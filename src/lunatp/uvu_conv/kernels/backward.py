"""UVU convolution backward CUDA and native launch."""

from __future__ import annotations
from collections import defaultdict
import re
import torch
from ...utils.cg import weight_channel_offset
from ...utils.runtime import get_overlap_state
from ....utils.precision import float_literal
from ....utils.extension import load_tp_extension
from ...utils.layout import build_signed_ij_classes, l0_feature_layout
from .utils import group_source_input_paths
from ..graph import SourceSegmentTables


def emit_raw_l0_contractions(
    lines,
    spec,
    path,
    offsets,
    width,
    *,
    outputs,
    operands,
    scalar_type="float",
    channel="u",
    indent="                ",
):
    """Insert the unscaled scalar-path product rule into the existing traversal.

    The envelope contraction uses the node seed alone. Scalar seeds contribute
    the trilinear x*y*w derivatives independently, so their direct envelope
    derivative is zero, including when the envelope contains zeros.
    """
    if path not in offsets:
        return
    address = f"int64_t(e) * {width} + {offsets[path]} + {channel}"
    lines.extend(
        f"{indent}{{\n{indent}const {scalar_type} raw_l0_seed = grad_l0 ? __ldg(grad_l0 + {address}) : 0.0;".split(
            "\n"
        )
    )
    for i, j, _k, coefficient in spec["terms"]:
        x, y, w = (operands["x"][i], operands["y"][j], operands["w"])
        expressions = {
            "x": f"{y} * {w} * raw_l0_seed",
            "y": f"{x} * {w} * raw_l0_seed",
            "w": f"{x} * {y} * raw_l0_seed",
        }
        literal = (
            f"{float(coefficient):.17e}"
            if scalar_type == "double"
            else float_literal(float(coefficient))
        )
        for role, target in outputs.items():
            variable = (
                target[i if role in {"x", "rx"} else j]
                if not isinstance(target, str)
                else target
            )
            lines.append(f"{indent}{variable} += {literal} * ({expressions[role]});")
    lines.append(f"{indent}}}")


def make_uvu_conv_bwd_source(
    op, *, with_env=False, need_grad_env=False, with_l0=False
):
    """Generate a precision-specialized first-order derivative and its scratch layout."""
    kernel_params = ", const double* env" if with_env else ""
    kernel_args = ", env->data_ptr<double>()" if with_env else ""
    if need_grad_env:
        kernel_params += ", double* ge_partial"
        kernel_args += ", ge_partial->data_ptr<double>()"
    if with_l0:
        kernel_params += ", const double* grad_l0"
        kernel_args += ", grad_l0->data_ptr<double>()"
    l0_offsets, l0_dim = l0_feature_layout(op.generated_specs) if with_l0 else ({}, 0)
    groups = group_source_input_paths(op.generated_specs)
    widths = {int(s["channels"]) for s in op.generated_specs}
    small_program = sum(d for (_, d), _ in groups) <= 16
    program_live_words = []
    for (_, dim), paths in groups:
        go_by_y = defaultdict(int)
        for _, spec in paths:
            go_by_y[int(spec["y_offset"])] += int(spec["d3"])
        program_live_words.append(
            4 * dim
            + 6 * max(int(spec["d2"]) for _, spec in paths)
            + 2 * max(go_by_y.values())
            + 32
        )
    wide_program = max(program_live_words) <= 120
    fuse = op.compute_dtype == torch.float32 and len(widths) == 1 and small_program
    bundles = [groups] if fuse else [[g] for g in groups]
    segmented = not fuse
    bodies, launches, kernels = [], [], []
    partial_count = 0
    covered_x = set()
    zero_y = False
    vectorized = False
    for group_id, bundle in enumerate(bundles):
        paths = [(pid, s) for _, gp in bundle for pid, s in gp]
        d1 = sum(d for (_, d), _ in bundle)
        prefixes = {
            xo: f"g{g}_" if fuse else "" for g, ((xo, _), _) in enumerate(bundle)
        }
        channels = int(paths[0][1]["channels"])
        by_y = defaultdict(list)
        covered_y = set()
        for path_id, spec in paths:
            yo, d2 = int(spec["y_offset"]), int(spec["d2"])
            by_y[(yo, d2)].append((path_id, spec))
            covered_y.update(range(yo, yo + d2))
        go_values = max(
            sum(int(spec["d3"]) for _, spec in ypaths) for ypaths in by_y.values()
        )
        max_d2 = max(d2 for _, d2 in by_y)
        live_words = 4 * d1 + 6 * max_d2 + 2 * go_values + 32
        vector_live = 8 * d1 + 8 * max_d2 + 4 * go_values + 32
        aligned = (
            op.out_dim % 2 == 0
            and op.weight_numel % 2 == 0
            and all(
                int(spec["out_offset"]) % 2 == 0 and int(spec["weight_offset"]) % 2 == 0
                for _, spec in paths
            )
        )
        resident_warps = 16 if live_words <= 120 else (12 if live_words <= 156 else 8)
        vector_limit = 192 if resident_warps == 12 else 160
        vector = (
            2
            if channels % 64 == 0
            and vector_live <= vector_limit
            and aligned
            and all(int(spec.get("weight_stride", 1)) == 1 for _, spec in paths)
            else 1
        )
        # Small FP32 programs expose channels as lanes to keep more warps resident.
        wide_block = (
            op.compute_dtype == torch.float32
            and small_program
            and (fuse or wide_program)
        )
        if wide_block:
            vector = 1
            if not fuse:
                resident_warps = 16
        padded_channels = ((channels // vector + 31) // 32) * 32
        limit = min(128 if wide_block else 32, padded_channels)
        threads = max(t for t in range(32, limit + 1, 32) if padded_channels % t == 0)
        vectorized |= vector == 2
        tiles = (channels + threads * vector - 1) // (threads * vector)
        min_blocks = max(1, resident_warps // (vector * (threads // 32)))
        kernel = f"lunatp_uvuconv_bwd_source_fp64<{group_id}, {tiles}, {vector}, {min_blocks}, {threads}, INDEX>"
        kernels.append(kernel)
        partial_base = partial_count
        partial_count += (channels + 32 * vector - 1) // (32 * vector)
        for (xo, dim), _ in bundle:
            covered_x.update(range(xo, xo + channels * dim))
        zero_y |= len(covered_y) != int(op.in2_dim)
        tail = channels % 32 != 0
        lines = [f"case {group_id}: {{"]
        if tail:
            lines.append(f"const bool active_channel = u < {channels};")
        else:
            lines.append(f"if (u >= {channels}) break;")
        for (x_offset, d1), _ in bundle:
            prefix = prefixes[x_offset]
            lines.append(
                f"const INDEX {prefix}xb = INDEX(node) * X_DIM + {x_offset} + u * {d1};"
            )
            for v in range(vector):
                for i in range(d1):
                    lines += [
                        f"const double {prefix}x{v}_{i} = "
                        + ("active_channel ? " if tail else "")
                        + f"__ldg(x + {prefix}xb + {v * d1 + i})"
                        + (" : 0.0" if tail else "")
                        + ";",
                        f"double {prefix}gx{v}_{i} = 0.0;",
                    ]
        lines += r"""for (Index p = begin; p < end; ++p) {{
{edge_index}
const INDEX yb = INDEX(e) * Y_DIM;
const INDEX gb = INDEX(d) * GO_DIM;
const INDEX wb = INDEX(e) * W_DIM;""".format(
            edge_index=(
                "const Index e = edge_ids[p];\nconst int d = __ldg(dst + e);"
            ),
        ).split("\n")
        if with_env:
            lines.append("const double env_scale = __ldg(env + e);")
        if need_grad_env:
            lines.append("double ge_value = 0.0;")
        for (yo, d2), ypaths in sorted(
            by_y.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            lines.append("{")
            for j in range(d2):
                lines += f"const double y{j} = __ldg(y + yb + {yo + j});\ndouble gy{j} = 0.0;".split(
                    "\n"
                )
            if tail:
                lines.append("if (active_channel) {")
            for path_id, spec in ypaths:
                d1 = int(spec["d1"])
                prefix = prefixes[int(spec["x_offset"])]
                wo, go, d3 = (
                    int(spec["weight_offset"]),
                    int(spec["out_offset"]),
                    int(spec["d3"]),
                )
                lines.append("{")
                if vector == 2:
                    lines.append(
                        f"const double2 weights = __ldg(reinterpret_cast<const double2*>(weight + wb + {wo} + {weight_channel_offset(spec)}));"
                    )
                    for k in range(d3):
                        lines.append(
                            f"const double2 g{k} = __ldg(reinterpret_cast<const double2*>(grad_out + gb + {go} + u * {d3}) + {k});"
                        )
                for v in range(vector):
                    lines.append(f"double gw{v} = 0.0;")
                for v in range(vector):
                    lines.append("{")
                    weight_expr = (
                        f"""weights.{("x" if v == 0 else "y")}"""
                        if vector == 2
                        else f"__ldg(weight + wb + {wo} + {weight_channel_offset(spec)})"
                    )
                    lines.append(
                        f"const double w = {weight_expr}"
                        + (" * env_scale;" if with_env else ";")
                    )
                    lines += [f"double a{i} = 0.0;" for i in range(d1)]
                    lines += [f"double b{j} = 0.0;" for j in range(d2)]
                    for ci, cl in enumerate(
                        build_signed_ij_classes(
                            list(spec["terms"]), spec_y_offset=yo, y_name_base=yo
                        )
                    ):
                        products = []
                        for _, _, k, co in cl.members[0].terms:
                            k = int(k)
                            if vector == 2:
                                flat = v * d3 + k
                                value = f"""g{flat // 2}.{("x" if flat % 2 == 0 else "y")}"""
                            else:
                                value = f"__ldg(grad_out + gb + {go} + u * {d3} + {k})"
                            if path_id in l0_offsets and not with_env:
                                value = f"({value} + __ldg(grad_l0 + int64_t(e) * {l0_dim} + {l0_offsets[path_id]} + u + {v}))"
                            products.append(f"{float(co):.17e} * {value}")
                        lines.append(
                            f"""const double q{ci} = {" + ".join(products)};"""
                        )
                        for m in cl.members:
                            sign = "+=" if m.relative_sign > 0 else "-="
                            lines += f"a{m.i_idx} {sign} y{m.j_name} * q{ci};\nb{m.j_name} {sign} {prefix}x{v}_{m.i_idx} * q{ci};".split(
                                "\n"
                            )
                    for i in range(d1):
                        lines += f"{prefix}gx{v}_{i} += w * a{i};\ngw{v} += {prefix}x{v}_{i} * a{i};".split(
                            "\n"
                        )
                    lines += [f"gy{j} += w * b{j};" for j in range(d2)]
                    lines.append("}")
                for v in range(vector):
                    if need_grad_env:
                        raw_weight = (
                            f"""weights.{("x" if v == 0 else "y")}"""
                            if vector == 2
                            else f"__ldg(weight + wb + {wo} + {weight_channel_offset(spec)})"
                        )
                        lines.append(f"ge_value += gw{v} * {raw_weight};")
                    if with_env:
                        lines.append(f"gw{v} *= env_scale;")
                    if with_env and path_id in l0_offsets:
                        channel = "u" if v == 0 else f"u + {v}"
                        emit_raw_l0_contractions(
                            lines,
                            spec,
                            path_id,
                            l0_offsets,
                            l0_dim,
                            scalar_type="double",
                            channel=channel,
                            outputs={
                                "x": [
                                    f"{prefix}gx{v}_{i}" for i in range(int(spec["d1"]))
                                ],
                                "y": [f"gy{j}" for j in range(int(spec["d2"]))],
                                "w": f"gw{v}",
                            },
                            operands={
                                "x": [
                                    f"{prefix}x{v}_{i}" for i in range(int(spec["d1"]))
                                ],
                                "y": [f"y{j}" for j in range(int(spec["d2"]))],
                                "w": (
                                    f"""weights.{("x" if v == 0 else "y")}"""
                                    if vector == 2
                                    else f"__ldg(weight + wb + {wo} + {weight_channel_offset(spec)})"
                                ),
                            },
                            indent="",
                        )
                if vector == 2:
                    lines.append(
                        f'asm volatile("st.global.cs.v2.f64 [%0], {{%1, %2}};" :: "l"(grad_weight + wb + {wo} + {weight_channel_offset(spec)}), "d"(gw0), "d"(gw1) : "memory");'
                    )
                else:
                    lines.append(
                        f"store_cs(grad_weight + wb + {wo} + {weight_channel_offset(spec)}, gw0);"
                    )
                lines.append("}")
            if tail:
                lines.append("}")
            for j in range(d2):
                lines.append(f"const double sy{j} = warp_sum(gy{j});")
                if with_env:
                    lines.append(
                        f"store_lane_zero(gy_partial + (int64_t({partial_base} + tile * (blockDim.x / 32) + warp) * edges + e) * Y_DIM + {yo + j}, sy{j}, lane);"
                    )
                else:
                    lines += [
                        "if (lane == 0) {",
                        f"const int part = {partial_base} + tile * (blockDim.x / 32) + warp;",
                        f"gy_partial[(int64_t(part) * edges + e) * Y_DIM + {yo + j}] = sy{j};",
                        "}",
                    ]
            lines.append("}")
        if need_grad_env:
            lines += [
                "const double ge_sum = warp_sum(ge_value);",
                f"store_lane_zero(ge_partial + int64_t({partial_base} + tile * (blockDim.x / 32) + warp) * edges + e, ge_sum, lane);",
            ]
        lines.append("}")
        for (x_offset, d1), _ in bundle:
            prefix = prefixes[x_offset]
            for v in range(vector):
                for i in range(d1):
                    lines.append(
                        ("if (active_channel) " if tail else "")
                        + f"gx_partial[int64_t(segment) * X_DIM + {x_offset} + u * {d1} + {v * d1 + i}] = {prefix}gx{v}_{i};"
                    )
        lines += ["break;", "}"]
        bodies.append("\n".join(lines))
        launches.append(
            r"""{kernel}<<<segments * {tiles}, {threads}, 0, stream>>>(
            x.data_ptr<double>(), y.data_ptr<double>(), weight.data_ptr<double>(), go.data_ptr<double>(),
            dst.data_ptr<int32_t>(), edge_ids.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
            segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(), segment_order.data_ptr<{source_index_type}>(),
            gx_partial.data_ptr<double>(), gy_partial.data_ptr<double>(), gw.data_ptr<double>(), edges{kernel_args});""".format(
                kernel=kernel,
                tiles=tiles,
                threads=threads,
                source_index_type="EdgeIndex" if segmented else "int32_t",
                kernel_args=kernel_args,
            )
        )
    index_types = ("Index",) if op.compute_dtype == torch.float32 else ("int64_t",)
    cache_setup = "\n".join(
        f"""        C10_CUDA_CHECK(cudaFuncSetCacheConfig({kernel.replace(", INDEX>", ", " + index_type + ">")}, cudaFuncCachePreferL1));"""
        for index_type in index_types
        for kernel in kernels
    )
    stream_state, state_parameter, stream_binding = "", "", ""
    if len(launches) > 1:
        # Source groups own disjoint input slices and partial-buffer slots.
        # Their reductions run only after all groups have completed.
        stream_state = r"""
struct SourceStreams {
    SourceStreams(int device, const std::vector<int64_t>& handles)
        : worker(at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handles.at(0)), device)) {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    }
    ~SourceStreams() noexcept {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        const int device = worker.device_index();
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        cudaEventDestroy(start);
        cudaEventDestroy(done);
        if (previous != device) cudaSetDevice(previous);
    }
    SourceStreams(const SourceStreams&) = delete;
    SourceStreams& operator=(const SourceStreams&) = delete;
    at::cuda::CUDAStream worker;
    cudaEvent_t start = nullptr, done = nullptr;
};
"""
        state_parameter = "SourceStreams& state, "
        stream_binding = r"""
pybind11::class_<SourceStreams>(m, "_Streams", pybind11::module_local())
    .def(pybind11::init<int, const std::vector<int64_t>&>());
m.attr("_Streams").attr("stream_count") = 1;
m.attr("_Streams").attr("stream_priority") = 0;
"""
        launches = ["C10_CUDA_CHECK(cudaEventRecord(state.start, stream));", "C10_CUDA_CHECK(cudaStreamWaitEvent(state.worker, state.start, 0));"] + [
            launch.replace(", 0, stream>>>", ", 0, state.worker>>>")
            for launch in launches[1::2]
        ] + ["C10_CUDA_CHECK(cudaEventRecord(state.done, state.worker));"] + launches[::2] + ["C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done, 0));"]
    launch_body = "\n".join(launches)
    launch_body = launch_body.replace(", INDEX>", ", " + index_types[0] + ">")
    env_reduce = r"""
__global__ void lunatp_uvuconv_bwd_env_reduce_fp64(const double* partial, double* ge, int64_t edges) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    double value = 0.0;
    for (int p = 0; p < PARTS; ++p) value += partial[int64_t(p) * edges + e];
    ge[e] = value;
}
""" if need_grad_env else ""
    env_launch = (
        """    if (edges > 0) lunatp_uvuconv_bwd_env_reduce_fp64<<<(edges + 255) / 256, 256, 0, stream>>>(
        ge_partial->data_ptr<double>(), ge->data_ptr<double>(), edges);
"""
        if need_grad_env
        else ""
    )
    # Keep warp-leader writes predicated when optional operands change NVCC's
    # branch heuristics; divergent stores otherwise duplicate address arithmetic.
    env_store = r"""
__device__ __forceinline__ void store_lane_zero(double* ptr, double value, int lane) {
    asm volatile("{ .reg .pred p; setp.eq.u32 p, %2, 0; @p st.global.f64 [%0], %1; }"
        :: "l"(ptr), "d"(value), "r"(lane) : "memory");
}
""" if with_env else ""
    source = r"""
#include <torch/extension.h>
#include <optional>
#include <ATen/cuda/CUDAContext.h>
{stream_includes}#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
namespace {{
constexpr int X_DIM = {in1_dim};
constexpr int Y_DIM = {in2_dim};
constexpr int GO_DIM = {out_dim};
constexpr int W_DIM = {weight_numel};
constexpr int PARTS = {partial_count};
__device__ __forceinline__ double warp_sum(double x) {{
    x += __shfl_down_sync(0xffffffff, x, 16);
    x += __shfl_down_sync(0xffffffff, x, 8);
    x += __shfl_down_sync(0xffffffff, x, 4);
    x += __shfl_down_sync(0xffffffff, x, 2);
    x += __shfl_down_sync(0xffffffff, x, 1);
    return x;
}}
__device__ __forceinline__ void store_cs(double* ptr, double value) {{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(ptr), "d"(value) : "memory");
}}
{env_store}template<int GROUP, int TILES, int VECTOR, int MIN_BLOCKS, int THREADS, typename INDEX>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS) void lunatp_uvuconv_bwd_source_fp64(
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ weight, const double* __restrict__ grad_out,
    const int32_t* __restrict__ dst, const EdgeIndex* __restrict__ edge_ids,
    const int32_t* __restrict__ segment_nodes, const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end, const {source_index_type}* __restrict__ segment_order,
    double* __restrict__ gx_partial, double* __restrict__ gy_partial,
    double* __restrict__ grad_weight, int64_t edges{kernel_params}) {{
    const Index segment = segment_order[blockIdx.x / TILES];
    const int tile = blockIdx.x % TILES;
    const int node = {source_node};
    const Index begin = segment_begin[segment], end = segment_end[{source_row}];
    const int u = (tile * blockDim.x + threadIdx.x) * VECTOR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    switch (GROUP) {{
{bodies}
    }}
}}
__global__ void lunatp_uvuconv_bwd_reduce_fp64(const double* __restrict__ gx_partial,
    const double* __restrict__ gy_partial, const EdgeIndex* __restrict__ rowptr,
    double* __restrict__ gx, double* __restrict__ gy, int64_t nodes, int64_t edges) {{
    const int64_t node_blocks = {clear_input_grid};
    if (blockIdx.x < node_blocks) {{
        const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= nodes * X_DIM) return;
        const int n = index / X_DIM, c = index % X_DIM;
        double value = 0.0;
        for (Index s = rowptr[n]; s < rowptr[n + 1]; ++s)
            value += gx_partial[int64_t(s) * X_DIM + c];
        gx[index] = value;
    }} else {{
        const int64_t index = (int64_t(blockIdx.x) - node_blocks) * blockDim.x + threadIdx.x;
        if (index >= edges * Y_DIM) return;
        double value = 0.0;
        for (int p = 0; p < PARTS; ++p)
            value += gy_partial[int64_t(p) * edges * Y_DIM + index];
        gy[index] = value;
    }}
}}
{env_reduce}{stream_state}}}
void run_uvu_conv_bwd({state_parameter}torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor go,
    torch::Tensor dst, torch::Tensor edge_ids, torch::Tensor segment_nodes,
    torch::Tensor segment_begin, torch::Tensor segment_end, torch::Tensor segment_order,
    torch::Tensor segment_rowptr, torch::Tensor gx_partial, torch::Tensor gy_partial,
    torch::Tensor gx, torch::Tensor gy, torch::Tensor gw, int64_t nodes, int64_t edges,
    int64_t segments, std::optional<torch::Tensor> env,
    std::optional<torch::Tensor> ge_partial, std::optional<torch::Tensor> ge,
    std::optional<torch::Tensor> grad_l0) {{
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    static thread_local int configured_device = -1;
    if (configured_device != x.get_device()) {{
{cache_setup}
        configured_device = x.get_device();
    }}
    if (segments > 0) {{
{launch_body}
    }}
    const int64_t node_blocks = {clear_input_grid};
    const int64_t edge_blocks = (edges * Y_DIM + 255) / 256;
    if (node_blocks + edge_blocks > 0) lunatp_uvuconv_bwd_reduce_fp64<<<node_blocks + edge_blocks, 256, 0, stream>>>(
        gx_partial.data_ptr<double>(), gy_partial.data_ptr<double>(), segment_rowptr.data_ptr<EdgeIndex>(),
        gx.data_ptr<double>(), gy.data_ptr<double>(), nodes, edges);
{env_launch}    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ {stream_binding}m.def("run_uvu_conv_bwd", &run_uvu_conv_bwd); }}
""".format(
        stream_includes="#include <vector>\n" if stream_state else "",
        stream_state=stream_state,
        state_parameter=state_parameter,
        stream_binding=stream_binding,
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        out_dim=op.out_dim,
        weight_numel=op.weight_numel,
        partial_count=partial_count,
        env_store=env_store,
        source_index_type="EdgeIndex" if segmented else "int32_t",
        kernel_params=kernel_params,
        source_node="segment_nodes[segment]" if segmented else "segment",
        source_row="segment" if segmented else "segment + 1",
        bodies=chr(10).join(bodies),
        clear_input_grid="(nodes * X_DIM + 255) / 256" if segmented else "0",
        env_reduce=env_reduce,
        cache_setup=cache_setup,
        launch_body=launch_body,
        env_launch=env_launch,
    )
    if op.compute_dtype == torch.float32:
        source = source.replace("_fp64", "_fp32")
        source = re.sub(r"\bdouble([234]?)\b", r"float\1", source)
        source = source.replace(".f64", ".f32").replace('"d"(', '"f"(')
        source = re.sub(
            r"(?<![\w.])(\d+\.\d+(?:e[+-]\d+)?)(?![\w.])",
            r"\1f",
            source,
        )
    return (
        source,
        segmented,
        partial_count,
        vectorized,
        len(covered_x) != int(op.in1_dim),
        zero_y,
    )


def run(op, x, y, w, g, graph, env, gl0, *, needs=(True, True, True, False)):
    node_features = x
    edge_attr = y
    weight = w
    src = graph.src
    dst = graph.dst
    grad_out = g
    src_rowptr = graph.src_rowptr
    edge_ids_by_src = graph.edge_ids_by_src
    source_node_order = graph.source_node_order
    need_grad_x, need_grad_edge_attr, need_grad_weight, need_grad_env = needs
    grad_l0 = gl0
    x, y, go = node_features, edge_attr, grad_out
    with_env = env is not None
    with_l0 = grad_l0 is not None
    flags = int(with_env) + 2 * int(need_grad_env) + 4 * int(with_l0)
    cache_name = "_source_first_ext" + (f"_{flags}" if flags else "")
    ext = getattr(op, cache_name, None)
    if ext is None:
        (
            source,
            op._source_first_segmented,
            op._source_first_partials,
            op._source_first_vectorized,
            op._source_first_zero_x,
            op._source_first_zero_y,
        ) = make_uvu_conv_bwd_source(
            op,
            with_env=with_env,
            need_grad_env=need_grad_env,
            with_l0=with_l0,
        )
        name = (
            "lunatp_uvuconv_bwd_source_fp64"
            if op.compute_dtype == torch.float64
            else "lunatp_uvuconv_bwd_source_fp32"
        )
        ext = load_tp_extension(
            name, source, max_channels=op.max_channels, index_bits=op.index_bits
        )
        setattr(op, cache_name, ext)
    if op._source_first_segmented:
        segments = graph.source_segments
    else:
        segments = SourceSegmentTables(
            nodes=source_node_order,
            begin=src_rowptr,
            end=src_rowptr,
            order=source_node_order,
            rowptr=src_rowptr,
        )
    # Contiguous offset views can still violate double2 base alignment.
    if op._source_first_vectorized:
        if weight.data_ptr() % 16:
            weight = weight.clone()
        if go.data_ptr() % 16:
            go = go.clone()

    gx, gy, gw = torch.empty_like(x), torch.empty_like(y), torch.empty_like(weight)
    if op._source_first_segmented:
        gx_partial = (torch.zeros if op._source_first_zero_x else torch.empty)(
            (segments.nodes.numel(), op.in1_dim), device=x.device, dtype=x.dtype
        )
    else:
        gx_partial = gx
        if op._source_first_zero_x:
            gx.zero_()
    gy_shape = (op._source_first_partials, src.numel(), op.in2_dim)
    gy_partial = (torch.zeros if op._source_first_zero_y else torch.empty)(
        gy_shape, device=x.device, dtype=x.dtype
    )
    grad_env = None
    ge_partial = None
    if need_grad_env:
        grad_env = y.new_empty((src.numel(), 1))
        ge_partial = y.new_empty((op._source_first_partials, src.numel()))

    state_args = (get_overlap_state(op, x.device, ext._Streams),) if hasattr(ext, "_Streams") else ()
    ext.run_uvu_conv_bwd(
        *state_args,
        x,
        y,
        weight,
        go,
        dst,
        edge_ids_by_src,
        segments.nodes,
        segments.begin,
        segments.end,
        segments.order,
        segments.rowptr,
        gx_partial,
        gy_partial,
        gx,
        gy,
        gw,
        x.shape[0],
        src.numel(),
        segments.nodes.numel(),
        env,
        ge_partial,
        grad_env,
        grad_l0,
    )
    return (
        gx if need_grad_x else None,
        gy if need_grad_edge_attr else None,
        gw if need_grad_weight else None,
        grad_env,
    )
