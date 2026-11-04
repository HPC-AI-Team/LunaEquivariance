"""UVW convolution backward CUDA and native launch."""

from __future__ import annotations
import textwrap
from collections.abc import Mapping
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal
from ....utils.extension import load_tp_extension, load_cuda_extension
from ...utils.runtime import auxiliary_streams, get_overlap_state
from ...utils.contractions import (
    ENV_INPUTS,
    ENV_PROGRAMS,
    INPUTS,
    PROGRAMS,
    ENV_OUTPUT_SHAPES,
    OUTPUT_SHAPES,
)
from .forward import make_uvw_conv_forward_external_kernel
from .utils import (
    get_uvw_conv_node_layout, get_uvw_conv_weight_layout, set_arch,
    get_uvw_conv_backward_layout, get_uvw_conv_tr_layout, make_tr_plan,
    MAX_WEIGHT_BLOCKS, MAX_WEIGHT_SPLITS, MIN_WEIGHT_SPLITS, WEIGHT_SPLIT_NODES,
    BACKWARD_NEIGHBOR_SPLITS,
)


def make_uvw_conv_node_cutlass_source(meta, kernel_name):
    """Contract node gradients and weights without packing either operand."""
    offsets, total = get_uvw_conv_node_layout(meta["generated_specs"])
    cases = []
    for index, spec in enumerate(meta["generated_specs"]):
        d, channels, k = spec["d3"], spec["path_channels"], spec["out_channels"]
        cases.append(f"""
case {index}: {{
    using Kernel=typename {kernel_name}_Gemm<{d}>::GemmKernel;
    using Mma=typename Kernel::Mma;
    using Epilogue=typename Kernel::Epilogue;
    auto& storage=shared.s{index};
    const int64_t first_node=int64_t(blockIdx.y)*64/{d};
    const int row=int64_t(blockIdx.y)*64%{d},column=blockIdx.x*64;
    const int rows=min(int64_t(64+{d}), (nodes-first_node)*{d});
    if(first_node>=nodes || column>={channels}) return;
    typename Mma::FragmentC acc; acc.clear();
#pragma unroll
    for(int p=0;p<Products;++p) {{
        const float* grad=p==0?g0:g1;
        const float* weight=p==0?w0:w1;
        typename Mma::IteratorA a(
        typename Mma::IteratorA::Params(cutlass::layout::RowMajor({meta['out_dim']})),
        const_cast<float*>(grad+first_node*{meta['out_dim']}+{spec['out_offset']}),{{rows,{k}}},threadIdx.x,{{row,0}});
        typename Mma::IteratorB b(
        typename Mma::IteratorB::Params(cutlass::layout::ColumnMajor({k})),
        const_cast<float*>(weight+{spec['weight_offset']}),{{{k},{channels}}},threadIdx.x,{{0,column}});
        Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
        mma(({k}+7)/8,acc,a,b,acc);
        __syncthreads();
    }}
    typename Epilogue::OutputTileIterator output(
    typename Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor({total})),
    z+first_node*{total}+{offsets[index]},{{rows,{channels}}},threadIdx.x,{{row,column}});
    Epilogue epilogue(storage.epilogue,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    epilogue(typename Kernel::EpilogueOutputOp(1.0f,0.0f),output,acc,output);
    return;
}}
""")
    return f"""
#include <cutlass/gemm/device/gemm_universal.h>
// Read and write [node, channel, component] directly, without packing.
template<int Components, int Stride> struct {kernel_name}_Layout {{
    using Index=int32_t; using LongIndex=int64_t;
    CUTLASS_HOST_DEVICE {kernel_name}_Layout(cutlass::MatrixCoord,Index) {{}}
    CUTLASS_HOST_DEVICE {kernel_name}_Layout(cutlass::layout::PitchLinearCoord,Index) {{}}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::MatrixCoord q) const {{
        return LongIndex(q.row()/Components)*Stride+LongIndex(q.column())*Components+q.row()%Components;
    }}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord q) const {{
        return operator()(cutlass::MatrixCoord(q.strided(),q.contiguous()));
    }}
}};
template<int Components> using {kernel_name}_Gemm=cutlass::gemm::device::GemmUniversal<
    float,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,
    float,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64,64,8>,cutlass::gemm::GemmShape<32,32,8>,cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,1,1,cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,cutlass::ComplexTransform::kNone,false,false,false,
    {kernel_name}_Layout<Components,{total}>,{kernel_name}_Layout<Components,{meta['out_dim']}>>;
template<int Products>
__global__ void {kernel_name}(const float* w0,const float* g0,const float* w1,const float* g1,float* z,int64_t nodes) {{
    union Storage {{ {' '.join(f"typename {kernel_name}_Gemm<{s['d3']}>::GemmKernel::SharedStorage s{i};" for i,s in enumerate(meta["generated_specs"]))} }};
    __shared__ Storage shared;
    switch(blockIdx.z) {{
{textwrap.indent(chr(10).join(cases), '        ')}
    }}
}}
"""


def grad_x_allocator(meta):
    """Check contiguous input-irrep coverage without enumerating channels."""
    intervals = sorted(
        {
            (
                int(spec["x_offset"]),
                int(spec["x_offset"]) + int(spec["in_channels"]) * int(spec["d1"]),
            )
            for spec in meta["generated_specs"]
        }
    )
    end = 0
    for begin, stop in intervals:
        if begin > end:
            return "torch::zeros"
        end = max(end, stop)
    return "torch::empty" if end == int(meta["in1_dim"]) else "torch::zeros"


def emit_dx_lines(
    meta,
    z_offsets,
    kernel_name,
    threads,
    channel_bound,
    *,
    paired=False,
    wide_z=False,
    products=None,
):
    """Emit source-CSR contraction; large accumulator sets own separate regions.

    Keeping at most one input irrep live avoids the high register pressure of
    accumulating every input component across all neighbors. Small contractions
    retain one block per source/channel tile. Both variants write final outputs.
    """
    specs = list(meta["generated_specs"])
    in1_dim, in2_dim = int(meta["in1_dim"]), int(meta["in2_dim"])
    slots = []
    for spec in specs:
        for component in range(int(spec["d1"])):
            key = (
                int(spec["x_offset"]),
                int(spec["in_channels"]),
                int(spec["d1"]),
                component,
            )
            if key not in slots:
                slots.append(key)
    slot_index = {
        (offset, dim, component): index
        for index, (offset, _, dim, component) in enumerate(slots)
    }
    groups = {}
    for index, spec in enumerate(specs):
        # Limit the compile-time accumulator count.
        key = int(spec["x_offset"]) if len(slots) > 32 else 0
        groups.setdefault(key, []).append((index, spec))
    arrays = ("edge0", "z0", "edge1", "z1") if paired else ("edge_attr", "z")
    if products is not None:
        arrays = tuple(
            sorted(
                {
                    role
                    for terms in products.values()
                    for term in terms
                    for role in term
                    if role is not None
                }
            )
        )
    targets = tuple(products) if products is not None else ("grad_x",)
    parameters = [
        f"""    const float* __restrict__ {("ptr_" if products is not None else "")}{name},"""
        for name in arrays
    ]
    lines = [
        r"""__global__ __launch_bounds__({threads}, 4) void {kernel_name}(
{parameters}
    const int32_t *__restrict__ dst,
    const EdgeIndex *__restrict__ src_rowptr,
    const EdgeIndex *__restrict__ edge_by_src,
{output_parameters}
    int64_t num_nodes) {{
    const int s = blockIdx.x;
    const int u = blockIdx.y * {threads} + threadIdx.x;
    if (s >= num_nodes || u >= {channel_bound})
        return;""".format(
            threads=threads,
            kernel_name=kernel_name,
            parameters="\n".join(parameters),
            output_parameters="\n".join(
                [f"    float* __restrict__ {target}," for target in targets]
            ),
            channel_bound=channel_bound,
        )
    ]
    for group, group_specs in enumerate(groups.values()):
        offsets = {int(spec["x_offset"]) for _, spec in group_specs}
        owned = [
            (index, slot) for index, slot in enumerate(slots) if slot[0] in offsets
        ]
        if len(groups) > 1:
            lines.append(f"    if (blockIdx.z == {group}) {{")
        for slot, _ in owned:
            if products is None:
                lines.append(f"    float acc_x{slot} = 0.0f;")
            else:
                lines += [f"    float acc_{target}{slot} = 0.0f;" for target in targets]
        lines += [r"""    const Index e_begin = __ldg(src_rowptr + s);
    const Index e_end = __ldg(src_rowptr + s + 1);
    for (Index pos = e_begin; pos < e_end; ++pos) {
        const Index e = __ldg(edge_by_src + pos);
        const int t = __ldg(dst + e);"""]
        if products is not None:
            for scalar in sorted(
                {
                    term[2]
                    for terms in products.values()
                    for term in terms
                    if term[2] is not None
                }
            ):
                lines.append(
                    f"        const float scale_{scalar}=__ldg(ptr_{scalar}+e);"
                )
        for spec_idx, spec in group_specs:
            x_offset, y_offset = int(spec["x_offset"]), int(spec["y_offset"])
            in_channels, d1, d3 = (
                int(spec["in_channels"]),
                int(spec["d1"]),
                int(spec["d3"]),
            )
            terms = list(spec["terms"])
            lines.append(f"        if (u < {in_channels}) {{")
            edge_channels, d2 = int(spec["edge_channels"]), int(spec["d2"])
            if edge_channels != 1:
                lines.append(f"            for (int v = 0; v < {edge_channels}; ++v) {{")
            edge_offset = f"v * {d2} + " if edge_channels != 1 else ""
            channel = "int64_t(u)" if wide_z else "u"
            path = (
                f"({channel} * {edge_channels} + v)" if edge_channels != 1 else channel
            )
            for j in sorted({int(t[1]) for t in terms}):
                if products is not None:
                    for feature in sorted(
                        {term[0] for terms in products.values() for term in terms}
                    ):
                        lines.append(
                            f"            const float value_{feature}_{spec_idx}_{j}=__ldg(ptr_{feature}+e*{in2_dim}+{edge_offset}{y_offset + j});"
                        )
                elif paired:
                    for product in range(2):
                        lines.append(
                            f"            const float y{product}v{spec_idx}_{j} = __ldg(edge{product} + e * {in2_dim} + {edge_offset}{y_offset + j});"
                        )
                else:
                    lines.append(
                        f"            const float yv{spec_idx}_{j} = __ldg(edge_attr + e * {in2_dim} + {edge_offset}{y_offset + j});"
                    )
            for k in sorted({int(t[2]) for t in terms}):
                if products is not None:
                    for matrix in sorted(
                        {term[1] for terms in products.values() for term in terms}
                    ):
                        lines.append(
                            f"            const float value_{matrix}_{spec_idx}_{k}=__ldg(ptr_{matrix}+Index(t)*GEN_TOTAL_Z+{z_offsets[spec_idx]}+{path}*{d3}+{k});"
                        )
                elif paired:
                    for product in range(2):
                        lines.append(
                            f"            const float z{product}v{spec_idx}_{k} = __ldg(z{product} + Index(t) * GEN_TOTAL_Z + {z_offsets[spec_idx]} + {path} * {d3} + {k});"
                        )
                else:
                    lines.append(
                        f"            const float zv{spec_idx}_{k} = __ldg(z + Index(t) * GEN_TOTAL_Z + {z_offsets[spec_idx]} + {path} * {d3} + {k});"
                    )
            for i, j, k, coeff in terms:
                slot = slot_index[x_offset, d1, int(i)]
                if products is not None:
                    for target, contractions in products.items():
                        for feature, matrix, scalar in contractions:
                            factor = f"*scale_{scalar}" if scalar is not None else ""
                            lines.append(
                                f"""            acc_{target}{slot}=fmaf({float_literal(float(coeff), meta.get("dtype"))}*value_{feature}_{spec_idx}_{int(j)}{factor},value_{matrix}_{spec_idx}_{int(k)},acc_{target}{slot});"""
                            )
                elif paired:
                    lines.append(
                        f"""            acc_x{slot} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, y0v{spec_idx}_{int(j)} * z0v{spec_idx}_{int(k)} + y1v{spec_idx}_{int(j)} * z1v{spec_idx}_{int(k)}, acc_x{slot});"""
                    )
                else:
                    lines.append(
                        f"""            acc_x{slot} = fmaf({float_literal(float(coeff), meta.get("dtype"))} * yv{spec_idx}_{int(j)}, zv{spec_idx}_{int(k)}, acc_x{slot});"""
                    )
            if edge_channels != 1:
                lines.append("        }")
            lines.append("        }")
        lines.append("    }")
        for slot, (offset, channels, dim, component) in owned:
            if products is None:
                lines.append(
                    f"    if (u < {channels}) grad_x[Index(s) * {in1_dim} + {offset} + u * {dim} + {component}] = acc_x{slot};"
                )
            else:
                lines += [
                    f"    if (u < {channels}) {target}[Index(s)*{in1_dim}+{offset}+u*{dim}+{component}]=acc_{target}{slot};"
                    for target in targets
                ]
        if len(groups) > 1:
            lines += ["    return;", "}"]
    lines.append("}")
    return lines, (f", {len(groups)}" if len(groups) > 1 else "")


def make_uvw_conv_backward_dx_shared_source(meta: Mapping[str, object]) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    z_offsets, total_z = get_uvw_conv_node_layout(specs)
    wide_z = total_z > (1 << 31) - 1
    z_type = "int64_t" if wide_z else "int"
    max_channels = max(int(s["in_channels"]) for s in specs)
    max_paths = max(int(s["path_channels"]) for s in specs)
    z_channels = (
        "GEN_MAX_PATH_CHANNELS" if max_paths != max_channels else "GEN_MAX_CHANNELS"
    )
    path_type = "int64_t" if max_paths > (1 << 31) - 64 else "int"
    path_constant = (
        f"constexpr {path_type} GEN_MAX_PATH_CHANNELS = {max_paths};\n"
        if max_paths != max_channels
        else ""
    )
    max_d3 = max(int(s["d3"]) for s in specs)
    x_tile = 64 if max_channels <= 64 else 128
    channel_type = "int64_t" if max_channels > (1 << 31) - x_tile else "int"

    grad_x_alloc = grad_x_allocator(meta)

    x_lines, x_group_grid = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_x_src_apply_fp32",
        "GEN_X_TILE",
        "GEN_MAX_CHANNELS",
        wide_z=wide_z,
    )

    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_IN1_DIM = {in1_dim};
constexpr {z_type} GEN_TOTAL_Z = {total_z};
{path_constant}constexpr {channel_type} GEN_MAX_CHANNELS = {max_channels};
constexpr int GEN_MAX_D3 = {max_d3};
constexpr int GEN_X_TILE = {x_tile};
constexpr int GEN_Z_TILE_M = {node_tile_m};
constexpr int GEN_Z_TILE_N = 64;


{node_contraction_source}

{x_lines}
}}  // namespace

torch::Tensor lunatp_uvwconv_bwd_dx_shared_launch_fp32(
    torch::Tensor edge_attr,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge_attr.device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, edge_attr.options());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, edge_attr.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_x_zgemm_fp32<1><<<dim3(
                   static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
                   static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
                   GEN_NUM_SPECS), 128, 0, stream>>>(
                weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    if (num_nodes > 0) lunatp_uvwconv_bwd_x_src_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
                                                   static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_X_TILE - 1) / GEN_X_TILE){x_group_grid}), GEN_X_TILE, 0, stream>>>(
            edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
            src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_dx_shared_launch_fp32", &lunatp_uvwconv_bwd_dx_shared_launch_fp32, "Graph-conv shared grad_x source-CSR backward launch");
}}
""".format(
            specs_count=len(specs),
            in1_dim=in1_dim,
            z_type=z_type,
            total_z=total_z,
            path_constant=path_constant,
            channel_type=channel_type,
            max_channels=max_channels,
            max_d3=max_d3,
            x_tile=x_tile,
            node_tile_m=64,
            node_contraction_source=make_uvw_conv_node_cutlass_source(
                meta, "lunatp_uvwconv_bwd_x_zgemm_fp32"
            ),
            x_lines=chr(10).join(x_lines),
            grad_x_alloc=grad_x_alloc,
            z_channels=z_channels,
            x_group_grid=x_group_grid,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_dxdy_shared_source(
    meta: Mapping[str, object]
) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    in2_dim = int(meta["in2_dim"])
    z_offsets, total_z = get_uvw_conv_node_layout(specs)
    wide_z = total_z > (1 << 31) - 1
    z_type = "int64_t" if wide_z else "int"
    max_in_channels = max(int(s["in_channels"]) for s in specs)
    max_paths = max(int(s["path_channels"]) for s in specs)
    z_channels = (
        "GEN_MAX_PATH_CHANNELS"
        if max_paths != max_in_channels
        else "GEN_MAX_IN_CHANNELS"
    )
    path_type = "int64_t" if max_paths > (1 << 31) - 64 else "int"
    path_constant = (
        f"constexpr {path_type} GEN_MAX_PATH_CHANNELS = {max_paths};\n"
        if max_paths != max_in_channels
        else ""
    )
    max_d2 = max(int(s["d2"]) for s in specs)
    max_d3 = max(int(s["d3"]) for s in specs)
    x_threads = 64 if max_in_channels <= 64 else 128
    y_threads = 64 if max_in_channels <= 64 else 128
    channel_type = "int64_t" if max_in_channels > (1 << 31) - x_threads else "int"

    grad_x_alloc = grad_x_allocator(meta)

    x_lines, x_group_grid = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_xy_source_x_apply_fp32",
        "GEN_X_THREADS",
        "GEN_MAX_IN_CHANNELS",
        wide_z=wide_z,
    )
    x_pair_lines, _ = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_xy_source_x_pair_apply_fp32",
        "GEN_X_THREADS",
        "GEN_MAX_IN_CHANNELS",
        paired=True,
        wide_z=wide_z,
    )

    edge_kernels = []
    for paired, edge_kernel_name in (
        (False, "lunatp_uvwconv_bwd_xy_source_y_apply_fp32"),
        (True, "lunatp_uvwconv_bwd_xy_source_y_pair_apply_fp32"),
    ):
        edge_specs = list(meta["generated_specs"])
        edge_in1_dim, edge_in2_dim = int(meta["in1_dim"]), int(meta["in2_dim"])
        # Paired contractions keep two sets of feature and node values live.
        # Their smaller budget preserves occupancy without splitting compact singles.
        edge_component_budget = 32 if paired else 48
        edge_groups = {}
        for edge_index, edge_spec in enumerate(edge_specs):
            edge_key = (
                int(edge_spec["y_offset"])
                if edge_in2_dim > edge_component_budget
                else 0
            )
            edge_groups.setdefault(edge_key, []).append((edge_index, edge_spec))
        edge_covered = {
            int(edge_spec["y_offset"]) + edge_j
            for edge_spec in edge_specs
            for edge_j in range(int(edge_spec["edge_channels"]) * int(edge_spec["d2"]))
        }
        edge_arrays = ("x0", "z0", "x1", "z1") if paired else ("x", "z")
        edge_lines = [
            "__global__ __launch_bounds__(GEN_Y_THREADS, 4)",
            f"void {edge_kernel_name}(",
            *[
                f"    const float* __restrict__ {edge_name},"
                for edge_name in edge_arrays
            ],
            "const int32_t* __restrict__ src,",
            "const int32_t* __restrict__ dst,",
            "    float* __restrict__ grad_edge,",
            "int64_t num_edges) {",
            "    __shared__ float smem[GEN_Y_WARPS * GEN_IN2_DIM];",
            "    const Index e = blockIdx.x;",
            "    const int u = threadIdx.x;",
            "    if (e >= num_edges) return;",
            "    const int s = __ldg(src + e);",
            "    const int t = __ldg(dst + e);",
        ]
        for edge_group, edge_group_specs in enumerate(edge_groups.values()):
            edge_components = {
                int(edge_spec["y_offset"]) + edge_j
                for _, edge_spec in edge_group_specs
                for edge_j in range(
                    int(edge_spec["edge_channels"]) * int(edge_spec["d2"])
                )
            }
            if edge_group == 0:
                # Unused edge irreps still receive zeros, without a clearing kernel.
                edge_components.update(set(range(edge_in2_dim)) - edge_covered)
            edge_components = sorted(edge_components)
            if len(edge_groups) > 1:
                edge_lines.append(f"    if (blockIdx.y == {edge_group}) {{")
            edge_lines += [
                f"    float acc_y{edge_j} = 0.0f;" for edge_j in edge_components
            ]
            for edge_spec_idx, edge_spec, edge_v in (
                (edge_spec_idx, edge_spec, edge_v)
                for edge_spec_idx, edge_spec in edge_group_specs
                for edge_v in range(int(edge_spec["edge_channels"]))
            ):
                edge_x_offset = int(edge_spec["x_offset"])
                edge_in_channels, edge_d1, edge_d3 = (
                    int(edge_spec["in_channels"]),
                    int(edge_spec["d1"]),
                    int(edge_spec["d3"]),
                )
                edge_terms = list(edge_spec["terms"])
                edge_channels = int(edge_spec["edge_channels"])
                edge_channel = "int64_t(u)" if wide_z else "u"
                edge_path = (
                    f"({edge_channel} * {edge_channels} + {edge_v})"
                    if edge_channels != 1
                    else edge_channel
                )
                if edge_channels != 1:
                    edge_lines.append("    {")
                edge_channel_type = (
                    "int64_t" if edge_in_channels > (1 << 31) - y_threads else "int"
                )
                edge_lines.append(
                    f"    for ({edge_channel_type} u = threadIdx.x; u < {edge_in_channels}; u += GEN_Y_THREADS) {{"
                    if edge_in_channels > y_threads
                    else f"    if (u < {edge_in_channels}) {{"
                )
                for edge_i in sorted({int(edge_t[0]) for edge_t in edge_terms}):
                    if paired:
                        for edge_product in range(2):
                            edge_lines.append(
                                f"    const float x{edge_product}v{edge_spec_idx}_{edge_i} = __ldg(x{edge_product} + Index(s) * {edge_in1_dim} + {edge_x_offset} + u * {edge_d1} + {edge_i});"
                            )
                    else:
                        edge_lines.append(
                            f"    const float xv{edge_spec_idx}_{edge_i} = __ldg(x + Index(s) * {edge_in1_dim} + {edge_x_offset} + u * {edge_d1} + {edge_i});"
                        )
                for edge_k in sorted({int(edge_t[2]) for edge_t in edge_terms}):
                    if paired:
                        for edge_product in range(2):
                            edge_lines.append(
                                f"    const float z{edge_product}v{edge_spec_idx}_{edge_k} = __ldg(z{edge_product} + Index(t) * GEN_TOTAL_Z + {z_offsets[edge_spec_idx]} + {edge_path} * {edge_d3} + {edge_k});"
                            )
                    else:
                        edge_lines.append(
                            f"    const float zv{edge_spec_idx}_{edge_k} = __ldg(z + Index(t) * GEN_TOTAL_Z + {z_offsets[edge_spec_idx]} + {edge_path} * {edge_d3} + {edge_k});"
                        )
                for edge_i, edge_j, edge_k, edge_coeff in edge_terms:
                    edge_component = (
                        int(edge_spec["y_offset"])
                        + edge_v * int(edge_spec["d2"])
                        + int(edge_j)
                    )
                    if paired:
                        edge_lines.append(
                            f"""    acc_y{edge_component} = fmaf({float_literal(float(edge_coeff), meta.get("dtype"))}, x0v{edge_spec_idx}_{int(edge_i)} * z0v{edge_spec_idx}_{int(edge_k)} + x1v{edge_spec_idx}_{int(edge_i)} * z1v{edge_spec_idx}_{int(edge_k)}, acc_y{edge_component});"""
                        )
                    else:
                        edge_lines.append(
                            f"""    acc_y{edge_component} = fmaf({float_literal(float(edge_coeff), meta.get("dtype"))} * xv{edge_spec_idx}_{int(edge_i)}, zv{edge_spec_idx}_{int(edge_k)}, acc_y{edge_component});"""
                        )
                edge_lines.append("}")
                if edge_channels != 1:
                    edge_lines.append("}")
            for edge_j in edge_components:
                edge_lines += r"""    float sum{j} = acc_y{j};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    sum{j} += __shfl_down_sync(0xffffffff, sum{j}, offset);
    if ((threadIdx.x & 31) == 0)
    smem[(threadIdx.x >> 5) * GEN_IN2_DIM + {j}] = sum{j};""".format(j=edge_j).split("\n")
            edge_lines += "    __syncthreads();\n    if (threadIdx.x < 32) {".split(
                "\n"
            )
            for edge_j in edge_components:
                edge_lines += r"""        float block_sum{j} = (threadIdx.x < GEN_Y_WARPS) ? smem[threadIdx.x * GEN_IN2_DIM + {j}] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        block_sum{j} += __shfl_down_sync(0xffffffff, block_sum{j}, offset);
        if (threadIdx.x == 0)
        grad_edge[e * GEN_IN2_DIM + {j}] = block_sum{j};""".format(j=edge_j).split("\n")
            edge_lines.append("    }")
            if len(edge_groups) > 1:
                edge_lines += ["    return;", "}"]
        edge_lines.append("}")
        edge_grid = "static_cast<unsigned int>(num_edges)"
        if len(edge_groups) > 1:
            edge_grid = f"dim3({edge_grid}, {len(edge_groups)})"
        edge_kernels.append((edge_lines, edge_grid))
    (y_kernel_lines, y_grid), (y_pair_kernel_lines, y_pair_grid) = edge_kernels

    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_X_THREADS = {x_threads};
constexpr int GEN_Y_THREADS = {y_threads};
constexpr int GEN_Y_WARPS = (GEN_Y_THREADS + 31) / 32;
constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_MAX_D2 = {max_d2};
constexpr {z_type} GEN_TOTAL_Z = {total_z};
{path_constant}constexpr {channel_type} GEN_MAX_IN_CHANNELS = {max_in_channels};
constexpr int GEN_MAX_D3 = {max_d3};
constexpr int GEN_Z_TILE_M = {node_tile_m};
constexpr int GEN_Z_TILE_N = 64;


{node_contraction_source}

{x_lines}

{x_pair_lines}

{y_kernel_lines}

{y_pair_kernel_lines}
}}  // namespace

template<int Products>
void prepare_uvwconv_z_fp32(int device) {{
    static thread_local int prepared_device=-1;
    if(prepared_device != device) {{
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,lunatp_uvwconv_bwd_xy_source_zgemm_fp32<Products>));
        prepared_device=device;
    }}
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_z_fp32(
    torch::Tensor weight,
    torch::Tensor grad_out,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(weight.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<1>(weight.get_device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, grad_out.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<1><<<dim3(
                   static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
                   static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
                   GEN_NUM_SPECS), 128, 0, stream>>>(
                weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_z_pair_fp32(
    torch::Tensor weight0, torch::Tensor grad0, torch::Tensor weight1,
    torch::Tensor grad1, int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(weight0.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<2>(weight0.get_device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, grad0.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<2><<<dim3(
                   ({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N,
                   (num_nodes * GEN_MAX_D3 + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M,
                   GEN_NUM_SPECS), 128, 0, stream>>>(
                weight0.data_ptr<float>(), grad0.data_ptr<float>(),
                weight1.data_ptr<float>(), grad1.data_ptr<float>(), z.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_x_from_z_fp32(
    torch::Tensor edge_attr,
    torch::Tensor z,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge_attr.device());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, edge_attr.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
                                                        static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
            edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
            src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_y_from_z_fp32(
    torch::Tensor x,
    torch::Tensor z,
    torch::Tensor src,
    torch::Tensor dst) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_edges = src.size(0);
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_apply_fp32<<<{y_grid}, GEN_Y_THREADS, 0, stream>>>(
            x.data_ptr<float>(), z.data_ptr<float>(), src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(),
            grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_edge;
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32(
    torch::Tensor edge0,
    torch::Tensor z0,
    torch::Tensor edge1,
    torch::Tensor z1,
    torch::Tensor x0,
    torch::Tensor x1,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge0.device());
    const int64_t num_edges = src.size(0);
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, x0.options());
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x0.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_pair_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
                                                              static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
            edge0.data_ptr<float>(), z0.data_ptr<float>(), edge1.data_ptr<float>(), z1.data_ptr<float>(),
            dst.data_ptr<int32_t>(), src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(),
            grad_x.data_ptr<float>(), num_nodes);
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_pair_apply_fp32<<<{y_pair_grid}, GEN_Y_THREADS, 0, stream>>>(
            x0.data_ptr<float>(), z0.data_ptr<float>(), x1.data_ptr<float>(), z1.data_ptr<float>(),
            src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_x, grad_edge}};
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_dxdy_shared_launch_fp32(
    torch::Tensor x,
    torch::Tensor edge_attr,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<1>(x.get_device());
    const int64_t num_edges = src.size(0);
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, x.options());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, x.options());
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<1><<<dim3(
                   static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
                   static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
                   GEN_NUM_SPECS), 128, 0, stream>>>(
                weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
                                                        static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
            edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
            src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_apply_fp32<<<{y_grid}, GEN_Y_THREADS, 0, stream>>>(
            x.data_ptr<float>(), z.data_ptr<float>(), src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(),
            grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_x, grad_edge}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_xy_source_z_pair_fp32", &lunatp_uvwconv_bwd_xy_source_z_pair_fp32);
    m.def("lunatp_uvwconv_bwd_xy_source_z_fp32", &lunatp_uvwconv_bwd_xy_source_z_fp32, "Graph-conv shared first z precontract");
    m.def("lunatp_uvwconv_bwd_xy_source_x_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_x_from_z_fp32, "Graph-conv shared first grad_x from z");
    m.def("lunatp_uvwconv_bwd_xy_source_y_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_y_from_z_fp32, "Graph-conv shared first grad_edge from z");
    m.def("lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32, "Graph-conv shared paired grad_x+grad_edge from z");
    m.def("lunatp_uvwconv_bwd_dxdy_shared_launch_fp32", &lunatp_uvwconv_bwd_dxdy_shared_launch_fp32, "Graph-conv shared grad_x+grad_edge source-CSR backward launch");
}}
""".format(
            specs_count=len(specs),
            x_threads=x_threads,
            y_threads=y_threads,
            in1_dim=in1_dim,
            in2_dim=in2_dim,
            max_d2=max_d2,
            z_type=z_type,
            total_z=total_z,
            path_constant=path_constant,
            channel_type=channel_type,
            max_in_channels=max_in_channels,
            max_d3=max_d3,
            node_tile_m=64,
            node_contraction_source=make_uvw_conv_node_cutlass_source(
                meta, "lunatp_uvwconv_bwd_xy_source_zgemm_fp32"
            ),
            x_lines=chr(10).join(x_lines),
            x_pair_lines=chr(10).join(x_pair_lines),
            y_kernel_lines=chr(10).join(y_kernel_lines),
            y_pair_kernel_lines=chr(10).join(y_pair_kernel_lines),
            z_channels=z_channels,
            grad_x_alloc=grad_x_alloc,
            x_group_grid=x_group_grid,
            y_grid=y_grid,
            y_pair_grid=y_pair_grid,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_dw_preprocess_shared_source(
    meta: Mapping[str, object],
) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    in2_dim = int(meta["in2_dim"])
    groups, group_rows, spec_to_group = get_uvw_conv_weight_layout(specs)
    threads = 128 if max(int(s["path_channels"]) for s in specs) <= 128 else 256

    params = []
    allocs = []
    ptrs = []
    for group_id, ((_out_offset, _out_channels, d3), _group_specs) in enumerate(groups):
        params.append(f"    float* __restrict__ q_group{group_id},")
        allocs.append(
            f"q_groups.push_back(torch::empty({{{group_rows[group_id]}, num_nodes * int64_t({d3})}}, q_options));"
        )
        ptrs.append(f"        q_groups[{group_id}].data_ptr<float>(),")
    cases = []
    pair_cases = []
    for spec_idx, spec in enumerate(specs):
        group_id, row_offset = spec_to_group[spec_idx]
        x_offset = int(spec["x_offset"])
        y_offset = int(spec["y_offset"])
        in_channels = int(spec["path_channels"])
        channel_type = (
            "int64_t" if row_offset + in_channels > (1 << 31) - threads else "int"
        )
        edge_channels, d2 = int(spec["edge_channels"]), int(spec["d2"])
        input_channel = f"(u / {edge_channels})" if edge_channels != 1 else "u"
        edge_offset = f"(u % {edge_channels}) * {d2} + " if edge_channels != 1 else ""
        d1 = int(spec["d1"])
        d3 = int(spec["d3"])
        terms = list(spec["terms"])
        unique_i = sorted({int(t[0]) for t in terms})
        unique_j = sorted({int(t[1]) for t in terms})
        lines = [f"case {spec_idx}: {{"]
        lines.append(
            f"    for ({channel_type} u = threadIdx.x; u < {in_channels}; u += blockDim.x) {{"
        )
        for k in range(d3):
            lines.append(f"        float acc{k} = 0.0f;")
        lines.extend(r"""        for (Index p = e_begin; p < e_end; ++p) {{
            const Index e = __ldg(edge_by_dst + p);
            const int s = __ldg(src + p);""".format().split("\n"))
        for i in unique_i:
            lines.append(
                f"        const float xv{i} = __ldg(x + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});"
            )
        for j in unique_j:
            lines.append(
                f"        const float yv{j} = __ldg(edge_attr + e * {in2_dim} + {edge_offset}{y_offset + j});"
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            lines.append(
                f"""        acc{int(k_idx)} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, xv{int(i_idx)} * yv{int(j_idx)}, acc{int(k_idx)});"""
            )
        lines.append("    }")
        for k in range(d3):
            lines.append(
                f"    q_group{group_id}[({row_offset} + u) * (num_nodes * int64_t({d3})) + Index(node) * {d3} + {k}] = acc{k};"
            )
        lines.append("}")
        lines.append("break;")
        lines.append("}")
        cases.append("\n".join(lines))

        pair_lines = [f"case {spec_idx}: {{"]
        pair_lines.append(
            f"    for ({channel_type} u = threadIdx.x; u < {in_channels}; u += blockDim.x) {{"
        )
        for k in range(d3):
            pair_lines.append(f"        float acc{k} = 0.0f;")
        pair_lines.extend(r"""        for (Index p = e_begin; p < e_end; ++p) {{
            const Index e = __ldg(edge_by_dst + p);
            const int s = __ldg(src + p);""".format().split("\n"))
        for i in unique_i:
            pair_lines.extend(
                f"            const float x0v{i} = __ldg(x0 + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});\n            const float x1v{i} = __ldg(x1 + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});".split(
                    "\n"
                )
            )
        for j in unique_j:
            pair_lines.extend(
                f"            const float y0v{j} = __ldg(edge0 + e * {in2_dim} + {edge_offset}{y_offset + j});\n            const float y1v{j} = __ldg(edge1 + e * {in2_dim} + {edge_offset}{y_offset + j});".split(
                    "\n"
                )
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            pair_lines.append(
                f"""        acc{int(k_idx)} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, x0v{int(i_idx)} * y0v{int(j_idx)} + x1v{int(i_idx)} * y1v{int(j_idx)}, acc{int(k_idx)});"""
            )
        pair_lines.append("    }")
        for k in range(d3):
            pair_lines.append(
                f"    q_group{group_id}[({row_offset} + u) * (num_nodes * int64_t({d3})) + Index(node) * {d3} + {k}] = acc{k};"
            )
        pair_lines.append("}")
        pair_lines.append("break;")
        pair_lines.append("}")
        pair_cases.append("\n".join(pair_lines))

    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_THREADS = {threads};
constexpr int GEN_NUM_GROUPS = {groups_count};


__global__ __launch_bounds__(GEN_THREADS, 4)
    void lunatp_uvwconv_bwd_weight_grouped_q_fp32(
        const float* __restrict__ x,
        const float* __restrict__ edge_attr,
        const int32_t* __restrict__ src,
        const EdgeIndex* __restrict__ edge_by_dst,
        const EdgeIndex* __restrict__ rowptr,
        {params}
        int64_t num_nodes) {{
    const Index node = blockIdx.x;
    const int spec = blockIdx.y;
    if (node >= num_nodes) return;
    const Index e_begin = __ldg(rowptr + node);
    const Index e_end = __ldg(rowptr + node + 1);
    switch (spec) {{
{cases}
        default:
            break;
    }}
}}

__global__ __launch_bounds__(GEN_THREADS, 4)
    void lunatp_uvwconv_bwd_weight_grouped_q_pair_fp32(
        const float* __restrict__ x0,
        const float* __restrict__ edge0,
        const float* __restrict__ x1,
        const float* __restrict__ edge1,
        const int32_t* __restrict__ src,
        const EdgeIndex* __restrict__ edge_by_dst,
        const EdgeIndex* __restrict__ rowptr,
        {params}
        int64_t num_nodes) {{
    const Index node = blockIdx.x;
    const int spec = blockIdx.y;
    if (node >= num_nodes) return;
    const Index e_begin = __ldg(rowptr + node);
    const Index e_end = __ldg(rowptr + node + 1);
    switch (spec) {{
{pair_cases}
        default:
            break;
    }}
}}

}} // namespace

std::vector<torch::Tensor> lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32(
    torch::Tensor x,
    torch::Tensor edge_attr,
    torch::Tensor src,
    torch::Tensor edge_by_dst,
    torch::Tensor rowptr,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto q_options = x.options();
    std::vector<torch::Tensor> q_groups;
{allocs}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_weight_grouped_q_fp32<<<dim3(static_cast<unsigned int>(num_nodes), GEN_NUM_SPECS), GEN_THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            edge_attr.data_ptr<float>(),
            src.data_ptr<int32_t>(),
            edge_by_dst.data_ptr<EdgeIndex>(),
            rowptr.data_ptr<EdgeIndex>(),
{ptrs}
            num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32(
    torch::Tensor x0,
    torch::Tensor edge0,
    torch::Tensor x1,
    torch::Tensor edge1,
    torch::Tensor src,
    torch::Tensor edge_by_dst,
    torch::Tensor rowptr,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x0.device());
    auto q_options = x0.options();
    std::vector<torch::Tensor> q_groups;
{allocs}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_weight_grouped_q_pair_fp32<<<dim3(static_cast<unsigned int>(num_nodes), GEN_NUM_SPECS), GEN_THREADS, 0, stream>>>(
            x0.data_ptr<float>(),
            edge0.data_ptr<float>(),
            x1.data_ptr<float>(),
            edge1.data_ptr<float>(),
            src.data_ptr<int32_t>(),
            edge_by_dst.data_ptr<EdgeIndex>(),
            rowptr.data_ptr<EdgeIndex>(),
{ptrs}
            num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32", &lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32, "Graph-conv shared grouped Q precontraction");
    m.def("lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32", &lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32, "Graph-conv shared grouped pair-summed Q precontraction");
}}
""".format(
            specs_count=len(specs),
            threads=threads,
            groups_count=len(groups),
            params=chr(10).join(params),
            cases=textwrap.indent(chr(10).join(cases), '        '),
            pair_cases=textwrap.indent(chr(10).join(pair_cases), '        '),
            allocs=textwrap.indent(chr(10).join(allocs), '    '),
            ptrs=textwrap.indent(chr(10).join(ptrs), '    '),
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_weight_cutlass_source(meta):
    """Reduce Q × gradient with one fused CUTLASS mainloop and atomic epilogue."""
    groups, group_rows, _ = get_uvw_conv_weight_layout(meta["generated_specs"])
    tiles = sum(((rows + 63) // 64) * ((key[1] + 63) // 64)
                for rows, (key, _) in zip(group_rows, groups))
    split_cap = max(MIN_WEIGHT_SPLITS, min(MAX_WEIGHT_SPLITS, MAX_WEIGHT_BLOCKS // max(1, tiles)))
    declarations = []
    cases = []
    for group, ((offset, channels, d), specs) in enumerate(groups):
        stores = '\n'.join(f'if(r>={s["row_offset"]} && r<{s["row_offset"]+s["path_channels"]}) atomicAdd(pointer+{s["weight_offset"]}+int64_t(r-{s["row_offset"]})*{channels}+c,value);' for s in specs)
        declarations.append(f'''
template <typename Iterator> struct WeightOutput{group} : Iterator {{
    using ThreadMap = typename Iterator::ThreadMap;
    using Fragment = typename Iterator::Fragment;
    using Params = typename Iterator::Params;
    float *pointer;
    CUTLASS_DEVICE WeightOutput{group}(Params const &p, float *out, cutlass::MatrixCoord extent, int thread,
    cutlass::MatrixCoord offset = {{}}, int const *indices = nullptr)
    : Iterator(p, out, extent, thread, offset, indices), pointer(out) {{
    }}
    CUTLASS_DEVICE void store(Fragment const &fragment) const {{
        typename Iterator::Mask mask;
        this->get_mask(mask);
#pragma unroll
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {{
#pragma unroll
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {{
#pragma unroll
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {{
                    int r = this->thread_start_row() + row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                    cluster * ThreadMap::Delta::kCluster;
                    int fr = row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
#pragma unroll
                    for (int col = 0; col < ThreadMap::Iterations::kColumn; ++col) {{
                        int c = this->thread_start_column() + col * ThreadMap::Delta::kColumn;
                        float value = fragment[fr * ThreadMap::Iterations::kColumn + col];
                        if (r < this->extent_row() && mask.predicates[col]) {{
{textwrap.indent(stores, '                            ')}
                        }}
                    }}
                }}
            }}
        }}
    }}
}};
using Gemm{group} = cutlass::gemm::device::GemmUniversal<
float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 8>,
cutlass::gemm::GemmShape<32, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, cutlass::arch::OpMultiplyAdd,
cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone, false, false, false, cutlass::layout::NoPermute,
cutlass::layout::NoPermute, GradLayout<{d}, {meta['out_dim']}>>;
''')
        cases.append(f'''
case {group}: {{
    using Kernel=typename Gemm{group}::GemmKernel;
    using Mma=typename Kernel::Mma;
    using Base=typename Kernel::Epilogue;
    using Epilogue=cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,typename Base::WarpMmaOperator,Base::kPartitionsK,
    WeightOutput{group}<typename Base::OutputTileIterator>,typename Base::AccumulatorFragmentIterator,typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,typename Base::OutputOp,typename Base::Padding>;
    auto& storage=shared.s{group};
    const int row=blockIdx.y*64,column=blockIdx.x*64;
    if(row>={group_rows[group]} || column>={channels}) return;
    const int64_t total_k=nodes*{d};
    // Partition by complete nodes so the gradient AoS origin stays aligned.
    const int64_t nodes_per_split=(nodes+splits-1)/splits;
    const int64_t first=split*nodes_per_split,count=min(nodes_per_split,nodes-first);
    if(count<=0) return;
    typename Mma::FragmentC accum;accum.clear();
#pragma unroll
    for(int p=0;p<Products;++p) {{
        for(int64_t chunk=0;chunk<count;chunk+=(INT_MAX-8)/{d}) {{
            const int k=min(int64_t((INT_MAX-8)/{d}),count-chunk)*{d};
            typename Mma::IteratorA a(typename Mma::IteratorA::Params(cutlass::layout::RowMajor(total_k)),
            const_cast<float*>(inputs.q[p][{group}]+(first+chunk)*{d}),{{{group_rows[group]},k}},threadIdx.x,{{row,0}});
            typename Mma::IteratorB b(typename Mma::IteratorB::Params(cutlass::layout::RowMajor({meta['out_dim']})),
            const_cast<float*>(inputs.g[p]+(first+chunk)*{meta['out_dim']}+{offset}),{{k,{channels}}},threadIdx.x,{{0,column}});
            Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
            mma((k+7)/8,accum,a,b,accum);__syncthreads();
        }}
    }}
    typename Epilogue::OutputTileIterator output(typename Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor({channels})),out,{{{group_rows[group]},{channels}}},threadIdx.x,{{row,column}});
    Epilogue epilogue(storage.epilogue,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    epilogue(typename Kernel::EpilogueOutputOp(1.0f,0.0f),output,accum,output);
    return;
}}
''')
    source = f'''
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cutlass/gemm/device/gemm_universal.h>
#include <limits.h>
struct Inputs {{const float* q[2][{len(groups)}];const float* g[2];}};
template<int Components,int Stride> struct GradLayout {{
    using Index=int32_t;using LongIndex=int64_t;
    CUTLASS_HOST_DEVICE GradLayout(cutlass::layout::PitchLinearCoord,Index) {{}}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord q) const {{return LongIndex(q.strided()/Components)*Stride+LongIndex(q.contiguous())*Components+q.strided()%Components;}}
}};
{''.join(declarations)}
template<int Products> __global__ void lunatp_uvwconv_bwd_weight_shared_fp32(Inputs inputs,float* out,int64_t nodes) {{
    union Storage {{ {' '.join(f"typename Gemm{i}::GemmKernel::SharedStorage s{i};" for i in range(len(groups)))} }};
    __shared__ Storage shared;
    const int splits=min(int64_t({split_cap}),max(int64_t({MIN_WEIGHT_SPLITS}),(nodes+{WEIGHT_SPLIT_NODES-1})/{WEIGHT_SPLIT_NODES}));
    const int split=blockIdx.z%splits;
    switch(blockIdx.z/splits) {{{chr(10).join(cases)}}}
}}

template<int Products>
at::Tensor reduce(Inputs inputs,at::Tensor grad) {{
    c10::cuda::CUDAGuard guard(grad.device());
    auto out=at::zeros({{{meta['weight_numel']}}},grad.options());
    if(grad.size(0)==0) return out;
    const int64_t splits=std::min<int64_t>({split_cap},std::max<int64_t>({MIN_WEIGHT_SPLITS},(grad.size(0)+{WEIGHT_SPLIT_NODES-1})/{WEIGHT_SPLIT_NODES}));
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_weight_shared_fp32<Products><<<dim3({max((key[1]+63)//64 for key,_ in groups)},{max((r+63)//64 for r in group_rows)},splits*{len(groups)}),128,0,stream>>>(inputs,out.data_ptr<float>(),grad.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}}
at::Tensor launch(std::vector<at::Tensor> q,at::Tensor g) {{
    Inputs inputs{{}};
    for(int i=0;i<{len(groups)};++i) inputs.q[0][i]=q[i].data_ptr<float>();
    inputs.g[0]=g.data_ptr<float>();return reduce<1>(inputs,g);
}}
at::Tensor launch_pair(std::vector<at::Tensor> q0,std::vector<at::Tensor> q1,at::Tensor g0,at::Tensor g1) {{
    Inputs inputs{{}};
    for(int i=0;i<{len(groups)};++i) {{inputs.q[0][i]=q0[i].data_ptr<float>();inputs.q[1][i]=q1[i].data_ptr<float>();}}
    inputs.g[0]=g0.data_ptr<float>();inputs.g[1]=g1.data_ptr<float>();return reduce<2>(inputs,g0);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);m.def("launch_pair",&launch_pair);}}
'''
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def make_uvw_conv_tr_kernel(meta, program, inputs, outputs, kernel_name):
    """Stream each W cell through factored T/R contractions for one edge channel."""
    plan = make_tr_plan(program, outputs)
    threads, path_tile, warps = 128, 128, 4
    tasks, cases = 0, []
    t_storage, r_storage = 1, 1
    for spec in meta["generated_specs"]:
        d1, d2, d3 = (spec[key] for key in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (spec[key] for key in ("x_offset", "y_offset", "out_offset", "weight_offset"))
        edges, channels, paths = (spec[key] for key in ("edge_channels", "out_channels", "path_channels"))
        columns = get_uvw_conv_tr_layout(meta, program, outputs, spec)["column_tile"]
        parts, u_channels = columns // 8, paths // edges
        column_tiles = (channels + columns - 1) // columns
        u_tiles = (u_channels + path_tile - 1) // path_tile
        count = edges * u_tiles * column_tiles
        t_storage = max(t_storage, len(plan["t"]) * d1 * columns)
        r_storage = max(r_storage, len(plan["r"]) * warps * d1 * columns)
        lines = [f"if(local<{tasks+count}) {{",
                 f"    const int v=(local-{tasks})%{edges};",
                 f"    const int64_t begin=(local-{tasks})/{edges}/{column_tiles}*{path_tile};",
                 f"    const int column_base=(local-{tasks})/{edges}%{column_tiles}*{columns};"]
        for ti, group in enumerate(plan["t"]):
            lines += [f"    for(int c=threadIdx.x;c<{columns};c+={threads}) {{",
                      "        const int column=column_base+c;",
                      *[f"        float t{i}=0.f;" for i in range(d1)], f"        if(column<{channels}) {{"]
            for name in sorted({pair[0] for pair in group["products"]}):
                for j in sorted({term[1] for term in spec["terms"]}):
                    lines.append(f"            const float {name}{j}=__ldg({name}+int64_t(item)*{meta['in2_dim']}+{yo}+v*{d2}+{j});")
            for name in sorted({pair[1] for pair in group["products"]}):
                for k in sorted({term[2] for term in spec["terms"]}):
                    lines.append(f"            const float {name}{k}=__ldg({name}+int64_t(dest_node)*{meta['out_dim']}+{oo}+column*{d3}+{k});")
            for i, j, k, coefficient in spec["terms"]:
                for edge, grad in group["products"]:
                    lines.append(f"            t{i}=fmaf({float_literal(coefficient,meta.get('dtype'))}*{edge}{j},{grad}{k},t{i});")
            lines += ["        }", *[f"        t_values[{(ti*d1+i)*columns}+c]=t{i};" for i in range(d1)], "    }"]
        lines.append("    __syncthreads();")
        for ri in range(len(plan["r"])):
            lines += [f"    float r{ri}_{i}_{part}=0.f;" for i in range(d1) for part in range(parts)]
        x_targets = tuple(target for target in program["x"] if target in outputs)
        w_targets = tuple(target for target in program["w"] if target in outputs)
        x_names = {feature for group in plan["r"] for feature, _ in group["products"]}
        x_names.update(name for group in plan["t"] for (target, name), _ in group["consumers"] if target in w_targets)
        weights = {weight for group in plan["r"] for _, weight in group["products"]}
        weights.update(name for group in plan["t"] for (target, name), _ in group["consumers"] if target in x_targets)
        lines += [f"    for(int row=threadIdx.x/8;row<{path_tile};row+=16) {{",
                  "        const int64_t u=begin+row;",
                  f"        const unsigned mask=__ballot_sync(0xffffffffu,u<{u_channels});",
                  f"        if(u<{u_channels}) {{"]
        for target in x_targets:
            lines += [f"            float {target}{i}=0.f;" for i in range(d1)]
        for name in sorted(x_names):
            lines += [f"            const float {name}{i}=__ldg({name}+int64_t(source_node)*{meta['in1_dim']}+{xo}+u*{d1}+{i});" for i in range(d1)]
        for part in range(parts):
            lines += ["            {", f"                const int c={part*8}+lane%8,column=column_base+c;", f"                if(column<{channels}) {{"]
            for weight in sorted(weights):
                lines.append(f"                    const float value_{weight}=__ldg({weight}+int64_t(item)*{meta['weight_numel']}+{wo}+(u*{edges}+v)*{channels}+column);")
            for target in w_targets:
                lines.append(f"                    float value_{target}=0.f;")
            for ti, group in enumerate(plan["t"]):
                lines.append("                    {")
                for i in range(d1):
                    lines.append(f"                        const float t{i}=t_values[{(ti*d1+i)*columns}+c];")
                for (target, name), repeat in group["consumers"]:
                    for _ in range(repeat):
                        for i in range(d1):
                            if target in x_targets:
                                lines.append(f"                        {target}{i}=fmaf(value_{name},t{i},{target}{i});")
                            else:
                                lines.append(f"                        value_{target}=fmaf({name}{i},t{i},value_{target});")
                lines.append("                    }")
            for ri, group in enumerate(plan["r"]):
                for feature, weight in group["products"]:
                    lines += [f"                    r{ri}_{i}_{part}=fmaf({feature}{i},value_{weight},r{ri}_{i}_{part});" for i in range(d1)]
            for target in w_targets:
                lines.append(f"                    {target}[int64_t(item)*{meta['weight_numel']}+{wo}+(u*{edges}+v)*{channels}+column]=value_{target};")
            lines += ["                }", "            }"]
        for target in x_targets:
            for i in range(d1):
                lines += ["#pragma unroll", "            for(int delta=4;delta;delta/=2)",
                          f"            {target}{i}+=__shfl_down_sync(mask,{target}{i},delta,8);",
                          f"            if(lane%8==0)atomicAdd({target}+int64_t(source_node)*{meta['in1_dim']}+{xo}+u*{d1}+{i},{target}{i});"]
        lines += ["        }", "    }", "#pragma unroll", "    for(int delta=16;delta>=8;delta/=2) {"]
        for ri in range(len(plan["r"])):
            lines += [f"        r{ri}_{i}_{part}+=__shfl_down_sync(0xffffffffu,r{ri}_{i}_{part},delta);" for i in range(d1) for part in range(parts)]
        lines += ["    }", "    if(lane<8) {"]
        for ri in range(len(plan["r"])):
            lines += [f"        r_partial[(({ri}*{warps}+warp)*{d1}+{i})*{columns}+{part*8}+lane]=r{ri}_{i}_{part};" for i in range(d1) for part in range(parts)]
        lines += ["    }", "    __syncthreads();", f"    if(threadIdx.x<{columns}) {{",
                  "        const int c=threadIdx.x,column=column_base+c;",
                  f"        const int active_columns=min(32,{columns}-(threadIdx.x/32)*32);"]
        y_targets = tuple(target for target in program["y"] if target in outputs)
        for target in y_targets:
            lines += [f"        float {target}{j}=0.f;" for j in range(d2)]
        if "gg" in outputs:
            lines += [f"        float gg{k}=0.f;" for k in range(d3)]
        lines.append(f"        if(column<{channels}) {{")
        for ri, group in enumerate(plan["r"]):
            lines.append("            {")
            for i in range(d1):
                total = "+".join(f"r_partial[{((ri*warps+w)*d1+i)*columns}+c]" for w in range(warps))
                lines.append(f"                const float r{i}={total};")
            for (target, name), repeat in group["consumers"]:
                lines.append("                {")
                if target in y_targets:
                    lines += [f"                    const float a{k}=__ldg({name}+int64_t(dest_node)*{meta['out_dim']}+{oo}+column*{d3}+{k});" for k in sorted({term[2] for term in spec["terms"]})]
                else:
                    lines += [f"                    const float a{j}=__ldg({name}+int64_t(item)*{meta['in2_dim']}+{yo}+v*{d2}+{j});" for j in sorted({term[1] for term in spec["terms"]})]
                for _ in range(repeat):
                    for i, j, k, coefficient in spec["terms"]:
                        component, value = (j, k) if target in y_targets else (k, j)
                        lines.append(f"                    {target}{component}=fmaf({float_literal(coefficient,meta.get('dtype'))}*r{i},a{value},{target}{component});")
                lines.append("                }")
            lines.append("            }")
        if "gg" in outputs:
            lines += [f"            atomicAdd(gg+int64_t(dest_node)*{meta['out_dim']}+{oo}+column*{d3}+{k},gg{k});" for k in range(d3)]
        lines += ["        }", "        const unsigned active=__activemask();"]
        for target in y_targets:
            for j in range(d2):
                lines += ["#pragma unroll", "        for(int delta=16;delta;delta/=2) {",
                          f"            const float other=__shfl_down_sync(active,{target}{j},delta);",
                          f"            if(lane+delta<active_columns){target}{j}+=other;", "        }",
                          f"        if(lane==0)atomicAdd({target}+int64_t(item)*{meta['in2_dim']}+{yo}+v*{d2}+{j},{target}{j});"]
        lines += ["    }", "    return;", "}"]
        cases.append("\n".join(lines))
        tasks += count
    parameters = [f"const float* __restrict__ {name}" for name in inputs]
    parameters += ["const int32_t* __restrict__ graph_src", "const int32_t* __restrict__ graph_dst"]
    parameters += [f"float* __restrict__ {name}" for name in outputs]
    source = f"""
__global__ __launch_bounds__({threads}) void {kernel_name}({','.join(parameters)}) {{
    const int64_t task = blockIdx.x, item = task / {tasks}, local = task % {tasks};
    const int source_node = graph_src[item], dest_node = graph_dst[item];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float t_values[{t_storage}];
    __shared__ float r_partial[{r_storage}];
{textwrap.indent(chr(10).join(cases), '    ')}
}}
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32)), tasks, threads


def select_uvw_conv_t_specs(meta, outputs):
    """Choose the lower-FMA contraction order within one CTA's shared budget."""
    if not all(name in outputs for name in ("gx", "gy", "gw")):
        return ()
    specs = meta["generated_specs"]
    max_paths = max(spec["path_channels"] for spec in specs)
    path_tile = min(128, max(32, 1 << (max_paths - 1).bit_length()))
    element_size = 8 if meta.get("dtype", torch.float32) == torch.float64 else 4
    cg_values = path_tile * max(spec["d3"] for spec in specs)
    edge_values = max(spec["edge_channels"] * spec["d2"] for spec in specs)
    available = 32 * 1024 - (cg_values + edge_values) * element_size
    selected = []
    for spec in specs:
        columns = min(64, max(32, 1 << (spec["out_channels"] - 1).bit_length()))
        values = spec["d1"] * spec["edge_channels"] * columns
        if 3 * spec["d1"] < 2 * spec["d3"] and values * (1 + path_tile // 32) * element_size <= available:
            selected.append(spec)
    return tuple(selected)


def emit_uvw_conv_t_backward(meta, spec, path_tile, threads, task_begin):
    """Own one v so each W cell updates one R accumulator without predicates."""
    d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
    xo, yo, oo, wo = (spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset"))
    edges, channels, paths = (spec[k] for k in ("edge_channels", "out_channels", "path_channels"))
    column_tile = min(64, max(32, 1 << (channels - 1).bit_length()))
    column_parts = column_tile // 8
    column_tiles = (channels + column_tile - 1) // column_tile
    warps = threads // 32
    rows = (paths + edges - 1) // edges
    count = ((rows + path_tile - 1) // path_tile) * edges * column_tiles
    lines = [f"if (local < {task_begin + count}) {{",
             f"    const int64_t begin = (local - {task_begin}) / {edges * column_tiles} * {path_tile};",
             f"    const int64_t v = (local - {task_begin}) / {column_tiles} % {edges};",
             f"    const int column_base = (local - {task_begin}) % {column_tiles} * {column_tile};",
             f"    for (int c = threadIdx.x; c < {column_tile}; c += {threads}) {{",
             "        const int column = column_base + c;"]
    lines += [f"        float t{i} = 0.f;" for i in range(d1)]
    lines.append(f"        if (column < {channels}) {{")
    lines += [f"            const float y{j} = __ldg(y + int64_t(item) * {meta['in2_dim']} + {yo} + v * {d2} + {j});"
              for j in sorted({term[1] for term in spec["terms"]})]
    lines += [f"            const float g{k} = __ldg(g + int64_t(dest_node) * {meta['out_dim']} + {oo} + column * {d3} + {k});"
              for k in sorted({term[2] for term in spec["terms"]})]
    for i, j, k, coefficient in spec["terms"]:
        lines.append(f"            t{i} = fmaf({float_literal(coefficient, meta.get('dtype'))} * y{j}, g{k}, t{i});")
    lines.append("        }")
    lines += [f"        t_values[{i} * {column_tile} + c] = t{i};" for i in range(d1)]
    lines += ["    }", "    __syncthreads();"]
    lines += [f"    float r{i}_{part} = 0.f;" for i in range(d1) for part in range(column_parts)]
    lines += [f"    for (int row = warp * 4 + lane / 8; row < {path_tile}; row += {warps * 4}) {{",
              f"        const int64_t u = begin + row, p = u * {edges} + v;",
              f"        const unsigned active = __ballot_sync(0xffffffffu, p < {paths});",
              f"        if (p < {paths}) {{"]
    lines += [f"            float dx{i} = 0.f;" for i in range(d1)]
    lines += [f"            const float x{i} = __ldg(x + int64_t(source_node) * {meta['in1_dim']} + {xo} + u * {d1} + {i});" for i in range(d1)]
    for part in range(column_parts):
        lines += ["            {", f"                const int column = column_base + {part * 8} + lane % 8;",
                  f"                if (column < {channels}) {{",
                  f"                    const float value = __ldg(w + int64_t(item) * {meta['weight_numel']} + {wo} + p * {channels} + column);",
                  "                    float dw = 0.f;"]
        for i in range(d1):
            lines += [f"                    const float t{i} = t_values[{i} * {column_tile} + {part * 8} + lane % 8];",
                      f"                    dx{i} = fmaf(value, t{i}, dx{i});",
                      f"                    dw = fmaf(x{i}, t{i}, dw);",
                      f"                    r{i}_{part} = fmaf(x{i}, value, r{i}_{part});"]
        lines += [f"                    gw[int64_t(item) * {meta['weight_numel']} + {wo} + p * {channels} + column] = dw;", "                }", "            }"]
    lines += ["#pragma unroll", "            for (int delta = 4; delta; delta /= 2) {"]
    lines += [f"                dx{i} += __shfl_down_sync(active, dx{i}, delta, 8);" for i in range(d1)]
    lines += ["            }", "            if (lane % 8 == 0) {"]
    lines += [f"                atomicAdd(gx + int64_t(source_node) * {meta['in1_dim']} + {xo} + u * {d1} + {i}, dx{i});" for i in range(d1)]
    lines += ["            }", "        }", "    }"]
    lines += ["#pragma unroll", "    for (int delta = 16; delta >= 8; delta /= 2) {"]
    lines += [f"        r{i}_{part} += __shfl_down_sync(0xffffffffu, r{i}_{part}, delta);"
              for i in range(d1) for part in range(column_parts)]
    lines += ["    }", "    if (lane < 8) {"]
    lines += [f"        r_partial[(warp * {d1} + {i}) * {column_tile} + {part * 8} + lane] = r{i}_{part};"
              for i in range(d1) for part in range(column_parts)]
    lines += ["    }", "    __syncthreads();", "    if (warp == 0) {"]
    lines += [f"        float dy{j} = 0.f;" for j in range(d2)]
    for part in range(column_tile // 32):
        lines += ["        {", f"            const int column = column_base + {part * 32} + lane;"]
        for i in range(d1):
            values = " + ".join(f"r_partial[{(warp * d1 + i) * column_tile + part * 32} + lane]" for warp in range(warps))
            lines.append(f"            const float r{i} = {values};")
        lines.append(f"            if (column < {channels}) {{")
        lines += [f"                const float g{k} = __ldg(g + int64_t(dest_node) * {meta['out_dim']} + {oo} + column * {d3} + {k});"
                  for k in sorted({term[2] for term in spec["terms"]})]
        for i, j, k, coefficient in spec["terms"]:
            lines.append(f"                dy{j} = fmaf({float_literal(coefficient, meta.get('dtype'))} * r{i}, g{k}, dy{j});")
        lines += ["            }", "        }"]
    lines += ["#pragma unroll", "        for (int delta = 16; delta; delta /= 2) {"]
    lines += [f"            dy{j} += __shfl_down_sync(0xffffffffu, dy{j}, delta);" for j in range(d2)]
    lines += ["        }", "        if (lane == 0) {"]
    lines += [f"            atomicAdd(gy + int64_t(item) * {meta['in2_dim']} + {yo} + v * {d2} + {j}, dy{j});" for j in range(d2)]
    lines += ["        }", "    }", "    return;", "}"]
    return lines, count


def make_uvw_conv_input_gradient_kernel(meta, program, inputs, outputs, kernel_name):
    """Stream eight-lane channel reductions, then consume CG paths in parallel."""
    max_paths = max(spec["path_channels"] for spec in meta["generated_specs"])
    path_tile = min(128, max(32, 1 << (max_paths - 1).bit_length()))
    threads = path_tile
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    needed_z = {pair[1] for pairs in (*x_outputs.values(), *y_outputs.values()) for pair in pairs}
    contractions = {k: v for k, v in program["z"].items() if k in needed_z}
    cases, tasks = [], 0
    max_edge = max(s["edge_channels"] * s["d2"] for s in meta["generated_specs"])
    max_components = max(s["d3"] for s in meta["generated_specs"])
    t_specs = select_uvw_conv_t_specs(meta, outputs)
    for spec in meta["generated_specs"]:
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset"))
        edges, channels, paths = (spec[k] for k in ("edge_channels", "out_channels", "path_channels"))
        if spec in t_specs:
            lines, count = emit_uvw_conv_t_backward(meta, spec, path_tile, threads, tasks)
            cases.append("\n".join(lines))
            tasks += count
            continue
        count = (paths + path_tile - 1) // path_tile
        lines = [f"if (local < {tasks + count}) {{",
                 f"    const int64_t begin = (local - {tasks}) * {path_tile};"]
        for target in y_outputs:
            lines.append(f"    for (int i = threadIdx.x; i < {edges * d2}; i += {threads}) shared_{target}[i] = 0.f;")
        lines += [f"    for (int path = warp * 4 + lane / 8; path < {path_tile}; path += {threads // 8}) {{",
                  f"        const int64_t p = begin + path, u = p / {edges}, v = p % {edges};",
                  f"        const unsigned mask = __ballot_sync(0xffffffffu, p < {paths});",
                  f"        if (p < {paths}) {{"]
        for zname in contractions:
            lines += [f"            float {zname}_{k} = 0.f;" for k in range(d3)]
        lines += ["#pragma unroll 1", f"            for (int column_base = (lane % 8) * 4; column_base < {channels}; column_base += 32) {{",
                  "#pragma unroll", "                for (int value = 0; value < 4; ++value) {",
                  "                    const int column = column_base + value;",
                  f"                    if (column < {channels}) {{"]
        weights = tuple(dict.fromkeys(w for pairs in contractions.values() for w, _ in pairs))
        gradients = tuple(dict.fromkeys(g for pairs in contractions.values() for _, g in pairs))
        for weight in weights:
            lines.append(f"                        const float value_{weight} = __ldg({weight} + int64_t(item) * {meta['weight_numel']} + {wo} + p * {channels} + column);")
        for grad in gradients:
            lines += [f"                        const float value_{grad}_{k} = __ldg({grad} + int64_t(dest_node) * {meta['out_dim']} + {oo} + column * {d3} + {k});" for k in range(d3)]
        for zname, pairs in contractions.items():
            for w, g in pairs:
                lines += [f"                        {zname}_{k} = fmaf(value_{w}, value_{g}_{k}, {zname}_{k});" for k in range(d3)]
        lines += ["                    }", "                }", "            }", "#pragma unroll", "            for (int delta = 4; delta; delta /= 2) {"]
        for zname in contractions:
            lines += [f"                {zname}_{k} += __shfl_down_sync(mask, {zname}_{k}, delta, 8);" for k in range(d3)]
        lines += ["            }", "            if (lane % 8 == 0) {"]
        for zi, zname in enumerate(contractions):
            lines += [f"                z[path][{zi * max_components + k}] = {zname}_{k};" for k in range(d3)]
        lines += ["            }", "        }", "    }", "    __syncthreads();",
                  f"    if (begin + threadIdx.x < {paths}) {{",
                  f"        const int64_t p = begin + threadIdx.x, u = p / {edges}, v = p % {edges};"]
        for axis, targets in (("x", x_outputs), ("y", y_outputs)):
            for target, pairs in targets.items():
                width = d1 if axis == "x" else d2
                lines += ["        {", f"            float result[{width}] = {{}};"]
                for pair in pairs:
                    name, zname = pair[:2]
                    factor = f"*__ldg({pair[2]} + item)" if len(pair) == 3 and pair[2] is not None else ""
                    for i, j, k, coeff in spec["terms"]:
                        addr = (f"{name} + int64_t(item) * {meta['in2_dim']} + {yo} + v * {d2} + {j}" if axis == "x" else
                                f"{name} + int64_t(source_node) * {meta['in1_dim']} + {xo} + u * {d1} + {i}")
                        component = i if axis == "x" else j
                        lines.append(f"            result[{component}] = fmaf({float_literal(coeff, meta.get('dtype'))} * __ldg({addr}){factor}, z[threadIdx.x][{list(contractions).index(zname) * max_components + k}], result[{component}]);")
                for c in sorted({term[0 if axis == "x" else 1] for term in spec["terms"]}):
                    if axis == "x":
                        lines.append(f"            atomicAdd({target} + int64_t(source_node) * {meta['in1_dim']} + {xo} + u * {d1} + {c}, result[{c}]);")
                    else:
                        lines.append(f"            atomicAdd(shared_{target} + v * {d2} + {c}, result[{c}]);")
                lines.append("        }")
        lines.append("    }")
        lines.append("    __syncthreads();")
        for target in y_outputs:
            lines += [f"    for (int i = threadIdx.x; i < {edges * d2}; i += {threads}) {{",
                      f"        atomicAdd({target} + int64_t(item) * {meta['in2_dim']} + {yo} + i, shared_{target}[i]);", "    }"]
        lines += ["    return;", "}"]
        cases.append("\n".join(lines))
        tasks += count
    parameters = [f"const float* {name}" for name in inputs]
    parameters += ["const int32_t* graph_src", "const int32_t* graph_dst"]
    parameters += [f"float* {name}" for name in outputs]
    scratch = "\n".join(f"__shared__ float shared_{target}[{max_edge}];" for target in y_outputs)
    scratch += f"\n__shared__ float z[{path_tile}][{len(contractions) * max_components}];"
    if t_specs:
        values = max(s["d1"] * min(64, max(32, 1 << (s["out_channels"] - 1).bit_length())) for s in t_specs)
        scratch += f"\n__shared__ float t_values[{values}], r_partial[{threads // 32 * values}];"
    source = f"""
__global__ __launch_bounds__({threads}) void {kernel_name}({','.join(parameters)}) {{
    const int64_t task = blockIdx.x, item = task / {tasks}, local = task % {tasks};
    const int source_node = graph_src[item], dest_node = graph_dst[item];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
{textwrap.indent(scratch, '    ')}
{textwrap.indent(chr(10).join(cases), '    ')}
}}
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32)), tasks, threads


def make_uvw_conv_backward_external_kernel(meta, program, inputs, outputs, kernel_name):
    """Reuse each weight in both channel and path contractions within a CTA."""
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    e_outputs = {k: v for k, v in program.get("e", {}).items() if k in outputs}
    needed_z = {p[1] for pairs in (*x_outputs.values(), *y_outputs.values(), *e_outputs.values()) for p in pairs}
    contractions = {k: v for k, v in program["z"].items() if k in needed_z}
    output_products = program["forward"] if "gg" in outputs else ()
    weight_groups = {}
    for zi, pairs in enumerate(contractions.values()):
        for weight, grad in pairs:
            weight_groups.setdefault(weight, {}).setdefault(grad, []).append(zi)
    for _, weight in output_products:
        weight_groups.setdefault(weight, {})
    queries = tuple(dict.fromkeys(q for q, _ in output_products))
    layout = get_uvw_conv_backward_layout(meta, program, outputs)
    path_tile, column_tile = layout["path_tile"], layout["column_tile"]
    threads, groups = layout["threads"], layout["groups"]
    max_components = layout["components"]
    z_stride, partial_stride = layout["z_stride"], layout["partial_stride"]
    node_centered = bool(x_outputs)
    tasks, cases = 0, []
    for spec in meta["generated_specs"]:
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset"))
        edges, channels, paths = (spec[k] for k in ("edge_channels", "out_channels", "path_channels"))
        count = (paths + path_tile - 1) // path_tile
        lines = [f"if(local<{tasks+count}) {{", f"    const int64_t begin=(local-{tasks})*{path_tile};"]
        if node_centered:
            for target in x_outputs:
                lines.append(f"    float node_{target}[{d1}]={{}};")
            lines += [f"    for(int64_t position=int64_t(src_rowptr[source_node])+partition;position<src_rowptr[source_node+1];position+={BACKWARD_NEIGHBOR_SPLITS}) {{",
                      "        const EdgeIndex item=edge_by_src[position];",
                      "        const int dest_node=graph_dst[item];"]
        for target in y_outputs:
            lines.append(f"    for(int i=threadIdx.x;i<{edges*d2};i+={threads})edge_{target}[i]=0.f;")
        for target in e_outputs:
            lines.append(f"    if(threadIdx.x==0)scalar_{target}=0.f;")
        lines += [f"    for(int i=threadIdx.x;i<{path_tile*z_stride};i+={threads})z[i]=0.f;", "    __syncthreads();"]
        for qi, query in enumerate(queries):
            lines += [f"    if(threadIdx.x<{path_tile} && begin+threadIdx.x<{paths}) {{",
                      f"        const int64_t p=begin+threadIdx.x,u=p/{edges},v=p%{edges};",
                      *[f"        float z{k}=0.f;" for k in range(d3)],
                      textwrap.indent(emit_cg_source(meta, spec, pairs=program["cg"][query], x_row="source_node"), '        '),
                      *[f"        query_values[({qi}*{path_tile}+threadIdx.x)*{max_components}+{k}]=z{k};" for k in range(d3)], "    }"]
        if queries:
            lines.append("    __syncthreads();")
        weight_items = tuple(weight_groups.items())
        cached_gradients = tuple(dict.fromkeys(
            grad for products in contractions.values() for _, grad in products
        ))
        ncolumns = (min(channels, column_tile) + 7) // 8
        lines += ["    {", f"        for(int column_base=0;column_base<{channels};column_base+={column_tile}) {{"]
        for grad in cached_gradients:
            lines += [f"            float cached_{grad}[{ncolumns*d3}]={{}};",
                      "#pragma unroll", f"            for(int tile=0;tile<{ncolumns};++tile) {{",
                      "                const int column=column_base+tile*8+lane;", f"                if(column<{channels}) {{"]
            for k in range(d3):
                lines.append(f"                    cached_{grad}[tile*{d3}+{k}]=__ldg({grad}+int64_t(dest_node)*{meta['out_dim']}+{oo}+column*{d3}+{k});")
            lines += ["                }", "            }"]
        if output_products:
            lines.append(f"            float output_accum[{ncolumns*d3}]={{}};")
        for first in range(0, len(weight_items), 2):
            weights = tuple(weight for weight, _ in weight_items[first:first+2])
            updates = {
                zi: tuple((weight, grad) for weight, grad in products if weight in weights)
                for zi, products in enumerate(contractions.values())
            }
            updates = {zi: products for zi, products in updates.items() if products}
            pairs = {weight: tuple(q for q, w in output_products if w == weight) for weight in weights}
            gradients = tuple(dict.fromkeys(
                grad for products in updates.values() for _, grad in products
            ))
            lines.append("            {")
            lines.append(f"                for(int row=group;row<{path_tile} && begin+row<{paths};row+={groups}) {{")
            for zi in updates:
                lines.append(f"                    float grad_accum{zi}[{d3}]={{}};")
            for weight, queries_for_weight in pairs.items():
                for k in range(d3):
                    if queries_for_weight:
                        value = "+".join(f"query_values[({queries.index(q)}*{path_tile}+row)*{max_components}+{k}]" for q in queries_for_weight)
                        lines.append(f"                    const float q_{weight}{k}={value};")
            lines += ["#pragma unroll", f"                    for(int tile=0;tile<{ncolumns};++tile) {{",
                      "                        const int column=column_base+tile*8+lane;", f"                        if(column<{channels}) {{"]
            for weight in weights:
                lines.append(f"                            const float value_{weight}=__ldg({weight}+int64_t(item)*{meta['weight_numel']}+{wo}+(begin+row)*{channels}+column);")
            for grad in gradients:
                for k in range(d3):
                    lines.append(f"                            const float value_{grad}{k}=cached_{grad}[tile*{d3}+{k}];")
            for zi, products in updates.items():
                for weight, grad in products:
                    for k in range(d3):
                        lines.append(f"                            grad_accum{zi}[{k}]=fmaf(value_{weight},value_{grad}{k},grad_accum{zi}[{k}]);")
            for weight, queries_for_weight in pairs.items():
                if queries_for_weight:
                    for k in range(d3):
                        lines.append(f"                            output_accum[tile*{d3}+{k}]=fmaf(value_{weight},q_{weight}{k},output_accum[tile*{d3}+{k}]);")
            lines += ["                        }", "                    }"]
            for zi in updates:
                for k in range(d3):
                    lines += ["#pragma unroll", "                    for(int delta=4;delta>0;delta/=2)",
                              f"                    grad_accum{zi}[{k}]+=__shfl_down_sync(0xffu << (warp_lane & ~7),grad_accum{zi}[{k}],delta,8);",
                              f"                    if(lane==0)z[row*{z_stride}+{zi*d3+k}]+=grad_accum{zi}[{k}];"]
            lines.append("                }")
            lines.append("            }")
        if output_products:
            lines += ["#pragma unroll", f"            for(int tile=0;tile<{ncolumns};++tile)",
                      "#pragma unroll", f"            for(int k=0;k<{d3};++k)",
                      f"            output_partial[group*{partial_stride}+(tile*8+lane)*{max_components}+k]=output_accum[tile*{d3}+k];",
                      "            __syncthreads();", f"            for(int i=threadIdx.x;i<min({column_tile},{channels}-column_base)*{d3};i+={threads}) {{",
                      f"                const int column=i/{d3},k=i%{d3};float value=0.f;",
                      "#pragma unroll", f"                for(int p=0;p<{groups};++p)value+=output_partial[p*{partial_stride}+column*{max_components}+k];",
                      f"                atomicAdd(gg+int64_t(dest_node)*{meta['out_dim']}+{oo}+(column_base+column)*{d3}+k,value);", "            }", "            __syncthreads();"]
        lines += ["        }", "    }"]
        lines.append("    __syncthreads();")
        if y_outputs:
            lines.append(f"    const unsigned valid_path_mask=__ballot_sync(physical_warp_mask,threadIdx.x<{path_tile} && begin+threadIdx.x<{paths});")
        lines += [f"    if(threadIdx.x<{path_tile} && begin+threadIdx.x<{paths}) {{",
                  f"        const int row=threadIdx.x;const int64_t p=begin+row,u=p/{edges},v=p%{edges};"]
        for axis, targets in (("x", x_outputs), ("y", y_outputs)):
            for target, pairs in targets.items():
                width = d1 if axis == "x" else d2
                lines += ["        {", f"            float result[{width}]={{}};"]
                for pair in pairs:
                    name, zname = pair[:2]
                    zi = list(contractions).index(zname)
                    factor = f"*__ldg({pair[2]}+item)" if len(pair) == 3 and pair[2] is not None else ""
                    for i, j, k, coeff in spec["terms"]:
                        addr = f"{name}+int64_t(item)*{meta['in2_dim']}+{yo}+v*{d2}+{j}" if axis == "x" else f"{name}+int64_t(source_node)*{meta['in1_dim']}+{xo}+u*{d1}+{i}"
                        component = i if axis == "x" else j
                        lines.append(f"            result[{component}]=fmaf({float_literal(coeff,meta.get('dtype'))}*__ldg({addr}){factor},z[row*{z_stride}+{zi*d3+k}],result[{component}]);")
                for c in sorted({term[0 if axis == "x" else 1] for term in spec["terms"]}):
                    address = f"{target}+int64_t(source_node)*{meta['in1_dim']}+{xo}+u*{d1}+{c}" if axis == "x" else f"edge_{target}+v*{d2}+{c}"
                    if axis == "x" and node_centered:
                        lines.append(f"            node_{target}[{c}]+=result[{c}];")
                    elif axis == "y":
                        lines += ["            {", f"                float partial=result[{c}];", "                const unsigned mask=valid_path_mask;",
                                  "#pragma unroll", f"                for(int delta={edges};delta<32;delta*=2) {{",
                                  "                    const float other=__shfl_down_sync(mask,partial,delta);",
                                  f"                    if(warp_lane+delta<32 && row+delta<{path_tile} && begin+row+delta<{paths})partial+=other;", "                }",
                                  f"                if(warp_lane<{edges})atomicAdd({address},partial);", "            }"]
                    else:
                        lines.append(f"            atomicAdd({address},result[{c}]);")
                lines.append("        }")
        for target, pairs in e_outputs.items():
            lines += ["        {", "            float value=0.f;"]
            for q, zname in pairs:
                zi = list(contractions).index(zname)
                lines += ["            {", *[f"                float z{k}=0.f;" for k in range(d3)],
                          textwrap.indent(emit_cg_source(meta,spec,pairs=program["cg"][q],x_row="source_node"), '                ')]
                lines += [f"                value=fmaf(z{k},z[row*{z_stride}+{zi*d3+k}],value);" for k in range(d3)]
                lines.append("            }")
            lines += [f"            atomicAdd(&scalar_{target},value);", "        }"]
        lines += ["    }", "    __syncthreads();"]
        for target in y_outputs:
            lines.append(f"    for(int i=threadIdx.x;i<{edges*d2};i+={threads})atomicAdd({target}+int64_t(item)*{meta['in2_dim']}+{yo}+i,edge_{target}[i]);")
        for target in e_outputs:
            lines.append(f"    if(threadIdx.x==0)atomicAdd({target}+item,scalar_{target});")
        if node_centered:
            lines += ["    __syncthreads();", "}", f"if(threadIdx.x<{path_tile} && begin+threadIdx.x<{paths}) {{", f"    const int64_t u=(begin+threadIdx.x)/{edges};"]
            for target in x_outputs:
                for c in sorted({term[0] for term in spec["terms"]}):
                    lines.append(f"    atomicAdd({target}+int64_t(source_node)*{meta['in1_dim']}+{xo}+u*{d1}+{c},node_{target}[{c}]);")
            lines.append("}")
        lines += ["    return;", "}"]
        cases.append("\n".join(lines))
        tasks += count
    parameters = [f"const float* __restrict__ {name}" for name in inputs]
    parameters += ["const int32_t* graph_src", "const int32_t* graph_dst"]
    if node_centered:
        parameters += ["const EdgeIndex* src_rowptr", "const EdgeIndex* edge_by_src"]
    parameters += [f"float* __restrict__ {name}" for name in outputs]
    task_decode = (
        f"const int64_t task=blockIdx.x,source_node=task/{tasks*BACKWARD_NEIGHBOR_SPLITS},partition=task/{tasks}%{BACKWARD_NEIGHBOR_SPLITS},local=task%{tasks};\n"
        "if(src_rowptr[source_node]+partition>=src_rowptr[source_node+1]) return;"
        if node_centered else
        f"const int64_t task=blockIdx.x,item=task/{tasks},local=task%{tasks};\n"
        "const int source_node=graph_src[item],dest_node=graph_dst[item];"
    )
    scratch = [f"__shared__ float z[{path_tile*z_stride}];"]
    scratch.extend(f"__shared__ float edge_{name}[{max(s['edge_channels']*s['d2'] for s in meta['generated_specs'])}];" for name in y_outputs)
    scratch.extend(f"__shared__ float scalar_{name};" for name in e_outputs)
    if queries:
        scratch.append(f"__shared__ float query_values[{len(queries)*path_tile*max_components}];")
    if output_products:
        scratch.append(f"__shared__ float output_partial[{groups*partial_stride}];")
    source = f"""
__global__ __launch_bounds__({threads}) void {kernel_name}({','.join(parameters)}) {{
{textwrap.indent(task_decode, '    ')}
    const int lane = threadIdx.x % 8, group = threadIdx.x / 8, warp_lane = threadIdx.x % 32;
    const unsigned physical_warp_mask = 0x{(1 << min(threads,32))-1:08x}u;
{textwrap.indent(''.join(scratch), '    ')}
{textwrap.indent(chr(10).join(cases), '    ')}
}}
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32)), tasks, threads


def make_uvw_conv_weight_gradient_kernel(
    meta, program, inputs, outputs, kernel_name, *, static_columns=False,
):
    """Write the actual path/column cells after cooperative CG and gradient gathers."""
    path_tile, column_tile, threads = 64, 64, 128
    max_components = max(spec["d3"] for spec in meta["generated_specs"])
    queries = tuple(dict.fromkeys(q for target in outputs for q, _ in program["w"][target]))
    gradients = tuple(dict.fromkeys(g for target in outputs for _, g in program["w"][target]))
    scalar_bytes = 8 if meta.get("dtype") == torch.float64 else 4
    while max_components * (len(queries)*path_tile + len(gradients)*column_tile) * scalar_bytes > 47*1024:
        if column_tile > 1:
            column_tile //= 2
        elif path_tile > 1:
            path_tile //= 2
        else:
            raise ValueError("Weight-gradient tile exceeds shared memory capacity")
    cases, tasks = [], 0
    for spec in meta["generated_specs"]:
        d3, channels, paths = (spec[k] for k in ("d3", "out_channels", "path_channels"))
        columns = (channels + column_tile - 1) // column_tile
        count = ((paths + path_tile - 1) // path_tile) * columns
        lines = [f"if (local < {tasks + count}) {{",
                 f"    const int64_t begin = (local - {tasks}) / {columns} * {path_tile};",
                 f"    const int column = (local - {tasks}) % {columns} * {column_tile};",
                 f"    const int rows = int(min(int64_t({path_tile}), int64_t({paths}) - begin));",
                 f"    const int columns = min({column_tile}, {channels} - column);"]
        for qi, q in enumerate(queries):
            lines += [f"    if (threadIdx.x < {path_tile} && begin + threadIdx.x < {paths}) {{",
                      f"        const int64_t p = begin + threadIdx.x, u = p / {spec['edge_channels']}, v = p % {spec['edge_channels']};"]
            lines += [f"        float z{k} = 0.f;" for k in range(d3)]
            lines.append(textwrap.indent(emit_cg_source(meta, spec, pairs=program["cg"][q], x_row="source_node"), '        '))
            lines += [f"        q[{qi}][threadIdx.x][{k}] = z{k};" for k in range(d3)]
            lines.append("    }")
        for gi, grad in enumerate(gradients):
            lines += [f"    for (int i = threadIdx.x; i < {column_tile * d3}; i += {threads}) {{",
                      f"        const int c = i / {d3}, k = i % {d3};",
                      f"        if (column + c < {channels}) grad[{gi}][c][k] = __ldg({grad} + int64_t(dest_node) * {meta['out_dim']} + {spec['out_offset']} + (column + c) * {d3} + k);",
                      "    }"]
        lines.append("    __syncthreads();")
        worker = ["const int column_shift=min(5,32-__clz(unsigned(columns-1)));",
                  "const int column_width=1<<column_shift;",
                  "const int c=threadIdx.x&(column_width-1);",
                  f"const int row_stride={threads}>>column_shift;"]
        column_parts = (column_tile + 31) // 32
        for gi, g in enumerate(gradients):
            for part in range(column_parts):
                for k in range(d3):
                    worker.append(f"const float cached_{g}_{part}_{k}=c+{part}*column_width<columns?grad[{gi}][c+{part}*column_width][{k}]:0.f;")
        worker += [f"for(int row=threadIdx.x>>column_shift;row<rows;row+=row_stride) {{"]
        for qi, query in enumerate(queries):
            for k in range(d3):
                worker.append(f"    const float row_{query}_{k}=q[{qi}][row][{k}];")
        for part in range(column_parts):
            worker += [f"    if(c+{part}*column_width<columns) {{"]
            for target in outputs:
                worker.append(f"        float value_{target}=0.f;")
                for query, grad in program["w"][target]:
                    for k in range(d3):
                        worker.append(f"        value_{target}=fmaf(row_{query}_{k},cached_{grad}_{part}_{k},value_{target});")
                worker.append(f"        {target}[int64_t(item)*{meta['weight_numel']}+{spec['weight_offset']}+(begin+row)*{channels}+column+c+{part}*column_width]=value_{target};")
            worker.append("    }")
        worker.append("}")
        if static_columns:
            lines += ["    auto store_columns = [&](auto column_count) {",
                      "        constexpr int columns = decltype(column_count)::value;"]
            lines += worker
            lines.append("    };")
            tail = channels % column_tile
            if channels <= column_tile:
                lines.append(f"    store_columns(std::integral_constant<int,{channels}>{{}});")
            elif tail:
                lines += [f"    if (column < {channels - tail}) store_columns(std::integral_constant<int,{column_tile}>{{}});",
                          f"    else store_columns(std::integral_constant<int,{tail}>{{}});"]
            else:
                lines.append(f"    store_columns(std::integral_constant<int,{column_tile}>{{}});")
        else:
            lines += worker
        lines += ["    return;", "}"]
        cases.append("\n".join(lines))
        tasks += count
    parameters = [f"const float* {name}" for name in inputs]
    parameters += ["const int32_t* graph_src", "const int32_t* graph_dst"]
    parameters += [f"float* {name}" for name in outputs]
    source = f"""
__global__ __launch_bounds__({threads})
    void {kernel_name}({','.join(parameters)}) {{
    const int64_t task = blockIdx.x, item = task / {tasks}, local = task % {tasks};
    const int source_node = graph_src[item], dest_node = graph_dst[item];
    __shared__ float q[{len(queries)}][{path_tile}][{max_components}];
    __shared__ float grad[{len(gradients)}][{column_tile}][{max_components}];
{textwrap.indent(chr(10).join(cases), '    ')}
}}
"""
    if static_columns:
        source = "#include <type_traits>\n" + source
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32)), tasks, threads


def make_external_backward_source(meta, program, inputs, outputs, phase, *, env=False):
    """Launch fused T/R paths and separate native gradient branches."""
    requested_outputs = outputs
    outputs = tuple(name for name in outputs
                    if (name == "gg" and program["forward"])
                    or any(name in program.get(axis, {}) for axis in ("x", "y", "w", "e")))
    prefix = f"lunatp_uvwconv_{phase}_external"
    # Separate dense weight-gradient kernels own specs outside the T/R paths.
    matrix_outputs = tuple(name for name in outputs if name != "gg")
    input_outputs=tuple(k for k in matrix_outputs if k not in program["w"])
    fused_output="gg" in outputs and bool(input_outputs)
    native_outputs = tuple(name for name in outputs
                           if name not in program["w"] and (name != "gg" or fused_output))
    weight_outputs = tuple(name for name in outputs if name in program["w"])
    native_first = phase == "bwd" and not env and meta.get("dtype", torch.float32) == torch.float32
    kernels, branches = [], []
    if native_first:
        phases = (input_outputs, weight_outputs)
        t_specs = select_uvw_conv_t_specs(meta, outputs)
        if t_specs:
            phases = (phases[0] + phases[1], phases[1])
        for index, phase_outputs in enumerate(phases):
            if not phase_outputs:
                continue
            phase_meta = meta
            if t_specs and index == 1:
                phase_meta = {**meta, "generated_specs": tuple(
                    spec for spec in meta["generated_specs"] if spec not in t_specs
                )}
                if not phase_meta["generated_specs"]:
                    continue
            name = f"{prefix}_fused_{index}_fp32"
            if index == 0:
                kernel, tasks, threads = make_uvw_conv_input_gradient_kernel(
                    phase_meta, program, inputs, phase_outputs, name
                )
            else:
                name = f"{prefix}_weight_fp32"
                kernel, tasks, threads = make_uvw_conv_weight_gradient_kernel(
                    phase_meta, program, inputs, phase_outputs, name, static_columns=True
                )
            kernels.append(kernel)
            args = ",".join(f"{name}.data_ptr<float>()" for name in inputs)
            args += ",src.data_ptr<int32_t>(),dst.data_ptr<int32_t>()," + ",".join(
                f"{name}.data_ptr<float>()" for name in phase_outputs
            )
            branch_stream = "stream" if not branches else f"state.streams[{len(branches) - 1}]"
            branches.append([f"{name}<<<items*{tasks},{threads},0,{branch_stream}>>>({args});"])
    else:
        groups = {}
        for spec in meta["generated_specs"]:
            layout = get_uvw_conv_tr_layout(meta, program, outputs, spec)
            key = ("tr", spec["d1"], spec["d3"]) if layout is not None else ("z", spec["d3"])
            groups.setdefault(key, []).append(spec)
        main_branch, weight_groups = [], {}
        for key, specs in groups.items():
            tr_axis = key[0] == "tr"
            phase_outputs = outputs if tr_axis else native_outputs
            group_meta = {**meta, "generated_specs": specs}
            if phase_outputs:
                suffix = "_".join(str(part) for part in key)
                name = f"{prefix}_fused_{suffix}_fp32"
                generate = make_uvw_conv_tr_kernel if tr_axis else make_uvw_conv_backward_external_kernel
                kernel, tasks, threads = generate(group_meta, program, inputs, phase_outputs, name)
                kernels.append(kernel)
                node_centered = not tr_axis and any(name in program["x"] for name in phase_outputs)
                args = ",".join(f"{name}.data_ptr<float>()" for name in inputs)
                args += ",src.data_ptr<int32_t>(),dst.data_ptr<int32_t>()"
                if node_centered:
                    args += ",src_rowptr.data_ptr<EdgeIndex>(),edge_by_src.data_ptr<EdgeIndex>()"
                args += "," + ",".join(f"{name}.data_ptr<float>()" for name in phase_outputs)
                grid = f"nodes*{BACKWARD_NEIGHBOR_SPLITS}" if node_centered else "items"
                branch_stream = "stream" if not main_branch else f"state.streams[{len(main_branch) - 1}]"
                main_branch.append(f"{name}<<<{grid}*{tasks},{threads},0,{branch_stream}>>>({args});")
            if not tr_axis and weight_outputs:
                weight_groups.setdefault(key[1], []).extend(specs)
        branches.extend([launch] for launch in main_branch)
        weight_branch = []
        branch_stream = "stream" if not branches else f"state.streams[{len(branches) - 1}]"
        for components, specs in weight_groups.items():
            name = f"{prefix}_weight_d{components}_fp32"
            group_meta = {**meta, "generated_specs": specs}
            kernel, tasks, threads = make_uvw_conv_weight_gradient_kernel(
                group_meta, program, inputs, weight_outputs, name
            )
            kernels.append(kernel)
            args = ",".join(f"{name}.data_ptr<float>()" for name in inputs)
            args += ",src.data_ptr<int32_t>(),dst.data_ptr<int32_t>()"
            args += "," + ",".join(f"{name}.data_ptr<float>()" for name in weight_outputs)
            weight_branch.append(f"{name}<<<items*{tasks},{threads},0,{branch_stream}>>>({args});")
        if weight_branch:
            branches.append(weight_branch)
    if "gg" in outputs and not fused_output:
        products = tuple((program["cg"][q], w) for q, w in program["forward"])
        name = f"{prefix}_g_fp32"
        kernel, grid, clear, pointer_order, forward_threads = make_uvw_conv_forward_external_kernel(
            meta, products, name
        )
        kernels.insert(0, kernel)
        output_branch = []
        branch_stream = "stream" if not branches else f"state.streams[{len(branches) - 1}]"
        if clear:
            output_branch.append(
                f"C10_CUDA_CHECK(cudaMemsetAsync(gg.data_ptr<float>(),0,gg.numel()*sizeof(float),{branch_stream}));"
            )
        args = ",".join(f"{name}.data_ptr<float>()" for name in pointer_order)
        output_branch.append(
            f"{name}<<<{grid},{forward_threads},0,{branch_stream}>>>({args},src.data_ptr<int32_t>(),edge_by_dst.data_ptr<EdgeIndex>(),dst_rowptr.data_ptr<EdgeIndex>(),gg.data_ptr<float>(),nodes);"
        )
        branches.append(output_branch)
    launches = [line for branch in branches for line in branch]
    stream_state, state_parameter, stream_binding = "", "", ""
    # Contributions use atomic sums or disjoint weight slices. Fork after
    # zeroing accumulators, and join before returning to the caller's stream.
    if len(branches) > 1:
        count = len(branches) - 1
        stream_state = f"""
struct BranchStreams {{
    BranchStreams(int device, const std::vector<int64_t> &handles) {{
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        for (auto handle : handles)
            streams.push_back(at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handle), device));
        for (auto &event : done)
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }}
    ~BranchStreams() noexcept {{
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess)
            return;
        const int device = streams.front().device_index();
        if (previous != device && cudaSetDevice(device) != cudaSuccess)
            return;
        cudaEventDestroy(start);
        for (auto event : done)
            cudaEventDestroy(event);
        if (previous != device)
            cudaSetDevice(previous);
    }}
    BranchStreams(const BranchStreams &) = delete;
    BranchStreams &operator=(const BranchStreams &) = delete;
    std::vector<at::cuda::CUDAStream> streams;
    cudaEvent_t start = nullptr;
    cudaEvent_t done[{count}] = {{}};
}};
"""
        state_parameter = "BranchStreams& state,"
        stream_binding = f"""
    pybind11::class_<BranchStreams>(m, "_Streams", pybind11::module_local())
        .def(pybind11::init<int, const std::vector<int64_t> &>());
    m.attr("_Streams").attr("stream_count") = {count};
    m.attr("_Streams").attr("stream_priority") = 0;
"""
        launches = ["C10_CUDA_CHECK(cudaEventRecord(state.start, stream));"] + branches[0]
        for index, branch in enumerate(branches[1:]):
            launches += [f"C10_CUDA_CHECK(cudaStreamWaitEvent(state.streams[{index}], state.start, 0));"]
            launches += branch
            launches += [f"C10_CUDA_CHECK(cudaEventRecord(state.done[{index}], state.streams[{index}]));"]
        launches += [f"C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done[{index}], 0));" for index in range(count)]
    output_shapes = ENV_OUTPUT_SHAPES if env else OUTPUT_SHAPES
    allocations = [
        f"auto {name}=at::empty_like({output_shapes[name]});" for name in requested_outputs
    ]
    clears = [
        f"C10_CUDA_CHECK(cudaMemsetAsync({name}.data_ptr<float>(),0,{name}.numel()*sizeof(float),stream));"
        for name in requested_outputs
        if name not in outputs or name in (*program["x"], *program["y"], *program.get("e", {})) or (name=="gg" and fused_output)
    ]
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
"""
        + "\n".join(kernels) + stream_state
        + r"""
std::vector<at::Tensor> launch({state_parameter}{input_parameters}, at::Tensor src, at::Tensor dst, at::Tensor src_rowptr,
       at::Tensor edge_by_src, at::Tensor dst_rowptr, at::Tensor edge_by_dst) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items = y.size(0), nodes = src_rowptr.numel() - 1;
{allocations}
    if (items == 0) {{
{clear_outputs}
        return {{{outputs}}};
    }}
    auto stream = at::cuda::getCurrentCUDAStream();
{clears}
{launches}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    {stream_binding}m.def("launch", &launch);
}}
""".format(
            state_parameter=state_parameter,
            stream_binding=stream_binding,
            input_parameters=",".join((f"at::Tensor {name}" for name in inputs)),
            allocations=textwrap.indent(chr(10).join(allocations), '    '),
            clear_outputs=textwrap.indent(chr(10).join((f"{name}.zero_();" for name in requested_outputs)), '        '),
            outputs=",".join(requested_outputs),
            clears=textwrap.indent(chr(10).join(clears), '    '),
            launches=textwrap.indent(chr(10).join(launches), '    '),
        ),
        meta.get("dtype", torch.float32),
    )


def without_zero_directions(program, zero_roles):
    """Remove products containing a direction that autograd supplied as None."""
    if not zero_roles:
        return program
    # A literal None inside an ENV product means an identity factor.
    cg = {name: tuple(product for product in products
                      if all(role not in zero_roles for role in product if role is not None))
          for name, products in program["cg"].items()}
    z = {name: tuple(product for product in products
                     if all(role not in zero_roles for role in product))
         for name, products in program["z"].items()}
    filtered = {"cg": {name: products for name, products in cg.items() if products},
                "z": {name: products for name, products in z.items() if products}}
    for axis in ("x", "y"):
        consumers = {}
        for name, products in program[axis].items():
            products = tuple(pair for pair in products
                             if pair[0] not in zero_roles and z.get(pair[1])
                             and all(role not in zero_roles for role in pair[2:] if role is not None))
            if products:
                consumers[name] = products
        filtered[axis] = consumers
    filtered["w"] = {
        name: tuple((query, grad) for query, grad in products if cg.get(query) and grad not in zero_roles)
        for name, products in program["w"].items()
    }
    filtered["w"] = {name: products for name, products in filtered["w"].items() if products}
    if "e" in program:
        filtered["e"] = {
            name: tuple((query, contraction) for query, contraction in products
                        if cg.get(query) and z.get(contraction))
            for name, products in program["e"].items()
        }
        filtered["e"] = {name: products for name, products in filtered["e"].items() if products}
    filtered["forward"] = tuple((query, weight) for query, weight in program["forward"]
                                if cg.get(query) and weight not in zero_roles)
    needed_q = {query for products in filtered["w"].values() for query, _ in products}
    needed_q.update(query for query, _ in filtered["forward"])
    needed_q.update(query for products in filtered.get("e", {}).values() for query, _ in products)
    needed_z = {pair[1] for axis in ("x", "y", "e")
                for products in filtered.get(axis, {}).values() for pair in products}
    filtered["cg"] = {name: products for name, products in filtered["cg"].items() if name in needed_q}
    filtered["z"] = {name: products for name, products in filtered["z"].items() if name in needed_z}
    return {**program, **filtered}


def run_external(op, order, inputs, graph, *, needs=None, env=False):
    """Native graph gather/scatter with common first-, second- and third-order expressions."""
    outputs = (ENV_PROGRAMS if env else PROGRAMS)[order]["outputs"]
    selected = (
        outputs
        if needs is None
        else tuple(name for name, need in zip(outputs, needs) if need)
    )

    roles = (ENV_INPUTS if env else INPUTS)[:len(inputs)]
    zero_roles = tuple(name for name, value in zip(roles, inputs) if value is None)
    key = (order, selected)
    if env:
        key += (True,)
    if zero_roles:
        key += (zero_roles,)
    launch = op.external_kernels.get(key)
    if launch is None:
        meta = op.meta
        program = without_zero_directions((ENV_PROGRAMS if env else PROGRAMS)[order], zero_roles)
        active_roles = tuple(name for name, value in zip(roles, inputs) if value is not None)
        source = make_external_backward_source(
            meta,
            program,
            active_roles,
            selected,
            ("bwd", "double_bwd", "triple_bwd")[order - 1],
            env=env,
        )
        extension = load_cuda_extension(
            f"""lunatp_uvwconv_{("fwd", "bwd", "double_bwd", "triple_bwd")[order]}_external_{op.dtype_tag}""",
            source,
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )
        launch = (extension.launch, getattr(extension, "_Streams", None))
        op.external_kernels[key] = launch

    function, stream_factory = launch
    arguments = tuple(value for value in inputs if value is not None) if zero_roles else inputs
    graph_inputs = (graph.src, graph.dst, graph.src_rowptr, graph.edge_ids_by_src,
                    graph.dst_rowptr, graph.edge_ids_by_dst)
    if stream_factory is None:
        result = function(*arguments, *graph_inputs)
    else:
        state = get_overlap_state(op, inputs[0].device, stream_factory)
        result = function(state, *arguments, *graph_inputs)
    if needs is None:
        return tuple(result)
    values = iter(result)
    return tuple(next(values) if needed else None for needed in needs)


def run(op, x, edge_attr, weight, grad_out, *, graph, needs=None):
    set_arch(op, x.get_device())
    src, dst = graph.src, graph.dst
    num_nodes = graph.num_nodes
    if op.weight_mode == "shared":
        # Frozen edge attributes and weights still require propagation into
        # upstream node features. Reuse the source-CSR input contraction.
        if needs is not None and needs[0] and not any(needs[1:]):
            if op.input_gradient is None:
                extension = load_cuda_extension(
                    f"lunatp_uvwconv_bwd_x_shared_{op.dtype_tag}",
                    make_uvw_conv_backward_dx_shared_source(op.meta),
                    arch=op.arch,
                    use_fast_math=False,
                    index_bits=op.index_bits,
                )
                op.input_gradient = getattr(
                    extension,
                    f"lunatp_uvwconv_bwd_dx_shared_launch_{op.dtype_tag}",
                )
            grad_x = op.input_gradient(
                edge_attr,
                weight,
                grad_out,
                graph.dst,
                graph.src_rowptr,
                graph.edge_ids_by_src,
                num_nodes,
            )
            return grad_x, None, None
        if op.node_contraction is None:
            load_node_contraction(op)
        if needs is not None and not needs[2]:
            grad_x, grad_edge = op.launch_dxdy(
                x, edge_attr, weight, grad_out,
                graph.src, graph.dst, graph.src_rowptr,
                graph.edge_ids_by_src, num_nodes,
            )
            return grad_x, grad_edge, None
        if op.weight_contraction is None:
            load_weight_contraction(op)

        current_stream = torch.cuda.current_stream(x.device)
        weight_stream = auxiliary_streams(
            current_stream.device_index, 1, scope=current_stream.cuda_stream
        )[0]
        # Weight reduction is independent of x/edge gradients. Both streams
        # first see caller inputs, and the caller joins before returning.
        weight_stream.wait_stream(current_stream)
        with torch.cuda.stream(weight_stream):
            q = op.precontract_q(
                x,
                edge_attr,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            grad_weight = op.weight_reduce.launch(q, grad_out)
            del q
        grad_x, grad_edge = op.launch_dxdy(
            x,
            edge_attr,
            weight,
            grad_out,
            graph.src,
            graph.dst,
            graph.src_rowptr,
            graph.edge_ids_by_src,
            num_nodes,
        )
        current_stream.wait_stream(weight_stream)
        grad_weight.record_stream(current_stream)
        return grad_x, grad_edge, grad_weight
    return run_external(op, 1, (x, edge_attr, weight, grad_out), graph, needs=needs)


def load_node_contraction(op):
    extension = load_cuda_extension(
        f"lunatp_uvwconv_bwd_xy_shared_{op.dtype_tag}",
        make_uvw_conv_backward_dxdy_shared_source(op.meta),
        arch=op.arch,
        use_fast_math=False,
        index_bits=op.index_bits,
    )
    op.precontract_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_z_{op.dtype_tag}"
    )
    op.precontract_z_pair = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_z_pair_{op.dtype_tag}"
    )
    op.pair_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_pair_from_z_{op.dtype_tag}"
    )
    op.launch_dxdy = getattr(
        extension, f"lunatp_uvwconv_bwd_dxdy_shared_launch_{op.dtype_tag}"
    )
    op.grad_x_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_x_from_z_{op.dtype_tag}"
    )
    op.grad_edge_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_y_from_z_{op.dtype_tag}"
    )
    op.node_contraction = extension


def load_weight_contraction(op):
    extension = load_tp_extension(
        f"lunatp_uvwconv_bwd_weight_precontract_shared_{op.dtype_tag}",
        make_uvw_conv_backward_dw_preprocess_shared_source(op.meta),
        max_channels=max(op.meta["max_in_channels"], op.meta["max_out_channels"]),
        index_bits=op.index_bits,
        arch=op.arch,
    )
    op.weight_reduce = load_cuda_extension(
        f"lunatp_uvwconv_bwd_weight_shared_{op.dtype_tag}",
        make_uvw_conv_weight_cutlass_source(op.meta),
        arch=op.arch,
        use_fast_math=False,
        index_bits=op.index_bits,
    )
    op.precontract_q = getattr(
        extension, f"lunatp_uvwconv_bwd_weight_grouped_q_launch_{op.dtype_tag}"
    )
    op.precontract_q_pair = getattr(
        extension, f"lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_{op.dtype_tag}"
    )
    op.weight_contraction = extension
