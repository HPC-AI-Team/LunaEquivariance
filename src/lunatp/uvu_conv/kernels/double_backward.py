"""UVU convolution double backward CUDA and native launch."""

from __future__ import annotations
from collections import defaultdict
import re
import torch
from ...utils.cg import weight_channel_offset
from ...utils.layout import CUDA_TASK_TABLE_WARP_LIMIT, UvuPathSpec, build_signed_ij_classes, l0_feature_layout
from ....utils.precision import float_literal, cuda_source_to_fp64
from ....utils.extension import load_tp_extension
from ...utils.runtime import get_overlap_state
from .utils import (
    add_optional_arguments,
    flat_coverage,
    group_source_input_paths,
    output_gradient_load,
)


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
        rx, ry, rw = (operands["rx"][i], operands["ry"][j], operands["rw"])
        expressions = {
            "x": f"({ry} * {w} + {y} * {rw}) * raw_l0_seed",
            "y": f"({rx} * {w} + {x} * {rw}) * raw_l0_seed",
            "w": f"({rx} * {y} + {x} * {ry}) * raw_l0_seed",
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


def make_uvu_conv_double_bwd_dgrad_out_kernel(
    op,
    *,
    accumulate=False,
    edge_workers=1,
    index_type="Index",
    with_l0=False,
    with_env=False,
    with_re=True,
):
    edge = "e"
    specs = [dict(spec) for spec in op.generated_specs]
    output_specs = specs
    output_in1_dim = op.in1_dim
    output_in2_dim = op.in2_dim
    output_out_dim = op.out_dim
    output_weight_numel = op.weight_numel
    output_prefer_short_factorization = True
    output_accumulate = accumulate
    output_cooperative_rows = edge_workers > 1
    output_scratch_tile = 16 if edge_workers > 1 else None
    output_index_type = index_type
    output_with_l0 = with_l0
    output_with_env = with_env
    output_with_re = with_re
    output_cast = "" if output_index_type == "int" else output_index_type
    output_cases: list[str] = []
    for output_group_idx, (
        (output_x_offset, output_d1),
        output_raw_group_specs,
    ) in enumerate(group_double_bwd_dgrad_out_paths(output_specs)):
        output_y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for output_path_idx, output_spec in output_raw_group_specs:
            output_y_groups[
                (int(output_spec["y_offset"]), int(output_spec["d2"]))
            ].append((output_path_idx, output_spec))
        output_ordered_y_groups = sorted(
            output_y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        )
        output_group_specs = sorted(
            output_raw_group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
        )
        output_lines = [f"case {output_group_idx}: {{"]
        output_channels = int(output_group_specs[0][1]["channels"])
        if output_channels % 32:
            output_lines.append(f"if (u >= {output_channels}) break;")
        for output_path_idx, output_spec in output_group_specs:
            for output_k_idx in range(int(output_spec["d3"])):
                output_lines.append(
                    f"    float acc_p{output_path_idx}_{output_k_idx} = 0.0f;"
                )
        output_lines.extend(
            [
                "    const Index begin = rowptr[t];",
                "    const Index end = rowptr[t + 1];",
                (
                    f"    for (Index {edge} = begin + hgo_edge_worker; {edge} < end; {edge} += DGRAD_OUT_EDGE_WORKERS) {{"
                    if output_cooperative_rows
                    else f"    for (Index {edge} = begin; {edge} < end; ++{edge}) {{"
                ),
                f"        const int s = __ldg(src + {edge});",
                f"""        const {output_index_type} x_base = {(output_cast + "(s)" if output_cast else "s")} * {output_in1_dim} + {output_x_offset} + u * {output_d1};""",
                f"""        const {output_index_type} y_base = {(output_cast + "(e)" if output_cast else "e")} * {output_in2_dim};""",
                f"""        const {output_index_type} weight_base = {(output_cast + "(e)" if output_cast else "e")} * {output_weight_numel};""",
            ]
        )
        if output_with_env:
            output_lines.append(f"        const float env_e = __ldg(env + e);")
            if output_with_re:
                output_lines.append(
                    "        const float re_e = re ? __ldg(re + e) : 0.0f;"
                )
        for output_i_idx in range(output_d1):
            output_lines.extend(
                f"        const float x{output_i_idx} = __ldg(x + x_base + {output_i_idx});\n        const float vrx{output_i_idx} = __ldg(rx + x_base + {output_i_idx});".split(
                    "\n"
                )
            )
        for (
            output_y_offset,
            output_d2,
        ), output_y_group_specs in output_ordered_y_groups:
            output_lines.append("        {")
            for output_j_idx in range(output_d2):
                output_lines.extend(
                    f"            const float y{output_j_idx} = __ldg(y + y_base + {output_y_offset + output_j_idx});\n            const float vry{output_j_idx} = __ldg(ry + y_base + {output_y_offset + output_j_idx});".split(
                        "\n"
                    )
                )
                if output_with_env:
                    output_lines[-2:] = [
                        f"            const float raw_y{output_j_idx} = __ldg(y + y_base + {output_y_offset + output_j_idx});",
                        f"            const float raw_ry{output_j_idx} = __ldg(ry + y_base + {output_y_offset + output_j_idx});",
                        f"            const float y{output_j_idx} = env_e * raw_y{output_j_idx};",
                        f"            const float vry{output_j_idx} = env_e * raw_ry{output_j_idx}"
                        + (
                            f" + re_e * raw_y{output_j_idx};" if output_with_re else ";"
                        ),
                    ]
            for output_path_idx, output_spec in sorted(
                output_y_group_specs,
                key=lambda path: len(path[1]["terms"]),
                reverse=True,
            ):
                output_weight_offset = int(output_spec["weight_offset"])
                output_terms = list(output_spec["terms"])
                output_unique_i = sorted(
                    {int(output_term[0]) for output_term in output_terms}
                )
                output_unique_j = sorted(
                    {int(output_term[1]) for output_term in output_terms}
                )
                output_factor_on_i = output_prefer_short_factorization and len(
                    output_unique_i
                ) < len(output_unique_j)
                output_lines.extend(
                    f"            {{\n                const float w = __ldg(weight + weight_base + {output_weight_offset} + {weight_channel_offset(output_spec)});\n                const float vrw = __ldg(rw + weight_base + {output_weight_offset} + {weight_channel_offset(output_spec)});".split(
                        "\n"
                    )
                )
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_lines.append("                float scalar_value = 0.0f;")
                output_scalar_first = True
                if output_factor_on_i:
                    for output_i_idx in output_unique_i:
                        output_lines.extend(
                            f"                const float xw{output_i_idx} = x{output_i_idx} * w;\n                const float xd{output_i_idx} = vrx{output_i_idx} * w + x{output_i_idx} * vrw;".split(
                                "\n"
                            )
                        )
                else:
                    for output_j_idx in output_unique_j:
                        output_lines.extend(
                            f"                const float yw{output_j_idx} = y{output_j_idx} * w;\n                const float yd{output_j_idx} = vry{output_j_idx} * w + y{output_j_idx} * vrw;".split(
                                "\n"
                            )
                        )
                for output_class_idx, output_signature_class in enumerate(
                    build_signed_ij_classes(
                        output_terms,
                        spec_y_offset=output_y_offset,
                        y_name_base=output_y_offset,
                    )
                ):
                    output_representative = output_signature_class.members[0]

                    output_lines.append("                {")
                    if output_factor_on_i:
                        output_dual_product_value = f"xd{output_representative.i_idx} * y{output_representative.j_name} + xw{output_representative.i_idx} * vry{output_representative.j_name}"
                    else:
                        output_dual_product_value = f"vrx{output_representative.i_idx} * yw{output_representative.j_name} + x{output_representative.i_idx} * yd{output_representative.j_name}"
                    output_lines.append(
                        f"                    float p{output_class_idx} = {output_dual_product_value};"
                    )
                    for output_member in output_signature_class.members[1:]:
                        output_op = "+=" if output_member.relative_sign > 0 else "-="
                        if output_factor_on_i:
                            output_dual_product_value = f"xd{output_member.i_idx} * y{output_member.j_name} + xw{output_member.i_idx} * vry{output_member.j_name}"
                        else:
                            output_dual_product_value = f"vrx{output_member.i_idx} * yw{output_member.j_name} + x{output_member.i_idx} * yd{output_member.j_name}"
                        output_lines.append(
                            f"                    p{output_class_idx} {output_op} {output_dual_product_value};"
                        )
                    for (
                        output__,
                        output__,
                        output_k_idx,
                        output_coeff,
                    ) in output_representative.terms:
                        output_lines.append(
                            f"                    acc_p{output_path_idx}_{int(output_k_idx)} += {float_literal(float(output_coeff))} * p{output_class_idx};"
                        )
                        if output_with_l0 and int(output_spec["d3"]) == 1:
                            output_scalar_op = "=" if output_scalar_first else "+="
                            output_lines.append(
                                f"                scalar_value {output_scalar_op} {float_literal(float(output_coeff))} * p{output_class_idx};"
                            )
                            output_scalar_first = False
                    output_lines.append("                }")
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_offsets, output_width = l0_feature_layout(output_specs)
                    if output_with_env:
                        output_lines.append("                scalar_value = 0.0f;")
                        for output_i, output_j, output__k, output_coeff in output_spec[
                            "terms"
                        ]:
                            output_lines.append(
                                f"                scalar_value += {float_literal(float(output_coeff))} * (vrx{output_i} * raw_y{output_j} * w + x{output_i} * raw_ry{output_j} * w + x{output_i} * raw_y{output_j} * vrw);"
                            )
                    if "l0_path" in output_spec:
                        output_pointer = f"hl0 + Index(e) * {output_width} + {output_offsets[output_path_idx]} + u"
                        output_lines.append(
                            "                if (hl0) "
                            + f"atomicAdd({output_pointer}, scalar_value);"
                        )
                    else:
                        output_lines.append(
                            f"                if (hl0) hl0[Index(e) * {output_width} + {output_offsets[output_path_idx]} + u] = scalar_value;"
                        )
                output_lines.append("            }")
            output_lines.append("        }")
        output_lines.append("    }")
        if output_cooperative_rows and output_scratch_tile is not None:
            output_outputs = [
                (output_path_idx, output_k_idx)
                for output_path_idx, output_spec in output_group_specs
                for output_k_idx in range(int(output_spec["d3"]))
            ]
            output_lines.append("    if constexpr (DGRAD_OUT_EDGE_WORKERS > 1) {")
            for output_tile_start in range(0, len(output_outputs), output_scratch_tile):
                output_tile_outputs = output_outputs[
                    output_tile_start : output_tile_start + output_scratch_tile
                ]
                for output_output_index, (output_path_idx, output_k_idx) in enumerate(
                    output_tile_outputs
                ):
                    output_lines.append(
                        f"        hgo_scratch[(hgo_edge_worker * {output_scratch_tile} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread] = acc_p{output_path_idx}_{output_k_idx};"
                    )
                output_lines.extend(r"""        __syncthreads();
        if (hgo_edge_worker == 0) {{
            #pragma unroll
            for (int worker = 1; worker < DGRAD_OUT_EDGE_WORKERS; ++worker) {{""".format().split("\n"))
                for output_output_index, (output_path_idx, output_k_idx) in enumerate(
                    output_tile_outputs
                ):
                    output_lines.append(
                        f"                acc_p{output_path_idx}_{output_k_idx} += hgo_scratch[(worker * {output_scratch_tile} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread];"
                    )
                output_lines.extend(
                    "            }\n        }\n        __syncthreads();".split("\n")
                )
            output_lines.extend(["    }", "    if (hgo_edge_worker == 0) {"])
        elif output_cooperative_rows:
            output_output_width = sum(
                int(output_spec["d3"]) for output__, output_spec in output_group_specs
            )
            output_lines.append("    if constexpr (DGRAD_OUT_EDGE_WORKERS > 1) {")
            output_output_index = 0
            for output_path_idx, output_spec in output_group_specs:
                for output_k_idx in range(int(output_spec["d3"])):
                    output_lines.append(
                        f"        hgo_scratch[(hgo_edge_worker * {output_output_width} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread] = acc_p{output_path_idx}_{output_k_idx};"
                    )
                    output_output_index += 1
            output_lines.extend(r"""        __syncthreads();
        if (hgo_edge_worker == 0) {{
            #pragma unroll
            for (int worker = 1; worker < DGRAD_OUT_EDGE_WORKERS; ++worker) {{""".format().split("\n"))
            output_output_index = 0
            for output_path_idx, output_spec in output_group_specs:
                for output_k_idx in range(int(output_spec["d3"])):
                    output_lines.append(
                        f"                acc_p{output_path_idx}_{output_k_idx} += hgo_scratch[(worker * {output_output_width} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread];"
                    )
                    output_output_index += 1
            output_lines.extend(
                [
                    "            }",
                    "        }",
                    "    }",
                    "    if (hgo_edge_worker == 0) {",
                ]
            )
        output_lines.append(
            f"""    const {output_index_type} hgo_base = {(output_cast + "(t)" if output_cast else "t")} * {output_out_dim};"""
        )
        for output_path_idx, output_spec in output_group_specs:
            output_out_offset = int(output_spec["out_offset"])
            output_d3 = int(output_spec["d3"])
            for output_k_idx in range(output_d3):
                output_address = (
                    f"hgo_base + {output_out_offset} + u * {output_d3} + {output_k_idx}"
                )
                output_value = f"acc_p{output_path_idx}_{output_k_idx}"
                if output_accumulate:
                    output_lines.append(
                        f"    atomicAdd(hgo + {output_address}, {output_value});"
                    )
                else:
                    output_lines.append(f"    hgo[{output_address}] = {output_value};")
        if output_cooperative_rows:
            output_lines.append("    }")
        output_lines.extend(["    break;", "}"])
        output_cases.append("\n".join(output_lines))
    branches, groups = "\n".join(output_cases), len(output_cases)
    task_widths = [
        (int(paths[0][1]["channels"]) + 31) // 32
        for _, paths in group_double_bwd_dgrad_out_paths(specs)
    ]
    task_count = sum(task_widths)
    if task_count <= CUDA_TASK_TABLE_WARP_LIMIT:
        tasks = [
            (group, channel * 32)
            for group, width in enumerate(task_widths)
            for channel in range(width)
        ]
        task_groups = ",".join(str(group) for group, _ in tasks)
        task_channels = ",".join(str(channel) for _, channel in tasks)
        task_tables = f"__device__ __constant__ int DOUBLE_BWD_DGRAD_OUT_TASK_GROUPS[] = {{{task_groups}}};\n__device__ __constant__ int DOUBLE_BWD_DGRAD_OUT_TASK_CHANNELS[] = {{{task_channels}}};"
        group_setup = "    const int group = DOUBLE_BWD_DGRAD_OUT_TASK_GROUPS[task];"
    else:
        task_tables = ""
        decode = ["    int group;", "    int channel_base;"]
        base = 0
        for group, width in enumerate(task_widths):
            if group == len(task_widths) - 1:
                branch = "    else {" if group else "    {"
            else:
                condition = "if" if group == 0 else "else if"
                branch = f"    {condition} (task < {base + width}) {{"
            decode += [
                branch,
                f"        group = {group};",
                f"        channel_base = (task - {base}) * 32;",
                "    }",
            ]
            base += width
        group_setup = "\n".join(decode)
    channel_base = (
        "channel_base" if task_count > CUDA_TASK_TABLE_WARP_LIMIT
        else "DOUBLE_BWD_DGRAD_OUT_TASK_CHANNELS[task]"
    )
    thread_setup = f"    const int u = {channel_base} + threadIdx.x;"
    if edge_workers > 1:
        thread_setup = r"""    constexpr int DGRAD_OUT_CHANNEL_THREADS = 32;
    constexpr int DGRAD_OUT_EDGE_WORKERS = {edge_workers};
    const int hgo_channel_thread = threadIdx.x % DGRAD_OUT_CHANNEL_THREADS;
    const int hgo_edge_worker = threadIdx.x / DGRAD_OUT_CHANNEL_THREADS;
    __shared__ float hgo_scratch[{scratch_words}];
    const int u = {channel_base} + hgo_channel_thread;""".format(
            edge_workers=edge_workers,
            channel_base=channel_base,
            scratch_words=edge_workers * 32 * 16,
        )
    kernel = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
namespace {{
{task_tables}
__global__ void lunatp_uvuconv_double_bwd_dgrad_out_dst_tasks_fp32(
    const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ weight, const float* __restrict__ rx,
    const float* __restrict__ ry, const float* __restrict__ rw,
    const int32_t* __restrict__ src, const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    float* __restrict__ hgo, int num_nodes) {{
    const int task = blockIdx.x / num_nodes;
{group_setup}
    const int rank = blockIdx.x - Index(task) * num_nodes;
    const int t = node_order ? node_order[rank] : rank;
{thread_setup}
    switch (group) {{
{branches}
    default: break;
    }}
}}
}}
""".format(
        task_tables=task_tables,
        group_setup=group_setup,
        thread_setup=thread_setup,
        branches=branches,
    )
    return kernel, task_count


def make_uvu_conv_double_bwd_dgrad_out_source(
    op, *, with_l0=False, with_env=False, with_re=True
):
    fp64 = op.compute_dtype == torch.float64
    accumulate = not fp64 or op.has_forward_output_collision
    clear = (
        "    C10_CUDA_CHECK(cudaMemsetAsync(hgo.data_ptr<float>(), 0,\n"
        "                                  hgo.numel() * sizeof(float), stream));\n"
        if not fp64 or op.forward_output_needs_zero
        else ""
    )
    edge_workers = 1 if any(int(s["channels"]) % 32 for s in op.generated_specs) else 2
    kernel, tasks = make_uvu_conv_double_bwd_dgrad_out_kernel(
        op,
        accumulate=accumulate,
        edge_workers=edge_workers,
        index_type="Index",
        with_l0=with_l0,
        with_env=with_env,
        with_re=with_re,
    )
    if with_l0 and any("l0_path" in spec for spec in op.generated_specs):
        clear += "    C10_CUDA_CHECK(cudaMemsetAsync(hl0->data_ptr<float>(), 0, hl0->numel() * sizeof(float), stream));\n"
    return add_optional_arguments(
        kernel
        + r"""
void run_uvuconv_double_bwd_dgrad_out(torch::Tensor x, torch::Tensor y, torch::Tensor weight,
         torch::Tensor rx, torch::Tensor ry, torch::Tensor rw,
         torch::Tensor src, torch::Tensor rowptr, torch::Tensor order,
         torch::Tensor hgo) {{
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
{clear}    dim3 grid(static_cast<unsigned int>(x.size(0) * {tasks}));
    lunatp_uvuconv_double_bwd_dgrad_out_dst_tasks_fp32<<<grid, {threads}, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), rw.data_ptr<float>(),
        src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(),
        order.numel() ? order.data_ptr<int32_t>() : nullptr,
        hgo.data_ptr<float>(), static_cast<int>(x.size(0)));
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{ module.def("run_uvuconv_double_bwd_dgrad_out", &run_uvuconv_double_bwd_dgrad_out); }}
""".format(
            clear=clear,
            tasks=tasks,
            threads=32 * edge_workers,
        ),
        "run_uvuconv_double_bwd_dgrad_out",
        {
            **({"hl0": "float* __restrict__"} if with_l0 else {}),
            **(
                {"env": "const float* __restrict__", "re": "const float* __restrict__"}
                if with_env
                else {}
            ),
        },
        ("hl0", "env", "re"),
    )


def make_uvu_conv_double_bwd_source(
    op, *, with_l0=False, with_env=False, need_grad_env=True
):
    """Assemble the complete second-order derivative from source-owner and output contractions.

    Index follows the compiled address width. Accumulation precision is
    selected independently by the FP32/FP64 source conversion.
    """
    need_grad_env = with_env and need_grad_env
    edge = "e"
    specs = op.generated_specs
    dims = dict(
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        out_dim=op.out_dim,
        weight_numel=op.weight_numel,
        index_type="Index",
    )
    groups = group_source_input_paths(specs)
    widths = [int(paths[0][1]["channels"]) for _, paths in groups]
    if any(c <= 0 for c in widths):
        raise ValueError("second-order derivatives require positive channels")
    scalar_bytes = 8 if op.compute_dtype == torch.float64 else 4
    components = sum(d1 for (_, d1), _ in groups)
    aligned = all(c % 32 == 0 for c in widths)
    padded = [((c + 31) // 32) * 32 for c in widths]
    native_threads = max(
        t for t in range(32, min(padded[0], 128) + 1, 32) if padded[0] % t == 0
    )
    # Bucket groups only when one edge worker fills the block, or state is small.
    equal_width = len(set(widths)) == 1 and (
        components <= 25 or 128 // native_threads == 1
    )
    segmented = components > (25 if equal_width else 32)
    fused = equal_width and not segmented and components * scalar_bytes <= 96
    max_threads = 64 if scalar_bytes == 8 and not equal_width and not segmented else 128
    threads = [
        max(t for t in range(32, min(c, max_threads) + 1, 32) if c % t == 0)
        for c in padded
    ]
    bundles = (
        [tuple(range(len(groups)))] if fused else [(g,) for g in range(len(groups))]
    )
    buckets = [[b] for b in range(len(bundles))]
    if equal_width and not fused:
        buckets = [[], []]
        costs = [0, 0]
        for i, (_, paths) in enumerate(groups):
            target = (
                min(range(2), key=costs.__getitem__)
                if segmented
                else int(i >= (len(groups) + 1) // 2)
            )
            buckets[target].append(i)
            costs[target] += sum(len(s["terms"]) for _, s in paths)
        buckets = [b for b in buckets if b]
    bases, partial_count = [], 0
    for bundle in bundles:
        bases.append(partial_count)
        partial_count += (widths[bundle[0]] + 31) // 32
    bodies, launches = [], []
    for bucket_id, bucket in enumerate(buckets):
        cases = []
        for branch, bundle_id in enumerate(bucket):
            bundle = bundles[bundle_id]
            t = threads[bundle[0]]
            node_specs = specs
            node_group_indices = bundle
            node_in1_dim = dims["in1_dim"]
            node_in2_dim = dims["in2_dim"]
            node_out_dim = dims["out_dim"]
            node_weight_numel = dims["weight_numel"]
            node_prefer_short_factorization = (
                widths[0] // t > 1 if fused else not (equal_width and segmented)
            )
            node_lookahead = (2 if widths[0] // t > 1 else 1) if fused else 0
            node_row_index = "static_cast<int64_t>(row)" if segmented else "row"
            node_segmented_rows = segmented
            node_channels = widths[bundle[0]]
            node_index_type = "Index"
            node_with_l0 = with_l0
            node_with_env = with_env
            node_need_grad_env = need_grad_env
            node_need_grad_env = node_with_env and node_need_grad_env
            node_l0_offsets, node_l0_dim = (
                l0_feature_layout(node_specs) if node_with_l0 else ({}, 0)
            )
            node_node_seed_offsets = {} if node_with_env else node_l0_offsets
            node_cast = "" if node_index_type == "int" else node_index_type
            node_tail = node_channels % 32 != 0
            node_all_groups = group_source_input_paths(node_specs)
            node_x_groups = [node_all_groups[node_i] for node_i in node_group_indices]
            node_y_groups: dict[
                tuple[int, int], list[tuple[int, int, dict[str, object]]]
            ] = defaultdict(list)
            for node_group_idx, (node__, node_group_specs) in enumerate(node_x_groups):
                for node_path_idx, node_spec in node_group_specs:
                    node_y_groups[
                        (int(node_spec["y_offset"]), int(node_spec["d2"]))
                    ].append((node_group_idx, node_path_idx, node_spec))
            node_ordered_y_groups = sorted(
                node_y_groups.items(),
                key=lambda group: sum(len(spec["terms"]) for _, _, spec in group[1]),
                reverse=True,
            )

            node_lines = (
                [f"const bool active_channel = u < {node_channels};"]
                if node_tail
                else []
            )
            for node_group_idx, ((node_x_offset, node_d1), node__) in enumerate(
                node_x_groups
            ):
                node_base = [
                    f"""    const {node_index_type} x_base_g{node_group_idx} = {(node_cast + "(s_node)" if node_cast else "s_node")} * {node_in1_dim} + {node_x_offset} + u * {node_d1};"""
                ]
                node_accumulators = [
                    f"    float hx_g{node_group_idx}_{node_i} = 0.0f;"
                    for node_i in range(node_d1)
                ]
                node_operands = [
                    f"    const float x_g{node_group_idx}_{node_i} = __ldg(x + x_base_g{node_group_idx} + {node_i});\n    const float rx_g{node_group_idx}_{node_i} = __ldg(rx + x_base_g{node_group_idx} + {node_i});".split(
                        "\n"
                    )
                    for node_i in range(node_d1)
                ]
                if len(node_group_indices) > 1:
                    node_lines += node_base
                    for node_acc, node_vals in zip(node_accumulators, node_operands):
                        node_lines += [node_acc] + node_vals
                else:
                    node_lines += node_accumulators + node_base
                    for node_vals in node_operands:
                        node_lines += node_vals
            if not node_segmented_rows:
                node_lines += [
                    "    const Index begin = row_begin[s_node];",
                    "    const Index end = row_begin[s_node + 1];",
                ]

            node_lines.extend(
                r"""    for (Index p = begin + edge_worker; p < end; p += EDGE_WORKERS) {{
{edge_index}
        const {index_type} y_base = {edge_row} * {in2_dim};
        const {index_type} weight_base = {edge_row} * {weight_numel};
        const {index_type} go_base = {destination_row} * {out_dim};""".format(
                    edge_index=(
                        "        const Index e = edge_ids_by_src[p];\n"
                        "        const int d = __ldg(dst + e);"
                    ),
                    index_type=node_index_type,
                    edge_row=node_cast + "(e)" if node_cast else "e",
                    in2_dim=node_in2_dim,
                    weight_numel=node_weight_numel,
                    destination_row=node_cast + "(d)" if node_cast else "d",
                    out_dim=node_out_dim,
                ).split("\n")
            )
            if node_with_env:
                node_lines.append(f"        const float env_e = __ldg(env + e);")
                if node_need_grad_env:
                    node_lines.extend(
                        "        const float re_e = re ? __ldg(re + e) : 0.0f;\n        float he_value = 0.0f;".split(
                            "\n"
                        )
                    )
            for (node_y_offset, node_d2), node_y_group_specs in node_ordered_y_groups:
                node_lines.append("        {")
                for node_j_idx in range(node_d2):
                    node_lines.extend(
                        r"""            const float y{j} = __ldg(y + y_base + {y_offset_node_j});
            const float ry{j} = __ldg(ry + y_base + {y_offset_node_j});
            float hy{j} = 0.0f;""".format(
                            j=node_j_idx,
                            y_offset_node_j=node_y_offset + node_j_idx,
                        ).split("\n")
                    )

                node_ordered_paths = (
                    sorted(
                        node_y_group_specs,
                        key=lambda path: len(path[2]["terms"]),
                        reverse=True,
                    )
                    if len(node_group_indices) > 1
                    else node_y_group_specs
                )
                if node_tail:
                    node_lines.append("if (active_channel) {")
                for node_group_idx, node_path_idx, node_spec in node_ordered_paths:
                    node_out_offset = int(node_spec["out_offset"])
                    node_weight_offset = int(node_spec["weight_offset"])
                    node_d3 = int(node_spec["d3"])
                    node_terms = list(node_spec["terms"])
                    node_unique_j = sorted(
                        {int(node_term[1]) for node_term in node_terms}
                    )
                    node_unique_i = sorted(
                        {int(node_term[0]) for node_term in node_terms}
                    )
                    node_factor_on_i = node_prefer_short_factorization and len(
                        node_unique_i
                    ) < len(node_unique_j)
                    node_w_name = "raw_w" if node_with_env else "w"
                    node_rw_name = "raw_rw" if node_with_env else "rw_value"
                    node_lines.extend(
                        f"            {{\n                const float {node_w_name} = __ldg(weight + weight_base + {node_weight_offset} + {weight_channel_offset(node_spec)});\n                const float {node_rw_name} = __ldg(rw + weight_base + {node_weight_offset} + {weight_channel_offset(node_spec)});".split(
                            "\n"
                        )
                    )
                    if node_with_env:
                        # Scale the path weight and its direction; retain the original
                        # contractions without extra edge-component accumulators.
                        node_lines.extend(
                            [
                                f"                const float w = env_e * raw_w;",
                                f"                const float rw_value = env_e * raw_rw"
                                + (" + re_e * raw_w;" if node_need_grad_env else ";"),
                            ]
                        )
                    if node_need_grad_env:
                        node_lines.append("                float gw_value = 0.0f;")
                    node_lines.append("                float hw_value = 0.0f;")
                    if node_factor_on_i:
                        for node_i_idx in node_unique_i:
                            node_lines.extend(
                                r"""                float sy{i} = 0.0f;
                float sry{i} = 0.0f;
                const float xd{i} = rx_g{group}_{i} * w + x_g{group}_{i} * rw_value;""".format(
                                    i=node_i_idx, group=node_group_idx
                                ).split("\n")
                            )
                    else:
                        for node_j_idx in node_unique_j:
                            node_lines.extend(
                                r"""                float sx{j} = 0.0f;
                float srx{j} = 0.0f;
                const float yd{j} = ry{j} * w + y{j} * rw_value;""".format(j=node_j_idx).split(
                                    "\n"
                                )
                            )
                    node_signed_classes = list(
                        build_signed_ij_classes(
                            node_terms,
                            spec_y_offset=node_y_offset,
                            y_name_base=node_y_offset,
                        )
                    )
                    for node_prefetched in range(
                        min(node_lookahead, len(node_signed_classes))
                    ):
                        node_representative = node_signed_classes[
                            node_prefetched
                        ].members[0]
                        node_go_sum = " + ".join(
                            f"""{float_literal(float(node_coeff))} * {output_gradient_load("grad_out", node_out_offset, node_d3, int(node_k_idx), node_path_idx, node_node_seed_offsets, node_l0_dim)}"""
                            for node__, node__, node_k_idx, node_coeff in node_representative.terms
                        )
                        node_lines.append(
                            f"                const float q{node_prefetched} = {node_go_sum};"
                        )
                    for node_class_idx, node_signature_class in enumerate(
                        node_signed_classes
                    ):
                        node_next_class = node_class_idx + node_lookahead
                        if node_next_class < len(node_signed_classes):
                            node_representative = node_signed_classes[
                                node_next_class
                            ].members[0]
                            node_go_sum = " + ".join(
                                f"""{float_literal(float(node_coeff))} * {output_gradient_load("grad_out", node_out_offset, node_d3, int(node_k_idx), node_path_idx, node_node_seed_offsets, node_l0_dim)}"""
                                for node__, node__, node_k_idx, node_coeff in node_representative.terms
                            )
                            node_lines.append(
                                f"                const float q{node_next_class} = {node_go_sum};"
                            )
                        for node_member in node_signature_class.members:
                            node_op = "+=" if node_member.relative_sign > 0 else "-="
                            if node_factor_on_i:
                                node_lines.extend(
                                    r"""                sy{i} {op} y{j} * q{cg_class};
                sry{i} {op} ry{j} * q{cg_class};
                hy{j} {op} xd{i} * q{cg_class};""".format(
                                        i=node_member.i_idx,
                                        op=node_op,
                                        j=node_member.j_name,
                                        cg_class=node_class_idx,
                                    ).split("\n")
                                )
                            else:
                                node_lines.extend(
                                    r"""                hx_g{group}_{i} {op} yd{j} * q{cg_class};
                sx{j} {op} x_g{group}_{i} * q{cg_class};
                srx{j} {op} rx_g{group}_{i} * q{cg_class};""".format(
                                        group=node_group_idx,
                                        i=node_member.i_idx,
                                        op=node_op,
                                        j=node_member.j_name,
                                        cg_class=node_class_idx,
                                    ).split("\n")
                                )
                    if node_factor_on_i:
                        for node_i_idx in node_unique_i:
                            node_lines.extend(
                                f"                hx_g{node_group_idx}_{node_i_idx} += sry{node_i_idx} * w + sy{node_i_idx} * rw_value;\n                hw_value += rx_g{node_group_idx}_{node_i_idx} * sy{node_i_idx} + x_g{node_group_idx}_{node_i_idx} * sry{node_i_idx};".split(
                                    "\n"
                                )
                            )
                            if node_need_grad_env:
                                node_lines.append(
                                    f"                gw_value += x_g{node_group_idx}_{node_i_idx} * sy{node_i_idx};"
                                )
                    else:
                        for node_j_idx in node_unique_j:
                            node_lines.extend(
                                f"                hy{node_j_idx} += srx{node_j_idx} * w + sx{node_j_idx} * rw_value;\n                hw_value += srx{node_j_idx} * y{node_j_idx} + sx{node_j_idx} * ry{node_j_idx};".split(
                                    "\n"
                                )
                            )
                            if node_need_grad_env:
                                node_lines.append(
                                    f"                gw_value += sx{node_j_idx} * y{node_j_idx};"
                                )
                    if node_need_grad_env:
                        node_lines.append(
                            f"                he_value += raw_w * hw_value + raw_rw * gw_value;"
                        )
                    if node_with_env:
                        node_lines.append(
                            "                hw_value = env_e * hw_value"
                            + (" + re_e * gw_value;" if node_need_grad_env else ";")
                        )
                    if node_with_env and node_path_idx in node_l0_offsets:
                        emit_raw_l0_contractions(
                            node_lines,
                            node_spec,
                            node_path_idx,
                            node_l0_offsets,
                            node_l0_dim,
                            outputs={
                                "x": [
                                    f"hx_g{node_group_idx}_{i}"
                                    for i in range(int(node_spec["d1"]))
                                ],
                                "y": [f"hy{j}" for j in range(int(node_spec["d2"]))],
                                "w": "hw_value",
                            },
                            operands={
                                "x": [
                                    f"x_g{node_group_idx}_{i}"
                                    for i in range(int(node_spec["d1"]))
                                ],
                                "rx": [
                                    f"rx_g{node_group_idx}_{i}"
                                    for i in range(int(node_spec["d1"]))
                                ],
                                "y": [f"y{j}" for j in range(int(node_spec["d2"]))],
                                "ry": [f"ry{j}" for j in range(int(node_spec["d2"]))],
                                "w": "raw_w",
                                "rw": "raw_rw",
                            },
                        )
                    node_lines.extend(
                        f"                store_global_cs_f32(hw + weight_base + {node_weight_offset} + {weight_channel_offset(node_spec)}, hw_value);\n            }}".split(
                            "\n"
                        )
                    )

                if node_tail:
                    node_lines.append("}")
                for node_j_idx in range(node_d2):
                    node_lines.extend(
                        f"            const float sum{node_j_idx} = warp_reduce_sum(hy{node_j_idx});\n            if (lane == 0) hy_partial[(((static_cast<int64_t>(partial_base) + blockIdx.z * (CHANNEL_THREADS / 32) + warp_id) * num_edges + e) * GEN_IN2_DIM) + {node_y_offset + node_j_idx}] = sum{node_j_idx};".split(
                            "\n"
                        )
                    )
                node_lines.append("        }")
            if node_need_grad_env:
                node_lines.extend(
                    "        const float he_sum = warp_reduce_sum(he_value);\n        if (lane == 0) he_partial[Index(e) * GEN_ENV_PARTS + partial_base + blockIdx.z * (CHANNEL_THREADS / 32) + warp_id] = he_sum;".split(
                        "\n"
                    )
                )
            node_lines.append("    }")

            node_components = sum(
                node_d1 for (node__, node_d1), node__ in node_x_groups
            )
            node_lines += [
                "    if constexpr (EDGE_WORKERS > 1) {",
                f"        __shared__ float scratch[128 * {node_components}];",
            ]
            node_component = 0
            for node_group_idx, ((node__, node_d1), node__) in enumerate(node_x_groups):
                for node_i_idx in range(node_d1):
                    node_lines.append(
                        f"        scratch[threadIdx.x * {node_components} + {node_component}] = hx_g{node_group_idx}_{node_i_idx};"
                    )
                    node_component += 1
            node_lines += r"""        __syncthreads();
        if (edge_worker == 0) {{
            #pragma unroll
            for (int worker = 1; worker < EDGE_WORKERS; ++worker) {{""".format().split("\n")
            node_component = 0
            for node_group_idx, ((node__, node_d1), node__) in enumerate(node_x_groups):
                for node_i_idx in range(node_d1):
                    node_lines.append(
                        f"                hx_g{node_group_idx}_{node_i_idx} += scratch[(worker * CHANNEL_THREADS + threadIdx.x) * {node_components} + {node_component}];"
                    )
                    node_component += 1
            node_lines += [
                "            }",
                "        }",
                "    }",
                "    if (edge_worker == 0) {",
            ]
            for node_group_idx, ((node_x_offset, node_d1), node__) in enumerate(
                node_x_groups
            ):
                for node_i_idx in range(node_d1):
                    node_lines.append(
                        f"    hx[{node_row_index} * {node_in1_dim} + {node_x_offset} + u * {node_d1} + {node_i_idx}] = hx_g{node_group_idx}_{node_i_idx};"
                    )
            if 0 in node_group_indices:
                node_holes, node__ = flat_coverage(
                    node_specs, node_in1_dim, "x_offset", "d1"
                )
                for node_lo, node_hi in node_holes:
                    node_lines.append(
                        f"    if (blockIdx.z == 0) for (int i = threadIdx.x; i < {node_hi - node_lo}; i += CHANNEL_THREADS) hx[{node_row_index} * {node_in1_dim} + {node_lo} + i] = 0.0f;"
                    )
            node_lines.append("    }")
            if node_tail:
                node_lines = [
                    re.sub(
                        "(= )(__ldg\\((?:x|rx) \\+ [^;]+)(;)",
                        r"\1active_channel ? \2 : 0.0f\3",
                        node_line,
                    )
                    for node_line in node_lines
                ]
                node_lines = [
                    ("if (active_channel) " + node_line)
                    if node_line.strip().startswith("hx[")
                    else node_line
                    for node_line in node_lines
                ]
            body = "\n".join(node_lines)
            payload = f"const int partial_base = {bases[bundle_id]};\n{body}"
            cases.append(
                payload
                if len(bucket) == 1
                else f"case {branch}: {{\n{payload}\nbreak;\n}}"
            )
        body = "\n".join(cases)
        if len(bucket) > 1:
            body = "switch (blockIdx.y) {\n" + body + "\n}"
        bodies.append(
            body if len(buckets) == 1 else f"case {bucket_id}: {{\n{body}\nbreak;\n}}"
        )
        t = threads[bundles[bucket[0]][0]]
        tiles = (widths[bundles[bucket[0]][0]] + t - 1) // t
        block = t if equal_width and not segmented else t * (128 // t)
        # Round live-state demand to a register budget; one rule for all bundles.
        live = max(
            6
            * (
                sum(groups[g][0][1] for g in bundles[b])
                + max(int(path["d2"]) for g in bundles[b] for _, path in groups[g][1])
            )
            * (scalar_bytes // 4)
            + (32 if scalar_bytes == 4 else 16)
            for b in bucket
        )
        budget = min(256, ((live + 31) // 32) * 32)
        min_blocks = max(1, 65536 // (block * budget))
        if segmented:
            min_blocks = max(2 if scalar_bytes == 8 else 3, 65536 // (128 * budget))
        launches.append(
            r"""    if (rows > 0) lunatp_uvuconv_double_bwd_source_fp32<{bucket_id}, {t}, {block}, {min_blocks}><<<dim3(rows, {bucket_count}, {tiles}), {block}, 0, state.streams[{bucket_id}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), go.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), rw.data_ptr<float>(), dst.data_ptr<int32_t>(),
        edges.data_ptr<EdgeIndex>(), row_nodes.data_ptr<int32_t>(), row_begin.data_ptr<EdgeIndex>(),
        row_end.data_ptr<EdgeIndex>(), row_order.data_ptr<{source_index_type}>(), hx_rows.data_ptr<float>(),
        hy_partial.data_ptr<float>(), hw.data_ptr<float>(), num_edges, rows);
    cudaEventRecord(state.done[{bucket_id}], state.streams[{bucket_id}].stream());""".format(
                bucket_id=bucket_id,
                t=t,
                block=block,
                min_blocks=min_blocks,
                bucket_count=len(bucket),
                tiles=tiles,
                source_index_type="EdgeIndex" if segmented else "int32_t",
            )
        )
    if len(buckets) > 1:
        bodies = ["switch (BUCKET) {\n" + "\n".join(bodies) + "\n}"]
    holes, overlap = flat_coverage(specs, op.out_dim, "out_offset", "d3")
    hgo_clear = (
        holes or overlap or (not equal_width and not segmented and scalar_bytes == 4)
    )
    hgo_source, hgo_launch = "", ""
    external_hgo = segmented and not equal_width
    if equal_width:
        ht = threads[0]
        cooperative_hgo = segmented and aligned
        hwidth = max(
            sum(int(spec["d3"]) for _, spec in paths)
            for _, paths in group_double_bwd_dgrad_out_paths(specs)
        )
        scratch_tile = 16 if 128 * hwidth * scalar_bytes > 48 * 1024 else None
        output_specs = specs
        output_in1_dim = dims["in1_dim"]
        output_in2_dim = dims["in2_dim"]
        output_out_dim = dims["out_dim"]
        output_weight_numel = dims["weight_numel"]
        output_prefer_short_factorization = segmented or widths[0] // ht > 1
        output_accumulate = overlap
        output_cooperative_rows = cooperative_hgo
        output_scratch_tile = scratch_tile if cooperative_hgo else None
        output_index_type = "Index"
        output_with_l0 = with_l0
        output_with_env = with_env
        output_with_re = need_grad_env
        output_cast = "" if output_index_type == "int" else output_index_type
        output_cases: list[str] = []
        for output_group_idx, (
            (output_x_offset, output_d1),
            output_raw_group_specs,
        ) in enumerate(group_double_bwd_dgrad_out_paths(output_specs)):
            output_y_groups: dict[
                tuple[int, int], list[tuple[int, dict[str, object]]]
            ] = defaultdict(list)
            for output_path_idx, output_spec in output_raw_group_specs:
                output_y_groups[
                    (int(output_spec["y_offset"]), int(output_spec["d2"]))
                ].append((output_path_idx, output_spec))
            output_ordered_y_groups = sorted(
                output_y_groups.items(),
                key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
                reverse=True,
            )
            output_group_specs = sorted(
                output_raw_group_specs,
                key=lambda path: len(path[1]["terms"]),
                reverse=True,
            )
            output_lines = [f"case {output_group_idx}: {{"]
            output_channels = int(output_group_specs[0][1]["channels"])
            if output_channels % 32:
                output_lines.append(f"if (u >= {output_channels}) break;")
            for output_path_idx, output_spec in output_group_specs:
                for output_k_idx in range(int(output_spec["d3"])):
                    output_lines.append(
                        f"    float acc_p{output_path_idx}_{output_k_idx} = 0.0f;"
                    )
            output_lines.extend(
                [
                    "    const Index begin = rowptr[t];",
                    "    const Index end = rowptr[t + 1];",
                    (
                        f"    for (Index {edge} = begin + hgo_edge_worker; {edge} < end; {edge} += DGRAD_OUT_EDGE_WORKERS) {{"
                        if output_cooperative_rows
                        else f"    for (Index {edge} = begin; {edge} < end; ++{edge}) {{"
                    ),
                    f"        const int s = __ldg(src + {edge});",
                    f"""        const {output_index_type} x_base = {(output_cast + "(s)" if output_cast else "s")} * {output_in1_dim} + {output_x_offset} + u * {output_d1};""",
                    f"""        const {output_index_type} y_base = {(output_cast + "(e)" if output_cast else "e")} * {output_in2_dim};""",
                    f"""        const {output_index_type} weight_base = {(output_cast + "(e)" if output_cast else "e")} * {output_weight_numel};""",
                ]
            )
            if output_with_env:
                output_lines.append(f"        const float env_e = __ldg(env + e);")
                if output_with_re:
                    output_lines.append(
                        "        const float re_e = re ? __ldg(re + e) : 0.0f;"
                    )
            for output_i_idx in range(output_d1):
                output_lines.extend(
                    f"        const float x{output_i_idx} = __ldg(x + x_base + {output_i_idx});\n        const float vrx{output_i_idx} = __ldg(rx + x_base + {output_i_idx});".split(
                        "\n"
                    )
                )
            for (
                output_y_offset,
                output_d2,
            ), output_y_group_specs in output_ordered_y_groups:
                output_lines.append("        {")
                for output_j_idx in range(output_d2):
                    output_lines.extend(
                        f"            const float y{output_j_idx} = __ldg(y + y_base + {output_y_offset + output_j_idx});\n            const float vry{output_j_idx} = __ldg(ry + y_base + {output_y_offset + output_j_idx});".split(
                            "\n"
                        )
                    )
                    if output_with_env:
                        output_lines[-2:] = [
                            f"            const float raw_y{output_j_idx} = __ldg(y + y_base + {output_y_offset + output_j_idx});",
                            f"            const float raw_ry{output_j_idx} = __ldg(ry + y_base + {output_y_offset + output_j_idx});",
                            f"            const float y{output_j_idx} = env_e * raw_y{output_j_idx};",
                            f"            const float vry{output_j_idx} = env_e * raw_ry{output_j_idx}"
                            + (
                                f" + re_e * raw_y{output_j_idx};"
                                if output_with_re
                                else ";"
                            ),
                        ]
                for output_path_idx, output_spec in sorted(
                    output_y_group_specs,
                    key=lambda path: len(path[1]["terms"]),
                    reverse=True,
                ):
                    output_weight_offset = int(output_spec["weight_offset"])
                    output_terms = list(output_spec["terms"])
                    output_unique_i = sorted(
                        {int(output_term[0]) for output_term in output_terms}
                    )
                    output_unique_j = sorted(
                        {int(output_term[1]) for output_term in output_terms}
                    )
                    output_factor_on_i = output_prefer_short_factorization and len(
                        output_unique_i
                    ) < len(output_unique_j)
                    output_lines.extend(
                        f"            {{\n                const float w = __ldg(weight + weight_base + {output_weight_offset} + {weight_channel_offset(output_spec)});\n                const float vrw = __ldg(rw + weight_base + {output_weight_offset} + {weight_channel_offset(output_spec)});".split(
                            "\n"
                        )
                    )
                    if output_with_l0 and int(output_spec["d3"]) == 1:
                        output_lines.append(
                            "                float scalar_value = 0.0f;"
                        )
                    output_scalar_first = True
                    if output_factor_on_i:
                        for output_i_idx in output_unique_i:
                            output_lines.extend(
                                f"                const float xw{output_i_idx} = x{output_i_idx} * w;\n                const float xd{output_i_idx} = vrx{output_i_idx} * w + x{output_i_idx} * vrw;".split(
                                    "\n"
                                )
                            )
                    else:
                        for output_j_idx in output_unique_j:
                            output_lines.extend(
                                f"                const float yw{output_j_idx} = y{output_j_idx} * w;\n                const float yd{output_j_idx} = vry{output_j_idx} * w + y{output_j_idx} * vrw;".split(
                                    "\n"
                                )
                            )
                    for output_class_idx, output_signature_class in enumerate(
                        build_signed_ij_classes(
                            output_terms,
                            spec_y_offset=output_y_offset,
                            y_name_base=output_y_offset,
                        )
                    ):
                        output_representative = output_signature_class.members[0]

                        output_lines.append("                {")
                        if output_factor_on_i:
                            output_dual_product_value = f"xd{output_representative.i_idx} * y{output_representative.j_name} + xw{output_representative.i_idx} * vry{output_representative.j_name}"
                        else:
                            output_dual_product_value = f"vrx{output_representative.i_idx} * yw{output_representative.j_name} + x{output_representative.i_idx} * yd{output_representative.j_name}"
                        output_lines.append(
                            f"                    float p{output_class_idx} = {output_dual_product_value};"
                        )
                        for output_member in output_signature_class.members[1:]:
                            output_op = (
                                "+=" if output_member.relative_sign > 0 else "-="
                            )
                            if output_factor_on_i:
                                output_dual_product_value = f"xd{output_member.i_idx} * y{output_member.j_name} + xw{output_member.i_idx} * vry{output_member.j_name}"
                            else:
                                output_dual_product_value = f"vrx{output_member.i_idx} * yw{output_member.j_name} + x{output_member.i_idx} * yd{output_member.j_name}"
                            output_lines.append(
                                f"                    p{output_class_idx} {output_op} {output_dual_product_value};"
                            )
                        for (
                            output__,
                            output__,
                            output_k_idx,
                            output_coeff,
                        ) in output_representative.terms:
                            output_lines.append(
                                f"                    acc_p{output_path_idx}_{int(output_k_idx)} += {float_literal(float(output_coeff))} * p{output_class_idx};"
                            )
                            if output_with_l0 and int(output_spec["d3"]) == 1:
                                output_scalar_op = "=" if output_scalar_first else "+="
                                output_lines.append(
                                    f"                scalar_value {output_scalar_op} {float_literal(float(output_coeff))} * p{output_class_idx};"
                                )
                                output_scalar_first = False
                        output_lines.append("                }")
                    if output_with_l0 and int(output_spec["d3"]) == 1:
                        output_offsets, output_width = l0_feature_layout(output_specs)
                        if output_with_env:
                            output_lines.append("                scalar_value = 0.0f;")
                            for (
                                output_i,
                                output_j,
                                output__k,
                                output_coeff,
                            ) in output_spec["terms"]:
                                output_lines.append(
                                    f"                scalar_value += {float_literal(float(output_coeff))} * (vrx{output_i} * raw_y{output_j} * w + x{output_i} * raw_ry{output_j} * w + x{output_i} * raw_y{output_j} * vrw);"
                                )
                        if "l0_path" in output_spec:
                            output_pointer = f"hl0 + Index(e) * {output_width} + {output_offsets[output_path_idx]} + u"
                            output_lines.append(
                                "                if (hl0) "
                                + f"atomicAdd({output_pointer}, scalar_value);"
                            )
                        else:
                            output_lines.append(
                                f"                if (hl0) hl0[Index(e) * {output_width} + {output_offsets[output_path_idx]} + u] = scalar_value;"
                            )
                    output_lines.append("            }")
                output_lines.append("        }")
            output_lines.append("    }")
            if output_cooperative_rows and output_scratch_tile is not None:
                output_outputs = [
                    (output_path_idx, output_k_idx)
                    for output_path_idx, output_spec in output_group_specs
                    for output_k_idx in range(int(output_spec["d3"]))
                ]
                output_lines.append("    if constexpr (DGRAD_OUT_EDGE_WORKERS > 1) {")
                for output_tile_start in range(
                    0, len(output_outputs), output_scratch_tile
                ):
                    output_tile_outputs = output_outputs[
                        output_tile_start : output_tile_start + output_scratch_tile
                    ]
                    for output_output_index, (
                        output_path_idx,
                        output_k_idx,
                    ) in enumerate(output_tile_outputs):
                        output_lines.append(
                            f"        hgo_scratch[(hgo_edge_worker * {output_scratch_tile} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread] = acc_p{output_path_idx}_{output_k_idx};"
                        )
                    output_lines.extend(r"""        __syncthreads();
        if (hgo_edge_worker == 0) {{
            #pragma unroll
            for (int worker = 1; worker < DGRAD_OUT_EDGE_WORKERS; ++worker) {{""".format().split("\n"))
                    for output_output_index, (
                        output_path_idx,
                        output_k_idx,
                    ) in enumerate(output_tile_outputs):
                        output_lines.append(
                            f"                acc_p{output_path_idx}_{output_k_idx} += hgo_scratch[(worker * {output_scratch_tile} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread];"
                        )
                    output_lines.extend(
                        "            }\n        }\n        __syncthreads();".split("\n")
                    )
                output_lines.extend(["    }", "    if (hgo_edge_worker == 0) {"])
            elif output_cooperative_rows:
                output_output_width = sum(
                    int(output_spec["d3"])
                    for output__, output_spec in output_group_specs
                )
                output_lines.append("    if constexpr (DGRAD_OUT_EDGE_WORKERS > 1) {")
                output_output_index = 0
                for output_path_idx, output_spec in output_group_specs:
                    for output_k_idx in range(int(output_spec["d3"])):
                        output_lines.append(
                            f"        hgo_scratch[(hgo_edge_worker * {output_output_width} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread] = acc_p{output_path_idx}_{output_k_idx};"
                        )
                        output_output_index += 1
                output_lines.extend(r"""        __syncthreads();
        if (hgo_edge_worker == 0) {{
            #pragma unroll
            for (int worker = 1; worker < DGRAD_OUT_EDGE_WORKERS; ++worker) {{""".format().split("\n"))
                output_output_index = 0
                for output_path_idx, output_spec in output_group_specs:
                    for output_k_idx in range(int(output_spec["d3"])):
                        output_lines.append(
                            f"                acc_p{output_path_idx}_{output_k_idx} += hgo_scratch[(worker * {output_output_width} + {output_output_index}) * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread];"
                        )
                        output_output_index += 1
                output_lines.extend(
                    [
                        "            }",
                        "        }",
                        "    }",
                        "    if (hgo_edge_worker == 0) {",
                    ]
                )
            output_lines.append(
                f"""    const {output_index_type} hgo_base = {(output_cast + "(t)" if output_cast else "t")} * {output_out_dim};"""
            )
            for output_path_idx, output_spec in output_group_specs:
                output_out_offset = int(output_spec["out_offset"])
                output_d3 = int(output_spec["d3"])
                for output_k_idx in range(output_d3):
                    output_address = f"hgo_base + {output_out_offset} + u * {output_d3} + {output_k_idx}"
                    output_value = f"acc_p{output_path_idx}_{output_k_idx}"
                    if output_accumulate:
                        output_lines.append(
                            f"    atomicAdd(hgo + {output_address}, {output_value});"
                        )
                    else:
                        output_lines.append(
                            f"    hgo[{output_address}] = {output_value};"
                        )
            if output_cooperative_rows:
                output_lines.append("    }")
            output_lines.extend(["    break;", "}"])
            output_cases.append("\n".join(output_lines))
        hcases, hn = "\n".join(output_cases), len(output_cases)
        hsetup = (
            r"""constexpr int DGRAD_OUT_CHANNEL_THREADS = {ht};
    constexpr int DGRAD_OUT_EDGE_WORKERS = 128 / DGRAD_OUT_CHANNEL_THREADS;
    __shared__ float hgo_scratch[{scratch_words}];
    const int hgo_edge_worker = DGRAD_OUT_EDGE_WORKERS == 1 ? 0 : threadIdx.x / DGRAD_OUT_CHANNEL_THREADS;
    const int hgo_channel_thread = DGRAD_OUT_EDGE_WORKERS == 1 ? threadIdx.x : threadIdx.x % DGRAD_OUT_CHANNEL_THREADS;
    const int u = blockIdx.y * DGRAD_OUT_CHANNEL_THREADS + hgo_channel_thread;""".format(
                ht=ht,
                scratch_words=128 * (scratch_tile or hwidth),
            )
            if cooperative_hgo
            else f"const int u = blockIdx.y * {ht} + threadIdx.x;"
        )
        hgo_source = r"""
__global__ void lunatp_uvuconv_double_bwd_dgrad_out_dst_rows_fp32(const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ weight, const float* __restrict__ rx,
    const float* __restrict__ ry, const float* __restrict__ rw,
    const int32_t* __restrict__ src, const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order, float* __restrict__ hgo, int64_t num_nodes) {{
    const Index task_id = blockIdx.x;
    {hsetup}
    const int group = task_id / static_cast<int>(num_nodes);
    const int node_rank = static_cast<int>(task_id - int64_t(group) * num_nodes);
    if (group >= {hn} || u >= {widths_value}) return;
    const int t = node_order[node_rank];
    switch(group) {{
{hcases}
    default: break;
    }}
}}
""".format(
            hsetup=hsetup,
            hn=hn,
            widths_value=widths[0],
            hcases=hcases,
        )
        hgo_launch = f"    if (num_nodes > 0) lunatp_uvuconv_double_bwd_dgrad_out_dst_rows_fp32<<<dim3(num_nodes * {hn}, {(widths[0] + ht - 1) // ht}), {(ht * (128 // ht) if cooperative_hgo else ht)}, 0, current>>>(\n        x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), rx.data_ptr<float>(),\n        ry.data_ptr<float>(), rw.data_ptr<float>(), src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(),\n        node_order.data_ptr<int32_t>(), hgo.data_ptr<float>(), num_nodes);"
    elif not external_hgo:
        hgo_source, tasks = make_uvu_conv_double_bwd_dgrad_out_kernel(
            op,
            accumulate=scalar_bytes == 4 or overlap,
            index_type=dims["index_type"],
            with_l0=with_l0,
            with_env=with_env,
            with_re=need_grad_env,
        )
        hgo_launch = f"    if (num_nodes > 0) lunatp_uvuconv_double_bwd_dgrad_out_dst_tasks_fp32<<<num_nodes * {tasks}, 32, 0, current>>>(\n        x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), rx.data_ptr<float>(),\n        ry.data_ptr<float>(), rw.data_ptr<float>(), src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(),\n        node_order.data_ptr<int32_t>(), hgo.data_ptr<float>(), static_cast<int>(num_nodes));"
    hy_lines = []
    for bundle_id, bundle in enumerate(bundles):
        written = sorted(
            {
                int(s["y_offset"]) + j
                for g in bundle
                for _, s in groups[g][1]
                for j in range(int(s["d2"]))
            }
        )
        ranges = []
        for col in written:
            if ranges and ranges[-1][1] == col:
                ranges[-1] = (ranges[-1][0], col + 1)
            else:
                ranges.append((col, col + 1))
        condition = (
            " || ".join(f"(feature >= {lo} && feature < {hi})" for lo, hi in ranges)
            or "false"
        )
        guard = "" if written == list(range(op.in2_dim)) else f"if ({condition}) "
        for p in range(
            bases[bundle_id], bases[bundle_id] + (widths[bundle[0]] + 31) // 32
        ):
            hy_lines.append(
                f"    {guard}value += hy_partial[static_cast<int64_t>({p}) * total + index];"
            )
    row_loop_open = (
        "for (Index node_rank = blockIdx.x; node_rank < num_rows; node_rank += gridDim.x) {"
        if fused
        else ""
    )
    row_loop_close = "}" if fused else ""
    row_setup = (
        f"""const Index row = row_order[{"node_rank" if fused else "blockIdx.x"}];
    const int s_node = row_nodes[row];
    const Index begin = row_begin[row], end = row_end[row];"""
        if segmented
        else f"""const int s_node = row_order[{"node_rank" if fused else "blockIdx.x"}];
    const Index row = s_node;"""
    )
    bounds = (
        f"""__launch_bounds__({(128 if segmented else "BLOCK_THREADS")}, MIN_BLOCKS)"""
        if segmented or (equal_width and (not fused or scalar_bytes == 4))
        else ""
    )
    reduce_launch = (
        """    if (num_nodes > 0) lunatp_uvuconv_double_bwd_reduce_x_fp32<<<(num_nodes * GEN_IN1_DIM + 255) / 256, 256, 0, current>>>(
        hx_rows.data_ptr<float>(), segment_rowptr.data_ptr<EdgeIndex>(), hx.data_ptr<float>(), num_nodes);"""
        if segmented
        else ""
    )
    zero_hgo = (
        "    C10_CUDA_CHECK(cudaMemsetAsync(hgo.data_ptr<float>(), 0, hgo.numel()*sizeof(float), current));"
        if hgo_clear and not external_hgo
        else ""
    )
    # Channel-only launches need no worker mapping; cooperative schedules retain
    # bounded channel indices, including their single-worker specializations.
    if segmented:
        worker, channel = (
            "EDGE_WORKERS == 1 ? 0 : threadIdx.x / CHANNEL_THREADS",
            "EDGE_WORKERS == 1 ? threadIdx.x : threadIdx.x % CHANNEL_THREADS",
        )
    else:
        worker, channel = (
            ("0", "threadIdx.x")
            if equal_width
            else ("threadIdx.x / CHANNEL_THREADS", "threadIdx.x % CHANNEL_THREADS")
        )
    source = (r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>
namespace {{
"""
        + (f"constexpr int GEN_ENV_PARTS = {partial_count};\n" if need_grad_env else "")
        + r"""constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
__device__ __forceinline__ float warp_reduce_sum(float value) {{
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}}
__device__ __forceinline__ void store_global_cs_f32(float* ptr, float value) {{
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(ptr), "f"(value) : "memory");
}}
template<int BUCKET, int CHANNEL_THREADS, int BLOCK_THREADS, int MIN_BLOCKS>
__global__ {bounds} void lunatp_uvuconv_double_bwd_source_fp32(
    const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ weight, const float* __restrict__ grad_out,
    const float* __restrict__ rx, const float* __restrict__ ry, const float* __restrict__ rw,
    const int32_t* __restrict__ dst, const EdgeIndex* __restrict__ edge_ids_by_src,
    const int32_t* __restrict__ row_nodes, const EdgeIndex* __restrict__ row_begin,
    const EdgeIndex* __restrict__ row_end, const {source_index_type}* __restrict__ row_order,
    float* __restrict__ hx, float* __restrict__ hy_partial, float* __restrict__ hw, int64_t num_edges, int64_t num_rows) {{
    {row_loop_open}
    {row_setup}
    constexpr int EDGE_WORKERS = BLOCK_THREADS / CHANNEL_THREADS;
    const int edge_worker = {worker};
    const int channel_thread = {channel};
    const int u = blockIdx.z * CHANNEL_THREADS + channel_thread;
    const int lane = threadIdx.x & 31, warp_id = channel_thread >> 5;
{bodies}
    {row_loop_close}
}}
__global__ void lunatp_uvuconv_double_bwd_reduce_x_fp32(const float* __restrict__ partial, const EdgeIndex* __restrict__ rowptr,
    float* __restrict__ hx, int64_t nodes) {{
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nodes * GEN_IN1_DIM) return;
    const int node = i / GEN_IN1_DIM, c = i % GEN_IN1_DIM;
    float value = 0.0f;
    for (Index s = rowptr[node]; s<rowptr[node+1]; ++s) value += partial[int64_t(s)*GEN_IN1_DIM+c];
    hx[i] = value;
}}
__global__ void lunatp_uvuconv_double_bwd_reduce_y_fp32(const float* __restrict__ hy_partial, float* __restrict__ hy, int64_t edges) {{
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = edges * GEN_IN2_DIM;
    if (index >= total) return;
    const int feature = index % GEN_IN2_DIM;
    float value = 0.0f;
{hy_lines}
    hy[index] = value;"""
        + (r"""
    if (feature == 0) {{
        const Index edge = index / GEN_IN2_DIM;
        float sum = 0.0f;
        for (int p = 0; p < GEN_ENV_PARTS; ++p) sum += he_partial[edge * GEN_ENV_PARTS + p];
        he[edge] = sum;
    }}""" if need_grad_env else "")
        + r"""
}}
}}
{hgo_source}
namespace {{
struct SourceStreams {{
    ~SourceStreams() {{
        // Match CUDAEvent's device-aware, non-throwing shutdown behavior.
        try {{
            c10::cuda::CUDAGuard guard(streams.front().device_index());
            cudaEventDestroy(start);
            for (auto event : done) cudaEventDestroy(event);
        }} catch (...) {{ /* CUDA may already be shutting down. */ }}
    }}
    SourceStreams(const SourceStreams&) = delete;
    SourceStreams& operator=(const SourceStreams&) = delete;
    std::vector<at::cuda::CUDAStream> streams;
    std::vector<cudaEvent_t> done;
    cudaEvent_t start;
    explicit SourceStreams(int device, const std::vector<int64_t>& handles) {{
        c10::cuda::CUDAGuard guard(device);
        cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
        done.resize({buckets_count});
        for (int b=0; b<{buckets_count}; ++b) {{
            streams.push_back(at::cuda::getStreamFromExternal(
                reinterpret_cast<cudaStream_t>(handles.at(b)), device));
            cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming);
        }}
    }}
}};
}}
void run_uvu_conv_double_bwd(SourceStreams& state, torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor go,
    torch::Tensor rx, torch::Tensor ry, torch::Tensor rw, torch::Tensor src, torch::Tensor dst,
    torch::Tensor rowptr, torch::Tensor edges, torch::Tensor node_order,
    torch::Tensor row_nodes, torch::Tensor row_begin, torch::Tensor row_end, torch::Tensor row_order,
    torch::Tensor segment_rowptr, torch::Tensor hx, torch::Tensor hx_rows, torch::Tensor hy,
    torch::Tensor hy_partial, torch::Tensor hw, torch::Tensor hgo, int64_t num_nodes,
    int64_t num_edges, int64_t rows) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto current = at::cuda::getCurrentCUDAStream(x.get_device());
{zero_hgo}
"""
        + (r"""    C10_CUDA_CHECK(cudaMemsetAsync(hl0->data_ptr<float>(), 0, hl0->numel() * sizeof(float), current));
""" if with_l0 and not external_hgo and any("l0_path" in spec for spec in specs) else "")
        + r"""    cudaEventRecord(state.start, current);
    for (auto& stream : state.streams) cudaStreamWaitEvent(stream.stream(), state.start, 0);
{launches}
{hgo_launch}
    for (auto done : state.done) cudaStreamWaitEvent(current, done, 0);
{reduce_launch}
    if (num_edges > 0) lunatp_uvuconv_double_bwd_reduce_y_fp32<<<(num_edges * GEN_IN2_DIM + 255) / 256, 256, 0, current>>>(
        hy_partial.data_ptr<float>(), hy.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    pybind11::class_<SourceStreams>(m, "_Streams", pybind11::module_local())
        .def(pybind11::init<int, const std::vector<int64_t>&>());
    m.attr("_Streams").attr("stream_count") = {buckets_count};
    m.attr("_Streams").attr("stream_priority") = {worker_offset};
    m.def("run_uvu_conv_double_bwd", &run_uvu_conv_double_bwd);
}}
""").format(
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        bounds=bounds,
        source_index_type="EdgeIndex" if segmented else "int32_t",
        row_loop_open=row_loop_open,
        row_setup=row_setup,
        worker=worker,
        channel=channel,
        bodies=chr(10).join(bodies),
        row_loop_close=row_loop_close,
        hy_lines=chr(10).join(hy_lines),
        hgo_source=hgo_source,
        buckets_count=len(buckets),
        zero_hgo=zero_hgo,
        launches=chr(10).join(launches),
        hgo_launch=hgo_launch,
        reduce_launch=reduce_launch,
        worker_offset=-1 if fused else 0,
    )
    parameters = (
        {"grad_l0": "const float* __restrict__", "hl0": "float* __restrict__"}
        if with_l0
        else {}
    )
    if with_env:
        parameters.update(
            {
                "env": "const float* __restrict__",
                "re": "const float* __restrict__",
            }
        )
    if need_grad_env:
        parameters.update(
            {"he_partial": "float* __restrict__", "he": "float* __restrict__"}
        )
    return (
        add_optional_arguments(
            source,
            "run_uvu_conv_double_bwd",
            parameters,
            ("grad_l0", "hl0", "env", "re", "he_partial", "he"),
        ),
        partial_count,
        segmented,
        external_hgo,
    )


def group_double_bwd_dgrad_out_paths(specs: list[UvuPathSpec]):
    """Bound live output accumulators while keeping every path and FP32 term."""
    grouped = []
    for key, paths in group_source_input_paths(specs):
        # Reserve scratch for x/rx and their factored products. Small irreps can
        # retain more outputs without the register pressure of high-degree ones.
        _, input_components = key
        live_output_cap = max(24, 64 - 4 * input_components)
        current = []
        live_outputs = 0
        for item in sorted(paths, key=lambda path: len(path[1]["terms"]), reverse=True):
            width = int(item[1]["d3"])
            if current and live_outputs + width > live_output_cap:
                grouped.append((key, current))
                current = []
                live_outputs = 0
            current.append(item)
            live_outputs += width
        if current:
            grouped.append((key, current))
    return grouped


def run(op, x, y, w, g, rx, ry, rw, graph, env, re, gl0, *, need_grad_env):
    weight = w
    grad_out = g
    grad_l0 = gl0
    need_grad_env = env is not None and need_grad_env
    if graph.src.numel() == 0:
        values = (x, y, weight, grad_out)
        if need_grad_env:
            values += (env,)
        if grad_l0 is not None:
            values += (grad_l0,)
        return tuple(torch.zeros_like(t) for t in values)

    cache_name = "_source_second_state" + (
        "_l0" if grad_l0 is not None else "_env" if env is not None else ""
    )
    if env is not None and grad_l0 is not None:
        cache_name += "_env"
    if env is not None and not need_grad_env:
        cache_name += "_fixed"
    cached = getattr(op, cache_name, None)
    if cached is None:
        source, parts, segmented, external = make_uvu_conv_double_bwd_source(
            op,
            with_l0=grad_l0 is not None,
            with_env=env is not None,
            need_grad_env=need_grad_env,
        )
        ext = load_tp_extension(
            (
                "lunatp_uvuconv_double_bwd_fp64"
                if x.dtype == torch.float64
                else "lunatp_uvuconv_double_bwd_fp32"
            ),
            cuda_source_to_fp64(source) if x.dtype == torch.float64 else source,
            max_channels=op.max_channels,
            index_bits=op.index_bits,
        )
        cached = (ext, parts, segmented, external)
        setattr(op, cache_name, cached)

    ext, parts, segmented, external = cached
    hx = torch.empty_like(x)
    hy = torch.empty_like(y)
    hw = torch.empty_like(weight)
    hgo = torch.empty_like(grad_out)
    hl0 = torch.empty_like(grad_l0) if grad_l0 is not None else None
    he = torch.empty_like(env) if need_grad_env else None
    he_partial = (
        torch.empty((graph.src.numel(), parts), device=x.device, dtype=x.dtype)
        if need_grad_env
        else None
    )
    tables = graph.source_segments
    row_nodes, row_begin, row_end, row_order = (
        (tables.nodes, tables.begin, tables.end, tables.order)
        if segmented
        else (graph.src, graph.src_rowptr, graph.src_rowptr, graph.source_node_order)
    )
    rows = row_nodes.numel() if segmented else x.shape[0]
    hx_rows = (
        torch.empty((rows, op.in1_dim), device=x.device, dtype=x.dtype)
        if segmented
        else hx
    )
    hy_partial = torch.empty(
        (parts, graph.src.numel(), op.in2_dim), device=x.device, dtype=x.dtype
    )

    if external:
        state = get_overlap_state(op, x.device)
        current = torch.cuda.current_stream(x.device)
        state.start.record(current)
        state.stream.wait_event(state.start)
        with torch.cuda.stream(state.stream):
            cache_name = "_double_bwd_dgrad_out_ext" + (
                "_l0" if hl0 is not None else "_env" if env is not None else ""
            )
            if env is not None and hl0 is not None:
                cache_name += "_env"
            if env is not None and not need_grad_env:
                cache_name += "_fixed"
            dgrad_extension = getattr(op, cache_name, None)
            if dgrad_extension is None:
                source = make_uvu_conv_double_bwd_dgrad_out_source(
                    op,
                    with_l0=hl0 is not None,
                    with_env=env is not None,
                    with_re=need_grad_env,
                )
                dgrad_extension = load_tp_extension(
                    (
                        "lunatp_uvuconv_double_bwd_dgrad_out_dst_tasks_fp64"
                        if x.dtype == torch.float64
                        else "lunatp_uvuconv_double_bwd_dgrad_out_dst_tasks_fp32"
                    ),
                    cuda_source_to_fp64(source) if x.dtype == torch.float64 else source,
                    max_channels=op.max_channels,
                    index_bits=op.index_bits,
                )
                setattr(op, cache_name, dgrad_extension)
            if x.shape[0]:
                order = (
                    graph.node_order
                    if graph.node_order is not None
                    else graph.src.new_empty((0,))
                )
                dgrad_extension.run_uvuconv_double_bwd_dgrad_out(
                    x,
                    y,
                    weight,
                    rx,
                    ry,
                    rw,
                    graph.src,
                    graph.dst_rowptr,
                    order,
                    hgo,
                    hl0,
                    env,
                    re,
                )
            state.done.record()

    ext.run_uvu_conv_double_bwd(
        get_overlap_state(op, x.device, ext._Streams),
        x,
        y,
        weight,
        grad_out,
        rx,
        ry,
        rw,
        graph.src,
        graph.dst,
        graph.dst_rowptr,
        graph.edge_ids_by_src,
        graph.node_order,
        row_nodes,
        row_begin,
        row_end,
        row_order,
        tables.rowptr,
        hx,
        hx_rows,
        hy,
        hy_partial,
        hw,
        hgo,
        x.shape[0],
        graph.src.numel(),
        rows,
        grad_l0,
        hl0,
        env,
        re,
        he_partial,
        he,
    )

    if external:
        current.wait_event(state.done)
        hgo.record_stream(current)
    if he is not None:
        return (hx, hy, hw, hgo, he) if hl0 is None else (hx, hy, hw, hgo, he, hl0)
    return (hx, hy, hw, hgo) if hl0 is None else (hx, hy, hw, hgo, hl0)
