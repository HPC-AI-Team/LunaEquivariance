"""UVU backward CUDA templates."""

from collections import defaultdict
from collections.abc import Mapping
import torch
from ...utils.cg import weight_channel_offset
from ....utils.precision import cuda_source_for_dtype, float_literal
from ...utils.contractions import INPUTS, OUTPUT_SHAPES, PROGRAMS
from ...utils.runtime import DERIVATIVE_STREAM_SOURCE
from .utils import SHARED_WEIGHT_ROWS, make_grid_source, get_uvu_launch_config


# Instruction math and traversals used by the kernels above.
def make_uvu_backward_source(
    meta: Mapping[str, object], *, need_grad_weight: bool = True, needs=None
) -> str:
    outputs = ["dx", "gy"]
    if need_grad_weight:
        outputs.append("gw")
    gw_pointer = "gw.data_ptr<float>()" if need_grad_weight else "nullptr"
    gw_work_pointer = "gw_work.data_ptr<float>()" if need_grad_weight else "nullptr"
    gw_allocation = (
        "(shared_weight && !use_shared_weight_reduce && !GEN_GROUPED_Y) ? "
        "torch::zeros_like(weight) : torch::empty_like(weight)"
        if need_grad_weight
        else "torch::Tensor()"
    )
    gw_work_allocation = "gw" if need_grad_weight else "torch::Tensor()"
    specs = meta["generated_specs"]
    in1_dim = meta["in1_dim"]
    in2_dim = meta["in2_dim"]
    out_dim = meta["out_dim"]
    weight_numel = meta["weight_numel"]
    max_channels = meta["max_channels"]
    spec_channels = [spec["channels"] for spec in specs]
    channel_set = set(spec_channels)
    weight_mode = meta["weight_mode"]
    # Group long path lists to reduce repeated edge-gradient atomics.
    use_grouped_y = len(specs) >= 32
    use_y_block_reduce = len(channel_set) == 1 or max_channels <= 128
    scope_uniform_y = weight_mode == "external" and channel_set == {128}
    enable_shared_weight_reduce = weight_mode == "shared" and (
        max_channels >= 128 or not use_grouped_y
    )
    x_channel_tile, x_threads, x_min_blocks = get_uvu_launch_config(max_channels)
    if enable_shared_weight_reduce:
        spec_channel_tile, spec_threads, spec_min_blocks = get_uvu_launch_config(
            max_channels, spec_channels
        )
    else:
        spec_channel_tile, spec_threads, spec_min_blocks = get_uvu_launch_config(
            max_channels
        )

    # A grouped edge reduction cannot use more than 1024 threads. Wide
    # irreps reuse the normal channel tile and accumulate tile contributions.
    wide_channels = max_channels > 1024
    group_threads = spec_threads if wide_channels else (max_channels + 31) // 32 * 32
    group_u = (
        "blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x"
        if wide_channels
        else "threadIdx.x"
    )
    group_tiles = (
        ", static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1)"
        " / GEN_SPEC_CHANNEL_TILE)"
        if wide_channels
        else ""
    )

    edge_allocation = (
        "torch::zeros_like(y)"
        if wide_channels
        or (
            sum(
                size
                for _, size in {
                    (s["y_offset"], s["d2"]) for s in meta["generated_specs"]
                }
            )
            != meta["in2_dim"]
        )
        else "GEN_GROUPED_Y ? torch::empty_like(y) : torch::zeros_like(y)"
    )

    x_groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for spec in specs:
        key = (spec["x_offset"], spec["d1"], spec["channels"])
        x_groups.setdefault(key, []).append(spec)
    x_group_items = sorted(x_groups.items(), key=lambda item: item[0][0])

    grad_x_cases = []
    for group_idx, ((x_offset, d1, channels), group_specs) in enumerate(x_group_items):
        lines = [f"case {group_idx}: {{"]
        lines.append(f"    if (u >= {channels}) return;")
        lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
        lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
        lines.append("    const int64_t item_dx = int64_t(n) * GEN_IN1_DIM;")
        for i in range(d1):
            lines.append(f"    float acc{i} = 0.0f;")
        for spec_idx, spec in enumerate(group_specs):
            y_offset = spec["y_offset"]
            out_offset = spec["out_offset"]
            weight_offset = spec["weight_offset"]
            weight_u = weight_channel_offset(spec)
            d3 = spec["d3"]
            terms = spec["terms"]
            unique_j = sorted({t[1] for t in terms})
            unique_k = sorted({t[2] for t in terms})
            lines.append(
                f"    const float ww_s{spec_idx} = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});"
            )
            for j in unique_j:
                lines.append(
                    f"    const float y_s{spec_idx}_{j} = __ldg(y + item_y + {y_offset + j});"
                )
            for k in unique_k:
                lines.append(
                    f"    const float go_s{spec_idx}_{k} = __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k});"
                )
            for i_idx, j_idx, k_idx, coeff in terms:
                i = i_idx
                j = j_idx
                k = k_idx
                c = float_literal(coeff)
                lines.append(
                    f"    acc{i} = fmaf({c} * y_s{spec_idx}_{j}, ww_s{spec_idx} * go_s{spec_idx}_{k}, acc{i});"
                )
        for i in range(d1):
            lines.append(f"    dx[item_dx + {x_offset} + u * {d1} + {i}] = acc{i};")
        lines.append("    break;")
        lines.append("}")
        grad_x_cases.append("\n".join(lines))

    if use_uvu_dx_dy_dw_fusion(meta, 1):
        return make_uvu_dx_dy_dw_source(
            meta, 1, need_grad_weight=need_grad_weight, x_cases=grad_x_cases, needs=needs
        )

    yw_cases = []
    if not use_grouped_y:
        for spec_idx, spec in enumerate(specs):
            channels = spec["channels"]
            d1 = spec["d1"]
            d3 = spec["d3"]
            x_offset = spec["x_offset"]
            y_offset = spec["y_offset"]
            out_offset = spec["out_offset"]
            weight_offset = spec["weight_offset"]
            weight_u = weight_channel_offset(spec)
            terms = spec["terms"]
            unique_i = sorted({t[0] for t in terms})
            unique_j = sorted({t[1] for t in terms})
            unique_k = sorted({t[2] for t in terms})
            lines = [f"case {spec_idx}: {{"]
            lines.append(f"    const bool active = u < {channels};")
            lines.append(
                "    const unsigned active_mask = __ballot_sync(0xffffffff, active);"
            )
            lines.append("    const int64_t item_x = int64_t(n) * GEN_IN1_DIM;")
            lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
            lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
            for i in unique_i:
                lines.append(
                    f"    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;"
                )
            if not scope_uniform_y:
                for j in unique_j:
                    lines.extend(
                        f"    const float yv{j} = __ldg(y + item_y + {y_offset + j});\n    float gy{j} = 0.0f;".split(
                            "\n"
                        )
                    )
            for k in unique_k:
                lines.append(
                    f"    const float go{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;"
                )
            if need_grad_weight:
                lines.append("    float gw = 0.0f;")
            lines.extend(
                f"    float ww = 0.0f;\n    if (active) {{\n        ww = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});\n    }}".split(
                    "\n"
                )
            )
            if scope_uniform_y:
                for j in unique_j:
                    lines.extend(
                        f"    {{\n        const float yv = __ldg(y + item_y + {y_offset + j});\n        float gy = 0.0f;".split(
                            "\n"
                        )
                    )
                    for i_idx, j_idx, k_idx, coeff in terms:
                        if j_idx != j:
                            continue
                        i = i_idx
                        k = k_idx
                        c = float_literal(coeff)
                        lines.extend(
                            f"        const float cgo_{i}_{j}_{k} = {c} * go{k};\n        gy = fmaf(cgo_{i}_{j}_{k}, ww * xv{i}, gy);".split(
                                "\n"
                            )
                        )
                        if need_grad_weight:
                            lines.append(
                                f"        gw = fmaf(cgo_{i}_{j}_{k} * xv{i}, yv, gw);"
                            )
                    lines.extend(
                        f"        const float gy_warp_sum = warp_reduce_sum(gy, active_mask);\n        if ((threadIdx.x & 31) == 0) {{\n            atomicAdd(gy_out + item_y + {y_offset + j}, gy_warp_sum);\n        }}\n    }}".split(
                            "\n"
                        )
                    )
            else:
                for i_idx, j_idx, k_idx, coeff in terms:
                    i = i_idx
                    j = j_idx
                    k = k_idx
                    c = float_literal(coeff)
                    lines.extend(
                        f"    const float cgo_{i}_{j}_{k} = {c} * go{k};\n    gy{j} = fmaf(cgo_{i}_{j}_{k}, ww * xv{i}, gy{j});".split(
                            "\n"
                        )
                    )
                    if need_grad_weight:
                        lines.append(
                            f"    gw = fmaf(cgo_{i}_{j}_{k} * xv{i}, yv{j}, gw);"
                        )
                for j in unique_j:
                    if use_y_block_reduce:
                        lines.extend(
                            r"""    const float gy_sum{j} = block_reduce_sum(gy{j});
    if (threadIdx.x == 0) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_sum{j});
    }}""".format(
                                j=j, y_offset_j=y_offset + j
                            ).split("\n")
                        )
                    else:
                        lines.extend(
                            r"""    if (active) {{
        const float gy_warp_sum{j} = warp_reduce_sum(gy{j}, active_mask);
        if ((threadIdx.x & 31) == 0) {{
            atomicAdd(gy_out + item_y + {y_offset_j}, gy_warp_sum{j});
        }}
    }}""".format(
                                j=j, y_offset_j=y_offset + j
                            ).split("\n")
                        )
            lines.append("    if (active) {")
            lines.append("        if (shared_weight) {")
            if need_grad_weight:
                lines.append(
                    f"            atomicAdd(gw_out + {weight_offset} + {weight_u}, gw);"
                )
            lines.append("        } else {")
            if need_grad_weight:
                lines.append(
                    f"            gw_out[int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u}] = gw;"
                )
            lines.append("        }")
            lines.append("    }")
            lines.append("    break;")
            lines.append("}")
            yw_cases.append("\n".join(lines))
    y_groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    if use_grouped_y:
        for spec in specs:
            key = (spec["y_offset"], spec["d2"])
            y_groups.setdefault(key, []).append(spec)
    y_group_items = sorted(y_groups.items(), key=lambda item: item[0][0])
    y_group_cases = []
    for group_idx, ((y_offset, d2), group_specs) in enumerate(y_group_items):
        lines = [f"case {group_idx}: {{"]
        lines.append("    const int64_t item_x = int64_t(n) * GEN_IN1_DIM;")
        lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
        lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
        lines.append("    const int64_t item_w = int64_t(n) * GEN_WEIGHT_NUMEL;")
        weight_base = "0" if weight_mode == "shared" else "item_w"
        for j in range(d2):
            lines.extend(
                f"    const float yv{j} = __ldg(y + item_y + {y_offset + j});\n    float gy{j} = 0.0f;".split(
                    "\n"
                )
            )
        for spec_idx, spec in enumerate(group_specs):
            channels = spec["channels"]
            d1 = spec["d1"]
            d3 = spec["d3"]
            x_offset = spec["x_offset"]
            out_offset = spec["out_offset"]
            weight_offset = spec["weight_offset"]
            weight_u = weight_channel_offset(spec)
            terms = spec["terms"]
            unique_i = sorted({t[0] for t in terms})
            unique_k = sorted({t[2] for t in terms})
            lines.append("    {")
            lines.append(f"        const bool active = u < {channels};")
            for i in unique_i:
                lines.append(
                    f"        const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;"
                )
            for k in unique_k:
                lines.append(
                    f"        const float go{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;"
                )
            lines.append(
                f"        const float ww = active ? __ldg(weight + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;"
            )
            if need_grad_weight and weight_mode != "shared":
                lines.append("        float gw = 0.0f;")
            for i_idx, j_idx, k_idx, coeff in terms:
                i = i_idx
                j = j_idx
                k = k_idx
                c = float_literal(coeff)
                lines.extend(
                    f"        const float cgo_{i}_{j}_{k} = {c} * go{k};\n        gy{j} = fmaf(cgo_{i}_{j}_{k}, ww * xv{i}, gy{j});".split(
                        "\n"
                    )
                )
                if need_grad_weight and weight_mode != "shared":
                    lines.append(
                        f"        gw = fmaf(cgo_{i}_{j}_{k} * xv{i}, yv{j}, gw);"
                    )
            lines.append("        if (active) {")
            if need_grad_weight and weight_mode != "shared":
                lines.append(
                    f"            gw_out[item_w + {weight_offset} + {weight_u}] = gw;"
                )
            lines.append("        }")
            lines.append("    }")
        for j in range(d2):
            lines.extend(
                f"    const float gy_sum{j} = block_reduce_sum(gy{j});\n    if (threadIdx.x == 0) {{".split(
                    "\n"
                )
            )
            lines.append(
                f"        atomicAdd(gy_out + item_y + {y_offset + j}, gy_sum{j});"
                if wide_channels
                else f"        gy_out[item_y + {y_offset + j}] = gy_sum{j};"
            )
            lines.append("    }")
        lines.append("    break;")
        lines.append("}")
        y_group_cases.append("\n".join(lines))

    yw_grouped_kernel = (
        r"""
__global__ __launch_bounds__({group_threads})
void lunatp_uvu_bwd_yw_grouped_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    int64_t num_items) {{
    const Index n = blockIdx.x;
    const int group = blockIdx.y;
    const int u = {group_u};
    if (n >= num_items) return;
    switch (group) {{
{y_group_cases}
    default:
        break;
    }}
}}
""".format(
            group_threads=group_threads,
            group_u=group_u,
            y_group_cases=chr(10).join(y_group_cases),
        )
        if use_grouped_y
        else ""
    )
    group_weight_reduction = (
        r"""        if (shared_weight) {{
            lunatp_uvu_reduce_weight_fp32<1><<<(GEN_WEIGHT_NUMEL + 31) / 32, 256, 0, stream>>>(
                {gw_work_pointer}, nullptr, {gw_pointer}, nullptr, x.size(0));
        }}
""".format(
            gw_work_pointer=gw_work_pointer,
            gw_pointer=gw_pointer,
        )
        if need_grad_weight and weight_mode != "shared"
        else ""
    )

    group_launch = (
        r"""
        auto gw_work = {gw_work_allocation};
        const dim3 group_block({group_threads});
        const dim3 group_grid(static_cast<unsigned int>(x.size(0)), {y_group_items_count}{group_tiles});
        lunatp_uvu_bwd_yw_grouped_fp32<<<group_grid, group_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_work_pointer},
            x.size(0));
{group_weight_reduction}""".format(
            gw_work_allocation=gw_work_allocation,
            group_threads=group_threads,
            y_group_items_count=len(y_group_items),
            group_tiles=group_tiles,
            gw_work_pointer=gw_work_pointer,
            group_weight_reduction=group_weight_reduction,
        )
        if use_grouped_y
        else ""
    )

    y_shared_cases = []
    w_shared_reduce_cases = []
    for spec_idx, spec in enumerate(specs):
        channels = spec["channels"]
        d1 = spec["d1"]
        d3 = spec["d3"]
        x_offset = spec["x_offset"]
        y_offset = spec["y_offset"]
        out_offset = spec["out_offset"]
        weight_offset = spec["weight_offset"]
        weight_u = weight_channel_offset(spec)
        terms = spec["terms"]
        unique_i = sorted({t[0] for t in terms})
        unique_j = sorted({t[1] for t in terms})
        unique_k = sorted({t[2] for t in terms})

        lines = [f"case {spec_idx}: {{"]
        lines.append(f"    const bool active = u < {channels};")
        lines.append(
            "    const unsigned active_mask = __ballot_sync(0xffffffff, active);"
        )
        lines.append("    const int64_t item_x = int64_t(n) * GEN_IN1_DIM;")
        lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
        lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
        for i in unique_i:
            lines.append(
                f"    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;"
            )
        for k in unique_k:
            lines.append(
                f"    const float go{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;"
            )
        lines.append(
            f"    const float ww = active ? __ldg(weight + {weight_offset} + {weight_u}) : 0.0f;"
        )
        for j in unique_j:
            lines.append(f"    float gy{j} = 0.0f;")
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            lines.append(f"    gy{j} = fmaf({c} * go{k}, ww * xv{i}, gy{j});")
        for j in unique_j:
            if use_y_block_reduce:
                lines.extend(
                    r"""    const float gy_sum{j} = block_reduce_sum(gy{j});
    if (threadIdx.x == 0) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_sum{j});
    }}""".format(j=j, y_offset_j=y_offset + j).split(
                        "\n"
                    )
                )
            else:
                lines.extend(
                    r"""    if (active) {{
        const float gy_warp_sum{j} = warp_reduce_sum(gy{j}, active_mask);
        if ((threadIdx.x & 31) == 0) {{
            atomicAdd(gy_out + item_y + {y_offset_j}, gy_warp_sum{j});
        }}
    }}""".format(j=j, y_offset_j=y_offset + j).split("\n")
                )
        lines.append("    break;")
        lines.append("}")
        y_shared_cases.append("\n".join(lines))

        if not need_grad_weight:
            continue

        lines = [f"case {spec_idx}: {{"]
        lines.extend(r"""    if (u >= {channels}) return;
    const int64_t start_n = int64_t(blockIdx.z) * items_per_tile;
    const int64_t tile_end = start_n + int64_t(items_per_tile);
    const int64_t end_n = tile_end < num_items ? tile_end : num_items;
    float acc = 0.0f;
    for (int64_t n = start_n; n < end_n; ++n) {{
        const int64_t item_x = n * GEN_IN1_DIM;
        const int64_t item_y = n * GEN_IN2_DIM;
        const int64_t item_go = n * GEN_OUT_DIM;""".format(channels=channels).split("\n"))
        for i in unique_i:
            lines.append(
                f"        const float xv{i} = __ldg(x + item_x + {x_offset} + u * {d1} + {i});"
            )
        for j in unique_j:
            lines.append(
                f"        const float yv{j} = __ldg(y + item_y + {y_offset + j});"
            )
        for k in unique_k:
            lines.append(
                f"        const float go{k} = __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k});"
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            lines.append(f"        acc = fmaf({c} * go{k} * xv{i}, yv{j}, acc);")
        lines.append("    }")
        lines.append(
            f"    partials[int64_t(blockIdx.z) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u}] = acc;"
        )
        lines.append("    break;")
        lines.append("}")
        w_shared_reduce_cases.append("\n".join(lines))

    shared_weight_launch = (
        r"""        const int items_per_tile = (x.size(0) >= 4096) ? 128 : 64;
        const int64_t num_tiles = (x.size(0) + items_per_tile - 1) / items_per_tile;
        auto partials = num_tiles == 1 ? gw : torch::empty({{num_tiles, GEN_WEIGHT_NUMEL}}, weight.options());
        const dim3 w_grid(
            GEN_NUM_SPECS,
            static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1) / GEN_SPEC_CHANNEL_TILE),
            static_cast<unsigned int>(num_tiles));
        lunatp_uvu_bwd_w_shared_partial_fp32<<<w_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            partials.data_ptr<float>(),
            x.size(0),
            items_per_tile);
        const dim3 reduce_block(256);
        const dim3 reduce_grid(static_cast<unsigned int>((GEN_WEIGHT_NUMEL + 31) / 32));
        if (num_tiles > 1) lunatp_uvu_reduce_weight_fp32<1><<<reduce_grid, reduce_block, 0, stream>>>(
            partials.data_ptr<float>(), nullptr,
            {gw_pointer}, nullptr,
            num_tiles);
""".format(
            gw_pointer=gw_pointer,
        )
        if need_grad_weight
        else ""
    )
    if weight_mode == "shared" and use_grouped_y:
        group_launch += shared_weight_launch

    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

{derivative_stream_source}

namespace {{

constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_OUT_DIM = {out_dim};
constexpr int GEN_WEIGHT_NUMEL = {weight_numel};
constexpr int GEN_MAX_CHANNELS = {max_channels};
constexpr int GEN_NUM_X_GROUPS = {x_group_items_count};
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_X_CHANNEL_TILE = {x_channel_tile};
constexpr int GEN_X_THREADS = {x_threads};
constexpr int GEN_X_MIN_BLOCKS = {x_min_blocks};
constexpr int GEN_SPEC_CHANNEL_TILE = {spec_channel_tile};
constexpr int GEN_SPEC_THREADS = {spec_threads};
constexpr int GEN_SPEC_MIN_BLOCKS = {spec_min_blocks};
constexpr bool GEN_SHARED_WEIGHT_REDUCE = {shared_weight_reduction};
constexpr bool GEN_GROUPED_Y = {grouped_edges};
constexpr bool GEN_UNIFORM_CHANNELS = {uniform_channels};

{make_reduction_source}__global__ __launch_bounds__(GEN_X_THREADS, GEN_X_MIN_BLOCKS)
void lunatp_uvu_bwd_x_fp32(
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ dx,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int group = blockIdx.y;
    const int u = blockIdx.z * GEN_X_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (group) {{
{grad_x_cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_bwd_yw_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int spec = blockIdx.y;
    const int u = blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (spec) {{
{yw_cases}
    default:
        break;
    }}
}}

{yw_grouped_kernel}
__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_bwd_y_shared_fp32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ gy_out,
    int64_t num_items) {{
    const Index n = blockIdx.x;
    const int spec = blockIdx.y;
    const int u = blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (spec) {{
{y_shared_cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_bwd_w_shared_partial_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ grad_out,
    float* __restrict__ partials,
    int64_t num_items,
    int items_per_tile) {{
    const int spec = blockIdx.x;
    const int u = blockIdx.y * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    switch (spec) {{
{w_shared_reduce_cases}
    default:
        break;
    }}
}}


torch::Tensor run_grad_x(torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out, bool shared_weight) {{
    c10::cuda::CUDAGuard guard(grad_out.device());
    auto dx = torch::{input_allocator}({{grad_out.size(0), GEN_IN1_DIM}}, grad_out.options());
    if (grad_out.size(0) == 0) return dx;
    const dim3 block(GEN_X_THREADS);
    const dim3 grid(
        static_cast<unsigned int>(grad_out.size(0)),
        GEN_NUM_X_GROUPS,
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_X_CHANNEL_TILE - 1) / GEN_X_CHANNEL_TILE));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_out.get_device());
    lunatp_uvu_bwd_x_fp32<<<grid, block, 0, stream>>>(
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        dx.data_ptr<float>(),
        grad_out.size(0),
        shared_weight);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}}

std::vector<torch::Tensor> run_uvu_bwd(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out, bool shared_weight) {{
    c10::cuda::CUDAGuard guard(x.device());
    const bool use_shared_weight_reduce = shared_weight && GEN_SHARED_WEIGHT_REDUCE &&
        (!GEN_UNIFORM_CHANNELS || x.size(0) >= 2048);
    auto dx = torch::{input_like_allocator}(x);
    auto gy = {edge_allocation};
    auto gw = {gw_allocation};
    if (x.size(0) == 0) {{
        {clear_weight}
        return {{{outputs}}};
    }}
    const dim3 x_block(GEN_X_THREADS);
    const dim3 spec_block(GEN_SPEC_THREADS);
    const dim3 x_grid(
        static_cast<unsigned int>(x.size(0)),
        GEN_NUM_X_GROUPS,
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_X_CHANNEL_TILE - 1) / GEN_X_CHANNEL_TILE));
    const dim3 spec_grid(
        static_cast<unsigned int>(x.size(0)),
        GEN_NUM_SPECS,
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1) / GEN_SPEC_CHANNEL_TILE));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto& lane = derivative_stream(x.get_device(), stream);
    DerivativeFork fork(lane, stream);
    lunatp_uvu_bwd_x_fp32<<<x_grid, x_block, 0, lane.stream>>>(
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        dx.data_ptr<float>(),
        x.size(0),
        shared_weight);
    if constexpr (GEN_GROUPED_Y) {{
{group_launch}
    }} else if (use_shared_weight_reduce) {{
        lunatp_uvu_bwd_y_shared_fp32<<<spec_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            gy.data_ptr<float>(),
            x.size(0));
{shared_weight_launch}    }} else {{
        lunatp_uvu_bwd_yw_fp32<<<spec_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_pointer},
            x.size(0),
            shared_weight);
    }}
    fork.join();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}

}}  // namespace

torch::Tensor lunatp_uvu_bwd_x_external_fp32(torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out) {{
    return run_grad_x(y, weight, grad_out, false);
}}

torch::Tensor lunatp_uvu_bwd_x_shared_fp32(torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out) {{
    return run_grad_x(y, weight, grad_out, true);
}}

std::vector<torch::Tensor> lunatp_uvu_bwd_external_fp32(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out) {{
    return run_uvu_bwd(x, y, weight, grad_out, false);
}}

std::vector<torch::Tensor> lunatp_uvu_bwd_shared_fp32(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor grad_out) {{
    return run_uvu_bwd(x, y, weight, grad_out, true);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    // Resolve this module before CUDA instrumentation queries its first launch.
    cudaFuncAttributes attributes;
    const auto status = cudaFuncGetAttributes(&attributes, lunatp_uvu_bwd_x_fp32);
    TORCH_CHECK(status == cudaSuccess, "UVU first kernel initialization: ", cudaGetErrorString(status));
    m.def("lunatp_uvu_bwd_x_external_fp32", &lunatp_uvu_bwd_x_external_fp32, "UVU grad_x external weights (CUDA)");
    m.def("lunatp_uvu_bwd_x_shared_fp32", &lunatp_uvu_bwd_x_shared_fp32, "UVU grad_x shared weights (CUDA)");
    m.def("lunatp_uvu_bwd_external_fp32", &lunatp_uvu_bwd_external_fp32, "UVU backward external weights (CUDA)");
    m.def("lunatp_uvu_bwd_shared_fp32", &lunatp_uvu_bwd_shared_fp32, "UVU backward shared weights (CUDA)");
}}
""".format(
        derivative_stream_source=DERIVATIVE_STREAM_SOURCE,
        in1_dim=in1_dim,
        in2_dim=in2_dim,
        out_dim=out_dim,
        weight_numel=weight_numel,
        max_channels=max_channels,
        x_group_items_count=len(x_group_items),
        specs_count=len(specs),
        x_channel_tile=x_channel_tile,
        x_threads=x_threads,
        x_min_blocks=x_min_blocks,
        spec_channel_tile=spec_channel_tile,
        spec_threads=spec_threads,
        spec_min_blocks=spec_min_blocks,
        shared_weight_reduction="true" if enable_shared_weight_reduce else "false",
        grouped_edges="true" if use_grouped_y else "false",
        uniform_channels="true" if len(channel_set) == 1 else "false",
        make_reduction_source=make_reduction_source(meta),
        grad_x_cases=chr(10).join(grad_x_cases),
        yw_cases=chr(10).join(yw_cases),
        yw_grouped_kernel=yw_grouped_kernel,
        y_shared_cases=chr(10).join(y_shared_cases),
        w_shared_reduce_cases=chr(10).join(w_shared_reduce_cases),
        input_allocator="zeros"
        if (
            sum(
                size
                for _, size in {
                    (s["x_offset"], s["channels"] * s["d1"])
                    for s in meta["generated_specs"]
                }
            )
            != meta["in1_dim"]
        )
        else "empty",
        input_like_allocator="zeros_like"
        if (
            sum(
                size
                for _, size in {
                    (s["x_offset"], s["channels"] * s["d1"])
                    for s in meta["generated_specs"]
                }
            )
            != meta["in1_dim"]
        )
        else "empty_like",
        edge_allocation=edge_allocation,
        gw_allocation=gw_allocation,
        clear_weight="gw.zero_();" if need_grad_weight else "",
        outputs=", ".join(outputs),
        group_launch=group_launch,
        shared_weight_launch=shared_weight_launch,
        gw_pointer=gw_pointer,
    )

    source = make_grid_source(source, max(x_channel_tile, spec_channel_tile))
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def use_uvu_dx_dy_dw_fusion(meta, order):
    """Bound the live working set; retain tiled schedules for wide/high degrees."""
    coords = {(s["x_offset"], s["d1"], s["channels"]) for s in meta["generated_specs"]}
    return (
        meta.get("dtype", torch.float32) is torch.float32
        and meta["max_channels"] <= 256
        # Higher-order direction and accumulator arrays need a smaller budget.
        and sum(d for _, d, _ in coords) <= (16 if order == 1 else 9)
        and meta["in2_dim"] <= (16 if order == 1 else 9)
    )


def make_uvu_dx_dy_dw_source(
    meta,
    order,
    *,
    need_grad_weight=True,
    need_grad_rw=True,
    x_cases=None,
    weight_reduce_columns=32,
    needs=None,
):
    program = PROGRAMS[order]
    specs = meta["generated_specs"]
    edge_threads = (meta["max_channels"] + 31) // 32 * 32
    edges_per_block = max(1, 128 // edge_threads)
    threads = edge_threads * edges_per_block
    warps = edge_threads // 32
    inputs = INPUTS[: (4, 7, 11)[order - 1]]
    outputs = tuple(
        k
        for k in program["outputs"]
        if (k != "gw" or need_grad_weight) and (k != "grw" or need_grad_rw)
        and (needs is None or needs[program["outputs"].index(k)])
    )
    xouts = {k: v for k, v in program["x"].items() if k in outputs}
    youts = {k: v for k, v in program["y"].items() if k in outputs}
    wouts = {
        k: v for k, v in program["w"].items()
        if k in outputs and meta["weight_mode"] != "shared"
    }
    shared_weight = make_shared_weight_partial(meta, order, outputs)
    forward = program["forward"] if "gg" in outputs else ()
    needed_z = {z for pairs in (*xouts.values(), *youts.values()) for _, z in pairs}
    zforms = {k: v for k, v in program["z"].items() if k in needed_z}
    needed_q = {q for pairs in wouts.values() for q, _ in pairs} | {
        q for q, _ in forward
    }
    qforms = {k: v for k, v in program["cg"].items() if k in needed_q}
    xroles = {x for pairs in youts.values() for x, _ in pairs} | {
        x for pairs in qforms.values() for x, _ in pairs
    }
    yroles = {y for pairs in xouts.values() for y, _ in pairs} | {
        y for pairs in qforms.values() for _, y in pairs
    }
    xcoords = [
        (off, dim, c, i)
        for off, dim, c in sorted(
            {(s["x_offset"], s["d1"], s["channels"]) for s in specs}
        )
        for i in range(dim)
    ]
    xids = {coord: i for i, coord in enumerate(xcoords)}
    ycoords = sorted({s["y_offset"] + j for s in specs for _, j, _, _ in s["terms"]})
    yids = {coord: i for i, coord in enumerate(ycoords)}
    lines = []
    for idx, (off, dim, c, i) in enumerate(xcoords):
        for role in sorted(xroles):
            lines.append(
                f"const float {role}_{idx} = u < {c} ? __ldg({role} + item_x + {off} + u * {dim} + {i}) : 0.0f;"
            )
        for out in xouts:
            lines.append(f"float {out}_{idx} = 0.0f;")
    for idx, coord in enumerate(ycoords):
        for role in sorted(yroles):
            lines.append(
                f"const float {role}_{idx} = __ldg({role} + item_y + {coord});"
            )
        for out in youts:
            lines.append(f"float {out}_{idx} = 0.0f;")

    weight_roles = {w for pairs in zforms.values() for w, _ in pairs} | {
        w for _, w in forward
    }
    grad_roles = {g for pairs in zforms.values() for _, g in pairs} | {
        g for pairs in wouts.values() for _, g in pairs
    }
    for s in specs:
        c = s["channels"]
        wi = f"""{s["weight_offset"]} + {weight_channel_offset(s)}"""
        lines += [
            "{",
            f"const bool active = edge_active && u < {c};",
            f"const int64_t wi = weight_base + {wi};",
        ]
        for role in sorted(weight_roles):
            lines.append(f"const float {role}v = active ? __ldg({role} + wi) : 0.0f;")
        for out in wouts:
            lines.append(f"float {out}v = 0.0f;")
        by_k = defaultdict(list)
        for i, j, k, coef in s["terms"]:
            by_k[k].append((i, j, coef))
        for k, terms in sorted(by_k.items()):
            go_index = f"""item_g + {s["out_offset"]} + u * {s["d3"]} + {k}"""
            lines.append("{")
            for role in sorted(grad_roles):
                lines.append(
                    f"const float {role}v = active ? __ldg({role} + {go_index}) : 0.0f;"
                )
            for z, pairs in zforms.items():
                expression = " + ".join(f"{w}v * {g}v" for w, g in pairs)
                lines.append(f"const float {z} = {expression};")
            if order == 1 and not need_grad_weight and xouts and youts:
                # With no Q consumer, coefficient*Z feeds both gradients without
                # holding feature contractions live across output components.
                for i, j, coef in terms:
                    xi = xids[(s["x_offset"], s["d1"], c, i)]
                    yi = yids[s["y_offset"] + j]
                    lines += r"""{{
const float cz = {coefficient} * z0;
gx_{xi} = fmaf(cz, y_{yi}, gx_{xi});
gy_{yi} = fmaf(cz, x_{xi}, gy_{yi});
}}""".format(
                        coefficient=float_literal(coef), xi=xi, yi=yi
                    ).split("\n")
                lines.append("}")
                continue
            by_i, by_j = defaultdict(list), defaultdict(list)
            for i, j, coef in terms:
                by_i[i].append((j, coef))
                by_j[j].append((i, coef))
            for i, entries in sorted(by_i.items()):
                idx = xids[(s["x_offset"], s["d1"], c, i)]
                lines.append("{")
                for role in sorted({y for pairs in xouts.values() for y, _ in pairs}):
                    expr = " + ".join(
                        f"""({float_literal(coef)} * {role}_{yids[s["y_offset"] + j]})"""
                        for j, coef in entries
                    )
                    lines.append(f"const float a{role}_{i} = {expr};")
                for out, pairs in xouts.items():
                    for role, z in pairs:
                        lines.append(
                            f"{out}_{idx} = fmaf(a{role}_{i}, {z}, {out}_{idx});"
                        )
                lines.append("}")
            for q in qforms:
                lines.append(f"float {q} = 0.0f;")
            for j, entries in sorted(by_j.items()):
                idx = yids[s["y_offset"] + j]
                lines.append("{")
                for role in sorted(xroles):
                    expr = " + ".join(
                        f"""({float_literal(coef)} * {role}_{xids[s["x_offset"], s["d1"], c, i]})"""
                        for i, coef in entries
                    )
                    lines.append(f"const float a{role} = {expr};")
                for out, pairs in youts.items():
                    for role, z in pairs:
                        lines.append(f"{out}_{idx} = fmaf(a{role}, {z}, {out}_{idx});")
                for q, pairs in qforms.items():
                    for x, y in pairs:
                        lines.append(f"{q} = fmaf(a{x}, {y}_{idx}, {q});")
                lines.append("}")
            for out, pairs in wouts.items():
                for q, g in pairs:
                    lines.append(f"{out}v = fmaf({q}, {g}v, {out}v);")
            if forward:
                expr = " + ".join(f"{q} * {w}v" for q, w in forward)
                assign = "+=" if meta["zero_output"] else "="
                lines.append(f"if (active) gg_out[{go_index}] {assign} {expr};")
            lines.append("}")
        for out in wouts:
            lines.append(f"if (active) {out}_out[item_w + {wi}] = {out}v;")
        lines.append("}")
    for idx, (off, dim, c, i) in enumerate(xcoords):
        for out in xouts:
            lines.append(
                f"if (edge_active && u < {c}) {out}_out[item_x + {off} + u * {dim} + {i}] = {out}_{idx};"
            )
    for yi, out in enumerate(youts):
        for idx in range(len(ycoords)):
            lines += f"const float {out}_sum{idx} = warp_reduce_sum({out}_{idx}, 0xffffffff);\nif (lane == 0) dy_sums[{yi}][edge_slot][warp][{idx}] = {out}_sum{idx};".split(
                "\n"
            )
    lines.append("__syncthreads();")
    for yi, out in enumerate(youts):
        lines += r"""for (int j=u; j<{edge_components}; j+={threads}) {{
float total = 0.0f;
#pragma unroll
for (int v=0; v<{warps}; ++v) total += dy_sums[{yi}][edge_slot][v][j];
if (edge_active) {out}_out[item_y + y_coordinates[j]] = total;
}}""".format(
            edge_components=len(ycoords),
            threads=edge_threads,
            warps=warps,
            yi=yi,
            out=out,
        ).split("\n")
    phase = ("bwd", "double_bwd", "triple_bwd")[order - 1]
    gradient_outputs = "_".join(f"d{out[1:]}" for out in outputs)
    kernel = f"lunatp_uvu_{phase}_{gradient_outputs}_fp32"
    if order == 1 and outputs == ("gx",):
        kernel = f"lunatp_uvu_{phase}_{gradient_outputs}_fused_fp32"
    run_name = f"run_uvu_{phase}_{gradient_outputs}"
    ptrs = ",\n".join(
        [f"const float* __restrict__ {v}" for v in inputs]
        + [f"float* __restrict__ {v}_out" for v in outputs]
    )
    decls = ", ".join(f"torch::Tensor {v}" for v in inputs)
    alloc = []
    for out in outputs:
        role = OUTPUT_SHAPES[out]
        zeros = (
            (
                sum(
                    size
                    for _, size in {
                        (s["x_offset"], s["channels"] * s["d1"])
                        for s in meta["generated_specs"]
                    }
                )
                != meta["in1_dim"]
            )
            if role == "x"
            else (
                sum(
                    size
                    for _, size in {
                        (s["y_offset"], s["d2"]) for s in meta["generated_specs"]
                    }
                )
                != meta["in2_dim"]
            )
            if role == "y"
            else meta["zero_output"]
            if role == "g"
            else False
        )
        alloc.append(
            f"""auto {out} = torch::{("zeros_like" if zeros else "empty_like")}({role});"""
        )
    args = ", ".join(
        [f"{v}.data_ptr<float>()" for v in inputs]
        + [f"{v}.data_ptr<float>()" for v in outputs]
        + ["shared_weight", "x.size(0)"]
    )
    reduce = shared_weight["launch"]
    wrappers = []
    for mode in ("shared", "external"):
        fname = f"lunatp_uvu_{phase}_{mode}_fp32"
        wrappers.append(
            f"""std::vector<torch::Tensor> {fname}({decls}) {{ return {run_name}({", ".join(inputs)}, {("true" if mode == "shared" else "false")}); }}"""
        )
    xonly = ""
    if order == 1 and x_cases is not None:
        # Preserve the established source-group/channel-tiled x-only schedule.
        tile, x_threads, min_blocks = get_uvu_launch_config(meta["max_channels"])
        groups = len({(s["x_offset"], s["d1"], s["channels"]) for s in specs})
        xonly = r"""
__global__ __launch_bounds__({x_threads},{min_blocks}) void lunatp_uvu_bwd_dx_fp32(const float* y,const float* weight,const float* grad_out,float* dx,bool shared_weight) {{
const Index n=blockIdx.x;
const int u=blockIdx.z*{tile}+threadIdx.x;
switch(blockIdx.y) {{ {x_cases} }}
}}
torch::Tensor run_dx(torch::Tensor y,torch::Tensor w,torch::Tensor g,bool shared_weight) {{
c10::cuda::CUDAGuard guard(g.device());
auto gx=torch::{input_allocator}({{g.size(0),GEN_IN1_DIM}},g.options());
if(g.size(0)) {{ lunatp_uvu_bwd_dx_fp32<<<dim3(static_cast<unsigned int>(g.size(0)),{groups},{channel_tiles}),{x_threads},0,at::cuda::getCurrentCUDAStream(g.get_device())>>>(y.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),gx.data_ptr<float>(),shared_weight); C10_CUDA_KERNEL_LAUNCH_CHECK(); }}
return gx;
}}
""".format(
            x_threads=x_threads,
            min_blocks=min_blocks,
            tile=tile,
            x_cases=chr(10).join(x_cases),
            input_allocator="zeros"
            if (
                sum(
                    size
                    for _, size in {
                        (s["x_offset"], s["channels"] * s["d1"])
                        for s in meta["generated_specs"]
                    }
                )
                != meta["in1_dim"]
            )
            else "empty",
            groups=groups,
            channel_tiles=(meta["max_channels"] + tile - 1) // tile,
        )
        for mode in ("shared", "external"):
            wrappers.append(
                f"""torch::Tensor lunatp_uvu_bwd_x_{mode}_fp32(torch::Tensor y,torch::Tensor w,torch::Tensor g) {{ return run_dx(y,w,g,{("true" if mode == "shared" else "false")}); }}"""
            )
    funcs = [f"lunatp_uvu_{phase}_{mode}_fp32" for mode in ("shared", "external")]
    if order == 1 and x_cases is not None:
        funcs += [f"lunatp_uvu_bwd_x_{mode}_fp32" for mode in ("shared", "external")]
    binds = "\n".join(f'm.def("{fn}",&{fn});' for fn in funcs)
    weight_outputs = tuple(out for out in outputs if out in program["w"])
    feature_outputs = tuple(out for out in outputs if out not in weight_outputs)
    main_launch = (
        f"{kernel}<<<static_cast<unsigned int>((x.size(0)+{edges_per_block-1})/{edges_per_block}),{threads},0,stream>>>({args});"
        if feature_outputs or meta["weight_mode"] != "shared" else ""
    )
    reduction_source = make_reduction_source(
        meta, columns=weight_reduce_columns if wouts else 32
    )
    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
namespace {{
constexpr int GEN_IN1_DIM={in1_dim};
constexpr int GEN_IN2_DIM={in2_dim};
constexpr int GEN_OUT_DIM={out_dim};
constexpr int GEN_WEIGHT_NUMEL={weight_numel};
{reduction_source}{weight_partial_source}
__global__ __launch_bounds__({threads}) void {kernel}({ptrs},bool shared_weight,int64_t num_items) {{
const int edge_slot=threadIdx.x/{edge_threads};
const int u=threadIdx.x%{edge_threads},lane=u&31,warp=u>>5;
const int64_t item=int64_t(blockIdx.x)*{edges_per_block}+edge_slot;
const bool edge_active=item<num_items;
const int64_t n=min(item,num_items-1);
const int64_t item_x=n*GEN_IN1_DIM,item_y=n*GEN_IN2_DIM,item_g=n*GEN_OUT_DIM,item_w=n*GEN_WEIGHT_NUMEL;
const int64_t weight_base=shared_weight?0:item_w;
const int y_coordinates[{ycoords_count}]={{{map}}};
__shared__ float dy_sums[{youts_count}][{edges_per_block}][{warps}][{ycoords_count}];
{lines}
}}
std::vector<torch::Tensor> {run_name}({decls},bool shared_weight) {{
c10::cuda::CUDAGuard guard(x.device());
{alloc}
if(x.size(0)==0) {{ {contractions} return {{{outputs}}}; }}
cudaStream_t stream=at::cuda::getCurrentCUDAStream(x.get_device());
{main_launch}
{reduce}
C10_CUDA_KERNEL_LAUNCH_CHECK();
return {{{outputs}}};
}}
{xonly}
}} // namespace
{wrappers}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
cudaFuncAttributes attributes;
const auto status=cudaFuncGetAttributes(&attributes,{kernel});
TORCH_CHECK(status==cudaSuccess,"UVU gradient kernel initialization: ",cudaGetErrorString(status));
{binds}
}}
""".format(
        in1_dim=meta["in1_dim"],
        in2_dim=meta["in2_dim"],
        out_dim=meta["out_dim"],
        weight_numel=meta["weight_numel"],
        reduction_source=reduction_source,
        weight_partial_source=shared_weight["source"],
        threads=threads,
        edge_threads=edge_threads,
        edges_per_block=edges_per_block,
        kernel=kernel,
        run_name=run_name,
        ptrs=ptrs,
        ycoords_count=len(ycoords),
        map=", ".join(map(str, ycoords)),
        youts_count=max(1, len(youts)),
        warps=warps,
        lines=chr(10).join(lines),
        decls=decls,
        alloc=chr(10).join(alloc),
        contractions=" ".join(f"{v}.zero_();" for v in weight_outputs),
        outputs=", ".join(outputs),
        args=args,
        main_launch=main_launch,
        reduce=reduce,
        xonly=xonly,
        wrappers=chr(10).join(wrappers),
        binds=binds,
    )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def make_shared_weight_partial(meta, order, outputs):
    """Contract shared weights directly into block-local edge partials."""
    program = PROGRAMS[order]
    weights = {name: pairs for name, pairs in program["w"].items() if name in outputs}
    if not weights or meta["weight_mode"] != "shared":
        return {"source": "", "launch": ""}
    qforms = {
        name: pairs for name, pairs in program["cg"].items()
        if any(name == q for pairs in weights.values() for q, _ in pairs)
    }
    xroles = {x for pairs in qforms.values() for x, _ in pairs}
    yroles = {y for pairs in qforms.values() for _, y in pairs}
    groles = {g for pairs in weights.values() for _, g in pairs}
    inputs = tuple(role for role in INPUTS if role in xroles | yroles | groles)
    name = f'lunatp_uvu_{("bwd", "double_bwd", "triple_bwd")[order - 1]}_weight_partial_fp32'
    cases = []
    for index, spec in enumerate(meta["generated_specs"]):
        lines = [f"case {index}: {{", f"    const bool active = u < {spec['channels']};"]
        for output in weights:
            lines.append(f"    float {output}_sum = 0.0f;")
        lines.append("    if (active) for (int64_t n = begin + worker; n < end; n += 8) {")
        for role in sorted(xroles):
            for i in sorted({i for i, _, _, _ in spec["terms"]}):
                lines.append(f"        const float {role}_{i} = __ldg({role} + n * GEN_IN1_DIM + {spec['x_offset']} + u * {spec['d1']} + {i});")
        for role in sorted(yroles):
            for j in sorted({j for _, j, _, _ in spec["terms"]}):
                lines.append(f"        const float {role}_{j} = __ldg({role} + n * GEN_IN2_DIM + {spec['y_offset'] + j});")
        for output in weights:
            lines.append(f"        float {output}_edge = 0.0f;")
        by_k = defaultdict(list)
        for i, j, k, coefficient in spec["terms"]:
            by_k[k].append((i, j, coefficient))
        for k, terms in sorted(by_k.items()):
            lines.append("        {")
            for role in sorted(groles):
                lines.append(f"            const float {role}v = __ldg({role} + n * GEN_OUT_DIM + {spec['out_offset']} + u * {spec['d3']} + {k});")
            for q in qforms:
                lines.append(f"            float {q} = 0.0f;")
            by_j = defaultdict(list)
            for i, j, coefficient in terms:
                by_j[j].append((i, coefficient))
            for j, entries in sorted(by_j.items()):
                lines.append("            {")
                for role in sorted(xroles):
                    expression = " + ".join(f"({float_literal(coefficient)} * {role}_{i})" for i, coefficient in entries)
                    lines.append(f"                const float a{role} = {expression};")
                for q, pairs in qforms.items():
                    for x, y in pairs:
                        lines.append(f"                {q} = fmaf(a{x}, {y}_{j}, {q});")
                lines.append("            }")
            for output, pairs in weights.items():
                for q, g in pairs:
                    lines.append(f"            {output}_edge = fmaf({q}, {g}v, {output}_edge);")
            lines.append("        }")
        for output in weights:
            lines.append(f"        {output}_sum += {output}_edge;")
        lines.append("    }")
        for slot, output in enumerate(weights):
            lines.append(f"    sums[{slot}][worker][lane] = {output}_sum;")
        lines.append("    __syncthreads();")
        lines.append("    if (worker == 0 && active) {")
        for slot, output in enumerate(weights):
            lines.append(f"        float {output}_total = 0.0f;")
            lines.append(f"        for (int row = 0; row < 8; ++row) {output}_total += sums[{slot}][row][lane];")
            lines.append(f"        {output}_out[int64_t(blockIdx.x) * GEN_WEIGHT_NUMEL + {spec['weight_offset']} + {weight_channel_offset(spec)}] = {output}_total;")
        lines.extend(("    }", "    break;", "}"))
        cases.append("\n".join(lines))
    pointers = ", ".join([f"const float* __restrict__ {role}" for role in inputs]
                         + [f"float* __restrict__ {output}_out" for output in weights])
    source = f'''
__global__ __launch_bounds__(256) void {name}({pointers}, int64_t num_items) {{
    const int lane = threadIdx.x & 31;
    const int worker = threadIdx.x >> 5;
    const int u = blockIdx.z * 32 + lane;
    const int64_t begin = int64_t(blockIdx.x) * {SHARED_WEIGHT_ROWS};
    const int64_t end = min(num_items, begin + {SHARED_WEIGHT_ROWS});
    __shared__ float sums[{len(weights)}][8][32];
    switch (blockIdx.y) {{
{chr(10).join(cases)}
    }}
}}
'''
    allocations = "\n".join(
        f"auto {output}_partial = weight_tiles == 1 ? {output} : torch::empty({{weight_tiles, GEN_WEIGHT_NUMEL}}, w.options());"
        for output in weights
    )
    arguments = ", ".join([f"{role}.data_ptr<float>()" for role in inputs]
                          + [f"{output}_partial.data_ptr<float>()" for output in weights]
                          + ["x.size(0)"])
    partials = [f"{output}_partial.data_ptr<float>()" for output in weights]
    finals = [f"{output}.data_ptr<float>()" for output in weights]
    if len(weights) == 1:
        partials.append("nullptr")
        finals.append("nullptr")
    launch = f'''
const int64_t weight_tiles = (x.size(0) + {SHARED_WEIGHT_ROWS - 1}) / {SHARED_WEIGHT_ROWS};
{allocations}
{name}<<<dim3(static_cast<unsigned int>(weight_tiles), {len(meta["generated_specs"])}, {(meta["max_channels"] + 31) // 32}), 256, 0, stream>>>({arguments});
if (weight_tiles > 1) launch_weight_reduction<{len(weights)}>({", ".join(partials + finals)}, weight_tiles, w.options(), stream);
'''
    return {"source": source, "launch": launch}


def make_reduction_source(meta: Mapping[str, object], *, columns: int = 32) -> str:
    """Keep whole-warp reductions unchanged and mask partial-warp sources."""
    source = (r"""__device__ __forceinline__ float block_reduce_sum(float v) {
    static __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    v = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
    }
    __syncthreads();
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
"""
        + (r"""        v += __shfl_down_sync(mask, v, offset);""" if all(spec["channels"] % 32 == 0 for spec in meta["generated_specs"]) else r"""        const float other = __shfl_down_sync(mask, v, offset);
        const int source_lane = (threadIdx.x & 31) + offset;
        if (source_lane < 32 && (mask & (1u << source_lane))) v += other;""")
        + r"""
    }
    return v;
}

// Reduce row-major weight contributions; one or two gradients share the launch.
// The generated operator defines GEN_WEIGHT_NUMEL before this source fragment.
template<int N_GRADS, int ROW_TILE = 0>
__global__ __launch_bounds__(256)
void lunatp_uvu_reduce_weight_fp32(
    const float* __restrict__ partial0,
    const float* __restrict__ partial1,
    float* __restrict__ output0,
    float* __restrict__ output1,
    int64_t rows) {
    const int lane = threadIdx.x & %(lane_mask)d;
    const int %(row_name)s = threadIdx.x >> %(column_shift)d;
    constexpr int64_t column_blocks = (int64_t(GEN_WEIGHT_NUMEL) + %(columns)d - 1) / %(columns)d;
    const int64_t tile = ROW_TILE ? blockIdx.x / column_blocks : 0;
    const int column = (ROW_TILE ? blockIdx.x %% column_blocks : blockIdx.x) * %(columns)d + lane;
    const int64_t begin = tile * ROW_TILE;
    const int64_t end = ROW_TILE && begin + ROW_TILE < rows ? begin + ROW_TILE : rows;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (column < GEN_WEIGHT_NUMEL) {
        for (int64_t row = begin + %(row_name)s; row < end; row += %(row_step)d) {
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t r = row + part * %(row_lanes)d;
                if (r < end) {
                    const int64_t offset = r * GEN_WEIGHT_NUMEL + column;
                    acc0[part] += __ldg(partial0 + offset);
                    if constexpr (N_GRADS == 2) acc1[part] += __ldg(partial1 + offset);
                }
            }
        }
    }
    __shared__ float sums[N_GRADS][256];
    sums[0][threadIdx.x] = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
    if constexpr (N_GRADS == 2)
        sums[1][threadIdx.x] = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
    __syncthreads();
    if (%(row_name)s == 0 && column < GEN_WEIGHT_NUMEL) {
        float value0 = 0.0f;
        float value1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < %(row_lanes)d; ++w) {
            value0 += sums[0][w * %(columns)d + lane];
            if constexpr (N_GRADS == 2) value1 += sums[1][w * %(columns)d + lane];
        }
        output0[tile * GEN_WEIGHT_NUMEL + column] = value0;
        if constexpr (N_GRADS == 2) output1[tile * GEN_WEIGHT_NUMEL + column] = value1;
    }
}

template<int N_GRADS>
void launch_weight_reduction(
    const float* partial0, const float* partial1,
    float* output0, float* output1,
    int64_t rows, const torch::TensorOptions& options, cudaStream_t stream) {
    constexpr int64_t column_blocks = (int64_t(GEN_WEIGHT_NUMEL) + %(columns)d - 1) / %(columns)d;
    // Narrow weights need independent row tiles before the final column sum.
    constexpr int ROW_TILE = 256;
    const int64_t tiles = (rows + ROW_TILE - 1) / ROW_TILE;
    if (tiles == 1) {
        lunatp_uvu_reduce_weight_fp32<N_GRADS, ROW_TILE><<<column_blocks, 256, 0, stream>>>(
            partial0, partial1, output0, output1, rows);
        return;
    }
    auto intermediate = torch::empty({int64_t(N_GRADS), tiles, int64_t(GEN_WEIGHT_NUMEL)}, options);
    float* work0 = intermediate.data_ptr<float>();
    float* work1 = N_GRADS == 2 ? work0 + tiles * GEN_WEIGHT_NUMEL : nullptr;
    lunatp_uvu_reduce_weight_fp32<N_GRADS, ROW_TILE><<<column_blocks * tiles, 256, 0, stream>>>(
        partial0, partial1, work0, work1, rows);
    lunatp_uvu_reduce_weight_fp32<N_GRADS><<<column_blocks, 256, 0, stream>>>(
        work0, work1, output0, output1, tiles);
}

""") % dict(
        columns=columns,
        lane_mask=columns - 1,
        column_shift=columns.bit_length() - 1,
        row_name="warp" if columns == 32 else "row_lane",
        row_lanes=256 // columns,
        row_step=1024 // columns,
    )
    return source
