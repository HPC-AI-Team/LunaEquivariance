"""UVU double-backward CUDA templates."""

from collections.abc import Mapping
import torch
from ...utils.cg import weight_channel_offset
from ....utils.precision import cuda_source_for_dtype, float_literal
from ...utils.runtime import DERIVATIVE_STREAM_SOURCE
from .utils import make_grid_source, get_uvu_launch_config
from .backward import (
    use_uvu_dx_dy_dw_fusion,
    make_uvu_dx_dy_dw_source,
    make_reduction_source,
)


# Instruction math and traversals used by the kernels above.
def make_uvu_double_backward_source(
    meta: Mapping[str, object], *, need_grad_weight: bool = True, needs=None
) -> str:
    if use_uvu_dx_dy_dw_fusion(meta, 2):
        return make_uvu_dx_dy_dw_source(meta, 2, need_grad_weight=need_grad_weight, needs=needs)
    outputs = ["gx", "gy"]
    if need_grad_weight:
        outputs.append("gw")
    outputs.append("gg")
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
    use_y_warp_partial = weight_mode == "external" and channel_set == {128}
    enable_shared_weight_reduce = weight_mode == "shared" and (
        (len(channel_set) > 1 and max_channels > 128)
        or (len(channel_set) == 1 and (max_channels <= 64 or max_channels >= 512))
    )
    x_channel_tile, x_threads, x_min_blocks = get_uvu_launch_config(max_channels)
    if len(channel_set) == 1 and 256 < max_channels <= 512:
        spec_channel_tile, spec_threads, spec_min_blocks = 128, 128, 4
    elif enable_shared_weight_reduce:
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
    gx_cases = []
    for group_idx, ((x_offset, d1, channels), group_specs) in enumerate(x_group_items):
        lines = [f"case {group_idx}: {{"]
        lines.append(f"    if (u >= {channels}) return;")
        lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
        lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
        lines.append("    const int64_t item_gx = int64_t(n) * GEN_IN1_DIM;")
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
            lines.extend(
                f"    const float ww_s{spec_idx} = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});\n    const float rw_s{spec_idx} = shared_weight ? __ldg(rw + {weight_offset} + {weight_u}) : __ldg(rw + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});".split(
                    "\n"
                )
            )
            for j in unique_j:
                lines.extend(
                    f"    const float y_s{spec_idx}_{j} = __ldg(y + item_y + {y_offset + j});\n    const float ry_s{spec_idx}_{j} = __ldg(ry + item_y + {y_offset + j});".split(
                        "\n"
                    )
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
                    f"    acc{i} = fmaf({c} * go_s{spec_idx}_{k}, ww_s{spec_idx} * ry_s{spec_idx}_{j} + y_s{spec_idx}_{j} * rw_s{spec_idx}, acc{i});"
                )
        for i in range(d1):
            lines.append(f"    gx_out[item_gx + {x_offset} + u * {d1} + {i}] = acc{i};")
        lines.append("    break;")
        lines.append("}")
        gx_cases.append("\n".join(lines))

    cases = []
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
                lines.extend(
                    f"    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;\n    const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;".split(
                        "\n"
                    )
                )
            for j in unique_j:
                lines.extend(
                    r"""    const float yv{j} = __ldg(y + item_y + {y_offset_j});
    const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
    float gy{j} = 0.0f;""".format(
                        j=j, y_offset_j=y_offset + j
                    ).split("\n")
                )
            for k in unique_k:
                lines.extend(
                    f"    const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;\n    float gg{k} = 0.0f;".split(
                        "\n"
                    )
                )
            lines.extend(
                r"""    float ww = 0.0f;
    float rwv = 0.0f;
    if (active) {{
        ww = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});
        rwv = shared_weight ? __ldg(rw + {weight_offset} + {weight_u}) : __ldg(rw + int64_t(n) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u});
    }}""".format(
                    weight_offset=weight_offset,
                    weight_u=weight_u,
                ).split("\n")
            )
            if need_grad_weight:
                lines.append("    float gw = 0.0f;")
            for i_idx, j_idx, k_idx, coeff in terms:
                i = i_idx
                j = j_idx
                k = k_idx
                c = float_literal(coeff)
                lines.extend(
                    f"    const float cgo_{i}_{j}_{k} = {c} * gov{k};\n    gy{j} = fmaf(cgo_{i}_{j}_{k}, ww * rxv{i} + xv{i} * rwv, gy{j});".split(
                        "\n"
                    )
                )
                if need_grad_weight:
                    lines.append(
                        f"    gw = fmaf(cgo_{i}_{j}_{k}, yv{j} * rxv{i} + xv{i} * ryv{j}, gw);"
                    )
                lines.append(
                    f"    gg{k} = fmaf({c}, ww * (yv{j} * rxv{i} + xv{i} * ryv{j}) + xv{i} * yv{j} * rwv, gg{k});"
                )
            for j in unique_j:
                if use_y_warp_partial:
                    lines.extend(
                        f"    const float gy_warp_sum{j} = warp_reduce_sum(gy{j}, active_mask);\n    if ((threadIdx.x & 31) == 0) {{\n        atomicAdd(gy_out + item_y + {y_offset + j}, gy_warp_sum{j});\n    }}".split(
                            "\n"
                        )
                    )
                elif use_y_block_reduce:
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
            for k in unique_k:
                lines.append("    if (active) {")
                lines.append(
                    f"    atomicAdd(gg_out + item_go + {out_offset} + u * {d3} + {k}, gg{k});"
                    if spec["accumulate_output"]
                    else f"    gg_out[item_go + {out_offset} + u * {d3} + {k}] = gg{k};"
                )
                lines.append("    }")
            lines.append("    break;")
            lines.append("}")
            cases.append("\n".join(lines))

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
                r"""    const float yv{j} = __ldg(y + item_y + {y_offset_j});
    const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
    float gy{j} = 0.0f;""".format(
                    j=j, y_offset_j=y_offset + j
                ).split("\n")
            )
        for spec in group_specs:
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
                lines.extend(
                    f"        const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;\n        const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;".split(
                        "\n"
                    )
                )
            for k in unique_k:
                lines.extend(
                    f"        const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;\n        float gg{k} = 0.0f;".split(
                        "\n"
                    )
                )
            lines.extend(
                f"        const float ww = active ? __ldg(weight + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;\n        const float rwv = active ? __ldg(rw + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;".split(
                    "\n"
                )
            )
            if need_grad_weight and weight_mode != "shared":
                lines.append("        float gw = 0.0f;")
            for i_idx, j_idx, k_idx, coeff in terms:
                i = i_idx
                j = j_idx
                k = k_idx
                c = float_literal(coeff)
                lines.extend(
                    r"""        const float cgo_{i}_{j}_{k} = {c} * gov{k};
        gy{j} = fmaf(cgo_{i}_{j}_{k}, ww * rxv{i} + xv{i} * rwv, gy{j});
        const float xry_{i}_{j}_{k} = yv{j} * rxv{i} + xv{i} * ryv{j};""".format(i=i, j=j, k=k, c=c).split("\n")
                )
                if need_grad_weight and weight_mode != "shared":
                    lines.append(
                        f"        gw = fmaf(cgo_{i}_{j}_{k}, xry_{i}_{j}_{k}, gw);"
                    )
                lines.append(
                    f"        gg{k} = fmaf({c}, ww * xry_{i}_{j}_{k} + xv{i} * yv{j} * rwv, gg{k});"
                )
            lines.append("        if (active) {")
            if need_grad_weight and weight_mode != "shared":
                lines.append(
                    f"            gw_out[item_w + {weight_offset} + {weight_u}] = gw;"
                )
            for k in unique_k:
                lines.append(
                    f"            atomicAdd(gg_out + item_go + {out_offset} + u * {d3} + {k}, gg{k});"
                    if spec["accumulate_output"]
                    else f"            gg_out[item_go + {out_offset} + u * {d3} + {k}] = gg{k};"
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
    grouped_y_kernel = (
        r"""
__global__ __launch_bounds__({group_threads})
void lunatp_uvu_double_bwd_grouped_y_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    float* __restrict__ gg_out,
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
        lunatp_uvu_double_bwd_grouped_y_fp32<<<group_grid, group_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_work_pointer},
            gg.data_ptr<float>(),
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

    shared_ygg_cases = []
    shared_w_partial_cases = []
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
            lines.extend(
                f"    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;\n    const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;".split(
                    "\n"
                )
            )
        for j in unique_j:
            lines.extend(
                r"""    const float yv{j} = __ldg(y + item_y + {y_offset_j});
    const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
    float gy{j} = 0.0f;""".format(
                    j=j, y_offset_j=y_offset + j
                ).split("\n")
            )
        for k in unique_k:
            lines.extend(
                f"    const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;\n    float gg{k} = 0.0f;".split(
                    "\n"
                )
            )
        lines.extend(
            f"    const float ww = active ? __ldg(weight + {weight_offset} + {weight_u}) : 0.0f;\n    const float rwv = active ? __ldg(rw + {weight_offset} + {weight_u}) : 0.0f;".split(
                "\n"
            )
        )
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            lines.extend(r"""    const float cgo_{i}_{j}_{k} = {c} * gov{k};
    gy{j} = fmaf(cgo_{i}_{j}_{k}, ww * rxv{i} + xv{i} * rwv, gy{j});
    gg{k} = fmaf({c}, ww * (yv{j} * rxv{i} + xv{i} * ryv{j}) + xv{i} * yv{j} * rwv, gg{k});""".format(i=i, j=j, k=k, c=c).split("\n"))
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
        for k in unique_k:
            lines.append("    if (active) {")
            lines.append(
                f"        atomicAdd(gg_out + item_go + {out_offset} + u * {d3} + {k}, gg{k});"
                if spec["accumulate_output"]
                else f"        gg_out[item_go + {out_offset} + u * {d3} + {k}] = gg{k};"
            )
            lines.append("    }")
        lines.append("    break;")
        lines.append("}")
        shared_ygg_cases.append("\n".join(lines))
        if not need_grad_weight:
            continue
        lines = [f"case {spec_idx}: {{"]
        lines.extend(
            r"""    if (u >= {channels}) return;
    const int64_t start_n = int64_t(blockIdx.z) * items_per_tile;
    const int64_t tile_end = start_n + int64_t(items_per_tile);
    const int64_t end_n = tile_end < num_items ? tile_end : num_items;
    float acc = 0.0f;
    for (int64_t n = start_n; n < end_n; ++n) {{
        const int64_t item_x = n * GEN_IN1_DIM;
        const int64_t item_y = n * GEN_IN2_DIM;
        const int64_t item_go = n * GEN_OUT_DIM;""".format(channels=channels).split("\n")
        )
        for i in unique_i:
            lines.extend(
                f"        const float xv{i} = __ldg(x + item_x + {x_offset} + u * {d1} + {i});\n        const float rxv{i} = __ldg(rx + item_x + {x_offset} + u * {d1} + {i});".split(
                    "\n"
                )
            )
        for j in unique_j:
            lines.extend(
                f"        const float yv{j} = __ldg(y + item_y + {y_offset + j});\n        const float ryv{j} = __ldg(ry + item_y + {y_offset + j});".split(
                    "\n"
                )
            )
        for k in unique_k:
            lines.append(
                f"        const float gov{k} = __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k});"
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            lines.append(
                f"        acc = fmaf({c} * gov{k}, yv{j} * rxv{i} + xv{i} * ryv{j}, acc);"
            )
        lines.append("    }")
        lines.append(
            f"    partials[int64_t(blockIdx.z) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u}] = acc;"
        )
        lines.append("    break;")
        lines.append("}")
        shared_w_partial_cases.append("\n".join(lines))
    shared_weight_launch = (
        r"""        const int items_per_tile = (x.size(0) >= 4096) ? 128 : 64;
        const int64_t num_tiles = (x.size(0) + items_per_tile - 1) / items_per_tile;
        auto partials = num_tiles == 1 ? gw : torch::empty({{num_tiles, GEN_WEIGHT_NUMEL}}, weight.options());
        const dim3 w_partial_grid(
            GEN_NUM_SPECS,
            static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1) / GEN_SPEC_CHANNEL_TILE),
            static_cast<unsigned int>(num_tiles));
        lunatp_uvu_double_bwd_shared_w_partial_fp32<<<w_partial_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
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
constexpr bool GEN_UNIFORM_CHANNELS = {uniform_channels};
constexpr bool GEN_GROUPED_Y = {grouped_edges};

{make_reduction_source}__global__ __launch_bounds__(GEN_X_THREADS, GEN_X_MIN_BLOCKS)
void lunatp_uvu_double_bwd_gx_fp32(
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    float* __restrict__ gx_out,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int group = blockIdx.y;
    const int u = blockIdx.z * GEN_X_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (group) {{
{gx_cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_double_bwd_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    float* __restrict__ gx_out,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    float* __restrict__ gg_out,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int spec = blockIdx.y;
    const int u = blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (spec) {{
{cases}
    default:
        break;
    }}
}}

{grouped_y_kernel}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_double_bwd_shared_ygg_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    float* __restrict__ gy_out,
    float* __restrict__ gg_out,
    int64_t num_items) {{
    const Index n = blockIdx.x;
    const int spec = blockIdx.y;
    const int u = blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (spec) {{
{shared_ygg_cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_double_bwd_shared_w_partial_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    float* __restrict__ partials,
    int64_t num_items,
    int items_per_tile) {{
    const int spec = blockIdx.x;
    const int u = blockIdx.y * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    switch (spec) {{
{shared_w_partial_cases}
    default:
        break;
    }}
}}


std::vector<torch::Tensor> run_second(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    bool shared_weight) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto gx = torch::{input_allocator}(x);
    auto gy = {edge_allocation};
    const bool use_shared_weight_reduce = shared_weight && GEN_SHARED_WEIGHT_REDUCE && (!GEN_UNIFORM_CHANNELS || x.size(0) >= 2048);
    auto gw = {gw_allocation};
    auto gg = torch::{output_allocator}(grad_out);
    if (x.size(0) == 0) {{
        {clear_weight}
        return {{{outputs}}};
    }}
    const dim3 x_block(GEN_X_THREADS);
    const dim3 spec_block(GEN_SPEC_THREADS);
    const dim3 gx_grid(
        static_cast<unsigned int>(x.size(0)),
        GEN_NUM_X_GROUPS,
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_X_CHANNEL_TILE - 1) / GEN_X_CHANNEL_TILE));
    const dim3 grid(
        static_cast<unsigned int>(x.size(0)),
        GEN_NUM_SPECS,
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1) / GEN_SPEC_CHANNEL_TILE));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto& lane = derivative_stream(x.get_device(), stream);
    DerivativeFork fork(lane, stream);
    lunatp_uvu_double_bwd_gx_fp32<<<gx_grid, x_block, 0, lane.stream>>>(
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        ry.data_ptr<float>(),
        rw.data_ptr<float>(),
        gx.data_ptr<float>(),
        x.size(0),
        shared_weight);
    if constexpr (GEN_GROUPED_Y) {{
{group_launch}
    }} else if (use_shared_weight_reduce) {{
        lunatp_uvu_double_bwd_shared_ygg_fp32<<<grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            gy.data_ptr<float>(),
            gg.data_ptr<float>(),
            x.size(0));
{shared_weight_launch}    }} else {{
        lunatp_uvu_double_bwd_fp32<<<grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            gx.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_pointer},
            gg.data_ptr<float>(),
            x.size(0),
            shared_weight);
    }}
    fork.join();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}

}}  // namespace

std::vector<torch::Tensor> lunatp_uvu_double_bwd_external_fp32(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw) {{
    return run_second(x, y, weight, grad_out, rx, ry, rw, false);
}}

std::vector<torch::Tensor> lunatp_uvu_double_bwd_shared_fp32(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw) {{
    return run_second(x, y, weight, grad_out, rx, ry, rw, true);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    // Resolve this module before CUDA instrumentation queries its first launch.
    cudaFuncAttributes attributes;
    const auto status = cudaFuncGetAttributes(&attributes, lunatp_uvu_double_bwd_gx_fp32);
    TORCH_CHECK(status == cudaSuccess, "UVU second kernel initialization: ", cudaGetErrorString(status));
    m.def("lunatp_uvu_double_bwd_external_fp32", &lunatp_uvu_double_bwd_external_fp32, "UVU second-order derivative external weights (CUDA)");
    m.def("lunatp_uvu_double_bwd_shared_fp32", &lunatp_uvu_double_bwd_shared_fp32, "UVU second-order derivative shared weights (CUDA)");
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
        uniform_channels="true" if len(channel_set) == 1 else "false",
        grouped_edges="true" if use_grouped_y else "false",
        make_reduction_source=make_reduction_source(meta),
        gx_cases=chr(10).join(gx_cases),
        cases=chr(10).join(cases),
        grouped_y_kernel=grouped_y_kernel,
        shared_ygg_cases=chr(10).join(shared_ygg_cases),
        shared_w_partial_cases=chr(10).join(shared_w_partial_cases),
        input_allocator="zeros_like"
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
        output_allocator="zeros_like" if meta["zero_output"] else "empty_like",
        clear_weight="gw.zero_();" if need_grad_weight else "",
        outputs=", ".join(outputs),
        group_launch=group_launch,
        shared_weight_launch=shared_weight_launch,
        gw_pointer=gw_pointer,
    )

    source = make_grid_source(source, max(x_channel_tile, spec_channel_tile))
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))
