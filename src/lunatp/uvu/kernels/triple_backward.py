"""UVU triple-backward CUDA templates."""

from collections import defaultdict
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
def make_uvu_triple_backward_source(
    meta: Mapping[str, object],
    *,
    need_grad_weight: bool = True,
    need_grad_rw: bool = True,
    needs=None,
) -> str:
    if use_uvu_dx_dy_dw_fusion(meta, 3):
        return make_uvu_dx_dy_dw_source(
            meta,
            3,
            need_grad_weight=need_grad_weight,
            need_grad_rw=need_grad_rw,
            weight_reduce_columns=8 if meta["weight_mode"] == "shared" else 32,
            needs=needs,
        )
    outputs = ["gx", "gy"]
    if need_grad_weight:
        outputs.append("gw")
    outputs.extend(("ggo", "grx", "gry"))
    if need_grad_rw:
        outputs.append("grw")
    weight_outputs = tuple(
        name
        for name, needed in (("gw", need_grad_weight), ("grw", need_grad_rw))
        if needed
    )
    weight_reduce_count = len(weight_outputs)
    gw_pointer = "gw.data_ptr<float>()" if need_grad_weight else "nullptr"
    gw_work_pointer = "gw_work.data_ptr<float>()" if need_grad_weight else "nullptr"
    gw_partials_pointer = (
        "gw_partials.data_ptr<float>()" if need_grad_weight else "nullptr"
    )
    gw_allocation = (
        "(shared_weight && !use_shared_weight_reduce && !GEN_GROUPED_Y) ? "
        "torch::zeros_like(weight) : torch::empty_like(weight)"
        if need_grad_weight
        else "torch::Tensor()"
    )
    grw_pointer = "grw.data_ptr<float>()" if need_grad_rw else "nullptr"
    grw_work_pointer = "grw_work.data_ptr<float>()" if need_grad_rw else "nullptr"
    grw_partials_pointer = (
        "grw_partials.data_ptr<float>()" if need_grad_rw else "nullptr"
    )
    grw_allocation = (
        "(shared_weight && !use_shared_weight_reduce && !GEN_GROUPED_Y) ? "
        "torch::zeros_like(weight) : torch::empty_like(weight)"
        if need_grad_rw
        else "torch::Tensor()"
    )
    gw_work_allocation = "gw" if need_grad_weight else "torch::Tensor()"
    grw_work_allocation = "grw" if need_grad_rw else "torch::Tensor()"
    gw_partials_allocation = (
        "num_tiles == 1 ? gw : torch::empty({num_tiles, GEN_WEIGHT_NUMEL}, weight.options())"
        if need_grad_weight
        else "torch::Tensor()"
    )
    grw_partials_allocation = (
        "num_tiles == 1 ? grw : torch::empty({num_tiles, GEN_WEIGHT_NUMEL}, weight.options())"
        if need_grad_rw
        else "torch::Tensor()"
    )

    specs = meta["generated_specs"]
    in1_dim = meta["in1_dim"]
    in2_dim = meta["in2_dim"]
    out_dim = meta["out_dim"]
    weight_numel = meta["weight_numel"]
    max_channels = meta["max_channels"]
    spec_channels = [spec["channels"] for spec in specs]
    channel_set = set(spec_channels)
    total_terms = sum(len(spec["terms"]) for spec in specs)
    weight_mode = meta["weight_mode"]
    weight_reduce_columns = 8 if weight_mode == "shared" and weight_outputs else 32
    # Group long path lists to reduce repeated edge-gradient atomics.
    use_grouped_y = len(specs) >= 32
    # Bound rounding error before long shared atomic sums become unstable.
    shared_reduce_start = (
        1024 if weight_mode == "shared" and not use_grouped_y else 2048
    )
    # Large expressions shorten register lifetimes with factored products.
    factored_third_terms = total_terms > 2200
    specialize_external = weight_mode == "external" and (
        factored_third_terms or len(channel_set) > 1
    )
    specialize_shared = weight_mode == "shared" and factored_third_terms
    use_y_block_reduce = len(channel_set) == 1 or max_channels <= 128
    enable_shared_weight_reduce = weight_mode == "shared" and weight_numel <= 1024
    x_channel_tile, x_threads, x_min_blocks = get_uvu_launch_config(max_channels)
    spec_channel_tile, spec_threads, spec_min_blocks = get_uvu_launch_config(
        max_channels, spec_channels
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
    x_cases = []
    for group_idx, ((x_offset, d1, channels), group_specs) in enumerate(x_group_items):
        lines = [f"case {group_idx}: {{"]
        lines.append(f"    if (u >= {channels}) return;")
        lines.append("    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;")
        lines.append("    const int64_t item_go = int64_t(n) * GEN_OUT_DIM;")
        lines.append("    const int64_t item_x = int64_t(n) * GEN_IN1_DIM;")
        lines.append("    const int64_t item_w = int64_t(n) * GEN_WEIGHT_NUMEL;")
        for i in range(d1):
            lines.extend(
                f"    float gx{i} = 0.0f;\n    float grx{i} = 0.0f;".split("\n")
            )
        for spec_idx, spec in enumerate(group_specs):
            y_offset = spec["y_offset"]
            out_offset = spec["out_offset"]
            weight_offset = spec["weight_offset"]
            weight_u = weight_channel_offset(spec)
            d3 = spec["d3"]
            terms = spec["terms"]
            unique_j = sorted({t[1] for t in terms})
            unique_k = sorted({t[2] for t in terms})
            if specialize_external:
                lines.extend(
                    r"""    const float ww_s{spec} = __ldg(weight + item_w + {weight_offset} + {weight_u});
    const float rw_s{spec} = __ldg(rw + item_w + {weight_offset} + {weight_u});
    const float bw_s{spec} = __ldg(bw + item_w + {weight_offset} + {weight_u});""".format(
                        spec=spec_idx,
                        weight_offset=weight_offset,
                        weight_u=weight_u,
                    ).split("\n")
                )
            elif specialize_shared:
                lines.extend(
                    r"""    const float ww_s{spec} = __ldg(weight + {weight_offset} + {weight_u});
    const float rw_s{spec} = __ldg(rw + {weight_offset} + {weight_u});
    const float bw_s{spec} = __ldg(bw + {weight_offset} + {weight_u});""".format(
                        spec=spec_idx,
                        weight_offset=weight_offset,
                        weight_u=weight_u,
                    ).split("\n")
                )
            else:
                lines.extend(
                    r"""    const float ww_s{spec} = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + item_w + {weight_offset} + {weight_u});
    const float rw_s{spec} = shared_weight ? __ldg(rw + {weight_offset} + {weight_u}) : __ldg(rw + item_w + {weight_offset} + {weight_u});
    const float bw_s{spec} = shared_weight ? __ldg(bw + {weight_offset} + {weight_u}) : __ldg(bw + item_w + {weight_offset} + {weight_u});""".format(
                        spec=spec_idx,
                        weight_offset=weight_offset,
                        weight_u=weight_u,
                    ).split("\n")
                )
            for j in unique_j:
                lines.extend(
                    r"""    const float y_s{spec}_{j} = __ldg(y + item_y + {y_offset_j});
    const float ry_s{spec}_{j} = __ldg(ry + item_y + {y_offset_j});
    const float by_s{spec}_{j} = __ldg(by + item_y + {y_offset_j});""".format(
                        spec=spec_idx,
                        j=j,
                        y_offset_j=y_offset + j,
                    ).split("\n")
                )
            for k in unique_k:
                lines.extend(
                    f"    const float go_s{spec_idx}_{k} = __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k});\n    const float bg_s{spec_idx}_{k} = __ldg(bg + item_go + {out_offset} + u * {d3} + {k});".split(
                        "\n"
                    )
                )
            pairs = defaultdict(list)
            for i, j, k, coefficient in terms:
                pairs[i, j].append((k, coefficient))
            for (i, j), entries in sorted(pairs.items()):
                sy, sr, sb = (f"{r}_s{spec_idx}_{j}" for r in ("y", "ry", "by"))
                ww, rw, bw = (f"{r}_s{spec_idx}" for r in ("ww", "rw", "bw"))
                tg = " + ".join(
                    f"({float_literal(c)} * go_s{spec_idx}_{k})" for k, c in entries
                )
                tb = " + ".join(
                    f"({float_literal(c)} * bg_s{spec_idx}_{k})" for k, c in entries
                )
                lines += r"""    {{
    const float tg = {tg};
    const float tb = {tb};
    gx{i} = fmaf(tg, {sb} * {rw} + {bw} * {sr}, gx{i});
    gx{i} = fmaf(tb, {ww} * {sr} + {sy} * {rw}, gx{i});
    grx{i} = fmaf(tg, {sb} * {ww} + {bw} * {sy}, grx{i});
    grx{i} = fmaf(tb, {ww} * {sy}, grx{i});
    }}""".format(
                    tg=tg,
                    tb=tb,
                    i=i,
                    sb=sb,
                    rw=rw,
                    bw=bw,
                    sr=sr,
                    ww=ww,
                    sy=sy,
                ).split("\n")
        for i in range(d1):
            lines.append(f"    gx_out[item_x + {x_offset} + u * {d1} + {i}] = gx{i};")
            lines.append(f"    grx_out[item_x + {x_offset} + u * {d1} + {i}] = grx{i};")
        lines.append("    break;")
        lines.append("}")
        x_cases.append("\n".join(lines))

    spec_cases = []
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
            lines.append("    const int64_t item_w = int64_t(n) * GEN_WEIGHT_NUMEL;")
            for i in unique_i:
                lines.extend(
                    r"""    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
    const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
    const float bxv{i} = active ? __ldg(bx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;""".format(
                        i=i, x_offset=x_offset, d1=d1
                    ).split("\n")
                )
            for j in unique_j:
                lines.extend(
                    r"""    const float yv{j} = __ldg(y + item_y + {y_offset_j});
    const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
    const float byv{j} = __ldg(by + item_y + {y_offset_j});
    float gy{j} = 0.0f;
    float gry{j} = 0.0f;""".format(
                        j=j, y_offset_j=y_offset + j
                    ).split("\n")
                )
            for k in unique_k:
                lines.extend(
                    r"""    const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
    const float bgv{k} = active ? __ldg(bg + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
    float ggo{k} = 0.0f;""".format(
                        k=k, out_offset=out_offset, d3=d3
                    ).split("\n")
                )
            lines.extend(r"""    float ww = 0.0f;
    float rwv = 0.0f;
    float bwv = 0.0f;
    if (active) {{""".format().split("\n"))
            if specialize_external:
                lines.extend(
                    r"""        ww = __ldg(weight + item_w + {weight_offset} + {weight_u});
        rwv = __ldg(rw + item_w + {weight_offset} + {weight_u});
        bwv = __ldg(bw + item_w + {weight_offset} + {weight_u});""".format(
                        weight_offset=weight_offset,
                        weight_u=weight_u,
                    ).split("\n")
                )
            elif specialize_shared:
                lines.extend(
                    r"""        ww = __ldg(weight + {weight_offset} + {weight_u});
        rwv = __ldg(rw + {weight_offset} + {weight_u});
        bwv = __ldg(bw + {weight_offset} + {weight_u});""".format(
                        weight_offset=weight_offset,
                        weight_u=weight_u,
                    ).split("\n")
                )
            else:
                lines.extend(
                    r"""        ww = shared_weight ? __ldg(weight + {weight_offset} + {weight_u}) : __ldg(weight + item_w + {weight_offset} + {weight_u});
        rwv = shared_weight ? __ldg(rw + {weight_offset} + {weight_u}) : __ldg(rw + item_w + {weight_offset} + {weight_u});
        bwv = shared_weight ? __ldg(bw + {weight_offset} + {weight_u}) : __ldg(bw + item_w + {weight_offset} + {weight_u});""".format(
                        weight_offset=weight_offset, weight_u=weight_u
                    ).split("\n")
                )
            lines.append("    }")
            if need_grad_weight:
                lines.append("    float gw = 0.0f;")
            if need_grad_rw:
                lines.append("    float grw = 0.0f;")
            for i_idx, j_idx, k_idx, coeff in terms:
                i = i_idx
                j = j_idx
                k = k_idx
                c = float_literal(coeff)
                if factored_third_terms:
                    lines.extend(
                        f"    gy{j} = fmaf({c}, gov{k} * (bxv{i} * rwv + bwv * rxv{i}) + bgv{k} * (ww * rxv{i} + xv{i} * rwv), gy{j});\n    gry{j} = fmaf({c}, gov{k} * (bxv{i} * ww + bwv * xv{i}) + bgv{k} * ww * xv{i}, gry{j});".split(
                            "\n"
                        )
                    )
                    if need_grad_weight:
                        lines.append(
                            f"    gw = fmaf({c}, gov{k} * (bxv{i} * ryv{j} + byv{j} * rxv{i}) + bgv{k} * (yv{j} * rxv{i} + xv{i} * ryv{j}), gw);"
                        )
                    if need_grad_rw:
                        lines.append(
                            f"    grw = fmaf({c}, gov{k} * (bxv{i} * yv{j} + byv{j} * xv{i}) + bgv{k} * xv{i} * yv{j}, grw);"
                        )
                else:
                    lines.extend(
                        f"    gy{j} = fmaf({c}, bxv{i} * gov{k} * rwv + bwv * gov{k} * rxv{i} + bgv{k} * (ww * rxv{i} + xv{i} * rwv), gy{j});\n    gry{j} = fmaf({c}, bxv{i} * gov{k} * ww + bwv * gov{k} * xv{i} + bgv{k} * ww * xv{i}, gry{j});".split(
                            "\n"
                        )
                    )
                    if need_grad_weight:
                        lines.append(
                            f"    gw = fmaf({c}, bxv{i} * gov{k} * ryv{j} + byv{j} * gov{k} * rxv{i} + bgv{k} * (yv{j} * rxv{i} + xv{i} * ryv{j}), gw);"
                        )
                    if need_grad_rw:
                        lines.append(
                            f"    grw = fmaf({c}, bxv{i} * gov{k} * yv{j} + byv{j} * gov{k} * xv{i} + bgv{k} * xv{i} * yv{j}, grw);"
                        )
                lines.append(
                    f"    ggo{k} = fmaf({c}, bxv{i} * (ww * ryv{j} + yv{j} * rwv) + byv{j} * (ww * rxv{i} + xv{i} * rwv) + bwv * (yv{j} * rxv{i} + xv{i} * ryv{j}), ggo{k});"
                )
            for j in unique_j:
                if use_y_block_reduce:
                    lines.extend(
                        r"""    const float gy_sum{j} = block_reduce_sum(gy{j});
    const float gry_sum{j} = block_reduce_sum(gry{j});
    if (threadIdx.x == 0) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_sum{j});
        atomicAdd(gry_out + item_y + {y_offset_j}, gry_sum{j});
    }}""".format(j=j, y_offset_j=y_offset + j).split(
                            "\n"
                        )
                    )
                else:
                    lines.extend(
                        r"""    const float gy_warp_sum{j} = warp_reduce_sum(gy{j}, active_mask);
    const float gry_warp_sum{j} = warp_reduce_sum(gry{j}, active_mask);
    if ((threadIdx.x & 31) == 0 && active_mask != 0u) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_warp_sum{j});
        atomicAdd(gry_out + item_y + {y_offset_j}, gry_warp_sum{j});
    }}""".format(j=j, y_offset_j=y_offset + j).split(
                            "\n"
                        )
                    )
            lines.append("    if (active) {")
            if specialize_external:
                if need_grad_weight:
                    lines.append(
                        f"        gw_out[item_w + {weight_offset} + {weight_u}] = gw;"
                    )
                if need_grad_rw:
                    lines.append(
                        f"        grw_out[item_w + {weight_offset} + {weight_u}] = grw;"
                    )
            elif specialize_shared:
                if need_grad_weight:
                    lines.append(
                        f"        atomicAdd(gw_out + {weight_offset} + {weight_u}, gw);"
                    )
                if need_grad_rw:
                    lines.append(
                        f"        atomicAdd(grw_out + {weight_offset} + {weight_u}, grw);"
                    )
            else:
                lines.append("        if (shared_weight) {")
                if need_grad_weight:
                    lines.append(
                        f"            atomicAdd(gw_out + {weight_offset} + {weight_u}, gw);"
                    )
                if need_grad_rw:
                    lines.append(
                        f"            atomicAdd(grw_out + {weight_offset} + {weight_u}, grw);"
                    )
                lines.append("        } else {")
                if need_grad_weight:
                    lines.append(
                        f"            gw_out[item_w + {weight_offset} + {weight_u}] = gw;"
                    )
                if need_grad_rw:
                    lines.append(
                        f"            grw_out[item_w + {weight_offset} + {weight_u}] = grw;"
                    )
                lines.append("        }")
            for k in unique_k:
                lines.append(
                    f"        atomicAdd(ggo_out + item_go + {out_offset} + u * {d3} + {k}, ggo{k});"
                    if spec["accumulate_output"]
                    else f"        ggo_out[item_go + {out_offset} + u * {d3} + {k}] = ggo{k};"
                )
            lines.append("    }")
            lines.append("    break;")
            lines.append("}")
            spec_cases.append("\n".join(lines))

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
    const float byv{j} = __ldg(by + item_y + {y_offset_j});
    float gy{j} = 0.0f;
    float gry{j} = 0.0f;""".format(j=j, y_offset_j=y_offset + j).split(
                    "\n"
                )
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
                    r"""        const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
        const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
        const float bxv{i} = active ? __ldg(bx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;""".format(
                        i=i, x_offset=x_offset, d1=d1
                    ).split("\n")
                )
            for k in unique_k:
                lines.extend(
                    r"""        const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
        const float bgv{k} = active ? __ldg(bg + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
        float ggo{k} = 0.0f;""".format(
                        k=k, out_offset=out_offset, d3=d3
                    ).split("\n")
                )
            lines.extend(
                r"""        const float ww = active ? __ldg(weight + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;
        const float rwv = active ? __ldg(rw + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;
        const float bwv = active ? __ldg(bw + {weight_base} + {weight_offset} + {weight_u}) : 0.0f;""".format(
                    weight_base=weight_base,
                    weight_offset=weight_offset,
                    weight_u=weight_u,
                ).split("\n")
            )
            if need_grad_weight and weight_mode != "shared":
                lines.append("        float gw = 0.0f;")
            if need_grad_rw and weight_mode != "shared":
                lines.append("        float grw = 0.0f;")
            pairs = defaultdict(list)
            for i, j, k, coefficient in terms:
                pairs[i, j].append((k, coefficient))
            for (i, j), entries in sorted(pairs.items()):
                tg = " + ".join(f"({float_literal(c)} * gov{k})" for k, c in entries)
                tb = " + ".join(f"({float_literal(c)} * bgv{k})" for k, c in entries)
                lines += r"""        {{
        const float tg = {tg};
        const float tb = {tb};
        const float a = ww * bxv{i} + bwv * xv{i};
        const float b = rwv * bxv{i} + bwv * rxv{i};
        const float c = ww * rxv{i} + rwv * xv{i};
        gy{j} = fmaf(tg, b, gy{j});
        gy{j} = fmaf(tb, c, gy{j});
        gry{j} = fmaf(tg, a, gry{j});
        gry{j} = fmaf(tb, ww * xv{i}, gry{j});""".format(tg=tg, tb=tb, i=i, j=j).split("\n")
                if need_grad_weight and weight_mode != "shared":
                    lines += r"""        const float p = bxv{i} * ryv{j} + rxv{i} * byv{j};
        const float q = rxv{i} * yv{j} + xv{i} * ryv{j};
        gw = fmaf(p, tg, gw);
        gw = fmaf(q, tb, gw);""".format(i=i, j=j).split("\n")
                if need_grad_rw and weight_mode != "shared":
                    lines += r"""        const float r = bxv{i} * yv{j} + xv{i} * byv{j};
        grw = fmaf(r, tg, grw);
        grw = fmaf(xv{i} * yv{j}, tb, grw);""".format(i=i, j=j).split(
                        "\n"
                    )
                lines.append(
                    f"        const float gg = ryv{j} * a + yv{j} * b + byv{j} * c;"
                )
                for k, coefficient in entries:
                    lines.append(
                        f"        ggo{k} = fmaf({float_literal(coefficient)}, gg, ggo{k});"
                    )
                lines.append("        }")
            lines.append("        if (active) {")
            if need_grad_weight and weight_mode != "shared":
                lines.append(
                    f"            gw_out[item_w + {weight_offset} + {weight_u}] = gw;"
                )
            if need_grad_rw and weight_mode != "shared":
                lines.append(
                    f"            grw_out[item_w + {weight_offset} + {weight_u}] = grw;"
                )
            for k in unique_k:
                lines.append(
                    f"            atomicAdd(ggo_out + item_go + {out_offset} + u * {d3} + {k}, ggo{k});"
                    if spec["accumulate_output"]
                    else f"            ggo_out[item_go + {out_offset} + u * {d3} + {k}] = ggo{k};"
                )
            lines.append("        }")
            lines.append("    }")
        for j in range(d2):
            lines.extend(
                f"    const float gy_sum{j} = block_reduce_sum(gy{j});\n    const float gry_sum{j} = block_reduce_sum(gry{j});\n    if (threadIdx.x == 0) {{".split(
                    "\n"
                )
            )
            lines.append(
                f"        atomicAdd(gy_out + item_y + {y_offset + j}, gy_sum{j});"
                if wide_channels
                else f"        gy_out[item_y + {y_offset + j}] = gy_sum{j};"
            )
            lines.append(
                f"        atomicAdd(gry_out + item_y + {y_offset + j}, gry_sum{j});"
                if wide_channels
                else f"        gry_out[item_y + {y_offset + j}] = gry_sum{j};"
            )
            lines.append("    }")
        lines.append("    break;")
        lines.append("}")
        y_group_cases.append("\n".join(lines))
    grouped_y_kernel = (
        r"""
__global__ __launch_bounds__({group_threads})
void lunatp_uvu_triple_bwd_grouped_y_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bw,
    const float* __restrict__ bg,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    float* __restrict__ ggo_out,
    float* __restrict__ gry_out,
    float* __restrict__ grw_out,
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
            lunatp_uvu_reduce_weight_fp32<{max}><<<(GEN_WEIGHT_NUMEL + {column_rounding}) / {weight_reduce_columns}, 256, 0, stream>>>(
                {work_pointers}{second_pointer},
                {output_pointers}{second_pointer}, x.size(0));
        }}
""".format(
            max=max(1, weight_reduce_count),
            column_rounding=weight_reduce_columns - 1,
            weight_reduce_columns=weight_reduce_columns,
            work_pointers=", ".join(
                (name + "_work.data_ptr<float>()" for name in weight_outputs)
            ),
            second_pointer=", nullptr" if weight_reduce_count == 1 else "",
            output_pointers=", ".join(
                (name + ".data_ptr<float>()" for name in weight_outputs)
            ),
        )
        if weight_outputs and weight_mode != "shared"
        else ""
    )
    group_launch = (
        r"""
        auto gw_work = {gw_work_allocation};
        auto grw_work = {grw_work_allocation};
        const dim3 group_block({group_threads});
        const dim3 group_grid(static_cast<unsigned int>(x.size(0)), {y_group_items_count}{group_tiles});
        lunatp_uvu_triple_bwd_grouped_y_fp32<<<group_grid, group_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            bx.data_ptr<float>(),
            by.data_ptr<float>(),
            bw.data_ptr<float>(),
            bg.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_work_pointer},
            ggo.data_ptr<float>(),
            gry.data_ptr<float>(),
            {grw_work_pointer},
            x.size(0));
{group_weight_reduction}""".format(
            gw_work_allocation=gw_work_allocation,
            grw_work_allocation=grw_work_allocation,
            group_threads=group_threads,
            y_group_items_count=len(y_group_items),
            group_tiles=group_tiles,
            gw_work_pointer=gw_work_pointer,
            grw_work_pointer=grw_work_pointer,
            group_weight_reduction=group_weight_reduction,
        )
        if use_grouped_y
        else ""
    )

    shared_ygg_cases = []
    shared_wrw_partial_cases = []
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
                r"""    const float xv{i} = active ? __ldg(x + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
    const float rxv{i} = active ? __ldg(rx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;
    const float bxv{i} = active ? __ldg(bx + item_x + {x_offset} + u * {d1} + {i}) : 0.0f;""".format(i=i, x_offset=x_offset, d1=d1).split(
                    "\n"
                )
            )
        for j in unique_j:
            lines.extend(
                r"""    const float yv{j} = __ldg(y + item_y + {y_offset_j});
    const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
    const float byv{j} = __ldg(by + item_y + {y_offset_j});
    float gy{j} = 0.0f;
    float gry{j} = 0.0f;""".format(j=j, y_offset_j=y_offset + j).split(
                    "\n"
                )
            )
        for k in unique_k:
            lines.extend(
                r"""    const float gov{k} = active ? __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
    const float bgv{k} = active ? __ldg(bg + item_go + {out_offset} + u * {d3} + {k}) : 0.0f;
    float ggo{k} = 0.0f;""".format(
                    k=k, out_offset=out_offset, d3=d3
                ).split("\n")
            )
        lines.extend(
            r"""    const float ww = active ? __ldg(weight + {weight_offset} + {weight_u}) : 0.0f;
    const float rwv = active ? __ldg(rw + {weight_offset} + {weight_u}) : 0.0f;
    const float bwv = active ? __ldg(bw + {weight_offset} + {weight_u}) : 0.0f;""".format(
                weight_offset=weight_offset,
                weight_u=weight_u,
            ).split("\n")
        )
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            lines.extend(r"""    gy{j} = fmaf({c}, bxv{i} * gov{k} * rwv + bwv * gov{k} * rxv{i} + bgv{k} * (ww * rxv{i} + xv{i} * rwv), gy{j});
    gry{j} = fmaf({c}, bxv{i} * gov{k} * ww + bwv * gov{k} * xv{i} + bgv{k} * ww * xv{i}, gry{j});
    ggo{k} = fmaf({c}, bxv{i} * (ww * ryv{j} + yv{j} * rwv) + byv{j} * (ww * rxv{i} + xv{i} * rwv) + bwv * (yv{j} * rxv{i} + xv{i} * ryv{j}), ggo{k});""".format(j=j, c=c, i=i, k=k).split("\n"))
        for j in unique_j:
            if use_y_block_reduce:
                lines.extend(
                    r"""    const float gy_sum{j} = block_reduce_sum(gy{j});
    const float gry_sum{j} = block_reduce_sum(gry{j});
    if (threadIdx.x == 0) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_sum{j});
        atomicAdd(gry_out + item_y + {y_offset_j}, gry_sum{j});
    }}""".format(j=j, y_offset_j=y_offset + j).split(
                        "\n"
                    )
                )
            else:
                lines.extend(
                    r"""    const float gy_warp_sum{j} = warp_reduce_sum(gy{j}, active_mask);
    const float gry_warp_sum{j} = warp_reduce_sum(gry{j}, active_mask);
    if ((threadIdx.x & 31) == 0 && active_mask != 0u) {{
        atomicAdd(gy_out + item_y + {y_offset_j}, gy_warp_sum{j});
        atomicAdd(gry_out + item_y + {y_offset_j}, gry_warp_sum{j});
    }}""".format(j=j, y_offset_j=y_offset + j).split("\n")
                )
        lines.append("    if (active) {")
        for k in unique_k:
            lines.append(
                f"        atomicAdd(ggo_out + item_go + {out_offset} + u * {d3} + {k}, ggo{k});"
                if spec["accumulate_output"]
                else f"        ggo_out[item_go + {out_offset} + u * {d3} + {k}] = ggo{k};"
            )
        lines.append("    }")
        lines.append("    break;")
        lines.append("}")
        shared_ygg_cases.append("\n".join(lines))
        if not (need_grad_weight or need_grad_rw):
            continue
        lines = [f"case {spec_idx}: {{"]
        lines.append(f"    if (u >= {channels}) return;")
        lines.append(
            "    const int64_t start_n = int64_t(blockIdx.z) * items_per_tile;"
        )
        lines.append("    const int64_t tile_end = start_n + int64_t(items_per_tile);")
        lines.append(
            "    const int64_t end_n = tile_end < num_items ? tile_end : num_items;"
        )
        if need_grad_weight:
            lines.append("    float gw = 0.0f;")
        if need_grad_rw:
            lines.append("    float grw = 0.0f;")
        lines.append("    for (int64_t n = start_n; n < end_n; ++n) {")
        lines.append("        const int64_t item_x = n * GEN_IN1_DIM;")
        lines.append("        const int64_t item_y = n * GEN_IN2_DIM;")
        lines.append("        const int64_t item_go = n * GEN_OUT_DIM;")
        for i in unique_i:
            lines.extend(
                r"""        const float xv{i} = __ldg(x + item_x + {x_offset} + u * {d1} + {i});
        const float rxv{i} = __ldg(rx + item_x + {x_offset} + u * {d1} + {i});
        const float bxv{i} = __ldg(bx + item_x + {x_offset} + u * {d1} + {i});""".format(
                    i=i, x_offset=x_offset, d1=d1
                ).split("\n")
            )
        for j in unique_j:
            lines.extend(
                r"""        const float yv{j} = __ldg(y + item_y + {y_offset_j});
        const float ryv{j} = __ldg(ry + item_y + {y_offset_j});
        const float byv{j} = __ldg(by + item_y + {y_offset_j});""".format(j=j, y_offset_j=y_offset + j).split(
                    "\n"
                )
            )
        for k in unique_k:
            lines.extend(
                f"        const float gov{k} = __ldg(grad_out + item_go + {out_offset} + u * {d3} + {k});\n        const float bgv{k} = __ldg(bg + item_go + {out_offset} + u * {d3} + {k});".split(
                    "\n"
                )
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            i = i_idx
            j = j_idx
            k = k_idx
            c = float_literal(coeff)
            if need_grad_weight:
                lines.append(
                    f"        gw = fmaf({c}, bxv{i} * gov{k} * ryv{j} + byv{j} * gov{k} * rxv{i} + bgv{k} * (yv{j} * rxv{i} + xv{i} * ryv{j}), gw);"
                )
            if need_grad_rw:
                lines.append(
                    f"        grw = fmaf({c}, bxv{i} * gov{k} * yv{j} + byv{j} * gov{k} * xv{i} + bgv{k} * xv{i} * yv{j}, grw);"
                )
        lines.append("    }")
        if need_grad_weight:
            lines.append(
                f"    gw_partials[int64_t(blockIdx.z) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u}] = gw;"
            )
        if need_grad_rw:
            lines.append(
                f"    grw_partials[int64_t(blockIdx.z) * GEN_WEIGHT_NUMEL + {weight_offset} + {weight_u}] = grw;"
            )
        lines.append("    break;")
        lines.append("}")
        shared_wrw_partial_cases.append("\n".join(lines))
    shared_weight_launch = (
        r"""        const int items_per_tile = (x.size(0) >= 4096) ? 128 : 64;
        const int64_t num_tiles = (x.size(0) + items_per_tile - 1) / items_per_tile;
        auto gw_partials = {gw_partials_allocation};
        auto grw_partials = {grw_partials_allocation};
        const dim3 w_partial_grid(
            GEN_NUM_SPECS,
            static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_SPEC_CHANNEL_TILE - 1) / GEN_SPEC_CHANNEL_TILE),
            static_cast<unsigned int>(num_tiles));
        lunatp_uvu_triple_bwd_shared_wrw_partial_fp32<<<w_partial_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            bx.data_ptr<float>(),
            by.data_ptr<float>(),
            bg.data_ptr<float>(),
            {gw_partials_pointer},
            {grw_partials_pointer},
            x.size(0),
            items_per_tile);
        const dim3 reduce_block(256);
        const dim3 reduce_grid(static_cast<unsigned int>((GEN_WEIGHT_NUMEL + {column_rounding}) / {weight_reduce_columns}));
        if (num_tiles > 1) lunatp_uvu_reduce_weight_fp32<{max}><<<reduce_grid, reduce_block, 0, stream>>>(
            {partial0_pointer},
            {partial1_pointer},
            {output0_pointer},
            {output1_pointer},
            num_tiles);
""".format(
            gw_partials_allocation=gw_partials_allocation,
            grw_partials_allocation=grw_partials_allocation,
            gw_partials_pointer=gw_partials_pointer,
            grw_partials_pointer=grw_partials_pointer,
            column_rounding=weight_reduce_columns - 1,
            weight_reduce_columns=weight_reduce_columns,
            max=max(1, weight_reduce_count),
            partial0_pointer=weight_outputs[0] + "_partials.data_ptr<float>()"
            if weight_outputs
            else "nullptr",
            partial1_pointer=weight_outputs[1] + "_partials.data_ptr<float>()"
            if weight_reduce_count == 2
            else "nullptr",
            output0_pointer=weight_outputs[0] + ".data_ptr<float>()"
            if weight_outputs
            else "nullptr",
            output1_pointer=weight_outputs[1] + ".data_ptr<float>()"
            if weight_reduce_count == 2
            else "nullptr",
        )
        if need_grad_weight or need_grad_rw
        else ""
    )

    if weight_mode == "shared" and use_grouped_y:
        group_launch += shared_weight_launch

    reduction_source = make_reduction_source(meta, columns=weight_reduce_columns)
    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <cstdint>
#include <cstdio>
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

{reduction_source}__global__ __launch_bounds__(GEN_X_THREADS, GEN_X_MIN_BLOCKS)
void lunatp_uvu_triple_bwd_x_fp32(
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ by,
    const float* __restrict__ bw,
    const float* __restrict__ bg,
    float* __restrict__ gx_out,
    float* __restrict__ grx_out,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int group = blockIdx.y;
    const int u = blockIdx.z * GEN_X_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (group) {{
{x_cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_triple_bwd_ywg_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bw,
    const float* __restrict__ bg,
    float* __restrict__ gy_out,
    float* __restrict__ gw_out,
    float* __restrict__ ggo_out,
    float* __restrict__ gry_out,
    float* __restrict__ grw_out,
    int64_t num_items,
    bool shared_weight) {{
    const Index n = blockIdx.x;
    const int spec = blockIdx.y;
    const int u = blockIdx.z * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    if (n >= num_items) return;
    switch (spec) {{
{spec_cases}
    default:
        break;
    }}
}}

{grouped_y_kernel}

__global__ __launch_bounds__(GEN_SPEC_THREADS, GEN_SPEC_MIN_BLOCKS)
void lunatp_uvu_triple_bwd_shared_ygg_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bw,
    const float* __restrict__ bg,
    float* __restrict__ gy_out,
    float* __restrict__ ggo_out,
    float* __restrict__ gry_out,
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
void lunatp_uvu_triple_bwd_shared_wrw_partial_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bg,
    float* __restrict__ gw_partials,
    float* __restrict__ grw_partials,
    int64_t num_items,
    int items_per_tile) {{
    const int spec = blockIdx.x;
    const int u = blockIdx.y * GEN_SPEC_CHANNEL_TILE + threadIdx.x;
    switch (spec) {{
{shared_wrw_partial_cases}
    default:
        break;
    }}
}}


std::vector<torch::Tensor> run_third(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    torch::Tensor bx,
    torch::Tensor by,
    torch::Tensor bw,
    torch::Tensor bg,
    bool shared_weight) {{
    c10::cuda::CUDAGuard guard(x.device());
    const bool use_shared_weight_reduce = shared_weight && GEN_SHARED_WEIGHT_REDUCE && x.size(0) >= {shared_reduce_start};
    auto gx = torch::{input_allocator}(x);
    auto gy = {edge_allocation};
    auto gw = {gw_allocation};
    auto ggo = torch::{output_allocator}(grad_out);
    auto grx = torch::{input_allocator}(x);
    auto gry = {edge_allocation};
    auto grw = {grw_allocation};
    if (x.size(0) == 0) {{
        {clear_weight}
        {clear_weight_direction}
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
    // Allocations and zero fills above belong to caller; fork only after them.
    auto& lane = derivative_stream(x.get_device(), stream);
    DerivativeFork fork(lane, stream);
    lunatp_uvu_triple_bwd_x_fp32<<<x_grid, x_block, 0, lane.stream>>>(
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        ry.data_ptr<float>(),
        rw.data_ptr<float>(),
        by.data_ptr<float>(),
        bw.data_ptr<float>(),
        bg.data_ptr<float>(),
        gx.data_ptr<float>(),
        grx.data_ptr<float>(),
        x.size(0),
        shared_weight);
    if constexpr (GEN_GROUPED_Y) {{
{group_launch}
    }} else if (use_shared_weight_reduce) {{
        lunatp_uvu_triple_bwd_shared_ygg_fp32<<<spec_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            bx.data_ptr<float>(),
            by.data_ptr<float>(),
            bw.data_ptr<float>(),
            bg.data_ptr<float>(),
            gy.data_ptr<float>(),
            ggo.data_ptr<float>(),
            gry.data_ptr<float>(),
            x.size(0));
{shared_weight_launch}    }} else {{
        lunatp_uvu_triple_bwd_ywg_fp32<<<spec_grid, spec_block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            rx.data_ptr<float>(),
            ry.data_ptr<float>(),
            rw.data_ptr<float>(),
            bx.data_ptr<float>(),
            by.data_ptr<float>(),
            bw.data_ptr<float>(),
            bg.data_ptr<float>(),
            gy.data_ptr<float>(),
            {gw_pointer},
            ggo.data_ptr<float>(),
            gry.data_ptr<float>(),
            {grw_pointer},
            x.size(0),
            shared_weight);
    }}
    // All inputs/outputs stay live until this join is enqueued on their caller.
    // No recordStream is needed: caller allocator reuse is ordered after join.
    fork.join();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}

}}  // namespace

std::vector<torch::Tensor> lunatp_uvu_triple_bwd_external_fp32(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    torch::Tensor bx,
    torch::Tensor by,
    torch::Tensor bw,
    torch::Tensor bg) {{
    return run_third(x, y, weight, grad_out, rx, ry, rw, bx, by, bw, bg, false);
}}

std::vector<torch::Tensor> lunatp_uvu_triple_bwd_shared_fp32(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    torch::Tensor bx,
    torch::Tensor by,
    torch::Tensor bw,
    torch::Tensor bg) {{
    return run_third(x, y, weight, grad_out, rx, ry, rw, bx, by, bw, bg, true);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    // Resolve this module before CUDA instrumentation queries its first launch.
    cudaFuncAttributes attributes;
    const auto status = cudaFuncGetAttributes(&attributes, lunatp_uvu_triple_bwd_x_fp32);
    TORCH_CHECK(status == cudaSuccess, "UVU third kernel initialization: ", cudaGetErrorString(status));
    m.def("lunatp_uvu_triple_bwd_external_fp32", &lunatp_uvu_triple_bwd_external_fp32, "UVU third-order derivative external weights (CUDA)");
    m.def("lunatp_uvu_triple_bwd_shared_fp32", &lunatp_uvu_triple_bwd_shared_fp32, "UVU third-order derivative shared weights (CUDA)");
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
        reduction_source=reduction_source,
        x_cases=chr(10).join(x_cases),
        spec_cases=chr(10).join(spec_cases),
        grouped_y_kernel=grouped_y_kernel,
        shared_ygg_cases=chr(10).join(shared_ygg_cases),
        shared_wrw_partial_cases=chr(10).join(shared_wrw_partial_cases),
        shared_reduce_start=shared_reduce_start,
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
        grw_allocation=grw_allocation,
        clear_weight="gw.zero_();" if need_grad_weight else "",
        clear_weight_direction="grw.zero_();" if need_grad_rw else "",
        outputs=", ".join(outputs),
        group_launch=group_launch,
        shared_weight_launch=shared_weight_launch,
        gw_pointer=gw_pointer,
        grw_pointer=grw_pointer,
    )

    source = make_grid_source(source, max(x_channel_tile, spec_channel_tile))
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))
