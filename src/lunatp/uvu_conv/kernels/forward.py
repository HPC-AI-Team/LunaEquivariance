"""UVU convolution forward CUDA and native launch."""

from __future__ import annotations
import torch
from ...utils.cg import weight_channel_offset
from .utils import (
    UVU_CONV_CHANNEL_CHUNK,
    UVU_CONV_FWD_LIVE_OUTPUT_LIMIT,
    UVU_CONV_FWD_TILE_TABLE_LIMIT,
    UVU_CONV_L0_STREAM_STORE_THRESHOLD,
    UVU_CONV_NEIGHBOR_TASK_LIMIT,
)
from ....utils.precision import float_literal, cuda_source_to_fp64
from ....utils.extension import load_tp_extension
from ...utils.layout import (
    UvuPathSpec,
    build_signed_ij_classes,
    canonical_signed_signature,
    group_forward_paths,
    l0_feature_layout,
)


def emit_simple_terms(
    dst_lines: list[str],
    *,
    path_idx: int,
    spec_y_offset: int,
    terms: list[tuple[int, int, int, float]],
    weight_on_x: bool = False,
    weight_on_output: bool,
    y_name_base: int,
    accumulator_base: str | None,
) -> None:
    for i_idx, j_idx, k_idx, coeff in terms:
        c_lit = float_literal(float(coeff))
        j_name = spec_y_offset + int(j_idx) - y_name_base
        x_operand = f"wx_p{path_idx}_{int(i_idx)}" if weight_on_x else f"x{int(i_idx)}"
        y_operand = (
            f"y{j_name}"
            if weight_on_x or weight_on_output
            else f"wy_p{path_idx}_{j_name}"
        )
        accumulator_prefix = (
            f"acc_p{path_idx}" if accumulator_base is None else accumulator_base
        )
        accumulator = f"{accumulator_prefix}_{int(k_idx)}"
        dst_lines.append(
            f"        {accumulator} += {c_lit} * {x_operand} * {y_operand};"
        )


def emit_ij_terms(
    dst_lines: list[str],
    *,
    path_idx: int,
    spec_y_offset: int,
    terms: list[tuple[int, int, int, float]],
    weight_on_x: bool = False,
    weight_on_output: bool,
    y_name_base: int,
    accumulator_base: str | None,
) -> None:
    terms_by_ij: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for i_idx, j_idx, k_idx, coeff in terms:
        pair = (int(i_idx), spec_y_offset + int(j_idx) - y_name_base)
        terms_by_ij.setdefault(pair, []).append((int(k_idx), float(coeff)))
    for (i_idx, j_name), ij_terms in sorted(terms_by_ij.items()):
        x_operand = f"wx_p{path_idx}_{i_idx}" if weight_on_x else f"x{i_idx}"
        y_operand = (
            f"y{j_name}"
            if weight_on_x or weight_on_output
            else f"wy_p{path_idx}_{j_name}"
        )
        if len(ij_terms) == 1:
            k_idx, coeff = ij_terms[0]
            accumulator_prefix = (
                f"acc_p{path_idx}" if accumulator_base is None else accumulator_base
            )
            accumulator = f"{accumulator_prefix}_{k_idx}"
            dst_lines.append(
                f"        {accumulator} += {float_literal(coeff)} * {x_operand} * {y_operand};"
            )
        else:
            dst_lines.append(
                f"        const float xy_p{path_idx}_{i_idx}_{j_name} = {x_operand} * {y_operand};"
            )
            for k_idx, coeff in ij_terms:
                accumulator_prefix = (
                    f"acc_p{path_idx}" if accumulator_base is None else accumulator_base
                )
                accumulator = f"{accumulator_prefix}_{k_idx}"
                dst_lines.append(
                    f"        {accumulator} += {float_literal(coeff)} * xy_p{path_idx}_{i_idx}_{j_name};"
                )


def mode_switch(selector, plain_body, envelope_body, *, with_env, return_l0):
    if with_env or return_l0:
        body = envelope_body if with_env else plain_body
        return f"        switch ({selector}) {{\n{body}\n        default:\n            break;\n        }}"
    plain_switch = f"        switch ({selector}) {{\n{plain_body}\n        default:\n            break;\n        }}"
    envelope_switch = f"        switch ({selector}) {{\n{envelope_body}\n        default:\n            break;\n        }}"
    return f"    if (env == nullptr) {{\n{plain_switch}\n    }} else {{\n{envelope_switch}\n    }}"


def make_uvu_conv_fwd_source(
    specs: list[UvuPathSpec],
    *,
    in1_dim: int,
    in2_dim: int,
    out_dim: int,
    weight_numel: int,
    max_channels: int,
    term_grouping: str = "ij",
    reuse_duplicate_exprs: bool = False,
    signed_pair_signatures: bool = False,
    weight_on_x_paths: set[int] | None = None,
    return_l0: bool = False,
    with_env: bool = False,
    split_edge_chunk: int = 40,
    merge_channel_chunks: bool = False,
    variable_channels: bool = False,
    weight_index_type: str | None = None,
    resident_threads: int = 1024,
    task_warps: int | None = None,
    degree_tiers: bool = False,
) -> str:
    if split_edge_chunk <= 0:
        raise ValueError("split edge chunk must be positive")
    cases: list[str] = []
    env_cases: list[str] = []
    task_cases: list[str] = []
    env_task_cases: list[str] = []
    collaborative_cases: list[str] = []
    env_collaborative_cases: list[str] = []
    collaborative_store_layouts: dict[int, list[tuple[int, int]]] = {}
    exec_idx = 0
    weight_on_x_paths = set() if weight_on_x_paths is None else set(weight_on_x_paths)
    l0_offsets, l0_dim = l0_feature_layout(specs) if return_l0 else ({}, 0)
    l0_kernel_parameter = "    float* __restrict__ l0_feat,\n" if return_l0 else ""
    l0_launch_argument = "        l0_feat->data_ptr<float>(),\n" if return_l0 else ""
    l0_clear = (
        "    C10_CUDA_CHECK(cudaMemsetAsync(l0_feat->data_ptr<float>(), 0, l0_feat->numel() * sizeof(float), stream));\n"
        if return_l0 and any("l0_path" in spec for spec in specs)
        else ""
    )
    output_counts = {}
    for spec in specs:
        key = (int(spec["out_offset"]), int(spec["d3"]), int(spec["channels"]))
        output_counts[key] = output_counts.get(key, 0) + 1
    output_collision_keys = {key for key, count in output_counts.items() if count > 1}
    forward_groups = group_forward_paths(
        specs,
        axis="x",
        live_output_cap=UVU_CONV_FWD_LIVE_OUTPUT_LIMIT,
    )
    num_forward_execs = sum(
        ((int(group_key[3]) + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK)
        for group_key, _ in forward_groups
    )
    mode = "neighbors" if num_forward_execs <= UVU_CONV_NEIGHBOR_TASK_LIMIT else "tasks"
    channels = {int(spec["channels"]) for spec in specs}
    uniform_channels = channels.pop() if len(channels) == 1 and not return_l0 else None
    if (
        uniform_channels is not None
        and 4 * UVU_CONV_CHANNEL_CHUNK <= uniform_channels <= 8 * UVU_CONV_CHANNEL_CHUNK
        and uniform_channels % UVU_CONV_CHANNEL_CHUNK == 0
    ):
        warps = uniform_channels // UVU_CONV_CHANNEL_CHUNK
    elif (
        max_channels <= 32
        or max_channels >= 512
        or (max_channels <= 128 and num_forward_execs <= 32)
    ):
        warps = 4
    else:
        warps = 8 if num_forward_execs >= 32 else 16
    if warps >= 16:
        min_blocks = 2
    elif max_channels >= 512:
        min_blocks = 3
    elif max_channels > 256:
        min_blocks = 4
    else:
        min_blocks = 5
    use_launch_bounds = 256 < max_channels < 512
    signed_pair_signatures = signed_pair_signatures and not (
        use_launch_bounds and min_blocks >= 5
    )
    use_collaborative_order = mode == "neighbors"
    forward_edge_warps = 2 if use_collaborative_order else 1
    use_direct_tasks = mode == "tasks"
    if use_direct_tasks:
        group_tiles = [
            (int(key[3]) + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK
            for key, _ in forward_groups
        ]
        node_major = len(group_tiles) <= 16
        use_tile_table = node_major and sum(group_tiles) <= UVU_CONV_FWD_TILE_TABLE_LIMIT
        if (
            not node_major
            and not degree_tiers
            and all(tiles == (task_warps or warps) for tiles in group_tiles)
        ):
            ordered_task_prologue = r"""    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const Index group_task = blockIdx.x;
    const Index node_count = num_nodes;
    const int group = group_task / node_count;
    const Index node_rank = group_task - Index(group) * node_count;
    const int t = node_order[node_rank];
    const int channel_base = warp_id * GEN_CHANNEL_CHUNK;
"""
        else:
            if use_tile_table:
                decode = [
                    "        const Index node_rank = task_id / GEN_NUM_FORWARD_EXECS;",
                    "        const int tile = task_id % GEN_NUM_FORWARD_EXECS;",
                    "        t = node_order[node_rank];",
                    "        const int2 tile_info = FORWARD_TILE_INFO[tile];",
                    "        group = tile_info.x;",
                    "        channel_base = tile_info.y;",
                ]
            else:
                decode = []
                if node_major:
                    decode.extend([
                        "        const Index node_rank = task_id / GEN_NUM_FORWARD_EXECS;",
                        "        const int tile = task_id % GEN_NUM_FORWARD_EXECS;",
                        "        t = node_order[node_rank];",
                    ])
                else:
                    decode.extend([
                        "        Index node_begin = 0;",
                        "        Index node_end = num_nodes;",
                    ])
                    if degree_tiers:
                        decode.extend([
                            "        #pragma unroll",
                            "        for (int tier = 0; node_tiers && tier < 3; ++tier) {",
                            "            const Index end = node_tiers[tier];",
                            "            if (task_id < end * GEN_NUM_FORWARD_EXECS) {",
                            "                node_end = end;",
                            "                break;",
                            "            }",
                            "            node_begin = end;",
                            "        }",
                        ])
                    decode.extend([
                        "        const Index tier_nodes = node_end - node_begin;",
                        "        const Index local_task = task_id - node_begin * GEN_NUM_FORWARD_EXECS;",
                    ])
                tile_begin = 0
                for group, tiles in enumerate(group_tiles):
                    if group == len(group_tiles) - 1:
                        condition = "else" if group else ""
                    else:
                        bound = str(tile_begin + tiles)
                        selector = "tile" if node_major else "local_task"
                        if not node_major:
                            bound += " * tier_nodes"
                        condition = ("if" if group == 0 else "else if") + f" ({selector} < {bound})"
                    decode.append(f"        {condition} {{")
                    decode.append(f"            group = {group};")
                    if node_major:
                        decode.append(f"            channel_base = (tile - {tile_begin}) * GEN_CHANNEL_CHUNK;")
                    else:
                        decode.extend([
                            f"            const Index group_task = local_task - {tile_begin} * tier_nodes;",
                            f"            t = node_order[node_begin + group_task / {tiles}];",
                            f"            channel_base = (group_task % {tiles}) * GEN_CHANNEL_CHUNK;",
                        ])
                    decode.append("        }")
                    tile_begin += tiles
            ordered_task_prologue = r"""    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const Index task_id = Index(blockIdx.x) * GEN_FORWARD_TASK_TEAMS + warp_id;
    if (task_id >= num_tasks) return;
    int group, t, channel_base;
    {
""" + "\n".join(decode) + "\n    }"
    for group_idx, (
        (x_offset, y_offset, d1, channels),
        group_specs,
    ) in enumerate(forward_groups):
        group_specs = sorted(
            group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
        )
        num_chunks = (channels + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK
        group_exec_base = exec_idx
        if use_collaborative_order:
            for chunk_id in range(num_chunks):
                collaborative_store_layouts[group_exec_base + chunk_id] = [
                    (
                        int(spec["out_offset"])
                        + chunk_id * UVU_CONV_CHANNEL_CHUNK * int(spec["d3"])
                        + k,
                        int(spec["d3"]),
                    )
                    for _, spec in group_specs
                    for k in range(int(spec["d3"]))
                ]
        case_chunks = (
            [range(num_chunks)]
            if (merge_channel_chunks or use_direct_tasks) and num_chunks > 1
            else [range(chunk_id, chunk_id + 1) for chunk_id in range(num_chunks)]
        )
        for chunk_ids in case_chunks:
            chunk_ids = list(chunk_ids)
            case_labels = "\n".join(
                f"case {group_exec_base + chunk_id}:" for chunk_id in chunk_ids
            )
            u_expr = (
                f"(exec - {group_exec_base}) * GEN_CHANNEL_CHUNK + lane"
                if len(chunk_ids) > 1
                else f"{chunk_ids[0] * UVU_CONV_CHANNEL_CHUNK} + lane"
            )

            variants = []
            if not with_env:
                variants.append((None, False, False, cases))
                if use_direct_tasks:
                    variants.append((None, False, True, task_cases))
                if use_collaborative_order:
                    variants.append((None, True, False, collaborative_cases))
            if with_env or not return_l0:
                variants.append(("env", False, False, env_cases))
                if use_direct_tasks:
                    variants.append(("env", False, True, env_task_cases))
                if use_collaborative_order:
                    variants.append(("env", True, False, env_collaborative_cases))
            for env_name, collaborative, direct_task, destination_cases in variants:
                labels = f"case {group_idx}:" if direct_task else case_labels
                lines = [f"{labels} {{"]
                if direct_task:
                    u_expr_local = "channel_base + lane"
                else:
                    u_expr_local = u_expr
                lines.append(f"    const int u = {u_expr_local};")
                if not collaborative:
                    lines.append(f"    if (u >= {channels}) return;")
                for path_idx, spec in group_specs:
                    for k in range(int(spec["d3"])):
                        lines.append(f"    float acc_p{path_idx}_{k} = 0.0f;")
                if collaborative:
                    lines.append("    const Index begin = rowptr[t] + edge_warp;")
                    lines.append("    const Index end = rowptr[t + 1];")
                else:
                    lines.append("    const Index begin = rowptr[t];")
                    lines.append("    const Index end = rowptr[t + 1];")
                # All lanes reach the shared-memory barrier, including tails.
                masked_channels = (
                    collaborative and channels % UVU_CONV_CHANNEL_CHUNK != 0
                )
                if masked_channels:
                    lines.append(f"    if (u < {channels}) {{")
                begin_expr = "begin"
                end_expr = "end"
                y_load_base = None
                case_weight_paths = weight_on_x_paths
                case_l0_offsets = l0_offsets
                env_on_weight = with_env
                l0_output = "l0_feat" if return_l0 else None
                stream_l0_stores = (
                    return_l0 and l0_dim >= UVU_CONV_L0_STREAM_STORE_THRESHOLD
                )
                edge_stride = 2 if collaborative else 1
                wide_weight_product = variable_channels
                y_name_base = 0 if y_load_base is None else int(y_load_base)
                unique_i = sorted(
                    {int(term[0]) for _, spec in group_specs for term in spec["terms"]}
                )
                unique_j = sorted(
                    {
                        int(spec["y_offset"]) + int(term[1]) - y_name_base
                        for _, spec in group_specs
                        for term in spec["terms"]
                    }
                )
                case_weight_paths = (
                    set() if case_weight_paths is None else set(case_weight_paths)
                )
                case_l0_offsets = {} if case_l0_offsets is None else case_l0_offsets
                weight_edge = (
                    "Index(e)"
                    if weight_index_type is None
                    else f"static_cast<{weight_index_type}>(e)"
                )
                weight_base = f"{weight_edge} * {weight_numel}"

                if edge_stride <= 0:
                    raise ValueError("edge_stride must be positive")
                edge = "e"
                edge_step = f"++{edge}" if edge_stride == 1 else f"{edge} += {edge_stride}"
                lines.append(
                    f"    for (Index {edge} = {begin_expr}; {edge} < {end_expr}; {edge_step}) {{"
                )
                lines.append(f"        const int s = src[{edge}];")
                if weight_index_type is not None:
                    weight_pointer_index = (
                        f"static_cast<uint64_t>({weight_edge})"
                        if wide_weight_product
                        else weight_edge
                    )
                    lines.append(
                        f"        const float* weight_edge = weight + {weight_pointer_index} * {weight_numel};"
                    )
                if env_name is not None and not env_on_weight:
                    lines.append(f"        const float env_e = {env_name}[e];")
                for i_idx in unique_i:
                    x_expr = (
                        f"x[Index(s) * {in1_dim} + {x_offset} + u * {d1} + {i_idx}]"
                    )
                    lines.append(f"        const float x{i_idx} = {x_expr};")
                for j_name in unique_j:
                    if y_load_base is None:
                        y_expr = f"y[Index(e) * {in2_dim} + {j_name}]"
                    else:
                        y_expr = (
                            f"y[Index(e) * {in2_dim} + {int(y_load_base)} + {j_name}]"
                        )
                    if env_name is not None and not env_on_weight:
                        y_expr = f"{y_expr} * env_e"
                    lines.append(f"        const float y{j_name} = {y_expr};")
                for path_idx, spec in group_specs:
                    weight_offset = int(spec["weight_offset"])
                    terms = list(spec["terms"])
                    spec_y_offset = int(spec["y_offset"])
                    path_weight_on_x = path_idx in case_weight_paths
                    writes_l0 = l0_output is not None and path_idx in case_l0_offsets
                    reuse_l0_for_output = (
                        writes_l0
                        and (env_name is None or env_on_weight)
                        and int(spec["d3"]) == 1
                        and all(int(term[2]) == 0 for term in terms)
                    )
                    weight_expr = (
                        f"weight[{weight_base} + {weight_offset} + {weight_channel_offset(spec)}]"
                        if weight_index_type is None
                        else f"weight_edge[{weight_offset} + {weight_channel_offset(spec)}]"
                    )
                    if (
                        writes_l0
                        and env_name is not None
                        and env_on_weight
                        and not reuse_l0_for_output
                    ):
                        lines.append(
                            f"        const float raw_w_p{path_idx} = {weight_expr};"
                        )
                    if (
                        env_name is not None
                        and env_on_weight
                        and not reuse_l0_for_output
                    ):
                        weight_expr = f"{weight_expr} * {env_name}[e]"
                    lines.append(f"        const float w_p{path_idx} = {weight_expr};")
                    if writes_l0:
                        lines.append(f"        float l0_p{path_idx}_0 = 0.0f;")
                    if path_weight_on_x:
                        for i_idx in sorted({int(t[0]) for t in terms}):
                            lines.append(
                                f"        const float wx_p{path_idx}_{i_idx} = w_p{path_idx} * x{i_idx};"
                            )
                    else:
                        y_offset = int(spec["y_offset"])
                        for j_name in sorted(
                            {
                                y_offset + int(term[1]) - y_name_base
                                for term in spec["terms"]
                            }
                        ):
                            lines.append(
                                f"        const float wy_p{path_idx}_{j_name} = w_p{path_idx} * y{j_name};"
                            )
                    weight_on_x = path_weight_on_x
                    weight_on_output = False
                    signed_axis_signatures = False
                    accumulator_base = (
                        f"l0_p{path_idx}" if reuse_l0_for_output else None
                    )
                    if signed_pair_signatures:
                        if term_grouping not in ("ij", "simple"):
                            raise ValueError(
                                "signed pair signatures require ij/simple grouping"
                            )
                        singleton_mode = term_grouping
                        classes = build_signed_ij_classes(
                            terms,
                            spec_y_offset=spec_y_offset,
                            y_name_base=y_name_base,
                        )
                        if not any(
                            len(signature_class.members) > 1
                            for signature_class in classes
                        ):
                            if singleton_mode == "ij":
                                emit_ij_terms(
                                    lines,
                                    path_idx=path_idx,
                                    spec_y_offset=spec_y_offset,
                                    terms=terms,
                                    weight_on_x=weight_on_x,
                                    weight_on_output=weight_on_output,
                                    y_name_base=y_name_base,
                                    accumulator_base=accumulator_base,
                                )
                            else:
                                emit_simple_terms(
                                    lines,
                                    path_idx=path_idx,
                                    spec_y_offset=spec_y_offset,
                                    terms=terms,
                                    weight_on_x=weight_on_x,
                                    weight_on_output=weight_on_output,
                                    y_name_base=y_name_base,
                                    accumulator_base=accumulator_base,
                                )
                        else:
                            duplicate_idx = 0
                            for signature_class in classes:
                                members = signature_class.members
                                if len(members) == 1:
                                    member_terms = list(members[0].terms)
                                    if singleton_mode == "ij":
                                        emit_ij_terms(
                                            lines,
                                            path_idx=path_idx,
                                            spec_y_offset=spec_y_offset,
                                            terms=member_terms,
                                            weight_on_x=weight_on_x,
                                            weight_on_output=weight_on_output,
                                            y_name_base=y_name_base,
                                            accumulator_base=accumulator_base,
                                        )
                                    else:
                                        emit_simple_terms(
                                            lines,
                                            path_idx=path_idx,
                                            spec_y_offset=spec_y_offset,
                                            terms=member_terms,
                                            weight_on_x=weight_on_x,
                                            weight_on_output=weight_on_output,
                                            y_name_base=y_name_base,
                                            accumulator_base=accumulator_base,
                                        )
                                    continue

                                representative = members[0]
                                product_name = f"xy_p{path_idx}_{duplicate_idx}"
                                representative_x = (
                                    f"wx_p{path_idx}_{representative.i_idx}"
                                    if weight_on_x
                                    else f"x{representative.i_idx}"
                                )
                                representative_y = (
                                    f"y{representative.j_name}"
                                    if weight_on_x or weight_on_output
                                    else f"wy_p{path_idx}_{representative.j_name}"
                                )
                                lines.extend(
                                    f"        {{\n            float {product_name} = {representative_x} * {representative_y};".split(
                                        "\n"
                                    )
                                )
                                for member in members[1:]:
                                    op = "+=" if member.relative_sign > 0 else "-="
                                    member_x = (
                                        f"wx_p{path_idx}_{member.i_idx}"
                                        if weight_on_x
                                        else f"x{member.i_idx}"
                                    )
                                    member_y = (
                                        f"y{member.j_name}"
                                        if weight_on_x or weight_on_output
                                        else f"wy_p{path_idx}_{member.j_name}"
                                    )
                                    lines.append(
                                        f"            {product_name} {op} {member_x} * {member_y};"
                                    )
                                for _, _, k_idx, coeff in representative.terms:
                                    accumulator_prefix = (
                                        f"acc_p{path_idx}"
                                        if accumulator_base is None
                                        else accumulator_base
                                    )
                                    accumulator = f"{accumulator_prefix}_{k_idx}"
                                    lines.append(
                                        f"            {accumulator} += {float_literal(coeff)} * {product_name};"
                                    )
                                lines.append("        }")
                                duplicate_idx += 1
                    elif term_grouping == "jk":
                        reuse_duplicate_jk_exprs = reuse_duplicate_exprs
                        terms_by_jk: dict[tuple[int, int], list[tuple[int, float]]] = {}
                        for i_idx, j_idx, k_idx, coeff in terms:
                            key = (spec_y_offset + int(j_idx) - y_name_base, int(k_idx))
                            terms_by_jk.setdefault(key, []).append(
                                (int(i_idx), float(coeff))
                            )

                        x_prefix = f"wx_p{path_idx}_" if weight_on_x else "x"
                        y_prefix = (
                            "y"
                            if weight_on_x or weight_on_output
                            else f"wy_p{path_idx}_"
                        )

                        if not reuse_duplicate_jk_exprs or d1 <= 1:
                            for (j_name, k_idx), jk_terms in sorted(
                                terms_by_jk.items()
                            ):
                                accumulator_prefix = (
                                    f"acc_p{path_idx}"
                                    if accumulator_base is None
                                    else accumulator_base
                                )
                                accumulator = f"{accumulator_prefix}_{k_idx}"
                                if len(jk_terms) == 1:
                                    i_idx, coeff = jk_terms[0]
                                    lines.append(
                                        f"        {accumulator} += {float_literal(coeff)} * {x_prefix}{i_idx} * {y_prefix}{j_name};"
                                    )
                                else:
                                    expr = " + ".join(
                                        f"{float_literal(coeff)} * {x_prefix}{i_idx}"
                                        for i_idx, coeff in jk_terms
                                    )
                                    lines.extend(
                                        f"        const float sx_p{path_idx}_{j_name}_{k_idx} = {expr};\n        {accumulator} += sx_p{path_idx}_{j_name}_{k_idx} * {y_prefix}{j_name};".split(
                                            "\n"
                                        )
                                    )
                        else:
                            sig_to_entries: dict[
                                tuple[tuple[int, int], ...]
                                | tuple[tuple[int, float], ...],
                                list[tuple[int, int, list[tuple[int, float]], int]],
                            ] = {}
                            for (j_name, k_idx), jk_terms in sorted(
                                terms_by_jk.items()
                            ):
                                if signed_axis_signatures:
                                    signature, orientation = canonical_signed_signature(
                                        jk_terms
                                    )
                                else:
                                    signature = tuple(
                                        sorted(
                                            (int(i_idx), float(coeff))
                                            for i_idx, coeff in jk_terms
                                        )
                                    )
                                    orientation = 1
                                sig_to_entries.setdefault(signature, []).append(
                                    (j_name, k_idx, jk_terms, orientation)
                                )

                            dup_idx = 0
                            for signature, entries in sorted(
                                sig_to_entries.items(), key=lambda item: item[1][0][:2]
                            ):
                                if len(entries) == 1:
                                    j_name, k_idx, jk_terms, _ = entries[0]
                                    accumulator_prefix = (
                                        f"acc_p{path_idx}"
                                        if accumulator_base is None
                                        else accumulator_base
                                    )
                                    accumulator = f"{accumulator_prefix}_{k_idx}"
                                    if len(jk_terms) == 1:
                                        i_idx, coeff = jk_terms[0]
                                        lines.append(
                                            f"        {accumulator} += {float_literal(coeff)} * {x_prefix}{i_idx} * {y_prefix}{j_name};"
                                        )
                                    else:
                                        expr = " + ".join(
                                            f"{float_literal(coeff)} * {x_prefix}{i_idx}"
                                            for i_idx, coeff in jk_terms
                                        )
                                        lines.extend(
                                            f"        const float sx_p{path_idx}_{j_name}_{k_idx} = {expr};\n        {accumulator} += sx_p{path_idx}_{j_name}_{k_idx} * {y_prefix}{j_name};".split(
                                                "\n"
                                            )
                                        )
                                    continue
                                representative_terms = entries[0][2]
                                representative_orientation = entries[0][3]
                                expr = " + ".join(
                                    f"{float_literal(coeff)} * {x_prefix}{i_idx}"
                                    for i_idx, coeff in representative_terms
                                )
                                sx_name = f"sx_dup_p{path_idx}_{dup_idx}"
                                lines.append(f"        const float {sx_name} = {expr};")
                                for j_name, k_idx, _, orientation in entries:
                                    operator = (
                                        "+="
                                        if orientation == representative_orientation
                                        else "-="
                                    )
                                    accumulator_prefix = (
                                        f"acc_p{path_idx}"
                                        if accumulator_base is None
                                        else accumulator_base
                                    )
                                    accumulator = f"{accumulator_prefix}_{k_idx}"
                                    lines.append(
                                        f"        {accumulator} {operator} {sx_name} * {y_prefix}{j_name};"
                                    )
                                dup_idx += 1
                    elif term_grouping == "ik":
                        reuse_duplicate_ik_exprs = reuse_duplicate_exprs
                        terms_by_ik: dict[tuple[int, int], list[tuple[int, float]]] = {}
                        for i_idx, j_idx, k_idx, coeff in terms:
                            terms_by_ik.setdefault((int(i_idx), int(k_idx)), []).append(
                                (
                                    spec_y_offset + int(j_idx) - y_name_base,
                                    float(coeff),
                                )
                            )

                        x_prefix = f"wx_p{path_idx}_" if weight_on_x else "x"
                        y_prefix = (
                            "y"
                            if weight_on_x or weight_on_output
                            else f"wy_p{path_idx}_"
                        )

                        if not reuse_duplicate_ik_exprs:
                            for (i_idx, k_idx), ik_terms in sorted(terms_by_ik.items()):
                                accumulator_prefix = (
                                    f"acc_p{path_idx}"
                                    if accumulator_base is None
                                    else accumulator_base
                                )
                                accumulator = f"{accumulator_prefix}_{k_idx}"
                                if len(ik_terms) == 1:
                                    j_name, coeff = ik_terms[0]
                                    lines.append(
                                        f"        {accumulator} += {float_literal(coeff)} * {x_prefix}{i_idx} * {y_prefix}{j_name};"
                                    )
                                else:
                                    expr = " + ".join(
                                        f"{float_literal(coeff)} * {y_prefix}{j_name}"
                                        for j_name, coeff in ik_terms
                                    )
                                    lines.extend(
                                        f"        const float sy_p{path_idx}_{i_idx}_{k_idx} = {expr};\n        {accumulator} += {x_prefix}{i_idx} * sy_p{path_idx}_{i_idx}_{k_idx};".split(
                                            "\n"
                                        )
                                    )
                        else:
                            signatures: dict[
                                tuple[tuple[int, int], ...]
                                | tuple[tuple[int, float], ...],
                                list[tuple[int, int, list[tuple[int, float]], int]],
                            ] = {}
                            for (i_idx, k_idx), ik_terms in sorted(terms_by_ik.items()):
                                if signed_axis_signatures:
                                    signature, orientation = canonical_signed_signature(
                                        ik_terms
                                    )
                                else:
                                    signature = tuple(
                                        sorted(
                                            (int(j_name), float(coeff))
                                            for j_name, coeff in ik_terms
                                        )
                                    )
                                    orientation = 1
                                signatures.setdefault(signature, []).append(
                                    (i_idx, k_idx, ik_terms, orientation)
                                )

                            duplicate_idx = 0
                            for signature, entries in sorted(
                                signatures.items(),
                                key=lambda item: item[1][0][:2],
                            ):
                                if len(entries) == 1:
                                    i_idx, k_idx, ik_terms, _ = entries[0]
                                    accumulator_prefix = (
                                        f"acc_p{path_idx}"
                                        if accumulator_base is None
                                        else accumulator_base
                                    )
                                    accumulator = f"{accumulator_prefix}_{k_idx}"
                                    if len(ik_terms) == 1:
                                        j_name, coeff = ik_terms[0]
                                        lines.append(
                                            f"        {accumulator} += {float_literal(coeff)} * {x_prefix}{i_idx} * {y_prefix}{j_name};"
                                        )
                                    else:
                                        expr = " + ".join(
                                            f"{float_literal(coeff)} * {y_prefix}{j_name}"
                                            for j_name, coeff in ik_terms
                                        )
                                        lines.extend(
                                            f"        const float sy_p{path_idx}_{i_idx}_{k_idx} = {expr};\n        {accumulator} += {x_prefix}{i_idx} * sy_p{path_idx}_{i_idx}_{k_idx};".split(
                                                "\n"
                                            )
                                        )
                                    continue
                                representative_terms = entries[0][2]
                                representative_orientation = entries[0][3]
                                expr = " + ".join(
                                    f"{float_literal(coeff)} * {y_prefix}{j_name}"
                                    for j_name, coeff in representative_terms
                                )
                                sy_name = f"sy_dup_p{path_idx}_{duplicate_idx}"
                                lines.append(f"        const float {sy_name} = {expr};")
                                for i_idx, k_idx, _, orientation in entries:
                                    operator = (
                                        "+="
                                        if orientation == representative_orientation
                                        else "-="
                                    )
                                    accumulator_prefix = (
                                        f"acc_p{path_idx}"
                                        if accumulator_base is None
                                        else accumulator_base
                                    )
                                    accumulator = f"{accumulator_prefix}_{k_idx}"
                                    lines.append(
                                        f"        {accumulator} {operator} {x_prefix}{i_idx} * {sy_name};"
                                    )
                                duplicate_idx += 1
                    elif term_grouping == "ij":
                        emit_ij_terms(
                            lines,
                            path_idx=path_idx,
                            spec_y_offset=spec_y_offset,
                            terms=terms,
                            weight_on_x=weight_on_x,
                            weight_on_output=weight_on_output,
                            y_name_base=y_name_base,
                            accumulator_base=accumulator_base,
                        )
                    elif term_grouping == "simple":
                        emit_simple_terms(
                            lines,
                            path_idx=path_idx,
                            spec_y_offset=spec_y_offset,
                            terms=terms,
                            weight_on_x=weight_on_x,
                            weight_on_output=weight_on_output,
                            y_name_base=y_name_base,
                            accumulator_base=accumulator_base,
                        )
                    else:
                        raise ValueError(
                            f"unsupported forward term_grouping {term_grouping!r}"
                        )
                    if reuse_l0_for_output:
                        if env_name is None:
                            lines.append(
                                f"        acc_p{path_idx}_0 += l0_p{path_idx}_0;"
                            )
                        else:
                            lines.append(
                                f"        acc_p{path_idx}_0 += l0_p{path_idx}_0 * {env_name}[e];"
                            )
                    elif writes_l0:
                        l0_weight = (
                            f"raw_w_p{path_idx}"
                            if env_name is not None and env_on_weight
                            else f"w_p{path_idx}"
                        )
                        for i_idx, j_idx, _k_idx, coefficient in terms:
                            y_index = spec_y_offset + int(j_idx)
                            lines.append(
                                f"        l0_p{path_idx}_0 += {float_literal(float(coefficient))} * x{int(i_idx)} * {l0_weight} * y[Index(e) * {in2_dim} + {y_index}];"
                            )
                    if writes_l0:
                        l0_ptr = f"{l0_output} + Index(e) * {l0_dim} + {int(case_l0_offsets[path_idx])} + u"
                        l0_value = f"l0_p{path_idx}_0"
                        if "l0_path" in spec:
                            lines.append(f"        atomicAdd({l0_ptr}, {l0_value});")
                        elif stream_l0_stores:
                            lines.append(
                                f"        store_global_cs_f32({l0_ptr}, {l0_value});"
                            )
                        else:
                            lines.append(f"        *({l0_ptr}) = {l0_value};")
                lines.append("    }")
                if masked_channels:
                    lines.append("    }")
                if collaborative:
                    slot = 0
                    for path_idx, spec in group_specs:
                        for k in range(int(spec["d3"])):
                            lines.append(
                                f"    partials[(({slot} * GEN_FORWARD_EDGE_WARPS + edge_warp) * GEN_CHANNEL_CHUNK) + lane] = acc_p{path_idx}_{k};"
                            )
                            slot += 1
                else:
                    mode = "stream"
                    indent = "    "
                    case_collision_keys = output_collision_keys
                    case_collision_keys = (
                        set() if case_collision_keys is None else case_collision_keys
                    )
                    for path_idx, spec in group_specs:
                        d3 = int(spec["d3"])
                        out_offset = int(spec["out_offset"])
                        store_mode = (
                            "atomic"
                            if (
                                int(spec["out_offset"]),
                                int(spec["d3"]),
                                int(spec["channels"]),
                            )
                            in case_collision_keys
                            else mode
                        )
                        for k in range(d3):
                            if store_mode == "assign":
                                lines.append(
                                    f"{indent}out[Index(t) * {out_dim} + {out_offset} + u * {d3} + {k}] = acc_p{path_idx}_{k};"
                                )
                            elif store_mode == "stream":
                                lines.append(
                                    f"{indent}store_global_cs_f32(out + Index(t) * {out_dim} + {out_offset} + u * {d3} + {k}, acc_p{path_idx}_{k});"
                                )
                            elif store_mode == "atomic":
                                lines.append(
                                    f"{indent}atomicAdd(out + Index(t) * {out_dim} + {out_offset} + u * {d3} + {k}, acc_p{path_idx}_{k});"
                                )
                            else:
                                raise ValueError(
                                    f"unsupported forward store mode {mode!r}"
                                )
                lines.append("    break;")
                lines.append("}")
                destination_cases.append("\n".join(lines))
        exec_idx += num_chunks

    switch_body = "\n".join(cases)
    env_switch_body = "\n".join(env_cases)
    ordered_switch_body = "\n".join(task_cases) if use_direct_tasks else switch_body
    env_ordered_switch_body = (
        "\n".join(env_task_cases) if use_direct_tasks else env_switch_body
    )
    collaborative_switch_body = "\n".join(collaborative_cases)
    env_collaborative_switch_body = "\n".join(env_collaborative_cases)

    forward_task_warps = forward_edge_warps if use_collaborative_order else (task_warps or warps)
    forward_task_teams = forward_task_warps // forward_edge_warps
    forward_max_live_outputs = max(
        sum(int(spec["d3"]) for _, spec in group_specs)
        for _, group_specs in forward_groups
    )
    collaborative_store_counts: list[int] = []
    collaborative_store_offsets: list[int] = []
    collaborative_store_strides: list[int] = []
    for exec_id in range(exec_idx):
        layout = collaborative_store_layouts.get(exec_id, [])
        collaborative_store_counts.append(len(layout))
        padded = layout + [(0, 0)] * (forward_max_live_outputs - len(layout))
        collaborative_store_offsets.extend(offset for offset, _ in padded)
        collaborative_store_strides.extend(stride for _, stride in padded)
    collaborative_store_metadata = (
        "__device__ __constant__ int GEN_FORWARD_STORE_COUNTS[] = {"
        + ", ".join(map(str, collaborative_store_counts))
        + "};\n__device__ __constant__ int GEN_FORWARD_STORE_OFFSETS[] = {"
        + ", ".join(map(str, collaborative_store_offsets))
        + "};\n__device__ __constant__ int GEN_FORWARD_STORE_STRIDES[] = {"
        + ", ".join(map(str, collaborative_store_strides))
        + "};"
        if use_collaborative_order
        else ""
    )
    collaborative_tail_guard = ""
    if use_collaborative_order and any(
        int(spec["channels"]) % UVU_CONV_CHANNEL_CHUNK for spec in specs
    ):
        valid_lanes = [
            min(UVU_CONV_CHANNEL_CHUNK, channels - chunk * UVU_CONV_CHANNEL_CHUNK)
            for (_, _, _, channels), _ in forward_groups
            for chunk in range(
                ((channels + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK)
            )
        ]
        collaborative_store_metadata += (
            "\n__device__ __constant__ int GEN_FORWARD_VALID_LANES[] = {"
            + ", ".join(map(str, valid_lanes))
            + "};"
        )
        collaborative_tail_guard = " && lane < GEN_FORWARD_VALID_LANES[exec]"
    # Aliased outputs are zeroed by the launcher; atomic stores also cover
    # distinct outputs in the same launch without a per-store runtime branch.
    collaborative_store = (
        "atomicAdd" if output_collision_keys else "store_global_cs_f32"
    )
    # Use a common resident-thread budget across task geometries.
    # FP64 reserves twice the register space per live scalar.
    resident_threads = max(
        resident_threads, forward_task_warps * UVU_CONV_CHANNEL_CHUNK
    )
    task_launch_bounds = f"__launch_bounds__(GEN_FORWARD_TASK_WARPS * GEN_CHANNEL_CHUNK, {resident_threads} / (GEN_FORWARD_TASK_WARPS * GEN_CHANNEL_CHUNK))\n"
    cache_preference = (
        "cudaFuncCachePreferEqual"
        if use_collaborative_order
        else (
            "cudaFuncCachePreferNone" if use_launch_bounds else "cudaFuncCachePreferL1"
        )
    )
    store_body = (
        "*ptr = value;"
        if max_channels >= 320 and in2_dim >= 36
        else f'asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(ptr), "f"(value) : "memory");'
    )
    forward_task_min_blocks = min_blocks
    if use_collaborative_order:
        ordered_kernel_source = r"""
{task_launch_bounds}__global__ void lunatp_uvuconv_fwd_ordered_kernel_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ env,
{l0_kernel_parameter}    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    float* __restrict__ out,
    int64_t num_tasks,
    int64_t num_nodes) {{
    __shared__ float partials[
        GEN_FORWARD_MAX_LIVE_OUTPUTS * GEN_FORWARD_EDGE_WARPS
        * GEN_CHANNEL_CHUNK];
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int edge_warp = warp_id;
    const int t = node_order[blockIdx.x];
    const int exec = blockIdx.y;

{forward_cases}
    __syncthreads();
    if (edge_warp == 0{collaborative_tail_guard}) {{
        const int metadata_base = exec * GEN_FORWARD_MAX_LIVE_OUTPUTS;
        const int store_count = GEN_FORWARD_STORE_COUNTS[exec];
        #pragma unroll
        for (int slot = 0; slot < GEN_FORWARD_MAX_LIVE_OUTPUTS; ++slot) {{
            if (slot >= store_count) break;
            const int partial_base =
                (slot * GEN_FORWARD_EDGE_WARPS) * GEN_CHANNEL_CHUNK + lane;
            const float value = partials[partial_base]
                + partials[partial_base + GEN_CHANNEL_CHUNK];
            {collaborative_store}(
                out + Index(t) * {out_dim} + GEN_FORWARD_STORE_OFFSETS[metadata_base + slot]
                + lane * GEN_FORWARD_STORE_STRIDES[metadata_base + slot],
                value);
        }}
    }}
}}
""".format(
            task_launch_bounds=task_launch_bounds,
            l0_kernel_parameter=l0_kernel_parameter,
            forward_cases=mode_switch(
                "exec",
                collaborative_switch_body,
                env_collaborative_switch_body,
                with_env=with_env,
                return_l0=return_l0,
            ),
            collaborative_tail_guard=collaborative_tail_guard,
            collaborative_store=collaborative_store,
            out_dim=out_dim,
        )
    else:
        ordered_kernel_source = r"""
{task_launch_bounds}__global__ void lunatp_uvuconv_fwd_ordered_kernel_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ env,
{l0_kernel_parameter}    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    const int32_t* __restrict__ node_tiers,
    float* __restrict__ out,
    int64_t num_tasks,
    int64_t num_nodes) {{
{ordered_task_prologue}

{forward_cases}
}}
""".format(
            task_launch_bounds=task_launch_bounds,
            l0_kernel_parameter=l0_kernel_parameter,
            ordered_task_prologue=ordered_task_prologue,
            forward_cases=mode_switch(
                "group",
                ordered_switch_body,
                env_ordered_switch_body,
                with_env=with_env,
                return_l0=return_l0,
            ),
        )
    if use_direct_tasks and use_tile_table:
        entries = ", ".join(
            f"{{{group}, {tile * UVU_CONV_CHANNEL_CHUNK}}}"
            for group, tiles in enumerate(group_tiles)
            for tile in range(tiles)
        )
        ordered_kernel_source = (
            "__device__ __constant__ int2 FORWARD_TILE_INFO[] = {" + entries + "};\n"
            + ordered_kernel_source
        )
    if use_collaborative_order:
        ordered_grid = (
            "const dim3 grid(static_cast<unsigned int>(num_nodes), "
            "GEN_NUM_FORWARD_EXECS);"
        )
    elif use_direct_tasks:
        ordered_grid = "const dim3 grid(static_cast<unsigned int>((num_tasks + GEN_FORWARD_TASK_TEAMS - 1) / GEN_FORWARD_TASK_TEAMS));"
    split_kernel_source = ""
    split_cache_config = ""
    split_host_source = ""
    split_binding = ""
    ordered_launch = r"""    lunatp_uvuconv_fwd_ordered_kernel_fp32<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), env_ptr,
{l0_launch_argument}        src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(), node_order.data_ptr<int32_t>(),
{tier_argument}        out.data_ptr<float>(), num_tasks, num_nodes);"""
    ordered_launch = ordered_launch.format(
        l0_launch_argument=l0_launch_argument,
        tier_argument=(
            "" if use_collaborative_order else
            "        node_tiers ? node_tiers->data_ptr<int32_t>() : nullptr,\n"
        ),
    )
    ordered_cache_config = f"        cudaFuncSetCacheConfig(lunatp_uvuconv_fwd_ordered_kernel_fp32, {cache_preference});\n"
    ordered_host_source = r"""void run_uvu_conv_fwd_ordered(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    std::optional<torch::Tensor> env,
    std::optional<torch::Tensor> l0_feat,
    torch::Tensor src,
    torch::Tensor rowptr,
    torch::Tensor node_order,
    std::optional<torch::Tensor> node_tiers,
    torch::Tensor out,
    int64_t num_nodes) {{
    const float* env_ptr = env && env->numel() ? env->data_ptr<float>() : nullptr;
    const int64_t num_tasks = num_nodes * GEN_NUM_FORWARD_EXECS;
    const dim3 block(GEN_FORWARD_TASK_WARPS * GEN_CHANNEL_CHUNK);
    {ordered_grid}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
{l0_clear}    configure_kernel_cache();
{ordered_launch}
}}
""".format(
        ordered_grid=ordered_grid,
        l0_clear=l0_clear,
        ordered_launch=ordered_launch,
    )
    ordered_binding = '    m.def("run_uvu_conv_fwd_ordered", &run_uvu_conv_fwd_ordered, "LunaTP var CSR CGTP ordered forward (CUDA)");\n'
    return r"""
#include <torch/extension.h>
#include <optional>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {{

constexpr int GEN_NUM_FORWARD_EXECS = {exec_idx};
constexpr int GEN_CHANNEL_CHUNK = {UVU_CONV_CHANNEL_CHUNK};
constexpr int GEN_FORWARD_TASK_WARPS = {forward_task_warps};
constexpr int GEN_FORWARD_EDGE_WARPS = {forward_edge_warps};
constexpr int GEN_FORWARD_TASK_TEAMS = {forward_task_teams};
constexpr int GEN_FORWARD_MAX_LIVE_OUTPUTS = {forward_max_live_outputs};
constexpr int GEN_MAX_CHANNELS = {max_channels};
constexpr int GEN_FORWARD_TASK_MIN_BLOCKS = {forward_task_min_blocks};
constexpr int GEN_SPLIT_EDGE_CHUNK = {split_edge_chunk};
{collaborative_store_metadata}

__device__ __forceinline__ void store_global_cs_f32(float* ptr, float value) {{
    {store_body}
}}

__global__ void lunatp_uvuconv_fwd_kernel_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ env,
{l0_kernel_parameter}    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    float* __restrict__ out,
    int64_t num_nodes) {{
    const int t = node_order[blockIdx.x];
    const int exec = blockIdx.y;
    const int lane = threadIdx.x;
    if (t >= num_nodes || lane >= GEN_CHANNEL_CHUNK) return;

{forward_cases}
}}

{ordered_kernel_source}

{split_kernel_source}

inline void configure_kernel_cache() {{
    static bool configured = false;
    if (!configured) {{
        cudaFuncSetCacheConfig(lunatp_uvuconv_fwd_kernel_fp32, {cache_preference});
{ordered_cache_config}{split_cache_config}        configured = true;
    }}
}}

}}  // namespace

void run_uvu_conv_fwd(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    std::optional<torch::Tensor> env,
    std::optional<torch::Tensor> l0_feat,
    torch::Tensor src,
    torch::Tensor rowptr,
    torch::Tensor node_order,
    torch::Tensor out,
    int64_t num_nodes) {{
    const float* env_ptr = env && env->numel() ? env->data_ptr<float>() : nullptr;
    const dim3 block(GEN_CHANNEL_CHUNK);
    const dim3 grid(static_cast<unsigned int>(num_nodes), GEN_NUM_FORWARD_EXECS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
{l0_clear}    configure_kernel_cache();
    lunatp_uvuconv_fwd_kernel_fp32<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), env_ptr,
{l0_launch_argument}        src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(), node_order.data_ptr<int32_t>(),
        out.data_ptr<float>(), num_nodes);
}}

{ordered_host_source}
{split_host_source}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run_uvu_conv_fwd", &run_uvu_conv_fwd, "LunaTP var CSR CGTP forward (CUDA)");
{ordered_binding}{split_binding}}}
""".format(
        exec_idx=exec_idx,
        UVU_CONV_CHANNEL_CHUNK=UVU_CONV_CHANNEL_CHUNK,
        forward_task_warps=forward_task_warps,
        forward_edge_warps=forward_edge_warps,
        forward_task_teams=forward_task_teams,
        forward_max_live_outputs=forward_max_live_outputs,
        max_channels=max_channels,
        forward_task_min_blocks=forward_task_min_blocks,
        split_edge_chunk=split_edge_chunk,
        collaborative_store_metadata=collaborative_store_metadata,
        store_body=store_body,
        l0_kernel_parameter=l0_kernel_parameter,
        forward_cases=mode_switch(
            "exec", switch_body, env_switch_body, with_env=with_env, return_l0=return_l0
        ),
        ordered_kernel_source=ordered_kernel_source,
        split_kernel_source=split_kernel_source,
        cache_preference=cache_preference,
        ordered_cache_config=ordered_cache_config,
        split_cache_config=split_cache_config,
        l0_clear=l0_clear,
        l0_launch_argument=l0_launch_argument,
        ordered_host_source=ordered_host_source,
        split_host_source=split_host_source,
        ordered_binding=ordered_binding,
        split_binding=split_binding,
    )


def load_forward(op, *, return_l0=False, with_env=False):
    forward_uniform_channels = (
        len({int(spec["channels"]) for spec in op.generated_specs}) == 1
    )
    forward_source = make_uvu_conv_fwd_source(
        op.generated_specs,
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        out_dim=op.out_dim,
        weight_numel=op.weight_numel,
        max_channels=op.max_channels,
        signed_pair_signatures=True,
        weight_on_x_paths=(
            {
                index
                for index, spec in enumerate(op.generated_specs)
                if int(spec["d1"]) < int(spec["d2"])
            }
            if op.compute_dtype == torch.float64
            else None
        ),
        return_l0=return_l0,
        with_env=with_env,
        merge_channel_chunks=(forward_uniform_channels or op.in2_dim >= 25),
        variable_channels=not forward_uniform_channels,
        weight_index_type="WeightIndex",
        resident_threads=640 if op.compute_dtype == torch.float64 else 1024,
        degree_tiers=op.forward_degree_tiers,
    )
    forward_name_prefix = "lunatp_uvu_conv_fwd"
    forward_name_prefix += "_fp64" if op.compute_dtype == torch.float64 else "_fp32"
    plain_ext = load_tp_extension(
        forward_name_prefix,
        cuda_source_to_fp64(forward_source)
        if op.compute_dtype == torch.float64
        else forward_source,
        max_channels=op.max_channels,
        index_bits=op.index_bits,
    )
    return plain_ext


def run(
    op,
    x_node: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    src: torch.Tensor,
    dst_rowptr: torch.Tensor,
    node_order: torch.Tensor,
    env: torch.Tensor | None = None,
    return_l0: bool = False,
    *,
    node_tiers: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (x_node.shape[0], op.out_dim),
        device=x_node.device,
        dtype=op.compute_dtype,
    )
    if op.forward_output_needs_zero:
        out.zero_()
    l0 = y.new_empty((y.shape[0], op.l0_dim)) if return_l0 else None
    result = (out, l0) if return_l0 else out
    if not x_node.shape[0]:
        return result
    if return_l0 and not y.shape[0]:
        out.zero_()
        return result

    with_l0 = return_l0 and op.l0_dim > 0
    ext = op.fwd_ext
    if env is not None or with_l0:
        key = (env is not None, with_l0)
        ext = op._forward_extensions.get(key)
        if ext is None:
            ext = load_forward(
                op, return_l0=key[1], with_env=key[0],
            )
            op._forward_extensions[key] = ext

    ext.run_uvu_conv_fwd_ordered(
        x_node,
        y,
        weight,
        env,
        l0,
        src,
        dst_rowptr,
        node_order,
        node_tiers,
        out,
        x_node.shape[0],
    )
    return result
