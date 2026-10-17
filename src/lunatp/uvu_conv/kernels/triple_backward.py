"""UVU convolution triple backward CUDA and native launch."""

from __future__ import annotations
from collections import defaultdict
from math import gcd
import torch
from ...utils.cg import weight_channel_offset
from ....utils.precision import float_literal, cuda_source_to_fp64
from ....utils.extension import load_tp_extension
from ...utils.runtime import get_overlap_state
from ...utils.layout import UvuPathSpec, build_signed_ij_classes, l0_feature_layout
from .utils import (
    UVU_CONV_TRIPLE_BWD_SOURCE_COMPONENT_LIMIT,
    add_optional_arguments,
    flat_coverage,
    group_source_input_paths,
    output_gradient_load,
)


TAIL_WARP_REDUCTION = r"""
__device__ __forceinline__ float warp_reduce_sum(float value, int valid_lanes) {
    const unsigned mask = 0xffffffffu >> (32 - valid_lanes);
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(mask, value, offset);
        if (lane + offset < valid_lanes) value += other;
    }
    return value;
}
"""


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
    lines.append(
        f"{indent}const {scalar_type} raw_sl0_seed = sl0 ? __ldg(sl0 + {address}) : 0.0;"
    )
    for i, j, _k, coefficient in spec["terms"]:
        x, y, w = (operands["x"][i], operands["y"][j], operands["w"])
        rx, ry, rw = (operands["rx"][i], operands["ry"][j], operands["rw"])
        sx, sy, sw = (operands["sx"][i], operands["sy"][j], operands["sw"])
        expressions = {
            "x": f"({sy} * {rw} + {ry} * {sw}) * raw_l0_seed + ({ry} * {w} + {y} * {rw}) * raw_sl0_seed",
            "rx": f"({sy} * {w} + {y} * {sw}) * raw_l0_seed + {y} * {w} * raw_sl0_seed",
            "y": f"({sx} * {rw} + {rx} * {sw}) * raw_l0_seed + ({rx} * {w} + {x} * {rw}) * raw_sl0_seed",
            "ry": f"({sx} * {w} + {x} * {sw}) * raw_l0_seed + {x} * {w} * raw_sl0_seed",
            "w": f"({sx} * {ry} + {rx} * {sy}) * raw_l0_seed + ({rx} * {y} + {x} * {ry}) * raw_sl0_seed",
            "rw": f"({sx} * {y} + {x} * {sy}) * raw_l0_seed + {x} * {y} * raw_sl0_seed",
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


def make_uvu_conv_triple_bwd_source_edge_source(
    specs: list[UvuPathSpec],
    *,
    in1_dim: int,
    in2_dim: int,
    out_dim: int,
    weight_numel: int,
    scalar_bytes: int = 4,
    with_l0: bool = False,
    with_env: bool = False,
    need_grad_env: bool = True,
) -> tuple[str, int, int]:
    """Generate the source/edge gradient schedule used by small programs."""
    need_grad_env = with_env and need_grad_env
    groups = group_source_input_paths(specs)
    schedules: list[tuple[int, int]] = []
    partial_bases: dict[int, int] = {}
    partial_count = 0
    for group_idx, (_, group_specs) in enumerate(groups):
        channel_counts = {int(spec["channels"]) for _, spec in group_specs}
        if len(channel_counts) != 1:
            raise ValueError("each triple backward x group must have one channel width")
        channels = channel_counts.pop()
        if channels <= 0:
            raise ValueError("triple backward channels must be positive")
        # Round the schedule, not the tensors: one partial per (possibly tail) warp.
        padded_channels = (channels + 31) // 32 * 32
        threads = (
            32
            if scalar_bytes == 8
            else max(
                candidate
                for candidate in range(32, min(padded_channels, 128) + 1, 32)
                if padded_channels % candidate == 0
            )
        )
        channel_tiles = padded_channels // threads
        partial_bases[group_idx] = partial_count
        partial_count += channel_tiles * (threads // 32)
        schedules.append((threads, channel_tiles))

    partition_count = min(
        max(2, (sum(d1 for (_, d1), _ in groups) + 7) // 8),
        min(
            len({(int(spec["y_offset"]), int(spec["d2"])) for _, spec in group_specs})
            for _, group_specs in groups
        ),
    )
    source_tasks: list[
        tuple[
            int,
            int,
            tuple[int, int],
            list[tuple[int, dict[str, object]]],
        ]
    ] = []
    for group_idx, (x_group, group_specs) in enumerate(groups):
        y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for path_idx, spec in group_specs:
            y_groups[(int(spec["y_offset"]), int(spec["d2"]))].append((path_idx, spec))
        if len(y_groups) < 2:
            source_tasks = []
            break
        partitions: list[list[tuple[int, dict[str, object]]]] = [
            [] for _ in range(partition_count)
        ]
        partition_costs = [0] * partition_count
        for _, partition_specs in sorted(
            y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            node_part = min(range(partition_count), key=partition_costs.__getitem__)
            partitions[node_part].extend(partition_specs)
            partition_costs[node_part] += sum(
                len(spec["terms"]) for _, spec in partition_specs
            )
        for node_part, partition_specs in enumerate(partitions):
            source_tasks.append((group_idx, node_part, x_group, partition_specs))
    node_parts = partition_count if source_tasks else 1
    launch_tasks = source_tasks or [
        (group_idx, 0, x_group, group_specs)
        for group_idx, (x_group, group_specs) in enumerate(groups)
    ]
    env_partial_bases = {}
    env_parts = 0
    if need_grad_env:
        for task_idx, (group_idx, _, _, _) in enumerate(launch_tasks):
            env_partial_bases[task_idx] = env_parts
            threads, tiles = schedules[group_idx]
            env_parts += tiles * (threads // 32)
    # Finish node/weight factors before starting edge factors in the same task.
    roles = ("x", "y") if scalar_bytes == 8 else ("all",)
    role_branches = []
    for role in roles:
        source_edge_specs = specs
        source_edge_in2_dim = in2_dim
        source_edge_out_dim = out_dim
        source_edge_weight_numel = weight_numel
        source_edge_partial_bases = partial_bases
        emit_weight_grads = True
        source_edge_output_role = role
        source_edge_tasks = source_tasks or None
        source_edge_with_l0 = with_l0
        source_edge_with_env = with_env
        source_edge_need_grad_env = need_grad_env
        source_edge_env_partial_bases = env_partial_bases
        source_edge_need_grad_env = source_edge_with_env and source_edge_need_grad_env
        source_edge_l0_offsets, source_edge_l0_dim = (
            l0_feature_layout(source_edge_specs) if source_edge_with_l0 else ({}, 0)
        )
        source_edge_seed_offsets = (
            {} if source_edge_with_env else source_edge_l0_offsets
        )
        split_source_partials = source_edge_tasks is not None
        if source_edge_tasks is None:
            source_edge_tasks = [
                (source_edge_group_idx, 0, source_input_group, source_edge_group_specs)
                for source_edge_group_idx, (
                    source_input_group,
                    source_edge_group_specs,
                ) in enumerate(group_source_input_paths(source_edge_specs))
            ]
        source_edge_branches: list[str] = []
        for source_edge_task_idx, (
            source_edge_group_idx,
            source_partial_idx,
            (source_edge_x_offset, source_edge_d1),
            source_edge_group_specs,
        ) in enumerate(source_edge_tasks):
            # Primal, output, and factor state use about nine doubles per component.
            split_source_edge_roles = (
                source_edge_output_role != "all"
                and 72
                * (
                    source_edge_d1
                    + max(
                        int(source_edge_spec["d2"])
                        for _, source_edge_spec in source_edge_group_specs
                    )
                )
                > 1024
            )
            if source_edge_output_role == "y" and not split_source_edge_roles:
                continue
            source_edge_task_role = (
                source_edge_output_role if split_source_edge_roles else "all"
            )
            emit_source_grads = source_edge_task_role != "y"
            emit_edge_grads = source_edge_task_role != "x"
            source_edge_channel_counts = {
                int(source_edge_spec["channels"])
                for _, source_edge_spec in source_edge_group_specs
            }
            if len(source_edge_channel_counts) != 1:
                raise ValueError(
                    "each triple backward x group must have one channel width"
                )
            source_edge_channels = source_edge_channel_counts.pop()
            source_edge_warp_tail = (
                f", min(32, {source_edge_channels} - u + lane)"
                if source_edge_channels % 32
                else ""
            )
            source_edge_y_groups: dict[
                tuple[int, int], list[tuple[int, dict[str, object]]]
            ] = defaultdict(list)
            for source_edge_path_idx, source_edge_spec in source_edge_group_specs:
                source_edge_y_groups[
                    (int(source_edge_spec["y_offset"]), int(source_edge_spec["d2"]))
                ].append((source_edge_path_idx, source_edge_spec))

            source_edge_branch_idx = (
                source_edge_task_idx if split_source_partials else source_edge_group_idx
            )
            source_edge_lines = [
                f"case {source_edge_branch_idx}: {{",
                f"    if (u >= {source_edge_channels}) break;",
            ]
            for source_component_idx in range(source_edge_d1):
                source_edge_lines.extend(
                    [
                        *(
                            [
                                f"    float dx{source_component_idx} = 0.0f;",
                                f"    float drx{source_component_idx} = 0.0f;",
                            ]
                            if emit_source_grads
                            else []
                        ),
                        *(
                            r"""    const float x{i} = __ldg(x + x_base + {i});
    const float rx{i} = __ldg(rx + x_base + {i});
    const float sx{i} = __ldg(sx + x_base + {i});""".format(
                                i=source_component_idx
                            ).split("\n")
                            if source_edge_task_role != "x"
                            else []
                        ),
                    ]
                )
            source_edge_lines.extend(
                r"""    for (Index p = begin; p < end; ++p) {{
{edge_index}
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};
        const Index go_base = Index(d) * {out_dim};""".format(
                    edge_index=(
                        "        const Index e = edge_ids_by_src[p];\n"
                        "        const int d = __ldg(dst + e);"
                    ),
                    in2_dim=source_edge_in2_dim,
                    weight_numel=source_edge_weight_numel,
                    out_dim=source_edge_out_dim,
                ).split("\n")
            )

            if source_edge_with_env:
                source_edge_lines.append(f"        const float env_e = __ldg(env + e);")
                if source_edge_need_grad_env:
                    source_edge_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
                if (
                    source_edge_need_grad_env
                    and emit_weight_grads
                    and emit_source_grads
                ):
                    source_edge_lines.extend(
                        [
                            "        float de_value = 0.0f;",
                            "        float dre_value = 0.0f;",
                        ]
                    )
            for (
                source_edge_y_offset,
                source_edge_d2,
            ), source_edge_path_specs in sorted(
                source_edge_y_groups.items(),
                key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
                reverse=True,
            ):
                source_edge_lines.append("        {")
                for edge_component_idx in range(source_edge_d2):
                    edge_feature_idx = source_edge_y_offset + edge_component_idx
                    source_edge_lines.extend(
                        [
                            *(
                                r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(
                                    j=edge_component_idx, j_abs=edge_feature_idx
                                ).split("\n")
                                if emit_source_grads
                                else []
                            ),
                            *(
                                [
                                    f"            float dy{edge_component_idx} = 0.0f;",
                                    f"            float dry{edge_component_idx} = 0.0f;",
                                ]
                                if emit_edge_grads
                                else []
                            ),
                        ]
                    )

                for source_edge_path_idx, source_edge_spec in sorted(
                    source_edge_path_specs,
                    key=lambda path: len(path[1]["terms"]),
                    reverse=True,
                ):
                    source_edge_out_offset = int(source_edge_spec["out_offset"])
                    source_edge_weight_offset = int(source_edge_spec["weight_offset"])
                    source_edge_d3 = int(source_edge_spec["d3"])
                    source_edge_terms = list(source_edge_spec["terms"])
                    source_edge_lines.extend(
                        r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                            weight_offset=source_edge_weight_offset,
                            weight_channel_offset=weight_channel_offset(
                                source_edge_spec
                            ),
                        ).split("\n")
                    )
                    if source_edge_with_env:
                        source_edge_lines[-3:] = emit_triple_bwd_env_weight_loads(
                            source_edge_weight_offset,
                            need_grad_env=source_edge_need_grad_env,
                            channel=weight_channel_offset(source_edge_spec),
                        )
                        if (
                            source_edge_need_grad_env
                            and emit_weight_grads
                            and emit_source_grads
                        ):
                            source_edge_lines.extend(
                                [
                                    "                float gv = 0.0f;",
                                    "                float hv = 0.0f;",
                                ]
                            )
                    if emit_weight_grads:
                        source_edge_lines.extend(
                            [
                                "                float dw_value = 0.0f;",
                                "                float drw_value = 0.0f;",
                            ]
                        )
                        # Accumulate contractions before applying path weights.
                        # The same contractions also produce both weight outputs.
                        source_edge_lines.extend(
                            f"                float {factor_name}{source_component_idx} = 0.0f;"
                            for source_component_idx in range(
                                source_edge_d1 if emit_source_grads else 0
                            )
                            for factor_name in ("x_mix", "x_ryq", "x_ryqs", "x_yq")
                        )
                        source_edge_lines.extend(
                            f"                float {factor_name}{edge_component_idx} = 0.0f;"
                            for edge_component_idx in range(
                                source_edge_d2 if emit_edge_grads else 0
                            )
                            for factor_name in ("y_mix", "y_rxq", "y_rxqs", "y_xq")
                        )
                    for source_edge_cg_class_idx, source_edge_cg_class in enumerate(
                        build_signed_ij_classes(
                            source_edge_terms,
                            spec_y_offset=source_edge_y_offset,
                            y_name_base=source_edge_y_offset,
                        )
                    ):
                        source_edge_cg_representative = source_edge_cg_class.members[0]
                        source_edge_grad_out_sum = " + ".join(
                            f"""{float_literal(float(source_edge_cg_coeff))} * {output_gradient_load("grad_out", source_edge_out_offset, source_edge_d3, int(output_component_idx), source_edge_path_idx, source_edge_seed_offsets, source_edge_l0_dim)}"""
                            for _, _, output_component_idx, source_edge_cg_coeff in source_edge_cg_representative.terms
                        )
                        source_edge_sgo_sum = " + ".join(
                            f"""{float_literal(float(source_edge_cg_coeff))} * {output_gradient_load("sgo", source_edge_out_offset, source_edge_d3, int(output_component_idx), source_edge_path_idx, source_edge_seed_offsets, source_edge_l0_dim)}"""
                            for _, _, output_component_idx, source_edge_cg_coeff in source_edge_cg_representative.terms
                        )
                        source_edge_lines.extend(
                            [
                                f"                const float q{source_edge_path_idx}_{source_edge_cg_class_idx} = {source_edge_grad_out_sum};",
                                f"                const float qs{source_edge_path_idx}_{source_edge_cg_class_idx} = {source_edge_sgo_sum};",
                            ]
                        )
                        for source_edge_cg_member in source_edge_cg_class.members:
                            source_edge_cg_op = (
                                "+="
                                if source_edge_cg_member.relative_sign > 0
                                else "-="
                            )
                            source_component_idx = source_edge_cg_member.i_idx
                            edge_component_idx = source_edge_cg_member.j_name
                            source_edge_q = (
                                f"q{source_edge_path_idx}_{source_edge_cg_class_idx}"
                            )
                            source_edge_qs = (
                                f"qs{source_edge_path_idx}_{source_edge_cg_class_idx}"
                            )
                            if emit_weight_grads:
                                source_edge_lines.extend(
                                    [
                                        *(
                                            r"""                x_mix{i} {op} sy{j} * {q} + y{j} * {qs};
                x_ryq{i} {op} ry{j} * {q};
                x_ryqs{i} {op} ry{j} * {qs};
                x_yq{i} {op} y{j} * {q};""".format(
                                                i=source_component_idx,
                                                op=source_edge_cg_op,
                                                j=edge_component_idx,
                                                q=source_edge_q,
                                                qs=source_edge_qs,
                                            ).split("\n")
                                            if emit_source_grads
                                            else []
                                        ),
                                        *(
                                            r"""                y_mix{j} {op} sx{i} * {q} + x{i} * {qs};
                y_rxq{j} {op} rx{i} * {q};
                y_rxqs{j} {op} rx{i} * {qs};
                y_xq{j} {op} x{i} * {q};""".format(
                                                j=edge_component_idx,
                                                op=source_edge_cg_op,
                                                i=source_component_idx,
                                                q=source_edge_q,
                                                qs=source_edge_qs,
                                            ).split("\n")
                                            if emit_edge_grads
                                            else []
                                        ),
                                    ]
                                )
                            else:
                                source_edge_lines.extend(
                                    r"""                dx{i} {op} ((sy{j} * rw_value + ry{j} * sw_value) * {q} + (ry{j} * w + y{j} * rw_value) * {qs});
                drx{i} {op} ((sy{j} * w + y{j} * sw_value) * {q} + y{j} * w * {qs});
                dy{j} {op} ((sx{i} * rw_value + rx{i} * sw_value) * {q} + (rx{i} * w + x{i} * rw_value) * {qs});
                dry{j} {op} ((sx{i} * w + x{i} * sw_value) * {q} + x{i} * w * {qs});""".format(
                                        i=source_component_idx,
                                        op=source_edge_cg_op,
                                        j=edge_component_idx,
                                        q=source_edge_q,
                                        qs=source_edge_qs,
                                    ).split("\n")
                                )
                                if source_edge_need_grad_env:
                                    source_edge_lines.extend(
                                        f"                dx{source_component_idx} {source_edge_cg_op} cross_weight * y{edge_component_idx} * {source_edge_q};\n                dy{edge_component_idx} {source_edge_cg_op} cross_weight * x{source_component_idx} * {source_edge_q};".split(
                                            "\n"
                                        )
                                    )
                    if emit_weight_grads:
                        for source_component_idx in range(
                            source_edge_d1 if emit_source_grads else 0
                        ):
                            if source_edge_task_role == "x":
                                source_edge_lines.extend(
                                    r"""                {{
                const float x{i} = __ldg(x + x_base + {i});
                const float rx{i} = __ldg(rx + x_base + {i});
                const float sx{i} = __ldg(sx + x_base + {i});""".format(
                                        i=source_component_idx
                                    ).split("\n")
                                )
                            source_edge_lines.extend(
                                r"""                dx{i} += rw_value * x_mix{i} + sw_value * x_ryq{i} + w * x_ryqs{i};
                drx{i} += w * x_mix{i} + sw_value * x_yq{i};
                dw_value += rx{i} * x_mix{i} + sx{i} * x_ryq{i} + x{i} * x_ryqs{i};
                drw_value += x{i} * x_mix{i} + sx{i} * x_yq{i};""".format(
                                    i=source_component_idx
                                ).split("\n")
                            )
                            if source_edge_need_grad_env:
                                source_edge_lines.extend(
                                    r"""                dx{i} += cross_weight * x_yq{i};
                gv += x{i} * x_yq{i};
                hv += rx{i} * x_yq{i} + x{i} * x_ryq{i};""".format(
                                        i=source_component_idx
                                    ).split("\n")
                                )
                            if source_edge_task_role == "x":
                                source_edge_lines.append("                }")
                        for edge_component_idx in range(
                            source_edge_d2 if emit_edge_grads else 0
                        ):
                            source_edge_lines.extend(
                                f"                dy{edge_component_idx} += rw_value * y_mix{edge_component_idx} + sw_value * y_rxq{edge_component_idx} + w * y_rxqs{edge_component_idx};\n                dry{edge_component_idx} += w * y_mix{edge_component_idx} + sw_value * y_xq{edge_component_idx};".split(
                                    "\n"
                                )
                            )
                            if source_edge_need_grad_env:
                                source_edge_lines.append(
                                    f"                dy{edge_component_idx} += cross_weight * y_xq{edge_component_idx};"
                                )
                        if emit_source_grads:
                            if source_edge_need_grad_env:
                                source_edge_lines.extend(
                                    r"""                de_value += raw_w * dw_value + raw_rw * drw_value + raw_sw * hv;
                dre_value += raw_w * drw_value + raw_sw * gv;
                dw_value = env_e * dw_value + re_e * drw_value + se_e * hv;
                drw_value = env_e * drw_value + se_e * gv;""".split("\n")
                                )
                            elif source_edge_with_env:
                                source_edge_lines.extend(
                                    f"                dw_value *= env_e;\n                drw_value *= env_e;".split(
                                        "\n"
                                    )
                                )
                            if (
                                source_edge_with_env
                                and source_edge_path_idx in source_edge_l0_offsets
                            ):
                                emit_triple_bwd_raw_l0(
                                    source_edge_lines,
                                    source_edge_spec,
                                    source_edge_path_idx,
                                    source_edge_l0_offsets,
                                    source_edge_l0_dim,
                                    {"w": "dw_value", "rw": "drw_value"},
                                )
                            source_edge_lines.extend(
                                f"                store_global_cs_f32(dw + weight_base + {source_edge_weight_offset} + {weight_channel_offset(source_edge_spec)}, dw_value);\n                store_global_cs_f32(drw + weight_base + {source_edge_weight_offset} + {weight_channel_offset(source_edge_spec)}, drw_value);".split(
                                    "\n"
                                )
                            )
                    if (
                        source_edge_with_env
                        and source_edge_path_idx in source_edge_l0_offsets
                    ):
                        emit_triple_bwd_raw_l0(
                            source_edge_lines,
                            source_edge_spec,
                            source_edge_path_idx,
                            source_edge_l0_offsets,
                            source_edge_l0_dim,
                            {
                                **(
                                    {
                                        "x": [
                                            f"dx{i}"
                                            for i in range(int(source_edge_spec["d1"]))
                                        ],
                                        "rx": [
                                            f"drx{i}"
                                            for i in range(int(source_edge_spec["d1"]))
                                        ],
                                    }
                                    if emit_source_grads
                                    else {}
                                ),
                                **(
                                    {
                                        "y": [
                                            f"dy{j}"
                                            for j in range(int(source_edge_spec["d2"]))
                                        ],
                                        "ry": [
                                            f"dry{j}"
                                            for j in range(int(source_edge_spec["d2"]))
                                        ],
                                    }
                                    if emit_edge_grads
                                    else {}
                                ),
                            },
                        )
                    source_edge_lines.append("            }")

                for edge_component_idx in range(
                    source_edge_d2 if emit_edge_grads else 0
                ):
                    edge_feature_idx = source_edge_y_offset + edge_component_idx
                    source_edge_lines.extend(
                        r"""            const float dy_sum{j} = warp_reduce_sum(dy{j}{reduction_tail});
            const float dry_sum{j} = warp_reduce_sum(dry{j}{reduction_tail});
            if (lane == 0) {{
                const int64_t partial = static_cast<int64_t>({edge_partial_base}) + blockIdx.z * (blockDim.x >> 5) + warp_id;
                const int64_t offset = (partial * num_edges + e) * GEN_IN2_DIM + {j_abs};
                dy_partial[offset] = dy_sum{j};
                dry_partial[offset] = dry_sum{j};
            }}""".format(
                            j=edge_component_idx,
                            reduction_tail=source_edge_warp_tail,
                            edge_partial_base=source_edge_partial_bases[
                                source_edge_group_idx
                            ],
                            j_abs=edge_feature_idx,
                        ).split("\n")
                    )
                source_edge_lines.append("        }")

            if source_edge_need_grad_env and emit_weight_grads and emit_source_grads:
                source_edge_env_base = source_edge_env_partial_bases[
                    source_edge_task_idx
                ]
                source_edge_lines.extend(
                    r"""        const float de_sum = warp_reduce_sum(de_value{reduction_tail});
        const float dre_sum = warp_reduce_sum(dre_value{reduction_tail});
        if (lane == 0) {{
            const Index part = {env_base} + blockIdx.z * (blockDim.x >> 5) + warp_id;
            de_partial[Index(e) * GEN_ENV_PARTS + part] = de_sum;
            dre_partial[Index(e) * GEN_ENV_PARTS + part] = dre_sum;
        }}""".format(
                        reduction_tail=source_edge_warp_tail,
                        env_base=source_edge_env_base,
                    ).split("\n")
                )
            source_edge_lines.append("    }")
            for source_component_idx in range(
                source_edge_d1 if emit_source_grads else 0
            ):
                source_grad_feature = f"{source_edge_x_offset} + u * {source_edge_d1} + {source_component_idx}"
                source_partial_offset = (
                    f"(static_cast<int64_t>(segment_idx) * GEN_NODE_PARTS + {source_partial_idx})"
                    if split_source_partials
                    else "static_cast<int64_t>(segment_idx)"
                )
                source_edge_lines.extend(
                    [
                        f"    dx_partial[{source_partial_offset} * GEN_IN1_DIM + {source_grad_feature}] = dx{source_component_idx};",
                        f"    drx_partial[{source_partial_offset} * GEN_IN1_DIM + {source_grad_feature}] = drx{source_component_idx};",
                    ]
                )
            source_edge_lines.extend(["    break;", "}"])
            source_edge_branches.append("\n".join(source_edge_lines))
        role_source = "\n".join(source_edge_branches)
        role_branches.append(role_source)
    source_branches = "\n    }\n    switch (task) {\n".join(role_branches)

    output_specs = specs
    output_in1_dim = in1_dim
    output_in2_dim = in2_dim
    output_out_dim = out_dim
    output_weight_numel = weight_numel
    output_scalar_bytes = scalar_bytes
    output_with_l0 = with_l0
    output_with_env = with_env
    output_need_grad_env = need_grad_env
    output_need_grad_env = output_with_env and output_need_grad_env
    _, output_output_overlap = flat_coverage(
        output_specs, output_out_dim, "out_offset", "d3"
    )
    output_cases: list[str] = []
    for output_group_idx, ((output_x_offset, output_d1), output_raw_specs) in enumerate(
        group_source_input_paths(output_specs)
    ):
        output_channels = {
            int(output_spec["channels"]) for _, output_spec in output_raw_specs
        }
        if len(output_channels) != 1:
            raise ValueError(
                "each triple backward hgo group must have one channel width"
            )
        output_channel_count = output_channels.pop()
        output_y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for output_path_idx, output_spec in output_raw_specs:
            output_y_groups[
                (int(output_spec["y_offset"]), int(output_spec["d2"]))
            ].append((output_path_idx, output_spec))
        output_ordered_specs = sorted(
            output_raw_specs, key=lambda path: len(path[1]["terms"]), reverse=True
        )
        output_lines = [
            f"case {output_group_idx}: {{",
            f"    if (u >= {output_channel_count}) break;",
        ]
        for output_path_idx, output_spec in output_ordered_specs:
            for output_k_idx in range(int(output_spec["d3"])):
                output_lines.append(
                    f"    float acc_p{output_path_idx}_{output_k_idx} = 0.0f;"
                )
        output_lines.extend(
            r"""    const Index begin = rowptr[t];
    const Index end = rowptr[t + 1];
{edge_loop}
        const Index x_base = Index(s) * {in1_dim} + {x_offset} + u * {d1};
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};""".format(
                edge_loop=(
                    "    for (Index e = begin; e < end; ++e) {\n"
                    "        const int s = __ldg(src + e);"
                ),
                in1_dim=output_in1_dim,
                x_offset=output_x_offset,
                d1=output_d1,
                in2_dim=output_in2_dim,
                weight_numel=output_weight_numel,
            ).split("\n")
        )
        for output_i_idx in range(output_d1):
            output_lines.extend(
                r"""        const float x{i} = __ldg(x + x_base + {i});
        const float rx{i} = __ldg(rx + x_base + {i});
        const float sx{i} = __ldg(sx + x_base + {i});""".format(i=output_i_idx).split("\n")
            )
        if output_with_env:
            output_lines.append(f"        const float env_e = __ldg(env + e);")
            if output_need_grad_env:
                output_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
        for (output_y_offset, output_d2), output_group_specs in sorted(
            output_y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            output_lines.append("        {")
            for output_j_idx in range(output_d2):
                output_j_abs = output_y_offset + output_j_idx
                output_lines.extend(
                    r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(
                        j=output_j_idx, j_abs=output_j_abs
                    ).split("\n")
                )
            for output_path_idx, output_spec in sorted(
                output_group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
            ):
                output_weight_offset = int(output_spec["weight_offset"])
                output_terms = list(output_spec["terms"])
                output_lines.extend(
                    r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                        weight_offset=output_weight_offset,
                        weight_channel_offset=weight_channel_offset(output_spec),
                    ).split("\n")
                )
                if output_with_env:
                    output_lines[-3:] = emit_triple_bwd_env_weight_loads(
                        output_weight_offset,
                        need_grad_env=output_need_grad_env,
                        channel=weight_channel_offset(output_spec),
                    )
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_lines.append("                float scalar_value = 0.0f;")
                output_scalar_first = True
                for output_class_idx, output_signature_class in enumerate(
                    build_signed_ij_classes(
                        output_terms,
                        spec_y_offset=output_y_offset,
                        y_name_base=output_y_offset,
                    )
                ):
                    output_representative = output_signature_class.members[0]

                    output_i_idx = output_representative.i_idx
                    output_j_idx = output_representative.j_name
                    output_extra = (
                        f" + cross_weight * x{output_i_idx} * y{output_j_idx}"
                        if output_need_grad_env
                        else ""
                    )
                    if output_scalar_bytes == 8 and output_d1 < output_d2:
                        output_mixed_product_value = (
                            f"ry{output_j_idx} * (sx{output_i_idx} * w + x{output_i_idx} * sw_value) + sy{output_j_idx} * (rx{output_i_idx} * w + x{output_i_idx} * rw_value) + y{output_j_idx} * (sx{output_i_idx} * rw_value + rx{output_i_idx} * sw_value)"
                            + output_extra
                        )
                    else:
                        output_mixed_product_value = (
                            f"sx{output_i_idx} * (ry{output_j_idx} * w + y{output_j_idx} * rw_value) + rx{output_i_idx} * (sy{output_j_idx} * w + y{output_j_idx} * sw_value) + x{output_i_idx} * (sy{output_j_idx} * rw_value + ry{output_j_idx} * sw_value)"
                            + output_extra
                        )
                    output_lines.append(
                        f"                float p{output_class_idx} = {output_mixed_product_value};"
                    )
                    for output_member in output_signature_class.members[1:]:
                        output_op = "+=" if output_member.relative_sign > 0 else "-="
                        output_i_idx = output_member.i_idx
                        output_j_idx = output_member.j_name
                        output_extra = (
                            f" + cross_weight * x{output_i_idx} * y{output_j_idx}"
                            if output_need_grad_env
                            else ""
                        )
                        if output_scalar_bytes == 8 and output_d1 < output_d2:
                            output_mixed_product_value = (
                                f"ry{output_j_idx} * (sx{output_i_idx} * w + x{output_i_idx} * sw_value) + sy{output_j_idx} * (rx{output_i_idx} * w + x{output_i_idx} * rw_value) + y{output_j_idx} * (sx{output_i_idx} * rw_value + rx{output_i_idx} * sw_value)"
                                + output_extra
                            )
                        else:
                            output_mixed_product_value = (
                                f"sx{output_i_idx} * (ry{output_j_idx} * w + y{output_j_idx} * rw_value) + rx{output_i_idx} * (sy{output_j_idx} * w + y{output_j_idx} * sw_value) + x{output_i_idx} * (sy{output_j_idx} * rw_value + ry{output_j_idx} * sw_value)"
                                + output_extra
                            )
                        output_lines.append(
                            f"                p{output_class_idx} {output_op} {output_mixed_product_value};"
                        )
                    for _, _, output_k_idx, output_coeff in output_representative.terms:
                        output_lines.append(
                            f"                acc_p{output_path_idx}_{int(output_k_idx)} += {float_literal(float(output_coeff))} * p{output_class_idx};"
                        )
                        if output_with_l0 and int(output_spec["d3"]) == 1:
                            output_scalar_op = "=" if output_scalar_first else "+="
                            output_lines.append(
                                f"                scalar_value {output_scalar_op} {float_literal(float(output_coeff))} * p{output_class_idx};"
                            )
                            output_scalar_first = False
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_offsets, output_width = l0_feature_layout(output_specs)
                    if output_with_env:
                        output_lines.append("                scalar_value = 0.0f;")
                        for output_i, output_j, output__k, output_coeff in output_spec[
                            "terms"
                        ]:
                            output_lines.append(
                                f"                scalar_value += {float_literal(float(output_coeff))} * (sx{output_i} * (ry{output_j} * raw_w + y{output_j} * raw_rw) + rx{output_i} * (sy{output_j} * raw_w + y{output_j} * raw_sw) + x{output_i} * (sy{output_j} * raw_rw + ry{output_j} * raw_sw));"
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
        output_lines.append(f"    const Index dgo_base = Index(t) * {output_out_dim};")
        for output_path_idx, output_spec in output_ordered_specs:
            output_out_offset = int(output_spec["out_offset"])
            output_d3 = int(output_spec["d3"])
            for output_k_idx in range(output_d3):
                output_address = (
                    f"dgo_base + {output_out_offset} + u * {output_d3} + {output_k_idx}"
                )
                output_value = f"acc_p{output_path_idx}_{output_k_idx}"
                if output_output_overlap:
                    output_lines.append(
                        f"    atomicAdd(dgo + {output_address}, {output_value});"
                    )
                else:
                    output_lines.append(f"    dgo[{output_address}] = {output_value};")
        output_lines.extend(["    break;", "}"])
        output_cases.append("\n".join(output_lines))
    hgo_cases = "\n".join(output_cases)
    max_group_channels = max(
        next(iter({int(spec["channels"]) for _, spec in group_specs}))
        for _, group_specs in groups
    )
    hgo_threads = max(32, min(128, gcd(*(int(spec["channels"]) for spec in specs))))
    hgo_tiles = (max_group_channels + hgo_threads - 1) // hgo_threads

    launches: list[str] = []
    for task_idx, (group_idx, _, _, _) in enumerate(launch_tasks):
        threads, channel_tiles = schedules[group_idx]
        launches.append(
            r"""
    lunatp_uvuconv_triple_bwd_source_fp32<{task_idx}><<<
        dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
        state.streams[{task_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(),
        grad_out.data_ptr<float>(), rx.data_ptr<float>(), ry.data_ptr<float>(),
        rw.data_ptr<float>(), sx.data_ptr<float>(), sy.data_ptr<float>(),
        sw.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
        edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
        segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
        segment_order.data_ptr<EdgeIndex>(), dx_partial.data_ptr<float>(),
        drx_partial.data_ptr<float>(), dy_partial.data_ptr<float>(),
        dry_partial.data_ptr<float>(), dw.data_ptr<float>(), drw.data_ptr<float>(),
        num_edges);
    cudaEventRecord(state.done[{task_idx}], state.streams[{task_idx}].stream());
""".format(
                task_idx=task_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
        )

    tail_reduction = (
        TAIL_WARP_REDUCTION if any(int(s["channels"]) % 32 for s in specs) else ""
    )
    source = (
        (r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

namespace {{
"""
        + (f"constexpr int GEN_ENV_PARTS = {env_parts};\n" if need_grad_env else "")
        + r"""constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_OUT_DIM = {out_dim};
constexpr int GEN_NUM_GROUPS = {groups_count};
constexpr int GEN_NUM_SOURCE_TASKS = {launch_tasks_count};
constexpr int GEN_NODE_PARTS = {node_parts};
constexpr int GEN_NUM_PARTIALS = {partial_count};
constexpr int GEN_DGRAD_OUT_TILES = {hgo_tiles};

__device__ __forceinline__ float warp_reduce_sum(float value) {{
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}}

{tail_reduction}__device__ __forceinline__ void store_global_cs_f32(float* ptr, float value) {{
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(ptr), "f"(value) : "memory");
}}

template <int TASK>
__global__ void lunatp_uvuconv_triple_bwd_source_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sw,
    const float* __restrict__ sgo,
    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ edge_ids_by_src,
    const int32_t* __restrict__ segment_nodes,
    const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end,
    const EdgeIndex* __restrict__ segment_order,
    float* __restrict__ dx_partial,
    float* __restrict__ drx_partial,
    float* __restrict__ dy_partial,
    float* __restrict__ dry_partial,
    float* __restrict__ dw,
    float* __restrict__ drw,
    int64_t num_edges) {{
    const Index segment_idx = segment_order[blockIdx.x];
    const int s_node = segment_nodes[segment_idx];
    const Index begin = segment_begin[segment_idx];
    const Index end = segment_end[segment_idx];
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int task = TASK;
    Index x_base = 0;
    switch (task) {{
""").format(
            in1_dim=in1_dim,
            in2_dim=in2_dim,
            out_dim=out_dim,
            groups_count=len(groups),
            launch_tasks_count=len(launch_tasks),
            node_parts=node_parts,
            partial_count=partial_count,
            hgo_tiles=hgo_tiles,
            tail_reduction=tail_reduction,
        )
        + "\n".join(
            f"    case {task_idx}: x_base = Index(s_node) * GEN_IN1_DIM + {x_offset} + u * {d1}; break;"
            for task_idx, (_, _, (x_offset, d1), _) in enumerate(launch_tasks)
        )
        + (r"""
    default: return;
    }}
    switch (task) {{
{source_branches}
    default: break;
    }}
}}

__global__ void lunatp_uvuconv_triple_bwd_node_reduce_fp32(
    const float* __restrict__ dx_partial,
    const float* __restrict__ drx_partial,
    const EdgeIndex* __restrict__ segment_rowptr,
    float* __restrict__ dx,
    float* __restrict__ drx,
    int64_t num_nodes) {{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_nodes * GEN_IN1_DIM;
    if (index >= total) return;
    const int node = static_cast<int>(index / GEN_IN1_DIM);
    const int feature = static_cast<int>(index - static_cast<int64_t>(node) * GEN_IN1_DIM);
    float dx_value = 0.0f;
    float drx_value = 0.0f;
    for (Index segment = segment_rowptr[node]; segment < segment_rowptr[node + 1]; ++segment) {{
        #pragma unroll
        for (int part = 0; part < GEN_NODE_PARTS; ++part) {{
            const int64_t offset =
                (static_cast<int64_t>(segment) * GEN_NODE_PARTS + part) * GEN_IN1_DIM + feature;
            dx_value += dx_partial[offset];
            drx_value += drx_partial[offset];
        }}
    }}
    dx[index] = dx_value;
    drx[index] = drx_value;
}}

__global__ void lunatp_uvuconv_triple_bwd_edge_reduce_fp32(
    const float* __restrict__ dy_partial,
    const float* __restrict__ dry_partial,
    float* __restrict__ dy,
    float* __restrict__ dry,
    int64_t num_edges) {{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_edges * GEN_IN2_DIM;
    if (index >= total) return;
    float dy_value = 0.0f;
    float dry_value = 0.0f;
    #pragma unroll
    for (int partial = 0; partial < GEN_NUM_PARTIALS; ++partial) {{
        const int64_t offset = static_cast<int64_t>(partial) * total + index;
        dy_value += dy_partial[offset];
        dry_value += dry_partial[offset];
    }}
    dy[index] = dy_value;
    dry[index] = dry_value;"""
        + (r"""
    if (index % GEN_IN2_DIM == 0) {{
        const Index edge = index / GEN_IN2_DIM;
        float de_value = 0.0f, dre_value = 0.0f;
        for (int p = 0; p < GEN_ENV_PARTS; ++p) {{
            de_value += de_partial[edge * GEN_ENV_PARTS + p];
            dre_value += dre_partial[edge * GEN_ENV_PARTS + p];
        }}
        de[edge] = de_value;
        dre[edge] = dre_value;
    }}""" if need_grad_env else "")
        + r"""
}}

__global__ void lunatp_uvuconv_triple_bwd_dgrad_out_dst_rows_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sw,
    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    float* __restrict__ dgo,
    int64_t num_nodes) {{
    const Index task = blockIdx.x;
    const int group = task / static_cast<int>(num_nodes);
    const int rank = task - Index(group) * num_nodes;
    if (group >= GEN_NUM_GROUPS) return;
    const int t = node_order[rank];
    const int u = blockIdx.y * blockDim.x + threadIdx.x;
    switch (group) {{
{hgo_cases}
    default: break;
    }}
}}

struct TripleBackwardStreams {{
    ~TripleBackwardStreams() {{
        // Match CUDAEvent's device-aware, non-throwing shutdown behavior.
        try {{
            c10::cuda::CUDAGuard guard(streams.front().device_index());
            cudaEventDestroy(start);
            for (auto event : done) cudaEventDestroy(event);
        }} catch (...) {{ /* CUDA may already be shutting down. */ }}
    }}
    TripleBackwardStreams(const TripleBackwardStreams&) = delete;
    TripleBackwardStreams& operator=(const TripleBackwardStreams&) = delete;
    explicit TripleBackwardStreams(int device, const std::vector<int64_t>& handles) {{
        c10::cuda::CUDAGuard guard(device);
        streams.reserve(GEN_NUM_SOURCE_TASKS);
        done.resize(GEN_NUM_SOURCE_TASKS);
        cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
        for (int task = 0; task < GEN_NUM_SOURCE_TASKS; ++task) {{
            streams.push_back(at::cuda::getStreamFromExternal(
                reinterpret_cast<cudaStream_t>(handles.at(task)), device));
            cudaEventCreateWithFlags(&done[task], cudaEventDisableTiming);
        }}
    }}
    std::vector<at::cuda::CUDAStream> streams;
    std::vector<cudaEvent_t> done;
    cudaEvent_t start = nullptr;
}};

}}  // namespace

void run_uvu_conv_triple_bwd(
    TripleBackwardStreams& state,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    torch::Tensor sx,
    torch::Tensor sy,
    torch::Tensor sw,
    torch::Tensor sgo,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor rowptr,
    torch::Tensor edge_ids_by_src,
    torch::Tensor segment_nodes,
    torch::Tensor segment_begin,
    torch::Tensor segment_end,
    torch::Tensor segment_rowptr,
    torch::Tensor segment_order,
    torch::Tensor node_order,
    torch::Tensor dx,
    torch::Tensor dx_partial,
    torch::Tensor dy,
    torch::Tensor dy_partial,
    torch::Tensor dw,
    torch::Tensor dgo,
    torch::Tensor drx,
    torch::Tensor drx_partial,
    torch::Tensor dry,
    torch::Tensor dry_partial,
    torch::Tensor drw,
    int64_t num_nodes,
    int64_t num_edges,
    int64_t num_segments) {{
    const int device = x.get_device();
    cudaStream_t current = at::cuda::getCurrentCUDAStream(device);
"""
        + (r"""    C10_CUDA_CHECK(cudaMemsetAsync(hl0->data_ptr<float>(), 0, hl0->numel() * sizeof(float), current));
""" if with_l0 and any("l0_path" in spec for spec in specs) else "")
        + r"""    cudaEventRecord(state.start, current);
    for (int task = 0; task < GEN_NUM_SOURCE_TASKS; ++task)
        cudaStreamWaitEvent(state.streams[task].stream(), state.start, 0);
{launches}
    lunatp_uvuconv_triple_bwd_dgrad_out_dst_rows_fp32<<<
        dim3(static_cast<unsigned int>(num_nodes * GEN_NUM_GROUPS), GEN_DGRAD_OUT_TILES),
        {hgo_threads}, 0, current>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), rw.data_ptr<float>(),
        sx.data_ptr<float>(), sy.data_ptr<float>(), sw.data_ptr<float>(),
        src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(),
        node_order.data_ptr<int32_t>(), dgo.data_ptr<float>(), num_nodes);
    for (int task = 0; task < GEN_NUM_SOURCE_TASKS; ++task)
        cudaStreamWaitEvent(current, state.done[task], 0);
    constexpr int threads = 256;
    const int64_t node_total = num_nodes * GEN_IN1_DIM;
    lunatp_uvuconv_triple_bwd_node_reduce_fp32<<<
        static_cast<int>((node_total + threads - 1) / threads), threads, 0, current>>>(
        dx_partial.data_ptr<float>(), drx_partial.data_ptr<float>(),
        segment_rowptr.data_ptr<EdgeIndex>(), dx.data_ptr<float>(),
        drx.data_ptr<float>(), num_nodes);
    const int64_t edge_total = num_edges * GEN_IN2_DIM;
    lunatp_uvuconv_triple_bwd_edge_reduce_fp32<<<
        static_cast<int>((edge_total + threads - 1) / threads), threads, 0, current>>>(
        dy_partial.data_ptr<float>(), dry_partial.data_ptr<float>(),
        dy.data_ptr<float>(), dry.data_ptr<float>(), num_edges);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
    pybind11::class_<TripleBackwardStreams>(module, "_Streams", pybind11::module_local())
        .def(pybind11::init<int, const std::vector<int64_t>&>());
    module.attr("_Streams").attr("stream_count") = GEN_NUM_SOURCE_TASKS;
    module.attr("_Streams").attr("stream_priority") = 0;
"""
        + (r"""    module.attr("env_parts") = GEN_ENV_PARTS;
""" if need_grad_env else "")
        + r"""    module.def("run_uvu_conv_triple_bwd", &run_uvu_conv_triple_bwd,
               "LunaTP source-owned complete third-order derivative (CUDA)");
}}
""").format(
            source_branches=source_branches,
            hgo_cases=hgo_cases,
            launches="".join(launches),
            hgo_threads=hgo_threads,
        )
    )
    parameters = (
        {
            "grad_l0": "const float* __restrict__",
            "sl0": "const float* __restrict__",
            "hl0": "float* __restrict__",
        }
        if with_l0
        else {}
    )
    if with_env:
        parameters["env"] = "const float* __restrict__"
    if need_grad_env:
        parameters.update(
            {
                "re": "const float* __restrict__",
                "se": "const float* __restrict__",
                "de_partial": "float* __restrict__",
                "dre_partial": "float* __restrict__",
                "de": "float* __restrict__",
                "dre": "float* __restrict__",
            }
        )
    return (
        add_optional_arguments(
            source,
            "run_uvu_conv_triple_bwd",
            parameters,
            (
                "grad_l0",
                "sl0",
                "hl0",
                "env",
                "re",
                "se",
                "de_partial",
                "dre_partial",
                "de",
                "dre",
            ),
        ),
        partial_count,
        node_parts,
    )


def make_uvu_conv_triple_bwd_split_source(
    specs: list[UvuPathSpec],
    *,
    in1_dim: int,
    in2_dim: int,
    out_dim: int,
    weight_numel: int,
    variable_channels: bool,
    scalar_bytes: int = 4,
    with_l0: bool = False,
    with_env: bool = False,
    need_grad_env: bool = True,
) -> tuple[str, int]:
    """Generate the role-split source schedule used by large programs."""
    need_grad_env = with_env and need_grad_env
    groups = group_source_input_paths(specs)
    schedules: list[tuple[int, int]] = []
    partial_bases: dict[int, int] = {}
    partial_count = 0
    for group_idx, (_, group_specs) in enumerate(groups):
        channel_counts = {int(spec["channels"]) for _, spec in group_specs}
        if len(channel_counts) != 1:
            raise ValueError("each triple backward x group must have one channel width")
        channels = channel_counts.pop()
        if channels <= 0:
            raise ValueError("triple backward channels must be positive")
        # Round the schedule, not the tensors: one partial per (possibly tail) warp.
        padded_channels = (channels + 31) // 32 * 32
        threads = max(
            candidate
            for candidate in range(32, min(padded_channels, 128) + 1, 32)
            if padded_channels % candidate == 0
        )
        channel_tiles = padded_channels // threads
        partial_bases[group_idx] = partial_count
        partial_count += channel_tiles * (threads // 32)
        schedules.append((threads, channel_tiles))

    node_specs = specs
    node_in2_dim = in2_dim
    node_out_dim = out_dim
    node_weight_numel = weight_numel
    node_with_l0 = with_l0
    node_with_env = with_env
    node_need_grad_env = need_grad_env
    node_need_grad_env = node_with_env and node_need_grad_env
    node_l0_offsets, node_l0_dim = (
        l0_feature_layout(node_specs) if node_with_l0 else ({}, 0)
    )
    node_node_seed_offsets = {} if node_with_env else node_l0_offsets
    node_branches: list[str] = []
    for node_group_idx, ((node_x_offset, node_d1), node_raw_specs) in enumerate(
        group_source_input_paths(node_specs)
    ):
        node_channels = {int(node_spec["channels"]) for _, node_spec in node_raw_specs}
        if len(node_channels) != 1:
            raise ValueError("each triple backward x group must have one channel width")
        node_channel_count = node_channels.pop()
        node_y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for node_path_idx, node_spec in node_raw_specs:
            node_y_groups[(int(node_spec["y_offset"]), int(node_spec["d2"]))].append(
                (node_path_idx, node_spec)
            )

        node_lines = [
            f"case {node_group_idx}: {{",
            f"    if (u >= {node_channel_count}) break;",
        ]
        for node_i_idx in range(node_d1):
            node_lines.extend(
                [
                    f"    float dx{node_i_idx} = 0.0f;",
                    f"    float drx{node_i_idx} = 0.0f;",
                ]
            )
        node_lines.extend(
            r"""    for (Index p = begin; p < end; ++p) {{
{edge_index}
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};
        const Index go_base = Index(d) * {out_dim};""".format(
                edge_index=(
                    "        const Index e = edge_ids_by_src[p];\n"
                    "        const int d = __ldg(dst + e);"
                ),
                in2_dim=node_in2_dim,
                weight_numel=node_weight_numel,
                out_dim=node_out_dim,
            ).split("\n")
        )

        if node_with_env:
            node_lines.append(f"        const float env_e = __ldg(env + e);")
            if node_need_grad_env:
                node_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
        for (node_y_offset, node_d2), node_group_specs in sorted(
            node_y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            node_lines.append("        {")
            for node_j_idx in range(node_d2):
                node_j_abs = node_y_offset + node_j_idx
                node_lines.extend(
                    r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(
                        j=node_j_idx, j_abs=node_j_abs
                    ).split("\n")
                )

            for node_path_idx, node_spec in sorted(
                node_group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
            ):
                node_out_offset = int(node_spec["out_offset"])
                node_weight_offset = int(node_spec["weight_offset"])
                node_d3 = int(node_spec["d3"])
                node_terms = list(node_spec["terms"])
                node_lines.extend(
                    r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                        weight_offset=node_weight_offset,
                        weight_channel_offset=weight_channel_offset(node_spec),
                    ).split("\n")
                )
                if node_with_env:
                    node_lines[-3:] = emit_triple_bwd_env_weight_loads(
                        node_weight_offset,
                        need_grad_env=node_need_grad_env,
                        channel=weight_channel_offset(node_spec),
                    )
                for node_class_idx, node_signature_class in enumerate(
                    build_signed_ij_classes(
                        node_terms,
                        spec_y_offset=node_y_offset,
                        y_name_base=node_y_offset,
                    )
                ):
                    node_representative = node_signature_class.members[0]
                    node_go_sum = " + ".join(
                        f"""{float_literal(float(node_coeff))} * {output_gradient_load("grad_out", node_out_offset, node_d3, int(node_k_idx), node_path_idx, node_node_seed_offsets, node_l0_dim)}"""
                        for _, _, node_k_idx, node_coeff in node_representative.terms
                    )
                    node_sgo_sum = " + ".join(
                        f"""{float_literal(float(node_coeff))} * {output_gradient_load("sgo", node_out_offset, node_d3, int(node_k_idx), node_path_idx, node_node_seed_offsets, node_l0_dim)}"""
                        for _, _, node_k_idx, node_coeff in node_representative.terms
                    )
                    node_lines.extend(
                        [
                            f"                const float q{node_path_idx}_{node_class_idx} = {node_go_sum};",
                            f"                const float qs{node_path_idx}_{node_class_idx} = {node_sgo_sum};",
                        ]
                    )
                    for node_member in node_signature_class.members:
                        node_op = "+=" if node_member.relative_sign > 0 else "-="
                        node_i_idx = node_member.i_idx
                        node_j_idx = node_member.j_name
                        node_q = f"q{node_path_idx}_{node_class_idx}"
                        node_qs = f"qs{node_path_idx}_{node_class_idx}"
                        node_lines.extend(
                            f"                dx{node_i_idx} {node_op} ((sy{node_j_idx} * rw_value + ry{node_j_idx} * sw_value) * {node_q} + (ry{node_j_idx} * w + y{node_j_idx} * rw_value) * {node_qs});\n                drx{node_i_idx} {node_op} ((sy{node_j_idx} * w + y{node_j_idx} * sw_value) * {node_q} + y{node_j_idx} * w * {node_qs});".split(
                                "\n"
                            )
                        )
                        if node_need_grad_env:
                            node_lines.append(
                                f"                dx{node_i_idx} {node_op} cross_weight * y{node_j_idx} * {node_q};"
                            )
                if node_with_env and node_path_idx in node_l0_offsets:
                    emit_triple_bwd_raw_l0(
                        node_lines,
                        node_spec,
                        node_path_idx,
                        node_l0_offsets,
                        node_l0_dim,
                        {
                            "x": [f"dx{i}" for i in range(int(node_spec["d1"]))],
                            "rx": [f"drx{i}" for i in range(int(node_spec["d1"]))],
                        },
                    )
                node_lines.append("            }")
            node_lines.append("        }")

        node_lines.append("    }")
        for node_i_idx in range(node_d1):
            node_feature = f"{node_x_offset} + u * {node_d1} + {node_i_idx}"
            node_lines.extend(
                [
                    f"    dx_partial[static_cast<int64_t>(segment_idx) * GEN_IN1_DIM + {node_feature}] = dx{node_i_idx};",
                    f"    drx_partial[static_cast<int64_t>(segment_idx) * GEN_IN1_DIM + {node_feature}] = drx{node_i_idx};",
                ]
            )
        node_lines.extend(["    break;", "}"])
        node_branches.append("\n".join(node_lines))
    node_source_branches = "\n".join(node_branches)

    edge_specs = specs
    edge_out_dim = out_dim
    edge_weight_numel = weight_numel
    edge_partial_bases = partial_bases
    edge_with_l0 = with_l0
    edge_with_env = with_env
    edge_need_grad_env = need_grad_env
    edge_need_grad_env = edge_with_env and edge_need_grad_env
    edge_l0_offsets, edge_l0_dim = (
        l0_feature_layout(edge_specs) if edge_with_l0 else ({}, 0)
    )
    edge_node_seed_offsets = {} if edge_with_env else edge_l0_offsets
    edge_branches: list[str] = []
    for edge_group_idx, ((edge_x_offset, edge_d1), edge_raw_specs) in enumerate(
        group_source_input_paths(edge_specs)
    ):
        edge_channels = {int(edge_spec["channels"]) for _, edge_spec in edge_raw_specs}
        if len(edge_channels) != 1:
            raise ValueError("each triple backward x group must have one channel width")
        edge_channel_count = edge_channels.pop()
        edge_reduction_tail = (
            f", min(32, {edge_channel_count} - u + lane)"
            if edge_channel_count % 32
            else ""
        )
        edge_y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for edge_path_idx, edge_spec in edge_raw_specs:
            edge_y_groups[(int(edge_spec["y_offset"]), int(edge_spec["d2"]))].append(
                (edge_path_idx, edge_spec)
            )

        edge_lines = [
            f"case {edge_group_idx}: {{",
            f"    if (u >= {edge_channel_count}) break;",
        ]
        for edge_i_idx in range(edge_d1):
            edge_lines.extend(r"""    const float x{i} = __ldg(x + x_base + {i});
    const float rx{i} = __ldg(rx + x_base + {i});
    const float sx{i} = __ldg(sx + x_base + {i});""".format(i=edge_i_idx).split("\n"))
        edge_lines.extend(
            r"""    for (Index p = begin; p < end; ++p) {{
{edge_index}
        const Index weight_base = e * {weight_numel};
        const Index go_base = Index(d) * {out_dim};""".format(
                edge_index=(
                    "        const Index e = edge_ids_by_src[p];\n"
                    "        const int d = __ldg(dst + e);"
                ),
                weight_numel=edge_weight_numel,
                out_dim=edge_out_dim,
            ).split("\n")
        )

        if edge_with_env:
            edge_lines.append(f"        const float env_e = __ldg(env + e);")
            if edge_need_grad_env:
                edge_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
        for (edge_y_offset, edge_d2), edge_group_specs in sorted(
            edge_y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            edge_lines.append("        {")
            for edge_j_idx in range(edge_d2):
                edge_lines.extend(
                    [
                        f"            float dy{edge_j_idx} = 0.0f;",
                        f"            float dry{edge_j_idx} = 0.0f;",
                    ]
                )

            for edge_path_idx, edge_spec in sorted(
                edge_group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
            ):
                edge_out_offset = int(edge_spec["out_offset"])
                edge_weight_offset = int(edge_spec["weight_offset"])
                edge_d3 = int(edge_spec["d3"])
                edge_terms = list(edge_spec["terms"])
                edge_lines.extend(
                    r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                        weight_offset=edge_weight_offset,
                        weight_channel_offset=weight_channel_offset(edge_spec),
                    ).split("\n")
                )
                if edge_with_env:
                    edge_lines[-3:] = emit_triple_bwd_env_weight_loads(
                        edge_weight_offset,
                        need_grad_env=edge_need_grad_env,
                        channel=weight_channel_offset(edge_spec),
                    )
                if edge_d1 == 1:
                    edge_lines.extend(r"""                const float dy_q = sx0 * rw_value + rx0 * sw_value;
                const float dy_qs = rx0 * w + x0 * rw_value;
                const float dry_q = sx0 * w + x0 * sw_value;
                const float dry_qs = x0 * w;""".split("\n"))
                for edge_class_idx, edge_signature_class in enumerate(
                    build_signed_ij_classes(
                        edge_terms,
                        spec_y_offset=edge_y_offset,
                        y_name_base=edge_y_offset,
                    )
                ):
                    edge_representative = edge_signature_class.members[0]
                    edge_go_sum = " + ".join(
                        f"""{float_literal(float(edge_coeff))} * {output_gradient_load("grad_out", edge_out_offset, edge_d3, int(edge_k_idx), edge_path_idx, edge_node_seed_offsets, edge_l0_dim)}"""
                        for _, _, edge_k_idx, edge_coeff in edge_representative.terms
                    )
                    edge_sgo_sum = " + ".join(
                        f"""{float_literal(float(edge_coeff))} * {output_gradient_load("sgo", edge_out_offset, edge_d3, int(edge_k_idx), edge_path_idx, edge_node_seed_offsets, edge_l0_dim)}"""
                        for _, _, edge_k_idx, edge_coeff in edge_representative.terms
                    )
                    edge_lines.extend(
                        [
                            f"                const float q{edge_path_idx}_{edge_class_idx} = {edge_go_sum};",
                            f"                const float qs{edge_path_idx}_{edge_class_idx} = {edge_sgo_sum};",
                        ]
                    )
                    for edge_member in edge_signature_class.members:
                        edge_op = "+=" if edge_member.relative_sign > 0 else "-="
                        edge_i_idx = edge_member.i_idx
                        edge_j_idx = edge_member.j_name
                        edge_q = f"q{edge_path_idx}_{edge_class_idx}"
                        edge_qs = f"qs{edge_path_idx}_{edge_class_idx}"
                        if edge_d1 == 1:
                            edge_lines.extend(
                                f"                dy{edge_j_idx} {edge_op} dy_q * {edge_q} + dy_qs * {edge_qs};\n                dry{edge_j_idx} {edge_op} dry_q * {edge_q} + dry_qs * {edge_qs};".split(
                                    "\n"
                                )
                            )
                        else:
                            edge_lines.extend(
                                f"                dy{edge_j_idx} {edge_op} ((sx{edge_i_idx} * rw_value + rx{edge_i_idx} * sw_value) * {edge_q} + (rx{edge_i_idx} * w + x{edge_i_idx} * rw_value) * {edge_qs});\n                dry{edge_j_idx} {edge_op} ((sx{edge_i_idx} * w + x{edge_i_idx} * sw_value) * {edge_q} + x{edge_i_idx} * w * {edge_qs});".split(
                                    "\n"
                                )
                            )
                        if edge_need_grad_env:
                            edge_lines.append(
                                f"                dy{edge_j_idx} {edge_op} cross_weight * x{edge_i_idx} * {edge_q};"
                            )
                if edge_with_env and edge_path_idx in edge_l0_offsets:
                    emit_triple_bwd_raw_l0(
                        edge_lines,
                        edge_spec,
                        edge_path_idx,
                        edge_l0_offsets,
                        edge_l0_dim,
                        {
                            "y": [f"dy{j}" for j in range(int(edge_spec["d2"]))],
                            "ry": [f"dry{j}" for j in range(int(edge_spec["d2"]))],
                        },
                    )
                edge_lines.append("            }")

            for edge_j_idx in range(edge_d2):
                edge_j_abs = edge_y_offset + edge_j_idx
                edge_lines.extend(
                    r"""            const float dy_sum{j} = warp_reduce_sum(dy{j}{reduction_tail});
            const float dry_sum{j} = warp_reduce_sum(dry{j}{reduction_tail});
            if (lane == 0) {{
                const int64_t partial = static_cast<int64_t>({partial_bases_edge_group}) + blockIdx.z * (blockDim.x >> 5) + warp_id;
                const int64_t offset = (partial * num_edges + e) * GEN_IN2_DIM + {j_abs};
                dy_partial[offset] = dy_sum{j};
                dry_partial[offset] = dry_sum{j};
            }}""".format(
                        j=edge_j_idx,
                        reduction_tail=edge_reduction_tail,
                        partial_bases_edge_group=edge_partial_bases[edge_group_idx],
                        j_abs=edge_j_abs,
                    ).split("\n")
                )
            edge_lines.append("        }")

        edge_lines.extend(["    }", "    break;", "}"])
        edge_branches.append("\n".join(edge_lines))
    y_source_branches = "\n".join(edge_branches)

    weight_branches = emit_triple_bwd_weight_cases(
        specs,
        in2_dim=in2_dim,
        out_dim=out_dim,
        weight_numel=weight_numel,
        per_edge=variable_channels,
        outputs="dw",
        with_l0=with_l0,
        with_env=with_env,
        need_grad_env=need_grad_env,
        env_partial_bases=partial_bases,
    )
    drw_branches = emit_triple_bwd_weight_cases(
        specs,
        in2_dim=in2_dim,
        out_dim=out_dim,
        weight_numel=weight_numel,
        per_edge=variable_channels,
        outputs="drw",
        with_l0=with_l0,
        with_env=with_env,
        need_grad_env=need_grad_env,
        env_partial_bases=partial_bases,
    )
    paired_weight_groups = {
        group_idx
        for group_idx, ((_, d1), _) in enumerate(groups)
        if variable_channels and d1 <= 5
    }
    paired_weight_branches = emit_triple_bwd_weight_cases(
        specs,
        in2_dim=in2_dim,
        out_dim=out_dim,
        weight_numel=weight_numel,
        per_edge=variable_channels,
        outputs="both",
        with_l0=with_l0,
        with_env=with_env,
        need_grad_env=need_grad_env,
        env_partial_bases=partial_bases,
    )
    paired_node_y_branches = ""
    if paired_weight_groups:
        source_edge_specs = specs
        source_edge_in2_dim = in2_dim
        source_edge_out_dim = out_dim
        source_edge_weight_numel = weight_numel
        source_edge_partial_bases = partial_bases
        emit_weight_grads = False
        source_edge_output_role = "all"
        source_edge_tasks = None
        source_edge_with_l0 = with_l0
        source_edge_with_env = with_env
        source_edge_need_grad_env = need_grad_env
        source_edge_env_partial_bases = None
        source_edge_need_grad_env = source_edge_with_env and source_edge_need_grad_env
        source_edge_l0_offsets, source_edge_l0_dim = (
            l0_feature_layout(source_edge_specs) if source_edge_with_l0 else ({}, 0)
        )
        source_edge_seed_offsets = (
            {} if source_edge_with_env else source_edge_l0_offsets
        )
        split_source_partials = source_edge_tasks is not None
        if source_edge_tasks is None:
            source_edge_tasks = [
                (source_edge_group_idx, 0, source_input_group, source_edge_group_specs)
                for source_edge_group_idx, (
                    source_input_group,
                    source_edge_group_specs,
                ) in enumerate(group_source_input_paths(source_edge_specs))
            ]
        source_edge_branches: list[str] = []
        for source_edge_task_idx, (
            source_edge_group_idx,
            source_partial_idx,
            (source_edge_x_offset, source_edge_d1),
            source_edge_group_specs,
        ) in enumerate(source_edge_tasks):
            # Primal, output, and factor state use about nine doubles per component.
            split_source_edge_roles = (
                source_edge_output_role != "all"
                and 72
                * (
                    source_edge_d1
                    + max(
                        int(source_edge_spec["d2"])
                        for _, source_edge_spec in source_edge_group_specs
                    )
                )
                > 1024
            )
            if source_edge_output_role == "y" and not split_source_edge_roles:
                continue
            source_edge_task_role = (
                source_edge_output_role if split_source_edge_roles else "all"
            )
            emit_source_grads = source_edge_task_role != "y"
            emit_edge_grads = source_edge_task_role != "x"
            source_edge_channel_counts = {
                int(source_edge_spec["channels"])
                for _, source_edge_spec in source_edge_group_specs
            }
            if len(source_edge_channel_counts) != 1:
                raise ValueError(
                    "each triple backward x group must have one channel width"
                )
            source_edge_channels = source_edge_channel_counts.pop()
            source_edge_warp_tail = (
                f", min(32, {source_edge_channels} - u + lane)"
                if source_edge_channels % 32
                else ""
            )
            source_edge_y_groups: dict[
                tuple[int, int], list[tuple[int, dict[str, object]]]
            ] = defaultdict(list)
            for source_edge_path_idx, source_edge_spec in source_edge_group_specs:
                source_edge_y_groups[
                    (int(source_edge_spec["y_offset"]), int(source_edge_spec["d2"]))
                ].append((source_edge_path_idx, source_edge_spec))

            source_edge_branch_idx = (
                source_edge_task_idx if split_source_partials else source_edge_group_idx
            )
            source_edge_lines = [
                f"case {source_edge_branch_idx}: {{",
                f"    if (u >= {source_edge_channels}) break;",
            ]
            for source_component_idx in range(source_edge_d1):
                source_edge_lines.extend(
                    [
                        *(
                            [
                                f"    float dx{source_component_idx} = 0.0f;",
                                f"    float drx{source_component_idx} = 0.0f;",
                            ]
                            if emit_source_grads
                            else []
                        ),
                        *(
                            r"""    const float x{i} = __ldg(x + x_base + {i});
    const float rx{i} = __ldg(rx + x_base + {i});
    const float sx{i} = __ldg(sx + x_base + {i});""".format(
                                i=source_component_idx
                            ).split("\n")
                            if source_edge_task_role != "x"
                            else []
                        ),
                    ]
                )
            source_edge_lines.extend(
                r"""    for (Index p = begin; p < end; ++p) {{
{edge_index}
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};
        const Index go_base = Index(d) * {out_dim};""".format(
                    edge_index=(
                        "        const Index e = edge_ids_by_src[p];\n"
                        "        const int d = __ldg(dst + e);"
                    ),
                    in2_dim=source_edge_in2_dim,
                    weight_numel=source_edge_weight_numel,
                    out_dim=source_edge_out_dim,
                ).split("\n")
            )

            if source_edge_with_env:
                source_edge_lines.append(f"        const float env_e = __ldg(env + e);")
                if source_edge_need_grad_env:
                    source_edge_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
                if (
                    source_edge_need_grad_env
                    and emit_weight_grads
                    and emit_source_grads
                ):
                    source_edge_lines.extend(
                        [
                            "        float de_value = 0.0f;",
                            "        float dre_value = 0.0f;",
                        ]
                    )
            for (
                source_edge_y_offset,
                source_edge_d2,
            ), source_edge_path_specs in sorted(
                source_edge_y_groups.items(),
                key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
                reverse=True,
            ):
                source_edge_lines.append("        {")
                for edge_component_idx in range(source_edge_d2):
                    edge_feature_idx = source_edge_y_offset + edge_component_idx
                    source_edge_lines.extend(
                        [
                            *(
                                r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(
                                    j=edge_component_idx, j_abs=edge_feature_idx
                                ).split("\n")
                                if emit_source_grads
                                else []
                            ),
                            *(
                                [
                                    f"            float dy{edge_component_idx} = 0.0f;",
                                    f"            float dry{edge_component_idx} = 0.0f;",
                                ]
                                if emit_edge_grads
                                else []
                            ),
                        ]
                    )

                for source_edge_path_idx, source_edge_spec in sorted(
                    source_edge_path_specs,
                    key=lambda path: len(path[1]["terms"]),
                    reverse=True,
                ):
                    source_edge_out_offset = int(source_edge_spec["out_offset"])
                    source_edge_weight_offset = int(source_edge_spec["weight_offset"])
                    source_edge_d3 = int(source_edge_spec["d3"])
                    source_edge_terms = list(source_edge_spec["terms"])
                    source_edge_lines.extend(
                        r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                            weight_offset=source_edge_weight_offset,
                            weight_channel_offset=weight_channel_offset(
                                source_edge_spec
                            ),
                        ).split("\n")
                    )
                    if source_edge_with_env:
                        source_edge_lines[-3:] = emit_triple_bwd_env_weight_loads(
                            source_edge_weight_offset,
                            need_grad_env=source_edge_need_grad_env,
                            channel=weight_channel_offset(source_edge_spec),
                        )
                        if (
                            source_edge_need_grad_env
                            and emit_weight_grads
                            and emit_source_grads
                        ):
                            source_edge_lines.extend(
                                [
                                    "                float gv = 0.0f;",
                                    "                float hv = 0.0f;",
                                ]
                            )
                    if emit_weight_grads:
                        source_edge_lines.extend(
                            [
                                "                float dw_value = 0.0f;",
                                "                float drw_value = 0.0f;",
                            ]
                        )
                        # Accumulate contractions before applying path weights.
                        # The same contractions also produce both weight outputs.
                        source_edge_lines.extend(
                            f"                float {factor_name}{source_component_idx} = 0.0f;"
                            for source_component_idx in range(
                                source_edge_d1 if emit_source_grads else 0
                            )
                            for factor_name in ("x_mix", "x_ryq", "x_ryqs", "x_yq")
                        )
                        source_edge_lines.extend(
                            f"                float {factor_name}{edge_component_idx} = 0.0f;"
                            for edge_component_idx in range(
                                source_edge_d2 if emit_edge_grads else 0
                            )
                            for factor_name in ("y_mix", "y_rxq", "y_rxqs", "y_xq")
                        )
                    for source_edge_cg_class_idx, source_edge_cg_class in enumerate(
                        build_signed_ij_classes(
                            source_edge_terms,
                            spec_y_offset=source_edge_y_offset,
                            y_name_base=source_edge_y_offset,
                        )
                    ):
                        source_edge_cg_representative = source_edge_cg_class.members[0]
                        source_edge_grad_out_sum = " + ".join(
                            f"""{float_literal(float(source_edge_cg_coeff))} * {output_gradient_load("grad_out", source_edge_out_offset, source_edge_d3, int(output_component_idx), source_edge_path_idx, source_edge_seed_offsets, source_edge_l0_dim)}"""
                            for _, _, output_component_idx, source_edge_cg_coeff in source_edge_cg_representative.terms
                        )
                        source_edge_sgo_sum = " + ".join(
                            f"""{float_literal(float(source_edge_cg_coeff))} * {output_gradient_load("sgo", source_edge_out_offset, source_edge_d3, int(output_component_idx), source_edge_path_idx, source_edge_seed_offsets, source_edge_l0_dim)}"""
                            for _, _, output_component_idx, source_edge_cg_coeff in source_edge_cg_representative.terms
                        )
                        source_edge_lines.extend(
                            [
                                f"                const float q{source_edge_path_idx}_{source_edge_cg_class_idx} = {source_edge_grad_out_sum};",
                                f"                const float qs{source_edge_path_idx}_{source_edge_cg_class_idx} = {source_edge_sgo_sum};",
                            ]
                        )
                        for source_edge_cg_member in source_edge_cg_class.members:
                            source_edge_cg_op = (
                                "+="
                                if source_edge_cg_member.relative_sign > 0
                                else "-="
                            )
                            source_component_idx = source_edge_cg_member.i_idx
                            edge_component_idx = source_edge_cg_member.j_name
                            source_edge_q = (
                                f"q{source_edge_path_idx}_{source_edge_cg_class_idx}"
                            )
                            source_edge_qs = (
                                f"qs{source_edge_path_idx}_{source_edge_cg_class_idx}"
                            )
                            if emit_weight_grads:
                                source_edge_lines.extend(
                                    [
                                        *(
                                            r"""                x_mix{i} {op} sy{j} * {q} + y{j} * {qs};
                x_ryq{i} {op} ry{j} * {q};
                x_ryqs{i} {op} ry{j} * {qs};
                x_yq{i} {op} y{j} * {q};""".format(
                                                i=source_component_idx,
                                                op=source_edge_cg_op,
                                                j=edge_component_idx,
                                                q=source_edge_q,
                                                qs=source_edge_qs,
                                            ).split("\n")
                                            if emit_source_grads
                                            else []
                                        ),
                                        *(
                                            r"""                y_mix{j} {op} sx{i} * {q} + x{i} * {qs};
                y_rxq{j} {op} rx{i} * {q};
                y_rxqs{j} {op} rx{i} * {qs};
                y_xq{j} {op} x{i} * {q};""".format(
                                                j=edge_component_idx,
                                                op=source_edge_cg_op,
                                                i=source_component_idx,
                                                q=source_edge_q,
                                                qs=source_edge_qs,
                                            ).split("\n")
                                            if emit_edge_grads
                                            else []
                                        ),
                                    ]
                                )
                            else:
                                source_edge_lines.extend(
                                    r"""                dx{i} {op} ((sy{j} * rw_value + ry{j} * sw_value) * {q} + (ry{j} * w + y{j} * rw_value) * {qs});
                drx{i} {op} ((sy{j} * w + y{j} * sw_value) * {q} + y{j} * w * {qs});
                dy{j} {op} ((sx{i} * rw_value + rx{i} * sw_value) * {q} + (rx{i} * w + x{i} * rw_value) * {qs});
                dry{j} {op} ((sx{i} * w + x{i} * sw_value) * {q} + x{i} * w * {qs});""".format(
                                        i=source_component_idx,
                                        op=source_edge_cg_op,
                                        j=edge_component_idx,
                                        q=source_edge_q,
                                        qs=source_edge_qs,
                                    ).split("\n")
                                )
                                if source_edge_need_grad_env:
                                    source_edge_lines.extend(
                                        f"                dx{source_component_idx} {source_edge_cg_op} cross_weight * y{edge_component_idx} * {source_edge_q};\n                dy{edge_component_idx} {source_edge_cg_op} cross_weight * x{source_component_idx} * {source_edge_q};".split(
                                            "\n"
                                        )
                                    )
                    if emit_weight_grads:
                        for source_component_idx in range(
                            source_edge_d1 if emit_source_grads else 0
                        ):
                            if source_edge_task_role == "x":
                                source_edge_lines.extend(
                                    r"""                {{
                const float x{i} = __ldg(x + x_base + {i});
                const float rx{i} = __ldg(rx + x_base + {i});
                const float sx{i} = __ldg(sx + x_base + {i});""".format(
                                        i=source_component_idx
                                    ).split("\n")
                                )
                            source_edge_lines.extend(
                                r"""                dx{i} += rw_value * x_mix{i} + sw_value * x_ryq{i} + w * x_ryqs{i};
                drx{i} += w * x_mix{i} + sw_value * x_yq{i};
                dw_value += rx{i} * x_mix{i} + sx{i} * x_ryq{i} + x{i} * x_ryqs{i};
                drw_value += x{i} * x_mix{i} + sx{i} * x_yq{i};""".format(
                                    i=source_component_idx
                                ).split("\n")
                            )
                            if source_edge_need_grad_env:
                                source_edge_lines.extend(
                                    r"""                dx{i} += cross_weight * x_yq{i};
                gv += x{i} * x_yq{i};
                hv += rx{i} * x_yq{i} + x{i} * x_ryq{i};""".format(
                                        i=source_component_idx
                                    ).split("\n")
                                )
                            if source_edge_task_role == "x":
                                source_edge_lines.append("                }")
                        for edge_component_idx in range(
                            source_edge_d2 if emit_edge_grads else 0
                        ):
                            source_edge_lines.extend(
                                f"                dy{edge_component_idx} += rw_value * y_mix{edge_component_idx} + sw_value * y_rxq{edge_component_idx} + w * y_rxqs{edge_component_idx};\n                dry{edge_component_idx} += w * y_mix{edge_component_idx} + sw_value * y_xq{edge_component_idx};".split(
                                    "\n"
                                )
                            )
                            if source_edge_need_grad_env:
                                source_edge_lines.append(
                                    f"                dy{edge_component_idx} += cross_weight * y_xq{edge_component_idx};"
                                )
                        if emit_source_grads:
                            if source_edge_need_grad_env:
                                source_edge_lines.extend(
                                    r"""                de_value += raw_w * dw_value + raw_rw * drw_value + raw_sw * hv;
                dre_value += raw_w * drw_value + raw_sw * gv;
                dw_value = env_e * dw_value + re_e * drw_value + se_e * hv;
                drw_value = env_e * drw_value + se_e * gv;""".split("\n")
                                )
                            elif source_edge_with_env:
                                source_edge_lines.extend(
                                    f"                dw_value *= env_e;\n                drw_value *= env_e;".split(
                                        "\n"
                                    )
                                )
                            if (
                                source_edge_with_env
                                and source_edge_path_idx in source_edge_l0_offsets
                            ):
                                emit_triple_bwd_raw_l0(
                                    source_edge_lines,
                                    source_edge_spec,
                                    source_edge_path_idx,
                                    source_edge_l0_offsets,
                                    source_edge_l0_dim,
                                    {"w": "dw_value", "rw": "drw_value"},
                                )
                            source_edge_lines.extend(
                                f"                store_global_cs_f32(dw + weight_base + {source_edge_weight_offset} + {weight_channel_offset(source_edge_spec)}, dw_value);\n                store_global_cs_f32(drw + weight_base + {source_edge_weight_offset} + {weight_channel_offset(source_edge_spec)}, drw_value);".split(
                                    "\n"
                                )
                            )
                    if (
                        source_edge_with_env
                        and source_edge_path_idx in source_edge_l0_offsets
                    ):
                        emit_triple_bwd_raw_l0(
                            source_edge_lines,
                            source_edge_spec,
                            source_edge_path_idx,
                            source_edge_l0_offsets,
                            source_edge_l0_dim,
                            {
                                **(
                                    {
                                        "x": [
                                            f"dx{i}"
                                            for i in range(int(source_edge_spec["d1"]))
                                        ],
                                        "rx": [
                                            f"drx{i}"
                                            for i in range(int(source_edge_spec["d1"]))
                                        ],
                                    }
                                    if emit_source_grads
                                    else {}
                                ),
                                **(
                                    {
                                        "y": [
                                            f"dy{j}"
                                            for j in range(int(source_edge_spec["d2"]))
                                        ],
                                        "ry": [
                                            f"dry{j}"
                                            for j in range(int(source_edge_spec["d2"]))
                                        ],
                                    }
                                    if emit_edge_grads
                                    else {}
                                ),
                            },
                        )
                    source_edge_lines.append("            }")

                for edge_component_idx in range(
                    source_edge_d2 if emit_edge_grads else 0
                ):
                    edge_feature_idx = source_edge_y_offset + edge_component_idx
                    source_edge_lines.extend(
                        r"""            const float dy_sum{j} = warp_reduce_sum(dy{j}{reduction_tail});
            const float dry_sum{j} = warp_reduce_sum(dry{j}{reduction_tail});
            if (lane == 0) {{
                const int64_t partial = static_cast<int64_t>({edge_partial_base}) + blockIdx.z * (blockDim.x >> 5) + warp_id;
                const int64_t offset = (partial * num_edges + e) * GEN_IN2_DIM + {j_abs};
                dy_partial[offset] = dy_sum{j};
                dry_partial[offset] = dry_sum{j};
            }}""".format(
                            j=edge_component_idx,
                            reduction_tail=source_edge_warp_tail,
                            edge_partial_base=source_edge_partial_bases[
                                source_edge_group_idx
                            ],
                            j_abs=edge_feature_idx,
                        ).split("\n")
                    )
                source_edge_lines.append("        }")

            if source_edge_need_grad_env and emit_weight_grads and emit_source_grads:
                source_edge_env_base = source_edge_env_partial_bases[
                    source_edge_task_idx
                ]
                source_edge_lines.extend(
                    r"""        const float de_sum = warp_reduce_sum(de_value{reduction_tail});
        const float dre_sum = warp_reduce_sum(dre_value{reduction_tail});
        if (lane == 0) {{
            const Index part = {env_base} + blockIdx.z * (blockDim.x >> 5) + warp_id;
            de_partial[Index(e) * GEN_ENV_PARTS + part] = de_sum;
            dre_partial[Index(e) * GEN_ENV_PARTS + part] = dre_sum;
        }}""".format(
                        reduction_tail=source_edge_warp_tail,
                        env_base=source_edge_env_base,
                    ).split("\n")
                )
            source_edge_lines.append("    }")
            for source_component_idx in range(
                source_edge_d1 if emit_source_grads else 0
            ):
                source_grad_feature = f"{source_edge_x_offset} + u * {source_edge_d1} + {source_component_idx}"
                source_partial_offset = (
                    f"(static_cast<int64_t>(segment_idx) * GEN_NODE_PARTS + {source_partial_idx})"
                    if split_source_partials
                    else "static_cast<int64_t>(segment_idx)"
                )
                source_edge_lines.extend(
                    [
                        f"    dx_partial[{source_partial_offset} * GEN_IN1_DIM + {source_grad_feature}] = dx{source_component_idx};",
                        f"    drx_partial[{source_partial_offset} * GEN_IN1_DIM + {source_grad_feature}] = drx{source_component_idx};",
                    ]
                )
            source_edge_lines.extend(["    break;", "}"])
            source_edge_branches.append("\n".join(source_edge_lines))
        paired_node_y_branches = "\n".join(source_edge_branches)

    output_specs = specs
    output_in1_dim = in1_dim
    output_in2_dim = in2_dim
    output_out_dim = out_dim
    output_weight_numel = weight_numel
    output_scalar_bytes = scalar_bytes
    output_with_l0 = with_l0
    output_with_env = with_env
    output_need_grad_env = need_grad_env
    output_need_grad_env = output_with_env and output_need_grad_env
    _, output_output_overlap = flat_coverage(
        output_specs, output_out_dim, "out_offset", "d3"
    )
    output_cases: list[str] = []
    for output_group_idx, ((output_x_offset, output_d1), output_raw_specs) in enumerate(
        group_source_input_paths(output_specs)
    ):
        output_channels = {
            int(output_spec["channels"]) for _, output_spec in output_raw_specs
        }
        if len(output_channels) != 1:
            raise ValueError(
                "each triple backward hgo group must have one channel width"
            )
        output_channel_count = output_channels.pop()
        output_y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for output_path_idx, output_spec in output_raw_specs:
            output_y_groups[
                (int(output_spec["y_offset"]), int(output_spec["d2"]))
            ].append((output_path_idx, output_spec))
        output_ordered_specs = sorted(
            output_raw_specs, key=lambda path: len(path[1]["terms"]), reverse=True
        )
        output_lines = [
            f"case {output_group_idx}: {{",
            f"    if (u >= {output_channel_count}) break;",
        ]
        for output_path_idx, output_spec in output_ordered_specs:
            for output_k_idx in range(int(output_spec["d3"])):
                output_lines.append(
                    f"    float acc_p{output_path_idx}_{output_k_idx} = 0.0f;"
                )
        output_lines.extend(
            r"""    const Index begin = rowptr[t];
    const Index end = rowptr[t + 1];
{edge_loop}
        const Index x_base = Index(s) * {in1_dim} + {x_offset} + u * {d1};
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};""".format(
                edge_loop=(
                    "    for (Index e = begin; e < end; ++e) {\n"
                    "        const int s = __ldg(src + e);"
                ),
                in1_dim=output_in1_dim,
                x_offset=output_x_offset,
                d1=output_d1,
                in2_dim=output_in2_dim,
                weight_numel=output_weight_numel,
            ).split("\n")
        )
        for output_i_idx in range(output_d1):
            output_lines.extend(
                r"""        const float x{i} = __ldg(x + x_base + {i});
        const float rx{i} = __ldg(rx + x_base + {i});
        const float sx{i} = __ldg(sx + x_base + {i});""".format(i=output_i_idx).split("\n")
            )
        if output_with_env:
            output_lines.append(f"        const float env_e = __ldg(env + e);")
            if output_need_grad_env:
                output_lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
        for (output_y_offset, output_d2), output_group_specs in sorted(
            output_y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            output_lines.append("        {")
            for output_j_idx in range(output_d2):
                output_j_abs = output_y_offset + output_j_idx
                output_lines.extend(
                    r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(
                        j=output_j_idx, j_abs=output_j_abs
                    ).split("\n")
                )
            for output_path_idx, output_spec in sorted(
                output_group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
            ):
                output_weight_offset = int(output_spec["weight_offset"])
                output_terms = list(output_spec["terms"])
                output_lines.extend(
                    r"""            {{
                const float w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float rw_value = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float sw_value = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});""".format(
                        weight_offset=output_weight_offset,
                        weight_channel_offset=weight_channel_offset(output_spec),
                    ).split("\n")
                )
                if output_with_env:
                    output_lines[-3:] = emit_triple_bwd_env_weight_loads(
                        output_weight_offset,
                        need_grad_env=output_need_grad_env,
                        channel=weight_channel_offset(output_spec),
                    )
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_lines.append("                float scalar_value = 0.0f;")
                output_scalar_first = True
                for output_class_idx, output_signature_class in enumerate(
                    build_signed_ij_classes(
                        output_terms,
                        spec_y_offset=output_y_offset,
                        y_name_base=output_y_offset,
                    )
                ):
                    output_representative = output_signature_class.members[0]

                    output_i_idx = output_representative.i_idx
                    output_j_idx = output_representative.j_name
                    output_extra = (
                        f" + cross_weight * x{output_i_idx} * y{output_j_idx}"
                        if output_need_grad_env
                        else ""
                    )
                    if output_scalar_bytes == 8 and output_d1 < output_d2:
                        output_mixed_product_value = (
                            f"ry{output_j_idx} * (sx{output_i_idx} * w + x{output_i_idx} * sw_value) + sy{output_j_idx} * (rx{output_i_idx} * w + x{output_i_idx} * rw_value) + y{output_j_idx} * (sx{output_i_idx} * rw_value + rx{output_i_idx} * sw_value)"
                            + output_extra
                        )
                    else:
                        output_mixed_product_value = (
                            f"sx{output_i_idx} * (ry{output_j_idx} * w + y{output_j_idx} * rw_value) + rx{output_i_idx} * (sy{output_j_idx} * w + y{output_j_idx} * sw_value) + x{output_i_idx} * (sy{output_j_idx} * rw_value + ry{output_j_idx} * sw_value)"
                            + output_extra
                        )
                    output_lines.append(
                        f"                float p{output_class_idx} = {output_mixed_product_value};"
                    )
                    for output_member in output_signature_class.members[1:]:
                        output_op = "+=" if output_member.relative_sign > 0 else "-="
                        output_i_idx = output_member.i_idx
                        output_j_idx = output_member.j_name
                        output_extra = (
                            f" + cross_weight * x{output_i_idx} * y{output_j_idx}"
                            if output_need_grad_env
                            else ""
                        )
                        if output_scalar_bytes == 8 and output_d1 < output_d2:
                            output_mixed_product_value = (
                                f"ry{output_j_idx} * (sx{output_i_idx} * w + x{output_i_idx} * sw_value) + sy{output_j_idx} * (rx{output_i_idx} * w + x{output_i_idx} * rw_value) + y{output_j_idx} * (sx{output_i_idx} * rw_value + rx{output_i_idx} * sw_value)"
                                + output_extra
                            )
                        else:
                            output_mixed_product_value = (
                                f"sx{output_i_idx} * (ry{output_j_idx} * w + y{output_j_idx} * rw_value) + rx{output_i_idx} * (sy{output_j_idx} * w + y{output_j_idx} * sw_value) + x{output_i_idx} * (sy{output_j_idx} * rw_value + ry{output_j_idx} * sw_value)"
                                + output_extra
                            )
                        output_lines.append(
                            f"                p{output_class_idx} {output_op} {output_mixed_product_value};"
                        )
                    for _, _, output_k_idx, output_coeff in output_representative.terms:
                        output_lines.append(
                            f"                acc_p{output_path_idx}_{int(output_k_idx)} += {float_literal(float(output_coeff))} * p{output_class_idx};"
                        )
                        if output_with_l0 and int(output_spec["d3"]) == 1:
                            output_scalar_op = "=" if output_scalar_first else "+="
                            output_lines.append(
                                f"                scalar_value {output_scalar_op} {float_literal(float(output_coeff))} * p{output_class_idx};"
                            )
                            output_scalar_first = False
                if output_with_l0 and int(output_spec["d3"]) == 1:
                    output_offsets, output_width = l0_feature_layout(output_specs)
                    if output_with_env:
                        output_lines.append("                scalar_value = 0.0f;")
                        for output_i, output_j, output__k, output_coeff in output_spec[
                            "terms"
                        ]:
                            output_lines.append(
                                f"                scalar_value += {float_literal(float(output_coeff))} * (sx{output_i} * (ry{output_j} * raw_w + y{output_j} * raw_rw) + rx{output_i} * (sy{output_j} * raw_w + y{output_j} * raw_sw) + x{output_i} * (sy{output_j} * raw_rw + ry{output_j} * raw_sw));"
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
        output_lines.append(f"    const Index dgo_base = Index(t) * {output_out_dim};")
        for output_path_idx, output_spec in output_ordered_specs:
            output_out_offset = int(output_spec["out_offset"])
            output_d3 = int(output_spec["d3"])
            for output_k_idx in range(output_d3):
                output_address = (
                    f"dgo_base + {output_out_offset} + u * {output_d3} + {output_k_idx}"
                )
                output_value = f"acc_p{output_path_idx}_{output_k_idx}"
                if output_output_overlap:
                    output_lines.append(
                        f"    atomicAdd(dgo + {output_address}, {output_value});"
                    )
                else:
                    output_lines.append(f"    dgo[{output_address}] = {output_value};")
        output_lines.extend(["    break;", "}"])
        output_cases.append("\n".join(output_lines))
    hgo_cases = "\n".join(output_cases)
    weight_x_cases = "\n".join(
        f"    case {group_idx}: x_base = Index(s_node) * GEN_IN1_DIM + {x_offset} + u * {d1}; break;"
        for group_idx, ((x_offset, d1), _) in enumerate(groups)
    )
    if variable_channels:
        weight_graph_arguments = """    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,"""
        weight_indices = (
            "    const Index e = blockIdx.x;\n"
            "    const int s_node = __ldg(src + e);\n"
            "    const int d = __ldg(dst + e);"
        )
    else:
        weight_graph_arguments = r"""    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ edge_ids_by_src,
    const int32_t* __restrict__ segment_nodes,
    const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end,
    const EdgeIndex* __restrict__ segment_order,"""
        weight_indices = """    const Index segment_idx = segment_order[blockIdx.x];
    const int s_node = segment_nodes[segment_idx];
    const Index begin = segment_begin[segment_idx];
    const Index end = segment_end[segment_idx];"""
    weight_kernel = r"""
template <int GROUP, bool DRW>
__global__ void lunatp_uvuconv_triple_bwd_weight_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sgo,
{weight_graph_arguments}
    float* __restrict__ out) {{
{weight_indices}
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    constexpr int group = GROUP;
    Index x_base = 0;
    switch (group) {{
{weight_x_cases}
    default: return;
    }}
    if constexpr (DRW) {{
        switch (group) {{
{drw_branches}
        default: break;
        }}
    }} else {{
        switch (group) {{
{weight_branches}
        default: break;
        }}
    }}
}}

template <int GROUP>
__global__ void lunatp_uvuconv_triple_bwd_weight_pair_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sgo,
{weight_graph_arguments}
    float* __restrict__ dw,
    float* __restrict__ drw) {{
{weight_indices}
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    constexpr int group = GROUP;
    Index x_base = 0;
    switch (group) {{
{weight_x_cases}
    default: return;
    }}
    switch (group) {{
{paired_weight_branches}
    default: break;
    }}
}}
""".format(
        weight_graph_arguments=weight_graph_arguments,
        weight_indices=weight_indices,
        weight_x_cases=weight_x_cases,
        drw_branches=drw_branches,
        weight_branches=weight_branches,
        paired_weight_branches=paired_weight_branches,
    )
    paired_node_y_kernel = (
        r"""
template <int GROUP>
__global__ void lunatp_uvuconv_triple_bwd_node_y_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sw,
    const float* __restrict__ sgo,
    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ edge_ids_by_src,
    const int32_t* __restrict__ segment_nodes,
    const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end,
    const EdgeIndex* __restrict__ segment_order,
    float* __restrict__ dx_partial,
    float* __restrict__ drx_partial,
    float* __restrict__ dy_partial,
    float* __restrict__ dry_partial,
    int64_t num_edges) {{
    const Index segment_idx = segment_order[blockIdx.x];
    const int s_node = segment_nodes[segment_idx];
    const Index begin = segment_begin[segment_idx];
    const Index end = segment_end[segment_idx];
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int group = GROUP;
    Index x_base = 0;
    switch (group) {{
{weight_x_cases}
    default: return;
    }}
    switch (group) {{
{paired_node_y_branches}
    default: break;
    }}
}}
""".format(
            weight_x_cases=weight_x_cases,
            paired_node_y_branches=paired_node_y_branches,
        )
        if paired_weight_groups
        else ""
    )

    launches: list[str] = []
    paired_launches: list[str] = []
    for group_idx, (threads, channel_tiles) in enumerate(schedules):
        if group_idx in paired_weight_groups:
            node_y_launch = r"""
    lunatp_uvuconv_triple_bwd_node_y_fp32<{group_idx}><<<
        dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
        state.streams[{group_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(),
        grad_out.data_ptr<float>(), rx.data_ptr<float>(), ry.data_ptr<float>(),
        rw.data_ptr<float>(), sx.data_ptr<float>(), sy.data_ptr<float>(),
        sw.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
        edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
        segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
        segment_order.data_ptr<EdgeIndex>(), dx_partial.data_ptr<float>(),
        drx_partial.data_ptr<float>(), dy_partial.data_ptr<float>(),
        dry_partial.data_ptr<float>(), num_edges);
    cudaEventRecord(state.done[{group_idx}], state.streams[{group_idx}].stream());
    cudaEventRecord(state.done[GEN_NUM_GROUPS + {group_idx}],
                    state.streams[{group_idx}].stream());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
        else:
            node_y_launch = r"""
    lunatp_uvuconv_triple_bwd_node_source_fp32<{group_idx}><<<
        dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
        state.streams[{group_idx}].stream()>>>(
        y.data_ptr<float>(), weight.data_ptr<float>(), grad_out.data_ptr<float>(),
        ry.data_ptr<float>(), rw.data_ptr<float>(), sy.data_ptr<float>(),
        sw.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
        edge_ids_by_src.data_ptr<EdgeIndex>(), segment_begin.data_ptr<EdgeIndex>(),
        segment_end.data_ptr<EdgeIndex>(),
        segment_order.data_ptr<EdgeIndex>(), dx_partial.data_ptr<float>(),
        drx_partial.data_ptr<float>());
    cudaEventRecord(state.done[{group_idx}], state.streams[{group_idx}].stream());

    lunatp_uvuconv_triple_bwd_y_source_fp32<{group_idx}><<<
        dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
        state.streams[GEN_NUM_GROUPS + {group_idx}].stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), grad_out.data_ptr<float>(),
        rx.data_ptr<float>(), rw.data_ptr<float>(), sx.data_ptr<float>(),
        sw.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
        edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
        segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
        segment_order.data_ptr<EdgeIndex>(), dy_partial.data_ptr<float>(),
        dry_partial.data_ptr<float>(), num_edges);
    cudaEventRecord(state.done[GEN_NUM_GROUPS + {group_idx}],
                    state.streams[GEN_NUM_GROUPS + {group_idx}].stream());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
        if group_idx in paired_weight_groups:
            weight_launch = r"""
    lunatp_uvuconv_triple_bwd_weight_pair_fp32<{group_idx}><<<
        dim3(static_cast<unsigned int>(num_edges), 1, {channel_tiles}), {threads}, 0,
        state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
        sy.data_ptr<float>(), sgo.data_ptr<float>(), src.data_ptr<int32_t>(),
        dst.data_ptr<int32_t>(), dw.data_ptr<float>(), drw.data_ptr<float>());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
            drw_launch = f"\n    cudaEventRecord(state.done[4 * GEN_NUM_GROUPS + {group_idx}],\n                    state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream());\n"
        elif variable_channels:
            weight_launch = r"""
    lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, false><<<
        dim3(static_cast<unsigned int>(num_edges), 1, {channel_tiles}), {threads}, 0,
        state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
        sy.data_ptr<float>(), sgo.data_ptr<float>(), src.data_ptr<int32_t>(),
        dst.data_ptr<int32_t>(), dw.data_ptr<float>());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
            drw_launch = r"""
    lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, true><<<
        dim3(static_cast<unsigned int>(num_edges), 1, {channel_tiles}), {threads}, 0,
        state.streams[4 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
        sy.data_ptr<float>(), sgo.data_ptr<float>(), src.data_ptr<int32_t>(),
        dst.data_ptr<int32_t>(), drw.data_ptr<float>());
    cudaEventRecord(state.done[4 * GEN_NUM_GROUPS + {group_idx}],
                    state.streams[4 * GEN_NUM_GROUPS + {group_idx}].stream());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
        else:
            pair_weight_call = r"""
        lunatp_uvuconv_triple_bwd_weight_pair_fp32<{group_idx}><<<
            dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
            state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
            rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
            sy.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
            edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
            segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
            segment_order.data_ptr<EdgeIndex>(), dw.data_ptr<float>(), drw.data_ptr<float>());
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
            )
            weight_launch = r"""
    if (state.split_weights[{group_idx}]) {{
        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, false><<<
            dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
            state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
            rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
            sy.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
            edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
            segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
            segment_order.data_ptr<EdgeIndex>(), dw.data_ptr<float>());
        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, true><<<
            dim3(static_cast<unsigned int>(num_segments), 1, {channel_tiles}), {threads}, 0,
            state.streams[4 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), grad_out.data_ptr<float>(),
            rx.data_ptr<float>(), ry.data_ptr<float>(), sx.data_ptr<float>(),
            sy.data_ptr<float>(), sgo.data_ptr<float>(), dst.data_ptr<int32_t>(),
            edge_ids_by_src.data_ptr<EdgeIndex>(), segment_nodes.data_ptr<int32_t>(),
            segment_begin.data_ptr<EdgeIndex>(), segment_end.data_ptr<EdgeIndex>(),
            segment_order.data_ptr<EdgeIndex>(), drw.data_ptr<float>());
        cudaEventRecord(state.done[4 * GEN_NUM_GROUPS + {group_idx}],
                        state.streams[4 * GEN_NUM_GROUPS + {group_idx}].stream());
    }} else {{
{pair_weight_call}
    }}
""".format(
                group_idx=group_idx,
                channel_tiles=channel_tiles,
                threads=threads,
                pair_weight_call=pair_weight_call,
            )
            drw_launch = ""
        hgo_launch = r"""    lunatp_uvuconv_triple_bwd_dgrad_out_dst_rows_fp32<{group_idx}><<<
        dim3(static_cast<unsigned int>(num_nodes), {channel_tiles}), {threads}, 0,
        state.streams[3 * GEN_NUM_GROUPS + {group_idx}].stream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(), rw.data_ptr<float>(),
        sx.data_ptr<float>(), sy.data_ptr<float>(), sw.data_ptr<float>(),
        src.data_ptr<int32_t>(), rowptr.data_ptr<EdgeIndex>(),
        node_order.data_ptr<int32_t>(), dgo.data_ptr<float>());
    cudaEventRecord(state.done[3 * GEN_NUM_GROUPS + {group_idx}],
                    state.streams[3 * GEN_NUM_GROUPS + {group_idx}].stream());
""".format(
            group_idx=group_idx,
            channel_tiles=channel_tiles,
            threads=threads,
        )
        launches.append(
            r"""
{node_y_launch}

{weight_launch}
    cudaEventRecord(state.done[2 * GEN_NUM_GROUPS + {group_idx}],
                    state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream());

{hgo_launch}
{drw_launch}
""".format(
                node_y_launch=node_y_launch,
                weight_launch=weight_launch,
                group_idx=group_idx,
                hgo_launch=hgo_launch,
                drw_launch=drw_launch,
            )
        )
        if not variable_channels:
            paired_launches.append(
                r"""
{node_y_launch}

{pair_weight_call}
    cudaEventRecord(state.done[2 * GEN_NUM_GROUPS + {group_idx}],
                    state.streams[2 * GEN_NUM_GROUPS + {group_idx}].stream());

{hgo_launch}""".format(
                    node_y_launch=node_y_launch,
                    pair_weight_call=pair_weight_call,
                    group_idx=group_idx,
                    hgo_launch=hgo_launch,
                )
            )

    num_roles = 5
    split_weight_init = (
        ""
        if variable_channels
        else """
        split_weights.resize(GEN_NUM_GROUPS);
"""
        + "\n".join(
            r"""
        cudaFuncAttributes pair_attributes_{group_idx}{{}};
        cudaFuncAttributes dw_attributes_{group_idx}{{}};
        cudaFuncAttributes drw_attributes_{group_idx}{{}};
        TORCH_CHECK(cudaFuncGetAttributes(
                        &pair_attributes_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_pair_fp32<{group_idx}>) == cudaSuccess,
                    "triple backward paired-weight resource query failed");
        TORCH_CHECK(cudaFuncGetAttributes(
                        &dw_attributes_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, false>) == cudaSuccess,
                    "triple backward dw resource query failed");
        TORCH_CHECK(cudaFuncGetAttributes(
                        &drw_attributes_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, true>) == cudaSuccess,
                    "triple backward drw resource query failed");
        int pair_blocks_{group_idx} = 0;
        int dw_blocks_{group_idx} = 0;
        int drw_blocks_{group_idx} = 0;
        TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &pair_blocks_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_pair_fp32<{group_idx}>, {threads}, 0) == cudaSuccess,
                    "triple backward paired-weight occupancy query failed");
        TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &dw_blocks_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, false>, {threads}, 0) == cudaSuccess,
                    "triple backward dw occupancy query failed");
        TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &drw_blocks_{group_idx},
                        lunatp_uvuconv_triple_bwd_weight_fp32<{group_idx}, true>, {threads}, 0) == cudaSuccess,
                    "triple backward drw occupancy query failed");
        const size_t split_local_{group_idx} =
            dw_attributes_{group_idx}.localSizeBytes > drw_attributes_{group_idx}.localSizeBytes ?
            dw_attributes_{group_idx}.localSizeBytes : drw_attributes_{group_idx}.localSizeBytes;
        const int split_blocks_{group_idx} =
            dw_blocks_{group_idx} < drw_blocks_{group_idx} ?
            dw_blocks_{group_idx} : drw_blocks_{group_idx};
        split_weights[{group_idx}] =
            split_local_{group_idx} < pair_attributes_{group_idx}.localSizeBytes &&
            split_blocks_{group_idx} >= pair_blocks_{group_idx};
        any_split = any_split || split_weights[{group_idx}];""".format(
                group_idx=group_idx,
                threads=threads,
            )
            for group_idx, (threads, _) in enumerate(schedules)
        )
    )
    split_weight_fields = (
        ""
        if variable_channels
        else "\n    std::vector<bool> split_weights;\n    bool any_split = false;"
    )
    stream_start = (
        f"    for (int group = 0; group < {num_roles} * GEN_NUM_GROUPS; ++group)\n        cudaStreamWaitEvent(state.streams[group].stream(), state.start, 0);\n"
        if variable_channels
        else r"""    for (int group = 0; group < 4 * GEN_NUM_GROUPS; ++group)
        cudaStreamWaitEvent(state.streams[group].stream(), state.start, 0);
    for (int group = 0; group < GEN_NUM_GROUPS; ++group)
        if (state.split_weights[group])
            cudaStreamWaitEvent(
                state.streams[4 * GEN_NUM_GROUPS + group].stream(), state.start, 0);
"""
    )
    stream_finish = (
        f"    for (int group = 0; group < {num_roles} * GEN_NUM_GROUPS; ++group)\n        cudaStreamWaitEvent(current, state.done[group], 0);\n"
        if variable_channels
        else r"""    for (int group = 0; group < 4 * GEN_NUM_GROUPS; ++group)
        cudaStreamWaitEvent(current, state.done[group], 0);
    for (int group = 0; group < GEN_NUM_GROUPS; ++group)
        if (state.split_weights[group])
            cudaStreamWaitEvent(current, state.done[4 * GEN_NUM_GROUPS + group], 0);
"""
    )
    launch_dispatch = (
        (r""""""
        + (r"""    C10_CUDA_CHECK(cudaMemsetAsync(hl0->data_ptr<float>(), 0, hl0->numel() * sizeof(float), current));
""" if with_l0 and any("l0_path" in spec for spec in specs) else "")
        + r"""    cudaEventRecord(state.start, current);
""")
        + stream_start
        + "".join(launches)
        + "\n"
        + stream_finish
        if variable_channels
        else ((r"""    C10_CUDA_CHECK(cudaMemsetAsync(hl0->data_ptr<float>(), 0, hl0->numel() * sizeof(float), current));
""" if with_l0 and any("l0_path" in spec for spec in specs) else "")
        + r"""    if (!state.any_split) {{
        cudaEventRecord(state.start, current);
        for (int group = 0; group < 4 * GEN_NUM_GROUPS; ++group)
            cudaStreamWaitEvent(state.streams[group].stream(), state.start, 0);
{paired_launches}
        for (int group = 0; group < 4 * GEN_NUM_GROUPS; ++group)
            cudaStreamWaitEvent(current, state.done[group], 0);
    }} else {{
        cudaEventRecord(state.start, current);
{stream_start}{launches}
{stream_finish}    }}
""").format(
            paired_launches="".join(paired_launches),
            stream_start=stream_start,
            launches="".join(launches),
            stream_finish=stream_finish,
        )
    )
    tail_reduction = (
        TAIL_WARP_REDUCTION if any(int(s["channels"]) % 32 for s in specs) else ""
    )
    source = (
        (r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

namespace {{
"""
        + (f"constexpr int GEN_ENV_PARTS = {partial_count};\n" if need_grad_env else "")
        + r"""constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_OUT_DIM = {out_dim};
constexpr int GEN_NUM_GROUPS = {groups_count};
constexpr int GEN_NUM_PARTIALS = {partial_count};

__device__ __forceinline__ float warp_reduce_sum(float value) {{
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}}

{tail_reduction}__device__ __forceinline__ void store_global_cs_f32(float* ptr, float value) {{
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(ptr), "f"(value) : "memory");
}}

template <int GROUP>
__global__ void lunatp_uvuconv_triple_bwd_node_source_fp32(
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ sy,
    const float* __restrict__ sw,
    const float* __restrict__ sgo,
    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ edge_ids_by_src,
    const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end,
    const EdgeIndex* __restrict__ segment_order,
    float* __restrict__ dx_partial,
    float* __restrict__ drx_partial) {{
    const Index segment_idx = segment_order[blockIdx.x];
    const Index begin = segment_begin[segment_idx];
    const Index end = segment_end[segment_idx];
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    constexpr int group = GROUP;
    switch (group) {{
{node_source_branches}
    default: break;
    }}
}}

template <int GROUP>
__global__ void lunatp_uvuconv_triple_bwd_y_source_fp32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    const float* __restrict__ rw,
    const float* __restrict__ sx,
    const float* __restrict__ sw,
    const float* __restrict__ sgo,
    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ edge_ids_by_src,
    const int32_t* __restrict__ segment_nodes,
    const EdgeIndex* __restrict__ segment_begin,
    const EdgeIndex* __restrict__ segment_end,
    const EdgeIndex* __restrict__ segment_order,
    float* __restrict__ dy_partial,
    float* __restrict__ dry_partial,
    int64_t num_edges) {{
    const Index segment_idx = segment_order[blockIdx.x];
    const int s_node = segment_nodes[segment_idx];
    const Index begin = segment_begin[segment_idx];
    const Index end = segment_end[segment_idx];
    const int u = blockIdx.z * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int group = GROUP;
    Index x_base = 0;
    switch (group) {{
""").format(
            in1_dim=in1_dim,
            in2_dim=in2_dim,
            out_dim=out_dim,
            groups_count=len(groups),
            partial_count=partial_count,
            tail_reduction=tail_reduction,
            node_source_branches=node_source_branches,
        )
        + "\n".join(
            f"    case {group_idx}: x_base = Index(s_node) * GEN_IN1_DIM + {x_offset} + u * {d1}; break;"
            for group_idx, ((x_offset, d1), _) in enumerate(groups)
        )
        + (r"""
    default: return;
    }}
    switch (group) {{
{y_source_branches}
    default: break;
    }}
}}

{weight_kernel}

{paired_node_y_kernel}

__global__ void lunatp_uvuconv_triple_bwd_node_reduce_fp32(
    const float* __restrict__ dx_partial,
    const float* __restrict__ drx_partial,
    const EdgeIndex* __restrict__ segment_rowptr,
    float* __restrict__ dx,
    float* __restrict__ drx,
    int64_t num_nodes) {{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_nodes * GEN_IN1_DIM;
    if (index >= total) return;
    const int node = static_cast<int>(index / GEN_IN1_DIM);
    const int feature = static_cast<int>(index - static_cast<int64_t>(node) * GEN_IN1_DIM);
    float dx_value = 0.0f;
    float drx_value = 0.0f;
    for (Index segment = segment_rowptr[node]; segment < segment_rowptr[node + 1]; ++segment) {{
        const int64_t offset = static_cast<int64_t>(segment) * GEN_IN1_DIM + feature;
        dx_value += dx_partial[offset];
        drx_value += drx_partial[offset];
    }}
    dx[index] = dx_value;
    drx[index] = drx_value;
}}

__global__ void lunatp_uvuconv_triple_bwd_edge_reduce_fp32(
    const float* __restrict__ dy_partial,
    const float* __restrict__ dry_partial,
    float* __restrict__ dy,
    float* __restrict__ dry,
    int64_t num_edges) {{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_edges * GEN_IN2_DIM;
    if (index >= total) return;
    float dy_value = 0.0f;
    float dry_value = 0.0f;
    #pragma unroll
    for (int partial = 0; partial < GEN_NUM_PARTIALS; ++partial) {{
        const int64_t offset = static_cast<int64_t>(partial) * total + index;
        dy_value += dy_partial[offset];
        dry_value += dry_partial[offset];
    }}
    dy[index] = dy_value;
    dry[index] = dry_value;"""
        + (r"""
    if (index % GEN_IN2_DIM == 0) {{
        const Index edge = index / GEN_IN2_DIM;
        float de_value = 0.0f, dre_value = 0.0f;
        for (int p = 0; p < GEN_ENV_PARTS; ++p) {{
            de_value += de_partial[edge * GEN_ENV_PARTS + p];
            dre_value += dre_partial[edge * GEN_ENV_PARTS + p];
        }}
        de[edge] = de_value;
        dre[edge] = dre_value;
    }}""" if need_grad_env else "")
        + r"""
}}

template <int GROUP>
__global__ void lunatp_uvuconv_triple_bwd_dgrad_out_dst_rows_fp32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rw,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sw,
    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ rowptr,
    const int32_t* __restrict__ node_order,
    float* __restrict__ dgo) {{
    const int rank = blockIdx.x;
    const int t = node_order[rank];
    const int u = blockIdx.y * blockDim.x + threadIdx.x;
    constexpr int group = GROUP;
    switch (group) {{
{hgo_cases}
    default: break;
    }}
}}

struct TripleBackwardStreams {{
    ~TripleBackwardStreams() {{
        // Match CUDAEvent's device-aware, non-throwing shutdown behavior.
        try {{
            c10::cuda::CUDAGuard guard(streams.front().device_index());
            cudaEventDestroy(start);
            for (auto event : done) cudaEventDestroy(event);
        }} catch (...) {{ /* CUDA may already be shutting down. */ }}
    }}
    TripleBackwardStreams(const TripleBackwardStreams&) = delete;
    TripleBackwardStreams& operator=(const TripleBackwardStreams&) = delete;
    explicit TripleBackwardStreams(int device, const std::vector<int64_t>& handles) {{
        c10::cuda::CUDAGuard guard(device);
        streams.reserve({num_roles} * GEN_NUM_GROUPS);
        done.resize({num_roles} * GEN_NUM_GROUPS);{split_weight_init}
        cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
        for (int group = 0; group < {num_roles} * GEN_NUM_GROUPS; ++group) {{
            streams.push_back(at::cuda::getStreamFromExternal(
                reinterpret_cast<cudaStream_t>(handles.at(group)), device));
            cudaEventCreateWithFlags(&done[group], cudaEventDisableTiming);
        }}
    }}
    std::vector<at::cuda::CUDAStream> streams;
    std::vector<cudaEvent_t> done;
    cudaEvent_t start = nullptr;{split_weight_fields}
}};

}}  // namespace

void run_uvu_conv_triple_bwd(
    TripleBackwardStreams& state,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor rx,
    torch::Tensor ry,
    torch::Tensor rw,
    torch::Tensor sx,
    torch::Tensor sy,
    torch::Tensor sw,
    torch::Tensor sgo,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor rowptr,
    torch::Tensor edge_ids_by_src,
    torch::Tensor segment_nodes,
    torch::Tensor segment_begin,
    torch::Tensor segment_end,
    torch::Tensor segment_rowptr,
    torch::Tensor segment_order,
    torch::Tensor node_order,
    torch::Tensor dx,
    torch::Tensor dx_partial,
    torch::Tensor dy,
    torch::Tensor dy_partial,
    torch::Tensor dw,
    torch::Tensor dgo,
    torch::Tensor drx,
    torch::Tensor drx_partial,
    torch::Tensor dry,
    torch::Tensor dry_partial,
    torch::Tensor drw,
    int64_t num_nodes,
    int64_t num_edges,
    int64_t num_segments) {{
    const int device = x.get_device();
    cudaStream_t current = at::cuda::getCurrentCUDAStream(device);
{launch_dispatch}    constexpr int threads = 256;
    const int64_t node_total = num_nodes * GEN_IN1_DIM;
    lunatp_uvuconv_triple_bwd_node_reduce_fp32<<<
        static_cast<int>((node_total + threads - 1) / threads), threads, 0, current>>>(
        dx_partial.data_ptr<float>(), drx_partial.data_ptr<float>(),
        segment_rowptr.data_ptr<EdgeIndex>(), dx.data_ptr<float>(),
        drx.data_ptr<float>(), num_nodes);
    const int64_t edge_total = num_edges * GEN_IN2_DIM;
    lunatp_uvuconv_triple_bwd_edge_reduce_fp32<<<
        static_cast<int>((edge_total + threads - 1) / threads), threads, 0, current>>>(
        dy_partial.data_ptr<float>(), dry_partial.data_ptr<float>(),
        dy.data_ptr<float>(), dry.data_ptr<float>(), num_edges);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
    pybind11::class_<TripleBackwardStreams>(module, "_Streams", pybind11::module_local())
        .def(pybind11::init<int, const std::vector<int64_t>&>());
    module.attr("_Streams").attr("stream_count") = {num_roles} * GEN_NUM_GROUPS;
    module.attr("_Streams").attr("stream_priority") = 0;
"""
        + (r"""    module.attr("env_parts") = GEN_ENV_PARTS;
""" if need_grad_env else "")
        + r"""    module.def("run_uvu_conv_triple_bwd", &run_uvu_conv_triple_bwd,
               "LunaTP source-owned complete third-order derivative (CUDA)");
}}
""").format(
            y_source_branches=y_source_branches,
            weight_kernel=weight_kernel,
            paired_node_y_kernel=paired_node_y_kernel,
            hgo_cases=hgo_cases,
            num_roles=num_roles,
            split_weight_init=split_weight_init,
            split_weight_fields=split_weight_fields,
            launch_dispatch=launch_dispatch,
        )
    )
    parameters = (
        {
            "grad_l0": "const float* __restrict__",
            "sl0": "const float* __restrict__",
            "hl0": "float* __restrict__",
        }
        if with_l0
        else {}
    )
    if with_env:
        parameters["env"] = "const float* __restrict__"
    if need_grad_env:
        parameters.update(
            {
                "re": "const float* __restrict__",
                "se": "const float* __restrict__",
                "de_partial": "float* __restrict__",
                "dre_partial": "float* __restrict__",
                "de": "float* __restrict__",
                "dre": "float* __restrict__",
                "weight": "const float* __restrict__",
                "rw": "const float* __restrict__",
                "sw": "const float* __restrict__",
            }
        )
    return add_optional_arguments(
        source,
        "run_uvu_conv_triple_bwd",
        parameters,
        (
            "grad_l0",
            "sl0",
            "hl0",
            "env",
            "re",
            "se",
            "de_partial",
            "dre_partial",
            "de",
            "dre",
        ),
    ), partial_count


def emit_triple_bwd_raw_l0(lines, spec, path, offsets, width, outputs):
    """Add raw L0 derivatives without changing the node/envelope contraction."""
    yo, wo = int(spec["y_offset"]), int(spec["weight_offset"])
    address = f"weight_base + {wo} + {weight_channel_offset(spec)}"
    operands = {
        role: ([f"__ldg({name} + x_base + {i})" for i in range(int(spec["d1"]))])
        for role, name in (("x", "x"), ("rx", "rx"), ("sx", "sx"))
    }
    operands.update(
        {
            role: (
                [
                    f"__ldg({name} + Index(e) * GEN_IN2_DIM + {yo + j})"
                    for j in range(int(spec["d2"]))
                ]
            )
            for role, name in (("y", "y"), ("ry", "ry"), ("sy", "sy"))
        }
    )
    operands.update(
        {
            role: f"__ldg({name} + {address})"
            for role, name in (("w", "weight"), ("rw", "rw"), ("sw", "sw"))
        }
    )
    emit_raw_l0_contractions(
        lines,
        spec,
        path,
        offsets,
        width,
        outputs=outputs,
        operands=operands,
    )


def emit_triple_bwd_env_weight_loads(offset, *, need_grad_env=True, channel="u"):
    """Effective weight directions for the existing factorized contractions."""
    lines = [
        f"                const float raw_w = __ldg(weight + weight_base + {offset} + {channel});",
        f"                const float raw_rw = __ldg(rw + weight_base + {offset} + {channel});",
        f"                const float raw_sw = __ldg(sw + weight_base + {offset} + {channel});",
        f"                const float w = env_e * raw_w;",
        f"                const float rw_value = env_e * raw_rw"
        + (" + re_e * raw_w;" if need_grad_env else ";"),
        f"                const float sw_value = env_e * raw_sw"
        + (" + se_e * raw_w;" if need_grad_env else ";"),
    ]
    if need_grad_env:
        lines.append(
            f"                const float cross_weight = re_e * raw_sw + se_e * raw_rw;"
        )
    return lines


def emit_triple_bwd_weight_cases(
    specs: list[UvuPathSpec],
    *,
    in2_dim: int,
    out_dim: int,
    weight_numel: int,
    per_edge: bool,
    outputs: str,
    with_l0: bool = False,
    with_env: bool = False,
    need_grad_env: bool = True,
    env_partial_bases: dict[int, int] | None = None,
) -> str:
    """Emit source-aggregated or per-edge weight-output branches."""
    need_grad_env = with_env and need_grad_env
    l0_offsets, l0_dim = l0_feature_layout(specs) if with_l0 else ({}, 0)
    node_seed_offsets = {} if with_env else l0_offsets
    if outputs not in {"both", "dw", "drw"}:
        raise ValueError("triple backward weight outputs must be both, dw, or drw")

    branches: list[str] = []
    for group_idx, ((x_offset, d1), raw_specs) in enumerate(
        group_source_input_paths(specs)
    ):
        channels = {int(spec["channels"]) for _, spec in raw_specs}
        if len(channels) != 1:
            raise ValueError("each triple backward x group must have one channel width")
        channel_count = channels.pop()
        y_groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = (
            defaultdict(list)
        )
        for path_idx, spec in raw_specs:
            y_groups[(int(spec["y_offset"]), int(spec["d2"]))].append((path_idx, spec))

        lines = [f"case {group_idx}: {{", f"    if (u >= {channel_count}) break;"]
        for i_idx in range(d1):
            if outputs == "drw":
                lines.extend(
                    f"    const float x{i_idx} = __ldg(x + x_base + {i_idx});\n    const float sx{i_idx} = __ldg(sx + x_base + {i_idx});".split(
                        "\n"
                    )
                )
            else:
                lines.extend(r"""    const float x{i} = __ldg(x + x_base + {i});
    const float rx{i} = __ldg(rx + x_base + {i});
    const float sx{i} = __ldg(sx + x_base + {i});""".format(i=i_idx).split("\n"))
        if per_edge:
            lines.extend(
                [
                    f"    const Index y_base = e * {in2_dim};",
                    f"    const Index weight_base = e * {weight_numel};",
                    f"    const Index go_base = Index(d) * {out_dim};",
                ]
            )
        else:
            lines.extend(
                r"""    for (Index p = begin; p < end; ++p) {{
{edge_index}
        const Index y_base = e * {in2_dim};
        const Index weight_base = e * {weight_numel};
        const Index go_base = Index(d) * {out_dim};""".format(
                    edge_index=(
                        "        const Index e = edge_ids_by_src[p];\n"
                        "        const int d = __ldg(dst + e);"
                    ),
                    in2_dim=in2_dim,
                    weight_numel=weight_numel,
                    out_dim=out_dim,
                ).split("\n")
            )

        if with_env:
            lines.append(f"        const float env_e = __ldg(env + e);")
            if need_grad_env:
                lines.extend(r"""        const float re_e = re ? __ldg(re + e) : 0.0f;
        const float se_e = se ? __ldg(se + e) : 0.0f;""".split("\n"))
            if need_grad_env and outputs != "drw":
                lines.extend(
                    [
                        "        float de_value = 0.0f;",
                        "        float dre_value = 0.0f;",
                    ]
                )
        for (y_offset, d2), group_specs in sorted(
            y_groups.items(),
            key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
            reverse=True,
        ):
            lines.append("        {")
            for j_idx in range(d2):
                j_abs = y_offset + j_idx
                if outputs == "drw":
                    lines.extend(
                        f"            const float y{j_idx} = __ldg(y + y_base + {j_abs});\n            const float sy{j_idx} = __ldg(sy + y_base + {j_abs});".split(
                            "\n"
                        )
                    )
                else:
                    lines.extend(
                        r"""            const float y{j} = __ldg(y + y_base + {j_abs});
            const float ry{j} = __ldg(ry + y_base + {j_abs});
            const float sy{j} = __ldg(sy + y_base + {j_abs});""".format(j=j_idx, j_abs=j_abs).split(
                            "\n"
                        )
                    )

            for path_idx, spec in sorted(
                group_specs, key=lambda path: len(path[1]["terms"]), reverse=True
            ):
                out_offset = int(spec["out_offset"])
                weight_offset = int(spec["weight_offset"])
                d3 = int(spec["d3"])
                terms = list(spec["terms"])
                lines.append("            {")
                if outputs != "drw":
                    lines.append("                float dw_value = 0.0f;")
                if outputs != "dw" or need_grad_env:
                    lines.append("                float drw_value = 0.0f;")
                if need_grad_env:
                    lines.append("                float gv = 0.0f;")
                    if outputs != "drw":
                        lines.append("                float hv = 0.0f;")
                for class_idx, signature_class in enumerate(
                    build_signed_ij_classes(
                        terms, spec_y_offset=y_offset, y_name_base=y_offset
                    )
                ):
                    representative = signature_class.members[0]
                    go_sum = " + ".join(
                        f"""{float_literal(float(coeff))} * {output_gradient_load("grad_out", out_offset, d3, int(k_idx), path_idx, node_seed_offsets, l0_dim)}"""
                        for _, _, k_idx, coeff in representative.terms
                    )
                    sgo_sum = " + ".join(
                        f"""{float_literal(float(coeff))} * {output_gradient_load("sgo", out_offset, d3, int(k_idx), path_idx, node_seed_offsets, l0_dim)}"""
                        for _, _, k_idx, coeff in representative.terms
                    )
                    lines.extend(
                        [
                            f"                const float q{path_idx}_{class_idx} = {go_sum};",
                            f"                const float qs{path_idx}_{class_idx} = {sgo_sum};",
                        ]
                    )
                    for member in signature_class.members:
                        op = "+=" if member.relative_sign > 0 else "-="
                        i_idx = member.i_idx
                        j_idx = member.j_name
                        q = f"q{path_idx}_{class_idx}"
                        qs = f"qs{path_idx}_{class_idx}"
                        if outputs != "drw":
                            lines.append(
                                f"                dw_value {op} ((sx{i_idx} * ry{j_idx} + rx{i_idx} * sy{j_idx}) * {q} + (rx{i_idx} * y{j_idx} + x{i_idx} * ry{j_idx}) * {qs});"
                            )
                        if outputs != "dw" or need_grad_env:
                            lines.append(
                                f"                drw_value {op} ((sx{i_idx} * y{j_idx} + x{i_idx} * sy{j_idx}) * {q} + x{i_idx} * y{j_idx} * {qs});"
                            )
                        if need_grad_env:
                            lines.append(
                                f"                gv {op} x{i_idx} * y{j_idx} * {q};"
                            )
                            if outputs != "drw":
                                lines.append(
                                    f"                hv {op} (rx{i_idx} * y{j_idx} + x{i_idx} * ry{j_idx}) * {q};"
                                )
                if need_grad_env:
                    if outputs != "drw":
                        lines.extend(
                            r"""                const float raw_w = __ldg(weight + weight_base + {weight_offset} + {weight_channel_offset});
                const float raw_rw = __ldg(rw + weight_base + {weight_offset} + {weight_channel_offset});
                const float raw_sw = __ldg(sw + weight_base + {weight_offset} + {weight_channel_offset});
                de_value += raw_w * dw_value + raw_rw * drw_value + raw_sw * hv;
                dre_value += raw_w * drw_value + raw_sw * gv;
                dw_value = env_e * dw_value + re_e * drw_value + se_e * hv;""".format(
                                weight_offset=weight_offset,
                                weight_channel_offset=weight_channel_offset(spec),
                            ).split("\n")
                        )
                    lines.append(
                        "                drw_value = env_e * drw_value + se_e * gv;"
                    )
                elif with_env:
                    if outputs != "drw":
                        lines.append(f"                dw_value *= env_e;")
                    if outputs != "dw":
                        lines.append(f"                drw_value *= env_e;")
                if with_env and path_idx in l0_offsets:
                    emit_triple_bwd_raw_l0(
                        lines,
                        spec,
                        path_idx,
                        l0_offsets,
                        l0_dim,
                        {
                            **({"w": "dw_value"} if outputs != "drw" else {}),
                            **({"rw": "drw_value"} if outputs != "dw" else {}),
                        },
                    )
                if outputs == "both":
                    lines.extend(
                        f"                store_global_cs_f32(dw + weight_base + {weight_offset} + {weight_channel_offset(spec)}, dw_value);\n                store_global_cs_f32(drw + weight_base + {weight_offset} + {weight_channel_offset(spec)}, drw_value);".split(
                            "\n"
                        )
                    )
                elif outputs == "dw":
                    lines.append(
                        f"                store_global_cs_f32(out + weight_base + {weight_offset} + {weight_channel_offset(spec)}, dw_value);"
                    )
                else:
                    lines.append(
                        f"                store_global_cs_f32(out + weight_base + {weight_offset} + {weight_channel_offset(spec)}, drw_value);"
                    )
                lines.append("            }")
            lines.append("        }")

        if need_grad_env and outputs != "drw":
            env_base = env_partial_bases[group_idx]
            reduction_tail = (
                f", min(32, {channel_count} - u + (threadIdx.x & 31))"
                if channel_count % 32
                else ""
            )
            lines.extend(
                r"""        const float de_sum = warp_reduce_sum(de_value{reduction_tail});
        const float dre_sum = warp_reduce_sum(dre_value{reduction_tail});
        if ((threadIdx.x & 31) == 0) {{
            const Index part = {env_base} + blockIdx.z * (blockDim.x >> 5) + (threadIdx.x >> 5);
            de_partial[Index(e) * GEN_ENV_PARTS + part] = de_sum;
            dre_partial[Index(e) * GEN_ENV_PARTS + part] = dre_sum;
        }}""".format(
                    reduction_tail=reduction_tail,
                    env_base=env_base,
                ).split("\n")
            )
        if not per_edge:
            lines.append("    }")
        lines.extend(["    break;", "}"])
        branches.append("\n".join(lines))
    return "\n".join(branches)


def run(
    op,
    x,
    y,
    w,
    g,
    rx,
    ry,
    rw,
    sx,
    sy,
    sw,
    sg,
    graph,
    env,
    re,
    gl0,
    se,
    sl0,
    *,
    need_grad_env,
):
    weight = w
    grad_out = g
    sgo = sg
    grad_l0 = gl0
    need_grad_env = env is not None and need_grad_env
    src, dst, dst_rowptr = graph.src, graph.dst, graph.dst_rowptr
    edge_ids_by_src, node_order = graph.edge_ids_by_src, graph.node_order
    source_segments = graph.source_segments
    uniform_channels = op.uniform_channels

    if src.numel() == 0:
        values = (x, y, weight, grad_out)
        if need_grad_env:
            values += (env,)
        if grad_l0 is not None:
            values += (grad_l0,)
        values += (rx, ry, rw)
        if need_grad_env:
            values += (env if re is None else re,)
        outputs = tuple(torch.zeros_like(t) for t in values)
    else:
        cache_name = "_triple_bwd_source_ext" + (
            "_l0" if grad_l0 is not None else "_env" if env is not None else ""
        )
        if env is not None and grad_l0 is not None:
            cache_name += "_env"
        if env is not None and not need_grad_env:
            cache_name += "_fixed"
        extension = getattr(op, cache_name, None)
        if extension is None:
            specs = [dict(spec) for spec in op.generated_specs]
            x_components = sum(d1 for (_, d1), _ in group_source_input_paths(specs))
            if x_components <= UVU_CONV_TRIPLE_BWD_SOURCE_COMPONENT_LIMIT:
                source, partial_count, node_parts = (
                    make_uvu_conv_triple_bwd_source_edge_source(
                        specs,
                        in1_dim=op.in1_dim,
                        in2_dim=op.in2_dim,
                        out_dim=op.out_dim,
                        weight_numel=op.weight_numel,
                        scalar_bytes=x.element_size(),
                        with_l0=grad_l0 is not None,
                        with_env=env is not None,
                        need_grad_env=need_grad_env,
                    )
                )
            else:
                source, partial_count = make_uvu_conv_triple_bwd_split_source(
                    specs,
                    in1_dim=op.in1_dim,
                    in2_dim=op.in2_dim,
                    out_dim=op.out_dim,
                    weight_numel=op.weight_numel,
                    variable_channels=not uniform_channels,
                    scalar_bytes=x.element_size(),
                    with_l0=grad_l0 is not None,
                    with_env=env is not None,
                    need_grad_env=need_grad_env,
                )
                node_parts = 1
            name = (
                "lunatp_uvuconv_triple_bwd_fp64"
                if x.dtype == torch.float64
                else "lunatp_uvuconv_triple_bwd_fp32"
            )
            if x.dtype == torch.float64:
                source = cuda_source_to_fp64(source)
            extension = load_tp_extension(
                name, source, max_channels=op.max_channels, index_bits=op.index_bits
            )
            op._triple_bwd_source = source
            setattr(op, cache_name, extension)
            op._triple_bwd_partial_count = partial_count
            op._triple_bwd_node_parts = node_parts
            x_holes, _ = flat_coverage(specs, op.in1_dim, "x_offset", "d1")
            out_holes, out_overlap = flat_coverage(
                specs, op.out_dim, "out_offset", "d3"
            )
            op._triple_bwd_node_partials_need_zero = bool(x_holes)
            op._triple_bwd_dgo_need_zero = bool(out_holes) or out_overlap

        # Like the compiled extension, this policy is cached for the immutable
        # operator metadata. It needs no tensor reads or per-call graph analysis.
        needs_zero = getattr(op, "_triple_bwd_edge_partials_need_zero", None)
        if needs_zero is None:
            groups = group_source_input_paths(op.generated_specs)
            needs_zero = not groups
            required = set(range(op.in2_dim))
            for _, group_specs in groups:
                covered = {
                    int(spec["y_offset"]) + component
                    for _, spec in group_specs
                    for component in range(int(spec["d2"]))
                }
                if covered != required:
                    needs_zero = True
                    break
            op._triple_bwd_edge_partials_need_zero = needs_zero

        src_i32 = src
        segment_nodes = source_segments.nodes
        segment_begin = source_segments.begin
        segment_end = source_segments.end
        segment_rowptr = source_segments.rowptr
        segment_order = source_segments.order
        num_nodes = x.shape[0]
        num_edges = src.numel()
        num_segments = segment_nodes.numel()

        dx = torch.empty_like(x)
        dy = torch.empty_like(y)
        dw = torch.empty_like(weight)
        dgo = (torch.zeros_like if op._triple_bwd_dgo_need_zero else torch.empty_like)(
            grad_out
        )
        drx = torch.empty_like(x)
        dry = torch.empty_like(y)
        drw = torch.empty_like(weight)
        node_alloc = (
            torch.zeros if op._triple_bwd_node_partials_need_zero else torch.empty
        )
        dx_partial = node_alloc(
            (op._triple_bwd_node_parts * num_segments, op.in1_dim),
            device=x.device,
            dtype=x.dtype,
        )
        drx_partial = (
            torch.zeros_like
            if op._triple_bwd_node_partials_need_zero
            else torch.empty_like
        )(dx_partial)
        partial_shape = (
            op._triple_bwd_partial_count,
            num_edges,
            op.in2_dim,
        )
        if needs_zero:
            dy_partial = torch.zeros(partial_shape, device=x.device, dtype=x.dtype)
            dry_partial = torch.zeros_like(dy_partial)
        else:
            dy_partial = torch.empty(partial_shape, device=x.device, dtype=x.dtype)
            dry_partial = torch.empty_like(dy_partial)
        dl0 = torch.empty_like(grad_l0) if grad_l0 is not None else None
        de = torch.empty_like(env) if need_grad_env else None
        dre = torch.empty_like(env) if need_grad_env else None
        de_partial = (
            torch.empty(
                (num_edges, extension.env_parts), device=x.device, dtype=x.dtype
            )
            if need_grad_env
            else None
        )
        dre_partial = torch.empty_like(de_partial) if need_grad_env else None

        extension.run_uvu_conv_triple_bwd(
            get_overlap_state(op, x.device, extension._Streams),
            x,
            y,
            weight,
            grad_out,
            rx,
            ry,
            rw,
            sx,
            sy,
            sw,
            sgo,
            src_i32,
            dst,
            dst_rowptr,
            edge_ids_by_src,
            segment_nodes,
            segment_begin,
            segment_end,
            segment_rowptr,
            segment_order,
            node_order,
            dx,
            dx_partial,
            dy,
            dy_partial,
            dw,
            dgo,
            drx,
            drx_partial,
            dry,
            dry_partial,
            drw,
            num_nodes,
            num_edges,
            num_segments,
            grad_l0,
            sl0,
            dl0,
            env,
            re,
            se,
            de_partial,
            dre_partial,
            de,
            dre,
        )
        if de is not None:
            if dl0 is not None:
                outputs = dx, dy, dw, dgo, de, dl0, drx, dry, drw, dre
            else:
                outputs = dx, dy, dw, dgo, de, drx, dry, drw, dre
        elif dl0 is not None:
            outputs = dx, dy, dw, dgo, dl0, drx, dry, drw
        else:
            outputs = dx, dy, dw, dgo, drx, dry, drw
    return outputs
