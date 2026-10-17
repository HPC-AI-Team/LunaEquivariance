"""UUU convolution triple backward CUDA."""

from __future__ import annotations
import torch
from ....utils.precision import float_literal
from ...utils.layout import build_signed_ij_classes


def emit_contraction(
    op,
    spec,
    targets,
    lines,
    selected,
    *,
    with_env=False,
    need_grad_env=True,
    source_axis_first=False,
):
    if (
        {"gy", "gry", "gw", "grw"}
        <= set(selected)
        <= {"gx", "grx", "gy", "gry", "gw", "grw", "ge", "gre"}
        or {"gx", "grx", "gw", "grw"}
        <= set(selected)
        <= {"gx", "grx", "gw", "grw", "ge", "gre"}
        or (with_env and {"gw", "grw"} <= set(selected) <= {"gw", "grw", "ge", "gre"})
    ):
        dtype = op.compute_dtype
        scalar = "double" if dtype == torch.float64 else "float"
        dims = {"x": int(spec["d1"]), "y": int(spec["d2"]), "g": int(spec["d3"])}
        # Differentiate the scalar w*e before projecting CG. Envelope products
        # then reuse the same four feature projections as ordinary UUU.
        scale, direction, seed_scale = "w", "rw_value", "bw"
        if with_env and set(selected) & {"gx", "grx", "gy", "gry"}:
            env_direction = "re_value" if need_grad_env else "0.0"
            env_seed = "be" if need_grad_env else "0.0"
            lines.append(
                r"""const {scalar} envelope_scale = w * e;
const {scalar} envelope_direction = rw_value * e + w * {env_direction};
const {scalar} envelope_seed = bw * e + w * {env_seed};
const {scalar} envelope_cross = rw_value * {env_seed} + bw * {env_direction};""".format(
                    scalar=scalar,
                    env_direction=env_direction,
                    env_seed=env_seed,
                )
            )
            scale, direction, seed_scale = (
                "envelope_scale",
                "envelope_direction",
                "envelope_seed",
            )

        source_stream = source_axis_first and dims["x"] <= dims["y"]
        if (
            {"gx", "grx"} <= set(selected)
            and (dims["x"] < dims["y"] or source_stream or "gy" not in selected)
        ) or (
            with_env
            and set(selected) <= {"gw", "grw", "ge", "gre"}
            and dims["x"] <= dims["y"]
        ):
            # The source contraction has the same four projected families;
            # choose its shorter angular state before forming scalar grads.
            components = range(dims["x"])
            chunks = [(i,) for i in components] if source_stream else [components]
            classes = build_signed_ij_classes(
                spec["terms"], spec_y_offset=0, y_name_base=0
            )
            for chunk in chunks:
                if source_stream:
                    lines.append("{")
                for i in chunk:
                    for name in ("base", "direction", "mixed", "cross"):
                        lines.append(f"{scalar} source_{name}{i} = 0.0;")
                    if "gy" in selected:
                        cross = f" + envelope_cross * x{i}" if with_env else ""
                        lines.append(
                            f"const {scalar} edge_g{i} = {seed_scale} * rx{i} + {direction} * bx{i}{cross};\nconst {scalar} edge_bg{i} = {scale} * rx{i} + {direction} * x{i};"
                        )
                    if "gry" in selected:
                        lines.append(
                            f"const {scalar} edge_rg{i} = {scale} * bx{i} + {seed_scale} * x{i};\nconst {scalar} edge_rbg{i} = {scale} * x{i};"
                        )
                for pairs in classes:
                    members = [
                        member for member in pairs.members if member.i_idx in chunk
                    ]
                    if not members:
                        continue
                    lines.append("{")
                    for gradient in ("g", "bg"):
                        expression = " + ".join(
                            f"{float_literal(coefficient, dtype)} * {gradient}{k}"
                            for _i, _j, k, coefficient in pairs.members[0].terms
                        )
                        lines.append(
                            f"const {scalar} projection_{gradient} = {expression};"
                        )
                    for member in members:
                        operation = "+=" if member.relative_sign > 0 else "-="
                        i, j = member.i_idx, member.j_name
                        projection_updates = r"""source_base{i} {operation} y{j} * projection_g;
source_direction{i} {operation} ry{j} * projection_g;
source_mixed{i} {operation} by{j} * projection_g + y{j} * projection_bg;
source_cross{i} {operation} ry{j} * projection_bg;
{edge_gradient} {operation} edge_g{i} * projection_g + edge_bg{i} * projection_bg;
{edge_direction_gradient} {operation} edge_rg{i} * projection_g + edge_rbg{i} * projection_bg;""".format(
                            i=i,
                            j=j,
                            operation=operation,
                            edge_gradient=targets["gy"][j] if "gy" in selected else "",
                            edge_direction_gradient=targets["gry"][j]
                            if "gry" in selected
                            else "",
                        ).splitlines()
                        lines.extend(projection_updates[:4])
                        if "gy" in selected:
                            lines.append(projection_updates[4])
                        if "gry" in selected:
                            lines.append(projection_updates[5])
                    lines.append("}")
                for i in chunk:
                    updates = f"""{(targets["gx"][i] if "gx" in selected else "")} += {seed_scale} * source_direction{i} + {direction} * source_mixed{i} + {scale} * source_cross{i}{(f" + envelope_cross * source_base{i}" if with_env else "")};\n{(targets["grx"][i] if "grx" in selected else "")} += {scale} * source_mixed{i} + {seed_scale} * source_base{i};""".splitlines()
                    if "gx" in selected:
                        lines.append(updates[0])
                    if "grx" in selected:
                        lines.append(updates[1])
                    if with_env:
                        lines.append(r"""{{
const {scalar} scalar_cross = bx{i} * source_direction{i} + rx{i} * source_mixed{i} + x{i} * source_cross{i};
const {scalar} scalar_mixed = x{i} * source_mixed{i} + bx{i} * source_base{i};""".format(scalar=scalar, i=i))
                        if need_grad_env:
                            lines.append(
                                f"const {scalar} scalar_direction = rx{i} * source_base{i} + x{i} * source_direction{i};\nconst {scalar} scalar_base = x{i} * source_base{i};"
                            )
                        cross = (
                            " + be * scalar_direction + re_value * scalar_mixed"
                            if need_grad_env
                            else ""
                        )
                        seed = " + be * scalar_base" if need_grad_env else ""
                        lines.append(
                            f"""{targets["gw"]} += e * scalar_cross{cross};\n{targets["grw"]} += e * scalar_mixed{seed};"""
                        )
                        envelope_updates = f"""{(targets["ge"] if "ge" in selected else "")} += w * scalar_cross + bw * scalar_direction + rw_value * scalar_mixed;\n{(targets["gre"] if "gre" in selected else "")} += w * scalar_mixed + bw * scalar_base;""".splitlines()
                        if "ge" in selected:
                            lines.append(envelope_updates[0])
                        if "gre" in selected:
                            lines.append(envelope_updates[1])
                        lines.append("}")
                    else:
                        lines.append(
                            f"""{targets["gw"]} += bx{i} * source_direction{i} + rx{i} * source_mixed{i} + x{i} * source_cross{i};\n{targets["grw"]} += x{i} * source_mixed{i} + bx{i} * source_base{i};"""
                        )
                if source_stream:
                    lines.append("}")
            lines.append("}")
            return "\n".join(lines)
        # Four unweighted projections provide both edge-feature derivatives
        # and both scalar derivatives; Q never needs to be materialized.
        for j in range(dims["y"]):
            for name in ("base", "direction", "mixed", "cross"):
                lines.append(f"{scalar} edge_{name}{j} = 0.0;")
            if "gx" in selected:
                cross = f" + envelope_cross * y{j}" if with_env else ""
                lines.append(
                    f"const {scalar} node_g{j} = {seed_scale} * ry{j} + {direction} * by{j}{cross};\nconst {scalar} node_bg{j} = {scale} * ry{j} + {direction} * y{j};"
                )
            if "grx" in selected:
                lines.append(
                    f"const {scalar} direction_g{j} = {scale} * by{j} + {seed_scale} * y{j};\nconst {scalar} direction_bg{j} = {scale} * y{j};"
                )
        for pairs in build_signed_ij_classes(
            spec["terms"], spec_y_offset=0, y_name_base=0
        ):
            lines.append("{")
            for gradient in ("g", "bg"):
                expression = " + ".join(
                    f"{float_literal(coefficient, dtype)} * {gradient}{k}"
                    for _i, _j, k, coefficient in pairs.members[0].terms
                )
                lines.append(f"const {scalar} projection_{gradient} = {expression};")
            for member in pairs.members:
                operation = "+=" if member.relative_sign > 0 else "-="
                i, j = member.i_idx, member.j_name
                projection_updates = r"""edge_base{j} {operation} x{i} * projection_g;
edge_direction{j} {operation} rx{i} * projection_g;
edge_mixed{j} {operation} bx{i} * projection_g + x{i} * projection_bg;
edge_cross{j} {operation} rx{i} * projection_bg;
{source_gradient} {operation} node_g{j} * projection_g + node_bg{j} * projection_bg;
{source_direction_gradient} {operation} direction_g{j} * projection_g + direction_bg{j} * projection_bg;""".format(
                    i=i,
                    j=j,
                    operation=operation,
                    source_gradient=targets["gx"][i] if "gx" in selected else "",
                    source_direction_gradient=targets["grx"][i]
                    if "grx" in selected
                    else "",
                ).splitlines()
                lines.extend(projection_updates[:4])
                if "gx" in selected:
                    lines.append(projection_updates[4])
                if "grx" in selected:
                    lines.append(projection_updates[5])
            lines.append("}")
        for j in range(dims["y"]):
            updates = f"""{(targets["gy"][j] if "gy" in selected else "")} += {seed_scale} * edge_direction{j} + {direction} * edge_mixed{j} + {scale} * edge_cross{j}{(f" + envelope_cross * edge_base{j}" if with_env else "")};\n{(targets["gry"][j] if "gry" in selected else "")} += {scale} * edge_mixed{j} + {seed_scale} * edge_base{j};""".splitlines()
            if "gy" in selected:
                lines.append(updates[0])
            if "gry" in selected:
                lines.append(updates[1])
            if with_env:
                lines.append(r"""{{
const {scalar} scalar_cross = by{j} * edge_direction{j} + ry{j} * edge_mixed{j} + y{j} * edge_cross{j};
const {scalar} scalar_mixed = y{j} * edge_mixed{j} + by{j} * edge_base{j};""".format(scalar=scalar, j=j))
                if need_grad_env:
                    lines.append(
                        f"const {scalar} scalar_direction = y{j} * edge_direction{j} + ry{j} * edge_base{j};\nconst {scalar} scalar_base = y{j} * edge_base{j};"
                    )
                cross = (
                    " + be * scalar_direction + re_value * scalar_mixed"
                    if need_grad_env
                    else ""
                )
                seed = " + be * scalar_base" if need_grad_env else ""
                lines.append(
                    f"""{targets["gw"]} += e * scalar_cross{cross};\n{targets["grw"]} += e * scalar_mixed{seed};"""
                )
                envelope_updates = f"""{(targets["ge"] if "ge" in selected else "")} += w * scalar_cross + bw * scalar_direction + rw_value * scalar_mixed;\n{(targets["gre"] if "gre" in selected else "")} += w * scalar_mixed + bw * scalar_base;""".splitlines()
                if "ge" in selected:
                    lines.append(envelope_updates[0])
                if "gre" in selected:
                    lines.append(envelope_updates[1])
                lines.append("}")
            else:
                lines.append(
                    f"""{targets["gw"]} += by{j} * edge_direction{j} + ry{j} * edge_mixed{j} + y{j} * edge_cross{j};\n{targets["grw"]} += y{j} * edge_mixed{j} + by{j} * edge_base{j};"""
                )
        lines.append("}")
        return "\n".join(lines)
    return None


BOUNDED_EDGE_SOURCE = r"""struct {result_type} {{ {fields} }};
__device__ __noinline__ {result_type} {helper}({input_params}, Index edge, int s, int d, int u) {{
{nulls}
{result_type} result{{}};
{body}
return result;
}}
"""


BOUNDED_NODE_SOURCE = r"""struct {result_type} {{ {fields} }};
__device__ __noinline__ {result_type} {helper}({input_params}, const EdgeIndex* __restrict__ src_rowptr, const EdgeIndex* __restrict__ edge_ids, const int32_t* __restrict__ dst, int s, int u, int neighbor_worker) {{
{nulls}
{result_type} result{{}};
for (Index p = src_rowptr[s] + neighbor_worker; p < src_rowptr[s + 1]; p += {workers}) {{
const Index edge = __ldg(edge_ids + p);
const int d = __ldg(dst + edge);

{body}
}}
return result;
}}
"""
