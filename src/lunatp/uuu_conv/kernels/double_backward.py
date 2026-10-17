"""UUU convolution double backward CUDA."""

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
        "gy" in selected
        and set(selected) <= {"gx", "gy", "gw", "ge"}
        and ("gw" in selected or "ge" in selected)
    ):
        dtype = op.compute_dtype
        scalar = "double" if dtype == torch.float64 else "float"
        dims = {"x": int(spec["d1"]), "y": int(spec["d2"]), "g": int(spec["d3"])}
        # As in UVUConv, the two unweighted edge projections supply both
        # feature and scalar derivatives without a separate directional Q.
        scale = "w * e" if with_env else "w"
        direction = "rw_value * e" if with_env else "rw_value"
        if with_env and need_grad_env:
            direction += " + w * re_value"
        lines.append(
            f"const {scalar} scale = {scale};\nconst {scalar} direction = {direction};"
        )

        if (
            source_axis_first
            and not with_env
            and (scalar == "double" or dims["x"] < dims["y"])
            and "gx" in selected
            and dims["x"] <= dims["y"]
        ):
            # Finish one source component before advancing to the next. Its
            # unweighted projections supply both node and scalar gradients.
            classes = build_signed_ij_classes(
                spec["terms"], spec_y_offset=0, y_name_base=0
            )
            for i in range(dims["x"]):
                lines.append(r"""{{
{scalar} source_base = 0.0;
{scalar} source_direction = 0.0;
const {scalar} edge_basis = scale * rx{i} + direction * x{i};""".format(scalar=scalar, i=i))
                for pairs in classes:
                    members = [member for member in pairs.members if member.i_idx == i]
                    if not members:
                        continue
                    projection = " + ".join(
                        f"{float_literal(coefficient, dtype)} * g{k}"
                        for _i, _j, k, coefficient in pairs.members[0].terms
                    )
                    lines += ["{", f"const {scalar} projection = {projection};"]
                    for member in members:
                        operation = "+=" if member.relative_sign > 0 else "-="
                        j = member.j_name
                        lines.append(
                            r"""source_base {operation} y{j} * projection;
source_direction {operation} ry{j} * projection;
{edge_gradient} {operation} edge_basis * projection;""".format(
                                operation=operation,
                                j=j,
                                edge_gradient=targets["gy"][j],
                            )
                        )
                    lines.append("}")
                updates = f"""{targets["gx"][i]} += scale * source_direction + direction * source_base;\n{(targets["gw"] if "gw" in selected else "")} += rx{i} * source_base + x{i} * source_direction;""".splitlines()
                lines.append(updates[0])
                if "gw" in selected:
                    lines.append(updates[1])
                lines.append("}")
            lines.append("}")
            return "\n".join(lines)
        for j in range(dims["y"]):
            basis = r"""{scalar} edge_base{j} = 0.0;
{scalar} edge_direction{j} = 0.0;
const {scalar} node_basis{j} = ry{j} * scale + y{j} * direction;""".format(scalar=scalar, j=j).splitlines()
            lines.extend(basis[:3] if "gx" in selected else basis[:2])
        for pairs in build_signed_ij_classes(
            spec["terms"], spec_y_offset=0, y_name_base=0
        ):
            projection = " + ".join(
                f"{float_literal(coefficient, dtype)} * g{k}"
                for _i, _j, k, coefficient in pairs.members[0].terms
            )
            lines += ["{", f"const {scalar} projection = {projection};"]
            for member in pairs.members:
                operation = "+=" if member.relative_sign > 0 else "-="
                projection_updates = r"""edge_base{j} {operation} x{i} * projection;
edge_direction{j} {operation} rx{i} * projection;
{source_gradient} {operation} node_basis{j} * projection;""".format(
                    j=member.j_name,
                    i=member.i_idx,
                    operation=operation,
                    source_gradient=targets["gx"][member.i_idx]
                    if "gx" in selected
                    else "",
                ).splitlines()
                lines.extend(
                    projection_updates[:3]
                    if "gx" in selected
                    else projection_updates[:2]
                )
            lines.append("}")
        lines += [f"{scalar} scalar_direction = 0.0;"]
        if "ge" in selected or (with_env and need_grad_env):
            lines.append(f"{scalar} scalar_base = 0.0;")
        for j in range(dims["y"]):
            lines.append(
                f"""{targets["gy"][j]} += scale * edge_direction{j} + direction * edge_base{j};\nscalar_direction += y{j} * edge_direction{j} + ry{j} * edge_base{j};"""
            )
            if "ge" in selected or (with_env and need_grad_env):
                lines.append(f"scalar_base += y{j} * edge_base{j};")
        if "gw" in selected:
            value = "e * scalar_direction" if with_env else "scalar_direction"
            if with_env and need_grad_env:
                value += " + re_value * scalar_base"
            lines.append(f"""{targets["gw"]} += {value};""")
        if "ge" in selected:
            lines.append(
                f"""{targets["ge"]} += w * scalar_direction + rw_value * scalar_base;"""
            )
        lines.append("}")
        return "\n".join(lines)
    return None


DESTINATION_DERIVATIVE_KERNEL = r"""__global__ __launch_bounds__({threads}, 2)
void {name}({parameters}) {{
{null_declarations}
const Index task = Index(blockIdx.x);
if (task >= Index(num_nodes) * {tiles}) return;
const int lane = threadIdx.x & 31;
const int neighbor_worker = threadIdx.x >> 5;
const int tile = int(task % {tiles});
const int u = tile * 32 + lane;
const int rank = int(task / {tiles});
const int d = node_order ? __ldg(node_order + rank) : rank;
__shared__ {scalar} neighbor_sums[{shared_width}];
{initialize_sums}
for (Index p = Index(dst_rowptr[d]) + neighbor_worker; p < dst_rowptr[d + 1]; p += {workers}) {{
const Index edge = Index(p);
const int s = __ldg(src + edge);

{edge_contractions}
}}
__syncthreads();
if (neighbor_worker == 0) {{
{output_reductions}
}}
}}
"""
