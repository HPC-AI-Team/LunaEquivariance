"""UUU convolution backward CUDA."""

from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import torch
from ....utils.precision import float_literal
from ...utils.contractions import ENV_PROGRAMS, PROGRAMS
from ...utils.layout import build_signed_ij_classes
from .forward import emit_contraction as emit_forward_contraction
from .double_backward import emit_contraction as emit_double_backward_contraction
from .triple_backward import emit_contraction as emit_triple_backward_contraction
from .utils import group_paths, sum_product_expressions
from .utils import (
    CUDA_OPERANDS,
    BACKWARD_OUTPUTS,
    FP64_PATH_LEAF_TERM_LIMIT,
    SHARED_MEMORY_BYTES,
)
from . import double_backward as double_backward_cuda
from . import triple_backward as triple_backward_cuda


def emit_path_contractions(
    op,
    spec,
    path,
    order,
    targets,
    *,
    with_env=False,
    need_grad_env=True,
    raw_l0=False,
    combined_l0=False,
    consume_q=False,
    resident_values=None,
    source_axis_first=False,
):
    """Emit only the Q/Z contractions consumed by the selected output roles.

    CG pairs are grouped by exact signed FP64 coefficient vectors, as in UVU
    convolution. Every edge attribute load preserves u along the path.
    """

    dtype = op.compute_dtype
    scalar = "double" if dtype == torch.float64 else "float"
    if order == 0:
        program = {
            "cg": {"q0": (("x", "y", "e"),) if with_env else (("x", "y"),)},
            "z": {},
            "x": {},
            "y": {},
            "w": {},
            "e": {},
            "forward": (("q0", "w"),),
        }
    else:
        program = deepcopy((ENV_PROGRAMS if with_env else PROGRAMS)[order])
        if with_env and not need_grad_env:
            # Frozen envelopes remain in primal products; their directions are zero.
            for kind in ("cg", "z", "x", "y", "w", "e"):
                for name, products in program.get(kind, {}).items():
                    program[kind][name] = tuple(
                        p for p in products if not any(r in ("re", "be") for r in p)
                    )
    selected = {}
    q_names, z_names, roles = set(), set(), set()
    for output, target in targets.items():
        if not spec["has_weight"] and output.removeprefix("g") in ("w", "rw"):
            continue
        if output == "gg":
            products = program["forward"]
        else:
            axis = output.removeprefix("g").removeprefix("r")
            products = program.get(axis, {}).get(output, ())
        selected[output] = products
        for product in products:
            for role in product:
                if role is None:
                    continue
                if role.startswith("q"):
                    q_names.add(role)
                elif role.startswith("z"):
                    z_names.add(role)
                else:
                    roles.add(role)
    for name in q_names:
        roles.update(
            r for product in program["cg"][name] for r in product if r is not None
        )
    for name in z_names:
        roles.update(
            r for product in program["z"][name] for r in product if r is not None
        )
    # Factor the symbolic feature/weight/envelope products before CG. This
    # lowers pair-product counts without changing the derivative expressions.
    forward_basis = None
    if order >= 2 and set(selected) == {"gg"}:
        forward_basis = defaultdict(list)
        for product in selected["gg"]:
            q = next(r for r in product if r is not None and r.startswith("q"))
            for pair in program["cg"][q]:
                feature = next(
                    r
                    for r in pair
                    if r is not None and r.removeprefix("r").removeprefix("b") == "x"
                )
                forward_basis[feature].append(
                    tuple(r for r in pair if r != feature)
                    + tuple(r for r in product if r != q)
                )
        q_names = set()
    feature_basis = {}
    if order >= 2 or (order == 1 and op.split_groups):
        for output, products in selected.items():
            if output.removeprefix("g").removeprefix("r") not in ("x", "y"):
                continue
            basis = defaultdict(list)
            for product in products:
                for position, role in enumerate(product):
                    if role is None or not role.startswith("z"):
                        continue
                    for matrix_product in program["z"][role]:
                        gradient = next(
                            r
                            for r in matrix_product
                            if r is not None
                            and r.removeprefix("r").removeprefix("b") == "g"
                        )
                        factors = (
                            *product[:position],
                            *(r for r in matrix_product if r != gradient),
                            *product[position + 1 :],
                        )
                        basis[gradient].append(factors)
            feature_basis[output] = dict(basis)
        z_names = {
            r
            for output, products in selected.items()
            if output not in feature_basis
            for product in products
            for r in product
            if r is not None and r.startswith("z")
        }
    lines = ["{"]
    dims = {"x": int(spec["d1"]), "y": int(spec["d2"]), "g": int(spec["d3"])}
    addresses = {
        "x": f"""Index(s) * {op.in1_dim} + {int(spec["x_offset"])} + u * {dims["x"]}""",
        "y": f"""Index(edge) * {op.in2_dim} + {int(spec["y_offset"])} + u * {dims["y"]}""",
        "g": f"""Index(d) * {op.out_dim} + {int(spec["out_offset"])} + u * {dims["g"]}""",
        "w": f"""Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u""",
        "e": "edge",
    }
    l0_address = f"Index(edge) * {op.l0_dim} + {op.l0_offsets.get(path, 0)} + u"
    for role in sorted(roles):
        axis, pointer = (
            role.removeprefix("r").removeprefix("b"),
            CUDA_OPERANDS.get(role, role),
        )
        for component in range(dims.get(axis, 1)):
            address = addresses[axis] + (f" + {component}" if axis in dims else "")
            value = f"__ldg({pointer} + {address})"
            if axis == "w" and not spec["has_weight"]:
                value = "1.0" if role == "w" else "0.0"
            if role in ("re", "be"):
                value = f"({pointer} ? {value} : 0.0)"
            if raw_l0 and axis == "g":
                pointer = "grad_l0" if role == "g" else "sl0"
                value = f"({pointer} ? __ldg({pointer} + {l0_address}) : 0.0)"
            elif combined_l0 and axis == "g":
                pointer = "grad_l0" if role == "g" else "sl0"
                value = (
                    f"({value} + ({pointer} ? __ldg({pointer} + {l0_address}) : 0.0))"
                )
            name = (
                role + str(component)
                if axis in dims
                else role + "_value"
                if role in ("rw", "re")
                else role
            )
            if resident_values is not None:
                value = resident_values.get(name, value)
            lines.append(f"const {scalar} {name} = {value};")
    if order == 0:
        return emit_forward_contraction(op, spec, targets, lines, with_env=with_env)
    if (
        order == 1
        and "gy" in selected
        and set(selected) <= {"gx", "gy", "gw", "ge"}
        and ("gw" in selected or "ge" in selected)
    ):
        dtype = op.compute_dtype
        scalar = "double" if dtype == torch.float64 else "float"
        dims = {"x": int(spec["d1"]), "y": int(spec["d2"]), "g": int(spec["d3"])}
        if source_axis_first and "gx" in selected and dims["x"] < dims["y"]:
            # Contract the shorter source axis before weight scaling. The
            # same source projection supplies both source and scalar grads.
            scale = "w * e" if with_env else "w"
            lines.append(f"const {scalar} scale = {scale};")
            for i in range(dims["x"]):
                lines.append(
                    f"{scalar} source_base{i} = 0.0;\nconst {scalar} scaled_source{i} = scale * x{i};"
                )
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
                    i, j = member.i_idx, member.j_name
                    lines.append(
                        f"""source_base{i} {operation} y{j} * projection;\n{targets["gy"][j]} {operation} scaled_source{i} * projection;"""
                    )
                lines.append("}")
            lines.append(f"{scalar} scalar_grad = 0.0;")
            for i in range(dims["x"]):
                lines.append(
                    f"""{targets["gx"][i]} += scale * source_base{i};\nscalar_grad += x{i} * source_base{i};"""
                )
            if "gw" in selected:
                factor = "e * " if with_env else ""
                lines.append(f"""{targets["gw"]} += {factor}scalar_grad;""")
            if "ge" in selected:
                lines.append(f"""{targets["ge"]} += w * scalar_grad;""")
            lines.append("}")
            return "\n".join(lines)
        # The unweighted edge derivative also contracts with y to give the
        # weight/envelope derivative. Reuse its CG projection rather than
        # independently forming CG(x, y) and contracting with grad_out.
        scale = "w * e" if with_env else "w"
        if "gx" in selected:
            lines.append(f"const {scalar} scale = {scale};")
        classes = build_signed_ij_classes(spec["terms"], spec_y_offset=0, y_name_base=0)
        shared_basis = (
            "gx" in selected
            and op.split_groups
            and not with_env
            and not raw_l0
            and not combined_l0
            and dims["x"] >= 3
            and len(classes) > dims["y"]
        )
        for j in range(dims["y"]):
            lines.append(f"{scalar} edge_grad{j} = 0.0;")
            if shared_basis:
                lines.append(f"const {scalar} node_basis{j} = y{j} * scale;")
        for pairs in classes:
            projection = " + ".join(
                f"{float_literal(coefficient, dtype)} * g{k}"
                for _i, _j, k, coefficient in pairs.members[0].terms
            )
            lines += ["{", f"const {scalar} projection = {projection};"]
            if "gx" in selected and not shared_basis:
                lines.append(
                    f"const {scalar} weighted_projection = scale * projection;"
                )
            for member in pairs.members:
                operation = "+=" if member.relative_sign > 0 else "-="
                lines.append(
                    f"edge_grad{member.j_name} {operation} x{member.i_idx} * projection;"
                )
                if "gx" in selected:
                    value = (
                        f"node_basis{member.j_name} * projection"
                        if shared_basis
                        else f"y{member.j_name} * weighted_projection"
                    )
                    lines.append(
                        f"""{targets["gx"][member.i_idx]} {operation} {value};"""
                    )
            lines.append("}")
        if "gx" not in selected:
            lines.append(f"const {scalar} scale = {scale};")
        lines.append(f"{scalar} scalar_grad = 0.0;")
        for j in range(dims["y"]):
            lines.append(f"""{targets["gy"][j]} += scale * edge_grad{j};""")
            lines.append(f"scalar_grad += y{j} * edge_grad{j};")
        if "gw" in selected:
            scale = "e * " if with_env else ""
            lines.append(f"""{targets["gw"]} += {scale}scalar_grad;""")
        if "ge" in selected:
            lines.append(f"""{targets["ge"]} += w * scalar_grad;""")
        lines.append("}")
        return "\n".join(lines)
    if order == 2:
        result = emit_double_backward_contraction(
            op,
            spec,
            targets,
            lines,
            selected,
            with_env=with_env,
            need_grad_env=need_grad_env,
            source_axis_first=source_axis_first,
        )
        if result is not None:
            return result
    if order == 3:
        result = emit_triple_backward_contraction(
            op,
            spec,
            targets,
            lines,
            selected,
            with_env=with_env,
            need_grad_env=need_grad_env,
            source_axis_first=source_axis_first,
        )
        if result is not None:
            return result
    if forward_basis is not None:
        for feature, products in forward_basis.items():
            for component in range(dims["y"]):
                expression = sum_product_expressions(products, 0, component, 0)
                lines.append(
                    f"const {scalar} forward_{feature}_{component} = {expression};"
                )
    feature_family_stream = bool(feature_basis) and (
        op.compute_dtype == torch.float64
        and order == 3
        and op.split_groups
        and sum(len(spec["terms"]) for spec in op.generated_specs)
        >= FP64_PATH_LEAF_TERM_LIMIT
    )
    for output, basis in () if feature_family_stream else feature_basis.items():
        axis = output.removeprefix("g").removeprefix("r")
        for gradient, products in basis.items():
            for component in range(dims["y" if axis == "x" else "x"]):
                expression = sum_product_expressions(
                    products,
                    0 if axis == "x" else component,
                    component if axis == "x" else 0,
                    0,
                )
                lines.append(
                    f"const {scalar} basis_{output}_{gradient}_{component} = {expression};"
                )
    for name in sorted(z_names):
        for k in range(dims["g"]):
            expression = sum_product_expressions(program["z"][name], 0, 0, k)
            lines.append(f"const {scalar} {name}_{k} = {expression};")
    classes = build_signed_ij_classes(spec["terms"], spec_y_offset=0, y_name_base=0)
    if forward_basis is not None:
        for cls, pairs in enumerate(classes):
            additions = []
            for member in pairs.members:
                value = " + ".join(
                    f"{feature}{member.i_idx} * forward_{feature}_{member.j_name}"
                    for feature in forward_basis
                )
                additions.append(
                    ("+" if member.relative_sign > 0 else "-") + f" ({value})"
                )
            lines.append(
                f"""const {scalar} forward_pair{cls} = {" ".join(additions)};"""
            )
            for _i, _j, k, coefficient in pairs.members[0].terms:
                target = (
                    targets["gg"][k]
                    if not isinstance(targets["gg"], str)
                    else targets["gg"]
                )
                lines.append(
                    f"{target} += {float_literal(coefficient, dtype)} * forward_pair{cls};"
                )
    stream_q = (
        (order == 3 or consume_q)
        and op.split_groups
        and q_names
        and not any(
            output.removeprefix("g").removeprefix("r") in ("x", "y")
            for output in selected
        )
    )
    for name in sorted(q_names):
        if stream_q:
            lines.append("{")
        for k in range(dims["g"]):
            lines.append(f"{scalar} {name}_{k} = 0.0;")
        for cls, pairs in enumerate(classes):
            additions = []
            for member in pairs.members:
                value = sum_product_expressions(
                    program["cg"][name], member.i_idx, member.j_name, 0
                )
                additions.append(
                    ("+" if member.relative_sign > 0 else "-") + f" ({value})"
                )
            expression = " ".join(additions)
            lines.append(f"const {scalar} {name}_pair{cls} = {expression};")
            for _i, _j, k, coefficient in pairs.members[0].terms:
                lines.append(
                    f"{name}_{k} += {float_literal(coefficient, dtype)} * {name}_pair{cls};"
                )
        if stream_q:
            for output, products in selected.items():
                for k in range(dims["g"]):
                    parts = []
                    for product in products:
                        if name not in product:
                            continue
                        factors = [
                            r + f"_{k}"
                            if r.startswith(("q", "z"))
                            else r + str(k)
                            if r.removeprefix("r").removeprefix("b") in ("x", "y", "g")
                            else r + "_value"
                            if r in ("rw", "re")
                            else r
                            for r in product
                            if r is not None
                        ]
                        parts.append(" * ".join(factors))
                    if parts:
                        target = (
                            targets[output][k]
                            if not isinstance(targets[output], str)
                            else targets[output]
                        )
                        lines.append(f"""{target} += {" + ".join(parts)};""")
            lines.append("}")
    if feature_basis:
        # Transpose the sparse CG contraction: project each incoming-gradient
        # vector once per exact signed coefficient class, then reuse that
        # scalar for every member and both feature-gradient outputs. No Q/Z
        # vector or cross-edge temporary is needed by these consumers.
        gradients = sorted(
            {gradient for basis in feature_basis.values() for gradient in basis}
        )
        families = (
            [(gradient,) for gradient in gradients]
            if feature_family_stream
            else [tuple(gradients)]
        )
        for family in families:
            if feature_family_stream:
                lines.append("{")
                for output, basis in feature_basis.items():
                    axis = output.removeprefix("g").removeprefix("r")
                    for gradient in family:
                        if gradient not in basis:
                            continue
                        for component in range(dims["y" if axis == "x" else "x"]):
                            expression = sum_product_expressions(
                                basis[gradient],
                                0 if axis == "x" else component,
                                component if axis == "x" else 0,
                                0,
                            )
                            lines.append(
                                f"const {scalar} basis_{output}_{gradient}_{component} = {expression};"
                            )
            for pairs in classes:
                lines.append("{")
                for gradient in family:
                    expression = " + ".join(
                        f"{float_literal(coefficient, dtype)} * {gradient}{k}"
                        for _i, _j, k, coefficient in pairs.members[0].terms
                    )
                    lines.append(
                        f"const {scalar} projection_{gradient} = {expression};"
                    )
                for output, basis in feature_basis.items():
                    axis = output.removeprefix("g").removeprefix("r")
                    for member in pairs.members:
                        component = member.i_idx if axis == "x" else member.j_name
                        other = member.j_name if axis == "x" else member.i_idx
                        target = targets[output][component]
                        operation = "+=" if member.relative_sign > 0 else "-="
                        for gradient in basis:
                            if gradient in family:
                                lines.append(
                                    f"{target} {operation} basis_{output}_{gradient}_{other} * projection_{gradient};"
                                )
                lines.append("}")
            if feature_family_stream:
                lines.append("}")
    for output, products in selected.items():
        if output == "gg" and forward_basis is not None:
            continue
        axis = output.removeprefix("g").removeprefix("r")
        if axis in ("x", "y"):
            if output in feature_basis:
                continue
            for i, j, k, coefficient in spec["terms"]:
                parts = []
                for product in products:
                    factors = [
                        r + f"_{k}"
                        if r.startswith("z")
                        else r + str(j if axis == "x" else i)
                        if r.removeprefix("r").removeprefix("b") in ("x", "y", "g")
                        else r + "_value"
                        if r in ("rw", "re")
                        else r
                        for r in product
                        if r is not None
                    ]
                    parts.append(" * ".join(factors))
                if parts:
                    target = targets[output][i if axis == "x" else j]
                    lines.append(
                        f"""{target} += {float_literal(coefficient, dtype)} * ({" + ".join(parts)});"""
                    )
        elif not stream_q:
            for k in range(dims["g"]):
                parts = []
                for product in products:
                    factors = [
                        r + f"_{k}"
                        if r.startswith(("q", "z"))
                        else r + str(k)
                        if r.removeprefix("r").removeprefix("b") in ("x", "y", "g")
                        else r + "_value"
                        if r in ("rw", "re")
                        else r
                        for r in product
                        if r is not None
                    ]
                    parts.append(" * ".join(factors))
                if parts:
                    target = (
                        targets[output][k]
                        if not isinstance(targets[output], str)
                        else targets[output]
                    )
                    lines.append(f"""{target} += {" + ".join(parts)};""")
    lines.append("}")
    return "\n".join(lines)


def make_scalar_source_kernel(op, parameters, paths, scalar, kernel_suffix, source_names):
    """Distribute a diagonal scalar CG path's angular entries across lanes."""
    channels = int(paths[0][1]["channels"])
    angular = 2
    channels_per_warp = 32 // angular
    tiles = (channels + channels_per_warp - 1) // channels_per_warp
    offset = int(paths[0][1]["x_offset"])
    lines = []
    for _, spec in paths:
        dimension = int(spec["d2"])
        coefficients = {
            int(j): float_literal(c, op.compute_dtype) for _, j, _, c in spec["terms"]
        }
        lines.append(
            r"""{{
const {scalar} w = u < {channels} ? __ldg(weight + Index(edge) * {weight_numel} + {weight_offset} + u) : 0.0;
{scalar} weight_grad = 0.0;""".format(
                scalar=scalar,
                channels=channels,
                weight_numel=op.weight_numel,
                weight_offset=int(spec["weight_offset"]),
            )
        )
        for start in range(0, dimension, angular):
            coefficient = " : ".join(
                f"j == {j - start} ? {coefficients[j]}"
                for j in range(start, min(start + angular - 1, dimension))
            )
            coefficient = (coefficient + " : " if coefficient else "") + coefficients[
                min(start + angular - 1, dimension - 1)
            ]
            lines += [
                r"""{{
const int component = j + {start};
const bool active = u < {channels} && component < {dimension};
const {scalar} gradient = active ? __ldg(grad_out + Index(d) * {out_dim} + {out_offset} + u * {dimension} + component) : 0.0;
const {scalar} attribute = active ? __ldg(y + Index(edge) * {in2_dim} + {y_offset} + u * {dimension} + component) : 0.0;
const {scalar} projection = ({coefficient}) * gradient;
const {scalar} edge_grad = source * projection;
node_grad += attribute * (w * projection);
weight_grad += attribute * edge_grad;
if (active) out_y[Index(edge) * {in2_dim} + {y_offset} + u * {dimension} + component] = w * edge_grad;
}}""".format(
                    start=start,
                    channels=channels,
                    dimension=dimension,
                    scalar=scalar,
                    out_dim=op.out_dim,
                    out_offset=int(spec["out_offset"]),
                    in2_dim=op.in2_dim,
                    y_offset=int(spec["y_offset"]),
                    coefficient=coefficient,
                )
            ]
        lines += [
            r"""#pragma unroll
for (int delta = {first_delta}; delta; delta >>= 1) {{
weight_grad += __shfl_down_sync(0xffffffffu, weight_grad, delta, {angular});
}}
if (!j && u < {channels}) out_w[Index(edge) * {weight_numel} + {weight_offset} + u] = weight_grad;
}}""".format(
                first_delta=angular // 2,
                angular=angular,
                channels=channels,
                weight_numel=op.weight_numel,
                weight_offset=int(spec["weight_offset"]),
            )
        ]
    name = f"lunatp_uuuconv_bwd_source_edge{kernel_suffix}"
    return (
        r"""__global__ __launch_bounds__(128, 4)
void {name}({parameters}) {{
{null_declarations}
const Index task = Index(blockIdx.x) * 4 + (threadIdx.x >> 5);
if (task >= Index(num_nodes) * {tiles}) return;
const int lane = threadIdx.x & 31;
const int j = lane % {angular};
const int u = int(task % {tiles}) * {channels_per_warp} + lane / {angular};
const int rank = int(task / {tiles});
const int s = source_node_order ? __ldg(source_node_order + rank) : rank;
const {scalar} source = u < {channels} ? __ldg(x + Index(s) * {in1_dim} + {offset} + u) : 0.0;
{scalar} node_grad = 0.0;
for (Index p = src_rowptr[s]; p < src_rowptr[s + 1]; ++p) {{
const Index edge = __ldg(edge_ids + p);
const int d = __ldg(dst + edge);

{edge_contractions}
}}
#pragma unroll
for (int delta = {first_delta}; delta; delta >>= 1) {{
node_grad += __shfl_down_sync(0xffffffffu, node_grad, delta, {angular});
}}
if (!j && u < {channels}) out_x[Index(s) * {in1_dim} + {offset} + u] = node_grad;
}}
""".format(
            name=name,
            parameters=parameters,
            null_declarations="\n".join(
                f"const {scalar}* {name} = nullptr;"
                for name in ("re", "se", "grad_l0", "sl0") if name not in source_names
            ),
            tiles=tiles,
            angular=angular,
            channels_per_warp=channels_per_warp,
            scalar=scalar,
            channels=channels,
            in1_dim=op.in1_dim,
            offset=offset,
            first_delta=angular // 2,
            edge_contractions="\n".join(lines),
        ),
        name,
        tiles,
        128,
        4,
    )


def make_source_edge_kernel(
    op,
    order,
    parameters,
    *,
    outputs,
    with_env,
    with_l0,
    need_grad_env,
    env_parts,
    paths=None,
    tile_offset=0,
    kernel_suffix="",
    node_offsets=None,
    node_stride=None,
    accumulated_edges=(),
    resident_blocks=None,
    source_names,
):
    """Own complete edge derivatives in the same traversal as source gradients.

    A UUU channel has one owner in source CSR, so its complete edge-gradient
    slice can be stored directly without UVU's cross-channel partial buffers.
    """
    channels = (
        op.max_channels if paths is None else max(int(s["channels"]) for _, s in paths)
    )
    paths = list(enumerate(op.generated_specs)) if paths is None else paths
    specs = [spec for _, spec in paths]
    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    if (
        order == 1
        and scalar == "double"
        and set(outputs) == {"x", "y", "w"}
        and not with_env
        and not with_l0
        and node_offsets is None
        and not any(key in accumulated_edges for key, _ in group_paths(specs, "y"))
        and len(group_paths(specs, "x")) == 1
        and len(group_paths(specs, "y")) == len(specs)
        and max(int(spec["d2"]) for spec in specs) >= 7
        and all(
            int(spec["d1"]) == 1
            and int(spec["d2"]) == int(spec["d3"])
            and int(spec["d2"]) <= 16
            and spec["has_weight"]
            and len(spec["terms"]) == int(spec["d2"])
            and sorted(int(j) for _, j, _, _ in spec["terms"])
            == list(range(int(spec["d2"])))
            and all(i == 0 and j == k for i, j, k, coefficient in spec["terms"])
            for spec in specs
        )
    ):
        return make_scalar_source_kernel(op, parameters, paths, scalar, kernel_suffix, source_names)
    # Short first-order programs amortize two neighboring-edge workers.
    workers = (
        2 if order == 1 and op.source_edge_width <= 32 else int(op.source_edge_workers)
    )
    shared_inputs = order == 2 and node_offsets is not None and workers in (2, 4)
    threads = 128 if workers == 1 else 32 * workers
    tasks_per_block = 4 if workers == 1 else 1

    x_groups = group_paths(specs, "x")
    y_groups = group_paths(specs, "y")
    source_width = sum(component_width for (_, component_width, _), _ in x_groups)
    if (
        order == 2
        and scalar == "double"
        and node_offsets is None
        and workers in (2, 4)
        and not with_env
        and not with_l0
        and set(outputs) == {"x", "y", "w", "g"}
        and source_width >= 9
        and 32 * source_width * (workers + 2) * 8 <= SHARED_MEMORY_BYTES
    ):
        shared_inputs = True
    x_indices = {key: i for i, (key, _) in enumerate(x_groups)}
    y_indices = {key: i for i, (key, _) in enumerate(y_groups)}
    node_roles = [n for n in ("x", "rx") if n in outputs]
    edge_roles = [n for n in ("y", "ry") if n in outputs]
    env_roles = [n for n in ("e", "re") if n in outputs]
    edge_width = max(component_width for (_, component_width, _), _ in y_groups)
    coalesced_first = (
        order == 1
        and op.split_groups
        and not with_env
        and not with_l0
        and set(outputs) == {"x", "y", "w"}
        and len(y_groups) > 1
        and threads * (edge_width + source_width) * (8 if scalar == "double" else 4)
        <= SHARED_MEMORY_BYTES
    )
    if (
        order == 3
        and scalar == "double"
        and workers in (2, 4)
        and node_roles
        and all(role in outputs for role in ("y", "ry", "w", "rw"))
        and not with_env
        and not with_l0
        and 32
        * source_width
        * (workers * len(node_roles) + 3)
        * (8 if scalar == "double" else 4)
        <= SHARED_MEMORY_BYTES
    ):
        shared_inputs = True
    tiles = (channels + 31) // 32
    task = (
        "Index(blockIdx.x) * 4 + (threadIdx.x >> 5)"
        if workers == 1
        else "Index(blockIdx.x)"
    )
    lines = []
    if coalesced_first:
        lines.append(f"__shared__ {scalar} edge_stores[{threads * edge_width}];")
    if workers > 1:
        lines.append("const int neighbor_worker = threadIdx.x >> 5;")
    for role in node_roles:
        for group, ((_, dimension, _), _) in enumerate(x_groups):
            for component in range(dimension):
                lines.append(f"{scalar} acc_{role}_{group}_{component} = 0.0;")
    # All neighbor warps share one invariant node/channel feature slice.
    # Volatile shared loads bound its register lifetime to the current path.
    resident_values = {}
    if shared_inputs:
        width = sum(component_width for (_, component_width, _), _ in x_groups)
        source_roles = ("x", "rx", "bx") if order == 3 else ("x", "rx")
        lines.append(
            f"__shared__ volatile {scalar} source_inputs[{32 * len(source_roles) * width}];"
        )
        lines.append("if (neighbor_worker == 0) {")
        slot = 0
        for group, ((offset, dimension, channels), _) in enumerate(x_groups):
            for role in source_roles:
                for component in range(dimension):
                    address = f"Index(s) * {op.in1_dim} + {offset} + u * {dimension} + {component}"
                    pointer = CUDA_OPERANDS.get(role, role)
                    lines.append(
                        f"source_inputs[{slot} * 32 + lane] = u < {channels} ? __ldg({pointer} + {address}) : 0.0;"
                    )
                    resident_values[(group, role + str(component))] = (
                        f"source_inputs[{slot} * 32 + lane]"
                    )
                    slot += 1
        lines += ["}", "__syncthreads();"]
    begin = "src_rowptr[s]" + (" + neighbor_worker" if workers > 1 else "")
    step = f"p += {workers}" if workers > 1 else "++p"
    node_setup = "\n".join(lines) + "\n" if lines else ""
    lines = []
    for role in edge_roles:
        for group, ((_, dimension, _), _) in enumerate(y_groups):
            for component in range(dimension):
                lines.append(f"{scalar} acc_{role}_{group}_{component} = 0.0;")
    for role in env_roles:
        lines.append(f"{scalar} acc_{role} = 0.0;")
    for path, spec in paths:
        x_key = int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"])
        y_key = int(spec["y_offset"]), int(spec["d2"]), int(spec["channels"])
        lines.append(f"""if (u < {int(spec["channels"])}) {{""")
        targets = {}
        for output in BACKWARD_OUTPUTS:
            role = output.removeprefix("g")
            if role in node_roles:
                targets[output] = [
                    f"acc_{role}_{x_indices[x_key]}_{i}" for i in range(int(spec["d1"]))
                ]
            elif role in edge_roles:
                targets[output] = [
                    f"acc_{role}_{y_indices[y_key]}_{j}" for j in range(int(spec["d2"]))
                ]
            elif role in env_roles:
                targets[output] = f"acc_{role}"
            elif role in ("w", "rw") and role in outputs and spec["has_weight"]:
                lines.append(f"{scalar} acc_{role} = 0.0;")
                targets[output] = f"acc_{role}"
        raw = with_l0 and path in op.l0_offsets
        lines.append(
            emit_path_contractions(
                op,
                spec,
                path,
                order,
                targets,
                with_env=with_env,
                need_grad_env=need_grad_env,
                combined_l0=raw and not with_env,
                resident_values={
                    name: value
                    for (group, name), value in resident_values.items()
                    if group == x_indices[x_key]
                }
                if shared_inputs
                else None,
                source_axis_first=(
                    node_offsets is not None
                    and order != 3
                    and (order != 2 or scalar == "double" or resident_blocks == 3)
                ),
            )
        )
        if raw and with_env:
            raw_targets = {
                r: t
                for r, t in targets.items()
                if r.removeprefix("g") not in ("e", "re")
            }
            lines.append(
                emit_path_contractions(op, spec, path, order, raw_targets, raw_l0=True)
            )
        for role in ("w", "rw"):
            if role in outputs and spec["has_weight"]:
                lines.append(
                    f"""out_{role}[Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u] = acc_{role};"""
                )
        lines.append("}")
    for role in edge_roles:
        for group, ((offset, dimension, channels), _) in enumerate(y_groups):
            operation = (
                "+=" if (offset, dimension, channels) in accumulated_edges else "="
            )
            if coalesced_first and dimension > 1:
                lines.append("{")
                for component in range(dimension):
                    lines.append(
                        f"edge_stores[(threadIdx.x >> 5) * {32 * edge_width} + lane * {dimension} + {component}] = acc_{role}_{group}_{component};"
                    )
                lines += [
                    r"""__syncwarp();
const int valid = min(32, {channels} - tile * 32);
for (int local = lane; local < valid * {dimension}; local += 32) {{
out_{role}[Index(edge) * {in2_dim} + {offset} + tile * 32 * {dimension} + local] {operation} edge_stores[(threadIdx.x >> 5) * {warp_store_width} + local];
}}
__syncwarp();
}}""".format(
                        channels=channels,
                        dimension=dimension,
                        role=role,
                        in2_dim=op.in2_dim,
                        offset=offset,
                        operation=operation,
                        warp_store_width=32 * edge_width,
                    )
                ]
            else:
                for component in range(dimension):
                    lines.append(
                        f"if (u < {channels}) out_{role}[Index(edge) * {op.in2_dim} + {offset} + u * {dimension} + {component}] {operation} acc_{role}_{group}_{component};"
                    )
    for role in env_roles:
        lines += [
            r"""#pragma unroll
for (int delta = 16; delta; delta >>= 1) {{
acc_{role} += __shfl_down_sync(0xffffffffu, acc_{role}, delta);
}}
if (!lane) partial_{role}[Index(edge) * {env_parts} + {tile_offset} + tile] = acc_{role};""".format(
                role=role,
                env_parts=env_parts,
                tile_offset=tile_offset,
            )
        ]

    edge_contractions = "\n".join(lines)
    lines = []
    components = [
        (role, group, component, offset, dimension, channels)
        for role in node_roles
        for group, ((offset, dimension, channels), _) in enumerate(x_groups)
        for component in range(dimension)
    ]
    if workers > 1:
        lines.append(f"__shared__ {scalar} source_sums[{threads * len(components)}];")
        for index, (role, group, component, _, _, _) in enumerate(components):
            lines.append(
                f"source_sums[{index} * {threads} + threadIdx.x] = acc_{role}_{group}_{component};"
            )
        lines.append("__syncthreads();")
    for index, (role, group, component, offset, dimension, channels) in enumerate(
        components
    ):
        value = f"value_{index}" if workers > 1 else f"acc_{role}_{group}_{component}"
        if node_offsets is not None:
            offset = node_offsets[(offset, dimension, channels)]
        stride = op.in1_dim if node_stride is None else node_stride
        store = f"out_{role}[Index(s) * {stride} + {offset} + u * {dimension} + {component}] = {value};"
        if workers > 1:
            lines.append(
                r"""if (neighbor_worker == 0 && u < {channels}) {{
{scalar} {value} = 0.0;
#pragma unroll
for (int worker = 0; worker < {workers}; ++worker) {value} += source_sums[{index} * {threads} + worker * 32 + lane];
{store}
}}""".format(
                    channels=channels,
                    scalar=scalar,
                    value=value,
                    workers=workers,
                    index=index,
                    threads=threads,
                    store=store,
                )
            )
        else:
            lines.append(r"""if (u < {channels}) {{
{store}
}}""".format(channels=channels, store=store))
    node_reductions = "\n" + "\n".join(lines) if lines else ""
    name = f"""lunatp_uuuconv_{("fwd", "bwd", "double_bwd", "triple_bwd")[order]}_source_edge{kernel_suffix}"""
    # Two requested node-feature families retain both complete angular slices.
    if resident_blocks is None:
        resident_blocks = (
            2
            if scalar == "double"
            or order in (1, 2)
            or (len(node_roles) > 1 and len(components) > 8)
            else 4
        )
    return (
        r"""__global__ __launch_bounds__({threads}, {resident_blocks})
void {name}({parameters}) {{
{null_declarations}
const Index task = {task};
if (task >= Index(num_nodes) * {tiles}) return;
const int lane = threadIdx.x & 31;
const int tile = int(task % {tiles});
const int u = tile * 32 + lane;
const int rank = int(task / {tiles});
const int s = source_node_order ? __ldg(source_node_order + rank) : rank;
{node_setup}for (Index p = {begin}; p < src_rowptr[s + 1]; {step}) {{
const Index edge = __ldg(edge_ids + p);
const int d = __ldg(dst + edge);

{edge_contractions}
}}{node_reductions}
}}
""".format(
            threads=threads,
            resident_blocks=resident_blocks,
            name=name,
            parameters=parameters,
            null_declarations="\n".join(
                f"const {scalar}* {name} = nullptr;"
                for name in ("re", "se", "grad_l0", "sl0") if name not in source_names
            ),
            task=task,
            tiles=tiles,
            node_setup=node_setup,
            begin=begin,
            step=step,
            edge_contractions=edge_contractions,
            node_reductions=node_reductions,
        ),
        name,
        tiles,
        threads,
        tasks_per_block,
    )


def make_destination_reduction(op, order, scalar, destination_bundles):
    kernels, launches = [], []
    covered = sorted(
        {
            key
            for paths in destination_bundles
            for key, _ in group_paths([spec for _, spec in paths], "x")
        }
    )
    ranges = " || ".join(
        f"(component >= {offset} && component < {offset + dimension * channels})"
        for offset, dimension, channels in covered
    )
    roles = ("x", "rx") if order == 3 else ("x",)
    partials = {role: "partial_" + role if order == 3 else "partial" for role in roles}
    stores = {role: "out_" + role if order == 3 else "out" for role in roles}
    values = {role: "v" + role if order == 3 else "value" for role in roles}
    accumulators = "\n".join(f"{scalar} {values[role]} = 0.0;" for role in roles)
    partial_sums = "\n".join(
        f"{values[role]} += {partials[role]}[edge * {op.in1_dim} + component];"
        for role in roles
    )
    output_stores = "\n".join(
        f"{stores[role]}[index] = {values[role]};" for role in roles
    )
    name = "uuuconv_reduce_destination_source"
    params = ", ".join(
        [f"const {scalar}* __restrict__ {partials[role]}" for role in roles]
        + [f"{scalar}* __restrict__ {stores[role]}" for role in roles]
        + [
            "const EdgeIndex* __restrict__ src_rowptr",
            "const EdgeIndex* __restrict__ edge_ids",
            "int num_nodes",
        ]
    )
    args = ", ".join(
        ["destination_partial_" + role for role in roles]
        + ["out_" + role for role in roles]
        + ["srp", "ep", "num_nodes"]
    )
    kernels.append(
        r"""__global__ void {name}({params}) {{
const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
if (index >= Index(num_nodes) * {in1_dim}) return;
const int node = int(index / {in1_dim});
const int component = int(index % {in1_dim});
if (!({ranges})) return;
{accumulators}
for (Index p = src_rowptr[node]; p < src_rowptr[node + 1]; ++p) {{
const Index edge = __ldg(edge_ids + p);
{partial_sums}
}}
{stores}
}}
""".format(
            name=name,
            params=params,
            in1_dim=op.in1_dim,
            ranges=ranges,
            accumulators=accumulators,
            partial_sums=partial_sums,
            stores=output_stores,
        )
    )
    calls = [
        "if (num_nodes > 0) {",
        f"{name}<<<(Index(num_nodes) * {op.in1_dim} + 255) / 256, 256, 0, stream>>>({args});",
        "}",
    ]
    launches += calls
    return kernels, launches


def make_edge_reductions(op, scalar, partial_outputs, g_parts, g_slices, g_span):
    kernels, launches = [], []
    for traversal, names, dimension, rowptr in (
        (
            "x",
            [n for n in ("x", "rx") if n in partial_outputs],
            op.in1_dim,
            "src_rowptr",
        ),
        ("g", [n for n in ("g",) if n in partial_outputs], op.out_dim, "dst_rowptr"),
    ):
        if not names:
            continue
        params = ", ".join(
            f"const {scalar}* edge_{n}, {scalar}* tensor_{n}" for n in names
        )
        accumulators = "\n".join(f"{scalar} value_{n} = 0.0;" for n in names)
        csr_reductions = []
        if traversal == "g" and g_parts > 1:
            for group_id, ((offset, dim, channels), starts) in enumerate(
                g_slices.items()
            ):
                partial_sums = "\n".join(
                    f"value_{n} += edge_{n}[Index(edge) * {g_span} + {start} + local];"
                    for start in starts
                    for n in names
                )
                csr_reductions.append(
                    r"""{branch} (component >= {offset} && component < {end}) {{
const int local = component - {offset};
for (EdgeIndex edge = {rowptr}[node]; edge < {rowptr}[node + 1]; ++edge) {{
{partial_sums}
}}
}}""".format(
                        branch="if" if group_id == 0 else "else if",
                        offset=offset,
                        end=offset + dim * channels,
                        rowptr=rowptr,
                        partial_sums=partial_sums,
                    )
                )
        else:
            partial_sums = "\n".join(
                f"value_{n} += edge_{n}[Index(edge) * {dimension} + component];"
                for n in names
            )
            csr_reductions.append(
                r"""for (EdgeIndex p = {rowptr}[node]; p < {rowptr}[node + 1]; ++p) {{
const EdgeIndex edge = {edge};
{partial_sums}
}}""".format(
                    rowptr=rowptr,
                    edge="edge_ids[p]" if traversal == "x" else "p",
                    partial_sums=partial_sums,
                )
            )
        output_stores = "\n".join(f"tensor_{n}[index] = value_{n};" for n in names)
        name = f"lunatp_uuuconv_reduce_edge_{traversal}"
        kernels.append(
            r"""__global__ void {name}({params}, const EdgeIndex* {rowptr}, const EdgeIndex* edge_ids, int num_nodes) {{
const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
if (index >= Index(num_nodes) * {dimension}) return;
const int node = int(index / {dimension});
const int component = int(index % {dimension});
{accumulators}
{csr_reductions}
{stores}
}}""".format(
                name=name,
                params=params,
                rowptr=rowptr,
                dimension=dimension,
                accumulators=accumulators,
                csr_reductions="\n".join(csr_reductions),
                stores=output_stores,
            )
        )
        reduce_args = ", ".join(
            f"out_{n}, tensor_{n}.data_ptr<{scalar}>()" for n in names
        )
        graph_args = "srp, ep" if traversal == "x" else "drp, ep"
        calls = [
            r"""if (num_nodes > 0) {{
dim3 grid((Index(num_nodes) * {dimension} + 255) / 256);
{name}<<<grid, 256, 0, stream>>>({reduce_args}, {graph_args}, num_nodes);
}}""".format(
                dimension=dimension,
                name=name,
                reduce_args=reduce_args,
                graph_args=graph_args,
            )
        ]
        launches += calls
    return kernels, launches


def make_envelope_reduction(scalar, env_names, env_parts):
    kernels, launches = [], []
    env_parameters = ", ".join(
        f"const {scalar}* partial_{n}, {scalar}* out_{n}" for n in env_names
    )
    partial_reductions = "\n".join(
        r"""{scalar} value_{role} = 0.0;
for (int p = 0; p < {env_parts}; ++p) value_{role} += partial_{role}[edge * {env_parts} + p];
out_{role}[edge] = value_{role};""".format(scalar=scalar, role=n, env_parts=env_parts)
        for n in env_names
    )
    kernels.append(
        r"""__global__ void lunatp_uuuconv_reduce_env({env_parameters}, Index num_edges) {{
const Index edge = Index(blockIdx.x) * blockDim.x + threadIdx.x;
if (edge >= num_edges) return;
{partial_reductions}
}}""".format(
            env_parameters=env_parameters,
            partial_reductions=partial_reductions,
        )
    )
    env_args = ", ".join(f"partial_{n}, out_{n}" for n in env_names)
    calls = [r"""if (num_edges > 0) {{
dim3 grid((num_edges + 255) / 256);
lunatp_uuuconv_reduce_env<<<grid, 256, 0, stream>>>({env_args}, num_edges);
}}""".format(env_args=env_args)]
    launches += calls
    return kernels, launches


def make_source_node_reduction(op, scalar, source_node_parts, source_node_span):
    """Reduce bounded source-node partials into x and rx gradients."""
    group_reductions = []
    for key, _ in group_paths(op.generated_specs, "x"):
        offset, dimension, channels = key
        partial_sums = "\n".join(
            f"vx += partial_x[node * {source_node_span} + {start} + feature - {offset}];\nvrx += partial_rx[node * {source_node_span} + {start} + feature - {offset}];"
            for owner, _, start in source_node_parts
            if owner == key
        )
        group_reductions.append(
            f"""if (feature >= {offset} && feature < {offset + dimension * channels}) {{\n{(partial_sums + chr(10) if partial_sums else "")}}}"""
        )
    kernel = r"""__global__ void uuuconv_reduce_node_source_parts(const {scalar}* partial_x, const {scalar}* partial_rx, {scalar}* out_x, {scalar}* out_rx, int num_nodes) {{
const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
if (index >= Index(num_nodes) * {in1_dim}) return;
const Index node = index / {in1_dim};
const int feature = index % {in1_dim};
{scalar} vx = 0.0, vrx = 0.0;
{group_reductions}
out_x[index] = vx;
out_rx[index] = vrx;
}}
""".format(
        scalar=scalar,
        in1_dim=op.in1_dim,
        group_reductions="\n".join(group_reductions),
    )
    calls = [
        "if (num_nodes > 0) {",
        f"uuuconv_reduce_node_source_parts<<<(Index(num_nodes) * {op.in1_dim} + 255) / 256, 256, 0, stream>>>(out_x, out_rx, tensor_x.data_ptr<{scalar}>(), tensor_rx.data_ptr<{scalar}>(), num_nodes);",
        "}",
    ]
    return kernel, calls


def make_destination_backward_kernel(
    op, parameters, paths, workers, suffix, *, order=2, source_names
):
    """Fuse complete derivatives in destination CSR, with source partials."""
    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    specs = [spec for _, spec in paths]
    channels = max(int(spec["channels"]) for spec in specs)
    tiles = (channels + 31) // 32
    threads = 32 * workers
    x_groups, y_groups = group_paths(specs, "x"), group_paths(specs, "y")
    indices = {
        axis: {key: group for group, (key, _) in enumerate(groups)}
        for axis, groups in (("x", x_groups), ("y", y_groups))
    }
    bases, width = {}, 0
    for path, spec in paths:
        bases[path] = width
        width += int(spec["d3"])
    initialize_sums = "\n".join(
        f"neighbor_sums[{component} * {threads} + threadIdx.x] = 0.0;"
        for component in range(width)
    )
    lines = []
    feature_roles = ("x", "y") + (("rx", "ry") if order == 3 else ())
    groups_by_axis = {"x": x_groups, "y": y_groups}
    axes = {"x": "x", "rx": "x", "y": "y", "ry": "y"}
    # Retain the shorter complete family and finish the wider slices in turn.
    streams_y = False
    if order != 3:
        stream_specs = specs
        stream_x_groups, stream_y_groups = (
            group_paths(stream_specs, "x"),
            group_paths(stream_specs, "y"),
        )
        stream_width = (
            sum(
                component_width
                for groups in (stream_x_groups, stream_y_groups)
                for (_, component_width, _), _ in groups
            )
            + 2 * max(component_width for (_, component_width, _), _ in stream_y_groups)
            + 3 * max(component_width for (_, component_width, _), _ in stream_x_groups)
            + max(int(spec["d3"]) for spec in stream_specs)
        )
        streams_y = stream_width * (8 if op.compute_dtype == torch.float64 else 4) > 640
    axis_widths = {
        axis: sum(width for (_, width, _), _ in groups)
        for axis, groups in groups_by_axis.items()
    }
    stream_axis = (
        max(
            groups_by_axis,
            key=axis_widths.__getitem__,
        )
        if order == 3
        else "y"
        if streams_y
        else None
    )
    streamed_roles = tuple(role for role in feature_roles if axes[role] == stream_axis)
    retained_roles = tuple(role for role in feature_roles if axes[role] != stream_axis)
    for role in retained_roles:
        groups = x_groups if role in ("x", "rx") else y_groups
        for group, ((_, dimension, _), _) in enumerate(groups):
            for component in range(dimension):
                lines.append(f"{scalar} acc_{role}_{group}_{component} = 0.0;")
    chunks = (
        [
            [paths[local] for local, _ in contributors]
            for _, contributors in groups_by_axis[stream_axis]
        ]
        if stream_axis
        else [paths]
    )
    for chunk in chunks:
        if stream_axis:
            lines.append("{")
            for role in streamed_roles:
                for key, _ in group_paths([spec for _, spec in chunk], stream_axis):
                    group = indices[stream_axis][key]
                    for component in range(key[1]):
                        lines.append(f"{scalar} acc_{role}_{group}_{component} = 0.0;")
        for path, spec in chunk:
            x_key = int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"])
            y_key = int(spec["y_offset"]), int(spec["d2"]), int(spec["channels"])
            targets = {
                "gx": [
                    f"""acc_x_{indices["x"][x_key]}_{i}"""
                    for i in range(int(spec["d1"]))
                ],
                "gy": [
                    f"""acc_y_{indices["y"][y_key]}_{j}"""
                    for j in range(int(spec["d2"]))
                ],
                "gw": "acc_w",
            }
            lines += [
                f"""if (u < {int(spec["channels"])}) {{""",
                f"{scalar} acc_w = 0.0;",
            ]
            if order == 3:
                lines.append(f"{scalar} acc_rw = 0.0;")
                targets.update(
                    grx=[
                        f"""acc_rx_{indices["x"][x_key]}_{i}"""
                        for i in range(int(spec["d1"]))
                    ],
                    gry=[
                        f"""acc_ry_{indices["y"][y_key]}_{j}"""
                        for j in range(int(spec["d2"]))
                    ],
                    grw="acc_rw",
                )
            lines.append(
                emit_path_contractions(
                    op, spec, path, order, targets, source_axis_first=(order == 3),
                )
            )
            if spec["has_weight"]:
                for role in ("w", "rw") if order == 3 else ("w",):
                    lines.append(
                        f"""out_{role}[Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u] = acc_{role};"""
                    )
            for k in range(int(spec["d3"])):
                lines.append(f"{scalar} path_value_{k} = 0.0;")
            lines.append(
                emit_path_contractions(
                    op,
                    spec,
                    path,
                    order,
                    {"gg": [f"path_value_{k}" for k in range(int(spec["d3"]))]},
                )
            )
            for k in range(int(spec["d3"])):
                lines.append(
                    f"neighbor_sums[{bases[path] + k} * {threads} + threadIdx.x] += path_value_{k};"
                )
            lines.append("}")
        if stream_axis:
            stride = op.in1_dim if stream_axis == "x" else op.in2_dim
            for role in streamed_roles:
                for key, _ in group_paths([spec for _, spec in chunk], stream_axis):
                    offset, dimension, count = key
                    group = indices[stream_axis][key]
                    for component in range(dimension):
                        lines.append(
                            f"if (u < {count}) out_{role}[Index(edge) * {stride} + {offset} + u * {dimension} + {component}] = acc_{role}_{group}_{component};"
                        )
            lines.append("}")
    for role in retained_roles:
        groups, stride = (
            (x_groups, op.in1_dim) if role in ("x", "rx") else (y_groups, op.in2_dim)
        )
        for group, ((offset, dimension, count), _) in enumerate(groups):
            for component in range(dimension):
                lines.append(
                    f"if (u < {count}) out_{role}[Index(edge) * {stride} + {offset} + u * {dimension} + {component}] = acc_{role}_{group}_{component};"
                )

    edge_contractions = "\n".join(lines)
    lines = []
    for path, spec in paths:
        lines.append(f"""if (u < {int(spec["channels"])}) {{""")
        for k in range(int(spec["d3"])):
            lines += [
                r"""{{
{scalar} value = 0.0;
#pragma unroll
for (int worker = 0; worker < {workers}; ++worker) value += neighbor_sums[{reduction_slot} * {threads} + worker * 32 + lane];
out_g[Index(d) * {out_dim} + {out_offset} + u * {d3} + {k}] = value;
}}""".format(
                    scalar=scalar,
                    workers=workers,
                    reduction_slot=bases[path] + k,
                    threads=threads,
                    out_dim=op.out_dim,
                    out_offset=int(spec["out_offset"]),
                    d3=int(spec["d3"]),
                    k=k,
                )
            ]
        lines.append("}")
    name = f"""lunatp_uuuconv_{("fwd", "bwd", "double_bwd", "triple_bwd")[order]}_destination_derivatives{suffix}"""
    return (
        double_backward_cuda.DESTINATION_DERIVATIVE_KERNEL.format(
            threads=threads,
            name=name,
            parameters=parameters,
            null_declarations="\n".join(
                f"const {scalar}* {name} = nullptr;"
                for name in ("re", "se", "grad_l0", "sl0") if name not in source_names
            ),
            tiles=tiles,
            scalar=scalar,
            shared_width=width * threads,
            workers=workers,
            initialize_sums=initialize_sums,
            edge_contractions=edge_contractions,
            output_reductions="\n".join(lines),
        ),
        name,
        tiles,
        threads,
    )


def make_bounded_source_leaves(
    op,
    paths,
    order,
    targets,
    source_names,
    input_params,
    group_id,
    leaf_paths,
    workers,
    *,
    with_env,
    with_l0,
    need_grad_env,
    edge_centered=False,
    scalar_paths=(),
):
    """Finish a bounded path chunk for its source CSR row or one owned edge."""
    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    scalar_names = ("out_w", "out_rw") if scalar_paths else ()
    source_args = ", ".join((*source_names, *scalar_names))
    if scalar_names:
        input_params += ", " + ", ".join(
            f"{scalar}* __restrict__ {name}" for name in scalar_names
        )
    dim = int(paths[0][1]["d1"])
    definitions, calls = [], []
    # Reuse one edge irrep across its bounded source paths, as in UVUConv.
    by_edge = defaultdict(list)
    for path, spec in paths:
        by_edge[(int(spec["y_offset"]), int(spec["d2"]))].append((path, spec))
    chunks = [
        group[start : start + leaf_paths]
        for group in by_edge.values()
        for start in range(0, len(group), leaf_paths)
    ]
    for chunk_id, chunk in enumerate(chunks):
        helper = f"""uuuconv_x_{("edge_" if edge_centered else "")}group{group_id}_chunk{chunk_id}"""
        result_type = helper + "_result"
        fields = [f"{scalar} {output}[{dim}];" for output in targets]
        helper_targets = {
            output: ([f"result.{output}[{v}]" for v in range(dim)])
            for output in targets
        }
        body, resident = [], {}
        first = chunk[0][1]
        for role, pointer in (("y", "y"), ("ry", "ry"), ("by", "sy")):
            for j in range(int(first["d2"])):
                name = f"shared_{role}{j}"
                resident[f"{role}{j}"] = name
                body.append(
                    f"""const {scalar} {name} = __ldg({pointer} + Index(edge) * {op.in2_dim} + {int(first["y_offset"])} + u * {int(first["d2"])} + {j});"""
                )
        for path, spec in chunk:
            raw = with_l0 and path in op.l0_offsets
            path_targets = dict(helper_targets)
            if path in scalar_paths:
                body += ["{", f"{scalar} scalar_w = 0.0, scalar_rw = 0.0;"]
                path_targets.update(gw="scalar_w", grw="scalar_rw")
            body.append(
                emit_path_contractions(
                    op,
                    spec,
                    path,
                    order,
                    path_targets,
                    with_env=with_env,
                    need_grad_env=need_grad_env,
                    combined_l0=raw and not with_env,
                    resident_values=resident,
                )
            )
            if path in scalar_paths:
                for name in ("w", "rw"):
                    body.append(
                        f"""out_{name}[Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u] = scalar_{name};"""
                    )
                body.append("}")
            if raw and with_env:
                body.append(
                    emit_path_contractions(
                        op,
                        spec,
                        path,
                        order,
                        helper_targets,
                        raw_l0=True,
                        resident_values=resident,
                    )
                )
        nulls = "\n".join(
            f"const {scalar}* {n} = nullptr;"
            for n in ("re", "se", "grad_l0", "sl0")
            if n not in source_names
        )
        if edge_centered:
            definitions.append(
                triple_backward_cuda.BOUNDED_EDGE_SOURCE.format(
                    result_type=result_type,
                    fields=" ".join(fields),
                    helper=helper,
                    input_params=input_params,
                    nulls=nulls,
                    body="\n".join(body),
                )
            )
        else:
            definitions.append(
                triple_backward_cuda.BOUNDED_NODE_SOURCE.format(
                    result_type=result_type,
                    fields=" ".join(fields),
                    helper=helper,
                    input_params=input_params,
                    nulls=nulls,
                    workers=workers,
                    body="\n".join(body),
                )
            )
        calls.append("{")
        if edge_centered:
            calls.append(f"const auto result = {helper}({source_args}, edge, s, d, u);")
        else:
            calls.append(
                f"""const auto result = {helper}({source_args}, src_rowptr, edge_ids, dst, s, u, {("neighbor_worker" if workers > 1 else "0")});"""
            )
        for output, target in targets.items():
            for v in range(dim):
                calls.append(f"{target[v]} += result.{output}[{v}];")
        calls.append("}")
    return definitions, calls


SOURCE_GROUP_REDUCTION_KERNEL = r"""__global__ void uuuconv_reduce_source_groups(const {scalar}* __restrict__ partial, {scalar}* __restrict__ out, int num_nodes) {{
const Index item = Index(blockIdx.x) * blockDim.x + threadIdx.x;
if (item >= Index(num_nodes) * {in1_dim}) return;
const Index node = item / {in1_dim};
const int feature = item % {in1_dim};
{scalar} value = 0.0;
{covered_guard}{group_reductions}
out[item] = value;
}}
"""


CHANNEL_CSR_KERNEL = r"""__global__ __launch_bounds__({threads}, {min_blocks})
void {kernel_name}({parameters}) {{
{nulls}
const Index task = {task};
const int lane = threadIdx.x & 31;
const Index task_count = {rows} * {width};
int group = 0, channel_base = 0;
Index item; int node = 0, tile;
{setup}
switch (group) {{
{cases}
}}
}}
"""

PATH_LEAF = r"""struct {result_type} {{ {fields} }};
__device__ __noinline__ {result_type} {helper}({input_params}, Index edge, int s, int d, int u) {{
{helper_nulls}
{result_type} result{{}};
{helper_body}
return result;
}}
"""
