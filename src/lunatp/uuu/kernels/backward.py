"""UUU backward CUDA templates."""

import torch
from ...utils.contractions import INPUTS, PROGRAMS
from ...utils.layout import build_signed_ij_classes
from ....utils.precision import float_literal, cuda_source_for_dtype
from ....utils.extension import load_tp_extension
from .utils import (
    UUU_THREADS,
    CUDA_HEADERS,
    make_channel_schedule,
    find_root,
    prepare_fused_group,
    input_slice,
    output_slice,
    build_uuu_groups,
    contraction_dependencies,
)


def make_source(
    meta,
    order,
    *,
    need_grad_weight=True,
    need_grad_rw=True,
    only_x=False,
    needs=None,
):
    """Generate the requested backward order and its current-stream binding."""
    inputs = ("y", "w", "g") if only_x else INPUTS[: (4, 7, 11)[order - 1]]
    outputs = (
        ["gx"]
        if only_x
        else [
            name
            for name in PROGRAMS[order]["outputs"]
            if (name != "gw" or need_grad_weight) and (name != "grw" or need_grad_rw)
            and (needs is None or needs[PROGRAMS[order]["outputs"].index(name)])
        ]
    )
    extra_source = extra_bindings = ""
    if order == 1 and not only_x:
        grad_x = make_source(meta, 1, only_x=True)
        extra_source = "namespace uuu_grad_x {\n" + grad_x + "\n}\n"
        dtype_name = "fp64" if meta.get("dtype") is torch.float64 else "fp32"
        entry = f"""lunatp_uuu_bwd_x_{meta["weight_mode"]}_{dtype_name}"""
        extra_bindings = (
            f'    m.def("{entry}", &uuu_grad_x::{entry}, "UUU grad_x (CUDA)");\n'
        )
    program = PROGRAMS[order]
    expressions = {
        **program["x"],
        **program["y"],
        **program["w"],
        "gg": program["forward"],
    }
    weight_outputs = [name for name in outputs if name in ("gw", "grw")]
    if only_x:
        groups = build_uuu_groups(meta, outputs)
    else:
        specs = meta["generated_specs"]
        parent = list(range(len(specs)))

        owners = {}
        for index, spec in enumerate(specs):
            for layout, offset, width in (
                ("x", spec["x_offset"], spec["d1"]),
                ("y", spec["y_offset"], spec["d2"]),
                ("output", spec["out_offset"], spec["d3"]),
            ):
                key = (layout, offset, width, spec["channels"])
                if key in owners:
                    parent[find_root(parent, index)] = find_root(parent, owners[key])
                else:
                    owners[key] = index
        components = {}
        for index, spec in enumerate(specs):
            components.setdefault(find_root(parent, index), []).append(spec)

        groups = []
        for paths in components.values():
            fused = prepare_fused_group(order, meta.get("dtype"), paths, outputs)
            if fused is not None:
                groups.append(fused)
                continue
            # A large full component can still reuse feature rows within each
            # derivative family. Weight and output derivatives share Q terms;
            # x and y families share their two incoming-gradient contractions.
            families = [
                [name for name in outputs if name in ("gx", "grx")],
                [name for name in outputs if name in ("gy", "gry")],
                [name for name in outputs if name in ("gw", "grw", "out", "gg")],
            ]
            for family in families:
                if not family:
                    continue
                fused = prepare_fused_group(order, meta.get("dtype"), paths, family)
                if fused is not None:
                    groups.append(fused)
                    continue
                weight_names = [name for name in family if name in ("gw", "grw")]
                output_names = [name for name in family if name in ("out", "gg")]
                parts = (
                    [weight_names, output_names]
                    if weight_names and output_names
                    else [family]
                )
                for names in parts:
                    fused = prepare_fused_group(order, meta.get("dtype"), paths, names)
                    if fused is not None:
                        groups.append(fused)
                    else:
                        groups.extend(
                            build_uuu_groups({**meta, "generated_specs": paths}, names)
                        )
    # Share each path's Q contractions between its parameter and output
    # derivatives without keeping the whole connected feature component live.
    if order >= 2 and weight_outputs:
        weight_offsets = {
            spec["weight_offset"]
            for group in groups
            if group["layout"] == "weight"
            for spec in group["specs"]
        }
        paired_offsets = set()
        for group in groups:
            if group["layout"] == "output" and all(
                spec["has_weight"] and spec["weight_offset"] in weight_offsets
                for spec in group["specs"]
            ):
                group["outputs"] = group["outputs"] + weight_outputs
                group["pair_weight"] = True
                paired_offsets.update(spec["weight_offset"] for spec in group["specs"])
        groups = [
            group
            for group in groups
            if group["layout"] != "weight"
            or group["specs"][0]["weight_offset"] not in paired_offsets
        ]

    group_sources = []
    for group in groups:
        if group["layout"] == "fused":
            group_outputs = group["outputs"]
            cg, z, q_names, z_names, roles = contraction_dependencies(
                order, group_outputs
            )
            lines = [f"""if (u >= {group["channels"]}) return;"""]
            for output in group_outputs:
                if output not in ("gw", "grw"):
                    for offset, dim in group["output_slices"][output]:
                        for component in range(dim):
                            lines.append(f"float {output}_{offset}_{component} = 0.0f;")
            for role in INPUTS:
                if role not in roles or role.lstrip("rb") == "w":
                    continue
                kind = role.lstrip("rb")
                pointer = (
                    "weight" if role == "w" else "grad_out" if role == "g" else role
                )
                for value_role, offset, component, dim in sorted(
                    group["cached_values"]
                ):
                    if value_role != role:
                        continue
                    lines.append(
                        f"const float {role}v_{offset}_{component} = __ldg({pointer} + item_{kind} + {offset} + u * {dim} + {component});"
                    )
            for spec in group["specs"]:
                lines.append("{")

                components = [
                    sorted({term[axis] for term in spec["terms"]}) for axis in range(3)
                ]
                for role in INPUTS:
                    if role in roles and role.lstrip("rb") == "w":
                        pointer = (
                            "weight"
                            if role == "w"
                            else "grad_out"
                            if role == "g"
                            else role
                        )
                        operand = (
                            f"""__ldg({pointer} + item_w + {spec["weight_offset"]} + u)"""
                            if spec["has_weight"]
                            else "1.0f"
                            if role == "w"
                            else "0.0f"
                        )
                        lines.append(f"    const float {role}v = {operand};")
                for output in group_outputs:
                    if output in ("gw", "grw"):
                        lines.append(f"    float {output}0 = 0.0f;")
                for name in sorted(z_names):
                    for k in components[2]:
                        products = [
                            f"""{weight}v * {gradient}v_{spec["out_offset"]}_{k}"""
                            for weight, gradient in z[name]
                        ]
                        lines.append(
                            f"""    const float {name}_{k} = {" + ".join(products)};"""
                        )
                for q in sorted(q_names):
                    for k in components[2]:
                        lines.append(f"    float {q}_{k} = 0.0f;")
                    for i, j, k, coefficient in spec["terms"]:
                        products = [
                            f"""{x}v_{spec["x_offset"]}_{i} * {y}v_{spec["y_offset"]}_{j}"""
                            for x, y in cg[q]
                        ]
                        lines.append(
                            f"""    {q}_{k} = fmaf({float_literal(coefficient)}, {" + ".join(products)}, {q}_{k});"""
                        )
                for i, j, k, coefficient in spec["terms"]:
                    for output in group_outputs:
                        if output not in ("gx", "grx", "gy", "gry"):
                            continue
                        input_gradient = output in ("gx", "grx")
                        component = i if input_gradient else j
                        other = j if input_gradient else i
                        offset, _ = output_slice(spec, output)
                        acc = f"{output}_{offset}_{component}"
                        for feature, name in expressions[output]:
                            feature_offset = (
                                spec["y_offset"] if input_gradient else spec["x_offset"]
                            )
                            lines.append(
                                f"    {acc} = fmaf({float_literal(coefficient)} * {feature}v_{feature_offset}_{other}, {name}_{k}, {acc});"
                            )
                for output in group_outputs:
                    weight_output = output in ("gw", "grw")
                    if weight_output or output == "gg":
                        for q, role in expressions[output]:
                            for k in components[2]:
                                if weight_output:
                                    acc = f"{output}0"
                                else:
                                    offset, _ = output_slice(spec, output)
                                    acc = f"{output}_{offset}_{k}"
                                operand = (
                                    f"""{role}v_{spec["out_offset"]}_{k}"""
                                    if weight_output
                                    else f"{role}v"
                                )
                                lines.append(
                                    f"    {acc} = fmaf({q}_{k}, {operand}, {acc});"
                                )
                    if weight_output and spec["has_weight"]:
                        lines.append(
                            f"""    {output}_out[item_partial + {spec["weight_offset"]} + u] = {output}0;"""
                        )
                lines.append("}")
            for output in group_outputs:
                if output in ("gw", "grw"):
                    continue
                row_base = (
                    "item_x"
                    if output in ("gx", "grx")
                    else "item_y"
                    if output in ("gy", "gry")
                    else "item_g"
                )
                for offset, dim in group["output_slices"][output]:
                    for component in range(dim):
                        lines.append(
                            f"{output}_out[{row_base} + {offset} + u * {dim} + {component}] = {output}_{offset}_{component};"
                        )
        else:
            kind = group["layout"]
            names = group["outputs"]
            dim = group["dim"]
            group_expressions = {name: expressions[name] for name in names}
            cg, z, q_names, z_names, used = contraction_dependencies(order, names)
            lines = [f"""if (u >= {group["channels"]}) return;"""]
            weight_names = [name for name in names if name in ("gw", "grw")]
            for output in names:
                if group.get("pair_weight") and output in weight_names:
                    continue
                for component in range(dim):
                    lines.append(f"float {output}{component} = 0.0f;")
            for spec in group["specs"]:
                lines.append("{")
                if group.get("pair_weight"):
                    for output in weight_names:
                        lines.append(f"    float {output}0 = 0.0f;")
                components = [
                    sorted({term[axis] for term in spec["terms"]}) for axis in range(3)
                ]
                for role in INPUTS:
                    if role not in used:
                        continue
                    pointer = (
                        "weight" if role == "w" else "grad_out" if role == "g" else role
                    )
                    base_role = role.lstrip("rb")
                    if base_role == "w":
                        value = (
                            f"""__ldg({pointer} + item_w + {spec["weight_offset"]} + u)"""
                            if spec["has_weight"]
                            else "1.0f"
                            if role == "w"
                            else "0.0f"
                        )
                        lines.append(f"    const float {role}v = {value};")
                    else:
                        offset, width, axis = input_slice(spec, role)
                        for component in components[axis]:
                            lines.append(
                                f"    const float {role}v{component} = __ldg({pointer} + item_{base_role} + {offset} + u * {width} + {component});"
                            )
                feature_weight = {}
                if kind in ("x", "y") and order >= 2:
                    feature_weight = {}
                    # Reuse each feature-weight combination across all sparse CG terms;
                    # the per-term multiply then uses its output-gradient component.
                    for output, products in group_expressions.items():
                        gradients = {}
                        for feature, z_name in products:
                            for weight, gradient in z[z_name]:
                                gradients.setdefault(gradient, []).append(
                                    (feature, weight)
                                )
                        feature_weight[output] = gradients
                    other_axis = 1 if kind == "x" else 0
                    for output, gradients in feature_weight.items():
                        for gradient, products in gradients.items():
                            for component in components[other_axis]:
                                values = [
                                    f"{feature}v{component} * {weight}v"
                                    for feature, weight in products
                                ]
                                lines.append(
                                    f"""    const float fw_{output}_{gradient}_{component} = {" + ".join(values)};"""
                                )
                for z_name in () if feature_weight else sorted(z_names):
                    for k in components[2]:
                        products = [
                            f"{weight}v * {gradient}v{k}"
                            for weight, gradient in z[z_name]
                        ]
                        lines.append(
                            f"""    const float {z_name}_{k} = {" + ".join(products)};"""
                        )
                pending = set(q_names)
                q_order = []
                while pending:
                    # Consume one contraction before constructing the next, preserving
                    # each output's product order while shortening temporary lifetimes.
                    q = next(
                        q
                        for q in sorted(pending)
                        if all(
                            previous not in pending
                            for products in group_expressions.values()
                            for index, (candidate, _) in enumerate(products)
                            if candidate == q
                            for previous, _ in products[:index]
                        )
                    )
                    q_order.append(q)
                    pending.remove(q)
                for q in q_order:
                    lines.append("    {")
                    for k in components[2]:
                        lines.append(f"        float {q}_{k} = 0.0f;")
                    for i, j, k, coefficient in spec["terms"]:
                        pair_products = [f"{x}v{i} * {y}v{j}" for x, y in cg[q]]
                        lines.append(
                            f"""        {q}_{k} = fmaf({float_literal(coefficient)}, {" + ".join(pair_products)}, {q}_{k});"""
                        )
                    for output, products in group_expressions.items():
                        weight_output = output in ("gw", "grw")
                        for candidate, role in products:
                            if candidate != q:
                                continue
                            for k in components[2]:
                                component = 0 if weight_output else k
                                value = (
                                    f"{role}v" if not weight_output else f"{role}v{k}"
                                )
                                lines.append(
                                    f"        {output}{component} = fmaf({q}_{k}, {value}, {output}{component});"
                                )
                    lines.append("    }")
                if feature_weight:
                    # A complete (i,j)->k coefficient vector is shared up to an exact
                    # sign. Project each gradient once, then reuse it for every member
                    # and every output's feature-weight product within this path.
                    gradients = sorted(
                        {
                            gradient
                            for products in feature_weight.values()
                            for gradient in products
                        }
                    )
                    for cls in build_signed_ij_classes(
                        spec["terms"], spec_y_offset=0, y_name_base=0
                    ):
                        lines.append("    {")
                        representative = cls.members[0].terms
                        for gradient in gradients:
                            _, _, k, coefficient = representative[0]
                            lines.append(
                                f"        float projected_{gradient} = {float_literal(coefficient)} * {gradient}v{k};"
                            )
                            for _, _, k, coefficient in representative[1:]:
                                lines.append(
                                    f"        projected_{gradient} = fmaf({float_literal(coefficient)}, {gradient}v{k}, projected_{gradient});"
                                )
                        for member in cls.members:
                            component = member.i_idx if kind == "x" else member.j_name
                            other = member.j_name if kind == "x" else member.i_idx
                            sign = "" if member.relative_sign == 1 else "-"
                            for output, products in feature_weight.items():
                                for gradient in products:
                                    lines.append(
                                        f"        {output}{component} = fmaf({sign}projected_{gradient}, fw_{output}_{gradient}_{other}, {output}{component});"
                                    )
                        lines.append("    }")
                elif kind in ("x", "y"):
                    for i, j, k, coefficient in spec["terms"]:
                        component = i if kind == "x" else j
                        other_component = j if kind == "x" else i
                        for output, products in group_expressions.items():
                            for feature, z_name in products:
                                lines.append(
                                    f"    {output}{component} = fmaf({float_literal(coefficient)} * {feature}v{other_component}, {z_name}_{k}, {output}{component});"
                                )
                if group.get("pair_weight"):
                    for output in weight_names:
                        lines.append(
                            f"""    {output}_out[item_partial + {spec["weight_offset"]} + u] = {output}0;"""
                        )
                lines.append("}")
            item = (
                "item_partial"
                if kind == "weight"
                else "item_g"
                if kind == "output"
                else f"item_{kind}"
            )
            for output in names:
                if group.get("pair_weight") and output in weight_names:
                    continue
                for component in range(dim):
                    lines.append(
                        f"""{output}_out[{item} + {group["offset"]} + u * {dim} + {component}] = {output}{component};"""
                    )
        group_sources.append("\n".join(lines))

    phase_suffix = ("fwd", "bwd", "double_bwd", "triple_bwd")[order]
    weight_outputs = [name for name in outputs if name in ("gw", "grw")]
    entry = f"""lunatp_uuu_{phase_suffix}{("_x" if only_x else "")}_{meta["weight_mode"]}_fp32"""
    kernel = f"""lunatp_uuu_{phase_suffix}{("_x" if only_x else "")}_fp32"""
    input_names = [
        "weight" if name == "w" else "grad_out" if name == "g" else name
        for name in inputs
    ]
    declarations = [f"torch::Tensor {name}" for name in input_names]
    pointers = [f"const float* __restrict__ {name}" for name in input_names]
    pointers.extend(f"float* __restrict__ {name}_out" for name in outputs)
    arguments = [f"{name}.data_ptr<float>()" for name in input_names]
    arguments.extend(
        f"""{name}{("_work" if name in weight_outputs else "")}.data_ptr<float>()"""
        for name in outputs
    )
    # Scopes bound path-local register lifetimes even in long grouped path lists.
    cases = []
    for index, body in enumerate(group_sources):
        cases.append(
            f"case {index}: {{\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n    break;\n}"
        )
    # Large programs traverse one role at a time to keep hot instruction
    # regions nearby. Short programs retain interleaved feature-row reuse.
    index, grid, constants = make_channel_schedule(groups, cases)
    anchor = "grad_out" if only_x else "x"
    allocations = []
    for output in outputs:
        if output in ("gw", "grw"):
            allocations.append(f"auto {output} = torch::empty_like(weight);")
            allocations.append(
                f"auto {output}_work = "
                + (
                    "torch::empty({num_items, GEN_WEIGHT_NUMEL}, weight.options());"
                    if meta["weight_mode"] == "shared"
                    else f"{output};"
                )
            )
        else:
            if output in ("gx", "grx"):
                shape = "GEN_IN1_DIM"
            elif output in ("gy", "gry"):
                shape = "GEN_IN2_DIM"
            else:
                shape = "GEN_OUT_DIM"
            tensor_width = (
                meta["in1_dim"]
                if output in ("gx", "grx")
                else meta["in2_dim"]
                if output in ("gy", "gry")
                else meta["out_dim"]
            )
            slices = set()
            for spec in meta["generated_specs"]:
                offset, width = output_slice(spec, output)
                slices.add((offset, spec["channels"] * width))
            initialize = (
                "zeros" if sum(span for _, span in slices) != tensor_width else "empty"
            )
            allocations.append(
                f"auto {output} = torch::{initialize}({{num_items, {shape}}}, {anchor}.options());"
            )
    if only_x:
        returned = outputs[0]
        return_type = "torch::Tensor"
    else:
        returned = "{" + ", ".join(outputs) + "}"
        return_type = "std::vector<torch::Tensor>"

    reduction = ""
    if weight_outputs and meta["weight_mode"] == "shared" and meta["weight_numel"]:
        names = weight_outputs + [None] * (2 - len(weight_outputs))
        args = [
            f"{name}_work.data_ptr<float>()" if name else "nullptr" for name in names
        ]
        args += [f"{name}.data_ptr<float>()" if name else "nullptr" for name in names]
        # Short batches keep one launch. Partition a long batch only when its
        # column grid cannot fill four blocks per SM; the bounded FP64 partials
        # retain accurate, deterministic shared-weight sums.
        partial_args = args[:2] + [
            f"{name}_parts.data_ptr<double>()" if name else "nullptr" for name in names
        ]
        merge_args = partial_args[2:] + args[2:]
        partial_allocations = [
            f"auto {name}_parts = torch::empty({{parts, GEN_WEIGHT_NUMEL}}, weight.options().dtype(torch::kFloat64));"
            for name in names
            if name
        ]
        partial_allocation_source = "\n".join(partial_allocations)
        reduction = r"""
    constexpr int column_blocks = (GEN_WEIGHT_NUMEL + 31) / 32;
    int parts = 1;
    if (num_items > 512) {{
        const int target = 4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        parts = int(std::min(int64_t(32), (num_items + 127) / 128));
        parts = std::min(parts, (target + column_blocks - 1) / column_blocks);
    }}
    if (parts > 1) {{
        {partial_allocation_source}
        lunatp_uuu_reduce_weight_fp32<{weight_outputs_count}, 4, double><<<dim3(column_blocks, parts), 128, 0, stream>>>(
                {partial_args}, num_items);
        lunatp_uuu_merge_weight_fp32<{weight_outputs_count}><<<(GEN_WEIGHT_NUMEL + 127) / 128, 128, 0, stream>>>(
                {merge_args}, parts);
    }} else {{
        lunatp_uuu_reduce_weight_fp32<{weight_outputs_count}, 8, float><<<column_blocks, 256, 0, stream>>>({args}, num_items);
    }}""".format(
            partial_allocation_source=partial_allocation_source,
            weight_outputs_count=len(weight_outputs),
            partial_args=", ".join(partial_args),
            merge_args=", ".join(merge_args),
            args=", ".join(args),
        )
    empty = "\n".join(f"        {name}.zero_();" for name in weight_outputs)
    weight_item = (
        "0" if meta["weight_mode"] == "shared" else "int64_t(n) * GEN_WEIGHT_NUMEL"
    )

    # FP64's high-order derivatives benefit from a lower residency target;
    # forward and first-order derivatives retain four blocks for occupancy.
    case_source = "\n".join(cases)
    allocation_source = "\n".join(allocations)
    headers = CUDA_HEADERS if not only_x else "\n" * 6
    source = r"""
{headers}

namespace {{

constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_OUT_DIM = {out_dim};
constexpr int GEN_WEIGHT_NUMEL = {weight_numel};
constexpr int GEN_THREADS = {UUU_THREADS};
{constants}
{reduction_source}
__global__ __launch_bounds__(GEN_THREADS, {resident_blocks})
void {kernel}(
    {pointers},
    int64_t num_items) {{
    {index}
    const int64_t item_x = int64_t(n) * GEN_IN1_DIM;
    const int64_t item_y = int64_t(n) * GEN_IN2_DIM;
    const int64_t item_g = int64_t(n) * GEN_OUT_DIM;
    const int64_t item_w = {weight_item};
    const int64_t item_partial = int64_t(n) * GEN_WEIGHT_NUMEL;
    switch (group) {{
{case_source}
    default: break;
    }}
}}

{return_type} {entry}({declarations}) {{
    c10::cuda::CUDAGuard guard({anchor}.device());
    const int64_t num_items = {anchor}.size(0);
    {allocation_source}
    if (num_items == 0) {{
{empty}
        return {returned};
    }}
    {grid}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream({anchor}.get_device());
    {kernel}<<<grid, GEN_THREADS, 0, stream>>>(
        {arguments}, num_items);
    {reduction}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {returned};
}}

}}  // namespace

""".format(
        headers=headers,
        in1_dim=meta["in1_dim"],
        in2_dim=meta["in2_dim"],
        out_dim=meta["out_dim"],
        weight_numel=meta["weight_numel"],
        UUU_THREADS=UUU_THREADS,
        constants=constants,
        reduction_source=r"""
// Row-major partials retain one contribution per item and parameter. Reduce
// columns in a fixed tree; shared gradients never depend on atomic ordering.
template<int N_GRADS, int WARPS = 8, typename output_t = float>
__global__ __launch_bounds__(32 * WARPS)
void lunatp_uuu_reduce_weight_fp32(
    const float* __restrict__ partial0,
    const float* __restrict__ partial1,
    output_t* __restrict__ output0,
    output_t* __restrict__ output1,
    int64_t rows) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column = blockIdx.x * 32 + lane;
    int64_t begin = 0, end = rows;
    if constexpr (WARPS == 4) {
        const int64_t length = (rows + gridDim.y - 1) / gridDim.y;
        begin = int64_t(blockIdx.y) * length;
        end = min(rows, begin + length);
    }
    double acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    double acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (column < GEN_WEIGHT_NUMEL) {
        for (int64_t row = begin + warp; row < end; row += 4 * WARPS) {
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t r = row + part * WARPS;
                if (r < end) {
                    const int64_t offset = r * GEN_WEIGHT_NUMEL + column;
                    acc0[part] += __ldg(partial0 + offset);
                    if constexpr (N_GRADS == 2) acc1[part] += __ldg(partial1 + offset);
                }
            }
        }
    }
    __shared__ double sums[N_GRADS][32 * WARPS];
    sums[0][threadIdx.x] = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
    if constexpr (N_GRADS == 2)
        sums[1][threadIdx.x] = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
    __syncthreads();
    if (warp == 0 && column < GEN_WEIGHT_NUMEL) {
        double value0 = 0.0f;
        double value1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            value0 += sums[0][w * 32 + lane];
            if constexpr (N_GRADS == 2) value1 += sums[1][w * 32 + lane];
        }
        const int64_t offset = int64_t(blockIdx.y) * GEN_WEIGHT_NUMEL + column;
        output0[offset] = value0;
        if constexpr (N_GRADS == 2) output1[offset] = value1;
    }
}

template<int N_GRADS>
__global__ void lunatp_uuu_merge_weight_fp32(
    const double* __restrict__ partial0,
    const double* __restrict__ partial1,
    float* __restrict__ output0,
    float* __restrict__ output1,
    int parts) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= GEN_WEIGHT_NUMEL) return;
    double value0 = 0.0, value1 = 0.0;
    for (int part = 0; part < parts; ++part) {
        const int64_t offset = int64_t(part) * GEN_WEIGHT_NUMEL + column;
        value0 += partial0[offset];
        if constexpr (N_GRADS == 2) value1 += partial1[offset];
    }
    output0[column] = value0;
    if constexpr (N_GRADS == 2) output1[column] = value1;
}
""" if reduction else "",
        resident_blocks=2 if meta.get("dtype") is torch.float64 and order >= 2 else 4,
        kernel=kernel,
        pointers=", ".join(pointers),
        index=index,
        weight_item=weight_item,
        case_source=case_source,
        return_type=return_type,
        entry=entry,
        declarations=", ".join(declarations),
        anchor=anchor,
        allocation_source=allocation_source,
        empty=empty,
        returned=returned,
        grid=grid,
        arguments=", ".join(arguments),
        reduction=reduction,
    )
    if not only_x:
        source += extra_source + r"""PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    cudaFuncAttributes attributes;
    const auto status = cudaFuncGetAttributes(&attributes, {kernel});
    TORCH_CHECK(status == cudaSuccess, "UUU first kernel initialization: ", cudaGetErrorString(status));
    m.def("{entry}", &{entry}, "UUU {phase_suffix} {weight_mode} weights (CUDA)");
{extra_bindings}}}
""".format(
            kernel=kernel,
            entry=entry,
            phase_suffix=phase_suffix,
            weight_mode=meta["weight_mode"],
            extra_bindings=extra_bindings,
        )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, order, *inputs, only_x=False, needs=None):
    if not op.meta["generated_specs"]:
        if only_x:
            return inputs[0].new_zeros((inputs[0].shape[0], op.meta["in1_dim"]))
        return tuple(
            torch.zeros_like(value) if needs is None or needs[index] else None
            for index, value in enumerate(inputs[: (3, 4, 7)[order - 1]])
        )

    native_needs = tuple(needs) if needs is not None else (True,) * (3, 4, 7)[order - 1]
    if not any(native_needs) and not only_x:
        return (None,) * len(native_needs)
    need_grad_weight = native_needs[2]
    need_grad_rw = order == 3 and native_needs[6]
    key = order, native_needs
    launch = op.kernels.get(key)
    if launch is None:
        dtype_name = "fp64" if op.compute_dtype is torch.float64 else "fp32"
        phase = ("bwd", "double_bwd", "triple_bwd")[order - 1]
        extension = load_tp_extension(
            f"lunatp_uuu_{phase}_{op.weight_mode}_{dtype_name}",
            make_source(
                op.meta,
                order,
                need_grad_weight=need_grad_weight,
                need_grad_rw=need_grad_rw,
                needs=native_needs,
            ),
            max_channels=op.meta["max_channels"],
            index_bits=op.index_bits,
        )
        if order == 1:
            launch = (
                getattr(extension, f"lunatp_uuu_bwd_{op.weight_mode}_{dtype_name}"),
                getattr(extension, f"lunatp_uuu_bwd_x_{op.weight_mode}_{dtype_name}"),
            )
        else:
            launch = getattr(
                extension, f"lunatp_uuu_{phase}_{op.weight_mode}_{dtype_name}"
            )
        op.kernels[key] = launch
    if order == 1:
        launch = launch[only_x]

    result = launch(*inputs)
    if only_x:
        return result
    values = iter(result)
    return tuple(next(values) if needed else None for needed in native_needs)
