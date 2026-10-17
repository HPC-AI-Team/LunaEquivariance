"""UUU forward CUDA templates."""

from collections.abc import Mapping
import torch
from ....utils.precision import float_literal, cuda_source_for_dtype
from ....utils.extension import load_tp_extension
from .utils import (
    UUU_THREADS,
    CUDA_HEADERS,
    make_channel_schedule,
    find_root,
    prepare_fused_group,
    build_uuu_groups,
)


def make_source(meta: Mapping[str, object]) -> str:
    """Generate forward kernels and their current-stream tensor binding."""
    outputs = ["out"]
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
        fused = prepare_fused_group(0, meta.get("dtype"), paths, outputs)
        if fused is not None:
            groups.append(fused)
            continue
        groups.extend(build_uuu_groups({**meta, "generated_specs": paths}, outputs))
    bodies = []
    for group in groups:
        if group["layout"] == "fused":
            group_outputs = group["outputs"]
            lines = [f"""if (u >= {group["channels"]}) return;"""]
            for output in group_outputs:
                for offset, dim in group["output_slices"][output]:
                    for component in range(dim):
                        lines.append(f"float {output}_{offset}_{component} = 0.0f;")
            for role in ("x", "y"):
                for value_role, offset, component, dim in sorted(
                    group["cached_values"]
                ):
                    if value_role != role:
                        continue
                    lines.append(
                        f"const float {role}v_{offset}_{component} = __ldg({role} + item_{role} + {offset} + u * {dim} + {component});"
                    )
            for spec in group["specs"]:
                lines.append("{")
                components = [
                    sorted({term[axis] for term in spec["terms"]}) for axis in range(3)
                ]
                weight = (
                    f"""__ldg(weight + item_w + {spec["weight_offset"]} + u)"""
                    if spec["has_weight"]
                    else "1.0f"
                )
                lines.append(f"    const float wv = {weight};")
                for k in components[2]:
                    lines.append(f"    float q0_{k} = 0.0f;")
                for i, j, k, coefficient in spec["terms"]:
                    lines.append(
                        f"""    q0_{k} = fmaf({float_literal(coefficient)}, xv_{spec["x_offset"]}_{i} * yv_{spec["y_offset"]}_{j}, q0_{k});"""
                    )
                for output in group_outputs:
                    for k in components[2]:
                        accumulator = f"""{output}_{spec["out_offset"]}_{k}"""
                        lines.append(
                            f"    {accumulator} = fmaf(q0_{k}, wv, {accumulator});"
                        )
                lines.append("}")
            for output in group_outputs:
                for offset, dim in group["output_slices"][output]:
                    for component in range(dim):
                        lines.append(
                            f"{output}_out[item_g + {offset} + u * {dim} + {component}] = {output}_{offset}_{component};"
                        )
        else:
            names, dim = group["outputs"], group["dim"]
            lines = [f"""if (u >= {group["channels"]}) return;"""]
            for output in names:
                for component in range(dim):
                    lines.append(f"float {output}{component} = 0.0f;")
            for spec in group["specs"]:
                lines.append("{")
                components = [
                    sorted({term[axis] for term in spec["terms"]}) for axis in range(3)
                ]
                for role, offset, width, axis in (
                    ("x", spec["x_offset"], spec["d1"], 0),
                    ("y", spec["y_offset"], spec["d2"], 1),
                ):
                    for component in components[axis]:
                        lines.append(
                            f"    const float {role}v{component} = __ldg({role} + item_{role} + {offset} + u * {width} + {component});"
                        )
                weight = (
                    f"""__ldg(weight + item_w + {spec["weight_offset"]} + u)"""
                    if spec["has_weight"]
                    else "1.0f"
                )
                lines.append(f"    const float wv = {weight};")
                lines.append("    {")
                for k in components[2]:
                    lines.append(f"        float q0_{k} = 0.0f;")
                for i, j, k, coefficient in spec["terms"]:
                    lines.append(
                        f"        q0_{k} = fmaf({float_literal(coefficient)}, xv{i} * yv{j}, q0_{k});"
                    )
                for output in names:
                    for k in components[2]:
                        lines.append(
                            f"        {output}{k} = fmaf(q0_{k}, wv, {output}{k});"
                        )
                lines.append("    }")
                lines.append("}")
            for output in names:
                for component in range(dim):
                    lines.append(
                        f"""{output}_out[item_g + {group["offset"]} + u * {dim} + {component}] = {output}{component};"""
                    )
        bodies.append("\n".join(lines))
    inputs = ("x", "y", "w")
    entry = f"""lunatp_uuu_fwd_{meta["weight_mode"]}_fp32"""
    kernel = "lunatp_uuu_fwd_fp32"
    input_names = [
        "weight" if name == "w" else "grad_out" if name == "g" else name
        for name in inputs
    ]
    declarations = [f"torch::Tensor {name}" for name in input_names]
    pointers = [f"const float* __restrict__ {name}" for name in input_names]
    pointers.extend(f"float* __restrict__ {name}_out" for name in outputs)
    arguments = [f"{name}.data_ptr<float>()" for name in input_names]
    arguments.extend(f"{name}.data_ptr<float>()" for name in outputs)
    # Scopes bound path-local register lifetimes even in long grouped path lists.
    cases = []
    for index, body in enumerate(bodies):
        cases.append(
            f"case {index}: {{\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n    break;\n}"
        )
    # Large programs traverse one role at a time to keep hot instruction
    # regions nearby. Short programs retain interleaved feature-row reuse.
    index, grid, constants = make_channel_schedule(groups, cases)
    slices = {
        (spec["out_offset"], spec["channels"] * spec["d3"])
        for spec in meta["generated_specs"]
    }
    initialize = (
        "zeros" if sum(span for _, span in slices) != meta["out_dim"] else "empty"
    )
    allocation_source = (
        f"auto out = torch::{initialize}({{num_items, GEN_OUT_DIM}}, x.options());"
    )
    weight_item = (
        "0" if meta["weight_mode"] == "shared" else "int64_t(n) * GEN_WEIGHT_NUMEL"
    )
    case_source = "\n".join(cases)
    headers = CUDA_HEADERS
    source = r"""
{headers}

namespace {{

constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_OUT_DIM = {out_dim};
constexpr int GEN_WEIGHT_NUMEL = {weight_numel};
constexpr int GEN_THREADS = {UUU_THREADS};
{constants}

__global__ __launch_bounds__(GEN_THREADS, 4)
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

torch::Tensor {entry}({declarations}) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_items = x.size(0);
    {allocation_source}
    if (num_items == 0) {{

        return out;
    }}
    {grid}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    {kernel}<<<grid, GEN_THREADS, 0, stream>>>(
        {arguments}, num_items);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
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
        kernel=kernel,
        pointers=", ".join(pointers),
        index=index,
        weight_item=weight_item,
        case_source=case_source,
        entry=entry,
        declarations=", ".join(declarations),
        allocation_source=allocation_source,
        grid=grid,
        arguments=", ".join(arguments),
    )
    source += r"""PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    cudaFuncAttributes attributes;
    const auto status = cudaFuncGetAttributes(&attributes, {kernel});
    TORCH_CHECK(status == cudaSuccess, "UUU first kernel initialization: ", cudaGetErrorString(status));
    m.def("{entry}", &{entry}, "UUU fwd {weight_mode} weights (CUDA)");
}}
""".format(
        kernel=kernel, entry=entry, weight_mode=meta["weight_mode"]
    )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, x, edge_attr, weight):
    if not op.meta["generated_specs"]:
        return x.new_zeros((x.shape[0], op.meta["out_dim"]))
    key = (0, True, False)
    launch = op.kernels.get(key)
    if launch is None:
        dtype_name = "fp64" if op.compute_dtype is torch.float64 else "fp32"
        name = f"lunatp_uuu_fwd_{op.weight_mode}_{dtype_name}"
        extension = load_tp_extension(
            name,
            make_source(op.meta),
            max_channels=op.meta["max_channels"],
            index_bits=op.index_bits,
        )
        launch = getattr(extension, name)
        op.kernels[key] = launch
    return launch(x, edge_attr, weight)
