"""UUU forward CUDA templates."""

from collections.abc import Mapping
from textwrap import indent
import torch
from ....utils.precision import float_literal, cuda_source_for_dtype
from ....utils.extension import load_tp_extension
from .utils import (
    UUU_THREADS,
    CUDA_HEADERS,
    make_channel_schedule,
    connected_path_groups,
    prepare_fused_group,
    build_uuu_groups,
)


def make_source(meta: Mapping[str, object]) -> str:
    """Generate forward kernels and their current-stream tensor binding."""
    outputs = ["out"]
    groups = []
    for paths in connected_path_groups(meta["generated_specs"]):
        fused = prepare_fused_group(0, meta.get("dtype"), paths, outputs)
        if fused is not None:
            groups.append(fused)
            continue
        groups.extend(build_uuu_groups({**meta, "generated_specs": paths}, outputs))
    bodies = []
    for group in groups:
        lines = [f"""if (u >= {group["channels"]}) return;"""]
        if group["layout"] == "fused":
            for offset, dim in group["output_slices"]["out"]:
                for component in range(dim):
                    lines.append(f"float out_{offset}_{component} = 0.0f;")
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
                for k in components[2]:
                    accumulator = f"""out_{spec["out_offset"]}_{k}"""
                    lines.append(
                        f"    {accumulator} = fmaf(q0_{k}, wv, {accumulator});"
                    )
                lines.append("}")
            for offset, dim in group["output_slices"]["out"]:
                for component in range(dim):
                    lines.append(
                        f"out_out[item_g + {offset} + u * {dim} + {component}] = out_{offset}_{component};"
                    )
        else:
            dim = group["dim"]
            for component in range(dim):
                lines.append(f"float out{component} = 0.0f;")
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
                for k in components[2]:
                    lines.append(f"        out{k} = fmaf(q0_{k}, wv, out{k});")
                lines.append("    }")
                lines.append("}")
            for component in range(dim):
                lines.append(
                    f"""out_out[item_g + {group["offset"]} + u * {dim} + {component}] = out{component};"""
                )
        bodies.append("\n".join(lines))
    entry = f"""lunatp_uuu_fwd_{meta["weight_mode"]}_fp32"""
    kernel = "lunatp_uuu_fwd_fp32"
    # Scopes bound path-local register lifetimes even in long grouped path lists.
    cases = []
    for index, body in enumerate(bodies):
        cases.append(
            f"case {index}: {{\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n    break;\n}"
        )
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
    case_source = indent("\n".join(cases), "        ")
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
    const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ weight, float* __restrict__ out_out,
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

torch::Tensor {entry}(torch::Tensor x, torch::Tensor y, torch::Tensor weight) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_items = x.size(0);
    {allocation_source}
    if (num_items == 0) {{
        return out;
    }}
    {grid}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    {kernel}<<<grid, GEN_THREADS, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), num_items);

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
        index=index,
        weight_item=weight_item,
        case_source=case_source,
        entry=entry,
        allocation_source=allocation_source,
        grid=grid,
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
