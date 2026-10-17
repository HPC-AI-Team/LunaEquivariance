"""Gate's shared physical paths, operand dependencies, and native binding."""

import torch

from ....utils.indexing import LAYOUT_HOST, LAYOUT_SOURCE
from ...activation.kernels.scalar import scalar_literal
from ...activation.kernels.utils import CUDA_INPUT_NAMES, INPUT_NAMES, OUTPUT_NAMES


GATE_ARITHMETIC = r'''__device__ __forceinline__ float gate_multiply(float a, float b) {
    return __fmul_rn(a, b);
}
__device__ __forceinline__ double gate_multiply(double a, double b) {
    return __dmul_rn(a, b);
}
__device__ __forceinline__ float gate_add(float a, float b) {
    return __fadd_rn(a, b);
}
__device__ __forceinline__ double gate_add(double a, double b) {
    return __dadd_rn(a, b);
}
template<int Components>
__device__ __forceinline__ Element gate_component_sum(const Element (&values)[Components]) {
    // Match the registered contiguous component reduction, including its
    // three-coordinate xz_y association. Other widths keep component order.
    if constexpr (Components == 3) {
        return gate_add(gate_add(values[0], values[2]), values[1]);
    } else {
        Element result = values[0];
        #pragma unroll
        for (int component = 1; component < Components; ++component) {
            result = gate_add(result, values[component]);
        }
        return result;
    }
}'''


def make_path_types(meta, dtype):
    """Declare only fixed scalar/product addresses and activation parameters."""
    from ...activation.kernels.forward import ACTIVATION_TYPES, FORWARD_FUNCTIONS, FORWARD_MATH

    declarations = [ACTIVATION_TYPES]
    paths = []
    for product, entries in ((False, meta["scalar_paths"]), (True, meta["gated_paths"])):
        for path in entries:
            spec = path["activation"]
            kind = spec["kind"].capitalize()
            if kind == "Gelu" and spec["params"].get("approximate") == "tanh":
                kind = "GeluTanh"
            name = f"Path{len(paths)}"
            fields = dict(
                input_offset=path["input_offset"], output_offset=path["output_offset"],
                components=path["components"] if product else 1,
            )
            if product:
                fields["gate_offset"] = path["gate_offset"]
            constants = "\n".join(f"    static constexpr int {field} = {value};" for field, value in fields.items())
            scale = scalar_literal(spec["cst"], dtype)
            beta = scalar_literal(spec["params"].get("beta", 1.0), dtype)
            threshold = scalar_literal(spec["params"].get("threshold", 20.0), dtype)
            declarations.append(
                f"struct {name} {{\n{constants}\n"
                f"    static constexpr auto kind = ActivationKind::{kind};\n"
                f"    static constexpr bool normalize = {str(not spec['is_id']).lower()};\n"
                f"    __device__ static ActivationParameters parameters() {{\n"
                f"        return {{Element({scale}), Element({beta}), Element({threshold})}};\n"
                f"    }}\n}};"
            )
            paths.append((name, product, path["copies"] if product else path["width"]))
    declarations += ["\n\n".join(FORWARD_FUNCTIONS.values()), FORWARD_MATH, GATE_ARITHMETIC]
    return declarations, paths


def make_schedule(paths, scalar_call, product_call):
    """Use one thread per scalar or vector copy, matching the original launch."""
    statements = []
    begin = 0
    for name, product, copies in paths:
        end = begin + copies
        call = product_call if product else scalar_call
        statements.append(
            f"if (group >= {begin} && group < {end}) {{\n"
            f"    const Index copy = group - {begin};\n"
            f"    {call.format(path=name)}\n}}"
        )
        begin = end
    return begin or 1, "\n".join(statements)


def make_binding(meta, config, *, declarations, kernel_body, binding, module_binding,
                 launch, strided=False, tensor_only=False, fragments=False):
    """Allocate requested outputs and bind one phase's actual CUDA kernel."""
    from ...activation.kernels.forward import CUDA_HEADERS

    names = config["active_outputs"]
    allocations = "\n".join(
        f"auto result_{name}=torch::empty({{num_items,int64_t({meta['out_dim'] if name in ('out', 'sg', 'tg') else meta['in_dim']})}},options);"
        for name in names
    )
    if tensor_only:
        assert len(names) == 1 and config["outputs"] == names
        result_return = f"result_{names[0]}"
    else:
        result_return = "pybind11::make_tuple(" + ",".join(
            f"pybind11::cast(result_{name})" if name in names else "pybind11::none()"
            for name in config["outputs"]
        ) + ")"

    body = binding.format(
        scalar="double" if config["dtype"] == torch.float64 else "float",
        layout_declarations="\n" + LAYOUT_SOURCE if strided else "",
        declarations="\n".join(declarations), kernel_body=kernel_body,
        result_type="torch::Tensor" if tensor_only else "pybind11::tuple",
        tensor_parameters=",".join(f"torch::Tensor {role}" for role in config["inputs"]),
        first=config["inputs"][0], allocations=allocations,
        layout_host=LAYOUT_HOST if strided else "", launch=launch,
        result_return=result_return,
    )
    headers = CUDA_HEADERS.splitlines() + ["#include <cmath>", "#include <type_traits>"]
    if fragments:
        return headers, body
    return "\n".join(headers) + "\n" + body + module_binding


def get_kernel_config(meta, kernel_cache, order, *, needs=None, available=None,
                      grad_mode=False, dtype=None):
    """Cache live operands and connected outputs without tracing or saved graphs."""
    dtype = meta["dtype"] if dtype is None else dtype
    needs = tuple(needs) if needs is not None else None
    available = tuple(available) if available is not None else None
    key = ("gate_config", dtype, order, needs, available, grad_mode)
    config = kernel_cache.get(key)
    if config is not None:
        return config

    if order not in range(4):
        raise ValueError("Activation/Gate derivative order must be 0..3")
    names = OUTPUT_NAMES[order]
    needs = (True,) * len(names) if needs is None else needs
    if len(needs) != len(names):
        raise ValueError("Activation/Gate output request count is incorrect")
    available = INPUT_NAMES[order] if available is None else available
    if set(available).difference(INPUT_NAMES[order]):
        raise ValueError("Unknown Activation/Gate input")
    from . import backward, double_backward, triple_backward

    if order == 0:
        required = {"out": {"x"} if meta["scalar_paths"] or meta["gated_paths"] else set()}
    elif not meta["has_grad"]:
        required = {}
    elif order == 1:
        required = backward.required_inputs(meta)
    elif order == 2:
        required = double_backward.required_inputs(meta, available)
    else:
        required = triple_backward.required_inputs(meta, available)

    dependencies = set()
    active = []
    zeros = []
    for name, wanted in zip(names, needs):
        reads = required.get(name)
        if wanted and reads is not None and reads.issubset(available):
            active.append(name)
            dependencies.update(reads)
            if not reads:
                zeros.append(name)

    config = dict(
        dtype=dtype, order=order, grad_mode=grad_mode, available=available,
        outputs=names, inputs=tuple(role for role in CUDA_INPUT_NAMES if role in dependencies),
        active_outputs=tuple(active), zero_outputs=tuple(zeros),
    )
    kernel_cache[key] = config
    if len(kernel_cache) > 512:
        kernel_cache.popitem(last=False)
    return config
