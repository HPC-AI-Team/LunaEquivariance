"""Gate's shared physical paths, operand dependencies, and native binding."""

from math import prod
from textwrap import indent

import torch

from ....utils.extension import get_cuda_arch, load_cuda_extension
from ....utils.indexing import LAYOUT_HOST, LAYOUT_SOURCE
from ...activation.kernels.scalar import scalar_literal
from ...activation.kernels.utils import CUDA_HEADERS, CUDA_INPUT_NAMES, INPUT_NAMES, OUTPUT_NAMES


WARP_SIZE = 32
SHARED_MEMORY_LIMIT = 48 * 1024
MIN_COPY_TILE = 8
DIRECT_STORE_BYTES = 16
FORWARD_THREADS = 128
FORWARD_MIN_BLOCKS = 16
BACKWARD_THREADS = 128
DOUBLE_BACKWARD_THREADS = 128
TRIPLE_BACKWARD_THREADS = 128


GATE_STORES = r'''template<auto Width, auto Offset, int Components, int Copies, int CopyTile>
__device__ __forceinline__ void gate_store_components(
    const Element* values, Element* out, Index row, Index copy_base, Index num_items) {
    if (row >= num_items) return;
    const int lane = threadIdx.x % CopyTile;
    values += ((threadIdx.x % 32) / CopyTile) * CopyTile * Components;
    constexpr int VectorWidth = sizeof(Element) == 8 ? 2 : 4;
    if constexpr (Copies % VectorWidth == 0 && Width % VectorWidth == 0 &&
                  Offset % VectorWidth == 0) {
        using Vector = std::conditional_t<sizeof(Element) == 8, double2, float4>;
        #pragma unroll
        for (int column = lane; column < CopyTile * Components / VectorWidth; column += CopyTile) {
            if (column < (Copies - copy_base) * Components / VectorWidth) {
                reinterpret_cast<Vector*>(out + row * Width + Offset + copy_base * Components)[column] =
                    reinterpret_cast<const Vector*>(values)[column];
            }
        }
    } else {
        #pragma unroll
        for (int chunk = 0; chunk < Components; ++chunk) {
            const int column = chunk * CopyTile + lane;
            if (column < (Copies - copy_base) * Components) {
                out[row * Width + Offset + copy_base * Components + column] = values[column];
            }
        }
    }
}'''


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
    """Declare scalar/product offsets and activation parameters."""
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


def make_schedule(paths, scalar_call, product_call, *, warp_groups=False, copy_tile=WARP_SIZE):
    """Assign path ranges to copies or equal-sized warp subgroups."""
    statements = []
    begin = 0
    for name, product, copies in paths:
        end = begin + ((copies + copy_tile - 1) // copy_tile if warp_groups else copies)
        call = product_call if product else scalar_call
        indices = (f"    const Index copy_base = (group - {begin}) * {copy_tile};\n"
                   f"    const Index copy = copy_base + lane % {copy_tile};\n" if warp_groups else
                   f"    const Index copy = group - {begin};\n")
        statements.append(
            f"if (group >= {begin} && group < {end}) {{\n"
            f"{indices}"
            f"{indent(call.format(path=name, copies=copies), '    ')}\n}}"
        )
        begin = end
    return begin or 1, indent("\n".join(statements), "        ")


def make_binding(meta, config, *, declarations, kernel_body, binding, module_binding,
                 launch, strided=False, tensor_only=False, fragments=False):
    """Emit the CUDA binding and requested output allocations."""
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
        first=config["inputs"][0], allocations=indent(allocations, "    "),
        layout_host=LAYOUT_HOST if strided else "", launch=launch,
        result_return=result_return,
    )
    headers = CUDA_HEADERS.splitlines() + ["#include <cmath>", "#include <type_traits>"]
    if fragments:
        return headers, body
    return "\n".join(headers) + "\n" + body + module_binding


def get_kernel_config(meta, kernel_cache, order, *, needs=None, available=None,
                      grad_mode=False, dtype=None):
    """Cache input dependencies and requested output names."""
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


def launch_kernel(inputs, config, meta, index_bits, kernel_cache, source):
    """Load the kernel and return the requested output tensors."""
    order = config["order"]
    dtype = config["dtype"]
    input_names = INPUT_NAMES[order]
    available = config["available"]
    first = next((value for value in inputs if value is not None), None)
    if first is None:
        return tuple(None for _ in config["outputs"])
    shape, device = first.shape, first.device
    in_dim, out_dim = meta["in_dim"], meta["out_dim"]
    batch = shape[:-1]
    for role, value in zip(input_names, inputs):
        if value is None:
            continue
        width = out_dim if role in ("grad_out", "bg") else in_dim
        if value.dtype != dtype:
            raise ValueError(
                "Activation/Gate operands must match the computation dtype"
            )
        if value.shape != (*batch, width):
            raise ValueError(f"Activation/Gate operand {role} has an incorrect shape")

    n = prod(batch)
    bound = n * max(in_dim, out_dim, 1) + 65535 * 128
    strided_forward = order == 0 and n > 0 and first.ndim <= 64 and not first.is_contiguous()
    if strided_forward:
        physical = sum((size - 1) * stride for size, stride in zip(first.shape, first.stride()))
        bound = max(bound, physical)
    if bound > (1 << 63) - 1:
        raise OverflowError("Activation/Gate offsets exceed signed int64 capacity")
    index_bits = 64 if bound > (1 << 31) - 1 else index_bits
    key = ("kernel", order, config["active_outputs"], config["inputs"], available,
           config["grad_mode"], index_bits, device, dtype)
    if strided_forward:
        key += ("strided_forward",)
    entry = kernel_cache.get(key)
    if entry is None:
        launch = None
        if config["active_outputs"] and config["inputs"]:
            extension = kernel_cache.get(("forward_backward_module", dtype, device, index_bits))
            if extension is not None and order == 0:
                launch = (extension.strided_forward if strided_forward else extension.forward).launch
            elif extension is not None and order == 1:
                if config["grad_mode"]:
                    launch = extension.differentiable_backward.launch
                elif hasattr(extension, "training"):
                    launch = extension.training.launch_backward_tuple
                else:
                    launch = extension.backward.launch
            else:
                cuda_source = (
                    source({**meta, "dtype": dtype}, config, strided=True)
                    if strided_forward else source({**meta, "dtype": dtype}, config)
                )
            if launch is None:
                precision = "fp64" if dtype == torch.float64 else "fp32"
                name = (
                    f"lunaeq_gate_strided_fwd" if strided_forward
                    else f"lunaeq_gate_{order}_{precision}"
                )
                launch = load_cuda_extension(
                    name, cuda_source, arch=get_cuda_arch(device), index_bits=index_bits,
                    use_fast_math=False, minimal_headers=True,
                ).launch
        entry = config, launch
        kernel_cache[key] = entry
        if len(kernel_cache) > 512:
            kernel_cache.popitem(last=False)

    config, launch = entry
    if not config["active_outputs"]:
        result = tuple(None for _ in config["outputs"])
    elif n == 0 or launch is None:
        result = tuple(
            (
                torch.zeros(
                    (
                        *batch,
                        out_dim if name in ("out", "sg", "tg") else in_dim,
                    ),
                    device=device,
                    dtype=dtype,
                )
                if name in config["active_outputs"]
                else None
            )
            for name in config["outputs"]
        )
    else:
        operands = [
            value if strided_forward else value.reshape(n, value.shape[-1]).contiguous()
            for role, value in zip(input_names, inputs) if role in config["inputs"]
        ]
        result = tuple(
            value.reshape(*batch, value.shape[-1]) if value is not None else None
            for value in launch(*operands, n)
        )
    return result
