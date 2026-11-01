"""Activation specifications, CUDA source construction and launch helpers."""

from functools import partial
from math import prod
from textwrap import indent

import torch
import torch.nn.functional as F

from ....utils.extension import get_cuda_arch, load_cuda_extension
from ....utils.indexing import LAYOUT_SOURCE, LAYOUT_HOST


CUDA_INPUT_NAMES = ("x", "grad_out", "rx", "bx", "bg")
INPUT_NAMES = (("x",), ("x", "grad_out"), ("x", "grad_out", "rx"), CUDA_INPUT_NAMES)
OUTPUT_NAMES = (("out",), ("dx",), ("sx", "sg"), ("tx", "tg", "trx"))
VECTOR_MIN_ELEMENTS = 1 << 18


CUDA_HEADERS = r'''#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>'''

BINDING = r'''using Element={scalar};{layout_declarations}
{declarations}
{kernel_body}
{result_type} launch({tensor_parameters},int64_t num_items) {{
    c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();
{allocations}
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();{layout_host}
    {launch}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {result_return};
}}
'''

MODULE_BINDING = r'''PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("launch",&launch);
}
'''

TENSOR_BINDING = r'''
torch::Tensor launch_tensor({args}) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_items=x.numel()/{width};
    constexpr int64_t padding=65535*128;
    TORCH_CHECK(num_items<=(std::numeric_limits<int64_t>::max()-padding)/{max_width},"Pointwise task offsets exceed int64");
{checks}
{call}
    auto result={result_value};
    std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end());
    shape.back()={result_width};
    // The fresh output owns its storage. Remove the host reshape's view
    // metadata before Python autograd attaches the custom backward.
    return result.reshape(shape).detach();
}}
'''

TRAINING_MODULE_BINDING = r'''
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("launch_forward",&forward_impl::launch_tensor);
    m.def("launch_backward",&backward_impl::launch_tensor);
    m.def("launch_backward_tuple", [](torch::Tensor x, torch::Tensor grad_out, int64_t num_items) {
        return pybind11::make_tuple(backward_impl::contiguous::launch(x, grad_out, num_items));
    });
}
'''


_FUNCTIONS = {
    torch.tanh: ("tanh", ()),
    torch.sigmoid: ("sigmoid", ()),
    torch.relu: ("relu", ()),
    torch.abs: ("abs", ()),
    torch.absolute: ("abs", ()),
    F.tanh: ("tanh", ()),
    F.sigmoid: ("sigmoid", ()),
    F.silu: ("silu", ()),
    F.relu: ("relu", ()),
    F.softplus: ("softplus", (("beta", 1.0), ("threshold", 20.0))),
    F.gelu: ("gelu", (("approximate", "none"),)),
}

_MODULES = {
    torch.nn.Identity: "identity",
    torch.nn.Tanh: "tanh",
    torch.nn.Sigmoid: "sigmoid",
    torch.nn.ReLU: "relu",
    torch.nn.SiLU: "silu",
    torch.nn.Softplus: "softplus",
    torch.nn.GELU: "gelu",
}

_MODULE_FORWARDS = {cls: cls.forward for cls in _MODULES}


def parse_activation(function):
    if function is None:
        return {"kind": "identity", "params": {}}

    keywords = {}
    if isinstance(function, partial):
        if function.args:
            raise ValueError("Activation partials must bind parameters by keyword, leaving the input unbound")
        keywords = dict(function.keywords or {})
        function = function.func

    if isinstance(function, torch.nn.Module):
        kind = _MODULES.get(type(function))
        if (
            kind is None
            or keywords
            or getattr(function.forward, '__func__', function.forward)
            is not _MODULE_FORWARDS.get(type(function))
        ):
            raise ValueError("Unsupported activation module; use a built-in Identity, Tanh, Sigmoid, ReLU, SiLU, Softplus or GELU")
        if getattr(function, "inplace", False):
            raise ValueError("In-place activations are not supported")
        if kind == "softplus":
            return {
                "kind": kind,
                "params": {"beta": float(function.beta), "threshold": float(function.threshold)},
            }
        if kind == "gelu":
            if function.approximate not in ("none", "tanh"):
                raise ValueError("GELU approximate must be 'none' or 'tanh'")
            return {"kind": kind, "params": {"approximate": function.approximate}}
        return {"kind": kind, "params": {}}

    try:
        config = _FUNCTIONS.get(function)
        kind = None if config is None else config[0]
    except TypeError:
        kind = None
    if kind is None:
        raise ValueError("Unsupported activation; use None, torch.abs, tanh, sigmoid, relu, silu, softplus or gelu")

    names = ("inplace",) if function in (F.silu, F.relu) else {
        "softplus": ("beta", "threshold"),
        "gelu": ("approximate",),
    }.get(kind, ())
    unknown = set(keywords).difference(names)
    if unknown:
        raise TypeError(f"Unsupported activation arguments: {sorted(unknown)}")
    if keywords.pop("inplace", False):
        raise ValueError("In-place activations are not supported")

    if kind == "softplus":
        keywords = {
            "beta": float(keywords.get("beta", 1.0)),
            "threshold": float(keywords.get("threshold", 20.0)),
        }
    if kind == "gelu":
        keywords = {"approximate": keywords.get("approximate", "none")}
        if keywords["approximate"] not in ("none", "tanh"):
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
    return {"kind": kind, "params": keywords}


def activation_config(function):
    """Return the built-in activation settings."""
    if function is None:
        return None
    try:
        config = _FUNCTIONS.get(function)
    except TypeError:
        config = None
    if config is not None:
        return config
    spec = parse_activation(function)
    return spec["kind"], tuple(spec["params"].items())


def apply_activation_spec(x, spec):
    """Evaluate an activation specification with Torch."""
    kind, params = spec["kind"], spec["params"]
    if kind == "identity":
        result = x
    elif kind == "tanh":
        result = torch.tanh(x)
    elif kind == "sigmoid":
        result = torch.sigmoid(x)
    elif kind == "relu":
        result = torch.relu(x)
    elif kind == "abs":
        result = torch.abs(x)
    elif kind == "silu":
        result = F.silu(x)
    elif kind == "softplus":
        result = F.softplus(x, **params)
    elif kind == "gelu":
        result = F.gelu(x, **params)
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return result if spec["is_id"] else result.mul(spec["cst"])


def parse_activation_paths(activation, parsed=None):
    """Read each e3nn channel block and its activation normalization."""
    paths = []
    for index, (mul, (degree, _), function) in enumerate(activation.paths):
        if parsed is None:
            spec = parse_activation(function.f if function is not None else None)
        else:
            spec = parsed[index]
        spec = {
            **spec,
            "cst": float(function.cst) if function is not None else 1.0,
            "is_id": bool(function._is_id) if function is not None else True,
        }
        paths.append((mul * (2 * degree + 1), spec))
    return tuple(paths)


def make_path_types(activation_paths, in_dim, dtype):
    """Declare the static CUDA path types shared by forward and backward kernels."""
    # Source construction runs after the phase modules have loaded.
    from .forward import ACTIVATION_TYPES, FORWARD_FUNCTIONS, FORWARD_MATH
    from .scalar import scalar_literal

    paths = []
    path_declarations = []
    end = 0
    vectorized = dtype == torch.float32 and in_dim % 4 == 0
    previous = None
    for width, spec in activation_paths:
        end += width
        if not width:
            continue
        kind = spec["kind"].capitalize()
        if kind == "Gelu" and spec["params"].get("approximate") == "tanh":
            kind = "GeluTanh"
        normalize = not spec["is_id"]
        scale = scalar_literal(spec["cst"], dtype)
        beta = scalar_literal(spec["params"].get("beta", 1.0), dtype)
        threshold = scalar_literal(spec["params"].get("threshold", 20.0), dtype)
        signature = kind, normalize, scale if normalize else None, beta, threshold
        if previous is not None and previous["signature"] == signature:
            previous["end"] = end
        else:
            previous = dict(
                signature=signature, end=end, kind=kind, normalize=normalize,
                scale=scale, beta=beta, threshold=threshold,
            )
            paths.append(previous)

    for index, path in enumerate(paths):
        vectorized &= path["end"] % 4 == 0
        path_declarations.append(
            f"struct Path{index} {{\n"
            f"    static constexpr auto end = {path['end']};\n"
            f"    static constexpr auto kind = ActivationKind::{path['kind']};\n"
            f"    static constexpr bool normalize = {str(path['normalize']).lower()};\n"
            f"    __device__ static ActivationParameters parameters() {{\n"
            f"        return {{Element({path['scale']}), Element({path['beta']}), Element({path['threshold']})}};\n"
            f"    }}\n}};"
        )

    declarations = [
        f"constexpr int VectorMinElements = {VECTOR_MIN_ELEMENTS};",
        ACTIVATION_TYPES,
        "\n".join(path_declarations), "\n\n".join(FORWARD_FUNCTIONS.values()), FORWARD_MATH,
    ]
    return declarations, [f"Path{index}" for index in range(len(paths))], vectorized


def make_binding(in_dim, config, *, declarations, kernel_body, binding, module_binding,
                launch, strided=False, tensor_only=False, fragments=False):
    """Bind one order's CUDA launch to its live tensors and requested outputs."""
    width = in_dim
    names = config["active_outputs"]
    allocations = "\n".join(
        f"auto result_{name}=torch::empty({{num_items,int64_t({width})}},options);"
        for name in names
    )
    result_return = f"result_{names[0]}" if tensor_only else "pybind11::make_tuple(" + ",".join(
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
        layout_host=LAYOUT_HOST if strided else "", launch=launch, result_return=result_return,
    )
    headers = CUDA_HEADERS.splitlines() + ["#include <cmath>", "#include <type_traits>"]
    if fragments:
        return headers, body
    return "\n".join(headers) + "\n" + body + module_binding


def get_kernel_config(activation_paths, order, dtype, kernel_cache, *, needs=None, available=None, grad_mode=False):
    needs = tuple(needs) if needs is not None else None
    available = tuple(available) if available is not None else None
    key = ("config", order, needs, available, grad_mode, dtype)
    config = kernel_cache.get(key)
    if config is not None:
        return config
    from . import backward, double_backward, triple_backward

    if order not in range(4):
        raise ValueError("Activation derivative order must be 0..3")
    names = OUTPUT_NAMES[order]
    needs = (True,) * len(names) if needs is None else needs
    if len(needs) != len(names):
        raise ValueError("Activation output request count is incorrect")
    available = INPUT_NAMES[order] if available is None else available
    if set(available).difference(INPUT_NAMES[order]):
        raise ValueError("Unknown Activation input")
    if order == 0:
        required = {"out": {"x"} if any(width for width, _ in activation_paths) else set()}
    elif not activation_paths:
        required = {}
    elif order == 1:
        required = backward.required_inputs(activation_paths)
    elif order == 2:
        required = double_backward.required_inputs(activation_paths, available)
    else:
        required = triple_backward.required_inputs(activation_paths, available)

    dependencies = set()
    active = []
    zeros = []
    for name, wanted in zip(names, needs):
        reads = required.get(name)
        # None has no autograd edge; an empty set is a connected zero gradient.
        if wanted and reads is not None and reads.issubset(available):
            active.append(name)
            dependencies.update(reads)
            if not reads:
                zeros.append(name)

    config = dict(
        dtype=dtype,
        order=order,
        grad_mode=grad_mode,
        available=available,
        outputs=names,
        inputs=tuple(role for role in CUDA_INPUT_NAMES if role in dependencies),
        active_outputs=tuple(active),
        zero_outputs=tuple(zeros),
    )
    kernel_cache[key] = config
    if len(kernel_cache) > 512:
        kernel_cache.popitem(last=False)
    return config


def launch_kernel(inputs, config, activation_paths, index_bits, kernel_cache, source):
    """Load the kernel and return the requested output tensors."""
    order = config["order"]
    dtype = config["dtype"]
    input_names = INPUT_NAMES[order]
    available = config["available"]
    first = next((value for value in inputs if value is not None), None)
    if first is None:
        return tuple(None for _ in config["outputs"])
    shape, device = first.shape, first.device
    in_dim = out_dim = shape[-1]
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
                    source(activation_paths, in_dim, config, strided=True)
                    if strided_forward else source(activation_paths, in_dim, config)
                )
            if launch is None:
                precision = "fp64" if dtype == torch.float64 else "fp32"
                name = (
                    f"lunaeq_activation_strided_fwd" if strided_forward
                    else f"lunaeq_activation_{order}_{precision}"
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
