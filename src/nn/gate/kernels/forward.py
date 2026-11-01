"""Gate CUDA forward, dense/strided schedules, and training native binding."""

from math import gcd
from textwrap import indent

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension

from ...activation.kernels.backward import (
    BACKWARD_CALL, BACKWARD_CHECKS,
)
from ...activation.kernels.forward import FORWARD_CALL
from ...activation.kernels.utils import (
    BINDING, MODULE_BINDING, TENSOR_BINDING, TRAINING_MODULE_BINDING,
)
from .utils import (
    DIRECT_STORE_BYTES, FORWARD_MIN_BLOCKS, FORWARD_THREADS, GATE_STORES,
    MIN_COPY_TILE, SHARED_MEMORY_LIMIT, WARP_SIZE,
    get_kernel_config, launch_kernel, make_binding,
    make_path_types, make_schedule,
)

FORWARD_MATH = r'''template<class Path>
__device__ __forceinline__ void gate_forward_scalar(
    const Element* x, int64_t x_row, int64_t feature_stride,
    Element* out, Index row, Index copy) {
    const Element value = x[x_row + (Path::input_offset + copy) * feature_stride];
    out[row * OutWidth + Path::output_offset + copy] =
        activation_forward_value<Path::kind, Path::normalize>(value, Path::parameters());
}
template<class Path, bool SharedOutput = false>
__device__ __forceinline__ void gate_forward_product(
    const Element* x, int64_t x_row, int64_t feature_stride,
    Element* out, Index row, Index copy, Element* shared_values = nullptr) {
    const Element value = x[x_row + (Path::gate_offset + copy) * feature_stride];
    const Element gate = activation_forward_value<Path::kind, Path::normalize>(
        value, Path::parameters());
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        const Index input = Path::input_offset + copy * Path::components + component;
        const Index output = row * OutWidth + Path::output_offset +
                             copy * Path::components + component;
        const Element result = gate_multiply(gate, x[x_row + input * feature_stride]);
        if constexpr (SharedOutput) {
            shared_values[(threadIdx.x % 32) * Path::components + component] = result;
        } else {
            out[output] = result;
        }
    }
}'''


FORWARD_KERNEL = r'''__global__ __launch_bounds__(128, 1) void gate_forward(
    const Element* __restrict__ x, Element* __restrict__ out,
    int64_t num_items{layout_parameter}) {{
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < num_items * {count}; item += Index(gridDim.x) * 128) {{
        const Index row = item / {count};
        const Index group = item % {count};
{input_layout}
{schedule}
    }}
}}'''


COALESCED_FORWARD_KERNEL = r'''__global__ __launch_bounds__({threads}, {min_blocks}) void gate_forward(
    const Element* __restrict__ x, Element* __restrict__ out,
    int64_t num_items) {{
    __shared__ __align__(16) Element shared_values[{threads} * {max_components}];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const Index tasks = ((num_items + {rows_per_warp} - 1) / {rows_per_warp}) * {tiles};
    for (Index task = Index(blockIdx.x) * {warps} + warp;
         task < tasks; task += Index(gridDim.x) * {warps}) {{
        const Index row = (task / {tiles}) * {rows_per_warp} + lane / {copy_tile};
        const Index group = task % {tiles};
        const Index x_row = row * {in_width};
{schedule}
    }}
}}'''


GATE_FORWARD_BINDING = r'''
torch::Tensor launch_fast_forward(torch::Tensor x) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_items=x.numel()/{width};
    constexpr int64_t padding=65535*128;
    TORCH_CHECK(num_items<=(std::numeric_limits<int64_t>::max()-padding)/{max_width},"Gate task offsets exceed int64");
    pybind11::tuple results;
    if(x.is_contiguous() || x.dim()>64) {{
        results=dense_forward::launch(x.reshape({{num_items,int64_t({width})}}).contiguous(),num_items);
    }} else {{
        results=strided_forward::launch(x,num_items);
    }}
    auto out=results[0].cast<torch::Tensor>();
    std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end());
    shape.back()={out_width};
    return out.reshape(shape);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
    m.def("launch_fast_forward",&launch_fast_forward);
}}
'''


def make_source(meta, config, *, strided=False, tensor_only=False, fragments=False):
    assert config["order"] == 0 and config["inputs"] == ("x",)
    assert config["active_outputs"] == ("out",)
    declarations, paths = make_path_types(meta, config["dtype"])
    count, schedule = make_schedule(
        paths,
        "gate_forward_scalar<{path}>(x, x_row, feature_stride, out, row, copy);",
        "gate_forward_product<{path}>(x, x_row, feature_stride, out, row, copy);",
    )
    declarations += [f"constexpr int OutWidth = {meta['out_dim']};", FORWARD_MATH]
    max_components = max((path["components"] for path in meta["gated_paths"]), default=1)
    copy_tile = gcd(WARP_SIZE, *(copies for _, _, copies in paths))
    rows_per_warp = WARP_SIZE // copy_tile
    coalesced = (
        not strided and bool(meta["gated_paths"])
        and max_components * FORWARD_THREADS * (8 if config["dtype"] == torch.float64 else 4) <= SHARED_MEMORY_LIMIT
        and copy_tile >= MIN_COPY_TILE
    )
    if coalesced:
        declarations.append(GATE_STORES)
        tiles, schedule = make_schedule(
            paths,
            "if (row < num_items && copy < {copies}) gate_forward_scalar<{path}>(x, x_row, 1, out, row, copy);",
            "if constexpr ({path}::components * sizeof(Element) <= " + str(DIRECT_STORE_BYTES) + ") {{\n"
            "    if (row < num_items && copy < {copies}) gate_forward_product<{path}>(x, x_row, 1, out, row, copy);\n"
            "}} else {{\n"
            f"    Element* values = shared_values + warp * 32 * {max_components};\n"
            "    if (row < num_items && copy < {copies}) gate_forward_product<{path}, true>(x, x_row, 1, out, row, copy, values);\n"
            "    __syncwarp();\n"
            f"    gate_store_components<OutWidth, {{path}}::output_offset, {{path}}::components, {{copies}}, {copy_tile}>(values, out, row, copy_base, num_items);\n"
            "    __syncwarp();\n}}",
            warp_groups=True, copy_tile=copy_tile,
        )
        kernel = COALESCED_FORWARD_KERNEL.format(
            tiles=tiles, max_components=max_components, schedule=schedule,
            rows_per_warp=rows_per_warp, copy_tile=copy_tile,
            threads=FORWARD_THREADS, min_blocks=FORWARD_MIN_BLOCKS, warps=FORWARD_THREADS // WARP_SIZE,
            in_width=meta["in_dim"],
        )
        warps = FORWARD_THREADS // WARP_SIZE
        launch = (
            f"if(num_items>0) gate_forward<<<std::min<int64_t>(65535,(((num_items+{rows_per_warp-1})/{rows_per_warp})*{tiles}+{warps-1})/{warps}),{FORWARD_THREADS},0,stream>>>"
            f"(x.data_ptr<Element>(),result_out.data_ptr<Element>(),num_items);"
        )
        return make_binding(
            meta, config, declarations=declarations, kernel_body=kernel,
            binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
            tensor_only=tensor_only, fragments=fragments,
        )
    kernel = FORWARD_KERNEL.format(
        count=count, schedule=schedule,
        layout_parameter=",ForwardLayout x_layout" if strided else "",
        input_layout=indent(("const int64_t x_row = forward_row_offset(row, x_layout);\n"
                      "const int64_t feature_stride = x_layout.feature_stride;" if strided else
                      f"const int64_t x_row = row * {meta['in_dim']};\n"
                      "const int64_t feature_stride = 1;"), "        "),
    )
    layout_argument = ",x_layout" if strided else ""
    launch = (
        f"if(num_items>0) gate_forward<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
        f"(x.data_ptr<Element>(),result_out.data_ptr<Element>(),num_items{layout_argument});"
    )
    return make_binding(
        meta, config, declarations=declarations, kernel_body=kernel,
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        strided=strided, tensor_only=tensor_only, fragments=fragments,
    )


def make_training_source(meta, forward, backward):
    """Build one extension source for dense/strided forward and backward."""
    from .backward import make_source as make_backward_source

    kernels = []
    for config in (forward, backward):
        order = config["order"]
        assert order in (0, 1) and config["active_outputs"] == (("out",) if order == 0 else ("dx",))
        assert config["inputs"] == (("x",) if order == 0 else ("x", "grad_out"))
        build = make_backward_source if order else make_source
        includes, dense = build(meta, config, tensor_only=True, fragments=True)
        bodies = "\nnamespace contiguous {\n" + dense + "\n}\n"
        if order == 0:
            strided_headers, strided_source = make_source(
                meta, config, strided=True, tensor_only=True, fragments=True
            )
            includes += strided_headers
            bodies += "\nnamespace strided {\n" + strided_source + "\n}\n"
        includes += ["#include <limits>", "#include <vector>"]
        width, out_width = meta["in_dim"], meta["out_dim"]
        args = "torch::Tensor x" + (",torch::Tensor grad_out" if order else "")
        checks = BACKWARD_CHECKS.format(out_width=out_width) if order else ""
        call = (BACKWARD_CALL.format(width=width, out_width=out_width) if order else
                FORWARD_CALL.format(results_type="torch::Tensor", width=width))
        wrapper = TENSOR_BINDING.format(
            args=args, scalar="Double" if meta["dtype"] == torch.float64 else "Float",
            width=width, max_width=max(width, out_width, 1), checks=checks, call=call,
            result_value="results", result_width=width if order else out_width,
        )
        kernels.append((includes, bodies + wrapper))
    (includes0, forward_impl), (includes1, backward_impl) = kernels
    return ("\n".join(dict.fromkeys([*includes0, *includes1])) +
            "\nnamespace forward_impl {\n" + forward_impl + "\n}\n" +
            "\nnamespace backward_impl {\n" + backward_impl + "\n}\n" +
            TRAINING_MODULE_BINDING)


def make_host_source(meta, config):
    """Bind sorted Gate outputs while reading the input's original strides."""
    assert meta["dtype"] == torch.float32
    dense_headers, dense_source = make_source(meta, config, fragments=True)
    strided_headers, strided_source = make_source(meta, config, strided=True, fragments=True)
    includes = "\n".join(dict.fromkeys([
        *dense_headers, *strided_headers, "#include <limits>", "#include <vector>",
    ]))
    source = (includes + "\nnamespace dense_forward {\n" + dense_source + "\n}\n" +
              "\nnamespace strided_forward {\n" + strided_source + "\n}\n")
    return source + GATE_FORWARD_BINDING.format(
        width=meta["in_dim"], out_width=meta["out_dim"],
        max_width=max(meta["in_dim"], meta["out_dim"], 1),
    )


def gate_forward(features, meta, index_bits, kernel_cache):
    """Load Gate forward and both first-backward modes in one extension."""
    dtype = features.dtype
    device = features.device
    numel = features.numel()
    in_dim, out_dim = meta["in_dim"], meta["out_dim"]
    bound = numel + 65535 * 128
    if out_dim > in_dim and in_dim:
        bound = (numel // in_dim) * out_dim + 65535 * 128
    if not features.is_contiguous():
        bound = max(bound, sum(
            (size - 1) * stride for size, stride in zip(features.shape, features.stride())
        ) + 65535 * 128)

    bits = 64 if bound > (1 << 31) - 1 else index_bits
    key = "forward_backward", dtype, device, bits
    entry = kernel_cache.get(key)
    if entry is None:
        from . import backward

        arch = get_cuda_arch(device)
        forward_config = get_kernel_config(meta, kernel_cache, 0, dtype=dtype)
        backward_config = get_kernel_config(
            meta, kernel_cache, 1, available=("x", "grad_out"), dtype=dtype
        )
        differentiable_config = get_kernel_config(
            meta, kernel_cache, 1, grad_mode=True, dtype=dtype
        )
        differentiable = bool(differentiable_config["active_outputs"])
        forward_kernel = backward_kernel = None
        if (
            in_dim and out_dim and bound <= (1 << 63) - 1
            and forward_config["inputs"] == ("x",)
            and forward_config["active_outputs"] == ("out",)
            and backward_config["inputs"] in (("x", "grad_out"), ("grad_out",))
            and backward_config["active_outputs"] == ("dx",)
        ):
            layout = {**meta, "dtype": dtype}
            sources = {"differentiable_backward": backward.make_source(layout, differentiable_config)}
            if backward_config["inputs"] == ("x", "grad_out"):
                sources["training"] = make_training_source(layout, forward_config, backward_config)
            else:
                sources["backward"] = backward.make_source(layout, backward_config)
                if dtype == torch.float32:
                    sources["host_forward"] = make_host_source(layout, forward_config)
                else:
                    sources["forward"] = make_source(layout, forward_config)
                    sources["strided_forward"] = make_source(layout, forward_config, strided=True)
            extension = load_cuda_extension(
                "lunaeq_gate_forward_backward", combine_cuda_sources(sources),
                index_bits=bits, arch=arch, use_fast_math=False, minimal_headers=True,
            )
            kernel_cache[("forward_backward_module", dtype, device, bits)] = extension
            if "training" in sources:
                forward_kernel = extension.training.launch_forward
                backward_kernel = extension.training.launch_backward
            elif "host_forward" in sources:
                kernel_cache[("gate_forward", dtype, device, bits)] = extension.host_forward.launch_fast_forward

        save_input = bool(meta["gated_paths"]) or any(
            path["activation"]["kind"] != "identity" for path in meta["scalar_paths"]
        )
        save_input |= backward_kernel is not None
        entry = forward_kernel, backward_kernel, differentiable, save_input, forward_config
        kernel_cache[key] = entry
        while len(kernel_cache) > 512:
            kernel_cache.popitem(last=False)

    forward_kernel, backward_kernel, differentiable, save_input, config = entry
    if forward_kernel is not None:
        result = forward_kernel(features)
    elif dtype == torch.float32 and in_dim and out_dim:
        key = "gate_forward", dtype, device, bits
        launch = kernel_cache.get(key)
        if launch is None:
            launch = load_cuda_extension(
                "gate_host_forward", make_host_source({**meta, "dtype": dtype}, config),
                index_bits=bits, arch=get_cuda_arch(device), use_fast_math=False, minimal_headers=True,
            ).launch_fast_forward
            kernel_cache[key] = launch
        result = launch(features).detach()
    else:
        result = launch_kernel(
            (features,), config, meta, index_bits, kernel_cache, make_source
        )[0].detach()
    return result, backward_kernel, differentiable, save_input
