"""Gate CUDA input gradient: activation slope and component contraction."""

from math import gcd

import torch

from ...activation.kernels.backward import (
    BACKWARD_FUNCTIONS, BACKWARD_MATH,
    required_inputs as activation_required_inputs,
)
from ...activation.kernels.utils import BINDING, MODULE_BINDING
from .utils import (
    BACKWARD_THREADS, DIRECT_STORE_BYTES, GATE_STORES, MIN_COPY_TILE,
    SHARED_MEMORY_LIMIT, WARP_SIZE,
    get_kernel_config, launch_kernel, make_binding,
    make_path_types, make_schedule,
)

BACKWARD_MATH_SOURCE = r'''template<class Path>
__device__ __forceinline__ void gate_backward_scalar(
    const Element* x, const Element* grad_out, Element* out_dx,
    Index row, Index copy) {
    const Index input = row * InWidth + Path::input_offset + copy;
    const Index output = row * OutWidth + Path::output_offset + copy;
    Element value = Element(0);
    if constexpr (NeedX) value = x[input];
    out_dx[input] = activation_backward_value<Path::kind, Path::normalize, Differentiable>(
        value, grad_out[output], Path::parameters());
}
template<class Path, bool SharedOutput = false>
__device__ __forceinline__ void gate_backward_product(
    const Element* x, const Element* grad_out, Element* out_dx,
    Index row, Index copy, Element* shared_values = nullptr) {
    const Index gate_input = row * InWidth + Path::gate_offset + copy;
    const Index input = row * InWidth + Path::input_offset + copy * Path::components;
    const Index output = row * OutWidth + Path::output_offset + copy * Path::components;
    const Element value = x[gate_input];
    const auto parameters = Path::parameters();
    const Element gate = activation_forward_value<Path::kind, Path::normalize>(value, parameters);
    Element contraction[Path::components];
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        const Element grad = grad_out[output + component];
        const Element result = gate_multiply(grad, gate);
        if constexpr (SharedOutput) {
            shared_values[(threadIdx.x % 32) * Path::components + component] = result;
        } else {
            out_dx[input + component] = result;
        }
        contraction[component] = gate_multiply(grad, x[input + component]);
    }
    out_dx[gate_input] = activation_backward_value<Path::kind, Path::normalize, Differentiable>(
        value, gate_component_sum(contraction), parameters);
}'''


BACKWARD_KERNEL = r'''__global__ __launch_bounds__(128, 1) void gate_backward(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    Element* __restrict__ out_dx, int64_t num_items) {{
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < num_items * {count}; item += Index(gridDim.x) * 128) {{
        const Index row = item / {count};
        const Index group = item % {count};
{schedule}
    }}
}}'''


COALESCED_BACKWARD_KERNEL = r'''__global__ __launch_bounds__({threads}, 1) void gate_backward(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    Element* __restrict__ out_dx, int64_t num_items) {{
    __shared__ __align__(16) Element shared_values[{threads} * {max_components}];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const Index tasks = ((num_items + {rows_per_warp} - 1) / {rows_per_warp}) * {tiles};
    for (Index task = Index(blockIdx.x) * {warps} + warp;
         task < tasks; task += Index(gridDim.x) * {warps}) {{
        const Index row = (task / {tiles}) * {rows_per_warp} + lane / {copy_tile};
        const Index group = task % {tiles};
{schedule}
    }}
}}'''


def required_inputs(meta):
    required = activation_required_inputs(meta["scalar_activation"])
    if meta["gated_paths"]:
        reads = required.get("dx")
        required["dx"] = (set() if reads is None else reads) | {"x", "grad_out"}
    return required


def make_source(meta, config, *, tensor_only=False, fragments=False):
    assert config["order"] == 1
    declarations, paths = make_path_types(meta, config["dtype"])
    declarations += [
        f"constexpr int InWidth = {meta['in_dim']};\nconstexpr int OutWidth = {meta['out_dim']};",
        f"constexpr bool NeedX = {str('x' in config['inputs']).lower()};\n"
        f"constexpr bool Differentiable = {str(config['grad_mode']).lower()};",
        "\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH, BACKWARD_MATH_SOURCE,
    ]
    count, schedule = make_schedule(
        paths,
        "gate_backward_scalar<{path}>(x, grad_out, out_dx, row, copy);",
        "gate_backward_product<{path}>(x, grad_out, out_dx, row, copy);",
    )
    max_components = max((path["components"] for path in meta["gated_paths"]), default=1)
    copy_tile = gcd(WARP_SIZE, *(copies for _, _, copies in paths))
    rows_per_warp = WARP_SIZE // copy_tile
    coalesced = (
        bool(meta["gated_paths"])
        and max_components * BACKWARD_THREADS * (8 if config["dtype"] == torch.float64 else 4) <= SHARED_MEMORY_LIMIT
        and copy_tile >= MIN_COPY_TILE
    )
    if coalesced:
        declarations.append(GATE_STORES)
        tiles, schedule = make_schedule(
            paths,
            "if (row < num_items && copy < {copies}) gate_backward_scalar<{path}>(x, grad_out, out_dx, row, copy);",
            "if constexpr ({path}::components * sizeof(Element) <= " + str(DIRECT_STORE_BYTES) + ") {{\n"
            "    if (row < num_items && copy < {copies}) gate_backward_product<{path}>(x, grad_out, out_dx, row, copy);\n"
            "}} else {{\n"
            f"    Element* values = shared_values + warp * 32 * {max_components};\n"
            "    if (row < num_items && copy < {copies}) gate_backward_product<{path}, true>(x, grad_out, out_dx, row, copy, values);\n"
            "    __syncwarp();\n"
            f"    gate_store_components<InWidth, {{path}}::input_offset, {{path}}::components, {{copies}}, {copy_tile}>(values, out_dx, row, copy_base, num_items);\n"
            "    __syncwarp();\n}}",
            warp_groups=True, copy_tile=copy_tile,
        )
        warps = BACKWARD_THREADS // WARP_SIZE
        kernel = COALESCED_BACKWARD_KERNEL.format(
            threads=BACKWARD_THREADS, max_components=max_components, warps=warps,
            tiles=tiles, schedule=schedule, rows_per_warp=rows_per_warp, copy_tile=copy_tile,
        )
        geometry = f"<<<std::min<int64_t>(65535,(((num_items+{rows_per_warp-1})/{rows_per_warp})*{tiles}+{warps-1})/{warps}),{BACKWARD_THREADS},0,stream>>>"
    else:
        kernel = BACKWARD_KERNEL.format(count=count, schedule=schedule)
        geometry = f"<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
    arguments = [f"{name}.data_ptr<Element>()" if name in config["inputs"] else "nullptr"
                 for name in ("x", "grad_out")]
    arguments.append("result_dx.data_ptr<Element>()")
    launch = (
        f"if(num_items>0) gate_backward{geometry}"
        f"({','.join(arguments)},num_items);"
    )
    return make_binding(
        meta, config, declarations=declarations,
        kernel_body=kernel,
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        tensor_only=tensor_only, fragments=fragments,
    )


def gate_backward(features, grad_output, meta, index_bits, kernel_cache, dtype, *, grad_mode=False):
    inputs = features, grad_output
    available = tuple(name for name, value in zip(("x", "grad_out"), inputs) if value is not None)
    config = get_kernel_config(
        meta, kernel_cache, 1, available=available, grad_mode=grad_mode, dtype=dtype
    )
    return launch_kernel(inputs, config, meta, index_bits, kernel_cache, make_source)[0]
