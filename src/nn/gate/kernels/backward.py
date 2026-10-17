"""Gate CUDA input gradient: activation slope and component contraction."""

from ...activation.kernels.backward import (
    BACKWARD_FUNCTIONS, BACKWARD_MATH, BINDING, MODULE_BINDING,
    required_inputs as activation_required_inputs,
)
from .utils import make_binding, make_path_types, make_schedule

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
template<class Path>
__device__ __forceinline__ void gate_backward_product(
    const Element* x, const Element* grad_out, Element* out_dx,
    Index row, Index copy) {
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
        out_dx[input + component] = gate_multiply(grad, gate);
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
    arguments = [f"{name}.data_ptr<Element>()" if name in config["inputs"] else "nullptr"
                 for name in ("x", "grad_out")]
    arguments.append("result_dx.data_ptr<Element>()")
    launch = (
        f"if(num_items>0) gate_backward<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
        f"({','.join(arguments)},num_items);"
    )
    return make_binding(
        meta, config, declarations=declarations,
        kernel_body=BACKWARD_KERNEL.format(count=count, schedule=schedule),
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        tensor_only=tensor_only, fragments=fragments,
    )
