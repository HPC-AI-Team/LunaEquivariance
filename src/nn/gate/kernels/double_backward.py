"""Gate CUDA second gradients, including gate/vector mixed directions."""

from ...activation.kernels.backward import BACKWARD_FUNCTIONS, BACKWARD_MATH
from ...activation.kernels.double_backward import (
    BINDING, DOUBLE_BACKWARD_MATH, DOUBLE_BACKWARD_RESULT, MODULE_BINDING,
    required_inputs as activation_required_inputs,
)
from .utils import make_binding, make_path_types, make_schedule

DOUBLE_BACKWARD_MATH_SOURCE = r'''template<class Path>
__device__ __forceinline__ void gate_double_backward_scalar(
    const Element* x, const Element* grad_out, const Element* rx,
    Element* out_sx, Element* out_sg, Index row, Index copy) {
    const Index input = row * InWidth + Path::input_offset + copy;
    const Index output = row * OutWidth + Path::output_offset + copy;
    Element value = Element(0), grad = Element(0), direction = Element(0);
    if constexpr (NeedX) value = x[input];
    if constexpr (NeedGrad) grad = grad_out[output];
    if constexpr (NeedRx) direction = rx[input];
    const auto result = activation_double_backward_value<
        Path::kind, Path::normalize, WriteSX, WriteSG>(value, grad, direction, Path::parameters());
    if constexpr (WriteSX) out_sx[input] = result.sx;
    if constexpr (WriteSG) out_sg[output] = result.sg;
}
template<class Path>
__device__ __forceinline__ void gate_double_backward_product(
    const Element* x, const Element* grad_out, const Element* rx,
    Element* out_sx, Element* out_sg, Index row, Index copy) {
    const Index gate_input = row * InWidth + Path::gate_offset + copy;
    const Index input = row * InWidth + Path::input_offset + copy * Path::components;
    const Index output = row * OutWidth + Path::output_offset + copy * Path::components;
    Element value = Element(0), gate_direction = Element(0);
    if constexpr (NeedX) value = x[gate_input];
    if constexpr (NeedRx) gate_direction = rx[gate_input];
    Element vectors[Path::components], directions[Path::components], grads[Path::components];
    Element contraction[Path::components], direction_contraction[Path::components];
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        vectors[component] = directions[component] = grads[component] = Element(0);
        if constexpr (NeedX) vectors[component] = x[input + component];
        if constexpr (NeedRx) directions[component] = rx[input + component];
        if constexpr (NeedGrad) grads[component] = grad_out[output + component];
        contraction[component] = gate_multiply(grads[component], vectors[component]);
        direction_contraction[component] = gate_multiply(directions[component], grads[component]);
    }
    const auto parameters = Path::parameters();
    const Element moment = gate_component_sum(contraction);
    const Element direction_moment = gate_component_sum(direction_contraction);
    const auto result = activation_double_backward_value<
        Path::kind, Path::normalize, WriteSX, true>(value, moment, gate_direction, parameters);
    if constexpr (WriteSX) {
        const Element mixed = activation_backward_value<Path::kind, Path::normalize, true>(
            value, direction_moment, parameters);
        out_sx[gate_input] = gate_add(result.sx, mixed);
    }
    Element gate = Element(0);
    if constexpr (WriteSG) {
        gate = activation_forward_value<Path::kind, Path::normalize>(value, parameters);
    }
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        if constexpr (WriteSX) {
            out_sx[input + component] = gate_multiply(result.sg, grads[component]);
        }
        if constexpr (WriteSG) {
            out_sg[output + component] = gate_add(
                gate_multiply(result.sg, vectors[component]),
                gate_multiply(directions[component], gate));
        }
    }
}'''


DOUBLE_BACKWARD_KERNEL = r'''__global__ __launch_bounds__(128, 1) void gate_double_backward(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, Element* __restrict__ out_sx,
    Element* __restrict__ out_sg, int64_t num_items) {{
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < num_items * {count}; item += Index(gridDim.x) * 128) {{
        const Index row = item / {count};
        const Index group = item % {count};
        {schedule}
    }}
}}'''


def required_inputs(meta, available):
    required = activation_required_inputs(meta["scalar_activation"], available)
    if "rx" not in available:
        return required
    for path in meta["gated_paths"]:
        sx = required.get("sx")
        sg = required.get("sg")
        sx_reads = {"grad_out", "rx"}
        if path["activation"]["kind"] != "identity":
            sx_reads.add("x")
        required["sx"] = (set() if sx is None else sx) | sx_reads
        required["sg"] = (set() if sg is None else sg) | {"x", "rx"}
    return required


def make_source(meta, config, *, tensor_only=False, fragments=False):
    assert config["order"] == 2
    declarations, paths = make_path_types(meta, config["dtype"])
    flags = dict(NeedX="x" in config["inputs"], NeedGrad="grad_out" in config["inputs"],
                 NeedRx="rx" in config["inputs"], WriteSX="sx" in config["active_outputs"],
                 WriteSG="sg" in config["active_outputs"])
    declarations += [
        f"constexpr int InWidth = {meta['in_dim']};\nconstexpr int OutWidth = {meta['out_dim']};",
        "\n".join(f"constexpr bool {name} = {str(value).lower()};" for name, value in flags.items()),
        "\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH, DOUBLE_BACKWARD_RESULT,
        DOUBLE_BACKWARD_MATH, DOUBLE_BACKWARD_MATH_SOURCE,
    ]
    count, schedule = make_schedule(
        paths,
        "gate_double_backward_scalar<{path}>(x, grad_out, rx, out_sx, out_sg, row, copy);",
        "gate_double_backward_product<{path}>(x, grad_out, rx, out_sx, out_sg, row, copy);",
    )
    arguments = [f"{name}.data_ptr<Element>()" if name in config["inputs"] else "nullptr"
                 for name in ("x", "grad_out", "rx")]
    arguments += [f"result_{name}.data_ptr<Element>()" if name in config["active_outputs"] else "nullptr"
                  for name in ("sx", "sg")]
    launch = (
        f"if(num_items>0) gate_double_backward<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
        f"({','.join(arguments)},num_items);"
    )
    return make_binding(
        meta, config, declarations=declarations,
        kernel_body=DOUBLE_BACKWARD_KERNEL.format(count=count, schedule=schedule),
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        tensor_only=tensor_only, fragments=fragments,
    )
