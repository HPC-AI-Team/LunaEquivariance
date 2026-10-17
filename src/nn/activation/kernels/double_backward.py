"""CUDA computations for Activation double backward."""

from .utils import make_path_types, make_binding, get_kernel_config, launch_kernel
from .backward import BACKWARD_FUNCTIONS, BACKWARD_MATH


DOUBLE_BACKWARD_RESULT = r'''struct DoubleBackwardResult {
    Element sx;
    Element sg;
};'''


DOUBLE_BACKWARD_MATH = r'''template<ActivationKind Kind>
__device__ __forceinline__ Element activation_second_derivative(
    Element x, const ActivationParameters& parameters) {
    if constexpr (Kind == ActivationKind::Identity || Kind == ActivationKind::Relu ||
                  Kind == ActivationKind::Abs) {
        return Element(0);
    } else if constexpr (Kind == ActivationKind::Tanh) {
        const Element value = ::tanh(x);
        return (Element(-2) * value) * (Element(1) - value * value);
    } else if constexpr (Kind == ActivationKind::Sigmoid) {
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
        return (Element(1) - Element(2) * sigmoid) *
               (Element(1) - sigmoid) * sigmoid;
    } else if constexpr (Kind == ActivationKind::Silu) {
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
        const Element complement = Element(1) - sigmoid;
        const Element slope = x * complement + Element(1);
        const Element product_term = sigmoid * complement;
        const Element sigmoid_term = (slope - sigmoid * x) * complement * sigmoid;
        return product_term + sigmoid_term;
    } else if constexpr (Kind == ActivationKind::Softplus) {
        const Element inner = x * parameters.beta;
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-inner));
        // Torch's double backward excludes equality at the linear threshold.
        const Element mask = Element(inner < parameters.threshold);
        return ((Element(1) - sigmoid) * sigmoid) * mask * parameters.beta;
    } else if constexpr (Kind == ActivationKind::Gelu) {
        const Element square = x * x;
        const Element density = ::exp(square * Element(-0.5)) *
                               Element(0.3989422804014327);
        return density * Element(2) - square * density;
    } else if constexpr (Kind == ActivationKind::GeluTanh) {
        const Element inner = (x + (x * x * x) * Element(0.044715)) *
                             Element(0.7978845608028654);
        const Element value = ::tanh(inner);
        const Element sech = Element(1) / ::cosh(inner);
        const Element complement = Element(1) - value * value;
        const Element inner_slope = (Element(1) + (x * Element(0.134145)) * x) *
                                   Element(0.7978845608028654);
        const Element inner_curvature = (x * Element(0.26829)) *
                                       Element(0.7978845608028654);
        const Element half_x = x * Element(0.5);
        // Keep cosh here: replacing sech^2 with 1-tanh^2 changes the tails
        // of Torch's registered GELU double backward after tanh saturates.
        const Element tanh_curvature = ((sech * Element(2)) * (-sech * value)) *
                                      inner_slope;
        const Element first_term = ((complement * Element(0.5)) * inner_slope) *
                                  Element(2);
        const Element second_term = (half_x * inner_slope) * tanh_curvature;
        const Element third_term = (half_x * complement) * inner_curvature;
        return (first_term + second_term) + third_term;
    }
}

template<ActivationKind Kind, bool Normalize, bool WriteSX, bool WriteSG>
__device__ __forceinline__ DoubleBackwardResult activation_double_backward_value(
    Element x, Element grad_out, Element rx, const ActivationParameters& parameters) {
    DoubleBackwardResult result{Element(0), Element(0)};
    if constexpr (WriteSG) {
        result.sg = activation_backward_value<Kind, false, false>(x, rx, parameters);
        if constexpr (Normalize) result.sg *= parameters.scale;
    }
    if constexpr (WriteSX && Kind != ActivationKind::Identity &&
                  Kind != ActivationKind::Relu && Kind != ActivationKind::Abs) {
        Element normalized_grad = grad_out;
        if constexpr (Normalize) normalized_grad *= parameters.scale;
        if constexpr (Kind == ActivationKind::Tanh) {
            const Element value = ::tanh(x);
            const Element grad_value = rx * ((value * Element(-2)) * normalized_grad);
            result.sx = grad_value * (Element(1) - value * value);
        } else if constexpr (Kind == ActivationKind::Sigmoid) {
            const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
            const Element grad_value = (rx * normalized_grad) *
                                (Element(1) + sigmoid * Element(-2));
            result.sx = grad_value * (Element(1) - sigmoid) * sigmoid;
        } else if constexpr (Kind == ActivationKind::Silu) {
            const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
            const Element complement = Element(1) - sigmoid;
            const Element slope = x * complement + Element(1);
            const Element weighted_sigmoid = normalized_grad * sigmoid;
            const Element product_grad = rx * weighted_sigmoid;
            const Element sigmoid_grad = (rx * slope) * normalized_grad - product_grad * x;
            result.sx = product_grad * complement +
                        (sigmoid_grad * complement) * sigmoid;
        } else if constexpr (Kind == ActivationKind::Softplus) {
            const Element inner = x * parameters.beta;
            const Element sigmoid = Element(1) / (Element(1) + ::exp(-inner));
            const Element mask = Element(inner < parameters.threshold);
            result.sx = (((rx * normalized_grad) * (Element(1) - sigmoid)) * sigmoid) *
                        mask * parameters.beta;
        } else {
            result.sx = (rx * normalized_grad) *
                        activation_second_derivative<Kind>(x, parameters);
        }
    }
    return result;
}'''


DOUBLE_BACKWARD_KERNEL = r'''template<auto Width, bool NeedX, bool NeedGrad, bool NeedRx,
                                    bool WriteSX, bool WriteSG, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_double_backward_kernel(
    const Element* __restrict__ x,
    const Element* __restrict__ grad_out,
    const Element* __restrict__ rx,
    Element* __restrict__ out_sx,
    Element* __restrict__ out_sg,
    int64_t num_items) {
    const int64_t count = num_items * Width;
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = item % Width;
        Element x_value = Element(0);
        Element grad_out_value = Element(0);
        Element rx_value = Element(0);
        if constexpr (NeedX) x_value = x[item];
        if constexpr (NeedGrad) grad_out_value = grad_out[item];
        if constexpr (NeedRx) rx_value = rx[item];
        DoubleBackwardResult result{Element(0), Element(0)};
        ((column < Paths::end
            ? (result = activation_double_backward_value<
                   Paths::kind, Paths::normalize, WriteSX, WriteSG>(
                   x_value, grad_out_value, rx_value, Paths::parameters()), true)
            : false) || ...);
        if constexpr (WriteSX) out_sx[item] = result.sx;
        if constexpr (WriteSG) out_sg[item] = result.sg;
    }
}

template<auto Width, bool NeedX, bool NeedGrad, bool NeedRx,
         bool WriteSX, bool WriteSG, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_double_backward_vectorized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    const float* __restrict__ rx,
    float* __restrict__ out_sx,
    float* __restrict__ out_sg,
    int64_t num_items) {
    const int64_t count = num_items * (Width / 4);
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = (item * 4) % Width;
        float4 x_values = make_float4(0, 0, 0, 0);
        float4 grad_values = make_float4(0, 0, 0, 0);
        float4 rx_values = make_float4(0, 0, 0, 0);
        if constexpr (NeedX) x_values = reinterpret_cast<const float4*>(x)[item];
        if constexpr (NeedGrad) grad_values = reinterpret_cast<const float4*>(grad_out)[item];
        if constexpr (NeedRx) rx_values = reinterpret_cast<const float4*>(rx)[item];
        DoubleBackwardResult lane_x{Element(0), Element(0)};
        DoubleBackwardResult lane_y{Element(0), Element(0)};
        DoubleBackwardResult lane_z{Element(0), Element(0)};
        DoubleBackwardResult lane_w{Element(0), Element(0)};
        ((column < Paths::end
            ? (lane_x = activation_double_backward_value<
                   Paths::kind, Paths::normalize, WriteSX, WriteSG>(
                   x_values.x, grad_values.x, rx_values.x, Paths::parameters()),
               lane_y = activation_double_backward_value<
                   Paths::kind, Paths::normalize, WriteSX, WriteSG>(
                   x_values.y, grad_values.y, rx_values.y, Paths::parameters()),
               lane_z = activation_double_backward_value<
                   Paths::kind, Paths::normalize, WriteSX, WriteSG>(
                   x_values.z, grad_values.z, rx_values.z, Paths::parameters()),
               lane_w = activation_double_backward_value<
                   Paths::kind, Paths::normalize, WriteSX, WriteSG>(
                   x_values.w, grad_values.w, rx_values.w, Paths::parameters()), true)
            : false) || ...);
        if constexpr (WriteSX) {
            reinterpret_cast<float4*>(out_sx)[item] = make_float4(
                lane_x.sx, lane_y.sx, lane_z.sx, lane_w.sx);
        }
        if constexpr (WriteSG) {
            reinterpret_cast<float4*>(out_sg)[item] = make_float4(
                lane_x.sg, lane_y.sg, lane_z.sg, lane_w.sg);
        }
    }
}

template<auto Width, bool NeedX, bool NeedGrad, bool NeedRx,
         bool WriteSX, bool WriteSG, class... Paths>
void run_activation_double_backward(
    const Element* x, const Element* grad_out, const Element* rx,
    Element* out_sx, Element* out_sg, int64_t num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    const int64_t count = num_items * Width;
    // A float4 warp covers 128 columns; keep each warp on one activation path.
    if constexpr (std::is_same_v<Element, float> && Width % 4 == 0 &&
                  ((Paths::end % 4 == 0) && ...) &&
                  (sizeof...(Paths) == 1 || ((Paths::end % 128 == 0) && ...))) {
        if (count >= (int64_t(1) << 20) &&
            (!NeedX || reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
            (!NeedGrad || reinterpret_cast<uintptr_t>(grad_out) % 16 == 0) &&
            (!NeedRx || reinterpret_cast<uintptr_t>(rx) % 16 == 0)) {
            const int64_t vector_count = num_items * (Width / 4);
            const unsigned blocks = std::min<int64_t>(65535, (vector_count + 127) / 128);
            activation_double_backward_vectorized_kernel<Width, NeedX, NeedGrad, NeedRx, WriteSX, WriteSG, Paths...><<<blocks, 128, 0, stream>>>(
                x, grad_out, rx, out_sx, out_sg, num_items);
            return;
        }
    }
    const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
    activation_double_backward_kernel<Width, NeedX, NeedGrad, NeedRx, WriteSX, WriteSG, Paths...><<<blocks, 128, 0, stream>>>(
        x, grad_out, rx, out_sx, out_sg, num_items);
}'''


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


MODULE_BINDING = r'''PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&launch);}
'''


def required_inputs(paths, available):
    """Inputs for gradients with respect to x (sx) and grad_out (sg)."""
    if "rx" not in available:
        return {}
    sx = sg = None
    for width, spec in paths:
        kind = spec["kind"]
        if kind != "identity":
            if sx is None:
                sx = set()
            if width:
                if kind == "relu":
                    sx.add("x")
                elif kind != "abs":
                    sx.update(("x", "grad_out", "rx"))
        if sg is None:
            sg = set()
        if width:
            sg.add("rx")
            if kind != "identity":
                sg.add("x")
    return {"sx": sx, "sg": sg}


def make_source(activation_paths, in_dim, config, *, tensor_only=False, fragments=False):
    declarations, paths, _ = make_path_types(activation_paths, in_dim, config["dtype"])
    declarations += ["\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH]
    declarations += [DOUBLE_BACKWARD_RESULT, DOUBLE_BACKWARD_MATH]
    template_args = [str(in_dim)]
    inputs = ('x', 'grad_out', 'rx')
    template_args.extend(str(role in config["inputs"]).lower() for role in inputs)
    template_args.extend(str(name in config["active_outputs"]).lower() for name in config["outputs"])
    template_args.extend(paths)
    arguments = [f"{role}.data_ptr<Element>()" if role in config["inputs"] else "nullptr" for role in inputs]
    arguments.extend(
        f"result_{name}.data_ptr<Element>()" if name in config["active_outputs"] else "nullptr"
        for name in config["outputs"]
    )
    arguments.extend(("num_items", "stream"))
    launch = f"run_activation_double_backward<{','.join(template_args)}>({','.join(arguments)});"
    return make_binding(
        in_dim, config, declarations=declarations, kernel_body=DOUBLE_BACKWARD_KERNEL, binding=BINDING,
        module_binding=MODULE_BINDING, launch=launch, tensor_only=tensor_only, fragments=fragments,
    )


def activation_double_backward(features, grad_output, grad_grad_input, paths, index_bits, kernel_cache, gate_layout, dtype, *, needs=None):
    inputs = features, grad_output, grad_grad_input
    available = tuple(name for name, value in zip(("x", "grad_out", "rx"), inputs) if value is not None)
    config = get_kernel_config(
        paths, 2, dtype, kernel_cache, needs=needs, available=available, gate_layout=gate_layout
    )
    return launch_kernel(inputs, config, paths, index_bits, kernel_cache, make_source, gate_layout)
