"""Activation backward CUDA kernel and launch."""

from .utils import make_path_types, make_binding, get_kernel_config, launch_kernel


BACKWARD_MATH = r'''template<ActivationKind Kind, bool Normalize, bool Differentiable>
__device__ __forceinline__ Element activation_backward_value(
    Element x, Element grad_out, const ActivationParameters& parameters) {
    if constexpr (Normalize) grad_out *= parameters.scale;
    if constexpr (Kind == ActivationKind::Identity) {
        return grad_out;
    } else if constexpr (Kind == ActivationKind::Tanh) {
        const Element value = ::tanh(x);
        return activation_tanh_backward(grad_out, value);
    } else if constexpr (Kind == ActivationKind::Sigmoid) {
        const Element value = activation_sigmoid(x);
        return activation_sigmoid_backward(grad_out, value);
    } else if constexpr (Kind == ActivationKind::Silu) {
        if constexpr (Differentiable) {
            const Element sigmoid = activation_sigmoid(x);
            // Torch's differentiable backward rounds each tensor operation.
            if constexpr (std::is_same_v<Element, float>) {
                const Element correction = __fsub_rn(Element(1), sigmoid);
                const Element factor = __fadd_rn(__fmul_rn(x, correction), Element(1));
                return __fmul_rn(__fmul_rn(grad_out, sigmoid), factor);
            } else {
                const Element correction = __dsub_rn(Element(1), sigmoid);
                const Element factor = __dadd_rn(__dmul_rn(x, correction), Element(1));
                return __dmul_rn(__dmul_rn(grad_out, sigmoid), factor);
            }
        }
        return activation_silu_backward(grad_out, x);
    } else if constexpr (Kind == ActivationKind::Softplus) {
        return activation_softplus_backward(
            grad_out, x, parameters.beta, parameters.threshold);
    } else if constexpr (Kind == ActivationKind::Relu) {
        return x <= Element(0) ? Element(0) : grad_out;
    } else if constexpr (Kind == ActivationKind::Abs) {
        return grad_out * Element((Element(0) < x) - (x < Element(0)));
    } else if constexpr (Kind == ActivationKind::Gelu) {
        return activation_gelu_backward_exact(grad_out, x);
    } else if constexpr (Kind == ActivationKind::GeluTanh) {
        return activation_gelu_backward_tanh(grad_out, x);
    }
}'''


BACKWARD_KERNEL = r'''template<
    auto Width, bool Differentiable, bool NeedX, bool NeedGrad, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_backward_kernel(
    const Element* __restrict__ x,
    const Element* __restrict__ grad_out,
    Element* __restrict__ grad_x,
    int64_t num_items) {
    const int64_t count = num_items * Width;
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = item % Width;
        Element value = Element(0);
        if constexpr (NeedX) value = x[item];
        Element gradient = Element(0);
        if constexpr (NeedGrad) gradient = grad_out[item];
        Element result = Element(0);
        ((column < Paths::end
            ? (result = activation_backward_value<
                   Paths::kind, Paths::normalize, Differentiable>(
                   value, gradient, Paths::parameters()), true)
            : false) || ...);
        grad_x[item] = result;
    }
}

template<auto Width, bool Differentiable, bool NeedX, bool NeedGrad, class... Paths>
void run_activation_backward(
    const Element* x, const Element* grad_out, Element* grad_x,
    int64_t num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    const int64_t count = num_items * Width;
    const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
    activation_backward_kernel<Width, Differentiable, NeedX, NeedGrad, Paths...><<<blocks, 128, 0, stream>>>(
        x, grad_out, grad_x, num_items);
}'''

BACKWARD_FUNCTIONS = {
    'silu_backward': r'''template<class T>
__device__ __forceinline__ T activation_silu_backward(T grad_out, T x) {
    const T sigmoid = T(1) / (T(1) + ::exp(-x));
    return grad_out * sigmoid * (T(1) + x * (T(1) - sigmoid));
}''',
    'sigmoid_backward': r'''template<class T>
__device__ __forceinline__ T activation_sigmoid_backward(T grad_out, T y) {
    return grad_out * (T(1) - y) * y;
}''',
    'tanh_backward': r'''template<class T>
__device__ __forceinline__ T activation_tanh_backward(T grad_out, T y) {
    return grad_out * (T(1) - y * y);
}''',
    'softplus_backward': r'''template<class T>
__device__ __forceinline__ T activation_softplus_backward(T grad_out, T x, T beta, T threshold) {
    const T z = ::exp(x * beta);
    return x * beta > threshold ? grad_out : grad_out * z / (z + T(1));
}''',
    'gelu_backward_exact': r'''template<class T>
__device__ __forceinline__ T activation_gelu_backward_exact(T grad_out, T x) {
    constexpr T beta = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
    constexpr T alpha = M_SQRT1_2;
    const T cdf = T(0.5) * (T(1) + ::erf(x * alpha));
    const T pdf = ::exp(T(-0.5) * x * x) * beta;
    return grad_out * (cdf + x * pdf);
}''',
    'gelu_backward_tanh': r'''template<class T>
__device__ __forceinline__ T activation_gelu_backward_tanh(T grad_out, T x) {
    constexpr T beta = M_SQRT2 * M_2_SQRTPI * T(0.5);
    constexpr T kappa = 0.044715;
    const T square = x * x;
    const T cube = square * x;
    const T inner = beta * (x + kappa * cube);
    const T t = ::tanh(inner);
    const T left = T(0.5) * x;
    const T right = T(1) + t;
    const T left_derivative = T(0.5) * right;
    const T tanh_derivative = T(1) - t * t;
    const T inner_derivative = beta * (T(1) + T(3) * kappa * square);
    const T right_derivative = left * tanh_derivative * inner_derivative;
    return grad_out * (left_derivative + right_derivative);
}''',
}

BACKWARD_CHECKS = r'''
    TORCH_CHECK(grad_out.scalar_type()==x.scalar_type(),"Pointwise gradient dtype mismatch");
    TORCH_CHECK(grad_out.dim()==x.dim() && grad_out.size(-1)=={out_width},"Pointwise gradient shape mismatch");
    for(int64_t axis=0;axis<x.dim()-1;++axis) TORCH_CHECK(grad_out.size(axis)==x.size(axis),"Pointwise gradient batch mismatch");
'''


BACKWARD_CALL = r'''
    auto results=contiguous::launch(x.reshape({{num_items,int64_t({width})}}).contiguous(),
                  grad_out.reshape({{num_items,int64_t({out_width})}}).contiguous(),num_items);
'''


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


TENSOR_MODULE_BINDING = r'''    m.def("launch_backward",&backward_impl::launch_tensor);
    m.def("launch_backward_tuple", [](torch::Tensor x, torch::Tensor grad_out, int64_t num_items) {
        return pybind11::make_tuple(backward_impl::contiguous::launch(x, grad_out, num_items));
    });
}
'''


def required_inputs(paths):
    """Identity reads only grad_out; nonlinear activations also read x."""
    if not paths:
        return {"dx": None}
    dx = set()
    for width, spec in paths:
        if width:
            dx.add("grad_out")
            if spec["kind"] != "identity":
                dx.add("x")
    return {"dx": dx}


def make_source(activation_paths, in_dim, config, *, tensor_only=False, fragments=False):
    declarations, paths, _ = make_path_types(activation_paths, in_dim, config["dtype"])
    declarations += ["\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH]
    template_args = [str(in_dim)]
    template_args.append(str(config["grad_mode"]).lower())
    inputs = ('x', 'grad_out')
    template_args.extend(str(role in config["inputs"]).lower() for role in inputs)
    template_args.extend(paths)
    arguments = [f"{role}.data_ptr<Element>()" if role in config["inputs"] else "nullptr" for role in inputs]
    arguments.extend(
        f"result_{name}.data_ptr<Element>()" if name in config["active_outputs"] else "nullptr"
        for name in config["outputs"]
    )
    arguments.extend(("num_items", "stream"))
    launch = f"run_activation_backward<{','.join(template_args)}>({','.join(arguments)});"
    return make_binding(
        in_dim, config, declarations=declarations, kernel_body=BACKWARD_KERNEL, binding=BINDING,
        module_binding=MODULE_BINDING, launch=launch, tensor_only=tensor_only, fragments=fragments,
    )


def activation_backward(features, grad_output, paths, index_bits, kernel_cache, gate_layout, dtype, *, grad_mode=False):
    inputs = features, grad_output
    available = tuple(name for name, value in zip(("x", "grad_out"), inputs) if value is not None)
    config = get_kernel_config(
        paths, 1, dtype, kernel_cache, available=available, grad_mode=grad_mode, gate_layout=gate_layout
    )
    return launch_kernel(inputs, config, paths, index_bits, kernel_cache, make_source, gate_layout)[0]
