"""Activation forward CUDA math, kernels, and launches."""

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension

from .utils import (
    BINDING, MODULE_BINDING, TENSOR_BINDING, TRAINING_MODULE_BINDING,
    make_path_types, make_binding, get_kernel_config, launch_kernel,
)


FORWARD_MATH = r'''template<ActivationKind Kind, bool Normalize>
__device__ __forceinline__ Element activation_forward_value(
    Element x, const ActivationParameters& parameters) {
    Element value;
    if constexpr (Kind == ActivationKind::Identity) {
        value = x;
    } else if constexpr (Kind == ActivationKind::Tanh) {
        value = ::tanh(x);
    } else if constexpr (Kind == ActivationKind::Sigmoid) {
        value = activation_sigmoid(x);
    } else if constexpr (Kind == ActivationKind::Silu) {
        value = activation_silu(x);
    } else if constexpr (Kind == ActivationKind::Softplus) {
        value = activation_softplus(x, parameters.beta, parameters.threshold);
    } else if constexpr (Kind == ActivationKind::Relu) {
        value = activation_relu(x);
    } else if constexpr (Kind == ActivationKind::Abs) {
        value = ::fabs(x);
    } else if constexpr (Kind == ActivationKind::Gelu) {
        value = activation_gelu_exact(x);
    } else if constexpr (Kind == ActivationKind::GeluTanh) {
        value = activation_gelu_tanh(x);
    }
    if constexpr (Normalize) value *= parameters.scale;
    return value;
}'''


FORWARD_KERNEL = r'''template<auto Width, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_forward_kernel(
    const Element* __restrict__ x,
    Element* __restrict__ out,
    int64_t num_items) {
    const int64_t count = num_items * Width;
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = item % Width;
        const Element value = x[item];
        Element result = Element(0);
        // Paths cover successive feature ranges; select the first matching end.
        ((column < Paths::end
            ? (result = activation_forward_value<Paths::kind, Paths::normalize>(
                   value, Paths::parameters()), true)
            : false) || ...);
        out[item] = result;
    }
}

template<auto Width, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_forward_vectorized_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t num_items) {
    const int64_t count = num_items * (Width / 4);
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = (item * 4) % Width;
        const float4 values = reinterpret_cast<const float4*>(x)[item];
        float4 result;
        ((column < Paths::end
            ? (result = make_float4(
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.x, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.y, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.z, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.w, Paths::parameters())), true)
            : false) || ...);
        reinterpret_cast<float4*>(out)[item] = result;
    }
}

template<auto Width, bool Vectorized, class... Paths>
void run_activation_forward(
    const Element* x, Element* out, int64_t num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    if constexpr (Vectorized) {
        if (num_items * Width >= VectorMinElements &&
            reinterpret_cast<uintptr_t>(x) % 16 == 0) {
            const int64_t count = num_items * (Width / 4);
            const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
            activation_forward_vectorized_kernel<Width, Paths...><<<blocks, 128, 0, stream>>>(
                x, out, num_items);
            return;
        }
    }
    const int64_t count = num_items * Width;
    const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
    activation_forward_kernel<Width, Paths...><<<blocks, 128, 0, stream>>>(x, out, num_items);
}'''


STRIDED_FORWARD_KERNEL = r'''template<auto Width, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_forward_strided_kernel(
    const Element* __restrict__ x,
    Element* __restrict__ out,
    int64_t num_items,
    ForwardLayout layout) {
    const int64_t count = num_items * Width;
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = item % Width;
        const Index row = item / Width;
        const int64_t offset = forward_row_offset(row, layout);
        const Element value = x[offset + column * layout.feature_stride];
        Element result = Element(0);
        ((column < Paths::end
            ? (result = activation_forward_value<Paths::kind, Paths::normalize>(
                   value, Paths::parameters()), true)
            : false) || ...);
        out[item] = result;
    }
}

template<auto Width, class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_forward_strided_vectorized_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t num_items,
    ForwardLayout layout) {
    const int64_t count = num_items * (Width / 4);
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = (item * 4) % Width;
        const Index row = (item * 4) / Width;
        const int64_t offset = forward_row_offset(row, layout)
            + column * layout.feature_stride;
        const float4 values = make_float4(
            x[offset], x[offset + layout.feature_stride],
            x[offset + 2 * layout.feature_stride], x[offset + 3 * layout.feature_stride]);
        float4 result;
        ((column < Paths::end
            ? (result = make_float4(
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.x, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.y, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.z, Paths::parameters()),
                   activation_forward_value<Paths::kind, Paths::normalize>(
                       values.w, Paths::parameters())), true)
            : false) || ...);
        reinterpret_cast<float4*>(out)[item] = result;
    }
}

template<auto Width, bool Vectorized, class... Paths>
void run_activation_forward_strided(
    const Element* x, Element* out, int64_t num_items,
    ForwardLayout layout, cudaStream_t stream) {
    if (num_items == 0) return;
    if constexpr (Vectorized) {
        if (num_items >= 16384 && reinterpret_cast<uintptr_t>(x) % 16 == 0) {
            const int64_t count = num_items * (Width / 4);
            const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
            activation_forward_strided_vectorized_kernel<Width, Paths...><<<blocks, 128, 0, stream>>>(
                x, out, num_items, layout);
            return;
        }
    }
    const int64_t count = num_items * Width;
    const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
    activation_forward_strided_kernel<Width, Paths...><<<blocks, 128, 0, stream>>>(
        x, out, num_items, layout);
}'''

FORWARD_FUNCTIONS = {
    'silu': r'''template<class T>
__device__ __forceinline__ T activation_silu(T x) {
    return x / (T(1) + ::exp(-x));
}''',
    'sigmoid': r'''template<class T>
__device__ __forceinline__ T activation_sigmoid(T x) {
    return T(1) / (T(1) + ::exp(-x));
}''',
    'softplus': r'''template<class T>
__device__ __forceinline__ T activation_softplus(T x, T beta, T threshold) {
    return x * beta > threshold ? x : ::log1p(::exp(x * beta)) / beta;
}''',
    'relu': r'''template<class T>
__device__ __forceinline__ T activation_relu(T x) {
    return x != x ? x : ::max(x, T(0));
}''',
    'gelu_exact': r'''template<class T>
__device__ __forceinline__ T activation_gelu_exact(T x) {
    constexpr T alpha = M_SQRT1_2;
    return x * T(0.5) * (T(1) + ::erf(x * alpha));
}''',
    'gelu_tanh': r'''template<class T>
__device__ __forceinline__ T activation_gelu_tanh(T x) {
    constexpr T beta = M_SQRT2 * M_2_SQRTPI * T(0.5);
    constexpr T kappa = 0.044715;
    const T cube = x * x * x;
    const T inner = beta * (x + kappa * cube);
    return T(0.5) * x * (T(1) + ::tanh(inner));
}''',
}

TANH_APPROX = r'''__device__ __forceinline__ float activation_tanh_approx(float x) {
    float result;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}'''


FORWARD_CALL = r'''
    {results_type} results;
    if(x.is_contiguous() || x.dim()>64) {{
        results=contiguous::launch(x.reshape({{num_items,int64_t({width})}}).contiguous(),num_items);
    }} else {{ results=strided::launch(x,num_items); }}
'''


ACTIVATION_TYPES = r'''enum class ActivationKind {
    Identity, Tanh, Sigmoid, Silu, Softplus, Relu, Abs, Gelu, GeluTanh
};

struct ActivationParameters {
    Element scale;
    Element beta;
    Element threshold;
};'''


def make_source(activation_paths, in_dim, config, *, strided=False, tensor_only=False, fragments=False):
    declarations, paths, vectorized = make_path_types(activation_paths, in_dim, config["dtype"])
    template_args = [str(in_dim), str(vectorized).lower(), *paths]
    function = "run_activation_forward_strided" if strided else "run_activation_forward"
    arguments = ["x.data_ptr<Element>()", "result_out.data_ptr<Element>()", "num_items"]
    if strided:
        arguments.append("x_layout")
    arguments.append("stream")
    launch = f"{function}<{','.join(template_args)}>({','.join(arguments)});"
    return make_binding(
        in_dim, config, declarations=declarations,
        kernel_body=STRIDED_FORWARD_KERNEL if strided else FORWARD_KERNEL,
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        strided=strided, tensor_only=tensor_only, fragments=fragments,
    )


def make_forward_backward_source(activation_paths, in_dim, forward_config, backward_config):
    from . import backward

    headers = []
    namespaces = []
    width = in_dim
    for config in (forward_config, backward_config):
        order = config["order"]
        includes, body = (backward.make_source if order else make_source)(
            activation_paths, in_dim, config, tensor_only=True, fragments=True
        )
        headers.extend(includes)
        native = "namespace contiguous {\n" + body + "\n}\n"
        if order == 0:
            includes, body = make_source(activation_paths, in_dim, config, strided=True, tensor_only=True, fragments=True)
            headers.extend(includes)
            native += "namespace strided {\n" + body + "\n}\n"
        call = (
            backward.BACKWARD_CALL.format(width=width, out_width=width)
            if order else FORWARD_CALL.format(results_type="torch::Tensor", width=width)
        )
        wrapper = TENSOR_BINDING.format(
            args="torch::Tensor x" + (",torch::Tensor grad_out" if order else ""),
            scalar="Double" if config["dtype"] == torch.float64 else "Float",
            width=width, max_width=width,
            checks=backward.BACKWARD_CHECKS.format(out_width=width) if order else "",
            call=call, result_value="results", result_width=width,
        )
        namespaces.append("namespace " + ("backward_impl" if order else "forward_impl") + " {\n" + native + wrapper + "\n}\n")
    headers += ["#include <limits>", "#include <vector>"]
    return "\n".join(dict.fromkeys(headers)) + "\n" + "\n".join(namespaces) + TRAINING_MODULE_BINDING


def activation_forward(features, activation_paths, index_bits, kernel_cache):
    """Load forward and both first-backward modes in one extension."""
    dtype = features.dtype
    device = features.device
    numel = features.numel()
    bound = numel + 65535 * 128
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
        in_dim = sum(width for width, _ in activation_paths)
        forward_config = get_kernel_config(activation_paths, 0, dtype, kernel_cache)
        backward_config = get_kernel_config(
            activation_paths, 1, dtype, kernel_cache, available=("x", "grad_out")
        )
        differentiable_config = get_kernel_config(
            activation_paths, 1, dtype, kernel_cache, grad_mode=True
        )

        differentiable = bool(differentiable_config["active_outputs"])
        forward_kernel = backward_kernel = None
        if (
            in_dim and bound <= (1 << 63) - 1
            and forward_config["inputs"] == ("x",)
            and forward_config["active_outputs"] == ("out",)
            and backward_config["inputs"] in (("x", "grad_out"), ("grad_out",))
            and backward_config["active_outputs"] == ("dx",)
        ):
            sources = {"differentiable_backward": backward.make_source(
                activation_paths, in_dim, differentiable_config
            )}
            if backward_config["inputs"] == ("x", "grad_out"):
                sources["training"] = make_forward_backward_source(
                    activation_paths, in_dim, forward_config, backward_config
                )
            else:
                sources["forward"] = make_source(activation_paths, in_dim, forward_config)
                sources["strided_forward"] = make_source(activation_paths, in_dim, forward_config, strided=True)
                sources["backward"] = backward.make_source(activation_paths, in_dim, backward_config)
            extension = load_cuda_extension(
                "lunaeq_activation_forward_backward", combine_cuda_sources(sources),
                index_bits=bits, arch=arch, use_fast_math=False, minimal_headers=True,
            )
            kernel_cache[("forward_backward_module", dtype, device, bits)] = extension
            if "training" in sources:
                forward_kernel = extension.training.launch_forward
                backward_kernel = extension.training.launch_backward

        save_input = any(spec["kind"] != "identity" for _, spec in activation_paths)
        save_input |= backward_kernel is not None
        entry = forward_kernel, backward_kernel, differentiable, save_input, forward_config
        kernel_cache[key] = entry
        while len(kernel_cache) > 512:
            kernel_cache.popitem(last=False)

    forward_kernel, backward_kernel, differentiable, save_input, config = entry
    if forward_kernel is not None:
        result = forward_kernel(features)
    else:
        result = launch_kernel(
            (features,), config, activation_paths, index_bits, kernel_cache, make_source
        )[0].detach()
    return result, backward_kernel, differentiable, save_input
