"""CUDA computations for Activation triple backward."""

from .utils import make_path_types, make_binding, get_kernel_config, launch_kernel
from .backward import BACKWARD_FUNCTIONS, BACKWARD_MATH
from .double_backward import DOUBLE_BACKWARD_RESULT, DOUBLE_BACKWARD_MATH


TRIPLE_BACKWARD_RESULT = r'''struct TripleBackwardResult {
    Element tx;
    Element tg;
    Element trx;
};'''


TRIPLE_BACKWARD_MATH = r'''template<ActivationKind Kind>
__device__ __forceinline__ Element activation_third_derivative(
    Element x, const ActivationParameters& parameters) {
    if constexpr (Kind == ActivationKind::Identity || Kind == ActivationKind::Relu ||
                  Kind == ActivationKind::Abs) {
        return Element(0);
    } else if constexpr (Kind == ActivationKind::Tanh) {
        const Element value = ::tanh(x);
        const Element square = value * value;
        return (Element(1) - square) * (Element(6) * square - Element(2));
    } else if constexpr (Kind == ActivationKind::Sigmoid) {
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
        const Element slope = (Element(1) - sigmoid) * sigmoid;
        const Element curvature_factor = Element(1) - Element(2) * sigmoid;
        return slope * (curvature_factor * curvature_factor - Element(2) * slope);
    } else if constexpr (Kind == ActivationKind::Silu) {
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-x));
        const Element slope = (Element(1) - sigmoid) * sigmoid;
        const Element curvature_factor = Element(1) - Element(2) * sigmoid;
        return slope * (Element(3) * curvature_factor +
                        x * (curvature_factor * curvature_factor - Element(2) * slope));
    } else if constexpr (Kind == ActivationKind::Softplus) {
        const Element inner = x * parameters.beta;
        const Element sigmoid = Element(1) / (Element(1) + ::exp(-inner));
        const Element mask = Element(inner < parameters.threshold);
        const Element curvature_factor = Element(1) + sigmoid * Element(-2);
        return (((Element(1) - sigmoid) * sigmoid) * curvature_factor) * mask *
               parameters.beta * parameters.beta;
    } else if constexpr (Kind == ActivationKind::Gelu) {
        const Element square = x * x;
        const Element density = ::exp(square * Element(-0.5)) *
                               Element(0.3989422804014327);
        return (x * density) * (square - Element(4));
    } else if constexpr (Kind == ActivationKind::GeluTanh) {
        const Element inner = (x + (x * x * x) * Element(0.044715)) *
                             Element(0.7978845608028654);
        const Element value = ::tanh(inner);
        const Element cosh_inner = ::cosh(inner);
        const Element sech = Element(1) / cosh_inner;
        const Element complement = Element(1) - value * value;
        const Element inner_slope = (Element(1) + (x * Element(0.134145)) * x) *
                                   Element(0.7978845608028654);
        const Element inner_curvature = (x * Element(0.26829)) *
                                       Element(0.7978845608028654);
        const Element inner_third = Element(0.26829) * Element(0.7978845608028654);
        const Element half_x = x * Element(0.5);
        const Element complement_slope = (Element(-2) * value * complement) * inner_slope;
        // Differentiate the registered cosh expression instead of replacing it
        // with a tanh identity; sinh keeps its overflow/NaN behavior intact.
        const Element sech_slope = ((-sech / cosh_inner) * ::sinh(inner)) * inner_slope;
        const Element tanh_curvature = ((sech * Element(2)) * (-sech * value)) *
                                      inner_slope;
        const Element tanh_third =
            (((Element(-4) * sech) * sech_slope) * value) * inner_slope -
            (((Element(2) * sech) * sech) * complement) * inner_slope * inner_slope -
            (((Element(2) * sech) * sech) * value) * inner_curvature;
        const Element first_term = complement_slope * inner_slope +
                                  complement * inner_curvature;
        const Element second_term =
            (Element(0.5) * inner_slope + half_x * inner_curvature) * tanh_curvature +
            (half_x * inner_slope) * tanh_third;
        const Element third_term =
            (Element(0.5) * complement + half_x * complement_slope) * inner_curvature +
            (half_x * complement) * inner_third;
        return (first_term + second_term) + third_term;
    }
}

template<ActivationKind Kind, bool Normalize,
         bool WriteTX, bool WriteTG, bool WriteTRX, bool HasBx, bool HasBg>
__device__ __forceinline__ TripleBackwardResult activation_triple_backward_value(
    Element x, Element grad_out, Element rx, Element bx, Element bg,
    const ActivationParameters& parameters) {
    TripleBackwardResult result{Element(0), Element(0), Element(0)};
    if constexpr (Kind == ActivationKind::Identity || Kind == ActivationKind::Relu ||
                  Kind == ActivationKind::Abs) {
        if constexpr (WriteTRX && HasBg) {
            Element normalized_bg = bg;
            if constexpr (Normalize) normalized_bg *= parameters.scale;
            result.trx = activation_backward_value<Kind, false, false>(
                x, normalized_bg, parameters);
        }
    } else {
        Element normalized_grad = grad_out;
        if constexpr (Normalize && HasBx && (WriteTX || WriteTRX)) {
            normalized_grad *= parameters.scale;
        }
        Element normalized_bg = bg;
        if constexpr (Normalize && HasBg && (WriteTX || WriteTRX)) {
            normalized_bg *= parameters.scale;
        }
        Element second_derivative = Element(0);
        if constexpr ((WriteTX && HasBg) || (WriteTG && HasBx) ||
                      (WriteTRX && HasBx)) {
            second_derivative = activation_second_derivative<Kind>(x, parameters);
        }
        if constexpr (WriteTX) {
            if constexpr (HasBx) {
                result.tx = (bx * (rx * normalized_grad)) *
                            activation_third_derivative<Kind>(x, parameters);
            }
            if constexpr (HasBg) {
                const Element bg_term = (normalized_bg * rx) * second_derivative;
                if constexpr (HasBx) result.tx += bg_term;
                else result.tx = bg_term;
            }
        }
        if constexpr (WriteTG && HasBx) {
            result.tg = (bx * second_derivative) * rx;
            if constexpr (Normalize) result.tg *= parameters.scale;
        }
        if constexpr (WriteTRX) {
            if constexpr (HasBx) result.trx = (bx * second_derivative) * normalized_grad;
            if constexpr (HasBg) {
                const Element bg_term = activation_backward_value<Kind, false, false>(
                    x, normalized_bg, parameters);
                if constexpr (HasBx) result.trx += bg_term;
                else result.trx = bg_term;
            }
        }
    }
    return result;
}'''


TRIPLE_BACKWARD_KERNEL = r'''template<auto Width, bool NeedX, bool NeedGrad, bool NeedRx,
                                    bool NeedBx, bool NeedBg, bool HasBx, bool HasBg,
                                    bool WriteTX, bool WriteTG, bool WriteTRX,
                                    class... Paths>
__global__ __launch_bounds__(128, 1)
void activation_triple_backward_kernel(
    const Element* __restrict__ x,
    const Element* __restrict__ grad_out,
    const Element* __restrict__ rx,
    const Element* __restrict__ bx,
    const Element* __restrict__ bg,
    Element* __restrict__ out_tx,
    Element* __restrict__ out_tg,
    Element* __restrict__ out_trx,
    int64_t num_items) {
    const int64_t count = num_items * Width;
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < count; item += Index(gridDim.x) * 128) {
        const Index column = item % Width;
        Element x_value = Element(0);
        Element grad_out_value = Element(0);
        Element rx_value = Element(0);
        Element bx_value = Element(0);
        Element bg_value = Element(0);
        if constexpr (NeedX) x_value = x[item];
        if constexpr (NeedGrad) grad_out_value = grad_out[item];
        if constexpr (NeedRx) rx_value = rx[item];
        if constexpr (NeedBx) bx_value = bx[item];
        if constexpr (NeedBg) bg_value = bg[item];
        TripleBackwardResult result{Element(0), Element(0), Element(0)};
        ((column < Paths::end
            ? (result = activation_triple_backward_value<
                   Paths::kind, Paths::normalize, WriteTX, WriteTG, WriteTRX, HasBx, HasBg>(
                   x_value, grad_out_value, rx_value, bx_value, bg_value, Paths::parameters()), true)
            : false) || ...);
        if constexpr (WriteTX) out_tx[item] = result.tx;
        if constexpr (WriteTG) out_tg[item] = result.tg;
        if constexpr (WriteTRX) out_trx[item] = result.trx;
    }
}

template<auto Width, bool NeedX, bool NeedGrad, bool NeedRx,
         bool NeedBx, bool NeedBg, bool HasBx, bool HasBg,
         bool WriteTX, bool WriteTG, bool WriteTRX, class... Paths>
void run_activation_triple_backward(
    const Element* x, const Element* grad_out, const Element* rx,
    const Element* bx, const Element* bg, Element* out_tx, Element* out_tg, Element* out_trx,
    int64_t num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    const int64_t count = num_items * Width;
    const unsigned blocks = std::min<int64_t>(65535, (count + 127) / 128);
    activation_triple_backward_kernel<Width, NeedX, NeedGrad, NeedRx, NeedBx, NeedBg, HasBx, HasBg, WriteTX, WriteTG, WriteTRX, Paths...><<<blocks, 128, 0, stream>>>(
        x, grad_out, rx, bx, bg, out_tx, out_tg, out_trx, num_items);
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
    """Inputs for gradients with respect to x, grad_out, and rx, respectively."""
    if "bx" not in available and "bg" not in available:
        return {}
    tx = tg = trx = None
    for width, spec in paths:
        kind = spec["kind"]
        if kind != "identity" and (kind != "abs" or "bg" in available):
            if tx is None:
                tx = set()
            if width:
                if kind == "relu":
                    tx.add("x")
                elif kind != "abs":
                    tx.update(("x", "rx"))
                    if "bx" in available:
                        tx.update(("grad_out", "bx"))
                    if "bg" in available:
                        tx.add("bg")
        if kind not in ("identity", "relu", "abs") and "bx" in available:
            if tg is None:
                tg = set()
            if width:
                tg.update(("x", "rx", "bx"))
        if kind in ("identity", "relu", "abs"):
            if "bg" not in available:
                continue
            if trx is None:
                trx = set()
            if width:
                trx.add("bg")
                if kind != "identity":
                    trx.add("x")
        else:
            if trx is None:
                trx = set()
            if width:
                trx.add("x")
                if "bx" in available:
                    trx.update(("grad_out", "bx"))
                if "bg" in available:
                    trx.add("bg")
    return {"tx": tx, "tg": tg, "trx": trx}


def make_source(activation_paths, in_dim, config, *, tensor_only=False, fragments=False):
    declarations, paths, _ = make_path_types(activation_paths, in_dim, config["dtype"])
    declarations += ["\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH]
    declarations += [DOUBLE_BACKWARD_RESULT, DOUBLE_BACKWARD_MATH]
    declarations += [TRIPLE_BACKWARD_RESULT, TRIPLE_BACKWARD_MATH]
    template_args = [str(in_dim)]
    inputs = ('x', 'grad_out', 'rx', 'bx', 'bg')
    template_args.extend(str(role in config["inputs"]).lower() for role in inputs)
    template_args.extend(str(role in config["available"]).lower() for role in ("bx", "bg"))
    template_args.extend(str(name in config["active_outputs"]).lower() for name in config["outputs"])
    template_args.extend(paths)
    arguments = [f"{role}.data_ptr<Element>()" if role in config["inputs"] else "nullptr" for role in inputs]
    arguments.extend(
        f"result_{name}.data_ptr<Element>()" if name in config["active_outputs"] else "nullptr"
        for name in config["outputs"]
    )
    arguments.extend(("num_items", "stream"))
    launch = f"run_activation_triple_backward<{','.join(template_args)}>({','.join(arguments)});"
    return make_binding(
        in_dim, config, declarations=declarations, kernel_body=TRIPLE_BACKWARD_KERNEL, binding=BINDING,
        module_binding=MODULE_BINDING, launch=launch, tensor_only=tensor_only, fragments=fragments,
    )


def activation_triple_backward(features, grad_output, grad_grad_input, grad_input, grad_grad_output, paths, index_bits, kernel_cache, gate_layout, dtype, *, needs=None):
    inputs = features, grad_output, grad_grad_input, grad_input, grad_grad_output
    available = tuple(name for name, value in zip(("x", "grad_out", "rx", "bx", "bg"), inputs) if value is not None)
    config = get_kernel_config(
        paths, 3, dtype, kernel_cache, needs=needs, available=available, gate_layout=gate_layout
    )
    return launch_kernel(inputs, config, paths, index_bits, kernel_cache, make_source, gate_layout)
