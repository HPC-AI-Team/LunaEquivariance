"""Gate CUDA third gradients, with independent input/output gradients."""

from ...activation.kernels.backward import BACKWARD_FUNCTIONS, BACKWARD_MATH
from ...activation.kernels.double_backward import DOUBLE_BACKWARD_MATH, DOUBLE_BACKWARD_RESULT
from ...activation.kernels.triple_backward import (
    BINDING, MODULE_BINDING, TRIPLE_BACKWARD_MATH, TRIPLE_BACKWARD_RESULT,
    required_inputs as activation_required_inputs,
)
from .utils import make_binding, make_path_types, make_schedule

TRIPLE_BACKWARD_MATH_SOURCE = r'''struct GateTripleValues {
    Element tx;
    Element trx;
    Element curvature;
    Element input_slope;
    Element direction_slope;
};

__device__ __forceinline__ float gate_subtract(float a, float b) {
    return __fsub_rn(a, b);
}
__device__ __forceinline__ double gate_subtract(double a, double b) {
    return __dsub_rn(a, b);
}

template<class Path>
__device__ __forceinline__ GateTripleValues gate_triple_product_values(
    Element value, Element moment, Element direction_moment, Element mixed_moment,
    Element output_moment, Element direction, Element input_seed,
    const ActivationParameters& parameters) {
    GateTripleValues result{Element(0), Element(0), Element(0), Element(0), Element(0)};
    const Element scale = Path::normalize ? parameters.scale : Element(1);
    const Element normalized_moment = gate_multiply(moment, scale);
    const Element normalized_mixed = gate_multiply(mixed_moment, scale);
    Element normalized_direction = Element(0), normalized_output = Element(0);
    if constexpr (HasBx) normalized_direction = gate_multiply(direction_moment, scale);
    if constexpr (HasBg) normalized_output = gate_multiply(output_moment, scale);

    if constexpr (Path::kind == ActivationKind::Tanh) {
        const Element activation = ::tanh(value);
        const Element factor = gate_multiply(activation, Element(-2));
        const Element raw_direction = activation_tanh_backward(direction, activation);
        result.direction_slope = gate_multiply(raw_direction, scale);
        Element raw_seed = Element(0), seed_direction = Element(0);
        if constexpr (HasBx) {
            raw_seed = activation_tanh_backward(input_seed, activation);
            seed_direction = gate_multiply(raw_seed, direction);
            result.input_slope = gate_multiply(raw_seed, scale);
            result.curvature = gate_multiply(gate_multiply(seed_direction, factor), scale);
        }
        if constexpr (WriteTX) {
            Element accumulator = Element(0);
            if constexpr (HasBx) {
                const Element moment_factor = gate_multiply(factor, normalized_moment);
                const Element inner = gate_add(gate_multiply(direction, moment_factor), normalized_direction);
                const Element seed_term = gate_multiply(input_seed, gate_multiply(factor, inner));
                const Element slope_term = gate_multiply(gate_multiply(seed_direction, normalized_moment), Element(-2));
                accumulator = gate_add(seed_term, slope_term);
            }
            const Element mixed_term = gate_multiply(normalized_mixed, gate_multiply(factor, direction));
            accumulator = HasBx ? gate_add(accumulator, mixed_term) : mixed_term;
            if constexpr (HasBg) accumulator = gate_add(accumulator, normalized_output);
            result.tx = gate_add(Element(0), activation_tanh_backward(accumulator, activation));
        }
        if constexpr (WriteTRX) {
            const Element mixed_term = activation_tanh_backward(normalized_mixed, activation);
            if constexpr (HasBx) {
                const Element seed_term = gate_multiply(raw_seed, gate_multiply(factor, normalized_moment));
                result.trx = gate_add(Element(0), gate_add(seed_term, mixed_term));
            } else {
                result.trx = gate_add(Element(0), mixed_term);
            }
        }
    } else if constexpr (Path::kind == ActivationKind::Sigmoid) {
        const Element activation = activation_sigmoid(value);
        const Element factor = gate_add(gate_multiply(activation, Element(-2)), Element(1));
        const Element raw_direction = activation_sigmoid_backward(direction, activation);
        result.direction_slope = gate_multiply(raw_direction, scale);
        Element raw_seed = Element(0), seed_factor = Element(0);
        if constexpr (HasBx) {
            raw_seed = activation_sigmoid_backward(input_seed, activation);
            seed_factor = gate_multiply(raw_seed, factor);
            result.input_slope = gate_multiply(raw_seed, scale);
            result.curvature = gate_multiply(gate_multiply(seed_factor, direction), scale);
        }
        if constexpr (WriteTX) {
            Element accumulator = Element(0);
            if constexpr (HasBx) {
                const Element weighted_direction = gate_multiply(direction, normalized_moment);
                const Element inner = gate_add(gate_multiply(weighted_direction, factor), normalized_direction);
                const Element seed_term = gate_multiply(gate_multiply(input_seed, inner), factor);
                const Element slope_term = gate_multiply(gate_multiply(raw_seed, weighted_direction), Element(-2));
                accumulator = gate_add(seed_term, slope_term);
            }
            const Element mixed_term = gate_multiply(gate_multiply(normalized_mixed, direction), factor);
            accumulator = HasBx ? gate_add(accumulator, mixed_term) : mixed_term;
            if constexpr (HasBg) accumulator = gate_add(accumulator, normalized_output);
            result.tx = gate_add(Element(0), activation_sigmoid_backward(accumulator, activation));
        }
        if constexpr (WriteTRX) {
            const Element mixed_term = activation_sigmoid_backward(normalized_mixed, activation);
            if constexpr (HasBx) {
                result.trx = gate_add(Element(0), gate_add(gate_multiply(seed_factor, normalized_moment), mixed_term));
            } else {
                result.trx = gate_add(Element(0), mixed_term);
            }
        }
    } else if constexpr (Path::kind == ActivationKind::Silu) {
        const Element sigmoid = activation_sigmoid(value);
        const Element complement = gate_subtract(Element(1), sigmoid);
        const Element slope = gate_add(gate_multiply(value, complement), Element(1));
        const Element direction_slope = gate_multiply(direction, slope);
        result.direction_slope = gate_multiply(gate_multiply(direction_slope, sigmoid), scale);
        Element sigmoid_seed = Element(0), slope_seed = Element(0);
        if constexpr (HasBx) {
            sigmoid_seed = activation_sigmoid_backward(input_seed, sigmoid);
            slope_seed = gate_add(gate_multiply(input_seed, complement), gate_multiply(-sigmoid_seed, value));
            result.input_slope = gate_multiply(gate_multiply(gate_multiply(input_seed, slope), sigmoid), scale);
            const Element first = gate_multiply(sigmoid_seed, direction_slope);
            const Element second = gate_multiply(gate_multiply(slope_seed, direction), sigmoid);
            result.curvature = gate_multiply(gate_add(first, second), scale);
        }
        const Element moment_sigmoid = gate_multiply(normalized_moment, sigmoid);
        Element mixed_sigmoid = gate_multiply(normalized_mixed, sigmoid);
        if constexpr (HasBx) mixed_sigmoid = gate_add(mixed_sigmoid, gate_multiply(sigmoid_seed, normalized_moment));
        if constexpr (WriteTX) {
            const Element weighted_direction = gate_multiply(direction, moment_sigmoid);
            Element accumulator = Element(0);
            if constexpr (HasBx) {
                const Element weighted_seed = gate_multiply(input_seed, gate_multiply(normalized_direction, sigmoid));
                const Element first = gate_multiply(weighted_seed, complement);
                const Element sigmoid_term = gate_add(
                    gate_multiply(gate_multiply(input_seed, slope), normalized_direction),
                    -gate_multiply(weighted_seed, value));
                accumulator = gate_add(first, activation_sigmoid_backward(sigmoid_term, sigmoid));
                accumulator = gate_add(accumulator, gate_multiply(-sigmoid_seed, weighted_direction));
            }
            const Element mixed_direction = gate_multiply(mixed_sigmoid, direction);
            const Element mixed_term = gate_multiply(mixed_direction, complement);
            accumulator = HasBx ? gate_add(accumulator, mixed_term) : mixed_term;
            Element sigmoid_gradient = Element(0);
            if constexpr (HasBx) {
                const Element inner = gate_add(gate_multiply(direction_slope, normalized_moment),
                                              -gate_multiply(weighted_direction, value));
                const Element seed_inner = gate_multiply(input_seed, inner);
                const Element factor = gate_add(gate_multiply(sigmoid, Element(-2)), Element(1));
                sigmoid_gradient = gate_multiply(seed_inner, factor);
            }
            const Element mixed_slope = gate_multiply(normalized_mixed, direction_slope);
            sigmoid_gradient = HasBx ? gate_add(sigmoid_gradient, mixed_slope) : mixed_slope;
            if constexpr (HasBx) {
                sigmoid_gradient = gate_add(sigmoid_gradient,
                    gate_multiply(gate_multiply(slope_seed, direction), normalized_moment));
                const Element negative = -gate_add(gate_multiply(input_seed, weighted_direction),
                                                  gate_multiply(mixed_direction, value));
                sigmoid_gradient = gate_add(sigmoid_gradient, negative);
            } else {
                sigmoid_gradient = gate_add(sigmoid_gradient, -gate_multiply(mixed_direction, value));
            }
            accumulator = gate_add(accumulator, activation_sigmoid_backward(sigmoid_gradient, sigmoid));
            if constexpr (HasBg) accumulator = gate_add(accumulator, activation_silu_backward(normalized_output, value));
            result.tx = gate_add(Element(0), accumulator);
        }
        if constexpr (WriteTRX) {
            const Element first = gate_multiply(mixed_sigmoid, slope);
            if constexpr (HasBx) result.trx = gate_add(Element(0), gate_add(first, gate_multiply(slope_seed, moment_sigmoid)));
            else result.trx = gate_add(Element(0), first);
        }
    } else if constexpr (Path::kind == ActivationKind::Softplus) {
        const Element inner = gate_multiply(value, parameters.beta);
        const Element mask = Element(inner < parameters.threshold);
        const Element sigmoid = activation_sigmoid(inner);
        result.direction_slope = gate_multiply(
            activation_softplus_backward(direction, value, parameters.beta, parameters.threshold), scale);
        Element masked_seed = Element(0), sigmoid_seed = Element(0);
        if constexpr (HasBx) {
            masked_seed = gate_multiply(gate_multiply(input_seed, parameters.beta), mask);
            sigmoid_seed = activation_sigmoid_backward(masked_seed, sigmoid);
            result.curvature = gate_multiply(gate_multiply(sigmoid_seed, direction), scale);
            result.input_slope = gate_multiply(
                activation_softplus_backward(input_seed, value, parameters.beta, parameters.threshold), scale);
        }
        if constexpr (WriteTX) {
            Element accumulator = Element(0);
            if constexpr (HasBx) {
                const Element input_term = gate_multiply(gate_multiply(
                    activation_sigmoid_backward(gate_multiply(input_seed, normalized_direction), sigmoid), mask), parameters.beta);
                const Element weighted_direction = gate_multiply(direction, normalized_moment);
                const Element seed_direction = gate_multiply(masked_seed, weighted_direction);
                const Element factor = gate_add(gate_multiply(sigmoid, Element(-2)), Element(1));
                const Element curvature_term = gate_multiply(
                    activation_sigmoid_backward(gate_multiply(seed_direction, factor), sigmoid), parameters.beta);
                accumulator = gate_add(input_term, curvature_term);
            }
            const Element mixed_term = gate_multiply(gate_multiply(
                activation_sigmoid_backward(gate_multiply(normalized_mixed, direction), sigmoid), mask), parameters.beta);
            accumulator = HasBx ? gate_add(accumulator, mixed_term) : mixed_term;
            if constexpr (HasBg) accumulator = gate_add(accumulator,
                activation_softplus_backward(normalized_output, value, parameters.beta, parameters.threshold));
            result.tx = gate_add(Element(0), accumulator);
        }
        if constexpr (WriteTRX) {
            const Element mixed_term = activation_softplus_backward(normalized_mixed, value, parameters.beta, parameters.threshold);
            if constexpr (HasBx) result.trx = gate_add(Element(0), gate_add(gate_multiply(sigmoid_seed, normalized_moment), mixed_term));
            else result.trx = gate_add(Element(0), mixed_term);
        }
    } else if constexpr (Path::kind == ActivationKind::Gelu || Path::kind == ActivationKind::GeluTanh) {
        const Element second_derivative = activation_second_derivative<Path::kind>(value, parameters);
        result.direction_slope = gate_multiply(
            activation_backward_value<Path::kind, false, false>(value, direction, parameters), scale);
        if constexpr (HasBx) {
            result.input_slope = gate_multiply(
                activation_backward_value<Path::kind, false, false>(value, input_seed, parameters), scale);
            result.curvature = gate_multiply(gate_multiply(gate_multiply(input_seed, second_derivative), direction), scale);
        }
        if constexpr (WriteTX) {
            Element accumulator = Element(0);
            if constexpr (HasBx) {
                accumulator = gate_multiply(gate_multiply(input_seed, normalized_direction), second_derivative);
                const Element weighted_seed = gate_multiply(input_seed, gate_multiply(direction, normalized_moment));
                if constexpr (Path::kind == ActivationKind::Gelu) {
                    const Element square = gate_multiply(value, value);
                    const Element exponential = ::exp(gate_multiply(square, Element(-0.5)));
                    const Element density = gate_multiply(exponential, Element(0.3989422804014327));
                    const Element density_term = gate_multiply(-weighted_seed, density);
                    const Element polynomial = gate_add(gate_multiply(weighted_seed, Element(2)),
                                                       gate_multiply(-weighted_seed, square));
                    Element exponential_term = gate_multiply(polynomial, Element(0.3989422804014327));
                    exponential_term = gate_multiply(exponential_term, exponential);
                    exponential_term = gate_multiply(exponential_term, Element(-0.5));
                    const Element density_gradient = gate_multiply(gate_add(density_term, exponential_term), value);
                    accumulator = gate_add(accumulator, density_gradient);
                    accumulator = gate_add(accumulator, density_gradient);
                } else {
                    const Element beta = Element(0.7978845608028654);
                    const Element square = gate_multiply(value, value);
                    const Element cubic = gate_multiply(square, value);
                    const Element inner = gate_multiply(gate_add(value, gate_multiply(cubic, Element(0.044715))), beta);
                    const Element activation = ::tanh(inner);
                    const Element complement = gate_subtract(Element(1), gate_multiply(activation, activation));
                    const Element half = gate_multiply(value, Element(0.5));
                    const Element slope = gate_multiply(gate_add(Element(1),
                        gate_multiply(gate_multiply(value, Element(0.134145)), value)), beta);
                    const Element curvature = gate_multiply(gate_multiply(value, Element(0.26829)), beta);
                    const Element cosh_inner = ::cosh(inner);
                    const Element sech = Element(1) / cosh_inner;
                    const Element tanh_curvature = gate_multiply(gate_multiply(sech, Element(2)), gate_multiply(-sech, activation));
                    const Element tangent_curvature = gate_multiply(tanh_curvature, slope);
                    Element slope_gradient = gate_multiply(weighted_seed, tangent_curvature);
                    Element tangent_gradient = gate_multiply(weighted_seed, gate_multiply(half, slope));
                    const Element input_curvature = gate_multiply(gate_multiply(
                        gate_multiply(weighted_seed, gate_multiply(half, complement)), beta), Element(0.26829));
                    accumulator = gate_add(accumulator, input_curvature);
                    const Element polynomial_gradient = gate_multiply(gate_add(
                        gate_add(gate_multiply(slope_gradient, half), gate_multiply(tangent_gradient, tanh_curvature)),
                        gate_multiply(gate_multiply(weighted_seed, Element(2)), gate_multiply(complement, Element(0.5)))), beta);
                    accumulator = gate_add(accumulator, gate_multiply(polynomial_gradient, gate_multiply(value, Element(0.134145))));
                    accumulator = gate_add(accumulator, gate_multiply(gate_multiply(polynomial_gradient, value), Element(0.134145)));
                    const Element weighted_curvature = gate_multiply(weighted_seed, curvature);
                    const Element half_gradient = gate_multiply(gate_add(gate_multiply(weighted_curvature, complement),
                        gate_multiply(slope_gradient, slope)), Element(0.5));
                    accumulator = gate_add(accumulator, half_gradient);
                    const Element tangent_weight = gate_multiply(tangent_gradient, slope);
                    const Element sech_weight = gate_multiply(tangent_weight, gate_multiply(sech, Element(2)));
                    const Element sech_gradient = gate_add(
                        gate_multiply(gate_multiply(tangent_weight, gate_multiply(-sech, activation)), Element(2)),
                        -gate_multiply(sech_weight, activation));
                    const Element cosh_gradient = gate_multiply(gate_multiply(-sech_gradient, sech / cosh_inner), ::sinh(inner));
                    Element activation_gradient = gate_multiply(sech_weight, -sech);
                    const Element complement_gradient = -gate_add(gate_multiply(weighted_curvature, half),
                        gate_multiply(gate_multiply(gate_multiply(weighted_seed, Element(2)), slope), Element(0.5)));
                    const Element complement_term = gate_multiply(complement_gradient, activation);
                    activation_gradient = gate_add(activation_gradient, complement_term);
                    activation_gradient = gate_add(activation_gradient, complement_term);
                    const Element inner_gradient = gate_multiply(gate_add(cosh_gradient,
                        activation_tanh_backward(activation_gradient, activation)), beta);
                    accumulator = gate_add(accumulator, inner_gradient);
                    accumulator = gate_add(accumulator, gate_multiply(gate_multiply(inner_gradient, Element(0.044715)),
                        gate_multiply(square, Element(3))));
                }
            }
            const Element mixed_term = gate_multiply(gate_multiply(normalized_mixed, direction), second_derivative);
            accumulator = HasBx ? gate_add(accumulator, mixed_term) : mixed_term;
            if constexpr (HasBg) accumulator = gate_add(accumulator,
                activation_backward_value<Path::kind, false, false>(value, normalized_output, parameters));
            result.tx = gate_add(Element(0), accumulator);
        }
        if constexpr (WriteTRX) {
            const Element mixed_term = activation_backward_value<Path::kind, false, false>(value, normalized_mixed, parameters);
            if constexpr (HasBx) {
                const Element seed_term = gate_multiply(gate_multiply(input_seed, second_derivative), normalized_moment);
                result.trx = gate_add(Element(0), gate_add(seed_term, mixed_term));
            } else result.trx = gate_add(Element(0), mixed_term);
        }
    } else {
        const auto ordinary = activation_triple_backward_value<
            Path::kind, Path::normalize, WriteTX, WriteTX || WriteTG, WriteTRX, HasBx, HasBx || HasBg>(
            value, moment, direction, input_seed, mixed_moment, parameters);
        DoubleBackwardResult input_mixed{Element(0), Element(0)};
        if constexpr (HasBx) input_mixed = activation_double_backward_value<
            Path::kind, Path::normalize, WriteTX, WriteTG || WriteTRX>(
            value, direction_moment, input_seed, parameters);
        const auto direction_result = activation_double_backward_value<
            Path::kind, Path::normalize, false, true>(value, moment, direction, parameters);
        result.curvature = ordinary.tg;
        result.input_slope = input_mixed.sg;
        result.direction_slope = direction_result.sg;
        result.trx = ordinary.trx;
        if constexpr (WriteTX) {
            Element output_mixed = Element(0);
            if constexpr (HasBg) output_mixed = activation_backward_value<Path::kind, Path::normalize, true>(
                value, output_moment, parameters);
            result.tx = gate_add(gate_add(ordinary.tx, input_mixed.sx), output_mixed);
        }
    }
    return result;
}

template<class Path>
__device__ __forceinline__ void gate_triple_backward_scalar(
    const Element* x, const Element* grad_out, const Element* rx,
    const Element* bx, const Element* bg, Element* out_tx,
    Element* out_tg, Element* out_trx, Index row, Index copy) {
    const Index input = row * InWidth + Path::input_offset + copy;
    const Index output = row * OutWidth + Path::output_offset + copy;
    Element value = Element(0), grad = Element(0), direction = Element(0);
    Element input_seed = Element(0), output_seed = Element(0);
    if constexpr (NeedX) value = x[input];
    if constexpr (NeedGrad) grad = grad_out[output];
    if constexpr (NeedRx) direction = rx[input];
    if constexpr (NeedBx) input_seed = bx[input];
    if constexpr (NeedBg) output_seed = bg[output];
    const auto result = activation_triple_backward_value<
        Path::kind, Path::normalize, WriteTX, WriteTG, WriteTRX, HasBx, HasBg>(
        value, grad, direction, input_seed, output_seed, Path::parameters());
    if constexpr (WriteTX) out_tx[input] = result.tx;
    if constexpr (WriteTG) out_tg[output] = result.tg;
    if constexpr (WriteTRX) out_trx[input] = result.trx;
}
template<class Path>
__device__ __forceinline__ void gate_triple_backward_product(
    const Element* x, const Element* grad_out, const Element* rx,
    const Element* bx, const Element* bg, Element* out_tx,
    Element* out_tg, Element* out_trx, Index row, Index copy) {
    const Index gate_input = row * InWidth + Path::gate_offset + copy;
    const Index input = row * InWidth + Path::input_offset + copy * Path::components;
    const Index output = row * OutWidth + Path::output_offset + copy * Path::components;
    Element value = Element(0), gate_direction = Element(0), gate_seed = Element(0);
    if constexpr (NeedX) value = x[gate_input];
    if constexpr (NeedRx) gate_direction = rx[gate_input];
    if constexpr (NeedBx) gate_seed = bx[gate_input];
    Element vectors[Path::components], directions[Path::components], grads[Path::components];
    Element input_seeds[Path::components], output_seeds[Path::components];
    Element contraction[Path::components], direction_contraction[Path::components];
    Element mixed_contraction[Path::components], output_contraction[Path::components];
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        vectors[component] = directions[component] = grads[component] = Element(0);
        input_seeds[component] = output_seeds[component] = Element(0);
        if constexpr (NeedX) vectors[component] = x[input + component];
        if constexpr (NeedRx) directions[component] = rx[input + component];
        if constexpr (NeedGrad) grads[component] = grad_out[output + component];
        if constexpr (NeedBx) input_seeds[component] = bx[input + component];
        if constexpr (NeedBg) output_seeds[component] = bg[output + component];
        contraction[component] = gate_multiply(grads[component], vectors[component]);
        direction_contraction[component] = gate_multiply(directions[component], grads[component]);
        if constexpr (HasBg) {
            mixed_contraction[component] = gate_add(
                gate_multiply(output_seeds[component], vectors[component]),
                gate_multiply(input_seeds[component], grads[component]));
            output_contraction[component] = gate_multiply(output_seeds[component], directions[component]);
        } else {
            // None omits this gradient's products; multiplying a zero-filled
            // stand-in by a nonfinite component changes the registered chain.
            mixed_contraction[component] = gate_multiply(input_seeds[component], grads[component]);
            output_contraction[component] = Element(0);
        }
    }
    const auto parameters = Path::parameters();
    const Element moment = gate_component_sum(contraction);
    const Element direction_moment = gate_component_sum(direction_contraction);
    const Element mixed_moment = gate_component_sum(mixed_contraction);
    const Element output_moment = gate_component_sum(output_contraction);
    const auto result = gate_triple_product_values<Path>(
        value, moment, direction_moment, mixed_moment, output_moment,
        gate_direction, gate_seed, parameters);
    if constexpr (WriteTX) out_tx[gate_input] = result.tx;
    if constexpr (WriteTRX) out_trx[gate_input] = result.trx;
    Element gate = Element(0);
    if constexpr (WriteTRX && HasBg) {
        gate = activation_forward_value<Path::kind, Path::normalize>(value, parameters);
    }
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        if constexpr (WriteTX) {
            if constexpr (HasBg) {
                out_tx[input + component] = gate_add(
                    gate_multiply(output_seeds[component], result.direction_slope),
                    gate_multiply(result.curvature, grads[component]));
            } else {
                out_tx[input + component] = gate_add(
                    Element(0), gate_multiply(result.curvature, grads[component]));
            }
        }
        if constexpr (WriteTG) {
            out_tg[output + component] = gate_add(
                gate_add(gate_multiply(result.input_slope, directions[component]),
                         gate_multiply(input_seeds[component], result.direction_slope)),
                gate_multiply(result.curvature, vectors[component]));
        }
        if constexpr (WriteTRX) {
            if constexpr (HasBg) {
                out_trx[input + component] = gate_add(
                    gate_multiply(output_seeds[component], gate),
                    gate_multiply(result.input_slope, grads[component]));
            } else {
                out_trx[input + component] = gate_add(
                    Element(0), gate_multiply(result.input_slope, grads[component]));
            }
        }
    }
}'''


TRIPLE_BACKWARD_KERNEL = r'''__global__ __launch_bounds__(128, 1) void gate_triple_backward(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ bx,
    const Element* __restrict__ bg, Element* __restrict__ out_tx,
    Element* __restrict__ out_tg, Element* __restrict__ out_trx, int64_t num_items) {{
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < num_items * {count}; item += Index(gridDim.x) * 128) {{
        const Index row = item / {count};
        const Index group = item % {count};
        {schedule}
    }}
}}'''


def required_inputs(meta, available):
    required = activation_required_inputs(meta["scalar_activation"], available)
    has_bx, has_bg = "bx" in available, "bg" in available
    if not has_bx and not has_bg:
        return required
    for path in meta["gated_paths"]:
        kind = path["activation"]["kind"]
        tx = required.get("tx")
        tg = required.get("tg")
        trx = required.get("trx")
        if has_bx:
            if kind != "identity":
                reads = {"x"} if kind == "relu" else set() if kind == "abs" else {"x", "grad_out", "rx", "bx"}
                tx = (set() if tx is None else tx) | reads
            reads = {"rx", "bx"}
            if kind != "identity":
                reads.add("x")
            tg = (set() if tg is None else tg) | reads
            reads = {"grad_out", "bx"}
            if kind != "identity":
                reads.add("x")
            trx = (set() if trx is None else trx) | reads
        if has_bg:
            reads = {"rx", "bg"}
            if kind != "identity":
                reads.add("x")
            tx = (set() if tx is None else tx) | reads
            trx = (set() if trx is None else trx) | {"x", "bg"}
        required.update(tx=tx, tg=tg, trx=trx)
    return required


def make_source(meta, config, *, tensor_only=False, fragments=False):
    assert config["order"] == 3
    declarations, paths = make_path_types(meta, config["dtype"])
    flags = dict(
        NeedX="x" in config["inputs"], NeedGrad="grad_out" in config["inputs"],
        NeedRx="rx" in config["inputs"], NeedBx="bx" in config["inputs"],
        NeedBg="bg" in config["inputs"], HasBx="bx" in config["available"],
        HasBg="bg" in config["available"], WriteTX="tx" in config["active_outputs"],
        WriteTG="tg" in config["active_outputs"], WriteTRX="trx" in config["active_outputs"],
    )
    declarations += [
        f"constexpr int InWidth = {meta['in_dim']};\nconstexpr int OutWidth = {meta['out_dim']};",
        "\n".join(f"constexpr bool {name} = {str(value).lower()};" for name, value in flags.items()),
        "\n\n".join(BACKWARD_FUNCTIONS.values()), BACKWARD_MATH, DOUBLE_BACKWARD_RESULT, DOUBLE_BACKWARD_MATH,
        TRIPLE_BACKWARD_RESULT, TRIPLE_BACKWARD_MATH, TRIPLE_BACKWARD_MATH_SOURCE,
    ]
    count, schedule = make_schedule(
        paths,
        "gate_triple_backward_scalar<{path}>(x, grad_out, rx, bx, bg, out_tx, out_tg, out_trx, row, copy);",
        "gate_triple_backward_product<{path}>(x, grad_out, rx, bx, bg, out_tx, out_tg, out_trx, row, copy);",
    )
    arguments = [f"{name}.data_ptr<Element>()" if name in config["inputs"] else "nullptr"
                 for name in ("x", "grad_out", "rx", "bx", "bg")]
    arguments += [f"result_{name}.data_ptr<Element>()" if name in config["active_outputs"] else "nullptr"
                  for name in ("tx", "tg", "trx")]
    launch = (
        f"if(num_items>0) gate_triple_backward<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
        f"({','.join(arguments)},num_items);"
    )
    return make_binding(
        meta, config, declarations=declarations,
        kernel_body=TRIPLE_BACKWARD_KERNEL.format(count=count, schedule=schedule),
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        tensor_only=tensor_only, fragments=fragments,
    )
