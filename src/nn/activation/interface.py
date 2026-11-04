"""Scalar activations with e3nn normalization and recomputed CUDA gradients."""

from collections import OrderedDict
from functools import lru_cache, partial

import torch
from torch.utils._python_dispatch import _disable_current_modes

from .kernels.utils import apply_activation_spec, activation_config, get_kernel_config
from ...o3.irreps import PublicIrreps as Irreps, irreps_key
from .kernels import backward, double_backward, forward, triple_backward
from ...utils import is_compiling
from ...utils.operator_config import encode_config, decode_config

__all__ = ['activation']


_IRREPS_MISSING = object()


@lru_cache(maxsize=128)
def get_activation_operator(config):
    paths, bits = decode_config(config)
    return paths, bits, OrderedDict()


@torch.library.custom_op("lunaeq::activation_forward", mutates_args=())
def activation_forward_op(x: torch.Tensor, config: str) -> torch.Tensor:
    paths, bits, cache = get_activation_operator(config)
    return forward.activation_forward(x, paths, bits, cache)[0]


@activation_forward_op.register_fake
def activation_forward_fake(x, config):
    paths, bits, cache = get_activation_operator(config)
    width = sum(width for width, _ in paths)
    return x.new_empty((*x.shape[:-1], width))


def activation_backward_fake_outputs(config, order, inputs, needs=None, grad_mode=False):
    paths, bits, cache = get_activation_operator(config)
    names = (("x", "grad_out"), ("x", "grad_out", "rx"), ("x", "grad_out", "rx", "bx", "bg"))[order - 1]
    available = tuple(name for name, value in zip(names, inputs) if value is not None)
    x = next(value for value in inputs if value is not None)
    program = get_kernel_config(paths, order, x.dtype, cache, needs=needs, available=available,
                                grad_mode=grad_mode)
    width = sum(width for width, _ in paths)
    return tuple(
        x.new_empty((*x.shape[:-1], width))
        if name in program["active_outputs"] else None
        for name in program["outputs"]
    )


@torch.library.custom_op("lunaeq::activation_backward", mutates_args=())
def activation_backward_op(x: torch.Tensor, g: torch.Tensor, grad_mode: bool, config: str) -> torch.Tensor:
    paths, bits, cache = get_activation_operator(config)
    return backward.activation_backward(x, g, paths, bits, cache, x.dtype, grad_mode=grad_mode)


@activation_backward_op.register_fake
def activation_backward_fake(x, g, grad_mode, config):
    return activation_backward_fake_outputs(config, 1, (x, g), grad_mode=grad_mode)[0]


@torch.library.custom_op(
    "lunaeq::activation_double_backward", mutates_args=(),
    schema="(Tensor x, Tensor g, Tensor rx, bool[] needs, str config) -> (Tensor?, Tensor?)",
)
def activation_double_backward_op(x, g, rx, needs, config):
    paths, bits, cache = get_activation_operator(config)
    return double_backward.activation_double_backward(x, g, rx, paths, bits, cache, x.dtype, needs=needs)


@activation_double_backward_op.register_fake
def activation_double_backward_fake(x, g, rx, needs, config):
    return activation_backward_fake_outputs(config, 2, (x, g, rx), needs)


@torch.library.custom_op(
    "lunaeq::activation_triple_backward", mutates_args=(),
    schema=(
        "(Tensor x, Tensor g, Tensor rx, Tensor? bx, Tensor? bg, bool[] needs, "
        "str config) -> (Tensor?, Tensor?, Tensor?)"
    ),
)
def activation_triple_backward_op(x, g, rx, bx, bg, needs, config):
    paths, bits, cache = get_activation_operator(config)
    return triple_backward.activation_triple_backward(
        x, g, rx, bx, bg, paths, bits, cache, x.dtype, needs=needs,
    )


@activation_triple_backward_op.register_fake
def activation_triple_backward_fake(x, g, rx, bx, bg, needs, config):
    return activation_backward_fake_outputs(config, 3, (x, g, rx, bx, bg), needs)


def activation_forward_setup(ctx, inputs, output):
    x, ctx.config = inputs
    paths, bits, cache = get_activation_operator(ctx.config)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x)
    program = get_kernel_config(paths, 1, x.dtype, cache, grad_mode=True)
    if not program["active_outputs"]:
        ctx.mark_non_differentiable(output)


def activation_forward_backward(ctx, g):
    (x,) = ctx.saved_tensors
    return (None if g is None else activation_backward_op(x, g, torch.is_grad_enabled(), ctx.config)), None


def activation_backward_setup(ctx, inputs, output):
    ctx.config = inputs[-1]
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:-2])


def activation_backward_backward(ctx, rx):
    if rx is None:
        return None, None, None, None
    x, g = ctx.saved_tensors
    sx, sg = activation_double_backward_op(x, g, rx, list(ctx.needs_input_grad[:2]), ctx.config)
    return sx, sg, None, None


def activation_double_backward_backward(ctx, bx, bg):
    x, g, rx = ctx.saved_tensors
    result = activation_triple_backward_op(x, g, rx, bx, bg, list(ctx.needs_input_grad[:3]), ctx.config)
    return *result, None, None


activation_forward_op.register_autograd(activation_forward_backward, setup_context=activation_forward_setup)
activation_backward_op.register_autograd(activation_backward_backward, setup_context=activation_backward_setup)
activation_double_backward_op.register_autograd(
    activation_double_backward_backward, setup_context=activation_backward_setup,
)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def prepare_activation(irreps_in, acts):
    """Cache e3nn's normalized scalar paths without constructing a Module."""
    from e3nn.math import moment

    irreps = Irreps(irreps_in)
    if len(irreps) != len(acts):
        raise ValueError("Irreps and number of activation functions do not match")
    paths, outputs = [], []
    for (mul, ir), config in zip(irreps, acts):
        spec = (dict(kind="identity", params={}) if config is None
                else dict(kind=config[0], params=dict(config[1])))
        parity = ir.p
        if config is None:
            spec.update(cst=1.0, is_id=True)
        else:
            if ir.l != 0:
                raise ValueError("Activation cannot apply an activation function to a non-scalar input")
            # e3nn's fixed CPU normalization is configuration, not a tensor
            # input computation recorded by make_fx.
            with torch.no_grad(), _disable_current_modes():
                spec.update(cst=1.0, is_id=True)
                function = partial(apply_activation_spec, spec=spec)
                cst = moment(function, 2, dtype=torch.float64, device="cpu").pow(-0.5).item()
                spec.update(cst=cst, is_id=abs(cst - 1) < 1e-4)
                if ir.p == -1:
                    x = torch.linspace(0, 10, 256)
                    positive, negative = apply_activation_spec(x, spec), apply_activation_spec(-x, spec)
                    parity = 1 if (positive - negative).abs().max() < 1e-5 else (
                        -1 if (positive + negative).abs().max() < 1e-5 else 0
                    )
                    if parity == 0:
                        raise ValueError(
                            "Activation parity is violated: an odd scalar requires an even or odd activation"
                        )
        paths.append((mul * ir.dim, spec))
        outputs.append((mul, (ir.l, parity)))
    return tuple(paths), Irreps(outputs), OrderedDict()


def activation(features, irreps_in=None, acts=None, dim=-1, *, paths=None, kernel_cache=None, operator_config=None):
    """Apply normalized scalar activations, with e3nn irreps and activation functions.

    ``paths`` and ``kernel_cache`` let a Module reuse its prepared configuration.
    Ordinary function calls only need ``features``, ``irreps_in`` and ``acts``.
    """
    if paths is None:
        paths, _, cached_kernels = prepare_activation(
            irreps_key(irreps_in),
            tuple(map(activation_config, acts))
        )
        if kernel_cache is None:
            kernel_cache = cached_kernels
    shape = features.shape
    ndim = len(shape)
    cuda = features.is_cuda and features.dtype in (torch.float32, torch.float64)
    valid_dim = ndim and isinstance(dim, int) and not isinstance(dim, bool) and -ndim <= dim < ndim
    if not cuda or not valid_dim or shape[dim] != sum(width for width, _ in paths):
        outputs, offset = [], 0
        for width, spec in paths:
            outputs.append(apply_activation_spec(features.narrow(dim, offset, width), spec))
            offset += width
        return torch.cat(outputs, dim=dim) if len(outputs) > 1 else (
            outputs[0] if outputs else torch.zeros_like(features)
        )
    dim %= ndim
    if not paths:
        return torch.zeros_like(features)
    if len(paths) == 1 and paths[0][1]["kind"] == "identity" and paths[0][1]["is_id"]:
        return features.narrow(dim, 0, paths[0][0])
    value = features.movedim(dim, -1) if dim != ndim - 1 else features
    if is_compiling():
        config = encode_config((paths, 32)) if operator_config is None else operator_config
        result = torch.ops.lunaeq.activation_forward.default(value, config)
        return result.movedim(-1, dim) if dim != ndim - 1 else result
    if kernel_cache is None:
        kernel_cache = OrderedDict()
    if torch.is_inference_mode_enabled():
        with torch.no_grad():
            result = ActivationFunction.apply(value, paths, 32, kernel_cache)
    else:
        result = ActivationFunction.apply(value, paths, 32, kernel_cache)
    return result.movedim(-1, dim) if dim != ndim - 1 else result


class ActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, paths, index_bits, kernel_cache):
        result, backward_kernel, differentiable, save_input = forward.activation_forward(
            features, paths, index_bits, kernel_cache
        )
        if ctx.needs_input_grad[0]:
            ctx.paths = paths
            ctx.index_bits = index_bits
            ctx.kernel_cache = kernel_cache
            ctx.backward_kernel = backward_kernel
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(features if save_input or features.numel() == 0 else None)
            if not differentiable:
                ctx.mark_non_differentiable(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None

        (features,) = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            grad_input = ActivationBackward.apply(
                features, grad_output, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, grad_output.dtype,
            )
        elif ctx.backward_kernel is not None:
            grad_input = ctx.backward_kernel(features, grad_output)
        else:
            grad_input = backward.activation_backward(
                features, grad_output, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, grad_output.dtype,
            )
        return grad_input, None, None, None


class ActivationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, paths, index_bits, kernel_cache, dtype):
        ctx.paths = paths
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = backward.activation_backward(
            features, grad_output, paths, index_bits, kernel_cache, dtype, grad_mode=True
        )

        config = get_kernel_config(
            paths, 2, dtype, kernel_cache, needs=ctx.needs_input_grad[:2],
            available=("x", "grad_out", "rx"),
        )
        ctx.save_for_backward(
            features if features is not None and ("x" in config["inputs"] or features.numel() == 0) else None,
            grad_output if grad_output is not None and (
                "grad_out" in config["inputs"] or grad_output.numel() == 0
            ) else None,
        )
        if result is not None and not config["active_outputs"]:
            ctx.mark_non_differentiable(result)
        return result

    @staticmethod
    def backward(ctx, grad_grad_input):
        features, grad_output = ctx.saved_tensors
        needs = ctx.needs_input_grad[:2]
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            gradients = ActivationDoubleBackward.apply(
                features, grad_output, grad_grad_input, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, ctx.dtype, needs,
            )
        else:
            gradients = double_backward.activation_double_backward(
                features, grad_output, grad_grad_input, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, ctx.dtype, needs=needs,
            )
        return *gradients, None, None, None, None


class ActivationDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, grad_grad_input, paths, index_bits, kernel_cache, dtype, needs):
        ctx.paths = paths
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = double_backward.activation_double_backward(
            features, grad_output, grad_grad_input, paths, index_bits,
            kernel_cache, dtype, needs=needs,
        )

        required = set()
        for output, grad_name in zip(result, ("bx", "bg")):
            if output is None:
                continue
            config = get_kernel_config(
                paths, 3, dtype, kernel_cache, needs=ctx.needs_input_grad[:3],
                available=("x", "grad_out", "rx", grad_name),
            )
            required.update(config["inputs"])
            if not config["active_outputs"]:
                ctx.mark_non_differentiable(output)
        ctx.save_for_backward(
            features if features is not None and ("x" in required or features.numel() == 0) else None,
            grad_output if grad_output is not None and ("grad_out" in required or grad_output.numel() == 0) else None,
            grad_grad_input if grad_grad_input is not None and (
                "rx" in required or grad_grad_input.numel() == 0
            ) else None,
        )
        return result

    @staticmethod
    def backward(ctx, grad_input, grad_grad_output):
        features, grad_output, grad_grad_input = ctx.saved_tensors
        gradients = triple_backward.activation_triple_backward(
            features, grad_output, grad_grad_input, grad_input, grad_grad_output,
            ctx.paths, ctx.index_bits, ctx.kernel_cache, ctx.dtype,
            needs=ctx.needs_input_grad[:3],
        )
        return *gradients, None, None, None, None, None
