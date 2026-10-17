"""Scalar activations with e3nn normalization and recomputed CUDA gradients."""

from collections import OrderedDict
from functools import lru_cache, partial
from hashlib import sha256

import torch
from torch.utils._python_dispatch import _disable_current_modes

from .kernels.utils import apply_activation_spec, activation_config, get_kernel_config
from ...o3.irreps import PublicIrreps as Irreps, irreps_key
from .kernels import backward, double_backward, forward, triple_backward
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling


_IRREPS_MISSING = object()


@torch.compiler.assume_constant_result
def register_ops(paths, index_bits, gate_layout=None):
    """Register a static activation or gate; runtime operators accept tensors and flags."""
    layout = None if gate_layout is None else {
        key: gate_layout[key]
        for key in ("in_dim", "out_dim", "scalar_paths", "gated_paths", "scalar_activation", "has_grad")
    }
    suffix = sha256(repr((paths, index_bits, layout)).encode()).hexdigest()[:24]
    name = f"activation_{suffix}_forward"
    with REGISTER_OPERATOR_LOCK:
        if hasattr(torch.ops.lunaeq, name):
            return name
        bits, cache = index_bits, OrderedDict()

        @torch.library.custom_op(f"lunaeq::activation_{suffix}_forward", mutates_args=())
        def activation_forward_op(x: torch.Tensor) -> torch.Tensor:
            return forward.activation_forward(x, paths, bits, cache, layout)[0]

        @activation_forward_op.register_fake
        def activation_forward_fake(x):
            width = sum(width for width, _ in paths) if layout is None else layout["out_dim"]
            return x.new_empty((*x.shape[:-1], width))

        def activation_backward_fake_outputs(order, inputs, needs=None, grad_mode=False):
            names = (("x", "grad_out"), ("x", "grad_out", "rx"), ("x", "grad_out", "rx", "bx", "bg"))[order - 1]
            available = tuple(name for name, value in zip(names, inputs) if value is not None)
            x = next(value for value in inputs if value is not None)
            program = get_kernel_config(paths, order, x.dtype, cache, needs=needs, available=available,
                                        grad_mode=grad_mode, gate_layout=layout)
            in_dim = sum(width for width, _ in paths) if layout is None else layout["in_dim"]
            out_dim = in_dim if layout is None else layout["out_dim"]
            return tuple(
                x.new_empty((*x.shape[:-1], out_dim if name in ("sg", "tg") else in_dim))
                if name in program["active_outputs"] else None
                for name in program["outputs"]
            )

        @torch.library.custom_op(f"lunaeq::activation_{suffix}_backward", mutates_args=())
        def activation_backward_op(x: torch.Tensor, g: torch.Tensor, grad_mode: bool) -> torch.Tensor:
            return backward.activation_backward(x, g, paths, bits, cache, layout, x.dtype, grad_mode=grad_mode)

        @activation_backward_op.register_fake
        def activation_backward_fake(x, g, grad_mode):
            return activation_backward_fake_outputs(1, (x, g), grad_mode=grad_mode)[0]

        @torch.library.custom_op(f"lunaeq::activation_{suffix}_double_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor g, Tensor rx, bool[] needs) -> (Tensor?, Tensor?)")
        def activation_double_backward_op(x: torch.Tensor, g: torch.Tensor, rx: torch.Tensor,
                                          needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            return double_backward.activation_double_backward(x, g, rx, paths, bits, cache, layout, x.dtype, needs=needs)

        @activation_double_backward_op.register_fake
        def activation_double_backward_fake(x, g, rx, needs):
            return activation_backward_fake_outputs(2, (x, g, rx), needs)

        @torch.library.custom_op(f"lunaeq::activation_{suffix}_triple_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor g, Tensor rx, Tensor? bx, Tensor? bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def activation_triple_backward_op(x: torch.Tensor, g: torch.Tensor, rx: torch.Tensor,
                                          bx: torch.Tensor | None, bg: torch.Tensor | None,
                                          needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return triple_backward.activation_triple_backward(x, g, rx, bx, bg, paths, bits, cache,
                                                               layout, x.dtype, needs=needs)

        @activation_triple_backward_op.register_fake
        def activation_triple_backward_fake(x, g, rx, bx, bg, needs):
            return activation_backward_fake_outputs(3, (x, g, rx, bx, bg), needs)

        def activation_forward_setup(ctx, inputs, output):
            (x,) = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x)
            program = get_kernel_config(paths, 1, x.dtype, cache, grad_mode=True, gate_layout=layout)
            if not program["active_outputs"]:
                ctx.mark_non_differentiable(output)

        def activation_forward_backward(ctx, g):
            (x,) = ctx.saved_tensors
            return (None if g is None else activation_backward_op(x, g, torch.is_grad_enabled())),

        def activation_backward_setup(ctx, inputs, output):
            x, g, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, g)

        def activation_backward_backward(ctx, rx):
            if rx is None:
                return None, None, None
            x, g = ctx.saved_tensors
            sx, sg = activation_double_backward_op(x, g, rx, list(ctx.needs_input_grad[:2]))
            return sx, sg, None

        def activation_double_backward_setup(ctx, inputs, output):
            x, g, rx, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, g, rx)

        def activation_double_backward_backward(ctx, bx, bg):
            x, g, rx = ctx.saved_tensors
            result = activation_triple_backward_op(x, g, rx, bx, bg, list(ctx.needs_input_grad[:3]))
            return *result, None

        activation_forward_op.register_autograd(activation_forward_backward, setup_context=activation_forward_setup)
        activation_backward_op.register_autograd(activation_backward_backward, setup_context=activation_backward_setup)
        activation_double_backward_op.register_autograd(activation_double_backward_backward,
                                                       setup_context=activation_double_backward_setup)
        return name


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
                        raise ValueError("Activation parity is violated: an odd scalar requires an even or odd activation")
        paths.append((mul * ir.dim, spec))
        outputs.append((mul, (ir.l, parity)))
    return tuple(paths), Irreps(outputs), OrderedDict()


def activation(features, irreps_in=None, acts=None, dim=-1, *, paths=None, kernel_cache=None):
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
        result = getattr(torch.ops.lunaeq, register_ops(paths, 32)).default(value)
        return result.movedim(-1, dim) if dim != ndim - 1 else result
    if kernel_cache is None:
        kernel_cache = OrderedDict()
    if torch.is_inference_mode_enabled():
        with torch.no_grad():
            result = ActivationFunction.apply(value, paths, 32, kernel_cache, None)
    else:
        result = ActivationFunction.apply(value, paths, 32, kernel_cache, None)
    return result.movedim(-1, dim) if dim != ndim - 1 else result


class ActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, paths, index_bits, kernel_cache, gate_layout):
        result, backward_kernel, differentiable, save_input = forward.activation_forward(
            features, paths, index_bits, kernel_cache, gate_layout
        )
        if ctx.needs_input_grad[0]:
            ctx.paths = paths
            ctx.index_bits = index_bits
            ctx.kernel_cache = kernel_cache
            ctx.gate_layout = gate_layout
            ctx.backward_kernel = backward_kernel
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(features if save_input or features.numel() == 0 else None)
            if not differentiable:
                ctx.mark_non_differentiable(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None

        (features,) = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            grad_input = ActivationBackward.apply(
                features, grad_output, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, ctx.gate_layout, grad_output.dtype,
            )
        elif ctx.backward_kernel is not None:
            grad_input = ctx.backward_kernel(features, grad_output)
        else:
            grad_input = backward.activation_backward(
                features, grad_output, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, ctx.gate_layout, grad_output.dtype,
            )
        return grad_input, None, None, None, None


class ActivationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, paths, index_bits, kernel_cache, gate_layout, dtype):
        ctx.paths = paths
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.gate_layout = gate_layout
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = backward.activation_backward(
            features, grad_output, paths, index_bits, kernel_cache, gate_layout, dtype, grad_mode=True
        )

        config = get_kernel_config(
            paths, 2, dtype, kernel_cache, needs=ctx.needs_input_grad[:2],
            available=("x", "grad_out", "rx"), gate_layout=gate_layout,
        )
        ctx.save_for_backward(
            features if features is not None and ("x" in config["inputs"] or features.numel() == 0) else None,
            grad_output if grad_output is not None and ("grad_out" in config["inputs"] or grad_output.numel() == 0) else None,
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
                ctx.kernel_cache, ctx.gate_layout, ctx.dtype, needs,
            )
        else:
            gradients = double_backward.activation_double_backward(
                features, grad_output, grad_grad_input, ctx.paths, ctx.index_bits,
                ctx.kernel_cache, ctx.gate_layout, ctx.dtype, needs=needs,
            )
        return *gradients, None, None, None, None, None


class ActivationDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, grad_grad_input, paths, index_bits, kernel_cache, gate_layout, dtype, needs):
        ctx.paths = paths
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.gate_layout = gate_layout
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = double_backward.activation_double_backward(
            features, grad_output, grad_grad_input, paths, index_bits,
            kernel_cache, gate_layout, dtype, needs=needs,
        )

        required = set()
        for output, grad_name in zip(result, ("bx", "bg")):
            if output is None:
                continue
            config = get_kernel_config(
                paths, 3, dtype, kernel_cache, needs=ctx.needs_input_grad[:3],
                available=("x", "grad_out", "rx", grad_name), gate_layout=gate_layout,
            )
            required.update(config["inputs"])
            if not config["active_outputs"]:
                ctx.mark_non_differentiable(output)
        ctx.save_for_backward(
            features if features is not None and ("x" in required or features.numel() == 0) else None,
            grad_output if grad_output is not None and ("grad_out" in required or grad_output.numel() == 0) else None,
            grad_grad_input if grad_grad_input is not None and ("rx" in required or grad_grad_input.numel() == 0) else None,
        )
        return result

    @staticmethod
    def backward(ctx, grad_input, grad_grad_output):
        features, grad_output, grad_grad_input = ctx.saved_tensors
        gradients = triple_backward.activation_triple_backward(
            features, grad_output, grad_grad_input, grad_input, grad_grad_output,
            ctx.paths, ctx.index_bits, ctx.kernel_cache, ctx.gate_layout, ctx.dtype,
            needs=ctx.needs_input_grad[:3],
        )
        return *gradients, None, None, None, None, None, None


__all__ = ['activation']
