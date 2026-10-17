"""NormActivation with built-in scalar activations and fused CUDA backward."""

from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256

import torch

from ...o3.norm.interface import NATIVE_DTYPES, SUPPORTED_DTYPES, build_norm_metadata
from .kernels.utils import NormActivationKernels, run_backward
from .kernels.forward import make_forward_source
from ..activation.kernels.utils import parse_activation, activation_config
from ...o3.irreps import PublicIrreps as Irreps, irreps_key
from .torch import continue_triple_backward, norm_activation_torch
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling


@torch.compiler.assume_constant_result
def register_module_ops(self, dtype):
    return register_ops(self.meta, 32, dtype)


@torch.compiler.assume_constant_result
def register_ops(metadata, index_bits, dtype):
    metadata = {key: value for key, value in metadata.items() if key not in ("irreps_in", "irreps_out", "dtype")}
    suffix = sha256(repr((metadata, index_bits, dtype)).encode()).hexdigest()[:24]
    name = f"norm_activation_{suffix}_forward"
    with REGISTER_OPERATOR_LOCK:
        if hasattr(torch.ops.lunaeq, name):
            return name
        metadata = {**metadata, "dtype": dtype}
        op = NormActivationKernels(meta=metadata, forward_source=make_forward_source,
                                  compute_dtype=dtype, index_bits=index_bits,
                                  kernels=OrderedDict(), programs=OrderedDict())

        @torch.library.custom_op(f"lunaeq::norm_activation_{suffix}_forward", mutates_args=(),
                                 schema="(Tensor x, Tensor? bias) -> (Tensor, Tensor?)")
        def norm_activation_forward_op(x: torch.Tensor, bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
            if bias is None:
                return op.run(0, x, None), None
            return op.run(0, x, bias, save_bias=True)

        @norm_activation_forward_op.register_fake
        def norm_activation_forward_fake(x, bias):
            metadata = op.meta
            return x.new_empty((*x.shape[:-1], metadata["out_dim"])), None if bias is None else bias.new_empty(bias.shape)

        def norm_activation_backward_fake_outputs(order, inputs, needs, lower_seeds=None, grad_mode=False):
            from .kernels.graph import build_program
            from .kernels.utils import INPUT_ROLES, BIAS_OUTPUTS, output_width

            x = inputs[0]
            available = tuple(role for role, value in zip(INPUT_ROLES[order], inputs) if value is not None)
            key = (order, tuple(needs), available, grad_mode, lower_seeds)
            program = op.programs.get(key)
            if program is None:
                program = build_program(op.meta, order, needs=needs, available=available,
                                        grad_mode=grad_mode, lower_seeds=lower_seeds)
                op.programs[key] = program
                if len(op.programs) > 128:
                    op.programs.popitem(last=False)
            return tuple(
                x.new_empty((op.meta["bias_dim"],) if name in BIAS_OUTPUTS
                            else (*x.shape[:-1], output_width(op.meta, name)))
                if name in program["active_outputs"] else None
                for name in program["outputs"]
            )

        @torch.library.custom_op(f"lunaeq::norm_activation_{suffix}_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? bias, Tensor g, bool[] needs, bool grad_mode) -> (Tensor?, Tensor?)")
        def norm_activation_backward_op(x: torch.Tensor, bias: torch.Tensor | None, g: torch.Tensor,
                                        needs: list[bool], grad_mode: bool) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            return run_backward(op, 1, (x, bias, g), needs=needs, grad_mode=grad_mode)

        @norm_activation_backward_op.register_fake
        def norm_activation_backward_fake(x, bias, g, needs, grad_mode):
            return norm_activation_backward_fake_outputs(1, (x, bias, g), needs, grad_mode=grad_mode)

        @torch.library.custom_op(f"lunaeq::norm_activation_{suffix}_double_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? bias, Tensor g, Tensor? rx, Tensor? rb, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def norm_activation_double_backward_op(x: torch.Tensor, bias: torch.Tensor | None, g: torch.Tensor,
                                               rx: torch.Tensor | None, rb: torch.Tensor | None,
                                               needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return run_backward(op, 2, (x, bias, g, rx, rb), needs=needs)

        @norm_activation_double_backward_op.register_fake
        def norm_activation_double_backward_fake(x, bias, g, rx, rb, needs):
            return norm_activation_backward_fake_outputs(2, (x, bias, g, rx, rb), needs)

        @torch.library.custom_op(f"lunaeq::norm_activation_{suffix}_triple_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? bias, Tensor g, Tensor? rx, Tensor? rb, Tensor? bx, Tensor? bb, Tensor? bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def norm_activation_triple_backward_op(x: torch.Tensor, bias: torch.Tensor | None, g: torch.Tensor,
                                               rx: torch.Tensor | None, rb: torch.Tensor | None,
                                               bx: torch.Tensor | None, bb: torch.Tensor | None, bg: torch.Tensor | None,
                                               needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return run_backward(op, 3, (x, bias, g, rx, rb, bx, bb, bg), needs=needs,
                                 lower_seeds=(rx is not None, rb is not None))

        @norm_activation_triple_backward_op.register_fake
        def norm_activation_triple_backward_fake(x, bias, g, rx, rb, bx, bb, bg, needs):
            return norm_activation_backward_fake_outputs(3, (x, bias, g, rx, rb, bx, bb, bg), needs,
                                                         (rx is not None, rb is not None))

        def norm_activation_forward_setup(ctx, inputs, output):
            x, bias = inputs
            _, saved_bias = output
            ctx.set_materialize_grads(False)
            ctx.needed = ctx.needs_input_grad[:2]
            if saved_bias is not None and not ctx.needed[1]:
                ctx.mark_non_differentiable(saved_bias)
            ctx.save_for_backward(x, saved_bias)

        def norm_activation_forward_backward(ctx, g, bias_gradient):
            x, bias = ctx.saved_tensors
            dx, db = (None, None) if g is None else norm_activation_backward_op(
                x, bias, g, list(ctx.needed), torch.is_grad_enabled()
            )
            if bias_gradient is not None:
                db = bias_gradient if db is None else db + bias_gradient
            return dx, db

        def norm_activation_backward_setup(ctx, inputs, output):
            x, bias, g, _, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, bias, g)

        def norm_activation_backward_backward(ctx, rx, rb):
            x, bias, g = ctx.saved_tensors
            result = norm_activation_double_backward_op(x, bias, g, rx, rb, list(ctx.needs_input_grad[:3]))
            return *result, None, None

        def norm_activation_double_backward_setup(ctx, inputs, output):
            x, bias, g, rx, rb, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, bias, g, rx, rb)

        def norm_activation_double_backward_backward(ctx, bx, bb, bg):
            x, bias, g, rx, rb = ctx.saved_tensors
            result = norm_activation_triple_backward_op(x, bias, g, rx, rb, bx, bb, bg,
                                                       list(ctx.needs_input_grad[:5]))
            return *result, None

        norm_activation_forward_op.register_autograd(norm_activation_forward_backward,
                                                    setup_context=norm_activation_forward_setup)
        norm_activation_backward_op.register_autograd(norm_activation_backward_backward,
                                                     setup_context=norm_activation_backward_setup)
        norm_activation_double_backward_op.register_autograd(norm_activation_double_backward_backward,
                                                            setup_context=norm_activation_double_backward_setup)
        return name


class NormActivationDoubleBackward(torch.autograd.Function):
    """Recompute the complete double-backward in one registered CUDA kernel."""

    @staticmethod
    def forward(ctx, op, x, bias, g, rx, rb, needs):
        ctx.op = op
        ctx.lower_seeds = rx is not None, rb is not None
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, bias, g, rx, rb)
        return run_backward(op, 2, (x, bias, g, rx, rb), needs=needs)

    @staticmethod
    def backward(ctx, bx, bb, bg):
        x, bias, g, rx, rb = ctx.saved_tensors
        needs = ctx.needs_input_grad[1:6]  # x, bias, g, rx, rb
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            # Preserve arbitrary later derivatives through the original
            # expressions; the normal terminal triple-backward never takes this path.
            result = continue_triple_backward(ctx.op.meta, x, bias, g, rx, rb, bx, bb, bg, needs)
            return (None, *result, None)

        ctx.op.backward_backend = "cuda_fused_recompute"
        tx, tb, tg, trx, trb = run_backward(
            ctx.op, 3, (x, bias, g, rx, rb, bx, bb, bg), needs=needs, lower_seeds=ctx.lower_seeds,
        )
        return None, tx, tb, tg, trx, trb, None


class NormActivationBackward(torch.autograd.Function):
    """A single AD node with fused DX and shared-DB recomputation."""

    @staticmethod
    def forward(ctx, op, x, bias, g, needs):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, bias, g)
        return run_backward(op, 1, (x, bias, g), needs=needs, grad_mode=True)

    @staticmethod
    def backward(ctx, rx, rb):
        x, bias, g = ctx.saved_tensors
        needs = ctx.needs_input_grad[1:4]  # x, bias, g

        ctx.op.backward_backend = "cuda_fused_recompute"
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            sx, sb, sg = NormActivationDoubleBackward.apply(
                ctx.op, x, bias, g, rx, rb, needs
            )
        else:
            sx, sb, sg = run_backward(ctx.op, 2, (x, bias, g, rx, rb), needs=needs)
        return None, sx, sb, sg, None


class NormActivationForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, bias):
        ctx.op = op
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.needed = ctx.needs_input_grad[1:3]
        ctx.set_materialize_grads(False)

        normalized_bias = bias
        if bias is not None and bias.dtype != x.dtype:
            normalized_bias = bias.to(dtype=x.dtype)
        output, saved_bias = (
            op.run(0, x, normalized_bias, save_bias=True, backward_needs=ctx.needed)
            if bias is not None
            else (op.run(0, x, None, backward_needs=ctx.needed), None)
        )

        if saved_bias is not None and not ctx.needed[1]:
            ctx.mark_non_differentiable(saved_bias)
        ctx.save_for_backward(x, saved_bias)
        # Saving this auxiliary output retains its bias edge without keeping
        # the mutable parameter itself or adding another autograd node.
        return output, saved_bias

    @staticmethod
    def backward(ctx, gradient, saved_bias_gradient):
        if gradient is None:
            db = None if saved_bias_gradient is None else saved_bias_gradient.to(ctx.bias_dtype)
            return None, None, db

        x, bias = ctx.saved_tensors
        enabled = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        ctx.op.backward_backend = "cuda_fused_recompute" if enabled else "cuda_fused"
        if enabled:
            dx, db = NormActivationBackward.apply(
                ctx.op, x, bias, gradient, ctx.needed
            )
        else:
            dx, db = run_backward(
                ctx.op, 1, (x, bias, gradient), needs=ctx.needed, grad_mode=False
            )

        if db is not None:
            db = db.to(ctx.bias_dtype)
        if saved_bias_gradient is not None:
            auxiliary = saved_bias_gradient.to(ctx.bias_dtype)
            db = auxiliary if db is None else db + auxiliary
        return None, dx, db


def build_norm_activation_metadata(
    activation_or_irreps,
    scalar_nonlinearity=None,
    normalize=True,
    epsilon=None,
    bias=False,
    *,
    dtype=torch.float32,
    activation_spec=None,
):
    official = hasattr(activation_or_irreps, "scalar_multiplier") and hasattr(
        activation_or_irreps, "norm"
    )
    if official:
        irreps = Irreps(activation_or_irreps.irreps_in)
        scalar_nonlinearity = activation_or_irreps.scalar_nonlinearity
        normalize, epsilon, bias = (
            activation_or_irreps.normalize,
            activation_or_irreps.epsilon,
            activation_or_irreps.bias,
        )
    else:
        irreps = Irreps(activation_or_irreps)

    if epsilon is None and normalize:
        epsilon = 1e-8
    elif epsilon is not None and not normalize:
        raise ValueError("epsilon and normalize = False don't make sense together")
    elif not epsilon > 0:
        raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive")
    if scalar_nonlinearity is None and activation_spec is None:
        raise TypeError("NormActivation requires a scalar activation")

    spec = {**(parse_activation(scalar_nonlinearity) if activation_spec is None else activation_spec),
            "cst": 1.0, "is_id": True}
    meta = build_norm_metadata(irreps, True, dtype=dtype)
    meta.update(
        kind="norm_activation",
        irreps_in=irreps,
        irreps_out=irreps,
        out_dim=irreps.dim,
        bias=bool(bias),
        normalize=bool(normalize),
        epsilon=epsilon,
        eps_squared=epsilon * epsilon if epsilon is not None else 0.0,
        activation=spec,
    )
    return meta, official


IRREPS_MISSING = object()


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def prepare_norm_activation(irreps_in, activation, normalize, epsilon, bias, dtype, index_bits):
    if activation is None:
        raise TypeError("NormActivation requires a scalar activation")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("NormActivation supports float16, bfloat16, float32 and float64")
    if index_bits not in (32, 64):
        raise ValueError("index_bits must be 32 or 64")
    metadata, _ = build_norm_activation_metadata(
        irreps_in, None, normalize, epsilon, bias, dtype=dtype,
        activation_spec=dict(kind=activation[0], params=dict(activation[1])),
    )
    metadata["norm_squared"] = metadata["epsilon"] is not None
    op = NormActivationKernels(
        meta=metadata, forward_source=make_forward_source,
        compute_dtype=dtype, index_bits=index_bits,
        kernels=OrderedDict(), programs=OrderedDict(),
    )
    # Dynamo needs a materialized attribute on the constant factory result.
    op.registered_forward = register_ops(metadata, index_bits, dtype)
    return op


def norm_activation(
    features, irreps_in, scalar_nonlinearity, normalize=True, epsilon=None, bias=None,
    *, config=None,
):
    """Apply a scalar activation to irrep norms with an explicit optional bias tensor."""
    op = config
    if op is None:
        op = prepare_norm_activation(
            irreps_key(irreps_in), activation_config(scalar_nonlinearity), normalize, epsilon, bias is not None,
            features.dtype, 32,
        )
    op.backward_backend = None
    if features.ndim == 0 or features.shape[-1] != op.meta["in_dim"]:
        raise ValueError("NormActivation feature dimension does not match irreps")
    native = features.is_cuda and features.dtype in NATIVE_DTYPES
    if native and bias is not None:
        native = not (
            bias.shape != (op.meta["bias_dim"],)
            or (bias.dtype == torch.float64 and features.dtype == torch.float32)
        )
    if not native or (features.dtype == torch.float32 and torch.is_autocast_enabled("cuda")):
        op.last_execution_backend = "torch_compat"
        op.backward_backend = "torch_compat"
        return norm_activation_torch(features, op.meta, bias)

    op.last_execution_backend = "cuda_fused"
    if is_compiling():
        if bias is not None and bias.dtype != features.dtype:
            bias = bias.to(dtype=features.dtype)
        name = op.registered_forward if op.compute_dtype == features.dtype else None
        if name is None:
            metadata = {key: value for key, value in op.meta.items()
                        if key not in ("irreps_in", "irreps_out", "dtype")}
            name = register_ops(metadata, op.index_bits, features.dtype)
        return getattr(torch.ops.lunaeq, name).default(features, bias)[0]
    if (
        op.meta["paths"]
        and not torch.is_inference_mode_enabled()
        and torch.is_grad_enabled()
        and (features.requires_grad or (bias is not None and bias.requires_grad))
    ):
        result, _ = NormActivationForward.apply(op, features, bias)
        return result
    with torch.no_grad():
        if bias is not None and bias.dtype != features.dtype:
            bias = bias.to(dtype=features.dtype)
    return op.run(0, features, bias)


__all__ = ['norm_activation', 'build_norm_activation_metadata']
