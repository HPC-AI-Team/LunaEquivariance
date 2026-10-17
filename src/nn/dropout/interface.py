"""Equivariant Dropout function, native dispatch and compact-mask autograd."""

from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256

import torch

from ...o3.irreps import PublicIrreps as Irreps, irreps_key
from .kernels.forward import load_dropout_kernel
from .kernels.utils import MAX_DIMS
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling


UNSET = object()


@torch.compiler.assume_constant_result
def register_ops(groups):
    suffix = sha256(repr(groups).encode()).hexdigest()[:24]
    name = f"dropout_{suffix}_forward"
    with REGISTER_OPERATOR_LOCK:
        if hasattr(torch.ops.lunaeq, name):
            return name
        copies = sum(mul for mul, _ in groups)

        @lru_cache(maxsize=128)
        def runtime(dtype, device):
            major, minor = torch.cuda.get_device_capability(device)
            return load_dropout_kernel(groups, dtype, f"{major}{minor}")

        @torch.library.custom_op(f"lunaeq::dropout_{suffix}_forward", mutates_args=(),
                                 tags=(torch.Tag.nondeterministic_seeded,))
        def dropout_forward_op(x: torch.Tensor, probability: float) -> tuple[torch.Tensor, torch.Tensor]:
            return runtime(x.dtype, x.device).launch(x, probability)

        @dropout_forward_op.register_fake
        def dropout_forward_fake(x, probability):
            return x.new_empty(x.shape), x.new_empty((x.shape[0], copies), dtype=torch.uint8)

        @torch.library.custom_op(f"lunaeq::dropout_{suffix}_backward", mutates_args=())
        def dropout_backward_op(direction: torch.Tensor, state: torch.Tensor,
                               probability: float) -> torch.Tensor:
            return runtime(direction.dtype, direction.device).apply_mask(direction, state, probability)

        @dropout_backward_op.register_fake
        def dropout_backward_fake(direction, state, probability):
            return direction.new_empty(direction.shape)

        def dropout_forward_setup(ctx, inputs, output):
            _, ctx.probability = inputs
            _, state = output
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(state)
            ctx.save_for_backward(state)

        def dropout_forward_backward(ctx, direction, state_gradient):
            (state,) = ctx.saved_tensors
            value = None if direction is None else dropout_backward_op(direction, state, ctx.probability)
            return value, None

        def dropout_backward_setup(ctx, inputs, output):
            _, state, ctx.probability = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(state)

        def dropout_backward_backward(ctx, direction):
            (state,) = ctx.saved_tensors
            value = None if direction is None else dropout_backward_op(direction, state, ctx.probability)
            return value, None, None

        dropout_forward_op.register_autograd(dropout_forward_backward, setup_context=dropout_forward_setup)
        dropout_backward_op.register_autograd(dropout_backward_backward, setup_context=dropout_backward_setup)
        return name


class DropoutBackward(torch.autograd.Function):
    """Backward, double-backward and triple-backward reuse the fixed mask kernel."""

    @staticmethod
    def forward(ctx, direction, state, probability, runtime):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(state)
        ctx.probability = probability
        ctx.runtime = runtime
        return runtime.apply_mask(direction, state, probability)

    @staticmethod
    def backward(ctx, direction):
        if direction is None:
            return None, None, None, None

        (state,) = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            value = DropoutBackward.apply(direction, state, ctx.probability, ctx.runtime)
        else:
            value = ctx.runtime.apply_mask(direction, state, ctx.probability)
        return value, None, None, None


class DropoutFunction(torch.autograd.Function):
    """Sample once, retaining only one decision byte for each irrep copy."""

    @staticmethod
    def forward(ctx, x, probability, runtime):
        output, state = runtime.launch(x, probability)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(state)
        ctx.probability = probability
        ctx.runtime = runtime
        return output

    @staticmethod
    def backward(ctx, direction):
        if direction is None:
            return None, None, None

        (state,) = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            value = DropoutBackward.apply(direction, state, ctx.probability, ctx.runtime)
        else:
            value = ctx.runtime.apply_mask(direction, state, ctx.probability)
        return value, None, None


def layout_metadata(irreps):
    return dict(
        _groups=tuple((int(mul), int(ir.dim)) for mul, ir in irreps),
        _feature_width=irreps.dim,
        _group_count=len(irreps),
        _max_multiplicity=max((mul for mul, _ in irreps), default=0),
    )


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def functional_plan(irreps):
    irreps = Irreps(irreps)
    return dict(
        irreps=irreps, **layout_metadata(irreps),
        _native_launches=OrderedDict(), _last_native_key=None, _last_native_runtime=None,
    )


def dropout(x, irreps, p, training=True, *, config=None):
    """Drop irrep copies; the functional cache owns static layout metadata only."""
    if not training:
        if config is not None:
            config["last_execution_backend"] = "identity"
        return x
    if config is None:
        key = irreps_key(irreps)
        config = functional_plan(key)
    width = config["_feature_width"]
    dtype = x.dtype
    if (
        x.is_cuda
        and x.layout == torch.strided
        and dtype in (torch.float32, torch.float64)
        and 2 <= x.ndim <= MAX_DIMS
        and x.shape[-1] == width
        and config["_group_count"] <= 128
        and x.shape[0] * config["_max_multiplicity"] * (4 if dtype == torch.float32 else 8)
        <= (1 << 31) - 1
        and 0 < width <= (1 << 31) - 1
        and isinstance(p, (float, int))
        and p == p
    ):
        if is_compiling():
            probability = 0.0 if p <= 0 else 1.0 if p >= 1 else float(p)
            return getattr(torch.ops.lunaeq, register_ops(config["_groups"])).default(x, probability)[0]
        irreps = config['irreps']
        key = dtype, x.device
        if key == config["_last_native_key"]:
            runtime = config["_last_native_runtime"]
        else:
            cache_key = irreps, *key
            launches = config["_native_launches"]
            runtime = launches.get(cache_key, UNSET)
            if runtime is UNSET:
                major, minor = torch.cuda.get_device_capability(x.device)
                if major >= 8:
                    runtime = load_dropout_kernel(config["_groups"], dtype, f"{major}{minor}")
                else:
                    runtime = False
                launches[cache_key] = runtime
                if len(launches) > 16:
                    launches.popitem(last=False)
            config["_last_native_key"] = key
            config["_last_native_runtime"] = runtime
        if runtime is not False:
            config["last_execution_backend"] = "cuda_fused_recompute"
            probability = 0.0 if p <= 0 else 1.0 if p >= 1 else float(p)
            # Inference mode takes precedence over a nested enable_grad.
            if torch.is_inference_mode_enabled():
                with torch.no_grad():
                    return DropoutFunction.apply(x, probability, runtime)
            return DropoutFunction.apply(x, probability, runtime)

    config["last_execution_backend"] = "torch_compat"
    # This also preserves e3nn's broadcasting and error behavior for inputs
    # outside the documented (batch, ..., irreps.dim) native shape.
    batch = x.shape[0]
    noises = []
    for mul, dimension in config["_groups"]:
        noise = x.new_empty(batch, mul)
        if p >= 1:
            noise.fill_(0)
        elif p <= 0:
            noise.fill_(1)
        else:
            noise.bernoulli_(1 - p).div_(1 - p)
        noises.append(
            noise[:, :, None].expand(-1, -1, dimension).reshape(batch, mul * dimension)
        )
    noise = torch.cat(noises, dim=-1)
    while noise.dim() < x.dim():
        noise = noise[:, None]
    return x * noise


__all__ = ['dropout']
