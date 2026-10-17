"""Per-irrep Norm with e3nn state and fused CUDA backward computations."""

from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256

import torch

from ...utils import is_compiling, REGISTER_OPERATOR_LOCK

from ..irreps import PublicIrreps as Irreps, irreps_key
from .kernels.utils import NormKernels, norm_product, SUPPORTED_DTYPES, NATIVE_DTYPES


def continue_triple_backward(op, x, g, rx, bx, bg, needs):
    """Replay the original graph when triple-backward must remain differentiable."""
    with torch.enable_grad():
        # Local partials must stop at these views even when G/R are nonleaves
        # with their own path to X. The outer engine joins those paths once.
        x, g, rx = x.view_as(x), g.view_as(g), rx.view_as(rx)
        squares = []
        for path in op.meta["paths"]:
            width = path["copies"] * path["components"]
            section = x[..., path["input_offset"]:path["input_offset"] + width].reshape(
                *x.shape[:-1], path["copies"], path["components"]
            )
            squares.append(norm_product(path["components"], section, section).squeeze(-1))
        square = torch.cat(squares, -1)
        y = square if op.meta["squared"] else square.relu().sqrt()

        (first,) = torch.autograd.grad(y, (x,), g, create_graph=True, retain_graph=True)
        second_targets = tuple(value for value in (x, g) if value.requires_grad)
        second_values = iter(
            torch.autograd.grad(
                first, second_targets, rx, create_graph=True, retain_graph=True, allow_unused=True
            )
        )
        second = tuple(next(second_values) if value.requires_grad else None for value in (x, g))

        active = tuple(
            (value, seed)
            for value, seed in zip(second, (bx, bg))
            if value is not None and seed is not None and value.requires_grad
        )
        targets = tuple(value for value, needed in zip((x, g, rx), needs) if needed)
        if active and targets:
            values = torch.autograd.grad(
                tuple(value for value, _ in active),
                targets,
                tuple(seed for _, seed in active),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            values = (None,) * len(targets)
        iterator = iter(values)
        return tuple(next(iterator) if needed else None for needed in needs)


class NormDoubleBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op, x, g, rx, needs):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, g, rx)
        sx, sg = op.run(2, x, g, rx, needs=needs)
        if (
            sx is not None
            and op.meta["squared"]
            and not (ctx.needs_input_grad[2] or ctx.needs_input_grad[3])
        ):
            ctx.mark_non_differentiable(sx)
        if sg is not None and not (ctx.needs_input_grad[1] or ctx.needs_input_grad[3]):
            ctx.mark_non_differentiable(sg)
        return sx, sg

    @staticmethod
    def backward(ctx, bx, bg):
        if bx is None and bg is None:
            return (None,) * 5

        x, g, rx = ctx.saved_tensors
        needs = ctx.needs_input_grad[1:4]
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            ctx.op.backward_backend = "torch_compat_higher_than_three"
            tx, tg, trx = continue_triple_backward(ctx.op, x, g, rx, bx, bg, needs)
        else:
            tx, tg, trx = ctx.op.run(3, x, g, rx, bx, bg, needs=needs)
        return None, tx, tg, trx, None


class NormBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op, x, g):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, g)
        dx = op.run(1, x, g)
        return dx

    @staticmethod
    def backward(ctx, rx):
        if rx is None:
            return (None,) * 3

        x, g = ctx.saved_tensors
        needs = ctx.needs_input_grad[1:3]
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            sx, sg = NormDoubleBackward.apply(ctx.op, x, g, rx, needs)
        else:
            sx, sg = ctx.op.run(2, x, g, rx, needs=needs)
        return None, sx, sg


class NormForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op, x):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x)
        return op.run(0, x)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None
        (x,) = ctx.saved_tensors
        ctx.op.backward_backend = "cuda_fused"
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            dx = NormBackward.apply(ctx.op, x, g)
        else:
            dx = ctx.op.run(1, x, g)
        return None, dx


def build_norm_metadata(norm_or_irreps, squared=False, *, dtype=torch.float32):
    official = hasattr(norm_or_irreps, "tp") and hasattr(norm_or_irreps, "squared")
    irreps = Irreps(norm_or_irreps.irreps_in if official else norm_or_irreps).simplify()
    if official:
        squared = norm_or_irreps.squared

    paths = []
    offset = scalar = 0
    for copies, irrep in irreps:
        if copies:
            paths.append(
                dict(copies=copies, components=irrep.dim, input_offset=offset, scalar_offset=scalar)
            )
        offset += copies * irrep.dim
        scalar += copies
    return dict(
        kind="norm",
        irreps_in=irreps,
        irreps_out=Irreps([(scalar, "0e")]).simplify(),
        in_dim=offset,
        out_dim=scalar,
        bias_dim=scalar,
        paths=tuple(paths),
        squared=bool(squared),
        bias=False,
        dtype=dtype,
    )


IRREPS_MISSING = object()


@lru_cache(maxsize=128)
def prepare_norm(irreps_in, squared, dtype, index_bits):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("Norm supports float16, bfloat16, float32 and float64")
    if index_bits not in (32, 64):
        raise ValueError("index_bits must be 32 or 64")
    return NormKernels(
        meta=build_norm_metadata(irreps_in, squared, dtype=dtype),
        compute_dtype=dtype, index_bits=index_bits,
        kernels=OrderedDict(),
    )


def norm(features, irreps_in, squared=False, *, config=None):
    """Compute per-irrep norms without constructing a Module.

    ``config`` optionally reuses prepared metadata and CUDA launch caches.
    It contains no parameters or Module references.
    """
    native = features.is_cuda and features.dtype in NATIVE_DTYPES
    if is_compiling() and native:
        irreps = irreps_key(irreps_in if config is None else config.meta["irreps_in"])
        if not isinstance(irreps, str):
            irreps = " + ".join(
                f"{mul}x{ir}" if isinstance(ir, str) else f"{mul}x{ir[0]}{'e' if ir[1] == 1 else 'o'}"
                for mul, ir in irreps
            )
        name = register_ops(
            irreps, bool(squared) if config is None else config.meta["squared"],
            32 if config is None else config.index_bits, features.dtype,
        )
        return getattr(torch.ops.lunaeq, name).default(features)
    op = config
    if op is None:
        if not isinstance(irreps_in, (str, Irreps)):
            irreps_in = Irreps(irreps_in)
        op = prepare_norm(irreps_in, bool(squared), features.dtype, 32)
    op.backward_backend = None
    if features.ndim == 0 or features.shape[-1] != op.meta["in_dim"]:
        raise ValueError("Norm feature dimension does not match irreps")
    if not native:
        op.last_execution_backend = "torch_compat"
        op.backward_backend = "torch_compat"
        outputs = []
        for path in op.meta["paths"]:
            width = path["copies"] * path["components"]
            section = features[..., path["input_offset"]:path["input_offset"] + width].reshape(
                *features.shape[:-1], path["copies"], path["components"]
            )
            outputs.append(norm_product(path["components"], section, section).squeeze(-1))
        result = torch.cat(outputs, -1) if outputs else features.new_zeros((*features.shape[:-1], 0))
        return result if op.meta["squared"] else result.relu().sqrt()

    op.last_execution_backend = "cuda_fused"
    if (
        op.meta["paths"]
        and torch.is_grad_enabled()
        and features.requires_grad
        and not torch.is_inference_mode_enabled()
    ):
        return NormForward.apply(op, features)
    return op.run(0, features)


__all__ = ['norm', 'build_norm_metadata']


@torch.compiler.assume_constant_result
def register_ops(irreps, squared, index_bits, dtype):
    """Bind static Norm metadata outside the compiled tensor graph."""
    with REGISTER_OPERATOR_LOCK:
        name = sha256(repr((irreps, squared, index_bits, dtype)).encode()).hexdigest()[:24]
        if hasattr(torch.ops.lunaeq, f"norm_{name}_forward"):
            return f"norm_{name}_forward"
        op = prepare_norm(irreps, squared, dtype, index_bits)

        @torch.library.custom_op(f"lunaeq::norm_{name}_forward", mutates_args=())
        def norm_forward(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 0 or x.shape[-1] != op.meta["in_dim"]:
                raise ValueError("Norm feature dimension does not match irreps")
            return op.run(0, x)

        @norm_forward.register_fake
        def norm_forward_fake(x):
            if x.ndim == 0 or x.shape[-1] != op.meta["in_dim"]:
                raise ValueError("Norm feature dimension does not match irreps")
            return x.new_empty((*x.shape[:-1], op.meta["out_dim"]))

        @torch.library.custom_op(f"lunaeq::norm_{name}_backward", mutates_args=())
        def norm_backward(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return op.run(1, x, g)

        @norm_backward.register_fake
        def norm_backward_fake(x, g):
            return torch.empty_like(x, memory_format=torch.contiguous_format)

        @torch.library.custom_op(f"lunaeq::norm_{name}_double_backward", mutates_args=(),
            schema="(Tensor x, Tensor g, Tensor rx, bool[] needs) -> (Tensor?, Tensor?)")
        def norm_double_backward(x: torch.Tensor, g: torch.Tensor, rx: torch.Tensor,
                                 needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            return op.run(2, x, g, rx, needs=needs)

        @norm_double_backward.register_fake
        def norm_double_backward_fake(x, g, rx, needs):
            if g.shape[-1] == 0:
                return None, None
            return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) if needed else None
                         for t, needed in zip((x, g), needs))

        @torch.library.custom_op(f"lunaeq::norm_{name}_triple_backward", mutates_args=(),
            schema="(Tensor x, Tensor g, Tensor rx, Tensor? bx, Tensor? bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def norm_triple_backward(x: torch.Tensor, g: torch.Tensor, rx: torch.Tensor,
                                 bx: torch.Tensor | None, bg: torch.Tensor | None,
                                 needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return op.run(3, x, g, rx, bx, bg, needs=needs)

        @norm_triple_backward.register_fake
        def norm_triple_backward_fake(x, g, rx, bx, bg, needs):
            if g.shape[-1] == 0:
                return None, None, None
            needs = (needs[0] and (not squared or bg is not None), needs[1] and bx is not None, needs[2])
            return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) if needed else None
                         for t, needed in zip((x, g, rx), needs))

        def norm_forward_backward(ctx, g):
            if g is None:
                return None
            (x,) = ctx.saved_tensors
            return norm_backward(x, g)

        def norm_backward_backward(ctx, rx):
            if rx is None:
                return None, None
            x, g = ctx.saved_tensors
            return norm_double_backward(x, g, rx, list(ctx.needs_input_grad))

        def norm_setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)
            ctx.set_materialize_grads(False)

        def norm_double_backward_setup(ctx, inputs, output):
            ctx.save_for_backward(*inputs[:3])
            ctx.set_materialize_grads(False)

        def norm_double_backward_backward(ctx, bx, bg):
            if bx is None and bg is None:
                return (None,) * 4
            x, g, rx = ctx.saved_tensors
            result = norm_triple_backward(x, g, rx, bx, bg, list(ctx.needs_input_grad[:3]))
            return *result, None

        norm_forward.register_autograd(norm_forward_backward, setup_context=norm_setup_context)
        norm_backward.register_autograd(norm_backward_backward, setup_context=norm_setup_context)
        norm_double_backward.register_autograd(norm_double_backward_backward, setup_context=norm_double_backward_setup)

        return f"norm_{name}_forward"
