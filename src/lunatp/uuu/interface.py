"""UUU tensor products preserve one matching channel across both inputs and the output.

Both input irreps contract independently per item and channel, using shared or per-item
weights. CUDA kernels and autograd support FP32/FP64 through third order."""

from __future__ import annotations
from hashlib import sha256
from math import prod
from typing import TYPE_CHECKING
import torch
from ..utils.layout import select_indexing
from ..utils.runtime import prepare_gradient
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling
from .kernels.forward import run as run_forward
from .kernels.backward import run as run_backward


if TYPE_CHECKING:
    from ..interface import LunaTPUUU


@torch.compiler.assume_constant_result
def register_ops(op):
    key = sha256(repr((op.index_mode, op.compute_dtype, op.index_bits, op.meta)).encode()).hexdigest()
    with REGISTER_OPERATOR_LOCK:
        name = f"tp_{key}_forward"
        if hasattr(torch.ops.lunatp, name):
            return name

        @torch.library.custom_op(f"lunatp::tp_{key}_forward", mutates_args=())
        def tp_forward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return run_forward(op, x, y, w)

        @tp_forward.register_fake
        def tp_forward_fake(x, y, w):
            return x.new_empty((x.shape[0], op.meta["out_dim"]))

        @torch.library.custom_op(f"lunatp::tp_{key}_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, bool[] needs, bool only_x=False) -> (Tensor?, Tensor?, Tensor?)")
        def tp_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                        needs: list[bool], only_x: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            if only_x:
                return run_backward(op, 1, y, w, g, only_x=True), None, None
            return run_backward(op, 1, x, y, w, g, needs=needs)

        @tp_backward.register_fake
        def tp_backward_fake(x, y, w, g, needs, only_x=False):
            return tuple(torch.empty_like(value) if needed else None for value, needed in zip((x, y, w), needs))

        @torch.library.custom_op(f"lunatp::tp_{key}_double_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, Tensor rx, Tensor ry, Tensor rw, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def tp_double_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                               rx: torch.Tensor, ry: torch.Tensor, rw: torch.Tensor, needs: list[bool]
                               ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return run_backward(op, 2, x, y, w, g, rx, ry, rw, needs=needs)

        @tp_double_backward.register_fake
        def tp_double_backward_fake(x, y, w, g, rx, ry, rw, needs):
            return tuple(torch.empty_like(value) if needed else None for value, needed in zip((x, y, w, g), needs))

        @torch.library.custom_op(f"lunatp::tp_{key}_triple_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, Tensor rx, Tensor ry, Tensor rw, Tensor bx, Tensor by, Tensor bw, Tensor bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def tp_triple_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                               rx: torch.Tensor, ry: torch.Tensor, rw: torch.Tensor,
                               bx: torch.Tensor, by: torch.Tensor, bw: torch.Tensor, bg: torch.Tensor,
                               needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
                                                          torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
                                                          torch.Tensor | None]:
            return run_backward(op, 3, x, y, w, g, rx, ry, rw, bx, by, bw, bg, needs=needs)

        @tp_triple_backward.register_fake
        def tp_triple_backward_fake(x, y, w, g, rx, ry, rw, bx, by, bw, bg, needs):
            return tuple(torch.empty_like(value) if needed else None for value, needed in zip((x, y, w, g, rx, ry, rw), needs))

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*(inputs if len(inputs) == 3 else inputs[:-1]))
            ctx.set_materialize_grads(False)

        def forward_backward(ctx, g):
            x, y, w = ctx.saved_tensors
            needs = list(ctx.needs_input_grad[:3])
            only_x = not torch.is_grad_enabled() and needs == [True, False, False]
            gradients = tp_backward(x, y, w, g.contiguous(), needs, only_x)
            return gradients

        def backward_setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs[:4])
            ctx.set_materialize_grads(False)

        def backward_backward(ctx, rx, ry, rw):
            x, y, w, g = ctx.saved_tensors
            rx, ry, rw = (prepare_gradient(value, template) for value, template in zip((rx, ry, rw), (x, y, w)))
            gradients = tp_double_backward(x, y, w, g, rx, ry, rw, list(ctx.needs_input_grad[:4]))
            return *gradients, None, None

        def double_backward_backward(ctx, bx, by, bw, bg):
            x, y, w, g, rx, ry, rw = ctx.saved_tensors
            bx, by, bw, bg = (prepare_gradient(value, template) for value, template in zip((bx, by, bw, bg), (x, y, w, g)))
            gradients = tp_triple_backward(x, y, w, g, rx, ry, rw, bx, by, bw, bg,
                                           list(ctx.needs_input_grad[:7]))
            return *gradients, None

        # The CUDA triple backward is terminal for AOT autograd.
        op.triple_backward_library = torch.library.Library("lunatp", "FRAGMENT")
        for dispatch_key in ("AutogradCPU", "AutogradCUDA"):
            op.triple_backward_library.impl(
                f"tp_{key}_triple_backward", torch.library.fallthrough_kernel, dispatch_key
            )

        tp_forward.register_autograd(forward_backward, setup_context=setup_context)
        tp_backward.register_autograd(backward_backward, setup_context=backward_setup_context)
        tp_double_backward.register_autograd(double_backward_backward, setup_context=setup_context)

        return name


class UUUDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, weight, grad_out, rx, ry, rw):
        ctx.op = op
        ctx.save_for_backward(x, y, weight, grad_out, rx, ry, rw)
        ctx.set_materialize_grads(False)
        return run_backward(
            op, 2, x, y, weight, grad_out, rx, ry, rw, needs=ctx.needs_input_grad[1:5]
        )

    @staticmethod
    def backward(ctx, bx, by, bw, bg):
        x, y, weight, grad_out, rx, ry, rw = ctx.saved_tensors
        # Missing incoming gradients contribute zero to third-order derivatives.
        bx = prepare_gradient(bx, x)
        by = prepare_gradient(by, y)
        bw = prepare_gradient(bw, weight)
        bg = prepare_gradient(bg, grad_out)
        gradients = run_backward(
            ctx.op, 3, x, y, weight, grad_out, rx, ry, rw, bx, by, bw, bg, needs=ctx.needs_input_grad[1:]
        )
        return (None, *gradients)


class UUUBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, weight, grad_out):
        ctx.op = op
        ctx.save_for_backward(x, y, weight, grad_out)
        ctx.set_materialize_grads(False)
        return run_backward(
            op, 1, x, y, weight, grad_out, needs=ctx.needs_input_grad[1:4]
        )

    @staticmethod
    def backward(ctx, rx, ry, rw):
        x, y, weight, grad_out = ctx.saved_tensors
        rx = prepare_gradient(rx, x)
        ry = prepare_gradient(ry, y)
        rw = prepare_gradient(rw, weight)
        if torch.is_grad_enabled():
            gradients = UUUDoubleBackward.apply(ctx.op, x, y, weight, grad_out, rx, ry, rw)
        else:
            gradients = run_backward(
                ctx.op, 2, x, y, weight, grad_out, rx, ry, rw, needs=ctx.needs_input_grad[1:]
            )
        return (None, *gradients)


class UUUForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, weight):
        ctx.op = op
        ctx.save_for_backward(x, y, weight)
        return run_forward(op, x, y, weight)

    @staticmethod
    def backward(ctx, grad_out):
        x, y, weight = ctx.saved_tensors
        if grad_out is None:
            grad_out = x.new_zeros((x.shape[0], ctx.op.meta["out_dim"]))
        else:
            grad_out = grad_out.contiguous()
        if torch.is_grad_enabled():
            gradients = UUUBackward.apply(ctx.op, x, y, weight, grad_out)
        else:
            needs = ctx.needs_input_grad[1:]
            if needs[0] and not needs[1] and not needs[2]:
                return (
                    None,
                    run_backward(ctx.op, 1, y, weight, grad_out, only_x=True),
                    None,
                    None,
                )
            gradients = run_backward(ctx.op, 1, x, y, weight, grad_out, needs=needs)
        return (None, *gradients)


def lunatp_uuu_func(
    op: LunaTPUUU,
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute UUU tensor products with broadcast batch dimensions.

    Inputs have shapes [..., in1_dim] and [..., in2_dim]. Shared weights
    have shape [weight_numel]; per-item weights have shape [..., weight_numel].
    Batch dimensions follow PyTorch broadcasting. Weights are always supplied explicitly. Weight-free products accept
    None. Inputs share one CUDA device and the operator's FP32/FP64 dtype.
    """
    shared = op.weight_mode == "shared"
    meta = op.meta
    if weight is None:
        if meta["weight_numel"] == 0:
            weight = x.new_empty((0,))
        else:
            raise TypeError("UUU requires a weight tensor")
    if any(value.dtype is not op.compute_dtype for value in (x, edge_attr, weight)):
        raise ValueError(f"UUU inputs must use {op.compute_dtype}")
    if (
        x.ndim == 0
        or x.shape[-1] != meta["in1_dim"]
        or edge_attr.ndim == 0
        or edge_attr.shape[-1] != meta["in2_dim"]
    ):
        raise ValueError("UUU input feature dimensions do not match the configuration")
    if weight.ndim == 0 or weight.shape[-1] != meta["weight_numel"]:
        raise ValueError(
            f"""UUU expected last weight dimension {meta["weight_numel"]}"""
        )
    if shared and weight.ndim != 1:
        raise ValueError("Shared UUU weights must have shape [weight_numel]")

    batches = [x.shape[:-1], edge_attr.shape[:-1]]
    if not shared:
        batches.append(weight.shape[:-1])
    if all(batch == batches[0] for batch in batches[1:]):
        batch_shape = batches[0]
    else:
        try:
            batch_shape = torch.broadcast_shapes(*batches)
        except RuntimeError as error:
            raise ValueError(
                "UUU input batch dimensions are not broadcastable"
            ) from error

    num_items = prod(batch_shape)
    if x.shape != (num_items, meta["in1_dim"]):
        x = x.expand(*batch_shape, meta["in1_dim"]).reshape(num_items, meta["in1_dim"])
    if edge_attr.shape != (num_items, meta["in2_dim"]):
        edge_attr = edge_attr.expand(*batch_shape, meta["in2_dim"]).reshape(
            num_items, meta["in2_dim"]
        )
    if not shared and weight.shape != (num_items, meta["weight_numel"]):
        weight = weight.expand(*batch_shape, meta["weight_numel"]).reshape(
            num_items, meta["weight_numel"]
        )
    x = x.contiguous()
    edge_attr = edge_attr.contiguous()
    weight = weight.contiguous()

    if num_items > op.index_item_limit:
        op = select_indexing(op, num_items, num_items)
    if is_compiling():
        result = getattr(torch.ops.lunatp, op.register_ops()).default(x, edge_attr, weight)
    elif not torch.is_grad_enabled() or not (
        x.requires_grad or edge_attr.requires_grad or weight.requires_grad
    ):
        result = run_forward(op, x, edge_attr, weight)
    else:
        result = UUUForward.apply(op, x, edge_attr, weight)
    return (
        result
        if len(batch_shape) == 1
        else result.reshape(*batch_shape, meta["out_dim"])
    )


__all__ = ["lunatp_uuu_func"]
