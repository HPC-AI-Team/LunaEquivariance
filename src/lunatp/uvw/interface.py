"""UVW tensor products mix independent input, edge and output channels.

Sparse Clebsch-Gordan contractions feed shared or per-item weighted matrix
products. Kernel selection follows the input device and requested derivatives."""

from __future__ import annotations
from hashlib import sha256
from typing import TYPE_CHECKING
import torch
from ..utils.layout import select_indexing
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling
from .kernels.forward import run as run_forward
from .kernels.backward import run as run_backward


if TYPE_CHECKING:
    from ..interface import LunaTPUVW


@torch.compiler.assume_constant_result
def register_ops(op):
    key = sha256(repr((op.index_mode, op.compute_dtype, op.index_bits, op.meta)).encode()).hexdigest()
    with REGISTER_OPERATOR_LOCK:
        name = f"tp_{key}_forward"
        if hasattr(torch.ops.lunatp, name):
            return name

        def cuda_backward(order, inputs, needs):
            # CUTLASS vector loads require aligned contiguous offset views.
            inputs = tuple(value.contiguous() if value is not None else None for value in inputs)
            inputs = tuple(value.clone() if value is not None and value.data_ptr() % 16 else value for value in inputs)
            return run_backward(op, order, *inputs, needs=needs)

        @torch.library.custom_op(f"lunatp::tp_{key}_forward", mutates_args=())
        def tp_forward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            x, y, w = (value.clone() if value.data_ptr() % 16 else value for value in (x, y, w))
            return run_forward(op, x, y, w)

        @tp_forward.register_fake
        def tp_forward_fake(x, y, w):
            return x.new_empty((x.shape[0], op.meta["out_dim"]))

        @torch.library.custom_op(f"lunatp::tp_{key}_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def tp_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                        needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return cuda_backward(1, (x, y, w, g), needs)

        @tp_backward.register_fake
        def tp_backward_fake(x, y, w, g, needs):
            return tuple(torch.empty_like(value) if needed else None for value, needed in zip((x, y, w), needs))

        @torch.library.custom_op(f"lunatp::tp_{key}_double_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, Tensor? rx, Tensor? ry, Tensor? rw, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def tp_double_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                               rx: torch.Tensor | None, ry: torch.Tensor | None, rw: torch.Tensor | None, needs: list[bool]
                               ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return cuda_backward(2, (x, y, w, g, rx, ry, rw), needs)

        @tp_double_backward.register_fake
        def tp_double_backward_fake(x, y, w, g, rx, ry, rw, needs):
            return tuple(torch.empty_like(value) if needed else None for value, needed in zip((x, y, w, g), needs))

        @torch.library.custom_op(f"lunatp::tp_{key}_triple_backward", mutates_args=(), schema=
            "(Tensor x, Tensor y, Tensor w, Tensor g, Tensor? rx, Tensor? ry, Tensor? rw, Tensor? bx, Tensor? by, Tensor? bw, Tensor? bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def tp_triple_backward(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, g: torch.Tensor,
                               rx: torch.Tensor | None, ry: torch.Tensor | None, rw: torch.Tensor | None,
                               bx: torch.Tensor | None, by: torch.Tensor | None, bw: torch.Tensor | None, bg: torch.Tensor | None,
                               needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
                                                          torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
                                                          torch.Tensor | None]:
            return cuda_backward(3, (x, y, w, g, rx, ry, rw, bx, by, bw, bg), needs)

        @tp_triple_backward.register_fake
        def tp_triple_backward_fake(x, y, w, g, rx, ry, rw, bx, by, bw, bg, needs):
            return tuple(torch.empty_like(value) if needed else None
                         for value, needed in zip((x, y, w, g, x, y, w), needs))

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*(inputs if len(inputs) == 3 else inputs[:-1]))
            ctx.set_materialize_grads(False)

        def forward_backward(ctx, g):
            x, y, w = ctx.saved_tensors
            needs = list(ctx.needs_input_grad[:3])
            gradients = tp_backward(x, y, w, g, needs)
            return gradients

        def backward_setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs[:4])
            ctx.set_materialize_grads(False)

        def backward_backward(ctx, rx, ry, rw):
            x, y, w, g = ctx.saved_tensors
            gradients = tp_double_backward(x, y, w, g, rx, ry, rw, list(ctx.needs_input_grad[:4]))
            return *gradients, None

        def double_backward_backward(ctx, bx, by, bw, bg):
            x, y, w, g, rx, ry, rw = ctx.saved_tensors
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


class UVWDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w, g, rx, ry, rw):
        ctx.op = op
        ctx.save_for_backward(x, y, w, g, rx, ry, rw)
        ctx.set_materialize_grads(False)
        return run_backward(
            op, 2, x, y, w, g, rx, ry, rw, needs=ctx.needs_input_grad[1:5]
        )

    @staticmethod
    def backward(ctx, bx, by, bw, bg):
        x, y, w, g, rx, ry, rw = ctx.saved_tensors
        bx, by, bw, bg = (value.contiguous() if value is not None else None for value in (bx, by, bw, bg))
        bx, by, bw, bg = (value.clone() if value is not None and value.data_ptr() % 16 else value for value in (bx, by, bw, bg))
        gradients = run_backward(
            ctx.op, 3, x, y, w, g, rx, ry, rw, bx, by, bw, bg, needs=ctx.needs_input_grad[1:]
        )
        return (None, *gradients)


class UVWBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w, g):
        ctx.op = op
        ctx.save_for_backward(x, y, w, g)
        ctx.set_materialize_grads(False)
        return run_backward(op, 1, x, y, w, g, needs=ctx.needs_input_grad[1:4])

    @staticmethod
    def backward(ctx, rx, ry, rw):
        x, y, w, g = ctx.saved_tensors
        rx, ry, rw = (value.contiguous() if value is not None else None for value in (rx, ry, rw))
        rx, ry, rw = (value.clone() if value is not None and value.data_ptr() % 16 else value for value in (rx, ry, rw))
        if torch.is_grad_enabled():
            gradients = UVWDoubleBackward.apply(ctx.op, x, y, w, g, rx, ry, rw)
        else:
            gradients = run_backward(
                ctx.op, 2, x, y, w, g, rx, ry, rw, needs=ctx.needs_input_grad[1:]
            )
        return (None, *gradients)


class UVWForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w):
        ctx.op = op
        ctx.save_for_backward(x, y, w)
        return run_forward(op, x, y, w)

    @staticmethod
    def backward(ctx, g):
        x, y, w = ctx.saved_tensors
        if g is None:
            g = x.new_zeros((x.shape[0], ctx.op.meta["out_dim"]))
        else:
            g = g.contiguous()
            if g.data_ptr() % 16:
                g = g.clone()
        if torch.is_grad_enabled():
            gradients = UVWBackward.apply(ctx.op, x, y, w, g)
        else:
            gradients = run_backward(
                ctx.op, 1, x, y, w, g, needs=ctx.needs_input_grad[1:]
            )
        return (None, *gradients)


def lunatp_uvw_func(
    op: LunaTPUVW,
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute UVW tensor products with derivatives through third order.

    Args:
        op: Independent path configuration and kernel cache.
        x: Features of shape [items, in1_dim].
        edge_attr: Attributes of shape [items, in2_dim].
        weight: External weights [items, weight_numel]. Shared mode takes a one-dimensional weight tensor.

    Returns:
        Output features of shape [items, out_dim].

    Inputs share one CUDA device and the operator's FP32/FP64 dtype.
    Aligned contiguous inputs are reused."""
    shared = op.weight_mode == "shared"
    if weight is None:
        raise TypeError("UVW requires a weight tensor")
    if (
        x.dtype != op.compute_dtype
        or edge_attr.dtype != op.compute_dtype
        or weight.dtype != op.compute_dtype
    ):
        raise ValueError(
            f"UVW inputs must use the initialized dtype {op.compute_dtype}"
        )

    x_shape = x.shape
    edge_shape = edge_attr.shape
    weight_shape = weight.shape
    if len(x_shape) != 2 or len(edge_shape) != 2:
        raise ValueError("UVW x and edge_attr must be matrices")
    meta = op.meta
    if x_shape[1] != meta["in1_dim"] or edge_shape != (
        x_shape[0],
        meta["in2_dim"],
    ):
        raise ValueError("UVW input shapes do not match the configuration")
    expected = (meta["weight_numel"],) if shared else (x_shape[0], meta["weight_numel"])
    if weight_shape != expected:
        raise ValueError(
            f"UVW expected weight shape {expected}, got {tuple(weight_shape)}"
        )

    # A contiguous slice can still start between float4 boundaries.
    x = x.contiguous()
    edge_attr = edge_attr.contiguous()
    weight = weight.contiguous()
    compiling = is_compiling()
    if not compiling:
        if x.data_ptr() % 16:
            x = x.clone()
        if edge_attr.data_ptr() % 16:
            edge_attr = edge_attr.clone()
        if weight.data_ptr() % 16:
            weight = weight.clone()
    if x_shape[0] > op.index_item_limit:
        op = select_indexing(op, x_shape[0], x_shape[0])
    if compiling:
        return getattr(torch.ops.lunatp, op.register_ops()).default(x, edge_attr, weight)
    return UVWForward.apply(op, x, edge_attr, weight)


__all__ = ["lunatp_uvw_func"]
