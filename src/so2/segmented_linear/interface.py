"""Segmented mixed-weight linear maps with derivatives through third order."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.autograd.function import once_differentiable

from .kernels.backward import input_gradient, mixed_input_vjp, mixed_weight_vjp, segmented_vjp, use_gate_weight_vjp
from .kernels.double_backward import segmented_double_backward, segmented_hvp_input_direction
from .kernels.forward import mixed_forward, segmented_forward
from .kernels.triple_backward import segmented_triple_backward
from .kernels.utils import (
    grouped_device, grouped_weight_device, mix_weights, split_flattened,
    streamed_weight_device, use_grouped,
)


def autograd_segmented_linear(x, weights, coefficients, bias, sizes):
    """Apply segment-specific expert mixtures using PyTorch autograd.

    ``x`` is ``(N, ..., Cin)``, ``weights`` is ``(experts, Cout, Cin)``,
    and ``coefficients`` is ``(segments, experts)``. The tuple ``sizes``
    partitions the leading axis, sums to ``N``, and has one entry per
    segment. Optional shared ``bias`` is ``(Cout,)``. Return
    ``(N, ..., Cout)``. All floating tensors share device and dtype.
    Features, weights, mixing coefficients, and bias support derivatives
    through third order; segment sizes are fixed.
    """
    mixed = mix_weights(weights, coefficients)
    outputs = [
        torch.nn.functional.linear(local, local_mixed, bias)
        for local, local_mixed in zip(
            torch.split(x, sizes, dim=0),
            mixed.unbind(0),
            strict=True,
        )
    ]
    if len(outputs) == 1:
        return outputs[0].contiguous()
    return concat_with_split_backward(outputs, dim=0)


class SegmentedLinearInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, coefficients, bias, sizes):
        mixed = mix_weights(weights, coefficients)
        ctx.save_for_backward(mixed)
        ctx.sizes = sizes
        return segmented_forward(x, mixed, bias, sizes)

    @staticmethod
    def backward(ctx, dy):
        (mixed,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Frozen parameters still allow a force loss to differentiate the
            # input VJP through its cotangent. Avoid out= in this graph.
            return (
                concat_with_split_backward(
                    tuple(
                        torch.matmul(dy_s, W_s)
                        for dy_s, W_s in zip(
                            torch.split(dy, ctx.sizes, dim=0),
                            mixed.unbind(0),
                            strict=True,
                        )
                    ),
                    dim=0,
                ),
                None,
                None,
                None,
                None,
            )
        dx = torch.empty(
            (*dy.shape[:-1], mixed.shape[-1]),
            device=dy.device,
            dtype=dy.dtype,
        )
        for dy_s, W_s, dx_s in zip(
            split_flattened(dy, ctx.sizes),
            mixed.unbind(0),
            split_flattened(dx, ctx.sizes),
            strict=True,
        ):
            torch.mm(dy_s, W_s, out=dx_s)
        return dx, None, None, None, None


class SegmentedLinearBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, x, weights, coefficients, bias, sizes):
        gradients, intermediates = segmented_vjp(
            dy, x, weights, coefficients, bias, sizes, save_intermediates=True
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(dy, x, weights, coefficients, *intermediates)
        ctx.bias = bias
        ctx.sizes = sizes
        return gradients

    @staticmethod
    def backward(ctx, hx, hweights, hcoefficients, hbias):
        dy, x, weights, coefficients, mixed, dmixed = ctx.saved_tensors
        args = (
            dy, x, weights, coefficients, ctx.bias, ctx.sizes,
            hx, hweights, hcoefficients, hbias,
            mixed, dmixed,
        )
        gradients = (
            SegmentedLinearDoubleBackward.apply(*args)
            if torch.is_grad_enabled()
            else segmented_double_backward(*args)
        )
        return *gradients, None, None


class SegmentedLinearDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, dy, x, weights, coefficients, bias, sizes,
        hx, hweights, hcoefficients, hbias, mixed, dmixed,
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(dy, x, weights, coefficients, hx, hweights, hcoefficients, mixed, dmixed)
        ctx.sizes = sizes
        return segmented_double_backward(
            dy, x, weights, coefficients, bias, sizes,
            hx, hweights, hcoefficients, hbias, mixed, dmixed,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, by, bx, bweights, bcoefficients):
        dy, x, weights, coefficients, hx, hweights, hcoefficients, mixed, dmixed = ctx.saved_tensors
        grad_primals, grad_seeds = segmented_triple_backward(
            dy, x, weights, coefficients, ctx.sizes, hx, hweights, hcoefficients,
            by, bx, bweights, bcoefficients, mixed, dmixed, ctx.needs_input_grad[6:10],
        )
        return *grad_primals, None, None, *grad_seeds, None, None


class SegmentedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, coefficients, bias, sizes):
        mixed = mix_weights(weights, coefficients)
        ctx.save_for_backward(x, weights, coefficients)
        ctx.bias = bias
        ctx.sizes = sizes
        return segmented_forward(x, mixed, bias, sizes)

    @staticmethod
    def backward(ctx, dy):
        x, weights, coefficients = ctx.saved_tensors
        arguments = dy, x, weights, coefficients, ctx.bias, ctx.sizes
        gradients = (
            SegmentedLinearBackward.apply(*arguments)
            if torch.is_grad_enabled()
            else segmented_vjp(*arguments)
        )
        return *gradients, None


class SegmentedLinearForceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, weights, coefficients, sizes, mixed):
        if mixed is None:
            mixed = mix_weights(weights, coefficients)
        if use_grouped(dy, mixed, sizes, input_gradient=True):
            dx = input_gradient(dy, mixed, sizes)
        else:
            dx = torch.empty(
                (*dy.shape[:-1], mixed.shape[-1]),
                device=dy.device,
                dtype=dy.dtype,
            )
            for dy_s, W_s, dx_s in zip(
                split_flattened(dy, sizes),
                mixed.unbind(0),
                split_flattened(dx, sizes),
                strict=True,
            ):
                torch.mm(dy_s, W_s, out=dx_s)
        ctx.save_for_backward(dy, weights, coefficients, mixed)
        ctx.sizes = sizes
        return dx

    @staticmethod
    @once_differentiable
    def backward(ctx, hx):
        dy, weights, coefficients, mixed = ctx.saved_tensors
        d_dy, _, dweights, dcoefficients = segmented_hvp_input_direction(
            dy, weights, coefficients, ctx.sizes, hx, mixed
        )
        return d_dy, dweights, dcoefficients, None, None


def reuse_force_mix(weights):
    return (
        weights.is_cuda
        and weights.dtype == torch.float32
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.is_autocast_enabled("cuda")
    )


class SegmentedLinearForce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, coefficients, bias, sizes):
        mixed = mix_weights(weights, coefficients)
        # Cache the value. The force VJP still computes gradients of the mixture.
        retained = mixed if reuse_force_mix(weights) else None
        ctx.save_for_backward(x, weights, coefficients, retained)
        ctx.bias = bias
        ctx.sizes = sizes
        return segmented_forward(x, mixed, bias, sizes)

    @staticmethod
    def backward(ctx, dy):
        x, weights, coefficients, mixed = ctx.saved_tensors
        if mixed is not None and not reuse_force_mix(weights):
            mixed = None
        if torch.is_grad_enabled():
            dx = SegmentedLinearForceBackward.apply(
                dy, weights, coefficients, ctx.sizes, mixed
            )
            return dx, None, None, None, None
        gradients = segmented_vjp(
            dy, x, weights, coefficients, ctx.bias, ctx.sizes, mixed=mixed
        )
        return *gradients, None


class MixedSegmentVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, x, mixed, sizes, offsets, grouped, need_x, need_weight, gate_cotangent_offsets=None):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(dy, x if need_weight else None, mixed, offsets, gate_cotangent_offsets)
        ctx.sizes, ctx.grouped = sizes, grouped
        dx = mixed_input_vjp(dy, mixed, sizes, grouped) if need_x else None
        dw = mixed_weight_vjp(dy, x, mixed, sizes, offsets, gate_cotangent_offsets) if need_weight else None
        return dx, dw

    @staticmethod
    @once_differentiable
    def backward(ctx, hx, hw):
        dy, x, mixed, offsets, gate_cotangent_offsets = ctx.saved_tensors
        sizes, grouped = ctx.sizes, ctx.grouped
        need_dy, need_x, need_weight = ctx.needs_input_grad[:3]
        ddy = dx = dw = None
        if hx is not None:
            hx = hx.resolve_neg().contiguous()
            if need_dy:
                ddy = mixed_forward(hx, mixed, sizes, grouped)
            if need_weight:
                dw = mixed_weight_vjp(dy, hx, mixed, sizes, offsets, gate_cotangent_offsets)
        if hw is not None:
            hw = hw.resolve_neg().contiguous()
            if need_dy:
                ddy = mixed_forward(x, hw, sizes, grouped, ddy)
            if need_x:
                dx = mixed_input_vjp(dy, hw, sizes, grouped)
        return ddy, dx, dw, None, None, None, None, None, None


class MixedSegmentProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mixed, bias, sizes, offsets, grouped, force_only, gate_cotangent_offsets=None):
        ctx.save_for_backward(x, mixed, offsets, gate_cotangent_offsets)
        ctx.sizes, ctx.grouped, ctx.force_only = sizes, grouped, force_only
        if offsets is None:
            return segmented_forward(x, mixed, bias, sizes)
        return mixed_forward(x, mixed, sizes, grouped)

    @staticmethod
    def backward(ctx, dy):
        x, mixed, offsets, gate_cotangent_offsets = ctx.saved_tensors
        sizes, grouped = ctx.sizes, ctx.grouped
        dy = dy.resolve_neg().contiguous()
        need_x, need_weight, need_bias = ctx.needs_input_grad[:3]
        db = None
        if torch.is_grad_enabled():
            dx, dw = MixedSegmentVJP.apply(
                dy, x, mixed, sizes, offsets, grouped, need_x,
                need_weight and (not ctx.force_only or not need_x), gate_cotangent_offsets,
            )
        else:
            dx = mixed_input_vjp(dy, mixed, sizes, grouped) if need_x else None
            dw = mixed_weight_vjp(dy, x, mixed, sizes, offsets, gate_cotangent_offsets) if need_weight else None
            db = dy.flatten(0, -2).sum(0) if need_bias else None
        return dx, dw, db, None, None, None, None, None


def project_mixed_segments(x, mixed, sizes, offsets, *, force_only=False):
    """Apply one premixed matrix per segment to a matrix or a stack of rows.

    Offsets contain the cumulative leading-axis sizes and are prepared with
    the graph. The CUDA path uses FP32 with autocast and TF32 disabled.
    Set ``force_only`` when a differentiable VJP needs only the input gradient.
    The ordinary VJP and the subsequent force derivative still reach weights.
    """
    sizes = tuple(sizes)
    min_size = min(sizes, default=0)
    if (
        x.ndim not in (2, 3) or mixed.ndim != 3
        or min(x.shape[1:]) <= 0 or mixed.shape[1] <= 0 or x.shape[-1] != mixed.shape[2]
        or not sizes or len(sizes) != mixed.shape[0] or min_size < 0 or sum(sizes) != x.shape[0]
    ):
        raise ValueError("mixed matrices and segment sizes must match the input")
    if not x.is_floating_point() or mixed.dtype != x.dtype or mixed.device != x.device:
        raise ValueError("mixed matrices and input must share a floating dtype and device")
    if (
        offsets.shape != (len(sizes) + 1,) or offsets.dtype != torch.int32
        or offsets.device != x.device or not offsets.is_contiguous() or offsets.is_neg()
    ):
        raise ValueError("segment offsets must be contiguous int32 on the input device")
    if x.is_cuda and (
        x.dtype != torch.float32 or torch.is_autocast_enabled("cuda")
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("CUDA mixed projection requires FP32 without autocast or TF32")
    grouped = (
        x.is_cuda and 1 < len(sizes) <= 64 and min_size > 0
        and not torch.are_deterministic_algorithms_enabled() and grouped_device(x.device)
    )
    return MixedSegmentProjection.apply(
        x.resolve_neg().contiguous(), mixed.resolve_neg().contiguous(), None,
        sizes, offsets, grouped, force_only,
    )


def segmented_linear_with_gradients(
    x, weights, coefficients, bias, sizes, *, gradient_mode, gate_cotangent_offsets=None,
):
    """Select the derivative schedule used by the UMA adapter.

    ``input`` is for frozen parameters. ``force`` computes an input VJP
    first, then parameter derivatives through double backward. ``full``
    provides every input and parameter derivative in either pass.

    The UMA adapter passes ``gate_cotangent_offsets`` only when the raw
    output goes directly into GroupedGate. Those cotangents have paired
    real and imaginary halves with a fixed sign pattern.
    """
    if gradient_mode == "input":
        function = SegmentedLinearInput
    elif gradient_mode == "full":
        function = SegmentedLinear
    elif gradient_mode == "force":
        if x.requires_grad and x.is_cuda and reuse_force_mix(weights) and (
            grouped_weight_device(x.device)
            or (
                (x.ndim, *weights.shape[1:]) in ((2, 640, 768), (3, 512, 512), (3, 256, 256))
                and streamed_weight_device(x.device)
            )
        ):
            # One mixture VJP receives both the energy and force contributions.
            mixed = mix_weights(weights, coefficients)
            grouped = use_grouped(x, mixed, sizes)
            if gate_cotangent_offsets is not None and (
                not grouped or bias is not None or mixed.shape[1:] != (256, 256)
                or not use_gate_weight_vjp(x, sizes)
                or gate_cotangent_offsets.shape != (len(sizes) + 1,)
                or gate_cotangent_offsets.dtype != torch.int32
                or gate_cotangent_offsets.device != x.device
                or not gate_cotangent_offsets.is_contiguous() or gate_cotangent_offsets.is_neg()
            ):
                gate_cotangent_offsets = None
            return MixedSegmentProjection.apply(
                x, mixed, bias, sizes, None, grouped, True, gate_cotangent_offsets,
            )
        function = SegmentedLinearForce if x.requires_grad else SegmentedLinear
    else:
        raise ValueError(f"unknown segmented linear gradient mode: {gradient_mode}")
    return function.apply(x, weights, coefficients, bias, sizes)


def segmented_linear(x, weights, coefficients, bias, sizes):
    """Apply segment-specific mixtures of external linear weights.

    ``x`` is ``(N, ..., Cin)``, ``weights`` is ``(experts, Cout, Cin)``,
    and ``coefficients`` is ``(segments, experts)``. Within each segment,
    the expert weights are mixed with that segment's coefficients. The
    same mixed map acts on every feature row in the segment.
    Explicitly, ``W_s=sum(coefficients[s,k]*weights[k] for k in experts)``
    and ``y_s=functional.linear(x_s,W_s,bias)``.

    ``sizes`` is a sequence of nonnegative integers that partitions the
    leading input axis. It contains one entry per segment and sums to
    ``N``. At least one segment is required; individual segments can be
    empty. Optional shared ``bias`` is ``(Cout,)``. Floating tensors share
    device and dtype. The output has shape ``(N, ..., Cout)``.

    Uses PyTorch matrix operations on CPU or CUDA. Features, weights,
    mixing coefficients, and bias support derivatives through third order.
    Segment sizes are fixed metadata and are not differentiable.
    """
    if x.ndim < 2 or weights.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(
            "segmented linear requires input (N, ..., Cin), weights "
            "(experts, Cout, Cin), and coefficients (segments, experts)"
        )
    experts, cout, cin = weights.shape
    if (
        min(experts, cout, cin) <= 0
        or x.shape[-1] != cin
        or coefficients.shape[1] != experts
    ):
        raise ValueError("segmented linear input and mixture dimensions do not match")
    sizes = tuple(sizes)
    if (
        not sizes
        or len(sizes) != coefficients.shape[0]
        or any(not isinstance(size, int) or size < 0 for size in sizes)
        or sum(sizes) != x.shape[0]
    ):
        raise ValueError(
            "sizes must contain one nonnegative integer per segment "
            "and sum to the input length"
        )
    if bias is not None and tuple(bias.shape) != (cout,):
        raise ValueError("segmented linear bias must have shape (Cout,)")
    tensors = (x, weights, coefficients)
    if bias is not None:
        tensors = (*tensors, bias)
    if any(
        not value.is_floating_point()
        or value.device != x.device
        or value.dtype != x.dtype
        for value in tensors
    ):
        raise ValueError(
            "segmented linear requires floating tensors with one device and dtype"
        )
    return SegmentedLinear.apply(x, weights, coefficients, bias, sizes)


class ConcatWithSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, *values):
        output = torch.cat(values, dim=dim)
        ctx.dim = dim
        ctx.sizes = tuple(value.shape[dim] for value in values)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return None, *torch.split(gradient, ctx.sizes, dim=ctx.dim)


def concat_with_split_backward(
    values: Sequence[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    """Concatenate equal-rank segments without separate slice VJPs in BW2."""
    return ConcatWithSplitBackward.apply(dim, *values)
