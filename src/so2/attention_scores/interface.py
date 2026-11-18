"""Fused LayerNorm, smooth activation, and per-head attention projection."""

from __future__ import annotations

from math import isfinite

import torch
from torch.autograd.function import once_differentiable

from .kernels.backward import backward
from .kernels.double_backward import double_backward
from .kernels.forward import forward
from .kernels.triple_backward import triple_backward
from .kernels.utils import validate


class AlphaProjectionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, inputs, weight, bias, dot, eps):
        gradients = backward(grad_output, inputs, weight, bias, dot, eps)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_output, inputs, weight, bias, dot)
        ctx.eps = eps
        return gradients

    @staticmethod
    def backward(
        ctx, input_direction, weight_direction, bias_direction, dot_direction
    ):
        args = (
            *ctx.saved_tensors, input_direction, weight_direction, bias_direction,
            dot_direction, ctx.eps,
        )
        gradients = (
            AlphaProjectionDoubleBackward.apply(*args)
            if torch.is_grad_enabled()
            else double_backward(*args)
        )
        return *gradients, None


class AlphaProjectionDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, inputs, weight, bias, dot, hx, hw, hb, hd, eps):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_output, inputs, weight, bias, dot, hx, hw, hb, hd)
        ctx.eps = eps
        return double_backward(grad_output, inputs, weight, bias, dot, hx, hw, hb, hd, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        gradients = triple_backward(
            ctx.saved_tensors, cotangents, ctx.needs_input_grad[:-1], ctx.eps,
        )
        return *gradients, None


class AlphaProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, dot, eps):
        output = forward(inputs, weight, bias, dot, eps)
        ctx.save_for_backward(inputs, weight, bias, dot)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias, dot = ctx.saved_tensors
        return (
            *AlphaProjectionBackward.apply(
                grad_output.contiguous(), inputs, weight, bias, dot, ctx.eps
            ),
            None,
        )


def attention_scores(
    inputs: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    projection: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fuse LayerNorm, SmoothLeakyReLU(0.2), and head-wise projection.

    ``inputs`` is ``(E,H,A)``, affine ``norm_weight`` and ``norm_bias`` are ``(A,)``,
    and ``projection`` is ``(H,A)``. Apply LayerNorm over the last axis with
    epsilon ``eps``. For its affine output ``z``, compute
    ``a=0.6*z+0.4*z*tanh(z/2)`` and return logits
    ``(a*projection).sum(-1)`` of shape ``(E,H)``. The reduction operator
    applies softmax and weights the messages.

    All inputs must be CUDA FP32 on one device. Parameters and the input
    channel axis must be contiguous. Inputs, normalization parameters,
    and projection weights support derivatives through third order.
    """

    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    _, _, channels = validate(inputs, norm_weight, norm_bias, projection)
    if channels == 1:
        # A one-channel LayerNorm is exactly zero before its affine bias.
        # Keep the zero dependencies so both reverse passes return tensors.
        affine = inputs * 0.0 + norm_weight * 0.0 + norm_bias
        activated = 0.6 * affine + 0.4 * affine * torch.tanh(affine / 2)
        return (activated * projection).sum(dim=-1)
    return AlphaProjection.apply(inputs, norm_weight, norm_bias, projection, eps)
