"""PyTorch normalization, smooth activation and attention scores."""

import torch
from torch.nn import functional as F


def fused_alpha_projection(inputs, weight, bias, dot, eps):
    """Compute affine LayerNorm, smooth leaky activation, and head dot products.

    Inputs ``(E,H,A)`` normalize over ``A`` with affine vectors ``(A,)``.
    For normalized affine value ``z``, the activation is
    ``0.6*z + 0.4*z*tanh(z/2)``. Multiplication by ``dot[H,A]`` and sum
    over ``A`` return logits ``(E,H)``. All four tensors are differentiable.
    """
    affine = F.layer_norm(inputs, (inputs.shape[-1],), weight, bias, eps)
    activated = 0.6 * affine + 0.4 * affine * torch.tanh(affine / 2)
    return (activated * dot).sum(dim=-1)


def attention_scores(inputs, norm_weight, norm_bias, projection, *, eps=1e-5):
    """Materialize affine LayerNorm, the smooth activation, and head projections."""
    z = F.layer_norm(inputs, (inputs.shape[-1],), norm_weight, norm_bias, eps)
    return ((0.6 * z + 0.4 * z * torch.tanh(z / 2)) * projection).sum(-1)
