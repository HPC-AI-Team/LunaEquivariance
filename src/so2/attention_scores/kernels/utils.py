"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def smooth_leaky(value):
    tangent = 2.0 * tl.sigmoid(value) - 1.0
    return 0.6 * value + 0.4 * value * tangent


@triton.jit
def smooth_leaky_prime(value):
    tangent = 2.0 * tl.sigmoid(value) - 1.0
    return 0.6 + 0.4 * tangent + 0.2 * value * (1.0 - tangent * tangent)


@triton.jit
def smooth_leaky_second(value):
    tangent = 2.0 * tl.sigmoid(value) - 1.0
    return (1.0 - tangent * tangent) * (0.4 - 0.2 * value * tangent)


def validate(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dot: torch.Tensor,
) -> tuple[int, int, int]:
    if (
        not inputs.is_cuda
        or inputs.dtype != torch.float32
        or inputs.ndim != 3
        or inputs.stride(-1) != 1
    ):
        raise ValueError(
            "alpha projection requires CUDA FP32 input with contiguous channels"
        )
    edges, heads, channels = inputs.shape
    if heads < 1 or channels < 1:
        raise ValueError("alpha projection heads and channels must be positive")
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("unexpected alpha LayerNorm parameters")
    if dot.shape != (heads, channels):
        raise ValueError("unexpected alpha dot parameter")
    if any(
        tensor.device != inputs.device or tensor.dtype != inputs.dtype
        for tensor in (weight, bias, dot)
    ):
        raise ValueError("alpha projection parameters must match the input")
    if any(not tensor.is_contiguous() for tensor in (weight, bias, dot)):
        raise ValueError("alpha projection parameters must be contiguous")
    return edges, heads, channels
