"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

import torch

from ....utils import is_compiling


def use_tiled_grouped_norm(inputs, *, max_rows=32768):
    groups, rows, channels = inputs.shape
    return (
        groups in (1, 4)
        and channels == 128
        and 8192 <= rows <= max_rows
        and not is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(inputs.device) == (9, 0)
    )


def grouped_layer_norm_silu_validate(inputs, input_bias, weight, bias):
    if (
        not inputs.is_cuda
        or inputs.dtype != torch.float32
        or not inputs.is_contiguous()
        or inputs.ndim != 3
    ):
        raise ValueError("grouped LayerNorm-SiLU requires contiguous CUDA FP32 [G,R,C]")
    groups, rows_per_group, channels = inputs.shape
    if input_bias.shape != (groups, channels) or weight.shape != (groups, channels):
        raise ValueError(f"unexpected grouped LayerNorm-SiLU contract: {inputs.shape=}")
    if bias.shape != weight.shape or any(
        parameter.dtype != inputs.dtype
        or parameter.device != inputs.device
        or not parameter.is_contiguous()
        for parameter in (input_bias, weight, bias)
    ):
        raise ValueError(
            "grouped LayerNorm-SiLU affine parameters must match input device/dtype and be contiguous"
        )
    return groups, rows_per_group, channels


def layer_norm_silu_validate(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[int, int]:
    if (
        inputs.ndim < 1
        or not inputs.is_cuda
        or inputs.dtype != torch.float32
        or not inputs.is_contiguous()
    ):
        raise ValueError("fused LayerNorm-SiLU requires contiguous CUDA FP32 input")
    channels = inputs.shape[-1]
    if channels < 1 or weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError(f"unexpected LayerNorm-SiLU contract: {inputs.shape=}")
    if any(
        parameter.dtype != inputs.dtype
        or parameter.device != inputs.device
        or not parameter.is_contiguous()
        for parameter in (weight, bias)
    ):
        raise ValueError(
            "LayerNorm-SiLU affine parameters must match input device/dtype and be contiguous"
        )
    return inputs.numel() // channels, channels
