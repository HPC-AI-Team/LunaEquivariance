from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import smooth_leaky, validate


@triton.jit
def alpha_projection_forward_kernel(
    input_ptr, weight_ptr, bias_ptr, dot_ptr, output_ptr,
    edges, eps,
    input_edge_stride: tl.constexpr, input_head_stride: tl.constexpr,
    heads: tl.constexpr, channels: tl.constexpr,
    block_channels: tl.constexpr, block_heads: tl.constexpr,
):
    edge = tl.program_id(0)
    head_1d = tl.arange(0, block_heads)
    head = head_1d[:, None]
    channel_1d = tl.arange(0, block_channels)
    channel = channel_1d[None, :]
    mask = (edge < edges) & (head < heads) & (channel < channels)
    input_offset = edge * input_edge_stride + head * input_head_stride + channel
    inputs = tl.load(input_ptr + input_offset, mask=mask, other=0.0).to(tl.float32)
    inv_channels = 1.0 / channels
    mean = tl.sum(inputs, axis=1)[:, None] * inv_channels
    centered = tl.where(channel < channels, inputs - mean, 0.0)
    variance = tl.sum(centered * centered, axis=1)[:, None] * inv_channels
    rstd = tl.rsqrt(variance + eps)
    normalized = centered * rstd
    weight = tl.load(weight_ptr + channel, mask=channel < channels, other=0.0)
    bias = tl.load(bias_ptr + channel, mask=channel < channels, other=0.0)
    affine = normalized * weight + bias
    activated = smooth_leaky(affine)
    dot = tl.load(dot_ptr + head * channels + channel, mask=mask, other=0.0).to(
        tl.float32
    )
    output = tl.sum(activated * dot, axis=1)
    tl.store(
        output_ptr + edge * heads + head_1d,
        output,
        mask=(edge < edges) & (head_1d < heads),
    )


def forward(inputs, weight, bias, dot, eps):
    edges, heads, channels = validate(inputs, weight, bias, dot)
    output = torch.empty((edges, heads), device=inputs.device, dtype=inputs.dtype)
    alpha_projection_forward_kernel[(edges,)](
        inputs, weight, bias, dot, output,
        edges, eps,
        input_edge_stride=inputs.stride(0),
        input_head_stride=inputs.stride(1),
        heads=heads,
        channels=channels,
        block_channels=triton.next_power_of_2(channels),
        block_heads=triton.next_power_of_2(heads),
        num_warps=8,
    )
    return output
