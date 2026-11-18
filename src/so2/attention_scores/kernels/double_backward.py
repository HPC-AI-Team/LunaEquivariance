from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.tensor_contracts import allow_atomic_reduction
from .backward import blocked_gradient_config, blocked_parameter_gradients
from .utils import smooth_leaky, smooth_leaky_prime, smooth_leaky_second, validate


@triton.jit
def alpha_projection_double_backward_kernel(
    grad_output_ptr, input_ptr, weight_ptr, bias_ptr, dot_ptr,
    input_direction_ptr, weight_direction_ptr, bias_direction_ptr, dot_direction_ptr,
    grad_grad_output_ptr, grad_input_ptr, grad_weight_ptr, grad_bias_ptr, grad_dot_ptr,
    edges, eps,
    input_edge_stride: tl.constexpr, input_head_stride: tl.constexpr,
    heads: tl.constexpr, channels: tl.constexpr,
    block_channels: tl.constexpr, block_heads: tl.constexpr,
    has_input_direction: tl.constexpr,
    has_weight_direction: tl.constexpr,
    has_bias_direction: tl.constexpr,
    has_dot_direction: tl.constexpr,
):
    edge = tl.program_id(0)
    head_1d = tl.arange(0, block_heads)
    head = head_1d[:, None]
    channel_1d = tl.arange(0, block_channels)
    channel = channel_1d[None, :]
    mask = (edge < edges) & (head < heads) & (channel < channels)
    offset = (edge * heads + head) * channels + channel
    input_offset = edge * input_edge_stride + head * input_head_stride + channel
    inputs = tl.load(input_ptr + input_offset, mask=mask, other=0.0).to(tl.float32)
    inv_channels = 1.0 / channels
    mean = tl.sum(inputs, axis=1)[:, None] * inv_channels
    centered = tl.where(channel < channels, inputs - mean, 0.0)
    variance = tl.sum(centered * centered, axis=1)[:, None] * inv_channels
    rstd = tl.rsqrt(variance + eps)
    xhat = centered * rstd

    # hx, hw, hb and hdot are incoming derivatives of the first backward outputs.
    # xhat is the normalized input.
    if has_input_direction:
        hx = tl.load(input_direction_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    else:
        hx = tl.zeros(inputs.shape, dtype=tl.float32)
    mean_hx = tl.sum(hx, axis=1)[:, None] * inv_channels
    mean_hx_xhat = tl.sum(hx * xhat, axis=1)[:, None] * inv_channels
    hxhat = rstd * (hx - mean_hx - xhat * mean_hx_xhat)
    hrstd = -rstd * rstd * mean_hx_xhat

    weight = tl.load(weight_ptr + channel, mask=channel < channels, other=0.0)
    bias = tl.load(bias_ptr + channel, mask=channel < channels, other=0.0)
    dot = tl.load(dot_ptr + head * channels + channel, mask=mask, other=0.0)
    if has_weight_direction:
        hw = tl.load(
            weight_direction_ptr + channel, mask=channel < channels, other=0.0
        ).to(tl.float32)
    else:
        hw = tl.zeros((1, block_channels), dtype=tl.float32)
    if has_bias_direction:
        hb = tl.load(
            bias_direction_ptr + channel, mask=channel < channels, other=0.0
        ).to(tl.float32)
    else:
        hb = tl.zeros((1, block_channels), dtype=tl.float32)
    if has_dot_direction:
        hdot = tl.load(
            dot_direction_ptr + head * channels + channel, mask=mask, other=0.0
        ).to(tl.float32)
    else:
        hdot = tl.zeros(inputs.shape, dtype=tl.float32)
    dy = tl.load(
        grad_output_ptr + edge * heads + head,
        mask=(edge < edges) & (head < heads),
        other=0.0,
    ).to(tl.float32)

    z = xhat * weight + bias
    hz = hw * xhat + weight * hxhat + hb
    a = smooth_leaky(z)
    ap = smooth_leaky_prime(z)
    app = smooth_leaky_second(z)

    phi_xhat = dot * ap * weight
    hphi_xhat = hdot * ap * weight + dot * app * hz * weight + dot * ap * hw
    dxhat = dy * phi_xhat
    hdxhat = dy * hphi_xhat
    mean_dxhat = tl.sum(dxhat, axis=1)[:, None] * inv_channels
    mean_dxhat_xhat = tl.sum(dxhat * xhat, axis=1)[:, None] * inv_channels
    projected = dxhat - mean_dxhat - xhat * mean_dxhat_xhat
    mean_hdxhat = tl.sum(hdxhat, axis=1)[:, None] * inv_channels
    mean_hdxhat_xhat = tl.sum(hdxhat * xhat, axis=1)[:, None] * inv_channels
    hprojected = hdxhat - mean_hdxhat - xhat * mean_hdxhat_xhat
    mean_dxhat_hxhat = tl.sum(dxhat * hxhat, axis=1)[:, None] * inv_channels
    dx = hrstd * projected + rstd * (
        hprojected - hxhat * mean_dxhat_xhat - xhat * mean_dxhat_hxhat
    )
    tl.store(grad_input_ptr + offset, dx, mask=mask)

    d_dy = tl.sum(hdot * a + dot * ap * hz, axis=1)
    tl.store(
        grad_grad_output_ptr + edge * heads + head_1d,
        d_dy,
        mask=(edge < edges) & (head_1d < heads),
    )
    dw = tl.sum(
        dy * (hdot * ap * xhat + dot * app * hz * xhat + dot * ap * hxhat),
        axis=0,
    )
    db = tl.sum(dy * (hdot * ap + dot * app * hz), axis=0)
    ddot = dy * ap * hz
    tl.atomic_add(
        grad_weight_ptr + channel_1d, dw,
        mask=channel_1d < channels, sem="relaxed", scope="gpu",
    )
    tl.atomic_add(
        grad_bias_ptr + channel_1d, db,
        mask=channel_1d < channels, sem="relaxed", scope="gpu",
    )
    tl.atomic_add(
        grad_dot_ptr + head * channels + channel, ddot,
        mask=mask, sem="relaxed", scope="gpu",
    )


def double_backward(
    grad_output, inputs, weight, bias, dot,
    input_direction, weight_direction, bias_direction, dot_direction,
    eps,
):
    allow_atomic_reduction("attention_logits double_backward")
    edges, heads, channels = validate(inputs, weight, bias, dot)
    config = blocked_gradient_config(edges, heads, channels)
    if config is not None:
        return blocked_parameter_gradients(
            inputs, weight, bias, dot, grad_output, eps, config,
            (input_direction, weight_direction, bias_direction, dot_direction),
        )
    dummy = inputs
    grad_grad_output = torch.empty_like(grad_output)
    grad_inputs = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    grad_dot = torch.zeros_like(dot)
    alpha_projection_double_backward_kernel[(edges,)](
        grad_output, inputs, weight, bias, dot,
        dummy if input_direction is None else input_direction.contiguous(),
        dummy if weight_direction is None else weight_direction.contiguous(),
        dummy if bias_direction is None else bias_direction.contiguous(),
        dummy if dot_direction is None else dot_direction.contiguous(),
        grad_grad_output, grad_inputs, grad_weight, grad_bias, grad_dot,
        edges, eps,
        input_edge_stride=inputs.stride(0),
        input_head_stride=inputs.stride(1),
        heads=heads,
        channels=channels,
        block_channels=triton.next_power_of_2(channels),
        block_heads=triton.next_power_of_2(heads),
        has_input_direction=input_direction is not None,
        has_weight_direction=weight_direction is not None,
        has_bias_direction=bias_direction is not None,
        has_dot_direction=dot_direction is not None,
        num_warps=8,
    )
    return grad_grad_output, grad_inputs, grad_weight, grad_bias, grad_dot
