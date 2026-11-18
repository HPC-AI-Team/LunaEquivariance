from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import smooth_leaky, smooth_leaky_prime, smooth_leaky_second, validate


@triton.jit
def alpha_projection_backward_kernel(
    grad_output_ptr, input_ptr, weight_ptr, bias_ptr, dot_ptr,
    grad_input_ptr, grad_weight_ptr, grad_bias_ptr, grad_dot_ptr,
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
    offset = (edge * heads + head) * channels + channel
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
    dot = tl.load(dot_ptr + head * channels + channel, mask=mask, other=0.0)
    grad_output = tl.load(
        grad_output_ptr + edge * heads + head,
        mask=(edge < edges) & (head < heads),
        other=0.0,
    ).to(tl.float32)
    affine = normalized * weight + bias
    activated = smooth_leaky(affine)
    prime = smooth_leaky_prime(affine)
    grad_affine = grad_output * dot * prime
    grad_normalized = grad_affine * weight
    mean_grad = tl.sum(grad_normalized, axis=1)[:, None] * inv_channels
    mean_grad_normalized = (
        tl.sum(grad_normalized * normalized, axis=1)[:, None] * inv_channels
    )
    grad_input = rstd * (
        grad_normalized - mean_grad - normalized * mean_grad_normalized
    )
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)

    grad_weight = tl.sum(grad_affine * normalized, axis=0)
    grad_bias = tl.sum(grad_affine, axis=0)
    tl.atomic_add(
        grad_weight_ptr + channel_1d, grad_weight,
        mask=channel_1d < channels, sem="relaxed", scope="gpu",
    )
    tl.atomic_add(
        grad_bias_ptr + channel_1d, grad_bias,
        mask=channel_1d < channels, sem="relaxed", scope="gpu",
    )
    tl.atomic_add(
        grad_dot_ptr + head * channels + channel, grad_output * activated,
        mask=mask, sem="relaxed", scope="gpu",
    )


@triton.jit
def tiled_parameter_gradients_kernel(
    inputs, weight, bias, projection, dy,
    hx_ptr, hw_ptr, hb_ptr, hp_ptr, dx_ptr, ddy_ptr, partials,
    edges, eps,
    edge_stride: tl.constexpr, head_stride: tl.constexpr,
    heads: tl.constexpr, channels: tl.constexpr,
    block_heads: tl.constexpr, block_channels: tl.constexpr,
    tile_edges: tl.constexpr, tiles_per_block: tl.constexpr,
    double: tl.constexpr,
    has_hx: tl.constexpr, has_hw: tl.constexpr, has_hb: tl.constexpr,
    has_hp: tl.constexpr,
):
    block = tl.program_id(0)
    rows = tl.arange(0, tile_edges * block_heads)
    heads_in_tile = rows % block_heads
    edges_in_tile = rows // block_heads
    cols = tl.arange(0, block_channels)

    w = tl.load(weight + cols, cols < channels, 0.0)[None, :]
    b = tl.load(bias + cols, cols < channels, 0.0)[None, :]
    param_offset = heads_in_tile[:, None] * channels + cols[None, :]
    param_mask = (heads_in_tile[:, None] < heads) & (cols[None, :] < channels)
    p = tl.load(projection + param_offset, param_mask, 0.0)
    if double:
        if has_hw:
            hw = tl.load(hw_ptr + cols, cols < channels, 0.0)[None, :]
        else:
            hw = 0.0
        if has_hb:
            hb = tl.load(hb_ptr + cols, cols < channels, 0.0)[None, :]
        else:
            hb = 0.0
        if has_hp:
            hp = tl.load(hp_ptr + param_offset, param_mask, 0.0)
        else:
            hp = 0.0

    # One block owns disjoint edges and emits a single partial per parameter.
    dw_sum = tl.zeros((tile_edges * block_heads, block_channels), tl.float32)
    db_sum = tl.zeros((tile_edges * block_heads, block_channels), tl.float32)
    dp_sum = tl.zeros((tile_edges * block_heads, block_channels), tl.float32)
    for tile in range(tiles_per_block):
        edge = (block * tiles_per_block + tile) * tile_edges + edges_in_tile
        row_mask = (edge < edges) & (heads_in_tile < heads)
        mask = row_mask[:, None] & (cols[None, :] < channels)
        row = edge * heads + heads_in_tile
        offset = row[:, None] * channels + cols[None, :]
        source = edge[:, None] * edge_stride + heads_in_tile[:, None] * head_stride + cols[None, :]
        x = tl.load(inputs + source, mask, 0.0)
        mean = (tl.sum(x, axis=1) / channels)[:, None]
        centered = tl.where(cols[None, :] < channels, x - mean, 0.0)
        rstd = tl.rsqrt((tl.sum(centered * centered, axis=1) / channels)[:, None] + eps)
        norm = centered * rstd
        z = norm * w + b
        a = smooth_leaky(z)
        ap = smooth_leaky_prime(z)
        cot = tl.load(dy + row, row_mask, 0.0)[:, None]

        if double:
            if has_hx:
                hx = tl.load(hx_ptr + offset, mask, 0.0)
                mean_hx = (tl.sum(hx, axis=1) / channels)[:, None]
                mean_hx_norm = (tl.sum(hx * norm, axis=1) / channels)[:, None]
                hnorm = rstd * (hx - mean_hx - norm * mean_hx_norm)
                # Padded channels must not enter the LayerNorm tangent reduction.
                hnorm = tl.where(cols[None, :] < channels, hnorm, 0.0)
                hrstd = -rstd * rstd * mean_hx_norm
            else:
                hnorm = 0.0
                hrstd = 0.0
            hz = hw * norm + w * hnorm + hb
            app = smooth_leaky_second(z)
            da = cot * p * ap
            hda = cot * (hp * ap + p * app * hz)
            dnorm = da * w
            hdnorm = hda * w + da * hw
            mean_dnorm = (tl.sum(dnorm, axis=1) / channels)[:, None]
            mean_dnorm_norm = (tl.sum(dnorm * norm, axis=1) / channels)[:, None]
            projected = dnorm - mean_dnorm - norm * mean_dnorm_norm
            mean_hdnorm = (tl.sum(hdnorm, axis=1) / channels)[:, None]
            mean_projected = (tl.sum(hdnorm * norm + dnorm * hnorm, axis=1) / channels)[:, None]
            dx = hrstd * projected + rstd * (
                hdnorm - mean_hdnorm - hnorm * mean_dnorm_norm - norm * mean_projected
            )
            ddy = tl.sum(hp * a + p * ap * hz, axis=1)
            tl.store(ddy_ptr + row, ddy, row_mask)
            dw = hda * norm + da * hnorm
            db = hda
            dp = cot * ap * hz
        else:
            dz = cot * p * ap
            dnorm = dz * w
            mean_dnorm = (tl.sum(dnorm, axis=1) / channels)[:, None]
            mean_dnorm_norm = (tl.sum(dnorm * norm, axis=1) / channels)[:, None]
            dx = rstd * (dnorm - mean_dnorm - norm * mean_dnorm_norm)
            dw = dz * norm
            db = dz
            dp = cot * a
        tl.store(dx_ptr + offset, dx, mask)
        dw_sum += tl.where(mask, dw, 0.0)
        db_sum += tl.where(mask, db, 0.0)
        dp_sum += tl.where(mask, dp, 0.0)

    block_width = (heads + 2) * channels
    base = partials + block * block_width
    tl.store(base + cols, tl.sum(dw_sum, axis=0), cols < channels)
    tl.store(base + channels + cols, tl.sum(db_sum, axis=0), cols < channels)
    grad_projection = tl.sum(
        dp_sum.reshape(tile_edges, block_heads, block_channels), axis=0,
    )
    head = tl.arange(0, block_heads)[:, None]
    tl.store(
        base + (2 + head) * channels + cols[None, :],
        grad_projection, (head < heads) & (cols[None, :] < channels),
    )


@triton.jit
def reduce_parameter_gradients_kernel(
    partials, grad_weight, grad_bias, grad_projection, blocks,
    channels: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_cols: tl.constexpr,
):
    row = tl.arange(0, block_rows)[:, None]
    col = tl.program_id(0) * block_cols + tl.arange(0, block_cols)
    # Bound the combining tile independently of the edge count.
    accum = tl.zeros((block_rows, block_cols), tl.float32)
    for start in range(0, blocks, block_rows):
        values = tl.load(
            partials + (start + row) * width + col[None, :],
            (start + row < blocks) & (col[None, :] < width),
            0.0,
        )
        accum += values
    result = tl.sum(accum, axis=0)
    tl.store(grad_weight + col, result, col < channels)
    tl.store(grad_bias + col - channels, result, (col >= channels) & (col < 2 * channels))
    tl.store(grad_projection + col - 2 * channels, result, (col >= 2 * channels) & (col < width))


def blocked_parameter_gradients(
    inputs, weight, bias, projection, dy, eps, config, directions=None,
):
    edges, heads, channels = inputs.shape
    dx = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
    double = directions is not None
    ddy = torch.empty_like(dy) if double else None

    tile_edges, tiles_per_block, warps = config
    block_heads = triton.next_power_of_2(heads)
    block_channels = triton.next_power_of_2(channels)
    blocks = triton.cdiv(edges, tile_edges * tiles_per_block)
    width = (heads + 2) * channels
    partials = torch.empty((blocks, width), device=inputs.device, dtype=inputs.dtype)
    dw, db, dp = (torch.empty_like(t) for t in (weight, bias, projection))
    directions = (None, None, None, None) if directions is None else directions
    pointers = tuple(inputs if t is None else t.contiguous() for t in directions)
    tiled_parameter_gradients_kernel[(blocks,)](
        inputs, weight, bias, projection, dy, *pointers,
        dx, dy if ddy is None else ddy, partials, edges, eps,
        inputs.stride(0), inputs.stride(1), heads, channels,
        block_heads, block_channels, tile_edges, tiles_per_block, double,
        *(t is not None for t in directions), num_warps=warps,
    )
    reduce_rows = min(triton.next_power_of_2(blocks), 256)
    # The narrower HVP reduction gives the large partials array more CTAs.
    reduce_cols = 4 if double else 16
    reduce_parameter_gradients_kernel[(triton.cdiv(width, reduce_cols),)](
        partials, dw, db, dp, blocks, channels, width, reduce_rows, reduce_cols, num_warps=4,
    )
    result = (dx, dw, db, dp)
    return (ddy, *result) if double else result


def blocked_gradient_config(edges, heads, channels):
    # The tiled reduction wins both derivative stages at this crossover
    # on A100 and H100 for the tested eight-head widths.
    if edges < 4_096 or heads != 8 or channels not in (16, 32, 64):
        return None
    return (8, 8, 4) if channels == 16 else (4, 16, 4)


def backward(grad_output, inputs, weight, bias, dot, eps):
    allow_atomic_reduction("attention_logits backward")
    edges, heads, channels = validate(inputs, weight, bias, dot)
    config = blocked_gradient_config(edges, heads, channels)
    if config is not None:
        return blocked_parameter_gradients(
            inputs, weight, bias, dot, grad_output, eps, config,
        )
    grad_inputs = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    grad_dot = torch.zeros_like(dot)
    alpha_projection_backward_kernel[(edges,)](
        grad_output, inputs, weight, bias, dot,
        grad_inputs, grad_weight, grad_bias, grad_dot,
        edges, eps,
        input_edge_stride=inputs.stride(0),
        input_head_stride=inputs.stride(1),
        heads=heads,
        channels=channels,
        block_channels=triton.next_power_of_2(channels),
        block_heads=triton.next_power_of_2(heads),
        num_warps=8,
    )
    return grad_inputs, grad_weight, grad_bias, grad_dot
