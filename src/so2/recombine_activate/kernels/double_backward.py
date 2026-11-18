from __future__ import annotations

import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value
from .utils import silu_terms


@triton.jit
def double_backward_kernel(
    grad_ptrs,
    gating_ptr,
    x_ptrs,
    gating_direction_ptr,
    direction_ptrs,
    grad_gradient_ptrs,
    gating_gradient_ptr,
    x_gradient_ptrs,
    gating_stride_edge,
    x0_stride_edge,
    gating_gradient_stride_edge: tl.constexpr,
    x0_gradient_stride_edge: tl.constexpr,
    gating_direction_stride_item,
    x0_direction_stride_edge,
    x0_direction_stride_row,
    direction_strides,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    fold_gate_adjoint: tl.constexpr = False,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    if channels > block_channels:
        channel += tl.program_id(1) * block_channels
    mask = channel < channels
    x0_scalar_address = edge * x0_stride_edge + channel
    scalar_address = edge * (lmax + 1) * channels + channel
    scalar = tl.load(x_ptrs[0] + x0_scalar_address, mask=mask, other=0.0).to(tl.float32)
    scalar_grad = tl.load(grad_ptrs[0] + scalar_address, mask=mask, other=0.0).to(
        tl.float32
    )
    scalar_direction = tl.load(
        direction_ptrs[0] + edge * x0_direction_stride_edge + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    _, silu_first, silu_second = silu_terms(scalar)
    tl.store(
        grad_gradient_ptrs[0] + scalar_address,
        silu_first * scalar_direction,
        mask=mask,
    )
    tl.store(
        x_gradient_ptrs[0] + edge * x0_gradient_stride_edge + channel,
        scalar_grad * silu_second * scalar_direction,
        mask=mask,
    )

    for degree in tl.static_range(1, lmax + 1):
        gate_input_address = (
            edge * gating_stride_edge + (degree - 1) * channels + channel
        )
        gate_output_address = edge * gating_gradient_stride_edge + (degree - 1) * channels + channel
        raw_gate = tl.load(gating_ptr + gate_input_address, mask=mask, other=0.0).to(
            tl.float32
        )
        gate_direction = tl.load(
            gating_direction_ptr
            + edge * gating_direction_stride_item
            + (degree - 1) * channels
            + channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(raw_gate)
        gate_first = gate * (1.0 - gate)
        gate_second = gate_first * (1.0 - 2.0 * gate)
        address0 = edge * (lmax + 1) * channels + degree * channels + channel
        x0_address = edge * x0_stride_edge + degree * channels + channel
        grad0 = tl.load(grad_ptrs[0] + address0, mask=mask, other=0.0).to(tl.float32)
        value0 = tl.load(x_ptrs[0] + x0_address, mask=mask, other=0.0).to(tl.float32)
        direction0 = tl.load(
            direction_ptrs[0]
            + edge * x0_direction_stride_edge
            + degree * x0_direction_stride_row
            + channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_gradient_ptrs[0] + address0,
            gate * direction0 + value0 * gate_first * gate_direction,
            mask=mask,
        )
        dx0_address = edge * x0_gradient_stride_edge + degree * channels + channel
        tl.store(
            x_gradient_ptrs[0] + dx0_address,
            grad0 * gate_first * gate_direction,
            mask=mask,
        )
        gating_hvp = (
            gate_first * grad0 * direction0
            + gate_second * gate_direction * grad0 * value0
        )
        contraction = tl.zeros(channel.shape, dtype=tl.float32)
        for order in tl.static_range(1, mmax + 1):
            if order <= degree:
                width = lmax - order + 1
                local = degree - order
                grad_ptr = grad_ptrs[order]
                x_ptr = x_ptrs[order]
                direction_ptr = direction_ptrs[order]
                direction_stride_edge = direction_strides[order][0]
                direction_stride_plane = direction_strides[order][1]
                direction_stride_item = direction_strides[order][2]
                grad_gradient_ptr = grad_gradient_ptrs[order]
                x_gradient_ptr = x_gradient_ptrs[order]
                for sign in tl.static_range(0, 2):
                    row = local + sign * width
                    grad_address = (
                        edge * 2 * width * channels + row * channels + channel
                    )
                    gradient = tl.load(
                        grad_ptr + grad_address, mask=mask, other=0.0
                    ).to(tl.float32)
                    value = tl.load(x_ptr + grad_address, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    direction = load_sector_value(
                        direction_ptr, edge, row, channel,
                        direction_stride_edge, direction_stride_plane, direction_stride_item,
                        channels, width, 1, raw_complex, mask,
                    )
                    tl.store(
                        grad_gradient_ptr + grad_address,
                        gate * direction + value * (1.0 - gate) * gate_direction,
                        mask=mask,
                    )
                    tl.store(
                        x_gradient_ptr + grad_address,
                        gradient * (1.0 - gate) * gate_direction,
                        mask=mask,
                    )
                    if fold_gate_adjoint:
                        contraction += gradient * (
                            gate * direction + (1.0 - 2.0 * gate) * gate_direction * value
                        )
                    else:
                        contraction += gradient * (direction - gate_direction * value)
        factor = (1.0 - gate) if fold_gate_adjoint else gate_first
        tl.store(
            gating_gradient_ptr + gate_output_address,
            gating_hvp + factor * contraction,
            mask=mask,
        )
