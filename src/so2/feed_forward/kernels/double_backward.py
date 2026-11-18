from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.coefficient_gate import silu_terms
from ...utils.degree_norm import (
    hvp_with_cotangent_direction,
    hvp_with_cotangent_input_and_output,
    jvp,
    jvp_input,
)
from .backward import bias_gradients, parameter_reduce, use_grouped_weight_gradients, weight_derivative
from .forward import epilogue_forward
from .utils import activation_layout, validate


@triton.jit
def hvp_kernel(
    grad_ptr,
    grad_direction_ptr,
    hidden_ptr,
    gating_ptr,
    hidden_bias_ptr,
    gating_bias_ptr,
    hidden_direction_ptr,
    gating_direction_ptr,
    hidden_bias_direction_ptr,
    gating_bias_direction_ptr,
    grad_output_gradient_ptr,
    hidden_gradient_ptr,
    gating_gradient_ptr,
    grad_stride_row,
    grad_stride_node,
    grad_stride_channel,
    grad_direction_stride_row,
    grad_direction_stride_node,
    grad_direction_stride_channel,
    hidden_stride_row,
    hidden_stride_node,
    hidden_stride_channel,
    gating_stride_node,
    hidden_direction_stride_row,
    hidden_direction_stride_node,
    hidden_direction_stride_channel,
    gating_direction_stride_node,
    grad_output_gradient_stride_row,
    grad_output_gradient_stride_node,
    grad_output_gradient_stride_channel,
    hidden_gradient_stride_row,
    hidden_gradient_stride_node,
    hidden_gradient_stride_channel,
    gating_gradient_stride_node,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    row_degrees: tl.constexpr,
    rows: tl.constexpr,
    has_grad_direction: tl.constexpr,
    has_bias_direction: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    mask = channel < channels

    scalar = tl.load(
        hidden_ptr + node * hidden_stride_node + channel * hidden_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scalar += tl.load(hidden_bias_ptr + channel, mask=mask, other=0.0)
    scalar_direction = tl.load(
        hidden_direction_ptr
        + node * hidden_direction_stride_node
        + channel * hidden_direction_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_bias_direction:
        scalar_direction += tl.load(hidden_bias_direction_ptr + channel, mask=mask, other=0.0)
    scalar_grad = tl.load(
        grad_ptr + node * grad_stride_node + channel * grad_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    _, scalar_first, scalar_second = silu_terms(scalar)
    scalar_grad_direction = tl.zeros(channel.shape, dtype=tl.float32)
    if has_grad_direction:
        scalar_grad_direction = tl.load(
            grad_direction_ptr
            + node * grad_direction_stride_node
            + channel * grad_direction_stride_channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        grad_output_gradient_ptr
        + node * grad_output_gradient_stride_node
        + channel * grad_output_gradient_stride_channel,
        scalar_first * scalar_direction,
        mask=mask,
    )
    scalar_gradient = (
        scalar_grad * scalar_second * scalar_direction
        + scalar_grad_direction * scalar_first
    )
    tl.store(
        hidden_gradient_ptr
        + node * hidden_gradient_stride_node
        + channel * hidden_gradient_stride_channel,
        scalar_gradient,
        mask=mask,
    )

    for degree in tl.static_range(1, lmax + 1):
        gate_offset = (degree - 1) * channels + channel
        raw_gate = tl.load(gating_ptr + node * gating_stride_node + gate_offset, mask=mask, other=0.0).to(tl.float32)
        raw_gate += tl.load(gating_bias_ptr + gate_offset, mask=mask, other=0.0)
        gate_direction = tl.load(
            gating_direction_ptr + node * gating_direction_stride_node + gate_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if has_bias_direction:
            gate_direction += tl.load(gating_bias_direction_ptr + gate_offset, mask=mask, other=0.0)
        silu_gate, silu_first, silu_second = silu_terms(raw_gate)
        gate = tl.sigmoid(silu_gate)
        sigmoid_first = gate * (1.0 - gate)
        gate_first = sigmoid_first * silu_first
        gate_second = (
            sigmoid_first * (1.0 - 2.0 * gate) * silu_first * silu_first
            + sigmoid_first * silu_second
        )
        contraction_value = tl.zeros(channel.shape, dtype=tl.float32)
        contraction_direction = tl.zeros(channel.shape, dtype=tl.float32)
        contraction_grad_direction = tl.zeros(channel.shape, dtype=tl.float32)
        for row in tl.static_range(1, rows):
            if row >= degree * degree and row < (degree + 1) * (degree + 1):
                address = (
                    row * hidden_stride_row
                    + node * hidden_stride_node
                    + channel * hidden_stride_channel
                )
                grad_address = (
                    row * grad_stride_row
                    + node * grad_stride_node
                    + channel * grad_stride_channel
                )
                direction_address = (
                    row * hidden_direction_stride_row
                    + node * hidden_direction_stride_node
                    + channel * hidden_direction_stride_channel
                )
                value = tl.load(hidden_ptr + address, mask=mask, other=0.0).to(tl.float32)
                value_direction = tl.load(hidden_direction_ptr + direction_address, mask=mask, other=0.0).to(tl.float32)
                value_grad = tl.load(grad_ptr + grad_address, mask=mask, other=0.0).to(tl.float32)
                value_grad_direction = tl.zeros(channel.shape, dtype=tl.float32)
                if has_grad_direction:
                    value_grad_direction = tl.load(
                        grad_direction_ptr
                        + row * grad_direction_stride_row
                        + node * grad_direction_stride_node
                        + channel * grad_direction_stride_channel,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                tl.store(
                    grad_output_gradient_ptr
                    + row * grad_output_gradient_stride_row
                    + node * grad_output_gradient_stride_node
                    + channel * grad_output_gradient_stride_channel,
                    gate * value_direction + value * gate_first * gate_direction,
                    mask=mask,
                )
                tl.store(
                    hidden_gradient_ptr
                    + row * hidden_gradient_stride_row
                    + node * hidden_gradient_stride_node
                    + channel * hidden_gradient_stride_channel,
                    value_grad * gate_first * gate_direction
                    + value_grad_direction * gate,
                    mask=mask,
                )
                contraction_value += value_grad * value
                contraction_direction += value_grad * value_direction
                contraction_grad_direction += value_grad_direction * value
        gate_gradient = (
            gate_first * contraction_direction
            + gate_second * gate_direction * contraction_value
            + gate_first * contraction_grad_direction
        )
        tl.store(
            gating_gradient_ptr + node * gating_gradient_stride_node + gate_offset,
            gate_gradient,
            mask=mask,
        )


def activation_hvp(
    grad,
    hidden, gating, hidden_bias, gating_bias,
    hidden_direction, gating_direction,
    hidden_bias_direction=None,
    gating_bias_direction=None,
    *,
    lmax,
    grad_direction=None,
    need_bias_gradients=True,
):
    original_gating_shape = gating.shape
    gating, rows, nodes, channels, row_degrees = activation_layout(
        hidden, gating, hidden_bias, gating_bias, lmax
    )
    gating_direction = gating_direction.view_as(gating)
    has_grad_direction = grad_direction is not None
    has_bias_direction = (
        hidden_bias_direction is not None or gating_bias_direction is not None
    )
    if has_bias_direction:
        if hidden_bias_direction is None:
            hidden_bias_direction = torch.zeros_like(hidden_bias)
        if gating_bias_direction is None:
            gating_bias_direction = torch.zeros_like(gating_bias)
    incoming_direction = grad if grad_direction is None else grad_direction
    hidden_bias_direction = (
        hidden_bias if hidden_bias_direction is None else hidden_bias_direction
    )
    gating_bias_direction = (
        gating_bias if gating_bias_direction is None else gating_bias_direction
    )
    grad_output_gradient = torch.empty_like(grad)
    hidden_gradient = torch.empty_like(hidden)
    gating_gradient = torch.empty_like(gating)
    hvp_kernel[(nodes,)](
        grad,
        incoming_direction,
        hidden, gating, hidden_bias, gating_bias,
        hidden_direction, gating_direction,
        hidden_bias_direction,
        gating_bias_direction,
        grad_output_gradient,
        hidden_gradient, gating_gradient,
        *grad.stride(),
        *incoming_direction.stride(),
        *hidden.stride(),
        gating.stride(0),
        *hidden_direction.stride(),
        gating_direction.stride(0),
        *grad_output_gradient.stride(),
        *hidden_gradient.stride(),
        gating_gradient.stride(0),
        channels=channels,
        lmax=lmax,
        row_degrees=row_degrees,
        rows=rows,
        has_grad_direction=has_grad_direction,
        has_bias_direction=has_bias_direction,
        block_channels=1 << (channels - 1).bit_length(),
        num_warps=4,
    )
    gradients = (
        grad_output_gradient,
        hidden_gradient,
        gating_gradient.view(original_gating_shape),
    )
    if need_bias_gradients:
        return (
            *gradients,
            *bias_gradients(
                hidden_gradient, gating_gradient, hidden_bias, gating_bias
            ),
        )
    return gradients


@triton.jit
def grid_product_double_backward_kernel(
    grad_output_ptr, pair_ptr, gate_ptr,
    pair_direction_ptr, gate_direction_ptr,
    grad_output_gradient_ptr,
    pair_gradient_ptr, gate_gradient_ptr,
    grad_output_sn, grad_output_sg, grad_output_sc,
    pair_direction_sn, pair_direction_sg, pair_direction_sc,
    gate_direction_sn, gate_direction_sc,
    nodes,
    grids: tl.constexpr,
    channels: tl.constexpr,
    wide_index: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    if wide_index:
        node = node.to(tl.int64)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    mask = channel < channels
    gate = tl.load(gate_ptr + node * channels + channel, mask=mask)
    gate_direction = tl.load(
        gate_direction_ptr + node * gate_direction_sn + channel * gate_direction_sc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate)
    sigmoid_prime = sigmoid_gate * (1.0 - sigmoid_gate)
    sigmoid_second = sigmoid_prime * (1.0 - 2.0 * sigmoid_gate)
    gate_gradient = tl.zeros((block_channels,), dtype=tl.float32)
    for grid in tl.range(0, grids):
        row = grid * nodes + node
        pair_base = row * 2 * channels + channel
        output_base = row * channels + channel
        grad_output_offset = (
            grid * grad_output_sn + node * grad_output_sg + channel * grad_output_sc
        )
        pair_direction_offset = (
            grid * pair_direction_sn
            + node * pair_direction_sg
            + channel * pair_direction_sc
        )
        cotangent = tl.load(grad_output_ptr + grad_output_offset, mask=mask, other=0.0).to(tl.float32)
        first = tl.load(pair_ptr + pair_base, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(pair_ptr + pair_base + channels, mask=mask, other=0.0).to(tl.float32)
        first_direction = tl.load(pair_direction_ptr + pair_direction_offset, mask=mask, other=0.0).to(tl.float32)
        second_direction = tl.load(
            pair_direction_ptr + pair_direction_offset + channels * pair_direction_sc,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_output_gradient = (
            sigmoid_gate * (first_direction * second + first * second_direction)
            + first * second * sigmoid_prime * gate_direction
        )
        tl.store(grad_output_gradient_ptr + output_base, grad_output_gradient, mask=mask)
        common = cotangent * sigmoid_prime * gate_direction
        tl.store(
            pair_gradient_ptr + pair_base,
            cotangent * second_direction * sigmoid_gate + common * second,
            mask=mask,
        )
        tl.store(
            pair_gradient_ptr + pair_base + channels,
            cotangent * first_direction * sigmoid_gate + common * first,
            mask=mask,
        )
        gate_gradient += cotangent * (
            sigmoid_prime * (first_direction * second + first * second_direction)
            + first * second * sigmoid_second * gate_direction
        )
    tl.store(gate_gradient_ptr + node * channels + channel, gate_gradient, mask=mask)


def double_backward(
    grad_output: torch.Tensor,
    pair: torch.Tensor,
    gate: torch.Tensor,
    pair_direction: torch.Tensor,
    gate_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes, grids, channels = validate(pair, gate)
    if grad_output.shape != (*pair.shape[:-1], channels):
        raise ValueError(f"unexpected grid-product gradient: {grad_output.shape}")
    if pair_direction.shape != pair.shape or gate_direction.shape != gate.shape:
        raise ValueError("unexpected grid-product double-backward direction")
    grad_output_gradient = torch.empty(
        (*pair.shape[:-1], channels),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    pair_gradient = torch.empty_like(pair)
    gate_gradient = torch.empty_like(gate)
    wide_index = pair.numel() > torch.iinfo(torch.int32).max
    block_channels = 128
    grid_product_double_backward_kernel[
        (nodes, triton.cdiv(channels, block_channels))
    ](
        grad_output, pair, gate,
        pair_direction, gate_direction,
        grad_output_gradient,
        pair_gradient, gate_gradient,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        pair_direction.stride(0), pair_direction.stride(1), pair_direction.stride(2),
        gate_direction.stride(0), gate_direction.stride(2),
        nodes,
        grids=grids, channels=channels,
        wide_index=wide_index,
        block_channels=block_channels,
        num_warps=4,
    )
    return grad_output_gradient, pair_gradient, gate_gradient


@triton.jit
def degree_expand_four_kernel(
    first_ptr,
    second_ptr,
    third_ptr,
    fourth_ptr,
    expand_index_ptr,
    first_output_ptr,
    second_output_ptr,
    third_output_ptr,
    fourth_output_ptr,
    matrix_elements,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    row = offsets // matrix_elements
    column = offsets - row * matrix_elements
    degree = tl.load(expand_index_ptr + row, mask=mask, other=0)
    source = degree * matrix_elements + column
    tl.store(first_output_ptr + offsets, tl.load(first_ptr + source, mask=mask, other=0.0), mask=mask)
    tl.store(second_output_ptr + offsets, tl.load(second_ptr + source, mask=mask, other=0.0), mask=mask)
    tl.store(third_output_ptr + offsets, tl.load(third_ptr + source, mask=mask, other=0.0), mask=mask)
    tl.store(fourth_output_ptr + offsets, tl.load(fourth_ptr + source, mask=mask, other=0.0), mask=mask)


@triton.jit
def degree_expand_pair_kernel(
    first_ptr,
    second_ptr,
    expand_index_ptr,
    first_output_ptr,
    second_output_ptr,
    matrix_elements,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    row = offsets // matrix_elements
    column = offsets - row * matrix_elements
    degree = tl.load(expand_index_ptr + row, mask=mask, other=0)
    source = degree * matrix_elements + column
    tl.store(first_output_ptr + offsets, tl.load(first_ptr + source, mask=mask, other=0.0), mask=mask)
    tl.store(second_output_ptr + offsets, tl.load(second_ptr + source, mask=mask, other=0.0), mask=mask)


def degree_expand_four(first, second, third, fourth, expand_index):
    shape = first.shape
    if any(
        value.shape[0] != shape[0] or value.numel() != first.numel()
        for value in (second, third, fourth)
    ):
        raise ValueError("spectral FFN maps must share degree count and matrix size")
    rows = expand_index.numel()
    outputs = tuple(
        torch.empty((rows, value.shape[1], value.shape[2]), device=first.device, dtype=first.dtype)
        for value in (first, second, third, fourth)
    )
    block = 256
    elements = outputs[0].numel()
    degree_expand_four_kernel[(triton.cdiv(elements, block),)](
        first,
        second,
        third,
        fourth,
        expand_index,
        *outputs,
        shape[1] * shape[2],
        elements,
        block=block,
        num_warps=4,
    )
    return outputs


def degree_expand_pair(first, second, expand_index):
    if second.shape[0] != first.shape[0] or second.numel() != first.numel():
        raise ValueError("spectral FFN maps must share degree count and matrix size")
    rows = expand_index.numel()
    outputs = tuple(
        torch.empty((rows, value.shape[1], value.shape[2]), device=first.device, dtype=first.dtype)
        for value in (first, second)
    )
    block = 256
    elements = outputs[0].numel()
    degree_expand_pair_kernel[(triton.cdiv(elements, block),)](
        first,
        second,
        expand_index,
        *outputs,
        first.shape[1] * first.shape[2],
        elements,
        block=block,
        num_warps=4,
    )
    return outputs


def norm_residual_hvp(
    x, norm_weight, norm_bias,
    coefficient_input, gating, hidden, activated,
    normalized_gradient, raw_gradient, activated_gradient,
    hidden_gradient, gating_gradient,
    gating_weight, gating_bias,
    first_weight, first_bias,
    second_weight, second_bias,
    expand_index,
    input_direction,
    norm_weight_direction, norm_bias_direction,
    gating_weight_direction, gating_bias_direction,
    first_weight_direction, first_bias_direction,
    second_weight_direction, second_bias_direction,
    *,
    lmax, epsilon,
):
    normalized_direction = jvp(
        x, norm_weight, norm_bias,
        input_direction,
        norm_weight_direction, norm_bias_direction,
        lmax=lmax, epsilon=epsilon,
        coefficient_major_output=True,
    )
    coefficient_direction = normalized_direction.permute(1, 0, 2)
    del normalized_direction
    gating_direction = torch.mm(coefficient_direction[0], gating_weight.transpose(0, 1))
    torch.addmm(
        gating_direction, coefficient_input[0], gating_weight_direction.transpose(0, 1),
        beta=1.0, alpha=1.0, out=gating_direction,
    )
    (
        first_expanded, first_expanded_direction,
        second_expanded, second_expanded_direction,
    ) = degree_expand_four(
        first_weight, first_weight_direction,
        second_weight, second_weight_direction,
        expand_index,
    )
    hidden_direction = torch.bmm(coefficient_direction, first_expanded.transpose(1, 2))
    torch.baddbmm(
        hidden_direction, coefficient_input, first_expanded_direction.transpose(1, 2),
        beta=1.0, alpha=1.0, out=hidden_direction,
    )

    activated_cotangent_direction = torch.bmm(raw_gradient, second_expanded_direction)
    (
        activated_direction,
        hidden_hessian, gating_hessian,
    ) = activation_hvp(
        activated_gradient,
        hidden, gating, first_bias, gating_bias,
        hidden_direction, gating_direction,
        first_bias_direction, gating_bias_direction,
        lmax=lmax,
        grad_direction=activated_cotangent_direction,
        need_bias_gradients=False,
    )
    del hidden_direction, gating_direction, activated_cotangent_direction
    raw_direction = torch.bmm(activated_direction, second_expanded.transpose(1, 2))
    torch.baddbmm(
        raw_direction, activated, second_expanded_direction.transpose(1, 2),
        beta=1.0, alpha=1.0, out=raw_direction,
    )
    del second_expanded, second_expanded_direction
    grad_output_gradient = epilogue_forward(
        raw_direction, second_bias_direction, input_direction
    )
    del raw_direction

    coefficient_hessian = torch.bmm(hidden_hessian, first_expanded)
    torch.baddbmm(
        coefficient_hessian, hidden_gradient, first_expanded_direction,
        beta=1.0, alpha=1.0, out=coefficient_hessian,
    )
    torch.addmm(
        coefficient_hessian[0], gating_hessian, gating_weight,
        beta=1.0, alpha=1.0, out=coefficient_hessian[0],
    )
    torch.addmm(
        coefficient_hessian[0], gating_gradient, gating_weight_direction,
        beta=1.0, alpha=1.0, out=coefficient_hessian[0],
    )
    del first_expanded, first_expanded_direction

    gating_weight_hessian = torch.mm(gating_hessian.transpose(0, 1), coefficient_input[0])
    torch.addmm(
        gating_weight_hessian, gating_gradient.transpose(0, 1), coefficient_direction[0],
        beta=1.0, alpha=1.0, out=gating_weight_hessian,
    )
    grouped = use_grouped_weight_gradients(x, hidden, lmax)
    first_product = weight_derivative(
        hidden_hessian, coefficient_input, lmax + 1, grouped=grouped,
        tangent=(hidden_gradient, coefficient_direction),
    )
    del coefficient_direction
    second_product = weight_derivative(raw_gradient, activated_direction, lmax + 1, grouped=grouped)
    del activated_direction
    (
        first_weight_hessian, second_weight_hessian,
        first_bias_hessian, gating_bias_hessian, second_bias_hessian,
    ) = parameter_reduce(
        first_product, second_product,
        hidden_hessian, gating_hessian,
        degree_count=first_weight.shape[0],
        weights_grouped=grouped,
    )
    del first_product, second_product
    del hidden_hessian, gating_hessian
    input_hessian, norm_weight_hessian, norm_bias_hessian = hvp_with_cotangent_direction(
        normalized_gradient,
        coefficient_hessian.permute(1, 0, 2),
        x, norm_weight, norm_bias,
        input_direction,
        norm_weight_direction,
        lmax=lmax, epsilon=epsilon,
    )
    return (
        grad_output_gradient,
        input_hessian,
        norm_weight_hessian, norm_bias_hessian,
        gating_weight_hessian, gating_bias_hessian,
        first_weight_hessian, first_bias_hessian,
        second_weight_hessian, second_bias_hessian,
    )


def hvp_input_direction(
    x, norm_weight, norm_bias,
    coefficient_input, gating, hidden, activated,
    normalized_gradient, raw_gradient, activated_gradient,
    hidden_gradient, gating_gradient,
    gating_weight, gating_bias,
    first_weight, first_bias,
    second_weight, second_bias,
    expand_index,
    input_direction,
    *,
    lmax, epsilon,
):
    normalized_direction = jvp_input(
        x, norm_weight, norm_bias,
        input_direction,
        lmax=lmax, epsilon=epsilon,
        coefficient_major_output=True,
    )
    coefficient_direction = normalized_direction.permute(1, 0, 2)
    gating_direction = torch.mm(coefficient_direction[0], gating_weight.transpose(0, 1))
    first_expanded, second_expanded = degree_expand_pair(
        first_weight, second_weight, expand_index
    )
    hidden_direction = torch.bmm(coefficient_direction, first_expanded.transpose(1, 2))
    (
        activated_direction,
        hidden_hessian, gating_hessian,
    ) = activation_hvp(
        activated_gradient,
        hidden, gating, first_bias, gating_bias,
        hidden_direction, gating_direction,
        lmax=lmax,
        need_bias_gradients=False,
    )
    raw_direction = torch.bmm(activated_direction, second_expanded.transpose(1, 2))

    coefficient_hessian = torch.bmm(hidden_hessian, first_expanded)
    torch.addmm(
        coefficient_hessian[0], gating_hessian, gating_weight,
        beta=1.0, alpha=1.0, out=coefficient_hessian[0],
    )
    gating_weight_hessian = torch.mm(gating_hessian.transpose(0, 1), coefficient_input[0])
    torch.addmm(
        gating_weight_hessian, gating_gradient.transpose(0, 1), coefficient_direction[0],
        beta=1.0, alpha=1.0, out=gating_weight_hessian,
    )
    grouped = use_grouped_weight_gradients(x, hidden, lmax)
    first_product = weight_derivative(
        hidden_hessian, coefficient_input, lmax + 1, grouped=grouped,
        tangent=(hidden_gradient, coefficient_direction),
    )
    second_product = weight_derivative(raw_gradient, activated_direction, lmax + 1, grouped=grouped)
    (
        first_weight_hessian, second_weight_hessian,
        first_bias_hessian, gating_bias_hessian, second_bias_hessian,
    ) = parameter_reduce(
        first_product, second_product,
        hidden_hessian, gating_hessian,
        degree_count=first_weight.shape[0],
        weights_grouped=grouped,
    )
    (
        grad_output_gradient,
        input_hessian,
        norm_weight_hessian, norm_bias_hessian,
    ) = hvp_with_cotangent_input_and_output(
        normalized_gradient,
        coefficient_hessian.permute(1, 0, 2),
        raw_direction,
        x, norm_weight, norm_bias,
        input_direction,
        lmax=lmax, epsilon=epsilon,
    )
    return (
        grad_output_gradient,
        input_hessian,
        norm_weight_hessian, norm_bias_hessian,
        gating_weight_hessian, gating_bias_hessian,
        first_weight_hessian, first_bias_hessian,
        second_weight_hessian, second_bias_hessian,
    )
