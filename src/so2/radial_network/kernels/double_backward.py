from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import grouped_layer_norm_silu_validate, layer_norm_silu_validate


@triton.jit
def double_backward_kernel(
    grad_output_ptr, input_ptr, input_bias_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    grad_input_cotangent_ptr,
    grad_input_bias_cotangent_ptr,
    grad_weight_cotangent_ptr,
    grad_bias_cotangent_ptr,
    grad_grad_output_ptr,
    grad_input_ptr, grad_input_bias_ptr, grad_weight_ptr, grad_bias_ptr,
    rows_per_group,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    rows_per_program: tl.constexpr,
    has_grad_weight_cotangent: tl.constexpr,
    has_grad_bias_cotangent: tl.constexpr,
):
    group = tl.program_id(1)
    row_in_group = (
        tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)[:, None]
    )
    channel_1d = tl.arange(0, block_channels)
    channel = channel_1d[None, :]
    row = group * rows_per_group + row_in_group
    mask = (row_in_group < rows_per_group) & (channel < channels)
    parameter_mask = channel < channels
    offset = row * channels + channel
    parameter_offset = group * channels + channel
    inputs = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    inputs += tl.load(
        input_bias_ptr + parameter_offset, mask=parameter_mask, other=0.0
    ).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + parameter_offset, mask=parameter_mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + parameter_offset, mask=parameter_mask, other=0.0).to(
        tl.float32
    )
    input_direction = tl.load(
        grad_input_cotangent_ptr + offset, mask=mask, other=0.0
    ).to(tl.float32)
    input_direction += tl.load(
        grad_input_bias_cotangent_ptr + parameter_offset,
        mask=parameter_mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + row, mask=row_in_group < rows_per_group, other=0.0).to(
        tl.float32
    )
    rstd = tl.load(rstd_ptr + row, mask=row_in_group < rows_per_group, other=0.0).to(
        tl.float32
    )

    inv_channels = 1.0 / channels
    normalized = (inputs - mean) * rstd
    affine = normalized * weight + bias
    sigmoid = tl.sigmoid(affine)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    silu_prime = sigmoid + affine * sigmoid_prime
    silu_second = sigmoid_prime * (2.0 + affine * (1.0 - 2.0 * sigmoid))
    affine_gradient = grad_output * silu_prime

    mean_direction = tl.sum(input_direction, axis=1)[:, None] * inv_channels
    mean_direction_normalized = (
        tl.sum(input_direction * normalized, axis=1)[:, None] * inv_channels
    )
    projected_direction = (
        input_direction - mean_direction - normalized * mean_direction_normalized
    )
    normalized_direction = rstd * projected_direction

    weighted_affine_gradient = affine_gradient * weight
    mean_weighted_normalized = (
        tl.sum(weighted_affine_gradient * normalized, axis=1)[:, None] * inv_channels
    )
    intermediate = (
        -mean_direction_normalized * weighted_affine_gradient
        - mean_weighted_normalized * input_direction
    )
    mean_intermediate = tl.sum(intermediate, axis=1)[:, None] * inv_channels
    mean_intermediate_normalized = (
        tl.sum(intermediate * normalized, axis=1)[:, None] * inv_channels
    )
    mean_weighted_projected = (
        tl.sum(weighted_affine_gradient * projected_direction, axis=1)[:, None]
        * inv_channels
    )
    grad_input = (
        rstd
        * rstd
        * (
            intermediate
            - mean_intermediate
            - normalized * mean_intermediate_normalized
            - normalized * mean_weighted_projected
        )
    )

    affine_direction = weight * normalized_direction
    if has_grad_weight_cotangent:
        weight_direction = tl.load(
            grad_weight_cotangent_ptr + parameter_offset,
            mask=parameter_mask,
            other=0.0,
        ).to(tl.float32)
        affine_direction += weight_direction * normalized
        weighted_parameter = affine_gradient * weight_direction
        mean_weighted_parameter = (
            tl.sum(weighted_parameter, axis=1)[:, None] * inv_channels
        )
        mean_weighted_parameter_normalized = (
            tl.sum(weighted_parameter * normalized, axis=1)[:, None] * inv_channels
        )
        grad_input += rstd * (
            weighted_parameter
            - mean_weighted_parameter
            - normalized * mean_weighted_parameter_normalized
        )
    if has_grad_bias_cotangent:
        bias_direction = tl.load(
            grad_bias_cotangent_ptr + parameter_offset,
            mask=parameter_mask,
            other=0.0,
        ).to(tl.float32)
        affine_direction += bias_direction

    second_affine_gradient = affine_direction * grad_output * silu_second
    weighted_second = second_affine_gradient * weight
    mean_weighted_second = tl.sum(weighted_second, axis=1)[:, None] * inv_channels
    mean_weighted_second_normalized = (
        tl.sum(weighted_second * normalized, axis=1)[:, None] * inv_channels
    )
    grad_input += rstd * (
        weighted_second
        - mean_weighted_second
        - normalized * mean_weighted_second_normalized
    )
    grad_grad_output = affine_direction * silu_prime
    grad_weight = (
        affine_gradient * normalized_direction + second_affine_gradient * normalized
    )

    tl.store(grad_grad_output_ptr + offset, grad_grad_output, mask=mask)
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)
    flat_parameter_offset = group * channels + channel_1d
    tl.atomic_add(
        grad_input_bias_ptr + flat_parameter_offset,
        tl.sum(tl.where(mask, grad_input, 0.0), axis=0),
        mask=channel_1d < channels,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        grad_weight_ptr + flat_parameter_offset,
        tl.sum(tl.where(mask, grad_weight, 0.0), axis=0),
        mask=channel_1d < channels,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        grad_bias_ptr + flat_parameter_offset,
        tl.sum(tl.where(mask, second_affine_gradient, 0.0), axis=0),
        mask=channel_1d < channels,
        sem="relaxed",
        scope="gpu",
    )


def grouped_layer_norm_silu_double_backward(
    grad_output, inputs, input_bias, weight, bias, mean, rstd,
    grad_input_cotangent,
    grad_input_bias_cotangent,
    grad_weight_cotangent,
    grad_bias_cotangent,
):
    groups, rows_per_group, channels = grouped_layer_norm_silu_validate(inputs, input_bias, weight, bias)
    allow_atomic_reduction("grouped radial LayerNorm-SiLU double backward")
    grad_grad_output = torch.empty_like(grad_output)
    grad_inputs = torch.empty_like(inputs)
    grad_input_bias = torch.zeros_like(input_bias)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    dummy = inputs
    block_channels = triton.next_power_of_2(channels)
    rows_per_program = 4
    double_backward_kernel[(triton.cdiv(rows_per_group, rows_per_program), groups)](
        grad_output, inputs, input_bias, weight, bias, mean, rstd,
        grad_input_cotangent,
        grad_input_bias_cotangent,
        dummy if grad_weight_cotangent is None else grad_weight_cotangent,
        dummy if grad_bias_cotangent is None else grad_bias_cotangent,
        grad_grad_output,
        grad_inputs, grad_input_bias, grad_weight, grad_bias,
        rows_per_group,
        channels=channels,
        block_channels=block_channels,
        rows_per_program=rows_per_program,
        has_grad_weight_cotangent=grad_weight_cotangent is not None,
        has_grad_bias_cotangent=grad_bias_cotangent is not None,
        num_warps=8,
    )
    return grad_grad_output, grad_inputs, grad_input_bias, grad_weight, grad_bias


@triton.jit
def layer_norm_silu_double_backward_kernel(
    grad_output_ptr, input_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    grad_input_cotangent_ptr,
    grad_weight_cotangent_ptr,
    grad_bias_cotangent_ptr,
    grad_grad_output_ptr,
    grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    has_grad_weight_cotangent: tl.constexpr,
    has_grad_bias_cotangent: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    mask = channel < channels
    offset = row * channels + channel
    inputs = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    input_direction = tl.load(
        grad_input_cotangent_ptr + offset, mask=mask, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    inv_channels = 1.0 / channels
    normalized = (inputs - mean) * rstd
    affine = normalized * weight + bias
    sigmoid = tl.sigmoid(affine)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    silu_prime = sigmoid + affine * sigmoid_prime
    silu_second = sigmoid_prime * (2.0 + affine * (1.0 - 2.0 * sigmoid))
    affine_gradient = grad_output * silu_prime

    mean_direction = tl.sum(input_direction, axis=0) * inv_channels
    mean_direction_normalized = (
        tl.sum(input_direction * normalized, axis=0) * inv_channels
    )
    projected_direction = (
        input_direction - mean_direction - normalized * mean_direction_normalized
    )
    normalized_direction = rstd * projected_direction

    weighted_affine_gradient = affine_gradient * weight
    mean_weighted_normalized = (
        tl.sum(weighted_affine_gradient * normalized, axis=0) * inv_channels
    )
    intermediate = (
        -mean_direction_normalized * weighted_affine_gradient
        - mean_weighted_normalized * input_direction
    )
    mean_intermediate = tl.sum(intermediate, axis=0) * inv_channels
    mean_intermediate_normalized = (
        tl.sum(intermediate * normalized, axis=0) * inv_channels
    )
    mean_weighted_projected = (
        tl.sum(weighted_affine_gradient * projected_direction, axis=0) * inv_channels
    )
    grad_input = (
        rstd
        * rstd
        * (
            intermediate
            - mean_intermediate
            - normalized * mean_intermediate_normalized
            - normalized * mean_weighted_projected
        )
    )

    affine_direction = weight * normalized_direction
    if has_grad_weight_cotangent:
        weight_direction = tl.load(
            grad_weight_cotangent_ptr + channel, mask=mask, other=0.0
        ).to(tl.float32)
        affine_direction += weight_direction * normalized
        weighted_parameter = affine_gradient * weight_direction
        mean_weighted_parameter = tl.sum(weighted_parameter, axis=0) * inv_channels
        mean_weighted_parameter_normalized = (
            tl.sum(weighted_parameter * normalized, axis=0) * inv_channels
        )
        grad_input += rstd * (
            weighted_parameter
            - mean_weighted_parameter
            - normalized * mean_weighted_parameter_normalized
        )
    if has_grad_bias_cotangent:
        bias_direction = tl.load(
            grad_bias_cotangent_ptr + channel, mask=mask, other=0.0
        ).to(tl.float32)
        affine_direction += bias_direction

    second_affine_gradient = affine_direction * grad_output * silu_second
    weighted_second = second_affine_gradient * weight
    mean_weighted_second = tl.sum(weighted_second, axis=0) * inv_channels
    mean_weighted_second_normalized = (
        tl.sum(weighted_second * normalized, axis=0) * inv_channels
    )
    grad_input += rstd * (
        weighted_second
        - mean_weighted_second
        - normalized * mean_weighted_second_normalized
    )
    grad_grad_output = affine_direction * silu_prime
    grad_weight = (
        affine_gradient * normalized_direction + second_affine_gradient * normalized
    )

    tl.store(grad_grad_output_ptr + offset, grad_grad_output, mask=mask)
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)
    tl.atomic_add(
        grad_weight_ptr + channel,
        grad_weight,
        mask=mask,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        grad_bias_ptr + channel,
        second_affine_gradient,
        mask=mask,
        sem="relaxed",
        scope="gpu",
    )


def layer_norm_silu_double_backward(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_input_cotangent: torch.Tensor,
    grad_weight_cotangent: torch.Tensor | None,
    grad_bias_cotangent: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, channels = layer_norm_silu_validate(inputs, weight, bias)
    allow_atomic_reduction("radial LayerNorm-SiLU double backward")
    grad_grad_output = torch.empty_like(grad_output)
    grad_inputs = torch.empty_like(inputs)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    dummy = inputs
    block_channels = triton.next_power_of_2(channels)
    layer_norm_silu_double_backward_kernel[(rows,)](
        grad_output, inputs, weight, bias, mean, rstd,
        grad_input_cotangent,
        dummy if grad_weight_cotangent is None else grad_weight_cotangent,
        dummy if grad_bias_cotangent is None else grad_bias_cotangent,
        grad_grad_output,
        grad_inputs, grad_weight, grad_bias,
        channels=channels,
        block_channels=block_channels,
        has_grad_weight_cotangent=grad_weight_cotangent is not None,
        has_grad_bias_cotangent=grad_bias_cotangent is not None,
        num_warps=4,
    )
    return grad_grad_output, grad_inputs, grad_weight, grad_bias
