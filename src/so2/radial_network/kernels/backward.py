from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import grouped_layer_norm_silu_validate, layer_norm_silu_validate, use_tiled_grouped_norm


@triton.jit
def distance_atom_pair_backward_kernel(
    grad_output_ptr,
    atomic_numbers_ptr,
    edge_index_ptr,
    grad_distance_ptr,
    grad_source_weight_ptr,
    grad_target_weight_ptr,
    rows,
    distance_channels: tl.constexpr,
    embedding_channels: tl.constexpr,
    total_channels: tl.constexpr,
    block_channels: tl.constexpr,
    compute_embedding_gradients: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    distance_mask = channel < distance_channels
    distance_gradient = tl.load(
        grad_output_ptr + row * total_channels + channel,
        mask=distance_mask,
        other=0.0,
    )
    tl.store(
        grad_distance_ptr + row * distance_channels + channel,
        distance_gradient,
        mask=distance_mask,
    )
    if compute_embedding_gradients:
        source_node = tl.load(edge_index_ptr + row)
        target_node = tl.load(edge_index_ptr + rows + row)
        source = tl.load(atomic_numbers_ptr + source_node)
        target = tl.load(atomic_numbers_ptr + target_node)
        embedding_mask = channel < embedding_channels
        source_gradient = tl.load(
            grad_output_ptr + row * total_channels + distance_channels + channel,
            mask=embedding_mask,
            other=0.0,
        )
        target_gradient = tl.load(
            grad_output_ptr
            + row * total_channels
            + distance_channels
            + embedding_channels
            + channel,
            mask=embedding_mask,
            other=0.0,
        )
        tl.atomic_add(
            grad_source_weight_ptr + source * embedding_channels + channel,
            source_gradient,
            mask=embedding_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.atomic_add(
            grad_target_weight_ptr + target * embedding_channels + channel,
            target_gradient,
            mask=embedding_mask,
            sem="relaxed",
            scope="gpu",
        )


def distance_atom_pair_backward(
    grad_output,
    source_weight,
    target_weight,
    atomic_numbers,
    edge_index,
    distance_channels,
    *,
    compute_embedding_gradients=True,
    packed_gradients=False,
):
    allow_atomic_reduction("distance_atom_pair backward")
    rows = grad_output.shape[0]
    embedding_channels = source_weight.shape[1]
    grad_distance = torch.empty(
        (rows, distance_channels),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    grad_source_weight = grad_target_weight = None
    if compute_embedding_gradients:
        if packed_gradients:
            grad_source_weight, grad_target_weight = source_weight.new_zeros(
                (2, *source_weight.shape),
            ).unbind(0)
        else:
            grad_source_weight = torch.zeros_like(source_weight)
            grad_target_weight = torch.zeros_like(target_weight)
    distance_atom_pair_backward_kernel[(rows,)](
        grad_output,
        atomic_numbers,
        edge_index,
        grad_distance,
        grad_source_weight if grad_source_weight is not None else source_weight,
        grad_target_weight if grad_target_weight is not None else target_weight,
        rows,
        distance_channels=distance_channels,
        embedding_channels=embedding_channels,
        total_channels=grad_output.shape[1],
        block_channels=triton.next_power_of_2(
            max(distance_channels, embedding_channels)
        ),
        compute_embedding_gradients=compute_embedding_gradients,
        num_warps=4,
    )
    return grad_distance, grad_source_weight, grad_target_weight


@triton.jit
def grouped_layer_norm_silu_backward_kernel(
    grad_output_ptr, input_ptr, input_bias_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    grad_input_ptr, grad_input_bias_ptr, grad_weight_ptr, grad_bias_ptr,
    rows_per_group,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    rows_per_program: tl.constexpr,
    compute_parameter_gradients: tl.constexpr,
    one_row_tile: tl.constexpr = False,
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
    flat_parameter_offset = group * channels + channel_1d
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
    mean = tl.load(mean_ptr + row, mask=row_in_group < rows_per_group, other=0.0)
    rstd = tl.load(rstd_ptr + row, mask=row_in_group < rows_per_group, other=0.0)
    normalized = (inputs - mean) * rstd
    affine = normalized * weight + bias
    sigmoid = tl.sigmoid(affine)
    silu_prime = sigmoid + affine * sigmoid * (1.0 - sigmoid)
    affine_gradient = grad_output * silu_prime
    weighted_gradient = affine_gradient * weight
    inv_channels = 1.0 / channels
    weighted_mean = tl.sum(weighted_gradient, axis=1)[:, None] * inv_channels
    weighted_normalized_mean = (
        tl.sum(weighted_gradient * normalized, axis=1)[:, None] * inv_channels
    )
    grad_input = rstd * (
        weighted_gradient - weighted_mean - normalized * weighted_normalized_mean
    )
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)
    if compute_parameter_gradients:
        offset_mask = channel_1d < channels
        input_bias_gradient = tl.sum(tl.where(mask, grad_input, 0.0), axis=0)
        weight_gradient = tl.sum(tl.where(mask, affine_gradient * normalized, 0.0), axis=0)
        bias_gradient = tl.sum(tl.where(mask, affine_gradient, 0.0), axis=0)
        if one_row_tile:
            tl.store(grad_input_bias_ptr + flat_parameter_offset, input_bias_gradient, offset_mask)
            tl.store(grad_weight_ptr + flat_parameter_offset, weight_gradient, offset_mask)
            tl.store(grad_bias_ptr + flat_parameter_offset, bias_gradient, offset_mask)
        else:
            tl.atomic_add(grad_input_bias_ptr + flat_parameter_offset, input_bias_gradient,
                          offset_mask, sem="relaxed", scope="gpu")
            tl.atomic_add(grad_weight_ptr + flat_parameter_offset, weight_gradient,
                          offset_mask, sem="relaxed", scope="gpu")
            tl.atomic_add(grad_bias_ptr + flat_parameter_offset, bias_gradient,
                          offset_mask, sem="relaxed", scope="gpu")


def grouped_layer_norm_silu_backward(
    grad_output, inputs, input_bias, weight, bias, mean, rstd,
    num_warps, *, packed_gradients=False,
):
    allow_atomic_reduction("grouped radial LayerNorm-SiLU parameter gradients")
    if packed_gradients:
        # The indexed hidden-stage kernel has already validated these tensors.
        groups, rows_per_group, channels = inputs.shape
        rows_per_program = 8
    else:
        groups, rows_per_group, channels = grouped_layer_norm_silu_validate(inputs, input_bias, weight, bias)
        rows_per_program = 8 if num_warps == 4 and use_tiled_grouped_norm(inputs) else 4
    one_row_tile = 0 < rows_per_group <= 8 and channels <= 256
    if one_row_tile:
        rows_per_program = triton.next_power_of_2(rows_per_group)
    if packed_gradients:
        allocate = input_bias.new_empty if one_row_tile else input_bias.new_zeros
        grad_input_bias, grad_weight, grad_bias = allocate((3, groups, channels)).unbind(0)
    else:
        allocate = torch.empty_like if one_row_tile else torch.zeros_like
        grad_input_bias, grad_weight, grad_bias = allocate(input_bias), allocate(weight), allocate(bias)
    grad_inputs = torch.empty_like(inputs)
    block_channels = triton.next_power_of_2(channels)
    grouped_layer_norm_silu_backward_kernel[(triton.cdiv(rows_per_group, rows_per_program), groups)](
        grad_output, inputs, input_bias, weight, bias, mean, rstd,
        grad_inputs, grad_input_bias, grad_weight, grad_bias,
        rows_per_group,
        channels=channels,
        block_channels=block_channels,
        rows_per_program=rows_per_program,
        compute_parameter_gradients=True,
        one_row_tile=one_row_tile,
        num_warps=num_warps,
    )
    return grad_inputs, grad_input_bias, grad_weight, grad_bias


def grouped_layer_norm_silu_backward_input(
    grad_output, inputs, input_bias, weight, bias, mean, rstd,
    num_warps,
):
    groups, rows_per_group, channels = grouped_layer_norm_silu_validate(inputs, input_bias, weight, bias)
    grad_inputs = torch.empty_like(inputs)
    rows_per_program = 4
    block_channels = triton.next_power_of_2(channels)
    grouped_layer_norm_silu_backward_kernel[(triton.cdiv(rows_per_group, rows_per_program), groups)](
        grad_output, inputs, input_bias, weight, bias, mean, rstd,
        grad_inputs, input_bias, weight, bias, rows_per_group,
        channels=channels,
        block_channels=block_channels,
        rows_per_program=rows_per_program,
        compute_parameter_gradients=False,
        one_row_tile=False,
        num_warps=num_warps,
    )
    return grad_inputs


@triton.jit
def layer_norm_silu_backward_kernel(
    grad_output_ptr, input_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
    rows,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    rows_per_program: tl.constexpr,
    compute_parameter_gradients: tl.constexpr,
):
    row = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)[:, None]
    channel_1d = tl.arange(0, block_channels)
    channel = channel_1d[None, :]
    mask = (row < rows) & (channel < channels)
    offset = row * channels + channel
    inputs = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=channel < channels, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + channel, mask=channel < channels, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    normalized = (inputs - mean) * rstd
    affine = normalized * weight + bias
    sigmoid = tl.sigmoid(affine)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    silu_prime = sigmoid + affine * sigmoid_prime
    affine_gradient = grad_output * silu_prime
    weighted_gradient = affine_gradient * weight
    inv_channels = 1.0 / channels
    weighted_mean = (tl.sum(weighted_gradient, axis=1) * inv_channels)[:, None]
    weighted_normalized_mean = (
        tl.sum(weighted_gradient * normalized, axis=1) * inv_channels
    )[:, None]
    grad_input = rstd * (
        weighted_gradient - weighted_mean - normalized * weighted_normalized_mean
    )
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)
    if compute_parameter_gradients:
        grad_weight = tl.sum(tl.where(mask, affine_gradient * normalized, 0.0), axis=0)
        grad_bias = tl.sum(tl.where(mask, affine_gradient, 0.0), axis=0)
        tl.atomic_add(
            grad_weight_ptr + channel_1d,
            grad_weight,
            mask=channel_1d < channels,
            sem="relaxed",
            scope="gpu",
        )
        tl.atomic_add(
            grad_bias_ptr + channel_1d,
            grad_bias,
            mask=channel_1d < channels,
            sem="relaxed",
            scope="gpu",
        )


def layer_norm_silu_backward(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, channels = layer_norm_silu_validate(inputs, weight, bias)
    allow_atomic_reduction("radial LayerNorm-SiLU parameter gradients")
    grad_inputs = torch.empty_like(inputs)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    rows_per_program = 8
    block_channels = triton.next_power_of_2(channels)
    layer_norm_silu_backward_kernel[(triton.cdiv(rows, rows_per_program),)](
        grad_output, inputs, weight, bias, mean, rstd,
        grad_inputs, grad_weight, grad_bias,
        rows,
        channels=channels,
        block_channels=block_channels,
        rows_per_program=rows_per_program,
        compute_parameter_gradients=True,
        num_warps=8,
    )
    return grad_inputs, grad_weight, grad_bias


def layer_norm_silu_backward_input(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor:
    rows, channels = layer_norm_silu_validate(inputs, weight, bias)
    grad_inputs = torch.empty_like(inputs)
    rows_per_program = 8
    block_channels = triton.next_power_of_2(channels)
    layer_norm_silu_backward_kernel[(triton.cdiv(rows, rows_per_program),)](
        grad_output, inputs, weight, bias, mean, rstd,
        grad_inputs, weight, bias, rows,
        channels=channels,
        block_channels=block_channels,
        rows_per_program=rows_per_program,
        compute_parameter_gradients=False,
        num_warps=8,
    )
    return grad_inputs


@triton.jit
def frozen_radial_norm_backward(
    gy, x, input_bias, weight, bias, mean, rstd, gx, rows,
    groups: tl.constexpr, channels: tl.constexpr, tile: tl.constexpr, packed: tl.constexpr,
):
    group = tl.program_id(1)
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    c = tl.arange(0, channels)
    live = row < rows
    grouped_row = group * rows + row
    if packed:
        x_row = row * groups + group
    else:
        x_row = grouped_row
    parameter = group * channels + c
    x_offset = x_row[:, None] * channels + c[None, :]
    y_offset = grouped_row[:, None] * channels + c[None, :]
    z = tl.load(x + x_offset, live[:, None], 0.) + tl.load(input_bias + parameter)[None, :]
    grad = tl.load(gy + y_offset, live[:, None], 0.)
    avg = tl.load(mean + grouped_row, live, 0.)
    inv = tl.load(rstd + grouped_row, live, 0.)
    coeff = tl.load(weight + parameter)[None, :]
    normal = (z - avg[:, None]) * inv[:, None]
    affine = normal * coeff + tl.load(bias + parameter)[None, :]
    sigmoid = tl.sigmoid(affine)
    prime = sigmoid + affine * sigmoid * (1.0 - sigmoid)
    affine_gradient = grad * prime
    weighted_gradient = affine_gradient * coeff
    avg_weighted = tl.sum(weighted_gradient, 1) / channels
    avg_normalized = tl.sum(weighted_gradient * normal, 1) / channels
    gz = inv[:, None] * (weighted_gradient - avg_weighted[:, None] - normal * avg_normalized[:, None])
    tl.store(gx + x_offset, gz, live[:, None])
