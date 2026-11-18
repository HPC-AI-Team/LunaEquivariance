from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value, store_sector_gradient
from ...utils.tensor_contracts import allow_atomic_reduction
from .backward import compact_tail_backward, store_attention_gradient
from .forward import receiver_sparse_forward_direction_kernel
from .utils import (
    ValueStorage,
    attention_dimensions,
    component_degree,
    empty_values,
    kernel_metadata,
    load_attention_value,
    logical_rows,
    m0_row_offset,
    attention_row_layout,
    receiver_softmax_direction_kernel,
    sum_forward_validate,
    use_degree_sum_node,
    validate_inputs,
    validate_row_ptr,
    value_blocks,
    wigner_offset,
)


@triton.jit
def receiver_attention_double_backward_kernel(
    logits_ptr,
    envelope_ptr,
    row_ptr,
    stats_ptr,
    logits_direction_ptr,
    envelope_direction_ptr,
    softmax_direction_ptr,
    value_contraction_ptr,
    contraction_direction_ptr,
    logits_hvp_ptr,
    envelope_hvp_ptr,
    eps,
    heads: tl.constexpr,
    degree_count: tl.constexpr,
    block_heads: tl.constexpr,
):
    receiver = tl.program_id(0)
    head = tl.arange(0, block_heads)
    head_mask = head < heads
    stats_offset = (receiver * heads + head) * 2
    receiver_max = tl.load(
        stats_ptr + stats_offset, mask=head_mask, other=0.0,
    ).to(tl.float32)
    receiver_denominator = (
        tl.load(stats_ptr + stats_offset + 1, mask=head_mask, other=0.0).to(tl.float32)
        + eps
    )
    normalization_direction = tl.load(
        softmax_direction_ptr + receiver * heads + head,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    edge_begin = tl.load(row_ptr + receiver)
    edge_end = tl.load(row_ptr + receiver + 1)

    correction = tl.zeros((block_heads,), tl.float32)
    correction_direction = tl.zeros((block_heads,), tl.float32)
    for edge in tl.range(edge_begin, edge_end):
        logit = tl.load(
            logits_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        logit_direction = tl.load(
            logits_direction_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
        normalized_exp = tl.exp(logit - receiver_max) / receiver_denominator
        probability = envelope * normalized_exp
        beta = envelope * probability
        beta_direction = (
            beta * (logit_direction - normalization_direction)
            + 2.0 * probability * envelope_direction
        )
        value_contraction = tl.zeros((block_heads,), tl.float32)
        contraction_direction = tl.zeros((block_heads,), tl.float32)
        for degree in tl.static_range(0, degree_count):
            partial_offset = (edge * degree_count + degree) * heads + head
            value_contraction += tl.load(
                value_contraction_ptr + partial_offset,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
            contraction_direction += tl.load(
                contraction_direction_ptr + partial_offset,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
        correction += beta * value_contraction
        correction_direction += (
            beta_direction * value_contraction + beta * contraction_direction
        )

    for edge in tl.range(edge_begin, edge_end):
        logit = tl.load(
            logits_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        logit_direction = tl.load(
            logits_direction_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
        normalized_exp = tl.exp(logit - receiver_max) / receiver_denominator
        normalized_exp_direction = normalized_exp * (
            logit_direction - normalization_direction
        )
        probability = envelope * normalized_exp
        probability_direction = (
            probability * (logit_direction - normalization_direction)
            + normalized_exp * envelope_direction
        )
        beta = envelope * probability
        beta_direction = (
            beta * (logit_direction - normalization_direction)
            + 2.0 * probability * envelope_direction
        )
        value_contraction = tl.zeros((block_heads,), tl.float32)
        contraction_direction = tl.zeros((block_heads,), tl.float32)
        for degree in tl.static_range(0, degree_count):
            partial_offset = (edge * degree_count + degree) * heads + head
            value_contraction += tl.load(
                value_contraction_ptr + partial_offset,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
            contraction_direction += tl.load(
                contraction_direction_ptr + partial_offset,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            logits_hvp_ptr + edge * heads + head,
            beta_direction * value_contraction
            + beta * contraction_direction
            - probability_direction * correction
            - probability * correction_direction,
            mask=head_mask,
        )
        envelope_hvp = tl.sum(
            normalized_exp_direction * (2.0 * envelope * value_contraction - correction)
            + normalized_exp
            * (
                2.0 * envelope_direction * value_contraction
                + 2.0 * envelope * contraction_direction
                - correction_direction
            ),
            axis=0,
        )
        tl.store(envelope_hvp_ptr + edge, envelope_hvp)


@triton.jit
def double_backward_degree_group_reuse(
    grad_output_ptr,
    values_ptr,
    values_direction_ptr,
    wigner_inv_ptr,
    wigner_direction_ptr,
    values_hvp_ptr,
    wigner_hvp_ptr,
    edge,
    receiver,
    channel,
    channel_mask,
    beta,
    beta_direction,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    full_components: tl.constexpr,
    degree_major_rows: tl.constexpr,
    degree_major_compact_offsets: tl.constexpr,
    compact_elements: tl.constexpr,
    row_offset: tl.constexpr,
    row_count: tl.constexpr,
    full_start: tl.constexpr,
    full_width: tl.constexpr,
    block_full: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """Reuse one cotangent degree block across all HVP reduced rows."""

    local_full = tl.arange(0, block_full)
    full = full_start + local_full
    full_mask = local_full < full_width
    cotangent = tl.load(
        grad_output_ptr
        + receiver * full_components * channels
        + full[:, None] * channels
        + channel[None, :],
        mask=full_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    value_contraction = tl.zeros(channel.shape, tl.float32)
    contraction_direction = tl.zeros(channel.shape, tl.float32)
    for local_reduced in tl.static_range(0, row_count):
        coefficient_offset = (
            edge * compact_elements
            + list(degree_major_compact_offsets)[row_offset + local_reduced]
            + local_full
        )
        coefficient = tl.load(
            wigner_inv_ptr + coefficient_offset,
            mask=full_mask,
            other=0.0,
        ).to(tl.float32)
        coefficient_direction = tl.load(
            wigner_direction_ptr + coefficient_offset,
            mask=full_mask,
            other=0.0,
        ).to(tl.float32)
        backprojected = tl.sum(
            coefficient[:, None] * cotangent,
            axis=0,
        )
        backprojected_direction = tl.sum(
            coefficient_direction[:, None] * cotangent,
            axis=0,
        )
        value = load_attention_value(
            values_ptr, edge,
            tl.constexpr(list(degree_major_rows)[row_offset + local_reduced]),
            channel, channel_mask,
            channels, reduced_components, sector_layout,
        )
        value_direction = load_attention_value(
            values_direction_ptr, edge,
            tl.constexpr(list(degree_major_rows)[row_offset + local_reduced]),
            channel, channel_mask,
            channels, reduced_components, sector_layout,
        )
        value_contraction += backprojected * value
        contraction_direction += (
            backprojected_direction * value + backprojected * value_direction
        )
        store_attention_gradient(
            values_hvp_ptr, beta_direction * backprojected + beta * backprojected_direction, edge,
            tl.constexpr(list(degree_major_rows)[row_offset + local_reduced]),
            channel, channel_mask,
            channels, reduced_components, sector_layout,
        )
        weighted_direction = beta_direction * value + beta * value_direction
        wigner_hvp = tl.sum(
            cotangent * weighted_direction[None, :],
            axis=1,
        )
        tl.atomic_add(
            wigner_hvp_ptr + coefficient_offset,
            wigner_hvp,
            mask=full_mask,
            sem="relaxed",
        )
    return value_contraction, contraction_direction


@triton.jit
def edge_all_degree_double_backward_kernel(
    grad_output_ptr,
    values_ptr,
    logits_ptr,
    envelope_ptr,
    wigner_inv_ptr,
    receivers_ptr,
    stats_ptr,
    values_direction_ptr,
    logits_direction_ptr,
    envelope_direction_ptr,
    wigner_direction_ptr,
    softmax_direction_ptr,
    values_hvp_ptr,
    value_contraction_ptr,
    contraction_direction_ptr,
    wigner_hvp_ptr,
    eps,
    heads: tl.constexpr,
    value_channels: tl.constexpr,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    full_components: tl.constexpr,
    degree_major_rows: tl.constexpr,
    degree_major_compact_offsets: tl.constexpr,
    degree_count: tl.constexpr,
    degree_row_counts: tl.constexpr,
    degree_row_offsets: tl.constexpr,
    degree_component_blocks: tl.constexpr,
    compact_elements: tl.constexpr,
    block_channels: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """All degree blocks for half the channels of one edge HVP."""

    edge = tl.program_id(0)
    channel_tile = tl.program_id(1)
    padded_value_channels: tl.constexpr = triton.next_power_of_2(value_channels)
    if padded_value_channels == value_channels:
        channel = channel_tile * block_channels + tl.arange(0, block_channels)
        channel_mask = channel < channels
        head = channel // value_channels
    else:
        lane = tl.arange(0, block_channels)
        head = (
            channel_tile * (block_channels // padded_value_channels)
            + lane // padded_value_channels
        )
        channel = head * value_channels + lane % padded_value_channels
        channel_mask = (head < heads) & (lane % padded_value_channels < value_channels)
    receiver = tl.load(receivers_ptr + edge)
    stats_offset = (receiver * heads + head) * 2
    receiver_max = tl.load(
        stats_ptr + stats_offset,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    receiver_denominator = (
        tl.load(
            stats_ptr + stats_offset + 1,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        + eps
    )
    logit = tl.load(
        logits_ptr + edge * heads + head,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    logit_direction = tl.load(
        logits_direction_ptr + edge * heads + head,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    normalization_direction = tl.load(
        softmax_direction_ptr + receiver * heads + head,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    normalized_exp = tl.exp(logit - receiver_max) / receiver_denominator
    probability = envelope * normalized_exp
    beta = envelope * probability
    beta_direction = (
        beta * (logit_direction - normalization_direction)
        + 2.0 * probability * envelope_direction
    )
    value_contraction = tl.zeros(channel.shape, tl.float32)
    contraction_direction = tl.zeros(channel.shape, tl.float32)
    for degree in tl.static_range(0, degree_count):
        next_value, next_direction = double_backward_degree_group_reuse(
            grad_output_ptr,
            values_ptr,
            values_direction_ptr,
            wigner_inv_ptr,
            wigner_direction_ptr,
            values_hvp_ptr,
            wigner_hvp_ptr,
            edge,
            receiver,
            channel,
            channel_mask,
            beta,
            beta_direction,
            channels,
            reduced_components,
            full_components,
            degree_major_rows,
            degree_major_compact_offsets,
            compact_elements,
            tl.constexpr(list(degree_row_offsets)[degree]),
            tl.constexpr(list(degree_row_counts)[degree]),
            degree * degree,
            2 * degree + 1,
            tl.constexpr(list(degree_component_blocks)[degree]),
            sector_layout,
        )
        value_contraction += next_value
        contraction_direction += next_direction

    for local_head in tl.static_range(0, block_channels // padded_value_channels):
        output_head = (
            channel_tile * (block_channels // padded_value_channels) + local_head
        )
        head_mask = head == output_head
        output_offset = edge * heads + output_head
        tl.store(
            value_contraction_ptr + output_offset,
            tl.sum(
                tl.where(head_mask, value_contraction, 0.0),
                axis=0,
            ),
            mask=(heads % (block_channels // padded_value_channels) == 0)
            | (output_head < heads),
        )
        tl.store(
            contraction_direction_ptr + output_offset,
            tl.sum(
                tl.where(head_mask, contraction_direction, 0.0),
                axis=0,
            ),
            mask=(heads % (block_channels // padded_value_channels) == 0)
            | (output_head < heads),
        )


def attention_double_backward_launch(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: ValueStorage,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    values_direction: ValueStorage,
    logits_direction: torch.Tensor,
    envelope_direction: torch.Tensor,
    wigner_direction: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> tuple[
    torch.Tensor,
    ValueStorage,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Differentiate the VJP along the four supplied input directions.

    The first result is the derivative with respect to the VJP's incoming
    cotangent, J @ direction. The other four results are Hessian-vector
    products of the scalar contraction with that cotangent held fixed.
    """
    channels, heads, value_channels = attention_dimensions(values, logits)
    allow_atomic_reduction("aggregate_attention double backward")
    envelope_flat, spec = validate_inputs(
        values,
        logits,
        envelope,
        wigner_inv,
        receivers,
        num_nodes,
        lmax,
        mmax,
    )
    validate_row_ptr(row_ptr, receivers, num_nodes)
    expected_output = (num_nodes, spec.full_components, channels)
    if tuple(grad_output.shape) != expected_output:
        raise ValueError(
            f"unexpected output cotangent shape: {tuple(grad_output.shape)}"
        )
    if tuple(forward_output.shape) != expected_output:
        raise ValueError(
            f"unexpected forward output shape: {tuple(forward_output.shape)}"
        )
    expected_directions = tuple(
        tuple(value.shape)
        for value in (
            *value_blocks(values),
            logits,
            envelope,
            wigner_inv,
        )
    )
    actual_directions = tuple(
        tuple(direction.shape)
        for direction in (
            *value_blocks(values_direction),
            logits_direction,
            envelope_direction,
            wigner_direction,
        )
    )
    if actual_directions != expected_directions:
        raise ValueError(
            f"unexpected structured-scatter directions: {actual_directions=}"
        )
    if tuple(stats.shape) != (num_nodes, heads, 2):
        raise ValueError(f"unexpected stats shape: {tuple(stats.shape)}")
    hvp_inputs = (
        grad_output,
        stats,
        *value_blocks(values_direction),
        logits_direction,
        envelope_direction,
        wigner_direction,
    )
    if not all(tensor.is_cuda for tensor in hvp_inputs):
        raise ValueError("structured-scatter HVP inputs must be CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in hvp_inputs):
        raise ValueError("structured-scatter HVP inputs must be contiguous")

    edges = value_blocks(values)[0].shape[0]
    envelope_direction_flat = envelope_direction[:, 0]
    grad_output_direction = torch.empty_like(grad_output)
    values_hvp = empty_values(values)
    logits_hvp = torch.empty_like(logits)
    envelope_hvp = torch.empty_like(envelope_flat)
    wigner_hvp = torch.zeros_like(wigner_inv)
    softmax_direction = torch.empty(
        (num_nodes, heads),
        device=value_blocks(values)[0].device,
        dtype=value_blocks(values)[0].dtype,
    )
    value_contraction = torch.empty(
        (edges, 1, heads),
        device=logits.device,
        dtype=logits.dtype,
    )
    contraction_direction = torch.empty_like(value_contraction)

    receiver_softmax_direction_kernel[(num_nodes,)](
        logits,
        envelope_flat,
        logits_direction,
        envelope_direction_flat,
        row_ptr,
        stats,
        softmax_direction,
        eps,
        heads=heads,
        block_heads=triton.next_power_of_2(heads),
        num_warps=1,
    )
    derivative_block = max(32, triton.next_power_of_2(value_channels))
    derivative_tiles = triton.cdiv(
        heads, derivative_block // triton.next_power_of_2(value_channels)
    )
    edge_all_degree_double_backward_kernel[(edges, derivative_tiles)](
        grad_output,
        values,
        logits,
        envelope_flat,
        wigner_inv,
        receivers,
        stats,
        values_direction,
        logits_direction,
        envelope_direction_flat,
        wigner_direction,
        softmax_direction,
        values_hvp,
        value_contraction,
        contraction_direction,
        wigner_hvp,
        eps,
        heads=heads,
        value_channels=value_channels,
        channels=channels,
        reduced_components=spec.reduced_components,
        full_components=spec.full_components,
        degree_major_rows=tuple(row for rows in spec.degree_rows for row in rows),
        degree_major_compact_offsets=tuple(
            spec.compact_row_offsets[row] for rows in spec.degree_rows for row in rows
        ),
        degree_count=spec.degree_count,
        degree_row_counts=spec.degree_row_counts,
        degree_row_offsets=spec.degree_row_offsets,
        degree_component_blocks=spec.degree_component_blocks,
        compact_elements=spec.compact_elements,
        block_channels=derivative_block,
        sector_layout=attention_row_layout(values, spec),
        num_warps=1,
        num_stages=1,
    )
    receiver_attention_double_backward_kernel[(num_nodes,)](
        logits,
        envelope_flat,
        row_ptr,
        stats,
        logits_direction,
        envelope_direction_flat,
        softmax_direction,
        value_contraction,
        contraction_direction,
        logits_hvp,
        envelope_hvp,
        eps,
        heads=heads,
        degree_count=1,
        block_heads=triton.next_power_of_2(heads),
        num_warps=1,
    )
    block_full = triton.next_power_of_2(spec.full_components)
    block_channels = 32 if block_full > 32 else 64
    receiver_sparse_forward_direction_kernel[
        (num_nodes, triton.cdiv(channels, block_channels))
    ](
        forward_output,
        values,
        logits,
        envelope_flat,
        wigner_inv,
        row_ptr,
        stats,
        values_direction,
        logits_direction,
        envelope_direction_flat,
        wigner_direction,
        grad_output_direction,
        eps,
        heads=heads,
        value_channels=value_channels,
        channels=channels,
        reduced_components=spec.reduced_components,
        full_components=spec.full_components,
        reduced_degrees=spec.reduced_degrees,
        compact_elements=spec.compact_elements,
        compact_row_offsets=spec.compact_row_offsets,
        block_full=block_full,
        block_channels=block_channels,
        sector_layout=attention_row_layout(values, spec),
        num_warps=4, num_stages=1,
    )
    return (
        grad_output_direction,
        values_hvp,
        logits_hvp,
        envelope_hvp[:, None],
        wigner_hvp,
    )


@triton.jit
def edge_degree_grad_output_hvp_kernel(
    radial_ptr, envelope_ptr, wigner_ptr,
    row_ptr,
    radial_direction_ptr, envelope_direction_ptr, wigner_direction_ptr,
    grad_output_gradient_ptr,
    rescale,
    components: tl.constexpr,
    degrees: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    compact_elements: tl.constexpr,
    m0_offsets: tl.constexpr,
    block_elements: tl.constexpr,
):
    receiver = tl.program_id(0)
    tile = tl.program_id(1)
    offset = tile * block_elements + tl.arange(0, block_elements)
    mask = offset < components * channels
    component = offset // channels
    channel = offset - component * channels
    degree = component_degree(component, lmax)
    accumulator = tl.zeros((block_elements,), dtype=tl.float32)
    edge_begin = tl.load(row_ptr + receiver)
    edge_end = tl.load(row_ptr + receiver + 1)
    for edge in tl.range(edge_begin, edge_end):
        radial_offset = edge * degrees * channels + degree * channels + channel
        wigner_offset = (
            edge * compact_elements
            + m0_row_offset(degree, m0_offsets, degrees)
            + component
            - degree * degree
        )
        radial = tl.load(radial_ptr + radial_offset, mask=mask, other=0.0).to(tl.float32)
        radial_direction = tl.load(
            radial_direction_ptr + radial_offset, mask=mask, other=0.0
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
        wigner = tl.load(wigner_ptr + wigner_offset, mask=mask, other=0.0).to(tl.float32)
        wigner_direction = tl.load(
            wigner_direction_ptr + wigner_offset, mask=mask, other=0.0
        ).to(tl.float32)
        accumulator += (
            envelope * wigner * radial_direction
            + envelope_direction * wigner * radial
            + envelope * wigner_direction * radial
        )
    tl.store(
        grad_output_gradient_ptr + receiver * components * channels + offset,
        accumulator / rescale,
        mask=mask,
    )


@triton.jit
def edge_degree_radial_hvp_kernel(
    grad_output_ptr,
    radial_ptr, envelope_ptr, wigner_ptr,
    receivers_ptr,
    radial_direction_ptr, envelope_direction_ptr, wigner_direction_ptr,
    radial_gradient_ptr, envelope_gradient_ptr,
    rescale,
    components: tl.constexpr,
    degrees: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    compact_elements: tl.constexpr,
    m0_offsets: tl.constexpr,
    block_elements: tl.constexpr,
):
    edge = tl.program_id(0)
    offset = tl.arange(0, block_elements)
    mask = offset < degrees * channels
    degree = offset // channels
    channel = offset - degree * channels
    receiver = tl.load(receivers_ptr + edge)
    projection = tl.zeros((block_elements,), dtype=tl.float32)
    direction_projection = tl.zeros((block_elements,), dtype=tl.float32)
    component_start = degree * degree
    component_width = 2 * degree + 1
    for local_component in tl.static_range(0, 2 * lmax + 1):
        active = mask & (local_component < component_width)
        component = component_start + local_component
        grad_output = tl.load(
            grad_output_ptr + receiver * components * channels + component * channels + channel,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        wigner_offset = (
            edge * compact_elements
            + m0_row_offset(degree, m0_offsets, degrees)
            + component
            - degree * degree
        )
        wigner = tl.load(wigner_ptr + wigner_offset, mask=active, other=0.0).to(tl.float32)
        wigner_direction = tl.load(
            wigner_direction_ptr + wigner_offset,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        projection += wigner * grad_output
        direction_projection += wigner_direction * grad_output
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    radial = tl.load(
        radial_ptr + edge * degrees * channels + offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    radial_direction = tl.load(
        radial_direction_ptr + edge * degrees * channels + offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        radial_gradient_ptr + edge * degrees * channels + offset,
        (envelope_direction * projection + envelope * direction_projection) / rescale,
        mask=mask,
    )
    envelope_gradient = (
        tl.sum(
            tl.where(
                mask,
                radial_direction * projection + radial * direction_projection,
                0.0,
            ),
            axis=0,
        )
        / rescale
    )
    tl.store(envelope_gradient_ptr + edge, envelope_gradient)


@triton.jit
def edge_degree_wigner_hvp_kernel(
    grad_output_ptr,
    radial_ptr,
    envelope_ptr,
    receivers_ptr,
    radial_direction_ptr,
    envelope_direction_ptr,
    wigner_gradient_ptr,
    rescale,
    components: tl.constexpr,
    degrees: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    compact_elements: tl.constexpr,
    m0_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    components_per_program: tl.constexpr,
):
    edge = tl.program_id(0)
    component = (
        tl.program_id(1) * components_per_program
        + tl.arange(0, components_per_program)[:, None]
    )
    channel = tl.arange(0, block_channels)[None, :]
    mask = (component < components) & (channel < channels)
    receiver = tl.load(receivers_ptr + edge)
    degree = component_degree(component, lmax)
    grad_output = tl.load(
        grad_output_ptr + receiver * components * channels + component * channels + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    radial_offset = edge * degrees * channels + degree * channels + channel
    radial = tl.load(radial_ptr + radial_offset, mask=mask, other=0.0).to(tl.float32)
    radial_direction = tl.load(
        radial_direction_ptr + radial_offset, mask=mask, other=0.0
    ).to(tl.float32)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    gradient = (
        tl.sum(
            grad_output * (envelope * radial_direction + envelope_direction * radial),
            axis=1,
        )[:, None]
        / rescale
    )
    tl.store(
        wigner_gradient_ptr
        + edge * compact_elements
        + m0_row_offset(degree, m0_offsets, degrees)
        + component
        - degree * degree,
        gradient,
        mask=component < components,
    )


def scalar_embedding_double_backward(
    grad_output,
    radial, envelope, wigner,
    receivers, row_ptr,
    radial_direction, envelope_direction, wigner_direction,
    rescale,
    spec,
):
    nodes = row_ptr.shape[0] - 1
    edges = radial.shape[0]
    grad_output_gradient = torch.empty_like(grad_output)
    radial_gradient = torch.empty_like(radial)
    envelope_gradient = torch.empty_like(envelope)
    wigner_gradient = torch.zeros_like(wigner)
    output_elements = spec.full_components * spec.channels
    output_block = min(4096, triton.next_power_of_2(output_elements))
    output_grid = (nodes, triton.cdiv(output_elements, output_block))
    edge_degree_grad_output_hvp_kernel[output_grid](
        radial,
        envelope,
        wigner,
        row_ptr,
        radial_direction, envelope_direction, wigner_direction,
        grad_output_gradient,
        rescale,
        **kernel_metadata(spec),
        block_elements=output_block,
        num_warps=8 if output_block >= 2048 else 4,
    )
    radial_elements = spec.degree_count * spec.channels
    radial_block = triton.next_power_of_2(radial_elements)
    edge_degree_radial_hvp_kernel[(edges,)](
        grad_output, radial, envelope, wigner,
        receivers,
        radial_direction, envelope_direction, wigner_direction,
        radial_gradient,
        envelope_gradient,
        rescale,
        **kernel_metadata(spec),
        block_elements=radial_block,
        num_warps=8 if radial_block >= 1024 else 4,
    )
    channel_block = triton.next_power_of_2(spec.channels)
    components_per_program = 8
    edge_degree_wigner_hvp_kernel[
        (edges, triton.cdiv(spec.full_components, components_per_program))
    ](
        grad_output,
        radial,
        envelope,
        receivers,
        radial_direction,
        envelope_direction,
        wigner_gradient,
        rescale,
        **kernel_metadata(spec),
        block_channels=channel_block,
        components_per_program=components_per_program,
        num_warps=4,
    )
    return (
        grad_output_gradient,
        radial_gradient, envelope_gradient, wigner_gradient,
    )


@triton.jit
def hvp_kernel(
    edge_direction_ptr,
    weight_direction_ptr,
    atomic_ptr,
    output_ptr,
    nodes,
    elements,
    row_stride,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    node = offset // row_stride
    within = offset - node * row_stride
    channel = within % channels
    atomic = tl.load(atomic_ptr + node, mask=mask, other=0)
    value = tl.load(edge_direction_ptr + offset, mask=mask, other=0.0)
    value += tl.load(
        weight_direction_ptr + atomic * channels + channel,
        mask=mask & (within < channels),
        other=0.0,
    )
    tl.store(output_ptr + offset, value, mask=mask)


def initial_atom_hvp(node_direction, weight_direction, atomic_numbers):
    node_direction = node_direction.contiguous()
    weight_direction = weight_direction.contiguous()
    output = torch.empty_like(node_direction)
    elements = node_direction.numel()
    row_stride = node_direction.shape[1] * node_direction.shape[2]
    block = 256
    hvp_kernel[(triton.cdiv(elements, block),)](
        node_direction,
        weight_direction,
        atomic_numbers,
        output,
        node_direction.shape[0],
        elements,
        row_stride,
        channels=node_direction.shape[2],
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def scalar_hvp_kernel(
    node_direction_ptr,
    scalar_direction_ptr,
    output_ptr,
    elements,
    row_stride,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    node = offset // row_stride
    within = offset - node * row_stride
    channel = within % channels
    value = tl.load(node_direction_ptr + offset, mask=mask, other=0.0)
    value += tl.load(
        scalar_direction_ptr + node * channels + channel,
        mask=mask & (within < channels),
        other=0.0,
    )
    tl.store(output_ptr + offset, value, mask=mask)


@triton.jit
def sector_message_double_backward_kernel(
    grad_output_ptr, message_ptr,
    envelope_ptr, wigner_ptr,
    message_direction_ptr, envelope_direction_ptr, wigner_direction_ptr,
    receivers_ptr,
    grad_output_gradient_ptr,
    message_gradient_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
    message_direction_stride_edge,
    message_direction_stride_row,
    message_direction_stride_channel,
    message_gradient_stride_edge,
    message_gradient_stride_row,
    message_gradient_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    wigner_direction_stride_edge,
    wigner_direction_stride_row,
    wigner_direction_stride_column,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    rows: tl.constexpr,
    wigner_layout: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    accumulate_nodes: tl.constexpr = True,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    receiver = tl.load(receivers_ptr + edge)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
    width = lmax - order + 1
    for row in tl.static_range(0, rows):
        if order == 0:
            degree = row
            dense_row = degree * degree + degree
        elif row < width:
            degree = order + row
            dense_row = degree * degree + degree + order
        else:
            degree = order + row - width
            dense_row = degree * degree + degree - order
        message = load_sector_value(
            message_ptr, edge, row, channel,
            message_stride_edge, message_stride_row, message_stride_channel,
            channels, width, order, raw_complex, channel_mask,
        )
        message_direction = load_sector_value(
            message_direction_ptr,
            edge,
            row,
            channel,
            message_direction_stride_edge,
            message_direction_stride_row,
            message_direction_stride_channel,
            channels,
            width,
            order,
            raw_complex,
            channel_mask,
        )
        message_gradient = tl.zeros(channel.shape, dtype=tl.float32)
        for local_full in tl.static_range(0, 2 * lmax + 1):
            full_mask = local_full < 2 * degree + 1
            full = degree * degree + local_full
            cotangent = tl.load(
                grad_output_ptr + receiver * full_components * channels + full * channels + channel,
                mask=channel_mask & full_mask,
                other=0.0,
            ).to(tl.float32)
            address = wigner_offset(
                edge, dense_row, full,
                wigner_stride_edge, wigner_stride_row, wigner_stride_column,
                wigner_layout,
            )
            coefficient = tl.load(wigner_ptr + address, mask=full_mask, other=0.0).to(tl.float32)
            coefficient_direction = tl.load(
                wigner_direction_ptr
                + wigner_offset(
                    edge,
                    dense_row,
                    full,
                    wigner_direction_stride_edge,
                    wigner_direction_stride_row,
                    wigner_direction_stride_column,
                    wigner_layout,
                ),
                mask=full_mask,
                other=0.0,
            ).to(tl.float32)
            message_gradient += (
                envelope_direction * coefficient * cotangent
                + envelope * coefficient_direction * cotangent
            )
            if accumulate_nodes:
                tl.atomic_add(
                    grad_output_gradient_ptr
                    + receiver * full_components * channels
                    + full * channels
                    + channel,
                    envelope_direction * coefficient * message
                    + envelope * coefficient_direction * message
                    + envelope * coefficient * message_direction,
                    mask=channel_mask & full_mask,
                )
        store_sector_gradient(
            message_gradient_ptr,
            message_gradient,
            edge,
            row,
            channel,
            message_gradient_stride_edge,
            message_gradient_stride_row,
            message_gradient_stride_channel,
            channels,
            width,
            order,
            raw_complex,
            channel_mask,
        )


@triton.jit
def receiver_node_double_backward_kernel(
    messages,
    directions,
    envelope_ptr,
    envelope_direction_ptr,
    wigner_ptr,
    wigner_direction_ptr,
    row_ptr,
    output_ptr,
    message_strides: tl.constexpr,
    direction_strides: tl.constexpr,
    wigner_strides: tl.constexpr,
    wigner_direction_strides: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr,
    wigner_layout: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
):
    receiver = tl.program_id(0)
    full = tl.program_id(1)
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    center = degree * degree + degree
    channel = tl.arange(0, block_channels)
    mask = channel < channels
    start = tl.load(row_ptr + receiver)
    end = tl.load(row_ptr + receiver + 1)
    total = tl.zeros((block_channels,), tl.float32)
    for edge in range(start, end):
        envelope = tl.load(envelope_ptr + edge)
        envelope_direction = tl.load(envelope_direction_ptr + edge)
        for order in tl.static_range(mmax + 1):
            for sign in tl.static_range(1 if order == 0 else 2):
                width = lmax - order + 1
                row = degree - order + sign * width
                dense_row = center + (order if sign == 0 else -order)
                valid = mask & (degree >= order)
                message = load_sector_value(
                    messages[order],
                    edge,
                    row,
                    channel,
                    list(message_strides)[order][0],
                    list(message_strides)[order][1],
                    list(message_strides)[order][2],
                    channels,
                    width,
                    order,
                    raw_complex,
                    valid,
                )
                direction = load_sector_value(
                    directions[order],
                    edge,
                    row,
                    channel,
                    list(direction_strides)[order][0],
                    list(direction_strides)[order][1],
                    list(direction_strides)[order][2],
                    channels,
                    width,
                    order,
                    raw_complex,
                    valid,
                )
                address = wigner_offset(
                    edge, dense_row, full,
                    list(wigner_strides)[0], list(wigner_strides)[1], list(wigner_strides)[2],
                    wigner_layout,
                )
                direction_address = wigner_offset(
                    edge,
                    dense_row,
                    full,
                    list(wigner_direction_strides)[0],
                    list(wigner_direction_strides)[1],
                    list(wigner_direction_strides)[2],
                    wigner_layout,
                )
                coefficient = tl.load(
                    wigner_ptr + address, mask=degree >= order, other=0.0
                )
                coefficient_direction = tl.load(
                    wigner_direction_ptr + direction_address,
                    mask=degree >= order,
                    other=0.0,
                )
                total += (
                    envelope_direction * coefficient * message
                    + envelope * coefficient_direction * message
                    + envelope * coefficient * direction
                )
    tl.store(
        output_ptr + (receiver * (lmax + 1) ** 2 + full) * channels + channel,
        total,
        mask=mask,
    )


@triton.jit
def receiver_degree_double_backward_kernel(
    messages, directions, e, ev, w, wv, row_ptr, output,
    message_strides: tl.constexpr, direction_strides: tl.constexpr,
    wigner_strides: tl.constexpr, wigner_direction_strides: tl.constexpr,
    channels: tl.constexpr, lmax: tl.constexpr, mmax: tl.constexpr,
    raw_complex: tl.constexpr,
    degree: tl.constexpr, tile: tl.constexpr,
):
    receiver = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, tile)
    width: tl.constexpr = 2 * degree + 1
    live = column < width
    channel = tl.arange(0, channels)
    degree_offset: tl.constexpr = degree if degree < 2 else 10 + 5 * (degree * degree - 4)
    start = tl.load(row_ptr + receiver)
    stop = tl.load(row_ptr + receiver + 1)
    # Receiver messages and envelope values are shared across the degree.
    total = tl.full((tile, channels), 0., tl.float32)
    for edge in range(start, stop):
        edge = edge.to(tl.int64)
        scale = tl.load(e + edge)
        vscale = tl.load(ev + edge)
        for m in tl.static_range(min(degree, mmax) + 1):
            for sign in tl.static_range(1 if m == 0 else 2):
                sector_width = lmax - m + 1
                row = degree - m + sign * sector_width
                message = load_sector_value(
                    messages[m], edge, row, channel,
                    message_strides[m][0], message_strides[m][1], message_strides[m][2],
                    channels, sector_width, m, raw_complex, channel < channels,
                )
                tangent = load_sector_value(
                    directions[m], edge, row, channel,
                    direction_strides[m][0], direction_strides[m][1], direction_strides[m][2],
                    channels, sector_width, m, raw_complex, channel < channels,
                )
                retained = 0 if m == 0 else 2 * m - 1 + sign
                idx = degree_offset + retained * width + column
                coeff = tl.load(w + edge * wigner_strides[0] + idx * wigner_strides[1], live, 0.)
                vcoeff = tl.load(wv + edge * wigner_direction_strides[0] + idx * wigner_direction_strides[1],
                                 live, 0.)
                total += ((vscale * coeff + scale * vcoeff)[:, None] * message[None, :]
                          + (scale * coeff)[:, None] * tangent[None, :])
    dst = ((receiver * (lmax + 1) ** 2 + degree * degree + column[:, None]) * channels
           + channel[None, :])
    tl.store(output + dst, total, live[:, None])



@triton.jit
def sector_parameter_double_backward_kernel(
    grad_output_ptr, message_ptr,
    envelope_ptr, wigner_ptr,
    message_direction_ptr, envelope_direction_ptr, wigner_direction_ptr,
    receivers_ptr,
    envelope_gradient_ptr, wigner_gradient_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
    message_direction_stride_edge,
    message_direction_stride_row,
    message_direction_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    wigner_direction_stride_edge,
    wigner_direction_stride_row,
    wigner_direction_stride_column,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    rows: tl.constexpr,
    wigner_layout: tl.constexpr,
    raw_complex: tl.constexpr,
    full_block: tl.constexpr,
    channel_block: tl.constexpr,
    write_envelope: tl.constexpr = True,
    write_wigner: tl.constexpr = True,
):
    edge = tl.program_id(0)
    row = tl.program_id(1)
    width = lmax - order + 1
    if order == 0:
        degree = row
        dense_row = degree * degree + degree
    else:
        positive = row < width
        degree = order + tl.where(positive, row, row - width)
        dense_row = degree * degree + degree + tl.where(positive, order, -order)
    local_full_1d = tl.arange(0, full_block)
    local_full = local_full_1d[:, None]
    full_mask_1d = local_full_1d < 2 * degree + 1
    full_mask = full_mask_1d[:, None]
    channel = tl.arange(0, channel_block)[None, :]
    channel_mask = channel < channels
    value_mask = full_mask & channel_mask
    receiver = tl.load(receivers_ptr + edge)
    full = degree * degree + local_full
    message = load_sector_value(
        message_ptr, edge, row, channel,
        message_stride_edge, message_stride_row, message_stride_channel,
        channels, width, order, raw_complex, channel_mask,
    )
    message_direction = load_sector_value(
        message_direction_ptr,
        edge,
        row,
        channel,
        message_direction_stride_edge,
        message_direction_stride_row,
        message_direction_stride_channel,
        channels,
        width,
        order,
        raw_complex,
        channel_mask,
    )
    cotangent = tl.load(
        grad_output_ptr + receiver * full_components * channels + full * channels + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    channel_dot = tl.sum(cotangent * message, axis=1)
    channel_dot_direction = tl.sum(cotangent * message_direction, axis=1)
    wigner_address = wigner_offset(
        edge, dense_row, degree * degree + local_full_1d,
        wigner_stride_edge, wigner_stride_row, wigner_stride_column,
        wigner_layout,
    )
    if write_wigner:
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        envelope_direction = tl.load(envelope_direction_ptr + edge).to(tl.float32)
        tl.store(
            wigner_gradient_ptr + wigner_address,
            envelope_direction * channel_dot + envelope * channel_dot_direction,
            mask=full_mask_1d,
        )
    if write_envelope:
        coefficient = tl.load(
            wigner_ptr + wigner_address,
            mask=full_mask_1d,
            other=0.0,
        ).to(tl.float32)
        coefficient_direction = tl.load(
            wigner_direction_ptr
            + wigner_offset(
                edge,
                dense_row,
                degree * degree + local_full_1d,
                wigner_direction_stride_edge,
                wigner_direction_stride_row,
                wigner_direction_stride_column,
                wigner_layout,
            ),
            mask=full_mask_1d,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            envelope_gradient_ptr + edge,
            tl.sum(
                coefficient_direction * channel_dot + coefficient * channel_dot_direction,
                axis=0,
            ),
        )


@triton.jit
def load_degree_values(
    messages, strides, edge, channel, mask,
    degree: tl.constexpr, channels: tl.constexpr, raw_complex: tl.constexpr,
):
    values = (load_sector_value(
        messages[0], edge, degree, channel,
        strides[0][0], strides[0][1], strides[0][2],
        channels, 3, 0, False, mask,
    ),)
    for order in tl.static_range(1, degree + 1):
        width = tl.constexpr(3 - order)
        positive = load_sector_value(
            messages[order], edge, degree - order, channel,
            strides[order][0], strides[order][1], strides[order][2],
            channels, width, order, raw_complex, mask,
        )
        negative = load_sector_value(
            messages[order], edge, width + degree - order, channel,
            strides[order][0], strides[order][1], strides[order][2],
            channels, width, order, raw_complex, mask,
        )
        values += (positive, negative)
    return values


@triton.jit
def receiver_degree_direction(
    messages, directions, message_strides, direction_strides,
    envelope_ptr, wigner_ptr, wigner_direction_ptr, row_ptr, output_ptr,
    receiver, degree: tl.constexpr, raw_complex: tl.constexpr,
):
    channels: tl.constexpr = 128
    edge_tile: tl.constexpr = 8
    channel_tile: tl.constexpr = 32
    width: tl.constexpr = 2 * degree + 1
    block_start: tl.constexpr = degree * (4 * degree * degree - 1) // 3
    start = tl.load(row_ptr + receiver)
    end = tl.load(row_ptr + receiver + 1)
    channel = tl.program_id(2) * channel_tile + tl.arange(0, channel_tile)
    edge_offsets = tl.arange(0, edge_tile)[:, None]
    accumulators = ()
    for col in tl.static_range(width):
        accumulators += (tl.zeros((edge_tile, channel_tile), tl.float32),)

    for block in range(start, end, edge_tile):
        edge = block + edge_offsets
        active = edge < end
        envelope = tl.load(envelope_ptr + edge, mask=active, other=0.0)
        values = load_degree_values(
            messages, message_strides, edge, channel[None, :], active,
            degree, channels, raw_complex,
        )
        value_directions = load_degree_values(
            directions, direction_strides, edge, channel[None, :], active,
            degree, channels, raw_complex,
        )
        next_accumulators = ()
        for col in tl.static_range(width):
            contribution = tl.zeros((edge_tile, channel_tile), tl.float32)
            for row in tl.static_range(width):
                offset = edge * 35 + block_start + row * width + col
                coefficient = tl.load(wigner_ptr + offset, mask=active, other=0.0)
                direction = tl.load(wigner_direction_ptr + offset, mask=active, other=0.0)
                contribution += coefficient * value_directions[row]
                contribution += direction * values[row]
            next_accumulators += (accumulators[col] + envelope * contribution,)
        accumulators = next_accumulators

    for col in tl.static_range(width):
        target = (receiver * 9 + degree * degree + col) * channels + channel
        tl.store(output_ptr + target, tl.sum(accumulators[col], axis=0))


@triton.jit
def compact_receiver_direction_kernel(
    messages, directions, message_strides, direction_strides,
    envelope, wigner, wigner_direction, row_ptr, output,
    raw_complex: tl.constexpr,
):
    receiver = tl.program_id(0)
    for degree in tl.static_range(3):
        if tl.program_id(1) == degree:
            receiver_degree_direction(
                messages, directions, message_strides, direction_strides,
                envelope, wigner, wigner_direction, row_ptr, output,
                receiver, degree, raw_complex,
            )


def compact_receiver_double_backward(
    grad_output, messages, envelope, wigner, receivers, row_ptr,
    message_directions, wigner_direction, *, raw_complex,
):
    # The edge gradients are the ordinary VJP evaluated on the two directions.
    dwigner, *dmessages = compact_tail_backward(
        grad_output, message_directions, envelope, wigner_direction, receivers,
        lmax=2, mmax=2, raw_complex=raw_complex, wigner_layout=6,
    )
    dgrad_output = torch.empty_like(grad_output)
    compact_receiver_direction_kernel[(grad_output.shape[0], 3, 4)](
        messages, message_directions,
        tuple(message.stride() for message in messages),
        tuple(direction.stride() for direction in message_directions),
        envelope, wigner, wigner_direction, row_ptr, dgrad_output,
        raw_complex, num_warps=4, num_stages=1,
    )
    return dgrad_output, None, dwigner, *dmessages


def sum_double_backward(
    grad_output,
    messages, envelope, wigner,
    receivers, nodes,
    envelope_direction, wigner_direction, message_directions,
    *,
    lmax, mmax,
    wigner_layout,
    raw_complex=False,
    row_ptr=None,
    output_mask=(True, True, True, True),
):
    allow_atomic_reduction("dataflow/edge_to_node/sum_aggregation/derivatives.double_backward")
    spec, edges = sum_forward_validate(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout, raw_complex=raw_complex,
    )
    # The mask follows the order of the returned gradients.
    want_nodes, want_envelope, want_wigner, want_messages = output_mask
    accumulate_nodes = want_nodes and row_ptr is None
    grad_output_gradient = None
    if want_nodes:
        allocate_nodes = messages[0].new_zeros if accumulate_nodes else messages[0].new_empty
        grad_output_gradient = allocate_nodes((nodes, spec.full_components, spec.channels))
    message_gradients = [
        torch.empty_like(message) if want_messages or accumulate_nodes else None for message in messages
    ]
    envelope_gradient = torch.zeros_like(envelope) if want_envelope else None
    wigner_gradient = torch.zeros_like(wigner) if want_wigner else None
    for order, (message, message_direction, message_gradient) in enumerate(
        zip(messages, message_directions, message_gradients, strict=True)
    ):
        rows = logical_rows(spec, order)
        if message_gradient is not None:
            sector_message_double_backward_kernel[(edges,)](
                grad_output, message,
                envelope, wigner,
                message_direction, envelope_direction, wigner_direction,
                receivers,
                grad_output if grad_output_gradient is None else grad_output_gradient,
                message_gradient,
                *message.stride(),
                *message_direction.stride(),
                *message_gradient.stride(), *wigner.stride(),
                *wigner_direction.stride(),
                channels=spec.channels,
                full_components=spec.full_components,
                lmax=spec.lmax,
                order=order, rows=rows,
                wigner_layout=wigner_layout, raw_complex=raw_complex,
                accumulate_nodes=accumulate_nodes,
                block_channels=1 << (spec.channels - 1).bit_length(),
                num_warps=4,
            )
        if want_envelope or want_wigner:
            sector_parameter_double_backward_kernel[(edges, rows)](
                grad_output, message,
                envelope, wigner,
                message_direction, envelope_direction, wigner_direction,
                receivers,
                envelope if envelope_gradient is None else envelope_gradient,
                wigner if wigner_gradient is None else wigner_gradient,
                *message.stride(),
                *message_direction.stride(),
                *wigner.stride(),
                *wigner_direction.stride(),
                channels=spec.channels,
                full_components=spec.full_components,
                lmax=spec.lmax,
                order=order, rows=rows,
                wigner_layout=wigner_layout, raw_complex=raw_complex,
                full_block=1 << (2 * spec.lmax).bit_length(),
                channel_block=1 << (spec.channels - 1).bit_length(),
                write_envelope=want_envelope, write_wigner=want_wigner,
                num_warps=4,
            )
    if want_nodes and row_ptr is not None:
        node_inputs = (
            tuple(messages), tuple(message_directions), envelope, envelope_direction,
            wigner, wigner_direction, row_ptr, grad_output_gradient,
        )
        node_layout = dict(
            message_strides=tuple(x.stride() for x in messages),
            direction_strides=tuple(x.stride() for x in message_directions),
            wigner_strides=wigner.stride(), wigner_direction_strides=wigner_direction.stride(),
            channels=spec.channels, lmax=lmax, mmax=mmax, raw_complex=raw_complex,
        )
        if (
            use_degree_sum_node(edges, nodes, lmax, mmax, spec.channels, wigner_layout)
            and torch.version.hip is None and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(grad_output.device) == (9, 0)
        ):
            for degree in range(lmax + 1):
                tile = triton.next_power_of_2(2 * degree + 1)
                receiver_degree_double_backward_kernel[(nodes,)](
                    *node_inputs, **node_layout, degree=degree, tile=tile,
                    num_warps=2 if tile == 1 else 4,
                )
        else:
            receiver_node_double_backward_kernel[(nodes, spec.full_components)](
                *node_inputs, **node_layout, wigner_layout=wigner_layout,
                block_channels=triton.next_power_of_2(spec.channels), num_warps=1,
            )
    return (grad_output_gradient, envelope_gradient, wigner_gradient,
            *(message_gradients if want_messages else (None,) * len(messages)))
