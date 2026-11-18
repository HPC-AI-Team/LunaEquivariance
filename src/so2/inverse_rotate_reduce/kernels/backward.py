from __future__ import annotations

import ctypes
from functools import cache, lru_cache
import os
from pathlib import Path
import re
from shutil import which
import subprocess

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op
from torch.utils.cpp_extension import CUDA_HOME, get_cxx_compiler
from triton.language.extra.cuda import libdevice

from ....utils.jit import compiler_identity, load_cuda_library, load_extension
from ...utils.layout import get_retained_layout
from ...utils.sector_layout import load_sector_value, store_sector_gradient
from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import (
    ValueStorage,
    combine_preprojection_orders,
    attention_dimensions,
    component_degree,
    empty_values,
    kernel_metadata,
    load_attention_value,
    logical_rows,
    m0_row_offset,
    attention_row_layout,
    preprojection_layout,
    reduce_wigner_channel_tiles_kernel,
    sum_forward_validate,
    validate_inputs,
    validate_row_ptr,
    value_blocks,
    wigner_offset,
)



@triton.jit
def preprojection_adjoint_rows(
    x, rotation, receivers, terms, scale, dx, drotation,
    edge, node, valid_edge, channel,
    order: tl.constexpr, input_width: tl.constexpr, rotation_width: tl.constexpr,
    positive_offsets: tl.constexpr, negative_offsets: tl.constexpr, degrees: tl.constexpr,
    angular_widths: tl.constexpr, angular_indices: tl.constexpr,
    packed_widths: tl.constexpr, packed_offset: tl.constexpr,
    include_scale: tl.constexpr, edge_tile: tl.constexpr, channel_tile: tl.constexpr,
    share_seed: tl.constexpr,
):
    live = channel < input_width
    valid = valid_edge[:, None] & live[None, :]
    planes: tl.constexpr = 2 if order else 1
    addr = edge[:, None] * (planes * input_width) + channel[None, :]
    positive = tl.load(x + addr, valid, 0.)
    dx_pos = tl.zeros((edge_tile, channel_tile), tl.float32)
    if order:
        negative = tl.load(x + addr + input_width, valid, 0.)
        dx_neg = tl.zeros((edge_tile, channel_tile), tl.float32)
    if not share_seed:
        index = tl.load(receivers + edge, valid_edge, 0).to(tl.int32)
    slots: tl.constexpr = len(angular_indices)
    for i in tl.static_range(slots):
        if share_seed:
            offset = ((node * angular_widths[i] + angular_indices[i]) * packed_widths[i]
                      + packed_offset + channel)
            ga = tl.load(terms[degrees[i]] + offset, live, 0.)[None, :]
            if order:
                gb = tl.load(terms[degrees[i]] + offset + input_width, live, 0.)[None, :]
        else:
            offset = ((index[:, None] * angular_widths[i] + angular_indices[i]) * packed_widths[i]
                      + packed_offset + channel[None, :])
            ga = tl.load(terms[degrees[i]] + offset, valid, 0.)
            if order:
                gb = tl.load(terms[degrees[i]] + offset + input_width, valid, 0.)
        a = tl.load(rotation + edge * rotation_width + positive_offsets[i], valid_edge, 0.)
        if order:
            b = tl.load(rotation + edge * rotation_width + negative_offsets[i], valid_edge, 0.)
            dx_pos += ga * a[:, None] + gb * b[:, None]
            dx_neg += ga * b[:, None] - gb * a[:, None]
            da = tl.sum(ga * positive - gb * negative, axis=1)
            db = tl.sum(ga * negative + gb * positive, axis=1)
            tl.store(drotation + edge * rotation_width + negative_offsets[i], db, valid_edge)
        else:
            dx_pos += ga * a[:, None]
            da = tl.sum(ga * positive, axis=1)
            if include_scale:
                if share_seed:
                    da += tl.load(scale + node * slots + i)
                else:
                    da += tl.load(scale + index * slots + i, valid_edge, 0.)
        tl.store(drotation + edge * rotation_width + positive_offsets[i], da, valid_edge)
    tl.store(dx + addr, dx_pos, valid)
    if order:
        tl.store(dx + addr + input_width, dx_neg, valid)


@triton.jit
def preprojection_adjoint_kernel(
    x, rotation, receivers, terms, scale, dx, drotation, edges,
    order: tl.constexpr, input_width: tl.constexpr, rotation_width: tl.constexpr,
    positive_offsets: tl.constexpr, negative_offsets: tl.constexpr, degrees: tl.constexpr,
    angular_widths: tl.constexpr, angular_indices: tl.constexpr,
    packed_widths: tl.constexpr, packed_offset: tl.constexpr,
    include_scale: tl.constexpr, edge_tile: tl.constexpr, channel_tile: tl.constexpr,
):
    first = tl.program_id(0) * edge_tile
    node = tl.load(receivers + first).to(tl.int32)
    last = tl.load(receivers + tl.minimum(first + edge_tile - 1, edges - 1)).to(tl.int32)
    edge = first + tl.arange(0, edge_tile)
    channel = tl.arange(0, channel_tile)
    valid = edge < edges
    # In receiver-sorted CSR, equal endpoints imply one shared cotangent.
    if node == last:
        preprojection_adjoint_rows(
            x, rotation, receivers, terms, scale, dx, drotation, edge, node, valid, channel,
            order, input_width, rotation_width, positive_offsets, negative_offsets, degrees,
            angular_widths, angular_indices, packed_widths, packed_offset,
            include_scale, edge_tile, channel_tile, True,
        )
    else:
        preprojection_adjoint_rows(
            x, rotation, receivers, terms, scale, dx, drotation, edge, node, valid, channel,
            order, input_width, rotation_width, positive_offsets, negative_offsets, degrees,
            angular_widths, angular_indices, packed_widths, packed_offset,
            include_scale, edge_tile, channel_tile, False,
        )


@triton.jit
def preprojection_adjoint_order(
    x, rotation, receivers, terms, scale, dx, drotation, edges,
    order: tl.constexpr, spec: tl.constexpr, rotation_width: tl.constexpr,
    include_scale: tl.constexpr, edge_tile: tl.constexpr, channels: tl.constexpr,
):
    width: tl.constexpr = spec[0]
    positives: tl.constexpr = spec[1]
    negatives: tl.constexpr = spec[2]
    degrees: tl.constexpr = spec[3]
    angular_widths: tl.constexpr = spec[4]
    angular_indices: tl.constexpr = spec[5]
    packed_widths: tl.constexpr = spec[6]
    packed_offset: tl.constexpr = spec[7]
    preprojection_adjoint_kernel(
        x, rotation, receivers, terms, scale, dx, drotation, edges,
        order, width, rotation_width, positives, negatives, degrees, angular_widths,
        angular_indices, packed_widths, packed_offset, include_scale, edge_tile, channels,
    )


@triton.jit
def preprojection_adjoints_kernel(
    x0, x1, x2, rotation, receivers, terms, scale, dx0, dx1, dx2, drotation, edges,
    spec0: tl.constexpr, spec1: tl.constexpr, spec2: tl.constexpr, rotation_width: tl.constexpr,
    include_scale: tl.constexpr, first_order: tl.constexpr, edge_tile: tl.constexpr,
):
    order = tl.program_id(1) + first_order
    if first_order == 0 and order == 0:
        preprojection_adjoint_order(
            x0, rotation, receivers, terms, scale, dx0, drotation, edges,
            0, spec0, rotation_width, include_scale, edge_tile, 512,
        )
    elif order == 1:
        preprojection_adjoint_order(
            x1, rotation, receivers, terms, scale, dx1, drotation, edges,
            1, spec1, rotation_width, False, edge_tile, 256,
        )
    else:
        preprojection_adjoint_order(
            x2, rotation, receivers, terms, scale, dx2, drotation, edges,
            2, spec2, rotation_width, False, edge_tile, 128,
        )


def preprojection_adjoint(
    sectors, rotation, receivers, grad_terms, grad_scale, *, include_scale, layout=None,
):
    use_l2_schedule = layout is None
    layout = preprojection_layout() if use_l2_schedule else layout
    edges = rotation.shape[0]
    dsectors = tuple(torch.empty_like(value) for value in sectors)
    drotation = torch.empty_like(rotation)
    if edges == 0:
        return *dsectors, drotation
    nodes = grad_terms[0].shape[0]
    first_order = len(layout.orders)
    if use_l2_schedule and combine_preprojection_orders(rotation, nodes):
        first_order = 0
    elif use_l2_schedule and edges <= 4096 and torch.cuda.get_device_capability(rotation.device) == (8, 0):
        first_order = 1
    # Small Ampere graphs need more CTAs to fill the device.
    small_ampere = (
        first_order == 0 and 0 < nodes <= 16 and edges <= 512
        and torch.cuda.get_device_capability(rotation.device) == (8, 0)
    )
    edge_tile, warps = (1, 2) if small_ampere else (4, 4)
    if (
        not use_l2_schedule and layout is preprojection_layout(3, 2)
        and 16384 <= edges <= 32768
        and torch.version.hip is None
        and torch.cuda.get_device_capability(rotation.device) == (9, 0)
    ):
        edge_tile, warps = 2, 2
    blocks = triton.cdiv(edges, edge_tile)
    for order, spec in enumerate(layout.orders[:first_order]):
        preprojection_adjoint_kernel[(blocks,)](
            sectors[order], rotation, receivers, grad_terms, grad_scale, dsectors[order], drotation, edges,
            order, spec.input_width, layout.rotation_width,
            spec.positive_offsets, spec.negative_offsets, spec.degrees, spec.angular_widths,
            spec.angular_indices, spec.packed_widths, spec.packed_offset,
            include_scale, edge_tile, triton.next_power_of_2(spec.input_width), num_warps=warps,
        )
    if first_order < len(layout.orders):
        # Each order writes separate feature and rotation rows.
        preprojection_adjoints_kernel[(blocks, len(layout.orders) - first_order)](
            *sectors, rotation, receivers, grad_terms, grad_scale, *dsectors, drotation, edges,
            *(tuple(spec) for spec in layout.orders), layout.rotation_width,
            include_scale, first_order, edge_tile, num_warps=warps,
        )
    return *dsectors, drotation


@triton.jit
def receiver_attention_gradient_kernel(
    logits_ptr,
    envelope_ptr,
    row_ptr,
    stats_ptr,
    grad_beta_ptr,
    grad_logits_ptr,
    grad_envelope_ptr,
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
    edge_begin = tl.load(row_ptr + receiver)
    edge_end = tl.load(row_ptr + receiver + 1)

    correction = tl.zeros((block_heads,), tl.float32)
    for edge in tl.range(edge_begin, edge_end):
        logit = tl.load(
            logits_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        exp_over_denominator = tl.exp(logit - receiver_max) / receiver_denominator
        beta = envelope * envelope * exp_over_denominator
        grad_beta = tl.zeros((block_heads,), tl.float32)
        for degree in tl.static_range(0, degree_count):
            grad_beta += tl.load(
                grad_beta_ptr + (edge * degree_count + degree) * heads + head,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
        correction += beta * grad_beta

    for edge in tl.range(edge_begin, edge_end):
        logit = tl.load(
            logits_ptr + edge * heads + head,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        exp_over_denominator = tl.exp(logit - receiver_max) / receiver_denominator
        probability = envelope * exp_over_denominator
        beta = envelope * probability
        grad_beta = tl.zeros((block_heads,), tl.float32)
        for degree in tl.static_range(0, degree_count):
            grad_beta += tl.load(
                grad_beta_ptr + (edge * degree_count + degree) * heads + head,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            grad_logits_ptr + edge * heads + head,
            beta * grad_beta - probability * correction,
            mask=head_mask,
        )
        grad_envelope = tl.sum(
            exp_over_denominator * (2.0 * envelope * grad_beta - correction),
            axis=0,
        )
        tl.store(grad_envelope_ptr + edge, grad_envelope)


@triton.jit
def backward_degree_group_reuse(
    grad_output_ptr,
    values_ptr,
    wigner_inv_ptr,
    grad_values_ptr,
    grad_wigner_partial_ptr,
    edge,
    receiver,
    channel_tile,
    channel,
    channel_mask,
    beta,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    full_components: tl.constexpr,
    degree_major_rows: tl.constexpr,
    degree_major_compact_offsets: tl.constexpr,
    compact_elements: tl.constexpr,
    channel_tiles: tl.constexpr,
    row_offset: tl.constexpr,
    row_count: tl.constexpr,
    full_start: tl.constexpr,
    full_width: tl.constexpr,
    block_full: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """Reuse one cotangent degree block across all of its reduced rows."""

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
    contraction = tl.zeros(channel.shape, tl.float32)
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
        backprojected = tl.sum(
            coefficient[:, None] * cotangent,
            axis=0,
        )
        value = load_attention_value(
            values_ptr, edge,
            tl.constexpr(list(degree_major_rows)[row_offset + local_reduced]),
            channel, channel_mask,
            channels, reduced_components, sector_layout,
        )
        contraction += backprojected * value
        store_attention_gradient(
            grad_values_ptr, beta * backprojected, edge,
            tl.constexpr(list(degree_major_rows)[row_offset + local_reduced]),
            channel, channel_mask,
            channels, reduced_components, sector_layout,
        )
        wigner_gradient = tl.sum(
            cotangent * (beta * value)[None, :],
            axis=1,
        )
        tl.store(
            grad_wigner_partial_ptr
            + channel_tile * tl.num_programs(0) * compact_elements
            + coefficient_offset,
            wigner_gradient,
            mask=full_mask,
        )
    return contraction


@triton.jit
def edge_all_degree_backward_kernel(
    grad_output_ptr,
    values_ptr,
    logits_ptr,
    envelope_ptr,
    wigner_inv_ptr,
    receivers_ptr,
    stats_ptr,
    grad_values_ptr,
    grad_beta_ptr,
    grad_wigner_partial_ptr,
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
    channel_tiles: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """All degree blocks for half the channels of one edge."""

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
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    beta = envelope * envelope * tl.exp(logit - receiver_max) / receiver_denominator
    contraction = tl.zeros(channel.shape, tl.float32)
    for degree in tl.static_range(0, degree_count):
        contraction += backward_degree_group_reuse(
            grad_output_ptr,
            values_ptr,
            wigner_inv_ptr,
            grad_values_ptr,
            grad_wigner_partial_ptr,
            edge,
            receiver,
            channel_tile,
            channel,
            channel_mask,
            beta,
            channels,
            reduced_components,
            full_components,
            degree_major_rows,
            degree_major_compact_offsets,
            compact_elements,
            channel_tiles,
            tl.constexpr(list(degree_row_offsets)[degree]),
            tl.constexpr(list(degree_row_counts)[degree]),
            degree * degree,
            2 * degree + 1,
            tl.constexpr(list(degree_component_blocks)[degree]),
            sector_layout,
        )

    for local_head in tl.static_range(0, block_channels // padded_value_channels):
        output_head = (
            channel_tile * (block_channels // padded_value_channels) + local_head
        )
        head_contraction = tl.sum(
            tl.where(head == output_head, contraction, 0.0),
            axis=0,
        )
        tl.store(
            grad_beta_ptr + edge * heads + output_head,
            head_contraction,
            mask=(heads % (block_channels // padded_value_channels) == 0)
            | (output_head < heads),
        )


def attention_backward(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: ValueStorage,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> tuple[ValueStorage, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gradients for values, logits, envelope, and inverse rotations.

    Value gradients use the input's packed or raw-sector layout. Rotation
    gradients first reduce channel tiles, then merge those partial sums.
    """
    channels, heads, value_channels = attention_dimensions(values, logits)
    del forward_output
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
    if tuple(grad_output.shape) != (
        num_nodes,
        spec.full_components,
        channels,
    ):
        raise ValueError(
            f"unexpected output cotangent shape: {tuple(grad_output.shape)}"
        )
    # Public slicing already supplies dense storage; direct workspace gradients may not.
    grad_output = grad_output.contiguous()
    if tuple(stats.shape) != (num_nodes, heads, 2):
        raise ValueError(f"unexpected stats shape: {tuple(stats.shape)}")
    edges = value_blocks(values)[0].shape[0]
    grad_values = empty_values(values)
    grad_beta = torch.empty(
        (edges, 1, heads),
        device=logits.device,
        dtype=logits.dtype,
    )
    grad_logits = torch.empty_like(logits)
    grad_envelope = torch.empty_like(envelope_flat)
    block_channels = max(32, triton.next_power_of_2(value_channels))
    channel_tiles = triton.cdiv(
        heads, block_channels // triton.next_power_of_2(value_channels)
    )
    grad_wigner_partial = torch.empty(
        (edges, spec.compact_elements, channel_tiles),
        device=wigner_inv.device,
        dtype=wigner_inv.dtype,
    )
    grad_wigner = torch.empty_like(wigner_inv)
    edge_all_degree_backward_kernel[(edges, channel_tiles)](
        grad_output,
        values,
        logits,
        envelope_flat,
        wigner_inv,
        receivers,
        stats,
        grad_values,
        grad_beta,
        grad_wigner_partial,
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
        block_channels=block_channels,
        channel_tiles=channel_tiles,
        sector_layout=attention_row_layout(values, spec),
        num_warps=1,
        num_stages=1,
    )
    wigner_element_count = edges * spec.compact_elements
    reduce_wigner_channel_tiles_kernel[(triton.cdiv(wigner_element_count, 256),)](
        grad_wigner_partial,
        grad_wigner,
        wigner_element_count,
        channel_tiles=channel_tiles,
        block_elements=256,
        num_warps=4, num_stages=1,
    )
    receiver_attention_gradient_kernel[(num_nodes,)](
        logits,
        envelope_flat,
        row_ptr,
        stats,
        grad_beta,
        grad_logits,
        grad_envelope,
        eps,
        heads=heads,
        degree_count=1,
        block_heads=triton.next_power_of_2(heads),
        num_warps=1,
    )
    return grad_values, grad_logits, grad_envelope[:, None], grad_wigner


@triton.jit
def store_attention_gradient(
    pointers,
    value,
    edge,
    reduced: tl.constexpr,
    channel,
    mask,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    sector_layout: tl.constexpr,
):
    if len(sector_layout) > 0:
        order: tl.constexpr = list(sector_layout)[reduced][0]
        row: tl.constexpr = list(sector_layout)[reduced][1]
        width: tl.constexpr = list(sector_layout)[reduced][2]
        raw_complex: tl.constexpr = list(sector_layout)[reduced][3]
        rows: tl.constexpr = width if order == 0 else 2 * width
        stride_edge: tl.constexpr = rows * channels * (2 if raw_complex else 1)
        stride_row: tl.constexpr = 2 * width * channels if raw_complex else channels
        store_sector_gradient(
            pointers[order], value, edge, row, channel,
            stride_edge, stride_row, 1,
            channels, width, order, raw_complex, mask,
        )
    else:
        tl.store(
            pointers + (edge * reduced_components + reduced) * channels + channel,
            value,
            mask=mask,
        )


@triton.jit
def edge_degree_radial_backward_kernel(
    grad_output_ptr,
    radial_ptr, envelope_ptr, wigner_ptr,
    receivers_ptr,
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
        wigner = tl.load(
            wigner_ptr
            + edge * compact_elements
            + m0_row_offset(degree, m0_offsets, degrees)
            + component
            - degree * degree,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        projection += wigner * grad_output
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    radial = tl.load(
        radial_ptr + edge * degrees * channels + offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        radial_gradient_ptr + edge * degrees * channels + offset,
        envelope * projection / rescale,
        mask=mask,
    )
    envelope_gradient = (
        tl.sum(tl.where(mask, radial * projection, 0.0), axis=0) / rescale
    )
    tl.store(envelope_gradient_ptr + edge, envelope_gradient)


@triton.jit
def edge_degree_wigner_backward_kernel(
    grad_output_ptr,
    radial_ptr,
    envelope_ptr,
    receivers_ptr,
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
    radial = tl.load(
        radial_ptr + edge * degrees * channels + degree * channels + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    gradient = envelope * tl.sum(grad_output * radial, axis=1)[:, None] / rescale
    tl.store(
        wigner_gradient_ptr
        + edge * compact_elements
        + m0_row_offset(degree, m0_offsets, degrees)
        + component
        - degree * degree,
        gradient,
        mask=component < components,
    )


def scalar_embedding_backward(grad_output, radial, envelope, wigner, receivers, rescale, spec):
    edges = radial.shape[0]
    radial_gradient = torch.empty_like(radial)
    envelope_gradient = torch.empty_like(envelope)
    wigner_gradient = torch.zeros_like(wigner)
    radial_elements = spec.degree_count * spec.channels
    radial_block = triton.next_power_of_2(radial_elements)
    edge_degree_radial_backward_kernel[(edges,)](
        grad_output, radial, envelope, wigner,
        receivers,
        radial_gradient,
        envelope_gradient,
        rescale,
        **kernel_metadata(spec),
        block_elements=radial_block,
        num_warps=8 if radial_block >= 1024 else 4,
    )
    channel_block = triton.next_power_of_2(spec.channels)
    components_per_program = 8
    edge_degree_wigner_backward_kernel[
        (edges, triton.cdiv(spec.full_components, components_per_program))
    ](
        grad_output,
        radial,
        envelope,
        receivers,
        wigner_gradient,
        rescale,
        **kernel_metadata(spec),
        block_channels=channel_block,
        components_per_program=components_per_program,
        num_warps=4,
    )
    return radial_gradient, envelope_gradient, wigner_gradient


@triton.jit
def weight_vjp_kernel(
    grad_ptr,
    atomic_ptr,
    weight_grad_ptr,
    nodes,
    row_stride,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block)
    mask = (node < nodes) & (channel < channels)
    atomic = tl.load(atomic_ptr + node)
    gradient = tl.load(grad_ptr + node * row_stride + channel, mask=mask, other=0.0)
    tl.atomic_add(weight_grad_ptr + atomic * channels + channel, gradient, mask=mask)


def initial_atom_weight_vjp(gradient, atomic_numbers, rows):
    allow_atomic_reduction("scalar initial_atom_weight_vjp")
    gradient = gradient.contiguous()
    nodes, components, channels = gradient.shape
    output = torch.zeros((rows, channels), device=gradient.device, dtype=gradient.dtype)
    weight_vjp_kernel[(nodes,)](
        gradient,
        atomic_numbers,
        output,
        nodes,
        components * channels,
        channels=channels,
        block=triton.next_power_of_2(channels),
        num_warps=4,
    )
    return output


@triton.jit
def sector_message_backward_kernel(
    grad_output_ptr,
    envelope_ptr, wigner_ptr,
    receivers_ptr,
    message_gradient_ptr,
    message_gradient_stride_edge,
    message_gradient_stride_row,
    message_gradient_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    rows: tl.constexpr,
    wigner_layout: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    receiver = tl.load(receivers_ptr + edge)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
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
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
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
            coefficient = tl.load(
                wigner_ptr + address,
                mask=full_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += coefficient * cotangent
        store_sector_gradient(
            message_gradient_ptr,
            envelope * accumulator,
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
def sector_parameter_backward_kernel(
    grad_output_ptr, message_ptr,
    envelope_ptr, wigner_ptr,
    receivers_ptr,
    envelope_gradient_ptr, wigner_gradient_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    rows: tl.constexpr,
    wigner_layout: tl.constexpr,
    raw_complex: tl.constexpr,
    full_block: tl.constexpr,
    channel_block: tl.constexpr,
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
    cotangent = tl.load(
        grad_output_ptr + receiver * full_components * channels + full * channels + channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    channel_dot = tl.sum(cotangent * message, axis=1)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)
    wigner_address = wigner_offset(
        edge, dense_row, degree * degree + local_full_1d,
        wigner_stride_edge, wigner_stride_row, wigner_stride_column,
        wigner_layout,
    )
    tl.store(
        wigner_gradient_ptr + wigner_address,
        envelope * channel_dot,
        mask=full_mask_1d,
    )
    coefficient = tl.load(
        wigner_ptr + wigner_address,
        mask=full_mask_1d,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        envelope_gradient_ptr + edge,
        tl.sum(coefficient * channel_dot, axis=0),
    )


@triton.jit(do_not_specialize_on_alignment=("receivers_ptr",))
def compact_sector_tail_input_backward_kernel(
    grad_output_ptr,
    messages,
    envelope_ptr, wigner_ptr,
    receivers_ptr,
    message_gradients,
    wigner_gradient_ptr,
    message_strides,
    gradient_strides,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    compact_elements: tl.constexpr,
    lmax: tl.constexpr,
    reduced_degrees: tl.constexpr,
    reduced_orders: tl.constexpr,
    dense_rows: tl.constexpr,
    reduced_components: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    wigner_layout: tl.constexpr,
    reuse_cotangents: tl.constexpr = False,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    receiver = tl.load(receivers_ptr + edge)
    envelope = tl.load(envelope_ptr + edge).to(tl.float32)

    if reuse_cotangents:
        cotangents = ()
        for full_index in tl.static_range(0, full_components):
            cotangents += (
                tl.load(
                    grad_output_ptr
                    + receiver * full_components * channels
                    + full_index * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32),
            )

    rotation_mmax: tl.constexpr = wigner_layout - 4
    for slot in tl.static_range(0, reduced_components):
        degree = tl.constexpr(list(reduced_degrees)[slot])
        dense_row = tl.constexpr(list(dense_rows)[slot])
        degree_start = degree * degree
        width = 2 * degree + 1
        signed_order = tl.constexpr(dense_row - degree * degree - degree)
        order = tl.constexpr(list(reduced_orders)[slot])
        sector_width = lmax - order + 1
        sector_row = degree - order
        if dense_row < degree_start + degree:
            sector_row += sector_width

        message_ptr = messages[tl.constexpr(list(reduced_orders)[slot])]
        message_gradient_ptr = message_gradients[
            tl.constexpr(list(reduced_orders)[slot])
        ]
        stride_edge, stride_row, stride_channel = message_strides[
            tl.constexpr(list(reduced_orders)[slot])
        ]
        gradient_stride_edge, gradient_stride_row, gradient_stride_channel = (
            gradient_strides[tl.constexpr(list(reduced_orders)[slot])]
        )

        message = load_sector_value(
            message_ptr, edge, sector_row, channel,
            stride_edge, stride_row, stride_channel,
            channels, sector_width, order, raw_complex, channel_mask,
        )
        message_gradient = tl.zeros(channel.shape, dtype=tl.float32)
        for local_output in tl.static_range(0, 2 * list(reduced_degrees)[slot] + 1):
            full = degree_start + local_output
            if reuse_cotangents:
                cotangent = cotangents[
                    tl.constexpr(list(reduced_degrees)[slot]) ** 2 + local_output
                ]
            else:
                cotangent = tl.load(
                    grad_output_ptr
                    + receiver * full_components * channels
                    + full * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            complete_degrees = tl.minimum(degree, rotation_mmax + 1)
            retained_start = (
                complete_degrees * (4 * complete_degrees * complete_degrees - 1) // 3
            )
            retained_start += (2 * rotation_mmax + 1) * (
                degree * degree - complete_degrees * complete_degrees
            )
            retained_row = tl.where(
                signed_order == 0,
                0,
                tl.where(signed_order > 0, 2 * signed_order - 1, -2 * signed_order),
            )
            wigner_offset = retained_start + retained_row * width + local_output
            coefficient = tl.load(
                wigner_ptr + edge * compact_elements + wigner_offset
            ).to(tl.float32)
            message_gradient += envelope * coefficient * cotangent
            tl.store(
                wigner_gradient_ptr + edge * compact_elements + wigner_offset,
                envelope * tl.sum(cotangent * message, axis=0),
            )

        store_sector_gradient(
            message_gradient_ptr, message_gradient, edge, sector_row, channel,
            gradient_stride_edge, gradient_stride_row, gradient_stride_channel,
            channels, sector_width, order, raw_complex, channel_mask,
        )


def compact_tail_backward(
    grad_output,
    messages,
    envelope,
    wigner,
    receivers,
    *,
    lmax, mmax,
    raw_complex,
    wigner_layout,
):
    if use_native_receiver_backward(
        grad_output, messages, envelope, wigner, receivers, lmax, mmax, raw_complex, wigner_layout,
    ):
        return native_receiver_backward(
            grad_output, messages, envelope, wigner, receivers, raw_complex,
        )
    message_gradients = tuple(torch.empty_like(message) for message in messages)
    wigner_gradient = torch.zeros_like(wigner)
    torch.ops.so2.compact_sector_tail_input_backward(
        grad_output, tuple(messages),
        envelope, wigner, receivers,
        message_gradients,
        wigner_gradient,
        lmax, mmax,
        messages[0].shape[-1],
        raw_complex,
        wigner_layout,
    )
    return wigner_gradient, *message_gradients


def sum_backward(
    grad_output,
    messages, envelope, wigner,
    receivers, nodes,
    *,
    lmax, mmax,
    wigner_layout,
    raw_complex=False,
):
    allow_atomic_reduction("dataflow/edge_to_node/sum_aggregation/derivatives.backward")
    spec, edges = sum_forward_validate(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout, raw_complex=raw_complex,
    )
    message_gradients = [torch.empty_like(message) for message in messages]
    envelope_gradient = torch.zeros_like(envelope)
    wigner_gradient = torch.zeros_like(wigner)
    for order, (message, message_gradient) in enumerate(
        zip(messages, message_gradients, strict=True)
    ):
        rows = logical_rows(spec, order)
        sector_message_backward_kernel[(edges,)](
            grad_output,
            envelope,
            wigner,
            receivers,
            message_gradient,
            *message_gradient.stride(), *wigner.stride(),
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax,
            order=order, rows=rows,
            wigner_layout=wigner_layout, raw_complex=raw_complex,
            block_channels=1 << (spec.channels - 1).bit_length(),
            num_warps=4,
        )
        sector_parameter_backward_kernel[(edges, rows)](
            grad_output, message,
            envelope, wigner,
            receivers,
            envelope_gradient, wigner_gradient,
            *message.stride(), *wigner.stride(),
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax,
            order=order, rows=rows,
            wigner_layout=wigner_layout, raw_complex=raw_complex,
            full_block=1 << (2 * spec.lmax).bit_length(),
            channel_block=1 << (spec.channels - 1).bit_length(),
            num_warps=4,
        )
    return envelope_gradient, wigner_gradient, *message_gradients


@triton.jit
def receiver_backward(
    x0, x1, x2, rotation, receivers, t0, t1, t2, scale, dx0, dx1, dx2, drotation,
    edges, specs: tl.constexpr, widths: tl.constexpr, degree: tl.constexpr,
    tile_edges: tl.constexpr,
):
    terms = (t0, t1, t2)
    edge = tl.program_id(0) * tile_edges + tl.arange(0, tile_edges)
    channel = tl.arange(0, 128)
    live = edge < edges
    node = tl.load(receivers + edge, live, 0).to(tl.int32)
    gate_gradient = tl.full((tile_edges, 128), 0.0, tl.float32)
    if degree:
        gate_address = edge[:, None] * 640 + (degree - 1) * 128 + channel[None, :]
        logit = tl.load(x0 + gate_address, live[:, None], 0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-logit))
    for order in tl.static_range(degree + 1):
        size = specs[order][0]
        term_offset = specs[order][7] + (degree - order) * 128
        local = (degree - order) * 128 + channel
        if order == 0:
            addr = edge[:, None] * 640 + 256 + local[None, :]
            raw_pos = tl.load(x0 + addr, live[:, None], 0.0)
            if degree == 0:
                sigmoid = 1.0 / (1.0 + tl.exp(-raw_pos))
        elif order == 1:
            addr = edge[:, None] * (2 * size) + local[None, :]
            raw_pos = tl.load(x1 + addr, live[:, None], 0.0)
            raw_neg = tl.load(x1 + addr + size, live[:, None], 0.0)
        else:
            addr = edge[:, None] * (2 * size) + local[None, :]
            raw_pos = tl.load(x2 + addr, live[:, None], 0.0)
            raw_neg = tl.load(x2 + addr + size, live[:, None], 0.0)
        pos = raw_pos * sigmoid
        grad_pos = tl.full((tile_edges, 128), 0.0, tl.float32)
        if order:
            neg = raw_neg * sigmoid
            grad_neg = tl.full((tile_edges, 128), 0.0, tl.float32)
        for output_degree in tl.static_range(order, 3):
            for component in tl.static_range(2 * output_degree + 1):
                grad_offset = ((node[:, None] * (2 * output_degree + 1) + component)
                               * widths[output_degree] + term_offset + channel[None, :])
                ga = tl.load(terms[output_degree] + grad_offset, live[:, None], 0.0)
                pos_addr = edge * 35 + specs[order][1][output_degree * output_degree - order * order + component]
                a = tl.load(rotation + pos_addr, live, 0.0)
                grad_pos += ga * a[:, None]
                if order:
                    gb = tl.load(terms[output_degree] + grad_offset + size, live[:, None], 0.0)
                    neg_addr = edge * 35 + specs[order][2][output_degree * output_degree - order * order + component]
                    b = tl.load(rotation + neg_addr, live, 0.0)
                    grad_pos += gb * b[:, None]
                    grad_neg += ga * b[:, None] - gb * a[:, None]
                    da = tl.sum(ga * pos - gb * neg, axis=1)
                    db = tl.sum(ga * neg + gb * pos, axis=1)
                    tl.store(drotation + degree * edges * 35 + neg_addr, db, live)
                else:
                    da = tl.sum(ga * pos, axis=1)
                    if degree == 0:
                        bias_grad = tl.load(scale + node * 9 + output_degree * output_degree + component, live, 0.0)
                        da += bias_grad
                tl.store(drotation + degree * edges * 35 + pos_addr, da, live)
        if degree == 0:
            grad_pos *= sigmoid * (1.0 + raw_pos * (1.0 - sigmoid))
        else:
            gate_gradient += raw_pos * grad_pos
            grad_pos *= sigmoid
        if order:
            gate_gradient += raw_neg * grad_neg
            grad_neg *= sigmoid
        if order == 0:
            tl.store(dx0 + addr, grad_pos, live[:, None])
        elif order == 1:
            tl.store(dx1 + addr, grad_pos, live[:, None])
            tl.store(dx1 + addr + size, grad_neg, live[:, None])
        else:
            tl.store(dx2 + addr, grad_pos, live[:, None])
            tl.store(dx2 + addr + size, grad_neg, live[:, None])
    if degree:
        tl.store(dx0 + gate_address, gate_gradient * sigmoid * (1.0 - sigmoid), live[:, None])


@triton.jit
def backward_degree(gradient, radial, rotation, grad_radial, grad_rotation,
                    pair, first, reverse, node0, node1, c, live, matched,
                    scale, degree: tl.constexpr, tile_pair: tl.constexpr):
    width: tl.constexpr = 2 * degree + 1
    offset: tl.constexpr = (0, 1, 10)[degree]
    radial_offset = (pair[:, None] * 3 + degree) * 128 + c[None, :]
    value = tl.load(radial + radial_offset, live[:, None], 0.)
    dr0 = tl.full((tile_pair, 128), 0., tl.float32)
    dr1 = tl.full((tile_pair, 128), 0., tl.float32)
    for k in tl.static_range(width):
        g0 = tl.load(gradient + (node0[:, None] * 9 + degree * degree + k) * 128 + c[None, :],
                     live[:, None], 0.)
        g1 = tl.load(gradient + (node1[:, None] * 9 + degree * degree + k) * 128 + c[None, :],
                     matched[:, None], 0.)
        w0 = tl.load(rotation + first * 35 + offset + k, live, 0.)
        w1 = tl.load(rotation + reverse * 35 + offset + k, matched, 0.)
        dr0 += w0[:, None] * g0
        dr1 += w1[:, None] * g1
        tl.store(grad_rotation + first * 35 + offset + k, scale * tl.sum(value * g0, 1), live)
        tl.store(grad_rotation + reverse * 35 + offset + k, scale * tl.sum(value * g1, 1), matched)
    tl.store(grad_radial + radial_offset, libdevice.mul_rn(scale, dr0) + libdevice.mul_rn(scale, dr1), live[:, None])

@triton.jit
def paired_initial_backward(gradient, radial, rotation, receiver, pairs, grad_radial, grad_rotation,
                            count, scale, tile_pair: tl.constexpr):
    scale = scale.to(tl.float32)
    pair = tl.program_id(0) * tile_pair + tl.arange(0, tile_pair)
    live = pair < count
    first = tl.load(pairs + pair, live, 0)
    reverse = tl.load(pairs + count + pair, live, -1)
    matched = live & (reverse >= 0)
    node0 = tl.load(receiver + first, live, 0).to(tl.int32)
    node1 = tl.load(receiver + reverse, matched, 0).to(tl.int32)
    c = tl.arange(0, 128)
    for degree in tl.static_range(3):
        backward_degree(gradient, radial, rotation, grad_radial, grad_rotation,
                        pair, first, reverse, node0, node1, c, live, matched, scale, degree, tile_pair)
    col = tl.arange(0, 32)
    offset = tl.where(col < 6, col + 4, col + 9)
    tl.store(grad_rotation + first[:, None] * 35 + offset[None, :], 0.,
             live[:, None] & (col[None, :] < 26))
    tl.store(grad_rotation + reverse[:, None] * 35 + offset[None, :], 0.,
             matched[:, None] & (col[None, :] < 26))


SM80_RECEIVER_VJP = r"""#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int channels = 128;
constexpr int seed_count = 8832;

__device__ __forceinline__ float warp_sum(float value) {
    for (int stride = 16; stride > 0; stride /= 2)
        value += __shfl_down_sync(0xffffffff, value, stride);
    return value;
}

template <int Order>
__device__ __forceinline__ void vjp_order(
    const float* raw, const float* rot, const float* const* seed, const float* scale,
    const float (*gate)[4], const float (*dgate)[4], float (*gate_grad)[4],
    float* grad_raw, float* grad_rot, int lane
) {
    constexpr int groups = 3 - Order;
    constexpr int width = groups * channels;
    constexpr int plane = Order == 0 ? 256 : 0;
    constexpr int packed = Order == 0 ? 0 : (Order == 1 ? 384 : 896);
    float p[groups][4], n[groups][4];
    float gp[groups][4] = {}, gn[groups][4] = {};
    float sig[groups][4];
    #pragma unroll
    for (int i = 0; i < groups; ++i) {
        const float4 pv = reinterpret_cast<const float4*>(raw + plane + i * channels)[lane];
        p[i][0] = pv.x; p[i][1] = pv.y; p[i][2] = pv.z; p[i][3] = pv.w;
        if constexpr (Order != 0) {
            const float4 nv = reinterpret_cast<const float4*>(raw + width + i * channels)[lane];
            n[i][0] = nv.x; n[i][1] = nv.y; n[i][2] = nv.z; n[i][3] = nv.w;
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c)
            sig[i][c] = i + Order == 0 ? 1.0f / (1.0f + __expf(-p[i][c])) : gate[i + Order - 1][c];
    }

    #pragma unroll
    for (int degree = Order; degree < 3; ++degree) {
        const int block = degree == 0 ? 0 : (degree == 1 ? 1 : 10);
        const int stride = degree == 0 ? 384 : (degree == 1 ? 896 : 1152);
        #pragma unroll
        for (int index = 0; index < 2 * degree + 1; ++index) {
            constexpr int signed_row = Order == 0 ? 0 : 2 * Order - 1;
            const int a_idx = block + signed_row * (2 * degree + 1) + index;
            const int b_idx = a_idx + (2 * degree + 1);
            const float a = __shfl_sync(0xffffffff, lane == 0 ? rot[a_idx] : 0.0f, 0);
            const float b = Order ? __shfl_sync(0xffffffff, lane == 0 ? rot[b_idx] : 0.0f, 0) : 0.0f;
            float da = 0.0f, db = 0.0f;
            #pragma unroll
            for (int i = 0; i < groups; ++i) {
                const float* row = seed[degree] + index * stride + packed + i * channels;
                const float4 sa = reinterpret_cast<const float4*>(row)[lane];
                const float u[4] = {sa.x, sa.y, sa.z, sa.w};
                float v[4];
                if constexpr (Order != 0) {
                    const float4 sb = reinterpret_cast<const float4*>(row + width)[lane];
                    v[0] = sb.x; v[1] = sb.y; v[2] = sb.z; v[3] = sb.w;
                }
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float pos = p[i][c] * sig[i][c];
                    gp[i][c] += u[c] * a;
                    if constexpr (Order != 0) {
                        const float neg = n[i][c] * sig[i][c];
                        gp[i][c] += v[c] * b;
                        gn[i][c] += u[c] * b - v[c] * a;
                        da += u[c] * pos - v[c] * neg;
                        db += u[c] * neg + v[c] * pos;
                    } else {
                        da += u[c] * pos;
                    }
                }
            }
            da = warp_sum(da);
            if constexpr (Order != 0) db = warp_sum(db);
            if (lane == 0) {
                if constexpr (Order == 0) da += scale[degree * degree + index];
                grad_rot[a_idx] = da;
                if constexpr (Order != 0) grad_rot[b_idx] = db;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < groups; ++i) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (i + Order == 0) {
                gp[i][c] *= sig[i][c] * (1.0f + p[i][c] * (1.0f - sig[i][c]));
            } else {
                float g = p[i][c] * gp[i][c];
                if constexpr (Order != 0) g += n[i][c] * gn[i][c];
                gate_grad[i + Order - 1][c] += g;
                gp[i][c] *= sig[i][c];
                if constexpr (Order != 0) gn[i][c] *= sig[i][c];
            }
        }
        reinterpret_cast<float4*>(grad_raw + plane + i * channels)[lane] =
            make_float4(gp[i][0], gp[i][1], gp[i][2], gp[i][3]);
        if constexpr (Order != 0)
            reinterpret_cast<float4*>(grad_raw + width + i * channels)[lane] =
                make_float4(gn[i][0], gn[i][1], gn[i][2], gn[i][3]);
    }
}

template <int Warps, bool Shared>
__global__ void receiver_vjp(
    const float* x0, const float* x1, const float* x2, const float* rotation, const int* row_ptr,
    const float* t0, const float* t1, const float* t2, const float* scale,
    float* dx0, float* dx1, float* dx2, float* dr
) {
    extern __shared__ float scratch[];
    const int node = blockIdx.x;
    const int start = row_ptr[node], end = row_ptr[node + 1];
    if (start == end) return;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float* seed[3] = {t0 + node * 384, t1 + node * 2688, t2 + node * 5760};
    if constexpr (Shared) {
        for (int i = threadIdx.x; i < seed_count / 4; i += blockDim.x) {
            const float* src = i < 96 ? seed[0] + 4 * i :
                (i < 768 ? seed[1] + 4 * i - 384 : seed[2] + 4 * i - 3072);
            reinterpret_cast<float4*>(scratch)[i] = *reinterpret_cast<const float4*>(src);
        }
        __syncthreads();
        seed[0] = scratch;
        seed[1] = scratch + 384;
        seed[2] = scratch + 3072;
    }
    for (int e = start + blockIdx.y * Warps + warp; e < end; e += gridDim.y * Warps) {
        const float* r = rotation + 35 * e;
        float* grad_r = dr + 35 * e;
        const float* zero = x0 + 640 * e;
        float* grad_zero = dx0 + 640 * e;
        float gate[2][4], dgate[2][4], gate_grad[2][4] = {};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float4 z = reinterpret_cast<const float4*>(zero + i * channels)[lane];
            const float v[4] = {z.x, z.y, z.z, z.w};
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                gate[i][c] = 1.0f / (1.0f + __expf(-v[c]));
                dgate[i][c] = gate[i][c] * (1.0f - gate[i][c]);
            }
        }
        vjp_order<0>(zero, r, seed, scale + 9 * node, gate, dgate, gate_grad, grad_zero, grad_r, lane);
        vjp_order<1>(x1 + 512 * e, r, seed, scale + 9 * node, gate, dgate, gate_grad, dx1 + 512 * e, grad_r, lane);
        vjp_order<2>(x2 + 256 * e, r, seed, scale + 9 * node, gate, dgate, gate_grad, dx2 + 256 * e, grad_r, lane);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            reinterpret_cast<float4*>(grad_zero + i * channels)[lane] =
                make_float4(gate_grad[i][0] * dgate[i][0], gate_grad[i][1] * dgate[i][1],
                            gate_grad[i][2] * dgate[i][2], gate_grad[i][3] * dgate[i][3]);
    }
}
}  // namespace

extern "C" int launch_receiver(
    const float* x0, const float* x1, const float* x2, const float* rotation, const int* row_ptr,
    const float* t0, const float* t1, const float* t2, const float* scale,
    float* dx0, float* dx1, float* dx2, float* dr, int nodes, int warps, int slots,
    bool shared, cudaStream_t stream
) {
    dim3 grid(nodes, slots);
    #define LAUNCH(W, S) receiver_vjp<W, S><<<grid, (W)*32, (S)? seed_count * sizeof(float) : 0, stream>>>( \
        x0, x1, x2, rotation, row_ptr, t0, t1, t2, scale, dx0, dx1, dx2, dr)
    if (warps == 4) {
        if (shared) LAUNCH(4, true); else LAUNCH(4, false);
    } else if (warps == 8) {
        if (shared) LAUNCH(8, true); else LAUNCH(8, false);
    } else if (warps == 2) {
        if (shared) LAUNCH(2, true); else LAUNCH(2, false);
    } else return cudaErrorInvalidValue;
    #undef LAUNCH
    return cudaPeekAtLastError();
}
"""


@cache
def native_kernel():
    launch = load_cuda_library("sm80_shared_receiver.cu", SM80_RECEIVER_VJP, "80").launch_receiver
    launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 3 + [ctypes.c_bool, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


def launch_receiver_vjp_sm80(inputs, config):
    x0, x1, x2, rotation, row_ptr, *_ = inputs
    outputs = tuple(torch.empty_like(t) for t in (x0, x1, x2, rotation))
    nodes = row_ptr.numel() - 1
    if nodes == 0:
        return outputs
    with torch.cuda.device(x0.device):
        stream = torch.cuda.current_stream()
        status = native_kernel()(*(t.data_ptr() for t in (*inputs, *outputs)), nodes, *config, stream.cuda_stream)
    if status:
        raise RuntimeError(f"Shared receiver VJP launch failed with CUDA error {status}")
    return outputs


@custom_op("lunaeq_so2::receiver_vjp", mutates_args=())
def receiver_vjp(
    x0: Tensor, x1: Tensor, x2: Tensor, rotation: Tensor, row_ptr: Tensor,
    g0: Tensor, g1: Tensor, g2: Tensor, gscale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raw = (x0, x1, x2, rotation, row_ptr)
    terms = tuple(g.resolve_neg().contiguous() for g in (g0, g1, g2, gscale))
    assert row_ptr.dtype == torch.int32
    assert all(t.dtype == torch.float32 for t in (x0, x1, x2, rotation, *terms))
    assert all(t.is_contiguous() and t.is_cuda for t in raw)
    assert all(t.data_ptr() % 16 == 0 for t in (x0, x1, x2, *terms[:3]))
    return launch_receiver_vjp_sm80((*raw, *terms), (8, 1, True))

@receiver_vjp.register_fake
def receiver_vjp_fake(x0, x1, x2, rotation, row_ptr, g0, g1, g2, gscale):
    return tuple(torch.empty_like(t) for t in (x0, x1, x2, rotation))


SM90_RECEIVER_VJP = r"""#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int channels = 128;
constexpr int seed_count = 8832;

__device__ __forceinline__ float warp_sum(float value) {
    for (int stride = 16; stride > 0; stride /= 2)
        value += __shfl_down_sync(0xffffffff, value, stride);
    return value;
}

template <int Order>
__device__ __forceinline__ void vjp_order(
    const float* raw, const float* rot, const float* const* seed, const float* scale,
    const float (*gate)[4], float (*gate_grad)[4],
    float* grad_raw, float* grad_rot, int lane
) {
    constexpr int groups = 3 - Order;
    constexpr int width = groups * channels;
    constexpr int plane = Order == 0 ? 256 : 0;
    constexpr int packed = Order == 0 ? 0 : (Order == 1 ? 384 : 896);
    float p[groups][4], n[groups][4];
    float gp[groups][4] = {}, gn[groups][4] = {};
    float sig[groups][4];
    #pragma unroll
    for (int i = 0; i < groups; ++i) {
        const float4 pv = reinterpret_cast<const float4*>(raw + plane + i * channels)[lane];
        p[i][0] = pv.x; p[i][1] = pv.y; p[i][2] = pv.z; p[i][3] = pv.w;
        if constexpr (Order != 0) {
            const float4 nv = reinterpret_cast<const float4*>(raw + width + i * channels)[lane];
            n[i][0] = nv.x; n[i][1] = nv.y; n[i][2] = nv.z; n[i][3] = nv.w;
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c)
            sig[i][c] = i + Order == 0 ? 1.0f / (1.0f + __expf(-p[i][c])) : gate[i + Order - 1][c];
    }

    #pragma unroll
    for (int degree = Order; degree < 3; ++degree) {
        const int block = degree == 0 ? 0 : (degree == 1 ? 1 : 10);
        const int stride = degree == 0 ? 384 : (degree == 1 ? 896 : 1152);
        #pragma unroll
        for (int index = 0; index < 2 * degree + 1; ++index) {
            const int a_idx = block + index * (2 * degree + 1) + degree + Order;
            const int b_idx = a_idx - 2 * Order;
            const float a = __shfl_sync(0xffffffff, lane == 0 ? rot[a_idx] : 0.0f, 0);
            const float b = Order ? __shfl_sync(0xffffffff, lane == 0 ? rot[b_idx] : 0.0f, 0) : 0.0f;
            float da = 0.0f, db = 0.0f;
            #pragma unroll
            for (int i = 0; i < groups; ++i) {
                const float* row = seed[degree] + index * stride + packed + i * channels;
                const float4 sa = reinterpret_cast<const float4*>(row)[lane];
                const float u[4] = {sa.x, sa.y, sa.z, sa.w};
                float v[4];
                if constexpr (Order != 0) {
                    const float4 sb = reinterpret_cast<const float4*>(row + width)[lane];
                    v[0] = sb.x; v[1] = sb.y; v[2] = sb.z; v[3] = sb.w;
                }
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float pos = p[i][c] * sig[i][c];
                    gp[i][c] += u[c] * a;
                    if constexpr (Order != 0) {
                        const float neg = n[i][c] * sig[i][c];
                        gp[i][c] += v[c] * b;
                        gn[i][c] += u[c] * b - v[c] * a;
                        da += u[c] * pos - v[c] * neg;
                        db += u[c] * neg + v[c] * pos;
                    } else {
                        da += u[c] * pos;
                    }
                }
            }
            da = warp_sum(da);
            if constexpr (Order != 0) db = warp_sum(db);
            if (lane == 0) {
                if constexpr (Order == 0) da += scale[degree * degree + index];
                grad_rot[a_idx] = da;
                if constexpr (Order != 0) grad_rot[b_idx] = db;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < groups; ++i) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (i + Order == 0) {
                gp[i][c] *= sig[i][c] * (1.0f + p[i][c] * (1.0f - sig[i][c]));
            } else {
                float g = p[i][c] * gp[i][c];
                if constexpr (Order != 0) g += n[i][c] * gn[i][c];
                gate_grad[i + Order - 1][c] += g;
                gp[i][c] *= sig[i][c];
                if constexpr (Order != 0) gn[i][c] *= sig[i][c];
            }
        }
        reinterpret_cast<float4*>(grad_raw + plane + i * channels)[lane] =
            make_float4(gp[i][0], gp[i][1], gp[i][2], gp[i][3]);
        if constexpr (Order != 0)
            reinterpret_cast<float4*>(grad_raw + width + i * channels)[lane] =
                make_float4(gn[i][0], gn[i][1], gn[i][2], gn[i][3]);
    }
}

template <int Warps, bool Shared>
__global__ void receiver_vjp(
    const float* x0, const float* x1, const float* x2, const float* rotation, const int* row_ptr,
    const float* t0, const float* t1, const float* t2, const float* scale,
    float* dx0, float* dx1, float* dx2, float* dr
) {
    extern __shared__ __align__(16) float scratch[];
    const int node = blockIdx.x;
    const int start = row_ptr[node], end = row_ptr[node + 1];
    if (start == end) return;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float* seed[3] = {t0 + node * 384, t1 + node * 2688, t2 + node * 5760};
    if constexpr (Shared) {
        for (int i = threadIdx.x; i < seed_count / 4; i += blockDim.x) {
            const float* src = i < 96 ? seed[0] + 4 * i :
                (i < 768 ? seed[1] + 4 * i - 384 : seed[2] + 4 * i - 3072);
            reinterpret_cast<float4*>(scratch)[i] = *reinterpret_cast<const float4*>(src);
        }
        __syncthreads();
        seed[0] = scratch;
        seed[1] = scratch + 384;
        seed[2] = scratch + 3072;
    }
    for (int e = start + blockIdx.y * Warps + warp; e < end; e += gridDim.y * Warps) {
        const float* r = rotation + 35 * e;
        float* grad_r = dr + 35 * e;
        const float* zero = x0 + 640 * e;
        float* grad_zero = dx0 + 640 * e;
        float gate[2][4], dgate[2][4], gate_grad[2][4] = {};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float4 z = reinterpret_cast<const float4*>(zero + i * channels)[lane];
            const float v[4] = {z.x, z.y, z.z, z.w};
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                gate[i][c] = 1.0f / (1.0f + __expf(-v[c]));
                dgate[i][c] = gate[i][c] * (1.0f - gate[i][c]);
            }
        }
        vjp_order<0>(zero, r, seed, scale + 9 * node, gate, gate_grad, grad_zero, grad_r, lane);
        vjp_order<1>(x1 + 512 * e, r, seed, scale + 9 * node, gate, gate_grad, dx1 + 512 * e, grad_r, lane);
        vjp_order<2>(x2 + 256 * e, r, seed, scale + 9 * node, gate, gate_grad, dx2 + 256 * e, grad_r, lane);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            reinterpret_cast<float4*>(grad_zero + i * channels)[lane] =
                make_float4(gate_grad[i][0] * dgate[i][0], gate_grad[i][1] * dgate[i][1],
                            gate_grad[i][2] * dgate[i][2], gate_grad[i][3] * dgate[i][3]);
    }
}
}  // namespace

extern "C" int launch_receiver(
    const float* x0, const float* x1, const float* x2, const float* rotation, const int* row_ptr,
    const float* t0, const float* t1, const float* t2, const float* scale,
    float* dx0, float* dx1, float* dx2, float* dr, int nodes, int warps, int slots,
    bool shared, cudaStream_t stream
) {
    dim3 grid(nodes, slots);
    #define LAUNCH(W, S) receiver_vjp<W, S><<<grid, (W)*32, (S)? seed_count * sizeof(float) : 0, stream>>>( \
        x0, x1, x2, rotation, row_ptr, t0, t1, t2, scale, dx0, dx1, dx2, dr)
    if (warps == 4) {
        if (shared) LAUNCH(4, true); else LAUNCH(4, false);
    } else if (warps == 8) {
        if (shared) LAUNCH(8, true); else LAUNCH(8, false);
    } else if (warps == 2) {
        if (shared) LAUNCH(2, true); else LAUNCH(2, false);
    } else return cudaErrorInvalidValue;
    #undef LAUNCH
    return cudaPeekAtLastError();
}
"""


def configure_receiver_backward_sm90():
    global launch_receiver_vjp_sm90
    launch_receiver_vjp_sm90 = load_cuda_library(
        "sm90_shared_receiver.cu", SM90_RECEIVER_VJP, "90a", flags=("-Xptxas=-v",),
    ).launch_receiver
    launch_receiver_vjp_sm90.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 3 + [ctypes.c_bool, ctypes.c_void_p]
    launch_receiver_vjp_sm90.restype = ctypes.c_int


@custom_op("trim_h100::shared_receiver_vjp", mutates_args=())
def receiver_vjp_sm90(
    x0: Tensor, x1: Tensor, x2: Tensor, rotation: Tensor, row_ptr: Tensor,
    g0: Tensor, g1: Tensor, g2: Tensor, gscale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    outputs = tuple(torch.empty_like(t) for t in (x0, x1, x2, rotation))
    nodes = row_ptr.numel() - 1
    if nodes == 0:
        return outputs
    raw = (x0, x1, x2, rotation, row_ptr)
    terms = tuple(g.resolve_neg().contiguous() for g in (g0, g1, g2, gscale))
    assert all(t.is_contiguous() for t in raw)
    with torch.cuda.device(x0.device):
        stream = torch.cuda.current_stream()
        status = launch_receiver_vjp_sm90(*(t.data_ptr() for t in (*raw, *terms, *outputs)),
                            nodes, 8, 1, True, stream.cuda_stream)
    if status:
        raise RuntimeError(f"Shared receiver VJP launch failed with CUDA error {status}")
    return outputs

@receiver_vjp_sm90.register_fake
def receiver_vjp_sm90_fake(x0, x1, x2, rotation, row_ptr, g0, g1, g2, gscale):
    return tuple(torch.empty_like(t) for t in (x0, x1, x2, rotation))


NATIVE_RECEIVER_BACKWARD_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

struct Buffers {
    const float *x[3];
    float *dx[3];
    const float *g, *env, *w;
    float *dw;
};

__device__ float4 add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ float4 subtract(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__device__ float4 negate(float4 a) {
    return make_float4(-a.x,-a.y,-a.z,-a.w);
}
__device__ float4 madd(float w, float4 x, float4 out) {
    return make_float4(fmaf(w, x.x, out.x), fmaf(w, x.y, out.y),
                       fmaf(w, x.z, out.z), fmaf(w, x.w, out.w));
}
__device__ float dot(float4 a, float4 b) {
    return (a.x * b.x + a.y * b.y)+(a.z * b.z + a.w * b.w);
}


__device__ float4 load4(const float *address) {
    return *reinterpret_cast<const float4 *>(address);
}
__device__ void store4(float *address, float4 value) {
    *reinterpret_cast<float4 *>(address) = value;
}

template<int Degree, bool Raw>
__device__ void compute_degree(
    Buffers data, int edge, int receiver, int lane, float scale,
    float w0, float w1, float &dw0, float &dw1
) {
    constexpr int width = 2 * Degree + 1;
    constexpr int start = Degree == 0 ? 0 : Degree == 1 ? 1 : 10;
    int channel = lane * 4;
    float4 gradient[width];
    float4 value[width];
    int main[width], cross[width];
    #pragma unroll
    for (int col = 0; col < width; ++col) {
        int r = Degree * Degree + col;
        gradient[col] = load4(data.g + (receiver * 9 + r) * 128 + channel);
    }
    #pragma unroll
    for (int row = 0; row < width; ++row) {
        int m = (row + 1) / 2;
        int sector = 3 - m;
        bool negative = row > 0 && row % 2 == 0;
        if constexpr (Raw) {
            if (m > 0) {
                int half = sector * 128;
                int base = (edge * 4 * sector + Degree - m) * 128 + channel;
                main[row] = base + (negative ? 2 * half : 0);
                cross[row] = base + (negative ? half : 3 * half);
            } else {
                main[row] = (edge * 3 + Degree) * 128 + channel;
                cross[row] = main[row];
            }
        } else {
            int rows = m == 0 ? sector : 2 * sector;
            main[row] = (edge * rows + Degree - m + negative * sector) * 128 + channel;
            cross[row] = main[row];
        }
        value[row] = load4(data.x[m] + main[row]);
        if (Raw && m > 0) {
            const float4 other = load4(data.x[m] + cross[row]);
            value[row] = negative ? add(value[row], other) : subtract(value[row], other);
        }
    }
    #pragma unroll
    for (int row = 0; row < width; ++row) {
        int m = (row + 1) / 2;
        bool negative = row > 0 && row % 2 == 0;
        const float4 msg = value[row];
        float4 out = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int col = 0; col < width; ++col) {
            int index = start + row * width + col;
            const float coeff = __shfl_sync(0xffffffff, index < 32 ? w0 : w1, index % 32);
            out = madd(scale * coeff, gradient[col], out);
            float reduced = dot(gradient[col], msg);
            #pragma unroll
            for (int distance = 16; distance > 0; distance /= 2) {
                reduced += __shfl_xor_sync(0xffffffff, reduced, distance);
            }
            if (lane == index % 32) {
                if (index < 32) dw0 = scale * reduced;
                else dw1 = scale * reduced;
            }
        }
        store4(data.dx[m] + main[row], out);
        if (Raw && m > 0) {
            store4(data.dx[m] + cross[row], negative ? out : negate(out));
        }
    }
}

template<int Warps, bool Raw, class Index>
__global__ __launch_bounds__(Warps * 32) void receiver_kernel(
    Buffers data, const Index *receivers, int edges
) {
    int lane = threadIdx.x % 32;
    int edge = blockIdx.x * Warps + threadIdx.x / 32;
    if (edge >= edges) return;
    int receiver = receivers[edge];
    float scale = data.env[edge];
    // Lanes load 32 + 3 rotation entries and share them with warp shuffles.
    const float w0 = data.w[edge * 35 + lane];
    const float w1 = lane < 3 ? data.w[edge * 35 + 32 + lane] : 0.f;
    float dw0 = 0.f, dw1 = 0.f;
    compute_degree<0, Raw>(data, edge, receiver, lane, scale, w0, w1, dw0, dw1);
    compute_degree<1, Raw>(data, edge, receiver, lane, scale, w0, w1, dw0, dw1);
    compute_degree<2, Raw>(data, edge, receiver, lane, scale, w0, w1, dw0, dw1);
    // Each lane owns one or two entries of the complete rotation adjoint.
    data.dw[edge * 35 + lane] = dw0;
    if (lane < 3) data.dw[edge * 35 + 32 + lane] = dw1;
}


template<int Warps, class Index>
void launch(
    Buffers data, const Index *receivers, int edges, bool raw, cudaStream_t stream
) {
    const int blocks = (edges + Warps - 1) / Warps;
    if (raw) {
        receiver_kernel<Warps, true><<<blocks, Warps * 32, 0, stream>>>(data, receivers, edges);
    } else {
        receiver_kernel<Warps, false><<<blocks, Warps * 32, 0, stream>>>(data, receivers, edges);
    }
}

void receiver_backward(
    const at::Tensor &m0, const at::Tensor &m1, const at::Tensor &m2,
    const at::Tensor &d0, const at::Tensor &d1, const at::Tensor &d2,
    const at::Tensor &grad, const at::Tensor &envelope, const at::Tensor &wigner,
    const at::Tensor &dw, const at::Tensor &receivers, bool raw, int warps
) {
    const c10::cuda::CUDAGuard guard(grad.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(grad.get_device());
    const Buffers data{
        {m0.const_data_ptr < float>(), m1.const_data_ptr < float>(), m2.const_data_ptr < float>()},
        {d0.data_ptr < float>(), d1.data_ptr < float>(), d2.data_ptr < float>()},
        grad.const_data_ptr < float>(), envelope.const_data_ptr < float>(),
        wigner.const_data_ptr < float>(), dw.data_ptr < float>(),
    };
    const int edges = receivers.numel();
    if (receivers.scalar_type() == at::kLong) {
        const auto *indices = receivers.const_data_ptr < int64_t>();
        if (warps == 2) launch<2>(data, indices, edges, raw, stream);
        else launch<4>(data, indices, edges, raw, stream);
    } else {
        const auto *indices = receivers.const_data_ptr < int32_t>();
        if (warps == 2) launch<2>(data, indices, edges, raw, stream);
        else launch<4>(data, indices, edges, raw, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("receiver_backward", &receiver_backward);
}
"""

@lru_cache(maxsize=1)
def receiver_toolkit_available():
    if torch.version.cuda is None or CUDA_HOME is None:
        return False
    nvcc = Path(CUDA_HOME) / "bin/nvcc"
    if not nvcc.is_file() or not which("ninja") or not which(get_cxx_compiler()):
        return False
    try:
        _, version = compiler_identity(str(nvcc), os.environ.get("PATH", ""))
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return False
    release = re.search(r"release (\d+)\.(\d+)", version)
    if release is None:
        return False
    major, minor = map(int, release.groups())
    return (major, minor) >= (11, 8) and major == int(torch.version.cuda.split(".")[0])


@lru_cache(maxsize=1)
def build_receiver_backward():
    return load_extension(
        "luna_receiver_backward",
        cuda_sources=NATIVE_RECEIVER_BACKWARD_SOURCE,
        extra_cuda_cflags=("-O3", "-gencode=arch=compute_80,code=sm_80",
                          "-gencode=arch=compute_90,code=sm_90"),
        no_implicit_headers=True,
    )


def use_native_receiver_backward(
    grad_output, messages, envelope, wigner, receivers, lmax, mmax, raw_complex, wigner_layout,
):
    edges = envelope.numel()
    if (
        not envelope.is_cuda
        or torch.compiler.is_compiling()
        or (lmax, mmax, wigner_layout) != (2, 2, 6)
        or not 2048 <= edges < 2**21
        or grad_output.shape[1:] != (9, 128)
        or grad_output.numel() >= 2**31
        or wigner.shape != (edges, 35, 1)
        or receivers.shape != (edges,)
        or receivers.dtype not in (torch.int32, torch.int64)
    ):
        return False
    shapes = ((3, 128), (2, 512), (2, 256)) if raw_complex else ((3, 128), (4, 128), (2, 128))
    return (
        len(messages) == len(shapes)
        and all(message.shape == (edges, *shape)
                for message, shape in zip(messages, shapes, strict=True))
        and all(t.dtype == torch.float32 and t.device == envelope.device and t.is_contiguous()
                for t in (grad_output, envelope, wigner, *messages))
        and all(t.data_ptr() % 16 == 0 for t in (grad_output, *messages))
        and receivers.device == envelope.device
        and receivers.is_contiguous()
        and torch.cuda.get_device_capability(envelope.device) in ((8, 0), (9, 0))
        and receiver_toolkit_available()
    )


def native_receiver_backward(grad_output, messages, envelope, wigner, receivers, raw_complex):
    message_gradients = tuple(torch.empty_like(message) for message in messages)
    wigner_gradient = torch.empty_like(wigner)
    warps = 2 if torch.cuda.get_device_capability(envelope.device) == (8, 0) else 4
    build_receiver_backward().receiver_backward(
        *messages, *message_gradients, grad_output, envelope, wigner, wigner_gradient,
        receivers, raw_complex, warps,
    )
    return wigner_gradient, *message_gradients
