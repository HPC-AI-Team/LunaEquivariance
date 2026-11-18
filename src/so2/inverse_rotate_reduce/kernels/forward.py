from __future__ import annotations

import torch
import triton
import triton.language as tl

from ....utils.jit import load_dispatch
from ...utils.layout import get_angular_layout
from ...utils.sector_layout import load_sector_value
from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import (
    ValueStorage,
    combine_preprojection_orders,
    attention_dimensions,
    component_degree,
    kernel_metadata,
    load_attention_value,
    m0_row_offset,
    attention_row_layout,
    preprojection_layout,
    sum_forward_validate,
    validate_inputs,
    validate_row_ptr,
    value_blocks,
    wigner_offset,
)


@triton.jit
def preprojection_order_kernel(
    x, rotation, hx, hrotation, row_ptr, terms, scale,
    order: tl.constexpr, input_width: tl.constexpr, rotation_width: tl.constexpr,
    positive_offsets: tl.constexpr, negative_offsets: tl.constexpr, degrees: tl.constexpr,
    angular_widths: tl.constexpr, angular_indices: tl.constexpr,
    packed_widths: tl.constexpr, packed_offset: tl.constexpr,
    edge_tile: tl.constexpr, channel_tile: tl.constexpr, mixed: tl.constexpr, first_tile: tl.constexpr = 0,
):
    node = tl.program_id(0)
    tile = tl.program_id(1) - first_tile
    channel = tile * channel_tile + tl.arange(0, channel_tile)
    start = tl.load(row_ptr + node).to(tl.int32)
    stop = tl.load(row_ptr + node + 1).to(tl.int32)
    offsets = tl.arange(0, edge_tile)
    slots: tl.constexpr = len(angular_indices)
    real = ()
    imag = ()
    bias = ()
    for _ in tl.static_range(slots):
        real += (tl.zeros((edge_tile, channel_tile), tl.float32),)
        if order:
            imag += (tl.zeros((edge_tile, channel_tile), tl.float32),)
        else:
            bias += (tl.zeros((edge_tile,), tl.float32),)
    for base in range(start, stop, edge_tile):
        edge = base + offsets
        live = edge < stop
        valid = live[:, None] & (channel[None, :] < input_width)
        planes: tl.constexpr = 2 if order else 1
        addr = edge[:, None] * (planes * input_width) + channel[None, :]
        positive = tl.load(x + addr, valid, 0.)
        if mixed:
            h_pos = tl.load(hx + addr, valid, 0.)
        if order:
            negative = tl.load(x + addr + input_width, valid, 0.)
            if mixed:
                h_neg = tl.load(hx + addr + input_width, valid, 0.)
        next_real = ()
        next_imag = ()
        next_bias = ()
        for i in tl.static_range(slots):
            a = tl.load(rotation + edge * rotation_width + positive_offsets[i], live, 0.)
            if mixed:
                da = tl.load(hrotation + edge * rotation_width + positive_offsets[i], live, 0.)
            if order:
                b = tl.load(rotation + edge * rotation_width + negative_offsets[i], live, 0.)
                if mixed:
                    db = tl.load(hrotation + edge * rotation_width + negative_offsets[i], live, 0.)
                    next_real += (real[i] + a[:, None] * h_pos + b[:, None] * h_neg
                                  + da[:, None] * positive + db[:, None] * negative,)
                    next_imag += (imag[i] + b[:, None] * h_pos - a[:, None] * h_neg
                                  + db[:, None] * positive - da[:, None] * negative,)
                else:
                    next_real += (real[i] + a[:, None] * positive + b[:, None] * negative,)
                    next_imag += (imag[i] + b[:, None] * positive - a[:, None] * negative,)
            elif mixed:
                next_real += (real[i] + a[:, None] * h_pos + da[:, None] * positive,)
                next_bias += (bias[i] + da,)
            else:
                next_real += (real[i] + a[:, None] * positive,)
                next_bias += (bias[i] + a,)
        real = next_real
        if order:
            imag = next_imag
        else:
            bias = next_bias
    for i in tl.static_range(slots):
        addr = (terms[degrees[i]] + (node * angular_widths[i] + angular_indices[i]) * packed_widths[i]
                + packed_offset + channel)
        tl.store(addr, tl.sum(real[i], axis=0), channel < input_width)
        if order:
            tl.store(addr + input_width, tl.sum(imag[i], axis=0), channel < input_width)
        elif tile == 0:
            tl.store(scale + node * slots + i, tl.sum(bias[i], axis=0))


@triton.jit
def preprojection_order_rows(
    x, rotation, hx, hrotation, row_ptr, terms, scale,
    order: tl.constexpr, spec: tl.constexpr, rotation_width: tl.constexpr,
    mixed: tl.constexpr, first_tile: tl.constexpr,
):
    width: tl.constexpr = spec[0]
    positives: tl.constexpr = spec[1]
    negatives: tl.constexpr = spec[2]
    degrees: tl.constexpr = spec[3]
    angular_widths: tl.constexpr = spec[4]
    angular_indices: tl.constexpr = spec[5]
    packed_widths: tl.constexpr = spec[6]
    packed_offset: tl.constexpr = spec[7]
    preprojection_order_kernel(
        x, rotation, hx, hrotation, row_ptr, terms, scale,
        order, width, rotation_width, positives, negatives, degrees, angular_widths,
        angular_indices, packed_widths, packed_offset, 8, 32, mixed, first_tile,
    )


@triton.jit
def preprojection_orders_kernel(
    x0, x1, x2, rotation, hx0, hx1, hx2, hrotation, row_ptr, terms, scale,
    spec0: tl.constexpr, spec1: tl.constexpr, spec2: tl.constexpr,
    rotation_width: tl.constexpr, mixed: tl.constexpr,
):
    tile = tl.program_id(1)
    # The order widths 384, 256, and 128 occupy 12, 8, and 4 tiles.
    if tile < 12:
        preprojection_order_rows(
            x0, rotation, hx0, hrotation, row_ptr, terms, scale, 0, spec0, rotation_width, mixed, 0,
        )
    elif tile < 20:
        preprojection_order_rows(
            x1, rotation, hx1, hrotation, row_ptr, terms, scale, 1, spec1, rotation_width, mixed, 12,
        )
    else:
        preprojection_order_rows(
            x2, rotation, hx2, hrotation, row_ptr, terms, scale, 2, spec2, rotation_width, mixed, 20,
        )


def preprojection_forward(sectors, rotation, row_ptr, *, directions=None, layout=None):
    use_l2_schedule = layout is None
    layout = preprojection_layout() if use_l2_schedule else layout
    nodes = row_ptr.numel() - 1
    terms = tuple(
        rotation.new_empty((nodes, angular, packed))
        for angular, packed in zip(layout.angular_widths, layout.packed_widths, strict=True)
    )
    scale = rotation.new_empty((nodes, sum(layout.angular_widths)))
    if nodes == 0:
        return terms, scale
    mixed = directions is not None
    sector_directions, rotation_direction = directions if mixed else (sectors, rotation)
    edge_tile, channel_tile = 8, 32
    if use_l2_schedule and combine_preprojection_orders(rotation, nodes):
        preprojection_orders_kernel[(nodes, 24)](
            *sectors, rotation, *sector_directions, rotation_direction, row_ptr, terms, scale,
            *(tuple(spec) for spec in layout.orders), layout.rotation_width, mixed,
            num_warps=2, num_stages=1,
        )
        return terms, scale
    for order, (sector, direction, spec) in enumerate(
        zip(sectors, sector_directions, layout.orders, strict=True)
    ):
        preprojection_order_kernel[(nodes, triton.cdiv(spec.input_width, channel_tile))](
            sector, rotation, direction, rotation_direction, row_ptr, terms, scale,
            order, spec.input_width, layout.rotation_width,
            spec.positive_offsets, spec.negative_offsets, spec.degrees, spec.angular_widths,
            spec.angular_indices, spec.packed_widths, spec.packed_offset,
            edge_tile, channel_tile, mixed, num_warps=2, num_stages=1,
        )
    return terms, scale


@triton.jit
def receiver_sparse_forward_direction_kernel(
    forward_output_ptr,
    values_ptr,
    logits_ptr,
    envelope_ptr,
    wigner_inv_ptr,
    row_ptr,
    stats_ptr,
    values_direction_ptr,
    logits_direction_ptr,
    envelope_direction_ptr,
    wigner_direction_ptr,
    output_direction_ptr,
    eps,
    heads: tl.constexpr,
    value_channels: tl.constexpr,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    full_components: tl.constexpr,
    reduced_degrees: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    block_full: tl.constexpr,
    block_channels: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """One edge pass for the JVP of online softmax and sparse reduction."""

    receiver = tl.program_id(0)
    channel_tile = tl.program_id(1)
    full = tl.arange(0, block_full)
    channel = channel_tile * block_channels + tl.arange(0, block_channels)
    full_mask = full < full_components
    channel_mask = channel < channels
    head = channel // value_channels
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
    denominator_direction = tl.zeros((block_channels,), tl.float32)
    numerator_direction = tl.zeros((block_full, block_channels), tl.float32)
    edge_begin = tl.load(row_ptr + receiver)
    edge_end = tl.load(row_ptr + receiver + 1)

    for edge in tl.range(edge_begin, edge_end):
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
        shifted_exp = tl.exp(logit - receiver_max)
        denominator_direction += shifted_exp * (
            envelope_direction + envelope * logit_direction
        )
        primal_scale = envelope * envelope * shifted_exp
        scale_direction = shifted_exp * (
            2.0 * envelope * envelope_direction + envelope * envelope * logit_direction
        )

        for reduced in tl.static_range(0, reduced_components):
            degree = list(reduced_degrees)[reduced]
            degree_start = degree * degree
            degree_end = (degree + 1) * (degree + 1)
            degree_mask = full_mask & (full >= degree_start) & (full < degree_end)
            coefficient_offset = (
                edge * compact_elements
                + list(compact_row_offsets)[reduced]
                + full
                - degree_start
            )
            coefficient = tl.load(
                wigner_inv_ptr + coefficient_offset,
                mask=degree_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient_direction = tl.load(
                wigner_direction_ptr + coefficient_offset,
                mask=degree_mask,
                other=0.0,
            ).to(tl.float32)
            value = load_attention_value(
                values_ptr, edge, reduced, channel, channel_mask,
                channels, reduced_components, sector_layout,
            )
            value_direction = load_attention_value(
                values_direction_ptr, edge,
                reduced,
                channel, channel_mask,
                channels, reduced_components, sector_layout,
            )
            primal_scaled_value = primal_scale * value
            directional_scaled_value = (
                scale_direction * value + primal_scale * value_direction
            )
            numerator_direction += (
                coefficient[:, None] * directional_scaled_value[None, :]
                + coefficient_direction[:, None] * primal_scaled_value[None, :]
            )

    forward_output = tl.load(
        forward_output_ptr
        + receiver * full_components * channels
        + full[:, None] * channels
        + channel[None, :],
        mask=full_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    output_direction = (
        numerator_direction - forward_output * denominator_direction[None, :]
    ) / receiver_denominator[None, :]
    tl.store(
        output_direction_ptr
        + receiver * full_components * channels
        + full[:, None] * channels
        + channel[None, :],
        output_direction,
        mask=full_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def sorted_coo_to_csr_kernel(receivers_ptr, row_ptr_ptr, edges):
    receiver = tl.program_id(0)
    lower = 0
    upper = edges
    for _ in tl.static_range(0, 32):
        active = lower < upper
        middle = (lower + upper) // 2
        value = tl.load(
            receivers_ptr + middle,
            mask=active & (middle < edges),
            other=receiver,
        )
        move_lower = active & (value < receiver)
        lower = tl.where(move_lower, middle + 1, lower)
        upper = tl.where(active & ~move_lower, middle, upper)
    tl.store(row_ptr_ptr + receiver, lower)


@triton.jit
def receiver_online_sparse_rotate_reduce_kernel(
    values_ptr, logits_ptr, envelope_ptr, wigner_inv_ptr,
    row_ptr, workspace_ptr, stats_element_offset, eps,
    heads: tl.constexpr,
    value_channels: tl.constexpr,
    channels: tl.constexpr,
    reduced_components: tl.constexpr,
    full_components: tl.constexpr,
    reduced_degrees: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    block_full: tl.constexpr,
    block_channels: tl.constexpr,
    sector_layout: tl.constexpr = (),
):
    """One-pass online softmax, sparse inverse rotation, and reduction."""

    receiver = tl.program_id(0)
    channel_tile = tl.program_id(1)
    full = tl.arange(0, block_full)
    channel = channel_tile * block_channels + tl.arange(0, block_channels)
    full_mask = full < full_components
    channel_mask = channel < channels
    head = channel // value_channels

    running_max = tl.full((block_channels,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_channels,), tl.float32)
    accumulator = tl.zeros((block_full, block_channels), tl.float32)
    edge_begin = tl.load(row_ptr + receiver)
    edge_end = tl.load(row_ptr + receiver + 1)

    for edge in tl.range(edge_begin, edge_end):
        logit = tl.load(
            logits_ptr + edge * heads + head,
            mask=channel_mask,
            other=-float("inf"),
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        next_max = tl.maximum(running_max, logit)
        old_scale = tl.exp(running_max - next_max)
        old_scale = tl.where(running_max == -float("inf"), 0.0, old_scale)
        edge_exp = tl.exp(logit - next_max)
        running_sum = running_sum * old_scale + envelope * edge_exp
        accumulator *= old_scale[None, :]
        weighted_value_scale = envelope * envelope * edge_exp

        for reduced in tl.static_range(0, reduced_components):
            degree = list(reduced_degrees)[reduced]
            degree_start = degree * degree
            degree_end = (degree + 1) * (degree + 1)
            degree_mask = full_mask & (full >= degree_start) & (full < degree_end)
            wigner = tl.load(
                wigner_inv_ptr + edge * compact_elements
                + list(compact_row_offsets)[reduced] + full - degree_start,
                mask=degree_mask,
                other=0.0,
            ).to(tl.float32)
            value = load_attention_value(
                values_ptr, edge, reduced, channel, channel_mask,
                channels, reduced_components, sector_layout,
            )
            accumulator += wigner[:, None] * (value * weighted_value_scale)[None, :]
        running_max = next_max

    output = accumulator / (running_sum[None, :] + eps)
    tl.store(
        workspace_ptr + receiver * full_components * channels
        + full[:, None] * channels + channel[None, :],
        output,
        mask=full_mask[:, None] & channel_mask[None, :],
    )
    head_leader = (channel % value_channels) == 0
    stats_offset = stats_element_offset + (receiver * heads + head) * 2
    tl.store(
        workspace_ptr + stats_offset,
        running_max,
        mask=channel_mask & head_leader,
    )
    tl.store(
        workspace_ptr + stats_offset + 1,
        running_sum,
        mask=channel_mask & head_leader,
    )


def attention_forward(
    values: ValueStorage,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    """Write node values, then per-head softmax maxima and denominators.

    The autograd boundary saves this workspace. Each derivative stage
    reconstructs edge-local terms from the inputs and saved statistics.
    """
    channels, heads, value_channels = attention_dimensions(values, logits)
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
    output_elements_per_node = spec.full_components * channels
    workspace_elements_per_node = output_elements_per_node + (heads * 2)
    workspace = torch.empty(
        (num_nodes * workspace_elements_per_node,),
        device=value_blocks(values)[0].device,
        dtype=value_blocks(values)[0].dtype,
    )
    stats_element_offset = num_nodes * output_elements_per_node
    block_full = triton.next_power_of_2(spec.full_components)
    block_channels = 32 if block_full > 32 else 64
    receiver_online_sparse_rotate_reduce_kernel[
        (num_nodes, triton.cdiv(channels, block_channels))
    ](
        values, logits, envelope_flat, wigner_inv,
        row_ptr, workspace, stats_element_offset, eps,
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
    return workspace


@triton.jit
def edge_degree_forward_kernel(
    radial_ptr, envelope_ptr, wigner_ptr,
    row_ptr,
    output_ptr,
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
        radial = tl.load(
            radial_ptr + edge * degrees * channels + degree * channels + channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        wigner = tl.load(
            wigner_ptr
            + edge * compact_elements
            + m0_row_offset(degree, m0_offsets, degrees)
            + component
            - degree * degree,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += envelope * wigner * radial
    tl.store(
        output_ptr + receiver * components * channels + offset,
        accumulator / rescale,
        mask=mask,
    )


@triton.jit
def edge_degree_forward_tiled(
    radial, envelope, wigner, row_ptr, output, rescale,
    components: tl.constexpr, degrees: tl.constexpr, channels: tl.constexpr,
    lmax: tl.constexpr, compact_elements: tl.constexpr, m0_offsets: tl.constexpr,
    edge_tile: tl.constexpr, channel_tile: tl.constexpr,
):
    receiver = tl.program_id(0)
    component = tl.program_id(1)
    channel = tl.program_id(2) * channel_tile + tl.arange(0, channel_tile)
    degree = component_degree(component, lmax)
    rotation_row = m0_row_offset(degree, m0_offsets, degrees) + component - degree * degree
    first = tl.load(row_ptr + receiver)
    end = tl.load(row_ptr + receiver + 1)
    lane = tl.arange(0, edge_tile)
    accumulation = tl.zeros((edge_tile, channel_tile), tl.float32)
    for start in range(first, end, edge_tile):
        edge = start + lane
        live = edge < end
        radial_offset = (edge[:, None] * degrees + degree) * channels + channel[None, :]
        coefficients = tl.load(
            radial + radial_offset, live[:, None] & (channel[None, :] < channels), 0.0,
        )
        inverse = tl.load(wigner + edge * compact_elements + rotation_row, live, 0.0)
        factor = tl.load(envelope + edge, live, 0.0)
        accumulation += (factor * inverse)[:, None] * coefficients
    tl.store(
        output + (receiver * components + component) * channels + channel,
        tl.sum(accumulation, axis=0) / rescale,
        channel < channels,
    )


def scalar_embedding_forward(radial, envelope, wigner, row_ptr, rescale, spec):
    nodes = row_ptr.shape[0] - 1
    output = torch.empty(
        (nodes, spec.full_components, spec.channels),
        device=radial.device,
        dtype=radial.dtype,
    )
    if nodes == 0:
        return output
    edges = radial.shape[0]
    # Dense receiver lists benefit from reducing several edges per iteration.
    if (
        spec.lmax == 4
        and spec.channels == 128
        and edges >= max(16384, 32 * nodes)
        and not torch.compiler.is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(radial.device) in ((8, 0), (9, 0))
    ):
        edge_degree_forward_tiled[(nodes, spec.full_components, 2)](
            radial, envelope, wigner, row_ptr, output, rescale,
            **kernel_metadata(spec), edge_tile=16, channel_tile=64, num_warps=4,
        )
        return output
    elements = spec.full_components * spec.channels
    block_elements = min(4096, triton.next_power_of_2(elements))
    grid = (nodes, triton.cdiv(elements, block_elements))
    edge_degree_forward_kernel[grid](
        radial, envelope, wigner,
        row_ptr, output, rescale,
        **kernel_metadata(spec),
        block_elements=block_elements,
        num_warps=8 if block_elements >= 2048 else 4,
    )
    return output


@triton.jit
def forward_kernel(
    node_ptr,
    weight_ptr,
    atomic_ptr,
    nodes,
    row_stride,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block)
    mask = (node < nodes) & (channel < channels)
    atomic = tl.load(atomic_ptr + node)
    offset = node * row_stride + channel
    value = tl.load(node_ptr + offset, mask=mask, other=0.0)
    value += tl.load(weight_ptr + atomic * channels + channel, mask=mask, other=0.0)
    tl.store(node_ptr + offset, value, mask=mask)


@triton.jit
def scalar_add_kernel(
    node_ptr, scalar_ptr, nodes, row_stride, channels: tl.constexpr, block: tl.constexpr
):
    node = tl.program_id(0)
    channel = tl.arange(0, block)
    mask = (node < nodes) & (channel < channels)
    offset = node * row_stride + channel
    value = tl.load(node_ptr + offset, mask=mask, other=0.0)
    value += tl.load(scalar_ptr + node * channels + channel, mask=mask, other=0.0)
    tl.store(node_ptr + offset, value, mask=mask)


@triton.jit(
    do_not_specialize=("nodes",),
    do_not_specialize_on_alignment=("row_ptr",),
)
def receiver_forward_kernel(
    messages,
    envelope_ptr, wigner_ptr,
    row_ptr, output_ptr, nodes,
    message_strides,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    output_stride_node, output_stride_full,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr,
    raw_complex: tl.constexpr, wigner_layout: tl.constexpr,
    block_channels: tl.constexpr,
    receiver_tile: tl.constexpr,
):
    receiver_group = tl.program_id(0)
    full = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    dense_center = degree * degree + degree

    for tile_offset in tl.static_range(0, receiver_tile):
        receiver = receiver_group * receiver_tile + tile_offset
        receiver_mask = receiver < nodes
        start = tl.load(row_ptr + receiver, mask=receiver_mask, other=0).to(tl.int32)
        end = tl.load(row_ptr + receiver + 1, mask=receiver_mask, other=0).to(tl.int32)
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
        for edge in range(start, end):
            envelope = tl.load(envelope_ptr + edge).to(tl.float32)
            message0 = load_sector_value(
                messages[0], edge, degree, channel,
                message_strides[0][0], message_strides[0][1], message_strides[0][2],
                channels, lmax + 1, 0, False, channel_mask,
            )
            coefficient0 = tl.load(
                wigner_ptr
                + wigner_offset(
                    edge, dense_center, full,
                    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
                    wigner_layout,
                )
            ).to(tl.float32)
            local = coefficient0 * message0

            for order in tl.static_range(1, mmax + 1):
                m_mask = channel_mask & (degree >= order)
                m_row = degree - order
                m_width = tl.constexpr(lmax - order + 1)
                message_positive = load_sector_value(
                    messages[order], edge, m_row, channel,
                    message_strides[order][0], message_strides[order][1], message_strides[order][2],
                    channels, m_width, order, raw_complex, m_mask,
                )
                message_negative = load_sector_value(
                    messages[order], edge, m_width + m_row, channel,
                    message_strides[order][0], message_strides[order][1], message_strides[order][2],
                    channels, m_width, order, raw_complex, m_mask,
                )
                coefficient_positive = tl.load(
                    wigner_ptr
                    + wigner_offset(
                        edge, dense_center + order, full,
                        wigner_stride_edge, wigner_stride_row, wigner_stride_column,
                        wigner_layout,
                    ),
                    mask=degree >= order,
                    other=0.0,
                ).to(tl.float32)
                coefficient_negative = tl.load(
                    wigner_ptr
                    + wigner_offset(
                        edge, dense_center - order, full,
                        wigner_stride_edge, wigner_stride_row, wigner_stride_column,
                        wigner_layout,
                    ),
                    mask=degree >= order,
                    other=0.0,
                ).to(tl.float32)
                local += (
                    coefficient_positive * message_positive
                    + coefficient_negative * message_negative
                )

            accumulator += envelope * local
        tl.store(
            output_ptr + receiver * output_stride_node + full * output_stride_full + channel,
            accumulator,
            mask=receiver_mask & channel_mask,
        )


@triton.jit
def degree_forward(
    message0_ptr,
    message1_ptr,
    message2_ptr,
    envelope_ptr, wigner_ptr,
    row_ptr, output_ptr, nodes,
    message0_stride_edge, message0_stride_row, message0_stride_channel,
    message1_stride_edge, message1_stride_row, message1_stride_channel,
    message2_stride_edge, message2_stride_row, message2_stride_channel,
    wigner_stride_edge, output_stride_node, output_stride_full,
    channels: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    degree: tl.constexpr,
    wigner_layout: tl.constexpr,
):
    receiver = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    start = tl.load(row_ptr + receiver).to(tl.int32)
    end = tl.load(row_ptr + receiver + 1).to(tl.int32)
    width: tl.constexpr = 2 * degree + 1
    block_start: tl.constexpr = degree * (4 * degree * degree - 1) // 3
    accumulators = ()
    for out in tl.static_range(width):
        accumulators += (tl.zeros(channel.shape, dtype=tl.float32),)
    for edge in range(start, end):
        envelope = tl.load(envelope_ptr + edge).to(tl.float32)
        message0 = load_sector_value(
            message0_ptr, edge, degree, channel,
            message0_stride_edge, message0_stride_row, message0_stride_channel,
            channels, 3, 0, False, channel_mask,
        )
        if degree >= 1:
            message1_positive = load_sector_value(
                message1_ptr, edge, degree - 1, channel,
                message1_stride_edge, message1_stride_row, message1_stride_channel,
                channels, 2, 1, raw_complex, channel_mask,
            )
            message1_negative = load_sector_value(
                message1_ptr, edge, 2 + degree - 1, channel,
                message1_stride_edge, message1_stride_row, message1_stride_channel,
                channels, 2, 1, raw_complex, channel_mask,
            )
        if degree >= 2:
            message2_positive = load_sector_value(
                message2_ptr, edge, 0, channel,
                message2_stride_edge, message2_stride_row, message2_stride_channel,
                channels, 1, 2, raw_complex, channel_mask,
            )
            message2_negative = load_sector_value(
                message2_ptr, edge, 1, channel,
                message2_stride_edge, message2_stride_row, message2_stride_channel,
                channels, 1, 2, raw_complex, channel_mask,
            )
        next_accumulators = ()
        for out in tl.static_range(width):
            coefficient0 = tl.load(
                wigner_ptr + edge * wigner_stride_edge + (block_start + 0 * width + out)
            ).to(tl.float32)
            local = coefficient0 * message0
            if degree >= 1:
                coefficient1_positive = tl.load(
                    wigner_ptr + edge * wigner_stride_edge + (block_start + 1 * width + out)
                ).to(tl.float32)
                coefficient1_negative = tl.load(
                    wigner_ptr + edge * wigner_stride_edge + (block_start + 2 * width + out)
                ).to(tl.float32)
                local += (
                    coefficient1_positive * message1_positive
                    + coefficient1_negative * message1_negative
                )
            if degree >= 2:
                coefficient2_positive = tl.load(
                    wigner_ptr + edge * wigner_stride_edge + (block_start + 3 * width + out)
                ).to(tl.float32)
                coefficient2_negative = tl.load(
                    wigner_ptr + edge * wigner_stride_edge + (block_start + 4 * width + out)
                ).to(tl.float32)
                local += (
                    coefficient2_positive * message2_positive
                    + coefficient2_negative * message2_negative
                )
            next_accumulators += (accumulators[out] + envelope * local,)
        accumulators = next_accumulators
    for out in tl.static_range(width):
        tl.store(
            output_ptr
            + receiver * output_stride_node
            + (degree * degree + out) * output_stride_full
            + channel,
            accumulators[out],
            mask=channel_mask,
        )


@triton.jit(do_not_specialize=("nodes",), do_not_specialize_on_alignment=("row_ptr",))
def receiver_degree_forward_kernel(
    message0_ptr,
    message1_ptr,
    message2_ptr,
    message3_ptr,
    envelope_ptr, wigner_ptr,
    row_ptr, output_ptr, nodes,
    message0_stride_edge, message0_stride_row, message0_stride_channel,
    message1_stride_edge, message1_stride_row, message1_stride_channel,
    message2_stride_edge, message2_stride_row, message2_stride_channel,
    message3_stride_edge, message3_stride_row, message3_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    output_stride_node, output_stride_full,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr,
    raw_complex: tl.constexpr, wigner_layout: tl.constexpr,
    block_channels: tl.constexpr,
    receiver_tile: tl.constexpr,
):
    tl.static_assert(channels == 128)
    tl.static_assert(full_components == 9)
    tl.static_assert(lmax == 2)
    tl.static_assert(mmax == 2)
    tl.static_assert(wigner_layout == 6)
    tl.static_assert(block_channels == 128)
    tl.static_assert(receiver_tile == 1)
    for degree in tl.static_range(3):
        if tl.program_id(1) == degree:
            degree_forward(
                message0_ptr,
                message1_ptr,
                message2_ptr,
                envelope_ptr, wigner_ptr,
                row_ptr, output_ptr, nodes,
                message0_stride_edge, message0_stride_row, message0_stride_channel,
                message1_stride_edge, message1_stride_row, message1_stride_channel,
                message2_stride_edge, message2_stride_row, message2_stride_channel,
                wigner_stride_edge, output_stride_node, output_stride_full,
                channels,
                raw_complex,
                block_channels,
                degree,
                wigner_layout,
            )


@triton.jit
def initial_atom_kernel(
    output_ptr,
    weight_ptr,
    atomic_ptr,
    elements,
    node_stride,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    node = offset // node_stride
    within = offset - node * node_stride
    channel = within % channels
    atomic = tl.load(atomic_ptr + node, mask=mask, other=0)
    value = tl.load(
        weight_ptr + atomic * channels + channel,
        mask=mask & (within < channels),
        other=0.0,
    )
    tl.store(output_ptr + offset, value, mask=mask)


@triton.jit
def accumulate_degree_tile(
    messages, strides, envelope_ptr, wigner_ptr, row_ptr, output_ptr,
    wigner_stride, receiver, lmax: tl.constexpr, mmax: tl.constexpr,
    channels: tl.constexpr, raw_complex: tl.constexpr,
    degree: tl.constexpr, edge_tile: tl.constexpr, channel_tile: tl.constexpr,
):
    width: tl.constexpr = 2 * degree + 1
    retained_order: tl.constexpr = min(degree, mmax)
    complete: tl.constexpr = min(degree, mmax + 1)
    block_start: tl.constexpr = (
        complete * (4 * complete * complete - 1) // 3
        + (2 * mmax + 1) * (degree * degree - complete * complete)
    )
    start = tl.load(row_ptr + receiver).to(tl.int32)
    end = tl.load(row_ptr + receiver + 1).to(tl.int32)
    channel = tl.program_id(2) * channel_tile + tl.arange(0, channel_tile)[None, :]
    edge_offsets = tl.arange(0, edge_tile)[:, None]
    accumulators = ()
    for output in tl.static_range(width):
        accumulators += (tl.zeros((edge_tile, channel_tile), tl.float32),)

    for block in range(start, end, edge_tile):
        edge = block + edge_offsets
        edge_mask = edge < end
        value_mask = edge_mask & (channel < channels)
        envelope = tl.load(envelope_ptr + edge, mask=edge_mask, other=0)
        values = (load_sector_value(
            messages[0], edge, degree, channel,
            strides[0][0], strides[0][1], strides[0][2],
            channels, lmax + 1, 0, False, value_mask,
        ),)
        for order in tl.static_range(1, retained_order + 1):
            order_width = tl.constexpr(lmax + 1 - order)
            positive = load_sector_value(
                messages[order], edge, degree - order, channel,
                strides[order][0], strides[order][1], strides[order][2],
                channels, order_width, order, raw_complex, value_mask,
            )
            negative = load_sector_value(
                messages[order], edge, order_width + degree - order, channel,
                strides[order][0], strides[order][1], strides[order][2],
                channels, order_width, order, raw_complex, value_mask,
            )
            values += (positive, negative)

        next_accumulators = ()
        for output in tl.static_range(width):
            rotated = tl.zeros((edge_tile, channel_tile), tl.float32)
            for row in tl.static_range(2 * retained_order + 1):
                coefficient = tl.load(
                    wigner_ptr + edge * wigner_stride + block_start + row * width + output,
                    mask=edge_mask, other=0,
                )
                rotated += coefficient * values[row]
            next_accumulators += (accumulators[output] + envelope * rotated,)
        accumulators = next_accumulators

    output_channel = tl.program_id(2) * channel_tile + tl.arange(0, channel_tile)
    for output in tl.static_range(width):
        value = tl.sum(accumulators[output], axis=0)
        offset = (receiver * (lmax + 1) ** 2 + degree * degree + output) * channels
        tl.store(
            output_ptr + offset + output_channel, value, mask=output_channel < channels,
        )


@triton.jit
def receiver_edge_tiled_kernel(
    messages, strides, envelope_ptr, wigner_ptr, row_ptr, output_ptr,
    wigner_stride, lmax: tl.constexpr, mmax: tl.constexpr,
    channels: tl.constexpr, raw_complex: tl.constexpr,
    edge_tile: tl.constexpr, channel_tile: tl.constexpr,
):
    receiver = tl.program_id(0)
    for degree in tl.static_range(lmax + 1):
        if tl.program_id(1) == degree:
            accumulate_degree_tile(
                messages, strides, envelope_ptr, wigner_ptr, row_ptr, output_ptr,
                wigner_stride, receiver, lmax, mmax, channels, raw_complex,
                degree, edge_tile, channel_tile,
            )


@triton.jit
def sector_forward_kernel(
    message_ptr,
    envelope_ptr, wigner_ptr,
    receivers_ptr,
    output_ptr,
    message_stride_edge, message_stride_row, message_stride_channel,
    wigner_stride_edge, wigner_stride_row, wigner_stride_column,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    order: tl.constexpr,
    rows: tl.constexpr,
    wigner_layout: tl.constexpr,
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
        value = tl.load(
            message_ptr
            + edge * message_stride_edge
            + row * message_stride_row
            + channel * message_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        for local_full in tl.static_range(0, 2 * lmax + 1):
            full_mask = local_full < 2 * degree + 1
            full = degree * degree + local_full
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
            tl.atomic_add(
                output_ptr + receiver * full_components * channels + full * channels + channel,
                envelope * coefficient * value,
                mask=channel_mask & full_mask,
            )


def scatter_order(
    message, envelope, wigner, receivers, output, spec, order, edges, wigner_layout
):
    rows = message.shape[1]
    sector_forward_kernel[(edges,)](
        message, envelope, wigner,
        receivers, output,
        *message.stride(), *wigner.stride(),
        channels=spec.channels,
        full_components=spec.full_components,
        lmax=spec.lmax,
        order=order, rows=rows,
        wigner_layout=wigner_layout,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )


def receiver_sector_tail_forward(
    messages, envelope, wigner,
    row_ptr, nodes,
    *,
    lmax, mmax,
    receiver_tile=1,
    num_warps=1,
    raw_complex=False,
    wigner_layout=0,
):
    if not 0 <= mmax <= lmax or len(messages) != mmax + 1:
        raise ValueError("the receiver-owned tail requires 0 <= mmax <= lmax")
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=messages[0].shape[2])
    if row_ptr.dtype != torch.int32 or tuple(row_ptr.shape) != (nodes + 1,):
        raise ValueError("row_ptr must be int32 with nodes + 1 entries")
    output = torch.empty(
        (nodes, spec.full_components, spec.channels),
        device=messages[0].device,
        dtype=messages[0].dtype,
    )
    messages = tuple(messages)
    if wigner_layout >= 4:
        torch.ops.so2.compact_receiver_tail_forward(
            messages, envelope, wigner,
            row_ptr, output, nodes,
            lmax, mmax,
            spec.channels,
            raw_complex,
            wigner_layout,
        )
    else:
        receiver_forward_kernel[
            (triton.cdiv(nodes, receiver_tile), spec.full_components)
        ](
            messages, envelope, wigner,
            row_ptr, output, nodes,
            tuple(message.stride() for message in messages),
            *wigner.stride(), output.stride(0), output.stride(1),
            channels=spec.channels,
            full_components=spec.full_components,
            lmax=spec.lmax, mmax=spec.mmax,
            raw_complex=raw_complex, wigner_layout=wigner_layout,
            block_channels=triton.next_power_of_2(spec.channels),
            receiver_tile=receiver_tile,
            num_warps=num_warps,
        )
    return output


def sum_forward(
    messages, envelope, wigner, receivers, nodes, *, lmax, mmax, wigner_layout
):
    allow_atomic_reduction("dataflow/edge_to_node/sum_aggregation/forward.forward")
    spec, edges = sum_forward_validate(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout,
    )
    output = messages[0].new_zeros((nodes, spec.full_components, spec.channels))
    for order, message in enumerate(messages):
        scatter_order(
            message, envelope, wigner,
            receivers, output,
            spec, order, edges, wigner_layout,
        )
    return output


def initial_forward(
    messages,
    envelope,
    wigner,
    receivers,
    atom_weight,
    atomic_numbers,
    nodes,
    *,
    lmax, mmax,
    wigner_layout,
):
    allow_atomic_reduction(
        "dataflow/edge_to_node/sum_aggregation/forward.initial_forward"
    )
    spec, edges = sum_forward_validate(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout,
    )
    if tuple(atom_weight.shape[1:]) != (spec.channels,):
        raise ValueError("initial atom weight does not match sector channels")
    if tuple(atomic_numbers.shape) != (nodes,):
        raise ValueError("initial atomic-number shape does not match nodes")
    output = messages[0].new_empty((nodes, spec.full_components, spec.channels))
    elements = output.numel()
    block = 256
    initial_atom_kernel[(triton.cdiv(elements, block),)](
        output,
        atom_weight,
        atomic_numbers,
        elements,
        spec.full_components * spec.channels,
        channels=spec.channels,
        block=block,
        num_warps=4,
    )
    for order, message in enumerate(messages):
        scatter_order(
            message, envelope, wigner,
            receivers, output,
            spec, order, edges, wigner_layout,
        )
    return output


@triton.jit
def receiver_forward(
    x, gate, rotation, row_ptr, t0, t1, t2, scale,
    order: tl.constexpr, input_width: tl.constexpr, rotation_width: tl.constexpr,
    positive_offsets: tl.constexpr, negative_offsets: tl.constexpr, degrees: tl.constexpr,
    angular_widths: tl.constexpr, angular_indices: tl.constexpr,
    packed_widths: tl.constexpr, packed_offset: tl.constexpr,
    edge_tile: tl.constexpr, channel_tile: tl.constexpr,
):
    terms = (t0, t1, t2)
    node = tl.program_id(0)
    tile = tl.program_id(1)
    channel = tile * channel_tile + tl.arange(0, channel_tile)
    start = tl.load(row_ptr + node).to(tl.int32)
    stop = tl.load(row_ptr + node + 1).to(tl.int32)
    offsets = tl.arange(0, edge_tile)
    slots: tl.constexpr = len(angular_indices)
    real = ()
    imag = ()
    bias = ()
    for _ in tl.static_range(slots):
        real += (tl.zeros((edge_tile, channel_tile), tl.float32),)
        if order:
            imag += (tl.zeros((edge_tile, channel_tile), tl.float32),)
        else:
            bias += (tl.zeros((edge_tile,), tl.float32),)
    for base in range(start, stop, edge_tile):
        edge = base + offsets
        live = edge < stop
        valid = live[:, None] & (channel[None, :] < input_width)
        planes: tl.constexpr = 2 if order else 1
        degree = channel // 128 + order
        gate_offset = tl.maximum(degree - 1, 0) * 128 + channel % 128
        logits = tl.load(gate + edge[:, None] * 640 + gate_offset[None, :], valid, 0.)
        sigmoid = 1.0 / (1.0 + tl.exp(-logits))
        if order:
            addr = edge[:, None] * (planes * input_width) + channel[None, :]
            positive = tl.load(x + addr, valid, 0.) * sigmoid
            negative = tl.load(x + addr + input_width, valid, 0.) * sigmoid
        else:
            addr = edge[:, None] * 640 + 256 + channel[None, :]
            positive = tl.load(x + addr, valid, 0.)
            silu = positive / (1.0 + tl.exp(-positive))
            positive = tl.where(degree[None, :] == 0, silu, positive * sigmoid)
        next_real = ()
        next_imag = ()
        next_bias = ()
        for i in tl.static_range(slots):
            a = tl.load(rotation + edge * rotation_width + positive_offsets[i], live, 0.)
            if order:
                b = tl.load(rotation + edge * rotation_width + negative_offsets[i], live, 0.)
                next_real += (real[i] + a[:, None] * positive + b[:, None] * negative,)
                next_imag += (imag[i] + b[:, None] * positive - a[:, None] * negative,)
            else:
                next_real += (real[i] + a[:, None] * positive,)
                next_bias += (bias[i] + a,)
        real = next_real
        if order:
            imag = next_imag
        else:
            bias = next_bias
    for i in tl.static_range(slots):
        addr = (terms[degrees[i]] + (node * angular_widths[i] + angular_indices[i]) * packed_widths[i]
                + packed_offset + channel)
        tl.store(addr, tl.sum(real[i], axis=0), channel < input_width)
        if order:
            tl.store(addr + input_width, tl.sum(imag[i], axis=0), channel < input_width)
        elif tile == 0:
            tl.store(scale + node * slots + i, tl.sum(bias[i], axis=0))


@triton.jit
def initial_terms(degree: tl.constexpr, rows: tl.constexpr, channels: tl.constexpr):
    result = ()
    for order in tl.static_range(degree + 1):
        values = ()
        for _ in tl.static_range(9 - order * order):
            values += ((tl.zeros((rows, channels), tl.float32), tl.zeros((rows, channels), tl.float32)),)
        result += (values,)
    return result

@triton.jit
def degree_gate_forward(x0, x1, x2, rotation, row_ptr, terms, scale,
                        specs: tl.constexpr, degree: tl.constexpr,
                        edge_tile: tl.constexpr, channel_tile: tl.constexpr):
    node = tl.program_id(0)
    tile = tl.program_id(1)
    channel = tile * channel_tile + tl.arange(0, channel_tile)
    start = tl.load(row_ptr + node).to(tl.int32)
    stop = tl.load(row_ptr + node + 1).to(tl.int32)
    offset = tl.arange(0, edge_tile)
    sums = initial_terms(degree, edge_tile, channel_tile)
    bias = ()
    if degree == 0:
        for _ in tl.static_range(9):
            bias += (tl.zeros((edge_tile,), tl.float32),)
    for base in range(start, stop, edge_tile):
        edge = base + offset
        live = edge < stop
        if degree:
            z = tl.load(x0 + edge[:, None] * 640 + (degree - 1) * 128 + channel[None, :],
                        live[:, None], other=0.)
            gate = 1.0 / (1.0 + tl.exp(-z))
        else:
            gate = tl.zeros((edge_tile, channel_tile), tl.float32)
            next_bias = ()
            for i in tl.static_range(9):
                value = tl.load(rotation + edge * 35 + specs.value[0][1][i], live, other=0.)
                next_bias += (bias[i] + value,)
            bias = next_bias
        next_sums = ()
        for order in tl.static_range(degree + 1):
            prev = sums[order]
            width = (3 - order) * 128
            col = (degree - order) * 128 + channel
            if order == 0:
                positive = tl.load(x0 + edge[:, None] * 640 + 256 + col[None, :], live[:, None], other=0.)
                if degree == 0:
                    positive = positive / (1.0 + tl.exp(-positive))
                else:
                    positive = positive * gate
            else:
                addr = edge[:, None] * (2 * width) + col[None, :]
                if order == 1:
                    positive = tl.load(x1 + addr, live[:, None], other=0.) * gate
                    negative = tl.load(x1 + addr + width, live[:, None], other=0.) * gate
                else:
                    positive = tl.load(x2 + addr, live[:, None], other=0.) * gate
                    negative = tl.load(x2 + addr + width, live[:, None], other=0.) * gate
            result = ()
            for i in tl.static_range(9 - order * order):
                a = tl.load(rotation + edge * 35 + specs.value[order][1][i], live, other=0.)
                real = prev[i][0] + a[:, None] * positive
                imag = prev[i][1]
                if order:
                    b = tl.load(rotation + edge * 35 + specs.value[order][2][i], live, other=0.)
                    real += b[:, None] * negative
                    imag = imag + b[:, None] * positive - a[:, None] * negative
                result += ((real, imag),)
            next_sums += (result,)
        sums = next_sums
    for order in tl.static_range(degree + 1):
        width = (3 - order) * 128
        col = (degree - order) * 128 + channel + specs.value[order][7]
        for i in tl.static_range(9 - order * order):
            angular_row = node * specs.value[order][4][i] + specs.value[order][5][i]
            ptr = terms[specs.value[order][3][i]] + angular_row * specs.value[order][6][i] + col
            tl.store(ptr, tl.sum(sums[order][i][0], 0))
            if order:
                tl.store(ptr + width, tl.sum(sums[order][i][1], 0))
    if degree == 0:
        if tile == 0:
            for i in tl.static_range(9):
                tl.store(scale + node * 9 + i, tl.sum(bias[i], 0))

def degree_gated_receiver_forward(x0, x1, x2, rotation, row_ptr, schedule):
    layout = preprojection_layout()
    nodes = row_ptr.numel() - 1
    terms = tuple(rotation.new_empty((nodes, angular, width))
                  for angular, width in zip(layout.angular_widths, layout.packed_widths, strict=True))
    scale = rotation.new_empty((nodes, 9))
    specs = tuple(tuple(spec) for spec in layout.orders)
    for degree, (edge_tile, channel_tile, warps) in enumerate(schedule):
        degree_gate_forward[(nodes, 128 // channel_tile)](
            x0, x1, x2, rotation, row_ptr, terms, scale, specs, degree,
            edge_tile, channel_tile, num_warps=warps, num_stages=1,
        )
    return *terms, scale


@triton.jit
def forward_degree(radial, rotation, edge_rows, output, scale, node, begin, end, c, slots,
                   degree: tl.constexpr, tile_edge: tl.constexpr, tile_channel: tl.constexpr):
    width: tl.constexpr = 2 * degree + 1
    offset: tl.constexpr = (0, 1, 10)[degree]
    sums = ()
    for k in tl.static_range(width):
        sums += (tl.full((tile_edge, tile_channel), 0., tl.float32),)
    for block in range(begin, end, tile_edge):
        edge = block + slots
        live = edge < end
        row = tl.load(edge_rows + edge, live, 0)
        value = tl.load(radial + (row[:, None] * 3 + degree) * 128 + c[None, :], live[:, None], 0.)
        partials = ()
        for k in tl.static_range(width):
            w = tl.load(rotation + edge * 35 + offset + k, live, 0.)
            partials += (sums[k] + (scale * w[:, None]) * value,)
        sums = partials
    for k in tl.static_range(width):
        tl.store(output + (node * 9 + degree * degree + k) * 128 + c, tl.sum(sums[k], 0))

@triton.jit
def paired_initial_forward(radial, rotation, edge_rows, row_ptr, output, scale,
                    tile_edge: tl.constexpr, tile_channel: tl.constexpr):
    # Inductor can pass the scale as FP64. Keep the accumulation in FP32.
    scale = scale.to(tl.float32)
    node = tl.program_id(0)
    c = tl.program_id(1) * tile_channel + tl.arange(0, tile_channel)
    begin = tl.load(row_ptr + node)
    end = tl.load(row_ptr + node + 1)
    slots = tl.arange(0, tile_edge)
    for degree in tl.static_range(3):
        forward_degree(radial, rotation, edge_rows, output, scale, node, begin, end, c, slots, degree, tile_edge, tile_channel)


SM90_RECEIVER_DISPATCH = r"""#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

extern "C" int launch_csr_receiver(
    const float*, const float*, const float*, const float*, const int*,
    float*, float*, float*, float*, int, int, cudaStream_t
);

namespace {

void receiver_out(
    const at::Tensor& x0, const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& rotation, const at::Tensor& row_ptr,
    at::Tensor& t0, at::Tensor& t1, at::Tensor& t2, at::Tensor& scale
) {
    const auto device = x0.device();
    const auto edges = x0.size(0);
    const auto nodes = row_ptr.numel() - 1;
    TORCH_CHECK(device.is_cuda() && nodes >= 0 && row_ptr.dim() == 1
                && row_ptr.device() == device && row_ptr.is_contiguous()
                && row_ptr.scalar_type() == at::kInt, "Invalid CUDA CSR row pointer");
    auto checked = [&](const at::Tensor& value, at::IntArrayRef shape) {
        TORCH_CHECK(value.sizes() == shape && value.device() == device
                    && value.is_contiguous() && value.scalar_type() == at::kFloat,
                    "Invalid CUDA CSR receiver tensor");
    };
    checked(x0, {edges, 640});
    checked(x1, {edges, 512});
    checked(x2, {edges, 256});
    checked(rotation, {edges, 35});
    checked(t0, {nodes, 1, 384});
    checked(t1, {nodes, 3, 896});
    checked(t2, {nodes, 5, 1152});
    checked(scale, {nodes, 9});
    const c10::cuda::CUDAGuard guard(device);
    const auto code = launch_csr_receiver(
        x0.const_data_ptr<float>(), x1.const_data_ptr<float>(), x2.const_data_ptr<float>(),
        rotation.const_data_ptr<float>(), row_ptr.const_data_ptr<int>(),
        t0.mutable_data_ptr<float>(), t1.mutable_data_ptr<float>(), t2.mutable_data_ptr<float>(),
        scale.mutable_data_ptr<float>(), nodes, 2, at::cuda::getCurrentCUDAStream(device.index())
    );
    TORCH_CHECK(code == 0, "Native CSR receiver failed with CUDA status ", code);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trim_h100, m) {
    m.def("csr_receiver_out(Tensor x0, Tensor x1, Tensor x2, Tensor rotation, Tensor row_ptr, "
          "Tensor(a!) t0, Tensor(b!) t1, Tensor(c!) t2, Tensor(d!) scale) -> ()");
}

TORCH_LIBRARY_IMPL(trim_h100, CUDA, m) {
    m.impl("csr_receiver_out", receiver_out);
}
"""


SM90_RECEIVER_FORWARD = r"""#include <cuda_runtime.h>

namespace {

constexpr int channels = 128;
constexpr int order_offset[] = {0, 384, 896};

__device__ __forceinline__ float coefficient(float low, float high, int index) {
    return __shfl_sync(0xffffffffu, index < 32 ? low : high, index & 31);
}

template<int Degree, int Order>
__device__ __forceinline__ void accumulate(
    float* sum, const float* raw, int edge, int channel, float gate, float low, float high
) {
    constexpr int width = (3 - Order) * channels;
    constexpr int start = Order == 0 ? 0 : (Order == 1 ? 9 : 25);
    constexpr int values = 9 - Order * Order;
    const int column = (Degree - Order) * channels + channel;
    float positive, negative = 0.f;
    if constexpr (Order == 0) {
        const float input = raw[edge * 640 + 256 + column];
        if constexpr (Degree == 0)
            positive = input * __fdividef(1.f, 1.f + __expf(-input));
        else
            positive = input * gate;
    } else {
        const int address = edge * (2 * width) + column;
        positive = raw[address] * gate;
        negative = raw[address + width] * gate;
    }

    #pragma unroll
    for (int l = Order; l <= 2; ++l) {
        const int angular = 2 * l + 1;
        const int rotation = l == 0 ? 0 : (l == 1 ? 1 : 10);
        #pragma unroll
        for (int j = 0; j < angular; ++j) {
            const int i = l * l - Order * Order + j;
            const int row = rotation + j * angular + l + Order;
            const float a = coefficient(low, high, row);
            sum[start + i] = fmaf(a, positive, sum[start + i]);
            if constexpr (Order > 0) {
                const float b = coefficient(low, high, row - 2 * Order);
                sum[start + i] = fmaf(b, negative, sum[start + i]);
                sum[start + values + i] = fmaf(b, positive, sum[start + values + i]);
                sum[start + values + i] = fmaf(-a, negative, sum[start + values + i]);
            }
        }
    }
}

template<int Warps>
__device__ __forceinline__ float total(const float* shared, int i, int lane) {
    float value = 0.f;
    #pragma unroll
    for (int w = 0; w < Warps; ++w)
        value += shared[(i * Warps + w) * 32 + lane];
    return value;
}

template<int Degree, int Order, int Warps>
__device__ __forceinline__ void finish(
    float* const* output, const float* shared, int node, int lane, int channel
) {
    constexpr int width = (3 - Order) * channels;
    constexpr int start = Order == 0 ? 0 : (Order == 1 ? 9 : 25);
    constexpr int values = 9 - Order * Order;
    const int column = order_offset[Order] + (Degree - Order) * channels + channel;
    #pragma unroll
    for (int l = Order; l <= 2; ++l) {
        const int angular = 2 * l + 1;
        #pragma unroll
        for (int j = 0; j < angular; ++j) {
            const int i = l * l - Order * Order + j;
            float* address = output[l] + (node * angular + j) * (l == 0 ? 384 : (l == 1 ? 896 : 1152)) + column;
            *address = total<Warps>(shared, start + i, lane);
            if constexpr (Order > 0)
                address[width] = total<Warps>(shared, start + values + i, lane);
        }
    }
}

template<int Degree, int Warps>
__device__ __forceinline__ void reduce(
    const float* x0, const float* x1, const float* x2, const float* rotation,
    const int* ptr, float* t0, float* t1, float* t2, float* scale, float* shared
) {
    constexpr int terms = Degree == 0 ? 9 : (Degree == 1 ? 25 : 35);
    const int node = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int channel = blockIdx.y * 32 + lane;
    const int begin = ptr[node], end = ptr[node + 1];
    float sums[terms] = {};
    float bias = 0.f;
    for (int edge = begin + warp; edge < end; edge += Warps) {
        const float* r = rotation + edge * 35;
        const float low = r[lane];
        const float high = lane < 3 ? r[lane + 32] : 0.f;
        float gate = 0.f;
        if constexpr (Degree > 0) {
            const float logit = x0[edge * 640 + (Degree - 1) * 128 + channel];
            gate = __fdividef(1.f, 1.f + __expf(-logit));
        }
        accumulate<Degree, 0>(sums, x0, edge, channel, gate, low, high);
        if constexpr (Degree > 0)
            accumulate<Degree, 1>(sums, x1, edge, channel, gate, low, high);
        if constexpr (Degree > 1)
            accumulate<Degree, 2>(sums, x2, edge, channel, gate, low, high);
        if constexpr (Degree == 0) {
            if (blockIdx.y == 0) {
                const int index = lane == 0 ? 0 : (lane < 4 ? 2 + 3 * (lane - 1) : 12 + 5 * (lane - 4));
                if (lane < 9) bias += r[index];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < terms; ++i)
        shared[(i * Warps + warp) * 32 + lane] = sums[i];
    if constexpr (Degree == 0)
        if (blockIdx.y == 0 && lane < 9)
            shared[(terms * Warps + warp) * 32 + lane] = bias;
    __syncthreads();
    if (warp != 0) return;
    float* output[3] = {t0, t1, t2};
    finish<Degree, 0, Warps>(output, shared, node, lane, channel);
    if constexpr (Degree > 0) finish<Degree, 1, Warps>(output, shared, node, lane, channel);
    if constexpr (Degree > 1) finish<Degree, 2, Warps>(output, shared, node, lane, channel);
    if constexpr (Degree == 0)
        if (blockIdx.y == 0 && lane < 9)
            scale[node * 9 + lane] = total<Warps>(shared, terms, lane);
}

template<int Warps>
__global__ void receiver(
    const float* x0, const float* x1, const float* x2, const float* rotation,
    const int* ptr, float* t0, float* t1, float* t2, float* scale
) {
    __shared__ float shared[35 * Warps * 32];
    if (blockIdx.z == 0) reduce<0, Warps>(x0, x1, x2, rotation, ptr, t0, t1, t2, scale, shared);
    if (blockIdx.z == 1) reduce<1, Warps>(x0, x1, x2, rotation, ptr, t0, t1, t2, scale, shared);
    if (blockIdx.z == 2) reduce<2, Warps>(x0, x1, x2, rotation, ptr, t0, t1, t2, scale, shared);
}

} // namespace

extern "C" int launch_csr_receiver(
    const float* x0, const float* x1, const float* x2, const float* rotation,
    const int* ptr, float* t0, float* t1, float* t2, float* scale,
    int nodes, int warps, cudaStream_t stream
) {
    if (nodes == 0) return cudaSuccess;
    const dim3 grid(nodes, 4, 3);
    #define LAUNCH(W) receiver<W><<<grid, W * 32, 0, stream>>>(x0, x1, x2, rotation, ptr, t0, t1, t2, scale)
    switch (warps) {
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
        default: return cudaErrorInvalidValue;
    }
    #undef LAUNCH
    return cudaPeekAtLastError();
}
"""


def configure_receiver_forward_sm90():
    load_dispatch(
        "csr_receiver_dispatch",
        (("sm90_receiver_dispatch.cpp", SM90_RECEIVER_DISPATCH), ("sm90_receiver.cu", SM90_RECEIVER_FORWARD)),
        arch="90",
    )

    @torch.library.register_fake("trim_h100::csr_receiver_out")
    def receiver_out_fake(x0, x1, x2, rotation, row_ptr, t0, t1, t2, scale):
        return None
