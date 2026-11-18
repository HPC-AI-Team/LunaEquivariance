from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ...utils.layout import get_retained_layout
from .backward import launch_prefix_wigner_tiles, prefix_input_csr_degree_kernel
from .utils import (
    add_node_direction,
    gl,
    gluon,
    launch_degree_shared,
    channel_launch,
    generated_compact_row_offset,
    input_channel_warps,
    load_prefix_cotangent,
    load_sector_cotangent,
    prefix_wigner_address,
    reduced_row_for_degree,
    sector_metadata,
    store_prefix_direction,
    store_sector_direction,
    use_prefix_degree_tiles,
    validate_inputs,
)


@triton.jit
def structured_double_backward_degree(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr,
    input_direction_ptr,
    input_direction_stride_node,
    input_direction_stride_component,
    input_direction_stride_channel,
    compact_radial_ptr,
    radial_direction_ptr,
    radial_direction_stride_edge,
    radial_direction_stride_degree,
    radial_direction_stride_channel,
    wigner_ptr,
    wigner_stride_edge, wigner_stride_row, wigner_stride_component,
    wigner_direction_ptr,
    wigner_direction_stride_edge,
    wigner_direction_stride_row,
    wigner_direction_stride_component,
    grad_outputs,
    input_gradient_ptr,
    radial_gradient_ptr,
    edge,
    node,
    branch,
    channel_offsets,
    channel_mask,
    edges,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    degree_count: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    degree: tl.constexpr,
    component_start: tl.constexpr,
    component_width: tl.constexpr,
    row_count: tl.constexpr,
    sector_layout: tl.constexpr,
):
    branch_channels = branch * channels + channel_offsets
    radial = tl.load(
        compact_radial_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + branch_channels,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    radial_direction = tl.load(
        radial_direction_ptr
        + edge * radial_direction_stride_edge
        + degree * radial_direction_stride_degree
        + branch_channels * radial_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)

    radial_accumulator = tl.zeros(channel_offsets.shape, dtype=tl.float32)
    for local_component in tl.static_range(0, component_width):
        component = component_start + local_component
        feature = tl.load(
            x_ptr
            + node * full_components * channels
            + component * channels
            + channel_offsets,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        feature_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + component * input_direction_stride_component
            + channel_offsets * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        transposed_projection = tl.zeros(channel_offsets.shape, dtype=tl.float32)
        direction_transposed_projection = tl.zeros(channel_offsets.shape, dtype=tl.float32)
        for local_row in tl.static_range(0, row_count):
            row = reduced_row_for_degree(degree, local_row, lmax)
            cotangent = load_prefix_cotangent(
                grads,
                grad_stride_edge, grad_stride_row, grad_stride_channel,
                edge,
                row,
                branch_channels,
                channel_mask,
                channels,
                sector_layout,
                lmax,
            )
            coefficient = tl.load(
                wigner_ptr
                + edge * wigner_stride_edge
                + generated_compact_row_offset(
                    row, compact_row_offsets, reduced_components
                )
                + local_component
            ).to(tl.float32)
            coefficient_direction = tl.load(
                wigner_direction_ptr
                + edge * wigner_direction_stride_edge
                + generated_compact_row_offset(
                    row, compact_row_offsets, reduced_components
                )
                + local_component
            ).to(tl.float32)
            transposed_projection += coefficient * cotangent
            direction_transposed_projection += coefficient_direction * cotangent
        input_gradient = (
            radial_direction * transposed_projection
            + radial * direction_transposed_projection
        )
        tl.atomic_add(
            input_gradient_ptr
            + node * full_components * channels
            + component * channels
            + channel_offsets,
            input_gradient,
            mask=channel_mask,
        )
        radial_accumulator += (
            feature_direction * transposed_projection
            + feature * direction_transposed_projection
        )

    tl.store(
        radial_gradient_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + branch_channels,
        radial_accumulator,
        mask=channel_mask,
    )

    for local_row in tl.static_range(0, row_count):
        row = reduced_row_for_degree(degree, local_row, lmax)
        output_accumulator = tl.zeros(channel_offsets.shape, dtype=tl.float32)
        for local_component in tl.static_range(0, component_width):
            component = component_start + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel_offsets,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            feature_direction = tl.load(
                input_direction_ptr
                + node * input_direction_stride_node
                + component * input_direction_stride_component
                + channel_offsets * input_direction_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.load(
                wigner_ptr
                + edge * wigner_stride_edge
                + generated_compact_row_offset(
                    row, compact_row_offsets, reduced_components
                )
                + local_component
            ).to(tl.float32)
            coefficient_direction = tl.load(
                wigner_direction_ptr
                + edge * wigner_direction_stride_edge
                + generated_compact_row_offset(
                    row, compact_row_offsets, reduced_components
                )
                + local_component
            ).to(tl.float32)
            message = radial * feature
            message_direction = radial * feature_direction + radial_direction * feature
            output_accumulator += (
                coefficient * message_direction + coefficient_direction * message
            )
        store_prefix_direction(
            grad_outputs,
            edge,
            row,
            branch_channels,
            output_accumulator,
            channel_mask,
            channels,
            sector_layout,
            reduced_components,
            lmax,
        )


@triton.jit
def structured_prefix_double_backward_kernel(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr, edge_index_ptr, compact_radial_ptr, wigner_ptr,
    wigner_stride_edge, wigner_stride_row, wigner_stride_component,
    input_direction_ptr,
    input_direction_stride_node,
    input_direction_stride_component,
    input_direction_stride_channel,
    radial_direction_ptr,
    radial_direction_stride_edge,
    radial_direction_stride_degree,
    radial_direction_stride_channel,
    wigner_direction_ptr,
    wigner_direction_stride_edge,
    wigner_direction_stride_row,
    wigner_direction_stride_component,
    grad_outputs,
    input_gradient_ptr,
    radial_gradient_ptr,
    edges,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    degree_count: tl.constexpr,
    degree_row_counts: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    sector_layout: tl.constexpr,
):
    edge = tl.program_id(0)
    channel_tile = tl.program_id(1)
    tiles_per_branch: tl.constexpr = tl.cdiv(channels, block_channels)
    branch = channel_tile // tiles_per_branch
    tile_in_branch = channel_tile - branch * tiles_per_branch
    channel_offsets = tile_in_branch * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)

    for degree in tl.static_range(0, degree_count):
        structured_double_backward_degree(
            grads,
            grad_stride_edge, grad_stride_row, grad_stride_channel,
            x_ptr,
            input_direction_ptr,
            input_direction_stride_node,
            input_direction_stride_component,
            input_direction_stride_channel,
            compact_radial_ptr,
            radial_direction_ptr,
            radial_direction_stride_edge,
            radial_direction_stride_degree,
            radial_direction_stride_channel,
            wigner_ptr,
            wigner_stride_edge, wigner_stride_row, wigner_stride_component,
            wigner_direction_ptr,
            wigner_direction_stride_edge,
            wigner_direction_stride_row,
            wigner_direction_stride_component,
            grad_outputs,
            input_gradient_ptr,
            radial_gradient_ptr,
            edge,
            node,
            branch,
            channel_offsets,
            channel_mask,
            edges,
            channels,
            lmax,
            full_components,
            reduced_components,
            degree_count,
            compact_row_offsets,
            degree,
            degree * degree,
            2 * degree + 1,
            tl.constexpr(list(degree_row_counts)[degree]),
            sector_layout,
        )


@triton.jit
def structured_prefix_double_backward_wigner_gradient_kernel(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr,
    edge_index_ptr,
    compact_radial_ptr,
    input_direction_ptr,
    input_direction_stride_node,
    input_direction_stride_component,
    input_direction_stride_channel,
    radial_direction_ptr,
    radial_direction_stride_edge,
    radial_direction_stride_degree,
    radial_direction_stride_channel,
    wigner_gradient_ptr,
    edges,
    channels: tl.constexpr,
    channel_block: tl.constexpr,
    lmax: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    degree_count: tl.constexpr,
    degree: tl.constexpr,
    component_start: tl.constexpr,
    component_width: tl.constexpr,
    component_block: tl.constexpr,
    sector_layout: tl.constexpr,
):
    edge = tl.program_id(0)
    local_row = tl.program_id(1)
    row = reduced_row_for_degree(degree, local_row, lmax)
    component_offsets_1d = tl.arange(0, component_block)
    component_offsets = component_offsets_1d[:, None]
    channel_offsets = tl.arange(0, channel_block)[None, :]
    channel_mask = channel_offsets < channels
    component_mask_1d = component_offsets_1d < component_width
    component_mask = component_mask_1d[:, None]
    value_mask = component_mask & channel_mask
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)

    source_cotangent = load_prefix_cotangent(
        grads,
        grad_stride_edge, grad_stride_row, grad_stride_channel,
        edge,
        row,
        channel_offsets,
        channel_mask,
        channels,
        sector_layout,
        lmax,
    )
    target_cotangent = load_prefix_cotangent(
        grads,
        grad_stride_edge, grad_stride_row, grad_stride_channel,
        edge,
        row,
        channels + channel_offsets,
        channel_mask,
        channels,
        sector_layout,
        lmax,
    )
    source_radial = tl.load(
        compact_radial_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    target_radial = tl.load(
        compact_radial_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + channels
        + channel_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    source_radial_direction = tl.load(
        radial_direction_ptr
        + edge * radial_direction_stride_edge
        + degree * radial_direction_stride_degree
        + channel_offsets * radial_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    target_radial_direction = tl.load(
        radial_direction_ptr
        + edge * radial_direction_stride_edge
        + degree * radial_direction_stride_degree
        + (channels + channel_offsets) * radial_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    components = component_start + component_offsets
    source = tl.load(
        x_ptr
        + source_node * full_components * channels
        + components * channels
        + channel_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    target = tl.load(
        x_ptr
        + target_node * full_components * channels
        + components * channels
        + channel_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    source_direction = tl.load(
        input_direction_ptr
        + source_node * input_direction_stride_node
        + components * input_direction_stride_component
        + channel_offsets * input_direction_stride_channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    target_direction = tl.load(
        input_direction_ptr
        + target_node * input_direction_stride_node
        + components * input_direction_stride_component
        + channel_offsets * input_direction_stride_channel,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    source_message_direction = (
        source_radial * source_direction + source_radial_direction * source
    )
    target_message_direction = (
        target_radial * target_direction + target_radial_direction * target
    )
    gradient = tl.sum(
        source_cotangent * source_message_direction
        + target_cotangent * target_message_direction,
        axis=1,
    )
    tl.store(
        wigner_gradient_ptr
        + edge * compact_elements
        + generated_compact_row_offset(row, compact_row_offsets, reduced_components)
        + component_offsets_1d,
        gradient,
        mask=component_mask_1d,
    )


@triton.jit
def prefix_wigner_degree_double_backward_kernel(
    grads,
    x_ptr,
    edge_index_ptr,
    radial_ptr,
    input_direction_ptr,
    radial_direction_ptr,
    wigner_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_elements: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    degree: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    component_block: tl.constexpr,
    channel_block: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    edge = tl.program_id(0)
    slot = tl.program_id(1)
    is_scalar = slot == 0
    order = tl.where(is_scalar, 0, (slot + 1) // 2)
    sign = (slot - 1) & 1
    width = lmax - order + 1
    sector_start = lmax + 1 + (order - 1) * (2 * (lmax + 1) - order)
    sector_row = tl.where(is_scalar, degree, sector_start + degree - order + sign * width)
    dense_center = degree * degree + degree
    dense_row = tl.where(is_scalar, dense_center, dense_center + tl.where(sign == 0, order, -order))
    radial_start = (
        ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
        * 2
        * channels
    )
    radial_row_offset = tl.where(
        is_scalar,
        degree * 2 * channels,
        radial_start + (degree - order) * 2 * channels,
    )

    local_component_1d = tl.arange(0, component_block)
    local_component = local_component_1d[:, None]
    component_mask_1d = local_component_1d < 2 * degree + 1
    component_mask = component_mask_1d[:, None]
    channel = tl.arange(0, channel_block)[None, :]
    channel_mask = channel < channels
    value_mask = component_mask & channel_mask
    component = degree * degree + local_component
    source_node = tl.load(edge_index_ptr + edge)
    target_node = tl.load(edge_index_ptr + edges + edge)
    source_address = (
        source_node * full_components * channels + component * channels + channel
    )
    target_address = (
        target_node * full_components * channels + component * channels + channel
    )
    source = tl.load(x_ptr + source_address, mask=value_mask, other=0.0).to(tl.float32)
    target = tl.load(x_ptr + target_address, mask=value_mask, other=0.0).to(tl.float32)
    source_direction = tl.load(input_direction_ptr + source_address, mask=value_mask, other=0.0).to(tl.float32)
    target_direction = tl.load(input_direction_ptr + target_address, mask=value_mask, other=0.0).to(tl.float32)
    radial_base = edge * radial_elements + radial_row_offset
    source_radial = tl.load(radial_ptr + radial_base + channel, mask=channel_mask, other=0.0).to(tl.float32)
    target_radial = tl.load(radial_ptr + radial_base + channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
    source_radial_direction = tl.load(
        radial_direction_ptr + radial_base + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    target_radial_direction = tl.load(
        radial_direction_ptr + radial_base + channels + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    source_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    target_cotangent = load_sector_cotangent(
        grads,
        edge,
        sector_row,
        channels + channel,
        channel_mask,
        mmax,
        offsets,
        elements,
        channels,
    )
    gradient = tl.sum(
        source_cotangent
        * (source_radial * source_direction + source_radial_direction * source)
        + target_cotangent
        * (target_radial * target_direction + target_radial_direction * target),
        axis=1,
    )
    tl.store(
        wigner_gradient_ptr
        + prefix_wigner_address(
            edge,
            dense_row,
            degree * degree + local_component_1d,
            full_components,
            compact_elements,
            compact_wigner,
            retained_mmax,
        ),
        gradient,
        mask=component_mask_1d,
    )


@triton.jit
def prefix_edge_double_backward_kernel(
    grads,
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    input_direction_ptr,
    radial_direction_ptr,
    wigner_direction_ptr,
    grad_gradients,
    radial_gradient_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    radial_group_rows0: tl.constexpr,
    radial_group_rows1: tl.constexpr,
    radial_group_dense_rows0: tl.constexpr,
    radial_group_dense_rows1: tl.constexpr,
    radial_group_degrees: tl.constexpr,
    radial_group_offsets: tl.constexpr,
    radial_group_count: tl.constexpr,
    radial_elements: tl.constexpr,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
    write_sectors: tl.constexpr = True,
    write_radial: tl.constexpr = True,
):
    edge = tl.program_id(0)
    branch = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    output_channel = branch * channels + channel
    for group in tl.static_range(0, radial_group_count):
        row0 = list(radial_group_rows0)[group]
        row1 = list(radial_group_rows1)[group]
        degree = list(radial_group_degrees)[group]
        radial_address = (
            edge * radial_elements
            + tl.constexpr(list(radial_group_offsets)[group])
            + output_channel
        )
        radial = tl.load(radial_ptr + radial_address, mask=channel_mask, other=0.0).to(tl.float32)
        radial_direction = tl.load(radial_direction_ptr + radial_address, mask=channel_mask, other=0.0).to(tl.float32)
        cotangent0 = load_sector_cotangent(
            grads,
            edge,
            row0,
            output_channel,
            channel_mask,
            mmax,
            offsets,
            elements,
            channels,
        )
        cotangent1 = tl.zeros(channel.shape, dtype=tl.float32)
        if row1 >= 0:
            cotangent1 = load_sector_cotangent(
                grads,
                edge,
                row1,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
        projected0 = tl.zeros(channel.shape, dtype=tl.float32)
        projected1 = tl.zeros(channel.shape, dtype=tl.float32)
        projected_direction0 = tl.zeros(channel.shape, dtype=tl.float32)
        projected_direction1 = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(
            0, 2 * list(radial_group_degrees)[group] + 1
        ):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            feature_direction = tl.load(
                input_direction_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0_address = prefix_wigner_address(
                edge,
                tl.constexpr(list(radial_group_dense_rows0)[group]),
                component,
                full_components,
                compact_elements,
                compact_wigner,
                retained_mmax,
            )
            coefficient0 = tl.load(wigner_ptr + coefficient0_address).to(tl.float32)
            coefficient_direction0 = tl.load(wigner_direction_ptr + coefficient0_address).to(tl.float32)
            projected0 += coefficient0 * feature
            projected_direction0 += (
                coefficient0 * feature_direction + coefficient_direction0 * feature
            )
            if row1 >= 0:
                coefficient1_address = prefix_wigner_address(
                    edge,
                    tl.constexpr(list(radial_group_dense_rows1)[group]),
                    component,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                )
                coefficient1 = tl.load(wigner_ptr + coefficient1_address).to(tl.float32)
                coefficient_direction1 = tl.load(wigner_direction_ptr + coefficient1_address).to(tl.float32)
                projected1 += coefficient1 * feature
                projected_direction1 += (
                    coefficient1 * feature_direction + coefficient_direction1 * feature
                )
        if write_sectors:
            store_sector_direction(
                grad_gradients,
                edge,
                row0,
                output_channel,
                radial * projected_direction0 + radial_direction * projected0,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
            if row1 >= 0:
                store_sector_direction(
                    grad_gradients,
                    edge,
                    row1,
                    output_channel,
                    radial * projected_direction1 + radial_direction * projected1,
                    channel_mask,
                    mmax,
                    offsets,
                    elements,
                    channels,
                )
        if write_radial:
            tl.store(
                radial_gradient_ptr + radial_address,
                cotangent0 * projected_direction0 + cotangent1 * projected_direction1,
                mask=channel_mask,
            )


@triton.jit
def prefix_input_csr_double_backward_kernel(
    grads,
    radial_ptr,
    wigner_ptr,
    radial_direction_ptr,
    wigner_direction_ptr,
    source_order_ptr,
    source_row_ptr,
    receiver_row_ptr,
    input_gradient_ptr,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    radial_elements: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    node = tl.program_id(0)
    full = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    dense_center = degree * degree + degree
    accumulator = tl.zeros(channel.shape, dtype=tl.float32)
    for branch in tl.static_range(0, 2):
        if branch == 0:
            start = tl.load(source_row_ptr + node).to(tl.int32)
            end = tl.load(source_row_ptr + node + 1).to(tl.int32)
        else:
            start = tl.load(receiver_row_ptr + node).to(tl.int32)
            end = tl.load(receiver_row_ptr + node + 1).to(tl.int32)
        output_channel = branch * channels + channel
        for position in range(start, end):
            edge = (
                tl.load(source_order_ptr + position).to(tl.int32)
                if branch == 0
                else position
            )
            radial0_address = (
                edge * radial_elements + degree * 2 * channels + output_channel
            )
            radial0 = tl.load(radial_ptr + radial0_address, mask=channel_mask, other=0.0).to(tl.float32)
            radial0_direction = tl.load(
                radial_direction_ptr + radial0_address,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient0_address = prefix_wigner_address(
                edge,
                dense_center,
                full,
                full_components,
                compact_elements,
                compact_wigner,
                retained_mmax,
            )
            coefficient0 = tl.load(wigner_ptr + coefficient0_address).to(tl.float32)
            coefficient0_direction = tl.load(wigner_direction_ptr + coefficient0_address).to(tl.float32)
            cotangent0 = load_sector_cotangent(
                grads,
                edge,
                degree,
                output_channel,
                channel_mask,
                mmax,
                offsets,
                elements,
                channels,
            )
            local = cotangent0 * (
                radial0_direction * coefficient0 + radial0 * coefficient0_direction
            )

            for order in tl.static_range(1, mmax + 1):
                order_mask = channel_mask & (degree >= order)
                width = lmax - order + 1
                local_row = degree - order
                radial_start = (
                    ((lmax + 1) + (order - 1) * (lmax + 1) - ((order - 1) * order) // 2)
                    * 2
                    * channels
                )
                radial_address = (
                    edge * radial_elements
                    + radial_start
                    + local_row * 2 * channels
                    + output_channel
                )
                radial_value = tl.load(radial_ptr + radial_address, mask=order_mask, other=0.0).to(tl.float32)
                radial_value_direction = tl.load(
                    radial_direction_ptr + radial_address,
                    mask=order_mask,
                    other=0.0,
                ).to(tl.float32)
                for sign in tl.static_range(0, 2):
                    sector_row = local_row + sign * width
                    dense_row = dense_center + (order if sign == 0 else -order)
                    coefficient_address = prefix_wigner_address(
                        edge,
                        dense_row,
                        full,
                        full_components,
                        compact_elements,
                        compact_wigner,
                        retained_mmax,
                    )
                    coefficient = tl.load(wigner_ptr + coefficient_address, mask=degree >= order, other=0.0).to(tl.float32)
                    coefficient_direction = tl.load(
                        wigner_direction_ptr + coefficient_address,
                        mask=degree >= order,
                        other=0.0,
                    ).to(tl.float32)
                    cotangent = tl.load(
                        grads[order]
                        + edge * list(elements)[order]
                        + sector_row * 2 * channels
                        + output_channel,
                        mask=order_mask,
                        other=0.0,
                    ).to(tl.float32)
                    local += cotangent * (
                        radial_value_direction * coefficient
                        + radial_value * coefficient_direction
                    )
            accumulator += local
    tl.store(
        input_gradient_ptr
        + node * full_components * channels
        + full * channels
        + channel,
        accumulator,
        mask=channel_mask,
    )


@triton.jit
def prefix_edge_wigner_double_backward_kernel(
    seeds, x, indices, radial, wigner, hx, hr, hw, dseed, dradial, dwigner, edges,
    lmax: tl.constexpr, channels: tl.constexpr, degree: tl.constexpr,
    radial_stride: tl.constexpr, wigner_stride: tl.constexpr, wigner_start: tl.constexpr,
    radial_offsets: tl.constexpr, sector_elements: tl.constexpr,
):
    edge = tl.program_id(0)
    width: tl.constexpr = 2 * degree + 1
    channel = tl.arange(0, channels)
    column = tl.arange(0, triton.next_power_of_2(width))
    valid = column < width
    feature_offset = (degree * degree + column[:, None]) * channels + channel[None, :]
    node_stride: tl.constexpr = (lmax + 1) ** 2 * channels
    source = tl.load(indices + edge) * node_stride + feature_offset
    receiver = tl.load(indices + edges + edge) * node_stride + feature_offset
    xs = tl.load(x + source, valid[:, None], 0.)
    xr = tl.load(x + receiver, valid[:, None], 0.)
    hxs = tl.load(hx + source, valid[:, None], 0.)
    hxr = tl.load(hx + receiver, valid[:, None], 0.)

    for order in tl.static_range(len(radial_offsets)):
        radial_offset = edge * radial_stride + radial_offsets[order] + channel
        rs = tl.load(radial + radial_offset)
        rr = tl.load(radial + radial_offset + channels)
        hrs = tl.load(hr + radial_offset)
        hrr = tl.load(hr + radial_offset + channels)
        source_direction = rs[None, :] * hxs + hrs[None, :] * xs
        receiver_direction = rr[None, :] * hxr + hrr[None, :] * xr
        drs = tl.zeros((channels,), tl.float32)
        drr = tl.zeros((channels,), tl.float32)
        for sign in tl.static_range(1 if order == 0 else 2):
            row = degree - order + sign * (lmax + 1 - order)
            slot = 0 if order == 0 else 2 * order - 1 + sign
            sector = edge * sector_elements[order] + row * 2 * channels + channel
            rotation = edge * wigner_stride + wigner_start + slot * width + column
            w = tl.load(wigner + rotation, valid, 0.)[:, None]
            dw = tl.load(hw + rotation, valid, 0.)[:, None]
            gs = tl.load(seeds[order] + sector)
            gr = tl.load(seeds[order] + sector + channels)
            ds = tl.sum(w * hxs + dw * xs, axis=0)
            dr = tl.sum(w * hxr + dw * xr, axis=0)
            tl.store(dseed[order] + sector, rs * ds + hrs * tl.sum(w * xs, axis=0))
            tl.store(dseed[order] + sector + channels, rr * dr + hrr * tl.sum(w * xr, axis=0))
            drs += gs * ds
            drr += gr * dr
            rotation_gradient = tl.sum(
                gs[None, :] * source_direction + gr[None, :] * receiver_direction, axis=1,
            )
            tl.store(dwigner + rotation, rotation_gradient, valid)
        tl.store(dradial + radial_offset, drs)
        tl.store(dradial + radial_offset + channels, drr)


def use_edge_wigner_double_backward(x, lmax, mmax, edges, compact, values):
    return (
        compact and (lmax, mmax, x.shape[-1]) == (3, 2, 128)
        and 16384 <= edges <= 32768 and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(x.device) == (8, 0)
        and all(value.is_contiguous() and not value.is_neg() for value in values)
    )


def gather_double_backward_source_csr(
    grad_outputs,
    x, edge_index, radial, wigner,
    metadata,
    input_direction,
    radial_direction,
    wigner_direction,
    *,
    lmax,
    mmax,
    output_mask=(True, True, True, True),
):
    write_sectors, write_input, write_radial, write_wigner = output_mask
    block_channels, forward_warps = channel_launch(x.shape[2])
    spec, compact_wigner = validate_inputs(
        x, edge_index, radial, wigner,
        lmax=lmax, mmax=mmax,
        block_channels=block_channels,
        num_warps=forward_warps,
    )
    edges = edge_index.shape[1]
    offsets, sector_elements = sector_metadata(spec)
    sector_grads = tuple(grad_outputs)
    source_order, source_row_ptr, receiver_row_ptr = metadata
    fuse_edge_wigner = write_sectors and write_radial and write_wigner and use_edge_wigner_double_backward(
        x, lmax, mmax, edges, compact_wigner,
        (x, radial, wigner, input_direction, radial_direction, wigner_direction, *sector_grads),
    )
    output_directions = tuple(torch.empty_like(value) if write_sectors else None for value in sector_grads)
    input_gradient = torch.empty_like(x) if write_input else None
    radial_gradient = torch.empty_like(radial) if write_radial else None
    if fuse_edge_wigner:
        wigner_gradient = torch.empty_like(wigner)
        retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=spec.channels)
        for degree, rows in enumerate(retained.degree_rows):
            radial_offsets = tuple(
                spec.radial_row_offsets[rows[0 if order == 0 else 2 * order - 1]]
                for order in range(min(degree, mmax) + 1)
            )
            prefix_edge_wigner_double_backward_kernel[(edges,)](
                sector_grads, x, edge_index, radial, wigner,
                input_direction, radial_direction, wigner_direction,
                output_directions, radial_gradient, wigner_gradient, edges,
                lmax, spec.channels, degree, spec.radial_elements, wigner.stride(0),
                retained.compact_row_offsets[rows[0]], radial_offsets, sector_elements,
                num_warps=2, num_stages=1,
            )
    else:
        wigner_gradient = torch.zeros_like(wigner) if write_wigner else None
    if not fuse_edge_wigner and (write_sectors or write_radial):
        # The disabled output pointers are unused by the specialized kernel.
        prefix_edge_double_backward_kernel[(edges, 2)](
            sector_grads,
            x, edge_index, radial, wigner,
            input_direction,
            radial_direction,
            wigner_direction,
            output_directions if write_sectors else sector_grads,
            radial_gradient if write_radial else radial,
            edges,
            channels=spec.channels,
            full_components=spec.full_components,
            radial_group_rows0=spec.radial_group_rows0,
            radial_group_rows1=spec.radial_group_rows1,
            radial_group_dense_rows0=spec.radial_group_dense_rows0,
            radial_group_dense_rows1=spec.radial_group_dense_rows1,
            radial_group_degrees=spec.radial_group_degrees,
            radial_group_offsets=spec.radial_group_offsets,
            radial_group_count=len(spec.radial_group_rows0),
            radial_elements=spec.radial_elements,
            mmax=spec.mmax,
            offsets=offsets,
            elements=sector_elements,
            block_channels=block_channels,
            compact_elements=wigner.stride(0),
            compact_wigner=compact_wigner,
            retained_mmax=mmax if wigner.ndim == 2 else -1,
            write_sectors=write_sectors,
            write_radial=write_radial,
            num_warps=max(4, forward_warps),
        )
    input_launch = dict(
        channels=spec.channels,
        full_components=spec.full_components,
        lmax=spec.lmax,
        mmax=spec.mmax,
        radial_elements=spec.radial_elements,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
        compact_wigner=compact_wigner,
        retained_mmax=mmax if wigner.ndim == 2 else -1,
        num_warps=input_channel_warps(block_channels),
    )
    degree_tiles = use_prefix_degree_tiles(
        x, edges, spec, compact_wigner, sector_grads, directional=True,
    )
    if write_input:
        input_indices = source_order, source_row_ptr, receiver_row_ptr, input_gradient
        if degree_tiles:
            prefix_input_csr_degree_kernel[(x.shape[0], lmax + 1)](
                sector_grads, radial, wigner, *input_indices, **input_launch,
                radial_direction=radial_direction, wigner_direction=wigner_direction, directional=True,
            )
        else:
            prefix_input_csr_double_backward_kernel[(x.shape[0], spec.full_components)](
                sector_grads, radial, wigner, radial_direction, wigner_direction,
                *input_indices, **input_launch,
            )
    if write_wigner and not fuse_edge_wigner:
        if degree_tiles:
            launch_prefix_wigner_tiles(
                sector_grads, x, edge_index, radial, wigner_gradient, spec, sector_elements,
                input_direction=input_direction, radial_direction=radial_direction,
            )
        else:
            for degree in range(spec.lmax + 1):
                retained_rows = 1 + 2 * min(spec.mmax, degree)
                prefix_wigner_degree_double_backward_kernel[(edges, retained_rows)](
                    sector_grads,
                    x,
                    edge_index,
                    radial,
                    input_direction,
                    radial_direction,
                    wigner_gradient,
                    edges,
                    channels=spec.channels,
                    full_components=spec.full_components,
                    radial_elements=spec.radial_elements,
                    lmax=spec.lmax,
                    mmax=spec.mmax,
                    degree=degree,
                    offsets=offsets,
                    elements=sector_elements,
                    component_block=triton.next_power_of_2(2 * degree + 1),
                    channel_block=triton.next_power_of_2(spec.channels),
                    compact_elements=wigner.stride(0),
                    compact_wigner=compact_wigner,
                    retained_mmax=mmax if wigner.ndim == 2 else -1,
                    num_warps=4,
                )
    return (
        *output_directions,
        input_gradient,
        radial_gradient,
        wigner_gradient,
    )


if gluon is not None:
    @gluon.jit
    def degree_derivatives(
        edge, nodes, sector_grads, sector_direction,
        x, radial, rotation, x_direction, radial_direction, rotation_direction,
        grad_x, grad_radial, grad_rotation,
        edges,
        degree: gl.constexpr,
        channels: gl.constexpr,
    ):
        width: gl.constexpr = 2 * degree + 1
        full: gl.constexpr = 9
        radial_size: gl.constexpr = 6 * 2 * channels
        rotation_size: gl.constexpr = 35
        rotation_offset: gl.constexpr = (0, 1, 10)[degree]
        # Each warp stays on the channel axis. The input angular components
        # remain in registers and all Wigner-column reductions share a barrier.
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
        channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
        column_layout: gl.constexpr = gl.SliceLayout(1, layout)
        endpoint_channel = gl.arange(0, 2 * channels, layout=channel_layout)
        branch = endpoint_channel // channels
        channel = endpoint_channel % channels
        node = gl.load(nodes + branch * edges + edge)
        x_base = (node * full + degree * degree) * channels + channel
        radial_base = edge * radial_size + endpoint_channel
        rotation_base = edge * rotation_size + rotation_offset

        width_tile: gl.constexpr = triton.next_power_of_2(width)
        column = gl.arange(0, width_tile, layout=column_layout)
        x_address = x_base[None, :] + column[:, None] * channels
        live_column = column < width
        zeros = gl.full((width_tile, 2 * channels), 0, gl.float32, layout=layout)
        zero_columns = gl.full((width_tile,), 0, gl.float32, layout=column_layout)
        features = gl.load(x + x_address, mask=live_column[:, None], other=zeros)
        feature_directions = gl.load(
            x_direction + x_address, mask=live_column[:, None], other=zeros,
        )
        dx_components = zeros
        for order in tl.static_range(degree + 1):
            sector_width = 3 - order
            group_start = (0, 3, 5)[order]
            group = group_start + degree - order
            r_address = radial_base + group * 2 * channels
            r = gl.load(radial + r_address)
            dr = gl.load(radial_direction + r_address)
            dr_gradient = gl.full((2 * channels,), 0, gl.float32, layout=channel_layout)
            for sign in tl.static_range(1 if order == 0 else 2):
                row = 0 if order == 0 else 2 * order - 1 + sign
                row_in_sector = degree - order + sign * sector_width
                rows_in_sector = sector_width if order == 0 else 2 * sector_width
                sector_address = (
                    (edge * rows_in_sector + row_in_sector) * 2 * channels + endpoint_channel
                )
                g = gl.load(sector_grads[order] + sector_address)
                a = g * r
                b = g * dr
                w_address = rotation_base + row * width + column
                w = gl.load(rotation + w_address, mask=live_column, other=zero_columns)
                dw = gl.load(rotation_direction + w_address, mask=live_column, other=zero_columns)
                u = gl.sum(w[:, None] * features, axis=0)
                du = gl.sum(
                    w[:, None] * feature_directions + dw[:, None] * features, axis=0,
                )
                dw_gradient = gl.sum(
                    b[None, :] * features + a[None, :] * feature_directions, axis=1,
                )
                gl.store(grad_rotation + w_address, dw_gradient, mask=live_column)
                dx_components += w[:, None] * b[None, :] + dw[:, None] * a[None, :]
                gl.store(sector_direction[order] + sector_address, r * du + dr * u)
                dr_gradient += g * du
            gl.store(grad_radial + r_address, dr_gradient)

        # Each block owns all edge-local outputs for this degree. Only the node
        # derivative receives contributions from other blocks.
        add_node_direction(grad_x + x_address, dx_components, live_column[:, None])


    @gluon.jit
    def fused_edge_degree_kernel(
        sector_grads, sector_direction, x, nodes, radial, rotation,
        x_direction, radial_direction, rotation_direction,
        grad_x, grad_radial, grad_rotation,
        edges,
        channels: gl.constexpr,
        fixed_degree: gl.constexpr = -1,
    ):
        edge = gl.program_id(0).to(gl.int64)
        if fixed_degree >= 0:
            degree_derivatives(
                edge, nodes, sector_grads, sector_direction, x, radial, rotation,
                x_direction, radial_direction, rotation_direction,
                grad_x, grad_radial, grad_rotation, edges, fixed_degree, channels,
            )
        else:
            degree = gl.program_id(1)
            if degree == 0:
                degree_derivatives(
                    edge, nodes, sector_grads, sector_direction, x, radial, rotation,
                    x_direction, radial_direction, rotation_direction,
                    grad_x, grad_radial, grad_rotation, edges, 0, channels,
                )
            elif degree == 1:
                degree_derivatives(
                    edge, nodes, sector_grads, sector_direction, x, radial, rotation,
                    x_direction, radial_direction, rotation_direction,
                    grad_x, grad_radial, grad_rotation, edges, 1, channels,
                )
            else:
                degree_derivatives(
                    edge, nodes, sector_grads, sector_direction, x, radial, rotation,
                    x_direction, radial_direction, rotation_direction,
                    grad_x, grad_radial, grad_rotation, edges, 2, channels,
                )


    @custom_op("so2::compact_prefix_double_backward", mutates_args=())
    def compact_double_backward(
        sector_grads: list[torch.Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        radial: torch.Tensor,
        rotation: torch.Tensor,
        x_direction: torch.Tensor,
        radial_direction: torch.Tensor,
        rotation_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edges = edge_index.shape[1]
        channels = x.shape[2]
        sector_direction = tuple(torch.empty_like(g) for g in sector_grads)
        grad_x = torch.zeros_like(x)
        grad_radial = torch.empty_like(radial)
        grad_rotation = torch.empty_like(rotation)
        args = (
            tuple(sector_grads), sector_direction, x, edge_index, radial, rotation,
            x_direction, radial_direction, rotation_direction,
            grad_x, grad_radial, grad_rotation, edges,
        )
        with torch.cuda.device(x.device):
            if edges >= 16_384:
                # Isolating the low degrees reduces their register allocation.
                for degree in range(3):
                    fused_edge_degree_kernel[(edges,)](
                        *args, channels=channels, fixed_degree=degree, num_warps=4,
                    )
            elif edges:
                fused_edge_degree_kernel[(edges, 3)](*args, channels=channels, num_warps=4)
        return (*sector_direction, grad_x, grad_radial, grad_rotation)


    @compact_double_backward.register_fake
    def compact_double_backward_fake(
        sector_grads, x, edge_index, radial, rotation,
        x_direction, radial_direction, rotation_direction,
    ):
        return tuple(torch.empty_like(value) for value in (*sector_grads, x, radial, rotation))


    @gluon.jit
    def edge_degree_double_backward(
        seeds, output_directions,
        x, indices, radial, rotations,
        x_direction, radial_direction, rotation_direction,
        grad_x, grad_radial, grad_rotations,
        edges,
        lmax: gl.constexpr, mmax: gl.constexpr, channels: gl.constexpr,
        degree: gl.constexpr, rotation_size: gl.constexpr, degree_offset: gl.constexpr,
        vector: gl.constexpr, warps: gl.constexpr,
        output_mask: gl.constexpr = (True, True, True, True),
    ):
        # One block writes this degree's edge-local outputs. Node gradients
        # from different edges add to the same output through atomics.
        edge = gl.program_id(0).to(gl.int64)
        write_sectors: gl.constexpr = output_mask[0]
        write_input: gl.constexpr = output_mask[1]
        write_radial: gl.constexpr = output_mask[2]
        write_rotation: gl.constexpr = output_mask[3]
        width: gl.constexpr = 2 * degree + 1
        tile: gl.constexpr = triton.next_power_of_2(width)
        endpoints: gl.constexpr = 2 * channels
        full: gl.constexpr = (lmax + 1) ** 2
        layout: gl.constexpr = gl.BlockedLayout([1, vector], [1, 32], [1, warps], [1, 0])
        channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
        column_layout: gl.constexpr = gl.SliceLayout(1, layout)

        endpoint = gl.arange(0, endpoints, layout=channel_layout)
        source_or_receiver = endpoint // channels
        node = gl.load(indices + source_or_receiver * edges + edge)
        channel = endpoint % channels
        column = gl.arange(0, tile, layout=column_layout)
        valid = column < width
        feature_offset = ((node * full + degree * degree) * channels + channel)[None, :]
        feature_offset = feature_offset + column[:, None] * channels
        zeros = gl.full((tile, endpoints), 0.0, gl.float32, layout=layout)
        zero_columns = gl.full((tile,), 0.0, gl.float32, layout=column_layout)
        feature = gl.load(x + feature_offset, valid[:, None], zeros)
        feature_direction = gl.load(x_direction + feature_offset, valid[:, None], zeros)
        radial_offset = (edge * (lmax + 1) + degree) * endpoints + endpoint
        r = gl.load(radial + radial_offset)
        dr = gl.load(radial_direction + radial_offset)
        x_result = gl.full((tile, endpoints), 0.0, gl.float32, layout=layout)
        radial_result = gl.full((endpoints,), 0.0, gl.float32, layout=channel_layout)

        order_count: gl.constexpr = (degree if degree < mmax else mmax) + 1
        for order in tl.static_range(order_count):
            sector_width = lmax + 1 - order
            sector_rows = sector_width if order == 0 else 2 * sector_width
            for sign in tl.static_range(1 if order == 0 else 2):
                row = 0 if order == 0 else 2 * order - 1 + sign
                sector_row = degree - order + sign * sector_width
                sector_offset = (edge * sector_rows + sector_row) * endpoints + endpoint
                rotation_offset = edge * rotation_size + degree_offset + row * width + column

                w = gl.load(rotations + rotation_offset, valid, zero_columns)
                dw = gl.load(rotation_direction + rotation_offset, valid, zero_columns)
                g = gl.load(seeds[order] + sector_offset)
                g_radial = g * r
                g_radial_direction = g * dr
                if write_sectors:
                    u = gl.sum(w[:, None] * feature, axis=0)
                if write_sectors or write_radial:
                    du = gl.sum(w[:, None] * feature_direction + dw[:, None] * feature, axis=0)
                if write_rotation:
                    rotation_result = gl.sum(
                        g_radial_direction[None, :] * feature + g_radial[None, :] * feature_direction,
                        axis=1,
                    )
                if write_input:
                    x_result += w[:, None] * g_radial_direction[None, :] + dw[:, None] * g_radial[None, :]
                if write_radial:
                    radial_result += g * du

                if write_sectors:
                    gl.store(output_directions[order] + sector_offset, r * du + dr * u)
                if write_rotation:
                    gl.store(grad_rotations + rotation_offset, rotation_result, valid)

        if write_radial:
            gl.store(grad_radial + radial_offset, radial_result)
        if write_input:
            add_node_direction(grad_x + feature_offset, x_result, valid[:, None])


    def degree_shared_double_backward(
        seeds, x, edge_index, radial, rotations, x_direction, radial_direction, rotation_direction,
        *, lmax, mmax, output_mask=(True, True, True, True),
    ):
        write_sectors, write_input, write_radial, write_rotation = output_mask
        output_directions = tuple(torch.empty_like(seed) if write_sectors else None for seed in seeds)
        dx = torch.zeros_like(x) if write_input else None
        dr = torch.empty_like(radial) if write_radial else None
        dw = torch.empty_like(rotations) if write_rotation else None
        # Keep tensor argument types stable for pointers eliminated by specialization.
        launch_degree_shared(
            edge_degree_double_backward, x, edge_index, rotations,
            (tuple(seeds), output_directions if write_sectors else tuple(seeds), x, edge_index,
             radial, rotations, x_direction, radial_direction, rotation_direction,
             dx if write_input else x, dr if write_radial else radial, dw if write_rotation else rotations),
            lmax=lmax, mmax=mmax, output_mask=output_mask,
        )
        return (*output_directions, dx, dr, dw)
else:
    compact_double_backward = degree_shared_double_backward = None
