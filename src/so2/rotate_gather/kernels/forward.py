from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import (
    gl,
    gluon,
    channel_launch,
    prefix_wigner_address,
    sector_metadata,
    validate_inputs,
    wig_rot9,
)


@triton.jit
def wigner_conv1_fused_fwd_kernel(
    x_ptr,  # [N, 9, C] node feats (L-major)
    edge_index_ptr,  # [2, E]
    wigner_ptr,  # [E, 35] compact block-diagonal matrix
    radial_ptr,  # [E, 6*2C], six order-degree groups with both endpoints
    m0_ptr,  # [E, 3*2C]
    m1_ptr,  # [E, 4*2C]
    m2_ptr,  # [E, 2*2C]
    num_edges,
    C: tl.constexpr,
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    BLOCK_C: tl.constexpr,  # == C
    GRID_E_STRIDE: tl.constexpr,
):
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C

    C2 = 2 * C  # width of one packed M-row (src|tgt)
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    RAD = 6 * C2

    while edge_id < num_edges:
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        w_base = edge_id * 35

        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(x_ptr + s_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )
        xt = (
            tl.load(x_ptr + t_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )

        y0s, y1s, y2s, y3s, y4s, y5s, y6s, y7s, y8s = wig_rot9(wigner_ptr, w_base, *xs)
        y0t, y1t, y2t, y3t, y4t, y5t, y6t, y7t, y8t = wig_rot9(wigner_ptr, w_base, *xt)

        # Each radial group stores source channels, then target channels.
        rb = edge_id * RAD
        rs = (
            tl.load(radial_ptr + rb + 0 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + c, mask=c_mask, other=0.0),
        )
        rt = (
            tl.load(radial_ptr + rb + 0 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + C + c, mask=c_mask, other=0.0),
        )
        # Reorder by m before the unrolled stores. Opposite orders share radial weights.
        ys_perm = (y0s, y2s, y6s, y3s, y7s, y1s, y5s, y8s, y4s)
        yt_perm = (y0t, y2t, y6t, y3t, y7t, y1t, y5t, y8t, y4t)
        rs_m = (rs[0], rs[1], rs[2], rs[3], rs[4], rs[3], rs[4], rs[5], rs[5])
        rt_m = (rt[0], rt[1], rt[2], rt[3], rt[4], rt[3], rt[4], rt[5], rt[5])
        mb = edge_id * m0_row
        m1b = edge_id * m1_row
        m2b = edge_id * m2_row
        for m in tl.static_range(3):  # m0 buffer: rows M0,M1,M2
            tl.store(m0_ptr + mb + m * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m0_ptr + mb + m * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(3, 7):  # m1 buffer: rows M3,M4,M5,M6
            row = m - 3
            tl.store(m1_ptr + m1b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m1_ptr + m1b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(7, 9):  # m2 buffer: rows M7,M8
            row = m - 7
            tl.store(m2_ptr + m2b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m2_ptr + m2b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)

        edge_id += GRID_E_STRIDE


@triton.jit
def prefix_sector_forward_kernel(
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    outputs,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    reduced_degrees: tl.constexpr,
    dense_rows: tl.constexpr,
    radial_row_indices: tl.constexpr,
    radial_rows: tl.constexpr,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    block_channels: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_wigner: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    edge = tl.program_id(0)
    branch = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    output_channel = branch * channels + channel

    for row in tl.static_range(0, reduced_components):
        degree = list(reduced_degrees)[row]
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(0, 2 * list(reduced_degrees)[row] + 1):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.load(
                wigner_ptr
                + prefix_wigner_address(
                    edge,
                    tl.constexpr(list(dense_rows)[row]),
                    component,
                    full_components,
                    compact_elements,
                    compact_wigner,
                    retained_mmax,
                )
            ).to(tl.float32)
            accumulator += coefficient * feature
        radial_value = tl.load(
            radial_ptr
            + edge * radial_rows * 2 * channels
            + tl.constexpr(list(radial_row_indices)[row]) * 2 * channels
            + output_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value = accumulator * radial_value
        pointer = outputs[0] + edge * list(elements)[0] + row * 2 * channels
        for order in tl.static_range(1, mmax + 1):
            if row >= list(offsets)[order - 1]:
                pointer = (
                    outputs[order]
                    + edge * list(elements)[order]
                    + (row - list(offsets)[order - 1]) * 2 * channels
                )
        pointer += output_channel
        tl.store(pointer, value, mask=channel_mask)


@triton.jit
def sector_prefix_kernel(
    x_ptr, edge_index_ptr, radial_ptr, wigner_ptr,
    output_ptr,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    reduced_degrees: tl.constexpr,
    dense_rows: tl.constexpr,
    radial_row_indices: tl.constexpr,
    radial_rows: tl.constexpr,
    wide_index: tl.constexpr,
    block_channels: tl.constexpr,
    retained_rows: tl.constexpr,
    compact_elements: tl.constexpr,
    compact_row_offsets: tl.constexpr,
):
    edge = tl.program_id(0)
    if wide_index:
        edge = edge.to(tl.int64)
    branch = tl.program_id(1)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    output_channel = branch * channels + channel

    for row in tl.static_range(0, reduced_components):
        degree = list(reduced_degrees)[row]
        accumulator = tl.zeros(channel.shape, dtype=tl.float32)
        for local_component in tl.static_range(0, 2 * list(reduced_degrees)[row] + 1):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            if retained_rows:
                address = (
                    edge * compact_elements
                    + tl.constexpr(list(compact_row_offsets)[row])
                    + local_component
                )
            else:
                address = (
                    edge * full_components * full_components
                    + tl.constexpr(list(dense_rows)[row]) * full_components
                    + component
                )
            coefficient = tl.load(wigner_ptr + address).to(tl.float32)
            accumulator += coefficient * feature
        radial = tl.load(
            radial_ptr
            + edge * radial_rows * 2 * channels
            + tl.constexpr(list(radial_row_indices)[row]) * 2 * channels
            + output_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr
            + edge * reduced_components * 2 * channels
            + row * 2 * channels
            + output_channel,
            accumulator * radial,
            mask=channel_mask,
        )


def prefix_sector_forward(x, edge_index, radial, wigner, *, lmax, mmax):
    block_channels, num_warps = channel_launch(x.shape[2])
    spec, compact_wigner = validate_inputs(
        x, edge_index, radial, wigner,
        lmax=lmax, mmax=mmax,
        block_channels=block_channels, num_warps=num_warps,
    )
    edges = edge_index.shape[1]
    outputs = []
    for order, width in enumerate(spec.sector_widths):
        shape = (
            (edges, width * 2 * spec.channels)
            if order == 0
            else (edges, 2, width * 2 * spec.channels)
        )
        outputs.append(torch.empty(shape, device=x.device, dtype=x.dtype))
    offsets, sector_elements = sector_metadata(spec)
    radial_row_indices = tuple(
        offset // (2 * spec.channels) for offset in spec.radial_row_offsets
    )
    radial_rows = spec.radial_elements // (2 * spec.channels)
    prefix_sector_forward_kernel[(edges, 2)](
        x, edge_index, radial, wigner,
        tuple(outputs),
        edges,
        channels=spec.channels,
        full_components=spec.full_components,
        reduced_components=spec.reduced_components,
        reduced_degrees=spec.reduced_degrees,
        dense_rows=spec.dense_rows,
        radial_row_indices=radial_row_indices,
        radial_rows=radial_rows,
        mmax=spec.mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=block_channels,
        compact_elements=wigner.stride(0),
        compact_wigner=compact_wigner,
        retained_mmax=mmax if wigner.ndim == 2 else -1,
        num_warps=num_warps,
    )
    return tuple(outputs)


if gluon is not None:
    @gluon.jit
    def write_degree_sectors(
        edge, x, indices, radial, rotations, outputs, edges,
        lmax: gl.constexpr, mmax: gl.constexpr, channels: gl.constexpr,
        degree: gl.constexpr, rotation_size: gl.constexpr, degree_offset: gl.constexpr,
        vector: gl.constexpr, warps: gl.constexpr,
    ):
        width: gl.constexpr = 2 * degree + 1
        tile: gl.constexpr = triton.next_power_of_2(width)
        endpoints: gl.constexpr = 2 * channels
        full: gl.constexpr = (lmax + 1) ** 2
        layout: gl.constexpr = gl.BlockedLayout([1, vector], [1, 32], [1, warps], [1, 0])
        channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
        column_layout: gl.constexpr = gl.SliceLayout(1, layout)
        endpoint = gl.arange(0, endpoints, layout=channel_layout)
        node = gl.load(indices + endpoint // channels * edges + edge)
        channel = endpoint % channels
        column = gl.arange(0, tile, layout=column_layout)
        valid = column < width

        x_offset = ((node * full + degree * degree) * channels + channel)[None, :]
        x_offset = x_offset + column[:, None] * channels
        zeros = gl.full((tile, endpoints), 0.0, gl.float32, layout=layout)
        zero_columns = gl.full((tile,), 0.0, gl.float32, layout=column_layout)
        feature = gl.load(x + x_offset, valid[:, None], zeros)
        radial_offset = (edge * (lmax + 1) + degree) * endpoints + endpoint
        r = gl.load(radial + radial_offset)

        order_count: gl.constexpr = (degree if degree < mmax else mmax) + 1
        for order in tl.static_range(order_count):
            sector_width = lmax + 1 - order
            sector_rows = sector_width if order == 0 else 2 * sector_width
            for sign in tl.static_range(1 if order == 0 else 2):
                row = 0 if order == 0 else 2 * order - 1 + sign
                sector_row = degree - order + sign * sector_width
                sector_offset = (edge * sector_rows + sector_row) * endpoints + endpoint
                w_offset = edge * rotation_size + degree_offset + row * width + column
                w = gl.load(rotations + w_offset, valid, zero_columns)
                value = gl.sum(w[:, None] * feature, axis=0) * r
                gl.store(outputs[order] + sector_offset, value)


    @gluon.jit
    def edge_degree_forward_kernel(
        x, indices, radial, rotations, outputs, edges,
        lmax: gl.constexpr, mmax: gl.constexpr, channels: gl.constexpr,
        rotation_size: gl.constexpr, offsets: gl.constexpr,
    ):
        edge = gl.program_id(0).to(gl.int64)
        for degree in tl.static_range(lmax + 1):
            write_degree_sectors(
                edge, x, indices, radial, rotations, outputs, edges,
                lmax, mmax, channels, degree, rotation_size, list(offsets)[degree], 4, 2,
            )


    def degree_shared_forward(x, edge_index, radial, rotations, outputs, spec):
        edges = edge_index.shape[1]
        if not edges:
            return
        offsets = spec.compact_row_offsets[:spec.degree_count]
        with torch.cuda.device(x.device):
            edge_degree_forward_kernel[(edges,)](
                x, edge_index, radial, rotations, outputs, edges,
                spec.lmax, spec.mmax, spec.channels, spec.compact_elements, offsets, num_warps=2,
            )
else:
    degree_shared_forward = None
