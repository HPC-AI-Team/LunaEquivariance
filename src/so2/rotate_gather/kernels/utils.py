"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

from importlib.util import find_spec

import torch
import triton
import triton.language as tl

from ...utils.layout import RetainedRotationLayout, get_angular_layout, get_retained_layout


@triton.jit
def generated_compact_row_offset(
    row,
    row_offsets: tl.constexpr,
    reduced_components: tl.constexpr,
):
    """Resolve one ragged row offset from static layout metadata."""

    result = row * 0
    for candidate in tl.static_range(0, reduced_components):
        result = tl.where(
            row == candidate,
            list(row_offsets)[candidate],
            result,
        )
    return result


@triton.jit
def structured_prefix_generated_kernel(
    x_ptr, edge_index_ptr, compact_radial_ptr, wigner_ptr,
    wigner_stride_edge, wigner_stride_row, wigner_stride_component,
    outputs,
    edges,
    channels: tl.constexpr,
    full_components: tl.constexpr,
    reduced_components: tl.constexpr,
    degree_count: tl.constexpr,
    reduced_degrees: tl.constexpr,
    compact_row_offsets: tl.constexpr,
    block_channels: tl.constexpr,
    sector_output: tl.constexpr,
):
    """One generated launch whose row loop is fully unrolled per spec."""

    edge = tl.program_id(0)
    channel_tile = tl.program_id(1)
    tiles_per_branch: tl.constexpr = tl.cdiv(channels, block_channels)
    branch = channel_tile // tiles_per_branch
    tile_in_branch = channel_tile - branch * tiles_per_branch
    channel_offsets = tile_in_branch * block_channels + tl.arange(0, block_channels)
    channel_mask = channel_offsets < channels
    node = tl.load(edge_index_ptr + branch * edges + edge)
    branch_channels = branch * channels + channel_offsets

    for row in tl.static_range(0, reduced_components):
        degree = list(reduced_degrees)[row]
        accumulator = tl.zeros(channel_offsets.shape, dtype=tl.float32)
        for local_component in tl.static_range(0, 2 * list(reduced_degrees)[row] + 1):
            component = degree * degree + local_component
            feature = tl.load(
                x_ptr
                + node * full_components * channels
                + component * channels
                + channel_offsets,
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
            accumulator += coefficient * feature

        radial = tl.load(
            compact_radial_ptr
            + edge * degree_count * 2 * channels
            + degree * 2 * channels
            + branch_channels,
            mask=channel_mask,
            other=0.0,
        )
        if sector_output:
            pointer = sector_storage_pointer(
                outputs,
                edge,
                row,
                degree_count - 1,
                2 * channels,
            )
            tl.store(pointer + branch_channels, accumulator * radial, mask=channel_mask)
        else:
            tl.store(
                outputs[0]
                + edge * reduced_components * 2 * channels
                + row * 2 * channels
                + branch_channels,
                accumulator * radial,
                mask=channel_mask,
            )


@triton.jit
def reduced_row_for_degree(
    degree: tl.constexpr,
    local_row,
    lmax: tl.constexpr,
):
    """Map local ``m0,+m1,-m1,...`` order to the reduced-layout row."""

    # The m=0 sector occupies rows [0, L].  Each later order m contributes
    # a positive and a negative sector of width L-m+1.  ``local_row`` is
    # bounded by the spec-derived row count, so order never exceeds either
    # M or the current degree.
    sector_index = local_row - 1
    order = sector_index // 2 + 1
    negative = sector_index % 2
    preceding_orders = order - 1
    sector_base = (
        lmax + 1 + 2 * (preceding_orders * (lmax + 1) - preceding_orders * order // 2)
    )
    sector_width = lmax - order + 1
    sector_row = sector_base + negative * sector_width + degree - order
    return tl.where(local_row == 0, degree, sector_row)


@triton.jit
def sector_storage_pointer(
    bases,
    edge,
    row: tl.constexpr,
    lmax: tl.constexpr,
    row_width: tl.constexpr,
):
    """Address one row in independent m-sector output tensors."""

    width0: tl.constexpr = lmax + 1
    pointer = bases[0] + (edge * width0 + row) * row_width
    for order in tl.static_range(1, len(bases)):
        start = tl.constexpr(lmax + 1 + (order - 1) * (2 * lmax + 2 - order))
        rows = tl.constexpr(2 * (lmax - order + 1))
        if row >= start:
            pointer = bases[order] + (edge * rows + row - start) * row_width
    return pointer


@triton.jit
def load_prefix_cotangent(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    edge,
    row: tl.constexpr,
    channel,
    mask,
    channels: tl.constexpr,
    sector_layout: tl.constexpr,
    lmax: tl.constexpr,
):
    if sector_layout:
        row_width: tl.constexpr = 2 * channels
        pointer = sector_storage_pointer(grads, edge, row, lmax, row_width)
        return tl.load(pointer + channel, mask=mask, other=0.0).to(tl.float32)
    return tl.load(
        grads[0]
        + edge * grad_stride_edge
        + row * grad_stride_row
        + channel * grad_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def store_prefix_direction(
    outputs,
    edge,
    row: tl.constexpr,
    channel,
    value,
    mask,
    channels: tl.constexpr,
    sector_layout: tl.constexpr,
    reduced_components: tl.constexpr,
    lmax: tl.constexpr,
):
    if sector_layout:
        row_width: tl.constexpr = 2 * channels
        pointer = sector_storage_pointer(outputs, edge, row, lmax, row_width)
        tl.store(pointer + channel, value, mask=mask)
    else:
        tl.store(
            outputs[0]
            + edge * reduced_components * 2 * channels
            + row * 2 * channels
            + channel,
            value,
            mask=mask,
        )


@triton.jit
def structured_input_radial_degree(
    grads,
    grad_stride_edge, grad_stride_row, grad_stride_channel,
    x_ptr,
    compact_radial_ptr,
    wigner_ptr,
    wigner_stride_edge, wigner_stride_row, wigner_stride_component,
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
    degree_count: tl.constexpr,
    reduced_components: tl.constexpr,
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

    radial_accumulator = tl.zeros(channel_offsets.shape, dtype=tl.float32)
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
        rotated = tl.zeros(channel_offsets.shape, dtype=tl.float32)
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
            coefficient = tl.load(
                wigner_ptr
                + edge * wigner_stride_edge
                + generated_compact_row_offset(
                    row, compact_row_offsets, reduced_components
                )
                + local_component
            ).to(tl.float32)
            rotated += coefficient * feature
        radial_accumulator += cotangent * rotated

    tl.store(
        radial_gradient_ptr
        + edge * degree_count * 2 * channels
        + degree * 2 * channels
        + branch_channels,
        radial_accumulator,
        mask=channel_mask,
    )

    for local_component in tl.static_range(0, component_width):
        component = component_start + local_component
        input_accumulator = tl.zeros(channel_offsets.shape, dtype=tl.float32)
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
            input_accumulator += coefficient * cotangent * radial
        tl.atomic_add(
            input_gradient_ptr
            + node * full_components * channels
            + component * channels
            + channel_offsets,
            input_accumulator,
            mask=channel_mask,
        )


FUSED_WIGNER_GRID_E_STRIDE = 131072


@triton.jit
def wig_idx(i: tl.constexpr, j: tl.constexpr):
    # Global rows -l,...,+l map into stored rows 0,+1,-1,... .
    if i == 0:
        return 0
    if i < 4:
        degree_one_row: tl.constexpr = (2, 0, 1)[i - 1]
        return 1 + degree_one_row * 3 + j - 1
    degree_two_row: tl.constexpr = (4, 2, 0, 1, 3)[i - 4]
    return 10 + degree_two_row * 5 + j - 4


@triton.jit
def wig_rot9(w_ptr, wb, x0, x1, x2, x3, x4, x5, x6, x7, x8):
    """Return W @ x for the nine L=2 coefficients, using the three degree blocks."""
    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8)
    y0 = tl.load(w_ptr + wb + 0) * xs[0]
    y1 = tl.load(w_ptr + wb + wig_idx(1, 1)) * xs[1]
    y2 = tl.load(w_ptr + wb + wig_idx(2, 1)) * xs[1]
    y3 = tl.load(w_ptr + wb + wig_idx(3, 1)) * xs[1]
    for j in tl.static_range(2, 4):
        y1 += tl.load(w_ptr + wb + wig_idx(1, j)) * xs[j]
        y2 += tl.load(w_ptr + wb + wig_idx(2, j)) * xs[j]
        y3 += tl.load(w_ptr + wb + wig_idx(3, j)) * xs[j]
    y4 = tl.load(w_ptr + wb + wig_idx(4, 4)) * xs[4]
    y5 = tl.load(w_ptr + wb + wig_idx(5, 4)) * xs[4]
    y6 = tl.load(w_ptr + wb + wig_idx(6, 4)) * xs[4]
    y7 = tl.load(w_ptr + wb + wig_idx(7, 4)) * xs[4]
    y8 = tl.load(w_ptr + wb + wig_idx(8, 4)) * xs[4]
    for j in tl.static_range(5, 9):
        y4 += tl.load(w_ptr + wb + wig_idx(4, j)) * xs[j]
        y5 += tl.load(w_ptr + wb + wig_idx(5, j)) * xs[j]
        y6 += tl.load(w_ptr + wb + wig_idx(6, j)) * xs[j]
        y7 += tl.load(w_ptr + wb + wig_idx(7, j)) * xs[j]
        y8 += tl.load(w_ptr + wb + wig_idx(8, j)) * xs[j]
    return y0, y1, y2, y3, y4, y5, y6, y7, y8


@triton.jit
def wig_dw_store2(dw_ptr, wb, a1, b1, a2, b2):
    """Store dW[i,j] = sum_c (a1[i]*b1[j] + a2[i]*b2[j]) for both endpoints."""
    tl.store(dw_ptr + wb + 0, tl.sum(a1[0] * b1[0] + a2[0] * b2[0]))
    for i in tl.static_range(1, 4):
        for j in tl.static_range(1, 4):
            tl.store(dw_ptr + wb + wig_idx(i, j), tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))
    for i in tl.static_range(4, 9):
        for j in tl.static_range(4, 9):
            tl.store(dw_ptr + wb + wig_idx(i, j), tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))


@triton.jit
def prefix_wigner_address(
    edge,
    dense_row,
    full,
    full_components: tl.constexpr,
    compact_elements: tl.constexpr,
    compact: tl.constexpr,
    retained_mmax: tl.constexpr,
):
    if not compact:
        return (
            edge * full_components * full_components
            + dense_row * full_components
            + full
        )
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    block_start = degree * (4 * degree * degree - 1) // 3
    degree_start = degree * degree
    width = 2 * degree + 1
    # Degree-major selected rows, ordered 0,+1,-1,+2,-2,... .
    complete_degrees: tl.constexpr = retained_mmax + 1
    complete_elements: tl.constexpr = (
        complete_degrees * (4 * complete_degrees * complete_degrees - 1) // 3
    )
    block_start = tl.where(
        degree <= complete_degrees,
        block_start,
        complete_elements
        + (2 * retained_mmax + 1)
        * (degree * degree - complete_degrees * complete_degrees),
    )
    order = dense_row - degree_start - degree
    selected_row = tl.where(order == 0, 0, 2 * tl.abs(order) - (order > 0).to(tl.int32))
    return (
        edge * compact_elements
        + block_start
        + selected_row * width
        + full
        - degree_start
    )


@triton.jit
def sector_pointer(
    pointers,
    edge,
    row,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    if mmax == 0:
        pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
    elif mmax <= 4:
        offset1: tl.constexpr = list(offsets)[0]
        offset2: tl.constexpr = list(offsets)[1] if mmax > 1 else 0
        offset3: tl.constexpr = list(offsets)[2] if mmax > 2 else 0
        offset4: tl.constexpr = list(offsets)[3] if mmax > 3 else 0
        if row < offset1:
            pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
        elif mmax == 1 or row < offset2:
            pointer = (
                pointers[1] + edge * list(elements)[1] + (row - offset1) * 2 * channels
            )
        elif mmax == 2 or row < offset3:
            pointer = (
                pointers[2 if mmax >= 2 else 0]
                + edge * list(elements)[2 if mmax >= 2 else 0]
                + (row - offset2) * 2 * channels
            )
        elif mmax == 3 or row < offset4:
            pointer = (
                pointers[3 if mmax >= 3 else 0]
                + edge * list(elements)[3 if mmax >= 3 else 0]
                + (row - offset3) * 2 * channels
            )
        else:
            pointer = (
                pointers[4 if mmax >= 4 else 0]
                + edge * list(elements)[4 if mmax >= 4 else 0]
                + (row - offset4) * 2 * channels
            )
    else:
        pointer = pointers[0] + edge * list(elements)[0] + row * 2 * channels
        for order in tl.static_range(1, mmax + 1):
            if row >= list(offsets)[order - 1]:
                pointer = (
                    pointers[order]
                    + edge * list(elements)[order]
                    + (row - list(offsets)[order - 1]) * 2 * channels
                )
    return pointer


@triton.jit
def load_sector_cotangent(
    grads,
    edge,
    row,
    output_channel,
    mask,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    pointer = sector_pointer(grads, edge, row, mmax, offsets, elements, channels)
    return tl.load(pointer + output_channel, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def store_sector_direction(
    grads,
    edge,
    row,
    output_channel,
    value,
    mask,
    mmax: tl.constexpr,
    offsets: tl.constexpr,
    elements: tl.constexpr,
    channels: tl.constexpr,
):
    pointer = sector_pointer(grads, edge, row, mmax, offsets, elements, channels)
    tl.store(pointer + output_channel, value, mask=mask)


@triton.jit
def load_l2_node_features(base, channels: tl.constexpr, channel_mask):
    return (
        tl.load(base + 0 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 1 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 2 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 3 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 4 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 5 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 6 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 7 * channels, mask=channel_mask, other=0.0).to(tl.float32),
        tl.load(base + 8 * channels, mask=channel_mask, other=0.0).to(tl.float32),
    )


def default_block_channels(*, lmax: int, channels: int) -> int:
    if lmax >= 5:
        return max(32, min(64, 1 << (channels - 1).bit_length()))
    if channels <= 32:
        return 32
    if channels <= 64:
        return 64
    return 128


def validate_prefix_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    compact_radial: torch.Tensor,
    wigner: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    block_channels: int,
    num_warps: int,
) -> RetainedRotationLayout:
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
    edges = edge_index.shape[1]
    expected = {
        "x": (x.shape[0], spec.full_components, spec.channels),
        "edge_index": (2, edges),
        "compact_radial": (edges, spec.degree_count, 2 * spec.channels),
        "wigner": (edges, spec.compact_elements),
    }
    actual = {
        "x": tuple(x.shape),
        "edge_index": tuple(edge_index.shape),
        "compact_radial": tuple(compact_radial.shape),
        "wigner": tuple(wigner.shape),
    }
    if actual != expected:
        raise ValueError(f"unexpected structured prefix shapes: {actual=}")
    values = (x, edge_index, compact_radial, wigner)
    if not all(value.is_cuda for value in values):
        raise ValueError("structured prefix inputs must be CUDA tensors")
    if any(value.device != x.device for value in values):
        raise ValueError("structured prefix inputs must share a device")
    if not all(value.is_contiguous() for value in values):
        raise ValueError("structured prefix inputs must be contiguous")
    if (
        x.dtype != torch.float32
        or compact_radial.dtype != x.dtype
        or wigner.dtype != x.dtype
    ):
        raise ValueError("node-to-edge requires FP32 features")
    if edge_index.dtype != torch.int64:
        raise ValueError("edge_index must be int64")
    if block_channels not in (32, 64, 128):
        raise ValueError("block_channels must be one of 32, 64, or 128")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    return spec


def validate_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    block_channels: int,
    num_warps: int,
):
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
    edges = edge_index.shape[1]
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
    expected = {
        "x": (x.shape[0], spec.full_components, spec.channels),
        "edge_index": (2, edges),
        "radial": (edges, spec.radial_elements),
        "wigner": (
            (edges, retained.compact_elements)
            if wigner.ndim == 2
            else (edges, spec.full_components, spec.full_components)
        ),
    }
    actual = {
        "x": tuple(x.shape),
        "edge_index": tuple(edge_index.shape),
        "radial": tuple(radial.shape),
        "wigner": tuple(wigner.shape),
    }
    if expected != actual:
        raise ValueError(f"unexpected sector prefix shapes: {actual=}, {expected=}")
    if not all(value.is_cuda for value in (x, edge_index, radial, wigner)):
        raise ValueError("sector prefix inputs must be CUDA tensors")
    if not all(value.is_contiguous() for value in (x, edge_index, radial, wigner)):
        raise ValueError("sector prefix inputs must be contiguous")
    if x.dtype != torch.float32 or radial.dtype != x.dtype or wigner.dtype != x.dtype:
        raise ValueError("the sector prefix runtime is FP32 only")
    if any(value.device != x.device for value in (edge_index, radial, wigner)):
        raise ValueError("prefix inputs must share a device")
    if edge_index.dtype != torch.int64:
        raise ValueError("edge_index must be int64")
    if (
        block_channels < 32
        or block_channels & (block_channels - 1)
        or num_warps not in (1, 2, 4, 8)
    ):
        raise ValueError("unsupported launch configuration")
    return spec, wigner.ndim == 2


def use_l2_edge_schedule(
    x, edge_index, radial, wigner, lmax, mmax, channels, *, forward=False,
):
    """Use the edgewise kernel for its measured L=M=2, C=128 workload."""
    if (lmax, mmax, channels) != (2, 2, 128) or not x.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(x.device)
    return (
        x.ndim == 3
        and tuple(x.shape[1:]) == (9, 128)
        and edge_index.ndim == 2
        and edge_index.shape[0] == 2
        and (
            capability == (9, 0)
            or (
                capability == (8, 0)
                and (
                    22_528 <= edge_index.shape[1] <= 48_000
                    if forward
                    else 1_024 <= edge_index.shape[1] <= 32_768
                )
            )
        )
        and tuple(radial.shape) == (edge_index.shape[1], 1536)
        and tuple(wigner.shape) == (edge_index.shape[1], 35)
        and edge_index.dtype == torch.int64
        and all(t.dtype == torch.float32 for t in (x, radial, wigner))
        and all(
            t.device == x.device and t.is_contiguous()
            for t in (x, edge_index, radial, wigner)
        )
    )


def channel_launch(channels: int) -> tuple[int, int]:
    block = max(128, triton.next_power_of_2(channels))
    if block == 128 and torch.cuda.get_device_capability()[0] >= 9:
        return block, 4
    return block, 1 if block == 128 else 4


def input_channel_warps(block_channels: int) -> int:
    if block_channels != 128:
        return 4
    return 4 if torch.cuda.get_device_capability()[0] >= 9 else 2


def sector_metadata(spec):
    starts = tuple(spec.sector_row_offsets[1:])
    row_counts = tuple(
        width if order == 0 else 2 * width
        for order, width in enumerate(spec.sector_widths)
    )
    elements = tuple(rows * 2 * spec.channels for rows in row_counts)
    return starts, elements


def use_prefix_degree_tiles(x, edges, spec, compact_wigner, grads, *, directional=False):
    esen_shape = (
        (spec.lmax, spec.mmax, spec.channels) == (3, 2, 128)
        and 8192 <= edges <= 32768
        and 24 * x.shape[0] <= edges <= 40 * x.shape[0]
    )
    compact_adjoint = (
        directional and 2 <= spec.lmax <= 6 and (spec.mmax, spec.channels) == (2, 64)
        and edges >= 8192 and edges * spec.radial_elements < 2**31 and x.numel() < 2**31
    )
    return (
        compact_wigner
        and (esen_shape or compact_adjoint)
        and not torch.compiler.is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (9, 0)
        and all(grad.dtype == torch.float32 and grad.is_contiguous() for grad in grads)
    )


if (
    torch.version.cuda
    and find_spec("triton.experimental") is not None
    and find_spec("triton.experimental.gluon") is not None
):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
else:
    gluon = gl = None


def launch_degree_shared(kernel, x, edge_index, rotations, args, *, lmax, mmax, **kernel_options):
    edges = edge_index.shape[1]
    if not edges:
        return
    channels = x.shape[2]
    warps = 4
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    offsets = spec.compact_row_offsets[:spec.degree_count]
    with torch.cuda.device(x.device):
        for degree, offset in enumerate(offsets):
            kernel[(edges,)](
                *args, edges, lmax, mmax, channels, degree, rotations.shape[1], offset,
                8 // warps, warps, num_warps=warps, **kernel_options,
            )


if gluon is not None:
    @gluon.jit
    def add_node_direction(pointer, value, mask):
        # Gluon 3.4 has no pointer atomic builtin. Use the same FP32 GPU-scope
        # relaxed reduction as the Triton kernel, without a returned old value.
        gl.inline_asm_elementwise(
            """
        {
            .reg .pred valid;
            setp.ne.u32 valid, $3, 0;
            @valid red.relaxed.gpu.global.add.f32 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
            constraints="=r,l,f,r",
            args=(pointer.to(gl.uint64), value, mask.to(gl.int32)),
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
else:
    add_node_direction = None
