"""SO(2) sector assembly and complex recombination with first/second derivatives."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def load_sector_value(
    pointer, edge, row, channel,
    stride_edge, stride_row, stride_channel,
    channels: tl.constexpr,
    width: tl.constexpr,
    order: tl.constexpr,
    raw_complex: tl.constexpr,
    mask,
):
    if raw_complex and order > 0:
        positive = row < width
        local_row = tl.where(positive, row, row - width)
        flat = local_row * channels + channel
        half = width * channels
        first_address = (
            edge * stride_edge
            + tl.where(positive, 0, stride_row)
            + flat * stride_channel
        )
        second_address = (
            edge * stride_edge
            + tl.where(positive, stride_row, 0)
            + (half + flat) * stride_channel
        )
        first = tl.load(pointer + first_address, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(pointer + second_address, mask=mask, other=0.0).to(tl.float32)
        return tl.where(positive, first - second, first + second)
    return tl.load(
        pointer + edge * stride_edge + row * stride_row + channel * stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def store_sector_gradient(
    pointer, value, edge, row, channel,
    stride_edge, stride_row, stride_channel,
    channels: tl.constexpr,
    width: tl.constexpr,
    order: tl.constexpr,
    raw_complex: tl.constexpr,
    mask,
):
    if raw_complex and order > 0:
        positive = row < width
        local_row = tl.where(positive, row, row - width)
        flat = local_row * channels + channel
        half = width * channels
        first_address = (
            edge * stride_edge
            + tl.where(positive, 0, stride_row)
            + flat * stride_channel
        )
        second_address = (
            edge * stride_edge
            + tl.where(positive, stride_row, 0)
            + (half + flat) * stride_channel
        )
        tl.store(pointer + first_address, value, mask=mask)
        tl.store(
            pointer + second_address,
            tl.where(positive, -value, value),
            mask=mask,
        )
    else:
        tl.store(
            pointer + edge * stride_edge + row * stride_row + channel * stride_channel,
            value,
            mask=mask,
        )


@triton.jit
def merge_sectors_kernel(
    sectors,
    output,
    strides: tl.constexpr,
    elements,
    channels: tl.constexpr,
    mmax: tl.constexpr,
    widths: tl.constexpr,
    starts: tl.constexpr,
    reduced_components: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    edge = offset // (reduced_components * channels)
    local = offset - edge * reduced_components * channels
    row = local // channels
    channel = local - row * channels
    value = tl.zeros(offset.shape, tl.float32)

    m0 = row < tl.constexpr(list(widths)[0])
    value += tl.load(
        sectors[0]
        + edge * tl.constexpr(list(strides)[0][0])
        + (row * channels + channel) * tl.constexpr(list(strides)[0][2]),
        mask=mask & m0,
        other=0.0,
    )
    for order in tl.static_range(1, mmax + 1):
        active = (row >= tl.constexpr(list(starts)[order])) & (
            row
            < tl.constexpr(list(starts)[order]) + 2 * tl.constexpr(list(widths)[order])
        )
        local_row = row - tl.constexpr(list(starts)[order])
        pair = local_row // tl.constexpr(list(widths)[order])
        component = local_row - pair * tl.constexpr(list(widths)[order])
        value += tl.load(
            sectors[order]
            + edge * tl.constexpr(list(strides)[order][0])
            + pair * tl.constexpr(list(strides)[order][1])
            + (component * channels + channel) * tl.constexpr(list(strides)[order][2]),
            mask=mask & active,
            other=0.0,
        )
    tl.store(output + offset, value, mask=mask)


@triton.jit
def pack_kernel(
    ys,
    output,
    extra_output,
    strides: tl.constexpr,
    edges,
    row_size: tl.constexpr,
    iteration_size: tl.constexpr,
    reduced_components: tl.constexpr,
    channels: tl.constexpr,
    extra_channels: tl.constexpr,
    mmax: tl.constexpr,
    widths: tl.constexpr,
    starts: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    edge = offset // iteration_size
    local_iteration = offset - edge * iteration_size
    representation_mask = (edge < edges) & (local_iteration < row_size)
    row = local_iteration // channels
    channel = local_iteration - row * channels
    value = tl.zeros(offset.shape, tl.float32)

    m0 = row < tl.constexpr(list(widths)[0])
    value += tl.load(
        ys[0]
        + edge * tl.constexpr(list(strides)[0][0])
        + extra_channels
        + row * channels
        + channel,
        mask=representation_mask & m0,
        other=0.0,
    )
    for order in tl.static_range(1, mmax + 1):
        positive = (row >= tl.constexpr(list(starts)[order])) & (
            row < tl.constexpr(list(starts)[order]) + tl.constexpr(list(widths)[order])
        )
        negative = (
            row >= tl.constexpr(list(starts)[order]) + tl.constexpr(list(widths)[order])
        ) & (
            row
            < tl.constexpr(list(starts)[order]) + 2 * tl.constexpr(list(widths)[order])
        )
        local = tl.where(
            positive,
            row - tl.constexpr(list(starts)[order]),
            row - tl.constexpr(list(starts)[order]) - tl.constexpr(list(widths)[order]),
        )
        component = local * channels + channel
        real0 = tl.load(
            ys[order] + edge * tl.constexpr(list(strides)[order][0]) + component,
            mask=representation_mask & positive,
            other=0.0,
        )
        imag1 = tl.load(
            ys[order]
            + edge * tl.constexpr(list(strides)[order][0])
            + tl.constexpr(list(strides)[order][1])
            + tl.constexpr(list(widths)[order]) * channels
            + component,
            mask=representation_mask & positive,
            other=0.0,
        )
        real1 = tl.load(
            ys[order]
            + edge * tl.constexpr(list(strides)[order][0])
            + tl.constexpr(list(strides)[order][1])
            + component,
            mask=representation_mask & negative,
            other=0.0,
        )
        imag0 = tl.load(
            ys[order]
            + edge * tl.constexpr(list(strides)[order][0])
            + tl.constexpr(list(widths)[order]) * channels
            + component,
            mask=representation_mask & negative,
            other=0.0,
        )
        value += real0 - imag1 + real1 + imag0
    tl.store(
        output + edge * row_size + local_iteration,
        value,
        mask=representation_mask,
    )

    if extra_channels:
        extra_mask = (edge < edges) & (local_iteration < extra_channels)
        extra = tl.load(
            ys[0] + edge * tl.constexpr(list(strides)[0][0]) + local_iteration,
            mask=extra_mask,
            other=0.0,
        )
        tl.store(
            extra_output + edge * extra_channels + local_iteration,
            extra,
            mask=extra_mask,
        )


@triton.jit
def unpack_kernel(
    representation,
    extra,
    ys,
    representation_edge_stride: tl.constexpr,
    representation_row_stride: tl.constexpr,
    representation_channel_stride: tl.constexpr,
    extra_edge_stride: tl.constexpr,
    extra_channel_stride: tl.constexpr,
    edges,
    row_size: tl.constexpr,
    iteration_size: tl.constexpr,
    channels: tl.constexpr,
    extra_channels: tl.constexpr,
    mmax: tl.constexpr,
    widths: tl.constexpr,
    starts: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    edge = offset // iteration_size
    local_iteration = offset - edge * iteration_size
    representation_mask = (edge < edges) & (local_iteration < row_size)
    row = local_iteration // channels
    channel = local_iteration - row * channels
    gradient = tl.load(
        representation
        + edge * representation_edge_stride
        + row * representation_row_stride
        + channel * representation_channel_stride,
        mask=representation_mask,
        other=0.0,
    )
    m0 = row < tl.constexpr(list(widths)[0])
    tl.store(
        ys[0]
        + edge * (extra_channels + tl.constexpr(list(widths)[0]) * channels)
        + extra_channels
        + row * channels
        + channel,
        gradient,
        mask=representation_mask & m0,
    )
    for order in tl.static_range(1, mmax + 1):
        positive = (row >= tl.constexpr(list(starts)[order])) & (
            row < tl.constexpr(list(starts)[order]) + tl.constexpr(list(widths)[order])
        )
        negative = (
            row >= tl.constexpr(list(starts)[order]) + tl.constexpr(list(widths)[order])
        ) & (
            row
            < tl.constexpr(list(starts)[order]) + 2 * tl.constexpr(list(widths)[order])
        )
        local = tl.where(
            positive,
            row - tl.constexpr(list(starts)[order]),
            row - tl.constexpr(list(starts)[order]) - tl.constexpr(list(widths)[order]),
        )
        component = local * channels + channel
        base = edge * 4 * tl.constexpr(list(widths)[order]) * channels
        tl.store(
            ys[order] + base + component, gradient, mask=representation_mask & positive
        )
        tl.store(
            ys[order]
            + base
            + 3 * tl.constexpr(list(widths)[order]) * channels
            + component,
            -gradient,
            mask=representation_mask & positive,
        )
        tl.store(
            ys[order]
            + base
            + 2 * tl.constexpr(list(widths)[order]) * channels
            + component,
            gradient,
            mask=representation_mask & negative,
        )
        tl.store(
            ys[order] + base + tl.constexpr(list(widths)[order]) * channels + component,
            gradient,
            mask=representation_mask & negative,
        )
    if extra_channels:
        extra_mask = (edge < edges) & (local_iteration < extra_channels)
        value = tl.load(
            extra + edge * extra_edge_stride + local_iteration * extra_channel_stride,
            mask=extra_mask,
            other=0.0,
        )
        tl.store(
            ys[0]
            + edge * (extra_channels + tl.constexpr(list(widths)[0]) * channels)
            + local_iteration,
            value,
            mask=extra_mask,
        )


def layout(lmax, mmax):
    if not 0 <= mmax <= lmax:
        raise ValueError("sector assembly requires 0 <= M <= L")
    widths = tuple(lmax - order + 1 for order in range(mmax + 1))
    starts = [0]
    offset = widths[0]
    for width in widths[1:]:
        starts.append(offset)
        offset += 2 * width
    return (
        widths,
        tuple(starts),
        offset,
    )


def normalize_sectors(sectors, lmax, mmax, channels):
    """Give physical coefficients the flat two-plane GEMM layout."""
    widths, _, _ = layout(lmax, mmax)
    if channels <= 0 or len(sectors) != mmax + 1:
        raise ValueError(
            "one physical sector is required per order, with positive channels"
        )
    if sectors[0].ndim not in (2, 3):
        raise ValueError(
            "the zero-order sector must have shape (E, (L+1)*C) or (E, L+1, C)"
        )
    edges = sectors[0].shape[0]
    normalized = []
    for order, (sector, width) in enumerate(zip(sectors, widths, strict=True)):
        if order == 0:
            target = (edges, width * channels)
            accepted = (target, (edges, width, channels))
        else:
            target = (edges, 2, width * channels)
            accepted = (
                target,
                (edges, 2 * width, channels),
                (edges, 2, width, channels),
            )
        if tuple(sector.shape) not in accepted:
            raise ValueError(
                f"unexpected physical sector shape for order {order}: {tuple(sector.shape)}"
            )
        normalized.append(
            sector if tuple(sector.shape) == target else sector.reshape(target)
        )
    return tuple(normalized)


def validate_float_inputs(values, operation):
    first = values[0]
    if any(
        not value.is_cuda
        or value.dtype != torch.float32
        or value.device != first.device
        for value in values
    ):
        raise ValueError(f"{operation} requires CUDA FP32 tensors on one device")


def merge_forward(sectors, lmax, mmax, channels):
    widths, starts, reduced_components = layout(lmax, mmax)
    edges = sectors[0].shape[0]
    output = torch.empty(
        (edges, reduced_components, channels),
        device=sectors[0].device,
        dtype=sectors[0].dtype,
    )
    elements = output.numel()
    block = 256
    merge_sectors_kernel[(triton.cdiv(elements, block),)](
        tuple(sectors),
        output,
        strides=tuple(
            (
                value.stride(0),
                value.stride(1) if value.ndim == 3 else 0,
                value.stride(-1),
            )
            for value in sectors
        ),
        elements=elements,
        channels=channels,
        mmax=mmax,
        widths=widths,
        starts=starts,
        reduced_components=reduced_components,
        block=block,
        num_warps=4,
    )
    return output


def split_sectors(representation, lmax, mmax):
    widths, starts, _ = layout(lmax, mmax)
    edges = representation.shape[0]
    channels = representation.shape[2]
    outputs = [representation[:, : widths[0]].reshape(edges, widths[0] * channels)]
    for order in range(1, mmax + 1):
        start = starts[order]
        width = widths[order]
        outputs.append(
            representation[:, start : start + 2 * width].reshape(
                edges, 2, width * channels
            )
        )
    return tuple(outputs)


class MergeSectors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        sectors = inputs[:-3]
        lmax, mmax, channels = inputs[-3:]
        if len(sectors) != mmax + 1:
            raise ValueError("sector count does not match mmax")
        ctx.dimensions = (lmax, mmax, channels)
        return merge_forward(sectors, *ctx.dimensions)

    @staticmethod
    def backward(ctx, representation_gradient):
        sectors = SplitSectors.apply(representation_gradient, *ctx.dimensions)
        return (*sectors, None, None, None)


class SplitSectors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, representation, lmax, mmax, channels):
        ctx.dimensions = (lmax, mmax, channels)
        return split_sectors(representation, lmax, mmax)

    @staticmethod
    def backward(ctx, *sector_gradients):
        if any(gradient is None for gradient in sector_gradients):
            raise RuntimeError("all SO2 sectors must participate")
        merged = MergeSectors.apply(*sector_gradients, *ctx.dimensions)
        return merged, None, None, None


def merge_sectors(
    sectors: tuple[torch.Tensor, ...],
    *,
    lmax: int,
    mmax: int,
    channels: int,
) -> torch.Tensor:
    """Assemble recombined coefficients into an ``(E, R, C)`` tensor.

    Sector zero is ``(E, (L+1)*C)`` or ``(E, L+1, C)``. At order
    ``m > 0``, let ``w = L+1-m``. A sector can use the GEMM layout
    ``(E, 2, w*C)``, the logical layout ``(E, 2*w, C)``, or explicit
    planes ``(E, 2, w, C)``. The positive-order rows precede the negative
    rows. Inside each signed order, degrees increase from ``m`` to ``L``.
    Output rows concatenate ``0,+1,-1,+2,-2,...``. All sectors are CUDA
    FP32 on one device. The retained count is
    ``R=L+1+2*sum(L+1-m for m in range(1,M+1))``.

    Shape conversion preserves existing flat buffers and uses views when
    possible. Autograd preserves each input's own shape.
    This function does not perform complex recombination.
    """

    sectors = normalize_sectors(tuple(sectors), lmax, mmax, channels)
    validate_float_inputs(sectors, "sector assembly")
    return MergeSectors.apply(*sectors, lmax, mmax, channels)


def pack(raw, lmax, mmax, channels, extra_channels):
    widths, starts, reduced_components = layout(lmax, mmax)
    edges = raw[0].shape[0]
    common = {"device": raw[0].device, "dtype": raw[0].dtype}
    output = torch.empty((edges, reduced_components, channels), **common)
    extra = torch.empty((edges, extra_channels), **common)
    row_size = reduced_components * channels
    iteration_size = max(row_size, extra_channels)
    elements = edges * iteration_size
    block = 256
    pack_kernel[(triton.cdiv(elements, block),)](
        tuple(raw),
        output,
        extra,
        strides=tuple(value.stride() for value in raw),
        edges=edges,
        row_size=row_size,
        iteration_size=iteration_size,
        reduced_components=reduced_components,
        channels=channels,
        extra_channels=extra_channels,
        mmax=mmax,
        widths=widths,
        starts=starts,
        block=block,
        num_warps=4,
    )
    return output, extra


def unpack(representation, extra, lmax, mmax, channels, extra_channels):
    widths, starts, reduced_components = layout(lmax, mmax)
    edges = representation.shape[0]
    common = {"device": representation.device, "dtype": representation.dtype}
    raw = [torch.empty((edges, extra_channels + widths[0] * channels), **common)]
    raw.extend(
        torch.empty((edges, 2, 2 * widths[order] * channels), **common)
        for order in range(1, mmax + 1)
    )
    row_size = reduced_components * channels
    iteration_size = max(row_size, extra_channels)
    elements = edges * iteration_size
    block = 256
    unpack_kernel[(triton.cdiv(elements, block),)](
        representation,
        extra,
        tuple(raw),
        representation_edge_stride=representation.stride(0),
        representation_row_stride=representation.stride(1),
        representation_channel_stride=representation.stride(2),
        extra_edge_stride=extra.stride(0),
        extra_channel_stride=extra.stride(1),
        edges=edges,
        row_size=row_size,
        iteration_size=iteration_size,
        channels=channels,
        extra_channels=extra_channels,
        mmax=mmax,
        widths=widths,
        starts=starts,
        block=block,
        num_warps=4,
    )
    return tuple(raw)


class Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, representation, extra, lmax, mmax, channels, extra_channels):
        ctx.dimensions = (lmax, mmax, channels, extra_channels)
        raw = unpack(representation, extra, *ctx.dimensions)
        return raw

    @staticmethod
    def backward(ctx, *raw_gradients):
        lmax, mmax, channels, extra_channels = ctx.dimensions
        representation, extra = Pack.apply(
            *raw_gradients, lmax, mmax, channels, extra_channels
        )
        return representation, extra, None, None, None, None


class Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        raw, dimensions = args[:-4], args[-4:]
        ctx.dimensions = dimensions
        return pack(raw, *dimensions)

    @staticmethod
    def backward(ctx, representation_gradient, extra_gradient):
        lmax, mmax, channels, extra_channels = ctx.dimensions
        if extra_gradient is None:
            extra_gradient = representation_gradient.new_empty(
                (representation_gradient.shape[0], 0)
            )
        raw = Unpack.apply(
            representation_gradient, extra_gradient, lmax, mmax, channels, extra_channels
        )
        return (*raw, None, None, None, None)
