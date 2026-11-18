"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ...utils.layout import RetainedRotationLayout, get_angular_layout, get_retained_layout
from ...utils.sector_layout import load_sector_value


class PreprojectionOrder(NamedTuple):
    input_width: int
    positive_offsets: tuple[int, ...]
    negative_offsets: tuple[int, ...]
    degrees: tuple[int, ...]
    angular_widths: tuple[int, ...]
    angular_indices: tuple[int, ...]
    packed_widths: tuple[int, ...]
    packed_offset: int


class PreprojectionLayout(NamedTuple):
    orders: tuple[PreprojectionOrder, ...]
    angular_widths: tuple[int, ...]
    packed_widths: tuple[int, ...]
    rotation_width: int


def combine_preprojection_orders(rotation, nodes):
    edges = rotation.shape[0]
    if not (0 < nodes <= 80 and 0 < edges <= 4096):
        return False
    return torch.cuda.get_device_capability(rotation.device) in ((8, 0), (9, 0))


@lru_cache(maxsize=8)
def preprojection_layout(lmax=2, mmax=2):
    angular = get_angular_layout(lmax=lmax, mmax=mmax, channels=128)
    rotation = get_retained_layout(lmax=lmax, mmax=mmax, channels=128)
    input_widths = tuple(width * angular.channels for width in angular.sector_widths)
    angular_widths = tuple(2 * degree + 1 for degree in range(lmax + 1))
    packed_widths = tuple(
        input_widths[0] + 2 * sum(input_widths[1:degree + 1])
        for degree in range(lmax + 1)
    )
    orders = []
    packed_offset = 0
    for order, input_width in enumerate(input_widths):
        width = angular.sector_widths[order]
        positive_offsets, negative_offsets = [], []
        degrees, widths, indices = [], [], []
        for degree in range(order, lmax + 1):
            row = angular.sector_row_offsets[order] + degree - order
            positive = rotation.compact_row_offsets[row]
            negative = rotation.compact_row_offsets[row + width] if order else positive
            components = angular_widths[degree]
            positive_offsets.extend(range(positive, positive + components))
            negative_offsets.extend(range(negative, negative + components))
            degrees.extend((degree,) * components)
            widths.extend((components,) * components)
            indices.extend(range(components))
        orders.append(PreprojectionOrder(
            input_width, tuple(positive_offsets), tuple(negative_offsets),
            tuple(degrees), tuple(widths), tuple(indices),
            tuple(packed_widths[degree] for degree in degrees), packed_offset,
        ))
        packed_offset += (2 if order else 1) * input_width
    return PreprojectionLayout(
        tuple(orders), angular_widths, packed_widths, rotation.compact_elements,
    )


@triton.jit
def reduce_wigner_channel_tiles_kernel(
    partial_ptr,
    output_ptr,
    element_count,
    channel_tiles: tl.constexpr,
    block_elements: tl.constexpr,
):
    element = tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    mask = element < element_count
    total = tl.zeros((block_elements,), tl.float32)
    for channel_tile in tl.static_range(0, channel_tiles):
        total += tl.load(
            partial_ptr + channel_tile * element_count + element,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(output_ptr + element, total, mask=mask)


@triton.jit
def receiver_softmax_direction_kernel(
    logits_ptr,
    envelope_ptr,
    logits_direction_ptr,
    envelope_direction_ptr,
    row_ptr,
    stats_ptr,
    softmax_direction_ptr,
    eps,
    heads: tl.constexpr,
    block_heads: tl.constexpr,
):
    """Compute delta(S) / (S + eps) for every receiver and head."""

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

    direction = tl.zeros((block_heads,), tl.float32)
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
        direction += normalized_exp * (envelope_direction + envelope * logit_direction)
    tl.store(
        softmax_direction_ptr + receiver * heads + head,
        direction,
        mask=head_mask,
    )


def attention_row_layout(values, spec):
    if isinstance(values, torch.Tensor):
        return ()
    return tuple(
        (order, row, width, order > 0 and values[order].shape[-1] == 2 * width * spec.channels)
        for order, width in enumerate(spec.sector_widths)
        for row in range(width if order == 0 else 2 * width)
    )


@triton.jit
def load_attention_value(
    pointers,
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
        return load_sector_value(
            pointers[order], edge, row, channel,
            stride_edge, stride_row, 1,
            channels, width, order, raw_complex, mask,
        )
    else:
        return tl.load(
            pointers + (edge * reduced_components + reduced) * channels + channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)


ValueStorage = torch.Tensor | tuple[torch.Tensor, ...]


def value_blocks(values: ValueStorage) -> tuple[torch.Tensor, ...]:
    return (values,) if isinstance(values, torch.Tensor) else tuple(values)


def attention_dimensions(
    values: ValueStorage, logits: torch.Tensor
) -> tuple[int, int, int]:
    tensors = value_blocks(values)
    if not tensors or tensors[0].ndim != 3 or logits.ndim != 2:
        raise ValueError("attention requires rank-three values and rank-two logits")
    channels, heads = tensors[0].shape[-1], logits.shape[1]
    if heads < 1 or channels < 1 or channels % heads:
        raise ValueError(
            "value channels must be positive and divisible by the number of heads"
        )
    return channels, heads, channels // heads


def empty_values(values: ValueStorage) -> ValueStorage:
    if isinstance(values, torch.Tensor):
        return torch.empty_like(values)
    return tuple(torch.empty_like(value) for value in values)


def validate_inputs(
    values: ValueStorage,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
    lmax: int,
    mmax: int,
) -> tuple[torch.Tensor, RetainedRotationLayout]:
    channels, heads, _ = attention_dimensions(values, logits)
    spec = get_retained_layout(
        lmax=lmax,
        mmax=mmax,
        channels=channels,
    )
    value_tensors = value_blocks(values)
    if not value_tensors:
        raise ValueError("attention values require at least the zero-order sector")
    first = value_tensors[0]
    edges = first.shape[0]
    expected = {
        "values": (edges, spec.reduced_components, channels),
        "logits": (edges, heads),
        "wigner_inv": (edges, spec.compact_elements),
        "receivers": (edges,),
    }
    if not isinstance(values, torch.Tensor):
        actual_sectors = tuple(tuple(value.shape) for value in values)
        raw_sectors = tuple(
            (edges, width, channels) if order == 0 else (edges, 2, 2 * width * channels)
            for order, width in enumerate(spec.sector_widths)
        )
        signed_sectors = tuple(
            (edges, width, channels) if order == 0 else (edges, 2, width * channels)
            for order, width in enumerate(spec.sector_widths)
        )
        expected["values"] = signed_sectors if actual_sectors == signed_sectors else raw_sectors
    actual = {
        "values": (
            tuple(values.shape)
            if isinstance(values, torch.Tensor)
            else tuple(tuple(value.shape) for value in values)
        ),
        "logits": tuple(logits.shape),
        "wigner_inv": tuple(wigner_inv.shape),
        "receivers": tuple(receivers.shape),
    }
    if actual != expected:
        raise ValueError(f"unexpected structured-scatter shapes: {actual=}")
    if tuple(envelope.shape) != (edges, 1):
        raise ValueError(f"unexpected envelope shape: {tuple(envelope.shape)}")
    tensors = (*value_tensors, logits, envelope, wigner_inv, receivers)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("structured-scatter inputs must be CUDA tensors")
    if not all(tensor.device == first.device for tensor in tensors):
        raise ValueError("attention reduction inputs must share one CUDA device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("structured-scatter inputs must be contiguous")
    if not all(
        tensor.dtype == torch.float32
        for tensor in (*value_tensors, logits, envelope, wigner_inv)
    ):
        raise ValueError("attention reduction requires FP32 inputs")
    if receivers.dtype != torch.int64:
        raise ValueError("receivers must be int64")
    return envelope[:, 0], spec


def validate_row_ptr(
    row_ptr: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
) -> None:
    if tuple(row_ptr.shape) != (num_nodes + 1,):
        raise ValueError(f"unexpected CSR row-pointer shape: {tuple(row_ptr.shape)}")
    if not row_ptr.is_cuda or not row_ptr.is_contiguous():
        raise ValueError("CSR row pointers must be contiguous CUDA tensors")
    if (
        row_ptr.dtype not in (torch.int32, torch.int64)
        or row_ptr.device != receivers.device
    ):
        raise ValueError(
            "CSR row pointers must be int32 or int64 on the receiver device"
        )


@triton.jit
def component_degree(component, lmax: tl.constexpr):
    """Map one flattened spherical-harmonic component to its degree."""

    degree = component * 0
    for candidate in tl.static_range(1, lmax + 1):
        degree = tl.where(component >= candidate * candidate, candidate, degree)
    return degree


@triton.jit
def m0_row_offset(degree, m0_offsets: tl.constexpr, degrees: tl.constexpr):
    """Resolve the compact inverse-Wigner base of the m=0 row."""

    offset = degree * 0
    for candidate in tl.static_range(0, degrees):
        offset = tl.where(degree == candidate, list(m0_offsets)[candidate], offset)
    return offset


def scalar_embedding_validate(radial, envelope, wigner, receivers, row_ptr, spec):
    if (
        not radial.is_cuda
        or radial.dtype != torch.float32
        or not radial.is_contiguous()
    ):
        raise ValueError("edge degree requires contiguous CUDA FP32 radial input")
    expected_radial = (spec.degree_count, spec.channels)
    if radial.ndim != 3 or tuple(radial.shape[1:]) != expected_radial:
        raise ValueError(f"unexpected edge-degree radial shape: {radial.shape}")
    if envelope.numel() != radial.shape[0] or not envelope.is_contiguous():
        raise ValueError("unexpected edge-degree envelope")
    if tuple(wigner.shape) != (radial.shape[0], spec.compact_elements):
        raise ValueError(f"unexpected inverse Wigner shape: {wigner.shape}")
    if receivers.shape != (radial.shape[0],) or row_ptr.ndim != 1:
        raise ValueError("unexpected receiver CSR")
    for name, tensor in (("envelope", envelope), ("wigner", wigner)):
        if (
            tensor.device != radial.device
            or tensor.dtype != radial.dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous FP32 on the radial input device"
            )
    for name, tensor in (("receivers", receivers), ("row_ptr", row_ptr)):
        if (
            tensor.device != radial.device
            or tensor.dtype not in (torch.int32, torch.int64)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous integer indices on the radial input device"
            )
    if row_ptr.numel() < 1:
        raise ValueError("receiver CSR must include its terminal offset")


def kernel_metadata(spec):
    return {
        "components": spec.full_components,
        "degrees": spec.degree_count,
        "channels": spec.channels,
        "lmax": spec.lmax,
        "compact_elements": spec.compact_elements,
        "m0_offsets": tuple(
            spec.compact_row_offsets[degree] for degree in range(spec.degree_count)
        ),
    }


def check_inplace_input(value):
    if not torch.is_grad_enabled():
        return
    base = value._base
    if (value.is_leaf and value.requires_grad) or (
        base is not None and base.is_leaf and base.requires_grad
    ):
        raise RuntimeError(
            "initial embedding cannot modify a leaf tensor requiring gradients or its view"
        )


def prepare_inputs(
    messages, envelope, wigner,
    receivers, nodes,
    *,
    lmax,
    mmax,
    wigner_layout,
    raw_complex=False,
):
    if not messages or messages[0].ndim != 3:
        raise ValueError("sum aggregation requires rank-three message sectors")
    edges = messages[0].shape[0]
    if tuple(envelope.shape) not in ((edges,), (edges, 1)):
        raise ValueError("envelope must contain one scalar per edge")
    sum_forward_validate(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout, raw_complex=raw_complex,
    )
    floats = (envelope, wigner, *messages)
    if not all(
        value.is_cuda
        and value.dtype == torch.float32
        and value.device == messages[0].device
        for value in floats
    ):
        raise ValueError("sum aggregation requires FP32 tensors on one CUDA device")
    if (
        receivers.dtype not in (torch.int32, torch.int64)
        or receivers.device != messages[0].device
    ):
        raise ValueError("receivers must be integer indices on the message device")
    # Parameter-gradient kernels share Wigner strides with their dense outputs.
    return envelope.contiguous(), wigner.contiguous(), receivers.contiguous()


def use_degree_sum_node(edges, nodes, lmax, mmax, channels, layout):
    return (
        edges >= 65536 and nodes >= 1024 and 2 <= lmax <= 6
        and (mmax, channels, layout) == (2, 64, 6)
    )



def use_receiver_double_backward(device, *, nodes, lmax, channels, wigner_layout):
    """Keep CSR except for measured large, dense-rotation SM80 workloads."""
    if wigner_layout != 0 or nodes < 512 or lmax < 3 or channels < 128:
        return True
    return torch.cuda.get_device_capability(device) != (8, 0)


@triton.jit
def wigner_offset(
    edge, dense_row, full,
    stride_edge, stride_row, stride_column,
    layout: tl.constexpr,
):
    if layout == 0:
        return edge * stride_edge + dense_row * stride_row + full * stride_column
    if layout == 1:
        return edge * stride_edge + full * stride_row + dense_row * stride_column
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    # IDs >= 4 encode retained-row storage and its independent maximum order.
    if layout >= 4:
        rotation_mmax: tl.constexpr = layout - 4
        complete_degrees = tl.minimum(degree, rotation_mmax + 1)
        block_start = (
            complete_degrees * (4 * complete_degrees * complete_degrees - 1) // 3
        )
        block_start += (2 * rotation_mmax + 1) * (
            degree * degree - complete_degrees * complete_degrees
        )
        signed_order = dense_row - degree * degree - degree
        retained_row = tl.where(
            signed_order == 0,
            0,
            tl.where(signed_order > 0, 2 * signed_order - 1, -2 * signed_order),
        )
        if stride_row == 1:
            address = (
                edge * stride_edge
                + block_start
                + retained_row * (2 * degree + 1)
                + full
                - degree * degree
            )
        else:
            offset = (
                block_start + retained_row * (2 * degree + 1) + full - degree * degree
            )
            address = edge * stride_edge + offset * stride_row
        return address


def logical_rows(spec, order):
    width = spec.sector_widths[order]
    return width if order == 0 else 2 * width


def sum_forward_validate(
    messages,
    envelope,
    wigner,
    receivers,
    nodes,
    *,
    lmax,
    mmax,
    wigner_layout,
    raw_complex=False,
):
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=messages[0].shape[2])
    edges = envelope.shape[0]
    if len(messages) != mmax + 1:
        raise ValueError("one message tensor is required for every retained m sector")
    for order, (message, width) in enumerate(
        zip(messages, spec.sector_widths, strict=True)
    ):
        if raw_complex and order > 0:
            expected = (edges, 2, 2 * width * spec.channels)
        else:
            expected = (edges, logical_rows(spec, order), spec.channels)
        if tuple(message.shape) != expected:
            raise ValueError(f"unexpected message shape for sector {order}")
    if wigner_layout in (0, 1):
        if tuple(wigner.shape) != (edges, spec.full_components, spec.full_components):
            raise ValueError("dense Wigner shape does not match L")
    elif 4 <= wigner_layout <= 4 + lmax:
        rotation_mmax = wigner_layout - 4
        retained_spec = get_retained_layout(
            lmax=lmax, mmax=rotation_mmax, channels=spec.channels
        )
        if rotation_mmax < mmax or tuple(wigner.shape) != (
            edges,
            retained_spec.compact_elements,
            1,
        ):
            raise ValueError(
                "retained Wigner shape or maximum order does not match the consumer"
            )
    else:
        raise ValueError(f"unknown Wigner layout: {wigner_layout}")
    if tuple(receivers.shape) != (edges,) or nodes <= 0:
        raise ValueError("unexpected receiver metadata")
    return spec, edges


def use_l2_tail_reuse(messages, envelope, wigner, lmax, mmax, channels):
    return (
        (lmax, mmax, channels) == (2, 2, 128)
        and envelope.is_cuda
        and torch.cuda.get_device_capability(envelope.device) == (9, 0)
        and len(messages) == 3
        and messages[0].shape[-1] == 128
        and tuple(wigner.shape[1:]) == (35, 1)
        and envelope.numel() > 0
        and all(
            t.dtype == torch.float32
            and t.is_contiguous()
            and t.device == envelope.device
            for t in (envelope, wigner, *messages)
        )
    )
