"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

from functools import cache

import torch
import triton
import triton.language as tl

from ...utils.sector_layout import load_sector_value, store_sector_gradient


def gate_scalar_views(scalar, input0, lmax):
    if input0 is not None:
        return scalar, input0
    edges, width = scalar.shape
    channels = width // (2 * lmax + 1)
    gates, coefficients = scalar.split((lmax * channels, (lmax + 1) * channels), dim=1)
    return gates, coefficients.view(edges, lmax + 1, channels)


def gate_gradient_buffers(scalar, input0, lmax):
    scalar_gradient = scalar.new_empty(scalar.shape)
    input0_gradient = input0.new_empty(input0.shape) if input0 is not None else None
    gates, coefficients = gate_scalar_views(scalar_gradient, input0_gradient, lmax)
    return scalar_gradient, input0_gradient, gates, coefficients


@triton.jit
def silu_terms(value):
    sigmoid = tl.sigmoid(value)
    first = sigmoid * (1.0 + value * (1.0 - sigmoid))
    second = sigmoid * (1.0 - sigmoid) * (2.0 + value * (1.0 - 2.0 * sigmoid))
    return value * sigmoid, first, second


@triton.jit
def load_raw(
    pointer, edge, row, channel,
    width: tl.constexpr,
    channels: tl.constexpr,
    raw_complex: tl.constexpr,
    mask,
):
    stride_edge = (4 if raw_complex else 2) * width * channels
    stride_row = 2 * width * channels if raw_complex else channels
    return load_sector_value(
        pointer, edge, row, channel,
        stride_edge, stride_row, 1,
        channels, width, 1, raw_complex, mask,
    )


@triton.jit
def store_raw(
    pointer, value, edge, row, channel,
    width: tl.constexpr,
    channels: tl.constexpr,
    raw_complex: tl.constexpr,
    mask,
):
    stride_edge = (4 if raw_complex else 2) * width * channels
    stride_row = 2 * width * channels if raw_complex else channels
    store_sector_gradient(
        pointer, value, edge, row, channel,
        stride_edge, stride_row, 1,
        channels, width, 1, raw_complex, mask,
    )


@triton.jit
def load_generated_component(
    pointer,
    edge_stride: tl.constexpr,
    pair_stride: tl.constexpr,
    edge,
    channel,
    valid,
    sector: tl.constexpr,
    width: tl.constexpr,
    local: tl.constexpr,
    negative: tl.constexpr,
    channels: tl.constexpr,
    extras: tl.constexpr,
    physical_input: tl.constexpr = False,
):
    base = edge * edge_stride
    if sector == 0:
        a = tl.load(
            pointer + base + extras + local * 2 * channels + channel,
            mask=valid,
            other=0.0,
        )
        b = tl.load(
            pointer + base + extras + local * 2 * channels + channel + channels,
            mask=valid,
            other=0.0,
        )
    elif physical_input:
        component = local * 2 * channels + channel
        if negative:
            component += pair_stride
        a = tl.load(pointer + base + component, mask=valid, other=0.0)
        b = tl.load(pointer + base + component + channels, mask=valid, other=0.0)
    else:
        component = local * 2 * channels + channel
        half = width * 2 * channels
        pair = pair_stride
        if negative:
            a = tl.load(
                pointer + base + pair + component, mask=valid, other=0.0
            ) + tl.load(pointer + base + half + component, mask=valid, other=0.0)
            b = tl.load(
                pointer + base + pair + component + channels, mask=valid, other=0.0
            ) + tl.load(
                pointer + base + half + component + channels, mask=valid, other=0.0
            )
        else:
            a = tl.load(pointer + base + component, mask=valid, other=0.0) - tl.load(
                pointer + base + pair + half + component, mask=valid, other=0.0
            )
            b = tl.load(
                pointer + base + component + channels, mask=valid, other=0.0
            ) - tl.load(
                pointer + base + pair + half + component + channels,
                mask=valid,
                other=0.0,
            )
    return a, b


def row_layout(lmax: int, mmax: int) -> tuple[tuple[int, int, int, bool], ...]:
    rows = [(0, lmax + 1, local, False) for local in range(lmax + 1)]
    for order in range(1, mmax + 1):
        width = lmax - order + 1
        rows.extend((order, width, local, False) for local in range(width))
        rows.extend((order, width, local, True) for local in range(width))
    return tuple(rows)


@cache
def projected_product_terms():
    """Upper-triangle product terms in retained real SO(2) row order."""
    rows = tuple((m + local, m, negative) for m, _, local, negative in row_layout(4, 2))
    terms = []
    for row, (degree, order, negative) in enumerate(rows):
        for i, (degree_a, order_a, negative_a) in enumerate(rows):
            for j in range(i, len(rows)):
                degree_b, order_b, negative_b = rows[j]
                if (
                    abs(degree_a - degree_b) <= degree <= degree_a + degree_b
                    and (degree + degree_a + degree_b) % 2 == 0
                    and 2 * max(order, order_a, order_b) == order + order_a + order_b
                    and (negative + negative_a + negative_b) % 2 == 0
                ):
                    terms.append((row, i, j))
    return tuple(terms)


@torch.inference_mode(False)
@torch.no_grad()
def prepare_projected_product(to_grid, from_grid):
    """Pack the product when the quadrature tables obey the L4/M2 support."""
    if (
        to_grid.shape != (112, 19) or from_grid.shape != (19, 112)
        or to_grid.dtype != torch.float32 or from_grid.dtype != torch.float32
        or to_grid.device != from_grid.device
        or to_grid.requires_grad or from_grid.requires_grad
    ):
        return None

    to = to_grid.to(device="cpu", dtype=torch.float64)
    back = from_grid.to(device="cpu", dtype=torch.float64)
    product = torch.einsum("rg,gi,gj->rij", back, to, to)
    product = (product + product.transpose(1, 2)) * 0.5
    if not product.isfinite().all():
        return None

    row, i, j = (torch.tensor(indices, dtype=torch.long, device="cpu")
                 for indices in zip(*projected_product_terms(), strict=True))
    residual = product.clone()
    residual[row, i, j] = 0
    residual[row, j, i] = 0
    residual_max = residual.abs().max().item()
    relative = (residual.norm() / product.norm().clamp_min(1e-30)).item()
    if relative > 1e-6 or residual_max > 1e-6 * product.abs().max().item():
        return None

    packed = product[row, i, j].float()
    if not packed.isfinite().all():
        return None
    # Higher derivatives use the same rounded coefficients as the kernel.
    dense = torch.zeros_like(product, dtype=torch.float32)
    dense[row, i, j] = packed
    dense[row, j, i] = packed
    return tuple(packed.tolist()), dense.to(to_grid.device)


def edge_tile_size(reduced_components: int) -> int:
    if reduced_components <= 9:
        return 8
    if reduced_components <= 20:
        return 4
    if reduced_components <= 40:
        return 2
    return 1
