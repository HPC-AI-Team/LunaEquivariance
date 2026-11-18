from __future__ import annotations

from functools import lru_cache
import linecache

import torch
import triton
import triton.language as tl

from ...utils.layout import get_retained_layout
from .utils import (
    edge_tile_size,
    load_generated_component,
    load_raw,
    projected_product_terms,
    row_layout,
    silu_terms,
)


@triton.jit
def forward_kernel(
    gating_ptr, x_ptrs, y_ptrs,
    gating_stride_edge, x0_stride_edge,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    three_products: tl.constexpr = False,
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    if channels > block_channels:
        channel += tl.program_id(1) * block_channels
    mask = channel < channels
    scalar = tl.load(
        x_ptrs[0] + edge * x0_stride_edge + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu, _, _ = silu_terms(scalar)
    tl.store(y_ptrs[0] + edge * (lmax + 1) * channels + channel, silu, mask=mask)

    for degree in tl.static_range(1, lmax + 1):
        gate = tl.sigmoid(
            tl.load(
                gating_ptr
                + edge * gating_stride_edge
                + (degree - 1) * channels
                + channel,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        )
        x0_address = edge * x0_stride_edge + degree * channels + channel
        y0_address = edge * (lmax + 1) * channels + degree * channels + channel
        tl.store(
            y_ptrs[0] + y0_address,
            tl.load(x_ptrs[0] + x0_address, mask=mask, other=0.0) * gate,
            mask=mask,
        )
        for order in tl.static_range(1, mmax + 1):
            if order <= degree:
                width = lmax - order + 1
                local = degree - order
                y_ptr = y_ptrs[order]
                if three_products:
                    source = (edge * width + local) * channels + channel
                    p = tl.load(x_ptrs[3 * order - 2] + source, mask, 0.)
                    q = tl.load(x_ptrs[3 * order - 1] + source, mask, 0.)
                    t = tl.load(x_ptrs[3 * order] + source, mask, 0.)
                    target = (edge * 2 * width + local) * channels + channel
                    tl.store(y_ptr + target, (p - q) * gate, mask)
                    tl.store(y_ptr + target + width * channels, (t - p - q) * gate, mask)
                else:
                    for sign in tl.static_range(0, 2):
                        row = local + sign * width
                        value = load_raw(
                            x_ptrs[order], edge, row, channel, width, channels, raw_complex, mask,
                        )
                        tl.store(
                            y_ptr + edge * 2 * width * channels + row * channels + channel,
                            value * gate,
                            mask=mask,
                        )


def kernel_source(
    lmax: int, mmax: int, channels: int, auxiliary_channels: int, physical_input=False,
    *, coefficients=None,
) -> tuple[str, str]:
    rows = row_layout(lmax, mmax)
    channel_block = triton.next_power_of_2(channels)
    extras = auxiliary_channels + 3 * channels
    name = f"generated_raw_l{lmax}_m{mmax}_c{channels}_a{auxiliary_channels}_kernel"
    input_flag = ""
    if physical_input:
        name += "_physical"
        input_flag = ", physical_input=True"
    if coefficients is not None:
        if (lmax, mmax) != (4, 2):
            raise ValueError("projected products require L=4 and M=2")
        name += "_projected"
    stride_arguments = ", ".join(
        f"s{m}: tl.constexpr, p{m}: tl.constexpr" for m in range(mmax + 1)
    )
    arguments = (
        "inputs, outputs, to_grid, from_grid, "
        + stride_arguments
        + """,
    to_grid_stride_grid: tl.constexpr, to_grid_stride_row: tl.constexpr,
    from_grid_stride_row: tl.constexpr, from_grid_stride_grid: tl.constexpr,
    edges, grids: tl.constexpr, edges_per_program: tl.constexpr,
    block: tl.constexpr"""
    )
    lines = [
        "@triton.jit",
        f"def {name}({arguments}):",
        "    lane = tl.arange(0, block)",
        f"    edge = tl.program_id(0) * edges_per_program + lane // {channel_block}",
        f"    channel = lane % {channel_block}",
        f"    valid = (edge < edges) & (channel < {channels})",
    ]
    for row, (sector, width, local, negative) in enumerate(rows):
        lines.append(
            f"    a{row}, b{row} = load_generated_component("
            f"inputs[{sector}], s{sector}, p{sector}, edge, channel, valid, sector={sector}, width={width}, local={local}, "
            f"negative={negative}, channels={channels}, extras={extras}{input_flag})"
        )
    for row in range(len(rows)):
        lines.append(f"    acc{row} = tl.zeros((block,), tl.float32)")
    if coefficients is not None:
        for (row, i, j), value in zip(projected_product_terms(), coefficients, strict=True):
            product = f"a{i} * b{j}"
            if i != j:
                product = f"({product} + a{j} * b{i})"
            lines.append(f"    acc{row} = tl.fma({product}, tl.full((), {value!r}, tl.float32), acc{row})")
    else:
        lines.extend(
            (
                "    for grid in tl.range(0, grids):",
                "        to_base = to_grid + grid * to_grid_stride_grid",
            )
        )
        for row in range(len(rows)):
            lines.append(f"        t{row} = tl.load(to_base + {row} * to_grid_stride_row)")
        pa = " + ".join(f"t{row} * a{row}" for row in range(len(rows)))
        pb = " + ".join(f"t{row} * b{row}" for row in range(len(rows)))
        lines.extend(
            (
                f"        pa = {pa}",
                f"        pb = {pb}",
                "        product = pa * pb",
                "        from_base = from_grid + grid * from_grid_stride_grid",
            )
        )
        for row in range(len(rows)):
            lines.append(
                f"        acc{row} += tl.load(from_base + {row} * "
                f"from_grid_stride_row) * product"
            )
    lines.extend(
        (
            f"    scalar_a = tl.load(inputs[0] + edge * s0 + {auxiliary_channels} + channel, mask=valid, other=0.0)",
            f"    scalar_b = tl.load(inputs[0] + edge * s0 + {auxiliary_channels + channels} + channel, mask=valid, other=0.0)",
            f"    raw_gate = tl.load(inputs[0] + edge * s0 + {auxiliary_channels + 2 * channels} + channel, mask=valid, other=0.0)",
            "    scalar = scalar_a * tl.sigmoid(scalar_a) * scalar_b",
            "    gate = tl.sigmoid(raw_gate)",
        )
    )
    for row, (sector, width, local, negative) in enumerate(rows):
        output_row = local + (width if negative else 0)
        scalar_term = " + scalar" if row == 0 else ""
        lines.extend(
            (
                f"    tl.store(outputs[{sector}] + edge * {((lmax + 1) if sector == 0 else 2 * width) * channels} + "
                f"{output_row * channels} + channel, acc{row} * gate{scalar_term}, mask=valid)",
            )
        )
    return name, "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def generated_kernel(
    lmax: int, mmax: int, channels: int, auxiliary_channels: int, physical_input=False,
    *, coefficients=None,
):
    name, source = kernel_source(
        lmax, mmax, channels, auxiliary_channels, physical_input, coefficients=coefficients,
    )
    filename = f"<so2-generated-{name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {
        "triton": triton,
        "tl": tl,
        "load_generated_component": load_generated_component,
    }
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def grid_forward(
    raw, to_grid, from_grid, lmax, mmax, channels, auxiliary_channels, *,
    physical_input=False, coefficients=None,
):
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    edges = raw[0].shape[0]
    outputs = tuple(
        torch.empty(
            (edges, width * channels) if order == 0 else (edges, 2, width * channels),
            device=raw[0].device,
            dtype=raw[0].dtype,
        )
        for order, width in enumerate(spec.sector_widths)
    )
    if edges == 0:
        return outputs
    channel_block = triton.next_power_of_2(channels)
    edges_per_program = max(
        1, edge_tile_size(spec.reduced_components) * 32 // channel_block
    )
    block = channel_block * edges_per_program
    kernel = generated_kernel(
        lmax, mmax, channels, auxiliary_channels, physical_input, coefficients=coefficients,
    )
    kernel[(triton.cdiv(edges, edges_per_program),)](
        raw, outputs, to_grid, from_grid,
        **{
            key: value
            for m, x in enumerate(raw)
            for key, value in (
                (f"s{m}", x.stride(0)),
                (f"p{m}", x.stride(1) if m else 0),
            )
        },
        to_grid_stride_grid=to_grid.stride(0),
        to_grid_stride_row=to_grid.stride(1),
        from_grid_stride_row=from_grid.stride(0),
        from_grid_stride_grid=from_grid.stride(1),
        edges=edges,
        grids=to_grid.shape[0],
        edges_per_program=edges_per_program,
        block=block,
        num_warps=min(8, max(1, block // 32)),
        num_stages=1,
    )
    return outputs


@triton.jit
def recombine_kernel(raw_ptr, output_ptr, elements, half, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    edge = offset // half
    channel = offset - edge * half
    raw_base = edge * 4 * half + channel
    real0 = tl.load(raw_ptr + raw_base, mask=mask, other=0.0)
    real1 = tl.load(raw_ptr + raw_base + 2 * half, mask=mask, other=0.0)
    imag0 = tl.load(raw_ptr + raw_base + half, mask=mask, other=0.0)
    imag1 = tl.load(raw_ptr + raw_base + 3 * half, mask=mask, other=0.0)
    output_base = edge * 2 * half + channel
    tl.store(output_ptr + output_base, real0 - imag1, mask=mask)
    tl.store(output_ptr + output_base + half, real1 + imag0, mask=mask)


def recombine(raw, half):
    if raw.dtype != torch.float32 or not raw.is_cuda or not raw.is_contiguous():
        raise ValueError("SO(2) complex recombination requires contiguous CUDA FP32")
    if raw.ndim != 3 or raw.shape[1] != 2 or raw.shape[2] != 2 * half:
        raise ValueError("unexpected SO(2) real-block shape")
    elements = raw.shape[0] * half
    output = torch.empty((raw.shape[0], 2, half), device=raw.device, dtype=raw.dtype)
    recombine_kernel[(triton.cdiv(elements, 256),)](
        raw, output, elements, half, block=256, num_warps=4
    )
    return output
