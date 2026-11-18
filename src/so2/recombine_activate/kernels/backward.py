from __future__ import annotations

from functools import lru_cache
from itertools import groupby
import linecache

import torch
import triton
import triton.language as tl

from .utils import (
    gate_gradient_buffers, gate_scalar_views, load_generated_component,
    projected_product_terms, row_layout, silu_terms, store_raw,
)


@triton.jit
def load_gate_seed(pointer, address, mask, present: tl.constexpr):
    if present:
        return tl.load(pointer + address, mask=mask, other=0.0).to(tl.float32)
    return tl.full(address.shape, 0.0, tl.float32)


@triton.jit
def backward_kernel(
    grad_ptrs,
    gating_ptr,
    x_ptrs,
    gating_gradient_ptr,
    x_gradient_ptrs,
    gating_stride_edge,
    x0_stride_edge,
    raw_grad_ptrs,
    gating_gradient_stride_edge: tl.constexpr,
    x0_gradient_stride_edge: tl.constexpr,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    mmax: tl.constexpr,
    raw_complex: tl.constexpr,
    block_channels: tl.constexpr,
    grad_present: tl.constexpr,
    raw_grad_present: tl.constexpr,
    write_scalar: tl.constexpr,
    three_products: tl.constexpr = False,
    skip_sectors: tl.constexpr = (),
):
    edge = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    if channels > block_channels:
        channel += tl.program_id(1) * block_channels
    mask = channel < channels
    if write_scalar:
        x0_scalar_address = edge * x0_stride_edge + channel
        grad0_scalar_address = edge * (lmax + 1) * channels + channel
        scalar = tl.load(x_ptrs[0] + x0_scalar_address, mask=mask, other=0.0).to(tl.float32)
        scalar_grad = load_gate_seed(grad_ptrs[0], grad0_scalar_address, mask, grad_present[0])
        _, silu_first, _ = silu_terms(scalar)
        tl.store(x_gradient_ptrs[0] + edge * x0_gradient_stride_edge + channel,
                 scalar_grad * silu_first, mask=mask)

    for degree in tl.static_range(1, lmax + 1):
        gate_input_address = (
            edge * gating_stride_edge + (degree - 1) * channels + channel
        )
        gate_output_address = edge * gating_gradient_stride_edge + (degree - 1) * channels + channel
        gate = tl.sigmoid(
            tl.load(gating_ptr + gate_input_address, mask=mask, other=0.0).to(
                tl.float32
            )
        )
        if write_scalar:
            grad0_address = edge * (lmax + 1) * channels + degree * channels + channel
            x0_address = edge * x0_stride_edge + degree * channels + channel
            grad0 = load_gate_seed(grad_ptrs[0], grad0_address, mask, grad_present[0])
            value0 = tl.load(x_ptrs[0] + x0_address, mask=mask, other=0.0).to(tl.float32)
            dx0_address = edge * x0_gradient_stride_edge + degree * channels + channel
            tl.store(x_gradient_ptrs[0] + dx0_address, grad0 * gate, mask=mask)
            contraction = grad0 * (gate * value0)
        for order in tl.static_range(1, mmax + 1):
            if order <= degree and not (skip_sectors[order - 1] if len(skip_sectors) else False):
                width = lmax - order + 1
                local = degree - order
                grad_ptr = grad_ptrs[order]
                x_ptr = x_ptrs[order]
                if three_products:
                    offset = (edge * 2 * width + local) * channels + channel
                    positive = load_gate_seed(grad_ptr, offset, mask, grad_present[order])
                    negative = load_gate_seed(grad_ptr, offset + width * channels, mask, grad_present[order])
                    contraction += positive * tl.load(x_ptr + offset, mask, 0.)
                    contraction += negative * tl.load(x_ptr + offset + width * channels, mask, 0.)
                    target = (edge * width + local) * channels + channel
                    tl.store(x_gradient_ptrs[3 * order - 2] + target, (positive - negative) * gate, mask)
                    tl.store(x_gradient_ptrs[3 * order - 1] + target, (-positive - negative) * gate, mask)
                    tl.store(x_gradient_ptrs[3 * order] + target, negative * gate, mask)
                else:
                    x_gradient_ptr = x_gradient_ptrs[order]
                    for sign in tl.static_range(0, 2):
                        row = local + sign * width
                        grad_address = (
                            edge * 2 * width * channels + row * channels + channel
                        )
                        gradient = load_gate_seed(grad_ptr, grad_address, mask, grad_present[order])
                        if write_scalar:
                            value = tl.load(x_ptr + grad_address, mask=mask, other=0.0).to(tl.float32)
                            contraction += gradient * value
                        if len(raw_grad_present):
                            gradient += load_gate_seed(raw_grad_ptrs[order - 1], grad_address, mask,
                                                       raw_grad_present[order - 1])
                        store_raw(
                            x_gradient_ptr, gradient * gate, edge, row, channel,
                            width, channels, raw_complex, mask,
                        )
        if write_scalar:
            tl.store(gating_gradient_ptr + gate_output_address, (1.0 - gate) * contraction, mask=mask)


def gate_backward(grads, scalar, input0, sectors, *, lmax, mmax, raw_complex=True, three_products=False,
                  raw_cotangents=(), skip_sectors=()):
    gating, x0 = gate_scalar_views(scalar, input0, lmax)
    edges, channels = x0.shape[0], x0.shape[-1]
    write_scalar = not raw_cotangents or any(grad is not None for grad in grads)
    if write_scalar:
        ds, d0, dgates, dx0 = gate_gradient_buffers(scalar, input0, lmax)
    else:
        ds = d0 = None
        dgates, dx0 = gating, x0
    if not raw_complex:
        dsectors = tuple(torch.empty_like(value) for value in sectors)
    elif three_products:
        dsectors = tuple(x0.new_empty((edges, (lmax - order + 1) * channels))
                         for order in range(1, mmax + 1) for _ in range(3))
    else:
        dsectors = tuple(
            None if skip_sectors and skip_sectors[order - 1]
            else x0.new_empty((edges, 2, 2 * (lmax - order + 1) * channels))
            for order in range(1, mmax + 1)
        )
    block = min(256, triton.next_power_of_2(channels))
    backward_kernel[(edges, triton.cdiv(channels, block))](
        tuple(x0 if grad is None else grad for grad in grads),
        gating, (x0, *sectors),
        dgates, (dx0, *(value if value is not None else x0 for value in dsectors)),
        gating.stride(0), x0.stride(0),
        tuple(x0 if grad is None else grad for grad in raw_cotangents),
        dgates.stride(0), dx0.stride(0),
        channels=channels,
        lmax=lmax,
        mmax=mmax,
        raw_complex=raw_complex,
        block_channels=block,
        grad_present=tuple(grad is not None for grad in grads),
        raw_grad_present=tuple(grad is not None for grad in raw_cotangents),
        write_scalar=write_scalar,
        three_products=three_products,
        skip_sectors=skip_sectors,
        num_warps=4,
    )
    return ds, d0, *dsectors


def projected_product_vjp(second, coefficients):
    lines = []
    if second:
        lines += ["    direction_gate = gp * hsc", "    direction_prime = direction_gate * (1. - 2. * gate)"]
    # The bilinear product is symmetric in its two feature axes. Each row
    # reuses the weighted cotangent for all entries in its sparse stencil.
    terms = groupby(zip(projected_product_terms(), coefficients, strict=True), key=lambda item: item[0][0])
    for row, entries in terms:
        lines += ["    projection = tl.full((B,), 0., tl.float32)", f"    gated_seed = g{row} * gate"]
        if second:
            lines += ["    projection_direction = tl.full((B,), 0., tl.float32)",
                      f"    direction_seed = g{row} * direction_gate"]
        for (_, i, j), coefficient in entries:
            pairs = ((i, j), (j, i)) if i != j else ((i, j),)
            value = " + ".join(f"a{u} * b{v}" for u, v in pairs)
            lines += [f"    p = tl.full((), {coefficient!r}, tl.float32)", f"    projection = tl.fma(p, {value}, projection)",
                      "    scale = p * gated_seed"]
            if second:
                tangent = " + ".join(f"ha{u} * b{v} + a{u} * hb{v}" for u, v in pairs)
                lines += [f"    projection_direction = tl.fma(p, {tangent}, projection_direction)",
                          "    direction_scale = p * direction_seed"]
            for u, v in pairs:
                if second:
                    lines += [f"    da{u} = tl.fma(scale, hb{v}, tl.fma(direction_scale, b{v}, da{u}))",
                              f"    db{v} = tl.fma(scale, ha{u}, tl.fma(direction_scale, a{u}, db{v}))"]
                else:
                    lines += [f"    da{u} = tl.fma(scale, b{v}, da{u})", f"    db{v} = tl.fma(scale, a{u}, db{v})"]
        if second:
            lines += [f"    hv{row} = tl.fma(direction_gate, projection, gate * projection_direction)",
                      f"    dsc += g{row} * (gp * projection_direction + direction_prime * projection)"]
        else:
            lines.append(f"    dsc += g{row} * gp * projection")
    return lines


def derivative_kernel_source(
    lmax, mmax, second, channels, auxiliary_channels, physical_input=False,
    sector_cotangents=False, coefficients=None,
):
    """Rebuild T a, T b and F.T g at each grid point in both derivatives.

    The VJP contracts F.T g with each factor to differentiate the other.
    The second pass differentiates those contractions along the supplied
    directions and returns the derivative with respect to g.
    """
    rows = row_layout(lmax, mmax)
    r = len(rows)
    channel_block = triton.next_power_of_2(channels)
    extras = auxiliary_channels + 3 * channels
    hidden = 2 * channels
    name = f"local_grid_derivative_{lmax}_{mmax}_{int(second)}_{channels}_{auxiliary_channels}"
    input_flag = ""
    if physical_input:
        name += "_physical"
        input_flag = ", physical_input=True"
    if coefficients is not None:
        name += "_projected"
    stride_arguments = ", ".join(
        f"s{m}: tl.constexpr, p{m}: tl.constexpr, hs{m}: tl.constexpr, hp{m}: tl.constexpr"
        for m in range(mmax + 1)
    )
    if sector_cotangents:
        name += "_sectors"
        grad_strides = ", ".join(
            f"gse{m}: tl.constexpr, gsp{m}: tl.constexpr, gsc{m}: tl.constexpr"
            for m in range(mmax + 1)
        )
    else:
        grad_strides = "gs0: tl.constexpr, gs1: tl.constexpr"
    args = (
        f"inputs, directions, outputs, g, alpha, t, f, og, {stride_arguments}, {grad_strides}"
        + """,
    a_s0: tl.constexpr, a_s1: tl.constexpr, ts0: tl.constexpr, ts1: tl.constexpr,
    fs0: tl.constexpr, fs1: tl.constexpr,
    edges, grids: tl.constexpr, ep: tl.constexpr, B: tl.constexpr"""
    )
    lines = [
        "@triton.jit",
        f"def {name}({args}):",
        "    lane = tl.arange(0, B)",
        f"    edge = tl.program_id(0) * ep + lane // {channel_block}",
        f"    channel = lane % {channel_block}",
        f"    valid = (edge < edges) & (channel < {channels})",
    ]
    add = lines.append
    for j, (sector, width, local, negative) in enumerate(rows):
        kw = (
            f"sector={sector}, width={width}, local={local}, negative={negative}, "
            f"channels={channels}, extras={extras}"
        )
        add(
            f"    a{j}, b{j} = load_generated_component("
            f"inputs[{sector}], s{sector}, p{sector}, edge, channel, valid, {kw}{input_flag})"
        )
        if second:
            add(
                f"    ha{j}, hb{j} = load_generated_component("
                f"directions[{sector}], hs{sector}, hp{sector}, edge, channel, valid, {kw}{input_flag})"
            )
        if sector_cotangents:
            pair = f" + gsp{sector}" if negative else ""
            g = (
                f"g[{sector}] + edge * gse{sector}{pair} + "
                f"({local * channels} + channel) * gsc{sector}"
            )
        else:
            g = f"g + edge * gs0 + {j} * gs1 + channel"
        add(f"    g{j} = tl.load({g}, mask=valid, other=0.)")
        add(f"    da{j} = tl.full((B,), 0., tl.float32)")
        add(f"    db{j} = tl.full((B,), 0., tl.float32)")
        if second:
            add(f"    hv{j} = tl.full((B,), 0., tl.float32)")
    for key, offset in (
        ("sa", auxiliary_channels),
        ("sb", auxiliary_channels + channels),
        ("sc", auxiliary_channels + 2 * channels),
    ):
        add(
            f"    {key} = tl.load(inputs[0] + edge * s0 + {offset} + channel, "
            f"mask=valid, other=0.)"
        )
        if second:
            add(
                f"    h{key} = tl.load(directions[0] + edge * hs0 + {offset} + channel, "
                f"mask=valid, other=0.)"
            )
    lines += [
        "    sig = tl.sigmoid(sa)",
        "    silu = sa * sig",
        "    sp = sig + sa * sig * (1. - sig)",
        "    gate = tl.sigmoid(sc)",
        "    gp = gate * (1. - gate)",
        "    dsc = tl.full((B,), 0., tl.float32)",
    ]
    if coefficients is not None:
        lines.extend(projected_product_vjp(second, coefficients))
    else:
        add("    for grid in tl.range(0, grids):")
        for j in range(r):
            add(f"        t{j} = tl.load(t + grid * ts0 + {j} * ts1)")
            add(f"        f{j} = tl.load(f + {j} * fs0 + grid * fs1)")
        for var, terms in (
            ("pa", [f"t{j} * a{j}" for j in range(r)]),
            ("pb", [f"t{j} * b{j}" for j in range(r)]),
            ("q", [f"f{j} * g{j}" for j in range(r)]),
        ):
            add(f"        {var} = " + " + ".join(terms))
        if second:
            add("        hpa = " + " + ".join(f"t{j} * ha{j}" for j in range(r)))
            add("        hpb = " + " + ".join(f"t{j} * hb{j}" for j in range(r)))
            lines += [
                "        hprod = hpa * pb + pa * hpb",
                "        dprod = gate * hprod + gp * hsc * pa * pb",
                "        dsc += q * (gp * hprod + gp * (1. - 2. * gate) * hsc * pa * pb)",
                "        qa = q * (gate * hpb + gp * hsc * pb)",
                "        qb = q * (gate * hpa + gp * hsc * pa)",
            ]
        else:
            lines += [
                "        qa = q * gate * pb",
                "        qb = q * gate * pa",
                "        dsc += q * pa * pb * gp",
            ]
        for j in range(r):
            add(f"        da{j} += t{j} * qa")
            add(f"        db{j} += t{j} * qb")
            if second:
                add(f"        hv{j} += f{j} * dprod")
    if second:
        lines += [
            "    spp = 2. * sig * (1. - sig) + sa * sig * (1. - sig) * (1. - 2. * sig)",
            "    dsa = g0 * (spp * hsa * sb + sp * hsb)",
            "    dsb = g0 * sp * hsa",
        ]
    else:
        lines += ["    dsa = g0 * sb * sp", "    dsb = g0 * silu"]
    for key, offset in (
        ("dsa", auxiliary_channels),
        ("dsb", auxiliary_channels + channels),
        ("dsc", auxiliary_channels + 2 * channels),
    ):
        add(
            f"    tl.store(outputs[0] + edge * {extras + (lmax + 1) * hidden} + "
            f"{offset} + channel, {key}, mask=valid)"
        )
    for j in range(triton.cdiv(auxiliary_channels, channels)):
        value = (
            "0."
            if second
            else (
                f"tl.load(alpha + edge * a_s0 + ({channels * j} + channel) * a_s1, "
                f"mask=valid & ({channels * j} + channel < {auxiliary_channels}), other=0.)"
            )
        )
        add(
            f"    tl.store(outputs[0] + edge * {extras + (lmax + 1) * hidden} + "
            f"{channels * j} + channel, {value}, "
            f"mask=valid & ({channels * j} + channel < {auxiliary_channels}))"
        )
    for j, (sector, width, local, negative) in enumerate(rows):
        for high, value in ((False, f"da{j}"), (True, f"db{j}")):
            c = f"channel + {channels if high else 0}"
            if sector == 0:
                add(
                    f"    tl.store(outputs[0] + edge * {extras + (lmax + 1) * hidden} + "
                    f"{extras} + {local * hidden} + {c}, {value}, mask=valid)"
                )
            elif physical_input:
                offset = (width if negative else 0) + local
                add(
                    f"    tl.store(outputs[{sector}] + edge * {2 * width * hidden} + "
                    f"{offset * hidden} + {c}, {value}, mask=valid)"
                )
            else:
                base = f"outputs[{sector}] + edge * {4 * width * hidden}"
                real = (2 * width * hidden if negative else 0) + local * hidden
                imag = (width * hidden if negative else 3 * width * hidden) + local * hidden
                add(f"    tl.store({base} + {real} + {c}, {value}, mask=valid)")
                add(
                    f"    tl.store({base} + {imag} + {c}, "
                    f"{value if negative else '-' + value}, mask=valid)"
                )
        if second:
            scalar = " + sp * hsa * sb + silu * hsb" if j == 0 else ""
            if sector_cotangents:
                stride = (2 if sector else 1) * width * channels
                offset = ((width if negative else 0) + local) * channels
                og = f"og[{sector}] + edge * {stride} + {offset} + channel"
            else:
                og = f"og + edge * {r * channels} + {j * channels} + channel"
            add(f"    tl.store({og}, hv{j}{scalar}, mask=valid)")
    return name, "\n".join(lines) + "\n"


@lru_cache(None)
def generated_derivative_kernel(
    lmax, mmax, second, channels, auxiliary_channels, physical_input=False,
    sector_cotangents=False, coefficients=None,
):
    name, source = derivative_kernel_source(
        lmax, mmax, second, channels, auxiliary_channels, physical_input, sector_cotangents, coefficients,
    )
    filename = f"<so2-generated-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    space = dict(
        triton=triton, tl=tl, load_generated_component=load_generated_component
    )
    exec(compile(source, filename, "exec"), space)
    return space[name]


def raw_activation_derivative(
    raw, dy, daux, to_grid, from_grid,
    lmax, mmax, channels, auxiliary_channels, directions=None, *, physical_input=False, coefficients=None,
):
    second = directions is not None
    tangents = directions or raw
    draw = tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in raw)
    sector_cotangents = isinstance(dy, tuple)
    if sector_cotangents:
        grad_strides = {
            key: value
            for m, grad in enumerate(dy)
            for key, value in (
                (f"gse{m}", grad.stride(0)),
                (f"gsp{m}", grad.stride(1) if m else 0),
                (f"gsc{m}", grad.stride(-1)),
            )
        }
        d_dy = tuple(value.new_empty(value.shape) for value in dy) if second else dy
    else:
        grad_strides = {"gs0": dy.stride(0), "gs1": dy.stride(1)}
        d_dy = dy.new_empty(dy.shape) if second else dy
    if raw[0].shape[0] == 0:
        return draw, d_dy
    channel_block = triton.next_power_of_2(channels)
    edges_per_program = max(1, 128 // channel_block)
    generated_derivative_kernel(
        lmax, mmax, second, channels, auxiliary_channels, physical_input,
        sector_cotangents, coefficients,
    )[
        (triton.cdiv(raw[0].shape[0], edges_per_program),)
    ](
        raw, tangents, draw,
        dy, daux, to_grid, from_grid, d_dy,
        **{
            key: value
            for m, (x, h) in enumerate(zip(raw, tangents))
            for key, value in (
                (f"s{m}", x.stride(0)),
                (f"p{m}", x.stride(1) if m else 0),
                (f"hs{m}", h.stride(0)),
                (f"hp{m}", h.stride(1) if m else 0),
            )
        },
        **grad_strides,
        a_s0=daux.stride(0), a_s1=daux.stride(1),
        ts0=to_grid.stride(0), ts1=to_grid.stride(1),
        fs0=from_grid.stride(0), fs1=from_grid.stride(1),
        edges=raw[0].shape[0],
        grids=to_grid.shape[0],
        ep=edges_per_program,
        B=channel_block * edges_per_program,
        num_warps=4,
        num_stages=1,
    )
    return draw, d_dy


@triton.jit
def recombine_vjp_kernel(grad_ptr, raw_gradient_ptr, elements, half, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < elements
    edge = offset // half
    channel = offset - edge * half
    output_base = edge * 2 * half + channel
    grad0 = tl.load(grad_ptr + output_base, mask=mask, other=0.0)
    grad1 = tl.load(grad_ptr + output_base + half, mask=mask, other=0.0)
    raw_base = edge * 4 * half + channel
    tl.store(raw_gradient_ptr + raw_base, grad0, mask=mask)
    tl.store(raw_gradient_ptr + raw_base + half, grad1, mask=mask)
    tl.store(raw_gradient_ptr + raw_base + 2 * half, grad1, mask=mask)
    tl.store(raw_gradient_ptr + raw_base + 3 * half, -grad0, mask=mask)
