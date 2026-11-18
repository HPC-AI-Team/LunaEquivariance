from functools import partial

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from ...utils.sector_layout import load_sector_value, split_sectors
from .utils import gate_gradient_buffers, gate_scalar_views, silu_terms, store_raw


class ActivationDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, third_backward, constants, edge_chunk, *inputs):
        ctx.save_for_backward(*constants, *inputs)
        ctx.constant_count = len(constants)
        ctx.third_backward = third_backward
        ctx.edge_chunk = edge_chunk
        ctx.set_materialize_grads(False)
        return kernel(*constants, *inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *directions):
        saved = ctx.saved_tensors
        constants = saved[:ctx.constant_count]
        inputs = saved[ctx.constant_count:]
        tile = ctx.edge_chunk
        if tile is None or inputs[0].shape[0] <= tile:
            gradients = ctx.third_backward(constants, inputs, directions)
        else:
            # The projected product is independent across edges. Release each
            # contraction graph before constructing the next one.
            gradients = [None] * len(inputs)
            for start in range(0, inputs[0].shape[0], tile):
                block = slice(start, start + tile)
                values = tuple(value[block] for value in inputs)
                seeds = tuple(value[block] if value is not None else None for value in directions)
                partials = ctx.third_backward(constants, values, seeds)
                for index, gradient in enumerate(partials):
                    if gradient is not None:
                        if gradients[index] is None:
                            gradients[index] = torch.zeros_like(inputs[index])
                        gradients[index][block].copy_(gradient)
        return None, None, None, None, *gradients


def activation_third_backward(differentiable, constants, inputs, directions):
    inputs = tuple(value.detach().requires_grad_() if value is not None else None for value in inputs)
    targets = tuple(value for value in inputs if value is not None)
    with torch.enable_grad(), torch.autocast(device_type=targets[0].device.type, enabled=False):
        outputs = differentiable(*constants, *inputs)
        active = [(value, direction) for value, direction in zip(outputs, directions, strict=True)
                  if value is not None and value.requires_grad and direction is not None]
        gradients = torch.autograd.grad(
            tuple(value for value, _ in active), targets,
            tuple(direction for _, direction in active), allow_unused=True,
        ) if active else (None,) * len(targets)
    gradients = iter(gradients)
    return tuple(next(gradients) if value is not None else None for value in inputs)


def physical_planes(raw, width, channels):
    maps = raw.reshape(raw.shape[0], 2, 2, width, channels)
    return torch.stack((maps[:, 0, 0] - maps[:, 1, 1], maps[:, 1, 0] + maps[:, 0, 1]), dim=1)


def gate_hvp(scalar, input0, scalar_direction, input0_direction, *inputs, lmax, mmax, raw_complex=True):
    count = mmax + 1
    grads, sectors, sector_directions = inputs[:count], inputs[count:2 * count - 1], inputs[2 * count - 1:]
    gating, x0 = gate_scalar_views(scalar, input0, lmax)
    hgate, hx0 = gate_scalar_views(scalar_direction, input0_direction, lmax)
    edges, _, channels = x0.shape
    gate = gating.reshape(edges, lmax, channels).sigmoid()
    hgate = hgate.reshape_as(gate)
    gate_first = gate * (1 - gate)
    gate_second = gate_first * (1 - 2 * gate)

    # Degree zero uses SiLU. Other degrees share the same gate across orders.
    x_scalar, h_scalar = x0[:, :1], hx0[:, :1]
    sigmoid = x_scalar.sigmoid()
    silu_first = sigmoid * (1 + x_scalar * (1 - sigmoid))
    silu_second = sigmoid * (1 - sigmoid) * (2 + x_scalar * (1 - 2 * sigmoid))
    grad0 = grads[0]
    dy0 = torch.cat((
        silu_first * h_scalar,
        gate * hx0[:, 1:] + gate_first * hgate * x0[:, 1:],
    ), dim=1)
    dx0 = torch.cat((
        grad0[:, :1] * silu_second * h_scalar,
        grad0[:, 1:] * gate_first * hgate,
    ), dim=1)
    dgate = grad0[:, 1:] * (gate_first * hx0[:, 1:] + gate_second * hgate * x0[:, 1:])

    seed_gradients = [dy0]
    physical_gradients = []
    for order, (grad, sector, direction) in enumerate(zip(grads[1:], sectors, sector_directions, strict=True), 1):
        width = lmax + 1 - order
        shape = (edges, 2, width, channels)
        grad, sector = grad.reshape(shape), sector.reshape(shape)
        h = physical_planes(direction, width, channels) if raw_complex else direction.reshape(shape)
        g = gate[:, None, order - 1:]
        hg = hgate[:, None, order - 1:]
        # The saved sector is already gated. Differentiate while treating
        # that output as independent, then autograd propagates its gradient.
        seed_gradients.append((g * h + (1 - g) * hg * sector).reshape(edges, 2 * width, channels))
        physical_gradients.append((grad * (1 - g) * hg).reshape(edges, 2 * width, channels))
        reduction = (grad * (h - hg * sector)).sum(dim=1)
        dgate = dgate + F.pad(gate_first[:, order - 1:] * reduction, (0, 0, order - 1, 0))
    if input0 is None:
        ds = torch.cat((dgate.flatten(1), dx0.flatten(1)), dim=1)
        dx0 = None
    else:
        ds = dgate.flatten(1)
    return ds, dx0, *seed_gradients, *physical_gradients


@triton.jit
def seed_value(pointer, address, valid, present: tl.constexpr):
    if present:
        return tl.load(pointer + address, valid, 0.)
    return tl.full(address.shape, 0., tl.float32)


@triton.jit
def gate_third_kernel(
    scalar, seeds, sectors, directions, outer_scalar, outer_seeds, outer_sectors,
    dscalar, dseeds, dsectors, ddirections,
    scalar_strides: tl.constexpr, output_strides: tl.constexpr,
    outer_strides: tl.constexpr, direction_strides: tl.constexpr,
    present: tl.constexpr, L: tl.constexpr, M: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
    raw_complex: tl.constexpr,
    write_seeds: tl.constexpr, write_sectors: tl.constexpr,
):
    edge = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    valid = c < C
    ix = edge * scalar_strides[1][0] + c
    ih = edge * scalar_strides[3][0] + c
    iq = edge * (L + 1) * C + c
    x = tl.load(scalar[1] + ix, valid, 0.)
    h = tl.load(scalar[3] + ih, valid, 0.)
    q = tl.load(seeds[0] + iq, valid, 0.)
    a = seed_value(outer_scalar[1], edge * outer_strides[1][0] + c, valid, present[1])
    u = seed_value(outer_seeds[0], iq, valid, present[2][0])
    _, f1, f2 = silu_terms(x)
    sig = tl.sigmoid(x)
    gp = sig * (1 - sig)
    f3 = gp * (3 * (1 - 2 * sig) + x * (1 - 6 * gp))
    tl.store(dscalar[1] + edge * output_strides[1] + c, h * (a * q * f3 + u * f2), valid)
    tl.store(dscalar[3] + edge * output_strides[3] + c, a * q * f2 + u * f1, valid)
    if write_seeds[0]:
        tl.store(dseeds[0] + iq, a * h * f2, valid)

    for degree in tl.static_range(1, L + 1):
        gz = (degree - 1) * C + c
        z = tl.load(scalar[0] + edge * scalar_strides[0][0] + gz, valid, 0.)
        hz = tl.load(scalar[2] + edge * scalar_strides[2][0] + gz, valid, 0.)
        zg = seed_value(outer_scalar[0], edge * outer_strides[0][0] + gz, valid, present[0])
        f = tl.sigmoid(z)
        one_minus_f = 1 - f
        f1 = f * one_minus_f
        f2 = f1 * (1 - 2 * f)
        f3 = f1 * (1 - 6 * f1)
        x = tl.load(scalar[1] + ix + degree * scalar_strides[1][1], valid, 0.)
        h = tl.load(scalar[3] + ih + degree * scalar_strides[3][1], valid, 0.)
        q = tl.load(seeds[0] + iq + degree * C, valid, 0.)
        a = seed_value(outer_scalar[1], edge * outer_strides[1][0] + degree * outer_strides[1][1] + c, valid, present[1])
        u = seed_value(outer_seeds[0], iq + degree * C, valid, present[2][0])
        dz = zg * q * (f2 * h + f3 * hz * x) + a * q * f2 * hz + u * (f1 * h + f2 * hz * x)
        dhz = zg * q * f2 * x + a * q * f1 + u * f1 * x
        tl.store(dscalar[1] + edge * output_strides[1] + degree * C + c, hz * (zg * q * f2 + u * f1), valid)
        tl.store(dscalar[3] + edge * output_strides[3] + degree * C + c, zg * q * f1 + u * f, valid)
        if write_seeds[0]:
            tl.store(dseeds[0] + iq + degree * C, zg * (f1 * h + f2 * hz * x) + a * f1 * hz, valid)

        for order in tl.static_range(1, M + 1):
            if order <= degree:
                width = L + 1 - order
                for sign in tl.static_range(2):
                    row = degree - order + sign * width
                    index = (edge * 2 * width + row) * C + c
                    q = tl.load(seeds[order] + index, valid, 0.)
                    y = tl.load(sectors[order - 1] + index, valid, 0.)
                    h = load_sector_value(
                        directions[order - 1], edge, row, c,
                        direction_strides[order - 1][0], direction_strides[order - 1][1],
                        direction_strides[order - 1][2], C, width, 1, raw_complex, valid,
                    )
                    a = seed_value(outer_sectors[order - 1], index, valid, present[3][order - 1])
                    u = seed_value(outer_seeds[order], index, valid, present[2][order])
                    difference = h - hz * y
                    dz += zg * q * f2 * difference - a * q * f1 * hz + u * f1 * difference
                    dhz += -zg * q * f1 * y + a * q * one_minus_f + u * one_minus_f * y
                    if write_seeds[order]:
                        tl.store(dseeds[order] + index, zg * f1 * difference + a * one_minus_f * hz, valid)
                    if write_sectors[order - 1]:
                        tl.store(dsectors[order - 1] + index, hz * (-zg * q * f1 + u * one_minus_f), valid)
                        store_raw(
                            ddirections[order - 1], zg * q * f1 + u * f, edge, row, c, width, C,
                            raw_complex, valid,
                        )
        tl.store(dscalar[0] + edge * output_strides[0] + gz, dz, valid)
        tl.store(dscalar[2] + edge * output_strides[2] + gz, dhz, valid)


def fused_gate_third_backward(inputs, outer, *, lmax, mmax, raw_complex=True, num_warps=4):
    """Reverse the D2 inputs, keeping the saved gated nonzero sectors independent."""
    if all(seed is None for seed in outer):
        return (None,) * len(inputs)
    count = mmax + 1
    scalar, x0, hscalar, hx0 = inputs[:4]
    grads = inputs[4:4 + count]
    sectors = inputs[4 + count:4 + count + mmax]
    directions = inputs[4 + count + mmax:]
    oscalar, ox0 = outer[:2]
    oseeds, osectors = outer[2:2 + count], outer[2 + count:]

    def last_dense(value):
        return (None if value is None else
                value.resolve_neg() if value.stride(-1) == 1 else value.resolve_neg().contiguous())

    def contiguous(values):
        return tuple(value.resolve_neg().contiguous() if value is not None else None for value in values)

    gating, x0_view = gate_scalar_views(last_dense(scalar), last_dense(x0), lmax)
    hgating, hx0_view = gate_scalar_views(last_dense(hscalar), last_dense(hx0), lmax)
    if x0 is None:
        ogating, ox0_view = (None, None) if oscalar is None else gate_scalar_views(last_dense(oscalar), None, lmax)
    else:
        ogating, ox0_view = last_dense(oscalar), last_dense(ox0)
    has_g, has_x = ogating is not None, ox0_view is not None
    depends_x = has_g or has_x or oseeds[0] is not None
    write_seeds = (has_g or has_x, *(has_g or value is not None for value in osectors))
    write_sectors = tuple(has_g or value is not None for value in oseeds[1:])

    def buffers(values, writes):
        return tuple(value.new_empty(value.shape) if write else None
                     for value, write in zip(values, writes, strict=True))

    ds, dx, dg, d0 = gate_gradient_buffers(scalar, x0, lmax)
    dhs, dhx, dhg, dh0 = gate_gradient_buffers(hscalar, hx0, lmax)
    dseeds = buffers(grads, write_seeds)
    dsectors = buffers(sectors, write_sectors)
    ddirections = buffers(directions, write_sectors)
    edges, channels = x0_view.shape[0], x0_view.shape[-1]
    if edges:
        block = min(256, triton.next_power_of_2(channels))
        outer_scalar = ogating, ox0_view
        oseeds, osectors = contiguous(oseeds), contiguous(osectors)
        tensors = gating, x0_view, hgating, hx0_view
        dscalar = dg, d0, dhg, dh0
        directions = tuple(value.resolve_neg() for value in directions)

        def pointers(values):
            return tuple(scalar if value is None else value for value in values)

        def strides(values):
            return tuple((0, 0) if value is None else value.stride()[:2] for value in values)

        gate_third_kernel[(edges, triton.cdiv(channels, block))](
            tensors, contiguous(grads), contiguous(sectors), directions,
            pointers(outer_scalar), pointers(oseeds), pointers(osectors),
            dscalar, pointers(dseeds), pointers(dsectors), pointers(ddirections),
            scalar_strides=strides(tensors), output_strides=tuple(value.stride(0) for value in dscalar),
            outer_strides=strides(outer_scalar), direction_strides=tuple(value.stride() for value in directions),
            present=(ogating is not None, ox0_view is not None, tuple(value is not None for value in oseeds),
                     tuple(value is not None for value in osectors)),
            L=lmax, M=mmax, C=channels, B=block, raw_complex=raw_complex, num_warps=num_warps,
            write_seeds=write_seeds, write_sectors=write_sectors,
        )
    return ds, dx if depends_x else None, dhs, dhx if depends_x else None, *dseeds, *dsectors, *ddirections


def gate_third_backward(constants, inputs, directions, *, lmax, mmax, raw_complex=True):
    scalar, input0 = inputs[:2]
    channels = scalar.shape[1] // (2 * lmax + 1) if input0 is None else input0.shape[-1]
    if (
        scalar.is_cuda and torch.version.hip is None and not torch.compiler.is_compiling()
        and 0 <= mmax <= min(lmax, 2) and lmax <= 6 and 0 < channels <= 256
        and all(
            value is None or (
                value.dtype == torch.float32 and value.device == scalar.device
                and value.numel() <= 2**31
                and sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) < 2**31
            )
            for value in (*inputs, *directions)
        )
    ):
        capability = torch.cuda.get_device_capability(scalar.device)
        if capability == (9, 0):
            return fused_gate_third_backward(
                inputs, directions, lmax=lmax, mmax=mmax, raw_complex=raw_complex,
            )
        if capability == (8, 0) and channels == 64 and mmax == 2:
            return fused_gate_third_backward(
                inputs, directions, lmax=lmax, mmax=mmax, raw_complex=raw_complex, num_warps=1,
            )
    hvp = partial(gate_hvp, lmax=lmax, mmax=mmax, raw_complex=raw_complex)
    return activation_third_backward(hvp, constants, inputs, directions)


def grid_product(raw, to_grid, from_grid, lmax, mmax, channels, auxiliary_channels, physical_input, polynomial):
    edges = raw[0].shape[0]
    hidden = 2 * channels
    extra = auxiliary_channels + 3 * channels
    coefficients = [raw[0][:, extra:].reshape(edges, lmax + 1, hidden)]
    for order, sector in enumerate(raw[1:], 1):
        width = lmax + 1 - order
        planes = sector if physical_input else physical_planes(sector, width, hidden)
        coefficients.append(planes.reshape(edges, 2 * width, hidden))
    a, b = torch.cat(coefficients, dim=1).split(channels, dim=-1)
    scalar_a, scalar_b, gate = raw[0][:, auxiliary_channels:extra].split(channels, dim=-1)
    if polynomial is None:
        grid = from_grid @ ((to_grid @ a) * (to_grid @ b))
    else:
        grid = torch.einsum("rij,eic,ejc->erc", polynomial, a, b)
    grid = grid * gate.sigmoid().unsqueeze(1)
    scalar = F.silu(scalar_a) * scalar_b
    return torch.cat((grid[:, :1] + scalar.unsqueeze(1), grid[:, 1:]), dim=1)


def grid_hvp(to_grid, from_grid, polynomial, *inputs, dimensions, physical_input):
    lmax, mmax = dimensions[:2]
    count = mmax + 1
    raw, directions, cotangents = inputs[:count], inputs[count:2 * count], inputs[2 * count:]
    output = grid_product(raw, to_grid, from_grid, *dimensions, physical_input, polynomial)
    outputs = split_sectors(output, lmax, mmax) if cotangents[0].ndim == 2 else (output,)
    first = torch.autograd.grad(outputs, raw, cotangents, create_graph=True)
    return torch.autograd.grad(first, (*raw, *cotangents), directions, create_graph=True)


def scalar_product_third(a, b, ua, ub, va, vb, grad, outer):
    s = a.sigmoid()
    sp = s * (1 - s)
    spp = sp * (1 - 2 * s)
    sppp = spp * (1 - 2 * s) - 2 * sp.square()
    silu, d1, d2, d3 = a * s, s + a * sp, 2 * sp + a * spp, 3 * spp + a * sppp
    cross, mixed = ua * vb + va * ub, ua * va
    dg = d1 * cross + d2 * mixed * b
    da = outer * (d1 * ub + d2 * ua * b) + grad * (d2 * cross + d3 * mixed * b)
    db = outer * d1 * ua + grad * d2 * mixed
    dua = outer * d1 * b + grad * (d1 * vb + d2 * va * b)
    dub = outer * silu + grad * d1 * va
    return dg, da, db, dua, dub


@triton.jit
def mixed_outer_kernel(
    pair, u_pair, v_pair, s, sp, spp, u_gate, v_gate, uv, output,
    stride_ug: tl.constexpr, stride_vg: tl.constexpr,
    rows: tl.constexpr, channels: tl.constexpr, block_row: tl.constexpr, block_channel: tl.constexpr,
):
    edge, tile = tl.program_id(0), tl.program_id(1)
    c = tile * block_channel + tl.arange(0, block_channel)
    row = tl.arange(0, block_row)
    at = (edge * rows + row[None, :]) * (2 * channels) + c[:, None]
    active = (c[:, None] < channels) & (row[None, :] < rows)
    a = tl.load(pair + at, active, 0)
    ua = tl.load(u_pair + at, active, 0)
    va = tl.load(v_pair + at, active, 0)
    b = tl.load(pair + at + channels, active, 0)
    ub = tl.load(u_pair + at + channels, active, 0)
    vb = tl.load(v_pair + at + channels, active, 0)
    ordinary = tl.load(s + edge * channels + c, c < channels, 0)[:, None]
    first = tl.load(sp + edge * channels + c, c < channels, 0)[:, None]
    second = tl.load(spp + edge * channels + c, c < channels, 0)[:, None]
    ug = tl.load(u_gate + edge * stride_ug + c, c < channels, 0)[:, None]
    vg = tl.load(v_gate + edge * stride_vg + c, c < channels, 0)[:, None]
    both = tl.load(uv + edge * channels + c, c < channels, 0)[:, None]
    right_a = first * (vg * ub + ug * vb) + second * both * b
    right_u = ordinary * vb + first * vg * b
    right_v = ordinary * ub + first * ug * b
    mixed = a[:, :, None] * right_a[:, None, :]
    mixed += ua[:, :, None] * right_u[:, None, :]
    mixed += va[:, :, None] * right_v[:, None, :]
    out = ((edge * channels + c[:, None, None]) * rows + row[None, :, None]) * rows + row[None, None, :]
    valid = (c[:, None, None] < channels) & (row[None, :, None] < rows) & (row[None, None, :] < rows)
    tl.store(output + out, mixed, valid)


def use_fused_mixed_outer(pair, u_pair, v_pair, u_gate, v_gate):
    return (
        pair.is_cuda and pair.dtype == torch.float32
        and 0 < pair.shape[0] <= 1024 and pair.shape[1:] == (19, 64)
        and u_pair.shape == v_pair.shape == pair.shape
        and all(value.is_contiguous() and not value.is_neg() for value in (pair, u_pair, v_pair))
        and all(value.shape == (pair.shape[0], 1, 32) and value.stride(-1) == 1 and not value.is_neg()
                for value in (u_gate, v_gate))
        and torch.cuda.get_device_capability(pair.device) == (9, 0)
    )


@triton.jit
def adjoint_epilogue_kernel(
    pair, u_pair, v_pair, g, h, s, sp, spp, sppp, uv, u_gate, v_gate,
    d_pair, d_u_pair, dq, duq, stride_u: tl.constexpr, stride_v: tl.constexpr,
    rows: tl.constexpr, channels: tl.constexpr, block_rows: tl.constexpr, block_channels: tl.constexpr,
):
    edge, tile = tl.program_id(0), tl.program_id(1)
    c = tile * block_channels + tl.arange(0, block_channels)
    r = tl.arange(0, block_rows)
    valid = (r[:, None] < rows) & (c[None, :] < channels)
    feature = (edge * rows + r[:, None]) * (2 * channels) + c[None, :]
    product = (edge * channels + c[None, :]) * rows + r[:, None]
    a = tl.load(pair + feature, valid, 0.0)
    ua = tl.load(u_pair + feature, valid, 0.0)
    va = tl.load(v_pair + feature, valid, 0.0)
    ga = tl.load(g + product * 6, valid, 0.0)
    gb = tl.load(g + product * 6 + 1, valid, 0.0)
    gua = tl.load(g + product * 6 + 2, valid, 0.0)
    gub = tl.load(g + product * 6 + 3, valid, 0.0)
    gva = tl.load(g + product * 6 + 4, valid, 0.0)
    gvb = tl.load(g + product * 6 + 5, valid, 0.0)
    ha = tl.load(h + product * 4, valid, 0.0)
    hb = tl.load(h + product * 4 + 1, valid, 0.0)
    hua = tl.load(h + product * 4 + 2, valid, 0.0)
    hub = tl.load(h + product * 4 + 3, valid, 0.0)
    active = c < channels
    gate = edge * channels + c
    ordinary = tl.load(s + gate, active, 0.0)
    first = tl.load(sp + gate, active, 0.0)
    second = tl.load(spp + gate, active, 0.0)
    third = tl.load(sppp + gate, active, 0.0)
    both = tl.load(uv + gate, active, 0.0)
    u = tl.load(u_gate + edge * stride_u + c, active, 0.0)
    v = tl.load(v_gate + edge * stride_v + c, active, 0.0)
    da = ordinary * hub + first * (u * hb + v * gub + u * gvb) + second * both * gb
    db = ordinary * hua + first * (u * ha + v * gua + u * gva) + second * both * ga
    dua = ordinary * (hb + gvb) + first * v * gb
    dub = ordinary * (ha + gva) + first * v * ga
    tl.store(d_pair + feature, da, valid)
    tl.store(d_pair + feature + channels, db, valid)
    tl.store(d_u_pair + feature, dua, valid)
    tl.store(d_u_pair + feature + channels, dub, valid)

    g0 = tl.sum(a * gb, axis=0)
    gu = tl.sum(ua * gb + a * gub, axis=0)
    gv = tl.sum(va * gb + a * gvb, axis=0)
    guv = tl.sum(ua * gvb + va * gub, axis=0)
    h0 = tl.sum(a * hb, axis=0)
    hu = tl.sum(ua * hb + a * hub, axis=0)
    gate_grad = first * (hu + guv) + second * (u * h0 + v * gu + u * gv) + third * both * g0
    gate_direction_grad = first * (h0 + gv) + second * v * g0
    tl.store(dq + gate, gate_grad, active)
    tl.store(duq + gate, gate_direction_grad, active)


def use_adjoint_epilogue(g, h):
    return (
        g.shape[:-1] == h.shape[:-1] and (g.shape[-1], h.shape[-1]) == (6, 4)
        and g.is_contiguous() and h.is_contiguous()
    )


def edge_polynomial_third(polynomial, values, directions):
    grad, pair, gate, u_pair, u_gate = values
    h_grad, v_pair, v_gate = directions
    a, b = pair.chunk(2, dim=-1)
    ua, ub = u_pair.chunk(2, dim=-1)
    va, vb = v_pair.chunk(2, dim=-1)
    edges, rows, channels = a.shape
    packed = polynomial.flatten(1)
    s = gate.sigmoid()
    sp = s * (1 - s)
    spp = sp * (1 - 2 * s)
    sppp = spp * (1 - 2 * s) - 2 * sp.square()
    uv = u_gate * v_gate

    # The polynomial is symmetric in its two input rows. Reuse one
    # cotangent contraction for both fields and their directions.
    def adjoints(cotangent, *columns):
        matrix = (cotangent.transpose(1, 2).reshape(-1, rows) @ packed).reshape(-1, rows, rows)
        rhs = torch.stack(columns, dim=-1).transpose(1, 2).reshape(-1, rows, len(columns))
        return torch.bmm(matrix, rhs).reshape(edges, channels, rows, len(columns))

    g = adjoints(grad, a, b, ua, ub, va, vb)
    h = adjoints(h_grad, a, b, ua, ub)
    fused_outer = use_fused_mixed_outer(pair, u_pair, v_pair, u_gate, v_gate)
    if fused_outer and use_adjoint_epilogue(g, h):
        pair_grad, pair_direction_grad = torch.empty_like(pair), torch.empty_like(u_pair)
        dq, duq = torch.empty_like(s), torch.empty_like(s)
        block_channels = 4
        with torch.cuda.device(pair.device):
            adjoint_epilogue_kernel[(edges, triton.cdiv(channels, block_channels))](
                pair, u_pair, v_pair, g, h, s, sp, spp, sppp, uv, u_gate, v_gate,
                pair_grad, pair_direction_grad, dq, duq, u_gate.stride(0), v_gate.stride(0),
                rows, channels, triton.next_power_of_2(rows), block_channels,
                num_warps=4, enable_fp_fusion=False,
            )
    else:
        ga, gb, gua, gub, gva, gvb = g.transpose(1, 2).unbind(-1)
        ha, hb, hua, hub = h.transpose(1, 2).unbind(-1)
        da = s * hub + sp * (u_gate * hb + v_gate * gub + u_gate * gvb) + spp * uv * gb
        db = s * hua + sp * (u_gate * ha + v_gate * gua + u_gate * gva) + spp * uv * ga
        dua = s * (hb + gvb) + sp * v_gate * gb
        dub = s * (ha + gva) + sp * v_gate * ga

        g0 = (a * gb).sum(1, keepdim=True)
        gu = (ua * gb + a * gub).sum(1, keepdim=True)
        gv = (va * gb + a * gvb).sum(1, keepdim=True)
        guv = (ua * gvb + va * gub).sum(1, keepdim=True)
        h0 = (a * hb).sum(1, keepdim=True)
        hu = (ua * hb + a * hub).sum(1, keepdim=True)
        dq = sp * (hu + guv) + spp * (u_gate * h0 + v_gate * gu + u_gate * gv) + sppp * uv * g0
        duq = sp * (h0 + gv) + spp * v_gate * g0

        pair_grad = torch.cat((da, db), -1)
        pair_direction_grad = torch.cat((dua, dub), -1)

    if fused_outer:
        mixed = pair.new_empty((edges, channels, rows, rows))
        block_channel = 4
        with torch.cuda.device(pair.device):
            mixed_outer_kernel[(edges, triton.cdiv(channels, block_channel))](
                pair, u_pair, v_pair, s, sp, spp, u_gate, v_gate, uv, mixed,
                u_gate.stride(0), v_gate.stride(0), rows, channels,
                triton.next_power_of_2(rows), block_channel,
                num_warps=4, enable_fp_fusion=False,
            )
    else:
        def outer(left, right):
            return left.transpose(1, 2).unsqueeze(-1) * right.transpose(1, 2).unsqueeze(-2)

        # Combine the direction terms before projecting the mixed product.
        mixed = outer(a, sp * (v_gate * ub + u_gate * vb) + spp * uv * b)
        mixed.add_(outer(ua, s * vb + sp * v_gate * b))
        mixed.add_(outer(va, s * ub + sp * u_gate * b))
    dg = (mixed.flatten(2) @ packed.T).transpose(1, 2).contiguous()
    return dg, pair_grad, dq, pair_direction_grad, duq


def projected_edge_grid_third(constants, inputs, outer, *, dimensions):
    if (
        any(value is None for value in outer)
        or torch.is_autocast_enabled("cuda")
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
    ):
        hvp = partial(grid_hvp, dimensions=dimensions, physical_input=True)
        return activation_third_backward(hvp, constants, inputs, outer)

    _, _, polynomial = constants
    lmax, mmax, channels, aux = dimensions
    count = mmax + 1
    edges = inputs[0].shape[0]
    raw, tangent, cotangent = inputs[:count], inputs[count:2 * count], inputs[2 * count:]
    widths = lmax + 1, *(2 * (lmax + 1 - order) for order in range(1, count))

    def fields(values):
        first = values[0]
        a, b, gate = first[:, aux:aux + 3 * channels].split(channels, dim=1)
        sectors = first[:, aux + 3 * channels:], *values[1:]
        pair = torch.cat(tuple(value.reshape(edges, width, 2 * channels)
                               for value, width in zip(sectors, widths, strict=True)), dim=1)
        return pair, a, b, gate[:, None]

    def seeds(values):
        return torch.cat(tuple(value.reshape(edges, width, channels)
                               for value, width in zip(values, widths, strict=True)), dim=1)

    def pack(pair, a, b, gate):
        order0, *nonzero = pair.split(widths, dim=1)
        scalar = torch.cat((raw[0].new_zeros((edges, aux)), a, b, gate[:, 0], order0.flatten(1)), dim=1)
        return scalar, *(value.reshape_as(sector) for value, sector in zip(nonzero, raw[1:], strict=True))

    with torch.autocast(device_type=raw[0].device.type, enabled=False):
        pair, a, b, gate = fields(raw)
        u_pair, ua, ub, u_gate = fields(tangent)
        v_pair, va, vb, v_gate = fields(outer[:count])
        grad, h_grad = seeds(cotangent), seeds(outer[count:])
        dg, d_pair, d_gate, d_u_pair, d_u_gate = edge_polynomial_third(
            polynomial, (grad, pair, gate, u_pair, u_gate), (h_grad, v_pair, v_gate),
        )
        scalar, da, db, dua, dub = scalar_product_third(a, b, ua, ub, va, vb, grad[:, 0], h_grad[:, 0])
        dg[:, 0].add_(scalar)
        return (*pack(d_pair, da, db, d_gate), *pack(d_u_pair, dua, dub, d_u_gate),
                *split_sectors(dg, lmax, mmax))
