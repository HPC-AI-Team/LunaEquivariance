"""Third reverse derivatives of the grouped radial normalization."""

import torch
import triton
import triton.language as tl

from ...utils.normalization import normalized_directions, normalized_directions_vjp, reduce_columns


@triton.jit
def triple_backward_rows(
    saved, outer, outputs, partial, rows, tiles,
    C: tl.constexpr, B_C: tl.constexpr, B_R: tl.constexpr, EPS: tl.constexpr,
    has_outer: tl.constexpr, has_inner: tl.constexpr,
    requested: tl.constexpr, param_slots: tl.constexpr,
):
    tile, group = tl.program_id(0), tl.program_id(1)
    row = tile * B_R + tl.arange(0, B_R)
    channel = tl.arange(0, B_C)
    valid = (row[:, None] < rows) & (channel[None, :] < C)
    columns = group * C + channel
    index = (group * rows + row[:, None]) * C + channel[None, :]
    inv_c = 1. / C

    dy = tl.load(saved[0] + index, valid, 0.)
    x = tl.load(saved[1] + index, valid, 0.) + tl.load(saved[2] + columns, channel < C, 0.)[None, :]
    w = tl.load(saved[3] + columns, channel < C, 0.)[None, :]
    b = tl.load(saved[4] + columns, channel < C, 0.)[None, :]
    v = tl.full((B_R, B_C), 0., tl.float32)
    if has_inner[0]:
        v += tl.load(saved[5] + index, valid, 0.)
    if has_inner[1]:
        v += tl.load(saved[6] + columns, channel < C, 0.)[None, :]
    u = tl.full((B_R, B_C), 0., tl.float32)
    if has_outer[1]:
        u += tl.load(outer[1] + index, valid, 0.)
    if has_outer[2]:
        u += tl.load(outer[2] + columns, channel < C, 0.)[None, :]

    n, nu, nv, nuv, state = normalized_directions(x, u, v, valid, inv_c, EPS)
    a = w * n + b
    au, av, auv = w * nu, w * nv, w * nuv
    if has_outer[3]:
        wu = tl.load(outer[3] + columns, channel < C, 0.)[None, :]
        au += wu * n
        auv += wu * nv
    if has_outer[4]:
        au += tl.load(outer[4] + columns, channel < C, 0.)[None, :]
    if has_inner[2]:
        wv = tl.load(saved[7] + columns, channel < C, 0.)[None, :]
        av += wv * n
        auv += wv * nu
    if has_inner[3]:
        av += tl.load(saved[8] + columns, channel < C, 0.)[None, :]

    sigmoid = tl.sigmoid(a)
    prime = sigmoid * (1 - sigmoid)
    second = prime * (1 - 2 * sigmoid)
    third = prime * (1 - 6 * prime)
    f1, f2, f3 = sigmoid + a * prime, 2 * prime + a * second, 3 * second + a * third
    # Reverse the scalar contraction dy * mixed_direction + z * inner_direction.
    da = dy * (f3 * au * av + f2 * auv)
    dau, dav, dauv = dy * f2 * av, dy * f2 * au, dy * f1
    if has_outer[0]:
        z = tl.load(outer[0] + index, valid, 0.)
        da += z * f2 * av
        dav += z * f1
    if requested[0]:
        tl.store(outputs[0] + index, f2 * au * av + f1 * auv, valid)

    dn, dnu, dnv, dnuv = w * da, w * dau, w * dav, w * dauv
    if has_outer[3]:
        dn += wu * dau
        dnv += wu * dauv
    if has_inner[2]:
        dn += wv * dav
        dnu += wv * dauv

    dx, dhx = normalized_directions_vjp(state, dn, dnu, dnv, dnuv, valid, inv_c)
    if requested[1]:
        tl.store(outputs[1] + index, dx, valid)
    if requested[5]:
        tl.store(outputs[2] + index, dhx, valid)

    # The parameter gradients sum over rows within each group.
    pa = partial + (group * tiles + tile) * C + channel
    stride = tl.num_programs(1) * tiles * C
    if requested[2]:
        tl.store(pa + param_slots[0] * stride, tl.sum(tl.where(valid, dx, 0.), 0), channel < C)
    if requested[3]:
        dw = da * n + dau * nu + dav * nv + dauv * nuv
        tl.store(pa + param_slots[1] * stride, tl.sum(tl.where(valid, dw, 0.), 0), channel < C)
    if requested[4]:
        tl.store(pa + param_slots[2] * stride, tl.sum(tl.where(valid, da, 0.), 0), channel < C)
    if requested[6]:
        tl.store(pa + param_slots[3] * stride, tl.sum(tl.where(valid, dhx, 0.), 0), channel < C)
    if requested[7]:
        dhw = dav * n + dauv * nu
        tl.store(pa + param_slots[4] * stride, tl.sum(tl.where(valid, dhw, 0.), 0), channel < C)
    if requested[8]:
        tl.store(pa + param_slots[5] * stride, tl.sum(tl.where(valid, dav, 0.), 0), channel < C)


def fused_triple_backward(saved, cotangents, needs, eps):
    inputs = saved[1]
    groups, rows, channels = inputs.shape
    requested = tuple(
        need and value is not None and (slot != 0 or any(seed is not None for seed in cotangents[1:]))
        for slot, (need, value) in enumerate(zip(needs, saved, strict=True))
    )
    if not any(requested) or all(seed is None for seed in cotangents):
        return (None,) * len(saved)
    result = [None] * len(saved)
    row_slots = (0, 1, 5)
    for slot in row_slots:
        if requested[slot]:
            result[slot] = inputs.new_empty(inputs.shape)

    parameter_slots = (2, 3, 4, 6, 7, 8)
    active = [slot for slot in parameter_slots if requested[slot]]
    indices = tuple(active.index(slot) if slot in active else -1 for slot in parameter_slots)
    rows_per_program = 8 if rows >= 32768 else 4
    tiles = triton.cdiv(rows, rows_per_program)
    partial = inputs.new_empty((len(active), groups, tiles, channels)) if active else inputs

    def pointers(values):
        return tuple(inputs if value is None else value.contiguous() for value in values)

    triple_backward_rows[(tiles, groups)](
        pointers(saved), pointers(cotangents), pointers(tuple(result[slot] for slot in row_slots)),
        partial, rows, tiles, C=channels, B_C=triton.next_power_of_2(channels),
        B_R=rows_per_program, EPS=eps,
        has_outer=tuple(value is not None for value in cotangents),
        has_inner=tuple(value is not None for value in saved[5:]),
        requested=requested, param_slots=indices, num_warps=8, num_stages=1,
    )
    if active:
        reduced = inputs.new_empty((len(active), groups, channels))
        reduce_columns[(triton.cdiv(channels, 8), groups, len(active))](
            partial, reduced, tiles, groups, C=channels,
            B_T=min(2048, triton.next_power_of_2(tiles)), B_C=8, num_warps=8, num_stages=1,
        )
        for slot, value in zip(active, reduced.unbind(), strict=True):
            result[slot] = value
    return tuple(result)


def directional_activation(inputs, input_bias, weight, bias, hx, hi, hw, hb, eps):
    values = inputs + input_bias[:, None, :]
    centered = values - values.mean(dim=-1, keepdim=True)
    rstd = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normalized = centered * rstd
    affine = normalized * weight[:, None, :] + bias[:, None, :]
    sigmoid = affine.sigmoid()

    direction = hx + hi[:, None, :]
    normalized_direction = rstd * (
        direction - direction.mean(dim=-1, keepdim=True)
        - normalized * (direction * normalized).mean(dim=-1, keepdim=True)
    )
    affine_direction = weight[:, None, :] * normalized_direction
    if hw is not None:
        affine_direction = affine_direction + normalized * hw[:, None, :]
    if hb is not None:
        affine_direction = affine_direction + hb[:, None, :]
    return affine_direction * sigmoid * (1 + affine * (1 - sigmoid))


def grouped_layer_norm_silu_triple_backward(saved, cotangents, needs, eps):
    inputs = saved[1]
    groups, rows, channels = inputs.shape
    if (
        inputs.is_cuda and inputs.dtype == torch.float32 and torch.version.hip is None
        and 0 < groups <= 65535 and rows > 0 and 0 < channels <= 256
        and inputs.numel() <= 2**30
    ):
        capability = torch.cuda.get_device_capability(inputs.device)
        small_fused = (
            groups == 1 and channels == 128
            and ((capability == (8, 0) and rows < 8192) or (capability == (9, 0) and rows < 1024))
        )
        ampere_grouped = (
            capability == (8, 0) and 2 <= groups <= 4 and channels <= 128
            and 4096 <= rows <= 16384
        )
        tiled = (
            (small_fused or ampere_grouped)
            and not any(value.is_neg() for value in (*saved, *cotangents) if value is not None)
        )
        if (capability == (9, 0) and rows >= 1024) or tiled or (
            capability == (8, 0)
            and groups == 1 and channels == 128 and 8192 <= rows <= 131072
            and all(value is not None for value in (*saved, *cotangents))
        ):
            with torch.cuda.device(inputs.device):
                return fused_triple_backward(saved, cotangents, needs, eps)
    return autograd_triple_backward(saved, cotangents, needs, eps)


def autograd_triple_backward(saved, cotangents, needs, eps):
    with torch.enable_grad():
        variables = tuple(
            None if value is None else value.detach().requires_grad_()
            for value in saved
        )
        dy, inputs, input_bias, weight, bias, hx, hi, hw, hb = variables
        direction = directional_activation(inputs, input_bias, weight, bias, hx, hi, hw, hb, eps)
        # Recompute normalization statistics from the inputs so their derivatives
        # contribute to the two terms of the double-backward VJP.
        gradients = torch.autograd.grad(
            direction, (inputs, input_bias, weight, bias), dy,
            create_graph=True, allow_unused=True,
        )
        active = [
            (value, seed)
            for value, seed in zip((direction, *gradients), cotangents, strict=True)
            if value is not None and value.requires_grad and seed is not None
        ]
        requested = tuple(need and value is not None for value, need in zip(variables, needs, strict=True))
        if not active or not any(requested):
            return (None,) * len(saved)
        outputs, seeds = zip(*active)
        targets = tuple(value for value, need in zip(variables, requested, strict=True) if need)
        result = iter(torch.autograd.grad(outputs, targets, seeds, allow_unused=True))
        return tuple(next(result) if need else None for need in requested)
