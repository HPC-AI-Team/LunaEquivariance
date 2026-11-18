"""Third reverse derivatives of the attention score."""

import torch
import triton
import triton.language as tl

from ...utils.normalization import normalized_directions, normalized_directions_vjp, reduce_columns


@triton.jit
def attention_third_rows(
    saved, outer, outputs, partial, rows, heads, tiles,
    C: tl.constexpr, B_C: tl.constexpr, B_R: tl.constexpr, EPS: tl.constexpr,
    row_strides: tl.constexpr, score_strides: tl.constexpr,
    has: tl.constexpr, requested: tl.constexpr, param_slots: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    row = tile * B_R + tl.arange(0, B_R)
    c = tl.arange(0, B_C)
    rmask, cmask = row < rows, c < C
    valid = rmask[:, None] & cmask[None, :]
    output_index = (row * heads + head)[:, None] * C + c[None, :]
    dot_index = head * C + c
    inv_c = 1. / C

    dy = tl.load(saved[0] + row * score_strides[0][0] + head * score_strides[0][1], rmask, 0.)[:, None]
    x_index = row[:, None] * row_strides[0][0] + head * row_strides[0][1] + c[None, :] * row_strides[0][2]
    x = tl.load(saved[1] + x_index, valid, 0.)
    w = tl.load(saved[2] + c, cmask, 0.)[None, :]
    b = tl.load(saved[3] + c, cmask, 0.)[None, :]
    dot = tl.load(saved[4] + dot_index, cmask, 0.)[None, :]
    udot = tl.full((1, B_C), 0., tl.float32)
    vdot = tl.full((1, B_C), 0., tl.float32)
    if has[4]:
        udot = tl.load(outer[4] + dot_index, cmask, 0.)[None, :]
    if has[8]:
        vdot = tl.load(saved[8] + dot_index, cmask, 0.)[None, :]

    u = tl.full((B_R, B_C), 0., tl.float32)
    v = tl.full((B_R, B_C), 0., tl.float32)
    if has[1]:
        u_idx = row[:, None] * row_strides[1][0] + head * row_strides[1][1] + c[None, :] * row_strides[1][2]
        u = tl.load(outer[1] + u_idx, valid, 0.)
    if has[5]:
        v_idx = row[:, None] * row_strides[2][0] + head * row_strides[2][1] + c[None, :] * row_strides[2][2]
        v = tl.load(saved[5] + v_idx, valid, 0.)

    n, nu, nv, nuv, state = normalized_directions(x, u, v, valid, inv_c, EPS)
    a = w * n + b
    au, av, auv = w * nu, w * nv, w * nuv
    if has[2]:
        wu = tl.load(outer[2] + c, cmask, 0.)[None, :]
        au += wu * n
        auv += wu * nv
    if has[3]:
        au += tl.load(outer[3] + c, cmask, 0.)[None, :]
    if has[6]:
        wv = tl.load(saved[6] + c, cmask, 0.)[None, :]
        av += wv * n
        auv += wv * nu
    if has[7]:
        av += tl.load(saved[7] + c, cmask, 0.)[None, :]

    tangent = 2 * tl.sigmoid(a) - 1
    sech_squared = 1 - tangent * tangent
    f = 0.6 * a + 0.4 * a * tangent
    f1 = 0.6 + 0.4 * tangent + 0.2 * a * sech_squared
    f2 = sech_squared * (0.4 - 0.2 * a * tangent)
    f3 = sech_squared * (-0.6 * tangent + 0.1 * a * (3 * tangent * tangent - 1))
    z = tl.full((B_R, 1), 0., tl.float32)
    if has[0]:
        z = tl.load(outer[0] + row * score_strides[1][0] + head * score_strides[1][1], rmask, 0.)[:, None]

    duv = f2 * au * av + f1 * auv
    da = dy * (dot * (f3 * au * av + f2 * auv) + vdot * f2 * au + udot * f2 * av)
    da += z * (dot * f2 * av + vdot * f1)
    dau = dy * (dot * f2 * av + vdot * f1)
    dav = dy * (dot * f2 * au + udot * f1) + z * dot * f1
    dauv = dy * dot * f1
    if requested[0]:
        ddy = tl.sum(tl.where(valid, dot * duv + f1 * (vdot * au + udot * av), 0.), axis=1)
        tl.store(outputs[0] + row * heads + head, ddy, rmask)

    dn, dnu, dnv, dnuv = w * da, w * dau, w * dav, w * dauv
    if has[2]:
        dn += wu * dau
        dnv += wu * dauv
    if has[6]:
        dn += wv * dav
        dnu += wv * dauv

    dx, dhx = normalized_directions_vjp(state, dn, dnu, dnv, dnuv, valid, inv_c)
    if requested[1]:
        tl.store(outputs[1] + output_index, dx, valid)
    if requested[5]:
        tl.store(outputs[2] + output_index, dhx, valid)

    pa = partial + (head * tiles + tile) * C + c
    stride = heads * tiles * C
    if requested[2]:
        dw = da * n + dau * nu + dav * nv + dauv * nuv
        tl.store(pa + param_slots[0] * stride, tl.sum(tl.where(valid, dw, 0.), 0), cmask)
    if requested[3]:
        tl.store(pa + param_slots[1] * stride, tl.sum(tl.where(valid, da, 0.), 0), cmask)
    if requested[4]:
        dd = dy * duv + z * f1 * av
        tl.store(pa + param_slots[2] * stride, tl.sum(tl.where(valid, dd, 0.), 0), cmask)
    if requested[6]:
        dhw = dav * n + dauv * nu
        tl.store(pa + param_slots[3] * stride, tl.sum(tl.where(valid, dhw, 0.), 0), cmask)
    if requested[7]:
        tl.store(pa + param_slots[4] * stride, tl.sum(tl.where(valid, dav, 0.), 0), cmask)
    if requested[8]:
        dhd = dy * f1 * au + z * f
        tl.store(pa + param_slots[5] * stride, tl.sum(tl.where(valid, dhd, 0.), 0), cmask)


def fused_triple_backward(saved, outer, needs, eps):
    """Use ordered per-head reduction, shared LayerNorm vectors and per-head dot vectors."""
    x = saved[1]
    rows, heads, c = x.shape
    rows_per_program = 8
    requested = [need and value is not None for need, value in zip(needs, saved, strict=True)]
    requested[0] &= any(seed is not None for seed in outer[1:])
    requested[4] &= any(seed is not None for seed in outer[:4])
    requested[8] &= any(seed is not None for seed in outer[:4])
    if not any(requested) or all(seed is None for seed in outer):
        return (None,) * len(saved)
    if not rows:
        return tuple(torch.zeros_like(value) if want else None for value, want in zip(saved, requested, strict=True))
    outputs = [None] * len(saved)
    for position in (0, 1, 5):
        if requested[position]:
            outputs[position] = x.new_empty(saved[position].shape)
    shared = (2, 3, 4, 6, 7, 8)
    active = [position for position in shared if requested[position]]
    indices = tuple(active.index(position) if position in active else -1 for position in shared)
    tiles = triton.cdiv(rows, rows_per_program)
    partial = x.new_empty((len(active), heads, tiles, c)) if active else x

    def prepare(values, preserve):
        return tuple(None if value is None else
                     value.resolve_neg() if i in preserve else value.resolve_neg().contiguous()
                     for i, value in enumerate(values))

    saved = prepare(saved, (0, 1, 5))
    outer = prepare(outer, (0, 1))
    dummy = lambda value: x if value is None else value
    strides = lambda value, dims: (0,) * dims if value is None else value.stride()
    attention_third_rows[(tiles, heads)](
        tuple(map(dummy, saved)), tuple(map(dummy, outer)),
        tuple(dummy(outputs[position]) for position in (0, 1, 5)), partial, rows, heads, tiles,
        C=c, B_C=triton.next_power_of_2(c), B_R=rows_per_program, EPS=eps,
        row_strides=tuple(strides(value, 3) for value in (saved[1], outer[1], saved[5])),
        score_strides=tuple(strides(value, 2) for value in (saved[0], outer[0])),
        has=tuple(value is not None for value in (*outer, *saved[5:])),
        requested=tuple(requested), param_slots=indices, num_warps=4, num_stages=1,
    )
    if active:
        parameters = x.new_empty((len(active), heads, c))
        reduce_columns[(triton.cdiv(c, 8), heads, len(active))](
            partial, parameters, tiles, heads, C=c, B_T=min(2048, triton.next_power_of_2(tiles)),
            B_C=8, num_warps=8, num_stages=1,
        )
        for position, parameter in zip(active, parameters.unbind(), strict=True):
            outputs[position] = parameter if position in (4, 8) else parameter.sum(dim=0)
    return tuple(outputs)


def triple_backward(saved, cotangents, needs, eps):
    inputs = saved[1]
    edges, heads, channels = inputs.shape
    if (
        inputs.is_cuda and torch.version.hip is None and not torch.compiler.is_compiling()
        and 8192 <= edges <= 131072 and (heads, channels) == (8, 32)
        and inputs.stride(-1) == 1
        and torch.cuda.get_device_capability(inputs.device) in ((8, 0), (9, 0))
        and all(
            value is None or (
                value.dtype == torch.float32 and value.device == inputs.device
                and sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) < 2**31
            )
            for value in (*saved, *cotangents)
        )
    ):
        return fused_triple_backward(saved, cotangents, needs, eps)
    return autograd_triple_backward(saved, cotangents, needs, eps)


def directional_scores(inputs, weight, bias, dot, hx, hw, hb, hd, eps):
    centered = inputs - inputs.mean(dim=-1, keepdim=True)
    rstd = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normalized = centered * rstd
    affine = normalized * weight + bias
    tangent = (affine * 0.5).tanh()
    derivative = 0.6 + 0.4 * tangent + 0.2 * affine * (1 - tangent.square())

    direction = 0.0
    if hx is not None:
        normalized_direction = rstd * (
            hx - hx.mean(dim=-1, keepdim=True)
            - normalized * (hx * normalized).mean(dim=-1, keepdim=True)
        )
        direction = weight * normalized_direction
    if hw is not None:
        direction = direction + normalized * hw
    if hb is not None:
        direction = direction + hb
    scores = dot * derivative * direction
    if hd is not None:
        activation = 0.6 * affine + 0.4 * affine * tangent
        scores = scores + activation * hd
    return scores.sum(dim=-1)


def autograd_triple_backward(saved, cotangents, needs, eps):
    with torch.enable_grad():
        variables = tuple(
            None if value is None else value.detach().requires_grad_()
            for value in saved
        )
        grad_output, inputs, weight, bias, dot, hx, hw, hb, hd = variables
        scores = directional_scores(inputs, weight, bias, dot, hx, hw, hb, hd, eps)
        # Double backward returns the score direction and its VJP with the
        # original output cotangent. Differentiate both against the primals
        # and the two sets of earlier directions.
        gradients = torch.autograd.grad(
            scores, (inputs, weight, bias, dot), grad_output,
            create_graph=True, allow_unused=True,
        )
        active = [
            (gradient, cotangent)
            for gradient, cotangent in zip((scores, *gradients), cotangents, strict=True)
            if gradient is not None and gradient.requires_grad and cotangent is not None
        ]
        requested = tuple(need and value is not None for value, need in zip(variables, needs, strict=True))
        if not active or not any(requested):
            return (None,) * len(saved)
        outputs, directions = zip(*active)
        targets = tuple(value for value, need in zip(variables, requested, strict=True) if need)
        result = iter(torch.autograd.grad(outputs, targets, directions, allow_unused=True))
        return tuple(next(result) if need else None for need in requested)
