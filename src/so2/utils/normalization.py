"""Row normalization derivatives and fixed-order parameter reduction."""

import triton
import triton.language as tl


@triton.jit
def normalized_directions(x, u, v, valid, inv_c, EPS: tl.constexpr):
    # Scaling the directions by r avoids explicit high powers of r.
    p = tl.where(valid, x - tl.sum(x, axis=1)[:, None] * inv_c, 0.)
    u = tl.where(valid, u - tl.sum(u, axis=1)[:, None] * inv_c, 0.)
    v = tl.where(valid, v - tl.sum(v, axis=1)[:, None] * inv_c, 0.)
    r = tl.rsqrt(tl.sum(p * p, axis=1)[:, None] * inv_c + EPS)
    n, u, v = r * p, r * u, r * v
    alpha = tl.sum(n * u, axis=1)[:, None] * inv_c
    beta = tl.sum(n * v, axis=1)[:, None] * inv_c
    gamma = tl.sum(u * v, axis=1)[:, None] * inv_c
    cross = 3 * alpha * beta - gamma
    nu, nv = u - alpha * n, v - beta * n
    nuv = cross * n - beta * u - alpha * v
    return n, nu, nv, nuv, (r, n, u, v, alpha, beta, cross)


@triton.jit
def normalized_directions_vjp(state, dn, dnu, dnv, dnuv, valid, inv_c):
    r, n, u, v, alpha, beta, cross = state
    # Reverse nu, nv, nuv through their row moments.
    dcross = tl.sum(dnuv * n, axis=1)[:, None]
    dalpha = 3 * beta * dcross - tl.sum(dnu * n + dnuv * v, axis=1)[:, None]
    dbeta = 3 * alpha * dcross - tl.sum(dnv * n + dnuv * u, axis=1)[:, None]
    dn = dn - alpha * dnu - beta * dnv + cross * dnuv + inv_c * (dalpha * u + dbeta * v)
    du = dnu - beta * dnuv + inv_c * (dalpha * n - dcross * v)
    dv = dnv - alpha * dnuv + inv_c * (dbeta * n - dcross * u)
    # The same r scales all three vectors. Only x changes r.
    dr_scaled = tl.sum(dn * n + du * u + dv * v, axis=1)[:, None] * inv_c
    dp = tl.where(valid, r * (dn - n * dr_scaled), 0.)
    dv = tl.where(valid, r * dv, 0.)
    dx = dp - tl.sum(dp, axis=1)[:, None] * inv_c
    dhx = dv - tl.sum(dv, axis=1)[:, None] * inv_c
    return dx, dhx


@triton.jit
def reduce_columns(partial, output, tiles, groups, C: tl.constexpr, B_T: tl.constexpr, B_C: tl.constexpr):
    group, slot = tl.program_id(1), tl.program_id(2)
    column = tl.program_id(0) * B_C + tl.arange(0, B_C)
    tile = tl.arange(0, B_T)
    base = (slot * groups + group) * tiles * C + column[None, :]
    accumulator = tl.full((B_T, B_C), 0., tl.float32)
    for start in range(0, tiles, B_T):
        index = start + tile
        valid = (index[:, None] < tiles) & (column[None, :] < C)
        accumulator += tl.load(partial + base + index[:, None] * C, valid, 0.)
    tl.store(output + (slot * groups + group) * C + column, tl.sum(accumulator, axis=0), column < C)
