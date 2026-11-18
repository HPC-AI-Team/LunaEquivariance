"""Third reverse derivatives of the inverse-rotation receiver reduction."""

import torch
import triton
import triton.language as tl

from ...utils.layout import get_retained_layout
from ...utils.sector_layout import load_sector_value, store_sector_gradient
from .backward import sum_backward
from .double_backward import sum_double_backward
from .utils import use_degree_sum_node, use_receiver_double_backward, wigner_offset


def sum_triple_backward(cotangent, values, directions, gradients, receivers, row_ptr, dimensions):
    nodes, lmax, mmax, raw_complex, wigner_layout = dimensions
    if row_ptr is not None and not use_receiver_double_backward(
        cotangent.device, nodes=nodes, lmax=lmax, channels=values[2].shape[-1],
        wigner_layout=wigner_layout,
    ):
        row_ptr = None

    def hessian(seed, primals, tangents, mask):
        envelope, wigner, *messages = primals
        de, dw, *dm = tangents
        return sum_double_backward(
            seed, messages, envelope, wigner, receivers, nodes, de, dw, dm,
            lmax=lmax, mmax=mmax, raw_complex=raw_complex, wigner_layout=wigner_layout,
            row_ptr=row_ptr,
            output_mask=mask if row_ptr is not None else (True, True, True, True),
        )

    grad_seed, *grad_primals = gradients
    has_outer_primals = any(value is not None for value in grad_primals)
    direct_node = has_outer_primals and use_direct_sum_node(cotangent, values[2], row_ptr, dimensions)
    direct_messages = direct_node and use_direct_sum_messages(grad_seed, *grad_primals[:2])
    direct_parameters = direct_node and use_direct_sum_parameters(grad_seed, grad_primals)
    if use_joint_sum_directions(grad_seed, direct_messages, direct_parameters):
        grad_cotangent = sum_node_mixed_direction(values, directions, grad_primals, row_ptr, dimensions)
        grad_values, grad_directions = sum_edge_mixed_direction(
            cotangent, values, directions, grad_primals, grad_seed, receivers, dimensions,
            write_messages=True, write_parameters=True, write_directions=True,
        )
        return grad_cotangent, grad_values, grad_directions

    grad_cotangent = None
    grad_values = [None] * len(values)
    grad_directions = [None] * len(values)
    zeros = tuple(torch.zeros_like(value) for value in values)
    tangents = tuple(
        zero if direction is None else direction.contiguous()
        for direction, zero in zip(directions, zeros, strict=True)
    )
    if grad_seed is not None:
        grad_seed = grad_seed.contiguous()
        mask = (False, not direct_parameters, not direct_parameters, not direct_messages)
        if any(mask):
            grad_values = list(hessian(grad_seed, values, tangents, mask)[1:])
        envelope, wigner, *messages = values
        grad_directions = list(sum_backward(
            grad_seed, messages, envelope, wigner, receivers, nodes,
            lmax=lmax, mmax=mmax, raw_complex=raw_complex, wigner_layout=wigner_layout,
        ))

    if has_outer_primals:
        weights = tuple(
            zero if direction is None else direction.contiguous()
            for direction, zero in zip(grad_primals, zeros, strict=True)
        )
        for i, value in enumerate(hessian(cotangent, values, weights, (False, True, True, True))[1:]):
            grad_directions[i] = value if grad_directions[i] is None else grad_directions[i] + value

        # The receiver sum is linear in each of the envelope, rotation and
        # the entire message tuple. Replace one slot in the Hessian at a time.
        for start, end in ((0, 1), (1, 2), (2, len(values))):
            if all(value is None for value in grad_primals[start:end]):
                continue
            mask = (
                not direct_node,
                start != 0 and not direct_parameters,
                start != 1 and not direct_parameters,
                start != 2 and not direct_messages,
            )
            if not any(mask):
                continue
            result = hessian(
                cotangent,
                (*values[:start], *weights[start:end], *values[end:]),
                (*tangents[:start], *zeros[start:end], *tangents[end:]),
                mask,
            )
            if not direct_node:
                grad_cotangent = result[0] if grad_cotangent is None else grad_cotangent + result[0]
            for i, value in enumerate(result[1:]):
                if value is not None and not start <= i < end:
                    grad_values[i] = value if grad_values[i] is None else grad_values[i] + value
        if direct_node:
            grad_cotangent = sum_node_mixed_direction(values, directions, grad_primals, row_ptr, dimensions)
        if direct_messages or direct_parameters:
            fused, _ = sum_edge_mixed_direction(
                cotangent, values, directions, grad_primals, grad_seed, receivers, dimensions,
                write_messages=direct_messages, write_parameters=direct_parameters,
            )
            if direct_messages:
                grad_values[2:] = fused[2:]
            if direct_parameters:
                grad_values[:2] = fused[:2]

    return grad_cotangent, grad_values, tuple(
        None if direction is None else gradient
        for direction, gradient in zip(directions, grad_directions, strict=True)
    )


def use_direct_sum_node(cotangent, scalar, row_ptr, dimensions):
    _, lmax, mmax, _, layout = dimensions
    return (
        row_ptr is not None
        and 2 <= lmax <= 6 and (mmax, scalar.shape[-1], layout) == (2, 64, 6)
        and cotangent.is_cuda and cotangent.dtype == torch.float32 and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and torch.cuda.get_device_capability(cotangent.device) in ((8, 0), (9, 0))
    )


def use_direct_sum_messages(grad_seed, grad_envelope, grad_rotation):
    return sum(value is not None for value in (grad_seed, grad_envelope, grad_rotation)) >= 2


def use_direct_sum_parameters(grad_seed, grad_primals):
    sources = sum(value is not None for value in (grad_seed, *grad_primals[:2]))
    return sources + any(value is not None for value in grad_primals[2:]) >= 2


def use_joint_sum_directions(grad_seed, direct_messages, direct_parameters):
    return grad_seed is not None and direct_messages and direct_parameters


@triton.jit
def load_rotation_entry(ptr, strides, edge, row, full, valid, layout: tl.constexpr):
    offset = wigner_offset(edge, row, full, strides[0], strides[1], strides[2], layout)
    return tl.load(ptr + offset, valid, 0.).to(tl.float32)


@triton.jit
def load_message_row(ptr, strides, edge, row, channel, valid, channels: tl.constexpr,
                   width: tl.constexpr, order: tl.constexpr, raw: tl.constexpr):
    return load_sector_value(ptr, edge, row, channel, strides[0], strides[1], strides[2],
                             channels, width, order, raw, valid)


@triton.jit
def sum_node_mixed_direction_kernel(
    env, env_v, env_u, rotation, rotation_v, rotation_u, messages, messages_v, messages_u,
    row_ptr, output, rotation_stride: tl.constexpr, rotation_v_stride: tl.constexpr,
    rotation_u_stride: tl.constexpr, strides: tl.constexpr, strides_v: tl.constexpr,
    strides_u: tl.constexpr, has_ev: tl.constexpr, has_eu: tl.constexpr,
    has_wv: tl.constexpr, has_wu: tl.constexpr, has_mv: tl.constexpr, has_mu: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr, channels: tl.constexpr, layout: tl.constexpr,
    raw: tl.constexpr, block: tl.constexpr,
):
    # With p = env * rotation, the mixed direction is p_uv*m + p_u*m_v + p_v*m_u.
    node, full = tl.program_id(0), tl.program_id(1)
    degree = tl.sqrt(full.to(tl.float32)).to(tl.int32)
    center = degree * degree + degree
    channel = tl.arange(0, block)
    valid_channel = channel < channels
    begin, end = tl.load(row_ptr + node), tl.load(row_ptr + node + 1)
    total = tl.zeros((block,), tl.float32)
    for edge in range(begin, end):
        e = tl.load(env + edge).to(tl.float32)
        ev = tl.load(env_v + edge).to(tl.float32) if has_ev else 0.0
        eu = tl.load(env_u + edge).to(tl.float32) if has_eu else 0.0
        for order in tl.static_range(mmax + 1):
            for sign in tl.static_range(1 if order == 0 else 2):
                width = tl.constexpr(lmax - order + 1)
                row = degree - order + sign * width
                dense_row = center + (order if sign == 0 else -order)
                active = degree >= order
                valid = valid_channel & active
                w = load_rotation_entry(rotation, rotation_stride, edge, dense_row, full, active, layout)
                wv = (load_rotation_entry(rotation_v, rotation_v_stride, edge, dense_row, full, active, layout)
                      if has_wv else 0.0)
                wu = (load_rotation_entry(rotation_u, rotation_u_stride, edge, dense_row, full, active, layout)
                      if has_wu else 0.0)
                if (has_eu and has_wv) or (has_ev and has_wu):
                    m = load_message_row(messages[order], strides[order], edge, row, channel,
                                       valid, channels, width, order, raw)
                    total += (eu * wv + ev * wu) * m
                if has_mv[order] and (has_eu or has_wu):
                    mv = load_message_row(messages_v[order], strides_v[order], edge, row, channel,
                                        valid, channels, width, order, raw)
                    total += (eu * w + e * wu) * mv
                if has_mu[order] and (has_ev or has_wv):
                    mu = load_message_row(messages_u[order], strides_u[order], edge, row, channel,
                                        valid, channels, width, order, raw)
                    total += (ev * w + e * wv) * mu
    tl.store(output + (node * (lmax + 1)**2 + full) * channels + channel,
             total, mask=valid_channel)


@triton.jit
def sum_node_degree_mixed_direction_kernel(
    env, env_v, env_u, rotation, rotation_v, rotation_u, messages, messages_v, messages_u,
    row_ptr, output, rotation_stride: tl.constexpr, rotation_v_stride: tl.constexpr,
    rotation_u_stride: tl.constexpr, strides: tl.constexpr, strides_v: tl.constexpr,
    strides_u: tl.constexpr, has_ev: tl.constexpr, has_eu: tl.constexpr,
    has_wv: tl.constexpr, has_wu: tl.constexpr, has_mv: tl.constexpr, has_mu: tl.constexpr,
    lmax: tl.constexpr, mmax: tl.constexpr, channels: tl.constexpr,
    rotation_mmax: tl.constexpr, raw: tl.constexpr, degree: tl.constexpr,
    columns: tl.constexpr,
):
    node = tl.program_id(0)
    column = tl.arange(0, columns)[:, None]
    channel = tl.arange(0, channels)[None, :]
    angular_width: tl.constexpr = 2 * degree + 1
    retained: tl.constexpr = min(degree, rotation_mmax + 1)
    degree_offset: tl.constexpr = (
        retained * (4 * retained * retained - 1) // 3
        + (2 * rotation_mmax + 1) * (degree * degree - retained * retained)
    )
    # Each degree shares the same retained message rows across its output columns.
    begin = tl.load(row_ptr + node)
    end = tl.load(row_ptr + node + 1)
    valid = column < angular_width
    total = tl.zeros((columns, channels), tl.float32)
    for edge in range(begin, end):
        e = tl.load(env + edge).to(tl.float32)
        ev = tl.load(env_v + edge).to(tl.float32) if has_ev else 0.0
        eu = tl.load(env_u + edge).to(tl.float32) if has_eu else 0.0
        for order in tl.static_range(min(mmax, degree) + 1):
            width = lmax - order + 1
            for sign in tl.static_range(1 if order == 0 else 2):
                row = degree - order + sign * width
                retained_row = 0 if order == 0 else 2 * order - 1 + sign
                offset = degree_offset + retained_row * angular_width + column

                w = tl.load(rotation + edge * rotation_stride[0] + offset * rotation_stride[1],
                            valid, 0.0).to(tl.float32)
                if has_wv:
                    wv = tl.load(rotation_v + edge * rotation_v_stride[0] + offset * rotation_v_stride[1],
                                 valid, 0.0).to(tl.float32)
                else:
                    wv = 0.0
                if has_wu:
                    wu = tl.load(rotation_u + edge * rotation_u_stride[0] + offset * rotation_u_stride[1],
                                 valid, 0.0).to(tl.float32)
                else:
                    wu = 0.0
                if (has_eu and has_wv) or (has_ev and has_wu):
                    m = load_message_row(messages[order], strides[order], edge, row, channel,
                                         True, channels, width, order, raw)
                    total += (eu * wv + ev * wu) * m
                if has_mv[order] and (has_eu or has_wu):
                    mv = load_message_row(messages_v[order], strides_v[order], edge, row, channel,
                                          True, channels, width, order, raw)
                    total += (eu * w + e * wu) * mv
                if has_mu[order] and (has_ev or has_wv):
                    mu = load_message_row(messages_u[order], strides_u[order], edge, row, channel,
                                          True, channels, width, order, raw)
                    total += (ev * w + e * wv) * mu
    full: tl.constexpr = (lmax + 1)**2
    address = ((node * full + degree * degree + column) * channels + channel)
    tl.store(output + address, total, valid)



def direction_pointers(values, primals):
    return tuple(primal if value is None else value.resolve_neg().contiguous()
                 for value, primal in zip(values, primals, strict=True))


def sum_node_mixed_direction(primals, directions, outer, row_ptr, dimensions):
    nodes, lmax, mmax, raw, layout = dimensions
    e, w, *messages = primals
    ev, wv, *mv = direction_pointers(directions, primals)
    eu, wu, *mu = direction_pointers(outer, primals)
    channels = messages[0].shape[-1]
    full = (lmax + 1)**2
    result = e.new_empty((nodes, full, channels))
    operands = (
        e, ev, eu, w, wv, wu, tuple(messages), tuple(mv), tuple(mu), row_ptr, result,
        w.stride(), wv.stride(), wu.stride(), tuple(x.stride() for x in messages),
        tuple(x.stride() for x in mv), tuple(x.stride() for x in mu),
        directions[0] is not None, outer[0] is not None,
        directions[1] is not None, outer[1] is not None,
        tuple(x is not None for x in directions[2:]), tuple(x is not None for x in outer[2:]),
        lmax, mmax, channels,
    )
    if use_degree_sum_node(e.numel(), nodes, lmax, mmax, channels, layout):
        for degree in range(lmax + 1):
            sum_node_degree_mixed_direction_kernel[(nodes,)](
                *operands, layout - 4, raw, degree, triton.next_power_of_2(2 * degree + 1),
                num_warps=4,
            )
    else:
        sum_node_mixed_direction_kernel[(nodes, full)](
            *operands, layout, raw, triton.next_power_of_2(channels), num_warps=1,
        )
    return result


@triton.jit
def sum_message_mixed_direction_kernel(
    env, env_v, env_u, rotation, rotation_v, rotation_u, seed, cotangent, receivers, output, grad_direction,
    w_stride: tl.constexpr, wv_stride: tl.constexpr, wu_stride: tl.constexpr,
    out_stride: tl.constexpr, want_direction: tl.constexpr,
    has_seed: tl.constexpr, has_ev: tl.constexpr,
    has_eu: tl.constexpr, has_wv: tl.constexpr, has_wu: tl.constexpr,
    lmax: tl.constexpr, order: tl.constexpr, channels: tl.constexpr,
    width: tl.constexpr, layout: tl.constexpr, raw: tl.constexpr,
    block_rows: tl.constexpr, block_channels: tl.constexpr,
):
    # Apply p_v^T to the outer node seed and p_uv^T to the original
    # cotangent, where p = env * rotation.
    edge = tl.program_id(0)
    receiver = tl.load(receivers + edge)
    channel = tl.arange(0, block_channels)
    valid_channel = channel < channels
    has_first: tl.constexpr = has_seed and (has_ev or has_wv)
    has_mixed: tl.constexpr = (has_eu and has_wv) or (has_ev and has_wu)
    want_seed: tl.constexpr = has_first or (want_direction and has_seed)
    want_mixed: tl.constexpr = has_mixed or (want_direction and (has_eu or has_wu))
    ev = tl.load(env_v + edge).to(tl.float32) if has_ev else 0.0
    eu = tl.load(env_u + edge).to(tl.float32) if has_eu else 0.0
    e = tl.load(env + edge).to(tl.float32) if (has_first and has_wv) or want_direction else 0.0

    for degree in tl.static_range(order, lmax + 1):
        local = tl.arange(0, block_rows)
        full = degree * degree + local
        center = degree * degree + degree
        valid_row = local < 2 * degree + 1
        offset = (receiver * (lmax + 1)**2 + full[:, None]) * channels + channel[None, :]
        mask = valid_row[:, None] & valid_channel[None, :]
        if want_seed:
            a = tl.load(seed + offset, mask, 0.0).to(tl.float32)
        if want_mixed:
            g = tl.load(cotangent + offset, mask, 0.0).to(tl.float32)
        for sign in tl.static_range(1 if order == 0 else 2):
            row = degree - order + sign * width
            dense_row = center + (order if sign == 0 else -order)
            acc = tl.full((block_rows, block_channels), 0.0, tl.float32)
            if has_first:
                p_v = tl.full((block_rows,), 0.0, tl.float32)
                if has_ev:
                    w = load_rotation_entry(rotation, w_stride, edge, dense_row, full, valid_row, layout)
                    p_v += ev * w
                if has_wv:
                    wv = load_rotation_entry(rotation_v, wv_stride, edge, dense_row, full, valid_row, layout)
                    p_v += e * wv
                acc += p_v[:, None] * a
            if has_mixed:
                p_uv = tl.full((block_rows,), 0.0, tl.float32)
                if has_eu and has_wv:
                    wv = load_rotation_entry(rotation_v, wv_stride, edge, dense_row, full, valid_row, layout)
                    p_uv += eu * wv
                if has_ev and has_wu:
                    wu = load_rotation_entry(rotation_u, wu_stride, edge, dense_row, full, valid_row, layout)
                    p_uv += ev * wu
                acc += p_uv[:, None] * g
            grad = tl.sum(acc, axis=0)
            store_sector_gradient(
                output, grad, edge, row, channel, *out_stride,
                channels, width, order, raw, valid_channel,
            )

            if want_direction:
                d = tl.zeros((block_rows, block_channels), tl.float32)
                if has_seed or has_eu:
                    w = load_rotation_entry(rotation, w_stride, edge, dense_row, full, valid_row, layout)
                if has_seed:
                    d += (e * w)[:, None] * a
                if has_eu or has_wu:
                    p_u = tl.zeros((block_rows,), tl.float32)
                    if has_eu:
                        p_u += eu * w
                    if has_wu:
                        wu = load_rotation_entry(rotation_u, wu_stride, edge, dense_row, full, valid_row, layout)
                        p_u += e * wu
                    d += p_u[:, None] * g
                store_sector_gradient(
                    grad_direction, tl.sum(d, axis=0), edge, row, channel, *out_stride,
                    channels, width, order, raw, valid_channel,
                )


@triton.jit
def sum_parameter_mixed_direction_kernel(
    env, env_v, env_u, rotation, rotation_v, rotation_u,
    message, message_v, message_u, seed, cotangent, receivers, grad_env, grad_rotation, grad_v_env, grad_v_rotation,
    w_stride: tl.constexpr, wv_stride: tl.constexpr, wu_stride: tl.constexpr,
    m_stride: tl.constexpr, mv_stride: tl.constexpr, mu_stride: tl.constexpr,
    has_seed: tl.constexpr, has_ev: tl.constexpr, has_eu: tl.constexpr,
    has_wv: tl.constexpr, has_wu: tl.constexpr, has_mv: tl.constexpr, has_mu: tl.constexpr,
    want_v_env: tl.constexpr, want_v_rotation: tl.constexpr, lmax: tl.constexpr,
    order: tl.constexpr, channels: tl.constexpr, width: tl.constexpr,
    layout: tl.constexpr, raw: tl.constexpr, block_rows: tl.constexpr, block_channels: tl.constexpr,
):
    # The envelope and Wigner adjoints share the same three channel contractions.
    edge, row = tl.program_id(0), tl.program_id(1)
    positive = row < width
    degree = order + tl.where(positive, row, row - width)
    dense_row = degree * degree + degree + tl.where(positive, order, -order)
    local = tl.arange(0, block_rows)
    full = degree * degree + local
    channel = tl.arange(0, block_channels)
    live_full = local < 2 * degree + 1
    live_channel = channel < channels
    live = live_full[:, None] & live_channel[None, :]
    receiver = tl.load(receivers + edge)
    stride = (lmax + 1)**2 * channels
    offset = receiver * stride + full[:, None] * channels + channel[None, :]

    want_q0: tl.constexpr = has_ev or has_wv or want_v_env or want_v_rotation
    want_qv: tl.constexpr = has_seed and has_mv
    want_qu: tl.constexpr = has_mv and (has_eu or has_wu)
    want_q1: tl.constexpr = (want_v_env and has_wu) or (want_v_rotation and has_eu)
    if (want_q0 and has_seed) or want_qv:
        a = tl.load(seed + offset, live, 0.0).to(tl.float32)
    if (want_q0 and has_mu) or want_qu or want_q1:
        g = tl.load(cotangent + offset, live, 0.0).to(tl.float32)
    if (want_q0 and has_seed) or want_q1:
        m = load_message_row(message, m_stride, edge, row, channel, live_channel,
                             channels, width, order, raw)
    if want_q0 and (has_seed or has_mu):
        q0_terms = tl.zeros((block_rows, block_channels), tl.float32)
        if has_seed:
            q0_terms += a * m[None, :]
        if has_mu:
            mu = load_message_row(message_u, mu_stride, edge, row, channel, live_channel,
                                  channels, width, order, raw)
            q0_terms += g * mu[None, :]
        q0 = tl.sum(q0_terms, axis=1)
    else:
        q0 = tl.zeros((block_rows,), tl.float32)
    if want_qv or want_qu:
        mv = load_message_row(message_v, mv_stride, edge, row, channel, live_channel,
                              channels, width, order, raw)
    if want_qv:
        qv = tl.sum(a * mv[None, :], axis=1)
    else:
        qv = tl.zeros((block_rows,), tl.float32)
    if want_qu:
        qu = tl.sum(g * mv[None, :], axis=1)
    else:
        qu = tl.zeros((block_rows,), tl.float32)

    if want_q1:
        q1 = tl.sum(g * m[None, :], axis=1)

    dw = tl.zeros((block_rows,), tl.float32)
    if has_ev and (has_seed or has_mu):
        dw += tl.load(env_v + edge).to(tl.float32) * q0
    if want_qv:
        dw += tl.load(env + edge).to(tl.float32) * qv
    if want_qu and has_eu:
        dw += tl.load(env_u + edge).to(tl.float32) * qu
    address = wigner_offset(edge, dense_row, full, *w_stride, layout)
    tl.store(grad_rotation + address, dw, live_full)

    de = tl.zeros((block_rows,), tl.float32)
    if has_wv and (has_seed or has_mu):
        wv = load_rotation_entry(rotation_v, wv_stride, edge, dense_row, full, live_full, layout)
        de += wv * q0
    if want_qv:
        w = load_rotation_entry(rotation, w_stride, edge, dense_row, full, live_full, layout)
        de += w * qv
    if want_qu and has_wu:
        wu = load_rotation_entry(rotation_u, wu_stride, edge, dense_row, full, live_full, layout)
        de += wu * qu
    tl.atomic_add(grad_env + edge, tl.sum(de, axis=0), sem="relaxed")

    if want_v_rotation:
        dvw = tl.load(env + edge).to(tl.float32) * q0
        if has_eu:
            dvw += tl.load(env_u + edge).to(tl.float32) * q1
        tl.store(grad_v_rotation + address, dvw, live_full)
    if want_v_env:
        w = load_rotation_entry(rotation, w_stride, edge, dense_row, full, live_full, layout)
        dve = w * q0
        if has_wu:
            wu = load_rotation_entry(rotation_u, wu_stride, edge, dense_row, full, live_full, layout)
            dve += wu * q1
        tl.atomic_add(grad_v_env + edge, tl.sum(dve, axis=0), sem="relaxed")


def sum_edge_mixed_direction(
    cotangent, primals, directions, outer, grad_seed, receivers, dimensions,
    *, write_messages, write_parameters, write_directions=False,
):
    _, lmax, _, raw, layout = dimensions
    e, w, *messages = primals
    ev, wv, *mv = direction_pointers(directions, primals)
    eu, wu, *mu = direction_pointers(outer, primals)
    channels = messages[0].shape[-1]
    cotangent = cotangent.resolve_neg().contiguous()
    has_seed = grad_seed is not None
    grad_seed = cotangent if grad_seed is None else grad_seed.resolve_neg().contiguous()
    grad_e = torch.zeros_like(e) if write_parameters else None
    grad_w = torch.empty_like(w) if write_parameters else None
    grad_v_e = torch.zeros_like(e) if write_directions and directions[0] is not None else None
    grad_v_w = torch.empty_like(w) if write_directions and directions[1] is not None else None
    grad_messages = tuple(torch.empty_like(message) if write_messages else None for message in messages)
    grad_v_messages = tuple(
        torch.empty_like(message) if write_directions and v is not None else None
        for message, v in zip(messages, directions[2:], strict=True)
    )
    block_rows = triton.next_power_of_2(2 * lmax + 1)
    block_channels = triton.next_power_of_2(channels)
    for order, (message, grad_m, grad_v_m) in enumerate(zip(messages, grad_messages, grad_v_messages, strict=True)):
        width = lmax - order + 1
        if grad_m is not None:
            sum_message_mixed_direction_kernel[(e.numel(),)](
                e, ev, eu, w, wv, wu, grad_seed, cotangent, receivers, grad_m,
                grad_m if grad_v_m is None else grad_v_m,
                w.stride(), wv.stride(), wu.stride(), grad_m.stride(), grad_v_m is not None,
                has_seed, directions[0] is not None, outer[0] is not None,
                directions[1] is not None, outer[1] is not None,
                lmax, order, channels, width, layout, raw, block_rows, block_channels, num_warps=2,
            )
        if write_parameters:
            rows = width if order == 0 else 2 * width
            sum_parameter_mixed_direction_kernel[(e.numel(), rows)](
                e, ev, eu, w, wv, wu, message, mv[order], mu[order], grad_seed, cotangent, receivers,
                grad_e, grad_w, grad_e if grad_v_e is None else grad_v_e, grad_w if grad_v_w is None else grad_v_w,
                w.stride(), wv.stride(), wu.stride(), message.stride(), mv[order].stride(), mu[order].stride(),
                has_seed, directions[0] is not None, outer[0] is not None,
                directions[1] is not None, outer[1] is not None,
                directions[2 + order] is not None, outer[2 + order] is not None,
                grad_v_e is not None, grad_v_w is not None, lmax, order, channels, width,
                layout, raw, block_rows, block_channels, num_warps=2,
            )
    return [grad_e, grad_w, *grad_messages], (grad_v_e, grad_v_w, *grad_v_messages)


def attention_sum(messages, logits, envelope, wigner, receivers, dimensions):
    nodes, eps, lmax, mmax = dimensions
    edges, _, channels = messages[0].shape
    heads = logits.shape[1]
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)

    coefficients = [messages[0]]
    for order, message in enumerate(messages[1:], start=1):
        width = lmax + 1 - order
        if message.shape[-1] == width * channels:
            coefficients.append(message.reshape(edges, 2 * width, channels))
        else:
            raw = message.reshape(edges, 2, 2, width, channels)
            coefficients.extend((raw[:, 0, 0] - raw[:, 1, 1], raw[:, 1, 0] + raw[:, 0, 1]))
    local = torch.cat(coefficients, dim=1)

    # Neither the location nor the value of the stabilizing maximum contributes
    # a derivative. The envelope weights both the normalization and the value.
    index = receivers[:, None].expand_as(logits)
    maxima = logits.new_full((nodes, heads), -torch.inf).scatter_reduce(
        0, index, logits.detach(), reduce="amax",
    )
    scores = (logits - maxima[receivers]).exp() * envelope
    denominator = logits.new_zeros(nodes, heads).index_add(0, receivers, scores)
    probabilities = scores / (denominator[receivers] + eps)
    factors = probabilities * envelope
    local = local.reshape(edges, spec.reduced_components, heads, channels // heads)
    local = (local * factors[:, None, :, None]).reshape(edges, spec.reduced_components, channels)

    global_parts = []
    for degree, rows in enumerate(spec.degree_rows):
        start = spec.compact_row_offsets[rows[0]]
        width = 2 * degree + 1
        columns = wigner[:, start : start + len(rows) * width].reshape(edges, len(rows), width)
        global_parts.append(torch.bmm(columns.transpose(1, 2), local[:, rows]))
    rotated = torch.cat(global_parts, dim=1)
    return rotated.new_zeros(nodes, spec.full_components, channels).index_add(0, receivers, rotated)


def attention_triple_backward(saved, gradients, needs, receivers, dimensions):
    if not any(needs):
        return (None,) * len(saved)
    if use_degree_attention_triple_backward(saved, dimensions):
        return attention_degree_triple_backward(saved, gradients, needs, receivers, dimensions)
    return attention_triple_backward_reference(saved, gradients, needs, receivers, dimensions)


def use_degree_attention_triple_backward(saved, dimensions):
    _, _, lmax, mmax = dimensions
    if (lmax, mmax) != (4, 2):
        return False
    seed, coefficients, _, _, logits = saved[:5]
    edges = coefficients.shape[0]
    device = seed.device
    return (
        8192 <= edges <= 32768
        and coefficients.shape[1:] == (5, 128)
        and logits.shape == (edges, 8)
        and device.type == "cuda"
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and all(value.dtype == torch.float32 and value.device == device for value in saved)
        and torch.cuda.get_device_capability(device) in ((8, 0), (9, 0))
    )


def attention_triple_backward_reference(saved, gradients, needs, receivers, dimensions):
    if not any(needs):
        return (None,) * len(saved)
    with torch.enable_grad():
        variables = tuple(value.detach().requires_grad_() for value in saved)
        count = (len(variables) - 1) // 2
        seed = variables[0]
        primals, tangents = variables[1:count + 1], variables[count + 1:]
        *messages, logits, envelope, wigner = primals
        values = attention_sum(messages, logits, envelope, wigner, receivers, dimensions)
        first = torch.autograd.grad(values, primals, seed, create_graph=True)
        second = torch.autograd.grad(
            first, (seed, *primals), tangents, create_graph=True, materialize_grads=True,
        )
        active = [
            (value, gradient)
            for value, gradient in zip(second, gradients, strict=True)
            if value.requires_grad and gradient is not None
        ]
        if not active:
            return (None,) * len(saved)
        outputs, cotangents = zip(*active)
        targets = tuple(value for value, need in zip(variables, needs, strict=True) if need)
        result = iter(torch.autograd.grad(outputs, targets, cotangents, allow_unused=True))
        return tuple(next(result) if need else None for need in needs)


def softmax_jets(logits, env, logits_v, env_v, receivers, nodes, eps, logits_u=None, env_u=None):
    with torch.no_grad():
        index = receivers[:, None].expand_as(logits)
        maximum = logits.new_full((nodes, logits.shape[1]), -torch.inf).scatter_reduce_(
            0, index, logits, reduce="amax",
        )
    exp = (logits - maximum[receivers]).exp()

    def receiver_sum(score):
        return torch.zeros_like(maximum).index_add(0, receivers, score)[receivers]

    # The envelope affects both the softmax normalization and the edge value.
    s = exp * env
    s_v = exp * (env * logits_v + env_v)
    den = receiver_sum(s) + eps
    den_v = receiver_sum(s_v)
    q = s * env / den
    q_v = (s_v * env + s * env_v - q * den_v) / den
    if logits_u is None:
        return q, None, q_v, None
    s_u = exp * (env * logits_u + env_u)
    s_uv = s_u * logits_v + exp * logits_u * env_v
    den_u = receiver_sum(s_u)
    den_uv = receiver_sum(s_uv)
    q_u = (s_u * env + s * env_u - q * den_u) / den
    q_uv = (s_uv * env + s_u * env_v + s_v * env_u - q * den_uv - q_u * den_v - q_v * den_u) / den
    return q, q_u, q_v, q_uv


def sector_view(message, edges, columns, channels):
    planes = message.shape[-1] // (columns * channels)
    return message.reshape(edges, 2, planes, columns, channels)


def degree_messages(sectors, degree, lmax, mmax, prototype):
    first = None if sectors[0] is None else sectors[0][:, degree]
    zero = torch.zeros_like(prototype) if any(value is None for value in sectors[:degree + 1]) else None
    rows = [zero if first is None else first]
    edges, channels = prototype.shape
    for order in range(1, min(mmax, degree) + 1):
        message = sectors[order]
        if message is None:
            rows.extend((zero, zero))
            continue
        value = sector_view(message, edges, lmax + 1 - order, channels)[:, :, :, degree - order]
        if value.shape[2] == 1:
            rows.extend((value[:, 0, 0], value[:, 1, 0]))
        else:
            rows.extend((value[:, 0, 0] - value[:, 1, 1], value[:, 1, 0] + value[:, 0, 1]))
    return torch.stack(rows, dim=1)


def write_degree(targets, gradient, degree, lmax, mmax):
    if targets[0] is not None:
        targets[0][:, degree].copy_(gradient[:, 0])
    edges, _, channels = gradient.shape
    for order in range(1, min(mmax, degree) + 1):
        if targets[order] is None:
            continue
        target = sector_view(targets[order], edges, lmax + 1 - order, channels)[:, :, :, degree - order]
        positive, negative = gradient[:, 2 * order - 1], gradient[:, 2 * order]
        target[:, 0, 0].copy_(positive)
        if target.shape[2] == 1:
            target[:, 1, 0].copy_(negative)
        else:
            target[:, 0, 1].copy_(negative)
            target[:, 1, 0].copy_(negative)
            target[:, 1, 1].copy_(-positive)


def attention_degree_vjp(messages, rotations, seeds, jets, jet_grads, receivers,
                         terms, want_m, want_w, p_target):
    message_grads = [None, None]
    rotation_grads = [None, None]
    seed_grad = None
    for pairs, contractions in terms:
        weighted = projected = None
        grad_message_factor = any((m < 2 and want_m[m]) or jet_grads[q] is not None for m, q in pairs)
        for seed_index, rotation_index in contractions:
            node_seed = seeds[seed_index]
            rotation = rotations[rotation_index]
            grad_w = rotation_index < 2 and want_w[rotation_index]
            grad_p = seed_index == 1 and p_target is not None
            if (grad_w or grad_p) and weighted is None:
                m, q = pairs[0]
                weighted = messages[m] * jets[q]
                for m, q in pairs[1:]:
                    weighted.addcmul_(messages[m], jets[q])
                weighted = weighted.flatten(2)
            if grad_w:
                dw = torch.bmm(weighted, node_seed.transpose(1, 2))
                if rotation_grads[rotation_index] is None:
                    rotation_grads[rotation_index] = dw
                else:
                    rotation_grads[rotation_index].add_(dw)
            if grad_p:
                dp = torch.bmm(rotation.transpose(1, 2), weighted)
                if seed_grad is None:
                    seed_grad = dp
                else:
                    seed_grad.add_(dp)
            if grad_message_factor:
                if projected is None:
                    projected = torch.bmm(rotation, node_seed).reshape_as(messages[0])
                else:
                    projected.add_(torch.bmm(rotation, node_seed).reshape_as(projected))

        if grad_message_factor:
            for m, q in pairs:
                if m < 2 and want_m[m]:
                    if message_grads[m] is None:
                        message_grads[m] = projected * jets[q]
                    else:
                        message_grads[m].addcmul_(projected, jets[q])
                if jet_grads[q] is not None:
                    jet_grads[q].add_((projected * messages[m]).sum((1, 3)))

    if seed_grad is not None:
        p_target.index_add_(0, receivers, seed_grad)
    return message_grads, rotation_grads


@torch.no_grad()
def attention_degree_triple_backward(saved, gradients, needs, receivers, dimensions):
    if not any(needs) or all(gradient is None for gradient in gradients):
        return (None,) * len(saved)
    nodes, eps, lmax, mmax = dimensions
    count = mmax + 4
    p, *values = saved
    *messages, logits, env, wigner = values[:count]
    *messages_v, logits_v, env_v, wigner_v = values[count:]
    r, *messages_u, logits_u, env_u, wigner_u = gradients
    has_r = r is not None
    has_u_messages = any(value is not None for value in messages_u)
    has_u_softmax = logits_u is not None or env_u is not None
    has_u_wigner = wigner_u is not None
    mixed = has_u_messages or has_u_softmax or has_u_wigner

    def allocate(value):
        return torch.empty_like(value, memory_format=torch.contiguous_format)

    want_p = needs[0] and mixed
    grad_p = torch.zeros_like(p) if want_p else None
    need_messages = has_r or has_u_softmax or has_u_wigner
    need_wigner = has_r or has_u_softmax or has_u_messages
    grad_messages = tuple(
        tuple(allocate(value) if need_messages and needs[offset + i] else None for i, value in enumerate(group))
        for offset, group in ((1, messages), (1 + count, messages_v))
    )
    grad_wigner = tuple(
        allocate(value) if need_wigner and needs[slot] else None
        for slot, value in ((count, wigner), (2 * count, wigner_v))
    )
    want_m = tuple(any(value is not None for value in group) for group in grad_messages)
    want_w = tuple(value is not None for value in grad_wigner)

    softmax_slots = (count - 2, count - 1, 2 * count - 2, 2 * count - 1)
    with torch.enable_grad():
        softmax_values = tuple(saved[slot].detach().requires_grad_(needs[slot]) for slot in softmax_slots)
        if has_u_softmax:
            logits_u = torch.zeros_like(logits) if logits_u is None else logits_u.detach()
            env_u = torch.zeros_like(env) if env_u is None else env_u.detach()
        q = softmax_jets(*softmax_values, receivers, nodes, eps, logits_u, env_u)
    q_grads = tuple(torch.zeros_like(value) if value is not None and value.requires_grad else None for value in q)
    q_view = tuple(None if value is None else value.detach()[:, None, :, None] for value in q)

    # Differentiate <r, DF[v]> + <p, D²F[u,v]> with respect to p, inputs, and v.
    # Messages and rotations are indexed as primal, v, u. Factors are q, q_u,
    # q_v, q_uv. A weighted message can serve several contractions.
    first_v = ((0, 2), (1, 0))
    terms = []
    first_contractions = ((0, 0),) if has_r else ()
    if has_u_wigner:
        first_contractions += ((1, 2),)
    if first_contractions:
        terms.append((first_v, first_contractions))
    if has_r:
        terms.append((((0, 0),), ((0, 1),)))
    if has_u_softmax or has_u_messages:
        primal = ((0, 3), (1, 1)) if has_u_softmax else ()
        direction = ((0, 1),) if has_u_softmax else ()
        if has_u_messages:
            primal += ((2, 2),)
            direction += ((2, 0),)
        terms.extend(((primal, ((1, 0),)), (direction, ((1, 1),))))

    edges, _, channels = messages[0].shape
    heads = logits.shape[1]
    layout = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    for degree, rows in enumerate(layout.degree_rows):
        width = 2 * degree + 1
        start = layout.compact_row_offsets[rows[0]]
        cols = slice(start, start + len(rows) * width)
        node_rows = slice(degree * degree, (degree + 1) ** 2)
        rotations = tuple(None if value is None else value[:, cols].reshape(edges, len(rows), width)
                          for value in (wigner, wigner_v, wigner_u))
        seeds = (r[:, node_rows][receivers] if has_r else None, p[:, node_rows][receivers] if mixed else None)
        degree_values = tuple(
            degree_messages(group, degree, lmax, mmax, messages[0][:, 0]).reshape(
                edges, len(rows), heads, channels // heads,
            )
            for group in ((messages, messages_v, messages_u) if has_u_messages else (messages, messages_v))
        )
        dm, dw = attention_degree_vjp(
            degree_values, rotations, seeds, q_view, q_grads, receivers,
            terms, want_m, want_w, None if grad_p is None else grad_p[:, node_rows],
        )
        for targets, value in zip(grad_messages, dm, strict=True):
            if value is not None:
                write_degree(targets, value.flatten(2), degree, lmax, mmax)
        for target, value in zip(grad_wigner, dw, strict=True):
            if value is not None:
                target[:, cols].copy_(value.flatten(1))

    grad_softmax = [None] * len(softmax_values)
    active_jets = [(value, grad) for value, grad in zip(q, q_grads, strict=True) if grad is not None]
    if active_jets:
        selectors = [i for i, value in enumerate(softmax_values) if value.requires_grad]
        outputs, cotangents = zip(*active_jets, strict=True)
        derivatives = torch.autograd.grad(outputs, [softmax_values[i] for i in selectors], cotangents,
                                          allow_unused=True)
        for index, value in zip(selectors, derivatives, strict=True):
            grad_softmax[index] = value
    return (grad_p, *grad_messages[0], *grad_softmax[:2], grad_wigner[0],
            *grad_messages[1], *grad_softmax[2:], grad_wigner[1])
