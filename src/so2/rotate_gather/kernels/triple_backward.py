import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from ...utils.layout import get_angular_layout, get_retained_layout
from ...utils.tensor_contracts import allow_atomic_reduction
from .utils import (
    add_node_direction, gl, gluon, launch_degree_shared, prefix_wigner_address,
    sector_metadata, store_sector_direction,
)


def use_mixed_sector_gradients(x, radial, rotation, dimensions):
    lmax, mmax, _, atomic = dimensions
    if not (
        not atomic and 2 <= lmax <= 6 and mmax == 2
        and x.is_cuda and x.dtype == torch.float32 and x.shape[-1] == 64
        and radial.ndim == rotation.ndim == 2
        and all(value.is_contiguous() and not value.is_neg() for value in (x, radial, rotation))
    ):
        return False
    capability = torch.cuda.get_device_capability(x.device)
    return capability == (9, 0) or (
        capability == (8, 0) and lmax >= 3 and radial.shape[0] >= 8192
    )


@triton.jit
def sectors_mixed_direction(
    x, r, w, u_x, u_r, u_w, v_x, v_r, v_w, edge_index, sectors, edges,
    channels: tl.constexpr, full: tl.constexpr, compact: tl.constexpr,
    mmax: tl.constexpr, radial_elements: tl.constexpr, groups: tl.constexpr,
    rows0: tl.constexpr, rows1: tl.constexpr, dense0: tl.constexpr, dense1: tl.constexpr,
    degrees: tl.constexpr, radial_offsets: tl.constexpr, offsets: tl.constexpr,
    elements: tl.constexpr, has_ux: tl.constexpr, has_ur: tl.constexpr, has_uw: tl.constexpr,
    has_vx: tl.constexpr, has_vr: tl.constexpr, has_vw: tl.constexpr, block: tl.constexpr,
):
    edge, branch = tl.program_id(0).to(tl.int64), tl.program_id(1)
    lane = tl.arange(0, block)
    valid = lane < channels
    endpoint = branch * channels + lane
    node = tl.load(edge_index + branch * edges + edge)
    for group in tl.static_range(groups):
        degree = tl.constexpr(degrees[group])
        radial_index = edge * radial_elements + tl.constexpr(radial_offsets[group]) + endpoint
        scale = tl.load(r + radial_index, valid, 0.0) if (has_uw and has_vx) or (has_vw and has_ux) else 0.0
        ru = tl.load(u_r + radial_index, valid, 0.0) if has_ur else 0.0
        rv = tl.load(v_r + radial_index, valid, 0.0) if has_vr else 0.0
        for sign in tl.static_range(1 if rows1[group] < 0 else 2):
            row = tl.constexpr(rows0[group] if sign == 0 else rows1[group])
            dense = tl.constexpr(dense0[group] if sign == 0 else dense1[group])
            mixed = tl.zeros((block,), tl.float32)
            direction_u = tl.zeros((block,), tl.float32)
            direction_v = tl.zeros((block,), tl.float32)
            for local in tl.static_range(2 * degrees[group] + 1):
                component = degree * degree + local
                address = (node * full + component) * channels + lane
                coefficient_address = prefix_wigner_address(edge, dense, component, full, compact, True, mmax)
                if (has_ur and has_vw) or (has_vr and has_uw):
                    feature = tl.load(x + address, valid, 0.0)
                if has_ux and (has_vw or has_vr):
                    xu = tl.load(u_x + address, valid, 0.0)
                if has_vx and (has_uw or has_ur):
                    xv = tl.load(v_x + address, valid, 0.0)
                if (has_ur and has_vx) or (has_vr and has_ux):
                    weight = tl.load(w + coefficient_address)
                if has_uw and (has_vx or has_vr):
                    wu = tl.load(u_w + coefficient_address)
                if has_vw and (has_ux or has_ur):
                    wv = tl.load(v_w + coefficient_address)
                if has_uw and has_vx:
                    mixed += wu * xv
                if has_vw and has_ux:
                    mixed += wv * xu
                if has_ur and has_vx:
                    direction_v += weight * xv
                if has_ur and has_vw:
                    direction_v += wv * feature
                if has_vr and has_ux:
                    direction_u += weight * xu
                if has_vr and has_uw:
                    direction_u += wu * feature
            value = scale * mixed + ru * direction_v + rv * direction_u
            store_sector_direction(sectors, edge, row, endpoint, value, valid, mmax, offsets, elements, channels)


def sector_direction_constants(lmax, mmax, channels, compact, outer, directions):
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    offsets, elements = sector_metadata(spec)
    return dict(
        channels=channels, full=spec.full_components, compact=compact, mmax=mmax,
        radial_elements=spec.radial_elements, groups=len(spec.radial_group_rows0),
        rows0=spec.radial_group_rows0, rows1=spec.radial_group_rows1,
        dense0=spec.radial_group_dense_rows0, dense1=spec.radial_group_dense_rows1,
        degrees=spec.radial_group_degrees, radial_offsets=spec.radial_group_offsets,
        offsets=offsets, elements=elements,
        has_ux=outer[0], has_ur=outer[1], has_uw=outer[2],
        has_vx=directions[0], has_vr=directions[1], has_vw=directions[2],
        block=max(32, triton.next_power_of_2(channels)),
    )


def mixed_sector_gradients(values, directions, outer, cotangents, *, edge_index, lmax, mmax):
    if all(value is None for value in outer):
        return (None,) * len(cotangents)
    outputs = tuple(value.new_empty(value.shape) for value in cotangents)
    edges = edge_index.shape[1]
    if not edges:
        return outputs

    def pointers(items):
        return tuple(value if item is None else item.resolve_neg().contiguous()
                     for value, item in zip(values, items, strict=True))

    spec = sector_direction_constants(
        lmax, mmax, values[0].shape[-1], values[2].shape[-1],
        tuple(item is not None for item in outer), tuple(item is not None for item in directions),
    )
    sectors_mixed_direction[(edges, 2)](
        *values, *pointers(outer), *pointers(directions), edge_index, outputs, edges,
        **spec, num_warps=2 if spec["block"] == 64 else 1,
    )
    return outputs


@triton.jit
def gather_mixed_edge_adjoint_kernel(
    x, r, w, vx, vr, vw, ux, ur, uw, p, q, indices,
    dr, dw, dvr, dvw, edges,
    L: tl.constexpr, M: tl.constexpr, C: tl.constexpr, degree: tl.constexpr,
    rotation_stride: tl.constexpr, degree_offset: tl.constexpr,
    radial_stride: tl.constexpr, radial_offsets: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    width: tl.constexpr = 2 * degree + 1
    columns: tl.constexpr = triton.next_power_of_2(width)
    endpoints: tl.constexpr = 2 * C
    c = tl.arange(0, endpoints)
    column = tl.arange(0, columns)
    active = column < width
    node = tl.load(indices + c // C * edges + edge)
    address = ((node[None, :] * (L + 1) ** 2 + degree * degree + column[:, None]) * C + c[None, :] % C)
    a = tl.load(x + address, active[:, None], 0.)
    a_v = tl.load(vx + address, active[:, None], 0.)
    a_u = tl.load(ux + address, active[:, None], 0.)
    for m in tl.static_range(min(degree, M) + 1):
        sector_width = L + 1 - m
        sector_rows = sector_width if m == 0 else 2 * sector_width
        r_address = edge * radial_stride + tl.constexpr(radial_offsets[m]) + c
        factor = tl.load(r + r_address)
        factor_v = tl.load(vr + r_address)
        factor_u = tl.load(ur + r_address)
        r_grad = tl.full((endpoints,), 0., tl.float32)
        rv_grad = tl.full((endpoints,), 0., tl.float32)
        for sign in tl.static_range(1 if m == 0 else 2):
            sector_row = degree - m + sign * sector_width
            seed_address = (edge * sector_rows + sector_row) * endpoints + c
            seed = tl.load(p[m] + seed_address)
            outer_seed = tl.load(q[m] + seed_address)
            rotation_row = 0 if m == 0 else 2 * m - 1 + sign
            w_address = edge * rotation_stride + degree_offset + rotation_row * width + column
            b = tl.load(w + w_address, active, 0.)[:, None]
            b_v = tl.load(vw + w_address, active, 0.)[:, None]
            b_u = tl.load(uw + w_address, active, 0.)[:, None]

            seed_rotation = outer_seed[None, :] * b + seed[None, :] * b_u
            outer_feature = seed[None, :] * a_u
            r_grad += tl.sum(seed_rotation * a_v + b_v * (outer_seed[None, :] * a + outer_feature), 0)
            rv_grad += tl.sum(seed_rotation * a + b * outer_feature, 0)

            seed_radial = (outer_seed * factor + seed * factor_u)[None, :]
            w_grad = tl.sum(seed_radial * a_v + factor_v[None, :] * (outer_seed[None, :] * a + outer_feature), 1)
            wv_grad = tl.sum(seed_radial * a + factor[None, :] * outer_feature, 1)
            tl.store(dw + w_address, w_grad, active)
            tl.store(dvw + w_address, wv_grad, active)
        tl.store(dr + r_address, r_grad)
        tl.store(dvr + r_address, rv_grad)


@triton.jit
def gather_mixed_node_adjoint_kernel(
    p, q, r, w, vr, vw, ur, uw,
    source_order, source_ptr, receiver_ptr, dx, dv,
    L: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    radial_stride: tl.constexpr, rotation_stride: tl.constexpr, degree: tl.constexpr,
    column_tile: tl.constexpr,
):
    node = tl.program_id(0).to(tl.int64)
    # Neighbor seeds and radial weights are shared by all columns of this degree.
    column = tl.program_id(1) * column_tile + tl.arange(0, column_tile)
    full = degree * degree + column
    valid = column < 2 * degree + 1
    lane = tl.arange(0, C)
    dx_sum = tl.full((column_tile, C), 0., tl.float32)
    dv_sum = tl.full((column_tile, C), 0., tl.float32)
    for branch in tl.static_range(2):
        ptr = source_ptr if branch == 0 else receiver_ptr
        begin, end = tl.load(ptr + node), tl.load(ptr + node + 1)
        channel = branch * C + lane
        for position in range(begin, end):
            edge = tl.full((), 0, tl.int64) + (tl.load(source_order + position) if branch == 0 else position)
            dxi = tl.full((column_tile, C), 0., tl.float32)
            dvi = tl.full((column_tile, C), 0., tl.float32)
            for m in tl.static_range(min(M, degree) + 1):
                width = L + 1 - m
                rows = width if m == 0 else 2 * width
                row = degree - m
                group = m * (L + 1) - m * (m - 1) // 2 + row
                radial_index = edge * radial_stride + group * (2 * C) + channel
                factor = tl.load(r + radial_index)
                factor_v = tl.load(vr + radial_index)
                factor_u = tl.load(ur + radial_index)
                for sign in tl.static_range(1 if m == 0 else 2):
                    sector_row = row + sign * width
                    seed_index = (edge * rows + sector_row) * (2 * C) + channel
                    seed = tl.load(p[m] + seed_index)
                    outer_seed = tl.load(q[m] + seed_index)
                    dense = degree * degree + degree + (m if sign == 0 else -m)
                    rotation_index = prefix_wigner_address(
                        edge, dense, full, (L + 1) ** 2, rotation_stride, True, M,
                    )
                    weight = tl.load(w + rotation_index, valid, 0.)
                    weight_v = tl.load(vw + rotation_index, valid, 0.)
                    weight_u = tl.load(uw + rotation_index, valid, 0.)
                    scaled = outer_seed * factor + seed * factor_u
                    dxi += (outer_seed * factor_v)[None, :] * weight[:, None]
                    dxi += scaled[None, :] * weight_v[:, None]
                    dxi += (seed * factor_v)[None, :] * weight_u[:, None]
                    dvi += scaled[None, :] * weight[:, None]
                    dvi += (seed * factor)[None, :] * weight_u[:, None]
            dx_sum += dxi
            dv_sum += dvi
    dst = ((node * (L + 1) ** 2 + full[:, None]) * C) + lane[None, :]
    tl.store(dx + dst, dx_sum, valid[:, None])
    tl.store(dv + dst, dv_sum, valid[:, None])


def mixed_gather_gradients(
    values, directions, outer, cotangents, seed_directions, *,
    sector_hessian, edge_index, metadata, lmax, mmax,
):
    x, radial, rotation = values
    with torch.cuda.device(x.device):
        grad_cotangents = sector_hessian(values, directions, outer, cotangents)

        # The two adjoints share each edge's seeds and the primal radial and rotation entries.
        p = tuple(value.resolve_neg().contiguous() for value in cotangents)
        q = tuple(value.resolve_neg().contiguous() for value in seed_directions)
        u = tuple(value.resolve_neg().contiguous() for value in outer)
        v = tuple(value.resolve_neg().contiguous() for value in directions)
        channels = x.shape[-1]
        spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
        retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
        dx, dvx = torch.empty_like(x), torch.empty_like(x)
        dr, dvr = torch.empty_like(radial), torch.empty_like(radial)
        dw, dvw = torch.empty_like(rotation), torch.empty_like(rotation)
        edges = edge_index.shape[1]
        for degree in range(lmax + 1):
            width = 2 * degree + 1
            tile = min(8, triton.next_power_of_2(width))
            gather_mixed_node_adjoint_kernel[(x.shape[0], triton.cdiv(width, tile))](
                p, q, radial, rotation, *v[1:], *u[1:], *metadata, dx, dvx,
                lmax, mmax, channels, spec.radial_elements, retained.compact_elements,
                degree, tile, num_warps=2 if tile == 1 else 4,
            )
        for degree, rows in enumerate(retained.degree_rows):
            radial_offsets = tuple(
                spec.radial_row_offsets[rows[0 if order == 0 else 2 * order - 1]]
                for order in range(min(degree, mmax) + 1)
            )
            gather_mixed_edge_adjoint_kernel[(edges,)](
                *values, *v, *u, p, q, edge_index, dr, dw, dvr, dvw, edges,
                lmax, mmax, channels, degree, retained.compact_elements,
                retained.compact_row_offsets[rows[0]], spec.radial_elements, radial_offsets,
                num_warps=2,
            )
    return (dx, dr, dw), (dvx, dvr, dvw), grad_cotangents


if gluon is not None:
    @gluon.jit
    def degree_shared_third_kernel(
        p, q, dp, x, index, r, w, vx, vr, vw, ux, ur, uw,
        dx, dr, dw, dvx, dvr, dvw, edges,
        lmax: gl.constexpr, mmax: gl.constexpr, channels: gl.constexpr,
        degree: gl.constexpr, rotation_size: gl.constexpr, degree_offset: gl.constexpr,
        vector: gl.constexpr, warps: gl.constexpr,
    ):
        edge = gl.program_id(0).to(gl.int64)
        width: gl.constexpr = 2 * degree + 1
        rows: gl.constexpr = triton.next_power_of_2(width)
        endpoints: gl.constexpr = 2 * channels
        layout: gl.constexpr = gl.BlockedLayout([1, vector], [1, 32], [1, warps], [1, 0])
        row_layout: gl.constexpr = gl.SliceLayout(1, layout)
        channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
        col = gl.arange(0, rows, layout=row_layout)
        channel = gl.arange(0, endpoints, layout=channel_layout)
        live_col = col < width
        live = live_col[:, None]
        node = gl.load(index + channel // channels * edges + edge)
        x_at = ((node[None, :] * (lmax + 1) ** 2 + degree * degree + col[:, None])
                * channels + channel[None, :] % channels)
        ax = gl.load(x + x_at, live, 0.)
        av = gl.load(vx + x_at, live, 0.)
        au = gl.load(ux + x_at, live, 0.)
        r_at = (edge * (lmax + 1) + degree) * endpoints + channel
        rr = gl.load(r + r_at)
        rv = gl.load(vr + r_at)
        ru = gl.load(ur + r_at)
        gx = gl.full((rows, endpoints), 0., gl.float32, layout)
        gvx = gl.full((rows, endpoints), 0., gl.float32, layout)
        gr = gl.full((endpoints,), 0., gl.float32, channel_layout)
        gvr = gl.full((endpoints,), 0., gl.float32, channel_layout)

        order_count: gl.constexpr = min(degree, mmax) + 1
        for m in gl.static_range(order_count):
            sector_width = lmax + 1 - m
            sector_rows = sector_width if m == 0 else 2 * sector_width
            for sign in gl.static_range(1 if m == 0 else 2):
                sector_row = degree - m + sign * sector_width
                p_at = (edge * sector_rows + sector_row) * endpoints + channel
                pp, qq = gl.load(p[m] + p_at), gl.load(q[m] + p_at)
                w_row = 0 if m == 0 else 2 * m - 1 + sign
                w_at = edge * rotation_size + degree_offset + w_row * width + col
                bw = gl.load(w + w_at, live_col, 0.)[:, None]
                bv = gl.load(vw + w_at, live_col, 0.)[:, None]
                bu = gl.load(uw + w_at, live_col, 0.)[:, None]

                # Reverse q*Df[v] + p*D²f[v,u] for f = r*(w @ x).
                pr = pp * rr
                pvr = pp * rv
                qr_ur = qq * rr + pp * ru
                qx_ux = qq[None, :] * ax + pp[None, :] * au
                qw_uw = qq[None, :] * bw + pp[None, :] * bu
                gx += (qq * rv)[None, :] * bw + qr_ur[None, :] * bv + pvr[None, :] * bu
                gvx += qr_ur[None, :] * bw + pr[None, :] * bu
                gr += gl.sum(qw_uw * av + bv * qx_ux, 0)
                gvr += gl.sum(qw_uw * ax + bw * pp[None, :] * au, 0)
                gw = gl.sum(qr_ur[None, :] * av + rv[None, :] * qx_ux, 1)
                gvw = gl.sum(qr_ur[None, :] * ax + pr[None, :] * au, 1)
                gl.store(dw + w_at, gw, live_col)
                gl.store(dvw + w_at, gvw, live_col)

                dseed = rr[None, :] * (bv * au + bu * av)
                dseed += rv[None, :] * (bu * ax + bw * au)
                dseed += ru[None, :] * (bv * ax + bw * av)
                gl.store(dp[m] + p_at, gl.sum(dseed, 0))

        gl.store(dr + r_at, gr)
        gl.store(dvr + r_at, gvr)
        add_node_direction(dx + x_at, gx, live)
        add_node_direction(dvx + x_at, gvx, live)


    def degree_shared_mixed_gradients(values, directions, outer_values, seeds, outer_seeds, *, edge_index):
        allow_atomic_reduction("degree-shared node-to-edge triple backward")
        x, radial, rotation = values

        def contiguous(tensors):
            return tuple(t.resolve_neg().contiguous() for t in tensors)

        dx, dvx = torch.zeros_like(x), torch.zeros_like(x)
        dr, dvr = torch.empty_like(radial), torch.empty_like(radial)
        dw, dvw = torch.empty_like(rotation), torch.empty_like(rotation)
        ds = tuple(torch.empty_like(seed, memory_format=torch.contiguous_format) for seed in seeds)
        args = (contiguous(seeds), contiguous(outer_seeds), ds, x, edge_index, radial, rotation,
                *contiguous(directions), *contiguous(outer_values), dx, dr, dw, dvx, dvr, dvw)
        launch_degree_shared(degree_shared_third_kernel, x, edge_index, rotation, args, lmax=4, mmax=2)
        return (dx, dr, dw), (dvx, dvr, dvw), ds

else:
    degree_shared_mixed_gradients = None

def accumulate(total, value):
    return value if total is None else total + value


class GatherDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, radial, rotation, hx, hr, hw, hvp, vjp,
                mask_outputs, mixed_gradients, sector_hessian, *cotangents):
        values = x, radial, rotation
        directions = hx, hr, hw
        ctx.save_for_backward(*values, *directions, *cotangents)
        ctx.set_materialize_grads(False)
        ctx.hvp = hvp
        ctx.vjp = vjp
        ctx.mask_outputs = mask_outputs
        ctx.mixed_gradients = mixed_gradients
        ctx.sector_hessian = sector_hessian
        return tuple(hvp(cotangents, values, directions))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        x, radial, rotation, hx, hr, hw, *cotangents = ctx.saved_tensors
        values = x, radial, rotation
        directions = hx, hr, hw
        count = len(cotangents)
        seed_directions = gradients[:count]
        value_directions = gradients[count:]
        if (
            ctx.mixed_gradients is not None
            and (ctx.sector_hessian is None or radial.shape[0] >= 8192)
            and not torch.compiler.is_compiling()
            and all(gradient is not None for gradient in gradients)
        ):
            grad_values, grad_directions, grad_cotangents = ctx.mixed_gradients(
                values, directions, value_directions, cotangents, seed_directions,
            )
            return *grad_values, *grad_directions, None, None, None, None, None, *grad_cotangents
        grad_values = [None] * len(values)
        grad_cotangents = [None] * count
        grad_directions = [None] * len(directions)

        def contract(seeds, values, directions, mask):
            if ctx.mask_outputs:
                return ctx.hvp(seeds, values, directions, output_mask=mask)
            return ctx.hvp(seeds, values, directions)

        # The contracted gather is linear in each of x, radial, rotation, and
        # the output cotangent. An HVP with one slot substituted gives the
        # third derivative terms that still contain that slot.
        primal_only = (False, True, True, True)
        if any(gradient is not None for gradient in seed_directions):
            seeds = tuple(
                torch.zeros_like(seed) if grad is None else grad.contiguous()
                for seed, grad in zip(cotangents, seed_directions, strict=True)
            )
            grad_values = list(contract(seeds, values, directions, primal_only)[count:])
            grad_directions = list(ctx.vjp(seeds, values))

        if any(gradient is not None for gradient in value_directions):
            weights = tuple(
                torch.zeros_like(value) if grad is None else grad.contiguous()
                for value, grad in zip(values, value_directions, strict=True)
            )
            for index, gradient in enumerate(
                contract(cotangents, values, weights, primal_only)[count:]
            ):
                grad_directions[index] = accumulate(grad_directions[index], gradient)

            split_sectors = ctx.sector_hessian is None
            for index, grad in enumerate(value_directions):
                if grad is None:
                    continue
                sub_values = list(values)
                sub_values[index] = weights[index]
                sub_directions = list(directions)
                # A term that already replaced this slot with h is constant
                # with respect to its original value.
                sub_directions[index] = torch.zeros_like(directions[index])
                mask = (split_sectors, index != 0, index != 1, index != 2)
                result = contract(cotangents, sub_values, sub_directions, mask)
                if split_sectors:
                    for sector, gradient in enumerate(result[:count]):
                        grad_cotangents[sector] = accumulate(grad_cotangents[sector], gradient)
                for slot, gradient in enumerate(result[count:]):
                    if slot != index:
                        grad_values[slot] = accumulate(grad_values[slot], gradient)

            if not split_sectors:
                grad_cotangents = ctx.sector_hessian(values, directions, value_directions, cotangents)

        return *grad_values, *grad_directions, None, None, None, None, None, *grad_cotangents
