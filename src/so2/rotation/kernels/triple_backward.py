from functools import partial

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .utils import (
    ROTATION_MASK_THRESHOLD,
    reduce_angles,
    retained_metadata,
    selected_local_rows,
    selected_row_indices,
    supports_factored_retained_rotation,
)


def third_adjoint(operation, inputs, cotangents, directions, weights):
    with torch.enable_grad():
        inputs = tuple(value.detach().requires_grad_() for value in inputs)
        cotangents = tuple(value.detach().requires_grad_() for value in cotangents)
        directions = tuple(value.detach().requires_grad_() for value in directions)
        first = torch.autograd.grad(
            operation(*inputs), inputs, cotangents, create_graph=True,
        )
        second = torch.autograd.grad(
            first, (*cotangents, *inputs), directions, create_graph=True,
        )
        terms = [(value, weight) for value, weight in zip(second, weights, strict=True)
                 if weight is not None and value.requires_grad]
        targets = (*cotangents, *inputs, *directions)
        if not terms:
            return (None,) * len(targets)
        outputs, grad_outputs = zip(*terms)
        return torch.autograd.grad(outputs, targets, grad_outputs, allow_unused=True)


def active_frame_angles(frames):
    # Both conventions have the same derivatives in the open Euler chart.
    active = frames[:, 1, 1].abs() < ROTATION_MASK_THRESHOLD
    chart_frame = frames.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    frames = torch.where(active[:, None, None], frames, chart_frame)
    x, y, z = frames[:, :, 1].unbind(1)
    alpha = torch.atan2(x, z)
    beta = torch.atan2((x.square() + z.square()).sqrt(), y)
    cosine, sine = alpha.cos(), alpha.sin()
    gamma = torch.atan2(
        cosine * frames[:, 0, 2] - sine * frames[:, 2, 2],
        cosine * frames[:, 0, 0] - sine * frames[:, 2, 0],
    )
    return alpha, beta, gamma


# Each tuple holds a value, two directional derivatives, and their mixed derivative.
@triton.jit
def multiply(a, b):
    x, xu, xv, xuv = a
    y, yu, yv, yuv = b
    return x * y, xu * y + x * yu, xv * y + x * yv, xuv * y + xu * yv + xv * yu + x * yuv


@triton.jit
def subtract(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]


@triton.jit
def negate(a):
    return -a[0], -a[1], -a[2], -a[3]


@triton.jit
def reciprocal(r):
    x, xu, xv, xuv = r
    q = 1. / x
    qu = -xu * q * q
    qv = -xv * q * q
    quv = q * q * (2. * q * xu * xv - xuv)
    return q, qu, qv, quv


@triton.jit
def radius(x, y):
    a, au, av, auv = x
    b, bu, bv, buv = y
    r = tl.sqrt(a * a + b * b)
    ru = (a * au + b * bu) / r
    rv = (a * av + b * bv) / r
    ruv = (au * av + a * auv + bu * bv + b * buv - ru * rv) / r
    return r, ru, rv, ruv


@triton.jit
def atan2_jacobian(y, x):
    x2, y2 = multiply(x, x), multiply(y, y)
    q = reciprocal((x2[0] + y2[0], x2[1] + y2[1], x2[2] + y2[2], x2[3] + y2[3]))
    return multiply(x, q), negate(multiply(y, q))


@triton.jit
def mixed_angle(jx, jy, x, y):
    return jx[2] * x[1] + jx[0] * x[3] + jy[2] * y[1] + jy[0] * y[3]


@triton.jit
def adjoint(j, seed, linear):
    # Reverse the direction contraction and the mixed chart derivative.
    return linear * j[1] + seed * j[3], linear * j[0] + seed * j[2]


@triton.jit
def value_jet(f, d, v, edge, column: tl.constexpr, active, inactive: tl.constexpr):
    address = 9 * edge + column
    z = tl.load(f + address, active, inactive)
    u = tl.load(d + address, active, 0.)
    w = tl.load(v + address, active, 0.)
    return z, u, w, tl.full(z.shape, 0., tl.float32)


@triton.jit
def chart_third_kernel(
    frames, direction, weight, cotangent, linear, d_cotangent, d_frame, d_direction,
    edges, threshold: tl.constexpr, block: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    live = edge < edges
    regular = tl.abs(tl.load(frames + edge * 9 + 4, live, 1.)) < threshold
    active = live & regular

    # The positive horizontal norm cancels inside gamma's atan2.
    x = value_jet(frames, direction, weight, edge, 1, active, 0.)
    y = value_jet(frames, direction, weight, edge, 4, active, 0.)
    z = value_jet(frames, direction, weight, edge, 7, active, 1.)
    a = value_jet(frames, direction, weight, edge, 2, active, 0.)
    b = value_jet(frames, direction, weight, edge, 8, active, 0.)
    c = value_jet(frames, direction, weight, edge, 0, active, 1.)
    d = value_jet(frames, direction, weight, edge, 6, active, 0.)

    alpha_x, alpha_z = atan2_jacobian(x, z)
    r = radius(x, z)
    rinv = reciprocal(r)
    beta_r, beta_y = atan2_jacobian(r, y)
    beta_x = multiply(beta_r, multiply(x, rinv))
    beta_z = multiply(beta_r, multiply(z, rinv))

    num = subtract(multiply(z, a), multiply(x, b))
    den = subtract(multiply(z, c), multiply(x, d))
    gamma_num, gamma_den = atan2_jacobian(num, den)
    gamma_x = subtract(negate(multiply(gamma_num, b)), multiply(gamma_den, d))
    gz = multiply(gamma_num, a)
    gm = multiply(gamma_den, c)
    gamma_z = (gz[0] + gm[0], gz[1] + gm[1], gz[2] + gm[2], gz[3] + gm[3])

    ca = tl.load(cotangent[0] + edge, live, 0.)
    cb = tl.load(cotangent[1] + edge, live, 0.)
    cg = tl.load(cotangent[2] + edge, live, 0.)
    ka = tl.load(linear[0] + edge, live, 0.)
    kb = tl.load(linear[1] + edge, live, 0.)
    kg = tl.load(linear[2] + edge, live, 0.)
    dax, duax = adjoint(alpha_x, ca, ka)
    daz, duaz = adjoint(alpha_z, ca, ka)
    dbx, dubx = adjoint(beta_x, cb, kb)
    dby, duby = adjoint(beta_y, cb, kb)
    dbz, dubz = adjoint(beta_z, cb, kb)
    dgx, dugx = adjoint(gamma_x, cg, kg)
    dgz, dugz = adjoint(gamma_z, cg, kg)
    dga, duga = adjoint(multiply(gamma_num, z), cg, kg)
    dgb, dugb = adjoint(negate(multiply(gamma_num, x)), cg, kg)
    dgc, dugc = adjoint(multiply(gamma_den, z), cg, kg)
    dgd, dugd = adjoint(negate(multiply(gamma_den, x)), cg, kg)
    f = (dgc, dax + dbx + dgx, dga, 0., dby, 0., dgd, daz + dbz + dgz, dgb)
    u = (dugc, duax + dubx + dugx, duga, 0., duby, 0., dugd, duaz + dubz + dugz, dugb)
    for i in tl.static_range(9):
        tl.store(d_frame + edge * 9 + i, tl.where(regular, f[i], 0.), live)
        tl.store(d_direction + edge * 9 + i, tl.where(regular, u[i], 0.), live)

    grad_seeds = (mixed_angle(alpha_x, alpha_z, x, z),
                  mixed_angle(beta_r, beta_y, r, y),
                  mixed_angle(gamma_num, gamma_den, num, den))
    for i in tl.static_range(3):
        tl.store(d_cotangent[i] + edge, grad_seeds[i], live)


def chart_third(frames, direction, cotangents, linears, weight):
    d_cotangents = tuple(torch.empty_like(item) for item in cotangents)
    d_frames = torch.empty_like(frames)
    d_direction = torch.empty_like(direction)
    weight = weight.resolve_neg().contiguous()
    linears = tuple(item.resolve_neg().contiguous() for item in linears)
    with torch.cuda.device(frames.device):
        chart_third_kernel[(triton.cdiv(frames.shape[0], 128),)](
            frames, direction, weight, cotangents, linears, d_cotangents, d_frames, d_direction,
            frames.shape[0], ROTATION_MASK_THRESHOLD, 128, num_warps=4, enable_fp_fusion=False,
        )
    return *d_cotangents, d_frames, d_direction


def use_chart_third(frames, direction, weights):
    return (
        frames.shape[0] >= 16384 and frames.is_cuda and frames.dtype == torch.float32
        and torch.version.hip is None and torch.cuda.get_device_capability(frames.device) == (9, 0)
        and not torch.compiler.is_compiling()
        and frames.is_contiguous() and not frames.is_neg()
        and direction.is_contiguous() and not direction.is_neg()
        and all(weight is not None for weight in weights)
    )


class AnglesDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ga, gb, gc, frames, direction, hvp):
        ctx.save_for_backward(ga, gb, gc, frames, direction)
        ctx.set_materialize_grads(False)
        return hvp(ga, gb, gc, frames, direction)

    @staticmethod
    @once_differentiable
    def backward(ctx, *weights):
        ga, gb, gc, frames, direction = ctx.saved_tensors
        if use_chart_third(frames, direction, weights):
            gradients = chart_third(frames, direction, (ga, gb, gc), weights[:3], weights[3])
        else:
            gradients = third_adjoint(
                active_frame_angles, (frames,), (ga, gb, gc), (direction,), weights,
            )
        return (*gradients, None)


def retained_block(alpha, beta, gamma, *, matrix, degree, rows):
    frequencies = torch.arange(degree, -degree - 1, -1, device=alpha.device,
                               dtype=alpha.dtype)
    selected = torch.stack(tuple(matrix[row] for row in rows))
    opposite = torch.stack(tuple(matrix[2 * degree - row] for row in rows))
    selected_frequencies = torch.stack(tuple(frequencies[row] for row in rows))
    phase = alpha[:, None, None] * selected_frequencies[None, :, None]
    first = torch.addcmul(phase.cos() * selected, phase.sin(), opposite)

    def rotate_columns(values, angle):
        # Right multiplication by the sine antidiagonal reverses columns.
        phase = angle[:, None, None] * frequencies
        return torch.addcmul(values * phase.cos(), values.flip(-1), phase.sin().flip(-1))

    return rotate_columns(rotate_columns(first, beta) @ matrix, gamma).flatten(1)


@triton.jit
def angular_direction(value, da, dg, fa, fg, mirror_row, mirror_column):
    # Alpha and gamma act by signed row and column exchanges, respectively.
    return (da * fa * tl.gather(value, mirror_row, axis=0)
            - dg * fg * tl.gather(value, mirror_column, axis=1))


@triton.jit
def beta_derivatives(alpha, beta, gamma, j, fa, fg, selected, row_live, col, col_live, mirror_column,
                        degree: tl.constexpr, width: tl.constexpr, br: tl.constexpr, bc: tl.constexpr,
                        need_third: tl.constexpr):
    ca, sa = tl.cos(alpha * fa), tl.sin(alpha * fa)
    flipped = width - 1 - selected
    c0 = tl.zeros((br, bc), tl.float32)
    c1 = tl.zeros((br, bc), tl.float32)
    c2 = tl.zeros((br, bc), tl.float32)
    c3 = tl.zeros((br, bc), tl.float32)
    for inner in tl.static_range(width):
        opposite = width - 1 - inner
        x = ca * tl.load(j + selected * width + inner, row_live, 0.)
        x += sa * tl.load(j + flipped * width + inner, row_live, 0.)
        xf = ca * tl.load(j + selected * width + opposite, row_live, 0.)
        xf += sa * tl.load(j + flipped * width + opposite, row_live, 0.)
        fb = degree - inner
        cb, sb = tl.cos(beta * fb), tl.sin(beta * fb)
        b0 = x * cb - xf * sb
        b1 = fb * (-x * sb - xf * cb)
        coef = tl.load(j + inner * width + col, col_live, 0.)
        c0 += b0 * coef
        c1 += b1 * coef
        c2 += (-fb * fb) * b0 * coef
        if need_third:
            c3 += (-fb * fb) * b1 * coef

    cg, sg = tl.cos(gamma * fg), tl.sin(gamma * fg)
    f0 = c0 * cg - tl.gather(c0, mirror_column, axis=1) * sg
    f1 = c1 * cg - tl.gather(c1, mirror_column, axis=1) * sg
    f2 = c2 * cg - tl.gather(c2, mirror_column, axis=1) * sg
    f3 = c3 * cg - tl.gather(c3, mirror_column, axis=1) * sg if need_third else c3
    return f0, f1, f2, f3


@triton.jit
def retained_third_kernel(
    angles, directions, outer, gradients, weights, j, scale, carrier_outputs, angle_outputs, direction_outputs,
    compact: tl.constexpr, offset: tl.constexpr, degree: tl.constexpr, selected_rows: tl.constexpr,
    row_count: tl.constexpr, width: tl.constexpr, br: tl.constexpr, bc: tl.constexpr,
    has_scale: tl.constexpr, has_weight: tl.constexpr, has_inverse_weight: tl.constexpr,
    has_va: tl.constexpr, has_vb: tl.constexpr, has_vg: tl.constexpr,
):
    edge = tl.program_id(0)
    # Keep signed row partners adjacent for the alpha generator.
    row = tl.arange(0, br)[:, None]
    col = tl.arange(0, bc)[None, :]
    row_live, col_live = row < row_count, col < width
    live = row_live & col_live
    stored = tl.where(row + 1 < row_count, row + 1, 0)
    selected = selected_row_indices(stored, selected_rows, row_count)
    fa = degree - selected
    fg = degree - col
    mirror_row = tl.broadcast_to(tl.where(row < row_count - 1, row ^ 1, row), (br, bc))
    mirror_col = tl.broadcast_to(tl.where(col_live, width - 1 - col, col), (br, bc))
    address = edge * compact + offset + stored * width + col
    if has_scale:
        factor = tl.load(scale + offset + stored * width + col, live, 0.)

    a, b, g = (tl.load(angles[0] + edge), tl.load(angles[1] + edge), tl.load(angles[2] + edge))
    da, db, dg = (
        tl.load(directions[0] + edge), tl.load(directions[1] + edge), tl.load(directions[2] + edge),
    )
    has_angle: tl.constexpr = has_va or has_vb or has_vg
    f0, f1, f2, f3 = beta_derivatives(
        a, b, g, j, fa, fg, selected, row_live, col, col_live, mirror_col,
        degree, width, br, bc, has_vb,
    )
    # Differentiate <K, DF[D]> + <G, D²F[D, V]>. K seeds the carrier
    # direction and V seeds the angle HVP. G is the original cotangent.
    ha, hb, hg = 0.0, 0.0, 0.0
    dta, dtb, dtg = 0.0, 0.0, 0.0
    if has_weight or has_inverse_weight:
        k = tl.load(weights[0] + address, live, 0.) if has_weight else tl.zeros((br, bc), tl.float32)
        if has_inverse_weight:
            k += tl.load(weights[1] + address, live, 0.) * factor
        ka = -fa * tl.gather(k, mirror_row, axis=0)
        kg = fg * tl.gather(k, mirror_col, axis=1)
        f_d = angular_direction(f0, da, dg, fa, fg, mirror_row, mirror_col) + db * f1
        f_db = angular_direction(f1, da, dg, fa, fg, mirror_row, mirror_col) + db * f2
        ha, hb, hg = reduce_angles(ka * f_d), reduce_angles(k * f_db), reduce_angles(kg * f_d)
        dta, dtb, dtg = reduce_angles(ka * f0), reduce_angles(k * f1), reduce_angles(kg * f0)

    carrier = tl.zeros((br, bc), tl.float32)
    if has_angle:
        va = tl.load(outer[0] + edge) if has_va else 0.
        vb = tl.load(outer[1] + edge) if has_vb else 0.
        vg = tl.load(outer[2] + edge) if has_vg else 0.
        cotangent = tl.load(gradients[0] + address, live, 0.)
        if has_scale:
            cotangent += tl.load(gradients[1] + address, live, 0.) * factor
        ca = -fa * tl.gather(cotangent, mirror_row, axis=0)
        cg = fg * tl.gather(cotangent, mirror_col, axis=1)
        f_v = angular_direction(f0, va, vg, fa, fg, mirror_row, mirror_col) + vb * f1
        f_vb = angular_direction(f1, va, vg, fa, fg, mirror_row, mirror_col) + vb * f2
        f_vbb = angular_direction(f2, va, vg, fa, fg, mirror_row, mirror_col) + vb * f3
        carrier = angular_direction(f_v, da, dg, fa, fg, mirror_row, mirror_col) + db * f_vb
        carrier_beta = angular_direction(f_vb, da, dg, fa, fg, mirror_row, mirror_col) + db * f_vbb
        ha += reduce_angles(ca * carrier)
        hb += reduce_angles(cotangent * carrier_beta)
        hg += reduce_angles(cg * carrier)
        dta += reduce_angles(ca * f_v)
        dtb += reduce_angles(cotangent * f_vb)
        dtg += reduce_angles(cg * f_v)

    tl.store(carrier_outputs[0] + address, carrier, live)
    if has_scale:
        tl.store(carrier_outputs[1] + address, carrier * factor, live)
    if degree == 1:
        tl.store(carrier_outputs[0] + edge * compact, 0.)
        if has_scale:
            tl.store(carrier_outputs[1] + edge * compact, 0.)
    for axis in tl.static_range(3):
        angle = (ha, hb, hg)[axis]
        direction = (dta, dtb, dtg)[axis]
        if degree > 1:
            angle += tl.load(angle_outputs[axis] + edge)
            direction += tl.load(direction_outputs[axis] + edge)
        tl.store(angle_outputs[axis] + edge, angle)
        tl.store(direction_outputs[axis] + edge, direction)


def retained_third(
    direct, inverse, angles, directions, scale, basis, spec,
    direct_weight, inverse_weight, outer,
):
    inverse_weight = inverse_weight if scale is not None else None
    if not any(value is not None for value in (direct_weight, inverse_weight, *outer)):
        inverse_carrier = torch.zeros_like(inverse) if scale is not None else None
        return torch.zeros_like(direct), inverse_carrier, *(
            torch.zeros_like(angle) for angle in (*angles, *directions)
        )

    def pointer(value, backup):
        return backup if value is None else value.resolve_neg().contiguous()

    has_scale = scale is not None
    carrier = torch.empty_like(direct)
    inverse_carrier = torch.empty_like(inverse) if has_scale else None
    outputs = tuple(torch.empty_like(angle) for angle in (*angles, *directions))
    grad_pairs = direct, direct if inverse is None else inverse
    weight_pairs = pointer(direct_weight, direct), pointer(inverse_weight, direct)
    output_pairs = carrier, carrier if inverse_carrier is None else inverse_carrier
    outer_pointers = tuple(pointer(value, angle) for value, angle in zip(outer, angles, strict=True))
    for degree, metadata in enumerate(retained_metadata(spec)[1:], start=1):
        retained_third_kernel[(direct.shape[0],)](
            angles, directions, outer_pointers, grad_pairs, weight_pairs, basis[degree],
            scale if has_scale else direct, output_pairs, outputs[:3], outputs[3:],
            spec.compact_elements, metadata["matrix_offset"], degree,
            metadata["selected_rows"], metadata["row_count"], metadata["width"],
            metadata["block_rows"], metadata["block_width"], has_scale,
            direct_weight is not None, inverse_weight is not None,
            *(value is not None for value in outer), num_warps=2,
        )
    return carrier, inverse_carrier, *outputs


def use_analytic_retained_third(direct, inverse, angles, directions, scale, matrices, spec):
    if (
        spec.mmax != 2 or not 2 <= spec.lmax <= 6
        or direct.shape[0] < 16384 or not direct.is_cuda or torch.version.hip is not None
        or torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled()
        or torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
        or not supports_factored_retained_rotation(direct.device, spec.lmax)
    ):
        return False
    tensors = (direct, *angles, *directions, *matrices)
    if scale is not None:
        tensors += (inverse, scale)
    return all(
        value is not None and value.dtype == torch.float32 and value.device == direct.device
        and value.is_contiguous() and not value.is_neg() for value in tensors
    )


class RetainedWignerDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direct, inverse, alpha, beta, gamma, da, db, dc, scale, *inputs):
        *matrices, spec, hvp = inputs
        ctx.save_for_backward(direct, inverse, alpha, beta, gamma, da, db, dc,
                              scale, *matrices)
        ctx.spec = spec
        ctx.set_materialize_grads(False)
        matrix_directions, angle_hvp = hvp(direct, inverse, alpha, beta, gamma,
                                           (da, db, dc), matrices, scale, spec)
        return (*matrix_directions, *angle_hvp)

    @staticmethod
    @once_differentiable
    def backward(ctx, direct_weight, inverse_weight, *angle_weights):
        direct, inverse, alpha, beta, gamma, da, db, dc, scale, *matrices = ctx.saved_tensors
        angles = alpha, beta, gamma
        directions = da, db, dc
        if use_analytic_retained_third(
            direct, inverse, angles, directions, scale, matrices, ctx.spec,
        ):
            gradients = retained_third(
                direct, inverse, angles, directions, scale, matrices, ctx.spec,
                direct_weight, inverse_weight, angle_weights,
            )
            return (*gradients, None, *(None for _ in matrices), None, None)

        carrier_gradient = torch.empty_like(direct)
        angle_gradients = tuple(torch.zeros_like(value) for value in (*angles, *directions))
        rows_by_degree = selected_local_rows(ctx.spec.lmax, ctx.spec.mmax)
        # The degree-zero block is constant with respect to all three angles.
        carrier_gradient[:, 0].zero_()
        start = 1
        # Degrees are independent of one another. Rebuild only one degree's
        # computation graph at a time to keep the third-stage tape bounded.
        blocks = zip(matrices[1:], rows_by_degree[1:], strict=True)
        for degree, (matrix, rows) in enumerate(blocks, 1):
            stop = start + len(rows) * matrix.shape[0]
            band = slice(start, stop)
            cotangent = direct[:, band]
            weight = direct_weight[:, band] if direct_weight is not None else None
            if scale is not None:
                block_scale = scale[band]
                cotangent = cotangent + inverse[:, band] * block_scale
                if inverse_weight is not None:
                    extra = inverse_weight[:, band] * block_scale
                    weight = extra if weight is None else weight + extra
            operation = partial(retained_block, matrix=matrix, degree=degree, rows=rows)
            grad_carrier, *grad_angles = third_adjoint(
                operation, angles, (cotangent,), directions, (weight, *angle_weights),
            )
            output = carrier_gradient[:, band]
            if grad_carrier is None:
                output.zero_()
            else:
                output.copy_(grad_carrier)
            for total, gradient in zip(angle_gradients, grad_angles, strict=True):
                if gradient is not None:
                    total.add_(gradient)
            start = stop
        inverse_gradient = carrier_gradient * scale if scale is not None else None
        return (carrier_gradient, inverse_gradient, *angle_gradients,
                None, *(None for _ in matrices), None, None)
