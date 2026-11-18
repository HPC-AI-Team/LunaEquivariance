from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ...utils.tensor_contracts import allow_atomic_reduction
from .forward import INVERSE, frame
from .utils import (
    adaptive_quaternion_tile,
    QUADRATIC_COUNT_TL,
    QUADRATIC_W_TL,
    QUADRATIC_X_TL,
    QUADRATIC_Y_TL,
    QUADRATIC_Z_TL,
    QUARTIC_COUNT_TL,
    QUARTIC_W_TL,
    QUARTIC_X_TL,
    QUARTIC_Y_TL,
    QUARTIC_Z_TL,
    ROTATION_MASK_THRESHOLD_TL,
    angle_derivatives,
    angle_jets,
    edge_quaternion_jets,
    frame_angle_edges,
    frame_angle_tile,
    launch_constants,
    matrix_address,
    monomial_derivative,
    monomial_values,
    polynomial_integer_power,
    reduce_angles,
    pack_sparse_quaternion,
    retained_metadata,
    retained_tile_metadata,
    retained_tile_rows,
    retained_tiles_kernel,
    supports_factored_retained_rotation,
    use_retained_tiles,
    rotation_metadata,
    rotation_values,
    selected_row_indices,
    use_quaternion_tiles,
    use_sparse_quaternion,
    wigner_terms,
)


@triton.jit
def edge_quaternion_backward_kernel(q_gradient_ptr, edge_ptr, gamma_ptr, edge_gradient_ptr, edges=0, tile_edges: tl.constexpr = 1):
    edge, live = frame_angle_edges(edges, tile_edges)
    axis = tl.arange(0, 4)
    if tile_edges > 1:
        edge = edge[:, None]
        live = live[:, None]
        axis = axis[None, :]
    base = edge * 3
    jets = edge_quaternion_jets(
        tl.load(edge_ptr + base).to(tl.float32),
        tl.load(edge_ptr + base + 1).to(tl.float32),
        tl.load(edge_ptr + base + 2).to(tl.float32),
        tl.load(gamma_ptr + edge).to(tl.float32),
        0.0,
        0.0,
        0.0,
        axis,
        second_order=False,
    )
    accumulator_shape: tl.constexpr = (4,) if tile_edges == 1 else (tile_edges, 4)
    gradient = tl.zeros(accumulator_shape, tl.float32)
    for component in tl.static_range(4):
        upstream = tl.load(q_gradient_ptr + edge * 4 + component).to(tl.float32)
        gradient += upstream * jets[component][1]
    tl.store(edge_gradient_ptr + base + axis, gradient, mask=live & (axis < 3))


def edge_quaternion_backward(q_gradient, edge_vectors, gamma, compile_visible):
    edge_gradient = torch.empty_like(edge_vectors)
    if compile_visible:
        torch.ops.so2.edge_quaternion_backward(
            q_gradient,
            edge_vectors,
            gamma,
            edge_gradient,
        )
    else:
        tile, warps = (16, 2) if use_quaternion_tiles(edge_vectors) else (1, 1)
        edge_quaternion_backward_kernel[(triton.cdiv(edge_vectors.shape[0], tile),)](
            q_gradient,
            edge_vectors,
            gamma,
            edge_gradient,
            edge_vectors.shape[0],
            tile,
            num_warps=warps,
        )
    return edge_gradient


@triton.jit
def angles_backward_kernel(
    alpha_gradient_ptr,
    beta_gradient_ptr,
    gamma_gradient_ptr,
    rotation_ptr,
    rotation_gradient_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    gradient_stride_edge,
    gradient_stride_row,
    gradient_stride_column,
    block_variables: tl.constexpr,
    edges,
    tile_edges: tl.constexpr,
):
    edge, valid = frame_angle_edges(edges, tile_edges)
    axis = tl.arange(0, block_variables)
    if tile_edges > 1:
        edge = edge[:, None]
        valid = valid[:, None]
        axis = axis[None, :]
    mask = (axis < 9) & valid
    alpha, beta, gamma, yprod = angle_jets(
        rotation_ptr,
        rotation_ptr,
        edge,
        axis,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        False,
    )
    active = (yprod > -ROTATION_MASK_THRESHOLD_TL) & (
        yprod < ROTATION_MASK_THRESHOLD_TL
    )
    alpha_gradient = tl.load(alpha_gradient_ptr + edge).to(tl.float32)
    beta_gradient = tl.load(beta_gradient_ptr + edge).to(tl.float32)
    gamma_gradient = tl.load(gamma_gradient_ptr + edge).to(tl.float32)
    gradient = tl.where(
        active,
        alpha_gradient * alpha[1] + beta_gradient * beta[1] + gamma_gradient * gamma[1],
        0.0,
    )
    tl.store(
        rotation_gradient_ptr
        + edge * gradient_stride_edge
        + (axis // 3) * gradient_stride_row
        + (axis % 3) * gradient_stride_column,
        gradient,
        mask=mask,
    )


def masked_angles_backward(alpha_gradient, beta_gradient, gamma_gradient, rotation):
    rotation_gradient = torch.empty_like(rotation)
    tile, warps = frame_angle_tile(rotation, masked_derivatives=True)
    angles_backward_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        alpha_gradient,
        beta_gradient,
        gamma_gradient,
        rotation,
        rotation_gradient,
        rotation_stride_edge=rotation.stride(0),
        rotation_stride_row=rotation.stride(1),
        rotation_stride_column=rotation.stride(2),
        gradient_stride_edge=rotation_gradient.stride(0),
        gradient_stride_row=rotation_gradient.stride(1),
        gradient_stride_column=rotation_gradient.stride(2),
        block_variables=16,
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return rotation_gradient


@triton.jit
def monomial_vjp_kernel(
    gradient_ptr,
    q_ptr,
    exponents_ptr,
    q_gradient_ptr,
    monomial_count,
    block_monomials: tl.constexpr,
    max_power: tl.constexpr,
):
    edge = tl.program_id(0)
    axis = tl.program_id(1)
    index = tl.arange(0, block_monomials)
    mask = index < monomial_count
    _, exponents, values = monomial_values(
        q_ptr, exponents_ptr, edge, index, mask, max_power
    )
    gradient = tl.load(
        gradient_ptr + edge * monomial_count + index,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    derivative = monomial_derivative(exponents, values, axis, max_power)
    tl.store(q_gradient_ptr + edge * 4 + axis, tl.sum(gradient * derivative))


@triton.jit
def pole_clamped_angles_backward_kernel(
    alpha_gradient_ptr,
    beta_gradient_ptr,
    gamma_gradient_ptr,
    rotation_ptr,
    rotation_gradient_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    gradient_stride_edge,
    gradient_stride_row,
    gradient_stride_column,
    edges,
    tile_edges: tl.constexpr,
):
    edge, valid = frame_angle_edges(edges, tile_edges)
    values = rotation_values(
        rotation_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
    )
    derivatives = angle_derivatives(values)
    regular = values[14]
    alpha_gradient = tl.load(alpha_gradient_ptr + edge).to(tl.float32)
    beta_gradient = tl.load(beta_gradient_ptr + edge).to(tl.float32)
    gamma_gradient = tl.load(gamma_gradient_ptr + edge).to(tl.float32)
    combined_alpha = alpha_gradient + gamma_gradient * derivatives[9]
    gradient_x = combined_alpha * derivatives[0] + beta_gradient * derivatives[2]
    gradient_y = beta_gradient * derivatives[3]
    gradient_z = combined_alpha * derivatives[1] + beta_gradient * derivatives[4]
    gradient_r02 = gamma_gradient * derivatives[5] * values[10]
    gradient_r22 = -gamma_gradient * derivatives[5] * values[11]
    gradient_r00 = gamma_gradient * derivatives[6] * values[10]
    gradient_r20 = -gamma_gradient * derivatives[6] * values[11]
    gradient_base = edge * gradient_stride_edge
    for offset in tl.static_range(0, 9):
        gradient = tl.zeros((), tl.float32)
        if offset == 0:
            gradient = gradient_r00
        elif offset == 1:
            gradient = gradient_x
        elif offset == 2:
            gradient = gradient_r02
        elif offset == 4:
            gradient = gradient_y
        elif offset == 6:
            gradient = gradient_r20
        elif offset == 7:
            gradient = gradient_z
        elif offset == 8:
            gradient = gradient_r22
        tl.store(
            rotation_gradient_ptr
            + gradient_base
            + (offset // 3) * gradient_stride_row
            + (offset % 3) * gradient_stride_column,
            tl.where(regular, gradient, 0.0),
            valid,
        )


def pole_clamped_angles_backward(alpha_gradient, beta_gradient, gamma_gradient, rotation):
    rotation_gradient = torch.empty_like(rotation)
    tile, warps = frame_angle_tile(rotation)
    pole_clamped_angles_backward_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        alpha_gradient,
        beta_gradient,
        gamma_gradient,
        rotation,
        rotation_gradient,
        *rotation.stride(),
        *rotation_gradient.stride(),
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return rotation_gradient


@triton.jit
def monomial_gradient(
    w,
    x,
    y,
    z,
    index: tl.constexpr,
    axis,
    exponents_w: tl.constexpr,
    exponents_x: tl.constexpr,
    exponents_y: tl.constexpr,
    exponents_z: tl.constexpr,
):
    values = (w, x, y, z)
    exponents = (
        list(exponents_w)[index],
        list(exponents_x)[index],
        list(exponents_y)[index],
        list(exponents_z)[index],
    )
    result = tl.zeros(axis.shape, tl.float32)
    for variable in tl.static_range(4):
        exponent = exponents[variable]
        derivative = exponent
        for local in tl.static_range(4):
            power = tl.maximum(0, exponents[local] - (1 if local == variable else 0))
            derivative *= polynomial_integer_power(values[local], power)
        result += tl.where(axis == variable, derivative, 0.0)
    return result


@triton.jit
def accumulate_gradient(
    coefficients_ptr,
    matrix_gradient_ptr,
    matrix_base,
    offsets: tl.constexpr,
    w,
    x,
    y,
    z,
    axis,
    output_count: tl.constexpr,
    monomial_count: tl.constexpr,
    exponents_w: tl.constexpr,
    exponents_x: tl.constexpr,
    exponents_y: tl.constexpr,
    exponents_z: tl.constexpr,
    block_outputs: tl.constexpr,
):
    output = tl.arange(0, block_outputs)[None, :]
    mask = output < output_count
    offset = tl.full((1, block_outputs), list(offsets)[0], tl.int32)
    for index in tl.static_range(1, output_count):
        offset = tl.where(output == index, list(offsets)[index], offset)
    matrix_gradient = tl.load(
        matrix_gradient_ptr + matrix_base + offset, mask=mask, other=0.0
    ).to(tl.float32)
    accumulator = tl.zeros((4, block_outputs), tl.float32)
    for index in tl.static_range(monomial_count):
        coefficient = tl.load(
            coefficients_ptr + output * monomial_count + index,
            mask=mask,
            other=0.0,
        )
        derivative = monomial_gradient(
            w,
            x,
            y,
            z,
            index,
            axis,
            exponents_w,
            exponents_x,
            exponents_y,
            exponents_z,
        )
        accumulator += derivative * coefficient * matrix_gradient
    return tl.sum(accumulator, axis=1)


@triton.jit
def retained_backward_edge(
    matrix_gradient_ptr,
    q_ptr,
    quadratic_ptr,
    quartic_ptr,
    q_gradient_ptr,
    edge,
    linear_count: tl.constexpr,
    quartic_count: tl.constexpr,
    linear_offsets: tl.constexpr,
    quartic_offsets: tl.constexpr,
):
    q_base = edge * 4
    w = tl.load(q_ptr + q_base).to(tl.float32)
    x = tl.load(q_ptr + q_base + 1).to(tl.float32)
    y = tl.load(q_ptr + q_base + 2).to(tl.float32)
    z = tl.load(q_ptr + q_base + 3).to(tl.float32)
    axis = tl.arange(0, 4)[:, None]
    matrix_base = edge * (1 + linear_count + quartic_count)
    gradient = accumulate_gradient(
        quadratic_ptr,
        matrix_gradient_ptr,
        matrix_base,
        linear_offsets,
        w,
        x,
        y,
        z,
        axis,
        output_count=linear_count,
        monomial_count=QUADRATIC_COUNT_TL,
        exponents_w=QUADRATIC_W_TL,
        exponents_x=QUADRATIC_X_TL,
        exponents_y=QUADRATIC_Y_TL,
        exponents_z=QUADRATIC_Z_TL,
        block_outputs=16,
    )
    gradient += accumulate_gradient(
        quartic_ptr,
        matrix_gradient_ptr,
        matrix_base,
        quartic_offsets,
        w,
        x,
        y,
        z,
        axis,
        output_count=quartic_count,
        monomial_count=QUARTIC_COUNT_TL,
        exponents_w=QUARTIC_W_TL,
        exponents_x=QUARTIC_X_TL,
        exponents_y=QUARTIC_Y_TL,
        exponents_z=QUARTIC_Z_TL,
        block_outputs=32,
    )
    tl.store(q_gradient_ptr + q_base + tl.arange(0, 4), gradient)


@triton.jit
def retained_backward_kernel(
    matrix_gradient_ptr,
    q_ptr,
    quadratic_ptr,
    quartic_ptr,
    q_gradient_ptr,
    linear_count: tl.constexpr,
    quartic_count: tl.constexpr,
    linear_offsets: tl.constexpr,
    quartic_offsets: tl.constexpr,
):
    retained_backward_edge(
        matrix_gradient_ptr, q_ptr, quadratic_ptr, quartic_ptr, q_gradient_ptr, tl.program_id(0),
        linear_count, quartic_count, linear_offsets, quartic_offsets,
    )


@triton.jit
def retained_sparse_backward_kernel(q, seed, quadratic, quartic, coeff, codes, usable, q_out, edges):
    adaptive_quaternion_tile(
        q, None, seed, quadratic, quartic, coeff, codes, usable, None,
        q_out, edges, retained_backward_edge, 1,
    )


def retained_rows_backward(matrix_gradient, q, quadratic, quartic, mmax, compile_visible):
    edges = q.shape[0]
    q_gradient = torch.empty_like(q)
    if use_sparse_quaternion(q, quadratic, quartic, mmax, compile_visible):
        packed = pack_sparse_quaternion(q, quadratic, quartic)
        retained_sparse_backward_kernel[(triton.cdiv(edges, 4),)](
            q, matrix_gradient, quadratic, quartic, *packed, q_gradient, edges, num_warps=4,
        )
    elif compile_visible:
        torch.ops.so2.retained_quaternion_backward(
            matrix_gradient, q, quadratic, quartic, q_gradient, mmax
        )
    elif q.shape[0]:
        retained_backward_kernel[(q.shape[0],)](
            matrix_gradient,
            q,
            quadratic,
            quartic,
            q_gradient,
            **launch_constants(mmax),
            num_warps=4,
        )
    return q_gradient


@triton.jit
def wigner_backward_kernel(
    matrix_gradient_ptr, alpha_ptr, beta_ptr, gamma_ptr, matrix_j_ptr,
    alpha_gradient_ptr, beta_gradient_ptr, gamma_gradient_ptr,
    degree: tl.constexpr, selected_rows: tl.constexpr,
    row_count: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_width: tl.constexpr,
    inverse_gradient_ptr=None, inverse_rescale_ptr=None,
    matrix_stride: tl.constexpr = 0,
    matrix_offset: tl.constexpr = 0,
):
    edge = tl.program_id(0)
    row = tl.arange(0, block_rows)[:, None]
    column = tl.arange(0, block_width)[None, :]
    mask = (row < row_count) & (column < width)
    address = matrix_address(
        edge, row, column, row_count, width, matrix_stride, matrix_offset
    )
    terms = wigner_terms(
        tl.load(alpha_ptr + edge).to(tl.float32),
        tl.load(beta_ptr + edge).to(tl.float32),
        tl.load(gamma_ptr + edge).to(tl.float32),
        matrix_j_ptr, degree, selected_rows,
        row_count, width, block_rows, block_width,
    )
    gradient = tl.load(matrix_gradient_ptr + address, mask=mask, other=0.0).to(tl.float32)
    if inverse_gradient_ptr is not None:
        rescale = tl.load(
            inverse_rescale_ptr + matrix_offset + row * width + column,
            mask=mask,
            other=0.0,
        )
        gradient += rescale * tl.load(
            inverse_gradient_ptr + address, mask=mask, other=0.0
        )
    tl.atomic_add(alpha_gradient_ptr + edge, reduce_angles(gradient * terms[1]))
    tl.atomic_add(beta_gradient_ptr + edge, reduce_angles(gradient * terms[2]))
    tl.atomic_add(gamma_gradient_ptr + edge, reduce_angles(gradient * terms[3]))


def wigner_backward(matrix_gradients, alpha, beta, gamma, matrices_j, rows_by_degree):
    allow_atomic_reduction("rotation/euler/wigner.backward")
    angle_gradients = tuple(torch.zeros_like(angle) for angle in (alpha, beta, gamma))
    edges = alpha.shape[0]
    for degree, metadata in enumerate(rotation_metadata(rows_by_degree)):
        wigner_backward_kernel[(edges,)](
            matrix_gradients[degree],
            alpha, beta, gamma,
            matrices_j[degree],
            *angle_gradients,
            **metadata,
        )
    return angle_gradients


@triton.jit
def retained_degree_vjp(
    alpha, beta, gamma, direct, inverse, scale, matrix, edge,
    compact: tl.constexpr, degree: tl.constexpr, metadata: tl.constexpr,
):
    rows: tl.constexpr = metadata[degree * 5]
    width: tl.constexpr = metadata[degree * 5 + 1]
    block_rows: tl.constexpr = metadata[degree * 5 + 2]
    block_width: tl.constexpr = metadata[degree * 5 + 3]
    offset: tl.constexpr = metadata[degree * 5 + 4]
    selected_rows = retained_tile_rows(degree, rows)
    row = tl.arange(0, block_rows)[:, None]
    col = tl.arange(0, block_width)[None, :]
    row_live, col_live = row < rows, col < width
    live = row_live & col_live
    # Signed row partners are adjacent in the tile. Storage starts at m=0.
    stored = tl.where(row + 1 < rows, row + 1, 0)
    selected = selected_row_indices(stored, selected_rows, rows)
    fa, fg = degree - selected, degree - col
    shape: tl.constexpr = (block_rows, block_width)
    mirror_row = tl.broadcast_to(tl.where(row < rows - 1, row ^ 1, row), shape)
    mirror_col = tl.broadcast_to(tl.where(col_live, width - 1 - col, col), shape)
    local = offset + stored * width + col
    address = edge * compact + local
    seed = tl.load(direct + address, live, 0.0)
    if scale is not None:
        seed += tl.load(inverse + address, live, 0.0) * tl.load(scale + local, live, 0.0)

    # Undo the outer Euler rotations on the cotangent before contraction.
    ap, gp = alpha * fa, gamma * fg
    seed = tl.cos(ap) * seed - tl.sin(ap) * tl.gather(seed, mirror_row, axis=0)
    seed = tl.cos(gp) * seed + tl.sin(gp) * tl.gather(seed, mirror_col, axis=1)

    value = tl.zeros(shape, tl.float32)
    d_beta = tl.zeros(shape, tl.float32)
    for k in tl.static_range(width):
        near = tl.load(matrix + selected * width + k, row_live, 0.0)
        far = tl.load(matrix + selected * width + width - 1 - k, row_live, 0.0)
        frequency = degree - k
        phase = beta * frequency
        cosine, sine = tl.cos(phase), tl.sin(phase)
        coefficient = tl.load(matrix + k * width + col, col_live, 0.0)
        value += (near * cosine - far * sine) * coefficient
        d_beta += (frequency * (-near * sine - far * cosine)) * coefficient

    alpha_seed = -fa * tl.gather(seed, mirror_row, axis=0)
    gamma_seed = fg * tl.gather(seed, mirror_col, axis=1)
    return (
        reduce_angles(alpha_seed * value),
        reduce_angles(seed * d_beta),
        reduce_angles(gamma_seed * value),
    )


@triton.jit
def analytic_retained_backward_kernel(
    angles, matrices, direct, inverse, scale, outputs,
    compact: tl.constexpr, metadata: tl.constexpr,
):
    edge = tl.program_id(0)
    alpha = tl.load(angles[0] + edge)
    beta = tl.load(angles[1] + edge)
    gamma = tl.load(angles[2] + edge)
    da, db, dg = 0.0, 0.0, 0.0
    for degree in tl.static_range(1, len(metadata) // 5):
        a, b, g = retained_degree_vjp(
            alpha, beta, gamma, direct, inverse, scale, matrices[degree], edge,
            compact, degree, metadata,
        )
        da += a
        db += b
        dg += g
    tl.store(outputs[0] + edge, da)
    tl.store(outputs[1] + edge, db)
    tl.store(outputs[2] + edge, dg)


def use_analytic_retained_backward(direct, inverse, angles, matrices, scale, spec):
    if (
        spec.mmax != 2 or not 2 <= spec.lmax <= 6
        or direct.shape[0] < 16384 or not direct.is_cuda or torch.version.hip is not None
        or torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled()
        or torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
        or not supports_factored_retained_rotation(direct.device, spec.lmax)
    ):
        return False
    tensors = (direct, *angles, *matrices)
    if scale is not None:
        tensors += (inverse, scale)
    return all(
        value is not None and value.dtype == torch.float32 and value.device == direct.device
        and value.is_contiguous() and not value.is_neg() for value in tensors
    )


def retained_backward(
    direct_gradient, inverse_gradient, alpha, beta, gamma,
    matrices_j, inverse_rescale, spec,
):
    allow_atomic_reduction("rotation/euler/wigner.retained_backward")
    angles = alpha, beta, gamma
    if use_analytic_retained_backward(
        direct_gradient, inverse_gradient, angles, matrices_j, inverse_rescale, spec,
    ):
        angle_gradients = tuple(torch.empty_like(angle) for angle in angles)
        analytic_retained_backward_kernel[(alpha.numel(),)](
            angles, tuple(matrices_j), direct_gradient, inverse_gradient, inverse_rescale,
            angle_gradients, spec.compact_elements, retained_tile_metadata(spec),
            num_warps=1,
        )
        return angle_gradients
    if use_retained_tiles(alpha, spec, backward=True):
        angle_gradients = tuple(torch.empty_like(angle) for angle in angles)
        gradients = (direct_gradient,) if inverse_gradient is None else (
            direct_gradient, inverse_gradient,
        )
        retained_tiles_kernel[(triton.cdiv(alpha.shape[0], 4),)](
            angles, tuple(matrices_j), inverse_rescale, None, None, gradients, angle_gradients,
            alpha.shape[0], spec.compact_elements, retained_tile_metadata(spec), True,
            num_warps=4,
        )
        return angle_gradients
    angle_gradients = tuple(torch.zeros_like(angle) for angle in (alpha, beta, gamma))
    for degree, metadata in enumerate(retained_metadata(spec)):
        wigner_backward_kernel[(alpha.shape[0],)](
            direct_gradient,
            alpha, beta, gamma,
            matrices_j[degree],
            *angle_gradients,
            inverse_gradient_ptr=inverse_gradient,
            inverse_rescale_ptr=inverse_rescale,
            **metadata,
        )
    return angle_gradients


TRANSPOSED = tl.constexpr((0, *(1 + j * 3 + i for i in range(3) for j in range(3)),
                           *(10 + j * 5 + i for i in range(5) for j in range(5))))


@triton.jit
def cross_vjp(v, g):
    x, y, z = v
    a, b, c, d, e = g
    return (
        a * z + b * y - e * x,
        b * x + 1.7320508075688772 * c * y + d * z,
        a * x + d * y + e * z,
    )


@triton.jit
def axial_vjp(v, g):
    x, y, z = v
    a, b, c, d, e = g
    return (
        1.7320508075688772 * (a * z + b * y - e * x),
        1.7320508075688772 * (b * x + d * z) + 3 * c * y,
        1.7320508075688772 * (a * x + d * y + e * z),
    )


@triton.jit
def column_seed_stride(g, live, stride, j: tl.constexpr):
    return (
        tl.load(g + list(INVERSE)[10 + j] * stride, live, 0),
        tl.load(g + list(INVERSE)[15 + j] * stride, live, 0),
        tl.load(g + list(INVERSE)[20 + j] * stride, live, 0),
        tl.load(g + list(INVERSE)[25 + j] * stride, live, 0),
        tl.load(g + list(INVERSE)[30 + j] * stride, live, 0),
    )


@triton.jit
def matrix_vjp(q, seed, e, live, seed_row, seed_col):
    q_values, s, _, u, v, t = frame(q, e, live)
    w, x, y, z = q_values
    g = seed + e * seed_row
    c0 = column_seed_stride(g, live, seed_col, 0)
    c1 = column_seed_stride(g, live, seed_col, 1)
    c2 = column_seed_stride(g, live, seed_col, 2)
    c3 = column_seed_stride(g, live, seed_col, 3)
    c4 = column_seed_stride(g, live, seed_col, 4)
    du0, du1, du2 = cross_vjp(t, c0), cross_vjp(v, c1), cross_vjp(u, c4)
    dv0, dv1, dv2 = cross_vjp(u, c1), axial_vjp(v, c2), cross_vjp(t, c3)
    dt0, dt1, dt2 = cross_vjp(u, c0), cross_vjp(v, c3), cross_vjp(t, c4)
    du = (du0[0] + du1[0] - du2[0], du0[1] + du1[1] - du2[1], du0[2] + du1[2] - du2[2])
    dv = (dv0[0] + dv1[0] + dv2[0], dv0[1] + dv1[1] + dv2[1], dv0[2] + dv1[2] + dv2[2])
    dt = (dt0[0] + dt1[0] + dt2[0], dt0[1] + dt1[1] + dt2[1], dt0[2] + dt1[2] + dt2[2])
    ds = du[0] + dv[1] + dt[2] - c2[2] * s
    a00 = du[0] + tl.load(g + list(INVERSE)[1] * seed_col, live, 0)
    a01 = dv[0] + tl.load(g + list(INVERSE)[2] * seed_col, live, 0)
    a02 = dt[0] + tl.load(g + list(INVERSE)[3] * seed_col, live, 0)
    a10 = du[1] + tl.load(g + list(INVERSE)[4] * seed_col, live, 0)
    a11 = dv[1] + tl.load(g + list(INVERSE)[5] * seed_col, live, 0)
    a12 = dt[1] + tl.load(g + list(INVERSE)[6] * seed_col, live, 0)
    a20 = du[2] + tl.load(g + list(INVERSE)[7] * seed_col, live, 0)
    a21 = dv[2] + tl.load(g + list(INVERSE)[8] * seed_col, live, 0)
    a22 = dt[2] + tl.load(g + list(INVERSE)[9] * seed_col, live, 0)
    skew_x, skew_y, skew_z = a21 - a12, a02 - a20, a10 - a01
    sym_x, sym_y, sym_z = a21 + a12, a02 + a20, a10 + a01
    return (
        2 * (w * ds + x * skew_x + y * skew_y + z * skew_z),
        2 * (x * (ds - 2 * (a11 + a22)) + y * sym_z + z * sym_y + w * skew_x),
        2 * (y * (ds - 2 * (a00 + a22)) + x * sym_z + z * sym_x + w * skew_y),
        2 * (z * (ds - 2 * (a00 + a11)) + x * sym_y + y * sym_x + w * skew_z),
    )


@triton.jit
def fused_rotation_vjp_kernel(q, vectors, gamma, seed, out, edges, seed_row, seed_col, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    live = e < edges
    dq = matrix_vjp(q, seed, e, live, seed_row, seed_col)
    base = vectors + e * 3
    ex = tl.load(base, live, 0)[:, None]
    ey = tl.load(base + 1, live, 0)[:, None]
    ez = tl.load(base + 2, live, 0)[:, None]
    angle = tl.load(gamma + e, live, 0)[:, None]
    axis = tl.arange(0, 4)[None, :]
    derivatives = edge_quaternion_jets(ex, ey, ez, angle, 0.0, 0.0, 0.0, axis, second_order=False)
    grad = (dq[0][:, None] * derivatives[0][1] + dq[1][:, None] * derivatives[1][1]
            + dq[2][:, None] * derivatives[2][1] + dq[3][:, None] * derivatives[3][1])
    tl.store(out + e[:, None] * 3 + axis, grad, live[:, None] & (axis < 3))


@triton.jit
def seed_value(g, h, live, stride, inverse_stride, column: tl.constexpr):
    transposed: tl.constexpr = list(TRANSPOSED)[column]
    return tl.load(g + column * stride, live, 0) + tl.load(h + transposed * inverse_stride, live, 0)


@triton.jit
def column_seed_pair(g, h, live, stride, inverse_stride, j: tl.constexpr):
    return (
        seed_value(g, h, live, stride, inverse_stride, 10 + j),
        seed_value(g, h, live, stride, inverse_stride, 15 + j),
        seed_value(g, h, live, stride, inverse_stride, 20 + j),
        seed_value(g, h, live, stride, inverse_stride, 25 + j),
        seed_value(g, h, live, stride, inverse_stride, 30 + j),
    )


@triton.jit
def backward_pair(q, seed, inverse_seed, out, edges, seed_row, seed_col,
                  inverse_row, inverse_col, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    live = e < edges
    q_values, s, _, u, v, t = frame(q, e, live)
    w, x, y, z = q_values
    g = seed + e * seed_row
    h = inverse_seed + e * inverse_row
    c0 = column_seed_pair(g, h, live, seed_col, inverse_col, 0)
    c1 = column_seed_pair(g, h, live, seed_col, inverse_col, 1)
    c2 = column_seed_pair(g, h, live, seed_col, inverse_col, 2)
    c3 = column_seed_pair(g, h, live, seed_col, inverse_col, 3)
    c4 = column_seed_pair(g, h, live, seed_col, inverse_col, 4)
    du0, du1, du2 = cross_vjp(t, c0), cross_vjp(v, c1), cross_vjp(u, c4)
    dv0, dv1, dv2 = cross_vjp(u, c1), axial_vjp(v, c2), cross_vjp(t, c3)
    dt0, dt1, dt2 = cross_vjp(u, c0), cross_vjp(v, c3), cross_vjp(t, c4)
    du = (du0[0] + du1[0] - du2[0], du0[1] + du1[1] - du2[1], du0[2] + du1[2] - du2[2])
    dv = (dv0[0] + dv1[0] + dv2[0], dv0[1] + dv1[1] + dv2[1], dv0[2] + dv1[2] + dv2[2])
    dt = (dt0[0] + dt1[0] + dt2[0], dt0[1] + dt1[1] + dt2[1], dt0[2] + dt1[2] + dt2[2])
    ds = du[0] + dv[1] + dt[2] - c2[2] * s
    a00 = du[0] + seed_value(g, h, live, seed_col, inverse_col, 1)
    a01 = dv[0] + seed_value(g, h, live, seed_col, inverse_col, 2)
    a02 = dt[0] + seed_value(g, h, live, seed_col, inverse_col, 3)
    a10 = du[1] + seed_value(g, h, live, seed_col, inverse_col, 4)
    a11 = dv[1] + seed_value(g, h, live, seed_col, inverse_col, 5)
    a12 = dt[1] + seed_value(g, h, live, seed_col, inverse_col, 6)
    a20 = du[2] + seed_value(g, h, live, seed_col, inverse_col, 7)
    a21 = dv[2] + seed_value(g, h, live, seed_col, inverse_col, 8)
    a22 = dt[2] + seed_value(g, h, live, seed_col, inverse_col, 9)
    skew_x, skew_y, skew_z = a21 - a12, a02 - a20, a10 - a01
    sym_x, sym_y, sym_z = a21 + a12, a02 + a20, a10 + a01
    base = out + e * 4
    tl.store(base, 2 * (w * ds + x * skew_x + y * skew_y + z * skew_z), live)
    tl.store(base + 1, 2 * (x * (ds - 2 * (a11 + a22)) + y * sym_z + z * sym_y + w * skew_x), live)
    tl.store(base + 2, 2 * (y * (ds - 2 * (a00 + a22)) + x * sym_z + z * sym_x + w * skew_y), live)
    tl.store(base + 3, 2 * (z * (ds - 2 * (a00 + a11)) + x * sym_y + y * sym_x + w * skew_z), live)
