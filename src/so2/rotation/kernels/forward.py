from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

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
    edge_quaternion_values,
    frame_angle_edges,
    frame_angle_tile,
    launch_constants,
    matrix_address,
    monomial_values,
    polynomial_values,
    pack_sparse_quaternion,
    retained_metadata,
    retained_tile_metadata,
    retained_tiles_kernel,
    supports_factored_retained_rotation,
    use_retained_tiles,
    rotation_metadata,
    rotation_values,
    use_quaternion_tiles,
    use_sparse_quaternion,
    wigner_terms,
)


@triton.jit
def edge_quaternion_forward_kernel(edge_ptr, gamma_ptr, output_ptr, edges=0, tile_edges: tl.constexpr = 1):
    edge, live = frame_angle_edges(edges, tile_edges)
    base = edge * 3
    values = edge_quaternion_values(
        tl.load(edge_ptr + base).to(tl.float32),
        tl.load(edge_ptr + base + 1).to(tl.float32),
        tl.load(edge_ptr + base + 2).to(tl.float32),
        tl.load(gamma_ptr + edge).to(tl.float32),
    )
    for component in tl.static_range(4):
        tl.store(output_ptr + edge * 4 + component, values[component], live)


def edge_quaternion_forward(edge_vectors, gamma, compile_visible):
    output = edge_vectors.new_empty((edge_vectors.shape[0], 4))
    if compile_visible:
        torch.ops.so2.edge_quaternion_forward(edge_vectors, gamma, output)
    else:
        tile, warps = (64, 2) if use_quaternion_tiles(edge_vectors) else (1, 1)
        edge_quaternion_forward_kernel[(triton.cdiv(edge_vectors.shape[0], tile),)](
            edge_vectors, gamma, output, edge_vectors.shape[0], tile, num_warps=warps,
        )
    return output


@triton.jit
def angles_forward_kernel(
    rotation_ptr,
    alpha_ptr,
    beta_ptr,
    gamma_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    edges,
    tile_edges: tl.constexpr,
):
    edge, valid = frame_angle_edges(edges, tile_edges)
    base = rotation_ptr + edge * rotation_stride_edge
    x0 = tl.load(base + rotation_stride_column).to(tl.float32)
    x1 = tl.load(base + rotation_stride_row + rotation_stride_column).to(tl.float32)
    x2 = tl.load(base + 2 * rotation_stride_row + rotation_stride_column).to(tl.float32)
    alpha = libdevice.atan2(x0, x2)
    # The transverse components retain precision as the polar cosine approaches ±1.
    beta = libdevice.atan2(tl.sqrt(x0 * x0 + x2 * x2), x1)
    beta = tl.where(
        x1 > ROTATION_MASK_THRESHOLD_TL,
        0.0,
        tl.where(
            x1 < -ROTATION_MASK_THRESHOLD_TL,
            3.141592653589793,
            beta,
        ),
    )
    alpha_cos = tl.cos(alpha)
    alpha_sin = tl.sin(alpha)
    gamma = libdevice.atan2(
        alpha_cos * tl.load(base + 2 * rotation_stride_column)
        - alpha_sin
        * tl.load(base + 2 * rotation_stride_row + 2 * rotation_stride_column),
        alpha_cos * tl.load(base) - alpha_sin * tl.load(base + 2 * rotation_stride_row),
    )
    tl.store(alpha_ptr + edge, alpha, valid)
    tl.store(beta_ptr + edge, beta, valid)
    tl.store(gamma_ptr + edge, gamma, valid)


def masked_angles_forward(rotation):
    common = {"device": rotation.device, "dtype": rotation.dtype}
    alpha = torch.empty((rotation.shape[0],), **common)
    beta = torch.empty_like(alpha)
    gamma = torch.empty_like(alpha)
    tile, warps = frame_angle_tile(rotation)
    angles_forward_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        rotation,
        alpha,
        beta,
        gamma,
        rotation_stride_edge=rotation.stride(0),
        rotation_stride_row=rotation.stride(1),
        rotation_stride_column=rotation.stride(2),
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return alpha, beta, gamma


@triton.jit
def monomial_forward_kernel(
    q_ptr,
    exponents_ptr,
    output_ptr,
    monomial_count,
    block_monomials: tl.constexpr,
    max_power: tl.constexpr,
):
    edge = tl.program_id(0)
    index = tl.arange(0, block_monomials)
    mask = index < monomial_count
    value, _, _ = monomial_values(q_ptr, exponents_ptr, edge, index, mask, max_power)
    tl.store(output_ptr + edge * monomial_count + index, value, mask=mask)


@triton.jit
def pole_clamped_angles_forward_kernel(
    rotation_ptr,
    alpha_ptr,
    beta_ptr,
    gamma_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
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
    tl.store(alpha_ptr + edge, values[7], valid)
    tl.store(beta_ptr + edge, values[8], valid)
    tl.store(gamma_ptr + edge, values[9], valid)


def pole_clamped_angles_forward(rotation):
    edges = rotation.shape[0]
    alpha = torch.empty(edges, device=rotation.device, dtype=rotation.dtype)
    beta = torch.empty_like(alpha)
    gamma = torch.empty_like(alpha)
    tile, warps = frame_angle_tile(rotation)
    pole_clamped_angles_forward_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        rotation,
        alpha,
        beta,
        gamma,
        *rotation.stride(),
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return alpha, beta, gamma


@triton.jit
def retained_forward_edge(
    q_ptr,
    quadratic_ptr,
    quartic_ptr,
    output_ptr,
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
    output_base = edge * (1 + linear_count + quartic_count)
    tl.store(output_ptr + output_base, 1.0)
    linear, linear_mask = polynomial_values(
        quadratic_ptr,
        w,
        x,
        y,
        z,
        output_count=linear_count,
        monomial_count=QUADRATIC_COUNT_TL,
        exponents_w=QUADRATIC_W_TL,
        exponents_x=QUADRATIC_X_TL,
        exponents_y=QUADRATIC_Y_TL,
        exponents_z=QUADRATIC_Z_TL,
        block_outputs=16,
    )
    linear_output = tl.arange(0, 16)
    linear_offset = tl.full((16,), list(linear_offsets)[0], tl.int32)
    for index in tl.static_range(1, linear_count):
        linear_offset = tl.where(
            linear_output == index, list(linear_offsets)[index], linear_offset
        )
    tl.store(output_ptr + output_base + linear_offset, linear, mask=linear_mask)
    quartic, quartic_mask = polynomial_values(
        quartic_ptr,
        w,
        x,
        y,
        z,
        output_count=quartic_count,
        monomial_count=QUARTIC_COUNT_TL,
        exponents_w=QUARTIC_W_TL,
        exponents_x=QUARTIC_X_TL,
        exponents_y=QUARTIC_Y_TL,
        exponents_z=QUARTIC_Z_TL,
        block_outputs=32,
    )
    quartic_output = tl.arange(0, 32)
    quartic_offset = 1 + linear_count + quartic_output
    tl.store(output_ptr + output_base + quartic_offset, quartic, mask=quartic_mask)


@triton.jit
def retained_forward_kernel(
    q_ptr,
    quadratic_ptr,
    quartic_ptr,
    output_ptr,
    linear_count: tl.constexpr,
    quartic_count: tl.constexpr,
    linear_offsets: tl.constexpr,
    quartic_offsets: tl.constexpr,
):
    retained_forward_edge(
        q_ptr, quadratic_ptr, quartic_ptr, output_ptr, tl.program_id(0),
        linear_count, quartic_count, linear_offsets, quartic_offsets,
    )


@triton.jit
def retained_sparse_forward_kernel(q, quadratic, quartic, coeff, codes, usable, matrix, edges):
    adaptive_quaternion_tile(
        q, None, None, quadratic, quartic, coeff, codes, usable, matrix,
        None, edges, retained_forward_edge, 0,
    )


def retained_rows_forward(q, quadratic, quartic, mmax, compile_visible):
    edges = q.shape[0]
    elements = 1 + quadratic.shape[0] + quartic.shape[0]
    output = q.new_empty((edges, elements))
    if use_sparse_quaternion(q, quadratic, quartic, mmax, compile_visible):
        packed = pack_sparse_quaternion(q, quadratic, quartic)
        retained_sparse_forward_kernel[(triton.cdiv(edges, 4),)](
            q, quadratic, quartic, *packed, output, edges, num_warps=4,
        )
    elif compile_visible:
        torch.ops.so2.retained_quaternion_forward(q, quadratic, quartic, output, mmax)
    elif q.shape[0]:
        retained_forward_kernel[(q.shape[0],)](
            q, quadratic, quartic, output, **launch_constants(mmax), num_warps=4
        )
    return output


@triton.jit
def wigner_forward_kernel(
    alpha_ptr, beta_ptr, gamma_ptr, matrix_j_ptr, output_ptr,
    degree: tl.constexpr, selected_rows: tl.constexpr,
    row_count: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_width: tl.constexpr,
    inverse_ptr=None, inverse_rescale_ptr=None,
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
    tl.store(output_ptr + address, terms[0], mask=mask)
    if inverse_ptr is not None:
        rescale = tl.load(
            inverse_rescale_ptr + matrix_offset + row * width + column,
            mask=mask,
            other=0.0,
        )
        tl.store(inverse_ptr + address, terms[0] * rescale, mask=mask)


def wigner_forward(alpha, beta, gamma, matrices_j, rows_by_degree):
    edges = alpha.shape[0]
    outputs = []
    for degree, metadata in enumerate(rotation_metadata(rows_by_degree)):
        outputs.append(
            alpha.new_empty((edges, metadata["row_count"], metadata["width"]))
        )
        wigner_forward_kernel[(edges,)](
            alpha, beta, gamma,
            matrices_j[degree],
            outputs[degree],
            **metadata,
        )
    return tuple(outputs)


@triton.jit
def factored_retained_degree(
    alpha, beta, gamma, matrix_j, scale, direct, inverse, edge, edge_mask,
    compact: tl.constexpr, degree: tl.constexpr, tile: tl.constexpr,
):
    # The retained rows keep orders 0, +1, -1, +2, -2 up to the degree.
    rows: tl.constexpr = 2 * min(degree, 2) + 1
    width: tl.constexpr = 2 * degree + 1
    block_rows: tl.constexpr = triton.next_power_of_2(rows)
    block_width: tl.constexpr = triton.next_power_of_2(width)
    # The first two blocks occupy ten coefficients. Higher blocks have five rows.
    start: tl.constexpr = degree if degree < 2 else 10 + 5 * (degree * degree - 4)
    row = tl.arange(0, block_rows)[:, None]
    column = tl.arange(0, block_width)[None, :]
    row_mask, column_mask = row < rows, column < width
    # Use adjacent signed orders during calculation, then write the zero order first.
    stored_row = tl.where(row < rows - 1, row + 1, 0)
    order = (stored_row + 1) // 2
    selected = degree + tl.where(stored_row % 2 != 0, order, -order)

    # Contract J Z(beta) J once, then apply the outer rotations by exchanging
    # signed row and column partners within the tile.
    value = tl.zeros((tile, block_rows, block_width), tl.float32)
    for k in tl.static_range(width):
        near = tl.load(matrix_j + selected * width + k, row_mask, 0.0)[None, :, :]
        far = tl.load(matrix_j + selected * width + width - 1 - k, row_mask, 0.0)[None, :, :]
        beta_phase = beta * (degree - k)
        middle = near * tl.cos(beta_phase) - far * tl.sin(beta_phase)
        weight = tl.load(matrix_j + k * width + column, column_mask, 0.0)[None, :, :]
        value += middle * weight

    mirror_row = tl.where(row < rows - 1, row ^ 1, row)
    mirror_row = tl.broadcast_to(mirror_row[None, :, :], (tile, block_rows, block_width))
    alpha_phase = alpha * (degree - selected)[None, :, :]
    value = tl.cos(alpha_phase) * value + tl.sin(alpha_phase) * tl.gather(value, mirror_row, axis=1)
    mirror_column = tl.where(column_mask, width - 1 - column, column)
    mirror_column = tl.broadcast_to(mirror_column[None, :, :], (tile, block_rows, block_width))
    gamma_phase = gamma * (degree - column)[None, :, :]
    value = tl.cos(gamma_phase) * value - tl.sin(gamma_phase) * tl.gather(value, mirror_column, axis=2)
    local = start + stored_row * width + column
    address = edge[:, None, None] * compact + local[None, :, :]
    mask = edge_mask[:, None, None] & (row_mask & column_mask)[None, :, :]
    tl.store(direct + address, value, mask)
    if scale is not None:
        factor = tl.load(scale + local, row_mask & column_mask, 0.0)
        tl.store(inverse + address, value * factor[None, :, :], mask)


@triton.jit
def factored_retained_forward_kernel(
    angles, basis, scale, direct, inverse, edges,
    compact: tl.constexpr, lmax: tl.constexpr, tile: tl.constexpr, degree: tl.constexpr,
):
    edge = tl.program_id(0) * tile + tl.arange(0, tile)
    mask = edge < edges
    alpha = tl.load(angles[0] + edge, mask, 0.0)[:, None, None]
    beta = tl.load(angles[1] + edge, mask, 0.0)[:, None, None]
    gamma = tl.load(angles[2] + edge, mask, 0.0)[:, None, None]
    if degree < 0:
        for l in tl.static_range(lmax + 1):
            factored_retained_degree(
                alpha, beta, gamma, basis[l], scale, direct, inverse, edge, mask, compact, l, tile,
            )
    else:
        factored_retained_degree(
            alpha, beta, gamma, basis[degree], scale, direct, inverse, edge, mask, compact, degree, tile,
        )


def use_factored_retained_forward(alpha, spec):
    return (
        spec.mmax == 2
        and 2 <= spec.lmax <= 6
        and alpha.shape[0] >= 16384
        and alpha.dtype == torch.float32
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and supports_factored_retained_rotation(alpha.device, spec.lmax)
    )


def retained_forward(alpha, beta, gamma, matrices_j, inverse_rescale, spec):
    direct = alpha.new_empty((alpha.shape[0], spec.compact_elements))
    inverse = torch.empty_like(direct) if inverse_rescale is not None else None
    if use_factored_retained_forward(alpha, spec):
        # Wider degrees split across launches to keep each program small.
        tile = 2 if spec.lmax <= 4 else 1
        degrees = (-1,) if tile == 2 else range(spec.lmax + 1)
        for degree in degrees:
            factored_retained_forward_kernel[(triton.cdiv(alpha.shape[0], tile),)](
                (alpha, beta, gamma), tuple(matrices_j), inverse_rescale, direct, inverse,
                alpha.shape[0], spec.compact_elements, spec.lmax, tile, degree, num_warps=tile,
            )
    elif use_retained_tiles(alpha, spec):
        retained_tiles_kernel[(triton.cdiv(alpha.shape[0], 4),)](
            (alpha, beta, gamma), tuple(matrices_j), inverse_rescale, direct, inverse, (), (),
            alpha.shape[0], spec.compact_elements, retained_tile_metadata(spec), False,
            num_warps=4,
        )
    else:
        for degree, metadata in enumerate(retained_metadata(spec)):
            wigner_forward_kernel[(alpha.shape[0],)](
                alpha, beta, gamma,
                matrices_j[degree],
                direct,
                inverse_ptr=inverse,
                inverse_rescale_ptr=inverse_rescale,
                **metadata,
            )
    return (direct, inverse) if inverse is not None else (direct,)


degree_one = (1, 2, 0)
degree_two = (2, 3, 1, 4, 0)
retained = (0, *(1 + 3 * row + i for row in degree_one for i in range(3)),
            *(10 + 5 * row + i for row in degree_two for i in range(5)))
INVERSE = tl.constexpr(tuple(retained.index(i) for i in range(35)))


@triton.jit
def cross(u, v, sqrt3):
    # Rotation of the symmetrized product of two distinct Cartesian axes.
    ux, uy, uz = u
    vx, vy, vz = v
    return (
        ux * vz + vx * uz,
        ux * vy + vx * uy,
        sqrt3 * uy * vy,
        uy * vz + vy * uz,
        uz * vz - ux * vx,
    )


@triton.jit
def axial(v, norm_fourth, sqrt3):
    vx, vy, vz = v
    return (
        sqrt3 * vx * vz,
        sqrt3 * vx * vy,
        1.5 * (vy * vy) - 0.5 * norm_fourth,
        sqrt3 * vy * vz,
        (0.5 * sqrt3) * (vz * vz - vx * vx),
    )


@triton.jit
def difference(u, v, sqrt3):
    ux, uy, uz = u
    vx, vy, vz = v
    return (
        vx * vz - ux * uz,
        vx * vy - ux * uy,
        (0.5 * sqrt3) * (vy * vy - uy * uy),
        vy * vz - uy * uz,
        0.5 * (vz * vz - uz * uz - vx * vx + ux * ux),
    )


@triton.jit
def frame(q, e, live):
    base = q + e * 4
    w = tl.load(base, live, 0)
    x = tl.load(base + 1, live, 0)
    y = tl.load(base + 2, live, 0)
    z = tl.load(base + 3, live, 0)
    x2, y2, z2 = x * x, y * y, z * z
    xx, yy, zz = 2 * (y2 + z2), 2 * (x2 + z2), 2 * (x2 + y2)
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r01, r02 = 2 * (xy - wz), 2 * (xz + wy)
    r10, r12 = 2 * (xy + wz), 2 * (yz - wx)
    r20, r21 = 2 * (xz - wy), 2 * (yz + wx)
    s = w * w + x2 + y2 + z2
    u = (s - xx, r10, r20)
    v = (r01, s - yy, r21)
    t = (r02, r12, s - zz)
    return (w, x, y, z), s, (xx, yy, zz), u, v, t


@triton.jit
def frame_values(w, x, y, z):
    x2, y2, z2 = x * x, y * y, z * z
    xx, yy, zz = 2 * (y2 + z2), 2 * (x2 + z2), 2 * (x2 + y2)
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r01, r02 = 2 * (xy - wz), 2 * (xz + wy)
    r10, r12 = 2 * (xy + wz), 2 * (yz - wx)
    r20, r21 = 2 * (xz - wy), 2 * (yz + wx)
    s = w * w + x2 + y2 + z2
    u = (s - xx, r10, r20)
    v = (r01, s - yy, r21)
    t = (r02, r12, s - zz)
    return (w, x, y, z), s, (xx, yy, zz), u, v, t


@triton.jit
def fused_rotation_kernel(vectors, gamma, out, q, edges, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    live = e < edges
    base = vectors + e * 3
    ex = tl.load(base, live, 0)
    ey = tl.load(base + 1, live, 0)
    ez = tl.load(base + 2, live, 0)
    angle = tl.load(gamma + e, live, 0)
    w, x, y, z = edge_quaternion_values(ex, ey, ez, angle)
    q_base = q + e * 4
    tl.store(q_base, w, live)
    tl.store(q_base + 1, x, live)
    tl.store(q_base + 2, y, live)
    tl.store(q_base + 3, z, live)
    _, s, diag, u, v, t = frame_values(w, x, y, z)
    dx, dy, dz = diag
    d1 = (1 - dx, v[0], t[0], u[1], 1 - dy, t[1], u[2], v[2], 1 - dz)
    sqrt3: tl.constexpr = 1.7320508075688772
    d2 = (cross(u, t, sqrt3), cross(u, v, sqrt3), axial(v, s * s, sqrt3),
          cross(v, t, sqrt3), difference(u, t, sqrt3))
    row = out + e * 35
    tl.store(row, 1.0, live)
    for i in tl.static_range(9):
        tl.store(row + list(INVERSE)[1 + i], d1[i], live)
    for j in tl.static_range(5):
        for i in tl.static_range(5):
            tl.store(row + list(INVERSE)[10 + i * 5 + j], d2[j][i], live)


@triton.jit
def forward_pair(q, out, inverse, edges, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    live = e < edges
    _, s, diag, u, v, t = frame(q, e, live)
    dx, dy, dz = diag
    d1 = (1 - dx, v[0], t[0], u[1], 1 - dy, t[1], u[2], v[2], 1 - dz)
    sqrt3: tl.constexpr = 1.7320508075688772
    d2 = (cross(u, t, sqrt3), cross(u, v, sqrt3), axial(v, s * s, sqrt3),
          cross(v, t, sqrt3), difference(u, t, sqrt3))
    row = out + e * 35
    transposed = inverse + e * 35
    tl.store(row, 1.0, live)
    tl.store(transposed, 1.0, live)
    for i in tl.static_range(9):
        tl.store(row + 1 + i, d1[i], live)
        tl.store(transposed + 1 + (i % 3) * 3 + i // 3, d1[i], live)
    for j in tl.static_range(5):
        for i in tl.static_range(5):
            tl.store(row + 10 + i * 5 + j, d2[j][i], live)
            tl.store(transposed + 10 + j * 5 + i, d2[j][i], live)
