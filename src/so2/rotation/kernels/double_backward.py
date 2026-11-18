from __future__ import annotations

from importlib.util import find_spec

import torch
import triton
import triton.language as tl
from torch import Tensor

from ...utils.tensor_contracts import allow_atomic_reduction
from .backward import monomial_gradient
from .utils import (
    adaptive_quaternion_tile,
    canonical_l6_fourier,
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
    monomials_integer_power,
    polynomial_integer_power,
    reduce_angles,
    pack_sparse_quaternion,
    retained_metadata,
    retained_tile_metadata,
    retained_tile_rows,
    rotation_metadata,
    rotation_values,
    use_quaternion_tiles,
    use_sparse_quaternion,
    selected_row_indices,
    wigner_terms,
)


@triton.jit
def edge_quaternion_hvp_kernel(
    q_gradient_ptr,
    edge_ptr,
    gamma_ptr,
    edge_direction_ptr,
    q_direction_ptr,
    edge_hvp_ptr,
    edges=0,
    tile_edges: tl.constexpr = 1,
):
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
        tl.load(edge_direction_ptr + base).to(tl.float32),
        tl.load(edge_direction_ptr + base + 1).to(tl.float32),
        tl.load(edge_direction_ptr + base + 2).to(tl.float32),
        axis,
        second_order=True,
    )
    accumulator_shape: tl.constexpr = (4,) if tile_edges == 1 else (tile_edges, 4)
    edge_hvp = tl.zeros(accumulator_shape, tl.float32)
    for component in tl.static_range(4):
        upstream = tl.load(q_gradient_ptr + edge * 4 + component).to(tl.float32)
        tl.store(q_direction_ptr + edge * 4 + component, jets[component][2], live)
        edge_hvp += upstream * jets[component][3]
    tl.store(edge_hvp_ptr + base + axis, edge_hvp, mask=live & (axis < 3))


def edge_quaternion_hvp(q_gradient, edge_vectors, gamma, edge_direction, compile_visible):
    q_direction = torch.empty_like(q_gradient)
    edge_hvp = torch.empty_like(edge_vectors)
    if compile_visible:
        torch.ops.so2.edge_quaternion_hvp(
            q_gradient,
            edge_vectors,
            gamma,
            edge_direction,
            q_direction,
            edge_hvp,
        )
    else:
        tile, warps = (16, 2) if use_quaternion_tiles(edge_vectors) else (1, 1)
        edge_quaternion_hvp_kernel[(triton.cdiv(edge_vectors.shape[0], tile),)](
            q_gradient,
            edge_vectors,
            gamma,
            edge_direction,
            q_direction,
            edge_hvp,
            edge_vectors.shape[0],
            tile,
            num_warps=warps,
        )
    return q_direction, edge_hvp


@triton.jit
def angles_hvp_kernel(
    alpha_gradient_ptr,
    beta_gradient_ptr,
    gamma_gradient_ptr,
    rotation_ptr,
    rotation_direction_ptr,
    alpha_direction_ptr,
    beta_direction_ptr,
    gamma_direction_ptr,
    rotation_hvp_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    direction_stride_edge,
    direction_stride_row,
    direction_stride_column,
    hvp_stride_edge,
    hvp_stride_row,
    hvp_stride_column,
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
        rotation_direction_ptr,
        edge,
        axis,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        True,
    )
    active = (yprod > -ROTATION_MASK_THRESHOLD_TL) & (
        yprod < ROTATION_MASK_THRESHOLD_TL
    )
    alpha_gradient = tl.load(alpha_gradient_ptr + edge).to(tl.float32)
    beta_gradient = tl.load(beta_gradient_ptr + edge).to(tl.float32)
    gamma_gradient = tl.load(gamma_gradient_ptr + edge).to(tl.float32)
    tl.store(alpha_direction_ptr + edge, tl.where(active, alpha[2], 0.0), valid)
    tl.store(beta_direction_ptr + edge, tl.where(active, beta[2], 0.0), valid)
    tl.store(gamma_direction_ptr + edge, tl.where(active, gamma[2], 0.0), valid)
    rotation_hvp = tl.where(
        active,
        alpha_gradient * alpha[3] + beta_gradient * beta[3] + gamma_gradient * gamma[3],
        0.0,
    )
    tl.store(
        rotation_hvp_ptr
        + edge * hvp_stride_edge
        + (axis // 3) * hvp_stride_row
        + (axis % 3) * hvp_stride_column,
        rotation_hvp,
        mask=mask,
    )


def masked_angles_hvp(
    alpha_gradient,
    beta_gradient,
    gamma_gradient,
    rotation,
    rotation_direction,
):
    common = {"device": rotation.device, "dtype": rotation.dtype}
    alpha_direction = torch.empty((rotation.shape[0],), **common)
    beta_direction = torch.empty_like(alpha_direction)
    gamma_direction = torch.empty_like(alpha_direction)
    rotation_hvp = torch.empty_like(rotation)
    tile, warps = frame_angle_tile(rotation, masked_derivatives=True)
    angles_hvp_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        alpha_gradient,
        beta_gradient,
        gamma_gradient,
        rotation,
        rotation_direction,
        alpha_direction,
        beta_direction,
        gamma_direction,
        rotation_hvp,
        rotation_stride_edge=rotation.stride(0),
        rotation_stride_row=rotation.stride(1),
        rotation_stride_column=rotation.stride(2),
        direction_stride_edge=rotation_direction.stride(0),
        direction_stride_row=rotation_direction.stride(1),
        direction_stride_column=rotation_direction.stride(2),
        hvp_stride_edge=rotation_hvp.stride(0),
        hvp_stride_row=rotation_hvp.stride(1),
        hvp_stride_column=rotation_hvp.stride(2),
        block_variables=16,
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return alpha_direction, beta_direction, gamma_direction, rotation_hvp


@triton.jit
def monomial_second_direction(
    exponents, values, direction, axis, max_power: tl.constexpr
):
    result = tl.zeros(exponents[0].shape, tl.float32)
    for column in tl.static_range(4):
        term = tl.full(exponents[0].shape, 1.0, tl.float32)
        row_exponent = tl.zeros(exponents[0].shape, tl.int32)
        column_exponent = exponents[column]
        for variable in tl.static_range(4):
            exponent = exponents[variable]
            row_exponent = tl.where(axis == variable, exponent, row_exponent)
            decrement = tl.where(axis == variable, 1, 0) + (
                1 if variable == column else 0
            )
            term *= monomials_integer_power(
                values[variable], tl.maximum(0, exponent - decrement), max_power
            )
        factor = row_exponent * (column_exponent - tl.where(axis == column, 1, 0))
        result += factor.to(tl.float32) * term * direction[column]
    return result


@triton.jit
def monomial_direction_kernel(
    q_ptr,
    q_direction_ptr,
    exponents_ptr,
    output_ptr,
    monomial_count,
    block_monomials: tl.constexpr,
    max_power: tl.constexpr,
):
    edge = tl.program_id(0)
    index = tl.arange(0, block_monomials)
    mask = index < monomial_count
    _, exponents, values = monomial_values(
        q_ptr, exponents_ptr, edge, index, mask, max_power
    )
    direction = (
        tl.load(q_direction_ptr + edge * 4).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 1).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 2).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 3).to(tl.float32),
    )
    result = tl.zeros(index.shape, tl.float32)
    for axis in tl.static_range(4):
        result += (
            monomial_derivative(exponents, values, axis, max_power) * direction[axis]
        )
    tl.store(output_ptr + edge * monomial_count + index, result, mask=mask)


@triton.jit
def monomial_q_hvp_kernel(
    gradient_ptr,
    q_ptr,
    q_direction_ptr,
    exponents_ptr,
    q_hvp_ptr,
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
    direction = (
        tl.load(q_direction_ptr + edge * 4).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 1).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 2).to(tl.float32),
        tl.load(q_direction_ptr + edge * 4 + 3).to(tl.float32),
    )
    gradient = tl.load(
        gradient_ptr + edge * monomial_count + index,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    second = monomial_second_direction(exponents, values, direction, axis, max_power)
    tl.store(q_hvp_ptr + edge * 4 + axis, tl.sum(gradient * second))


@triton.jit
def pole_clamped_angles_hvp_kernel(
    alpha_gradient_ptr,
    beta_gradient_ptr,
    gamma_gradient_ptr,
    rotation_ptr,
    rotation_direction_ptr,
    alpha_direction_ptr,
    beta_direction_ptr,
    gamma_direction_ptr,
    rotation_hvp_ptr,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    direction_stride_edge,
    direction_stride_row,
    direction_stride_column,
    hvp_stride_edge,
    hvp_stride_row,
    hvp_stride_column,
    edges,
    tile_edges: tl.constexpr,
):
    edge, valid = frame_angle_edges(edges, tile_edges)
    direction_base = edge * direction_stride_edge
    hvp_base = edge * hvp_stride_edge
    values = rotation_values(
        rotation_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
    )
    derivatives = angle_derivatives(values)
    (
        x,
        y,
        z,
        r02,
        r22,
        r00,
        r20,
        alpha,
        beta,
        gamma,
        cosine_alpha,
        sine_alpha,
        gamma_y,
        gamma_x,
        regular,
    ) = values
    (
        alpha_x,
        alpha_z,
        beta_x,
        beta_y,
        beta_z,
        gamma_y_gradient,
        gamma_x_gradient,
        gamma_y_alpha,
        gamma_x_alpha,
        gamma_alpha,
    ) = derivatives

    direction_r00 = tl.load(rotation_direction_ptr + direction_base).to(tl.float32)
    direction_x = tl.load(
        rotation_direction_ptr + direction_base + direction_stride_column
    ).to(tl.float32)
    direction_r02 = tl.load(
        rotation_direction_ptr + direction_base + 2 * direction_stride_column
    ).to(tl.float32)
    direction_y = tl.load(
        rotation_direction_ptr
        + direction_base
        + direction_stride_row
        + direction_stride_column
    ).to(tl.float32)
    direction_r20 = tl.load(
        rotation_direction_ptr + direction_base + 2 * direction_stride_row
    ).to(tl.float32)
    direction_z = tl.load(
        rotation_direction_ptr
        + direction_base
        + 2 * direction_stride_row
        + direction_stride_column
    ).to(tl.float32)
    direction_r22 = tl.load(
        rotation_direction_ptr
        + direction_base
        + 2 * direction_stride_row
        + 2 * direction_stride_column
    ).to(tl.float32)

    radial_squared = tl.where(regular, x * x + z * z, 1.0)
    radial = tl.sqrt(radial_squared)
    norm_squared = radial_squared + y * y
    radial_squared_direction = 2.0 * (x * direction_x + z * direction_z)
    radial_direction = (x * direction_x + z * direction_z) / radial
    norm_squared_direction = radial_squared_direction + 2.0 * y * direction_y
    alpha_x_direction = (
        direction_z * radial_squared - z * radial_squared_direction
    ) / (radial_squared * radial_squared)
    alpha_z_direction = (
        -direction_x * radial_squared + x * radial_squared_direction
    ) / (radial_squared * radial_squared)
    beta_x_direction = (direction_y * x + y * direction_x) / (
        radial * norm_squared
    ) - beta_x * (radial_direction / radial + norm_squared_direction / norm_squared)
    beta_z_direction = (direction_y * z + y * direction_z) / (
        radial * norm_squared
    ) - beta_z * (radial_direction / radial + norm_squared_direction / norm_squared)
    beta_y_direction = (
        -radial_direction / norm_squared
        + radial * norm_squared_direction / (norm_squared * norm_squared)
    )

    alpha_direction = alpha_x * direction_x + alpha_z * direction_z
    beta_direction = beta_x * direction_x + beta_y * direction_y + beta_z * direction_z
    cosine_alpha_direction = -sine_alpha * alpha_direction
    sine_alpha_direction = cosine_alpha * alpha_direction
    gamma_y_direction = (
        cosine_alpha_direction * r02
        + cosine_alpha * direction_r02
        - sine_alpha_direction * r22
        - sine_alpha * direction_r22
    )
    gamma_x_direction = (
        cosine_alpha_direction * r00
        + cosine_alpha * direction_r00
        - sine_alpha_direction * r20
        - sine_alpha * direction_r20
    )
    gamma_denominator = tl.where(regular, gamma_y * gamma_y + gamma_x * gamma_x, 1.0)
    gamma_denominator_direction = 2.0 * (
        gamma_y * gamma_y_direction + gamma_x * gamma_x_direction
    )
    gamma_y_gradient_direction = (
        gamma_x_direction / gamma_denominator
        - gamma_y_gradient * gamma_denominator_direction / gamma_denominator
    )
    gamma_x_gradient_direction = (
        -gamma_y_direction / gamma_denominator
        - gamma_x_gradient * gamma_denominator_direction / gamma_denominator
    )
    gamma_direction = (
        gamma_y_gradient * gamma_y_direction + gamma_x_gradient * gamma_x_direction
    )
    gamma_y_alpha_direction = (
        -cosine_alpha * alpha_direction * r02
        - sine_alpha * direction_r02
        + sine_alpha * alpha_direction * r22
        - cosine_alpha * direction_r22
    )
    gamma_x_alpha_direction = (
        -cosine_alpha * alpha_direction * r00
        - sine_alpha * direction_r00
        + sine_alpha * alpha_direction * r20
        - cosine_alpha * direction_r20
    )
    gamma_alpha_direction = (
        gamma_y_gradient_direction * gamma_y_alpha
        + gamma_y_gradient * gamma_y_alpha_direction
        + gamma_x_gradient_direction * gamma_x_alpha
        + gamma_x_gradient * gamma_x_alpha_direction
    )

    alpha_gradient = tl.load(alpha_gradient_ptr + edge).to(tl.float32)
    beta_gradient = tl.load(beta_gradient_ptr + edge).to(tl.float32)
    gamma_gradient = tl.load(gamma_gradient_ptr + edge).to(tl.float32)
    combined_alpha = alpha_gradient + gamma_gradient * gamma_alpha
    combined_alpha_direction = gamma_gradient * gamma_alpha_direction
    hvp_x = (
        combined_alpha_direction * alpha_x
        + combined_alpha * alpha_x_direction
        + beta_gradient * beta_x_direction
    )
    hvp_y = beta_gradient * beta_y_direction
    hvp_z = (
        combined_alpha_direction * alpha_z
        + combined_alpha * alpha_z_direction
        + beta_gradient * beta_z_direction
    )
    hvp_r02 = gamma_gradient * (
        gamma_y_gradient_direction * cosine_alpha
        + gamma_y_gradient * cosine_alpha_direction
    )
    hvp_r22 = -gamma_gradient * (
        gamma_y_gradient_direction * sine_alpha
        + gamma_y_gradient * sine_alpha_direction
    )
    hvp_r00 = gamma_gradient * (
        gamma_x_gradient_direction * cosine_alpha
        + gamma_x_gradient * cosine_alpha_direction
    )
    hvp_r20 = -gamma_gradient * (
        gamma_x_gradient_direction * sine_alpha
        + gamma_x_gradient * sine_alpha_direction
    )

    tl.store(alpha_direction_ptr + edge, tl.where(regular, alpha_direction, 0.0), valid)
    tl.store(beta_direction_ptr + edge, tl.where(regular, beta_direction, 0.0), valid)
    tl.store(gamma_direction_ptr + edge, tl.where(regular, gamma_direction, 0.0), valid)
    for offset in tl.static_range(0, 9):
        hvp = tl.zeros((), tl.float32)
        if offset == 0:
            hvp = hvp_r00
        elif offset == 1:
            hvp = hvp_x
        elif offset == 2:
            hvp = hvp_r02
        elif offset == 4:
            hvp = hvp_y
        elif offset == 6:
            hvp = hvp_r20
        elif offset == 7:
            hvp = hvp_z
        elif offset == 8:
            hvp = hvp_r22
        tl.store(
            rotation_hvp_ptr
            + hvp_base
            + (offset // 3) * hvp_stride_row
            + (offset % 3) * hvp_stride_column,
            tl.where(regular, hvp, 0.0),
            valid,
        )


def pole_clamped_angles_hvp(
    alpha_gradient,
    beta_gradient,
    gamma_gradient,
    rotation,
    rotation_direction,
):
    edges = rotation.shape[0]
    alpha_direction = torch.empty(edges, device=rotation.device, dtype=rotation.dtype)
    beta_direction = torch.empty_like(alpha_direction)
    gamma_direction = torch.empty_like(alpha_direction)
    rotation_hvp = torch.empty_like(rotation)
    tile, warps = frame_angle_tile(rotation)
    pole_clamped_angles_hvp_kernel[(triton.cdiv(rotation.shape[0], tile),)](
        alpha_gradient,
        beta_gradient,
        gamma_gradient,
        rotation,
        rotation_direction,
        alpha_direction,
        beta_direction,
        gamma_direction,
        rotation_hvp,
        *rotation.stride(),
        *rotation_direction.stride(),
        *rotation_hvp.stride(),
        edges=rotation.shape[0],
        tile_edges=tile,
        num_warps=warps,
    )
    return alpha_direction, beta_direction, gamma_direction, rotation_hvp


@triton.jit
def monomial_hessian_direction(
    w,
    x,
    y,
    z,
    dw,
    dx,
    dy,
    dz,
    index: tl.constexpr,
    axis,
    exponents_w: tl.constexpr,
    exponents_x: tl.constexpr,
    exponents_y: tl.constexpr,
    exponents_z: tl.constexpr,
):
    values = (w, x, y, z)
    directions = (dw, dx, dy, dz)
    exponents = (
        list(exponents_w)[index],
        list(exponents_x)[index],
        list(exponents_y)[index],
        list(exponents_z)[index],
    )
    result = tl.zeros(axis.shape, tl.float32)
    for row in tl.static_range(4):
        row_exponent = exponents[row]
        local_result = 0.0
        for column in tl.static_range(4):
            column_exponent = exponents[column]
            factor = row_exponent * (column_exponent - (1 if row == column else 0))
            term = factor * directions[column]
            for local in tl.static_range(4):
                decrement = (1 if local == row else 0) + (1 if local == column else 0)
                term *= polynomial_integer_power(
                    values[local],
                    tl.maximum(0, exponents[local] - decrement),
                )
            local_result += term
        result += tl.where(axis == row, local_result, 0.0)
    return result


@triton.jit
def hvp_terms(
    coefficients_ptr,
    matrix_gradient_ptr,
    matrix_base,
    offsets: tl.constexpr,
    w,
    x,
    y,
    z,
    dw,
    dx,
    dy,
    dz,
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
    matrix_direction = tl.zeros((1, block_outputs), tl.float32)
    q_hvp = tl.zeros((4, block_outputs), tl.float32)
    direction = tl.full((4, 1), dw, tl.float32)
    direction = tl.where(axis == 1, dx, direction)
    direction = tl.where(axis == 2, dy, direction)
    direction = tl.where(axis == 3, dz, direction)
    for index in tl.static_range(monomial_count):
        coefficient = tl.load(
            coefficients_ptr + output * monomial_count + index,
            mask=mask,
            other=0.0,
        )
        gradient = monomial_gradient(
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
        directional = tl.sum(gradient * direction, axis=0)
        matrix_direction += tl.sum(coefficient, axis=0) * directional
        hessian_direction = monomial_hessian_direction(
            w,
            x,
            y,
            z,
            dw,
            dx,
            dy,
            dz,
            index,
            axis,
            exponents_w,
            exponents_x,
            exponents_y,
            exponents_z,
        )
        q_hvp += hessian_direction * coefficient * matrix_gradient
    return (
        tl.reshape(matrix_direction, (block_outputs,)),
        tl.sum(q_hvp, axis=1),
        tl.reshape(mask, (block_outputs,)),
    )


@triton.jit
def retained_hvp_edge(
    matrix_gradient_ptr,
    q_ptr,
    q_direction_ptr,
    quadratic_ptr,
    quartic_ptr,
    matrix_direction_ptr,
    q_hvp_ptr,
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
    dw = tl.load(q_direction_ptr + q_base).to(tl.float32)
    dx = tl.load(q_direction_ptr + q_base + 1).to(tl.float32)
    dy = tl.load(q_direction_ptr + q_base + 2).to(tl.float32)
    dz = tl.load(q_direction_ptr + q_base + 3).to(tl.float32)
    axis = tl.arange(0, 4)[:, None]
    matrix_base = edge * (1 + linear_count + quartic_count)
    linear_direction, q_hvp, linear_mask = hvp_terms(
        quadratic_ptr,
        matrix_gradient_ptr,
        matrix_base,
        linear_offsets,
        w,
        x,
        y,
        z,
        dw,
        dx,
        dy,
        dz,
        axis,
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
    tl.store(
        matrix_direction_ptr + matrix_base + linear_offset,
        linear_direction,
        mask=linear_mask,
    )
    quartic_direction, quartic_hvp, quartic_mask = hvp_terms(
        quartic_ptr,
        matrix_gradient_ptr,
        matrix_base,
        quartic_offsets,
        w,
        x,
        y,
        z,
        dw,
        dx,
        dy,
        dz,
        axis,
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
    tl.store(
        matrix_direction_ptr + matrix_base + quartic_offset,
        quartic_direction,
        mask=quartic_mask,
    )
    tl.store(q_hvp_ptr + q_base + tl.arange(0, 4), q_hvp + quartic_hvp)


@triton.jit
def retained_hvp_kernel(
    matrix_gradient_ptr,
    q_ptr,
    q_direction_ptr,
    quadratic_ptr,
    quartic_ptr,
    matrix_direction_ptr,
    q_hvp_ptr,
    linear_count: tl.constexpr,
    quartic_count: tl.constexpr,
    linear_offsets: tl.constexpr,
    quartic_offsets: tl.constexpr,
):
    retained_hvp_edge(
        matrix_gradient_ptr, q_ptr, q_direction_ptr, quadratic_ptr, quartic_ptr,
        matrix_direction_ptr, q_hvp_ptr, tl.program_id(0),
        linear_count, quartic_count, linear_offsets, quartic_offsets,
    )


@triton.jit
def retained_sparse_hvp_kernel(
    q, direction, seed, quadratic, quartic, coeff, codes, usable, matrix, q_out, edges,
):
    adaptive_quaternion_tile(
        q, direction, seed, quadratic, quartic, coeff, codes, usable, matrix,
        q_out, edges, retained_hvp_edge, 2,
    )


def retained_rows_hvp(matrix_gradient, q, q_direction, quadratic, quartic, mmax, compile_visible):
    edges = q.shape[0]
    q_hvp = torch.empty_like(q)
    if use_sparse_quaternion(q, quadratic, quartic, mmax, compile_visible):
        matrix_direction = torch.empty_like(matrix_gradient)
        packed = pack_sparse_quaternion(q, quadratic, quartic)
        retained_sparse_hvp_kernel[(triton.cdiv(edges, 4),)](
            q, q_direction, matrix_gradient, quadratic, quartic, *packed,
            matrix_direction, q_hvp, edges, num_warps=4,
        )
        return matrix_direction, q_hvp
    matrix_direction = torch.zeros_like(matrix_gradient)
    if compile_visible:
        torch.ops.so2.retained_quaternion_hvp(
            matrix_gradient,
            q,
            q_direction,
            quadratic,
            quartic,
            matrix_direction,
            q_hvp,
            mmax,
        )
    elif q.shape[0]:
        retained_hvp_kernel[(q.shape[0],)](
            matrix_gradient,
            q,
            q_direction,
            quadratic,
            quartic,
            matrix_direction,
            q_hvp,
            **launch_constants(mmax),
            num_warps=4,
        )
    return matrix_direction, q_hvp


@triton.jit
def wigner_hvp_kernel(
    matrix_gradient_ptr, alpha_ptr, beta_ptr, gamma_ptr,
    alpha_direction_ptr, beta_direction_ptr, gamma_direction_ptr,
    matrix_j_ptr, matrix_direction_ptr, alpha_hvp_ptr, beta_hvp_ptr, gamma_hvp_ptr,
    degree: tl.constexpr, selected_rows: tl.constexpr,
    row_count: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_width: tl.constexpr,
    inverse_gradient_ptr=None, inverse_direction_ptr=None, inverse_rescale_ptr=None,
    matrix_stride: tl.constexpr = 0,
    matrix_offset: tl.constexpr = 0,
):
    edge = tl.program_id(0)
    row = tl.arange(0, block_rows)[:, None]
    column = tl.arange(0, block_width)[None, :]
    mask = (row < row_count) & (column < width)
    alpha_direction = tl.load(alpha_direction_ptr + edge).to(tl.float32)
    beta_direction = tl.load(beta_direction_ptr + edge).to(tl.float32)
    gamma_direction = tl.load(gamma_direction_ptr + edge).to(tl.float32)
    _, da, db, dg, daa, dab, dag, dbb, dbg, dgg = wigner_terms(
        tl.load(alpha_ptr + edge).to(tl.float32),
        tl.load(beta_ptr + edge).to(tl.float32),
        tl.load(gamma_ptr + edge).to(tl.float32),
        matrix_j_ptr, degree, selected_rows,
        row_count, width, block_rows, block_width,
    )
    address = matrix_address(
        edge, row, column, row_count, width, matrix_stride, matrix_offset
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
    matrix_direction = da * alpha_direction + db * beta_direction + dg * gamma_direction
    tl.store(matrix_direction_ptr + address, matrix_direction, mask=mask)
    if inverse_direction_ptr is not None:
        tl.store(inverse_direction_ptr + address, matrix_direction * rescale, mask=mask)
    alpha_curvature = daa * alpha_direction + dab * beta_direction + dag * gamma_direction
    beta_curvature = dab * alpha_direction + dbb * beta_direction + dbg * gamma_direction
    gamma_curvature = dag * alpha_direction + dbg * beta_direction + dgg * gamma_direction
    tl.atomic_add(alpha_hvp_ptr + edge, reduce_angles(gradient * alpha_curvature))
    tl.atomic_add(beta_hvp_ptr + edge, reduce_angles(gradient * beta_curvature))
    tl.atomic_add(gamma_hvp_ptr + edge, reduce_angles(gradient * gamma_curvature))


def wigner_hvp(matrix_gradients, alpha, beta, gamma, directions, matrices_j, rows_by_degree):
    allow_atomic_reduction("rotation/euler/wigner.hvp")
    angle_hvp = tuple(torch.zeros_like(angle) for angle in (alpha, beta, gamma))
    matrix_directions = []
    edges = alpha.shape[0]
    for degree, metadata in enumerate(rotation_metadata(rows_by_degree)):
        index = degree
        matrix_directions.append(torch.empty_like(matrix_gradients[index]))
        wigner_hvp_kernel[(edges,)](
            matrix_gradients[index],
            alpha, beta, gamma,
            *directions,
            matrices_j[degree],
            matrix_directions[index],
            *angle_hvp,
            **metadata,
        )
    return tuple(matrix_directions), angle_hvp


@triton.jit
def wigner_generator_terms(
    alpha, beta, gamma, va, vb, vg, j,
    degree: tl.constexpr, rows: tl.constexpr, row_count: tl.constexpr,
    width: tl.constexpr, br: tl.constexpr, bc: tl.constexpr,
):
    row = tl.arange(0, br)[:, None]
    col = tl.arange(0, bc)[None, :]
    # Put each pair of signed orders in adjacent rows for the lane exchange.
    # Compact storage still uses m=0 followed by the signed order pairs.
    stored_row = tl.where(row + 1 < row_count, row + 1, 0)
    selected = selected_row_indices(stored_row, rows, row_count)
    fa = degree - selected
    fa_cos = tl.cos(alpha * fa)
    fa_sin = tl.sin(alpha * fa)
    selected_flip = width - 1 - selected
    c = tl.zeros((br, bc), tl.float32)
    cb = tl.zeros((br, bc), tl.float32)
    cbb = tl.zeros((br, bc), tl.float32)

    for inner in tl.static_range(width):
        inner_flip = width - 1 - inner
        a = fa_cos * tl.load(j + selected * width + inner, row < row_count, 0)
        a += fa_sin * tl.load(j + selected_flip * width + inner, row < row_count, 0)
        af = fa_cos * tl.load(j + selected * width + inner_flip, row < row_count, 0)
        af += fa_sin * tl.load(j + selected_flip * width + inner_flip, row < row_count, 0)
        fb = degree - inner
        fb_cos = tl.cos(beta * fb)
        fb_sin = tl.sin(beta * fb)
        b = a * fb_cos - af * fb_sin
        db = fb * (-a * fb_sin - af * fb_cos)
        ddb = -(fb * fb) * b
        coeff = tl.load(j + inner * width + col, col < width, 0)
        c += b * coeff
        cb += db * coeff
        cbb += ddb * coeff

    mirror_col = tl.broadcast_to(tl.where(col < width, width - 1 - col, col), c.shape)
    mirror_row = tl.broadcast_to(tl.where(row < row_count - 1, row ^ 1, row), c.shape)
    cf = tl.gather(c, mirror_col, axis=1)
    cbf = tl.gather(cb, mirror_col, axis=1)
    cbbf = tl.gather(cbb, mirror_col, axis=1)
    fg = degree - col
    g_cos = tl.cos(gamma * fg)
    g_sin = tl.sin(gamma * fg)
    d = c * g_cos - cf * g_sin
    db = cb * g_cos - cbf * g_sin
    dbb = cbb * g_cos - cbbf * g_sin

    # Z'(a) = A Z(a), A_(r,opposite(r)) = degree-r.
    # Left and right multiplication by A have opposite frequency signs.
    adot = fa * va
    gdot = -fg * vg
    jvp = adot * tl.gather(d, mirror_row, axis=0) + vb * db + gdot * tl.gather(d, mirror_col, axis=1)
    ha = fa * tl.gather(jvp, mirror_row, axis=0)
    hb = adot * tl.gather(db, mirror_row, axis=0) + vb * dbb + gdot * tl.gather(db, mirror_col, axis=1)
    hg = -fg * tl.gather(jvp, mirror_col, axis=1)
    return jvp, ha, hb, hg


@triton.jit
def retained_generator_hvp_kernel(
    angles, directions, basis, scale, gradients, outputs, angle_hvp,
    degree: tl.constexpr, rows: tl.constexpr, row_count: tl.constexpr,
    width: tl.constexpr, br: tl.constexpr, bc: tl.constexpr,
    compact: tl.constexpr, offset: tl.constexpr,
    edge=None,
):
    if edge is None:
        edge = tl.program_id(0)
    alpha, beta, gamma = (tl.load(angles[0] + edge), tl.load(angles[1] + edge),
                          tl.load(angles[2] + edge))
    va, vb, vg = (tl.load(directions[0] + edge), tl.load(directions[1] + edge),
                  tl.load(directions[2] + edge))
    jvp, ha, hb, hg = wigner_generator_terms(
        alpha, beta, gamma, va, vb, vg, basis, degree, rows,
        row_count, width, br, bc,
    )
    row = tl.arange(0, br)[:, None]
    col = tl.arange(0, bc)[None, :]
    stored_row = tl.where(row + 1 < row_count, row + 1, 0)
    mask = (row < row_count) & (col < width)
    local = offset + stored_row * width + col
    address = edge * compact + local
    gradient = tl.load(gradients[0] + address, mask, 0)
    tl.store(outputs[0] + address, jvp, mask)
    if scale is not None:
        factor = tl.load(scale + local, mask, 0)
        gradient += tl.load(gradients[1] + address, mask, 0) * factor
        tl.store(outputs[1] + address, jvp * factor, mask)
    for axis in tl.static_range(3):
        update = tl.sum(tl.sum(gradient * (ha, hb, hg)[axis], axis=1), axis=0)
        if degree > 0:
            update += tl.load(angle_hvp[axis] + edge)
        tl.store(angle_hvp[axis] + edge, update)


def use_retained_generator(alpha, spec):
    return (
        (spec.lmax, spec.mmax) in ((4, 2), (4, 4), (6, 2))
        and alpha.shape[0] >= 16384
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(alpha.device) in ((8, 0), (9, 0))
    )


@triton.jit
def compact_columns(slot):
    tile = slot // 8
    index = slot % 8
    degree = tl.where(tile < 1, 1, tl.where(tile < 4, 2, tl.where(tile < 9, 3, 4)))
    begin = tl.where(degree == 1, 0, tl.where(degree == 2, 1, tl.where(degree == 3, 4, 9)))
    local = tile - begin
    full = tl.where(degree == 1, index < 4, local < degree)
    full_group = 2 * local + index // 4
    axis_group = 4 * (local - degree) + index // 2
    row = tl.where(full, 1 + 2 * (full_group // degree) + (index // 2 % 2),
                   tl.where(axis_group < 2, 1 + 2 * axis_group + index % 2, 0))
    order = tl.where(full, full_group % degree + 1, tl.where(axis_group < 2, 0, axis_group - 1))
    col = degree + tl.where(index % 2 == 0, -order, order)

    # Degree one has one complete 2x2 group, then a reflected row and column.
    row1 = tl.where(index < 4, 1 + index // 2,
                    tl.where(index < 6, 1 + index % 2, 0))
    col1 = tl.where((index >= 4) & (index < 6), 1, 2 * (index % 2))
    row = tl.where(degree == 1, row1, row)
    col = tl.where(degree == 1, col1, col)
    valid = full | (axis_group < degree + 2) | (degree == 1)
    valid = tl.where(tile < 15, valid, slot < 125)
    row = tl.where(tile < 15, row, 0)
    degree = tl.where(tile < 15, degree, tl.where(slot < 124, slot - 119, 0))
    col = tl.where(tile < 15, col, degree)

    row_flip = tl.where(full, 2, tl.where((row > 0) & (col == degree), 1, 0))
    col_flip = tl.where(col != degree, 1, 0)
    row_flip = tl.where(tile < 15, row_flip, 0)
    offsets = tl.where(degree == 4, 70, tl.where(degree == 3, 35,
                      tl.where(degree == 2, 10, tl.where(degree == 1, 1, 0))))
    width = 2 * degree + 1
    column = offsets + row * width + col
    row_index = degree + tl.where(row % 2 == 0, -(row // 2), (row + 1) // 2)
    return degree, width, row_index, col, column, row_flip, col_flip, valid


@triton.jit
def store_nonzero(weight, harmonic, count, values, orders, lane, valid):
    keep = (weight != 0.) & valid
    index = count * 128 + lane
    tl.store(values + index, weight, keep)
    tl.store(orders + index, harmonic, keep)
    return count + keep.to(tl.int32)


@triton.jit
def stable_basis_value(value):
    # Products and their exact residuals remain normal FP32 in this interval.
    magnitude = tl.abs(value)
    return (magnitude <= 2.) & ((magnitude == 0.) | (magnitude >= 2.3283064365386963e-10))


@triton.jit
def rounded_fourier_term(a, b, c, d):
    first, second = a * b, c * d
    combined = first + second
    # If separately rounded products cancel, recover the low product bits.
    residual = tl.fma(a, b, -first) + tl.fma(c, d, -second)
    return tl.where(combined == 0., residual, combined)


@triton.jit
def pack_basis_kernel(j0, j1, j2, j3, j4, metadata, harmonics, values, maximum):
    slot = tl.arange(0, 128)
    degree, width, row, col, column, rf, cf, valid = compact_columns(slot)
    j = tl.where(degree == 4, j4, tl.where(degree == 3, j3,
                 tl.where(degree == 2, j2, tl.where(degree == 1, j1, j0))))
    rp = tl.load(j + row * width + degree, valid, 0.)
    cp = tl.load(j + degree * width + col, valid, 0.)
    count = tl.full((128,), 0, tl.int32)
    basis_ok = stable_basis_value(rp) & stable_basis_value(cp)
    count = store_nonzero(rp * cp, 0, count, values, harmonics, slot, valid)
    for order in tl.static_range(1, 5):
        use_order = valid & (order <= degree)
        rp = tl.load(j + row * width + degree - order, use_order, 0.)
        rn = tl.load(j + row * width + degree + order, use_order, 0.)
        cp = tl.load(j + (degree - order) * width + col, use_order, 0.)
        cn = tl.load(j + (degree + order) * width + col, use_order, 0.)
        basis_ok &= (stable_basis_value(rp) & stable_basis_value(rn)
                     & stable_basis_value(cp) & stable_basis_value(cn))
        sine = rounded_fourier_term(rp, cn, -rn, cp)
        cosine = rounded_fourier_term(rp, cp, rn, cn)
        count = store_nonzero(sine, 2 * order - 1, count, values, harmonics, slot, use_order)
        count = store_nonzero(cosine, 2 * order, count, values, harmonics, slot, use_order)
    meta = column | (rf << 7) | (cf << 9) | ((degree - row + 4) << 11)
    meta |= ((degree - col + 4) << 15) | (count << 19)
    tl.store(metadata + slot, tl.where(valid, meta, 127))
    terms = tl.maximum(tl.max(count, 0), 1)
    tl.store(maximum, tl.where(tl.sum((~basis_ok).to(tl.int32), 0) == 0, terms, -1))


if (
    torch.version.cuda
    and find_spec("triton.experimental") is not None
    and find_spec("triton.experimental.gluon") is not None
):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def shuffle_xor(x, distance):
        return gl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
            constraints="=f,f,r", args=(x, distance), dtype=gl.float32, is_pure=True, pack=1,
        )


    @gluon.jit
    def shuffle_index(x, lane):
        return gl.inline_asm_elementwise(
            "shfl.sync.idx.b32 $0, $1, $2, 31, -1;",
            constraints="=f,f,r", args=(x, lane), dtype=gl.float32, is_pure=True, pack=1,
        )


    @gluon.jit
    def rotate(value, sa, ca, sg, cg, row_xor, col_xor):
        left = gl.fma(sa, shuffle_xor(value, row_xor), ca * value)
        return gl.fma(-sg, shuffle_xor(left, col_xor), cg * left)


    @gluon.jit
    def bounded_sincos(phase):
        # Split 2*pi into a leading float and a residual for Euler harmonics.
        turns = gl.inline_asm_elementwise(
            "cvt.rni.f32.f32 $0, $1;", constraints="=f,f",
            args=(phase * 0.15915494309189535,), dtype=gl.float32, is_pure=True, pack=1,
        )
        angle = gl.fma(-turns, 6.2831854820251465, phase)
        angle = gl.fma(turns, 1.748455600074497e-7, angle)
        return gl.inline_asm_elementwise(
            "{ sin.approx.ftz.f32 $0, $2; cos.approx.ftz.f32 $1, $2; }",
            constraints="=f,=f,f", args=(angle,),
            dtype=(gl.float32, gl.float32), is_pure=True, pack=1,
        )


    @gluon.jit
    def sparse_hvp_kernel(
        alpha, beta, gamma, alpha_dir, beta_dir, gamma_dir,
        direct_seed, inverse_seed, scale, metadata, harmonics, values, maximum,
        direct_out, inverse_out, angle_out, exceptional, edges,
        compact: gl.constexpr, slots: gl.constexpr, max_order: gl.constexpr,
        column_bits: gl.constexpr, fixed_terms: gl.constexpr,
        edge_tile: gl.constexpr, coefficient_warps: gl.constexpr,
    ):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [edge_tile, coefficient_warps], [1, 0])
        edge = gl.program_id(0) * edge_tile + gl.arange(0, edge_tile, layout=gl.SliceLayout(1, layout))
        valid_edge = edge < edges
        terms = gl.load(maximum) if fixed_terms == 0 else fixed_terms
        if terms < 0:
            gl.store(exceptional + edge, valid_edge, valid_edge)
        else:
            slot = gl.arange(0, slots, layout=gl.SliceLayout(0, layout))
            meta = gl.load(metadata + slot)
            column = meta & ((1 << column_bits) - 1)
            valid_col = column < compact
            zero_edge = gl.full((edge_tile,), 0., gl.float32, layout=gl.SliceLayout(1, layout))
            zero_col = gl.full((slots,), 0., gl.float32, layout=gl.SliceLayout(0, layout))
            zero = gl.full((edge_tile, slots), 0., gl.float32, layout=layout)

            a = gl.load(alpha + edge, valid_edge, zero_edge)[:, None]
            b = gl.load(beta + edge, valid_edge, zero_edge)[:, None]
            g = gl.load(gamma + edge, valid_edge, zero_edge)[:, None]
            va = gl.load(alpha_dir + edge, valid_edge, zero_edge)[:, None]
            vb = gl.load(beta_dir + edge, valid_edge, zero_edge)[:, None]
            vg = gl.load(gamma_dir + edge, valid_edge, zero_edge)[:, None]
            af = ((meta >> (column_bits + 4) & 15) - max_order)[None, :]
            gf = ((meta >> (column_bits + 8) & 15) - max_order)[None, :]
            # Pad lanes still participate in every shuffle and the final block reduction.
            row_xor = (meta >> column_bits & 3)[None, :] + gl.full((edge_tile, 1), 0, gl.int32, layout=layout)
            col_xor = (meta >> (column_bits + 2) & 3)[None, :] + gl.full((edge_tile, 1), 0, gl.int32, layout=layout)
            lane = slot[None, :] % 32
            phase = gl.where(lane < max_order + 1, a, gl.where(lane < 2 * (max_order + 1), b, g)) * (lane % (max_order + 1))
            bounded = ((gl.abs(a) <= 3.141592741) & (gl.abs(b) <= 3.141592741)
                       & (gl.abs(g) <= 3.141592741) & (gl.abs(va) <= 4294967296.)
                       & (gl.abs(vb) <= 4294967296.) & (gl.abs(vg) <= 4294967296.))
            ss, cc = bounded_sincos(phase)
            ai = gl.abs(af) + gl.full((edge_tile, 1), 0, gl.int32, layout=layout)
            gi = gl.abs(gf) + 2 * (max_order + 1) + gl.full((edge_tile, 1), 0, gl.int32, layout=layout)
            sa, ca = shuffle_index(ss, ai), shuffle_index(cc, ai)
            sg, cg = shuffle_index(ss, gi), shuffle_index(cc, gi)
            sa = gl.where(af < 0, -sa, sa)
            sg = gl.where(gf < 0, -sg, sg)

            k, kb, kbb = zero, zero, zero
            for term in range(terms):
                offset = term * slots + slot
                if fixed_terms == 0:
                    count = meta >> (column_bits + 12)
                    present = term < count
                    zero_order = gl.full((slots,), 0, gl.int32, layout=gl.SliceLayout(0, layout))
                    harmonic = gl.load(harmonics + offset, present, zero_order)[None, :]
                    weight = gl.load(values + offset, present, zero_col)[None, :]
                else:
                    harmonic = gl.load(harmonics + offset)[None, :]
                    weight = gl.load(values + offset)[None, :]
                order = (harmonic + 1) >> 1
                idx = order + max_order + 1 + gl.full((edge_tile, 1), 0, gl.int32, layout=layout)
                sin, cos = shuffle_index(ss, idx), shuffle_index(cc, idx)
                odd = harmonic & 1 != 0
                h = gl.where(odd, sin, cos)
                h1 = gl.where(odd, cos, -sin) * order
                k = gl.fma(weight, h, k)
                kb = gl.fma(weight, h1, kb)
                kbb = gl.fma(-weight, h * order * order, kbb)

            d = rotate(k, sa, ca, sg, cg, row_xor, col_xor)
            db = rotate(kb, sa, ca, sg, cg, row_xor, col_xor)
            dbb = rotate(kbb, sa, ca, sg, cg, row_xor, col_xor)
            jvp = gl.fma(va * af, shuffle_xor(d, row_xor), vb * db)
            jvp = gl.fma(-vg * gf, shuffle_xor(d, col_xor), jvp)
            hb = gl.fma(va * af, shuffle_xor(db, row_xor), vb * dbb)
            hb = gl.fma(-vg * gf, shuffle_xor(db, col_xor), hb)
            ha = af * shuffle_xor(jvp, row_xor)
            hg = -gf * shuffle_xor(jvp, col_xor)
            address = edge[:, None] * compact + column[None, :]
            mask = valid_edge[:, None] & valid_col[None, :]
            seed = gl.load(direct_seed + address, mask, zero)
            seed_ok = gl.abs(seed) <= 4294967296.
            if inverse_seed is not None:
                factor = gl.load(scale + column, valid_col, zero_col)[None, :]
                inverse = gl.load(inverse_seed + address, mask, zero)
                seed_ok &= (gl.abs(inverse) <= 4294967296.) & (gl.abs(factor) <= 65536.)
                seed = gl.fma(inverse, factor, seed)
            # The fallback runs on the same stream and overwrites every flagged edge.
            gl.store(direct_out + address, jvp, mask)
            if inverse_out is not None:
                gl.store(inverse_out + address, jvp * factor, mask)
            gl.store(angle_out + edge, gl.sum(gl.where(valid_col[None, :], ha * seed, 0.), 1), valid_edge)
            gl.store(angle_out + edges + edge, gl.sum(gl.where(valid_col[None, :], hb * seed, 0.), 1), valid_edge)
            gl.store(angle_out + 2 * edges + edge, gl.sum(gl.where(valid_col[None, :], hg * seed, 0.), 1), valid_edge)
            bad_seed = gl.max((valid_col[None, :] & ~seed_ok).to(gl.int32), 1)
            good = (gl.sum(bounded.to(gl.int32), 1) == 1) & (bad_seed == 0)
            gl.store(exceptional + edge, ~good, valid_edge)

else:
    gluon = gl = None


@triton.jit
def grouped_paired_exception_hvp_kernel(
    angles, directions, basis, scale, seeds, outputs, hessian, exceptional, edges,
    metadata: tl.constexpr, compact: tl.constexpr, group: tl.constexpr,
):
    start = tl.program_id(0) * group
    lanes = tl.arange(0, group)
    flagged = tl.load(exceptional + start + lanes, start + lanes < edges, False)
    one = tl.full((group,), 1, tl.uint32)
    bits = tl.sum(tl.where(flagged, one << lanes, 0), axis=0).to(tl.uint32)
    while bits != 0:
        index = tl.inline_asm_elementwise(
            "bfind.u32 $0, $1;", constraints="=r,r", args=(bits,), dtype=tl.int32,
            is_pure=True, pack=1,
        )
        edge = start + index
        bits ^= tl.full((), 1, tl.uint32) << index
        for degree in tl.static_range(len(metadata) // 5):
            retained_generator_hvp_kernel(
                angles, directions, basis[degree], scale, seeds, outputs, hessian,
                degree, retained_tile_rows(degree, metadata[degree * 5]),
                metadata[degree * 5], metadata[degree * 5 + 1],
                metadata[degree * 5 + 2], metadata[degree * 5 + 3],
                compact, metadata[degree * 5 + 4],
                edge=edge,
            )


def sparse_hvp_from_terms(angles, directions, basis, scale, direct, inverse, spec, terms, maximum):
    angles, directions, basis = tuple(angles), tuple(directions), tuple(basis)
    edges = angles[0].shape[0]
    direct_out = torch.empty_like(direct)
    inverse_out = None if inverse is None else torch.empty_like(inverse)
    angular = direct.new_empty((3, edges))
    exceptional = direct.new_empty((edges,), dtype=torch.bool)
    if maximum is None:
        # Keep a full edge in one warp on A100. H100 benefits from two warps.
        edge_tile = 2 if torch.cuda.get_device_capability(direct.device) == (9, 0) else 4
        slots, bits, fixed = 256, 8, 4
    else:
        edge_tile = 4
        slots, bits, fixed = 128, 7, 0
    sparse_hvp_kernel[(triton.cdiv(edges, edge_tile),)](
        *angles, *directions, direct, inverse, scale, *terms, maximum,
        direct_out, inverse_out, angular, exceptional, edges, spec.compact_elements,
        slots, spec.lmax, bits, fixed, edge_tile, 4 // edge_tile, num_warps=4,
    )
    hessian = angular.unbind(0)
    seeds, outputs = ((direct,), (direct_out,)) if inverse is None else ((direct, inverse), (direct_out, inverse_out))
    group = 16
    grouped_paired_exception_hvp_kernel[(triton.cdiv(edges, group),)](
        angles, directions, basis, scale, seeds, outputs, hessian, exceptional, edges,
        retained_tile_metadata(spec), spec.compact_elements, group, num_warps=4,
    )
    return (direct_out, inverse_out), hessian


def paired_sparse_hvp(angles, directions, basis, scale, direct, inverse, spec):
    metadata = direct.new_empty((128,), dtype=torch.int32)
    harmonics = direct.new_empty((9, 128), dtype=torch.int32)
    weights = direct.new_empty((9, 128))
    maximum = direct.new_empty((), dtype=torch.int32)
    pack_basis_kernel[(1,)](
        *basis, metadata, harmonics, weights, maximum, num_warps=4,
        enable_fp_fusion=False,
    )
    return sparse_hvp_from_terms(
        angles, directions, basis, scale, direct, inverse, spec, (metadata, harmonics, weights), maximum,
    )


def frame_paired_retained_hvp(
    direct, inverse, alpha, beta, gamma, directions, basis, scale, spec,
):
    angles = (alpha, beta, gamma)
    if (
        gluon is not None
        and use_retained_generator(alpha, spec)
        and all(value.dtype == torch.float32 and value.device == alpha.device
                and value.is_contiguous()
                for value in (*angles, *directions, *basis, scale, direct, inverse))
    ):
        allow_atomic_reduction("rotation/euler/wigner.retained_hvp")
        return paired_sparse_hvp(angles, directions, basis, scale, direct, inverse, spec)
    return retained_hvp(direct, inverse, alpha, beta, gamma, directions, basis, scale, spec)


def default_retained_hvp(
    direct, inverse, alpha, beta, gamma, directions, basis, scale, spec,
):
    if gluon is None or not use_retained_generator(alpha, spec):
        return retained_hvp(direct, inverse, alpha, beta, gamma, directions, basis, scale, spec)
    allow_atomic_reduction("rotation/euler/wigner.retained_hvp")
    angles = (alpha, beta, gamma)
    return sparse_hvp_from_terms(
        angles, directions, basis, scale, direct, inverse, spec,
        canonical_l6_fourier(alpha.device), None,
    )


def retained_hvp(
    direct_gradient, inverse_gradient, alpha, beta, gamma,
    directions,
    matrices_j, inverse_rescale, spec,
):
    allow_atomic_reduction("rotation/euler/wigner.retained_hvp")
    angles = (alpha, beta, gamma)
    direct_direction = torch.empty_like(direct_gradient)
    inverse_direction = (
        torch.empty_like(inverse_gradient) if inverse_gradient is not None else None
    )
    if use_retained_generator(alpha, spec):
        angle_hvp = tuple(torch.empty_like(angle) for angle in angles)
        gradients = (direct_gradient,)
        outputs = (direct_direction,)
        if inverse_gradient is not None:
            gradients += (inverse_gradient,)
            outputs += (inverse_direction,)
        for degree, metadata in enumerate(retained_metadata(spec)):
            retained_generator_hvp_kernel[(alpha.shape[0],)](
                angles, directions, matrices_j[degree], inverse_rescale, gradients,
                outputs, angle_hvp, degree, metadata["selected_rows"],
                metadata["row_count"], metadata["width"], metadata["block_rows"],
                metadata["block_width"], spec.compact_elements, metadata["matrix_offset"],
                num_warps=4,
            )
        return (direct_direction, inverse_direction), angle_hvp
    angle_hvp = tuple(torch.zeros_like(angle) for angle in angles)
    for degree, metadata in enumerate(retained_metadata(spec)):
        wigner_hvp_kernel[(alpha.shape[0],)](
            direct_gradient,
            alpha, beta, gamma,
            *directions,
            matrices_j[degree],
            direct_direction,
            *angle_hvp,
            inverse_gradient_ptr=inverse_gradient,
            inverse_direction_ptr=inverse_direction,
            inverse_rescale_ptr=inverse_rescale,
            **metadata,
        )
    return (direct_direction, inverse_direction), angle_hvp
