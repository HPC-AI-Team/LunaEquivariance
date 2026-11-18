"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import pi
from operator import index
from threading import Lock
from typing import NamedTuple
from weakref import ref
import math
from math import comb, sqrt

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from ...utils.layout import get_angular_layout, get_retained_layout


@lru_cache(maxsize=128)
def degree_basis(degree):
    from e3nn import o3
    from e3nn.o3._wigner import change_basis_real_to_complex

    # Explicit dtypes keep setup independent of the caller's default dtype.
    # The real-harmonic transformation is the same one used by e3nn's Wigner D.
    with torch.inference_mode(False), torch.no_grad(), torch.device("cpu"):
        m = torch.arange(-degree, degree, dtype=torch.float64, device="cpu")
        raising = torch.diag(-(degree * (degree + 1) - m * (m + 1)).sqrt(), diagonal=-1)
        m = torch.arange(-degree + 1, degree + 1, dtype=torch.float64, device="cpu")
        lowering = torch.diag((degree * (degree + 1) - m * (m - 1)).sqrt(), diagonal=1)
        m = torch.arange(-degree, degree + 1, dtype=torch.float64, device="cpu")
        generators = torch.stack((
            0.5 * (raising + lowering),
            torch.diag(1j * m),
            -0.5j * (raising - lowering),
        ))
        basis = change_basis_real_to_complex(degree, dtype=torch.float64, device="cpu")
        generators = (basis.conj().T @ generators @ basis).real

        # P is an improper involution. -P is a proper rotation, so its
        # representation acquires the natural spherical-harmonic parity.
        proper = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]],
                              dtype=torch.float64, device="cpu")
        alpha, beta, gamma = (angle.remainder(2 * pi) for angle in o3.matrix_to_angles(proper))
        result = (
            torch.matrix_exp(alpha * generators[1])
            @ torch.matrix_exp(beta * generators[0])
            @ torch.matrix_exp(gamma * generators[1])
        )
        return ((-1) ** degree * result).contiguous()


@lru_cache(maxsize=256)
def device_degree_basis(degree, device, dtype):
    with torch.inference_mode(False), torch.no_grad():
        return degree_basis(degree).to(device=device, dtype=dtype).contiguous()


def standard_basis(lmax, device=None, dtype=None):
    """Return fixed J matrices for degrees zero through ``lmax``.

    The constants use e3nn's real spherical-harmonic basis and natural parity.
    They satisfy ``D(alpha,beta,gamma) = Z(alpha) J Z(beta) J Z(gamma)``.
    Build them in FP64 on CPU, then cache the requested FP32 or FP64 device
    copies. Returned matrices are shared constants and must not be modified.

    This setup needs only the core e3nn dependency. It does not load model
    packages, checkpoints, or stored basis files, and never changes PyTorch's
    default dtype. No geometry-dependent tensor is cached.
    """
    if isinstance(lmax, bool):
        raise TypeError("lmax must be an integer degree")
    lmax = index(lmax)
    if lmax < 0:
        raise ValueError("lmax must be nonnegative")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("standard Wigner basis requires float32 or float64")
    device = torch.device("cpu" if device is None else device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return tuple(device_degree_basis(degree, device, dtype) for degree in range(lmax + 1))


YTOL = 0.999999


def branchless_edge_rotation(
    edge_vec: torch.Tensor, frame_random: torch.Tensor
) -> torch.Tensor:
    """Construct ``(E,3,3)`` edge frames from vectors and external frame seeds.

    ``edge_vec`` and ``frame_random`` are floating ``(E,3)`` tensors on one
    device. Edge vectors must be nonzero. The second output row follows the
    normalized edge. ``frame_random`` contains the
    caller's uniform [0,1] seeds used to choose transverse directions.
    The output uses eSEN's frame convention, including its pole clipping.
    This function is ordinary PyTorch and exposes its full computation here.
    The caller keeps frame seeds fixed while differentiating the geometry.
    """
    distance = torch.linalg.vector_norm(edge_vec, dim=1)
    norm_x = edge_vec / distance[:, None]

    yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
    positive_y = norm_x.new_tensor([0.0, 1.0, 0.0]).expand_as(norm_x)
    negative_y = norm_x.new_tensor([0.0, -1.0, 0.0]).expand_as(norm_x)
    norm_x = torch.where((yprod > YTOL)[:, None], positive_y, norm_x)
    norm_x = torch.where((yprod < -YTOL)[:, None], negative_y, norm_x)

    candidate = frame_random - 0.5
    candidate = candidate / torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
    candidate_b = torch.stack(
        (-candidate[:, 1], candidate[:, 0], candidate[:, 2]), dim=1
    )
    candidate_c = torch.stack(
        (candidate[:, 0], -candidate[:, 2], candidate[:, 1]), dim=1
    )
    dot = torch.abs(torch.sum(candidate * norm_x, dim=1, keepdim=True))
    dot_b = torch.abs(torch.sum(candidate_b * norm_x, dim=1, keepdim=True))
    candidate = torch.where(dot > dot_b, candidate_b, candidate)
    dot = torch.abs(torch.sum(candidate * norm_x, dim=1, keepdim=True))
    dot_c = torch.abs(torch.sum(candidate_c * norm_x, dim=1, keepdim=True))
    candidate = torch.where(dot > dot_c, candidate_c, candidate)

    norm_z = torch.cross(norm_x, candidate, dim=1)
    norm_z = norm_z / torch.linalg.vector_norm(norm_z, dim=1, keepdim=True)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.linalg.vector_norm(norm_y, dim=1, keepdim=True)
    edge_rot_mat_inv = torch.stack((norm_z, norm_x, -norm_y), dim=2)
    return edge_rot_mat_inv.transpose(1, 2)


FLOAT_EPS = tl.constexpr(1.1920928955078125e-7)


NORMALIZE_EPS = tl.constexpr(1.0e-12)


@triton.jit
def normalize3_values(x, y, z, epsilon: tl.constexpr):
    norm_square = x * x + y * y + z * z
    inverse = 1.0 / tl.sqrt(tl.maximum(norm_square, epsilon * epsilon))
    return x * inverse, y * inverse, z * inverse


@triton.jit
def normalize4_values(w, x, y, z, epsilon: tl.constexpr):
    norm_square = w * w + x * x + y * y + z * z
    inverse = 1.0 / tl.sqrt(tl.maximum(norm_square, epsilon * epsilon))
    return w * inverse, x * inverse, y * inverse, z * inverse


@triton.jit
def edge_quaternion_values(ex, ey, ez, gamma):
    ex, ey, ez = normalize3_values(ex, ey, ez, NORMALIZE_EPS)
    chart1 = normalize4_values(1.0 + ey, -ez, 0.0, ex, tl.sqrt(FLOAT_EPS))
    chart2 = normalize4_values(-ez, 1.0 - ey, ex, 0.0, tl.sqrt(FLOAT_EPS))
    t = tl.maximum(0.0, tl.minimum(1.0, (ey + 0.9) / 1.8))
    numerator = 2.0 * t - 1.0
    denominator = tl.maximum(t * (1.0 - t), FLOAT_EPS)
    blend = tl.sigmoid(numerator / denominator)
    blend = tl.where(t < FLOAT_EPS, 0.0, blend)
    blend = tl.where(t > 1.0 - FLOAT_EPS, 1.0, blend)
    dot = (
        chart2[0] * chart1[0]
        + chart2[1] * chart1[1]
        + chart2[2] * chart1[2]
        + chart2[3] * chart1[3]
    )
    chart2 = (
        tl.where(dot < 0.0, -chart2[0], chart2[0]),
        tl.where(dot < 0.0, -chart2[1], chart2[1]),
        tl.where(dot < 0.0, -chart2[2], chart2[2]),
        tl.where(dot < 0.0, -chart2[3], chart2[3]),
    )
    edge_q = normalize4_values(
        (1.0 - blend) * chart2[0] + blend * chart1[0],
        (1.0 - blend) * chart2[1] + blend * chart1[1],
        (1.0 - blend) * chart2[2] + blend * chart1[2],
        (1.0 - blend) * chart2[3] + blend * chart1[3],
        NORMALIZE_EPS,
    )
    cosine = tl.cos(0.5 * gamma)
    sine = tl.sin(0.5 * gamma)
    w, x, y, z = edge_q
    return (
        cosine * w - sine * y,
        cosine * x + sine * z,
        cosine * y + sine * w,
        cosine * z - sine * x,
    )


@triton.jit
def normalize3_jets(x, y, z, epsilon, axis, second_order: tl.constexpr):
    norm_square = jet_add(
        jet_add(
            jet_multiply(x, x, second_order),
            jet_multiply(y, y, second_order),
        ),
        jet_multiply(z, z, second_order),
    )
    lower = jet_constant(epsilon * epsilon, axis)
    norm_square = jet_where(norm_square[0] >= lower[0], norm_square, lower)
    inverse = jet_reciprocal(jet_sqrt(norm_square, second_order), second_order)
    return (
        jet_multiply(x, inverse, second_order),
        jet_multiply(y, inverse, second_order),
        jet_multiply(z, inverse, second_order),
    )


@triton.jit
def normalize4_jets(w, x, y, z, epsilon, axis, second_order: tl.constexpr):
    norm_square = jet_add(
        jet_add(
            jet_multiply(w, w, second_order),
            jet_multiply(x, x, second_order),
        ),
        jet_add(
            jet_multiply(y, y, second_order),
            jet_multiply(z, z, second_order),
        ),
    )
    lower = jet_constant(epsilon * epsilon, axis)
    norm_square = jet_where(norm_square[0] >= lower[0], norm_square, lower)
    inverse = jet_reciprocal(jet_sqrt(norm_square, second_order), second_order)
    return (
        jet_multiply(w, inverse, second_order),
        jet_multiply(x, inverse, second_order),
        jet_multiply(y, inverse, second_order),
        jet_multiply(z, inverse, second_order),
    )


@triton.jit
def edge_quaternion_jets(
    ex,
    ey,
    ez,
    gamma,
    dex,
    dey,
    dez,
    axis,
    second_order: tl.constexpr,
):
    edge = normalize3_jets(
        jet_input(ex, 0, dex, axis),
        jet_input(ey, 1, dey, axis),
        jet_input(ez, 2, dez, axis),
        NORMALIZE_EPS,
        axis,
        second_order,
    )
    ex_jet, ey_jet, ez_jet = edge
    zero = jet_constant(0.0, axis)
    one = jet_constant(1.0, axis)
    chart1 = normalize4_jets(
        jet_add(one, ey_jet),
        jet_negate(ez_jet),
        zero,
        ex_jet,
        tl.sqrt(FLOAT_EPS),
        axis,
        second_order,
    )
    chart2 = normalize4_jets(
        jet_negate(ez_jet),
        jet_subtract(one, ey_jet),
        ex_jet,
        zero,
        tl.sqrt(FLOAT_EPS),
        axis,
        second_order,
    )
    t = jet_scale(
        jet_add(ey_jet, jet_constant(0.9, axis)),
        1.0 / 1.8,
        axis,
        second_order,
    )
    t = jet_where(t[0] > 0.0, t, zero)
    t = jet_where(t[0] < 1.0, t, one)
    numerator = jet_subtract(jet_scale(t, 2.0, axis, second_order), one)
    denominator = jet_multiply(t, jet_subtract(one, t), second_order)
    denominator = jet_where(
        denominator[0] >= FLOAT_EPS,
        denominator,
        jet_constant(FLOAT_EPS, axis),
    )
    argument = jet_multiply(
        numerator, jet_reciprocal(denominator, second_order), second_order
    )
    blend = jet_sigmoid(argument, second_order)
    blend = jet_where(t[0] >= FLOAT_EPS, blend, zero)
    blend = jet_where(t[0] <= 1.0 - FLOAT_EPS, blend, one)
    dot = jet_add(
        jet_add(
            jet_multiply(chart2[0], chart1[0], second_order),
            jet_multiply(chart2[1], chart1[1], second_order),
        ),
        jet_add(
            jet_multiply(chart2[2], chart1[2], second_order),
            jet_multiply(chart2[3], chart1[3], second_order),
        ),
    )
    chart2 = (
        jet_where(dot[0] < 0.0, jet_negate(chart2[0]), chart2[0]),
        jet_where(dot[0] < 0.0, jet_negate(chart2[1]), chart2[1]),
        jet_where(dot[0] < 0.0, jet_negate(chart2[2]), chart2[2]),
        jet_where(dot[0] < 0.0, jet_negate(chart2[3]), chart2[3]),
    )
    one_minus_blend = jet_subtract(one, blend)
    mixed = (
        jet_add(
            jet_multiply(one_minus_blend, chart2[0], second_order),
            jet_multiply(blend, chart1[0], second_order),
        ),
        jet_add(
            jet_multiply(one_minus_blend, chart2[1], second_order),
            jet_multiply(blend, chart1[1], second_order),
        ),
        jet_add(
            jet_multiply(one_minus_blend, chart2[2], second_order),
            jet_multiply(blend, chart1[2], second_order),
        ),
        jet_add(
            jet_multiply(one_minus_blend, chart2[3], second_order),
            jet_multiply(blend, chart1[3], second_order),
        ),
    )
    edge_q = normalize4_jets(
        mixed[0],
        mixed[1],
        mixed[2],
        mixed[3],
        NORMALIZE_EPS,
        axis,
        second_order,
    )
    cosine = tl.cos(0.5 * gamma)
    sine = tl.sin(0.5 * gamma)
    w, x, y, z = edge_q
    return (
        jet_subtract(
            jet_scale(w, cosine, axis, second_order),
            jet_scale(y, sine, axis, second_order),
        ),
        jet_add(
            jet_scale(x, cosine, axis, second_order),
            jet_scale(z, sine, axis, second_order),
        ),
        jet_add(
            jet_scale(y, cosine, axis, second_order),
            jet_scale(w, sine, axis, second_order),
        ),
        jet_subtract(
            jet_scale(z, cosine, axis, second_order),
            jet_scale(x, sine, axis, second_order),
        ),
    )


@triton.jit
def jet_constant(value, axis):
    zero = tl.zeros(axis.shape, tl.float32)
    return value, zero, 0.0, zero


@triton.jit
def jet_input(value, variable: tl.constexpr, direction, axis):
    gradient = tl.where(axis == variable, 1.0, 0.0)
    return value, gradient, direction, tl.zeros(axis.shape, tl.float32)


@triton.jit
def jet_add(left, right):
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    )


@triton.jit
def jet_negate(value):
    return -value[0], -value[1], -value[2], -value[3]


@triton.jit
def jet_subtract(left, right):
    return jet_add(left, jet_negate(right))


@triton.jit
def jet_multiply(left, right, second_order: tl.constexpr):
    value = left[0] * right[0]
    gradient = left[1] * right[0] + left[0] * right[1]
    if second_order:
        direction = left[2] * right[0] + left[0] * right[2]
        derivative_gradient = (
            left[3] * right[0]
            + left[1] * right[2]
            + left[2] * right[1]
            + left[0] * right[3]
        )
    else:
        direction = 0.0
        derivative_gradient = tl.zeros(gradient.shape, tl.float32)
    return value, gradient, direction, derivative_gradient


@triton.jit
def jet_unary(
    value,
    output,
    first_derivative,
    second_derivative,
    second_order: tl.constexpr,
):
    gradient = first_derivative * value[1]
    if second_order:
        direction = first_derivative * value[2]
        derivative_gradient = (
            second_derivative * value[2] * value[1] + first_derivative * value[3]
        )
    else:
        direction = 0.0
        derivative_gradient = tl.zeros(gradient.shape, tl.float32)
    return output, gradient, direction, derivative_gradient


@triton.jit
def jet_reciprocal(value, second_order: tl.constexpr):
    output = 1.0 / value[0]
    return jet_unary(
        value,
        output,
        -(output * output),
        2.0 * output * output * output,
        second_order,
    )


@triton.jit
def jet_sqrt(value, second_order: tl.constexpr):
    output = tl.sqrt(value[0])
    return jet_unary(
        value,
        output,
        0.5 / output,
        -0.25 / (value[0] * output),
        second_order,
    )


@triton.jit
def jet_sigmoid(value, second_order: tl.constexpr):
    output = tl.sigmoid(value[0])
    first = output * (1.0 - output)
    return jet_unary(
        value,
        output,
        first,
        first * (1.0 - 2.0 * output),
        second_order,
    )


@triton.jit
def jet_where(condition, left, right):
    return (
        tl.where(condition, left[0], right[0]),
        tl.where(condition, left[1], right[1]),
        tl.where(condition, left[2], right[2]),
        tl.where(condition, left[3], right[3]),
    )


@triton.jit
def jet_scale(value, scalar, axis, second_order: tl.constexpr):
    return jet_multiply(value, jet_constant(scalar, axis), second_order)


ROTATION_MASK_THRESHOLD = 0.999999


ROTATION_MASK_THRESHOLD_TL = tl.constexpr(ROTATION_MASK_THRESHOLD)


@triton.jit
def jet_sin(x, second_order: tl.constexpr):
    value = tl.sin(x[0])
    first = tl.cos(x[0])
    return jet_unary(x, value, first, -value, second_order)


@triton.jit
def jet_cos(x, second_order: tl.constexpr):
    value = tl.cos(x[0])
    first = -tl.sin(x[0])
    return jet_unary(x, value, first, -value, second_order)


@triton.jit
def jet_atan2(y, x, second_order: tl.constexpr):
    denominator = x[0] * x[0] + y[0] * y[0]
    y_partial = x[0] / denominator
    x_partial = -y[0] / denominator
    gradient = y_partial * y[1] + x_partial * x[1]
    if second_order:
        denominator_direction = 2.0 * (x[0] * x[2] + y[0] * y[2])
        y_partial_direction = x[2] / denominator - x[0] * denominator_direction / (
            denominator * denominator
        )
        x_partial_direction = -y[2] / denominator + y[0] * denominator_direction / (
            denominator * denominator
        )
        direction = y_partial * y[2] + x_partial * x[2]
        derivative_gradient = (
            y_partial_direction * y[1]
            + y_partial * y[3]
            + x_partial_direction * x[1]
            + x_partial * x[3]
        )
    else:
        direction = 0.0
        derivative_gradient = tl.zeros(y[1].shape, tl.float32)
    return (
        libdevice.atan2(y[0], x[0]),
        gradient,
        direction,
        derivative_gradient,
    )


@triton.jit
def load_rotation_jet(
    rotation_ptr,
    direction_ptr,
    edge,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    direction_stride_edge,
    direction_stride_row,
    direction_stride_column,
    variable: tl.constexpr,
    axis,
    second_order: tl.constexpr,
):
    row: tl.constexpr = variable // 3
    column: tl.constexpr = variable % 3
    value = tl.load(
        rotation_ptr
        + edge * rotation_stride_edge
        + row * rotation_stride_row
        + column * rotation_stride_column
    ).to(tl.float32)
    if second_order:
        direction = tl.load(
            direction_ptr
            + edge * direction_stride_edge
            + row * direction_stride_row
            + column * direction_stride_column
        ).to(tl.float32)
    else:
        direction = 0.0
    return jet_input(value, variable, direction, axis)


@triton.jit
def angle_jets(
    rotation_ptr,
    direction_ptr,
    edge,
    axis,
    rotation_stride_edge,
    rotation_stride_row,
    rotation_stride_column,
    direction_stride_edge,
    direction_stride_row,
    direction_stride_column,
    second_order: tl.constexpr,
):
    r00 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        0,
        axis,
        second_order,
    )
    x0 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        1,
        axis,
        second_order,
    )
    r02 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        2,
        axis,
        second_order,
    )
    x1 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        4,
        axis,
        second_order,
    )
    r20 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        6,
        axis,
        second_order,
    )
    x2 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        7,
        axis,
        second_order,
    )
    r22 = load_rotation_jet(
        rotation_ptr,
        direction_ptr,
        edge,
        rotation_stride_edge,
        rotation_stride_row,
        rotation_stride_column,
        direction_stride_edge,
        direction_stride_row,
        direction_stride_column,
        8,
        axis,
        second_order,
    )
    radial_squared = jet_add(
        jet_multiply(x0, x0, second_order), jet_multiply(x2, x2, second_order)
    )
    radial = jet_sqrt(radial_squared, second_order)
    alpha = jet_atan2(x0, x2, second_order)
    beta = jet_atan2(radial, x1, second_order)
    alpha_cos = jet_cos(alpha, second_order)
    alpha_sin = jet_sin(alpha, second_order)
    gamma_numerator = jet_subtract(
        jet_multiply(alpha_cos, r02, second_order),
        jet_multiply(alpha_sin, r22, second_order),
    )
    gamma_denominator = jet_subtract(
        jet_multiply(alpha_cos, r00, second_order),
        jet_multiply(alpha_sin, r20, second_order),
    )
    gamma = jet_atan2(
        gamma_numerator,
        gamma_denominator,
        second_order,
    )
    return alpha, beta, gamma, x1[0]


def monomial_exponents(degree: int, device: torch.device) -> Tensor:
    """Enumerate degree-D quaternion monomials as int32 ``(binom(D+3,3),4)``.

    Columns are powers of (w,x,y,z), each row sums to ``degree``, and rows
    follow lexicographic order. This setup tensor is constant during geometry
    differentiation. A row a represents ``q[:,0]**a[0] * ... * q[:,3]**a[3]``.
    """
    values = tuple(
        exponents
        for exponents in product(range(degree + 1), repeat=4)
        if sum(exponents) == degree
    )
    return torch.tensor(values, device=device, dtype=torch.int32).contiguous()


@triton.jit
def monomials_integer_power(value, exponent, max_power: tl.constexpr):
    if max_power > 12:
        result = tl.full(exponent.shape, 1.0, tl.float32)
        factor = value
        for bit in tl.static_range(31):
            if (1 << bit) <= max_power:
                result *= tl.where((exponent & (1 << bit)) != 0, factor, 1.0)
                factor *= factor
        return result
    square = value * value
    cube = square * value
    fourth = square * square
    fifth = fourth * value
    sixth = cube * cube
    seventh = sixth * value
    eighth = fourth * fourth
    ninth = eighth * value
    tenth = fifth * fifth
    eleventh = tenth * value
    twelfth = sixth * sixth
    result = tl.where(exponent == 0, 1.0, value)
    result = tl.where(exponent == 2, square, result)
    result = tl.where(exponent == 3, cube, result)
    result = tl.where(exponent == 4, fourth, result)
    result = tl.where(exponent == 5, fifth, result)
    result = tl.where(exponent == 6, sixth, result)
    result = tl.where(exponent == 7, seventh, result)
    result = tl.where(exponent == 8, eighth, result)
    result = tl.where(exponent == 9, ninth, result)
    result = tl.where(exponent == 10, tenth, result)
    result = tl.where(exponent == 11, eleventh, result)
    return tl.where(exponent == 12, twelfth, result)


@triton.jit
def load_quaternion(q_ptr, edge):
    base = edge * 4
    return (
        tl.load(q_ptr + base).to(tl.float32),
        tl.load(q_ptr + base + 1).to(tl.float32),
        tl.load(q_ptr + base + 2).to(tl.float32),
        tl.load(q_ptr + base + 3).to(tl.float32),
    )


@triton.jit
def monomial_values(q_ptr, exponents_ptr, edge, index, mask, max_power: tl.constexpr):
    w, x, y, z = load_quaternion(q_ptr, edge)
    exponent_base = index * 4
    ew = tl.load(exponents_ptr + exponent_base, mask=mask, other=0)
    ex = tl.load(exponents_ptr + exponent_base + 1, mask=mask, other=0)
    ey = tl.load(exponents_ptr + exponent_base + 2, mask=mask, other=0)
    ez = tl.load(exponents_ptr + exponent_base + 3, mask=mask, other=0)
    value = (
        monomials_integer_power(w, ew, max_power)
        * monomials_integer_power(x, ex, max_power)
        * monomials_integer_power(y, ey, max_power)
        * monomials_integer_power(z, ez, max_power)
    )
    return value, (ew, ex, ey, ez), (w, x, y, z)


@triton.jit
def monomial_derivative(exponents, values, axis, max_power: tl.constexpr):
    result = tl.full(exponents[0].shape, 1.0, tl.float32)
    factor = tl.zeros(exponents[0].shape, tl.float32)
    for variable in tl.static_range(4):
        exponent = exponents[variable]
        factor = tl.where(axis == variable, exponent.to(tl.float32), factor)
        result *= monomials_integer_power(
            values[variable],
            tl.maximum(0, exponent - tl.where(axis == variable, 1, 0)),
            max_power,
        )
    return factor * result


def block_monomials(count: int) -> int:
    return triton.next_power_of_2(count)


POLE_TOLERANCE = tl.constexpr(0.999999)


PI = tl.constexpr(math.pi)


@triton.jit
def frame_angle_edges(edges, tile: tl.constexpr):
    if tile == 1:
        return tl.program_id(0), True
    else:
        edge = tl.program_id(0) * tile + tl.arange(0, tile)
        return tl.minimum(edge, edges - 1), edge < edges


def frame_angle_tile(rotation, *, masked_derivatives=False):
    if (
        rotation.shape[0] >= 16384
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(rotation.device) in ((8, 0), (9, 0))
    ):
        return (16, 4) if masked_derivatives else (64, 2)
    return 1, 1


def use_quaternion_tiles(tensor):
    return (
        tensor.is_cuda
        and torch.version.hip is None
        and tensor.shape[0] >= 16384
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(tensor.device) in ((8, 0), (9, 0))
    )


@triton.jit
def rotation_values(rotation_ptr, edge, stride_edge, stride_row, stride_column):
    base = edge * stride_edge
    x = tl.load(rotation_ptr + base + stride_column).to(tl.float32)
    y = tl.load(rotation_ptr + base + stride_row + stride_column).to(tl.float32)
    z = tl.load(rotation_ptr + base + 2 * stride_row + stride_column).to(tl.float32)
    r02 = tl.load(rotation_ptr + base + 2 * stride_column).to(tl.float32)
    r22 = tl.load(rotation_ptr + base + 2 * stride_row + 2 * stride_column).to(
        tl.float32
    )
    r00 = tl.load(rotation_ptr + base).to(tl.float32)
    r20 = tl.load(rotation_ptr + base + 2 * stride_row).to(tl.float32)
    alpha = libdevice.atan2(x, z)
    norm = tl.sqrt(x * x + y * y + z * z)
    cosine_beta = tl.maximum(-1.0, tl.minimum(1.0, y / norm))
    beta = libdevice.acos(cosine_beta)
    cosine_alpha = tl.cos(alpha)
    sine_alpha = tl.sin(alpha)
    gamma_y = cosine_alpha * r02 - sine_alpha * r22
    gamma_x = cosine_alpha * r00 - sine_alpha * r20
    gamma = libdevice.atan2(gamma_y, gamma_x)
    regular = (y > -POLE_TOLERANCE) & (y < POLE_TOLERANCE)
    # Equality detaches the angle in upstream eSEN but preserves its value.
    beta = tl.where(
        y > POLE_TOLERANCE,
        0.0,
        tl.where(y < -POLE_TOLERANCE, PI, beta),
    )
    return (
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
    )


@triton.jit
def angle_derivatives(values):
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
    radial_squared = tl.where(regular, x * x + z * z, 1.0)
    radial = tl.sqrt(radial_squared)
    norm_squared = radial_squared + y * y
    alpha_x = z / radial_squared
    alpha_z = -x / radial_squared
    beta_x = y * x / (radial * norm_squared)
    beta_y = -radial / norm_squared
    beta_z = y * z / (radial * norm_squared)
    gamma_denominator = tl.where(regular, gamma_y * gamma_y + gamma_x * gamma_x, 1.0)
    gamma_y_gradient = gamma_x / gamma_denominator
    gamma_x_gradient = -gamma_y / gamma_denominator
    gamma_y_alpha = -sine_alpha * r02 - cosine_alpha * r22
    gamma_x_alpha = -sine_alpha * r00 - cosine_alpha * r20
    gamma_alpha = gamma_y_gradient * gamma_y_alpha + gamma_x_gradient * gamma_x_alpha
    return (
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
    )


QUADRATIC_EXPONENTS = ((0, 0, 0, 0),) + tuple(
    exponents for exponents in product(range(3), repeat=4) if sum(exponents) == 2
)


QUARTIC_EXPONENTS = tuple(
    exponents for exponents in product(range(5), repeat=4) if sum(exponents) == 4
)


QUADRATIC_W = tuple(value[0] for value in QUADRATIC_EXPONENTS)


QUADRATIC_X = tuple(value[1] for value in QUADRATIC_EXPONENTS)


QUADRATIC_Y = tuple(value[2] for value in QUADRATIC_EXPONENTS)


QUADRATIC_Z = tuple(value[3] for value in QUADRATIC_EXPONENTS)


QUARTIC_W = tuple(value[0] for value in QUARTIC_EXPONENTS)


QUARTIC_X = tuple(value[1] for value in QUARTIC_EXPONENTS)


QUARTIC_Y = tuple(value[2] for value in QUARTIC_EXPONENTS)


QUARTIC_Z = tuple(value[3] for value in QUARTIC_EXPONENTS)


QUADRATIC_COUNT_TL = tl.constexpr(len(QUADRATIC_EXPONENTS))


QUARTIC_COUNT_TL = tl.constexpr(len(QUARTIC_EXPONENTS))


QUADRATIC_W_TL = tl.constexpr(QUADRATIC_W)


QUADRATIC_X_TL = tl.constexpr(QUADRATIC_X)


QUADRATIC_Y_TL = tl.constexpr(QUADRATIC_Y)


QUADRATIC_Z_TL = tl.constexpr(QUADRATIC_Z)


QUARTIC_W_TL = tl.constexpr(QUARTIC_W)


QUARTIC_X_TL = tl.constexpr(QUARTIC_X)


QUARTIC_Y_TL = tl.constexpr(QUARTIC_Y)


QUARTIC_Z_TL = tl.constexpr(QUARTIC_Z)


def coefficient_index(exponents, table):
    return table.index(tuple(exponents))


def build_quadratic_coefficients() -> torch.Tensor:
    coefficients = torch.zeros(9, len(QUADRATIC_EXPONENTS), dtype=torch.float32)

    def add(output, value, exponents):
        coefficients[output, coefficient_index(exponents, QUADRATIC_EXPONENTS)] += (
            value
        )

    for output in (0, 4, 8):
        add(output, 1.0, (0, 0, 0, 0))
    add(0, -2.0, (0, 0, 2, 0))
    add(0, -2.0, (0, 0, 0, 2))
    add(1, 2.0, (0, 1, 1, 0))
    add(1, -2.0, (1, 0, 0, 1))
    add(2, 2.0, (0, 1, 0, 1))
    add(2, 2.0, (1, 0, 1, 0))
    add(3, 2.0, (0, 1, 1, 0))
    add(3, 2.0, (1, 0, 0, 1))
    add(4, -2.0, (0, 2, 0, 0))
    add(4, -2.0, (0, 0, 0, 2))
    add(5, 2.0, (0, 0, 1, 1))
    add(5, -2.0, (1, 1, 0, 0))
    add(6, 2.0, (0, 1, 0, 1))
    add(6, -2.0, (1, 0, 1, 0))
    add(7, 2.0, (0, 0, 1, 1))
    add(7, 2.0, (1, 1, 0, 0))
    add(8, -2.0, (0, 2, 0, 0))
    add(8, -2.0, (0, 0, 2, 0))
    return coefficients


def compress_l2_coefficients(coefficients: torch.Tensor) -> torch.Tensor:
    """Collect UMA's ``(5,5,4,4,4,4)`` tensor into a ``(25,35)`` quartic table.

    Each ordered product of four quaternion components contributes to the
    corresponding exponent row in ``QUARTIC_EXPONENTS``. Row-major 5x5
    outputs remain in natural degree-two matrix order. Constants are detached,
    accumulated in CPU FP32, and returned on the supplied table's device.
    This setup operation is not an edge-dependent differentiable calculation.
    """
    if tuple(coefficients.shape) != (5, 5, 4, 4, 4, 4):
        raise ValueError(f"unexpected L2 coefficient shape: {coefficients.shape}")
    source = coefficients.detach().to(device="cpu", dtype=torch.float32)
    compressed = torch.zeros(25, len(QUARTIC_EXPONENTS), dtype=torch.float32)
    for a, b, c, d in product(range(4), repeat=4):
        exponents = tuple((a, b, c, d).count(variable) for variable in range(4))
        index = coefficient_index(exponents, QUARTIC_EXPONENTS)
        compressed[:, index] += source[:, :, a, b, c, d].reshape(-1)
    return compressed.to(device=coefficients.device).contiguous()


def quadratic_coefficients(device: torch.device) -> torch.Tensor:
    """Return the fixed FP32 ``(9,11)`` table for the degree-one rotation.

    The 3x3 outputs are flattened in row order, using the standard (w,x,y,z)
    quaternion matrix. Columns are a constant 1 followed by the degree-two
    monomials in lexicographic order. ``build_quadratic_coefficients`` shows every
    nonzero term explicitly. Select retained rows before per-edge evaluation.
    """
    return build_quadratic_coefficients().to(device=device).contiguous()


def polynomial_wigner_coefficients(
    degree: int,
    transformation: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Prepare a fixed real-basis degree-l quaternion Wigner polynomial table.

    ``transformation`` contains real and imaginary parts of the supplied
    complex-to-real ``(2*l+1,2*l+1)`` basis change. Return FP32
    ``((2*l+1)**2, binom(2*l+3,3))`` coefficients on ``device``. Each row is
    one natural-order matrix entry. Columns match degree-2*l quaternion
    monomials from ``monomial_exponents``. Construction uses CPU complex128
    and checks that the transformed coefficients are real. Basis constants
    are detached and generated once, outside geometry-dependent computation.
    """
    if degree < 1:
        raise ValueError("polynomial Wigner degree must be positive")
    total_degree = 2 * degree
    exponents = tuple(
        value
        for value in product(range(total_degree + 1), repeat=4)
        if sum(value) == total_degree
    )
    exponent_index = {value: index for index, value in enumerate(exponents)}
    width = 2 * degree + 1
    coefficients = torch.zeros(width, width, len(exponents), dtype=torch.complex128)
    for row, mp in enumerate(range(-degree, degree + 1)):
        for column, m in enumerate(range(-degree, degree + 1)):
            square_root = math.sqrt(
                math.factorial(degree + m)
                * math.factorial(degree - m)
                / (math.factorial(degree + mp) * math.factorial(degree - mp))
            )
            rho_min = max(0, mp - m)
            rho_max = min(degree + mp, degree - m)
            for rho in range(rho_min, rho_max + 1):
                scalar = (
                    square_root
                    * math.comb(degree + mp, rho)
                    * math.comb(degree - mp, degree - m - rho)
                    * ((-1) ** rho)
                )
                ra_power = 2 * degree + mp - m - 2 * rho
                rb_power = m - mp + 2 * rho
                ra_phase = mp + m
                rb_phase = m - mp
                ra = (ra_power + ra_phase) // 2
                ra_conjugate = (ra_power - ra_phase) // 2
                rb = (rb_power + rb_phase) // 2
                rb_conjugate = (rb_power - rb_phase) // 2
                for z_positive in range(ra + 1):
                    for z_negative in range(ra_conjugate + 1):
                        z_power = z_positive + z_negative
                        w_power = ra + ra_conjugate - z_power
                        ra_coefficient = (
                            math.comb(ra, z_positive)
                            * (1j**z_positive)
                            * math.comb(ra_conjugate, z_negative)
                            * ((-1j) ** z_negative)
                        )
                        for x_positive in range(rb + 1):
                            for x_negative in range(rb_conjugate + 1):
                                x_power = x_positive + x_negative
                                y_power = rb + rb_conjugate - x_power
                                rb_coefficient = (
                                    math.comb(rb, x_positive)
                                    * (1j**x_positive)
                                    * math.comb(rb_conjugate, x_negative)
                                    * ((-1j) ** x_negative)
                                )
                                index = exponent_index[
                                    (w_power, x_power, y_power, z_power)
                                ]
                                coefficients[row, column, index] += (
                                    scalar * ra_coefficient * rb_coefficient
                                )
    real, imaginary = transformation
    transform = real.detach().cpu().double() + 1j * imaginary.detach().cpu().double()
    transformed = torch.einsum(
        "ai,ijm,bj->abm", transform, coefficients, transform.conj()
    )
    if transformed.imag.abs().max().item() >= 1e-9:
        raise RuntimeError("real Wigner polynomial has a complex residual")
    return transformed.real.flatten(0, 1).float().to(device=device).contiguous()


@triton.jit
def polynomial_integer_power(value, exponent):
    square = value * value
    cube = square * value
    fourth = square * square
    result = tl.where(exponent == 0, 1.0, value)
    result = tl.where(exponent == 2, square, result)
    result = tl.where(exponent == 3, cube, result)
    return tl.where(exponent == 4, fourth, result)


@triton.jit
def monomial(
    w,
    x,
    y,
    z,
    index: tl.constexpr,
    exponents_w: tl.constexpr,
    exponents_x: tl.constexpr,
    exponents_y: tl.constexpr,
    exponents_z: tl.constexpr,
):
    return (
        polynomial_integer_power(w, list(exponents_w)[index])
        * polynomial_integer_power(x, list(exponents_x)[index])
        * polynomial_integer_power(y, list(exponents_y)[index])
        * polynomial_integer_power(z, list(exponents_z)[index])
    )


@triton.jit
def polynomial_values(
    coefficients_ptr,
    w,
    x,
    y,
    z,
    output_count: tl.constexpr,
    monomial_count: tl.constexpr,
    exponents_w: tl.constexpr,
    exponents_x: tl.constexpr,
    exponents_y: tl.constexpr,
    exponents_z: tl.constexpr,
    block_outputs: tl.constexpr,
):
    output = tl.arange(0, block_outputs)
    mask = output < output_count
    values = tl.zeros((block_outputs,), tl.float32)
    for index in tl.static_range(monomial_count):
        coefficient = tl.load(
            coefficients_ptr + output * monomial_count + index,
            mask=mask,
            other=0.0,
        )
        values += coefficient * monomial(
            w,
            x,
            y,
            z,
            index,
            exponents_w,
            exponents_x,
            exponents_y,
            exponents_z,
        )
    return values, mask


def quaternion_exponent_codes(powers):
    return tl.constexpr(tuple(sum(power << (3 * j) for j, power in enumerate(row))
                              for row in powers))


QUADRATIC_CODES_TL = quaternion_exponent_codes(QUADRATIC_EXPONENTS)
QUARTIC_CODES_TL = quaternion_exponent_codes(QUARTIC_EXPONENTS)
QUADRATIC_L2_OFFSETS_TL = tl.constexpr(tuple(range(1, 10)))
QUARTIC_L2_OFFSETS_TL = tl.constexpr(tuple(range(10, 35)))


@triton.jit
def pack_sparse_quaternion_kernel(quadratic, quartic, coefficients, codes, usable):
    row = tl.arange(0, 64)[:, None]
    col = tl.arange(0, 64)[None, :]
    is_quadratic = (row > 0) & (row < 10)
    is_quartic = (row >= 10) & (row < 35)
    quad = tl.load(quadratic + (row - 1) * 11 + col, is_quadratic & (col < 11), 0)
    quart = tl.load(quartic + (row - 10) * 35 + col, is_quartic & (col < 35), 0)
    value = tl.where(row == 0, tl.where(col == 0, 1.0, 0.0), quad + quart)
    nonzero = value != 0.0
    max_terms = tl.max(tl.sum(nonzero.to(tl.int32), axis=1), axis=0)
    in_bounds = tl.max(tl.max((~(tl.abs(value) <= 1e8)).to(tl.int32), axis=1), axis=0) == 0
    tl.store(usable, (max_terms <= 10) & in_bounds)

    # Sorting by unique original column preserves each row's contraction order.
    index = tl.broadcast_to(col, value.shape)
    ordered = tl.sort(tl.where(nonzero, index, index + 64), descending=False) % 64
    packed = tl.gather(value, ordered, axis=1)
    qcode = tl.zeros((1, 64), tl.int32)
    wcode = tl.zeros((1, 64), tl.int32)
    for j in tl.static_range(11):
        qcode = tl.where(col == j, list(QUADRATIC_CODES_TL)[j], qcode)
    for j in tl.static_range(35):
        wcode = tl.where(col == j, list(QUARTIC_CODES_TL)[j], wcode)
    power = tl.gather(tl.where(is_quadratic, qcode, tl.where(is_quartic, wcode, 0)),
                      ordered, axis=1)
    out = col * 35 + row
    active = (col < 10) & (row < 35)
    tl.store(coefficients + out, packed, active)
    tl.store(codes + out, power, active)

@triton.jit
def quaternion_power_terms(q, direction, power, phase: tl.constexpr):
    square = q * q
    cube = square * q
    value = tl.where(power <= 2, tl.where(power == 0, 1.0, tl.where(power == 1, q, square)),
                     tl.where(power == 3, cube, square * square))
    gradient = 0.0
    directional = 0.0
    second = 0.0
    if phase:
        gradient = tl.where(power <= 2, tl.where(power == 0, 0.0, tl.where(power == 1, 1.0, 2.0 * q)),
                            tl.where(power == 3, 3.0 * square, 4.0 * cube))
    if phase == 2:
        directional = gradient * direction
        second = tl.where(power <= 1, 0.0, tl.where(power == 2, 2.0,
                          tl.where(power == 3, 6.0 * q, 12.0 * square))) * direction
    return value, gradient, directional, second


@triton.jit
def sparse_quaternion_tile(
    q, direction, matrix_gradient, coefficient, exponent, matrix, q_out, edges, start,
    tile_edges: tl.constexpr, max_terms: tl.constexpr, phase: tl.constexpr,
):
    edge = start + tl.arange(0, tile_edges)[:, None]
    col = tl.arange(0, 64)[None, :]
    edge_mask = edge < edges
    mask = edge_mask & (col < 35)
    address = edge * 35 + col
    w = tl.load(q + edge * 4, edge_mask, 0)
    x = tl.load(q + edge * 4 + 1, edge_mask, 0)
    y = tl.load(q + edge * 4 + 2, edge_mask, 0)
    z = tl.load(q + edge * 4 + 3, edge_mask, 0)
    dw, dx, dy, dz = 0.0, 0.0, 0.0, 0.0
    if phase == 2:
        dw = tl.load(direction + edge * 4, edge_mask, 0)
        dx = tl.load(direction + edge * 4 + 1, edge_mask, 0)
        dy = tl.load(direction + edge * 4 + 2, edge_mask, 0)
        dz = tl.load(direction + edge * 4 + 3, edge_mask, 0)
    if phase:
        seed = tl.load(matrix_gradient + address, mask, 0)
    value = tl.zeros((tile_edges, 64), tl.float32)
    aw = tl.zeros((tile_edges, 64), tl.float32)
    ax = tl.zeros((tile_edges, 64), tl.float32)
    ay = tl.zeros((tile_edges, 64), tl.float32)
    az = tl.zeros((tile_edges, 64), tl.float32)
    for index in tl.static_range(max_terms):
        coeff = tl.load(coefficient + index * 35 + col, col < 35, 0)
        code = tl.load(exponent + index * 35 + col, col < 35, 0)
        wv, wg, wd, wh = quaternion_power_terms(w, dw, code & 7, phase)
        xv, xg, xd, xh = quaternion_power_terms(x, dx, (code >> 3) & 7, phase)
        yv, yg, yd, yh = quaternion_power_terms(y, dy, (code >> 6) & 7, phase)
        zv, zg, zd, zh = quaternion_power_terms(z, dz, (code >> 9) & 7, phase)
        left = wv * xv
        right = yv * zv
        if phase == 0:
            value += coeff * (left * right)
        elif phase == 1:
            weighted = coeff * seed
            aw += weighted * (wg * xv * right)
            ax += weighted * (xg * wv * right)
            ay += weighted * (yg * zv * left)
            az += weighted * (zg * yv * left)
        else:
            dleft = wd * xv + wv * xd
            dright = yd * zv + yv * zd
            value += coeff * (dleft * right + left * dright)
            weighted = coeff * seed
            aw += weighted * (wh * xv * right + wg * (xd * right + xv * dright))
            ax += weighted * (xh * wv * right + xg * (wd * right + wv * dright))
            ay += weighted * (yh * zv * left + yg * (zd * left + zv * dleft))
            az += weighted * (zh * yv * left + zg * (yd * left + yv * dleft))
    if phase != 1:
        tl.store(matrix + address, value, mask)
    if phase:
        destination = tl.reshape(edge * 4, (tile_edges,))
        valid = tl.reshape(edge_mask, (tile_edges,))
        tl.store(q_out + destination, tl.sum(aw, axis=1), valid)
        tl.store(q_out + destination + 1, tl.sum(ax, axis=1), valid)
        tl.store(q_out + destination + 2, tl.sum(ay, axis=1), valid)
        tl.store(q_out + destination + 3, tl.sum(az, axis=1), valid)


@triton.jit
def adaptive_quaternion_tile(
    q, direction, seed, quadratic, quartic, coefficients, codes, usable, matrix,
    q_out, edges, dense_edge: tl.constexpr, phase: tl.constexpr,
):
    tile: tl.constexpr = 4
    start = tl.program_id(0) * tile
    edge = start + tl.arange(0, tile)[:, None]
    vector = tl.arange(0, 4)[None, :]
    live = edge < edges
    value = tl.load(q + 4 * edge + vector, live, 0)
    safe = tl.max(tl.max((~(tl.abs(value) <= 2.0)).to(tl.int32), axis=1), axis=0) == 0
    if phase:
        column = tl.arange(0, 64)[None, :]
        upstream = tl.load(seed + 35 * edge + column, live & (column < 35), 0)
        safe &= tl.max(tl.max((~(tl.abs(upstream) <= 1e8)).to(tl.int32), axis=1), axis=0) == 0
    if phase == 2:
        vector_direction = tl.load(direction + 4 * edge + vector, live, 0)
        safe &= tl.max(tl.max((~(tl.abs(vector_direction) <= 1e8)).to(tl.int32), axis=1), axis=0) == 0
    if safe & (tl.load(usable) != 0):
        sparse_quaternion_tile(
            q, direction, seed, coefficients, codes, matrix, q_out, edges, start, tile, 10, phase,
        )
    else:
        for i in range(tile):
            item = start + i
            if item < edges:
                if phase == 0:
                    dense_edge(
                        q, quadratic, quartic, matrix, item, 9, 25,
                        QUADRATIC_L2_OFFSETS_TL, QUARTIC_L2_OFFSETS_TL,
                    )
                elif phase == 1:
                    dense_edge(
                        seed, q, quadratic, quartic, q_out, item, 9, 25,
                        QUADRATIC_L2_OFFSETS_TL, QUARTIC_L2_OFFSETS_TL,
                    )
                else:
                    tl.store(matrix + item * 35, 0.0)
                    dense_edge(
                        seed, q, direction, quadratic, quartic, matrix, q_out, item, 9, 25,
                        QUADRATIC_L2_OFFSETS_TL, QUARTIC_L2_OFFSETS_TL,
                    )


sparse_quaternion_bases = {}
sparse_quaternion_registration_lock = Lock()


def register_sparse_quaternion_basis(quadratic, quartic):
    """Validate model-owned coefficient sparsity outside CUDA capture."""
    if torch.compiler.is_compiling():
        return None
    if (
        quadratic.shape != (9, 11) or quartic.shape != (25, 35)
        or quadratic.dtype != torch.float32 or quartic.dtype != torch.float32
        or quadratic.device != quartic.device
        or quadratic.requires_grad or quartic.requires_grad
        or quadratic.is_inference() or quartic.is_inference()
    ):
        return False
    if quadratic.is_cuda:
        with torch.cuda.device(quadratic.device):
            if torch.cuda.is_current_stream_capturing():
                return None
    versions = quadratic._version, quartic._version
    for table in (quadratic, quartic):
        host = table.detach().cpu()
        if not (host.abs() <= 1e8).all() or host.count_nonzero(dim=1).max() > 10:
            return False
    if versions != (quadratic._version, quartic._version):
        return False

    # The registry stores no CUDA tensors or derived coefficients.
    key = id(quadratic), id(quartic)
    with sparse_quaternion_registration_lock:
        sparse_quaternion_bases.pop(key, None)
        if len(sparse_quaternion_bases) >= 64:
            sparse_quaternion_bases.pop(next(iter(sparse_quaternion_bases)))
        sparse_quaternion_bases[key] = ref(quadratic), ref(quartic), versions
    return True


def use_sparse_quaternion(q, quadratic, quartic, mmax, compile_visible):
    if mmax != 2 or compile_visible or torch.compiler.is_compiling():
        return False
    registration = sparse_quaternion_bases.get((id(quadratic), id(quartic)))
    if registration is None:
        return False
    quad_owner, quart_owner, versions = registration
    return (
        quad_owner() is quadratic and quart_owner() is quartic
        and versions == (quadratic._version, quartic._version)
        and use_quaternion_tiles(q)
    )


def pack_sparse_quaternion(q, quadratic, quartic):
    # Capture places this scratch in the graph's own allocation pool.
    coefficient = q.new_empty((10, 35))
    codes = q.new_empty((10, 35), dtype=torch.int32)
    usable = q.new_empty((), dtype=torch.int32)
    pack_sparse_quaternion_kernel[(1,)](
        quadratic, quartic, coefficient, codes, usable, num_warps=8,
    )
    return coefficient, codes, usable


def selected_outputs(degrees: tuple[int, ...], mmax: int) -> tuple[int, ...]:
    """Indices in original, natural-order full polynomial coefficient tables."""
    lmax = max(degrees)
    layout = get_retained_layout(lmax=lmax, mmax=min(mmax, lmax), channels=1)
    sectors = get_angular_layout(lmax=lmax, mmax=min(mmax, lmax), channels=1)
    indices = []
    offset = 0
    for degree in degrees:
        width = 2 * degree + 1
        for row in layout.degree_rows[degree]:
            local = sectors.dense_rows[row] - degree * degree
            indices.extend(offset + local * width + col for col in range(width))
        offset += width * width
    return tuple(indices)


def select_retained_coefficients(
    coefficients: Tensor, *, degrees: tuple[int, ...], mmax: int
) -> Tensor:
    """Select retained output rows from fixed quaternion polynomial tables.

    ``coefficients`` has ``sum((2*l+1)**2 for l in degrees)`` rows and one
    column per monomial. Degrees are consecutive natural-order matrix blocks.
    Return the same monomial columns and selected ``m=0,+1,-1,...`` rows of
    each block, including all 2*l+1 matrix columns. The result can be used by
    the retained quaternion generator. Tables must not require gradients.
    """
    if coefficients.requires_grad:
        raise ValueError("Wigner polynomial coefficients must be fixed")
    if coefficients.ndim != 2 or coefficients.shape[0] != sum(
        (2 * degree + 1) ** 2 for degree in degrees
    ):
        raise ValueError("coefficient rows do not match their degrees")
    indices = torch.tensor(selected_outputs(degrees, mmax), device=coefficients.device)
    return coefficients.index_select(0, indices).contiguous()


def launch_constants(mmax: int):
    linear_count = 3 * (2 * min(1, mmax) + 1)
    quartic_count = 5 * (2 * min(2, mmax) + 1)
    return dict(
        linear_count=linear_count,
        quartic_count=quartic_count,
        linear_offsets=tuple(range(1, 1 + linear_count)),
        quartic_offsets=tuple(
            range(1 + linear_count, 1 + linear_count + quartic_count)
        ),
    )


class PolynomialBlock(NamedTuple):
    """A fixed polynomial map for one or more consecutive Wigner degrees.

    ``exponents`` is int32 ``(P,4)`` for powers of quaternion components.
    Each power is at most ``2*max(degrees)``. ``coefficients`` has shape
    ``(sum((2*min(l,M)+1)*(2*l+1) for l in degrees), P)`` and selected rows
    in degree order. A block with degrees ``(3,4)`` shares its monomials and
    one matrix product across both degrees.
    """

    degrees: tuple[int, ...]
    exponents: Tensor
    coefficients: Tensor


def validate_polynomial_blocks(blocks, *, lmax, mmax, edge_vectors):
    next_degree = 3
    for degrees, powers, table in blocks:
        if not degrees or degrees != tuple(
            range(next_degree, next_degree + len(degrees))
        ):
            raise ValueError(
                "polynomial blocks must cover consecutive degrees from 3 through L"
            )
        if degrees[-1] > lmax:
            raise ValueError("polynomial block degree exceeds L")
        rows = sum((2 * min(degree, mmax) + 1) * (2 * degree + 1) for degree in degrees)
        if (
            powers.ndim != 2
            or powers.shape[1] != 4
            or powers.shape[0] == 0
            or powers.dtype != torch.int32
            or powers.device != edge_vectors.device
        ):
            raise ValueError(
                "polynomial powers must be nonempty int32 [P,4] on the edge device"
            )
        if (
            table.shape != (rows, powers.shape[0])
            or table.device != edge_vectors.device
            or table.dtype != edge_vectors.dtype
            or table.requires_grad
        ):
            raise ValueError(
                "polynomial coefficients must contain fixed retained rows on the edge device"
            )
        next_degree = degrees[-1] + 1
    if next_degree != max(lmax + 1, 3):
        raise ValueError(
            "polynomial blocks must cover consecutive degrees from 3 through L"
        )


def selected_local_rows(lmax: int, mmax: int) -> tuple[tuple[int, ...], ...]:
    """Return local row indices ``(l,l+1,l-1,...)`` for each degree 0 through L.

    Row indices address the natural ``-l..+l`` rows of a ``(2*l+1)`` block.
    Only orders with ``abs(m) <= min(l,M)`` are retained. The result contains
    Python tuples, is independent of device and dtype, and has no gradients.
    """
    if lmax < 0 or not 0 <= mmax <= lmax:
        raise ValueError(f"invalid Wigner truncation: {lmax=} {mmax=}")
    return tuple(
        (degree,)
        + tuple(
            component
            for order in range(1, min(mmax, degree) + 1)
            for component in (degree + order, degree - order)
        )
        for degree in range(lmax + 1)
    )


@triton.jit
def selected_row_indices(
    row, selected_rows: tl.constexpr, row_count: tl.constexpr,
):
    selected = tl.full(row.shape, list(selected_rows)[0], tl.int32)
    for index in tl.static_range(1, row_count):
        selected = tl.where(row == index, list(selected_rows)[index], selected)
    return selected


@triton.jit
def wigner_terms(
    alpha, beta, gamma, matrix_j,
    degree: tl.constexpr, selected_rows: tl.constexpr,
    row_count: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_width: tl.constexpr,
):
    """Return D, its three angle derivatives, and the upper-triangle Hessian.

    Hessian entries follow (aa, ab, ag, bb, bg, gg), where a, b, g are the
    Euler angles alpha, beta, gamma.
    """
    row = tl.arange(0, block_rows)[:, None]
    column = tl.arange(0, block_width)[None, :]
    row_mask = row < row_count
    column_mask = column < width
    selected = selected_row_indices(row, selected_rows, row_count)
    flipped_selected = width - 1 - selected
    alpha_frequency = degree - selected
    alpha_phase = alpha * alpha_frequency
    alpha_cos = tl.cos(alpha_phase)
    alpha_sin = tl.sin(alpha_phase)

    c = tl.zeros((block_rows, block_width), tl.float32)
    ca = tl.zeros((block_rows, block_width), tl.float32)
    cb = tl.zeros((block_rows, block_width), tl.float32)
    caa = tl.zeros((block_rows, block_width), tl.float32)
    cab = tl.zeros((block_rows, block_width), tl.float32)
    cbb = tl.zeros((block_rows, block_width), tl.float32)
    c_flip = tl.zeros((block_rows, block_width), tl.float32)
    ca_flip = tl.zeros((block_rows, block_width), tl.float32)
    cb_flip = tl.zeros((block_rows, block_width), tl.float32)
    caa_flip = tl.zeros((block_rows, block_width), tl.float32)
    cab_flip = tl.zeros((block_rows, block_width), tl.float32)
    cbb_flip = tl.zeros((block_rows, block_width), tl.float32)

    for inner in tl.static_range(0, width):
        flipped_inner = width - 1 - inner
        j_selected = tl.load(
            matrix_j + selected * width + inner, mask=row_mask, other=0.0
        )
        j_flipped_selected = tl.load(
            matrix_j + flipped_selected * width + inner,
            mask=row_mask,
            other=0.0,
        )
        j_selected_flip = tl.load(
            matrix_j + selected * width + flipped_inner,
            mask=row_mask,
            other=0.0,
        )
        j_flipped_selected_flip = tl.load(
            matrix_j + flipped_selected * width + flipped_inner,
            mask=row_mask,
            other=0.0,
        )
        a = alpha_cos * j_selected + alpha_sin * j_flipped_selected
        a_flip = alpha_cos * j_selected_flip + alpha_sin * j_flipped_selected_flip
        a_alpha = alpha_frequency * (
            -alpha_sin * j_selected + alpha_cos * j_flipped_selected
        )
        a_flip_alpha = alpha_frequency * (
            -alpha_sin * j_selected_flip + alpha_cos * j_flipped_selected_flip
        )
        a_alpha_alpha = -(alpha_frequency * alpha_frequency) * a
        a_flip_alpha_alpha = -(alpha_frequency * alpha_frequency) * a_flip

        beta_frequency = degree - inner
        beta_phase = beta * beta_frequency
        beta_cos = tl.cos(beta_phase)
        beta_sin = tl.sin(beta_phase)
        b = a * beta_cos - a_flip * beta_sin
        b_alpha = a_alpha * beta_cos - a_flip_alpha * beta_sin
        b_beta = beta_frequency * (-a * beta_sin - a_flip * beta_cos)
        b_alpha_alpha = a_alpha_alpha * beta_cos - a_flip_alpha_alpha * beta_sin
        b_alpha_beta = beta_frequency * (-a_alpha * beta_sin - a_flip_alpha * beta_cos)
        b_beta_beta = -(beta_frequency * beta_frequency) * b

        j = tl.load(matrix_j + inner * width + column, mask=column_mask, other=0.0)
        j_flip = tl.load(
            matrix_j + inner * width + width - 1 - column,
            mask=column_mask,
            other=0.0,
        )
        c += b * j
        ca += b_alpha * j
        cb += b_beta * j
        caa += b_alpha_alpha * j
        cab += b_alpha_beta * j
        cbb += b_beta_beta * j
        c_flip += b * j_flip
        ca_flip += b_alpha * j_flip
        cb_flip += b_beta * j_flip
        caa_flip += b_alpha_alpha * j_flip
        cab_flip += b_alpha_beta * j_flip
        cbb_flip += b_beta_beta * j_flip

    gamma_frequency = degree - column
    gamma_phase = gamma * gamma_frequency
    gamma_cos = tl.cos(gamma_phase)
    gamma_sin = tl.sin(gamma_phase)
    value = c * gamma_cos - c_flip * gamma_sin
    d_alpha = ca * gamma_cos - ca_flip * gamma_sin
    d_beta = cb * gamma_cos - cb_flip * gamma_sin
    d_gamma = gamma_frequency * (-c * gamma_sin - c_flip * gamma_cos)
    d_alpha_alpha = caa * gamma_cos - caa_flip * gamma_sin
    d_alpha_beta = cab * gamma_cos - cab_flip * gamma_sin
    d_beta_beta = cbb * gamma_cos - cbb_flip * gamma_sin
    d_alpha_gamma = gamma_frequency * (-ca * gamma_sin - ca_flip * gamma_cos)
    d_beta_gamma = gamma_frequency * (-cb * gamma_sin - cb_flip * gamma_cos)
    d_gamma_gamma = -(gamma_frequency * gamma_frequency) * value
    return (
        value, d_alpha, d_beta, d_gamma,
        d_alpha_alpha, d_alpha_beta, d_alpha_gamma,
        d_beta_beta, d_beta_gamma, d_gamma_gamma,
    )


@triton.jit
def matrix_address(
    edge, row, column, row_count: tl.constexpr, width: tl.constexpr,
    matrix_stride: tl.constexpr = 0,
    matrix_offset: tl.constexpr = 0,
):
    stride: tl.constexpr = matrix_stride if matrix_stride else row_count * width
    return edge * stride + matrix_offset + row * width + column


@triton.jit
def reduce_angles(value):
    return tl.sum(tl.sum(value, axis=1), axis=0)


@lru_cache(maxsize=128)
def rotation_metadata(rows_by_degree):
    metadata = []
    for degree, rows in enumerate(rows_by_degree):
        width = 2 * degree + 1
        metadata.append(
            dict(
                degree=degree,
                selected_rows=rows,
                row_count=len(rows),
                width=width,
                block_rows=triton.next_power_of_2(len(rows)),
                block_width=triton.next_power_of_2(width),
                num_warps=4,
            )
        )
    return tuple(metadata)


@lru_cache(maxsize=128)
def retained_metadata(spec):
    rows = selected_local_rows(spec.lmax, spec.mmax)
    return tuple(
        {
            **metadata,
            "matrix_stride": spec.compact_elements,
            "matrix_offset": spec.compact_row_offsets[degree_rows[0]],
        }
        for metadata, degree_rows in zip(rotation_metadata(rows), spec.degree_rows, strict=True)
    )



@triton.jit
def retained_tile_degree(
    alpha, beta, gamma, matrix_j, scale, direct, inverse, gradients,
    edge, edge_mask, compact: tl.constexpr, degree: tl.constexpr,
    rows: tl.constexpr, row_count: tl.constexpr, width: tl.constexpr,
    block_rows: tl.constexpr, block_width: tl.constexpr, start: tl.constexpr,
    backward: tl.constexpr,
):
    row = tl.arange(0, block_rows)[:, None]
    col = tl.arange(0, block_width)[None, :]
    local = start + row * width + col
    retained = (row < row_count) & (col < width)
    mask = edge_mask[:, None, None] & retained[None, :, :]
    address = edge[:, None, None] * compact + local[None, :, :]
    terms = wigner_terms(
        alpha, beta, gamma, matrix_j, degree, rows,
        row_count, width, block_rows, block_width,
    )
    if scale is not None:
        factor = tl.load(scale + local, retained, 0)[None, :, :]
    if backward:
        cotangent = tl.load(gradients[0] + address, mask, 0)
        if scale is not None:
            cotangent += tl.load(gradients[1] + address, mask, 0) * factor
        return (
            tl.sum(tl.sum(cotangent * terms[1], axis=2), axis=1),
            tl.sum(tl.sum(cotangent * terms[2], axis=2), axis=1),
            tl.sum(tl.sum(cotangent * terms[3], axis=2), axis=1),
        )
    else:
        tl.store(direct + address, terms[0], mask)
        if scale is not None:
            tl.store(inverse + address, terms[0] * factor, mask)
        return 0.0, 0.0, 0.0


@triton.jit
def retained_tile_rows(degree: tl.constexpr, count: tl.constexpr):
    rows = ()
    for index in tl.static_range(count):
        order = (index + 1) // 2
        if index % 2:
            rows += (degree + order,)
        else:
            rows += (degree - order,)
    return rows


@triton.jit
def retained_tiles_kernel(
    angles, basis, scale, direct, inverse, gradients, angle_gradients, edges,
    compact: tl.constexpr, metadata: tl.constexpr, backward: tl.constexpr,
):
    edge = tl.program_id(0) * 4 + tl.arange(0, 4)
    edge_mask = edge < edges
    alpha = tl.load(angles[0] + edge, edge_mask, 0)[:, None, None]
    beta = tl.load(angles[1] + edge, edge_mask, 0)[:, None, None]
    gamma = tl.load(angles[2] + edge, edge_mask, 0)[:, None, None]
    if backward:
        da = tl.zeros((4,), tl.float32)
        db = tl.zeros((4,), tl.float32)
        dg = tl.zeros((4,), tl.float32)
    for degree in tl.static_range(len(metadata) // 5):
        da_degree, db_degree, dg_degree = retained_tile_degree(
            alpha, beta, gamma, basis[degree], scale, direct, inverse, gradients,
            edge, edge_mask, compact, degree,
            retained_tile_rows(degree, metadata[degree * 5]), metadata[degree * 5],
            metadata[degree * 5 + 1], metadata[degree * 5 + 2],
            metadata[degree * 5 + 3], metadata[degree * 5 + 4],
            backward,
        )
        if backward:
            da += da_degree
            db += db_degree
            dg += dg_degree
    if backward:
        tl.store(angle_gradients[0] + edge, da, edge_mask)
        tl.store(angle_gradients[1] + edge, db, edge_mask)
        tl.store(angle_gradients[2] + edge, dg, edge_mask)


@lru_cache(maxsize=128)
def retained_tile_metadata(spec):
    # A flat constexpr avoids nested tuple indexing in older Triton compilers.
    keys = ("row_count", "width", "block_rows", "block_width", "matrix_offset")
    return tuple(degree[key] for degree in retained_metadata(spec) for key in keys)


def supports_factored_retained_rotation(device, lmax):
    capability = torch.cuda.get_device_capability(device)
    return capability == (9, 0) or (capability == (8, 0) and lmax <= 5)


def use_retained_tiles(alpha, spec, *, backward=False):
    # Wide retained rows improve in forward, but cost too many registers in VJP.
    shape = (spec.lmax, spec.mmax)
    if shape not in ((2, 2), (4, 2), (6, 2)) and (backward or shape != (4, 4)):
        return False
    return (
        alpha.shape[0] >= 16384
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(alpha.device) in ((8, 0), (9, 0))
    )

def validate_inputs(alpha, beta, gamma, matrices_j, mmax):
    rows = selected_local_rows(len(matrices_j) - 1, mmax)
    if not all(
        angle.is_cuda and angle.dtype == torch.float32 for angle in (alpha, beta, gamma)
    ):
        raise ValueError("sparse Wigner requires CUDA FP32 angles")
    if not all(
        angle.ndim == 1
        and angle.shape == alpha.shape
        and angle.device == alpha.device
        and angle.is_contiguous()
        for angle in (alpha, beta, gamma)
    ):
        raise ValueError(
            "Wigner angles must be contiguous matching vectors on one device"
        )
    for degree, matrix in enumerate(matrices_j):
        if (
            matrix.shape != (2 * degree + 1, 2 * degree + 1)
            or matrix.device != alpha.device
            or matrix.dtype != alpha.dtype
            or not matrix.is_contiguous()
            or matrix.requires_grad
        ):
            raise ValueError(
                "J matrices must be fixed contiguous FP32 degree blocks on the angle device"
            )
    return rows


def real_column(m):
    if m == 0:
        return ((0, 1),)
    order = abs(m)
    sign = (-1) ** order
    if m > 0:
        return ((-order, 1), (order, sign))
    return ((-order, -1j), (order, 1j * sign))


def rotation_integer(degree, output, source):
    # exp(pi/2 * X) acts on the real fundamental spinor as [[1,1],[-1,1]]/sqrt(2).
    size = 2 * degree
    k, u = output + degree, source + degree
    return sum((-1) ** t * comb(size - u, t) * comb(u, k - t)
               for t in range(max(0, k - u), min(k, size - u) + 1))


def canonical_j(degree):
    rows = []
    for row in range(-degree, degree + 1):
        data = []
        for col in range(-degree, degree + 1):
            total = sum(
                q_row.conjugate() * (1j ** ((k + u) % 4)) * rotation_integer(degree, k, u) * q_col
                for k, q_row in real_column(row) for u, q_col in real_column(col)
            )
            if total.imag != 0:
                raise AssertionError((degree, row, col, total))
            radical = comb(2 * degree, degree + col) / comb(2 * degree, degree + row)
            radical *= 0.5 ** ((row != 0) + (col != 0))
            data.append((-1) ** degree * total.real * sqrt(radical) / (2 ** degree))
        rows.append(data)
    return torch.tensor(rows, dtype=torch.float64, device="cpu")



def row_index(degree, compact_row):
    if compact_row % 2:
        return degree + (compact_row + 1) // 2
    return degree - compact_row // 2


@lru_cache(maxsize=1)
def canonical_l6_fourier_tables():
    """Use the analytic canonical basis only for this explicit default path."""
    basis = tuple(canonical_j(degree).float() for degree in range(7))
    groups = {(degree, size): []
              for degree in range(7) for size in (4, 2, 1)}
    offsets = (0, 1, 10, 35, 70, 115, 170)
    for degree in range(7):
        width = 2 * degree + 1
        for compact_row in range(0, 2 * min(degree, 2) + 1):
            if compact_row and compact_row % 2 == 0:
                continue
            row_group = (compact_row, compact_row + 1) if compact_row else (0,)
            for col in range(degree + 1):
                col_group = (col, 2 * degree - col) if col != degree else (col,)
                group = tuple((degree, r, c, offsets[degree] + r * width + c)
                              for r in row_group for c in col_group)
                groups[degree, len(group)].append(
                    (len(row_group), len(col_group), group)
                )

    metadata = torch.full((256,), 255, dtype=torch.int32, device="cpu")
    harmonics = torch.zeros((4, 256), dtype=torch.int32, device="cpu")
    values = torch.zeros((4, 256), dtype=torch.float32, device="cpu")
    lane = 0
    for (_, size), orbits in groups.items():
        if orbits:
            lane = (lane + size - 1) // size * size
        for rows, cols, group in orbits:
            for degree, row, col, compact in group:
                row = row_index(degree, row)
                j = basis[degree]
                data = [j[row, degree] * j[degree, col]]
                for order in range(1, degree + 1):
                    rp, rn = j[row, degree - order], j[row, degree + order]
                    cp, cn = j[degree - order, col], j[degree + order, col]
                    data.extend((rp * cn - rn * cp, rp * cp + rn * cn))
                data = torch.stack(data)
                nz = data.nonzero().flatten()
                if nz.numel() > 4:
                    raise ValueError("analytic basis has more than four exact harmonics per coefficient")
                harmonics[:nz.numel(), lane] = nz
                values[:nz.numel(), lane] = data[nz]
                reflected_row = cols if rows == 2 else 0
                reflected_col = 1 if cols == 2 else 0
                alpha_order = degree - row
                gamma_order = degree - col
                metadata[lane] = (compact | (reflected_row << 8) | (reflected_col << 10)
                                  | ((alpha_order + 6) << 12) | ((gamma_order + 6) << 16))
                lane += 1
    assert sorted((metadata[metadata & 255 < 235] & 255).tolist()) == list(range(235))
    return metadata, harmonics, values



@lru_cache(maxsize=16)
def canonical_l6_fourier(device):
    # Static mathematical constants, independent of all user-provided tensors.
    with torch.inference_mode(False), torch.no_grad():
        return tuple(value.to(device) for value in canonical_l6_fourier_tables())
