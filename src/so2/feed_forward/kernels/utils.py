"""Shared layout, indexing, and launch helpers."""

from __future__ import annotations

from itertools import product
from math import isqrt, pi, sqrt

import torch
import triton
import triton.language as tl

from ...utils.tensor_contracts import require_cuda_fp32


# EqV3 stores each degree in increasing signed order. On a uniform longitude
# grid, only triples whose signed frequencies sum to zero survive integration.
GRID_MODES = tuple(range(-4, 5))
GRID_ROW_MODE = tuple(row - isqrt(row) * (isqrt(row) + 1) + 4 for row in range(25))
GRID_MODE_ROWS = tuple(tuple(row for row, mode in enumerate(GRID_ROW_MODE) if mode == k) for k in range(9))


def real_fourier_mode(mode):
    if mode == 0:
        return ((0, 1.),)
    amplitude = 1. / sqrt(2.)
    if mode > 0:
        return ((mode, amplitude), (-mode, amplitude))
    return ((-mode, -1j * amplitude), (mode, 1j * amplitude))


def longitude_product_terms():
    harmonics = tuple(real_fourier_mode(mode) for mode in GRID_MODES)
    terms = []
    for output in harmonics:
        row = []
        for i, j in product(range(len(harmonics)), repeat=2):
            coefficient = sum(a * b * c for (m, a), (n, b), (p, c)
                              in product(output, harmonics[i], harmonics[j]) if m + n + p == 0)
            if coefficient:
                # Fourteen samples integrate every triple of orders up to four.
                row.append((i, j, 14. * coefficient.real))
        terms.append(tuple(row))
    return tuple(terms)


GRID_LONGITUDE_TERMS = longitude_product_terms()


@triton.jit
def grid_latitude_modes(coefficients, latitude, mode_rows: tl.constexpr, block: tl.constexpr):
    spectral = ()
    for mode in tl.static_range(len(mode_rows)):
        value = tl.zeros((block,), tl.float32)
        for m in tl.static_range(len(mode_rows[mode])):
            factor = tl.load(latitude + mode_rows[mode][m])
            value = tl.fma(factor, coefficients[mode_rows[mode][m]], value)
        spectral += (value,)
    return spectral


@triton.jit
def grid_longitude_product(a, b, terms: tl.constexpr, block: tl.constexpr):
    value = ()
    for mode in tl.static_range(len(terms)):
        acc = tl.zeros((block,), tl.float32)
        for term in tl.static_range(len(terms[mode])):
            scale = tl.full((), terms[mode][term][2], tl.float32)
            acc = tl.fma(a[terms[mode][term][0]], b[terms[mode][term][1]] * scale, acc)
        value += (acc,)
    return value


def compact_grid_factors(alpha, row_mode):
    """Build seven-latitude factors for the canonical L=4 EqV3 node grid."""
    from e3nn.o3 import Legendre
    from numpy.polynomial.legendre import leggauss

    # Longitude integration eliminates triples with an odd sum of absolute
    # orders. The remaining latitude integrands have degree at most 12.
    nodes, weights = leggauss(7)
    z = torch.from_numpy(nodes)
    quadrature = torch.from_numpy(weights) / (2 * alpha.shape[0])
    latitude = Legendre(list(range(5)))(z, (1 - z.square()).sqrt())
    norm = z.new_tensor([sqrt(5 * (2 * isqrt(row) + 1)) for row in range(25)])
    to_grid = latitude * (sqrt(4 * pi) / norm)
    from_grid = latitude * (sqrt(4 * pi) * norm) * quadrature[:, None]
    return alpha, to_grid.to(alpha).contiguous(), from_grid.to(alpha).contiguous(), row_mode


def activation_layout(hidden, gating, hidden_bias, gating_bias, lmax):
    rows, nodes, channels = hidden.shape
    expected_gating = (nodes, lmax * channels)
    if gating.ndim == 3:
        gating = gating.view(nodes, lmax * channels)
    if tuple(gating.shape) != expected_gating:
        raise ValueError(f"unexpected gating shape: {tuple(gating.shape)}")
    if tuple(hidden_bias.shape) != (channels,):
        raise ValueError("unexpected hidden bias shape")
    if tuple(gating_bias.shape) != (lmax * channels,):
        raise ValueError("unexpected gating bias shape")
    if rows != (lmax + 1) ** 2:
        raise ValueError("unexpected hidden row count")
    row_degrees = tuple(
        degree for degree in range(lmax + 1) for _ in range(2 * degree + 1)
    )
    return gating, rows, nodes, channels, row_degrees


def validate(pair: torch.Tensor, gate: torch.Tensor) -> tuple[int, int, int]:
    require_cuda_fp32(pair, gate, operation="grid product", contiguous=True)
    if pair.ndim != 3 or gate.ndim != 3 or gate.shape[1] != 1:
        raise ValueError(f"unexpected grid-product ranks: {pair.shape=}, {gate.shape=}")
    grids, nodes, doubled_channels = pair.shape
    channels = doubled_channels // 2
    if (
        channels <= 0
        or doubled_channels != 2 * channels
        or gate.shape != (nodes, 1, channels)
    ):
        raise ValueError(
            f"incompatible grid-product shapes: {pair.shape=}, {gate.shape=}"
        )
    return nodes, grids, channels


def validate_epilogue(raw, bias, residual):
    rows, nodes, channels = raw.shape
    if tuple(residual.shape) != (nodes, rows, channels):
        raise ValueError("unexpected FFN residual shape")
    if tuple(bias.shape) != (channels,):
        raise ValueError("unexpected FFN output bias shape")
    if not raw.is_contiguous() or not residual.is_contiguous():
        raise ValueError("FFN epilogue requires contiguous inputs")
    return rows, nodes, channels
