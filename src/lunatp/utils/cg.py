"""Tensor-product coefficients, weight addresses and CUDA expressions."""

from __future__ import annotations

import functools
from fractions import Fraction
from math import factorial
from typing import Union

import torch

from ...utils.precision import float_literal
from .layout import CGTerm, UvuPathSpec

CG_COEFFICIENT_THRESHOLD = 1.0e-12

CG_FEATURE_LOAD = r"""{scalar_type} {name}v{component}=__ldg({name}+int64_t({row})*{width}+{offset}+{channel}*{components}+{component});"""

CG_SCALAR_LOAD = r"""{scalar_type} {name}v=__ldg({name}+int64_t(item));"""

CG_SCALED_EDGE = r"""({edge}v{component}*{scale}v)"""

# Keep the caller's association: multiplying the coefficient into X before
# the FMA is distinct from multiplying X/Y first. The edge factor can include
# a per-edge scalar, such as a convolution envelope or its direction.
CG_COEFFICIENT_X = r"""{accumulator}{output}=fma{suffix}({coefficient}*{feature}v{i},{edge_factor},{accumulator}{output});"""

CG_X_EDGE = r"""{accumulator}{output}=fma{suffix}({coefficient},{feature}v{i}*{edge_factor},{accumulator}{output});"""


def build_cg_terms(cg: torch.Tensor, path_weight: float) -> list[CGTerm]:
    """Extract sparse coefficients in the original component order."""
    terms = []
    d1, d2, d3 = cg.shape
    for i_idx in range(d1):
        for j_idx in range(d2):
            for k_idx in range(d3):
                coeff = cg[i_idx, j_idx, k_idx].item() * path_weight
                if abs(coeff) > CG_COEFFICIENT_THRESHOLD:
                    terms.append((i_idx, j_idx, k_idx, coeff))
    return terms


def weight_channel_offset(spec: UvuPathSpec, channel: str = "u") -> str:
    """Address one input channel in e3nn's [input, edge] weight layout."""
    stride = int(spec.get("weight_stride", 1))
    return channel if stride == 1 else f"{channel} * {stride}"


def emit_cg_source(
    meta, spec, pairs=(("x", "y"),), *, x_row="item", coefficient_first=True,
    accumulator="z",
):
    """Emit sparse CG products with the caller's multiplication grouping.

    x_row selects the node expression at code generation.
    """
    terms = spec["terms"]
    dtype = meta.get("dtype", torch.float32)
    scalar_type = "double" if dtype == torch.float64 else "float"
    suffix = "" if dtype == torch.float64 else "f"
    lines = []
    for name in sorted({pair[0] for pair in pairs}):
        for i in sorted({t[0] for t in terms}):
            lines.append(
                CG_FEATURE_LOAD.format(
                    scalar_type=scalar_type, name=name, component=i, row=x_row,
                    width=meta['in1_dim'], offset=spec['x_offset'], channel='u',
                    components=spec['d1'],
                )
            )
    for name in sorted({pair[1] for pair in pairs}):
        for j in sorted({t[1] for t in terms}):
            lines.append(
                CG_FEATURE_LOAD.format(
                    scalar_type=scalar_type, name=name, component=j, row='item',
                    width=meta['in2_dim'], offset=spec['y_offset'], channel='v',
                    components=spec['d2'],
                )
            )
    scalars = sorted({pair[2] for pair in pairs if len(pair) == 3 and pair[2] is not None})
    for name in scalars:
        lines.append(CG_SCALAR_LOAD.format(scalar_type=scalar_type, name=name))
    for i, j, k, value in terms:
        for pair in pairs:
            x, y = pair[:2]
            scalar = pair[2] if len(pair) == 3 else None
            y_value = (CG_SCALED_EDGE.format(edge=y, component=j, scale=scalar)
                       if scalar is not None else f"{y}v{j}")
            source = CG_COEFFICIENT_X if coefficient_first else CG_X_EDGE
            lines.append(source.format(
                accumulator=accumulator, output=k, suffix=suffix,
                coefficient=float_literal(value, dtype), feature=x, i=i,
                edge_factor=y_value,
            ))
    return "\n".join(lines)


# Coefficient arithmetic adapted from e3nn 0.6.0; operation order is retained.
# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html
@torch.compiler.disable
def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2 ** 0.5
        q[l + m, l - abs(m)] = -1j / 2 ** 0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2 ** 0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2 ** 0.5
    q = (-1j) ** l * q
    if dtype is None:
        dtype = torch.empty(0).dtype
    if device is None:
        device = torch.empty(0).device
    dtype = {torch.float32: torch.complex64, torch.float64: torch.complex128}[dtype]
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


def wigner_3j(l1: int, l2: int, l3: int, dtype=None, device=None) -> torch.Tensor:
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)
    C = so3_clebsch_gordan(l1, l2, l3)
    if dtype is None:
        dtype = torch.empty(0).dtype
    if device is None:
        device = torch.empty(0).device
    return C.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


@functools.lru_cache(maxsize=None)
def so3_clebsch_gordan(l1: int, l2: int, l3: int) -> torch.Tensor:
    Q1 = change_basis_real_to_complex(l1, dtype=torch.float64)
    Q2 = change_basis_real_to_complex(l2, dtype=torch.float64)
    Q3 = change_basis_real_to_complex(l3, dtype=torch.float64)
    C = su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, torch.conj(Q3.T), C)
    assert torch.all(torch.abs(torch.imag(C)) < 1e-05)
    C = torch.real(C)
    C = C / torch.norm(C)
    return C


def integer_factorial(n: int) -> int:
    assert n == round(n)
    return factorial(round(n))


def su2_clebsch_gordan(
    j1: Union[int, float], j2: Union[int, float], j3: Union[int, float]
) -> torch.Tensor:
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = torch.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype=torch.float64)
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:

                    m3 = m1 + m2
                    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
                    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

                    C = (
                        (2.0 * j3 + 1.0)
                        * Fraction(
                            integer_factorial(j3 + j1 - j2) * integer_factorial(j3 - j1 + j2) * integer_factorial(j1 + j2 - j3) * integer_factorial(j3 + m3) * integer_factorial(j3 - m3),
                            integer_factorial(j1 + j2 + j3 + 1) * integer_factorial(j1 - m1) * integer_factorial(j1 + m1) * integer_factorial(j2 - m2) * integer_factorial(j2 + m2),
                        )
                    ) ** 0.5
                    S = 0
                    for v in range(vmin, vmax + 1):
                        S += (-1) ** int(v + j2 + m2) * Fraction(
                            integer_factorial(j2 + j3 + m1 - v) * integer_factorial(j1 - m1 + v),
                            integer_factorial(v) * integer_factorial(j3 - j1 + j2 - v) * integer_factorial(j3 + m3 - v) * integer_factorial(v + j1 - j2 - m3),
                        )
                    C = C * S
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m3)] = C
    return mat

