"""Dense PyTorch definitions of retained rotation storage and Euler Wigner maps.

These functions use ordinary tensor operations on CPU or CUDA, in the input
dtype. They make the stored rows explicit and remain differentiable. The
optimized generators evaluate only the entries retained here.
"""

import math

import torch
import torch.nn.functional as F


POLE_TOLERANCE = 0.9999989867210388  # 0.999999 rounded to FP32 in CUDA kernels.


def retained_row_indices(lmax, mmax):
    """List (degree, order) in the feature layout, independently of kernels."""
    if lmax < 0 or not 0 <= mmax <= lmax:
        raise ValueError("rotation degrees require 0 <= mmax <= lmax")
    return [(degree, 0) for degree in range(lmax + 1)] + [
        (degree, sign * order)
        for order in range(1, mmax + 1)
        for sign in (1, -1)
        for degree in range(order, lmax + 1)
    ]


def pack_retained_rotations(rotations, *, lmax, mmax, input_layout="full"):
    """Select degree-local rows from a dense forward rotation.

    ``rotations`` is ``(E, d, d)`` for ``input_layout='full'`` or
    ``(E, R, d)`` for ``'reduced_rows'``, with ``d=(L+1)**2`` and
    ``R=sum(2*min(l,M)+1 for l in range(L+1))``. Reduced rows follow
    ``m=0,+1,-1,...``, with ascending degree within each order. The result
    is ``(E, K)`` with ``K=sum((2*l+1)*(2*min(l,M)+1) for l in range(L+1))``.

    The packed order is degree first, then ``m=0,+1,-1,...``, then the
    ``2*l+1`` full columns. Entries that couple different degrees are not
    stored. The supplied matrix must represent independent degree blocks.
    Slicing and concatenation preserve first and second derivatives.
    """
    rows = retained_row_indices(lmax, mmax)
    reduced_index = {pair: index for index, pair in enumerate(rows)}
    if input_layout not in ("full", "reduced_rows"):
        raise ValueError("input_layout must be 'full' or 'reduced_rows'")
    pieces = []
    for degree in range(lmax + 1):
        orders = [
            0,
            *(sign * m for m in range(1, min(degree, mmax) + 1) for sign in (1, -1)),
        ]
        for order in orders:
            row = (
                degree * degree + degree + order
                if input_layout == "full"
                else reduced_index[degree, order]
            )
            pieces.append(rotations[:, row, degree * degree : (degree + 1) ** 2])
    return torch.cat(pieces, dim=1)


def unpack_retained_rotations(rotations, *, lmax, mmax):
    """Expand ``(E, K)`` storage to forward rows ``(E, R, d)``.

    The output rows follow ``m=0,+1,-1,...`` and ascending degree.
    Each row has zeros outside its degree block. This recovers only the
    retained rows, not a complete inverse of the packing operation.
    With local coefficients ``v`` in ``(E,R,C)``, forward rotation is
    ``rows @ x`` and inverse rotation is ``rows.transpose(1,2) @ v``.
    For a separately packed inverse carrier, that transpose recovers the
    supplied inverse entries, including any convention-dependent scales.
    """
    rows = retained_row_indices(lmax, mmax)
    full_width = (lmax + 1) ** 2
    expected = sum(
        (2 * degree + 1) * (2 * min(degree, mmax) + 1) for degree in range(lmax + 1)
    )
    if rotations.ndim != 2 or rotations.shape[1] != expected:
        raise ValueError(f"rotations must have shape (E, {expected})")
    expanded = {}
    offset = 0
    for degree in range(lmax + 1):
        width = 2 * degree + 1
        for order in [
            0,
            *(sign * m for m in range(1, min(degree, mmax) + 1) for sign in (1, -1)),
        ]:
            values = rotations[:, offset : offset + width]
            expanded[degree, order] = F.pad(
                values, (degree * degree, full_width - (degree + 1) ** 2)
            )
            offset += width
    return torch.stack([expanded[pair] for pair in rows], dim=1)


def pack_inverse_rotations(rotations, *, lmax, mmax):
    """Select ``(E,d,R)`` inverse entries into the same ``(E,K)`` storage.

    Columns follow the reduced feature order of :func:`pack_retained_rotations`.
    Only within-degree entries are retained. This function packs entries,
    without computing an inverse or adding a normalization factor.
    """
    return pack_retained_rotations(
        rotations.transpose(1, 2), lmax=lmax, mmax=mmax, input_layout="reduced_rows"
    )


def compact_inverse_rescale(inverse_rescale, *, lmax, mmax):
    """Select a ``(1,d,R)`` inverse scale into a differentiable ``(K,)`` vector."""
    return pack_inverse_rotations(inverse_rescale, lmax=lmax, mmax=mmax)[0]


def rotation_angles(frames, *, angle_convention):
    """Extract Euler vectors from ``(E,3,3)`` frames with masked pole gradients.

    The second column supplies the edge direction. Both supported conventions
    freeze angle derivatives at ``abs(frames[:,1,1]) >= 0.999999`` (FP32). The beta
    value is clamped to 0 or pi only beyond that threshold. At equality its
    original value is retained. ``'masked'`` normalizes direction components
    before computing alpha, matching the corresponding generator.

    The safe frame keeps unused derivative branches finite at exact poles.
    All other frame entries remain differentiable through double backward.
    """
    if angle_convention not in ("pole_clamped", "masked"):
        raise ValueError("angle_convention must be 'pole_clamped' or 'masked'")

    def angles(value):
        direction = value[:, :, 1]
        normalized = F.normalize(direction, dim=1).clamp(-1, 1)
        x, _, z = (normalized if angle_convention == "masked" else direction).unbind(1)
        alpha = torch.atan2(x, z)
        beta = torch.acos(normalized[:, 1])
        gamma = torch.atan2(
            alpha.cos() * value[:, 0, 2] - alpha.sin() * value[:, 2, 2],
            alpha.cos() * value[:, 0, 0] - alpha.sin() * value[:, 2, 0],
        )
        return alpha, beta, gamma

    y = frames[:, 1, 1]
    regular = (y > -POLE_TOLERANCE) & (y < POLE_TOLERANCE)
    safe_frame = frames.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    live = angles(torch.where(regular[:, None, None], frames, safe_frame))
    alpha, beta, gamma = angles(frames.detach())
    beta = torch.where(
        y > POLE_TOLERANCE, 0, torch.where(y < -POLE_TOLERANCE, math.pi, beta)
    )
    return tuple(
        torch.where(regular, a, b.detach()) for a, b in zip(live, (alpha, beta, gamma))
    )


def z_rotation(angle, degree):
    frequencies = torch.arange(
        degree, -degree - 1, -1, device=angle.device, dtype=angle.dtype
    )
    phase = angle[:, None] * frequencies
    identity = torch.eye(2 * degree + 1, device=angle.device, dtype=angle.dtype)
    return phase.cos()[:, :, None] * identity + phase.sin()[:, :, None] * identity.flip(
        1
    )


def sparse_wigner(alpha, beta, gamma, matrices_j, *, mmax, storage="blocks"):
    """Form full ``Z(alpha) @ J @ Z(beta) @ J @ Z(gamma)`` degree blocks.

    Angles are three ``(E,)`` vectors. ``matrices_j[l]`` is a fixed real-basis
    ``(2*l+1,2*l+1)`` matrix. ``storage='blocks'`` returns selected rows of each
    degree as a tuple, with local order ``0,+1,-1,...``. ``'retained_rows'``
    concatenates those blocks into ``(E,K)``. The reference materializes the
    full degree matrices that the optimized generator avoids. Derivatives
    propagate to every angle and to incoming cotangents through double backward.
    """
    retained_row_indices(len(matrices_j) - 1, mmax)
    if storage not in ("blocks", "retained_rows"):
        raise ValueError("storage must be 'blocks' or 'retained_rows'")
    blocks = []
    for degree, matrix_j in enumerate(matrices_j):
        full = (
            z_rotation(alpha, degree)
            @ matrix_j
            @ z_rotation(beta, degree)
            @ matrix_j
            @ z_rotation(gamma, degree)
        )
        rows = [
            degree,
            *(
                degree + sign * m
                for m in range(1, min(degree, mmax) + 1)
                for sign in (1, -1)
            ),
        ]
        blocks.append(full[:, rows])
    return (
        torch.cat([block.flatten(1) for block in blocks], dim=1)
        if storage == "retained_rows"
        else tuple(blocks)
    )


def sparse_wigner_pair(alpha, beta, gamma, matrices_j, inverse_rescale, *, mmax):
    """Return direct ``(E,K)`` rows and their elementwise scaled inverse carrier."""
    direct = sparse_wigner(
        alpha, beta, gamma, matrices_j, mmax=mmax, storage="retained_rows"
    )
    return direct, direct * inverse_rescale


def retained_wigner_from_frames(
    frames, matrices_j, *, mmax, angle_convention, inverse_rescale=None
):
    """Dense PyTorch definition of the frame-to-retained-rotation operation.

    Frames have shape ``(E,3,3)`` and basis matrices span degrees 0 through L.
    Return ``(E,K)`` direct rows, or ``(direct, scaled_inverse)`` when a fixed
    ``(K,)`` inverse scale is provided. Frame construction and basis selection
    remain with the caller. Pole derivatives follow :func:`rotation_angles`.
    """
    angles = rotation_angles(frames, angle_convention=angle_convention)
    direct = sparse_wigner(*angles, matrices_j, mmax=mmax, storage="retained_rows")
    return direct if inverse_rescale is None else (direct, direct * inverse_rescale)


def fused_edge_quaternion(edge_vectors, gamma, *, compile_visible=False):
    """Map ``(E,3)`` vectors and fixed ``(E,)`` angles to ``(E,4)`` quaternions.

    This is the complete UMA chart-blending formula. Quaternion components
    use ``(w,x,y,z)`` order. The frame angle is supplied by the caller and
    held fixed during geometry differentiation. The FP32 chart epsilon is
    part of this convention even when evaluating the reference in FP64.
    ``compile_visible`` affects only optimized launches and is unused here.
    """
    if gamma.requires_grad:
        raise ValueError("gamma is a fixed local-frame angle")
    eps = torch.finfo(torch.float32).eps

    def normalize(value, epsilon):
        # Clamp before sqrt so an unused zero chart has finite second derivatives.
        return value * value.square().sum(1, keepdim=True).clamp_min(epsilon**2).rsqrt()

    x, y, z = normalize(edge_vectors, 1e-12).unbind(1)
    zero = torch.zeros_like(x)
    chart1 = normalize(torch.stack((1 + y, -z, zero, x), dim=1), math.sqrt(eps))
    chart2 = normalize(torch.stack((-z, 1 - y, x, zero), dim=1), math.sqrt(eps))
    t = ((y + 0.9) / 1.8).clamp(0, 1)
    blend = torch.sigmoid((2 * t - 1) / (t * (1 - t)).clamp_min(eps))
    blend = torch.where(t < eps, 0, torch.where(t > 1 - eps, 1, blend))[:, None]
    chart2 = torch.where((chart1 * chart2).sum(1, keepdim=True) < 0, -chart2, chart2)
    q = normalize((1 - blend) * chart2 + blend * chart1, 1e-12)
    w, x, y, z = q.unbind(1)
    cosine, sine = (gamma / 2).cos(), (gamma / 2).sin()
    return torch.stack(
        (
            cosine * w - sine * y,
            cosine * x + sine * z,
            cosine * y + sine * w,
            cosine * z - sine * x,
        ),
        dim=1,
    )


def full_rows(lmax, mmax):
    """Full coefficient indices, grouped by local order and increasing degree."""
    rows = [[degree * degree + degree for degree in range(lmax + 1)]]
    rows += [
        [
            degree * degree + degree + sign * order
            for sign in (1, -1)
            for degree in range(order, lmax + 1)
        ]
        for order in range(1, mmax + 1)
    ]
    return rows


def dense_rotation_rows(rotations, lmax, mmax):
    """Select retained rows of the within-degree rotation blocks.

    Rotations do not couple different degrees. Masking those entries also
    makes their gradients zero, matching the compact representation.
    """
    full = (lmax + 1) ** 2
    blocks = [
        F.pad(
            rotations[:, degree**2 : (degree + 1) ** 2, degree**2 : (degree + 1) ** 2],
            (degree**2, full - (degree + 1) ** 2),
        )
        for degree in range(lmax + 1)
    ]
    block_diagonal = torch.cat(blocks, dim=1)
    rows = [row for group in full_rows(lmax, mmax) for row in group]
    return block_diagonal[:, rows]


def rotation(geometry, basis=None, *, lmax=None, mmax, convention=None, inverse_scale=None):
    """Materialized PyTorch equations for ``rotation`` in the input dtype."""
    if basis is None:
        if lmax is None:
            raise ValueError("rotation requires lmax or explicit basis tables")
        from LunaEquivariance.so2.rotation.kernels.utils import standard_basis

        basis = standard_basis(lmax, device=geometry.device, dtype=geometry.dtype)
    elif lmax is not None and lmax != len(basis) - 1:
        raise ValueError("lmax must match the supplied basis tables")
    return retained_wigner_from_frames(
        geometry, tuple(basis), mmax=mmax,
        angle_convention="pole_clamped" if convention is None else convention,
        inverse_rescale=inverse_scale,
    )


def quaternion_wigner_from_edges_retained(
    geometry, quadratic, quartic, *, lmax, mmax, polynomial_blocks=(),
):
    """Reference for UMA edge vectors and preselected polynomial tables."""
    from LunaEquivariance.so2.rotation.kernels.utils import (
        QUADRATIC_EXPONENTS,
        QUARTIC_EXPONENTS,
    )

    gamma = (torch.rand(geometry.shape[0], device=geometry.device, dtype=torch.float32) * (2 * torch.pi)).to(geometry.dtype)
    q = fused_edge_quaternion(geometry, gamma)

    def monomials(powers):
        powers = torch.as_tensor(powers, dtype=torch.int64, device=q.device)
        return (q[:, None, :] ** powers[None, :, :]).prod(-1)

    scalar = q[:, :1].square() * 0 + 1
    values = [scalar]
    if lmax >= 1:
        values.append(monomials(QUADRATIC_EXPONENTS) @ quadratic.T)
    if lmax >= 2:
        values.append(monomials(QUARTIC_EXPONENTS) @ quartic.T)
    for block in polynomial_blocks:
        values.append(monomials(block.exponents) @ block.coefficients.T)
    return torch.cat(values, dim=1)
