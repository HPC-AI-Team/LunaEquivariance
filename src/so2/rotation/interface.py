"""Frame-to-Wigner preparation with explicit basis and pole conventions."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op, wrap_triton

from ..utils.layout import RetainedRotationLayout, get_angular_layout, get_retained_layout
from ..utils.tensor_contracts import require_int32_offsets
from .kernels.backward import (
    backward_pair,
    fused_rotation_vjp_kernel,
    edge_quaternion_backward,
    edge_quaternion_backward_kernel,
    masked_angles_backward,
    monomial_vjp_kernel,
    pole_clamped_angles_backward,
    retained_backward,
    retained_backward_kernel,
    retained_rows_backward,
    wigner_backward,
)
from .kernels.double_backward import (
    edge_quaternion_hvp,
    edge_quaternion_hvp_kernel,
    default_retained_hvp,
    frame_paired_retained_hvp,
    masked_angles_hvp,
    monomial_direction_kernel,
    monomial_q_hvp_kernel,
    pole_clamped_angles_hvp,
    retained_hvp,
    retained_hvp_kernel,
    retained_rows_hvp,
    wigner_hvp,
)
from .kernels.forward import (
    forward_pair,
    fused_rotation_kernel,
    edge_quaternion_forward,
    edge_quaternion_forward_kernel,
    masked_angles_forward,
    monomial_forward_kernel,
    pole_clamped_angles_forward,
    retained_forward,
    retained_forward_kernel,
    retained_rows_forward,
    wigner_forward,
)
from .kernels.triple_backward import AnglesDoubleBackward, RetainedWignerDoubleBackward
from .kernels.utils import (
    PolynomialBlock,
    QUADRATIC_COUNT_TL,
    QUARTIC_COUNT_TL,
    ROTATION_MASK_THRESHOLD,
    block_monomials,
    compress_l2_coefficients,
    launch_constants,
    monomial_exponents,
    polynomial_wigner_coefficients,
    quadratic_coefficients,
    register_sparse_quaternion_basis,
    select_retained_coefficients,
    selected_local_rows,
    validate_inputs,
    validate_polynomial_blocks,
)


def pack_retained_rotations(rotations, *, lmax, mmax, input_layout="full"):
    """Select retained rows within each degree and pack them into ``(E, K)``.

    The input layout determines the row order, where ``F=(L+1)^2``:

    - ``"full"``: ``(E, F, F)``, with degree-major rows ``l*l+l+m``.
    - ``"reduced_rows"``: ``(E, R, F)``, with orders ``0,+1,-1,...`` and
      ascending degree within each order, as in EquiformerV3.

    These layouts have the same shape at M=L but different row orders.
    Rotations must be block diagonal and use the features' real-harmonic
    basis. Cross-degree entries are discarded. The contiguous output uses
    the offsets in ``get_retained_layout`` and supports first and second
    derivatives. It can be reused across layers for the same geometry.
    """
    layout = get_angular_layout(lmax=lmax, mmax=mmax, channels=1)
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    if input_layout == "full":
        row_indices = layout.dense_rows
        rows = spec.full_components
    elif input_layout == "reduced_rows":
        row_indices = tuple(range(spec.reduced_components))
        rows = spec.reduced_components
    else:
        raise ValueError("input_layout must be 'full' or 'reduced_rows'")
    if rotations.ndim != 3 or rotations.shape[1:] != (rows, spec.full_components):
        raise ValueError(f"rotations must have shape (E, {rows}, {spec.full_components})")
    return torch.cat(
        tuple(
            rotations[:, row_indices[row], degree * degree : (degree + 1) ** 2]
            for degree, rows in enumerate(spec.degree_rows)
            for row in rows
        ),
        dim=1,
    ).contiguous()


def pack_inverse_rotations(rotations, *, lmax, mmax):
    """Pack rectangular inverse rotations ``(E, F, R)`` into ``(E, K)``.

    Full rows use degree-major order. Reduced columns use orders
    ``0,+1,-1,...`` with ascending degree, as in EquiformerV3.
    Each reduced column contributes its within-degree entries, including
    any inverse scale. This packs an existing inverse map and does not
    compute a matrix inverse. The output supports first and second
    derivatives and can be reused across layers for the same geometry.
    """
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    if rotations.ndim != 3 or rotations.shape[1:] != (
        spec.full_components,
        spec.reduced_components,
    ):
        raise ValueError(
            "inverse rotations must have shape (E, full_components, reduced_components)"
        )
    return torch.cat(
        tuple(
            rotations[:, degree * degree : (degree + 1) ** 2, row]
            for degree, rows in enumerate(spec.degree_rows)
            for row in rows
        ),
        dim=1,
    ).contiguous()


def retained_wigner_from_frames(
    frames, matrices_j, *, mmax, angle_convention, inverse_rescale=None, canonical_d2=False,
):
    """Build retained ``(E, K)`` rotations from CUDA FP32 ``(E, 3, 3)`` frames.

    ``matrices_j`` contains one contiguous CUDA FP32 real-basis matrix of
    shape ``(2*l+1, 2*l+1)`` per degree 0..L. The caller constructs the
    frames and chooses the vector orientation and harmonic basis.

    ``angle_convention`` selects the Euler-angle and pole convention:
    ``"pole_clamped"`` follows eSEN and ``"masked"`` follows EquiformerV3.
    Both support up to three derivatives with respect to frames and output
    cotangents. The basis matrices and inverse scale are constants.

    Return retained forward rows, which inverse consumers can transpose.
    If ``inverse_rescale`` is a contiguous CUDA FP32 ``(K,)`` tensor,
    return ``(direct, scaled_inverse)`` instead. Storage offsets follow
    ``get_retained_layout``. No dense Wigner matrix is built.
    """
    if angle_convention == "pole_clamped":
        fused_rotation_angles = pole_clamped_angles_fused_rotation_angles
    elif angle_convention == "masked":
        fused_rotation_angles = masked_angles_fused_rotation_angles
    else:
        raise ValueError("angle_convention must be 'pole_clamped' or 'masked'")

    angles = fused_rotation_angles(frames)
    if inverse_rescale is None:
        schedule = "canonical" if canonical_d2 else None
        return retained_wigner_angles(*angles, matrices_j, mmax, schedule=schedule)[0]
    schedule = "frame_pair" if (len(matrices_j), mmax) == (5, 2) else None
    return retained_wigner_angles(*angles, matrices_j, mmax, inverse_rescale, schedule=schedule)


def compact_inverse_rescale(inverse_rescale, *, lmax, mmax):
    """Pack ``(1, (L+1)^2, R)`` inverse scales into retained ``(K,)``.

    ``R`` follows the order-major layout in ``get_angular_layout``. Only
    within-degree entries are kept. Packing supports first and second
    derivatives. A differentiable scale can be reused within one forward
    pass, while an unchanged constant can be cached across passes.
    ``retained_wigner_from_frames`` accepts only constant scales.
    """
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    if inverse_rescale.shape != (1, spec.full_components, spec.reduced_components):
        raise ValueError(
            "inverse rescale must have shape (1, full_components, reduced_components)"
        )
    return torch.cat(
        tuple(
            inverse_rescale[0, degree * degree : (degree + 1) ** 2, row]
            for degree, rows in enumerate(spec.degree_rows)
            for row in rows
        )
    ).contiguous()


class FrozenQuaternionBasis(NamedTuple):
    """Prepared degree-two UMA rotation for one geometry derivative."""

    quaternion_frame: Callable[[Tensor], Tensor] | None


def prepare_frozen_rotation(device, *, quaternion_frame=None):
    """Select the compact representation required by the frozen receiver."""
    device = torch.device(device)
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    if capability == (8, 0) and quaternion_frame is None:
        return FrozenQuaternionBasis(None)
    if capability == (9, 0) and quaternion_frame is not None:
        return FrozenQuaternionBasis(quaternion_frame)
    raise ValueError("Frozen UMA rotation requires the Ampere frame or a Hopper quaternion constructor")


def frozen_rotation(geometry, basis, *, mmax):
    """Construct the two compact rotations for frozen UMA inference."""
    if mmax != 2:
        raise ValueError("Frozen UMA rotation requires unscaled degree-two quaternion rotations")
    if basis.quaternion_frame is not None:
        return rotation_pair(basis.quaternion_frame(geometry))
    gamma = torch.rand(geometry.shape[0], device=geometry.device, dtype=geometry.dtype) * (2 * math.pi)
    retained = fused_rotation(geometry, gamma)
    return retained, retained


def rotation(geometry, basis=None, *, lmax=None, mmax, convention=None, inverse_scale=None):
    """Construct retained Wigner rows without materializing dense rotations.

    For (E,3,3) frames, supply ``lmax`` to use the standard e3nn real basis.
    Alternatively, supply a sequence of fixed J matrices for a custom basis.
    ``convention`` selects ``pole_clamped`` or ``masked`` Euler angles.
    The default is ``pole_clamped``.

    Floating inputs use CUDA FP32. Basis tables and optional (K,) inverse
    scale are fixed. Geometry supports derivatives through third order.
    Return the forward (E,K) tensor, or (forward, scaled_inverse) when an
    inverse scale is supplied. Inverse consumers transpose degree blocks.

    """
    default_basis = basis is None
    if default_basis:
        if lmax is None:
            raise ValueError("rotation requires lmax or explicit basis tables")
        from .kernels.utils import standard_basis

        basis = standard_basis(lmax, device=geometry.device, dtype=geometry.dtype)
    elif lmax is not None and lmax != len(basis) - 1:
        raise ValueError("lmax must match the supplied basis tables")
    return retained_wigner_from_frames(
        geometry, tuple(basis), mmax=mmax,
        angle_convention="pole_clamped" if convention is None else convention,
        inverse_rescale=inverse_scale,
        canonical_d2=default_basis and (lmax, mmax) == (6, 2) and inverse_scale is None,
    )


@triton_op("so2::edge_quaternion_forward", mutates_args=("output",))
def edge_quaternion_forward_launch(
    edge_vectors: Tensor, gamma: Tensor, output: Tensor
) -> None:
    wrap_triton(edge_quaternion_forward_kernel)[(edge_vectors.shape[0],)](
        edge_vectors, gamma, output, edge_vectors.shape[0], 1, num_warps=1
    )


@triton_op("so2::edge_quaternion_backward", mutates_args=("edge_gradient",))
def edge_quaternion_backward_launch(
    q_gradient: Tensor,
    edge_vectors: Tensor,
    gamma: Tensor,
    edge_gradient: Tensor,
) -> None:
    wrap_triton(edge_quaternion_backward_kernel)[(edge_vectors.shape[0],)](
        q_gradient,
        edge_vectors,
        gamma,
        edge_gradient,
        edge_vectors.shape[0],
        1,
        num_warps=1,
    )


@triton_op(
    "so2::edge_quaternion_hvp",
    mutates_args=("q_direction", "edge_hvp"),
)
def edge_quaternion_hvp_launch(
    q_gradient: Tensor,
    edge_vectors: Tensor,
    gamma: Tensor,
    edge_direction: Tensor,
    q_direction: Tensor,
    edge_hvp: Tensor,
) -> None:
    wrap_triton(edge_quaternion_hvp_kernel)[(edge_vectors.shape[0],)](
        q_gradient,
        edge_vectors,
        gamma,
        edge_direction,
        q_direction,
        edge_hvp,
        edge_vectors.shape[0],
        1,
        num_warps=1,
    )


class EdgeQuaternionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_gradient, edge_vectors, gamma, compile_visible):
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(q_gradient, edge_vectors, gamma)
        return edge_quaternion_backward(q_gradient, edge_vectors, gamma, compile_visible)

    @staticmethod
    @once_differentiable
    def backward(ctx, edge_direction):
        q_gradient, edge_vectors, gamma = ctx.saved_tensors
        q_direction, edge_hvp = edge_quaternion_hvp(
            q_gradient,
            edge_vectors,
            gamma,
            edge_direction.contiguous(),
            ctx.compile_visible,
        )
        return q_direction, edge_hvp, None, None


class EdgeQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_vectors, gamma, compile_visible):
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(edge_vectors, gamma)
        return edge_quaternion_forward(edge_vectors, gamma, compile_visible)

    @staticmethod
    def backward(ctx, q_gradient):
        edge_vectors, gamma = ctx.saved_tensors
        return (
            EdgeQuaternionBackward.apply(
                q_gradient.contiguous(), edge_vectors, gamma, ctx.compile_visible
            ),
            None,
            None,
        )


def fused_edge_quaternion(
    edge_vectors: torch.Tensor, gamma: torch.Tensor, *, compile_visible: bool = False
) -> torch.Tensor:
    """Construct UMA-convention quaternions ``(E,4)`` in ``(w,x,y,z)`` order.

    ``edge_vectors`` is CUDA FP32 ``(E,3)``. ``gamma`` is a fixed CUDA FP32
    ``(E,)`` angle on the same device, in radians. The caller supplies the
    random local-frame angle, typically uniform on [0,2*pi). The operation
    blends two normalized quaternion charts and rotates around the local
    edge axis.

    Geometry and incoming cotangents support backward and double backward.
    Gamma defines the sampled frame and is not differentiable. Noncontiguous
    inputs are normalized to contiguous storage. ``compile_visible`` exposes
    launches to PyTorch tracing without changing the mathematical operation.
    """
    if (
        edge_vectors.dtype != torch.float32
        or not edge_vectors.is_cuda
        or edge_vectors.ndim != 2
        or edge_vectors.shape[1] != 3
    ):
        raise ValueError("UMA edge quaternion requires CUDA FP32 [E,3]")
    if (
        gamma.shape != (edge_vectors.shape[0],)
        or gamma.device != edge_vectors.device
        or gamma.dtype != torch.float32
        or gamma.requires_grad
    ):
        raise ValueError("gamma must be a fixed FP32 (E,) angle on the edge device")
    return EdgeQuaternion.apply(
        edge_vectors.contiguous(), gamma.contiguous(), compile_visible
    )


class AnglesBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_gradient, beta_gradient, gamma_gradient, rotation):
        ctx.save_for_backward(alpha_gradient, beta_gradient, gamma_gradient, rotation)
        return masked_angles_backward(
            alpha_gradient.contiguous(), beta_gradient.contiguous(),
            gamma_gradient.contiguous(), rotation,
        )

    @staticmethod
    def backward(ctx, rotation_direction):
        alpha_gradient, beta_gradient, gamma_gradient, rotation = ctx.saved_tensors
        inputs = (
            alpha_gradient.contiguous(),
            beta_gradient.contiguous(),
            gamma_gradient.contiguous(),
            rotation,
            rotation_direction,
        )
        if torch.is_grad_enabled():
            return AnglesDoubleBackward.apply(*inputs, masked_angles_hvp)
        return masked_angles_hvp(*inputs)


class Angles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation):
        ctx.save_for_backward(rotation)
        return masked_angles_forward(rotation)

    @staticmethod
    def backward(ctx, alpha_gradient, beta_gradient, gamma_gradient):
        if any(
            gradient is None
            for gradient in (alpha_gradient, beta_gradient, gamma_gradient)
        ):
            raise RuntimeError("all Euler angles must participate")
        (rotation,) = ctx.saved_tensors
        return AnglesBackward.apply(
            alpha_gradient.contiguous(),
            beta_gradient.contiguous(),
            gamma_gradient.contiguous(),
            rotation,
        )


def masked_angles_fused_rotation_angles(rotation):
    """Return three ``(E,)`` Euler-angle vectors from CUDA FP32 ``(E,3,3)`` frames.

    The second frame column supplies the normalized edge direction. Follows the
    masked pole convention and supports derivatives through third order.
    Noncontiguous frame views are accepted.
    """

    if tuple(rotation.shape[1:]) != (3, 3):
        raise ValueError(f"unexpected rotation shape: {tuple(rotation.shape)}")
    if not rotation.is_cuda or rotation.dtype != torch.float32:
        raise ValueError("fused Euler angles require CUDA FP32")
    return Angles.apply(rotation)


@triton_op("so2::quaternion_monomials_forward", mutates_args=("output",))
def monomials_forward_launch(
    q: Tensor, exponents: Tensor, output: Tensor, max_power: int
) -> None:
    count = exponents.shape[0]
    wrap_triton(monomial_forward_kernel)[(q.shape[0],)](
        q,
        exponents,
        output,
        count,
        block_monomials=block_monomials(count),
        max_power=max_power,
        num_warps=4,
    )


@triton_op("so2::quaternion_monomials_vjp", mutates_args=("q_gradient",))
def monomials_backward_launch(
    gradient: Tensor, q: Tensor, exponents: Tensor, q_gradient: Tensor, max_power: int
) -> None:
    count = exponents.shape[0]
    wrap_triton(monomial_vjp_kernel)[(q.shape[0], 4)](
        gradient,
        q,
        exponents,
        q_gradient,
        count,
        block_monomials=block_monomials(count),
        max_power=max_power,
        num_warps=4,
    )


@triton_op(
    "so2::quaternion_monomials_hvp",
    mutates_args=("monomial_direction", "q_hvp"),
)
def monomials_double_backward_launch(
    gradient: Tensor,
    q: Tensor,
    q_direction: Tensor,
    exponents: Tensor,
    monomial_direction: Tensor,
    q_hvp: Tensor,
    max_power: int,
) -> None:
    count = exponents.shape[0]
    block = block_monomials(count)
    wrap_triton(monomial_direction_kernel)[(q.shape[0],)](
        q,
        q_direction,
        exponents,
        monomial_direction,
        count,
        block_monomials=block,
        max_power=max_power,
        num_warps=4,
    )
    wrap_triton(monomial_q_hvp_kernel)[(q.shape[0], 4)](
        gradient,
        q,
        q_direction,
        exponents,
        q_hvp,
        count,
        block_monomials=block,
        max_power=max_power,
        num_warps=4,
    )


def monomials_forward(
    q: Tensor, exponents: Tensor, compile_visible: bool, max_power: int
) -> Tensor:
    output = q.new_empty((q.shape[0], exponents.shape[0]))
    if compile_visible:
        torch.ops.so2.quaternion_monomials_forward(q, exponents, output, max_power)
    else:
        monomials_forward_launch(q, exponents, output, max_power)
    return output


def monomials_backward(
    gradient: Tensor,
    q: Tensor,
    exponents: Tensor,
    compile_visible: bool,
    max_power: int,
) -> Tensor:
    q_gradient = torch.empty_like(q)
    if compile_visible:
        torch.ops.so2.quaternion_monomials_vjp(
            gradient, q, exponents, q_gradient, max_power
        )
    else:
        monomials_backward_launch(gradient, q, exponents, q_gradient, max_power)
    return q_gradient


def monomials_double_backward(
    gradient: Tensor,
    q: Tensor,
    q_direction: Tensor,
    exponents: Tensor,
    compile_visible: bool,
    max_power: int,
) -> tuple[Tensor, Tensor]:
    monomial_direction = torch.empty_like(gradient)
    q_hvp = torch.empty_like(q)
    if compile_visible:
        torch.ops.so2.quaternion_monomials_hvp(
            gradient,
            q,
            q_direction,
            exponents,
            monomial_direction,
            q_hvp,
            max_power,
        )
    else:
        monomials_double_backward_launch(
            gradient,
            q,
            q_direction,
            exponents,
            monomial_direction,
            q_hvp,
            max_power,
        )
    return monomial_direction, q_hvp


class QuaternionMonomialsBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient, q, exponents, compile_visible, max_power):
        ctx.max_power = max_power
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(gradient, q, exponents)
        return monomials_backward(gradient, q, exponents, compile_visible, max_power)

    @staticmethod
    @once_differentiable
    def backward(ctx, q_direction):
        gradient, q, exponents = ctx.saved_tensors
        monomial_direction, q_hvp = monomials_double_backward(
            gradient,
            q,
            q_direction.contiguous(),
            exponents,
            ctx.compile_visible,
            ctx.max_power,
        )
        return monomial_direction, q_hvp, None, None, None


class QuaternionMonomials(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, exponents, compile_visible, max_power):
        ctx.max_power = max_power
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(q, exponents)
        return monomials_forward(q, exponents, compile_visible, max_power)

    @staticmethod
    def backward(ctx, gradient):
        q, exponents = ctx.saved_tensors
        return (
            QuaternionMonomialsBackward.apply(
                gradient.contiguous(), q, exponents, ctx.compile_visible, ctx.max_power
            ),
            None,
            None,
            None,
        )


def quaternion_monomials(
    q: Tensor,
    exponents: Tensor,
    *,
    compile_visible: bool = False,
    max_power: int = 2147483647,
) -> Tensor:
    if q.dtype != torch.float32 or not q.is_cuda or tuple(q.shape[1:]) != (4,):
        raise ValueError("quaternion monomials require CUDA FP32 [E,4]")
    if (
        exponents.dtype != torch.int32
        or not exponents.is_cuda
        or tuple(exponents.shape[1:]) != (4,)
    ):
        raise ValueError("monomial exponents require CUDA int32 [K,4]")
    if max_power < 0:
        raise ValueError("maximum monomial power must be nonnegative")
    return QuaternionMonomials.apply(
        q.contiguous(), exponents.contiguous(), compile_visible, max_power
    )


class RotationAnglesBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_gradient, beta_gradient, gamma_gradient, rotation):
        ctx.save_for_backward(alpha_gradient, beta_gradient, gamma_gradient, rotation)
        return pole_clamped_angles_backward(
            alpha_gradient.contiguous(), beta_gradient.contiguous(),
            gamma_gradient.contiguous(), rotation,
        )

    @staticmethod
    def backward(ctx, rotation_direction):
        alpha_gradient, beta_gradient, gamma_gradient, rotation = ctx.saved_tensors
        inputs = (
            alpha_gradient.contiguous(),
            beta_gradient.contiguous(),
            gamma_gradient.contiguous(),
            rotation,
            rotation_direction.contiguous(),
        )
        if torch.is_grad_enabled():
            return AnglesDoubleBackward.apply(*inputs, pole_clamped_angles_hvp)
        return pole_clamped_angles_hvp(*inputs)


class RotationAngles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation):
        outputs = pole_clamped_angles_forward(rotation)
        ctx.save_for_backward(rotation)
        return outputs

    @staticmethod
    def backward(ctx, alpha_gradient, beta_gradient, gamma_gradient):
        (rotation,) = ctx.saved_tensors
        return RotationAnglesBackward.apply(
            alpha_gradient.contiguous(),
            beta_gradient.contiguous(),
            gamma_gradient.contiguous(),
            rotation,
        )


def pole_clamped_angles_fused_rotation_angles(rotation):
    """Return three ``(E,)`` Euler-angle vectors from CUDA FP32 ``(E,3,3)`` frames.

    The second frame column supplies the edge direction. At the poles, angle
    values follow the eSEN convention and their geometry derivatives vanish.
    Derivatives through third order are supported, including for noncontiguous
    frame views.
    """
    if (
        not rotation.is_cuda
        or rotation.dtype != torch.float32
        or rotation.ndim != 3
        or rotation.shape[-2:] != (3, 3)
    ):
        raise ValueError("fused rotation angles require CUDA FP32 [E,3,3]")
    return RotationAngles.apply(rotation)


@triton_op("so2::retained_quaternion_forward", mutates_args=("output",))
def retained_rows_forward_launch(
    q: Tensor, quadratic: Tensor, quartic: Tensor, output: Tensor, mmax: int
) -> None:
    wrap_triton(retained_forward_kernel)[(q.shape[0],)](
        q, quadratic, quartic, output, **launch_constants(mmax), num_warps=4
    )


@triton_op("so2::retained_quaternion_backward", mutates_args=("q_gradient",))
def retained_rows_backward_launch(
    matrix_gradient: Tensor,
    q: Tensor,
    quadratic: Tensor,
    quartic: Tensor,
    q_gradient: Tensor,
    mmax: int,
) -> None:
    wrap_triton(retained_backward_kernel)[(q.shape[0],)](
        matrix_gradient,
        q,
        quadratic,
        quartic,
        q_gradient,
        **launch_constants(mmax),
        num_warps=4,
    )


@triton_op("so2::retained_quaternion_hvp", mutates_args=("matrix_direction", "q_hvp"))
def retained_rows_double_backward_launch(
    matrix_gradient: Tensor,
    q: Tensor,
    q_direction: Tensor,
    quadratic: Tensor,
    quartic: Tensor,
    matrix_direction: Tensor,
    q_hvp: Tensor,
    mmax: int,
) -> None:
    wrap_triton(retained_hvp_kernel)[(q.shape[0],)](
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


class RetainedQuaternionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix_gradient, q, quadratic, quartic, mmax, compile_visible):
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(matrix_gradient, q, quadratic, quartic)
        ctx.mmax = mmax
        return retained_rows_backward(matrix_gradient, q, quadratic, quartic, mmax, compile_visible)

    @staticmethod
    @once_differentiable
    def backward(ctx, q_direction):
        matrix_gradient, q, quadratic, quartic = ctx.saved_tensors
        matrix_direction, q_hvp = retained_rows_hvp(
            matrix_gradient,
            q,
            q_direction.contiguous(),
            quadratic,
            quartic,
            ctx.mmax,
            ctx.compile_visible,
        )
        return matrix_direction, q_hvp, None, None, None, None


class RetainedQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, quadratic, quartic, mmax, compile_visible):
        ctx.compile_visible = compile_visible
        ctx.save_for_backward(q, quadratic, quartic)
        ctx.mmax = mmax
        return retained_rows_forward(q, quadratic, quartic, mmax, compile_visible)

    @staticmethod
    def backward(ctx, matrix_gradient):
        q, quadratic, quartic = ctx.saved_tensors
        return (
            RetainedQuaternionBackward.apply(
                matrix_gradient.contiguous(),
                q,
                quadratic,
                quartic,
                ctx.mmax,
                ctx.compile_visible,
            ),
            None,
            None,
            None,
            None,
        )


def retained_quaternion_l2(
    q: Tensor,
    quadratic: Tensor,
    quartic: Tensor,
    *,
    mmax: int,
    compile_visible: bool = False,
) -> Tensor:
    """Directly evaluate canonical selected rows, with analytic VJP and HVP."""
    if not 0 <= mmax <= 2:
        raise ValueError("L2 Wigner requires 0 <= M <= 2")
    if q.ndim != 2 or q.shape[1] != 4 or not q.is_cuda or q.dtype != torch.float32:
        raise ValueError("quaternion Wigner requires CUDA FP32 [E,4]")
    expected = (
        (3 * (2 * min(1, mmax) + 1), int(QUADRATIC_COUNT_TL.value)),
        (5 * (2 * mmax + 1), int(QUARTIC_COUNT_TL.value)),
    )
    for coefficients, shape in zip((quadratic, quartic), expected, strict=True):
        if (
            coefficients.shape != shape
            or coefficients.device != q.device
            or coefficients.dtype != q.dtype
            or not coefficients.is_contiguous()
            or coefficients.requires_grad
        ):
            raise ValueError(
                "fixed retained polynomial coefficients must match M, device and dtype"
            )
    return RetainedQuaternion.apply(
        q.contiguous(), quadratic, quartic, mmax, compile_visible
    )


def quaternion_wigner_from_edges_retained(
    edge_vectors: Tensor,
    quadratic: Tensor,
    quartic: Tensor,
    *,
    lmax: int,
    mmax: int,
    compile_visible: bool = False,
    polynomial_blocks: tuple[PolynomialBlock, ...] = (),
) -> Tensor:
    """Generate [E,K] forward rows; inverse consumers use the same rows transposed.

    Every coefficient table must have been selected with
    ``select_retained_coefficients`` at installation. ``polynomial_blocks``
    covers degrees 3 through L in order. Each block shares one monomial
    evaluation and matrix product across its degrees. Degrees up to two use
    the direct polynomial kernel. Coefficient tables are fixed during geometry
    differentiation. The random local-frame angle and quaternion convention
    match the original UMA generator.
    """
    pass

    if lmax < 0 or not 0 <= mmax <= lmax:
        raise ValueError("retained quaternion Wigner requires 0 <= M <= L")
    if lmax > 2 or polynomial_blocks:
        validate_polynomial_blocks(
            polynomial_blocks, lmax=lmax, mmax=mmax, edge_vectors=edge_vectors
        )
    gamma = torch.rand(
        edge_vectors.shape[0], dtype=edge_vectors.dtype, device=edge_vectors.device
    ) * (2 * math.pi)
    quaternion = fused_edge_quaternion(
        edge_vectors, gamma, compile_visible=compile_visible
    )
    if lmax < 2:
        scalar = quaternion[:, :1].square() * 0.0 + 1.0
        if lmax == 0:
            return scalar
        from .kernels.utils import QUADRATIC_EXPONENTS

        powers = torch.tensor(
            QUADRATIC_EXPONENTS, dtype=torch.int32, device=quaternion.device
        )
        expected = (3 * (2 * mmax + 1), len(QUADRATIC_EXPONENTS))
        if (
            quadratic.shape != expected
            or quadratic.requires_grad
            or quadratic.device != quaternion.device
            or quadratic.dtype != quaternion.dtype
        ):
            raise ValueError("fixed quadratic coefficients must match retained L1 rows")
        degree_one = quaternion_monomials(
            quaternion, powers, compile_visible=compile_visible, max_power=2
        ) @ quadratic.transpose(0, 1)
        return torch.cat((scalar, degree_one), dim=1)
    base = retained_quaternion_l2(
        quaternion,
        quadratic,
        quartic,
        mmax=min(mmax, 2),
        compile_visible=compile_visible,
    )
    if lmax == 2:
        return base
    blocks = [
        base,
        *(
            quaternion_monomials(
                quaternion,
                block.exponents,
                compile_visible=compile_visible,
                max_power=2 * max(block.degrees),
            )
            @ block.coefficients.transpose(0, 1)
            for block in polynomial_blocks
        ),
    ]
    return torch.cat(blocks, dim=1).contiguous()


class WignerBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        rows_by_degree = inputs[-1]
        count = len(rows_by_degree)
        matrix_gradients = inputs[:count]
        alpha, beta, gamma = inputs[count : count + 3]
        matrices_j = inputs[count + 3 : -1]
        ctx.save_for_backward(*matrix_gradients, alpha, beta, gamma, *matrices_j)
        ctx.rows_by_degree = rows_by_degree
        ctx.count = count
        return wigner_backward(
            matrix_gradients, alpha, beta, gamma, matrices_j, rows_by_degree
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, alpha_direction, beta_direction, gamma_direction):
        saved = ctx.saved_tensors
        matrix_gradients = saved[: ctx.count]
        alpha, beta, gamma = saved[ctx.count : ctx.count + 3]
        matrices_j = saved[ctx.count + 3 :]
        matrix_directions, angle_hvp = wigner_hvp(
            matrix_gradients,
            alpha, beta, gamma,
            tuple(
                direction.contiguous()
                for direction in (alpha_direction, beta_direction, gamma_direction)
            ),
            matrices_j,
            ctx.rows_by_degree,
        )
        return (*matrix_directions, *angle_hvp, *(None for _ in matrices_j), None)


class Wigner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        alpha, beta, gamma = inputs[:3]
        matrices_j = inputs[3:-1]
        rows_by_degree = inputs[-1]
        ctx.save_for_backward(alpha, beta, gamma, *matrices_j)
        ctx.rows_by_degree = rows_by_degree
        return wigner_forward(alpha, beta, gamma, matrices_j, rows_by_degree)

    @staticmethod
    def backward(ctx, *matrix_gradients):
        if any(gradient is None for gradient in matrix_gradients):
            raise RuntimeError("all sparse Wigner blocks must participate")
        alpha, beta, gamma, *matrices_j = ctx.saved_tensors
        angle_gradients = WignerBackward.apply(
            *(gradient.contiguous() for gradient in matrix_gradients),
            alpha, beta, gamma,
            *matrices_j,
            ctx.rows_by_degree,
        )
        return (*angle_gradients, *(None for _ in matrices_j), None)


class RetainedD2(NamedTuple):
    layout: RetainedRotationLayout
    schedule: str


class RetainedWignerBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, direct_gradient, inverse_gradient,
        alpha, beta, gamma, inverse_rescale, *inputs,
    ):
        *matrices_j, spec = inputs
        ctx.schedule = spec.schedule if isinstance(spec, RetainedD2) else None
        if ctx.schedule is not None:
            spec = spec.layout
        ctx.save_for_backward(
            direct_gradient, inverse_gradient, alpha, beta, gamma,
            inverse_rescale,
            *matrices_j,
        )
        ctx.spec = spec
        return retained_backward(
            direct_gradient, inverse_gradient, alpha, beta, gamma,
            matrices_j, inverse_rescale, spec,
        )

    @staticmethod
    def backward(ctx, *directions):
        (
            direct_gradient, inverse_gradient, alpha, beta, gamma,
            inverse_rescale,
            *matrices_j,
        ) = ctx.saved_tensors
        calculate = retained_hvp
        if ctx.schedule == "canonical":
            calculate = default_retained_hvp
        elif ctx.schedule == "frame_pair":
            calculate = frame_paired_retained_hvp
        directions = tuple(direction.contiguous() for direction in directions)
        args = direct_gradient, inverse_gradient, alpha, beta, gamma
        if torch.is_grad_enabled():
            gradients = RetainedWignerDoubleBackward.apply(
                *args, *directions, inverse_rescale, *matrices_j, ctx.spec, calculate,
            )
        else:
            matrix_directions, angle_hvp = calculate(
                *args, directions, matrices_j, inverse_rescale, ctx.spec,
            )
            gradients = (*matrix_directions, *angle_hvp)
        return (*gradients, None, *(None for _ in matrices_j), None)


class RetainedWigner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, beta, gamma, inverse_rescale, *inputs):
        *matrices_j, spec = inputs
        ctx.save_for_backward(alpha, beta, gamma, inverse_rescale, *matrices_j)
        ctx.spec = spec
        layout = spec.layout if isinstance(spec, RetainedD2) else spec
        return retained_forward(alpha, beta, gamma, matrices_j, inverse_rescale, layout)

    @staticmethod
    def backward(ctx, *gradients):
        alpha, beta, gamma, inverse_rescale, *matrices_j = ctx.saved_tensors
        direct_gradient = gradients[0].contiguous()
        inverse_gradient = (
            gradients[1].contiguous() if inverse_rescale is not None else None
        )
        angle_gradients = RetainedWignerBackward.apply(
            direct_gradient, inverse_gradient, alpha, beta, gamma,
            inverse_rescale,
            *matrices_j,
            ctx.spec,
        )
        return (*angle_gradients, None, *(None for _ in matrices_j), None)


def retained_wigner_angles(
    alpha, beta, gamma, matrices_j, mmax, inverse_rescale=None, *, schedule=None,
):
    validate_inputs(alpha, beta, gamma, matrices_j, mmax)
    spec = get_retained_layout(lmax=len(matrices_j) - 1, mmax=mmax, channels=1)
    require_int32_offsets(
        (alpha.shape[0], spec.compact_elements),
        (spec.compact_elements, 1),
        name="retained Wigner carrier",
    )
    if inverse_rescale is not None and (
        inverse_rescale.shape != (spec.compact_elements,)
        or inverse_rescale.device != alpha.device
        or inverse_rescale.dtype != alpha.dtype
        or not inverse_rescale.is_contiguous()
        or inverse_rescale.requires_grad
    ):
        raise ValueError(
            "inverse rescale must be a fixed contiguous FP32 retained vector on the angle device"
        )
    if schedule is not None:
        spec = RetainedD2(spec, schedule)
    return RetainedWigner.apply(alpha, beta, gamma, inverse_rescale, *matrices_j, spec)


def sparse_wigner(
    alpha, beta, gamma, matrices_j, *, mmax: int, storage: str = "blocks"
):
    """Evaluate degrees 0..L and retain orders -M..M with differentiable angles.

    ``blocks`` returns one [E, min(2*M+1, 2*l+1), 2*l+1] tensor per
    degree. ``retained_rows`` writes canonical [E, K] storage directly.
    J matrices are fixed basis constants, not trainable parameters. Inputs
    are CUDA FP32 vectors. First and second derivatives propagate to all
    three Euler angles, including the incoming first-derivative cotangent.
    """
    if storage not in ("blocks", "retained_rows"):
        raise ValueError(f"unknown Wigner storage: {storage!r}")
    if storage == "retained_rows":
        return retained_wigner_angles(alpha, beta, gamma, matrices_j, mmax)[0]
    rows = validate_inputs(alpha, beta, gamma, matrices_j, mmax)
    return Wigner.apply(alpha, beta, gamma, *matrices_j, rows)


def sparse_wigner_pair(alpha, beta, gamma, matrices_j, inverse_rescale, *, mmax: int):
    """Directly generate retained direct and scaled inverse carriers.

    ``inverse_rescale`` is a fixed [K] convention tensor. Inverse entries
    are multiplied before storage; both carriers contribute to angle D1/D2,
    including gradients with respect to both incoming output cotangents.
    """
    if inverse_rescale is None:
        raise ValueError("paired Wigner requires a fixed inverse scale")
    return retained_wigner_angles(alpha, beta, gamma, matrices_j, mmax, inverse_rescale)


@triton_op("lunaeq_so2::edge_wigner", mutates_args=())
def fused_rotation_with_quaternion(vectors: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    vectors, gamma = vectors.contiguous(), gamma.contiguous()
    edges = vectors.shape[0]
    out = vectors.new_empty((edges, 35))
    q = vectors.new_empty((edges, 4))
    wrap_triton(fused_rotation_kernel)[(triton.cdiv(edges, 128),)](
        vectors, gamma, out, q, edges, 128, num_warps=4,
    )
    return out, q


@triton_op("lunaeq_so2::edge_wigner_vjp", mutates_args=())
def fused_rotation_input_vjp(q: Tensor, vectors: Tensor, gamma: Tensor, seed: Tensor) -> Tensor:
    vectors, gamma = vectors.contiguous(), gamma.contiguous()
    seed = seed.resolve_neg()
    edges = vectors.shape[0]
    out = torch.empty_like(vectors)
    wrap_triton(fused_rotation_vjp_kernel)[(triton.cdiv(edges, 32),)](
        q, vectors, gamma, seed, out, edges, *seed.stride(), 32, num_warps=4,
    )
    return out


def fused_rotation_setup_context(ctx, inputs, output):
    _, q = output
    ctx.mark_non_differentiable(q)
    ctx.save_for_backward(q, *inputs)


def fused_rotation_backward(ctx, grad, _):
    return fused_rotation_input_vjp(*ctx.saved_tensors, grad), None


fused_rotation_with_quaternion.register_autograd(fused_rotation_backward, setup_context=fused_rotation_setup_context)


def fused_rotation(vectors, gamma):
    return fused_rotation_with_quaternion(vectors, gamma)[0]


@triton_op("trim_h100::natural_rotation_pair", mutates_args=())
def rotation_pair(q: Tensor) -> tuple[Tensor, Tensor]:
    q = q.contiguous()
    out = q.new_empty((q.shape[0], 35))
    inverse = torch.empty_like(out)
    wrap_triton(forward_pair)[(triton.cdiv(q.shape[0], 128),)](
        q, out, inverse, q.shape[0], 128, num_warps=4,
    )
    return out, inverse


@triton_op("trim_h100::natural_rotation_pair_vjp", mutates_args=())
def rotation_pair_vjp(q: Tensor, g: Tensor, h: Tensor) -> Tensor:
    q = q.contiguous()
    out = torch.empty_like(q)
    g, h = g.resolve_neg(), h.resolve_neg()
    wrap_triton(backward_pair)[(triton.cdiv(q.shape[0], 128),)](
        q, g, h, out, q.shape[0], *g.stride(), *h.stride(), 128, num_warps=4,
    )
    return out


def rotation_pair_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def rotation_pair_backward(ctx, g, h):
    return rotation_pair_vjp(*ctx.saved_tensors, g, h)


rotation_pair.register_autograd(rotation_pair_backward, setup_context=rotation_pair_setup_context)
