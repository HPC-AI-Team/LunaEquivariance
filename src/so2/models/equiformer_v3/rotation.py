"""Bind the official rotation convention to compact tensor operations."""

from __future__ import annotations
from types import MethodType

import torch
from fairchem.experimental.models.equiformer_v3.wigner import _Jd
from LunaEquivariance.so2 import rotation as build_rotation
from LunaEquivariance.so2.rotation.interface import (
    ROTATION_MASK_THRESHOLD,
    compact_inverse_rescale,
)
from LunaEquivariance.so2.utils.layout import get_retained_layout


def generated_set_wigner(rotation, rot_mat3x3: torch.Tensor) -> None:
    """Populate compact direct/inverse Wigner carriers without dense assembly."""

    from fairchem.experimental.models.equiformer_v3.edge_rot_mat import (
        _ROTATION_MASK_THRESHOLD,
    )
    if _ROTATION_MASK_THRESHOLD != ROTATION_MASK_THRESHOLD:
        raise ValueError("unsupported rotation-mask convention")
    lmax, mmax = rotation.lmax, rotation.mmax
    if not 0 <= mmax <= lmax or lmax >= len(_Jd):
        raise ValueError(
            "rotation requires valid L/M and an official J matrix for every degree"
        )
    if rot_mat3x3.dtype != torch.float32 or not rot_mat3x3.is_cuda:
        raise ValueError("generated compact Wigner requires CUDA FP32")
    if not rotation.use_rotation_mask:
        raise ValueError("generated Euler derivatives require rotation masking")
    matrices_j, inverse_rescale = rotation_constants(
        rotation, device=rot_mat3x3.device, dtype=rot_mat3x3.dtype
    )
    rotation.wigner, rotation.wigner_inv = compact_wigner(
        rot_mat3x3, matrices_j, inverse_rescale, lmax=lmax, mmax=mmax
    )


def rotation_constants(rotation, *, device, dtype):
    lmax, mmax = rotation.lmax, rotation.mmax
    mapping = rotation.wigner_index_to_m_array
    dense_inverse_rescale = rotation.wigner_inv_rescale
    sources = (mapping, dense_inverse_rescale, *_Jd[: lmax + 1])
    signature = None
    if not torch.is_inference_mode_enabled() and not any(
        value.requires_grad for value in sources
    ):
        try:
            signature = (
                lmax,
                mmax,
                device,
                dtype,
                tuple((id(value), value._version) for value in sources),
            )
        except RuntimeError:
            # Inference tensors have no version counter, so mutation is not observable.
            pass
    cached = getattr(rotation, "so2_rotation_constants", None)
    if signature is not None and cached is not None and cached[0] == signature:
        return cached[1:3]
    rotation.__dict__.pop("so2_rotation_constants", None)
    expected_rows = reduced_to_full(lmax, mmax)
    actual_rows = tuple(mapping.abs().argmax(dim=1).tolist())
    if actual_rows != expected_rows:
        raise ValueError(f"unexpected reduced Wigner mapping: {actual_rows}")
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    expected_shape = (1, spec.full_components, spec.reduced_components)
    if tuple(dense_inverse_rescale.shape) != expected_shape:
        raise ValueError(
            f"unexpected inverse Wigner rescale: {dense_inverse_rescale.shape}"
        )
    inverse_rescale = compact_inverse_rescale(
        dense_inverse_rescale, lmax=lmax, mmax=mmax
    )
    matrices_j = tuple(
        value.to(dtype=dtype, device=device).contiguous() for value in _Jd[: lmax + 1]
    )
    if signature is not None:
        rotation.so2_rotation_constants = (
            signature,
            matrices_j,
            inverse_rescale,
            sources,
        )
    return matrices_j, inverse_rescale


@torch.compiler.disable
def runtime_set_wigner_forward(self, rot_mat3x3: torch.Tensor) -> None:
    generated_set_wigner(self, rot_mat3x3)


def install_runtime_rotation(rotation):
    rotation.set_wigner = MethodType(runtime_set_wigner_forward, rotation)


def restore_runtime_rotation(rotation, state):
    for name in ("set_wigner", "so2_rotation_constants"):
        if name in state:
            setattr(rotation, name, state[name])
        else:
            rotation.__dict__.pop(name, None)


def reduced_to_full(lmax: int, mmax: int) -> tuple[int, ...]:
    mapping = tuple(degree * degree + degree for degree in range(lmax + 1))
    for order in range(1, mmax + 1):
        mapping += tuple(
            degree * degree + degree + order for degree in range(order, lmax + 1)
        )
        mapping += tuple(
            degree * degree + degree - order for degree in range(order, lmax + 1)
        )
    return mapping


def compact_wigner(rot_mat3x3, matrices_j, inverse_rescale, *, lmax, mmax):
    """Construct retained rotation blocks from explicit convention tensors."""
    if len(matrices_j) != lmax + 1:
        raise ValueError("Wigner J matrices do not match lmax")
    return build_rotation(
        rot_mat3x3,
        matrices_j,
        mmax=mmax,
        convention="masked",
        inverse_scale=inverse_rescale,
    )
