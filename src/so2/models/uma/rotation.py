"""Bind UMA edge-frame construction to retained Wigner blocks."""

from __future__ import annotations

from types import MethodType

import torch

from LunaEquivariance.so2.rotation.interface import (
    PolynomialBlock,
    compress_l2_coefficients,
    frozen_rotation,
    monomial_exponents,
    polynomial_wigner_coefficients,
    prepare_frozen_rotation,
    quadratic_coefficients,
    quaternion_wigner_from_edges_retained,
    register_sparse_quaternion_basis,
    select_retained_coefficients,
)


def update_sparse_basis_registration(module):
    if (
        (module.lmax, module.mmax) != (2, 2)
        or module.operator_rotation_compile_visible or torch.compiler.is_compiling()
    ):
        return
    quadratic = module.operator_wigner_quadratic
    quartic = module.operator_wigner_quartic
    identity = id(quadratic), id(quartic)
    if identity != getattr(module, "operator_wigner_basis_ids", None):
        if register_sparse_quaternion_basis(quadratic, quartic) is not None:
            module.operator_wigner_basis_ids = identity


def fused_edge_frame_forward(module, edge_vectors):
    update_sparse_basis_registration(module)
    blocks = tuple(
        PolynomialBlock(
            degrees,
            getattr(module, f"operator_wigner_l{degrees[-1]}_exponents"),
            getattr(module, f"operator_wigner_l{degrees[-1]}_coefficients"),
        )
        for degrees in module.operator_wigner_degrees
    )
    retained = quaternion_wigner_from_edges_retained(
        edge_vectors, module.operator_wigner_quadratic, module.operator_wigner_quartic,
        lmax=module.lmax, mmax=module.mmax, polynomial_blocks=blocks,
        compile_visible=module.operator_rotation_compile_visible,
    )
    return retained, retained


def frozen_edge_frame_forward(module, edge_vectors):
    return frozen_rotation(edge_vectors, module.operator_frozen_rotation, mmax=module.mmax)


def install_frozen_edge_frame(backbone, device, *, quaternion_frame=None):
    backbone.operator_frozen_rotation = prepare_frozen_rotation(device, quaternion_frame=quaternion_frame)
    backbone._get_rotmat_and_wigner = MethodType(frozen_edge_frame_forward, backbone)


def install_fused_edge_frame(
    backbone: torch.nn.Module, *, compile_visible: bool = False
) -> int:
    if backbone.lmax < 0 or not 0 <= backbone.mmax <= backbone.lmax:
        raise RuntimeError(f"unsupported UMA Wigner degree L={backbone.lmax}")
    coefficients = backbone.wigner_data.custom_kernels.C_l2
    backbone.register_buffer(
        "operator_wigner_quadratic",
        select_retained_coefficients(
            quadratic_coefficients(coefficients.device),
            degrees=(1,),
            mmax=backbone.mmax,
        ),
        persistent=False,
    )
    backbone.register_buffer(
        "operator_wigner_quartic",
        select_retained_coefficients(
            compress_l2_coefficients(coefficients), degrees=(2,), mmax=backbone.mmax
        ),
        persistent=False,
    )
    if backbone.lmax == 3:
        groups = ((3,),)
    elif 4 <= backbone.lmax <= 6:
        groups = ((3, 4), *((degree,) for degree in range(5, backbone.lmax + 1)))
    else:
        groups = tuple((degree,) for degree in range(3, backbone.lmax + 1))
    for degrees in groups:
        degree = degrees[-1]
        if degrees == (3, 4):
            table = backbone.wigner_data.custom_kernels.C_combined_l3l4
        elif degree <= 4:
            table = getattr(backbone.wigner_data.custom_kernels, f"C_l{degree}")
        else:
            table = polynomial_wigner_coefficients(
                degree,
                backbone.wigner_data.U_blocks[degree - 5],
                coefficients.device,
            )
        backbone.register_buffer(
            f"operator_wigner_l{degree}_exponents",
            monomial_exponents(2 * degree, coefficients.device),
            persistent=False,
        )
        backbone.register_buffer(
            f"operator_wigner_l{degree}_coefficients",
            select_retained_coefficients(table, degrees=degrees, mmax=backbone.mmax),
            persistent=False,
        )
    backbone.operator_wigner_degrees = groups
    backbone.operator_rotation_compile_visible = compile_visible
    update_sparse_basis_registration(backbone)
    backbone._get_rotmat_and_wigner = MethodType(fused_edge_frame_forward, backbone)
    return 1
