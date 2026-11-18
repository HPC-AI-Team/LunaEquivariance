"""Bind UMA edge-degree initialization to common scalar aggregation."""

from __future__ import annotations
from types import MethodType
import torch
from LunaEquivariance.so2.inverse_rotate_reduce.interface import fused_initial_scalar, structured_sector_tail
from ..radial import (
    fused_radial_mlp_from_module as fused_radial_mlp,
    validate_radial_mlp,
)


def structured_edge_degree_forward(
    module,
    x,
    x_edge,
    scatter_target,
    wigner_inv_envelope,
):
    radial = fused_radial_mlp(
        module.rad_func,
        x_edge,
    ).reshape(
        x_edge.shape[0],
        module.m_0_num_coefficients,
        module.sphere_channels,
    )
    rescale_envelope = torch.full(
        (x_edge.shape[0],),
        1.0 / module.rescale_factor,
        device=x_edge.device,
        dtype=x_edge.dtype,
    )
    embedding = structured_sector_tail(
        (radial,),
        rescale_envelope,
        wigner_inv_envelope,
        scatter_target,
        x.shape[0],
        lmax=module.operator_lmax,
        mmax=0,
        wigner_layout="retained_rows",
        wigner_mmax=module.operator_mmax,
    )
    return fused_initial_scalar(embedding, x[:, 0, :].contiguous())


def install_structured_edge_degree(model: torch.nn.Module) -> int:
    from fairchem.core.models.uma.nn.embedding import EdgeDegreeEmbedding

    installed = 0
    for module in model.modules():
        if isinstance(module, EdgeDegreeEmbedding):
            if module.activation_checkpoint_chunk_size is not None:
                raise ValueError(
                    "the UMA edge embedding does not support activation checkpoint chunking"
                )
            validate_radial_mlp(module.rad_func)
            module.operator_lmax = model.backbone.lmax
            module.operator_mmax = model.backbone.mmax
            module.forward = MethodType(structured_edge_degree_forward, module)
            installed += 1
    if installed != 1:
        raise RuntimeError(f"expected one UMA edge-degree embedding, found {installed}")
    return installed
