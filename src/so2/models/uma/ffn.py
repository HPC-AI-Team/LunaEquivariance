"""Bind UMA equivariant FFNs and normalization operators."""

from __future__ import annotations
from types import MethodType
import torch
from ..spectral import (
    validate_spectral_ffn,
    fused_equivariant_ffn,
    rms_norm_forward,
)


def fused_block_forward(
    module,
    x,
    x_edge,
    edge_index,
    wigner,
    wigner_inv_envelope,
    total_atoms_across_gp_ranks,
    sys_node_embedding=None,
    scatter_target=None,
    gp_ctx=None,
):
    residual = x
    x = module.norm_1(x)
    if sys_node_embedding is not None:
        x[:, 0, :] = x[:, 0, :] + sys_node_embedding
    x = module.edge_wise(
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks=total_atoms_across_gp_ranks,
        scatter_target=scatter_target,
        gp_ctx=gp_ctx,
    )
    x = x + residual
    return fused_equivariant_ffn(module.norm_2, module.atom_wise, x)


opaque_fused_block_forward = torch.compiler.disable(fused_block_forward)


def install_fused_equivariant_ffns(model: torch.nn.Module) -> int:
    from fairchem.core.models.uma.escn_md_block import eSCNMD_Block, SpectralAtomwise

    for module in model.modules():
        if isinstance(module, SpectralAtomwise):
            validate_spectral_ffn(module)
    installed = 0
    for module in model.modules():
        if isinstance(module, eSCNMD_Block) and isinstance(
            module.atom_wise, SpectralAtomwise
        ):
            module.forward = MethodType(opaque_fused_block_forward, module)
            installed += 1
    if installed == 0:
        raise RuntimeError("no UMA spectral equivariant FFNs were found")
    return installed


def install_fused_rms_norms(model: torch.nn.Module) -> int:
    from fairchem.core.models.uma.nn.layer_norm import (
        EquivariantRMSNormArraySphericalHarmonicsV2,
    )

    installed = 0
    for module in model.modules():
        if isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2):
            if not (
                module.affine
                and module.centering
                and module.std_balance_degrees
                and module.normalization == "component"
            ):
                raise RuntimeError("unsupported UMA RMSNorm configuration")
            module.forward = MethodType(
                torch.compiler.disable(rms_norm_forward), module
            )
            installed += 1
    if installed == 0:
        raise RuntimeError("no UMA equivariant RMSNorm modules were found")
    return installed
