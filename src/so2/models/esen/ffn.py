"""Coefficient-major execution for the eSEN spectral equivariant FFN."""

from __future__ import annotations

from types import MethodType

import torch
from ..spectral import (
    validate_spectral_ffn,
    fused_equivariant_ffn,
)

from LunaEquivariance.so2 import feed_forward


def coefficient_major_spectral_forward(module, inputs):
    scalar_linear = module.scalar_mlp[0]
    first = module.so3_linear_1
    second = module.so3_linear_2
    return feed_forward(
        inputs, first.weight, second.weight, scalar_linear.weight,
        input_bias=first.bias, output_bias=second.bias,
        gate_bias=scalar_linear.bias,
    )


def fused_equivariant_ffn_block_forward(
    module,
    x,
    x_edge,
    edge_distance,
    edge_index,
    wigner,
    wigner_inv,
    node_offset=0,
):
    residual = x
    x = module.norm_1(x)
    x = module.edge_wise(
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_offset,
    )
    x = x + residual
    return opaque_fused_equivariant_ffn(module.norm_2, module.atom_wise, x)


def install_coefficient_major_ffns(model: torch.nn.Module) -> int:
    from fairchem.core.models.esen.esen_block import SpectralAtomwise, eSEN_Block

    for module in model.modules():
        if isinstance(module, SpectralAtomwise):
            validate_spectral_ffn(module)
    installed = 0
    for module in model.modules():
        if isinstance(module, SpectralAtomwise):
            module.forward = MethodType(coefficient_major_spectral_forward, module)
            installed += 1
        if isinstance(module, eSEN_Block) and isinstance(
            module.atom_wise, SpectralAtomwise
        ):
            module.forward = MethodType(fused_equivariant_ffn_block_forward, module)
    if (
        hasattr(model, "norm")
        and hasattr(model, "atomwise")
        and isinstance(model.atomwise, SpectralAtomwise)
    ):
        model.norm.coefficient_major_output = True
    if installed == 0:
        raise RuntimeError("no eSEN spectral equivariant FFNs were found")
    return installed


@torch.compiler.disable
def opaque_fused_equivariant_ffn(norm, atomwise, x):
    # Keep the custom backward and double backward outside Dynamo tracing.
    # Their optional saved tensors are interpreted by our autograd functions.
    return fused_equivariant_ffn(norm, atomwise, x)
