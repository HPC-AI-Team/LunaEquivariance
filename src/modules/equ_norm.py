"""Fused weighted equivariant layer and RMS normalization."""

from __future__ import annotations

import torch
from torch import nn

from ..nn.equ_norm.interface import EquNormOperator, equivariant_norm


class LunaEqEquivariantNorm(nn.Module):
    """Fused weighted equivariant layer normalization or RMS normalization.

    ``grouping`` selects per_irrep, separable or merged denominators.
    ``norm='layer'`` centers even scalars; ``norm='rms'`` preserves their mean.
    ``weighting='inverse_dim_squared'`` weights by the inverse squared irrep
    dimension, retaining the channel sum.
    Gamma always shares one value across all m of an irrep copy.
    """

    def __init__(
        self,
        irreps,
        *,
        norm="layer",
        grouping="merged",
        weighting="component",
        layout="mul_ir",
        eps=None,
        affine=True,
        bias=None,
        stat_weights=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if norm not in ("layer", "rms"):
            raise ValueError("norm must be layer or rms")
        if eps is None:
            eps = 1e-12 if weighting == "inverse_dim_squared" else 1e-5

        self.op = EquNormOperator(
            irreps,
            grouping=grouping,
            center_scalars=norm == "layer",
            weighting=weighting,
            layout=layout,
            eps=eps,
            stat_weights=stat_weights,
        )

        self.irreps = irreps
        plan = self.op.plan
        self.weight = (
            nn.Parameter(torch.ones(plan.param_dim, device=device, dtype=dtype)) if affine else None
        )
        if bias is None:
            bias = affine and norm == "layer"
        self.bias = (
            nn.Parameter(torch.zeros(plan.bias_dim, device=device, dtype=dtype))
            if bias and plan.bias_dim
            else None
        )

    def forward(self, x):
        return equivariant_norm(x, self.irreps, self.weight, self.bias, config=self.op)


EquivariantNorm = LunaEqEquivariantNorm


__all__ = ["EquivariantNorm", "LunaEqEquivariantNorm"]
