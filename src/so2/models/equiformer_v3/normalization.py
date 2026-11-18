"""EqV3 normalization forwards and temporary installation."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Callable, Iterator, Sequence

import torch

from LunaEquivariance.so2.utils.degree_norm import degree_norm


# Keep custom Triton launch and autograd state outside Dynamo tracing.
@torch.compiler.disable
def fused_equivariant_norm_forward(
    self,
    inputs: torch.Tensor,
) -> torch.Tensor:
    return degree_norm(
        inputs,
        self.affine_weight,
        self.affine_bias,
        epsilon=self.eps,
    )


def installed_equivariant_norm_count(model):
    return sum(
        getattr(module.forward, "__func__", None) is fused_equivariant_norm_forward
        for module in model.modules()
    )


def install_equivariant_norms(
    model: torch.nn.Module,
    block_indices: Sequence[int],
) -> Callable[[], None]:
    """Bind affine, degree-balanced LayerNorm in the selected model blocks."""

    indices = tuple(block_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("block indices must be a non-empty unique sequence")
    if min(indices) < 0 or max(indices) >= len(model.blocks):
        raise ValueError(f"block index outside [0, {len(model.blocks)})")
    norms = []
    for index in indices:
        norms.extend((model.blocks[index].norm_1, model.blocks[index].norm_2))
    if indices == tuple(range(len(model.blocks))):
        norms.append(model.norm)
    for norm in norms:
        expected = {
            "affine": True,
            "std_balance_degrees": True,
            "centering": True,
            "normalization": "component",
        }
        actual = {name: getattr(norm, name, None) for name in expected}
        if actual != expected or norm.affine_bias is None:
            raise ValueError(f"unexpected EqV3 merge LayerNorm: {actual=}")
    old_forwards = [norm.__dict__.get("forward") for norm in norms]
    for norm in norms:
        norm.forward = MethodType(fused_equivariant_norm_forward, norm)

    def restore():
        for norm, old_forward in zip(norms, old_forwards, strict=True):
            if old_forward is None:
                del norm.forward
            else:
                norm.forward = old_forward

    return restore


@contextmanager
def fused_equivariant_norm_blocks(
    model: torch.nn.Module, block_indices: Sequence[int]
) -> Iterator[None]:
    restore = install_equivariant_norms(model, block_indices)
    try:
        yield
    finally:
        restore()
