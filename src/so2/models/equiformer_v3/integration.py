"""Install the complete EqV3 operator configuration for one context."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch

from .attention import (
    install_attention,
    runtime_attention_blocks,
    validate_checkpointing,
)
from .embedding import install_runtime_embedding, restore_runtime_embedding
from .ffn import install_grid_ffns, reassociated_grid_ffn_blocks
from .normalization import (
    install_equivariant_norms,
    fused_equivariant_norm_blocks,
    installed_equivariant_norm_count,
)
from .rotation import install_runtime_rotation, restore_runtime_rotation


def install_equiformer_v3_operator(model: torch.nn.Module) -> dict[str, int]:
    validate_checkpointing(model)
    indices = tuple(range(len(model.blocks)))
    install_runtime_rotation(model.so3_rotation)
    install_runtime_embedding(model, model.edge_degree_embedding)
    install_equivariant_norms(model, indices)
    install_grid_ffns(model, indices)
    install_attention(model, indices)
    return {
        "rotation": 1,
        "embedding": 1,
        "attention_blocks": len(indices),
        "grid_ffn_blocks": len(indices),
        "equivariant_norms": installed_equivariant_norm_count(model),
    }


@contextmanager
def eqv3_runtime_blocks(
    model: torch.nn.Module,
    block_indices: Sequence[int],
) -> Iterator[None]:
    """Install SO(2) optimizations in the EquiformerV3 model."""

    validate_checkpointing(model)
    indices = tuple(block_indices)
    rotation = model.so3_rotation
    edge_degree = model.edge_degree_embedding
    rotation_state = {
        name: rotation.__dict__[name]
        for name in ("set_wigner", "so2_rotation_constants")
        if name in rotation.__dict__
    }
    embedding_state = {
        name: owner.__dict__[name]
        for owner, name in ((edge_degree, "forward"), (model, "_forward_embedding"))
        if name in owner.__dict__
    }
    try:
        install_runtime_rotation(rotation)
        install_runtime_embedding(model, edge_degree)
        with fused_equivariant_norm_blocks(model, indices):
            with reassociated_grid_ffn_blocks(model, indices):
                with runtime_attention_blocks(model, indices):
                    yield
    finally:
        restore_runtime_rotation(rotation, rotation_state)
        restore_runtime_embedding(model, edge_degree, embedding_state)
