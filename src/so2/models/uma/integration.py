"""Install the UMA operator components."""

from __future__ import annotations
import torch
from .radial import install_sequential_radials
from .mole import (
    enable_segmented_mole_training,
    install_segmented_mole,
)
from .rotation import install_fused_edge_frame
from .convolution import (
    install_structured_edgewise,
    validate_structured_edgewise,
)
from .embedding import install_structured_edge_degree
from .ffn import (
    install_fused_equivariant_ffns,
    install_fused_rms_norms,
)
from .convolution import UMAOperatorBackend as UMAOperatorBackend


def install_uma_operator(model: torch.nn.Module) -> dict[str, int]:
    from fairchem.core.models.uma.escn_md_block import Edgewise
    from fairchem.core.models.uma.nn.mole import MOLE

    edgewise_modules = tuple(
        module for module in model.modules() if isinstance(module, Edgewise)
    )
    if not edgewise_modules:
        raise RuntimeError("no UMA Edgewise modules were found")
    if getattr(model.backbone, "ff_type", "spectral") != "spectral":
        raise ValueError(
            "the UMA operator requires ff_type='spectral'; grid FFNs are unsupported"
        )
    for module in edgewise_modules:
        validate_structured_edgewise(module)
    if (
        model.backbone.edge_degree_embedding.activation_checkpoint_chunk_size
        is not None
    ):
        raise ValueError(
            "the UMA edge embedding does not support activation checkpoint chunking"
        )
    edge_frame_count = install_fused_edge_frame(model.backbone)
    radial_count = install_sequential_radials(edgewise_modules)
    has_mole = any(isinstance(module, MOLE) for module in model.modules())
    mole_count = install_segmented_mole(model) if has_mole else 0
    if has_mole:
        for module in edgewise_modules:
            enable_segmented_mole_training(module.so2_conv_1, force_training=True)
            if any(isinstance(layer, MOLE) for layer in module.so2_conv_2.modules()):
                enable_segmented_mole_training(module.so2_conv_2, force_training=True)
    edgewise_count = install_structured_edgewise(model, shared_radial=True)
    initial_count = install_structured_edge_degree(model)
    ffn_count = install_fused_equivariant_ffns(model)
    norm_count = install_fused_rms_norms(model)
    return {
        "edge_frame_construction": edge_frame_count,
        "radial_weight_generation": radial_count,
        "segmented_mole": mole_count,
        "initial_feature_construction": initial_count,
        "node_to_edge_transform": edgewise_count,
        "so2_activation": edgewise_count,
        "edge_to_node_aggregation": edgewise_count,
        "equivariant_ffn": ffn_count,
        "rms_norm": norm_count,
    }
