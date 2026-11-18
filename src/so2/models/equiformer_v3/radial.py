"""Grouped first stage for EquiformerV3 edge-conditioned radial MLPs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from LunaEquivariance.so2.radial_network.interface import AtomPairRadialRuntime, packed_radial_net
from ..radial import parameters_from_module, validate_radial_mlp


def radial_layers(attention: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    layers = tuple(attention.rad_func.net)
    expected = (
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.SiLU,
        torch.nn.Linear,
    )
    if tuple(type(layer) for layer in layers) != expected:
        raise ValueError("unsupported EqV3 radial MLP structure")
    return layers


def grouped_radial_runtime(attentions: Sequence[torch.nn.Module]):
    """Bind model parameters to the common grouped radial computation."""
    attentions = tuple(attentions)
    if not attentions:
        raise ValueError("at least one attention block is required")
    for attention in attentions:
        validate_radial_mlp(attention.rad_func)
    first = attentions[0]
    distance_channels = radial_layers(first)[0].in_features - 2 * first.source_embedding.embedding_dim
    return AtomPairRadialRuntime(
        tuple(parameters_from_module(attention.rad_func) for attention in attentions),
        tuple(attention.source_embedding.weight for attention in attentions),
        tuple(attention.target_embedding.weight for attention in attentions),
        distance_channels=distance_channels,
    )


def radial_from_module(
    net: torch.nn.Sequential,
    edge_distance: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    source_embedding: torch.nn.Embedding,
    target_embedding: torch.nn.Embedding,
) -> torch.Tensor:
    return packed_radial_net(
        edge_distance,
        atomic_numbers,
        edge_index,
        source_embedding.weight,
        target_embedding.weight,
        first_weight=net[0].weight,
        first_bias=net[0].bias,
        first_norm_weight=net[1].weight,
        first_norm_bias=net[1].bias,
        first_norm_eps=net[1].eps,
        second_weight=net[3].weight,
        second_bias=net[3].bias,
        second_norm_weight=net[4].weight,
        second_norm_bias=net[4].bias,
        second_norm_eps=net[4].eps,
        output_weight=net[6].weight,
        output_bias=net[6].bias,
    )
