"""Physical-sparse eSEN initial edge-degree embedding."""

from __future__ import annotations

from types import MethodType

import torch

from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
    fused_initial_scalar,
    receiver_row_ptr,
    scalar_edge_to_node,
    structured_initial_sector_tail,
    structured_sector_tail,
)
from ..radial import (
    fused_radial_mlp_from_module as fused_radial_mlp,
    validate_radial_mlp,
)


def structured_edge_degree_forward(
    module,
    x,
    x_edge,
    edge_distance,
    edge_index,
    wigner_inv,
    node_offset=0,
):
    embedding = structured_edge_degree_embedding(
        module,
        x_edge,
        edge_distance,
        edge_index,
        wigner_inv,
        node_offset=node_offset,
        nodes=x.shape[0],
    )
    return fused_initial_scalar(embedding, x[:, 0, :].contiguous())


def structured_edge_degree_embedding(
    module,
    x_edge,
    edge_distance,
    edge_index,
    wigner_inv,
    *,
    node_offset=0,
    nodes,
):
    m0, envelope = edge_degree_inputs(module, x_edge, edge_distance)
    embedding = structured_sector_tail(
        (m0,),
        envelope,
        wigner_inv,
        edge_index[1] - node_offset,
        nodes,
        lmax=module.lmax,
        mmax=0,
        wigner_layout="retained_rows",
        wigner_mmax=module.so2_rotation_mmax,
    )
    return embedding


def structured_initial_edge_degree_embedding(
    module,
    x_edge,
    edge_distance,
    edge_index,
    wigner_inv,
    atom_weight,
    atomic_numbers,
    *,
    node_offset=0,
):
    m0, envelope = edge_degree_inputs(module, x_edge, edge_distance)
    return structured_initial_sector_tail(
        (m0,),
        envelope,
        wigner_inv,
        edge_index[1] - node_offset,
        atom_weight,
        atomic_numbers,
        atomic_numbers.shape[0],
        lmax=module.lmax,
        mmax=0,
        wigner_layout="retained_rows",
        wigner_mmax=module.so2_rotation_mmax,
    )


def edge_degree_inputs(module, x_edge, edge_distance):
    m0 = fused_radial_mlp(module.rad_func, x_edge).reshape(
        x_edge.shape[0], module.m_0_num_coefficients, module.sphere_channels
    )
    if not module.use_envelope:
        raise RuntimeError("the structured edge embedding requires the model envelope")
    envelope = module.envelope(edge_distance / module.cutoff) / module.rescale_factor
    return m0, envelope


def receiver_edge_degree_forward(
    module,
    x,
    x_edge,
    edge_distance,
    edge_index,
    wigner_inv,
    node_offset=0,
):
    """Bound model path: reuse the native sorted-receiver graph contract."""
    radial, envelope = edge_degree_inputs(module, x_edge, edge_distance)
    receivers = (edge_index[1] - node_offset).contiguous()
    row_ptr = receiver_row_ptr(receivers, x.shape[0])
    embedding = scalar_edge_to_node(
        radial.contiguous(),
        envelope.contiguous(),
        wigner_inv.contiguous(),
        receivers,
        row_ptr,
        1.0,
        lmax=module.lmax,
        mmax=module.so2_rotation_mmax,
    )
    return fused_initial_scalar(embedding, x[:, 0, :].contiguous())


def install_structured_edge_embedding(
    model: torch.nn.Module,
    *,
    receiver_owned: bool = False,
) -> int:
    """Use receiver-owned reduction only for an explicitly sorted graph contract."""
    from fairchem.core.models.esen.nn.embedding import EdgeDegreeEmbedding

    installed = 0
    for module in model.modules():
        if isinstance(module, EdgeDegreeEmbedding):
            if not module.use_envelope:
                raise ValueError("the eSEN edge embedding requires use_envelope=True")
            validate_radial_mlp(module.rad_func)
            module.so2_rotation_mmax = model.backbone.mmax
            module.forward = MethodType(
                receiver_edge_degree_forward
                if receiver_owned
                else structured_edge_degree_forward,
                module,
            )
            installed += 1
    if installed != 1:
        raise RuntimeError(
            f"expected one eSEN edge-degree embedding, found {installed}"
        )
    return installed
