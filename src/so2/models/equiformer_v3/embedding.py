"""EqV3 edge-degree and initial node embedding adapters."""

from __future__ import annotations

from types import MethodType

import torch

from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
    build_receiver_csr,
    fused_initial_atom,
    scalar_edge_to_node,
)
from LunaEquivariance.so2.utils.layout import get_retained_layout
from LunaEquivariance.so2.rotation.interface import pack_inverse_rotations
from .radial import radial_from_module


@torch.compiler.disable
def compact_edge_degree_forward(
    self,
    atomic_numbers,
    edge_distance,
    edge_index,
    edge_envelope_weight=None,
):
    state = getattr(self, "so2_scatter_state", None)
    if state is not None:
        state.clear()
    if edge_envelope_weight is None:
        raise ValueError("compact edge degree requires an envelope")
    spec = get_retained_layout(
        lmax=self.lmax, mmax=self.mmax, channels=self.num_channels
    )
    receivers = edge_index[1].contiguous()
    row_ptr = build_receiver_csr(receivers, atomic_numbers.shape[0])
    wigner_inverse = self.so3_rotation.wigner_inv
    if tuple(wigner_inverse.shape[1:]) != (spec.compact_elements,):
        wigner_inverse = pack_inverse_rotations(
            wigner_inverse,
            lmax=spec.lmax,
            mmax=spec.mmax,
        )
    radial = radial_from_module(
        self.rad_func.net,
        edge_distance,
        atomic_numbers.contiguous(),
        edge_index.contiguous(),
        self.source_embedding,
        self.target_embedding,
    )
    radial = radial.view(edge_index.shape[1], self.lmax + 1, self.num_channels)
    arguments = (
        radial.contiguous(),
        edge_envelope_weight.contiguous(),
        wigner_inverse.contiguous(),
        receivers,
        row_ptr,
        self.rescale_factor,
    )
    result = scalar_edge_to_node(
        *arguments,
        lmax=spec.lmax,
        mmax=spec.mmax,
    )
    if state is not None:
        from .attention import receiver_key

        key = receiver_key(receivers, atomic_numbers.shape[0])
        if key is not None:
            state["embedding_csr"] = (edge_index, key, row_ptr)
    return result


def runtime_forward_embedding(
    self,
    atomic_numbers,
    edge_distance,
    edge_index,
    edge_envelope_weight,
):
    try:
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        )
        return fused_initial_atom(
            edge_degree,
            self.sphere_embedding.weight,
            atomic_numbers.contiguous(),
        )
    except Exception:
        state = getattr(self.edge_degree_embedding, "so2_scatter_state", None)
        if state is not None:
            state.clear()
        raise


def install_runtime_embedding(model, edge_degree):
    edge_degree.forward = MethodType(compact_edge_degree_forward, edge_degree)
    model._forward_embedding = MethodType(runtime_forward_embedding, model)


def restore_runtime_embedding(model, edge_degree, state):
    for owner, name in ((edge_degree, "forward"), (model, "_forward_embedding")):
        if name in state:
            setattr(owner, name, state[name])
        else:
            owner.__dict__.pop(name, None)
