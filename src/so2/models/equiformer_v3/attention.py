"""EqV3 attention execution, per-forward caches, and installation."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Callable, Iterator, Sequence

import torch

from fairchem.experimental.models.equiformer_v3.activation import SmoothLeakyReLU
from LunaEquivariance.so2 import attention_scores
from LunaEquivariance.so2.utils.graph import validate_edge_index
from LunaEquivariance.so2.utils.layout import RetainedRotationLayout, get_retained_layout
from LunaEquivariance.so2.rotate_gather.interface import gather_rotated_sectors
from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
    attention_aggregation,
    build_receiver_csr,
)
from LunaEquivariance.so2.rotation.interface import pack_retained_rotations, pack_inverse_rotations
from .activation import linear_grid_activation, validate_linear_grid_activation
from .linear import project_attention_values
from .radial import grouped_radial_runtime


def receiver_key(receivers: torch.Tensor, num_nodes: int):
    try:
        version = receivers._version
    except RuntimeError:
        return None
    return (
        receivers.data_ptr(),
        receivers.shape[0],
        receivers.stride(0),
        num_nodes,
        receivers.device,
        version,
    )


def prepared_receiver_csr(
    state: dict[str, object],
    receivers: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    key = receiver_key(receivers, num_nodes)
    if key is None or state.get("key") != key:
        state.clear()
        state["key"] = key
        state["receivers"] = receivers
        state["row_ptr"] = build_receiver_csr(receivers, num_nodes)
    return state["row_ptr"]


def reset_forward_caches(
    attention: torch.nn.Module, edge_index=None, num_nodes=None
) -> None:
    if attention.so2_grouped_radial_index == 0:
        state = attention.so2_scatter_state
        pending = state.pop("embedding_csr", None)
        attention.so2_prefix_state.clear()
        state.clear()
        if pending is not None and edge_index is not None:
            graph, key, row_ptr = pending
            receivers = edge_index[1]
            if (
                graph is edge_index
                and key is not None
                and key == receiver_key(receivers, num_nodes)
            ):
                state.update(key=key, receivers=receivers, row_ptr=row_ptr)


def validate_attention(attention: torch.nn.Module) -> None:
    get_retained_layout(
        lmax=attention.lmax,
        mmax=attention.mmax,
        channels=attention.num_in_channels,
    )
    if attention.activation != "sep-merge_gates2_swiglu":
        raise ValueError(
            "the structured EqV3 path requires sep-merge_gates2_swiglu activation"
        )
    if not isinstance(attention.act.grid_drop, torch.nn.Identity):
        raise ValueError("the structured EqV3 activation requires zero value dropout")
    validate_linear_grid_activation(attention.so2_linear_1)
    if attention.so2_linear_1.num_in_channels != 2 * attention.num_in_channels:
        raise ValueError(
            "the structured EqV3 prefix requires concatenated endpoint channels"
        )
    expected = {
        "use_add_merge": False,
        "use_atom_edge_embedding": True,
        "use_rad_l_parametrization": True,
    }
    actual = {name: getattr(attention, name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"the structured prefix requires the EqV3 concat/shared-l path: {actual=}"
        )
    radial = attention.rad_func
    if (
        radial.lmax != attention.lmax
        or radial.mmax != attention.lmax
        or not radial.use_expand
    ):
        raise ValueError("unexpected radial-function contract")


def shared_compact_wigner(
    attention: torch.nn.Module,
    spec: RetainedRotationLayout,
) -> torch.Tensor:
    """Pack one model-forward Wigner carrier once for all attention blocks."""

    state = attention.so2_prefix_state
    wigner = attention.so3_rotation.wigner
    if tuple(wigner.shape[1:]) == (spec.compact_elements,):
        return wigner
    key = (wigner.data_ptr(), tuple(wigner.shape), spec.lmax, spec.mmax)
    if state.get("wigner_key") != key:
        state.clear()
        state["wigner_key"] = key
        state["compact_wigner"] = pack_retained_rotations(
            wigner,
            lmax=spec.lmax, mmax=spec.mmax,
            input_layout="reduced_rows",
        )
    return state["compact_wigner"]


def shared_compact_inverse_wigner(
    attention: torch.nn.Module,
    spec: RetainedRotationLayout,
) -> torch.Tensor:
    """Pack one inverse Wigner carrier once for all attention blocks."""

    state = attention.so2_prefix_state
    wigner_inv = attention.so3_rotation.wigner_inv
    if tuple(wigner_inv.shape[1:]) == (spec.compact_elements,):
        return wigner_inv
    key = (wigner_inv.data_ptr(), tuple(wigner_inv.shape), spec.lmax, spec.mmax)
    if state.get("wigner_inv_key") != key:
        state["wigner_inv_key"] = key
        state["compact_wigner_inv"] = pack_inverse_rotations(
            wigner_inv,
            lmax=spec.lmax, mmax=spec.mmax,
        )
    return state["compact_wigner_inv"]


@torch.compiler.disable
def runtime_attention_forward(
    self,
    x,
    source_atomic_numbers,
    target_atomic_numbers,
    edge_distance,
    edge_index,
    edge_envelope_weight=None,
):
    try:
        num_nodes = x.shape[0]
        reset_forward_caches(self, edge_index, num_nodes)
        if self.so2_grouped_radial_index == 0:
            validate_edge_index(edge_index, num_nodes)
        spec = get_retained_layout(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.num_in_channels,
        )
        compact_radial = self.so2_grouped_radial.consume(
            self.so2_grouped_radial_index,
            edge_distance,
            source_atomic_numbers,
            target_atomic_numbers,
        )
        compact_radial = compact_radial.view(-1, self.lmax + 1, 2 * self.num_in_channels)
        prefix_sectors = gather_rotated_sectors(
            x.to(compact_radial.dtype), edge_index, compact_radial.contiguous(),
            shared_compact_wigner(self, spec),
            lmax=spec.lmax, mmax=spec.mmax,
            radial_sharing="degree",
            rotation_storage="retained_rows",
        )
        if self.activation != "sep-merge_gates2_swiglu":
            raise ValueError("SO(2) optimization requires the gated spherical-grid activation")
        activated_sectors, x_alpha = linear_grid_activation(
            self.so2_linear_1, prefix_sectors,
            self.act.so3_grid.to_grid_mat,
            self.act.so3_grid.from_grid_mat,
        )
        # Release temporary edge buffers before the next projection.
        del prefix_sectors
        x_message = project_attention_values(self.so2_linear_2, activated_sectors)
        x_alpha = x_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
        alpha = attention_scores(
            x_alpha,
            self.alpha_norm.weight, self.alpha_norm.bias,
            self.alpha_dot,
            eps=self.alpha_norm.eps,
        )
        if edge_envelope_weight is None:
            raise ValueError("the receiver-owned tail requires an edge envelope")
        if torch.is_autocast_enabled():
            raise ValueError("the receiver-owned tail currently requires FP32")
        receivers = edge_index[1].contiguous()
        row_ptr = prepared_receiver_csr(self.so2_scatter_state, receivers, num_nodes)
        x_message = attention_aggregation(
            x_message,
            alpha.contiguous(),
            edge_envelope_weight.contiguous(),
            shared_compact_inverse_wigner(self, spec),
            receivers, row_ptr, num_nodes, self.attn_softmax.eps,
            lmax=spec.lmax, mmax=spec.mmax,
        )
        output = self.proj(x_message)
        if self.so2_grouped_radial_index == self.so2_grouped_radial.block_count - 1:
            self.so2_scatter_state.clear()
        return output
    except Exception:
        self.so2_prefix_state.clear()
        self.so2_scatter_state.clear()
        radial = getattr(self, "so2_grouped_radial", None)
        if radial is not None:
            radial.reset()
        raise


def validate_scatter_schedule(attention: torch.nn.Module) -> None:
    if attention.num_heads <= 0 or attention.attn_value_channels <= 0:
        raise ValueError(
            "the receiver-owned scatter schedule currently requires "
            "positive head and per-head value dimensions"
        )
    if not isinstance(attention.attn_softmax.softcap, torch.nn.Identity):
        raise ValueError("the receiver-owned scatter schedule requires no softcap")
    if not isinstance(attention.attn_softmax.dropout, torch.nn.Identity):
        raise ValueError(
            "the receiver-owned scatter schedule requires zero exponent dropout"
        )
    if not isinstance(attention.attn_weights_dropout, torch.nn.Identity):
        raise ValueError(
            "the receiver-owned scatter schedule requires zero attention dropout"
        )
    if (
        not isinstance(attention.alpha_norm, torch.nn.LayerNorm)
        or not attention.alpha_norm.elementwise_affine
        or attention.alpha_norm.bias is None
        or attention.alpha_norm.normalized_shape != (attention.attn_alpha_channels,)
    ):
        raise ValueError(
            "the fused alpha projection requires affine LayerNorm matching alpha channels"
        )
    if (
        not isinstance(attention.alpha_act, SmoothLeakyReLU)
        or attention.alpha_act.alpha != 0.2
        or not isinstance(attention.alpha_dropout, torch.nn.Identity)
    ):
        raise ValueError(
            "the fused alpha projection requires SmoothLeakyReLU(0.2) and zero dropout"
        )


def install_attention(
    model: torch.nn.Module,
    block_indices: Sequence[int],
) -> Callable[[], None]:
    """Install streaming gather, activation and receiver reduction."""

    validate_checkpointing(model)
    indices = tuple(block_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("block indices must be a non-empty unique sequence")
    if min(indices) < 0 or max(indices) >= len(model.blocks):
        raise ValueError(f"block index outside [0, {len(model.blocks)})")
    attentions = [model.blocks[index].ga for index in indices]
    grouped_radial = grouped_radial_runtime(attentions)
    prefix_state: dict[str, object] = {}
    scatter_state: dict[str, object] = {}
    for attention in attentions:
        validate_attention(attention)
        validate_scatter_schedule(attention)
        if attention.so3_rotation is not model.so3_rotation:
            raise ValueError("attention must share the model SO3Rotation")
        if hasattr(attention, "so2_prefix_state") or hasattr(attention, "so2_scatter_state"):
            raise RuntimeError("runtime attention state is already installed")
        if hasattr(attention, "so2_grouped_radial"):
            raise RuntimeError("grouped radial state is already installed")
    edge_degree = getattr(model, "edge_degree_embedding", None)
    missing = object()
    old_embedding_state = (
        getattr(edge_degree, "so2_scatter_state", missing)
        if edge_degree is not None
        else missing
    )
    if edge_degree is not None:
        edge_degree.so2_scatter_state = scatter_state
    old_forwards = [attention.__dict__.get("forward") for attention in attentions]
    old_grouped_indices = [
        attention.__dict__.get("so2_grouped_radial_index") for attention in attentions
    ]
    for radial_index, attention in enumerate(attentions):
        attention.so2_prefix_state = prefix_state
        attention.so2_scatter_state = scatter_state
        attention.so2_grouped_radial = grouped_radial
        attention.so2_grouped_radial_index = radial_index
        attention.forward = MethodType(runtime_attention_forward, attention)

    def restore():
        prefix_state.clear()
        scatter_state.clear()
        if edge_degree is not None:
            if old_embedding_state is missing:
                edge_degree.__dict__.pop("so2_scatter_state", None)
            else:
                edge_degree.so2_scatter_state = old_embedding_state
        for attention, old_forward, old_grouped_index in zip(
            attentions, old_forwards, old_grouped_indices, strict=True,
        ):
            if old_forward is None:
                del attention.forward
            else:
                attention.forward = old_forward
            # Installation requires these three attributes to be absent.
            del attention.so2_prefix_state
            del attention.so2_scatter_state
            del attention.so2_grouped_radial
            if old_grouped_index is None:
                del attention.so2_grouped_radial_index
            else:
                attention.so2_grouped_radial_index = old_grouped_index

    return restore


def validate_checkpointing(model):
    # Upstream stores one 0/1 switch per layer, not a list of layer indices.
    if any(getattr(model, "gradient_checkpointing_block_list", None) or ()):
        raise ValueError(
            "the EqV3 operator does not support gradient_checkpointing_block_list"
        )


@contextmanager
def runtime_attention_blocks(
    model: torch.nn.Module, block_indices: Sequence[int]
) -> Iterator[None]:
    restore = install_attention(model, block_indices)
    try:
        yield
    finally:
        restore()
