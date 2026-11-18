"""Bind UMA edge convolutions to common dataflow operators."""

from __future__ import annotations
from functools import partial
from itertools import accumulate
from types import MethodType
import torch
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend
from LunaEquivariance.so2.segmented_linear.kernels.utils import grouped_device
from LunaEquivariance.so2.linear.interface import (
    gather_project,
    project_after_inverse_reduce,
    project_sectors,
)
from LunaEquivariance.so2.recombine_activate import recombine_activate
from LunaEquivariance.so2.utils.graph import GraphCSRRuntime
from LunaEquivariance.so2.inverse_rotate_reduce.interface import sum_aggregation
from .mole import use_gate_cotangent
from .radial import layer_radial_forward, layer_radial_inputs, shared_radial_layers


class UMAOperatorBackend(ExecutionBackend):
    """Keep retained degree rows for the structured Triton edge path."""

    @staticmethod
    def prepare_wigner(wigner, wigner_inv, mapping_reduced, coefficient_index):
        del mapping_reduced, coefficient_index
        return wigner, wigner_inv

    @staticmethod
    def get_layer_radial_emb(x_edge, model):
        return layer_radial_inputs(x_edge, model, model.operator_shared_radials)


class MoLENodeSegments:
    value = None


def mole_node_segments(backbone, batch):
    if (
        not backbone.training or backbone.regress_config.direct_forces
        or not (backbone.regress_config.forces or backbone.regress_config.stress)
        or not torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
    ):
        return
    if batch is None or not hasattr(batch, "get_batch_stats"):
        return
    positions = batch.pos
    if (
        not positions.is_cuda or positions.dtype != torch.float32
        or torch.get_float32_matmul_precision() != "highest"
        or not grouped_device(positions.device)
    ):
        return
    sizes = batch.get_batch_stats()[3]
    if not sizes or not 1 <= len(sizes) <= 64 or min(sizes) <= 0:
        return
    if batch.natoms.numel() != len(sizes) or sum(sizes) != positions.shape[0]:
        return
    offsets = torch.tensor((0, *accumulate(sizes)), dtype=torch.int32, pin_memory=True)
    offsets = offsets.to(positions.device, non_blocking=True)
    return tuple(sizes), offsets


def forward_with_mole_node_segments(backbone, data_dict):
    forward = backbone.operator_original_forward
    if torch.compiler.is_compiling():
        return forward(data_dict)
    segments = backbone.operator_mole_node_segments
    try:
        segments.value = mole_node_segments(backbone, data_dict)
        return forward(data_dict)
    finally:
        segments.value = None


def preprojection_layers(module):
    from fairchem.core.models.uma.nn.mole import MOLE

    second = module.so2_conv_2
    if (module.lmax, module.mmax, second.m_output_channels) != (2, 2, 128):
        return ()
    layers = (second.fc_m0, *(order.fc for order in second.so2_m_conv))
    if len(layers) != 3 or not all(isinstance(layer, MOLE) for layer in layers):
        return ()
    scalar = layers[0]
    if (
        scalar.bias is None or scalar.bias.shape != (384,)
        or any(layer.bias is not None for layer in layers[1:])
        or tuple(tuple(layer.weights.shape[1:]) for layer in layers)
        != ((384, 384), (512, 256), (256, 128))
        or any(layer.global_mole_tensors is not scalar.global_mole_tensors for layer in layers)
        or not all(getattr(layer, "operator_force_training", False) for layer in layers)
    ):
        return ()
    return layers


def mole_raw_linear(linear, inputs, *, gate_map=None):
    """Apply a MoLE linear without flattening the edge dimension."""

    if inputs.ndim != 3 or inputs.shape[1] != 2:
        raise ValueError("nonzero-m MoLE input must have shape [E, 2, K]")
    if linear is gate_map and use_gate_cotangent(linear):
        return linear(inputs, gate_cotangent=True)
    return linear(inputs)


def structured_edgewise_forward(
    module, x, x_edge, edge_index, wigner, wigner_inv_envelope,
    total_atoms_across_gp_ranks,
    scatter_target=None,
    gp_ctx=None,
):
    try:
        from fairchem.core.common import gp_utils

        if gp_utils.initialized() or gp_ctx is not None:
            raise RuntimeError("the UMA structured edge path is single-GPU")
        if module.activation_checkpoint_chunk_size is not None:
            raise RuntimeError(
                "the UMA structured edge path does not use edge chunking"
            )

        metadata = module.csr_runtime.consume(
            module.csr_index, edge_index, x.shape[0], 0
        )
        if scatter_target is None:
            scatter_target = edge_index[1]
        output = module.operator_core(
            x, x_edge, edge_index, wigner, wigner_inv_envelope,
            scatter_target, metadata,
        )
        module.csr_runtime.finish(module.csr_index)
        return output
    except Exception:
        module.csr_runtime.reset()
        raise


def structured_edgewise_core(
    module, x, x_edge, edge_index, wigner, wigner_inv_envelope,
    scatter_target,
    metadata,
):
    first = module.so2_conv_1
    torch._dynamo.mark_static(x, 1)
    torch._dynamo.mark_static(x, 2)
    radial = first.rad_func(x_edge)
    apply_pair = mole_raw_linear
    if module.training and (module.lmax, module.mmax) == (2, 2) and not torch.compiler.is_compiling():
        apply_pair = partial(mole_raw_linear, gate_map=first.so2_m_conv[1].fc)
    raw_outputs = gather_project(
        x, edge_index, radial, wigner, first.fc_m0,
        tuple(layer.fc for layer in first.so2_m_conv), lmax=module.lmax, mmax=module.mmax,
        metadata=metadata, apply_pair=apply_pair,
    )
    activated, _ = recombine_activate(
        raw_outputs, lmax=module.lmax, mmax=module.mmax, activation="gate",
    )

    second = module.so2_conv_2
    if not torch.compiler.is_compiling() and module.operator_preprojection:
        partition = module.operator_mole_node_segments.value
        if (
            partition is not None and scatter_target.numel()
            and scatter_target.data_ptr() == edge_index[1].data_ptr()
        ):
            sizes, offsets = partition
            layers = module.operator_preprojection
            return project_after_inverse_reduce(
                activated, wigner_inv_envelope, scatter_target, metadata[2],
                tuple(layer.weights for layer in layers),
                layers[0].global_mole_tensors.expert_mixing_coefficients,
                layers[0].bias, sizes, offsets,
            )
    _, second_outputs = project_sectors(
        activated,
        second.fc_m0,
        tuple(layer.fc for layer in second.so2_m_conv),
        output_channels=second.m_output_channels,
        apply_pair=mole_raw_linear,
    )

    unit_envelope = torch.ones(x_edge.shape[0], device=x_edge.device, dtype=x_edge.dtype)
    output = sum_aggregation(
        second_outputs, unit_envelope, wigner_inv_envelope,
        scatter_target, x.shape[0], lmax=module.lmax, mmax=module.mmax,
        row_ptr=metadata[2],
        raw_complex=True,
        rotation_storage="retained_rows",
    )
    return output


def install_structured_edgewise(model: torch.nn.Module, *, shared_radial=False) -> int:
    from fairchem.core.models.uma.escn_md_block import Edgewise

    backbone = model.backbone
    edgewise_modules = tuple(
        module for module in backbone.modules() if isinstance(module, Edgewise)
    )
    for module in edgewise_modules:
        validate_structured_edgewise(module)
    backbone.operator_shared_radials = shared_radial_layers(backbone) if shared_radial else ()
    for radial in backbone.operator_shared_radials:
        radial.forward = MethodType(layer_radial_forward, radial)
    backend = UMAOperatorBackend()
    backbone.backend = backend
    backbone.edge_degree_embedding.backend = backend
    runtime = GraphCSRRuntime(len(edgewise_modules))
    segments = MoLENodeSegments()
    uses_node_projection = False
    for index, module in enumerate(edgewise_modules):
        torch._dynamo.mark_static(type(module))
        module.backend = backend
        module.forward = MethodType(structured_edgewise_forward, module)
        module.csr_runtime = runtime
        module.csr_index = index
        module.operator_core = MethodType(structured_edgewise_core, module)
        module.operator_preprojection = preprojection_layers(module)
        module.operator_mole_node_segments = segments
        uses_node_projection |= bool(module.operator_preprojection)
    if not edgewise_modules:
        raise RuntimeError("no UMA Edgewise modules were found")
    if uses_node_projection:
        backbone.operator_mole_node_segments = segments
        if not hasattr(backbone, "operator_original_forward"):
            backbone.operator_original_forward = backbone.forward
            backbone.forward = MethodType(forward_with_mole_node_segments, backbone)
    return len(edgewise_modules)


def validate_structured_edgewise(module):
    first = module.so2_conv_1
    if first.extra_m0_output_channels != module.lmax * first.m_output_channels:
        raise ValueError("the UMA structured edge path requires one gate per degree and channel")
    if module.act_type != "gate":
        raise ValueError("the UMA structured edge path requires act_type='gate'")
    if (
        type(module.act.scalar_act) is not torch.nn.SiLU
        or type(module.act.gate_act) is not torch.nn.Sigmoid
    ):
        raise ValueError(
            "the UMA structured edge path requires SiLU scalars and sigmoid gates"
        )
    if module.activation_checkpoint_chunk_size is not None:
        raise ValueError(
            "the UMA structured edge path does not support activation checkpoint chunking"
        )
