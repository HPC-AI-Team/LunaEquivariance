"""eSEN operator integration and installation."""

from __future__ import annotations

from types import MethodType
from weakref import ref

import torch
from torch.nn.modules import module as nn_module

from LunaEquivariance.so2.recombine_activate import interface as activation
from LunaEquivariance.so2.recombine_activate.interface import complex_recombination, gated_activation
from LunaEquivariance.so2.linear import interface as projection
from LunaEquivariance.so2.linear.interface import (
    flattened_so2_linear,
    project_sectors,
    raw_sector_linear,
)
from LunaEquivariance.so2.rotate_gather import interface as gather
from LunaEquivariance.so2.inverse_rotate_reduce.interface import sum_aggregation
from LunaEquivariance.so2.utils.graph import GraphCSRRuntime
from .radial import install_unified_radials, unified_radial_forward
from ..spectral import rms_norm_forward

ESEN_OPERATOR_COMPONENTS = (
    "edgewise",
    "edge_embedding",
    "node_gates",
    "rms_norms",
    "complex",
    "sparse_wigner",
)
project_reduced_sectors = getattr(projection, "project_reduced_sectors", None)
frozen_complex_projection = getattr(projection, "FrozenComplexProjection", None)
three_product_gate = getattr(activation, "ThreeProductGate", None)
packed_signed_weight = getattr(projection, "PackedSignedWeight", None)
frozen_radial_gather = getattr(gather, "FrozenRadialGather", None)
rotate_gather_order_degree = gather.rotate_gather_order_degree


class EdgewiseRuntime(GraphCSRRuntime):
    def __init__(self, block_count):
        from fairchem.core.models.esen.nn.radial import PolynomialEnvelope

        self.envelope_type = PolynomialEnvelope
        self.envelope_forward = PolynomialEnvelope.forward
        super().__init__(block_count)

    def reset(self):
        super().reset()
        self.receiver_geometry = None

    def receiver_rotation(self, envelope, distance, cutoff, rotation):
        key = None
        if (
            type(envelope) is self.envelope_type
            and self.envelope_type.forward is self.envelope_forward and "forward" not in vars(envelope)
            and not (
                envelope._forward_pre_hooks or envelope._forward_hooks
                or envelope._backward_pre_hooks or envelope._backward_hooks
            )
            and not (distance.is_inference() or rotation.is_inference())
        ):
            coefficients = cutoff, envelope.p, envelope.a, envelope.b, envelope.c
            if all(type(value) in (int, float) for value in coefficients):
                key = coefficients, distance._version, rotation._version, torch.is_grad_enabled()
                saved = self.receiver_geometry
                if saved is not None:
                    saved_distance, saved_rotation, saved_key, saved_weighted = saved
                    if distance is saved_distance() and rotation is saved_rotation() and key == saved_key:
                        weighted = saved_weighted()
                        if weighted is not None:
                            return weighted

        weighted = rotation * envelope(distance / cutoff).reshape(-1, 1)
        # The receiver's autograd graph owns the tensor. The runtime only
        # borrows it so an interrupted model forward can release the graph.
        self.receiver_geometry = (ref(distance), ref(rotation), key, ref(weighted)) if key is not None else None
        return weighted


def plain_linear_maps(layers):
    if (
        nn_module._global_forward_pre_hooks or nn_module._global_forward_hooks
        or nn_module._global_backward_pre_hooks or nn_module._global_backward_hooks
    ):
        return False
    return all(
        type(layer) is torch.nn.Linear and "forward" not in vars(layer)
        and not (
            layer._forward_pre_hooks or layer._forward_hooks
            or layer._backward_pre_hooks or layer._backward_hooks
        )
        for layer in layers
    )


def use_receiver_projection(module, x, wigner, node_offset, layers):
    nodes = x.shape[0]
    edges = wigner.shape[0]
    return (
        not torch.compiler.is_compiling()
        and node_offset == 0
        and (module.lmax, module.mmax, x.shape[-1], module.so2_conv_2.m_output_channels)
        == (3, 2, 128, 128)
        and tuple(layer.weight.shape for layer in layers) == ((512, 512), (768, 384), (512, 256))
        and layers[0].bias is not None
        and layers[0].bias.shape == (512,)
        and all(layer.bias is None for layer in layers[1:])
        and 16384 <= edges <= 32768
        and 24 * nodes <= edges <= 40 * nodes
        and wigner.shape == (edges, 70)
        and wigner.is_contiguous()
        and x.is_cuda and x.dtype == wigner.dtype == torch.float32
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.is_autocast_enabled("cuda")
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))
        and (module.training or not any(parameter.requires_grad for parameter in module.parameters()))
        and plain_linear_maps(layers)
    )


def use_frozen_gather(module, x, edge_index, wigner):
    if (
        frozen_radial_gather is None or module.training or not torch.is_grad_enabled()
        or torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    radial = module.so2_conv_1.rad_func
    edges = edge_index.shape[1]
    return (
        (module.lmax, module.mmax, x.shape[-1]) == (3, 2, 128)
        and 16384 <= edges <= 32768 and 24 * x.shape[0] <= edges <= 40 * x.shape[0]
        and wigner.shape == (edges, 70) and x.is_cuda and torch.version.hip is None
        and x.dtype == wigner.dtype == torch.float32
        and all(t.is_contiguous() and not t.is_neg() for t in (x, edge_index, wigner))
        and torch.cuda.get_device_capability(x.device) == (9, 0)
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and getattr(radial.forward, "__func__", None) is unified_radial_forward
        and not radial.so2_unified_radials.native_eval_grad
        and not (radial._forward_pre_hooks or radial._forward_hooks
                 or radial._backward_pre_hooks or radial._backward_hooks)
        and not (nn_module._global_forward_pre_hooks or nn_module._global_forward_hooks
                 or nn_module._global_backward_pre_hooks or nn_module._global_backward_hooks)
        and not any(parameter.requires_grad for parameter in module.parameters())
    )


def gather_project_and_gate(module, x, edge_index, radial, wigner, metadata, *, frozen_gather=False):
    if isinstance(radial, tuple):
        sectors = frozen_radial_gather.apply(
            x, edge_index, *radial, wigner, module.lmax, module.mmax,
        )
    else:
        sectors = rotate_gather_order_degree(
            x, edge_index, radial, wigner,
            lmax=module.lmax, mmax=module.mmax, metadata=metadata, sectorized=True,
            derivative_schedule="fused_atomic" if frozen_gather else "split_csr",
        )
    first = module.so2_conv_1
    scalar, *orders = layers = first.fc_m0, *(layer.fc for layer in first.so2_m_conv)
    features = sectors[0]
    channels, prefix = first.m_output_channels, first.extra_m0_output_channels
    coefficient_shape = module.lmax + 1, channels
    scalar_sizes = prefix, coefficient_shape[0] * channels
    if (
        packed_signed_weight is not None
        and module.training and torch.is_grad_enabled() and not torch.compiler.is_compiling()
        and (module.lmax, module.mmax, channels, prefix) == (3, 2, 128, 384)
        and 16384 <= features.shape[0] <= 32768 and features.dtype == torch.float32
        and features.is_cuda and torch.version.hip is None
        and torch.cuda.get_device_capability(features.device) in ((8, 0), (9, 0))
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and tuple(layer.weight.shape for layer in layers) == ((896, 1024), (768, 768), (512, 512))
        and all(layer.bias is None for layer in orders)
        and plain_linear_maps(layers)
    ):
        output0, *outputs = projection.project_recombined_sectors(
            (features.flatten(1), *sectors[1:]), tuple(layer.weight for layer in layers), scalar.bias,
            pack_weight=packed_signed_weight.apply,
        )
        gates, coefficients = output0.split(scalar_sizes, dim=-1)
        physical = tuple(value.reshape(value.shape[0], -1, channels) for value in outputs)
        return activation.apply_gate(
            gates, coefficients.unflatten(-1, coefficient_shape), physical,
            module.lmax, module.mmax, channels, raw_complex=False,
        )
    if (
        frozen_complex_projection is not None and three_product_gate is not None
        and not module.training and not torch.compiler.is_compiling()
        and (module.lmax, module.mmax, channels, prefix) == (3, 2, 128, 384)
        and 16384 <= features.shape[0] <= 32768 and features.dtype == torch.float32
        and features.is_cuda and torch.version.hip is None
        and torch.cuda.get_device_capability(features.device) == (9, 0)
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and tuple(layer.weight.shape for layer in layers) == ((896, 1024), (768, 768), (512, 512))
        and all(layer.bias is None for layer in orders)
        and not any(parameter.requires_grad for layer in layers for parameter in layer.parameters())
        and plain_linear_maps(layers)
    ):
        # A frozen projection's input adjoint only needs its weights.
        features, first_order, second_order = sectors
        del sectors
        gates, coefficients = scalar(features.flatten(1)).split(scalar_sizes, dim=-1)
        del features
        products = frozen_complex_projection.apply(first_order, orders[0].weight)
        del first_order
        products += frozen_complex_projection.apply(second_order, orders[1].weight)
        del second_order
        return three_product_gate.apply(
            gates, coefficients.unflatten(-1, coefficient_shape),
            module.lmax, module.mmax, *products,
        )
    output0, *outputs = raw_sector_linear(sectors, scalar, tuple(orders))
    gates, coefficients = output0.split(scalar_sizes, dim=-1)
    return gated_activation(
        gates, coefficients.unflatten(-1, coefficient_shape), outputs,
        lmax=module.lmax, mmax=module.mmax,
    )


def structured_edgewise_forward(
    module, x, x_edge, edge_distance, edge_index, wigner, wigner_inv,
    node_offset: int = 0,
):
    try:
        first = module.so2_conv_1
        csr_metadata = module.csr_runtime.consume(
            module.csr_index, edge_index, x.shape[0], node_offset
        )
        frozen_gather = use_frozen_gather(module, x, edge_index, wigner)
        # Early blocks run last in the geometry backward, after the later blocks' state is released.
        keep_hidden = frozen_gather and module.csr_index < 4
        radial = first.rad_func(x_edge, keep_hidden=True) if keep_hidden else first.rad_func(x_edge)
        activated = gather_project_and_gate(
            module, x, edge_index, radial, wigner, csr_metadata, frozen_gather=frozen_gather,
        )

        second = module.so2_conv_2
        if not module.use_envelope:
            raise RuntimeError("the structured eSEN tail requires the model envelope")
        order_layers = tuple(layer.fc for layer in second.so2_m_conv)
        if project_reduced_sectors is not None and use_receiver_projection(
            module, x, wigner, node_offset, (second.fc_m0, *order_layers),
        ):
            rotation = module.csr_runtime.receiver_rotation(
                module.envelope, edge_distance, module.cutoff, wigner,
            )
            output = project_reduced_sectors(
                activated, rotation,
                edge_index[1].contiguous(), csr_metadata[2],
                (second.fc_m0.weight, *(layer.weight for layer in order_layers)),
                second.fc_m0.bias,
                lmax=module.lmax, mmax=module.mmax,
            )
        else:
            _, second_outputs = project_sectors(
                activated,
                second.fc_m0,
                order_layers,
                output_channels=second.m_output_channels,
            )
            envelope = module.envelope(edge_distance / module.cutoff)
            output = sum_aggregation(
                second_outputs, envelope, wigner,
                edge_index[1] - node_offset, x.shape[0],
                lmax=module.lmax, mmax=module.mmax,
                row_ptr=csr_metadata[2],
                raw_complex=True,
                rotation_storage="retained_rows",
            )
        module.csr_runtime.finish(module.csr_index)
        return output
    except Exception:
        module.csr_runtime.reset()
        module.so2_conv_1.rad_func.so2_unified_radials.reset()
        raise


def install_structured_edgewise(model: torch.nn.Module) -> int:
    from fairchem.core.models.esen.esen_block import Edgewise

    edgewise_modules = tuple(
        child for child in model.modules() if isinstance(child, Edgewise)
    )
    if getattr(model.backbone, "activation_checkpointing", False):
        raise ValueError(
            "the eSEN structured edge path does not support activation_checkpointing"
        )
    for child in edgewise_modules:
        validate_structured_edgewise(child)
    csr_runtime = EdgewiseRuntime(len(edgewise_modules))
    for index, child in enumerate(edgewise_modules):
        child.forward = MethodType(structured_edgewise_forward, child)
        child.csr_runtime = csr_runtime
        child.csr_index = index
    if not edgewise_modules:
        raise RuntimeError("no eSEN Edgewise modules were found")
    install_unified_radials(edgewise_modules)
    return len(edgewise_modules)


def validate_structured_edgewise(module):
    if module.act_type != "gate":
        raise ValueError("the eSEN structured edge path requires act_type='gate'")
    if (
        type(module.act.scalar_act) is not torch.nn.SiLU
        or type(module.act.gate_act) is not torch.nn.Sigmoid
    ):
        raise ValueError(
            "the eSEN structured edge path requires SiLU scalars and sigmoid gates"
        )
    if not module.use_envelope:
        raise ValueError("the eSEN structured edge path requires use_envelope=True")


def fused_so2_m_conv_forward(module, x_m):
    return complex_recombination(
        flattened_so2_linear(module.fc, x_m), module.out_channels_half
    )


def install_fused_so2_complex(model: torch.nn.Module) -> int:
    from fairchem.core.models.esen.nn.so2_layers import SO2_m_Conv

    installed = 0
    for module in model.modules():
        if isinstance(module, SO2_m_Conv):
            module.forward = MethodType(fused_so2_m_conv_forward, module)
            installed += 1
    if installed == 0:
        raise RuntimeError("no eSEN nonzero-m SO2 convolutions were found")
    return installed


def install_fused_rms_norms(model: torch.nn.Module) -> int:
    from fairchem.core.models.esen.nn.layer_norm import (
        EquivariantRMSNormArraySphericalHarmonicsV2,
    )

    installed = 0
    for module in model.modules():
        if isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2):
            if not (
                module.affine
                and module.centering
                and module.std_balance_degrees
                and module.normalization == "component"
            ):
                raise RuntimeError("unsupported eSEN RMSNorm configuration")
            module.forward = MethodType(rms_norm_forward, module)
            installed += 1
    if installed == 0:
        raise RuntimeError("no eSEN equivariant RMSNorm modules were found")
    return installed


def install_esen_operator(
    model: torch.nn.Module,
    *,
    components: tuple[str, ...] = ESEN_OPERATOR_COMPONENTS,
    official_wigner: bool = False,
) -> dict[str, int | str]:
    selected = tuple(components)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"invalid eSEN operator component list: {selected}")
    unknown = set(selected) - set(ESEN_OPERATOR_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown eSEN operator components: {sorted(unknown)}")

    installed: dict[str, int | str] = {}
    if "edgewise" in selected and getattr(
        model.backbone, "activation_checkpointing", False
    ):
        raise ValueError(
            "the eSEN structured edge path does not support activation_checkpointing"
        )
    if (
        "node_gates" in selected
        and getattr(model.backbone, "mlp_type", "spectral") != "spectral"
    ):
        raise ValueError(
            "the eSEN operator requires mlp_type='spectral'; grid FFNs are unsupported"
        )
    if "sparse_wigner" in selected:
        from .rotation import install_sparse_wigner

        installed["sparse_wigner"] = install_sparse_wigner(model, official=official_wigner)
    if "edgewise" in selected:
        installed["structured_edgewise_blocks"] = install_structured_edgewise(model)
    if "node_gates" in selected:
        from .ffn import (
            install_coefficient_major_ffns,
        )
        from .gate import (
            install_fused_gates,
        )

        installed["fused_gates"] = install_fused_gates(model)
        installed["coefficient_major_ffns"] = install_coefficient_major_ffns(model)
    if "rms_norms" in selected:
        installed["fused_rms_norms"] = install_fused_rms_norms(model)
    if "complex" in selected:
        installed["complex_recombination"] = install_fused_so2_complex(model)
    if "edge_embedding" in selected:
        from .embedding import (
            install_structured_edge_embedding,
        )

        installed["structured_edge_embedding"] = install_structured_edge_embedding(
            model,
            receiver_owned="edgewise" in selected,
        )
    return installed
