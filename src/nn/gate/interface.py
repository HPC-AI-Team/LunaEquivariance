"""Sorted e3nn scalar gates with fused native CUDA gradients."""

from collections import OrderedDict

import torch

from ..activation.kernels.utils import apply_activation_spec, activation_config
from ...o3.irreps import irreps_key
from ..activation.interface import ActivationFunction, register_ops
from ...utils import is_compiling
from .utils import prepare_gate


@torch.compiler.assume_constant_result
def register_module_ops(self):
    return register_ops(self.activation_paths, 32, self.meta)


def gate(features, irreps_scalars=None, act_scalars=None, irreps_gates=None,
         act_gates=None, irreps_gated=None, *, layout=None, kernel_cache=None):
    """Apply scalar gates using e3nn's five Gate configuration arguments."""
    if layout is None:
        layout, cached_kernels = prepare_gate(
            irreps_key(irreps_scalars),
            tuple(map(activation_config, act_scalars)),
            irreps_key(irreps_gates),
            tuple(map(activation_config, act_gates)),
            irreps_key(irreps_gated),
        )
        if kernel_cache is None:
            kernel_cache = cached_kernels
    assert features.shape[-1] == layout["in_dim"], "Gate input feature shape is incorrect"
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        return gate_torch(features, layout)
    if kernel_cache is None:
        kernel_cache = OrderedDict()
    if is_compiling():
        operator_layout = {
            key: layout[key]
            for key in ("in_dim", "out_dim", "scalar_paths", "gated_paths", "scalar_activation", "has_grad")
        }
        name = register_ops(layout["scalar_activation"], 32, operator_layout)
        return getattr(torch.ops.lunaeq, name).default(features)
    if torch.is_inference_mode_enabled():
        with torch.no_grad():
            return ActivationFunction.apply(features, layout["scalar_activation"], 32, kernel_cache, layout)
    return ActivationFunction.apply(features, layout["scalar_activation"], 32, kernel_cache, layout)


def gate_torch(features, layout):
    # Use the same normalized paths and extraction order as the CUDA kernels.
    extracted = []
    for width, spans, whole_input in zip(
        layout["extract_dims"], layout["extract_specs"], layout["extract_identity"],
    ):
        values = features.new_zeros((*features.shape[:-1], width))
        if whole_input:
            values.copy_(features)
        else:
            for span in spans:
                values.narrow(-1, span["output_offset"], span["width"]).copy_(
                    features.narrow(-1, span["input_offset"], span["width"])
                )
        extracted.append(values)

    scalars, gates, gated = extracted
    activated = []
    for values, paths in (
        (scalars, layout["scalar_activation"]), (gates, layout["gate_activation"]),
    ):
        outputs = []
        offset = 0
        for width, spec in paths:
            outputs.append(apply_activation_spec(values.narrow(-1, offset, width), spec))
            offset += width
        activated.append(torch.cat(outputs, dim=-1) if len(outputs) > 1 else
                         outputs[0] if outputs else torch.zeros_like(values))
    scalars, gates = activated
    if not gates.shape[-1]:
        return scalars

    vectors = [
        gated[..., section].reshape(-1, entry.mul, entry.ir.dim)
        for entry, section in zip(layout["tp_irreps1"], layout["tp_irreps1"].slices())
    ]
    weights = [
        gates[..., section].reshape(-1, entry.mul)
        for entry, section in zip(layout["tp_irreps2"], layout["tp_irreps2"].slices())
    ]
    outputs = []
    for vector, scalar in zip(vectors, weights):
        product = torch.einsum("cb,cba->cba", scalar, vector)
        outputs.append(product.flatten(1))
    result = torch.cat(outputs, dim=-1).reshape(
        *features.shape[:-1], layout["out_dim"] - scalars.shape[-1]
    )
    return torch.cat((scalars, result), dim=-1)


__all__ = ['gate']
