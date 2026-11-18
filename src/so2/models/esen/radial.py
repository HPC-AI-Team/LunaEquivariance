from types import MethodType

import torch

from ..radial import (
    parameters_from_module,
    validate_radial_mlp,
)
from LunaEquivariance.so2.radial_network.interface import RadialMLPRuntime


class UnifiedRadials:
    """Own this model's radial modules while keeping tensor caches transient."""

    def __init__(
        self, radial_mlps, *, checkpoint_hidden, native_eval_grad, forward_num_warps
    ):
        self.radial_mlps = tuple(radial_mlps)
        self.checkpoint_hidden = checkpoint_hidden
        self.native_eval_grad = native_eval_grad
        self.runtime = RadialMLPRuntime(
            len(radial_mlps), forward_num_warps=forward_num_warps
        )

    def reset(self):
        self.runtime.reset()


def unified_radial_forward(module, x_edge, *, keep_hidden=False):
    owner = module.so2_unified_radials
    if owner.native_eval_grad and torch.is_grad_enabled() and not module.training:
        return module.net(x_edge)
    index = module.so2_radial_index
    parameters = (
        tuple(parameters_from_module(value) for value in owner.radial_mlps) if index == 0 else None
    )
    options = {"keep_hidden": True} if keep_hidden else {}
    return owner.runtime.consume(
        index,
        x_edge,
        parameters=parameters,
        checkpoint_hidden=owner.checkpoint_hidden and not owner.radial_mlps[0].training,
        **options,
    )


def install_unified_radials(
    edgewise_modules,
    *,
    checkpoint_hidden=True,
    native_eval_grad=False,
    forward_num_warps=8,
) -> int:
    radial_mlps = tuple(module.so2_conv_1.rad_func for module in edgewise_modules)
    if not radial_mlps:
        raise ValueError("at least one radial MLP is required")
    reference = radial_mlps[0]
    for radial_mlp in radial_mlps:
        validate_radial_mlp(radial_mlp)
        if any(radial_mlp.net[i].eps != reference.net[i].eps for i in (1, 4)):
            raise ValueError("batched radial normalizations must share epsilon")
        for layer_index in (0, 3, 6):
            layer = radial_mlp.net[layer_index]
            reference_layer = reference.net[layer_index]
            if (
                layer.in_features != reference_layer.in_features
                or layer.out_features != reference_layer.out_features
            ):
                raise ValueError("radial MLP dimensions are not uniform")
    owner = UnifiedRadials(
        radial_mlps,
        checkpoint_hidden=checkpoint_hidden,
        native_eval_grad=native_eval_grad,
        forward_num_warps=forward_num_warps,
    )

    for index, radial_mlp in enumerate(radial_mlps):
        radial_mlp.so2_unified_radials = owner
        radial_mlp.so2_radial_index = index
        radial_mlp.forward = MethodType(unified_radial_forward, radial_mlp)
    return len(radial_mlps)
