from types import MethodType

import torch
from torch.nn.modules import module as nn_module

from LunaEquivariance.so2.radial_network.interface import unified_radial_hidden
from ..radial import (
    fused_radial_mlp_from_module,
    parameters_from_module,
    validate_radial_mlp,
)


def install_sequential_radials(edgewise_modules) -> int:
    radial_mlps = tuple(module.so2_conv_1.rad_func for module in edgewise_modules)
    for radial_mlp in radial_mlps:
        validate_radial_mlp(radial_mlp)
    for radial_mlp in radial_mlps:
        radial_mlp.forward = MethodType(fused_radial_mlp_from_module, radial_mlp)
    return len(radial_mlps)


def shared_radial_layers(backbone):
    radial_mlps = tuple(block.edge_wise.so2_conv_1.rad_func for block in backbone.blocks)
    if len(radial_mlps) != 4:
        return ()
    for radial in radial_mlps:
        first, second, final = radial.net[0], radial.net[3], radial.net[6]
        widths = tuple((layer.in_features, layer.out_features) for layer in (first, second, final))
        if widths != ((288, 128), (128, 128), (128, 1536)):
            return ()
    return radial_mlps


def layer_radial_forward(radial, inputs):
    if inputs.shape[-1] == 128:
        final = radial.net[6]
        return torch.nn.functional.linear(inputs, final.weight, final.bias)
    return fused_radial_mlp_from_module(radial, inputs)


def layer_radial_inputs(x_edge, model, radial_mlps):
    if (
        radial_mlps and model.training and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and x_edge.is_cuda and x_edge.dtype == torch.float32
        and 8192 <= x_edge.shape[0] <= 32768 and x_edge.shape[-1] == 288
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and torch.version.hip is None and torch.cuda.get_device_capability(x_edge.device) == (9, 0)
        and not (
            nn_module._global_forward_pre_hooks or nn_module._global_forward_hooks
            or nn_module._global_backward_pre_hooks or nn_module._global_backward_hooks
        )
        and all(
            getattr(radial.forward, "__func__", None) is layer_radial_forward
            and not (radial._forward_pre_hooks or radial._forward_hooks
                     or radial._backward_pre_hooks or radial._backward_hooks)
            for radial in radial_mlps
        )
    ):
        parameters = tuple(parameters_from_module(radial) for radial in radial_mlps)
        eps = parameters[0].first_eps, parameters[0].second_eps
        if all((p.first_eps, p.second_eps) == eps for p in parameters):
            return unified_radial_hidden(parameters, x_edge).unbind(0)
    return [x_edge] * len(model.blocks)
