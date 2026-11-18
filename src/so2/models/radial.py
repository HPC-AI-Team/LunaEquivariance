"""Bind the shared seven-layer radial network to external tensor parameters."""

import torch

from LunaEquivariance.so2.radial_network.interface import RadialParameters, radial_mlp_forward


def validate_radial_mlp(radial_mlp):
    layers = tuple(radial_mlp.net)
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
        raise ValueError("radial MLP requires Linear-LayerNorm-SiLU twice, then Linear")
    for linear, norm in ((layers[0], layers[1]), (layers[3], layers[4])):
        if linear.bias is None or norm.weight is None or norm.bias is None:
            raise ValueError(
                "radial hidden layers require biases and affine normalization"
            )
        if norm.normalized_shape != (linear.out_features,):
            raise ValueError("radial normalization must act on the hidden channels")
    if (
        layers[3].in_features != layers[0].out_features
        or layers[6].in_features != layers[3].out_features
    ):
        raise ValueError("radial linear dimensions do not match")


def parameters_from_module(radial_mlp):
    layers = tuple(radial_mlp.net)
    return RadialParameters(
        layers[0].weight, layers[0].bias,
        layers[1].weight, layers[1].bias, layers[1].eps,
        layers[3].weight, layers[3].bias,
        layers[4].weight, layers[4].bias, layers[4].eps,
        layers[6].weight, layers[6].bias,
    )


def fused_radial_mlp_from_module(radial_mlp, x_edge, *, forward_num_warps=8):
    return radial_mlp_forward(
        parameters_from_module(radial_mlp), x_edge, forward_num_warps=forward_num_warps
    )
