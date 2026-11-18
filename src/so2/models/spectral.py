"""Bind eSEN and UMA spectral blocks to tensor operators."""

import torch

from LunaEquivariance.so2 import feed_forward
from LunaEquivariance.so2.utils.degree_norm import degree_norm


def rms_norm_forward(module, node_input):
    return degree_norm(
        node_input,
        module.affine_weight,
        module.affine_bias,
        lmax=module.lmax,
        epsilon=module.eps,
        coefficient_major_output=getattr(module, "coefficient_major_output", False),
    )


def validate_spectral_ffn(module):
    if tuple(type(layer) for layer in module.scalar_mlp) != (
        torch.nn.Linear,
        torch.nn.SiLU,
    ):
        raise ValueError("spectral FFN requires a linear map followed by SiLU")
    gate = module.act
    if (
        type(gate.scalar_act) is not torch.nn.SiLU
        or type(gate.gate_act) is not torch.nn.Sigmoid
    ):
        raise ValueError("spectral FFN requires SiLU scalars and sigmoid gates")
    if gate.lmax != module.lmax or gate.mmax != module.lmax:
        raise ValueError("spectral FFN requires complete node angular degrees")
    if (
        module.so3_linear_1.lmax != module.lmax
        or module.so3_linear_2.lmax != module.lmax
    ):
        raise ValueError("spectral FFN linear maps must share lmax")
    channels = module.sphere_channels
    hidden = module.hidden_channels
    scalar = module.scalar_mlp[0]
    if (
        scalar.in_features != channels
        or scalar.out_features != module.lmax * hidden
        or scalar.bias is None
    ):
        raise ValueError("unexpected spectral FFN scalar-map dimensions")
    for linear, inputs, outputs in (
        (module.so3_linear_1, channels, hidden),
        (module.so3_linear_2, hidden, channels),
    ):
        if (
            tuple(linear.weight.shape) != (module.lmax + 1, outputs, inputs)
            or linear.bias is None
            or linear.bias.numel() != outputs
        ):
            raise ValueError("unexpected spectral FFN linear-map dimensions")


def fused_equivariant_ffn(norm, atomwise, x):
    scalar_linear = atomwise.scalar_mlp[0]
    first = atomwise.so3_linear_1
    second = atomwise.so3_linear_2
    if first.lmax != norm.lmax or second.lmax != norm.lmax:
        raise ValueError("equivariant FFN modules must share lmax")
    return feed_forward(
        x, first.weight, second.weight, scalar_linear.weight,
        input_bias=first.bias, output_bias=second.bias,
        gate_bias=scalar_linear.bias,
        norm_weight=norm.affine_weight, norm_bias=norm.affine_bias, norm_eps=norm.eps,
        residual=True,
    )
