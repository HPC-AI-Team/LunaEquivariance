"""PyTorch node FFN with separate degree maps, nonlinearities and residual."""

from math import isqrt

import torch
from torch.nn import functional as F

from tests.so2.reference.norm import fused_rms_norm
from tests.so2.reference.gate import node_gate
from LunaEquivariance.so2.feed_forward.interface import degree_index


def coefficient_channel_linear(
    inputs,
    degree_weight,
    component_degree,
    scalar_bias=None,
    *,
    input_node_major=True,
):
    """Apply a degree's channel map to each row, returning (rows, nodes, Cout)."""
    coefficients = inputs.transpose(0, 1) if input_node_major else inputs
    bias = scalar_bias.reshape(-1) if scalar_bias is not None else None
    return torch.stack(
        [
            F.linear(row, degree_weight[degree], bias if index == 0 else None)
            for index, (row, degree) in enumerate(
                zip(coefficients, component_degree, strict=True)
            )
        ]
    )


def project_before_grid(
    inputs,
    degree_weight,
    channel_weight,
    component_degree,
    scalar_bias=None,
    *,
    input_node_major=True,
):
    """Apply the two original channel maps in sequence, before any grid map.

    The optimized operator composes their weights. This reference retains
    the coefficient intermediate to show the same result without composition.
    """
    coefficients = coefficient_channel_linear(
        inputs,
        degree_weight,
        component_degree,
        scalar_bias,
        input_node_major=input_node_major,
    )
    return F.linear(coefficients, channel_weight)


def project_after_grid(
    coefficients,
    channel_weight,
    degree_weight,
    component_degree,
    scalar_bias=None,
    *,
    input_node_major=False,
):
    """Apply the shared channel map, then each degree map, without composition."""
    projected = F.linear(coefficients, channel_weight)
    return coefficient_channel_linear(
        projected,
        degree_weight,
        component_degree,
        scalar_bias,
        input_node_major=input_node_major,
    )


def coefficient_ffn(
    inputs,
    scalar_weight,
    scalar_bias,
    first_weight,
    first_expand_index,
    first_bias,
    second_weight,
    second_expand_index,
    second_bias,
    *,
    lmax,
):
    """Degree-wise linear maps with a scalar SiLU and sigmoid-of-SiLU gates.

    All weights use PyTorch's (output, input) channel convention. The scalar
    branch maps input coefficient zero to one gate per positive degree and
    hidden channel. Biases on the degree maps affect coefficient zero only.
    """
    hidden = coefficient_channel_linear(
        inputs, first_weight, first_expand_index, first_bias
    )
    hidden = hidden.transpose(0, 1)
    gates = F.silu(F.linear(inputs[:, 0], scalar_weight, scalar_bias))
    activated = node_gate(gates, hidden, lmax=lmax)
    return coefficient_channel_linear(
        activated,
        second_weight,
        second_expand_index,
        second_bias,
    ).transpose(0, 1)


def norm_residual_ffn(
    x,
    norm_weight,
    norm_bias,
    gating_weight,
    gating_bias,
    first_weight,
    first_bias,
    second_weight,
    second_bias,
    expand_index,
    *,
    lmax,
    epsilon,
):
    """Return x + coefficient_ffn(fused_rms_norm(x)) using separate operations."""
    normalized = fused_rms_norm(x, norm_weight, norm_bias, lmax=lmax, epsilon=epsilon)
    return x + coefficient_ffn(
        normalized,
        gating_weight,
        gating_bias,
        first_weight,
        expand_index,
        first_bias,
        second_weight,
        expand_index,
        second_bias,
        lmax=lmax,
    )


def grid_ffn(
    inputs,
    input_degree_weight,
    input_channel_weight,
    input_component_degree,
    input_scalar_bias,
    to_grid,
    gate_weight,
    gate_bias,
    from_grid,
    output_channel_weight,
    output_degree_weight,
    output_component_degree,
    output_scalar_bias,
):
    """Expand, project channels, multiply grid factors, then return to coefficients.

    Unlike the optimized path, this keeps the shared channel projections on
    the grid side of the angular transforms and retains each intermediate.
    The result is coefficient-major (output rows, nodes, output channels).
    """
    coefficients = coefficient_channel_linear(
        inputs,
        input_degree_weight,
        input_component_degree,
        input_scalar_bias,
    )
    grid = torch.einsum("gr,rnc->gnc", to_grid, coefficients)
    first, second = F.linear(grid, input_channel_weight).chunk(2, dim=-1)
    gates = F.linear(inputs[:, 0], gate_weight, gate_bias).sigmoid()
    product = first * second * gates.unsqueeze(0)
    projected_grid = F.linear(product, output_channel_weight)
    coefficients = torch.einsum("rg,gnc->rnc", from_grid, projected_grid)
    return coefficient_channel_linear(
        coefficients,
        output_degree_weight,
        output_component_degree,
        output_scalar_bias,
        input_node_major=False,
    )


def merge_scalar_branch(outputs, scalar_hidden, output_degree_weight):
    """Add the bias-free degree-zero scalar map and return node-major features."""
    scalar = F.linear(scalar_hidden, output_degree_weight[0]).transpose(0, 1)
    return torch.cat((outputs[:1] + scalar, outputs[1:]), dim=0).transpose(0, 1)


def feed_forward(
    x, input_weight, output_weight, gate_weight, *,
    input_bias, output_bias, gate_bias,
    to_grid=None, from_grid=None, grid_input_weight=None, grid_output_weight=None,
    norm_weight=None, norm_bias=None, norm_eps=1e-5,
    residual=False, scalar_branch=None, input_only=False,
):
    """Separate PyTorch transforms, nonlinearities and residual for comparison."""
    index = degree_index(x.shape[1], x.device)
    lmax = isqrt(x.shape[1]) - 1
    if input_only and input_weight.shape[0] == output_weight.shape[0] == x.shape[1]:
        index = torch.arange(x.shape[1], device=x.device)
    original = x
    if norm_weight is not None:
        x = fused_rms_norm(
            x, norm_weight, norm_bias, lmax=lmax, epsilon=norm_eps,
        )
    if to_grid is None:
        result = coefficient_ffn(
            x, gate_weight, gate_bias, input_weight, index, input_bias,
            output_weight, index, output_bias, lmax=lmax,
        )
    else:
        result = grid_ffn(
            x, input_weight, grid_input_weight, index, input_bias,
            to_grid, gate_weight, gate_bias, from_grid,
            grid_output_weight, output_weight, index, output_bias,
        )
        if scalar_branch is None:
            result = result.permute(1, 0, 2)
        else:
            first, second = F.linear(x[:, 0], *scalar_branch).chunk(2, dim=-1)
            result = merge_scalar_branch(
                result, (F.silu(first) * second).unsqueeze(1), output_weight
            )
    return original + result if residual else result
