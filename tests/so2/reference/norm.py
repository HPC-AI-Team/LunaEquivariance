"""PyTorch gating and normalization of node coefficients."""

import torch


def fused_rms_norm(
    x,
    weight,
    bias,
    *,
    lmax,
    epsilon,
    coefficient_major_output=False,
):
    """Center scalars, average degree mean squares, then apply affine maps.

    Each degree contributes equally, irrespective of its number of rows.
    ``coefficient_major_output`` changes storage strides, not logical axes.
    """
    centered = torch.cat((x[:, :1] - x[:, :1].mean(-1, keepdim=True), x[:, 1:]), dim=1)
    variance = torch.stack(
        [
            centered[:, degree**2 : (degree + 1) ** 2].square().mean((1, 2))
            for degree in range(lmax + 1)
        ]
    ).mean(0)
    scale = torch.rsqrt(variance + epsilon)[:, None, None]
    output = torch.cat(
        [
            centered[:, degree**2 : (degree + 1) ** 2] * scale * weight[degree]
            + (bias if degree == 0 else 0)
            for degree in range(lmax + 1)
        ],
        dim=1,
    )
    if coefficient_major_output:
        output = output.transpose(0, 1).contiguous().transpose(0, 1)
    return output


def fused_equivariant_norm(inputs, weight, bias, eps):
    """The same degree-balanced normalization, inferring L from the weights."""
    return fused_rms_norm(inputs, weight, bias, lmax=weight.shape[0] - 1, epsilon=eps)
