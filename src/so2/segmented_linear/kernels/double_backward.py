from __future__ import annotations

import torch

from .backward import input_gradient, mixed_input_vjp, segmented_weight_gradient
from .forward import fused_projection, project, use_fused_projection
from .utils import mix_weights, split_flattened, use_grouped


def segmented_double_backward(
    dy, x, weights, coefficients, bias, sizes,
    hx, hweights, hcoefficients, hbias, mixed, dmixed,
):
    if hx is not None and hweights is None and hcoefficients is None and hbias is None:
        return segmented_hvp_input_direction(dy, weights, coefficients, sizes, hx, mixed)
    if hx is None:
        hx = torch.zeros_like(x)
    if hweights is None:
        hweights = torch.zeros_like(weights)
    if hcoefficients is None:
        hcoefficients = torch.zeros_like(coefficients)
    if bias is not None and hbias is None:
        hbias = torch.zeros_like(bias)
    return segmented_hvp(
        dy, x, weights, coefficients, bias, sizes,
        hx, hweights, hcoefficients, hbias, mixed, dmixed,
    )


def segmented_hvp(
    dy, x, weights, coefficients, bias, sizes,
    hx, hweights, hcoefficients, hbias,
    mixed, dmixed,
):
    # h denotes the incoming derivatives of the VJP outputs. The mixed
    # direction is hW_s = sum_k (a_sk hW_k + ha_sk W_k).
    hmixed = mix_weights(hweights, coefficients)
    second = hx, mixed
    if use_fused_projection(x, hmixed, hbias, sizes, second=second):
        hmixed.flatten(1).addmm_(hcoefficients, weights.flatten(1))
        d_dy = fused_projection(x, hmixed, hbias, sizes, second=second)
        dx = mixed_input_vjp(dy, hmixed, sizes, use_grouped(dy, hmixed, sizes, input_gradient=True))
        h_dmixed = segmented_weight_gradient(dy, hx, sizes, mixed)
    elif dy.is_contiguous() and hx.is_contiguous() and use_grouped(x, mixed, sizes):
        hmixed.add_(mix_weights(weights, hcoefficients))
        d_dy = project(x, hmixed, sizes)
        if hbias is not None:
            d_dy.add_(hbias)
        project(hx, mixed, sizes, d_dy, add=True)
        dx = input_gradient(dy, hmixed, sizes)
        h_dmixed = segmented_weight_gradient(dy, hx, sizes, mixed)
    else:
        hmixed.add_(mix_weights(weights, hcoefficients))
        xs = split_flattened(x, sizes)
        dys = split_flattened(dy, sizes)
        hxs = split_flattened(hx, sizes)
        d_dy = torch.empty_like(dy, memory_format=torch.contiguous_format)
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        h_dmixed = torch.empty(
            (coefficients.shape[0], weights.shape[1], weights.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )
        d_dys = split_flattened(d_dy, sizes)
        dxs = split_flattened(dx, sizes)
        for x_s, dy_s, hx_s, W_s, hW_s, d_dy_s, dx_s, h_dW_s in zip(
            xs, dys, hxs, mixed.unbind(0), hmixed.unbind(0),
            d_dys, dxs, h_dmixed.unbind(0), strict=True,
        ):
            if hbias is None:
                torch.mm(x_s, hW_s.transpose(0, 1), out=d_dy_s)
            else:
                torch.addmm(hbias, x_s, hW_s.transpose(0, 1), out=d_dy_s)
            torch.addmm(d_dy_s, hx_s, W_s.transpose(0, 1), beta=1.0, alpha=1.0, out=d_dy_s)
            torch.mm(dy_s, hW_s, out=dx_s)
            torch.mm(dy_s.transpose(0, 1), hx_s, out=h_dW_s)

    h_dmixed = h_dmixed.flatten(1)
    dmixed = dmixed.flatten(1)
    flat_weights = weights.flatten(1)
    flat_hweights = hweights.flatten(1)
    dweights = torch.mm(coefficients.transpose(0, 1), h_dmixed)
    torch.addmm(
        dweights, hcoefficients.transpose(0, 1), dmixed,
        beta=1.0, alpha=1.0, out=dweights,
    )
    dcoefficients = torch.mm(h_dmixed, flat_weights.transpose(0, 1))
    torch.addmm(
        dcoefficients, dmixed, flat_hweights.transpose(0, 1),
        beta=1.0, alpha=1.0, out=dcoefficients,
    )
    return d_dy, dx, dweights.view_as(weights), dcoefficients


def segmented_hvp_input_direction(dy, weights, coefficients, sizes, hx, mixed):
    if dy.is_contiguous() and use_grouped(hx, mixed, sizes):
        d_dy = project(hx, mixed, sizes)
        dmixed = segmented_weight_gradient(dy, hx, sizes, mixed)
    else:
        d_dy = torch.empty_like(dy, memory_format=torch.contiguous_format)
        dmixed = torch.empty(
            (coefficients.shape[0], weights.shape[1], weights.shape[2]),
            device=dy.device,
            dtype=dy.dtype,
        )
        for dy_s, hx_s, W_s, d_dy_s, dW_s in zip(
            split_flattened(dy, sizes),
            split_flattened(hx, sizes),
            mixed.unbind(0),
            split_flattened(d_dy, sizes),
            dmixed.unbind(0),
            strict=True,
        ):
            torch.mm(hx_s, W_s.transpose(0, 1), out=d_dy_s)
            torch.mm(dy_s.transpose(0, 1), hx_s, out=dW_s)
    flat_dmixed = dmixed.flatten(1)
    dweights = torch.mm(coefficients.transpose(0, 1), flat_dmixed).view_as(weights)
    dcoefficients = torch.mm(flat_dmixed, weights.flatten(1).transpose(0, 1))
    return d_dy, None, dweights, dcoefficients
