"""Differentiate the segmented linear double backward using matrix products."""

import torch

from .backward import mixed_input_vjp, segmented_weight_gradient
from .forward import mixed_forward, segmented_forward
from .utils import mix_weights, use_grouped


def segmented_triple_backward(
    dy, x, weights, coefficients, sizes, hx, hweights, hcoefficients,
    by, bx, bweights, bcoefficients, mixed, dmixed, need_seeds,
):
    # h and b are cotangents of the first and second backward outputs.
    hx, hweights, hcoefficients = (
        torch.zeros_like(value) if seed is None else seed
        for value, seed in zip((x, weights, coefficients), (hx, hweights, hcoefficients), strict=True)
    )
    by, bx, bweights, bcoefficients = (
        torch.zeros_like(value) if seed is None else seed
        for value, seed in zip(
            (dy, x, weights, coefficients), (by, bx, bweights, bcoefficients), strict=True,
        )
    )

    hmixed = mix_weights(hweights, coefficients).add_(mix_weights(weights, hcoefficients))
    bmixed = mix_weights(bweights, coefficients).add_(mix_weights(weights, bcoefficients))
    hbmixed = mix_weights(hweights, bcoefficients).add_(mix_weights(bweights, hcoefficients))

    # The derivative with respect to dy is the mixed second direction of Y.
    grad_dy = segmented_forward(x, hbmixed, None, sizes)
    mixed_forward(hx, bmixed, sizes, use_grouped(hx, bmixed, sizes), output=grad_dy)
    mixed_forward(bx, hmixed, sizes, use_grouped(bx, hmixed, sizes), output=grad_dy)
    grad_x = mixed_input_vjp(by, hmixed, sizes, use_grouped(by, hmixed, sizes, input_gradient=True))
    grad_x.add_(mixed_input_vjp(dy, hbmixed, sizes, use_grouped(dy, hbmixed, sizes, input_gradient=True)))

    # The remaining parameter derivatives use adjoints of the weight mixture.
    h_dmixed = segmented_weight_gradient(dy, hx, sizes, mixed).flatten(1)
    b_dmixed = segmented_weight_gradient(by, x, sizes, mixed)
    b_dmixed.add_(segmented_weight_gradient(dy, bx, sizes, mixed))
    b_dmixed = b_dmixed.flatten(1)
    hb_dmixed = segmented_weight_gradient(by, hx, sizes, mixed).flatten(1)
    dmixed = dmixed.flatten(1)
    weights_t = weights.flatten(1).T
    hweights_t = hweights.flatten(1).T
    bweights_t = bweights.flatten(1).T

    grad_weights = coefficients.T @ hb_dmixed
    grad_weights.addmm_(hcoefficients.T, b_dmixed).addmm_(bcoefficients.T, h_dmixed)
    grad_coefficients = hb_dmixed @ weights_t
    grad_coefficients.addmm_(b_dmixed, hweights_t).addmm_(h_dmixed, bweights_t)

    need_hx, need_hweights, need_hcoefficients, need_hbias = need_seeds
    grad_hx = grad_hweights = grad_hcoefficients = grad_hbias = None
    if need_hx:
        grad_hx = mixed_input_vjp(by, mixed, sizes, use_grouped(by, mixed, sizes, input_gradient=True))
        grad_hx.add_(mixed_input_vjp(dy, bmixed, sizes, use_grouped(dy, bmixed, sizes, input_gradient=True)))
    if need_hweights:
        grad_hweights = (coefficients.T @ b_dmixed).addmm_(bcoefficients.T, dmixed).view_as(weights)
    if need_hcoefficients:
        grad_hcoefficients = (b_dmixed @ weights_t).addmm_(dmixed, bweights_t)
    if need_hbias:
        grad_hbias = by.flatten(0, -2).sum(0)

    return (
        (grad_dy, grad_x, grad_weights.view_as(weights), grad_coefficients),
        (grad_hx, grad_hweights, grad_hcoefficients, grad_hbias),
    )
