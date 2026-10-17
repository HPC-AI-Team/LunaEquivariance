"""Lazy reference evaluation for CPU, low precision and higher derivatives.

CUDA FP32/FP64 forward through triple-backward use native recomputation.
A differentiable triple-backward replays live Torch math for later derivatives.
"""

import builtins
import math

import torch

from .kernels.forward import evaluate_polynomial

try:
    from e3nn.o3._spherical_harmonics import _spherical_harmonics as e3nn_polynomial
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    E3NN_IMPORT_ERROR = error
    e3nn_polynomial = None


def reference_forward(x, *, degrees, normalize, normalization):
    if e3nn_polynomial is None:
        raise E3NN_IMPORT_ERROR
    # Same expression and scaling order as e3nn SphericalHarmonics.forward.
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
    sh = e3nn_polynomial(max(degrees), x[..., 0], x[..., 1], x[..., 2])
    if degrees != tuple(range(max(degrees) + 1)):
        sh = torch.cat([sh[..., l * l:(l + 1) * (l + 1)] for l in degrees], dim=-1)
    if normalization == 'integral':
        sh.div_(math.sqrt(4 * math.pi))
    elif normalization == 'norm':
        sh.div_(torch.cat([
            math.sqrt(2 * l + 1) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
            for l in degrees
        ]))
    return sh


def higher_backward(op, x, g, rx, bx, bg, needs):
    """Return differentiable triple-backward contractions from the live original seeds."""
    if bx is None and bg is None:
        return None, None, None
    if x is None or rx is None:
        raise RuntimeError('Normalized SH replay needs saved X and R')

    # An SG-only consumer does not read G. Supply its differentiation slot
    # without inventing an edge to an omitted original G operand.
    # Local views make the inner differentiation slots independent. G/R may
    # themselves depend on the original X; their real upstream edges remain
    # attached to these views for the outer engine to traverse once.
    local_x = x.view_as(x)
    local_rx = rx.view_as(rx)
    seed_g = (
        g.view_as(g)
        if g is not None
        else x.new_zeros((*x.shape[:-1], op.meta['out_dim']), requires_grad=True)
    )
    local_bx = bx.view_as(bx) if bx is not None else None
    local_bg = bg.view_as(bg) if bg is not None else None
    primary = polynomial_forward if e3nn_polynomial is None else reference_forward

    with torch.enable_grad():
        y = primary(local_x, degrees=tuple(op.meta['degrees']),
                    normalize=op.normalize, normalization=op.normalization)
        (dx,) = torch.autograd.grad(y, local_x, seed_g, create_graph=True, retain_graph=True)

        second_targets, second_seeds = [], []
        if bx is not None:
            second_targets.append(local_x)
            second_seeds.append(local_bx)
        if bg is not None:
            second_targets.append(seed_g)
            second_seeds.append(local_bg)
        second = torch.autograd.grad(
            dx,
            tuple(second_targets),
            local_rx,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        pairs = [
            (value, seed)
            for value, seed in zip(second, second_seeds)
            if value is not None and value.requires_grad
        ]
        operands = (local_x, None if g is None else seed_g, local_rx)
        live = [
            (index, value)
            for index, (value, needed) in enumerate(zip(operands, needs))
            if needed and value is not None and value.requires_grad
        ]
        result = [None, None, None]
        if live and pairs:
            gradients = torch.autograd.grad(
                tuple(value for value, _ in pairs),
                tuple(value for _, value in live),
                tuple(seed for _, seed in pairs),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            for (index, _), gradient in zip(live, gradients):
                result[index] = gradient
        return tuple(result)


def polynomial_forward(value, *, normalize, degrees, normalization):
    coordinates = torch.nn.functional.normalize(value, dim=-1) if normalize else value
    return evaluate_polynomial(
        coordinates, degrees, normalization,
        # Bundled expressions may trigger lazy Torch imports through Tensor dispatch.
        evaluator=lambda expression, namespace, values: builtins.eval(
            expression, {**namespace, '__builtins__': builtins.__dict__}, values
        ),
    )
