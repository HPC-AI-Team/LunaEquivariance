"""Torch fallback and higher derivatives of the same normalized activation."""

import torch

from ...o3.norm.kernels.utils import norm_product
from .kernels.graph import trace_scalar_product
from ..activation.kernels.utils import apply_activation_spec


def norm_activation_torch(features, meta, bias):
    """Evaluate the shared Torch expression for fallback and higher derivatives."""
    squares = []
    sections = []
    for path in meta["paths"]:
        width = path["copies"] * path["components"]
        section = features[..., path["input_offset"]:path["input_offset"] + width].reshape(
            *features.shape[:-1], path["copies"], path["components"]
        )
        sections.append(section)
        squares.append(norm_product(path["components"], section, section).squeeze(-1))
    norms = torch.cat(squares, -1) if squares else features.new_zeros((*features.shape[:-1], 0))
    if not meta["norm_squared"]:
        norms = norms.relu().sqrt()
    if meta["eps_squared"] > 0:
        norms[norms < meta["eps_squared"]] = meta["eps_squared"]
        norms = norms.sqrt()
    scalings = apply_activation_spec(norms + bias if bias is not None else norms, meta["activation"])
    if meta["normalize"]:
        scalings = scalings / norms
    outputs = []
    for path, section in zip(meta["paths"], sections):
        scales = scalings[..., path["scalar_offset"]:path["scalar_offset"] + path["copies"]]
        outputs.append(trace_scalar_product(path["components"], scales.unsqueeze(-1), section).flatten(-2))
    return torch.cat(outputs, -1) if outputs else features.new_zeros(features.shape)


def continue_triple_backward(meta, x, bias, g, rx, rb, bx, bb, bg, needs):
    """Continue the saved forward expression beyond the native triple-backward."""
    with torch.enable_grad():
        # These inputs can share ancestors: a loss gradient G or a
        # higher direction R can already depend on X. Fresh views make
        # each local derivative target independent of its siblings.
        # They retain their original edges so the outer engine alone
        # propagates the returned partials to shared ancestors.
        x, bias, g, rx, rb = tuple(
            None if value is None else value.view_as(value) for value in (x, bias, g, rx, rb)
        )

        y = norm_activation_torch(x, meta, bias)

        primal = (x, bias)
        first_indices = [i for i, v in enumerate(primal) if v is not None and v.requires_grad]
        first = torch.autograd.grad(
            y,
            tuple(primal[i] for i in first_indices),
            g,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        lower = (rx, rb)
        first_live = [
            (v, lower[i])
            for i, v in zip(first_indices, first)
            if v is not None and v.requires_grad and lower[i] is not None
        ]
        second_inputs = (x, bias, g)
        second_indices = [
            i for i, v in enumerate(second_inputs) if v is not None and v.requires_grad
        ]
        second = (
            torch.autograd.grad(
                tuple(v for v, _ in first_live),
                tuple(second_inputs[i] for i in second_indices),
                tuple(s for _, s in first_live),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            if first_live
            else ()
        )

        seeds = bx, bb, bg
        second_live = [
            (v, seeds[i])
            for i, v in zip(second_indices, second)
            if v is not None and v.requires_grad and seeds[i] is not None
        ]
        targets = (x, bias, g, rx, rb)
        target_indices = [
            i for i, (v, need) in enumerate(zip(targets, needs)) if v is not None and need
        ]
        values = (
            torch.autograd.grad(
                tuple(v for v, _ in second_live),
                tuple(targets[i] for i in target_indices),
                tuple(s for _, s in second_live),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            if second_live
            else ()
        )

        result = [None] * 5
        for i, value in zip(target_indices, values):
            result[i] = value
    return tuple(result)
