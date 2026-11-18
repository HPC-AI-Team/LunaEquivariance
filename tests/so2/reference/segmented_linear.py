"""PyTorch expert mixtures and per-segment channel maps."""

import torch
from torch.nn import functional as F


def segmented_linear(x, weights, coefficients, bias, sizes):
    """Apply a mixture of expert maps to each contiguous input segment.

    ``x`` is ``(N,...,Cin)``, ``weights`` is ``(experts,Cout,Cin)``,
    ``coefficients`` is ``(segments,experts)``, and optional shared bias
    is ``(Cout,)``. ``sizes`` partitions the leading input axis and may
    include empty segments. For segment ``s``, compute
    ``W_s = sum(coefficients[s,k] * weights[k] for k in experts)`` then
    ``linear(x_s, W_s, bias)``. Return ``(N,...,Cout)``. Every floating
    input supports backward and double backward. Segment sizes are fixed.
    """
    mixed = torch.einsum("sk,koi->soi", coefficients, weights)
    return torch.cat(
        [
            F.linear(segment, weight, bias)
            for segment, weight in zip(x.split(sizes, dim=0), mixed, strict=True)
        ],
        dim=0,
    )
