"""PyTorch gate for degree-major coefficients."""

import torch
from torch.nn import functional as F


def node_gate(gating, x, *, lmax):
    """Apply SiLU to row zero and sigmoid gates to each positive degree."""
    gates = gating.reshape(x.shape[0], lmax, x.shape[-1]).sigmoid()
    return torch.cat(
        [F.silu(x[:, :1])]
        + [
            x[:, degree**2 : (degree + 1) ** 2] * gates[:, degree - 1 : degree]
            for degree in range(1, lmax + 1)
        ],
        dim=1,
    )
