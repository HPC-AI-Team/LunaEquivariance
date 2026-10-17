"""Unweighted per-irrep Torch products for the no-e3nn construction path."""

import torch

from ..irreps import PublicIrreps as Irreps


class TensorProductState(torch.nn.Module):
    """Standalone state for unweighted per-irrep dot or scalar multiplication."""

    def __init__(self, irreps, *, norm=False):
        super().__init__()
        self.irreps_in2 = Irreps(irreps)
        self.irreps_in1 = (
            self.irreps_in2 if norm else Irreps([(self.irreps_in2.num_irreps, "0e")]).simplify()
        )
        self.irreps_out = (
            Irreps([(self.irreps_in2.num_irreps, "0e")]).simplify() if norm else self.irreps_in2
        )
        self.norm = norm
        self.register_buffer("weight", torch.empty(0))
        self.register_buffer("output_mask", torch.ones(self.irreps_out.dim))

    def forward(self, left, right):
        outputs = []
        offset = scalar = 0
        for copies, irrep in self.irreps_in2:
            if copies:
                shape = (*right.shape[:-1], copies, irrep.dim)
                section = right[..., offset : offset + copies * irrep.dim].reshape(shape)
                if self.norm:
                    other = left[..., offset : offset + copies * irrep.dim].reshape(
                        *left.shape[:-1], copies, irrep.dim
                    )
                    outputs.append((other * section).sum(-1))
                else:
                    outputs.append(
                        (left[..., scalar : scalar + copies].unsqueeze(-1) * section).flatten(-2)
                    )
            offset += copies * irrep.dim
            scalar += copies
        return (
            torch.cat(outputs, -1)
            if outputs
            else right.new_zeros((*right.shape[:-1], self.irreps_out.dim))
        )
