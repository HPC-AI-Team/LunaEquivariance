"""Symmetric contractions with trainable species weights."""

import json

import torch
from e3nn import o3

from ..o3.symmetric_contraction.interface import (
    SymmetricContractionForward, get_operator_config, kernel_metadata,
)


class LunaEqSymmetricContraction(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, correlation=3, *, num_elements,
                 use_reduced_cg=True, dtype=None, device=None):
        super().__init__()
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("SymmetricContraction supports float32 and float64")
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.correlation, self.num_elements = correlation, num_elements
        self.use_reduced_cg = use_reduced_cg
        self.config = get_operator_config(str(self.irreps_in), str(self.irreps_out), correlation,
                                          num_elements, dtype, use_reduced_cg)
        self.meta = kernel_metadata(self.config)
        uniform = len({mul for mul, _ in (*self.irreps_in, *self.irreps_out)}) == 1
        parts = []
        for index, count in enumerate(self.meta["path_counts"]):
            channels = self.irreps_out[index // correlation].mul
            value = torch.randn(num_elements, count, channels, device=device, dtype=dtype) / max(1, count)
            parts.append(value if uniform else value.flatten(1))
        self.weight = torch.nn.Parameter(torch.cat(parts, dim=1))

    def get_extra_state(self):
        return json.loads(self.config)

    def set_extra_state(self, spec):
        self.config = json.dumps(spec, separators=(",", ":"))
        self.meta = kernel_metadata(self.config)
        self.irreps_in, self.irreps_out = o3.Irreps(spec["irreps_in"]), o3.Irreps(spec["irreps_out"])
        self.correlation, self.num_elements = spec["correlation"], spec["num_elements"]
        self.use_reduced_cg = spec["use_reduced_cg"]

    def forward(self, x, attrs):
        return SymmetricContractionForward.apply(x, self.weight.flatten(1), attrs, self.config)


__all__ = ['LunaEqSymmetricContraction']
