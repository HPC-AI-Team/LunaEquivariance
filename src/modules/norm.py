"""PyTorch module for norm."""

from collections import OrderedDict
from copy import deepcopy

import torch

try:
    from e3nn.o3 import Norm as E3nnNorm
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    E3nnNorm = None

from ..o3.norm.interface import (
    IRREPS_MISSING,
    build_norm_metadata,
    norm,
    register_ops,
)
from ..o3.norm.kernels.utils import NormKernels, SUPPORTED_DTYPES
from ..o3.norm.state import TensorProductState


class LunaEqNorm(torch.nn.Module):
    def __init__(
        self,
        norm_or_irreps=IRREPS_MISSING,
        squared=False,
        *,
        irreps_in=IRREPS_MISSING,
        dtype=None,
    ):
        super().__init__()
        if irreps_in is not IRREPS_MISSING:
            if norm_or_irreps is IRREPS_MISSING or norm_or_irreps is None:
                norm_or_irreps = irreps_in
            elif irreps_in is not None:
                raise TypeError("irreps_in was supplied twice")
        if norm_or_irreps is IRREPS_MISSING:
            raise TypeError("Norm requires irreps_in")

        self.training = getattr(norm_or_irreps, "training", self.training)
        compute_dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype is None and isinstance(norm_or_irreps, torch.nn.Module):
            for value in (*norm_or_irreps.parameters(), *norm_or_irreps.buffers()):
                if value.is_floating_point():
                    compute_dtype = value.dtype
                    break
        if compute_dtype not in SUPPORTED_DTYPES:
            raise ValueError("Norm supports floating point float16, bfloat16, float32 and float64")

        metadata = build_norm_metadata(norm_or_irreps, squared, dtype=compute_dtype)
        self.meta = metadata
        self.compute_dtype = compute_dtype
        self.kernels = OrderedDict()
        self._specializations = OrderedDict()
        self.active_config = None
        self._device_marker = torch.empty(0, dtype=self.compute_dtype)
        self.irreps_in, self.irreps_out = self.meta["irreps_in"], self.meta["irreps_out"]
        self.squared = self.meta["squared"]
        self.compile_irreps = str(self.irreps_in)

        if hasattr(norm_or_irreps, "tp") and hasattr(norm_or_irreps, "squared"):
            self.tp = deepcopy(norm_or_irreps.tp)
        else:
            self.tp = (
                TensorProductState(self.irreps_in, norm=True)
                if E3nnNorm is None
                else E3nnNorm(self.irreps_in, squared=self.squared).tp
            )
        if dtype is not None:
            self.tp.to(dtype=dtype)

    def __getstate__(self):
        state = torch.nn.Module.__getstate__(self).copy()
        for name in ("kernels", "_specializations"):
            state[name] = OrderedDict()
        state["active_config"] = None
        return state

    def _apply(self, fn, recurse=True):
        marker = fn(self._device_marker)
        dtype = marker.dtype
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError("Norm supports floating point float16, bfloat16, float32 and float64")
        result = super()._apply(fn, recurse=recurse)
        self._device_marker = marker
        if dtype != self.compute_dtype:
            self.compute_dtype = dtype
            self.meta = {**self.meta, "dtype": dtype}

        # Configuration can change without a dtype change. Replace the cache
        # so saved backward contexts keep their original specialization.
        self.kernels = OrderedDict()
        self._specializations.clear()
        self.active_config = None
        return result

    def specialize(self, dtype):
        key = (dtype, bool(self.squared))
        result = self._specializations.get(key)
        if result is None:
            metadata = {**self.meta, "dtype": dtype, "squared": bool(self.squared)}
            primary = dtype == self.compute_dtype and len(self._specializations) == 0
            result = NormKernels(
                meta=metadata,
                compute_dtype=dtype,
                index_bits=32,
                kernels=self.kernels if primary else OrderedDict(),
            )
            self._specializations[key] = result
            if len(self._specializations) > 32:
                self._specializations.popitem(last=False)
        return result

    @property
    def last_execution_backend(self):
        return None if self.active_config is None else self.active_config.last_execution_backend

    @property
    def backward_backend(self):
        return None if self.active_config is None else self.active_config.backward_backend

    def forward(self, features):
        if torch.compiler.is_compiling() and features.is_cuda and features.dtype in (torch.float32, torch.float64):
            name = register_ops(self.compile_irreps, bool(self.squared), 32, features.dtype)
            return getattr(torch.ops.lunaeq, name).default(features)
        config = self.active_config
        if config is None or config.compute_dtype != features.dtype or config.meta["squared"] != bool(self.squared):
            config = self.specialize(features.dtype)
            self.active_config = config
        return norm(features, self.irreps_in, self.squared, config=config)


__all__ = ['LunaEqNorm']
