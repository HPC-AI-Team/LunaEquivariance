"""PyTorch module for dropout."""

from collections import OrderedDict

import torch

from ..nn.dropout.interface import UNSET, dropout, layout_metadata
from ..o3.irreps import PublicIrreps as Irreps


class LunaEqDropout(torch.nn.Module):
    """Drop whole irrep copies, sharing decisions across the middle axes.

    ``Dropout(irreps, p)`` accepts the e3nn constructor arguments. An existing
    e3nn Dropout can also be supplied as the first argument. Input precision
    determines execution precision. A numeric probability needs no parameters
    or buffers; a probability Parameter retains e3nn's registration behavior.
    """

    def __init__(self, irreps, p=UNSET):
        super().__init__()
        if (
            isinstance(irreps, torch.nn.Module)
            and hasattr(irreps, "irreps")
            and hasattr(irreps, "p")
        ):
            reference = irreps
            irreps = reference.irreps
            if p is UNSET:
                p = reference.p
            self.training = reference.training
        elif p is UNSET:
            raise TypeError("Dropout requires the probability p")

        # Preserve the original groups: simplification changes RNG consumption.
        self.irreps = Irreps(irreps)
        self.p = p
        self._layout_irreps = self.irreps
        self.config = dict(
            irreps=self.irreps, **layout_metadata(self.irreps),
            _native_launches=OrderedDict(), _last_native_key=None, _last_native_runtime=None,
            last_execution_backend=None,
        )

    @property
    def last_execution_backend(self):
        return self.config["last_execution_backend"]

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, p={self.p})"

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["config"] = dict(self.config, _native_launches=OrderedDict(),
                               _last_native_key=None, _last_native_runtime=None)
        return state

    def forward(self, x):
        if torch.compiler.is_compiling():
            if not self.training:
                return x
            return dropout(x, None, self.p, self.training, config=self.config)
        if self._layout_irreps is not self.irreps:
            self._layout_irreps = self.irreps
            self.config.update(irreps=self.irreps, **layout_metadata(self.irreps),
                               _last_native_key=None, _last_native_runtime=None)
        return dropout(x, self.irreps, self.p, self.training, config=self.config)


__all__ = ['LunaEqDropout']
