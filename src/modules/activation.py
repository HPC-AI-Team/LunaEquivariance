"""PyTorch module for activation."""

from collections import OrderedDict

import torch

try:
    from e3nn.nn import Activation
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    Activation = None

from ..nn.activation.interface import _IRREPS_MISSING, activation
from ..nn.activation.kernels.utils import parse_activation, parse_activation_paths
from ..o3.irreps import PublicIrreps as Irreps


class LunaEqActivation(torch.nn.Module):
    """Scalar activation with mathematical settings fixed at construction."""

    def __init__(
        self,
        activation_or_irreps=_IRREPS_MISSING,
        acts=None,
        *,
        irreps_in=_IRREPS_MISSING,
        dtype=None,
    ):
        super().__init__()
        if irreps_in is not _IRREPS_MISSING:
            if activation_or_irreps is _IRREPS_MISSING or activation_or_irreps is None:
                activation_or_irreps = irreps_in
            elif irreps_in is not None:
                raise TypeError("irreps_in was supplied twice")
        if activation_or_irreps is _IRREPS_MISSING:
            raise TypeError("Activation requires irreps_in")

        self.compute_dtype = torch.get_default_dtype() if dtype is None else dtype

        parsed = None
        if acts is None and hasattr(activation_or_irreps, "paths"):
            official = activation_or_irreps
        else:
            if Activation is None:
                raise ModuleNotFoundError("No module named 'e3nn'", name="e3nn")

            acts = list(acts)
            parsed = [parse_activation(act) for act in acts]
            official = Activation(activation_or_irreps, acts)

        # Reuse e3nn's normalized paths and normalization constants.
        self.acts = official.acts
        self.paths = official.paths
        self.training = official.training
        self.irreps_in, self.irreps_out = Irreps(str(official.irreps_in)), Irreps(str(official.irreps_out))
        self.in_dim, self.out_dim = self.irreps_in.dim, self.irreps_out.dim
        self.activation_paths = parse_activation_paths(official, parsed)
        self.kernel_cache = OrderedDict()
        object.__setattr__(self, "last_execution_backend", None)
        self._device_marker = torch.empty(0, dtype=self.compute_dtype)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["kernel_cache"] = OrderedDict()
        return state

    def _apply(self, fn, recurse=True):
        marker = fn(self._device_marker)
        dtype = marker.dtype
        result = super()._apply(fn, recurse=recurse)
        self._device_marker = marker
        if dtype != self.compute_dtype:
            self.compute_dtype = dtype
            if hasattr(self, "meta"):
                self.meta = {**self.meta, "dtype": dtype}
        return result

    def forward(self, features, dim=-1):
        cuda = features.is_cuda and features.dtype in (torch.float32, torch.float64)
        valid_dim = features.ndim and isinstance(dim, int) and not isinstance(dim, bool) and -features.ndim <= dim < features.ndim
        backend = "cuda_fused" if cuda and valid_dim and features.shape[dim] == self.in_dim else "torch_compat"
        object.__setattr__(self, "last_execution_backend", backend)
        return activation(
            features, dim=dim, paths=self.activation_paths,
            kernel_cache=self.kernel_cache,
        )


__all__ = ['LunaEqActivation']
