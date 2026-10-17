"""PyTorch module for norm activation."""

from collections import OrderedDict
from copy import deepcopy

import torch

try:
    from e3nn.nn import NormActivation as E3nnNormActivation
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    E3nnNormActivation = None

from ..nn.norm_activation.kernels.forward import make_forward_source
from ..nn.norm_activation.interface import (
    IRREPS_MISSING,
    build_norm_activation_metadata,
    norm_activation,
    register_module_ops,
)
from ..o3.norm.interface import SUPPORTED_DTYPES
from ..nn.norm_activation.kernels.utils import NormActivationKernels
from ..o3.norm.state import TensorProductState
from .norm import LunaEqNorm
from ..o3.norm.interface import NATIVE_DTYPES


class LunaEqNormActivation(LunaEqNorm):
    """Apply a built-in scalar activation with settings fixed at construction."""

    def __init__(
        self,
        activation_or_irreps=IRREPS_MISSING,
        scalar_nonlinearity=None,
        normalize=True,
        epsilon=None,
        bias=False,
        *,
        irreps_in=IRREPS_MISSING,
        dtype=None,
    ):
        torch.nn.Module.__init__(self)
        if irreps_in is not IRREPS_MISSING:
            if activation_or_irreps is IRREPS_MISSING or activation_or_irreps is None:
                activation_or_irreps = irreps_in
            elif irreps_in is not None:
                raise TypeError("irreps_in was supplied twice")
        if activation_or_irreps is IRREPS_MISSING:
            raise TypeError("NormActivation requires irreps_in")

        self.training = getattr(activation_or_irreps, "training", self.training)
        compute_dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype is None and isinstance(activation_or_irreps, torch.nn.Module):
            for value in (*activation_or_irreps.parameters(), *activation_or_irreps.buffers()):
                if value.is_floating_point():
                    compute_dtype = value.dtype
                    break
        if compute_dtype not in SUPPORTED_DTYPES:
            raise ValueError("NormActivation supports float16, bfloat16, float32 and float64")

        metadata, official = build_norm_activation_metadata(
            activation_or_irreps,
            scalar_nonlinearity,
            normalize,
            epsilon,
            bias,
            dtype=compute_dtype,
        )
        self.meta = metadata
        self.compute_dtype = compute_dtype
        self.kernels, self.programs = OrderedDict(), OrderedDict()
        self._specializations = OrderedDict()
        self.active_config = None
        self._device_marker = torch.empty(0, dtype=self.compute_dtype)
        self.irreps_in, self.irreps_out = (
            self.meta["irreps_in"],
            self.meta["irreps_out"],
        )
        self.normalize, self.epsilon, self.bias = (
            self.meta["normalize"],
            self.meta["epsilon"],
            self.meta["bias"],
        )
        self._eps_squared = (
            activation_or_irreps._eps_squared if official else self.meta["eps_squared"]
        )

        function = activation_or_irreps.scalar_nonlinearity if official else scalar_nonlinearity
        if hasattr(activation_or_irreps, "scalar_multiplier") and hasattr(
            activation_or_irreps, "norm"
        ):
            state = deepcopy(activation_or_irreps)
        else:
            state = (
                None
                if E3nnNormActivation is None
                else E3nnNormActivation(
                    self.irreps_in,
                    function,
                    normalize=self.normalize,
                    epsilon=self.epsilon,
                    bias=self.bias,
                )
            )

        if state is None:
            self.norm = LunaEqNorm(self.irreps_in, squared=self.epsilon is not None)
            self.scalar_multiplier = TensorProductState(self.irreps_in)
            self.scalar_nonlinearity = function
            if self.bias:
                self.biases = torch.nn.Parameter(torch.zeros(self.meta["bias_dim"]))
        else:
            self.norm = state.norm
            self.scalar_nonlinearity = state.scalar_nonlinearity
            self.scalar_multiplier = state.scalar_multiplier
            if self.bias:
                self.biases = state.biases
        self.meta["norm_squared"] = bool(self.norm.squared)
        self.in_dim = self.meta["in_dim"]
        if dtype is not None:
            self.to(dtype=dtype)

    def specialize(self, dtype):
        result = self._specializations.get(dtype)
        if result is None:
            metadata = {**self.meta, "dtype": dtype}
            primary = dtype == self.compute_dtype and len(self._specializations) == 0
            result = NormActivationKernels(
                meta=metadata,
                forward_source=make_forward_source,
                compute_dtype=dtype,
                index_bits=32,
                kernels=self.kernels if primary else OrderedDict(),
                programs=self.programs if primary else OrderedDict(),
            )
            self._specializations[dtype] = result
        return result

    register_module_ops = register_module_ops

    def forward(self, features):
        if torch.compiler.is_compiling() and features.is_cuda and features.dtype in NATIVE_DTYPES:
            bias = self.biases if self.bias else None
            native_bias = bias is None or (
                not (bias.dtype == torch.float64 and features.dtype == torch.float32)
            )
            if native_bias and not (features.dtype == torch.float32 and torch.is_autocast_enabled("cuda")):
                if features.ndim == 0 or features.shape[-1] != self.in_dim:
                    raise ValueError("NormActivation feature dimension does not match irreps")
                if bias is not None and bias.dtype != features.dtype:
                    bias = bias.to(dtype=features.dtype)
                return getattr(torch.ops.lunaeq, self.register_module_ops(features.dtype)).default(features, bias)[0]
        config = self.active_config
        if config is None or config.compute_dtype != features.dtype:
            config = self.specialize(features.dtype)
            self.active_config = config
        return norm_activation(
            features, self.irreps_in, self.scalar_nonlinearity,
            self.normalize, self.epsilon, self.biases if self.bias else None,
            config=config,
        )


__all__ = ['LunaEqNormActivation']
