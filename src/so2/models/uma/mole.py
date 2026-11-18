"""Segmented cuBLAS MoLE linears for UMA."""

from __future__ import annotations

from itertools import accumulate
from types import MethodType

import torch
from torch.nn.modules import module as nn_module

from LunaEquivariance.so2.segmented_linear.interface import (
    autograd_segmented_linear,
    segmented_linear_with_gradients,
)
from LunaEquivariance.so2.segmented_linear.kernels import backward as segmented_backward


use_gate_weight_vjp = getattr(segmented_backward, "use_gate_weight_vjp", None)


class SegmentSizes:
    def __init__(self) -> None:
        self.key = None
        self.value = None
        self.gate_offsets = None

    def get(self, sizes: torch.Tensor) -> tuple[int, ...]:
        key = (id(sizes), sizes.data_ptr(), sizes._version, sizes.numel())
        if key != self.key:
            self.key = key
            self.value = tuple(int(value) for value in sizes.tolist())
        return self.value

    def offsets_for_gate(self, sizes, device):
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                return None
            key = sizes, torch.cuda.current_stream()
            cached = self.gate_offsets
            if cached is not None and cached[0] == key:
                return cached[1]
            offsets = torch.tensor((0, *accumulate(sizes)), dtype=torch.int32, device=device)
        self.gate_offsets = key, offsets
        return offsets


def use_gate_cotangent(module):
    if (
        use_gate_weight_vjp is None or torch.compiler.is_compiling()
        or getattr(module.forward, "__func__", None) is not segmented_mole_forward
        or type(module).__call__ is not torch.nn.Module.__call__
        or module._compiled_call_impl is not None or torch._C._get_tracing_state() is not None
    ):
        return False
    # Hooks can transform the raw output or its cotangent before the linear VJP.
    return not (
        module._forward_hooks or module._forward_pre_hooks
        or module._backward_hooks or module._backward_pre_hooks
        or nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks
        or nn_module._global_backward_hooks or nn_module._global_backward_pre_hooks
    )


def segmented_mole_forward(module, x, *, gate_cotangent=False):
    if torch.compiler.is_compiling():
        sizes = module.operator_compiled_segment_sizes
    else:
        sizes = module.operator_segment_sizes.get(
            module.global_mole_tensors.mole_sizes
        )
    trainable_parameters = any(
        parameter is not None and parameter.requires_grad
        for parameter in (
            module.weights,
            module.global_mole_tensors.expert_mixing_coefficients,
            module.bias,
        )
    )
    if torch.is_grad_enabled() and trainable_parameters and not module.training:
        return autograd_segmented_mole_forward(module, x, sizes)
    if (
        module.training
        and torch.is_grad_enabled()
        and trainable_parameters
        and module.operator_explicit_training
        and module.operator_force_training
    ):
        gradient_mode = "force"
    else:
        gradient_mode = (
            "full"
            if (
                torch.is_grad_enabled()
                and trainable_parameters
                and module.operator_explicit_training
            )
            else "input"
        )
        if (
            torch.is_grad_enabled()
            and trainable_parameters
            and not module.operator_explicit_training
        ):
            return autograd_segmented_mole_forward(module, x, sizes)
    arguments = (
        x,
        module.weights,
        module.global_mole_tensors.expert_mixing_coefficients,
        module.bias,
        sizes,
    )
    if (gate_cotangent and gradient_mode == "force" and use_gate_weight_vjp is not None
            and use_gate_weight_vjp(x, sizes)):
        offsets = module.operator_segment_sizes.offsets_for_gate(sizes, x.device)
        if offsets is not None:
            return segmented_linear_with_gradients(
                *arguments, gradient_mode=gradient_mode, gate_cotangent_offsets=offsets,
            )
    return segmented_linear_with_gradients(*arguments, gradient_mode=gradient_mode)


def autograd_segmented_mole_forward(module, x, sizes):
    return autograd_segmented_linear(
        x,
        module.weights,
        module.global_mole_tensors.expert_mixing_coefficients,
        module.bias,
        sizes,
    )


def install_segmented_mole(model: torch.nn.Module) -> int:
    from fairchem.core.models.uma.nn.mole import MOLE

    installed = 0
    runtimes = {}
    for module in model.modules():
        if isinstance(module, MOLE):
            key = id(module.global_mole_tensors)
            if key not in runtimes:
                runtimes[key] = SegmentSizes()
            module.operator_segment_sizes = runtimes[key]
            module.operator_explicit_training = False
            module.operator_force_training = False
            module.forward = MethodType(segmented_mole_forward, module)
            installed += 1
    if installed == 0:
        raise RuntimeError("no UMA MoLE layers were found")
    return installed


def enable_segmented_mole_training(
    model: torch.nn.Module, *, force_training: bool
) -> int:
    from fairchem.core.models.uma.nn.mole import MOLE

    enabled = 0
    for module in model.modules():
        if isinstance(module, MOLE):
            if not hasattr(module, "operator_segment_sizes"):
                raise RuntimeError("segmented MoLE must be installed first")
            module.operator_explicit_training = True
            module.operator_force_training = force_training
            enabled += 1
    if enabled == 0:
        raise RuntimeError("no UMA MoLE layers were found")
    return enabled
