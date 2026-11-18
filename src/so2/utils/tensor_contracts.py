"""Validate tensor metadata and CUDA launch constraints."""

import warnings

import torch


def require_cuda_fp32(
    *tensors: torch.Tensor,
    operation: str,
    contiguous: bool = False,
) -> None:
    """Check tensor metadata without synchronizing the CUDA stream."""
    if not tensors or not isinstance(tensors[0], torch.Tensor):
        raise ValueError(f"{operation} requires tensor inputs")
    device = tensors[0].device
    for value in tensors:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{operation} requires tensor inputs")
        if value.dtype != torch.float32:
            raise ValueError(f"{operation} requires CUDA FP32 tensors")
        if contiguous and not value.is_contiguous():
            raise ValueError(f"{operation} requires contiguous tensors")
        if value.device != device:
            raise ValueError(f"{operation} inputs must share one device")
    if device.type != "cuda":
        raise ValueError(f"{operation} requires CUDA FP32 tensors")


def allow_atomic_reduction(operation):
    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        f"{operation} uses floating-point atomic reductions and is not deterministic"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        raise RuntimeError(message)


def require_int32_offsets(shape, strides, *, name):
    if any(size < 0 for size in shape) or any(stride < 0 for stride in strides):
        raise ValueError(f"{name} requires nonnegative sizes and strides")
    if not all(shape):
        return
    last = sum((size - 1) * stride for size, stride in zip(shape, strides))
    if last > 2**31 - 1:
        raise ValueError(f"{name} exceeds the supported 32-bit element offset range")
