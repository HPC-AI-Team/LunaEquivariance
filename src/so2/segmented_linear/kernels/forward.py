from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import (
    PROJECT, batched_segment_gemm, grouped_gemm, row_sizes, split_flattened,
    use_equal_segment_bmm, use_grouped,
)


def project(x, mixed, sizes, output=None, *, add=False):
    if output is None:
        output = x.new_empty((*x.shape[:-1], mixed.shape[1]))
    grouped_gemm(
        x.reshape(-1, x.shape[-1]).contiguous(), mixed,
        output.reshape(-1, output.shape[-1]), row_sizes(x, sizes), PROJECT, add,
    )
    return output


@triton.jit
def segmented_projection_kernel(
    x, weights, hx, hweights, bias, output, rows: tl.constexpr,
    two_products: tl.constexpr, add_bias: tl.constexpr, input_vjp: tl.constexpr,
):
    group = tl.program_id(1)
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 32)
    x_offset = (group * rows + m[:, None]) * 64 + k[None, :]
    if input_vjp:
        w_offset = (group * 64 + k[:, None]) * 64 + n[None, :]
        w_step: tl.constexpr = 32 * 64
    else:
        w_offset = (group * 64 + n[None, :]) * 64 + k[:, None]
        w_step: tl.constexpr = 32
    result = tl.zeros((64, 64), tl.float32)
    for part in range(2):
        features = tl.load(x + x_offset + part * 32)
        matrix = tl.load(weights + w_offset + part * w_step)
        result = tl.dot(features, matrix, result, input_precision="ieee")
    if two_products:
        for part in range(2):
            features = tl.load(hx + x_offset + part * 32)
            matrix = tl.load(hweights + w_offset + part * w_step)
            result = tl.dot(features, matrix, result, input_precision="ieee")
    if add_bias:
        result += tl.load(bias + n)[None, :]
    tl.store(output + (group * rows + m[:, None]) * 64 + n[None, :], result)


def use_fused_projection(x, mixed, bias, sizes, *, second=None, input_vjp=False):
    if (
        len(sizes) != 4 or sizes[0] not in (2048, 4096, 8192, 16384, 32768)
        or any(size != sizes[0] for size in sizes)
        or x.shape != (4 * sizes[0], 64) or mixed.shape != (4, 64, 64)
        or not x.is_cuda or torch.version.hip is not None
        or torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled()
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled("cuda")
    ):
        return False
    capability = torch.cuda.get_device_capability(x.device)
    tensors = (x, mixed)
    if input_vjp:
        if capability != (8, 0) or sizes[0] < 16384:
            return False
    elif second is not None:
        hx, hweights = second
        if capability != (8, 0) or hx.shape != x.shape or hweights.shape != mixed.shape:
            return False
        tensors += second
    elif capability != (8, 0) and not (capability == (9, 0) and sizes[0] == 2048):
        return False
    if bias is not None:
        if bias.shape != (64,):
            return False
        tensors += (bias,)
    return all(
        value.device == x.device and value.dtype == torch.float32
        and value.is_contiguous() and not value.is_neg()
        for value in tensors
    )


def fused_projection(x, mixed, bias, sizes, *, second=None, input_vjp=False):
    output = torch.empty_like(x)
    hx, hweights = second if second is not None else (x, mixed)
    with torch.cuda.device(x.device):
        segmented_projection_kernel[(sizes[0] // 64, 4)](
            x, mixed, hx, hweights, bias if bias is not None else x, output, sizes[0],
            second is not None, bias is not None, input_vjp, num_warps=4, num_stages=1,
        )
    return output


def segmented_forward(x, mixed, bias, sizes):
    if bias is not None and use_fused_projection(x, mixed, bias, sizes):
        return fused_projection(x, mixed, bias, sizes)
    # The bias epilogue selects a slower GEMM for this range on Ampere.
    if (
        len(sizes) == 1 and 2048 <= sizes[0] <= 4096
        and x.ndim == 2 and mixed.shape == (1, 640, 768)
        and bias is not None and x.is_cuda and x.dtype == torch.float32
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.is_autocast_enabled("cuda")
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    ):
        return (x @ mixed[0].T).add_(bias)
    if use_equal_segment_bmm(x, mixed, sizes, large=True):
        output = x.new_empty((x.shape[0], mixed.shape[1]))
        batched_segment_gemm(x, mixed, output, sizes, PROJECT, False)
        return output if bias is None else output.add_(bias)
    if use_grouped(x, mixed, sizes):
        output = project(x, mixed, sizes)
        if bias is not None:
            output.add_(bias)
        return output
    outputs = tuple(
        torch.nn.functional.linear(local, local_mixed, bias)
        for local, local_mixed in zip(
            torch.split(x, sizes, dim=0), mixed.unbind(0), strict=True,
        )
    )
    return outputs[0].contiguous() if len(outputs) == 1 else torch.cat(outputs, dim=0)


def mixed_forward(x, mixed, sizes, grouped, output=None):
    if grouped:
        return project(x, mixed, sizes, output, add=output is not None)
    if output is None:
        return segmented_forward(x, mixed, None, sizes)
    for local, weight, destination in zip(
        split_flattened(x, sizes), mixed.unbind(0), split_flattened(output, sizes), strict=True,
    ):
        torch.addmm(destination, local, weight.T, out=destination)
    return output
