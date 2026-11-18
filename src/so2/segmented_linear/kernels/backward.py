from __future__ import annotations

import torch
import triton
import triton.language as tl

from .forward import fused_projection, use_fused_projection
from .utils import (
    INPUT, PROJECT, WEIGHT, batched_segment_gemm, grouped_gemm, grouped_weight_device,
    mix_weights, row_sizes, split_flattened, streamed_weight_device, use_equal_segment_bmm,
    use_grouped, use_uniform_node_gemm,
)


def input_gradient(dy, mixed, sizes):
    output = dy.new_empty((*dy.shape[:-1], mixed.shape[2]))
    grouped_gemm(
        dy.reshape(-1, dy.shape[-1]).contiguous(), mixed,
        output.reshape(-1, output.shape[-1]), row_sizes(dy, sizes), INPUT, False,
    )
    return output


@triton.jit
def weight_from_offsets_kernel(
    x, dy, offsets, gradient,
    factor: tl.constexpr, inputs: tl.constexpr, outputs: tl.constexpr,
    tile_inputs: tl.constexpr, tile_outputs: tl.constexpr, tile_rows: tl.constexpr,
):
    segment = tl.program_id(0)
    i = tl.program_id(1) * tile_inputs + tl.arange(0, tile_inputs)
    o = tl.program_id(2) * tile_outputs + tl.arange(0, tile_outputs)
    start = tl.load(offsets + segment).to(tl.int64) * factor
    end = tl.load(offsets + segment + 1).to(tl.int64) * factor
    r = tl.arange(0, tile_rows)
    acc = tl.zeros((tile_outputs, tile_inputs), tl.float32)
    for base in range(start, end, tile_rows):
        row = base + r
        lhs = tl.load(dy + row[None, :] * outputs + o[:, None],
                      (row[None, :] < end) & (o[:, None] < outputs), 0.)
        rhs = tl.load(x + row[:, None] * inputs + i[None, :],
                      (row[:, None] < end) & (i[None, :] < inputs), 0.)
        acc = tl.dot(lhs, rhs, acc, input_precision="ieee")
    destination = segment.to(tl.int64) * (inputs * outputs)
    tl.store(gradient + destination + o[:, None] * inputs + i[None, :],
             acc, (o[:, None] < outputs) & (i[None, :] < inputs))


def weight_gradient_from_offsets(dy, x, mixed, offsets):
    gradient = torch.empty_like(mixed)
    groups, outputs, inputs = mixed.shape
    factor = x.shape[1] if x.ndim == 3 else 1
    tile_inputs, tile_outputs, tile_rows = 32, 16, 16
    grid = (groups, triton.cdiv(inputs, tile_inputs), triton.cdiv(outputs, tile_outputs))
    with torch.cuda.device(x.device):
        weight_from_offsets_kernel[grid](
            x, dy, offsets, gradient, factor, inputs, outputs,
            tile_inputs, tile_outputs, tile_rows, num_warps=4,
        )
    return gradient


def use_gate_weight_vjp(x, sizes):
    groups = len(sizes)
    if not 16 <= groups <= 32 or x.ndim != 3 or x.shape[1:] != (2, 256):
        return False
    edges = x.shape[0]
    longest = max(sizes)
    # The unsplit reduction loses throughput when a long segment delays the batch.
    return (
        80 * groups <= edges <= 960 * groups and sum(sizes) == edges
        and min(sizes) >= 80 and longest <= min(1152, 2.25 * edges / groups)
        and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and not x.is_neg()
        and not torch.compiler.is_compiling() and not torch.are_deterministic_algorithms_enabled()
        and not torch.is_autocast_enabled("cuda") and not torch.backends.cuda.matmul.allow_tf32
        and torch.get_float32_matmul_precision() == "highest" and streamed_weight_device(x.device)
    )


@triton.jit
def gate_weight_kernel(x, dy, offsets, gradient,
                       inputs: tl.constexpr, half: tl.constexpr,
                       tile_outputs: tl.constexpr, tile_inputs: tl.constexpr, tile_rows: tl.constexpr):
    segment = tl.program_id(1)
    start = tl.load(offsets + segment)
    end = tl.load(offsets + segment + 1)
    tile = tl.program_id(0)
    input_tiles: tl.constexpr = (inputs + tile_inputs - 1) // tile_inputs
    i = (tile % input_tiles) * tile_inputs + tl.arange(0, tile_inputs)
    o = (tile // input_tiles) * tile_outputs + tl.arange(0, tile_outputs)
    k = tl.arange(0, tile_rows)
    u = tl.zeros((tile_outputs, tile_inputs), tl.float32)
    v = tl.zeros((tile_outputs, tile_inputs), tl.float32)
    cross = tl.zeros((tile_outputs, tile_inputs), tl.float32)
    for base in range(start, end, tile_rows):
        row = base + k
        features = x + row[:, None] * (2 * inputs) + i[None, :]
        seeds = dy + row[None, :] * (4 * half) + o[:, None]
        input_mask = (row[:, None] < end) & (i[None, :] < inputs)
        output_mask = (row[None, :] < end) & (o[:, None] < half)
        a = tl.load(features, input_mask, 0)
        b = tl.load(features + inputs, input_mask, 0)
        p = tl.load(seeds, output_mask, 0)
        q = tl.load(seeds + half, output_mask, 0)
        u = tl.dot(p, a, u, input_precision="ieee")
        v = tl.dot(q, b, v, input_precision="ieee")
        cross = tl.dot(p + q, a - b, cross, input_precision="ieee")
    # GroupedGate provides [[p, q], [q, -p]]. These products give both weight halves.
    target = gradient + (segment * (2 * half) + o[:, None]) * inputs + i[None, :]
    valid = (o[:, None] < half) & (i[None, :] < inputs)
    tl.store(target, u + v, valid)
    tl.store(target + half * inputs, cross - u + v, valid)


def gate_weight_vjp(dy, x, mixed, offsets):
    gradient = torch.empty_like(mixed)
    segments, outputs, inputs = mixed.shape
    half = outputs // 2
    tile_outputs, tile_inputs, tile_rows = 32, 64, 16
    tiles = triton.cdiv(half, tile_outputs) * triton.cdiv(inputs, tile_inputs)
    with torch.cuda.device(x.device):
        offsets.record_stream(torch.cuda.current_stream())
        gate_weight_kernel[(tiles, segments)](
            x, dy, offsets, gradient, inputs, half, tile_outputs, tile_inputs, tile_rows,
            num_warps=4, num_stages=2,
        )
    return gradient


def mixed_input_vjp(dy, mixed, sizes, grouped):
    if grouped:
        return input_gradient(dy, mixed, sizes)
    dx = dy.new_empty((*dy.shape[:-1], mixed.shape[-1]))
    for local_dy, weight, local_dx in zip(
        split_flattened(dy, sizes), mixed.unbind(0), split_flattened(dx, sizes), strict=True,
    ):
        torch.mm(local_dy, weight, out=local_dx)
    return dx


def mixed_weight_vjp(dy, x, mixed, sizes, offsets, gate_cotangent_offsets=None):
    if (
        gate_cotangent_offsets is not None and dy.shape == x.shape
        and dy.dtype == x.dtype and x.is_contiguous() and dy.is_contiguous()
        and not torch.compiler.is_compiling() and not torch.are_deterministic_algorithms_enabled()
        and not torch.is_autocast_enabled("cuda") and not torch.backends.cuda.matmul.allow_tf32
        and torch.get_float32_matmul_precision() == "highest"
    ):
        with torch.cuda.device(x.device):
            if not torch.cuda.is_current_stream_capturing():
                return gate_weight_vjp(dy, x, mixed, gate_cotangent_offsets)
    if x.is_cuda and offsets is not None:
        if sizes == (32,) * 16 and x.ndim == dy.ndim == 3 and x.is_contiguous() and dy.is_contiguous():
            rows = row_sizes(x, sizes)
            if use_uniform_node_gemm(x.flatten(0, 1), mixed, dy.flatten(0, 1), rows, PROJECT, False):
                # The receiver offsets are the prefix sums of these equal-sized groups.
                inputs = x.view(len(sizes), rows[0], x.shape[-1])
                gradients = dy.view(len(sizes), rows[0], dy.shape[-1])
                return torch.bmm(gradients.transpose(1, 2), inputs, out=torch.empty_like(mixed))
        return weight_gradient_from_offsets(dy, x, mixed, offsets)
    return segmented_weight_gradient(dy, x, sizes, mixed)


def segmented_weight_gradient(dy, x, sizes, mixed):
    gradient = torch.empty_like(mixed)
    if (
        dy.is_contiguous()
        and dy.dtype == torch.float32
        and dy.device == x.device
        and dy.shape[:-1] == x.shape[:-1]
        and dy.shape[-1] == mixed.shape[1]
        and not dy.is_neg()
        and not x.is_neg()
        and torch.get_float32_matmul_precision() == "highest"
        and use_grouped(x, mixed, sizes)
        and (
            streamed_weight_device(x.device)
            or (
                (x.ndim, *mixed.shape[1:]) in ((2, 640, 768), (3, 512, 512), (3, 256, 256))
                and grouped_weight_device(x.device)
            )
        )
    ):
        grouped_gemm(
            x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1]),
            gradient, row_sizes(x, sizes), WEIGHT, False,
        )
        return gradient

    for dy_s, x_s, dW_s in zip(
        split_flattened(dy, sizes), split_flattened(x, sizes),
        gradient.unbind(0), strict=True,
    ):
        torch.mm(dy_s.transpose(0, 1), x_s, out=dW_s)
    return gradient


def segmented_vjp(
    dy, x, weights, coefficients, bias, sizes, *, save_intermediates=False, mixed=None
):
    if mixed is None:
        mixed = mix_weights(weights, coefficients)
    if use_fused_projection(dy, mixed, None, sizes, input_vjp=True):
        dx = fused_projection(dy, mixed, None, sizes, input_vjp=True)
        dmixed = segmented_weight_gradient(dy, x, sizes, mixed)
    elif dy.is_contiguous() and use_grouped(x, mixed, sizes):
        dx = input_gradient(dy, mixed, sizes)
        dmixed = segmented_weight_gradient(dy, x, sizes, mixed)
    elif (
        use_equal_segment_bmm(dy, mixed, sizes, large=True)
        and torch.cuda.get_device_capability(dy.device) == (9, 0)
    ):
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        batched_segment_gemm(dy, mixed, dx, sizes, INPUT, False)
        dmixed = segmented_weight_gradient(dy, x, sizes, mixed)
    else:
        xs = split_flattened(x, sizes)
        dys = split_flattened(dy, sizes)
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        dmixed = torch.empty(
            (coefficients.shape[0], weights.shape[1], weights.shape[2]),
            device=dy.device,
            dtype=dy.dtype,
        )
        dxs = split_flattened(dx, sizes)
        for x_s, dy_s, W_s, dx_s, dW_s in zip(
            xs, dys, mixed.unbind(0), dxs, dmixed.unbind(0), strict=True
        ):
            torch.mm(dy_s, W_s, out=dx_s)
            torch.mm(dy_s.transpose(0, 1), x_s, out=dW_s)

    flat_dmixed = dmixed.flatten(1)
    flat_weights = weights.flatten(1)
    dweights = torch.mm(coefficients.transpose(0, 1), flat_dmixed).view_as(weights)
    dcoefficients = torch.mm(flat_dmixed, flat_weights.transpose(0, 1))
    dbias = dy.reshape(-1, dy.shape[-1]).sum(dim=0) if bias is not None else None
    gradients = dx, dweights, dcoefficients, dbias
    if save_intermediates:
        return gradients, (mixed, dmixed)
    return gradients
