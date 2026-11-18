"""Gate SO(2) node and edge coefficients with explicit scalar logits."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from ...utils import is_compiling
from .layout import get_angular_layout
from .tensor_contracts import require_cuda_fp32


class CoefficientGateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, x, gating, lmax, mmax, layout, preserve_strides):
        values = coefficient_gate_backward(
            grad, x, gating, lmax=lmax, mmax=mmax, layout=layout,
            preserve_strides=preserve_strides,
        )
        ctx.save_for_backward(grad, x, gating)
        ctx.settings = (lmax, mmax, layout, preserve_strides)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, input_direction, gating_direction):
        grad, x, gating = ctx.saved_tensors
        if input_direction is None:
            input_direction = torch.zeros_like(x)
        if gating_direction is None:
            gating_direction = torch.zeros_like(gating)
        lmax, mmax, layout, preserve_strides = ctx.settings
        values = coefficient_gate_double_backward(
            grad,
            x,
            gating,
            input_direction,
            gating_direction.contiguous(),
            lmax=lmax,
            mmax=mmax,
            layout=layout,
            preserve_strides=preserve_strides,
        )
        return *values, None, None, None, None


class FusedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gating, lmax, mmax, layout, preserve_strides):
        output = coefficient_gate_forward(
            x, gating, lmax=lmax, mmax=mmax, layout=layout,
            preserve_strides=preserve_strides,
        )
        ctx.save_for_backward(x, gating)
        ctx.settings = (lmax, mmax, layout, preserve_strides)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, gating = ctx.saved_tensors
        lmax, mmax, layout, preserve_strides = ctx.settings
        if not torch.is_grad_enabled():
            values = coefficient_gate_backward(
                grad.contiguous(), x, gating, lmax=lmax, mmax=mmax, layout=layout,
                preserve_strides=preserve_strides,
            )
            return *values, None, None, None, None
        values = CoefficientGateBackward.apply(grad, x, gating, lmax, mmax, layout, preserve_strides)
        return *values, None, None, None, None


def fused_gate(gating, x, *, lmax, mmax, layout="lmajor"):
    """Gate edge coefficients while preserving their storage strides."""
    layout_id = {"lmajor": 0, "mmajor": 1}[layout]
    if gating.stride(-1) != 1:
        gating = gating.contiguous()
    return FusedGate.apply(x, gating, lmax, mmax, layout_id, True)


def node_gate(gating, x, *, lmax):
    """Apply a scalar SiLU and one sigmoid gate per positive degree.

    Parameters
    ----------
    gating : Tensor
        CUDA FP32 logits with ``N * L * C`` entries and first dimension N.
        The remaining entries are flattened in degree/channel order, for
        example ``(N, L, C)``. These are logits, before sigmoid.
    x : Tensor, shape (N, (L+1)**2, C)
        CUDA FP32 degree-major coefficients. Degree l occupies rows
        ``l*l:(l+1)**2``, with the scalar coefficient at row zero.
    lmax : int
        Maximum degree L, including L=0.

    Returns
    -------
    Tensor
        Features with the shape of x. Row zero is ``silu(x[:, 0])``.
        Degree l>0 is multiplied by ``sigmoid(gating[:, l-1])``.

    The kernel combines scalar activation, gate evaluation and feature writes.
    Backward and double backward cover x and gating, including cotangents.
    """
    if x.ndim != 3 or lmax < 0 or x.shape[1] != (lmax + 1) ** 2:
        raise ValueError("node gate requires complete degree-major node features")
    if (
        gating.ndim < 1
        or gating.shape[0] != x.shape[0]
        or gating.numel() != x.shape[0] * lmax * x.shape[2]
    ):
        raise ValueError(
            "node gates must provide one value per positive degree and channel"
        )
    require_cuda_fp32(x, gating, operation="node gate")
    return FusedGate.apply(x, gating.contiguous(), lmax, lmax, 0, False)


@triton.jit
def silu_terms(value):
    sigmoid = tl.sigmoid(value)
    silu = value * sigmoid
    first = sigmoid + value * sigmoid * (1.0 - sigmoid)
    second = 2.0 * sigmoid * (1.0 - sigmoid) + value * sigmoid * (1.0 - sigmoid) * (
        1.0 - 2.0 * sigmoid
    )
    return silu, first, second


def gate_layout(x, lmax, mmax, layout):
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
    if layout == 0:
        row_degrees = spec.lmajor_degrees
        has_scalar = True
        expected_rows = spec.reduced_components
    elif layout == 1:
        row_degrees = spec.reduced_degrees
        has_scalar = True
        expected_rows = spec.reduced_components
    else:
        raise ValueError(f"unknown gate layout: {layout}")
    if x.shape[1] != expected_rows:
        raise ValueError(
            f"gate input does not match L/M/layout: {x.shape[1]} != {expected_rows}"
        )
    degree_row_masks = tuple(
        row_degree == degree for degree in range(lmax + 1) for row_degree in row_degrees
    )
    return spec, row_degrees, degree_row_masks, has_scalar


def empty_features(x, preserve_strides):
    if preserve_strides:
        return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    return torch.empty_like(x)


def use_node_tiles(x, lmax, mmax, layout, preserve_strides):
    return (
        not preserve_strides and layout == 0 and lmax == mmax
        and lmax in (3, 4) and x.shape[2] == 128
        and 96 <= x.shape[0] <= 2083
        and not is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))
    )


@triton.jit
def apply_gate(
    X, G, DY, HX, HG, Y, DX, DG,
    xs, gs, dys, hxs, hgs, ys, dxs, dgs,
    nodes, channels: tl.constexpr, block_rows: tl.constexpr,
    block_nodes: tl.constexpr, block_channels: tl.constexpr, phase: tl.constexpr,
):
    node = tl.program_id(0) * block_nodes + tl.arange(0, block_nodes)[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    n = node[:, None, :]
    c = channel[:, None, :]
    ell = tl.program_id(1)
    valid_2d = (node < nodes) & (channel < channels)
    valid = valid_2d[:, None, :]
    if ell == 0:
        scalar_x = tl.load(X + n * xs[0] + c * xs[2], valid, 0.0)
        scalar_y, scalar_slope, scalar_curvature = silu_terms(scalar_x)
        if phase == 0:
            tl.store(Y + n * ys[0] + c * ys[2], scalar_y, valid)
        else:
            scalar_dy = tl.load(DY + n * dys[0] + c * dys[2], valid, 0.0)
            if phase == 1:
                scalar_dx = scalar_dy * scalar_slope
            else:
                scalar_hx = tl.load(HX + n * hxs[0] + c * hxs[2], valid, 0.0)
                scalar_dx = scalar_dy * scalar_curvature * scalar_hx
                tl.store(Y + n * ys[0] + c * ys[2], scalar_slope * scalar_hx, valid)
            tl.store(DX + n * dxs[0] + c * dxs[2], scalar_dx, valid)
    else:
        i = tl.arange(0, block_rows)[None, :, None]
        row = ell * ell + i
        mask = valid & (i < 2 * ell + 1)
        gate_col = (ell - 1) * channels + channel
        g = tl.load(G + node * gs[0] + gate_col * gs[1], valid_2d, 0.0)
        gate_2d = tl.sigmoid(g)
        gate = gate_2d[:, None, :]
        x = tl.load(X + n * xs[0] + row * xs[1] + c * xs[2], mask, 0.0)
        if phase == 0:
            tl.store(Y + n * ys[0] + row * ys[1] + c * ys[2], x * gate, mask)
        else:
            dy = tl.load(DY + n * dys[0] + row * dys[1] + c * dys[2], mask, 0.0)
            slope = gate_2d * (1.0 - gate_2d)
            if phase == 1:
                dx = dy * gate
                dg = tl.sum(dy * x, axis=1) * slope
            else:
                hx = tl.load(HX + n * hxs[0] + row * hxs[1] + c * hxs[2], mask, 0.0)
                hg = tl.load(HG + node * hgs[0] + gate_col * hgs[1], valid_2d, 0.0)
                hg_slope = hg * slope
                dx = dy * hg_slope[:, None, :]
                dg = (
                    tl.sum(dy * hx, axis=1) * slope
                    + tl.sum(dy * x, axis=1) * (hg_slope * (1.0 - 2.0 * gate_2d))
                )
                tl.store(Y + n * ys[0] + row * ys[1] + c * ys[2], gate * hx + x * hg_slope[:, None, :], mask)
            tl.store(DX + n * dxs[0] + row * dxs[1] + c * dxs[2], dx, mask)
            tl.store(DG + node * dgs[0] + gate_col * dgs[1], dg, valid_2d)


def tiled_node_gate(x, gate, *, lmax, grad=None, directions=None):
    nodes, _, channels = x.shape
    gate_shape = gate.shape
    gate = gate.reshape(nodes, -1)
    mode = 0 if grad is None else 1 if directions is None else 2
    if mode == 2:
        hx, hg = directions
        hg = hg.reshape_as(gate)
    else:
        hx, hg = x, gate
    if grad is None:
        grad = x

    result = x
    if mode == 0:
        result = torch.empty_like(x)
    elif mode == 2:
        result = torch.empty_like(grad)
    dx = torch.empty_like(x) if mode else x
    dg = torch.empty_like(gate) if mode else gate
    block_nodes = 1 if mode == 0 else 4
    apply_gate[(triton.cdiv(nodes, block_nodes), lmax + 1)](
        x, gate, grad, hx, hg, result, dx, dg,
        x.stride(), gate.stride(), grad.stride(), hx.stride(), hg.stride(),
        result.stride(), dx.stride(), dg.stride(),
        nodes, channels, triton.next_power_of_2(2 * lmax + 1), block_nodes, channels, mode,
        num_warps=4,
    )
    if mode == 0:
        return result
    if mode == 1:
        return dx, dg.reshape(gate_shape)
    return result, dx, dg.reshape(gate_shape)


@triton.jit
def gate_forward_kernel(
    x_ptr, gating_ptr, output_ptr, items,
    x_stride_item, x_stride_row, x_stride_channel,
    gating_stride_item,
    output_stride_item, output_stride_row, output_stride_channel,
    channels: tl.constexpr,
    degree_count: tl.constexpr,
    degree_reduced_counts: tl.constexpr,
    degree_reduced_offsets: tl.constexpr,
    row_degrees: tl.constexpr,
    reduced_components: tl.constexpr,
    has_scalar: tl.constexpr,
    first_vector_row: tl.constexpr,
    block_channels: tl.constexpr,
):
    item = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    if has_scalar:
        scalar = tl.load(
            x_ptr + item * x_stride_item + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        silu, _, _ = silu_terms(scalar)
        tl.store(
            output_ptr + item * output_stride_item + channel * output_stride_channel,
            silu,
            mask=channel_mask,
        )
    for row in tl.static_range(first_vector_row, reduced_components):
        degree = list(row_degrees)[row]
        raw_gate = tl.load(
            gating_ptr + item * gating_stride_item + (degree - 1) * channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(raw_gate)
        value = tl.load(
            x_ptr
            + item * x_stride_item
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr
            + item * output_stride_item
            + row * output_stride_row
            + channel * output_stride_channel,
            value * gate,
            mask=channel_mask,
        )


def coefficient_gate_forward(x, gating, *, lmax, mmax, layout, preserve_strides):
    spec, row_degrees, degree_row_masks, has_scalar = gate_layout(x, lmax, mmax, layout)
    if use_node_tiles(x, lmax, mmax, layout, preserve_strides):
        return tiled_node_gate(x, gating, lmax=lmax)
    reduced_components = x.shape[1]
    output = empty_features(x, preserve_strides)
    gate_forward_kernel[(x.shape[0],)](
        x, gating, output, x.shape[0],
        *x.stride(),
        gating.stride(0),
        *output.stride(),
        channels=spec.channels,
        degree_count=spec.lmax + 1,
        degree_reduced_counts=spec.degree_reduced_counts,
        degree_reduced_offsets=spec.degree_reduced_offsets,
        row_degrees=row_degrees,
        reduced_components=reduced_components,
        has_scalar=has_scalar,
        first_vector_row=int(has_scalar),
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return output


@triton.jit
def gate_backward_kernel(
    grad_ptr,
    x_ptr,
    gating_ptr,
    input_gradient_ptr,
    gating_gradient_ptr,
    grad_stride_item, grad_stride_row, grad_stride_channel,
    x_stride_item, x_stride_row, x_stride_channel,
    input_gradient_stride_item, input_gradient_stride_row, input_gradient_stride_channel,
    gating_stride_item,
    gating_gradient_stride_item,
    channels: tl.constexpr,
    degree_count: tl.constexpr,
    degree_reduced_counts: tl.constexpr,
    degree_reduced_offsets: tl.constexpr,
    row_degrees: tl.constexpr,
    degree_row_masks: tl.constexpr,
    reduced_components: tl.constexpr,
    has_scalar: tl.constexpr,
    first_vector_row: tl.constexpr,
    block_channels: tl.constexpr,
):
    item = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    if has_scalar:
        scalar = tl.load(
            x_ptr + item * x_stride_item + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        _, silu_first, _ = silu_terms(scalar)
        scalar_grad = tl.load(
            grad_ptr + item * grad_stride_item + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            input_gradient_ptr
            + item * input_gradient_stride_item
            + channel * input_gradient_stride_channel,
            scalar_grad * silu_first,
            mask=channel_mask,
        )
    for degree in tl.static_range(1, degree_count):
        raw_gate = tl.load(
            gating_ptr + item * gating_stride_item + (degree - 1) * channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(raw_gate)
        gate_first = gate * (1.0 - gate)
        contraction = tl.zeros(channel.shape, dtype=tl.float32)
        for row in tl.static_range(first_vector_row, reduced_components):
            if list(degree_row_masks)[degree * reduced_components + row]:
                grad = tl.load(
                    grad_ptr
                    + item * grad_stride_item
                    + row * grad_stride_row
                    + channel * grad_stride_channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                value = tl.load(
                    x_ptr
                    + item * x_stride_item
                    + row * x_stride_row
                    + channel * x_stride_channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    input_gradient_ptr
                    + item * input_gradient_stride_item
                    + row * input_gradient_stride_row
                    + channel * input_gradient_stride_channel,
                    grad * gate,
                    mask=channel_mask,
                )
                contraction += grad * value
        tl.store(
            gating_gradient_ptr
            + item * gating_gradient_stride_item
            + (degree - 1) * channels
            + channel,
            gate_first * contraction,
            mask=channel_mask,
        )


def coefficient_gate_backward(grad, x, gating, *, lmax, mmax, layout, preserve_strides):
    spec, row_degrees, degree_row_masks, has_scalar = gate_layout(x, lmax, mmax, layout)
    if use_node_tiles(x, lmax, mmax, layout, preserve_strides):
        return tiled_node_gate(x, gating, lmax=lmax, grad=grad)
    reduced_components = x.shape[1]
    input_gradient = empty_features(x, preserve_strides)
    gating_gradient = torch.empty_like(gating)
    gate_backward_kernel[(x.shape[0],)](
        grad, x, gating, input_gradient, gating_gradient,
        *grad.stride(),
        *x.stride(),
        *input_gradient.stride(),
        gating.stride(0),
        gating_gradient.stride(0),
        channels=spec.channels,
        degree_count=spec.lmax + 1,
        degree_reduced_counts=spec.degree_reduced_counts,
        degree_reduced_offsets=spec.degree_reduced_offsets,
        row_degrees=row_degrees,
        degree_row_masks=degree_row_masks,
        reduced_components=reduced_components,
        has_scalar=has_scalar,
        first_vector_row=int(has_scalar),
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return input_gradient, gating_gradient


@triton.jit
def gate_double_backward_kernel(
    grad_ptr,
    x_ptr,
    gating_ptr,
    input_direction_ptr,
    gating_direction_ptr,
    grad_output_gradient_ptr,
    input_gradient_ptr,
    gating_gradient_ptr,
    grad_stride_item, grad_stride_row, grad_stride_channel,
    x_stride_item, x_stride_row, x_stride_channel,
    input_direction_stride_item,
    input_direction_stride_row,
    input_direction_stride_channel,
    grad_output_gradient_stride_item,
    grad_output_gradient_stride_row,
    grad_output_gradient_stride_channel,
    input_gradient_stride_item, input_gradient_stride_row, input_gradient_stride_channel,
    gating_stride_item,
    gating_direction_stride_item,
    gating_gradient_stride_item,
    channels: tl.constexpr,
    degree_count: tl.constexpr,
    degree_reduced_counts: tl.constexpr,
    degree_reduced_offsets: tl.constexpr,
    row_degrees: tl.constexpr,
    degree_row_masks: tl.constexpr,
    reduced_components: tl.constexpr,
    has_scalar: tl.constexpr,
    first_vector_row: tl.constexpr,
    block_channels: tl.constexpr,
):
    item = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    if has_scalar:
        scalar = tl.load(
            x_ptr + item * x_stride_item + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        scalar_direction = tl.load(
            input_direction_ptr
            + item * input_direction_stride_item
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        scalar_grad = tl.load(
            grad_ptr + item * grad_stride_item + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        _, silu_first, silu_second = silu_terms(scalar)
        tl.store(
            grad_output_gradient_ptr
            + item * grad_output_gradient_stride_item
            + channel * grad_output_gradient_stride_channel,
            silu_first * scalar_direction,
            mask=channel_mask,
        )
        tl.store(
            input_gradient_ptr
            + item * input_gradient_stride_item
            + channel * input_gradient_stride_channel,
            scalar_grad * silu_second * scalar_direction,
            mask=channel_mask,
        )
    for degree in tl.static_range(1, degree_count):
        gate_address = item * gating_stride_item + (degree - 1) * channels + channel
        raw_gate = tl.load(gating_ptr + gate_address, mask=channel_mask, other=0.0).to(
            tl.float32
        )
        gate_direction = tl.load(
            gating_direction_ptr
            + item * gating_direction_stride_item
            + (degree - 1) * channels
            + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(raw_gate)
        gate_first = gate * (1.0 - gate)
        gate_second = gate_first * (1.0 - 2.0 * gate)
        contraction_value = tl.zeros(channel.shape, dtype=tl.float32)
        contraction_direction = tl.zeros(channel.shape, dtype=tl.float32)
        for row in tl.static_range(first_vector_row, reduced_components):
            if list(degree_row_masks)[degree * reduced_components + row]:
                grad_address = (
                    item * grad_stride_item
                    + row * grad_stride_row
                    + channel * grad_stride_channel
                )
                value_address = (
                    item * x_stride_item
                    + row * x_stride_row
                    + channel * x_stride_channel
                )
                direction_address = (
                    item * input_direction_stride_item
                    + row * input_direction_stride_row
                    + channel * input_direction_stride_channel
                )
                grad = tl.load(
                    grad_ptr + grad_address, mask=channel_mask, other=0.0
                ).to(tl.float32)
                value = tl.load(x_ptr + value_address, mask=channel_mask, other=0.0).to(
                    tl.float32
                )
                value_direction = tl.load(
                    input_direction_ptr + direction_address,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    grad_output_gradient_ptr
                    + item * grad_output_gradient_stride_item
                    + row * grad_output_gradient_stride_row
                    + channel * grad_output_gradient_stride_channel,
                    gate * value_direction + value * gate_first * gate_direction,
                    mask=channel_mask,
                )
                tl.store(
                    input_gradient_ptr
                    + item * input_gradient_stride_item
                    + row * input_gradient_stride_row
                    + channel * input_gradient_stride_channel,
                    grad * gate_first * gate_direction,
                    mask=channel_mask,
                )
                contraction_value += grad * value
                contraction_direction += grad * value_direction
        tl.store(
            gating_gradient_ptr
            + item * gating_gradient_stride_item
            + (degree - 1) * channels
            + channel,
            gate_first * contraction_direction
            + gate_second * gate_direction * contraction_value,
            mask=channel_mask,
        )


def coefficient_gate_double_backward(grad, x, gating, input_direction, gating_direction, *, lmax, mmax, layout, preserve_strides):
    spec, row_degrees, degree_row_masks, has_scalar = gate_layout(x, lmax, mmax, layout)
    if use_node_tiles(x, lmax, mmax, layout, preserve_strides):
        return tiled_node_gate(x, gating, lmax=lmax, grad=grad,
            directions=(input_direction, gating_direction))
    reduced_components = x.shape[1]
    grad_output_gradient = torch.empty_like(grad)
    input_gradient = empty_features(x, preserve_strides)
    gating_gradient = torch.empty_like(gating)
    gate_double_backward_kernel[(x.shape[0],)](
        grad, x, gating, input_direction, gating_direction,
        grad_output_gradient, input_gradient, gating_gradient,
        *grad.stride(),
        *x.stride(),
        *input_direction.stride(),
        *grad_output_gradient.stride(),
        *input_gradient.stride(),
        gating.stride(0),
        gating_direction.stride(0),
        gating_gradient.stride(0),
        channels=spec.channels,
        degree_count=spec.lmax + 1,
        degree_reduced_counts=spec.degree_reduced_counts,
        degree_reduced_offsets=spec.degree_reduced_offsets,
        row_degrees=row_degrees,
        degree_row_masks=degree_row_masks,
        reduced_components=reduced_components,
        has_scalar=has_scalar,
        first_vector_row=int(has_scalar),
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return grad_output_gradient, input_gradient, gating_gradient
