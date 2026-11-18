"""SO(2) degree normalization for training and frozen inference."""

from __future__ import annotations

import math
from importlib.util import find_spec

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op, wrap_triton

from ...utils import is_compiling
from .layout import get_angular_layout
from .tensor_contracts import allow_atomic_reduction, require_cuda_fp32


@torch.compiler.assume_constant_result
def hopper_degree_kernels(device):
    return torch.version.hip is None and torch.cuda.get_device_capability(device) == (9, 0)


def degree_norm(
    x, weight, bias, *, epsilon=1e-5, lmax=None, coefficient_major_output=False,
    input_only=False,
):
    """Normalize degree-major coefficients with equal weight per degree.

    ``input_only`` selects a first-order path where available. The affine
    parameters must be frozen and only the input gradient is required.
    """
    if input_only:
        if weight.requires_grad or bias.requires_grad:
            raise ValueError("input-only normalization requires frozen affine parameters")
        if (
            lmax in (None, 2) and x.ndim == 3 and x.shape[1:] == (9, 128)
            and x.shape[0] > 0 and x.is_cuda and x.dtype == torch.float32
            and weight.shape == (3, 128) and bias.shape == (128,)
            and weight.is_contiguous() and bias.is_contiguous()
            and hopper_degree_kernels(x.device)
        ):
            require_cuda_fp32(x, weight, bias, operation="degree normalization")
            if coefficient_major_output:
                return coefficient_norm(x, weight, bias, epsilon)
            return frozen_degree_norm(x.contiguous(), weight, bias, epsilon)[0]
    if lmax is None and not coefficient_major_output:
        return fused_equivariant_norm(x, weight, bias, epsilon)
    return fused_rms_norm(
        x, weight, bias, lmax=weight.shape[0] - 1 if lmax is None else lmax,
        epsilon=epsilon, coefficient_major_output=coefficient_major_output,
    )


class EquivariantNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, inputs, weight, bias, mean, rstd):
        gradients = backward(grad_output, inputs, weight, mean, rstd)
        ctx.save_for_backward(grad_output, inputs, weight, bias, mean, rstd)
        return gradients

    @staticmethod
    @once_differentiable
    def backward(ctx, input_direction, weight_direction, bias_direction):
        grad_output, inputs, weight, bias, mean, rstd = ctx.saved_tensors
        if input_direction is None:
            input_direction = torch.zeros_like(inputs)
        if weight_direction is None:
            weight_direction = torch.zeros_like(weight)
        if bias_direction is None:
            bias_direction = torch.zeros_like(bias)
        dgrad, dx, dweight = double_backward(
            grad_output, inputs, weight,
            mean, rstd,
            input_direction.contiguous(),
            weight_direction.contiguous(),
            bias_direction.contiguous(),
        )
        return dgrad, dx, dweight, torch.zeros_like(bias), None, None


class EquivariantNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, eps):
        outputs, mean, rstd = equivariant_norm_forward(inputs, weight, bias, eps)
        ctx.save_for_backward(inputs, weight, bias, mean, rstd)
        ctx.epsilon = eps
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias, mean, rstd = ctx.saved_tensors
        if (
            not torch.is_grad_enabled()
            and ctx.needs_input_grad[0]
            and not any(ctx.needs_input_grad[1:3])
        ):
            return (
                backward_input(grad_output.contiguous(), inputs, weight, mean, rstd),
                None,
                None,
                None,
            )
        lmax = weight.shape[0] - 1
        if any(ctx.needs_input_grad[1:3]) and use_flat_schedule(inputs, lmax):
            gradients = RMSNormBackward.apply(
                grad_output, inputs, weight, bias, lmax, ctx.epsilon
            )
            return (*gradients, None)
        gradients = EquivariantNormBackward.apply(
            grad_output.contiguous(), inputs, weight, bias, mean, rstd
        )
        return (*gradients, None)


def fused_equivariant_norm(inputs, weight, bias, eps):
    """Apply centered, degree-balanced normalization with L from the weights.

    Parameters
    ----------
    inputs : Tensor, shape (N, (L+1)**2, C)
        CUDA FP32 degree-major coefficients. Coefficient zero is scalar.
    weight : Tensor, shape (L+1, C)
        Contiguous CUDA FP32 degree/channel scales. Its first axis defines L.
    bias : Tensor, shape (C,)
        Contiguous CUDA FP32 scalar bias.
    eps : float
        Positive value added to the mean square before rsqrt.

    Returns
    -------
    Tensor, shape (N, (L+1)**2, C)
        Scalar channels are centered. Each degree contributes its mean square
        equally to the shared normalization factor. The affine bias applies
        only to coefficient zero. Logical axes match inputs.

    This entry chooses a flattened or row-wise reduction kernel from the node
    width. Derivatives through third order cover inputs, affine tensors and
    cotangents. Input strides are normalized internally.
    """

    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("equivariant norm affine parameters must be tensors")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("equivariant norm affine tensors must be contiguous")
    require_cuda_fp32(inputs, weight, bias, operation="equivariant norm")
    inputs = inputs.contiguous()
    if inputs.ndim == 3 and inputs.shape[1] * inputs.shape[2] > MAX_BLOCK_ELEMENTS:
        _, degrees, _, _, _ = validate(inputs, weight, bias)
        return fused_rms_norm(inputs, weight, bias, lmax=degrees - 1, epsilon=eps)
    return EquivariantNorm.apply(inputs, weight, bias, eps)


class RMSNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, x, weight, bias, lmax, epsilon):
        gradients = vjp(grad, x, weight, bias, lmax=lmax, epsilon=epsilon)
        ctx.save_for_backward(grad, x, weight, bias)
        ctx.norm_config = (lmax, epsilon)
        return gradients

    @staticmethod
    def backward(ctx, input_direction, weight_direction, bias_direction):
        grad, x, weight, bias = ctx.saved_tensors
        if input_direction is None:
            input_direction = torch.zeros_like(x)
        if weight_direction is None:
            weight_direction = torch.zeros_like(weight)
        if bias_direction is None:
            bias_direction = torch.zeros_like(bias)
        weight_direction = weight_direction.contiguous()
        bias_direction = bias_direction.contiguous()
        lmax, epsilon = ctx.norm_config
        arguments = (
            grad,
            x, weight, bias,
            input_direction,
            weight_direction,
            bias_direction,
        )
        if torch.is_grad_enabled():
            gradients = RMSNormDoubleBackward.apply(lmax, epsilon, *arguments)
        else:
            gradients = hvp(*arguments, lmax=lmax, epsilon=epsilon)
        return gradients[0], gradients[1], gradients[2], gradients[3], None, None


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lmax, epsilon, coefficient_major_output):
        output = rms_norm_forward(
            x, weight, bias,
            lmax=lmax, epsilon=epsilon,
            coefficient_major_output=coefficient_major_output,
        )
        ctx.save_for_backward(x, weight, bias)
        ctx.norm_config = (lmax, epsilon)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors
        lmax, epsilon = ctx.norm_config
        gradients = RMSNormBackward.apply(grad, x, weight, bias, lmax, epsilon)
        return gradients[0], gradients[1], gradients[2], None, None, None


def fused_rms_norm(
    x, weight, bias,
    *,
    lmax, epsilon,
    coefficient_major_output=False,
):
    """Center scalar channels and apply degree-balanced RMS normalization.

    Parameters
    ----------
    x : Tensor, shape (N, (L+1)**2, C)
        CUDA FP32 degree-major coefficients, with coefficient zero scalar.
    weight : Tensor, shape (L+1, C)
        Contiguous CUDA FP32 scale for each degree and channel.
    bias : Tensor, shape (C,)
        Contiguous CUDA FP32 bias for coefficient zero only.
    lmax : int
        Maximum degree L.
    epsilon : float
        Positive value added to the degree-balanced mean square before rsqrt.
    coefficient_major_output : bool, default False
        Return the same logical axes with coefficient-major storage when true.

    Returns
    -------
    Tensor, shape (N, (L+1)**2, C)
        Center row zero over channels. For each node, average the mean squares
        within each degree, then average these L+1 values. Normalize all rows
        by that shared RMS, multiply by their degree scale, and add scalar bias.
        Coefficient-major storage has strides ``(C, N*C, 1)``.

    The kernel combines centering, reductions and affine feature writes.
    Derivatives through third order cover x, weight, bias and cotangents.
    """
    if x.ndim != 3:
        raise ValueError("equivariant RMSNorm input must have shape (N, (L+1)**2, C)")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("equivariant RMSNorm affine parameters must be tensors")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("equivariant RMSNorm affine tensors must be contiguous")
    require_cuda_fp32(x, weight, bias, operation="equivariant RMSNorm")
    normalization_layout(x, weight, bias, lmax)
    return FusedRMSNorm.apply(x, weight, bias, lmax, epsilon, coefficient_major_output)


FROZEN_NORM_DEGREES = (0, 1, 1, 1, 2, 2, 2, 2, 2)


@triton_op("trim_h100::uma_visible_norm_forward", mutates_args=())
def frozen_degree_norm(x: Tensor, weight: Tensor, bias: Tensor, epsilon: float) -> tuple[Tensor, Tensor, Tensor]:
    out = torch.empty_like(x)
    mean = x.new_empty((x.shape[0],))
    rstd = torch.empty_like(mean)
    wrap_triton(equivariant_norm_forward_kernel)[(x.shape[0],)](
        x, weight, bias, out, mean, rstd, epsilon,
        elements_per_node=1152, channels=128, degrees=3, block_elements=2048, num_warps=8,
    )
    return out, mean, rstd


@triton_op("trim_h100::uma_visible_norm_input_vjp", mutates_args=())
def norm_input_vjp(grad: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor) -> Tensor:
    dx = torch.empty_like(x)
    grad = grad.resolve_neg().contiguous()
    wrap_triton(equivariant_norm_backward_kernel)[(x.shape[0],)](
        grad, x, weight, mean, rstd, dx, dx, dx,
        elements_per_node=1152, channels=128, degrees=3, block_elements=2048,
        compute_parameter_gradients=False, num_warps=8,
    )
    return dx


def frozen_degree_norm_setup_context(ctx, inputs, output):
    x, weight, _, _ = inputs
    _, mean, rstd = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(mean, rstd)
    ctx.save_for_backward(x, weight, mean, rstd)


def frozen_degree_norm_backward(ctx, grad, _mean_grad, _rstd_grad):
    if grad is None:
        return None, None, None, None
    return norm_input_vjp(grad, *ctx.saved_tensors), None, None, None


frozen_degree_norm.register_autograd(frozen_degree_norm_backward, setup_context=frozen_degree_norm_setup_context)


@triton_op("trim_h100::uma_ffn_coefficient_norm", mutates_args=())
def coefficient_norm(x: Tensor, weight: Tensor, bias: Tensor, epsilon: float) -> Tensor:
    nodes = x.shape[0]
    out = torch.empty_strided(x.shape, (128, nodes * 128, 1), dtype=x.dtype, device=x.device)
    wrap_triton(forward_kernel)[(nodes,)](
        x, weight, bias, out, *x.stride(), *out.stride(),
        channels=128, rows=9, degrees=FROZEN_NORM_DEGREES, degree_count=3,
        epsilon=epsilon, block_channels=128, num_warps=4,
    )
    return out


@triton_op("trim_h100::uma_ffn_coefficient_norm_vjp", mutates_args=())
def coefficient_norm_vjp(grad: Tensor, x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    grad = grad.resolve_neg()
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    wrap_triton(vjp_kernel)[(x.shape[0],)](
        grad, grad, x, weight, dx, dx, dx, *grad.stride(), *x.stride(), *dx.stride(),
        channels=128, rows=9, degrees=FROZEN_NORM_DEGREES, degree_count=3, epsilon=epsilon,
        has_residual=False, compute_parameter_gradients=False, block_channels=128, num_warps=4,
    )
    return dx


def save_norm(ctx, inputs, output):
    x, weight, _, epsilon = inputs
    ctx.save_for_backward(x, weight)
    ctx.epsilon = epsilon


def backward_norm(ctx, grad):
    return coefficient_norm_vjp(grad, *ctx.saved_tensors, ctx.epsilon), None, None, None


coefficient_norm.register_autograd(backward_norm, setup_context=save_norm)


MAX_BLOCK_ELEMENTS = 4096


@triton.jit
def component_degree(component, degrees: tl.constexpr):
    degree = tl.zeros_like(component)
    for candidate in tl.static_range(1, degrees):
        degree += component >= candidate * candidate
    return degree


def validate(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if (
        not inputs.is_cuda
        or inputs.dtype != torch.float32
        or not inputs.is_contiguous()
    ):
        raise ValueError("equivariant norm requires contiguous CUDA FP32 input")
    if inputs.ndim != 3:
        raise ValueError(f"unexpected equivariant norm input: {inputs.shape}")
    components = inputs.shape[1]
    channels = inputs.shape[2]
    if components < 1 or channels < 1:
        raise ValueError(
            "equivariant norm requires positive component and channel counts"
        )
    degrees = math.isqrt(components)
    if degrees * degrees != components:
        raise ValueError(f"component count is not a square: {components}")
    if weight.shape != (degrees, channels) or (
        bias is not None and bias.shape != (channels,)
    ):
        raise ValueError("unexpected equivariant norm affine parameters")
    elements_per_node = components * channels
    block_elements = triton.next_power_of_2(elements_per_node)
    num_warps = 8 if block_elements >= 2048 else 4
    return channels, degrees, elements_per_node, block_elements, num_warps


@triton.jit
def jvp_kernel(
    x_ptr,
    weight_ptr,
    input_direction_ptr,
    weight_direction_ptr,
    bias_direction_ptr,
    output_ptr,
    x_stride_node, x_stride_row, x_stride_channel,
    input_direction_stride_node, input_direction_stride_row, input_direction_stride_channel,
    output_stride_node, output_stride_row, output_stride_channel,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    has_parameter_direction: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    scalar = tl.load(
        x_ptr + node * x_stride_node + channel * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_direction = tl.load(
        input_direction_ptr
        + node * input_direction_stride_node
        + channel * input_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_mean = tl.sum(scalar, axis=0) / channels
    scalar_direction_mean = tl.sum(scalar_direction, axis=0) / channels
    norm = tl.zeros((), dtype=tl.float32)
    norm_direction_half = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        norm += balance * tl.sum(value * value, axis=0)
        norm_direction_half += balance * tl.sum(value * value_direction, axis=0)
    inverse = tl.rsqrt(norm + epsilon)
    inverse_direction = -inverse * inverse * inverse * norm_direction_half
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        weight_direction = tl.zeros(channel.shape, dtype=tl.float32)
        if has_parameter_direction:
            weight_direction = tl.load(
                weight_direction_ptr + degree * channels + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
        result = (
            value_direction * inverse * weight
            + value * inverse_direction * weight
            + value * inverse * weight_direction
        )
        if row == 0 and has_parameter_direction:
            result += tl.load(bias_direction_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr
            + node * output_stride_node
            + row * output_stride_row
            + channel * output_stride_channel,
            result,
            mask=channel_mask,
        )


def normalization_layout(x, weight, bias, lmax):
    spec = get_angular_layout(lmax=lmax, mmax=lmax, channels=x.shape[2])
    if tuple(x.shape[1:]) != (spec.full_components, spec.channels):
        raise ValueError("unexpected equivariant RMSNorm input shape")
    if tuple(weight.shape) != (lmax + 1, spec.channels):
        raise ValueError("unexpected equivariant RMSNorm weight shape")
    if tuple(bias.shape) != (spec.channels,):
        raise ValueError("unexpected equivariant RMSNorm bias shape")
    return spec


def jvp_input(
    x, weight, bias,
    input_direction,
    *,
    lmax, epsilon,
    coefficient_major_output,
):
    spec = normalization_layout(x, weight, bias, lmax)
    output_stride = (
        (spec.channels, x.shape[0] * spec.channels, 1)
        if coefficient_major_output
        else (spec.full_components * spec.channels, spec.channels, 1)
    )
    output = torch.empty_strided(x.shape, output_stride, device=x.device, dtype=x.dtype)
    jvp_kernel[(x.shape[0],)](
        x,
        weight,
        input_direction,
        weight,
        bias,
        output,
        *x.stride(),
        *input_direction.stride(),
        *output.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_parameter_direction=False,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return output


def jvp(
    x, weight, bias,
    input_direction,
    weight_direction,
    bias_direction,
    *,
    lmax, epsilon,
    coefficient_major_output,
):
    spec = normalization_layout(x, weight, bias, lmax)
    output_stride = (
        (spec.channels, x.shape[0] * spec.channels, 1)
        if coefficient_major_output
        else (spec.full_components * spec.channels, spec.channels, 1)
    )
    output = torch.empty_strided(x.shape, output_stride, device=x.device, dtype=x.dtype)
    jvp_kernel[(x.shape[0],)](
        x,
        weight,
        input_direction,
        weight_direction,
        bias_direction,
        output,
        *x.stride(),
        *input_direction.stride(),
        *output.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_parameter_direction=True,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return output


@triton.jit
def equivariant_norm_forward_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    mean_ptr, rstd_ptr,
    eps,
    elements_per_node: tl.constexpr,
    channels: tl.constexpr,
    degrees: tl.constexpr,
    block_elements: tl.constexpr,
):
    node = tl.program_id(0)
    offset = tl.arange(0, block_elements)
    mask = offset < elements_per_node
    component = offset // channels
    channel = offset - component * channels
    value = tl.load(input_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    scalar_mask = mask & (component == 0)
    scalar_mean = tl.sum(tl.where(scalar_mask, value, 0.0), axis=0) / channels
    centered = tl.where(component == 0, value - scalar_mean, value)
    degree = component_degree(component, degrees)
    degree_width = 2 * degree + 1
    variance_weight = 1.0 / (degrees * degree_width * channels)
    variance = tl.sum(tl.where(mask, variance_weight * centered * centered, 0.0), axis=0)
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + degree * channels + channel, mask=mask, other=0.0).to(tl.float32)
    output = centered * rstd * weight
    bias = tl.load(bias_ptr + channel, mask=scalar_mask, other=0.0).to(tl.float32)
    output += tl.where(component == 0, bias, 0.0)
    tl.store(output_ptr + node * elements_per_node + offset, output, mask=mask)
    tl.store(mean_ptr + node, scalar_mean)
    tl.store(rstd_ptr + node, rstd)


def equivariant_norm_forward(inputs, weight, bias, eps):
    channels, degrees, elements_per_node, block_elements, num_warps = validate(inputs, weight, bias)
    outputs = torch.empty_like(inputs)
    mean = torch.empty(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    rstd = torch.empty_like(mean)
    equivariant_norm_forward_kernel[(inputs.shape[0],)](
        inputs, weight, bias, outputs,
        mean, rstd,
        eps,
        elements_per_node=elements_per_node,
        channels=channels, degrees=degrees,
        block_elements=block_elements, num_warps=num_warps,
    )
    return outputs, mean, rstd


@triton.jit
def forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    x_stride_node, x_stride_row, x_stride_channel,
    output_stride_node, output_stride_row, output_stride_channel,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    scalar = tl.load(
        x_ptr + node * x_stride_node + channel * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_mean = tl.sum(scalar, axis=0) / channels
    norm = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            # Centering would otherwise turn padded lanes into -mean and make
            # non-power-of-two channel counts contribute to the reduction.
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        norm += balance * tl.sum(value * value, axis=0)
    inverse = tl.rsqrt(norm + epsilon)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        result = value * inverse * weight
        if row == 0:
            result += tl.load(bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr
            + node * output_stride_node
            + row * output_stride_row
            + channel * output_stride_channel,
            result,
            mask=channel_mask,
        )


def rms_norm_forward(
    x, weight, bias,
    *,
    lmax, epsilon,
    coefficient_major_output,
):
    spec = normalization_layout(x, weight, bias, lmax)
    output_stride = (
        (spec.channels, x.shape[0] * spec.channels, 1)
        if coefficient_major_output
        else (spec.full_components * spec.channels, spec.channels, 1)
    )
    output = torch.empty_strided(x.shape, output_stride, device=x.device, dtype=x.dtype)
    forward_kernel[(x.shape[0],)](
        x, weight, bias,
        output,
        *x.stride(),
        *output.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return output


@triton.jit
def equivariant_norm_backward_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr, rstd_ptr,
    input_gradient_ptr,
    weight_gradient_ptr,
    bias_gradient_ptr,
    elements_per_node: tl.constexpr,
    channels: tl.constexpr,
    degrees: tl.constexpr,
    block_elements: tl.constexpr,
    compute_parameter_gradients: tl.constexpr,
):
    node = tl.program_id(0)
    offset = tl.arange(0, block_elements)
    mask = offset < elements_per_node
    component = offset // channels
    channel = offset - component * channels
    degree = component_degree(component, degrees)
    degree_width = 2 * degree + 1
    variance_weight = 1.0 / (degrees * degree_width * channels)
    value = tl.load(input_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + degree * channels + channel, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + node).to(tl.float32)
    rstd = tl.load(rstd_ptr + node).to(tl.float32)
    centered = tl.where(component == 0, value - mean, value)
    weighted_grad = grad_output * weight
    contraction = tl.sum(tl.where(mask, weighted_grad * centered, 0.0), axis=0)
    dx_raw = (
        rstd * weighted_grad
        - rstd * rstd * rstd * contraction * variance_weight * centered
    )
    dx_mean = tl.sum(tl.where(mask & (component == 0), dx_raw, 0.0), axis=0) / channels
    dx = tl.where(component == 0, dx_raw - dx_mean, dx_raw)
    tl.store(input_gradient_ptr + node * elements_per_node + offset, dx, mask=mask)
    if compute_parameter_gradients:
        tl.atomic_add(
            weight_gradient_ptr + degree * channels + channel,
            grad_output * centered * rstd,
            mask=mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.atomic_add(
            bias_gradient_ptr + channel,
            grad_output,
            mask=mask & (component == 0),
            sem="relaxed",
            scope="gpu",
        )


def backward(grad_output, inputs, weight, mean, rstd):
    allow_atomic_reduction("node_to_node.equivariant_norm.backward")
    channels, degrees, elements_per_node, block_elements, num_warps = validate(inputs, weight, None)
    dx = torch.empty_like(inputs)
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros(channels, device=inputs.device, dtype=inputs.dtype)
    equivariant_norm_backward_kernel[(inputs.shape[0],)](
        grad_output, inputs, weight,
        mean, rstd,
        dx, dweight, dbias,
        elements_per_node=elements_per_node,
        channels=channels, degrees=degrees,
        block_elements=block_elements,
        compute_parameter_gradients=True,
        num_warps=num_warps,
    )
    return dx, dweight, dbias


def backward_input(grad_output, inputs, weight, mean, rstd):
    channels, degrees, elements_per_node, block_elements, num_warps = validate(inputs, weight, None)
    dx = torch.empty_like(inputs)
    equivariant_norm_backward_kernel[(inputs.shape[0],)](
        grad_output, inputs, weight,
        mean, rstd,
        dx,
        weight,
        weight,
        elements_per_node=elements_per_node,
        channels=channels, degrees=degrees,
        block_elements=block_elements,
        compute_parameter_gradients=False,
        num_warps=num_warps,
    )
    return dx


@triton.jit
def vjp_kernel(
    grad_ptr,
    residual_gradient_ptr,
    x_ptr,
    weight_ptr,
    input_gradient_ptr,
    weight_gradient_ptr,
    bias_gradient_ptr,
    grad_stride_node,
    grad_stride_row,
    grad_stride_channel,
    x_stride_node, x_stride_row, x_stride_channel,
    input_gradient_stride_node,
    input_gradient_stride_row,
    input_gradient_stride_channel,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    has_residual: tl.constexpr,
    compute_parameter_gradients: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    scalar = tl.load(
        x_ptr + node * x_stride_node + channel * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_mean = tl.sum(scalar, axis=0) / channels
    norm = tl.zeros((), dtype=tl.float32)
    contraction = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        norm += balance * tl.sum(value * value, axis=0)
        contraction += tl.sum(grad * value * weight, axis=0)
    inverse = tl.rsqrt(norm + epsilon)
    inverse_cubed = inverse * inverse * inverse
    scalar_gradient_mean = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        input_gradient = (
            grad * weight * inverse - contraction * inverse_cubed * balance * value
        )
        if row == 0:
            scalar_gradient_mean = tl.sum(input_gradient, axis=0) / channels
            input_gradient -= scalar_gradient_mean
            if compute_parameter_gradients:
                tl.atomic_add(bias_gradient_ptr + channel, grad, mask=channel_mask)
        if compute_parameter_gradients:
            tl.atomic_add(
                weight_gradient_ptr + degree * channels + channel,
                grad * value * inverse,
                mask=channel_mask,
            )
        if has_residual:
            input_gradient += tl.load(
                residual_gradient_ptr
                + node * input_gradient_stride_node
                + row * input_gradient_stride_row
                + channel * input_gradient_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            input_gradient_ptr
            + node * input_gradient_stride_node
            + row * input_gradient_stride_row
            + channel * input_gradient_stride_channel,
            input_gradient,
            mask=channel_mask,
        )


if (
    torch.version.cuda
    and find_spec("triton.experimental") is not None
    and find_spec("triton.experimental.gluon") is not None
):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def gl_sum_node(values):
        return gl.sum(gl.sum(values, axis=0), axis=0)


    @gluon.jit
    def reduce_add(ptr, value, mask, unique_threads: gl.constexpr):
        # Some layouts replicate the parameter reduction across warps.
        # Each affine channel has exactly one writer per node.
        gl.inline_asm_elementwise(
            """
            {
                .reg .u32 thread;
                .reg .pred valid, unique;
                mov.u32 thread, %tid.x;
                setp.ne.u32 valid, $3, 0;
                setp.lt.u32 unique, thread, $4;
                and.pred valid, valid, unique;
                @valid red.relaxed.gpu.global.add.f32 [$1], $2;
                mov.u32 $0, 0;
            }
            """,
            constraints="=r,l,f,r,r",
            args=(ptr.to(gl.uint64), value, mask.to(gl.int32), unique_threads),
            dtype=gl.int32, is_pure=False, pack=1,
        )


    @gluon.jit
    def channel_derivatives(
        g, x, w, hx, hw, hb, dx, dg, dw, db,
        g_n, g_r, g_c, x_n, x_r, x_c, hx_n, hx_r, hx_c,
        rows: gl.constexpr, channels: gl.constexpr, count: gl.constexpr,
        epsilon: gl.constexpr, block_rows: gl.constexpr, block_channels: gl.constexpr,
        double: gl.constexpr,
        unique_threads: gl.constexpr,
    ):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
        node = gl.program_id(0)
        row = gl.arange(0, block_rows, layout=gl.SliceLayout(1, layout))[:, None]
        ch = gl.arange(0, block_channels, layout=gl.SliceLayout(0, layout))[None, :]
        mask = (row < rows) & (ch < channels)
        zero = gl.full((block_rows, block_channels), 0.0, gl.float32, layout=layout)
        zero_bias = gl.full((1, block_channels), 0.0, gl.float32, layout=layout)
        degree = gl.full((block_rows, 1), 0, gl.int32, layout=layout)
        for ell in tl.static_range(1, count):
            degree += row >= ell * ell
        balance = 1.0 / (count * channels * (2 * degree + 1))

        raw = gl.load(x + node * x_n + row * x_r + ch * x_c, mask, zero)
        scalar = gl_sum_node(gl.where(row == 0, raw, 0.0)) / channels
        value = gl.where(mask, raw - gl.where(row == 0, scalar, 0.0), 0.0)
        grad = gl.load(g + node * g_n + row * g_r + ch * g_c, mask, zero)
        weight = gl.load(w + degree * channels + ch, mask, zero)

        inv = gl.rsqrt(gl_sum_node(balance * value * value) + epsilon)
        inv_cubed = inv * inv * inv
        contraction = gl_sum_node(grad * value * weight)

        if double:
            h_raw = gl.load(hx + node * hx_n + row * hx_r + ch * hx_c, mask, zero)
            h_scalar = gl_sum_node(gl.where(row == 0, h_raw, 0.0)) / channels
            h_value = gl.where(mask, h_raw - gl.where(row == 0, h_scalar, 0.0), 0.0)
            h_weight = gl.load(hw + degree * channels + ch, mask, zero)
            h_bias = gl.load(hb + ch, ch < channels, zero_bias)
            h_inv = -inv_cubed * gl_sum_node(balance * value * h_value)
            h_contraction = gl_sum_node(grad * (h_value * weight + value * h_weight))
            h_normalized = h_value * inv + value * h_inv

            dout = h_normalized * weight + value * inv * h_weight
            dout += gl.where(row == 0, h_bias, 0.0)
            gl.store(dg + (node * rows + row) * channels + ch, dout, mask)
            dinput = grad * (h_weight * inv + weight * h_inv)
            h_contraction_scaled = h_contraction * inv_cubed + 3.0 * contraction * inv * inv * h_inv
            dinput -= h_contraction_scaled * balance * value
            dinput -= contraction * inv_cubed * balance * h_value
            dparam = grad * h_normalized
        else:
            dinput = grad * weight * inv - contraction * inv_cubed * balance * value
            dparam = grad * value * inv
            dbias = gl.sum(gl.where(row == 0, grad, 0.0), axis=0)
            cols = gl.arange(0, block_channels, layout=gl.SliceLayout(0, layout))
            reduce_add(db + cols, dbias, cols < channels, unique_threads)

        scalar_dx = gl_sum_node(gl.where((row == 0) & mask, dinput, 0.0)) / channels
        dinput -= gl.where(row == 0, scalar_dx, 0.0)
        gl.store(dx + (node * rows + row) * channels + ch, dinput, mask)

        cols = gl.arange(0, block_channels, layout=gl.SliceLayout(0, layout))
        for ell in tl.static_range(0, count):
            result = gl.sum(gl.where(degree == ell, dparam, 0.0), axis=0)
            reduce_add(dw + ell * channels + cols, result, cols < channels, unique_threads)

    def channel_node_derivatives(grad, x, weight, bias, *, lmax, epsilon, directions=None):
        nodes, rows, channels = x.shape
        dx = x.new_empty(x.shape)
        double = directions is not None
        if double:
            hx, hw, hb = directions
            dg = x.new_empty(x.shape)
        else:
            dg = hx = hw = hb = x
        reduced = x.new_zeros(weight.numel() + bias.numel())
        dw = reduced[:weight.numel()].view_as(weight)
        db = reduced[weight.numel():]
        channel_derivatives[(nodes,)](
            grad, x, weight, hx, hw, hb, dx, dg, dw, db,
            *grad.stride(), *x.stride(), *hx.stride(),
            rows, channels, lmax + 1, epsilon,
            triton.next_power_of_2(rows), channels, double, channels,
            num_warps=4,
        )
        return (dg, dx, dw, db) if double else (dx, dw, db)
else:
    channel_node_derivatives = None


def use_flat_schedule(x, lmax):
    # A node tile avoids row-by-row reloads in the measured widths and degrees.
    return (
        lmax in (2, 3, 4)
        and 256 <= x.shape[0] <= 4096
        and x.shape[2] in (64, 128)
        and not is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))
    )


@triton.jit
def sum_node(values):
    return tl.sum(tl.sum(values, axis=0), axis=0)


@triton.jit
def flat_derivatives(
    g, x, w, hx, hw, hb, dx, dg, dw, db,
    g_n, g_r, g_c, x_n, x_r, x_c, hx_n, hx_r, hx_c,
    rows: tl.constexpr, channels: tl.constexpr, count: tl.constexpr,
    epsilon: tl.constexpr, block_rows: tl.constexpr, block_channels: tl.constexpr,
    double: tl.constexpr,
):
    node = tl.program_id(0)
    row = tl.arange(0, block_rows)[:, None]
    ch = tl.arange(0, block_channels)[None, :]
    mask = (row < rows) & (ch < channels)
    degree = tl.full((block_rows, 1), 0, tl.int32)
    for ell in tl.static_range(1, count):
        degree += row >= ell * ell
    balance = 1.0 / (count * channels * (2 * degree + 1))

    raw = tl.load(x + node * x_n + row * x_r + ch * x_c, mask, 0.0)
    scalar = sum_node(tl.where(row == 0, raw, 0.0)) / channels
    value = tl.where(mask, raw - tl.where(row == 0, scalar, 0.0), 0.0)
    grad = tl.load(g + node * g_n + row * g_r + ch * g_c, mask, 0.0)
    weight = tl.load(w + degree * channels + ch, mask, 0.0)

    inv = tl.rsqrt(sum_node(balance * value * value) + epsilon)
    inv_cubed = inv * inv * inv
    contraction = sum_node(grad * value * weight)

    if double:
        h_raw = tl.load(hx + node * hx_n + row * hx_r + ch * hx_c, mask, 0.0)
        h_scalar = sum_node(tl.where(row == 0, h_raw, 0.0)) / channels
        h_value = tl.where(mask, h_raw - tl.where(row == 0, h_scalar, 0.0), 0.0)
        h_weight = tl.load(hw + degree * channels + ch, mask, 0.0)
        h_bias = tl.load(hb + ch, ch < channels, 0.0)
        h_inv = -inv_cubed * sum_node(balance * value * h_value)
        h_contraction = sum_node(grad * (h_value * weight + value * h_weight))
        h_normalized = h_value * inv + value * h_inv

        dout = h_normalized * weight + value * inv * h_weight
        dout += tl.where(row == 0, h_bias, 0.0)
        tl.store(dg + (node * rows + row) * channels + ch, dout, mask)
        dinput = grad * (h_weight * inv + weight * h_inv)
        h_contraction_scaled = h_contraction * inv_cubed + 3.0 * contraction * inv * inv * h_inv
        dinput -= h_contraction_scaled * balance * value
        dinput -= contraction * inv_cubed * balance * h_value
        dparam = grad * h_normalized
    else:
        dinput = grad * weight * inv - contraction * inv_cubed * balance * value
        dparam = grad * value * inv
        dbias = tl.sum(tl.where(row == 0, grad, 0.0), axis=0)
        tl.atomic_add(db + tl.arange(0, block_channels), dbias,
                      tl.arange(0, block_channels) < channels, sem="relaxed")

    scalar_dx = sum_node(tl.where((row == 0) & mask, dinput, 0.0)) / channels
    dinput -= tl.where(row == 0, scalar_dx, 0.0)
    tl.store(dx + (node * rows + row) * channels + ch, dinput, mask)

    cols = tl.arange(0, block_channels)
    for ell in tl.static_range(0, count):
        result = tl.sum(tl.where(degree == ell, dparam, 0.0), axis=0)
        tl.atomic_add(dw + ell * channels + cols, result, cols < channels, sem="relaxed")


def flat_node_derivatives(grad, x, weight, bias, *, lmax, epsilon, directions=None):
    if channel_node_derivatives is not None:
        return channel_node_derivatives(
            grad, x, weight, bias, lmax=lmax, epsilon=epsilon, directions=directions,
        )
    nodes, rows, channels = x.shape
    dx = x.new_empty(x.shape)
    double = directions is not None
    if double:
        hx, hw, hb = directions
        dg = x.new_empty(x.shape)
    else:
        dg = hx = hw = hb = x
    dw = torch.zeros_like(weight)
    db = torch.zeros_like(bias)
    flat_derivatives[(nodes,)](
        grad, x, weight, hx, hw, hb, dx, dg, dw, db,
        *grad.stride(), *x.stride(), *hx.stride(),
        rows, channels, lmax + 1, epsilon,
        triton.next_power_of_2(rows), triton.next_power_of_2(channels), double,
        num_warps=4,
    )
    return (dg, dx, dw, db) if double else (dx, dw, db)


def vjp(grad, x, weight, bias, *, lmax, epsilon):
    allow_atomic_reduction("node_to_node.rms_norm.vjp")
    spec = normalization_layout(x, weight, bias, lmax)
    if use_flat_schedule(x, lmax):
        return flat_node_derivatives(
            grad, x, weight, bias, lmax=lmax, epsilon=epsilon,
        )
    input_gradient = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    weight_gradient = torch.zeros_like(weight)
    bias_gradient = torch.zeros_like(bias)
    vjp_kernel[(x.shape[0],)](
        grad,
        grad,
        x,
        weight,
        input_gradient,
        weight_gradient,
        bias_gradient,
        *grad.stride(),
        *x.stride(),
        *input_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_residual=False,
        compute_parameter_gradients=True,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return input_gradient, weight_gradient, bias_gradient


def vjp_with_residual(grad, residual_gradient, x, weight, bias, *, lmax, epsilon):
    allow_atomic_reduction("node_to_node.rms_norm.vjp_with_residual")
    spec = normalization_layout(x, weight, bias, lmax)
    input_gradient = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    weight_gradient = torch.zeros_like(weight)
    bias_gradient = torch.zeros_like(bias)
    vjp_kernel[(x.shape[0],)](
        grad,
        residual_gradient,
        x,
        weight,
        input_gradient,
        weight_gradient,
        bias_gradient,
        *grad.stride(),
        *x.stride(),
        *input_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_residual=True,
        compute_parameter_gradients=True,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return input_gradient, weight_gradient, bias_gradient


def vjp_input_with_residual(
    grad,
    residual_gradient,
    x, weight, bias,
    *,
    lmax, epsilon,
):
    spec = normalization_layout(x, weight, bias, lmax)
    input_gradient = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    vjp_kernel[(x.shape[0],)](
        grad,
        residual_gradient,
        x,
        weight,
        input_gradient,
        weight,
        bias,
        *grad.stride(),
        *x.stride(),
        *input_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_residual=True,
        compute_parameter_gradients=False,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return input_gradient


@triton.jit
def equivariant_norm_double_backward_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr, rstd_ptr,
    input_direction_ptr, weight_direction_ptr, bias_direction_ptr,
    grad_output_gradient_ptr, input_gradient_ptr, weight_gradient_ptr,
    elements_per_node: tl.constexpr,
    channels: tl.constexpr,
    degrees: tl.constexpr,
    block_elements: tl.constexpr,
):
    node = tl.program_id(0)
    offset = tl.arange(0, block_elements)
    mask = offset < elements_per_node
    component = offset // channels
    channel = offset - component * channels
    degree = component_degree(component, degrees)
    degree_width = 2 * degree + 1
    variance_weight = 1.0 / (degrees * degree_width * channels)
    value = tl.load(input_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    input_direction = tl.load(input_direction_ptr + node * elements_per_node + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + degree * channels + channel, mask=mask, other=0.0).to(tl.float32)
    weight_direction = tl.load(weight_direction_ptr + degree * channels + channel, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + node).to(tl.float32)
    rstd = tl.load(rstd_ptr + node).to(tl.float32)
    scalar_direction_mean = (
        tl.sum(tl.where(mask & (component == 0), input_direction, 0.0), axis=0)
        / channels
    )
    centered = tl.where(component == 0, value - mean, value)
    centered_direction = tl.where(component == 0, input_direction - scalar_direction_mean, input_direction)
    weighted_grad = grad_output * weight
    contraction = tl.sum(tl.where(mask, weighted_grad * centered, 0.0), axis=0)
    variance_direction = tl.sum(
        tl.where(mask, variance_weight * centered * centered_direction, 0.0),
        axis=0,
    )
    mixed_contraction = tl.sum(
        tl.where(mask, grad_output * (weight * centered_direction + weight_direction * centered), 0.0),
        axis=0,
    )
    dgrad = (
        rstd * (weight * centered_direction + weight_direction * centered)
        - rstd * rstd * rstd * variance_direction * weight * centered
    )
    bias_direction = tl.load(bias_direction_ptr + channel, mask=mask & (component == 0), other=0.0).to(tl.float32)
    dgrad += tl.where(component == 0, bias_direction, 0.0)
    tl.store(grad_output_gradient_ptr + node * elements_per_node + offset, dgrad, mask=mask)

    dx_raw = (
        -rstd * rstd * rstd * variance_weight * centered * mixed_contraction
        + rstd * grad_output * weight_direction
        + 3.0 * rstd * rstd * rstd * rstd * rstd
        * variance_weight * centered * variance_direction * contraction
        - rstd * rstd * rstd * variance_weight * centered_direction * contraction
        - rstd * rstd * rstd * variance_direction * weighted_grad
    )
    dx_mean = tl.sum(tl.where(mask & (component == 0), dx_raw, 0.0), axis=0) / channels
    dx = tl.where(component == 0, dx_raw - dx_mean, dx_raw)
    tl.store(input_gradient_ptr + node * elements_per_node + offset, dx, mask=mask)
    tl.atomic_add(
        weight_gradient_ptr + degree * channels + channel,
        rstd * grad_output * centered_direction
        - rstd * rstd * rstd * variance_direction * grad_output * centered,
        mask=mask,
        sem="relaxed",
        scope="gpu",
    )


def double_backward(
    grad_output, inputs, weight,
    mean, rstd,
    input_direction, weight_direction, bias_direction,
):
    allow_atomic_reduction("node_to_node.equivariant_norm.double_backward")
    channels, degrees, elements_per_node, block_elements, num_warps = validate(inputs, weight, None)
    dgrad = torch.empty_like(grad_output)
    dx = torch.empty_like(inputs)
    dweight = torch.zeros_like(weight)
    equivariant_norm_double_backward_kernel[(inputs.shape[0],)](
        grad_output, inputs, weight,
        mean, rstd,
        input_direction, weight_direction, bias_direction,
        dgrad, dx, dweight,
        elements_per_node=elements_per_node,
        channels=channels, degrees=degrees,
        block_elements=block_elements, num_warps=num_warps,
    )
    return dgrad, dx, dweight


@triton.jit
def hvp_kernel(
    grad_ptr,
    x_ptr,
    weight_ptr,
    input_direction_ptr,
    weight_direction_ptr,
    bias_direction_ptr,
    grad_output_gradient_ptr,
    input_gradient_ptr,
    weight_gradient_ptr,
    grad_stride_node,
    grad_stride_row,
    grad_stride_channel,
    x_stride_node, x_stride_row, x_stride_channel,
    input_direction_stride_node, input_direction_stride_row, input_direction_stride_channel,
    grad_output_gradient_stride_node,
    grad_output_gradient_stride_row,
    grad_output_gradient_stride_channel,
    input_gradient_stride_node,
    input_gradient_stride_row,
    input_gradient_stride_channel,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    scalar = tl.load(
        x_ptr + node * x_stride_node + channel * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_direction = tl.load(
        input_direction_ptr
        + node * input_direction_stride_node
        + channel * input_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_mean = tl.sum(scalar, axis=0) / channels
    scalar_direction_mean = tl.sum(scalar_direction, axis=0) / channels
    norm = tl.zeros((), dtype=tl.float32)
    norm_direction_half = tl.zeros((), dtype=tl.float32)
    contraction = tl.zeros((), dtype=tl.float32)
    contraction_direction = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        weight_direction = tl.load(
            weight_direction_ptr + degree * channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        norm += balance * tl.sum(value * value, axis=0)
        norm_direction_half += balance * tl.sum(value * value_direction, axis=0)
        contraction += tl.sum(grad * value * weight, axis=0)
        contraction_direction += tl.sum(
            grad * (value_direction * weight + value * weight_direction), axis=0
        )
    inverse = tl.rsqrt(norm + epsilon)
    inverse_cubed = inverse * inverse * inverse
    inverse_direction = -inverse_cubed * norm_direction_half
    scalar_input_gradient_mean = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        weight_direction = tl.load(
            weight_direction_ptr + degree * channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        grad_output_gradient = (
            value_direction * inverse * weight
            + value * inverse_direction * weight
            + value * inverse * weight_direction
        )
        if row == 0:
            grad_output_gradient += tl.load(bias_direction_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        input_gradient = (
            grad * weight_direction * inverse
            + grad * weight * inverse_direction
            - contraction_direction * inverse_cubed * balance * value
            - contraction
            * 3.0
            * inverse
            * inverse
            * inverse_direction
            * balance
            * value
            - contraction * inverse_cubed * balance * value_direction
        )
        if row == 0:
            scalar_input_gradient_mean = tl.sum(input_gradient, axis=0) / channels
            input_gradient -= scalar_input_gradient_mean
        tl.atomic_add(
            weight_gradient_ptr + degree * channels + channel,
            grad * (value_direction * inverse + value * inverse_direction),
            mask=channel_mask,
        )
        tl.store(
            grad_output_gradient_ptr
            + node * grad_output_gradient_stride_node
            + row * grad_output_gradient_stride_row
            + channel * grad_output_gradient_stride_channel,
            grad_output_gradient,
            mask=channel_mask,
        )
        tl.store(
            input_gradient_ptr
            + node * input_gradient_stride_node
            + row * input_gradient_stride_row
            + channel * input_gradient_stride_channel,
            input_gradient,
            mask=channel_mask,
        )


@triton.jit
def hvp_with_cotangent_kernel(
    grad_ptr,
    grad_direction_ptr,
    x_ptr,
    weight_ptr,
    input_direction_ptr,
    weight_direction_ptr,
    input_gradient_ptr,
    weight_gradient_ptr,
    bias_gradient_ptr,
    raw_direction_ptr,
    grad_output_gradient_ptr,
    grad_stride_node,
    grad_stride_row,
    grad_stride_channel,
    grad_direction_stride_node,
    grad_direction_stride_row,
    grad_direction_stride_channel,
    x_stride_node, x_stride_row, x_stride_channel,
    input_direction_stride_node, input_direction_stride_row, input_direction_stride_channel,
    input_gradient_stride_node,
    input_gradient_stride_row,
    input_gradient_stride_channel,
    raw_direction_stride_row,
    raw_direction_stride_node,
    raw_direction_stride_channel,
    grad_output_gradient_stride_node,
    grad_output_gradient_stride_row,
    grad_output_gradient_stride_channel,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    has_weight_direction: tl.constexpr,
    write_grad_output: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    channel_mask = channel < channels
    scalar = tl.load(
        x_ptr + node * x_stride_node + channel * x_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_direction = tl.load(
        input_direction_ptr
        + node * input_direction_stride_node
        + channel * input_direction_stride_channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    scalar_mean = tl.sum(scalar, axis=0) / channels
    scalar_direction_mean = tl.sum(scalar_direction, axis=0) / channels
    norm = tl.zeros((), dtype=tl.float32)
    norm_direction_half = tl.zeros((), dtype=tl.float32)
    contraction = tl.zeros((), dtype=tl.float32)
    contraction_direction = tl.zeros((), dtype=tl.float32)
    cotangent_contraction = tl.zeros((), dtype=tl.float32)
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        grad_direction = tl.load(
            grad_direction_ptr
            + node * grad_direction_stride_node
            + row * grad_direction_stride_row
            + channel * grad_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        weight_direction = tl.zeros(channel.shape, dtype=tl.float32)
        if has_weight_direction:
            weight_direction = tl.load(
                weight_direction_ptr + degree * channels + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        norm += balance * tl.sum(value * value, axis=0)
        norm_direction_half += balance * tl.sum(value * value_direction, axis=0)
        contraction += tl.sum(grad * value * weight, axis=0)
        contraction_direction += tl.sum(
            grad * (value_direction * weight + value * weight_direction), axis=0
        )
        cotangent_contraction += tl.sum(grad_direction * value * weight, axis=0)
    inverse = tl.rsqrt(norm + epsilon)
    inverse_cubed = inverse * inverse * inverse
    inverse_direction = -inverse_cubed * norm_direction_half
    for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
        if rows * channels > 4096:
            degree = tl.full((), 0, tl.int32)
            for candidate in tl.static_range(1, degree_count):
                degree += row >= candidate * candidate
        else:
            degree = list(degrees)[row]
        value = tl.load(
            x_ptr
            + node * x_stride_node
            + row * x_stride_row
            + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        value_direction = tl.load(
            input_direction_ptr
            + node * input_direction_stride_node
            + row * input_direction_stride_row
            + channel * input_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if row == 0:
            value = tl.where(channel_mask, value - scalar_mean, 0.0)
            value_direction = tl.where(channel_mask, value_direction - scalar_direction_mean, 0.0)
        grad = tl.load(
            grad_ptr
            + node * grad_stride_node
            + row * grad_stride_row
            + channel * grad_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        grad_direction = tl.load(
            grad_direction_ptr
            + node * grad_direction_stride_node
            + row * grad_direction_stride_row
            + channel * grad_direction_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + degree * channels + channel, mask=channel_mask, other=0.0).to(tl.float32)
        weight_direction = tl.zeros(channel.shape, dtype=tl.float32)
        if has_weight_direction:
            weight_direction = tl.load(
                weight_direction_ptr + degree * channels + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
        balance = 1.0 / (degree_count * (2 * degree + 1) * channels)
        input_gradient = (
            grad * weight_direction * inverse
            + grad * weight * inverse_direction
            - contraction_direction * inverse_cubed * balance * value
            - contraction
            * 3.0
            * inverse
            * inverse
            * inverse_direction
            * balance
            * value
            - contraction * inverse_cubed * balance * value_direction
            + grad_direction * weight * inverse
            - cotangent_contraction * inverse_cubed * balance * value
        )
        if row == 0:
            input_gradient -= tl.sum(input_gradient, axis=0) / channels
            tl.atomic_add(bias_gradient_ptr + channel, grad_direction, mask=channel_mask)
        tl.atomic_add(
            weight_gradient_ptr + degree * channels + channel,
            grad * (value_direction * inverse + value * inverse_direction)
            + grad_direction * value * inverse,
            mask=channel_mask,
        )
        tl.store(
            input_gradient_ptr
            + node * input_gradient_stride_node
            + row * input_gradient_stride_row
            + channel * input_gradient_stride_channel,
            input_gradient,
            mask=channel_mask,
        )
        if write_grad_output:
            raw_direction = tl.load(
                raw_direction_ptr
                + row * raw_direction_stride_row
                + node * raw_direction_stride_node
                + channel * raw_direction_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            residual_direction = tl.load(
                input_direction_ptr
                + node * input_direction_stride_node
                + row * input_direction_stride_row
                + channel * input_direction_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                grad_output_gradient_ptr
                + node * grad_output_gradient_stride_node
                + row * grad_output_gradient_stride_row
                + channel * grad_output_gradient_stride_channel,
                raw_direction + residual_direction,
                mask=channel_mask,
            )


def hvp(
    grad,
    x, weight, bias,
    input_direction,
    weight_direction,
    bias_direction,
    *,
    lmax, epsilon,
):
    allow_atomic_reduction("node_to_node.rms_norm.hvp")
    spec = normalization_layout(x, weight, bias, lmax)
    if use_flat_schedule(x, lmax):
        return flat_node_derivatives(
            grad, x, weight, bias, lmax=lmax, epsilon=epsilon,
            directions=(input_direction, weight_direction, bias_direction),
        )
    grad_output_gradient = torch.empty(grad.shape, device=grad.device, dtype=grad.dtype)
    input_gradient = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    weight_gradient = torch.zeros_like(weight)
    bias_gradient = torch.zeros_like(bias)
    hvp_kernel[(x.shape[0],)](
        grad,
        x,
        weight,
        input_direction,
        weight_direction,
        bias_direction,
        grad_output_gradient,
        input_gradient,
        weight_gradient,
        *grad.stride(),
        *x.stride(),
        *input_direction.stride(),
        *grad_output_gradient.stride(),
        *input_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return grad_output_gradient, input_gradient, weight_gradient, bias_gradient


def hvp_with_cotangent_direction(
    grad,
    grad_direction,
    x, weight, bias,
    input_direction,
    weight_direction,
    *,
    lmax, epsilon,
):
    allow_atomic_reduction("node_to_node.rms_norm.hvp_with_cotangent_direction")
    spec = normalization_layout(x, weight, bias, lmax)
    input_gradient = torch.empty_like(x)
    weight_gradient = torch.zeros_like(weight)
    bias_gradient = torch.zeros_like(bias)
    hvp_with_cotangent_kernel[(x.shape[0],)](
        grad,
        grad_direction,
        x,
        weight,
        input_direction,
        weight_direction,
        input_gradient,
        weight_gradient,
        bias_gradient,
        grad,
        input_gradient,
        *grad.stride(),
        *grad_direction.stride(),
        *x.stride(),
        *input_direction.stride(),
        *input_gradient.stride(),
        *grad.stride(),
        *input_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_weight_direction=True,
        write_grad_output=False,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return input_gradient, weight_gradient, bias_gradient


def hvp_with_cotangent_input_and_output(
    grad,
    grad_direction,
    raw_direction,
    x, weight, bias,
    input_direction,
    *,
    lmax, epsilon,
):
    allow_atomic_reduction("node_to_node.rms_norm.hvp_with_cotangent_input_and_output")
    spec = normalization_layout(x, weight, bias, lmax)
    expected_raw_shape = (spec.full_components, x.shape[0], spec.channels)
    if tuple(raw_direction.shape) != expected_raw_shape:
        raise ValueError("unexpected FFN output direction shape")
    input_gradient = torch.empty_like(x)
    weight_gradient = torch.zeros_like(weight)
    bias_gradient = torch.zeros_like(bias)
    grad_output_gradient = torch.empty_like(x)
    hvp_with_cotangent_kernel[(x.shape[0],)](
        grad,
        grad_direction,
        x,
        weight,
        input_direction,
        weight,
        input_gradient,
        weight_gradient,
        bias_gradient,
        raw_direction,
        grad_output_gradient,
        *grad.stride(),
        *grad_direction.stride(),
        *x.stride(),
        *input_direction.stride(),
        *input_gradient.stride(),
        *raw_direction.stride(),
        *grad_output_gradient.stride(),
        channels=spec.channels,
        rows=spec.full_components,
        degrees=spec.lmajor_degrees,
        degree_count=spec.lmax + 1,
        epsilon=epsilon,
        has_weight_direction=False,
        write_grad_output=True,
        block_channels=1 << (spec.channels - 1).bit_length(),
        num_warps=4,
    )
    return (
        grad_output_gradient,
        input_gradient,
        weight_gradient,
        bias_gradient,
    )


@triton.jit
def load_rows(ptr, node, row, channel, valid, S: tl.constexpr):
    return tl.load(ptr + node * S[0] + row * S[1] + channel * S[2], valid, 0.)


@triton.jit
def center_scalar(value, scalar, valid, C: tl.constexpr):
    mean = tl.sum(tl.where(scalar, value, 0.)) / C
    return tl.where(valid, tl.where(scalar, value - mean, value), 0.)


@triton.jit
def third_rows(
    G, X, W, U, UW, A, V, VW, DG, DX, DU, PARTIAL,
    N: tl.constexpr, R: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
    CG: tl.constexpr, CX: tl.constexpr, CU: tl.constexpr,
    CA: tl.constexpr, CV: tl.constexpr,
    AW: tl.constexpr, AX: tl.constexpr, AVW: tl.constexpr,
    REQUIRE_DG: tl.constexpr, REQUIRE_DX: tl.constexpr,
    REQUIRE_DW: tl.constexpr, REQUIRE_DU: tl.constexpr,
    REQUIRE_DUW: tl.constexpr, REQUIRE_DUB: tl.constexpr,
    EPS: tl.constexpr, B: tl.constexpr,
):
    node = tl.program_id(0)
    offset = tl.arange(0, B)
    row, channel = offset // C, offset % C
    valid = row < R
    scalar = row == 0
    degree = tl.full((B,), 0, tl.int32)
    for d in tl.static_range(1, L + 1):
        degree += row >= d * d
    w_offset = degree * C + channel
    b = 1. / ((L + 1) * (2 * degree + 1) * C)
    x = center_scalar(load_rows(X, node, row, channel, valid, CX), scalar, valid, C)
    u = center_scalar(load_rows(U, node, row, channel, valid, CU), scalar, valid, C)
    v = tl.full((B,), 0., tl.float32)
    a = tl.full((B,), 0., tl.float32)
    tw = tl.full((B,), 0., tl.float32)
    if AX:
        v = center_scalar(load_rows(V, node, row, channel, valid, CV), scalar, valid, C)
    if AW:
        a = load_rows(A, node, row, channel, valid, CA)
    if AVW:
        tw = tl.load(VW + w_offset, valid, 0.)
    g = load_rows(G, node, row, channel, valid, CG)
    w = tl.load(W + w_offset, valid, 0.)
    uw = tl.load(UW + w_offset, valid, 0.)

    # Normalize both direction vectors with the same degree-weighted RMS.
    r = tl.rsqrt(tl.sum(b * x * x) + EPS)
    n, u, v = r * x, r * u, r * v
    alpha, beta = tl.sum(b * n * u), tl.sum(b * n * v)
    cross = 3. * alpha * beta - tl.sum(b * u * v)
    nu, nv = u - alpha * n, v - beta * n
    nuv = cross * n - beta * u - alpha * v

    if REQUIRE_DG:
        tl.store(DG + node * R * C + offset, w * nuv + tw * nu + uw * nv, valid)
    if REQUIRE_DW:
        tl.store(PARTIAL + node * R * C + offset, a * nu + g * nuv, valid)
    if REQUIRE_DUW:
        tl.store(PARTIAL + (N + node) * R * C + offset, a * n + g * nv, valid)
    if REQUIRE_DUB:
        tl.store(PARTIAL + (2 * N * R + node) * C + channel, a, scalar)
    if REQUIRE_DX or REQUIRE_DU:
        # Reverse the first and mixed directional derivatives through the row moments.
        dn, dnu, dnv, dnuv = a * uw, a * w + g * tw, g * uw, g * w
        dcross = tl.sum(dnuv * n)
        dalpha = 3. * beta * dcross - tl.sum(dnu * n + dnuv * v)
        dbeta = 3. * alpha * dcross - tl.sum(dnv * n + dnuv * u)
        d_u = dnu - beta * dnuv + b * (dalpha * n - dcross * v)
        if REQUIRE_DU:
            du = center_scalar(r * d_u, scalar, valid, C)
            tl.store(DU + node * R * C + offset, du, valid)
        if REQUIRE_DX:
            d_n = dn - alpha * dnu - beta * dnv + cross * dnuv + b * (dalpha * u + dbeta * v)
            d_v = dnv - alpha * dnuv + b * (dbeta * n - dcross * u)
            dr = tl.sum(d_n * n + d_u * u + d_v * v)
            dx = center_scalar(r * (d_n - b * n * dr), scalar, valid, C)
            tl.store(DX + node * R * C + offset, dx, valid)


@triton.jit
def third_parameters(
    partial, dw, duw, dub, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    DW: tl.constexpr, DUW: tl.constexpr, DUB: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    intervals=None, DEGREES: tl.constexpr = 1, SHARDS: tl.constexpr = 1, CHUNK_N: tl.constexpr = 0,
):
    group = tl.program_id(1)
    degree = group // SHARDS
    slot = tl.program_id(2)
    channel = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    shard = group % SHARDS
    lane = shard * CHUNK_N + tl.arange(0, BLOCK_N)
    if (slot == 0 and DW) or (slot == 1 and DUW) or (slot == 2 and DUB and degree == 0):
        stride = tl.where(slot == 2, C, R * C)
        base = partial + slot * N * R * C + degree * degree * C
        acc = tl.full((BLOCK_N, BLOCK_C), 0., tl.float32)
        for start in range(0, CHUNK_N if CHUNK_N else N, BLOCK_N):
            node = start + lane
            for component in range(2 * degree + 1):
                idx = node[:, None] * stride + component * C + channel[None, :]
                acc += tl.load(base + idx, (node[:, None] < N) & (channel[None, :] < C), 0.)
        value = tl.sum(acc, axis=0)
        if CHUNK_N:
            offset = ((slot * DEGREES + degree) * SHARDS + shard) * C + channel
            tl.store(intervals + offset, value, channel < C)
        elif slot == 0:
            tl.store(dw + degree * C + channel, value, channel < C)
        elif slot == 1:
            tl.store(duw + degree * C + channel, value, channel < C)
        else:
            tl.store(dub + channel, value, channel < C)


@triton.jit
def finish_third_parameters(
    intervals, dw, duw, dub, C: tl.constexpr, DEGREES: tl.constexpr, SHARDS: tl.constexpr,
    DW: tl.constexpr, DUW: tl.constexpr, DUB: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
):
    degree = tl.program_id(1)
    slot = tl.program_id(2)
    channel = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    if (slot == 0 and DW) or (slot == 1 and DUW) or (slot == 2 and DUB and degree == 0):
        shard = tl.arange(0, BLOCK_S)
        offset = ((slot * DEGREES + degree) * SHARDS + shard[:, None]) * C + channel[None, :]
        partial = tl.load(intervals + offset, (shard[:, None] < SHARDS) & (channel[None, :] < C), 0.)
        value = tl.sum(partial, axis=0)
        if slot == 0:
            tl.store(dw + degree * C + channel, value, channel < C)
        elif slot == 1:
            tl.store(duw + degree * C + channel, value, channel < C)
        else:
            tl.store(dub + channel, value, channel < C)


def fused_third(values, directions, needs, lmax, epsilon):
    g, x, w, bias, u, uw, ub = values
    a, v, tw, _ = directions
    need_g, need_x, need_w, _, need_u, need_uw, need_ub = needs
    has_input_curvature = v is not None
    has_parameter_curvature = tw is not None
    active = a is not None or has_input_curvature or has_parameter_curvature
    need_g = need_g and (has_input_curvature or has_parameter_curvature)
    need_x = need_x and active
    need_u = need_u and active
    need_w = need_w and (a is not None or has_input_curvature)
    need_uw = need_uw and (a is not None or has_input_curvature)
    need_ub = need_ub and a is not None
    n, r, c = x.shape
    dg = x.new_empty(x.shape) if need_g else None
    dx = x.new_empty(x.shape) if need_x else None
    du = x.new_empty(x.shape) if need_u else None
    dw = torch.empty_like(w) if need_w else None
    duw = torch.empty_like(w) if need_uw else None
    dub = torch.empty_like(bias) if need_ub else None
    param = need_w or need_uw or need_ub
    if not (need_g or need_x or need_u or param):
        return (None,) * len(values)
    partial = x.new_empty((n * (2 * r + 1), c)) if param else x
    third_rows[(n,)](
        g, x, w, u, uw, x if a is None else a, x if v is None else v,
        w if tw is None else tw.contiguous(), x if dg is None else dg,
        x if dx is None else dx, x if du is None else du, partial,
        n, r, c, lmax, g.stride(), x.stride(), u.stride(),
        x.stride() if a is None else a.stride(), x.stride() if v is None else v.stride(),
        a is not None, v is not None, tw is not None,
        need_g, need_x, need_w, need_u, need_uw, need_ub, epsilon,
        triton.next_power_of_2(r * c), num_warps=8,
    )
    if param:
        outputs = w if dw is None else dw, w if duw is None else duw, bias if dub is None else dub
        grid = triton.cdiv(c, 16), lmax + 1, 3
        requested = need_w, need_uw, need_ub
        if n > 1024:
            chunk = 128
            shards = triton.cdiv(n, chunk)
            intervals = partial.new_empty((3, lmax + 1, shards, c))
            third_parameters[(grid[0], grid[1] * shards, grid[2])](
                partial, *outputs, n, r, c, *requested, 16, 16,
                intervals=intervals, DEGREES=lmax + 1, SHARDS=shards, CHUNK_N=chunk, num_warps=4,
            )
            finish_third_parameters[grid](
                intervals, *outputs, c, lmax + 1, shards, *requested,
                triton.next_power_of_2(shards), 16, num_warps=4,
            )
        else:
            third_parameters[grid](partial, *outputs, n, r, c, *requested, 16, 16, num_warps=4)
    return dg, dx, dw, None, du, duw, dub


def degree_rms(x, weight, bias, lmax, epsilon):
    components = x.split(tuple(2 * degree + 1 for degree in range(lmax + 1)), dim=1)
    scalar = components[0] - components[0].mean(dim=-1, keepdim=True)
    components = scalar, *components[1:]
    variance = torch.stack(tuple(value.square().mean(dim=(1, 2)) for value in components)).mean(dim=0)
    scale = (variance + epsilon).rsqrt()[:, None, None]
    normalized = [value * scale * weight[degree] for degree, value in enumerate(components)]
    return torch.cat((normalized[0] + bias, *normalized[1:]), dim=1)


class RMSNormDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lmax, epsilon, *values):
        ctx.save_for_backward(*values)
        ctx.norm_config = lmax, epsilon
        ctx.set_materialize_grads(False)
        return hvp(*values, lmax=lmax, epsilon=epsilon)

    @staticmethod
    @once_differentiable
    def backward(ctx, *directions):
        lmax, epsilon = ctx.norm_config
        values = ctx.saved_tensors
        x = values[1]
        if (
            x.is_cuda and x.dtype == torch.float32 and 0 < x.shape[0] <= 8192
            and x.shape[2] <= 128 and lmax <= 6
            and torch.version.hip is None and torch.cuda.get_device_capability(x.device) == (9, 0)
            and not any(value.is_neg() for value in (*values, *directions) if value is not None)
        ):
            gradients = fused_third(values, directions, ctx.needs_input_grad[2:], lmax, epsilon)
            return None, None, *gradients
        grad, x, weight, bias, *tangents = (
            value.detach().requires_grad_() for value in values
        )
        inputs = x, weight, bias
        with torch.enable_grad():
            normalized = degree_rms(*inputs, lmax, epsilon)
            first = torch.autograd.grad(normalized, inputs, grad, create_graph=True)
            second = torch.autograd.grad(
                first, (grad, *inputs), tuple(tangents),
                create_graph=True, materialize_grads=True,
            )
            active = [(value, direction) for value, direction in zip(second, directions, strict=True)
                      if value.requires_grad and direction is not None]
            targets = grad, *inputs, *tangents
            gradients = torch.autograd.grad(
                tuple(value for value, _ in active), targets,
                tuple(direction for _, direction in active), allow_unused=True,
            ) if active else (None,) * len(targets)
        return None, None, *gradients
