"""Apply degree-wise channel maps and gated nonlinearities to node coefficients."""

from __future__ import annotations

from functools import lru_cache, partial
from math import isqrt
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import triton_op, wrap_triton

from ..utils.tensor_contracts import require_cuda_fp32
from ..utils.degree_norm import degree_norm, hopper_degree_kernels
from .kernels.backward import (
    activation_vjp, backward, input_vjp_from_shape, norm_residual_vjp, vjp_input,
    grid_longitude_backward_kernel, separable_grid_backward_kernel,
    vjp_kernel as gate_backward_kernel,
)
from .kernels.double_backward import (
    activation_hvp,
    double_backward,
    hvp_input_direction,
    norm_residual_hvp,
)
from .kernels.forward import (
    activation_forward,
    add_scalar_row,
    angular_grid_product,
    forward_kernel as gate_forward_kernel,
    grid_longitude_forward_kernel,
    grid_product_forward,
    merge_scalar_branch,
    norm_residual_forward,
    project_after_grid,
    project_before_grid,
    rms_expand_forward,
    separable_grid_forward_kernel,
)
from .kernels.triple_backward import (
    CoefficientActivationDoubleBackward, FFNDoubleBackward, StreamedGridDoubleBackward, contracted_hvp,
    grid_product_hvp, norm_residual_hvp_reference, streamed_grid_hvp,
)
from .kernels.utils import GRID_LONGITUDE_TERMS, GRID_MODE_ROWS, GRID_ROW_MODE, compact_grid_factors


class CanonicalGridFactors(NamedTuple):
    to_grid: Tensor
    from_grid: Tensor
    values: tuple[Tensor, ...]
    versions: tuple[int, ...]

    @classmethod
    def bind(cls, to_grid, from_grid, values):
        versions = tuple(value._version for value in (to_grid, from_grid, *values))
        return cls(to_grid, from_grid, values, versions)

    def matches(self, to_grid, from_grid):
        if to_grid is not self.to_grid or from_grid is not self.from_grid:
            return False
        values = (to_grid, from_grid, *self.values)
        return (
            not any(value.requires_grad or value.is_inference() for value in values)
            and tuple(value._version for value in values) == self.versions
        )


def coefficient_ffn(
    inputs: torch.Tensor,
    scalar_weight: torch.Tensor,
    scalar_bias: torch.Tensor,
    first_weight: torch.Tensor,
    first_expand_index: torch.Tensor | None,
    first_bias: torch.Tensor,
    second_weight: torch.Tensor,
    second_expand_index: torch.Tensor | None,
    second_bias: torch.Tensor,
    *,
    lmax: int,
    input_only=False,
) -> torch.Tensor:
    """Apply two degree-wise channel maps with a gated coefficient activation.

    Parameters
    ----------
    inputs : Tensor, shape (N, (L+1)**2, Cin)
        CUDA FP32 node coefficients in complete degree-major order.
    scalar_weight, scalar_bias : Tensor
        Gate map with shapes ``(L*H, Cin)`` and ``(L*H,)``. Its input is
        coefficient zero. Each positive degree has H gate values.
    first_weight, first_bias : Tensor
        First maps ``(L+1, H, Cin)`` and scalar-only bias ``(H,)``.
    first_expand_index, second_expand_index : Tensor, shape ((L+1)**2,)
        CUDA int64 component-to-degree maps, with degree l repeated 2*l+1
        times. Both maps describe the supplied degree-major basis. With
        ``input_only=True``, use None for both to pass pre-expanded maps.
    second_weight, second_bias : Tensor
        Second maps ``(L+1, Cout, H)`` and scalar-only bias ``(Cout,)``.
    lmax : int
        Maximum degree L. All floating tensors are CUDA FP32 on one device.
        first_bias and scalar_bias must be contiguous.
    input_only : bool
        Use the frozen parameter path, which only differentiates the input.

    Returns
    -------
    Tensor, shape (N, (L+1)**2, Cout)
        Apply first_weight independently per degree. Row zero uses SiLU after
        first_bias. Positive degrees use ``sigmoid(silu(gate_logits))``.
        Apply second_weight and add second_bias to row zero only. Storage is
        coefficient-major, while the logical axes remain node/row/channel.

    Dense products remain PyTorch products. A kernel combines scalar bias,
    activation and degree gating. Derivatives through third order cover inputs,
    all floating parameters and incoming cotangents (when ``input_only=False``).
    """
    rows = (lmax + 1) ** 2
    if lmax < 0 or inputs.ndim != 3 or inputs.shape[1] != rows or inputs.shape[2] <= 0:
        raise ValueError("unexpected spectral FFN input shape")
    parameters = scalar_weight, scalar_bias, first_weight, first_bias, second_weight, second_bias
    require_cuda_fp32(inputs, *parameters, operation="spectral FFN")
    if input_only and any(weight.requires_grad for weight in parameters):
        raise ValueError("input-only FFN requires frozen parameters")
    expanded = input_only and first_expand_index is None and second_expand_index is None
    weight_rows = rows if expanded else lmax + 1
    channels = inputs.shape[2]
    if (
        first_weight.ndim != 3
        or first_weight.shape[0] != weight_rows
        or first_weight.shape[2] != channels
    ):
        raise ValueError("first spectral FFN weight must match the input degrees and channels")
    hidden = first_weight.shape[1]
    if (
        hidden <= 0
        or second_weight.ndim != 3
        or second_weight.shape[0] != weight_rows
        or second_weight.shape[1] <= 0
        or second_weight.shape[2] != hidden
    ):
        raise ValueError("second spectral FFN weight must match the input degrees and hidden width")
    if (
        scalar_weight.shape != (lmax * hidden, channels)
        or scalar_bias.shape != (lmax * hidden,)
        or first_bias.shape != (hidden,)
        or second_bias.shape != (second_weight.shape[1],)
    ):
        raise ValueError(
            "spectral FFN scalar maps and biases must match the channel widths"
        )
    if not scalar_bias.is_contiguous() or not first_bias.is_contiguous():
        raise ValueError("spectral FFN activation biases must be contiguous")
    if not expanded:
        for index in (first_expand_index, second_expand_index):
            if (
                index is None
                or index.shape != (rows,)
                or index.dtype != torch.int64
                or index.device != inputs.device
            ):
                raise ValueError(
                    "spectral FFN degree indices must be int64 (components,) on the input device"
                )

    hopper = (
        input_only and lmax == 2 and channels == second_weight.shape[1] == 128
        and inputs.shape[0] > 0 and hidden in (128, 256, 512)
        and hopper_degree_kernels(inputs.device)
    )
    if hopper:
        coefficient_inputs = inputs.transpose(0, 1).contiguous()
        gating = coefficient_inputs[0] @ scalar_weight.T
    else:
        gating = F.linear(inputs.select(1, 0), scalar_weight, bias=None).unsqueeze(1)
    if not expanded:
        first_weight = first_weight.index_select(0, first_expand_index)
    if not hopper:
        coefficient_inputs = inputs.transpose(0, 1)
    hidden = torch.bmm(coefficient_inputs, first_weight.transpose(1, 2))
    if hopper:
        hidden = spectral_gate(hidden, gating, first_bias, scalar_bias)
    else:
        hidden = fused_ffn_activation(hidden, gating, first_bias, scalar_bias, lmax=lmax)

    if not expanded:
        second_weight = second_weight.index_select(0, second_expand_index)
    outputs = torch.bmm(hidden, second_weight.transpose(1, 2))
    if hopper:
        return torch.cat((outputs[:1] + second_bias, outputs[1:])).transpose(0, 1)
    return add_scalar_row(outputs, second_bias, reuse_output=True).transpose(0, 1)


def grid_ffn(
    inputs: torch.Tensor,
    input_degree_weight: torch.Tensor,
    input_channel_weight: torch.Tensor,
    input_component_degree: torch.Tensor,
    input_scalar_bias: torch.Tensor,
    to_grid: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    from_grid: torch.Tensor,
    output_channel_weight: torch.Tensor,
    output_degree_weight: torch.Tensor,
    output_component_degree: torch.Tensor,
    output_scalar_bias: torch.Tensor,
    *,
    factors=(),
) -> torch.Tensor:
    """Apply a gated grid-factor product between degree-wise channel maps.

    Parameters
    ----------
    inputs : Tensor, shape (N, Rin, Cin)
        CUDA FP32 node coefficients, scalar coefficient first.
    input_degree_weight, input_scalar_bias : Tensor
        Input degree maps ``(Din, I, Cin)`` and scalar-only bias ``(I,)``.
    input_channel_weight : Tensor, shape (2*H, I)
        Bias-free shared channel map producing two H-channel grid factors.
    input_component_degree : Tensor, shape (Rin,)
        CUDA int64 degree index of each input coefficient.
    to_grid, from_grid : Tensor
        Angular transforms ``(G, Rin)`` and ``(Rout, G)``. Both use the same
        G grid points. Their angular basis and normalization are caller-owned.
    gate_weight, gate_bias : Tensor
        Scalar-input gate map ``(H, Cin)`` and optional bias ``(H,)``.
    output_channel_weight : Tensor, shape (J, H)
        Bias-free shared output channel map.
    output_degree_weight, output_scalar_bias : Tensor
        Output degree maps ``(Dout, Cout, J)`` and scalar-only bias ``(Cout,)``.
    output_component_degree : Tensor, shape (Rout,)
        CUDA int64 degree index of each output coefficient. All floating
        tensors are CUDA FP32 on the input device.

    Returns
    -------
    Tensor, shape (Rout, N, Cout)
        Coefficient-major output. On the grid, split the projected channels
        into a and b, then compute ``a * b * sigmoid(gate(inputs[:, 0]))``.
        Project the product back to coefficients and apply output maps.

    Channel maps are composed with degree maps and factor multiplication is
    combined with gating. Derivatives through third order cover inputs, maps,
    biases and angular transform tensors.
    """
    floats = (
        inputs,
        input_degree_weight,
        input_channel_weight,
        input_scalar_bias,
        to_grid,
        gate_weight,
        gate_bias,
        from_grid,
        output_channel_weight,
        output_degree_weight,
        output_scalar_bias,
    )
    require_cuda_fp32(
        *(value for value in floats if value is not None), operation="grid FFN"
    )
    for index in (input_component_degree, output_component_degree):
        if (
            index.ndim != 1
            or index.dtype != torch.int64
            or index.device != inputs.device
        ):
            raise ValueError(
                "grid FFN degree indices must be int64 vectors on the input device"
            )
    coefficient_pair = project_before_grid(
        inputs,
        input_degree_weight, input_channel_weight, input_component_degree,
        input_scalar_bias,
        input_node_major=True,
    )
    nodes = inputs.shape[0]
    hidden_channels = input_channel_weight.shape[0] // 2
    if use_streamed_grid(coefficient_pair, gate_weight, gate_bias, to_grid, from_grid):
        gate_logits = F.linear(inputs.select(1, 0), gate_weight, gate_bias).unsqueeze(1)
        grid_args = (factors,) if isinstance(factors, CanonicalGridFactors) else factors
        coefficients = StreamedGridProduct.apply(
            coefficient_pair, gate_logits.contiguous(), to_grid, from_grid, *grid_args,
        )
        del coefficient_pair, gate_logits
    else:
        grid_pair = torch.matmul(to_grid, coefficient_pair.flatten(1)).view(
            to_grid.shape[0], nodes, 2 * hidden_channels
        )
        del coefficient_pair

        gate_logits = F.linear(inputs.select(1, 0), gate_weight, gate_bias).unsqueeze(1)
        product = fused_grid_product_grid_major(
            grid_pair.contiguous(), gate_logits.contiguous()
        )
        del grid_pair, gate_logits

        coefficients = torch.matmul(from_grid, product.flatten(1)).view(
            from_grid.shape[0], nodes, hidden_channels
        )
        del product
    outputs = project_after_grid(
        coefficients,
        output_channel_weight, output_degree_weight, output_component_degree,
        output_scalar_bias,
        input_node_major=False,
    )
    del coefficients
    return outputs


@lru_cache(maxsize=64)
def degree_index(rows, device):
    degree_count = isqrt(rows)
    if degree_count * degree_count != rows or rows == 0:
        raise ValueError("node coefficients must contain complete degrees")
    return torch.tensor([isqrt(row) for row in range(rows)], dtype=torch.int64, device=device)


def grid_feed_forward(
    x, input_weight, output_weight, gate_weight, index, *,
    input_bias, output_bias, gate_bias, to_grid, from_grid,
    grid_input_weight, grid_output_weight, scalar_branch, factors=(),
):
    output = grid_ffn(
        x, input_weight, grid_input_weight, index, input_bias,
        to_grid, gate_weight, gate_bias, from_grid,
        grid_output_weight, output_weight, index, output_bias, factors=factors,
    )
    if scalar_branch is None:
        return output.permute(1, 0, 2)
    first, second = F.linear(x[:, 0], *scalar_branch).chunk(2, dim=-1)
    return merge_scalar_branch(output, (F.silu(first) * second).unsqueeze(1), output_weight, reuse_output=True)


def feed_forward(
    x, input_weight, output_weight, gate_weight, *,
    input_bias, output_bias, gate_bias,
    to_grid=None, from_grid=None, grid_input_weight=None, grid_output_weight=None,
    norm_weight=None, norm_bias=None, norm_eps=1e-5,
    residual=False, scalar_branch=None, input_only=False,
):
    """Apply a gated coefficient or grid-factor equivariant FFN.

    ``x`` is (N,d,Cin), with complete degree-major coefficients. Input and
    output weights are (L+1,Hin,Cin) and (L+1,Cout,Hout). Biases affect only
    coefficient zero. In coefficient mode Hin=Hout=H and the scalar-input
    gate map is (L*H,Cin), using sigmoid(SiLU(logits)). For grid mode, provide
    ``to_grid[G,d]``, ``from_grid[d,G]``, ``grid_input_weight[2*H,Hin]``, and
    ``grid_output_weight[Hout,H]`` together. The gate map is (H,Cin) and uses
    sigmoid(logits) on grid-factor products.

    Provide ``norm_weight[L+1,Cin]`` and ``norm_bias[Cin]`` together to apply
    degree normalization with ``norm_eps`` before the first map.
    ``residual=True`` adds x to the result and requires Cout=Cin.

    A grid FFN may supply ``scalar_branch=(weight,bias)`` with a (2*Hout,Cin)
    map. Its SwiGLU output joins coefficient zero before the final degree
    map, matching the separate scalar branch in EquiformerV3. Inputs and all
    caller-owned parameters support derivatives through third order.

    ``input_only`` selects the first-order coefficient path where available.
    Its parameters must be frozen. Input and output maps can also be supplied
    with one weight per component, repeating each degree map in degree order.
    """
    if x.ndim != 3:
        raise ValueError("node coefficients must have shape (N,d,C)")
    grid = to_grid is not None
    if any((value is not None) != grid for value in (from_grid, grid_input_weight, grid_output_weight)):
        raise ValueError("provide all four grid parameters together")
    normalized = norm_weight is not None
    if (norm_bias is not None) != normalized:
        raise ValueError("provide norm_weight and norm_bias together")
    if input_only and grid:
        raise ValueError("input-only FFN requires coefficient mode")
    rows = x.shape[1]
    expanded = input_only and input_weight.shape[0] == output_weight.shape[0] == rows
    if expanded:
        if input_weight.ndim != 3 or input_weight.shape[1] <= 0 or gate_weight.ndim != 2:
            raise ValueError("expanded FFN weights must define a positive hidden width and a scalar gate map")
        lmax = gate_weight.shape[0] // input_weight.shape[1]
    else:
        lmax = isqrt(rows) - 1
    if lmax < 0 or (lmax + 1) ** 2 != rows:
        raise ValueError("node coefficients must contain complete degrees")
    index = None if expanded else degree_index(rows, x.device)
    weight_rows = rows if expanded else lmax + 1
    if input_weight.shape[0] != weight_rows or output_weight.shape[0] != weight_rows:
        raise ValueError("FFN weight degrees must match the node coefficients")
    if residual and output_weight.shape[1] != x.shape[2]:
        raise ValueError("residual FFN input and output channels must match")
    if not grid and scalar_branch is not None:
        raise ValueError("the independent scalar branch belongs to the grid FFN")
    if not grid and normalized and residual and not input_only:
        return norm_residual_ffn(
            x, norm_weight, norm_bias, gate_weight, gate_bias,
            input_weight, input_bias, output_weight, output_bias, index,
            lmax=lmax, epsilon=norm_eps,
        )
    original = x
    if normalized:
        x = degree_norm(
            x, norm_weight, norm_bias, lmax=lmax, epsilon=norm_eps, input_only=input_only,
            coefficient_major_output=input_only,
        )
    if not grid:
        result = coefficient_ffn(
            x, gate_weight, gate_bias, input_weight, index, input_bias,
            output_weight, index, output_bias, lmax=lmax, input_only=input_only,
        )
    else:
        result = grid_feed_forward(
            x, input_weight, output_weight, gate_weight, index,
            input_bias=input_bias, output_bias=output_bias, gate_bias=gate_bias,
            to_grid=to_grid, from_grid=from_grid,
            grid_input_weight=grid_input_weight, grid_output_weight=grid_output_weight,
            scalar_branch=scalar_branch,
        )
    return original + result if residual else result


class FFNActivationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, hidden, gating, hidden_bias, gating_bias, lmax):
        values = activation_vjp(grad, hidden, gating, hidden_bias, gating_bias, lmax=lmax)
        ctx.save_for_backward(grad, hidden, gating, hidden_bias, gating_bias)
        ctx.lmax = lmax
        return values

    @staticmethod
    def backward(
        ctx,
        hidden_direction, gating_direction,
        hidden_bias_direction,
        gating_bias_direction,
    ):
        grad, hidden, gating, hidden_bias, gating_bias = ctx.saved_tensors
        if hidden_direction is None:
            hidden_direction = torch.zeros_like(hidden)
        if gating_direction is None:
            gating_direction = torch.zeros_like(gating)
        if hidden_bias_direction is None:
            hidden_bias_direction = torch.zeros_like(hidden_bias)
        if gating_bias_direction is None:
            gating_bias_direction = torch.zeros_like(gating_bias)
        gating_direction = gating_direction.contiguous()
        hidden_bias_direction = hidden_bias_direction.contiguous()
        gating_bias_direction = gating_bias_direction.contiguous()
        args = (
            grad,
            hidden, gating, hidden_bias, gating_bias,
            hidden_direction, gating_direction,
            hidden_bias_direction,
            gating_bias_direction,
        )
        if torch.is_grad_enabled():
            values = CoefficientActivationDoubleBackward.apply(ctx.lmax, *args)
        else:
            values = activation_hvp(*args, lmax=ctx.lmax)
        return *values, None


class FusedFFNActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, gating, hidden_bias, gating_bias, lmax):
        output = activation_forward(hidden, gating, hidden_bias, gating_bias, lmax=lmax)
        ctx.save_for_backward(hidden, gating, hidden_bias, gating_bias)
        ctx.lmax = lmax
        return output

    @staticmethod
    def backward(ctx, grad):
        hidden, gating, hidden_bias, gating_bias = ctx.saved_tensors
        if not torch.is_grad_enabled():
            values = activation_vjp(
                grad.contiguous(),
                hidden, gating, hidden_bias, gating_bias,
                lmax=ctx.lmax,
            )
            return values[0], values[1], values[2], values[3], None
        values = FFNActivationBackward.apply(
            grad, hidden, gating, hidden_bias, gating_bias, ctx.lmax
        )
        return values[0], values[1], values[2], values[3], None


def fused_ffn_activation(hidden, gating, hidden_bias, gating_bias, *, lmax):
    return FusedFFNActivation.apply(hidden, gating, hidden_bias, gating_bias, lmax)


def use_streamed_grid(pair, gate_weight, gate_bias, to_grid, from_grid):
    if torch.compiler.is_compiling():
        return False
    rows, nodes, width = pair.shape
    return (
        rows == 25 and width == 1024 and 256 <= nodes <= 1024
        and gate_weight.ndim == 2 and gate_weight.shape[0] == 512
        and (gate_bias is None or gate_bias.shape == (512,))
        and to_grid.shape == (196, 25) and from_grid.shape == (25, 196)
        and pair.is_cuda and pair.dtype == torch.float32 and pair.is_contiguous()
        and to_grid.stride() == (25, 1) and from_grid.stride() in ((1, 25), (196, 1))
        and not to_grid.requires_grad and not from_grid.requires_grad
        and torch.version.hip is None
        and not torch._C._are_functorch_transforms_active()
        and torch.autograd.forward_ad._current_level < 0
        and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability(pair.device) in ((8, 0), (9, 0))
    )


def use_grid_longitude(pair, gate, to_grid, from_grid, factors):
    if (
        not pair.is_cuda
        or pair.ndim != 3 or pair.shape[0] != 25 or pair.shape[2] != 1024
        or not 512 <= pair.shape[1] <= 1024
        or gate.shape != (pair.shape[1], 1, 512)
        or to_grid.shape != (196, 25) or from_grid.shape != (25, 196)
        or tuple(value.shape for value in factors) != ((14, 9), (7, 25), (7, 25), (25,))
        or torch.compiler.is_compiling()
        or torch.get_float32_matmul_precision() != "highest"
        or torch.is_autocast_enabled("cuda") or torch.are_deterministic_algorithms_enabled()
        or not hopper_degree_kernels(pair.device)
    ):
        return False
    sources = (pair, gate, to_grid, from_grid, *factors)
    return (
        all(value.device == pair.device and not value.is_neg() for value in sources)
        and all(value.dtype == torch.float32 for value in sources[:-1])
        and factors[-1].dtype == torch.int32
        and all(value.is_contiguous() for value in (pair, gate, *factors))
    )


def streamed_grid_product(pair, gate, to_grid, from_grid, *factors,
                          grad=None, directions=None, longitude=False):
    if not factors:
        return angular_grid_product(pair, gate, to_grid, from_grid, grad=grad, directions=directions)

    rows, nodes, width = pair.shape
    channels = width // 2
    if longitude:
        gradient_inputs = (grad,) if directions is None else (grad, *directions)
        longitude = not any(value is not None and value.is_neg() for value in gradient_inputs)
    grid = (nodes * (channels // 32),)
    _, beta_to, beta_from, _ = factors
    beta_count = beta_to.shape[0]
    if grad is None:
        output = pair.new_empty((rows, nodes, channels))
        if longitude:
            grid_longitude_forward_kernel[(nodes * (channels // 128),)](
                pair, gate, beta_to, beta_from, output, nodes,
                GRID_MODE_ROWS, GRID_ROW_MODE, GRID_LONGITUDE_TERMS, 128, num_warps=4,
            )
        else:
            separable_grid_forward_kernel[grid](
                pair, gate, *factors, output, nodes, 32,
                beta_count=beta_count, num_warps=4, num_stages=1,
            )
        return (output,)

    second = directions is not None
    d_pair, d_gate = torch.empty_like(pair), torch.empty_like(gate)
    output = pair.new_empty((rows, nodes, channels)) if second else None
    hpair, hgate = directions if second else (None, None)
    strides = (
        *grad.stride(), *(hpair.stride() if second else (0, 0, 0)),
        *((hgate.stride(0), hgate.stride(2)) if second else (0, 0)),
    )
    if longitude:
        grid_longitude_backward_kernel[(nodes * (channels // 128),)](
            pair, gate, beta_to, beta_from, grad, hpair, hgate, output, d_pair, d_gate,
            *strides, nodes, GRID_MODE_ROWS, GRID_ROW_MODE, GRID_LONGITUDE_TERMS, 128, second,
            num_warps=4,
        )
    else:
        separable_grid_backward_kernel[grid](
            pair, gate, *factors, grad, hpair, hgate, output, d_pair, d_gate,
            *strides, nodes, second, 32, beta_count=beta_count, num_warps=4, num_stages=1,
        )
    return (output, d_pair, d_gate) if second else (d_pair, d_gate)


class StreamedGridBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, pair, gate, to_grid, from_grid, longitude, *factors):
        ctx.save_for_backward(grad, pair, gate, to_grid, from_grid, *factors)
        ctx.longitude = longitude
        return streamed_grid_product(pair, gate, to_grid, from_grid, *factors, grad=grad, longitude=longitude)

    @staticmethod
    def backward(ctx, pair_direction, gate_direction):
        grad, pair, gate, to_grid, from_grid, *factors = ctx.saved_tensors
        args = grad, pair, gate, pair_direction, gate_direction
        longitude = ctx.longitude and not torch.is_grad_enabled()
        kernel = partial(streamed_grid_double_backward, factors=factors, longitude=longitude)
        if torch.is_grad_enabled():
            rows, nodes, width = pair.shape
            double_backward_op = FFNDoubleBackward
            if (
                rows == 25 and 256 <= nodes <= 1024 and width == 1024
                and to_grid.shape[0] == 196 and pair.dtype == torch.float32 and pair.is_cuda
                and hopper_degree_kernels(pair.device)
            ):
                double_backward_op = StreamedGridDoubleBackward
                if factors:
                    alpha, beta_to, beta_from, row_mode = factors
                    longitude = alpha.index_select(1, row_mode)
                    to_grid = (beta_to[:, None, :] * longitude).flatten(0, 1)
                    from_grid = (beta_from[:, None, :] * longitude).flatten(0, 1).T
            values = double_backward_op.apply(
                kernel, streamed_grid_hvp, (to_grid, from_grid), *args,
            )
        else:
            values = kernel(to_grid, from_grid, *args)
        return *values, None, None, None, *(None,) * len(factors)


def streamed_grid_double_backward(
    to_grid, from_grid, grad, pair, gate, pair_direction, gate_direction, *, factors=(), longitude=False,
):
    return streamed_grid_product(
        pair, gate, to_grid, from_grid, *factors, grad=grad,
        directions=(pair_direction, gate_direction), longitude=longitude,
    )


class StreamedGridProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, gate, to_grid, from_grid, *factors):
        ctx.factor_args = len(factors)
        ctx.longitude = False
        if len(factors) == 1 and isinstance(factors[0], CanonicalGridFactors):
            plan, = factors
            factors = plan.values if plan.matches(to_grid, from_grid) else ()
            ctx.longitude = use_grid_longitude(pair, gate, to_grid, from_grid, factors)
        ctx.save_for_backward(pair, gate, to_grid, from_grid, *factors)
        return streamed_grid_product(pair, gate, to_grid, from_grid, *factors, longitude=ctx.longitude)[0]

    @staticmethod
    def backward(ctx, grad):
        pair, gate, to_grid, from_grid, *factors = ctx.saved_tensors
        values = StreamedGridBackward.apply(grad, pair, gate, to_grid, from_grid, ctx.longitude, *factors)
        return *values, *(None,) * (2 + ctx.factor_args)


class GridProductBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, pair, gate):
        pair_gradient, gate_gradient = backward(grad_output, pair, gate)
        ctx.save_for_backward(grad_output, pair, gate)
        return pair_gradient, gate_gradient

    @staticmethod
    def backward(ctx, pair_direction, gate_direction):
        grad_output, pair, gate = ctx.saved_tensors
        args = grad_output, pair, gate, pair_direction, gate_direction
        if torch.is_grad_enabled():
            return FFNDoubleBackward.apply(double_backward, grid_product_hvp, (), *args)
        return double_backward(*args)


class GridProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, gate):
        output = grid_product_forward(pair, gate)
        ctx.save_for_backward(pair, gate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pair, gate = ctx.saved_tensors
        return GridProductBackward.apply(grad_output, pair, gate)


def fused_grid_product_grid_major(pair: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Grid-major form with pair/output layouts ``[G, N, 2C/C]``."""

    return GridProduct.apply(pair, gate)


def norm_residual_hvp_for_third(expand_index, grad, *inputs, lmax, epsilon):
    values, directions = inputs[:9], inputs[9:]
    x, norm_weight, norm_bias, gating_weight, gating_bias, first, first_bias, second, second_bias = values
    if (
        x.is_cuda and x.is_contiguous() and 0 < x.shape[0] <= 8192 and x.shape[2] <= 128
        and lmax <= 6 and not torch.compiler.is_compiling()
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.are_deterministic_algorithms_enabled()
        and all(value.dtype == torch.float32 and not value.is_neg() for value in (grad, *inputs))
        and hopper_degree_kernels(x.device)
    ):
        normalized = degree_norm(
            x, norm_weight, norm_bias, lmax=lmax, epsilon=epsilon, coefficient_major_output=True,
        )
        output = coefficient_ffn(
            normalized, gating_weight, gating_bias, first, expand_index, first_bias,
            second, expand_index, second_bias, lmax=lmax,
        )
        return contracted_hvp((output + x).transpose(0, 1), grad, values, directions)
    return norm_residual_hvp_reference(expand_index, grad, *inputs, lmax=lmax, epsilon=epsilon)


class EquivariantFFNBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_gradient,
        grad,
        x, norm_weight, norm_bias,
        coefficient_input, gating, hidden, activated,
        gating_weight, gating_bias,
        first_weight, first_bias,
        second_weight, second_bias,
        expand_index,
        lmax, epsilon,
    ):
        gradients, derivative_saved = norm_residual_vjp(
            grad,
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
            lmax=lmax, epsilon=epsilon,
            raw_gradient=raw_gradient,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            *derivative_saved,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
        )
        ctx.config = (lmax, epsilon)
        return gradients

    @staticmethod
    def backward(ctx, *directions):
        directions = tuple(
            None if direction is None else direction.contiguous()
            for direction in directions
        )
        (
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            normalized_gradient, raw_gradient, activated_gradient,
            hidden_gradient, gating_gradient,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
        ) = ctx.saved_tensors
        references = (
            x, norm_weight, norm_bias,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
        )
        lmax, epsilon = ctx.config
        input_direction_only = directions[0] is not None and all(
            direction is None for direction in directions[1:]
        )
        arguments = (
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            normalized_gradient, raw_gradient, activated_gradient,
            hidden_gradient, gating_gradient,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
        )
        if not input_direction_only or torch.is_grad_enabled():
            full_directions = tuple(
                torch.zeros_like(reference) if direction is None else direction
                for direction, reference in zip(directions, references, strict=True)
            )
        with torch.no_grad():
            if input_direction_only:
                values = hvp_input_direction(*arguments, directions[0], lmax=lmax, epsilon=epsilon)
            else:
                values = norm_residual_hvp(*arguments, *full_directions, lmax=lmax, epsilon=epsilon)
            values = (values[0].transpose(0, 1), *values[1:])
        if torch.is_grad_enabled():
            values = FFNDoubleBackward.apply(
                lambda *_: values,
                partial(norm_residual_hvp_for_third, lmax=lmax, epsilon=epsilon),
                (expand_index,),
                raw_gradient, *references, *full_directions,
            )
        return (
            values[0],
            None,
            values[1],
            values[2],
            values[3],
            None,
            None,
            None,
            None,
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            None,
            None,
            None,
        )


class ReorderCotangent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, rows):
        return input_vjp_from_shape(grad, rows)

    @staticmethod
    def backward(ctx, grad):
        return grad.transpose(0, 1), None


class FusedEquivariantFFN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, norm_weight, norm_bias,
        gating_weight, gating_bias,
        first_weight, first_bias,
        second_weight, second_bias,
        expand_index,
        lmax, epsilon,
    ):
        output, saved = norm_residual_forward(
            x, norm_weight, norm_bias,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
            lmax=lmax, epsilon=epsilon,
        )
        if ctx.needs_input_grad[0] and not any(ctx.needs_input_grad[1:9]):
            gating, hidden = saved[1:3]
            saved = None, gating, hidden, None
        ctx.save_for_backward(
            x, norm_weight, norm_bias,
            *saved,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
        )
        ctx.config = (lmax, epsilon)
        return output

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        (
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
        ) = ctx.saved_tensors
        lmax, epsilon = ctx.config
        if coefficient_input is None and not torch.is_grad_enabled():
            input_gradient = vjp_input(
                grad,
                x, norm_weight, norm_bias,
                gating, hidden,
                gating_weight,
                first_weight, first_bias,
                second_weight,
                gating_bias,
                expand_index,
                lmax=lmax, epsilon=epsilon,
            )
            return (input_gradient,) + (None,) * 11
        if coefficient_input is None:
            # The higher-order path also needs the normalized and activated features.
            with torch.no_grad():
                normalized = rms_expand_forward(
                    x, norm_weight, norm_bias, first_weight, second_weight,
                    expand_index, lmax=lmax, epsilon=epsilon,
                )[0]
                coefficient_input = normalized.permute(1, 0, 2)
                activated = activation_forward(hidden, gating, first_bias, gating_bias, lmax=lmax)
        arguments = (
            x, norm_weight, norm_bias,
            coefficient_input, gating, hidden, activated,
            gating_weight, gating_bias,
            first_weight, first_bias,
            second_weight, second_bias,
            expand_index,
            lmax, epsilon,
        )
        if torch.is_grad_enabled():
            # Keep the existing coefficient-major buffer connected to the
            # incoming cotangent without saving an additional full tensor.
            raw_gradient = ReorderCotangent.apply(grad, coefficient_input.shape[0])
            gradients = EquivariantFFNBackward.apply(raw_gradient, grad.detach(), *arguments)
        else:
            gradients, _ = norm_residual_vjp(
                grad,
                x, norm_weight, norm_bias,
                coefficient_input, gating, hidden, activated,
                gating_weight, gating_bias,
                first_weight, first_bias,
                second_weight, second_bias,
                expand_index,
                lmax=lmax, epsilon=epsilon,
                save_derivatives=False,
            )
        return (*gradients, None, None, None)


def norm_residual_ffn(
    x, norm_weight, norm_bias,
    gating_weight, gating_bias,
    first_weight, first_bias,
    second_weight, second_bias,
    expand_index,
    *,
    lmax, epsilon,
):
    """Return x + coefficient_ffn(fused_rms_norm(x)) with caller-owned weights.

    Parameters
    ----------
    x : Tensor, shape (N, (L+1)**2, C)
        Contiguous CUDA FP32 coefficients in complete degree-major order.
    norm_weight, norm_bias : Tensor
        Degree-balanced normalization scale ``(L+1, C)`` and scalar bias ``(C,)``.
    gating_weight, gating_bias : Tensor
        Scalar-input gate map ``(L*H, C)`` and bias ``(L*H,)``.
    first_weight, first_bias : Tensor
        First degree maps ``(L+1, H, C)`` and scalar-only bias ``(H,)``.
    second_weight, second_bias : Tensor
        Second degree maps ``(L+1, C, H)`` and scalar-only bias ``(C,)``.
    expand_index : Tensor, shape ((L+1)**2,)
        Contiguous CUDA int64 degree l repeated 2*l+1 times.
    lmax : int
        Maximum degree L.
    epsilon : float
        Positive normalization epsilon. All parameters must be contiguous
        CUDA FP32 on the same device as x.

    Returns
    -------
    Tensor, shape (N, (L+1)**2, C)
        Normalize x, apply the gated coefficient FFN, and add the residual x.
        The FFN scalar uses SiLU. Positive degrees use sigmoid-of-SiLU gates.

    Kernels combine normalization with feature preparation, and activation
    with surrounding writes. Dense products use cuBLAS. Derivatives through
    third order cover x, all floating parameters and incoming cotangents.
    """
    if lmax < 0 or x.ndim != 3 or x.shape[1] != (lmax + 1) ** 2 or x.shape[2] <= 0:
        raise ValueError(
            "residual FFN input must have shape (N, (L+1)**2, C) with C > 0"
        )
    require_cuda_fp32(
        x, norm_weight, norm_bias,
        gating_weight, gating_bias,
        first_weight, first_bias,
        second_weight, second_bias,
        operation="residual FFN",
        contiguous=True,
    )
    channels = x.shape[2]
    if (
        first_weight.ndim != 3
        or first_weight.shape[0] != lmax + 1
        or first_weight.shape[2] != channels
    ):
        raise ValueError("first residual FFN weight must have shape (L+1, H, C)")
    hidden = first_weight.shape[1]
    expected = (
        (norm_weight, (lmax + 1, channels)),
        (norm_bias, (channels,)),
        (gating_weight, (lmax * hidden, channels)),
        (gating_bias, (lmax * hidden,)),
        (first_bias, (hidden,)),
        (second_weight, (lmax + 1, channels, hidden)),
        (second_bias, (channels,)),
    )
    if hidden <= 0 or any(value.shape != shape for value, shape in expected):
        raise ValueError(
            "residual FFN parameters must match the input and hidden channel widths"
        )
    if (
        expand_index.shape != (x.shape[1],)
        or expand_index.dtype != torch.int64
        or expand_index.device != x.device
        or not expand_index.is_contiguous()
    ):
        raise ValueError(
            "residual FFN degree indices must be contiguous int64 on the input device"
        )
    return FusedEquivariantFFN.apply(
        x, norm_weight, norm_bias,
        gating_weight, gating_bias,
        first_weight, first_bias,
        second_weight, second_bias,
        expand_index,
        lmax, epsilon,
    )


SPECTRAL_FFN_DEGREES = (0, 1, 1, 1, 2, 2, 2, 2, 2)


@triton_op("trim_h100::uma_ffn_activation", mutates_args=())
def spectral_gate(hidden: Tensor, gating: Tensor, hidden_bias: Tensor, gate_bias: Tensor) -> Tensor:
    out = torch.empty_like(hidden)
    nodes = hidden.shape[1]
    wrap_triton(gate_forward_kernel)[(nodes,)](
        hidden, gating, hidden_bias, gate_bias, out, nodes,
        *hidden.stride(), gating.stride(0), *out.stride(),
        channels=hidden.shape[2], lmax=2, row_degrees=SPECTRAL_FFN_DEGREES, rows=9,
        block_channels=hidden.shape[2], num_warps=4,
    )
    return out


@triton_op("trim_h100::uma_ffn_activation_vjp", mutates_args=())
def spectral_gate_vjp(
    grad: Tensor, hidden: Tensor, gating: Tensor, hidden_bias: Tensor, gate_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    grad = grad.resolve_neg()
    dh = torch.empty_like(hidden)
    dg = torch.empty_like(gating)
    wrap_triton(gate_backward_kernel)[(hidden.shape[1],)](
        grad, hidden, gating, hidden_bias, gate_bias, dh, dg,
        *grad.stride(), *hidden.stride(), gating.stride(0), *dh.stride(), dg.stride(0),
        channels=hidden.shape[2], lmax=2, row_degrees=SPECTRAL_FFN_DEGREES, rows=9,
        block_channels=hidden.shape[2], num_warps=4,
    )
    return dh, dg


def save_spectral_gate(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def backward_spectral_gate(ctx, grad):
    dh, dg = spectral_gate_vjp(grad, *ctx.saved_tensors)
    return dh, dg, None, None


spectral_gate.register_autograd(backward_spectral_gate, setup_context=save_spectral_gate)
