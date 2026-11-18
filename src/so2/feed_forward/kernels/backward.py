from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...utils.degree_norm import vjp_input_with_residual, vjp_with_residual
from ...utils.coefficient_gate import silu_terms
from .utils import grid_latitude_modes, grid_longitude_product
from ....utils import is_compiling
from .forward import degree_expand
from .utils import activation_layout, validate


@triton.jit
def vjp_kernel(
    grad_ptr,
    hidden_ptr,
    gating_ptr,
    hidden_bias_ptr,
    gating_bias_ptr,
    hidden_gradient_ptr,
    gating_gradient_ptr,
    grad_stride_row,
    grad_stride_node,
    grad_stride_channel,
    hidden_stride_row,
    hidden_stride_node,
    hidden_stride_channel,
    gating_stride_node,
    hidden_gradient_stride_row,
    hidden_gradient_stride_node,
    hidden_gradient_stride_channel,
    gating_gradient_stride_node,
    channels: tl.constexpr,
    lmax: tl.constexpr,
    row_degrees: tl.constexpr,
    rows: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    channel = tl.arange(0, block_channels)
    mask = channel < channels

    scalar = tl.load(
        hidden_ptr + node * hidden_stride_node + channel * hidden_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scalar += tl.load(hidden_bias_ptr + channel, mask=mask, other=0.0)
    scalar_grad = tl.load(
        grad_ptr + node * grad_stride_node + channel * grad_stride_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    _, scalar_first, _ = silu_terms(scalar)
    scalar_gradient = scalar_grad * scalar_first
    tl.store(
        hidden_gradient_ptr
        + node * hidden_gradient_stride_node
        + channel * hidden_gradient_stride_channel,
        scalar_gradient,
        mask=mask,
    )

    for degree in tl.static_range(1, lmax + 1):
        gate_offset = (degree - 1) * channels + channel
        raw_gate = tl.load(gating_ptr + node * gating_stride_node + gate_offset, mask=mask, other=0.0).to(tl.float32)
        raw_gate += tl.load(gating_bias_ptr + gate_offset, mask=mask, other=0.0)
        silu_gate, silu_first, _ = silu_terms(raw_gate)
        gate = tl.sigmoid(silu_gate)
        gate_first = gate * (1.0 - gate) * silu_first
        contraction = tl.zeros(channel.shape, dtype=tl.float32)
        for row in tl.static_range(1, rows):
            if row >= degree * degree and row < (degree + 1) * (degree + 1):
                address = (
                    row * hidden_stride_row
                    + node * hidden_stride_node
                    + channel * hidden_stride_channel
                )
                grad_address = (
                    row * grad_stride_row
                    + node * grad_stride_node
                    + channel * grad_stride_channel
                )
                value = tl.load(hidden_ptr + address, mask=mask, other=0.0).to(tl.float32)
                value_grad = tl.load(grad_ptr + grad_address, mask=mask, other=0.0).to(tl.float32)
                tl.store(
                    hidden_gradient_ptr
                    + row * hidden_gradient_stride_row
                    + node * hidden_gradient_stride_node
                    + channel * hidden_gradient_stride_channel,
                    value_grad * gate,
                    mask=mask,
                )
                contraction += value_grad * value
        gate_gradient = gate_first * contraction
        tl.store(
            gating_gradient_ptr + node * gating_gradient_stride_node + gate_offset,
            gate_gradient,
            mask=mask,
        )


@triton.jit
def bias_reduce_kernel(
    hidden_gradient_ptr,
    gating_gradient_ptr,
    hidden_bias_gradient_ptr,
    gating_bias_gradient_ptr,
    hidden_gradient_stride_node,
    hidden_gradient_stride_channel,
    gating_gradient_stride_node,
    nodes: tl.constexpr,
    channels: tl.constexpr,
    gating_channels: tl.constexpr,
    block_nodes: tl.constexpr,
):
    index = tl.program_id(0)
    node = tl.arange(0, block_nodes)
    mask = node < nodes
    if index < channels:
        values = tl.load(
            hidden_gradient_ptr
            + node * hidden_gradient_stride_node
            + index * hidden_gradient_stride_channel,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(hidden_bias_gradient_ptr + index, tl.sum(values, axis=0))
    if index < gating_channels:
        values = tl.load(
            gating_gradient_ptr + node * gating_gradient_stride_node + index,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(gating_bias_gradient_ptr + index, tl.sum(values, axis=0))


def bias_gradients(hidden_gradient, gating_gradient, hidden_bias, gating_bias):
    nodes, channels = hidden_gradient.shape[1:]
    hidden_bias_gradient = torch.empty_like(hidden_bias)
    gating_bias_gradient = torch.empty_like(gating_bias)
    bias_reduce_kernel[(max(channels, gating_bias.numel()),)](
        hidden_gradient, gating_gradient,
        hidden_bias_gradient,
        gating_bias_gradient,
        hidden_gradient.stride(1),
        hidden_gradient.stride(2),
        gating_gradient.stride(0),
        nodes=nodes,
        channels=channels,
        gating_channels=gating_bias.numel(),
        block_nodes=1 << (nodes - 1).bit_length(),
        num_warps=4,
    )
    return hidden_bias_gradient, gating_bias_gradient


def activation_vjp(
    grad,
    hidden, gating, hidden_bias, gating_bias,
    *,
    lmax,
    need_bias_gradients=True,
):
    original_gating_shape = gating.shape
    gating, rows, nodes, channels, row_degrees = activation_layout(
        hidden, gating, hidden_bias, gating_bias, lmax
    )
    hidden_gradient = torch.empty_like(hidden)
    gating_gradient = torch.empty_like(gating)
    vjp_kernel[(nodes,)](
        grad,
        hidden, gating, hidden_bias, gating_bias,
        hidden_gradient, gating_gradient,
        *grad.stride(),
        *hidden.stride(),
        gating.stride(0),
        *hidden_gradient.stride(),
        gating_gradient.stride(0),
        channels=channels,
        lmax=lmax,
        row_degrees=row_degrees,
        rows=rows,
        block_channels=1 << (channels - 1).bit_length(),
        num_warps=4,
    )
    gradients = hidden_gradient, gating_gradient.view(original_gating_shape)
    if need_bias_gradients:
        return (
            *gradients,
            *bias_gradients(
                hidden_gradient, gating_gradient, hidden_bias, gating_bias
            ),
        )
    return gradients


@triton.jit
def grid_product_backward_kernel(
    grad_output_ptr, pair_ptr, gate_ptr,
    pair_gradient_ptr, gate_gradient_ptr,
    grad_output_sn, grad_output_sg, grad_output_sc,
    nodes,
    grids: tl.constexpr,
    channels: tl.constexpr,
    wide_index: tl.constexpr,
    block_channels: tl.constexpr,
):
    node = tl.program_id(0)
    if wide_index:
        node = node.to(tl.int64)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    mask = channel < channels
    gate = tl.load(gate_ptr + node * channels + channel, mask=mask)
    sigmoid_gate = tl.sigmoid(gate)
    sigmoid_prime = sigmoid_gate * (1.0 - sigmoid_gate)
    gate_gradient = tl.zeros((block_channels,), dtype=tl.float32)
    for grid in tl.range(0, grids):
        row = grid * nodes + node
        pair_base = row * 2 * channels + channel
        grad_output_offset = (
            grid * grad_output_sn + node * grad_output_sg + channel * grad_output_sc
        )
        cotangent = tl.load(grad_output_ptr + grad_output_offset, mask=mask, other=0.0).to(tl.float32)
        first = tl.load(pair_ptr + pair_base, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(pair_ptr + pair_base + channels, mask=mask, other=0.0).to(tl.float32)
        tl.store(pair_gradient_ptr + pair_base, cotangent * second * sigmoid_gate, mask=mask)
        tl.store(pair_gradient_ptr + pair_base + channels, cotangent * first * sigmoid_gate, mask=mask)
        gate_gradient += cotangent * first * second * sigmoid_prime
    tl.store(gate_gradient_ptr + node * channels + channel, gate_gradient, mask=mask)


def backward(
    grad_output: torch.Tensor,
    pair: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, grids, channels = validate(pair, gate)
    if grad_output.shape != (*pair.shape[:-1], channels):
        raise ValueError(f"unexpected grid-product gradient: {grad_output.shape}")
    pair_gradient = torch.empty_like(pair)
    gate_gradient = torch.empty_like(gate)
    wide_index = pair.numel() > torch.iinfo(torch.int32).max
    block_channels = 128
    grid_product_backward_kernel[(nodes, triton.cdiv(channels, block_channels))](
        grad_output, pair, gate,
        pair_gradient, gate_gradient,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        nodes,
        grids=grids, channels=channels,
        wide_index=wide_index,
        block_channels=block_channels,
        num_warps=4,
    )
    return pair_gradient, gate_gradient


@triton.jit
def parameter_reduce_kernel(
    first_expanded_ptr,
    second_expanded_ptr,
    hidden_gradient_ptr,
    gating_gradient_ptr,
    output_gradient_ptr,
    first_weight_ptr,
    second_weight_ptr,
    hidden_bias_ptr,
    gating_bias_ptr,
    second_bias_ptr,
    hidden_stride_node,
    hidden_stride_channel,
    gating_stride_node,
    output_stride_node,
    output_stride_channel,
    nodes,
    matrix_elements,
    weight_elements,
    weight_programs,
    reduce_weights: tl.constexpr,
    channels: tl.constexpr,
    output_channels: tl.constexpr,
    lmax: tl.constexpr,
    has_second_bias: tl.constexpr,
    block: tl.constexpr,
    block_nodes: tl.constexpr,
):
    program = tl.program_id(0)
    if reduce_weights and program < weight_programs:
        weight_offset = program * block + tl.arange(0, block)
        weight_mask = weight_offset < weight_elements
        weight_degree = weight_offset // matrix_elements
        matrix_offset = weight_offset - weight_degree * matrix_elements
        first_sum = tl.zeros(weight_offset.shape, dtype=tl.float32)
        second_sum = tl.zeros(weight_offset.shape, dtype=tl.float32)
        start = weight_degree * weight_degree
        count = 2 * weight_degree + 1
        for index in tl.static_range(0, 2 * lmax + 1):
            expanded_offset = (start + index) * matrix_elements + matrix_offset
            active = weight_mask & (index < count)
            first_sum += tl.load(first_expanded_ptr + expanded_offset, mask=active, other=0.0).to(tl.float32)
            second_sum += tl.load(second_expanded_ptr + expanded_offset, mask=active, other=0.0).to(tl.float32)
        tl.store(first_weight_ptr + weight_offset, first_sum, mask=weight_mask)
        tl.store(second_weight_ptr + weight_offset, second_sum, mask=weight_mask)
    else:
        bias_index = program - weight_programs
        node_offset = tl.arange(0, block_nodes)
        node_mask = node_offset < nodes
        if bias_index < channels:
            hidden_values = tl.load(
                hidden_gradient_ptr
                + node_offset * hidden_stride_node
                + bias_index * hidden_stride_channel,
                mask=node_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(hidden_bias_ptr + bias_index, tl.sum(hidden_values, axis=0))
        if bias_index < output_channels:
            second_value = tl.zeros((), dtype=tl.float32)
            if has_second_bias:
                second_values = tl.load(
                    output_gradient_ptr
                    + node_offset * output_stride_node
                    + bias_index * output_stride_channel,
                    mask=node_mask,
                    other=0.0,
                ).to(tl.float32)
                second_value = tl.sum(second_values, axis=0)
            tl.store(second_bias_ptr + bias_index, second_value)
        if bias_index < lmax * channels:
            gating_values = tl.load(
                gating_gradient_ptr + node_offset * gating_stride_node + bias_index,
                mask=node_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(gating_bias_ptr + bias_index, tl.sum(gating_values, axis=0))


def use_grouped_weight_gradients(x, hidden, lmax):
    # Measured for the spectral residual blocks used in eSEN and UMA.
    return (
        lmax in (2, 3)
        and 384 <= x.shape[0] <= 4096
        and x.shape[2] == hidden.shape[2] == 128
        and x.is_cuda
        and x.dtype == hidden.dtype == torch.float32
        and torch.version.hip is None
        and not is_compiling()
        and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    )


def weight_derivative(grad, features, degree_count, *, grouped, tangent=None):
    if not grouped:
        product = torch.bmm(grad.transpose(1, 2), features)
        if tangent is not None:
            tangent_grad, tangent_features = tangent
            torch.baddbmm(product, tangent_grad.transpose(1, 2), tangent_features, out=product)
        return product

    # Flatten the coefficients and nodes of each degree into one contraction.
    # The batch product of the ungrouped path materializes one matrix per coefficient.
    products = []
    for degree in range(degree_count):
        start, width = degree * degree, 2 * degree + 1
        left = grad.narrow(0, start, width).flatten(0, 1).transpose(0, 1)
        right = features.narrow(0, start, width).flatten(0, 1)
        product = left @ right
        if tangent is not None:
            tangent_grad, tangent_features = tangent
            tangent_left = tangent_grad.narrow(0, start, width).flatten(0, 1).transpose(0, 1)
            tangent_right = tangent_features.narrow(0, start, width).flatten(0, 1)
            torch.addmm(product, tangent_left, tangent_right, out=product)
        products.append(product)
    return torch.stack(products)


def parameter_reduce(
    first_expanded,
    second_expanded,
    hidden_gradient, gating_gradient,
    output_gradient=None,
    *,
    degree_count,
    weights_grouped=False,
):
    has_second_bias = output_gradient is not None
    output_pointer = hidden_gradient if output_gradient is None else output_gradient
    output_strides = (
        (0, 0)
        if output_gradient is None
        else (output_gradient.stride(0), output_gradient.stride(2))
    )
    if weights_grouped:
        first_weight, second_weight = first_expanded, second_expanded
    else:
        first_weight = torch.empty(
            (degree_count, first_expanded.shape[1], first_expanded.shape[2]),
            device=first_expanded.device,
            dtype=first_expanded.dtype,
        )
        second_weight = torch.empty(
            (degree_count, second_expanded.shape[1], second_expanded.shape[2]),
            device=second_expanded.device,
            dtype=second_expanded.dtype,
        )
    channels = hidden_gradient.shape[2]
    hidden_bias = torch.empty((channels,), device=hidden_gradient.device, dtype=hidden_gradient.dtype)
    gating_bias = torch.empty(
        ((degree_count - 1) * channels,),
        device=hidden_gradient.device,
        dtype=hidden_gradient.dtype,
    )
    output_channels = second_expanded.shape[1]
    second_bias = torch.empty(
        (output_channels,), device=hidden_gradient.device, dtype=hidden_gradient.dtype
    )
    block = 256
    weight_elements = first_weight.numel()
    weight_programs = 0 if weights_grouped else triton.cdiv(weight_elements, block)
    grid = weight_programs + max(channels, output_channels, gating_bias.numel())
    parameter_reduce_kernel[(grid,)](
        None if weights_grouped else first_expanded,
        None if weights_grouped else second_expanded,
        hidden_gradient, gating_gradient,
        output_pointer,
        None if weights_grouped else first_weight,
        None if weights_grouped else second_weight,
        hidden_bias,
        gating_bias,
        second_bias,
        hidden_gradient.stride(1),
        hidden_gradient.stride(2),
        gating_gradient.stride(0),
        *output_strides,
        hidden_gradient.shape[1],
        first_expanded.shape[1] * first_expanded.shape[2],
        weight_elements,
        weight_programs,
        reduce_weights=not weights_grouped,
        channels=channels,
        output_channels=output_channels,
        lmax=degree_count - 1,
        has_second_bias=has_second_bias,
        block=block,
        block_nodes=1 << (hidden_gradient.shape[1] - 1).bit_length(),
        num_warps=4,
    )
    return first_weight, second_weight, hidden_bias, gating_bias, second_bias


@triton.jit(do_not_specialize=["nodes"])
def input_vjp_kernel(
    grad_ptr,
    raw_gradient_ptr,
    elements, nodes,
    rows: tl.constexpr,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    channel = offsets % channels
    row = (offsets // channels) % rows
    node = offsets // (rows * channels)
    raw_offset = (row * nodes + node) * channels + channel
    value = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    tl.store(raw_gradient_ptr + raw_offset, value, mask=mask)


def input_vjp_from_shape(grad, rows):
    nodes, observed_rows, channels = grad.shape
    if observed_rows != rows:
        raise ValueError("unexpected FFN epilogue gradient shape")
    grad = grad.contiguous()
    raw_gradient = torch.empty((rows, nodes, channels), device=grad.device, dtype=grad.dtype)
    block = 256
    input_vjp_kernel[(triton.cdiv(grad.numel(), block),)](
        grad,
        raw_gradient,
        grad.numel(),
        nodes=nodes, rows=rows, channels=channels,
        block=block,
        num_warps=4,
    )
    return raw_gradient


def norm_residual_vjp(
    grad,
    x, norm_weight, norm_bias,
    coefficient_input, gating, hidden, activated,
    gating_weight, gating_bias,
    first_weight, first_bias,
    second_weight, second_bias,
    expand_index,
    *,
    lmax, epsilon,
    save_derivatives=True,
    raw_gradient=None,
):
    grouped = use_grouped_weight_gradients(x, hidden, lmax)
    rows = coefficient_input.shape[0]
    if raw_gradient is None:
        raw_gradient = input_vjp_from_shape(grad, rows)
    second_expanded = degree_expand(second_weight, expand_index)
    activated_gradient = torch.bmm(raw_gradient, second_expanded)
    del second_expanded
    second_product = weight_derivative(raw_gradient, activated, lmax + 1, grouped=grouped)
    hidden_gradient, gating_gradient = activation_vjp(
        activated_gradient,
        hidden, gating, first_bias, gating_bias,
        lmax=lmax,
        need_bias_gradients=False,
    )
    gating_weight_gradient = torch.mm(gating_gradient.transpose(0, 1), coefficient_input[0])
    first_product = weight_derivative(hidden_gradient, coefficient_input, lmax + 1, grouped=grouped)
    if not save_derivatives:
        del raw_gradient, activated_gradient, hidden, gating, activated
    (
        first_weight_gradient, second_weight_gradient,
        first_bias_gradient, gating_bias_gradient, second_bias_gradient,
    ) = parameter_reduce(
        first_product, second_product,
        hidden_gradient, gating_gradient,
        grad,
        degree_count=first_weight.shape[0],
        weights_grouped=grouped,
    )
    if not save_derivatives:
        del first_product, second_product
    first_expanded = degree_expand(first_weight, expand_index)
    coefficient_gradient = torch.bmm(hidden_gradient, first_expanded)
    del first_expanded
    torch.addmm(
        coefficient_gradient[0], gating_gradient, gating_weight,
        beta=1.0, alpha=1.0, out=coefficient_gradient[0],
    )
    normalized_gradient = coefficient_gradient.permute(1, 0, 2)
    input_gradient, norm_weight_gradient, norm_bias_gradient = vjp_with_residual(
        normalized_gradient,
        grad,
        x, norm_weight, norm_bias,
        lmax=lmax, epsilon=epsilon,
    )
    gradients = (
        input_gradient, norm_weight_gradient, norm_bias_gradient,
        gating_weight_gradient, gating_bias_gradient,
        first_weight_gradient, first_bias_gradient,
        second_weight_gradient, second_bias_gradient,
    )
    derivative_saved = None
    if save_derivatives:
        derivative_saved = (
            normalized_gradient, raw_gradient, activated_gradient,
            hidden_gradient, gating_gradient,
        )
    return gradients, derivative_saved


def vjp_input(
    grad,
    x, norm_weight, norm_bias,
    gating, hidden,
    gating_weight,
    first_weight, first_bias,
    second_weight,
    gating_bias,
    expand_index,
    *,
    lmax, epsilon,
):
    raw_gradient = input_vjp_from_shape(grad, x.shape[1])
    second_expanded = degree_expand(second_weight, expand_index)
    activated_gradient = torch.bmm(raw_gradient, second_expanded)
    hidden_gradient, gating_gradient = activation_vjp(
        activated_gradient,
        hidden, gating, first_bias, gating_bias,
        lmax=lmax,
        need_bias_gradients=False,
    )
    first_expanded = degree_expand(first_weight, expand_index)
    coefficient_gradient = torch.bmm(hidden_gradient, first_expanded)
    torch.addmm(
        coefficient_gradient[0], gating_gradient, gating_weight,
        beta=1.0, alpha=1.0, out=coefficient_gradient[0],
    )
    return vjp_input_with_residual(
        coefficient_gradient.permute(1, 0, 2),
        grad,
        x, norm_weight, norm_bias,
        lmax=lmax, epsilon=epsilon,
    )


@triton.jit
def load_coefficients(x, node, channel, order, valid, stride_row, stride_node, stride_channel):
    values = ()
    for ell in tl.static_range(5):
        row = ell * (ell + 1) + order
        address = row[:, None] * stride_row + node[None, :] * stride_node + channel[None, :] * stride_channel
        mask = (tl.abs(order) <= ell)[:, None] & valid[None, :]
        values += (tl.load(x + address, mask, 0.0),)
    return values


@triton.jit
def latitude_weights(beta_weights, beta, order):
    weights = ()
    for ell in tl.static_range(5):
        row = ell * (ell + 1) + order
        weights += (tl.load(beta_weights + beta * 25 + row, tl.abs(order) <= ell, 0.0),)
    return weights


@triton.jit
def evaluate_longitude(alpha, weights, coefficients, channels: tl.constexpr):
    modes = tl.zeros((16, channels), tl.float32)
    for ell in tl.static_range(5):
        modes = tl.fma(weights[ell][:, None], coefficients[ell], modes)
    return tl.dot(alpha, modes, input_precision="ieee")


@triton.jit
def project_longitude(alpha, grid, mode_indices):
    projected = tl.dot(tl.trans(alpha), grid, input_precision="ieee")
    return tl.gather(projected, mode_indices, axis=0)


@triton.jit
def separable_grid_backward_kernel(
    pair, gate, alpha, beta_to, beta_from, row_mode,
    grad, hpair, hgate, result, d_pair, d_gate,
    gs0, gs1, gs2, hs0, hs1, hs2, hzs0, hzs2,
    nodes, second: tl.constexpr, channels: tl.constexpr, beta_count: tl.constexpr = 14,
):
    ch = tl.program_id(0) * channels + tl.arange(0, channels)
    node, channel = ch // 512, ch % 512
    valid = node < nodes
    s = tl.sigmoid(tl.load(gate + ch, valid, 0.0))
    ds = s * (1.0 - s)
    axis = tl.arange(0, 16)
    order = axis - 4
    longitude = tl.load(alpha + axis[:, None] * 9 + axis[None, :],
                        (axis[:, None] < 14) & (axis[None, :] < 9), 0.0)
    rows = tl.arange(0, 32)
    modes = tl.load(row_mode + rows, rows < 25, 0)
    mode_indices = tl.broadcast_to(modes[:, None], (32, channels))
    x = load_coefficients(pair, node, channel, order, valid, nodes * 1024, 1024, 1)
    y = load_coefficients(pair + 512, node, channel, order, valid, nodes * 1024, 1024, 1)
    g = load_coefficients(grad, node, channel, order, valid, gs0, gs1, gs2)
    da = tl.zeros((32, channels), tl.float32)
    db = tl.zeros((32, channels), tl.float32)
    dz = tl.zeros((16, channels), tl.float32)
    if second:
        tx = load_coefficients(hpair, node, channel, order, valid, hs0, hs1, hs2)
        ty = load_coefficients(hpair + 512 * hs2, node, channel, order, valid, hs0, hs1, hs2)
        tz = tl.load(hgate + node * hzs0 + channel * hzs2, valid, 0.0)
        ts = ds * tz
        tds = ts * (1.0 - 2.0 * s)
        output = tl.zeros((32, channels), tl.float32)

    for beta in range(beta_count):
        bt = latitude_weights(beta_to, beta, order)
        bf = latitude_weights(beta_from, beta, order)
        a = evaluate_longitude(longitude, bt, x, channels)
        b = evaluate_longitude(longitude, bt, y, channels)
        q = evaluate_longitude(longitude, bf, g, channels)
        ab = a * b
        if second:
            ha = evaluate_longitude(longitude, bt, tx, channels)
            hb = evaluate_longitude(longitude, bt, ty, channels)
            hab = ha * b + a * hb
            ga = q * (hb * s[None, :] + b * ts[None, :])
            gb = q * (ha * s[None, :] + a * ts[None, :])
            dz += q * (hab * ds[None, :] + ab * tds[None, :])
            qo = project_longitude(longitude, hab * s[None, :] + ab * ts[None, :], mode_indices)
            bf_column = tl.load(beta_from + beta * 25 + rows, rows < 25, 0.0)
            output = tl.fma(bf_column[:, None], qo, output)
        else:
            ga = q * b * s[None, :]
            gb = q * a * s[None, :]
            dz += q * ab * ds[None, :]
        bt_column = tl.load(beta_to + beta * 25 + rows, rows < 25, 0.0)
        pa = project_longitude(longitude, ga, mode_indices)
        pb = project_longitude(longitude, gb, mode_indices)
        da = tl.fma(bt_column[:, None], pa, da)
        db = tl.fma(bt_column[:, None], pb, db)

    mask = (rows[:, None] < 25) & valid[None, :]
    pair_offset = (rows[:, None] * nodes + node[None, :]) * 1024 + channel[None, :]
    tl.store(d_pair + pair_offset, da, mask)
    tl.store(d_pair + pair_offset + 512, db, mask)
    tl.store(d_gate + ch, tl.sum(dz, axis=0), valid)
    if second:
        tl.store(result + rows[:, None] * (nodes * 512) + ch[None, :], output, mask)


@triton.jit
def load_grid_rows(value, node, channel, valid, rs, ns, cs):
    values = ()
    for row in tl.static_range(25):
        index = row * rs + node * ns + channel * cs
        values += (tl.load(value + index, valid, 0.),)
    return values


@triton.jit
def mix_grid_modes(left, right, sl, sr):
    mixed = ()
    for mode in tl.static_range(9):
        mixed += (left[mode] * sl + right[mode] * sr,)
    return mixed


@triton.jit
def contract_grid_modes(a, b, scale, value):
    for mode in tl.static_range(9):
        value = tl.fma(a[mode], b[mode] * scale, value)
    return value


@triton.jit
def add_grid_latitude(previous, modes, weights, scale, row_mode: tl.constexpr):
    out = ()
    for row in tl.static_range(25):
        factor = tl.load(weights + row)
        out += (tl.fma(modes[row_mode[row]] * scale, factor, previous[row]),)
    return out


@triton.jit
def grid_longitude_backward_kernel(
    pair, gate, beta_to, beta_from, gradient, hpair, hgate, output, d_pair, d_gate,
    gs0, gs1, gs2, hs0, hs1, hs2, hzs0, hzs2, nodes,
    mode_rows: tl.constexpr, row_mode: tl.constexpr, terms: tl.constexpr,
    block: tl.constexpr, second: tl.constexpr,
):
    linear = tl.program_id(0) * block + tl.arange(0, block)
    node, channel = linear // 512, linear % 512
    valid = node < nodes
    s = tl.sigmoid(tl.load(gate + linear, valid, 0.))
    ds = s * (1. - s)
    if second:
        dz_dir = tl.load(hgate + node * hzs0 + channel * hzs2, valid, 0.)
        ts = ds * dz_dir
        tds = ts * (1. - 2. * s)

    x = load_grid_rows(pair, node, channel, valid, nodes * 1024, 1024, 1)
    y = load_grid_rows(pair + 512, node, channel, valid, nodes * 1024, 1024, 1)
    g = load_grid_rows(gradient, node, channel, valid, gs0, gs1, gs2)
    da, db, do = (), (), ()
    for _ in tl.static_range(25):
        da += (tl.zeros((block,), tl.float32),)
        db += (tl.zeros((block,), tl.float32),)
        if second:
            do += (tl.zeros((block,), tl.float32),)
    dz = tl.zeros((block,), tl.float32)

    for latitude in range(7):
        bt, bf = beta_to + latitude * 25, beta_from + latitude * 25
        am = grid_latitude_modes(x, bt, mode_rows, block)
        bm = grid_latitude_modes(y, bt, mode_rows, block)
        q = grid_latitude_modes(g, bf, mode_rows, block)
        ab = grid_longitude_product(am, bm, terms, block)
        if second:
            hx = load_grid_rows(hpair, node, channel, valid, hs0, hs1, hs2)
            hy = load_grid_rows(hpair + 512 * hs2, node, channel, valid, hs0, hs1, hs2)
            ham = grid_latitude_modes(hx, bt, mode_rows, block)
            hbm = grid_latitude_modes(hy, bt, mode_rows, block)
            av = grid_longitude_product(q, mix_grid_modes(hbm, bm, s, ts), terms, block)
            bv = grid_longitude_product(q, mix_grid_modes(ham, am, s, ts), terms, block)
            hab = mix_grid_modes(grid_longitude_product(ham, bm, terms, block),
                            grid_longitude_product(am, hbm, terms, block), 1., 1.)
            spectral = mix_grid_modes(hab, ab, s, ts)
            dz = contract_grid_modes(q, mix_grid_modes(hab, ab, ds, tds), 1., dz)
            do = add_grid_latitude(do, spectral, bf, 1., row_mode)
            da = add_grid_latitude(da, av, bt, 1., row_mode)
            db = add_grid_latitude(db, bv, bt, 1., row_mode)
        else:
            av = grid_longitude_product(q, bm, terms, block)
            bv = grid_longitude_product(q, am, terms, block)
            dz = contract_grid_modes(q, ab, ds, dz)
            da = add_grid_latitude(da, av, bt, s, row_mode)
            db = add_grid_latitude(db, bv, bt, s, row_mode)

    for row in tl.static_range(25):
        index = (row * nodes + node) * 1024 + channel
        tl.store(d_pair + index, da[row], valid)
        tl.store(d_pair + index + 512, db[row], valid)
        if second:
            tl.store(output + row * nodes * 512 + linear, do[row], valid)
    tl.store(d_gate + linear, dz, valid)
