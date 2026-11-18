from __future__ import annotations

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from ...utils.coefficient_gate import silu_terms
from .utils import grid_latitude_modes, grid_longitude_product
from .utils import activation_layout, validate, validate_epilogue


class AddScalarRow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, scalar):
        ctx.scalar_shape = scalar.shape
        ctx.mark_dirty(output)
        output[:1].add_(scalar)
        return output

    @staticmethod
    def backward(ctx, grad):
        scalar_grad = grad[:1].sum_to_size(ctx.scalar_shape) if ctx.needs_input_grad[1] else None
        return grad, scalar_grad


def add_scalar_row(output, scalar, *, reuse_output):
    if (
        reuse_output and output.device.type in ("cpu", "cuda")
        and scalar.dtype == output.dtype and scalar.device == output.device
        and not torch.is_autocast_enabled(output.device.type)
        and not torch.compiler.is_compiling()
        and not torch._C._are_functorch_transforms_active()
        and torch.autograd.forward_ad._current_level < 0
    ):
        return AddScalarRow.apply(output, scalar)
    return torch.cat((output[:1] + scalar, output[1:]), dim=0)


def merge_scalar_branch(
    outputs: torch.Tensor,
    scalar_hidden: torch.Tensor,
    output_degree_weight: torch.Tensor,
    *,
    reuse_output: bool = False,
) -> torch.Tensor:
    """Add an independent scalar branch after the grid FFN output map.

    Parameters
    ----------
    outputs : Tensor, shape (R, N, Cout)
        Coefficient-major main output. Row zero is scalar.
    scalar_hidden : Tensor, shape (N, 1, J)
        Activated scalar branch, before the final degree-zero map.
    output_degree_weight : Tensor, shape (D, Cout, J)
        Final degree maps. Only degree zero is used, without adding its bias
        again. Inputs follow PyTorch's device and floating dtype rules.
    reuse_output : bool
        Permit updates to a fresh, unshared GEMM result.

    Returns
    -------
    Tensor, shape (N, R, Cout)
        Add ``linear(scalar_hidden, output_degree_weight[0])`` to row zero.
        Return node-major logical axes. Keeping this merge separate preserves
        the caller's scalar activation and avoids evaluating that branch on the
        grid.
    """
    scalar_output = functional.linear(scalar_hidden, output_degree_weight[0])
    outputs = add_scalar_row(outputs, scalar_output.permute(1, 0, 2), reuse_output=reuse_output)
    return outputs.permute(1, 0, 2)


def coefficient_channel_linear(
    inputs: torch.Tensor,
    degree_weight: torch.Tensor,
    component_degree: torch.Tensor,
    scalar_bias: torch.Tensor | None = None,
    *,
    input_node_major: bool = True,
) -> torch.Tensor:
    """Apply a channel map selected by each coefficient's degree.

    Parameters
    ----------
    inputs : Tensor
        ``(N, R, Cin)`` if input_node_major is true, otherwise ``(R, N, Cin)``.
        Coefficient zero must be the scalar when a scalar bias is supplied.
    degree_weight : Tensor, shape (D, Cout, Cin)
        Caller-owned degree maps, following PyTorch's (output, input) convention.
    component_degree : Tensor, shape (R,)
        Integer degree index in [0, D) for each coefficient. Truncated angular
        bases are allowed when this mapping matches the supplied rows.
    scalar_bias : Tensor or None
        Optional ``(Cout,)`` or ``(1, 1, Cout)`` bias for coefficient zero only.
    input_node_major : bool, default True
        Select the logical input axes. Device and dtype follow PyTorch rules.

    Returns
    -------
    Tensor, shape (R, N, Cout)
        Row r is ``linear(inputs[:, r], degree_weight[component_degree[r]])``.
        Only row zero receives scalar_bias. All weights remain caller-owned.
    """
    component_weight = degree_weight.index_select(0, component_degree)
    coefficient_inputs = inputs.permute(1, 0, 2) if input_node_major else inputs
    outputs = torch.bmm(coefficient_inputs, component_weight.transpose(1, 2))
    if scalar_bias is not None:
        outputs = add_scalar_row(outputs, scalar_bias, reuse_output=True)
    return outputs


def project_before_grid(
    inputs: torch.Tensor,
    degree_weight: torch.Tensor,
    channel_weight: torch.Tensor,
    component_degree: torch.Tensor,
    scalar_bias: torch.Tensor | None = None,
    *,
    input_node_major: bool = True,
) -> torch.Tensor:
    """Apply ``channel_weight @ degree_weight[l]`` before grid expansion.

    Channel maps and angular transforms act on separate axes. Composing the
    channel maps here avoids applying the shared map to expanded grid values.

    ``inputs`` is ``(N, R, Cin)`` or ``(R, N, Cin)`` according to
    ``input_node_major``. ``degree_weight`` is ``(D, I, Cin)`` and
    ``channel_weight`` is ``(Cout, I)``. The integer ``component_degree``
    vector has shape ``(R,)`` and selects the degree in ``[0, D)`` for each row.
    The output has shape ``(R, N, Cout)``.

    ``scalar_bias`` is the optional first-map bias, shaped ``(I,)`` or
    ``(1, 1, I)``. It applies to coefficient zero and is projected through
    ``channel_weight``. Coefficient zero must therefore be the scalar row.
    Device, dtype, and autograd support follow PyTorch rules.
    """
    weight = torch.einsum("oh,lhi->loi", channel_weight, degree_weight)
    bias = functional.linear(scalar_bias, channel_weight) if scalar_bias is not None else None
    return coefficient_channel_linear(
        inputs, weight, component_degree, bias,
        input_node_major=input_node_major,
    )


def project_after_grid(
    coefficients: torch.Tensor,
    channel_weight: torch.Tensor,
    degree_weight: torch.Tensor,
    component_degree: torch.Tensor,
    scalar_bias: torch.Tensor | None = None,
    *,
    input_node_major: bool = False,
) -> torch.Tensor:
    """Apply ``degree_weight[l] @ channel_weight`` after grid projection.

    Moving the shared channel map across the angular projection lets it
    combine with the next degree map, without moving the nonlinearity.

    ``coefficients`` is ``(R, N, Cin)`` by default, or ``(N, R, Cin)`` when
    ``input_node_major=True``. ``channel_weight`` is ``(I, Cin)`` and
    ``degree_weight`` is ``(D, Cout, I)``. The integer ``component_degree``
    vector has shape ``(R,)`` and selects the degree in ``[0, D)`` for each row.
    The output has shape ``(R, N, Cout)``.

    ``scalar_bias`` is the optional final-map bias, shaped ``(Cout,)`` or
    ``(1, 1, Cout)``. It applies to coefficient zero after both maps.
    Coefficient zero must therefore be the scalar row. Device, dtype, and
    autograd support follow PyTorch rules.
    """
    weight = torch.einsum("loh,hj->loj", degree_weight, channel_weight)
    return coefficient_channel_linear(
        coefficients, weight, component_degree, scalar_bias,
        input_node_major=input_node_major,
    )


@triton.jit
def forward_kernel(
    hidden_ptr,
    gating_ptr,
    hidden_bias_ptr,
    gating_bias_ptr,
    output_ptr,
    nodes,
    hidden_stride_row,
    hidden_stride_node,
    hidden_stride_channel,
    gating_stride_node,
    output_stride_row,
    output_stride_node,
    output_stride_channel,
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
    scalar_output, _, _ = silu_terms(scalar)
    tl.store(
        output_ptr + node * output_stride_node + channel * output_stride_channel,
        scalar_output,
        mask=mask,
    )

    for degree in tl.static_range(1, lmax + 1):
        gate_offset = (degree - 1) * channels + channel
        raw_gate = tl.load(gating_ptr + node * gating_stride_node + gate_offset, mask=mask, other=0.0).to(tl.float32)
        raw_gate += tl.load(gating_bias_ptr + gate_offset, mask=mask, other=0.0)
        silu_gate, _, _ = silu_terms(raw_gate)
        gate = tl.sigmoid(silu_gate)
        for row in tl.static_range(1, rows):
            if row >= degree * degree and row < (degree + 1) * (degree + 1):
                value = tl.load(
                    hidden_ptr
                    + row * hidden_stride_row
                    + node * hidden_stride_node
                    + channel * hidden_stride_channel,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    output_ptr
                    + row * output_stride_row
                    + node * output_stride_node
                    + channel * output_stride_channel,
                    value * gate,
                    mask=mask,
                )


def activation_forward(hidden, gating, hidden_bias, gating_bias, *, lmax):
    gating, rows, nodes, channels, row_degrees = activation_layout(
        hidden, gating, hidden_bias, gating_bias, lmax
    )
    output = torch.empty_like(hidden)
    forward_kernel[(nodes,)](
        hidden, gating, hidden_bias, gating_bias,
        output,
        nodes,
        *hidden.stride(),
        gating.stride(0),
        *output.stride(),
        channels=channels,
        lmax=lmax,
        row_degrees=row_degrees,
        rows=rows,
        block_channels=1 << (channels - 1).bit_length(),
        num_warps=4,
    )
    return output


@triton.jit
def grid_product_forward_kernel(
    pair_ptr, gate_ptr, output_ptr,
    elements, nodes,
    grids: tl.constexpr,
    channels: tl.constexpr,
    wide_index: tl.constexpr,
    block: tl.constexpr,
):
    if wide_index:
        offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block).to(tl.int64)
    else:
        offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    channel = offsets % channels
    row = offsets // channels
    node = row - (row // nodes) * nodes
    pair_base = row * 2 * channels + channel
    gate = tl.load(gate_ptr + node * channels + channel, mask=mask)
    sigmoid_gate = tl.sigmoid(gate)
    first = tl.load(pair_ptr + pair_base, mask=mask)
    second = tl.load(pair_ptr + pair_base + channels, mask=mask)
    tl.store(output_ptr + offsets, first * second * sigmoid_gate, mask=mask)


def grid_product_forward(pair: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    nodes, grids, channels = validate(pair, gate)
    output = torch.empty((*pair.shape[:-1], channels), device=pair.device, dtype=pair.dtype)
    elements = output.numel()
    wide_index = pair.numel() > torch.iinfo(torch.int32).max
    block = 256
    grid_product_forward_kernel[(triton.cdiv(elements, block),)](
        pair, gate, output,
        elements, nodes,
        grids=grids, channels=channels,
        wide_index=wide_index,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def rms_expand_forward_kernel(
    x_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    first_weight_ptr,
    second_weight_ptr,
    expand_index_ptr,
    normalized_ptr,
    first_expanded_ptr,
    second_expanded_ptr,
    x_stride_node, x_stride_row, x_stride_channel,
    normalized_stride_node,
    normalized_stride_row,
    normalized_stride_channel,
    nodes,
    matrix_elements,
    expanded_elements,
    channels: tl.constexpr,
    rows: tl.constexpr,
    degrees: tl.constexpr,
    degree_count: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    program = tl.program_id(0)
    offsets = tl.arange(0, block)
    if program < nodes:
        channel = offsets
        channel_mask = channel < channels
        scalar = tl.load(
            x_ptr + program * x_stride_node + channel * x_stride_channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        scalar_mean = tl.sum(scalar, axis=0) / channels
        norm = tl.zeros((), dtype=tl.float32)
        for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
            if rows * channels > 4096:
                row_degree = tl.full((), 0, tl.int32)
                for candidate in tl.static_range(1, degree_count):
                    row_degree += row >= candidate * candidate
            else:
                row_degree = list(degrees)[row]
            value = tl.load(
                x_ptr
                + program * x_stride_node
                + row * x_stride_row
                + channel * x_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            if row == 0:
                value = tl.where(channel_mask, value - scalar_mean, 0.0)
            balance = 1.0 / (degree_count * (2 * row_degree + 1) * channels)
            norm += balance * tl.sum(value * value, axis=0)
        inverse = tl.rsqrt(norm + epsilon)
        for row in (tl.range if rows * channels > 4096 else tl.static_range)(0, rows):
            if rows * channels > 4096:
                row_degree = tl.full((), 0, tl.int32)
                for candidate in tl.static_range(1, degree_count):
                    row_degree += row >= candidate * candidate
            else:
                row_degree = list(degrees)[row]
            value = tl.load(
                x_ptr
                + program * x_stride_node
                + row * x_stride_row
                + channel * x_stride_channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            if row == 0:
                value = tl.where(channel_mask, value - scalar_mean, 0.0)
            weight = tl.load(
                norm_weight_ptr + row_degree * channels + channel,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            result = value * inverse * weight
            if row == 0:
                result += tl.load(norm_bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
            tl.store(
                normalized_ptr
                + program * normalized_stride_node
                + row * normalized_stride_row
                + channel * normalized_stride_channel,
                result,
                mask=channel_mask,
            )
    else:
        element = (program - nodes) * block + offsets
        mask = element < expanded_elements
        expand_row = element // matrix_elements
        column = element - expand_row * matrix_elements
        weight_degree = tl.load(expand_index_ptr + expand_row, mask=mask, other=0)
        source = weight_degree * matrix_elements + column
        first = tl.load(first_weight_ptr + source, mask=mask, other=0.0)
        second = tl.load(second_weight_ptr + source, mask=mask, other=0.0)
        tl.store(first_expanded_ptr + element, first, mask=mask)
        tl.store(second_expanded_ptr + element, second, mask=mask)


def rms_expand_forward(
    x, norm_weight, norm_bias,
    first_weight,
    second_weight,
    expand_index,
    *,
    lmax, epsilon,
):
    if second_weight.shape != (
        first_weight.shape[0],
        first_weight.shape[2],
        first_weight.shape[1],
    ):
        raise ValueError("spectral FFN maps must have reversed input and hidden widths")
    rows = (lmax + 1) ** 2
    channels = x.shape[2]
    normalized = torch.empty_strided(
        x.shape,
        (channels, x.shape[0] * channels, 1),
        device=x.device,
        dtype=x.dtype,
    )
    first_expanded = torch.empty(
        (rows, first_weight.shape[1], first_weight.shape[2]),
        device=x.device,
        dtype=x.dtype,
    )
    second_expanded = torch.empty(
        (rows, second_weight.shape[1], second_weight.shape[2]),
        device=x.device,
        dtype=x.dtype,
    )
    matrix_elements = first_weight.shape[1] * first_weight.shape[2]
    expanded_elements = first_expanded.numel()
    block = 1 << (channels - 1).bit_length()
    degrees = tuple(degree for degree in range(lmax + 1) for _ in range(2 * degree + 1))
    grid = x.shape[0] + triton.cdiv(expanded_elements, block)
    rms_expand_forward_kernel[(grid,)](
        x, norm_weight, norm_bias,
        first_weight,
        second_weight,
        expand_index,
        normalized,
        first_expanded,
        second_expanded,
        *x.stride(),
        *normalized.stride(),
        x.shape[0],
        matrix_elements,
        expanded_elements,
        channels=channels,
        rows=rows,
        degrees=degrees,
        degree_count=lmax + 1,
        epsilon=epsilon,
        block=block,
        num_warps=4,
    )
    return normalized, first_expanded, second_expanded


@triton.jit
def expand_weight_kernel(
    weight_ptr,
    expand_index_ptr,
    output_ptr,
    elements,
    matrix_elements: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    row = offsets // matrix_elements
    column = offsets - row * matrix_elements
    degree = tl.load(expand_index_ptr + row, mask=mask, other=0)
    value = tl.load(weight_ptr + degree * matrix_elements + column, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def degree_expand(weight, expand_index):
    output = torch.empty(
        (expand_index.numel(), weight.shape[1], weight.shape[2]),
        device=weight.device,
        dtype=weight.dtype,
    )
    elements = output.numel()
    matrix_elements = weight.shape[1] * weight.shape[2]
    block = 256
    expand_weight_kernel[(triton.cdiv(elements, block),)](
        weight,
        expand_index,
        output,
        elements,
        matrix_elements=matrix_elements,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit(do_not_specialize=["nodes"])
def epilogue_kernel(
    raw_ptr,
    bias_ptr,
    residual_ptr,
    output_ptr,
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
    value = tl.load(raw_ptr + raw_offset, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value += tl.where(row == 0, tl.load(bias_ptr + channel, mask=mask, other=0.0), 0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def epilogue_forward(raw, bias, residual):
    rows, nodes, channels = validate_epilogue(raw, bias, residual)
    output = torch.empty_like(residual)
    elements = output.numel()
    block = 256
    epilogue_kernel[(triton.cdiv(elements, block),)](
        raw, bias, residual, output,
        elements,
        nodes=nodes, rows=rows, channels=channels,
        block=block,
        num_warps=4,
    )
    return output


def norm_residual_forward(
    x, norm_weight, norm_bias,
    gating_weight, gating_bias,
    first_weight, first_bias,
    second_weight, second_bias,
    expand_index,
    *,
    lmax, epsilon,
):
    normalized, first_expanded, second_expanded = rms_expand_forward(
        x, norm_weight, norm_bias,
        first_weight,
        second_weight,
        expand_index,
        lmax=lmax, epsilon=epsilon,
    )
    coefficient_input = normalized.permute(1, 0, 2)
    gating = torch.mm(coefficient_input[0], gating_weight.transpose(0, 1))
    hidden = torch.bmm(coefficient_input, first_expanded.transpose(1, 2))
    activated = activation_forward(hidden, gating, first_bias, gating_bias, lmax=lmax)
    raw = torch.bmm(activated, second_expanded.transpose(1, 2))
    output = epilogue_forward(raw, second_bias, x)
    return output, (coefficient_input, gating, hidden, activated)


@triton.jit
def angular_product_kernel(
    coefficients, gate, to_grid, from_grid, gradient, coefficient_direction, gate_direction,
    output, dcoefficients, dgate,
    xs0, xs1, xs2, ts0, ts1, fs0, fs1,
    gs0, gs1, gs2, hs0, hs1, hs2, hgate_s0, hgate_s2,
    nodes, channels: tl.constexpr, rows: tl.constexpr, grids: tl.constexpr,
    stage: tl.constexpr, channel_tile: tl.constexpr, grid_tile: tl.constexpr,
    row_tile: tl.constexpr,
):
    ch = tl.program_id(0) * channel_tile + tl.arange(0, channel_tile)
    node, channel = ch // channels, ch % channels
    ch_mask = node < nodes
    row = tl.arange(0, row_tile)
    row_mask = row < rows
    mask = row_mask[:, None] & ch_mask[None, :]
    x_offset = row[:, None] * xs0 + node[None, :] * xs1 + channel[None, :] * xs2
    xa = tl.load(coefficients + x_offset, mask, 0.)
    xb = tl.load(coefficients + x_offset + channels * xs2, mask, 0.)
    s = tl.sigmoid(tl.load(gate + ch, ch_mask, 0.))
    if stage >= 1:
        sp = s * (1. - s)
        g_offset = row[:, None] * gs0 + node[None, :] * gs1 + channel[None, :] * gs2
        incoming = tl.load(gradient + g_offset, mask, 0.)
        dda = tl.zeros((row_tile, channel_tile), tl.float32)
        ddb = tl.zeros((row_tile, channel_tile), tl.float32)
        dz = tl.zeros((grid_tile, channel_tile), tl.float32)
    if stage == 2:
        h_offset = row[:, None] * hs0 + node[None, :] * hs1 + channel[None, :] * hs2
        hxa = tl.load(coefficient_direction + h_offset, mask, 0.)
        hxb = tl.load(coefficient_direction + h_offset + channels * hs2, mask, 0.)
        hz = tl.load(gate_direction + node * hgate_s0 + channel * hgate_s2, ch_mask, 0.)
        hs = sp * hz
        hsp = sp * (1. - 2. * s) * hz
    if stage == 0 or stage == 2:
        value = tl.zeros((row_tile, channel_tile), tl.float32)

    for block in range(tl.cdiv(grids, grid_tile)):
        point = block * grid_tile + tl.arange(0, grid_tile)
        on_grid = point[:, None] < grids
        keep = on_grid & row_mask[None, :]
        t = tl.load(to_grid + point[:, None] * ts0 + row[None, :] * ts1, keep, 0.)
        f = tl.load(from_grid + row[None, :] * fs0 + point[:, None] * fs1, keep, 0.)
        a = tl.dot(t, xa, input_precision="ieee")
        b = tl.dot(t, xb, input_precision="ieee")
        ab = a * b
        if stage == 0:
            value = tl.dot(tl.trans(f), tl.where(on_grid, ab * s[None, :], 0.),
                           value, input_precision="ieee")
        else:
            q = tl.dot(f, incoming, input_precision="ieee")
            if stage == 1:
                da = q * b * s[None, :]
                db = q * a * s[None, :]
                dz += tl.where(on_grid, q * ab * sp[None, :], 0.)
            else:
                ha = tl.dot(t, hxa, input_precision="ieee")
                hb = tl.dot(t, hxb, input_precision="ieee")
                hab = ha * b + a * hb
                da = q * (hb * s[None, :] + b * hs[None, :])
                db = q * (ha * s[None, :] + a * hs[None, :])
                dz += tl.where(on_grid, q * (hab * sp[None, :] + ab * hsp[None, :]), 0.)
                value = tl.dot(tl.trans(f), tl.where(on_grid, hab * s[None, :] + ab * hs[None, :], 0.),
                               value, input_precision="ieee")
            dda = tl.dot(tl.trans(t), tl.where(on_grid, da, 0.), dda, input_precision="ieee")
            ddb = tl.dot(tl.trans(t), tl.where(on_grid, db, 0.), ddb, input_precision="ieee")

    if stage == 0 or stage == 2:
        out_offset = row[:, None] * (nodes * channels) + ch[None, :]
        tl.store(output + out_offset, value, mask)
    if stage >= 1:
        dx_offset = row[:, None] * (nodes * 2 * channels) + node[None, :] * (2 * channels) + channel[None, :]
        tl.store(dcoefficients + dx_offset, dda, mask)
        tl.store(dcoefficients + dx_offset + channels, ddb, mask)
        tl.store(dgate + ch, tl.sum(dz, axis=0), ch_mask)


def angular_grid_product(pair, gate, to_grid, from_grid, *, grad=None, directions=None):
    rows, nodes, width = pair.shape
    channels = width // 2
    grids = to_grid.shape[0]
    stage = 2 if directions is not None else 1 if grad is not None else 0
    out = pair.new_empty((rows, nodes, channels)) if stage != 1 else None
    dcoefficients = pair.new_empty(pair.shape) if stage else None
    dgate = gate.new_empty(gate.shape) if stage else None
    if directions is None:
        d_pair = d_gate = None
    else:
        d_pair, d_gate = directions
    channel_tile = grid_tile = 32
    angular_product_kernel[(triton.cdiv(nodes * channels, channel_tile),)](
        pair, gate, to_grid, from_grid, grad, d_pair, d_gate, out, dcoefficients, dgate,
        *pair.stride(), *to_grid.stride(), *from_grid.stride(),
        *(grad.stride() if grad is not None else (0, 0, 0)),
        *(d_pair.stride() if d_pair is not None else (0, 0, 0)),
        *((d_gate.stride(0), d_gate.stride(2)) if d_gate is not None else (0, 0)),
        nodes, channels, rows, grids, stage, channel_tile, grid_tile,
        triton.next_power_of_2(rows), num_warps=4, num_stages=1,
    )
    if stage == 0:
        return (out,)
    if stage == 1:
        return dcoefficients, dgate
    return out, dcoefficients, dgate


@triton.jit
def separable_grid_forward_kernel(
    pair, gate, alpha, beta_to, beta_from, row_mode, output,
    nodes, channel_tile: tl.constexpr, beta_count: tl.constexpr = 14,
):
    ch = tl.program_id(0) * channel_tile + tl.arange(0, channel_tile)
    node, channel = ch // 512, ch % 512
    valid = node < nodes
    s = tl.sigmoid(tl.load(gate + ch, valid, 0.0))
    axis = tl.arange(0, 16)
    order = axis - 4
    longitude = tl.load(alpha + axis[:, None] * 9 + axis[None, :],
                        (axis[:, None] < 14) & (axis[None, :] < 9), 0.0)
    rows = tl.arange(0, 32)
    output_mode = tl.load(row_mode + rows, rows < 25, 0)
    # Keep the coefficient rows grouped by signed order across latitude steps.
    plane_a, plane_b = (), ()
    for ell in tl.static_range(5):
        row = ell * (ell + 1) + order
        addr = (row[:, None] * nodes + node[None, :]) * 1024 + channel[None, :]
        mask = (tl.abs(order) <= ell)[:, None] & valid[None, :]
        plane_a += (tl.load(pair + addr, mask, 0.0),)
        plane_b += (tl.load(pair + addr + 512, mask, 0.0),)

    result = tl.zeros((32, channel_tile), tl.float32)
    for beta in range(beta_count):
        modes_a = tl.zeros((16, channel_tile), tl.float32)
        modes_b = tl.zeros((16, channel_tile), tl.float32)
        for ell in tl.static_range(5):
            row = ell * (ell + 1) + order
            coeff = tl.load(beta_to + beta * 25 + row, tl.abs(order) <= ell, 0.0)
            modes_a = tl.fma(coeff[:, None], plane_a[ell], modes_a)
            modes_b = tl.fma(coeff[:, None], plane_b[ell], modes_b)
        a = tl.dot(longitude, modes_a, input_precision="ieee")
        b = tl.dot(longitude, modes_b, input_precision="ieee")
        modes = tl.dot(tl.trans(longitude), a * b * s[None, :], input_precision="ieee")
        expanded = tl.gather(modes, tl.broadcast_to(output_mode[:, None], (32, channel_tile)), axis=0)
        coeff = tl.load(beta_from + beta * 25 + rows, rows < 25, 0.0)
        result = tl.fma(coeff[:, None], expanded, result)

    tl.store(output + rows[:, None] * (nodes * 512) + ch[None, :], result,
             (rows[:, None] < 25) & valid[None, :])


@triton.jit
def grid_longitude_forward_kernel(
    pair, gate, beta_to, beta_from, output, nodes,
    mode_rows: tl.constexpr, row_mode: tl.constexpr, terms: tl.constexpr, block: tl.constexpr,
):
    linear = tl.program_id(0) * block + tl.arange(0, block)
    node, channel = linear // 512, linear % 512
    valid = node < nodes
    s = tl.sigmoid(tl.load(gate + linear, valid, 0.))
    a, b, result = (), (), ()
    for row in tl.static_range(25):
        index = (row * nodes + node) * 1024 + channel
        a += (tl.load(pair + index, valid, 0.),)
        b += (tl.load(pair + index + 512, valid, 0.),)
        result += (tl.zeros((block,), tl.float32),)
    for latitude in range(7):
        am = grid_latitude_modes(a, beta_to + latitude * 25, mode_rows, block)
        bm = grid_latitude_modes(b, beta_to + latitude * 25, mode_rows, block)
        q = grid_longitude_product(am, bm, terms, block)
        updated = ()
        for row in tl.static_range(25):
            weight = tl.load(beta_from + latitude * 25 + row)
            updated += (tl.fma(q[row_mode[row]] * s, weight, result[row]),)
        result = updated
    for row in tl.static_range(25):
        tl.store(output + row * nodes * 512 + linear, result[row], valid)
