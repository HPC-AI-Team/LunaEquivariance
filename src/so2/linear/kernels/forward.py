"""Apply each channel map to its order sector."""

from math import prod

import triton
import triton.language as tl


@triton.jit
def pack_signed_kernel(
    weight, packed, rows, cols,
    stride_wr, stride_wc, stride_pr, stride_pc,
    tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    col = tl.program_id(1) * tile + tl.arange(0, tile)
    mask = (row[:, None] < rows) & (col[None, :] < cols)
    source = row[:, None] * stride_wr + col[None, :] * stride_wc
    target = row[:, None] * stride_pr + col[None, :] * stride_pc
    real = tl.load(weight + source, mask, 0.)
    imag = tl.load(weight + source + rows * stride_wr, mask, 0.)
    tl.store(packed + target, real, mask)
    tl.store(packed + target + cols * stride_pc, -imag, mask)
    tl.store(packed + target + rows * stride_pr, imag, mask)
    tl.store(packed + target + rows * stride_pr + cols * stride_pc, real, mask)


def pack_signed_weight(weight):
    rows, cols = weight.shape
    packed = weight.new_empty((2 * cols, rows)).t()
    tile = 32
    pack_signed_kernel[(triton.cdiv(rows // 2, tile), triton.cdiv(cols, tile))](
        weight, packed, rows // 2, cols, *weight.stride(), *packed.stride(),
        tile, num_warps=4,
    )
    return packed


def flattened_so2_linear(linear, inputs):
    if inputs.ndim != 3 or inputs.shape[1] != 2:
        raise ValueError("SO(2) nonzero-m linear requires [E, 2, K] input")
    outputs = linear(inputs.reshape(-1, inputs.shape[-1]))
    return outputs.reshape(inputs.shape[0], 2, outputs.shape[-1])


def raw_sector_linear(sectors, scalar_map, order_maps, *, apply_pair=flattened_so2_linear):
    """Apply order-wise channel maps without complex recombination.

    ``sectors`` contains orders ``0..M``. The zero-order tensor is
    ``(E,K0)`` or ``(E,L+1,Cin)``. For ``m>0``, let ``w_m=L+1-m``.
    Accepted shapes are ``(E,2,w_m*Cin)``, ``(E,2*w_m,Cin)``, and
    ``(E,2,w_m,Cin)``. The two planes contain positive then negative
    orders, with degrees increasing from ``m`` to ``L`` in each plane.

    ``scalar_map`` consumes flattened ``(E,K0)`` rows. Each callable in
    ``order_maps`` consumes the last axis of paired ``(E,2,w_m*Cin)``
    rows. With ordinary ``functional.linear`` maps, use weight shapes
    ``(A+(L+1)*Cout,K0)`` and ``(2*w_m*Cout,w_m*Cin)``. The nonzero
    maps are bias-free to preserve equivariance. Their output channels
    concatenate real-map then imaginary-map products.

    Return ``(scalar_output, *raw_order_outputs)`` with scalar shape
    ``(E,A+(L+1)*Cout)`` and raw order shapes ``(E,2,2*w_m*Cout)``.
    No additions or subtractions occur here. ``apply_pair(map, paired)``
    can preserve the edge axis for edge-conditioned maps. Device, dtype,
    and autograd support follow the supplied maps. PyTorch linear maps
    support feature and external parameter derivatives through third order.
    """
    if len(sectors) != len(order_maps) + 1:
        raise ValueError("one channel map is required for each nonzero order")
    scalar = sectors[0]
    outputs = [scalar_map(scalar.reshape(scalar.shape[0], prod(scalar.shape[1:])))]
    for value, channel_map in zip(sectors[1:], order_maps):
        elements = prod(value.shape[1:])
        if (
            value.ndim not in (3, 4)
            or elements % 2
            or (value.ndim == 3 and value.shape[1] % 2)
            or (value.ndim == 4 and value.shape[1] != 2)
        ):
            raise ValueError("nonzero-order features require two planes per edge")
        paired = value.reshape(value.shape[0], 2, elements // 2)
        outputs.append(apply_pair(channel_map, paired))
    return tuple(outputs)


@triton.jit
def degree_coordinates(nodes, channels: tl.constexpr, block_rows: tl.constexpr,
                block_channels: tl.constexpr):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    channel = tl.arange(0, block_channels)
    node = row // 9
    angle = row % 9
    degree = tl.where(angle < 1, 0, tl.where(angle < 4, 1, 2))
    local = angle - degree * degree
    valid = (node[:, None] < nodes) & (channel[None, :] < channels)
    return row, node, degree, local, channel, valid


@triton.jit
def load_degree_rows(values, node, degree, local, channel, valid,
                 channels: tl.constexpr, present: tl.constexpr):
    value = tl.full(valid.shape, 0., tl.float32)
    for d in tl.static_range(3):
        if present[d]:
            row = node * (2 * d + 1) + local
            at = row[:, None] * channels + channel[None, :]
            select = degree[:, None] == d
            part = tl.load(values[d] + at, valid & select, 0.)
            value = tl.where(select, part, value)
    return value


@triton.jit
def assemble_projected_degrees(y0, y1, y2, scale, bias, output, nodes,
             channels: tl.constexpr, block_rows: tl.constexpr,
             block_channels: tl.constexpr):
    row, node, degree, local, channel, valid = degree_coordinates(
        nodes, channels, block_rows, block_channels,
    )
    value = load_degree_rows((y0, y1, y2), node, degree, local, channel, valid,
                         channels, (True, True, True))
    s = tl.load(scale + row, node < nodes, 0.)
    b = tl.load(bias + degree[:, None] * channels + channel[None, :],
                channel[None, :] < channels, 0.)
    result = value + s[:, None] * b
    tl.store(output + row[:, None] * channels + channel[None, :], result, valid)


def launch_degree_assembly(kernel, nodes, channels, args, *flags):
    block_rows = 4
    kernel[(triton.cdiv(nodes * 9, block_rows),)](
        *args, nodes, channels, block_rows, triton.next_power_of_2(channels),
        *flags, num_warps=4, enable_fp_fusion=False,
    )


@triton.jit
def mix_degree_weights(
    coefficients, w0, w1, w2, output, segments, degree,
    width: tl.constexpr, block_segments: tl.constexpr, block_features: tl.constexpr,
    block_experts: tl.constexpr,
):
    segment = tl.program_id(0) * block_segments + tl.arange(0, block_segments)
    feature = tl.program_id(1) * block_features + tl.arange(0, block_features)
    valid = feature < width * 128
    out_channel = feature // width
    in_channel = feature % width
    k = tl.arange(0, block_experts)
    total = tl.full((block_segments, block_features), 0., tl.float32)
    for step in range(64 // block_experts):
        expert = step * block_experts + k
        mix = tl.load(
            coefficients + segment[:, None] * 64 + expert[None, :],
            segment[:, None] < segments, 0.,
        )
        row0 = degree * 128 + out_channel
        weight = tl.load(
            w0 + expert[:, None] * (384 * 384) + row0[None, :] * 384 + in_channel[None, :],
            (valid & (in_channel < 384))[None, :], 0.,
        )
        if degree > 0:
            index = in_channel - 384
            row1 = (degree - 1) * 128 + out_channel + index // 256 * 256
            weight += tl.load(
                w1 + expert[:, None] * (512 * 256) + row1[None, :] * 256 + (index % 256)[None, :],
                (valid & (index >= 0) & (index < 512))[None, :], 0.,
            )
        if degree > 1:
            index = in_channel - 896
            row2 = out_channel + index // 128 * 128
            weight += tl.load(
                w2 + expert[:, None] * (256 * 128) + row2[None, :] * 128 + (index % 128)[None, :],
                (valid & (index >= 0))[None, :], 0.,
            )
        total = tl.dot(mix, weight, total, input_precision="ieee")
    tl.store(output + segment[:, None] * (width * 128) + feature[None, :], total,
             (segment[:, None] < segments) & valid[None, :])


def mix_receiver_weights(experts, coefficients):
    segments = coefficients.shape[0]
    weights = []
    for degree, width in enumerate((384, 896, 1152)):
        weight = coefficients.new_empty((segments, 128, width))
        mix_degree_weights[(triton.cdiv(segments, 16), triton.cdiv(width * 128, 128))](
            coefficients, *experts, weight, segments, degree, width, 16, 128, 16,
            num_warps=4, num_stages=2,
        )
        weights.append(weight)
    return tuple(weights)


@triton.jit
def projection_degree_tile(nodes, m: tl.constexpr, n: tl.constexpr, adjoint: tl.constexpr):
    index = tl.program_id(0)
    if adjoint:
        n0: tl.constexpr = 384 // n
        n1: tl.constexpr = 896 // n
        n2: tl.constexpr = 1152 // n
    else:
        n0: tl.constexpr = 128 // n
        n1: tl.constexpr = 128 // n
        n2: tl.constexpr = 128 // n
    stop0 = tl.cdiv(nodes, m) * n0
    stop1 = stop0 + tl.cdiv(3 * nodes, m) * n1
    degree = tl.where(index < stop0, 0, tl.where(index < stop1, 1, 2))
    index -= tl.where(degree == 0, 0, tl.where(degree == 1, stop0, stop1))
    tiles = tl.where(degree == 0, n0, tl.where(degree == 1, n1, n2))
    size = tl.multiple_of(tl.where(degree == 0, 384, tl.where(degree == 1, 896, 1152)), 128)
    row = index // tiles * m + tl.arange(0, m)
    tile = index % tiles
    node = tl.where(degree == 0, row, tl.where(degree == 1, row // 3, row // 5))
    position = 9 * node + degree * degree + row - node * (2 * degree + 1)
    return degree, size, row, tile, position, node < nodes


@triton.jit
def project_small_kernel(x0, x1, x2, w0, w1, w2, scale, bias, output, nodes,
                         bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    degree, width, row, tile, position, valid = projection_degree_tile(nodes, bm, bn, False)
    x = tl.where(degree == 0, x0, tl.where(degree == 1, x1, x2))
    w = tl.where(degree == 0, w0, tl.where(degree == 1, w1, w2))
    channel = tile * bn + tl.arange(0, bn)
    k = tl.arange(0, bk)
    a = x + row[:, None] * width + k[None, :]
    b = w + channel[:, None] * width + k[None, :]
    products = tl.full((bm, bn, bk), 0., tl.float32)
    for _ in range(width // bk):
        inputs = tl.load(a, valid[:, None], 0.)
        weights = tl.load(b)
        products = tl.fma(inputs[:, None, :], weights[None, :, :], products)
        a += bk
        b += bk
    acc = tl.sum(products, axis=2)
    multiplier = tl.load(scale + position, valid, 0.)
    addend = tl.load(bias + degree * 128 + channel)
    tl.store(output + position[:, None] * 128 + channel[None, :],
             acc + multiplier[:, None] * addend[None, :], valid[:, None])


def project_small_degrees(terms, weights, scale, bias, *, ampere):
    nodes = scale.shape[0]
    if nodes <= (4 if ampere else 5):
        bm, bn, bk = 2, 8, 128
    else:
        wide_tile = (ampere and nodes == 16) or nodes > 16
        bm, bn, bk = 4, 16 if wide_tile else 8, 64
    out = scale.new_empty((nodes, 9, 128))
    grid = (sum(triton.cdiv(nodes * q, bm) * (128 // bn) for q in (1, 3, 5)),)
    project_small_kernel[grid](
        *terms, *weights, scale, bias, out, nodes, bm, bn, bk,
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return out


@triton.jit
def natural_wig_index(row: tl.constexpr, column: tl.constexpr):
    if row == 0:
        return 0
    if row < 4:
        return 1 + (row - 1) * 3 + column - 1
    return 10 + (row - 4) * 5 + column - 4


@triton.jit
def rotate_ordered(w_ptr, wb, x0, x1, x2, x3, x4, x5, x6, x7, x8):
    """
    Block-diagonal Wigner rotate y = W @ x for the 9 lmax=2 coefficients.

    Loops over the three diagonal blocks (rows/cols only over nonzeros).
    Returns the 9 rotated coefficients y0..y8 as a tuple.
    """
    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8)
    y0 = tl.load(w_ptr + wb + 0) * xs[0]
    y1 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(1, 1)), xs[1],
                tl.load(w_ptr + wb + natural_wig_index(1, 2)) * xs[2])
    y2 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(2, 1)), xs[1],
                tl.load(w_ptr + wb + natural_wig_index(2, 2)) * xs[2])
    y3 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(3, 1)), xs[1],
                tl.load(w_ptr + wb + natural_wig_index(3, 2)) * xs[2])
    for j in tl.static_range(3, 4):
        y1 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(1, j)), xs[j], y1)
        y2 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(2, j)), xs[j], y2)
        y3 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(3, j)), xs[j], y3)
    y4 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(4, 4)), xs[4],
                tl.load(w_ptr + wb + natural_wig_index(4, 5)) * xs[5])
    y5 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(5, 4)), xs[4],
                tl.load(w_ptr + wb + natural_wig_index(5, 5)) * xs[5])
    y6 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(6, 4)), xs[4],
                tl.load(w_ptr + wb + natural_wig_index(6, 5)) * xs[5])
    y7 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(7, 4)), xs[4],
                tl.load(w_ptr + wb + natural_wig_index(7, 5)) * xs[5])
    y8 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(8, 4)), xs[4],
                tl.load(w_ptr + wb + natural_wig_index(8, 5)) * xs[5])
    for j in tl.static_range(6, 9):
        y4 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(4, j)), xs[j], y4)
        y5 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(5, j)), xs[j], y5)
        y6 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(6, j)), xs[j], y6)
        y7 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(7, j)), xs[j], y7)
        y8 = tl.fma(tl.load(w_ptr + wb + natural_wig_index(8, j)), xs[j], y8)
    return y0, y1, y2, y3, y4, y5, y6, y7, y8


@triton.jit
def rounded(value):
    bits = value.to(tl.int32, bitcast=True)
    rne = (bits + 4095 + ((bits >> 13) & 1)) & -8192
    return tl.where((bits & 0x7f800000) == 0x7f800000, bits, rne).to(tl.float32, bitcast=True)


@triton.jit
def power_of_two(exponent):
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def pack_row(out, inverse, counts, cols, delta, edge, c, values,
             C: tl.constexpr, groups: tl.constexpr, weight_shift: tl.constexpr, cap: tl.constexpr):
    maximum = tl.full((C,), 0.0, tl.float32)
    for i in tl.static_range(groups):
        maximum = tl.maximum(maximum, tl.abs(rounded(values[i])))
    maximum = tl.max(maximum, axis=0)
    unbiased = (maximum.to(tl.int32, bitcast=True) >> 23) - 127
    shift = tl.where(maximum > 0.0, tl.minimum(100, tl.maximum(-100, 15 - unbiased)), 0)
    scale = power_of_two(shift)
    unscale = power_of_two(-shift)
    inverse_exponent = -shift - weight_shift
    inverse_is_normal = (inverse_exponent >= -126) & (inverse_exponent <= 127)
    tl.store(inverse + edge, power_of_two(tl.where(inverse_is_normal, inverse_exponent, 0)))
    count = 0
    for i in tl.static_range(groups):
        value = rounded(values[i])
        scaled = value * scale
        half = tl.where(tl.abs(scaled) <= 65504.0, scaled, 0.0).to(tl.float16)
        residual = value - half.to(tl.float32) * unscale
        tl.store(out + edge * groups * C + i * C + c, half)
        missing = residual != 0.0
        failures = tl.sum(missing.to(tl.int32), axis=0)
        if failures > 0:
            rank = count + tl.cumsum(missing.to(tl.int32)) - 1
            write = missing & (rank < cap)
            tl.store(cols + edge * cap + rank, i * C + c, write)
            tl.store(delta + edge * cap + rank, residual, write)
        count += failures
    # A scale outside the normal range requires recomputing the row.
    tl.store(counts + edge, tl.where(inverse_is_normal, count, C * groups + 1))


@triton.jit
def gather_with_row_scale(
    x_ptr,  # [N, 9, C] node feats (L-major)
    edge_index_ptr,  # [2, E]
    wigner_ptr,  # [E, 35] compact block-diagonal matrix
    radial_ptr,  # [E, RTOT] conv1 radial (RTOT = 6*C for lmax2: 768+512+256)
    m0_ptr,  # [E, 3*2C]
    m1_ptr,  # [E, 4*2C]
    m2_ptr,  # [E, 2*2C]
    scale1, scale2, counts1, counts2, cols1, cols2, delta1, delta2,
    num_edges,
    C: tl.constexpr,  # sphere_channels (128)
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    BLOCK_C: tl.constexpr,  # == C
    GRID_E_STRIDE: tl.constexpr,
    weight_shift1: tl.constexpr, weight_shift2: tl.constexpr, cap: tl.constexpr,
):
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C

    C2 = 2 * C  # width of one packed M-row (src|tgt)
    m0_row = 3 * C2
    # radial block offsets (in units of C, radial row is [E, 6C])
    RAD = 6 * C2  # 1536 for C=128 (radial row width = 768+512+256)

    while edge_id < num_edges:
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        w_base = edge_id * 35

        # ---- load all 9 L-major coeffs, src & tgt ----
        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(x_ptr + s_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )
        xt = (
            tl.load(x_ptr + t_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )

        # ---- block-diagonal Wigner rotate (M->L y indices below) ----
        y0s, y1s, y2s, y3s, y4s, y5s, y6s, y7s, y8s = rotate_ordered(wigner_ptr, w_base, *xs)
        y0t, y1t, y2t, y3t, y4t, y5t, y6t, y7t, y8t = rotate_ordered(wigner_ptr, w_base, *xt)

        # ---- M-major rows (M<-L permute) then per-m radial scale + pack ----
        # M0<-y0 M1<-y2 M2<-y6 | M3<-y3 M4<-y7 M5<-y1 M6<-y5 | M7<-y8 M8<-y4
        # radial src at [base + c], tgt at [base + C + c]
        rb = edge_id * RAD
        rs = (
            tl.load(radial_ptr + rb + 0 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + c, mask=c_mask, other=0.0),
        )
        rt = (
            tl.load(radial_ptr + rb + 0 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + C + c, mask=c_mask, other=0.0),
        )
        # per-M-row operands, pre-permuted to M-order (radial block per row:
        # 0,1,2 | 3,4,3,4 | 5,5) so only the direct loop var indexes each tuple
        ys_perm = (y0s, y2s, y6s, y3s, y7s, y1s, y5s, y8s, y4s)
        yt_perm = (y0t, y2t, y6t, y3t, y7t, y1t, y5t, y8t, y4t)
        rs_m = (rs[0], rs[1], rs[2], rs[3], rs[4], rs[3], rs[4], rs[5], rs[5])
        rt_m = (rt[0], rt[1], rt[2], rt[3], rt[4], rt[3], rt[4], rt[5], rt[5])
        mb = edge_id * m0_row
        for m in tl.static_range(3):  # m0 buffer: rows M0,M1,M2
            tl.store(m0_ptr + mb + m * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m0_ptr + mb + m * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        m1 = (
            y3s * rs[3], y3t * rt[3], y7s * rs[4], y7t * rt[4],
            y1s * rs[3], y1t * rt[3], y5s * rs[4], y5t * rt[4],
        )
        m2 = (y8s * rs[5], y8t * rt[5], y4s * rs[5], y4t * rt[5])
        pack_row(m1_ptr, scale1, counts1, cols1, delta1, edge_id, c, m1, C, 8, weight_shift1, cap)
        pack_row(m2_ptr, scale2, counts2, cols2, delta2, edge_id, c, m2, C, 4, weight_shift2, cap)
        edge_id += GRID_E_STRIDE


@triton.jit
def project_row(wt, x, index, rotation, radial, edge, column,
                stride_n, stride_l, stride_c, stride_index,
                order: tl.constexpr, width: tl.constexpr, tile: tl.constexpr):
    source = tl.load(index + edge).to(tl.int64)
    target = tl.load(index + stride_index + edge).to(tl.int64)
    value = tl.full((tile,), 0.0, tl.float32)
    for block in tl.static_range(2 * (3 - order)):
        degree = order + block % (3 - order)
        signed_order = order if block < 3 - order else -order
        degree_width = 2 * degree + 1
        rotation_base = 1 if degree == 1 else 10
        offset = rotation_base + (degree + signed_order) * degree_width
        w = rotation + edge * 35 + offset
        w0 = tl.load(w)
        w1 = tl.load(w + 1)
        w2 = tl.load(w + 2)
        w3 = tl.load(w + 3, degree == 2, other=0.0)
        w4 = tl.load(w + 4, degree == 2, other=0.0)
        radial_group = 2 + degree if order == 1 else 5
        for i in range(256):
            node = tl.where(i < 128, source, target)
            first = x + node * stride_n + degree * degree * stride_l + (i % 128) * stride_c
            x0 = tl.load(first)
            x1 = tl.load(first + stride_l)
            x2 = tl.load(first + 2 * stride_l)
            rotated = tl.fma(w0, x0, w1 * x1)
            rotated = tl.fma(w2, x2, rotated)
            if degree == 2:
                x3 = tl.load(first + 3 * stride_l)
                x4 = tl.load(first + 4 * stride_l)
                rotated = tl.fma(w3, x3, rotated)
                rotated = tl.fma(w4, x4, rotated)
            radial_weight = tl.load(radial + edge * 1536 + radial_group * 256 + i)
            gathered = rounded(rotated * radial_weight)
            weight = tl.load(wt + (block * 256 + i) * width + column, column < width, other=0.0)
            value = tl.fma(gathered, weight, value)
    return value


@triton.jit
def restore_sparse_row(value, weight, packed, unscale, x, index, rotation, radial,
                       edge, column, stride_n, stride_l, stride_c, stride_index,
                       order: tl.constexpr, width: tl.constexpr):
    source = tl.load(index + edge).to(tl.int64)
    target = tl.load(index + stride_index + edge).to(tl.int64)
    lane = tl.arange(0, 256)
    node = tl.where(lane < 128, source, target)
    pos = (lane % 128) * stride_c
    for block in tl.static_range(2 * (3 - order)):
        degree = order + block % (3 - order)
        signed_order = order if block < 3 - order else -order
        degree_width = 2 * degree + 1
        offset = (1 if degree == 1 else 10) + (degree + signed_order) * degree_width
        w = rotation + edge * 35 + offset
        base = x + node * stride_n + degree * degree * stride_l + pos
        a = tl.fma(tl.load(w), tl.load(base), tl.load(w + 1) * tl.load(base + stride_l))
        a = tl.fma(tl.load(w + 2), tl.load(base + 2 * stride_l), a)
        if degree == 2:
            a = tl.fma(tl.load(w + 3), tl.load(base + 3 * stride_l), a)
            a = tl.fma(tl.load(w + 4), tl.load(base + 4 * stride_l), a)
        radial_group = 2 + degree if order == 1 else 5
        a = rounded(a * tl.load(radial + edge * 1536 + radial_group * 256 + lane))
        index_in_row = block * 256 + lane
        b = tl.load(packed + edge * (2 * (3 - order) * 256) + index_in_row)
        residual = a - b.to(tl.float32) * unscale
        changed = residual != 0.0
        # Visit only the terms lost in the half-precision packing.
        k = tl.min(tl.where(changed, lane, 256), 0)
        while k < 256:
            correction = tl.sum(tl.where(lane == k, residual, 0.0), 0)
            weight_offset = (block * 256 + k) * width + column
            row_weight = tl.load(weight + weight_offset, column < width, other=0.0)
            value = tl.fma(correction, row_weight, value)
            k = tl.min(tl.where(changed & (lane > k), lane, 256), 0)
    return value


@triton.jit
def apply_sparse(y1, y2, w1, w2, counts1, counts2, cols1, cols2, delta1, delta2,
                 half1, half2, inv1, inv2,
                 x, index, rotation, radial, edges, stride_n, stride_l, stride_c, stride_index,
                 shift1: tl.constexpr, shift2: tl.constexpr, cap: tl.constexpr, batch: tl.constexpr):
    order = tl.program_id(1) + 1
    first = tl.program_id(0) * batch
    group = first + tl.arange(0, batch)
    cols = tl.arange(0, 512)
    counts = counts1 if order == 1 else counts2
    failures = tl.load(counts + group, group < edges, other=0)
    if tl.sum(failures, axis=0) > 0:
        for slot in range(batch):
            edge = first + slot
            n = tl.load(counts + edge, edge < edges, other=0)
            if n > 0:
                if order == 1:
                    correct(y1, w1, cols1, delta1, half1, inv1, x, index, rotation, radial,
                            edge, n, cols, stride_n, stride_l, stride_c, stride_index, 1, 512, shift1, cap)
                else:
                    correct(y2, w2, cols2, delta2, half2, inv2, x, index, rotation, radial,
                            edge, n, cols, stride_n, stride_l, stride_c, stride_index, 2, 256, shift2, cap)


@triton.jit
def correct(out, weight, indices, residuals, packed, inverse, x, index, rotation, radial,
            edge, count, col, stride_n, stride_l, stride_c, stride_index,
            order: tl.constexpr, width: tl.constexpr, weight_shift: tl.constexpr, cap: tl.constexpr):
    address = out + edge * width + col
    if count > cap:
        combined_inverse = tl.load(inverse + edge)
        unscale = combined_inverse * (2.0 ** weight_shift)
        scale_valid = (combined_inverse > 0.0) & (combined_inverse < float("inf"))
        scale_valid &= (unscale > 0.0) & (unscale < float("inf"))
        if (count <= (3 - order) * 256) & scale_valid:
            value = tl.load(address, col < width, other=0.0)
            value = restore_sparse_row(value, weight, packed, unscale, x, index, rotation, radial,
                                       edge, col, stride_n, stride_l, stride_c, stride_index, order, width)
        else:
            value = project_row(weight, x, index, rotation, radial, edge, col,
                                stride_n, stride_l, stride_c, stride_index, order, width, 512)
    else:
        value = tl.load(address, col < width, other=0.0)
        for k in range(count):
            offset = tl.load(indices + edge * cap + k)
            residual = tl.load(residuals + edge * cap + k)
            w = tl.load(weight + offset * width + col, col < width, other=0.0)
            value = tl.fma(residual, w, value)
    tl.store(address, value, col < width)


AMPERE_GATHER_FORWARD_SOURCE = r"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int channels = 128;
constexpr int radial_width = 1536;

__device__ __forceinline__ void read4(float* values, const float* ptr, int lane) {
    const float4 v = reinterpret_cast<const float4*>(ptr)[lane];
    values[0] = v.x;
    values[1] = v.y;
    values[2] = v.z;
    values[3] = v.w;
}

__device__ __forceinline__ void write4(float* ptr, const float* v, int lane) {
    reinterpret_cast<float4*>(ptr)[lane] = make_float4(v[0], v[1], v[2], v[3]);
}

struct Residual {
    int* active;
    int* total;
    int* counts;
    int* columns;
    int* edge_pair;
    float* values;
    float scale;
    int cap;
};

struct Residuals { Residual order[3]; };

__device__ __forceinline__ void write_packed(
    __half* ptr, const float* value, int edge, int col, int lane, int& running, const Residual& residual
) {
    __half encoded[4];
    float difference[4];
    unsigned misses = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned bits = __float_as_uint(value[i]);
        const float rounded = __uint_as_float((bits + 4096u) & ~8191u);
        const float scaled = rounded * residual.scale;
        const float mag = fabsf(scaled);
        const bool keep = mag >= 0x1p-14f && mag <= 65504.0f;
        encoded[i] = __float2half_rn(keep ? scaled : 0.0f);
        difference[i] = keep ? 0.0f : rounded;
        misses |= (difference[i] != 0.0f) << i;
    }
    const unsigned p0 = static_cast<unsigned>(__half_as_ushort(encoded[0])) |
                        (static_cast<unsigned>(__half_as_ushort(encoded[1])) << 16);
    const unsigned p1 = static_cast<unsigned>(__half_as_ushort(encoded[2])) |
                        (static_cast<unsigned>(__half_as_ushort(encoded[3])) << 16);
    reinterpret_cast<uint2*>(ptr)[lane] = make_uint2(p0, p1);
    const unsigned live = __ballot_sync(0xffffffff, misses != 0);
    if (!live) return;

    const int count = __popc(misses);
    int position = running;
    if ((live & (live - 1)) == 0) {
        running += __shfl_sync(0xffffffff, count, __ffs(live) - 1);
    } else {
        const unsigned below = (1u << lane) - 1u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned item = __ballot_sync(0xffffffff, (misses >> i) & 1);
            position += __popc(item & below);
            running += __popc(item);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if ((misses >> i) & 1) {
            if (position < residual.cap) {
                residual.columns[edge * residual.cap + position] = col + 4 * lane + i;
                residual.values[edge * residual.cap + position] = difference[i];
            }
            ++position;
        }
    }
}

__device__ __forceinline__ void publish_residual(int edge, int pair, int count, const Residual& residual) {
    if (count != 0) {
        const int slot = atomicAdd(residual.total, 1);
        residual.active[slot] = edge;
        residual.counts[edge] = count;
        residual.edge_pair[edge] = pair;
    }
}

template <int Degree>
__device__ __forceinline__ void load_node(float (*values)[4], const float* x, int node, int lane) {
    #pragma unroll
    for (int c = 0; c < 2 * Degree + 1; ++c)
        read4(values[c], x + (node * 9 + Degree * Degree + c) * channels, lane);
}

template <int Degree, int Order, int Packed>
__device__ __forceinline__ void store_order(
    const float* radial, const float* rot0, const float* rot1, float lo0, float lo1, const float (*xs)[4],
    const float (*xt)[4], int first, int reverse, int lane, float* out,
    int& count0, int& count1, const Residual& residual
) {
    constexpr int width = 2 * Degree + 1;
    constexpr int rows = Order ? 2 * (3 - Order) : 3;
    constexpr int radial_group = (Order == 0 ? 0 : (Order == 1 ? 3 : 5)) + Degree - Order;
    constexpr int wbase = Degree == 0 ? 0 : (Degree == 1 ? 1 : 10);
    float rs[4], rt[4];
    read4(rs, radial + radial_group * 2 * channels, lane);
    read4(rt, radial + (radial_group * 2 + 1) * channels, lane);
    #pragma unroll
    for (int sign = 0; sign < (Order ? 2 : 1); ++sign) {
        const int row = Degree - Order + sign * (3 - Order);
        const int rid = Order ? 2 * Order - 1 + sign : 0;
        const int start = wbase + rid * width;
        float y0s[4], y0t[4], y1s[4], y1t[4];
        const float w0 = Degree > 0 ? __shfl_sync(0xffffffff, lo0, rid * width)
                : __shfl_sync(0xffffffff, lane == 0 ? rot0[start] : 0.0f, 0);
        const float w1 = Degree > 0 ? __shfl_sync(0xffffffff, lo1, rid * width)
                : __shfl_sync(0xffffffff, lane == 0 && reverse >= 0 ? rot1[start] : 0.0f, 0);
        if constexpr (width > 1) {
            const float a = Degree > 0 ? __shfl_sync(0xffffffff, lo0, rid * width + 1)
                : __shfl_sync(0xffffffff, lane == 0 ? rot0[start + 1] : 0.0f, 0);
            const float b = Degree > 0 ? __shfl_sync(0xffffffff, lo1, rid * width + 1)
                : __shfl_sync(0xffffffff, lane == 0 && reverse >= 0 ? rot1[start + 1] : 0.0f, 0);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                y0s[k] = __fmaf_rn(w0, xs[0][k], __fmul_rn(a, xs[1][k]));
                y0t[k] = __fmaf_rn(w0, xt[0][k], __fmul_rn(a, xt[1][k]));
                y1s[k] = __fmaf_rn(w1, xt[0][k], __fmul_rn(b, xt[1][k]));
                y1t[k] = __fmaf_rn(w1, xs[0][k], __fmul_rn(b, xs[1][k]));
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                y0s[k] = __fmul_rn(w0, xs[0][k]);
                y0t[k] = __fmul_rn(w0, xt[0][k]);
                y1s[k] = __fmul_rn(w1, xt[0][k]);
                y1t[k] = __fmul_rn(w1, xs[0][k]);
            }
        }
        #pragma unroll
        for (int c = 2; c < width; ++c) {
            const float a = Degree > 0 ? __shfl_sync(0xffffffff, lo0, rid * width + c)
                : __shfl_sync(0xffffffff, lane == 0 ? rot0[start + c] : 0.0f, 0);
            const float b = Degree > 0 ? __shfl_sync(0xffffffff, lo1, rid * width + c)
                : __shfl_sync(0xffffffff, lane == 0 && reverse >= 0 ? rot1[start + c] : 0.0f, 0);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                y0s[k] = __fmaf_rn(a, xs[c][k], y0s[k]);
                y0t[k] = __fmaf_rn(a, xt[c][k], y0t[k]);
                y1s[k] = __fmaf_rn(b, xt[c][k], y1s[k]);
                y1t[k] = __fmaf_rn(b, xs[c][k], y1t[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            y0s[k] = __fmul_rn(y0s[k], rs[k]);
            y0t[k] = __fmul_rn(y0t[k], rt[k]);
            y1s[k] = __fmul_rn(y1s[k], rs[k]);
            y1t[k] = __fmul_rn(y1t[k], rt[k]);
        }
        const int row_offset = row * 2 * channels;
        const int off0 = first * rows * 2 * channels + row_offset;
        const int off1 = reverse * rows * 2 * channels + row_offset;
        if constexpr ((Packed & (1 << Order)) != 0) {
            __half* p = reinterpret_cast<__half*>(out);
            write_packed(p + off0, y0s, first, row_offset, lane, count0, residual);
            write_packed(p + off0 + channels, y0t, first, row_offset + channels, lane, count0, residual);
            if (reverse >= 0) {
                write_packed(p + off1, y1s, reverse, row_offset, lane, count1, residual);
                write_packed(p + off1 + channels, y1t, reverse, row_offset + channels, lane, count1, residual);
            }
        } else {
            write4(out + off0, y0s, lane);
            write4(out + off0 + channels, y0t, lane);
            if (reverse >= 0) {
                write4(out + off1, y1s, lane);
                write4(out + off1 + channels, y1t, lane);
            }
        }
    }
}

template <int Degree, int Packed>
__device__ __forceinline__ void store_degree(
    const float* radial, const float* rot0, const float* rot1, const float (*xs)[4],
    const float (*xt)[4], int first, int reverse, int lane, float* m0, float* m1, float* m2,
    int (&count0)[3], int (&count1)[3], const Residuals& residuals
) {
    float lo0 = 0.0f, lo1 = 0.0f;
    if constexpr (Degree > 0) {
        constexpr int width = 2 * Degree + 1;
        constexpr int base = Degree == 1 ? 1 : 10;
        if (lane < width * width) {
            lo0 = rot0[base + lane];
            if (reverse >= 0) lo1 = rot1[base + lane];
        }
    }
    store_order<Degree, 0, Packed>(radial, rot0, rot1, lo0, lo1, xs, xt, first, reverse, lane, m0, count0[0], count1[0], residuals.order[0]);
    if constexpr (Degree > 0)
        store_order<Degree, 1, Packed>(radial, rot0, rot1, lo0, lo1, xs, xt, first, reverse, lane, m1, count0[1], count1[1], residuals.order[1]);
    if constexpr (Degree > 1)
        store_order<Degree, 2, Packed>(radial, rot0, rot1, lo0, lo1, xs, xt, first, reverse, lane, m2, count0[2], count1[2], residuals.order[2]);
}

template <bool Owned, int Packed>
__global__ void gather_forward(
    const float* x, const int64_t* index, const float* radial, const float* rotation,
    const int* pair_edges, const int* pair_row_ptr, float* m0, float* m1, float* m2,
    int edges, int pairs, Residuals residuals
) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    int begin, end, node;
    if constexpr (Owned) {
        begin = pair_row_ptr[blockIdx.x] + blockIdx.y * warps + warp;
        end = pair_row_ptr[blockIdx.x + 1];
        node = blockIdx.x;
    } else {
        begin = blockIdx.x * warps + warp;
        end = min(begin + 1, pairs);
        node = begin < end ? index[edges + pair_edges[begin]] : 0;
    }
    if (begin >= end) return;
    float xt0[1][4], xt1[3][4], xt2[5][4];
    load_node<0>(xt0, x, node, lane);
    load_node<1>(xt1, x, node, lane);
    load_node<2>(xt2, x, node, lane);
    for (int pair = begin; pair < end; pair += warps * gridDim.y) {
        const int first = pair_edges[pair], reverse = pair_edges[pairs + pair];
        const int src = index[first];
        const float* r = radial + pair * radial_width;
        const float* w0 = rotation + first * 35;
        const float* w1 = reverse >= 0 ? rotation + reverse * 35 : w0;
        float xs0[1][4], xs1[3][4], xs2[5][4];
        int count0[3] = {}, count1[3] = {};
        load_node<0>(xs0, x, src, lane);
        store_degree<0, Packed>(r, w0, w1, xs0, xt0, first, reverse, lane, m0, m1, m2, count0, count1, residuals);
        load_node<1>(xs1, x, src, lane);
        store_degree<1, Packed>(r, w0, w1, xs1, xt1, first, reverse, lane, m0, m1, m2, count0, count1, residuals);
        load_node<2>(xs2, x, src, lane);
        store_degree<2, Packed>(r, w0, w1, xs2, xt2, first, reverse, lane, m0, m1, m2, count0, count1, residuals);
        if (lane == 0) {
            #pragma unroll
            for (int order = 0; order < 3; ++order) {
                if (Packed & (1 << order)) {
                    publish_residual(first, pair, count0[order], residuals.order[order]);
                    if (reverse >= 0)
                        publish_residual(reverse, pair, count1[order], residuals.order[order]);
                }
            }
        }
    }
}
}

extern "C" int launch_gather_forward(
    const float* x, const int64_t* index, const float* radial, const float* rotation,
    const int* pair_edges, float* m0, float* m1, float* m2,
    const void* metadata, int edges, int pairs, cudaStream_t stream
) {
    constexpr int warps = 4;
    if (pairs == 0) return cudaSuccess;
    const auto* residuals = static_cast<const Residuals*>(metadata);
    gather_forward<false, 7><<<(pairs + warps - 1) / warps, warps * 32, 0, stream>>>(
        x, index, radial, rotation, pair_edges, nullptr, m0, m1, m2, edges, pairs, *residuals);
    return cudaPeekAtLastError();
}
"""


AMPERE_CORRECTION_SOURCE = r"""#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int channels = 128;

struct Residual {
    int* active;
    int* total;
    int* counts;
    int* columns;
    int* edge_pair;
    float* values;
    float scale;
    int cap;
};

// A CTA owns one active output row. For an overflowing row, reconstruct all
// its rounded inputs and overwrite its FP32 output, without using capped data.
template <int Order>
__global__ void correct_order(
    float* output, const float* wt, const float* x, const int64_t* index,
    const float* radial, const float* rotation, const int* active, const int* total,
    const int* counts, const int* columns, const int* edge_pair, const float* values,
    const float* bias, int edges, int cap, int output_width
) {
    constexpr int rows = Order == 0 ? 3 : 2 * (3 - Order);
    constexpr int inner = 256 * rows;
    __shared__ float row[inner];
    for (int slot = blockIdx.x; slot < *total; slot += gridDim.x) {
        const int edge = active[slot];
        const int count = counts[edge];
        if (count <= cap) {
            for (int c = threadIdx.x; c < output_width; c += blockDim.x) {
                float v = output[edge * output_width + c];
                for (int i = 0; i < count; ++i) {
                    const int k = columns[edge * cap + i];
                    v = fmaf(values[edge * cap + i], wt[k * output_width + c], v);
                }
                output[edge * output_width + c] = v;
            }
        } else {
            for (int k = threadIdx.x; k < inner; k += blockDim.x) {
                const int sector = k / 256;
                const int degree = Order + sector % (3 - Order);
                const int sign = sector / (3 - Order);
                const int endpoint = k / 128 % 2;
                const int channel = k % 128;
                const int width = 2 * degree + 1;
                const int start = degree * degree;
                const int rid = (degree == 0 ? 0 : (degree == 1 ? 1 : 10)) + (Order ? 2 * Order - 1 + sign : 0) * width;
                const int node = index[endpoint * edges + edge];
                const float* input = x + (node * 9 + start) * channels + channel;
                const float* w = rotation + edge * 35 + rid;
                float v = width == 1 ? __fmul_rn(w[0], input[0])
                    : __fmaf_rn(w[0], input[0], __fmul_rn(w[1], input[channels]));
                for (int col = 2; col < width; ++col)
                    v = __fmaf_rn(w[col], input[col * channels], v);
                const int g = (Order == 0 ? 0 : (Order == 1 ? 3 : 5)) + degree - Order;
                v = __fmul_rn(v, radial[(edge_pair[edge] * 6 + g) * 256 + endpoint * channels + channel]);
                row[k] = __uint_as_float((__float_as_uint(v) + 4096u) & ~8191u);
            }
            __syncthreads();
            for (int c = threadIdx.x; c < output_width; c += blockDim.x) {
                float v = 0.0f;
                for (int k = 0; k < inner; ++k)
                    v = fmaf(row[k], wt[k * output_width + c], v);
                output[edge * output_width + c] = bias ? v + bias[c] : v;
            }
            __syncthreads();
        }
    }
}
}

extern "C" int launch_correction(
    float* output, const float* wt, const float* x, const int64_t* index,
    const float* radial, const float* rotation, const void* metadata,
    const float* bias, int edges, int output_width, int order, cudaStream_t stream
) {
    const auto* residual = static_cast<const Residual*>(metadata);
    #define LAUNCH(O) correct_order<O><<<256, 128, 0, stream>>>( \
        output, wt, x, index, radial, rotation, residual->active, residual->total, \
        residual->counts, residual->columns, residual->edge_pair, residual->values, \
        bias, edges, residual->cap, output_width)
    switch (order) {
        case 0: LAUNCH(0); break;
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        default: return cudaErrorInvalidValue;
    }
    #undef LAUNCH
    return cudaPeekAtLastError();
}
"""


PROJECTION_LT_SOURCE = r"""#include <cublasLt.h>
#include <cstdint>
#include <vector>

struct Plans {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
    uint64_t outputs{};
    cublasLtMatmulPreference_t preference{};
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;

    ~Plans() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }

    cublasStatus_t create(uint64_t edges, uint64_t outputs, uint64_t inner,
                          bool row_rhs, const void* bias, size_t workspace, int requested,
                          bool split_output = false) {
        this->outputs = outputs;
        // Express a row-major result Y as a column-major view of Y transpose.
        const cublasOperation_t trans_a = row_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        auto status = cublasLtCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        auto compute = split_output ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        auto input_type = split_output ? CUDA_R_32F : CUDA_R_16F;
        status = cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        if (bias) {
            const cudaDataType_t bias_type = CUDA_R_32F;
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
            if (status != CUBLAS_STATUS_SUCCESS) return status;
        }
        const uint64_t a_rows = row_rhs ? inner : outputs;
        const uint64_t a_cols = row_rhs ? outputs : inner;
        status = cublasLtMatrixLayoutCreate(&a, input_type, a_rows, a_cols, a_rows);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        status = cublasLtMatrixLayoutCreate(&b, input_type, inner, edges, inner);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        if (split_output) {
            status = cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, outputs, edges, outputs);
            if (status != CUBLAS_STATUS_SUCCESS) return status;
        }
        status = cublasLtMatrixLayoutCreate(&d, CUDA_R_32F, outputs, edges, split_output ? 2 * outputs : outputs);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        status = cublasLtMatmulPreferenceCreate(&preference);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        algorithms.resize(requested);
        int count = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            handle, op, a, b, c ? c : d, d, preference, requested, algorithms.data(), &count);
        algorithms.resize(count);
        if (!split_output || status != CUBLAS_STATUS_SUCCESS) return status;

        constexpr uint64_t strict_fp32 = CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA
                                       | CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F
                                       | CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F;
        constexpr uint64_t precision = CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK
                                     | CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK
                                     | CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK;
        const cublasLtMatmulHeuristicResult_t* chosen = nullptr;
        for (const auto& candidate : algorithms) {
            if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize) continue;
            uint64_t flags = 0;
            uint32_t tile = 0;
            size_t written = 0;
            status = cublasLtMatmulAlgoCapGetAttribute(
                &candidate.algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &flags, sizeof(flags), &written);
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            if ((flags & precision) != strict_fp32) continue;
            status = cublasLtMatmulAlgoConfigGetAttribute(
                &candidate.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &written);
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            if (!chosen || tile == CUBLASLT_MATMUL_TILE_128x128) chosen = &candidate;
            if (tile == CUBLASLT_MATMUL_TILE_128x128) break;
        }
        if (!chosen) return CUBLAS_STATUS_NOT_SUPPORTED;
        const auto selection = *chosen;
        algorithms.assign(1, selection);
        return CUBLAS_STATUS_SUCCESS;
    }
};

extern "C" {
void* plans_create(uint64_t edges, uint64_t outputs, uint64_t inner, bool row_rhs,
                   const void* bias, size_t workspace, int requested, int* status) {
    auto* plans = new Plans;
    *status = plans->create(edges, outputs, inner, row_rhs, bias, workspace, requested);
    if (*status != CUBLAS_STATUS_SUCCESS) {
        delete plans;
        return nullptr;
    }
    return plans;
}

void* plans_create_frozen(uint64_t edges, uint64_t outputs, uint64_t inner, int* status) {
    auto* plans = new Plans;
    *status = plans->create(edges, outputs, inner, false, nullptr, 0, 16, true);
    if (*status != CUBLAS_STATUS_SUCCESS) {
        delete plans;
        if (*status == CUBLAS_STATUS_NOT_SUPPORTED) *status = CUBLAS_STATUS_SUCCESS;
        return nullptr;
    }
    return plans;
}

void plans_destroy(Plans* plans) {
    delete plans;
}

int plans_count(Plans* plans) {
    return plans->algorithms.size();
}

int plans_run(Plans* plans, int index, const void* a, const void* b, void* d,
              void* workspace, size_t bytes, float alpha, void* stream) {
    const float beta = 0.f;
    return cublasLtMatmul(
        plans->handle, plans->op, &alpha, a, plans->a, b, plans->b, &beta,
        d, plans->d, d, plans->d, &plans->algorithms[index].algo,
        workspace, bytes, static_cast<cudaStream_t>(stream));
}

int plans_run_frozen(Plans* plans, const void* dp, const void* dq, const void* real,
                     const void* imag, const void* shared, float* output, cudaStream_t stream) {
    const float alpha = 1.f, beta = 1.f;
    auto apply = [&](const void* seed, const void* weight, float* destination) {
        return cublasLtMatmul(
            plans->handle, plans->op, &alpha, weight, plans->a, seed, plans->b, &beta,
            shared, plans->c, destination, plans->d, &plans->algorithms[0].algo, nullptr, 0, stream);
    };
    auto status = apply(dp, real, output);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    return apply(dq, imag, output + plans->outputs);
}

}
"""


HOPPER_ROW_SCALED_SOURCE = r"""#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>

namespace {

using namespace cute;
using namespace cutlass::epilogue::fusion;

template<bool Cooperative>
struct ScaledProjection {
    using M = conditional_t<Cooperative, _128, _64>;
    using Tile = Shape<M, _128, _64>;
    using Cluster = Shape<_1, _1, _1>;
    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using EpiSchedule = conditional_t<Cooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecialized>;
    using MainSchedule = conditional_t<Cooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                       cutlass::gemm::KernelTmaWarpSpecialized>;
    using Scale = Sm90ColBroadcast<0, Tile, float>;
    using Multiply = Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Fusion = Sm90EVT<Multiply, Scale, Sm90AccFetch>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster, Shape<M, _128>,
        float, float, void, RowMajor, 4, float, RowMajor, 4, EpiSchedule, Fusion
    >::CollectiveOp;
    using Half = cutlass::half_t;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Half, RowMajor, 8, Half, ColumnMajor, 8,
        float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        MainSchedule
    >::CollectiveOp;
    using Universal = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Universal>;

    static int run(
        const void* input, const void* weight, const float* scale, float* output,
        int rows, int cols, int inner, int sms, cudaStream_t stream
    ) {
        const typename Universal::StrideA stride_a{inner, _1{}, int64_t(rows) * inner};
        const typename Universal::StrideB stride_b{inner, _1{}, int64_t(cols) * inner};
        const typename Universal::StrideC stride_c{cols, _1{}, int64_t(rows) * cols};
        const typename Universal::StrideD stride_d{cols, _1{}, int64_t(rows) * cols};
        cutlass::KernelHardwareInfo hardware{};
        hardware.sm_count = sms;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {rows, cols, inner, 1},
            {static_cast<const Half*>(input), stride_a, static_cast<const Half*>(weight), stride_b},
            {{{scale, 1.f, {}}, {}, {}}, nullptr, stride_c, output, stride_d},
            hardware,
        };
        if (Gemm::get_workspace_size(args) != 0)
            return -1;
        Gemm gemm;
        auto status = gemm.can_implement(args);
        if (status != cutlass::Status::kSuccess)
            return static_cast<int>(status);
        status = gemm.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess)
            return static_cast<int>(status);
        return static_cast<int>(gemm.run(stream));
    }
};

}

extern "C" int row_scaled_projection(
    const void* input, const void* weight, const float* scale, float* output,
    int rows, int cols, int inner, int sms, bool cooperative, cudaStream_t stream
) {
    if (cooperative)
        return ScaledProjection<true>::run(input, weight, scale, output, rows, cols, inner, sms, stream);
    return ScaledProjection<false>::run(input, weight, scale, output, rows, cols, inner, sms, stream);
}
"""


HOPPER_ROW_DISPATCH_SOURCE = r"""#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <limits>

extern "C" int row_scaled_projection(const void*, const void*, const float*, float*,
                                     int, int, int, int, bool, cudaStream_t);

namespace {

at::Tensor project(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale) {
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && scale.dim() == 1
                && x.size(0) == scale.size(0) && x.size(1) == weight.size(1),
                "Invalid UMA row-scaled projection dimensions");
    TORCH_CHECK(x.is_cuda() && x.device() == weight.device() && x.device() == scale.device()
                && x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf
                && scale.scalar_type() == at::kFloat && x.is_contiguous()
                && weight.is_contiguous() && scale.is_contiguous(),
                "Expected contiguous CUDA FP16 operands and FP32 scale");
    const auto rows = x.size(0), cols = weight.size(0), inner = x.size(1);
    TORCH_CHECK(rows > 0 && rows <= std::numeric_limits<int>::max()
                && ((inner == 1024 && cols == 512) || (inner == 512 && cols == 256)),
                "Expected supported nonzero-order UMA-S projections");
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty({rows, cols}, x.options().dtype(at::kFloat));
    auto properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0, "Expected Hopper device");
    const auto code = row_scaled_projection(
        x.const_data_ptr<at::Half>(), weight.const_data_ptr<at::Half>(),
        scale.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
        rows, cols, inner, properties->multiProcessorCount, true,
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(code == 0, "Row-scaled CUTLASS projection failed with status ", code);
    return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trim_h100, m) {
    m.def("uma_row_scaled_matmul(Tensor x, Tensor weight, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(trim_h100, CUDA, m) {
    m.impl("uma_row_scaled_matmul", project);
}
"""
