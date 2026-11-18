"""Differentiate degree assembly and its rotation-weighted bias."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from .forward import degree_coordinates, launch_degree_assembly, projection_degree_tile


@triton.jit
def fold_signed_kernel(
    gradient, folded, rows, cols, stride_gr, stride_gc, tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    col = tl.program_id(1) * tile + tl.arange(0, tile)
    mask = (row[:, None] < rows) & (col[None, :] < cols)
    source = row[:, None] * stride_gr + col[None, :] * stride_gc
    target = row[:, None] * cols + col[None, :]
    q00 = tl.load(gradient + source, mask, 0.)
    q01 = tl.load(gradient + source + cols * stride_gc, mask, 0.)
    q10 = tl.load(gradient + source + rows * stride_gr, mask, 0.)
    q11 = tl.load(gradient + source + rows * stride_gr + cols * stride_gc, mask, 0.)
    tl.store(folded + target, q00 + q11, mask)
    tl.store(folded + target + rows * cols, q10 - q01, mask)


def fold_signed_weight(gradient):
    rows, cols = gradient.shape
    folded = gradient.new_empty((rows, cols // 2))
    tile = 32
    fold_signed_kernel[(triton.cdiv(rows // 2, tile), triton.cdiv(cols // 2, tile))](
        gradient, folded, rows // 2, cols // 2, *gradient.stride(),
        tile, num_warps=4,
    )
    return folded


@triton.jit
def merge_complex_input_gradient(
    real, imaginary, shared, output, elements, width: tl.constexpr,
    has_real: tl.constexpr, has_imaginary: tl.constexpr, has_shared: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    live = i < elements
    common = tl.load(shared + i, live, 0.) if has_shared else tl.full((block,), 0., tl.float32)
    left, right = common, common
    if has_real:
        left += tl.load(real + i, live, 0.)
    if has_imaginary:
        right += tl.load(imaginary + i, live, 0.)
    dest = i + i // width * width
    tl.store(output + dest, left, live)
    tl.store(output + dest + width, right, live)


@triton.jit
def split_degree_gradients(g, bias, gy0, gy1, gy2, gscale, nodes,
             channels: tl.constexpr, block_rows: tl.constexpr,
             block_channels: tl.constexpr, need_degrees: tl.constexpr,
             need_scale: tl.constexpr):
    row, node, degree, local, channel, valid = degree_coordinates(
        nodes, channels, block_rows, block_channels,
    )
    grad = tl.load(g + row[:, None] * channels + channel[None, :], valid, 0.)
    gy = (gy0, gy1, gy2)
    for d in tl.static_range(3):
        if need_degrees[d]:
            at = (node * (2 * d + 1) + local)[:, None] * channels + channel[None, :]
            tl.store(gy[d] + at, grad, valid & (degree[:, None] == d))
    if need_scale:
        b = tl.load(bias + degree[:, None] * channels + channel[None, :],
                    channel[None, :] < channels, 0.)
        tl.store(gscale + row, tl.sum(grad * b, axis=1), node < nodes)


@triton.jit
def reduce_bias_gradient(g, scale, gbias, nodes, channels: tl.constexpr,
             block_rows: tl.constexpr, block_channels: tl.constexpr):
    degree = tl.program_id(0)
    width = 2 * degree + 1
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    lane = tl.arange(0, block_rows)
    acc = tl.full((block_rows, block_channels), 0., tl.float32)
    for start in range(0, nodes * width, block_rows):
        item = start + lane
        row = (item // width) * 9 + degree * degree + item % width
        live = item < nodes * width
        s = tl.load(scale + row, live, 0.)
        v = tl.load(g + row[:, None] * channels + channel[None, :],
                    live[:, None] & (channel[None, :] < channels), 0.)
        acc += v * s[:, None]
    tl.store(gbias + degree * channels + channel, tl.sum(acc, axis=0), channel < channels)


def projected_bias_gradient(g, scale, bias):
    nodes, _, channels = g.shape
    result = torch.empty_like(bias)
    block_rows, block_channels = 128, 4
    reduce_bias_gradient[(3, triton.cdiv(channels, block_channels))](
        g, scale, result, nodes, channels, block_rows, block_channels,
        num_warps=4, enable_fp_fusion=False,
    )
    return result


def degree_assembly_backward(g, scale, bias, needs):
    nodes, _, channels = g.shape
    need_degrees, need_scale, need_bias = needs[:3], needs[3], needs[4]
    gy = tuple(g.new_empty((nodes, 2 * degree + 1, channels)) if need else None
               for degree, need in enumerate(need_degrees))
    gs = torch.empty_like(scale) if need_scale else None
    if any(need_degrees) or need_scale:
        launch_degree_assembly(split_degree_gradients, nodes, channels, (g, bias, *gy, gs), need_degrees, need_scale)
    gb = projected_bias_gradient(g, scale, bias) if need_bias else None
    return *gy, gs, gb


@triton.jit
def project_adjoint_kernel(grad, w0, w1, w2, bias, x0, x1, x2, dscale, nodes,
                           bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    degree, width, row, tile, position, valid = projection_degree_tile(nodes, bm, bn, True)
    x = tl.where(degree == 0, x0, tl.where(degree == 1, x1, x2))
    w = tl.where(degree == 0, w0, tl.where(degree == 1, w1, w2))
    channel = tile * bn + tl.arange(0, bn)
    k = tl.arange(0, bk)
    a = grad + position[:, None] * 128 + k[None, :]
    b = w + k[:, None] * width + channel[None, :]
    acc = tl.zeros((bm, bn), tl.float32)
    for _ in range(128 // bk):
        inputs = tl.load(a, valid[:, None], 0.)
        weights = tl.load(b)
        acc = tl.dot(inputs, weights, acc, input_precision="ieee")
        a += bk
        b += bk * width
    tl.store(x + row[:, None] * width + channel[None, :], acc, valid[:, None])

    if tile == 0:
        k_all = tl.arange(0, 128)
        g = tl.load(grad + position[:, None] * 128 + k_all[None, :], valid[:, None], 0.)
        bias_values = tl.load(bias + degree * 128 + k_all)
        tl.store(dscale + position, tl.sum(g * bias_values[None, :], axis=1), valid)


def small_degree_input_vjp(grad, weights, bias, *, ampere):
    nodes = grad.shape[0]
    bm, bk = 16, 32
    bn = 64 if ampere else 128 if nodes > 32 else 32
    shape = ((1, 384), (3, 896), (5, 1152))
    terms = tuple(grad.new_empty((nodes, *dims)) for dims in shape)
    dscale = grad.new_empty((nodes, 9))
    grid = (sum(triton.cdiv(nodes * q, bm) * (width // bn) for q, width in shape),)
    project_adjoint_kernel[grid](
        grad, *weights, bias, *terms, dscale, nodes, bm, bn, bk,
        num_warps=4, num_stages=3, enable_fp_fusion=False,
    )
    return terms, dscale


def projection_input_vjp(x, index, rotation, radial, weights, grad, gather_vjp):
    """Differentiate projection and use the caller's gather-layout VJP."""
    x, index, rotation, radial = (tensor.contiguous() for tensor in (x, index, rotation, radial))
    weights = tuple(weight.contiguous() for weight in weights)
    packed = tuple(F.linear(value.resolve_neg().contiguous(), weight)
                   for value, weight in zip(grad, weights, strict=True))
    dx, dw, dr = torch.zeros_like(x), torch.zeros_like(rotation), torch.empty_like(radial)
    gather_vjp(*packed, rotation, radial, x, index, dx, dw, dr, 128, True)
    return dx, dw, dr


AMPERE_GATHER_VJP_SOURCE = r"""#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int channels = 128;
constexpr int radial_width = 1536;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__device__ __forceinline__ void read4(float* values, const float* p, int lane) {
    const float4 v = reinterpret_cast<const float4*>(p)[lane];
    values[0] = v.x;
    values[1] = v.y;
    values[2] = v.z;
    values[3] = v.w;
}

__device__ __forceinline__ void write4(float* p, const float* values, int lane) {
    reinterpret_cast<float4*>(p)[lane] = make_float4(values[0], values[1], values[2], values[3]);
}

template <int Degree>
__device__ __forceinline__ void load_node(float (*values)[4], const float* x, int node, int lane) {
    #pragma unroll
    for (int c = 0; c < 2 * Degree + 1; ++c)
        read4(values[c], x + (node * 9 + Degree * Degree + c) * channels, lane);
}

template <int Degree>
__device__ __forceinline__ void add_node(float* dx, int node, int lane, const float (*values)[4]) {
    #pragma unroll
    for (int c = 0; c < 2 * Degree + 1; ++c) {
        float* row = dx + (node * 9 + Degree * Degree + c) * channels + 4 * lane;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            atomicAdd(row + k, values[c][k]);
    }
}

template <int Degree, int Order, bool Striped>
__device__ __forceinline__ void vjp_order(
    const float* seeds, const float* radial, const float* rot0, const float* rot1, float lane0, float lane1,
    int first, int reverse, int lane,
    const float (*xs)[4], const float (*xt)[4], float (*dxs)[4], float (*dxt)[4],
    float* dr, float* dw0, float* dw1
) {
    constexpr int width = 2 * Degree + 1;
    constexpr int rows = Order ? 2 * (3 - Order) : 3;
    constexpr int radial_group = (Order == 0 ? 0 : (Order == 1 ? 3 : 5)) + Degree - Order;
    constexpr int wbase = Degree == 0 ? 0 : (Degree == 1 ? 1 : 10);
    float rs[4], rt[4], drs[4] = {}, drt[4] = {};
    read4(rs, radial + radial_group * 2 * channels, lane);
    read4(rt, radial + (radial_group * 2 + 1) * channels, lane);
    #pragma unroll
    for (int sign = 0; sign < (Order ? 2 : 1); ++sign) {
        const int row = Degree - Order + sign * (3 - Order);
        const int rid = Order ? 2 * Order - 1 + sign : 0;
        const int start = wbase + rid * width;
        float q0s[4], q0t[4], q1s[4] = {}, q1t[4] = {};
        const float* p0 = seeds + (first * rows + row) * 2 * channels;
        read4(q0s, p0, lane);
        read4(q0t, p0 + channels, lane);
        if (reverse >= 0) {
            const float* p1 = seeds + (reverse * rows + row) * 2 * channels;
            read4(q1s, p1, lane);
            read4(q1t, p1 + channels, lane);
        }
        float g0s[4], g0t[4], g1s[4], g1t[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            g0s[k] = q0s[k] * rs[k];
            g0t[k] = q0t[k] * rt[k];
            g1s[k] = q1s[k] * rs[k];
            g1t[k] = q1t[k] * rt[k];
        }
        #pragma unroll
        for (int c = 0; c < width; ++c) {
            const int col = rid * width + c;
            const float w0 = Striped ? __shfl_sync(0xffffffff, lane0, col)
                                     : __shfl_sync(0xffffffff, lane == 0 ? rot0[start + c] : 0.0f, 0);
            const float w1 = Striped ? __shfl_sync(0xffffffff, lane1, col)
                                     : __shfl_sync(0xffffffff, lane == 0 && reverse >= 0 ? rot1[start + c] : 0.0f, 0);
            float dw0_part = 0.0f, dw1_part = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                dxs[c][k] += w0 * g0s[k] + w1 * g1t[k];
                dxt[c][k] += w0 * g0t[k] + w1 * g1s[k];
                dw0_part += g0s[k] * xs[c][k] + g0t[k] * xt[c][k];
                dw1_part += g1s[k] * xt[c][k] + g1t[k] * xs[c][k];
                drs[k] += (q0s[k] * w0) * xs[c][k] + (q1s[k] * w1) * xt[c][k];
                drt[k] += (q0t[k] * w0) * xt[c][k] + (q1t[k] * w1) * xs[c][k];
            }
            const float da = warp_sum(dw0_part), db = warp_sum(dw1_part);
            if (lane == 0) {
                dw0[start + c] = da;
                if (reverse >= 0) dw1[start + c] = db;
            }
        }
    }
    write4(dr + radial_group * 2 * channels, drs, lane);
    write4(dr + (radial_group * 2 + 1) * channels, drt, lane);
}

template <int Degree, bool Owned, bool Striped>
__global__ void gather_vjp(
    const float* q0, const float* q1, const float* q2,
    const float* x, const int64_t* index, const float* radial, const float* rotation,
    const int* pair_edges, const int* pair_row_ptr, float* dx, float* dr, float* dw,
    int edges, int pairs
) {
    constexpr int width = 2 * Degree + 1;
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
    float xt[width][4], dxt[width][4] = {};
    load_node<Degree>(xt, x, node, lane);

    for (int pair = begin; pair < end; pair += warps * gridDim.y) {
        const int first = pair_edges[pair], reverse = pair_edges[pairs + pair];
        const int source = index[first];
        float xs[width][4], dxs[width][4] = {};
        load_node<Degree>(xs, x, source, lane);
        const float* r = radial + pair * radial_width;
        const float* w0 = rotation + first * 35;
        const float* w1 = reverse >= 0 ? rotation + reverse * 35 : w0;
        float lane0 = 0.0f, lane1 = 0.0f;
        if constexpr (Striped && Degree > 0) {
            constexpr int base = Degree == 1 ? 1 : 10;
            if (lane < width * width) {
                lane0 = w0[base + lane];
                if (reverse >= 0) lane1 = w1[base + lane];
            }
        }
        float* rr = dr + pair * radial_width;
        float* rw0 = dw + first * 35;
        float* rw1 = reverse >= 0 ? dw + reverse * 35 : rw0;
        vjp_order<Degree, 0, Striped && (Degree > 0)>(q0, r, w0, w1, lane0, lane1, first, reverse, lane, xs, xt, dxs, dxt, rr, rw0, rw1);
        if constexpr (Degree > 0)
            vjp_order<Degree, 1, Striped && (Degree > 0)>(q1, r, w0, w1, lane0, lane1, first, reverse, lane, xs, xt, dxs, dxt, rr, rw0, rw1);
        if constexpr (Degree > 1)
            vjp_order<Degree, 2, Striped && (Degree > 0)>(q2, r, w0, w1, lane0, lane1, first, reverse, lane, xs, xt, dxs, dxt, rr, rw0, rw1);
        add_node<Degree>(dx, source, lane, dxs);
    }
    add_node<Degree>(dx, node, lane, dxt);
}
}


extern "C" int launch_gather_strip(
    const float* q0, const float* q1, const float* q2, const float* x,
    const int64_t* index, const float* radial, const float* rotation,
    const int* pair_edges, const int* pair_row_ptr, float* dx, float* dr, float* dw,
    int edges, int pairs, int nodes, cudaStream_t stream
) {
    constexpr int warps = 4;
    constexpr int slots = 2;
    if (pairs == 0) return cudaSuccess;
    const dim3 grid(nodes, slots);
    gather_vjp<0, true, false><<<grid, warps * 32, 0, stream>>>(
        q0, q1, q2, x, index, radial, rotation, pair_edges, pair_row_ptr, dx, dr, dw, edges, pairs);
    gather_vjp<1, true, true><<<grid, warps * 32, 0, stream>>>(
        q0, q1, q2, x, index, radial, rotation, pair_edges, pair_row_ptr, dx, dr, dw, edges, pairs);
    gather_vjp<2, true, true><<<grid, warps * 32, 0, stream>>>(
        q0, q1, q2, x, index, radial, rotation, pair_edges, pair_row_ptr, dx, dr, dw, edges, pairs);
    return cudaPeekAtLastError();
}
"""
