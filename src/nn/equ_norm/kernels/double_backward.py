"""Equivariant normalization double-backward CUDA."""

from operator import attrgetter
from textwrap import indent

import torch

from ....utils.precision import suffix_float32_literals
from .utils import (
    AFFINE_SHARED_LIMIT,
    DOUBLE_BACKWARD_RESIDENT_THREADS,
    FEATURE_THREADS,
    ROW_CACHE_WORDS,
    fast_view_source,
    make_binding,
    make_reduction_prelude,
)


def make_feature_reduce():
    return r"""
__device__ __forceinline__ void feature_reduce(Acc (&values)[FEATURE_K], Acc* shared) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 0; k < FEATURE_K; ++k) {
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            values[k] += __shfl_down_sync(0xffffffff, values[k], delta);
        if (lane == 0)
            shared[k * (FEATURE_THREADS / 32) + threadIdx.x / 32] = values[k];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < FEATURE_K; ++k) {
        Acc total = 0;
        #pragma unroll
        for (int w = 0; w < FEATURE_THREADS / 32; ++w)
            total += shared[k * (FEATURE_THREADS / 32) + w];
        values[k] = total;
    }
    __syncthreads();
}
"""


def make_double_backward_source(plan, dtype):
    group_widths = [0] * plan.num_groups
    starts = []
    for block in plan.blocks:
        starts.append(group_widths[block.group])
        group_widths[block.group] += block.channels * block.components
    width = max(group_widths)
    word = 2 if dtype == torch.float64 else 1
    row_threads = FEATURE_THREADS
    if (width + row_threads - 1) // row_threads * 4 * word > ROW_CACHE_WORDS and (
        (width + 511) // 512 * 4 * word <= ROW_CACHE_WORDS
    ):
        row_threads = 512
    slots = (width + row_threads - 1) // row_threads
    cached = slots * 4 * word <= ROW_CACHE_WORDS
    expanded_slots = [0] * plan.num_groups
    for block, start in zip(plan.blocks, starts):
        stop = start + block.channels * block.components
        expanded_slots[block.group] += (
            stop + row_threads - 1
        ) // row_threads - start // row_threads
    limit_registers = (
        dtype == torch.float32
        and cached
        and row_threads == FEATURE_THREADS
        and max(expanded_slots) * 4 > ROW_CACHE_WORDS
    )
    launch_bounds = (
        f"FEATURE_THREADS, {DOUBLE_BACKWARD_RESIDENT_THREADS} / FEATURE_THREADS"
        if limit_registers
        else "FEATURE_THREADS"
    )
    affine_width = max(b.channels * b.components for b in plan.blocks)
    shared_affine = (
        plan.layout == "ir_mul" and cached and affine_width * 4 * word <= AFFINE_SHARED_LIMIT
    )
    descriptors = {
        b: "{%d,%d,%d,%d,%d,%d,%d,0,%.17e}"
        % (
            b.offset,
            b.channels,
            b.components,
            b.param_offset,
            b.bias_offset,
            b.group,
            b.center_index,
            b.weight,
        )
        for b in plan.blocks
    }
    prefix = make_reduction_prelude(plan, dtype, 2)
    cuda = f"""
constexpr int FEATURE_THREADS = {row_threads}, FEATURE_K = {3 + 3 * plan.num_centers};
""" + make_feature_reduce()
    cuda += r"""
__global__ void zero_parameters(Outputs out) {
    const Index p = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out.v[1] && p < P)
        out.v[1][p] = 0;
    if (out.v[2] && p < BIAS)
        out.v[2][p] = 0;
}
"""
    cuda += f"__global__ __launch_bounds__({launch_bounds}) void double_backward_rows(\n"
    cuda += r"""
    Inputs in, Outputs out, const Acc* fs, int64_t N, int workers) {
    __shared__ Acc reduced[FEATURE_K * (FEATURE_THREADS / 32)];
"""
    if shared_affine:
        cuda += f"""
    __shared__ Acc gamma_shared[{affine_width}];
"""
    cuda += r"""
    const int worker = blockIdx.x / NG, first_group = blockIdx.x % NG;
    const int lane = threadIdx.x & 31;
    for (int64_t group = first_group; group < NG; group += gridDim.x) {
"""
    if cached:
        cuda += f"""
        Acc gamma_acc[{slots}] = {{0}};
"""
    cuda += r"""
        #pragma unroll 1
        for (int64_t row = worker; row < N; row += workers) {
            const Acc r = fs[row * FW + NCENTER + group];
            Acc sums[FEATURE_K] = {0};
"""
    if cached:
        cuda += f"""
            Acc z_cache[{slots}], h_cache[{slots}], g_cache[{slots}];
"""
    anchors = r"""
const Acc ax = b.center_index >= 0 ? in.v[0].get<UnitStride>(row, b.offset) : 0;
const Acc mx = b.center_index >= 0 ? fs[row * FW + b.center_index] : 0;
const Acc ah = b.center_index >= 0 ? in.v[4].get<UnitStride>(row, b.offset) : 0;
"""
    loads = r"""
const Index c = channel(b, j);
const Acc xv = in.v[0].get<UnitStride>(row, b.offset + j), hv = in.v[4].get<UnitStride>(row, b.offset + j);
const Acc z = (b.center_index >= 0 ? (xv - ax) - mx : xv) * r;
const Acc h = (b.center_index >= 0 ? hv - ah : hv) * r;
const Acc g = in.v[3].get<UnitStride>(row, b.offset + j);
const Acc ga = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0), rg = in.v[5].parameter<UnitStride>(b.param_offset + c);
"""
    for b, start in zip(plan.blocks, starts):
        cuda += f"""
            do {{
                constexpr Block b = {descriptors[b]};
                if (b.group != group)
                    break;
{indent(anchors, '                ')}
"""
        inner = loads + r"""
sums[0] += in.v[4].data ? g * ga * z : 0;
sums[1] += b.weight * z * h;
sums[2] += h * g * ga + g * rg * z;
"""
        if b.center_index >= 0:
            ci = b.center_index
            inner += f"""
sums[{3 + 3 * ci}] += h;
sums[{4 + 3 * ci}] += in.v[4].data ? g * ga : 0;
sums[{5 + 3 * ci}] += g * rg;
"""
        if cached:
            first = start // row_threads
            end = (start + b.channels * b.components + row_threads - 1) // row_threads
            for slot in range(first, end):
                cuda += f"""
                {{
                    const Index j = Index(threadIdx.x) + Index({slot * row_threads - start});
                    if (j >= 0 && j < b.channels * b.components) {{
{indent(inner, '                        ')}
                        z_cache[{slot}] = z;
                        h_cache[{slot}] = h;
                        g_cache[{slot}] = g;
                    }}
                }}
"""
        else:
            cuda += f"""
                #pragma unroll 4
                for (Index chunk = 0; chunk < 1 + (b.channels * b.components - 1) / FEATURE_THREADS; ++chunk) {{
                    const Index j = Index(threadIdx.x) + chunk * FEATURE_THREADS;
                    if (j < b.channels * b.components) {{
{indent(inner, '                        ')}
                    }}
                }}
"""
        cuda += """
            } while (false);
"""
    cuda += r"""
            feature_reduce(sums, reduced);
            const Acc S = sums[0], C = sums[1];
            Acc T = sums[2];
"""
    for b in sorted(
        (b for b in plan.blocks if b.center_index >= 0), key=attrgetter("center_index")
    ):
        ci = b.center_index
        cuda += f"""
            const Acc hm{ci} = sums[{3 + 3 * ci}] / {b.channels}.0;
            T -= hm{ci} * sums[{4 + 3 * ci}];
            const Acc pm{ci} = r * (sums[{5 + 3 * ci}] / {b.channels}.0 - C * sums[{4 + 3 * ci}] / {b.channels}.0);
"""
    cuda += r"""
            const Acc Q = T - 3.0 * S * C;
"""
    for b, start in zip(plan.blocks, starts):
        hm = f"hm{b.center_index}" if b.center_index >= 0 else "0.0"
        pm = f"pm{b.center_index}" if b.center_index >= 0 else "0.0"
        rb = "in.v[6].parameter<UnitStride>(b.bias_offset + c)" if b.bias_offset >= 0 else "0.0"
        dx = (
            "0.0"
            if b.center_index >= 0 and b.channels == 1
            else f"r * (g * A - h * wh - z * wz) - {pm}"
        )
        output = f"""
const Acc A = rg - ga * C;
const Acc wh = b.weight * S, wz = b.weight * Q;
if (out.v[0])
    out.v[0][row * D + b.offset + j] = {dx};
if (out.v[3])
    out.v[3][row * D + b.offset + j] = ga * h + A * z + {rb};
"""
        cuda += f"""
            do {{
                constexpr Block b = {descriptors[b]};
                if (b.group != group)
                    break;
"""
        if cached:
            first = start // row_threads
            end = (start + b.channels * b.components + row_threads - 1) // row_threads
            for slot in range(first, end):
                inner = f"""
const Index c = channel(b, j);
const Acc z = z_cache[{slot}], h = h_cache[{slot}] - {hm}, g = g_cache[{slot}];
const Acc ga = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0), rg = in.v[5].parameter<UnitStride>(b.param_offset + c);
if (out.v[1])
    gamma_acc[{slot}] += g * (h - z * C);
""" + output
                cuda += f"""
                {{
                    const Index j = Index(threadIdx.x) + Index({slot * row_threads - start});
                    if (j >= 0 && j < b.channels * b.components) {{
{indent(inner, '                        ')}
                    }}
                }}
"""
        else:
            cuda += indent(anchors, "                ")
            inner = (
                loads.replace("const Acc h =", "const Acc raw_h =")
                + f"const Acc h = raw_h - {hm};\n"
                + output
            )
            cuda += f"""
                #pragma unroll 4
                for (Index chunk = 0; chunk < 1 + (b.channels * b.components - 1) / FEATURE_THREADS; ++chunk) {{
                    const Index j = Index(threadIdx.x) + chunk * FEATURE_THREADS;
                    const Index c = channel(b, j);
                    Acc gp = 0;
                    if (j < b.channels * b.components) {{
{indent(inner, '                        ')}
                        gp = g * (h - z * C);
                    }}
                    if (out.v[1]) {{
"""
            if plan.layout == "mul_ir":
                cuda += r"""
                        #pragma unroll
                        for (int delta = 1; delta < min(32, b.components); delta <<= 1) {
                            const Acc other = __shfl_down_sync(0xffffffff, gp, delta);
                            const Index other_channel = __shfl_down_sync(0xffffffff, c, delta);
                            if (lane + delta < 32 && other_channel == c)
                                gp += other;
                        }
                        if (j < b.channels * b.components && (lane == 0 || j % b.components == 0))
                            atomicAdd(out.v[1] + b.param_offset + c, gp);
"""
            else:
                cuda += r"""
                        if (j < b.channels * b.components)
                            atomicAdd(out.v[1] + b.param_offset + c, gp);
"""
            cuda += """
                    }
                }
"""
        cuda += """
            } while (false);
"""
    cuda += """
        }
"""
    if cached:
        cuda += """
        if (out.v[1]) {
"""
        for b, start in zip(plan.blocks, starts):
            cuda += f"""
            do {{
                constexpr Block b = {descriptors[b]};
                if (b.group != group)
                    break;
"""
            if shared_affine:
                first = start // row_threads
                end = (start + b.channels * b.components + row_threads - 1) // row_threads
                for slot in range(first, end):
                    cuda += f"""
                {{
                    const Index j = Index(threadIdx.x) + Index({slot * row_threads - start});
                    if (j >= 0 && j < b.channels * b.components)
                        gamma_shared[j] = gamma_acc[{slot}];
                }}
"""
                cuda += r"""
                __syncthreads();
                for (Index c = threadIdx.x; c < b.channels; c += FEATURE_THREADS) {
                    Acc gp = 0;
                    #pragma unroll 4
                    for (Index m = 0; m < b.components; ++m)
                        gp += gamma_shared[local_index(b, c, m)];
                    atomicAdd(out.v[1] + b.param_offset + c, gp);
                }
                __syncthreads();
            } while (false);
"""
                continue
            first = start // row_threads
            end = (start + b.channels * b.components + row_threads - 1) // row_threads
            for slot in range(first, end):
                cuda += f"""
                {{
                    const Index j = Index(threadIdx.x) + Index({slot * row_threads - start});
                    const Index c = channel(b, j);
                    Acc gp = j >= 0 && j < b.channels * b.components ? gamma_acc[{slot}] : 0;
"""
                if plan.layout == "mul_ir":
                    cuda += r"""
                    #pragma unroll
                    for (int delta = 1; delta < min(32, b.components); delta <<= 1) {
                        const Acc other = __shfl_down_sync(0xffffffff, gp, delta);
                        const Index other_channel = __shfl_down_sync(0xffffffff, c, delta);
                        if (lane + delta < 32 && other_channel == c)
                            gp += other;
                    }
                    if (j >= 0 && j < b.channels * b.components && (lane == 0 || j % b.components == 0))
                        atomicAdd(out.v[1] + b.param_offset + c, gp);
"""
                else:
                    cuda += r"""
                    if (j >= 0 && j < b.channels * b.components)
                        atomicAdd(out.v[1] + b.param_offset + c, gp);
"""
                cuda += """
                }
"""
            cuda += """
            } while (false);
"""
        cuda += """
        }
"""
    cuda += r"""
    }
}
int row_capacity() {
    static thread_local int cached_device = -1, cached_capacity = 0;
    int device = 0;
    cudaGetDevice(&device);
    if (device != cached_device) {
        int blocks = 0, multiprocessors = 0;
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
        const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, double_backward_rows, FEATURE_THREADS, 0);
        if (error != cudaSuccess || blocks <= 0)
            throw std::runtime_error("double-backward row capacity unavailable");
        cached_device = device;
        cached_capacity = blocks * multiprocessors;
    }
    return cached_capacity;
}
void launch(Inputs in, Outputs out, const Acc* fs, Acc* stats, int64_t N, cudaStream_t stream) {
    if (out.v[1] || out.v[2])
        zero_parameters<<<(int64_t(max(P, BIAS)) + 127) / 128, 128, 0, stream>>>(out);
    const int capacity = row_capacity();
    const int workers = int(min(int64_t(max(1, capacity / NG)), N));
    const int blocks = int(min(int64_t(capacity), int64_t(workers) * NG));
    double_backward_rows<<<blocks, FEATURE_THREADS, 0, stream>>>(in, out, fs, N, workers);
}
"""
    if max(plan.dim, plan.param_dim, plan.bias_dim) > 2147483647:
        cuda = cuda.replace("delta < min(32, b.components)", "delta < 32 && delta < b.components")
    if dtype == torch.float32:
        cuda = suffix_float32_literals(cuda)
    cpp = make_binding(
        dtype,
        r"""
int64_t workspace_size(int64_t n, bool parameters = true) {
    return 0;
}
""",
        stats_width=0,
        parameter_outputs=(1, 2),
    )
    return fast_view_source(
        {"cpp": cpp, "body": cuda, "prefix": prefix, "includes": ("#include <stdexcept>",)},
        plan,
        dtype,
        active_inputs=(0, 1, 3, 4, 5, 6),
    )
