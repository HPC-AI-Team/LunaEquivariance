"""Equivariant normalization forward CUDA."""

from textwrap import indent

import torch

from ....utils.precision import suffix_float32_literals
from .utils import (
    AFFINE_SHARED_LIMIT,
    FORWARD_MAX_ROW_THREADS,
    FORWARD_PACK_THREADS,
    ROW_CACHE_WORDS,
    fast_view_source,
    make_binding,
    make_reduction_prelude,
)


def make_forward_source(plan, dtype):
    word = 2 if dtype == torch.float64 else 1
    for lanes in (32, 64, 128, 256, FORWARD_MAX_ROW_THREADS):
        prefix = make_reduction_prelude(
            plan, dtype, 0, row_cap=lanes, row_cache_budget=ROW_CACHE_WORDS
        )
        if prefix["resources"]["cache_enabled"]:
            break
    resource = prefix["resources"]
    lanes = resource["row_threads"]
    threads = max(lanes, 128, min(FORWARD_PACK_THREADS, 4 * lanes))
    cached = resource["cache_enabled"]
    descriptors = resource["descriptors"]
    affine = ""
    if (plan.param_dim + plan.bias_dim) * 4 * word <= AFFINE_SHARED_LIMIT:
        affine = f"""
    __shared__ Element gamma[{max(1, plan.param_dim)}];
    __shared__ Element beta[{max(1, plan.bias_dim)}];
    if (in.v[1].data)
        for (Index p = threadIdx.x; p < P; p += FORWARD_THREADS)
            gamma[p] = Element(in.v[1].parameter<UnitStride>(p));
    if (in.v[2].data)
        for (Index p = threadIdx.x; p < BIAS; p += FORWARD_THREADS)
            beta[p] = Element(in.v[2].parameter<UnitStride>(p));
    __syncthreads();
    if (in.v[1].data) {{
        in.v[1].data = gamma;
        in.v[1].s1 = 1;
    }}
    if (in.v[2].data) {{
        in.v[2].data = beta;
        in.v[2].s1 = 1;
    }}
"""
    load, center, resquare, maximum, scaled, output = [], [], [], [], [], []
    unroll = "#pragma unroll" if cached else "#pragma unroll 4"
    for block, descriptor in zip(plan.blocks, descriptors):
        size = block.channels * block.components
        begin = f"""
    {{
        constexpr Block b = {descriptor};
        if (b.group == group) {{
            {unroll}
            for (int slot = 0; slot < ({size} + ROW_THREADS - 1) / ROW_THREADS; ++slot) {{
                const Index j = Index(lane) + Index(slot) * ROW_THREADS;
                if (j >= {size})
                    continue;
"""
        end = """
            }
        }
    }
"""
        centered = (
            "(in.v[0].get<UnitStride>(row, b.offset + j) - anchor - sums[b.center_index + 1])"
            if block.center_index >= 0
            else "in.v[0].get<UnitStride>(row, b.offset + j)"
        )
        anchor = (
            f"    const Acc anchor{block.center_index} = group == {block.group} ? in.v[0].get<UnitStride>(row, {block.offset}) : 0;\n"
            if block.center_index >= 0
            else ""
        )
        load.append(
            anchor
            + begin
            + (
                f"                const Acc u = in.v[0].get<UnitStride>(row, b.offset + j) - anchor{block.center_index};\n"
                if block.center_index >= 0
                else "                const Acc u = in.v[0].get<UnitStride>(row, b.offset + j);\n"
            )
        )
        if cached:
            load[-1] += "                cache[b.cache_offset + slot] = u;\n"
        load[-1] += "                sums[0] += b.weight * u * u;\n"
        if block.center_index >= 0:
            load[-1] += "                sums[b.center_index + 1] += u;\n"
        load[-1] += end
        if block.center_index >= 0:
            center.append(f"""
    if (group == {block.group}) {{
        const Acc sum = sums[{block.center_index + 1}];
        sums[{block.center_index + 1}] /= {block.channels};
        q -= Acc({block.weight:.17e}) * sums[{block.center_index + 1}] * sum;
        if (lane == 0)
            stats[row * FW + {block.center_index}] = sums[{block.center_index + 1}];
    }}
""")
            if cached:
                center[-1] += (
                    begin
                    + "                cache[b.cache_offset + slot] -= sums[b.center_index + 1];\n"
                    + end
                )
        if cached:
            value = "cache[b.cache_offset + slot]"
        else:
            value = centered.replace("anchor", f"anchor{block.center_index}")
        resquare.append(
            begin
            + f"                const Acc u = {value};\n                stable[0] += b.weight * u * u;\n"
            + end
        )
        maximum.append(
            begin + f"                magnitude[0] = fmax(magnitude[0], fabs({value}));\n" + end
        )
        scaled.append(
            begin
            + f"                const Acc u = ({value}) / magnitude[0];\n                stable[0] += b.weight * u * u;\n"
            + end
        )
        output.append(begin + f"""
                const Index c = channel(b, j);
                const Acc gamma = in.v[1].parameter<UnitStride>(b.param_offset + c, 1);
                const Acc beta = b.bias_offset >= 0 ? in.v[2].parameter<UnitStride>(b.bias_offset + c) : 0;
                out.v[0][row * D + b.offset + j] = Element(gamma * ({value} * inv) + beta);
""" + end)

    cuda = f"""
constexpr int FORWARD_THREADS = {threads}, ROWS_PER_BLOCK = {threads // lanes};

template <int K, bool Maximum = false>
__device__ __forceinline__ void row_reduce(Acc (&values)[K]) {{
    constexpr int WARPS = ROW_THREADS / 32, STRIDE = WARPS + 1;
    __shared__ Acc storage[ROWS_PER_BLOCK * K * STRIDE];
    const int row_group = threadIdx.x / ROW_THREADS;
    const int lane = threadIdx.x & 31, warp = (threadIdx.x % ROW_THREADS) / 32;
    Acc* shared = storage + row_group * K * STRIDE;
    #pragma unroll
    for (int k = 0; k < K; ++k) {{
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {{
            const Acc other = __shfl_down_sync(0xffffffff, values[k], delta);
            values[k] = Maximum ? fmax(values[k], other) : values[k] + other;
        }}
        if constexpr (WARPS == 1)
            values[k] = __shfl_sync(0xffffffff, values[k], 0);
        else if (lane == 0)
            shared[k * STRIDE + warp] = values[k];
    }}
    if constexpr (WARPS > 1) {{
        asm volatile("bar.sync %0, %1;" ::"r"(row_group + 1), "n"(ROW_THREADS) : "memory");
        if (warp == 0) {{
            #pragma unroll
            for (int k = 0; k < K; ++k) {{
                Acc sum = lane < WARPS ? shared[k * STRIDE + lane] : 0;
                #pragma unroll
                for (int delta = WARPS / 2; delta > 0; delta >>= 1) {{
                    const Acc other = __shfl_down_sync(0xffffffff, sum, delta);
                    sum = Maximum ? fmax(sum, other) : sum + other;
                }}
                if (lane == 0)
                    shared[k * STRIDE + WARPS] = sum;
            }}
        }}
        asm volatile("bar.sync %0, %1;" ::"r"(row_group + 1), "n"(ROW_THREADS) : "memory");
        #pragma unroll
        for (int k = 0; k < K; ++k)
            values[k] = shared[k * STRIDE + WARPS];
    }}
}}

__global__ __launch_bounds__(FORWARD_THREADS) void rows(
    Inputs in, Outputs out, Acc* stats, int64_t N, int64_t row_start
) {{
{affine}
    const int lane = threadIdx.x % ROW_THREADS, group = blockIdx.y;
    const int64_t row = row_start + int64_t(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.x / ROW_THREADS;
    if (row >= N)
        return;
"""
    if cached:
        cuda += f"    Acc cache[{resource['cache_slots']}];\n"
    cuda += f"""
    Acc sums[{plan.num_centers + 1}] = {{}};
{''.join(load)}
    row_reduce<{plan.num_centers + 1}>(sums);
    const Acc raw = sums[0];
    Acc q = raw;
{''.join(center)}
"""
    if plan.num_centers:
        cuda += f"""
    if (q <= Acc(0.02) * raw || !isfinite(q)) {{
        Acc stable[1] = {{0}};
{indent(''.join(resquare), '    ')}
        row_reduce<1>(stable);
        q = stable[0];
    }}
"""
    cuda += f"""
    Acc inv = inverse_sqrt(q + Acc({plan.eps:.17e}));
    if (!isfinite(q + Acc({plan.eps:.17e}))) {{
        Acc magnitude[1] = {{0}};
{indent(''.join(maximum), '    ')}
        row_reduce<1, true>(magnitude);
        Acc stable[1] = {{0}};
{indent(''.join(scaled), '    ')}
        row_reduce<1>(stable);
        inv = (Acc(1) / magnitude[0]) * inverse_sqrt(stable[0] + (Acc({plan.eps:.17e}) / magnitude[0]) / magnitude[0]);
    }}
    if (lane == 0)
        stats[row * FW + NCENTER + group] = inv;
{''.join(output)}
}}

void launch(Inputs in, Outputs out, const Acc*, Acc* stats, int64_t N, cudaStream_t stream) {{
    const int64_t blocks = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    for (int64_t start = 0; start < blocks;) {{
        const int64_t count = min(blocks - start, int64_t(2147483647));
        rows<<<dim3(count, NG), FORWARD_THREADS, 0, stream>>>(in, out, stats, N, start * ROWS_PER_BLOCK);
        start += count;
    }}
}}
"""
    if dtype == torch.float32:
        cuda = suffix_float32_literals(cuda)
    host = make_binding(
        dtype,
        f"""
int64_t workspace_size(int64_t n, bool = true) {{
    return n * {plan.num_centers + plan.num_groups};
}}
""",
        stats_width=plan.num_centers + plan.num_groups,
    )
    return fast_view_source(
        {"cpp": host, "body": cuda, "prefix": prefix}, plan, dtype, active_inputs=(0, 1, 2)
    )
