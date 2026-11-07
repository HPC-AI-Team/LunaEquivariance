"""Equivariant normalization backward CUDA."""

import torch

from ....utils.precision import suffix_float32_literals
from .utils import (
    BACKWARD_CACHE_WORDS,
    BACKWARD_COMPONENTS_PER_LANE,
    BACKWARD_MAX_COMPONENT_LANES,
    FEATURE_THREADS,
    color_segments,
    fast_view_source,
    make_binding,
    make_reduction_prelude,
)


def make_backward_source(plan, dtype):
    """Fuse feature and affine gradients with bounded copy reductions."""
    block_descriptors = {
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
    common_source = make_reduction_prelude(plan, dtype, 1)

    # Native Element atomics need no global workspace.
    word = 2 if dtype == torch.float64 else 1
    group_dims = [
        sum(b.channels * b.components for b in plan.blocks if b.group == group)
        for group in range(plan.num_groups)
    ]
    component_lanes = 1
    cooperative_lanes = 1
    if plan.layout == "mul_ir":
        component_features = {}
        for block in plan.blocks:
            key = (block.group, block.components)
            component_features[key] = (
                component_features.get(key, 0) + block.channels * block.components
            )
        # Keep component cooperation for long irreps that dominate a group.
        dominant_components = max(
            (features * components + group_dims[group] - 1) // group_dims[group]
            for (group, components), features in component_features.items()
        )
        mean_components = (plan.dim + plan.param_dim - 1) // plan.param_dim
        affine_channels = plan.param_dim + plan.bias_dim
        packed_components = (plan.dim + affine_channels - 1) // affine_channels
        components = min(mean_components, max(packed_components, dominant_components))
        budget = BACKWARD_COMPONENTS_PER_LANE * word
        cooperative_work = (mean_components + budget - 1) // budget
        cooperative_lanes = min(
            BACKWARD_MAX_COMPONENT_LANES,
            1 << (cooperative_work - 1).bit_length(),
        )
        component_work = (components + budget - 1) // budget
        component_lanes = min(BACKWARD_MAX_COMPONENT_LANES, 1 << (component_work - 1).bit_length())
    gang = FEATURE_THREADS
    group_params = [
        sum(b.channels for b in plan.blocks if b.group == group) for group in range(plan.num_groups)
    ]
    feature_cache_slots = (max(group_dims) + gang - 1) // gang
    parameter_cache_slots = (max(group_params) + gang - 1) // gang
    bias_cache_slots = 0
    if word * (2 * feature_cache_slots + 2 * parameter_cache_slots) <= BACKWARD_CACHE_WORDS:
        for lanes in sorted({component_lanes, cooperative_lanes}):
            for candidate in (32, 64, 128, FEATURE_THREADS):
                owners, counts = color_segments(plan, candidate // lanes, lanes)
                cached = word * (2 * counts[0] + 2 * counts[1] + counts[2]) <= BACKWARD_CACHE_WORDS
                if cached:
                    gang, component_lanes = candidate, lanes
                    break
            if cached:
                break
        owners, counts = color_segments(plan, gang // component_lanes, component_lanes)
        feature_cache_slots, parameter_cache_slots, bias_cache_slots = counts
    if (
        word * (2 * feature_cache_slots + 2 * parameter_cache_slots + bias_cache_slots)
        > BACKWARD_CACHE_WORDS
    ):
        cpp = make_binding(
            dtype,
            r"""
int64_t workspace_size(int64_t n, bool parameters = true);
""",
            stats_width=0,
            parameter_outputs=(1, 2),
        )
        parameter_offsets, bias_offsets = [], []
        parameter_slots, bias_slots = [0] * plan.num_groups, [0] * plan.num_groups
        for block in plan.blocks:
            parameter_offsets.append(parameter_slots[block.group])
            bias_offsets.append(bias_slots[block.group])
            parameter_slots[block.group] += (block.channels + 127) // 128
            if block.bias_offset >= 0:
                bias_slots[block.group] += (block.channels + 127) // 128
        accumulate = word * (max(parameter_slots) + max(bias_slots)) <= 64
        cuda = ""
        cuda += r"""
__device__ __forceinline__ Acc backward_stream_sum(Acc value) {
    __shared__ Acc shared[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        value += __shfl_down_sync(0xffffffff, value, d);
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    value = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w)
        value += shared[w];
    __syncthreads();
    return value;
}
__global__ void backward_stream_zero(Outputs out) {
    const Index p = Index(blockIdx.x) * 128 + threadIdx.x;
    if (out.v[1] && p < P)
        out.v[1][p] = 0;
    if (out.v[2] && p < BIAS)
        out.v[2][p] = 0;
}
__global__ __launch_bounds__(128) void backward_stream_rows(Inputs in, Outputs out, const Acc* fs, int64_t N) {
    const int group = blockIdx.y, tid = threadIdx.x;
"""
        if accumulate:
            cuda += f"""
    Acc gamma_acc[{max(parameter_slots)}] = {{0}};
    Acc beta_acc[{max(1, max(bias_slots))}] = {{0}};
"""
        cuda += r"""
    #pragma unroll 1
    for (int64_t row = blockIdx.x; row < N; row += gridDim.x) {
        const Acc r = fs[row * FW + NCENTER + group];
        Acc H = 0;
"""
        for bi, block in enumerate(plan.blocks):
            cuda += f"""
        do {{
            constexpr Block b = {block_descriptors[block]};
            if (b.group != group)
                break;

            const Acc anchor = b.center_index >= 0 ? in.v[0].get<UnitStride>(row, b.offset) : 0;
            const Acc mean = b.center_index >= 0 ? fs[row * FW + b.center_index] : 0;
            for (Index c = tid; c < b.channels; c += 128) {{
                const Acc gamma = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0);
                Acc gp = 0, gb = 0;
                #pragma unroll 4
                for (Index m = 0; m < b.components; ++m) {{
                    const Index j = b.offset + local_index(b, c, m);
                    const Acc x = in.v[0].get<UnitStride>(row, j), g = in.v[3].get<UnitStride>(row, j);
                    const Acc z = (b.center_index >= 0 ? (x - anchor) - mean : x) * r;
                    gp += g * z;
                    if (m == 0)
                        gb = g;
                }}
                H += gamma * gp;
"""
            cuda += (
                f"""
                if (out.v[1])
                    gamma_acc[{parameter_offsets[bi]} + c / 128] += gp;
"""
                if accumulate
                else r"""
                if (out.v[1])
                    atomicAdd(out.v[1] + b.param_offset + c, Element(gp));
"""
            )
            if block.bias_offset >= 0:
                cuda += (
                    f"""
                if (out.v[2])
                    beta_acc[{bias_offsets[bi]} + c / 128] += gb;
"""
                    if accumulate
                    else r"""
                if (out.v[2])
                    atomicAdd(out.v[2] + b.bias_offset + c, Element(gb));
"""
                )
            cuda += """
            }
        } while (false);
"""
        cuda += r"""
        H = backward_stream_sum(H);
        if (out.v[0]) {
"""
        for block in plan.blocks:
            cuda += f"""
            do {{
                constexpr Block b = {block_descriptors[block]};
                if (b.group != group)
                    break;

                const Acc anchor = b.center_index >= 0 ? in.v[0].get<UnitStride>(row, b.offset) : 0;
                const Acc mean = b.center_index >= 0 ? fs[row * FW + b.center_index] : 0;
"""
            if block.center_index >= 0 and block.channels == 1:
                # The centering projection on a one-channel scalar is exactly zero.
                cuda += r"""
                if (tid == 0)
                    out.v[0][row * D + b.offset] = Element(0);
"""
            elif block.center_index >= 0:
                cuda += r"""
                Acc projection = 0;
                for (Index c = tid; c < b.channels; c += 128) {
                    const Acc gamma = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0);
                    const Acc x = in.v[0].get<UnitStride>(row, b.offset + c), g = in.v[3].get<UnitStride>(row, b.offset + c);
                    const Acc z = ((x - anchor) - mean) * r;
                    projection += r * (gamma * g - b.weight * z * H);
                }
                projection = backward_stream_sum(projection) / b.channels;
                for (Index c = tid; c < b.channels; c += 128) {
                    const Acc gamma = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0);
                    const Acc x = in.v[0].get<UnitStride>(row, b.offset + c), g = in.v[3].get<UnitStride>(row, b.offset + c);
                    const Acc z = ((x - anchor) - mean) * r;
                    out.v[0][row * D + b.offset + c] = Element(r * (gamma * g - b.weight * z * H) - projection);
                }
"""
            else:
                cuda += r"""
                for (Index c = tid; c < b.channels; c += 128) {
                    const Acc gamma = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0);
                    #pragma unroll 4
                    for (Index m = 0; m < b.components; ++m) {
                        const Index j = b.offset + local_index(b, c, m);
                        const Acc x = in.v[0].get<UnitStride>(row, j), g = in.v[3].get<UnitStride>(row, j), z = x * r;
                        out.v[0][row * D + j] = Element(r * (gamma * g - b.weight * z * H));
                    }
                }
"""
            cuda += """
            } while (false);
"""
        cuda += r"""
        }
    }
"""
        if accumulate:
            for bi, block in enumerate(plan.blocks):
                cuda += f"""
    if (group == {block.group}) {{
        for (Index c = tid; c < {block.channels}; c += 128) {{
            if (out.v[1])
                atomicAdd(out.v[1] + {block.param_offset} + c, Element(gamma_acc[{parameter_offsets[bi]} + c / 128]));
"""
                if block.bias_offset >= 0:
                    cuda += f"""
            if (out.v[2])
                atomicAdd(out.v[2] + {block.bias_offset} + c, Element(beta_acc[{bias_offsets[bi]} + c / 128]));
"""
                cuda += """
        }
    }
"""
        cuda += r"""
}
int streaming_capacity() {
    static thread_local int cached_device = -1, capacity = 0;
    int device = 0;
    cudaGetDevice(&device);
    if (device != cached_device) {
        int multiprocessors = 0, blocks = 0;
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, backward_stream_rows, 128, 0);
        cached_device = device;
        capacity = max(1, blocks * multiprocessors);
    }
    return capacity;
}
int64_t workspace_size(int64_t, bool) {
    return 0;
}
void launch(Inputs in, Outputs out, const Acc* fs, Acc*, int64_t N, cudaStream_t stream) {
    if (out.v[1] || out.v[2])
        backward_stream_zero<<<(int64_t(max(P, BIAS)) + 127) / 128, 128, 0, stream>>>(out);
    if (N > 0)
        backward_stream_rows<<<dim3(int(min(N, int64_t(max(1, streaming_capacity() / NG)))), NG), 128, 0, stream>>>(in, out, fs, N);
}
"""
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        return fast_view_source(
            {"cpp": cpp, "body": cuda, "prefix": common_source, "cuda_workspace": True},
            plan,
            dtype,
            active_inputs=(0, 1, 3),
        )

    bias_cache_slots = max(1, bias_cache_slots)
    threads = max(128, gang)
    cpp = make_binding(
        dtype,
        r"""
int64_t workspace_size(int64_t n, bool parameters = true);
""",
        stats_width=0,
        parameter_outputs=(1, 2),
    )
    cuda = ""
    cuda += "\n\n\n"
    cuda += f"""
constexpr int OWN_THREADS = {threads}, GANG = {gang}, GANGS = {threads // gang};
constexpr int COMPONENT_LANES = {component_lanes};
constexpr int DATA_SLOTS = {feature_cache_slots}, PARAM_SLOTS = {parameter_cache_slots}, BETA_SLOTS = {bias_cache_slots};
"""
    cuda += r"""
__device__ __forceinline__ Acc gang_sum(Acc value, Acc* shared) {
    const int lane = threadIdx.x & 31, local_lane = threadIdx.x % GANG;
    const int gang_id = threadIdx.x / GANG;
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        value += __shfl_down_sync(0xffffffff, value, d);
    if constexpr (GANG == 32)
        return __shfl_sync(0xffffffff, value, 0);
    else {
        if (lane == 0)
            shared[threadIdx.x / 32] = value;
        asm volatile("bar.sync %0, %1;" ::"r"(gang_id + 1), "n"(GANG) : "memory");
        value = 0;
        #pragma unroll
        for (int w = 0; w < GANG / 32; ++w)
            value += shared[gang_id * (GANG / 32) + w];
        asm volatile("bar.sync %0, %1;" ::"r"(gang_id + 1), "n"(GANG) : "memory");
        return value;
    }
}
"""
    cuda += r"""
__global__ __launch_bounds__(OWN_THREADS) void backward_channel_colored(
    Inputs in, Outputs out, const Acc* fs, Acc* partials, int64_t N, int workers) {
    __shared__ Acc row_shared[OWN_THREADS / 32];
    __shared__ Acc parameter_shared[GANGS * PARAM_SLOTS * GANG];
    __shared__ Acc beta_shared[GANGS * BETA_SLOTS * GANG];
    const int worker = blockIdx.x / NG, first_group = blockIdx.x % NG;
    const int gang_id = threadIdx.x / GANG, lane = threadIdx.x % GANG;
    const int component_lane = lane % COMPONENT_LANES, channel_lane = lane / COMPONENT_LANES;
    for (int group = first_group; group < NG; group += gridDim.x) {
        Acc z_cache[DATA_SLOTS], g_cache[DATA_SLOTS], gamma_cache[PARAM_SLOTS];
        Acc gamma_acc[PARAM_SLOTS] = {0}, beta_acc[BETA_SLOTS] = {0};
"""
    for o in owners:
        cuda += f"""
        do {{
            constexpr Block b = {block_descriptors[o['block']]};
            if (b.group != group)
                break;
            const Index c = Index(channel_lane) + ({o['channel_shift']});
            const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
            if (owns)
                gamma_cache[{o['param']}] = in.v[1].parameter<UnitStride>(b.param_offset + c, 1.0);
        }} while (false);
"""
    cuda += r"""
        for (int64_t start = int64_t(worker) * GANGS; start < N; start += int64_t(workers) * GANGS) {
            const int64_t row = start + gang_id;
            const bool active = row < N;
            const Acc r = active ? fs[row * FW + NCENTER + group] : 0;
            Acc H = 0;
"""
    for o in owners:
        b, data_slot, parameter_slot, bias_slot = o["block"], o["data"], o["param"], o["bias"]
        cuda += f"""
            do {{
                constexpr Block b = {block_descriptors[o['block']]};
                if (b.group != group)
                    break;
                const Index c = Index(channel_lane) + ({o['channel_shift']});
                const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
                if (!owns)
                    break;
                const Acc anchor = active && b.center_index >= 0 ? in.v[0].get<UnitStride>(row, b.offset) : 0;
                const Acc mean_delta = active && b.center_index >= 0 ? fs[row * FW + b.center_index] : 0;
                Acc gp = 0;
"""
        cuda += f"""
                #pragma unroll
                for (Index m = component_lane; m < b.components; m += COMPONENT_LANES) {{
                    const Index j = local_index(b, c, m);
                    const Acc u = active ? in.v[0].get<UnitStride>(row, b.offset + j) : 0, g = active ? in.v[3].get<UnitStride>(row, b.offset + j) : 0;
                    const Acc z = (b.center_index >= 0 ? (u - anchor) - mean_delta : u) * r;
                    z_cache[{data_slot} + m / COMPONENT_LANES] = z;
                    g_cache[{data_slot} + m / COMPONENT_LANES] = g;
                    gp += g * z;
                }}
                H += gamma_cache[{parameter_slot}] * gp;
                if (out.v[1])
                    gamma_acc[{parameter_slot}] += gp;
"""
        if b.bias_offset >= 0:
            cuda += f"""
                if (out.v[2] && component_lane == 0)
                    beta_acc[{bias_slot}] += g_cache[{data_slot}];
"""
        cuda += """
            } while (false);
"""
    cuda += r"""
            H = gang_sum(H, row_shared);
            if (out.v[0]) {
"""
    for b in plan.blocks:
        own = [o for o in owners if o["block"] is b]
        if b.center_index >= 0:
            cuda += r"""
                Acc projection%d = 0;
""" % b.center_index
        for o in own:
            data_slot, parameter_slot = o["data"], o["param"]
            cuda += f"""
                do {{
                    constexpr Block b = {block_descriptors[o['block']]};
                    if (b.group != group)
                        break;
                    const Index c = Index(channel_lane) + ({o['channel_shift']});
                    const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
                    if (!owns)
                        break;
                    #pragma unroll
                    for (Index m = component_lane; m < b.components; m += COMPONENT_LANES) {{
                        const Acc dx = r * (g_cache[{data_slot} + m / COMPONENT_LANES] * gamma_cache[{parameter_slot}] - b.weight * z_cache[{data_slot} + m / COMPONENT_LANES] * H);
"""
            if b.center_index >= 0:
                cuda += f"""
                        z_cache[{data_slot} + m / COMPONENT_LANES] = dx;
                        projection{b.center_index} += dx;
"""
            else:
                cuda += r"""
                        if (active)
                            out.v[0][row * D + b.offset + local_index(b, c, m)] = dx;
"""
            cuda += """
                    }
                } while (false);
"""
        if b.center_index >= 0:
            cuda += f"""
                projection{b.center_index} = gang_sum(projection{b.center_index}, row_shared) / {b.channels};
"""
            for o in own:
                cuda += f"""
                do {{
                    constexpr Block b = {block_descriptors[o['block']]};
                    if (b.group != group)
                        break;
                    const Index c = Index(channel_lane) + ({o['channel_shift']});
                    const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
                    if (owns && active && component_lane == 0)
                        out.v[0][row * D + b.offset + c] = z_cache[{o['data']}] - projection{b.center_index};
                }} while (false);
"""
    cuda += r"""
            }
        }
        if (out.v[1] || out.v[2]) {
"""
    for o in owners:
        b, parameter_slot, bias_slot = o["block"], o["param"], o["bias"]
        cuda += f"""
            do {{
                constexpr Block b = {block_descriptors[o['block']]};
                if (b.group != group)
                    break;
                const Index c = Index(channel_lane) + ({o['channel_shift']});
                const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
                if (owns) {{
                    parameter_shared[gang_id * PARAM_SLOTS * GANG + {parameter_slot} * GANG + lane] = gamma_acc[{parameter_slot}];
"""
        if b.bias_offset >= 0:
            cuda += f"""
                    beta_shared[gang_id * BETA_SLOTS * GANG + {bias_slot} * GANG + lane] = beta_acc[{bias_slot}];
"""
        cuda += """
                }
            } while (false);
"""
    cuda += r"""
            __syncthreads();
            if (gang_id == 0) {
"""
    for o in owners:
        b, parameter_slot, bias_slot = o["block"], o["param"], o["bias"]
        cuda += f"""
                do {{
                    constexpr Block b = {block_descriptors[o['block']]};
                    if (b.group != group)
                        break;
                    const Index c = Index(channel_lane) + ({o['channel_shift']});
                    const bool owns = channel_lane >= {o['lo']} && channel_lane < {o['hi']};
                    if (!owns)
                        break;
                    Acc gp = 0, bp = 0;
                    #pragma unroll
                    for (int gang = 0; gang < GANGS; ++gang) {{
                        gp += parameter_shared[gang * PARAM_SLOTS * GANG + {parameter_slot} * GANG + lane];
"""
        if b.bias_offset >= 0:
            cuda += f"""
                        bp += beta_shared[gang * BETA_SLOTS * GANG + {bias_slot} * GANG + lane];
"""
        cuda += r"""
                    }
                    #pragma unroll
                    for (int delta = COMPONENT_LANES / 2; delta > 0; delta >>= 1)
                        gp += __shfl_down_sync(__activemask(), gp, delta);
                    if (out.v[1] && component_lane == 0)
                        atomicAdd(out.v[1] + b.param_offset + c, gp);
"""
        if b.bias_offset >= 0:
            cuda += r"""
                    if (out.v[2] && component_lane == 0)
                        atomicAdd(out.v[2] + b.bias_offset + c, bp);
"""
        cuda += """
                } while (false);
"""
    cuda += r"""
            }
            __syncthreads();
        }
    }
"""
    cuda += r"""
}
__global__ void backward_colored_zero(Outputs out) {
    const Index p = Index(blockIdx.x) * 128 + threadIdx.x;
    if (out.v[1] && p < P)
        out.v[1][p] = Element(0);
    if (out.v[2] && p < BIAS)
        out.v[2][p] = Element(0);
}
"""
    cuda += r"""
thread_local int row_sm_count = 1;
int row_capacity() {
    static thread_local int cached_device = -1, cached_capacity = 0;
    int device = 0;
    cudaGetDevice(&device);
    if (device != cached_device) {
        int per_sm = 0;
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, backward_channel_colored, OWN_THREADS, 0);
        if (err != cudaSuccess || per_sm <= 0)
            throw std::runtime_error("backward row capacity is unavailable");
        cached_device = device;
        cached_capacity = per_sm * prop.multiProcessorCount;
        row_sm_count = prop.multiProcessorCount;
    }
    return cached_capacity;
}
int worker_count(int64_t N) {
    const int available = max(1, row_capacity() / NG);
    const int minimum_grid = max(1, (row_sm_count + NG - 1) / NG);
    const int64_t contractions = (N + 4 * GANGS - 1) / (4 * GANGS);
    const int64_t workers = min(int64_t(available), max(int64_t(minimum_grid), contractions));
    return int(min(workers, (N + GANGS - 1) / GANGS));
}
int64_t workspace_size(int64_t n, bool parameters) {
    return 0;
}
"""
    cuda += r"""
void launch(Inputs in, Outputs out, const Acc* fs, Acc* stats, int64_t N, cudaStream_t stream) {
    const int workers = worker_count(N), blocks = min(row_capacity(), workers * NG);
    if (out.v[1] || out.v[2])
        backward_colored_zero<<<(int64_t(max(P, BIAS)) + 127) / 128, 128, 0, stream>>>(out);
    backward_channel_colored<<<blocks, OWN_THREADS, 0, stream>>>(in, out, fs, stats, N, workers);
}
"""
    if dtype == torch.float32:
        cuda = suffix_float32_literals(cuda)
    return fast_view_source(
        {
            "cpp": cpp,
            "body": cuda,
            "prefix": common_source,
            "cuda_workspace": True,
            "trailing_includes": ("#include <stdexcept>",),
        },
        plan,
        dtype,
        active_inputs=(0, 1, 3),
    )
