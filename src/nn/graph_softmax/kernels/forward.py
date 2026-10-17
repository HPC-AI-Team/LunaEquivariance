"""Fused online CSR softmax and envelope weighting."""

CPP_SOURCE = r'''
void launch_graph_softmax_row_ptr(
    const int64_t*, int64_t*, int64_t, int64_t, cudaStream_t);
void launch_graph_softmax_forward(
    const float*, const float*, const void*, bool, float*,
    int64_t, int64_t, int64_t, float, int, bool, cudaStream_t);
void launch_graph_softmax_forward(
    const double*, const double*, const void*, bool, double*,
    int64_t, int64_t, int64_t, double, int, bool, cudaStream_t);

torch::Tensor graph_softmax_build_row_ptr(
    torch::Tensor segment_ids, int64_t num_segments)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(segment_ids));
    auto row_ptr = torch::empty(
        {num_segments + 1}, segment_ids.options().dtype(torch::kInt64));
    if (segment_ids.numel() == 0) return row_ptr.zero_();
    launch_graph_softmax_row_ptr(
        segment_ids.data_ptr<int64_t>(), row_ptr.data_ptr<int64_t>(),
        segment_ids.numel(), num_segments, at::cuda::getCurrentCUDAStream());
    return row_ptr;
}

torch::Tensor graph_softmax_forward_cuda(
    torch::Tensor data, torch::Tensor env, torch::Tensor row_ptr,
    double eps, int64_t env_mode, bool smooth_out)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(data));
    auto output = torch::empty_like(data);
    int64_t num_segments = row_ptr.numel() - 1;
    int64_t embed_dim = data.size(1);
    if (data.numel() == 0) return output;
    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "graph_softmax_forward", [&] {
        launch_graph_softmax_forward(
            data.data_ptr<scalar_t>(), env.numel() ? env.data_ptr<scalar_t>() : nullptr,
            row_ptr.data_ptr(), row_ptr.scalar_type() == torch::kInt32, output.data_ptr<scalar_t>(),
            num_segments, embed_dim, data.size(0), static_cast<scalar_t>(eps),
            static_cast<int>(env_mode), smooth_out,
            at::cuda::getCurrentCUDAStream());
    });
    return output;
}
'''


MODULE_BINDING = r'''    m.def("graph_softmax_build_row_ptr", &graph_softmax_build_row_ptr);
    m.def("graph_softmax_forward_cuda", &graph_softmax_forward_cuda);
'''


CUDA_SOURCE = r'''
__global__ void build_row_ptr_kernel(
    const int64_t* __restrict__ segment_ids, int64_t* __restrict__ row_ptr,
    int64_t num_edges, int64_t num_segments)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;
    int64_t current = segment_ids[tid];
    if (tid == 0) {
        for (int64_t s = 0; s <= current; ++s) row_ptr[s] = 0;
    } else if (current != segment_ids[tid - 1]) {
        for (int64_t s = segment_ids[tid - 1] + 1; s <= current; ++s) row_ptr[s] = tid;
    }
    if (tid == num_edges - 1) {
        for (int64_t s = current + 1; s <= num_segments; ++s) row_ptr[s] = num_edges;
    }
}

template <typename T, int Width>
struct ForwardPacket { T value[Width]; };

template <typename T, int Width>
__device__ __forceinline__ ForwardPacket<T, Width> load_forward_packet(
    const T* pointer, int64_t offset, int64_t dim, int64_t width, bool vectorized)
{
    ForwardPacket<T, Width> result = {};
    if constexpr (Width == 1) {
        if (dim < width) result.value[0] = pointer[offset];
    } else if (vectorized && dim + Width - 1 < width) {
        if constexpr (sizeof(T) == sizeof(float)) {
            float4 v = *reinterpret_cast<const float4*>(pointer + offset);
            result = {{v.x, v.y, v.z, v.w}};
        } else {
            double2 v = *reinterpret_cast<const double2*>(pointer + offset);
            result = {{v.x, v.y}};
        }
    } else {
#pragma unroll
        for (int c = 0; c < Width; ++c) {
            if (dim + c < width) result.value[c] = pointer[offset + c];
        }
    }
    return result;
}

template <typename T, int Width>
__device__ __forceinline__ void store_forward_packet(
    T* pointer, int64_t offset, int64_t dim, int64_t width,
    bool vectorized, const ForwardPacket<T, Width>& value)
{
    if constexpr (Width == 1) {
        if (dim < width) pointer[offset] = value.value[0];
    } else if (vectorized && dim + Width - 1 < width) {
        if constexpr (sizeof(T) == sizeof(float)) {
            *reinterpret_cast<float4*>(pointer + offset) = make_float4(
                value.value[0], value.value[1], value.value[2], value.value[3]);
        } else {
            *reinterpret_cast<double2*>(pointer + offset) = make_double2(value.value[0], value.value[1]);
        }
    } else {
#pragma unroll
        for (int c = 0; c < Width; ++c) {
            if (dim + c < width) pointer[offset + c] = value.value[c];
        }
    }
}

template <typename T, int EnvMode, int Width, int Threads>
__global__ void __launch_bounds__(Threads, Threads >= 512 ? 1 : 2) graph_softmax_forward_online_kernel(
    const T* __restrict__ data, const T* __restrict__ env,
    const void* __restrict__ row_ptr, bool csr_int32, T* __restrict__ output,
    int64_t embed_dim, T eps, bool smooth_out)
{
    constexpr int kOnlineDimTile = 32 * Width;
    constexpr int Groups = Threads / 32;
    constexpr int CacheEdges = sizeof(T) == sizeof(double) && EnvMode == 2 ? 16 : 32;
    constexpr int Alignment = sizeof(T) == sizeof(float) ? 4 : 2;
    const int64_t seg = static_cast<int64_t>(blockIdx.x);
    const int64_t start = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[seg]
        : static_cast<const int64_t*>(row_ptr)[seg]);
    const int64_t end = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[seg + 1]
        : static_cast<const int64_t*>(row_ptr)[seg + 1]);
    if (start == end) return;
    const int lane = threadIdx.x % 32;
    const int group = threadIdx.x / 32;
    const int local_dim = lane * Width;
    const bool aligned_stride = embed_dim % Alignment == 0;
    const bool vector_data = aligned_stride && (reinterpret_cast<uintptr_t>(data) & 15) == 0;
    const bool vector_env = aligned_stride && (reinterpret_cast<uintptr_t>(env) & 15) == 0;
    __shared__ T normals[2][Groups][kOnlineDimTile];
    __shared__ T cached_data[CacheEdges * kOnlineDimTile];
    __shared__ T cached_env[EnvMode == 2 ? CacheEdges * kOnlineDimTile : 1];
    for (int64_t dim_base = static_cast<int64_t>(blockIdx.y) * kOnlineDimTile;
         dim_base < embed_dim; dim_base += static_cast<int64_t>(gridDim.y) * kOnlineDimTile) {
        const int64_t dim = dim_base + local_dim;
        OnlineNormalizer<T> states[Width];

        for (int64_t edge = start + group; edge < end; edge += Groups) {
            const int64_t offset = edge * embed_dim + dim;
            auto value = load_forward_packet<T, Width>(data, offset, dim, embed_dim, vector_data);
            ForwardPacket<T, Width> envelope;
            if constexpr (EnvMode == 2) {
                envelope = load_forward_packet<T, Width>(env, offset, dim, embed_dim, vector_env);
            } else {
                T e = EnvMode == 0 ? T(1) : env[edge];
                for (int c = 0; c < Width; ++c) envelope.value[c] = e;
            }
#pragma unroll
            for (int c = 0; c < Width; ++c) {
                if (edge - start < CacheEdges) {
                    int64_t cache = (edge - start) * kOnlineDimTile + local_dim + c;
                    cached_data[cache] = value.value[c];
                    if constexpr (EnvMode == 2) cached_env[cache] = envelope.value[c];
                }
                if (dim + c < embed_dim) {
                    states[c].add(value.value[c], envelope.value[c]);
                }
            }
        }
        // Merge the independently accumulated edge groups once per row tile.
#pragma unroll
        for (int c = 0; c < Width; ++c) {
            normals[0][group][local_dim+c] = states[c].maximum;
            normals[1][group][local_dim+c] = states[c].weighted_sum;
        }
        __syncthreads();
        if (group == 0) {
#pragma unroll
            for (int other = 1; other < Groups; ++other) {
#pragma unroll
                for (int c = 0; c < Width; ++c) {
                    states[c].merge(normals[0][other][local_dim+c], normals[1][other][local_dim+c]);
                }
            }
#pragma unroll
            for (int c = 0; c < Width; ++c) {
                normals[0][0][local_dim+c] = states[c].maximum;
                normals[1][0][local_dim+c] = states[c].weighted_sum;
            }
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < Width; ++c) {
            states[c].maximum = normals[0][0][local_dim+c];
            states[c].weighted_sum = normals[1][0][local_dim+c];
        }
        ForwardPacket<T, Width> inverse;
#pragma unroll
        for (int c = 0; c < Width; ++c) inverse.value[c] = T(1) / (states[c].weighted_sum + eps);
        for (int64_t edge = start + group; edge < end; edge += Groups) {
            const int64_t offset = edge * embed_dim + dim;
            ForwardPacket<T, Width> value;
            ForwardPacket<T, Width> envelope;
            if (edge - start < CacheEdges) {
#pragma unroll
                for (int c = 0; c < Width; ++c) {
                    int64_t cache = (edge - start) * kOnlineDimTile + local_dim + c;
                    value.value[c] = cached_data[cache];
                    if constexpr (EnvMode == 2) envelope.value[c] = cached_env[cache];
                }
            } else {
                value = load_forward_packet<T, Width>(data, offset, dim, embed_dim, vector_data);
                if constexpr (EnvMode == 2) envelope = load_forward_packet<T, Width>(env, offset, dim, embed_dim, vector_env);
            }
            if constexpr (EnvMode != 2) {
                T e = EnvMode == 0 ? T(1) : env[edge];
                for (int c = 0; c < Width; ++c) envelope.value[c] = e;
            }
            ForwardPacket<T, Width> result;
#pragma unroll
            for (int c = 0; c < Width; ++c) {
                T e = envelope.value[c];
                result.value[c] = exp_op<T>(value.value[c] - states[c].maximum) * inverse.value[c]
                    * e * (smooth_out ? e : T(1));
            }
            store_forward_packet(output, offset, dim, embed_dim, aligned_stride, result);
        }
        __syncthreads();
    }
}
'''


LAUNCH_SOURCE = r'''
void launch_graph_softmax_row_ptr(
    const int64_t* segment_ids, int64_t* row_ptr, int64_t num_edges,
    int64_t num_segments, cudaStream_t stream)
{
    build_row_ptr_kernel<<<(num_edges + 255) / 256, 256, 0, stream>>>(
        segment_ids, row_ptr, num_edges, num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int Width, int Threads>
void launch_graph_softmax_forward_impl(
    const scalar_t* data, const scalar_t* env, const void* row_ptr, bool csr_int32,
    scalar_t* output, int64_t num_segments,
    int64_t embed_dim, scalar_t eps, int env_mode, bool smooth_out,
    cudaStream_t stream)
{
    constexpr int kOnlineDimTile = 32 * Width;
    const int64_t tiles = (embed_dim + kOnlineDimTile - 1) / kOnlineDimTile;
    dim3 grid(num_segments, tiles < 65535 ? tiles : 65535);
    if (env_mode == 0) {
        graph_softmax_forward_online_kernel<scalar_t, 0, Width, Threads><<<grid, Threads, 0, stream>>>(
            data, env, row_ptr, csr_int32, output, embed_dim, eps, smooth_out);
    } else if (env_mode == 1) {
        graph_softmax_forward_online_kernel<scalar_t, 1, Width, Threads><<<grid, Threads, 0, stream>>>(
            data, env, row_ptr, csr_int32, output, embed_dim, eps, smooth_out);
    } else {
        graph_softmax_forward_online_kernel<scalar_t, 2, Width, Threads><<<grid, Threads, 0, stream>>>(
            data, env, row_ptr, csr_int32, output, embed_dim, eps, smooth_out);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int Width>
void launch_graph_softmax_forward_rows(
    const T* data, const T* env, const void* row_ptr, bool csr_int32,
    T* output, int64_t segments, int64_t dim, int64_t edges,
    T eps, int mode, bool smooth, cudaStream_t stream)
{
    // Bound packet work per CTA; give longer rows more cooperating warps.
    if (edges <= segments * 64) {
        launch_graph_softmax_forward_impl<T, Width, 128>(
            data, env, row_ptr, csr_int32, output, segments, dim, eps, mode, smooth, stream);
    } else {
        launch_graph_softmax_forward_impl<T, Width, 1024 / Width>(
            data, env, row_ptr, csr_int32, output, segments, dim, eps, mode, smooth, stream);
    }
}

void launch_graph_softmax_forward(
    const float* data, const float* env, const void* row_ptr, bool csr_int32,
    float* output, int64_t num_segments, int64_t embed_dim, int64_t edges,
    float eps, int env_mode, bool smooth_out, cudaStream_t stream)
{
    if (embed_dim <= 32 || embed_dim % 4 != 0) {
        launch_graph_softmax_forward_rows<float, 1>(
            data, env, row_ptr, csr_int32, output, num_segments, embed_dim, edges,
            eps, env_mode, smooth_out, stream);
    } else {
        launch_graph_softmax_forward_rows<float, 4>(
            data, env, row_ptr, csr_int32, output, num_segments, embed_dim, edges,
            eps, env_mode, smooth_out, stream);
    }
}

void launch_graph_softmax_forward(
    const double* data, const double* env, const void* row_ptr, bool csr_int32,
    double* output, int64_t num_segments, int64_t embed_dim, int64_t edges,
    double eps, int env_mode, bool smooth_out, cudaStream_t stream)
{
    if (embed_dim <= 32 || embed_dim % 2 != 0) {
        launch_graph_softmax_forward_rows<double, 1>(
            data, env, row_ptr, csr_int32, output, num_segments, embed_dim, edges,
            eps, env_mode, smooth_out, stream);
    } else {
        launch_graph_softmax_forward_rows<double, 2>(
            data, env, row_ptr, csr_int32, output, num_segments, embed_dim, edges,
            eps, env_mode, smooth_out, stream);
    }
}
'''
