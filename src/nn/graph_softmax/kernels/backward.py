"""Fused CSR softmax backward with on-chip normalization recomputation."""

CPP_SOURCE = r"""
void launch_graph_softmax_backward(
    const float*, const float*, const float*, const void*, bool, float*, float*,
    int64_t, int64_t, int64_t, int, bool, float, cudaStream_t);
void launch_graph_softmax_backward(
    const double*, const double*, const double*, const void*, bool, double*, double*,
    int64_t, int64_t, int64_t, int, bool, double, cudaStream_t);

std::vector<torch::Tensor> graph_softmax_backward_cuda(
    torch::Tensor grad_output, torch::Tensor data, torch::Tensor env,
    torch::Tensor row_ptr, int64_t env_mode, bool need_grad_env,
    bool smooth_out, double eps)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    grad_output = grad_output.contiguous();
    auto grad_data = torch::empty_like(data);
    auto grad_env = need_grad_env ? torch::empty_like(env) : torch::empty({0}, env.options());
    if (data.numel() == 0) return {grad_data, grad_env};
    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "graph_softmax_backward", [&] {
        launch_graph_softmax_backward(
            grad_output.data_ptr<scalar_t>(), data.data_ptr<scalar_t>(),
            env.numel() ? env.data_ptr<scalar_t>() : nullptr,
            row_ptr.data_ptr(), row_ptr.scalar_type() == torch::kInt32, grad_data.data_ptr<scalar_t>(),
            grad_env.numel() ? grad_env.data_ptr<scalar_t>() : nullptr,
            row_ptr.numel() - 1, data.size(1), data.size(0), static_cast<int>(env_mode),
            smooth_out, static_cast<scalar_t>(eps), at::cuda::getCurrentCUDAStream());
    });
    return {grad_data, grad_env};
}
"""

MODULE_BINDING = '    m.def("graph_softmax_backward_cuda", &graph_softmax_backward_cuda);\n'


CUDA_SOURCE = r"""
// Each warp owns an edge subsequence; lanes hold adjacent feature packets.
// Scalar envelopes keep a whole-row CTA; independent features use tiled CTAs.
template <typename scalar_t, int Width>
struct BackwardPacket {
    scalar_t value[Width];
};

template <int Width, typename scalar_t>
__device__ __forceinline__ BackwardPacket<scalar_t, Width> load_backward_packet(
    const scalar_t* pointer, int64_t offset, int64_t dim,
    int64_t embed_dim, bool vectorized)
{
    BackwardPacket<scalar_t, Width> packet = {};
    if constexpr (Width == 1) {
        if (dim < embed_dim) packet.value[0] = pointer[offset];
    } else if (vectorized && dim + (Width - 1) < embed_dim) {
        if constexpr (sizeof(scalar_t) == sizeof(float)) {
            float4 v = *reinterpret_cast<const float4*>(pointer + offset);
            packet.value[0] = v.x;
            packet.value[1] = v.y;
            packet.value[2] = v.z;
            packet.value[3] = v.w;
        } else {
            double2 lo = *reinterpret_cast<const double2*>(pointer + offset);
            if constexpr (Width == 4) {
                double2 hi = *reinterpret_cast<const double2*>(pointer + offset + 2);
                packet.value[0] = lo.x;
                packet.value[1] = lo.y;
                packet.value[2] = hi.x;
                packet.value[3] = hi.y;
            } else {
                packet.value[0] = lo.x;
                packet.value[1] = lo.y;
            }
        }
    } else {
        #pragma unroll
        for (int component = 0; component < Width; ++component) {
            if (dim + component < embed_dim) {
                packet.value[component] = pointer[offset + component];
            }
        }
    }
    return packet;
}

template <typename scalar_t, int Width>
__device__ __forceinline__ void store_backward_packet(
    scalar_t* pointer, int64_t offset, int64_t dim, int64_t embed_dim,
    const BackwardPacket<scalar_t, Width>& packet, bool vectorized)
{
    if constexpr (Width == 1) {
        if (dim < embed_dim) pointer[offset] = packet.value[0];
    } else if (vectorized && dim + (Width - 1) < embed_dim) {
        if constexpr (sizeof(scalar_t) == sizeof(float)) {
            *reinterpret_cast<float4*>(pointer + offset) = make_float4(
                packet.value[0], packet.value[1], packet.value[2], packet.value[3]);
        } else {
            *reinterpret_cast<double2*>(pointer + offset) = make_double2(
                packet.value[0], packet.value[1]);
            if constexpr (Width == 4) {
                *reinterpret_cast<double2*>(pointer + offset + 2) = make_double2(
                    packet.value[2], packet.value[3]);
            }
        }
    } else {
        #pragma unroll
        for (int component = 0; component < Width; ++component) {
            if (dim + component < embed_dim) {
                pointer[offset + component] = packet.value[component];
            }
        }
    }
}

template <typename T, int EnvMode, int Width, int Threads>
__global__ void __launch_bounds__(Threads, Threads >= 512 ? 1 : 2) graph_softmax_backward_kernel(
    const T* __restrict__ grad_output, const T* __restrict__ data,
    const T* __restrict__ env, const void* __restrict__ row_ptr, bool csr_int32,
    T* __restrict__ grad_data, T* __restrict__ grad_env,
    int64_t embed_dim, bool smooth, T eps)
{
    constexpr int Warps = Threads / 32;
    constexpr int Tile = 32 * Width;
    constexpr int CacheEdges = 32;
    constexpr bool FeatureGrid = EnvMode != 1;
    using Packet = BackwardPacket<T, Width>;
    const int64_t segment = blockIdx.x;
    const int64_t begin = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[segment]
        : static_cast<const int64_t*>(row_ptr)[segment]);
    const int64_t end = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[segment + 1]
        : static_cast<const int64_t*>(row_ptr)[segment + 1]);
    if (begin == end) return;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int group = warp;
    const int local_dim = lane * Width;
    const int64_t tile_start = FeatureGrid ? int64_t(blockIdx.y) * Tile : 0;
    const int64_t tile_stride = FeatureGrid ? int64_t(gridDim.y) * Tile : Tile;
    const bool aligned_stride = embed_dim % Width == 0;
    const bool vector_x = aligned_stride && (reinterpret_cast<uintptr_t>(data) & 15) == 0;
    const bool vector_g = aligned_stride && (reinterpret_cast<uintptr_t>(grad_output) & 15) == 0;
    const bool vector_e = aligned_stride && (reinterpret_cast<uintptr_t>(env) & 15) == 0;
    const bool vector_dx = aligned_stride && (reinterpret_cast<uintptr_t>(grad_data) & 15) == 0;
    const bool vector_de = aligned_stride && (reinterpret_cast<uintptr_t>(grad_env) & 15) == 0;
    __shared__ __align__(16) T cache[CacheEdges][Tile];
    __shared__ T moments[3][Warps][Tile];

    for (int64_t base_dim = tile_start; base_dim < embed_dim; base_dim += tile_stride) {
        const int64_t dim = base_dim + local_dim;
        Packet maximum = {}, inverse = {};
        T dot[Width] = {};
        if (end - begin != 1) {
            OnlineMoments<T, 1> states[Width];
            for (int64_t edge = begin + group; edge < end; edge += Warps) {
                const int64_t offset = edge * embed_dim + dim;
                const auto x = load_backward_packet<Width>(data, offset, dim, embed_dim, vector_x);
                const auto g = load_backward_packet<Width>(grad_output, offset, dim, embed_dim, vector_g);
                #pragma unroll
                for (int c = 0; c < Width; ++c) {
                    if (dim + c < embed_dim) {
                        const T e = env_value<T, EnvMode>(env, edge, dim + c, embed_dim);
                        const T moment[1] = {g.value[c] * e * (smooth ? e : T(1))};
                        states[c].add(x.value[c], e, moment);
                    }
                }
                if (edge - begin < CacheEdges) {
                    store_backward_packet(&cache[0][0], (edge - begin) * Tile + local_dim,
                                          local_dim, int64_t(Tile), x, true);
                }
            }
            #pragma unroll
            for (int c = 0; c < Width; ++c) {
                moments[0][warp][local_dim+c] = states[c].maximum;
                moments[1][warp][local_dim+c] = states[c].weighted_sum;
                moments[2][warp][local_dim+c] = states[c].moments[0];
            }
            __syncthreads();
            #pragma unroll
            for (int stride = Warps / 2; stride > 0; stride /= 2) {
                if (warp < stride) {
                    #pragma unroll
                    for (int c = 0; c < Width; ++c) {
                        T value[1] = {moments[2][warp+stride][local_dim+c]};
                        states[c].merge(moments[0][warp+stride][local_dim+c],
                                        moments[1][warp+stride][local_dim+c], value);
                        moments[0][warp][local_dim+c] = states[c].maximum;
                        moments[1][warp][local_dim+c] = states[c].weighted_sum;
                        moments[2][warp][local_dim+c] = states[c].moments[0];
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (int c = 0; c < Width; ++c) {
                states[c].maximum = moments[0][0][local_dim+c];
                states[c].weighted_sum = moments[1][0][local_dim+c];
                states[c].moments[0] = moments[2][0][local_dim+c];
            }
            #pragma unroll
            for (int c = 0; c < Width; ++c) {
                maximum.value[c] = states[c].maximum;
                inverse.value[c] = T(1) / (states[c].weighted_sum + eps);
                dot[c] = states[c].moments[0] * inverse.value[c];
            }
        }

        // Each warp writes its edge subsequence and reduces envelope channels.
        for (int64_t edge = begin + warp; edge < end; edge += Warps) {
            T env_sum = T(0);
            {
                const int64_t offset = edge * embed_dim + dim;
                const auto g = load_backward_packet<Width>(grad_output, offset, dim, embed_dim, vector_g);
                Packet q = {}, envelope = {}, dx = {}, de = {};
                if constexpr (EnvMode == 2) {
                    envelope = load_backward_packet<Width>(env, offset, dim, embed_dim, vector_e);
                }
                if (end - begin != 1) {
                    const auto x = edge - begin < CacheEdges
                        ? load_backward_packet<Width>(&cache[0][0], (edge - begin) * Tile + local_dim,
                                                       local_dim, int64_t(Tile), true)
                        : load_backward_packet<Width>(data, offset, dim, embed_dim, vector_x);
                    #pragma unroll
                    for (int c = 0; c < Width; ++c) {
                        q.value[c] = exp_op(x.value[c] - maximum.value[c]) * inverse.value[c];
                    }
                }
                #pragma unroll
                for (int c = 0; c < Width; ++c) {
                    if (dim + c < embed_dim) {
                        const T e = EnvMode == 0 ? T(1)
                            : (EnvMode == 1 ? env[edge] : envelope.value[c]);
                        T derivative_e;
                        if (end - begin == 1) {
                            const T qi = T(1) / (e + eps);
                            const T p = qi * e, residual = eps * qi;
                            dx.value[c] = g.value[c] * p * residual * (smooth ? e : T(1));
                            derivative_e = g.value[c] * (smooth ? p * (T(1) + residual) : qi * residual);
                        } else {
                            const T p = q.value[c] * e;
                            const T y = p * (smooth ? e : T(1));
                            dx.value[c] = g.value[c] * y - p * dot[c];
                            derivative_e = q.value[c] * (g.value[c] * (smooth ? T(2) * e : T(1)) - dot[c]);
                        }
                        if constexpr (EnvMode == 1) env_sum += derivative_e;
                        else if constexpr (EnvMode == 2) de.value[c] = derivative_e;
                    }
                }
                store_backward_packet(grad_data, offset, dim, embed_dim, dx, vector_dx);
                if constexpr (EnvMode == 2) {
                    if (grad_env) store_backward_packet(grad_env, offset, dim, embed_dim, de, vector_de);
                }
            }
            if constexpr (EnvMode == 1) {
                if (grad_env) {
                    env_sum = warp_reduce_sum(env_sum);
                    if (lane == 0) grad_env[edge] = env_sum + (base_dim ? grad_env[edge] : T(0));
                }
            }
        }
        __syncthreads();
    }

}
"""

LAUNCH_SOURCE = r"""
template <typename T, int Width, int Threads>
void launch_graph_softmax_backward_impl(
    const T* grad_output, const T* data, const T* env, const void* row_ptr, bool csr_int32,
    T* grad_data, T* grad_env, int64_t segments, int64_t dim,
    int mode, bool smooth, T eps, cudaStream_t stream)
{
    constexpr int Tile = 32 * Width;
    const int64_t tiles = (dim + Tile - 1) / Tile;
    const unsigned int grid_y = static_cast<unsigned int>(min(tiles, int64_t(65535)));
    if (mode == 0) {
        graph_softmax_backward_kernel<T, 0, Width, Threads><<<dim3(segments, grid_y), Threads, 0, stream>>>(
            grad_output, data, env, row_ptr, csr_int32, grad_data, grad_env, dim, smooth, eps);
    } else if (mode == 1) {
        graph_softmax_backward_kernel<T, 1, Width, Threads><<<segments, Threads, 0, stream>>>(
            grad_output, data, env, row_ptr, csr_int32, grad_data, grad_env, dim, smooth, eps);
    } else {
        graph_softmax_backward_kernel<T, 2, Width, Threads><<<dim3(segments, grid_y), Threads, 0, stream>>>(
            grad_output, data, env, row_ptr, csr_int32, grad_data, grad_env, dim, smooth, eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int Width>
void launch_graph_softmax_backward_rows(
    const T* g, const T* data, const T* env, const void* row_ptr, bool csr_int32,
    T* dx, T* de, int64_t segments, int64_t dim, int64_t edges,
    int mode, bool smooth, T eps, cudaStream_t stream)
{
    if (edges <= segments * 64) {
        launch_graph_softmax_backward_impl<T, Width, (Width == 1 ? 256 : 128)>(
            g, data, env, row_ptr, csr_int32, dx, de, segments, dim, mode, smooth, eps, stream);
    } else {
        launch_graph_softmax_backward_impl<T, Width, 1024 / Width>(
            g, data, env, row_ptr, csr_int32, dx, de, segments, dim, mode, smooth, eps, stream);
    }
}

void launch_graph_softmax_backward(
    const float* g, const float* data, const float* env, const void* row_ptr, bool csr_int32,
    float* dx, float* de, int64_t segments, int64_t dim, int64_t edges,
    int mode, bool smooth, float eps, cudaStream_t stream)
{
    if (dim <= 32 || dim % 4 != 0) {
        launch_graph_softmax_backward_rows<float, 1>(g, data, env, row_ptr, csr_int32, dx, de, segments, dim, edges, mode, smooth, eps, stream);
    } else {
        launch_graph_softmax_backward_rows<float, 4>(g, data, env, row_ptr, csr_int32, dx, de, segments, dim, edges, mode, smooth, eps, stream);
    }
}

void launch_graph_softmax_backward(
    const double* g, const double* data, const double* env, const void* row_ptr, bool csr_int32,
    double* dx, double* de, int64_t segments, int64_t dim, int64_t edges,
    int mode, bool smooth, double eps, cudaStream_t stream)
{
    if (dim <= 32 || dim % 2 != 0) {
        launch_graph_softmax_backward_rows<double, 1>(g, data, env, row_ptr, csr_int32, dx, de, segments, dim, edges, mode, smooth, eps, stream);
    } else {
        launch_graph_softmax_backward_rows<double, 2>(g, data, env, row_ptr, csr_int32, dx, de, segments, dim, edges, mode, smooth, eps, stream);
    }
}
"""
