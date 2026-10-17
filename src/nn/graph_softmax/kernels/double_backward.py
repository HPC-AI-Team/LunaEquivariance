"""Graph softmax double backward: fused CUDA computation and binding."""

CPP_SOURCE = r"""
template <typename T>
void launch_graph_softmax_double_backward_fused(
    const T* grad_grad_data, const T* grad_grad_env, const T* grad_output,
    const T* data, const T* env,
    const void* row_ptr, bool csr_int32, T* grad_grad_output, T* grad_data, T* grad_env,
    int64_t num_segments, int64_t embed_dim, int64_t edges, int env_mode, bool smooth_out,
    T eps, cudaStream_t stream);

std::vector<torch::Tensor> graph_softmax_double_backward_cuda(
    torch::Tensor grad_grad_data,
    torch::Tensor grad_grad_env,
    torch::Tensor grad_output,
    torch::Tensor data,
    torch::Tensor env,
    torch::Tensor row_ptr,
    int64_t env_mode,
    bool need_grad_env,
    bool smooth_out,
    double eps)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    grad_grad_data = grad_grad_data.contiguous();
    grad_grad_env = grad_grad_env.contiguous();
    grad_output = grad_output.contiguous();
    auto grad_grad_output = torch::empty_like(data);
    auto grad_data = torch::empty_like(data);
    auto grad_env = need_grad_env ? torch::empty_like(env)
                                 : torch::empty({0}, env.options());
    if (data.numel() == 0) {
        return {grad_grad_output, grad_data, grad_env};
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "graph_softmax_double_backward", [&] {
        launch_graph_softmax_double_backward_fused<scalar_t>(
            grad_grad_data.numel() ? grad_grad_data.data_ptr<scalar_t>() : nullptr,
            grad_grad_env.numel() ? grad_grad_env.data_ptr<scalar_t>() : nullptr,
            grad_output.data_ptr<scalar_t>(), data.data_ptr<scalar_t>(),
            env.numel() ? env.data_ptr<scalar_t>() : nullptr,
            row_ptr.data_ptr(), row_ptr.scalar_type() == torch::kInt32, grad_grad_output.data_ptr<scalar_t>(),
            grad_data.data_ptr<scalar_t>(),
            grad_env.numel() ? grad_env.data_ptr<scalar_t>() : nullptr,
            row_ptr.numel() - 1, data.size(1), data.size(0), static_cast<int>(env_mode),
            smooth_out, static_cast<scalar_t>(eps), stream);
    });
    return {grad_grad_output, grad_data, grad_env};
}
"""

MODULE_BINDING = '    m.def("graph_softmax_double_backward_cuda", &graph_softmax_double_backward_cuda);\n'


CUDA_SOURCE = r"""
// Scalar envelopes keep one CTA per CSR segment for unique reductions.
// Independent feature tiles use separate CTAs when no envelope reduction is needed.
template <typename T, int EnvMode, int Threads, int FeatureTile>
__global__ void __launch_bounds__(Threads, EnvMode == 0 ? 1 : 0) graph_softmax_double_backward_fused_kernel(
    const T* __restrict__ grad_grad_data,
    const T* __restrict__ grad_grad_env,
    const T* __restrict__ grad_output,
    const T* __restrict__ data,
    const T* __restrict__ env,
    const void* __restrict__ row_ptr, bool csr_int32,
    T* __restrict__ grad_grad_output,
    T* __restrict__ grad_data,
    T* __restrict__ grad_env,
    int64_t embed_dim,
    bool smooth_out,
    T eps)
{
    const int64_t seg = blockIdx.x;
    const int64_t start = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[seg]
        : static_cast<const int64_t*>(row_ptr)[seg]);
    const int64_t end = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[seg + 1]
        : static_cast<const int64_t*>(row_ptr)[seg + 1]);
    if (start == end) return;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int feature = threadIdx.x % FeatureTile;
    const int edge_group = threadIdx.x / FeatureTile;
    constexpr int EdgeGroups = Threads / FeatureTile;
    constexpr int Warps = Threads / kWarpSize;
    const int64_t dim_stride = EnvMode == 1 ? FeatureTile : int64_t(gridDim.y) * FeatureTile;
    const T two = T(2);
    // Five online fields become A/C/B/max/inverse after the shared merge.
    __shared__ T sums[5][Threads];
    constexpr int CacheBudget = (48 * 1024 / sizeof(T) - 5 * Threads) / FeatureTile;
    constexpr int CachedEdges = CacheBudget < 31 ? CacheBudget : 31;
    __shared__ T cache[CachedEdges][FeatureTile];

    // Preserve the epsilon residual directly on singleton rows; subtracting
    // nearly unit probabilities loses the envelope derivatives in FP32.
    if (end - start == 1) {
        for (int64_t dim_base = EnvMode == 1 ? 0 : int64_t(blockIdx.y) * FeatureTile; dim_base < embed_dim;
             dim_base += EnvMode == 1 ? Threads : dim_stride) {
            const int64_t dim_end = EnvMode == 1 ? embed_dim : min(embed_dim, dim_base + FeatureTile);
            const int64_t dim = dim_base + threadIdx.x;
            T de = T(0);
            if (dim < dim_end) {
                const int64_t offset = start * embed_dim + dim;
                const T e = env_value<T, EnvMode>(env, start, dim, embed_dim);
                const T q = T(1) / (e + eps);
                const T p = q * e;
                const T w = eps * q;
                const T J = T(1) - two * p;
                const T Fx = (smooth_out ? e : T(1)) * p * w;
                const T Fe = smooth_out ? p * (T(1) + w) : q * w;
                const T Fxx = Fx * J;
                const T Fxe = smooth_out ? two * p * w * w : q * w * J;
                const T Fee = smooth_out ? two * q * w * w : -two * q * q * w;
                const T g = grad_output[offset];
                const T U = grad_grad_data ? grad_grad_data[offset] : T(0);
                const T V = EnvMode == 0 ? T(0) : (grad_grad_env ? env_value<T, EnvMode>(grad_grad_env, start, dim, embed_dim) : T(0));
                grad_grad_output[offset] = U * Fx + V * Fe;
                grad_data[offset] = g * (U * Fxx + V * Fxe);
                if (grad_env) {
                    de = g * (U * Fxe + V * Fee);
                    if constexpr (EnvMode == 2) grad_env[offset] = de;
                }
            }
            if constexpr (EnvMode == 1) {
                if (grad_env) {
                    de = warp_reduce_sum(de);
                    if (lane == 0) sums[0][warp] = de;
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        T total = T(0);
                        #pragma unroll
                        for (int group = 0; group < Warps; ++group) total += sums[0][group];
                        grad_env[start] = dim_base == 0 ? total : grad_env[start] + total;
                    }
                    __syncthreads();
                }
            }
        }
        return;
    }

    for (int64_t dim_base = EnvMode == 1 ? 0 : int64_t(blockIdx.y) * FeatureTile; dim_base < embed_dim; dim_base += dim_stride) {
        const int64_t dim = dim_base + feature;
        OnlineMoments<T, 3> state;
        if (dim < embed_dim) {
            for (int64_t edge = start + edge_group; edge < end; edge += EdgeGroups) {
                const int64_t offset = edge * embed_dim + dim;
                const T value = data[offset];
                const int64_t local_edge = edge - start;
                if (local_edge < CachedEdges) cache[local_edge][feature] = value;
                const T e = env_value<T, EnvMode>(env, edge, dim, embed_dim);
                const T g = grad_output[offset];
                const T U = grad_grad_data ? grad_grad_data[offset] : T(0);
                const T V = EnvMode == 0 ? T(0) : (grad_grad_env ? env_value<T, EnvMode>(grad_grad_env, edge, dim, embed_dim) : T(0));
                const T f = e * (smooth_out ? e : T(1));
                const T t = smooth_out ? two * e : T(1);
                const T weights[3] = {g * f, U * e + V, g * (U * f + V * t)};
                state.add(value, e, weights);
            }
        }
        sums[0][threadIdx.x] = state.maximum;
        sums[1][threadIdx.x] = state.weighted_sum;
        #pragma unroll
        for (int moment = 0; moment < 3; ++moment) {
            sums[2 + moment][threadIdx.x] = state.moments[moment];
        }
        __syncthreads();
        if (threadIdx.x < FeatureTile) {
            OnlineMoments<T, 3> merged;
            #pragma unroll
            for (int group = 0; group < EdgeGroups; ++group) {
                const int index = group * FeatureTile + threadIdx.x;
                const T moments[3] = {sums[2][index], sums[3][index], sums[4][index]};
                merged.merge(sums[0][index], sums[1][index], moments);
            }
            const T inverse = T(1) / (merged.weighted_sum + eps);
            // Each merge owns one feature column, so these writes cannot
            // overwrite another thread's unread normalizer or moments.
            #pragma unroll
            for (int moment = 0; moment < 3; ++moment) {
                sums[moment][threadIdx.x] = merged.moments[moment] * inverse;
            }
            sums[3][threadIdx.x] = merged.maximum;
            sums[4][threadIdx.x] = inverse;
        }
        __syncthreads();

        for (int64_t edge = start + warp; edge < end; edge += Warps) {
            T env_sum = T(0);
            for (int local_dim = lane; local_dim < FeatureTile; local_dim += kWarpSize) {
                const int64_t dim = dim_base + local_dim;
                if (dim >= embed_dim) continue;
                const int64_t offset = edge * embed_dim + dim;
                const T maximum = sums[3][local_dim];
                const T inverse = sums[4][local_dim];
                const int64_t local_edge = edge - start;
                const T value = local_edge < CachedEdges ? cache[local_edge][local_dim] : data[offset];
                const T q = exp_op(value - maximum) * inverse;
                const T e = env_value<T, EnvMode>(env, edge, dim, embed_dim);
                const T p = q * e;
                const T y = q * e * (smooth_out ? e : T(1));
                const T g = grad_output[offset];
                const T U = grad_grad_data ? grad_grad_data[offset] : T(0);
                const T V = EnvMode == 0 ? T(0) : (grad_grad_env ? env_value<T, EnvMode>(grad_grad_env, edge, dim, embed_dim) : T(0));
                const T A = sums[0][local_dim];
                const T C = sums[1][local_dim];
                const T B = sums[2][local_dim];
                const T t = smooth_out ? two * e : T(1);
                const T tprime = smooth_out ? two : T(0);
                const T dg = U * y + V * q * t - y * C;
                grad_grad_output[offset] = dg;
                grad_data[offset] = g * dg - p * B - A * (U * p + V * q) + two * A * p * C;
                if (grad_env) {
                    const T de = q * (g * (U * t + V * tprime) - B - A * U
                                     + C * (two * A - g * t));
                    if constexpr (EnvMode == 1) env_sum += de;
                    else if constexpr (EnvMode == 2) grad_env[offset] = de;
                }
            }
            if constexpr (EnvMode == 1) {
                if (grad_env) {
                    env_sum = warp_reduce_sum(env_sum);
                    if (lane == 0) {
                        // This warp is the unique owner of this envelope data.
                        grad_env[edge] = dim_base == 0 ? env_sum : grad_env[edge] + env_sum;
                    }
                }
            }
        }
        __syncthreads();
    }
}
"""

LAUNCH_SOURCE = r"""
template <typename T, int scalar_tile, int independent_tile>
void launch_graph_softmax_double_backward_impl(
    const T* grad_grad_data, const T* grad_grad_env, const T* grad_output,
    const T* data, const T* env,
    const void* row_ptr, bool csr_int32, T* grad_grad_output, T* grad_data, T* grad_env,
    int64_t num_segments, int64_t embed_dim, int env_mode, bool smooth_out,
    T eps, cudaStream_t stream)
{
#define LAUNCH_DOUBLE(Mode, Threads, Tile, Grid) \
    graph_softmax_double_backward_fused_kernel<T, Mode, Threads, Tile><<<Grid, Threads, 0, stream>>>( \
        grad_grad_data, grad_grad_env, grad_output, data, env, \
        row_ptr, csr_int32, grad_grad_output, grad_data, grad_env, embed_dim, smooth_out, eps)
    const dim3 scalar_grid(num_segments);
    const int64_t feature_tiles = (embed_dim + independent_tile - 1) / independent_tile;
    const unsigned grid_y = feature_tiles < 65535 ? feature_tiles : 65535;
    const dim3 independent_grid(num_segments, grid_y);
    switch (env_mode) {
        case 0: LAUNCH_DOUBLE(0, 512, independent_tile, independent_grid); break;
        case 1: LAUNCH_DOUBLE(1, 1024, scalar_tile, scalar_grid); break;
        default: LAUNCH_DOUBLE(2, 512, independent_tile, independent_grid); break;
    }
#undef LAUNCH_DOUBLE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_graph_softmax_double_backward_fused(
    const T* u, const T* v, const T* g, const T* data, const T* env,
    const void* row_ptr, bool csr_int32, T* dg, T* dx, T* de,
    int64_t segments, int64_t dim, int64_t edges, int mode, bool smooth,
    T eps, cudaStream_t stream)
{
    // Long rows need edge workers; short rows amortize the CTA over channels.
    if (edges <= segments * 64) {
        launch_graph_softmax_double_backward_impl<T, sizeof(T) == 4 ? 128 : 64, sizeof(T) == 4 ? 256 : 128>(
            u, v, g, data, env, row_ptr, csr_int32, dg, dx, de, segments, dim, mode, smooth, eps, stream);
    } else {
        launch_graph_softmax_double_backward_impl<T, 32, 32>(
            u, v, g, data, env, row_ptr, csr_int32, dg, dx, de, segments, dim, mode, smooth, eps, stream);
    }
}

template void launch_graph_softmax_double_backward_fused<float>(
    const float*, const float*, const float*, const float*, const float*,
    const void*, bool, float*, float*, float*,
    int64_t, int64_t, int64_t, int, bool, float, cudaStream_t);
template void launch_graph_softmax_double_backward_fused<double>(
    const double*, const double*, const double*, const double*, const double*,
    const void*, bool, double*, double*, double*,
    int64_t, int64_t, int64_t, int, bool, double, cudaStream_t);
"""
