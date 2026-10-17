"""Fused third reverse derivative of envelope-weighted CSR softmax."""

CPP_SOURCE = r"""
template <typename T>
void launch_graph_softmax_triple_backward(
    const T*, const T*, const T*, const T*, const T*, const T*,
    const T*, const T*, const void*, bool,
    T*, T*, T*, T*, T*, int64_t, int64_t, int64_t, int, bool, T, cudaStream_t);

std::vector<torch::Tensor> graph_softmax_triple_backward_cuda(
    torch::Tensor wg, torch::Tensor wx, torch::Tensor we,
    torch::Tensor u, torch::Tensor v, torch::Tensor g,
    torch::Tensor data, torch::Tensor env,
    torch::Tensor row_ptr, int64_t env_mode, bool need_grad_env,
    bool smooth_out, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(data));
    auto du = torch::empty_like(u);
    auto dv = torch::empty_like(v);
    auto dg = torch::empty_like(data);
    auto dx = torch::empty_like(data);
    auto de = need_grad_env ? torch::empty_like(env) : torch::empty({0}, env.options());
    if (data.numel() == 0) return {du, dv, dg, dx, de};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define CALL_TRIPLE(T) \
    launch_graph_softmax_triple_backward( \
        wg.numel() ? wg.data_ptr<T>() : nullptr, \
        wx.numel() ? wx.data_ptr<T>() : nullptr, \
        we.numel() ? we.data_ptr<T>() : nullptr, \
        u.numel() ? u.data_ptr<T>() : nullptr, \
        v.numel() ? v.data_ptr<T>() : nullptr, g.data_ptr<T>(), \
        data.data_ptr<T>(), \
        env.numel() ? env.data_ptr<T>() : nullptr, row_ptr.data_ptr(), row_ptr.scalar_type() == torch::kInt32, \
        du.numel() ? du.data_ptr<T>() : nullptr, dv.numel() ? dv.data_ptr<T>() : nullptr, \
        dg.data_ptr<T>(), dx.data_ptr<T>(), de.numel() ? de.data_ptr<T>() : nullptr, \
        row_ptr.numel() - 1, data.size(1), data.size(0), static_cast<int>(env_mode), \
        smooth_out, static_cast<T>(eps), stream);
    if (data.scalar_type() == torch::kFloat32) {
        CALL_TRIPLE(float)
    } else {
        CALL_TRIPLE(double)
    }
#undef CALL_TRIPLE
    return {du, dv, dg, dx, de};
}
"""

MODULE_BINDING = '    m.def("graph_softmax_triple_backward_cuda", &graph_softmax_triple_backward_cuda);\n'


CUDA_SOURCE = r"""
// A, C, T are the three row moments in double backward.  Reverse their
// scalar equations, then reverse q_i = exp(x_i-m)/(sum(exp(x-m)*e)+eps).
// Smooth output only changes f=e*e, t=2*e and t'=2 in this same calculation.
template <typename T>
__device__ __forceinline__ T triple_env(
    const T* ptr, int64_t edge, int64_t offset, int mode, T absent)
{
    return ptr ? ptr[mode == 1 ? edge : offset] : absent;
}

template <typename T, int Threads>
__global__ void __launch_bounds__(Threads, 1) graph_softmax_triple_backward_kernel(
    const T* wg, const T* wx, const T* we,
    const T* u, const T* v, const T* g,
    const T* data, const T* env,
    const void* row_ptr, bool csr_int32, T* du, T* dv, T* dg, T* dx, T* de,
    int64_t dim, int env_mode, bool smooth, T eps)
{
    constexpr int Warps = Threads / 32;
    __shared__ T row_moments[9][Warps][32];
    const int64_t segment = blockIdx.x;
    const int64_t begin = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[segment]
        : static_cast<const int64_t*>(row_ptr)[segment]);
    const int64_t end = (csr_int32 ? static_cast<const int32_t*>(row_ptr)[segment + 1]
        : static_cast<const int64_t*>(row_ptr)[segment + 1]);
    if (begin == end) return;
    const bool scalar_grad = env_mode == 1 && (dv != nullptr || de != nullptr);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

    // Warps partition the row; lanes own adjacent features.
    for (int64_t feature_base = env_mode == 1 ? 0 : int64_t(blockIdx.y) * 32;
         feature_base < dim; feature_base += env_mode == 1 ? 32 : int64_t(gridDim.y) * 32) {
        const int64_t feature = feature_base + lane;
        const bool active = feature < dim;
        T maximum = T(0), inverse = T(0);
        T a = T(0), c = T(0), total = T(0);
        T k = T(0), l = T(0), m = T(0), n = T(0);
        OnlineMoments<T, 7> state;
        if (active) {
            if (end - begin == 1) {
                const int64_t off = begin * dim + feature;
                maximum = data[off];
                inverse = T(1) / (triple_env(env, begin, off, env_mode, T(1)) + eps);
            } else {
                for (int64_t edge = begin + warp; edge < end; edge += Warps) {
                const int64_t off = edge * dim + feature;
                const T e = triple_env(env, edge, off, env_mode, T(1));
                const T vv = triple_env(v, edge, off, env_mode, T(0));
                const T uu = u ? u[off] : T(0), gg = g[off];
                const T w = wg ? wg[off] : T(0), z = wx ? wx[off] : T(0);
                const T r = triple_env(we, edge, off, env_mode, T(0));
                const T f = smooth ? e * e : e;
                const T t = smooth ? T(2) * e : T(1);
                const T tp = smooth ? T(2) : T(0);
                const T b = uu * e + vv, h = uu * f + vv * t;
                const T hbar = w + z * gg;
                const T weights[7] = {
                    gg * f, b, gg * h, z * e + r,
                    z * b + r * uu, hbar * f + r * gg * t,
                    hbar * h + r * gg * (uu * t + vv * tp),
                };
                state.add(data[off], e, weights);
            }
        }
    }
    if (end - begin != 1) {
        row_moments[0][warp][lane] = state.maximum;
        row_moments[1][warp][lane] = state.weighted_sum;
        #pragma unroll
        for (int j = 0; j < 7; ++j) row_moments[j+2][warp][lane] = state.moments[j];
        __syncthreads();
        #pragma unroll
        for (int stride = Warps / 2; stride > 0; stride /= 2) {
            if (warp < stride) {
                T moments[7];
                #pragma unroll
                for (int j = 0; j < 7; ++j) moments[j] = row_moments[j+2][warp+stride][lane];
                state.merge(row_moments[0][warp+stride][lane], row_moments[1][warp+stride][lane], moments);
                row_moments[0][warp][lane] = state.maximum;
                row_moments[1][warp][lane] = state.weighted_sum;
                #pragma unroll
                for (int j = 0; j < 7; ++j) row_moments[j+2][warp][lane] = state.moments[j];
            }
            __syncthreads();
        }
        if (warp == 0) {
            const T inverse = T(1) / (state.weighted_sum + eps);
            row_moments[1][0][lane] = inverse;
            #pragma unroll
            for (int j = 0; j < 7; ++j) row_moments[j+2][0][lane] = state.moments[j] * inverse;
        }
        __syncthreads();
        maximum = row_moments[0][0][lane];
        inverse = row_moments[1][0][lane];
        a = row_moments[2][0][lane]; c = row_moments[3][0][lane];
        total = row_moments[4][0][lane]; k = row_moments[5][0][lane];
        l = row_moments[6][0][lane]; m = row_moments[7][0][lane];
        n = row_moments[8][0][lane];
    }
    const T da = T(2) * c * k - l;
    const T dc = T(2) * a * k - m;
    const T dt = -k;
    const T q_total = n - T(2) * (a * l + c * m)
                    + k * (T(6) * a * c - T(2) * total);
    for (int64_t edge = begin + warp; edge < end; edge += Warps) {
            T local_dv = T(0), local_de = T(0);
            if (active) {
                const int64_t off = edge * dim + feature;
                const T q = exp_op(data[off] - maximum) * inverse;
                const T e = triple_env(env, edge, off, env_mode, T(1));
                const T vv = triple_env(v, edge, off, env_mode, T(0));
                const T uu = u ? u[off] : T(0), gg = g[off];
                const T f = smooth ? e * e : e;
                const T t = smooth ? T(2) * e : T(1);
                const T tp = smooth ? T(2) : T(0);
                const T w = wg ? wg[off] : T(0), z = wx ? wx[off] : T(0);
                const T r = triple_env(we, edge, off, env_mode, T(0));
                if (end - begin == 1) {
                    const T p = q * e, residual = eps * q;
                    const T residual2 = residual * residual;
                    const T normal = smooth ? T(0) : T(1);
                    const T smoothed = T(1) - normal;
                    const T polynomial = T(1) - T(6) * p + T(6) * p * p;
                    const T fx = p * residual * (smooth ? e : T(1));
                    const T fxx = fx * (T(1) - T(2) * p);
                    const T fxxx = fx * polynomial;
                    const T fe = normal * q * residual + smoothed * p * (T(1) + residual);
                    const T fxe = normal * q * residual * (T(1) - T(2) * p)
                                + smoothed * T(2) * p * residual2;
                    const T fee = T(2) * q * residual * (smoothed * residual - normal * q);
                    const T fxxe = normal * q * residual * polynomial
                                 + smoothed * T(2) * p * residual2 * (T(1) - T(3) * p);
                    const T fxee = T(2) * q * residual
                                 * (smoothed * residual * (T(1) - T(3) * p)
                                    - normal * q * (T(2) - T(3) * p));
                    const T feee = T(6) * q * q * residual * (normal * q - smoothed * residual);
                    if (du) du[off] = w * fx + gg * (z * fxx + r * fxe);
                    local_dv = w * fe + gg * (z * fxe + r * fee);
                    dg[off] = z * (uu * fxx + vv * fxe) + r * (uu * fxe + vv * fee);
                    dx[off] = w * (uu * fxx + vv * fxe)
                            + gg * (z * (uu * fxxx + vv * fxxe)
                                  + r * (uu * fxxe + vv * fxee));
                    local_de = w * (uu * fxe + vv * fee)
                             + gg * (z * (uu * fxxe + vv * fxee)
                                   + r * (uu * fxee + vv * feee));
                } else {
                    const T hbar = w + z * gg;
                    const T shift = -total + T(2) * a * c - a * uu;
                    const T h = (uu - c) * f + vv * t;
                    const T j = (uu - c) * t + vv * tp;
                    const T qbar = z * (e * shift - a * vv) + hbar * h
                                 + r * (shift + gg * j) + da * gg * f
                                 + dc * (uu * e + vv) + dt * gg * (uu * f + vv * t);
                    // Factor q outside cancelling terms in all five gradients.
                    if (du) {
                        du[off] = q * (r * (-a + gg * t) - z * a * e + hbar * f
                                     + dc * e + dt * gg * f);
                    }
                    dg[off] = q * (z * h + r * j + da * f + dt * (uu * f + vv * t));
                    dx[off] = q * (qbar - e * q_total);
                    local_dv = q * (r * gg * tp - z * a + hbar * t + dc + dt * gg * t);
                    local_de = q * (z * shift + hbar * (uu - c) * t
                                 + tp * (r * gg * (uu - c) + hbar * vv)
                                 + da * gg * t + dc * uu
                                 + dt * gg * (uu * t + vv * tp) - q_total);
                }
                if (env_mode == 2) {
                    if (dv) dv[off] = local_dv;
                    if (de) de[off] = local_de;
                }
            }
            if (scalar_grad) {
                local_dv = warp_reduce_sum(local_dv);
                local_de = warp_reduce_sum(local_de);
                if (lane == 0) {
                    if (dv) dv[edge] = local_dv + (feature_base ? dv[edge] : T(0));
                    if (de) de[edge] = local_de + (feature_base ? de[edge] : T(0));
                }
            }
        }
        __syncthreads();
    }

}
"""

LAUNCH_SOURCE = r"""
template <typename T>
void launch_graph_softmax_triple_backward(
    const T* wg, const T* wx, const T* we, const T* u, const T* v, const T* g,
    const T* data, const T* env, const void* row_ptr, bool csr_int32,
    T* du, T* dv, T* dg, T* dx, T* de,
    int64_t segments, int64_t dim, int64_t edges, int env_mode, bool smooth, T eps, cudaStream_t stream)
{
    const int64_t tiles = (dim + 31) / 32;
    const dim3 grid(segments, env_mode == 1 ? 1 : min(tiles, int64_t(65535)));
    if (edges <= segments * 64) {
        graph_softmax_triple_backward_kernel<T, 256><<<grid, 256, 0, stream>>>(
            wg, wx, we, u, v, g, data, env, row_ptr, csr_int32, du, dv, dg, dx, de,
            dim, env_mode, smooth, eps);
    } else {
        constexpr int Threads = sizeof(T) == 4 ? 1024 : 512;
        graph_softmax_triple_backward_kernel<T, Threads><<<grid, Threads, 0, stream>>>(
            wg, wx, we, u, v, g, data, env, row_ptr, csr_int32, du, dv, dg, dx, de,
            dim, env_mode, smooth, eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template void launch_graph_softmax_triple_backward<float>(
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const void*, bool,
    float*, float*, float*, float*, float*, int64_t, int64_t, int64_t, int, bool, float, cudaStream_t);
template void launch_graph_softmax_triple_backward<double>(
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const void*, bool,
    double*, double*, double*, double*, double*, int64_t, int64_t, int64_t, int, bool, double, cudaStream_t);
"""
