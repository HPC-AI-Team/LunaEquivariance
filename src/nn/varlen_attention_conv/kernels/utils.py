"""FP32 differential arithmetic and stride-aware views shared by fused kernels."""
import torch
from ....utils.extension import get_cuda_arch, load_cuda_extension

CUDA_SOURCE = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

#if defined(LOWP_BF16)
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#endif

// Value, two directional derivatives, and their mixed derivative.
// Compile-time width removes unused arithmetic from lower derivative orders.
template <int N> struct Differential {
    float x[N];
    __device__ Differential(float value = 0.f) {
        x[0] = value;
        for (int i = 1; i < N; ++i)
            x[i] = 0.f;
    }
};
template <int N> __device__ Differential<N> operator+(Differential<N> a, Differential<N> b) {
    for (int i = 0; i < N; ++i)
        a.x[i] += b.x[i];
    return a;
}
template <int N> __device__ Differential<N> operator-(Differential<N> a, Differential<N> b) {
    for (int i = 0; i < N; ++i)
        a.x[i] -= b.x[i];
    return a;
}
template <int N> __device__ Differential<N> operator*(Differential<N> a, Differential<N> b) {
    Differential<N> c(a.x[0] * b.x[0]);
    if constexpr (N >= 2)
        c.x[1] = fmaf(a.x[1], b.x[0], a.x[0] * b.x[1]);
    if constexpr (N == 4) {
        c.x[2] = fmaf(a.x[2], b.x[0], a.x[0] * b.x[2]);
        c.x[3] = fmaf(a.x[3], b.x[0], fmaf(a.x[1], b.x[2], fmaf(a.x[2], b.x[1], a.x[0] * b.x[3])));
    }
    return c;
}
template <int N> __device__ Differential<N> reciprocal(Differential<N> a) {
    float inv = 1.f / a.x[0];
    Differential<N> b(inv);
    if constexpr (N >= 2)
        b.x[1] = -a.x[1] * inv * inv;
    if constexpr (N == 4) {
        b.x[2] = -a.x[2] * inv * inv;
        b.x[3] = (2.f * a.x[1] * a.x[2] * inv - a.x[3]) * inv * inv;
    }
    return b;
}
template <int N> __device__ Differential<N> exponential(Differential<N> a) {
    float e = expf(a.x[0]);
    Differential<N> b(e);
    if constexpr (N >= 2)
        b.x[1] = e * a.x[1];
    if constexpr (N == 4) {
        b.x[2] = e * a.x[2];
        b.x[3] = e * fmaf(a.x[1], a.x[2], a.x[3]);
    }
    return b;
}
template <int N> __device__ Differential<N> warp_sum(Differential<N> a) {
    for (int offset = 16; offset; offset /= 2)
        for (int c = 0; c < N; ++c)
            a.x[c] += __shfl_xor_sync(0xffffffff, a.x[c], offset);
    return a;
}
// Only storage access depends on dtype; addressing and FP32 arithmetic are shared.
struct View {
#if defined(LOWP_BF16)
    void *data = nullptr;
    int dtype = 0;
#else
    float *data = nullptr;
#endif
    __device__ float* fp32_data() const {
#if defined(LOWP_BF16)
        assert(dtype == 0);
        return static_cast<float*>(data);
#else
        return data;
#endif
    }
    int64_t stride[4] = {}, size[4] = {};
    __device__ int64_t index(int64_t i, int64_t j = 0, int64_t k = 0, int64_t l = 0) const {
        return i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3];
    }
    __device__ float load(int64_t offset) const {
#if defined(LOWP_BF16)
        if (dtype == 0) return static_cast<const float *>(data)[offset];
        if (dtype == 1) return __half2float(static_cast<const __half *>(data)[offset]);
        return __bfloat162float(static_cast<const __nv_bfloat16 *>(data)[offset]);
#else
        return data[offset];
#endif
    }
    // Four adjacent storage values, with one aligned global load.
    __device__ float4 load4(int64_t offset) const {
#if defined(LOWP_BF16)
        if(dtype != 0) {
            uint2 bits=*reinterpret_cast<const uint2*>(static_cast<const char*>(data)+offset*2);
            float2 a,b;
            if(dtype == 1) {
                a=__half22float2(*reinterpret_cast<const __half2*>(&bits.x));
                b=__half22float2(*reinterpret_cast<const __half2*>(&bits.y));
            } else {
                a=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.x));
                b=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.y));
            }
            return {a.x,a.y,b.x,b.y};
        }
        return *reinterpret_cast<const float4*>(static_cast<const float*>(data)+offset);
#else
        return *reinterpret_cast<const float4*>(data+offset);
#endif
    }
    __device__ void store(int64_t offset, float v) const {
#if defined(LOWP_BF16)
        if (dtype == 0) static_cast<float *>(data)[offset] = v;
        else if (dtype == 1) static_cast<__half *>(data)[offset] = __float2half_rn(v);
        else static_cast<__nv_bfloat16 *>(data)[offset] = __float2bfloat16_rn(v);
#else
        data[offset] = v;
#endif
    }
    __device__ void accumulate(int64_t offset, float v) const {
#if defined(LOWP_BF16)
        assert(dtype == 0);
        static_cast<float *>(data)[offset] += v;
#else
        data[offset] += v;
#endif
    }
    __device__ void atomic_add(int64_t offset, float v) const {
#if defined(LOWP_BF16)
        assert(dtype == 0);
        atomicAdd(static_cast<float *>(data) + offset, v);
#else
        atomicAdd(data + offset, v);
#endif
    }
    __device__ float get(int64_t i, int64_t j = 0, int64_t k = 0, int64_t l = 0) const {
        return data ? load(index(i, j, k, l)) : 0.f;
    }
    __device__ void add(float v, int64_t i, int64_t j = 0, int64_t k = 0, int64_t l = 0) const {
        if (data) accumulate(index(i, j, k, l), v);
    }
};
View view(std::optional<at::Tensor> const &t) {
    View v;
    if (t) {
#if defined(LOWP_BF16)
        v.data = t->data_ptr();
        v.dtype = t->scalar_type() == at::kFloat ? 0 : (t->scalar_type() == at::kHalf ? 1 : 2);
#else
        v.data = t->data_ptr<float>();
#endif
        for (int i = 0; i < t->dim(); ++i) {
            v.stride[i] = t->stride(i);
            v.size[i] = t->size(i);
        }
    }
    return v;
}
struct Params {
    View in[19], out[12];
    const void *csr;
    bool csr64;
    int64_t csr_stride;
    float scale;
    __device__ int64_t offset(int64_t a) const {
        return csr64 ? static_cast<int64_t const *>(csr)[a * csr_stride]
                     : static_cast<int const *>(csr)[a * csr_stride];
    }
};
Params params(std::vector<std::optional<at::Tensor>> const &in,
              std::vector<std::optional<at::Tensor>> const &out, at::Tensor csr, double scale) {
    Params p{};
    for (size_t i = 0; i < in.size(); ++i)
        p.in[i] = view(in[i]);
    for (size_t i = 0; i < out.size(); ++i)
        p.out[i] = view(out[i]);
    p.csr = csr.data_ptr();
    p.csr64 = csr.scalar_type() == at::kLong;
    p.csr_stride = csr.stride(0);
    p.scale = scale;
    return p;
}
template <int N>
__device__ Differential<N> input(Params const &p, int slot, int64_t i, int64_t j = 0, int64_t k = 0,
                                 int64_t l = 0) {
    Differential<N> v(p.in[slot].get(i, j, k, l));
    if constexpr (N >= 2)
        v.x[1] = p.in[7 + slot].get(i, j, k, l);
    if constexpr (N == 4)
        v.x[2] = p.in[12 + slot].get(i, j, k, l);
    return v;
}
template <int N>
__device__ Differential<N> upstream(Params const &p, int slot, int64_t i, int h, int c) {
    Differential<N> v(p.in[slot].get(i, h, c));
    if constexpr (N == 4)
        v.x[2] = p.in[slot + 12].get(i, h, c);
    return v;
}
template <int N>
__device__ Differential<N> score(Params const &p, int64_t atom, int64_t begin, int64_t q, int64_t k,
                                 int h) {
    Differential<N> a;
    for (int c = 0; c < DIM; ++c)
        a = a + input<N>(p, 0, q, h, c) * input<N>(p, 1, k, h, c);
    a = a * Differential<N>(p.scale);
    if constexpr (BIAS_KIND == 1)
        a = a + input<N>(p, 3, k, h);
    if constexpr (BIAS_KIND == 2)
        a = a + input<N>(p, 3, atom, h, q - begin, k - begin);
    return a;
}
// Every CTA owns one complete atom, including its heads. Thus envelope gradients
// and all node/key reductions need neither atomics nor global scratch storage.
__device__ void clear_outputs(Params const &p, int slots, int64_t atom, int64_t begin,
                              int64_t end) {
    for (int slot = 0; slot < slots; ++slot) {
        View v = p.out[slot];
        if (!v.data)
            continue;
        int field = slot >= 7 ? slot - 7 : slot;
        if (field == 5) {
            for (int h = 0; h < HEADS; ++h)
                for (int c = threadIdx.x; c < VALUE_DIM; c += 32)
                    v.store(v.index(atom, h, c), 0.f);
        } else if (field == 3 && BIAS_KIND == 2) {
            int64_t count = v.size[2] * v.size[3];
            for (int h = 0; h < HEADS; ++h)
                for (int64_t i = threadIdx.x; i < count; i += 32)
                    v.store(v.index(atom, h, i / v.size[3], i % v.size[3]), 0.f);
        } else if (field == 4) {
            for (int64_t i = begin + threadIdx.x; i < end; i += 32)
                v.store(v.index(i), 0.f);
        } else {
            int width = field < 2 ? DIM : (field == 3 ? 1 : VALUE_DIM);
            for (int64_t i = begin + threadIdx.x; i < end; i += 32)
                for (int h = 0; h < HEADS; ++h)
                    for (int c = 0; c < width; ++c)
                        v.store(v.index(i, h, c), 0.f);
        }
    }
    __syncwarp();
}
"""


def load_kernel(name, source, config, device_index, dtype=torch.float32):
    dim, value_dim, heads, scale, causal, bias_kind = config
    defines = f"#define DIM {dim}\n#define VALUE_DIM {value_dim}\n#define HEADS {heads}\n#define CAUSAL {int(causal)}\n#define BIAS_KIND {bias_kind}\n"
    if dtype != torch.float32:
        defines += f"#define LOWP_BF16 {int(dtype == torch.bfloat16)}\n"
    return load_cuda_extension(
        name,
        defines + CUDA_SOURCE + source,
        arch=get_cuda_arch(torch.device("cuda", device_index)),
        use_fast_math=False,
        minimal_headers=True,
    )


def allocate(inputs, needs, accumulate=False):
    return tuple(
        torch.empty(x.shape, device=x.device, dtype=torch.float32 if accumulate else x.dtype)
        if x is not None and need
        else None
        for x, need in zip(inputs, needs)
    )


def finish_gradients(outputs, inputs):
    return tuple(y.to(x.dtype) if y is not None and y.dtype != x.dtype else y
                 for y, x in zip(outputs, inputs))
