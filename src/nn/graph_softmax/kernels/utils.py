"""Shared CUDA math and the existing LunaEquivariance extension loader."""

from functools import lru_cache

from ....utils.jit import load_extension
from ....utils.indexing import name_cuda_kernels

CPP_SOURCE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
'''


CUDA_SOURCE = r'''
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cfloat>

namespace {

constexpr int kWarpSize = 32;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
    for (int offset = 16; offset; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, value, offset);
        value = value > other ? value : other;
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T neg_inf() {
    if constexpr (sizeof(T) == sizeof(float)) return -FLT_MAX;
    else return -DBL_MAX;
}

template <typename T>
__device__ __forceinline__ T exp_op(T value) {
    if constexpr (sizeof(T) == sizeof(float)) return __expf(value);
    else return exp(value);
}

template <typename T>
struct OnlineNormalizer {
    T maximum = neg_inf<T>();
    T weighted_sum = T(0);

    __device__ __forceinline__ void add(T value, T weight) {
        const bool replace = value > maximum;
        const T ratio = exp_op(replace ? maximum - value : value - maximum);
        weighted_sum = weighted_sum * (replace ? ratio : T(1))
                     + weight * (replace ? T(1) : ratio);
        maximum = replace ? value : maximum;
    }

    __device__ __forceinline__ void merge(T other_max, T other_sum) {
        const bool replace = other_max > maximum;
        const T ratio = exp_op(replace ? maximum - other_max : other_max - maximum);
        weighted_sum = weighted_sum * (replace ? ratio : T(1))
                     + other_sum * (replace ? T(1) : ratio);
        maximum = replace ? other_max : maximum;
    }
};

// Gradient moments share the normalizer's scale and input scan.
template <typename T, int Count>
struct OnlineMoments {
    T maximum = neg_inf<T>();
    T weighted_sum = T(0);
    T moments[Count] = {};

    __device__ __forceinline__ void add(
        T value, T weight, const T (&values)[Count]) {
        const bool replace = value > maximum;
        const T ratio = exp_op(replace ? maximum - value : value - maximum);
        const T old_scale = replace ? ratio : T(1);
        const T new_scale = replace ? T(1) : ratio;
        weighted_sum = weighted_sum * old_scale + weight * new_scale;
        #pragma unroll
        for (int i = 0; i < Count; ++i) {
            moments[i] = moments[i] * old_scale + values[i] * new_scale;
        }
        maximum = replace ? value : maximum;
    }

    __device__ __forceinline__ void merge(
        T other_max, T other_sum, const T (&other_moments)[Count]) {
        const bool replace = other_max > maximum;
        const T ratio = exp_op(replace ? maximum - other_max : other_max - maximum);
        const T old_scale = replace ? ratio : T(1);
        const T new_scale = replace ? T(1) : ratio;
        weighted_sum = weighted_sum * old_scale + other_sum * new_scale;
        #pragma unroll
        for (int i = 0; i < Count; ++i) {
            moments[i] = moments[i] * old_scale + other_moments[i] * new_scale;
        }
        maximum = replace ? other_max : maximum;
    }
};

template <typename T, int EnvMode>
__device__ __forceinline__ T env_value(
    const T* env, int64_t edge, int64_t dim, int64_t embed_dim) {
    if constexpr (EnvMode == 0) return T(1);
    else if constexpr (EnvMode == 1) return env[edge];
    else return env[edge * embed_dim + dim];
}
'''


@lru_cache(maxsize=1)
def load_graph_softmax_kernel():
    from . import backward, double_backward, forward, triple_backward

    phases = (forward, backward, double_backward, triple_backward)
    binding = CPP_SOURCE + '\n'.join(phase.CPP_SOURCE for phase in phases)
    binding += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
    binding += ''.join(phase.MODULE_BINDING for phase in phases)
    binding += '}\n'
    cuda = CUDA_SOURCE + '\n'.join(phase.CUDA_SOURCE for phase in phases)
    cuda += '\n}  // namespace\n'
    cuda += '\n'.join(phase.LAUNCH_SOURCE for phase in phases)
    # T is specialized by CUDA templates; CSR width is selected at runtime.
    cuda = name_cuda_kernels(cuda, "graph_softmax", precision="")
    return load_extension(
        'lunaeq_graph_softmax', cpp_sources=binding, cuda_sources=cuda,
        extra_cflags=('-O3',), extra_cuda_cflags=('-O3', '--use_fast_math'),
        no_implicit_headers=True,
    )
