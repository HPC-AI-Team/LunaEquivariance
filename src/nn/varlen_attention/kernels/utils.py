"""Shared dtype-aware tensor addressing and FP32 accumulation for CSR attention.

Input slots follow interface.py: q, k, v, bias, grad_out, then incoming
backward gradients (rq, rk, rv, rb) and double-backward gradients
(sq, sk, sv, sb, sg). Each output tuple follows its differentiated inputs.
"""

import torch


def dtype_defines(dtype):
    if dtype == torch.float32:
        return "#define LOW_PRECISION 0\nusing Storage = float;\n"
    storage = "at::Half" if dtype == torch.float16 else "at::BFloat16"
    mma = "cutlass::half_t" if dtype == torch.float16 else "cutlass::bfloat16_t"
    return f"#include <cutlass/arch/mma_sm80.h>\n#include <cutlass/numeric_types.h>\nusing LowMmaElement = {mma};\n#define LOW_PRECISION 1\n#include <torch/extension.h>\nusing Storage = {storage};\n"


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/mma.h>
#include <cutlass/arch/mma_sm50.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <optional>
#include <vector>

using Element = float;
using ThreadMma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<1, 1, 1>, 1,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

__device__ __forceinline__ Element multiply_add(Element a, Element b, Element c) {
    cutlass::Array<Element, 1> left, right, accumulator, result;
    left[0] = a;
    right[0] = b;
    accumulator[0] = c;
    ThreadMma{}(result, left, right, accumulator);
    return result[0];
}

struct AttentionTensor {
    Storage* data = nullptr;
    bool fp32 = false;
    int64_t stride[4] = {};
    int64_t size[4] = {};

#if LOW_PRECISION
    struct Reference {
        void* pointer;
        bool fp32;
        __device__ operator float() const {
            return fp32 ? *static_cast<float*>(pointer) : float(*static_cast<Storage*>(pointer));
        }
        __device__ Reference& operator=(float value) {
            if (fp32) *static_cast<float*>(pointer) = value;
            else *static_cast<Storage*>(pointer) = Storage(value);
            return *this;
        }
        __device__ Storage* operator&() const { return static_cast<Storage*>(pointer); }
    };
    __device__ __forceinline__ Reference feature(int64_t i, int64_t j, int64_t k) const {
        int64_t offset = i * stride[0] + j * stride[1] + k * stride[2];
        return {fp32 ? static_cast<void*>(reinterpret_cast<float*>(data) + offset) :
                       static_cast<void*>(data + offset), fp32};
    }
    __device__ __forceinline__ Reference at(
        int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        int64_t offset = i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3];
        return {fp32 ? static_cast<void*>(reinterpret_cast<float*>(data) + offset) :
                       static_cast<void*>(data + offset), fp32};
    }
#else
    __device__ __forceinline__ Element& feature(int64_t i, int64_t j, int64_t k) const {
        return data[i * stride[0] + j * stride[1] + k * stride[2]];
    }
    __device__ __forceinline__ Element& at(
        int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        return data[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
    }
#endif
    // Split reductions use a float workspace even when inputs are low precision.
    __device__ __forceinline__ float& fp32_at(int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        return reinterpret_cast<float*>(data)[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
    }
    __device__ __forceinline__ Element load(int64_t row, int64_t head, int64_t channel) const {
        return data ? feature(row, head, channel) : 0.0f;
    }
    __device__ __forceinline__ void store(int64_t row, int64_t head, int64_t channel, Element value) const {
        if (data) feature(row, head, channel) = value;
    }
};

AttentionTensor make_tensor(const std::optional<at::Tensor>& tensor) {
    AttentionTensor view;
    if (tensor.has_value()) {
        view.data = reinterpret_cast<Storage*>(tensor->data_ptr());
        view.fp32 = tensor->scalar_type() == at::kFloat;
        for (int axis = 0; axis < tensor->dim(); ++axis) {
            view.stride[axis] = tensor->stride(axis);
            view.size[axis] = tensor->size(axis);
        }
    }
    return view;
}

struct AttentionState {
    float* data;
    int64_t heads;
    __device__ float at(int64_t row, int64_t head, int channel) const {
        return data[(row * heads + head) * 2 + channel];
    }
};

struct AttentionParams {
    AttentionTensor inputs[14], outputs[9];
    AttentionState state;
    const void* rowptr;
    bool csr64;
    int64_t groups, rowptr_stride;
    Element scale;

    __device__ __forceinline__ int64_t offset(int64_t group) const {
        return csr64 ? static_cast<const int64_t*>(rowptr)[group * rowptr_stride]
                     : static_cast<const int32_t*>(rowptr)[group * rowptr_stride];
    }
};

AttentionParams make_params(
    const std::vector<std::optional<at::Tensor>>& inputs,
    const std::vector<std::optional<at::Tensor>>& outputs,
    const at::Tensor& rowptr, const at::Tensor& state, double scale) {
    AttentionParams params{};
    for (size_t i = 0; i < inputs.size(); ++i) params.inputs[i] = make_tensor(inputs[i]);
    for (size_t i = 0; i < outputs.size(); ++i) params.outputs[i] = make_tensor(outputs[i]);
    params.state = {state.data_ptr<float>(), state.size(1)};
    params.rowptr = rowptr.data_ptr();
    params.csr64 = rowptr.scalar_type() == at::kLong;
    params.groups = rowptr.numel() - 1;
    params.rowptr_stride = rowptr.stride(0);
    params.scale = static_cast<Element>(scale);
    return params;
}




"""
