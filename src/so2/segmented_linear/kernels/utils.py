"""Shared layout, indexing, and grouped CUDA dispatch."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from shutil import which
import subprocess

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import CUDA_HOME, get_cxx_compiler

from ....utils.extension import find_cutlass_include
from ....utils.jit import load_extension


def mix_weights(weights: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    mixed = coefficients @ weights.flatten(1)
    return mixed.view(coefficients.shape[0], weights.shape[1], weights.shape[2])


def split_flattened(tensor, sizes):
    return tuple(
        value.reshape(-1, value.shape[-1])
        for value in torch.split(tensor, sizes, dim=0)
    )


SOURCE = r"""
#include <ATen/DynamicLibrary.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

namespace {

constexpr int MaxGroups = 64;
enum class Product : int { Project, Input, Weight };

struct SegmentStarts {
    int64_t rows[MaxGroups];
};

template <bool Weight>
__global__ void make_group_pointers(
    const float* a, const float* b, float* c, int64_t* arrays,
    SegmentStarts starts, int groups, int64_t a_step, int64_t b_step,
    int64_t c_step
) {
    int group = threadIdx.x;
    if (group >= groups) return;
    int64_t offset = starts.rows[group];
    arrays[group] = reinterpret_cast<int64_t>(a + (Weight ? offset : group) * a_step);
    arrays[groups + group] = reinterpret_cast<int64_t>(b + offset * b_step);
    arrays[2 * groups + group] = reinterpret_cast<int64_t>(c + (Weight ? group : offset) * c_step);
}

using GroupedGemm = cublasStatus_t (CUBLASWINAPI *)(
    cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*,
    const int*, const int*, const int*, const void*, const void* const*,
    cudaDataType_t, const int*, const void* const*, cudaDataType_t,
    const int*, const void*, void* const*, cudaDataType_t, const int*,
    int, const int*, cublasComputeType_t
);

at::DynamicLibrary& cublas_library() {
    // Use the CUDA runtime already loaded with Torch, including for CUDA Graphs.
    static at::DynamicLibrary library("libcublas.so.12", nullptr, true);
    return library;
}

GroupedGemm grouped_gemm() {
    static auto function = reinterpret_cast<GroupedGemm>(
        cublas_library().sym("cublasGemmGroupedBatchedEx")
    );
    return function;
}

void grouped(
    const at::Tensor& left, const at::Tensor& right, at::Tensor& output,
    const std::vector<int64_t>& sizes, int64_t product, bool add
) {
    TORCH_CHECK(left.is_cuda() && right.device() == left.device() &&
                output.device() == left.device());
    TORCH_CHECK(left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat &&
                output.scalar_type() == at::kFloat);
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(!output.is_neg(), "Grouped output must not be a negative view");
    TORCH_CHECK(product >= 0 && product <= static_cast<int>(Product::Weight));
    bool project = static_cast<Product>(product) == Product::Project;
    bool weight = static_cast<Product>(product) == Product::Weight;
    TORCH_CHECK(left.dim() == 2 && right.dim() == (weight ? 2 : 3) && output.dim() == (weight ? 3 : 2));
    int groups = sizes.size();
    TORCH_CHECK(groups > 0 && groups <= MaxGroups);
    int64_t cin = weight ? left.size(1) : right.size(2);
    int64_t cout = right.size(1);
    TORCH_CHECK(cin > 0 && cin < INT_MAX && cout > 0 && cout < INT_MAX);
    if (weight) {
        TORCH_CHECK(!add, "Grouped weight gradients do not support accumulation");
        TORCH_CHECK(!left.is_neg() && !right.is_neg());
        TORCH_CHECK(!output.is_alias_of(left) && !output.is_alias_of(right),
                    "Grouped weight output must not alias its inputs");
        TORCH_CHECK(right.size(0) == left.size(0) && output.size(0) == groups &&
                    output.size(1) == cout && output.size(2) == cin);
    } else {
        TORCH_CHECK(right.size(0) == groups && left.size(1) == (project ? cin : cout));
        TORCH_CHECK(output.size(0) == left.size(0) && output.size(1) == (project ? cout : cin));
    }

    SegmentStarts starts{};
    std::array<int, MaxGroups> m, count, inner, lda, ldb, ldc, multiplicity;
    std::array<float, MaxGroups> alpha, beta;
    std::array<cublasOperation_t, MaxGroups> trans_a, trans_b;
    int64_t rows = 0;
    for (int group = 0; group < groups; ++group) {
        int64_t size = sizes[group];
        TORCH_CHECK(size > 0 && size < INT_MAX);
        starts.rows[group] = rows;
        rows += size;
        // The column-major weight product is dW^T = X^T dY.
        m[group] = project ? cout : cin;
        count[group] = weight ? cout : size;
        inner[group] = weight ? size : (project ? cin : cout);
        trans_a[group] = project ? CUBLAS_OP_T : CUBLAS_OP_N;
        trans_b[group] = weight ? CUBLAS_OP_T : CUBLAS_OP_N;
        lda[group] = cin;
        ldb[group] = project ? cin : cout;
        ldc[group] = m[group];
        multiplicity[group] = 1;
        alpha[group] = 1.0f;
        beta[group] = add ? 1.0f : 0.0f;
    }
    TORCH_CHECK(rows == left.size(0));

    c10::cuda::CUDAGuard guard(left.device());
    auto left_values = left.is_neg() ? left.resolve_neg() : left;
    auto right_values = right.is_neg() ? right.resolve_neg() : right;
    auto stream = c10::cuda::getCurrentCUDAStream(left.get_device()).stream();
    auto pointers = at::empty({3, groups}, left.options().dtype(at::kLong));
    auto arrays = pointers.data_ptr<int64_t>();
    if (weight) {
        make_group_pointers<true><<<1, MaxGroups, 0, stream>>>(
            left_values.data_ptr<float>(), right_values.data_ptr<float>(),
            output.data_ptr<float>(), arrays, starts, groups, cin, cout, cin * cout
        );
    } else {
        make_group_pointers<false><<<1, MaxGroups, 0, stream>>>(
            right_values.data_ptr<float>(), left_values.data_ptr<float>(),
            output.data_ptr<float>(), arrays, starts, groups, cin * cout, left.size(1), output.size(1)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto status = grouped_gemm()(
        handle, trans_a.data(), trans_b.data(), m.data(), count.data(), inner.data(),
        alpha.data(), reinterpret_cast<const void* const*>(arrays), CUDA_R_32F, lda.data(),
        reinterpret_cast<const void* const*>(arrays + groups), CUDA_R_32F, ldb.data(),
        beta.data(), reinterpret_cast<void* const*>(arrays + 2 * groups), CUDA_R_32F, ldc.data(),
        groups, multiplicity.data(), CUBLAS_COMPUTE_32F_PEDANTIC
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Grouped cuBLAS status: ", int(status));
}

using Stream = c10::cuda::CUDAStream;
constexpr int StreamLanes = 4;
constexpr int64_t StreamWorkspace = 256 * 1024;

struct StreamBlas {
    decltype(&cublasSetStream) stream = symbol<decltype(&cublasSetStream)>("cublasSetStream_v2");
    decltype(&cublasSetWorkspace) workspace = symbol<decltype(&cublasSetWorkspace)>("cublasSetWorkspace_v2");
    decltype(&cublasSgemm) gemm = symbol<decltype(&cublasSgemm)>("cublasSgemm_v2");

    template <typename F>
    F symbol(const char* name) {
        return reinterpret_cast<F>(cublas_library().sym(name));
    }
};

const StreamBlas& stream_blas() {
    static const StreamBlas functions;
    return functions;
}

void check_blas(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Streamed cuBLAS status: ", int(status));
}

template <typename F>
void release_on_device(c10::DeviceIndex device, F&& release) noexcept {
    // CUDA may already be shutting down when the owner thread exits.
    int current;
    if (cudaGetDevice(&current) != cudaSuccess) return;
    if (current != device && cudaSetDevice(device) != cudaSuccess) return;
    release();
    if (current != device) (void)cudaSetDevice(current);
}

struct Event {
    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    ~Event() {
        if (handle) release_on_device(device, [&] { (void)cudaEventDestroy(handle); });
    }

    void record(Stream stream) {
        if (!handle) {
            device = stream.device_index();
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&handle, cudaEventDisableTiming));
        }
        C10_CUDA_CHECK(cudaEventRecord(handle, stream));
    }

    void block(Stream stream) const {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, handle, 0));
    }

    c10::DeviceIndex device = -1;
    cudaEvent_t handle = nullptr;
};

struct Worker {
    explicit Worker(c10::DeviceIndex device) : device(device) {
        C10_CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
    }

    ~Worker() {
        release_on_device(device, [&] { (void)cudaStreamDestroy(handle); });
    }

    Stream stream() const {
        return c10::cuda::getStreamFromExternal(handle, device);
    }

    c10::DeviceIndex device;
    cudaStream_t handle = nullptr;
    Event done;
};

struct Pool {
    Event start;
    std::vector<std::unique_ptr<Worker>> workers;
};

struct RestoreHandle {
    ~RestoreHandle() {
        if (active) {
            try { (void)at::cuda::getCurrentCUDABlasHandle(); } catch (...) {}
        }
    }

    void restore() {
        if (active) {
            (void)at::cuda::getCurrentCUDABlasHandle();
            active = false;
        }
    }

    bool active;
};

thread_local std::unordered_map<c10::DeviceIndex, Pool> pools;

void streamed_weight_gradient(
    const at::Tensor& x, const at::Tensor& dy, const at::Tensor& output,
    const std::vector<int64_t>& sizes
) {
    TORCH_CHECK(x.is_cuda() && dy.device() == x.device() && output.device() == x.device());
    TORCH_CHECK(x.scalar_type() == at::kFloat && dy.scalar_type() == at::kFloat &&
                output.scalar_type() == at::kFloat);
    TORCH_CHECK(x.is_contiguous() && dy.is_contiguous() && output.is_contiguous() &&
                !x.is_neg() && !dy.is_neg() && !output.is_neg());
    TORCH_CHECK(x.dim() == 2 && dy.dim() == 2 && output.dim() == 3);
    TORCH_CHECK(output.size(0) == sizes.size() && output.size(1) == dy.size(1) &&
                output.size(2) == x.size(1) && x.size(0) == dy.size(0));
    TORCH_CHECK(!output.is_alias_of(x) && !output.is_alias_of(dy));
    TORCH_CHECK(sizes.size() >= StreamLanes && sizes.size() <= MaxGroups);
    int lane_count = StreamLanes;

    constexpr auto max_int = std::numeric_limits<int>::max();
    int64_t cin = x.size(1);
    int64_t cout = dy.size(1);
    TORCH_CHECK(cin > 0 && cout > 0 && cin <= max_int && cout <= max_int);
    std::vector<int64_t> starts(sizes.size());
    int64_t count = 0;
    for (size_t s = 0; s < sizes.size(); ++s) {
        TORCH_CHECK(sizes[s] > 0 && sizes[s] <= max_int);
        starts[s] = count;
        count += sizes[s];
    }
    TORCH_CHECK(count == x.size(0));

    auto device = x.get_device();
    c10::cuda::CUDAGuard guard(device);
    auto parent = c10::cuda::getCurrentCUDAStream(device);
    auto& pool = pools[device];
    if (pool.workers.size() < size_t(lane_count - 1)) {
        cudaStreamCaptureStatus capturing;
        C10_CUDA_CHECK(cudaStreamIsCapturing(parent, &capturing));
        if (capturing != cudaStreamCaptureStatusNone) {
            // A caller can capture without first warming this thread's pool.
            lane_count = 1;
        } else {
            while (pool.workers.size() < size_t(lane_count - 1)) {
                pool.workers.push_back(std::make_unique<Worker>(device));
            }
        }
    }

    std::vector<int> order(sizes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&sizes](int a, int b) { return sizes[a] > sizes[b]; });
    std::vector<int64_t> loads(lane_count, 0);
    std::vector<std::vector<int>> lanes(lane_count);
    for (int segment : order) {
        int lane = std::min_element(loads.begin(), loads.end()) - loads.begin();
        lanes[lane].push_back(segment);
        loads[lane] += sizes[segment] + 128;
    }

    auto x_ptr = x.data_ptr<float>();
    auto dy_ptr = dy.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    // Allocating on the parent stream preserves scratch lifetime across graph replays.
    at::Tensor scratch;
    if (lane_count > 1) {
        scratch = at::empty({lane_count - 1, StreamWorkspace}, x.options().dtype(at::kByte));
    }
    const float alpha = 1.f, beta = 0.f;
    auto contract = [&](cublasHandle_t handle, const std::vector<int>& indices) {
        for (int s : indices) {
            auto row = starts[s];
            // The column-major cuBLAS view computes dW^T = X^T dY.
            auto status = stream_blas().gemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, cin, cout, sizes[s],
                &alpha, x_ptr + row * cin, cin, dy_ptr + row * cout, cout,
                &beta, out_ptr + s * cout * cin, cin);
            check_blas(status);
        }
    };

    // Each worker waits for inputs on the caller stream, and the caller
    // waits for every worker before consuming or reusing its allocations.
    if (lane_count > 1) pool.start.record(parent);
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    contract(handle, lanes.front());
    RestoreHandle restore{lane_count > 1};
    for (int lane = 1; lane < lane_count; ++lane) {
        auto& worker = *pool.workers[lane - 1];
        auto stream = worker.stream();
        pool.start.block(stream);
        try {
            auto workspace = scratch.data_ptr<uint8_t>() + (lane - 1) * StreamWorkspace;
            check_blas(stream_blas().stream(handle, stream));
            check_blas(stream_blas().workspace(handle, workspace, StreamWorkspace));
            contract(handle, lanes[lane]);
        } catch (...) {
            worker.done.record(stream);
            worker.done.block(parent);
            throw;
        }
        worker.done.record(stream);
        worker.done.block(parent);
    }
    restore.restore();
}

constexpr int WeightChunk = 1024;
constexpr int MaxWeightTasks = 128;
constexpr int MaxWeightReductions = 32;

struct WeightPlan {
    int tasks[MaxWeightTasks][4];
    int reductions[MaxWeightReductions][3];
};

__global__ void pack_weight_plan(
    int* tasks, int* reductions, WeightPlan plan, int task_count, int reduction_count
) {
    int lane = threadIdx.x;
    for (int i = lane; i < 4 * task_count; i += blockDim.x) {
        tasks[i] = plan.tasks[i / 4][i % 4];
    }
    for (int i = lane; i < 3 * reduction_count; i += blockDim.x) {
        reductions[i] = plan.reductions[i / 3][i % 3];
    }
}

std::tuple<at::Tensor, at::Tensor, int> weight_plan(
    const at::Tensor& x, const at::Tensor& dy, const at::Tensor& output,
    const std::vector<int64_t>& sizes
) {
    TORCH_CHECK(x.is_cuda() && x.device() == dy.device() && x.device() == output.device(),
                "Grouped weight tensors must share one CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && dy.scalar_type() == at::kFloat &&
                output.scalar_type() == at::kFloat, "Grouped weight tensors must be FP32");
    TORCH_CHECK(x.is_contiguous() && dy.is_contiguous() && output.is_contiguous() &&
                !x.is_neg() && !dy.is_neg() && !output.is_neg(),
                "Grouped weight tensors must be contiguous with resolved values");
    TORCH_CHECK(!output.is_alias_of(x) && !output.is_alias_of(dy),
                "Grouped weight output must not alias its inputs");
    TORCH_CHECK(x.dim() == 2 && dy.dim() == 2 && output.dim() == 3,
                "Invalid grouped weight ranks");
    TORCH_CHECK(x.size(0) == dy.size(0) && output.size(0) == sizes.size() &&
                output.size(1) == dy.size(1) && output.size(2) == x.size(1),
                "Invalid grouped weight dimensions");
    const bool paired = x.size(1) != 768;
    TORCH_CHECK((!paired && dy.size(1) == 640) ||
                (paired && x.size(1) == dy.size(1) &&
                 (x.size(1) == 512 || x.size(1) == 256)),
                "Unsupported grouped weight channels");
    TORCH_CHECK(sizes.size() >= 16 && sizes.size() <= MaxGroups && x.size(0) <= 65536,
                "Unsupported grouped weight row or group count");
    int factor = paired ? 2 : 1;
    TORCH_CHECK(x.size(0) % factor == 0 && x.size(0) <= 32768 * factor,
                "Invalid grouped weight row count");

    WeightPlan plan{};
    int row = 0, tasks = 0, reductions = 0, partials = 0;
    for (int group = 0; group < int(sizes.size()); ++group) {
        TORCH_CHECK(sizes[group] >= 32 * factor && sizes[group] <= 4608 * factor &&
                    sizes[group] % factor == 0, "Unsupported grouped weight segment length");
        int count = sizes[group];
        bool split = paired && count >= 2 * WeightChunk;
        int parts = split ? (count + WeightChunk - 1) / WeightChunk : 1;
        TORCH_CHECK(tasks + parts <= MaxWeightTasks, "Grouped weight plan has too many tasks");
        for (int part = 0; part < parts; ++part) {
            int offset = part * WeightChunk;
            int* task = plan.tasks[tasks++];
            task[0] = row + offset;
            task[1] = split ? std::min(count - offset, WeightChunk) : count;
            task[2] = split ? partials + part : group;
            task[3] = split;
        }
        if (split) {
            TORCH_CHECK(reductions < MaxWeightReductions, "Grouped weight plan has too many reductions");
            int* reduction = plan.reductions[reductions++];
            reduction[0] = group;
            reduction[1] = partials;
            reduction[2] = parts;
            partials += parts;
        }
        row += count;
    }
    TORCH_CHECK(row == x.size(0), "Segment lengths do not partition grouped weight rows");

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    auto table = at::empty({4 * tasks + 3 * reductions}, x.options().dtype(at::kInt));
    auto task_table = table.narrow(0, 0, 4 * tasks).view({tasks, 4});
    auto reduction_table = table.narrow(0, 4 * tasks, 3 * reductions).view({reductions, 3});
    pack_weight_plan<<<1, 128, 0, stream>>>(
        task_table.data_ptr<int>(), reduction_table.data_ptr<int>(), plan, tasks, reductions
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {task_table, reduction_table, partials};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("grouped", &grouped);
    module.def("weight_plan", &weight_plan);
    module.def("streamed_weight_gradient", &streamed_weight_gradient);
}
"""

DIRECT_GROUPED_SOURCE = r"""
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {
constexpr int MaxGroups = 64;
constexpr int TileM = 64;
constexpr int TileN = 128;

struct Rows {
  int prefix[MaxGroups + 1];
  int tiles[MaxGroups + 1];
  int groups;
};

template <class LayoutB>
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TileM, TileN, 8>, cutlass::gemm::GemmShape<TileM / 2, TileN / 2, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2>::GemmKernel;

// Bound each tail tile to its segment so the input and output iterators mask the same rows.
template <int Cin, int Cout, bool InputVjp, bool Ragged>
__global__ void direct_cutlass(const float* a, const float* weights, float* result,
                               const __grid_constant__ Rows rows, int total_rows) {
  using LayoutB = std::conditional_t<InputVjp, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using Operation = Kernel<LayoutB>;
  using Mma = typename Operation::Mma;
  using Epilogue = typename Operation::Epilogue;
  constexpr int K = InputVjp ? Cout : Cin;
  constexpr int N = InputVjp ? Cin : Cout;
  union Storage {
    typename Mma::SharedStorage mma;
    typename Epilogue::SharedStorage epilogue;
  };
  __shared__ Storage storage;
  int first_row = blockIdx.x * TileM;
  int first_col = blockIdx.y * TileN;
  int segment = 0;
  const float* input = a;
  if constexpr (Ragged) {
    int end = rows.groups;
    while (segment + 1 < end) {
      int middle = (segment + end) / 2;
      if (rows.tiles[middle] <= blockIdx.x) segment = middle;
      else end = middle;
    }
    int start = rows.prefix[segment];
    first_row = (blockIdx.x - rows.tiles[segment]) * TileM;
    total_rows = rows.prefix[segment + 1] - start;
    input += static_cast<int64_t>(start) * K;
    result += static_cast<int64_t>(start) * N;
  } else {
    while (first_row >= rows.prefix[segment + 1]) ++segment;
  }
  auto* weight = const_cast<float*>(weights + segment * Cout * Cin);

  int thread = threadIdx.x;
  int warp = cutlass::canonical_warp_idx_sync();
  int lane = thread % 32;
  typename Mma::IteratorA lhs(cutlass::layout::RowMajor(K), const_cast<float*>(input),
                               {total_rows, K}, thread, {first_row, 0});
  typename Mma::IteratorB rhs(LayoutB(Cin), weight, {K, N}, thread, {0, first_col});
  typename Mma::FragmentC accum;
  accum.clear();
  Mma mma(storage.mma, thread, warp, lane);
  mma(K / Mma::Shape::kK, accum, lhs, rhs, accum);

  __syncthreads();
  typename Epilogue::OutputTileIterator out(
      typename Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor(N)),
      result, {total_rows, N}, thread, {first_row, first_col});
  Epilogue epilogue(storage.epilogue, thread, warp, lane);
  epilogue(typename Epilogue::OutputOp(1.0f, 0.0f), out, accum, out);
}

template <int Cin, int Cout, bool Ragged>
void launch_direct(const float* a, const float* weight, float* out, Rows rows,
                  int total_rows, bool input_vjp, cudaStream_t stream) {
  dim3 grid(Ragged ? rows.tiles[rows.groups] : total_rows / TileM,
            (input_vjp ? Cin : Cout) / TileN);
  if (input_vjp) {
    using Layout = cutlass::layout::RowMajor;
    direct_cutlass<Cin, Cout, true, Ragged><<<grid, Kernel<Layout>::kThreadCount, 0, stream>>>(
        a, weight, out, rows, total_rows);
  } else {
    using Layout = cutlass::layout::ColumnMajor;
    direct_cutlass<Cin, Cout, false, Ragged><<<grid, Kernel<Layout>::kThreadCount, 0, stream>>>(
        a, weight, out, rows, total_rows);
  }
}

template <int Cin, int Cout>
void launch_projection(const float* a, const float* weight, float* out, Rows rows,
                       int total_rows, bool input_vjp, cudaStream_t stream) {
  if (rows.groups <= 32 && rows.tiles[rows.groups] * TileM == total_rows) {
    launch_direct<Cin, Cout, false>(a, weight, out, rows, total_rows, input_vjp, stream);
  } else {
    launch_direct<Cin, Cout, true>(a, weight, out, rows, total_rows, input_vjp, stream);
  }
}

void grouped_direct(const at::Tensor& left, const at::Tensor& weights, at::Tensor& out,
                    const std::vector<int64_t>& sizes, bool input_vjp) {
  TORCH_CHECK(!sizes.empty() && sizes.size() <= MaxGroups);
  TORCH_CHECK(left.size(0) <= 61440 && left.size(0) == out.size(0));
  Rows rows{};
  rows.groups = sizes.size();
  for (int group = 0; group < rows.groups; ++group) {
    auto count = sizes[group];
    TORCH_CHECK(count > 0 && count <= left.size(0) - rows.prefix[group],
                "Invalid grouped projection segment size");
    rows.prefix[group + 1] = rows.prefix[group] + count;
    rows.tiles[group + 1] = rows.tiles[group] + (count + TileM - 1) / TileM;
  }
  TORCH_CHECK(rows.prefix[rows.groups] == left.size(0), "Segment sizes must partition the input");
  c10::cuda::CUDAGuard guard(left.device());
  auto stream = c10::cuda::getCurrentCUDAStream(left.get_device()).stream();
  const float* a = left.data_ptr<float>();
  const float* b = weights.data_ptr<float>();
  float* result = out.data_ptr<float>();
  if (weights.size(1) == 512 && weights.size(2) == 512) {
    launch_projection<512, 512>(a, b, result, rows, left.size(0), input_vjp, stream);
  } else if (weights.size(1) == 256 && weights.size(2) == 256) {
    launch_projection<256, 256>(a, b, result, rows, left.size(0), input_vjp, stream);
  } else {
    TORCH_CHECK(weights.size(1) == 640 && weights.size(2) == 768);
    launch_projection<768, 640>(a, b, result, rows, left.size(0), input_vjp, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("grouped_direct", &grouped_direct);
}
"""


PROJECT, INPUT, WEIGHT, BLAS_WEIGHT = range(4)


@lru_cache(maxsize=1)
def build_grouped():
    return load_extension(
        "luna_segmented_grouped", cuda_sources=SOURCE,
        extra_cuda_cflags=("-O3", "-gencode=arch=compute_80,code=sm_80",
                           "-gencode=arch=compute_90,code=sm_90"),
        no_implicit_headers=True,
    )


@lru_cache(maxsize=1)
def grouped_direct_include():
    try:
        path = find_cutlass_include()
    except RuntimeError:
        return None
    return path if (path / "cutlass/gemm/kernel/default_gemm_grouped.h").is_file() else None


@lru_cache(maxsize=1)
def build_grouped_direct(include_path):
    return load_extension(
        "luna_segmented_direct", cuda_sources=DIRECT_GROUPED_SOURCE,
        extra_cuda_cflags=(
            "-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90,code=sm_90",
        ),
        extra_include_paths=(include_path,), no_implicit_headers=True,
    ).grouped_direct


grouped_direct_launch = None


def use_grouped_direct(left, right, output, sizes, product, add):
    groups = len(sizes)
    if (product not in (PROJECT, INPUT) or add or not 16 <= groups <= 64
            or right.ndim != 3 or right.shape[0] != groups or left.ndim != 2):
        return False
    cout, cin = right.shape[1:]
    if (cout, cin) == (512, 512):
        min_rows, max_rows = 16384, 61440
    elif (cout, cin) == (256, 256):
        min_rows, max_rows = 30720, 61440
    elif (cout, cin) == (640, 768):
        min_rows, max_rows = 15360, 30720
    else:
        return False
    row_count, inputs = left.shape
    expected_in, expected_out = (cin, cout) if product == PROJECT else (cout, cin)
    if (not min_rows <= row_count <= max_rows or inputs != expected_in
            or output.shape != (row_count, expected_out) or sum(sizes) != row_count
            or any(size <= 0 for size in sizes)):
        return False
    # Too many partial tiles erase the gain from direct dispatch.
    padded_rows = sum((size + 63) // 64 for size in sizes) * 64
    if padded_rows * 100 > 112 * row_count:
        return False
    return (
        left.is_cuda and torch.version.hip is None
        and torch.cuda.get_device_capability(left.device) in ((8, 0), (9, 0))
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and all(value.device == left.device and value.dtype == torch.float32
                and value.is_contiguous() and not value.is_neg() for value in (left, right, output))
        and not torch._C._is_alias_of(output, left) and not torch._C._is_alias_of(output, right)
    )


def run_grouped_direct(left, right, output, sizes, product):
    global grouped_direct_launch
    launch = grouped_direct_launch
    if launch is None:
        # A first use inside capture can use the already configured cuBLAS route.
        with torch.cuda.device(left.device):
            if torch.cuda.is_current_stream_capturing():
                return False
        include_path = grouped_direct_include()
        if include_path is None:
            return False
        launch = grouped_direct_launch = build_grouped_direct(include_path)
    launch(left, right, output, sizes, product == INPUT)
    return True


@lru_cache(maxsize=1)
def toolkit_available():
    version = torch.version.cuda
    if version is None or CUDA_HOME is None:
        return False
    if not which("ninja") or not which(get_cxx_compiler()):
        return False
    runtime = tuple(map(int, version.split(".")[:2]))
    if not (12, 5) <= runtime < (13, 0):
        return False
    toolkit = Path(CUDA_HOME)
    compiler = toolkit / "bin/nvcc"
    if not compiler.is_file():
        return False
    # Use a toolkit compatible with the grouped cuBLAS major version.
    metadata = toolkit / "version.json"
    if metadata.is_file():
        try:
            toolkit_version = json.loads(metadata.read_text())["cuda"]["version"]
            return (12, 5) <= tuple(map(int, toolkit_version.split(".")[:2])) < (13, 0)
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            pass
    try:
        compiler_version = subprocess.check_output((str(compiler), "--version"), text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    release = re.search(r"release (\d+)\.(\d+)", compiler_version)
    return release is not None and (12, 5) <= tuple(map(int, release.groups())) < (13, 0)


@torch.compiler.assume_constant_result
def grouped_device(device):
    return toolkit_available() and torch.cuda.get_device_capability(device) in ((8, 0), (9, 0))


def use_equal_segment_bmm(x, mixed, sizes, *, large=False):
    rows_per_segment = (16384, 32768) if large else (2048, 4096, 8192)
    return (
        len(sizes) == 4 and sizes[0] in rows_per_segment
        and all(size == sizes[0] for size in sizes)
        and x.shape == (4 * sizes[0], 64) and mixed.shape == (4, 64, 64)
        and x.is_cuda and torch.version.hip is None and mixed.device == x.device
        and x.dtype == mixed.dtype == torch.float32
        and x.is_contiguous() and mixed.is_contiguous()
        and not torch.compiler.is_compiling()
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.is_autocast_enabled("cuda") and not torch.are_deterministic_algorithms_enabled()
        and (
            torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))
            and not x.is_neg() and not mixed.is_neg()
            if large else streamed_weight_device(x.device)
        )
    )


def use_grouped(x, mixed, sizes, *, input_gradient=False):
    """Use grouping where it amortizes launch and pointer setup."""
    groups = len(sizes)
    if groups == 4:
        return use_equal_segment_bmm(x, mixed, sizes)
    edges = x.shape[0]
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or mixed.dtype != torch.float32
        or mixed.device != x.device
        or not mixed.is_contiguous()
        or not x.is_contiguous()
        or not 16 <= groups <= 64
        or not 80 * groups <= edges <= 960 * groups
        or edges > 32768
        or min(sizes) < 32
        or max(sizes) > 4608
        or torch.backends.cuda.matmul.allow_tf32
        or torch.is_autocast_enabled("cuda")
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    shape = x.ndim, *mixed.shape[1:]
    if shape not in (
        (2, 640, 768), (3, 512, 512), (3, 256, 256),
        (2, 384, 384), (3, 512, 256), (3, 256, 128),
    ):
        return False
    if (x.ndim == 3 and x.shape[1] != 2) or x.shape[-1] != mixed.shape[1 if input_gradient else 2]:
        return False
    return grouped_device(x.device)


@torch.compiler.assume_constant_result
def grouped_weight_device(device):
    return torch.cuda.get_device_capability(device) == (8, 0)


@torch.compiler.assume_constant_result
def streamed_weight_device(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def use_hopper_weight_tiles(left, right, sizes):
    if left.ndim != 2 or right.ndim != 2 or not 16 <= len(sizes) <= 64:
        return False
    shape = right.shape[1], left.shape[1]
    if shape not in ((640, 768), (512, 512), (256, 256)):
        return False
    planes = 1 if shape == (640, 768) else 2
    return left.shape[0] <= 32768 * planes and all(
        32 * planes <= size <= 4608 * planes and size % planes == 0 for size in sizes
    )


def use_hopper_blas_weight(left, right, sizes):
    if (
        not 16 <= len(sizes) <= 32 or left.ndim != 2 or right.ndim != 2
        or torch.compiler.is_compiling()
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled("cuda")
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    shape = right.shape[1], left.shape[1]
    if shape not in ((640, 768), (512, 512)):
        return False
    planes = 1 if shape == (640, 768) else 2
    return streamed_weight_device(left.device) and all(
        400 * planes <= size <= 1900 * planes and size % planes == 0 for size in sizes
    )


@triton.jit
def grouped_ieee_kernel(
    dy, x, output, partial, tasks, O: tl.constexpr, I: tl.constexpr,
    BO: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
):
    task = tl.program_id(1)
    start = tl.load(tasks + task * 4)
    count = tl.load(tasks + task * 4 + 1)
    dest = tl.load(tasks + task * 4 + 2)
    pointer = output
    if SPLIT:
        is_split = tl.load(tasks + task * 4 + 3)
        pointer = tl.where(is_split != 0, partial, output)

    tile = tl.program_id(0)
    tiles_i: tl.constexpr = (I + BI - 1) // BI
    o = (tile // tiles_i) * BO + tl.arange(0, BO)
    i = (tile % tiles_i) * BI + tl.arange(0, BI)
    k = tl.arange(0, BK)
    acc = tl.full((BO, BI), 0, tl.float32)
    for block in range(tl.cdiv(count, BK)):
        pos = block * BK + k
        row = start + pos
        lhs = tl.load(dy + row[None, :] * O + o[:, None],
                      (o[:, None] < O) & (pos[None, :] < count), 0.)
        rhs = tl.load(x + row[:, None] * I + i[None, :],
                      (pos[:, None] < count) & (i[None, :] < I), 0.)
        acc = tl.dot(lhs, rhs, acc, input_precision="ieee")
    tl.store(pointer + (dest * O + o[:, None]) * I + i[None, :],
             acc, (o[:, None] < O) & (i[None, :] < I))


@triton.jit
def reduce_weight_kernel(
    partial, output, reduction, width: tl.constexpr,
    split_count: tl.constexpr, block: tl.constexpr,
):
    rid = tl.program_id(1)
    group = tl.load(reduction + rid * 3)
    start = tl.load(reduction + rid * 3 + 1)
    count = tl.load(reduction + rid * 3 + 2)
    split = tl.arange(0, split_count)
    col = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(partial + (start + split[:, None]) * width + col[None, :],
                    (split[:, None] < count) & (col[None, :] < width), 0.)
    tl.store(output + group * width + col, tl.sum(value, axis=0), col < width)


def grouped_weight(x, dy, output, tasks, reductions, partial_count, *, output_tile=32):
    _, out_width, in_width = output.shape
    with torch.cuda.device(x.device):
        partial = x.new_empty((partial_count, out_width, in_width)) if partial_count else output
        block_o, block_i, block_k = output_tile, 64, 16
        tiles = triton.cdiv(out_width, block_o) * triton.cdiv(in_width, block_i)
        grouped_ieee_kernel[(tiles, tasks.shape[0])](
            dy, x, output, partial, tasks, out_width, in_width, block_o, block_i,
            block_k, bool(partial_count), num_warps=4, num_stages=2,
        )
        if reductions.shape[0]:
            width = out_width * in_width
            # Paired segments up to 4608 edges have at most nine row tiles.
            reduce_weight_kernel[(triton.cdiv(width, 128), reductions.shape[0])](
                partial, output, reductions, width, 16, 128, num_warps=4,
            )


@torch.library.custom_op("lunaeq::segmented_grouped_gemm", mutates_args=("output",),
                         device_types="cuda")
def grouped_gemm_cuda(left: torch.Tensor, right: torch.Tensor, output: torch.Tensor,
                      sizes: list[int], product: int, add: bool) -> None:
    if use_grouped_direct(left, right, output, sizes, product, add):
        if run_grouped_direct(left, right, output, sizes, product):
            return
    extension = build_grouped()
    if product == BLAS_WEIGHT:
        extension.grouped(left, right, output, sizes, WEIGHT, add)
    elif product == WEIGHT:
        if add:
            raise ValueError("Grouped weight gradients do not support accumulation")
        hopper = streamed_weight_device(left.device)
        tiled = hopper and use_hopper_weight_tiles(left, right, sizes)
        if hopper and not tiled:
            extension.streamed_weight_gradient(left, right, output, sizes)
        else:
            tasks, reductions, partials = extension.weight_plan(left, right, output, sizes)
            grouped_weight(
                left, right, output, tasks, reductions, partials,
                output_tile=64 if tiled else 32,
            )
    else:
        extension.grouped(left, right, output, sizes, product, add)


@grouped_gemm_cuda.register_fake
def grouped_gemm_fake(left, right, output, sizes, product, add):
    return None


def use_uniform_node_gemm(left, right, output, sizes, product, add):
    if (
        product not in (PROJECT, INPUT) or add or torch.compiler.is_compiling()
        or len(sizes) != 16 or right.ndim != 3 or right.shape[0] != len(sizes)
        or right.shape[1] != 128 or (right.shape[2], sizes[0]) not in ((384, 32), (896, 96), (1152, 160))
        or any(size != sizes[0] for size in sizes)
        or not streamed_weight_device(left.device)
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.is_autocast_enabled("cuda") or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    rows = len(sizes) * sizes[0]
    inputs, outputs = right.shape[2], right.shape[1]
    if product == INPUT:
        inputs, outputs = outputs, inputs
    return (
        left.shape == (rows, inputs) and output.shape == (rows, outputs)
        and all(
            value.device == left.device and value.dtype == torch.float32
            and value.is_contiguous() and not value.is_neg()
            for value in (left, right, output)
        )
        and not torch._C._is_alias_of(output, left)
        and not torch._C._is_alias_of(output, right)
    )


@torch.no_grad()
def batched_segment_gemm(left, right, output, sizes, product, add):
    groups, rows = len(sizes), sizes[0]
    if product == WEIGHT:
        # Shorter reductions keep the four equal segments parallel on Hopper.
        chunk = 512 if rows <= 8192 else 2048
        splits = rows // chunk
        x = left.view(groups * splits, chunk, left.shape[1])
        dy = right.view(groups * splits, chunk, right.shape[1]).transpose(1, 2)
        if splits == 1:
            torch.bmm(dy, x, out=output)
        else:
            partials = torch.bmm(dy, x).view(groups, splits, *output.shape[1:])
            torch.sum(partials, dim=1, out=output)
        return

    lhs = left.view(groups, rows, left.shape[1])
    rhs = right.transpose(1, 2) if product == PROJECT else right
    target = output.view(groups, rows, output.shape[1])
    if add:
        torch.baddbmm(target, lhs, rhs, out=target)
    else:
        torch.bmm(lhs, rhs, out=target)


def grouped_gemm(left, right, output, sizes, product, add):
    if len(sizes) == 4 and (product in (PROJECT, INPUT) or (product == WEIGHT and not add)):
        mixed, row_matrix = (output, right) if product == WEIGHT else (right, output)
        if (
            use_equal_segment_bmm(left, mixed, sizes)
            and row_matrix.shape == left.shape and row_matrix.device == left.device
            and row_matrix.dtype == left.dtype and row_matrix.is_contiguous() and not output.is_neg()
            and (product != WEIGHT or not (left.is_neg() or right.is_neg()))
            and not torch._C._is_alias_of(output, left) and not torch._C._is_alias_of(output, right)
        ):
            return batched_segment_gemm(left, right, output, sizes, product, add)
    if product == WEIGHT:
        # Select outside the custom op so compiled calls retain WEIGHT.
        if not add and use_hopper_blas_weight(left, right, sizes):
            product = BLAS_WEIGHT
    elif use_uniform_node_gemm(left, right, output, sizes, product, add):
        return batched_segment_gemm(left, right, output, sizes, product, add)
    return grouped_gemm_cuda(left, right, output, sizes, product, add)


def row_sizes(x, sizes):
    factor = x.shape[1] if x.ndim == 3 else 1
    return [size * factor for size in sizes]
