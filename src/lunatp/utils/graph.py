"""CUDA construction of graph CSR, traversal orders and source segments."""

from functools import lru_cache

from ...utils.extension import BINDING_CFLAGS
from ...utils.jit import load_extension


TOPOLOGY_SOURCE = r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

template <typename Offset>
__global__ void node_counts(
    const Offset* src_rowptr, const Offset* dst_rowptr, const int32_t* node_order,
    int64_t* counts, int64_t* dst_degrees, int64_t num_nodes, int segment_size) {
    const int64_t node = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node == 0) counts[0] = 0;
    if (node >= num_nodes) return;
    const int64_t degree = int64_t(src_rowptr[node + 1]) - int64_t(src_rowptr[node]);
    const int64_t pieces = degree / segment_size + (degree % segment_size != 0);
    counts[node + 1] = pieces > 0 ? pieces : 1;
    const int32_t dst_node = node_order[node];
    dst_degrees[node] = int64_t(dst_rowptr[int64_t(dst_node) + 1]) - int64_t(dst_rowptr[dst_node]);
}

template <typename Offset>
__global__ void fill_segments(
    const Offset* src_rowptr, const int64_t* segment_rowptr,
    const int64_t* dst_cumulative, int32_t* tiers, int32_t* segment_nodes,
    Offset* begin, Offset* end, Offset* rowptr, int64_t* degrees,
    int32_t* lengths, int64_t num_nodes, int64_t num_edges, int segment_size, bool packed_order) {
    const int64_t node = blockIdx.x;
    if (threadIdx.x == 0 && node < 3) {
        int64_t cut = num_nodes;
        if (num_nodes && num_edges) {
            const int64_t part = node + 1;
            const int64_t threshold = (num_edges / 4) * part + (num_edges % 4) * part / 4;
            int64_t low = 0, high = num_nodes;
            while (low < high) {
                const int64_t middle = low + (high - low) / 2;
                if (dst_cumulative[middle] < threshold) low = middle + 1;
                else high = middle;
            }
            cut = low + 1;
        }
        tiers[node] = static_cast<int32_t>(cut);
    }
    if constexpr (sizeof(Offset) == 4) {
        if (node == 0 && threadIdx.x == 0) rowptr[0] = 0;
    }
    if (node >= num_nodes) return;
    const int64_t first = segment_rowptr[node];
    const int64_t last = segment_rowptr[node + 1];
    if constexpr (sizeof(Offset) == 4) {
        if (threadIdx.x == 0) rowptr[node + 1] = static_cast<Offset>(last);
    }
    const int64_t source_begin = src_rowptr[node];
    const int64_t source_end = src_rowptr[node + 1];
    for (int64_t segment = first + threadIdx.x; segment < last; segment += blockDim.x) {
        const int64_t start = source_begin + (segment - first) * segment_size;
        const int64_t remaining = source_end - start;
        const int64_t length = remaining < segment_size ? remaining : segment_size;
        segment_nodes[segment] = static_cast<int32_t>(node);
        begin[segment] = static_cast<Offset>(start);
        end[segment] = static_cast<Offset>(start + length);
        degrees[segment] = packed_order
            ? length * (num_edges + 1) + (source_end - source_begin)
            : source_end - source_begin;
        if (!packed_order) lengths[segment] = static_cast<int32_t>(length);
    }
}

__global__ void ordered_lengths(
    const int32_t* lengths, const int64_t* degree_order, int32_t* output, int64_t count) {
    const int64_t segment = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (segment < count) output[segment] = lengths[degree_order[segment]];
}

template <typename Offset>
__global__ void segment_order(
    const int64_t* degree_order, const int64_t* length_order, Offset* output, int64_t count) {
    const int64_t segment = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (segment < count) output[segment] = static_cast<Offset>(degree_order[length_order[segment]]);
}

template <typename Offset>
std::vector<at::Tensor> build_topology(
    const at::Tensor& src_rowptr, const at::Tensor& dst_rowptr,
    const at::Tensor& node_order, int64_t num_edges, int segment_size) {
    const int64_t num_nodes = src_rowptr.numel() - 1;
    const auto wide = src_rowptr.options().dtype(at::kLong);
    const auto integer = src_rowptr.options().dtype(at::kInt);
    const auto stream = at::cuda::getCurrentCUDAStream(src_rowptr.get_device());
    auto counts = at::empty({num_nodes + 1}, wide);
    auto dst_degrees = at::empty({num_nodes}, wide);
    node_counts<Offset><<<std::max<int64_t>(1, (num_nodes + 255) / 256), 256, 0, stream>>>(
        src_rowptr.data_ptr<Offset>(), dst_rowptr.data_ptr<Offset>(),
        node_order.data_ptr<int32_t>(), counts.data_ptr<int64_t>(),
        dst_degrees.data_ptr<int64_t>(), num_nodes, segment_size);
    auto offsets = at::cumsum(counts, 0, at::kLong);
    auto dst_cumulative = at::cumsum(dst_degrees, 0, at::kLong);
    // Exact-sized output needs the same single count transfer as the Torch construction.
    const int64_t num_segments = offsets.select(0, num_nodes).item<int64_t>();
    auto tiers = at::empty({3}, integer);
    auto nodes = at::empty({num_segments}, integer);
    auto begin = at::empty({num_segments}, src_rowptr.options());
    auto end = at::empty({num_segments}, src_rowptr.options());
    auto rowptr = sizeof(Offset) == 8 ? offsets : at::empty({num_nodes + 1}, src_rowptr.options());
    at::Tensor order;
    auto degrees = at::empty({num_segments}, wide);
    const bool packed_order = num_edges <=
        (std::numeric_limits<int64_t>::max() - segment_size) / (int64_t(segment_size) + 1);
    auto lengths = packed_order ? at::Tensor() : at::empty({num_segments}, integer);
    fill_segments<Offset><<<std::max<int64_t>(3, num_nodes), 128, 0, stream>>>(
        src_rowptr.data_ptr<Offset>(), offsets.data_ptr<int64_t>(),
        dst_cumulative.data_ptr<int64_t>(), tiers.data_ptr<int32_t>(), nodes.data_ptr<int32_t>(),
        begin.data_ptr<Offset>(), end.data_ptr<Offset>(), rowptr.data_ptr<Offset>(),
        degrees.data_ptr<int64_t>(), packed_order ? nullptr : lengths.data_ptr<int32_t>(), num_nodes, num_edges, segment_size, packed_order);
    if (num_segments && packed_order) {
        // Length is primary, degree secondary; stable sorting preserves segment-ID ties.
        order = at::argsort(degrees, true, 0, true).to(src_rowptr.scalar_type());
    } else {
        order = at::empty({num_segments}, src_rowptr.options());
        if (num_segments) {
            // Stable passes preserve the existing order: length, degree, then original segment ID.
            auto by_degree = at::argsort(degrees, true, 0, true);
            auto sorted_lengths = at::empty({num_segments}, integer);
            const int64_t blocks = (num_segments + 255) / 256;
            ordered_lengths<<<blocks, 256, 0, stream>>>(
                lengths.data_ptr<int32_t>(), by_degree.data_ptr<int64_t>(),
                sorted_lengths.data_ptr<int32_t>(), num_segments);
            auto by_length = at::argsort(sorted_lengths, true, 0, true);
            segment_order<Offset><<<blocks, 256, 0, stream>>>(
                by_degree.data_ptr<int64_t>(), by_length.data_ptr<int64_t>(),
                order.data_ptr<Offset>(), num_segments);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {tiers, nodes, begin, end, rowptr, order};
}

std::vector<at::Tensor> make_topology(
    const at::Tensor& src_rowptr, const at::Tensor& dst_rowptr,
    const at::Tensor& node_order, int64_t num_edges, int segment_size) {
    const c10::cuda::CUDAGuard guard(src_rowptr.device());
    if (src_rowptr.scalar_type() == at::kInt)
        return build_topology<int32_t>(src_rowptr, dst_rowptr, node_order, num_edges, segment_size);
    return build_topology<int64_t>(src_rowptr, dst_rowptr, node_order, num_edges, segment_size);
}

template <typename Offset>
__global__ void fill_csr(
    const Offset* src_cumulative, const Offset* dst_cumulative,
    Offset* src_rowptr, Offset* dst_rowptr, int64_t num_nodes) {
    const int64_t node = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node == 0) src_rowptr[0] = dst_rowptr[0] = 0;
    if (node < num_nodes) {
        src_rowptr[node + 1] = src_cumulative[node];
        dst_rowptr[node + 1] = dst_cumulative[node];
    }
}

template <typename SrcIndex, typename DstIndex>
__global__ void edge_counts(
    const SrcIndex* src, const DstIndex* dst, int64_t src_stride, int64_t dst_stride,
    int64_t* src_counts, int64_t* dst_counts, int64_t num_edges) {
    for (int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < num_edges; edge += int64_t(gridDim.x) * blockDim.x) {
        atomicAdd(reinterpret_cast<unsigned long long*>(src_counts + src[edge * src_stride]), 1ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(dst_counts + dst[edge * dst_stride]), 1ULL);
    }
}

template <typename Offset>
std::vector<at::Tensor> build_graph_tables(
    const at::Tensor& src, const at::Tensor& dst,
    int64_t num_nodes, int segment_size) {
    const auto offset_type = sizeof(Offset) == 4 ? at::kInt : at::kLong;
    const auto options = src.options().dtype(offset_type);
    const auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
    auto src_counts = at::zeros({num_nodes}, src.options().dtype(at::kLong));
    auto dst_counts = at::zeros({num_nodes}, src.options().dtype(at::kLong));
    const int64_t num_edges = src.numel();
    if (num_edges) {
        const int blocks = static_cast<int>(std::min<int64_t>((num_edges + 255) / 256, 4096));
        if (src.scalar_type() == at::kInt) {
            if (dst.scalar_type() == at::kInt)
                edge_counts<int32_t, int32_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), src.stride(0), dst.stride(0),
                    src_counts.data_ptr<int64_t>(), dst_counts.data_ptr<int64_t>(), num_edges);
            else
                edge_counts<int32_t, int64_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int32_t>(), dst.data_ptr<int64_t>(), src.stride(0), dst.stride(0),
                    src_counts.data_ptr<int64_t>(), dst_counts.data_ptr<int64_t>(), num_edges);
        } else {
            if (dst.scalar_type() == at::kInt)
                edge_counts<int64_t, int32_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int64_t>(), dst.data_ptr<int32_t>(), src.stride(0), dst.stride(0),
                    src_counts.data_ptr<int64_t>(), dst_counts.data_ptr<int64_t>(), num_edges);
            else
                edge_counts<int64_t, int64_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(), src.stride(0), dst.stride(0),
                    src_counts.data_ptr<int64_t>(), dst_counts.data_ptr<int64_t>(), num_edges);
        }
    }
    auto src_cumulative = at::cumsum(src_counts, 0, offset_type);
    auto dst_cumulative = at::cumsum(dst_counts, 0, offset_type);
    auto src_rowptr = at::empty({num_nodes + 1}, options);
    auto dst_rowptr = at::empty({num_nodes + 1}, options);
    fill_csr<Offset><<<std::max<int64_t>(1, (num_nodes + 255) / 256), 256, 0, stream>>>(
        src_cumulative.data_ptr<Offset>(), dst_cumulative.data_ptr<Offset>(),
        src_rowptr.data_ptr<Offset>(), dst_rowptr.data_ptr<Offset>(), num_nodes);
    auto source_edges = at::argsort(src).to(offset_type);
    auto node_order = at::argsort(dst_counts, 0, true).to(at::kInt);
    // The old source-degree sort uses the CSR dtype, including its tie ordering.
    auto source_node_order = at::argsort(src_counts.to(offset_type), 0, true).to(at::kInt);
    auto topology = build_topology<Offset>(
        src_rowptr, dst_rowptr, node_order, src.numel(), segment_size);
    std::vector<at::Tensor> result = {
        src_rowptr, dst_rowptr, source_edges, node_order, source_node_order};
    result.insert(result.end(), topology.begin(), topology.end());
    return result;
}

std::vector<at::Tensor> make_graph_tables(
    const at::Tensor& src, const at::Tensor& dst,
    int64_t num_nodes, int offset_bits, int segment_size) {
    const c10::cuda::CUDAGuard guard(src.device());
    if (offset_bits == 32)
        return build_graph_tables<int32_t>(src, dst, num_nodes, segment_size);
    return build_graph_tables<int64_t>(src, dst, num_nodes, segment_size);
}

__global__ void initialize_bounds(int64_t* bounds) {
    bounds[0] = INT64_MAX;
    bounds[1] = INT64_MIN;
    bounds[2] = 0;
}

template <typename SrcIndex, typename DstIndex>
__global__ void scan_bounds(
    const SrcIndex* src, const DstIndex* dst, int64_t src_stride, int64_t dst_stride,
    int64_t num_edges, int64_t* bounds) {
    long long minimum = INT64_MAX, maximum = INT64_MIN;
    int unsorted = 0;
    for (int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < num_edges; edge += int64_t(gridDim.x) * blockDim.x) {
        const long long source = src[edge * src_stride];
        const long long destination = dst[edge * dst_stride];
        minimum = min(minimum, min(source, destination));
        maximum = max(maximum, max(source, destination));
        unsorted |= edge > 0 && destination < dst[(edge - 1) * dst_stride];
    }
    for (int offset = 16; offset; offset /= 2) {
        minimum = min(minimum, __shfl_down_sync(0xffffffff, minimum, offset));
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        unsorted |= __shfl_down_sync(0xffffffff, unsorted, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMin(reinterpret_cast<long long*>(bounds), minimum);
        atomicMax(reinterpret_cast<long long*>(bounds + 1), maximum);
        if (unsorted) atomicOr(reinterpret_cast<unsigned long long*>(bounds + 2), 1ULL);
    }
}

std::vector<int64_t> graph_bounds(const at::Tensor& src, const at::Tensor& dst) {
    const c10::cuda::CUDAGuard guard(src.device());
    const auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
    auto bounds = at::empty({3}, src.options().dtype(at::kLong));
    initialize_bounds<<<1, 1, 0, stream>>>(bounds.data_ptr<int64_t>());
    const int64_t num_edges = src.numel();
    if (num_edges) {
        const int blocks = static_cast<int>(std::min<int64_t>((num_edges + 255) / 256, 4096));
        if (src.scalar_type() == at::kInt) {
            if (dst.scalar_type() == at::kInt)
                scan_bounds<int32_t, int32_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), src.stride(0), dst.stride(0),
                    num_edges, bounds.data_ptr<int64_t>());
            else
                scan_bounds<int32_t, int64_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int32_t>(), dst.data_ptr<int64_t>(), src.stride(0), dst.stride(0),
                    num_edges, bounds.data_ptr<int64_t>());
        } else {
            if (dst.scalar_type() == at::kInt)
                scan_bounds<int64_t, int32_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int64_t>(), dst.data_ptr<int32_t>(), src.stride(0), dst.stride(0),
                    num_edges, bounds.data_ptr<int64_t>());
            else
                scan_bounds<int64_t, int64_t><<<blocks, 256, 0, stream>>>(
                    src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(), src.stride(0), dst.stride(0),
                    num_edges, bounds.data_ptr<int64_t>());
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto cpu_bounds = bounds.to(at::kCPU);
    const int64_t* values = cpu_bounds.data_ptr<int64_t>();
    return {values[0], values[1], values[2]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("graph_bounds", &graph_bounds);
    module.def("make_topology", &make_topology);
    module.def("make_graph_tables", &make_graph_tables);
}
"""


@lru_cache(maxsize=None)
def get_topology_kernel(arch):
    return load_extension(
        "lunaeq_graph", cuda_sources=TOPOLOGY_SOURCE,
        extra_cflags=["-O3", *BINDING_CFLAGS],
        extra_cuda_cflags=["-O3", *BINDING_CFLAGS,
                          f"-gencode=arch=compute_{arch},code=sm_{arch}",
                          f"-gencode=arch=compute_{arch},code=compute_{arch}"],
        no_implicit_headers=True,
    )
