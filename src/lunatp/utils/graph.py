"""CUDA construction of graph CSR, traversal orders and source segments."""

from functools import lru_cache

from ...utils.extension import BINDING_CFLAGS
from ...utils.jit import load_extension


TOPOLOGY_SOURCE = r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <cooperative_groups.h>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_rank.cuh>

namespace {

namespace cg = cooperative_groups;

constexpr int graph_block_threads = 256;
constexpr int graph_items_per_thread = 8;
constexpr int max_radix_bits = 8;
constexpr int max_radix_counts = (32 / max_radix_bits) * (1 << max_radix_bits);

template <typename Offset>
struct GraphPrefix {
    Offset src_offset;
    Offset dst_offset;
    Offset segment_offset;
    Offset src_degree_max;
    Offset dst_degree_max;
};

struct AddGraphPrefix {
    template <typename Offset>
    __device__ GraphPrefix<Offset> operator()(GraphPrefix<Offset> a, GraphPrefix<Offset> b) const {
        return {a.src_offset + b.src_offset, max(a.dst_offset, b.dst_offset),
                a.segment_offset + b.segment_offset, max(a.src_degree_max, b.src_degree_max),
                max(a.dst_degree_max, b.dst_degree_max)};
    }
};

struct GraphStatus {
    int64_t min_index;
    int64_t max_index;
    int64_t unsorted;
    int64_t num_segments;
    int64_t max_src_degree;
    int64_t max_dst_degree;
    unsigned src_bits_or;
    unsigned src_bits_and;
};

enum GraphOrder { DestinationNodes, SourceNodes, SourceSegments };

template <typename Offset>
struct GraphBuffers {
    int32_t* src;
    int32_t* dst;
    int32_t* node_order;
    int32_t* source_node_order;
    int32_t* node_tiers;
    int32_t* segment_nodes;
    Offset* src_rowptr;
    Offset* dst_rowptr;
    Offset* edge_ids_by_src;
    Offset* edge_ids_by_dst;
    Offset* segment_begin;
    Offset* segment_end;
    Offset* segment_rowptr;
    Offset* segment_order;
    int32_t* node_order_scratch;
    int32_t* source_order_scratch;
    Offset* segment_order_scratch;
    Offset* order_histogram;
    Offset* order_counts;
    Offset* edge_counts;
    cub::ScanTileState<Offset> radix_scan_state;
    cub::ScanTileState<GraphPrefix<Offset>> csr_scan_state;
    cub::ScanTileState<Offset, false> degree_scan_state;
    GraphStatus* status;
};

template <typename Offset>
__device__ uint64_t order_key(
    const GraphBuffers<Offset>& graph, GraphOrder order, Offset value, bool sort_lengths) {
    if (order == SourceSegments) {
        if (sort_lengths) return uint64_t(graph.segment_end[value]) - graph.segment_begin[value];
        value = graph.segment_nodes[value];
    }
    const Offset* csr = order == DestinationNodes ? graph.dst_rowptr : graph.src_rowptr;
    return uint64_t(csr[int64_t(value) + 1]) - csr[value];
}

__device__ int64_t graph_index(const void* values, bool wide, int64_t offset) {
    return wide ? static_cast<const int64_t*>(values)[offset]
                : static_cast<const int32_t*>(values)[offset];
}

struct GraphDigit {
    int shift, mask;
    __device__ __forceinline__ unsigned Digit(int32_t key) const {
        return (static_cast<unsigned>(key) >> shift) & mask;
    }
};

template <typename Offset>
struct GraphRadixCounts {
    using State = cub::ScanTileState<Offset>;
    State state;
    Offset* counts;
    int64_t tile;
    int radix_bins, digit_bins, valid_items;
    typename State::StatusWord partial_tag, inclusive_tag;

    __device__ __forceinline__ void operator()(int (&bins)[1]) const {
        const int bin = threadIdx.x;
        if (bin >= digit_bins) return;
        const Offset total = bins[0] - (bin + 1 == digit_bins
            ? graph_block_threads * graph_items_per_thread - valid_items : 0);
        counts[bin] = total;
        typename State::TileDescriptor descriptor{tile == 0 ? inclusive_tag : partial_tag, total};
        typename State::TxnWord word;
        *reinterpret_cast<typename State::TileDescriptor*>(&word) = descriptor;
        cub::detail::store_relaxed(state.d_tile_descriptors + State::TILE_STATUS_PADDING
                                  + tile * radix_bins + bin, word);
    }
};

template <typename Offset>
__global__ void __launch_bounds__(256, 2) make_graph_kernel(
    const void* input_src, const void* input_dst, bool src_wide, bool dst_wide,
    int64_t src_stride, int64_t dst_stride, int64_t num_nodes, int64_t num_edges,
    int segment_size, int radix_bits, GraphBuffers<Offset> graph) {
    constexpr int threads = graph_block_threads;
    constexpr int max_edge_items = graph_items_per_thread;
    const int radix_bins = 1 << radix_bits;
    const int radix_counts = ((32 + radix_bits - 1) / radix_bits) * radix_bins;
    using Prefix = GraphPrefix<Offset>;
    using Scan = cub::BlockScan<Prefix, threads>;
    using CountScan = cub::BlockScan<Offset, threads>;
    using PrefixCallback = cub::TilePrefixCallbackOp<Prefix, AddGraphPrefix, cub::ScanTileState<Prefix>>;
    using DegreeCallback = cub::TilePrefixCallbackOp<Offset, cub::Sum, cub::ScanTileState<Offset, false>>;
    using EdgeState = cub::ScanTileState<Offset>;
    using RadixRank = cub::BlockRadixRankMatch<threads, max_radix_bits, false>;
    __shared__ union {
        typename RadixRank::TempStorage rank;
        typename Scan::TempStorage scan;
        int warp_histogram[8 * (threads + 1)];
        struct {
            int32_t keys[threads * max_edge_items];
            Offset values[threads * max_edge_items];
        } exchange;
        typename CountScan::TempStorage counts;
        typename PrefixCallback::TempStorage prefix_callback;
        typename DegreeCallback::TempStorage degree_callback;
        struct {
            int64_t minimum[8], maximum[8];
            unsigned src_bits_or[8], src_bits_and[8];
            int unsorted[8];
        } bounds;
    } storage;
    __shared__ Offset bucket_base[max_radix_counts];
    const auto grid = cg::this_grid();
    const int64_t thread = int64_t(blockIdx.x) * threads + threadIdx.x;
    const int64_t stride = int64_t(gridDim.x) * threads;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x / 32;
    const int edge_items = static_cast<int>(max(int64_t(1), min(int64_t(max_edge_items),
        num_edges / stride + (num_edges % stride != 0))));
    const int edge_tile_size = threads * edge_items;
    const int node_tiles = static_cast<int>((num_nodes + threads - 1) / threads);
    constexpr int tile_padding = cub::ScanTileState<Prefix>::TILE_STATUS_PADDING;
    for (int64_t tile = thread; tile < node_tiles; tile += stride)
        graph.csr_scan_state.d_tile_status[tile_padding + tile] = cub::SCAN_TILE_INVALID;
    if (thread < tile_padding) {
        graph.csr_scan_state.d_tile_status[thread] = cub::SCAN_TILE_OOB;
        graph.csr_scan_state.d_tile_inclusive[thread] = Prefix{};
    }
    for (int64_t node = thread; node <= num_nodes; node += stride) {
        graph.src_rowptr[node] = graph.dst_rowptr[node] = 0;
    }
    for (int64_t index = thread; index < radix_counts; index += stride) graph.edge_counts[index] = 0;
    if (thread == 0) {
        graph.status->min_index = INT64_MAX;
        graph.status->max_index = INT64_MIN;
        graph.status->unsorted = graph.status->num_segments =
            graph.status->max_src_degree = graph.status->max_dst_degree = 0;
        graph.status->src_bits_or = 0;
        graph.status->src_bits_and = ~0u;
        graph.segment_rowptr[0] = 0;
        graph.node_tiers[0] = graph.node_tiers[1] = graph.node_tiers[2] = static_cast<int32_t>(num_nodes);
    }
    grid.sync();

    // Count source edges and locate destination runs.
    int64_t minimum = INT64_MAX, maximum = INT64_MIN;
    int unsorted = 0;
    unsigned src_bits_or = 0, src_bits_and = ~0u;
    using AtomicCount = std::conditional_t<sizeof(Offset) == 4, int, unsigned long long>;
    auto* source_counts = reinterpret_cast<AtomicCount*>(graph.src_rowptr);
    int32_t pending_source = -1;
    Offset pending_count = 0;
    // Destination starts reuse node-order workspace.
    auto* destination_begin = reinterpret_cast<Offset*>(graph.node_order_scratch);
    for (int64_t edge = thread; edge < num_edges; edge += stride) {
        const int64_t source = graph_index(input_src, src_wide, edge * src_stride);
        const int64_t destination = graph_index(input_dst, dst_wide, edge * dst_stride);
        minimum = min(minimum, min(source, destination));
        maximum = max(maximum, max(source, destination));
        const int64_t previous_destination = edge > 0 ? graph_index(input_dst, dst_wide, (edge - 1) * dst_stride) : -1;
        unsorted |= edge > 0 && destination < previous_destination;
        graph.src[edge] = static_cast<int32_t>(source);
        src_bits_or |= static_cast<unsigned>(source);
        src_bits_and &= static_cast<unsigned>(source);
        if (source >= 0 && source < num_nodes && destination >= 0 && destination < num_nodes) {
            const unsigned active = __activemask();
            const unsigned source_peers = __match_any_sync(active, static_cast<int32_t>(source));
            if (lane == __ffs(source_peers) - 1) {
                if (pending_count && pending_source != source) {
                    atomicAdd(source_counts + int64_t(pending_source) + 1, static_cast<AtomicCount>(pending_count));
                    pending_count = 0;
                }
                pending_source = static_cast<int32_t>(source);
                pending_count += __popc(source_peers);
            }
            if (edge == 0 || destination != previous_destination)
                destination_begin[destination] = static_cast<Offset>(edge);
            if (edge + 1 == num_edges || destination != graph_index(input_dst, dst_wide, (edge + 1) * dst_stride))
                graph.dst_rowptr[destination + 1] = static_cast<Offset>(edge + 1);
        }
    }
    if (pending_count)
        atomicAdd(source_counts + int64_t(pending_source) + 1, static_cast<AtomicCount>(pending_count));
    for (int shift = 16; shift; shift /= 2) {
        minimum = min(minimum, __shfl_down_sync(0xffffffff, minimum, shift));
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, shift));
        unsorted |= __shfl_down_sync(0xffffffff, unsorted, shift);
        src_bits_or |= __shfl_down_sync(0xffffffff, src_bits_or, shift);
        src_bits_and &= __shfl_down_sync(0xffffffff, src_bits_and, shift);
    }
    if (!lane) {
        storage.bounds.minimum[warp_in_block] = minimum;
        storage.bounds.maximum[warp_in_block] = maximum;
        storage.bounds.src_bits_or[warp_in_block] = src_bits_or;
        storage.bounds.src_bits_and[warp_in_block] = src_bits_and;
        storage.bounds.unsorted[warp_in_block] = unsorted;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        minimum = lane < 8 ? storage.bounds.minimum[lane] : INT64_MAX;
        maximum = lane < 8 ? storage.bounds.maximum[lane] : INT64_MIN;
        src_bits_or = lane < 8 ? storage.bounds.src_bits_or[lane] : 0;
        src_bits_and = lane < 8 ? storage.bounds.src_bits_and[lane] : ~0u;
        unsorted = lane < 8 ? storage.bounds.unsorted[lane] : 0;
        for (int distance = 16; distance; distance /= 2) {
            minimum = min(minimum, __shfl_down_sync(0xffffffff, minimum, distance));
            maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, distance));
            src_bits_or |= __shfl_down_sync(0xffffffff, src_bits_or, distance);
            src_bits_and &= __shfl_down_sync(0xffffffff, src_bits_and, distance);
            unsorted |= __shfl_down_sync(0xffffffff, unsorted, distance);
        }
        if (!lane) {
            atomicMin(reinterpret_cast<long long*>(&graph.status->min_index), static_cast<long long>(minimum));
            atomicMax(reinterpret_cast<long long*>(&graph.status->max_index), static_cast<long long>(maximum));
            if (unsorted) atomicOr(reinterpret_cast<unsigned long long*>(&graph.status->unsorted), 1ULL);
            atomicOr(&graph.status->src_bits_or, src_bits_or);
            atomicAnd(&graph.status->src_bits_and, src_bits_and);
        }
    }
    grid.sync();
    if (graph.status->min_index < 0 || graph.status->max_index >= num_nodes || graph.status->unsorted) return;

    // Scan CSR and source-segment offsets.
    for (int tile = blockIdx.x; tile < node_tiles; tile += gridDim.x) {
        const int64_t node = int64_t(tile) * threads + threadIdx.x;
        const int64_t source = node < num_nodes ? graph.src_rowptr[node + 1] : 0;
        const int64_t destination = node < num_nodes ? graph.dst_rowptr[node + 1] : 0;
        const int64_t destination_degree = destination ? destination - destination_begin[node] : 0;
        const int64_t pieces = node < num_nodes ? max(int64_t(1), (source + segment_size - 1) / segment_size) : 0;
        Prefix prefix, total;
        Scan(storage.scan).InclusiveScan(
            Prefix{Offset(source), Offset(destination), Offset(pieces),
                   Offset(source), Offset(destination_degree)},
            prefix, AddGraphPrefix{}, total);
        __syncthreads();
        Prefix base{0, 0, 0, 0, 0}, inclusive = total;
        if (tile == 0) {
            if (threadIdx.x == 0) graph.csr_scan_state.SetInclusive(0, total);
        } else {
            PrefixCallback callback(graph.csr_scan_state, storage.prefix_callback, AddGraphPrefix{}, tile);
            if (threadIdx.x < 32) callback(total);
            __syncthreads();
            base = callback.GetExclusivePrefix();
            inclusive = callback.GetInclusivePrefix();
        }
        prefix = AddGraphPrefix{}(base, prefix);
        if (node < num_nodes) {
            graph.src_rowptr[node + 1] = prefix.src_offset;
            graph.dst_rowptr[node + 1] = prefix.dst_offset;
            graph.segment_rowptr[node + 1] = prefix.segment_offset;
        }
        if (threadIdx.x == 0 && tile + 1 == node_tiles) {
            graph.status->num_segments = inclusive.segment_offset;
            graph.status->max_src_degree = inclusive.src_degree_max;
            graph.status->max_dst_degree = inclusive.dst_degree_max;
        }
        __syncthreads();
    }
    grid.sync();
    // Stable radix sort of source edge IDs.
    const int64_t num_segments = graph.status->num_segments;
    const int64_t warp = thread / 32;
    const int64_t edge_tiles = (num_edges + edge_tile_size - 1) / edge_tile_size;
    const unsigned changing = num_edges ? graph.status->src_bits_or ^ graph.status->src_bits_and : 0;
    const int first_bit = changing ? __ffs(changing) - 1 : 0;
    const int last_bit = 32 - __clz(changing);
    int32_t* key_input = graph.src;
    int32_t* key_output = graph.dst;
    Offset* edge_input = graph.edge_ids_by_dst;
    Offset* edge_output = graph.edge_ids_by_src;
    typename EdgeState::TxnWord invalid{}, padding{};
    reinterpret_cast<typename EdgeState::TileDescriptor*>(&invalid)->status = cub::SCAN_TILE_INVALID;
    reinterpret_cast<typename EdgeState::TileDescriptor*>(&padding)->status = cub::SCAN_TILE_OOB;
    if (changing) {
        for (int index = threadIdx.x; index < radix_counts; index += threads) bucket_base[index] = 0;
        __syncthreads();
        for (int64_t node = thread; node < num_nodes; node += stride) {
            const Offset degree = graph.src_rowptr[node + 1] - graph.src_rowptr[node];
            if (degree) {
                for (int pass = 0, shift = first_bit; shift < last_bit; ++pass, shift += radix_bits) {
                    const unsigned mask = (1u << min(radix_bits, last_bit - shift)) - 1;
                    const int index = pass * radix_bins + ((node >> shift) & mask);
                    if constexpr (sizeof(Offset) == 4) atomicAdd(bucket_base + index, degree);
                    else atomicAdd(reinterpret_cast<unsigned long long*>(bucket_base + index),
                                   static_cast<unsigned long long>(degree));
                }
            }
        }
        __syncthreads();
        for (int index = threadIdx.x; index < radix_counts; index += threads) {
            if (bucket_base[index]) {
                if constexpr (sizeof(Offset) == 4) atomicAdd(graph.edge_counts + index, bucket_base[index]);
                else atomicAdd(reinterpret_cast<unsigned long long*>(graph.edge_counts + index),
                               static_cast<unsigned long long>(bucket_base[index]));
            }
        }
        for (int64_t descriptor = thread; descriptor < radix_bins * edge_tiles; descriptor += stride)
            graph.radix_scan_state.d_tile_descriptors[tile_padding + descriptor] = invalid;
        if (thread < tile_padding) graph.radix_scan_state.d_tile_descriptors[thread] = padding;
        grid.sync();
    }
    // Pass tags reuse one packed prefix buffer across radix rounds.
    for (int pass = 0, shift = first_bit; shift < last_bit; ++pass, shift += radix_bits) {
        const int digit_bins = 1 << min(radix_bits, last_bit - shift);
        const typename EdgeState::StatusWord partial_tag = 2 * (pass + 1);
        const typename EdgeState::StatusWord inclusive_tag = partial_tag + 1;
        Offset* digit_counts = graph.edge_counts + pass * radix_bins;
        const Offset digit_count = threadIdx.x < digit_bins ? digit_counts[threadIdx.x] : 0;
        Offset bucket_start;
        CountScan(storage.counts).ExclusiveSum(digit_count, bucket_start);
        __syncthreads();

        for (int64_t tile = blockIdx.x; tile < edge_tiles; tile += gridDim.x) {
            Offset values[max_edge_items];
            int32_t keys[max_edge_items];
            int ranks[max_edge_items];
            const int valid_items = static_cast<int>(min(int64_t(edge_tile_size), num_edges - tile * edge_tile_size));
            #pragma unroll
            for (int item = 0; item < max_edge_items; ++item) {
                const int64_t position = tile * edge_tile_size + warp_in_block * edge_items * 32 + item * 32 + lane;
                keys[item] = -1;
                if (item < edge_items && position < num_edges) {
                    values[item] = shift == first_bit ? static_cast<Offset>(position) : edge_input[position];
                    keys[item] = key_input[position];
                }
            }
            int digit_prefix[1];
            RadixRank(storage.rank).RankKeys(keys, ranks, GraphDigit{shift, digit_bins - 1}, digit_prefix,
                GraphRadixCounts<Offset>{graph.radix_scan_state, bucket_base, tile, radix_bins,
                                         digit_bins, valid_items, partial_tag, inclusive_tag});
            if (threadIdx.x < digit_bins)
                bucket_base[radix_bins + threadIdx.x] = static_cast<Offset>(digit_prefix[0]);
            if (threadIdx.x < digit_bins) {
                const int state_tile = static_cast<int>(tile * radix_bins + threadIdx.x);
                const Offset total = bucket_base[threadIdx.x];
                Offset before = 0;
                typename EdgeState::TileDescriptor descriptor;
                typename EdgeState::TxnWord word;
                auto* states = graph.radix_scan_state.d_tile_descriptors + tile_padding;
                if (tile != 0) {
                    int predecessor = static_cast<int>((tile - 1) * radix_bins + threadIdx.x);
                    bool done = false;
                    const unsigned active = __activemask();
                    while (__any_sync(active, !done)) {
                        if (!done) {
                            auto previous = cub::detail::load_relaxed(states + predecessor);
                            const auto value = *reinterpret_cast<typename EdgeState::TileDescriptor*>(&previous);
                            if (value.status == partial_tag || value.status == inclusive_tag) {
                                before += value.value;
                                done = value.status == inclusive_tag;
                                if (!done) predecessor -= radix_bins;
                            }
                        }
                    }
                    descriptor = {inclusive_tag, before + total};
                    *reinterpret_cast<typename EdgeState::TileDescriptor*>(&word) = descriptor;
                    cub::detail::store_relaxed(states + state_tile, word);
                }
                bucket_base[threadIdx.x] = bucket_start + before;
            }
            __syncthreads();
            #pragma unroll
            for (int item = 0; item < max_edge_items; ++item) {
                const int64_t position = tile * edge_tile_size + warp_in_block * edge_items * 32 + item * 32 + lane;
                if (item < edge_items && position < num_edges) {
                    const int digit = (keys[item] >> shift) & (digit_bins - 1);
                    const int padding = digit + 1 == digit_bins
                        ? warp_in_block * 32 * (max_edge_items - edge_items) : 0;
                    const int local = ranks[item] - padding;
                    storage.exchange.keys[local] = keys[item];
                    storage.exchange.values[local] = values[item];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int item = 0; item < max_edge_items; ++item) {
                const int local = item * threads + threadIdx.x;
                if (item < edge_items && tile * edge_tile_size + local < num_edges) {
                    const int32_t key = storage.exchange.keys[local];
                    const int digit = (key >> shift) & (digit_bins - 1);
                    const int64_t target = int64_t(bucket_base[digit]) + local - bucket_base[radix_bins + digit];
                    key_output[target] = key;
                    edge_output[target] = storage.exchange.values[local];
                }
            }
            __syncthreads();
        }
        grid.sync();
        int32_t* key_swap = key_input;
        key_input = key_output;
        key_output = key_swap;
        Offset* edge_swap = edge_input;
        edge_input = edge_output;
        edge_output = edge_swap;
    }
    for (int64_t node = thread; node < num_nodes; node += stride) {
        const Offset begin = graph.src_rowptr[node];
        if (begin == graph.src_rowptr[node + 1]) {
            const Offset segment = graph.segment_rowptr[node];
            graph.segment_nodes[segment] = static_cast<int32_t>(node);
            graph.segment_begin[segment] = graph.segment_end[segment] = begin;
        }
    }
    for (int64_t position = thread; position < num_edges; position += stride) {
        const int32_t source = key_input[position];
        const Offset value = changing ? edge_input[position] : static_cast<Offset>(position);
        graph.edge_ids_by_src[position] = value;
        graph.src[position] = static_cast<int32_t>(graph_index(input_src, src_wide, position * src_stride));
        graph.dst[position] = static_cast<int32_t>(graph_index(input_dst, dst_wide, position * dst_stride));
        const int64_t begin = graph.src_rowptr[source];
        if ((position - begin) % segment_size == 0) {
            const int64_t segment = int64_t(graph.segment_rowptr[source]) + (position - begin) / segment_size;
            graph.segment_nodes[segment] = source;
            graph.segment_begin[segment] = static_cast<Offset>(position);
            graph.segment_end[segment] = static_cast<Offset>(min(position + segment_size,
                                                               int64_t(graph.src_rowptr[int64_t(source) + 1])));
        }
    }
    grid.sync();

    // Degree order for nodes; length/degree order for source segments.
    const int degree_bits = max(1, 64 - __clzll(static_cast<unsigned long long>(
        max(graph.status->max_src_degree, graph.status->max_dst_degree))));
    const int degree_passes = (degree_bits + 7) / 8;
    const int64_t segment_tiles = (num_segments + threads - 1) / threads;
    const int64_t order_tiles = 2 * node_tiles + segment_tiles;
    for (int pass = 0; pass <= degree_passes; ++pass) {
        const bool sort_lengths = pass == degree_passes;
        const int shift = sort_lengths ? 0 : pass * 8;
        const int bits = sort_lengths ? 6 : min(8, degree_bits - shift);
        const unsigned mask = (1u << bits) - 1;
        const int64_t first_tile = sort_lengths ? 2 * node_tiles : 0;
        const int buckets = 1 << bits;
        for (int64_t job = first_tile + blockIdx.x; job < order_tiles; job += gridDim.x) {
            const GraphOrder order = job >= 2 * node_tiles ? SourceSegments
                : (job >= node_tiles ? SourceNodes : DestinationNodes);
            const int64_t tile = job - (order == SourceSegments ? 2 * node_tiles : order * node_tiles);
            const int64_t position = tile * threads + threadIdx.x;
            const int64_t count = order == SourceSegments ? num_segments : num_nodes;
            for (int index = threadIdx.x; index < 8 * buckets; index += threads)
                storage.warp_histogram[index] = 0;
            __syncthreads();
            const unsigned active = __ballot_sync(0xffffffff, position < count);
            if (position < count) {
                const Offset value = pass == 0 ? static_cast<Offset>(position)
                    : order == SourceSegments ? graph.segment_order[position]
                    : order == SourceNodes ? graph.source_node_order[position] : graph.node_order[position];
                const int digit = int((order_key(graph, order, value, sort_lengths) >> shift) & mask);
                const unsigned peers = __match_any_sync(active, digit);
                if (lane == __ffs(peers) - 1)
                    storage.warp_histogram[(threadIdx.x / 32) * buckets + digit] = __popc(peers);
                if (order == SourceSegments) graph.segment_order_scratch[position] = value;
                else if (order == SourceNodes) graph.source_order_scratch[position] = static_cast<int32_t>(value);
                else graph.node_order_scratch[position] = static_cast<int32_t>(value);
            }
            __syncthreads();
            if (threadIdx.x < buckets) {
                int total = 0;
                #pragma unroll
                for (int warp = 0; warp < 8; ++warp)
                    total += storage.warp_histogram[warp * buckets + threadIdx.x];
                graph.order_histogram[job * threads + threadIdx.x] = static_cast<Offset>(total);
            }
            __syncthreads();
        }
        grid.sync();

        // Prefix each bucket across node_tiles.
        for (int task = (sort_lengths ? 2 * buckets : 0) + int(warp); task < 3 * buckets; task += int(stride / 32)) {
            const auto order = static_cast<GraphOrder>(task >> bits);
            const int digit = task & mask;
            const int64_t first = order == SourceSegments ? 2 * node_tiles : order * node_tiles;
            const int64_t count = order == SourceSegments ? segment_tiles : node_tiles;
            Offset carry = 0;
            for (int64_t first_tile = 0; first_tile < count; first_tile += 32) {
                const int64_t tile = first_tile + lane;
                const Offset own = tile < count ? graph.order_histogram[(first + tile) * threads + digit] : 0;
                Offset prefix = own;
                for (int offset = 1; offset < 32; offset *= 2) {
                    const Offset value = __shfl_up_sync(0xffffffff, prefix, offset);
                    if (lane >= offset) prefix += value;
                }
                if (tile < count)
                    graph.order_histogram[(first + tile) * threads + digit] = carry + prefix - own;
                carry += __shfl_sync(0xffffffff, prefix, 31);
            }
            if (!lane) graph.order_counts[order * threads + digit] = carry;
        }
        grid.sync();

        for (int64_t job = first_tile + blockIdx.x; job < order_tiles; job += gridDim.x) {
            const GraphOrder order = job >= 2 * node_tiles ? SourceSegments
                : (job >= node_tiles ? SourceNodes : DestinationNodes);
            const int64_t tile = job - (order == SourceSegments ? 2 * node_tiles : order * node_tiles);
            const int64_t position = tile * threads + threadIdx.x;
            const int64_t count = order == SourceSegments ? num_segments : num_nodes;
            const int bucket = buckets - 1 - threadIdx.x;
            Offset base;
            const Offset total = threadIdx.x < buckets ? graph.order_counts[order * threads + bucket] : 0;
            CountScan(storage.counts).ExclusiveSum(total, base);
            if (threadIdx.x < buckets) bucket_base[bucket] = base;
            __syncthreads();
            for (int index = threadIdx.x; index < 8 * buckets; index += threads)
                storage.warp_histogram[index] = 0;
            Offset value = 0;
            int digit = 0;
            unsigned peers = 0;
            __syncthreads();
            const unsigned active = __ballot_sync(0xffffffff, position < count);
            if (position < count) {
                value = order == SourceSegments ? graph.segment_order_scratch[position]
                    : order == SourceNodes ? graph.source_order_scratch[position] : graph.node_order_scratch[position];
                digit = int((order_key(graph, order, value, sort_lengths) >> shift) & mask);
                peers = __match_any_sync(active, digit);
                if (lane == __ffs(peers) - 1)
                    storage.warp_histogram[(threadIdx.x / 32) * buckets + digit] = __popc(peers);
            }
            __syncthreads();
            if (position < count) {
                int rank = __popc(peers & ((1u << lane) - 1));
                for (int warp = 0; warp < threadIdx.x / 32; ++warp)
                    rank += storage.warp_histogram[warp * buckets + digit];
                const int64_t target = int64_t(bucket_base[digit])
                    + graph.order_histogram[job * threads + digit] + rank;
                if (order == SourceSegments) graph.segment_order[target] = value;
                else if (order == SourceNodes) graph.source_node_order[target] = static_cast<int32_t>(value);
                else graph.node_order[target] = static_cast<int32_t>(value);
            }
            __syncthreads();
        }
        if (sort_lengths) {
            for (int64_t tile = thread; tile < node_tiles; tile += stride)
                graph.degree_scan_state.d_tile_status[tile_padding + tile] = cub::SCAN_TILE_INVALID;
            if (thread < tile_padding) {
                graph.degree_scan_state.d_tile_status[thread] = cub::SCAN_TILE_OOB;
                graph.degree_scan_state.d_tile_inclusive[thread] = 0;
            }
        }
        grid.sync();
    }

    if (num_nodes && num_edges) {
        for (int tile = blockIdx.x; tile < node_tiles; tile += gridDim.x) {
            const int64_t rank = int64_t(tile) * threads + threadIdx.x;
            const int32_t node = rank < num_nodes ? graph.node_order[rank] : 0;
            const Offset degree = rank < num_nodes ? graph.dst_rowptr[int64_t(node) + 1] - graph.dst_rowptr[node] : 0;
            Offset prefix, total;
            CountScan(storage.counts).InclusiveSum(degree, prefix, total);
            __syncthreads();
            Offset base = 0;
            if (tile == 0) {
                if (threadIdx.x == 0) graph.degree_scan_state.SetInclusive(0, total);
            } else {
                DegreeCallback callback(graph.degree_scan_state, storage.degree_callback, cub::Sum{}, tile);
                if (threadIdx.x < 32) callback(total);
                __syncthreads();
                base = callback.GetExclusivePrefix();
            }
            if (rank < num_nodes) {
                const int64_t end = int64_t(base) + prefix;
                const int64_t begin = end - degree;
                for (int part = 1; part <= 3; ++part) {
                    const int64_t threshold = (num_edges / 4) * part + (num_edges % 4) * part / 4;
                    if ((begin < threshold && end >= threshold) || (!rank && !threshold))
                        graph.node_tiers[part - 1] = static_cast<int32_t>(rank + 1);
                }
            }
            __syncthreads();
        }
    }
    for (int64_t edge = thread; edge < num_edges; edge += stride)
        graph.edge_ids_by_dst[edge] = static_cast<Offset>(edge);
}

template <typename Offset>
std::vector<at::Tensor> fused_graph_tables(
    const at::Tensor& src, const at::Tensor& dst, int64_t num_nodes, int segment_size) {
    int64_t num_edges = src.numel();
    const int node_bits = num_nodes > 1 ? 64 - __builtin_clzll(num_nodes - 1) : 1;
    const int radix_passes = (node_bits + max_radix_bits - 1) / max_radix_bits;
    int radix_bits = (node_bits + radix_passes - 1) / radix_passes;
    const int radix_bins = 1 << radix_bits;
    const int radix_counts = ((32 + radix_bits - 1) / radix_bits) * radix_bins;
    const auto* properties = at::cuda::getDeviceProperties(src.get_device());
    static const int workers_per_sm = [] {
        int active = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, make_graph_kernel<Offset>, 256, 0));
        return std::max(1, std::min(2, active));
    }();
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(
        int64_t(properties->multiProcessorCount) * workers_per_sm,
        (std::max(num_nodes, num_edges) + 255) / 256)));
    const int64_t segment_capacity = num_nodes + num_edges / segment_size;
    const int64_t node_tiles = (num_nodes + 255) / 256;
    const auto integer = src.options().dtype(at::kInt);
    const auto wide = src.options().dtype(at::kLong);
    const auto offsets = src.options().dtype(sizeof(Offset) == 4 ? at::kInt : at::kLong);
    auto src_out = at::empty({num_edges}, integer);
    auto dst_out = at::empty({num_edges}, integer);
    auto src_rowptr = at::empty({num_nodes + 1}, offsets);
    auto dst_rowptr = at::empty({num_nodes + 1}, offsets);
    auto edge_ids_by_src = at::empty({num_edges}, offsets);
    auto edge_ids_by_dst = at::empty({num_edges}, offsets);
    auto node_order = at::empty({num_nodes}, integer);
    auto source_node_order = at::empty({num_nodes}, integer);
    auto node_tiers = at::empty({3}, integer);
    auto segment_nodes = at::empty({segment_capacity}, integer);
    auto segment_begin = at::empty({segment_capacity}, offsets);
    auto segment_end = at::empty({segment_capacity}, offsets);
    auto segment_rowptr = at::empty({num_nodes + 1}, offsets);
    auto segment_order = at::empty({segment_capacity}, offsets);
    const int64_t histogram_slots = (2 * node_tiles + (segment_capacity + 255) / 256) * 256;
    const int64_t order_scratch_words = ((segment_capacity + histogram_slots + 768) * sizeof(Offset) + 7) / 8;
    cub::ScanTileState<GraphPrefix<Offset>> csr_scan_state;
    cub::ScanTileState<Offset, false> degree_scan_state;
    size_t prefix_bytes = 0, degree_bytes = 0;
    C10_CUDA_CHECK(decltype(csr_scan_state)::AllocationSize(node_tiles, prefix_bytes));
    C10_CUDA_CHECK(decltype(degree_scan_state)::AllocationSize(node_tiles, degree_bytes));
    const size_t state_bytes = std::max(prefix_bytes, degree_bytes);
    const int64_t scan_words = (state_bytes + 7) / 8;
    const int64_t edge_stride = int64_t(blocks) * 256;
    const int64_t edge_items = std::max<int64_t>(1, std::min<int64_t>(8,
        num_edges / edge_stride + (num_edges % edge_stride != 0)));
    const int64_t edge_tile_size = 256 * edge_items;
    const int64_t edge_tiles = (num_edges + edge_tile_size - 1) / edge_tile_size;
    cub::ScanTileState<Offset> radix_scan_state;
    size_t edge_bytes = 0;
    C10_CUDA_CHECK(decltype(radix_scan_state)::AllocationSize(static_cast<int>(radix_bins * edge_tiles), edge_bytes));
    const int64_t radix_scratch_words = (edge_bytes + 15 + 7) / 8;
    const bool reuse_edges = num_edges >= segment_capacity + histogram_slots + 768;
    const bool reuse_nodes = reuse_edges && num_edges * sizeof(Offset) >=
        num_nodes * sizeof(int64_t) + (segment_capacity + histogram_slots + 768) * sizeof(Offset);
    const int64_t node_scratch_words = reuse_nodes ? 0 : num_nodes;
    const int64_t scratch_words = std::max(reuse_edges ? int64_t(0) : order_scratch_words, radix_scratch_words);
    auto workspace = at::empty({node_scratch_words + scan_words + scratch_words
                               + radix_counts * int64_t(sizeof(Offset)) / 8}, wide);
    const int64_t status_words = (sizeof(GraphStatus) + 7) / 8;
    auto status = at::empty({status_words}, wide);
    auto* work = workspace.data_ptr<int64_t>();
    C10_CUDA_CHECK(csr_scan_state.Init(node_tiles, work + node_scratch_words, state_bytes));
    C10_CUDA_CHECK(degree_scan_state.Init(node_tiles, work + node_scratch_words, state_bytes));
    auto* node_order_scratch = reuse_nodes ? reinterpret_cast<int32_t*>(edge_ids_by_dst.data_ptr<Offset>())
        : reinterpret_cast<int32_t*>(work);
    auto* scratch = work + node_scratch_words + scan_words;
    auto* segment_order_scratch = reuse_edges ? edge_ids_by_dst.data_ptr<Offset>()
        + (reuse_nodes ? 2 * num_nodes * sizeof(int32_t) / sizeof(Offset) : 0)
        : reinterpret_cast<Offset*>(scratch);
    void* edge_storage = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(scratch) + 15) & ~uintptr_t(15));
    C10_CUDA_CHECK(radix_scan_state.Init(static_cast<int>(radix_bins * edge_tiles), edge_storage, edge_bytes));
    GraphBuffers<Offset> graph{};
    graph.src = src_out.data_ptr<int32_t>();
    graph.dst = dst_out.data_ptr<int32_t>();
    graph.src_rowptr = src_rowptr.data_ptr<Offset>();
    graph.dst_rowptr = dst_rowptr.data_ptr<Offset>();
    graph.edge_ids_by_src = edge_ids_by_src.data_ptr<Offset>();
    graph.edge_ids_by_dst = edge_ids_by_dst.data_ptr<Offset>();
    graph.node_order = node_order.data_ptr<int32_t>();
    graph.source_node_order = source_node_order.data_ptr<int32_t>();
    graph.node_tiers = node_tiers.data_ptr<int32_t>();
    graph.segment_nodes = segment_nodes.data_ptr<int32_t>();
    graph.segment_begin = segment_begin.data_ptr<Offset>();
    graph.segment_end = segment_end.data_ptr<Offset>();
    graph.segment_rowptr = segment_rowptr.data_ptr<Offset>();
    graph.segment_order = segment_order.data_ptr<Offset>();
    graph.node_order_scratch = node_order_scratch;
    graph.source_order_scratch = node_order_scratch + num_nodes;
    graph.segment_order_scratch = segment_order_scratch;
    graph.order_histogram = segment_order_scratch + segment_capacity;
    graph.order_counts = graph.order_histogram + histogram_slots;
    graph.edge_counts = reinterpret_cast<Offset*>(work + node_scratch_words + scan_words + scratch_words);
    graph.radix_scan_state = radix_scan_state;
    graph.csr_scan_state = csr_scan_state;
    graph.degree_scan_state = degree_scan_state;
    graph.status = reinterpret_cast<GraphStatus*>(status.data_ptr<int64_t>());
    const void* src_pointer = src.const_data_ptr();
    const void* dst_pointer = dst.const_data_ptr();
    bool src_wide = src.scalar_type() == at::kLong, dst_wide = dst.scalar_type() == at::kLong;
    int64_t src_stride = src.stride(0), dst_stride = dst.stride(0);
    const auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
    void* arguments[] = {&src_pointer, &dst_pointer, &src_wide, &dst_wide, &src_stride, &dst_stride,
                         &num_nodes, &num_edges, &segment_size, &radix_bits, &graph};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(make_graph_kernel<Offset>),
                                             dim3(blocks), dim3(256), arguments, 0, stream));
    auto cpu = status.to(at::kCPU);
    const auto* values = reinterpret_cast<const GraphStatus*>(cpu.data_ptr<int64_t>());
    if (num_edges && values->min_index < 0)
        throw pybind11::value_error("graph indices must be nonnegative.");
    if (num_edges && values->max_index >= num_nodes)
        throw pybind11::value_error("num_nodes=" + std::to_string(num_nodes) +
            " must be nonnegative and at least " +
            std::to_string(static_cast<unsigned long long>(values->max_index) + 1ULL) +
            " for the supplied indices.");
    if (values->unsorted)
        throw pybind11::value_error(
            "dst must be sorted in ascending order; src and edge data must follow the same order.");
    const int64_t num_segments = values->num_segments;
    return {src_out, dst_out, src_rowptr, dst_rowptr, edge_ids_by_src, edge_ids_by_dst,
            node_order, source_node_order, node_tiers,
            segment_nodes.narrow(0, 0, num_segments), segment_begin.narrow(0, 0, num_segments),
            segment_end.narrow(0, 0, num_segments), segment_rowptr,
            segment_order.narrow(0, 0, num_segments)};
}

std::vector<at::Tensor> make_graph_tables(
    const at::Tensor& src, const at::Tensor& dst,
    int64_t num_nodes, int offset_bits, int segment_size) {
    const c10::cuda::CUDAGuard guard(src.device());
    if (offset_bits == 32) return fused_graph_tables<int32_t>(src, dst, num_nodes, segment_size);
    return fused_graph_tables<int64_t>(src, dst, num_nodes, segment_size);
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

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("graph_bounds", &graph_bounds);
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
