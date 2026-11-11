"""Fused feature and parameter gradients for symmetric contractions."""

from collections import Counter, defaultdict
from string import Template

from .utils import (
    BLOCK_SIZE, CHANNEL_TILE, MAX_BLOCKS, SHARED_MEMORY_BYTES, WEIGHT_GROUP_TILE,
    BACKWARD_CG_ROWS, BACKWARD_PRIVATE_REGISTER_WORDS, BACKWARD_PACKET_EDGE_BUDGET,
    BACKWARD_PACKET_ROOT_BUDGET, BACKWARD_PARAMETER_CTA_WAVES, CG_PACKET_REGISTER_OVERHEAD,
    BACKWARD_REGISTER_ROWS, BACKWARD_REGISTER_PARAMETER_WORDS, BACKWARD_REGISTER_LIVE_WORDS,
    BACKWARD_REGISTER_TERM_BUDGET, BACKWARD_REGISTER_CTA_WAVES, BACKWARD_REGISTER_SINGLE_CTA_WAVES,
    BACKWARD_FP64_REGISTER_PARAMETER_WORDS,
    BACKWARD_FP64_REGISTER_CTA_WAVES, BACKWARD_FP64_BLOCK_MIN_CTAS,
    polynomial_expression, polynomial_groups,
    bounded_cg_polynomials, cg_member_constants, reduced_pair_groups,
)


DIRECT_SCATTER_LIVE_WORDS = 44

DIRECT_SCATTER_CHECK = r"""
            bool direct_scatter = false;
            if constexpr (DirectScatter) {
#if __CUDA_ARCH__ >= 700
                // Each warp checks the complete row tile. Direct scatter costs at
                // most 4/3 as many weight updates as reduction for these tiles.
                const int warp_lane = threadIdx.x & 31;
                const int check_row = warp_lane % RowsPerBlock;
                const int nonzeros = attribute_counts[check_row];
                if (!__any_sync(0xffffffffu, nonzeros > 1)) {
                    const unsigned active = __ballot_sync(0xffffffffu, nonzeros && warp_lane < RowsPerBlock);
                    const int element = nonzeros ? attribute_elements[check_row][0] : -1;
                    const unsigned peers = __match_any_sync(0xffffffffu, element);
                    const bool leader = nonzeros && !(peers & ((1u << warp_lane) - 1));
                    const unsigned distinct = __ballot_sync(0xffffffffu, leader);
                    direct_scatter = __popc(distinct) * 4 >= __popc(active) * 3;
                }
#endif
            }
""".rstrip()


WARP_EXCHANGE_SOURCE = r"""
#if CUB_VERSION >= 300000
template <typename T, int Items, int Threads>
using ShuffleWarpExchange = cub::WarpExchange<T, Items, Threads, cub::WARP_EXCHANGE_SHUFFLE>;
#else
template <typename T, int Items, int Threads>
using ShuffleWarpExchange = cub::WarpExchange<T, Items, Threads, 0, cub::WARP_EXCHANGE_SHUFFLE>;
#endif
"""


CUDA_HEADER = r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros_like.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/warp/warp_exchange.cuh>
#include <algorithm>
#include <optional>
#include <limits>
#include <tuple>

using Element = ${element_type};
using Packet = ${packet_type};
using Index = int64_t;${layout_constants}
constexpr int Channels = ${channels};
constexpr int InputDim = ${input_dim};
constexpr int OutputDim = ${output_dim};
constexpr int Elements = ${elements};
constexpr int Paths = ${paths};
constexpr int Groups = ${groups};
constexpr int ReductionGroups = ${reduce_tile};
constexpr int ComputeGroups = ${compute_tile};
constexpr int ReductionChunks = ReductionGroups / ComputeGroups;
constexpr int ChannelTile = ${channel_tile};
constexpr int RowsPerBlock = ${rows_per_block};
constexpr int PacketWidth = 16 / sizeof(Element);
constexpr int ProductStride = RowsPerBlock + PacketWidth;
constexpr int ElementWorkers = ${element_workers};
constexpr int GroupWorkers = RowsPerBlock / ElementWorkers;
constexpr int Threads = RowsPerBlock * ChannelTile;
constexpr int ChannelTiles = (Channels + ChannelTile - 1) / ChannelTile;
constexpr int ChannelGroup = ChannelTiles < 4 ? ChannelTiles : 4;
constexpr int ChannelGroups = (ChannelTiles + ChannelGroup - 1) / ChannelGroup;
constexpr int LastChannelGroup = ChannelTiles - (ChannelGroups - 1) * ChannelGroup;

template<typename Index>
__device__ __forceinline__ void add_weight_gradient(
    int group, int element, int channel, Element value, Element* dw
) {
    if (value == Element(0)) return;
    const Index base = Index(element) * ${weight_element_stride} + channel;
    switch (group) {
        /* weight_scatter */
    }
}

"""

CUDA_HEADER += WARP_EXCHANGE_SOURCE


BLOCK_CUDA_SOURCE = r"""template<typename Index, bool WriteX, bool WriteWeights, bool AccumulateWeights = false>
__global__ __launch_bounds__(${thread_count}, AccumulateWeights ? ${fp64_block_min_ctas} : /* weight_launch_bounds */) void symmetric_contraction_bwd_${precision}(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    Element* __restrict__ dx, Element* __restrict__ dw, Index rows, Index splits
) {
    constexpr int ActiveGroups = ${active_groups};/* direct_scatter_constant */
    __shared__ __align__(16) Element attribute_values[Elements][ProductStride];
    __shared__ int attribute_elements[RowsPerBlock][Elements];
    __shared__ int attribute_counts[RowsPerBlock];
    __shared__ int active_count;
    __shared__ int active_species[WriteWeights && !AccumulateWeights ? Elements : 1];
    __shared__ __align__(16) Element products[WriteWeights ? ReductionGroups * ChannelTile * ProductStride : 1];
    __shared__ Element projected_weights[AccumulateWeights && WriteX && ActiveGroups ? ActiveGroups * Elements * ChannelTile : 1];
    __shared__ Element weight_totals[AccumulateWeights && WriteWeights && ActiveGroups ? ActiveGroups * Elements * ChannelTile : 1];
    const int lane = threadIdx.x & (ChannelTile - 1);
    const int local_row = threadIdx.x / ChannelTile;
    constexpr int ChannelTasks = AccumulateWeights ? ChannelTiles : ChannelGroups;
    for (int channel_group = blockIdx.y; channel_group < ChannelTasks; channel_group += gridDim.y) {
        const int channel = channel_group * ChannelTile + lane;
        if constexpr (AccumulateWeights) {
            for (int item = threadIdx.x; item < ActiveGroups * Elements * ChannelTile; item += Threads) {
                if constexpr (WriteWeights) weight_totals[item] = 0;
                if constexpr (WriteX) {
                    const int element = (item / ChannelTile) % Elements;
                    const int group = item / (ChannelTile * Elements);
                    Element projected = 0;
                    if (channel < Channels) {
                        const Index base = Index(element) * ${weight_element_stride} + channel;
                        switch (group) {
                            /* weight_projection */
                        }
                    }
                    projected_weights[item] = projected;
                }
            }
            __syncthreads();
        }
        const int width = AccumulateWeights ? 1 : (channel_group == ChannelGroups - 1 ? LastChannelGroup : ChannelGroup);
        for (Index tile = blockIdx.x; tile < Index(splits) * width; tile += gridDim.x) {
            const Index node_tile = AccumulateWeights ? tile : (channel_group == ChannelGroups - 1 ? tile / LastChannelGroup : tile / ChannelGroup);
            const int column = channel_group == ChannelGroups - 1 ? tile % LastChannelGroup : tile % ChannelGroup;
            const int channel_tile = AccumulateWeights ? channel_group : channel_group * ChannelGroup + column;
            const int channel = channel_tile * ChannelTile + lane;
            const Index batch = node_tile * RowsPerBlock;
            const Index end = batch + RowsPerBlock < rows ? batch + RowsPerBlock : rows;
            const Index row = batch + local_row;
/* initialize_tile_weights */            int count = 0;
            for (int first = 0; first < Elements; first += ChannelTile) {
                const int element = first + lane;
                const Element attribute = row < end && element < Elements
                    ? attrs[row * Elements + element] : Element(0);
                if (element < Elements) attribute_values[element][local_row] = attribute;
                const unsigned active = (__ballot_sync(0xffffffffu, attribute != Element(0)) >> ((threadIdx.x & 31) - lane)) & ((1u << ChannelTile) - 1);
                if (attribute != Element(0)) {
                    const int position = count + __popc(active & ((1u << lane) - 1));
                    attribute_elements[local_row][position] = element;
                }
                count += __popc(active);
            }
            if (lane == 0) attribute_counts[local_row] = count;
            if constexpr (WriteWeights && !AccumulateWeights) {
                if (threadIdx.x == 0) active_count = 0;
            }
            __syncthreads();/* direct_scatter_check */
            if constexpr (WriteWeights && !AccumulateWeights) {
                for (int word = threadIdx.x >> 5; /* skip_direct_scatter */word < (Elements + 31) / 32; word += Threads / 32) {
                    const int element = word * 32 + (threadIdx.x & 31);
                    bool active = false;
                    if (element < Elements) {
                        #pragma unroll
                        for (int local = 0; local < RowsPerBlock; ++local)
                            active |= attribute_values[element][local] != Element(0);
                    }
                    const unsigned mask = __ballot_sync(0xffffffffu, active);
                    int first = 0;
                    if ((threadIdx.x & 31) == 0) first = atomicAdd(&active_count, __popc(mask));
                    first = __shfl_sync(0xffffffffu, first, 0);
                    if (active) active_species[first + __popc(mask & ((1u << (threadIdx.x & 31)) - 1))] = element;
                }
            }
            const bool valid = row < end && channel < Channels;
            const Element* features = valid ? x + ${feature_row_offset} : x;
            const Element* grad = valid ? grad_out + row * OutputDim : grad_out;
            /* feature_loads */
            Element gradient[WriteX ? InputDim : 1] = {};
/* sparse_begin */            int num_windows = ${num_compute_windows};
            /* active_windows */
            #pragma unroll ${window_unroll}
            for (int window = 0; window < num_windows; ++window) {
                if constexpr (WriteWeights) {
                    if (/* skip_direct_scatter */window % ReductionChunks == 0) {
                        #pragma unroll
                        for (int group = 0; group < ReductionGroups; ++group)
                            products[(group * ChannelTile + lane) * ProductStride + local_row] = 0;
                    }
                }
                if (valid && /* active_attributes */) {
                    switch (window) {
                        /* polynomials */
                    }
                }
                if constexpr (WriteWeights) {
                    if (/* reduce_weights */) {
                        __syncthreads();
                        if (channel < Channels) {
                            const int num_active = AccumulateWeights ? Elements : active_count;
                            const int element_worker = local_row & (ElementWorkers - 1);
                            const int group_worker = (local_row / ElementWorkers) % GroupWorkers;
                            #pragma unroll
                            for (int local_group = group_worker; local_group < ReductionGroups; local_group += GroupWorkers) {
                                const int group = (window / ReductionChunks) * ReductionGroups + local_group;
                                #pragma unroll ${reduction_unroll}
                                for (int element_index = element_worker; element_index < Elements && element_index < num_active;
                                     element_index += ElementWorkers) {
                                    if (group < (AccumulateWeights ? ActiveGroups : Groups)${group_support_guard}) {
                                        const int element = AccumulateWeights ? element_index : active_species[element_index];
                                        Element sum = 0;
                                        #pragma unroll
                                        for (int local = 0; local < RowsPerBlock; local += PacketWidth) {
                                            const Packet attributes = *reinterpret_cast<const Packet*>(&attribute_values[element][local]);
                                            const Packet values = *reinterpret_cast<const Packet*>(&products[(local_group * ChannelTile + lane) * ProductStride + local]);
                                            /* reduction_values */
                                        }
                                        if constexpr (AccumulateWeights)
                                            weight_totals[(group * Elements + element) * ChannelTile + lane] += sum;
                                        else add_weight_gradient<Index>(group, element, channel, sum, dw);
                                    }
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
/* sparse_end */            /* feature_stores_block */
            __syncthreads();
        }
        if constexpr (AccumulateWeights) {
            if constexpr (WriteWeights) {
                const int element_worker = local_row & (ElementWorkers - 1);
                const int group_worker = (local_row / ElementWorkers) % GroupWorkers;
                if (channel < Channels) {
                    for (int group = group_worker; group < ActiveGroups; group += GroupWorkers) {
                        for (int element = element_worker; element < Elements; element += ElementWorkers) {
                            const Element value = weight_totals[(group * Elements + element) * ChannelTile + lane];
                            add_weight_gradient<Index>(group, element, channel, value, dw);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

"""


ATTRIBUTE_CUDA_SOURCE = r"""__device__ __forceinline__ Element contraction_weight_gradient(
    int group, const Element* features, const Element* grad, int channel
) {
    /* attribute_loads */
    switch (group) {
        /* attribute_polynomials */
        default: return Element(0);
    }
}

__global__ void symmetric_contraction_bwd_da_${precision}_i64(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ grad_out, Element* __restrict__ da, int64_t rows
) {
    __shared__ Element sums[128];
    for (Index task = blockIdx.x; task < rows * Elements; task += gridDim.x) {
        const Index row = task / Elements;
        const int element = task % Elements;
        Element sum = 0;
        for (int channel = threadIdx.x; channel < Channels; channel += blockDim.x) {
            const Element* features = x + ${feature_row_offset};
            const Element* grad = grad_out + row * OutputDim;
            const Element* weight = weights + Index(element) * ${weight_element_stride} + channel;
            /* attribute_gradient */
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        #pragma unroll
        for (int stride = 64; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) da[task] = sums[0];
        __syncthreads();
    }
}

"""


CUDA_BINDING = r"""std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>> run(
    at::Tensor x, at::Tensor weights, at::Tensor attrs, at::Tensor grad_out,
    bool need_x, bool need_weights, bool need_attrs
) {
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t rows = x.size(0);
    std::optional<at::Tensor> dx, dw, da;
    if (need_x) dx = at::empty_like(x);
    if (need_weights) dw = /* weight_allocation */;
    if (need_attrs) da = at::empty_like(attrs);
    if (rows == 0) return {dx, dw, da};
    auto stream = at::cuda::getCurrentCUDAStream();
    if (need_x || need_weights) {
        const int64_t splits = (rows + RowsPerBlock - 1) / RowsPerBlock;
        const dim3 blocks(/* grid_x */,
                          /* grid_y */);
        const int64_t limit = std::numeric_limits<int32_t>::max();
        const bool small_index = rows <= limit - /* row_limit */ &&
            splits * ChannelGroup <= limit - blocks.x &&
            x.numel() <= limit && weights.numel() <= limit &&
            attrs.numel() <= limit && grad_out.numel() <= limit;
        /* launch_index */        #define LAUNCH_BACKWARD(WRITE_X, WRITE_W) do { \
            if (small_index) { LAUNCH_INDEX(int32_t, WRITE_X, WRITE_W); } \
            else { LAUNCH_INDEX(int64_t, WRITE_X, WRITE_W); } \
        } while (false)
        if (need_x) {
            if (need_weights) LAUNCH_BACKWARD(true, true);
            else LAUNCH_BACKWARD(true, false);
        } else LAUNCH_BACKWARD(false, true);
        #undef LAUNCH_BACKWARD
        #undef LAUNCH_INDEX
    }
    if (need_attrs) {
        const int blocks = std::min<int64_t>(${max_blocks}, rows * Elements);
        symmetric_contraction_bwd_da_${precision}_i64<<<blocks, 128, 0, stream>>>(
            x.data_ptr<Element>(), weights.data_ptr<Element>(), grad_out.data_ptr<Element>(), da->data_ptr<Element>(), rows);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dw, da};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""


def polynomial_backward_code(terms, value, scale, depth=0):
    if not terms[0][0]:
        coefficient = sum(coefficient for _, coefficient in terms)
        return [f"{value} = Element({coefficient:.17g});"]
    children = defaultdict(list)
    for inputs, coefficient in terms:
        children[inputs[0]].append((inputs[1:], coefficient))
    code = [f"{value} = 0;"]
    for component, remaining in sorted(children.items()):
        child, child_scale = f"child{depth}", f"scale{depth}"
        code += ["{", f"    Element {child};", f"    const Element {child_scale} = ({scale}) * values[{component}];"]
        code += ["    " + line for line in polynomial_backward_code(remaining, child, child_scale, depth + 1)]
        code += [f"    if constexpr (WriteX) gradient[{component}] = fma(({scale}), {child}, gradient[{component}]);",
                f"    {value} = fma(values[{component}], {child}, {value});", "}"]
    return code



def binary_groups(meta, group_tile):
    binary = meta.get("binary")
    from_dag = "cg_dag" in meta
    if not from_dag and (binary is None or meta.get("correlation", 3) > 3):
        input_counts = meta.get("input_channels", (meta["channels"],) * meta["input_dim"])
        output_counts = meta.get("output_channels", (meta["channels"],) * len(meta["output_offsets"]))
        groups = list(polynomial_groups(meta)) or [((), ())]
        groups.sort(key=lambda group: -max((min(output_counts[output],
                                               *(input_counts[component] for component in inputs))
                                           for output, inputs, _ in group[0]), default=0))
        windows = []
        for first in range(0, len(groups), group_tile):
            consumers = [(local, terms) for local, (terms, _) in enumerate(groups[first:first + group_tile]) if terms]
            windows.append([({}, consumers)] if consumers else [])
        groups.extend([((), ())] * (len(windows) * group_tile - len(groups)))
        return groups, windows
    if from_dag:
        pairs, weight_paths, limits = reduced_pair_groups(meta)
        input_dim = meta["input_dim"]
        items = []
        for pair, paths in pairs:
            components = {component for component, _, _ in pair}
            consumers = defaultdict(list)
            for basis, outputs in paths:
                outer = tuple(sorted(
                    (output, (input_dim + inner,) if raw is None else (raw, input_dim + inner), coefficient)
                    for output, entries in outputs.items()
                    for raw, inner, coefficient in entries if inner in components
                ))
                consumers[limits[basis]].append((basis, outer))
            for bucket in consumers.values():
                for first in range(0, len(bucket), group_tile):
                    items.append((pair, bucket[first:first + group_tile]))
    else:
        weight_paths = binary["compute_weight_paths"]
        selected = {path["basis_path"]: path for path in binary["selected_paths"]}
        blocks = tuple(binary["local_cg_blocks"].values())
        items = []
        for pair in binary["pair_groups"]:
            consumers = []
            available = {component for component, _, _ in pair["terms"]}
            for basis, _, _, sign in pair["consumers"]:
                path = selected[basis]
                outer = defaultdict(float)
                if path["degree"] == 2:
                    for component in range(path["output_components"]):
                        if component not in available:
                            continue
                        outer[path["output_component"] + component, (meta["input_dim"] + component,)] += sign
                else:
                    tensor = next(block["tensor"] for block in blocks if block["tree"] == path["tree"])
                    right = path["tree"]["args"][1]["start"]
                    for output, inner, component in tensor.nonzero().tolist():
                        if inner not in available:
                            continue
                        inputs = (right + component, meta["input_dim"] + inner)
                        outer[path["output_component"] + output, inputs] += sign * tensor[output, inner, component].item()
                terms = tuple((*key, value) for key, value in sorted(outer.items()) if value != 0)
                consumers.append((basis, terms))
            for first in range(0, len(consumers), group_tile):
                items.append((pair["terms"], consumers[first:first + group_tile]))
        for path in binary["selected_paths"]:
            if path["degree"] == 1:
                terms = tuple((path["output_component"] + component,
                               (path["tree"]["start"] + component,), 1.0)
                              for component in range(path["output_components"]))
                items.append(((), [(path["basis_path"], terms)]))
    input_counts = meta.get("input_channels", (meta["channels"],) * meta["input_dim"])
    output_counts = meta.get("output_channels", (meta["channels"],) * len(meta["output_offsets"]))
    supported_items = []
    for pair, consumers in items:
        pair_inputs = {component for _, inputs, _ in pair for component in inputs}
        support = max((min(output_counts[output], *(input_counts[component] for component in pair_inputs),
                           *(input_counts[component] for component in inputs if component < meta["input_dim"]))
                       for _, terms in consumers for output, inputs, _ in terms), default=0)
        supported_items.append((support, pair, consumers))
    packed = []
    for support, pair, consumers in sorted(supported_items, key=lambda item: (-item[0], -len(item[2]))):
        for window in packed:
            if sum(len(paths) for _, paths in window) + len(consumers) <= group_tile:
                window.append((pair, consumers))
                break
        else:
            packed.append([(pair, consumers)])
    packed = packed or [[]]
    groups, windows = [], []
    for window in packed:
        local_group = 0
        entries = []
        for pair, consumers in window:
            pair_terms = defaultdict(list)
            for component, inputs, coefficient in pair:
                pair_terms[component].append((inputs, coefficient))
            outputs = []
            for basis, outer in consumers:
                polynomial = defaultdict(float)
                for output, inputs, coefficient in outer:
                    if inputs[-1] < meta["input_dim"]:
                        polynomial[output, inputs] += coefficient
                    else:
                        for left, value in pair_terms[inputs[-1] - meta["input_dim"]]:
                            polynomial[output, tuple(sorted((*left, *inputs[:-1])))] += value * coefficient
                expanded = tuple((*key, value) for key, value in sorted(polynomial.items()) if value != 0)
                groups.append((expanded, weight_paths[basis]))
                outputs.append((local_group, outer))
                local_group += 1
            entries.append((pair_terms, outputs))
        groups.extend([((), ())] * (group_tile - local_group))
        windows.append(entries)
    return groups, windows


def binary_code(meta, groups, pair_windows, reduction_groups, projected_weights=False, scatter_rows=False):
    compute_groups = len(groups) // len(pair_windows)
    windows, attribute_polynomials, attribute_gradient, weight_scatter = [], [], [], []
    prune = any(count != meta["channels"] for count in (*meta["input_channels"], *meta["output_channels"]))
    weight_stride = "WeightSize" if prune else "Paths * Channels"
    gradients = [f"grad[{offset} + channel * {stride}]"
                for offset, stride in zip(meta["output_offsets"], meta["output_strides"])]
    weight_loads = [f"weights[base + Index({path}) * Channels]" for path in range(meta["num_paths"])]
    attribute_weight_loads = [f"weight[Index({path}) * Channels]" for path in range(meta["num_paths"])]
    if prune:
        gradients = [f"(channel < {count} ? {value} : Element(0))"
                    for count, value in zip(meta["output_channels"], gradients)]
        weight_loads = [f"(channel < {count} ? weights[base + Index({offset})] : Element(0))"
                        for count, offset in zip(meta["path_channels"], meta["path_offsets"])]
        attribute_weight_loads = [f"(channel < {count} ? weight[Index({offset})] : Element(0))"
                                  for count, offset in zip(meta["path_channels"], meta["path_offsets"])]
    active_channels = [max((min(meta["output_channels"][output], *(meta["input_channels"][component] for component in inputs))
                            for output, inputs, _ in terms), default=0)
                       if prune else meta["channels"] for terms, _ in groups]
    if not scatter_rows:
        for group, (terms, members) in enumerate(groups):
            scatter = [f"case {group}: {{"]
            for path, coefficient in members:
                if prune:
                    scatter += [f"    if (channel < {meta['path_channels'][path]}) atomicAdd(dw + base + Index({meta['path_offsets'][path]}), Element({coefficient:.17g}) * value);"]
                else:
                    scatter += [f"    atomicAdd(dw + base + Index({path}) * Channels, Element({coefficient:.17g}) * value);"]
            weight_scatter += scatter + ["    break;", "}"]
            by_output = defaultdict(list)
            for output, inputs, coefficient in terms:
                by_output[output].append((inputs, coefficient))
            expressions = ["Element value = 0;"]
            for output, polynomial in sorted(by_output.items()):
                grad = gradients[output]
                expressions += [f"value = fma({grad}, {polynomial_expression(polynomial)}, value);"]
            attribute_polynomials += [f"case {group}: {{"] + ([f"    if (channel >= {active_channels[group]}) return Element(0);"] if prune else []) + ["    " + line for line in expressions] + ["    return value;", "}"]
            attribute_gradient += ["{", "    Element channel_weight = 0;"]
            for path, coefficient in members:
                attribute_gradient += [f"    channel_weight = fma(Element({coefficient:.17g}), {attribute_weight_loads[path]}, channel_weight);"]
            attribute_gradient += [f"    sum = fma(channel_weight, contraction_weight_gradient({group}, features, grad, channel), sum);", "}"]

    if scatter_rows:
        gradients = [f"output_gradient_values[{index}]" for index in range(len(gradients))]
    for window, pairs in enumerate(pair_windows):
        code = [f"case {window}: {{"]
        first_weights, remaining_weights = [], []
        for local, (_, members) in enumerate(groups[window * compute_groups:(window + 1) * compute_groups]):
            code += [f"    Element mixed_weight{local} = 0;"]
            arithmetic = ["{", "    Element channel_weight = 0;"]
            if projected_weights:
                group = window * compute_groups + local
                value = (f"projected_weights[({group} * Elements + element) * ChannelTile + lane]"
                         if groups[group][0] and members else 'Element(0)')
                arithmetic += [f"    if constexpr (AccumulateWeights) channel_weight = {value};", "    else {"]
            arithmetic += [f"    channel_weight = fma(Element({coefficient:.17g}), {weight_loads[path]}, channel_weight);" for path, coefficient in members]
            if projected_weights:
                arithmetic += ["    }"]
            support = active_channels[window * compute_groups + local]
            guard = prune and support < meta["channels"]
            first_weight = "channel_weight" if scatter_rows else "attribute * channel_weight"
            first_weights += ([f"if (channel < {support}) {{"] if guard else []) + arithmetic + [f"    mixed_weight{local} = {first_weight};", "}"] + (["}"] if guard else [])
            remaining_weights += ([f"if (channel < {support}) {{"] if guard else []) + arithmetic + [f"    mixed_weight{local} = fma(attribute, channel_weight, mixed_weight{local});", "}"] + (["}"] if guard else [])
        code += ["    if constexpr (WriteX) {", "        {",
                ("            const int element = first_element;" if scatter_rows else
                 "            const int element = attribute_elements[local_row][0];"),
                ("            const Element attribute = first_attribute;" if scatter_rows else
                 "            const Element attribute = attribute_values[element][local_row];"),
                f"            const Index base = Index(element) * {weight_stride} + channel;"]
        code += ["            " + line for line in first_weights]
        count = "num_attributes" if scatter_rows else "attribute_counts[local_row]"
        code += ["        }"]
        if not scatter_rows:
            code += [f"        for (int item = 1; item < {count}; ++item) {{",
                     "            const int element = attribute_elements[local_row][item];",
                     "            const Element attribute = attribute_values[element][local_row];",
                     f"            const Index base = Index(element) * {weight_stride} + channel;"]
            code += ["            " + line for line in remaining_weights]
            code += ["        }"]
        code += ["    }"]
        for pair, consumers in pairs:
            used = sorted({component - meta["input_dim"] for _, terms in consumers
                           for _, inputs, _ in terms for component in inputs if component >= meta["input_dim"]})
            support = max(active_channels[window * compute_groups + local] for local, _ in consumers)
            code += [f"    if (channel < {support}) {{" if prune and support < meta["channels"] else "    {"]
            for local, _ in consumers:
                code += [f"        Element gradient_product{local} = 0;"]
            for component in used or [None]:
                code += ["        {"]
                entries = []
                for local, terms in consumers:
                    by_output = defaultdict(list)
                    for output, inputs, coefficient in terms:
                        if component is not None:
                            target = meta["input_dim"] + component
                            if target not in inputs:
                                continue
                            inputs = tuple(value for value in inputs if value != target)
                        by_output[output].append((inputs, coefficient))
                    entries.extend((local, output, polynomial) for output, polynomial in sorted(by_output.items()))
                if component is not None and scatter_rows:
                    by_local = defaultdict(list)
                    for local, output, polynomial in entries:
                        by_local[local].append((output, polynomial))
                    code.append("            Element pair_seed = 0;")
                    for local, outputs in by_local.items():
                        support = active_channels[window * compute_groups + local]
                        code.append(f"            Element pair_gradient{local} = 0;")
                        for output, polynomial in outputs:
                            grad = gradients[output]
                            expression = polynomial_expression(polynomial)
                            if prune and support < meta["channels"]:
                                grad = f"(channel < {support} ? {grad} : Element(0))"
                                expression = f"(channel < {support} ? {expression} : Element(0))"
                            code.append(f"            pair_gradient{local} = fma({grad}, {expression}, pair_gradient{local});")
                        code.append(f"            if constexpr (WriteX) pair_seed = fma(mixed_weight{local}, pair_gradient{local}, pair_seed);")
                    code.append("            Element pair_value;")
                    code += ["            " + line for line in polynomial_backward_code(pair[component], "pair_value", "pair_seed")]
                    for local, outputs in by_local.items():
                        support = active_channels[window * compute_groups + local]
                        code += [f"            gradient_product{local} = fma(pair_value, pair_gradient{local}, gradient_product{local});",
                                 "            if constexpr (WriteX) {",
                                 f"                const Element weighted_pair = mixed_weight{local} * pair_value;"]
                        for output, polynomial in outputs:
                            if not any(inputs for inputs, _ in polynomial):
                                continue
                            grad = gradients[output]
                            if prune and support < meta["channels"]:
                                grad = f"(channel < {support} ? {grad} : Element(0))"
                            code += ["                {", f"                    const Element scale = weighted_pair * {grad};"]
                            for inputs, coefficient in polynomial:
                                if inputs:
                                    code.append(f"                    gradient[{inputs[0]}] = fma(scale, Element({coefficient:.17g}), gradient[{inputs[0]}]);")
                            code.append("                }")
                        code.append("            }")
                    code.append("        }")
                    continue
                if component is not None:
                    code += ["            Element pair_seed = 0;"]
                    for index, (local, output, polynomial) in enumerate(entries):
                        grad = gradients[output]
                        support = active_channels[window * compute_groups + local]
                        expression = polynomial_expression(polynomial)
                        if prune and support < meta["channels"]:
                            grad = f"(channel < {support} ? {grad} : Element(0))"
                            expression = f"(channel < {support} ? {expression} : Element(0))"
                        code += [f"            const Element outer_value{index} = {expression};",
                                f"            if constexpr (WriteX) pair_seed = fma(mixed_weight{local} * {grad}, outer_value{index}, pair_seed);"]
                    code += ["            Element pair_value;"]
                    code += ["            " + line for line in polynomial_backward_code(pair[component], "pair_value", "pair_seed")]
                for index, (local, output, polynomial) in enumerate(entries):
                    grad = gradients[output]
                    support = active_channels[window * compute_groups + local]
                    if prune and support < meta["channels"]:
                        grad = f"(channel < {support} ? {grad} : Element(0))"
                    code += ["            {", f"                const Element output_gradient = {grad};",
                            f"                const Element weighted_gradient = mixed_weight{local} * output_gradient;"]
                    if component is None:
                        code += ["                Element polynomial;"]
                        code += ["                " + line for line in polynomial_backward_code(polynomial, "polynomial", "weighted_gradient")]
                        value = "polynomial"
                    else:
                        code += ["                if constexpr (WriteX) {", "                    const Element scale = weighted_gradient * pair_value;"]
                        for inputs, coefficient in polynomial:
                            if inputs:
                                code += [f"                    gradient[{inputs[0]}] = fma(scale, Element({coefficient:.17g}), gradient[{inputs[0]}]);"]
                        code += ["                }"]
                        value = f"pair_value * outer_value{index}"
                    code += [f"                gradient_product{local} = fma(output_gradient, {value}, gradient_product{local});", "            }"]
                code += ["        }"]
            for local, _ in consumers:
                if scatter_rows:
                    group = window * compute_groups + local
                    code.append(f"        if constexpr (WriteWeights) add_weight_gradient<Index>({group}, first_element, channel, gradient_product{local}, dw);")
                else:
                    code += [
                        f"        if constexpr (WriteWeights) products[(({local + (window * compute_groups) % reduction_groups}) * ChannelTile + lane) * ProductStride + local_row] = gradient_product{local};"
                    ]
            code += ["    }"]
        windows += code + ["    break;", "}"]
    return windows, attribute_polynomials, attribute_gradient, weight_scatter


DEFAULT_BLOCK_LAUNCH = r"""#define LAUNCH_INDEX(INDEX, WRITE_X, WRITE_W) \
            symmetric_contraction_bwd_${precision}<INDEX, WRITE_X, WRITE_W><<<blocks, Threads, 0, stream>>>( \
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(), grad_out.data_ptr<Element>(), \
                need_x ? dx->data_ptr<Element>() : nullptr, need_weights ? dw->data_ptr<Element>() : nullptr, \
                INDEX(rows), INDEX(splits))
"""


FP64_BLOCK_LAUNCH = r"""#define LAUNCH_INDEX(INDEX, WRITE_X, WRITE_W) do { \
            auto kernel = WRITE_W ? /* weights_kernel */ \
                                  : symmetric_contraction_bwd_${precision}<INDEX, WRITE_X, WRITE_W>; \
            kernel<<<blocks, Threads, 0, stream>>>( \
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(), grad_out.data_ptr<Element>(), \
                need_x ? dx->data_ptr<Element>() : nullptr, need_weights ? dw->data_ptr<Element>() : nullptr, \
                INDEX(rows), INDEX(splits)); \
        } while (false)
"""


REGISTER_CUDA_SOURCE = r"""template<typename Index, bool WriteX, bool WriteWeights>
__global__ __launch_bounds__(${thread_count}, /* register_min_blocks */) void symmetric_contraction_bwd/* register_suffix */_${precision}(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    Element* __restrict__ dx, Element* __restrict__ dw, Index rows, Index splits
) {
    __shared__ Element partials[WriteWeights ? /* register_values */ * ChannelTile * (RowsPerBlock + 1) : 1];
    __shared__ Element projected_weights[WriteX ? /* register_values */ * ChannelTile : 1];
    const int lane = threadIdx.x % ChannelTile;
    const int local_row = threadIdx.x / ChannelTile;
    for (int channel_tile = blockIdx.y; channel_tile < ChannelTiles; channel_tile += gridDim.y) {
        const int channel = channel_tile * ChannelTile + lane;
        if constexpr (WriteX) {
/* weight_projection */
            __syncthreads();
        }
/* weight_totals */
        for (Index row = Index(blockIdx.x) * RowsPerBlock + local_row; row < rows; row += Index(gridDim.x) * RowsPerBlock) {
            const Index end = rows;
            const bool valid = channel < Channels;
            const Element* features = valid ? x + ${feature_row_offset} : x;
            const Element* grad = grad_out + row * OutputDim;
/* register_loads */
            Element gradient[WriteX ? InputDim : 1] = {};
            if (valid) {
/* register_attributes */
/* register_polynomials */
            }
            if constexpr (WriteX) {
/* register_stores */
            }
        }
        if constexpr (WriteWeights) {
/* weight_partials */
            __syncthreads();
            if (channel < Channels) {
                for (int item = local_row; item < /* register_values */; item += RowsPerBlock) {
                    const int element = item % Elements;
                    Element sum = 0;
                    #pragma unroll
                    for (int row = 0; row < RowsPerBlock; ++row)
                        sum += partials[(item * ChannelTile + lane) * (RowsPerBlock + 1) + row];
                    switch (item / Elements) {
/* register_scatter */
                    }
                }
            }
            __syncthreads();
        }
/* channel_tile_barrier */    }
}
"""


def register_backward_kernel(meta, groups, loads, stores, *, fp64=False):
    """Consume each atom's parameter products before a single CTA reduction."""
    elements = meta['num_elements']
    live_groups = [index for index, (terms, members) in enumerate(groups) if terms and members]
    projection, totals, attributes, polynomials, partials, scatter = [], [], [], [], [], []
    for local, group in enumerate(live_groups):
        _, members = groups[group]
        projection += ['            for (int element = local_row; element < Elements; element += RowsPerBlock) {',
                       '                Element weight = 0;', '                if (channel < Channels) {']
        for path, coefficient in members:
            offset = meta['path_offsets'][path]
            value = f'weights[Index(element) * ${{weight_element_stride}} + Index({offset}) + channel]'
            value = f'(channel < {meta["path_channels"][path]} ? {value} : Element(0))'
            projection.append(f'                    weight = fma(Element({coefficient:.17g}), {value}, weight);')
        projection += ['                }',
                       f'                projected_weights[({local} * Elements + element) * ChannelTile + lane] = weight;',
                       '            }']
    for group in live_groups:
        for element in range(elements):
            totals.append(f'        Element total{group}_{element} = 0;')
    for element in range(elements):
        attributes.append(f'                const Element attribute{element} = attrs[row * Elements + {element}];')
    for local, group in enumerate(live_groups):
        terms, _ = groups[group]
        polynomials += ['                {', '                    Element mixed_weight = 0;']
        for element in range(elements):
            polynomials += ['                    if constexpr (WriteX) {',
                            f'                        if (attribute{element} != Element(0)) {{',
                            f'                            const Element channel_weight = projected_weights[({local} * Elements + {element}) * ChannelTile + lane];',
                            f'                            mixed_weight = fma(attribute{element}, channel_weight, mixed_weight);',
                            '                        }', '                    }']
        polynomials.append('                    Element parameter = 0;')
        by_output = defaultdict(list)
        for output, inputs, coefficient in terms:
            by_output[output].append((inputs, coefficient))
        for output, polynomial in sorted(by_output.items()):
            offset, stride = meta['output_offsets'][output], meta['output_strides'][output]
            count = meta['output_channels'][output]
            polynomials += ['                    {', f'                        const Element output_gradient = channel < {count} ? grad[{offset} + channel * {stride}] : Element(0);',
                            '                        Element polynomial;']
            polynomials += ['                        ' + line for line in polynomial_backward_code(polynomial, 'polynomial', 'mixed_weight * output_gradient')]
            polynomials += ['                        parameter = fma(output_gradient, polynomial, parameter);', '                    }']
        for element in range(elements):
            polynomials.append(f'                    if constexpr (WriteWeights) total{group}_{element} = fma(attribute{element}, parameter, total{group}_{element});')
        polynomials.append('                }')
    for local, group in enumerate(live_groups):
        for element in range(elements):
            partials.append(f'            partials[(({local * elements + element}) * ChannelTile + lane) * (RowsPerBlock + 1) + local_row] = total{group}_{element};')
    for local, group in enumerate(live_groups):
        scatter.append(f'                    case {local}: add_weight_gradient<Index>({group}, element, channel, sum, dw); break;')
    # The barrier is only needed when a block can advance to another channel tile.
    barrier = ("        if constexpr (WriteX && !WriteWeights) __syncthreads();\n"
               if (meta["channels"] + 15) // 16 > MAX_BLOCKS else "")
    return (REGISTER_CUDA_SOURCE
            .replace("/* register_suffix */", "_block" if fp64 else "")
            .replace("/* register_min_blocks */", str(BACKWARD_FP64_REGISTER_CTA_WAVES if fp64
                                                     else BACKWARD_REGISTER_CTA_WAVES))
            .replace("/* register_values */", str(len(live_groups) * elements))
            .replace("/* channel_tile_barrier */", barrier)
            .replace("/* weight_projection */", "\n".join(projection))
            .replace("/* weight_totals */", "\n".join(totals))
            .replace("/* register_loads */", "\n".join("            " + line for line in loads))
            .replace("/* register_attributes */", "\n".join(attributes))
            .replace("/* register_polynomials */", "\n".join(polynomials))
            .replace("/* register_stores */", "\n".join("                " + line for line in stores))
            .replace("/* weight_partials */", "\n".join(partials))
            .replace("/* register_scatter */", "\n".join(scatter)))


def make_backward_source(meta, dtype="float32"):
    """Choose the CG packet or polynomial path, then emit its kernel and launch."""
    dtype = str(dtype).removeprefix("torch.")
    element_bytes = 8 if dtype in ("float64", "double") else 4
    reuse_pairs = element_bytes == 4 and meta.get("use_reduced_cg", False) and meta["correlation"] == 3
    if "cg_dag" in meta and not reuse_pairs:
        polynomial = bounded_cg_polynomials(meta)
        if polynomial is None:
            if any(len(mapping) > 1 for mapping in meta['cg_dag']['weight_members']):
                return make_cg_packet_source(meta, dtype)
            return make_cg_backward_source(meta, dtype)
        meta = polynomial
    input_dim, channels = meta["input_dim"], meta["channels"]
    elements = meta["num_elements"]
    input_counts = meta["input_channels"]
    input_offsets = meta["input_offsets"]
    input_strides = meta["input_strides"]
    rows_per_block = 16 // (element_bytes // 4)
    live_words = 2 * input_dim + len(meta["output_offsets"]) + WEIGHT_GROUP_TILE
    attribute_bytes = rows_per_block * (elements * (element_bytes + 4) + 4) + elements * 16
    mask_bytes = (elements * 4 + 4 + element_bytes - 1) // element_bytes * element_bytes
    product_capacity = (SHARED_MEMORY_BYTES - attribute_bytes - mask_bytes) // ((rows_per_block + 16 // element_bytes) * 16 * element_bytes)
    reduction_groups = max(1, min(2 * WEIGHT_GROUP_TILE // (element_bytes // 4), 1 << (max(1, product_capacity).bit_length() - 1)))
    compute_groups = min(WEIGHT_GROUP_TILE, reduction_groups)
    groups, pair_windows = binary_groups(meta, compute_groups)
    if (element_bytes == 4 and elements <= rows_per_block
            and live_words <= BACKWARD_REGISTER_LIVE_WORDS
            and min(len(groups), product_capacity) >= 2 * reduction_groups):
        reduction_groups *= 2
    active_groups = max((index + 1 for index, (terms, members) in enumerate(groups) if terms and members), default=0)
    block_shared_bytes = ((2 * active_groups * elements * 16
                           + (reduction_groups * 16 + elements) * (rows_per_block + 16 // element_bytes))
                          * element_bytes + rows_per_block * (elements + 1) * 4)
    live_groups = [(terms, members) for terms, members in groups if terms and members]
    register_words = element_bytes // 4
    window_register_words = (2 * input_dim + len(meta["output_offsets"]) + len(live_groups)) * register_words
    parameter_budget = (BACKWARD_FP64_REGISTER_PARAMETER_WORDS if element_bytes == 8
                        else BACKWARD_REGISTER_PARAMETER_WORDS)
    register_totals = (len(live_groups) * elements * register_words <= parameter_budget
                       and (input_dim + len(meta['output_offsets']) + len(live_groups) * (elements + 1))
                       * register_words <= BACKWARD_REGISTER_LIVE_WORDS
                       and sum(len(terms) for terms, _ in live_groups) <= BACKWARD_REGISTER_TERM_BUDGET)
    fp64_register_totals = element_bytes == 8 and register_totals
    fp64_parameters = element_bytes == 8 and (fp64_register_totals or block_shared_bytes <= SHARED_MEMORY_BYTES)
    if register_totals and element_bytes == 4:
        rows_per_block = BACKWARD_REGISTER_ROWS
    channel_tile = 16
    uniform_channels = all(count == channels for count in (*input_counts, *meta["output_channels"]))
    if element_bytes == 4 and uniform_channels:
        channel_tile = min(16, max(4, 1 << (max(channels, input_dim) - 1).bit_length()))
    loads = [f"Element values[{input_dim}];", "if (valid) {"]
    loads += [f"    values[{component}] = channel < {count} ? features[{offset} + Index(channel) * {stride}] : Element(0);"
              for component, (count, offset, stride) in enumerate(zip(input_counts, input_offsets, input_strides))]
    loads += ["}"]

    stores, first = [], 0
    while first < input_dim:
        stop = first + 1
        while (stop < input_dim and input_counts[stop] == input_counts[first]
               and input_strides[stop] == input_strides[first]
               and input_offsets[stop] == input_offsets[first] + stop - first):
            stop += 1
        width = min(channel_tile, 32 // element_bytes, 1 << ((stop - first).bit_length() - 1))
        stores += ["{", f"    Element items[{width}] = {{" + ", ".join(f"gradient[{first + item}]" for item in range(width)) + "};",
                   f"    using Exchange = ShuffleWarpExchange<Element, {width}, {width}>;",
                   "    typename Exchange::TempStorage storage;",
                   "    Exchange(storage).BlockedToStriped(items, items);",
                   f"    const int local_lane = lane & {width - 1};",
                   "    const int first_channel = channel - local_lane;"]
        for item in range(width):
            stores += [f"    if (row < end && first_channel + {item} < {input_counts[first]})",
                       f"        dx[row * InputSize + {input_offsets[first]} + local_lane + Index(first_channel + {item}) * {input_strides[first]}] = items[{item}];"]
        stores += ["}"]
        first += width
    block_stores = stores
    max_stride = max(input_strides)
    if (max_stride > 1 and (max_stride * channel_tile + 1) * rows_per_block
            <= reduction_groups * channel_tile * (rows_per_block + 16 // element_bytes)):
        block_stores = ["if constexpr (WriteWeights) {",
                        f"using FeatureStorage = typename cub::WarpExchange<Element, {max_stride}, ChannelTile>::TempStorage;",
                        "static_assert(sizeof(FeatureStorage) * RowsPerBlock <= sizeof(products));",
                        "auto* feature_storage = reinterpret_cast<FeatureStorage*>(products);"]
        first = 0
        while first < input_dim:
            stride, count, offset = input_strides[first], input_counts[first], input_offsets[first]
            values = ", ".join(f"gradient[{first + item}]" for item in range(stride))
            block_stores += ["{", f"Element items[{stride}] = {{{values}}};",
                             f"using Exchange = cub::WarpExchange<Element, {stride}, ChannelTile>;",
                             "auto& storage = *reinterpret_cast<typename Exchange::TempStorage*>(feature_storage + local_row);",
                             "Exchange(storage).BlockedToStriped(items, items);"]
            for item in range(stride):
                index = f"Index(channel - lane) * {stride} + lane + {item * channel_tile}"
                block_stores += [f"if (row < end && {index} < {count * stride})",
                                 f"    dx[row * InputSize + {offset} + {index}] = items[{item}];"]
            block_stores += ["__syncwarp();", "}"]
            first += stride
        block_stores += ["} else {", *stores, "}"]
    store_block = "if constexpr (WriteX) {\n"
    store_block += "\n".join(block_stores) + "\n            }"
    feature_values = [f"(channel < {count} ? features[{offset} + Index(channel) * {stride}] : Element(0))"
                      for count, offset, stride in zip(input_counts, input_offsets, input_strides)]
    live_words = 2 * input_dim + len(meta['output_offsets']) + compute_groups
    # Keep the original launch bounds for dense tiles.
    weight_ctas = 3 if live_words <= 32 else (512 // (element_bytes // 4)) // (rows_per_block * channel_tile)
    direct_scatter = (element_bytes == 4 and not register_totals and elements > rows_per_block
                      and live_words <= DIRECT_SCATTER_LIVE_WORDS)
    sparse_begin, sparse_end = "", ""
    if direct_scatter:
        gradients = [f"grad[{offset} + channel * {stride}]"
                     for offset, stride in zip(meta["output_offsets"], meta["output_strides"])]
        gradients = [f"(channel < {count} ? {value} : Element(0))"
                     for count, value in zip(meta["output_channels"], gradients)]
        row_attribute_state = (
            "            const int num_attributes = attribute_counts[local_row];\n"
            "            const int first_element = num_attributes ? attribute_elements[local_row][0] : 0;\n"
            "            const Element first_attribute = num_attributes ? attribute_values[first_element][local_row] : Element(0);\n"
            "            const Element output_gradient_values[] = {"
            + ", ".join(f"first_attribute * ({value})" for value in gradients) + "};\n"
        )
        sparse_windows, *_ = binary_code(meta, groups, pair_windows, reduction_groups, scatter_rows=True)
        sparse_body = "\n".join("{" if line.startswith("case ") else line
                                for line in sparse_windows if line.strip() != "break;")
        sparse_begin = ("            if (direct_scatter) {\n"
                        "                if (valid && attribute_counts[local_row]) {\n"
                        + row_attribute_state + sparse_body + "\n"
                        "                }\n"
                        "            } else {\n")
        sparse_end = "            }\n"
    windows, attribute_polynomials, attribute_gradient, weight_scatter = binary_code(
        meta, groups, pair_windows, reduction_groups, fp64_parameters and not fp64_register_totals)
    active_windows, group_guard = "", ""
    active_channels = [max((min(meta["output_channels"][output], *(meta["input_channels"][component] for component in inputs))
                            for output, inputs, _ in terms), default=0) for terms, _ in groups]
    window_channels = [max(active_channels[first:first + compute_groups]) for first in range(0, len(groups), compute_groups)]
    layout_constants = (f"\nconstexpr int InputSize = {meta['input_size']};\n"
                        f"constexpr int WeightSize = {meta['weight_numel']};\n")
    if any(count != channels for count in (*input_counts, *meta["output_channels"])):
        layout_constants += "__device__ const int GroupChannels[] = {" + ", ".join(map(str, active_channels)) + "};\n"
        layout_constants += "__device__ const int WindowChannels[] = {" + ", ".join(map(str, window_channels)) + "};"
        active_windows = f"num_windows = 0;\n            while (num_windows < {len(pair_windows)} && channel_tile * ChannelTile < WindowChannels[num_windows]) ++num_windows;"
        group_guard = " && channel < GroupChannels[group]"
    projection = []
    if fp64_parameters and not fp64_register_totals:
        for group, (_, members) in enumerate(groups):
            projection.append(f'case {group}: {{')
            for path, coefficient in members:
                offset = meta['path_offsets'][path]
                value = f'weights[base + Index({offset})]'
                value = f'(channel < {meta["path_channels"][path]} ? {value} : Element(0))'
                projection.append(f'    projected = fma(Element({coefficient:.17g}), {value}, projected);')
            projection += ['    break;', '}']
    # FP32 register totals replace the tiled kernel. FP64 keeps the tiled
    # kernel for input-only gradients and selects a separate weight launch.
    block_kernel = BLOCK_CUDA_SOURCE
    grid_x = "std::min<int64_t>(${max_blocks}, int64_t(splits) * ChannelGroup)"
    grid_y = "std::min(ChannelGroups, 65535)"
    row_limit = "RowsPerBlock"
    launch = DEFAULT_BLOCK_LAUNCH
    if fp64_parameters:
        if fp64_register_totals:
            block_kernel += register_backward_kernel(meta, groups, loads, stores, fp64=True)
            row_limit = "(need_weights ? int64_t(blocks.x) * RowsPerBlock : RowsPerBlock)"
            weights_kernel = "symmetric_contraction_bwd_block_${precision}<INDEX, WRITE_X, WRITE_W>"
        else:
            weights_kernel = "symmetric_contraction_bwd_${precision}<INDEX, WRITE_X, WRITE_W, true>"
        channel_tiles = (channels + channel_tile - 1) // channel_tile
        waves = BACKWARD_FP64_REGISTER_CTA_WAVES if fp64_register_totals else BACKWARD_PARAMETER_CTA_WAVES
        blocks = (max(1, waves * meta["multiprocessor_count"] // channel_tiles)
                  if "multiprocessor_count" in meta else MAX_BLOCKS)
        fused_blocks = (max(1, BACKWARD_FP64_BLOCK_MIN_CTAS * meta["multiprocessor_count"] // channel_tiles)
                        if not fp64_register_totals and "multiprocessor_count" in meta else blocks)
        grid_x = (f"std::min<int64_t>(need_weights ? (need_x ? {fused_blocks} : {blocks}) : "
                  "${max_blocks}, int64_t(splits) * (need_weights ? 1 : ChannelGroup))")
        grid_y = "std::min(need_weights ? ChannelTiles : ChannelGroups, 65535)"
        launch = FP64_BLOCK_LAUNCH.replace("/* weights_kernel */", weights_kernel)
    elif register_totals and element_bytes == 4:
        block_kernel = register_backward_kernel(meta, groups, loads, stores)
        channel_tiles = (channels + channel_tile - 1) // channel_tile
        sm_count = meta.get("multiprocessor_count", 1)
        blocks = max(1, BACKWARD_REGISTER_SINGLE_CTA_WAVES * sm_count // channel_tiles)
        fused_blocks = max(1, BACKWARD_REGISTER_CTA_WAVES * sm_count // channel_tiles)
        grid_x = f"std::min<int64_t>(need_x && need_weights ? {fused_blocks} : {blocks}, splits)"
        grid_y = "std::min(ChannelTiles, 65535)"
        row_limit = "int64_t(blocks.x) * RowsPerBlock"
    block_thread_budget = (768 if element_bytes == 4 and window_register_words <= BACKWARD_REGISTER_LIVE_WORDS
                           else 512 // register_words)
    initialize_tile_weights = ""
    weight_allocation = "at::zeros_like(weights)"
    if (element_bytes == 4 and elements <= rows_per_block
            and window_register_words <= BACKWARD_REGISTER_LIVE_WORDS
            and uniform_channels and not register_totals):
        weight_allocation = "(rows > 0 && rows <= RowsPerBlock ? at::empty_like(weights) : at::zeros_like(weights))"
        initialize_tile_weights = """            if constexpr (WriteWeights) {
                if (rows <= RowsPerBlock) {
                    #pragma unroll 1
                    for (int64_t item = threadIdx.x; item < int64_t(Elements) * Paths * ChannelTile; item += Threads) {
                        const int column = channel_tile * ChannelTile + item % ChannelTile;
                        if (column < Channels) dw[(item / ChannelTile) * Channels + column] = Element(0);
                    }
                    __syncthreads();
                }
            }
"""
    source = CUDA_HEADER + block_kernel + ATTRIBUTE_CUDA_SOURCE + CUDA_BINDING
    reduce_weights = "(window + 1) % ReductionChunks == 0 || window + 1 == num_windows"
    for marker, code in {
        "grid_x": grid_x, "grid_y": grid_y, "row_limit": row_limit, "launch_index": launch,
        "initialize_tile_weights": initialize_tile_weights, "weight_allocation": weight_allocation,
        "reduction_values": "\n".join(
            f"sum = fma(attributes.{component}, values.{component}, sum);"
            for component in "xyzw"[:16 // element_bytes]
        ),
        "feature_loads": "\n".join(loads), "feature_stores_block": store_block,
        "sparse_begin": sparse_begin, "sparse_end": sparse_end,
        "active_attributes": "attribute_counts[local_row]",
        "polynomials": "\n".join(windows), "weight_scatter": "\n".join(weight_scatter),
        "attribute_loads": f"const Element values[{input_dim}] = {{" + ", ".join(feature_values) + "};",
        "attribute_polynomials": "\n".join(attribute_polynomials), "attribute_gradient": "\n".join(attribute_gradient),
        "active_windows": active_windows, "weight_projection": "\n".join(projection),
        "weight_launch_bounds": (
            f"(WriteWeights && sizeof(Element) == sizeof(float) && Elements > RowsPerBlock ? {weight_ctas} : ${{min_blocks}})"
            if direct_scatter else "${min_blocks}"
        ),
        "direct_scatter_constant": (
            "\n    constexpr bool DirectScatter = WriteWeights && !AccumulateWeights &&\n"
            "                                   sizeof(Element) == sizeof(float) && Elements > RowsPerBlock;"
            if direct_scatter else ""
        ),
        "direct_scatter_check": DIRECT_SCATTER_CHECK if direct_scatter else "",
        "skip_direct_scatter": "!direct_scatter && " if direct_scatter else "",
        "reduce_weights": f"!direct_scatter && ({reduce_weights})" if direct_scatter else reduce_weights,
    }.items():
        source = source.replace(f"/* {marker} */", code)
    return Template(source).substitute(
        precision="fp64" if element_bytes == 8 else "fp32",
        element_type="double" if element_bytes == 8 else "float",
        packet_type="double2" if element_bytes == 8 else "float4",
        channels=channels, input_dim=input_dim, output_dim=meta["output_dim"],
        elements=elements, paths=meta["num_paths"], active_groups=active_groups,
        groups=len(groups), num_compute_windows=len(pair_windows),
        reduce_tile=reduction_groups, compute_tile=compute_groups, rows_per_block=rows_per_block,
        element_workers=min(rows_per_block, 1 << (elements - 1).bit_length()),
        thread_count=rows_per_block * channel_tile, channel_tile=channel_tile,
        min_blocks=block_thread_budget // (rows_per_block * channel_tile),
        max_blocks=MAX_BLOCKS, layout_constants=layout_constants,
        fp64_block_min_ctas=BACKWARD_FP64_BLOCK_MIN_CTAS,
        reduction_unroll="" if fp64_parameters and not fp64_register_totals else "1",
        window_unroll="" if window_register_words <= BACKWARD_REGISTER_LIVE_WORDS else "1",
        weight_element_stride="WeightSize",
        feature_row_offset="row * InputSize",
        group_support_guard=group_guard,
    )


def packet_schedule(meta, selected):
    """Reuse primal and gradient slots after their final consumer."""
    graph = meta['cg_dag']
    nodes, roots = (graph['nodes'], graph['roots'])
    closure, root_by_node = (set(), defaultdict(list))

    def visit(index):
        node = nodes[index]
        if node['kind'] == 'input' or index in closure:
            return
        visit(node['left_id'])
        visit(node['right_id'])
        closure.add(index)
    for root in selected:
        root_by_node[roots[root]['node_id']].append(root)
        visit(roots[root]['node_id'])
    pending = Counter()
    for index in closure:
        node = nodes[index]
        pending.update(set((node['left_id'], node['right_id'])))
    values = {index: node['input_start'] for index, node in enumerate(nodes) if node['kind'] == 'input'}
    totals = dict(values)
    occupied = [[True] * meta['input_dim'], [True] * meta['input_dim']]
    peaks = [meta['input_dim'], meta['input_dim']]
    operations, computed, finished = ([], set(values), set())

    def allocate(plane, dimension):
        begin = meta['input_dim']
        storage = occupied[plane]
        while True:
            if len(storage) < begin + dimension:
                storage.extend([False] * (begin + dimension - len(storage)))
            if not any(storage[begin:begin + dimension]):
                storage[begin:begin + dimension] = [True] * dimension
                peaks[plane] = max(peaks[plane], begin + dimension)
                return begin
            begin += 1

    def gradient_slot(index):
        if index not in totals:
            base = allocate(1, nodes[index]['irrep_dim'])
            totals[index] = base
            operations.append(('zero', base, nodes[index]['irrep_dim']))
        return totals[index]

    def reverse(index):
        node = nodes[index]
        if node['kind'] == 'input' or index not in computed or index in finished or pending[index]:
            return
        left, right = (node['left_id'], node['right_id'])
        seed = gradient_slot(index)
        lt, rt = (gradient_slot(left), gradient_slot(right))
        operations.append(('reverse', node['cg_id'], values[left], values[right], lt, rt, seed))
        dimension = node['irrep_dim']
        for plane, mapping in ((0, values), (1, totals)):
            base = mapping.pop(index)
            occupied[plane][base:base + dimension] = [False] * dimension
        finished.add(index)
        for parent in set((left, right)):
            pending[parent] -= 1
            reverse(parent)

    def consume(index):
        for root in root_by_node[index]:
            operations.append(('root', root, values[index], gradient_slot(index)))
        reverse(index)
    for index, node in enumerate(nodes):
        if node['kind'] == 'input':
            consume(index)
    for index in sorted(closure):
        node = nodes[index]
        left, right = (node['left_id'], node['right_id'])
        destination = allocate(0, node['irrep_dim'])
        operations.append(('forward', node['cg_id'], values[left], values[right], destination))
        values[index] = destination
        computed.add(index)
        consume(index)
    assert finished == closure
    blocks = [graph['local_cg'][operation[1]] for operation in operations if operation[0] == 'forward']
    return dict(
        roots=tuple(selected),
        operations=operations,
        value_slots=peaks[0],
        total_slots=peaks[1],
        edges=sum((len(block['coefficients']) for block in blocks)),
        parameter_work=2 * meta['num_elements'] * sum((len(graph['weight_members'][roots[root]['selected_basis_id']]) for root in selected)),
        operands=max((sum(block['shape']) for block in blocks), default=0)
    )

def register_packets(meta, element_words=1, private_words=BACKWARD_PRIVATE_REGISTER_WORDS,
                     edge_budget=BACKWARD_PACKET_EDGE_BUDGET, root_budget=BACKWARD_PACKET_ROOT_BUDGET,
                     parameter_totals=False):
    graph = meta['cg_dag']
    groups = defaultdict(list)
    for index, root in enumerate(graph['roots']):
        node = graph['nodes'][root['node_id']]
        groups[node['left_id'] if node['kind'] == 'coupling' else root['node_id']].append(index)

    def fits(packet):
        extra = len(packet['roots']) * meta['num_elements'] if parameter_totals else 0
        words = (packet['value_slots'] + packet['total_slots'] + packet['operands'] + 2 * meta['num_elements'] + extra) * element_words + CG_PACKET_REGISTER_OVERHEAD
        return words <= private_words and packet['edges'] <= edge_budget and (parameter_totals or packet['parameter_work'] <= edge_budget) and (len(packet['roots']) <= root_budget)
    shared_slots = meta.get('shared_memory_bytes', SHARED_MEMORY_BYTES) // (BACKWARD_CG_ROWS * 4 * element_words) - meta['input_dim'] - meta['num_elements']

    def fits_shared(packet):
        return packet['value_slots'] + packet['total_slots'] <= shared_slots and packet['edges'] <= edge_budget and (parameter_totals or packet['parameter_work'] <= edge_budget) and (len(packet['roots']) <= root_budget)

    def finish(selected):
        packet = packet_schedule(meta, selected)
        packet['shared'] = not fits(packet)
        return packet
    selected, packets = ([], [])
    for group in groups.values():
        for root in group:
            candidate = packet_schedule(meta, (*selected, root))
            if selected and (not (fits(candidate) or (not fits(packet_schedule(meta, selected)) and fits_shared(candidate)))):
                packets.append(finish(selected))
                selected = []
            selected.append(root)
    if selected:
        packets.append(finish(selected))
    return packets

def local_functions(meta, block_ids, shared=False):
    """Keep each small CG contraction in registers before writing its result."""
    code = []
    prefix = 'shared_' if shared else ''
    worker = ', int worker' if shared else ''

    def address(base, component):
        return f'({base}+{component})*Threads+worker' if shared else f'{base}+{component}'
    for index in sorted(block_ids):
        block = meta['cg_dag']['local_cg'][index]
        edges = [(output, block['left_components'][edge], block['right_components'][edge], block['coefficients'][edge]) for output in range(block['shape'][0]) for edge in range(block['output_offsets'][output], block['output_offsets'][output + 1])]
        inline = '__noinline__' if shared else '__forceinline__'
        forward = [f'__device__ {inline} void {prefix}cg_forward_{index}(Element* values, int left, int right, int destination{worker}) {{']
        reverse = [f'__device__ {inline} void {prefix}cg_reverse_{index}(const Element* values, Element* totals, int left, int right, int left_total, int right_total, int seed{worker}) {{']
        for body in (forward, reverse):
            for side, dimension in (('left', block['shape'][1]), ('right', block['shape'][2])):
                for component in range(dimension):
                    body.append(f'    const Element {side}{component}=values[{address(side, component)}];')
        for output in range(block['shape'][0]):
            forward.append(f'    Element output{output}=0;')
            for component, left, right, coefficient in edges:
                if component == output:
                    forward.append(f'    output{output}=fma(Element({coefficient:.17g})*left{left},right{right},output{output});')
            forward.append(f"    values[{address('destination', output)}]=output{output};")
            reverse.append(f"    const Element seed{output}=totals[{address('seed', output)}];")
        for side, dimension in (('left', block['shape'][1]), ('right', block['shape'][2])):
            for component in range(dimension):
                target = address(side + '_total', component)
                reverse += ['    {', f'        Element total=totals[{target}];']
                for output, left, right, coefficient in edges:
                    if (left if side == 'left' else right) == component:
                        value = f'right{right}' if side == 'left' else f'left{left}'
                        reverse.append(f'        total=fma(Element({coefficient:.17g})*seed{output},{value},total);')
                reverse += [f'        totals[{target}]=total;', '    }']
        code += ['\n'.join(forward + ['}']), '\n'.join(reverse + ['}'])]
    return '\n'.join(code)

CG_BINDING = Template(r"""
#include <ATen/core/Tensor.h>
${empty_header}#include <ATen/ops/zeros_like.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <optional>
#include <tuple>
namespace symmetric_backward_${schedule} {
std::tuple<std::optional<at::Tensor>,std::optional<at::Tensor>,std::optional<at::Tensor>> run(
    at::Tensor x,at::Tensor weights,at::Tensor attrs,at::Tensor g,bool need_x,bool need_w,bool need_a) {
    c10::cuda::CUDAGuard guard(x.device());
    std::optional<at::Tensor> dx,dw,da;
    if (need_x) dx=at::${x_allocation}(x);
    if (need_w) dw=at::zeros_like(weights);
    if (need_a) da=at::zeros_like(attrs);
    const int64_t atoms=x.size(0);
    if (atoms && (need_x || need_w || need_a)) {
        ${set_shared}
        ${grid_setup}
        symmetric_contraction_bwd_${schedule}_${precision}<<<${grid_variable},Threads,${shared_bytes},at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<Element>(),weights.data_ptr<Element>(),attrs.data_ptr<Element>(),g.data_ptr<Element>(),
            need_x?dx->data_ptr<Element>():nullptr,need_w?dw->data_ptr<Element>():nullptr,need_a?da->data_ptr<Element>():nullptr,
            atoms,need_x,need_w,need_a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {dx,dw,da};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) { module.def("run",&symmetric_backward_${schedule}::run); }
""")


def make_cg_backward_source(meta, dtype='float32'):
    """Keep singleton parameter maps on the existing node-owned schedule."""
    scalar = 'double' if str(dtype).removeprefix('torch.') in ('float64', 'double') else 'float'
    item_bytes = 8 if scalar == 'double' else 4
    packets = register_packets(meta, item_bytes // 4)
    shared_values = max((packet['value_slots'] for packet in packets if packet['shared']), default=0)
    shared_totals = max((packet['total_slots'] for packet in packets if packet['shared']), default=0)
    threads = BLOCK_SIZE
    while threads > 1 and (shared_values + shared_totals + meta['input_dim'] + meta['num_elements']) * threads * item_bytes > meta.get('shared_memory_bytes', 48 * 1024):
        threads //= 2
    rows = min(BACKWARD_CG_ROWS, threads)
    columns = threads // rows
    shared_bytes = (shared_values + shared_totals + meta['input_dim'] + meta['num_elements']) * threads * item_bytes
    graph = meta['cg_dag']
    input_dim = meta['input_dim']
    elements = meta['num_elements']
    fields = [f'Element x{index};' for index in range(input_dim)] + [f'Element a{index};' for index in range(elements)]
    code = [
        '#include <cuda_runtime.h>',
        '#include <cstdint>',
        'namespace symmetric_backward_register {',
        f'using Element={scalar};',
        f'constexpr int Threads={threads}, Rows={rows}, Columns={columns};',
        f"constexpr int Channels={meta['channels']}, NumElements={elements};",
        f"constexpr int64_t InputSize={meta['input_size']}, OutputSize={meta['output_dim']}, WeightSize={meta['weight_numel']};",
        'struct Result {' + ' '.join(fields) + '};'
    ]
    private_used = {op[1] for packet in packets if not packet['shared'] for op in packet['operations'] if op[0] == 'forward'}
    shared_used = {op[1] for packet in packets if packet['shared'] for op in packet['operations'] if op[0] == 'forward'}
    code.append(local_functions(meta, private_used))
    if shared_used:
        code.append(local_functions(meta, shared_used, True))
    for index, packet in enumerate(packets):
        shared = packet['shared']
        extra = ', Element* shared_values, Element* shared_totals, int worker' if shared else ''
        code += [
            f'__device__ __noinline__ void packet_{index}(const Element* x,const Element* weights,const Element* attrs,const Element* g,Element* dw,int64_t atom,int channel,bool valid,bool need_x,bool need_w,bool need_a,Element* __restrict__ carry{extra}) {{'
        ]
        if shared:
            code += ['    Element* values=shared_values;', '    Element* totals=shared_totals;']
        else:
            code += [f"    Element values[{packet['value_slots']}];", f"    Element totals[{packet['total_slots']}];"]

        def slot(value):
            return f'{value}*Threads+worker' if shared else str(value)
        code += [
            '    Result result={};',
            '    Element attributes[NumElements];',
            '    #pragma unroll',
            '    for (int element=0;element<NumElements;++element) attributes[element]=valid?attrs[atom*NumElements+element]:Element(0);'
        ]
        for component in range(input_dim):
            offset, stride, count = (meta[field][component] for field in ('input_offsets', 'input_strides', 'input_channels'))
            code.append(f'    values[{slot(component)}]=valid && channel<{count}?x[atom*InputSize+{offset}+int64_t(channel)*{stride}]:Element(0);')
            code.append(f'    totals[{slot(component)}]=0;')
        for operation in packet['operations']:
            kind = operation[0]
            if kind == 'zero':
                _, first, dimension = operation
                for component in range(dimension):
                    code.append(f'    totals[{slot(first + component)}]=0;')
            elif kind == 'forward':
                _, block, left, right, destination = operation
                extra = ',worker' if shared else ''
                code.append(f"    {('shared_' if shared else '')}cg_forward_{block}(values,{left},{right},{destination}{extra});")
            elif kind == 'reverse':
                _, block, left, right, lt, rt, seed = operation
                extra = ',worker' if shared else ''
                code.append(f"    {('shared_' if shared else '')}cg_reverse_{block}(values,totals,{left},{right},{lt},{rt},{seed}{extra});")
            else:
                _, root_id, value, total = operation
                root = graph['roots'][root_id]
                basis = root['selected_basis_id']
                output = root['output_component_offset']
                dimension = graph['nodes'][root['node_id']]['irrep_dim']
                mapping = graph['weight_members'][basis]
                code += ['    {']
                for component in range(dimension):
                    offset, stride, count = (meta[field][output + component] for field in ('output_offsets', 'output_strides', 'output_channels'))
                    code.append(f'        const Element g{component}=valid && channel<{count}?g[atom*OutputSize+{offset}+int64_t(channel)*{stride}]:Element(0);')
                code += ['        Element parameter=0;', '        {']
                for component in range(dimension):
                    code.append(f'            parameter=fma(g{component},values[{slot(value + component)}],parameter);')
                code += [
                    '        }',
                    '        Element mixed=0;',
                    f'        #pragma unroll {(1 if shared else 4)}',
                    '        for (int element=0;element<NumElements;++element) {'
                ]
                attribute = 'attributes[element]'
                if shared:
                    selection = ' : '.join([f'element=={i}?attributes[{i}]' for i in range(elements - 1)] + [f'attributes[{elements - 1}]'])
                    code.append(f'            const Element attribute={selection};')
                    attribute = 'attribute'
                code += ['            {', '                Element projected=0;']
                if len(mapping) == 1:
                    path, coefficient = mapping[0]
                    offset = meta['path_offsets'][path]
                    count = meta['path_channels'][path]
                    factor = '' if coefficient == 1 else f'Element({coefficient:.17g})*'
                    code.append(f'                if (channel<{count}) projected={factor}weights[int64_t(element)*WeightSize+{offset}+channel];')
                code += [f'                mixed=fma({attribute},projected,mixed);', '                {', '                    switch(element) {']
                for element in range(elements):
                    code.append(f'                    case {element}: result.a{element}=fma(projected,parameter,result.a{element});break;')
                code += [
                    '                    }',
                    '                }',
                    '            }',
                    '            {',
                    f'                Element sum={attribute}*parameter;',
                    '                constexpr unsigned Mask=0xffffffffu>>(32-(Threads<32?Threads:32));',
                    '                #pragma unroll',
                    '                for (int offset=Rows/2;offset>0;offset/=2) sum+=__shfl_down_sync(Mask,sum,offset,Rows);',
                    '                if (need_w && (threadIdx.x%Rows)==0 && sum!=Element(0)) {'
                ]
                if len(mapping) == 1:
                    path, coefficient = mapping[0]
                    offset = meta['path_offsets'][path]
                    count = meta['path_channels'][path]
                    factor = '' if coefficient == 1 else f'Element({coefficient:.17g})*'
                    code.append(f'                    if (channel<{count}) atomicAdd(dw+int64_t(element)*WeightSize+{offset}+channel,{factor}sum);')
                code += ['                }', '            }', '        }', '        {']
                for component in range(dimension):
                    code.append(f'            totals[{slot(total + component)}]=fma(mixed,g{component},totals[{slot(total + component)}]);')
                code += ['        }', '    }']
        for component in range(input_dim):
            code.append(f'    carry[{component}*Threads+threadIdx.x]+=totals[{slot(component)}];')
        for element in range(elements):
            code.append(f'    carry[{input_dim + element}*Threads+threadIdx.x]+=result.a{element};')
        code += ['}']
    precision = 'fp64' if scalar == 'double' else 'fp32'
    code += [
        f'__global__ void symmetric_contraction_bwd_register_{precision}(const Element* x,const Element* weights,const Element* attrs,const Element* g,Element* dx,Element* dw,Element* da,int64_t atoms,bool need_x,bool need_w,bool need_a) {{'
    ]
    code += ['    extern __shared__ Element scratch[];', f'    Element* carry=scratch+{shared_values + shared_totals}*Threads;']
    code += [
        '    const int channel=blockIdx.y*Columns+threadIdx.x/Rows;',
        '    for (int64_t first=int64_t(blockIdx.x)*Rows;first<atoms;first+=int64_t(gridDim.x)*Rows) {',
        '        const int64_t atom=first+threadIdx.x%Rows;',
        '        const bool valid=atom<atoms && channel<Channels;'
    ]
    for component in range(input_dim + elements):
        code.append(f'        carry[{component}*Threads+threadIdx.x]=0;')
    code += [f'        #pragma unroll 1', f'        for (int packet=0;packet<{len(packets)};++packet) {{', '            switch(packet) {']
    for index, packet in enumerate(packets):
        extra = f',scratch,scratch+{shared_values}*Threads,threadIdx.x' if packet['shared'] else ''
        code += [f'            case {index}: packet_{index}(x,weights,attrs,g,dw,atom,channel,valid,need_x,need_w,need_a,carry{extra});break;']
    code += ['            }', '        }']
    code.append('        if (valid) {')
    for component in range(input_dim):
        offset, stride, count = (meta[field][component] for field in ('input_offsets', 'input_strides', 'input_channels'))
        code.append(f'            if (need_x && channel<{count}) dx[atom*InputSize+{offset}+int64_t(channel)*{stride}]=carry[{component}*Threads+threadIdx.x];')
    for element in range(elements):
        code.append(f'            if (need_a && carry[{input_dim + element}*Threads+threadIdx.x]!=Element(0)) atomicAdd(da+atom*NumElements+{element},carry[{input_dim + element}*Threads+threadIdx.x]);')
    code += ['        }', '    }', '}', '}']
    device = '\n'.join(code) + '\n'
    attribute = (f"C10_CUDA_CHECK(cudaFuncSetAttribute(symmetric_contraction_bwd_register_{precision},"
                 f"cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_bytes}));"
                 if shared_bytes > SHARED_MEMORY_BYTES else "")
    return device + CG_BINDING.substitute(
        empty_header="#include <ATen/ops/empty_like.h>\n",
        schedule="register", x_allocation="empty_like", grid_variable="blocks",
        grid_setup="const dim3 blocks(std::min<int64_t>((atoms+Rows-1)/Rows,65535),(Channels+Columns-1)/Columns);",
        set_shared=attribute, precision=precision, shared_bytes=shared_bytes,
    )


def make_cg_packet_source(meta, dtype='float32'):
    """Reuse a packet projection across atoms and reduce before transposing T."""
    scalar = 'double' if str(dtype).removeprefix('torch.') in ('float64', 'double') else 'float'
    item_bytes = 8 if scalar == 'double' else 4
    graph = meta['cg_dag']
    packets = register_packets(meta, item_bytes // 4, parameter_totals=True)
    inputs, elements = meta['input_dim'], meta['num_elements']
    shared_values = max((packet['value_slots'] for packet in packets if packet['shared']), default=0)
    shared_totals = max((packet['total_slots'] for packet in packets if packet['shared']), default=0)
    root_capacity = max((len(packet['roots']) for packet in packets), default=0)
    threads = BLOCK_SIZE
    while True:
        columns = min(CHANNEL_TILE, threads)
        shared_bytes = ((shared_values + shared_totals + 1) * threads + root_capacity * elements * columns) * item_bytes
        if threads == 1 or shared_bytes <= meta.get('shared_memory_bytes', 48 * 1024):
            break
        threads //= 2
    members, ranges = [], []
    for mapping in graph['weight_members']:
        first = len(members)
        members.extend(mapping)
        ranges.append((first, len(members)))
    code = ['#include <cuda_runtime.h>', '#include <cstdint>', '#include <cub/warp/warp_exchange.cuh>',
            WARP_EXCHANGE_SOURCE,
            'namespace symmetric_backward_packets {',
            f'using Element={scalar};', f'constexpr int Threads={threads}, Columns={columns}, Rows=Threads/Columns;',
            f'constexpr int Channels={meta["channels"]}, NumElements={elements};',
            f'constexpr int64_t InputSize={meta["input_size"]}, OutputSize={meta["output_dim"]}, WeightSize={meta["weight_numel"]};',
            cg_member_constants(members, item_bytes),
            '__device__ const int64_t PathOffsets[]={' + ','.join(map(str, meta['path_offsets'])) + '};',
            '__device__ const int PathChannels[]={' + ','.join(map(str, meta['path_channels'])) + '};',
            'struct MemberRange { int first,last; };',
            '__device__ volatile const MemberRange MemberRanges[]={' + ','.join('{' + str(a) + ',' + str(b) + '}' for a,b in ranges) + '};']
    code += [r'''
__device__ __noinline__ void project_root(const Element* weights,Element* projected,int basis) {
    const int begin=MemberRanges[basis].first,end=MemberRanges[basis].last;
    for (int point=threadIdx.x;point<NumElements*Columns;point+=Threads) {
        const int element=point/Columns;
        const int channel=blockIdx.y*Columns+point%Columns;
        Element sum=0;
        #pragma unroll 1
        for (int index=begin;index<end;++index) {
            const CgMember member=load_cg_member(index);
            if (channel<PathChannels[member.path])
                sum=fma(member.coefficient,weights[int64_t(element)*WeightSize+PathOffsets[member.path]+channel],sum);
        }
        projected[point]=sum;
    }
}
__device__ __noinline__ void scatter_root(Element* dw,Element value,int element,int channel,int basis,Element* reduction) {
    reduction[threadIdx.x]=value;
    __syncthreads();
    const int column=threadIdx.x%Columns;
    Element sum=0;
    #pragma unroll
    for (int row=0;row<Rows;++row) sum+=reduction[row*Columns+column];
    const int begin=MemberRanges[basis].first,end=MemberRanges[basis].last;
    #pragma unroll 1
    for (int index=begin+threadIdx.x/Columns;index<end;index+=Rows) {
        const CgMember member=load_cg_member(index);
        if (channel<PathChannels[member.path] && sum!=Element(0))
            atomicAdd(dw+int64_t(element)*WeightSize+PathOffsets[member.path]+channel,member.coefficient*sum);
    }
    __syncthreads();
}
''']
    private_used = {op[1] for packet in packets if not packet['shared'] for op in packet['operations'] if op[0]=='forward'}
    shared_used = {op[1] for packet in packets if packet['shared'] for op in packet['operations'] if op[0]=='forward'}
    code.append(local_functions(meta, private_used))
    if shared_used:
        code.append(local_functions(meta, shared_used, True))
    for packet_id, packet in enumerate(packets):
        shared = packet['shared']
        roots = packet['roots']
        positions = {root: index for index, root in enumerate(roots)}
        code += [f'__device__ __noinline__ void packet_{packet_id}(const Element* x,const Element* weights,const Element* attrs,const Element* g,Element* dx,Element* dw,Element* da,int64_t atoms,int channel,bool need_x,bool need_w,bool need_a,Element* projected_weights,Element* reduction,Element* shared_values,Element* shared_totals) {{',
                 '    const int column=threadIdx.x%Columns;',
                 '    const int row=threadIdx.x/Columns;']
        for local, root_id in enumerate(roots):
            basis = graph['roots'][root_id]['selected_basis_id']
            code.append(f'    project_root(weights,projected_weights+{local}*NumElements*Columns,{basis});')
        code.append('    __syncthreads();')
        for local in range(len(roots)):
            for element in range(elements):
                code.append(f'    Element weight_total_{local}_{element}=0;')
        if shared:
            code += ['    Element* values=shared_values;', '    Element* totals=shared_totals;']
        else:
            code += [f'    Element values[{packet["value_slots"]}];', f'    Element totals[{packet["total_slots"]}];']
        code += ['    for (int64_t atom=int64_t(blockIdx.x)*Rows+row;atom<atoms;atom+=int64_t(gridDim.x)*Rows) {',
                 '        const bool valid=channel<Channels;', f'        Element attributes[{elements}];',
                 f'        Element attribute_totals[{elements}]={{}};', '        #pragma unroll',
                 '        for (int element=0;element<NumElements;++element) attributes[element]=attrs[atom*NumElements+element];']

        def slot(index):
            return f'{index}*Threads+threadIdx.x' if shared else str(index)

        for component in range(inputs):
            offset, stride, count = (meta[field][component] for field in ('input_offsets', 'input_strides', 'input_channels'))
            code += [f'        values[{slot(component)}]=valid && channel<{count}?x[atom*InputSize+{offset}+int64_t(channel)*{stride}]:Element(0);',
                     f'        totals[{slot(component)}]=0;']
        for operation in packet['operations']:
            kind = operation[0]
            if kind == 'zero':
                for component in range(operation[2]):
                    code.append(f'        totals[{slot(operation[1]+component)}]=0;')
            elif kind in ('forward', 'reverse'):
                prefix, extra = ('shared_', ',threadIdx.x') if shared else ('', '')
                block, *arguments = operation[1:]
                pointers = 'values' if kind == 'forward' else 'values,totals'
                code.append(f'        {prefix}cg_{kind}_{block}({pointers},' + ','.join(map(str, arguments)) + extra + ');')
            else:
                _, root_id, value, total = operation
                root = graph['roots'][root_id]
                local = positions[root_id]
                first = root['output_component_offset']
                dimension = graph['nodes'][root['node_id']]['irrep_dim']
                code += ['        {', '            Element parameter=0;']
                for component in range(dimension):
                    offset, stride, count = (meta[field][first+component] for field in ('output_offsets', 'output_strides', 'output_channels'))
                    code += [f'            const Element g{component}=valid && channel<{count}?g[atom*OutputSize+{offset}+int64_t(channel)*{stride}]:Element(0);',
                             f'            parameter=fma(g{component},values[{slot(value+component)}],parameter);']
                code.append('            Element mixed=0;')
                for element in range(elements):
                    code += [f'            const Element projected{element}=projected_weights[({local}*NumElements+{element})*Columns+column];',
                             f'            mixed=fma(attributes[{element}],projected{element},mixed);',
                             f'            if (need_a) attribute_totals[{element}]=fma(projected{element},parameter,attribute_totals[{element}]);',
                             f'            if (need_w) weight_total_{local}_{element}=fma(attributes[{element}],parameter,weight_total_{local}_{element});']
                for component in range(dimension):
                    code.append(f'            totals[{slot(total+component)}]=fma(mixed,g{component},totals[{slot(total+component)}]);')
                code.append('        }')
        code.append('        if (need_x) {')
        first = 0
        while first < inputs:
            stop = first + 1
            while (stop < inputs and meta['input_channels'][stop] == meta['input_channels'][first]
                   and meta['input_strides'][stop] == meta['input_strides'][first]
                   and meta['input_offsets'][stop] == meta['input_offsets'][first] + stop - first):
                stop += 1
            width = min(columns, 32 // item_bytes, 1 << ((stop-first).bit_length()-1))
            offset, stride, count = (meta[field][first] for field in ('input_offsets', 'input_strides', 'input_channels'))
            if width == 1:
                code += [f'            if (valid && channel<{count} && totals[{slot(first)}]!=Element(0))',
                         f'                atomicAdd(dx+atom*InputSize+{offset}+int64_t(channel)*{stride},totals[{slot(first)}]);']
            else:
                values = ','.join(f'totals[{slot(first+item)}]' for item in range(width))
                code += ['            {', f'                Element items[{width}]={{{values}}};',
                         f'                using Exchange=ShuffleWarpExchange<Element,{width},{width}>;',
                         '                typename Exchange::TempStorage storage;',
                         '                Exchange(storage).BlockedToStriped(items,items);',
                         f'                const int component=threadIdx.x%{width};',
                         '                const int first_channel=channel-component;']
                for item in range(width):
                    code += [f'                if (first_channel+{item}<{count} && items[{item}]!=Element(0))',
                             f'                    atomicAdd(dx+atom*InputSize+{offset}+component+int64_t(first_channel+{item})*{stride},items[{item}]);']
                code.append('            }')
            first += width
        code.append('        }')
        for element in range(elements):
            code += [f'        if (need_a && valid && attribute_totals[{element}]!=Element(0))',
                     f'            atomicAdd(da+atom*NumElements+{element},attribute_totals[{element}]);']
        code += ['    }', '    if (need_w) {']
        for local, root_id in enumerate(roots):
            basis = graph['roots'][root_id]['selected_basis_id']
            for element in range(elements):
                code.append(f'        scatter_root(dw,weight_total_{local}_{element},{element},channel,{basis},reduction);')
        code += ['    }', '}']
    precision = 'fp64' if scalar == 'double' else 'fp32'
    code += [f'__global__ void symmetric_contraction_bwd_packets_{precision}(const Element* x,const Element* weights,const Element* attrs,const Element* g,Element* dx,Element* dw,Element* da,int64_t atoms,bool need_x,bool need_w,bool need_a) {{',
             '    extern __shared__ Element scratch[];',
             f'    Element* projected=scratch+{shared_values+shared_totals}*Threads;',
             f'    Element* reduction=projected+{root_capacity}*NumElements*Columns;',
             '    const int channel=blockIdx.y*Columns+threadIdx.x%Columns;',
             f'    for (int packet=blockIdx.z;packet<{len(packets)};packet+=gridDim.z) {{',
             '        switch (packet) {']
    for packet_id in range(len(packets)):
        code.append(f'        case {packet_id}: packet_{packet_id}(x,weights,attrs,g,dx,dw,da,atoms,channel,need_x,need_w,need_a,projected,reduction,scratch,scratch+{shared_values}*Threads);break;')
    code += ['        }', '        __syncthreads();', '    }', '}', '}']
    packet_count = max(1, len(packets))
    grid = f"""const int columns=(Channels+Columns-1)/Columns;
        const int atom_groups=std::max(1,{BACKWARD_PARAMETER_CTA_WAVES}*{meta["multiprocessor_count"]}/({packet_count}*columns));
        const dim3 grid(std::min<int64_t>((atoms+Rows-1)/Rows,atom_groups),columns,std::min(65535,{packet_count}));"""
    attribute = (f"C10_CUDA_CHECK(cudaFuncSetAttribute(symmetric_contraction_bwd_packets_{precision},"
                 f"cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_bytes}));"
                 if shared_bytes > SHARED_MEMORY_BYTES else "")
    host = CG_BINDING.substitute(
        empty_header="", schedule="packets", x_allocation="zeros_like", grid_variable="grid",
        grid_setup=grid, set_shared=attribute, precision=precision, shared_bytes=shared_bytes,
    )
    return "\n".join(code) + "\n" + host
