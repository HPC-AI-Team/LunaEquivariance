"""Fused third derivatives of symmetric contractions."""

from collections import defaultdict
from textwrap import indent

from .utils import (
    ContractionLayout, consumer_outputs, contraction_groups,
    SHARED_MEMORY_BYTES, CG_SHARED_BUDGET_DIVISOR, TRIPLE_CG_SHARED_BUDGET_DIVISOR,
    bounded_cg_polynomials, reduced_pair_groups,
    build_arena_plan, packet_intervals,
)


CUDA_SOURCE = r"""
#include <ATen/core/Tensor.h>
/* row_workspace_header */#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/_foreach_zero.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace symmetric_triple_backward {
/* constants */
/* group_rows */

template<bool WriteGrad, int ChildDegree, int Component>
__device__ __forceinline__ void polynomial_triple_backward_edge(
    Element feature, Element rx, Element bx,
    Element child, Element child_rx, Element child_bx, Element child_rx_bx,
    Element grad_p, Element grad_pr, Element grad_pb, Element grad_prb,
    bool write_x, bool write_rx, Element* feature_sum, Element* feature_rx_sum,
    Element& polynomial, Element& polynomial_rx, Element& polynomial_bx, Element& polynomial_rx_bx
) {
    if constexpr (WriteGrad) {
        if (write_x) {
            if constexpr (ChildDegree == 0) feature_sum[Component] += grad_p * child;
            else if constexpr (ChildDegree == 1)
                feature_sum[Component] = fma(grad_p, child, fma(grad_pr, child_rx, fma(grad_pb, child_bx, feature_sum[Component])));
            else feature_sum[Component] = fma(grad_p, child,
                fma(grad_pr, child_rx, fma(grad_pb, child_bx, fma(grad_prb, child_rx_bx, feature_sum[Component]))));
        }
        if (write_rx) {
            if constexpr (ChildDegree == 0) feature_rx_sum[Component] += grad_pr * child;
            else feature_rx_sum[Component] = fma(grad_pr, child, fma(grad_prb, child_bx, feature_rx_sum[Component]));
        }
    }
    polynomial = fma(feature, child, polynomial);
    if constexpr (ChildDegree == 0) {
        polynomial_rx = fma(rx, child, polynomial_rx);
        polynomial_bx = fma(bx, child, polynomial_bx);
    } else {
        polynomial_rx = fma(rx, child, fma(feature, child_rx, polynomial_rx));
        polynomial_bx = fma(bx, child, fma(feature, child_bx, polynomial_bx));
        if constexpr (ChildDegree == 1)
            polynomial_rx_bx = fma(rx, child_bx, fma(bx, child_rx, polynomial_rx_bx));
        else polynomial_rx_bx = fma(rx, child_bx, fma(bx, child_rx,
            fma(feature, child_rx_bx, polynomial_rx_bx)));
    }
}


/* power_helpers */template<bool WriteAttrs, bool HasAttrsDirections, bool AllGrads, bool WeightDirections/* band_template */>
__global__ __launch_bounds__(NODE_THREADS) void symmetric_contraction_triplebwd_pairs_/* precision */_i64(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ rw, const Element* __restrict__ ra,
    const Element* __restrict__ bx, const Element* __restrict__ bw, const Element* __restrict__ ba,
    const Element* __restrict__ bg, Element* __restrict__ tx, Element* __restrict__ ta,
    Element* __restrict__ tg, Element* __restrict__ trx, Element* __restrict__ tra,
    Element* __restrict__ tw, Element* __restrict__ trw, int64_t nodes/* row_arguments */
) {
/* shared_declaration */
    __shared__ Element attribute_values[HasAttrsDirections ? 3 : 1][ATOM_TILE][NUM_ELEMENTS];
    __shared__ unsigned active_rows[NUM_ELEMENTS];
    __shared__ unsigned element_words[ELEMENT_WORDS];
    __shared__ int active_elements[NUM_ELEMENTS];
    __shared__ int active_elements_count;
    __shared__ Element projected_weights[4][ELEMENT_TILE][3][FUSED_CHANNEL_TILE + 1];
    __shared__ Element products[4][HasAttrsDirections ? 4 : 2][ATOM_TILE][FUSED_CHANNEL_TILE];
    const int local_node = threadIdx.x / FUSED_CHANNEL_TILE;
    const int local_channel = threadIdx.x % FUSED_CHANNEL_TILE;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
/* task_mapping */
        const Index node = /* row_index */;
        Element attr_sum[WriteAttrs ? NUM_ELEMENTS : 1] = {};
        Element attr_rx_sum[WriteAttrs ? NUM_ELEMENTS : 1] = {};
        {
            const int channel_begin = channel_tile * FUSED_CHANNEL_TILE;
            const int channel = channel_begin + local_channel;
/* initialize_weights */
/* grouped_initialization */
            for (int word = threadIdx.x; word < ELEMENT_WORDS; word += NODE_THREADS) element_words[word] = 0;
            for (int element = threadIdx.x; element < NUM_ELEMENTS; element += NODE_THREADS) active_rows[element] = 0;
/* grouped_initialization_end */
/* feature_loads */
            __syncthreads();

/* grouped_attributes */
            for (int word = 0; word < ELEMENT_WORDS; ++word) {
                unsigned elements = 0;
                #pragma unroll
                for (int first = 0; first < 32; first += FUSED_CHANNEL_TILE) {
                    const int element = word * 32 + first + local_channel;
                    bool active = false;
                    if (element < NUM_ELEMENTS) {
                        const Index index = node * NUM_ELEMENTS + element;
                        const Element a = node < nodes ? attrs[index] : Element(0);
                        attribute_values[0][local_node][element] = a;
                        active = a != Element(0);
                        if constexpr (HasAttrsDirections) {
                            const Element ar = node < nodes && ra ? ra[index] : Element(0);
                            const Element ab = node < nodes && ba ? ba[index] : Element(0);
                            attribute_values[1][local_node][element] = ar;
                            attribute_values[2][local_node][element] = ab;
                            active |= ar != Element(0) || ab != Element(0);
                        }
                        if (active) atomicOr(active_rows + element, 1u << local_node);
                    }
                    const unsigned mask = __ballot_sync(0xffffffff, active);
                    const unsigned row = (mask >> (lane - local_channel)) & ((1u << FUSED_CHANNEL_TILE) - 1);
                    elements |= row << first;
                }
                if (local_channel == 0) {
                    atomicOr(element_words + word, elements);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int count = 0;
                for (int word = 0; word < ELEMENT_WORDS; ++word) {
                    unsigned remaining = element_words[word];
                    while (remaining) {
                        active_elements[count++] = word * 32 + __ffs(remaining) - 1;
                        remaining &= remaining - 1;
                    }
                }
                active_elements_count = count;
            }
/* grouped_attributes_end */
            __syncthreads();
            volatile Element* feature = data[0];
            volatile Element* direction_rx = data[1];
            volatile Element* direction_bx = data[2];
            const bool valid = node < nodes && /* channel_guard */;
            Element feature_sum[INPUT_DIM] = {};
            Element feature_rx_sum[INPUT_DIM] = {};
            Element output_sum[OUTPUT_COMPONENTS] = {};
/* saved_gradients */
/* channel_contractions */
/* gradient_staging */
            __syncthreads();
/* feature_stores */
            if (node < nodes && /* channel_guard */) {
                if (AllGrads || tg) {
/* output_stores */
                }
            }
        }
        if constexpr (WriteAttrs) {
            for (int element = 0; element < NUM_ELEMENTS; ++element) {
                Element a = attr_sum[element], ar = attr_rx_sum[element];
                #pragma unroll
                for (int offset = FUSED_CHANNEL_TILE / 2; offset > 0; offset >>= 1) {
                    a += __shfl_down_sync(0xffffffff, a, offset, FUSED_CHANNEL_TILE);
                    ar += __shfl_down_sync(0xffffffff, ar, offset, FUSED_CHANNEL_TILE);
                }
                if (local_channel == 0 && node < nodes) {
                    if (ta) atomicAdd(ta + node * NUM_ELEMENTS + element, a);
                    if (tra) atomicAdd(tra + node * NUM_ELEMENTS + element, ar);
                }
            }
        }
        __syncthreads();
    }
}

pybind11::tuple run(
    at::Tensor x, at::Tensor weights, at::Tensor attrs, at::Tensor grad_out,
    c10::optional<at::Tensor> rx, c10::optional<at::Tensor> rw, c10::optional<at::Tensor> ra,
    c10::optional<at::Tensor> bx, c10::optional<at::Tensor> bw, c10::optional<at::Tensor> ba,
    c10::optional<at::Tensor> bg, bool need_x, bool need_weights, bool need_attrs,
    bool need_grad_out, bool need_rx, bool need_rw, bool need_ra
) {
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t nodes = x.size(0);
    at::Tensor tx, tw, ta, tg, trx, trw, tra;
    if (need_x) tx = at::empty_like(x);
    if (need_weights) tw = at::empty_like(weights);
    if (need_attrs) ta = at::empty_like(attrs);
    if (need_grad_out) tg = at::empty_like(grad_out);
    if (need_rx) trx = at::empty_like(x);
    if (need_rw) trw = at::empty_like(weights);
    if (need_ra) tra = at::empty_like(attrs);
/* batch_weight_initialization */
    const auto ptr = [](const c10::optional<at::Tensor>& value) {
        return value.has_value() ? value->data_ptr<Element>() : nullptr;
    };
    const Element* rx_ptr = ptr(rx), *rw_ptr = ptr(rw), *ra_ptr = ptr(ra);
    const Element* bx_ptr = ptr(bx), *bw_ptr = ptr(bw), *ba_ptr = ptr(ba), *bg_ptr = ptr(bg);

    auto stream = at::cuda::getCurrentCUDAStream();
    if (nodes > 0 && (need_x || need_weights || need_attrs || need_grad_out || need_rx || need_rw || need_ra)) {
/* row_workspace */
/* launch */
        if (need_x && need_weights && need_grad_out && need_rx && need_rw
            && !need_attrs && !need_ra && !ra_ptr && !ba_ptr
            && rx_ptr && rw_ptr && bx_ptr && bw_ptr && bg_ptr) {
            LAUNCH_FUSED(false, false, true, true);
        } else if (need_x && need_weights && need_grad_out && need_rx && !need_rw
            && !need_attrs && !need_ra && !ra_ptr && !ba_ptr && !rw_ptr && !bw_ptr
            && rx_ptr && bx_ptr && bg_ptr) {
            LAUNCH_FUSED(false, false, true, false);
        } else {
        if (need_attrs || need_ra) {
            if (ra_ptr || ba_ptr) LAUNCH_FUSED(true, true, false, true);
            else LAUNCH_FUSED(true, false, false, true);
        } else {
            if (ra_ptr || ba_ptr) LAUNCH_FUSED(false, true, false, true);
            else LAUNCH_FUSED(false, false, false, true);
        }
        }
        #undef LAUNCH_FUSED
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(
        need_x ? pybind11::cast(tx) : pybind11::none(), need_weights ? pybind11::cast(tw) : pybind11::none(),
        need_attrs ? pybind11::cast(ta) : pybind11::none(), need_grad_out ? pybind11::cast(tg) : pybind11::none(),
        need_rx ? pybind11::cast(trx) : pybind11::none(), need_rw ? pybind11::cast(trw) : pybind11::none(),
        need_ra ? pybind11::cast(tra) : pybind11::none());
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
/* configure */ module.def("run", &symmetric_triple_backward::run); }
"""


GROUP_ROWS = r"""
constexpr int GroupThreads = 128;

__global__ void classify_rows(const Element* attrs, int2* rows, int* counts, int nodes) {
    const int node = blockIdx.x * (GroupThreads / 32) + threadIdx.x / 32;
    if (node >= nodes) return;
    const int lane = threadIdx.x % 32;
    int element = 0, nonzeros = 0;
    for (int first = 0; first < NUM_ELEMENTS; first += 32) {
        const int column = first + lane;
        const unsigned active = __ballot_sync(0xffffffffu,
            column < NUM_ELEMENTS && attrs[Index(node) * NUM_ELEMENTS + column] != Element(0));
        nonzeros += __popc(active);
        if (nonzeros > 1) break;
        if (active) element = first + __ffs(active) - 1;
    }
    if (lane == 0) {
        if (nonzeros > 1) {
            atomicExch(counts + NUM_ELEMENTS, 1);
        } else {
            rows[node] = make_int2(element, atomicAdd(counts + element, 1));
        }
    }
}

__global__ void pack_rows(const int2* rows, int* counts, int* permutation, int2* tiles, int nodes) {
    if (counts[NUM_ELEMENTS]) return;
    __shared__ int starts[NUM_ELEMENTS];
    __shared__ int warp_tiles[GroupThreads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int tile_base = 0;
    for (int first = 0; first < NUM_ELEMENTS; first += GroupThreads) {
        const int element = first + threadIdx.x;
        const int count = element < NUM_ELEMENTS ? counts[element] : 0;
        const int ntiles = (count + ATOM_TILE - 1) / ATOM_TILE;
        int scan = ntiles;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const int prior = __shfl_up_sync(0xffffffffu, scan, offset);
            if (lane >= offset) scan += prior;
        }
        if (lane == 31) warp_tiles[warp] = scan;
        __syncthreads();
        int start = tile_base + scan - ntiles;
        #pragma unroll
        for (int w = 0; w < GroupThreads / 32; ++w) {
            const int total = warp_tiles[w];
            if (w < warp) start += total;
            tile_base += total;
        }
        if (element < NUM_ELEMENTS) {
            starts[element] = start;
            if (blockIdx.x == 0) {
                for (int tile = 0; tile < ntiles; ++tile) {
                    tiles[start + tile] = make_int2(element, min(ATOM_TILE, count - tile * ATOM_TILE));
                }
            }
        }
        __syncthreads();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) counts[NUM_ELEMENTS + 1] = tile_base;
    const int node = blockIdx.x * GroupThreads + threadIdx.x;
    if (node < nodes) {
        const int2 row = rows[node];
        permutation[starts[row.x] * ATOM_TILE + row.y] = node;
    }
}
"""


ROW_WORKSPACE = r"""
        at::Tensor row_workspace;
        const int* row_counts = nullptr;
        const int* row_permutation = nullptr;
        const int2* row_tiles = nullptr;
        Index max_row_tiles = (nodes + ATOM_TILE - 1) / ATOM_TILE;
        // Group only fixed attributes, and amortize preprocessing over at least a tile per element.
        if (!need_attrs && !need_ra && !ra_ptr && !ba_ptr
            && nodes >= Index(NUM_ELEMENTS) * ATOM_TILE && nodes < (Index(1) << 30)) {
            max_row_tiles += NUM_ELEMENTS;
            row_workspace = at::empty({2 * max_row_tiles + 2 * nodes + NUM_ELEMENTS + 2
                                       + max_row_tiles * ATOM_TILE}, attrs.options().dtype(at::kInt));
            auto* tiles = reinterpret_cast<int2*>(row_workspace.data_ptr<int>());
            auto* classes = tiles + max_row_tiles;
            int* counts = reinterpret_cast<int*>(classes + nodes);
            int* permutation = counts + NUM_ELEMENTS + 2;
            C10_CUDA_CHECK(cudaMemsetAsync(counts, 0, (NUM_ELEMENTS + 2) * sizeof(int), stream));
            classify_rows<<<(nodes + GroupThreads / 32 - 1) / (GroupThreads / 32), GroupThreads, 0, stream>>>(
                attrs.data_ptr<Element>(), classes, counts, nodes);
            pack_rows<<<(nodes + GroupThreads - 1) / GroupThreads, GroupThreads, 0, stream>>>(
                classes, counts, permutation, tiles, nodes);
            row_counts = counts;
            row_permutation = permutation;
            row_tiles = tiles;
        }
"""


WINDOW_REDUCTION = r"""
                __syncthreads();
                for (int group_slot = warp; group_slot < 4; group_slot += NODE_THREADS / 32) {
                const int group = /* group_offset */window * 4 + group_slot;
                const int c = channel_begin + lane % FUSED_CHANNEL_TILE;
/* reduction_mask */
                if (group < NUM_GROUPS && /* reduction_guard */ && (AllGrads || tw || trw)) {
/* grouped_reduction */
                    for (int active = lane / FUSED_CHANNEL_TILE; active < active_elements_count;
                         active += 32 / FUSED_CHANNEL_TILE) {
                        const int element = active_elements[active];
                        Element total = 0, total_rx = 0;
                        const auto accumulate = [&](int n) {
                            const Element a = attribute_values[0][n][element];
                            const int column = lane % FUSED_CHANNEL_TILE;
                            if (AllGrads || tw) total = fma(a, products[group_slot][1][n][column], total);
                            if ((AllGrads && WeightDirections) || trw) total_rx = fma(a, products[group_slot][0][n][column], total_rx);
                            if constexpr (HasAttrsDirections) {
                                const Element ar = ra ? attribute_values[1][n][element] : Element(0);
                                const Element ab = ba ? attribute_values[2][n][element] : Element(0);
                                if (AllGrads || tw) total = fma(ar, products[group_slot][0][n][column],
                                    fma(ab, products[group_slot][3][n][column], total));
                                if ((AllGrads && WeightDirections) || trw) total_rx = fma(ab, products[group_slot][2][n][column], total_rx);
                            }
                        };
                        unsigned remaining = active_rows[element];
                        if (remaining == (0xffffffffu >> (32 - ATOM_TILE))) {
                            for (int n = 0; n < ATOM_TILE; ++n) accumulate(n);
                        } else {
                            while (remaining) {
                                const int n = __ffs(remaining) - 1;
                                remaining &= remaining - 1;
                                accumulate(n);
                            }
                        }
/* store_weight_gradient */
                    }
/* grouped_reduction_end */
                }
                }
                __syncthreads();
            }"""


GROUPED_REDUCTION = r"""                    if (grouped) {
                        const int element = tile.x;
                        const int column = lane % FUSED_CHANNEL_TILE;
                        constexpr int RowsPerWarp = 32 / FUSED_CHANNEL_TILE;
                        Element total = 0, total_rx = 0;
                        const unsigned rows = active_rows[element];
                        for (int n = lane / FUSED_CHANNEL_TILE; n < ATOM_TILE; n += RowsPerWarp) {
                            if (rows & (1u << n)) {
                                const Element a = attribute_values[0][n][element];
                                if (AllGrads || tw) total = fma(a, products[group_slot][1][n][column], total);
                                if ((AllGrads && WeightDirections) || trw) {
                                    total_rx = fma(a, products[group_slot][0][n][column], total_rx);
                                }
                            }
                        }
                        #pragma unroll
                        for (int offset = 16; offset >= FUSED_CHANNEL_TILE; offset /= 2) {
                            total += __shfl_down_sync(reduction_mask, total, offset);
                            total_rx += __shfl_down_sync(reduction_mask, total_rx, offset);
                        }
                        if (lane < FUSED_CHANNEL_TILE) {
/* store_weight_gradient */
                        }
                    } else {"""


WINDOW_STORE = r"""                        for (int member = GROUP_OFFSETS[group]; member < GROUP_OFFSETS[group + 1]; ++member) {
                            const Index index = /* reduction_index */;
                            const Element coefficient = GROUP_SIGNS[member];
                            if ((AllGrads || tw) && total != 0) atomicAdd(tw + index, coefficient * total);
                            if (((AllGrads && WeightDirections) || trw) && total_rx != 0) atomicAdd(trw + index, coefficient * total_rx);
                        }"""


POWER_CUDA_SOURCE = r"""
template<bool WriteGrad, int Degree, int Component>
__device__ __forceinline__ void polynomial_triple_backward_power(
    Element coefficient, Element x, Element rx, Element bx,
    Element grad_p, Element grad_pr, Element grad_pb, Element grad_prb,
    bool write_x, bool write_rx, Element* feature_sum, Element* feature_rx_sum,
    Element& polynomial, Element& polynomial_rx, Element& polynomial_bx, Element& polynomial_rx_bx
) {
    Element powers[Degree + 1];
    powers[0] = coefficient;
    #pragma unroll
    for (int degree = 1; degree <= Degree; ++degree) powers[degree] = powers[degree - 1] * x;
    const Element first = Element(Degree) * powers[Degree - 1];
    const Element second = Element(Degree * (Degree - 1)) * powers[Degree - 2];
    polynomial += powers[Degree];
    polynomial_rx = fma(first, rx, polynomial_rx);
    polynomial_bx = fma(first, bx, polynomial_bx);
    polynomial_rx_bx = fma(second * rx, bx, polynomial_rx_bx);
    if constexpr (WriteGrad) {
        if (write_x) {
            Element gradient = fma(grad_p, first, fma(grad_pr, second * rx, grad_pb * second * bx));
            if constexpr (Degree > 2) {
                const Element third = Element(Degree * (Degree - 1) * (Degree - 2)) * powers[Degree - 3];
                gradient = fma(grad_prb, third * rx * bx, gradient);
            }
            feature_sum[Component] += gradient;
        }
        if (write_rx) feature_rx_sum[Component] += fma(grad_pr, first, grad_prb * second * bx);
    }
}
"""


def polynomial_triple_backward_code(terms, values, gradients, write_grad, write_x, write_rx,
                                    total_x, total_rx, depth=0, combine_squares=False, combine_powers=False):
    p, pr, pb, prb = values
    gp, gpr, gpb, gprb = gradients
    if not terms[0][0]:
        coefficient = sum(coefficient for inputs, coefficient in terms)
        return [f"{p} = Element({coefficient:.17g});", f"{pr} = 0;", f"{pb} = 0;", f"{prb} = 0;"]
    children = defaultdict(list)
    for inputs, coefficient in terms:
        children[inputs[0]].append((inputs[1:], coefficient))
    child_values = tuple(f"child_{name}{depth}" for name in ("p", "pr", "pb", "prb"))
    child_gradients = tuple(f"child_g{name}{depth}" for name in ("p", "pr", "pb", "prb"))
    cp, cr, cb, crb = child_values
    cgp, cgr, cgb, cgrb = child_gradients
    code = []
    for component, remaining in sorted(children.items()):
        x, r, b = f"input_x{depth}", f"input_rx{depth}", f"input_bx{depth}"
        degree = len(remaining[0][0]) + 1
        if (combine_powers and degree >= 2 and (degree > 2 or depth > 0)
                and all(inputs == (component,) * (degree - 1) for inputs, _ in remaining)):
            coefficient = sum(coefficient for _, coefficient in remaining)
            inputs = [f"{name}[{component} * (NODE_THREADS + 1)]"
                      for name in ("feature", "direction_rx", "direction_bx")]
            arguments = [f"Element({coefficient:.17g})", *inputs, *gradients,
                         write_x, write_rx, total_x, total_rx, *values]
            code.append(f"polynomial_triple_backward_power<{write_grad}, {degree}, {component}>("
                        + ", ".join(arguments) + ");")
            continue
        if combine_squares and depth == 0 and all(inputs == (component,) for inputs, _ in remaining):
            coefficient = sum(coefficient for inputs, coefficient in remaining)
            code += ["{", f"    const Element {x} = feature[{component} * (NODE_THREADS + 1)];",
                     f"    const Element {r} = direction_rx[{component} * (NODE_THREADS + 1)];",
                     f"    const Element {b} = direction_bx[{component} * (NODE_THREADS + 1)];",
                     f"    const Element weighted_x = Element({coefficient:.17g}) * {x};",
                     f"    {p} = fma(weighted_x, {x}, {p});",
                     f"    {pr} = fma(Element(2) * weighted_x, {r}, {pr});",
                     f"    {pb} = fma(Element(2) * weighted_x, {b}, {pb});",
                     f"    {prb} = fma(Element({2 * coefficient:.17g}) * {r}, {b}, {prb});"]
            if write_grad != "false":
                code += [f"    if constexpr ({write_grad}) {{", f"        if ({write_x}) {{",
                         f"            const Element gradient = fma({gp}, {x}, fma({gpr}, {r}, ({gpb}) * {b}));",
                         f"            {total_x}[{component}] = fma(Element({2 * coefficient:.17g}), gradient, {total_x}[{component}]);", "        }",
                         f"        if ({write_rx}) {{",
                         f"            const Element gradient = fma({gpr}, {x}, ({gprb}) * {b});",
                         f"            {total_rx}[{component}] = fma(Element({2 * coefficient:.17g}), gradient, {total_rx}[{component}]);", "        }", "    }"]
            code.append("}")
            continue
        code += ["{", f"    const Element {x} = feature[{component} * (NODE_THREADS + 1)];",
                 f"    const Element {r} = direction_rx[{component} * (NODE_THREADS + 1)];",
                 f"    const Element {b} = direction_bx[{component} * (NODE_THREADS + 1)];"]
        degree = len(remaining[0][0])
        if degree == 0:
            coefficient = sum(coefficient for inputs, coefficient in remaining)
            child = (f"Element({coefficient:.17g})", "Element(0)", "Element(0)", "Element(0)")
            args = [x, r, b, *child, *gradients, write_x, write_rx, total_x, total_rx, *values]
            code.append(f"polynomial_triple_backward_edge<{write_grad}, 0, {component}>(" + ", ".join(args) + ");")
            code.append("}")
            continue
        code += ["{", f"    Element {cp} = 0, {cr} = 0, {cb} = 0, {crb} = 0;"]
        if write_grad == "false":
            selected_gradients = ("Element(0)",) * 4
        else:
            selected_gradients = child_gradients
            code += [f"    Element {cgp} = 0, {cgr} = 0, {cgb} = 0, {cgrb} = 0;",
                     f"    if constexpr ({write_grad}) {{",
                     f"        {cgp} = fma({gp}, {x}, fma({gpr}, {r}, ({gpb}) * {b}));",
                     f"        {cgr} = fma({gpr}, {x}, ({gprb}) * {b});",
                     f"        {cgb} = fma({gpb}, {x}, ({gprb}) * {r});",
                     f"        {cgrb} = ({gprb}) * {x};", "    }"]
        code.extend("    " + line for line in polynomial_triple_backward_code(
            remaining, child_values, selected_gradients, write_grad, write_x, write_rx,
            total_x, total_rx, depth + 1, combine_powers=combine_powers))
        args = [x, r, b, *child_values, *gradients, write_x, write_rx, total_x, total_rx, *values]
        code += [f"    polynomial_triple_backward_edge<{write_grad}, {degree}, {component}>(" + ", ".join(args) + ");", "}", "}"]
    return code


def path_channel_count(meta, path):
    counts = meta.get("input_channels", (meta["channels"],) * meta["input_dim"])
    limit = meta.get("output_channels", (meta["channels"],) * len(meta["output_offsets"]))[path["output_component"]]
    pending = [path["tree"]]
    while pending:
        node = pending.pop()
        if "input" in node:
            limit = min(limit, *counts[node["start"]:node["stop"]])
        else:
            pending.extend(node["args"])
    return limit


def packed_windows(binary, tile=4):
    selected = {path["basis_path"]: path for path in binary["selected_paths"]}
    entries, paired = [], set()
    for pair in binary["pair_groups"]:
        consumers = list(pair["consumers"])
        paired.update(item[0] for item in consumers)
        entries.extend((pair["terms"], consumers[first:first + tile])
                       for first in range(0, len(consumers), tile))
    linear = [basis for basis in selected if basis not in paired]
    entries.extend((None, [(basis, selected[basis]["path"], None, 1)
                           for basis in linear[first:first + tile]])
                   for first in range(0, len(linear), tile))
    return pack_windows(entries, tile)


def pack_windows(entries, tile=4):
    windows = []
    for entry in sorted(entries, key=lambda item: -len(item[1])):
        for window in windows:
            if sum(len(consumers) for _, consumers in window) + len(entry[1]) <= tile:
                window.append(entry)
                break
        else:
            windows.append([entry])
    return windows or [[]]


def outer_projection_items(outer, component):
    selected = []
    for raw, items in sorted(outer.items(), key=lambda item: -1 if item[0] is None else item[0]):
        values = tuple((output, value) for output, inner, value in items if inner == component)
        if values:
            selected.append((raw, values))
    return tuple(selected)


def pack_projection_windows(entries, meta, limits, layout=None):
    layout = layout or ContractionLayout(meta, limits, False, meta["channels"])
    windows = pack_windows(entries)
    entries = [entry for window in windows for entry in window]
    paths = []
    for terms, consumers in entries:
        projections = set()
        for component in {m for m, _, _ in terms} if terms else ():
            for basis, outer in consumers:
                items = outer_projection_items(outer, component)
                if items:
                    guard = (layout.guard("channel", limits[basis])
                             if limits[basis] < meta["channels"] and limits[basis] < layout.end else None)
                    projections.add((component, items, guard))
        paths.append(projections)
    sizes = [len(consumers) for _, consumers in entries]
    packed, first = [], 0
    for window in windows:
        packed.append(list(range(first, first + len(window))))
        first += len(window)

    def loads(group):
        return len(set().union(*(paths[item] for item in group)))

    while True:
        improvement, move = 0, None
        for left in range(len(packed)):
            for right in range(left + 1, len(packed)):
                before = loads(packed[left]) + loads(packed[right])
                for a in [None, *packed[left]]:
                    for b in [None, *packed[right]]:
                        if a is None and b is None:
                            continue
                        lhs = [item for item in packed[left] if item != a] + ([] if b is None else [b])
                        rhs = [item for item in packed[right] if item != b] + ([] if a is None else [a])
                        if sum(sizes[item] for item in lhs) > 4 or sum(sizes[item] for item in rhs) > 4:
                            continue
                        delta = loads(lhs) + loads(rhs) - before
                        if delta < improvement:
                            improvement, move = delta, (left, right, lhs, rhs)
        if move is None:
            break
        left, right, packed[left], packed[right] = move
    return [[entries[item] for item in group] for group in packed if group] or [[]]


def input_groups(meta):
    counts, offsets, strides = (meta[key] for key in ("input_channels", "input_offsets", "input_strides"))
    groups, component = [], 0
    while component < meta["input_dim"]:
        count, offset, stride = counts[component], offsets[component], strides[component]
        dimension = max(1, stride)
        if component + dimension > meta["input_dim"] or any(
                counts[component + m] != count or strides[component + m] != stride
                or offsets[component + m] != offset + m for m in range(dimension)):
            dimension = 1
        groups.append((component, dimension, count, offset, stride))
        component += dimension
    return groups


def shared_layout(groups):
    stride, locations = 0, {}
    for group, (first, dimension, count, offset, item_stride) in enumerate(groups):
        padded = dimension + (dimension % 2 == 0)
        for component in range(dimension):
            locations[first + component] = f"{stride} * NODE_THREADS + {group} + threadIdx.x * {padded} + {component}"
        stride += padded
    return stride, locations


def band_expression(values):
    expression = str(values[-1])
    for index in range(len(values) - 2, -1, -1):
        expression = f"(ChannelBand == {index} ? {values[index]} : {expression})"
    return expression


def feature_code(groups, banded):
    loads, stores, staging = [], [], []
    shared_stride = 0
    for group, (first, dimension, count, offset, stride) in enumerate(groups):
        padded = dimension + (dimension % 2 == 0)
        shared_base = f"{shared_stride} * NODE_THREADS + {group}"
        padding = "" if padded == dimension else f" + item / {dimension}"
        position = f"{shared_base} + local_node * FUSED_CHANNEL_TILE * {padded} + item{padding}"
        tail = "item" if dimension == stride else f"Index(item) * {stride}"
        index = f"node * INPUT_SIZE + Index({offset}) + Index(channel_begin) * {stride} + {tail}"
        begin = f"            if constexpr (BandBegin < {count}) {{" if banded else "            {"
        common = [begin,
                  f"                const int items = {dimension} * min(FUSED_CHANNEL_TILE, {count} - channel_begin);",
                  f"                for (int item = local_channel; item < {dimension} * FUSED_CHANNEL_TILE; item += FUSED_CHANNEL_TILE) {{",
                  f"                    const Index offset = {index};",
                  "                    const bool present = node < nodes && item < items;"]
        loads += common + [f"                    data[0][{position}] = present ? x[offset] : Element(0);",
                           f"                    data[1][{position}] = present && (AllGrads || rx) ? rx[offset] : Element(0);",
                           f"                    data[2][{position}] = present && (AllGrads || bx) ? bx[offset] : Element(0);",
                           "                }", "            }"]
        stores += common + ["                    if (present) {",
                            f"                        if (AllGrads || tx) tx[offset] = data[0][{position}];",
                            f"                        if (AllGrads || trx) trx[offset] = data[1][{position}];",
                            "                    }", "                }", "            }"]
        for component in range(dimension):
            location = f"{shared_base} + threadIdx.x * {padded} + {component}"
            for plane, pointer, total in ((0, "tx", "feature_sum"), (1, "trx", "feature_rx_sum")):
                statement = f"if (AllGrads || {pointer}) data[{plane}][{location}] = {total}[{first + component}];"
                if banded:
                    statement = f"if constexpr (BandBegin < {count}) {{ {statement} }}"
                staging.append("            " + statement)
        shared_stride += padded
    return "\n".join(loads), "\n".join(stores), "\n".join(staging)


def launch_code(bands, shared_dims, banded, grouped_rows):
    node_tiles = "max_row_tiles" if grouped_rows else "(nodes + ATOM_TILE - 1) / ATOM_TILE"
    row_tables = ", row_counts, row_permutation, row_tiles" if grouped_rows else ""
    function = "symmetric_contraction_triplebwd_pairs_/* precision */_i64"
    if not banded:
        launch = r'''        const int64_t tasks = /* node tiles */ * CHANNEL_GROUP_TILE;
        const int64_t groups = ((CHANNELS + FUSED_CHANNEL_TILE - 1) / FUSED_CHANNEL_TILE + CHANNEL_GROUP_TILE - 1) / CHANNEL_GROUP_TILE;
        const dim3 blocks(std::min<int64_t>(tasks, 65535), std::min<int64_t>(groups, 65535));
        #define LAUNCH_FUSED(WRITE_A, ATTRS_DIRECTIONS, ALL_GRADS, WEIGHT_DIRECTIONS) \
            symmetric_contraction_triplebwd_pairs_/* precision */_i64<WRITE_A, ATTRS_DIRECTIONS, ALL_GRADS, WEIGHT_DIRECTIONS><<<blocks, NODE_THREADS, DYNAMIC_SHARED_BYTES, stream>>>( \
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(), grad_out.data_ptr<Element>(), \
                rx_ptr, rw_ptr, ra_ptr, bx_ptr, bw_ptr, ba_ptr, bg_ptr, \
                need_x ? tx.data_ptr<Element>() : nullptr, need_attrs ? ta.data_ptr<Element>() : nullptr, \
                need_grad_out ? tg.data_ptr<Element>() : nullptr, need_rx ? trx.data_ptr<Element>() : nullptr, \
                need_ra ? tra.data_ptr<Element>() : nullptr, need_weights ? tw.data_ptr<Element>() : nullptr, \
                need_rw ? trw.data_ptr<Element>() : nullptr, nodes/* row tables */)'''
        launch = (launch.replace("/* node tiles */", node_tiles if grouped_rows else f"({node_tiles})")
                  .replace("/* row tables */", row_tables))
    else:
        lines = [f"        const int64_t node_tiles = {node_tiles};"]
        calls = []
        for index, (begin, end) in enumerate(bands):
            lines += [f"        const int64_t band_tasks{index} = node_tiles * {end - begin};",
                      f"        const int band_x{index} = std::min<int64_t>(band_tasks{index}, 65535);",
                      f"        const dim3 band_blocks{index}(band_x{index}, std::min<int64_t>((band_tasks{index} + band_x{index} - 1) / band_x{index}, 65535));"]
            calls += [f"{function}<WRITE_A, ATTRS_DIRECTIONS, ALL_GRADS, WEIGHT_DIRECTIONS, {index}><<<band_blocks{index}, NODE_THREADS, 3 * {shared_dims[index]} * sizeof(Element), stream>>>(",
                      "x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(), grad_out.data_ptr<Element>(),",
                      "rx_ptr, rw_ptr, ra_ptr, bx_ptr, bw_ptr, ba_ptr, bg_ptr,",
                      "need_x ? tx.data_ptr<Element>() : nullptr, need_attrs ? ta.data_ptr<Element>() : nullptr,",
                      "need_grad_out ? tg.data_ptr<Element>() : nullptr, need_rx ? trx.data_ptr<Element>() : nullptr,",
                      "need_ra ? tra.data_ptr<Element>() : nullptr, need_weights ? tw.data_ptr<Element>() : nullptr,",
                      f"need_rw ? trw.data_ptr<Element>() : nullptr, nodes{row_tables});"]
        continuation = " " + chr(92) + "\n"
        launch = "\n".join(lines) + "\n        #define LAUNCH_FUSED(WRITE_A, ATTRS_DIRECTIONS, ALL_GRADS, WEIGHT_DIRECTIONS) do {" + continuation
        launch += "".join("            " + line + continuation for line in calls)
        launch += "        } while (false)"
    configure = []
    for flags in ((False, False, False, True), (False, True, False, True),
                  (True, False, False, True), (True, True, False, True),
                  (False, False, True, False), (False, False, True, True)):
        arguments = ", ".join(str(flag).lower() for flag in flags)
        for index in range(len(bands) if banded else 1):
            args = arguments + (f", {index}" if banded else "")
            configure.append("    C10_CUDA_CHECK(cudaFuncSetAttribute(\n"
                             f"        symmetric_triple_backward::{function}<{args}>,\n"
                             "        cudaFuncAttributeMaxDynamicSharedMemorySize, symmetric_triple_backward::DYNAMIC_SHARED_BYTES));")
    return launch, "\n".join(configure)


def component_blocks(terms, outputs, guards, polynomial_code, projection_cache=None, projection_code=None):
    count = len(outputs)
    prefix = [f"Element grad_products[{count}] = {{}}, grad_rx_products[{count}] = {{}};",
            f"Element grad_bx_products[{count}] = {{}}, grad_rx_bx_products[{count}] = {{}};"]
    blocks = []
    for component in sorted({m for m, inputs, value in terms}):
        code = ["{", "    Element ap = 0, ar = 0, ab = 0, arb = 0;"]
        channels, names = [], {}
        for slot, outer in enumerate(outputs):
            by_raw = dict(outer_projection_items(outer, component))
            channels.append(by_raw)
            if not by_raw:
                continue
            first = len(code)
            code += [f"    Element h0_{slot} = 0, hr_{slot} = 0, hb_{slot} = 0, hbr_{slot} = 0;"]
            if guards[slot]:
                code.append(f"    if ({guards[slot]}) {{")
            for raw, items in sorted(by_raw.items(), key=lambda item: -1 if item[0] is None else item[0]):
                code += ["    {", "        Element gradient = 0, direction = 0;"]
                for output, coefficient in items:
                    code += [f"        gradient = fma(Element({coefficient:.17g}), saved_gradient{output}, gradient);",
                             f"        direction = fma(Element({coefficient:.17g}), saved_direction{output}, direction);"]
                if raw is None:
                    code += [f"        h0_{slot} += gradient;", f"        hb_{slot} += direction;"]
                else:
                    code += [f"        const Element x = feature[{raw} * (NODE_THREADS + 1)];",
                             f"        const Element r = direction_rx[{raw} * (NODE_THREADS + 1)];",
                             f"        const Element b = direction_bx[{raw} * (NODE_THREADS + 1)];",
                             f"        h0_{slot} = fma(gradient, x, h0_{slot});",
                             f"        hr_{slot} = fma(gradient, r, hr_{slot});",
                             f"        hb_{slot} = fma(gradient, b, fma(direction, x, hb_{slot}));",
                             f"        hbr_{slot} = fma(direction, r, hbr_{slot});"]
                code.append("    }")
            if projection_cache is not None:
                cached = projection_cache.setdefault(component, {})
                key = tuple(by_raw.items()), guards[slot]
                if key not in cached:
                    number = len(cached)
                    assert number < 4
                    cached[key] = {name: f"projection_{component}_{number}_{name}"
                                   for name in ("h0", "hr", "hb", "hbr")}
                    projection = "\n".join(code[first:] + (["    }"] if guards[slot] else []))
                    for before, after in cached[key].items():
                        projection = projection.replace(f"{before}_{slot}", after)
                    projection_code.setdefault(component, []).append(projection)
                names.update({f"{name}_{slot}": value for name, value in cached[key].items()})
                code[first:] = [f"    if ({guards[slot]}) {{"] if guards[slot] else []
            code += ["    if (AllGrads || tx || trx) {",
                     f"        const Element w = mixed[{slot}][0], wr = mixed[{slot}][1];",
                     f"        const Element wb = mixed[{slot}][2], wrb = mixed[{slot}][3];",
                     "        if constexpr (WeightDirections || HasAttrsDirections) {",
                     f"            ap = fma(wrb, h0_{slot}, fma(wb, hr_{slot}, fma(wr, hb_{slot}, fma(w, hbr_{slot}, ap))));",
                     f"            ar = fma(wb, h0_{slot}, fma(w, hb_{slot}, ar));",
                     f"            ab = fma(wr, h0_{slot}, fma(w, hr_{slot}, ab));",
                     "        } else {",
                     f"            ap = fma(w, hbr_{slot}, ap);",
                     f"            ar = fma(w, hb_{slot}, ar);",
                     f"            ab = fma(w, hr_{slot}, ab);", "        }",
                     f"        arb = fma(w, h0_{slot}, arb);", "    }"]
            if guards[slot]:
                code.append("    }")
        code.append("    Element p = 0, pr = 0, pb = 0, prb = 0;")
        selected = [(inputs, value) for m, inputs, value in terms if m == component]
        code.extend("    " + line for line in polynomial_code(
            selected, ("p", "pr", "pb", "prb"), ("ap", "ar", "ab", "arb"),
            "true", "(AllGrads || tx)", "(AllGrads || trx)", "feature_sum", "feature_rx_sum"))
        for slot, by_raw in enumerate(channels):
            if not by_raw:
                continue
            if guards[slot]:
                code.append(f"    if ({guards[slot]}) {{")
            code += ["    {", "        if constexpr (WriteAttrs || HasAttrsDirections) {",
                     f"            grad_products[{slot}] = fma(h0_{slot}, p, grad_products[{slot}]);",
                     f"            grad_rx_products[{slot}] = fma(h0_{slot}, pr, fma(hr_{slot}, p, grad_rx_products[{slot}]));", "        }",
                     f"        grad_bx_products[{slot}] = fma(h0_{slot}, pb, fma(hb_{slot}, p, grad_bx_products[{slot}]));",
                     f"        grad_rx_bx_products[{slot}] = fma(h0_{slot}, prb, fma(hr_{slot}, pb, fma(hb_{slot}, pr, fma(hbr_{slot}, p, grad_rx_bx_products[{slot}]))));",
                     f"        const Element w = mixed[{slot}][0], wr = mixed[{slot}][1];",
                     f"        const Element wb = mixed[{slot}][2], wrb = mixed[{slot}][3];",
                     "        Element value, value_rx, value_bx;",
                     "        if constexpr (WeightDirections || HasAttrsDirections) {",
                     "            value = fma(wrb, p, fma(wb, pr, fma(wr, pb, w * prb)));",
                     "            value_rx = fma(wb, p, w * pb);",
                     "            value_bx = fma(wr, p, w * pr);",
                     "        } else { value = w * prb; value_rx = w * pb; value_bx = w * pr; }",
                     "        const Element value_rx_bx = w * p;"]
            for raw, items in sorted(by_raw.items(), key=lambda item: -1 if item[0] is None else item[0]):
                code.append("        {")
                if raw is None:
                    code.append("            if (AllGrads || tg) {")
                    for output, coefficient in items:
                        code.append(f"                output_sum[{output}] = fma(Element({coefficient:.17g}), value, output_sum[{output}]);")
                    code += ["            }", "        }"]
                    continue
                code += ["            if (AllGrads || tg) {",
                         f"                const Element x = feature[{raw} * (NODE_THREADS + 1)];",
                         f"                const Element r = direction_rx[{raw} * (NODE_THREADS + 1)];",
                         f"                const Element b = direction_bx[{raw} * (NODE_THREADS + 1)];",
                         "                const Element contribution = fma(value, x, fma(value_rx, r, value_bx * b));"]
                for output, coefficient in items:
                    code.append(f"                output_sum[{output}] = fma(Element({coefficient:.17g}), contribution, output_sum[{output}]);")
                code += ["            }", "            if (AllGrads || tx || trx) {",
                         "                Element gradient = 0, direction = 0;"]
                for output, coefficient in items:
                    code += [f"                gradient = fma(Element({coefficient:.17g}), saved_gradient{output}, gradient);",
                             f"                direction = fma(Element({coefficient:.17g}), saved_direction{output}, direction);"]
                code += [f"                if (AllGrads || tx) feature_sum[{raw}] = fma(value, gradient, fma(value_bx, direction, feature_sum[{raw}]));",
                         f"                if (AllGrads || trx) feature_rx_sum[{raw}] = fma(value_rx, gradient, fma(value_rx_bx, direction, feature_rx_sum[{raw}]));",
                         "            }", "        }"]
            code.append("    }")
            if guards[slot]:
                code.append("    }")
        code.append("}")
        block = "\n".join(code)
        for before, after in names.items():
            block = block.replace(before, after)
        blocks.append(block)
    return prefix, blocks


def consumer_outer(binary, consumers):
    paths = {path["basis_path"]: path for path in binary["selected_paths"]}
    blocks = tuple(binary["local_cg_blocks"].values())
    outputs = []
    for basis, original, key, sign in consumers:
        by_raw = defaultdict(list)
        for output, items in consumer_outputs(paths[basis], blocks, sign).items():
            for raw, inner, coefficient in items:
                by_raw[raw].append((output, inner, coefficient))
        outputs.append(by_raw)
    return outputs


def cache_component_inputs(code, reference, reads, max_inputs=6):
    counts = [(value, reference.count(value)) for value in reads if reference.count(value) >= 3]
    counts.sort(key=lambda item: (-item[1], reference.index(item[0])))
    declarations = []
    for index, (value, _) in enumerate(counts[:max_inputs]):
        name = f"component_input{index}"
        declarations.append(f"const Element {name} = {value};")
        code = code.replace(value, name)
    return "{\n" + "\n".join(declarations) + code[1:]


def component_math_blocks(terms, outputs, guards, locations, combine_powers=False,
                          projection_cache=None, projection_code=None, cache_inputs=True):
    prefix, original = component_blocks(terms, outputs, guards, polynomial_triple_backward_code)

    def polynomial(terms, *arguments):
        return polynomial_triple_backward_code(terms, *arguments, combine_squares=True, combine_powers=combine_powers)

    _, optimized = component_blocks(terms, outputs, guards, polynomial, projection_cache, projection_code)
    blocks = []
    for before, after in zip(original, optimized):
        for component, location in locations.items():
            for name in ("feature", "direction_rx", "direction_bx"):
                source, target = f"{name}[{component} * (NODE_THREADS + 1)]", f"{name}[{location}]"
                before, after = before.replace(source, target), after.replace(source, target)
        if not any(guards) and cache_inputs:
            reads = [f"{name}[{location}]" for location in locations.values()
                     for name in ("feature", "direction_rx", "direction_bx")]
            source = before if projection_cache is None else after
            after = cache_component_inputs(after, source, reads)
        blocks.append(after)
    return prefix, blocks


def projected_weights(layout, consumers, weight_paths, limits, grouped_rows):
    count = len(consumers)
    declarations = [f"Element w{slot} = 0, r{slot} = 0, b{slot} = 0;" for slot in range(count)]
    base = "WEIGHT_SIZE" if layout.compact else "NUM_PATHS * CHANNELS"
    operations = []
    for slot, basis in enumerate(consumers):
        lines = []
        for path, coefficient in weight_paths[basis]:
            for pointer, value in (("weights", "w"), ("rw", "r"), ("bw", "b")):
                guard = "" if pointer == "weights" else f"if constexpr (WeightDirections) if (AllGrads || {pointer}) "
                lines.append(f"{guard}{value}{slot} = fma(Element({coefficient:.17g}), {layout.load(pointer, path)}, {value}{slot});")
        operations.append(lines)
    projections = [lines[member] for member in range(max(map(len, operations), default=0))
                   for lines in operations if member < len(lines)]
    code = [f"Element mixed[{count}][4] = {{}};"]
    if grouped_rows:
        code += ["if (grouped) {",
                 *("    " + line for line in declarations),
                 f"    if (lane < FUSED_CHANNEL_TILE && {layout.guard('channel', 'CHANNELS')}) {{",
                 f"        const Index base = Index(tile.x) * {base} + channel;",
                 *("        " + line for line in projections), "    }",
                 "    const Element a = valid ? attribute_values[0][local_node][tile.x] : Element(0);"]
        for slot, basis in enumerate(consumers):
            condition = "a != Element(0)"
            if limits[basis] < layout.end:
                condition += f" && {layout.guard('channel', limits[basis])}"
            code += [f"    const Element wg{slot} = __shfl_sync(0xffffffffu, w{slot}, local_channel);",
                     f"    if ({condition}) mixed[{slot}][0] = a * wg{slot};",
                     "    if constexpr (WeightDirections) {",
                     f"        const Element r = __shfl_sync(0xffffffffu, r{slot}, local_channel);",
                     f"        const Element b = __shfl_sync(0xffffffffu, b{slot}, local_channel);",
                     f"        if ({condition}) {{",
                     f"            mixed[{slot}][1] = a * r;",
                     f"            mixed[{slot}][2] = a * b;",
                     "        }", "    }"]
        code += ["} else {"]
    code += ["for (int first = 0; first < active_elements_count; first += ELEMENT_TILE) {",
             "    if (local_node < ELEMENT_TILE) {", "        const int position = local_node;",
             "        const int element = first + position < active_elements_count ? active_elements[first + position] : -1;",
             *("        " + line for line in declarations),
             f"        if (element >= 0 && {layout.guard('channel', 'CHANNELS')}) {{",
             f"            const Index base = Index(element) * {base} + channel;",
             *("            " + line for line in projections)]
    code.append("        }")
    for slot in range(count):
        for plane, value in enumerate(("w", "r", "b")):
            code.append(f"        projected_weights[{slot}][position][{plane}][local_channel] = {value}{slot};")
    code += ["    }", "    __syncthreads();"]
    for slot, basis in enumerate(consumers):
        active = "valid" if limits[basis] >= layout.end else f"(valid && {layout.guard('channel', limits[basis])})"
        code += ["    {", f"        Element weighted = mixed[{slot}][0], weighted_rx = mixed[{slot}][1];",
                 f"        Element weighted_bx = mixed[{slot}][2], weighted_rx_bx = mixed[{slot}][3];",
                 "        #pragma unroll 4", "        for (int position = 0; position < ELEMENT_TILE; ++position) {",
                 "            if (first + position >= active_elements_count) break;",
                 "            const int element = active_elements[first + position];",
                 "            const Index ai = node * NUM_ELEMENTS + element;",
                 f"            const Element a = {active} ? attribute_values[0][local_node][element] : Element(0);",
                 f"            const Element ar = {active} && HasAttrsDirections && ra ? attribute_values[1][local_node][element] : Element(0);",
                 f"            const Element ab = {active} && HasAttrsDirections && ba ? attribute_values[2][local_node][element] : Element(0);",
                 "            if (a == 0 && ar == 0 && ab == 0) continue;"]
        for plane, value in enumerate(("w", "r", "b")):
            code.append(f"            const Element {value} = projected_weights[{slot}][position][{plane}][local_channel];")
        code += ["            weighted = fma(a, w, weighted);",
                 "            if constexpr (WeightDirections) weighted_rx = fma(a, r, weighted_rx);",
                 "            if constexpr (WeightDirections) weighted_bx = fma(a, b, weighted_bx);",
                 "            if constexpr (HasAttrsDirections) {",
                 "                weighted_rx = fma(ar, w, weighted_rx);",
                 "                weighted_bx = fma(ab, w, weighted_bx);",
                 "                weighted_rx_bx = fma(ar, b, fma(ab, r, weighted_rx_bx));", "            }", "        }",
                 f"        mixed[{slot}][0] = weighted; mixed[{slot}][1] = weighted_rx;",
                 f"        mixed[{slot}][2] = weighted_bx; mixed[{slot}][3] = weighted_rx_bx;", "    }"]
    return code + ["    __syncthreads();", "}"] + (["}"] if grouped_rows else [])


def attribute_products(layout, mapping, slot):
    base = "WEIGHT_SIZE" if layout.compact else "NUM_PATHS * CHANNELS"
    code = ["if constexpr (WriteAttrs) {", "if (valid) {", "    #pragma unroll 4",
            "    for (int element = 0; element < NUM_ELEMENTS; ++element) {",
            f"        const Index base = Index(element) * {base} + channel;", "        Element w = 0, r = 0, b = 0;"]
    for path, coefficient in mapping:
        for pointer, value in (("weights", "w"), ("rw", "r"), ("bw", "b")):
            guard = "" if pointer == "weights" else f"if constexpr (WeightDirections) if (AllGrads || {pointer}) "
            code.append(f"        {guard}{value} = fma(Element({coefficient:.17g}), {layout.load(pointer, path)}, {value});")
    code += ["        if (ta) attr_sum[element] = fma(w, grb, fma(r, gb, fma(b, gr, attr_sum[element])));",
             "        if (tra) attr_rx_sum[element] = fma(w, gb, fma(b, g0, attr_rx_sum[element]));",
             "    }", "}", "}",
             f"if constexpr (!AllGrads || WeightDirections || HasAttrsDirections) products[{slot}][0][local_node][local_channel] = gb;",
             f"products[{slot}][1][local_node][local_channel] = grb;", "if constexpr (HasAttrsDirections) {",
             f"    products[{slot}][2][local_node][local_channel] = g0;",
             f"    products[{slot}][3][local_node][local_channel] = gr;", "}"]
    return code


def linear_consumer(meta, selected, slot, locations):
    code = [f"const Element weighted = mixed[{slot}][0], weighted_rx = mixed[{slot}][1];",
            f"const Element weighted_bx = mixed[{slot}][2], weighted_rx_bx = mixed[{slot}][3];",
            "Element g0 = 0, gr = 0, gb = 0, grb = 0;"]
    for component in range(selected["output_components"]):
        output = selected["output_component"] + component
        feature = selected["tree"]["start"] + component
        index = f"node * OUTPUT_DIM + {meta['output_offsets'][output]} + Index(channel) * {meta['output_strides'][output]}"
        code += ["{", f"    const Index oi = {index};",
                 f"    const Element g = saved_gradient{output}, bgrad = saved_direction{output};",
                 "    const Element gp = WeightDirections || HasAttrsDirections ? fma(g, weighted_rx_bx, bgrad * weighted_rx) : Element(0);",
                 "    const Element gpr = WeightDirections || HasAttrsDirections ? fma(g, weighted_bx, bgrad * weighted) : bgrad * weighted;",
                 "    const Element gpb = WeightDirections || HasAttrsDirections ? g * weighted_rx : Element(0), gprb = g * weighted;",
                 "    Element p = 0, pr = 0, pb = 0, prb = 0;",
                 f"    p += feature[{locations[feature]}];", f"    pr += direction_rx[{locations[feature]}];",
                 f"    pb += direction_bx[{locations[feature]}];",
                 f"    if (AllGrads || tx) feature_sum[{feature}] += gp;",
                 f"    if (AllGrads || trx) feature_rx_sum[{feature}] += gpr;",
                 f"    if (AllGrads || tg) output_sum[{output}] = WeightDirections || HasAttrsDirections ? fma(weighted, prb, fma(weighted_rx, pb, fma(weighted_bx, pr, fma(weighted_rx_bx, p, output_sum[{output}])))) : fma(weighted, prb, output_sum[{output}]);",
                 "    if constexpr (WriteAttrs || HasAttrsDirections) { g0 = fma(g, p, g0); gr = fma(g, pr, gr); }",
                 "    gb = fma(g, pb, fma(bgrad, p, gb));", "    grb = fma(g, prb, fma(bgrad, pr, grb));", "}"]
    return code


def zero_products(slot):
    return [f"if constexpr (!AllGrads || WeightDirections || HasAttrsDirections) products[{slot}][0][local_node][local_channel] = 0;",
            f"products[{slot}][1][local_node][local_channel] = 0;", "if constexpr (HasAttrsDirections) {",
            f"    products[{slot}][2][local_node][local_channel] = 0;",
            f"    products[{slot}][3][local_node][local_channel] = 0;", "}"]


def shared_projection_window(meta, layout, packet, weight_paths, limits, locations,
                             groups, combine_powers, grouped_rows):
    if any(not terms for terms, _ in packet):
        return None
    cache, projections, records, prefixes = {}, {}, [], []
    offset = occurrences = 0
    guarded = any(limits[basis] < meta["channels"] and limits[basis] < layout.end
                  for _, consumers in packet for basis, _ in consumers)
    cache_shared_inputs = not guarded and meta["input_dim"] > 16
    if grouped_rows and not cache_shared_inputs:
        return None
    for terms, consumers in packet:
        bases = [basis for basis, _ in consumers]
        guards = [layout.guard("channel", limits[basis])
                  if limits[basis] < meta["channels"] and limits[basis] < layout.end else None for basis in bases]
        outputs = [outer for _, outer in consumers]
        prefix, blocks = component_math_blocks(terms, outputs, guards, locations, combine_powers,
                                                cache, projections, cache_inputs=not cache_shared_inputs)
        components = sorted({component for component, _, _ in terms})
        occurrences += sum(bool(outer_projection_items(outer, component))
                           for outer in outputs for component in components)
        names = {name: f"{name}_{offset}" for name in
                 ("grad_products", "grad_rx_products", "grad_bx_products", "grad_rx_bx_products")}
        prefix, blocks = "\n".join(prefix), dict(zip(components, blocks))
        for before, after in names.items():
            prefix = prefix.replace(before, after)
            blocks = {component: block.replace(before, after) for component, block in blocks.items()}
        prefixes.append(prefix)
        records.append((offset, bases, guards, blocks, names))
        offset += len(consumers)
    if occurrences == sum(len(items) for items in cache.values()):
        return None
    bases = [basis for _, consumers in packet for basis, _ in consumers]
    code = projected_weights(layout, bases, weight_paths, limits, grouped_rows)
    code += ["const auto* window_weights = mixed;", *prefixes]
    for component in sorted(projections):
        projection = "\n".join(projections[component])
        for raw, location in locations.items():
            for name in ("feature", "direction_rx", "direction_bx"):
                projection = projection.replace(f"{name}[{raw} * (NODE_THREADS + 1)]", f"{name}[{location}]")
        body = ["{", projection]
        for first, _, _, blocks, _ in records:
            if component in blocks:
                body += ["{", f"const auto* mixed = window_weights + {first};", blocks[component], "}"]
        body.append("}")
        if cache_shared_inputs:
            reads = [f"{name}[{location}]" for location in locations.values()
                     for name in ("feature", "direction_rx", "direction_bx")]
            source = "\n".join(body)
            body = [cache_component_inputs(source, source, reads, max_inputs=24)]
        code += body
    for first, bases, guards, _, names in records:
        for local, basis in enumerate(bases):
            mapping = weight_paths[basis]
            groups.append((mapping, limits[basis]))
            slot = first + local
            body = ["{", f"Element g0 = {names['grad_products']}[{local}], gr = {names['grad_rx_products']}[{local}];",
                    f"Element gb = {names['grad_bx_products']}[{local}], grb = {names['grad_rx_bx_products']}[{local}];",
                    *attribute_products(layout, mapping, slot), "}"]
            if limits[basis] < meta["channels"] and guards[local]:
                body = [f"if ({guards[local]}) {{", *body, "} else {", *zero_products(slot), "}"]
            code += body
    for slot in range(offset, 4):
        groups.append(((), 0))
        code += zero_products(slot)
    return code


def window_code(meta, layout, packet, binary, weight_paths, limits, locations, groups,
                combine_powers=False, grouped_rows=False, project_window=False):
    if project_window and not binary:
        shared = shared_projection_window(meta, layout, packet, weight_paths, limits, locations,
                                          groups, combine_powers, grouped_rows)
        if shared is not None:
            return shared
    code, offset = [], 0
    paths = {item["basis_path"]: item for item in binary["selected_paths"]} if binary else {}
    if project_window and packet:
        bases = [item[0] for _, consumers in packet for item in consumers]
        code += projected_weights(layout, bases, weight_paths, limits, grouped_rows)
        code.append("const auto* window_weights = mixed;")
    for terms, consumers in packet:
        bases = [item[0] for item in consumers]
        pair_start = len(code)
        code.append("{")
        if project_window:
            code.append(f"const auto* mixed = window_weights + {offset};")
        else:
            code += projected_weights(layout, bases, weight_paths, limits, grouped_rows)
        guards = [layout.guard("channel", limits[basis])
                  if limits[basis] < meta["channels"] and limits[basis] < layout.end else None for basis in bases]
        if terms:
            outputs = consumer_outer(binary, consumers) if binary else [item[1] for item in consumers]
            prefix, blocks = component_math_blocks(terms, outputs, guards, locations, combine_powers)
            code.append("\n".join(prefix + blocks))
        for local, basis in enumerate(bases):
            mapping = weight_paths[basis]
            groups.append((mapping, limits[basis]))
            slot = offset + local
            body = ["{"]
            if terms:
                body += [f"    Element g0 = grad_products[{local}], gr = grad_rx_products[{local}];",
                         f"    Element gb = grad_bx_products[{local}], grb = grad_rx_bx_products[{local}];"]
            else:
                body += linear_consumer(meta, paths[basis], local, locations)
            body += attribute_products(layout, mapping, slot) + ["}"]
            if limits[basis] < meta["channels"]:
                body = ([f"if ({guards[local]}) {{", *body, "} else {", *zero_products(slot), "}"]
                        if guards[local] else ["{", *body, "}"])
            code += body
        offset += len(consumers)
        code.append("}")
        if max(limits[basis] for basis in bases) < meta["channels"]:
            code[pair_start:] = ["{", *code[pair_start:], "}"]
    for slot in range(offset, 4):
        groups.append(((), 0))
        code += zero_products(slot)
    return code


def make_cg_triple_blocks(meta, plan, compact):
    forward_functions, backward_functions = [], []
    active = {node[3] for node in plan["arrays"]["CgNodes"]}
    for block_id, block in enumerate(meta["cg_dag"]["local_cg"]):
        if block_id not in active:
            continue
        edges = [(output, block["left_components"][edge], block["right_components"][edge],
                  block["coefficients"][edge])
                 for output in range(block["shape"][0])
                 for edge in range(block["output_offsets"][output], block["output_offsets"][output + 1])]
        forward = [f"__device__ __noinline__ void cg_forward_{block_id}(Element* values, int left, int right, int output, int worker) {{"]
        for name, field in (("left", 1), ("right", 2)):
            for component in sorted({edge[field] for edge in edges}):
                forward.append(f"    const Jet4 {name}{component} = load_jet(values, {name} + {component}, worker);")
        for component in range(block["shape"][0]):
            forward += ["    {", "        Jet4 total = {};"]
            for target, a, b, coefficient in edges:
                if target == component:
                    forward.append(f"        jet4_add_scaled(total, Element({coefficient:.17g}), "
                                   f"jet4_multiply(left{a}, right{b}));")
            forward += [f"        store_jet(values, output + {component}, worker, total);", "    }"]
        forward.append("}")
        forward_functions.append("\n".join(forward))

        arguments = ("int grad_output, int grad_left, int grad_right, int clear" if compact else "int output")
        output = "grad_output" if compact else "output"
        backward = [f"__device__ __noinline__ void cg_backward_{block_id}(const Element* values, Element* totals, int left, int right, {arguments}, int worker) {{"]
        if compact:
            for flag, name, dimension in ((1, "grad_output", block["shape"][0]), (2, "grad_left", block["shape"][1]), (4, "grad_right", block["shape"][2])):
                backward += [f"    if (CompactArena && (clear & {flag})) {{", "        #pragma unroll",
                             f"        for (int component = 0; component < {dimension}; ++component)",
                             f"            store_jet(totals, {name} + component, worker, {{}});", "    }"]
        for side, other, field in (("left", "right", 1), ("right", "left", 2)):
            backward.append("    {")
            for component in sorted({edge[0] for edge in edges}):
                backward.append(f"    const Jet4 output{component} = load_jet(totals, {output} + {component}, worker);")
            for component in sorted({edge[2 if field == 1 else 1] for edge in edges}):
                backward.append(f"    const Jet4 {other}{component} = load_jet(values, {other} + {component}, worker);")
            for component in sorted({edge[field] for edge in edges}):
                backward += ["    {", "        Jet4 gradient = {};"]
                for target, a, b, coefficient in edges:
                    if (a if field == 1 else b) == component:
                        other_component = b if field == 1 else a
                        backward += ["        {", "            Jet4 total = {}, unused = {};",
                                     f"            jet4_add_scaled(total, Element({coefficient:.17g}), output{target});",
                                     f"            jet4_reverse({{}}, {other}{other_component}, total, gradient, unused);",
                                     "        }"]
                total = f"grad_{side}" if compact else side
                backward += [f"        add_jet(totals, {total} + {component}, worker, gradient);", "    }"]
            backward.append("    }")
        backward.append("}")
        backward_functions.append("\n".join(backward))

    forward_dispatch = ["__device__ __forceinline__ void cg_forward(Element* values, int left, int right, int output, int block, int worker) {",
                        "    switch (block) {"]
    arguments = "int grad_output, int grad_left, int grad_right, int clear" if compact else "int output"
    backward_arguments = "grad_output, grad_left, grad_right, clear" if compact else "output"
    backward_dispatch = [f"__device__ __forceinline__ void cg_backward(const Element* values, Element* totals, int left, int right, {arguments}, int block, int worker) {{",
                         "    switch (block) {"]
    for block_id in sorted(active):
        forward_dispatch.append(f"        case {block_id}: cg_forward_{block_id}(values, left, right, output, worker); break;")
        backward_dispatch.append(f"        case {block_id}: cg_backward_{block_id}(values, totals, left, right, {backward_arguments}, worker); break;")
    forward_dispatch += ["    }", "}"]
    backward_dispatch += ["    }", "}"]
    return "\n\n".join((*forward_functions, *backward_functions,
                          "\n".join(forward_dispatch), "\n".join(backward_dispatch)))


def make_triple_backward_source(meta, dtype="float32"):
    """Use the pair or polynomial path when available; otherwise schedule a CG arena."""
    scalar = "double" if "64" in str(dtype) else "float"
    reduced_pairs = ("cg_dag" in meta and meta.get("use_reduced_cg", False)
                     and meta["correlation"] <= 3)
    if "cg_dag" in meta and not reduced_pairs:
        polynomials = bounded_cg_polynomials(meta)
        if polynomials is not None:
            meta = polynomials
    if "cg_dag" in meta and not reduced_pairs:
        from .utils import (CG_ELEMENT_TILE, CG_MEMBER_LITERAL_LIMIT, TRIPLE_CG_NODE_TILE,
                            build_cg_plan, cg_member_constants, cg_plan_constants)

        plan_meta = dict(meta, source_dtype="float64" if scalar == "double" else "float32")
        shared_bytes = meta.get("shared_memory_bytes", SHARED_MEMORY_BYTES)
        identity = all(len(members) == 1 and tuple(members[0]) == (basis, 1.0)
                       for basis, members in enumerate(meta["cg_dag"]["weight_members"]))
        original_plan = build_cg_plan(plan_meta, planes_per_slot=8,
                                      shared_bytes=shared_bytes // CG_SHARED_BUDGET_DIVISOR)
        compact_plan = build_arena_plan(plan_meta, planes_per_value=4)
        if not identity:
            larger_plan = build_cg_plan(plan_meta, planes_per_slot=8, shared_bytes=shared_bytes // TRIPLE_CG_SHARED_BUDGET_DIVISOR)
            larger = larger_plan["constants"]
            if larger["Workers"] > max(original_plan["constants"]["Workers"],
                                       compact_plan["constants"]["Workers"]):
                roots = max(end - begin for _, _, begin, end, _ in larger_plan["arrays"]["CgPackets"])
                projection_slots = roots * min(meta["num_elements"], CG_ELEMENT_TILE) * larger["ChannelColumns"]
                larger_bytes = (8 * larger["MaxSlots"] * larger["ValueStride"]
                                + 6 * projection_slots) * larger_plan["item_bytes"]
                if larger_bytes <= shared_bytes:
                    original_plan = larger_plan
        compact = compact_plan["constants"]["Workers"] > original_plan["constants"]["Workers"]
        plan = arena_addresses(meta, compact_plan) if compact else original_plan
        if not compact:
            plan["arrays"].update(CgNodeTotals=[0], CgNodeZeros=[0], CgRootTotals=[0], CgRootZeros=[0])
        constants = f"using Element = {scalar};\nusing JetVector = {scalar}4;\nconstexpr bool CompactArena = {str(compact).lower()};\n"
        constants += f"constexpr bool IdentityWeights = {'true' if identity else 'false'};\n"
        members = plan["arrays"]["CgMembers"]
        packed_members = len(members) > CG_MEMBER_LITERAL_LIMIT
        table_plan = dict(plan, arrays={key: value for key, value in plan["arrays"].items()
                                       if key != "CgMembers"}) if packed_members else plan
        constants += cg_plan_constants(table_plan, constant_budget=0) + "\n"
        if packed_members:
            constants += cg_member_constants(members, 8 if scalar == "double" else 4) + "\n"
        max_roots = max(end - begin for _, _, begin, end, _ in plan['arrays']['CgPackets'])
        constants += f'constexpr int MaxPacketRoots = {max_roots};\n'
        constants += f"constexpr int CgNodeTile = {TRIPLE_CG_NODE_TILE};\nconstexpr int CgElementTile = {CG_ELEMENT_TILE};\n"
        backward_call = """                cg_backward(values, totals, operation_node.left, operation_node.right,
                            CompactArena ? CgNodeTotals[3 * operation] : operation_node.value,
                            CompactArena ? CgNodeTotals[3 * operation + 1] : operation_node.left,
                            CompactArena ? CgNodeTotals[3 * operation + 2] : operation_node.right,
                            CompactArena ? CgNodeZeros[operation] : 0,
                            operation_node.block, worker);""" if compact else """                cg_backward(values, totals, operation_node.left, operation_node.right,
                            operation_node.value, operation_node.block, worker);"""
        return CG_CUDA_SOURCE.replace("/* constants */", constants).replace(
            "/* backward_call */", backward_call).replace(
            "/* cg_blocks */", make_cg_triple_blocks(meta, plan, compact)).replace(
            "/* member_load */", "load_cg_member(member_id)" if packed_members else "CgMembers[member_id]").replace(
            "/* precision */", "fp64" if scalar == "double" else "fp32")
    item_bytes = 8 if scalar == "double" else 4
    compact = any(count != meta["channels"] for count in (*meta["input_channels"], *meta["output_channels"]))
    binary = meta.get("binary") if meta.get("correlation", 3) <= 3 else None
    inputs = meta.get("input_channels", (meta["channels"],) * meta["input_dim"])
    outputs = meta.get("output_channels", (meta["channels"],) * len(meta["output_offsets"]))
    threads = 128
    shared_bytes = meta.get("shared_memory_bytes", 96 * 1024)
    while threads > 32 and (3 * meta["input_dim"] * (threads + 1) +
                            16 * threads + 3 * meta["num_elements"] * (threads // 8)) * item_bytes > shared_bytes:
        threads //= 2
    grouped_rows = scalar == "float" and meta["num_elements"] > threads // 8
    share_projections = not grouped_rows or meta["input_dim"] > 16
    if binary:
        paths = {path["basis_path"]: path for path in binary["selected_paths"]}
        limits = {basis: path_channel_count(meta, path) for basis, path in paths.items()}
        weight_paths = binary["compute_weight_paths"]
        base_windows = packed_windows(binary)
    else:
        if reduced_pairs:
            groups, weight_paths, limits = reduced_pair_groups(meta)
        else:
            groups, weight_paths = contraction_groups(meta)
            limits = {}
        entries = []
        for terms, consumers in groups:
            converted = []
            for basis, outer in consumers:
                by_raw = defaultdict(list)
                for output, items in outer.items():
                    for raw, inner, coefficient in items:
                        by_raw[raw].append((output, inner, coefficient))
                converted.append((basis, by_raw))
                if not reduced_pairs:
                    limits[basis] = max(min(outputs[output], *(inputs[component] for component in components))
                                        for output, components, coefficient in terms)
            entries.extend((terms, converted[first:first + 4]) for first in range(0, len(converted), 4))
        if reduced_pairs:
            base_windows = (pack_projection_windows(entries, meta, limits) if share_projections
                            else pack_windows(entries))
        else:
            base_windows = []
            for terms, selected in entries:
                if (scalar == "double" and base_windows
                        and sum(len(items) for _, items in base_windows[-1]) + len(selected) <= 4):
                    base_windows[-1].append((terms, selected))
                else:
                    base_windows.append([(terms, selected)])
            base_windows = base_windows or [[]]
    boundaries = sorted({0, meta["channels"], *(min(meta["channels"], ((limit + 7) // 8) * 8)
                                               for limit in limits.values() if limit)})
    banded = compact and len(boundaries) > 2
    bands = [(begin // 8, (end + 7) // 8) for begin, end in zip(boundaries, boundaries[1:])] if banded else [(0, (meta["channels"] + 7) // 8)]
    layouts, plans = [], []
    for begin, end in bands:
        layout = ContractionLayout(meta, limits, banded, end * 8)
        layouts.append(layout)
        if not banded:
            plans.append(base_windows)
        elif binary:
            active = {basis for basis, limit in limits.items() if limit > begin * 8}
            selected = dict(binary, selected_paths=[path for path in binary["selected_paths"]
                                                   if path["basis_path"] in active],
                            pair_groups=[dict(pair, consumers=[item for item in pair["consumers"] if item[0] in active])
                                         for pair in binary["pair_groups"]
                                         if any(item[0] in active for item in pair["consumers"])])
            plans.append(packed_windows(selected))
        elif reduced_pairs:
            active = [(terms, [item for item in consumers if limits[item[0]] > begin * 8])
                      for terms, consumers in entries]
            selected = [entry for entry in active if entry[1]]
            plans.append(pack_projection_windows(selected, meta, limits, layout) if share_projections
                         else pack_windows(selected))
        else:
            windows = [[(terms, [item for item in consumers if limits[item[0]] > begin * 8])
                        for terms, consumers in packet] for packet in base_windows]
            plans.append([[entry for entry in packet if entry[1]] for packet in windows])
    feature_groups = input_groups(meta)
    shared_stride, locations = shared_layout(feature_groups)
    shared_dims = []
    for begin, end in bands:
        active = [(first + dimension, index + 1)
                  for index, (first, dimension, count, offset, stride) in enumerate(feature_groups)
                  if begin * 8 < count]
        component, padding = max(active, default=(0, 1))
        shared_dims.append(f"({component} * NODE_THREADS + {padding})")
    group_records, contractions = [], []
    constraint = "d" if scalar == "double" else "f"
    operands = ", ".join(f'"+{constraint}"({prefix}{component})' for component in range(len(outputs))
                         for prefix in ("saved_gradient", "saved_direction"))
    for band, (layout, windows) in enumerate(zip(layouts, plans)):
        group_offset = len(group_records)
        cases = []
        for window, packet in enumerate(windows):
            cases += [f"case {window}: {{",
                      *window_code(meta, layout, packet, binary, weight_paths, limits, locations,
                                   group_records, scalar == "double", grouped_rows, reduced_pairs),
                      "    break;", "}"]
        count = str(len(windows)) if banded else "FUSED_WINDOWS"
        code = ["            #pragma unroll 1", f"            for (int window = 0; window < {count}; ++window) {{",
                f'                asm volatile("" : {operands});', "                switch (window) {",
                indent("\n".join(cases), " " * 20), "                }"]
        reduction_index = ("Index(element) * WEIGHT_SIZE + PATH_OFFSETS[GROUP_PATHS[member]] + c"
                           if compact else "(Index(element) * NUM_PATHS + GROUP_PATHS[member]) * CHANNELS + c")
        reduction = WINDOW_REDUCTION.replace("/* group_offset */", f"{group_offset} + " if banded else "")
        if grouped_rows:
            reduction = reduction.replace("/* reduction_mask */", "                const unsigned reduction_mask = "
                                          "grouped ? __ballot_sync(0xffffffffu, group < NUM_GROUPS && /* reduction_guard */) : 0;")
            reduction = reduction.replace("/* grouped_reduction */", GROUPED_REDUCTION)
            reduction = reduction.replace("/* grouped_reduction_end */", "                    }")
        else:
            reduction = reduction.replace("/* reduction_mask */\n", "")
            reduction = reduction.replace("/* grouped_reduction */\n", "").replace("/* grouped_reduction_end */\n", "")
        reduction = reduction.replace("/* store_weight_gradient */", WINDOW_STORE)
        reduction = reduction.replace("/* reduction_guard */", "c < GROUP_CHANNELS[group]" if compact else "c < CHANNELS")
        reduction = reduction.replace("/* reduction_index */", reduction_index)
        code.append(reduction)
        body = "\n".join(code)
        if banded:
            body = f"            {'if' if band == 0 else 'else if'} constexpr (ChannelBand == {band}) {{\n" + body + "\n            }"
        contractions.append(body)
    group_paths, group_signs, group_offsets = [], [], [0]
    for members, limit in group_records:
        group_paths.extend(path for path, coefficient in members)
        group_signs.extend(coefficient for path, coefficient in members)
        group_offsets.append(len(group_paths))
    constants = [f"using Element = {scalar};", "using Index = int64_t;"]
    if not compact:
        constants.append(f"constexpr Index INPUT_SIZE = Index({meta['channels']}) * {meta['input_dim']};")
    constants.append(f"constexpr int CHANNEL_GROUP_TILE = {min(4, (meta['channels'] + 7) // 8)};")
    if compact:
        constants += [f"constexpr int INPUT_SIZE = {meta['input_size']};", f"constexpr int WEIGHT_SIZE = {meta['weight_numel']};",
                      "__device__ const int PATH_OFFSETS[] = {" + ", ".join(map(str, meta["path_offsets"])) + "};",
                      "__device__ const int GROUP_CHANNELS[] = {" + ", ".join(str(limit) for members, limit in group_records) + "};"]
    sizes = dict(CHANNELS=meta["channels"], INPUT_DIM=meta["input_dim"], OUTPUT_DIM=meta["output_dim"],
                 NUM_ELEMENTS=meta["num_elements"], NUM_PATHS=meta["num_paths"], NUM_GROUPS=len(group_records),
                 NODE_THREADS=threads, FUSED_CHANNEL_TILE=8, ATOM_TILE=threads // 8,
                 OUTPUT_COMPONENTS=len(outputs), FUSED_WINDOWS=len(base_windows))
    for name, value in sizes.items():
        constants.append(f"constexpr int {name} = {value};")
        if name == "NUM_ELEMENTS":
            constants.append("constexpr int ELEMENT_WORDS = (NUM_ELEMENTS + 31) / 32;")
        if name == "ATOM_TILE":
            constants.append("constexpr int ELEMENT_TILE = NUM_ELEMENTS < ATOM_TILE ? NUM_ELEMENTS : ATOM_TILE;")
    constants += [f"constexpr int INPUT_SHARED_DIM = {shared_stride} * NODE_THREADS + {len(feature_groups)};",
                  "constexpr int DYNAMIC_SHARED_BYTES = 3 * INPUT_SHARED_DIM * sizeof(Element);"]
    arrays = [("GROUP_PATHS", group_paths), ("GROUP_OFFSETS", group_offsets), ("GROUP_SIGNS", group_signs)]
    for name, values in arrays:
        kind = "Element" if name == "GROUP_SIGNS" else "int"
        literals = ", ".join(f"Element({value:.17g})" if kind == "Element" else str(value) for value in (values or [0]))
        constants.append(f"__device__ const {kind} {name}[] = {{{literals}}};")
    node_mapping = ["    const Index node_tiles = (nodes + ATOM_TILE - 1) / ATOM_TILE;"]
    if grouped_rows:
        node_mapping = [
            "    const bool grouped = row_counts && !row_counts[NUM_ELEMENTS];",
            "    const Index node_tiles = grouped ? row_counts[NUM_ELEMENTS + 1] : (nodes + ATOM_TILE - 1) / ATOM_TILE;",
        ]
    if banded:
        mapping = [f"    constexpr int FirstTile = {band_expression([begin for begin, end in bands])};",
                   f"    constexpr int BandTiles = {band_expression([end - begin for begin, end in bands])};",
                   "    constexpr int BandBegin = FirstTile * FUSED_CHANNEL_TILE;",
                   "    constexpr int BandEnd = (FirstTile + BandTiles) * FUSED_CHANNEL_TILE;",
                   *node_mapping,
                   "    for (Index task = Index(blockIdx.y) * gridDim.x + blockIdx.x; task < node_tiles * BandTiles;",
                   "         task += Index(gridDim.x) * gridDim.y) {",
                   "        const Index channel_tile = FirstTile + task % BandTiles;",
                   "        const Index first_node = (task / BandTiles) * ATOM_TILE;"]
        shared = f"    constexpr int BandSharedDim = {band_expression(shared_dims)};\n    extern __shared__ __align__(16) unsigned char shared_memory[];\n    auto* data = reinterpret_cast<volatile Element(*)[BandSharedDim]>(shared_memory);"
    else:
        mapping = ["    const Index channel_tiles = (CHANNELS + FUSED_CHANNEL_TILE - 1) / FUSED_CHANNEL_TILE;",
                   *node_mapping,
                   "    for (Index channel_group = blockIdx.y; channel_group < (channel_tiles + CHANNEL_GROUP_TILE - 1) / CHANNEL_GROUP_TILE; channel_group += gridDim.y)",
                   "    for (Index task = blockIdx.x; task < node_tiles * CHANNEL_GROUP_TILE; task += gridDim.x) {",
                   "        const Index channel_tile = channel_group * CHANNEL_GROUP_TILE + task % CHANNEL_GROUP_TILE;",
                   "        if (channel_tile >= channel_tiles) continue;",
                   "        const Index first_node = (task / CHANNEL_GROUP_TILE) * ATOM_TILE;"]
        shared = "    extern __shared__ __align__(16) unsigned char shared_memory[];\n    auto* data = reinterpret_cast<volatile Element(*)[INPUT_SHARED_DIM]>(shared_memory);"
    loads, stores, staging = feature_code(feature_groups, banded)
    gradients, output_stores = [], []
    layout = layouts[0]
    for component, (offset, stride) in enumerate(zip(meta["output_offsets"], meta["output_strides"])):
        count = outputs[component] if compact else "CHANNELS"
        guard = layout.guard("channel", count)
        index = f"node * OUTPUT_DIM + {offset} + Index(channel) * {stride}"
        gradients += [f"Element saved_gradient{component} = node < nodes && {guard} ? grad_out[{index}] : Element(0);",
                      f"Element saved_direction{component} = node < nodes && {guard} && (AllGrads || bg) ? bg[{index}] : Element(0);"]
        output_stores.append((f"if ({guard}) " if compact else "") + f"tg[{index}] = output_sum[{component}];")
    launch, configure = launch_code(bands, shared_dims, banded, grouped_rows)
    row_code = dict(group_rows="", row_arguments="", row_index="first_node + local_node", row_workspace="",
                    grouped_initialization="", grouped_initialization_end="",
                    grouped_attributes="", grouped_attributes_end="")
    if grouped_rows:
        mapping += [
            "        const int2 tile = grouped ? row_tiles[first_node / ATOM_TILE] : make_int2(0, 0);",
        ]
        row_code.update(
            group_rows=GROUP_ROWS,
            row_arguments=", const int* row_counts, const int* row_permutation, const int2* row_tiles",
            row_index="grouped ? (local_node < tile.y ? row_permutation[first_node + local_node] : nodes) : first_node + local_node",
            row_workspace=ROW_WORKSPACE,
            grouped_initialization="""            if (grouped) {
                if (threadIdx.x == 0) {
                    active_elements_count = 1;
                    active_elements[0] = tile.x;
                    active_rows[tile.x] = 0;
                }
            } else {""",
            grouped_initialization_end="            }",
            grouped_attributes="""            if (grouped) {
                if (local_channel == 0) {
                    const Element a = node < nodes ? attrs[node * NUM_ELEMENTS + tile.x] : Element(0);
                    attribute_values[0][local_node][tile.x] = a;
                    if (a != Element(0)) atomicOr(active_rows + tile.x, 1u << local_node);
                }
            } else {""",
            grouped_attributes_end="            }",
        )
    initialize_weights = ""
    weight_initialization = "true"
    if not compact:
        weight_initialization = "!(nodes > 0 && nodes <= ATOM_TILE)"
        initialize_weights = """            if (nodes <= ATOM_TILE) {
                if (channel < CHANNELS) {
                    for (int element = local_node; element < NUM_ELEMENTS; element += ATOM_TILE) {
                        for (int path = 0; path < NUM_PATHS; ++path) {
                            const Index index = (Index(element) * NUM_PATHS + path) * CHANNELS + channel;
                            if (tw) tw[index] = Element(0);
                            if (trw) trw[index] = Element(0);
                        }
                    }
                }
                __syncthreads();
            }"""
    batch_initialization = """    std::vector<at::Tensor> zero_outputs;
    if (need_attrs) zero_outputs.push_back(ta);
    if (need_ra) zero_outputs.push_back(tra);
    if (/* weight_initialization */) {
        if (need_weights) zero_outputs.push_back(tw);
        if (need_rw) zero_outputs.push_back(trw);
    }
    if (zero_outputs.size() == 1) zero_outputs[0].zero_();
    else if (!zero_outputs.empty()) at::_foreach_zero_(zero_outputs);""".replace(
        "/* weight_initialization */", weight_initialization)
    replacements = dict(constants="\n".join(constants), band_template=", int ChannelBand" if banded else "",
                        initialize_weights=initialize_weights, batch_weight_initialization=batch_initialization,
                        shared_declaration=shared, task_mapping="\n".join(mapping), feature_loads=loads,
                        saved_gradients=indent("\n".join(gradients), " " * 12), channel_contractions="\n".join(contractions),
                        gradient_staging=staging, feature_stores=stores, output_stores=indent("\n".join(output_stores), " " * 20),
                        channel_guard=layout.guard("channel", "CHANNELS"), launch=launch, configure=configure, **row_code)
    channel_contractions = replacements["channel_contractions"]
    powers = POWER_CUDA_SOURCE + "\n\n" if "polynomial_triple_backward_power<" in channel_contractions else ""
    source = (CUDA_SOURCE.replace("/* power_helpers */", powers)
              .replace("/* row_workspace_header */", "#include <ATen/ops/empty.h>\n" if grouped_rows else ""))
    for key, value in replacements.items():
        if not value:
            source = source.replace(f"\n/* {key} */\n", "\n")
        source = source.replace(f"/* {key} */", value)
    return source.replace("/* precision */", "fp64" if scalar == "double" else "fp32")


CG_CUDA_SOURCE = r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/_foreach_zero.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace symmetric_triple_backward {
/* constants */

struct Jet4 { Element value, r, b, rb; };

__device__ __forceinline__ Jet4 jet4_multiply(Jet4 a, Jet4 b) {
    return {a.value*b.value, fma(a.r,b.value,a.value*b.r),
            fma(a.b,b.value,a.value*b.b),
            fma(a.rb,b.value,fma(a.r,b.b,fma(a.b,b.r,a.value*b.rb)))};
}

__device__ __forceinline__ void jet4_add_scaled(Jet4& dst, Element c, Jet4 a) {
    dst.value=fma(c,a.value,dst.value);
    dst.r=fma(c,a.r,dst.r);
    dst.b=fma(c,a.b,dst.b);
    dst.rb=fma(c,a.rb,dst.rb);
}

__device__ __forceinline__ void jet4_reverse(Jet4 a, Jet4 b, Jet4 total,
                                           Jet4& ga, Jet4& gb) {
    ga.value=fma(total.value,b.value,fma(total.r,b.r,fma(total.b,b.b,fma(total.rb,b.rb,ga.value))));
    ga.r=fma(total.r,b.value,fma(total.rb,b.b,ga.r));
    ga.b=fma(total.b,b.value,fma(total.rb,b.r,ga.b));
    ga.rb=fma(total.rb,b.value,ga.rb);
    gb.value=fma(total.value,a.value,fma(total.r,a.r,fma(total.b,a.b,fma(total.rb,a.rb,gb.value))));
    gb.r=fma(total.r,a.value,fma(total.rb,a.b,gb.r));
    gb.b=fma(total.b,a.value,fma(total.rb,a.r,gb.b));
    gb.rb=fma(total.rb,a.value,gb.rb);
}

constexpr bool ReuseWeights = !IdentityWeights;
constexpr int PacketNodeTile = ReuseWeights ? CgNodeTile : Rows;
constexpr int PacketTasks = ReuseWeights ? PacketCount : 1;
constexpr int ElementTile = ReuseWeights && NumElements > CgElementTile ? CgElementTile : NumElements;
constexpr int ElementTiles = (NumElements + ElementTile - 1) / ElementTile;
constexpr int ArenaBytes = (CompactArena ? 4 : 8) * MaxSlots * ValueStride * sizeof(Element);
constexpr int ProjectionSlots = MaxPacketRoots * ElementTile * ChannelColumns;
constexpr int SharedBytes = ArenaBytes + (ReuseWeights ? 6 * ProjectionSlots * sizeof(Element) : 0);
constexpr unsigned WorkerMask = 0xffffffffu >> (32 - (Workers < 32 ? Workers : 32));


__device__ __forceinline__ Jet4 load_jet(const Element* values, int slot, int worker) {
    const JetVector value = reinterpret_cast<const JetVector*>(values)[slot * ValueStride + worker];
    return {value.x, value.y, value.z, value.w};
}

__device__ __forceinline__ void store_jet(Element* values, int slot, int worker, Jet4 value) {
    reinterpret_cast<JetVector*>(values)[slot * ValueStride + worker] =
        JetVector{value.value, value.r, value.b, value.rb};
}

__device__ __forceinline__ void add_jet(Element* values, int slot, int worker, Jet4 value) {
    JetVector* address = reinterpret_cast<JetVector*>(values) + slot * ValueStride + worker;
    JetVector total = *address;
    total.x += value.value;
    total.y += value.r;
    total.z += value.b;
    total.w += value.rb;
    *address = total;
}

/* cg_blocks */

__global__ void lunaeq_symmetric_contraction_triplebwd_/* precision */_i64(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ rw, const Element* __restrict__ ra,
    const Element* __restrict__ bx, const Element* __restrict__ bw, const Element* __restrict__ ba,
    const Element* __restrict__ bg, Element* __restrict__ tx, Element* __restrict__ tw,
    Element* __restrict__ ta, Element* __restrict__ tg, Element* __restrict__ trx,
    Element* __restrict__ trw, Element* __restrict__ tra, int64_t nodes
) {
    extern __shared__ __align__(16) unsigned char shared_memory[];
    Element* values = reinterpret_cast<Element*>(shared_memory);
    Element* totals = CompactArena ? values : values + 4 * MaxSlots * ValueStride;
    const int worker = threadIdx.x;
    const int row = worker / ChannelColumns;
    const int column = worker % ChannelColumns;
    constexpr int ChannelTiles = (Channels + ChannelColumns - 1) / ChannelColumns;
    const int64_t node_tiles = (nodes + PacketNodeTile - 1) / PacketNodeTile;
    const int64_t tasks = node_tiles * ChannelTiles * PacketTasks * ElementTiles;

    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int packet_id = ReuseWeights ? (ElementTiles == 1 ? task / (node_tiles * ChannelTiles)
                             : (task / (node_tiles * ChannelTiles)) % PacketCount) : 0;
        const int element_begin = ElementTiles == 1 ? 0
                                 : (task / (node_tiles * ChannelTiles * PacketCount)) * ElementTile;
        const int64_t node_begin = ReuseWeights ? ((task / ChannelTiles) % node_tiles) * PacketNodeTile
                                               : (task / ChannelTiles) * Rows;
        const int channel = (task % ChannelTiles) * ChannelColumns + column;
        const CgPacket projected_packet = CgPackets[packet_id];
        Jet4* projected_weights = reinterpret_cast<Jet4*>(shared_memory + ArenaBytes);
        Element* projected_totals = reinterpret_cast<Element*>(projected_weights + ProjectionSlots);
        if constexpr (ReuseWeights) {
            for (int root_id = projected_packet.root_begin; root_id < projected_packet.root_end; ++root_id) {
                const CgRoot root = CgRoots[root_id];
                for (int element = element_begin; element < NumElements && element < element_begin + ElementTile; ++element) {
                    Jet4 projected = {};
                    if (channel < OutputChannels[root.output]) {
                        for (int member_id = root.member_begin + row; member_id < root.member_end; member_id += Rows) {
                            const CgMember member = /* member_load */;
                            const int64_t index = int64_t(element) * WeightSize + PathOffsets[member.path] + channel;
                            jet4_add_scaled(projected, member.coefficient,
                                            {weights[index], rw ? rw[index] : Element(0),
                                             bw ? bw[index] : Element(0), Element(0)});
                        }
                    }
                    for (int offset = Rows / 2; offset > 0; offset >>= 1) {
                        projected.value += __shfl_down_sync(WorkerMask, projected.value, offset * ChannelColumns);
                        projected.r += __shfl_down_sync(WorkerMask, projected.r, offset * ChannelColumns);
                        projected.b += __shfl_down_sync(WorkerMask, projected.b, offset * ChannelColumns);
                    }
                    if (row == 0) {
                        const int slot = ((root_id - projected_packet.root_begin) * ElementTile + element - element_begin) * ChannelColumns + column;
                        projected_weights[slot] = projected;
                        projected_totals[2 * slot] = 0;
                        projected_totals[2 * slot + 1] = 0;
                    }
                }
            }
        }
        if constexpr (ReuseWeights) __syncthreads();
        #pragma unroll 1
        for (int node_offset = 0; node_offset < PacketNodeTile; node_offset += Rows) {
            const int64_t node_base = node_begin + node_offset;
            if (ReuseWeights && node_base >= nodes) break;
            const int64_t node = node_base + row;
            Element attr_total[ElementTile] = {};
            Element attr_r_total[ElementTile] = {};
            Element output_total[OutputComponents] = {};
            for (int component = 0; component < InputDim; ++component) {
                const int64_t index = node * InputSize + InputOffsets[component]
                                      + int64_t(channel) * InputStrides[component];
                const bool valid = node < nodes && channel < InputChannels[component];
                store_jet(values, component, worker,
                          {valid ? x[index] : Element(0),
                           valid && rx ? rx[index] : Element(0),
                           valid && bx ? bx[index] : Element(0), Element(0)});
                store_jet(totals, (CompactArena ? InputDim : 0) + component, worker, {});
            }

            for (int local_packet = ReuseWeights ? packet_id : 0;
                 local_packet < (ReuseWeights ? packet_id + 1 : PacketCount); ++local_packet) {
                const CgPacket packet = CgPackets[local_packet];
                if constexpr (!CompactArena) {
                    for (int slot = InputDim; slot < packet.slots; ++slot)
                        store_jet(totals, slot, worker, {});
                }

                for (int operation = packet.node_begin; operation < packet.node_end; ++operation) {
                    const CgNode operation_node = CgNodes[operation];
                    cg_forward(values, operation_node.left, operation_node.right,
                               operation_node.value, operation_node.block, worker);
                }

                for (int root_id = packet.root_begin; root_id < packet.root_end; ++root_id) {
                    const CgRoot root = CgRoots[root_id];
                    const bool valid = node < nodes && channel < OutputChannels[root.output];
                    if (CompactArena && CgRootZeros[root_id]) {
                        for (int component = 0; component < root.dimension; ++component)
                            store_jet(totals, CgRootTotals[root_id] + component, worker, {});
                    }
                    Jet4 mixed = {};
                    for (int element = element_begin; element < NumElements && element < element_begin + ElementTile; ++element) {
                        const int64_t attribute_index = node * NumElements + element;
                        const Jet4 attribute = {valid ? attrs[attribute_index] : Element(0),
                                               valid && ra ? ra[attribute_index] : Element(0),
                                               valid && ba ? ba[attribute_index] : Element(0), Element(0)};
                        const int slot = ((root_id - packet.root_begin) * ElementTile + element - element_begin) * ChannelColumns + column;
                        Jet4 projected = {};
                        if constexpr (ReuseWeights) projected = projected_weights[slot];
                        else if (channel < OutputChannels[root.output]) {
                            const int64_t index = int64_t(element) * WeightSize + PathOffsets[root.basis] + channel;
                            projected = {weights[index], rw ? rw[index] : Element(0),
                                         bw ? bw[index] : Element(0), Element(0)};
                        }
                        jet4_add_scaled(mixed, Element(1), jet4_multiply(attribute, projected));
                    }
                    Jet4 mixed_total = {};
                    for (int component = 0; component < root.dimension; ++component) {
                        const int output = root.output + component;
                        const int64_t index = node * OutputSize + OutputOffsets[output]
                                              + int64_t(channel) * OutputStrides[output];
                        const Jet4 polynomial = load_jet(values, root.value + component, worker);
                        const Jet4 total = {Element(0), valid && bg ? bg[index] : Element(0),
                                            Element(0), valid ? grad_out[index] : Element(0)};
                        Jet4 polynomial_total = {};
                        jet4_reverse(mixed, polynomial, total, mixed_total, polynomial_total);
                        add_jet(totals, (CompactArena ? CgRootTotals[root_id] : root.value) + component, worker, polynomial_total);
                        if (tg) output_total[output] += jet4_multiply(mixed, polynomial).rb;
                    }

                    if (ta || tra || tw || trw) for (int element = element_begin; element < NumElements && element < element_begin + ElementTile; ++element) {
                        const int64_t attribute_index = node * NumElements + element;
                        const Jet4 attribute = {valid ? attrs[attribute_index] : Element(0),
                                               valid && ra ? ra[attribute_index] : Element(0),
                                               valid && ba ? ba[attribute_index] : Element(0), Element(0)};
                        const int slot = ((root_id - packet.root_begin) * ElementTile + element - element_begin) * ChannelColumns + column;
                        Jet4 projected = {};
                        if constexpr (ReuseWeights) projected = projected_weights[slot];
                        else if ((ta || tra) && channel < OutputChannels[root.output]) {
                            const int64_t index = int64_t(element) * WeightSize + PathOffsets[root.basis] + channel;
                            projected = {weights[index], rw ? rw[index] : Element(0),
                                         bw ? bw[index] : Element(0), Element(0)};
                        }
                        Jet4 attribute_total = {}, weight_total = {};
                        jet4_reverse(attribute, projected, mixed_total, attribute_total, weight_total);
                        attr_total[element - element_begin] += attribute_total.value;
                        attr_r_total[element - element_begin] += attribute_total.r;
                        Element weight = weight_total.value, weight_r = weight_total.r;
                        for (int offset = Rows / 2; offset > 0; offset >>= 1) {
                            weight += __shfl_down_sync(WorkerMask, weight, offset * ChannelColumns);
                            weight_r += __shfl_down_sync(WorkerMask, weight_r, offset * ChannelColumns);
                        }
                        if (row == 0 && channel < OutputChannels[root.output]) {
                            if constexpr (ReuseWeights) {
                                projected_totals[2 * slot] += weight;
                                projected_totals[2 * slot + 1] += weight_r;
                            } else {
                                const int64_t index = int64_t(element) * WeightSize + PathOffsets[root.basis] + channel;
                                if (tw && weight != Element(0)) atomicAdd(tw + index, weight);
                                if (trw && weight_r != Element(0)) atomicAdd(trw + index, weight_r);
                            }
                        }
                    }
                }

                for (int operation = packet.node_end - 1; operation >= packet.node_begin; --operation) {
                    const CgNode operation_node = CgNodes[operation];
    /* backward_call */
                }
            }

            for (int component = 0; component < InputDim; ++component) {
                if (node < nodes && channel < InputChannels[component]) {
                    const int64_t index = node * InputSize + InputOffsets[component]
                                          + int64_t(channel) * InputStrides[component];
                    const Jet4 total = load_jet(totals, (CompactArena ? InputDim : 0) + component, worker);
                    if (tx) {
                        if constexpr (ReuseWeights) atomicAdd(tx + index, total.value);
                        else tx[index] = total.value;
                    }
                    if (trx) {
                        if constexpr (ReuseWeights) atomicAdd(trx + index, total.r);
                        else trx[index] = total.r;
                    }
                }
            }
            for (int component = 0; component < OutputComponents; ++component) {
                if (tg && node < nodes && channel < OutputChannels[component]) {
                    const int64_t index = node * OutputSize + OutputOffsets[component]
                                          + int64_t(channel) * OutputStrides[component];
                    if constexpr (ReuseWeights) atomicAdd(tg + index, output_total[component]);
                    else tg[index] = output_total[component];
                }
            }
            for (int element = element_begin; element < NumElements && element < element_begin + ElementTile; ++element) {
                Element attribute = attr_total[element - element_begin], attribute_r = attr_r_total[element - element_begin];
                for (int offset = ChannelColumns / 2; offset > 0; offset >>= 1) {
                    attribute += __shfl_down_sync(WorkerMask, attribute, offset, ChannelColumns);
                    attribute_r += __shfl_down_sync(WorkerMask, attribute_r, offset, ChannelColumns);
                }
                if (column == 0 && node < nodes) {
                    if (ta) atomicAdd(ta + node * NumElements + element, attribute);
                    if (tra) atomicAdd(tra + node * NumElements + element, attribute_r);
                }
            }
            if constexpr (ReuseWeights) __syncthreads();
        }
        if constexpr (ReuseWeights) {
            for (int root_id = projected_packet.root_begin; root_id < projected_packet.root_end; ++root_id) {
                const CgRoot root = CgRoots[root_id];
                if (channel < OutputChannels[root.output]) for (int element = element_begin; element < NumElements && element < element_begin + ElementTile; ++element) {
                    const int slot = ((root_id - projected_packet.root_begin) * ElementTile + element - element_begin) * ChannelColumns + column;
                    const Element weight = projected_totals[2 * slot];
                    const Element weight_r = projected_totals[2 * slot + 1];
                    for (int member_id = root.member_begin + row; member_id < root.member_end; member_id += Rows) {
                        const CgMember member = /* member_load */;
                        const int64_t index = int64_t(element) * WeightSize + PathOffsets[member.path] + channel;
                        if (tw && weight != Element(0)) atomicAdd(tw + index, member.coefficient * weight);
                        if (trw && weight_r != Element(0)) atomicAdd(trw + index, member.coefficient * weight_r);
                    }
                }
            }
            if constexpr (ReuseWeights) __syncthreads();
        }
    }
}

pybind11::tuple run(
    at::Tensor x, at::Tensor weights, at::Tensor attrs, at::Tensor grad_out,
    c10::optional<at::Tensor> rx, c10::optional<at::Tensor> rw, c10::optional<at::Tensor> ra,
    c10::optional<at::Tensor> bx, c10::optional<at::Tensor> bw, c10::optional<at::Tensor> ba,
    c10::optional<at::Tensor> bg, bool need_x, bool need_weights, bool need_attrs,
    bool need_grad_out, bool need_rx, bool need_rw, bool need_ra
) {
    c10::cuda::CUDAGuard guard(x.device());
    at::Tensor tx, tw, ta, tg, trx, trw, tra;
    if (need_x) tx = at::empty_like(x);
    if (need_weights) tw = at::empty_like(weights);
    if (need_attrs) ta = at::empty_like(attrs);
    if (need_grad_out) tg = at::empty_like(grad_out);
    if (need_rx) trx = at::empty_like(x);
    if (need_rw) trw = at::empty_like(weights);
    if (need_ra) tra = at::empty_like(attrs);
    std::vector<at::Tensor> zero_outputs;
    if constexpr (ReuseWeights) {
        if (need_x) zero_outputs.push_back(tx);
        if (need_grad_out) zero_outputs.push_back(tg);
        if (need_rx) zero_outputs.push_back(trx);
    }
    if (need_weights) zero_outputs.push_back(tw);
    if (need_attrs) zero_outputs.push_back(ta);
    if (need_rw) zero_outputs.push_back(trw);
    if (need_ra) zero_outputs.push_back(tra);
    if (zero_outputs.size() == 1) zero_outputs[0].zero_();
    else if (!zero_outputs.empty()) at::_foreach_zero_(zero_outputs);
    const auto ptr = [](const c10::optional<at::Tensor>& value) {
        return value ? value->data_ptr<Element>() : nullptr;
    };
    const int64_t nodes = x.size(0);
    if (nodes && (need_x || need_weights || need_attrs || need_grad_out || need_rx || need_rw || need_ra)) {
        const int64_t tasks = ((nodes + PacketNodeTile - 1) / PacketNodeTile) * ((Channels + ChannelColumns - 1) / ChannelColumns) * PacketTasks * ElementTiles;
        const int blocks = static_cast<int>(std::min<int64_t>(tasks, 65535));
        lunaeq_symmetric_contraction_triplebwd_/* precision */_i64<<<blocks, Workers, SharedBytes,
            at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(),
            grad_out.data_ptr<Element>(), ptr(rx), ptr(rw), ptr(ra), ptr(bx), ptr(bw), ptr(ba), ptr(bg),
            need_x ? tx.data_ptr<Element>() : nullptr, need_weights ? tw.data_ptr<Element>() : nullptr,
            need_attrs ? ta.data_ptr<Element>() : nullptr, need_grad_out ? tg.data_ptr<Element>() : nullptr,
            need_rx ? trx.data_ptr<Element>() : nullptr, need_rw ? trw.data_ptr<Element>() : nullptr,
            need_ra ? tra.data_ptr<Element>() : nullptr, nodes);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(
        need_x ? pybind11::cast(tx) : pybind11::none(), need_weights ? pybind11::cast(tw) : pybind11::none(),
        need_attrs ? pybind11::cast(ta) : pybind11::none(), need_grad_out ? pybind11::cast(tg) : pybind11::none(),
        need_rx ? pybind11::cast(trx) : pybind11::none(), need_rw ? pybind11::cast(trw) : pybind11::none(),
        need_ra ? pybind11::cast(tra) : pybind11::none());
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    if (symmetric_triple_backward::SharedBytes > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            symmetric_triple_backward::lunaeq_symmetric_contraction_triplebwd_/* precision */_i64,
            cudaFuncAttributeMaxDynamicSharedMemorySize, symmetric_triple_backward::SharedBytes));
    module.def("run", &symmetric_triple_backward::run);
}
"""


def arena_addresses(meta, plan):
    """Map logical value and gradient slots to the shared arena offsets."""
    arrays = dict(plan["arrays"])
    inputs = plan["constants"]["InputDim"]
    nodes, roots, node_totals, node_zeros, root_totals, root_zeros = [], [], [], [], [], []
    for index, packet in enumerate(arrays["CgPackets"]):
        begin, end, root_begin, root_end, _ = packet
        allocation = plan["layouts"][index]
        intervals = packet_intervals(meta, plan, packet)

        def physical(kind, base):
            return allocation.get((kind, base), base + (inputs if kind == "total" else 0))

        for offset, (left, right, value, block) in enumerate(arrays["CgNodes"][begin:end]):
            nodes.append((physical("value", left), physical("value", right), physical("value", value), block))
            node_totals.extend(physical("total", base) for base in (value, left, right))
            event = 2 * (end - begin) + root_end - root_begin - 1 - offset
            clear = sum(flag for flag, base in ((1, value), (2, left), (4, right))
                        if ("total", base) in intervals and intervals["total", base][0] == event)
            node_zeros.append(clear)
        for offset, (value, *fields) in enumerate(arrays["CgRoots"][root_begin:root_end]):
            roots.append((physical("value", value), *fields))
            root_totals.append(physical("total", value))
            interval = intervals.get(("total", value))
            root_zeros.append(int(interval is not None and interval[0] == end - begin + offset))
    arrays.update(CgNodes=nodes, CgRoots=roots, CgNodeTotals=node_totals, CgNodeZeros=node_zeros,
                  CgRootTotals=root_totals, CgRootZeros=root_zeros)
    arrays.pop("CgBlocks")
    arrays.pop("CgEdges")
    return dict(plan, arrays=arrays)
