"""Fused second derivatives of channel-aligned symmetric contractions."""

from collections import defaultdict
from string import Template
from textwrap import indent

from .utils import (
    bounded_cg_polynomials,
    polynomial_groups,
    reduced_pair_groups,
    CG_CONSTANT_BYTES,
    CG_MEMBER_LITERAL_LIMIT,
    SHARED_MEMORY_BYTES,
)


CUDA_SOURCE = Template(r"""
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
#include <cstdint>

namespace symmetric_double_backward {
$constants

$feature_loader
template<bool WriteX, int ChildDegree, int Component>
__device__ __forceinline__ void polynomial_double_backward_edge(
    Element feature, Element direction, Element child, Element child_direction,
    Element grad_value, Element grad_direction, Element* feature_sum,
    Element& polynomial, Element& polynomial_direction
) {
    if constexpr (WriteX) {
        if constexpr (ChildDegree == 0) feature_sum[Component] += grad_value * child;
        else feature_sum[Component] = fma(grad_value, child, fma(grad_direction, child_direction, feature_sum[Component]));
    }
    polynomial = fma(feature, child, polynomial);
    if constexpr (ChildDegree == 0) polynomial_direction = fma(direction, child, polynomial_direction);
    else polynomial_direction = fma(direction, child, fma(feature, child_direction, polynomial_direction));
}

template<bool WriteX, bool HasAttrs, bool HasWeightDirection$band_template, bool FullOutputs = false>
__global__ __launch_bounds__($launch_threads, $min_blocks) void $function(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* __restrict__ sx,
    Element* __restrict__ sw, Element* __restrict__ sa,
    Element* __restrict__ sg, int64_t nodes, int64_t tasks
) {
$row_constants
    if constexpr (!HasWeightDirection) rw = nullptr;
    if constexpr (!HasAttrs) ra = nullptr;
    __shared__ int active_species[$rows][NUM_ELEMENTS];
    __shared__ int active_counts[$rows];
    __shared__ Element attr_values[$rows][NUM_ELEMENTS];
    __shared__ Element attr_directions[$rows][NUM_ELEMENTS];
    __shared__ unsigned active_rows[NUM_ELEMENTS];
$sparse_shared    __shared__ $weight_vector projected_weights[WEIGHT_PATH_TILE][NUM_ELEMENTS][FUSED_CHANNEL_TILE];
    __shared__ Element products[GROUPS_PER_WORKER][2][$rows][FUSED_CHANNEL_TILE];
    const int local_channel = threadIdx.x % FUSED_CHANNEL_TILE;
    const int node_row = threadIdx.x / FUSED_CHANNEL_TILE;
$task_mapping
        const Index node = first_node + node_row;
        const bool valid = node_row < $rows && node < nodes && $valid_channel;
        if ($track_active_rows) {
            for (int element = threadIdx.x; element < NUM_ELEMENTS; element += $threads)
                active_rows[element] = 0;
            __syncthreads();
        }
${parallel_attributes}if (local_channel == 0 && node_row < $rows) {
            int count = 0;
            for (int element = 0; element < NUM_ELEMENTS; ++element) {
                const bool valid_node = node < nodes;
                const Element a = valid_node ? attrs[node * NUM_ELEMENTS + element] : Element(0);
                const Element ar = valid_node && ra ? ra[node * NUM_ELEMENTS + element] : Element(0);
                attr_values[node_row][element] = a;
                attr_directions[node_row][element] = ar;
                if (a != Element(0) || ar != Element(0)) {
                    active_species[node_row][count++] = element;
                    if ($track_active_rows) atomicOr(active_rows + element, 1u << node_row);
                }
            }
            active_counts[node_row] = count;
        }
        __syncthreads();
$sparse_scheduling$preloads
        const int active_count = valid ? active_counts[node_row] : 0;
$first_species$feature_loads
        Element output_gradient[OUTPUT_COMPONENTS];
        $gradient_loads
        Element feature_sum[WriteX ? INPUT_DIM : 1] = {};
        Element output_sum[OUTPUT_COMPONENTS] = {};
        Element attr_sum[HasAttrs ? NUM_ELEMENTS : 1] = {};
$contractions
        if constexpr (WriteX) {
            $feature_stores
        }
        if (valid) {
            if (FullOutputs || sg) {
                $output_stores
            }
        }
        if constexpr (HasAttrs) {
          if (sa) {
            for (int element = 0; element < NUM_ELEMENTS; ++element) {
                Element value = attr_sum[element];
                #pragma unroll
                for (int offset = FUSED_CHANNEL_TILE / 2; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, offset, FUSED_CHANNEL_TILE);
                if (local_channel == 0 && node_row < $rows && node < nodes)
                    atomicAdd(sa + node * NUM_ELEMENTS + element, value);
            }
          }
        }

}


void configure_cache(const void* kernel, int threads, int shared_per_sm) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                      cudaSharedmemCarveoutMaxShared));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    const size_t required = size_t(std::max(1, blocks)) * attributes.sharedSizeBytes;
    const int percent = std::min(100, int((100 * required + shared_per_sm - 1) / shared_per_sm));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, percent));
}

pybind11::tuple run(
    at::Tensor x, at::Tensor weights, at::Tensor attrs, at::Tensor grad_out,
    c10::optional<at::Tensor> rx, c10::optional<at::Tensor> rw,
    c10::optional<at::Tensor> ra, bool need_x, bool need_weights,
    bool need_attrs, bool need_grad_out
) {
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t nodes = x.size(0);
    at::Tensor sx, sw, sa, sg;
    if (need_x) sx = at::empty_like(x);
    if (need_weights) sw = at::zeros_like(weights);
    if (need_attrs) sa = at::zeros_like(attrs);
    if (need_grad_out) sg = at::empty_like(grad_out);
    const Element* rx_ptr = rx.has_value() ? rx->data_ptr<Element>() : nullptr;
    const Element* rw_ptr = rw.has_value() ? rw->data_ptr<Element>() : nullptr;
    const Element* ra_ptr = ra.has_value() ? ra->data_ptr<Element>() : nullptr;
    if (nodes > 0 && (need_x || need_weights || need_attrs || need_grad_out)) {
        auto stream = at::cuda::getCurrentCUDAStream();
$dimensions
$launches
        #define LAUNCH(WRITE_X, WRITE_A) do { \
            if (rw.has_value()) LAUNCH_RW(WRITE_X, WRITE_A, true, false); \
            else LAUNCH_RW(WRITE_X, WRITE_A, false, false); \
        } while (false)
        if (need_x && need_weights && need_grad_out && rx.has_value() && rw.has_value()
            && !need_attrs && !ra.has_value()) {
            LAUNCH_RW(true, false, true, true);
        } else if (need_x) {
            if (need_attrs || ra.has_value()) LAUNCH(true, true);
            else LAUNCH(true, false);
        } else {
            if (need_attrs || ra.has_value()) LAUNCH(false, true);
            else LAUNCH(false, false);
        }
        #undef LAUNCH
        #undef LAUNCH_RW
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(
        need_x ? pybind11::cast(sx) : pybind11::none(),
        need_weights ? pybind11::cast(sw) : pybind11::none(),
        need_attrs ? pybind11::cast(sa) : pybind11::none(),
        need_grad_out ? pybind11::cast(sg) : pybind11::none());
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    int device = 0, shared_per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
$configuration
    m.def("run", &symmetric_double_backward::run);
}
""")


SPARSE_SHARED = r"""    __shared__ bool scatter_block, inline_block;
"""


PARALLEL_ATTRIBUTES = Template(r"""        if constexpr (FUSED_CHANNEL_TILE <= 32
                      && (32 % FUSED_CHANNEL_TILE) == 0 && ($threads % 32) == 0) {
            constexpr unsigned RowMask = 0xffffffffu >> (32 - FUSED_CHANNEL_TILE);
            const int row_lane = (threadIdx.x % 32) - local_channel;
            int count = 0;
            for (int first = 0; first < NUM_ELEMENTS; first += FUSED_CHANNEL_TILE) {
                const int element = first + local_channel;
                const bool live = node_row < $rows && node < nodes && element < NUM_ELEMENTS;
                const Element a = live ? attrs[node * NUM_ELEMENTS + element] : Element(0);
                const Element ar = live && ra ? ra[node * NUM_ELEMENTS + element] : Element(0);
                const bool active = a != Element(0) || ar != Element(0);
                const unsigned mask = (__ballot_sync(0xffffffffu, active) >> row_lane) & RowMask;
                if (active) {
                    const unsigned previous = mask & ((1u << local_channel) - 1);
                    active_species[node_row][count + __popc(previous)] = element;
                    attr_values[node_row][element] = a;
                    attr_directions[node_row][element] = ar;
                    if ($track_active_rows) atomicOr(active_rows + element, 1u << node_row);
                }
                count += __popc(mask);
            }
            if (local_channel == 0 && node_row < $rows) active_counts[node_row] = count;
        } else """)


SPARSE_PROBE = Template(r"""        if constexpr (FullOutputs) {
            if (threadIdx.x < 32) {
                const int lane = threadIdx.x;
                const int count = lane < $rows ? active_counts[lane] : 0;
                const unsigned live = __ballot_sync(0xffffffffu, count != 0);
                const unsigned multiple = __ballot_sync(0xffffffffu, count > 1);
                const int species = count ? active_species[lane][0] : -1;
                const unsigned matches = __match_any_sync(0xffffffffu, species);
                const unsigned unique = __ballot_sync(0xffffffffu, count && lane == __ffs(matches) - 1);
                if (lane == 0) {
                    scatter_block = !multiple && 4 * __popc(unique) >= 3 * __popc(live);
                    inline_block = scatter_block;
                }
            }
            __syncthreads();
            if (!scatter_block) {
                for (int element = threadIdx.x; element < NUM_ELEMENTS; element += $threads)
                    active_rows[element] = 0;
                __syncthreads();
                if (node_row < $rows) {
                    for (int slot = local_channel; slot < active_counts[node_row]; slot += FUSED_CHANNEL_TILE)
                        atomicOr(active_rows + active_species[node_row][slot], 1u << node_row);
                }
                __syncthreads();
            }
        }
""")


SPARSE_SCHEDULING = Template(r"""        if ($track_block) {
            if (threadIdx.x < 32) {
                const int lane = threadIdx.x;
                int count = 0;
                for (int begin = 0; begin < NUM_ELEMENTS; begin += 32) {
                    const int element = begin + lane;
                    const bool active = element < NUM_ELEMENTS && active_rows[element] != 0;
                    const unsigned mask = __ballot_sync(0xffffffffu, active);
                    count += __popc(mask);
                }
                const int row_count = lane < $rows ? active_counts[lane] : 0;
                const unsigned multiple = __ballot_sync(0xffffffffu, row_count > 1);
                const unsigned live_rows = __ballot_sync(0xffffffffu, row_count != 0);
                int active_entries = row_count;
                #pragma unroll
                for (int offset = 16; offset; offset >>= 1)
                    active_entries += __shfl_down_sync(0xffffffffu, active_entries, offset);
                if (lane == 0) {
                    inline_block = active_entries <= 2 * count;
                    scatter_block = !ra && !multiple && 4 * count >= 3 * __popc(live_rows);
                }
            }
            __syncthreads();
        }
        const bool scatter_weights = sw && scatter_block;
        const bool load_inline = (sw || WriteX || sg) && inline_block;
""")


def consumer_outputs(path, blocks, sign):
    outer = defaultdict(list)
    if path["degree"] == 2:
        for component in range(path["output_components"]):
            outer[path["output_component"] + component].append((None, component, sign))
    elif path["degree"] == 3:
        tensor = next(block["tensor"] for block in blocks if block["tree"] == path["tree"])
        start = path["tree"]["args"][1]["start"]
        for output, inner, right in tensor.nonzero().tolist():
            outer[path["output_component"] + output].append(
                (start + right, inner, sign * tensor[output, inner, right].item()))
    else:
        raise ValueError("binary pair consumers must have degree two or three")
    return outer


def contraction_groups(meta):
    payload = meta.get("binary")
    if payload is None or meta.get("correlation", 3) > 3:
        groups, paths = [], []
        for terms, members in polynomial_groups(meta):
            if not terms:
                continue
            basis = len(paths)
            outer = {output: [(None, output, 1.)] for output, inputs, coefficient in terms}
            groups.append((terms, [(basis, outer)]))
            paths.append(tuple(members))
        return groups, tuple(paths)

    selected = {item["basis_path"]: item for item in payload["selected_paths"]}
    blocks = tuple(payload["local_cg_blocks"].values())
    groups, weight_paths = [], list(payload["compute_weight_paths"])
    for pair in payload["pair_groups"]:
        if not pair["terms"]:
            continue
        scale = max(abs(value) for output, inputs, value in pair["terms"])
        terms = tuple((output, tuple(inputs), value / scale) for output, inputs, value in pair["terms"])
        consumers = []
        for basis, original, key, sign in pair["consumers"]:
            outer = consumer_outputs(selected[basis], blocks, sign)
            outer_scale = max(abs(value) for items in outer.values() for raw, inner, value in items)
            weight_paths[basis] = tuple((path, coefficient * scale * outer_scale)
                                        for path, coefficient in payload["compute_weight_paths"][basis])
            consumers.append((basis, {output: [(raw, inner, value / outer_scale) for raw, inner, value in items]
                                      for output, items in outer.items()}))
        groups.append((terms, consumers))
    for basis, item in selected.items():
        if item["degree"] == 1:
            groups.append(((), [(basis, {item["output_component"] + component:
                                         [(item["tree"]["start"] + component, None, 1.)]
                                         for component in range(item["output_components"])})]))
    return groups, tuple(weight_paths)


class ContractionLayout:
    def __init__(self, meta, limits, banded, begin, end):
        self.meta, self.limits = meta, limits
        self.compact = meta.get("input_layout", "channels") != "channels"
        self.banded, self.begin, self.end = banded, begin, end

    def guard(self, value, limit):
        if not self.banded:
            return f"{value} < {limit}"
        count = self.meta["channels"] if limit == "CHANNELS" else limit
        return f"(BandEnd <= {count} || (BandBegin < {count} && {value} < {count}))"

    def basis_guard(self, basis):
        if self.limits is None:
            return None
        limit = self.limits[basis]
        return "true" if self.banded and limit >= self.end else self.guard("channel", limit)

    def load(self, pointer, path):
        if not self.compact:
            return f"{pointer}[base + Index({path}) * CHANNELS]"
        address = f"{pointer}[base + Index({self.meta['path_offsets'][path]})]"
        limit = self.meta["path_channels"][path]
        if self.banded and limit >= self.end:
            return address
        return f"({self.guard('channel', limit)} ? {address} : Element(0))"


def polynomial_code(terms, value, direction, grad_value, grad_direction, total, depth=0,
                     combine_squares=False, combine_powers=False):
    if not terms[0][0]:
        coefficient = sum(coefficient for inputs, coefficient in terms)
        return [f"{value} = Element({coefficient:.17g});", f"{direction} = Element(0);"]
    children = defaultdict(list)
    for inputs, coefficient in terms:
        children[inputs[0]].append((inputs[1:], coefficient))
    code = []
    child, child_rx = f"child{depth}", f"child_rx{depth}"
    child_a, child_ar = f"child_a{depth}", f"child_ar{depth}"
    for component, remaining in sorted(children.items()):
        degree = len(remaining[0][0])
        if (combine_powers and degree >= 1 and (degree > 1 or depth > 0)
                and all(indices == (component,) * degree for indices, _ in remaining)):
            coefficient = sum(coefficient for _, coefficient in remaining)
            power = " * ".join([f"Element({coefficient:.17g})"] + [f"feature[{component}]"] * (degree - 1))
            code += ["{", f"const Element power = {power};",
                     f"const Element weighted_feature = power * feature[{component}];",
                     f"{value} = fma(weighted_feature, feature[{component}], {value});",
                     f"{direction} = fma(Element({degree + 1}) * weighted_feature, direction[{component}], {direction});",
                     "if constexpr (WriteX) {",
                     f"const Element gradient = fma({grad_value}, feature[{component}], Element({degree}) * ({grad_direction}) * direction[{component}]);",
                     f"{total}[{component}] = fma(Element({degree + 1}) * power, gradient, {total}[{component}]);",
                     "}", "}"]
            continue
        if depth == 0 and combine_squares and all(inputs == (component,) for inputs, _ in remaining):
            coefficient = sum(coefficient for inputs, coefficient in remaining)
            code += ["{",
                     f"const Element weighted_feature = Element({coefficient:.17g}) * feature[{component}];",
                     f"{value} = fma(weighted_feature, feature[{component}], {value});",
                     f"{direction} = fma(Element(2) * weighted_feature, direction[{component}], {direction});",
                     "if constexpr (WriteX) {",
                     f"const Element gradient = fma({grad_value}, feature[{component}], {grad_direction} * direction[{component}]);",
                     f"{total}[{component}] = fma(Element({2 * coefficient:.17g}), gradient, {total}[{component}]);",
                     "}", "}"]
            continue
        if degree == 0:
            coefficient = sum(coefficient for inputs, coefficient in remaining)
            code.append(f"polynomial_double_backward_edge<WriteX, 0, {component}>(feature[{component}], direction[{component}], Element({coefficient:.17g}), Element(0), {grad_value}, {grad_direction}, {total}, {value}, {direction});")
            continue
        code += ["{", f"    Element {child} = 0, {child_rx} = 0;",
                 f"    Element {child_a} = 0, {child_ar} = 0;",
                 "    if constexpr (WriteX) {",
                 f"        {child_a} = fma({grad_value}, feature[{component}], ({grad_direction}) * direction[{component}]);",
                 f"        {child_ar} = ({grad_direction}) * feature[{component}];", "    }"]
        code.extend("    " + line for line in polynomial_code(
            remaining, child, child_rx, child_a, child_ar, total, depth + 1, combine_powers=combine_powers))
        code += [f"    polynomial_double_backward_edge<WriteX, {degree}, {component}>(feature[{component}], direction[{component}], {child}, {child_rx}, {grad_value}, {grad_direction}, {total}, {value}, {direction});", "}"]
    return code


def component_code(terms, consumers, guards, scalar):
    count = len(consumers)
    code = [f"Element stream_product[{count}] = {{}}, stream_direction[{count}] = {{}};"]
    components = sorted({component for component, inputs, value in terms})
    bits, constraint = (64, "d") if scalar == "double" else (32, "f")
    for component in components:
        inner = [(tuple(inputs), value) for output, inputs, value in terms if output == component]
        channels = []
        for basis, outputs in consumers:
            by_raw = defaultdict(list)
            for output, items in outputs.items():
                for raw, selected, coefficient in items:
                    if selected == component:
                        by_raw[raw].append((output, coefficient))
            channels.append(by_raw)
        inputs = sorted({value for indices, coefficient in inner for value in indices}
                        | {raw for by_raw in channels for raw in by_raw if raw is not None})
        outputs = sorted({output for by_raw in channels for items in by_raw.values() for output, _ in items})
        code += ["{", "    Element pair_gradient = 0, pair_gradient_direction = 0;"]
        for value, indices in (("feature", inputs), ("direction", inputs), ("output_gradient", outputs)):
            for index in indices:
                field = f"{value}[{index}]"
                code.append(f'asm volatile("mov.b{bits} %0, %1;" : "={constraint}"({field}) : "{constraint}"({field}));')
        for slot, by_raw in enumerate(channels):
            if not by_raw:
                continue
            code.append(f"    Element seed{slot} = 0, seed_direction{slot} = 0;")
            if guards[slot]:
                code.append(f"    if ({guards[slot]}) {{")
            for raw, items in sorted(by_raw.items(), key=lambda item: -1 if item[0] is None else item[0]):
                code += ["    {", "        Element gradient = 0;"]
                for output, coefficient in items:
                    code.append(f"        gradient = fma(Element({coefficient:.17g}), output_gradient[{output}], gradient);")
                if raw is None:
                    code.append(f"        seed{slot} += gradient;")
                else:
                    code += [f"        seed{slot} = fma(gradient, feature[{raw}], seed{slot});",
                             f"        seed_direction{slot} = fma(gradient, direction[{raw}], seed_direction{slot});"]
                code.append("    }")
            code += ["    if constexpr (WriteX) {",
                     f"        pair_gradient = fma(mixed_direction[{slot}], seed{slot}, fma(mixed_weight[{slot}], seed_direction{slot}, pair_gradient));",
                     f"        pair_gradient_direction = fma(mixed_weight[{slot}], seed{slot}, pair_gradient_direction);", "    }"]
            if guards[slot]:
                code.append("    }")
        children = defaultdict(list)
        for indices, coefficient in inner:
            children[indices[0]].append(indices[1:])
        squares = sum(all(indices == (index,) for indices in values) for index, values in children.items())
        code += ["    Element pair_value = 0, pair_direction = 0;"]
        code += ["    " + line for line in polynomial_code(
            inner, "pair_value", "pair_direction", "pair_gradient", "pair_gradient_direction",
            "feature_sum", combine_squares=squares >= 2, combine_powers=scalar == "double")]
        for slot, by_raw in enumerate(channels):
            if not by_raw:
                continue
            if guards[slot]:
                code.append(f"    if ({guards[slot]}) {{")
            code += ["    {", f"        if constexpr (HasAttrs) stream_product[{slot}] = fma(seed{slot}, pair_value, stream_product[{slot}]);",
                     f"        stream_direction[{slot}] = fma(seed{slot}, pair_direction, fma(seed_direction{slot}, pair_value, stream_direction[{slot}]));",
                     f"        const Element direct_value = fma(mixed_direction[{slot}], pair_value, mixed_weight[{slot}] * pair_direction);",
                     f"        const Element direct_direction = mixed_weight[{slot}] * pair_value;"]
            for raw, items in sorted(by_raw.items(), key=lambda item: -1 if item[0] is None else item[0]):
                code.append("        {")
                if raw is None:
                    code.append("            if (FullOutputs || sg) {")
                    for output, coefficient in items:
                        code.append(f"                output_sum[{output}] = fma(Element({coefficient:.17g}), direct_value, output_sum[{output}]);")
                    code.append("            }")
                else:
                    code += ["            if (FullOutputs || sg) {",
                             f"                const Element contribution = fma(direct_value, feature[{raw}], direct_direction * direction[{raw}]);"]
                    for output, coefficient in items:
                        code.append(f"                output_sum[{output}] = fma(Element({coefficient:.17g}), contribution, output_sum[{output}]);")
                    code += ["            }", "            if constexpr (WriteX) {", "                Element gradient = 0;"]
                    for output, coefficient in items:
                        code.append(f"                gradient = fma(Element({coefficient:.17g}), output_gradient[{output}], gradient);")
                    code += [f"                feature_sum[{raw}] = fma(direct_value, gradient, feature_sum[{raw}]);", "            }"]
                code.append("        }")
            code.append("    }")
            if guards[slot]:
                code.append("    }")
        code.append("}")
    return code


def projected_weights_code(bases, weight_paths, layout, threads, vector):
    rows = (len(bases) + 3) // 4
    code = [f"for (int item = threadIdx.x; item < {rows} * NUM_ELEMENTS * FUSED_CHANNEL_TILE; item += {threads}) {{",
            "    const int local_c = item % FUSED_CHANNEL_TILE;",
            "    const int channel = channel_begin + local_c;",
            "    const int element = (item / FUSED_CHANNEL_TILE) % NUM_ELEMENTS;",
            "    const int group = 4 * (item / (NUM_ELEMENTS * FUSED_CHANNEL_TILE));",
            "    const Index base = " + ("Index(element) * WEIGHT_SIZE + channel;" if layout.compact
                                       else "Index(element) * NUM_PATHS * CHANNELS + channel;")]
    for slot in range(4):
        code.append(f"Element weight{slot} = 0, weight_direction{slot} = 0;")
    code += [f"    if ({layout.guard('channel', 'CHANNELS')} && active_rows[element]) {{", "        switch (group) {"]
    for first in range(0, len(bases), 4):
        code.append(f"case {first}: {{")
        for slot, basis in enumerate(bases[first:first + 4]):
            limit = layout.limits[basis] if layout.limits is not None else layout.meta["channels"]
            guard = "true" if layout.banded and limit >= layout.end else layout.guard("channel", limit)
            code += ["{", f"if ({guard}) {{"]
            for path, coefficient in weight_paths[basis]:
                code += [f"weight{slot} = fma(Element({coefficient:.17g}), {layout.load('weights', path)}, weight{slot});",
                         f"if constexpr (HasWeightDirection) weight_direction{slot} = fma(Element({coefficient:.17g}), {layout.load('rw', path)}, weight_direction{slot});"]
            code += ["}", "}"]
        code += ["break;", "}"]
    code += ["        }", "    }"]
    for slot in range(4):
        code += [f"if (group + {slot} < {len(bases)}) {{",
                 f"projected_weights[group + {slot}][element][local_c] = make_{vector}(weight{slot}, weight_direction{slot});", "}"]
    code += ["}", "__syncthreads();"]
    return code


def attribute_code(basis, weight_paths, layout):
    code = ["if constexpr (HasAttrs) {", "if (sa && valid) {", "#pragma unroll 4",
            "for (int element = 0; element < NUM_ELEMENTS; ++element) {",
            "const Index base = " + ("Index(element) * WEIGHT_SIZE + channel;" if layout.compact
                                     else "Index(element) * NUM_PATHS * CHANNELS + channel;"),
            "Element weight = 0, weight_direction = 0;"]
    for path, coefficient in weight_paths[basis]:
        code += [f"weight = fma(Element({coefficient:.17g}), {layout.load('weights', path)}, weight);",
                 f"if constexpr (HasWeightDirection) weight_direction = fma(Element({coefficient:.17g}), {layout.load('rw', path)}, weight_direction);"]
    return code + ["attr_sum[element] = fma(weight, gradient_direction, fma(weight_direction, gradient_product, attr_sum[element]));",
                   "}", "}", "}"]


def weight_reduction_code(chunk, weight_paths, layout, threads, inline_weights):
    active = "CHANNELS"
    if layout.limits is not None and not (layout.banded and min(layout.limits[basis] for basis in chunk) >= layout.end):
        active = str(layout.limits[chunk[-1]])
        for slot in range(len(chunk) - 2, -1, -1):
            active = f"(slot == {slot} ? {layout.limits[chunk[slot]]} : {active})"
    guard = f"c < {active}" if active.startswith("(") else layout.guard("c", active if active == "CHANNELS" else int(active))
    work = f"NUM_ELEMENTS * {len(chunk)} * FUSED_CHANNEL_TILE"
    index = ["const int slot = item / (FUSED_CHANNEL_TILE * NUM_ELEMENTS);",
             "const int element = (item / FUSED_CHANNEL_TILE) % NUM_ELEMENTS;"]
    reduce_weights = "(FullOutputs || sw) && !scatter_weights" if inline_weights else "FullOutputs || sw"
    code = [f"if ({reduce_weights}) {{", "__syncthreads();",
            f"for (int item = threadIdx.x; item < {work}; item += {threads}) {{",
            "const int c_local = item % FUSED_CHANNEL_TILE;", *index,
            "const int c = channel_begin + c_local;", f"if ({guard}) {{",
            "unsigned remaining = active_rows[element];", "Element total = 0;",
            "#pragma unroll 4", "while (remaining) {", "const int row = __ffs(remaining) - 1;",
            "total = fma(attr_values[row][element], products[slot][1][row][c_local], total);",
            "if (ra) total = fma(attr_directions[row][element], products[slot][0][row][c_local], total);",
            "remaining &= remaining - 1;", "}", "if (total != Element(0)) {", "switch (slot) {"]
    for slot, basis in enumerate(chunk):
        code.append(f"case {slot}: {{")
        for path, coefficient in weight_paths[basis]:
            if layout.compact:
                limit = layout.meta["path_channels"][path]
                limit = "CHANNELS" if layout.banded and limit >= layout.end else limit
                code += [f"if ({layout.guard('c', limit)}) {{",
                         f"const Index index = Index(element) * WEIGHT_SIZE + Index({layout.meta['path_offsets'][path]}) + c;",
                         f"atomicAdd(sw + index, Element({coefficient:.17g}) * total);", "}"]
            else:
                code += [f"const Index index{path} = (Index(element) * NUM_PATHS + {path}) * CHANNELS + c;",
                         f"atomicAdd(sw + index{path}, Element({coefficient:.17g}) * total);"]
        code += ["break;", "}"]
    return code + ["}", "}", "}", "}", "__syncthreads();", "}"]


def pair_code(groups, weight_paths, layout, rows, threads, path_tile, scalar, inline_weights, reuse_species):
    order = [basis for terms, consumers in groups for basis, outputs in consumers]
    offsets = {basis: index for index, basis in enumerate(order)}
    tiles = [order[first:first + path_tile] for first in range(0, len(order), path_tile)]
    vector = scalar + "2"
    current_tile = 0
    code = []
    bits, constraint = (64, "d") if scalar == "double" else (32, "f")
    for pair, (terms, consumers) in enumerate(groups):
        code.append(f"case {pair}: {{")
        needed = sorted({index for other_terms, others in groups[pair:] for output, inputs, value in other_terms for index in inputs}
                        | {raw for other_terms, others in groups[pair:] for basis, outputs in others
                           for items in outputs.values() for raw, inner, value in items if raw is not None})
        for component in needed:
            for value in ("feature", "direction"):
                code.append(f'asm volatile("mov.b{bits} %0, %1;" : "={constraint}"({value}[{component}]) : "{constraint}"({value}[{component}]));')
        for output in range(len(layout.meta["output_offsets"])):
            code.append(f'asm volatile("mov.b{bits} %0, %1;" : "={constraint}"(output_gradient[{output}]) : "{constraint}"(output_gradient[{output}]));')
        count = len(consumers)
        code += [f"Element mixed_weight[{count}] = {{}}, mixed_direction[{count}] = {{}};", "if (WriteX || sg) {"]
        first = 0
        if inline_weights:
            base = "WEIGHT_SIZE" if layout.compact else "(NUM_PATHS * CHANNELS)"
            mix = []
            for slot, (basis, _) in enumerate(consumers):
                mix += ["{", "Element weight = 0, weight_direction = 0;"]
                for path, coefficient in weight_paths[basis]:
                    mix += [f"weight = fma(Element({coefficient:.17g}), {layout.load('weights', path)}, weight);",
                            f"if constexpr (HasWeightDirection) weight_direction = fma(Element({coefficient:.17g}), {layout.load('rw', path)}, weight_direction);"]
                mix += [f"mixed_weight[{slot}] = fma(attribute, weight, mixed_weight[{slot}]);",
                        f"if constexpr (HasWeightDirection) mixed_direction[{slot}] = fma(attribute, weight_direction, mixed_direction[{slot}]);",
                        f"if (ra) mixed_direction[{slot}] = fma(attr_directions[node_row][element], weight, mixed_direction[{slot}]);",
                        "}"]
            code += ["if (load_inline) {"]
            if reuse_species:
                code += ["if constexpr (FullOutputs) {", "if (active_count) {", "int element = first_species;",
                         "Element attribute = first_attribute;", "Index base = first_weight_base;",
                         "#pragma unroll 1", "for (int active = 0;;) {", *mix,
                         "if (++active == active_count) break;", "element = active_species[node_row][active];",
                         "attribute = attr_values[node_row][element];", f"base = Index(element) * {base} + channel;",
                         "}", "}", "} else {"]
            code += ["#pragma unroll 1", "for (int active = 0; active < active_count; ++active) {",
                     "const int element = active_species[node_row][active];",
                     "const Element attribute = attr_values[node_row][element];",
                     f"const Index base = Index(element) * {base} + channel;", *mix, "}"]
            if reuse_species:
                code += ["}"]
            code += ["} else {"]
        while first < count:
            tile = offsets[consumers[first][0]] // path_tile
            last = first
            while last < count and offsets[consumers[last][0]] // path_tile == tile:
                last += 1
            if tile != current_tile:
                code.append("__syncthreads();")
                code += projected_weights_code(tiles[tile], weight_paths, layout, threads, vector)
                current_tile = tile
            code += ["#pragma unroll 4", "for (int active = 0; active < active_count; ++active) {",
                     "const int element = active_species[node_row][active];",
                     "const Element attribute = attr_values[node_row][element];"]
            for slot in range(first, last):
                cache_slot = offsets[consumers[slot][0]] % path_tile
                code += ["{", f"const {vector} projected = projected_weights[{cache_slot}][element][local_channel];",
                         "const Element weight = projected.x;",
                         f"mixed_weight[{slot}] = fma(attribute, weight, mixed_weight[{slot}]);",
                         f"if constexpr (HasWeightDirection) mixed_direction[{slot}] = fma(attribute, projected.y, mixed_direction[{slot}]);",
                         f"if (ra) mixed_direction[{slot}] = fma(attr_directions[node_row][element], weight, mixed_direction[{slot}]);", "}"]
            code.append("}")
            first = last
        if inline_weights:
            code.append("}")
        code.append("}")
        guards = [layout.basis_guard(basis) for basis, outputs in consumers]
        if terms:
            code += component_code(terms, consumers, guards, scalar)
        for slot, (basis, outputs) in enumerate(consumers):
            code.append("{")
            if terms:
                code += [f"Element gradient_product = stream_product[{slot}];",
                         f"Element gradient_direction = stream_direction[{slot}];"]
            else:
                code += [f"const Element weighted = mixed_weight[{slot}];",
                         f"const Element weighted_direction = mixed_direction[{slot}];",
                         "Element gradient_product = 0, gradient_direction = 0;"]
            if guards[slot]:
                code.append(f"if ({guards[slot]}) {{")
            if not terms:
                for output, items in sorted(outputs.items()):
                    code += ["{", f"const Element gradient = output_gradient[{output}];",
                             "const Element grad_value = gradient * weighted_direction;",
                             "const Element grad_direction = gradient * weighted;",
                             "Element polynomial = 0, polynomial_direction = 0;"]
                    by_raw = defaultdict(list)
                    for raw, inner, coefficient in items:
                        by_raw[raw].append((inner, coefficient))
                    for raw, inner_terms in sorted(by_raw.items(), key=lambda item: -1 if item[0] is None else item[0]):
                        coefficient = inner_terms[0][1]
                        code += [f"polynomial = fma(Element({coefficient:.17g}), feature[{raw}], polynomial);",
                                 f"polynomial_direction = fma(Element({coefficient:.17g}), direction[{raw}], polynomial_direction);",
                                 f"if constexpr (WriteX) feature_sum[{raw}] = fma(Element({coefficient:.17g}), grad_value, feature_sum[{raw}]);"]
                    code += [f"if (FullOutputs || sg) output_sum[{output}] = fma(weighted, polynomial_direction, fma(weighted_direction, polynomial, output_sum[{output}]));",
                             "if (HasAttrs || ra) gradient_product = fma(gradient, polynomial, gradient_product);",
                             "gradient_direction = fma(gradient, polynomial_direction, gradient_direction);", "}"]
            code += attribute_code(basis, weight_paths, layout)
            if guards[slot]:
                code.append("}")
            if inline_weights:
                code += ["if (scatter_weights && active_count) {"]
                if reuse_species:
                    code += ["const int element = FullOutputs ? first_species : active_species[node_row][0];",
                             "const Element attribute = FullOutputs ? first_attribute : attr_values[node_row][element];",
                             "const Element total = attribute * gradient_direction;"]
                else:
                    code += ["const int element = active_species[node_row][0];",
                             "const Element total = attr_values[node_row][element] * gradient_direction;"]
                code += ["if (total != Element(0)) {"]
                for path, coefficient in weight_paths[basis]:
                    if layout.compact:
                        limit = layout.meta["path_channels"][path]
                        offset = layout.meta['path_offsets'][path]
                        address = (f"(FullOutputs ? first_weight_base : Index(element) * WEIGHT_SIZE + channel) + Index({offset})" if reuse_species else
                                   f"Index(element) * WEIGHT_SIZE + Index({offset}) + channel")
                        code += [f"if ({layout.guard('channel', limit)})",
                                 f"atomicAdd(sw + {address}, Element({coefficient:.17g}) * total);"]
                    else:
                        address = (f"(FullOutputs ? first_weight_base : Index(element) * NUM_PATHS * CHANNELS + channel) + Index({path}) * CHANNELS" if reuse_species else
                                   f"(Index(element) * NUM_PATHS + {path}) * CHANNELS + channel")
                        code += [f"atomicAdd(sw + {address}, Element({coefficient:.17g}) * total);"]
                code += ["}", "}"]
            needs_reduction = " && !scatter_weights" if inline_weights else ""
            code += [f"if ((FullOutputs || sw){needs_reduction} && node_row < {rows}) {{",
                     f"products[{offsets[basis] % 4}][0][node_row][local_channel] = gradient_product;",
                     f"products[{offsets[basis] % 4}][1][node_row][local_channel] = gradient_direction;", "}", "}"]
            consumer = offsets[basis]
            if consumer % 4 == 3 or consumer == len(order) - 1:
                code += weight_reduction_code(order[(consumer // 4) * 4:consumer + 1], weight_paths, layout,
                                              threads, inline_weights)
        code += ["break;", "}"]
    needs_projection = "(WriteX || sg) && !load_inline" if inline_weights else "WriteX || sg"
    preload = ([f"if ({needs_projection}) {{"] +
               projected_weights_code(tiles[0], weight_paths, layout, threads, vector) + ["}"] if tiles else [])
    body = ["auto contraction_pair = [&](int pair) __attribute__((always_inline)) {", "switch (pair) {"]
    body += code + ["}", "};", "if constexpr (HasAttrs) {", "#pragma unroll 1",
                    f"for (int pair = 0; pair < {len(groups)}; ++pair) contraction_pair(pair);",
                    "} else {"]
    body += [f"contraction_pair({pair});" for pair in range(len(groups))]
    body.append("}")
    return "\n".join(preload), "\n".join(body)


def make_double_backward_source(meta, dtype="float32"):
    """Use reduced pairs or bounded polynomials before falling back to a CG plan."""
    reduced_pairs = "cg_dag" in meta and meta.get("use_reduced_cg") and meta["correlation"] <= 3
    if "cg_dag" in meta and not reduced_pairs:
        polynomial_meta = bounded_cg_polynomials(meta)
        if polynomial_meta is None:
            return make_cg_double_backward_source(meta, dtype)
        meta = polynomial_meta
    scalar = "double" if str(dtype) in ("float64", "torch.float64", "double") else "float"
    item_bytes = 8 if scalar == "double" else 4
    if reduced_pairs:
        groups, weight_paths, support = reduced_pair_groups(meta)
    else:
        groups, weight_paths = contraction_groups(meta)
    compact = meta.get("input_layout", "channels") != "channels"
    limits = None
    if compact:
        if reduced_pairs:
            computed = tuple(support[basis] for basis in range(len(weight_paths)))
        elif not meta["terms"] and "polynomials" in meta:
            computed = tuple(max((min(meta["output_channels"][output],
                                       *(meta["input_channels"][index] for index in inputs))
                                  for output, inputs, coefficient in terms), default=0)
                             for terms, consumers in groups)
        else:
            support = [0] * meta["num_paths"]
            for output, path, inputs, coefficient in meta["terms"]:
                support[path] = max(support[path], min(meta["output_channels"][output],
                                                      *(meta["input_channels"][index] for index in inputs)))
            computed = tuple(max((support[path] for path, coefficient in members), default=0) for members in weight_paths)
        if not all(limit == meta["channels"] for limit in computed):
            limits = computed
    channels = meta.get("channel_tile", 16)
    boundaries = sorted({0, meta["channels"], *(min(meta["channels"], (limit + channels - 1) // channels * channels)
                                                  for limit in (limits or ()) if limit)})
    banded = len(boundaries) > 2
    order = [basis for terms, consumers in groups for basis, outputs in consumers]
    shared_limit = SHARED_MEMORY_BYTES
    path_bytes = 2 * meta["num_elements"] * channels * item_bytes
    for original_rows in (16, 8, 4, 2, 1):
        required = (original_rows * meta["num_elements"] * (4 + 2 * item_bytes) + original_rows * 4
                    + meta["num_elements"] * 4 + 8 * original_rows * channels * item_bytes)
        if required + path_bytes <= shared_limit:
            break
    path_capacity = max(1, (shared_limit - required) // path_bytes)
    path_tile = min(max(1, len(order)), path_capacity)
    sparse_bytes = 8
    inline_weights = (scalar == "float" and path_capacity < 4
                      and required + path_bytes + sparse_bytes <= shared_limit)
    if inline_weights:
        path_tile = min(path_tile, (shared_limit - required - sparse_bytes) // path_bytes)
    max_consumers = max((len(consumers) for terms, consumers in groups), default=1)
    registers = 3 * meta["input_dim"] + 1 + 6 * max_consumers + 2 * len(meta["output_offsets"]) + 8 + 24
    register_threads = max(32, (32768 // (32 * registers)) * 32)
    node_rows = min(original_rows, max(1, register_threads // channels))
    min_blocks = 3 if inline_weights else 2
    fixed_rows = node_rows
    if banded:
        fixed = 2 * path_tile * meta["num_elements"] * channels * item_bytes + 4 * meta["num_elements"]
        if inline_weights:
            fixed += sparse_bytes
        per_row = meta["num_elements"] * (4 + 2 * item_bytes) + 4 + 8 * channels * item_bytes
        fixed_rows = max(node_rows, min(16, (shared_limit - fixed) // per_row))
    separate_rows = fixed_rows > node_rows
    # The fast scatter keeps the first species live for the whole tile.
    estimated_blocks = round((2 * 32768) / (registers * max(32, fixed_rows * channels)))
    reuse_species = inline_weights and estimated_blocks >= min_blocks
    warps_per_block = (max(32, node_rows * channels) + 31) // 32
    min_blocks = min(min_blocks, max(2, estimated_blocks, (12 + warps_per_block - 1) // warps_per_block))
    rows, threads = ("node_rows", "node_threads") if separate_rows else ("NODE_TILE", "FUSED_THREADS")
    function = "symmetric_contraction_doublebwd_" + ("fp64" if scalar == "double" else "fp32") + "_i64"
    vector_width = next(width for width in ((2, 1) if scalar == "double" else (4, 2, 1)) if meta["input_dim"] % width == 0)
    sizes = dict(CHANNELS=meta["channels"], INPUT_DIM=meta["input_dim"], OUTPUT_DIM=meta["output_dim"],
                 NUM_ELEMENTS=meta["num_elements"], NUM_PATHS=meta["num_paths"],
                 OUTPUT_COMPONENTS=len(meta["output_offsets"]), NODE_TILE=node_rows, FUSED_CHANNEL_TILE=channels,
                 FUSED_THREADS=max(32, node_rows * channels), GROUPS_PER_WORKER=4,
                 CHANNEL_TILES=(meta["channels"] + channels - 1) // channels, WEIGHT_PATH_TILE=path_tile)
    constants = [f"using Element = {scalar};", "using Index = int64_t;",
                 f"using FeatureVector = {scalar + str(vector_width) if vector_width > 1 else scalar};",
                 f"constexpr int FEATURE_VECTOR_WIDTH = {vector_width};"]
    for name, value in sizes.items():
        constants.append(f"constexpr int {name} = {value};")
        if name == "FUSED_THREADS" and separate_rows:
            constants += [f"constexpr int NODE_TILE_NO_ATTRS = {fixed_rows};",
                          f"constexpr int THREADS_NO_ATTRS = {max(32, fixed_rows * channels)};"]
    if compact:
        constants += ["", f"constexpr int INPUT_SIZE = {meta['input_size']};",
                      f"constexpr int WEIGHT_SIZE = {meta['weight_numel']};"]
        loader = ["__device__ __forceinline__ void load_feature(",
                  "const Element* input, Index node, int channel, bool valid, Element* values", ") {"]
        for component, (offset, stride, count) in enumerate(zip(meta["input_offsets"], meta["input_strides"], meta["input_channels"])):
            loader.append(f"values[{component}] = valid && input && channel < {count} ? input[node * INPUT_SIZE + Index({offset}) + Index(channel) * {stride}] : Element(0);")
        loader += ["}", ""]
        feature_loads = ["Element feature[INPUT_DIM], direction[INPUT_DIM];",
                         "load_feature(x, node, channel, valid, feature);",
                         "load_feature(rx, node, channel, valid, direction);"]
    else:
        loader = ["__device__ __forceinline__ void load_feature(const Element* input, Element* values) {",
                  "if (!input) {", "#pragma unroll",
                  "for (int i = 0; i < INPUT_DIM; ++i) values[i] = Element(0);", "return;", "}",
                  "if ((reinterpret_cast<uintptr_t>(input) & (sizeof(FeatureVector) - 1)) == 0) {",
                  "#pragma unroll", "for (int i = 0; i < INPUT_DIM / FEATURE_VECTOR_WIDTH; ++i) {",
                  "const FeatureVector value = reinterpret_cast<const FeatureVector*>(input)[i];"]
        loader += [f"values[i * FEATURE_VECTOR_WIDTH + {index}] = value{'.' + component if vector_width > 1 else ''};"
                   for index, component in enumerate("xyzw"[:vector_width])]
        loader += ["}", "return;", "}", "#pragma unroll",
                   "for (int i = 0; i < INPUT_DIM; ++i) values[i] = input[i];", "}", ""]
        feature_loads = ["const Index input_offset = (node * CHANNELS + channel) * INPUT_DIM;",
                         "Element feature[INPUT_DIM], direction[INPUT_DIM];",
                         "load_feature(valid ? x + input_offset : nullptr, feature);",
                         "load_feature(valid && (FullOutputs || rx) ? rx + input_offset : nullptr, direction);"]
    layout = ContractionLayout(meta, limits, banded, 0, meta["channels"])
    gradient_loads, stores = [], []
    for output, (offset, stride) in enumerate(zip(meta["output_offsets"], meta["output_strides"])):
        address = f"node * OUTPUT_DIM + {offset} + Index(channel) * {stride}"
        valid = "valid" + (f" && {layout.guard('channel', meta['output_channels'][output])}" if compact else "")
        gradient_loads.append(f"output_gradient[{output}] = {valid} ? grad_out[{address}] : Element(0);")
        guard = f"if ({layout.guard('channel', meta['output_channels'][output])}) " if compact else ""
        stores.append(guard + f"sg[{address}] = output_sum[{output}];")
    feature_stores = []
    if compact:
        for component, (offset, stride, count) in enumerate(zip(meta["input_offsets"], meta["input_strides"], meta["input_channels"])):
            feature_stores.append(f"if (node_row < {rows} && node < nodes && {layout.guard('channel', count)}) sx[node * INPUT_SIZE + Index({offset}) + Index(channel) * {stride}] = feature_sum[{component}];")
    else:
        first = 0
        while first < meta["input_dim"]:
            width = min(4 if scalar == "double" else 8, 1 << ((meta["input_dim"] - first).bit_length() - 1))
            guard = f"node_row < {rows} && " if node_rows * channels < 32 else ""
            feature_stores += ["{", f"Element items[{width}] = {{" + ", ".join(f"feature_sum[{first + item}]" for item in range(width)) + "};",
                              f"using Exchange = cub::WarpExchange<Element, {width}, {width}, 0, cub::WARP_EXCHANGE_SHUFFLE>;",
                              "typename Exchange::TempStorage storage;", "Exchange(storage).BlockedToStriped(items, items);",
                              f"const int local_lane = threadIdx.x & {width - 1};", "const int first_channel = channel - local_lane;"]
            for item in range(width):
                feature_stores += [f"if ({guard}node < nodes && first_channel + {item} < CHANNELS)",
                                   f"sx[(node * CHANNELS + first_channel + {item}) * INPUT_DIM + {first} + local_lane] = items[{item}];"]
            feature_stores.append("}")
            first += width
    preloads, contractions = [], []
    tiles = [0] + [(bound + channels - 1) // channels for bound in boundaries[1:]]
    for band, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        active_groups = groups
        if banded:
            active_groups = [(terms, [(basis, outputs) for basis, outputs in consumers if limits[basis] > begin])
                             for terms, consumers in groups]
            active_groups = [(terms, consumers) for terms, consumers in active_groups if consumers]
        band_path_tile = min(path_tile, max(1, sum(len(consumers) for terms, consumers in active_groups)))
        preload, contraction = pair_code(active_groups, weight_paths, ContractionLayout(meta, limits, banded, begin, end),
                                        rows, threads, band_path_tile, scalar, inline_weights, reuse_species)
        if banded:
            prefix = "if constexpr" if band == 0 else "else if constexpr"
            preloads.append(f"{prefix} (ChannelBand == {band}) {{\n{preload}\n}}")
            contractions.append(f"{prefix} (ChannelBand == {band}) {{\n{contraction}\n}}")
        else:
            preloads.append(preload)
            contractions.append(contraction)
    if banded:
        first_tile, band_tiles = str(tiles[-2]), str(tiles[-1] - tiles[-2])
        for band in range(len(tiles) - 3, -1, -1):
            first_tile = f"(ChannelBand == {band} ? {tiles[band]} : {first_tile})"
            band_tiles = f"(ChannelBand == {band} ? {tiles[band + 1] - tiles[band]} : {band_tiles})"
        mapping = [f"constexpr int FirstTile = {first_tile};", f"constexpr int BandTiles = {band_tiles};",
                   "constexpr int BandBegin = FirstTile * FUSED_CHANNEL_TILE;",
                   "constexpr int BandEnd = (FirstTile + BandTiles) * FUSED_CHANNEL_TILE;",
                   "const Index task = Index(blockIdx.y) * gridDim.x + blockIdx.x;",
                   "if (task >= tasks) return;", "const int channel_tile = FirstTile + task % BandTiles;",
                   "const Index node_tile = task / BandTiles;", "const int channel_begin = channel_tile * FUSED_CHANNEL_TILE;",
                   "const int channel = channel_begin + local_channel;", f"const Index first_node = node_tile * {rows};"]
    else:
        mapping = ["const Index task = Index(blockIdx.y) * gridDim.x + blockIdx.x;", "if (task >= tasks) return;",
                   "const int channel_begin = (task % CHANNEL_TILES) * FUSED_CHANNEL_TILE;",
                   "const int channel = channel_begin + local_channel;", f"const Index first_node = (task / CHANNEL_TILES) * {rows};"]
    dimensions = []
    if separate_rows:
        dimensions.append("const int64_t node_rows = need_attrs || ra.has_value() ? NODE_TILE : NODE_TILE_NO_ATTRS;")
    if banded:
        dimensions.append(f"const int64_t node_tiles = (nodes + {rows} - 1) / {rows};")
        for band, (begin, end) in enumerate(zip(tiles, tiles[1:])):
            dimensions += [f"const int64_t band_tasks{band} = node_tiles * {end - begin};",
                           f"const int band_blocks_x{band} = std::min<int64_t>(band_tasks{band}, 2147483647);",
                           f"const dim3 band_blocks{band}(band_blocks_x{band}, static_cast<unsigned>((band_tasks{band} + band_blocks_x{band} - 1) / band_blocks_x{band}));"]
    else:
        dimensions += [f"const int64_t tasks = ((nodes + {rows} - 1) / {rows}) * CHANNEL_TILES;",
                       "const int blocks_x = std::min<int64_t>(tasks, 2147483647);",
                       "const dim3 blocks(blocks_x, static_cast<unsigned>((tasks + blocks_x - 1) / blocks_x));"]
    launch_threads = "(WRITE_A ? FUSED_THREADS : THREADS_NO_ATTRS)" if separate_rows else "FUSED_THREADS"
    calls = []
    for band in range(len(boundaries) - 1):
        suffix = f", {band}" if banded else ""
        blocks, tasks = (f"band_blocks{band}", f"band_tasks{band}") if banded else ("blocks", "tasks")
        calls += [f"{function}<WRITE_X, WRITE_A, WEIGHT_DIRECTION{suffix}, FULL_OUTPUTS><<<{blocks}, {launch_threads}, 0, stream>>>(",
                  "x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(),",
                  "grad_out.data_ptr<Element>(), rx_ptr, rw_ptr, ra_ptr,",
                  "need_x ? sx.data_ptr<Element>() : nullptr,", "need_weights ? sw.data_ptr<Element>() : nullptr,",
                  "need_attrs ? sa.data_ptr<Element>() : nullptr,",
                  f"need_grad_out ? sg.data_ptr<Element>() : nullptr, nodes, {tasks})" + (";" if banded else "")]
    continuation = " " + chr(92) + "\n"
    launches = "#define LAUNCH_RW(WRITE_X, WRITE_A, WEIGHT_DIRECTION, FULL_OUTPUTS)"
    if banded:
        launches += " do {" + continuation + "".join("    " + call + continuation for call in calls) + "} while (false)"
    else:
        launches += continuation + continuation.join("    " + call for call in calls)
    configurations = [(True, False, True, True)]
    configurations += [(write_x, has_attrs, direction, False) for has_attrs in (True, False)
                       for write_x in (False, True) for direction in (False, True)]
    configuration = []
    for write_x, has_attrs, direction, full in configurations:
        for band in range(len(boundaries) - 1):
            flags = [str(write_x).lower(), str(has_attrs).lower(), str(direction).lower()]
            if banded:
                flags.append(str(band))
            if full:
                flags.append("true")
            block_threads = "THREADS_NO_ATTRS" if separate_rows and not has_attrs else "FUSED_THREADS"
            configuration += ["symmetric_double_backward::configure_cache(",
                              f"reinterpret_cast<const void*>(symmetric_double_backward::{function}<" + ", ".join(flags) + ">),",
                              f"symmetric_double_backward::{block_threads}, shared_per_sm);"]
    row_constants = ("\n    constexpr int node_rows = HasAttrs ? NODE_TILE : NODE_TILE_NO_ATTRS;\n"
                     "    constexpr int node_threads = HasAttrs ? FUSED_THREADS : THREADS_NO_ATTRS;\n" if separate_rows else "")
    first_species = ""
    track_active_rows = "sw || WriteX || sg"
    track_block = track_active_rows
    sparse_scheduling = ""
    if reuse_species:
        track_active_rows = "!FullOutputs && (sw || WriteX || sg)"
        track_block = "(sw || WriteX || sg) && (!FullOutputs || !scatter_block)"
        sparse_scheduling = SPARSE_PROBE.substitute(rows=rows, threads=threads)
    if inline_weights:
        sparse_scheduling += SPARSE_SCHEDULING.substitute(rows=rows, track_block=track_block)
    if reuse_species:
        stride = "WEIGHT_SIZE" if compact else "(NUM_PATHS * CHANNELS)"
        first_species = f"""        const bool has_inline_weight = FullOutputs && load_inline && active_count;
        const int first_species = has_inline_weight ? active_species[node_row][0] : 0;
        const Element first_attribute = has_inline_weight ? attr_values[node_row][first_species] : Element(0);
        const Index first_weight_base = Index(first_species) * {stride} + channel;
"""
    return CUDA_SOURCE.substitute(
        constants="\n".join(constants), feature_loader="\n".join(loader),
        band_template=", int ChannelBand" if banded else "",
        launch_threads="HasAttrs ? FUSED_THREADS : THREADS_NO_ATTRS" if separate_rows else "FUSED_THREADS",
        min_blocks=min_blocks,
        sparse_shared=SPARSE_SHARED if inline_weights else "",
        parallel_attributes=PARALLEL_ATTRIBUTES.substitute(
            rows=rows, threads=threads, track_active_rows=track_active_rows) if inline_weights else "        ",
        sparse_scheduling=sparse_scheduling, track_active_rows=track_active_rows,
        function=function, row_constants=row_constants, rows=rows, threads=threads,
        weight_vector=scalar + "2", task_mapping=indent("\n".join(mapping), "    "),
        valid_channel=layout.guard("channel", "CHANNELS"), preloads=indent("\n".join(preloads), "        "),
        first_species=first_species, feature_loads=indent("\n".join(feature_loads), "        "),
        gradient_loads=indent("\n".join(gradient_loads), "        "),
        contractions=indent("\n".join(contractions), "        "),
        feature_stores=indent("\n".join(feature_stores), "            "),
        output_stores=indent("\n".join(stores), "                "),
        dimensions=indent("\n".join(dimensions), "        "), launches=indent(launches, "        "),
        configuration=indent("\n".join(configuration), "    "))


CG_JET_SOURCE = r"""
struct Jet2 { Element value, r; };

__device__ __forceinline__ Jet2 jet2_multiply(Jet2 a, Jet2 b) {
    return {a.value*b.value, fma(a.r,b.value,a.value*b.r)};
}

__device__ __forceinline__ void jet2_add_scaled(Jet2& dst, Element c, Jet2 a) {
    dst.value=fma(c,a.value,dst.value);
    dst.r=fma(c,a.r,dst.r);
}

__device__ __forceinline__ void jet2_reverse(Jet2 a, Jet2 b, Jet2 total,
                                           Jet2& ga, Jet2& gb) {
    ga.value=fma(total.value,b.value,fma(total.r,b.r,ga.value));
    ga.r=fma(total.r,b.value,ga.r);
    gb.value=fma(total.value,a.value,fma(total.r,a.r,gb.value));
    gb.r=fma(total.r,a.value,gb.r);
}

"""


CG_PAIR_ACCESSORS = r"""__device__ __forceinline__ Jet2 load_pair(const Element* storage, int slot, int worker) {
    return {storage[(slot * ValuePlanes) * ValueStride + worker],
            storage[(slot * ValuePlanes + 1) * ValueStride + worker]};
}

__device__ __forceinline__ void store_pair(Element* storage, int slot, int worker, Jet2 value) {
    storage[(slot * ValuePlanes) * ValueStride + worker] = value.value;
    storage[(slot * ValuePlanes + 1) * ValueStride + worker] = value.r;
}

__device__ __forceinline__ void add_pair(Element* storage, int slot, int worker, Jet2 value) {
    storage[(slot * ValuePlanes) * ValueStride + worker] += value.value;
    storage[(slot * ValuePlanes + 1) * ValueStride + worker] += value.r;
}

"""


CG_BINDING_SOURCE = Template(r"""pybind11::tuple run(
    at::Tensor x, at::Tensor weights, at::Tensor attrs, at::Tensor grad_out,
    c10::optional<at::Tensor> rx, c10::optional<at::Tensor> rw,
    c10::optional<at::Tensor> ra, bool need_x, bool need_weights,
    bool need_attrs, bool need_grad_out
) {
    c10::cuda::CUDAGuard guard(x.device());
    const Index nodes = x.size(0);
    at::Tensor sx, sw, sa, sg;
    if (need_x) sx = $output_allocation(x);
    if (need_weights) sw = at::zeros_like(weights);
    if (need_attrs) sa = at::zeros_like(attrs);
    if (need_grad_out) sg = $output_allocation(grad_out);
    if (nodes > 0 && (need_x || need_weights || need_attrs || need_grad_out)) {
        auto stream = at::cuda::getCurrentCUDAStream();
$grid
        const size_t shared_bytes = ValueElements * sizeof(Element);
        #define LAUNCH(WRITE_X, WRITE_ATTRS) \
            $function<WRITE_X, WRITE_ATTRS><<<blocks, Workers, shared_bytes, stream>>>( \
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(), \
                grad_out.data_ptr<Element>(), rx.has_value() ? rx->data_ptr<Element>() : nullptr, \
                rw.has_value() ? rw->data_ptr<Element>() : nullptr, \
                ra.has_value() ? ra->data_ptr<Element>() : nullptr, \
                need_x ? sx.data_ptr<Element>() : nullptr, \
                need_weights ? sw.data_ptr<Element>() : nullptr, \
                need_attrs ? sa.data_ptr<Element>() : nullptr, \
                need_grad_out ? sg.data_ptr<Element>() : nullptr, nodes)
        if (need_x) {
            if (need_attrs) LAUNCH(true, true);
            else LAUNCH(true, false);
        } else {
            if (need_attrs) LAUNCH(false, true);
            else LAUNCH(false, false);
        }
        #undef LAUNCH
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(
        need_x ? pybind11::cast(sx) : pybind11::none(),
        need_weights ? pybind11::cast(sw) : pybind11::none(),
        need_attrs ? pybind11::cast(sa) : pybind11::none(),
        need_grad_out ? pybind11::cast(sg) : pybind11::none());
}""")


CG_CUDA_SOURCE = Template(r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace symmetric_double_backward {
using Element = $scalar;
using Index = int64_t;
$constants
$cg_jet_source
constexpr bool IdentityWeights = $identity_weights;
constexpr int ChannelTiles = (Channels + ChannelColumns - 1) / ChannelColumns;
constexpr int ValuePlanes = 2;
constexpr int ValueElements = MaxSlots * ValuePlanes * ValueStride;
constexpr unsigned WorkerMask = 0xffffffffu >> (32 - (Workers < 32 ? Workers : 32));

$pair_accessors$cg_functions
$packet_functions

// Reverse g*F_r on bounded packets of CG trees. Each worker owns one
// atom/channel; packet-local values and gradients are recomputed on every call.
template<bool WriteX, bool WriteAttrs>
__global__ __launch_bounds__(Workers) void symmetric_contraction_doublebwd_$suffix(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* __restrict__ sx,
    Element* __restrict__ sw, Element* __restrict__ sa,
    Element* __restrict__ sg, Index nodes
) {
    extern __shared__ Element storage[];
    Element* totals = storage;
    const int worker = threadIdx.x;
    const int local_channel = worker % ChannelColumns;
    const int row = worker / ChannelColumns;
    const Index task = Index(blockIdx.y) * gridDim.x + blockIdx.x;
    const Index tasks = ((nodes + Rows - 1) / Rows) * ChannelTiles;
    if (task >= tasks) return;
    const int channel = (task % ChannelTiles) * ChannelColumns + local_channel;
    const Index node = (task / ChannelTiles) * Rows + row;
    const bool valid = node < nodes && channel < Channels;
    for (int component = 0; component < InputDim; ++component) {
        const bool active = valid && channel < InputChannels[component];
        const Index index = node * InputSize + InputOffsets[component]
                            + Index(channel) * InputStrides[component];
        store_pair(storage, component, worker,
                   {active ? x[index] : Element(0), active && rx ? rx[index] : Element(0)});
    }
    if constexpr (WriteX) {
        for (int slot = 0; slot < InputDim; ++slot)
            store_pair(totals, InputDim + slot, worker, {Element(0), Element(0)});
    }
    Element attr_total[WriteAttrs ? NumElements : 1] = {};
    Element output_sum[OutputComponents] = {};
$packet_dispatch

    if constexpr (WriteX) {
        for (int component = 0; component < InputDim; ++component) {
            if (!valid || channel >= InputChannels[component]) continue;
            const Index index = node * InputSize + InputOffsets[component]
                                + Index(channel) * InputStrides[component];
            sx[index] = load_pair(totals, InputDim + component, worker).value;
        }
    }
    if (sg && valid) {
        for (int output = 0; output < OutputComponents; ++output) {
            if (channel >= OutputChannels[output]) continue;
            const Index index = node * OutputSize + OutputOffsets[output]
                                + Index(channel) * OutputStrides[output];
            sg[index] = output_sum[output];
        }
    }
    if constexpr (WriteAttrs) {
        for (int element = 0; element < NumElements; ++element) {
            Element total = attr_total[element];
            #pragma unroll
            for (int offset = ChannelColumns / 2; offset > 0; offset >>= 1)
                total += __shfl_down_sync(WorkerMask, total, offset, ChannelColumns);
            if (local_channel == 0 && node < nodes)
                atomicAdd(sa + node * NumElements + element, total);
        }
    }
}

$binding
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &symmetric_double_backward::run);
}
""")


def make_cg_double_backward_source(meta, dtype):
    from .utils import build_arena_plan, cg_member_constants, cg_plan_constants

    scalar = "double" if str(dtype) in ("float64", "torch.float64", "double") else "float"
    plan = build_arena_plan(dict(meta, source_dtype="float64" if scalar == "double" else "float32"))
    identity = all(len(members) == 1 and members[0][0] == basis and members[0][1] == 1.
                   for basis, members in enumerate(meta["cg_dag"]["weight_members"]))
    projected_members = (not identity and len(plan["arrays"]["CgMembers"])
                         > plan["constants"]["Rows"] * len(plan["arrays"]["CgRoots"]))
    packed_members = len(plan["arrays"]["CgMembers"]) > CG_MEMBER_LITERAL_LIMIT
    excluded = ("CgBlocks", "CgEdges", "CgMembers") if packed_members else ("CgBlocks", "CgEdges")
    table_plan = dict(plan, arrays={name: values for name, values in plan["arrays"].items()
                                   if name not in excluded})
    constants = cg_plan_constants(table_plan, constant_budget=CG_CONSTANT_BYTES)
    if packed_members:
        constants += "\n" + cg_member_constants(plan["arrays"]["CgMembers"], plan["item_bytes"])
    constants += "\nstruct CgReverseNode { int left, right, value, left_total, right_total; };\n"
    max_roots = max((packet[3] - packet[2] for packet in plan["arrays"]["CgPackets"]), default=0)
    projection_bytes = max_roots * meta["num_elements"] * plan["constants"]["ChannelColumns"] * 3 * plan["item_bytes"]
    if projected_members and plan["shared_bytes"] + projection_bytes <= SHARED_MEMORY_BYTES:
        if not packed_members:
            constants += "\n__device__ __forceinline__ CgMember load_cg_member(int index) { return CgMembers[index]; }\n"
        constants += f"\nconstexpr int MaxPacketRoots = {max_roots};\n"
        dispatch = ["            switch (packet_index) {"]
        for index in range(plan["constants"]["PacketCount"]):
            dispatch.append(f"            case {index}: double_backward_packet_{index}<WriteX, WriteAttrs>({PACKET_ARGUMENTS}, projected_weights, projected_totals); break;")
        dispatch.append("            }")
        grid = f"""        const int resident = 4 * {meta["multiprocessor_count"]};
        const int node_blocks = std::min<Index>((nodes + Rows - 1) / Rows,
                                               std::max(1, resident / (PacketCount * ChannelTiles)));
        const dim3 blocks(node_blocks, ChannelTiles, std::min(65535, PacketCount));"""
        binding = CG_BINDING_SOURCE.substitute(
            output_allocation="at::zeros_like", grid=grid,
            function="symmetric_contraction_doublebwd_projected_" + ("fp64_i64" if scalar == "double" else "fp32_i64"))
        return PROJECTED_CG_CUDA_SOURCE.substitute(
            cg_jet_source=CG_JET_SOURCE, pair_accessors=CG_PAIR_ACCESSORS, binding=binding,
            scalar=scalar, suffix="fp64_i64" if scalar == "double" else "fp32_i64",
            constants=constants, cg_functions=cg_native_functions(meta, plan),
            packet_functions=PROJECTED_ROOT_SOURCE + cg_packet_functions(meta, plan, projected=True),
            packet_dispatch="\n".join(dispatch), multiprocessors=meta["multiprocessor_count"])
    binding = CG_BINDING_SOURCE.substitute(
        output_allocation="at::empty_like", grid=r"""        const Index tasks = ((nodes + Rows - 1) / Rows) * ChannelTiles;
        const int blocks_x = std::min<Index>(tasks, 2147483647);
        const dim3 blocks(blocks_x, static_cast<unsigned>((tasks + blocks_x - 1) / blocks_x));""",
        function="symmetric_contraction_doublebwd_" + ("fp64_i64" if scalar == "double" else "fp32_i64"))
    return CG_CUDA_SOURCE.substitute(
        cg_jet_source=CG_JET_SOURCE, pair_accessors=CG_PAIR_ACCESSORS, binding=binding,
        identity_weights=str(identity).lower(), scalar=scalar, suffix="fp64_i64" if scalar == "double" else "fp32_i64",
        constants=constants,
        cg_functions=cg_native_functions(meta, plan),
        packet_functions=STATIC_ROOT_SOURCE.substitute(
            member_load="load_cg_member(member)" if packed_members else "CgMembers[member]")
            + cg_packet_functions(meta, plan),
        packet_dispatch=cg_packet_dispatch(plan))


STATIC_ROOT_SOURCE = Template(r"""
template<bool WriteX, bool WriteAttrs, int RootValue, int RootTotal, int RootDimension, int RootOutput,
         int RootBasis, int MemberBegin, int MemberEnd, int RootChannels, Index WeightOffset, int WeightChannels>
__device__ __forceinline__ void double_backward_root(
    const Element* __restrict__ weights, const Element* __restrict__ attrs,
    const Element* __restrict__ grad_out, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* storage, Element* totals,
    Element* __restrict__ sw, Element* __restrict__ attr_total, Element* __restrict__ output_sum,
    Index node, int channel, int worker, bool valid, bool write_output
) {
    const int row = worker / ChannelColumns;

    const bool active = valid && channel < RootChannels;
    Jet2 mixed = {Element(0), Element(0)};
    #pragma unroll 4
    for (int element = 0; element < NumElements; ++element) {
        const Index index = node * NumElements + element;
        const Jet2 attribute = {active ? attrs[index] : Element(0),
                                active && ra ? ra[index] : Element(0)};
        if (attribute.value == Element(0) && attribute.r == Element(0)) continue;
        Jet2 weight = {Element(0), Element(0)};
        if constexpr (IdentityWeights) {
            if (channel < WeightChannels) {
                const Index address = Index(element) * WeightSize + WeightOffset + channel;
                weight.value = weights[address];
                weight.r = rw ? rw[address] : Element(0);
            }
        } else {
            #pragma unroll 1
            for (int member = MemberBegin; member < MemberEnd; ++member) {
                const CgMember entry = $member_load;
                if (channel >= PathChannels[entry.path]) continue;
                const Index address = Index(element) * WeightSize + PathOffsets[entry.path] + channel;
                weight.value = fma(entry.coefficient, weights[address], weight.value);
                if (rw) weight.r = fma(entry.coefficient, rw[address], weight.r);
            }
        }
        jet2_add_scaled(mixed, Element(1), jet2_multiply(attribute, weight));
    }
    Jet2 mixed_total = {Element(0), Element(0)};
    #pragma unroll
    for (int component = 0; component < RootDimension; ++component) {
        const int output = RootOutput + component;
        const Index index = node * OutputSize + OutputOffsets[output]
                            + Index(channel) * OutputStrides[output];
        const Jet2 value = load_pair(storage, RootValue + component, worker);
        const Element gradient = active ? grad_out[index] : Element(0);
        if (write_output && active) {
            const Element direction = fma(mixed.value, value.r, mixed.r * value.value);
            output_sum[output] += direction;
        }
        if (WriteX || sw || WriteAttrs) {
            const Jet2 total = {Element(0), gradient};
            Jet2 value_total = {Element(0), Element(0)};
            jet2_reverse(mixed, value, total, mixed_total, value_total);
            if constexpr (WriteX) add_pair(totals, RootTotal + component, worker, value_total);
        }
    }
    if (sw || WriteAttrs) {
        #pragma unroll 4
        for (int element = 0; element < NumElements; ++element) {
            const Index index = node * NumElements + element;
            const Jet2 attribute = {active ? attrs[index] : Element(0),
                                    active && ra ? ra[index] : Element(0)};
            Jet2 weight = {Element(0), Element(0)};
            if constexpr (WriteAttrs) {
                if constexpr (IdentityWeights) {
                    if (channel < WeightChannels) {
                        const Index address = Index(element) * WeightSize + WeightOffset + channel;
                        weight.value = weights[address];
                        weight.r = rw ? rw[address] : Element(0);
                    }
                } else {
                    #pragma unroll 1
                    for (int member = MemberBegin; member < MemberEnd; ++member) {
                        const CgMember entry = $member_load;
                        if (channel >= PathChannels[entry.path]) continue;
                        const Index address = Index(element) * WeightSize + PathOffsets[entry.path] + channel;
                        weight.value = fma(entry.coefficient, weights[address], weight.value);
                        if (rw) weight.r = fma(entry.coefficient, rw[address], weight.r);
                    }
                }
            }
            Jet2 attribute_total = {Element(0), Element(0)};
            Jet2 weight_total = {Element(0), Element(0)};
            jet2_reverse(attribute, weight, mixed_total, attribute_total, weight_total);
            if constexpr (WriteAttrs) attr_total[element] += attribute_total.value;
            if (sw) {
                Element total = weight_total.value;
                #pragma unroll
                for (int offset = Workers / 2; offset >= ChannelColumns; offset >>= 1)
                    total += __shfl_down_sync(WorkerMask, total, offset);
                if (row == 0 && channel < Channels && total != Element(0)) {
                    if constexpr (IdentityWeights) {
                        if (channel < WeightChannels) {
                            const Index address = Index(element) * WeightSize + WeightOffset + channel;
                            atomicAdd(sw + address, total);
                        }
                    } else {
                        #pragma unroll 1
                        for (int member = MemberBegin; member < MemberEnd; ++member) {
                            const CgMember entry = $member_load;
                            if (channel >= PathChannels[entry.path]) continue;
                            const Index address = Index(element) * WeightSize + PathOffsets[entry.path] + channel;
                            atomicAdd(sw + address, entry.coefficient * total);
                        }
                    }
                }
            }
        }
    }

}
""")


PACKET_PARAMETERS = """const Element* __restrict__ weights, const Element* __restrict__ attrs,
    const Element* __restrict__ grad_out, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* storage, Element* totals,
    Element* __restrict__ sw, Element* __restrict__ attr_total_pointer, Element* __restrict__ output_sum_pointer,
    Index node, int channel, int worker, bool valid, bool write_output"""
PACKET_ARGUMENTS = "weights, attrs, grad_out, rw, ra, storage, totals, sw, attr_total, output_sum, node, channel, worker, valid, write_output"


def cg_native_functions(meta, plan):
    """Generate one bounded function per referenced local CG block."""
    forward, reverse = [], []
    used_blocks = {node[3] for node in plan["arrays"]["CgNodes"]}
    for index, block in enumerate(meta["cg_dag"]["local_cg"]):
        if index not in used_blocks:
            continue
        outputs, left_dim, right_dim = block["shape"]
        edges = [(output, block["left_components"][edge], block["right_components"][edge], block["coefficients"][edge])
                 for output in range(outputs)
                 for edge in range(block["output_offsets"][output], block["output_offsets"][output + 1])]
        values = []
        for side, width, name in (("left", left_dim, "a"), ("right", right_dim, "b")):
            values.extend(f"        const Jet2 {name}{component} = load_pair(values, node.{side} + {component}, worker);"
                          for component in range(width))
        for output in range(outputs):
            values += ["        {", "            Jet2 value = {Element(0), Element(0)};"]
            for component, left, right, coefficient in edges:
                if component == output:
                    values.append(f"            jet2_add_scaled(value, Element({coefficient:.17g}), jet2_multiply(a{left}, b{right}));")
            values += [f"            store_pair(values, node.value + {output}, worker, value);", "        }"]
        forward.append(f"__device__ __noinline__ void cg_forward_{index}(Element* values, int worker, CgNode node) {{\n\n"
                       + "\n".join(values) + "\n        }\n")
        totals = []
        for side, width, other, other_dim, name in (("left", left_dim, "right", right_dim, "b"),
                                                   ("right", right_dim, "left", left_dim, "a")):
            totals.append("        {")
            totals.extend(f"            const Jet2 {name}{component} = load_pair(values, node.{other} + {component}, worker);"
                          for component in range(other_dim))
            totals.extend(f"            const Jet2 t{output} = load_pair(totals, node.value + {output}, worker);"
                          for output in range(outputs))
            for component in range(width):
                totals += ["            {", "                Jet2 value = {Element(0), Element(0)}, unused = {Element(0), Element(0)};"]
                for output, left, right, coefficient in edges:
                    if (left if side == "left" else right) == component:
                        total = f"{{Element({coefficient:.17g}) * t{output}.value, Element({coefficient:.17g}) * t{output}.r}}"
                        arguments = (f"{{Element(0), Element(0)}}, b{right}, {total}, value, unused" if side == "left"
                                     else f"a{left}, {{Element(0), Element(0)}}, {total}, unused, value")
                        totals.append(f"                jet2_reverse({arguments});")
                totals += [f"                add_pair(totals, node.{side}_total + {component}, worker, value);", "            }"]
            totals.append("        }")
        reverse.append(f"__device__ __noinline__ void cg_reverse_{index}(const Element* values, Element* totals, int worker, CgReverseNode node) {{\n\n"
                       + "\n".join(totals) + "\n        }\n")
    return "\n".join(forward + reverse)


def cg_packet_functions(meta, plan, projected=False):
    from .utils import packet_intervals

    identity = all(len(members) == 1 and members[0][0] == basis and members[0][1] == 1.
                   for basis, members in enumerate(meta["cg_dag"]["weight_members"]))
    code = []
    arrays = plan["arrays"]
    inputs = plan["constants"]["InputDim"]
    parameters = PACKET_PARAMETERS
    if projected:
        parameters += ", const Jet2* projected_weights, Element* projected_totals"
    for index, packet in enumerate(arrays["CgPackets"]):
        node_begin, node_end, root_begin, root_end, _ = packet
        nodes = arrays["CgNodes"][node_begin:node_end]
        roots = arrays["CgRoots"][root_begin:root_end]
        allocation = plan["layouts"][index]
        intervals = packet_intervals(meta, plan, packet)
        zero_events = {}
        for key, (begin, _, width) in intervals.items():
            if key[0] == "total":
                zero_events.setdefault(begin, []).append((allocation[key], width))

        def physical(kind, base):
            return allocation.get((kind, base), base + (inputs if kind == "total" else 0))

        def zeros(event, indentation="    "):
            for slot, width in zero_events.get(event, ()):
                code.extend([indentation + "if constexpr (WriteX) {", indentation + "    #pragma unroll",
                             indentation + f"    for (int component = 0; component < {width}; ++component)",
                             indentation + f"        store_pair(totals, {slot} + component, worker, {{Element(0), Element(0)}});",
                             indentation + "}"])

        code.extend(["template<bool WriteX, bool WriteAttrs>",
                     f"__device__ __noinline__ void double_backward_packet_{index}({parameters}) {{",
                     "    Element attr_total[WriteAttrs ? NumElements : 1] = {};",
                     "    Element output_sum[OutputComponents];", "    #pragma unroll",
                     "    for (int output = 0; output < OutputComponents; ++output) output_sum[output] = output_sum_pointer[output];",
                     "    if constexpr (WriteAttrs) {", "        #pragma unroll",
                     "        for (int element = 0; element < NumElements; ++element) attr_total[element] = attr_total_pointer[element];", "    }"])
        for event, (left, right, value, block) in enumerate(nodes):
            zeros(event)
            fields = (physical("value", left), physical("value", right), physical("value", value), block)
            code.append(f"    cg_forward_{block}(storage, worker, CgNode{{{', '.join(map(str, fields))}}});")
        for offset, root in enumerate(roots):
            zeros(len(nodes) + offset)
            value, dimension, output, basis, member_begin, member_end = root
            output_channels = meta["output_channels"][output]
            weight_offset = meta["path_offsets"][basis] if identity else 0
            weight_channels = meta["path_channels"][basis] if identity else output_channels
            fields = (physical("value", value), physical("total", value), dimension, output, basis,
                      member_begin, member_end, output_channels, weight_offset, weight_channels)
            if projected:
                offset = offset * meta["num_elements"] * plan["constants"]["ChannelColumns"]
                arguments = PACKET_ARGUMENTS + f", projected_weights + {offset}, projected_totals + {offset}"
                code.append(
                    f"    double_backward_projected_root<WriteX, WriteAttrs, {dimension}, {output}, {output_channels}>"
                    f"({arguments}, {physical('value', value)}, {physical('total', value)});")
            else:
                code.append(f"    double_backward_root<WriteX, WriteAttrs, {', '.join(map(str, fields))}>({PACKET_ARGUMENTS});")
        code.append("    if constexpr (WriteX) {")
        for offset, (left, right, value, block) in enumerate(reversed(nodes)):
            zeros(len(nodes) + len(roots) + offset, "        ")
            fields = (physical("value", left), physical("value", right), physical("total", value),
                      physical("total", left), physical("total", right))
            code.append(f"        cg_reverse_{block}(storage, totals, worker, CgReverseNode{{{', '.join(map(str, fields))}}});")
        code.extend(["    }", "    #pragma unroll",
                     "    for (int output = 0; output < OutputComponents; ++output) output_sum_pointer[output] = output_sum[output];",
                     "    if constexpr (WriteAttrs) {", "        #pragma unroll",
                     "        for (int element = 0; element < NumElements; ++element) attr_total_pointer[element] = attr_total[element];", "    }", "}"])
    return "\n".join(code)


def cg_packet_dispatch(plan):
    code = ["    const bool write_output = sg != nullptr;",
            "    for (int packet_index = 0; packet_index < PacketCount; ++packet_index) {",
            "        switch (packet_index) {"]
    for index in range(plan["constants"]["PacketCount"]):
        code += [f"        case {index}: double_backward_packet_{index}<WriteX, WriteAttrs>({PACKET_ARGUMENTS}); break;"]
    code += ["        }", "    }"]
    return "\n".join(code)


PROJECTED_ROOT_SOURCE = r"""
// Share root arithmetic across packet slots; only output layout remains static.
template<bool WriteX, bool WriteAttrs, int RootDimension, int RootOutput, int RootChannels>
__device__ __noinline__ void double_backward_projected_root(
    const Element* __restrict__ weights, const Element* __restrict__ attrs,
    const Element* __restrict__ grad_out, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* storage, Element* totals,
    Element* __restrict__ sw, Element* __restrict__ attr_total, Element* __restrict__ output_sum,
    Index node, int channel, int worker, bool valid, bool write_output,
    const Jet2* __restrict__ projected_weights, Element* __restrict__ projected_totals,
    int RootValue, int RootTotal
) {
    const int column = worker % ChannelColumns;
    const int row = worker / ChannelColumns;
    const bool active = valid && channel < RootChannels;
    Jet2 mixed = {Element(0), Element(0)};
    #pragma unroll
    for (int element = 0; element < NumElements; ++element) {
        const Index index = node * NumElements + element;
        const Jet2 attribute = {active ? attrs[index] : Element(0),
                                active && ra ? ra[index] : Element(0)};
        const Jet2 weight = projected_weights[element * ChannelColumns + column];
        jet2_add_scaled(mixed, Element(1), jet2_multiply(attribute, weight));
    }
    Jet2 mixed_total = {Element(0), Element(0)};
    #pragma unroll
    for (int component = 0; component < RootDimension; ++component) {
        const int output = RootOutput + component;
        const Index index = node * OutputSize + OutputOffsets[output]
                            + Index(channel) * OutputStrides[output];
        const Jet2 value = load_pair(storage, RootValue + component, worker);
        const Element gradient = active ? grad_out[index] : Element(0);
        if (write_output && active)
            output_sum[output] += fma(mixed.value, value.r, mixed.r * value.value);
        if (WriteX || sw || WriteAttrs) {
            const Jet2 total = {Element(0), gradient};
            Jet2 value_total = {Element(0), Element(0)};
            jet2_reverse(mixed, value, total, mixed_total, value_total);
            if constexpr (WriteX) add_pair(totals, RootTotal + component, worker, value_total);
        }
    }
    if (sw || WriteAttrs) {
        #pragma unroll
        for (int element = 0; element < NumElements; ++element) {
            const Index index = node * NumElements + element;
            const Jet2 attribute = {active ? attrs[index] : Element(0),
                                    active && ra ? ra[index] : Element(0)};
            const Jet2 weight = projected_weights[element * ChannelColumns + column];
            if constexpr (WriteAttrs)
                attr_total[element] += fma(weight.value, mixed_total.value, weight.r * mixed_total.r);
            if (sw) {
                Element total = fma(attribute.value, mixed_total.value, attribute.r * mixed_total.r);
                #pragma unroll
                for (int offset = Workers / 2; offset >= ChannelColumns; offset >>= 1)
                    total += __shfl_down_sync(WorkerMask, total, offset);
                if (row == 0) projected_totals[element * ChannelColumns + column] += total;
            }
        }
    }
}
"""


PROJECTED_CG_CUDA_SOURCE = Template(r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace symmetric_double_backward {
using Element = $scalar;
using Index = int64_t;
$constants
$cg_jet_source
constexpr int ChannelTiles = (Channels + ChannelColumns - 1) / ChannelColumns;
constexpr int ValuePlanes = 2;
constexpr int ValueElements = MaxSlots * ValuePlanes * ValueStride;
constexpr unsigned WorkerMask = 0xffffffffu >> (32 - Workers);

$pair_accessors$cg_functions
$packet_functions

// TwoSum preserves the rounding residual while the parameter map is reused
// over the node partition. All products and sums remain native Element ops.
__device__ __forceinline__ void add_projected_value(Element value, Element& total, Element& correction) {
    Element next;
    if constexpr (sizeof(Element) == sizeof(float)) next = __fadd_rn(total, value);
    else next = __dadd_rn(total, value);
    const Element restored = next - total;
    correction += (total - (next - restored)) + (value - restored);
    total = next;
}

// A block owns a packet and a channel tile. Project parameters once, consume
// its node partition, then apply the transpose map to the reduced root totals.
template<bool WriteX, bool WriteAttrs>
__global__ __launch_bounds__(Workers) void symmetric_contraction_doublebwd_projected_$suffix(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ rw,
    const Element* __restrict__ ra, Element* __restrict__ sx,
    Element* __restrict__ sw, Element* __restrict__ sa,
    Element* __restrict__ sg, Index nodes
) {
    extern __shared__ Element storage[];
    __shared__ Jet2 projected_weights[MaxPacketRoots * NumElements * ChannelColumns];
    __shared__ Element projected_totals[MaxPacketRoots * NumElements * ChannelColumns];
    Element* totals = storage;
    const int worker = threadIdx.x;
    const int column = worker % ChannelColumns;
    const int row = worker / ChannelColumns;
    const int channel = blockIdx.y * ChannelColumns + column;
    const bool write_output = sg != nullptr;
    for (int packet_index = blockIdx.z; packet_index < PacketCount; packet_index += gridDim.z) {
        const CgPacket packet = CgPackets[packet_index];
        const int root_count = packet.root_end - packet.root_begin;
        for (int root_slot = 0; root_slot < root_count; ++root_slot) {
            const CgRoot root = CgRoots[packet.root_begin + root_slot];
            for (int element = 0; element < NumElements; ++element) {
                Jet2 weight = {Element(0), Element(0)};
                Jet2 correction = {Element(0), Element(0)};
                if (channel < Channels) {
                    #pragma unroll 1
                    for (int member = root.member_begin + row; member < root.member_end; member += Rows) {
                        const CgMember entry = load_cg_member(member);
                        if (channel >= PathChannels[entry.path]) continue;
                        const Index address = Index(element) * WeightSize + PathOffsets[entry.path] + channel;
                        const Element value = weights[address];
                        Element product;
                        if constexpr (sizeof(Element) == sizeof(float)) product = __fmul_rn(entry.coefficient, value);
                        else product = __dmul_rn(entry.coefficient, value);
                        add_projected_value(product, weight.value, correction.value);
                        correction.value += fma(entry.coefficient, value, -product);
                        if (rw) {
                            const Element direction = rw[address];
                            if constexpr (sizeof(Element) == sizeof(float)) product = __fmul_rn(entry.coefficient, direction);
                            else product = __dmul_rn(entry.coefficient, direction);
                            add_projected_value(product, weight.r, correction.r);
                            correction.r += fma(entry.coefficient, direction, -product);
                        }
                    }
                }
                #pragma unroll
                for (int offset = ChannelColumns; offset < Workers; offset <<= 1) {
                    const Jet2 other = {__shfl_xor_sync(WorkerMask, weight.value, offset),
                                        __shfl_xor_sync(WorkerMask, weight.r, offset)};
                    const Jet2 other_correction = {__shfl_xor_sync(WorkerMask, correction.value, offset),
                                                   __shfl_xor_sync(WorkerMask, correction.r, offset)};
                    add_projected_value(other.value, weight.value, correction.value);
                    add_projected_value(other.r, weight.r, correction.r);
                    correction.value += other_correction.value;
                    correction.r += other_correction.r;
                }
                if (row == 0) {
                    const int index = (root_slot * NumElements + element) * ChannelColumns + column;
                    projected_weights[index] = {weight.value + correction.value, weight.r + correction.r};
                    projected_totals[index] = Element(0);
                }
            }
        }
        __syncthreads();
        for (Index first_node = Index(blockIdx.x) * Rows; first_node < nodes;
             first_node += Index(gridDim.x) * Rows) {
            const Index node = first_node + row;
            const bool valid = node < nodes && channel < Channels;
            for (int component = 0; component < InputDim; ++component) {
                const bool active = valid && channel < InputChannels[component];
                const Index index = node * InputSize + InputOffsets[component]
                                    + Index(channel) * InputStrides[component];
                store_pair(storage, component, worker,
                           {active ? x[index] : Element(0), active && rx ? rx[index] : Element(0)});
            }
            if constexpr (WriteX) {
                for (int slot = 0; slot < InputDim; ++slot)
                    store_pair(totals, InputDim + slot, worker, {Element(0), Element(0)});
            }
            Element attr_total[WriteAttrs ? NumElements : 1] = {};
            Element output_sum[OutputComponents] = {};
$packet_dispatch
            if constexpr (WriteX) {
                for (int component = 0; component < InputDim; ++component) {
                    if (!valid || channel >= InputChannels[component]) continue;
                    const Index index = node * InputSize + InputOffsets[component]
                                        + Index(channel) * InputStrides[component];
                    atomicAdd(sx + index, load_pair(totals, InputDim + component, worker).value);
                }
            }
            if (sg && valid) {
                for (int output = 0; output < OutputComponents; ++output) {
                    if (channel >= OutputChannels[output]) continue;
                    const Index index = node * OutputSize + OutputOffsets[output]
                                        + Index(channel) * OutputStrides[output];
                    atomicAdd(sg + index, output_sum[output]);
                }
            }
            if constexpr (WriteAttrs) {
                for (int element = 0; element < NumElements; ++element) {
                    Element total = attr_total[element];
                    #pragma unroll
                    for (int offset = ChannelColumns / 2; offset > 0; offset >>= 1)
                        total += __shfl_down_sync(WorkerMask, total, offset, ChannelColumns);
                    if (column == 0 && node < nodes)
                        atomicAdd(sa + node * NumElements + element, total);
                }
            }
        }
        __syncthreads();
        if (sw && channel < Channels) {
            for (int root_slot = 0; root_slot < root_count; ++root_slot) {
                const CgRoot root = CgRoots[packet.root_begin + root_slot];
                for (int element = 0; element < NumElements; ++element) {
                    const Element total = projected_totals[(root_slot * NumElements + element) * ChannelColumns + column];
                    if (total == Element(0)) continue;
                    #pragma unroll 1
                    for (int member = root.member_begin + row; member < root.member_end; member += Rows) {
                        const CgMember entry = load_cg_member(member);
                        if (channel >= PathChannels[entry.path]) continue;
                        const Index index = Index(element) * WeightSize + PathOffsets[entry.path] + channel;
                        atomicAdd(sw + index, entry.coefficient * total);
                    }
                }
            }
        }
        __syncthreads();
    }
}

$binding
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &symmetric_double_backward::run); }
""")
