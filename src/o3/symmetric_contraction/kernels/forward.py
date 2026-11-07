"""Fused symmetric contractions with static CG packets and polynomial terms."""

from collections import Counter, defaultdict
import struct
from textwrap import indent

from e3nn import o3

from .utils import (
    BLOCK_SIZE, FORWARD_RESIDENT_THREADS, FORWARD_WINDOW_WORK,
    MAX_BLOCKS, SHARED_MEMORY_BYTES, CG_PACKET_LIVE_SLOTS, CG_PACKET_EDGE_BUDGET,
    CG_PACKET_ROOT_BUDGET, CG_PRIVATE_REGISTER_WORDS, CG_PACKET_REGISTER_OVERHEAD,
    FORWARD_CG_THREADS,
    cg_member_constants,
    CG_MEMBER_LITERAL_LIMIT,
    bounded_cg_polynomials, reduced_pair_groups,
)


CUDA_SOURCE = r"""
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>

using Element = /* element */;
using Index = int64_t;/* layout constants */
constexpr int Channels = /* channels */;
constexpr int InputDim = /* input_dim */;
constexpr int OutputDim = /* output_dim */;
constexpr int NumElements = /* num_elements */;
constexpr int Threads = /* threads */;
constexpr int RowThreads = /* row_threads */;
constexpr int AttributeThreads = RowThreads < 32 ? RowThreads : 32;
constexpr bool RegisterAttributes = /* register_attributes */;
constexpr bool ShuffleAttributes = /* shuffle_attributes */;
constexpr int ChannelColumns = /* channel_columns */;
constexpr int Rows = Threads / RowThreads;
constexpr int ChannelTiles = (Channels + ChannelColumns - 1) / ChannelColumns;

template<typename Index>
__global__ __launch_bounds__(Threads, /* resident_threads */ / Threads)
void symmetric_contraction_fwd_/* precision */(
    const Element* __restrict__ x, const Element* __restrict__ weights,
    const Element* __restrict__ attrs, Element* __restrict__ output, Index atoms
) {
    __shared__ Element attr_values[Rows][NumElements];
    __shared__ int attr_elements[Rows][NumElements];
    __shared__ int active_counts[Rows];
    const int lane = threadIdx.x % RowThreads;
    const int row = threadIdx.x / RowThreads;
    for (int tile = blockIdx.y; tile < ChannelTiles; tile += gridDim.y) {
        for (Index first_atom = Index(blockIdx.x) * Rows; first_atom < atoms;
             first_atom += Index(gridDim.x) * Rows) {
            const Index atom = first_atom + row;
            const int channel_start = tile * ChannelColumns;
            const int channel = channel_start + lane;
            const int warp_offset = (threadIdx.x % 32) - lane;
            const unsigned lanes = (0xffffffffu >> (32 - AttributeThreads)) << warp_offset;
            Element attributes[RegisterAttributes ? NumElements : 1];
            Element lane_attribute = 0;
            unsigned attribute_mask = 0;
            if constexpr (RegisterAttributes) {
                const Element value = atom < atoms && lane < NumElements
                    ? attrs[atom * NumElements + lane] : Element(0);
                #pragma unroll
                for (int element = 0; element < NumElements; ++element)
                    attributes[element] = __shfl_sync(lanes, value, element, RowThreads);/* register attribute mask */
            } else if constexpr (ShuffleAttributes) {
                lane_attribute = atom < atoms && lane < NumElements
                    ? attrs[atom * NumElements + lane] : Element(0);
                attribute_mask = __ballot_sync(lanes, lane_attribute != Element(0)) >> warp_offset;
            } else if (lane < AttributeThreads) {
                int count = 0;
                #pragma unroll
                for (int begin = 0; begin < NumElements; begin += AttributeThreads) {
                    const int element = begin + lane;
                    const Element value = atom < atoms && element < NumElements
                        ? attrs[atom * NumElements + element] : Element(0);
                    const unsigned mask = __ballot_sync(lanes, value != Element(0)) >> warp_offset;
                    if (value != Element(0)) {
                        const int position = count + __popc(mask & ((1u << lane) - 1));
                        attr_values[row][position] = value;
                        attr_elements[row][position] = element;
                    }
                    count += __popc(mask);
                }
                if (lane == 0) active_counts[row] = count;
            }
            if constexpr (!RegisterAttributes && !ShuffleAttributes) __syncwarp();
            if (atom < atoms && lane < ChannelColumns && channel < Channels) {
                int active_count = 0, first_element = 0;
                Element first_attribute = 0;
                if constexpr (!RegisterAttributes && !ShuffleAttributes) {
                    active_count = active_counts[row];
                    if (active_count) {
                        first_element = attr_elements[row][0];
                        first_attribute = attr_values[row][0];
                    }
                }
                /* input loads */
                /* outputs */
                /* contractions */
                /* stores */
            }
            if constexpr (!RegisterAttributes && !ShuffleAttributes) __syncwarp();
        }
    }
}

at::Tensor run(at::Tensor x, at::Tensor weights, at::Tensor attrs) {
/* layout checks */    TORCH_CHECK(attrs.dim() == 2 && attrs.size(0) == x.size(0) && attrs.size(1) == NumElements,
                "attrs must have shape [atoms, elements]");
    c10::cuda::CUDAGuard guard(x.device());
    const Index atoms = x.size(0);
    auto output = at::empty({atoms, Index(OutputDim)}, x.options());
    if (atoms) {
        const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
        const dim3 grid(std::min<Index>((atoms + Rows - 1) / Rows, /* max_blocks */),
                        std::min(ChannelTiles, /* max_blocks */));
        const int64_t limit = std::numeric_limits<int32_t>::max();
        const bool small_index = atoms <= limit - Index(grid.x) * Rows - Rows &&
            x.numel() <= limit && weights.numel() <= limit &&
            attrs.numel() <= limit && output.numel() <= limit;
        if (small_index) {
            symmetric_contraction_fwd_/* precision */<int32_t><<<grid, Threads, 0, stream>>>(
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(),
                output.data_ptr<Element>(), int32_t(atoms));
        } else {
            symmetric_contraction_fwd_/* precision */<int64_t><<<grid, Threads, 0, stream>>>(
                x.data_ptr<Element>(), weights.data_ptr<Element>(), attrs.data_ptr<Element>(),
                output.data_ptr<Element>(), atoms);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
"""


def binary_windows(meta):
    binary = meta["binary"]
    selected = {path["basis_path"]: path for path in binary["selected_paths"]}
    blocks = tuple(binary["local_cg_blocks"].values())
    entries = []
    for pair in binary["pair_groups"]:
        available = {component for component, _, _ in pair["terms"]}
        consumers = []
        for basis, _, _, sign in pair["consumers"]:
            path = selected[basis]
            outer = defaultdict(float)
            if path["degree"] == 2:
                for component in available:
                    outer[path["output_component"] + component, (meta["input_dim"] + component,)] += sign
            else:
                tensor = next(block["tensor"] for block in blocks if block["tree"] == path["tree"])
                right = path["tree"]["args"][1]["start"]
                for output, component, right_component in tensor.nonzero().tolist():
                    if component in available:
                        outer[path["output_component"] + output,
                              (right + right_component, meta["input_dim"] + component)] += sign * tensor[output, component, right_component].item()
            consumers.append((basis, tuple((*key, value) for key, value in sorted(outer.items()) if value != 0)))
        for first in range(0, len(consumers), 8):
            entries.append((pair["terms"], consumers[first:first + 8]))
    for path in binary["selected_paths"]:
        if path["degree"] == 1:
            terms = tuple((path["output_component"] + component,
                           (path["tree"]["start"] + component,), 1.0)
                          for component in range(path["output_components"]))
            entries.append(((), [(path["basis_path"], terms)]))
    return pack_forward_windows(entries, binary["compute_weight_paths"])


def pack_forward_windows(entries, weight_paths=None):
    windows = []
    for entry in sorted(entries, key=lambda item: -len(item[1])):
        for window in windows:
            if sum(len(consumers) for _, consumers in window) + len(entry[1]) <= 8:
                window.append(entry)
                break
        else:
            windows.append([entry])
    return repack_windows(windows, weight_paths) if weight_paths is not None else windows


def repack_windows(windows, weight_paths):
    entries = [entry for window in windows for entry in window]
    paths = [{path for basis, _ in entry[1] for path, _ in weight_paths[basis]}
             for entry in entries]
    sizes = [len(entry[1]) for entry in entries]
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
                        if sum(sizes[item] for item in lhs) > 8 or sum(sizes[item] for item in rhs) > 8:
                            continue
                        delta = loads(lhs) + loads(rhs) - before
                        if delta < improvement:
                            improvement, move = delta, (left, right, lhs, rhs)
        if move is None:
            break
        left, right, packed[left], packed[right] = move
    return [[entries[item] for item in group] for group in packed if group]


def element_coefficient(value, scalar):
    return value if scalar == "double" else struct.unpack("f", struct.pack("f", value))[0]


def factored_polynomial_expression(terms, scalar):
    """Emit the forward expression, using add/subtract for represented ±1 coefficients."""
    if not terms:
        return "Element(0)"
    common = abs(element_coefficient(terms[0][1], scalar))
    if len(terms) > 1 and common not in (0, 1) and all(
            abs(element_coefficient(coefficient, scalar)) == common for _, coefficient in terms):
        normalized = [(inputs, 1. if coefficient > 0 else -1.) for inputs, coefficient in terms]
        expression = factored_polynomial_expression(normalized, scalar)
        return f"(Element({common:.17g}) * {expression})"
    if not terms[0][0]:
        return f"Element({terms[0][1]:.17g})"
    groups = defaultdict(list)
    for inputs, coefficient in terms:
        groups[inputs[0]].append((inputs[1:], coefficient))
    expression = None
    for component, remaining in sorted(groups.items()):
        child = factored_polynomial_expression(remaining, scalar)
        constant = (element_coefficient(remaining[0][1], scalar)
                    if len(remaining) == 1 and not remaining[0][0] else None)
        value = f"values[{component}]"
        if expression is None:
            expression = value if constant == 1 else f"(-{value})" if constant == -1 else f"({value} * {child})"
        elif constant == 1:
            expression = f"({expression} + {value})"
        elif constant == -1:
            expression = f"({expression} - {value})"
        else:
            expression = f"fma({value}, {child}, {expression})"
    return expression


def contraction_loop(meta, windows, limits, channel_end=None, scalar="float", register_attributes=False, shuffle_attributes=False, single_attribute=False):
    binary = meta["binary"]
    weight_stride = "WeightSize"
    bound = meta["channels"] if channel_end is None else channel_end
    blocks = []
    register_words = meta["input_dim"] + len(meta["output_offsets"]) + max(
        (sum(len(consumers) for _, consumers in window) for window in windows), default=0)
    straight_line = register_words * (2 if scalar == "double" else 1) <= 64
    for window_id, pairs in enumerate(windows):
        consumers = [consumer for _, items in pairs for consumer in items]
        members = [binary["compute_weight_paths"][basis] for basis, _ in consumers]
        code = ["{" if straight_line else f"case {window_id}: {{"]
        active = max(limits[basis] for basis, _ in consumers)
        if active < bound:
            if straight_line:
                code.append(f"    if (channel < {active}) {{")
            else:
                code.append(f"    if (channel >= {active}) break;")
        code.extend(f"    Element weight_{local} = 0;" for local in range(len(consumers)))
        elements = (range(meta["num_elements"]) if register_attributes else
                    (None,) if shuffle_attributes or single_attribute else (None, None))
        for item, element in enumerate(elements):
            first = not item
            if single_attribute:
                code += ["    {", f"        const Index base = Index(first_element) * {weight_stride} + channel;",
                         "        const Element attribute = first_attribute;"]
            elif shuffle_attributes:
                code += ["    unsigned remaining = attribute_mask;", "    #pragma unroll 1",
                         "    while (remaining) {", "        const int element = __ffs(remaining) - 1;",
                         f"        const Index base = Index(element) * {weight_stride} + channel;",
                         "        const Element attribute = __shfl_sync(lanes, lane_attribute, element, RowThreads);"]
            elif register_attributes:
                code += [f"    if (attributes[{element}] != Element(0)) {{",
                         f"        const Index base = Index({element}) * {weight_stride} + channel;",
                         f"        const Element attribute = attributes[{element}];"]
            elif first:
                code += ["    if (active_count > 0) {",
                         f"        const Index base = Index(first_element) * {weight_stride} + channel;",
                         "        const Element attribute = first_attribute;"]
            else:
                code += [f"    #pragma unroll {2 if meta['num_elements'] > 32 else 1}",
                         "    for (int item = 1; item < NumElements; ++item) {",
                         "        if (item >= active_count) break;",
                         f"        const Index base = Index(attr_elements[row][item]) * {weight_stride} + channel;",
                         "        const Element attribute = attr_values[row][item];"]
            for local, items in enumerate(members):
                limit = limits[consumers[local][0]]
                guard = f"if (channel < {limit}) " if limit < bound else ""
                code += [f"        {guard}{{", "            Element value = 0;"]
                for position, (path, coefficient) in enumerate(items):
                    value = (f"weights[base + Index({meta['path_offsets'][path]})]")
                    if position == 0:
                        represented = element_coefficient(coefficient, scalar)
                        value = value if represented == 1 else f"-{value}" if represented == -1 else f"Element({coefficient:.17g}) * {value}"
                        code.append(f"            value = {value};")
                    elif abs(coefficient) == 1:
                        code.append(f"            value {'+=' if coefficient == 1 else '-='} {value};")
                    else:
                        code.append(f"            value = fma(Element({coefficient:.17g}), {value}, value);")
                value = "attribute * value" if first and not register_attributes and not shuffle_attributes else f"fma(attribute, value, weight_{local})"
                code += [f"            weight_{local} = {value};", "        }"]
            if shuffle_attributes:
                code.append("        remaining &= remaining - 1;")
            code.append("    }")
        local = 0
        for pair, consumers in pairs:
            active = max(limits[basis] for basis, _ in consumers)
            pair_terms = defaultdict(list)
            for component, inputs, coefficient in pair:
                pair_terms[component].append((inputs, coefficient))
            used_pair = sorted({component - meta["input_dim"] for _, terms in consumers
                                for _, inputs, _ in terms for component in inputs if component >= meta["input_dim"]})
            pair_inputs = {component for m in used_pair for inputs, _ in pair_terms[m] for component in inputs}
            outer_inputs = {component for _, terms in consumers for _, inputs, _ in terms
                            for component in inputs if component < meta["input_dim"]}
            guard = f"if (channel < {active}) " if active < bound else ""
            code += [f"    {guard}{{", "        Element values[InputDim];"]
            code.extend(f"        values[{component}] = input[{component}];"
                        for component in sorted(pair_inputs))
            code.extend(f"        const Element pair_{component} = {factored_polynomial_expression(pair_terms[component], scalar)};"
                        for component in used_pair)
            code.extend(f"        values[{component}] = input[{component}];"
                        for component in sorted(outer_inputs - pair_inputs))
            for basis, terms in consumers:
                guarded = limits[basis] < min(active, bound)
                if guarded:
                    code.append(f"        if (channel < {limits[basis]}) {{")
                outputs = defaultdict(list)
                for output, inputs, coefficient in terms:
                    outputs[output].append((inputs, coefficient))
                for output, polynomial in sorted(outputs.items()):
                    expression = factored_polynomial_expression(polynomial, scalar)
                    for component in used_pair:
                        expression = expression.replace(f"values[{meta['input_dim'] + component}]", f"pair_{component}")
                    code.append(f"        out_{output} = fma(weight_{local}, {expression}, out_{output});")
                if guarded:
                    code.append("        }")
                local += 1
            code.append("    }")
        if straight_line:
            blocks += code + (["    }"] if active < bound else []) + ["}"]
        else:
            blocks += code + ["    break;", "}"]

    if straight_line:
        body = "\n".join(blocks)
    else:
        body = ("#pragma unroll 1\n"
                f"for (int window = 0; window < {len(windows)}; ++window) {{\n"
                "    switch (window) {\n" + indent("\n".join(blocks), " " * 8) + "\n    }\n}")
    if register_attributes:
        sparse = contraction_loop(meta, windows, limits, channel_end, scalar, single_attribute=True)
        return ("if (active_count == 1) {\n" + indent(sparse, " " * 4) +
                "\n} else {\n" + indent(body, " " * 4) + "\n}")
    return body


def polynomial_windows(meta, allowed=None):
    polynomials = defaultdict(list)
    for output, path, inputs, coefficient in meta.get("polynomials", meta)["terms"]:
        if allowed is None or path in allowed:
            polynomials[path].append((output, tuple(inputs), coefficient))
    consumers = [(path, tuple(terms)) for path, terms in sorted(polynomials.items())]
    return [[((), consumers[first:first + 8])] for first in range(0, len(consumers), 8)]


def forward_packet_schedule(meta, selected, reuse_inputs=True):
    """Schedule the selected roots and reuse a CG vector's slots after its last read."""
    graph = meta['cg_dag']
    nodes, roots = graph['nodes'], graph['roots']
    closure = set()
    def visit(index):
        node = nodes[index]
        if node['kind'] == 'input' or index in closure:
            return
        visit(node["left_id"])
        visit(node["right_id"])
        closure.add(index)
    root_by_node = defaultdict(list)
    for index in selected:
        root_by_node[roots[index]['node_id']].append(index)
        visit(roots[index]['node_id'])
    uses = Counter()
    for index in closure:
        node = nodes[index]
        uses.update((node['left_id'], node['right_id']))
    bases = {i: node["input_start"] for i, node in enumerate(nodes) if node["kind"] == "input"}
    occupied = [True] * meta['input_dim']
    peak = len(occupied)
    operations = []
    def release(index):
        if nodes[index]['kind'] != 'input':
            begin = bases[index]
            occupied[begin:begin + nodes[index]['irrep_dim']] = [False] * nodes[index]['irrep_dim']
    def allocate(dimension):
        nonlocal peak
        begin = meta['input_dim']
        while True:
            if len(occupied) < begin + dimension:
                occupied.extend([False] * (begin + dimension - len(occupied)))
            if not any(occupied[begin:begin + dimension]):
                occupied[begin:begin + dimension] = [True] * dimension
                peak = max(peak, begin + dimension)
                return begin
            begin += 1
    for index, node in enumerate(nodes):
        if node["kind"] == "input":
            for root in root_by_node[index]:
                operations.append(("root", root, bases[index]))
    for index in sorted(closure):
        node = nodes[index]
        left, right = node["left_id"], node["right_id"]
        left_base, right_base = bases[left], bases[right]
        # The inlined local CG function reads both vectors before its first store.
        if reuse_inputs:
            for child in (left, right):
                uses[child] -= 1
                if not uses[child]:
                    release(child)
        destination = allocate(node["irrep_dim"])
        bases[index] = destination
        operations.append(("cg", node["cg_id"], left_base, right_base, destination))
        for root in root_by_node[index]:
            operations.append(("root", root, destination))
        if not reuse_inputs:
            for child in (left, right):
                uses[child] -= 1
                if not uses[child]:
                    release(child)
        if not uses[index]:
            release(index)
    blocks = [graph["local_cg"][op[1]] for op in operations if op[0] == "cg"]
    return dict(slots=peak, operations=operations, roots=tuple(selected),
                edges=sum(len(block["coefficients"]) for block in blocks),
                operands=max((block["shape"][1] + block["shape"][2] for block in blocks), default=0))


def forward_packets(meta, slot_budget=CG_PACKET_LIVE_SLOTS, edge_budget=CG_PACKET_EDGE_BUDGET, root_budget=CG_PACKET_ROOT_BUDGET,
                     element_words=1, private_words=CG_PRIVATE_REGISTER_WORDS):
    roots,nodes = meta['cg_dag']['roots'],meta['cg_dag']['nodes']
    groups = defaultdict(list)
    for index,root in enumerate(roots):
        node = nodes[root['node_id']]
        groups[node['left_id'] if node['kind']=='coupling' else root['node_id']].append(index)
    def fits(packet):
        registers = (packet['slots']+packet['operands']+len(meta['output_offsets']))*element_words+CG_PACKET_REGISTER_OVERHEAD
        return (packet['slots']<=slot_budget and packet['edges']<=edge_budget
                and len(packet['roots'])<=root_budget and registers<=private_words)
    def finish(packet):
        packet['shared'] = not fits(packet)
        if packet['shared']:
            # Shared operations keep their destinations disjoint from their two inputs.
            packet.update(forward_packet_schedule(meta,packet['roots'],reuse_inputs=False))
        return packet
    selected,packets = [],[]
    for group in groups.values():
        for root in group:
            candidate = forward_packet_schedule(meta,(*selected,root))
            if not fits(candidate) and selected:
                packets.append(finish(packet))
                selected = []
                candidate = forward_packet_schedule(meta,(root,))
            selected.append(root)
            packet = candidate
    if selected:
        packets.append(finish(packet))
    return packets


def shared_cg_functions(meta):
    edges,functions=[],[]
    for block_id,block in enumerate(meta['cg_dag']['local_cg']):
        begin=len(edges)
        edges.extend(zip(block['left_components'],block['right_components'],block['coefficients']))
        code=[f'__device__ __noinline__ void shared_cg_block_{block_id}(Element* values, int left, int right, int destination, int worker) {{']
        for output in range(block['shape'][0]):
            first,last=(begin+block['output_offsets'][output],begin+block['output_offsets'][output+1])
            code += ['    {','        Element sum = 0;','        #pragma unroll 1',
                     f'        for (int index={first}; index<{last}; ++index) {{',
                     '            const SharedEdge edge=SharedEdges[index];',
                     '            sum=fma(edge.coefficient * values[(left+edge.left)*Threads+worker],',
                     '                    values[(right+edge.right)*Threads+worker],sum);','        }',
                     f'        values[(destination+{output})*Threads+worker]=sum;','    }']
        code.append('}')
        functions.append('\n'.join(code))
    data=','.join('{'+str(left)+','+str(right)+f',Element({coefficient:.17g})'+'}' for left,right,coefficient in edges) or '{0,0,Element(0)}'
    return ('struct SharedEdge { int left,right; Element coefficient; };\n'
            '__device__ const SharedEdge SharedEdges[] = {'+data+'};\n'+'\n'.join(functions))


def local_cg_functions(meta):
    code=[]
    for block_id,block in enumerate(meta['cg_dag']['local_cg']):
        code.append(f'__device__ __forceinline__ void cg_block_{block_id}(Element* values, int left, int right, int destination) {{')
        for side,dimension in (('left',block['shape'][1]),('right',block['shape'][2])):
            for component in range(dimension):code.append(f'    const Element {side}{component} = values[{side} + {component}];')
        for output in range(block['shape'][0]):
            code.append(f'    Element sum{output} = Element(0);')
            for edge in range(block['output_offsets'][output],block['output_offsets'][output+1]):
                l,r,c = block['left_components'][edge],block['right_components'][edge],block['coefficients'][edge]
                code.append(f'    sum{output} = fma(Element({c:.17g}) * left{l}, right{r}, sum{output});')
            code.append(f'    values[destination + {output}] = sum{output};')
        code.append('}')
    return '\n'.join(code)


def make_cg_forward_source(meta, dtype='float32', private_words=CG_PRIVATE_REGISTER_WORDS):
    """Keep CG packets in registers when they fit; use shared slots otherwise."""
    threads = FORWARD_CG_THREADS
    scalar = 'double' if str(dtype) in ('float64','torch.float64','double') else 'float'
    item_bytes = 8 if scalar=='double' else 4
    packets = forward_packets(meta,element_words=item_bytes//4,private_words=private_words)
    shared_slots = max((p['slots'] for p in packets if p['shared']),default=0)
    shared_budget = meta.get('shared_memory_bytes', SHARED_MEMORY_BYTES)
    while threads>1 and shared_slots*threads*item_bytes>shared_budget:
        threads//=2
    shared_bytes=shared_slots*threads*item_bytes
    graph = meta['cg_dag']
    outputs = len(meta['output_offsets'])
    mixed_members = any(len(mapping) > 1 for mapping in graph['weight_members'])
    parallel_projection = threads >= 32 and sum(map(len, graph['weight_members'])) > CG_MEMBER_LITERAL_LIMIT
    projection_columns = 1 << (min(8 if parallel_projection else 32, meta['channels'], threads).bit_length() - 1)
    projection_bytes = len(graph['weight_members']) * meta['num_elements'] * projection_columns * item_bytes
    share_projection = (not shared_slots
                        and mixed_members and projection_bytes <= min(48 * 1024, shared_budget))
    packet_bases = [tuple(dict.fromkeys(graph['roots'][root]['selected_basis_id'] for root in packet['roots']))
                    for packet in packets]
    projection_capacity = max(map(len, packet_bases), default=0)
    packet_projection_bytes = projection_capacity * meta['num_elements'] * projection_columns * item_bytes
    # FP64 static projection arrays must stay within CUDA's 48 KiB allocation.
    projection_budget = min(48 * 1024, shared_budget) if scalar == 'double' else shared_budget
    share_packets = (not share_projection and mixed_members
                     and shared_bytes + packet_projection_bytes <= projection_budget)
    projected = share_projection or share_packets
    atom_groups = max(1, min(8, 32 // outputs)) if share_packets else 1
    members, member_ranges = [], []
    for mapping in graph['weight_members']:
        start = len(members)
        members.extend(mapping)
        member_ranges.append((start, len(members)))
    code = [rf'''#include <cuda_runtime.h>
#include <cstdint>
namespace symmetric_forward_register {{
using Element = {scalar};
constexpr int Threads = {threads};
constexpr int Channels = {meta["channels"]};
constexpr int NumElements = {meta["num_elements"]};
constexpr int64_t InputSize = {meta["input_size"]};
constexpr int64_t OutputSize = {meta["output_dim"]};
constexpr int64_t WeightSize = {meta["weight_numel"]};
struct Output {{ {' '.join(f'Element c{i};' for i in range(outputs))} }};
struct Member {{ int path; Element coefficient; }};''']
    compact_members = len(members) > CG_MEMBER_LITERAL_LIMIT
    member_value = 'load_cg_member(member_id)' if compact_members else 'Members[member_id]'
    member_type = 'CgMember' if compact_members else 'Member'
    if mixed_members:
        if compact_members:
            code.append(cg_member_constants(members, item_bytes))
        else:
            entries = ','.join('{'+str(path)+f',Element({coefficient:.17g})'+'}' for path,coefficient in members) or '{0,Element(0)}'
            code.append('__device__ const Member Members[] = {'+entries+'};')
        code.append('__device__ const int64_t PathOffsets[] = {'+','.join(map(str,meta['path_offsets']))+'};')
    if projected:
        bases = tuple(range(len(member_ranges))) if share_projection else tuple(basis for packet in packet_bases for basis in packet)
        capacity = len(member_ranges) if share_projection else projection_capacity
        code.append(rf'''constexpr int ProjectionColumns = {projection_columns};
constexpr int ProjectionRows = Threads / ProjectionColumns;
constexpr int ProjectionBases = {capacity};
constexpr int ProjectionAtoms = ProjectionRows * {atom_groups};''')
        if share_packets:
            code.append('__device__ const int ProjectionBasisIds[] = {' + ','.join(map(str, bases)) + '};')
        code += ['__device__ const int MemberOffsets[] = {' + ','.join(str(begin) for begin, _ in member_ranges) + f',{len(members)}' + '};',
                 '__device__ const int BasisChannels[] = {' + ','.join(str(min((meta['path_channels'][path] for path, _ in mapping), default=0)) for mapping in graph['weight_members']) + '};']
        if parallel_projection:
            # A warp divides the long member reduction between channel groups.
            basis_index = 'ProjectionBasisIds[first_basis + local_basis]' if share_packets else 'local_basis'
            code.append(r'''__device__ __forceinline__ void project_weights(const Element* weights, Element* projected, int channel_begin, int first_basis, int basis_count) {
    const int column = threadIdx.x % ProjectionColumns;
    const int worker = (threadIdx.x % 32) / ProjectionColumns;
    constexpr int workers = 32 / ProjectionColumns;
    const int channel = channel_begin + column;
    for (int job = threadIdx.x / 32; job < basis_count * NumElements; job += Threads / 32) {
        const int element = job % NumElements;
        const int local_basis = job / NumElements;
        const int basis = /* basis */;
        Element value = 0;
        if (channel < BasisChannels[basis]) {
            #pragma unroll 1
            for (int member_id = MemberOffsets[basis] + worker; member_id < MemberOffsets[basis + 1]; member_id += workers) {
                const /* member type */ member = /* member */;
                value = fma(member.coefficient, weights[int64_t(element) * WeightSize + PathOffsets[member.path] + channel], value);
            }
        }
        #pragma unroll
        for (int offset = 16; offset >= ProjectionColumns; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (worker == 0) projected[(local_basis * NumElements + element) * ProjectionColumns + column] = value;
    }
}'''.replace('/* basis */', basis_index).replace('/* member type */', member_type).replace('/* member */', member_value))
        else:
            basis_index = 'point / (ProjectionColumns * NumElements)'
            if share_packets:
                basis_index = f'ProjectionBasisIds[first_basis + {basis_index}]'
            code.append(r'''__device__ __forceinline__ void project_weights(const Element* weights, Element* projected, int channel_begin, int first_basis, int basis_count) {
    for (int point = threadIdx.x; point < basis_count * NumElements * ProjectionColumns; point += Threads) {
        const int column = point % ProjectionColumns;
        const int element = point / ProjectionColumns % NumElements;
        const int basis = /* basis */;
        const int channel = channel_begin + column;
        Element value = 0;
        if (channel < BasisChannels[basis]) {
            #pragma unroll 1
            for (int member_id = MemberOffsets[basis]; member_id < MemberOffsets[basis + 1]; ++member_id) {
                const /* member type */ member = /* member */;
                value = fma(member.coefficient, weights[int64_t(element) * WeightSize + PathOffsets[member.path] + channel], value);
            }
        }
        projected[point] = value;
    }
}'''.replace('/* basis */', basis_index).replace('/* member type */', member_type).replace('/* member */', member_value))
    code.append(local_cg_functions(meta))
    if shared_slots:
        code.append(shared_cg_functions(meta))
    for packet_id, packet in enumerate(packets):
        shared = packet['shared']
        extra = ', Element* values, int worker' if shared else ''
        if projected:
            extra += ', const Element* projected_weights, int column'
        code.append(f'__device__ __noinline__ Output packet_{packet_id}(const Element* x, const Element* weights, const Element* attrs, int64_t atom, int channel{extra}) {{')
        if not shared:
            code.append(f'    Element values[{packet["slots"]}];')
        code.append('    Output output = {};')
        if projected:
            code.append(r'''    Element attributes[NumElements];
    #pragma unroll
    for (int element = 0; element < NumElements; ++element)
        attributes[element] = attrs[atom * NumElements + element];''')
        for component in range(meta['input_dim']):
            offset,stride,channels = (meta[field][component] for field in ('input_offsets','input_strides','input_channels'))
            address=f'{component}*Threads+worker' if shared else str(component)
            code.append(f'    values[{address}] = channel < {channels} ? x[atom * InputSize + {offset} + int64_t(channel) * {stride}] : Element(0);')
        for operation in packet['operations']:
            if operation[0]=='cg':
                _,block,left,right,destination=operation
                call=f'shared_cg_block_{block}(values,{left},{right},{destination},worker)' if shared else f'cg_block_{block}(values,{left},{right},{destination})'
                code.append(f'    {call};')
                continue
            _, root_id, value = operation
            root = graph['roots'][root_id]
            basis = root['selected_basis_id']
            component = root['output_component_offset']
            dimension = graph['nodes'][root['node_id']]['irrep_dim']
            begin, end = member_ranges[basis]
            attribute = 'attributes[element]' if projected else 'attrs[atom * NumElements + element]'
            code.append(r'''    {
        Element mixed = 0;
        if (channel < /* channels */) {
            /* unroll */
            for (int element = 0; element < NumElements; ++element) {
                const Element a = /* attribute */;
                if (a != Element(0)) {'''.replace('/* channels */', str(meta['output_channels'][component]))
                .replace('/* unroll */', '#pragma unroll' if projected else '#pragma unroll 1')
                .replace('/* attribute */', attribute))
            if projected:
                position = basis if share_projection else packet_bases[packet_id].index(basis)
                code.append(f'                    const Element projected = projected_weights[({position} * NumElements + element) * ProjectionColumns + column];')
            elif end == begin:
                code.append('                    const Element projected = Element(0);')
            elif end - begin == 1:
                path, coefficient = members[begin]
                code.append(f'                    const Element projected = Element({coefficient:.17g}) * weights[int64_t(element) * WeightSize + {meta["path_offsets"][path]} + channel];')
            else:
                code.append(rf'''                    Element projected = 0;
                    #pragma unroll 1
                    for (int member_id = {begin}; member_id < {end}; ++member_id) {{
                        const {member_type} member = {member_value};
                        projected = fma(member.coefficient, weights[int64_t(element) * WeightSize + PathOffsets[member.path] + channel], projected);
                    }}''')
            code.append(r'''                    mixed = fma(a, projected, mixed);
                }
            }
        }''')
            for index in range(dimension):
                address=f'{value+index}*Threads+worker' if shared else str(value+index)
                code.append(f'        output.c{component+index} = fma(mixed, values[{address}], output.c{component+index});')
            code.append('    }')
        code += ['    return output;','}']
    precision = 'fp64' if scalar == 'double' else 'fp32'
    code.append(f'__global__ void symmetric_contraction_fwd_register_{precision}(const Element* x,const Element* weights,const Element* attrs,Element* result,int64_t atoms) {{')
    if shared_slots:
        code.append('    extern __shared__ Element shared_values[];')
    if projected:
        code.append(r'''    __shared__ Element projected_weights[ProjectionBases * NumElements * ProjectionColumns];
    const int column = threadIdx.x % ProjectionColumns;
    const int channel = blockIdx.y * ProjectionColumns + column;''')
    if share_packets:
        code.append(rf'''    for (int packet = blockIdx.z; packet < {len(packets)}; packet += gridDim.z) {{
        switch (packet) {{''')
        first_basis = 0
        for index, packet in enumerate(packets):
            extra = ',shared_values,threadIdx.x' if packet['shared'] else ''
            extra += ',projected_weights,column'
            code.append(rf'''        case {index}: {{
            project_weights(weights, projected_weights, blockIdx.y * ProjectionColumns, {first_basis}, {len(packet_bases[index])});
            __syncthreads();
            if (channel < Channels) {{
                for (int64_t atom = int64_t(blockIdx.x) * ProjectionRows + threadIdx.x / ProjectionColumns; atom < atoms; atom += int64_t(gridDim.x) * ProjectionRows) {{
                    const Output part = packet_{index}(x,weights,attrs,atom,channel{extra});''')
            for output in range(outputs):
                offset, stride, channels = (meta[field][output] for field in ('output_offsets','output_strides','output_channels'))
                code.append(f'                    if (channel < {channels}) atomicAdd(result + atom * OutputSize + {offset} + int64_t(channel) * {stride}, part.c{output});')
            code.append(r'''                }
            }
            __syncthreads();
            break;
        }''')
            first_basis += len(packet_bases[index])
        code += ['        }', '    }']
    else:
        if share_projection:
            code.append(r'''    project_weights(weights, projected_weights, blockIdx.y * ProjectionColumns, 0, ProjectionBases);
    __syncthreads();
    for (int64_t atom = int64_t(blockIdx.x) * ProjectionRows + threadIdx.x / ProjectionColumns; atom < atoms; atom += int64_t(gridDim.x) * ProjectionRows) {
        if (channel >= Channels) continue;
        Output sum = {};''')
        else:
            code.append(r'''    const int64_t tasks = atoms * Channels;
    for (int64_t point = int64_t(blockIdx.x) * Threads + threadIdx.x; point < tasks; point += int64_t(gridDim.x) * Threads) {
        const int64_t atom = point / Channels;
        const int channel = point % Channels;
        Output sum = {};''')
        for index, packet in enumerate(packets):
            extra = ',shared_values,threadIdx.x' if packet['shared'] else ''
            if projected:
                extra += ',projected_weights,column'
            code += ['        {', f'            const Output part = packet_{index}(x,weights,attrs,atom,channel{extra});']
            code.extend(f'            sum.c{output} += part.c{output};' for output in range(outputs))
            code.append('        }')
        for output in range(outputs):
            offset, stride, channels = (meta[field][output] for field in ('output_offsets','output_strides','output_channels'))
            code.append(f'        if (channel < {channels}) result[atom * OutputSize + {offset} + int64_t(channel) * {stride}] = sum.c{output};')
        code.append('    }')
    code += ['}', '}']
    device='\n'.join(code)+'\n'
    if projected:
        resident_blocks = 4 * meta["multiprocessor_count"]
        if share_packets:
            channel_tiles = (meta["channels"] + projection_columns - 1) // projection_columns
            resident_blocks = max(1, resident_blocks // (len(packets) * channel_tiles))
        grid = (f'const int blocks = int(std::min<int64_t>((atoms + ProjectionAtoms - 1) / ProjectionAtoms, {resident_blocks}));\n'
                '        const dim3 grid(blocks, (Channels + ProjectionColumns - 1) / ProjectionColumns'
                + (f', {min(65535, len(packets))}' if share_packets else '') + ');')
    else:
        grid = f'const int blocks=int(std::min<int64_t>((atoms*Channels+Threads-1)/Threads,{MAX_BLOCKS}));'
    shared_setup = ''
    if shared_slots:
        condition = 'true' if share_packets and shared_bytes + packet_projection_bytes > 48 * 1024 else f'{shared_bytes}>48*1024'
        shared_setup = f'\n        if ({condition}) C10_CUDA_CHECK(cudaFuncSetAttribute(symmetric_contraction_fwd_register_/* precision */,cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_bytes}));'
    allocation = 'zeros' if share_packets else 'empty'
    host=r'''
#include <ATen/core/Tensor.h>
/* allocation header */
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
namespace symmetric_forward_register {
at::Tensor run(at::Tensor x,at::Tensor weights,at::Tensor attrs) {
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && weights.is_cuda() && attrs.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weights.is_contiguous() && attrs.is_contiguous(), "inputs must be contiguous");
    const int64_t atoms=x.size(0);
    TORCH_CHECK(x.numel() == atoms * InputSize, "input shape does not match contraction configuration");
    TORCH_CHECK(weights.numel() == int64_t(NumElements) * WeightSize, "weight shape does not match contraction configuration");
    TORCH_CHECK(attrs.dim() == 2 && attrs.size(0) == atoms && attrs.size(1) == NumElements,
                "attribute shape does not match contraction configuration");
    auto output=at::/* allocation */({atoms,OutputSize},x.options());
    if (atoms) {/* shared setup */
        /* grid */
        symmetric_contraction_fwd_register_/* precision */<<</* launch grid */,Threads,/* shared bytes */,at::cuda::getCurrentCUDAStream(x.get_device())>>>(x.data_ptr<Element>(),weights.data_ptr<Element>(),attrs.data_ptr<Element>(),output.data_ptr<Element>(),atoms);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) { module.def("run",&symmetric_forward_register::run); }
'''
    host = (host.replace('/* allocation header */', f'#include <ATen/ops/{allocation}.h>')
            .replace('/* allocation */', allocation).replace('/* shared setup */', shared_setup)
            .replace('/* grid */', grid).replace('/* launch grid */', 'grid' if projected else 'blocks')
            .replace('/* shared bytes */', str(shared_bytes)))
    return device + host.replace('/* precision */', precision)


def make_forward_source(meta, dtype="float32"):
    """Emit a reduced-pair or polynomial forward, or fall back to CG packets."""
    scalar = "double" if str(dtype) in ("float64", "torch.float64", "double") else "float"
    reduced_entries = None
    if scalar == "float" and meta.get("use_reduced_cg") and "cg_dag" in meta and meta["correlation"] <= 3:
        groups, weights, limits = reduced_pair_groups(meta)
        meta = dict(meta, binary=dict(compute_weight_paths=weights))
        reduced_entries = []
        offset = meta["input_dim"]
        for pair, consumers in groups:
            present = {component for component, _, _ in pair}
            entries = []
            for basis, outer in consumers:
                terms = tuple(
                    (output, (offset + inner,) if raw is None else (raw, offset + inner), value)
                    for output, row in outer.items()
                    for raw, inner, value in row if inner in present)
                entries.append((basis, terms))
            reduced_entries.extend((pair, entries[first:first + 8]) for first in range(0, len(entries), 8))
    elif "cg_dag" in meta:
        identity = all(len(members) == 1 and tuple(members[0]) == (basis, 1.0)
                       for basis, members in enumerate(meta["cg_dag"]["weight_members"]))
        if identity and str(dtype) in (
            "float", "float32", "torch.float32", "float64", "torch.float64", "double"
        ):
            polynomial_meta = bounded_cg_polynomials(meta)
            if polynomial_meta is not None:
                meta = polynomial_meta
        if "cg_dag" in meta:
            return make_cg_forward_source(meta,dtype)
    binary = meta.get("binary")
    use_binary = reduced_entries is None and binary is not None and meta["correlation"] <= 3
    if not use_binary and reduced_entries is None:
        weights = meta.get("polynomials", {}).get("compute_weight_paths")
        if weights is None:
            weights = tuple(((path, 1.0),) for path in range(meta["num_paths"]))
        meta = dict(meta, binary=dict(compute_weight_paths=weights))
    item_bytes = 8 if scalar == "double" else 4
    binary = meta["binary"]
    inputs = meta.get("input_channels", (meta["channels"],) * meta["input_dim"])
    output_channels = meta.get("output_channels", (meta["channels"],) * len(meta["output_offsets"]))
    if reduced_entries is not None:
        windows = pack_forward_windows(reduced_entries, binary["compute_weight_paths"])
    else:
        limits = defaultdict(int)
        terms = meta["terms"] if use_binary else meta.get("polynomials", meta)["terms"]
        for output, path, components, _ in terms:
            limits[path] = max(limits[path], min(output_channels[output], min(inputs[i] for i in components)))
        if use_binary:
            limits = {entry["basis_path"]: limits[entry["path"]] for entry in binary["selected_paths"]}
        windows = binary_windows(meta) if use_binary else polynomial_windows(meta)
    weight_work = max((sum(len(binary["compute_weight_paths"][basis])
                           for _, consumers in window for basis, _ in consumers)
                       for window in windows), default=0)
    weight_work = max(1, meta["num_elements"] * weight_work)
    weight_registers = max((sum(len(consumers) for _, consumers in window) for window in windows), default=0)
    pair_registers = max((len({term[0] for term in pair}) for window in windows for pair, _ in window), default=0)
    live_words = (meta["input_dim"] + len(output_channels) + weight_registers + pair_registers) * (item_bytes // 4) + 16
    register_attributes = meta["num_elements"] * item_bytes <= 16 and live_words <= 64
    input_stride = meta["input_dim"] + (meta["input_dim"] % 2 == 0)
    shared_bytes = SHARED_MEMORY_BYTES - (meta["input_dim"] * item_bytes)
    available = (shared_bytes - meta["num_elements"] * (item_bytes + 4) - 16) // (input_stride * item_bytes)
    alignment = min(count & -count for count in (*inputs, *output_channels))
    work_rows = max(1, (FORWARD_WINDOW_WORK * 4 // item_bytes) // weight_work)
    row_threads = max(16, min(32, alignment, 16 * work_rows))
    # For a multi-pass scan, use one warp per row when it can cover a full
    # channel tile. Keep several warps per block even if dense work is large.
    full_warp_scan = meta["num_elements"] > row_threads and min(alignment, available) >= 32
    if full_warp_scan:
        row_threads = 32
    block_threads = max(32, BLOCK_SIZE * row_threads // 16 * 4 // item_bytes)
    columns = min(row_threads, max(1, available))
    row_bytes = columns * input_stride * item_bytes + meta["num_elements"] * (item_bytes + 4) + 16
    min_rows = BLOCK_SIZE * 4 // item_bytes // row_threads if full_warp_scan else 32 // row_threads
    rows = min(block_threads // row_threads, max(1, shared_bytes // row_bytes),
               max(min_rows, work_rows))
    threads = row_threads * (1 << (rows.bit_length() - 1))
    shuffle_attributes = (not register_attributes and columns == row_threads and alignment >= row_threads
                          and meta["num_elements"] <= row_threads)
    contractions = None
    boundaries = sorted({0, *limits.values(), meta["channels"]})
    if len(boundaries) > 2:
        branches = []
        for upper in boundaries[1:]:
            keep = {basis for basis, limit in limits.items() if limit >= upper}
            if reduced_entries is not None:
                entries = [(pair, [item for item in consumers if item[0] in keep])
                           for pair, consumers in reduced_entries]
                band_windows = pack_forward_windows([entry for entry in entries if entry[1]], binary["compute_weight_paths"])
            elif use_binary:
                pairs = [dict(pair, consumers=[item for item in pair["consumers"] if item[0] in keep])
                         for pair in binary["pair_groups"]]
                filtered = dict(binary, selected_paths=[path for path in binary["selected_paths"] if path["basis_path"] in keep],
                                pair_groups=[pair for pair in pairs if pair["consumers"]])
                band_windows = binary_windows(dict(meta, binary=filtered))
            else:
                band_windows = polynomial_windows(meta, keep)
            body = contraction_loop(meta, band_windows, limits, upper, scalar, register_attributes, shuffle_attributes)
            branch = ("if" if not branches else "else if") + f" (channel < {upper}) {{"
            branches.append(branch + "\n" + indent(body, " " * 4) + "\n}")
        contractions = "\n".join(branches)
    if contractions is None:
        contractions = contraction_loop(meta, windows, limits, scalar=scalar,
                                        register_attributes=register_attributes, shuffle_attributes=shuffle_attributes)

    source = CUDA_SOURCE.replace("                /* outputs */", "\n".join(f"                Element out_{m} = 0;" for m in range(len(output_channels))))
    stores = [(f"if (channel < {output_channels[m]}) " if output_channels[m] < meta["channels"] else "") +
              f"output[atom * OutputDim + {offset} + Index(channel) * {stride}] = out_{m};"
              for m, (offset, stride) in enumerate(zip(meta["output_offsets"], meta["output_strides"]))]
    source = source.replace("                /* contractions */", indent(contractions, " " * 16))
    source = source.replace("                /* stores */", indent("\n".join(stores), " " * 16))
    replacements = {key: meta[key] for key in ("channels", "input_dim", "output_dim", "num_elements")}
    resident_threads = min((FORWARD_RESIDENT_THREADS if meta["input_dim"] * item_bytes <= 64
                            else FORWARD_RESIDENT_THREADS * 3 // 4) * 4 // item_bytes, threads * 32)
    register_tile = (live_words + 15) // 16 * 16
    resident_threads = min(resident_threads, max(1, 65536 // (threads * register_tile)) * threads)
    replacements.update(element=scalar, threads=threads, row_threads=row_threads, channel_columns=columns,
                        resident_threads=resident_threads, max_blocks=MAX_BLOCKS,
                        register_attributes=str(register_attributes).lower(),
                        shuffle_attributes=str(shuffle_attributes).lower())
    for key, value in replacements.items():
        source = source.replace(f"/* {key} */", str(value))
    layout_constants = (f"\nconstexpr int InputSize = {meta['input_size']};\n"
                        f"constexpr int WeightSize = {meta['weight_numel']};\n")
    layout_checks = """    TORCH_CHECK(x.dim() == 2 && x.size(1) == InputSize,
                "x must have shape [atoms, packed input components]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == NumElements && weights.size(1) == WeightSize,
                "weights must have shape [elements, packed weights]");
"""
    uniform = len(set((*inputs, *output_channels))) == 1
    input_loads = ["                Element input[InputDim];"]
    if uniform:
        input_loads.append("                const Element* channel_input = x + atom * InputSize + channel;")
    component = offset = 0
    for channels, irrep in o3.Irreps(meta["irreps_in"]):
        for m in range(irrep.dim):
            value = (f"__ldg(channel_input + {offset + m} + Index(channel) * {irrep.dim - 1})" if uniform else
                     f"x[atom * InputSize + {offset} + Index(channel) * {irrep.dim} + {m}]")
            if channels < meta["channels"]:
                value = f"channel < {channels} ? {value} : Element(0)"
            input_loads.append(f"                input[{component + m}] = {value};")
        component += irrep.dim
        offset += channels * irrep.dim
    if uniform and full_warp_scan:
        input_loads = [line.replace("__ldg", "__ldcs") for line in input_loads]
    attribute_mask = ""
    if register_attributes:
        attribute_mask = """
                attribute_mask = __ballot_sync(lanes, value != Element(0)) >> warp_offset;
                lane_attribute = __shfl_sync(lanes, value, attribute_mask ? __ffs(attribute_mask) - 1 : 0, RowThreads);"""
        input_loads[:0] = ["                active_count = __popc(attribute_mask);",
                          "                first_element = attribute_mask ? __ffs(attribute_mask) - 1 : 0;",
                          "                first_attribute = lane_attribute;"]
    source = source.replace("/* register attribute mask */", attribute_mask)
    source = (source.replace("/* layout constants */", layout_constants)
              .replace("/* layout checks */", layout_checks)
              .replace("                /* input loads */", "\n".join(input_loads)))
    return source.replace("/* precision */", f"fp{item_bytes * 8}")
