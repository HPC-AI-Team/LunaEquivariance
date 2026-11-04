"""Contraction layouts, static polynomial helpers and CUDA builds."""

from collections import defaultdict
import struct

import torch
from e3nn import o3

from ..couplings import build_couplings, build_polynomial_couplings, polynomial_dag
from ....utils.extension import BINDING_CFLAGS, combine_cuda_sources, get_cuda_arch
from ....utils.jit import load_extension

BLOCK_SIZE = 128
CHANNEL_TILE = 32
WEIGHT_GROUP_TILE = 4
MAX_BLOCKS = 65535
SHARED_MEMORY_BYTES = 48 * 1024
FORWARD_RESIDENT_THREADS = 1024
FORWARD_WINDOW_WORK = 1024

CG_PACKET_LIVE_SLOTS = 64
CG_PACKET_EDGE_BUDGET = 1024
CG_PACKET_ROOT_BUDGET = 32
CG_PRIVATE_REGISTER_WORDS = 255
CG_PACKET_REGISTER_OVERHEAD = 48
FORWARD_CG_THREADS = 128
CG_PLAN_ROWS = 4
CG_PLAN_CHANNEL_COLUMNS = 8
BACKWARD_CG_ROWS = 32
BACKWARD_CG_CHANNEL_COLUMNS = 1
BACKWARD_PRIVATE_REGISTER_WORDS = 192
BACKWARD_PACKET_EDGE_BUDGET = 512
BACKWARD_PACKET_ROOT_BUDGET = 8
BACKWARD_PARAMETER_CTA_WAVES = 4
BACKWARD_REGISTER_ROWS = 8
BACKWARD_REGISTER_PARAMETER_WORDS = 32
BACKWARD_REGISTER_LIVE_WORDS = 96
BACKWARD_REGISTER_TERM_BUDGET = 128
BACKWARD_REGISTER_CTA_WAVES = 6
BACKWARD_REGISTER_SINGLE_CTA_WAVES = 8
BACKWARD_FP64_REGISTER_PARAMETER_WORDS = 48
BACKWARD_FP64_REGISTER_CTA_WAVES = 4
BACKWARD_FP64_BLOCK_MIN_CTAS = 3
CG_SHARED_PACKET_BYTES = 24 * 1024
CG_SHARED_BUDGET_DIVISOR = 4
CG_CONSTANT_BYTES = 14 * 1024
CG_MEMBER_LITERAL_LIMIT = 8192
CG_ELEMENT_TILE = 4
TRIPLE_CG_NODE_TILE = 256
TRIPLE_CG_SHARED_BUDGET_DIVISOR = 2
CG_POLYNOMIAL_COMPONENT_TERMS = 128
CG_POLYNOMIAL_TOTAL_TERMS = 2048


def polynomial_groups(meta):
    basis = meta.get("polynomials")
    source = meta if basis is None else basis
    polynomials = defaultdict(list)
    for output, path, inputs, coefficient in source["terms"]:
        polynomials[path].append((output, tuple(inputs), coefficient))
    groups = defaultdict(list)
    count = meta["num_paths"] if basis is None else len(basis["compute_weight_paths"])
    for path in range(count):
        terms = tuple(sorted(polynomials[path]))
        sign = -1 if terms and terms[0][-1] < 0 else 1
        signature = tuple((output, inputs, coefficient * sign)
                          for output, inputs, coefficient in terms)
        if basis is None:
            groups[signature].append((path, sign))
        else:
            groups[signature].extend((original, sign * coefficient)
                                     for original, coefficient in basis["compute_weight_paths"][path])
    return tuple((terms, tuple(members)) for terms, members in groups.items())


def polynomial_expression(terms):
    if not terms:
        return "Element(0)"
    if not terms[0][0]:
        return f"Element({terms[0][1]:.17g})"
    groups = defaultdict(list)
    for inputs, coefficient in terms:
        groups[inputs[0]].append((inputs[1:], coefficient))
    expression = "Element(0)"
    for component, remaining in sorted(groups.items()):
        expression = f"fma(values[{component}], {polynomial_expression(remaining)}, {expression})"
    return expression


def bounded_cg_polynomials(meta):
    """Factor small static CG polynomials without a per-node arena."""
    graph = meta["cg_dag"]
    values = []
    total_terms = 0
    for node in graph["nodes"]:
        if node["kind"] == "input":
            values.append([{(node["input_start"] + component,): 1.}
                           for component in range(node["input_width"])])
            continue
        block = graph["local_cg"][node["cg_id"]]
        left, right = values[node["left_id"]], values[node["right_id"]]
        outputs = []
        for component in range(node["irrep_dim"]):
            terms = defaultdict(float)
            for edge in range(block["output_offsets"][component], block["output_offsets"][component + 1]):
                coefficient = block["coefficients"][edge]
                for first, a in left[block["left_components"][edge]].items():
                    for second, b in right[block["right_components"][edge]].items():
                        terms[tuple(sorted(first + second))] += coefficient * a * b
                if len(terms) > CG_POLYNOMIAL_COMPONENT_TERMS:
                    return None
            terms = {indices: coefficient for indices, coefficient in terms.items() if coefficient != 0.}
            total_terms += len(terms)
            if total_terms > CG_POLYNOMIAL_TOTAL_TERMS:
                return None
            outputs.append(terms)
        values.append(outputs)
    terms = []
    for root in graph["roots"]:
        basis = root["selected_basis_id"]
        for component, polynomial in enumerate(values[root["node_id"]]):
            output = root["output_component_offset"] + component
            for indices, coefficient in polynomial.items():
                terms.append((output, basis, indices, coefficient))
    result = {key: value for key, value in meta.items() if key != "cg_dag"}
    result["terms"] = ()
    result["polynomials"] = dict(terms=tuple(terms), compute_weight_paths=graph["weight_members"])
    return result


def reduced_pair_groups(meta):
    """Keep the shared prefix of each linear, quadratic or cubic CG root."""
    graph = meta["cg_dag"]
    nodes, blocks = graph["nodes"], graph["local_cg"]
    prefixes, consumers, limits = {}, defaultdict(list), {}
    input_channels = [meta["input_channels"][node["input_start"]]
                      for node in nodes if node["kind"] == "input"]
    for root in graph["roots"]:
        node_id, degree = root["node_id"], root["degree"]
        node = nodes[node_id]
        prefix_id = node["left_id"] if degree == 3 else node_id
        prefix = nodes[prefix_id]
        if prefix_id not in prefixes:
            if prefix["kind"] == "input":
                prefixes[prefix_id] = tuple((component, (prefix["input_start"] + component,), 1.)
                                            for component in range(prefix["input_width"]))
            else:
                block = blocks[prefix["cg_id"]]
                left, right = (nodes[prefix[key]]["input_start"] for key in ("left_id", "right_id"))
                terms = defaultdict(float)
                for component, (begin, end) in enumerate(zip(block["output_offsets"], block["output_offsets"][1:])):
                    for edge in range(begin, end):
                        indices = tuple(sorted((left + block["left_components"][edge],
                                                right + block["right_components"][edge])))
                        terms[component, indices] += block["coefficients"][edge]
                prefixes[prefix_id] = tuple((component, indices, value)
                                            for (component, indices), value in sorted(terms.items()) if value != 0.)
        basis, offset = root["selected_basis_id"], root["output_component_offset"]
        outer = {}
        if degree == 3:
            block = blocks[node["cg_id"]]
            right = nodes[node["right_id"]]["input_start"]
            for component, (begin, end) in enumerate(zip(block["output_offsets"], block["output_offsets"][1:])):
                outer[offset + component] = [(right + block["right_components"][edge],
                                               block["left_components"][edge], block["coefficients"][edge])
                                              for edge in range(begin, end)]
        else:
            outer = {offset + component: [(None, component, 1.)] for component in range(node["irrep_dim"])}
        consumers[prefix_id].append((basis, outer))
        limits[basis] = min(meta["output_channels"][offset],
                            *(count for count, used in zip(input_channels, root["composition"]) if used))
    groups = [(prefixes[prefix], members) for prefix, members in consumers.items()]
    return groups, graph["weight_members"], limits


def build_cg_plan(meta, planes_per_slot=1, shared_bytes=None, rows=CG_PLAN_ROWS, channel_columns=CG_PLAN_CHANNEL_COLUMNS):
    """Keep the CG values required by each packet's backward computation."""
    graph = meta["cg_dag"]
    nodes, roots = graph["nodes"], graph["roots"]
    dtype = meta.get("source_dtype", "float32")
    item_bytes = 8 if dtype == "float64" else 4
    shared_bytes = shared_bytes or meta.get("shared_memory_bytes", SHARED_MEMORY_BYTES)
    input_dim = meta["input_dim"]
    closures = []
    for index, node in enumerate(nodes):
        if node["kind"] == "input":
            closures.append(frozenset())
        else:
            left, right = node["left_id"], node["right_id"]
            if not 0 <= left < index or not 0 <= right < index:
                raise ValueError("CG nodes must be in topological order")
            closures.append(closures[left] | closures[right] | {index})

    def slots(indices):
        return input_dim + sum(nodes[index]["irrep_dim"] for index in indices)

    largest_root = max((slots(closures[root["node_id"]]) for root in roots), default=input_dim)
    while largest_root * (rows * channel_columns + 1) * planes_per_slot * item_bytes > shared_bytes:
        if rows > 1:
            rows //= 2
        elif channel_columns > 1:
            channel_columns //= 2
        else:
            raise ValueError(f"CG root needs {largest_root} slots and cannot fit in {shared_bytes} shared bytes")
    workers = rows * channel_columns
    value_stride = workers + 1
    slot_budget = shared_bytes // (planes_per_slot * value_stride * item_bytes)
    root_groups = defaultdict(list)
    for index, root in enumerate(roots):
        node = nodes[root["node_id"]]
        prefix = node["left_id"] if node["kind"] == "coupling" else root["node_id"]
        root_groups[prefix].append(index)

    packets, packet_roots, closure = [], [], set()
    for group in root_groups.values():
        for root in group:
            candidate = closure | closures[roots[root]["node_id"]]
            if slots(candidate) > slot_budget and packet_roots:
                packets.append((tuple(sorted(closure)), tuple(packet_roots)))
                packet_roots, closure = [], set()
                candidate = set(closures[roots[root]["node_id"]])
            if slots(candidate) > slot_budget:
                raise ValueError(f"CG root {root} needs {slots(candidate)} slots; budget allows {slot_budget}")
            packet_roots.append(root)
            closure = candidate
    if packet_roots:
        packets.append((tuple(sorted(closure)), tuple(packet_roots)))

    cg_blocks, cg_edges = [], []
    for block in graph["local_cg"]:
        begin = len(cg_edges)
        offsets = block["output_offsets"]
        for component in range(block["shape"][0]):
            for edge in range(offsets[component], offsets[component + 1]):
                cg_edges.append((component, block["left_components"][edge],
                                 block["right_components"][edge], block["coefficients"][edge]))
        cg_blocks.append((begin, len(cg_edges), block["shape"][0]))

    cg_nodes, cg_roots, cg_members, cg_packets = [], [], [], []
    max_slots = input_dim
    for internal, selected in packets:
        bases = {index: node["input_start"] for index, node in enumerate(nodes) if node["kind"] == "input"}
        next_slot = input_dim
        node_begin, root_begin = len(cg_nodes), len(cg_roots)
        for index in internal:
            node = nodes[index]
            bases[index] = next_slot
            cg_nodes.append((bases[node["left_id"]], bases[node["right_id"]], next_slot, node["cg_id"]))
            next_slot += node["irrep_dim"]
        for index in selected:
            root = roots[index]
            basis = root["selected_basis_id"]
            begin = len(cg_members)
            cg_members.extend(graph["weight_members"][basis])
            cg_roots.append((bases[root["node_id"]], nodes[root["node_id"]]["irrep_dim"],
                             root["output_component_offset"], basis, begin, len(cg_members)))
        cg_packets.append((node_begin, len(cg_nodes), root_begin, len(cg_roots), next_slot))
        max_slots = max(max_slots, next_slot)

    channels = meta["channels"]
    uniform = meta.get("input_layout", "channels") == "channels"
    if not uniform and any(key not in meta for key in ("input_offsets", "input_strides", "input_channels")):
        raise ValueError("compact CG input layout requires explicit component addresses")
    input_offsets = tuple(meta.get("input_offsets", range(input_dim)))
    input_strides = tuple(meta.get("input_strides", (input_dim,) * input_dim))
    input_channels = tuple(meta.get("input_channels", (channels,) * input_dim))
    output_components = len(meta["output_offsets"])
    paths = meta["num_paths"]
    arrays = dict(CgNodes=cg_nodes, CgBlocks=cg_blocks, CgEdges=cg_edges,
                  CgRoots=cg_roots, CgMembers=cg_members, CgPackets=cg_packets,
                  InputOffsets=input_offsets, InputStrides=input_strides, InputChannels=input_channels,
                  OutputOffsets=tuple(meta["output_offsets"]), OutputStrides=tuple(meta["output_strides"]),
                  OutputChannels=tuple(meta["output_channels"]),
                  PathOffsets=tuple(meta.get("path_offsets", (path * channels for path in range(paths)))),
                  PathChannels=tuple(meta.get("path_channels", (channels,) * paths)))
    constants = dict(Workers=workers, Rows=rows, ChannelColumns=channel_columns,
                     ValueStride=value_stride, MaxSlots=max_slots, PacketCount=len(cg_packets),
                     Channels=channels, InputDim=input_dim,
                     InputSize=meta.get("input_size", input_dim * channels),
                     OutputComponents=output_components, OutputSize=meta["output_dim"],
                     NumElements=meta["num_elements"], WeightSize=meta.get("weight_numel", paths * channels))
    return dict(constants=constants, arrays=arrays, shared_bytes=max_slots * value_stride * planes_per_slot * item_bytes,
                slot_budget=slot_budget, planes_per_slot=planes_per_slot, source_dtype=dtype, item_bytes=item_bytes)


def cg_member_constants(values, item_bytes):
    """Store parameter indices and coefficients as exact bit patterns."""
    storage = "__device__ const"
    values = values or ((0, 0.0),)
    largest_path = max(int(path) for path, _ in values)
    path_type = "int" if largest_path < 1 << 31 else "long long"
    lines = [f"struct CgMember {{ {path_type} path; Element coefficient; }};"]
    if item_bytes == 4 and largest_path < 1 << 32:
        words = []
        for path, coefficient in values:
            bits = struct.unpack("<I", struct.pack("<f", float(coefficient)))[0]
            words.append(f"0x{(bits << 32) | int(path):016x}ull")
        lines += [f"{storage} unsigned long long CgMemberBits[]={{\n" + ",\n".join(words) + "\n};",
                  "__device__ __forceinline__ CgMember load_cg_member(int index) {",
                  "    const unsigned long long bits = CgMemberBits[index];",
                  f"    return {{static_cast<{path_type}>(static_cast<unsigned>(bits)),",
                  "            __uint_as_float(static_cast<unsigned>(bits >> 32))};", "}"]
    else:
        coefficient_type, pack_type, unpack_type = (("unsigned", "<f", "<I") if item_bytes == 4
                                                   else ("unsigned long long", "<d", "<Q"))
        coefficients = [struct.unpack(unpack_type, struct.pack(pack_type, float(coefficient)))[0]
                        for _, coefficient in values]
        paths = ",\n".join(str(int(path)) for path, _ in values)
        coefficients = ",\n".join(f"0x{bits:x}" + ("u" if item_bytes == 4 else "ull")
                                    for bits in coefficients)
        lines += [f"{storage} {path_type} CgMemberPaths[]={{\n{paths}\n}};",
                  f"{storage} {coefficient_type} CgMemberBits[]={{\n{coefficients}\n}};",
                  "__device__ __forceinline__ CgMember load_cg_member(int index) {"]
        value = ("__uint_as_float(CgMemberBits[index])" if item_bytes == 4 else
                 "__longlong_as_double(static_cast<long long>(CgMemberBits[index]))")
        lines += [f"    return {{CgMemberPaths[index], {value}}};", "}"]
    return "\n".join(lines)


def cg_plan_constants(plan, constant_budget=0):
    """Emit data tables, leaving the caller's Element typedef and CUDA body intact."""
    records = {
        "CgNodes": ("CgNode", "int left,right,value,block;", 4),
        "CgBlocks": ("CgBlock", "int begin,end,dimension;", 3),
        "CgEdges": ("CgEdge", "int output,left,right; Element coefficient;", 4),
        "CgRoots": ("CgRoot", "int value,dimension,output,basis,member_begin,member_end;", 6),
        "CgMembers": ("CgMember", "int path; Element coefficient;", 2),
        "CgPackets": ("CgPacket", "int node_begin,node_end,root_begin,root_end,slots;", 5),
    }
    constant_budget = min(constant_budget, CG_CONSTANT_BYTES)
    record_bytes = dict(CgNodes=16, CgBlocks=12, CgRoots=24, CgPackets=20,
                        CgEdges=24 if plan["item_bytes"] == 8 else 16,
                        CgMembers=16 if plan["item_bytes"] == 8 else 8)
    priority = ("CgRoots", "CgNodes", "CgPackets", "InputOffsets", "InputStrides", "InputChannels",
                "OutputOffsets", "OutputStrides", "OutputChannels", "PathOffsets", "PathChannels",
                "CgMembers", "CgBlocks", "CgEdges")
    constant_arrays = set()
    for name in priority:
        if name not in plan["arrays"]:
            continue
        element_bytes = record_bytes.get(name, 8 if name.endswith("Offsets") else 4)
        size = ((max(1, len(plan["arrays"][name])) * element_bytes + 15) // 16) * 16
        if size <= constant_budget:
            constant_arrays.add(name)
            constant_budget -= size
    lines = [f"constexpr long long {key}={value};" if key in ("InputSize", "OutputSize", "WeightSize")
             else f"constexpr int {key}={value};" for key, value in plan["constants"].items()]
    for name, values in plan["arrays"].items():
        storage = "__device__ __constant__ const" if name in constant_arrays else "__device__ const"
        if name in records:
            typename, fields, width = records[name]
            lines.append(f"struct {typename} {{ {fields} }};")
            rows = []
            for value in values or [(0,) * width]:
                parts = [f"Element({float(item):.17g})" if name in ("CgEdges", "CgMembers") and index == width - 1
                         else str(int(item)) for index, item in enumerate(value)]
                rows.append("{" + ",".join(parts) + "}")
            content = ",\n".join(rows)
            lines.append(f"{storage} {typename} {name}[]={{\n{content}\n}};")
        else:
            typename = "long long" if name in ("InputOffsets", "OutputOffsets", "PathOffsets") else "int"
            content = ",".join(str(int(value)) for value in values) or "0"
            lines.append(f"{storage} {typename} {name}[]={{ {content} }};")
    return "\n".join(lines)


def packet_intervals(meta, plan, packet):
    """Track CG values and gradient totals across forward, root and reverse events."""
    node_begin, node_end, root_begin, root_end, _ = packet
    nodes = plan["arrays"]["CgNodes"][node_begin:node_end]
    roots = plan["arrays"]["CgRoots"][root_begin:root_end]
    dimensions = {value: meta["cg_dag"]["local_cg"][block]["shape"][0]
                  for _, _, value, block in nodes}
    intervals = {}

    def touch(kind, base, event):
        if base not in dimensions:
            return
        key = kind, base
        if key not in intervals:
            intervals[key] = [event, event, dimensions[base]]
        intervals[key][1] = max(intervals[key][1], event)

    # Inputs occupy reserved slots; only temporaries enter the arena allocator.
    for event, (left, right, value, _) in enumerate(nodes):
        touch("value", value, event)
        touch("value", left, event)
        touch("value", right, event)
    # Roots consume forward values and seed the reverse gradient totals.
    for index, (value, *_) in enumerate(roots):
        event = len(nodes) + index
        touch("value", value, event)
        touch("total", value, event)
    for index, (left, right, value, _) in enumerate(reversed(nodes)):
        event = len(nodes) + len(roots) + index
        touch("value", left, event)
        touch("value", right, event)
        touch("total", value, event)
        touch("total", left, event)
        touch("total", right, event)
    return intervals


def allocate_intervals(intervals, prefix):
    """Reuse contiguous vector slots only after their final read or write."""
    allocation, active, peak = {}, [], prefix
    for key, (begin, end, width) in sorted(intervals.items(), key=lambda item: (item[1][0], -item[1][2], item[0])):
        # Keep slots alive through the event that last reads or writes them.
        active = [item for item in active if item[0] >= begin]
        offset = prefix
        for _, current, current_width in sorted(active, key=lambda item: item[1]):
            if offset + width <= current:
                break
            offset = max(offset, current + current_width)
        allocation[key] = offset
        active.append((end, offset, width))
        peak = max(peak, offset + width)
    return allocation, peak


def build_arena_plan(meta, planes_per_value=2, *, rows=None, channel_columns=None, shared_bytes=None):
    """Pack live values and gradients, retaining both input ranges across packets."""
    rows = CG_PLAN_ROWS if rows is None else rows
    columns = CG_PLAN_CHANNEL_COLUMNS if channel_columns is None else channel_columns
    shared_bytes = CG_SHARED_PACKET_BYTES if shared_bytes is None else shared_bytes
    while True:
        plan = build_cg_plan(meta, planes_per_slot=planes_per_value, shared_bytes=shared_bytes,
                             rows=rows, channel_columns=columns)
        constants = plan["constants"]
        prefix = 2 * constants["InputDim"]
        budget = shared_bytes // (planes_per_value * constants["ValueStride"] * plan["item_bytes"])
        packets, layouts, nodes_out, roots_out = [], [], [], []
        largest, failed = prefix, False

        def add_packet(nodes, roots, allocation, peak):
            packets.append((len(nodes_out), len(nodes_out) + len(nodes), len(roots_out),
                            len(roots_out) + len(roots), peak))
            nodes_out.extend(nodes)
            roots_out.extend(roots)
            layouts.append(allocation)

        for node_begin, node_end, root_begin, root_end, _ in plan["arrays"]["CgPackets"]:
            nodes = plan["arrays"]["CgNodes"][node_begin:node_end]
            lookup = {node[2]: node for node in nodes}

            def closure(roots):
                needed, pending = set(), [root[0] for root in roots]
                while pending:
                    value = pending.pop()
                    if value not in lookup or value in needed:
                        continue
                    needed.add(value)
                    pending.extend(lookup[value][:2])
                return [node for node in nodes if node[2] in needed]

            current, previous = [], None
            for root in plan["arrays"]["CgRoots"][root_begin:root_end]:
                trial = current + [root]
                selected = closure(trial)
                temporary = {"arrays": {"CgNodes": selected, "CgRoots": trial}}
                intervals = packet_intervals(meta, temporary, (0, len(selected), 0, len(trial), 0))
                allocation, peak = allocate_intervals(intervals, prefix)
                if peak > budget and current:
                    add_packet(*previous)
                    trial = [root]
                    selected = closure(trial)
                    temporary = {"arrays": {"CgNodes": selected, "CgRoots": trial}}
                    intervals = packet_intervals(meta, temporary, (0, len(selected), 0, 1, 0))
                    allocation, peak = allocate_intervals(intervals, prefix)
                if peak > budget:
                    failed = True
                    break
                current = trial
                previous = selected, trial, allocation, peak
                largest = max(largest, peak)
            if failed:
                break
            if previous is not None:
                add_packet(*previous)
        if not failed:
            arrays = dict(plan["arrays"], CgNodes=nodes_out, CgRoots=roots_out, CgPackets=packets)
            constants = dict(constants, MaxSlots=largest, PacketCount=len(packets))
            return dict(plan, arrays=arrays, constants=constants, layouts=layouts,
                        shared_bytes=largest * planes_per_value * constants["ValueStride"] * plan["item_bytes"])
        rows, columns = constants["Rows"], constants["ChannelColumns"]
        if rows > 1:
            rows //= 2
        elif columns > 1:
            columns //= 2
        else:
            raise ValueError("CG closure cannot fit one-worker shared state")


def build_metadata(irreps_in, irreps_out, correlation, num_elements, dtype, use_reduced_cg=False):
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    if not irreps_in or not irreps_out:
        raise ValueError("input and output irreps must be nonempty")
    if any(mul <= 0 for mul, _ in (*irreps_in, *irreps_out)):
        raise ValueError("input and output multiplicities must be positive")
    if correlation < 1 or num_elements < 1:
        raise ValueError("correlation and num_elements must be positive")
    dtype_name = str(dtype).removeprefix("torch.")
    meta = dict(build_couplings(tuple(str(irrep) for _, irrep in irreps_in),
                                tuple(str(irrep) for _, irrep in irreps_out),
                                correlation, dtype_name, use_reduced_cg=use_reduced_cg))
    uniform = len({mul for mul, _ in (*irreps_in, *irreps_out)}) == 1
    input_offsets, input_strides, input_channels = [], [], []
    component_offset = flat_offset = 0
    for channels, irrep in irreps_in:
        input_channels.extend([channels] * irrep.dim)
        input_offsets.extend(range(component_offset if uniform else flat_offset,
                                   (component_offset if uniform else flat_offset) + irrep.dim))
        input_strides.extend([meta["input_dim"] if uniform else irrep.dim] * irrep.dim)
        component_offset += irrep.dim
        flat_offset += channels * irrep.dim

    output_offsets, output_strides, output_channels, output_irreps = [], [], [], []
    path_offsets, path_channels = [], []
    flat_offset = weight_numel = 0
    for block, (channels, irrep) in enumerate(irreps_out):
        output_irreps.append((irrep.dim, flat_offset))
        output_offsets.extend(range(flat_offset, flat_offset + irrep.dim))
        output_strides.extend([irrep.dim] * irrep.dim)
        output_channels.extend([channels] * irrep.dim)
        paths = sum(meta["path_counts"][block * correlation:(block + 1) * correlation])
        for _ in range(paths):
            path_offsets.append(weight_numel)
            path_channels.append(channels)
            weight_numel += channels
        flat_offset += channels * irrep.dim
    meta.update(
        irreps_in=str(irreps_in), irreps_out=str(irreps_out), source_dtype=dtype_name,
        use_reduced_cg=use_reduced_cg,
        channels=max(mul for mul, _ in (*irreps_in, *irreps_out)), num_elements=num_elements,
        input_layout="channels" if uniform else "irreps", input_size=irreps_in.dim,
        input_offsets=tuple(input_offsets), input_strides=tuple(input_strides),
        input_channels=tuple(input_channels), output_dim=irreps_out.dim,
        output_offsets=tuple(output_offsets), output_strides=tuple(output_strides),
        output_channels=tuple(output_channels), output_irreps=tuple(output_irreps),
        path_offsets=tuple(path_offsets), path_channels=tuple(path_channels), weight_numel=weight_numel,
    )
    return meta


def load_kernels(meta, dtype, device):
    from .forward import make_forward_source
    from .backward import make_backward_source
    from .double_backward import make_double_backward_source
    from .triple_backward import make_triple_backward_source

    arch = get_cuda_arch(device)
    properties = torch.cuda.get_device_properties(device)
    compile_meta = dict(meta,
                        shared_memory_bytes=min(96 * 1024, properties.shared_memory_per_multiprocessor // 2),
                        multiprocessor_count=properties.multi_processor_count)
    if dtype == torch.float64 and meta["source_dtype"] == "float32":
        compile_meta.pop("binary", None)
        compile_meta.pop("polynomials", None)
        if not meta.get("use_reduced_cg", False) and "cg_dag" in meta:
            exact = build_polynomial_couplings(
                tuple(str(irrep) for _, irrep in o3.Irreps(meta["irreps_in"])),
                tuple(str(irrep) for _, irrep in o3.Irreps(meta["irreps_out"])),
                meta["correlation"], "float32",
            )
            if exact["num_paths"] != meta["num_paths"]:
                raise ValueError("original polynomial paths do not match the weight layout")
            compile_meta["cg_dag"] = polynomial_dag(exact)
    sources = dict(
        forward=make_forward_source(compile_meta, dtype),
        backward=make_backward_source(compile_meta, dtype),
        double_backward=make_double_backward_source(compile_meta, dtype),
        triple_backward=make_triple_backward_source(compile_meta, dtype),
    )
    source = combine_cuda_sources(sources).replace(
        "#include <torch/extension.h>",
        "#include <ATen/core/Tensor.h>\n#include <ATen/ops/empty.h>\n#include <ATen/ops/empty_like.h>\n#include <ATen/ops/zeros.h>\n#include <ATen/ops/zeros_like.h>\n#include <torch/csrc/utils/pybind.h>",
    )
    extension = load_extension(
        "symmetric_contraction", cuda_sources=source,
        extra_cflags=["-O3", *BINDING_CFLAGS],
        extra_cuda_cflags=["-O3", *BINDING_CFLAGS,
                          f"-gencode=arch=compute_{arch},code=sm_{arch}",
                          f"-gencode=arch=compute_{arch},code=compute_{arch}"],
        no_implicit_headers=True,
    )
    return extension
