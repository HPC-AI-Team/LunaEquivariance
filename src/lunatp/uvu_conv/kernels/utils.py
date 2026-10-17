"""UVU convolution path metadata, scheduling and CUDA helpers."""

from __future__ import annotations

import re
from math import prod

import torch

from ...utils.cg import build_cg_terms, wigner_3j
from ...utils.layout import UvuPathSpec, configure_indexing, l0_feature_layout

UVU_CONV_CHANNEL_CHUNK = 32
UVU_CONV_FWD_LIVE_OUTPUT_LIMIT = 16
# Keep the generated int2 tile map within 8 KiB of constant memory.
UVU_CONV_FWD_TILE_TABLE_LIMIT = 1024
UVU_CONV_NEIGHBOR_TASK_LIMIT = 16
UVU_CONV_TRIPLE_BWD_SOURCE_COMPONENT_LIMIT = 64
UVU_CONV_L0_STREAM_STORE_THRESHOLD = 1024
_SOURCE_SEGMENT_EDGES = 32


def build_uvu_conv_metadata(tp, coefficient_dtype):
    if coefficient_dtype not in {torch.float32, torch.float64}:
        raise ValueError(
            f"UVU convolution coefficients require float32 or float64; got {coefficient_dtype}."
        )

    if tp.internal_weights:
        raise ValueError(
            "Generated SO3 kernels require internal_weights=False."
        )
    if tp.shared_weights:
        raise ValueError("Generated SO3 kernels require shared_weights=False.")

    in_slices = tp.irreps_in1.slices()
    edge_slices = tp.irreps_in2.slices()
    out_slices = tp.irreps_out.slices()
    specs: list[UvuPathSpec] = []
    weight_offset = 0
    max_channels = 0

    for instruction_idx, ins in enumerate(tp.instructions):
        if ins.connection_mode != "uvu":
            raise ValueError(
                f"Only connection_mode='uvu' is supported, got {ins.connection_mode!r}."
            )
        if not ins.has_weight:
            raise ValueError(
                "Generated SO3 kernels require weighted uvu instructions."
            )

        mul_in, ir_in = tp.irreps_in1[ins.i_in1]
        mul_edge, ir_edge = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]
        channels = mul_in
        if mul_in != mul_out:
            raise ValueError(
                f"UVU requires equal input/output multiplicities; got {mul_in} and {mul_out} for path ({ins.i_in1}, {ins.i_in2}, {ins.i_out})."
            )
        if ins.path_shape != (mul_in, mul_edge):
            raise ValueError(
                f"UVU expects path_shape=({channels}, {mul_edge}); got {ins.path_shape}."
            )
        max_channels = max(max_channels, channels)

        cg = wigner_3j(
            ir_in.l,
            ir_edge.l,
            ir_out.l,
            dtype=coefficient_dtype,
            device="cpu",
        )
        path_weight = float(ins.path_weight)
        terms = build_cg_terms(cg, path_weight)

        for edge_channel in range(mul_edge):
            spec: UvuPathSpec = {
                "x_offset": in_slices[ins.i_in1].start,
                "y_offset": edge_slices[ins.i_in2].start
                + edge_channel * ir_edge.dim,
                "out_offset": out_slices[ins.i_out].start,
                "weight_offset": weight_offset + edge_channel,
                "channels": channels,
                "d1": ir_in.dim,
                "d2": ir_edge.dim,
                "d3": ir_out.dim,
                "l1": ir_in.l,
                "l2": ir_edge.l,
                "l3": ir_out.l,
                "terms": terms,
            }
            if mul_edge != 1:
                # e3nn stores weighted UVU paths as [input_channel, edge_channel].
                spec["weight_stride"] = mul_edge
                spec["l0_path"] = instruction_idx
            specs.append(spec)
        weight_offset += prod(ins.path_shape)

    if not specs:
        raise ValueError("No uvu instructions found.")
    if weight_offset != tp.weight_numel:
        raise ValueError(
            f"Lowered weight_numel={weight_offset}, but configuration reports {tp.weight_numel}."
        )

    # Channels are tiled across fixed-size CUDA blocks.
    if any(spec["channels"] <= 0 for spec in specs):
        raise ValueError("SO3 uvu kernels require positive channels per irrep.")

    meta = {
        "in1_dim": tp.irreps_in1.dim,
        "in2_dim": tp.irreps_in2.dim,
        "out_dim": tp.irreps_out.dim,
        "weight_numel": tp.weight_numel,
        "max_channels": max_channels,
        "generated_specs": specs,
    }
    return meta

def configure_uvu_conv(op, meta, dtype, index_bits):
    from .forward import load_forward

    op.meta = dict(meta)
    configure_indexing(op, op.meta, "uvu_conv", index_bits)
    op.compute_dtype = dtype
    op.in1_dim = int(meta["in1_dim"])
    op.in2_dim = int(meta["in2_dim"])
    op.out_dim = int(meta["out_dim"])
    op.weight_numel = int(meta["weight_numel"])
    op.max_channels = int(meta["max_channels"])

    op.generated_specs = list(meta["generated_specs"])
    op.l0_offsets, op.l0_dim = l0_feature_layout(op.generated_specs)
    output_counts = {}
    for spec in op.generated_specs:
        key = (int(spec["out_offset"]), int(spec["d3"]), int(spec["channels"]))
        output_counts[key] = output_counts.get(key, 0) + 1
    op.has_forward_output_collision = any(
        count > 1 for count in output_counts.values()
    )
    op.forward_output_needs_zero = (
        op.has_forward_output_collision
        or len(
            {
                int(spec["out_offset"]) + component
                for spec in op.generated_specs
                for component in range(int(spec["channels"]) * int(spec["d3"]))
            }
        )
        != op.out_dim
    )
    op.uniform_channels = (
        len({int(spec["channels"]) for spec in op.generated_specs}) == 1
    )
    op.forward_degree_tiers = dtype == torch.float64 or op.max_channels > 128
    op.fwd_ext = load_forward(op)
    op._forward_extensions = {}


def output_gradient_load(name, out_offset, d3, k, path, l0_offsets, l0_dim):
    """Load the node seed plus its optional per-edge scalar-path seed."""
    node = f"__ldg({name} + go_base + {out_offset} + u * {d3} + {k})"
    if path not in l0_offsets:
        return node
    scalar = "grad_l0" if name == "grad_out" else "sl0"
    address = f"Index(e) * {l0_dim} + {l0_offsets[path]} + u"
    return f"({scalar} ? ({node} + __ldg({scalar} + {address})) : {node})"


def closing(text, start, opening, end):
    depth = 1
    position = start + 1
    while depth:
        depth += (text[position] == opening) - (text[position] == end)
        position += 1
    return position - 1


def add_optional_arguments(source, host_name, parameters, optional_tensors):
    """Thread optional tensor pointers through existing kernels and their launcher.

    The host signature always includes optional tensors; disabled modes pass None.
    Only pointers used by the generated computation reach the CUDA kernels.
    Launch geometry and call order are preserved.
    """

    edits = []
    pattern = r"__global__[^;{}]*?\bvoid\s+(\w+)\s*\("
    for match in re.finditer(pattern, source):
        name = match.group(1)
        end = closing(source, match.end() - 1, "(", ")")
        begin = source.index("{", end)
        body = source[begin : closing(source, begin, "{", "}") + 1]
        declared = source[match.end() : end]
        used = [
            key
            for key in parameters
            if re.search(r"\b" + key + r"\b", body)
            and not re.search(r"\b" + key + r"\b", declared)
        ]
        if not used:
            continue
        declarations = ", " + ", ".join(parameters[key] + " " + key for key in used)
        edits.append((end, declarations))
        calls = re.compile(r"\b" + name + r"(?:<[^;{}\n]*?>)?<<<.*?>>>\(", re.S)
        arguments = ", " + ", ".join(
            key + ".data_ptr<float>()"
            for key in used
        )
        for call in calls.finditer(source):
            end = closing(source, call.end() - 1, "(", ")")
            edits.append((end, arguments))
    host = re.search(r"\bvoid\s+" + host_name + r"\s*\(", source)
    if host is None:
        raise ValueError(f"Missing native launcher {host_name}")
    end = closing(source, host.end() - 1, "(", ")")
    declared = source[host.end() : end]
    if any(re.search(r"\b" + key + r"\b", declared) for key in optional_tensors):
        raise ValueError(f"Optional tensors already declared in {host_name}")
    edits.append(
        (
            end,
            ", "
            + ", ".join(
                "std::optional<torch::Tensor> " + key for key in optional_tensors
            ),
        )
    )
    for position, value in sorted(edits, reverse=True):
        source = source[:position] + value + source[position:]
    for key in optional_tensors:
        source = re.sub(
            r"\b" + key + r"\.data_ptr<([^>]+)>\(\)",
            f"({key} ? {key}->data_ptr<\\1>() : nullptr)",
            source,
        )
    return "#include <optional>\n" + source


def group_source_input_paths(specs: list[UvuPathSpec]):
    groups: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = {}
    for path_idx, spec in enumerate(specs):
        groups.setdefault((int(spec["x_offset"]), int(spec["d1"])), []).append(
            (path_idx, spec)
        )
    return sorted(
        groups.items(),
        key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
        reverse=True,
    )


def flat_coverage(specs, total_dim, offset_key, component_key):
    """Return uncovered flat intervals and whether any instruction overlaps."""
    intervals = sorted(
        (
            int(spec[offset_key]),
            int(spec[offset_key]) + int(spec["channels"]) * int(spec[component_key]),
        )
        for spec in specs
    )
    holes, covered, overlaps = [], 0, False
    for begin, end in intervals:
        if not 0 <= begin < end <= total_dim:
            raise ValueError("instruction extent is outside its tensor dimension")
        if begin > covered:
            holes.append((covered, begin))
        overlaps = overlaps or begin < covered
        covered = max(covered, end)
    if covered < total_dim:
        holes.append((covered, total_dim))
    return holes, overlaps
