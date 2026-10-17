"""UUU convolution traversal CUDA."""

from __future__ import annotations
from collections import defaultdict
import torch
from ....utils.extension import load_tp_extension
from ...utils.runtime import get_overlap_state
from ...utils.layout import group_forward_paths
from .forward import make_forward_node_planes, make_grouped_forward_partition
from .backward import (
    emit_path_contractions,
    make_bounded_source_leaves,
    make_destination_backward_kernel,
    make_destination_reduction,
    make_edge_reductions,
    make_envelope_reduction,
    make_source_edge_kernel,
    make_source_node_reduction,
)
from .utils import (
    component_root,
    group_paths,
    make_source_task,
    output_roles,
    BACKWARD_OUTPUTS,
    DESTINATION_GROUP_TERM_LIMIT,
    EDGE_PARTIAL_TERM_LIMIT,
    ENVELOPE_TRIPLE_SOURCE_WIDTH_LIMIT,
    FORWARD_REUSE_TERM_LIMIT,
    FP64_PATH_LEAF_TERM_LIMIT,
    NEIGHBOR_EDGES_PER_NODE,
    NODE_TILES_PER_SM_LIMIT,
    PARTIAL_NODE_TILES_PER_SM_LIMIT,
    SHARED_MEMORY_BYTES,
    SOURCE_BUCKET_STREAMS,
    TENSOR_BINDING,
    SOURCE_BUNDLE_WIDTH_LIMIT,
    SOURCE_EDGE_WIDTH_LIMIT,
)
from ...uuu.kernels.utils import find_root
from . import backward as backward_cuda


def make_uuu_conv_source(
    op,
    order,
    *,
    with_env=False,
    with_l0=False,
    need_grad_env=False,
    outputs=None,
    node_edge_workers=None,
    edge_partials=False,
):
    """Generate fused source, edge and destination traversals for one phase."""

    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    destination_workers = int(
        op.node_edge_workers if node_edge_workers is None else node_edge_workers
    )
    # A function boundary removes cross-path spills in large FP64 programs.
    # Smaller programs retain inline reuse and avoid the extra call lifetime.
    outlined_paths = (
        op.compute_dtype == torch.float64
        and order == 3
        and op.split_groups
        and sum(len(spec["terms"]) for spec in op.generated_specs)
        >= FP64_PATH_LEAF_TERM_LIMIT
    )

    source_names = ["x", "y", "weight"]
    if order:
        source_names += ["grad_out"]
    if order >= 2:
        source_names += ["rx", "ry", "rw"]
    if order == 3:
        source_names += ["sx", "sy", "sw", "sgo"]
    if with_env:
        source_names += ["env"]
        if order >= 2 and need_grad_env:
            source_names += ["re"]
        if order == 3 and need_grad_env:
            source_names += ["se"]
    if with_l0 and order:
        source_names += ["grad_l0"]
        if order == 3:
            source_names += ["sl0"]
    if outputs is None:
        outputs = output_roles(order, with_l0=with_l0, need_grad_env=need_grad_env)
    fused_edge_gradients = (
        order == 3
        and not with_env
        and op.split_groups
        and all(len(paths) == 1 for _, paths in group_paths(op.generated_specs, "g"))
    )
    node_scalars = (
        outlined_paths
        and not need_grad_env
        and not fused_edge_gradients
        and not op.fuse_source
    )
    node_destination = (
        order == 3
        and edge_partials
        and fused_edge_gradients
        and not with_l0
        and not need_grad_env
    )
    paired_scalars = (
        ((edge_partials and not node_destination) or node_scalars)
        and order >= 2
        and "g" in outputs
    )
    overlap_candidate = (
        order in (1, 2, 3)
        and not with_env
        and not with_l0
        and not need_grad_env
        and (not edge_partials or node_destination)
        and not paired_scalars
        and set(outputs) == set(output_roles(order))
    )
    scalar_outputs = (
        [n for n in ("w", "rw", "e", "re") if n in outputs] if paired_scalars else []
    )
    g_groups = group_paths(op.generated_specs, "g")
    slices = sorted(
        (
            int(spec["out_offset"]),
            int(spec["out_offset"]) + int(spec["channels"]) * int(spec["d3"]),
        )
        for spec in op.generated_specs
    )
    grouped_forward = (
        order == 0
        and not with_l0
        and not edge_partials
        and all(left[1] <= right[0] for left, right in zip(slices, slices[1:]))
    )
    packed_g = (
        order >= 2
        and (not edge_partials or node_destination)
        and not need_grad_env
        and op.split_groups
        and all(len(paths) == 1 for _, paths in g_groups)
    )
    # A single angular output cannot be split by path grouping. Fall back to
    # independent warp owners when its neighbor reduction exceeds 48 KiB.
    if (
        packed_g
        and order >= 2
        and scalar == "double"
        and destination_workers > 1
        and any(
            int(spec["d3"]) * 32 * destination_workers * 8 > SHARED_MEMORY_BYTES
            for spec in op.generated_specs
        )
    ):
        destination_workers = 1
    # FP64 output state is expensive in registers. Reuse the neighbor-sum
    # workspace throughout the CSR traversal, retaining one final output owner.
    shared_g_acc = (
        packed_g
        and ((order == 3 and scalar == "float") or (order == 2 and scalar == "double"))
        and destination_workers > 1
        and not with_env
        and not with_l0
        and all(
            int(spec["d3"])
            * 32
            * destination_workers
            * (8 if scalar == "double" else 4)
            <= SHARED_MEMORY_BYTES
            for spec in op.generated_specs
        )
    )
    if packed_g:
        # UVUConv keeps independent destination slices with their shared input
        # group. Bound the live output state before joining their CSR traversal.
        g_groups = []
        limit = 20 if scalar == "double" else 64 if order == 2 else 32
        if shared_g_acc:
            # Keep static shared storage within CUDA's portable 48 KiB limit.
            limit = SHARED_MEMORY_BYTES // (
                32 * destination_workers * (8 if scalar == "double" else 4)
            )
        # Edge features are the large shared operand for channel-preserving
        # higher derivatives; keep their paths in the same CSR owner.
        input_axis = (
            "y" if order == 2 or (order == 3 and not with_env and not with_l0) else "x"
        )
        for (_, _, channels), paths in group_paths(op.generated_specs, input_axis):
            chunk, dimension, terms = [], 0, 0
            for path, spec in paths:
                width, count = int(spec["d3"]), len(spec["terms"])
                if chunk and (
                    dimension + width > limit
                    or terms + count > DESTINATION_GROUP_TERM_LIMIT
                ):
                    g_groups.append(((0, dimension, channels), chunk))
                    chunk, dimension, terms = [], 0, 0
                chunk.append((path, spec))
                dimension += width
                terms += count
            if chunk:
                g_groups.append(((0, dimension, channels), chunk))
    g_parts = 1
    if paired_scalars and order == 3 and edge_partials:
        split = []
        for key, paths in g_groups:
            chunk, terms, part = [], 0, 0
            for path in paths:
                count = len(path[1]["terms"])
                if chunk and terms + count > DESTINATION_GROUP_TERM_LIMIT:
                    split.append((key, chunk))
                    part += 1
                    chunk, terms = [], 0
                chunk.append(path)
                terms += count
            split.append((key, chunk))
            g_parts = max(g_parts, part + 1)
        g_groups = split
    g_starts, g_span = [], 0
    g_slices = defaultdict(list)
    if g_parts > 1:
        for key, _ in g_groups:
            g_starts.append(g_span)
            g_slices[key].append(g_span)
            g_span += key[1] * key[2]

    input_params = ", ".join(f"const {scalar}* __restrict__ {n}" for n in source_names)
    output_params = ", ".join(f"{scalar}* __restrict__ out_{n}" for n in outputs)
    graph_params = (
        "const int32_t* __restrict__ src, const int32_t* __restrict__ dst, "
        "const EdgeIndex* __restrict__ dst_rowptr, const EdgeIndex* __restrict__ src_rowptr, "
        "const EdgeIndex* __restrict__ edge_ids, const int32_t* __restrict__ node_order, "
        "const int32_t* __restrict__ source_node_order, "
        "int num_nodes, Index num_edges"
    )
    parameters = ", ".join((input_params, graph_params, output_params))
    source_args = ", ".join(source_names)
    graph_args = (
        "srcp, dstp, drp, srp, ep, nop, sop, "
        "num_nodes, num_edges"
    )
    output_args = ", ".join("out_" + n for n in outputs)
    arguments = ", ".join((source_args, graph_args, output_args))
    declarations = []
    for i, name in enumerate(source_names):
        pointer = f"inputs[{i}]->data_ptr<{scalar}>()"
        if name in ("re", "se", "grad_l0", "sl0"):
            pointer = f"inputs[{i}] ? {pointer} : nullptr"
        declarations.append(f"const {scalar}* {name} = {pointer};")
    for name in ("re", "se", "grad_l0", "sl0"):
        if name not in source_names:
            declarations.append(f"const {scalar}* {name} = nullptr;")

    source_node_parts, source_node_span = [], 0
    if (
        order == 3
        and edge_partials
        and op.split_groups
        and not with_env
        and not with_l0
        and not need_grad_env
        and all(name in outputs for name in ("x", "rx"))
        and not op.fuse_source
    ):
        if (
            sum(
                component_width
                for axis in ("x", "y")
                for (_, component_width, _), _ in group_paths(op.generated_specs, axis)
            )
            > 32
        ):
            candidates = []
            for key, paths in group_paths(op.generated_specs, "x"):
                for _, neighbors in group_paths([spec for _, spec in paths], "y"):
                    chunk, terms = [], 0
                    for local, spec in neighbors:
                        count = len(spec["terms"])
                        if chunk and terms + count > DESTINATION_GROUP_TERM_LIMIT:
                            candidates.append((key, chunk))
                            chunk, terms = [], 0
                        chunk.append(paths[local])
                        terms += count
                    if chunk:
                        candidates.append((key, chunk))
            if all(
                (
                    32
                    * int(op.source_edge_workers)
                    * component_width
                    * 2
                    * (8 if scalar == "double" else 4)
                    if op.source_edge_workers > 1
                    else 0
                )
                <= SHARED_MEMORY_BYTES
                for (_, component_width, _), _ in candidates
            ):
                for key, paths in candidates:
                    source_node_parts.append((key, paths, source_node_span))
                    source_node_span += key[1] * key[2]

    allocations = []
    input_indices = {name: i for i, name in enumerate(source_names)}
    destination_bundles = []
    if (
        order in (2, 3)
        and not edge_partials
        and not with_env
        and not with_l0
        and not need_grad_env
        and set(outputs) == set(output_roles(order))
        and op.in2_dim > 4 * op.in1_dim
        and all(len(paths) == 1 for _, paths in group_paths(op.generated_specs, "g"))
    ):
        component_specs = op.generated_specs
        source_slices = {
            (int(s["x_offset"]), int(s["d1"]), int(s["channels"]))
            for s in component_specs
        }
        component_roots = {key: key for key in source_slices}

        for _, paths in group_paths(component_specs, "y"):
            first = paths[0][1]
            owner = component_root(
                component_roots,
                (int(first["x_offset"]), int(first["d1"]), int(first["channels"])),
            )
            for _, spec in paths[1:]:
                key = (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
                component_roots[component_root(component_roots, key)] = owner
        component_bundles = defaultdict(list)
        for path, spec in enumerate(component_specs):
            key = (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
            component_bundles[component_root(component_roots, key)].append((path, spec))
        source_component_groups = sorted(
            component_bundles.values(),
            key=lambda paths: sum(len(spec["terms"]) for _, spec in paths),
            reverse=True,
        )
        for paths in source_component_groups:
            if (
                sum(int(spec["d3"]) for _, spec in paths)
                * 32
                * (8 if scalar == "double" else 4)
                > SHARED_MEMORY_BYTES
            ):
                continue
            groups = {
                axis: group_paths([spec for _, spec in paths], axis)
                for axis in ("x", "y")
            }
            widths = {
                axis: sum(
                    component_width for (_, component_width, _), _ in groups[axis]
                )
                for axis in groups
            }
            accumulator_width = sum(widths.values())
            if order == 3:
                full_width = 2 * accumulator_width
                # Extend the route only when compact source partials are small.
                stream_axis = max(widths, key=widths.get)
                accumulator_width = 2 * (
                    accumulator_width
                    - widths[stream_axis]
                    + max(
                        component_width
                        for (_, component_width, _), _ in groups[stream_axis]
                    )
                )
                # GG retains its X/RX/BX forward bases across the CG classes.
                forward_width = 3 * max(
                    component_width for (_, component_width, _), _ in groups["y"]
                )
                if full_width > 64 and (
                    2 * widths["x"] > widths["y"]
                    or accumulator_width + forward_width > 64
                ):
                    continue
            if accumulator_width <= 64:
                destination_bundles.append(paths)
    destination_paths = {path for paths in destination_bundles for path, _ in paths}
    partial_outputs = (
        [
            name
            for name in outputs
            if name in ("x", "rx", "g") and not (node_destination and name == "g")
        ]
        if edge_partials
        else []
    )
    if source_node_parts:
        partial_outputs = [name for name in partial_outputs if name not in ("x", "rx")]
    for name in outputs:
        role = {"rx": "x", "ry": "y", "rw": "w", "re": "e"}.get(name, name)
        reference = {
            "x": "x",
            "y": "y",
            "w": "weight",
            "g": "grad_out",
            "e": "env",
            "l0": "grad_l0",
        }[role]
        allocation = (
            "zeros" if role in op._needs_zero and op._needs_zero[role] else "empty"
        )
        if role in ("g", "l0"):
            shape = (
                f"{{num_nodes, {op.out_dim}}}"
                if role == "g"
                else f"{{num_edges, {op.l0_dim}}}"
            )
            value = f"torch::{allocation}({shape}, features.options())"
        else:
            value = f"torch::{allocation}_like(*inputs[{input_indices[reference]}])"
        allocations.append(f"auto tensor_{name} = {value};")
        if source_node_parts and name in ("x", "rx"):
            declarations += [
                f"auto node_partials_{name} = torch::empty({{num_nodes, {source_node_span}}}, features.options());",
                f"{scalar}* out_{name} = node_partials_{name}.data_ptr<{scalar}>();",
            ]
        elif name in partial_outputs:
            dimension = op.in1_dim if name in ("x", "rx") else op.out_dim
            if name == "g" and g_parts > 1:
                dimension = g_span
                allocation = "empty"
            declarations.append(
                f"auto edge_{name} = torch::{allocation}({{num_edges, {dimension}}}, features.options());"
            )
            declarations.append(
                f"{scalar}* out_{name} = edge_{name}.data_ptr<{scalar}>();"
            )
        else:
            declarations.append(
                f"{scalar}* out_{name} = tensor_{name}.data_ptr<{scalar}>();"
            )

    kernels, launches = [], []
    source_tasks, destination_launches, reduction_launches = [], [], []
    if destination_bundles:
        for role in ("x", "rx") if order == 3 else ("x",):
            declarations += [
                f"auto destination_source_{role} = torch::empty({{num_edges, {op.in1_dim}}}, features.options());",
                f"{scalar}* destination_partial_{role} = destination_source_{role}.data_ptr<{scalar}>();",
            ]
        g_groups = [
            (
                key,
                [(path, spec) for path, spec in paths if path not in destination_paths],
            )
            for key, paths in g_groups
        ]
        g_groups = [(key, paths) for key, paths in g_groups if paths]
    use_node_planes = False
    if grouped_forward:
        use_node_planes = False
        if (
            op.compute_dtype == torch.float32
            and sum(len(spec["terms"]) for spec in op.generated_specs)
            >= FORWARD_REUSE_TERM_LIMIT
        ):
            plane_families = defaultdict(set)
            for spec in op.generated_specs:
                plane_families[int(spec["channels"])].add(
                    (spec["x_offset"], spec["d1"])
                )
            if max(map(len, plane_families.values())) >= 5:
                plane_slices = sorted(
                    {
                        (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
                        for spec in op.generated_specs
                    }
                )
                use_node_planes = max(dim for _, dim, _ in plane_slices) <= 32 and all(
                    left[0] + left[1] * left[2] <= right[0]
                    for left, right in zip(plane_slices, plane_slices[1:])
                )
    if use_node_planes:
        declarations.append(
            f"auto packed_forward_x = torch::empty({{num_nodes, {op.in1_dim}}}, features.options());"
        )
        kernel, calls = make_forward_node_planes(op)
        kernels.append(kernel)
        launches.extend(calls)

    env_names = [n for n in ("e", "re") if n in outputs]
    source_components = sum(
        component_width
        for (_, component_width, _), _ in group_paths(op.generated_specs, "x")
    )
    edge_components = sum(
        component_width
        for (_, component_width, _), _ in group_paths(op.generated_specs, "y")
    )
    # Source and edge projections share feature and scalar derivatives. Bound
    # the complete source/edge accumulator state before joining their owners.
    fused_source = (
        not edge_partials
        and (
            (op.fuse_source and source_components <= 9)
            or (order in (1, 2) and source_components + edge_components <= 32)
        )
        and order > 0
        and any(n in outputs for n in ("x", "rx"))
        and any(n in outputs for n in ("y", "w", "ry", "rw", "e", "re"))
    )
    source_bundles = []
    remaining_paths = [
        (path, spec)
        for path, spec in enumerate(op.generated_specs)
        if path not in destination_paths
    ]
    # Third order retains both primal and directional gradient accumulators.
    source_bundle_limit = (
        SOURCE_BUNDLE_WIDTH_LIMIT
        if order in (1, 2)
        else ENVELOPE_TRIPLE_SOURCE_WIDTH_LIMIT
        if order == 3 and with_env
        else 0
    )
    if destination_bundles:
        fused_source = False
        destination_arguments = ", ".join(
            (
                source_args,
                graph_args,
                ", ".join(
                    "destination_partial_" + role
                    if role in ("x", "rx")
                    else "out_" + role
                    for role in outputs
                ),
            )
        )
        for bundle, paths in enumerate(destination_bundles):
            width = sum(int(spec["d3"]) for _, spec in paths)
            workers = min(
                int(op.source_edge_workers),
                max(
                    1,
                    SHARED_MEMORY_BYTES
                    // (width * 32 * (8 if scalar == "double" else 4)),
                ),
            )
            workers = 4 if workers >= 4 else 2 if workers >= 2 else 1
            kernel, name, tiles, threads = make_destination_backward_kernel(
                op, parameters, paths, workers, f"_bundle{bundle}",
                order=order, source_names=source_names,
            )
            kernels.append(kernel)
            calls = [
                "if (num_nodes > 0) {",
                f"dim3 grid(Index(num_nodes) * {tiles});",
                f"{name}<<<grid, {threads}, 0, stream>>>({destination_arguments});",
                "}",
            ]
            launches += calls
            # Independent complete components reuse the existing joined queues.
            overlap_components = (
                order == 2
                and overlap_candidate
                and len(destination_paths) == len(op.generated_specs)
            )
            if overlap_components:
                for component in destination_bundles:
                    stream_specs = [spec for _, spec in component]
                    stream_x_groups, stream_y_groups = (
                        group_paths(stream_specs, "x"),
                        group_paths(stream_specs, "y"),
                    )
                    stream_width = (
                        sum(
                            component_width
                            for groups in (stream_x_groups, stream_y_groups)
                            for (_, component_width, _), _ in groups
                        )
                        + 2
                        * max(
                            component_width
                            for (_, component_width, _), _ in stream_y_groups
                        )
                        + 3
                        * max(
                            component_width
                            for (_, component_width, _), _ in stream_x_groups
                        )
                        + max(int(spec["d3"]) for spec in stream_specs)
                    )
                    streams_y = (
                        stream_width * (8 if op.compute_dtype == torch.float64 else 4)
                        > 640
                    )
                    if streams_y:
                        overlap_components = False
                        break
            if overlap_components:
                source_tasks.append(make_source_task(name, calls, paths, outputs))
            else:
                destination_launches += calls
    if (
        source_bundle_limit
        and not edge_partials
        and "x" in outputs
        and any(n in outputs for n in ("y", "w", "e"))
    ):
        component_specs = [spec for _, spec in remaining_paths]
        source_slices = {
            (int(s["x_offset"]), int(s["d1"]), int(s["channels"]))
            for s in component_specs
        }
        component_roots = {key: key for key in source_slices}

        for _, paths in group_paths(component_specs, "y"):
            first = paths[0][1]
            owner = component_root(
                component_roots,
                (int(first["x_offset"]), int(first["d1"]), int(first["channels"])),
            )
            for _, spec in paths[1:]:
                key = (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
                component_roots[component_root(component_roots, key)] = owner
        component_bundles = defaultdict(list)
        for path, spec in enumerate(component_specs):
            key = (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
            component_bundles[component_root(component_roots, key)].append((path, spec))
        source_component_groups = sorted(
            component_bundles.values(),
            key=lambda paths: sum(len(spec["terms"]) for _, spec in paths),
            reverse=True,
        )
        candidates = [
            [remaining_paths[path] for path, _ in paths]
            for paths in source_component_groups
        ]
        if len(candidates) > 1:
            # Bound both live accumulators and their shared neighbor reduction.
            source_bundles = [
                paths
                for paths in candidates
                if sum(
                    component_width
                    for axis in ("x", "y")
                    for (_, component_width, _), _ in group_paths(
                        [s for _, s in paths], axis
                    )
                )
                <= source_bundle_limit
                and (
                    order != 3
                    or int(op.source_edge_workers) == 1
                    or 32
                    * int(op.source_edge_workers)
                    * sum(n in outputs for n in ("x", "rx"))
                    * sum(
                        component_width
                        for (_, component_width, _), _ in group_paths(
                            [s for _, s in paths], "x"
                        )
                    )
                    * (8 if scalar == "double" else 4)
                    <= SHARED_MEMORY_BYTES
                )
            ]
        if source_bundles:
            bundled_paths = {path for paths in source_bundles for path, _ in paths}
            remaining_paths = [
                (path, spec)
                for path, spec in remaining_paths
                if path not in bundled_paths
            ]
            fused_source = not remaining_paths
    # A complete edge-irrep owner can share its projections with all source
    # slices. Source slices shared by different edge groups use node partials.
    node_offsets, node_partial_span = [], 0
    if (
        (
            order == 1
            or (order == 2 and op.split_groups)
            or (
                order == 3
                and not op.split_groups
                and all(role in outputs for role in ("rx", "ry", "rw"))
            )
        )
        and not edge_partials
        and (not fused_source or (order == 1 and source_components >= 16))
        and not with_env
        and not with_l0
        and all(role in outputs for role in ("x", "y", "w"))
    ):
        candidates = [
            [remaining_paths[path] for path, _ in paths]
            for _, paths in group_paths([spec for _, spec in remaining_paths], "y")
        ]
        node_channels = max(
            (int(spec["channels"]) for _, spec in remaining_paths), default=0
        )
        live_limit = (
            48 if order == 1 and (scalar == "double" or node_channels <= 32) else 64
        )
        bounded_first = (
            order == 1
            and scalar == "double"
            and any(
                sum(
                    component_width
                    for axis in ("x", "y")
                    for (_, component_width, _), _ in group_paths(
                        [spec for _, spec in paths], axis
                    )
                )
                > live_limit
                for paths in candidates
            )
        )
        if bounded_first or (
            order == 2 and (scalar == "double" or destination_workers >= 4)
        ):
            # Narrow node grids benefit from bounded path owners; wider grids
            # already supply parallel work and retain the complete source owner.
            partitioned = []
            source_width_limit = (
                32 if order == 1 else (16 if scalar == "double" else 24)
            )
            for paths in candidates:
                chunk, width, terms, present = [], 0, 0, set()
                for key, source_paths in group_paths([spec for _, spec in paths], "x"):
                    for path, spec in source_paths:
                        added = 0 if key in present else key[1]
                        count = len(spec["terms"])
                        if chunk and (
                            width + added > source_width_limit
                            or (
                                scalar == "float"
                                and terms + count > DESTINATION_GROUP_TERM_LIMIT
                            )
                        ):
                            partitioned.append(chunk)
                            chunk, width, terms, present = [], 0, 0, set()
                            added = key[1]
                        chunk.append(paths[path])
                        width += added
                        terms += count
                        present.add(key)
                if chunk:
                    partitioned.append(chunk)
            candidates = partitioned
        source_workers = int(op.source_edge_workers)
        shared_per_component = (
            32
            * (8 if scalar == "double" else 4)
            * (
                source_workers * (2 if order == 3 else 1)
                + (2 if order == 2 and source_workers in (2, 4) else 0)
            )
        )
        if len(candidates) > 1 and all(
            (
                order != 3
                or sum(
                    component_width
                    for (_, component_width, _), _ in group_paths(
                        [spec for _, spec in paths], "x"
                    )
                )
                <= (12 if scalar == "double" else 16)
            )
            and sum(
                component_width
                for axis in ("x", "y")
                for (_, component_width, _), _ in group_paths(
                    [spec for _, spec in paths], axis
                )
            )
            <= live_limit
            and sum(
                component_width
                for (_, component_width, _), _ in group_paths(
                    [spec for _, spec in paths], "x"
                )
            )
            * shared_per_component
            <= SHARED_MEMORY_BYTES
            for paths in candidates
        ):
            node_offsets = [None] * len(source_bundles)
            for paths in candidates:
                offsets = {}
                for key, _ in group_paths([spec for _, spec in paths], "x"):
                    offsets[key] = node_partial_span
                    node_partial_span += key[1] * key[2]
                node_offsets.append(offsets)
            # Keep the scratch address range inside existing node index bounds.
            if order == 3 or node_partial_span <= max(op.in1_dim, op.out_dim):
                source_bundles += candidates
                remaining_paths = []
                fused_source = True
                declarations += [
                    f"auto source_partials = torch::empty({{num_nodes, {node_partial_span}}}, features.options());",
                    f"{scalar}* partial_node_x = source_partials.data_ptr<{scalar}>();",
                ]
                if order == 3:
                    declarations += [
                        f"auto source_partials_rx = torch::empty({{num_nodes, {node_partial_span}}}, features.options());",
                        f"{scalar}* partial_node_rx = source_partials_rx.data_ptr<{scalar}>();",
                    ]
            else:
                node_offsets = []

    remaining_specs = [spec for _, spec in remaining_paths]
    source_scalar_paths = (
        {
            path
            for path, spec in remaining_paths
            if spec["has_weight"] and 2 * int(spec["d1"]) <= int(spec["d2"])
        }
        if fused_edge_gradients
        and not edge_partials
        and not fused_source
        and not source_bundles
        and not with_l0
        and not paired_scalars
        and op.in2_dim > 4 * op.in1_dim
        and {"x", "rx", "y", "ry", "w", "rw"} <= set(outputs)
        else set()
    )
    edge_groups = g_groups if paired_scalars else group_paths(remaining_specs, "y")
    bundle_tiles = sum(
        (max(int(s["channels"]) for _, s in paths) + 31) // 32
        for paths in source_bundles
    )
    edge_env_slot = f"{bundle_tiles} + tile" if bundle_tiles else "tile"
    env_parts = (
        bundle_tiles + sum((channels + 31) // 32 for (_, _, channels), _ in edge_groups)
        if source_bundles
        else (op.max_channels + 31) // 32
        if fused_source
        else sum((channels + 31) // 32 for (_, _, channels), _ in edge_groups)
    )
    if env_names:
        parameters += ", " + ", ".join(
            f"{scalar}* __restrict__ partial_{n}" for n in env_names
        )
        arguments += ", " + ", ".join("partial_" + n for n in env_names)
        for n in env_names:
            declarations += [
                f"auto scratch_{n} = torch::empty({{num_edges, {env_parts}}}, features.options());",
                f"{scalar}* partial_{n} = scratch_{n}.data_ptr<{scalar}>();",
            ]

    if fused_source or source_bundles:
        tile_offset = 0
        written_edge_groups = set()
        for bundle_id, paths in enumerate(source_bundles or [None]):
            kernel, name, tiles, threads, tasks_per_block = make_source_edge_kernel(
                op,
                order,
                parameters,
                outputs=outputs,
                with_env=with_env,
                with_l0=with_l0,
                need_grad_env=need_grad_env,
                env_parts=env_parts,
                paths=paths,
                tile_offset=tile_offset,
                kernel_suffix=f"_bundle{bundle_id}" if paths else "",
                node_offsets=node_offsets[bundle_id] if node_offsets else None,
                node_stride=node_partial_span
                if node_offsets and node_offsets[bundle_id]
                else None,
                accumulated_edges=written_edge_groups,
                resident_blocks=(
                    3
                    if order == 2
                    and scalar == "float"
                    and destination_workers < 4
                    and node_offsets
                    and node_offsets[bundle_id] is not None
                    else None
                ),
                source_names=source_names,
            )
            written_edge_groups.update(
                key
                for key, _ in group_paths(
                    [spec for _, spec in paths] if paths else op.generated_specs, "y"
                )
            )
            kernels.append(kernel)
            source_edge_arguments = arguments
            if node_offsets and node_offsets[bundle_id]:
                source_edge_arguments = ", ".join(
                    (
                        source_args,
                        graph_args,
                        ", ".join(
                            "partial_node_x"
                            if role == "x"
                            else "partial_node_rx"
                            if order == 3 and role == "rx"
                            else "out_" + role
                            for role in outputs
                        ),
                    )
                )
            calls = [
                "if (num_nodes > 0) {",
                f"dim3 grid((Index(num_nodes) * {tiles} + {tasks_per_block - 1}) / {tasks_per_block});",
                f"{name}<<<grid, {threads}, 0, stream>>>({source_edge_arguments});",
                "}",
            ]
            launches += calls
            if overlap_candidate:
                source_tasks.append(
                    make_source_task(
                        name,
                        calls,
                        paths or list(enumerate(op.generated_specs)),
                        outputs,
                        node_offsets[bundle_id] if node_offsets else None,
                    )
                )
            tile_offset += tiles
        if node_offsets:
            group_reductions = []
            covered_guard = ""
            covered = {
                key
                for offsets in node_offsets
                if offsets is not None
                for key in offsets
            }
            if destination_bundles or any(offsets is None for offsets in node_offsets):
                ranges = " || ".join(
                    f"(feature >= {offset} && feature < {offset + dimension * channels})"
                    for offset, dimension, channels in sorted(covered)
                )
                covered_guard = f"if (!({ranges})) return;\n"
            for key, _ in group_paths(op.generated_specs, "x"):
                if key not in covered:
                    continue
                offset, dimension, channels = key
                partial_sums = "\n".join(
                    f"value += partial[node * {node_partial_span} + {offsets[key]} + feature - {offset}];"
                    for offsets in node_offsets
                    if offsets is not None and key in offsets
                )
                group_reductions.append(
                    f"""if (feature >= {offset} && feature < {offset + dimension * channels}) {{\n{(partial_sums + chr(10) if partial_sums else "")}}}"""
                )
            kernels.append(
                backward_cuda.SOURCE_GROUP_REDUCTION_KERNEL.format(
                    scalar=scalar,
                    in1_dim=op.in1_dim,
                    covered_guard=covered_guard,
                    group_reductions="\n".join(group_reductions),
                )
            )
            calls = [
                "if (num_nodes > 0) {",
                f"uuuconv_reduce_source_groups<<<(Index(num_nodes) * {op.in1_dim} + 255) / 256, 256, 0, stream>>>(partial_node_x, out_x, num_nodes);",
                "}",
            ]
            if order == 3:
                calls += [
                    "if (num_nodes > 0) {",
                    f"uuuconv_reduce_source_groups<<<(Index(num_nodes) * {op.in1_dim} + 255) / 256, 256, 0, stream>>>(partial_node_rx, out_rx, num_nodes);",
                    "}",
                ]
            launches += calls
            reduction_launches += calls

    traversals = (
        []
        if grouped_forward
        else [
            ("x", [n for n in ("x", "rx") if n in outputs], ""),
            ("y", [n for n in ("y", "w", "ry", "rw", "e", "re") if n in outputs], ""),
            ("g", [n for n in ("g", "l0") if n in outputs], ""),
        ]
    )
    if order == 3 and op.split_groups and not fused_source and not fused_edge_gradients:
        # Third-order Q/scalar and Z/feature families otherwise spill together.
        # Each family retains all contributors and owns its final output slice.
        traversals[1:2] = [
            ("y", [n for n in ("y", "ry") if n in outputs], "_feature"),
            ("y", [n for n in ("w", "rw", "e", "re") if n in outputs], "_scalar"),
        ]
    if paired_scalars:
        traversals = [
            (
                role,
                [n for n in wanted if n not in scalar_outputs]
                + (scalar_outputs if role == "g" else []),
                suffix,
            )
            for role, wanted, suffix in traversals
        ]
    for traversal, wanted, kernel_suffix in traversals:
        if (
            not wanted
            or (fused_source and traversal in ("x", "y"))
            or (source_node_parts and traversal == "x")
        ):
            continue
        traversal_env_names = [n for n in env_names if n in wanted]
        paired_group = paired_scalars and traversal == "g"
        shared_g = shared_g_acc and traversal == "g"
        direct_g_owner = (
            (order == 0 or (order == 2 and scalar == "double" and op.split_groups))
            and traversal == "g"
            and "g" in wanted
            and not with_l0
            and not shared_g
        )
        edge_env = bool(traversal_env_names) and (traversal == "y" or paired_group)
        groups = (
            g_groups
            if traversal == "g"
            else [
                (key, [(remaining_paths[path][0], spec) for path, spec in paths])
                for key, paths in group_paths(remaining_specs, traversal)
            ]
        )
        if not groups:
            continue
        edge_centered = edge_partials and (
            traversal == "x" or (traversal == "g" and not node_destination)
        )
        workers = (
            destination_workers
            if traversal == "g"
            else (int(op.source_edge_workers) if traversal == "x" else 1)
        )
        if edge_centered:
            workers = 1
        # Source derivatives may retain both x and rx in the neighbor workspace.
        if (
            traversal == "x"
            and workers > 1
            and 32
            * workers
            * max(component_width for (_, component_width, _), _ in groups)
            * sum(n in ("x", "rx") for n in wanted)
            * (8 if scalar == "double" else 4)
            > SHARED_MEMORY_BYTES
        ):
            workers = 1
        if workers not in (1, 2, 4, 8):
            raise ValueError("node_edge_workers must be 1, 2, 4 or 8")
        tasks_per_block = 4 if workers == 1 else 1
        threads = 128 if workers == 1 else 32 * workers
        maximum_dimension = max(
            component_width for (_, component_width, _), _ in groups
        )
        coalesced_edge = (
            order in (1, 3)
            and traversal == "y"
            and op.split_groups
            and (
                order != 1
                or len({int(spec["channels"]) for spec in op.generated_specs}) > 1
            )
            and (
                scalar == "float"
                or (
                    not node_destination
                    and len({int(spec["channels"]) for spec in op.generated_specs}) > 1
                )
            )
            and not with_env
            and not with_l0
            and threads * maximum_dimension * (8 if scalar == "double" else 4)
            <= SHARED_MEMORY_BYTES
        )
        neighbor_reduction = workers > 1 and (traversal == "x" or "g" in wanted)
        total_tiles = sum((channels + 31) // 32 for (_, _, channels), _ in groups)
        branches, tile_base = [], 0
        for group_id, ((offset, dim, channels), paths) in enumerate(groups):
            width = (channels + 31) // 32
            shared_stride = dim if op.split_groups else maximum_dimension
            g_components, path_bases = [], {}
            if packed_g and traversal == "g":
                for path, spec in paths:
                    path_bases[path] = len(g_components)
                    g_components.extend(
                        (int(spec["out_offset"]), int(spec["d3"]), k)
                        for k in range(int(spec["d3"]))
                    )
            case = [f"case {group_id}: {{", "const int u = channel_base + lane;"]
            if not edge_env and not neighbor_reduction:
                case.append(f"if (u >= {channels}) break;")
            accum_roles = [
                n
                for n in wanted
                if n
                in (
                    ("x", "rx")
                    if traversal == "x"
                    else ("y", "ry")
                    if traversal == "y"
                    else ("g",)
                )
            ]
            for n in accum_roles:
                for v in range(dim):
                    if shared_g:
                        case.append(
                            f"neighbor_sums[{v} * {threads} + threadIdx.x] = 0.0;"
                        )
                    else:
                        case.append(f"{scalar} acc_{n}_{v} = 0.0;")
            for n in traversal_env_names if edge_env else ():
                case.append(f"{scalar} acc_{n} = 0.0;")
            bounded_source = outlined_paths and traversal == "x" and len(paths) >= 8
            if edge_centered or traversal == "y":
                case.append(r"""const Index edge = item;
const int s = __ldg(src + edge);
const int d = __ldg(dst + edge);
""")
                if edge_env:
                    case.append(f"if (u < {channels}) {{")
            elif traversal == "x":
                if neighbor_reduction:
                    case.append(f"if (u < {channels}) {{")
                begin = "src_rowptr[node]" + (
                    " + neighbor_worker" if workers > 1 else ""
                )
                step = f"p += {workers}" if workers > 1 else "++p"
                case.append("const int s = node;")
                if not bounded_source:
                    case.append(r"""for (Index p = {begin}; p < src_rowptr[node + 1]; {step}) {{
const Index edge = __ldg(edge_ids + p);
const int d = __ldg(dst + edge);
""".format(begin=begin, step=step))
            elif traversal == "g":
                if neighbor_reduction:
                    case.append(f"if (u < {channels}) {{")
                edge_begin = "dst_rowptr[node]" + (
                    " + neighbor_worker" if workers > 1 else ""
                )
                edge_step = f"edge += {workers}" if workers > 1 else "++edge"
                case.append(
                    r"""const int d = node;
for (Index edge = {edge_begin}; edge < dst_rowptr[node + 1]; {edge_step}) {{
const int s = __ldg(src + edge);
""".format(
                        edge_begin=edge_begin, edge_step=edge_step,
                    )
                )
            if bounded_source:
                source_targets = {}
                for output in BACKWARD_OUTPUTS:
                    operand = output.removeprefix("g")
                    if operand in accum_roles:
                        source_targets[output] = [
                            f"acc_{operand}_{component}"
                            for component in range(int(paths[0][1]["d1"]))
                        ]
                helpers, calls = make_bounded_source_leaves(
                    op,
                    paths,
                    order,
                    source_targets,
                    source_names,
                    input_params,
                    group_id,
                    8,
                    workers,
                    with_env=with_env,
                    with_l0=with_l0,
                    need_grad_env=need_grad_env,
                    edge_centered=edge_centered,
                    scalar_paths={path for path, _ in paths} & source_scalar_paths,
                )
                kernels.extend(helpers)
                case.extend(calls)
            for path, spec in () if bounded_source else paths:
                case.append("{")
                path_targets = {}
                for role in BACKWARD_OUTPUTS:
                    n = role.removeprefix("g")
                    if n in accum_roles:
                        base = path_bases.get(path, 0) if n == "g" else 0
                        path_targets[role] = [
                            f"acc_{n}_{base + v}"
                            for v in range(
                                int(
                                    spec[
                                        {"x": "d1", "y": "d2", "g": "d3"}[
                                            n.removeprefix("r")
                                        ]
                                    ]
                                )
                            )
                        ]
                    elif (
                        (
                            (traversal == "y" or paired_group)
                            and n in wanted
                            and not (path in source_scalar_paths and n in ("w", "rw"))
                        )
                        or (
                            traversal == "x"
                            and path in source_scalar_paths
                            and n in ("w", "rw")
                        )
                    ) and (n not in ("w", "rw") or spec["has_weight"]):
                        if n in ("w", "rw"):
                            case.append(f"{scalar} acc_{n} = 0.0;")
                        path_targets[role] = f"acc_{n}"
                if traversal == "g" and not direct_g_owner:
                    # One temporary per path separates node and raw L0 scaling.
                    for k in range(int(spec["d3"])):
                        case.append(f"{scalar} path_value_{k} = 0.0;")
                    path_targets["gg"] = [
                        f"path_value_{k}" for k in range(int(spec["d3"]))
                    ]
                raw = with_l0 and path in op.l0_offsets
                if outlined_paths:
                    # Keep each path's primitive loads and CG intermediates in
                    # an independent leaf function while retaining output owners.
                    helper = f"uuuconv_{traversal}{kernel_suffix}_path{path}"
                    result_type = helper + "_result"
                    path_dims = {
                        "x": int(spec["d1"]),
                        "y": int(spec["d2"]),
                        "g": int(spec["d3"]),
                    }
                    fields, helper_targets = [], {}
                    for output in path_targets:
                        axis = output.removeprefix("g").removeprefix("r")
                        if axis in path_dims:
                            fields.append(f"{scalar} {output}[{path_dims[axis]}];")
                            helper_targets[output] = [
                                f"result.{output}[{v}]" for v in range(path_dims[axis])
                            ]
                        else:
                            fields.append(f"{scalar} {output};")
                            helper_targets[output] = f"result.{output}"
                    helper_body = emit_path_contractions(
                        op,
                        spec,
                        path,
                        order,
                        helper_targets,
                        with_env=with_env,
                        need_grad_env=need_grad_env,
                        combined_l0=raw
                        and not with_env
                        and (traversal != "g" or paired_group),
                    )
                    helper_nulls = "\n".join(
                        f"const {scalar}* {n} = nullptr;"
                        for n in ("re", "se", "grad_l0", "sl0")
                        if n not in source_names
                    )
                    kernels.append(
                        backward_cuda.PATH_LEAF.format(
                            result_type=result_type,
                            fields=" ".join(fields),
                            helper=helper,
                            input_params=input_params,
                            helper_nulls=helper_nulls,
                            helper_body=helper_body,
                        )
                    )
                    case.append(
                        f"const auto result = {helper}({source_args}, edge, s, d, u);"
                    )
                    for output, target in path_targets.items():
                        axis = output.removeprefix("g").removeprefix("r")
                        if axis in path_dims:
                            for v in range(path_dims[axis]):
                                case.append(f"{target[v]} += result.{output}[{v}];")
                        else:
                            case.append(f"{target} += result.{output};")
                else:
                    case.append(
                        emit_path_contractions(
                            op,
                            spec,
                            path,
                            order,
                            path_targets,
                            with_env=with_env,
                            need_grad_env=need_grad_env,
                            combined_l0=raw
                            and not with_env
                            and (traversal != "g" or paired_group),
                            consume_q=direct_g_owner,
                        )
                    )
                if raw and with_env and (traversal != "g" or paired_group):
                    raw_targets = {
                        r: t
                        for r, t in path_targets.items()
                        if r.removeprefix("g") not in ("e", "re") and r != "gg"
                    }
                    case.append(
                        emit_path_contractions(
                            op, spec, path, order, raw_targets, raw_l0=True,
                        )
                    )
                if (
                    traversal == "y"
                    or paired_group
                    or (traversal == "x" and path in source_scalar_paths)
                ):
                    for n in ("w", "rw"):
                        owner = (n in wanted and path not in source_scalar_paths) or (
                            traversal == "x" and path in source_scalar_paths
                        )
                        if owner and spec["has_weight"]:
                            case.append(
                                f"""out_{n}[Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u] = acc_{n};"""
                            )
                if traversal == "g":
                    for k in range(int(spec["d3"])):
                        if "g" in wanted and not direct_g_owner:
                            target = (
                                f"neighbor_sums[{path_bases.get(path, 0) + k} * {threads} + threadIdx.x]"
                                if shared_g
                                else f"acc_g_{path_bases.get(path, 0) + k}"
                            )
                            case.append(f"{target} += path_value_{k};")
                    if raw and "l0" in wanted:
                        if with_env:
                            case.append(f"{scalar} raw_value = 0.0;")
                            case.append(
                                emit_path_contractions(
                                    op,
                                    spec,
                                    path,
                                    order,
                                    {"gg": "raw_value"},
                                    raw_l0=True,
                                )
                            )
                            value = "raw_value"
                        else:
                            value = "path_value_0"
                        case.append(
                            f"out_l0[Index(edge) * {op.l0_dim} + {op.l0_offsets[path]} + u] = {value};"
                        )
                case.append("}")
            if traversal in ("x", "g") and not edge_centered:
                if not bounded_source:
                    case.append("}")
                if neighbor_reduction:
                    case.append("}")
            elif traversal_env_names:
                case.append("}")
            if neighbor_reduction:
                for role_index, n in enumerate(accum_roles):
                    for v in range(dim):
                        if not shared_g:
                            case.append(
                                f"neighbor_sums[({role_index * threads} + threadIdx.x) * {shared_stride} + {v}] = acc_{n}_{v};"
                            )
                case.append("__syncthreads();")
            for role_index, n in enumerate(accum_roles):
                row, stride = (
                    ("node", op.in1_dim)
                    if traversal == "x"
                    else ("item", op.in2_dim)
                    if traversal == "y"
                    else ("node", op.out_dim)
                )
                if edge_centered:
                    row = "item"
                guard = f"if (u < {channels}) " if edge_env else ""
                if neighbor_reduction:
                    case.append(f"if (neighbor_worker == 0 && u < {channels}) {{")
                    for v in range(dim):
                        output_offset, output_dim, output_component = (
                            g_components[v]
                            if n == "g" and g_components
                            else (offset, dim, v)
                        )
                        shared_index = (
                            f"{v} * {threads} + worker * 32 + lane"
                            if shared_g
                            else f"({role_index * threads} + worker * 32 + lane) * {shared_stride} + {v}"
                        )
                        case.append(
                            r"""{scalar} value_{v} = 0.0;
#pragma unroll
for (int worker = 0; worker < {workers}; ++worker) value_{v} += neighbor_sums[{shared_index}];
out_{n}[Index({row}) * {stride} + {output_offset} + u * {output_dim} + {output_component}] = value_{v};""".format(
                                scalar=scalar,
                                v=v,
                                workers=workers,
                                shared_index=shared_index,
                                n=n,
                                row=row,
                                stride=stride,
                                output_offset=output_offset,
                                output_dim=output_dim,
                                output_component=output_component,
                            )
                        )
                    case.append("}")
                    continue
                if coalesced_edge and dim > 1:
                    case.append("{")
                    for v in range(dim):
                        case.append(
                            f"edge_stores[threadIdx.x * {dim} + {v}] = acc_{n}_{v};"
                        )
                    case.append(
                        r"""const unsigned active = __activemask();
__syncwarp(active);
const int valid = min(32, {channels} - channel_base);
for (int local = lane; local < valid * {dim}; local += valid) {{
out_{n}[Index({edge_row}) * {in2_dim} + {offset} + channel_base * {dim} + local] = edge_stores[(threadIdx.x >> 5) * 32 * {dim} + local];
}}
__syncwarp(active);
}}""".format(
                            channels=channels,
                            dim=dim,
                            n=n,
                            in2_dim=op.in2_dim,
                            offset=offset,
                            edge_row="item",
                        )
                    )
                    continue
                for v in range(dim):
                    output_offset, output_dim, output_component = (
                        g_components[v]
                        if n == "g" and g_components
                        else (offset, dim, v)
                    )
                    address = f"Index({row}) * {stride} + {output_offset} + u * {output_dim} + {output_component}"
                    if n == "g" and g_parts > 1:
                        address = f"Index({row}) * {g_span} + {g_starts[group_id]} + u * {dim} + {v}"
                    case.append(f"{guard}out_{n}[{address}] = acc_{n}_{v};")
            if traversal == "y" or paired_group:
                for n in traversal_env_names:
                    case.append(
                        r"""#pragma unroll
for (int delta = 16; delta; delta >>= 1) {{
acc_{n} += __shfl_down_sync(0xffffffffu, acc_{n}, delta);
}}
if (!lane) partial_{n}[Index({edge_row}) * {env_parts} + {edge_env_slot}] = acc_{n};""".format(
                            n=n,
                            env_parts=env_parts,
                            edge_env_slot=edge_env_slot,
                            edge_row="item",
                        )
                    )
            case += ["break;", "}"]
            branches.append("\n".join(case))
            tile_base += width
        decode, base = [], 0
        for gid, (key, _) in enumerate(groups):
            width = (key[2] + 31) // 32
            decode += [
                f"""{("if" if gid == 0 else "else if")} (tile < {base + width}) {{""",
                f"group = {gid}; channel_base = (tile - {base}) * 32;",
                "}",
            ]
            base += width
        rows = "num_edges" if traversal == "y" or edge_centered else "Index(num_nodes)"
        task_position = (
            "Index(blockIdx.x) * 4 + (threadIdx.x >> 5)"
            if workers == 1
            else "Index(blockIdx.x)"
        )
        # A large angular program gets one entry point per tensor slice. All paths
        # contributing to that slice stay together, preserving direct final writes.
        selected_groups = range(len(groups)) if op.split_groups else (None,)
        group_base = 0
        for selected_group in selected_groups:
            width = (
                total_tiles
                if selected_group is None
                else (groups[selected_group][0][2] + 31) // 32
            )
            live_dimension = (
                maximum_dimension
                if selected_group is None
                else groups[selected_group][0][1]
            )
            setup = []
            if workers > 1:
                setup.append("const int neighbor_worker = threadIdx.x >> 5;")
            if neighbor_reduction:
                setup.append(
                    f"__shared__ {scalar} neighbor_sums[{threads * live_dimension * len(accum_roles)}];"
                )
            if coalesced_edge and live_dimension > 1:
                setup.append(
                    f"__shared__ {scalar} edge_stores[{threads * live_dimension}];"
                )
            setup += [
                "if (task >= task_count) return;",
                f"item = task / {width};",
                f"tile = {group_base} + int(task % {width});",
            ]
            if traversal != "y" and not edge_centered:
                pointer = "source_node_order" if traversal == "x" else "node_order"
                setup.append(f"node = {pointer} ? __ldg({pointer} + item) : int(item);")
            if selected_group is None:
                setup += decode
                body = "\n".join(branches)
            else:
                setup += [
                    f"group = {selected_group}; channel_base = (tile - {group_base}) * 32;"
                ]
                body = branches[selected_group]
            # Optional derivative pointers are compile-time null in each variant.
            nulls = "\n".join(
                f"const {scalar}* {n} = nullptr;"
                for n in ("re", "se", "grad_l0", "sl0")
                if n not in source_names
            )
            kernel_name = f"""lunatp_uuuconv_{("fwd", "bwd", "double_bwd", "triple_bwd")[order]}_{traversal}"""
            kernel_name += kernel_suffix
            if selected_group is not None:
                kernel_name += f"_group{selected_group}"
            if edge_centered:
                kernel_name += "_edge"
            min_blocks = (
                2 if scalar == "double" or (order >= 2 and op.split_groups) else 4
            )
            kernels.append(
                backward_cuda.CHANNEL_CSR_KERNEL.format(
                    threads=threads,
                    min_blocks=min_blocks,
                    kernel_name=kernel_name,
                    parameters=parameters,
                    nulls=nulls,
                    setup="\n".join(setup),
                    cases=body,
                    task=task_position,
                    rows=rows,
                    width=width,
                )
            )
            count = f"{rows} * {width}"
            calls = [
                f"if ({rows} > 0) {{",
                f"dim3 grid(({count} + {tasks_per_block - 1}) / {tasks_per_block});",
                f"{kernel_name}<<<grid, {threads}, 0, stream>>>({arguments});",
                "}",
            ]
            launches += calls
            if overlap_candidate:
                if traversal == "g":
                    destination_launches += calls
                else:
                    owners = (
                        groups if selected_group is None else [groups[selected_group]]
                    )
                    paths = [item for _, owner_paths in owners for item in owner_paths]
                    task = make_source_task(
                        kernel_name,
                        calls,
                        paths,
                        wanted,
                        scalar_paths=source_scalar_paths,
                        scalar_owner=traversal if traversal in ("x", "y") else None,
                    )
                    source_tasks.append(task)
            group_base += width
    if source_node_parts:
        source_launches, partial_tasks = [], []
        for part, (key, paths, start) in enumerate(source_node_parts):
            kernel, name, tiles, threads, tasks_per_block = make_source_edge_kernel(
                op,
                order,
                parameters,
                outputs=("x", "rx"),
                with_env=False,
                with_l0=False,
                need_grad_env=False,
                env_parts=0,
                paths=paths,
                kernel_suffix=f"_nodepart{part}",
                node_offsets={key: start},
                node_stride=source_node_span,
                source_names=source_names,
            )
            kernels.append(kernel)
            calls = [
                "if (num_nodes > 0) {",
                f"dim3 grid((Index(num_nodes) * {tiles} + {tasks_per_block - 1}) / {tasks_per_block});",
                f"{name}<<<grid, {threads}, 0, stream>>>({arguments});",
                "}",
            ]
            source_launches += calls
            if overlap_candidate:
                partial_tasks.append(
                    make_source_task(name, calls, paths, ("x", "rx"), {key: start})
                )
        launches[:0] = source_launches
        source_tasks[:0] = partial_tasks
        kernel, calls = make_source_node_reduction(
            op, scalar, source_node_parts, source_node_span
        )
        kernels.append(kernel)
        launches += calls
        reduction_launches += calls
    # Independent output slices can share input ownership without partials.
    if grouped_forward:
        forward_scalar = "double" if op.compute_dtype == torch.float64 else "float"
        forward_share_basis = (
            sum(len(forward_spec["terms"]) for forward_spec in op.generated_specs)
            >= 4096
        )
        forward_input_width = max(
            int(forward_spec["d1"]) + int(forward_spec["d2"])
            for forward_spec in op.generated_specs
        )
        forward_x_cap = (
            16 if forward_scalar == "double" and forward_input_width > 24 else 24
        )
        forward_consumers = defaultdict(set)
        for forward_spec in op.generated_specs:
            forward_consumers[
                (forward_spec["y_offset"], forward_spec["d2"], forward_spec["channels"])
            ].add(
                (forward_spec["x_offset"], forward_spec["d1"], forward_spec["channels"])
            )
        forward_channel_families, forward_family_terms = (
            defaultdict(set),
            defaultdict(int),
        )
        for forward_spec in op.generated_specs:
            forward_width = int(forward_spec["channels"])
            forward_channel_families[forward_width].add(
                (forward_spec["x_offset"], forward_spec["d1"])
            )
            forward_family_terms[forward_width] += len(forward_spec["terms"])
        forward_reused_widths = {
            forward_width
            for forward_width, forward_inputs in forward_channel_families.items()
            if len(forward_inputs) >= 5
            and (
                forward_scalar == "double"
                or forward_family_terms[forward_width] >= FORWARD_REUSE_TERM_LIMIT
            )
        }
        forward_by_axis = {"x": [], "y": []}
        for forward_path, forward_spec in enumerate(op.generated_specs):
            forward_owner = (
                "y"
                if forward_spec["channels"] in forward_reused_widths
                and len(
                    forward_consumers[
                        (
                            forward_spec["y_offset"],
                            forward_spec["d2"],
                            forward_spec["channels"],
                        )
                    ]
                )
                >= 3
                else "x"
            )
            forward_by_axis[forward_owner].append((forward_path, forward_spec))
        forward_batches = []
        for forward_axis, forward_selected in forward_by_axis.items():
            if not forward_selected:
                continue
            forward_cap = (
                forward_x_cap
                if forward_axis == "x"
                else 24
                if forward_scalar == "double"
                else 32
            )
            if forward_axis == "y" and forward_share_basis:
                forward_x_families = defaultdict(list)
                for forward_pair in forward_selected:
                    forward_spec = forward_pair[1]
                    forward_x_families[
                        (
                            forward_spec["x_offset"],
                            forward_spec["d1"],
                            forward_spec["channels"],
                        )
                    ].append(forward_pair)
                forward_groups = []
                for forward_chunk in forward_x_families.values():
                    forward_groups.extend(
                        (
                            forward_key,
                            [
                                forward_chunk[forward_local_path]
                                for forward_local_path, _ in forward_paths
                            ],
                        )
                        for forward_key, forward_paths in group_forward_paths(
                            [forward_spec for _, forward_spec in forward_chunk],
                            axis=forward_axis,
                            live_output_cap=forward_cap,
                        )
                    )
                forward_groups.sort(
                    key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
                    reverse=True,
                )
            else:
                forward_groups = group_forward_paths(
                    [forward_spec for _, forward_spec in forward_selected],
                    axis=forward_axis,
                    live_output_cap=forward_cap,
                )
                forward_groups = [
                    (
                        forward_key,
                        [
                            forward_selected[forward_local_path]
                            for forward_local_path, _ in forward_paths
                        ],
                    )
                    for forward_key, forward_paths in forward_groups
                ]
            if (forward_axis == "x" and forward_scalar == "double") or sum(
                len(forward_spec["terms"]) for forward_spec in op.generated_specs
            ) < FORWARD_REUSE_TERM_LIMIT:
                forward_batches.append((forward_axis, forward_groups))
            else:
                forward_partitions = defaultdict(list)
                for forward_key, forward_paths in forward_groups:
                    (
                        forward_input_offset,
                        _,
                        forward_input_components,
                        forward_channels,
                    ) = forward_key
                    forward_partitions[
                        (
                            forward_input_offset,
                            forward_input_components,
                            forward_channels,
                        )
                    ].append((forward_key, forward_paths))
                forward_batches.extend(
                    (forward_axis, forward_subset)
                    for forward_subset in forward_partitions.values()
                )
        forward_kernels, forward_launches = [], []
        for forward_partition, (forward_axis, forward_subset) in enumerate(
            forward_batches
        ):
            forward_suffix = forward_partition if len(forward_batches) > 1 else None
            forward_use_node_planes = False
            if (
                op.compute_dtype == torch.float32
                and sum(
                    len(forward_spec["terms"]) for forward_spec in op.generated_specs
                )
                >= FORWARD_REUSE_TERM_LIMIT
            ):
                forward_plane_families = defaultdict(set)
                for forward_spec in op.generated_specs:
                    forward_plane_families[int(forward_spec["channels"])].add(
                        (forward_spec["x_offset"], forward_spec["d1"])
                    )
                if max(map(len, forward_plane_families.values())) >= 5:
                    forward_plane_slices = sorted(
                        {
                            (
                                int(forward_spec["x_offset"]),
                                int(forward_spec["d1"]),
                                int(forward_spec["channels"]),
                            )
                            for forward_spec in op.generated_specs
                        }
                    )
                    forward_use_node_planes = max(
                        forward_dim for _, forward_dim, _ in forward_plane_slices
                    ) <= 32 and all(
                        forward_left[0] + forward_left[1] * forward_left[2]
                        <= forward_right[0]
                        for forward_left, forward_right in zip(
                            forward_plane_slices, forward_plane_slices[1:]
                        )
                    )
            forward_kernel, forward_calls = make_grouped_forward_partition(
                op,
                parameters,
                arguments,
                forward_subset,
                forward_suffix,
                emit_path_contractions=emit_path_contractions,
                with_env=with_env,
                group_axis=forward_axis,
                packed_x=forward_use_node_planes,
                share_basis=forward_share_basis,
            )
            forward_kernels.append(forward_kernel)
            forward_launches.extend(forward_calls)
        kernel, grouped_launches = "\n".join(forward_kernels), forward_launches
        kernels.append(kernel)
        launches += grouped_launches
    if destination_bundles:
        reduction_kernels, calls = make_destination_reduction(
            op, order, scalar, destination_bundles
        )
        kernels += reduction_kernels
        launches += calls
        reduction_launches += calls

    reduction_kernels, calls = make_edge_reductions(
        op, scalar, partial_outputs, g_parts, g_slices, g_span
    )
    kernels += reduction_kernels
    launches += calls
    reduction_launches += calls
    if env_names:
        reduction_kernels, calls = make_envelope_reduction(scalar, env_names, env_parts)
        kernels += reduction_kernels
        launches += calls
        reduction_launches += calls

    buckets = []
    if overlap_candidate:
        roots = list(range(len(source_tasks)))

        for right, task in enumerate(source_tasks):
            for left in range(right):
                if any(
                    left_buffer == right_buffer
                    and left_extent == right_extent
                    and left_begin < right_end
                    and right_begin < left_end
                    for left_buffer, left_extent, left_begin, left_end in source_tasks[
                        left
                    ]["spans"]
                    for right_buffer, right_extent, right_begin, right_end in task[
                        "spans"
                    ]
                ):
                    roots[find_root(roots, right)] = find_root(
                        roots, left
                    )
        components = defaultdict(list)
        for index in range(len(source_tasks)):
            components[find_root(roots, index)].append(index)
        if len(components) >= (5 if order == 3 else 3):
            buckets, costs = [[], []], [0, 0]
            component_costs = {
                tuple(indices): sum(source_tasks[i]["cost"] for i in indices)
                for indices in components.values()
            }
            ordered = sorted(
                component_costs,
                key=component_costs.__getitem__,
                reverse=True,
            )
            for indices in ordered:
                target = min(range(2), key=costs.__getitem__)
                buckets[target].extend(indices)
                costs[target] += sum(source_tasks[i]["cost"] for i in indices)
            if not all(buckets):
                buckets = []

    stream_state, state_parameter, stream_binding = "", "", ""
    launch_body = "\n".join(launches)
    if buckets:
        stream_state = SOURCE_BUCKET_STREAMS
        state_parameter = "SourceBucketStreams& state, "
        stream_binding = r"""    pybind11::class_<SourceBucketStreams>(module, "_Streams", pybind11::module_local())
        .def(pybind11::init<int, const std::vector<int64_t>&>());
    module.attr("_Streams").attr("stream_count") = 2;
    module.attr("_Streams").attr("stream_priority") = 0;
"""
        body = ["C10_CUDA_CHECK(cudaEventRecord(state.start, stream));"]
        for bucket in range(2):
            body += [
                f"auto source_stream{bucket} = state.streams[{bucket}].stream();",
                f"C10_CUDA_CHECK(cudaStreamWaitEvent(source_stream{bucket}, state.start, 0));",
            ]
        for bucket, indices in enumerate(buckets):
            for index in indices:
                body += [
                    line.replace(", 0, stream>>>", f", 0, source_stream{bucket}>>>")
                    for line in source_tasks[index]["calls"]
                ]
            body.append(
                f"C10_CUDA_CHECK(cudaEventRecord(state.done[{bucket}], source_stream{bucket}));"
            )
        body += destination_launches
        for bucket in range(2):
            body.append(
                f"C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done[{bucket}], 0));"
            )
        body += reduction_launches
        launch_body = "\n".join(body)

    allocation_source = "\n".join(allocations)
    declaration_source = "\n".join(declarations)
    source = TENSOR_BINDING.format(
        kernels="\n".join(kernels),
        stream_state=stream_state,
        state_parameter=state_parameter,
        allocation_source=allocation_source,
        declaration_source=declaration_source,
        launch_body=launch_body,
        contractions=", ".join(("tensor_" + name for name in outputs)),
        stream_binding=stream_binding,
    )
    return source, tuple(source_names), tuple(outputs)


def empty_table(op, src, *, offsets=False):
    dtype = (
        (torch.int64 if op.index_bits == 64 else torch.int32)
        if offsets
        else torch.int32
    )
    key = src.device, dtype
    table = op._empty_tables.get(key)
    if table is None:
        table = torch.empty(0, device=src.device, dtype=dtype)
        op._empty_tables[key] = table
    return table


def run(
    op,
    order,
    values,
    graph,
    *,
    with_env=False,
    with_l0=False,
    need_grad_env=False,
    outputs=None,
):
    x, y = values["x"], values["y"]
    if not op.generated_specs:
        if outputs is None:
            outputs = output_roles(order, with_l0=with_l0, need_grad_env=need_grad_env)
        shapes = {
            "x": x.shape,
            "y": y.shape,
            "w": values["weight"].shape,
            "g": (x.shape[0], op.out_dim),
            "l0": (y.shape[0], op.l0_dim),
            "e": (y.shape[0], 1),
        }
        return {
            name: x.new_zeros(shapes[name[1:] if name.startswith("r") else name])
            for name in outputs
        }

    edge_partials = graph.dst.numel() == graph.src.numel() and (
        op.output_group_terms > op.forward_group_limit
        if order == 0
        else op.split_groups
        if order == 1
        else op.output_group_terms >= EDGE_PARTIAL_TERM_LIMIT
    )
    node_workers = op.node_edge_workers
    fuse_double_bwd_source_edge = (
        order == 2 and op.source_edge_width <= SOURCE_EDGE_WIDTH_LIMIT
    )
    if fuse_double_bwd_source_edge:
        edge_partials = False
    if x.shape[0]:
        sm_count = op._device_sm_counts.get(x.device)
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
            op._device_sm_counts[x.device] = sm_count
        small_grid = (
            x.shape[0] * max(op.forward_group_tiles)
            <= NODE_TILES_PER_SM_LIMIT * sm_count
        )
        # Independent node groups need four tasks per CTA and two CTAs
        # per SM. Check the narrowest launch before allocating edge partials.
        if order in (1, 2) and op.split_groups and edge_partials:
            small_grid = (
                small_grid
                and x.shape[0] * op.min_node_group_tiles
                < PARTIAL_NODE_TILES_PER_SM_LIMIT * sm_count
            )
        # Large node grids already fill SMs without edge-sized workspaces.
        if op.split_groups:
            edge_partials = edge_partials and small_grid
        # Short FP64 forward programs do not amortize the shared neighbor sum.
        # Higher derivatives reuse enough work to benefit from parallel edges.
        neighbor_work = (
            order >= 2
            or op.compute_dtype == torch.float32
            or op.output_group_terms >= EDGE_PARTIAL_TERM_LIMIT
        )
        large_forward = order == 0 and op.output_group_terms > op.forward_group_limit
        if (
            order != 1
            and neighbor_work
            and y.shape[0] >= NEIGHBOR_EDGES_PER_NODE * x.shape[0]
            and (small_grid or large_forward or fuse_double_bwd_source_edge)
        ):
            node_workers = 4
    if (
        order == 2
        and node_workers == 1
        and op.split_groups
        and not edge_partials
        and op.min_node_group_tiles
        and not need_grad_env
        and x.shape[0]
        and y.shape[0] >= NEIGHBOR_EDGES_PER_NODE * x.shape[0]
    ):
        # More neighbor workers compensate for narrow independent node grids.
        node_workers = (
            4
            if x.shape[0] * op.min_node_group_tiles < NODE_TILES_PER_SM_LIMIT * sm_count
            else 2
        )

    key = (
        order,
        with_env,
        with_l0,
        need_grad_env,
        node_workers,
        op.source_edge_workers,
        op.fuse_source,
        op.split_groups,
        edge_partials,
        None if outputs is None else tuple(outputs),
    )
    cached = op.kernels.get(key)
    if cached is None:
        source, source_names, output_names = make_uuu_conv_source(
            op,
            order,
            with_env=with_env,
            with_l0=with_l0,
            need_grad_env=need_grad_env,
            outputs=outputs,
            node_edge_workers=node_workers,
            edge_partials=edge_partials,
        )
        phase = ("fwd", "bwd", "double_bwd", "triple_bwd")[order]
        dtype_name = "fp64" if op.compute_dtype == torch.float64 else "fp32"
        extension = load_tp_extension(
            f"lunatp_uuu_conv_{phase}_{dtype_name}",
            source,
            max_channels=op.max_channels,
            index_bits=op.index_bits,
        )
        cached = (
            extension.run_uuu_conv,
            source_names,
            output_names,
            getattr(extension, "_Streams", None),
        )
        op.kernels[key] = cached

    launch, source_names, output_names, stream_factory = cached
    table = empty_table(op, graph.src)
    arguments = (
        [values.get(name) for name in source_names],
        graph.src,
        graph.dst,
        graph.dst_rowptr,
        graph.src_rowptr,
        graph.edge_ids_by_src,
        graph.node_order if graph.node_order is not None else table,
        graph.source_node_order if graph.source_node_order is not None else table,
    )
    if stream_factory is not None:
        result = launch(get_overlap_state(op, x.device, stream_factory), *arguments)
    else:
        result = launch(*arguments)
    return dict(zip(output_names, result))


def launch_forward(op, x_node, y, weight, *, graph, env=None, return_l0=False):
    values = {"x": x_node, "y": y, "weight": weight, "env": env}
    result = run(
        op,
        0,
        values,
        graph,
        with_env=env is not None,
        with_l0=return_l0,
        outputs=("g", "l0") if return_l0 else ("g",),
    )
    return (result["g"], result["l0"]) if return_l0 else result["g"]
