"""Static contraction layouts, resource limits and launch caches for UVW convolution."""

from __future__ import annotations

from collections import Counter, defaultdict

import torch

from ....utils.extension import get_cuda_arch


FORWARD_SPLIT_CHANNELS = 256
NODE_NEIGHBOR_SPLITS = 16
BACKWARD_NEIGHBOR_SPLITS = 16
BACKWARD_PATH_TILE = 32
MAX_OUTPUT_SPLITS = 4
TARGET_OUTPUT_BLOCKS = 1024
MAX_WEIGHT_BLOCKS = 4096
MAX_WEIGHT_SPLITS = 128
MIN_WEIGHT_SPLITS = 8
WEIGHT_SPLIT_NODES = 128


def make_tr_plan(program, outputs):
    """Combine contractions with exactly the same consumers, before codegen."""
    t_uses, r_uses = defaultdict(Counter), defaultdict(Counter)
    for target, pairs in program["x"].items():
        if target in outputs:
            for feature, z_name in pairs:
                for weight, grad in program["z"][z_name]:
                    t_uses[feature, grad][target, weight] += 1
    for target, pairs in program["w"].items():
        if target in outputs:
            for query, grad in pairs:
                for feature, edge in program["cg"][query]:
                    t_uses[edge, grad][target, feature] += 1
    for target, pairs in program["y"].items():
        if target in outputs:
            for feature, z_name in pairs:
                for weight, grad in program["z"][z_name]:
                    r_uses[feature, weight][target, grad] += 1
    if "gg" in outputs:
        for query, weight in program["forward"]:
            for feature, edge in program["cg"][query]:
                r_uses[feature, weight]["gg", edge] += 1

    combined = []
    for uses in (t_uses, r_uses):
        groups = {}
        for pair, consumers in uses.items():
            signature = tuple(sorted(consumers.items()))
            groups.setdefault(signature, []).append(pair)
        combined.append([{"products": tuple(pairs), "consumers": signature}
                         for signature, pairs in groups.items()])
    t_groups, r_groups = combined
    return dict(t=t_groups, r=r_groups,
                stream_fmas=sum(sum(count for _, count in group["consumers"]) for group in t_groups)
                            + sum(len(group["products"]) for group in r_groups),
                cg_terms=sum(len(group["products"]) for group in t_groups)
                         + sum(sum(count for _, count in group["consumers"]) for group in r_groups))


def get_uvw_conv_tr_layout(meta, program, outputs, spec):
    """Choose the input angular axis from actual program work and local state."""
    if meta.get("dtype", torch.float32) != torch.float32 or program.get("e"):
        return None
    if set(outputs) != set(program["outputs"]):
        return None
    plan = make_tr_plan(program, outputs)
    d1, d3, edges, paths, channels = (spec[key] for key in ("d1", "d3", "edge_channels", "path_channels", "out_channels"))
    needed_z = {pair[1] for axis in ("x", "y") for pairs in program[axis].values() for pair in pairs}
    native_stream = len({pair for key, pairs in program["z"].items() if key in needed_z for pair in pairs})
    native_stream += sum(len(pairs) for pairs in program["w"].values())
    native_stream += len(program["forward"])
    if plan["stream_fmas"] * d1 >= native_stream * d3:
        return None
    queries = {query for pairs in program["w"].values() for query, _ in pairs}
    queries.update(query for query, _ in program["forward"])
    native_cg = sum(len(program["cg"][query]) for query in queries)
    native_cg += sum(len(pairs) for axis in ("x", "y") for pairs in program[axis].values())
    u_tiles = ((paths // edges) + 127) // 128
    terms = len(spec["terms"])
    native_cost = native_stream * d3 * paths * channels + 2 * native_cg * terms * paths
    tr_cost = plan["stream_fmas"] * d1 * paths * channels + 2 * plan["cg_terms"] * terms * edges * channels * u_tiles
    if tr_cost >= native_cost:
        return None
    columns = min(64, max(8, 1 << (channels - 1).bit_length()))
    while columns > 8 and len(plan["r"]) * d1 * (columns // 8) > 32:
        columns //= 2
    shared_bytes = (len(plan["t"]) + 4 * len(plan["r"])) * d1 * columns * 4
    if len(plan["r"]) * d1 * (columns // 8) > 32 or shared_bytes > 47 * 1024:
        return None
    return dict(column_tile=columns, path_tile=128, threads=128, shared_bytes=shared_bytes)


def get_uvw_conv_tr_grid_tasks(meta):
    """Include fixed-v and column tasks in the static edge-index launch bound."""
    specs = meta["generated_specs"]
    max_paths = max(spec["path_channels"] for spec in specs)
    first_tile = min(128, max(32, 1 << (max_paths - 1).bit_length()))
    first_tasks = 0
    for spec in specs:
        paths, edges, channels = (spec[key] for key in ("path_channels", "edge_channels", "out_channels"))
        columns = min(64, max(32, 1 << (channels - 1).bit_length()))
        rows = (paths + edges - 1) // edges
        first_tasks += max((paths + first_tile - 1) // first_tile,
                           edges * ((rows + first_tile - 1) // first_tile)
                           * ((channels + columns - 1) // columns))
    maximum = max(1, first_tasks)
    groups = {}
    for spec in specs:
        edges, paths, channels = (spec[key] for key in ("edge_channels", "path_channels", "out_channels"))
        tasks = edges * (((paths // edges) + 127) // 128) * ((channels + 7) // 8)
        key = spec["d1"], spec["d3"]
        groups[key] = groups.get(key, 0) + tasks
    maximum = max(maximum, max(groups.values(), default=1))
    return maximum


def get_uvw_conv_backward_layout(meta, program, outputs, *, allow_unsupported=False):
    """Fit native CTA scratch; index planning may accept an unsupported layout."""
    w_outputs = {key: pairs for key, pairs in program["w"].items() if key in outputs}
    x_outputs = {key: pairs for key, pairs in program["x"].items() if key in outputs}
    y_outputs = {key: pairs for key, pairs in program["y"].items() if key in outputs}
    e_outputs = {key: pairs for key, pairs in program.get("e", {}).items() if key in outputs}
    z_names = {pair[1] for pairs in (*x_outputs.values(), *y_outputs.values(), *e_outputs.values()) for pair in pairs}
    products = program["forward"] if "gg" in outputs else ()
    queries = {query for pairs in w_outputs.values() for query, _ in pairs}
    queries.update(query for query, _ in products)
    specs = meta["generated_specs"]
    components = max(spec["d3"] for spec in specs)
    columns = ((max(spec["out_channels"] for spec in specs) + 7) // 8) * 8
    z_stride = len(z_names) * components
    z_stride += int(z_stride % 2 == 0)
    edge_elements = len(y_outputs) * max(spec["edge_channels"] * spec["d2"] for spec in specs) + len(e_outputs)
    scalar_bytes = 8 if meta.get("dtype") == torch.float64 else 4
    budget = (48 * 1024 - 1024) // scalar_bytes
    path_tile, column_tile, groups = 128, 128, 16
    while True:
        partial_stride = min(column_tile, columns) * components
        partial_stride += 8 * int((partial_stride // 8) % 2 == 0)
        elements = path_tile * (len(queries) * components + z_stride) + edge_elements
        if products:
            elements += groups * partial_stride
        if elements <= budget:
            break
        if column_tile > 8:
            column_tile //= 2
        elif path_tile > 1:
            path_tile //= 2
        elif groups > 1:
            groups //= 2
        else:
            if allow_unsupported:
                return None
            raise ValueError("UVW convolution backward scratch exceeds 48 KiB with one path and one group")
    return dict(path_tile=path_tile, column_tile=column_tile, groups=groups,
                threads=groups * 8, components=components, z_stride=z_stride,
                partial_stride=partial_stride, shared_bytes=elements * scalar_bytes)


def get_uvw_conv_backward_grid_tasks(meta):
    """Conservatively bound native backward work for index planning."""
    from ...utils.contractions import PROGRAMS, ENV_PROGRAMS

    specs = meta["generated_specs"]
    tasks = sum((spec["path_channels"] + BACKWARD_PATH_TILE - 1) // BACKWARD_PATH_TILE for spec in specs)
    groups = {}
    for spec in specs:
        groups.setdefault(spec["d3"], []).append(spec)
    for group_specs in groups.values():
        group_meta = {**meta, "generated_specs": group_specs}
        for programs in (PROGRAMS, ENV_PROGRAMS):
            for program in programs.values():
                layout = get_uvw_conv_backward_layout(
                    group_meta, program, program["outputs"], allow_unsupported=True,
                )
                # Subsets may still fit; one path per task bounds any supported tile.
                tile = layout["path_tile"] if layout is not None else 1
                tasks = max(tasks, sum((spec["path_channels"] + tile - 1) // tile for spec in group_specs))
    return BACKWARD_NEIGHBOR_SPLITS * tasks


def get_uvw_conv_node_layout(specs):
    offsets = []
    total = 0
    for spec in specs:
        offsets.append(total)
        total += int(spec["path_channels"]) * int(spec["d3"])
    return offsets, total


def get_uvw_conv_weight_layout(specs):
    groups: dict[tuple[int, int, int], list[dict[str, int]]] = {}
    for spec_idx, spec in enumerate(specs):
        key = (int(spec["out_offset"]), int(spec["out_channels"]), int(spec["d3"]))
        groups.setdefault(key, []).append(
            {
                "spec_idx": spec_idx,
                "weight_offset": int(spec["weight_offset"]),
                "path_channels": int(spec["path_channels"]),
                "d3": int(spec["d3"]),
            }
        )

    ordered = []
    rows = []
    spec_to_group: dict[int, tuple[int, int]] = {}
    for group_id, (key, group_specs) in enumerate(
        sorted(groups.items(), key=lambda group: group[0][0])
    ):
        row = 0
        enriched = []
        for item in group_specs:
            item = dict(item)
            item["row_offset"] = row
            spec_to_group[int(item["spec_idx"])] = (group_id, row)
            row += int(item["path_channels"])
            enriched.append(item)
        ordered.append((key, enriched))
        rows.append(row)
    return ordered, rows, spec_to_group


def set_arch(op, device_index):
    if device_index == op.arch_device:
        return
    device = torch.device("cuda", device_index)
    arch = get_cuda_arch(device)
    op.arch_device = device_index
    if arch == op.arch:
        return
    # Native functions belong to their architecture. Changing devices
    # selects a separate extension cache and prepares its own contractions.
    op.arch = arch
    reset_launch_caches(op)


def reset_launch_caches(op):
    op.fused_forward = None
    op.external_kernels = {}
    op.output_contraction = None
    op.node_contraction = None
    op.input_gradient = None
    op.weight_contraction = None
    op.envelope_forward = None
    op.envelope_backward_kernels = {}
