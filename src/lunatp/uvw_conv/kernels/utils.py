"""UVW convolution utils CUDA and native launch."""

from __future__ import annotations
import torch
from ....utils.extension import get_cuda_arch


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
