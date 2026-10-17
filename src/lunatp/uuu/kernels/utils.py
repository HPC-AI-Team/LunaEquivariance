"""UUU operand slices, path groups and contraction dependencies."""

from __future__ import annotations
from collections import Counter
from math import prod
import torch
from ...utils.cg import build_cg_terms, wigner_3j
from ...utils.layout import TensorProductConfig, UuuMetadata, UuuTensorPathSpec, CUDA_TASK_TABLE_WARP_LIMIT
from ...utils.contractions import PROGRAMS

UUU_THREADS = 128
# Estimate cached scalars and accumulators, leaving room for address registers.
UUU_FUSED_REGISTER_BUDGET = 112
# Traverse large generated bodies in groups to improve instruction-cache reuse.
UUU_GROUP_MAJOR_SOURCE_BYTES = 512 * 1024


def input_slice(spec, operand):
    """Return the actual offset, irrep width and CG axis read by an operand."""
    if operand in ("x", "rx", "bx"):
        return spec["x_offset"], spec["d1"], 0
    if operand in ("y", "ry", "by"):
        return spec["y_offset"], spec["d2"], 1
    if operand in ("g", "bg"):
        return spec["out_offset"], spec["d3"], 2
    raise ValueError(f"Not a feature operand: {operand}")


def output_slice(spec, output):
    """Return the tensor slice written by one requested derivative."""
    if output in ("gx", "grx"):
        return spec["x_offset"], spec["d1"]
    if output in ("gy", "gry"):
        return spec["y_offset"], spec["d2"]
    if output in ("out", "gg"):
        return spec["out_offset"], spec["d3"]
    if output in ("gw", "grw"):
        return spec["weight_offset"], 1
    raise ValueError(f"Unknown UUU output: {output}")


def build_uuu_groups(meta, outputs):
    """Group outputs sharing an actual tensor slice, preserving family order."""
    groups = []
    for layout, family in (
        ("x", ("gx", "grx")),
        ("y", ("gy", "gry")),
        ("weight", ("gw", "grw")),
        ("output", ("out", "gg")),
    ):
        names = [name for name in outputs if name in family]
        if not names:
            continue
        paths = {}
        for spec in meta["generated_specs"]:
            if layout == "weight" and not spec["has_weight"]:
                continue
            offset, width = output_slice(spec, names[0])
            paths.setdefault((offset, width, spec["channels"]), []).append(spec)
        for (offset, width, channels), group_specs in sorted(paths.items()):
            groups.append(
                dict(
                    layout=layout,
                    outputs=names,
                    offset=offset,
                    dim=width,
                    channels=channels,
                    specs=group_specs,
                )
            )
    return groups


def contraction_dependencies(order, outputs):
    """Trace requested products to the contractions and operands they require."""
    cg = PROGRAMS[order]["cg"] if order else {"q0": (("x", "y"),)}
    z = PROGRAMS[order]["z"] if order else {}
    if order:
        program = PROGRAMS[order]
        expressions = {
            **program["x"],
            **program["y"],
            **program["w"],
            "gg": program["forward"],
        }
    else:
        expressions = {"out": (("q0", "w"),)}

    q_names, z_names, roles = set(), set(), set()
    for output in outputs:
        for left, right in expressions[output]:
            if output in ("gx", "grx", "gy", "gry"):
                roles.add(left)
                z_names.add(right)
            else:
                q_names.add(left)
                roles.add(right)
    for q in q_names:
        roles.update(role for pair in cg[q] for role in pair)
    for name in z_names:
        roles.update(role for pair in z[name] for role in pair)
    return cg, z, q_names, z_names, roles


def find_root(parent, index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def prepare_fused_group(order, dtype, paths, names):
    _, _, q_names, z_names, roles = contraction_dependencies(order, names)
    output_slices = {
        name: sorted({output_slice(spec, name) for spec in paths})
        for name in names
        if name not in ("gw", "grw")
    }

    cached_values = set()
    for operand in roles:
        if operand in ("w", "rw", "bw"):
            continue
        for spec in paths:
            offset, width, cg_axis = input_slice(spec, operand)
            for term in spec["terms"]:
                cached_values.add((operand, offset, term[cg_axis], width))

    accumulators = sum(
        width for slices in output_slices.values() for _, width in slices
    )
    max_d3 = max(spec["d3"] for spec in paths)
    local_state = max_d3 * (len(q_names) + len(z_names))
    local_state += sum(role.lstrip("rb") == "w" for role in roles) + 3
    registers = len(cached_values) + accumulators + local_state
    if dtype is torch.float64:
        registers *= 2
    if registers > UUU_FUSED_REGISTER_BUDGET:
        return None
    return {
        "layout": "fused",
        "outputs": names,
        "channels": paths[0]["channels"],
        "specs": paths,
        "output_slices": output_slices,
        "cached_values": cached_values,
    }


CUDA_HEADERS = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>"""


def make_channel_schedule(groups, cases):
    """Choose item/channel or warp owners once during kernel compilation."""
    group_major = sum(map(len, cases)) >= UUU_GROUP_MAJOR_SOURCE_BYTES
    channels = {group["channels"] for group in groups}
    if len(channels) == 1:
        count = next(iter(channels))
        index = r"""const Index item_channel = Index(blockIdx.x) * GEN_THREADS + threadIdx.x;
    const Index n = item_channel / {channels};
    const int u = item_channel % {channels};
    if (n >= num_items) return;
    const int group = blockIdx.y;""".format(channels=count)
        grid = f"const dim3 grid(static_cast<unsigned int>(\n        (int64_t(num_items) * {count} + GEN_THREADS - 1) / GEN_THREADS), {len(groups)});"
        if len(groups) > 1 and not group_major:
            # Nearby blocks process different roles of the same item/channel
            # tile. Reused x/y/gradient rows remain resident in L2 instead of
            # traversing the entire batch once for each destination group.
            index = r"""const int group = Index(blockIdx.x) % {groups};
    const Index item_channel = (Index(blockIdx.x) / {groups}) * GEN_THREADS + threadIdx.x;
    const Index n = item_channel / {channels};
    const int u = item_channel % {channels};
    if (n >= num_items) return;""".format(channels=count, groups=len(groups))
            grid = f"const dim3 grid(static_cast<unsigned int>(\n        ((int64_t(num_items) * {count} + GEN_THREADS - 1) / GEN_THREADS) * {len(groups)}));"
        constants = ""
    else:
        # Every warp executes one branch and one channel tile. No block reduction
        # or communication is needed, so final partial channel warps can return.
        tasks = [
            (index, base)
            for index, group in enumerate(groups)
            for base in range(0, group["channels"], 32)
        ]
        constants = f"constexpr int64_t GEN_NUM_WARP_TASKS = {len(tasks)};\n"
        index = r"""const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t item_task = int64_t(blockIdx.x) * (GEN_THREADS / 32) + warp;
    const Index n = item_task / GEN_NUM_WARP_TASKS;
    if (n >= num_items) return;
    const int64_t task = item_task % GEN_NUM_WARP_TASKS;
"""
        if len(tasks) <= CUDA_TASK_TABLE_WARP_LIMIT:
            constants += (
                "__device__ __constant__ int GEN_TASK_GROUP[GEN_NUM_WARP_TASKS] = {"
                + ", ".join(str(group) for group, _ in tasks)
                + "};\n"
                "__device__ __constant__ int GEN_TASK_CHANNEL[GEN_NUM_WARP_TASKS] = {"
                + ", ".join(str(base) for _, base in tasks)
                + "};\n"
            )
            index += r"""    const int group = GEN_TASK_GROUP[task];
    const int u = GEN_TASK_CHANNEL[task] + lane;"""
        else:
            index += "    int group;\n    int u;\n"
            start = 0
            for group_index, group in enumerate(groups):
                end = start + (group["channels"] + 31) // 32
                branch = (
                    f"if (task < {end})"
                    if not group_index
                    else "else"
                    if group_index == len(groups) - 1
                    else f"else if (task < {end})"
                )
                index += f"    {branch} {{ group = {group_index}; u = int(task - {start}) * 32 + lane; }}\n"
                start = end
        grid = r"""const dim3 grid(static_cast<unsigned int>((int64_t(num_items) * GEN_NUM_WARP_TASKS + GEN_THREADS / 32 - 1) / (GEN_THREADS / 32)));"""
    return index, grid, constants


def build_uuu_metadata(
    official_tp: TensorProductConfig, dtype: torch.dtype = torch.float32
) -> UuuMetadata:

    internal = bool(official_tp.internal_weights)
    shared = bool(official_tp.shared_weights)
    if internal and not shared:
        raise ValueError("Internal UUU weights must be shared across items.")
    weight_mode = "shared" if shared else "external"
    in_slices = official_tp.irreps_in1.slices()
    edge_slices = official_tp.irreps_in2.slices()
    out_slices = official_tp.irreps_out.slices()
    output_uses = Counter(ins.i_out for ins in official_tp.instructions)
    specs: list[UuuTensorPathSpec] = []
    weight_numel = int(official_tp.weight_numel)
    weight_offset = 0
    max_channels = 0

    for ins in official_tp.instructions:
        if ins.connection_mode != "uuu":
            raise ValueError(
                f"Only connection_mode='uuu' is supported, got {ins.connection_mode!r}."
            )
        mul_in, ir_in = official_tp.irreps_in1[ins.i_in1]
        mul_edge, ir_edge = official_tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = official_tp.irreps_out[ins.i_out]
        channels = int(mul_in)
        if channels != int(mul_edge) or channels != int(mul_out):
            raise ValueError(
                "UUU generated kernels require mul_in1 == mul_in2 == mul_out for every path."
            )
        if channels < 0:
            raise ValueError("UUU requires nonnegative channels for every path.")
        if tuple(int(v) for v in ins.path_shape) != (channels,):
            raise ValueError("UUU generated kernels expect uuu path_shape=(channels,).")
        if channels == 0:
            continue
        max_channels = max(max_channels, channels)

        cg = wigner_3j(ir_in.l, ir_edge.l, ir_out.l, dtype=dtype, device="cpu")
        path_weight = float(ins.path_weight)
        terms = (
            []
            if path_weight == 0
            else [
                (i, j, k, coefficient * path_weight)
                for i, j, k, coefficient in build_cg_terms(cg, 1.0)
            ]
        )
        spec: UuuTensorPathSpec = {
            "x_offset": int(in_slices[ins.i_in1].start),
            "y_offset": int(edge_slices[ins.i_in2].start),
            "out_offset": int(out_slices[ins.i_out].start),
            "accumulate_output": output_uses[ins.i_out] > 1,
            "weight_offset": weight_offset,
            "has_weight": bool(ins.has_weight),
            "path_weight": path_weight,
            "channels": channels,
            "d1": int(ir_in.dim),
            "d2": int(ir_edge.dim),
            "d3": int(ir_out.dim),
            "l1": int(ir_in.l),
            "l2": int(ir_edge.l),
            "l3": int(ir_out.l),
            "terms": terms,
        }
        specs.append(spec)
        if ins.has_weight:
            weight_offset += prod(ins.path_shape)

    if weight_offset != weight_numel:
        raise ValueError(
            f"lowered weight_numel={weight_offset}, configuration reports {official_tp.weight_numel}."
        )
    meta = {
        "in1_dim": int(official_tp.irreps_in1.dim),
        "in2_dim": int(official_tp.irreps_in2.dim),
        "out_dim": int(official_tp.irreps_out.dim),
        "weight_numel": weight_numel,
        "max_channels": max_channels,
        "weight_mode": weight_mode,
        "generated_specs": specs,
        # One grouped writer accumulates every path to an output slice.
        # Components outside those slices are explicitly initialized to zero.
        "zero_output": sum(
            channels * dim
            for _, channels, dim in {
                (spec["out_offset"], spec["channels"], spec["d3"]) for spec in specs
            }
        )
        != int(official_tp.irreps_out.dim),
    }
    if dtype is torch.float64:
        meta["dtype"] = dtype
    return meta
