"""UUU convolution path metadata, scheduling and CUDA helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from ...utils.layout import (
    TensorProductConfig,
    UuuConvMetadata,
    UuuPathSpec,
    configure_indexing,
    l0_feature_layout,
)
from ...uuu.kernels.utils import build_uuu_metadata
from ...uvu_conv.kernels.utils import UVU_CONV_CHANNEL_CHUNK

FP64_PATH_LEAF_TERM_LIMIT = 2048
FORWARD_REUSE_TERM_LIMIT = 2048

BACKWARD_OUTPUTS = ("gx", "gy", "gw", "gg", "ge", "grx", "gry", "grw", "gre")
CUDA_OPERANDS = {
    "w": "weight",
    "g": "grad_out",
    "e": "env",
    "bx": "sx",
    "by": "sy",
    "bw": "sw",
    "bg": "sgo",
    "be": "se",
}

SHARED_MEMORY_BYTES = 48 * 1024
DESTINATION_GROUP_TERM_LIMIT = 512
SOURCE_BUNDLE_WIDTH_LIMIT = 64
ENVELOPE_TRIPLE_SOURCE_WIDTH_LIMIT = 32
FUSED_SOURCE_WIDTH_LIMIT = 9
SOURCE_EDGE_WIDTH_LIMIT = 32
SPLIT_GROUP_TERM_LIMIT = 1000
EDGE_PARTIAL_TERM_LIMIT = 128
FORWARD_TERM_LIMITS = {torch.float32: 1024, torch.float64: 512}
NODE_TILES_PER_SM_LIMIT = 16
PARTIAL_NODE_TILES_PER_SM_LIMIT = 8
NEIGHBOR_EDGES_PER_NODE = 8


SOURCE_BUCKET_STREAMS = r"""
struct SourceBucketStreams {
    explicit SourceBucketStreams(int device, const std::vector<int64_t>& handles)
        : streams{at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handles.at(0)), device),
                  at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handles.at(1)), device)} {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        for (auto& event : done)
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~SourceBucketStreams() {
        try {
            c10::cuda::CUDAGuard guard(streams[0].device_index());
            cudaEventDestroy(start);
            for (auto event : done) cudaEventDestroy(event);
        } catch (...) { /* CUDA may already be shutting down. */ }
    }
    SourceBucketStreams(const SourceBucketStreams&) = delete;
    SourceBucketStreams& operator=(const SourceBucketStreams&) = delete;
    at::cuda::CUDAStream streams[2];
    cudaEvent_t start = nullptr;
    cudaEvent_t done[2] = {nullptr, nullptr};
};
"""


TENSOR_BINDING = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
namespace {{
{kernels}{stream_state}
}}
std::vector<torch::Tensor> run_uuu_conv({state_parameter}std::vector<std::optional<torch::Tensor>> inputs, torch::Tensor src,
    torch::Tensor dst, torch::Tensor dst_rowptr, torch::Tensor src_rowptr,
    torch::Tensor edge_ids, torch::Tensor node_order, torch::Tensor source_node_order) {{
    const auto& features = *inputs[0];
    const c10::cuda::CUDAGuard device_guard(features.device());
    auto stream = at::cuda::getCurrentCUDAStream(features.get_device());
    const int num_nodes = int(features.size(0));
    const Index num_edges = src.numel();
    const int32_t* srcp = src.data_ptr<int32_t>();
    const int32_t* dstp = dst.data_ptr<int32_t>();
    const EdgeIndex* drp = dst_rowptr.data_ptr<EdgeIndex>();
    const EdgeIndex* srp = src_rowptr.data_ptr<EdgeIndex>();
    const EdgeIndex* ep = edge_ids.data_ptr<EdgeIndex>();
    const int32_t* nop = node_order.numel() ? node_order.data_ptr<int32_t>() : nullptr;
    const int32_t* sop = source_node_order.numel() ? source_node_order.data_ptr<int32_t>() : nullptr;

    {allocation_source}
    {declaration_source}
    {launch_body}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{contractions}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
{stream_binding}    module.def("run_uuu_conv", &run_uuu_conv);
}}
"""


def configure_uuu_conv(op, meta, dtype, index_bits):
    """Lower layouts and reset launch caches for one precision/index width."""
    op.meta = dict(meta)
    op.compute_dtype = dtype
    (
        op.in1_dim,
        op.in2_dim,
        op.out_dim,
        op.weight_numel,
        op.max_channels,
    ) = (
        int(meta[key])
        for key in ("in1_dim", "in2_dim", "out_dim", "weight_numel", "max_channels")
    )

    op.generated_specs = list(meta["generated_specs"])
    if any(int(s["channels"]) < 0 for s in op.generated_specs):
        raise ValueError(
            "UUU convolution requires nonnegative channels for every path."
        )
    op.generated_specs = [
        s for s in op.generated_specs if int(s["channels"]) > 0
    ]
    op.meta["generated_specs"] = op.generated_specs
    configure_indexing(op, op.meta, "uuu_conv", index_bits)

    op.kernels = {}
    op.node_edge_workers = 1
    op.source_edge_workers = 4

    # Source and edge gradient writes reuse the scalar/vector program's loads.
    # Higher angular dimensions use separate slice owners to bound live state.
    op.fuse_source = (
        sum(
            component_width
            for (_, component_width, _), _ in group_paths(op.generated_specs, "x")
        )
        <= FUSED_SOURCE_WIDTH_LIMIT
    )
    op.source_edge_width = sum(
        component_width
        for axis in ("x", "y")
        for (_, component_width, _), _ in group_paths(op.generated_specs, axis)
    )
    op.split_groups = (
        sum(len(spec["terms"]) for spec in op.generated_specs)
        > SPLIT_GROUP_TERM_LIMIT
    )

    op.l0_offsets, op.l0_dim = l0_feature_layout(op.generated_specs)
    groups = group_paths(op.generated_specs, "g")
    op.output_group_terms = max(
        (sum(len(spec["terms"]) for _, spec in paths) for _, paths in groups),
        default=0,
    )
    op.forward_group_limit = FORWARD_TERM_LIMITS[dtype]
    op.forward_group_tiles = tuple(
        (channels + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK
        for (_, _, channels), _ in groups
    )
    op.min_node_group_tiles = 0
    if all(len(paths) == 1 for _, paths in groups):
        source_tiles = [
            (channels + UVU_CONV_CHANNEL_CHUNK - 1) // UVU_CONV_CHANNEL_CHUNK
            for (_, _, channels), _ in group_paths(op.generated_specs, "x")
        ]
        op.min_node_group_tiles = min(
            (*source_tiles, *op.forward_group_tiles), default=0
        )

    op._device_sm_counts = {}
    op._needs_zero = {}
    for role, dim, offset, width in (
        ("x", op.in1_dim, "x_offset", "d1"),
        ("y", op.in2_dim, "y_offset", "d2"),
        ("g", op.out_dim, "out_offset", "d3"),
    ):
        covered = {
            int(spec[offset]) + component
            for spec in op.generated_specs
            for component in range(int(spec["channels"]) * int(spec[width]))
        }
        op._needs_zero[role] = len(covered) != dim
    op._empty_tables = {}


def sum_product_expressions(products, i, j, k):
    if not products:
        return "0.0"
    expressions = []
    for roles in products:
        values = []
        for role in roles:
            if role is None:
                continue
            if role in ("x", "rx", "bx"):
                component = i
            elif role in ("y", "ry", "by"):
                component = j
            elif role in ("g", "bg"):
                component = k
            else:
                component = 0
            value = (
                role + str(component)
                if role.removeprefix("r").removeprefix("b") in ("x", "y", "g")
                else role + "_value"
                if role in ("rw", "re")
                else role
            )
            values.append(value)
        expressions.append(" * ".join(values) if values else "1.0")
    return " + ".join(expressions)


def group_paths(specs: list[UuuPathSpec], role: str):
    """Group every contributor to one channel-preserving tensor slice."""
    offset, dimension = {
        "x": ("x_offset", "d1"),
        "y": ("y_offset", "d2"),
        "g": ("out_offset", "d3"),
    }[role]
    groups = defaultdict(list)
    for path, spec in enumerate(specs):
        key = int(spec[offset]), int(spec[dimension]), int(spec["channels"])
        groups[key].append((path, spec))
    return sorted(
        groups.items(),
        key=lambda group: sum(len(spec["terms"]) for _, spec in group[1]),
        reverse=True,
    )


def output_roles(order, *, with_l0=False, need_grad_env=False):
    """Use the same output layout for empty programs and generated kernels."""
    outputs = ("g",) if order == 0 else ("x", "y", "w")
    if order >= 2:
        outputs += ("g",)
    if need_grad_env:
        outputs += ("e",)
    if with_l0 and order != 1:
        outputs += ("l0",)
    if order == 3:
        outputs += ("rx", "ry", "rw") + (("re",) if need_grad_env else ())
    return outputs


def get_source_write_spans(paths, wanted, node_offsets=None):
    """Describe whole-row write owners before assigning auxiliary streams."""
    spans = set()
    for name in wanted:
        role = {"rx": "x", "ry": "y", "rw": "w"}.get(name, name)
        if role in ("x", "y"):
            for key, _ in group_paths([spec for _, spec in paths], role):
                offset, dimension, channels = key
                buffer = name
                if role == "x" and node_offsets is not None:
                    offset = node_offsets[key]
                    buffer = "partial_" + name
                spans.add(
                    (
                        buffer,
                        "N" if role == "x" else "E",
                        offset,
                        offset + dimension * channels,
                    )
                )
        elif role == "w":
            for _, spec in paths:
                if spec["has_weight"]:
                    offset = int(spec["weight_offset"])
                    spans.add((name, "E", offset, offset + int(spec["channels"])))
    return tuple(sorted(spans))


def make_source_task(
    name, calls, paths, wanted, node_offsets=None, *, scalar_paths=(), scalar_owner=None
):
    roles = tuple(n for n in wanted if n in ("x", "rx", "y", "ry", "w", "rw"))
    cost = sum(int(spec["channels"]) * len(spec["terms"]) for _, spec in paths)
    task = dict(
        name=name,
        calls=calls,
        spans=get_source_write_spans(paths, roles, node_offsets),
        cost=cost * len(roles),
    )
    if scalar_paths and scalar_owner is not None:
        moved = [item for item in paths if item[0] in scalar_paths]
        owned_scalars = (
            moved
            if scalar_owner == "x"
            else [item for item in paths if item[0] not in scalar_paths]
        )
        features = [name for name in wanted if name not in ("w", "rw")]
        task["spans"] = tuple(
            sorted(
                set(get_source_write_spans(paths, features))
                | set(get_source_write_spans(owned_scalars, ("w", "rw")))
            )
        )
        scalar_cost = 2 * sum(
            int(spec["channels"]) * len(spec["terms"]) for _, spec in moved
        )
        task["cost"] += scalar_cost if scalar_owner == "x" else -scalar_cost
    return task


def component_root(component_roots, key):
    while component_roots[key] != key:
        key = component_roots[key]
    return key


@dataclass(frozen=True)
class UuuConvGraphTables:
    """Prepared traversal tables retained by each autograd context."""

    src: torch.Tensor
    dst: torch.Tensor
    dst_rowptr: torch.Tensor
    src_rowptr: torch.Tensor
    edge_ids_by_src: torch.Tensor
    source_node_order: torch.Tensor | None
    node_order: torch.Tensor | None


def build_uuu_conv_metadata(
    official_tp: TensorProductConfig, *, coefficient_dtype=torch.float32
) -> UuuConvMetadata:
    if coefficient_dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"UUU convolution coefficients require float32 or float64; got {coefficient_dtype}."
        )
    meta = build_uuu_metadata(official_tp, dtype=coefficient_dtype)
    # Convolution always lowers weights to one row per edge at its boundary.
    # Keep its metadata independent of the tensor operator's shared-weight mode.
    return {
        key: meta[key]
        for key in (
            "in1_dim",
            "in2_dim",
            "out_dim",
            "weight_numel",
            "max_channels",
            "generated_specs",
        )
    }


