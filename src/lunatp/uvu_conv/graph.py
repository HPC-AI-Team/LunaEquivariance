"""Graph-owned CSR segments and traversal tables for UVU convolution."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .kernels.utils import _SOURCE_SEGMENT_EDGES


@dataclass(frozen=True)
class SourceSegmentTables:
    """Bounded source-CSR segments shared by the higher-order kernels.

    nodes identifies each segment owner; begin/end index its adjacency slice.
    rowptr maps a source node to its segments, while order is the launch
    permutation. The permutation does not renumber adjacency entries.
    """

    nodes: torch.Tensor
    begin: torch.Tensor
    end: torch.Tensor
    rowptr: torch.Tensor
    order: torch.Tensor


@dataclass(frozen=True)
class UvuConvGraphTables:
    src: torch.Tensor
    dst: torch.Tensor
    dst_rowptr: torch.Tensor
    src_rowptr: torch.Tensor
    edge_ids_by_src: torch.Tensor
    source_node_order: torch.Tensor | None
    node_order: torch.Tensor | None
    source_segments: SourceSegmentTables | None
    node_tiers: torch.Tensor | None = None


def make_source_segments(src_rowptr):
    """Build source adjacency segments once during make_graph."""
    src_rowptr_i64 = (
        src_rowptr
        if src_rowptr.dtype == torch.int64
        else src_rowptr.to(torch.int64)
    )
    degree = src_rowptr_i64[1:] - src_rowptr_i64[:-1]
    # Isolated nodes retain an empty segment, so the segment CSR still covers every node.
    segment_count = (
        torch.div(
            degree,
            _SOURCE_SEGMENT_EDGES,
            rounding_mode="floor",
        )
        .add((degree % _SOURCE_SEGMENT_EDGES != 0).to(torch.int64))
        .clamp_min(1)
    )
    segment_rowptr_i64 = torch.cat(
        (
            torch.zeros(1, device=src_rowptr.device, dtype=torch.int64),
            torch.cumsum(segment_count, dim=0),
        )
    )
    num_segments = segment_rowptr_i64[-1].item()
    segment_nodes_i64 = torch.repeat_interleave(
        torch.arange(degree.numel(), device=src_rowptr.device, dtype=torch.int64),
        segment_count,
        output_size=num_segments,
    )
    local_segment = torch.arange(
        num_segments, device=src_rowptr.device, dtype=torch.int64
    ) - torch.repeat_interleave(
        segment_rowptr_i64[:-1], segment_count, output_size=num_segments
    )
    segment_begin_i64 = (
        src_rowptr_i64[segment_nodes_i64] + _SOURCE_SEGMENT_EDGES * local_segment
    )
    segment_end_i64 = segment_begin_i64 + torch.minimum(
        torch.full_like(segment_begin_i64, _SOURCE_SEGMENT_EDGES),
        src_rowptr_i64[segment_nodes_i64 + 1] - segment_begin_i64,
    )
    segment_length = segment_end_i64 - segment_begin_i64
    # Lexicographic sorting avoids multiplying an unbounded node degree.
    degree_order = torch.argsort(
        degree[segment_nodes_i64], descending=True, stable=True
    )
    segment_order_i64 = degree_order[
        torch.argsort(segment_length[degree_order], descending=True, stable=True)
    ]
    positions = (
        segment_begin_i64,
        segment_end_i64,
        segment_rowptr_i64,
        segment_order_i64,
    )
    if src_rowptr.dtype == torch.int32:
        positions = tuple(table.to(torch.int32) for table in positions)
    segment_tables = SourceSegmentTables(
        segment_nodes_i64.to(torch.int32), *positions
    )
    return segment_tables

