"""Graph validation, CSR construction and owned topology."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import index

import torch

from ..utils.extension import get_cuda_arch
from .utils.graph import get_topology_kernel
from .uuu_conv.kernels.utils import UuuConvGraphTables
from .uvu_conv.kernels.utils import _SOURCE_SEGMENT_EDGES
from .uvu_conv.graph import UvuConvGraphTables, SourceSegmentTables, make_source_segments


@dataclass(frozen=True)
class LunaTPGraph:
    """Prepared topology shared by convolutions; tables stay unchanged after construction."""

    src: torch.Tensor
    dst: torch.Tensor
    num_nodes: int
    dst_rowptr: torch.Tensor
    src_rowptr: torch.Tensor
    edge_ids_by_src: torch.Tensor  # Source CSR
    edge_ids_by_dst: torch.Tensor  # Contiguous edge IDs used by UVW CUDA kernels
    node_order: torch.Tensor | None = None
    source_node_order: torch.Tensor | None = None
    node_tiers: torch.Tensor | None = None
    source_segments: SourceSegmentTables | None = None
    # Each graph owns its derived tables; replace() starts with empty runtime state.
    _tables: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def store_topology(self):
        """Expose the same constructed topology to each convolution mode."""
        bits = 64 if self.src_rowptr.dtype == torch.int64 else 32
        inputs = (self.src, self.dst, self.dst_rowptr, self.src_rowptr,
                  self.edge_ids_by_src, self.source_node_order,
                  self.node_order)
        uuu = UuuConvGraphTables(*inputs)
        segments = self.source_segments
        uvu = UvuConvGraphTables(*inputs, segments,
                                node_tiers=self.node_tiers)
        self._tables[("uuu", bits)] = uuu
        self._tables[("uvu", bits)] = uvu


def make_graph(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int | None = None,
) -> LunaTPGraph:
    """Build the topology shared by all convolution modes.

    src and dst accept [edges] or [edges, 1], with dst sorted ascending.
    CSR, edge permutations and traversal orders are generated internally.
    Build once per topology and reuse the graph across convolution layers.
    """
    if src.ndim == 2 and src.shape[1] == 1:
        src = src.squeeze(-1)
    if dst.ndim == 2 and dst.shape[1] == 1:
        dst = dst.squeeze(-1)
    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError("src and dst must have equal shape [edges] or [edges, 1].")
    if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (torch.int32, torch.int64):
        raise TypeError("graph indices must be int32 or int64.")
    if num_nodes is not None and not isinstance(num_nodes, int):
        num_nodes = index(num_nodes)

    num_edges = src.numel()
    required_nodes = 0
    is_cuda = src.is_cuda
    if is_cuda:
        kernel = get_topology_kernel(get_cuda_arch(src.device))
        if num_nodes is None:
            low, high, unsorted = kernel.graph_bounds(src, dst)
            if num_edges and low < 0:
                raise ValueError("graph indices must be nonnegative.")
            if unsorted:
                raise ValueError("dst must be sorted in ascending order; src and edge data must follow the same order.")
            required_nodes = high + 1 if num_edges else 0
    elif num_edges:
        src_min, src_max = torch.aminmax(src)
        dst_min, dst_max = torch.aminmax(dst)
        low = min(src_min.item(), dst_min.item())
        required_nodes = max(src_max.item(), dst_max.item()) + 1
        if low < 0:
            raise ValueError("graph indices must be nonnegative.")
        if num_edges > 1 and bool((dst[1:] < dst[:-1]).any()):
            raise ValueError("dst must be sorted in ascending order; src and edge data must follow the same order.")

    if num_nodes is None:
        num_nodes = required_nodes
    if num_nodes < required_nodes:
        raise ValueError(f"num_nodes={num_nodes} must be nonnegative and at least {required_nodes} for the supplied indices.")
    if num_nodes >= (1 << 31):
        raise OverflowError("graph node count exceeds signed int32 capacity.")
    segment_limit = num_nodes + num_edges // _SOURCE_SEGMENT_EDGES
    bits = 64 if max(num_edges, segment_limit) >= (1 << 31) else 32

    if is_cuda:
        (src_out, dst_out, src_rowptr, dst_rowptr, edge_ids_by_src, edge_ids_by_dst,
         node_order, source_node_order, node_tiers,
         segment_nodes, segment_begin, segment_end, segment_rowptr, segment_order) = (
            kernel.make_graph_tables(src, dst, num_nodes, bits, _SOURCE_SEGMENT_EDGES)
        )
        source_segments = SourceSegmentTables(
            segment_nodes, segment_begin, segment_end, segment_rowptr, segment_order,
        )
    else:
        edge_dtype = torch.int64 if bits == 64 else torch.int32
        src_rowptr = torch.empty(num_nodes + 1, device=src.device, dtype=edge_dtype)
        src_rowptr[0] = 0
        torch.cumsum(torch.bincount(src, minlength=num_nodes), 0, dtype=edge_dtype, out=src_rowptr[1:])
        dst_rowptr = torch.empty_like(src_rowptr)
        dst_rowptr[0] = 0
        torch.cumsum(torch.bincount(dst, minlength=num_nodes), 0, dtype=edge_dtype, out=dst_rowptr[1:])
        node_order = torch.argsort(dst_rowptr.diff().to(torch.int64), descending=True).to(torch.int32)
        source_node_order = torch.argsort(src_rowptr.diff(), descending=True).to(torch.int32)
        if num_nodes and num_edges:
            cumulative = dst_rowptr.diff().to(torch.int64)[node_order.long()].cumsum(0)
            thresholds = torch.arange(1, 4, device=src.device, dtype=torch.int64) * num_edges // 4
            node_tiers = (torch.searchsorted(cumulative, thresholds) + 1).to(torch.int32)
        else:
            node_tiers = torch.full((3,), num_nodes, device=src.device, dtype=torch.int32)
        src_out = src.to(dtype=torch.int32, copy=True, memory_format=torch.contiguous_format)
        dst_out = dst.to(dtype=torch.int32, copy=True, memory_format=torch.contiguous_format)
        edge_ids_by_src = torch.argsort(src).to(edge_dtype)
        edge_ids_by_dst = torch.arange(num_edges, device=src.device, dtype=edge_dtype)
        source_segments = make_source_segments(src_rowptr)

    graph = LunaTPGraph(
        src=src_out, dst=dst_out, num_nodes=num_nodes,
        src_rowptr=src_rowptr, dst_rowptr=dst_rowptr,
        edge_ids_by_src=edge_ids_by_src, edge_ids_by_dst=edge_ids_by_dst,
        node_order=node_order, source_node_order=source_node_order,
        node_tiers=node_tiers, source_segments=source_segments,
    )
    graph.store_topology()
    return graph


__all__ = ["LunaTPGraph", "make_graph"]


def get_graph_with_int64_offsets(graph, *, cache=True):
    """Retain int64 offsets on this graph when first needed."""
    names = ("dst_rowptr", "src_rowptr", "edge_ids_by_src", "edge_ids_by_dst")
    tables = tuple(getattr(graph, name) for name in names)
    if all(table is None or table.dtype == torch.int64 for table in tables):
        return graph
    cached = graph._tables.get("int64")
    if cached is not None:
        return cached
    converted = {
        name: table.to(torch.int64)
        for name, table in zip(names, tables)
        if table is not None and table.dtype != torch.int64
    }
    segments = graph.source_segments
    converted["source_segments"] = replace(
        segments, begin=segments.begin.to(torch.int64), end=segments.end.to(torch.int64),
        rowptr=segments.rowptr.to(torch.int64), order=segments.order.to(torch.int64),
    )
    wide_graph = replace(graph, **converted)
    wide_graph.store_topology()
    if cache:
        graph._tables["int64"] = wide_graph
    return wide_graph
