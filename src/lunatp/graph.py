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
    *,
    dst_rowptr: torch.Tensor | None = None,
    src_rowptr: torch.Tensor | None = None,
    edge_ids_by_src: torch.Tensor | None = None,
    node_order: torch.Tensor | None = None,
    source_node_order: torch.Tensor | None = None,
) -> LunaTPGraph:
    """Build topology shared by all convolution modes.

    src and dst each accept shape [edges] or [edges, 1].
    dst must be ascending; src and all edge data must follow the same order.
    Edge order is preserved; this function does not sort destinations.
    Supplied CSR and source-edge order are validated and copied, not rebuilt.
    Kernels derive channel tasks from their configuration and graph traversal orders.
    Build once per topology and pass the graph to each convolution layer.
    """
    if src.ndim == 2 and src.shape[1] == 1:
        src = src.squeeze(-1)
    if dst.ndim == 2 and dst.shape[1] == 1:
        dst = dst.squeeze(-1)

    if dst_rowptr is not None and num_nodes is None:
        num_nodes = dst_rowptr.numel() - 1
    if num_nodes is not None and not isinstance(num_nodes, int):
        num_nodes = index(num_nodes)

    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError(
            f"src and dst must have shape [edges] or [edges, 1] with equal length; got {tuple(src.shape)} and {tuple(dst.shape)}."
        )
    if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            f"graph indices must be int32 or int64; got {src.dtype} and {dst.dtype}."
        )
    required = 0
    unsorted = False
    if src.numel():
        if src.is_cuda:
            low, high, unsorted = get_topology_kernel(get_cuda_arch(src.device)).graph_bounds(src, dst)
        else:
            src_min, src_max = torch.aminmax(src)
            dst_min, dst_max = torch.aminmax(dst)
            low, high = torch.stack(
                (torch.minimum(src_min, dst_min), torch.maximum(src_max, dst_max))
            ).tolist()
            unsorted = dst.numel() > 1 and bool((dst[1:] < dst[:-1]).any())
        if low < 0:
            raise ValueError("graph indices must be nonnegative.")
        required = high + 1
    if num_nodes is None:
        num_nodes = required
    if num_nodes < required or num_nodes < 0:
        raise ValueError(
            f"num_nodes={num_nodes} must be nonnegative and at least {required} for the supplied indices."
        )
    if num_nodes >= (1 << 31):
        raise OverflowError("graph node count exceeds signed int32 capacity.")
    segment_limit = num_nodes + src.numel() // _SOURCE_SEGMENT_EDGES
    edge_dtype = torch.int64 if max(src.numel(), segment_limit) >= (1 << 31) else torch.int32
    tables = (
        dst_rowptr,
        src_rowptr,
        edge_ids_by_src,
        node_order,
        source_node_order,
    )
    for table in tables:
        if table is not None and table.dtype not in (torch.int32, torch.int64):
            raise TypeError("graph tables must be int32 or int64.")
        if table is not None and table.ndim != 1:
            raise ValueError("graph tables must be one-dimensional.")

    # Supplied source CSR must cover every edge exactly once.
    if edge_ids_by_src is not None and not torch.equal(
        torch.sort(edge_ids_by_src).values,
        torch.arange(src.numel(), device=src.device),
    ):
        raise ValueError("edge_ids_by_src must be a permutation of the edge IDs.")
    input_edge_ids = edge_ids_by_src
    if unsorted:
        raise ValueError("dst must be sorted in ascending order; src and edge data must follow the same order.")

    native_tables = None
    if src.is_cuda and all(table is None for table in (
        dst_rowptr, src_rowptr, edge_ids_by_src, node_order, source_node_order
    )):
        native_tables = get_topology_kernel(get_cuda_arch(src.device)).make_graph_tables(
            src, dst, num_nodes, 64 if edge_dtype == torch.int64 else 32, _SOURCE_SEGMENT_EDGES
        )
        src_rowptr, dst_rowptr, edge_ids_by_src = native_tables[:3]
    else:
        rowptrs = []
        for name, nodes, rowptr in (
            ("dst_rowptr", dst, dst_rowptr),
            ("src_rowptr", src, src_rowptr),
        ):
            counts = torch.bincount(nodes, minlength=num_nodes)
            if rowptr is None:
                rowptr = torch.empty(num_nodes + 1, device=src.device, dtype=edge_dtype)
                rowptr[0] = 0
                torch.cumsum(counts, dim=0, dtype=edge_dtype, out=rowptr[1:])
            else:
                # Validate the supplied CSR offsets.
                if rowptr.numel() != num_nodes + 1:
                    raise ValueError(f"{name} must have num_nodes + 1 entries.")
                offsets = rowptr.to(torch.int64)
                if not bool((offsets[0] == 0) & torch.all(offsets.diff() == counts)):
                    raise ValueError(f"{name} does not match the graph indices.")
                rowptr = rowptr.to(
                    dtype=edge_dtype, copy=True, memory_format=torch.contiguous_format
                )
            rowptrs.append(rowptr)
        dst_rowptr, src_rowptr = rowptrs

    if edge_ids_by_src is None:
        edge_ids_by_src = torch.argsort(src)
    elif native_tables is None:
        grouped_src = src[edge_ids_by_src]
        if bool((grouped_src[1:] < grouped_src[:-1]).any()):
            raise ValueError("edge_ids_by_src is not grouped by source node.")
    for name, order in (
        ("node_order", node_order),
        ("source_node_order", source_node_order),
    ):
        if order is not None and not torch.equal(
            torch.sort(order).values,
            torch.arange(num_nodes, device=src.device),
        ):
            raise ValueError(f"{name} must be a permutation of the node IDs.")

    # Copy borrowed storage once. A dtype conversion and copy use the same allocation.
    owned = {}
    for name, table, borrowed in (
        ("src", src, True),
        ("dst", dst, True),
        ("edge_ids_by_src", edge_ids_by_src, edge_ids_by_src is input_edge_ids),
        ("node_order", node_order, True),
        ("source_node_order", source_node_order, True),
    ):
        if table is None:
            owned[name] = None
            continue
        dtype = edge_dtype if name == "edge_ids_by_src" else torch.int32
        owned[name] = table.to(
            dtype=dtype, copy=borrowed, memory_format=torch.contiguous_format
        )
    if native_tables is not None:
        owned["node_order"], owned["source_node_order"] = native_tables[3:5]
    else:
        if node_order is None:
            owned["node_order"] = torch.argsort(dst_rowptr.diff().to(torch.int64), descending=True).to(
                torch.int32
            )
        if source_node_order is None:
            owned["source_node_order"] = torch.argsort(src_rowptr.diff(), descending=True).to(
                torch.int32
            )
    if native_tables is not None:
        node_tiers, *segments = native_tables[5:]
        source_segments = SourceSegmentTables(*segments)
    elif src.is_cuda:
        node_tiers, *segments = get_topology_kernel(get_cuda_arch(src_rowptr.device)).make_topology(
            src_rowptr, dst_rowptr, owned["node_order"], src.numel(), _SOURCE_SEGMENT_EDGES
        )
        source_segments = SourceSegmentTables(*segments)
    else:
        if num_nodes and src.numel():
            cumulative = dst_rowptr.diff().to(torch.int64)[owned["node_order"].long()].cumsum(0)
            thresholds = torch.arange(1, 4, device=src.device, dtype=torch.int64) * src.numel() // 4
            node_tiers = (torch.searchsorted(cumulative, thresholds) + 1).to(torch.int32)
        else:
            node_tiers = torch.full((3,), num_nodes, device=src.device, dtype=torch.int32)
        source_segments = make_source_segments(src_rowptr)
    graph = LunaTPGraph(
        **owned,
        dst_rowptr=dst_rowptr,
        src_rowptr=src_rowptr,
        num_nodes=num_nodes,
        edge_ids_by_dst=torch.arange(src.numel(), device=src.device, dtype=edge_dtype),
        node_tiers=node_tiers,
        source_segments=source_segments,
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
