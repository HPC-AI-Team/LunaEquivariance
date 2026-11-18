"""Prepare reusable graph indices and periodic edge geometry."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
import math
from threading import local
from types import SimpleNamespace
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

__all__: list[str] = []


def validate_receivers(receivers, nodes, *, sorted=False):
    if receivers.ndim != 1 or receivers.dtype not in (torch.int32, torch.int64):
        raise ValueError("receivers must be a one-dimensional integer tensor")
    if not 0 <= nodes < 2**31 or receivers.numel() >= 2**31:
        raise ValueError("graph sizes must fit the 32-bit CSR representation")
    torch._assert_async(
        ((receivers >= 0) & (receivers < nodes)).all(),
        "receiver indices must lie within the node range",
    )
    if sorted:
        torch._assert_async(
            (receivers[1:] >= receivers[:-1]).all(),
            "CSR aggregation requires edges sorted by receiver",
        )


def validate_edge_index(edge_index, nodes, *, sorted_receivers=False):
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.int64
    ):
        raise ValueError("edge_index must have shape (2, E) and dtype int64")
    validate_receivers(edge_index[1], nodes, sorted=sorted_receivers)
    torch._assert_async(
        ((edge_index[0] >= 0) & (edge_index[0] < nodes)).all(),
        "source indices must lie within the node range",
    )


def prefix_csr_metadata(edge_index, nodes):
    validate_edge_index(edge_index, nodes, sorted_receivers=True)
    source_order = torch.argsort(edge_index[0], stable=True)
    source_sorted = edge_index[0].index_select(0, source_order)
    boundaries = torch.arange(
        nodes + 1, device=edge_index.device, dtype=edge_index.dtype
    )
    source_row_ptr = torch.searchsorted(source_sorted, boundaries).to(torch.int32)
    # Native COO conversion preserves this receiver view's offset under Inductor.
    receiver_row_ptr = torch._convert_indices_from_coo_to_csr(edge_index[1], nodes).to(
        torch.int32
    )
    return source_order.to(torch.int32), source_row_ptr, receiver_row_ptr


class GraphCSRRuntime:
    """Reuse one CSR topology across a sequence of edgewise blocks."""

    def __init__(self, block_count: int):
        self.block_count = block_count
        self.reset()

    def reset(self):
        self.edge_index = None
        self.metadata = None
        self.next_index = 0

    def __deepcopy__(self, memo):
        result = type(self)(self.block_count)
        memo[id(self)] = result
        return result

    def consume(self, index, edge_index, nodes, node_offset):
        try:
            if node_offset != 0:
                raise ValueError("the CSR runtime requires zero node_offset")
            if index == 0:
                self.reset()
                self.edge_index = edge_index
                self.metadata = prefix_csr_metadata(edge_index, nodes)
                self.next_index = 0
            if self.metadata is None or self.edge_index is not edge_index:
                raise RuntimeError("edgewise blocks must share one graph tensor")
            if index != self.next_index:
                raise RuntimeError(
                    f"edgewise order changed: expected {self.next_index}, got {index}"
                )
            self.next_index += 1
            return self.metadata
        except Exception:
            self.reset()
            raise

    def finish(self, index):
        if index == self.block_count - 1:
            self.reset()


class PreparedGraph(NamedTuple):
    """Receiver-sorted graph, new-to-original edge order, and CSR metadata."""

    edge_index: torch.Tensor
    edge_order: torch.Tensor
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def source_order(self) -> torch.Tensor:
        """Edge indices grouped by sending node, relative to this graph."""
        return self.metadata[0]

    @property
    def source_row_ptr(self) -> torch.Tensor:
        """CSR boundaries for sending nodes."""
        return self.metadata[1]

    @property
    def receiver_row_ptr(self) -> torch.Tensor:
        """CSR boundaries for receiving nodes."""
        return self.metadata[2]


def prepare_graph(edge_index: torch.Tensor, num_nodes: int) -> PreparedGraph:
    """Sort an int64 ``(2, E)`` graph by receiver and construct CSR indices.

    Row 0 is source, row 1 is receiver. Stable sorting preserves the order
    of equal receivers. Apply ``value.index_select(0, result.edge_order)``
    to every original edge tensor (vectors, rotations, radial weights).
    The returned graph equals ``edge_index[:, result.edge_order]``.
    Metadata is ``(source_order, source_row_ptr, receiver_row_ptr)`` with
    int32 entries; source_order indexes the *returned* edge ordering.
    Both sum and attention aggregation accept that int32 receiver pointer.
    Isolated nodes and empty graphs are supported. Sizes must fit int32.

    Reuse the result while topology and edge ordering stay unchanged.
    """
    validate_edge_index(edge_index, num_nodes)
    order = torch.argsort(edge_index[1], stable=True)
    ordered = edge_index.index_select(1, order)
    return PreparedGraph(ordered, order, prefix_csr_metadata(ordered, num_nodes))


def graph_tables(graph):
    """Reuse one edge ordering and its CSR tables across convolution layers."""
    key = ("so2", "int32")
    version_key = ("so2", "topology_version")
    topology = (graph.src, graph.dst, graph.edge_ids_by_src, graph.src_rowptr, graph.dst_rowptr)
    try:
        signature = tuple((id(value), value._version) for value in topology)
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        # Inference tensors can change without observable versions. Rebuild
        # this call's tables and discard all SO(2) derived traversal caches.
        for name in tuple(graph._tables):
            if isinstance(name, tuple) and name and name[0] == "so2":
                del graph._tables[name]
        signature = None
    if key in graph._tables and graph._tables[version_key] != signature:
        raise RuntimeError("graph topology changed after SO(2) preparation; rebuild the graph")
    if key not in graph._tables:
        if graph.src.numel() >= 2**31 or graph.num_nodes >= 2**31:
            raise ValueError("SO(2) graph sizes must fit signed int32 offsets")
        edge_index = torch.stack((graph.src, graph.dst)).to(torch.int64)
        metadata = (
            graph.edge_ids_by_src.to(torch.int32),
            graph.src_rowptr.to(torch.int32),
            graph.dst_rowptr.to(torch.int32),
        )
        graph._tables[key] = (edge_index, metadata)
        graph._tables[("so2", "receivers")] = edge_index[1]
        graph._tables[version_key] = signature
    return graph._tables[key]


@cache
def load_periodic_graph_backend():
    """Bind optional Warp helpers when a periodic graph is constructed."""
    global legacy_cell_list, empty_sentinel, get_build_cell_list_kernel, get_query_cell_list_kernel
    global wp, allocate_cell_list, estimate_max_neighbors, batch_cell_list, neighbor_list
    global get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype
    global _batch_estimate_cell_list_sizes_overload, _batch_cell_list_construct_bin_size_overload
    global _batch_cell_list_bin_atoms_overload, _batch_cell_list_count_atoms_per_bin_overload
    global _batch_cell_list_build_neighbor_matrix_overload

    import warp as wp
    from nvalchemiops.torch.neighbors import neighbor_list
    from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list
    from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list, estimate_max_neighbors
    from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype
    legacy_cell_list = version("nvalchemi-toolkit-ops") == "0.3.0"
    if legacy_cell_list:
        from nvalchemiops.neighbors.batch_cell_list import (
            _batch_cell_list_bin_atoms_overload,
            _batch_cell_list_build_neighbor_matrix_overload,
            _batch_cell_list_construct_bin_size_overload,
            _batch_cell_list_count_atoms_per_bin_overload,
            _batch_estimate_cell_list_sizes_overload,
        )
    else:
        from nvalchemiops.neighbors.cell_list import get_build_cell_list_kernel, get_query_cell_list_kernel
        from nvalchemiops.neighbors.neighbor_utils import empty_sentinel


# The sizing and binning kernels halve the grid dimensions to fit this workspace.
MAX_BINS = 8192
WORKSPACE_FIELDS = (
    "cells_per_dimension", "neighbor_search_radius", "atom_periodic_shifts",
    "atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_start_indices",
    "cell_atom_list",
)


class FixedCapacityCellList:
    """Reuse cell bins for one periodic system with a checked cell."""

    def __init__(self):
        load_periodic_graph_backend()
        self.scratch = local()

    def workspace(self, positions):
        device = positions.device
        key = (positions.shape[0], device, torch.cuda.current_stream(device).cuda_stream)
        if getattr(self.scratch, "key", None) != key:
            radii = torch.empty((1, 3), dtype=torch.int32, device=device)
            counts = torch.empty(1, dtype=torch.int32, device=device)
            buffers = allocate_cell_list(positions.shape[0], MAX_BINS, radii, device)
            self.scratch.buffers = counts, dict(zip(WORKSPACE_FIELDS, buffers, strict=True))
            self.scratch.key = key
        return self.scratch.buffers

    def __call__(self, *, positions, cutoff, cell, pbc, batch_idx, neighbor_matrix,
                 neighbor_matrix_shifts, num_neighbors, half_fill=False):
        if not legacy_cell_list:
            tensors = positions, cell, pbc, batch_idx
            stream = torch.cuda.current_stream(positions.device)
            scratch = bind_workspace(
                SimpleNamespace(), self, positions, neighbor_matrix, neighbor_matrix_shifts,
                num_neighbors, stream,
            )
            launch_cell_list(scratch, tensors, cutoff, half_fill=half_fill)
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

        counts, buffers = self.workspace(positions)
        dtype = get_wp_dtype(cell.dtype)
        wp.launch(
            _batch_estimate_cell_list_sizes_overload[dtype],
            dim=1,
            inputs=[
                wp.from_torch(cell, dtype=get_wp_mat_dtype(cell.dtype), return_ctype=True),
                wp.from_torch(pbc, dtype=wp.bool, return_ctype=True),
                dtype(cutoff),
                MAX_BINS,
                wp.from_torch(counts, dtype=wp.int32, return_ctype=True),
                wp.from_torch(buffers["neighbor_search_radius"], dtype=wp.vec3i, return_ctype=True),
            ],
            device=str(positions.device),
        )
        # The periodic dispatch in neighbor_list passes batch_idx unchanged and
        # discards batch_ptr. The batch builder clears the workspace every call.
        return batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, **buffers,
            neighbor_matrix=neighbor_matrix, neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors, half_fill=half_fill,
        )


class PeriodicCellList(FixedCapacityCellList):
    """Fixed-capacity path with nvalchemi fallback and cell validation."""

    def __call__(self, *, positions, cutoff, cell, pbc, batch_idx, method, **kwargs):
        supported_options = {"neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"}
        if (
            method != "batch_cell_list" or cell is None or cell.shape[0] != 1
            or not positions.is_cuda or cutoff <= 0 or positions.shape[0] == 0
            or not supported_options <= kwargs.keys() <= supported_options | {"half_fill"}
            or any(kwargs[name] is None for name in supported_options)
        ):
            return neighbor_list(
                positions, cutoff, cell=cell, pbc=pbc, batch_idx=batch_idx,
                method=method, **kwargs,
            )

        # Preserve the positive-volume check from the original size estimator.
        if invalid_cell(cell):
            raise RuntimeError(
                "Cells with volume <= 0 detected and are not supported. "
                "Please pass unit cells with `det(cell) > 0.0`."
            )
        return super().__call__(
            positions=positions, cutoff=cutoff, cell=cell, pbc=pbc, batch_idx=batch_idx, **kwargs,
        )


def tensor_version(value):
    if value.is_inference():
        return None
    return value._version


@dataclass(frozen=True)
class GeometryMetadata:
    pbc: torch.Tensor
    cell: torch.Tensor
    pbc_version: int | None
    cell_version: int | None
    periodic: bool
    positive_diagonal_cell: bool


geometry_metadata = ContextVar("neighbor_geometry_metadata", default=None)


def cpu_geometry_flags(pbc, cell):
    if pbc.device.type != "cpu" or cell.device.type != "cpu" or cell.shape != (1, 3, 3):
        return None
    values = cell.reshape(-1).tolist()
    # For positive diagonal FP32/FP64 cells with this margin, the LU determinant
    # is positive and every intermediate product is representable on either device.
    positive = (
        cell.dtype in (torch.float32, torch.float64)
        and all(values[i] == 0. for i in (1, 2, 3, 5, 6, 7))
        and all(2.**-30 <= values[i] <= 2.**30 for i in (0, 4, 8))
    )
    return bool(pbc.any()), positive


def bind_metadata(pbc, cell, flags):
    if flags is None:
        return None
    return GeometryMetadata(pbc, cell, tensor_version(pbc), tensor_version(cell), *flags)


@contextmanager
def device_geometry(pbc, cell, flags):
    token = geometry_metadata.set(bind_metadata(pbc, cell, flags))
    try:
        yield
    finally:
        geometry_metadata.reset(token)


def any_pbc(pbc):
    metadata = geometry_metadata.get()
    if (
        metadata is not None and metadata.pbc is pbc
        and metadata.pbc_version is not None and tensor_version(pbc) == metadata.pbc_version
    ):
        return metadata.periodic
    return bool(pbc.any().item())


def invalid_cell(cell):
    metadata = geometry_metadata.get()
    if (
        metadata is not None and metadata.cell is cell and metadata.positive_diagonal_cell
        and metadata.cell_version is not None and tensor_version(cell) == metadata.cell_version
    ):
        return False
    return bool(torch.any(cell.det() <= 0.0))


def view(tensor, dtype, *, scan=False):
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False, return_ctype=not scan)


def bind_workspace(scratch, workspace, positions, matrix, shifts, neighbors, stream):
    key = (positions.device, positions.shape[0], stream.cuda_stream,
           matrix.data_ptr(), shifts.data_ptr(), neighbors.data_ptr())
    if getattr(scratch, "key", None) != key:
        total, fields = workspace.workspace(positions)
        offset = torch.zeros(1, dtype=torch.int32, device=positions.device)
        scratch.offset = offset
        scratch.outputs = matrix, shifts, neighbors
        scratch.count = fields["atoms_per_cell_count"]
        scratch.warp_stream = wp.stream_from_torch(stream)
        scratch.bin_count = view(total, wp.int32)
        scratch.dims = view(fields["cells_per_dimension"], wp.vec3i)
        scratch.radius = view(fields["neighbor_search_radius"], wp.vec3i)
        scratch.shifts = view(fields["atom_periodic_shifts"], wp.vec3i)
        scratch.mapping = view(fields["atom_to_cell_mapping"], wp.vec3i)
        scratch.count_warp = view(scratch.count, wp.int32, scan=True)
        scratch.start_indices = fields["cell_atom_start_indices"]
        scratch.starts = view(scratch.start_indices, wp.int32, scan=True)
        scratch.atoms = view(fields["cell_atom_list"], wp.int32)
        scratch.offset_warp = view(offset, wp.int32)
        scratch.matrix = view(matrix, wp.int32)
        scratch.matrix_shifts = view(shifts, wp.vec3i)
        scratch.neighbors = view(neighbors, wp.int32)
        scratch.key = key
    return scratch


def cell_list_stages(scratch, geometry, cutoff, dtype, nodes, *, half_fill=False):
    pos, box, pbc, batch = geometry
    f = dtype(cutoff)
    if legacy_cell_list:
        return (
            (_batch_estimate_cell_list_sizes_overload[dtype], 1,
             (box, pbc, f, MAX_BINS, scratch.bin_count, scratch.radius), 2),
            (_batch_cell_list_construct_bin_size_overload[dtype], 1,
             (box, pbc, scratch.dims, f, MAX_BINS), 3),
            (_batch_cell_list_count_atoms_per_bin_overload[dtype], nodes,
             (pos, box, pbc, batch, scratch.dims, scratch.offset_warp,
              scratch.count_warp, scratch.shifts), None),
            (_batch_cell_list_bin_atoms_overload[dtype], nodes,
             (pos, box, pbc, batch, scratch.dims, scratch.offset_warp,
              scratch.mapping, scratch.count_warp, scratch.starts, scratch.atoms), None),
            (_batch_cell_list_build_neighbor_matrix_overload[dtype], nodes,
             (pos, box, pbc, batch, f, scratch.dims, scratch.radius, scratch.shifts,
              scratch.mapping, scratch.count_warp, scratch.starts, scratch.atoms, scratch.offset_warp,
              scratch.matrix, scratch.matrix_shifts, scratch.neighbors, half_fill), 4),
        )

    device = scratch.warp_stream.device
    wp_vec, wp_mat = (wp.vec3f, wp.mat33f) if dtype == wp.float32 else (wp.vec3d, wp.mat33d)
    # The batched specialization ignores the single-system and optional arguments.
    boolean = empty_sentinel(1, wp.bool, device)
    integer = empty_sentinel(1, wp.int32, device)
    no_cell = empty_sentinel(1, wp_mat, device)
    no_positions = empty_sentinel(1, wp_vec, device)
    no_shifts = empty_sentinel(1, wp.vec3i, device)
    vector_output = empty_sentinel(2, wp_vec, device)
    scalar_output = empty_sentinel(2, dtype, device)
    estimate = get_build_cell_list_kernel("estimate_sizes", dtype, batched=True, min_cells_per_dimension=1)
    construct = get_build_cell_list_kernel("construct_bin_size", dtype, batched=True, min_cells_per_dimension=1)
    count = get_build_cell_list_kernel("count_atoms", dtype, batched=True)
    bin_atoms = get_build_cell_list_kernel("bin_atoms", dtype, batched=True)
    query = get_query_cell_list_kernel(dtype, batched=True, half_fill=half_fill, atom_centric_path="direct")
    return (
        (estimate, 1, (box, boolean, pbc, f, MAX_BINS, scratch.bin_count, integer, scratch.radius), 3),
        (construct, 1, (box, boolean, pbc, integer, scratch.dims, f, MAX_BINS), 5),
        (count, nodes, (pos, box, no_cell, boolean, pbc, batch, integer, scratch.dims,
                        scratch.offset_warp, scratch.count_warp, scratch.shifts), None),
        (bin_atoms, nodes, (pos, box, no_cell, boolean, pbc, batch, integer, scratch.dims,
                            scratch.offset_warp, scratch.mapping,
                            scratch.count_warp, scratch.starts, scratch.atoms), None),
        (query, nodes, (pos, scratch.shifts, no_positions, no_shifts, box, boolean, pbc, batch,
                        f, integer, scratch.dims, integer, scratch.radius, scratch.mapping,
                        scratch.count_warp, scratch.starts, scratch.atoms, scratch.offset_warp, integer,
                        scratch.matrix, scratch.matrix_shifts, scratch.neighbors, vector_output,
                        scalar_output, scalar_output, scalar_output, vector_output, boolean), 8),
    )


def launch_cell_list(scratch, tensors, cutoff, *, half_fill):
    positions, cell, pbc, batch = tensors
    dtype = get_wp_dtype(positions.dtype)
    geometry = (
        view(positions, get_wp_vec_dtype(positions.dtype)),
        view(cell, get_wp_mat_dtype(cell.dtype)),
        view(pbc, wp.bool),
        view(batch, wp.int32),
    )
    stages = cell_list_stages(scratch, geometry, cutoff, dtype, positions.shape[0], half_fill=half_fill)
    scratch.outputs[2].zero_()
    scratch.count.zero_()
    stream = scratch.warp_stream
    for kernel, dim, inputs, _ in stages[:3]:
        wp.launch(kernel, dim=dim, inputs=inputs, stream=stream)
    with wp.ScopedStream(stream, sync_enter=False):
        wp.utils.array_scan(scratch.count_warp, scratch.starts, inclusive=False)
    scratch.count.zero_()
    for kernel, dim, inputs, _ in stages[3:]:
        wp.launch(kernel, dim=dim, inputs=inputs, stream=stream)


class WarpWorkspace:
    def __init__(self):
        self.native = PeriodicCellList()
        self.scratch = local()

    def buffers(self, positions, matrix, shifts, neighbors, stream):
        return bind_workspace(self.scratch, self.native, positions, matrix, shifts, neighbors, stream)

    @torch.no_grad()
    def __call__(self, *, positions, cutoff, cell, pbc, batch_idx, method,
                 neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill):
        args = dict(positions=positions, cutoff=cutoff, cell=cell, pbc=pbc,
                    batch_idx=batch_idx, method=method, neighbor_matrix=neighbor_matrix,
                    neighbor_matrix_shifts=neighbor_matrix_shifts, num_neighbors=num_neighbors,
                    half_fill=half_fill)
        if (
            method != "batch_cell_list" or not positions.is_cuda or cutoff <= 0
            or cell is None or cell.shape != (1, 3, 3) or pbc.shape != (1, 3)
            or positions.dtype != torch.float32 or cell.dtype != torch.float32
            or batch_idx.dtype != torch.int32 or positions.numel() == 0 or half_fill
        ):
            return self.native(**args)

        if invalid_cell(cell):
            raise RuntimeError("Cells with volume <= 0 detected and are not supported. "
                               "Please pass unit cells with `det(cell) > 0.0`.")

        stream = torch.cuda.current_stream(positions.device)
        scratch = self.buffers(positions, neighbor_matrix, neighbor_matrix_shifts, num_neighbors, stream)
        launch_cell_list(scratch, (positions, cell, pbc, batch_idx), cutoff, half_fill=False)
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def argument_descriptor(tensor, dtype):
    descriptor = view(tensor, dtype)
    # Retain the C layout, with no Python reference to the recording input.
    return type(descriptor).from_buffer_copy(descriptor)


class Commands:
    def __init__(self, scratch, tensors, cutoff):
        self.cutoff = cutoff
        self.geometry = tuple(argument_descriptor(t, dtype) for t, dtype in zip(
            tensors, (wp.vec3f, wp.mat33f, wp.bool, wp.int32), strict=True,
        ))
        self.stream = scratch.warp_stream
        stages = cell_list_stages(scratch, self.geometry, cutoff, wp.float32, tensors[0].shape[0])
        self.launches = []
        self.cutoff_args = []
        for kernel, dim, inputs, cutoff_index in stages:
            cmd = wp.launch(kernel, dim=dim, inputs=inputs, stream=self.stream,
                            block_dim=256, record_cmd=True)
            for i, arg in enumerate(inputs):
                if any(arg is descriptor for descriptor in self.geometry) and cmd.params[i + 1] is not arg:
                    raise RuntimeError("Warp no longer retains the mutable array descriptor")
            self.launches.append(cmd)
            if cutoff_index is not None:
                self.cutoff_args.append((cmd, cutoff_index))

    def update(self, tensors, cutoff):
        for descriptor, tensor in zip(self.geometry, tensors, strict=True):
            descriptor.data = tensor.data_ptr()
        if cutoff != self.cutoff:
            for cmd, index in self.cutoff_args:
                cmd.set_param_at_index_from_ctype(index, float(cutoff))
            self.cutoff = cutoff

    def launch(self, scratch):
        estimate, construct, count, bin_atoms, query = self.launches
        with wp.ScopedStream(self.stream, sync_enter=False):
            estimate.launch(stream=self.stream)
            construct.launch(stream=self.stream)
            count.launch(stream=self.stream)
            wp.utils.array_scan(scratch.count_warp, scratch.starts, inclusive=False)
            scratch.count.zero_()
            bin_atoms.launch(stream=self.stream)
            query.launch(stream=self.stream)


class RecordedWorkspace(WarpWorkspace):
    @torch.no_grad()
    def __call__(self, *, positions, cutoff, cell, pbc, batch_idx, method,
                 neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill):
        tensors = positions, cell, pbc, batch_idx
        if (
            method != "batch_cell_list" or not positions.is_cuda
            or not math.isfinite(cutoff) or cutoff <= 0 or cell is None
            or cell.shape != (1, 3, 3) or pbc.shape != (1, 3) or positions.numel() == 0
            or positions.dtype != torch.float32 or cell.dtype != torch.float32
            or pbc.dtype != torch.bool or batch_idx.dtype != torch.int32 or half_fill
            or not all(t.device == positions.device and t.is_contiguous() for t in tensors)
        ):
            return super().__call__(
                positions=positions, cutoff=cutoff, cell=cell, pbc=pbc, batch_idx=batch_idx,
                method=method, neighbor_matrix=neighbor_matrix, neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors, half_fill=half_fill,
            )

        if invalid_cell(cell):
            raise RuntimeError("Cells with volume <= 0 detected and are not supported. "
                               "Please pass unit cells with `det(cell) > 0.0`.")

        stream = torch.cuda.current_stream(positions.device)
        scratch = self.buffers(positions, neighbor_matrix, neighbor_matrix_shifts, num_neighbors, stream)
        key = scratch.key, tuple((t.shape, t.stride()) for t in tensors)
        if getattr(scratch, "record_key", None) != key:
            scratch.commands = Commands(scratch, tensors, cutoff)
            scratch.record_key = key
        else:
            scratch.commands.update(tensors, cutoff)

        num_neighbors.zero_()
        scratch.count.zero_()
        # Retain only launch metadata. The input tensors stay live through
        # CUDA submission and the enclosing graph's status check on this stream.
        scratch.commands.launch(scratch)
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


@triton.jit
def pair_csr(starts, cumulative, pair_rows, nodes: tl.constexpr, block: tl.constexpr):
    node = tl.program_id(0) * block + tl.arange(0, block)
    live = node <= nodes
    e = tl.load(starts + node, live, other=0)
    p = tl.load(cumulative + e - 1, live & (e > 0), other=0)
    tl.store(pair_rows + node, p, live)


@triton.jit
def compact_and_lookup(matrix, shifts, starts, positions, cell, index, offsets, cart, vectors, lookup,
                       nodes: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    node = tl.program_id(0)
    slot = tl.arange(0, block)
    start = tl.load(starts + node)
    stop = tl.load(starts + node + 1)
    total = tl.load(starts + nodes)
    live = (slot < width) & (start + slot < stop)
    src = tl.load(matrix + node * width + slot, live, other=0)
    sx = tl.load(shifts + (node * width + slot) * 3, live, other=0)
    sy = tl.load(shifts + (node * width + slot) * 3 + 1, live, other=0)
    sz = tl.load(shifts + (node * width + slot) * 3 + 2, live, other=0)
    e = start + slot
    tl.store(index + e, src.to(tl.int64), live)
    tl.store(index + total + e, node.to(tl.int64), live)
    offs = (sx, sy, sz)
    for dim in tl.static_range(3):
        tl.store(offsets + e * 3 + dim, offs[dim], live)
        shift = offs[dim].to(tl.float32) * tl.load(cell + 4 * dim)
        left = tl.load(positions + src * 3 + dim, live, other=0.)
        right = tl.load(positions + node * 3 + dim)
        tl.store(cart + e * 3 + dim, shift, live)
        tl.store(vectors + e * 3 + dim, (left - right) + shift, live)
    # High bits count endpoint multiplicity. The low bits store the edge
    # index plus one for unique keys. Ambiguous keys never share radial rows.
    key = src * nodes + node
    tl.atomic_add(lookup + key, (1 << 32) + (e.to(tl.int64) + 1), live, sem="relaxed")


@triton.jit
def find_compact_leaders(index, vector, offsets, distance, species, lookup, starts, reverse, leaders, zero,
                         capacity: tl.constexpr, nodes: tl.constexpr, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    edges = tl.load(starts + nodes)
    live = e < edges
    src = tl.load(index + e, live, other=0).to(tl.int32)
    dst = tl.load(index + edges + e, live, other=0).to(tl.int32)
    same_key = (tl.load(lookup + src * nodes + dst, live, other=0) >> 32) == 1
    back = tl.load(lookup + dst * nodes + src, live, other=0)
    r = back.to(tl.int32) - 1
    eligible = live & same_key & ((back >> 32) == 1) & (r != e)
    same_species = tl.load(species + src, eligible, other=0) == tl.load(species + dst, eligible, other=-1)
    exact = eligible & same_species
    for dim in tl.static_range(3):
        v0 = tl.load(vector + e * 3 + dim, eligible, other=0.)
        v1 = tl.load(vector + r * 3 + dim, eligible, other=0.)
        o0 = tl.load(offsets + e * 3 + dim, eligible, other=0)
        o1 = tl.load(offsets + r * 3 + dim, eligible, other=0)
        exact &= (v0 == -v1) & (o0 == -o1)
    d = tl.load(distance + e, live, other=1.)
    exact &= d == tl.load(distance + r, eligible, other=-1.)
    tl.store(reverse + e, tl.where(exact, r, e), live)
    tl.store(leaders + e, live & ((e <= r) | ~exact), e < capacity)
    if tl.sum((live & (d == 0)).to(tl.int32), 0) > 0:
        tl.atomic_or(zero, 1, sem="relaxed")


@triton.jit
def emit_pairs(reverse, cumulative, leaders, first, pairs, rows, starts,
               capacity: tl.constexpr, nodes: tl.constexpr, block: tl.constexpr):
    e = tl.program_id(0) * block + tl.arange(0, block)
    edges = tl.load(starts + nodes)
    num_pairs = tl.load(cumulative + capacity - 1)
    live = e < edges
    r = tl.load(reverse + e, live, other=0)
    p = tl.load(cumulative + tl.minimum(e, r), live, other=0) - 1
    leader = tl.load(leaders + e, live, other=0)
    tl.store(first + p, e.to(tl.int64), live & leader)
    tl.store(pairs + p, e, live & leader)
    tl.store(pairs + num_pairs + p, tl.where(e == r, -1, r), live & leader)
    tl.store(rows + e, p, live)


class CompactGraph:
    """Build a periodic directed graph and share exact reciprocal radial rows."""

    def __init__(self):
        self.scratch = local()
        self.workspace = FixedCapacityCellList()

    def __call__(self, positions, cell, pbc, batch, species, cutoff, max_neighbors):
        nodes = positions.shape[0]
        if (
            positions.dtype != torch.float32 or not positions.is_cuda
            or not positions.is_contiguous() or not cell.is_contiguous() or cell.dtype != torch.float32
            or cell.shape != (1, 3, 3) or not 0 < nodes <= 2048
        ):
            return None
        width = estimate_max_neighbors(cutoff=cutoff, atomic_density=0.4)
        if not 0 < max_neighbors <= width:
            return None
        diagonal = cell.diagonal(dim1=-2, dim2=-1)
        valid = (pbc.all() & (cell == torch.diag_embed(diagonal)).all()
                 & ((diagonal >= 2.**-30) & (diagonal <= 2.**30)).all())
        if not bool(valid):
            return None

        key = (nodes, width, positions.device, torch.cuda.current_stream(positions.device).cuda_stream)
        if getattr(self.scratch, "key", None) != key:
            self.scratch.buffers = (
                torch.empty((nodes, width), device=positions.device, dtype=torch.int32),
                torch.empty((nodes, width, 3), device=positions.device, dtype=torch.int32),
                torch.empty((nodes,), device=positions.device, dtype=torch.int32),
            )
            self.scratch.key = key
        matrix, shifts, counts = self.scratch.buffers
        self.workspace(
            positions=positions, cutoff=cutoff + 1e-6, cell=cell, pbc=pbc,
            batch_idx=batch.int(), neighbor_matrix=matrix,
            neighbor_matrix_shifts=shifts, num_neighbors=counts, half_fill=False,
        )

        counts_ok = (counts <= max_neighbors).all().to(torch.int32)
        starts = torch.nn.functional.pad(counts.clamp(0, max_neighbors).cumsum(0, dtype=torch.int32), (1, 0))
        capacity = nodes * max_neighbors
        index = torch.empty(2 * capacity, dtype=torch.int64, device=positions.device)
        offsets = matrix.new_empty((capacity, 3))
        cart = positions.new_empty((capacity, 3))
        vectors = positions.new_zeros((capacity, 3))
        lookup = index.new_zeros(nodes * nodes)
        compact_and_lookup[(nodes,)](matrix, shifts, starts, positions, cell, index, offsets, cart, vectors, lookup,
                                      nodes, width, triton.next_power_of_2(max_neighbors), num_warps=4,
                                      enable_fp_fusion=False)
        distance = vectors.norm(dim=-1)
        reverse = matrix.new_empty(capacity)
        leaders = torch.empty(capacity, dtype=torch.bool, device=positions.device)
        zero = counts.new_zeros(())
        grid = (triton.cdiv(capacity, 256),)
        find_compact_leaders[grid](index, vectors, offsets, distance, species, lookup, starts, reverse, leaders,
                                    zero, capacity, nodes, 256, num_warps=4)
        cumulative = leaders.cumsum(0, dtype=torch.int32)
        first, pairs, rows, pair_rows = (
            index.new_empty(capacity), matrix.new_empty(2 * capacity),
            matrix.new_empty(capacity), matrix.new_empty(nodes + 1),
        )
        emit_pairs[grid](reverse, cumulative, leaders, first, pairs, rows, starts, capacity, nodes, 256, num_warps=4)
        pair_csr[(triton.cdiv(nodes + 1, 256),)](starts, cumulative, pair_rows, nodes, 256, num_warps=4)
        edges, num_pairs, okay, had_zero = torch.stack((starts[-1], cumulative[-1], counts_ok, zero)).tolist()
        if not okay or not edges or had_zero:
            return None
        graph = {
            "edge_index": index[:2 * edges].view(2, edges), "edge_distance": distance[:edges],
            "edge_distance_vec": vectors[:edges], "cell_offsets": offsets[:edges], "offset_distances": cart[:edges],
            "neighbors": starts[-1:].to(torch.int64),
        }
        return graph, (first[:num_pairs], rows[:edges], pairs[:2 * num_pairs].view(2, num_pairs), pair_rows)


@triton.jit
def compact(matrix, shifts, row_ptr, positions, cell, index, offsets, cart, vectors,
            NODES: tl.constexpr, WIDTH: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr,
            DIAGONAL_CELL: tl.constexpr = True):
    node = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    start = tl.load(row_ptr + node)
    stop = tl.load(row_ptr + node + 1)
    edge = start + slot
    total = tl.load(row_ptr + NODES)
    live = (slot < LIMIT) & (edge < stop)

    source = tl.load(matrix + node * WIDTH + slot, live, other=0)
    tl.store(index + edge, source.to(tl.int64), live)
    tl.store(index + total + edge, node.to(tl.int64), live)
    shift_ptr = shifts + (node * WIDTH + slot) * 3
    if not DIAGONAL_CELL:
        sx = tl.load(shift_ptr, live, other=0).to(tl.float32)
        sy = tl.load(shift_ptr + 1, live, other=0).to(tl.float32)
        sz = tl.load(shift_ptr + 2, live, other=0).to(tl.float32)
    for dim in tl.static_range(3):
        shift = tl.load(shift_ptr + dim, live, other=0)
        if DIAGONAL_CELL:
            shift_cart = shift.to(tl.float32) * tl.load(cell + 4 * dim)
        else:
            shift_cart = sx * tl.load(cell + dim) + sy * tl.load(cell + 3 + dim) + sz * tl.load(cell + 6 + dim)
        left = tl.load(positions + 3 * source + dim, live, other=0.)
        right = tl.load(positions + 3 * node + dim)
        tl.store(offsets + 3 * edge + dim, shift, live)
        tl.store(cart + 3 * edge + dim, shift_cart, live)
        tl.store(vectors + 3 * edge + dim, (left - right) + shift_cart, live)


def eligible_cell(pbc, cell):
    metadata = geometry_metadata.get()
    if (
        metadata is not None and metadata.pbc is pbc and metadata.cell is cell
        and metadata.pbc_version is not None and metadata.cell_version is not None
        and tensor_version(pbc) == metadata.pbc_version and tensor_version(cell) == metadata.cell_version
    ):
        return metadata.periodic and metadata.positive_diagonal_cell

    diagonal = cell.diagonal(dim1=-2, dim2=-1)
    return bool(pbc.any() & (cell == torch.diag_embed(diagonal)).all()
                & ((diagonal >= 2.**-30) & (diagonal <= 2.**30)).all())


class NativeDirectedGraph:
    """Build a periodic directed graph with CSR rows for receiver reuse."""

    def __init__(self, workspace=None, *, diagonal_cell=True):
        load_periodic_graph_backend()
        self.workspace = WarpWorkspace() if workspace is None else workspace
        self.scratch = local()
        self.diagonal_cell = diagonal_cell

    def supports_cell(self, pbc, cell):
        if self.diagonal_cell:
            return eligible_cell(pbc, cell)
        return any_pbc(pbc) and bool(torch.isfinite(cell).all() & (cell.det() > 0).all())

    @torch.no_grad()
    def __call__(self, positions, cell, pbc, batch, cutoff, max_neighbors):
        nodes = positions.shape[0]
        if (
            not 0 < nodes <= 2048 or positions.shape != (nodes, 3)
            or cell.shape != (1, 3, 3) or pbc.shape != (1, 3) or batch.shape != (nodes,)
            or not positions.is_cuda or positions.dtype != torch.float32 or cell.dtype != torch.float32
            or pbc.dtype != torch.bool or batch.dtype not in (torch.int32, torch.int64)
            or not all(t.is_contiguous() and t.device == positions.device for t in (positions, cell, pbc, batch))
            or not 0 < cutoff or not self.supports_cell(pbc, cell)
        ):
            return None
        width = estimate_max_neighbors(cutoff=cutoff, atomic_density=0.4)
        if not 0 < max_neighbors <= width:
            return None

        stream = torch.cuda.current_stream(positions.device)
        key = (nodes, width, positions.device, stream.cuda_stream)
        if getattr(self.scratch, "key", None) != key:
            self.scratch.buffers = (
                torch.empty((nodes, width), device=positions.device, dtype=torch.int32),
                torch.empty((nodes, width, 3), device=positions.device, dtype=torch.int32),
                torch.empty(nodes, device=positions.device, dtype=torch.int32),
            )
            self.scratch.key = key
        matrix, shifts, counts = self.scratch.buffers
        with device_geometry(pbc, cell, (True, self.diagonal_cell)):
            self.workspace(
                positions=positions, cutoff=cutoff + 1e-6, cell=cell, pbc=pbc,
                batch_idx=batch.int(), method="batch_cell_list", neighbor_matrix=matrix,
                neighbor_matrix_shifts=shifts, num_neighbors=counts, half_fill=False,
            )

        row_ptr = torch.nn.functional.pad(counts.clamp(0, max_neighbors).cumsum(0, dtype=torch.int32), (1, 0))
        capacity = nodes * max_neighbors
        index = torch.empty(2 * capacity, dtype=torch.int64, device=positions.device)
        offsets = matrix.new_empty((capacity, 3))
        cart = positions.new_empty((capacity, 3))
        vectors = positions.new_zeros((capacity, 3))
        compact[(nodes,)](
            matrix, shifts, row_ptr, positions, cell, index, offsets, cart, vectors,
            nodes, width, max_neighbors, triton.next_power_of_2(max_neighbors),
            DIAGONAL_CELL=self.diagonal_cell, num_warps=4, enable_fp_fusion=False,
        )
        distances = vectors.norm(dim=-1)
        counts_ok = ((counts >= 0) & (counts <= max_neighbors)).all()
        # Only live slots count toward the fallback for coincident atoms.
        live = torch.arange(capacity, device=positions.device) < row_ptr[-1]
        invalid_distance = (live & ((distances == 0) | ~distances.isfinite())).any()
        edges, counts_ok, invalid_distance = torch.stack((row_ptr[-1], counts_ok, invalid_distance)).tolist()
        if not edges or not counts_ok or invalid_distance:
            return None
        graph = {
            "edge_index": index[:2 * edges].view(2, edges),
            "edge_distance": distances[:edges],
            "edge_distance_vec": vectors[:edges],
            "cell_offsets": offsets[:edges],
            "offset_distances": cart[:edges],
            "neighbors": row_ptr[-1:].long(),
        }
        return graph, row_ptr


class FixedDirectedGraph:
    """Rebuild one periodic graph inside CUDA capture with a fixed edge count."""

    def __init__(self, data, graph, row_ptr, cutoff, max_neighbors, *, diagonal_cell=True):
        self.data = data
        self.diagonal_cell = diagonal_cell
        self.max_neighbors = max_neighbors
        self.nodes = data.pos.shape[0]
        self.edges = graph["edge_index"].shape[1]
        self.capacity = self.nodes * max_neighbors
        self.width = estimate_max_neighbors(cutoff=cutoff, atomic_density=0.4)
        self.workspace = RecordedWorkspace()
        self.matrix = data.batch.new_empty((self.nodes, self.width), dtype=torch.int32)
        self.shifts = self.matrix.new_empty((self.nodes, self.width, 3))
        self.counts = self.matrix.new_empty(self.nodes)
        self.batch = data.batch.int()
        self.reference = {name: value.detach().clone() for name, value in graph.items()}
        self.reference_ptr = row_ptr.clone()
        stream = torch.cuda.current_stream(data.pos.device)
        self.scratch = self.workspace.buffers(data.pos, self.matrix, self.shifts, self.counts, stream)
        tensors = data.pos, data.cell, data.pbc, self.batch
        self.commands = Commands(self.scratch, tensors, cutoff + 1e-6)
        self.starts = self.scratch.start_indices

    @torch.no_grad()
    def build(self, graph, row_ptr):
        pos, cell = self.data.pos, self.data.cell
        scratch = self.scratch
        estimate, construct, count, bin_atoms, query = self.commands.launches
        stream = self.commands.stream
        self.counts.zero_()
        scratch.count.zero_()
        estimate.launch(stream=stream)
        construct.launch(stream=stream)
        count.launch(stream=stream)
        # Warp's scan releases scratch during an external CUDA capture.
        # Use a Torch scan on the same stream with an existing output buffer.
        self.starts[:1].zero_()
        torch.cumsum(scratch.count[:-1], dim=0, dtype=torch.int32, out=self.starts[1:])
        scratch.count.zero_()
        bin_atoms.launch(stream=stream)
        query.launch(stream=stream)
        ptr = torch.nn.functional.pad(
            self.counts.clamp(0, self.max_neighbors).cumsum(0, dtype=torch.int32), (1, 0),
        )
        index = torch.empty(2 * self.capacity, dtype=torch.int64, device=pos.device)
        offsets = self.matrix.new_empty((self.capacity, 3))
        cart = pos.new_empty((self.capacity, 3))
        vectors = pos.new_zeros((self.capacity, 3))
        compact[(self.nodes,)](
            self.matrix, self.shifts, ptr, pos, cell, index, offsets, cart, vectors,
            self.nodes, self.width, self.max_neighbors,
            triton.next_power_of_2(self.max_neighbors), DIAGONAL_CELL=self.diagonal_cell,
            num_warps=4, enable_fp_fusion=False,
        )
        distances = vectors[:self.edges].norm(dim=-1)
        valid = (
            (ptr[-1] == self.edges)
            & ((self.counts >= 0) & (self.counts <= self.max_neighbors)).all()
            & ((distances > 0) & distances.isfinite()).all()
        )
        candidates = {
            "edge_index": index[:2 * self.edges].view(2, self.edges),
            "edge_distance": distances,
            "edge_distance_vec": vectors[:self.edges],
            "cell_offsets": offsets[:self.edges],
            "offset_distances": cart[:self.edges],
            "neighbors": ptr[-1:].long(),
        }
        # An invalid geometry still executes the captured model. Use a known
        # valid graph in that case so the model cannot read invalid indices.
        for name, value in candidates.items():
            torch.where(valid, value, self.reference[name], out=graph[name].detach())
        torch.where(valid, ptr, self.reference_ptr, out=row_ptr)
        return valid


def compact_counts(matrix, shifts, counts, max_neigh):
    width = matrix.shape[1]
    if not counts.numel():
        return counts.new_empty((0,), dtype=torch.long), matrix.reshape(-1), shifts.reshape(-1, 3), False
    # nvalchemi increments the raw count even when the matrix row is full.
    # Use clipped counts for COO lengths and raw counts to detect a cap.
    live = counts.clamp(0, width)
    ends = live.cumsum(0, dtype=torch.int64)
    cap = (counts > max_neigh).any().to(ends.dtype)
    edges, capped = torch.stack((ends[-1], cap)).tolist()
    flat = torch.arange(edges, device=counts.device)
    center = torch.searchsorted(ends, flat, right=True)
    slots = flat - (ends - live)[center]
    selected = center * width + slots
    neighbor = matrix.flatten().index_select(0, selected)
    offset = shifts.reshape(-1, 3).index_select(0, selected)
    return center, neighbor, offset, bool(capped)


@torch.compiler.disable()
def edge_vjp(energy, vectors, distances):
    return torch.autograd.grad(energy, (vectors, distances))


@torch.no_grad()
def force_stress_torch(grad_v, grad_d, vectors, distances, edge_index, cell, nodes):
    # A dummy edge can be present in the model's graph for an isolated system.
    # It does not depend on positions or the cell.
    edges = edge_index.shape[1]
    vectors = vectors[:edges]
    grad = grad_v[:edges] + grad_d[:edges, None] * (vectors / distances[:edges, None])
    source, target = edge_index
    forces = grad.new_zeros((nodes, 3))
    forces.index_add_(0, source, -grad)
    forces.index_add_(0, target, grad)

    # For v_e = r_src - r_dst + offset @ cell, the symmetric strain derivative
    # is the symmetric outer product of dE/dv_e with the full periodic vector.
    virial = (grad[:, :, None] * vectors[:, None, :]).sum(dim=0, dtype=torch.float64)
    stress = (virial + virial.mT) / (2 * torch.det(cell).abs())
    return forces, stress.reshape(1, 9)


@triton.jit
def accumulate_edges(
    grad_v, grad_d, vectors, distance, indices, forces, virial,
    edges, index_stride, B: tl.constexpr,
):
    tile = tl.program_id(0)
    e = tile * B + tl.arange(0, B)
    live = e < edges
    src = tl.load(indices + e, live, 0)
    dst = tl.load(indices + e + index_stride, live, 0)
    d = tl.load(distance + e, live, 1.0)
    g_d = tl.load(grad_d + e, live, 0.0)
    displacement = ()
    grad = ()
    for axis in tl.static_range(3):
        v = tl.load(vectors + 3 * e + axis, live, 0.0)
        g = tl.load(grad_v + 3 * e + axis, live, 0.0) + g_d * (v / d)
        tl.atomic_add(forces + 3 * src + axis, -g, live & (src != dst), sem="relaxed")
        tl.atomic_add(forces + 3 * dst + axis, g, live & (src != dst), sem="relaxed")
        displacement += (v,)
        grad += (g,)
    for i in tl.static_range(3):
        for j in tl.static_range(i, 3):
            value = (grad[i] * displacement[j]).to(tl.float64)
            if i != j:
                value += (grad[j] * displacement[i]).to(tl.float64)
            value = tl.sum(value, 0)
            tl.store(virial + tile * 9 + i * 3 + j, value)
            if i != j:
                tl.store(virial + tile * 9 + j * 3 + i, value)


@triton.jit
def finish_stress(virial, cell, stress, tiles, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(virial + tile * 9 + i, tile < tiles, 0.0), 0)
    a0, a1, a2 = tl.load(cell), tl.load(cell + 1), tl.load(cell + 2)
    b0, b1, b2 = tl.load(cell + 3), tl.load(cell + 4), tl.load(cell + 5)
    c0, c1, c2 = tl.load(cell + 6), tl.load(cell + 7), tl.load(cell + 8)
    determinant = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
    factor = 1.0 if i % 4 == 0 else 0.5
    tl.store(stress + i, total * factor / tl.abs(determinant).to(tl.float64))


@triton_op("trim_h100::force_virial", mutates_args=())
def force_stress(
    grad_v: Tensor, grad_d: Tensor, vectors: Tensor, distances: Tensor,
    edge_index: Tensor, cell: Tensor, nodes: int,
) -> tuple[Tensor, Tensor]:
    edges = edge_index.shape[1]
    block = 256
    tiles = torch.sym_max(1, triton.cdiv(edges, block))
    forces = grad_v.new_zeros((nodes, 3))
    virial = grad_v.new_empty((tiles, 9), dtype=torch.float64)
    stress = grad_v.new_empty((1, 9), dtype=torch.float64)
    wrap_triton(accumulate_edges)[(tiles,)](
        grad_v.contiguous(), grad_d.contiguous(), vectors.contiguous(),
        distances.contiguous(), edge_index, forces, virial, edges, edge_index.stride(0),
        block, num_warps=4,
    )
    # Bound per-program scratch for all supported edge counts via one
    # reduction per component. The 1,000-atom trial has 305 tiles.
    sums = virial.sum(0, keepdim=True)
    wrap_triton(finish_stress)[(9,)](sums, cell.contiguous(), stress, 1, 1, num_warps=1)
    return forces, stress
