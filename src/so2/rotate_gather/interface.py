"""Gather, rotate and scale node features directly into SO(2) order sectors."""

from __future__ import annotations

from functools import partial

import torch
import triton
from torch.autograd.function import once_differentiable
from torch.library import triton_op, wrap_triton
from torch.nn.functional import linear

from ..utils.graph import graph_tables, prefix_csr_metadata
from ..utils.layout import get_angular_layout, get_retained_layout
from ..utils.tensor_contracts import allow_atomic_reduction
from .kernels.backward import (
    degree_shared_backward,
    paired_prefix_backward_kernel,
    gather_backward_atomic,
    gather_backward_source_csr,
    prefix_degree_backward_kernel,
    prefix_radial_wigner_backward_kernel,
    structured_prefix_input_radial_gradient_kernel,
    structured_prefix_wigner_gradient_kernel,
    use_order_degree_backward,
    wigner_conv1_fused_bwd_kernel,
)
from .kernels.double_backward import (
    compact_double_backward,
    degree_shared_double_backward,
    gather_double_backward_source_csr,
    structured_prefix_double_backward_kernel,
    structured_prefix_double_backward_wigner_gradient_kernel,
)
from .kernels.forward import (
    degree_shared_forward,
    prefix_sector_forward,
    prefix_sector_forward_kernel,
    sector_prefix_kernel,
    wigner_conv1_fused_fwd_kernel,
)
from .kernels.triple_backward import (
    GatherDoubleBackward,
    degree_shared_mixed_gradients,
    mixed_gather_gradients,
    mixed_sector_gradients,
    use_mixed_sector_gradients,
)
from .kernels.utils import (
    FUSED_WIGNER_GRID_E_STRIDE,
    channel_launch,
    default_block_channels,
    sector_metadata,
    structured_prefix_generated_kernel,
    use_l2_edge_schedule,
    validate_inputs,
    validate_prefix_inputs,
)


class FrozenRadialGather(torch.autograd.Function):
    """Recompute a frozen radial projection when differentiating geometry."""

    @staticmethod
    def forward(ctx, x, edge_index, hidden, weight, bias, rotations, lmax, mmax):
        ctx.save_for_backward(x, edge_index, hidden, weight, bias, rotations)
        ctx.dimensions = lmax, mmax
        return prefix_sector_forward(
            x, edge_index, linear(hidden, weight, bias), rotations, lmax=lmax, mmax=mmax,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_sectors):
        x, edge_index, hidden, weight, bias, rotations = ctx.saved_tensors
        lmax, mmax = ctx.dimensions
        radial = linear(hidden, weight, bias)
        dx, dr, dw = gather_backward_atomic(
            grad_sectors, x, edge_index, radial, rotations,
            lmax=lmax, mmax=mmax, channels=x.shape[-1],
        )
        del radial
        dhidden = dr @ weight if ctx.needs_input_grad[2] else None
        return dx, None, dhidden, None, None, dw, None, None


def rotate_gather(
    x, rotations, radial, graph, *, lmax, mmax, radial_sharing="order_degree"
):
    """Gather both endpoints, rotate and scale into physical order sectors.

    ``x`` is ``(N,(L+1)**2,C)``. ``rotations`` contains compact retained
    rows, and ``radial`` contains endpoint channel weights. For sharing
    ``"degree"``, radial has shape ``(E,L+1,2*C)``. For
    ``"order_degree"``, it is ``(E,2*C*sum(L+1-m for m in 0..M))``.
    ``graph`` is a receiver-sorted ``LunaTPGraph``. All edge tensors follow
    its ordering. Return one physical sector per order: ``(E,(L+1)*2*C)``
    for order zero, and ``(E,2,(L+1-m)*2*C)`` otherwise.

    Floating inputs use CUDA FP32 and support derivatives through third
    order. Graph indices are fixed during differentiation.
    """
    edge_index, metadata = graph_tables(graph)
    if x.shape[0] != graph.num_nodes:
        raise ValueError("node features must match the graph node count")
    return gather_rotated_sectors(
        x, edge_index, radial, rotations,
        lmax=lmax, mmax=mmax, radial_sharing=radial_sharing,
        rotation_storage="retained_rows",
        metadata=metadata if radial_sharing == "order_degree" else None,
    )


def gather_rotated_sectors(
    x, edge_index, radial, rotations,
    *,
    lmax, mmax,
    radial_sharing,
    rotation_storage,
    metadata=None,
):
    """Gather both endpoints, rotate to the edge frame, and scale by radial weights.

    The materialized PyTorch calculation is
    ``local = D @ cat((x[source], x[receiver]), dim=-1)``, followed by
    elementwise radial scaling and grouping by local angular order. The
    kernel combines these steps without allocating gathered node features.

    Parameters
    ----------
    x : Tensor, shape (N, F, C)
        Node features, with ``F=(L+1)**2``. Full angular rows use degree-major
        indices ``l*l+l+m`` for degree ``l`` and signed order ``m``.
    edge_index : int64 Tensor, shape (2, E)
        Row zero contains source nodes and row one contains receiver nodes.
        The order-degree schedule requires edges sorted by receiver.
    radial : Tensor
        External differentiable weights for the endpoint channels. With
        ``radial_sharing="degree"``, shape ``(E, L+1, 2*C)`` and one map
        per degree, shared by all its retained orders. With
        ``radial_sharing="order_degree"``, shape ``(E, 2*C*G)``, where
        ``G=sum(L+1-m for m in range(M+1))``. Groups run through ``m=0..M``
        and then ``l=m..L``. Each group contains source channels followed
        by receiver channels. The positive and negative orders share a map.
    rotations : Tensor
        Forward global-to-local rotations in the same real-harmonic basis as
        ``x``. ``rotation_storage="dense"`` takes ``(E, F, F)`` and uses
        only within-degree entries. ``"retained_rows"`` takes ``(E, K)``,
        where ``K=sum((2*min(l,M)+1)*(2*l+1) for l in range(L+1))``.
        Compact blocks run through increasing degree and then local rows
        ``0,+1,-1,...``. Each row stores all ``2*l+1`` input components.
        Use :func:`LunaEquivariance.so2.rotation.interface.pack_retained_rotations` to pack dense
        rotations. Degree-shared weights require retained-row storage.
    lmax, mmax : int
        ``L >= 0`` and ``0 <= M <= L``.
    radial_sharing : {"degree", "order_degree"}
        Select the radial weight contract above.
    rotation_storage : {"dense", "retained_rows"}
        Select the rotation contract above.
    metadata : tuple or None
        Reusable graph indices from
        :func:`LunaEquivariance.so2.utils.graph.prefix_csr_metadata` for the order-degree path.
        They must describe this exact edge order. The degree path takes
        ``None``. :func:`LunaEquivariance.so2.utils.graph.prepare_graph` returns sorted edges,
        their permutation, and reusable metadata. Apply the permutation to
        radial and rotation inputs too.

    Returns
    -------
    tuple[Tensor, ...]
        One sector per nonnegative order. Sector zero is
        ``(E, (L+1)*2*C)``. Sector ``m>0`` is
        ``(E, 2, (L+1-m)*2*C)``. Its two planes contain ``+m`` and ``-m``.
        Within each plane, degree increases and each degree contains source
        channels followed by receiver channels. These are physical
        coefficients ready for :func:`LunaEquivariance.so2.linear.interface.project_sectors`.

    All floating inputs must be contiguous CUDA FP32 on one device. Edge
    indices must also be contiguous on that device.
    Features, radial weights, and rotation entries support derivatives through
    third order. Indices, metadata, and dimensions are constant.
    """
    if x.ndim != 3 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "node features and edge_index must have shapes (N, d, C) and (2, E)"
        )
    if any(value.device != x.device for value in (edge_index, radial, rotations)):
        raise ValueError("node-to-edge inputs must share a device")

    if radial_sharing == "degree" and rotation_storage == "retained_rows":
        if metadata is not None:
            raise ValueError("degree-shared prefix does not consume CSR metadata")
        return rotate_gather_degree(
            x, edge_index, radial, rotations,
            lmax=lmax, mmax=mmax,
            channels=x.shape[2],
            num_warps=2,
        )

    if radial_sharing == "order_degree" and rotation_storage in (
        "dense",
        "retained_rows",
    ):
        layout = get_angular_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
        retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
        shape = (
            (layout.full_components, layout.full_components)
            if rotation_storage == "dense"
            else (retained.compact_elements,)
        )
        if tuple(rotations.shape[1:]) != shape:
            raise ValueError("rotation tensor does not match its declared storage")
        return rotate_gather_order_degree(
            x, edge_index, radial, rotations,
            lmax=lmax, mmax=mmax,
            metadata=metadata,
            sectorized=True,
        )

    raise ValueError("unsupported radial sharing and rotation storage combination")


def use_degree_shared_gather(x, edges, lmax, mmax, channels):
    if (
        degree_shared_double_backward is None
        or (lmax, mmax, channels) != (4, 2, 128)
        or not 16384 <= edges <= 32768
        or torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    return x.is_cuda and torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))


class DegreePrefixBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        sector_count = len(inputs) - 9
        grad_sectors = inputs[:sector_count]
        (
            x, edge_index, compact_radial, wigner,
            lmax, mmax, channels,
            block_channels, num_warps,
        ) = inputs[sector_count:]
        ctx.save_for_backward(*grad_sectors, x, edge_index, compact_radial, wigner)
        ctx.sector_count = sector_count
        ctx.dimensions = (lmax, mmax, channels, block_channels, num_warps)
        return DegreePrefixBackward.run_vjp(
            grad_sectors, (x, compact_radial, wigner),
            edge_index=edge_index, dimensions=ctx.dimensions,
        )

    @staticmethod
    def run_vjp(grad_sectors, values, *, edge_index, dimensions):
        x, compact_radial, wigner = values
        lmax, mmax, channels, block_channels, num_warps = dimensions
        allow_atomic_reduction("degree-shared node-to-edge backward")

        spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
        if len(grad_sectors) != spec.mmax + 1:
            raise ValueError("prefix sector cotangents do not match M")
        edges = edge_index.shape[1]
        if use_degree_shared_gather(x, edges, lmax, mmax, channels):
            return degree_shared_backward(
                grad_sectors, x, edge_index, compact_radial, wigner, lmax=lmax, mmax=mmax,
            )

        dx = torch.zeros_like(x)
        dradial = torch.empty_like(compact_radial)
        dwigner = torch.zeros_like(wigner)
        grid = (edges, 2 * triton.cdiv(spec.channels, block_channels))
        structured_prefix_input_radial_gradient_kernel[grid](
            tuple(grad_sectors), 0, 0, 0,
            x, edge_index, compact_radial, wigner,
            wigner.stride(0), 0, 1,
            dx, dradial, edges,
            channels=spec.channels,
            lmax=spec.lmax,
            full_components=spec.full_components,
            degree_count=spec.degree_count,
            degree_row_counts=spec.degree_row_counts,
            reduced_components=spec.reduced_components,
            compact_row_offsets=spec.compact_row_offsets,
            block_channels=block_channels,
            sector_layout=True,
            num_warps=num_warps,
        )
        for (
            degree,
            row_count,
            component_start,
            component_width,
            component_block,
        ) in spec.degree_settings:
            structured_prefix_wigner_gradient_kernel[(edges, row_count)](
                tuple(grad_sectors), 0, 0, 0,
                x, edge_index, compact_radial,
                dwigner,
                edges,
                channels=spec.channels,
                channel_block=1 << (spec.channels - 1).bit_length(),
                lmax=spec.lmax,
                full_components=spec.full_components,
                reduced_components=spec.reduced_components,
                compact_elements=spec.compact_elements,
                compact_row_offsets=spec.compact_row_offsets,
                degree_count=spec.degree_count,
                degree=degree,
                component_start=component_start,
                component_width=component_width,
                component_block=component_block,
                sector_layout=True,
                num_warps=num_warps,
            )
        return dx, dradial, dwigner

    @staticmethod
    def backward(ctx, input_direction, radial_direction, wigner_direction):
        saved = ctx.saved_tensors
        grad_sectors = saved[: ctx.sector_count]
        x, edge_index, compact_radial, wigner = saved[ctx.sector_count :]
        if input_direction is None:
            input_direction = torch.zeros_like(x)
        if radial_direction is None:
            radial_direction = torch.zeros_like(compact_radial)
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        values = x, compact_radial, wigner
        directions = (
            input_direction.contiguous(),
            radial_direction.contiguous(),
            wigner_direction.contiguous(),
        )
        if torch.is_grad_enabled():
            hvp = partial(DegreePrefixBackward.hvp, edge_index=edge_index, dimensions=ctx.dimensions)
            vjp = partial(DegreePrefixBackward.run_vjp, edge_index=edge_index, dimensions=ctx.dimensions)
            lmax, mmax, channels = ctx.dimensions[:3]
            mixed = None
            if (degree_shared_mixed_gradients is not None
                    and use_degree_shared_gather(x, edge_index.shape[1], lmax, mmax, channels)):
                mixed = partial(degree_shared_mixed_gradients, edge_index=edge_index)
            gradients = GatherDoubleBackward.apply(
                *values, *directions, hvp, vjp, True, mixed, None, *grad_sectors,
            )
        else:
            gradients = DegreePrefixBackward.hvp(
                grad_sectors, values, directions, edge_index=edge_index, dimensions=ctx.dimensions,
            )
        *grad_outputs, dx, dradial, dwigner = gradients
        return (*grad_outputs, dx, None, dradial, dwigner, *(None for _ in range(5)))

    @staticmethod
    def hvp(
        grad_sectors, values, directions, *, edge_index, dimensions,
        output_mask=(True, True, True, True),
    ):
        x, compact_radial, wigner = values
        input_direction, radial_direction, wigner_direction = directions
        lmax, mmax, channels, block_channels, num_warps = dimensions
        allow_atomic_reduction("degree-shared node-to-edge double backward")

        if use_degree_shared_gather(x, edge_index.shape[1], lmax, mmax, channels):
            return degree_shared_double_backward(
                grad_sectors, x, edge_index, compact_radial, wigner,
                input_direction, radial_direction, wigner_direction,
                lmax=lmax, mmax=mmax, output_mask=output_mask,
            )

        spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
        if len(grad_sectors) != spec.mmax + 1:
            raise ValueError("prefix sector cotangents do not match M")
        edges = edge_index.shape[1]
        grad_outputs = tuple(torch.empty_like(value) for value in grad_sectors)
        dx = torch.zeros_like(x)
        dradial = torch.empty_like(compact_radial)
        dwigner = torch.zeros_like(wigner)
        grid = (edges, 2 * triton.cdiv(spec.channels, block_channels))
        structured_prefix_double_backward_kernel[grid](
            tuple(grad_sectors), 0, 0, 0,
            x, edge_index, compact_radial, wigner,
            wigner.stride(0), 0, 1,
            input_direction,
            input_direction.stride(0), input_direction.stride(1), input_direction.stride(2),
            radial_direction,
            radial_direction.stride(0), radial_direction.stride(1), radial_direction.stride(2),
            wigner_direction,
            wigner_direction.stride(0), 0, 1,
            grad_outputs,
            dx, dradial, edges,
            channels=spec.channels,
            lmax=spec.lmax,
            full_components=spec.full_components,
            reduced_components=spec.reduced_components,
            degree_count=spec.degree_count,
            degree_row_counts=spec.degree_row_counts,
            compact_row_offsets=spec.compact_row_offsets,
            block_channels=block_channels,
            sector_layout=True,
            num_warps=num_warps,
        )
        for (
            degree,
            row_count,
            component_start,
            component_width,
            component_block,
        ) in spec.degree_settings:
            structured_prefix_double_backward_wigner_gradient_kernel[(edges, row_count)](
                tuple(grad_sectors), 0, 0, 0,
                x, edge_index, compact_radial,
                input_direction,
                input_direction.stride(0), input_direction.stride(1), input_direction.stride(2),
                radial_direction,
                radial_direction.stride(0), radial_direction.stride(1), radial_direction.stride(2),
                dwigner,
                edges,
                channels=spec.channels,
                channel_block=1 << (spec.channels - 1).bit_length(),
                lmax=spec.lmax,
                full_components=spec.full_components,
                reduced_components=spec.reduced_components,
                compact_elements=spec.compact_elements,
                compact_row_offsets=spec.compact_row_offsets,
                degree_count=spec.degree_count,
                degree=degree,
                component_start=component_start,
                component_width=component_width,
                component_block=component_block,
                sector_layout=True,
                num_warps=num_warps,
            )

        return *grad_outputs, dx, dradial, dwigner


class DegreePrefix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, compact_radial, wigner,
        lmax, mmax, channels,
        block_channels, num_warps,
    ):
        spec = validate_prefix_inputs(
            x, edge_index, compact_radial, wigner,
            lmax=lmax, mmax=mmax,
            block_channels=block_channels, num_warps=num_warps,
        )
        edges = edge_index.shape[1]
        common = {"device": x.device, "dtype": x.dtype}
        sectors = []
        for order, width in enumerate(spec.sector_widths):
            if order == 0:
                shape = (edges, width * 2 * spec.channels)
            else:
                shape = (edges, 2, width * 2 * spec.channels)
            sectors.append(torch.empty(shape, **common))
        sectors = tuple(sectors)
        if use_degree_shared_gather(x, edges, lmax, mmax, channels):
            degree_shared_forward(x, edge_index, compact_radial, wigner, sectors, spec)
        else:
            grid = (edges, 2 * triton.cdiv(spec.channels, block_channels))
            structured_prefix_generated_kernel[grid](
                x, edge_index, compact_radial, wigner,
                wigner.stride(0), 0, 1,
                sectors,
                edges,
                channels=spec.channels,
                full_components=spec.full_components,
                reduced_components=spec.reduced_components,
                degree_count=spec.degree_count,
                reduced_degrees=spec.reduced_degrees,
                compact_row_offsets=spec.compact_row_offsets,
                block_channels=block_channels,
                sector_output=True,
                num_warps=num_warps,
            )
        ctx.save_for_backward(x, edge_index, compact_radial, wigner)
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.channels = channels
        ctx.block_channels = block_channels
        ctx.num_warps = num_warps
        return sectors

    @staticmethod
    def backward(ctx, *grad_sectors):
        x, edge_index, compact_radial, wigner = ctx.saved_tensors
        if len(grad_sectors) != ctx.mmax + 1 or any(
            gradient is None for gradient in grad_sectors
        ):
            raise RuntimeError("all prefix sectors must participate in backward")
        gradients = DegreePrefixBackward.apply(
            *(gradient.contiguous() for gradient in grad_sectors),
            x, edge_index, compact_radial, wigner,
            ctx.lmax, ctx.mmax, ctx.channels,
            ctx.block_channels, ctx.num_warps,
        )
        return gradients[0], None, gradients[1], gradients[2], None, None, None, None, None


def rotate_gather_degree(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    compact_radial: torch.Tensor,
    wigner: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    channels: int,
    block_channels: int | None = None,
    num_warps: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Write independent cuBLAS sector outputs with explicit VJP and HVP."""

    if block_channels is None:
        block_channels = default_block_channels(lmax=lmax, channels=channels)
    return DegreePrefix.apply(
        x, edge_index, compact_radial, wigner,
        lmax, mmax, channels,
        block_channels, num_warps,
    )


@triton_op("so2::structured_prefix_forward", mutates_args=("output",))
def flat_prefix_forward_out(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    output: torch.Tensor,
    lmax: int,
    mmax: int,
    channels: int,
) -> None:
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    radial_row_indices = tuple(
        offset // (2 * channels) for offset in spec.radial_row_offsets
    )
    radial_rows = spec.radial_elements // (2 * channels)
    block_channels = max(128, triton.next_power_of_2(channels))
    num_warps = 1 if block_channels == 128 else 4

    edges = edge_index.shape[1]
    grid = (edges, 2 * triton.cdiv(channels, block_channels))
    wrap_triton(sector_prefix_kernel)[grid](
        x, edge_index, radial, wigner,
        output,
        edges,
        channels=channels,
        full_components=spec.full_components,
        reduced_components=spec.reduced_components,
        reduced_degrees=spec.reduced_degrees,
        dense_rows=spec.dense_rows,
        radial_row_indices=radial_row_indices,
        radial_rows=radial_rows,
        wide_index=False,
        block_channels=block_channels,
        retained_rows=wigner.ndim == 2,
        compact_elements=retained.compact_elements,
        compact_row_offsets=retained.compact_row_offsets,
        num_warps=num_warps,
    )


@triton_op(
    "so2::compact_prefix_forward",
    mutates_args=("outputs",),
)
def compact_prefix_forward(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    outputs: list[torch.Tensor],
    lmax: int,
    mmax: int,
    channels: int,
    num_warps: int,
) -> None:
    if use_l2_edge_schedule(
        x, edge_index, radial, wigner, lmax, mmax, channels, forward=True,
    ):
        edges = edge_index.shape[1]

        def grid(meta):
            return (torch.sym_max(1, torch.sym_min(edges, FUSED_WIGNER_GRID_E_STRIDE)),)

        wrap_triton(wigner_conv1_fused_fwd_kernel)[grid](
            x,
            edge_index,
            wigner.view(edges, 35),
            radial,
            outputs[0],
            outputs[1].view(edges, 1024),
            outputs[2].view(edges, 512),
            edges,
            channels,
            x.stride(0), x.stride(1), x.stride(2),
            edge_index.stride(0),
            BLOCK_C=channels,
            GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
            num_warps=1,
        )
        return
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    offsets, sector_elements = sector_metadata(spec)
    radial_row_indices = tuple(
        offset // (2 * channels) for offset in spec.radial_row_offsets
    )

    def grid(meta):
        return (edge_index.shape[1], 2)

    wrap_triton(prefix_sector_forward_kernel)[grid](
        x, edge_index, radial, wigner,
        tuple(outputs),
        edge_index.shape[1],
        channels=channels,
        full_components=spec.full_components,
        reduced_components=spec.reduced_components,
        reduced_degrees=spec.reduced_degrees,
        dense_rows=spec.dense_rows,
        radial_row_indices=radial_row_indices,
        radial_rows=spec.radial_elements // (2 * channels),
        mmax=mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=max(128, triton.next_power_of_2(channels)),
        compact_elements=wigner.stride(0),
        compact_wigner=True,
        retained_mmax=mmax if wigner.ndim == 2 else -1,
        num_warps=num_warps,
    )


@triton_op(
    "so2::compact_prefix_backward",
    mutates_args=("input_gradient", "radial_gradient", "wigner_gradient"),
)
def compact_prefix_backward(
    grads: list[torch.Tensor],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    input_gradient: torch.Tensor,
    radial_gradient: torch.Tensor,
    wigner_gradient: torch.Tensor,
    lmax: int,
    mmax: int,
    channels: int,
    num_warps: int,
) -> None:
    allow_atomic_reduction("compact prefix backward")
    if use_l2_edge_schedule(x, edge_index, radial, wigner, lmax, mmax, channels):
        edges = edge_index.shape[1]
        ampere = torch.cuda.get_device_capability(x.device) == (8, 0)
        if (ampere and paired_prefix_backward_kernel is not None
                and edges >= 8_192 and edges >= 8 * x.shape[0]):
            def paired_grid(meta):
                return (triton.cdiv(edges, 2),)

            for degree in range(3):
                wrap_triton(paired_prefix_backward_kernel)[paired_grid](
                    tuple(grads), x, edge_index, radial, wigner,
                    input_gradient, radial_gradient, wigner_gradient,
                    edges, degree, num_warps=4,
                )
            return

        def grid(meta):
            return (torch.sym_max(1, torch.sym_min(edges, FUSED_WIGNER_GRID_E_STRIDE)),)

        wrap_triton(wigner_conv1_fused_bwd_kernel)[grid](
            grads[0],
            grads[1].view(edges, 1024),
            grads[2].view(edges, 512),
            wigner.view(edges, 35),
            radial,
            x,
            edge_index,
            input_gradient,
            wigner_gradient.view(edges, 35),
            radial_gradient,
            edges,
            x.stride(0), x.stride(1), x.stride(2),
            edge_index.stride(0),
            C=channels,
            DIRECT_SCATTER=True,
            BLOCK_C=channels,
            GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
            num_warps=2 if ampere else 1,
        )
        return
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    edges = edge_index.shape[1]
    if use_order_degree_backward(grads, x, radial, wigner, edges, lmax, mmax, channels):
        ampere = torch.cuda.get_device_capability(x.device) == (8, 0)
        retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
        for degree, rows in enumerate(retained.degree_rows):
            radial_offsets = tuple(
                spec.radial_row_offsets[rows[0 if m == 0 else 2 * m - 1]]
                for m in range(min(degree, mmax) + 1)
            )
            wrap_triton(prefix_degree_backward_kernel)[(edges,)](
                tuple(grads), x, edge_index, radial, wigner,
                input_gradient, radial_gradient, wigner_gradient, edges,
                lmax, mmax, channels, degree, wigner.stride(0), retained.compact_row_offsets[rows[0]],
                radial.shape[1], radial_offsets, num_warps=4 if ampere and degree >= 2 else 1,
            )
        return
    offsets, sector_elements = sector_metadata(spec)
    # Finish each degree before starting the next to shorten BW1 accumulator
    # lifetimes. Other degrees retain the general schedule.
    degree_schedule = lmax == 2 and mmax > 0
    groups = (
        sorted(
            range(len(spec.radial_group_rows0)),
            key=lambda group: spec.radial_group_degrees[group],
        )
        if degree_schedule
        else range(len(spec.radial_group_rows0))
    )
    group_degrees = tuple(spec.radial_group_degrees[group] for group in groups)
    group_last = tuple(
        group == len(group_degrees) - 1 or degree != group_degrees[group + 1]
        for group, degree in enumerate(group_degrees)
    )
    wrap_triton(prefix_radial_wigner_backward_kernel)[(edge_index.shape[1],)](
        tuple(grads),
        x, edge_index, radial, wigner,
        radial_gradient,
        wigner_gradient,
        input_gradient,
        edge_index.shape[1],
        channels=channels,
        full_components=spec.full_components,
        radial_group_rows0=tuple(spec.radial_group_rows0[group] for group in groups),
        radial_group_rows1=tuple(spec.radial_group_rows1[group] for group in groups),
        radial_group_dense_rows0=tuple(
            spec.radial_group_dense_rows0[group] for group in groups
        ),
        radial_group_dense_rows1=tuple(
            spec.radial_group_dense_rows1[group] for group in groups
        ),
        radial_group_degrees=tuple(
            spec.radial_group_degrees[group] for group in groups
        ),
        radial_group_offsets=tuple(
            spec.radial_group_offsets[group] for group in groups
        ),
        radial_group_count=len(spec.radial_group_rows0),
        radial_elements=spec.radial_elements,
        mmax=mmax,
        offsets=offsets,
        elements=sector_elements,
        block_channels=max(128, triton.next_power_of_2(channels)),
        compact_elements=wigner.stride(0),
        compact_wigner=True,
        retained_mmax=mmax if wigner.ndim == 2 else -1,
        atomic_input=True,
        degree_schedule=degree_schedule,
        group_last=group_last,
        num_warps=max(4, num_warps),
    )


class CSRPrefixBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, radial, wigner,
        source_order,
        source_row_ptr,
        receiver_row_ptr,
        lmax,
        mmax,
        fused_vjp,
        atomic_vjp,
        *grad_outputs,
    ):
        ctx.save_for_backward(
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
            *grad_outputs,
        )
        ctx.dimensions = (lmax, mmax, fused_vjp, atomic_vjp)
        return CSRPrefixBackward.run_vjp(
            grad_outputs, (x, radial, wigner), edge_index=edge_index,
            metadata=(source_order, source_row_ptr, receiver_row_ptr), dimensions=ctx.dimensions,
        )

    @staticmethod
    def run_vjp(grad_outputs, values, *, edge_index, metadata, dimensions):
        x, radial, wigner = values
        lmax, mmax, fused_vjp, atomic_vjp = dimensions
        if atomic_vjp:
            return gather_backward_atomic(
                grad_outputs, x, edge_index, radial, wigner,
                lmax=lmax, mmax=mmax, channels=x.shape[2],
            )
        return gather_backward_source_csr(
            grad_outputs, x, edge_index, radial, wigner, metadata,
            lmax=lmax, mmax=mmax, fused_vjp=fused_vjp,
        )

    @staticmethod
    def backward(ctx, input_direction, radial_direction, wigner_direction):
        (
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
            *grad_outputs,
        ) = ctx.saved_tensors
        if input_direction is None:
            input_direction = torch.zeros_like(x)
        if radial_direction is None:
            radial_direction = torch.zeros_like(radial)
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        values = x, radial, wigner
        directions = (
            input_direction.contiguous(),
            radial_direction.contiguous(),
            wigner_direction.contiguous(),
        )
        metadata = source_order, source_row_ptr, receiver_row_ptr
        if torch.is_grad_enabled():
            hvp = partial(CSRPrefixBackward.hvp, edge_index=edge_index, metadata=metadata,
                          dimensions=ctx.dimensions)
            vjp = partial(CSRPrefixBackward.run_vjp, edge_index=edge_index, metadata=metadata,
                          dimensions=ctx.dimensions)
            atomic_vjp = ctx.dimensions[-1]
            mixed_gradients = sector_hessian = None
            if use_mixed_sector_gradients(x, radial, wigner, ctx.dimensions):
                lmax, mmax = ctx.dimensions[:2]
                sector_hessian = partial(mixed_sector_gradients, edge_index=edge_index, lmax=lmax, mmax=mmax)
                mixed_gradients = partial(
                    mixed_gather_gradients, sector_hessian=sector_hessian,
                    edge_index=edge_index, metadata=metadata, lmax=lmax, mmax=mmax,
                )
            gradients = GatherDoubleBackward.apply(
                *values, *directions, hvp, vjp, not atomic_vjp, mixed_gradients, sector_hessian, *grad_outputs,
            )
        else:
            gradients = CSRPrefixBackward.hvp(
                grad_outputs, values, directions, edge_index=edge_index,
                metadata=metadata, dimensions=ctx.dimensions,
            )
        *grad_sectors, dx, dradial, dwigner = gradients
        return (
            dx, None, dradial, dwigner,
            None, None, None,
            None, None, None, None,
            *grad_sectors,
        )

    @staticmethod
    def hvp(
        grad_outputs, values, directions, *, edge_index, metadata, dimensions,
        output_mask=(True, True, True, True),
    ):
        x, radial, wigner = values
        lmax, mmax, _, atomic_vjp = dimensions
        if (
            atomic_vjp
            and compact_double_backward is not None
            and not torch.are_deterministic_algorithms_enabled()
            and torch.cuda.get_device_capability(x.device) in ((8, 0), (9, 0))
        ):
            return compact_double_backward(
                list(grad_outputs), x, edge_index, radial, wigner, *directions,
            )
        return gather_double_backward_source_csr(
            grad_outputs, x, edge_index, radial, wigner, metadata, *directions,
            lmax=lmax, mmax=mmax, output_mask=output_mask,
        )


def flat_prefix_forward(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    block_channels: int,
    num_warps: int,
) -> torch.Tensor:
    spec, _ = validate_inputs(
        x, edge_index, radial, wigner,
        lmax=lmax, mmax=mmax,
        block_channels=block_channels, num_warps=num_warps,
    )
    output = torch.empty(
        (edge_index.shape[1], spec.reduced_components, 2 * spec.channels),
        device=x.device,
        dtype=x.dtype,
    )
    flat_prefix_forward_out(
        x, edge_index, radial, wigner,
        output,
        spec.lmax, spec.mmax, spec.channels,
    )
    return output


class CSRPrefix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, radial, wigner,
        source_order,
        source_row_ptr,
        receiver_row_ptr,
        lmax,
        mmax,
        sectorized,
        derivative_schedule,
    ):
        if sectorized and use_l2_edge_schedule(
            x, edge_index, radial, wigner, lmax, mmax, x.shape[2], forward=True,
        ):
            sectors = compact_prefix(
                x, edge_index, radial, wigner, lmax=lmax, mmax=mmax, channels=x.shape[2]
            )
        elif sectorized:
            sectors = prefix_sector_forward(
                x, edge_index, radial, wigner,
                lmax=lmax, mmax=mmax,
            )
        else:
            block_channels, num_warps = channel_launch(x.shape[2])
            output = flat_prefix_forward(
                x, edge_index, radial, wigner,
                lmax=lmax, mmax=mmax,
                block_channels=block_channels, num_warps=num_warps,
            )
            spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=x.shape[2])
            sectors = []
            for order, width in enumerate(spec.sector_widths):
                start = spec.sector_row_offsets[order]
                rows = width if order == 0 else 2 * width
                value = output.narrow(1, start, rows)
                if order == 0:
                    value = value.reshape(output.shape[0], width * 2 * spec.channels)
                else:
                    value = value.reshape(output.shape[0], 2, width * 2 * spec.channels)
                sectors.append(value)
        ctx.save_for_backward(
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
        )
        ctx.dimensions = (lmax, mmax, derivative_schedule)
        return tuple(sectors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            x, edge_index, radial, wigner,
            source_order,
            source_row_ptr,
            receiver_row_ptr,
        ) = ctx.saved_tensors
        lmax, mmax, derivative_schedule = ctx.dimensions
        fused_vjp = derivative_schedule != "split_csr"
        atomic_input = derivative_schedule == "fused_atomic"
        contiguous_grads = tuple(grad.contiguous() for grad in grad_outputs)
        if (
            not torch.is_grad_enabled()
            and not torch.are_deterministic_algorithms_enabled()
        ):
            if wigner.ndim == 2 and atomic_input:
                gradients = gather_backward_atomic(
                    contiguous_grads,
                    x, edge_index, radial, wigner,
                    lmax=lmax, mmax=mmax,
                    channels=x.shape[2],
                )
            else:
                gradients = gather_backward_source_csr(
                    contiguous_grads,
                    x, edge_index, radial, wigner,
                    (source_order, source_row_ptr, receiver_row_ptr),
                    lmax=lmax, mmax=mmax,
                    fused_vjp=fused_vjp,
                    atomic_input=atomic_input,
                )
        else:
            gradients = CSRPrefixBackward.apply(
                x, edge_index, radial, wigner,
                source_order,
                source_row_ptr,
                receiver_row_ptr,
                lmax,
                mmax,
                fused_vjp,
                atomic_input
                and wigner.ndim == 2
                and (lmax, mmax, x.shape[2]) == (2, 2, 128)
                and not torch.are_deterministic_algorithms_enabled(),
                *contiguous_grads,
            )
        return (
            gradients[0],
            None,
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CompactPrefix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, edge_index, radial, wigner,
        lmax, mmax, channels,
    ):
        edges = edge_index.shape[1]
        spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
        outputs = []
        for order in range(mmax + 1):
            width = spec.sector_widths[order]
            shape = (
                (edges, width * 2 * channels)
                if order == 0
                else (edges, 2, width * 2 * channels)
            )
            outputs.append(torch.empty(shape, device=x.device, dtype=x.dtype))
        torch.ops.so2.compact_prefix_forward(
            x, edge_index, radial, wigner,
            tuple(outputs),
            lmax, mmax, channels,
            channel_launch(channels)[1],
        )
        ctx.save_for_backward(x, edge_index, radial, wigner)
        ctx.dimensions = (lmax, mmax, channels)
        return tuple(outputs[: mmax + 1])

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, edge_index, radial, wigner = ctx.saved_tensors
        lmax, mmax, channels = ctx.dimensions
        gradients = gather_backward_atomic(
            grad_outputs,
            x, edge_index, radial, wigner,
            lmax=lmax, mmax=mmax,
            channels=channels,
        )
        return (
            gradients[0],
            None,
            gradients[1],
            gradients[2],
            None,
            None,
            None,
        )


def compact_prefix(x, edge_index, radial, wigner, *, lmax, mmax, channels):
    if not 0 <= mmax <= lmax:
        raise ValueError("compact prefix supports 0 <= mmax <= lmax")
    return CompactPrefix.apply(x, edge_index, radial, wigner, lmax, mmax, channels)


def rotate_gather_order_degree(
    x, edge_index, radial, wigner,
    *,
    lmax,
    mmax,
    metadata=None,
    sectorized=False,
    derivative_schedule=None,
):
    """Gather both endpoints into local order sectors with order-degree weights.

    ``sectorized`` chooses separate sector buffers or views of one flat buffer.
    The default backward uses split CSR reductions for the flat output and
    fused atomic reductions for separate sectors. ``derivative_schedule`` can
    override that choice without changing the output layout.

    The compact L=M=2, C=128 fused atomic schedule also supports building
    the backward graph. On A100 and H100 it can fuse the double backward
    with Gluon. Deterministic mode uses CSR.
    """
    if not 0 <= mmax <= lmax:
        raise ValueError("CSR prefix supports 0 <= mmax <= lmax")
    if not isinstance(sectorized, bool):
        raise ValueError("sectorized must be a boolean")
    if derivative_schedule is None:
        derivative_schedule = "fused_atomic" if sectorized else "split_csr"
    if derivative_schedule not in ("split_csr", "fused_csr", "fused_atomic"):
        raise ValueError("unsupported prefix derivative schedule")
    if metadata is None:
        metadata = prefix_csr_metadata(edge_index, x.shape[0])
    if len(metadata) != 3:
        raise ValueError(
            "prefix metadata must contain source order and two CSR row pointers"
        )
    expected = ((edge_index.shape[1],), (x.shape[0] + 1,), (x.shape[0] + 1,))
    if any(
        tuple(value.shape) != shape
        or value.dtype != torch.int32
        or value.device != edge_index.device
        or not value.is_contiguous()
        for value, shape in zip(metadata, expected)
    ):
        raise ValueError(
            "prefix metadata must be contiguous int32 tensors matching this graph"
        )
    return CSRPrefix.apply(
        x, edge_index, radial, wigner,
        *metadata,
        lmax,
        mmax,
        sectorized,
        derivative_schedule,
    )
