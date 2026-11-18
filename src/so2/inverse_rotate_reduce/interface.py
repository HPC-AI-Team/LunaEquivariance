"""Inverse-rotate edge messages and reduce them into receiver nodes."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, lru_cache
from typing import NamedTuple

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op, wrap_triton
from torch.nn.functional import linear

from ..utils.graph import graph_tables
from ..utils.layout import check_orders, get_angular_layout, get_retained_layout
from ..utils.tensor_contracts import require_cuda_fp32
from .kernels.backward import (
    attention_backward,
    compact_sector_tail_input_backward_kernel,
    compact_tail_backward,
    configure_receiver_backward_sm90,
    initial_atom_weight_vjp,
    native_kernel,
    paired_initial_backward,
    preprojection_adjoint,
    receiver_backward,
    receiver_vjp,
    receiver_vjp_sm90,
    scalar_embedding_backward,
    sum_backward,
)
from .kernels.double_backward import (
    attention_double_backward_launch,
    compact_receiver_double_backward,
    initial_atom_hvp,
    scalar_embedding_double_backward,
    scalar_hvp_kernel,
    sum_double_backward,
)
from .kernels.forward import (
    attention_forward,
    configure_receiver_forward_sm90,
    degree_gated_receiver_forward,
    forward_kernel,
    initial_forward,
    paired_initial_forward,
    preprojection_forward,
    receiver_degree_forward_kernel,
    receiver_edge_tiled_kernel,
    receiver_forward,
    receiver_forward_kernel,
    receiver_sector_tail_forward,
    scalar_add_kernel,
    scalar_embedding_forward,
    sorted_coo_to_csr_kernel,
    sum_forward,
)
from .kernels.triple_backward import attention_triple_backward, sum_triple_backward
from .kernels.utils import (
    attention_dimensions,
    check_inplace_input,
    preprojection_layout,
    prepare_inputs,
    scalar_embedding_validate,
    use_l2_tail_reuse,
    use_receiver_double_backward,
    validate_row_ptr,
)


@lru_cache(maxsize=None)
def compact_widths(lmax):
    return tuple(
        sum((2 * min(degree, order) + 1) * (2 * degree + 1) for degree in range(lmax + 1))
        for order in range(lmax + 1)
    )


def inverse_rotate_reduce(
    raw, rotations, graph, *, lmax, mmax, edge_weight=None, logits=None, eps=1e-8
):
    """Recombine raw products, inverse-rotate and reduce into receiver nodes.

    ``raw`` is the output of ``linear`` with no scalar side output. Its
    zero-order sector may also have shape ``(E,L+1,C)``; this form is
    consumed as supplied. With ``mmax=0``, the single sector describes
    initial local-frame messages without a preceding complex projection.
    ``rotations`` stores compact inverse-map columns in the same ordering
    as compact forward rows. Sum reduction can consume rotations retaining
    more orders than the messages; the stored order is inferred from the
    compact width. For orthogonal rotation blocks the forward
    representation can be reused here. ``edge_weight`` has one value per
    edge and defaults to one. Output is ``(N,(L+1)**2,C)``.

    With no ``logits``, each edge weight is applied once before summation.
    With logits ``(E,H)``, channels are split among H attention heads.
    For each receiver, normalized weights are
    ``q=exp(logits-detached_max)*edge_weight``, ``p=q/(sum(q)+eps)``;
    the final edge factor is ``p*edge_weight``. This preserves the model's
    two uses of the envelope. Inputs use CUDA FP32 and support derivatives
    through third order, including logits and edge weights.
    """
    raw = tuple(raw)
    check_orders(lmax, mmax, raw)
    edges = graph.src.numel()
    scalar = raw[0]
    if scalar.ndim == 3:
        if scalar.shape[1] != lmax + 1 or scalar.shape[2] <= 0:
            raise ValueError("shaped order zero must have shape (E,L+1,C) with positive C")
        messages = raw
    elif scalar.ndim == 2 and scalar.shape[1] > 0 and scalar.shape[1] % (lmax + 1) == 0:
        channels = scalar.shape[1] // (lmax + 1)
        messages = (scalar.reshape(edges, lmax + 1, channels), *raw[1:])
    else:
        raise ValueError("raw order zero must contain (L+1)*C coefficients without scalar extras")
    if rotations.ndim != 2:
        raise ValueError("compact rotations must be a matrix with one row per edge")
    try:
        stored_mmax = compact_widths(lmax).index(rotations.shape[1])
    except ValueError:
        raise ValueError("rotation width does not match a retained order for this lmax") from None
    if stored_mmax < mmax:
        raise ValueError("rotations retain fewer orders than the messages require")
    if logits is not None and stored_mmax != mmax:
        raise ValueError("attention reduction requires matching rotation and message orders")
    _, metadata = graph_tables(graph)
    if edge_weight is None:
        edge_weight = raw[0].new_ones(edges)
    receivers = graph._tables[("so2", "receivers")]
    if logits is None:
        return sum_aggregation(
            messages, edge_weight, rotations, receivers, graph.num_nodes,
            lmax=lmax, mmax=mmax, row_ptr=metadata[2], raw_complex=True,
            rotation_storage="retained_rows", wigner_mmax=stored_mmax,
        )
    if edge_weight.ndim == 1:
        edge_weight = edge_weight.unsqueeze(1)
    attention_csr = ("so2", "attention_csr")
    if attention_csr not in graph._tables:
        # The receiver-owned attention kernels retain the int64 traversal used
        # by the model. Keep the int32 tables for the gather and sum kernels.
        graph._tables[attention_csr] = graph.dst_rowptr.to(torch.int64)
    return attention_aggregation(
        messages, logits, edge_weight, rotations,
        receivers, graph._tables[attention_csr], graph.num_nodes, eps,
        lmax=lmax, mmax=mmax,
    )


def build_receiver_csr(receivers: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return CUDA int64 receiver CSR of shape ``(num_nodes+1,)``.

    ``receivers`` is a contiguous CUDA int64 ``(E,)`` tensor sorted in
    increasing order. The result gives edge intervals for attention
    aggregation, including empty intervals for isolated nodes. Build it once
    per graph. For metadata shared with sum aggregation, use the int32
    ``prepare_graph(...).receiver_row_ptr`` instead.
    """

    from ..utils.graph import validate_receivers

    validate_receivers(receivers, num_nodes, sorted=True)
    if not receivers.is_cuda or not receivers.is_contiguous():
        raise ValueError("receivers must be a contiguous CUDA tensor")
    if receivers.dtype != torch.int64:
        raise ValueError("receivers must be int64")
    row_ptr = torch.empty((num_nodes + 1,), device=receivers.device, dtype=torch.int64)
    sorted_coo_to_csr_kernel[(num_nodes + 1,)](
        receivers, row_ptr, receivers.shape[0], num_warps=1,
    )
    return row_ptr


def attention_aggregation(
    values: torch.Tensor | tuple[torch.Tensor, ...],
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    nodes: int,
    eps: float,
    *,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    """Recombine, attention-weight, inverse-rotate, and reduce edge values.

    For each receiver and attention head, let ``s`` be the detached maximum
    logit. This operation computes
    ``q[e] = exp(logits[e]-s) * envelope[e]``,
    ``p[e] = q[e] / (sum(q[e] for e -> i) + eps)``, and
    ``weight[e] = p[e] * envelope[e]``. The envelope therefore enters
    twice.

    Parameters
    ----------
    values : Tensor or tuple[Tensor, ...]
        Packed physical coefficients have shape ``(E, R, C)``. Retained rows
        run through all zero-order degrees, then all ``+1`` degrees, all
        ``-1`` degrees, and so on. ``R=sum(2*min(l,M)+1 for l in range(L+1))``.
        A tuple supplies raw matrix products: ``(E, L+1, C)`` at order zero
        and ``(E, 2, 2*(L+1-m)*C)`` at positive order. Reshape a raw sector
        to ``(E, 2, 2, L+1-m, C)``. Its positive and negative coefficients
        are ``raw[:,0,0]-raw[:,1,1]`` and ``raw[:,1,0]+raw[:,0,1]``.
        A signed-sector tuple stores these two planes directly as
        ``(E, 2, (L+1-m)*C)`` for each positive order and keeps the same
        order-zero shape.
    logits : Tensor, shape (E, H)
        Attention logits. ``C`` must be divisible by the number of heads
        ``H``. Each head weights a contiguous group of ``C/H`` channels.
    envelope : Tensor, shape (E, 1)
        Edge factors used in both normalization and final weighting.
    wigner_inv : Tensor, shape (E, K)
        Compact inverse-map columns. For each increasing degree, store
        local columns ``0,+1,-1,...`` and all ``2*l+1`` global entries of
        each column. ``K=sum((2*min(l,M)+1)*(2*l+1) for l in range(L+1))``.
        :func:`LunaEquivariance.so2.rotation.pack_inverse_rotations` packs a rectangular
        ``(E, F, R)`` inverse map. This function does not compute a matrix
        inverse or remove scales already present in the map.
    receivers : int64 Tensor, shape (E,)
        Receiver indices, sorted in increasing order.
    row_ptr : int32 or int64 Tensor, shape (nodes+1,)
        Receiver CSR for exactly this edge order.
    nodes : int
        Number of output nodes.
    eps : float
        Constant added to each receiver/head denominator after summation.
    lmax, mmax : int
        ``L >= 0`` and ``0 <= M <= L``.

    Returns
    -------
    Tensor, shape (nodes, (L+1)**2, C)
        Global degree-major node features. Each edge value is weighted,
        mapped by its inverse rotation, and added to its receiver.

    All tensors are contiguous on one CUDA device. Floating inputs are
    FP32. Values, logits, envelope, and inverse-map entries support backward
    and double backward. Raw and signed sectors also support triple backward
    through PyTorch recomputation. The stabilizing maximum is constant in
    each derivative. The fused kernels materialize
    neither recombined nor inverse-rotated edge arrays.
    """
    channels, _, _ = attention_dimensions(values, logits)
    if (
        not isinstance(nodes, (int, torch.SymInt))
        or isinstance(nodes, bool)
        or nodes < 0
    ):
        raise ValueError("nodes must be a nonnegative integer")
    if isinstance(values, torch.Tensor):
        operation = packed_attention_op
    else:
        values = list(values)
        operation = raw_attention_op
    workspace = operation(
        values, logits, envelope, wigner_inv,
        receivers, row_ptr,
        nodes, eps, lmax, mmax,
    )
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    output_elements_per_node = spec.full_components * channels
    return workspace[: nodes * output_elements_per_node].view(
        nodes, spec.full_components, channels
    )


@torch.library.triton_op("lunaeq_so2::structured_scatter", mutates_args={})
def packed_attention_op(
    values: torch.Tensor,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    return attention_forward(
        values, logits, envelope, wigner_inv,
        receivers, row_ptr,
        num_nodes, eps, lmax, mmax,
    )


@packed_attention_op.register_fake
def packed_attention_fake(
    values, logits, envelope, wigner_inv, receivers, row_ptr, num_nodes, eps, lmax, mmax
):
    channels, heads, _ = attention_dimensions(values, logits)
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    workspace_elements_per_node = spec.full_components * channels + (heads * 2)
    return values.new_empty((num_nodes * workspace_elements_per_node,))


@torch.library.triton_op("lunaeq_so2::structured_scatter_backward", mutates_args={})
def packed_attention_backward(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: torch.Tensor,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_backward(
        grad_output, forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
        num_nodes, eps, lmax, mmax,
    )


@packed_attention_backward.register_fake
def packed_attention_backward_fake(
    grad_output, forward_output,
    values, logits, envelope, wigner_inv,
    receivers, row_ptr, stats,
    num_nodes, eps, lmax, mmax,
):
    return (
        torch.empty_like(values),
        torch.empty_like(logits),
        torch.empty_like(envelope),
        torch.empty_like(wigner_inv),
    )


@torch.library.triton_op("lunaeq_so2::structured_scatter_hvp", mutates_args={})
def packed_attention_double_backward(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: torch.Tensor,
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner_inv: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    values_direction: torch.Tensor,
    logits_direction: torch.Tensor,
    envelope_direction: torch.Tensor,
    wigner_direction: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_double_backward_launch(
        grad_output, forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
        values_direction, logits_direction,
        envelope_direction, wigner_direction,
        num_nodes, eps, lmax, mmax,
    )


@packed_attention_double_backward.register_fake
def packed_attention_double_backward_fake(
    grad_output, forward_output,
    values, logits, envelope, wigner_inv,
    receivers, row_ptr, stats,
    values_direction, logits_direction,
    envelope_direction, wigner_direction,
    num_nodes, eps, lmax, mmax,
):
    return (
        torch.empty_like(grad_output),
        torch.empty_like(values),
        torch.empty_like(logits),
        torch.empty_like(envelope),
        torch.empty_like(wigner_inv),
    )


def save_backward_state(ctx, inputs, output) -> None:
    (
        grad_output, forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
        ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
    ) = inputs
    ctx.save_for_backward(
        grad_output, forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
    )


@once_differentiable
def attention_double_backward(
    ctx,
    values_direction: torch.Tensor,
    logits_direction: torch.Tensor,
    envelope_direction: torch.Tensor,
    wigner_direction: torch.Tensor,
):
    (
        grad_output, forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
    ) = ctx.saved_tensors
    references = (values, logits, envelope, wigner_inv)
    directions = tuple(
        torch.zeros_like(reference) if direction is None else direction.contiguous()
        for direction, reference in zip(
            (
                values_direction, logits_direction,
                envelope_direction, wigner_direction,
            ),
            references,
        )
    )
    double_gradients = packed_attention_double_backward(
        grad_output.contiguous(), forward_output,
        values, logits, envelope, wigner_inv,
        receivers, row_ptr, stats,
        *directions,
        ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
    )
    return (
        double_gradients[0], None, *double_gradients[1:],
        None, None, None,
        None, None, None, None,
    )


packed_attention_backward.register_autograd(
    attention_double_backward,
    setup_context=save_backward_state,
)


def save_attention_state(ctx, inputs, output) -> None:
    (
        values, logits, envelope, wigner_inv,
        receivers, row_ptr,
        ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
    ) = inputs
    ctx.save_for_backward(
        values, logits, envelope, wigner_inv, receivers, row_ptr, output
    )


def packed_attention_vjp(ctx, grad_workspace: torch.Tensor | None):
    values, logits, envelope, wigner_inv, receivers, row_ptr, workspace = ctx.saved_tensors
    channels, heads, _ = attention_dimensions(values, logits)
    spec = get_retained_layout(lmax=ctx.lmax, mmax=ctx.mmax, channels=channels)
    output_element_count = ctx.num_nodes * spec.full_components * channels
    forward_output = workspace[:output_element_count].view(
        ctx.num_nodes, spec.full_components, channels
    )
    stats = workspace[output_element_count:].view(ctx.num_nodes, heads, 2)
    if grad_workspace is None:
        grad_values = torch.zeros_like(values)
        grad_logits = torch.zeros_like(logits)
        grad_envelope = torch.zeros_like(envelope)
        grad_wigner = torch.zeros_like(wigner_inv)
    else:
        grad_output = grad_workspace[:output_element_count].view(
            ctx.num_nodes, spec.full_components, channels
        )
        grad_values, grad_logits, grad_envelope, grad_wigner = (
            packed_attention_backward(
                grad_output, forward_output.detach(),
                values, logits, envelope, wigner_inv,
                receivers, row_ptr,
                stats.detach(),
                stats.shape[0], ctx.eps, ctx.lmax, ctx.mmax,
            )
        )
    return (
        grad_values, grad_logits, grad_envelope, grad_wigner,
        None, None,
        None, None, None, None,
    )


packed_attention_op.register_autograd(
    packed_attention_vjp,
    setup_context=save_attention_state,
)


@torch.library.triton_op("lunaeq_so2::raw_attention_aggregation", mutates_args={})
def raw_attention_op(
    values: list[torch.Tensor],
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    return attention_forward(
        tuple(values), logits, envelope, wigner,
        receivers, row_ptr,
        num_nodes, eps, lmax, mmax,
    )


@raw_attention_op.register_fake
def raw_attention_fake(
    values, logits, envelope, wigner, receivers, row_ptr, num_nodes, eps, lmax, mmax
):
    channels = values[0].shape[-1]
    heads = logits.shape[1]
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    return values[0].new_empty(
        (num_nodes * (spec.full_components * channels + (heads * 2)),)
    )


@torch.library.triton_op("lunaeq_so2::raw_attention_backward", mutates_args={})
def raw_attention_backward_op(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: list[torch.Tensor],
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> list[torch.Tensor]:
    grad_values, grad_logits, grad_envelope, grad_wigner = attention_backward(
        grad_output, forward_output,
        tuple(values), logits, envelope, wigner,
        receivers, row_ptr, stats,
        num_nodes, eps, lmax, mmax,
    )
    return [*grad_values, grad_logits, grad_envelope, grad_wigner]


@raw_attention_backward_op.register_fake
def raw_attention_backward_fake(
    grad_output, forward_output,
    values, logits, envelope, wigner,
    receivers, row_ptr, stats,
    num_nodes, eps, lmax, mmax,
):
    return [torch.empty_like(value) for value in (*values, logits, envelope, wigner)]


@torch.library.triton_op("lunaeq_so2::raw_attention_hvp", mutates_args={})
def raw_attention_double_backward_op(
    grad_output: torch.Tensor,
    forward_output: torch.Tensor,
    values: list[torch.Tensor],
    logits: torch.Tensor,
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    receivers: torch.Tensor,
    row_ptr: torch.Tensor,
    stats: torch.Tensor,
    directions: list[torch.Tensor],
    num_nodes: int,
    eps: float,
    lmax: int,
    mmax: int,
) -> list[torch.Tensor]:
    count = len(values)
    result = attention_double_backward_launch(
        grad_output, forward_output,
        tuple(values), logits, envelope, wigner,
        receivers, row_ptr, stats,
        tuple(directions[:count]),
        *directions[count:],
        num_nodes, eps, lmax, mmax,
    )
    return [result[0], *result[1], *result[2:]]


@raw_attention_double_backward_op.register_fake
def raw_attention_double_backward_fake(
    grad_output, forward_output,
    values, logits, envelope, wigner,
    receivers, row_ptr, stats,
    directions,
    num_nodes, eps, lmax, mmax,
):
    return [
        torch.empty_like(value)
        for value in (grad_output, *values, logits, envelope, wigner)
    ]


def save_raw_backward_state(ctx, inputs, output):
    (
        grad_output, forward_output,
        values, logits, envelope, wigner,
        receivers, row_ptr, stats,
        ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
    ) = inputs
    ctx.value_count = len(values)
    ctx.save_for_backward(
        grad_output, forward_output,
        *values,
        logits,
        envelope,
        wigner,
        receivers,
        row_ptr,
        stats,
    )


def raw_double_backward_flat(ctx, *directions):
    grad_output, forward_output, *saved = ctx.saved_tensors
    count = ctx.value_count
    values = saved[:count]
    logits, envelope, wigner, receivers, row_ptr, stats = saved[count:]
    references = (*values, logits, envelope, wigner)
    directions = [
        torch.zeros_like(reference) if direction is None else direction.contiguous()
        for direction, reference in zip(directions, references, strict=True)
    ]
    if torch.is_grad_enabled():
        return RawAttentionDoubleBackward.apply(
            grad_output.contiguous(), forward_output, receivers, row_ptr, stats,
            (ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax), *references, *directions,
        )
    return tuple(
        raw_attention_double_backward_op(
            grad_output.contiguous(), forward_output,
            values, logits, envelope, wigner,
            receivers, row_ptr, stats,
            directions,
            ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
        )
    )


class RawAttentionDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, forward_output, receivers, row_ptr, stats, dimensions,
                *values_and_directions):
        nodes, eps, lmax, mmax = dimensions
        count = mmax + 4
        values = values_and_directions[:count]
        directions = values_and_directions[count:]
        *messages, logits, envelope, wigner = values
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_output, receivers, *values_and_directions)
        ctx.dimensions = dimensions
        return tuple(raw_attention_double_backward_op(
            grad_output, forward_output, messages, logits, envelope, wigner,
            receivers, row_ptr, stats, list(directions), nodes, eps, lmax, mmax,
        ))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        grad_output, receivers, *saved = ctx.saved_tensors
        needs = (ctx.needs_input_grad[0], *ctx.needs_input_grad[6:])
        grad_seed, *grad_values = attention_triple_backward(
            (grad_output, *saved), gradients, needs, receivers, ctx.dimensions,
        )
        return grad_seed, None, None, None, None, None, *grad_values


def raw_double_backward(ctx, directions):
    result = raw_double_backward_flat(ctx, *directions)
    count = ctx.value_count
    return (
        result[0],
        None,
        list(result[1 : count + 1]),
        *result[count + 1 :],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


raw_attention_backward_op.register_autograd(raw_double_backward, setup_context=save_raw_backward_state)


def save_raw_attention_state(ctx, inputs, output):
    (
        values, logits, envelope, wigner,
        receivers, row_ptr,
        ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
    ) = inputs
    ctx.value_count = len(values)
    ctx.save_for_backward(*values, logits, envelope, wigner, receivers, row_ptr, output)


def raw_attention_backward(ctx, grad_workspace):
    count = ctx.value_count
    saved = ctx.saved_tensors
    values = list(saved[:count])
    logits, envelope, wigner, receivers, row_ptr, workspace = saved[count:]
    channels = values[0].shape[-1]
    heads = logits.shape[1]
    spec = get_retained_layout(lmax=ctx.lmax, mmax=ctx.mmax, channels=channels)
    elements = ctx.num_nodes * spec.full_components * channels
    shape = (ctx.num_nodes, spec.full_components, channels)
    if grad_workspace is None:
        grads = [
            torch.zeros_like(value) for value in (*values, logits, envelope, wigner)
        ]
    else:
        grads = raw_attention_backward_op(
            grad_workspace[:elements].view(shape),
            workspace[:elements].view(shape).detach(),
            values, logits, envelope, wigner,
            receivers, row_ptr,
            workspace[elements:].view(ctx.num_nodes, heads, 2).detach(),
            ctx.num_nodes, ctx.eps, ctx.lmax, ctx.mmax,
        )
    return (grads[:count], *grads[count:], None, None, None, None, None, None)


raw_attention_op.register_autograd(raw_attention_backward, setup_context=save_raw_attention_state)


class EdgeDegreeBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        radial, envelope, wigner,
        receivers, row_ptr,
        rescale,
        lmax, mmax,
    ):
        spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=radial.shape[2])
        gradients = scalar_embedding_backward(
            grad_output, radial, envelope, wigner, receivers, rescale, spec
        )
        ctx.save_for_backward(grad_output, radial, envelope, wigner, receivers, row_ptr)
        ctx.rescale = rescale
        ctx.lmax = lmax
        ctx.mmax = mmax
        return gradients

    @staticmethod
    @once_differentiable
    def backward(ctx, radial_direction, envelope_direction, wigner_direction):
        grad_output, radial, envelope, wigner, receivers, row_ptr = ctx.saved_tensors
        if radial_direction is None:
            radial_direction = torch.zeros_like(radial)
        else:
            radial_direction = radial_direction.contiguous()
        if envelope_direction is None:
            envelope_direction = torch.zeros_like(envelope)
        else:
            envelope_direction = envelope_direction.contiguous()
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        else:
            wigner_direction = wigner_direction.contiguous()
        gradients = scalar_embedding_double_backward(
            grad_output,
            radial, envelope, wigner,
            receivers, row_ptr,
            radial_direction, envelope_direction, wigner_direction,
            ctx.rescale,
            get_retained_layout(
                lmax=ctx.lmax, mmax=ctx.mmax, channels=radial.shape[2],
            ),
        )
        return (*gradients, None, None, None, None, None)


class EdgeDegree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, radial, envelope, wigner, receivers, row_ptr, rescale, lmax, mmax):
        spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=radial.shape[2])
        scalar_embedding_validate(radial, envelope, wigner, receivers, row_ptr, spec)
        output = scalar_embedding_forward(radial, envelope, wigner, row_ptr, rescale, spec)
        ctx.save_for_backward(radial, envelope, wigner, receivers, row_ptr)
        ctx.rescale = rescale
        ctx.lmax = lmax
        ctx.mmax = mmax
        return output

    @staticmethod
    def backward(ctx, grad_output):
        radial, envelope, wigner, receivers, row_ptr = ctx.saved_tensors
        if not torch.is_grad_enabled():
            gradients = scalar_embedding_backward(
                grad_output.contiguous(),
                radial,
                envelope,
                wigner,
                receivers,
                ctx.rescale,
                get_retained_layout(
                    lmax=ctx.lmax, mmax=ctx.mmax, channels=radial.shape[2],
                ),
            )
            return (*gradients, None, None, None, None, None)
        gradients = EdgeDegreeBackward.apply(
            grad_output.contiguous(),
            radial, envelope, wigner,
            receivers, row_ptr,
            ctx.rescale,
            ctx.lmax,
            ctx.mmax,
        )
        return (*gradients, None, None, None, None, None)


def scalar_edge_to_node(
    radial, envelope, wigner,
    receivers, row_ptr,
    rescale,
    *,
    lmax, mmax,
):
    """Rotate local zero-order embeddings and sum into receiver nodes.

    For degree ``l``, compute
    ``out[i,l] = sum(envelope[e] * Dinv[e,l,:,m=0,None] * radial[e,l]
    for e -> i) / rescale``. The result has all ``2*l+1`` global components,
    although each local embedding has only order zero.

    Parameters
    ----------
    radial : Tensor, shape (E, L+1, C)
        A local zero-order embedding for every degree.
    envelope : Tensor, E entries
        Scalar edge factors, applied once.
    wigner : Tensor, shape (E, K)
        Compact inverse-map columns, using the retained rotation layout for
        ``L`` and ``M``. Blocks have increasing degree and local columns
        ``0,+1,-1,...``, each with ``2*l+1`` global components. Only the
        zero-order column of each degree is consumed. Other stored orders
        can serve subsequent convolutions on the same geometry.
    receivers : int32 or int64 Tensor, shape (E,)
        Receiver indices, sorted in increasing order.
    row_ptr : int32 or int64 Tensor, shape (N+1,)
        Matching receiver CSR, including the final edge offset.
    rescale : float
        Constant divisor applied after aggregation.
    lmax, mmax : int
        Maximum degree ``L`` and stored maximum local order ``M``, with
        ``0 <= M <= L``.

    Returns
    -------
    Tensor, shape (N, (L+1)**2, C)
        Global coefficients in degree-major order ``l*l+l+m``.

    Floating inputs must be contiguous CUDA FP32 on one device. Integer
    indices must be contiguous on that device. Radial weights, envelope,
    and rotation entries support backward and double backward.
    """

    return EdgeDegree.apply(
        radial, envelope, wigner,
        receivers, row_ptr,
        float(rescale),
        int(lmax),
        int(mmax),
    )


class AtomAddBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient, atomic_numbers, rows):
        ctx.save_for_backward(atomic_numbers)
        ctx.rows = rows
        return gradient, initial_atom_weight_vjp(gradient, atomic_numbers, rows)

    @staticmethod
    def backward(ctx, edge_direction, weight_direction):
        (atomic_numbers,) = ctx.saved_tensors
        return (
            initial_atom_hvp(edge_direction, weight_direction, atomic_numbers),
            None,
            None,
        )


class AtomAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_features, weight, atomic_numbers):
        ctx.mark_dirty(node_features)
        ctx.save_for_backward(atomic_numbers)
        ctx.rows = weight.shape[0]
        nodes, components, channels = node_features.shape
        forward_kernel[(nodes,)](
            node_features,
            weight,
            atomic_numbers,
            nodes,
            components * channels,
            channels=channels,
            block=triton.next_power_of_2(channels),
            num_warps=4,
        )
        return node_features

    @staticmethod
    def backward(ctx, gradient):
        (atomic_numbers,) = ctx.saved_tensors
        if not torch.is_grad_enabled():
            return (
                gradient if ctx.needs_input_grad[0] else None,
                (
                    initial_atom_weight_vjp(gradient, atomic_numbers, ctx.rows)
                    if ctx.needs_input_grad[1]
                    else None
                ),
                None,
            )
        return (*AtomAddBackward.apply(gradient, atomic_numbers, ctx.rows), None)


def fused_initial_atom(node_features, weight, atomic_numbers):
    """Add atom-table embeddings to coefficient zero in place.

    Parameters
    ----------
    node_features : Tensor, shape (N, R, C)
        Contiguous CUDA FP32 coefficients with row zero scalar. Storage must
        permit an autograd in-place operation. Clone a leaf that requires
        gradients before passing it to this function.
    weight : Tensor, shape (T, C)
        Contiguous CUDA FP32 atom embedding table on the input device.
    atomic_numbers : Tensor, shape (N,)
        Contiguous CUDA int64 table indices in [0, T). The caller provides
        the convention that maps chemical element numbers to table rows.

    Returns
    -------
    Tensor, shape (N, R, C)
        The same tensor and storage as node_features, with
        ``node_features[:, 0] += weight[atomic_numbers]``. Other rows are unchanged.

    The kernel combines embedding lookup with the scalar write and
    avoids a separate gathered tensor. Backward and double backward cover
    node features, weights and incoming cotangents. Indices are not differentiable.
    """
    if node_features.ndim != 3 or min(node_features.shape[1:]) < 1:
        raise ValueError(
            "initial atom fusion requires node features of shape (N, K, C)"
        )
    nodes, _, channels = node_features.shape
    if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] != channels:
        raise ValueError("initial atom weight must have shape (T, C) with T > 0")
    check_inplace_input(node_features)
    require_cuda_fp32(
        node_features, weight, operation="initial atom fusion", contiguous=True
    )
    if (
        atomic_numbers.shape != (nodes,)
        or atomic_numbers.dtype != torch.int64
        or atomic_numbers.device != node_features.device
        or not atomic_numbers.is_contiguous()
    ):
        raise ValueError(
            "initial atom indices must be contiguous int64 (N,) on the input device"
        )
    return AtomAdd.apply(node_features, weight, atomic_numbers)


class ScalarAddBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient):
        return gradient, gradient[:, 0, :]

    @staticmethod
    def backward(ctx, node_direction, scalar_direction):
        node_direction = node_direction.contiguous()
        scalar_direction = scalar_direction.contiguous()
        output = torch.empty_like(node_direction)
        elements = node_direction.numel()
        row_stride = node_direction.shape[1] * node_direction.shape[2]
        block = 256
        scalar_hvp_kernel[(triton.cdiv(elements, block),)](
            node_direction,
            scalar_direction,
            output,
            elements,
            row_stride,
            channels=node_direction.shape[2],
            block=block,
            num_warps=4,
        )
        return output


class ScalarAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_features, scalar_features):
        ctx.mark_dirty(node_features)
        nodes, components, channels = node_features.shape
        scalar_add_kernel[(nodes,)](
            node_features,
            scalar_features,
            nodes,
            components * channels,
            channels=channels,
            block=triton.next_power_of_2(channels),
            num_warps=4,
        )
        return node_features

    @staticmethod
    def backward(ctx, gradient):
        if not torch.is_grad_enabled():
            return gradient, gradient[:, 0, :]
        return ScalarAddBackward.apply(gradient)


def fused_initial_scalar(node_features, scalar_features):
    """Add externally computed scalar features to coefficient zero in place.

    Parameters
    ----------
    node_features : Tensor, shape (N, R, C)
        Contiguous CUDA FP32 coefficients, row zero scalar. Clone a leaf that
        requires gradients before passing it to this in-place operation.
    scalar_features : Tensor, shape (N, C)
        Contiguous CUDA FP32 additions on the same device as node_features.

    Returns
    -------
    Tensor, shape (N, R, C)
        The same tensor and storage, with
        ``node_features[:, 0] += scalar_features``. Other rows are unchanged.

    The kernel writes scalar additions without rebuilding the full
    coefficient tensor. Backward and double backward cover both inputs and
    incoming cotangents.
    """
    if node_features.ndim != 3 or min(node_features.shape[1:]) < 1:
        raise ValueError(
            "initial scalar fusion requires node features of shape (N, K, C)"
        )
    if scalar_features.shape != (node_features.shape[0], node_features.shape[2]):
        raise ValueError("initial scalar fusion shape mismatch")
    check_inplace_input(node_features)
    require_cuda_fp32(
        node_features,
        scalar_features,
        operation="initial scalar fusion",
        contiguous=True,
    )
    return ScalarAdd.apply(node_features, scalar_features)


def receiver_row_ptr(receivers: torch.Tensor, nodes: int) -> torch.Tensor:
    """Return int32 CSR boundaries ``(nodes+1,)`` for sorted integer receivers.

    ``receivers`` has shape ``(E,)`` and uses int32 or int64 on CPU or CUDA.
    The result stays on the same device and can be reused by sum and attention
    aggregation. Each boundary is the insertion point of a node index in
    ``receivers``.
    """
    from ..utils.graph import validate_receivers

    validate_receivers(receivers, nodes, sorted=True)
    boundaries = torch.arange(nodes + 1, device=receivers.device, dtype=receivers.dtype)
    return torch.searchsorted(receivers, boundaries).to(torch.int32)


class PreprojectionAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation, receivers, row_ptr, layout, include_scale, *inputs):
        orders = 3 if layout is None else len(layout.orders)
        sectors = inputs[:orders]
        grad_terms, grad_scale = inputs[orders:-1], inputs[-1]
        ctx.save_for_backward(rotation, receivers, row_ptr, *inputs)
        ctx.layout = layout
        ctx.orders = orders
        ctx.include_scale = include_scale
        *dsectors, drotation = preprojection_adjoint(
            sectors, rotation, receivers, grad_terms, grad_scale,
            include_scale=include_scale, layout=layout,
        )
        return drotation, *dsectors

    @staticmethod
    def backward(ctx, hrotation, *hsectors):
        if ctx.layout is None:
            return PreprojectionAdjoint.terminal_backward(ctx, hrotation, *hsectors)
        return PreprojectionAdjoint.run_backward(ctx, hrotation, *hsectors)

    @staticmethod
    @once_differentiable
    def terminal_backward(ctx, hrotation, *hsectors):
        return PreprojectionAdjoint.run_backward(ctx, hrotation, *hsectors)

    @staticmethod
    def run_backward(ctx, hrotation, *hsectors):
        saved = ctx.saved_tensors
        rotation, receivers, row_ptr = saved[:3]
        sectors = saved[3:3 + ctx.orders]
        grad_terms, grad_scale = saved[3 + ctx.orders:-1], saved[-1]
        hrotation = hrotation.resolve_neg().contiguous()
        hsectors = tuple(h.resolve_neg().contiguous() for h in hsectors)
        layout = ctx.layout
        if torch.is_grad_enabled():
            drotation, *dsectors = PreprojectionAdjoint.apply(
                hrotation, receivers, row_ptr, layout, False,
                *hsectors, *grad_terms, grad_scale,
            )
            # Each reduced term is bilinear. The bias scale is linear in rotation.
            *rotation_terms, dscale = PreprojectionReduce.apply(
                hrotation, receivers, row_ptr, layout, *sectors,
            )
            *sector_terms, _ = PreprojectionReduce.apply(
                rotation, receivers, row_ptr, layout, *hsectors,
            )
            dterms = tuple(a + b for a, b in zip(rotation_terms, sector_terms, strict=True))
        else:
            *dsectors, drotation = preprojection_adjoint(
                hsectors, hrotation, receivers, grad_terms, grad_scale,
                include_scale=False, layout=layout,
            )
            dterms, dscale = preprojection_forward(
                sectors, rotation, row_ptr, directions=(hsectors, hrotation), layout=layout,
            )
        return (
            drotation, None, None, None, None, *dsectors, *dterms,
            dscale if ctx.include_scale else None,
        )


class PreprojectionReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation, receivers, row_ptr, layout, *sectors):
        ctx.save_for_backward(rotation, receivers, row_ptr, *sectors)
        ctx.layout = layout
        terms, scale = preprojection_forward(sectors, rotation, row_ptr, layout=layout)
        return *terms, scale

    @staticmethod
    def backward(ctx, *grad_outputs):
        rotation, receivers, row_ptr, *sectors = ctx.saved_tensors
        grads = tuple(grad.resolve_neg().contiguous() for grad in grad_outputs)
        layout = ctx.layout
        if layout is None or torch.is_grad_enabled():
            drotation, *dsectors = PreprojectionAdjoint.apply(
                rotation, receivers, row_ptr, layout, True, *sectors, *grads,
            )
        else:
            *dsectors, drotation = preprojection_adjoint(
                sectors, rotation, receivers, grads[:-1], grads[-1],
                include_scale=True, layout=layout,
            )
        return drotation, None, None, None, *dsectors


def preprojection_reduce(
    sectors, rotation, receivers, row_ptr, *, frozen: FrozenReceiver | None = None,
    lmax=2, mmax=2,
):
    """Prepare receiver features for a projection shared by all incoming edges.

    For L=M=2 and 128 channels, supply physical sectors of shape (E,3,128),
    (E,4,128), and (E,2,128) or their flattened/GEMM views. For nonzero
    orders, the positive plane precedes the negative plane. Rotations have
    shape (E,35) in retained-row layout and may include an edge envelope.
    Supply contiguous integer receivers sorted in ascending order and the
    corresponding CSR offsets. Floating inputs use CUDA FP32 on one device.

    With inverse coefficients a and b for +m and -m, and features x+ and
    x-, accumulate a*x+ + b*x- for the real map and b*x+ - a*x- for the
    imaginary map. Each degree packs order zero, then the two terms of
    each retained nonzero order, returning tensors of shape (N,1,384),
    (N,3,896), and (N,5,1152). The separate (N,9) scale sums the order-zero
    coefficients for a shared bias. Both support double backward. The
    explicit (L=3, M=2) layout also supports differentiation of that pass.

    With a frozen receiver plan, supply the packed outputs of the first
    projection of shapes (E,640), (E,512), and (E,256). The selected kernel
    applies the gate during reduction and supports one input derivative.
    """
    sectors = tuple(sectors)
    if frozen is not None:
        if (lmax, mmax) != (2, 2):
            raise ValueError("the frozen receiver plan requires L=M=2")
        *terms, scale = frozen.reduction(*sectors, rotation, receivers, row_ptr)
        return tuple(terms), scale

    require_cuda_fp32(*sectors, rotation, operation="preprojection reduction")
    layout = preprojection_layout(lmax, mmax)
    default_schedule = (lmax, mmax) == (2, 2)
    if (
        rotation.ndim != 2
        or rotation.shape[1] != layout.rotation_width
        or not rotation.is_contiguous()
    ):
        raise ValueError(
            f"preprojection reduction requires contiguous rotations of shape (E,{layout.rotation_width})"
        )
    edges = rotation.shape[0]
    if len(sectors) != len(layout.orders):
        raise ValueError("preprojection reduction requires one sector per retained order")
    if (
        receivers.shape != (edges,)
        or receivers.device != rotation.device
        or receivers.dtype not in (torch.int32, torch.int64)
        or not receivers.is_contiguous()
    ):
        raise ValueError("preprojection reduction requires contiguous integer receivers of shape (E,)")
    nodes = row_ptr.numel() - 1
    validate_row_ptr(row_ptr, receivers, nodes)
    if nodes < 0 or (edges and nodes == 0):
        raise ValueError("preprojection reduction requires receiver CSR with a terminal offset")
    max_sector = max((2 if order else 1) * spec.input_width
                     for order, spec in enumerate(layout.orders))
    max_degree = layout.angular_widths[-1] * layout.packed_widths[-1]
    if edges * max_sector > 2**31 or nodes * max_degree > 2**31:
        raise ValueError("preprojection reduction exceeds the 32-bit offset range")

    contiguous_sectors = []
    for order, (sector, spec) in enumerate(zip(sectors, layout.orders, strict=True)):
        planes = 2 if order else 1
        width = planes * spec.input_width
        physical = (edges, width // 128, 128)
        shapes = ((edges, width), physical)
        if order:
            shapes += ((edges, planes, spec.input_width),)
        if sector.shape not in shapes:
            raise ValueError(f"unexpected physical sector shape at order {order}: {tuple(sector.shape)}")
        contiguous_sectors.append(sector.resolve_neg().contiguous())

    *terms, scale = PreprojectionReduce.apply(
        rotation.resolve_neg(), receivers, row_ptr, None if default_schedule else layout,
        *contiguous_sectors,
    )
    return tuple(terms), scale


class SumAggregationBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        envelope, wigner,
        receivers, row_ptr,
        nodes, lmax, mmax,
        raw_complex, wigner_layout,
        *messages,
    ):
        grad_output = grad_output.contiguous()
        if (
            not ctx.needs_input_grad[1]
            and (lmax, mmax, messages[0].shape[-1], wigner_layout) == (2, 2, 128, 6)
        ):
            gradients = (None, *compact_tail_backward(
                grad_output, messages, envelope, wigner, receivers,
                lmax=lmax, mmax=mmax,
                raw_complex=raw_complex, wigner_layout=wigner_layout,
            ))
        else:
            gradients = sum_backward(
                grad_output, messages, envelope, wigner, receivers, nodes,
                lmax=lmax, mmax=mmax,
                wigner_layout=wigner_layout, raw_complex=raw_complex,
            )
        ctx.save_for_backward(
            grad_output, envelope, wigner, receivers, row_ptr, *messages
        )
        ctx.dimensions = (nodes, lmax, mmax, raw_complex, wigner_layout)
        return gradients

    @staticmethod
    def backward(ctx, envelope_direction, wigner_direction, *message_directions):
        (
            grad_output,
            envelope, wigner,
            receivers, row_ptr,
            *messages,
        ) = ctx.saved_tensors
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        message_directions = tuple(
            torch.zeros_like(message) if direction is None else direction.contiguous()
            for message, direction in zip(messages, message_directions, strict=True)
        )
        directions = envelope_direction, wigner_direction, *message_directions
        frozen_envelope = not ctx.needs_input_grad[1]
        if torch.is_grad_enabled():
            gradients = SumAggregationDoubleBackward.apply(
                grad_output, envelope, wigner, receivers, row_ptr, ctx.dimensions,
                frozen_envelope, *messages, *directions,
            )
        else:
            gradients = SumAggregationBackward.hvp(
                grad_output, (envelope, wigner, *messages), directions,
                receivers, row_ptr, ctx.dimensions, frozen_envelope,
            )
        return (*gradients[:3], None, None, None, None, None, None, None, *gradients[3:])

    @staticmethod
    def hvp(grad_output, values, directions, receivers, row_ptr, dimensions, frozen_envelope):
        envelope, wigner, *messages = values
        envelope_direction, wigner_direction = directions[:2]
        message_directions = directions[2:]
        nodes, lmax, mmax, raw_complex, wigner_layout = dimensions
        if (
            frozen_envelope
            and row_ptr is not None
            and (lmax, mmax, messages[0].shape[-1], wigner_layout) == (2, 2, 128, 6)
            and nodes >= 32
            and envelope.numel() >= 16 * nodes
            and envelope.is_cuda
            and torch.version.hip is None
            and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(envelope.device) in ((8, 0), (9, 0))
        ):
            return compact_receiver_double_backward(
                grad_output, tuple(messages), envelope, wigner, receivers, row_ptr,
                message_directions, wigner_direction.contiguous(), raw_complex=raw_complex,
            )
        if envelope_direction is None:
            envelope_direction = torch.zeros_like(envelope)
        envelope_direction = envelope_direction.contiguous()
        if row_ptr is not None and not use_receiver_double_backward(
            envelope.device,
            nodes=nodes,
            lmax=lmax,
            channels=messages[0].shape[-1],
            wigner_layout=wigner_layout,
        ):
            row_ptr = None
        return sum_double_backward(
            grad_output.contiguous(),
            messages, envelope, wigner,
            receivers, nodes,
            envelope_direction, wigner_direction, message_directions,
            lmax=lmax, mmax=mmax,
            wigner_layout=wigner_layout, raw_complex=raw_complex,
            row_ptr=row_ptr,
        )


class SumAggregationDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, envelope, wigner, receivers, row_ptr, dimensions,
                frozen_envelope, *messages_and_directions):
        count = dimensions[2] + 1
        messages = messages_and_directions[:count]
        directions = messages_and_directions[count:]
        values = envelope, wigner, *messages
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_output, receivers, row_ptr, *values, *directions)
        ctx.dimensions = dimensions
        return SumAggregationBackward.hvp(
            grad_output, values, directions, receivers, row_ptr, dimensions, frozen_envelope,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        grad_output, receivers, row_ptr, *saved = ctx.saved_tensors
        count = ctx.dimensions[2] + 3
        values = saved[:count]
        directions = saved[count:]
        grad_cotangent, grad_values, grad_directions = sum_triple_backward(
            grad_output, values, directions, gradients, receivers, row_ptr, ctx.dimensions,
        )
        return (
            grad_cotangent, *grad_values[:2], None, None, None, None,
            *grad_values[2:], *grad_directions,
        )


class SumAggregation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        envelope, wigner,
        receivers, row_ptr,
        nodes, lmax, mmax,
        raw_complex, wigner_layout,
        *messages,
    ):
        output = receiver_sector_tail_forward(
            messages, envelope, wigner,
            row_ptr, nodes,
            lmax=lmax, mmax=mmax,
            raw_complex=raw_complex, wigner_layout=wigner_layout,
        )
        ctx.save_for_backward(envelope, wigner, receivers, row_ptr, *messages)
        ctx.dimensions = (
            nodes, lmax, mmax,
            raw_complex, wigner_layout,
            envelope.requires_grad,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        envelope, wigner, receivers, row_ptr, *messages = ctx.saved_tensors
        (
            nodes, lmax, mmax,
            raw_complex, wigner_layout,
            envelope_requires_grad,
        ) = ctx.dimensions
        if torch.is_grad_enabled():
            denvelope, dwigner, *dmessages = SumAggregationBackward.apply(
                grad_output.contiguous(),
                envelope, wigner,
                receivers, row_ptr,
                nodes, lmax, mmax,
                raw_complex, wigner_layout,
                *messages,
            )
        elif wigner_layout >= 4 and not envelope_requires_grad:
            dwigner, *dmessages = compact_tail_backward(
                grad_output.contiguous(), messages,
                envelope, wigner,
                receivers,
                lmax=lmax, mmax=mmax,
                raw_complex=raw_complex, wigner_layout=wigner_layout,
            )
            denvelope = None
        else:
            denvelope, dwigner, *dmessages = sum_backward(
                grad_output.contiguous(),
                messages, envelope, wigner,
                receivers, nodes,
                lmax=lmax, mmax=mmax,
                wigner_layout=wigner_layout, raw_complex=raw_complex,
            )
        return (
            denvelope, dwigner,
            None, None,
            None, None, None,
            None, None,
            *dmessages,
        )


def sum_aggregation(
    messages, envelope, wigner,
    receivers, nodes,
    *,
    lmax,
    mmax,
    row_ptr=None,
    raw_complex=False,
    rotation_storage=None,
    wigner_mmax=None,
):
    """Recombine edge products, inverse-rotate, and sum into receiver nodes.

    The materialized equation is
    ``out[i] = sum(envelope[e] * D[e].T @ message[e] for e -> i)``.
    Recombination, when requested, happens before inverse rotation. The
    kernel accumulates directly into node outputs.

    Parameters
    ----------
    messages : tuple[Tensor, ...]
        One sector per order ``0..M``. Order zero has shape ``(E, L+1, C)``.
        With ``raw_complex=False``, order ``m>0`` has shape
        ``(E, 2*(L+1-m), C)``, with positive-order degrees first and
        negative-order degrees second. With ``raw_complex=True``, it has
        shape ``(E, 2, 2*(L+1-m)*C)``. Reshaping this raw sector to
        ``(E, 2, 2, L+1-m, C)`` gives input-plane and weight-map axes.
        Recombined positive and negative coefficients are
        ``raw[:,0,0] - raw[:,1,1]`` and ``raw[:,1,0] + raw[:,0,1]``.
    envelope : Tensor, E entries
        A scalar factor per edge, applied once to the complete message.
    wigner : Tensor
        Forward rotation rows, applied transposed to return local messages
        to the global frame. Dense storage is ``(E, F, F)``, where
        ``F=(L+1)**2`` and rows use ``l*l+l+m``. Only within-degree entries
        are used. Retained-row storage is ``(E, K)`` with increasing degree,
        local rows ``0,+1,-1,...``, and ``2*l+1`` full components per row.
        It is the output of :func:`LunaEquivariance.so2.rotation.interface.pack_retained_rotations`.
        An inverse-specific scale can already be included in these entries.
    receivers : int32 or int64 Tensor, shape (E,)
        Receiver indices, sorted in increasing order.
    nodes : int
        Number of output nodes.
    lmax, mmax : int
        Maximum degree ``L`` and consumed local order ``M``, with
        ``0 <= M <= L``.
    row_ptr : int32 Tensor, shape (nodes+1,), optional
        Receiver CSR for the same sorted edges. If absent, it is built
        from ``receivers``. Reuse it across layers on the same graph.
    raw_complex : bool
        Select raw matrix products or already recombined sectors.
    rotation_storage : {None, "dense", "retained_rows"}
        ``None`` selects dense storage.
    wigner_mmax : int, optional
        Stored maximum order for compact rotations. Defaults to ``mmax``.
        A larger value lets several consumers share the same rotation data.

    Returns
    -------
    Tensor, shape (nodes, F, C)
        Global node features in degree-major order. Nodes without incoming
        edges receive zero.

    Floating inputs are CUDA FP32 on one device. Compact rotation storage
    must be contiguous. Messages, envelope, and rotations support backward
    through third order, including recombination.
    """
    if not messages or messages[0].ndim != 3:
        raise ValueError("sum aggregation requires rank-three message sectors")
    if not 0 <= mmax <= lmax:
        raise ValueError("the receiver-owned tail requires 0 <= mmax <= lmax")
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=messages[0].shape[-1])
    if rotation_storage == "retained_rows":
        wigner_mmax = mmax if wigner_mmax is None else wigner_mmax
        retained = get_retained_layout(
            lmax=lmax, mmax=wigner_mmax, channels=spec.channels
        )
        if (
            wigner_mmax < mmax
            or tuple(wigner.shape) != (messages[0].shape[0], retained.compact_elements)
            or not wigner.is_contiguous()
        ):
            raise ValueError(
                "expected contiguous retained rotations (E, K) covering all consumer orders"
            )
        wigner = wigner.unsqueeze(-1)
        # Kernel IDs: 0 = dense, 1 = transposed dense, 4 + M = retained rows.
        wigner_layout = 4 + wigner_mmax
    elif rotation_storage in (None, "dense"):
        if tuple(wigner.shape) != (
            messages[0].shape[0],
            spec.full_components,
            spec.full_components,
        ):
            raise ValueError("expected dense rotation shape (E, d, d)")
        wigner_layout = 0
    else:
        raise ValueError("unknown rotation storage")
    envelope, wigner, receivers = prepare_inputs(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout, raw_complex=raw_complex,
    )
    if row_ptr is None:
        row_ptr = receiver_row_ptr(receivers, nodes)
    elif (
        row_ptr.dtype != torch.int32
        or tuple(row_ptr.shape) != (nodes + 1,)
        or row_ptr.device != receivers.device
    ):
        raise ValueError(
            "row_ptr must be int32 with nodes + 1 entries on the receiver device"
        )
    row_ptr = row_ptr.contiguous()
    return SumAggregation.apply(
        envelope, wigner,
        receivers, row_ptr,
        nodes, lmax, mmax,
        raw_complex, wigner_layout,
        *messages,
    )


class ScatterSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, envelope, wigner, receivers, nodes, lmax, mmax, wigner_layout, *messages
    ):
        output = sum_forward(
            messages, envelope, wigner,
            receivers, nodes,
            lmax=lmax, mmax=mmax,
            wigner_layout=wigner_layout,
        )
        ctx.save_for_backward(envelope, wigner, receivers, *messages)
        ctx.dimensions = (nodes, lmax, mmax, wigner_layout)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        envelope, wigner, receivers, *messages = ctx.saved_tensors
        nodes, lmax, mmax, wigner_layout = ctx.dimensions
        if torch.is_grad_enabled():
            # A missing CSR selects the scatter schedule for double backward.
            denvelope, dwigner, *dmessages = SumAggregationBackward.apply(
                grad_output.contiguous(),
                envelope, wigner,
                receivers, None,
                nodes, lmax, mmax,
                False, wigner_layout,
                *messages,
            )
        else:
            denvelope, dwigner, *dmessages = sum_backward(
                grad_output.contiguous(),
                messages, envelope, wigner,
                receivers, nodes,
                lmax=lmax, mmax=mmax,
                wigner_layout=wigner_layout,
            )
        return denvelope, dwigner, None, None, None, None, None, *dmessages


def structured_sector_tail(
    messages, envelope, wigner,
    receivers, nodes,
    *,
    lmax,
    mmax,
    wigner_layout="row_full",
    wigner_mmax=None,
):
    """Scatter inverse-rotated sectors into nodes, without building a CSR."""
    if wigner_mmax is None:
        wigner_mmax = mmax
    if wigner_layout == "retained_rows":
        if wigner.ndim != 2 or not wigner.is_contiguous():
            raise ValueError("retained rotations must be a contiguous (E, K) tensor")
        wigner = wigner.unsqueeze(-1)
    wigner_layout_id = {
        "retained_rows": 4 + wigner_mmax,
        "row_full": 0,
        "full_row": 1,
    }[wigner_layout]
    envelope, wigner, receivers = prepare_inputs(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax, mmax=mmax,
        wigner_layout=wigner_layout_id,
    )
    return ScatterSum.apply(
        envelope, wigner, receivers, nodes, lmax, mmax, wigner_layout_id, *messages
    )


class InitialSumBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        envelope,
        wigner,
        receivers,
        atom_weight,
        atomic_numbers,
        nodes, lmax, mmax,
        wigner_layout,
        *messages,
    ):
        gradients = sum_backward(
            grad_output.contiguous(),
            messages, envelope, wigner,
            receivers, nodes,
            lmax=lmax, mmax=mmax,
            wigner_layout=wigner_layout,
        )
        weight_gradient = initial_atom_weight_vjp(
            grad_output, atomic_numbers, atom_weight.shape[0]
        )
        ctx.save_for_backward(
            grad_output,
            envelope,
            wigner,
            receivers,
            atomic_numbers,
            *messages,
        )
        ctx.dimensions = (nodes, lmax, mmax, wigner_layout)
        return (*gradients, weight_gradient)

    @staticmethod
    @once_differentiable
    def backward(ctx, envelope_direction, wigner_direction, *directions):
        (
            grad_output,
            envelope,
            wigner,
            receivers,
            atomic_numbers,
            *messages,
        ) = ctx.saved_tensors
        message_directions = directions[:-1]
        weight_direction = directions[-1]
        if envelope_direction is None:
            envelope_direction = torch.zeros_like(envelope)
        envelope_direction = envelope_direction.contiguous()
        if wigner_direction is None:
            wigner_direction = torch.zeros_like(wigner)
        message_directions = tuple(
            torch.zeros_like(message) if direction is None else direction.contiguous()
            for message, direction in zip(messages, message_directions, strict=True)
        )
        if weight_direction is None:
            rows = int(atomic_numbers.max()) + 1
            weight_direction = torch.zeros(
                (rows, messages[0].shape[-1]),
                device=messages[0].device,
                dtype=messages[0].dtype,
            )
        nodes, lmax, mmax, wigner_layout = ctx.dimensions
        gradients = sum_double_backward(
            grad_output.contiguous(),
            messages, envelope, wigner,
            receivers, nodes,
            envelope_direction, wigner_direction, message_directions,
            lmax=lmax, mmax=mmax,
            wigner_layout=wigner_layout,
        )
        grad_output_gradient = initial_atom_hvp(
            gradients[0], weight_direction, atomic_numbers
        )
        return (
            grad_output_gradient, gradients[1], gradients[2],
            None, None, None,
            None, None, None, None,
            *gradients[3:],
        )


class InitialSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        envelope,
        wigner,
        receivers,
        atom_weight,
        atomic_numbers,
        nodes, lmax, mmax,
        wigner_layout,
        *messages,
    ):
        output = initial_forward(
            messages,
            envelope,
            wigner,
            receivers,
            atom_weight,
            atomic_numbers,
            nodes,
            lmax=lmax, mmax=mmax,
            wigner_layout=wigner_layout,
        )
        ctx.save_for_backward(
            envelope,
            wigner,
            receivers,
            atom_weight,
            atomic_numbers,
            *messages,
        )
        ctx.dimensions = (nodes, lmax, mmax, wigner_layout)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            envelope,
            wigner,
            receivers,
            atom_weight,
            atomic_numbers,
            *messages,
        ) = ctx.saved_tensors
        nodes, lmax, mmax, wigner_layout = ctx.dimensions
        if torch.is_grad_enabled():
            denvelope, dwigner, *dmessages, dweight = InitialSumBackward.apply(
                grad_output.contiguous(),
                envelope,
                wigner,
                receivers,
                atom_weight,
                atomic_numbers,
                nodes, lmax, mmax,
                wigner_layout,
                *messages,
            )
        else:
            denvelope, dwigner, *dmessages = sum_backward(
                grad_output.contiguous(),
                messages, envelope, wigner,
                receivers, nodes,
                lmax=lmax, mmax=mmax,
                wigner_layout=wigner_layout,
            )
            dweight = initial_atom_weight_vjp(
                grad_output, atomic_numbers, atom_weight.shape[0]
            )
        return (
            denvelope, dwigner, None,
            dweight, None,
            None, None, None, None,
            *dmessages,
        )


def structured_initial_sector_tail(
    messages,
    envelope,
    wigner,
    receivers,
    atom_weight,
    atomic_numbers,
    nodes,
    *,
    lmax,
    mmax,
    wigner_layout="row_full",
    wigner_mmax=None,
):
    """Initialize scalar atomic features and add inverse-rotated edge sectors."""
    if wigner_mmax is None:
        wigner_mmax = mmax
    if wigner_layout == "retained_rows":
        if wigner.ndim != 2 or not wigner.is_contiguous():
            raise ValueError("retained rotations must be a contiguous (E, K) tensor")
        wigner = wigner.unsqueeze(-1)
    wigner_layout_id = {
        "retained_rows": 4 + wigner_mmax,
        "row_full": 0,
        "full_row": 1,
    }[wigner_layout]
    envelope, wigner, receivers = prepare_inputs(
        messages, envelope, wigner,
        receivers, nodes,
        lmax=lmax,
        mmax=mmax,
        wigner_layout=wigner_layout_id,
    )
    return InitialSum.apply(
        envelope,
        wigner,
        receivers,
        atom_weight,
        atomic_numbers,
        nodes,
        lmax,
        mmax,
        wigner_layout_id,
        *messages,
    )


@triton_op(
    "so2::compact_sector_tail_input_backward",
    mutates_args=("message_gradients", "wigner_gradient"),
)
def compact_sector_tail_input_backward(
    grad_output: torch.Tensor,
    messages: list[torch.Tensor],
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    receivers: torch.Tensor,
    message_gradients: list[torch.Tensor],
    wigner_gradient: torch.Tensor,
    lmax: int,
    mmax: int,
    channels: int,
    raw_complex: bool,
    wigner_layout: int,
) -> None:
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)

    def grid(meta):
        del meta
        return (envelope.shape[0],)

    wrap_triton(compact_sector_tail_input_backward_kernel)[grid](
        grad_output, tuple(messages),
        envelope, wigner, receivers,
        tuple(message_gradients),
        wigner_gradient,
        tuple(message.stride() for message in messages),
        tuple(gradient.stride() for gradient in message_gradients),
        channels=channels,
        full_components=spec.full_components,
        compact_elements=wigner.shape[1],
        lmax=lmax,
        reduced_degrees=spec.reduced_degrees,
        reduced_orders=tuple(
            abs(row - degree * degree - degree)
            for row, degree in zip(spec.dense_rows, spec.reduced_degrees)
        ),
        dense_rows=spec.dense_rows,
        reduced_components=spec.reduced_components,
        raw_complex=raw_complex,
        wigner_layout=wigner_layout,
        block_channels=max(128, triton.next_power_of_2(channels)),
        reuse_cotangents=use_l2_tail_reuse(
            messages[:3], envelope, wigner, lmax, mmax, channels
        ),
        num_warps=1,
    )


@triton_op(
    "so2::compact_receiver_tail_forward",
    mutates_args=("output",),
)
def compact_receiver_tail_forward(
    messages: list[torch.Tensor],
    envelope: torch.Tensor,
    wigner: torch.Tensor,
    row_ptr: torch.Tensor,
    output: torch.Tensor,
    nodes: int,
    lmax: int,
    mmax: int,
    channels: int,
    raw_complex: bool,
    wigner_layout: int,
) -> None:
    if (mmax, wigner_layout) == (2, 6) and envelope.numel() >= 16 * nodes and (
        (lmax, channels) == (2, 128)
        or (
            2 <= lmax <= 6 and channels == 64
            and torch.version.hip is None
            and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(output.device) in ((8, 0), (9, 0))
        )
    ):
        # On dense graphs, parallelize a receiver's incoming edges within a block.
        def grid(meta):
            return (nodes, lmax + 1, triton.cdiv(channels, 32))

        wrap_triton(receiver_edge_tiled_kernel)[grid](
            tuple(messages), tuple(message.stride() for message in messages),
            envelope, wigner, row_ptr, output, wigner.stride(0),
            lmax, mmax, channels, raw_complex, 8, 32, num_warps=4, num_stages=1,
        )
        return
    if use_l2_tail_reuse(messages[:3], envelope, wigner, lmax, mmax, channels):
        fast_messages = (*messages[:3], messages[0])

        def grid(meta):
            return (nodes, 3)

        wrap_triton(receiver_degree_forward_kernel)[grid](
            *fast_messages, envelope, wigner,
            row_ptr, output, nodes,
            *(stride for message in fast_messages for stride in message.stride()),
            *wigner.stride(), output.stride(0), output.stride(1),
            channels=128,
            full_components=9,
            lmax=2,
            mmax=2,
            raw_complex=raw_complex,
            wigner_layout=wigner_layout,
            block_channels=128,
            receiver_tile=1,
            num_warps=1,
        )
        return
    full_components = (lmax + 1) ** 2

    def grid(meta):
        del meta
        return (nodes, full_components)

    wrap_triton(receiver_forward_kernel)[grid](
        tuple(messages), envelope, wigner,
        row_ptr, output, nodes,
        tuple(message.stride() for message in messages),
        *wigner.stride(), output.stride(0), output.stride(1),
        channels=channels,
        full_components=full_components,
        lmax=lmax, mmax=mmax,
        raw_complex=raw_complex, wigner_layout=wigner_layout,
        block_channels=triton.next_power_of_2(channels),
        receiver_tile=1,
        num_warps=1,
    )


GATED_RECEIVER_LAYOUT = preprojection_layout()


@triton_op("trim_inference::receiver_reduce", mutates_args=())
def gated_receiver(x0: Tensor, x1: Tensor, x2: Tensor, rotation: Tensor, receiver: Tensor,
                   row_ptr: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    layout = GATED_RECEIVER_LAYOUT
    nodes = row_ptr.numel() - 1
    terms = tuple(
        rotation.new_empty((nodes, angular, width))
        for angular, width in zip(layout.angular_widths, layout.packed_widths, strict=True)
    )
    scale = rotation.new_empty((nodes, 9))
    for order, (x, spec) in enumerate(zip((x0, x1, x2), layout.orders, strict=True)):
        wrap_triton(receiver_forward)[(nodes, triton.cdiv(spec.input_width, 32))](
            x, x0, rotation, row_ptr, *terms, scale,
            order, spec.input_width, layout.rotation_width, spec.positive_offsets,
            spec.negative_offsets, spec.degrees, spec.angular_widths, spec.angular_indices,
            spec.packed_widths, spec.packed_offset, 8, 32, num_warps=2, num_stages=1,
        )
    return *terms, scale

@triton_op("trim_inference::receiver_reduce_vjp", mutates_args=())
def gated_receiver_vjp(x0: Tensor, x1: Tensor, x2: Tensor, rotation: Tensor, receiver: Tensor,
                       g0: Tensor, g1: Tensor, g2: Tensor, gscale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cotangents = g0, g1, g2, gscale
    gradients = tuple(torch.empty_like(value) for value in (x0, x1, x2))
    rotation_parts = rotation.new_zeros((3, *rotation.shape))
    *terms, scale = (value.resolve_neg().contiguous() for value in cotangents)
    layout = GATED_RECEIVER_LAYOUT
    edges = rotation.shape[0]
    for degree in range(3):
        wrap_triton(receiver_backward)[(triton.cdiv(edges, 4),)](
            x0, x1, x2, rotation, receiver, *terms, scale, *gradients, rotation_parts,
            edges, tuple(tuple(spec) for spec in layout.orders), layout.packed_widths,
            degree, 4, num_warps=4,
        )
    return *gradients, rotation_parts.sum(dim=0)

def save_gated_receiver(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:5])

def backward_receiver(ctx, *grad):
    return *gated_receiver_vjp(*ctx.saved_tensors, *grad), None, None


gated_receiver.register_autograd(backward_receiver, setup_context=save_gated_receiver)


def pack_projection_weights(scalar_weight, paired_weights, channels):
    """Group per-order projection weights by degree, with optional leading groups."""
    pairs = tuple(weight.chunk(2, dim=-2) for weight in paired_weights)
    weights = []
    for degree in range(scalar_weight.shape[-2] // channels):
        start = degree * channels
        parts = [scalar_weight[..., start:start + channels, :]]
        for order, planes in enumerate(pairs[:degree], start=1):
            start = (degree - order) * channels
            parts.extend(plane[..., start:start + channels, :] for plane in planes)
        if degree == 0 and scalar_weight.ndim > 2:
            weights.append(parts[0].contiguous())
        else:
            weights.append(torch.cat(parts, dim=-1).contiguous())
    return tuple(weights)

def assemble_receiver_rows(values, scale, bias_rows):
    return torch.cat(values, dim=1) + scale.unsqueeze(-1) * bias_rows


class FrozenReceiver(NamedTuple):
    reduction: Callable
    assemble: Callable
    weights: tuple[Tensor, ...]
    bias: Tensor

    @property
    def reference(self):
        return self.reduction is gated_receiver


def prepare_frozen_receiver(scalar_weight, paired_weights, bias, *, reference=False):
    """Prepare the L=M=2 gated receiver for frozen weights and one input VJP."""
    weights = (scalar_weight, *paired_weights)
    require_cuda_fp32(*weights, bias, operation="frozen receiver")
    if any(value.requires_grad for value in (*weights, bias)):
        raise ValueError("frozen receiver requires fixed projection weights and bias")
    if tuple(value.shape for value in weights) != ((384, 384), (512, 256), (256, 128)) or bias.shape != (384,):
        raise ValueError("frozen receiver requires L=M=2 and 128 channels")

    weights = pack_projection_weights(weights[0], weights[1:], 128)
    if not reference:
        capability = torch.cuda.get_device_capability(bias.device)
        if capability == (8, 0):
            from ..linear.interface import assemble_degrees_with_bias

            native_kernel()
            return FrozenReceiver(GatedReceiver.apply, assemble_degrees_with_bias, weights, bias)
        if capability != (9, 0):
            raise ValueError("the native frozen receiver requires an A100 or H100")
        configure_receiver()

    rows = bias.reshape(3, 128)
    bias_rows = torch.cat((rows[:1], rows[1:2].expand(3, -1), rows[2:].expand(5, -1)))
    reduction = gated_receiver if reference else receiver
    return FrozenReceiver(reduction, assemble_receiver_rows, weights, bias_rows)


def project_frozen_receiver(sectors, rotation, receivers, row_ptr, prepared):
    terms, scale = preprojection_reduce(sectors, rotation, receivers, row_ptr, frozen=prepared)
    values = tuple(linear(value, weight) for value, weight in zip(terms, prepared.weights, strict=True))
    return prepared.assemble(values, scale, prepared.bias)


class GatedReceiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, x2, rotation, receiver, row_ptr):
        ctx.save_for_backward(x0, x1, x2, rotation, row_ptr)
        return degree_gated_receiver_forward(x0, x1, x2, rotation, row_ptr, ((8, 32, 2),) * 3)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        return *receiver_vjp(*ctx.saved_tensors, *gradients), None, None


@triton_op("lunaeq_so2::initial_pair_vjp", mutates_args=())
def paired_vjp(gradient: Tensor, radial: Tensor, rotation: Tensor, receiver: Tensor,
               pairs: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    dr, dw = torch.empty_like(radial), torch.empty_like(rotation)
    count = radial.shape[0]
    if count:
        wrap_triton(paired_initial_backward)[(triton.cdiv(count, 2),)](
            gradient.contiguous(), radial, rotation, receiver, pairs, dr, dw, count, scale, 2, num_warps=4,
        )
    return dr, dw

@triton_op("lunaeq_so2::initial_pair", mutates_args=())
def paired(radial: Tensor, rotation: Tensor, receiver: Tensor, edge_rows: Tensor, pairs: Tensor,
           row_ptr: Tensor, scale: float) -> Tensor:
    nodes = row_ptr.numel() - 1
    output = radial.new_empty((nodes, 9, 128))
    if nodes:
        wrap_triton(paired_initial_forward)[(nodes, 4)](
            radial, rotation, edge_rows, row_ptr, output, scale, 16, 32, num_warps=4,
        )
    return output

def save_paired_receiver(ctx, inputs, output):
    radial, rotation, receiver, _, pairs, _, scale = inputs
    ctx.save_for_backward(radial, rotation, receiver, pairs)
    ctx.scale = scale

@torch.autograd.function.once_differentiable
def backward_paired_receiver(ctx, gradient):
    return *paired_vjp(gradient, *ctx.saved_tensors, ctx.scale), None, None, None, None, None


paired.register_autograd(backward_paired_receiver, setup_context=save_paired_receiver)


@cache
def configure_receiver():
    """Load the Hopper CSR receiver before tracing."""
    configure_receiver_backward_sm90()
    configure_receiver_forward_sm90()


@triton_op("trim_h100::cpp_csr_receiver", mutates_args=())
def receiver(
    x0: Tensor, x1: Tensor, x2: Tensor, rotation: Tensor, receivers: Tensor, row_ptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    nodes = row_ptr.numel() - 1
    outputs = (
        rotation.new_empty((nodes, 1, 384)),
        rotation.new_empty((nodes, 3, 896)),
        rotation.new_empty((nodes, 5, 1152)),
        rotation.new_empty((nodes, 9)),
    )
    torch.ops.trim_h100.csr_receiver_out.default(x0, x1, x2, rotation, row_ptr, *outputs)
    return outputs

def save_native_receiver(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4], inputs[5])

def backward_native_receiver(ctx, *grad):
    return *receiver_vjp_sm90(*ctx.saved_tensors, *grad), None, None


receiver.register_autograd(backward_native_receiver, setup_context=save_native_receiver)
