"""Radial MLP computation with caller-owned tensor parameters."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint

from ..utils.tensor_contracts import require_cuda_fp32
from .kernels.backward import (
    distance_atom_pair_backward,
    frozen_radial_norm_backward,
    grouped_layer_norm_silu_backward,
    grouped_layer_norm_silu_backward_input,
    layer_norm_silu_backward,
    layer_norm_silu_backward_input,
)
from .kernels.double_backward import (
    grouped_layer_norm_silu_double_backward,
    layer_norm_silu_double_backward,
)
from .kernels.forward import (
    add_projected_pair_kernel,
    checked_distance_atom_pair_forward,
    checked_radial_types,
    distance_atom_pair_forward,
    indexed_radial_linear_norm_kernel,
    indexed_radial_backend,
    native_indexed_radial,
    radial_linear_norm_kernel,
    tiled_radial_norm_forward,
    grouped_layer_norm_silu_forward,
    layer_norm_silu_forward,
)
from .kernels.triple_backward import grouped_layer_norm_silu_triple_backward


class RadialParameters(NamedTuple):
    """Parameters for two Linear-LayerNorm-SiLU stages and a final Linear.

    Weights use PyTorch's ``(output, input)`` order. For input width Cin,
    hidden widths H1 and H2, and output width Cout:

    - ``first_weight``: (H1, Cin). Its bias and norm vectors have shape (H1,).
    - ``second_weight``: (H2, H1). Its bias and norm vectors have shape (H2,).
    - ``final_weight``: (Cout, H2). ``final_bias`` is (Cout,) or None.

    ``first_eps`` and ``second_eps`` are positive LayerNorm epsilons. Tensors
    use CUDA FP32 on one device, with contiguous bias and norm vectors.
    This record references the caller's tensors, so parameter updates are
    visible on the next call.
    """

    first_weight: torch.Tensor
    first_bias: torch.Tensor
    first_norm_weight: torch.Tensor
    first_norm_bias: torch.Tensor
    first_eps: float
    second_weight: torch.Tensor
    second_bias: torch.Tensor
    second_norm_weight: torch.Tensor
    second_norm_bias: torch.Tensor
    second_eps: float
    final_weight: torch.Tensor
    final_bias: torch.Tensor | None


def radial_hidden_stage(values, bias, norm_weight, norm_bias, epsilon, *, input_only, forward_num_warps):
    if input_only:
        return FrozenRadialNorm.apply(values, bias, norm_weight, norm_bias, epsilon, False, 8, 4)
    return fused_grouped_layer_norm_silu(
        values, bias, norm_weight, norm_bias, epsilon,
        backward_num_warps=4, forward_num_warps=forward_num_warps,
    )


class TransposedRadialLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        need_input, need_weight, _ = ctx.needs_input_grad
        ctx.save_for_backward(inputs if need_weight else None, weight if need_input else None)
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight = ctx.saved_tensors
        need_input, need_weight, need_bias = ctx.needs_input_grad
        # For 8 to 16 rows, cuBLAS picks a much faster FP32 kernel with this layout.
        dx = (weight.mT @ gradient.mT).mT.contiguous() if need_input else None
        dw = gradient.mT @ inputs if need_weight else None
        db = gradient.sum(0) if need_bias else None
        return dx, dw, db


def radial_mlp_forward(
    parameters, inputs, *, forward_num_warps=8, input_only=False, bias_in_linear=False,
    transpose_cotangent=False,
):
    if input_only and any(torch.is_tensor(value) and value.requires_grad for value in parameters):
        raise ValueError("The input-only radial path requires frozen weights")
    first_bias, second_bias = parameters.first_bias, parameters.second_bias
    linear = TransposedRadialLinear.apply if transpose_cotangent else F.linear
    if bias_in_linear:
        hidden = linear(inputs, parameters.first_weight, first_bias).unsqueeze(0)
        first_bias = second_bias = torch.zeros_like(first_bias)
    else:
        hidden = F.linear(inputs, parameters.first_weight).unsqueeze(0)
    del inputs
    hidden = radial_hidden_stage(
        hidden,
        first_bias.unsqueeze(0),
        parameters.first_norm_weight.unsqueeze(0),
        parameters.first_norm_bias.unsqueeze(0),
        parameters.first_eps,
        input_only=input_only,
        forward_num_warps=forward_num_warps,
    )
    if bias_in_linear:
        hidden = linear(hidden.squeeze(0), parameters.second_weight, parameters.second_bias).unsqueeze(0)
    else:
        hidden = F.linear(hidden, parameters.second_weight)
    if not input_only:
        hidden = hidden.contiguous()
    hidden = radial_hidden_stage(
        hidden,
        second_bias.unsqueeze(0),
        parameters.second_norm_weight.unsqueeze(0),
        parameters.second_norm_bias.unsqueeze(0),
        parameters.second_eps,
        input_only=input_only,
        forward_num_warps=forward_num_warps,
    )
    return linear(hidden.squeeze(0), parameters.final_weight, parameters.final_bias)


def fused_radial_mlp(parameters, inputs):
    """Apply Linear-LayerNorm-SiLU twice, followed by a final Linear.

    ``inputs`` has shape (E, Cin). ``parameters`` is a ``RadialParameters``
    record whose tensors share the input's CUDA device and FP32 dtype.
    The output has shape (E, Cout). Each LayerNorm normalizes over the
    hidden channels of one edge and applies its learned scale and bias.

    Each hidden linear bias, LayerNorm and SiLU share one kernel. Dense
    products use PyTorch. Inputs and parameters support derivatives through
    third order.
    """

    return radial_mlp_forward(parameters, inputs)


def unified_radial_hidden(parameters, x_edge, *, forward_num_warps=8):
    """Compute both hidden stages as (networks, edges, hidden channels).

    The first projection shares the edge input. The second is a batched
    product with independent weights. Each network's final projection is
    deferred until its caller consumes the hidden features.
    """
    if isinstance(parameters, FrozenRadialHidden):
        first, second, norm1, norm2, eps1, eps2 = parameters
        hidden = F.linear(x_edge, first)
        hidden = FrozenRadialNorm.apply(hidden, *norm1, eps1, True, 8, 4)
        hidden = torch.bmm(hidden, second)
        return FrozenRadialNorm.apply(hidden, *norm2, eps2, False, 8, 4)

    first_weight = torch.cat(tuple(layer.first_weight for layer in parameters), dim=0)
    first_bias = torch.stack(tuple(layer.first_bias for layer in parameters))
    hidden = F.linear(x_edge, first_weight)
    hidden = hidden.unflatten(1, (len(parameters), -1)).transpose(0, 1).contiguous()
    first_norm_weight = torch.stack(tuple(layer.first_norm_weight for layer in parameters))
    first_norm_bias = torch.stack(tuple(layer.first_norm_bias for layer in parameters))
    hidden = fused_grouped_layer_norm_silu(
        hidden, first_bias, first_norm_weight, first_norm_bias, parameters[0].first_eps,
        backward_num_warps=4,
        forward_num_warps=forward_num_warps,
    )
    second_weight = torch.stack(tuple(layer.second_weight for layer in parameters))
    second_bias = torch.stack(tuple(layer.second_bias for layer in parameters))
    hidden = grouped_radial_linear(hidden, second_weight)
    second_norm_weight = torch.stack(tuple(layer.second_norm_weight for layer in parameters))
    second_norm_bias = torch.stack(tuple(layer.second_norm_bias for layer in parameters))
    return fused_grouped_layer_norm_silu(
        hidden, second_bias, second_norm_weight, second_norm_bias, parameters[0].second_eps,
        backward_num_warps=4,
        forward_num_warps=forward_num_warps,
    )


class ProjectedAtomPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, source, target, source_ids, target_ids):
        ctx.save_for_backward(source_ids, target_ids)
        ctx.shape = source.shape
        layers, edges, channels = hidden.shape
        output = torch.empty_like(hidden)
        with torch.cuda.device(hidden.device):
            add_projected_pair_kernel[(triton.cdiv(edges * channels, 512), layers)](
                hidden, source, target, source_ids, target_ids, output,
                edges, source.shape[1], channels, 512, num_warps=4,
            )
        return output

    @staticmethod
    def backward(ctx, grad):
        src, dst = ctx.saved_tensors
        table_grads = tuple(
            grad.new_zeros(ctx.shape).index_add_(1, ids, grad) if need else None
            for need, ids in zip(ctx.needs_input_grad[1:3], (src, dst), strict=True)
        )
        return grad, *table_grads, None, None


def use_projected_pair(hidden, source, target, source_ids, target_ids):
    if (
        not hidden.is_cuda or hidden.dtype != torch.float32
        or hidden.shape[-1] != 128 or hidden.shape[1] < 16384
        or torch.version.hip is not None or torch.compiler.is_compiling()
    ):
        return False
    capability = torch.cuda.get_device_capability(hidden.device)
    return (
        (capability == (9, 0) or (capability == (8, 0) and hidden.shape[1] >= 32768))
        and all(value.device == hidden.device and value.is_contiguous() and not value.is_neg()
                for value in (hidden, source, target, source_ids, target_ids))
    )


def add_projected_atom_pair(hidden, source, target, source_ids, target_ids):
    if use_projected_pair(hidden, source, target, source_ids, target_ids):
        return ProjectedAtomPair.apply(hidden, source, target, source_ids, target_ids)
    hidden = hidden + source.index_select(1, source_ids)
    return hidden + target.index_select(1, target_ids)


def indexed_single_radial(inputs, parameters, source_table, target_table, source_ids, target_ids):
    def group_affine(value):
        return value.resolve_neg().contiguous().unsqueeze(0)

    distance_channels = inputs.shape[-1]
    embedding_end = distance_channels + source_table.shape[-1]
    weight = parameters.first_weight
    source = F.linear(source_table, weight[:, distance_channels:embedding_end]).unsqueeze(0)
    target = F.linear(target_table, weight[:, embedding_end:]).unsqueeze(0)
    hidden = F.linear(inputs, weight[:, :distance_channels]).unsqueeze(0)
    hidden = add_projected_atom_pair(hidden, source, target, source_ids, target_ids)
    hidden = fused_grouped_layer_norm_silu(
        hidden, group_affine(parameters.first_bias),
        group_affine(parameters.first_norm_weight), group_affine(parameters.first_norm_bias),
        parameters.first_eps,
    )
    hidden = F.linear(hidden.squeeze(0), parameters.second_weight).unsqueeze(0)
    hidden = fused_grouped_layer_norm_silu(
        hidden, group_affine(parameters.second_bias),
        group_affine(parameters.second_norm_weight), group_affine(parameters.second_norm_bias),
        parameters.second_eps,
    )
    return F.linear(hidden.squeeze(0), parameters.final_weight, parameters.final_bias)


def grouped_radial_first(
    edge_distance,
    source_indices,
    target_indices,
    first_weights,
    first_biases,
    source_embeddings,
    target_embeddings,
    norm_weights,
    norm_biases,
    *,
    distance_channels,
    embedding_channels,
    norm_eps,
):
    """Project atom tables before gathering their rows for each edge.

    This avoids concatenating repeated atom embeddings on the edges.
    Return hidden features in (networks, edges, channels) order.
    """
    first_weight = torch.stack(first_weights)
    embedding_end = distance_channels + embedding_channels
    source_weight = first_weight[..., distance_channels:embedding_end]
    target_weight = first_weight[..., embedding_end:]
    source_tables = torch.bmm(
        torch.stack(source_embeddings), source_weight.transpose(1, 2)
    )
    target_tables = torch.bmm(
        torch.stack(target_embeddings), target_weight.transpose(1, 2)
    )
    hidden = torch.matmul(
        edge_distance, first_weight[..., :distance_channels].transpose(1, 2)
    )
    input_bias = torch.stack(first_biases)
    norm_weight = torch.stack(norm_weights)
    norm_bias = torch.stack(norm_biases)
    hidden = add_projected_atom_pair(hidden, source_tables, target_tables, source_indices, target_indices)
    return fused_grouped_layer_norm_silu(
        hidden, input_bias, norm_weight, norm_bias, norm_eps
    )


class GroupedRadialLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        return torch.bmm(inputs, weights.mT)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            if torch.is_grad_enabled() and weights.requires_grad:
                # Keep the per-group weight reduction when differentiating the input gradient.
                grad_inputs = GroupedRadialLinear.apply(grad_output, weights.mT)
            else:
                grad_inputs = torch.bmm(grad_output, weights)
        grad_weights = None
        if ctx.needs_input_grad[1]:
            # Separate products let cuBLAS split the long edge reduction on SM80.
            grad_weights = torch.stack(tuple(
                dy.mT.mm(x) for x, dy in zip(inputs.unbind(), grad_output.unbind(), strict=True)
            ))
        return grad_inputs, grad_weights


def grouped_radial_linear(inputs, weights):
    if (
        not torch.compiler.is_compiling()
        and weights.requires_grad and torch.is_grad_enabled()
        and inputs.is_cuda and not torch.is_autocast_enabled("cuda")
        and torch.version.hip is None
    ):
        groups, output_channels, input_channels = weights.shape
        rows = inputs.shape[1]
        small_groups = (
            2 <= groups <= 4 and 1024 <= rows <= 16384
            and max(output_channels, input_channels) <= 128
        )
        esen_groups = (
            groups == 10 and 16384 <= rows <= 32768
            and (output_channels, input_channels) == (128, 128)
        )
        if small_groups or esen_groups:
            capability = torch.cuda.get_device_capability(inputs.device)
            if capability == (8, 0) or (capability == (9, 0) and small_groups and rows >= 4096):
                return GroupedRadialLinear.apply(inputs, weights)
    return torch.bmm(inputs, weights.mT)


def radial_network(
    inputs, weights, biases, norm_weights, norm_biases, *, eps=1e-5,
    atom_types=None, edge_index=None, embeddings=None,
):
    """Evaluate two Linear-LayerNorm-SiLU stages and an output projection.

    ``inputs`` contains distance-basis or preassembled edge features ``(E,D)``.
    ``weights`` holds three PyTorch-order matrices ``(H1,Cin)``, ``(H2,H1)``,
    and ``(Cout,H2)``. ``biases`` holds their bias vectors; only the output
    bias may be ``None``. ``norm_weights`` and ``norm_biases`` each hold two
    vectors, of lengths H1 and H2. Return ``(E,Cout)``.

    To evaluate independent networks sharing the input, prepend the same
    network dimension S to every parameter tensor. Batched hidden stages and
    output products return ``(S,E,Cout)``. Parameters remain independent.

    Optional ``embeddings=(source_table, receiver_table)`` supplies atom-type
    tables ``(T,B)`` or, for independent tables, ``(S,T,B)``. Also provide
    int64 ``atom_types[N]`` and ``edge_index[2,E]``. The first map acts on
    ``[inputs, source_embedding, receiver_embedding]``. Tables are projected
    before their rows are gathered, avoiding an expanded edge embedding.

    ``eps`` is a positive scalar or a pair of hidden-stage epsilons.
    All floating tensors use CUDA FP32 on one device. Input and parameter
    gradients support derivatives through third order. No parameters are owned
    or copied into a module by this function.
    """
    weights, biases = tuple(weights), tuple(biases)
    norm_weights, norm_biases = tuple(norm_weights), tuple(norm_biases)
    if len(weights) != 3 or len(biases) != 3:
        raise ValueError("radial_network requires three linear weights and biases")
    if len(norm_weights) != 2 or len(norm_biases) != 2:
        raise ValueError("radial_network requires two affine normalization stages")
    eps = (eps, eps) if isinstance(eps, (int, float)) else tuple(eps)
    if len(eps) != 2 or any(not isfinite(value) or value <= 0 for value in eps):
        raise ValueError("eps must contain one or two positive finite values")
    if inputs.ndim != 2 or weights[0].ndim not in (2, 3):
        raise ValueError("inputs must be (E,D); weights must be matrices or batches")
    grouped = weights[0].ndim == 3
    groups = weights[0].shape[0] if grouped else 1
    if groups <= 0 or any(weight.ndim != weights[0].ndim for weight in weights):
        raise ValueError("all radial weights must use the same nonempty network axis")
    prefix = (groups,) if grouped else ()
    h1, cin = weights[0].shape[-2:]
    h2, previous = weights[1].shape[-2:]
    cout, previous_output = weights[2].shape[-2:]
    if min(h1, h2, cin, cout) <= 0 or previous != h1 or previous_output != h2:
        raise ValueError("radial linear dimensions do not form a chain")
    if any(weight.shape[:-2] != prefix for weight in weights):
        raise ValueError("radial linear weights must share their network axis")
    for index, (value, width) in enumerate(zip(biases, (h1, h2, cout), strict=True)):
        if value is None:
            if index == 2:
                continue
            raise ValueError("both hidden biases are required")
        if tuple(value.shape) != (*prefix, width):
            raise ValueError("radial bias dimensions do not match linear weights")
    for values in (norm_weights, norm_biases):
        if any(tuple(value.shape) != (*prefix, width)
               for value, width in zip(values, (h1, h2), strict=True)):
            raise ValueError("normalization vectors must match the hidden widths")
    float_inputs = (inputs, *weights, *biases[:2], *norm_weights, *norm_biases)
    if biases[2] is not None:
        float_inputs += (biases[2],)
    require_cuda_fp32(*float_inputs, operation="radial_network")

    indexed = embeddings is not None
    if indexed != (atom_types is not None and edge_index is not None):
        raise ValueError("embeddings, atom_types and edge_index must be supplied together")
    if not indexed and (atom_types is not None or edge_index is not None):
        raise ValueError("atom indices require source and receiver embedding tables")
    parameters = []
    for index in range(groups):
        def select(value):
            return value[index] if grouped and value is not None else value
        parameters.append(RadialParameters(
            select(weights[0]), select(biases[0]),
            select(norm_weights[0]), select(norm_biases[0]), eps[0],
            select(weights[1]), select(biases[1]),
            select(norm_weights[1]), select(norm_biases[1]), eps[1],
            select(weights[2]), select(biases[2]),
        ))

    if indexed:
        embeddings = tuple(embeddings)
        if len(embeddings) != 2:
            raise ValueError("embeddings must contain source and receiver tables")
        require_cuda_fp32(*embeddings, inputs, operation="radial_network")
        if any(table.ndim not in (2, 3) for table in embeddings):
            raise ValueError("embedding tables must be (T,B) or (S,T,B)")
        if (embeddings[0].shape[-2:] != embeddings[1].shape[-2:]
                or inputs.shape[1] + 2 * embeddings[0].shape[-1] != cin):
            raise ValueError("embedding and distance widths do not match the first map")
        if any(table.ndim == 3 and (not grouped or table.shape[0] != groups)
               for table in embeddings):
            raise ValueError("embedding network axes must match the radial weights")
        if (atom_types.dtype != torch.int64 or atom_types.ndim != 1
                or edge_index.dtype != torch.int64
                or tuple(edge_index.shape) != (2, inputs.shape[0])
                or atom_types.device != inputs.device or edge_index.device != inputs.device):
            raise ValueError("atom_types and edge_index must be int64 tensors on the input device")
        if (
            not grouped and 0 < inputs.shape[0] <= 65536
            and inputs.shape[1] == 64 and embeddings[0].shape[-1] == 32
            and h1 == h2 == cout == 128
            and torch.version.hip is None
            and not torch.compiler.is_compiling()
            and (inputs.shape[0] >= 128 or not torch.backends.cuda.matmul.allow_tf32)
            and not torch.are_deterministic_algorithms_enabled()
            and not torch.is_autocast_enabled("cuda")
            and all(value.is_contiguous() and not value.is_neg() for value in (
                inputs, *embeddings, atom_types, edge_index,
                *biases[:2], *norm_weights, *norm_biases,
            ))
        ):
            capability = torch.cuda.get_device_capability(inputs.device)
            if capability == (9, 0) or (
                capability == (8, 0) and inputs.shape[0] <= 16384
                and (inputs.shape[0] >= 8192 or not torch.backends.cuda.matmul.allow_tf32)
            ):
                with torch.cuda.device(inputs.device):
                    rows = inputs.shape[0]
                    training_graph = torch.is_grad_enabled()
                    fused_rows = 16384 if capability == (9, 0) else 12288
                    fused_shape = 8192 <= rows <= fused_rows
                    ampere_eager = (
                        capability == (8, 0) and 256 <= rows < 8192 and training_graph
                        and not torch.cuda.is_current_stream_capturing()
                    )
                    if (
                        (fused_shape or ampere_eager)
                        and not torch.backends.cuda.matmul.allow_tf32
                        and all(w.is_contiguous() and not w.is_neg() for w in weights[:2])
                    ):
                        arguments = (
                            inputs, *embeddings, atom_types, edge_index,
                            weights[0], biases[0].unsqueeze(0),
                            norm_weights[0].unsqueeze(0), norm_biases[0].unsqueeze(0),
                            weights[1], biases[1].unsqueeze(0),
                            norm_weights[1].unsqueeze(0), norm_biases[1].unsqueeze(0), *eps,
                        )
                        if training_graph:
                            backend = indexed_radial_backend(inputs, *embeddings, atom_types, *weights[:2])
                            if fused_shape or backend is not None:
                                hidden, *_ = IndexedRadialHidden.apply(*arguments, backend)
                                return F.linear(hidden, weights[2], biases[2])
                        else:
                            hidden = indexed_radial_hidden_inference(*arguments)
                            return F.linear(hidden, weights[2], biases[2])
                    return radial_mlp_forward(
                        parameters[0],
                        RadialInput.apply(inputs, *embeddings, atom_types, edge_index, True),
                        forward_num_warps=1 if capability == (9, 0) else 4,
                        # cuBLAS uses a faster GEMM with bias for these short FP32 matrices.
                        bias_in_linear=2 <= inputs.shape[0] <= 16,
                        transpose_cotangent=capability == (9, 0) and 8 <= rows <= 16 and training_graph,
                    )
        checked_types = False
        if (
            not grouped and 32768 <= inputs.shape[0] <= 131072
            and torch.version.hip is None and not torch.compiler.is_compiling()
            and atom_types.is_contiguous() and not atom_types.is_neg()
            and edge_index.is_contiguous() and not edge_index.is_neg()
        ):
            capability = torch.cuda.get_device_capability(inputs.device)
            checked_types = capability == ((8, 0) if inputs.shape[0] <= 65536 else (9, 0))
        if checked_types:
            with torch.cuda.device(inputs.device):
                source_types, receiver_types = checked_radial_types(
                    atom_types, edge_index, embeddings[0].shape[-2],
                )
        else:
            torch._assert_async(((edge_index >= 0) & (edge_index < atom_types.numel())).all(),
                                "radial edge endpoints must index atom_types")
            torch._assert_async(((atom_types >= 0) & (atom_types < embeddings[0].shape[-2])).all(),
                                "atom_types must index the embedding tables")
            source_types = atom_types[edge_index[0]]
            receiver_types = atom_types[edge_index[1]]
        tables = [tuple(table.unbind(0)) if table.ndim == 3 else (table,) * groups
                  for table in embeddings]
        # At larger sizes the batched layout uses less memory in third backward.
        if groups == 1 and inputs.shape[0] <= 65536:
            output = indexed_single_radial(
                inputs, parameters[0], tables[0][0], tables[1][0], source_types, receiver_types,
            )
            return output.unsqueeze(0) if grouped else output
        first = grouped_radial_first(
            inputs, source_types, receiver_types,
            tuple(p.first_weight for p in parameters),
            tuple(p.first_bias for p in parameters),
            *tables,
            tuple(p.first_norm_weight for p in parameters),
            tuple(p.first_norm_bias for p in parameters),
            distance_channels=inputs.shape[1], embedding_channels=embeddings[0].shape[-1],
            norm_eps=eps[0],
        )
        hidden = grouped_radial_linear(first, torch.stack([p.second_weight for p in parameters]))
        hidden = fused_grouped_layer_norm_silu(
            hidden, torch.stack([p.second_bias for p in parameters]),
            torch.stack([p.second_norm_weight for p in parameters]),
            torch.stack([p.second_norm_bias for p in parameters]), eps[1],
        )
    else:
        if inputs.shape[1] != cin:
            raise ValueError("input width does not match the first linear map")
        if not grouped:
            return radial_mlp_forward(parameters[0], inputs)
        hidden = unified_radial_hidden(parameters, inputs)
    outputs = grouped_radial_linear(hidden, torch.stack([p.final_weight for p in parameters]))
    if biases[2] is not None:
        outputs.add_((biases[2] if grouped else biases[2].unsqueeze(0)).unsqueeze(1))
    return outputs if grouped else outputs.squeeze(0)


class RadialInputBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        source_weight,
        target_weight,
        atomic_numbers,
        edge_index,
        distance_channels,
    ):
        gradients = distance_atom_pair_backward(
            grad_output,
            source_weight,
            target_weight,
            atomic_numbers,
            edge_index,
            distance_channels,
        )
        ctx.save_for_backward(atomic_numbers, edge_index)
        ctx.distance_channels = distance_channels
        ctx.weight_shape = tuple(source_weight.shape)
        ctx.dtype = source_weight.dtype
        return gradients

    @staticmethod
    def backward(
        ctx,
        distance_direction,
        source_weight_direction,
        target_weight_direction,
    ):
        atomic_numbers, edge_index = ctx.saved_tensors
        if source_weight_direction is None:
            source_weight_direction = torch.zeros(
                ctx.weight_shape,
                device=atomic_numbers.device,
                dtype=ctx.dtype,
            )
        if distance_direction is None:
            distance_direction = torch.zeros(
                (edge_index.shape[1], ctx.distance_channels),
                device=atomic_numbers.device,
                dtype=source_weight_direction.dtype,
            )
        if target_weight_direction is None:
            target_weight_direction = torch.zeros_like(source_weight_direction)
        arguments = (
            distance_direction.resolve_neg().contiguous(),
            source_weight_direction.resolve_neg().contiguous(),
            target_weight_direction.resolve_neg().contiguous(),
            atomic_numbers,
            edge_index,
        )
        # The pack is linear. Keep its adjoint available when differentiating
        # the double backward, including directions into the embedding tables.
        grad_grad_output = (
            RadialInput.apply(*arguments, False)
            if torch.is_grad_enabled()
            else distance_atom_pair_forward(*arguments)
        )
        return grad_grad_output, None, None, None, None, None


class RadialInput(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        distance,
        source_weight,
        target_weight,
        atomic_numbers,
        edge_index,
        checked,
    ):
        pack = checked_distance_atom_pair_forward if checked else distance_atom_pair_forward
        output = pack(
            distance,
            source_weight,
            target_weight,
            atomic_numbers,
            edge_index,
        )
        ctx.save_for_backward(source_weight, target_weight, atomic_numbers, edge_index)
        ctx.distance_channels = distance.shape[1]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        source_weight, target_weight, atomic_numbers, edge_index = ctx.saved_tensors
        arguments = (
            grad_output.contiguous(),
            source_weight,
            target_weight,
            atomic_numbers,
            edge_index,
            ctx.distance_channels,
        )
        if torch.is_grad_enabled():
            gradients = RadialInputBackward.apply(*arguments)
        elif ctx.needs_input_grad[0] and not any(ctx.needs_input_grad[1:3]):
            gradients = distance_atom_pair_backward(*arguments, compute_embedding_gradients=False)
        else:
            gradients = distance_atom_pair_backward(*arguments)
        return gradients[0], gradients[1], gradients[2], None, None, None


def fused_radial_input(
    distance,
    source_weight,
    target_weight,
    atomic_numbers,
    edge_index,
):
    """Concatenate distance features with source and receiver atom embeddings.

    Parameters
    ----------
    distance : Tensor, shape (E, D)
        Contiguous CUDA FP32 distance-basis features, already evaluated by
        the caller. These are features, not raw scalar distances.
    source_weight, target_weight : Tensor, shape (T, C)
        Contiguous CUDA FP32 embedding tables of equal shape.
    atomic_numbers : Tensor, shape (N,)
        Contiguous CUDA int64 table indices in [0, T).
    edge_index : Tensor, shape (2, E)
        Contiguous CUDA int64 source and receiver node indices, in that order.
        All tensors are on the same device.

    Returns
    -------
    Tensor, shape (E, D+2*C)
        Concatenate distance, source_weight[atomic_numbers[source]], and
        target_weight[atomic_numbers[receiver]] along channels, in that order.

    One kernel writes the final layout without separate embedding gathers.
    Derivatives through third order cover distance features, table weights and
    incoming cotangents. Indices are not differentiable.
    """
    if distance.ndim != 2:
        raise ValueError("radial input distance features must have shape (E, D)")
    if (
        source_weight.ndim != 2
        or min(source_weight.shape) < 1
        or source_weight.shape != target_weight.shape
    ):
        raise ValueError(
            "radial embedding tables must have equal shapes (T, C) with T, C > 0"
        )
    require_cuda_fp32(
        distance,
        source_weight,
        target_weight,
        operation="radial input",
        contiguous=True,
    )
    if (
        atomic_numbers.ndim != 1
        or atomic_numbers.dtype != torch.long
        or atomic_numbers.device != distance.device
        or not atomic_numbers.is_contiguous()
    ):
        raise ValueError(
            "radial input requires contiguous int64 atom indices on the input device"
        )
    if (
        edge_index.dtype != torch.long
        or edge_index.device != distance.device
        or not edge_index.is_contiguous()
        or edge_index.shape != (2, distance.shape[0])
    ):
        raise ValueError(
            "radial input requires contiguous int64 [2, E] edge indices on the input device"
        )
    return RadialInput.apply(
        distance,
        source_weight,
        target_weight,
        atomic_numbers,
        edge_index,
        False,
    )


class GroupedLayerNormSiluBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, grad_output, inputs, input_bias, weight, bias, mean, rstd, num_warps, eps
    ):
        gradients = grouped_layer_norm_silu_backward(
            grad_output, inputs, input_bias, weight, bias, mean, rstd, num_warps
        )
        ctx.save_for_backward(grad_output, inputs, input_bias, weight, bias, mean, rstd)
        ctx.eps = eps
        return gradients

    @staticmethod
    def backward(ctx, grad_inputs, grad_input_bias, grad_weight, grad_bias):
        grad_output, inputs, input_bias, weight, bias, mean, rstd = ctx.saved_tensors
        if grad_inputs is None:
            raise RuntimeError(
                "grouped LayerNorm-SiLU double backward requires input cotangent"
            )
        args = (
            grad_output.contiguous(), inputs, input_bias, weight, bias, mean, rstd,
            grad_inputs.contiguous(),
            (
                torch.zeros_like(input_bias)
                if grad_input_bias is None
                else grad_input_bias.contiguous()
            ),
            None if grad_weight is None else grad_weight.contiguous(),
            None if grad_bias is None else grad_bias.contiguous(),
        )
        gradients = (
            GroupedLayerNormSiluDoubleBackward.apply(*args, ctx.eps)
            if torch.is_grad_enabled()
            else grouped_layer_norm_silu_double_backward(*args)
        )
        return *gradients, None, None, None, None


class GroupedLayerNormSiluDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, inputs, input_bias, weight, bias, mean, rstd, hx, hi, hw, hb, eps):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(dy, inputs, input_bias, weight, bias, hx, hi, hw, hb)
        ctx.eps = eps
        return grouped_layer_norm_silu_double_backward(
            dy, inputs, input_bias, weight, bias, mean, rstd, hx, hi, hw, hb,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        needs = (*ctx.needs_input_grad[:5], *ctx.needs_input_grad[7:11])
        *primals, hx, hi, hw, hb = grouped_layer_norm_silu_triple_backward(
            ctx.saved_tensors, cotangents, needs, ctx.eps,
        )
        return *primals, None, None, hx, hi, hw, hb, None


class GroupedLayerNormSilu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs, input_bias, weight, bias, eps, num_warps, forward_num_warps
    ):
        output, mean, rstd = grouped_layer_norm_silu_forward(
            inputs, input_bias, weight, bias, eps, forward_num_warps
        )
        ctx.save_for_backward(inputs, input_bias, weight, bias, mean, rstd)
        ctx.backward_num_warps = num_warps
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, input_bias, weight, bias, mean, rstd = ctx.saved_tensors
        num_warps = ctx.backward_num_warps
        if not torch.is_grad_enabled():
            if ctx.needs_input_grad[0] and not any(ctx.needs_input_grad[1:4]):
                grad_inputs = grouped_layer_norm_silu_backward_input(
                    grad_output.contiguous(), inputs, input_bias, weight, bias, mean, rstd,
                    num_warps,
                )
                return grad_inputs, None, None, None, None, None, None
            gradients = grouped_layer_norm_silu_backward(
                grad_output.contiguous(), inputs, input_bias, weight, bias, mean, rstd,
                num_warps,
            )
            return (*gradients, None, None, None)
        grad_inputs, grad_input_bias, grad_weight, grad_bias = GroupedLayerNormSiluBackward.apply(
            grad_output.contiguous(), inputs, input_bias, weight, bias, mean, rstd,
            num_warps, ctx.eps,
        )
        return grad_inputs, grad_input_bias, grad_weight, grad_bias, None, None, None


def indexed_radial_hidden_inference(
    x, source, target, atoms, edges,
    w1, b1, scale1, shift1, w2, b2, scale2, shift2, eps1, eps2,
):
    rows, nodes = x.shape[0], atoms.numel()
    hidden1 = x.new_empty((rows, 128))
    indexed_radial_linear_norm_kernel[(triton.cdiv(max(rows, nodes), 32),)](
        x, source, target, atoms, edges, None, w1, b1, scale1, shift1, None, hidden1,
        None, None, rows, nodes, source.shape[0], eps1, 32, 16, False,
        num_warps=4, num_stages=2,
    )
    hidden2 = x.new_empty((rows, 128))
    radial_linear_norm_kernel[(triton.cdiv(rows, 64),)](
        hidden1, w2, b2, scale2, shift2, None, hidden2, None, None,
        rows, eps2, 64, 16, False, num_warps=4, num_stages=2,
    )
    return hidden2


class IndexedRadialHidden(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, source, target, atoms, edges,
        w1, b1, scale1, shift1, w2, b2, scale2, shift2, eps1, eps2, backend,
    ):
        if backend is not None:
            packed, raw1, hidden1, mean1, rstd1, raw2, hidden2, mean2, rstd2 = native_indexed_radial(
                backend, x, source, target, atoms, edges, w1, b1, scale1, shift1,
                w2, b2, scale2, shift2, eps1, eps2,
            )
        else:
            rows, nodes = x.shape[0], atoms.numel()
            packed = x.new_empty((rows, 128))
            raw1 = x.new_empty((1, rows, 128))
            hidden1 = x.new_empty((rows, 128))
            mean1, rstd1 = x.new_empty(rows), x.new_empty(rows)
            indexed_radial_linear_norm_kernel[(triton.cdiv(max(rows, nodes), 32),)](
                x, source, target, atoms, edges, packed, w1, b1, scale1, shift1, raw1, hidden1,
                mean1, rstd1, rows, nodes, source.shape[0], eps1, 32, 16, num_warps=4, num_stages=2,
            )
            raw2 = x.new_empty(raw1.shape)
            hidden2 = x.new_empty((rows, 128))
            mean2, rstd2 = x.new_empty(rows), x.new_empty(rows)
            radial_linear_norm_kernel[(triton.cdiv(rows, 64),)](
                hidden1, w2, b2, scale2, shift2, raw2, hidden2, mean2, rstd2,
                rows, eps2, 64, 16, num_warps=4, num_stages=2,
            )
        ctx.save_for_backward(source, target, atoms, edges, w1, b1, scale1, shift1, w2, b2, scale2, shift2,
                              packed, raw1, hidden1, raw2, mean1, rstd1, mean2, rstd2)
        ctx.eps = eps1, eps2
        ctx.set_materialize_grads(False)
        # Expose the saved activations to the adjoints of higher-order calls.
        return hidden2, packed, raw1, hidden1, raw2

    @staticmethod
    def backward(ctx, dh2, dpacked, draw1, dh1, draw2):
        (source, target, atoms, edges, w1, b1, scale1, shift1, w2, b2, scale2, shift2,
         packed, raw1, hidden1, raw2, mean1, rstd1, mean2, rstd2) = ctx.saved_tensors
        tracking = torch.is_grad_enabled()
        needed = ctx.needs_input_grad
        packed_gradients = (
            not tracking and dh2 is not None and dpacked is None
            and draw1 is None and dh1 is None and draw2 is None
            and all(needed[:3]) and all(needed[5:13])
            and not torch.compiler.is_compiling()
        )

        def norm_vjp(dh, dr, raw, bias, scale, shift, mean, rstd, eps, needs_params):
            if dh is None:
                return (None if dr is None else dr.squeeze(0)), None, None, None
            args = dh.unsqueeze(0).contiguous(), raw, bias, scale, shift, mean, rstd, 4
            if tracking:
                dz, db, dn, dv = GroupedLayerNormSiluBackward.apply(*args, eps)
            elif needs_params:
                dz, db, dn, dv = grouped_layer_norm_silu_backward(*args, packed_gradients=packed_gradients)
            else:
                dz = grouped_layer_norm_silu_backward_input(*args)
                db = dn = dv = None
            if dr is not None:
                dz = dz + dr if tracking else dz.add_(dr)
            return dz.squeeze(0), db, dn, dv

        dz2, db2, dn2, dv2 = norm_vjp(dh2, draw2, raw2, b2, scale2, shift2, mean2, rstd2,
                                       ctx.eps[1], any(needed[10:13]))
        dw2 = None
        if dz2 is not None:
            if needed[9]:
                dw2 = dz2.t().mm(hidden1)
            if any(needed[:9]):
                update = dz2.mm(w2)
                if dh1 is not None:
                    update = update + dh1 if tracking else update.add_(dh1)
                dh1 = update
                del update

        # The first projection does not need the second projection cotangents.
        del dz2, dh2, draw2
        dz1, db1, dn1, dv1 = norm_vjp(dh1, draw1, raw1, b1, scale1, shift1, mean1, rstd1,
                                       ctx.eps[0], any(needed[6:9]))
        del dh1, draw1
        dw1 = None
        if dz1 is not None:
            if needed[5]:
                dw1 = dz1.t().mm(packed)
            if any(needed[:3]):
                update = dz1.mm(w1)
                if dpacked is not None:
                    update = update + dpacked if tracking else update.add_(dpacked)
                dpacked = update
                del update

        del dz1
        dx = ds = dt = None
        if dpacked is not None and any(needed[:3]):
            args = dpacked.contiguous(), source, target, atoms, edges, 64
            if tracking:
                dx, ds, dt = RadialInputBackward.apply(*args)
            else:
                dx, ds, dt = distance_atom_pair_backward(
                    *args, compute_embedding_gradients=any(needed[1:3]),
                    packed_gradients=packed_gradients,
                )
        return dx, ds, dt, None, None, dw1, db1, dn1, dv1, dw2, db2, dn2, dv2, None, None, None


def fused_grouped_layer_norm_silu(
    inputs, input_bias, weight, bias, eps, *,
    backward_num_warps=8, forward_num_warps=8,
):
    """Apply one affine LayerNorm-SiLU parameter set to each leading group."""

    return GroupedLayerNormSilu.apply(
        inputs.contiguous(), input_bias, weight, bias, eps,
        backward_num_warps, forward_num_warps,
    )


class LayerNormSiluBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, inputs, weight, bias, mean, rstd):
        gradients = layer_norm_silu_backward(grad_output, inputs, weight, bias, mean, rstd)
        ctx.save_for_backward(grad_output, inputs, weight, bias, mean, rstd)
        return gradients

    @staticmethod
    def backward(ctx, grad_inputs, grad_weight, grad_bias):
        grad_output, inputs, weight, bias, mean, rstd = ctx.saved_tensors
        if grad_inputs is None:
            raise RuntimeError(
                "LayerNorm-SiLU force-training double backward requires "
                "an input-gradient cotangent"
            )
        gradients = layer_norm_silu_double_backward(
            grad_output.contiguous(), inputs, weight, bias, mean, rstd,
            grad_inputs.contiguous(),
            None if grad_weight is None else grad_weight.contiguous(),
            None if grad_bias is None else grad_bias.contiguous(),
        )
        return (*gradients, None, None)


class LayerNormSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, eps):
        output, mean, rstd = layer_norm_silu_forward(inputs, weight, bias, eps)
        ctx.save_for_backward(inputs, weight, bias, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias, mean, rstd = ctx.saved_tensors
        if not torch.is_grad_enabled():
            if ctx.needs_input_grad[0] and not any(ctx.needs_input_grad[1:3]):
                return (
                    layer_norm_silu_backward_input(
                        grad_output.contiguous(), inputs, weight, bias, mean, rstd
                    ),
                    None,
                    None,
                    None,
                )
            return (
                *layer_norm_silu_backward(
                    grad_output.contiguous(), inputs, weight, bias, mean, rstd,
                ),
                None,
            )
        grad_inputs, grad_weight, grad_bias = LayerNormSiluBackward.apply(
            grad_output.contiguous(), inputs, weight, bias, mean, rstd
        )
        return grad_inputs, grad_weight, grad_bias, None


def fused_layer_norm_silu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute affine LayerNorm followed by SiLU without saving the affine output."""

    return LayerNormSilu.apply(inputs.contiguous(), weight, bias, eps)


def packed_radial_net(
    edge_distance: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    source_embedding_weight: torch.Tensor,
    target_embedding_weight: torch.Tensor,
    *,
    first_weight: torch.Tensor,
    first_bias: torch.Tensor,
    first_norm_weight: torch.Tensor,
    first_norm_bias: torch.Tensor,
    first_norm_eps: float,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor,
    second_norm_weight: torch.Tensor,
    second_norm_bias: torch.Tensor,
    second_norm_eps: float,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Build atom-pair features, then apply a radial MLP.

    ``edge_distance`` is (E, D) and contains evaluated distance-basis
    features. ``atomic_numbers`` is (N,) and indexes the (T, C) source and
    target embedding tables. ``edge_index`` is (2, E), with source indices
    first and receiver indices second. The packed edge feature concatenates
    distance, source embedding and receiver embedding in that order.

    The MLP applies Linear-LayerNorm-SiLU twice, then a final Linear:

    - ``first_weight``: (H1, D+2*C), with bias and norm vectors (H1,).
    - ``second_weight``: (H2, H1), with bias and norm vectors (H2,).
    - ``output_weight``: (Cout, H2), with bias (Cout,).

    Weights follow PyTorch's (output, input) convention. The two norm
    epsilons are positive. Return an (E, Cout) tensor.

    Floating tensors use CUDA FP32 on one device. Distance features,
    embedding tables and norm vectors are contiguous. Indices are contiguous
    CUDA int64 tensors. Embedding indices lie in [0, T).

    Kernels gather embeddings directly into the packed input and fuse each
    LayerNorm with SiLU. All floating inputs and parameters support derivatives
    through third order.
    """
    packed = fused_radial_input(
        edge_distance, source_embedding_weight, target_embedding_weight,
        atomic_numbers, edge_index,
    )
    outputs = F.linear(packed, first_weight, first_bias)
    outputs = fused_layer_norm_silu(
        outputs, first_norm_weight, first_norm_bias, first_norm_eps
    )
    outputs = F.linear(outputs, second_weight, second_bias)
    outputs = fused_layer_norm_silu(
        outputs, second_norm_weight, second_norm_bias, second_norm_eps
    )
    return F.linear(outputs, output_weight, output_bias)


class RadialMLPRuntime:
    """Share hidden stages across radial networks consumed in block order."""

    def __init__(self, block_count, *, forward_num_warps=8):
        if block_count < 1:
            raise ValueError("at least one radial MLP is required")
        self.block_count = block_count
        self.forward_num_warps = forward_num_warps
        self.reset()

    def reset(self):
        self.input = None
        self.hidden = None
        self.parameters = None
        self.next_index = 0

    def __deepcopy__(self, memo):
        result = type(self)(self.block_count, forward_num_warps=self.forward_num_warps)
        memo[id(self)] = result
        return result

    def consume(self, index, x_edge, *, parameters=None, checkpoint_hidden=False, keep_hidden=False):
        """Consume one radial network and release the cache after the last one.

        When keep_hidden is set, return an independent hidden slice and the
        parameters of the final linear map.
        """
        try:
            if index == 0:
                # A new forward also recovers from an exception between blocks.
                self.reset()
                if parameters is None or len(parameters) != self.block_count:
                    raise ValueError("radial parameters must match the block count")
                self.input = x_edge
                self.parameters = tuple(parameters)

                def compute_hidden(value):
                    return unified_radial_hidden(
                        parameters, value, forward_num_warps=self.forward_num_warps
                    )

                # Return all layer views together so autograd assembles their gradients once.
                self.hidden = (
                    checkpoint(
                        compute_hidden,
                        x_edge,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if checkpoint_hidden and torch.is_grad_enabled()
                    else compute_hidden(x_edge)
                ).unbind(0)
                self.next_index = 0
            if self.hidden is None or self.input is not x_edge:
                raise RuntimeError(
                    "radial layers must consume one shared edge tensor in order"
                )
            if index != self.next_index:
                raise RuntimeError("radial layer order changed")
            final = self.parameters[index]
            if keep_hidden:
                # A view would retain the backing buffer for all layers.
                output = self.hidden[index].clone(), final.final_weight, final.final_bias
            else:
                output = F.linear(self.hidden[index], final.final_weight, final.final_bias)
            self.next_index += 1
            if self.next_index == self.block_count:
                self.reset()
            return output
        except Exception:
            self.reset()
            raise


class AtomPairRadialRuntime:
    """Batch the indexed first stage and consume each network's tail in order.

    Parameters are independent ``RadialParameters`` records. Source and target
    embedding tables contain one tensor per record. Only first-stage hidden
    values are cached; each is released as its network consumes it.
    """

    def __init__(self, parameters, source_embeddings, target_embeddings, *, distance_channels):
        self.parameters = tuple(parameters)
        self.source_embeddings = tuple(source_embeddings)
        self.target_embeddings = tuple(target_embeddings)
        self.block_count = len(self.parameters)
        self.distance_channels = distance_channels
        if not self.block_count:
            raise ValueError("at least one radial network is required")
        if len(self.source_embeddings) != self.block_count or len(self.target_embeddings) != self.block_count:
            raise ValueError("each radial network requires source and target embedding tables")
        reference = self.contract(self.parameters[0], self.source_embeddings[0], self.target_embeddings[0])
        for params, source, target in zip(self.parameters, self.source_embeddings, self.target_embeddings, strict=True):
            if self.contract(params, source, target) != reference:
                raise ValueError("grouped radial networks must have uniform dimensions and epsilons")
        self.reset()

    def contract(self, params, source, target):
        if source.ndim != 2 or target.shape != source.shape:
            raise ValueError("source and target embeddings must have equal (types, channels) shapes")
        if self.distance_channels < 0 or params.first_weight.shape[1] != self.distance_channels + 2 * source.shape[1]:
            raise ValueError("distance and embedding widths must match the first radial map")
        return (
            tuple(source.shape), tuple(target.shape),
            tuple(tuple(weight.shape) for weight in (params.first_weight, params.second_weight, params.final_weight)),
            tuple(params.first_norm_weight.shape), tuple(params.second_norm_weight.shape),
            params.first_eps, params.second_eps,
        )

    def reset(self):
        self.input_key = None
        self.hidden = None
        self.next_index = 0

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.parameters = deepcopy(self.parameters, memo)
        result.source_embeddings = deepcopy(self.source_embeddings, memo)
        result.target_embeddings = deepcopy(self.target_embeddings, memo)
        result.block_count = self.block_count
        result.distance_channels = self.distance_channels
        result.reset()
        return result

    def compute_first(self, edge_distance, source_indices, target_indices):
        params = self.parameters
        hidden = grouped_radial_first(
            edge_distance, source_indices, target_indices,
            tuple(p.first_weight for p in params), tuple(p.first_bias for p in params),
            self.source_embeddings, self.target_embeddings,
            tuple(p.first_norm_weight for p in params), tuple(p.first_norm_bias for p in params),
            distance_channels=self.distance_channels,
            embedding_channels=self.source_embeddings[0].shape[1], norm_eps=params[0].first_eps,
        )
        # Release each network's features when its block consumes them.
        return [value.clone() for value in hidden.unbind(0)]

    def consume(self, index, edge_distance, source_indices, target_indices):
        try:
            input_key = (edge_distance.data_ptr(), source_indices.data_ptr(), target_indices.data_ptr())
            if index == 0:
                self.reset()
                self.input_key = input_key
                self.hidden = self.compute_first(edge_distance, source_indices, target_indices)
            if self.hidden is None or input_key != self.input_key:
                raise RuntimeError("radial networks must share one radial input")
            if index != self.next_index:
                raise RuntimeError(f"radial block order changed: expected {self.next_index}, got {index}")
            params = self.parameters[index]
            hidden = self.hidden[index]
            self.hidden[index] = None
            output = F.linear(hidden, params.second_weight, params.second_bias)
            del hidden
            output = fused_layer_norm_silu(output, params.second_norm_weight, params.second_norm_bias, params.second_eps)
            output = F.linear(output, params.final_weight, params.final_bias)
            self.next_index += 1
            if self.next_index == self.block_count:
                self.reset()
            return output
        except Exception:
            self.reset()
            raise


class FrozenRadialNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_bias, weight, bias, eps, packed, tile, warps):
        groups, channels = weight.shape
        rows = x.shape[0 if packed else 1]
        output = x.new_empty((groups, rows, channels))
        mean = x.new_empty((groups * rows,))
        rstd = torch.empty_like(mean)
        tiled_radial_norm_forward[(triton.cdiv(rows, tile), groups)](
            x, input_bias, weight, bias, output, mean, rstd, rows, eps, groups, channels,
            tile, packed, num_warps=warps,
        )
        ctx.save_for_backward(x, input_bias, weight, bias, mean, rstd)
        ctx.launch = rows, groups, channels, tile, packed, warps
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x, input_bias, weight, bias, mean, rstd = ctx.saved_tensors
        rows, groups, channels, tile, packed, warps = ctx.launch
        out = torch.empty_like(x)
        frozen_radial_norm_backward[(triton.cdiv(rows, tile), groups)](
            grad.resolve_neg().contiguous(), x, input_bias, weight, bias, mean, rstd, out,
            rows, groups, channels, tile, packed, num_warps=warps,
        )
        return out, None, None, None, None, None, None, None


class FrozenRadialHidden(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor
    first_norm: tuple[torch.Tensor, ...]
    second_norm: tuple[torch.Tensor, ...]
    first_eps: float
    second_eps: float


def prepare_frozen_radial_hidden(parameters):
    if any(torch.is_tensor(value) and value.requires_grad for layer in parameters for value in layer):
        raise ValueError("The input-only radial path requires frozen weights")
    first_weight = torch.cat(tuple(layer.first_weight for layer in parameters))
    second_weight = torch.stack(tuple(layer.second_weight for layer in parameters)).transpose(1, 2)
    first_norm = tuple(torch.stack(tuple(getattr(layer, name) for layer in parameters))
                       for name in ("first_bias", "first_norm_weight", "first_norm_bias"))
    second_norm = tuple(torch.stack(tuple(getattr(layer, name) for layer in parameters))
                        for name in ("second_bias", "second_norm_weight", "second_norm_bias"))
    return FrozenRadialHidden(
        first_weight, second_weight, first_norm, second_norm, parameters[0].first_eps, parameters[0].second_eps,
    )


class FrozenJointLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, distance, weight, bias, w0, w1):
        ctx.save_for_backward(w0, w1)
        return F.linear(features, weight, bias).split((128, 512), dim=1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d0, d1):
        w0, w1 = ctx.saved_tensors
        return None, torch.mm(d0, w0) + torch.mm(d1, w1), None, None, None, None
