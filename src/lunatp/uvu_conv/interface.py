"""UVU graph convolution sums weighted edge messages at destination nodes.

Source and destination CSR tables serve feature gradients and aggregation.
Optional envelopes and per-edge scalar outputs extend the same fused kernels."""

from __future__ import annotations
from hashlib import sha256

import torch
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling
from ..graph import LunaTPGraph, get_graph_with_int64_offsets
from ..utils.layout import select_indexing
from ..utils.runtime import prepare_gradient, prepare_gradients
from .graph import SourceSegmentTables, UvuConvGraphTables
from .kernels.forward import run as run_forward
from .kernels.backward import run as run_backward
from .kernels.double_backward import run as run_double_backward
from .kernels.triple_backward import run as run_triple_backward


@torch.compiler.assume_constant_result
def register_ops(op):
    key = sha256(repr((op.index_mode, op.compute_dtype, op.index_bits, op.meta)).encode()).hexdigest()
    with REGISTER_OPERATOR_LOCK:
        name = f"conv_{key}_forward"
        if hasattr(torch.ops.lunatp, name):
            return name

        @torch.library.custom_op(f"lunatp::conv_{key}_forward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor?[] tables, SymInt num_nodes, bool return_l0) -> (Tensor, Tensor?)")
        def conv_forward(x, y, w, env, tables, num_nodes, return_l0):
            graph = UvuConvGraphTables(*tables[:7], SourceSegmentTables(*tables[7:12]), node_tiers=tables[12])
            result = run_forward(op, x, y, w, graph.src, graph.dst_rowptr, graph.node_order,
                                 env, return_l0, node_tiers=graph.node_tiers)
            return result if return_l0 else (result, None)

        @conv_forward.register_fake
        def conv_forward_fake(x, y, w, env, tables, num_nodes, return_l0):
            return x.new_empty((num_nodes, op.meta["out_dim"])), y.new_empty((y.shape[0], op.l0_dim)) if return_l0 else None

        @torch.library.custom_op(f"lunatp::conv_{key}_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_backward(x, y, w, env, g, gl0, tables, num_nodes, needs):
            graph = UvuConvGraphTables(*tables[:7], SourceSegmentTables(*tables[7:12]), node_tiers=tables[12])
            return run_backward(op, x, y, w, g, graph, env, gl0, needs=tuple(needs))

        @conv_backward.register_fake
        def conv_backward_fake(x, y, w, env, g, gl0, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_double_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor rx, Tensor ry, Tensor rw, Tensor? re, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_double_backward(x, y, w, env, g, gl0, rx, ry, rw, re, tables, num_nodes, needs):
            graph = UvuConvGraphTables(*tables[:7], SourceSegmentTables(*tables[7:12]), node_tiers=tables[12])
            result = run_double_backward(op, x, y, w, g, rx, ry, rw, graph, env, re, gl0, need_grad_env=needs[3])
            hx, hy, hw, hg = result[:4]
            he = result[4] if needs[3] else None
            hl0 = result[4 + int(needs[3])] if gl0 is not None else None
            return tuple(value if need else None for value, need in zip((hx, hy, hw, he, hg, hl0), needs))

        @conv_double_backward.register_fake
        def conv_double_backward_fake(x, y, w, env, g, gl0, rx, ry, rw, re, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env, g, gl0), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_triple_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor rx, Tensor ry, Tensor rw, Tensor? re, Tensor sx, Tensor sy, Tensor sw, Tensor sg, Tensor? se, Tensor? sl0, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_triple_backward(x, y, w, env, g, gl0, rx, ry, rw, re, sx, sy, sw, sg, se, sl0, tables, num_nodes, needs):
            graph = UvuConvGraphTables(*tables[:7], SourceSegmentTables(*tables[7:12]), node_tiers=tables[12])
            result = run_triple_backward(op, x, y, w, g, rx, ry, rw, sx, sy, sw, sg, graph, env, re, gl0,
                                         se, sl0, need_grad_env=needs[3])
            dx, dy, dw, dg = result[:4]
            offset = 4
            de = result[offset] if needs[3] else None
            offset += int(needs[3])
            dl0 = result[offset] if gl0 is not None else None
            offset += int(gl0 is not None)
            drx, dry, drw = result[offset:offset + 3]
            dre = result[-1] if needs[3] else None
            return tuple(value if need else None for value, need in zip((dx, dy, dw, de, dg, dl0, drx, dry, drw, dre), needs))

        @conv_triple_backward.register_fake
        def conv_triple_backward_fake(x, y, w, env, g, gl0, rx, ry, rw, re, sx, sy, sw, sg, se, sl0, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env, g, gl0, rx, ry, rw, re), needs))

        def setup_forward(ctx, inputs, output):
            x, y, w, env, tables, ctx.num_nodes, _ = inputs
            ctx.save_for_backward(x, y, w, env, *tables)
            ctx.table_grad = None if any(table is None for table in tables) else [None] * len(tables)
            ctx.set_materialize_grads(False)

        def setup_backward(ctx, inputs, output):
            ctx.num_nodes = inputs[-2]
            ctx.save_for_backward(*inputs[:-3], *inputs[-3])
            ctx.table_grad = None if any(table is None for table in inputs[-3]) else [None] * len(inputs[-3])
            ctx.set_materialize_grads(False)

        def forward_backward(ctx, g, gl0):
            x, y, w, env, *tables = ctx.saved_tensors
            g = x.new_zeros((ctx.num_nodes, op.meta["out_dim"])) if g is None else g.contiguous()
            gl0 = None if gl0 is None else gl0.contiguous()
            result = conv_backward(x, y, w, env, g, gl0, tables, ctx.num_nodes,
                                   list(ctx.needs_input_grad[:4]))
            return *result, ctx.table_grad, None, None

        def backward_backward(ctx, rx, ry, rw, re):
            x, y, w, env, g, gl0, *tables = ctx.saved_tensors
            rx, ry, rw = (prepare_gradient(value, template) for value, template in zip((rx, ry, rw), (x, y, w)))
            re = prepare_gradient(re, env) if env is not None else None
            result = conv_double_backward(x, y, w, env, g, gl0, rx, ry, rw, re,
                                          tables, ctx.num_nodes, list(ctx.needs_input_grad[:6]))
            return *result, ctx.table_grad, None, None

        def double_backward_backward(ctx, sx, sy, sw, se, sg, sl0):
            x, y, w, env, g, gl0, rx, ry, rw, re, *tables = ctx.saved_tensors
            sx, sy, sw, sg = (prepare_gradient(value, template) for value, template in zip((sx, sy, sw, sg), (x, y, w, g)))
            se = prepare_gradient(se, env) if env is not None else None
            sl0 = prepare_gradient(sl0, gl0) if gl0 is not None else None
            result = conv_triple_backward(x, y, w, env, g, gl0, rx, ry, rw, re,
                                          sx, sy, sw, sg, se, sl0, tables, ctx.num_nodes,
                                          list(ctx.needs_input_grad[:10]))
            return *result, ctx.table_grad, None, None

        conv_forward.register_autograd(forward_backward, setup_context=setup_forward)
        conv_backward.register_autograd(backward_backward, setup_context=setup_backward)
        conv_double_backward.register_autograd(double_backward_backward, setup_context=setup_backward)

        return name


class UVUConvDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w, g, rx, ry, rw, graph, env, re, gl0):
        ctx.op, ctx.graph = op, graph
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, y, w, g, rx, ry, rw, env, re, gl0)
        return run_double_backward(
            op,
            x,
            y,
            w,
            g,
            rx,
            ry,
            rw,
            graph,
            env,
            re,
            gl0,
            need_grad_env=ctx.needs_input_grad[9],
        )

    @staticmethod
    def backward(ctx, sx, sy, sw, sg, *extra):
        x, y, w, g, rx, ry, rw, env, re, gl0 = ctx.saved_tensors
        sx, sy, sw, sg = prepare_gradients((sx, sy, sw, sg), (x, y, w, g))
        need_grad_env = env is not None and ctx.needs_input_grad[9]
        se = extra[0] if need_grad_env and extra else None
        sl0 = (
            extra[int(need_grad_env)]
            if gl0 is not None and len(extra) > int(need_grad_env)
            else None
        )
        if se is not None:
            se = se.contiguous()
        if sl0 is not None:
            sl0 = sl0.contiguous()

        outputs = run_triple_backward(
            ctx.op,
            x,
            y,
            w,
            g,
            rx,
            ry,
            rw,
            sx,
            sy,
            sw,
            sg,
            ctx.graph,
            env,
            re,
            gl0,
            se,
            sl0,
            need_grad_env=need_grad_env,
        )

        de = dre = dl0 = None
        if need_grad_env and gl0 is not None:
            dx, dy, dw, dg, de, dl0, drx, dry, drw, dre = outputs
        elif need_grad_env:
            dx, dy, dw, dg, de, drx, dry, drw, dre = outputs
        elif gl0 is not None:
            dx, dy, dw, dg, dl0, drx, dry, drw = outputs
        else:
            dx, dy, dw, dg, drx, dry, drw = outputs
        needs = ctx.needs_input_grad
        return (
            None,
            dx if needs[1] else None,
            dy if needs[2] else None,
            dw if needs[3] else None,
            dg if needs[4] else None,
            drx if needs[5] else None,
            dry if needs[6] else None,
            drw if needs[7] else None,
            None,
            de if needs[9] else None,
            dre if needs[10] else None,
            dl0 if needs[11] else None,
        )


class UVUConvBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w, g, graph, env, gl0):
        ctx.op, ctx.graph = op, graph
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, y, w, g, env, gl0)
        gl0 = gl0 if gl0 is not None and gl0.numel() else None
        return run_backward(
            op,
            x,
            y,
            w,
            g,
            graph,
            env,
            gl0,
            needs=(True, True, True, ctx.needs_input_grad[6]),
        )

    @staticmethod
    def backward(ctx, rx, ry, rw, re):
        x, y, w, g, env, gl0 = ctx.saved_tensors
        rx, ry, rw = prepare_gradients((rx, ry, rw), (x, y, w))
        if env is not None and re is not None:
            re = re.contiguous()
        if torch.is_grad_enabled():
            outputs = UVUConvDoubleBackward.apply(
                ctx.op, x, y, w, g, rx, ry, rw, ctx.graph, env, re, gl0
            )
        else:
            outputs = run_double_backward(
                ctx.op,
                x,
                y,
                w,
                g,
                rx,
                ry,
                rw,
                ctx.graph,
                env,
                re,
                gl0,
                need_grad_env=ctx.needs_input_grad[6],
            )
        hx, hy, hw, hg = outputs[:4]
        need_grad_env = env is not None and ctx.needs_input_grad[6]
        he = outputs[4] if need_grad_env else None
        hl0 = outputs[4 + int(need_grad_env)] if gl0 is not None else None
        needs = ctx.needs_input_grad
        return (
            None,
            hx if needs[1] else None,
            hy if needs[2] else None,
            hw if needs[3] else None,
            hg if needs[4] else None,
            None,
            he if needs[6] else None,
            hl0 if needs[7] else None,
        )


class UVUConvForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, y, w, graph, env=None, return_l0=False):
        ctx.op = op
        ctx.graph = graph
        if env is None and not return_l0:
            ctx.save_for_backward(x, y, w)
            return run_forward(
                op,
                x,
                y,
                w,
                graph.src,
                graph.dst_rowptr,
                graph.node_order,
                node_tiers=graph.node_tiers,
            )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, y, w, env)
        return run_forward(
            op,
            x,
            y,
            w,
            graph.src,
            graph.dst_rowptr,
            graph.node_order,
            env,
            return_l0,
            node_tiers=graph.node_tiers,
        )

    @staticmethod
    def backward(ctx, grad_out, grad_l0=None):
        saved = ctx.saved_tensors
        x, y, w = saved[:3]
        env = saved[3] if len(saved) == 4 else None
        graph, op = ctx.graph, ctx.op
        if grad_out is None:
            g = x.new_zeros((x.shape[0], op.out_dim))
        else:
            g = grad_out.contiguous()
        gl0 = grad_l0
        if gl0 is not None:
            gl0 = gl0.contiguous()
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            gx, gy, gw, ge = UVUConvBackward.apply(op, x, y, w, g, graph, env, gl0)
        else:
            gx, gy, gw, ge = run_backward(
                op,
                x,
                y,
                w,
                g,
                graph,
                env,
                gl0,
                needs=(*needs[1:4], len(needs) > 5 and needs[5]),
            )
        if len(needs) == 5:
            return (
                None,
                gx if needs[1] else None,
                gy if needs[2] else None,
                gw if needs[3] else None,
                None,
            )
        return (
            None,
            gx if needs[1] else None,
            gy if needs[2] else None,
            gw if needs[3] else None,
            None,
            ge if needs[5] else None,
            None,
        )


def lunatp_uvu_conv_func(
    op,
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    weight: torch.Tensor,
    *,
    graph: LunaTPGraph,
    env: torch.Tensor | None = None,
    return_l0_feat: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Aggregate UVU edge messages with derivatives through third order.

    Args:
        op: Initialized UVU convolution operator.
        x: Node features [num_nodes, in1_dim].
        edge_attr: Edge attributes [num_edges, in2_dim].
        weight: External per-edge weights [num_edges, weight_numel].
        graph: Shared topology prepared with make_graph.
        env: Optional edge scalars [num_edges] or [num_edges, 1], applied to node messages.
        return_l0_feat: Also return raw per-edge scalar contributions, with one
            column per input channel of each original scalar-output instruction.
            These features do not include env.

    Returns:
        Node output [num_nodes, out_dim], or (node_output, l0_feat).

    Features, weights and env share one CUDA device and the operator's
    FP32/FP64 dtype. Edge inputs follow the graph builder's ordering convention."""
    if (graph.src_rowptr.dtype == torch.int64 or x.shape[0] > op.index_node_limit
            or edge_attr.shape[0] > op.index_edge_limit):
        op = select_indexing(op, x.shape[0], edge_attr.shape[0])
    if op.index_bits == 64:
        graph = get_graph_with_int64_offsets(graph, cache=not is_compiling())
    graph_tables = graph._tables[("uvu", op.index_bits)]
    x = x.contiguous()
    edge_attr = edge_attr.contiguous()
    weight = weight.contiguous()
    if env is not None:
        if env.ndim == 1:
            env = env.unsqueeze(-1)
        env = env.contiguous()
    if is_compiling():
        segments = graph.source_segments
        tables = [
            graph.src, graph.dst, graph.dst_rowptr, graph.src_rowptr,
            graph.edge_ids_by_src, graph.source_node_order,
            graph.node_order,
            segments.nodes, segments.begin, segments.end, segments.rowptr, segments.order,
            graph.node_tiers,
        ]
        result, l0_feat = getattr(torch.ops.lunatp, op.register_ops()).default(
            x, edge_attr, weight, env, tables,
            graph.num_nodes, return_l0_feat,
        )
        if return_l0_feat:
            return result, l0_feat
        return result
    if env is None and not return_l0_feat:
        return UVUConvForward.apply(op, x, edge_attr, weight, graph_tables)
    return UVUConvForward.apply(
        op, x, edge_attr, weight, graph_tables, env, return_l0_feat
    )


__all__ = ["lunatp_uvu_conv_func"]
