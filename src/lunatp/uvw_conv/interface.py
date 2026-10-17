"""UVW graph convolution aggregates tensor products with channel mixing.

Shared and per-edge weights use distinct contraction layouts; optional edge
envelopes and all mixed derivatives through third order use those same layouts."""

from __future__ import annotations
from hashlib import sha256
from types import SimpleNamespace
import torch
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling
from ..graph import LunaTPGraph, get_graph_with_int64_offsets
from ..utils.layout import select_indexing
from ..utils.runtime import prepare_gradient, prepare_gradients
from .kernels.forward import run as run_forward
from .kernels.backward import run as run_backward
from .kernels.double_backward import run as run_double_backward
from .kernels.triple_backward import run as run_triple_backward, run_envelope_backward


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
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            result = run_forward(op, x, y, w, graph=graph, env=env)
            return result if return_l0 else (result, None)

        @conv_forward.register_fake
        def conv_forward_fake(x, y, w, env, tables, num_nodes, return_l0):
            return x.new_empty((num_nodes, op.meta["out_dim"])), y.new_empty((y.shape[0], op.l0_dim)) if return_l0 else None

        @torch.library.custom_op(f"lunatp::conv_{key}_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_backward(x, y, w, env, g, gl0, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            result = run_backward(op, x, y, w, g, graph=graph, needs=needs[:3]) if env is None else run_envelope_backward(
                op, 1, (x, y, w, g, env), graph, needs=needs
            )
            result = (pad_node_gradient(result[0], x), *result[1:])
            return (*result, None) if env is None else result

        @conv_backward.register_fake
        def conv_backward_fake(x, y, w, env, g, gl0, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_double_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor rx, Tensor ry, Tensor rw, Tensor? re, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_double_backward(x, y, w, env, g, gl0, rx, ry, rw, re, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            if env is None:
                hx, hy, hw, hg = run_double_backward(op, x, y, w, g, rx, ry, rw, graph=graph, needs=(*needs[:3], needs[4]))
                he = None
            else:
                hx, hy, hw, hg, he = run_envelope_backward(op, 2, (x, y, w, g, env, rx, ry, rw, re),
                                                          graph, needs=(*needs[:3], needs[4], needs[3]))
            return pad_node_gradient(hx, x), hy, hw, he, hg, None

        @conv_double_backward.register_fake
        def conv_double_backward_fake(x, y, w, env, g, gl0, rx, ry, rw, re, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env, g, gl0), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_triple_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? gl0, Tensor rx, Tensor ry, Tensor rw, Tensor? re, Tensor sx, Tensor sy, Tensor sw, Tensor sg, Tensor? se, Tensor? sl0, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_triple_backward(x, y, w, env, g, gl0, rx, ry, rw, re, sx, sy, sw, sg, se, sl0, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            if env is None:
                dx, dy, dw, dg, drx, dry, drw = run_triple_backward(op, x, y, w, g, rx, ry, rw, sx, sy, sw, sg,
                                                   graph=graph, needs=(*needs[:3], needs[4], *needs[6:9]))
                de = dre = None
            else:
                dx, dy, dw, dg, de, drx, dry, drw, dre = run_envelope_backward(
                    op, 3, (x, y, w, g, env, rx, ry, rw, re, sx, sy, sw, sg, se), graph,
                    needs=(*needs[:3], needs[4], needs[3], *needs[6:9], needs[9]))
            return pad_node_gradient(dx, x), dy, dw, de, dg, None, pad_node_gradient(drx, rx), dry, drw, dre

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


class UVWConvDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        op,
        x,
        edge_attr,
        weight,
        grad_out,
        rx,
        ry,
        rw,
        src,
        dst,
        graph,
        num_nodes,
        env=None,
        re=None,
    ):
        ctx.op = op
        ctx.graph = graph
        ctx.num_nodes = num_nodes
        if env is not None:
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(
                x, edge_attr, weight, grad_out, rx, ry, rw, src, dst, env, re
            )
            result = run_envelope_backward(
                op, 2, (x, edge_attr, weight, grad_out, env, rx, ry, rw, re), graph
            )
            return (pad_node_gradient(result[0], x), *result[1:])
        ctx.save_for_backward(x, edge_attr, weight, grad_out, rx, ry, rw, src, dst)
        grad_x, grad_edge, grad_weight, grad_grad_out = run_double_backward(
            op, x, edge_attr, weight, grad_out, rx, ry, rw, graph=graph
        )
        if grad_x is not None and grad_x.shape[0] != x.shape[0]:
            grad_x = pad_node_gradient(grad_x, x)
        # Return the second-order derivatives in public (x, edge, weight, grad_out) order.
        return grad_x, grad_edge, grad_weight, grad_grad_out

    @staticmethod
    def backward(ctx, bx, by, bw, bg, be=None):
        op = ctx.op
        saved = ctx.saved_tensors
        if len(saved) == 11:
            x, edge_attr, weight, grad_out, rx, ry, rw, _, _, env, re = saved
            bx, by, bw, bg, be = prepare_gradients(
                (bx, by, bw, bg, be), (x, edge_attr, weight, grad_out, env)
            )
            needs = ctx.needs_input_grad
            selected = (*needs[1:5], needs[12], *needs[5:8], needs[13])
            gx, gy, gw, gg, ge, grx, gry, grw, gre = run_envelope_backward(
                op,
                3,
                (
                    x,
                    edge_attr,
                    weight,
                    grad_out,
                    env,
                    rx,
                    ry,
                    rw,
                    re,
                    bx,
                    by,
                    bw,
                    bg,
                    be,
                ),
                ctx.graph,
                selected,
            )
            return (
                None,
                pad_node_gradient(gx, x),
                gy,
                gw,
                gg,
                pad_node_gradient(grx, rx),
                gry,
                grw,
                None,
                None,
                None,
                None,
                ge,
                gre,
            )
        x, edge_attr, weight, grad_out, rx, ry, rw, _, _ = saved
        bx = prepare_gradient(bx, x)
        by = prepare_gradient(by, edge_attr)
        bw = prepare_gradient(bw, weight)
        bg = prepare_gradient(bg, grad_out)
        graph = ctx.graph
        needs = ctx.needs_input_grad[1:8]
        gx, gy, gw, gg, grx, gry, grw = run_triple_backward(
            op,
            x,
            edge_attr,
            weight,
            grad_out,
            rx,
            ry,
            rw,
            bx,
            by,
            bw,
            bg,
            graph=graph,
            needs=needs,
        )
        if gx is not None and gx.shape[0] != x.shape[0]:
            gx = pad_node_gradient(gx, x)
        if grx is not None and grx.shape[0] != rx.shape[0]:
            grx = pad_node_gradient(grx, rx)
        needs = ctx.needs_input_grad
        return (
            None,
            gx if needs[1] else None,
            gy if needs[2] else None,
            gw if needs[3] else None,
            gg if needs[4] else None,
            grx if needs[5] else None,
            gry if needs[6] else None,
            grw if needs[7] else None,
            None,
            None,
            None,
            None,
        )


class UVWConvBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        op,
        x,
        edge_attr,
        weight,
        grad_out,
        src,
        dst,
        graph,
        num_nodes,
        env=None,
    ):
        ctx.op = op
        ctx.graph = graph
        ctx.num_nodes = num_nodes
        if env is not None:
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, edge_attr, weight, grad_out, src, dst, env)
            result = run_envelope_backward(
                op, 1, (x, edge_attr, weight, grad_out, env), graph
            )
            return (pad_node_gradient(result[0], x), *result[1:])
        ctx.save_for_backward(x, edge_attr, weight, grad_out, src, dst)
        dx, dy, dw = run_backward(op, x, edge_attr, weight, grad_out, graph=graph)
        if dx is not None and dx.shape[0] != x.shape[0]:
            dx = pad_node_gradient(dx, x)
        return dx, dy, dw

    @staticmethod
    def backward(ctx, rx, ry, rw, re=None):
        op = ctx.op
        saved = ctx.saved_tensors
        if len(saved) == 7:
            x, edge_attr, weight, grad_out, src, dst, env = saved
            rx, ry, rw, re = prepare_gradients(
                (rx, ry, rw, re), (x, edge_attr, weight, env)
            )
            needs = ctx.needs_input_grad
            if torch.is_grad_enabled():
                gx, gy, gw, gg, ge = UVWConvDoubleBackward.apply(
                    op,
                    x,
                    edge_attr,
                    weight,
                    grad_out,
                    rx,
                    ry,
                    rw,
                    src,
                    dst,
                    ctx.graph,
                    ctx.num_nodes,
                    env,
                    re,
                )
            else:
                gx, gy, gw, gg, ge = run_envelope_backward(
                    op,
                    2,
                    (x, edge_attr, weight, grad_out, env, rx, ry, rw, re),
                    ctx.graph,
                    needs=(*needs[1:5], needs[9]),
                )
            return (
                None,
                pad_node_gradient(gx, x) if needs[1] else None,
                gy if needs[2] else None,
                gw if needs[3] else None,
                gg if needs[4] else None,
                None,
                None,
                None,
                None,
                ge if needs[9] else None,
            )
        x, edge_attr, weight, grad_out, src, dst = saved
        rx = prepare_gradient(rx, x)
        ry = prepare_gradient(ry, edge_attr)
        rw = prepare_gradient(rw, weight)
        if torch.is_grad_enabled():
            grad_x, grad_edge, grad_weight, grad_grad_out = UVWConvDoubleBackward.apply(
                op,
                x,
                edge_attr,
                weight,
                grad_out,
                rx,
                ry,
                rw,
                src,
                dst,
                ctx.graph,
                ctx.num_nodes,
            )
        else:
            grad_x, grad_edge, grad_weight, grad_grad_out = run_double_backward(
                op,
                x,
                edge_attr,
                weight,
                grad_out,
                rx,
                ry,
                rw,
                graph=ctx.graph,
                needs=ctx.needs_input_grad[1:5],
            )
        if grad_x is not None and grad_x.shape[0] != x.shape[0]:
            grad_x = pad_node_gradient(grad_x, x)
        needs = ctx.needs_input_grad
        return (
            None,
            grad_x if needs[1] else None,
            grad_edge if needs[2] else None,
            grad_weight if needs[3] else None,
            grad_grad_out if needs[4] else None,
            None,
            None,
            None,
            None,
        )


class UVWConvForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, edge_attr, weight, src, dst, graph, num_nodes, env=None):
        ctx.op = op
        ctx.graph = graph
        ctx.num_nodes = num_nodes
        if env is not None:
            ctx.save_for_backward(x, edge_attr, weight, src, dst, env)
            ctx.set_materialize_grads(False)
            return run_forward(op, x, edge_attr, weight, graph=graph, env=env)
        ctx.save_for_backward(x, edge_attr, weight, src, dst)
        return run_forward(
            op,
            x,
            edge_attr,
            weight,
            graph=graph,
        )

    @staticmethod
    def backward(ctx, grad_out):
        op = ctx.op
        saved = ctx.saved_tensors
        x, edge_attr, weight, src, dst = saved[:5]
        if grad_out is None:
            grad_out = torch.zeros(
                ctx.num_nodes,
                op.tp.irreps_out.dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            grad_out = grad_out.contiguous()
        needs = ctx.needs_input_grad
        if len(saved) == 6:
            env = saved[5]
            if torch.is_grad_enabled():
                dx, dy, dw, de = UVWConvBackward.apply(
                    op,
                    x,
                    edge_attr,
                    weight,
                    grad_out,
                    src,
                    dst,
                    ctx.graph,
                    ctx.num_nodes,
                    env,
                )
            else:
                dx, dy, dw, de = run_envelope_backward(
                    op,
                    1,
                    (x, edge_attr, weight, grad_out, env),
                    ctx.graph,
                    needs=(*needs[1:4], needs[8]),
                )
            return (
                None,
                pad_node_gradient(dx, x) if needs[1] else None,
                dy if needs[2] else None,
                dw if needs[3] else None,
                None,
                None,
                None,
                None,
                de if needs[8] else None,
            )
        if torch.is_grad_enabled():
            dx, dy, dw = UVWConvBackward.apply(
                op,
                x,
                edge_attr,
                weight,
                grad_out,
                src,
                dst,
                ctx.graph,
                ctx.num_nodes,
            )
        else:
            dx, dy, dw = run_backward(
                op,
                x,
                edge_attr,
                weight,
                grad_out,
                graph=ctx.graph,
                needs=ctx.needs_input_grad[1:4],
            )
        if dx is not None and dx.shape[0] != x.shape[0]:
            dx = pad_node_gradient(dx, x)
        return (
            None,
            dx if needs[1] else None,
            dy if needs[2] else None,
            dw if needs[3] else None,
            None,
            None,
            None,
            None,
        )


def pad_node_gradient(gradient, template):
    if gradient is not None and gradient.shape[0] != template.shape[0]:
        return torch.nn.functional.pad(
            gradient, (0, 0, 0, template.shape[0] - gradient.shape[0])
        )
    return gradient


def lunatp_uvw_conv_func(
    op,
    x,
    edge_attr,
    weight=None,
    *,
    graph: LunaTPGraph,
    env=None,
):
    """Aggregate UVW edge messages with derivatives through third order.

    Args:
        op: Initialized UVW convolution operator.
        x: Node features [input_nodes, in1_dim].
        edge_attr: Edge attributes [num_edges, in2_dim].
        weight: External weights [num_edges, weight_numel] or shared weights
            [weight_numel]. Shared weights are supplied explicitly.
        graph: Shared topology prepared with make_graph.
            x must contain all graph nodes. Edge inputs follow its ordering convention.
        env: Optional edge scalars [num_edges] or [num_edges, 1] multiplying edge messages.

    Returns:
        Aggregated node output of shape [num_nodes, out_dim].

    Features, weights and env share one CUDA device and the operator's
    FP32/FP64 dtype. Shared
    and external weights support all mixed derivatives, including env."""
    if (graph.src_rowptr.dtype == torch.int64 or x.shape[0] > op.index_node_limit
            or edge_attr.shape[0] > op.index_edge_limit):
        op = select_indexing(op, x.shape[0], edge_attr.shape[0])
    if op.index_bits == 64:
        graph = get_graph_with_int64_offsets(graph, cache=not is_compiling())
    graph_tables = graph
    x = x.contiguous()
    edge_attr = edge_attr.contiguous()
    weight = weight.contiguous()
    if env is not None:
        if env.ndim == 1:
            env = env.unsqueeze(-1)
        env = env.contiguous()
    num_nodes = graph.num_nodes
    if is_compiling():
        tables = [
            graph.src, graph.dst, graph.dst_rowptr, graph.src_rowptr,
            graph.edge_ids_by_src, graph.edge_ids_by_dst,
        ]
        result, _ = getattr(torch.ops.lunatp, op.register_ops()).default(
            x, edge_attr, weight, env, tables,
            num_nodes, False,
        )
        return result
    src, dst = graph_tables.src, graph_tables.dst
    if env is not None:
        return UVWConvForward.apply(
            op, x, edge_attr, weight, src, dst, graph_tables, num_nodes, env
        )
    return UVWConvForward.apply(
        op, x, edge_attr, weight, src, dst, graph_tables, num_nodes
    )


__all__ = ["lunatp_uvw_conv_func"]
