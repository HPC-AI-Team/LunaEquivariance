"""UVW graph convolution aggregates tensor products with channel mixing.

Shared and per-edge weights use distinct contraction layouts; optional edge
envelopes and all mixed derivatives through third order use those same layouts.
"""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import torch

from ...utils import REGISTER_OPERATOR_LOCK, is_compiling
from ..graph import LunaTPGraph, get_graph_with_int64_offsets
from ..utils.layout import select_indexing
from ..utils.runtime import prepare_gradient
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
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor?[] tables, SymInt num_nodes) -> Tensor")
        def conv_forward(x, y, w, env, tables, num_nodes):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            return run_forward(op, x, y, w, graph=graph, env=env)

        @conv_forward.register_fake
        def conv_forward_fake(x, y, w, env, tables, num_nodes):
            return x.new_empty((num_nodes, op.meta["out_dim"]))

        @torch.library.custom_op(f"lunatp::conv_{key}_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_backward(x, y, w, env, g, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            if env is None:
                dx, dy, dw = run_backward(op, x, y, w, g, graph=graph, needs=needs[:3])
                de = None
            else:
                dx, dy, dw, de = run_envelope_backward(
                    op, 1, (x, y, w, g, env), graph, needs=needs
                )
            return (
                pad_node_gradient(dx, x) if needs[0] else None,
                dy if needs[1] else None,
                dw if needs[2] else None,
                de if needs[3] else None,
            )

        @conv_backward.register_fake
        def conv_backward_fake(x, y, w, env, g, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_double_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? rx, Tensor? ry, Tensor? rw, Tensor? re, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_double_backward(x, y, w, env, g, rx, ry, rw, re, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            if env is None:
                hx, hy, hw, hg = run_double_backward(
                    op, x, y, w, g, rx, ry, rw, graph=graph,
                    needs=(*needs[:3], needs[4]),
                )
                he = None
            else:
                hx, hy, hw, hg, he = run_envelope_backward(
                    op, 2, (x, y, w, g, env, rx, ry, rw, re), graph,
                    needs=(*needs[:3], needs[4], needs[3]),
                )
            return (
                pad_node_gradient(hx, x) if needs[0] else None,
                hy if needs[1] else None,
                hw if needs[2] else None,
                he if needs[3] else None,
                hg if needs[4] else None,
            )

        @conv_double_backward.register_fake
        def conv_double_backward_fake(x, y, w, env, g, rx, ry, rw, re, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env, g), needs))

        @torch.library.custom_op(f"lunatp::conv_{key}_triple_backward", mutates_args=(),
            schema="(Tensor x, Tensor y, Tensor w, Tensor? env, Tensor g, Tensor? rx, Tensor? ry, Tensor? rw, Tensor? re, Tensor? sx, Tensor? sy, Tensor? sw, Tensor? sg, Tensor? se, Tensor?[] tables, SymInt num_nodes, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def conv_triple_backward(x, y, w, env, g, rx, ry, rw, re, sx, sy, sw, sg, se, tables, num_nodes, needs):
            graph = SimpleNamespace(
                src=tables[0], dst=tables[1], dst_rowptr=tables[2],
                src_rowptr=tables[3], edge_ids_by_src=tables[4],
                edge_ids_by_dst=tables[5], num_nodes=num_nodes,
            )
            if env is None:
                dx, dy, dw, dg, drx, dry, drw = run_triple_backward(
                    op, x, y, w, g, rx, ry, rw, sx, sy, sw, sg,
                    graph=graph, needs=(*needs[:3], needs[4], *needs[5:8]),
                )
                de = dre = None
            else:
                dx, dy, dw, dg, de, drx, dry, drw, dre = run_envelope_backward(
                    op, 3, (x, y, w, g, env, rx, ry, rw, re, sx, sy, sw, sg, se), graph,
                    needs=(*needs[:3], needs[4], needs[3], *needs[5:9]),
                )
            return (
                pad_node_gradient(dx, x) if needs[0] else None,
                dy if needs[1] else None,
                dw if needs[2] else None,
                de if needs[3] else None,
                dg if needs[4] else None,
                pad_node_gradient(drx, rx) if needs[5] else None,
                dry if needs[6] else None,
                drw if needs[7] else None,
                dre if needs[8] else None,
            )

        @conv_triple_backward.register_fake
        def conv_triple_backward_fake(x, y, w, env, g, rx, ry, rw, re, sx, sy, sw, sg, se, tables, num_nodes, needs):
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format) if need and value is not None else None
                         for value, need in zip((x, y, w, env, g, rx, ry, rw, re), needs))

        def setup_forward(ctx, inputs, output):
            x, y, w, env, tables, ctx.num_nodes = inputs
            ctx.save_for_backward(x, y, w, env, *tables)
            ctx.table_grad = None if any(table is None for table in tables) else [None] * len(tables)
            ctx.set_materialize_grads(False)

        def setup_backward(ctx, inputs, output):
            ctx.num_nodes = inputs[-2]
            ctx.save_for_backward(*inputs[:-3], *inputs[-3])
            ctx.table_grad = None if any(table is None for table in inputs[-3]) else [None] * len(inputs[-3])
            ctx.set_materialize_grads(False)

        def forward_backward(ctx, g):
            x, y, w, env, *tables = ctx.saved_tensors
            g = x.new_zeros((ctx.num_nodes, op.meta["out_dim"])) if g is None else g.contiguous()
            result = conv_backward(x, y, w, env, g, tables, ctx.num_nodes,
                                   list(ctx.needs_input_grad[:4]))
            return *result, ctx.table_grad, None

        def backward_backward(ctx, rx, ry, rw, re):
            x, y, w, env, g, *tables = ctx.saved_tensors
            rx, ry, rw = (prepare_gradient(value, template)
                          if value is not None or op.weight_mode == "shared" else None
                          for value, template in zip((rx, ry, rw), (x, y, w)))
            re = (prepare_gradient(re, env)
                  if env is not None and (re is not None or op.weight_mode == "shared") else None)
            result = conv_double_backward(x, y, w, env, g, rx, ry, rw, re,
                                          tables, ctx.num_nodes, list(ctx.needs_input_grad[:5]))
            return *result, ctx.table_grad, None, None

        def double_backward_backward(ctx, sx, sy, sw, se, sg):
            x, y, w, env, g, rx, ry, rw, re, *tables = ctx.saved_tensors
            sx, sy, sw, sg = (prepare_gradient(value, template)
                              if value is not None or op.weight_mode == "shared" else None
                              for value, template in zip((sx, sy, sw, sg), (x, y, w, g)))
            se = (prepare_gradient(se, env)
                  if env is not None and (se is not None or op.weight_mode == "shared") else None)
            result = conv_triple_backward(x, y, w, env, g, rx, ry, rw, re,
                                          sx, sy, sw, sg, se, tables, ctx.num_nodes,
                                          list(ctx.needs_input_grad[:9]))
            return *result, ctx.table_grad, None, None

        conv_forward.register_autograd(forward_backward, setup_context=setup_forward)
        conv_backward.register_autograd(backward_backward, setup_context=setup_backward)
        conv_double_backward.register_autograd(double_backward_backward, setup_context=setup_backward)
        return name


class UVWConvDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, edge_attr, weight, grad_out, rx, ry, rw, graph, env, re):
        ctx.op, ctx.graph = op, graph
        ctx.save_for_backward(x, edge_attr, weight, grad_out, rx, ry, rw, env, re)
        if env is not None or op.weight_mode != "shared":
            ctx.set_materialize_grads(False)
        if env is not None:
            result = run_envelope_backward(
                op, 2, (x, edge_attr, weight, grad_out, env, rx, ry, rw, re), graph,
                needs=(*ctx.needs_input_grad[1:5], ctx.needs_input_grad[9]),
            )
        else:
            result = run_double_backward(
                op, x, edge_attr, weight, grad_out, rx, ry, rw, graph=graph,
                needs=ctx.needs_input_grad[1:5],
            )
        return pad_node_gradient(result[0], x), *result[1:]

    @staticmethod
    def backward(ctx, bx, by, bw, bg, be=None):
        op, graph = ctx.op, ctx.graph
        x, edge_attr, weight, grad_out, rx, ry, rw, env, re = ctx.saved_tensors
        bx, by, bw, bg = (
            prepare_gradient(value, template)
            if value is not None or op.weight_mode == "shared" else None
            for value, template in zip((bx, by, bw, bg), (x, edge_attr, weight, grad_out))
        )
        needs = ctx.needs_input_grad
        if env is not None:
            be = prepare_gradient(be, env) if be is not None or op.weight_mode == "shared" else None
            gx, gy, gw, gg, ge, grx, gry, grw, gre = run_envelope_backward(
                op, 3, (x, edge_attr, weight, grad_out, env, rx, ry, rw, re, bx, by, bw, bg, be),
                graph, needs=(*needs[1:5], needs[9], *needs[5:8], needs[10]),
            )
        else:
            gx, gy, gw, gg, grx, gry, grw = run_triple_backward(
                op, x, edge_attr, weight, grad_out, rx, ry, rw, bx, by, bw, bg,
                graph=graph, needs=needs[1:8],
            )
            ge = gre = None
        return (
            None,
            pad_node_gradient(gx, x) if needs[1] else None,
            gy if needs[2] else None,
            gw if needs[3] else None,
            gg if needs[4] else None,
            pad_node_gradient(grx, rx) if needs[5] else None,
            gry if needs[6] else None,
            grw if needs[7] else None,
            None,
            ge if needs[9] else None,
            gre if needs[10] else None,
        )


class UVWConvBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, edge_attr, weight, grad_out, graph, env):
        ctx.op, ctx.graph = op, graph
        ctx.save_for_backward(x, edge_attr, weight, grad_out, env)
        if env is not None or op.weight_mode != "shared":
            ctx.set_materialize_grads(False)
        if env is not None:
            result = run_envelope_backward(
                op, 1, (x, edge_attr, weight, grad_out, env), graph,
                needs=(*ctx.needs_input_grad[1:4], ctx.needs_input_grad[6]),
            )
        else:
            result = run_backward(
                op, x, edge_attr, weight, grad_out, graph=graph,
                needs=ctx.needs_input_grad[1:4],
            )
        return pad_node_gradient(result[0], x), *result[1:]

    @staticmethod
    def backward(ctx, rx, ry, rw, re=None):
        op, graph = ctx.op, ctx.graph
        x, edge_attr, weight, grad_out, env = ctx.saved_tensors
        rx, ry, rw = (
            prepare_gradient(value, template)
            if value is not None or op.weight_mode == "shared" else None
            for value, template in zip((rx, ry, rw), (x, edge_attr, weight))
        )
        if env is not None:
            re = prepare_gradient(re, env) if re is not None or op.weight_mode == "shared" else None
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            result = UVWConvDoubleBackward.apply(
                op, x, edge_attr, weight, grad_out, rx, ry, rw, graph, env, re
            )
        elif env is not None:
            result = run_envelope_backward(
                op, 2, (x, edge_attr, weight, grad_out, env, rx, ry, rw, re), graph,
                needs=(*needs[1:5], needs[6]),
            )
        else:
            result = run_double_backward(
                op, x, edge_attr, weight, grad_out, rx, ry, rw,
                graph=graph, needs=needs[1:5],
            )
        gx, gy, gw, gg = result[:4]
        ge = result[4] if env is not None else None
        return (
            None,
            pad_node_gradient(gx, x) if needs[1] else None,
            gy if needs[2] else None,
            gw if needs[3] else None,
            gg if needs[4] else None,
            None,
            ge if needs[6] else None,
        )


class UVWConvForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, edge_attr, weight, graph, env):
        ctx.op, ctx.graph = op, graph
        ctx.save_for_backward(x, edge_attr, weight, env)
        if env is not None:
            ctx.set_materialize_grads(False)
        return run_forward(op, x, edge_attr, weight, graph=graph, env=env)

    @staticmethod
    def backward(ctx, grad_out):
        op, graph = ctx.op, ctx.graph
        x, edge_attr, weight, env = ctx.saved_tensors
        grad_out = (x.new_zeros((graph.num_nodes, op.meta["out_dim"]))
                    if grad_out is None else grad_out.contiguous())
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            result = UVWConvBackward.apply(op, x, edge_attr, weight, grad_out, graph, env)
        elif env is not None:
            result = run_envelope_backward(
                op, 1, (x, edge_attr, weight, grad_out, env), graph,
                needs=(*needs[1:4], needs[5]),
            )
        else:
            result = run_backward(
                op, x, edge_attr, weight, grad_out, graph=graph, needs=needs[1:4],
            )
        dx, dy, dw = result[:3]
        de = result[3] if env is not None else None
        return (
            None,
            pad_node_gradient(dx, x) if needs[1] else None,
            dy if needs[2] else None,
            dw if needs[3] else None,
            None,
            de if needs[5] else None,
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
    FP32/FP64 dtype. Shared and external weights support all mixed derivatives,
    including env.
    """
    if (graph.src_rowptr.dtype == torch.int64 or x.shape[0] > op.index_node_limit
            or edge_attr.shape[0] > op.index_edge_limit):
        op = select_indexing(op, x.shape[0], edge_attr.shape[0])
    if op.index_bits == 64:
        graph = get_graph_with_int64_offsets(graph, cache=not is_compiling())
    x = x.contiguous()
    edge_attr = edge_attr.contiguous()
    weight = weight.contiguous()
    if env is not None:
        if env.ndim == 1:
            env = env.unsqueeze(-1)
        env = env.contiguous()
    if is_compiling():
        tables = [
            graph.src, graph.dst, graph.dst_rowptr, graph.src_rowptr,
            graph.edge_ids_by_src, graph.edge_ids_by_dst,
        ]
        return getattr(torch.ops.lunatp, op.register_ops()).default(
            x, edge_attr, weight, env, tables, graph.num_nodes,
        )
    return UVWConvForward.apply(op, x, edge_attr, weight, graph, env)


__all__ = ["lunatp_uvw_conv_func"]
