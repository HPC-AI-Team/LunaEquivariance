"""Linear functional entry, layout dispatch, and autograd.

metadata.py owns configuration; kernels/ owns CUDA specialization and execution.
The PyTorch Module owns parameters; this interface owns saved operands.
"""

from copy import copy
from hashlib import sha256
from types import SimpleNamespace
from math import prod

import torch

from .kernels.utils import MAX_BATCH_RANK, TASK_ALIGNMENT, build_program, run_kernel, run_raw_forward, select_program
from .metadata import linear_config, LinearKernelConfig, Instruction
from ...utils import is_compiling, REGISTER_OPERATOR_LOCK
from ..irreps import Irreps, irreps_key


MISSING = object()


class LinearDoubleBackward(torch.autograd.Function):
    """Double-backward node saving only operands required by triple-backward."""

    @staticmethod
    def forward(ctx, op, x, weight, g, rx, rw, rb, num_items, needs):
        ctx.op = op
        ctx.num_items = num_items
        ctx.set_materialize_grads(False)

        # A derivative may be needed even when its original tensor value
        # was not needed or saved by the outer backward call.
        nx, nw, ng = needs
        sx, sw, _, sg = run_kernel(op,
            2, x, weight, None, g, rx, rw, rb, needs=(nx, nw, False, ng), num_items=num_items
        )

        nx, nw, ng, nrx, nrw, nrb = ctx.needs_input_grad[1:7]
        live = [
            role
            for role, value in zip(
                ("x", "weight", "grad_out", "rx", "rw", "rb"), (x, weight, g, rx, rw, rb)
            )
            if value is not None
        ]
        live += [role for role, value in zip(("bx", "bw", "bg"), (sx, sw, sg)) if value is not None]
        required = select_program(op, 3, available=live, needs=(nx, nw, False, ng, nrx, nrw, nrb))[
            "inputs"
        ]
        ctx.save_for_backward(
            x if "x" in required else None,
            weight if "weight" in required else None,
            g if "grad_out" in required else None,
            rx if "rx" in required else None,
            rw if "rw" in required else None,
        )
        return sx, sw, sg

    @staticmethod
    def backward(ctx, bx, bw, bg):
        x, weight, g, rx, rw = ctx.saved_tensors
        nx, nw, ng, nrx, nrw, nrb = ctx.needs_input_grad[1:7]
        result = run_kernel(ctx.op,
            3,
            x,
            weight,
            None,
            g,
            rx,
            rw,
            None,
            bx,
            bw,
            None,
            bg,
            needs=(nx, nw, False, ng, nrx, nrw, nrb),
            num_items=ctx.num_items,
        )
        tx, tw, _, tg, trx, trw, trb = result
        return None, tx, tw, tg, trx, trw, trb, None, None


class LinearBackward(torch.autograd.Function):
    """Backward node saving original operands required by double-backward."""

    @staticmethod
    def forward(ctx, op, x, weight, g, needs):
        ctx.op = op
        ctx.num_items = g.shape[0]
        ctx.set_materialize_grads(False)
        nx, nw, ng = ctx.needs_input_grad[1:4]
        live = [
            role
            for role, value in zip(("x", "weight", "grad_out"), (x, weight, g))
            if value is not None
        ]
        live += [role for role, wanted in zip(("rx", "rw", "rb"), needs) if wanted]
        required = select_program(op, 2, available=live, needs=(nx, nw, False, ng))["inputs"]
        ctx.save_for_backward(
            x if "x" in required else None,
            weight if "weight" in required else None,
            g if "grad_out" in required else None,
        )
        return run_kernel(op, 1, x, weight, None, g, needs=needs)

    @staticmethod
    def backward(ctx, rx, rw, rb):
        x, weight, g = ctx.saved_tensors
        nx, nw, ng = ctx.needs_input_grad[1:4]
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            sx, sw, sg = LinearDoubleBackward.apply(
                ctx.op, x, weight, g, rx, rw, rb, ctx.num_items, (nx, nw, ng)
            )
        else:
            sx, sw, _, sg = run_kernel(ctx.op,
                2,
                x,
                weight,
                None,
                g,
                rx,
                rw,
                rb,
                needs=(nx, nw, False, ng),
                num_items=ctx.num_items,
            )
        return None, sx, sw, sg, None


class LinearForward(torch.autograd.Function):
    """Save X for DW and weight for DX; restore raw-view gradient shapes."""

    @staticmethod
    def forward(ctx, op, x, weight, bias, plan):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.needs = ctx.needs_input_grad[1:4]
        nx, nw, _ = ctx.needs
        ctx.save_for_backward(x if nw else None, weight if nx else None)
        ctx.raw_shapes = None
        if plan is None:
            return run_kernel(op, 0, x, weight, bias, backward_needs=ctx.needs if any(ctx.needs) else None)

        ctx.num_items = plan["n"]
        ctx.raw_shapes = {
            role: tuple(value.shape) for role, value in plan["roles"].items() if value is not None
        }
        return run_raw_forward(op, plan, backward_needs=ctx.needs if any(ctx.needs) else None)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return (None,) * 5
        x, weight = ctx.saved_tensors
        if ctx.raw_shapes is not None:
            n = ctx.num_items
            x = x.reshape(n, ctx.op.meta["in_dim"]) if x is not None else None
            if weight is not None:
                shape = (
                    (ctx.op.meta["weight_numel"],)
                    if ctx.op.shared_weights
                    else (n, ctx.op.meta["weight_numel"])
                )
                weight = weight.reshape(shape)

        gradients = (
            LinearBackward.apply(ctx.op, x, weight, g, ctx.needs)
            if torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            else run_kernel(ctx.op, 1, x, weight, None, g, needs=ctx.needs)
        )
        if ctx.raw_shapes is not None:
            gradients = tuple(
                gradient.reshape(ctx.raw_shapes[role]) if gradient is not None else None
                for role, gradient in zip(("x", "weight", "bias"), gradients)
            )
        return (None, *gradients, None)


def linear(features, weight=None, bias=None, *, irreps_in=None, irreps_out=None,
           f_in=None, f_out=None, shared_weights=True, instructions=None, biases=False,
           path_normalization="element", dtype=None, config=None):
    """Equivariant linear transform with explicit parameters and optional prepared config.

    Supply irreps/options, or reuse ``module.config`` without retaining
    the Module. The bounded cache holds only static metadata and kernel records.
    """
    op = config
    if op is None:
        if irreps_in is None or irreps_out is None:
            raise TypeError("linear requires irreps_in and irreps_out, or config")
        op = linear_config(
            irreps_key(irreps_in), irreps_key(irreps_out),
            f_in, f_out, shared_weights,
            None if instructions is None else tuple(tuple(ins) for ins in instructions),
            tuple(biases) if isinstance(biases, (list, tuple)) else biases,
            path_normalization, dtype, 32,
        )
    compiling = is_compiling()
    x = features
    if weight is None and op.weight_numel:
        raise RuntimeError("Weights must be provided")
    if bias is None and op.bias_numel:
        raise RuntimeError("Biases must be provided")
    raw_weight, raw_bias = weight, bias
    active = ([x, raw_weight] if op.meta["generated_specs"] else []) + (
        [raw_bias] if op.meta["bias_specs"] else []
    )
    if not op._explicit_dtype:
        dtype = (
            raw_weight.dtype
            if op.meta["generated_specs"]
            else raw_bias.dtype if op.meta["bias_specs"] else x.dtype
        )
    else:
        dtype = op.compute_dtype
    op.last_execution_dtype = dtype

    # CPU, unsupported precision and mixed dtypes retain Torch behavior.
    # Native compilation and launch errors propagate to the caller.
    if (
        not x.is_cuda
        or dtype not in (torch.float32, torch.float64)
        or not op._cuda_dimensions_supported
        or any(value.dtype != active[0].dtype for value in active[1:])
    ):
        op.last_execution_backend = "torch_compat"
        result = torch_linear(features, weight, bias, config=op)
        op.last_execution_dtype = result.dtype
        return result

    op.last_execution_backend = "cuda_fused"
    if not op.meta["generated_specs"] and not op.meta["bias_specs"]:
        rank = 1 if op.f_in is None else 2
        output_shape = (op.output_dim,) if op.f_out is None else (op.f_out, op.output_dim)
        op.last_execution_backend = "cuda_native"
        op.last_execution_dtype = x.dtype
        return x.new_zeros(tuple(x.shape[:-rank]) + output_shape)

    differentiable = (
        torch.is_grad_enabled()
        and (compiling or not torch.is_inference_mode_enabled())
        and any(value.requires_grad for value in active)
    )
    # Only an outstanding graph or a different computation dtype needs a snapshot.
    # Inference uses the owner directly; saved parameter views still share storage.
    execution_op = copy(op) if not compiling and (differentiable or dtype != op.compute_dtype) else op
    if not compiling and dtype is not None and dtype != op.compute_dtype:
        execution_op.compute_dtype = dtype
        execution_op.meta = {**op.meta, "dtype": dtype}

    original = (x, raw_weight, raw_bias)
    weight, bias = raw_weight, raw_bias
    feature_shape = (
        (execution_op.input_dim,) if execution_op.f_in is None else (execution_op.f_in, execution_op.input_dim)
    )
    batch = tuple(x.shape[: -len(feature_shape)])
    if x.shape[-len(feature_shape) :] != feature_shape:
        x = x.reshape(batch + feature_shape)

    wshape = (() if execution_op.f_in is None else (execution_op.f_in, execution_op.f_out)) + (execution_op.weight_numel,)
    if execution_op.meta["weight_numel"]:
        if execution_op.shared_weights:
            if weight.shape != wshape:
                weight = weight.reshape(wshape)
        else:
            n = prod(batch)
            count = weight.numel() // execution_op.meta["weight_numel"]
            remainder = weight.numel() % execution_op.meta["weight_numel"]
            if remainder:
                raise RuntimeError(
                    "Linear external weight size cannot be reshaped to its feature axes"
                )
            if count == n:
                if weight.shape != batch + wshape:
                    weight = weight.reshape(batch + wshape)
            elif count == 1:
                # e3nn broadcasts its flattened weight batch of one.
                # Preserve an existing expanded/raw view when possible.
                if weight.shape[-len(wshape) :] != wshape:
                    weight = weight.reshape(wshape)
                elif len(weight.shape) - len(wshape) > len(batch):
                    weight = weight.reshape((1,) * len(batch) + wshape)
            elif weight.shape[-len(wshape) :] != wshape:
                raise RuntimeError(
                    "Linear external weight batch cannot broadcast to the feature batch"
                )
            batch = tuple(torch.broadcast_shapes(batch, weight.shape[: -len(wshape)]))
            if weight.shape != batch + wshape:
                weight = weight.expand(batch + wshape)
    else:
        weight = None

    bshape = (() if execution_op.f_out is None else (execution_op.f_out,)) + (execution_op.bias_numel,)
    if execution_op.meta["bias_numel"]:
        if bias.shape != bshape:
            bias = bias.reshape(bshape)
    else:
        bias = None

    # Ignore structurally unused operands just as e3nn's generated graph
    # does; their dtype/device and requires_grad cannot affect its result.
    if not execution_op.meta["generated_specs"]:
        x = x.detach()
        weight = None
    if not execution_op.meta["bias_specs"]:
        bias = None
    if execution_op._explicit_dtype and active[0].dtype != execution_op.compute_dtype:
        raise RuntimeError(f"Linear inputs must use {execution_op.compute_dtype}")

    # _apply may replace a Parameter's data while this forward still has
    # a live graph. A differentiable view keeps the saved storage/dtype
    # and preserves its link to that Parameter, as e3nn codegen does.
    if isinstance(weight, torch.nn.Parameter):
        weight = weight.view_as(weight)
    if isinstance(bias, torch.nn.Parameter):
        bias = bias.view_as(bias)
    if x.shape != batch + feature_shape:
        x = x.expand(batch + feature_shape)
    elif isinstance(x, torch.nn.Parameter):
        x = x.view_as(x)
    if compiling:
        name = register_ops(op.layout_key, dtype) if config is None else op.register_ops(dtype)
        return getattr(torch.ops.lunaeq, name).default(x, weight, bias)
    if any(
        after is not None and after.data_ptr() != before.data_ptr()
        for before, after in zip(original, (x, weight, bias))
    ):
        op.last_execution_backend = "cuda_native"

    return run_linear(execution_op, x, weight, bias, differentiable=differentiable)


def run_linear(op, x, weight, bias, *, differentiable=False):
    """Launch normalized native inputs, including strided and broadcast views."""
    feature_rank = 1 if op.f_in is None else 2
    weight_rank = 1 if op.f_in is None else 3
    contiguous = x.is_contiguous() and all(
        value is None or value.is_contiguous() for value in (weight, bias)
    )
    needs_raw = not contiguous or not op.meta["out_dim"]
    if not op.shared_weights and weight is not None and op.meta["weight_numel"]:
        needs_raw = needs_raw or tuple(x.shape[:-feature_rank]) != tuple(
            weight.shape[:-weight_rank]
        )
    if needs_raw:
        feature_shape = (op.input_dim,) if op.f_in is None else (op.f_in, op.input_dim)
        batch = tuple(x.shape[: -len(feature_shape)])
        wshape = (() if op.f_in is None else (op.f_in, op.f_out)) + (op.weight_numel,)
        bshape = (() if op.f_out is None else (op.f_out,)) + (op.bias_numel,)
        n = prod(batch)
        if n > (1 << 63) - 1:
            raise OverflowError("Linear item count exceeds signed int64 capacity")

        roles = {"x": x, "weight": weight, "bias": bias}
        x_batch = x.stride()[: -len(feature_shape)]
        w_batch = (
            (0,) * len(batch)
            if op.shared_weights or weight is None
            else weight.stride()[: -len(wshape)]
        )
        axes = []
        if n:
            for size, sx, sw in zip(batch, x_batch, w_batch):
                if size == 1:
                    continue
                if axes and all(left == size * right for left, right in zip(axes[-1][1:], (sx, sw))):
                    previous, _, _ = axes.pop()
                    axes.append((previous * size, sx, sw))
                else:
                    axes.append((size, sx, sw))
            if len(axes) > MAX_BATCH_RANK:
                raise OverflowError("Linear raw-view batch rank exceeds signed-int64 capacity")

        xf = (0,) * (2 - len(feature_shape)) + tuple(x.stride()[-len(feature_shape):])
        wf = (
            (0, 0, 0) if weight is None
            else (0,) * (3 - len(wshape)) + tuple(weight.stride()[-len(wshape):])
        )
        bf = (
            (0, 0) if bias is None
            else (0,) * (2 - len(bshape)) + tuple(bias.stride()[-len(bshape):])
        )
        descriptor = (
            len(axes),
            *(axis[0] for axis in axes),
            *(axis[1] for axis in axes),
            *(axis[2] for axis in axes),
            *xf,
            *wf,
            *bf,
        )

        spans = {
            role: (
                sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
                if value.numel()
                else 0
            )
            for role, value in roles.items()
            if value is not None
        }
        if any(span < 0 or span > (1 << 63) - 1 for span in spans.values()):
            raise OverflowError("Linear raw-view addresses exceed signed int64 capacity")
        logical_bound = (
            n * op.index_batch_span + op.meta["bias_numel"] + 3 * TASK_ALIGNMENT + 65535 * 256
        )
        bound = max(logical_bound, *spans.values(), *descriptor)
        if bound > (1 << 63) - 1:
            raise OverflowError("Linear raw-view launch indices exceed signed int64 capacity")
        bits = 64 if bound > (1 << 31) - 1 else op.index_bits
        output_shape = (op.output_dim,) if op.f_out is None else (op.f_out, op.output_dim)
        plan = dict(
            roles=roles,
            n=n,
            descriptor=descriptor,
            index_bits=bits,
            output_shape=batch + output_shape,
            contiguous=contiguous,
        )

        if not plan["contiguous"] or not op.meta["out_dim"]:
            roles = plan["roles"]
            if differentiable:
                result = LinearForward.apply(
                    op, roles["x"], roles["weight"], roles["bias"], plan
                )
            else:
                result = run_raw_forward(op, plan)
            return result.reshape(plan["output_shape"])

    batch = tuple(x.shape[:-feature_rank])
    if weight is not None:
        shape = (
            (op.meta["weight_numel"],)
            if op.shared_weights
            else (prod(batch), op.meta["weight_numel"])
        )
        weight = weight.reshape(shape)
    if bias is not None:
        bias = bias.reshape(op.meta["bias_numel"])
    x = x.reshape(prod(batch), op.meta["in_dim"])

    result = (
        LinearForward.apply(op, x, weight, bias, None)
        if differentiable
        else run_kernel(op, 0, x, weight, bias)
    )
    output_shape = (op.output_dim,) if op.f_out is None else (op.f_out, op.output_dim)
    return result.reshape(batch + output_shape)


__all__ = ['linear']


def torch_linear(features, weight, bias, *, config):
    """Existing CPU/low-precision path expressed directly with Torch tensors."""
    op = config
    feature_axes = () if op.f_in is None else (op.f_in,)
    output_axes = () if op.f_out is None else (op.f_out,)
    batch = features.shape[:-(len(feature_axes) + 1)]
    output_shape = batch + output_axes + (op.output_dim,)
    instructions = [ins for ins in op.instructions if 0 not in ins.path_shape]
    if not instructions and not op.bias_numel:
        return features.new_zeros(output_shape)
    x = features.reshape(-1, *feature_axes, op.input_dim)
    n = x.shape[0]
    inputs = [
        x.narrow(-1, part.start, ir.dim).reshape(n, *feature_axes, ir.mul, ir.ir.dim)
        for part, ir in zip(op.irreps_in.slices(), op.irreps_in)
    ]
    if op.weight_numel:
        weight_axes = () if op.f_in is None else (op.f_in, op.f_out)
        weight = weight.reshape(-1, *weight_axes, op.weight_numel)
    if op.bias_numel:
        bias = bias.reshape(-1, *output_axes, op.bias_numel)
    pieces = [[] for _ in op.irreps_out]
    wi = bi = 0
    z = '' if op.shared_weights else 'z'
    for ins in instructions:
        count = prod(ins.path_shape)
        if ins.i_out < 0:
            wi += count
            continue
        if ins.i_in == -1:
            value = (ins.path_weight * bias.narrow(-1, bi, count)).reshape(
                1, *output_axes, op.irreps_out[ins.i_out].dim
            )
            bi += count
        else:
            weights = weight.narrow(-1, wi, count).reshape(
                (() if op.shared_weights else (-1,)) + weight_axes + ins.path_shape
            )
            wi += count
            operand = inputs[ins.i_in]
            if op.optimize_einsums:
                # e3nn's two-operand optimization scales the smaller operand,
                # using its construction-time example batch of four.
                cin, cout = ins.path_shape
                x_cost = 4 * (op.f_in or 1) * cin * op.irreps_in[ins.i_in].ir.dim
                w_cost = (1 if op.shared_weights else 4) * (op.f_in or 1) * (op.f_out or 1) * cin * cout
                if ins.path_weight != 1:
                    if x_cost <= w_cost:
                        operand = operand * ins.path_weight
                    else:
                        weights = weights * ins.path_weight
                if op.shared_weights:
                    dims = ((1,), (0,)) if op.f_in is None else ((1, 2), (0, 2))
                    axes = (0, 2, 1) if op.f_in is None else (0, 2, 3, 1)
                    value = torch.tensordot(operand, weights, dims=dims).permute(axes)
                else:
                    expression = 'zui,zuw->zwi' if op.f_in is None else 'zxui,zxyuw->zywi'
                    value = torch.einsum(expression, operand, weights)
            else:
                expression = f'{z}uw,zui->zwi' if op.f_in is None else f'{z}xyuw,zxui->zywi'
                value = ins.path_weight * torch.einsum(expression, weights, operand)
            value = value.reshape(n, *output_axes, op.irreps_out[ins.i_out].dim)
        pieces[ins.i_out].append(value)
    outputs = []
    for ir, terms in zip(op.irreps_out, pieces):
        if not ir.mul:
            continue
        value = sum(terms[1:], terms[0]) if terms else x.new_zeros((n, *output_axes, ir.dim))
        outputs.append(value)
    return (torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]).reshape(output_shape)


@torch.compiler.assume_constant_result
def register_ops(layout_key, dtype):
    """Register a tensor-only boundary once for each immutable Linear layout."""
    with REGISTER_OPERATOR_LOCK:
        name = sha256(repr((layout_key, dtype)).encode()).hexdigest()[:24]
        if hasattr(torch.ops.lunaeq, f"linear_{name}_forward"):
            return f"linear_{name}_forward"
        in_irreps, out_irreps, instructions, weights, biases, shared, optimize, fin, fout, bits = layout_key
        layout = SimpleNamespace(
            irreps_in=Irreps(in_irreps), irreps_out=Irreps(out_irreps),
            instructions=tuple(Instruction(*item) for item in instructions),
            weight_numel=weights, bias_numel=biases,
            shared_weights=shared, _optimize_einsums=optimize,
        )
        op = LinearKernelConfig(layout, f_in=fin, f_out=fout, dtype=dtype, index_bits=bits)

        @torch.library.custom_op(f"lunaeq::linear_{name}_forward", mutates_args=())
        def linear_forward(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
            return run_linear(op, x, weight, bias)

        @linear_forward.register_fake
        def linear_forward_fake(x, weight, bias):
            feature_rank = 1 if op.f_in is None else 2
            output_shape = (op.output_dim,) if op.f_out is None else (op.f_out, op.output_dim)
            return x.new_empty(x.shape[:-feature_rank] + output_shape, dtype=dtype)

        @torch.library.custom_op(f"lunaeq::linear_{name}_backward", mutates_args=(), schema=
            "(Tensor x, Tensor? weight, Tensor? bias, Tensor g, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def linear_backward(x, weight, bias, g, needs):
            n = prod(g.shape[:-(1 if op.f_out is None else 2)])
            x_flat = x.reshape(n, op.meta["in_dim"])
            w_flat = None if weight is None else weight.reshape((-1,) if op.shared_weights else (n, -1))
            gradients = run_kernel(op, 1, x_flat, w_flat, None, g.reshape(n, op.meta["out_dim"]), needs=needs)
            return tuple(value.reshape_as(original) if value is not None else None
                         for value, original in zip(gradients, (x, weight, bias)))

        @linear_backward.register_fake
        def linear_backward_fake(x, weight, bias, g, needs):
            available = tuple(role for role, value in zip(("x", "weight", "bias", "grad_out"), (x, weight, None, g)) if value is not None)
            program = build_program(1, needs=needs, available=available)
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                         if output in program["active_outputs"] else None
                         for value, output in zip((x, weight, bias), program["outputs"]))

        @torch.library.custom_op(f"lunaeq::linear_{name}_double_backward", mutates_args=(), schema=
            "(Tensor x, Tensor? weight, Tensor? bias, Tensor g, Tensor? rx, Tensor? rw, Tensor? rb, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def linear_double_backward(x, weight, bias, g, rx, rw, rb, needs):
            n = prod(g.shape[:-(1 if op.f_out is None else 2)])
            xs, ws, gs = (n, op.meta["in_dim"]), ((-1,) if op.shared_weights else (n, -1)), (n, op.meta["out_dim"])
            gradients = run_kernel(
                op, 2, x.reshape(xs), None if weight is None else weight.reshape(ws), None, g.reshape(gs),
                None if rx is None else rx.reshape(xs), None if rw is None else rw.reshape(ws),
                None if rb is None else rb.reshape(-1), needs=needs, num_items=n,
            )
            return tuple(value.reshape_as(original) if value is not None else None
                         for value, original in zip(gradients, (x, weight, bias, g)))

        @linear_double_backward.register_fake
        def linear_double_backward_fake(x, weight, bias, g, rx, rw, rb, needs):
            available = tuple(role for role, value in zip(("x", "weight", "bias", "grad_out", "rx", "rw", "rb"), (x, weight, None, g, rx, rw, rb)) if value is not None)
            program = build_program(2, needs=needs, available=available)
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                         if output in program["active_outputs"] else None
                         for value, output in zip((x, weight, bias, g), program["outputs"]))

        @torch.library.custom_op(f"lunaeq::linear_{name}_triple_backward", mutates_args=(), schema=
            "(Tensor x, Tensor? weight, Tensor? bias, Tensor g, Tensor? rx, Tensor? rw, Tensor? rb, Tensor? bx, Tensor? bw, Tensor? bb, Tensor? bg, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def linear_triple_backward(x, weight, bias, g, rx, rw, rb, bx, bw, bb, bg, needs):
            n = prod(g.shape[:-(1 if op.f_out is None else 2)])
            xs, ws, gs = (n, op.meta["in_dim"]), ((-1,) if op.shared_weights else (n, -1)), (n, op.meta["out_dim"])
            gradients = run_kernel(
                op, 3, x.reshape(xs), None if weight is None else weight.reshape(ws), None, g.reshape(gs),
                None if rx is None else rx.reshape(xs), None if rw is None else rw.reshape(ws),
                None if rb is None else rb.reshape(-1), None if bx is None else bx.reshape(xs),
                None if bw is None else bw.reshape(ws), None, None if bg is None else bg.reshape(gs),
                needs=needs, num_items=n,
            )
            return tuple(value.reshape_as(original) if value is not None else None
                         for value, original in zip(gradients, (x, weight, bias, g, rx, rw, rb)))

        @linear_triple_backward.register_fake
        def linear_triple_backward_fake(x, weight, bias, g, rx, rw, rb, bx, bw, bb, bg, needs):
            available = tuple(role for role, value in zip(("x", "weight", "bias", "grad_out", "rx", "rw", "rb", "bx", "bw", "bb", "bg"), (x, weight, None, g, rx, rw, rb, bx, bw, None, bg)) if value is not None)
            program = build_program(3, needs=needs, available=available)
            return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                         if output in program["active_outputs"] else None
                         for value, output in zip((x, weight, bias, g, rx, rw, rb), program["outputs"]))

        def linear_setup_context(ctx, inputs, output):
            ctx.save_for_backward(*(inputs if len(inputs) == 3 else inputs[:-1]))
            ctx.set_materialize_grads(False)

        def linear_forward_backward(ctx, g):
            x, weight, bias = ctx.saved_tensors
            gradients = linear_backward(x, weight, bias, g, list(ctx.needs_input_grad[:3]))
            return gradients

        def linear_backward_backward(ctx, rx, rw, rb):
            x, weight, bias, g = ctx.saved_tensors
            needs = list(ctx.needs_input_grad[:4])
            needs[2] = False
            gradients = linear_double_backward(x, weight, bias, g, rx, rw, rb, needs)
            return *gradients, None

        def linear_double_backward_backward(ctx, bx, bw, bb, bg):
            x, weight, bias, g, rx, rw, rb = ctx.saved_tensors
            needs = list(ctx.needs_input_grad[:7])
            needs[2] = False
            gradients = linear_triple_backward(x, weight, bias, g, rx, rw, rb, bx, bw, bb, bg,
                                              needs)
            return *gradients, None

        # The CUDA triple backward is terminal for AOT autograd.
        op.triple_backward_library = torch.library.Library("lunaeq", "FRAGMENT")
        for dispatch_key in ("AutogradCPU", "AutogradCUDA"):
            op.triple_backward_library.impl(
                f"linear_{name}_triple_backward", torch.library.fallthrough_kernel, dispatch_key
            )

        linear_forward.register_autograd(linear_forward_backward, setup_context=linear_setup_context)
        linear_backward.register_autograd(linear_backward_backward, setup_context=linear_setup_context)
        linear_double_backward.register_autograd(linear_double_backward_backward, setup_context=linear_setup_context)

        return f"linear_{name}_forward"
