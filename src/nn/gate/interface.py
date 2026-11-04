"""Sorted e3nn scalar gates with fused native CUDA gradients."""

from collections import OrderedDict
from functools import lru_cache

import torch

from ..activation.kernels.utils import apply_activation_spec, activation_config
from ...o3.irreps import irreps_key
from ...utils import is_compiling
from ...utils.operator_config import encode_config, decode_config
from .kernels import backward, double_backward, forward, triple_backward
from .kernels.utils import get_kernel_config
from .utils import prepare_gate

__all__ = ['gate']


def gate(features, irreps_scalars=None, act_scalars=None, irreps_gates=None,
         act_gates=None, irreps_gated=None, *, layout=None, kernel_cache=None):
    """Apply scalar gates using e3nn's five Gate configuration arguments."""
    if layout is None:
        layout, cached_kernels = prepare_gate(
            irreps_key(irreps_scalars),
            tuple(map(activation_config, act_scalars)),
            irreps_key(irreps_gates),
            tuple(map(activation_config, act_gates)),
            irreps_key(irreps_gated),
        )
        if kernel_cache is None:
            kernel_cache = cached_kernels
    assert features.shape[-1] == layout["in_dim"], "Gate input feature shape is incorrect"
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        return gate_torch(features, layout)
    if kernel_cache is None:
        kernel_cache = OrderedDict()
    if is_compiling():
        return torch.ops.lunaeq.gate_forward.default(features, gate_config(layout))
    if torch.is_inference_mode_enabled():
        with torch.no_grad():
            return GateFunction.apply(features, layout, 32, kernel_cache)
    return GateFunction.apply(features, layout, 32, kernel_cache)


def gate_torch(features, layout):
    # Use the same normalized paths and extraction order as the CUDA kernels.
    extracted = []
    for width, spans, whole_input in zip(
        layout["extract_dims"], layout["extract_specs"], layout["extract_identity"],
    ):
        values = features.new_zeros((*features.shape[:-1], width))
        if whole_input:
            values.copy_(features)
        else:
            for span in spans:
                values.narrow(-1, span["output_offset"], span["width"]).copy_(
                    features.narrow(-1, span["input_offset"], span["width"])
                )
        extracted.append(values)

    scalars, gates, gated = extracted
    activated = []
    for values, paths in (
        (scalars, layout["scalar_activation"]), (gates, layout["gate_activation"]),
    ):
        outputs = []
        offset = 0
        for width, spec in paths:
            outputs.append(apply_activation_spec(values.narrow(-1, offset, width), spec))
            offset += width
        activated.append(torch.cat(outputs, dim=-1) if len(outputs) > 1 else
                         outputs[0] if outputs else torch.zeros_like(values))
    scalars, gates = activated
    if not gates.shape[-1]:
        return scalars

    vectors = [
        gated[..., section].reshape(-1, entry.mul, entry.ir.dim)
        for entry, section in zip(layout["tp_irreps1"], layout["tp_irreps1"].slices())
    ]
    weights = [
        gates[..., section].reshape(-1, entry.mul)
        for entry, section in zip(layout["tp_irreps2"], layout["tp_irreps2"].slices())
    ]
    outputs = []
    for vector, scalar in zip(vectors, weights):
        product = torch.einsum("cb,cba->cba", scalar, vector)
        outputs.append(product.flatten(1))
    result = torch.cat(outputs, dim=-1).reshape(
        *features.shape[:-1], layout["out_dim"] - scalars.shape[-1]
    )
    return torch.cat((scalars, result), dim=-1)


@torch.compiler.assume_constant_result
def gate_config(meta, index_bits=32):
    metadata = {key: meta[key] for key in (
        "in_dim", "out_dim", "scalar_paths", "gated_paths", "scalar_activation", "has_grad"
    )}
    return encode_config((metadata, index_bits))


@lru_cache(maxsize=128)
def get_gate_operator(config):
    meta, bits = decode_config(config)
    return meta, bits, OrderedDict()


@torch.library.custom_op("lunaeq::gate_forward", mutates_args=())
def gate_forward_op(x: torch.Tensor, config: str) -> torch.Tensor:
    meta, bits, cache = get_gate_operator(config)
    return forward.gate_forward(x, meta, bits, cache)[0]


@gate_forward_op.register_fake
def gate_forward_fake(x, config):
    meta, bits, cache = get_gate_operator(config)
    width = meta["out_dim"]
    return x.new_empty((*x.shape[:-1], width))


def gate_backward_fake_outputs(config, order, inputs, needs=None, grad_mode=False):
    meta, bits, cache = get_gate_operator(config)
    names = (("x", "grad_out"), ("x", "grad_out", "rx"), ("x", "grad_out", "rx", "bx", "bg"))[order - 1]
    available = tuple(name for name, value in zip(names, inputs) if value is not None)
    x = next(value for value in inputs if value is not None)
    program = get_kernel_config(meta, cache, order, dtype=x.dtype, needs=needs, available=available,
                                grad_mode=grad_mode)
    in_dim, out_dim = meta["in_dim"], meta["out_dim"]
    return tuple(
        x.new_empty((*x.shape[:-1], out_dim if name in ("sg", "tg") else in_dim))
        if name in program["active_outputs"] else None
        for name in program["outputs"]
    )


@torch.library.custom_op("lunaeq::gate_backward", mutates_args=())
def gate_backward_op(x: torch.Tensor, g: torch.Tensor, grad_mode: bool, config: str) -> torch.Tensor:
    meta, bits, cache = get_gate_operator(config)
    return backward.gate_backward(x, g, meta, bits, cache, x.dtype, grad_mode=grad_mode)


@gate_backward_op.register_fake
def gate_backward_fake(x, g, grad_mode, config):
    return gate_backward_fake_outputs(config, 1, (x, g), grad_mode=grad_mode)[0]


@torch.library.custom_op(
    "lunaeq::gate_double_backward", mutates_args=(),
    schema="(Tensor x, Tensor g, Tensor rx, bool[] needs, str config) -> (Tensor?, Tensor?)",
)
def gate_double_backward_op(x, g, rx, needs, config):
    meta, bits, cache = get_gate_operator(config)
    return double_backward.gate_double_backward(x, g, rx, meta, bits, cache, x.dtype, needs=needs)


@gate_double_backward_op.register_fake
def gate_double_backward_fake(x, g, rx, needs, config):
    return gate_backward_fake_outputs(config, 2, (x, g, rx), needs)


@torch.library.custom_op(
    "lunaeq::gate_triple_backward", mutates_args=(),
    schema=(
        "(Tensor x, Tensor g, Tensor rx, Tensor? bx, Tensor? bg, bool[] needs, "
        "str config) -> (Tensor?, Tensor?, Tensor?)"
    ),
)
def gate_triple_backward_op(x, g, rx, bx, bg, needs, config):
    meta, bits, cache = get_gate_operator(config)
    return triple_backward.gate_triple_backward(
        x, g, rx, bx, bg, meta, bits, cache, x.dtype, needs=needs,
    )


@gate_triple_backward_op.register_fake
def gate_triple_backward_fake(x, g, rx, bx, bg, needs, config):
    return gate_backward_fake_outputs(config, 3, (x, g, rx, bx, bg), needs)


def gate_forward_setup(ctx, inputs, output):
    x, ctx.config = inputs
    meta, bits, cache = get_gate_operator(ctx.config)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x)
    program = get_kernel_config(meta, cache, 1, dtype=x.dtype, grad_mode=True)
    if not program["active_outputs"]:
        ctx.mark_non_differentiable(output)


def gate_forward_backward(ctx, g):
    (x,) = ctx.saved_tensors
    return (None if g is None else gate_backward_op(x, g, torch.is_grad_enabled(), ctx.config)), None


def gate_backward_setup(ctx, inputs, output):
    ctx.config = inputs[-1]
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:-2])


def gate_backward_backward(ctx, rx):
    if rx is None:
        return None, None, None, None
    x, g = ctx.saved_tensors
    sx, sg = gate_double_backward_op(x, g, rx, list(ctx.needs_input_grad[:2]), ctx.config)
    return sx, sg, None, None


def gate_double_backward_backward(ctx, bx, bg):
    x, g, rx = ctx.saved_tensors
    result = gate_triple_backward_op(x, g, rx, bx, bg, list(ctx.needs_input_grad[:3]), ctx.config)
    return *result, None, None


gate_forward_op.register_autograd(gate_forward_backward, setup_context=gate_forward_setup)
gate_backward_op.register_autograd(gate_backward_backward, setup_context=gate_backward_setup)
gate_double_backward_op.register_autograd(
    gate_double_backward_backward, setup_context=gate_backward_setup,
)


class GateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, meta, index_bits, kernel_cache):
        result, backward_kernel, differentiable, save_input = forward.gate_forward(
            features, meta, index_bits, kernel_cache
        )
        if ctx.needs_input_grad[0]:
            ctx.meta = meta
            ctx.index_bits = index_bits
            ctx.kernel_cache = kernel_cache
            ctx.backward_kernel = backward_kernel
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(features if save_input or features.numel() == 0 else None)
            if not differentiable:
                ctx.mark_non_differentiable(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None

        (features,) = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            grad_input = GateBackward.apply(
                features, grad_output, ctx.meta, ctx.index_bits,
                ctx.kernel_cache, grad_output.dtype,
            )
        elif ctx.backward_kernel is not None:
            grad_input = ctx.backward_kernel(features, grad_output)
        else:
            grad_input = backward.gate_backward(
                features, grad_output, ctx.meta, ctx.index_bits,
                ctx.kernel_cache, grad_output.dtype,
            )
        return grad_input, None, None, None


class GateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, meta, index_bits, kernel_cache, dtype):
        ctx.meta = meta
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = backward.gate_backward(
            features, grad_output, meta, index_bits, kernel_cache, dtype, grad_mode=True
        )

        config = get_kernel_config(
            meta, kernel_cache, 2, dtype=dtype, needs=ctx.needs_input_grad[:2],
            available=("x", "grad_out", "rx"),
        )
        ctx.save_for_backward(
            features if features is not None and ("x" in config["inputs"] or features.numel() == 0) else None,
            grad_output if grad_output is not None and (
                "grad_out" in config["inputs"] or grad_output.numel() == 0
            ) else None,
        )
        if result is not None and not config["active_outputs"]:
            ctx.mark_non_differentiable(result)
        return result

    @staticmethod
    def backward(ctx, grad_grad_input):
        features, grad_output = ctx.saved_tensors
        needs = ctx.needs_input_grad[:2]
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            gradients = GateDoubleBackward.apply(
                features, grad_output, grad_grad_input, ctx.meta, ctx.index_bits,
                ctx.kernel_cache, ctx.dtype, needs,
            )
        else:
            gradients = double_backward.gate_double_backward(
                features, grad_output, grad_grad_input, ctx.meta, ctx.index_bits,
                ctx.kernel_cache, ctx.dtype, needs=needs,
            )
        return *gradients, None, None, None, None


class GateDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, grad_output, grad_grad_input, meta, index_bits, kernel_cache, dtype, needs):
        ctx.meta = meta
        ctx.index_bits = index_bits
        ctx.kernel_cache = kernel_cache
        ctx.dtype = dtype
        ctx.set_materialize_grads(False)

        result = double_backward.gate_double_backward(
            features, grad_output, grad_grad_input, meta, index_bits,
            kernel_cache, dtype, needs=needs,
        )

        required = set()
        for output, grad_name in zip(result, ("bx", "bg")):
            if output is None:
                continue
            config = get_kernel_config(
                meta, kernel_cache, 3, dtype=dtype, needs=ctx.needs_input_grad[:3],
                available=("x", "grad_out", "rx", grad_name),
            )
            required.update(config["inputs"])
            if not config["active_outputs"]:
                ctx.mark_non_differentiable(output)
        ctx.save_for_backward(
            features if features is not None and ("x" in required or features.numel() == 0) else None,
            grad_output if grad_output is not None and ("grad_out" in required or grad_output.numel() == 0) else None,
            grad_grad_input if grad_grad_input is not None and (
                "rx" in required or grad_grad_input.numel() == 0
            ) else None,
        )
        return result

    @staticmethod
    def backward(ctx, grad_input, grad_grad_output):
        features, grad_output, grad_grad_input = ctx.saved_tensors
        gradients = triple_backward.gate_triple_backward(
            features, grad_output, grad_grad_input, grad_input, grad_grad_output,
            ctx.meta, ctx.index_bits, ctx.kernel_cache, ctx.dtype,
            needs=ctx.needs_input_grad[:3],
        )
        return *gradients, None, None, None, None, None
