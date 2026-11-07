"""Symmetric contraction and its CUDA autograd registrations."""

from functools import lru_cache
import json

import torch
from e3nn import o3

from .kernels.utils import build_metadata, load_kernels

__all__ = ['symmetric_contraction']


@lru_cache(maxsize=128)
def kernel_metadata(config):
    spec = json.loads(config)
    return build_metadata(
        spec["irreps_in"],
        spec["irreps_out"],
        spec["correlation"],
        spec["num_elements"],
        getattr(torch, spec["source_dtype"]),
        use_reduced_cg=spec["use_reduced_cg"],
    )


@lru_cache(maxsize=128)
def get_kernels(config, dtype, device):
    return load_kernels(kernel_metadata(config), dtype, device)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def get_operator_config(irreps_in, irreps_out, correlation, num_elements, dtype, use_reduced_cg=True):
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    return json.dumps(
        dict(
            irreps_in=str(irreps_in),
            irreps_out=str(irreps_out),
            correlation=correlation,
            num_elements=num_elements,
            source_dtype=str(dtype).removeprefix("torch."),
            use_reduced_cg=use_reduced_cg,
            output_dim=irreps_out.dim,
        ),
        separators=(",", ":"),
    )


@torch.library.custom_op("lunaeq::symmetric_contraction_forward", mutates_args=(), device_types="cuda")
def symmetric_contraction_forward(
    x: torch.Tensor, weights: torch.Tensor, attrs: torch.Tensor, config: str
) -> torch.Tensor:
    kernels = get_kernels(config, x.dtype, x.device)
    return kernels.forward.run(x.contiguous(), weights.contiguous(), attrs.contiguous())


@symmetric_contraction_forward.register_fake
def forward_fake(x, weights, attrs, config):
    return x.new_empty((x.shape[0], json.loads(config)["output_dim"]))


@torch.library.custom_op(
    "lunaeq::symmetric_contraction_backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor x, Tensor weights, Tensor attrs, Tensor g, bool[] needs, str config) -> (Tensor?, Tensor?, Tensor?)",
)
def symmetric_contraction_backward(x, weights, attrs, g, needs, config):
    kernels = get_kernels(config, x.dtype, x.device)
    return kernels.backward.run(
        x.contiguous(), weights.contiguous(), attrs.contiguous(), g.contiguous(), *needs
    )


@symmetric_contraction_backward.register_fake
def backward_fake(x, weights, attrs, g, needs, config):
    return tuple(
        torch.empty_like(value, memory_format=torch.contiguous_format) if needed else None
        for value, needed in zip((x, weights, attrs), needs)
    )


@torch.library.custom_op(
    "lunaeq::symmetric_contraction_double_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        '(Tensor x, Tensor weights, Tensor attrs, Tensor g, Tensor? rx, Tensor? rw, Tensor? ra, bool[] '
        'needs, str config) -> (Tensor?, Tensor?, Tensor?, Tensor?)'
    ),
)
def symmetric_contraction_double_backward(x, weights, attrs, g, rx, rw, ra, needs, config):
    kernels = get_kernels(config, x.dtype, x.device)
    return kernels.double_backward.run(
        x.contiguous(),
        weights.contiguous(),
        attrs.contiguous(),
        g.contiguous(),
        None if rx is None else rx.contiguous(),
        None if rw is None else rw.contiguous(),
        None if ra is None else ra.contiguous(),
        *needs,
    )


@symmetric_contraction_double_backward.register_fake
def double_backward_fake(x, weights, attrs, g, rx, rw, ra, needs, config):
    return tuple(
        torch.empty_like(value, memory_format=torch.contiguous_format) if needed else None
        for value, needed in zip((x, weights, attrs, g), needs)
    )


@torch.library.custom_op(
    "lunaeq::symmetric_contraction_triple_backward",
    mutates_args=(),
    device_types="cuda",
    schema=(
        '(Tensor x, Tensor weights, Tensor attrs, Tensor g, Tensor? rx, Tensor? rw, Tensor? ra, Tensor? '
        'bx, Tensor? bw, Tensor? ba, Tensor? bg, bool[] needs, str config) -> (Tensor?, Tensor?, '
        'Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)'
    ),
)
def symmetric_contraction_triple_backward(x, weights, attrs, g, rx, rw, ra, bx, bw, ba, bg, needs, config):
    kernels = get_kernels(config, x.dtype, x.device)
    inputs = (x, weights, attrs, g, rx, rw, ra, bx, bw, ba, bg)
    return kernels.triple_backward.run(
        *(None if value is None else value.contiguous() for value in inputs), *needs
    )


@symmetric_contraction_triple_backward.register_fake
def triple_backward_fake(x, weights, attrs, g, rx, rw, ra, bx, bw, ba, bg, needs, config):
    return tuple(
        torch.empty_like(value, memory_format=torch.contiguous_format) if needed else None
        for value, needed in zip((x, weights, attrs, g, rx, rw, ra), needs)
    )


def forward_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:3])
    ctx.config = inputs[3]


def backward_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:4])
    ctx.config = inputs[5]


def double_backward_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*inputs[:7])
    ctx.config = inputs[8]


def symmetric_contraction(x, weights, attrs, irreps_in, irreps_out, correlation=3, *, use_reduced_cg=True):
    """Contract equivariant features in the selected symmetric CG basis."""
    config = get_operator_config(
        str(irreps_in), str(irreps_out), correlation, weights.shape[0], x.dtype, use_reduced_cg
    )
    return SymmetricContractionForward.apply(x, weights.flatten(1), attrs, config)


class SymmetricContractionForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, attrs, config):
        if x.ndim != 2:
            raise ValueError("SymmetricContraction expects a 2D input [atoms, irreps_in.dim]")
        ctx.config = config
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weights, attrs)
        return symmetric_contraction_forward(x, weights, attrs, config)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return (None,) * 4
        inputs = (*ctx.saved_tensors, g, list(ctx.needs_input_grad[:3]), ctx.config)
        if torch.is_grad_enabled():
            result = SymmetricContractionBackward.apply(*inputs)
        else:
            result = symmetric_contraction_backward(*inputs)
        return *result, None


class SymmetricContractionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, attrs, g, needs, config):
        ctx.config = config
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weights, attrs, g)
        return symmetric_contraction_backward(x, weights, attrs, g, needs, config)

    @staticmethod
    def backward(ctx, rx, rw, ra):
        if rx is None and rw is None and ra is None:
            return (None,) * 6
        inputs = (*ctx.saved_tensors, rx, rw, ra, list(ctx.needs_input_grad[:4]), ctx.config)
        if torch.is_grad_enabled():
            result = SymmetricContractionDoubleBackward.apply(*inputs)
        else:
            result = symmetric_contraction_double_backward(*inputs)
        return *result, None, None


class SymmetricContractionDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, attrs, g, rx, rw, ra, needs, config):
        ctx.config = config
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weights, attrs, g, rx, rw, ra)
        return symmetric_contraction_double_backward(x, weights, attrs, g, rx, rw, ra, needs, config)

    @staticmethod
    def backward(ctx, bx, bw, ba, bg):
        if bx is None and bw is None and ba is None and bg is None:
            return (None,) * 9
        result = symmetric_contraction_triple_backward(
            *ctx.saved_tensors, bx, bw, ba, bg, list(ctx.needs_input_grad[:7]), ctx.config
        )
        return *result, None, None


# make_fx exposes the CUDA ops without the Function nodes.
symmetric_contraction_forward.register_autograd(
    SymmetricContractionForward.backward, setup_context=forward_setup
)


symmetric_contraction_backward.register_autograd(
    SymmetricContractionBackward.backward, setup_context=backward_setup
)


symmetric_contraction_double_backward.register_autograd(
    SymmetricContractionDoubleBackward.backward, setup_context=double_backward_setup
)
