"""Spherical-harmonic function, Tensor boundary and recomputing autograd."""

from collections import OrderedDict
from functools import lru_cache
from math import prod

import torch
from torch.utils._python_dispatch import _disable_current_modes

from ...utils import is_compiling
from ...utils.operator_config import encode_config, decode_config

from ..irreps import Irreps as LocalIrreps, PublicIrreps as Irreps
from .kernels.backward import (
    build_double_backward_program,
    build_triple_backward_program,
    program_record,
    run_backward,
)
from .kernels.forward import normalization_layout, run_forward
from .replay import higher_backward, reference_forward


UNSET = object()


__all__ = ['spherical_harmonics']


def parse_irreps(irreps_out, irreps_in=None):
    is_irreps = isinstance(irreps_out, (str, LocalIrreps)) or hasattr(irreps_out, 'ls')
    output = Irreps(irreps_out) if is_irreps else None
    if irreps_in is None:
        irreps_in = '1e' if output is not None and any(ir.l % 2 and ir.p == 1 for _, ir in output) else '1o'
    irreps_in = Irreps(irreps_in)
    if irreps_in not in (Irreps('1o'), Irreps('1e')):
        raise ValueError('SphericalHarmonics input must be one vector (1o) or pseudovector (1e)')

    parity = irreps_in[0].ir.p
    if output is not None:
        if any(ir.p != parity**ir.l for _, ir in output):
            raise ValueError('SphericalHarmonics output parity is inconsistent with input parity')
        degrees = tuple(l for mul, ir in output for l in [ir.l] * mul)
    else:
        degrees = (irreps_out,) if isinstance(irreps_out, int) else tuple(irreps_out)

    if not degrees or any(not isinstance(l, int) or l < 0 for l in degrees):
        raise ValueError('SphericalHarmonics degrees must be nonempty nonnegative integers')
    if max(degrees) > 12:
        raise NotImplementedError('SphericalHarmonics supports degrees through 12')
    output = Irreps([(1, (l, parity**l)) for l in degrees]).simplify()

    return irreps_in, output, degrees


@lru_cache(maxsize=128)
def spherical_harmonics_config(degrees, normalize, normalization, dtype, index_bits=32, irreps_in=None):
    _, _, degrees = parse_irreps(degrees, irreps_in)
    return SphericalHarmonicsConfig(degrees, normalize, normalization, dtype, index_bits, irreps_in)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def spherical_harmonics_operator_config(degrees, normalize, normalization, index_bits, dtype):
    return encode_config((degrees, normalize, normalization, index_bits, dtype))


@lru_cache(maxsize=128)
def get_spherical_harmonics_config(config):
    degrees, normalize, normalization, index_bits, dtype = decode_config(config)
    with _disable_current_modes():
        return spherical_harmonics_config(degrees, normalize, normalization, dtype, index_bits)


@torch.library.custom_op("lunaeq::spherical_harmonics_forward", mutates_args=())
def spherical_harmonics_forward(x: torch.Tensor, config: str) -> torch.Tensor:
    return run_forward(get_spherical_harmonics_config(config), x)


@spherical_harmonics_forward.register_fake
def spherical_harmonics_forward_fake(x, config):
    op = get_spherical_harmonics_config(config)
    return x.new_empty((*x.shape[:-1], op.meta['out_dim']))


def spherical_harmonics_forward_backward(ctx, g):
    if g is None or not get_spherical_harmonics_config(ctx.config)._lmax:
        return None, None
    (x,) = ctx.saved_tensors
    return spherical_harmonics_backward(x, g, ctx.config), None


def spherical_harmonics_setup_context(ctx, inputs, output):
    ctx.config = inputs[-1]
    if get_spherical_harmonics_config(ctx.config)._lmax:
        ctx.save_for_backward(*inputs[:-1])
    ctx.set_materialize_grads(False)


@torch.library.custom_op(
    "lunaeq::spherical_harmonics_backward",
    mutates_args=(),
    schema="(Tensor x, Tensor g, str config) -> Tensor",
)
def spherical_harmonics_backward(x, g, config):
    op = get_spherical_harmonics_config(config)
    if not op._lmax:
        raise ValueError('Constant spherical harmonics have no input-gradient contribution')
    return run_backward(op, 1, x, g)[0]


@spherical_harmonics_backward.register_fake
def spherical_harmonics_backward_fake(x, g, config):
    if not get_spherical_harmonics_config(config)._lmax:
        raise ValueError('Constant spherical harmonics have no input-gradient contribution')
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "lunaeq::spherical_harmonics_double_backward",
    mutates_args=(),
    schema="(Tensor x, Tensor g, Tensor rx, bool[] needs, str config) -> (Tensor?, Tensor?)",
)
def spherical_harmonics_double_backward(x, g, rx, needs, config):
    op = get_spherical_harmonics_config(config)
    return run_backward(op, 2, x, g, rx, needs=needs)


@spherical_harmonics_double_backward.register_fake
def spherical_harmonics_double_backward_fake(x, g, rx, needs, config):
    op = get_spherical_harmonics_config(config)
    program = build_double_backward_program(op.meta, needs=needs, available=('x', 'grad_out', 'rx'))
    return tuple(
        (
            torch.empty_like(value, memory_format=torch.contiguous_format)
            if name in program['active_outputs']
            else None
        )
        for name, value in zip(program['outputs'], (x, g))
    )


@torch.library.custom_op(
    "lunaeq::spherical_harmonics_triple_backward",
    mutates_args=(),
    schema=(
        '(Tensor x, Tensor g, Tensor rx, Tensor? bx, Tensor? bg, bool[] needs, str config) -> (Tensor?, '
        'Tensor?, Tensor?)'
    ),
)
def spherical_harmonics_triple_backward(x, g, rx, bx, bg, needs, config):
    op = get_spherical_harmonics_config(config)
    return run_backward(op, 3, x, g, rx, bx, bg, needs=needs)


@spherical_harmonics_triple_backward.register_fake
def spherical_harmonics_triple_backward_fake(x, g, rx, bx, bg, needs, config):
    op = get_spherical_harmonics_config(config)
    available = (
        ('x', 'grad_out', 'rx') + (('bx',) if bx is not None else ()) + (('bg',) if bg is not None else ())
    )
    program = build_triple_backward_program(op.meta, needs=needs, available=available)
    return tuple(
        (
            torch.empty_like(value, memory_format=torch.contiguous_format)
            if name in program['active_outputs']
            else None
        )
        for name, value in zip(program['outputs'], (x, g, rx))
    )


def spherical_harmonics_backward_backward(ctx, rx):
    if rx is None:
        return None, None, None
    x, g = ctx.saved_tensors
    result = spherical_harmonics_double_backward(x, g, rx, list(ctx.needs_input_grad[:2]), ctx.config)
    return *result, None


def spherical_harmonics_double_backward_setup(ctx, inputs, output):
    ctx.config = inputs[-1]
    ctx.save_for_backward(*inputs[:3])
    ctx.set_materialize_grads(False)


def spherical_harmonics_double_backward_backward(ctx, bx, bg):
    if bx is None and bg is None:
        return (None,) * 5
    x, g, rx = ctx.saved_tensors
    result = spherical_harmonics_triple_backward(x, g, rx, bx, bg, list(ctx.needs_input_grad[:3]), ctx.config)
    return *result, None, None


def spherical_harmonics(l, x, normalize, normalization='integral', *, config=None):
    """Evaluate harmonics without a Module; cache only static CUDA configuration."""
    if config is None:
        degrees = l if isinstance(l, (int, str, LocalIrreps)) or hasattr(l, 'ls') else tuple(l)
        config = spherical_harmonics_config(degrees, bool(normalize), normalization, x.dtype)
    elif config.compute_dtype != x.dtype:
        config = spherical_harmonics_config(
            tuple(config._ls_list),
            config.normalize,
            config.normalization,
            x.dtype,
            config.index_bits,
            str(config.irreps_in),
        )
    op = config
    if x.ndim == 0 or x.shape[-1] < 3:
        raise ValueError('SphericalHarmonics input must have shape (...,3)')
    native = x.is_cuda and x.dtype in (torch.float32, torch.float64) and x.shape[-1] == 3
    if is_compiling() and native:
        descriptor = spherical_harmonics_operator_config(
            tuple(op._ls_list), op.normalize, op.normalization, op.index_bits, x.dtype
        )
        return torch.ops.lunaeq.spherical_harmonics_forward.default(x, descriptor)
    if not native:
        op.last_execution_backend = 'torch_compat'
        op.backward_backend = 'torch_compat'
        return reference_forward(
            x, degrees=tuple(op._ls_list), normalize=bool(op.normalize), normalization=op.normalization
        )

    op.last_execution_backend = 'cuda_fused'
    op.backward_backend = None
    norm_layout = None
    if op.normalize and op._lmax > 0:
        norm_layout = normalization_layout(x)

    tracked = (
        torch.is_grad_enabled() and not torch.is_inference_mode_enabled() and x.requires_grad and op._lmax > 0
    )
    if tracked:
        op.backward_backend = 'cuda_fused_recompute'

    return (
        SphericalHarmonicsFunction.apply(op, x, norm_layout)
        if tracked
        else run_forward(op, x, norm_layout=norm_layout)
    )


def registered_spherical_harmonics(x, degrees, normalize, normalization):
    return spherical_harmonics(degrees, x, normalize, normalization)


if not hasattr(torch.ops.lunaeq, 'spherical_harmonics'):
    _TORCH_LIBRARY = torch.library.Library('lunaeq', 'FRAGMENT')
    _TORCH_LIBRARY.define(
        'spherical_harmonics(Tensor x, int[] degrees, bool normalize, str normalization) -> Tensor'
    )
    _TORCH_LIBRARY.impl('spherical_harmonics', registered_spherical_harmonics, 'CompositeImplicitAutograd')


class SphericalHarmonicsConfig:
    """Static mathematical configuration and reusable CUDA launch records."""

    def __init__(self, degrees, normalize, normalization, dtype, index_bits=32, irreps_in=None):
        self.irreps_in, self.irreps_out, degrees = parse_irreps(degrees, irreps_in)
        if normalization not in ('component', 'norm', 'integral'):
            raise ValueError('normalization must be component, norm or integral')
        self._ls_list = degrees
        self._lmax = max(degrees)
        self.normalize, self.normalization = bool(normalize), normalization
        self.compute_dtype, self.index_bits = dtype, index_bits
        self.meta = dict(
            degrees=degrees,
            normalize=self.normalize,
            normalization=normalization,
            dtype=dtype,
            out_dim=self.irreps_out.dim,
        )
        self.kernels = OrderedDict()
        self.last_execution_backend = 'uninitialized'
        self.backward_backend = None

    def __getstate__(self):
        return {**self.__dict__, 'kernels': OrderedDict()}


class SphericalHarmonicsFunction(torch.autograd.Function):
    """Forward node saving X only when its derivative program needs it."""

    @staticmethod
    def forward(ctx, op, x, norm_layout):
        ctx.op = op
        ctx.norm_layout = norm_layout
        ctx.set_materialize_grads(False)
        required = program_record(op, 1)['save_roles'] if ctx.needs_input_grad[1] else ()
        ctx.save_for_backward(x if 'x' in required else None)
        return run_forward(op, x, norm_layout=norm_layout)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None
        (x,) = ctx.saved_tensors
        dx = (
            SphericalHarmonicsBackward.apply(ctx.op, x, g, ctx.norm_layout)
            if torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            else run_backward(ctx.op, 1, x, g, norm_layout=ctx.norm_layout)[0]
        )
        return None, dx, None


class SphericalHarmonicsBackward(torch.autograd.Function):
    """Backward saves only X/G required by double-backward."""

    @staticmethod
    def forward(ctx, op, x, g, norm_layout):
        ctx.op, ctx.num_items = op, prod(g.shape[:-1])
        ctx.norm_layout = norm_layout
        ctx.set_materialize_grads(False)
        ctx.needs = ctx.needs_input_grad[1:3]
        available = ['grad_out', 'rx'] + (['x'] if x is not None else [])
        required = program_record(op, 2, needs=ctx.needs, available=available)['save_roles']
        ctx.save_for_backward(x if 'x' in required else None, g if 'grad_out' in required else None)
        return run_backward(op, 1, x, g, num_items=ctx.num_items, norm_layout=norm_layout)[0]

    @staticmethod
    def backward(ctx, rx):
        x, g = ctx.saved_tensors
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            sx, sg = SphericalHarmonicsDoubleBackward.apply(
                ctx.op, x, g, rx, ctx.num_items, ctx.needs, ctx.norm_layout
            )
        else:
            sx, sg = run_backward(
                ctx.op, 2, x, g, rx, needs=ctx.needs, num_items=ctx.num_items, norm_layout=ctx.norm_layout
            )
        return None, sx, sg, None


class SphericalHarmonicsDoubleBackward(torch.autograd.Function):
    """Double-backward saves only X/G/R required by triple-backward."""

    @staticmethod
    def forward(ctx, op, x, g, rx, n, needs, norm_layout):
        ctx.op, ctx.num_items = op, n
        ctx.norm_layout = norm_layout
        ctx.set_materialize_grads(False)
        sx, sg = run_backward(op, 2, x, g, rx, needs=needs, num_items=n, norm_layout=norm_layout)
        ctx.needs = ctx.needs_input_grad[1:4]

        available = [
            role for role, v in zip(('x', 'grad_out', 'rx', 'bx', 'bg'), (x, g, rx, sx, sg)) if v is not None
        ]
        required = program_record(op, 3, needs=ctx.needs, available=available)['save_roles']
        ctx.save_for_backward(
            x if 'x' in required else None,
            g if 'grad_out' in required else None,
            rx if 'rx' in required else None,
        )
        return sx, sg

    @staticmethod
    def backward(ctx, bx, bg):
        x, g, rx = ctx.saved_tensors
        if ctx.op.normalize and torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            # Preserve fourth and later derivatives only when requested.
            # Ordinary terminal triple-backward retains the fused native recomputation.
            tx, tg, trx = higher_backward(ctx.op, x, g, rx, bx, bg, ctx.needs)
        else:
            tx, tg, trx = run_backward(
                ctx.op,
                3,
                x,
                g,
                rx,
                bx,
                bg,
                needs=ctx.needs,
                num_items=ctx.num_items,
                norm_layout=ctx.norm_layout,
            )
        return None, tx, tg, trx, None, None, None


spherical_harmonics_forward.register_autograd(
    spherical_harmonics_forward_backward, setup_context=spherical_harmonics_setup_context
)


spherical_harmonics_backward.register_autograd(
    spherical_harmonics_backward_backward, setup_context=spherical_harmonics_setup_context
)


spherical_harmonics_double_backward.register_autograd(
    spherical_harmonics_double_backward_backward, setup_context=spherical_harmonics_double_backward_setup
)
