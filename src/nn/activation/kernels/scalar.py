"""Scalar CUDA emission adapted from PyTorch 70d99e998b4955e0049d13a98d77ae1b14db1f45."""

from __future__ import annotations

import math
from numbers import Number
import struct

import torch

from .forward import FORWARD_FUNCTIONS, TANH_APPROX
from .backward import BACKWARD_FUNCTIONS

CUDA_ATOMS = {**FORWARD_FUNCTIONS, **BACKWARD_FUNCTIONS}
CUDA_MATH_OPERATIONS = {
    name: "fabs" if name == "abs" else name
    for name in (
        "abs", "exp", "tanh", "erf", "log1p", "cosh", "sinh",
        "sin", "cos", "log", "sqrt", "rsqrt", "expm1",
    )
}

def scalar_type(dtype):
    if dtype == torch.float32:
        return 'float'
    if dtype == torch.float64:
        return 'double'
    raise TypeError(f'Activation CUDA supports only float32/float64, got {dtype!r}')


def scalar_literal(value, dtype):
    value = float(value)
    scalar = scalar_type(dtype)
    if scalar == 'float':
        try:
            value = struct.unpack('f', struct.pack('f', value))[0]
        except OverflowError:
            value = math.copysign(math.inf, value)
    if math.isnan(value):
        return f'{scalar}(NAN)'
    if math.isinf(value):
        return ('-' if value < 0 else '') + f'{scalar}(INFINITY)'
    suffix = 'f' if scalar == 'float' else ''
    return value.hex() + suffix


def emit_activation_aten_scalar(opname, args, *, dtype, params=None, approximate_tanh=False):
    """Return one scalar CUDA expression for a registered ATen atom.

    ``args`` contains C++ expression strings for Tensor arguments.  Static
    Scalar arguments may either occur in their ATen positions as Python
    numbers/C++ expressions, or be supplied in ``params``.  GELU's mode is
    supplied as ``params['approximate']``. Powers 2 and 3 retain Torch's
    multiplication order; other static powers use CUDA's overloaded pow.

    The argument order is ATen's: backward atoms receive grad_output first.
    This preserves associations and exceptional values for the actual seed,
    rather than evaluating a derivative at a unit seed then multiplying.
    """
    scalar = scalar_type(dtype)
    name = str(opname).removeprefix('aten.').split('.')[0]
    params = {} if params is None else params
    arguments = tuple(scalar_literal(a, dtype) if isinstance(a, Number) else str(a) for a in args)
    arity = {
        'identity': 1,
        'silu': 1,
        'silu_backward': 2,
        'sigmoid': 1,
        'sigmoid_backward': 2,
        'tanh': 1,
        'tanh_backward': 2,
        'relu': 1,
        'abs': 1,
        'sgn': 1,
        'exp': 1,
        'cosh': 1,
        'sinh': 1,
        'gelu': 1,
        'gelu_backward': 2,
        'sin': 1,
        'cos': 1,
        'log': 1,
        'log1p': 1,
        'sqrt': 1,
        'rsqrt': 1,
        'reciprocal': 1,
        'expm1': 1,
        'erf': 1,
    }
    if name in ('gelu', 'gelu_backward'):
        mode = params.get('approximate', 'none')
        if mode not in ('none', 'tanh'):
            raise ValueError(f'Unsupported GELU approximation: {mode!r}')
        helper = name + ('_exact' if mode == 'none' else '_tanh')
    elif name in ('softplus', 'softplus_backward'):
        count = 1 if name == 'softplus' else 2
        if len(arguments) == count:
            arguments += (
                scalar_literal(params.get('beta', 1.0), dtype),
                scalar_literal(params.get('threshold', 20.0), dtype),
            )
        arity[name] = count + 2
        helper = name
    elif name == 'threshold_backward':
        if len(arguments) == 2:
            arguments += (scalar_literal(params.get('threshold', 0.0), dtype),)
        arity[name] = 3
        helper = name
    elif name == 'pow':
        exponent = params.get('exponent', args[1] if len(args) == 2 else None)
        if not isinstance(exponent, Number):
            raise ValueError(f'Activation pow needs a static scalar exponent, got {exponent!r}')
        arguments = arguments[:1]
        if exponent in (2, 3):
            arity[name] = 1
        else:
            arity[name] = 2
            arguments += (scalar_literal(exponent, dtype),)
    else:
        helper = name
    if name not in arity:
        raise ValueError(f'Unsupported activation scalar ATen atom: {opname}')
    if len(arguments) != arity[name]:
        raise ValueError(f'{opname} needs {arity[name]} scalar arguments, got {len(arguments)}')
    if name == 'identity':
        return arguments[0]
    if name in CUDA_MATH_OPERATIONS:
        if name == 'tanh' and approximate_tanh:
            if scalar != 'float':
                raise ValueError('Approximate tanh requires float32')
            return f'activation_tanh_approx({arguments[0]})'
        return f'::{CUDA_MATH_OPERATIONS[name]}({scalar}({arguments[0]}))'
    if name == 'sgn':
        x = f'{scalar}({arguments[0]})'
        return f'{scalar}(({scalar}(0)<{x})-({x}<{scalar}(0)))'
    if name == 'reciprocal':
        return f'({scalar}(1)/{scalar}({arguments[0]}))'
    if name == 'threshold_backward':
        g, x, threshold = (f'{scalar}({a})' for a in arguments)
        return f'({x}<={threshold}?{scalar}(0):{g})'
    if name == 'pow':
        x = f'{scalar}({arguments[0]})'
        if exponent == 2:
            return f'({x}*{x})'
        if exponent == 3:
            return f'({x}*{x}*{x})'
        return f'::pow({x},{scalar}({arguments[1]}))'
    return f'activation_{helper}<{scalar}>({",".join(arguments)})'


def activation_cuda_prelude(dtype, needed_ops, *, include_header=True, approximate_tanh=False):
    """Emit just the requested activation atoms, without derivative jets.

    ``needed_ops`` contains ATen names or ``(name, params)`` descriptors.
    Plain GELU names include both supported static variants. A descriptor
    can restrict them to the actual approximation. Simple math operations
    use CUDA overloads directly and need no device helper.
    CUDA compilation must use precise math (no --use_fast_math).  Ordinary
    tensor operations outside these atoms require explicit RN intrinsics;
    expressions within an ATen CUDA atom keep its original functor syntax.
    """
    scalar = scalar_type(dtype)
    selected = set()
    helpers = {
        'silu',
        'silu_backward',
        'sigmoid',
        'sigmoid_backward',
        'tanh_backward',
        'softplus',
        'softplus_backward',
        'relu',
    }
    direct = {*CUDA_MATH_OPERATIONS, 'sgn', 'reciprocal', 'threshold_backward', 'pow'}
    needed = False
    for item in needed_ops:
        name, params = item if isinstance(item, tuple) else (item, {})
        name = str(name).removeprefix('aten.').split('.')[0]
        if name == 'identity':
            continue
        needed = True
        if name in ('gelu', 'gelu_backward'):
            modes = (params['approximate'],) if 'approximate' in params else ('none', 'tanh')
            if any(mode not in ('none', 'tanh') for mode in modes):
                raise ValueError(f'Unsupported GELU approximation: {modes}')
            selected.update(name + ('_exact' if mode == 'none' else '_tanh') for mode in modes)
        elif name == 'tanh' and approximate_tanh:
            if scalar != 'float':
                raise ValueError('Approximate tanh requires float32')
            selected.add(name)
        elif name in helpers:
            selected.add(name)
        elif name in direct:
            continue
        else:
            raise ValueError(f'Unsupported activation scalar ATen atom: {name}')
    if not needed:
        return ''
    lines = []
    if include_header:
        lines.append('#include <cmath>')
    else:
        lines.append('')
    for name in sorted(selected):
        definition = (
            TANH_APPROX if name == "tanh" and approximate_tanh else CUDA_ATOMS[name]
        )
        lines.append(definition)
    return '\n'.join(lines) + '\n'
