"""Cartesian polynomial gradients and the analytic normalization chain.

Each order contracts polynomial derivatives before normalization.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import prod
import math

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from .forward import (
    build_forward_graph,
    build_forward_program,
    make_scalar_source,
    normalization_layout,
)
from .utils import (
    BLOCK_THREADS, BACKWARD_INPUTS, BACKWARD_OUTPUTS,
    TRIPLE_BACKWARD_INPUTS, TRIPLE_BACKWARD_OUTPUTS,
)
from .scalar import (
    ScalarGraph, harmonic_derivative, cuda_assignments, lower_expression,
)

NORMALIZATION_BACKWARD = r"""
const Element nv = nx * vx + ny * vy + nz * vz;
const Element out_x = (vx - nx * (active * nv)) * inv;
const Element out_y = (vy - ny * (active * nv)) * inv;
const Element out_z = (vz - nz * (active * nv)) * inv;
"""

NORMALIZATION_DOUBLE_BACKWARD = r"""
const Element nv = nx * vx + ny * vy + nz * vz;
const Element uv = ux * vx + uy * vy + uz * vz;
const Element factor = Element(3.0) * (nu * nv) - uv;
const Element out_x = (nx * factor - (vx * nu + ux * nv)) * inv2;
const Element out_y = (ny * factor - (vy * nu + uy * nv)) * inv2;
const Element out_z = (nz * factor - (vz * nu + uz * nv)) * inv2;
"""

NORMALIZATION_TRIPLE_BACKWARD = r"""
const Element nv = nx * vx + ny * vy + nz * vz;
const Element rv = rx * vx + ry * vy + rz * vz;
const Element bv = bx * vx + by * vy + bz * vz;
const Element factor_v = Element(3.0) * (nr * nb) - rb;
const Element factor_r = Element(3.0) * (nv * nb) - bv;
const Element factor_b = Element(3.0) * (nv * nr) - rv;
const Element factor_n = Element(3.0) * ((rv * nb + bv * nr) + nv * rb)
    - Element(15.0) * (nv * (nr * nb));
const Element out_x = (((vx * factor_v + rx * factor_r) + bx * factor_b) + nx * factor_n) * (inv2 * inv);
const Element out_y = (((vy * factor_v + ry * factor_r) + by * factor_b) + ny * factor_n) * (inv2 * inv);
const Element out_z = (((vz * factor_v + rz * factor_r) + bz * factor_b) + nz * factor_n) * (inv2 * inv);
"""


def build_backward_program(
    meta,
    *,
    needs: Sequence[bool] | None = None,
    available: Iterable[str] | None = None,
) -> dict[str, object]:
    """Select the requested input gradient and only its live operands.

    Degree zero has no derivative edge.  Without input normalization, a
    request containing only degrees zero and one has a constant Jacobian,
    so its backward reads ``grad_out`` alone.  Higher polynomial degrees and all
    normalized positive degrees also read ``x``.  A missing seed removes
    the contraction without allocating a zero seed tensor.
    """
    if needs is None:
        needs = (True,)
    if len(needs) != len(BACKWARD_OUTPUTS):
        raise ValueError("Spherical harmonics backward needs one output flag")
    live = set(BACKWARD_INPUTS if available is None else available)
    unknown = live.difference(BACKWARD_INPUTS)
    if unknown:
        raise ValueError(f"Unknown spherical harmonics backward inputs: {sorted(unknown)}")

    degrees = tuple(meta["degrees"])
    positive = any(degree > 0 for degree in degrees)
    required = (
        ("x", "grad_out")
        if meta["normalize"] or any(degree > 1 for degree in degrees)
        else ("grad_out",)
    )
    active = bool(needs[0]) and positive and all(role in live for role in required)
    return {
        "order": 1,
        "outputs": BACKWARD_OUTPUTS,
        "active_outputs": ("dx",) if active else (),
        "zero_outputs": ("dx",) if needs[0] and not active else (),
        "inputs": required if active else (),
    }


def build_double_backward_program(meta, *, needs=(True, True), available=None):
    if needs is None:
        needs = (True, True)
    if len(needs) != 2:
        raise ValueError('Spherical harmonics double-backward needs two output flags')
    roles = ('x', 'grad_out', 'rx')
    available = set(roles if available is None else available)
    if available.difference(roles):
        raise ValueError('Unknown spherical-harmonic double-backward inputs')

    degrees = tuple(meta['degrees'])
    active, inputs = set(), set()
    for target, wanted, derivative, base in (
        ('sx', needs[0], 2, ('grad_out', 'rx')),
        ('sg', needs[1], 1, ('rx',)),
    ):
        live = any(l > 0 if meta['normalize'] else l >= derivative for l in degrees)
        required = (
            *(('x',) if meta['normalize'] or any(l > derivative for l in degrees) else ()),
            *base,
        )
        if wanted and live and set(required).issubset(available):
            active.add(target)
            inputs.update(required)
    return dict(
        order=2,
        outputs=('sx', 'sg'),
        active_outputs=tuple(t for t in ('sx', 'sg') if t in active),
        zero_outputs=(),
        inputs=tuple(t for t in roles if t in inputs),
    )


def build_triple_backward_program(
    meta: Mapping[str, object],
    *,
    needs: Sequence[bool] | None = None,
    available: Iterable[str] | None = None,
) -> dict[str, object]:
    """Select complete contractions without zero seeds or constant inputs.

    ``needs`` follows ``(x, grad_out, rx)`` order. Every term is removed if
    its polynomial derivative is identically zero or a required value is
    absent. An order-q derivative of an unnormalized degree-q polynomial is
    constant and does not read x; mixed requests read x only if a contributing
    degree exceeds q. Only the arguments needed by selected outputs are saved.
    """
    if needs is None:
        needs = (True,) * len(TRIPLE_BACKWARD_OUTPUTS)
    if len(needs) != len(TRIPLE_BACKWARD_OUTPUTS):
        raise ValueError("Spherical-harmonic triple-backward needs three output flags")
    live = set(TRIPLE_BACKWARD_INPUTS if available is None else available)
    unknown = live.difference(TRIPLE_BACKWARD_INPUTS)
    if unknown:
        raise ValueError(f"Unknown spherical-harmonic triple-backward inputs: {sorted(unknown)}")
    selected = {
        output for output, needed in zip(TRIPLE_BACKWARD_OUTPUTS, needs) if needed
    }
    degrees = tuple(meta["degrees"])
    normalized_positive = bool(meta["normalize"]) and any(degree > 0 for degree in degrees)
    active, inputs = set(), set()

    def select_contraction(target, order, values):
        if target not in selected:
            return
        exists = normalized_positive or any(degree >= order for degree in degrees)
        if not exists:
            return
        needs_x = normalized_positive or any(degree > order for degree in degrees)
        dependencies = (("x",) if needs_x else ()) + values
        if not all(role in live for role in dependencies):
            return
        active.add(target)
        inputs.update(dependencies)

    # Input differentiation of double-backward contracts a third derivative
    # with grad_out/rx/bx, and a second derivative with rx/bg.
    select_contraction('tx', 3, ('grad_out', 'rx', 'bx'))
    select_contraction('tx', 2, ('rx', 'bg'))
    # grad_out differentiation contracts only the input-side seed.
    select_contraction('tg', 2, ('rx', 'bx'))
    # rx differentiation uses grad_out/bx or the first derivative with bg.
    select_contraction('trx', 2, ('grad_out', 'bx'))
    select_contraction('trx', 1, ('bg',))

    return {
        "order": 3,
        "outputs": TRIPLE_BACKWARD_OUTPUTS,
        "active_outputs": tuple(
            output for output in TRIPLE_BACKWARD_OUTPUTS if output in active
        ),
        "zero_outputs": tuple(
            output
            for output in TRIPLE_BACKWARD_OUTPUTS
            if output in selected and output not in active
        ),
        "inputs": tuple(
            role for role in TRIPLE_BACKWARD_INPUTS if role in inputs
        ),
    }


def make_backward_source(meta, program):
    """Contract polynomial gradients before applying the normalization chain."""
    order = program['order']
    if order < 3:
        raw, raw_outputs = build_forward_graph({
            **meta, 'normalize': False, 'normalization': 'component',
            'degrees': tuple(range(max(meta['degrees']) + 1)),
        })
    components = tuple(
        (degree, component) for degree in meta['degrees']
        for component in range(2 * degree + 1)
    )
    graph = ScalarGraph()

    def scale(v, s):
        return tuple((graph.binary('mul', x, s) for x in v))

    def plus(a, b):
        return tuple((graph.binary('add', x, y) for x, y in zip(a, b)))

    def minus(a, b):
        return tuple((graph.binary('sub', x, y) for x, y in zip(a, b)))

    def dot(a, b):
        value = graph.zero
        for x, y in zip(a, b):
            value = graph.binary('add', value, graph.binary('mul', x, y))
        return value

    zero = (graph.zero,) * 3
    x = (
        tuple((graph.input('x', i) for i in range(3)))
        if 'x' in program['inputs']
        else zero
    )
    r = (
        tuple((graph.input('rx', i) for i in range(3)))
        if 'rx' in program['inputs']
        else zero
    )
    b = (
        tuple((graph.input('bx', i) for i in range(3)))
        if 'bx' in program['inputs']
        else zero
    )
    if meta['normalize']:
        length = graph.node('norm', *x)
        denominator = graph.node('clamp', length, graph.constant(1e-12))
        inv = graph.binary('div', graph.one, denominator)
        active = graph.node(
            'where',
            graph.node('ge', length, graph.constant(1e-12)),
            graph.one,
            graph.zero,
        )
    else:
        inv, active = (graph.one, graph.zero)
    n = scale(x, inv)
    inv2 = graph.binary('mul', active, graph.binary('mul', inv, inv))
    nr, nb, rb = (dot(n, r), dot(n, b), dot(r, b))
    # a/d are normalization applied to rx/bx; c is their mixed second derivative.
    a = scale(minus(r, scale(n, graph.binary('mul', active, nr))), inv)
    d = scale(minus(b, scale(n, graph.binary('mul', active, nb))), inv)
    three = graph.constant(3.0)
    c = scale(
        minus(
            scale(
                n,
                graph.binary(
                    'sub', graph.binary('mul', three, graph.binary('mul', nr, nb)), rb
                ),
            ),
            plus(scale(r, nb), scale(b, nr)),
        ),
        inv2,
    )

    def normalization_backward(source, value, direction=None, dot_direction=None):
        u = zero if direction is None else direction
        variables = dict(
            nx=n[0],
            ny=n[1],
            nz=n[2],
            inv=inv,
            inv2=inv2,
            active=active,
            rx=r[0],
            ry=r[1],
            rz=r[2],
            bx=b[0],
            by=b[1],
            bz=b[2],
            nr=nr,
            nb=nb,
            rb=rb,
            vx=value[0],
            vy=value[1],
            vz=value[2],
            ux=u[0],
            uy=u[1],
            uz=u[2],
            nu=dot_direction,
        )
        for name, expression in cuda_assignments(source):
            variables[name] = lower_expression(graph, expression, variables)
        return tuple((variables[name] for name in ('out_x', 'out_y', 'out_z')))

    basis_values = {}

    def harmonic_value(node):
        if node.index in basis_values:
            return basis_values[node.index]
        if node.op == 'input':
            value = n[node.value[1]]
        elif node.op == 'constant':
            value = graph.constant(float.fromhex(node.value))
        else:
            args = tuple(harmonic_value(arg) for arg in node.args)
            if node.op in ('add', 'sub', 'mul', 'div'):
                value = graph.binary(node.op, *args)
            elif node.op == 'neg':
                value = graph.neg(args[0])
            elif node.op == 'pow':
                value = graph.power(args[0], node.value)
            else:
                value = graph.node(node.op, *args, value=node.value)
        basis_values[node.index] = value
        return value

    def polynomial_derivative(component, derivative):
        degree, column = component
        divisor = (math.sqrt(4 * math.pi) if meta['normalization'] == 'integral'
                   else math.sqrt(2 * degree + 1) if meta['normalization'] == 'norm' else 1.0)
        value = graph.zero
        for lower_degree, lower_column, coefficient in harmonic_derivative(degree, column, derivative):
            basis = harmonic_value(raw_outputs['out'][lower_degree * lower_degree + lower_column])
            value = graph.binary('add', value,
                                 graph.binary('mul', graph.constant(coefficient / divisor), basis))
        return value

    weighted_grad = weighted_hess_r = weighted_hess_b = weighted_third = zero
    weighted_hessian = (graph.zero,) * 6
    sg, tg = [], []
    if order == 3:
        harmonics = {(0, 0): (graph.one, graph.zero, graph.zero, graph.zero)}
        gradients = {}

        def harmonic_gradients(degree, component):
            key = (degree, component)
            if key not in gradients:
                columns = []
                for axis in range(3):
                    values = [graph.zero] * 4
                    powers = tuple(int(i == axis) for i in range(3))
                    for l, column, coefficient in harmonic_derivative(degree, component, powers):
                        lower = harmonic_values(l, column)
                        for direction in range(4):
                            values[direction] = graph.binary('add', values[direction],
                                graph.binary('mul', graph.constant(coefficient), lower[direction]))
                    columns.append(values)
                gradients[key] = tuple(tuple(columns[axis][direction] for axis in range(3))
                                       for direction in range(4))
            return gradients[key]

        def harmonic_values(degree, component):
            # a/d/c stay fixed while lowering the raw polynomial. Its input
            # normalization derivatives are applied once, after contraction.
            key = (degree, component)
            if key not in harmonics:
                grad, grad_a, _, _ = harmonic_gradients(degree, component)
                harmonics[key] = (graph.binary('mul', dot(n, grad), graph.constant(1.0/degree)),
                             dot(a, grad), dot(d, grad),
                             graph.binary('add', dot(d, grad_a), dot(c, grad)))
            return harmonics[key]

        tg = [graph.zero] * len(components)
        offset = 0
        for degree in meta['degrees']:
            divisor = (math.sqrt(4*math.pi) if meta['normalization'] == 'integral'
                       else math.sqrt(2*degree+1) if meta['normalization'] == 'norm' else 1.0)
            factor = graph.constant(1.0/divisor)
            # Adjacent +/-m components share their lower-degree basis. Keep
            # that pair close in the CUDA program without changing output slots.
            component_order = sorted(range(2*degree+1), key=lambda m: abs(m-degree))
            for component in component_order:
                column = offset + component
                grad, grad_a, grad_b, grad_m = harmonic_gradients(degree, component)
                _, _, _, mixed = harmonic_values(degree, component)
                g = graph.binary('mul', graph.input('grad_out', column), factor) if 'grad_out' in program['inputs'] else graph.zero
                h = graph.binary('mul', graph.input('bg', column), factor) if 'bg' in program['inputs'] else graph.zero
                weighted_grad = plus(weighted_grad, scale(grad, g))
                weighted_hess_r = plus(weighted_hess_r, scale(grad_a, g))
                weighted_hess_b = plus(weighted_hess_b, plus(scale(grad_b, g), scale(grad, h)))
                weighted_third = plus(weighted_third, plus(scale(grad_m, g), scale(grad_a, h)))
                tg[column] = graph.binary('mul', mixed, factor)
            offset += 2*degree+1
    else:
        for column, component in enumerate(components):
            grad = tuple(polynomial_derivative(component, tuple(int(k == i) for k in range(3)))
                         for i in range(3))
            g = graph.input('grad_out', column) if 'grad_out' in program['inputs'] else graph.zero
            weighted_grad = plus(weighted_grad, scale(grad, g))
            if order == 2:
                hess = [[polynomial_derivative(component,
                             tuple(int(k == i) + int(k == j) for k in range(3)))
                         for j in range(3)] for i in range(3)]
                weighted_hessian = tuple(
                    graph.binary('add', previous, graph.binary('mul', hess[i][j], g))
                    for previous, (i, j) in zip(weighted_hessian,
                                               ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))))
                sg.append(dot(grad, a))
    if order == 1:
        outputs = dict(dx=normalization_backward(NORMALIZATION_BACKWARD, weighted_grad))
    elif order == 2:
        h00,h01,h02,h11,h12,h22 = weighted_hessian
        weighted_hess_r = tuple(dot(row, a) for row in (
            (h00,h01,h02), (h01,h11,h12), (h02,h12,h22),
        ))
        outputs = dict(
            sx=plus(
                normalization_backward(NORMALIZATION_BACKWARD, weighted_hess_r),
                normalization_backward(
                    NORMALIZATION_DOUBLE_BACKWARD, weighted_grad, r, nr
                ),
            ),
            sg=tuple(sg),
        )
    else:
        outputs = dict(
            tx=plus(
                plus(
                    normalization_backward(NORMALIZATION_BACKWARD, weighted_third),
                    normalization_backward(
                        NORMALIZATION_DOUBLE_BACKWARD, weighted_hess_b, r, nr
                    ),
                ),
                plus(
                    normalization_backward(
                        NORMALIZATION_DOUBLE_BACKWARD, weighted_hess_r, b, nb
                    ),
                    normalization_backward(
                        NORMALIZATION_TRIPLE_BACKWARD, weighted_grad
                    ),
                ),
            ),
            tg=tuple(tg),
            trx=plus(
                normalization_backward(NORMALIZATION_BACKWARD, weighted_hess_b),
                normalization_backward(
                    NORMALIZATION_DOUBLE_BACKWARD, weighted_grad, b, nb
                ),
            ),
        )
    return make_scalar_source(meta, program, graph=graph, outputs=outputs)


def program_record(op, order, *, needs=None, available=None):
    """Cache the live dependency plan used by launch and autograd saving."""
    if order == 1:
        inputs, builder = ('x', 'grad_out'), build_backward_program
    elif order == 2:
        inputs, builder = ('x', 'grad_out', 'rx'), build_double_backward_program
    elif order == 3:
        inputs, builder = ('x', 'grad_out', 'rx', 'bx', 'bg'), build_triple_backward_program
    else:
        raise ValueError('Expected backward, double-backward or triple-backward')
    needs = tuple(needs) if needs is not None else None
    available = tuple(inputs if available is None else available)
    key = ('program', order, needs, tuple(sorted(available)))
    record = op.kernels.get(key)
    if record is None:
        program = builder(op.meta, needs=needs, available=available)
        record = {'program': program, 'save_roles': program['inputs'], 'launch': None}
        op.kernels[key] = record
        if len(op.kernels) > 128:
            op.kernels.popitem(last=False)
    else:
        op.kernels.move_to_end(key)
    return record


def run_backward(
    op, order, x, grad_out, rx=None, bx=None, bg=None,
    *, needs=None, num_items=None, norm_layout=None,
):
    """Run the fused polynomial contraction for the requested backward order."""
    values = {'x': x, 'grad_out': grad_out}
    if order >= 2:
        values['rx'] = rx
    if order == 3:
        values.update(bx=bx, bg=bg)
    available = tuple(name for name, value in values.items() if value is not None)
    record = program_record(op, order, needs=needs, available=available)
    program = record['program']
    if not program['inputs']:
        return tuple(None for _ in program['outputs'])

    first = values[program['inputs'][0]]
    batch = first.shape[:-1]
    num_items = prod(batch) if num_items is None else num_items
    if num_items == 0:
        return tuple(
            first.new_empty(batch + (op.meta['out_dim'] if name in ('sg', 'tg') else 3,))
            if name in program['active_outputs'] else None
            for name in program['outputs']
        )

    bound = max(3, op.meta['out_dim']) * num_items + 65535 * BLOCK_THREADS
    if bound > (1 << 63) - 1:
        raise OverflowError('SphericalHarmonics offsets exceed signed int64 capacity')
    index_bits = 64 if bound > (1 << 31) - 1 else op.index_bits
    arch = get_cuda_arch(first.device)
    if op.normalize and op._lmax > 0:
        norm_layout = normalization_layout(x) if norm_layout is None else norm_layout
    else:
        norm_layout = None

    key = (
        order, tuple(needs) if needs is not None else None, available,
        index_bits, arch, norm_layout,
    )
    launch_record = op.kernels.get(key)
    if launch_record is None:
        meta = op.meta
        if norm_layout is not None:
            meta = {**meta, 'norm_layout': norm_layout}
        precision = 'fp64' if meta['dtype'] == torch.float64 else 'fp32'
        phase = ('backward', 'double_backward', 'triple_backward')[order - 1]
        source = make_backward_source(meta, program)
        if order == 1:
            source = combine_cuda_sources({
                'forward': make_scalar_source(meta, build_forward_program(meta)),
                'backward': source,
            })
            phase = 'forward_backward'
        extension = load_cuda_extension(
            f'spherical_harmonics_{phase}_{precision}', source,
            index_bits=index_bits, arch=arch, use_fast_math=False, minimal_headers=True,
        )
        launch = extension.backward.launch if order == 1 else extension.launch
        op.kernels[key] = {**record, 'launch': launch}
        if len(op.kernels) > 128:
            op.kernels.popitem(last=False)
    else:
        launch = launch_record['launch']
        op.kernels.move_to_end(key)

    result = launch(*(values[name].contiguous() for name in program['inputs']), num_items)
    return tuple(
        value.reshape(batch + (value.shape[-1],)) if value is not None else None
        for value in result
    )
