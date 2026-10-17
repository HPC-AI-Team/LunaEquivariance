"""Reverse Cartesian arithmetic, shared by backward through triple-backward.

Each block is parsed into the ordered scalar graph. Reversing the graph
again produces the higher orders without duplicated polynomial formulas.
Registered normalized Torch graphs use these same scalar CUDA operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import prod
from threading import Lock

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from ....utils.torch_scalar import gradient
from .forward import (
    build_forward_graph,
    build_forward_program,
    evaluate_polynomial,
    make_scalar_source,
    normalization_layout,
)
from .utils import (
    INPUT_ROLES, BACKWARD_INPUTS, BACKWARD_OUTPUTS,
    TRIPLE_BACKWARD_INPUTS, TRIPLE_BACKWARD_OUTPUTS,
)
from .scalar import (
    HarmonicsScalarizer,
    ScalarGraph,
    cuda_assignments,
    lower_expression,
)

REVERSE_RULES = {
    'add': cuda_assignments(r"""
const Element grad_a = grad;
const Element grad_b = grad;
"""),
    'sub': cuda_assignments(r"""
const Element grad_a = grad;
const Element grad_b = -grad;
"""),
    'mul': cuda_assignments(r"""
const Element grad_a = grad * b;
const Element grad_b = grad * a;
"""),
    'div': cuda_assignments(r"""
const Element grad_a = grad / b;
const Element negative_seed = -grad;
const Element quotient = (a / b) / b;
const Element grad_b = negative_seed * quotient;
"""),
    'neg': cuda_assignments(r"""
const Element grad_a = -grad;
"""),
    'pow': cuda_assignments(r"""
const Element grad_a = grad * (Element(exponent) * Element(std::pow(a, exponent - 1)));
"""),
    'norm': cuda_assignments(r"""
const bool zero_norm = value == Element(0);
const Element unit = zero_norm ? Element(0) : a / value;
const Element grad_a = grad * unit;
"""),
    'clamp': cuda_assignments(r"""
const bool active = a >= b;
const Element grad_a = active ? grad : Element(0);
"""),
    'where': cuda_assignments(r"""
const Element grad_b = a ? grad : Element(0);
const Element grad_c = a ? Element(0) : grad;
"""),
}


def differentiate(graph, outputs, seeds, inputs):
    ancestors = graph.ancestors(outputs)
    ordered = [n for n in graph.nodes if n.index in ancestors]
    adj = {}

    for output, seed in zip(outputs, seeds):
        graph.accumulate_gradient(adj, output, seed)
    for n in reversed(ordered):
        g = adj.get(n.index)
        if g is None:
            continue
        a = n.args
        rule = REVERSE_RULES.get(n.op)
        if rule is None:
            if n.op not in ('input', 'constant', 'eq', 'ge'):
                raise ValueError(f'Unsupported scalar derivative: {n.op}')
            continue
        coordinates = a if n.op == 'norm' else (a[0],)
        for coordinate in coordinates:
            variables = {'grad': g, 'value': n, 'a': coordinate}
            variables.update(zip(('b', 'c'), a[1:]))
            if n.op == 'pow':
                variables['exponent'] = graph.constant(n.value)
            for name, expression in rule:
                value = lower_expression(graph, expression, variables)
                variables[name] = value
                if name.startswith('grad_'):
                    argument = coordinate if n.op == 'norm' else a[ord(name[-1]) - ord('a')]
                    graph.accumulate_gradient(adj, argument, value)
    return tuple(adj.get(n.index, graph.zero) for n in inputs)


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
        "terms": {"dx": ("dx_g",)} if active else {},
        "live_terms": ("dx_g",) if active else (),
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
    terms, inputs = {}, set()
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
            terms[target] = (target + '_rx',)
            inputs.update(required)
    return dict(
        order=2,
        outputs=('sx', 'sg'),
        active_outputs=tuple(t for t in ('sx', 'sg') if t in terms),
        zero_outputs=(),
        inputs=tuple(t for t in roles if t in inputs),
        terms=terms,
        live_terms=tuple(t for ts in terms.values() for t in ts),
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
    degree exceeds q. Seed groups select reverse contractions for all seeded outputs;
    term IDs describe the selected mathematics and do not control execution.
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
    terms = {}
    term_inputs = {}
    derivative_orders = {}
    seed_groups = {}
    live_terms = []

    def select_contraction(target, name, order, values, seed):
        if target not in selected:
            return
        exists = normalized_positive or any(degree >= order for degree in degrees)
        if not exists:
            return
        needs_x = normalized_positive or any(degree > order for degree in degrees)
        dependencies = (("x",) if needs_x else ()) + values
        if not all(role in live for role in dependencies):
            return
        terms.setdefault(target, []).append(name)
        if seed not in seed_groups.setdefault(target, []):
            seed_groups[target].append(seed)
        live_terms.append(name)
        term_inputs[name] = dependencies
        derivative_orders[name] = order

    # Input differentiation of double-backward contracts a third derivative
    # with grad_out/rx/bx, and a second derivative with rx/bg.
    select_contraction('tx', 'tx_bx', 3, ('grad_out', 'rx', 'bx'), 'bx')
    select_contraction('tx', 'tx_bg', 2, ('rx', 'bg'), 'bg')
    # grad_out differentiation contracts only the input-side seed.
    select_contraction('tg', 'tg_bx', 2, ('rx', 'bx'), 'bx')
    # rx differentiation uses grad_out/bx or the first derivative with bg.
    select_contraction('trx', 'trx_bx', 2, ('grad_out', 'bx'), 'bx')
    select_contraction('trx', 'trx_bg', 1, ('bg',), 'bg')

    dependencies = {role for values in term_inputs.values() for role in values}
    return {
        "order": 3,
        "outputs": TRIPLE_BACKWARD_OUTPUTS,
        "active_outputs": tuple(
            output for output in TRIPLE_BACKWARD_OUTPUTS if output in terms
        ),
        "zero_outputs": tuple(
            output
            for output in TRIPLE_BACKWARD_OUTPUTS
            if output in selected and output not in terms
        ),
        "inputs": tuple(
            role for role in TRIPLE_BACKWARD_INPUTS if role in dependencies
        ),
        "terms": {target: tuple(names) for target, names in terms.items()},
        "live_terms": tuple(live_terms),
        "term_inputs": term_inputs,
        "derivative_orders": derivative_orders,
        "seed_groups": {target: tuple(seeds) for target, seeds in seed_groups.items()},
        "needs_x": "x" in dependencies,
    }


def make_backward_source(meta, program, *, layout_specs=None):
    """Generate backward, double-backward or triple-backward CUDA.

    Normalized derivatives lower Torch's registered arithmetic. Polynomial
    derivatives reverse the same ordered graph used by forward.
    """
    if program['order'] not in (1, 2, 3):
        raise ValueError('Expected a spherical-harmonic backward program')
    if meta['normalize'] and max(meta['degrees']) > 0:
        graph, outputs = build_registered_graph(meta, program, layout_specs=layout_specs)
    else:
        graph, forward_outputs = build_forward_graph(meta)
        x = tuple(graph.input('x', i) for i in range(3))
        grad_out = tuple(graph.input('grad_out', i) for i in range(meta['out_dim']))
        rx = tuple(graph.input('rx', i) for i in range(3))
        bx = tuple(graph.input('bx', i) for i in range(3))
        bg = tuple(graph.input('bg', i) for i in range(meta['out_dim']))
        dx = differentiate(graph, forward_outputs['out'], grad_out, x)
        outputs = {'dx': dx}
        if program['order'] >= 2:
            second = differentiate(graph, dx, rx, (*x, *grad_out))
            input_width = len(x)
            sx, sg = second[:input_width], second[input_width:]
            outputs = {'sx': sx, 'sg': sg}
        if program['order'] == 3:
            outputs, groups = {}, {}
            output_end = input_width + len(grad_out)
            result_slices = {
                'tx': slice(0, input_width),
                'tg': slice(input_width, output_end),
                'trx': slice(output_end, None),
            }
            for target in program['active_outputs']:
                seed_group = program['seed_groups'][target]
                if seed_group not in groups:
                    # Reverse seeded outputs before shared norm/div nodes.
                    roots = (sx if 'bx' in seed_group else ()) + (sg if 'bg' in seed_group else ())
                    seeds = (bx if 'bx' in seed_group else ()) + (bg if 'bg' in seed_group else ())
                    groups[seed_group] = differentiate(graph, roots, seeds, (*x, *grad_out, *rx))
                outputs[target] = tuple(groups[seed_group][result_slices[target]])
    return make_scalar_source(meta, program, graph=graph, outputs=outputs)


def registered_triple_backward(
    program, names, input_gradient, x, grad_output, input_direction,
    grad_input_direction, grad_output_direction,
):
    """Differentiate the double-backward using each result's live seed group."""
    second_input, second_output = gradient(
        (input_gradient,), (x, grad_output), (input_direction,), create_graph=True
    )
    targets = {"tx": x, "tg": grad_output, "trx": input_direction}
    grouped = {}
    result = {}
    for name in names:
        grouped.setdefault(program['seed_groups'][name], []).append(name)
    for seed_group, output_names in grouped.items():
        # Combining seeds preserves the registered backward's rounding and pruning.
        # tx from bg and tg from bx, for example, do not need grad_output.
        values = ((second_input,) if 'bx' in seed_group else ()) + ((second_output,) if 'bg' in seed_group else ())
        seeds = ((grad_input_direction,) if 'bx' in seed_group else ()) + ((grad_output_direction,) if 'bg' in seed_group else ())
        gradients = gradient(
            values, tuple(targets[name] for name in output_names), seeds, create_graph=False,
        )
        result.update(zip(output_names, gradients))
    return tuple(result[name] for name in names)


def trace_registered_graph(meta, program, *, layout_specs=None):
    """Return the shared scalar DAG and the requested derivative output fields.

    ``layout_specs`` contains only each original role's shape and stride.  No
    tensor values, Torch derivative graph or FakeTensor are retained afterward.
    """
    graph = ScalarGraph()
    names = tuple(program['active_outputs'])
    if not names:
        return graph, {}

    run = partial(trace_harmonics_program, meta=meta, program=program, names=names)

    with torch.enable_grad():
        # Storage-free leaves retain every supplied batch dimension.
        width = int(meta['out_dim'])
        specs = {} if layout_specs is None else layout_specs
        batch_spec = next((specs[role] for role in INPUT_ROLES if role in specs), None)
        batch = tuple(batch_spec['shape'][:-1]) if batch_spec is not None else (2,)
        result = []
        for role in INPUT_ROLES:
            dimension = width if role in ('grad_out', 'bg') else 3
            spec = specs.get(role)
            if spec is None:
                shape = (*batch, dimension)
                stride = []
                step = 1
                for size in reversed(shape):
                    stride.append(step)
                    step *= max(1, size)
                stride = tuple(reversed(stride))
                if layout_specs is None and role == 'x' and meta.get('norm_layout') == 'xy_z':
                    # Native entry points have no original shape metadata.
                    stride = (1, 2)
            else:
                shape, stride = tuple(spec['shape']), tuple(spec['stride'])
            if not shape or shape[-1] != dimension or len(shape) != len(stride):
                raise ValueError(f'Invalid spherical-harmonic layout for {role}: {shape}, {stride}')
            result.append(
                torch.empty_strided(
                    shape,
                    stride,
                    device='meta',
                    dtype=meta['dtype'],
                    requires_grad=role in ('x', 'grad_out', 'rx'),
                )
            )
        inputs = tuple(result)
        traced = make_fx(run, tracing_mode='fake')(*inputs)

    outputs = HarmonicsScalarizer(graph, meta['dtype']).lower(traced, names)
    selected = tuple(field for name in names for field in outputs[name])
    live = graph.ancestors(selected)
    dependencies = {
        node.value[0] for node in graph.nodes if node.index in live and node.op == 'input'
    }
    omitted = dependencies.difference(program['inputs'])
    if omitted:
        raise ValueError(f'Torch graph has omitted runtime dependencies: {sorted(omitted)}')
    return graph, outputs


def build_registered_graph(meta, program, *, layout_specs=None):
    """Compile static metadata without advancing caller autograd counters.

    A single reusable host worker owns metadata differentiation. The submit
    lock permits at most one outstanding build; only the returned scalar DAG
    survives a build. No input values or Tensor derivative graphs are cached.
    """
    global metadata_executor
    if not program['active_outputs']:
        return trace_registered_graph(meta, program, layout_specs=layout_specs)
    with metadata_lock:
        if metadata_executor is None:
            metadata_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sh-metadata')
        return metadata_executor.submit(
            trace_registered_graph, meta, program, layout_specs=layout_specs
        ).result()


def trace_harmonics_program(x, grad_out, rx, bx, bg, *, meta, program, names):
    coordinates = torch.nn.functional.normalize(x, dim=-1) if meta['normalize'] else x
    out = evaluate_polynomial(coordinates, tuple(meta['degrees']), meta['normalization'])
    if program['order'] == 0:
        return (out,)
    (dx,) = gradient((out,), (x,), (grad_out,), create_graph=True)
    if program['order'] == 1:
        return (dx,)
    if program['order'] == 2:
        targets = {'sx': x, 'sg': grad_out}
        return gradient(
            (dx,), tuple(targets[name] for name in names), (rx,), create_graph=False,
        )
    return registered_triple_backward(program, names, dx, x, grad_out, rx, bx, bg)


metadata_lock = Lock()
metadata_executor = None


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
    """Run the requested derivative of the same ordered scalar polynomial."""
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

    bound = max(3, op.meta['out_dim']) * num_items + 65535 * 128
    if bound > (1 << 63) - 1:
        raise OverflowError('SphericalHarmonics offsets exceed signed int64 capacity')
    index_bits = 64 if bound > (1 << 31) - 1 else op.index_bits
    arch = get_cuda_arch(first.device)
    layout_specs = None
    layout_key = None
    if op.normalize and op._lmax > 0:
        norm_layout = normalization_layout(x) if norm_layout is None else norm_layout
        if all(
            value.is_contiguous() and value.shape[:-1] == batch
            for value in values.values() if value is not None
        ):
            # Feature reductions share one metadata trace across contiguous batch shapes.
            layout_key = 'contiguous'
        else:
            layout_key = tuple(
                (name, tuple(value.shape), tuple(value.stride()))
                for name, value in values.items() if value is not None
            )
    else:
        norm_layout = None

    key = (
        order, tuple(needs) if needs is not None else None, available,
        index_bits, arch, norm_layout, layout_key, None,
    )
    launch_record = op.kernels.get(key)
    if launch_record is None:
        meta = op.meta
        if norm_layout is not None:
            meta = {**meta, 'norm_layout': norm_layout}
            if layout_key == 'contiguous':
                layout_specs = {
                    name: dict(shape=(2, value.shape[-1]), stride=(value.shape[-1], 1))
                    for name, value in values.items() if value is not None
                }
            else:
                layout_specs = {name: dict(shape=shape, stride=stride) for name, shape, stride in layout_key}
        precision = 'fp64' if meta['dtype'] == torch.float64 else 'fp32'
        phase = ('backward', 'double_backward', 'triple_backward')[order - 1]
        source = make_backward_source(meta, program, layout_specs=layout_specs)
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
