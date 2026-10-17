"""Shared spherical harmonics scalar graph and registered Torch lowering."""

from __future__ import annotations
import ast
from functools import lru_cache
import math
import torch

from ....utils.torch_scalar import ScalarValue, Scalarizer
from .utils import INPUT_ROLES


@lru_cache(maxsize=None)
def cuda_assignments(source):
    """Parse scalar CUDA statements once, retaining their expression order."""
    statements = []
    source = '\n'.join(line.split('//', 1)[0] for line in source.splitlines())
    for statement in source.split(';'):
        if not statement.strip():
            continue
        declaration, expression = statement.split('=', 1)
        name = declaration.split()[-1]
        expression = expression.strip().replace('std::sqrt', 'math.sqrt').replace('std::pow', 'pow')
        if '?' in expression:
            condition, branches = expression.split('?', 1)
            yes, no = branches.split(':', 1)
            expression = f'({yes}) if ({condition}) else ({no})'
        statements.append((name, ast.parse(f'({expression})', mode='eval').body))
    return tuple(statements)


def lower_expression(graph, expression, variables):
    if isinstance(expression, ast.Name):
        return variables[expression.id]
    if isinstance(expression, ast.Constant):
        return graph.constant(expression.value)
    if isinstance(expression, ast.UnaryOp):
        value = lower_expression(graph, expression.operand, variables)
        if isinstance(expression.op, ast.USub):
            return graph.neg(value)
        if isinstance(expression.op, ast.UAdd):
            return value
    if isinstance(expression, ast.BinOp):
        a = lower_expression(graph, expression.left, variables)
        b = lower_expression(graph, expression.right, variables)
        op = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}.get(
            type(expression.op)
        )
        if op is not None:
            return graph.binary(op, a, b)
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
        function = expression.func
        if (
            isinstance(function.value, ast.Name)
            and function.value.id == 'math'
            and function.attr == 'sqrt'
        ):
            arg = lower_expression(graph, expression.args[0], variables)
            assert arg.op == 'constant'
            return graph.constant(math.sqrt(float.fromhex(arg.value)))
    if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
        op = {ast.Eq: 'eq', ast.GtE: 'ge'}.get(type(expression.ops[0]))
        if op is not None:
            return graph.node(
                op, lower_expression(graph, expression.left, variables),
                lower_expression(graph, expression.comparators[0], variables),
            )
    if isinstance(expression, ast.IfExp):
        return graph.node(
            'where', lower_expression(graph, expression.test, variables),
            lower_expression(graph, expression.body, variables),
            lower_expression(graph, expression.orelse, variables),
        )
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        name = expression.func.id
        if name == 'Element':
            return lower_expression(graph, expression.args[0], variables)
        if name == 'pow':
            value = lower_expression(graph, expression.args[0], variables)
            exponent = expression.args[1]
            if isinstance(exponent, ast.Constant):
                count = int(exponent.value)
            else:
                # The reverse rule uses the compile-time integer exponent - 1.
                # It is not a tensor operation or a node in the scalar graph.
                assert isinstance(exponent, ast.BinOp) and isinstance(exponent.op, ast.Sub)
                count = int(float.fromhex(variables[exponent.left.id].value)) - int(exponent.right.value)
            return graph.power(value, count)
        if name in ('vector_norm', 'clamp_min'):
            return graph.node(
                'norm' if name == 'vector_norm' else 'clamp',
                *(lower_expression(graph, arg, variables) for arg in expression.args),
            )
    raise ValueError(f'Unsupported scalar CUDA expression: {ast.dump(expression)}')


class ScalarNode:
    __slots__ = ('index', 'op', 'args', 'value')

    def __init__(self, index, op, args, value):
        self.index, self.op, self.args, self.value = index, op, args, value


class ScalarGraph:
    def __init__(self):
        self.nodes = []
        self.cache = {}
        self.zero = self.constant(0.0)
        self.one = self.constant(1.0)

    def node(self, op, *args, value=None):
        # Raw nodes retain tensor-operation rounding and order. The analytic
        # polynomial helpers below may instead fold Python-number constants.
        key = (op, tuple(n.index for n in args), value)
        if key not in self.cache:
            n = ScalarNode(len(self.nodes), op, args, value)
            self.cache[key] = n
            self.nodes.append(n)
        return self.cache[key]

    def constant(self, value):
        return self.node('constant', value=float(value).hex())

    def input(self, role, column):
        return self.node('input', value=(role, column))

    def binary(self, op, a, b):
        if a.op == b.op == 'constant':
            x, y = float.fromhex(a.value), float.fromhex(b.value)
            if op == 'add':
                return self.constant(x + y)
            if op == 'sub':
                return self.constant(x - y)
            if op == 'mul':
                return self.constant(x * y)
            if op == 'div':
                return self.constant(x / y)
        return self.node(op, a, b)

    def neg(self, a):
        return self.constant(-float.fromhex(a.value)) if a.op == 'constant' else self.node('neg', a)

    def power(self, a, exponent):
        if exponent == 0:
            return self.one
        if exponent == 1:
            return a
        if a.op == 'constant':
            return self.constant(float.fromhex(a.value) ** exponent)
        return self.node('pow', a, value=exponent)

    @staticmethod
    def ancestors(outputs):
        visited = set()

        for root in outputs:
            pending = [root]
            while pending:
                node = pending.pop()
                if node.index in visited:
                    continue
                visited.add(node.index)
                pending.extend(reversed(node.args))
        return visited

    def accumulate_gradient(self, node_gradients, node, value):
        if node.op == 'constant':
            return
        node_gradients[node.index] = (
            self.binary('add', node_gradients[node.index], value)
            if node.index in node_gradients else value
        )


class HarmonicsScalarizer(Scalarizer):
    unsupported_operator = 'spherical-harmonic'

    def slice_fields(self, shape, value, dimension, start, end, step):
        dimension %= len(value.shape)
        start, end, step = slice(start, end, step).indices(value.shape[dimension])
        if dimension not in value.axes:
            if (start, end, step) != (0, value.shape[dimension], 1):
                raise ValueError('Slicing batch coordinates cannot be represented by a scalar row')
            return ScalarValue(shape, value.axes, value.fields)
        return ScalarValue(
            shape,
            value.axes,
            [
                value.at({**item, dimension: start + item[dimension] * step})
                for item in self.indices(shape, value.axes)
            ],
        )

    def backward_view(self, shape, value, dimension, *, index=None, start=None, end=None, step=1):
        dimension %= len(shape)
        if index is not None:
            axes = tuple(sorted((*[axis + (axis >= dimension) for axis in value.axes], dimension)))
        else:
            axes = value.axes
            start, end, step = slice(start, end, step).indices(shape[dimension])
            if dimension not in axes:
                if (start, end, step) != (0, shape[dimension], 1):
                    raise ValueError('A batch slice backward needs cross-row materialization')
                return ScalarValue(shape, axes, value.fields)
        fields = []
        for item in self.indices(shape, axes):
            column = item[dimension]
            if index is not None:
                active = column == index
                coordinates = {
                    axis - (axis > dimension): i for axis, i in item.items() if axis != dimension
                }
            else:
                active = start <= column < end and (column - start) % step == 0
                coordinates = {**item, dimension: (column - start) // step}
            fields.append(value.at(coordinates) if active else self.graph.zero)
        return ScalarValue(shape, axes, fields)

    def lower(self, traced, names):
        result = super().lower(traced, names, INPUT_ROLES)
        for name, fields in result.items():
            if fields is None:
                raise ValueError(f'Active spherical-harmonic target {name} has no Torch gradient')
            if name in ('dx', 'sx', 'tx', 'trx') and len(fields) != 3:
                raise ValueError(f'Invalid scalarized output {name}: {len(fields)} components')
        return result
