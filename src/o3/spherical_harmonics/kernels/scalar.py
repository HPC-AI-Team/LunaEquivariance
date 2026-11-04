"""Ordered scalar arithmetic shared by Cartesian harmonics and their gradients."""

from __future__ import annotations
import ast
from functools import lru_cache
import math


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
            return graph.power(value, int(expression.args[1].value))
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
        # Share identical scalar operations; binary/power fold constants and
        # algebraic identities before CUDA emission.
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
        if op == 'add':
            if a is self.zero:
                return b
            if b is self.zero:
                return a
        elif op == 'mul':
            if a is self.zero or b is self.zero:
                return self.zero
            if a is self.one:
                return b
            if b is self.one:
                return a
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



@lru_cache(maxsize=None)
def harmonic_derivative(degree, component, powers):
    """Cartesian derivative as a sparse row of lower solid harmonics."""
    if sum(powers) == 0:
        return ((degree, component, 1.0),)
    if degree < sum(powers):
        return ()
    axis = next(i for i, count in enumerate(powers) if count)
    remaining = tuple(count - int(i == axis) for i, count in enumerate(powers))
    m = component - degree
    q = (2 * degree + 1) / (2 * degree - 1)
    terms = []
    if axis == 1:
        if abs(m) < degree:
            terms.append((m, math.sqrt(q * (degree * degree - m * m))))
    elif m == 0:
        value = -math.sqrt(q * degree * (degree - 1) / 2)
        terms.append((-1 if axis == 0 else 1, value))
    else:
        k = abs(m)
        lower = math.sqrt(q * (degree + k) * (degree + k - 1)) / 2
        upper = math.sqrt(q * (degree - k) * (degree - k - 1)) / 2
        if axis == 2:
            if k > 1:
                terms.append(((k - 1) if m > 0 else -(k - 1), lower))
            elif m > 0:
                terms.append((0, lower * math.sqrt(2)))
            terms.append(((k + 1) if m > 0 else -(k + 1), -upper))
        elif m > 0:
            if k > 1:
                terms.append((-(k - 1), -lower))
            terms.append((-(k + 1), -upper))
        else:
            terms.append(((k - 1), lower * (math.sqrt(2) if k == 1 else 1)))
            terms.append(((k + 1), upper))
    result = {}
    for target, coefficient in terms:
        if coefficient == 0 or abs(target) >= degree:
            continue
        for l, column, value in harmonic_derivative(degree - 1, target + degree - 1, remaining):
            key = (l, column)
            result[key] = result.get(key, 0.0) + coefficient * value
    return tuple((l, column, value) for (l, column), value in sorted(result.items()) if value != 0)
