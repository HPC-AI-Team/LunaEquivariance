"""Scalarize actual Torch graphs without retaining tensor values or FX graphs.

Feature fields retain Torch shape/stride metadata. Operations are lowered in
registered order; this representation is never differentiated symbolically.
"""

from __future__ import annotations

from functools import partial
import itertools
import math
from numbers import Number

import torch
from torch.fx import Node as FXNode


class ScalarValue:
    """A tensor's feature fields; batch coordinates never enter the DAG."""

    __slots__ = ('shape', 'axes', 'fields')

    def __init__(self, shape, axes, fields):
        self.shape = tuple(shape)
        self.axes = tuple(axes)
        self.fields = tuple(fields)
        expected = math.prod(self.shape[axis] for axis in self.axes)
        if len(self.fields) != expected:
            raise ValueError(f'Feature representation size mismatch: {self.shape}, {self.axes}')

    def at(self, indices):
        offset = 0
        for axis in self.axes:
            offset = offset * self.shape[axis] + indices.get(axis, 0)
        return self.fields[offset]


class Scalarizer:
    unsupported_operator = 'scalar-row'

    def __init__(self, graph, dtype):
        self.graph, self.dtype, self.values = graph, dtype, {}

    def constant(self, value):
        # Scalars converted to a Tensor first obey its dtype, before any
        # subsequent arithmetic (which must not be folded in Python double).
        return self.graph.constant(torch.tensor(value, dtype=self.dtype, device='cpu').item())

    @staticmethod
    def indices(shape, axes):
        return (
            dict(zip(axes, coordinates))
            for coordinates in itertools.product(*(range(shape[axis]) for axis in axes))
        )

    def resolve(self, value):
        return self.values[value] if isinstance(value, FXNode) else value

    def elementwise(self, shape, operands, operation):
        operands = tuple(self.resolve(value) for value in operands)
        axes = sorted(
            {
                axis + len(shape) - len(value.shape)
                for value in operands
                if isinstance(value, ScalarValue)
                for axis in value.axes
            }
        )
        fields = []
        for indices in self.indices(shape, axes):
            arguments = []
            for value in operands:
                if isinstance(value, ScalarValue):
                    shift = len(shape) - len(value.shape)
                    arguments.append(
                        value.at(
                            {
                                axis: 0 if value.shape[axis] == 1 else indices[axis + shift]
                                for axis in value.axes
                            }
                        )
                    )
                else:
                    arguments.append(self.constant(value))
            fields.append(operation(*arguments))
        return ScalarValue(shape, axes, fields)

    def slice_fields(self, shape, value, dimension, start, end, step):
        dimension %= len(value.shape)
        start, end, step = slice(start, end, step).indices(value.shape[dimension])
        if dimension not in value.axes:
            if dimension == 0 and (start, end, step) != (0, value.shape[dimension], 1):
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
                if dimension != 0:
                    # A broadcast scalar has no explicit feature axis until
                    # a slice backward scatters it into selected columns.
                    axes = tuple(sorted((*axes, dimension)))
                elif (start, end, step) != (0, shape[dimension], 1):
                    raise ValueError(
                        f'A batch slice backward needs cross-row materialization: output={shape}, value={value.shape}, axes={axes}, dimension={dimension}, slice={(start, end, step)}'
                    )
                else:
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

    def reduce_fields(self, node, value, dimensions, keepdim, *, norm=False):
        dimensions = (
            tuple(dimension % len(value.shape) for dimension in dimensions) if value.shape else ()
        )
        if not norm and (len(dimensions) != 1 or dimensions[0] not in value.axes):
            if any(d not in value.axes and value.shape[d] != 1 for d in dimensions):
                raise ValueError('Only a per-row feature reduction is supported')
            # Explicit small global parameter tensors may have several
            # feature axes; singleton axes carry no additional field.
            # Keep the original single-feature path below byte-for-byte
            # for registered normalization's layout-sensitive arithmetic.
            shape = tuple(node.meta['val'].shape)
            remaining = tuple(d for d in value.axes if d not in dimensions)
            axes = tuple(d if keepdim else d - sum(r < d for r in dimensions) for d in remaining)
            if keepdim:
                axes = tuple(sorted((*axes, *(d for d in dimensions if shape[d] == 1))))
            reduced = tuple(d for d in value.axes if d in dimensions)
            fields = []
            for coordinate in self.indices(shape, axes):
                old = {
                    d: coordinate[d if keepdim else d - sum(r < d for r in dimensions)]
                    for d in remaining
                }
                columns = [
                    value.at({**old, **indexes}) for indexes in self.indices(value.shape, reduced)
                ]
                result = columns[0] if columns else self.graph.zero
                for column in columns[1:]:
                    result = self.graph.node('add', result, column)
                fields.append(result)
            return ScalarValue(shape, axes, fields)
        if len(dimensions) != 1 or dimensions[0] not in value.axes:
            raise ValueError('Only a per-row feature reduction is supported')
        dimension = dimensions[0]
        shape = tuple(node.meta['val'].shape)
        axes = tuple(
            axis if keepdim else axis - (axis > dimension)
            for axis in value.axes
            if axis != dimension
        )
        if keepdim:
            axes = tuple(sorted((*axes, dimension)))
        operand = node.args[0].meta['val']
        operand_shape, operand_stride = tuple(operand.shape), tuple(operand.stride())
        batch_axes = [
            axis for axis, size in enumerate(operand_shape)
            if axis != dimension and size > 1
        ]
        # The last non-singleton batch axis fixes the reduction family.
        layout = (
            'xz_y' if not batch_axes or operand_stride[dimension] < operand_stride[batch_axes[-1]]
            else 'xy_z'
        )
        fields = []
        for item in self.indices(shape, axes):
            coordinates = {
                axis if keepdim else axis + (axis >= dimension): index
                for axis, index in item.items()
                if not (keepdim and axis == dimension)
            }
            columns = [
                value.at({**coordinates, dimension: column})
                for column in range(value.shape[dimension])
            ]
            if norm:
                if len(columns) != 3:
                    raise ValueError('Scalar-row normalization requires three coordinates')
                result = self.graph.node('norm', *columns, value=layout)
            else:
                if len(columns) == 3 and layout == 'xz_y':
                    columns = (columns[0], columns[2], columns[1])
                if not columns:
                    result = self.graph.zero
                else:
                    result = columns[0]
                    for column in columns[1:]:
                        result = self.graph.node('add', result, column)
            fields.append(result)
        return ScalarValue(shape, axes, fields)

    def binary(self, a, b, *, operation, alpha):
        if operation in ('add', 'sub') and alpha != 1:
            b = self.graph.node('mul', b, self.constant(alpha))
        return self.graph.node(operation, a, b)

    def operation(self, node):
        target, args, kwargs = str(node.target), node.args, node.kwargs
        shape = tuple(node.meta['val'].shape)
        unary_aliases = ('aten.alias.', 'aten.detach.')
        if target.startswith(unary_aliases):
            return self.resolve(args[0])
        if target.startswith(('aten.clone.', 'aten.contiguous.')):
            value = self.resolve(args[0])
            return ScalarValue(shape, value.axes, value.fields)
        if target.startswith(('aten.ones_like.', 'aten.zeros_like.', 'aten.full_like.')):
            value = self.resolve(args[0])
            fill = (
                1
                if target.startswith('aten.ones_like.')
                else 0
                if target.startswith('aten.zeros_like.')
                else args[1]
            )
            return ScalarValue(shape, value.axes, [self.constant(fill)] * len(value.fields))
        if target.startswith(('aten.scalar_tensor.', 'aten.ones.', 'aten.zeros.', 'aten.full.')):
            fill = (
                args[0]
                if target.startswith('aten.scalar_tensor.')
                else args[1]
                if target.startswith('aten.full.')
                else int(target.startswith('aten.ones.'))
            )
            # Such constructors are scalar or normalization's feature vector.
            axes = tuple(range(len(shape)))
            return ScalarValue(shape, axes, [self.constant(fill)] * math.prod(shape))
        if target.startswith('aten.select.int'):
            value, dimension, index = self.resolve(args[0]), args[1], args[2]
            dimension %= len(value.shape)
            if dimension not in value.axes:
                raise ValueError('Selecting a batch coordinate cannot be represented by a scalar row')
            axes = tuple(axis - (axis > dimension) for axis in value.axes if axis != dimension)
            return ScalarValue(
                shape,
                axes,
                [
                    value.at(
                        {
                            **{axis + (axis >= dimension): i for axis, i in item.items()},
                            dimension: index,
                        }
                    )
                    for item in self.indices(shape, axes)
                ],
            )
        if target.startswith('aten.slice.Tensor'):
            return self.slice_fields(
                shape,
                self.resolve(args[0]),
                args[1],
                args[2],
                args[3],
                args[4] if len(args) > 4 else 1,
            )
        if target.startswith('aten.select_backward.'):
            return self.backward_view(shape, self.resolve(args[0]), args[2], index=args[3])
        if target.startswith('aten.slice_backward.'):
            return self.backward_view(
                shape, self.resolve(args[0]), args[2], start=args[3], end=args[4], step=args[5]
            )
        if target.startswith('aten.stack.'):
            values = [self.resolve(value) for value in args[0]]
            dimension = args[1] if len(args) > 1 else 0
            dimension %= len(shape)
            axes = tuple(sorted((*[axis + (axis >= dimension) for axis in values[0].axes], dimension)))
            fields = [
                values[item[dimension]].at(
                    {axis - (axis > dimension): i for axis, i in item.items() if axis != dimension}
                )
                for item in self.indices(shape, axes)
            ]
            return ScalarValue(shape, axes, fields)
        if target.startswith('aten.cat.'):
            values = [self.resolve(value) for value in args[0]]
            dimension = args[1] if len(args) > 1 else 0
            dimension %= len(shape)
            if any(dimension not in value.axes for value in values):
                raise ValueError('Concatenating batches cannot be represented by a scalar row')
            fields = []
            for item in self.indices(shape, values[0].axes):
                column = item[dimension]
                for value in values:
                    if column < value.shape[dimension]:
                        fields.append(value.at({**item, dimension: column}))
                        break
                    column -= value.shape[dimension]
            return ScalarValue(shape, values[0].axes, fields)
        if target.startswith('aten.expand.'):
            return self.elementwise(shape, (args[0],), lambda value: value)
        if target.startswith('aten.linalg_vector_norm.'):
            if args[1] != 2:
                raise ValueError('Only Euclidean normalization is supported')
            return self.reduce_fields(node, self.resolve(args[0]), args[2], args[3], norm=True)
        if target.startswith('aten.sum.dim_IntList'):
            return self.reduce_fields(
                node, self.resolve(args[0]), args[1], args[2] if len(args) > 2 else False
            )
        if target.startswith('aten.clamp_min.'):
            return self.elementwise(shape, args[:2], partial(self.graph.node, 'clamp'))
        if target.startswith(('aten.ge.', 'aten.eq.')):
            operation = target.split('.')[1]
            return self.elementwise(shape, args[:2], partial(self.graph.node, operation))
        if target.startswith('aten.where.self'):
            return self.elementwise(
                shape, args[:3], partial(self.graph.node, 'where')
            )
        if target.startswith(('aten.masked_fill.', 'aten.masked_fill_.')):
            result = self.elementwise(
                shape, args[:3], lambda value, mask, fill: self.graph.node('where', mask, fill, value)
            )
            if target.startswith('aten.masked_fill_.'):
                value = self.resolve(args[0])
                value.fields = result.fields
                return value
            return result
        if target.startswith('aten.neg.'):
            return self.elementwise(shape, args[:1], partial(self.graph.node, 'neg'))
        if target.startswith('aten.pow.Tensor_Scalar'):
            exponent = args[1]
            if exponent not in (0, 1, 2):
                raise ValueError(f'Unexpected polynomial power: {exponent}')
            return self.elementwise(
                shape,
                args[:1],
                lambda value: self.graph.one if exponent == 0 else value if exponent == 1 else self.graph.node('mul', value, value),
            )
        operation = target.split('.')[1] if target.startswith('aten.') else ''
        if operation.rstrip('_') in ('add', 'sub', 'mul', 'div'):
            operation = operation.rstrip('_')
            if operation == 'div' and isinstance(args[1], Number):
                reciprocal = self.constant(1.0 / float(args[1]))
                result = self.elementwise(
                    shape, args[:1], partial(self.binary, b=reciprocal, operation='mul', alpha=1)
                )
            else:
                alpha = kwargs.get('alpha', args[2] if len(args) > 2 else 1)

                result = self.elementwise(
                    shape, args[:2], partial(self.binary, operation=operation, alpha=alpha)
                )
            if target.split('.')[1].endswith('_'):
                value = self.resolve(args[0])
                value.fields = result.fields
                return value
            return result
        raise ValueError(
            f'Unsupported {self.unsupported_operator} Torch graph operator: {target}, {args}, {kwargs}'
        )

    def lower(self, traced, names, roles):
        roles = iter(roles)
        outputs = None
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                role = next(roles)
                shape = tuple(node.meta['val'].shape)
                self.values[node] = ScalarValue(
                    shape,
                    (len(shape) - 1,),
                    [self.graph.input(role, column) for column in range(shape[-1])],
                )
            elif node.op == 'call_function':
                self.values[node] = self.operation(node)
            elif node.op == 'output':
                outputs = tuple(
                    self.resolve(value) if value is not None else None for value in node.args[0]
                )
            elif node.op == 'get_attr':
                tensor = getattr(traced, node.target)
                self.values[node] = ScalarValue(
                    tuple(tensor.shape),
                    tuple(range(tensor.ndim)),
                    [self.constant(value) for value in tensor.flatten().tolist()],
                )
            else:
                raise ValueError(f'Unsupported FX node kind: {node.op}')
        result = {}
        for name, value in zip(names, outputs):
            if value is None:
                result[name] = None
                continue
            if not value.axes and value.shape:
                value = ScalarValue(
                    value.shape, (len(value.shape) - 1,), value.fields * value.shape[-1]
                )
            if value.axes != (len(value.shape) - 1,):
                raise ValueError(f'Invalid scalarized output {name}: {value.shape}, {value.axes}')
            result[name] = value.fields
        return result


def gradient(outputs, inputs, seeds, *, create_graph):
    pairs = [
        (value, seed)
        for value, seed in zip(outputs, seeds)
        if value is not None and value.requires_grad and seed is not None
    ]
    if not pairs:
        return (None,) * len(inputs)
    return torch.autograd.grad(
        tuple(value for value, _ in pairs),
        inputs,
        tuple(seed for _, seed in pairs),
        allow_unused=True,
        retain_graph=True,
        create_graph=create_graph,
    )

