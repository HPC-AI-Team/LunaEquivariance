"""Registered per-irrep expressions and their scalar graph lowering.

Tracing happens only for a new dtype and requested derivative signature. The
original Torch operations preserve e3nn's rounding and threshold behavior.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
from threading import Lock

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ...activation.kernels.utils import apply_activation_spec
from .utils import Graph, ROLES, OUTPUTS, INPUT_ROLES
from ....utils.torch_scalar import gradient, ScalarValue, Scalarizer
from ....o3.norm.kernels.utils import norm_product


TRACE_LOCK = Lock()
TRACE_EXECUTOR = None


class RegisteredScalarizer(Scalarizer):
    """Extend structural lowering with registered nonlinear scalar atoms."""

    _ATOMS = (
        "abs",
        "sgn",
        "relu",
        "threshold_backward",
        "silu",
        "silu_backward",
        "sigmoid",
        "sigmoid_backward",
        "tanh",
        "tanh_backward",
        "softplus",
        "softplus_backward",
        "gelu",
        "gelu_backward",
        "exp",
        "cosh",
        "sinh",
        "erf",
        "sin",
        "cos",
        "log",
        "log1p",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "expm1",
    )

    def __init__(self, graph, dtype):
        super().__init__(graph, dtype)
        self.aliases = {}

    def resolve(self, value):
        result = super().resolve(value)
        if isinstance(result, ScalarValue) and id(result) in self.aliases:
            owner, indices = self.aliases[id(result)]
            result.fields = tuple(owner.fields[index] for index in indices)
        return result

    def alias_fields(self, result, original, indices):
        owner, parent_indices = self.aliases.get(
            id(original), (original, tuple(range(len(original.fields))))
        )
        self.aliases[id(result)] = owner, tuple(parent_indices[index] for index in indices)
        return result

    def assign_fields(self, value, fields):
        fields = tuple(fields)
        value.fields = fields
        if id(value) in self.aliases:
            owner, indices = self.aliases[id(value)]
            updated = list(owner.fields)
            for index, field in zip(indices, fields):
                updated[index] = field
            owner.fields = tuple(updated)
        return value

    def construct_fields(self, shape, fill):
        axes = tuple(range(1 if len(shape) > 1 else 0, len(shape)))
        return ScalarValue(shape, axes, [fill] * math.prod(shape[axis] for axis in axes))

    def slice_fields(self, shape, value, dimension, start, end, step):
        result = super().slice_fields(shape, value, dimension, start, end, step)
        dimension %= len(value.shape)
        if dimension not in value.axes:
            return self.alias_fields(result, value, range(len(value.fields)))
        start, end, step = slice(start, end, step).indices(value.shape[dimension])
        indices = []
        for coordinates in self.indices(shape, result.axes):
            coordinates = {**coordinates, dimension: start + coordinates[dimension] * step}
            offset = 0
            for axis in value.axes:
                offset = offset * value.shape[axis] + coordinates.get(axis, 0)
            indices.append(offset)
        return self.alias_fields(result, value, indices)

    def operation(self, node):
        target, args, kwargs = str(node.target), node.args, node.kwargs
        shape = tuple(node.meta['val'].shape)
        name = target.split(".")[1] if target.startswith("aten.") else ""
        if target == "prims.add.default":
            return self.elementwise(shape, args[:2], partial(self.graph.node, "add"))
        if name == "pow" and args[1] in (0, 1):
            return self.elementwise(
                shape, args[:1], partial(self.power, exponent=args[1])
            )
        if name in self._ATOMS or target.startswith("aten.pow.Tensor_Scalar"):
            params = {}
            operands = args
            if name in ("softplus", "softplus_backward"):
                count = 1 if name == "softplus" else 2
                params = {
                    "beta": args[count] if len(args) > count else kwargs.get("beta", 1),
                    "threshold": (
                        args[count + 1] if len(args) > count + 1 else kwargs.get("threshold", 20)
                    ),
                }
                operands = args[:count]
            elif name in ("gelu", "gelu_backward"):
                count = 1 if name == "gelu" else 2
                params = {
                    "approximate": (
                        args[count] if len(args) > count else kwargs.get("approximate", "none")
                    )
                }
                operands = args[:count]
            elif name == "threshold_backward":
                params, operands = {"threshold": args[2]}, args[:2]
            elif name == "pow":
                params, operands = {"exponent": args[1]}, args[:1]
            return self.elementwise(
                shape,
                operands,
                partial(self.graph.node, target, value=tuple(sorted(params.items()))),
            )
        if name in ("lt", "le", "gt", "ne"):
            return self.elementwise(shape, args[:2], partial(self.graph.node, name))
        if name == "_to_copy":
            return self.elementwise(shape, args[:1], partial(self.graph.node, "cast"))
        if name in ("empty_like", "new_empty", "new_empty_strided", "empty", "empty_strided"):
            return self.construct_fields(shape, self.graph.node("uninitialized"))
        if name == "_efficientzerotensor":
            return self.construct_fields(shape, self.graph.zero)
        if name in ("new_zeros", "new_ones"):
            return self.construct_fields(
                shape, self.graph.zero if name == "new_zeros" else self.graph.one
            )
        if name in ("zeros", "ones", "full"):
            fill = (
                self.constant(args[1])
                if name == "full"
                else self.graph.one if name == "ones" else self.graph.zero
            )
            return self.construct_fields(shape, fill)
        if name == "fill_":
            value = self.resolve(args[0])
            return self.assign_fields(value, (self.constant(args[1]),) * len(value.fields))
        if name in ("copy", "copy_"):
            result = self.elementwise(shape, args[1:2], lambda value: value)
            return (
                self.assign_fields(self.resolve(args[0]), result.fields)
                if name == "copy_"
                else result
            )
        if name == "permute":
            value, permutation = self.resolve(args[0]), tuple(args[1])
            axes = tuple(sorted(permutation.index(axis) for axis in value.axes))
            fields, indices = [], []
            for item in self.indices(shape, axes):
                coordinates = {permutation[axis]: index for axis, index in item.items()}
                offset = 0
                for axis in value.axes:
                    offset = offset * value.shape[axis] + coordinates.get(axis, 0)
                fields.append(value.fields[offset])
                indices.append(offset)
            return self.alias_fields(ScalarValue(shape, axes, fields), value, indices)
        if name in ("view", "_unsafe_view", "reshape", "unsqueeze", "squeeze"):
            value = self.resolve(args[0])
            width = len(value.fields)
            start_min = 1 if width == 0 and shape else 0
            axes = next(
                (
                    tuple(range(start, len(shape)))
                    for start in range(start_min, len(shape) + 1)
                    if math.prod(shape[start:]) == width
                ),
                None,
            )
            if axes is None:
                raise ValueError(
                    f"A view crosses scalar-row batch coordinates: {value.shape} -> {shape}"
                )
            return self.alias_fields(ScalarValue(shape, axes, value.fields), value, range(width))
        result = super().operation(node)
        if name.endswith("_") and isinstance(result, ScalarValue):
            fields = tuple(result.fields)
            self.assign_fields(self.resolve(args[0]), fields)
        return result


class NormActivationScalarizer(RegisteredScalarizer):
    def operation(self, node):
        target, args = str(node.target), node.args
        shape = tuple(node.meta['val'].shape)
        name = target.split(".")[1] if target.startswith("aten.") else ""
        if name == "sqrt":
            return self.elementwise(shape, args[:1], partial(self.graph.node, "sqrt"))
        if name == "transpose":
            value = self.resolve(args[0])
            rank = len(value.shape)
            permutation = list(range(rank))
            a, b = args[1] % rank, args[2] % rank
            permutation[a], permutation[b] = permutation[b], permutation[a]
            axes = tuple(sorted(permutation.index(axis) for axis in value.axes))
            fields = []
            indices = []
            for item in self.indices(shape, axes):
                coordinates = {permutation[axis]: index for axis, index in item.items()}
                offset = 0
                for axis in value.axes:
                    offset = offset * value.shape[axis] + coordinates.get(axis, 0)
                fields.append(value.fields[offset])
                indices.append(offset)
            return self.alias_fields(ScalarValue(shape, axes, fields), value, indices)

        if name == "bmm":
            left, right = self.resolve(args[0]), self.resolve(args[1])
            if len(shape) != 3 or any(0 in value.axes for value in (left, right)):
                raise ValueError("Norm BMM must preserve independent row batches")
            width = left.shape[2]
            fields = []
            for row in range(shape[1]):
                for column in range(shape[2]):
                    acc = self.graph.zero
                    for k in range(width):
                        acc = self.graph.node(
                            "fma", left.at({1: row, 2: k}), right.at({1: k, 2: column}), acc
                        )
                    fields.append(acc)
            return ScalarValue(shape, (1, 2), fields)

        if name in ("index_put", "index_put_"):
            if len(args[1]) != 1 or args[1][0] is None or (len(args) > 3 and args[3]):
                raise ValueError("Norm requires nonaccumulating elementwise boolean index updates")
            value = self.resolve(args[0])
            mask = self.resolve(args[1][0])
            if mask.shape != value.shape:
                raise ValueError("Norm mask shape must match its norm")
            result = self.elementwise(
                shape,
                (args[0], args[1][0], args[2]),
                lambda old, condition, new: self.graph.node("where", condition, new, old),
            )
            return self.assign_fields(value, result.fields) if name == "index_put_" else result
        return super().operation(node)


def trace_scalar_product(components, scalings, features):
    shape = torch.broadcast_tensors(scalings[..., :1], features[..., :1])[0].shape[:-1]
    left = scalings.expand(shape + (-1,)).reshape(-1, 1)
    right = features.expand(shape + (-1,)).reshape(-1, components)
    rows = left.shape[0]
    right = right.reshape(rows, 1, components)
    if components == 1:
        right = right.reshape(rows, 1)
        value = torch.einsum("ba,ba->ba", right, left)
    else:
        value = torch.einsum("cba,cb->cba", right, left)
    value = value.reshape(rows, components)
    return value.reshape(shape + (components,))


def trace_program(*values, copy_meta, order, names, available, grad_mode):
    # This is the one boundary between the native role ABI and the formulas.
    (
        x, bias, grad_output, input_direction, bias_direction,
        input_gradient, bias_gradient, output_gradient,
    ) = values
    components = copy_meta["components"]
    q = norm_product(components, x, x)
    if copy_meta["kind"] == "norm":
        output = q if copy_meta["squared"] else q.relu().sqrt()
    else:
        if not copy_meta.get("norm_squared", True):
            q = q.relu().sqrt()
        if copy_meta["eps_squared"] > 0:
            q[q < copy_meta["eps_squared"]] = copy_meta["eps_squared"]
            q = q.sqrt()
        argument = q + bias if copy_meta["bias"] else q
        scaling = apply_activation_spec(argument, copy_meta["activation"])
        if copy_meta["normalize"]:
            scaling = scaling / q
        output = trace_scalar_product(components, scaling, x)
    if order == 0:
        return (output,)

    has_bias = copy_meta["bias"]
    first_names = ("dx", "db") if has_bias else ("dx",)
    first_inputs = (x, bias) if has_bias else (x,)
    first = gradient(
        (output,), first_inputs, (grad_output,),
        create_graph=grad_mode if order == 1 else True,
    )
    if order == 1:
        first_values = dict(zip(first_names, first))
        return tuple(first_values.get(name) for name in names)

    if order == 2:
        targets = {"sx": x, "sb": bias, "sg": grad_output}
        input_seed = input_direction if "rx" in available else None
        bias_seed = bias_direction if "rb" in available else None
        directions = (input_seed, bias_seed) if has_bias else (input_seed,)
        return gradient(
            first, tuple(targets[name] for name in names), directions, create_graph=False
        )

    keep_input_direction, keep_bias_direction = copy_meta["lower_seeds"]
    # Direction targets remain differentiable even if their value was pruned.
    directions = (
        (input_direction if keep_input_direction else None,
         bias_direction if keep_bias_direction else None)
        if has_bias else (input_direction if keep_input_direction else None,)
    )
    second_targets = (x, bias, grad_output) if has_bias else (x, grad_output)
    second = gradient(first, second_targets, directions, create_graph=True)

    input_seed = input_gradient if "bx" in available else None
    bias_seed = bias_gradient if "bb" in available else None
    output_seed = output_gradient if "bg" in available else None
    gradients = (
        (input_seed, bias_seed, output_seed)
        if has_bias else (input_seed, output_seed)
    )
    targets = {
        "tx": x, "tb": bias, "tg": grad_output,
        "trx": input_direction, "trb": bias_direction,
    }
    return gradient(
        second, tuple(targets[name] for name in names), gradients, create_graph=False
    )


def build_registered_graph(meta, order, names, available, *, grad_mode=False, lower_seeds=None):
    global TRACE_EXECUTOR
    with TRACE_LOCK:
        if TRACE_EXECUTOR is None:
            TRACE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="norm-metadata")
        meta = {
            **meta,
            "lower_seeds": (
                (True, bool(meta["bias"])) if lower_seeds is None else tuple(lower_seeds)
            ),
        }
        return TRACE_EXECUTOR.submit(
            assemble_registered_graph, meta, order, names, available, grad_mode
        ).result()


def assemble_registered_graph(meta, order, names, available, grad_mode):
    graph = Graph()
    fields = {name: [] for name in names}
    local_cache = {}
    for path in meta["paths"]:
        components = path["components"]
        if components not in local_cache:
            copy_meta = {**meta, "components": components}

            run = partial(
                trace_program, copy_meta=copy_meta, order=order, names=names,
                available=available, grad_mode=grad_mode,
            )

            copy_graph = Graph()
            inputs = []
            for role in ROLES:
                if role in ("x", "rx", "bx"):
                    width = copy_meta["components"]
                elif role in ("bias", "rb", "bb") or copy_meta["kind"] == "norm":
                    width = 1
                else:
                    width = copy_meta["components"]
                inputs.append(
                    torch.empty(
                        (2, width), device="meta", dtype=copy_meta["dtype"], requires_grad=role in ROLES[:5]
                    )
                )
            with torch.enable_grad():
                traced = make_fx(run, tracing_mode="fake")(
                    *inputs
                )
            local_cache[components] = (
                copy_graph,
                NormActivationScalarizer(copy_graph, copy_meta["dtype"]).lower(traced, names, ROLES),
            )

        local, roots = local_cache[components]
        for copy in range(path["copies"]):
            memo = {}

            for name in names:
                if roots[name] is None:
                    fields[name] = None
                elif fields[name] is not None:
                    for root in roots[name]:
                        pending = [(root, False)]
                        while pending:
                            node, ready = pending.pop()
                            if node.index in memo:
                                continue
                            if not ready:
                                pending.append((node, True))
                                pending.extend((arg, False) for arg in reversed(node.args))
                                continue
                            value = node.value
                            if node.op == "input":
                                role, column = value
                                vector = role in ("x", "rx", "bx") or (
                                    role in ("grad_out", "bg") and meta["kind"] == "norm_activation"
                                )
                                value = (
                                    role,
                                    path["input_offset"] + copy * components + column
                                    if vector else path["scalar_offset"] + copy + column,
                                )
                            memo[node.index] = graph.node(
                                node.op, *(memo[arg.index] for arg in node.args), value=value
                            )
                        fields[name].append(memo[root.index])
    return (
        graph,
        {name: None if values is None else tuple(values) for name, values in fields.items()},
    )


def build_program(meta, order, *, needs=None, available=None, grad_mode=False, lower_seeds=None):
    if order not in range(4):
        raise ValueError("Norm computation order must be 0..3")
    names = OUTPUTS[order]
    needs = (True,) * len(names) if needs is None else tuple(needs)
    if len(needs) != len(names):
        raise ValueError("Norm output request count is incorrect")
    available = INPUT_ROLES[order] if available is None else tuple(available)
    if set(available).difference(INPUT_ROLES[order]):
        raise ValueError("Unknown Norm input")
    selected = tuple(name for name, wanted in zip(names, needs) if wanted)
    program = dict(
        order=order,
        grad_mode=grad_mode,
        outputs=names,
        inputs=(),
        active_outputs=(),
        zero_outputs=(),
        lower_seeds=lower_seeds,
    )
    if not selected or not meta["paths"]:
        return program

    graph, fields = build_registered_graph(
        meta, order, selected, available, grad_mode=grad_mode, lower_seeds=lower_seeds
    )
    active = []
    zero = []
    dependencies = set()
    for name in selected:
        if fields[name] is None:
            continue
        ancestors = graph.ancestors(fields[name])
        live = {
            node.value[0] for node in graph.nodes if node.index in ancestors and node.op == "input"
        }
        if live.issubset(available):
            active.append(name)
            dependencies.update(live)
            if not live:
                zero.append(name)

    program.update(
        graph=graph,
        fields=fields,
        inputs=tuple(role for role in ROLES if role in dependencies),
        active_outputs=tuple(active),
        zero_outputs=tuple(zero),
    )
    return program
