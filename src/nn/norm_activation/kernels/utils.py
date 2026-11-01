"""Per-irrep CUDA addressing and bounded execution caches."""

from functools import partial
from math import prod, isfinite

import torch

from ....utils.precision import float_literal
from ...activation.kernels.scalar import scalar_literal, emit_activation_aten_scalar

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from .graph import (
    OUTPUTS, INPUT_ROLES, VECTOR_INPUTS, VECTOR_OUTPUTS, BIAS_OUTPUTS,
    build_program, build_irrep_program,
)


MAX_SHARED_BYTES = 48 * 1024
FORWARD_BLOCK_SIZE = 128
FORWARD_MIN_BLOCKS = 12

COMPUTATION_NAMES = ('forward', 'backward', 'double_backward', 'triple_backward')

CUDA_HEADERS = (
    "#include <torch/extension.h>",
    "#include <c10/cuda/CUDAGuard.h>",
    "#include <c10/cuda/CUDAException.h>",
    "#include <ATen/cuda/CUDAContext.h>",
    "#include <cuda_runtime.h>",
    "#include <algorithm>",
)


def output_width(meta, name):
    if name in BIAS_OUTPUTS:
        return meta["bias_dim"]
    if name in ("out", "sg", "tg"):
        return meta["out_dim"]
    return meta["in_dim"]


def input_width(meta, role):
    if role in ("bias", "rb", "bb"):
        return meta["bias_dim"]
    if role in ("grad_out", "bg"):
        return meta["out_dim"]
    return meta["in_dim"]


def emit_copy_program(
    meta, program, *, strided=False, generic=False, save_bias=False,
    shared_outputs=False, expression_source=None,
):
    """Emit each irrep copy in scalar-graph order."""
    count = sum(path["copies"] for path in meta["paths"])
    lines = [f"const Index row=item/{count};"]
    if generic:
        lines.append("const int64_t x_row=forward_row_offset(row,x_layout);")
    lines.append(f"const Index group=item%{count};")
    if save_bias:
        address = "bias[(group)*bias_stride]" if strided or generic else "bias[group]"
        lines.append(f"if(row==0) saved_state[group]={address};")

    copy_programs = []
    atoms = []
    begin = 0
    for path in meta["paths"]:
        components, copies = path["components"], path["copies"]
        if not copies:
            continue

        local = build_irrep_program(meta, program, path, check_copies=True)
        lines.extend((
            f"if(group>={begin} && group<{begin + copies}) {{",
            f"    const Index copy=group-{begin};",
        ))

        input_address = partial(
            copy_input_address, meta=meta, path=path, components=components,
            strided=strided, generic=generic,
        )
        output_address = partial(
            copy_output_address, meta=meta, path=path, components=components,
            shared=shared_outputs and components * (8 if meta["dtype"] == torch.float64 else 4) > 16,
        )

        local_lines, local_atoms = emit_scalar_lines(
            meta["dtype"], local, input_address, output_address,
            expression_source=expression_source,
        )
        lines.extend((*("    " + line for line in local_lines), "}"))
        if shared_outputs:
            copy_programs.append((path, local_lines))
        atoms.extend(local_atoms)
        begin += copies
    return lines, atoms, count, copy_programs


def copy_input_address(role, column, *, meta, path, components, strided, generic):
    vector = role in VECTOR_INPUTS or (
        role in ("grad_out", "bg") and meta["kind"] == "norm_activation"
    )
    start = path["input_offset"] if vector else path["scalar_offset"]
    step = components if vector else 1
    width = input_width(meta, role)
    if role == "bias":
        if strided or generic:
            return f"bias[({start}+copy*{step}+{column})*bias_stride]"
        return f"bias[{start}+copy*{step}+{column}]"
    if generic:
        return f"x[x_row+({start}+copy*{step}+{column})*x_layout.feature_stride]"
    if strided:
        return f"x[row*x_row_stride+({start}+copy*{step}+{column})*x_col_stride]"
    return f"{role}[row*{width}+{start}+copy*{step}+{column}]"


def copy_output_address(name, column, *, meta, path, components, shared=False):
    if shared:
        return f"values[lane*{components}+{column}]"
    vector = (
        name in VECTOR_OUTPUTS
        or (name == "out" and meta["kind"] == "norm_activation")
        or (name in ("sg", "tg") and meta["kind"] == "norm_activation")
    )
    start = path["input_offset"] if vector else path["scalar_offset"]
    step = components if vector else 1
    return f"out_{name}[row*{output_width(meta, name)}+{start}+copy*{step}+{column}]"


def load_forward_backward(
    op, *, arch, index_bits, needs, available, strided=False, generic=False,
    save_bias=False, shared_bias=False,
):
    """Compile forward and both create_graph modes of backward together."""
    forward_available = tuple(role for role in available if role in INPUT_ROLES[0])
    forward_key = (0, None, forward_available, False, None)
    forward_program = op.programs.get(forward_key)
    if forward_program is None:
        forward_program = build_program(op.meta, 0, available=forward_available)
        op.programs[forward_key] = forward_program
    forward_source = op.forward_source
    if forward_source is None:
        from .forward import make_forward_source
        forward_source = make_forward_source
    backward_source = None if shared_bias else op.backward_source
    if backward_source is None:
        from .backward import make_backward_source
        backward_source = make_backward_source

    sources = {
        "forward": forward_source(
            op.meta, forward_program, strided=strided, generic=generic, save_bias=save_bias
        )
    }

    backward_available = (*forward_available, "grad_out")
    backward_bindings = []
    for grad_mode in (False, True):
        program_key = (
            1, tuple(needs) if needs is not None else None, backward_available, grad_mode, None
        )
        program = op.programs.get(program_key)
        if program is None:
            program = build_program(
                op.meta, 1, needs=needs, available=backward_available, grad_mode=grad_mode
            )
            op.programs[program_key] = program
        if not program["active_outputs"] or not program["inputs"]:
            continue
        source = backward_source(op.meta, program)
        name = "backward_create_graph" if grad_mode else "backward"
        if grad_mode and source == sources.get("backward"):
            name = "backward"
        else:
            sources[name] = source
        kernel_key = (*program_key, index_bits, arch)
        if shared_bias:
            kernel_key = ("shared_bias", *kernel_key)
        backward_bindings.append((kernel_key, name))
    precision = "fp64" if op.compute_dtype == torch.float64 else "fp32"
    extension = load_cuda_extension(
        f"{op.meta['kind']}_forward_backward_{precision}",
        combine_cuda_sources(sources), arch=arch, index_bits=index_bits,
        use_fast_math=False, minimal_headers=True,
    )
    for key, name in backward_bindings:
        op.kernels[key] = getattr(extension, name).launch
    while len(op.programs) > 128:
        op.programs.popitem(last=False)
    return extension.forward.launch


class NormActivationKernels:
    """Retain configuration and compiled launches for pending autograd contexts."""

    def __init__(
        self, *, meta, compute_dtype, index_bits, kernels, programs,
        forward_source=None, backward_source=None,
    ):
        self.meta = meta
        self.compute_dtype = compute_dtype
        self.index_bits = index_bits
        self.registered_forward = None
        self.last_execution_backend = None
        self.backward_backend = None
        self.kernels = kernels
        self.programs = programs
        self.forward_source = forward_source
        self.backward_source = backward_source

    def __getstate__(self):
        state = self.__dict__.copy()
        state["registered_forward"] = None
        return state

    def run(
        self, order, *inputs, needs=None, grad_mode=False, lower_seeds=None,
        save_bias=False, backward_needs=None,
    ):
        values = dict(zip(INPUT_ROLES[order], inputs))
        available = tuple(role for role, value in values.items() if value is not None)
        first = next(
            (value for role, value in values.items() if role != "bias" and value is not None), None
        )
        if first is None:
            return tuple(None for _ in OUTPUTS[order])
        batch = tuple(first.shape[:-1])
        n = prod(batch)
        # Public forward entries have already validated these operands.
        if order != 0:
            for role, value in values.items():
                if value is None:
                    continue
                shape = (self.meta["bias_dim"],) if role == "bias" else (*batch, input_width(self.meta, role))
                if (
                    value.dtype != self.compute_dtype
                    or tuple(value.shape) != shape
                ):
                    raise ValueError(
                        f"Norm operand {role} must have shape{shape} and the computation dtype"
                    )

        bound = n * max(self.meta["in_dim"], self.meta["out_dim"], self.meta["bias_dim"], 1) + 65535 * 128
        strided_forward = (
            order == 0
            and len(batch) <= 63
            and (
                not first.is_contiguous()
                or (values["bias"] is not None and not values["bias"].is_contiguous())
            )
        )
        generic_forward = strided_forward and len(batch) != 1
        if strided_forward and n:
            span = max(
                sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
                for value in values.values()
                if value is not None
            )
            bound = max(bound, span + 65535 * 128)
        if bound > (1 << 63) - 1:
            raise OverflowError("Norm offsets exceed signed int64 capacity")
        index_bits = 64 if bound > (1 << 31) - 1 else self.index_bits

        key = (
            order,
            tuple(needs) if needs is not None else None,
            available,
            grad_mode,
            lower_seeds,
        )
        program = self.programs.get(key)
        if program is None:
            program = build_program(
                self.meta,
                order,
                needs=needs,
                available=available,
                grad_mode=grad_mode,
                lower_seeds=lower_seeds,
            )
            self.programs[key] = program
            while len(self.programs) > 128:
                self.programs.popitem(last=False)

        if not self.meta["paths"] and order == 0:
            output = torch.zeros(
                (*batch, self.meta["out_dim"]), device=first.device, dtype=self.compute_dtype
            )
            return (output, values["bias"].clone()) if save_bias else output
        if not program["active_outputs"]:
            result = tuple(None for _ in program["outputs"])
        elif n == 0 or not program["inputs"]:
            result = tuple(
                (
                    torch.zeros(
                        (*batch, output_width(self.meta, name)),
                        device=first.device,
                        dtype=self.compute_dtype,
                    )
                    if name in program["active_outputs"]
                    else None
                )
                for name in program["outputs"]
            )
        else:
            arch = get_cuda_arch(first.device)
            key = (
                order,
                tuple(needs) if needs is not None else None,
                available,
                grad_mode,
                lower_seeds,
                index_bits,
                arch,
            )
            if order == 0:
                backward_needs = (
                    (True, self.meta["bias"]) if backward_needs is None else tuple(backward_needs)
                )
                key += (backward_needs,)
            if generic_forward:
                key += ("generic",)
            elif strided_forward:
                key += (True,)
            if save_bias:
                key += ("training_bias",)
            launch = self.kernels.get(key)
            if launch is None:
                if order in (0, 1):
                    forward_launch = load_forward_backward(
                        self, arch=arch, index_bits=index_bits,
                        needs=backward_needs if order == 0 else needs,
                        available=available, strided=strided_forward,
                        generic=generic_forward, save_bias=save_bias,
                        shared_bias=order == 0 and self.meta["kind"] == "norm_activation",
                    )
                    launch = forward_launch if order == 0 else self.kernels[key]
                else:
                    precision = "fp64" if self.meta["dtype"] == torch.float64 else "fp32"
                    source_builder = self.backward_source
                    if source_builder is None:
                        from .backward import make_backward_source
                        source_builder = make_backward_source
                    launch = load_cuda_extension(
                        f"{self.meta['kind']}_{COMPUTATION_NAMES[order]}_{precision}",
                        source_builder(self.meta, program), arch=arch, index_bits=index_bits,
                        use_fast_math=False, minimal_headers=True,
                    ).launch
                self.kernels[key] = launch
                while len(self.kernels) > 128:
                    self.kernels.popitem(last=False)

            identity_packed = order in (0, 1) and index_bits == 32 and len(batch) == 1
            if identity_packed:
                for identity_role in program["inputs"]:
                    identity_value = values[identity_role]
                    if not identity_value.is_contiguous() or identity_value.storage_offset() != 0:
                        identity_packed = False
                        break
                    if identity_role != "bias":
                        identity_width = input_width(self.meta, identity_role)
                        if (n == 1 or identity_width == 1) and identity_value.stride() != (
                            identity_width,
                            1,
                        ):
                            identity_packed = False
                            break
            operands = (
                [values[role] for role in program["inputs"]]
                if strided_forward or identity_packed
                else [
                    (
                        values[role].contiguous()
                        if role == "bias"
                        else values[role].reshape(n, input_width(self.meta, role)).contiguous()
                    )
                    for role in program["inputs"]
                ]
            )
            launched = launch(*operands, n)
            if save_bias:
                result = (launched[0].reshape(*batch, self.meta["out_dim"]), launched[1])
            else:
                result = tuple(
                    value.reshape(*batch, value.shape[-1]) if value is not None else None
                    for value in launched
                )

        if save_bias:
            # Empty batches have no kernel to copy the saved bias vector.
            return result[0], result[1] if len(result) > 1 else values["bias"].clone()
        return result[0] if order == 0 else result


def run_backward(op, order, inputs, *, needs, grad_mode=False, lower_seeds=None):
    """Launch the requested gradients with shared-bias reduction."""
    values = dict(zip(INPUT_ROLES[order], inputs))
    x = values["x"]
    batch = x.shape[:-1]
    n = prod(batch)
    available = tuple(role for role, value in values.items() if value is not None)
    key = (
        order,
        tuple(needs) if needs is not None else None,
        available,
        grad_mode,
        lower_seeds,
    )
    program = op.programs.get(key)
    if program is None:
        program = build_program(
            op.meta,
            order,
            needs=needs,
            available=available,
            grad_mode=grad_mode,
            lower_seeds=lower_seeds,
        )
        op.programs[key] = program
        if len(op.programs) > 128:
            op.programs.popitem(last=False)
    if not program["active_outputs"]:
        return (None,) * len(OUTPUTS[order])
    if not n or not program["inputs"]:
        return tuple(
            (
                x.new_zeros(
                    (op.meta["bias_dim"],)
                    if name in BIAS_OUTPUTS
                    else (*batch, output_width(op.meta, name))
                )
                if name in program["active_outputs"]
                else None
            )
            for name in OUTPUTS[order]
        )

    operands = []
    span = n * max(op.meta["in_dim"], op.meta["out_dim"], 1) + 65535 * 128
    for role in program["inputs"]:
        value = values[role]
        if value.dtype != op.compute_dtype:
            raise ValueError(f"NormActivation operand {role} must match the computation dtype")
        if role in ("bias", "rb", "bb"):
            if tuple(value.shape) != (op.meta["bias_dim"],):
                raise ValueError(f"NormActivation operand {role} must be a shared bias vector")
        else:
            if tuple(value.shape) != (*batch, input_width(op.meta, role)):
                raise ValueError(
                    f"NormActivation operand {role} must match the batch and feature shape"
                )
            value = value.reshape(n, input_width(op.meta, role))
        if not value.is_contiguous():
            span = max(
                span, sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())),
            )
        operands.append(value)
    if span > (1 << 63) - 1:
        raise OverflowError("NormActivation offsets exceed signed int64 capacity")

    index_bits = 64 if span > (1 << 31) - 1 else op.index_bits
    arch = get_cuda_arch(x.device)
    key = (
        "shared_bias",
        order,
        tuple(needs),
        available,
        grad_mode,
        lower_seeds,
        index_bits,
        arch,
    )
    launch = op.kernels.get(key)
    if launch is None:
        if order == 1:
            load_forward_backward(
                op, arch=arch, index_bits=index_bits, needs=needs,
                available=available, shared_bias=True,
            )
            launch = op.kernels[key]
        else:
            from .backward import make_backward_source

            precision = "fp64" if op.compute_dtype == torch.float64 else "fp32"
            launch = load_cuda_extension(
                f"norm_activation_{COMPUTATION_NAMES[order]}_{precision}_shared_bias",
                make_backward_source(op.meta, program),
                arch=arch,
                index_bits=index_bits,
                use_fast_math=False,
                minimal_headers=True,
            ).launch
        op.kernels[key] = launch
        while len(op.kernels) > 128:
            op.kernels.popitem(last=False)

    result = launch(*operands, n)
    return tuple(
        (
            value
            if name in BIAS_OUTPUTS
            else value.reshape(*batch, value.shape[-1])
            if value is not None
            else None
        )
        for name, value in zip(OUTPUTS[order], result)
    )


def emit_scalar_lines(
    dtype, program, input_address, output_address, *,
    expression_source=None, approximate_tanh=False,
):
    """Emit CUDA scalar expressions in graph order."""
    graph, fields = program["graph"], program["fields"]
    names = tuple(program["active_outputs"])
    if any(fields[name] is None for name in names):
        raise ValueError("Native program requested a structurally absent derivative")
    ancestors = graph.ancestors(tuple(field for name in names for field in fields[name]))
    nodes = [node for node in graph.nodes if node.index in ancestors]
    dependencies = {node.value[0] for node in nodes if node.op == "input"}
    if not dependencies.issubset(program["inputs"]):
        raise ValueError(
            f"Omitted live native operands: {dependencies.difference(program['inputs'])}"
        )
    scalar = "double" if dtype == torch.float64 else "float"
    prefix = "__d" if scalar == "double" else "__f"
    arithmetic = {name: prefix + name + "_rn" for name in ("add", "sub", "mul", "div")}

    lines, atoms = [], []
    for node in nodes:
        if node.op == "constant":
            continue
        arguments = []
        for argument in node.args:
            if argument.op != "constant":
                arguments.append(f"v{argument.index}")
            else:
                value = float.fromhex(argument.value)
                arguments.append(
                    float_literal(value, dtype) if isfinite(value)
                    else scalar_literal(value, dtype)
                )
        args = tuple(arguments)
        type_ = "bool" if node.op in ("eq", "ne", "lt", "le", "gt", "ge") else "Element"
        # Use the caller's rounding rules when supplied.
        expression = expression_source(node, args, dtype) if expression_source is not None else None
        if expression is None:
            if node.op == "input":
                role, column = node.value
                expression = input_address(role, column)
            elif node.op in arithmetic:
                expression = f"{arithmetic[node.op]}({args[0]},{args[1]})"
            elif node.op == "fma":
                expression = (
                    f"{'__fma_rn' if dtype == torch.float64 else '__fmaf_rn'}({','.join(args)})"
                )
            elif node.op == "sqrt":
                expression = f"{prefix}sqrt_rn({args[0]})"
            elif node.op == "neg":
                expression = f"(-{args[0]})"
            elif node.op == "cast":
                expression = f"Element({args[0]})"
            elif node.op in ("eq", "ne", "lt", "le", "gt", "ge"):
                symbol = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[node.op]
                expression = f"({args[0]}{symbol}{args[1]})"
            elif node.op == "where":
                expression = f"({args[0]}?{args[1]}:{args[2]})"
            elif node.op.startswith("aten."):
                params = dict(node.value or ())
                expression = emit_activation_aten_scalar(
                    node.op, args, dtype=dtype, params=params,
                    approximate_tanh=approximate_tanh,
                )
            else:
                raise ValueError(f"Unsupported live native node: {node.op}")
        if node.op.startswith("aten."):
            atoms.append((node.op, dict(node.value or ())))
        lines.append(f"const {type_} v{node.index}={expression};")
    for name in names:
        for column, value in enumerate(fields[name]):
            destination = output_address(name, column)
            if value.op != "constant":
                expression = f"v{value.index}"
            else:
                constant = float.fromhex(value.value)
                expression = (
                    float_literal(constant, dtype) if isfinite(constant)
                    else scalar_literal(constant, dtype)
                )
            lines.append(f"{destination}={expression};")
    return lines, atoms

