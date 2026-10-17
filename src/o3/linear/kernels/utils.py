"""Linear contraction selection, specialization cache, and CUDA execution."""

from collections.abc import Iterable, Sequence
import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension

TASK_ALIGNMENT = 32
MAX_BATCH_RANK = 63
MAX_WEIGHT_SPLITS = 128

LAYOUT_CACHE_LIMIT = 128

ORDER_INPUTS = (
    ('x', 'weight', 'bias'),
    ('x', 'weight', 'bias', 'grad_out'),
    ('x', 'weight', 'bias', 'grad_out', 'rx', 'rw', 'rb'),
    ('x', 'weight', 'bias', 'grad_out', 'rx', 'rw', 'rb', 'bx', 'bw', 'bb', 'bg'),
)


def build_program(
    order: int, *, needs: Sequence[bool] | None = None, available: Iterable[str] | None = None,
) -> dict[str, object]:
    """Select each order's requested outputs and the operands they read.

    Bias has no original-operand contribution after forward. Higher orders
    retain mixed derivatives of output seeds and external derivative directions.
    Missing directions remain None, rather than becoming zero allocations.
    """
    if order == 0:
        outputs = ('out',)
    elif order == 1:
        outputs = ('dx', 'dw', 'db')
        contractions = {
            'input': {'dx': (('grad_out', 'weight'),)},
            'weight': {'dw': (('x', 'grad_out'),)},
            'output': {},
            'bias': {'db': (('grad_out',),)},
        }
    elif order == 2:
        outputs = ('sx', 'sw', 'sb', 'sg')
        contractions = {
            'input': {'sx': (('grad_out', 'rw'),)},
            'weight': {'sw': (('rx', 'grad_out'),)},
            'output': {'sg': (('rx', 'weight'), ('x', 'rw'))},
            'broadcast': {'sg': (('rb',),)},
        }
    elif order == 3:
        outputs = ('tx', 'tw', 'tb', 'tg', 'trx', 'trw', 'trb')
        contractions = {
            'input': {
                'tx': (('bg', 'rw'),),
                'trx': (('grad_out', 'bw'), ('bg', 'weight')),
            },
            'weight': {
                'tw': (('rx', 'bg'),),
                'trw': (('bx', 'grad_out'), ('x', 'bg')),
            },
            'output': {'tg': (('bx', 'rw'), ('rx', 'bw'))},
            'bias': {'trb': (('bg',),)},
        }
    else:
        raise ValueError("Linear supports forward through triple-backward")

    inputs = ORDER_INPUTS[order]
    live = set(inputs if available is None else available)
    if order == 0:
        pairs = (("x", "weight"),) if {"x", "weight"} <= live else ()
        broadcast = (("bias",),) if "bias" in live else ()
        return {
            "order": 0,
            "outputs": outputs,
            "active_outputs": outputs,
            "zero_outputs": (),
            "inputs": tuple(role for role in inputs if role in live),
            "input": {},
            "weight": {},
            "bias": {},
            "output": {"out": pairs},
            "broadcast": {"out": broadcast},
        }
    if needs is None:
        needs = (True,) * len(outputs)
    if len(needs) != len(outputs):
        raise ValueError(f"Linear order {order} needs {len(outputs)} output flags")
    unknown = live.difference(inputs)
    if unknown:
        raise ValueError(f"Unknown Linear order {order} inputs: {sorted(unknown)}")

    selected = {output for output, needed in zip(outputs, needs) if needed}
    groups = {}
    for kind, targets in contractions.items():
        groups[kind] = {
            target: tuple(term for term in terms if all(role in live for role in term))
            for target, terms in targets.items()
            if target in selected
        }
        groups[kind] = {target: terms for target, terms in groups[kind].items() if terms}
    active = {target for targets in groups.values() for target in targets}
    dependencies = {
        role
        for targets in groups.values()
        for terms in targets.values()
        for term in terms
        for role in term
    }
    return {
        "order": order,
        "outputs": outputs,
        "active_outputs": tuple(output for output in outputs if output in active),
        "zero_outputs": tuple(
            output for output in outputs if output in selected and output not in active
        ),
        "inputs": tuple(role for role in inputs if role in dependencies),
        **groups,
    }


def select_program(op, order, *, needs=None, available=None):
    # Selection depends only on order/role presence, never live tensor
    # values, layout or dtype.
    programs = op._programs

    key = (
        order,
        tuple(needs) if needs is not None else None,
        tuple(available) if available is not None else None,
    )
    program = programs.get(key)
    if program is None:
        program = build_program(order, needs=needs, available=available)
        program["weight_terms"] = max((len(terms) for terms in program["weight"].values()), default=1)
        programs[key] = program
    return program


def run_kernel(op, order, *inputs, needs=None, num_items=None, backward_needs=None):
    if order not in range(4) or len(inputs) != len(ORDER_INPUTS[order]):
        raise ValueError("Linear derivative argument count is incorrect")

    values = dict(zip(ORDER_INPUTS[order], inputs))
    if order == 0 and not op.meta["generated_specs"]:
        # A bias-only graph uses features for shape, without reading or
        # converting its tensor. Remove that unused native pointer while
        # retaining the logical batch for allocation and derivatives.
        if num_items is None:
            num_items = values["x"].shape[0]
        values["x"] = values["weight"] = None
    available = tuple(role for role, tensor in values.items() if tensor is not None)
    program = select_program(op, order, available=available, needs=needs)
    if not program["inputs"]:
        return tuple(None for _ in program["outputs"])

    # Infer batch from any live feature, including higher-order seeds.
    n = num_items
    if n is None:
        n = next(
            (
                tensor.shape[0]
                for role, tensor in values.items()
                if tensor is not None and role in ("x", "grad_out", "rx", "bx", "bg")
            ),
            None,
        )
    if n is None:
        raise ValueError("A Linear derivative needs a feature tensor to determine its batch")
    if n < 0:
        raise ValueError("Linear batch size must be nonnegative")

    # Tensor addresses alone do not bound padded gradient tasks or split-K
    # workspaces. These bounds cover both native organizations and all
    # requested derivative roles. Also bound the final loop increment:
    # grid-stride kernels advance even after their last valid element.
    # External DW uses <= n*sum(ceil(path_weights/1024)) block tasks;
    # twice this count is covered by the padded weight span above.
    weight_terms = program["weight_terms"]
    bound = max(
        n * op.index_batch_span + op.meta["bias_numel"] + 3 * TASK_ALIGNMENT,
        weight_terms * op.index_workspace_span,
    )
    counter_bound = bound + 65535 * 256
    if counter_bound > (1 << 63) - 1:
        raise OverflowError(
            "Linear tensor, task or workspace offsets exceed signed int64 capacity"
        )

    first = values[program["inputs"][0]]
    arch = get_cuda_arch(first.device)
    index_bits = 64 if counter_bound > (1 << 31) - 1 else op.index_bits
    use_dx_dw_dg_db = False
    if order in (2, 3):
        max_components = op.max_components
        if op.meta["weight_numel"] <= 4096 and len(op.meta["generated_specs"]) <= 32 and max_components <= 17:
            if n <= 64:
                use_dx_dw_dg_db = op.meta["max_channels"] <= 64
            elif not (
                op.meta["shared_weights"]
                or op.meta["max_channels"] > 16
                or max_components > 7
                or op.meta["bias_numel"] > 0 and program.get("bias")
            ):
                consumers = (
                    (op.meta["in_dim"], set(program.get("input", ()))),
                    (op.meta["weight_numel"], set(program.get("weight", ()))),
                    (op.meta["out_dim"], set(program.get("output", ())) | set(program.get("broadcast", ()))),
                    (op.meta["bias_numel"], set(program.get("bias", ()))),
                )
                use_dx_dw_dg_db = sum(len(targets) for dimension, targets in consumers if dimension > 0) >= 2

    key = (
        order,
        tuple(needs) if needs is not None else None,
        available,
        arch,
        index_bits,
        use_dx_dw_dg_db,
        op.compute_dtype,
    )
    if order == 0:
        key += (backward_needs,)
    launch = op.kernels.get(key)
    if launch is None:
        from .forward import make_forward_source
        from .backward import make_backward_source
        from .double_backward import make_dx_dw_dg_db_source

        if order == 0:
            source = make_forward_source(op.meta, program, arch=arch)
        elif use_dx_dw_dg_db:
            source = make_dx_dw_dg_db_source(op.meta, program)
        else:
            source = make_backward_source(op.meta, program, arch=arch)
        precision = "fp64" if op.compute_dtype is torch.float64 else "fp32"
        if order == 2:
            name = "linear_double_bwd"
        elif order == 3 and use_dx_dw_dg_db:
            name = f"linear_triple_bwd_fused_{precision}"
        else:
            derivative = ("fwd", "bwd", "double_bwd", "triple_bwd")[order]
            name = f"linear_{derivative}_{precision}"
        if order == 0 and backward_needs is not None:
            backward_available = tuple(
                role for role, wanted in (
                    ("x", backward_needs[1]), ("weight", backward_needs[0]),
                    ("grad_out", True),
                ) if wanted
            )
            backward_program = select_program(
                op, 1, available=backward_available, needs=backward_needs
            )
            source = combine_cuda_sources({
                "forward": source,
                "backward": make_backward_source(op.meta, backward_program, arch=arch),
            })
            extension = load_cuda_extension(
                f"linear_forward_backward_{precision}", source,
                index_bits=index_bits, arch=arch, use_fast_math=False, minimal_headers=True,
            )
            backward_key = (
                1, backward_needs, backward_available, arch, index_bits, False, op.compute_dtype,
            )
            op.kernels[backward_key] = extension.backward.launch
            launch = extension.forward.launch
        else:
            extension = load_cuda_extension(
                name, source, index_bits=index_bits, arch=arch,
                use_fast_math=False, minimal_headers=True,
            )
            launch = extension.launch
        op.kernels[key] = launch
        if order == 0 and backward_needs is not None:
            op.kernels[key[:-1] + (None,)] = launch

    arguments = [values[role].contiguous() for role in program["inputs"]]
    result = launch(*arguments, n)
    return result[0] if order == 0 else result


def run_raw_forward(op, plan, *, backward_needs=None):
    roles = plan["roles"]
    first = (
        roles["x"] if op.meta["generated_specs"] or roles["bias"] is None else roles["bias"]
    )
    if not plan["n"] or not op.meta["out_dim"]:
        return torch.empty(
            (plan["n"], op.meta["out_dim"]), device=first.device, dtype=op.compute_dtype
        )

    available = tuple(
        role
        for role, value in roles.items()
        if value is not None and (role != "x" or op.meta["generated_specs"])
    )
    program = select_program(op, 0, available=available)
    arch = get_cuda_arch(first.device)

    scaled = []
    for role in ("x", "weight", "bias"):
        tensor = roles[role]
        if tensor is None or not tensor.numel():
            scaled.append(False)
            continue
        expected = tensor.stride()[-1]
        for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
            if size != 1 and stride != expected:
                scaled.append(False)
                break
            expected *= size
        else:
            scaled.append(True)
    scaled = tuple(scaled)

    key = (op.compute_dtype, available, arch, plan["index_bits"], plan["descriptor"], scaled, backward_needs)
    launch = op.raw_layout_kernels.get(key)
    if launch is None:
        from .forward import make_forward_source
        from .backward import make_backward_source

        source = make_forward_source(
            op.meta, program, arch=arch, f_in=op.f_in, f_out=op.f_out, scaled=scaled, layout=True
        )
        if backward_needs is not None:
            backward_available = tuple(
                role for role, wanted in (
                    ("x", backward_needs[1]), ("weight", backward_needs[0]),
                    ("grad_out", True),
                ) if wanted
            )
            backward_program = select_program(
                op, 1, available=backward_available, needs=backward_needs
            )
            source = combine_cuda_sources({
                "forward": source,
                "backward": make_backward_source(op.meta, backward_program, arch=arch),
            })
            extension = load_cuda_extension(
                "linear_forward_backward_raw_views", source,
                index_bits=plan["index_bits"], arch=arch,
                use_fast_math=False, minimal_headers=True,
            )

            # Backward uses contiguous saved operands; its index width depends on
            # contraction workspaces, independently of the original view strides.
            bound = max(
                plan["n"] * op.index_batch_span + op.meta["bias_numel"] + 3 * TASK_ALIGNMENT,
                op.index_workspace_span,
            ) + 65535 * 256
            backward_index_bits = 64 if bound > (1 << 31) - 1 else op.index_bits
            if backward_index_bits != plan["index_bits"]:
                backward_extension = load_cuda_extension(
                    "linear_forward_backward_raw_views", source,
                    index_bits=backward_index_bits, arch=arch,
                    use_fast_math=False, minimal_headers=True,
                )
            else:
                backward_extension = extension
            backward_key = (
                1, backward_needs, backward_available, arch, backward_index_bits,
                False, op.compute_dtype,
            )
            op.kernels[backward_key] = backward_extension.backward.launch
            launch = extension.forward.RawLayoutLauncher(plan["descriptor"]).launch
        else:
            extension = load_cuda_extension(
                "linear_forward_raw_views", source,
                index_bits=plan["index_bits"], arch=arch,
                use_fast_math=False, minimal_headers=True,
            )
            launch = extension.RawLayoutLauncher(plan["descriptor"]).launch
        op.raw_layout_kernels[key] = launch
        if backward_needs is not None:
            op.raw_layout_kernels[key[:-1] + (None,)] = launch
        while len(op.raw_layout_kernels) > LAYOUT_CACHE_LIMIT:
            op.raw_layout_kernels.popitem(last=False)
    else:
        op.raw_layout_kernels.move_to_end(key)

    return launch(*(roles[role] for role in program["inputs"]), plan["n"])[0]
