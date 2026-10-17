"""Check a user workload against e3nn or PyTorch and measure steady-state execution time."""

from copy import deepcopy
from math import prod
from statistics import median
from time import perf_counter

import torch
from torch.utils._pytree import tree_flatten, tree_map


class TensorProductConvReference(torch.nn.Module):
    """e3nn edge messages followed by destination aggregation."""

    def __init__(self, tp, scalar_outputs=False):
        super().__init__()
        self.tp = tp
        self.scalar_paths = torch.nn.ModuleList()
        self.weight_slices = []
        if scalar_outputs:
            from e3nn.o3 import TensorProduct

            paths, offset = [], 0
            for instruction in tp.instructions:
                width = prod(instruction.path_shape) if instruction.has_weight else 0
                if tp.irreps_out[instruction.i_out].ir.l == 0:
                    paths.append(
                        (instruction.i_out, instruction, slice(offset, offset + width))
                    )
                offset += width
            for index, instruction, section in sorted(paths, key=lambda item: item[0]):
                # Stored path_weight already includes e3nn's normalization.
                path = TensorProduct(
                    tp.irreps_in1,
                    tp.irreps_in2,
                    [tp.irreps_out[index]],
                    [
                        (
                            instruction.i_in1,
                            instruction.i_in2,
                            0,
                            instruction.connection_mode,
                            instruction.has_weight,
                            instruction.path_weight**2,
                        )
                    ],
                    internal_weights=False,
                    shared_weights=tp.shared_weights,
                    irrep_normalization="none",
                    path_normalization="none",
                ).to(tp.weight)
                self.scalar_paths.append(path)
                self.weight_slices.append(section)

    def forward(
        self, x, edge_attr, weight=None, *, graph, env=None, return_l0_feat=False
    ):
        source = x[graph.src.long()]
        edge = self.tp(source, edge_attr, weight)
        messages = edge if env is None else edge * env.reshape(-1, 1)
        output = edge.new_zeros(graph.num_nodes, self.tp.irreps_out.dim)
        output = output.index_add(0, graph.dst.long(), messages)
        if not return_l0_feat:
            return output
        weight = self.tp.weight if weight is None else weight
        scalars = [
            path(source, edge_attr, weight[..., section])
            for path, section in zip(self.scalar_paths, self.weight_slices)
        ]
        return output, torch.cat(scalars, -1) if scalars else edge[:, :0]


class EquivariantNormReference(torch.nn.Module):
    """PyTorch normalization using the module's groups and statistical weights."""

    def __init__(self, module):
        super().__init__()
        self.plan = module.op.plan
        self.weight = deepcopy(module.weight)
        self.bias = deepcopy(module.bias)

    def forward(self, x):
        plan = self.plan
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weight, bias = self.weight, self.bias
        if x.dtype in (torch.float16, torch.bfloat16):
            weight = weight.float() if weight is not None else None
            bias = bias.float() if bias is not None else None
        flat = values.reshape(x.shape[0], plan.dim)
        blocks = []
        for block in plan.blocks:
            c, d = block.channels, block.components
            value = flat[:, block.offset:block.offset + c * d]
            value = (value.reshape(-1, c, d) if plan.layout == "mul_ir"
                     else value.reshape(-1, d, c).transpose(1, 2))
            if block.center_index >= 0:
                value = value - value.mean(1, keepdim=True)
            blocks.append(value)
        scales = [
            (sum(value.square().sum((1, 2), keepdim=True) * block.weight
                 for block, value in zip(plan.blocks, blocks) if block.group == group)
             + plan.eps).rsqrt()
            for group in range(plan.num_groups)
        ]
        output = []
        for block, value in zip(plan.blocks, blocks):
            value = value * scales[block.group]
            if weight is not None:
                value = value * weight[
                    block.param_offset:block.param_offset + block.channels
                ][None, :, None]
            if bias is not None and block.bias_offset >= 0:
                value = value + bias[
                    block.bias_offset:block.bias_offset + block.channels
                ][None, :, None]
            if plan.layout == "ir_mul":
                value = value.transpose(1, 2)
            output.append(value.flatten(1))
        return torch.cat(output, -1).reshape_as(x).to(x.dtype)


class GraphSoftmaxReference(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.eps, self.smooth_out = module.eps, module.smooth_out

    def forward(self, data, csr, *, env=None):
        values = data.float() if data.dtype in (torch.float16, torch.bfloat16) else data
        if env is not None and data.dtype in (torch.float16, torch.bfloat16):
            env = env.float()
        parts = []
        for begin, end in zip(self.offsets, self.offsets[1:]):
            logits = values[begin:end]
            if begin == end:
                parts.append(logits)
                continue
            weight = 1 if env is None else env[begin:end].reshape(end - begin, -1)
            numerator = (logits - logits.detach().amax(0, keepdim=True)).exp() * weight
            output = numerator / (numerator.sum(0, keepdim=True) + self.eps)
            parts.append(output * weight if self.smooth_out else output)
        return torch.cat(parts).to(data.dtype) if parts else data[:0]


class AttentionReference(torch.nn.Module):
    """Segmented PyTorch attention, optionally followed by atom aggregation."""

    def __init__(self, module, aggregate=False):
        super().__init__()
        self.num_heads = module.num_heads
        self.scale = module.softmax_scale
        self.causal = module.causal
        self.aggregate = aggregate

    def forward(self, q, k, v, csr, *, bias=None, envelope=None, return_attn_out=False):
        flatten = v.ndim == 2
        heads = next((x.shape[1] for x in (q, k, v) if x.ndim == 3), self.num_heads)
        q, k, v = (x.unflatten(1, (heads, -1)) if x.ndim == 2 else x for x in (q, k, v))
        scale = q.shape[-1] ** -0.5 if self.scale is None else self.scale
        edges, atoms = [], []
        for index, (begin, end) in enumerate(zip(self.offsets, self.offsets[1:])):
            length = end - begin
            scores = torch.einsum("ihd,jhd->hij", q[begin:end], k[begin:end]) * scale
            if bias is not None:
                scores = scores + (bias[begin:end].T[:, None, :] if bias.ndim == 2
                                   else bias[index, :, :length, :length])
            if self.causal:
                mask = torch.ones(length, length, device=q.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, -torch.inf)
            output = torch.einsum("hij,jhv->ihv", scores.softmax(-1), v[begin:end])
            edges.append(output)
            if self.aggregate:
                weighted = output if envelope is None else output * envelope.reshape(-1)[begin:end, None, None]
                atoms.append(weighted.sum(0))
        edge_output = torch.cat(edges) if edges else v[:0]
        if flatten:
            edge_output = edge_output.flatten(1)
        if not self.aggregate:
            return edge_output
        output = torch.stack(atoms) if atoms else v[:0]
        if flatten:
            output = output.flatten(1)
        return (output, edge_output) if return_attn_out else output


def build_reference(module, scalar_outputs=False):
    from e3nn import nn, o3
    from LunaEquivariance import (
        Linear, SphericalHarmonics, Norm, Activation, Gate, NormActivation,
        Dropout, TensorProduct, TensorProductConv, EquivariantNorm,
        GraphSoftmax, VarlenAttention, VarlenAttentionConv,
    )

    if isinstance(module, (TensorProduct, TensorProductConv)):
        tp = module.tp
        reference = o3.TensorProduct(
            str(tp.irreps_in1), str(tp.irreps_in2), str(tp.irreps_out),
            [(p.i_in1, p.i_in2, p.i_out, p.connection_mode, p.has_weight,
              p.path_weight**2) for p in tp.instructions],
            internal_weights=tp.internal_weights, shared_weights=tp.shared_weights,
            irrep_normalization="none", path_normalization="none",
        ).to(tp.weight)
        with torch.no_grad():
            reference.weight.copy_(tp.weight)
        if isinstance(module, TensorProductConv):
            reference = TensorProductConvReference(reference, scalar_outputs)
    elif isinstance(module, Linear):
        reference = deepcopy(module.tp)
        reference.load_state_dict(module.state_dict())
    elif isinstance(module, SphericalHarmonics):
        reference = o3.SphericalHarmonics(
            str(module.irreps_out), module.normalize, module.normalization,
            irreps_in=str(module.irreps_in),
        )
    elif isinstance(module, NormActivation):
        reference = nn.NormActivation(
            o3.Irreps(str(module.irreps_in)), module.scalar_nonlinearity,
            normalize=module.normalize, epsilon=module.epsilon, bias=module.bias,
        )
        reference.load_state_dict(module.state_dict())
    elif isinstance(module, Norm):
        reference = o3.Norm(str(module.irreps_in), squared=module.squared)
    elif isinstance(module, Gate):
        reference = nn.Gate(
            module.act_scalars.irreps_in,
            [act.f if act is not None else None for act in module.act_scalars.acts],
            module.act_gates.irreps_in,
            [act.f if act is not None else None for act in module.act_gates.acts],
            module.irreps_gated,
        )
        reference.act_scalars = deepcopy(module.act_scalars)
        reference.act_gates = deepcopy(module.act_gates)
    elif isinstance(module, Activation):
        reference = nn.Activation(
            str(module.irreps_in), [act.f if act is not None else None for act in module.acts],
        )
        reference.acts = deepcopy(module.acts)
        reference.paths = [(mul, ir, act) for (mul, ir), act
                           in zip(reference.irreps_in, reference.acts)]
    elif isinstance(module, Dropout):
        reference = nn.Dropout(str(module.irreps), p=module.p)
    elif isinstance(module, EquivariantNorm):
        reference = EquivariantNormReference(module)
    elif isinstance(module, GraphSoftmax):
        reference = GraphSoftmaxReference(module)
    elif isinstance(module, (VarlenAttention, VarlenAttentionConv)):
        reference = AttentionReference(module, aggregate=isinstance(module, VarlenAttentionConv))
    else:
        raise TypeError(f"No automatic reference for {type(module).__name__}")
    for original, parameter in zip(module.parameters(), reference.parameters()):
        parameter.requires_grad_(original.requires_grad)
    return reference.train(module.training)


def gradient_targets(module, tensors):
    targets = {id(x): x for x in [*tensors, *module.parameters()] if x.requires_grad}
    return list(targets.values())


def evaluate(module, args, kwargs, loss_fn, targets, order, directions, collect=False):
    output = module(*args, **kwargs)
    loss = loss_fn(output)
    if loss.ndim != 0:
        raise ValueError("loss_fn should return a scalar tensor")
    results = {}
    if collect:
        results = {
            "output": tree_map(torch.Tensor.detach, output),
            "loss": loss.detach(),
        }
    values = (loss,)
    for level in range(order):
        active = [(i, value) for i, value in enumerate(values) if value.requires_grad]
        if active:
            gradients = torch.autograd.grad(
                tuple(value for _, value in active), targets,
                None if level == 0 else tuple(directions[level - 1][i] for i, _ in active),
                create_graph=level + 1 < order, allow_unused=True,
            )
            values = tuple(
                torch.zeros_like(x) if g is None else g
                for x, g in zip(targets, gradients)
            )
        else:
            values = tuple(torch.zeros_like(x) for x in targets)
        if collect:
            results[f"gradient_{level + 1}"] = tuple(x.detach() for x in values)
    return results


def error_metrics(actual, reference, atol, rtol):
    left, left_spec = tree_flatten(actual)
    right, right_spec = tree_flatten(reference)
    if left_spec != right_spec or any(a.shape != b.shape for a, b in zip(left, right)):
        return {"passed": False, "error": "Output or gradient shapes differ"}
    max_abs, diff_squared, ref_squared, passed = 0.0, 0.0, 0.0, True
    for a, b in zip(left, right):
        a, b = a.detach().double(), b.detach().double()
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        passed &= finite and torch.allclose(a, b, atol=atol, rtol=rtol)
        difference = a - b
        max_abs = (
            max(max_abs, difference.abs().max().item() if a.numel() else 0.0)
            if finite
            else float("inf")
        )
        diff_squared += difference.square().sum().item()
        ref_squared += b.square().sum().item()
    relative = (
        (diff_squared / ref_squared) ** 0.5
        if ref_squared
        else (0.0 if diff_squared == 0 else float("inf"))
    )
    return {"passed": bool(passed), "max_abs": max_abs, "relative_l2": relative}


def print_report(report):
    print(f"{report['operator']}: {'PASS' if report['passed'] else 'FAIL'} "
          f"({report['device']}, {report['dtype']})")
    name = report["reference"]
    print(f"  loss: LunaEq {report['loss']['lunaeq']:.8g}, {name} {report['loss']['reference']:.8g}")
    for stage, error in report["errors"].items():
        if "error" in error:
            print(f"  {stage}: {error['error']}")
        else:
            status = "PASS" if error["passed"] else "FAIL"
            print(f"  {stage}: max abs {error['max_abs']:.3e}, "
                  f"relative L2 {error['relative_l2']:.3e} ({status})")
    print("  Median end-to-end wall time")
    for stage, timing in report["timings"].items():
        print(f"  {stage}: LunaEq {timing['lunaeq_ms']:.4f} ms, "
              f"{name} {timing['reference_ms']:.4f} ms, speedup {timing['speedup']:.2f}x")


def check_compatibility(
    module, args, *, kwargs=None, loss_fn=None, derivative_order=1,
    warmup=3, repeats=20, atol=1e-5, rtol=1e-4, seed=0, verbose=True,
):
    """Compare outputs, loss, derivatives and timings with e3nn or PyTorch.

    Pass a LunaEquivariance module and its inputs. The reference uses its
    configuration and weights. Higher derivatives use random vector contractions.
    Timing covers forward, loss and the requested derivatives; order 0 times forward only.
    """
    if not isinstance(derivative_order, int) or derivative_order < 0:
        raise ValueError("derivative_order should be a nonnegative integer")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup should be nonnegative and repeats positive")
    kwargs = {} if kwargs is None else kwargs
    if loss_fn is None:
        def loss_fn(output):
            return sum(x.square().mean() if x.numel() else x.sum()
                       for x in tree_flatten(output)[0])
    tensors = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
    sample = next(x for x in tensors if x.is_floating_point())
    devices = sorted({x.device.index for x in tensors if x.is_cuda})

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    def reset_seed():
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)

    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        reset_seed()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(sample.dtype)
        try:
            reference = build_reference(module, kwargs.get("return_l0_feat", False))
        finally:
            torch.set_default_dtype(previous_dtype)
        reference = reference.to(device=sample.device, dtype=sample.dtype)
        if isinstance(reference, (GraphSoftmaxReference, AttentionReference)):
            csr_index = 1 if isinstance(reference, GraphSoftmaxReference) else 3
            csr = args[csr_index] if len(args) > csr_index else kwargs["csr"]
            reference.offsets = csr.detach().cpu().tolist()
        reference_name = ("PyTorch" if isinstance(reference, (
            EquivariantNormReference, GraphSoftmaxReference, AttentionReference,
        )) else "e3nn")
        modules = (module, reference)
        targets = [gradient_targets(module, tensors) for module in modules]
        if derivative_order:
            if not targets[0]:
                raise ValueError("Set requires_grad=True on inputs or use trainable parameters")
            if [x.shape for x in targets[0]] != [x.shape for x in targets[1]]:
                raise ValueError("Gradient target shapes differ between LunaEq and the reference")
        generator = torch.Generator().manual_seed(seed)
        directions = [
            tuple(
                torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
                for x in targets[0]
            )
            for _ in range(derivative_order - 1)
        ]
        results = []
        for candidate, module_targets in zip(modules, targets):
            reset_seed()
            results.append(evaluate(
                candidate, args, kwargs, loss_fn, module_targets,
                derivative_order, directions, collect=True,
            ))
        report = {
            "operator": type(module).__name__, "reference": reference_name, "device": str(sample.device), "dtype": str(sample.dtype),
            "errors": {
                stage: error_metrics(value, results[1][stage], atol, rtol)
                for stage, value in results[0].items()
            },
            "loss": {
                "lunaeq": results[0]["loss"].item(),
                "reference": results[1]["loss"].item(),
            },
            "timings": {},
        }
        del results
        report["passed"] = all(error["passed"] for error in report["errors"].values())
        order = derivative_order
        samples = [[], []]
        for iteration in range(warmup + repeats):
            for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                reset_seed()
                synchronize()
                start = perf_counter()
                if order == 0:
                    modules[index](*args, **kwargs)
                else:
                    evaluate(
                        modules[index], args, kwargs, loss_fn,
                        targets[index], order, directions,
                    )
                synchronize()
                if iteration >= warmup:
                    samples[index].append((perf_counter() - start) * 1000)
        luna_ms, reference_ms = map(median, samples)
        stage = "forward" if order == 0 else f"forward_loss_gradient_{order}"
        report["timings"][stage] = {
            "lunaeq_ms": luna_ms, "reference_ms": reference_ms,
            "speedup": reference_ms / luna_ms,
        }
    if verbose:
        print_report(report)
    return report
