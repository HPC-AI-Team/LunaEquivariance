"""Weighted equivariant normalization and autograd through triple-backward."""

from __future__ import annotations

import math
from hashlib import sha256
from dataclasses import dataclass
from functools import lru_cache

import torch

from ...o3.irreps import PublicIrreps, irreps_key
from .kernels.utils import load_kernel
from ...utils import REGISTER_OPERATOR_LOCK, is_compiling


@dataclass(frozen=True)
class NormBlock:
    offset: int
    channels: int
    components: int
    param_offset: int
    bias_offset: int
    group: int
    center_index: int
    weight: float


@dataclass(frozen=True)
class NormPlan:
    blocks: tuple[NormBlock, ...]
    dim: int
    param_dim: int
    bias_dim: int
    num_groups: int
    num_centers: int
    layout: str
    eps: float


def equivariant_norm_torch(x, weight, bias, plan):
    dtype = x.dtype
    values = x.float() if dtype in (torch.float16, torch.bfloat16) else x
    if dtype in (torch.float16, torch.bfloat16):
        weight = weight.float() if weight is not None else None
        bias = bias.float() if bias is not None else None
    values = values.reshape(x.shape[0], plan.dim)
    blocks = []
    for block in plan.blocks:
        c, d = block.channels, block.components
        value = values[:, block.offset:block.offset + c * d]
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
    return torch.cat(output, -1).reshape_as(x).to(dtype)


@torch.compiler.assume_constant_result
def register_ops(self):
    plan = self.plan
    suffix = sha256(repr(plan).encode()).hexdigest()[:24]
    name = f"equ_norm_{suffix}_forward"
    with REGISTER_OPERATOR_LOCK:
        if hasattr(torch.ops.lunaeq, name):
            return name
        op = object.__new__(EquNormOperator)
        op.plan = plan
        op.compiled_modules, op.device_arches = {}, {}
        channels = plan.blocks[0].channels
        op.input_channels = channels if all(block.channels == channels for block in plan.blocks) else None

        @torch.library.custom_op(f"lunaeq::equ_norm_{suffix}_forward", mutates_args=())
        def equ_norm_forward_op(x: torch.Tensor, gamma: torch.Tensor | None, beta: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            return op.forward(x, gamma, beta)

        @equ_norm_forward_op.register_fake
        def equ_norm_forward_fake(x, gamma, beta):
            plan = op.plan
            return x.new_empty(x.shape), x.new_empty((x.shape[0], plan.num_centers + plan.num_groups))

        def equ_norm_backward_fake_outputs(inputs, needs):
            return tuple(value.new_empty(value.shape) if need and value is not None else None
                         for value, need in zip(inputs, needs))

        @torch.library.custom_op(f"lunaeq::equ_norm_{suffix}_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? gamma, Tensor? beta, Tensor g, Tensor stats, bool[] needs) -> (Tensor?, Tensor?, Tensor?)")
        def equ_norm_backward_op(x: torch.Tensor, gamma: torch.Tensor | None, beta: torch.Tensor | None,
                                 g: torch.Tensor, stats: torch.Tensor,
                                 needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return op.run_backward(1, (x, gamma, beta, g), stats, needs)

        @equ_norm_backward_op.register_fake
        def equ_norm_backward_fake(x, gamma, beta, g, stats, needs):
            return equ_norm_backward_fake_outputs((x, gamma, beta), needs)

        @torch.library.custom_op(f"lunaeq::equ_norm_{suffix}_double_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? gamma, Tensor? beta, Tensor g, Tensor? rx, Tensor? rgamma, Tensor? rbeta, Tensor stats, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
        def equ_norm_double_backward_op(x: torch.Tensor, gamma: torch.Tensor | None, beta: torch.Tensor | None,
                                        g: torch.Tensor, rx: torch.Tensor | None, rgamma: torch.Tensor | None,
                                        rbeta: torch.Tensor | None, stats: torch.Tensor,
                                        needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return op.run_backward(2, (x, gamma, beta, g, rx, rgamma, rbeta), stats, needs)

        @equ_norm_double_backward_op.register_fake
        def equ_norm_double_backward_fake(x, gamma, beta, g, rx, rgamma, rbeta, stats, needs):
            return equ_norm_backward_fake_outputs((x, gamma, beta, g), needs)

        @torch.library.custom_op(f"lunaeq::equ_norm_{suffix}_triple_backward", mutates_args=(),
                                 schema="(Tensor x, Tensor? gamma, Tensor? beta, Tensor g, Tensor? rx, Tensor? rgamma, Tensor? rbeta, Tensor? sx, Tensor? sgamma, Tensor? sbeta, Tensor? sg, Tensor stats, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
        def equ_norm_triple_backward_op(x: torch.Tensor, gamma: torch.Tensor | None, beta: torch.Tensor | None,
                                        g: torch.Tensor, rx: torch.Tensor | None, rgamma: torch.Tensor | None,
                                        rbeta: torch.Tensor | None, sx: torch.Tensor | None, sgamma: torch.Tensor | None,
                                        sbeta: torch.Tensor | None, sg: torch.Tensor | None,
                                        stats: torch.Tensor, needs: list[bool]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            return op.run_backward(
                3, (x, gamma, beta, g, rx, rgamma, rbeta, sx, sgamma, sbeta, sg), stats, needs
            )

        @equ_norm_triple_backward_op.register_fake
        def equ_norm_triple_backward_fake(x, gamma, beta, g, rx, rgamma, rbeta, sx, sgamma, sbeta, sg, stats, needs):
            return equ_norm_backward_fake_outputs((x, gamma, beta, g, rx, rgamma, rbeta), needs)

        def equ_norm_forward_setup(ctx, inputs, output):
            x, gamma, beta = inputs
            _, stats = output
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(stats)
            ctx.save_for_backward(x, gamma, beta, stats)

        def equ_norm_forward_backward(ctx, g, stats_gradient):
            if g is None:
                return None, None, None
            x, gamma, beta, stats = ctx.saved_tensors
            result = equ_norm_backward_op(x, gamma, beta, g, stats, list(ctx.needs_input_grad[:3]))
            return result

        def equ_norm_backward_setup(ctx, inputs, output):
            x, gamma, beta, g, stats, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, gamma, beta, g, stats)

        def equ_norm_backward_backward(ctx, rx, rgamma, rbeta):
            x, gamma, beta, g, stats = ctx.saved_tensors
            result = equ_norm_double_backward_op(x, gamma, beta, g, rx, rgamma, rbeta, stats,
                                                 list(ctx.needs_input_grad[:4]))
            return *result, None, None

        def equ_norm_double_backward_setup(ctx, inputs, output):
            x, gamma, beta, g, rx, rgamma, rbeta, stats, _ = inputs
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, gamma, beta, g, rx, rgamma, rbeta, stats)

        def equ_norm_double_backward_backward(ctx, sx, sgamma, sbeta, sg):
            x, gamma, beta, g, rx, rgamma, rbeta, stats = ctx.saved_tensors
            result = equ_norm_triple_backward_op(x, gamma, beta, g, rx, rgamma, rbeta, sx, sgamma,
                                                 sbeta, sg, stats, list(ctx.needs_input_grad[:7]))
            return *result, None, None

        equ_norm_forward_op.register_autograd(equ_norm_forward_backward, setup_context=equ_norm_forward_setup)
        equ_norm_backward_op.register_autograd(equ_norm_backward_backward, setup_context=equ_norm_backward_setup)
        equ_norm_double_backward_op.register_autograd(equ_norm_double_backward_backward,
                                                     setup_context=equ_norm_double_backward_setup)
        return name


class EquNormOperator:
    def __init__(
        self, irreps, *, grouping="merged", center_scalars=True,
        weighting="component", layout="mul_ir", eps=1e-5, stat_weights=None,
    ):
        irreps = PublicIrreps(irreps)
        if not irreps or any(mul <= 0 for mul, _ in irreps):
            raise ValueError("irreps must contain positive multiplicities")
        if grouping not in ("per_irrep", "separable", "merged"):
            raise ValueError("grouping must be per_irrep, separable or merged")
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError("layout must be mul_ir or ir_mul")
        if weighting not in ("component", "norm", "inverse_dim_squared"):
            raise ValueError("weighting must be component, norm or inverse_dim_squared")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        ids = []
        if grouping == "per_irrep":
            ids = list(range(len(irreps)))
        elif grouping == "merged":
            ids = [0] * len(irreps)
        else:
            scalar_count = sum(ir.is_scalar() for _, ir in irreps)
            scalar_index = 0
            for _, ir in irreps:
                ids.append(scalar_index if ir.is_scalar() else scalar_count)
                scalar_index += int(ir.is_scalar())
        num_groups = max(ids) + 1
        group_counts = [ids.count(g) for g in range(num_groups)]

        if stat_weights is not None:
            stat_weights = tuple(float(w) for w in stat_weights)
            if len(stat_weights) != len(irreps) or any(
                not math.isfinite(w) or w <= 0 for w in stat_weights
            ):
                raise ValueError("stat_weights needs one finite positive weight per block")

        blocks = []
        offset = param_offset = bias_offset = center_index = 0
        for i, ((mul, ir), group) in enumerate(zip(irreps, ids)):
            if stat_weights is not None:
                weight = stat_weights[i]
            elif weighting == "inverse_dim_squared":
                weight = 1.0 / (group_counts[group] * ir.dim * ir.dim)
            else:
                weight = 1.0 / (group_counts[group] * mul)
                if weighting == "component":
                    weight /= ir.dim
            scalar = ir.is_scalar()
            centered = bool(center_scalars and scalar)
            blocks.append(
                NormBlock(
                    offset=offset,
                    channels=mul,
                    components=ir.dim,
                    param_offset=param_offset,
                    bias_offset=bias_offset if scalar else -1,
                    group=group,
                    center_index=center_index if centered else -1,
                    weight=weight,
                )
            )
            offset += mul * ir.dim
            param_offset += mul
            bias_offset += mul if scalar else 0
            center_index += int(centered)

        self.plan = NormPlan(
            blocks=tuple(blocks),
            dim=offset,
            param_dim=param_offset,
            bias_dim=bias_offset,
            num_groups=num_groups,
            num_centers=center_index,
            layout=layout,
            eps=float(eps),
        )
        self.registered_forward = None
        self.compiled_modules = {}
        self.device_arches = {}
        channels = blocks[0].channels
        self.input_channels = channels if all(b.channels == channels for b in blocks) else None

    register_ops = register_ops

    def __getstate__(self):
        state = self.__dict__.copy()
        state["registered_forward"] = None
        state["compiled_modules"] = {}
        state["device_arches"] = {}
        return state

    def __call__(self, x, gamma=None, beta=None):
        if not x.is_cuda or x.dtype in (torch.float16, torch.bfloat16):
            self.check_inputs(x, gamma, beta)
            return equivariant_norm_torch(x, gamma, beta, self.plan)
        if is_compiling():
            name = self.registered_forward if self.registered_forward is not None else self.register_ops()
            return getattr(torch.ops.lunaeq, name).default(x, gamma, beta)[0]
        return EquNormForward.apply(self, x, gamma, beta)

    def check_inputs(self, x, gamma, beta):
        p = self.plan
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("equ_norm requires floating-point inputs")
        if x.ndim not in (2, 3) or x.shape[1:].numel() != p.dim:
            raise ValueError(f"expected [N,{p.dim}] or compatible [N,M,C]")
        if x.ndim == 3 and (
            p.layout != "ir_mul" or x.shape[2] != self.input_channels
        ):
            raise ValueError("3D inputs require ir_mul layout and uniform channel counts")
        for name, value, width in (
            ("gamma", gamma, p.param_dim),
            ("beta", beta, p.bias_dim),
        ):
            if value is not None and (
                value.ndim != 1
                or value.numel() != width
                or value.dtype != x.dtype
            ):
                raise ValueError(f"{name} must have shape [{width}], input dtype")

    def forward(self, x, gamma=None, beta=None):
        self.check_inputs(x, gamma, beta)
        arch = self.device_arches.get(x.device)
        if arch is None:
            arch = torch.cuda.get_device_capability(x.device)
            self.device_arches[x.device] = arch
        key = (0, x.dtype, arch)
        ext = self.compiled_modules.get(key)
        if ext is None:
            extension = load_kernel(self.plan, 0, x.dtype, arch)
            ext = extension.forward
            self.compiled_modules[key] = ext
            self.compiled_modules[(1, x.dtype, arch)] = extension.backward
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        stats = torch.empty((x.shape[0], ext.stats_width), dtype=x.dtype, device=x.device)
        ext.run([x, gamma, beta], [out], stats, stats)
        return out, stats

    def derivative(self, order, inputs, forward_stats, needs=None):
        input_count = (4, 7, 11)
        if order not in (1, 2, 3) or len(inputs) != input_count[order - 1]:
            raise ValueError("derivative expects order 1/2/3 and 4/7/11 operands")
        x, gamma, beta = inputs[:3]
        self.check_inputs(x, gamma, beta)
        p = self.plan
        if (
            forward_stats.dtype != x.dtype
            or not forward_stats.is_contiguous()
            or forward_stats.shape != (x.shape[0], p.num_centers + p.num_groups)
        ):
            raise ValueError(
                "forward_stats must be the contiguous input-dtype state returned by this operator"
            )

        for i, seed in enumerate(inputs[3:], 3):
            if seed is None:
                continue
            if i in (3, 4, 7, 10):
                shape = x.shape
            elif i in (5, 8):
                shape = (p.param_dim,)
            else:
                shape = (p.bias_dim,)
            if seed.shape != shape or seed.dtype != x.dtype:
                raise ValueError(f"operand {i} has incompatible shape or dtype")

        count = (3, 4, 7)[order - 1]
        if needs is not None and len(needs) != count:
            raise ValueError("needs must match the derivative output count")
        return self.run_backward(order, inputs, forward_stats, needs)

    def run_backward(self, order, inputs, forward_stats, needs):
        """Launch with operands validated by the entry or produced by autograd."""
        x = inputs[0]
        arch = self.device_arches.get(x.device)
        if arch is None:
            arch = torch.cuda.get_device_capability(x.device)
            self.device_arches[x.device] = arch
        key = (order, x.dtype, arch)
        ext = self.compiled_modules.get(key)
        if ext is None:
            extension = load_kernel(self.plan, order, x.dtype, arch)
            if order == 1:
                self.compiled_modules[(0, x.dtype, arch)] = extension.forward
                ext = extension.backward
            else:
                ext = extension
            self.compiled_modules[key] = ext

        # Each backward differentiates the prefix x, gamma, beta, g, rx, rgamma, rbeta.
        count = (3, 4, 7)[order - 1]
        if needs is None:
            needs = tuple(value is not None for value in inputs[:count])
        outputs = [
            torch.empty(value.shape, dtype=x.dtype, device=x.device)
            if need and value is not None else None
            for value, need in zip(inputs[:count], needs)
        ]

        # All backward kernels fuse affine reductions with native atomics.
        workspace = torch.empty(0, dtype=x.dtype, device=x.device)
        ext.run(inputs, outputs, forward_stats, workspace)
        return tuple(outputs)


class EquNormForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, gamma, beta):
        out, stats = op.forward(x, gamma, beta)
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, gamma, beta, stats)
        return out

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None, None
        x, gamma, beta, stats = ctx.saved_tensors
        if torch.is_grad_enabled():
            result = EquNormBackward.apply(ctx.op, x, gamma, beta, g, stats)
        else:
            result = ctx.op.run_backward(1, [x, gamma, beta, g], stats, ctx.needs_input_grad[1:])
        return (None, *result)


class EquNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, gamma, beta, g, stats):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, gamma, beta, g, stats)
        return op.run_backward(1, [x, gamma, beta, g], stats, ctx.needs_input_grad[1:4])

    @staticmethod
    def backward(ctx, rx, rgamma, rbeta):
        x, gamma, beta, g, stats = ctx.saved_tensors
        if torch.is_grad_enabled():
            result = EquNormDoubleBackward.apply(ctx.op, x, gamma, beta, g, rx, rgamma, rbeta, stats)
        else:
            result = ctx.op.run_backward(
                2,
                [x, gamma, beta, g, rx, rgamma, rbeta],
                stats,
                ctx.needs_input_grad[1:5],
            )
        return (None, *result, None)


class EquNormDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, gamma, beta, g, rx, rgamma, rbeta, stats):
        ctx.op = op
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, gamma, beta, g, rx, rgamma, rbeta, stats)
        return op.run_backward(
            2, [x, gamma, beta, g, rx, rgamma, rbeta], stats, ctx.needs_input_grad[1:5]
        )

    @staticmethod
    def backward(ctx, sx, sgamma, sbeta, sg):
        x, gamma, beta, g, rx, rgamma, rbeta, stats = ctx.saved_tensors
        result = ctx.op.run_backward(
            3,
            [x, gamma, beta, g, rx, rgamma, rbeta, sx, sgamma, sbeta, sg],
            stats,
            ctx.needs_input_grad[1:8],
        )
        return (None, *result, None)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def prepare_equivariant_norm(irreps, norm, grouping, weighting, layout, eps, stat_weights):
    if norm not in ("layer", "rms"):
        raise ValueError("norm must be layer or rms")
    if eps is None:
        eps = 1e-12 if weighting == "inverse_dim_squared" else 1e-5
    op = EquNormOperator(
        irreps, grouping=grouping, center_scalars=norm == "layer",
        weighting=weighting, layout=layout, eps=eps, stat_weights=stat_weights,
    )
    # Dynamo needs a materialized attribute on the constant factory result.
    op.registered_forward = op.register_ops()
    return op


def equivariant_norm(
    x, irreps, weight=None, bias=None, *, norm="layer", grouping="merged",
    weighting="component", layout="mul_ir", eps=None, stat_weights=None, config=None,
):
    """Weighted equivariant normalization with explicit affine tensors.

    This function owns no trainable parameters and does not construct a Module.
    ``config`` may reuse an independently prepared EquNormOperator.
    """
    op = config
    if op is None:
        op = prepare_equivariant_norm(
            irreps_key(irreps), norm, grouping, weighting, layout, eps,
            None if stat_weights is None else tuple(float(w) for w in stat_weights),
        )
    return op(x, weight, bias)
