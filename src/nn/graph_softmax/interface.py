"""CSR graph softmax with fused recomputation through the third reverse order."""

import math

import torch

from .kernels.utils import load_graph_softmax_kernel


@torch.library.custom_op("lunaeq_graph::graph_softmax_forward", mutates_args=())
def graph_softmax_forward(
    data: torch.Tensor, env: torch.Tensor, row_ptr: torch.Tensor,
    eps: float, env_mode: int, smooth_out: bool,
) -> torch.Tensor:
    return load_graph_softmax_kernel().graph_softmax_forward_cuda(data, env, row_ptr, eps, env_mode, smooth_out)


@graph_softmax_forward.register_fake
def forward_fake(data, env, row_ptr, eps, env_mode, smooth_out):
    return data.new_empty(data.shape)


@torch.library.custom_op("lunaeq_graph::graph_softmax_backward", mutates_args=())
def graph_softmax_backward(
    grad_output: torch.Tensor, data: torch.Tensor, env: torch.Tensor,
    row_ptr: torch.Tensor, env_mode: int, need_grad_env: bool,
    smooth_out: bool, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return load_graph_softmax_kernel().graph_softmax_backward_cuda(
        grad_output, data, env, row_ptr, env_mode, need_grad_env, smooth_out, eps,
    )


@graph_softmax_backward.register_fake
def backward_fake(
    grad_output, data, env, row_ptr, env_mode, need_grad_env, smooth_out, eps,
):
    return data.new_empty(data.shape), env.new_empty(env.shape if need_grad_env else (0,))


@torch.library.custom_op("lunaeq_graph::graph_softmax_double_backward", mutates_args=())
def graph_softmax_double_backward(
    grad_grad_data: torch.Tensor, grad_grad_env: torch.Tensor,
    grad_output: torch.Tensor, data: torch.Tensor, env: torch.Tensor,
    row_ptr: torch.Tensor, env_mode: int, need_grad_env: bool,
    smooth_out: bool, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return load_graph_softmax_kernel().graph_softmax_double_backward_cuda(
        grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr,
        env_mode, need_grad_env, smooth_out, eps,
    )


@graph_softmax_double_backward.register_fake
def double_backward_fake(
    grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr,
    env_mode, need_grad_env, smooth_out, eps,
):
    return (
        grad_output.new_empty(grad_output.shape), data.new_empty(data.shape),
        env.new_empty(env.shape if need_grad_env else (0,)),
    )


@torch.library.custom_op("lunaeq_graph::graph_softmax_triple_backward", mutates_args=())
def graph_softmax_triple_backward(
    grad_grad_grad_output: torch.Tensor, grad_grad_grad_data: torch.Tensor,
    grad_grad_grad_env: torch.Tensor, grad_grad_data: torch.Tensor,
    grad_grad_env: torch.Tensor, grad_output: torch.Tensor, data: torch.Tensor,
    env: torch.Tensor, row_ptr: torch.Tensor, env_mode: int,
    need_grad_env: bool, smooth_out: bool, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return load_graph_softmax_kernel().graph_softmax_triple_backward_cuda(
        grad_grad_grad_output.contiguous(), grad_grad_grad_data.contiguous(),
        grad_grad_grad_env.contiguous(), grad_grad_data.contiguous(),
        grad_grad_env.contiguous(), grad_output.contiguous(), data, env, row_ptr,
        env_mode, need_grad_env, smooth_out, eps,
    )


@graph_softmax_triple_backward.register_fake
def triple_backward_fake(
    grad_grad_grad_output, grad_grad_grad_data, grad_grad_grad_env,
    grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr,
    env_mode, need_grad_env, smooth_out, eps,
):
    return (
        grad_grad_data.new_empty(grad_grad_data.shape),
        grad_grad_env.new_empty(grad_grad_env.shape),
        grad_output.new_empty(grad_output.shape), data.new_empty(data.shape),
        env.new_empty(env.shape if need_grad_env else (0,)),
    )


class GraphSoftmaxForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, env, row_ptr, eps, env_mode, smooth_out):
        ctx.config = env_mode, env_mode != 0 and ctx.needs_input_grad[1], smooth_out, eps
        ctx.save_for_backward(data, env, row_ptr)
        ctx.set_materialize_grads(False)
        return graph_softmax_forward(data, env, row_ptr, eps, env_mode, smooth_out)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 6
        grad_data, grad_env = GraphSoftmaxBackward.apply(
            grad_output, *ctx.saved_tensors, *ctx.config,
        )
        return grad_data, grad_env if ctx.config[1] else None, None, None, None, None


class GraphSoftmaxBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, grad_output, data, env, row_ptr, env_mode, need_grad_env, smooth_out, eps,
    ):
        ctx.config = env_mode, need_grad_env, smooth_out, eps
        ctx.save_for_backward(grad_output, data, env, row_ptr)
        ctx.set_materialize_grads(False)
        return graph_softmax_backward(grad_output, data, env, row_ptr, *ctx.config)

    @staticmethod
    def backward(ctx, grad_grad_data, grad_grad_env):
        grad_output, data, env, row_ptr = ctx.saved_tensors
        if grad_grad_data is None:
            grad_grad_data = data.new_empty((0,))
        if grad_grad_env is None or not ctx.config[1]:
            grad_grad_env = env.new_empty((0,))
        grad_grad_output, grad_data, grad_env = GraphSoftmaxDoubleBackward.apply(
            grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr, *ctx.config,
        )
        return (
            grad_grad_output, grad_data, grad_env if ctx.config[1] else None,
            None, None, None, None, None,
        )


class GraphSoftmaxDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr,
        env_mode, need_grad_env, smooth_out, eps,
    ):
        ctx.config = env_mode, need_grad_env, smooth_out, eps
        ctx.save_for_backward(grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr)
        ctx.set_materialize_grads(False)
        return graph_softmax_double_backward(
            grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr, *ctx.config,
        )

    @staticmethod
    def backward(ctx, grad_grad_grad_output, grad_grad_grad_data, grad_grad_grad_env):
        grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr = ctx.saved_tensors
        if grad_grad_grad_output is None:
            grad_grad_grad_output = data.new_empty((0,))
        if grad_grad_grad_data is None:
            grad_grad_grad_data = data.new_empty((0,))
        if grad_grad_grad_env is None:
            grad_grad_grad_env = env.new_empty((0,))
        grad_u, grad_v, grad_g, grad_data, grad_env = graph_softmax_triple_backward(
            grad_grad_grad_output, grad_grad_grad_data, grad_grad_grad_env,
            grad_grad_data, grad_grad_env, grad_output, data, env, row_ptr, *ctx.config,
        )
        return (
            grad_u if ctx.needs_input_grad[0] else None,
            grad_v if ctx.needs_input_grad[1] else None,
            grad_g, grad_data, grad_env if ctx.config[1] else None,
            None, None, None, None, None,
        )


def forward_setup(ctx, inputs, output):
    data, env, row_ptr, eps, env_mode, smooth_out = inputs
    ctx.config = env_mode, env_mode != 0 and ctx.needs_input_grad[1], smooth_out, eps
    ctx.save_for_backward(data, env, row_ptr)
    ctx.set_materialize_grads(False)


def backward_setup(ctx, inputs, output):
    ctx.config = inputs[-4:]
    ctx.save_for_backward(*inputs[:-4])
    ctx.set_materialize_grads(False)


# make_fx exposes these custom ops without their Python autograd.Function wrappers.
graph_softmax_forward.register_autograd(GraphSoftmaxForward.backward, setup_context=forward_setup)
graph_softmax_backward.register_autograd(GraphSoftmaxBackward.backward, setup_context=backward_setup)
graph_softmax_double_backward.register_autograd(
    GraphSoftmaxDoubleBackward.backward, setup_context=backward_setup,
)


def build_graph_softmax_row_ptr(segment_ids, num_segments):
    """Build int64 CSR offsets from sorted, nonnegative segment IDs.

    IDs must be smaller than num_segments. Empty segments are preserved.
    Build once with the graph and reuse the offsets across model layers.
    """
    if num_segments < 0:
        raise ValueError("num_segments must be a nonnegative integer")
    if (
        segment_ids.ndim != 1
        or segment_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("segment_ids must be a 1D int32/int64 tensor")
    if segment_ids.numel() and not num_segments:
        raise ValueError("Nonempty segment_ids require at least one segment")
    return load_graph_softmax_kernel().graph_softmax_build_row_ptr(segment_ids.to(dtype=torch.int64).contiguous(), num_segments)


def graph_softmax_torch(data, csr, env, eps, smooth_out):
    dtype = data.dtype
    values = data.float() if dtype in (torch.float16, torch.bfloat16) else data
    num_segments = csr.numel() - 1
    segments = torch.repeat_interleave(
        torch.arange(num_segments, device=data.device), csr[1:] - csr[:-1],
        output_size=data.shape[0],
    )
    index = segments[:, None].expand_as(values)
    maxima = values.new_full((num_segments, data.shape[1]), -torch.inf)
    maxima = maxima.scatter_reduce(0, index, values.detach(), reduce="amax")
    numerator = (values - maxima[segments]).exp()
    if env is not None:
        env = env.to(values.dtype)
        numerator = numerator * env
    denominator = values.new_zeros((num_segments, data.shape[1])).index_add(0, segments, numerator)
    output = numerator / (denominator[segments] + eps)
    if smooth_out:
        output = output * env
    return output.to(dtype)


def graph_softmax(data, csr, *, env=None, eps=1e-8, smooth_out=False):
    """Normalize [all_edge, dim] logits within each CSR segment.

    For each segment and channel, z = exp(data - stop_gradient(max(data))).
    The result is z * env / (sum(z * env) + eps), treating a missing env as
    one. smooth_out=True multiplies the result by env once more.

    CPU and FP16/BF16 inputs use PyTorch; CUDA FP32/FP64 use fused kernels.
    env accepts [all_edge], [all_edge, 1]
    or [all_edge, dim]. csr accepts int32/int64 and is read directly without dtype conversion.
    Offsets must start at zero, be monotonic and end at
    all_edge. Values are a graph-construction contract and are not copied
    to the CPU on each call. Noncontiguous operands are made contiguous.
    Reverse derivatives through order three are supported. The CUDA kernels
    recompute normalization during backward.
    """
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("data must have shape [all_edge, dim] with dim > 0")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("data must have a floating-point dtype")
    if (
        csr.ndim != 1
        or csr.numel() < 1
        or csr.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("csr must be a 1D int32/int64 tensor")
    if data.shape[0] and csr.numel() == 1:
        raise ValueError("Nonempty data requires at least one CSR segment")
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if smooth_out and env is None:
        raise ValueError("smooth_out requires env")
    data = data.contiguous()
    if env is None:
        env = data.new_empty((0,))
        env_mode = 0
    else:
        if env.ndim == 1:
            env = env.unsqueeze(1)
        if (
            env.dtype != data.dtype
            or env.ndim != 2
            or env.shape[0] != data.shape[0]
            or env.shape[1] not in (1, data.shape[1])
        ):
            raise ValueError(
                "env must match data dtype and have shape [E], [E, 1] or [E, dim]"
            )
        env = env.contiguous()
        env_mode = 1 if env.shape[1] == 1 else 2
    csr = csr.contiguous()
    if not data.is_cuda or data.dtype in (torch.float16, torch.bfloat16):
        return graph_softmax_torch(data, csr, env if env_mode else None, eps, smooth_out)
    return GraphSoftmaxForward.apply(data, env, csr, eps, env_mode, smooth_out)
