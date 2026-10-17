"""CSR self-attention followed by envelope-weighted sum, through third derivatives."""
import torch
from ...utils import is_compiling
from .kernels import forward, backward, double_backward, triple_backward


def configuration(inputs, scale, causal, bias_kind):
    q, _, v = inputs[:3]
    return (q.shape[2], v.shape[2], q.shape[1], scale, causal, bias_kind)


def varlen_attention_conv(
    q,
    k,
    v,
    csr,
    *,
    envelope=None,
    return_attn_out=False,
    num_heads=1,
    bias=None,
    softmax_scale=None,
    causal=False
):
    """Return CSR sums of attention outputs, optionally also unweighted edge outputs.

    Inputs are CUDA FP32/FP16/BF16 [edge, heads*dim] or [edge, heads, dim]. The same CSR
    defines attention sequences and atom aggregation. Envelope is [edge] or
    [edge, 1], including strided views. All floating inputs support three reverse
    derivative orders. Empty CSR segments produce zero. The optional second
    output is independent of envelope. Caller supplies valid monotone CSR
    offsets starting at zero and ending at edge count; no host CSR scan occurs.

    All tensors must be on the same CUDA device.
    Q/K/V share one dtype. Bias and envelope may use that dtype or FP32.
    Outputs retain V's dtype; each gradient retains its input's dtype.
    Low-precision matrix products accumulate in FP32, as do softmax and sums.
    """
    flatten = v.ndim == 2
    for tensor in (q, k, v):
        if tensor.ndim == 3:
            num_heads = tensor.shape[1]
            break
    if not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError("num_heads must be positive")
    q = q.unflatten(1, (num_heads, -1)) if q.ndim == 2 else q
    k = k.unflatten(1, (num_heads, -1)) if k.ndim == 2 else k
    v = v.unflatten(1, (num_heads, -1)) if v.ndim == 2 else v
    if (
        q.ndim != 3
        or k.shape != q.shape
        or v.ndim != 3
        or v.shape[:2] != q.shape[:2]
        or min(q.shape[1:]) < 1
        or v.shape[2] < 1
    ):
        raise ValueError(
            "q/k must match [edge, heads, dim]; v must match edges and heads"
        )
    edges, heads = q.shape[:2]
    if csr.ndim != 1 or csr.numel() < 1 or csr.dtype not in (torch.int32, torch.int64):
        raise ValueError("csr must be a nonempty 1D int32/int64 tensor")
    atoms = csr.numel() - 1
    if edges and not atoms:
        raise ValueError("nonempty inputs require CSR segments")
    if envelope is not None:
        if envelope.shape not in ((edges,), (edges, 1)):
            raise ValueError("envelope must have shape [edge] or [edge, 1]")
        envelope = envelope.squeeze(-1) if envelope.ndim == 2 else envelope
    bias_kind = 0
    if bias is not None:
        if bias.shape == (edges, heads):
            bias_kind = 1
        elif (
            bias.ndim == 4
            and bias.shape[:2] == (atoms, heads)
            and bias.shape[2] == bias.shape[3]
            and bias.shape[2] > 0
        ):
            bias_kind = 2
        else:
            raise ValueError(
                "bias must be [edge, heads] or [atom, heads, max_length, max_length]"
            )
    inputs = (q, k, v, bias, envelope)
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("q/k/v must use float32, float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same dtype")
    if any(t is not None and t.dtype not in (q.dtype, torch.float32) for t in (bias, envelope)):
        raise ValueError("bias and envelope must use the q/k/v dtype or float32")
    scale = q.shape[2] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if is_compiling():
        output, edge_output = attention_conv_forward(
            *inputs, csr, scale, causal, bias_kind, return_attn_out
        )
    else:
        config = configuration(inputs, scale, causal, bias_kind)
        output, edge_output = AttentionConvForward.apply(
            config, csr, return_attn_out, *inputs
        )
    output = output.flatten(1) if flatten else output
    if return_attn_out:
        return output, edge_output.flatten(1) if flatten else edge_output
    return output


class AttentionConvForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, csr, return_attn_out, *inputs):
        ctx.config = config
        ctx.save_for_backward(*inputs, csr)
        ctx.set_materialize_grads(False)
        return forward.attention_conv_forward(inputs, csr, config, return_attn_out)

    @staticmethod
    def backward(ctx, node_grad, edge_grad):
        *inputs, csr = ctx.saved_tensors
        result = AttentionConvBackward.apply(
            ctx.config, csr, ctx.needs_input_grad[3:], *inputs, node_grad, edge_grad
        )
        return (None, None, None, *result)


class AttentionConvBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, csr, needs, *inputs):
        ctx.config = config
        ctx.save_for_backward(*inputs, csr)
        ctx.set_materialize_grads(False)
        return backward.attention_conv_backward(inputs, csr, config, needs)

    @staticmethod
    def backward(ctx, *directions):
        *inputs, csr = ctx.saved_tensors
        result = AttentionConvDoubleBackward.apply(
            ctx.config, csr, ctx.needs_input_grad[3:], *inputs, *directions
        )
        return (None, None, None, *result)


class AttentionConvDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, csr, needs, *inputs):
        ctx.config = config
        ctx.save_for_backward(*inputs, csr)
        ctx.set_materialize_grads(False)
        return double_backward.attention_conv_double_backward(
            inputs, csr, config, needs
        )

    @staticmethod
    def backward(ctx, *directions):
        *inputs, csr = ctx.saved_tensors
        result = triple_backward.attention_conv_triple_backward(
            (*inputs, *directions), csr, ctx.config, ctx.needs_input_grad[3:]
        )
        return (None, None, None, *result)


# Use individual tensor arguments: autograd dispatch can miss
# requires_grad in a list containing None.
@torch.library.custom_op(
    "lunaeq::attention_conv_forward", mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor? envelope, Tensor csr, float scale, bool causal, int bias_kind, bool return_attn_out) -> (Tensor, Tensor?)",
)
def attention_conv_forward(
    q, k, v, bias, envelope,
    csr, scale, causal, bias_kind, return_attn_out,
):
    inputs = (q, k, v, bias, envelope)
    return forward.attention_conv_forward(
        inputs, csr, configuration(inputs, scale, causal, bias_kind), return_attn_out
    )


@attention_conv_forward.register_fake
def forward_fake(
    q, k, v, bias, envelope,
    csr, scale, causal, bias_kind, return_attn_out,
):
    return (
        v.new_empty((csr.numel() - 1, *v.shape[1:])),
        v.new_empty(v.shape) if return_attn_out else None,
    )


@torch.library.custom_op(
    "lunaeq::attention_conv_backward", mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor? envelope, Tensor? node_grad, Tensor? edge_grad, Tensor csr, float scale, bool causal, int bias_kind, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)",
)
def attention_conv_backward(
    q, k, v, bias, envelope, node_grad, edge_grad,
    csr, scale, causal, bias_kind, needs,
):
    inputs = (q, k, v, bias, envelope, node_grad, edge_grad)
    return backward.attention_conv_backward(
        inputs, csr, configuration(inputs, scale, causal, bias_kind), needs
    )


@attention_conv_backward.register_fake
def backward_fake(
    q, k, v, bias, envelope, node_grad, edge_grad,
    csr, scale, causal, bias_kind, needs,
):
    return tuple(
        x.new_empty(x.shape) if x is not None and need else None
        for x, need in zip((q, k, v, bias, envelope), needs)
    )


@torch.library.custom_op(
    "lunaeq::attention_conv_double_backward", mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor? envelope, Tensor? node_grad, Tensor? edge_grad, Tensor? rq, Tensor? rk, Tensor? rv, Tensor? rb, Tensor? re, Tensor csr, float scale, bool causal, int bias_kind, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)",
)
def attention_conv_double_backward(
    q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re,
    csr, scale, causal, bias_kind, needs,
):
    inputs = (q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re)
    return double_backward.attention_conv_double_backward(
        inputs, csr, configuration(inputs, scale, causal, bias_kind), needs
    )


@attention_conv_double_backward.register_fake
def double_backward_fake(
    q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re,
    csr, scale, causal, bias_kind, needs,
):
    return tuple(
        x.new_empty(x.shape) if x is not None and need else None
        for x, need in zip((q, k, v, bias, envelope, node_grad, edge_grad), needs)
    )


@torch.library.custom_op(
    "lunaeq::attention_conv_triple_backward", mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor? envelope, Tensor? node_grad, Tensor? edge_grad, Tensor? rq, Tensor? rk, Tensor? rv, Tensor? rb, Tensor? re, Tensor? sq, Tensor? sk, Tensor? sv, Tensor? sb, Tensor? se, Tensor? sg, Tensor? so, Tensor csr, float scale, bool causal, int bias_kind, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)",
)
def attention_conv_triple_backward(
    q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re, sq, sk, sv, sb, se, sg, so,
    csr, scale, causal, bias_kind, needs,
):
    inputs = (q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re, sq, sk, sv, sb, se, sg, so)
    return triple_backward.attention_conv_triple_backward(
        inputs, csr, configuration(inputs, scale, causal, bias_kind), needs
    )


@attention_conv_triple_backward.register_fake
def triple_backward_fake(
    q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re, sq, sk, sv, sb, se, sg, so,
    csr, scale, causal, bias_kind, needs,
):
    return tuple(
        x.new_empty(x.shape) if x is not None and need else None
        for x, need in zip((q, k, v, bias, envelope, node_grad, edge_grad, rq, rk, rv, rb, re), needs)
    )


def save_inputs(ctx, inputs, output):
    *values, csr, ctx.scale, ctx.causal, ctx.bias_kind, _ = inputs
    ctx.save_for_backward(*values, csr)
    ctx.needs = list(ctx.needs_input_grad[:len(values)])
    ctx.set_materialize_grads(False)


def forward_backward(ctx, node_grad, edge_grad):
    *values, csr = ctx.saved_tensors
    result = attention_conv_backward(
        *values, node_grad, edge_grad, csr,
        ctx.scale, ctx.causal, ctx.bias_kind, ctx.needs,
    )
    return (*result, None, None, None, None, None)


def backward_backward(ctx, *directions):
    *values, csr = ctx.saved_tensors
    result = attention_conv_double_backward(
        *values, *directions, csr,
        ctx.scale, ctx.causal, ctx.bias_kind, ctx.needs,
    )
    return (*result, None, None, None, None, None)


def double_backward_backward(ctx, *directions):
    *values, csr = ctx.saved_tensors
    result = attention_conv_triple_backward(
        *values, *directions, csr,
        ctx.scale, ctx.causal, ctx.bias_kind, ctx.needs,
    )
    return (*result, None, None, None, None, None)


attention_conv_forward.register_autograd(forward_backward, setup_context=save_inputs)
attention_conv_backward.register_autograd(backward_backward, setup_context=save_inputs)
attention_conv_double_backward.register_autograd(
    double_backward_backward, setup_context=save_inputs
)
