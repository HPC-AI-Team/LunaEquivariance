"""FP32/FP16/BF16 CSR attention and complete reverse derivatives through order three."""

import torch

from ...utils import is_compiling

from .kernels import backward, double_backward, forward, triple_backward


def varlen_attention(
    q,
    k,
    v,
    csr,
    *,
    num_heads=1,
    bias=None,
    softmax_scale=None,
    causal=False,
):
    """Attend within each CSR sequence; output has the same shape as v.

    q/k/v accept [edges, heads * dim] or [edges, heads, dim]. Already split
    tensors determine the head count; num_heads is used when all are 2D.
    Only 2D inputs are unflattened, without copying their data.
    csr is a CUDA int32/int64 row pointer. Input rows must already be packed
    in CSR sequence order; outputs keep that same order.

    bias may be [all_edge, num_heads] (key bias) or
    [num_sequences, num_heads, max_length, max_length] (pair bias). Pair-bias
    axes index CSR positions, while key-bias rows follow the same packed order as q/k/v.
    CSR pointers must start at zero, be monotonic, and end at all_edge;
    max_length must cover every sequence. Metadata is checked without copying
    CSR values to the CPU. All sequences use the variable-length implementation.
    Dropout is not applied.

    Q/K/V share FP32, FP16 or BF16 dtype; bias may also use FP32.
    Output and gradients retain their corresponding input dtype. Matrix
    products, softmax and saved normalization statistics accumulate in FP32.
    """
    flatten_output = v.ndim == 2
    if q.ndim == 3:
        num_heads = q.shape[1]
    elif k.ndim == 3:
        num_heads = k.shape[1]
    elif v.ndim == 3:
        num_heads = v.shape[1]
    if q.ndim == 2:
        q = q.unflatten(1, (num_heads, -1))
    if k.ndim == 2:
        k = k.unflatten(1, (num_heads, -1))
    if v.ndim == 2:
        v = v.unflatten(1, (num_heads, -1))
    if (
        q.ndim != 3 or k.shape != q.shape or v.ndim != 3
        or v.shape[:2] != q.shape[:2]
        or q.shape[1] <= 0 or q.shape[2] <= 0 or v.shape[2] <= 0
    ):
        raise ValueError("q/k must match [edges, heads, dim]; v must match edges and heads.")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("q/k/v must use float32, float16 or bfloat16.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same dtype.")
    if bias is not None and bias.dtype not in (q.dtype, torch.float32):
        raise ValueError("bias must use the q/k/v dtype or float32.")
    if (
        csr.ndim != 1
        or csr.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("csr must be a 1D int32/int64 tensor.")
    edges = q.shape[0]
    if csr.numel() < 1 or edges > 2**31 - 1 or csr.numel() > 2**31 - 1:
        raise ValueError(
            "CSR must contain its initial offset; row counts must fit int32."
        )
    sequences = csr.numel() - 1
    if edges and not sequences:
        raise ValueError("Nonempty edge input needs at least one CSR sequence.")
    kind = 0
    if bias is not None:
        if bias.ndim == 2 and bias.shape == (edges, num_heads):
            kind = 1
        elif (
            bias.ndim == 4
            and bias.shape[:2] == (sequences, num_heads)
            and bias.shape[2] == bias.shape[3]
            and bias.shape[2] > 0
        ):
            kind = 2
        else:
            raise ValueError(
                "bias must be [all_edge, heads] or [sequences, heads, max_length, max_length]."
            )
    if is_compiling():
        output, _ = attention_forward(
            q, k, v, bias, csr, softmax_scale, causal,
        )
    else:
        dim, value_dim = q.shape[2], v.shape[2]
        scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
        config = (
            dim, value_dim, num_heads, scale, causal,
            max(1, sequences.bit_length()), kind,
        )
        output = VarlenAttentionForward.apply(config, csr, q, k, v, bias)
    return output.flatten(1) if flatten_output else output


class VarlenAttentionForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, csr, q, k, v, bias):
        output, state = forward.attention_forward(
            (q, k, v, bias, None), csr, config
        )
        ctx.config = config
        ctx.save_for_backward(q, k, v, bias, csr, state)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return (None,) * 6
        q, k, v, bias, csr, state = ctx.saved_tensors
        gradients = VarlenAttentionBackward.apply(
            ctx.config,
            ctx.needs_input_grad[2:6],
            csr,
            state,
            q,
            k,
            v,
            bias,
            g,
        )
        return None, None, *gradients


class VarlenAttentionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, config, needs, csr, state, q, k, v, bias, g):
        inputs = (q, k, v, bias, g)
        ctx.config = config
        ctx.save_for_backward(*inputs, csr, state)
        ctx.set_materialize_grads(False)
        return backward.attention_backward(inputs, csr, state, config, needs)

    @staticmethod
    def backward(ctx, rq, rk, rv, rb):
        q, k, v, bias, g, csr, state = ctx.saved_tensors
        result = VarlenAttentionDoubleBackward.apply(
            ctx.config,
            ctx.needs_input_grad[4:9],
            csr,
            state,
            q,
            k,
            v,
            bias,
            g,
            rq,
            rk,
            rv,
            rb,
        )
        return (None,) * 4 + result


class VarlenAttentionDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, config, needs, csr, state, q, k, v, bias, g, rq, rk, rv, rb
    ):
        inputs = (q, k, v, bias, g, rq, rk, rv, rb)
        ctx.config = config
        ctx.save_for_backward(*inputs, csr, state)
        ctx.set_materialize_grads(False)
        return double_backward.attention_double_backward(
            inputs, csr, state, config, needs
        )

    @staticmethod
    def backward(ctx, sq, sk, sv, sb, sg):
        saved = ctx.saved_tensors
        inputs = saved[:9]
        csr, state = saved[9:]
        result = triple_backward.attention_triple_backward(
            (*inputs, sq, sk, sv, sb, sg),
            csr,
            state,
            ctx.config,
            ctx.needs_input_grad[4:13],
        )
        return (None,) * 4 + result


def attention_config(q, v, bias, csr, softmax_scale, causal):
    dim, value_dim, heads = q.shape[2], v.shape[2], q.shape[1]
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    return (dim, value_dim, heads, scale, causal, max(1, (csr.numel() - 1).bit_length()),
            0 if bias is None else 1 if bias.ndim == 2 else 2)


def register_ops():
    @torch.library.custom_op("lunaeq::attention_forward", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor csr, float? scale, bool causal) -> (Tensor, Tensor)")
    def attention_forward(q, k, v, bias, csr, scale, causal):
        config = attention_config(q, v, bias, csr, scale, causal)
        return forward.attention_forward((q, k, v, bias, None), csr, config)


    @attention_forward.register_fake
    def attention_forward_fake(q, k, v, bias, csr, scale, causal):
        return torch.empty_like(v, memory_format=torch.contiguous_format), v.new_empty((*v.shape[:2], 2), dtype=torch.float32)


    @torch.library.custom_op("lunaeq::attention_backward", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor g, Tensor csr, Tensor state, float? scale, bool causal, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?)")
    def attention_backward(q, k, v, bias, g, csr, state, scale, causal, needs):
        config = attention_config(q, v, bias, csr, scale, causal)
        return backward.attention_backward((q, k, v, bias, g), csr, state, config, needs)


    @attention_backward.register_fake
    def attention_backward_fake(q, k, v, bias, g, csr, state, scale, causal, needs):
        return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                     if need and value is not None else None
                     for value, need in zip((q, k, v, bias), needs))


    @torch.library.custom_op("lunaeq::attention_double_backward", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor g, Tensor? rq, Tensor? rk, Tensor? rv, Tensor? rb, Tensor csr, Tensor state, float? scale, bool causal, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
    def attention_double_backward(q, k, v, bias, g, rq, rk, rv, rb, csr, state, scale, causal, needs):
        config = attention_config(q, v, bias, csr, scale, causal)
        return double_backward.attention_double_backward((q, k, v, bias, g, rq, rk, rv, rb), csr, state, config, needs)


    @attention_double_backward.register_fake
    def attention_double_backward_fake(q, k, v, bias, g, rq, rk, rv, rb, csr, state, scale, causal, needs):
        return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                     if need and value is not None else None
                     for value, need in zip((q, k, v, bias, g), needs))


    @torch.library.custom_op("lunaeq::attention_triple_backward", mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor? bias, Tensor g, Tensor? rq, Tensor? rk, Tensor? rv, Tensor? rb, Tensor? sq, Tensor? sk, Tensor? sv, Tensor? sb, Tensor? sg, Tensor csr, Tensor state, float? scale, bool causal, bool[] needs) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)")
    def attention_triple_backward(q, k, v, bias, g, rq, rk, rv, rb, sq, sk, sv, sb, sg,
                                  csr, state, scale, causal, needs):
        config = attention_config(q, v, bias, csr, scale, causal)
        return triple_backward.attention_triple_backward(
            (q, k, v, bias, g, rq, rk, rv, rb, sq, sk, sv, sb, sg), csr, state, config, needs)


    @attention_triple_backward.register_fake
    def attention_triple_backward_fake(q, k, v, bias, g, rq, rk, rv, rb, sq, sk, sv, sb, sg,
                                       csr, state, scale, causal, needs):
        return tuple(torch.empty_like(value, memory_format=torch.contiguous_format)
                     if need and value is not None else None
                     for value, need in zip((q, k, v, bias, g, rq, rk, rv, rb), needs))


    def attention_forward_setup(ctx, inputs, output):
        ctx.config = inputs[5:]
        ctx.save_for_backward(*inputs[:5], output[1])
        ctx.mark_non_differentiable(output[1])
        ctx.set_materialize_grads(False)


    def attention_forward_backward(ctx, g, unused_state):
        if g is None:
            return (None,) * 7
        q, k, v, bias, csr, state = ctx.saved_tensors
        gradients = attention_backward(q, k, v, bias, g, csr, state,
                                       *ctx.config, list(ctx.needs_input_grad[:4]))
        return *gradients, None, None, None


    def attention_backward_setup(ctx, inputs, output):
        ctx.config = inputs[7:9]
        ctx.save_for_backward(*inputs[:7])
        ctx.set_materialize_grads(False)


    def attention_backward_backward(ctx, rq, rk, rv, rb):
        if all(value is None for value in (rq, rk, rv, rb)):
            return (None,) * 10
        q, k, v, bias, g, csr, state = ctx.saved_tensors
        gradients = attention_double_backward(q, k, v, bias, g, rq, rk, rv, rb, csr, state,
                                              *ctx.config, list(ctx.needs_input_grad[:5]))
        return *gradients, None, None, None, None, None


    def attention_double_backward_setup(ctx, inputs, output):
        ctx.config = inputs[11:13]
        ctx.save_for_backward(*inputs[:11])
        ctx.set_materialize_grads(False)


    def attention_double_backward_backward(ctx, sq, sk, sv, sb, sg):
        if all(value is None for value in (sq, sk, sv, sb, sg)):
            return (None,) * 14
        q, k, v, bias, g, rq, rk, rv, rb, csr, state = ctx.saved_tensors
        gradients = attention_triple_backward(q, k, v, bias, g, rq, rk, rv, rb, sq, sk, sv, sb, sg,
                                              csr, state, *ctx.config, list(ctx.needs_input_grad[:9]))
        return *gradients, None, None, None, None, None


    attention_forward.register_autograd(attention_forward_backward, setup_context=attention_forward_setup)
    attention_backward.register_autograd(attention_backward_backward, setup_context=attention_backward_setup)
    attention_double_backward.register_autograd(attention_double_backward_backward, setup_context=attention_double_backward_setup)

    return attention_forward, attention_backward, attention_double_backward, attention_triple_backward


attention_forward, attention_backward, attention_double_backward, attention_triple_backward = register_ops()
