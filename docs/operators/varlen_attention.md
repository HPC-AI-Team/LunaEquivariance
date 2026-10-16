# VarlenAttention

````{py:class} LunaEquivariance.nn.VarlenAttention(*, num_heads=1, softmax_scale=None, causal=False)

Computes [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
independently for each sequence and attention head. Input sequences are
concatenated along the first dimension, with boundaries given by `csr`.
For one sequence and head,

```{math}
\alpha_{ij}=\operatorname{softmax}_j\!\left(s\,q_i^T k_j+b_{ij}+m_{ij}\right),\qquad f_i^{\prime}=\sum_j\alpha_{ij}v_j.
```

{math}`q_i`, {math}`k_j` and {math}`v_j` are the query, key and value vectors
at sequence positions {math}`i` and {math}`j`; {math}`\alpha_{ij}` are attention weights and {math}`f'_i` is the output. {math}`s` is
`softmax_scale`, {math}`b_{ij}` is the optional bias, and {math}`m_{ij}` is zero for allowed
positions and negative infinity for masked positions. The sums run over one CSR sequence. The operation has no learnable parameters
and applies no dropout. Derivatives with respect to `q`, `k`, `v` and `bias`
are supported through third order.

:param num_heads: Number of attention heads for 2D inputs. If any input is 3D, its second dimension determines the head count. Default: `1`.
:type num_heads: `int`
:param softmax_scale: Scale applied to query-key dot products. `None` uses `1 / sqrt(head_dim)`.
:type softmax_scale: `float or None`
:param causal: Mask keys after the query position within each sequence. Default: `False`.
:type causal: `bool`

`num_edges` counts input rows; `num_sequences` counts CSR groups. `head_dim`
and `value_dim` are the query/key and value dimensions per head.

```{py:method} forward(q, k, v, csr, *, bias=None)

:param q: Queries of shape `(num_edges, num_heads * head_dim)` or `(num_edges, num_heads, head_dim)`.
:type q: `torch.Tensor`
:param k: Keys of shape `(num_edges, num_heads * head_dim)` or `(num_edges, num_heads, head_dim)`, with the same head count and dimension as `q`.
:type k: `torch.Tensor`
:param v: Values of shape `(num_edges, num_heads * value_dim)` or `(num_edges, num_heads, value_dim)`.
:type v: `torch.Tensor`
:param csr: Sequence offsets of shape `(num_sequences + 1,)`, with dtype `torch.int32` or `torch.int64`. Starts at zero, is nondecreasing and ends at `num_edges`.
:type csr: `torch.Tensor`
:param bias: Additive key bias of shape `(num_edges, num_heads)`, or pair bias of shape `(num_sequences, num_heads, max_length, max_length)`. `max_length` is the padded sequence length, at least as large as the longest sequence. Key bias is shared across queries; pair bias depends on query and key positions. Uses the Q/K/V dtype or FP32. `None` adds no bias.
:type bias: `torch.Tensor or None`
:returns: Tensor with the same shape as `v`, in the same input row order.

```
````

Supports FP32, FP16 and BF16 on CUDA.
Q, K and V use the same dtype. Outputs retain V's dtype, and gradients retain
the dtype of their corresponding inputs. Computation uses FP32 accumulation.

## Function

```{py:function} LunaEquivariance.varlen_attention(q, k, v, csr, *, num_heads=1, bias=None, softmax_scale=None, causal=False)

See the module parameters above.

:returns: Tensor with the same shape as `v`, in the same input row order.

```

## Examples

```python
import torch
from LunaEquivariance import nn

num_edges, num_sequences = 6, 2
num_heads, head_dim, value_dim = 4, 16, 16
dtype = torch.float32  # Also supports torch.float16 and torch.bfloat16.
attention = nn.VarlenAttention(num_heads=num_heads)
# Two independent groups: rows [0, 3) and [3, 6).
csr = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)  # (num_sequences + 1,)
# q: (num_edges, num_heads * head_dim)
q = torch.randn(num_edges, num_heads * head_dim, device="cuda", dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)  # (num_edges, num_heads * head_dim)
# v: (num_edges, num_heads * value_dim)
v = torch.randn(num_edges, num_heads * value_dim, device="cuda", dtype=dtype, requires_grad=True)
out = attention(q, k, v, csr)  # (num_edges, num_heads * value_dim)
assert out.shape == v.shape
out.square().mean().backward()
```

For the first sequence above, the corresponding PyTorch call is:

```python
import torch.nn.functional as F

sequence_length = 3
# q_seq, k_seq: (num_heads, sequence_length, head_dim)
q_seq = q[:sequence_length].unflatten(1, (num_heads, head_dim)).transpose(0, 1)
k_seq = k[:sequence_length].unflatten(1, (num_heads, head_dim)).transpose(0, 1)
# v_seq: (num_heads, sequence_length, value_dim)
v_seq = v[:sequence_length].unflatten(1, (num_heads, value_dim)).transpose(0, 1)
reference = F.scaled_dot_product_attention(
    q_seq, k_seq, v_seq, dropout_p=0.0,
    is_causal=attention.causal, scale=attention.softmax_scale,
).transpose(0, 1).flatten(1)  # (sequence_length, num_heads * value_dim)
```

Inputs can also use a separate head dimension. 2D and 3D inputs can be mixed
when their head counts agree. The output has the same shape as `v`.

```python
q_heads = q.unflatten(1, (num_heads, head_dim))  # (num_edges, num_heads, head_dim)
k_heads = k.unflatten(1, (num_heads, head_dim))
v_heads = v.unflatten(1, (num_heads, value_dim))
out = attention(q_heads, k_heads, v_heads, csr)  # (num_edges, num_heads, value_dim)
```

## Sequence order

Sequence `i` occupies rows `csr[i]:csr[i + 1]` in `q`, `k`, `v` and key bias.
Outputs keep this order.
The operator does not sort or reorder tensors. Keep `q`, `k`, `v` and key bias
in the same sequence order. With `causal=True`, this order determines which keys
each query can attend to. If positions within a sequence are reordered, apply
the same permutation to both position axes of pair bias.

## Limitations

- CUDA execution requires compute capability 8.0 or later.
- CPU and FP64 inputs are not supported.
