# VarlenAttentionConv

Combines [self-attention](varlen_attention.md) over incoming edges with
aggregation at destination atoms. Q, K and V are **edge features**; the output
contains one feature vector per **atom**, for each head.

For atom {math}`i`, let {math}`S_i` contain its incoming edges. Each head computes:

```{math}
\begin{aligned}
\alpha_{er} &= \operatorname{softmax}_{r\in S_i}
\!\left(s\,q_e^T k_r+b_{er}+m_{er}\right), \\
o_e &= \sum_{r\in S_i}\alpha_{er}v_r, \\
f_i^{\prime} &= \sum_{e\in S_i}a_e\,o_e.
\end{aligned}
```

The first two steps produce an attention output {math}`o_e` for each edge.
The last step multiplies by `envelope` ({math}`a_e`, one when omitted) and sums
at the destination atom. This edge-to-atom aggregation is the convolution.

{math}`s` is `softmax_scale`, {math}`b_{er}` is the optional bias, and
{math}`m_{er}` is zero for allowed positions and negative infinity for masked
positions. `return_attn_out=True` also returns the edge outputs {math}`o_e`.
All floating inputs support derivatives through third order.

````{py:class} LunaEquivariance.nn.VarlenAttentionConv(*, num_heads=1, softmax_scale=None, causal=False)

:param num_heads: Number of attention heads for 2D inputs. If any input is 3D, its second dimension determines the head count. Default: `1`.
:type num_heads: `int`
:param softmax_scale: Scale applied to query-key dot products. `None` uses `1 / sqrt(head_dim)`.
:type softmax_scale: `float or None`
:param causal: Mask keys after the query position within each edge group. Default: `False`.
:type causal: `bool`

```{py:method} forward(q, k, v, csr, *, envelope=None, return_attn_out=False, bias=None)

:param q: Queries of shape `(num_edges, num_heads * head_dim)` or `(num_edges, num_heads, head_dim)`.
:type q: `torch.Tensor`
:param k: Keys of shape `(num_edges, num_heads * head_dim)` or `(num_edges, num_heads, head_dim)`, with the same head count and dimension as `q`.
:type k: `torch.Tensor`
:param v: Values of shape `(num_edges, num_heads * value_dim)` or `(num_edges, num_heads, value_dim)`.
:type v: `torch.Tensor`
:param csr: Offsets of shape `(num_atoms + 1,)`, with dtype `torch.int32` or `torch.int64`. Starts at zero, is nondecreasing and ends at `num_edges`. Each segment defines both an attention sequence and the edges aggregated into one atom. Empty segments produce zero.
:type csr: `torch.Tensor`
:param envelope: Edge weights of shape `(num_edges,)` or `(num_edges, 1)`, applied to attention outputs before summation. Uses the Q/K/V dtype or FP32. `None` uses unit weights.
:type envelope: `torch.Tensor or None`
:param return_attn_out: Also return edge attention outputs before envelope scaling and aggregation. Default: `False`.
:type return_attn_out: `bool`
:param bias: Key bias of shape `(num_edges, num_heads)`, shared across queries, or pair bias of shape `(num_atoms, num_heads, max_length, max_length)`. `max_length` covers the longest segment. Uses the Q/K/V dtype or FP32. `None` adds no bias.
:type bias: `torch.Tensor or None`
:returns: Atom features of shape `(num_atoms, num_heads * value_dim)` or `(num_atoms, num_heads, value_dim)`, following the layout of `v`. With `return_attn_out=True`, returns `(output, attn_out)`, where `attn_out` has the same shape as `v` and is independent of `envelope`.

```
````

Supports FP32, FP16 and BF16 on CUDA.
Q, K and V use the same dtype. Outputs retain V's dtype, and gradients retain
the dtype of their corresponding inputs. Computation uses FP32 accumulation.

## Function

```{py:function} LunaEquivariance.varlen_attention_conv(q, k, v, csr, *, envelope=None, return_attn_out=False, num_heads=1, bias=None, softmax_scale=None, causal=False)

See the module parameters above.

:returns: Aggregated atom features, or `(output, attn_out)` when `return_attn_out=True`, with the shapes described for `forward`.

```

## Examples

The seven edges below are grouped by destination: three enter atom 0, none
enter atom 1, and four enter atom 2. Attention operates separately within
each group; aggregation produces three atom outputs.

```python
import torch
from LunaEquivariance import nn

num_atoms, num_edges = 3, 7
num_heads, head_dim, value_dim = 2, 8, 5
dtype = torch.float32  # Also supports torch.float16 and torch.bfloat16.

dst = torch.tensor([0, 0, 0, 2, 2, 2, 2], device="cuda")  # (num_edges,)
csr = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)  # (num_atoms + 1,)
# Atom 0: edges [0:3]; atom 1: no edges; atom 2: edges [3:7].
q = torch.randn(num_edges, num_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)  # (num_edges, num_heads, head_dim)
v = torch.randn(num_edges, num_heads, value_dim, device="cuda", dtype=dtype, requires_grad=True)
envelope = torch.rand(num_edges, device="cuda", requires_grad=True)  # (num_edges,)

conv = nn.VarlenAttentionConv(num_heads=num_heads)
out, attn_out = conv(q, k, v, csr, envelope=envelope, return_attn_out=True)
# attn_out: (num_edges, num_heads, value_dim), before envelope scaling.
# out: (num_atoms, num_heads, value_dim); out[1] is zero.
out.square().mean().backward()

# The default call returns only the atom features:
out = conv(q, k, v, csr, envelope=envelope)
```

The equivalent composition using `VarlenAttention` and PyTorch aggregation is:

```python
attention = nn.VarlenAttention(num_heads=num_heads)
edge_out = attention(q, k, v, csr)  # (num_edges, num_heads, value_dim)
messages = edge_out * envelope[:, None, None]
out = messages.new_zeros(num_atoms, num_heads, value_dim)
out.index_add_(0, dst, messages)  # Sum incoming edge messages at each atom.
```

## Limitations

- CUDA execution requires compute capability 8.0 or later.
- Edges in each CSR segment are contiguous, with `q`, `k`, `v`, `envelope` and key bias in the same order.
- CPU and FP64 inputs are not supported.
