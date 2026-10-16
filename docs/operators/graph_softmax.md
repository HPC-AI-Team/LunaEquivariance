# GraphSoftmax

````{py:class} LunaEquivariance.nn.GraphSoftmax(*, eps=1e-8, smooth_out=False)

Normalizes edge scores independently within each CSR segment and feature channel,
with an optional envelope. With `env=None` and `eps=0`, this is
`torch.softmax(data[start:end], dim=0)` for each segment. The module has no
learnable parameters. Derivatives with respect to `data` and `env` are supported
through third order.

For edge {math}`e` in segment {math}`S`, independently for each channel
(channel index omitted),

```{math}
m_S=\max_{r\in S}x_r,\qquad
z_e=\exp(x_e-m_S),\qquad
p_e=\frac{z_ea_e}{\sum_{r\in S}z_ra_r+\epsilon}.
```

The output is

```{math}
y_e=\begin{cases}
p_e, & \text{smooth_out=False},\\
a_e\,p_e, & \text{smooth_out=True}.
\end{cases}
```

Here {math}`x` is `data`, {math}`a` is `env` (one when omitted), and
{math}`\epsilon` is `eps`. The maximum is treated as a constant when computing
gradients, as in the PyTorch example below.

:param eps: Nonnegative constant added to each segment's denominator. Default: `1e-8`; `0` omits this term.
:type eps: `float`
:param smooth_out: Multiply the normalized weights by `env` once more, so the output vanishes as the envelope goes to zero. Available when `env` is provided. Default: `False`.
:type smooth_out: `bool`

```{py:method} forward(data, csr, *, env=None)

:param data: Scores of shape `(num_edges, dim)`, where `dim` is the number of independently normalized channels.
:type data: `torch.Tensor`
:param csr: Segment offsets of shape `(num_segments + 1,)`, with dtype `torch.int32` or `torch.int64`. Starts at zero, is nondecreasing and ends at `num_edges`. Segment `i` occupies rows `csr[i]:csr[i + 1]` in both `data` and `env`; repeated offsets represent empty segments.
:type csr: `torch.Tensor`
:param env: Envelope of shape `(num_edges,)`, `(num_edges, 1)` or `(num_edges, dim)`. The first two forms share a value across channels. `None` uses unit weights.
:type env: `torch.Tensor or None`
:returns: Tensor of shape `(num_edges, dim)`, in the same row order as `data`.

```
````

Supports FP32 and FP64 on CUDA. CPU and FP16/BF16 inputs use the PyTorch implementation.

## Functions

```{py:function} LunaEquivariance.graph_softmax(data, csr, *, env=None, eps=1e-8, smooth_out=False)

See the module parameters above.

:returns: Tensor of shape `(num_edges, dim)`.

```

```{py:function} LunaEquivariance.build_graph_softmax_row_ptr(segment_ids, num_segments)

Builds CSR offsets from sorted segment IDs, preserving empty segments. Build
these offsets once per graph and reuse them across layers.

:param segment_ids: Sorted CUDA integer tensor of shape `(num_edges,)`, with dtype `torch.int32` or `torch.int64`. IDs are in `[0, num_segments)`.
:type segment_ids: `torch.Tensor`
:param num_segments: Number of segments, including empty ones.
:type num_segments: `int`
:returns: CUDA `torch.int64` offsets of shape `(num_segments + 1,)`.

```

## Examples

`num_atoms` is the number of destination segments, `num_edges` the number of
edges, and `dim` the score width. Empty destination segments are retained.

```python
import torch
from LunaEquivariance import nn
from LunaEquivariance import build_graph_softmax_row_ptr

num_atoms, num_edges, dim = 3, 5, 8
# Destination 1 has no incoming edges.
dst = torch.tensor([0, 0, 2, 2, 2], device="cuda", dtype=torch.int64)
csr = build_graph_softmax_row_ptr(dst, num_atoms)  # (num_atoms + 1,): [0, 2, 2, 5]
data = torch.randn(num_edges, dim, device="cuda", requires_grad=True)
env = torch.rand(num_edges, 1, device="cuda", requires_grad=True)
softmax = nn.GraphSoftmax()
out = softmax(data, csr, env=env)  # (num_edges, dim)
assert out.shape == data.shape
out.square().sum().backward()
```

For one nonempty segment, the equivalent PyTorch expression is:

```python
# scores: (num_segment_edges, dim); envelope: (num_segment_edges, 1) or matching scores.
scores, envelope = data[:2], env[:2]
z = (scores - scores.amax(dim=0, keepdim=True).detach()).exp()
weighted = z * envelope
out = weighted / (weighted.sum(dim=0, keepdim=True) + softmax.eps)
if softmax.smooth_out:
    out = out * envelope
```

## Limitations

- Rows belonging to each segment are contiguous, with `data` and `env` in the same order. Neither the operator nor the CSR builder sorts input rows.
