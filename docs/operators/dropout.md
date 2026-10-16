# Dropout

````{py:class} LunaEquivariance.nn.Dropout(irreps, p)

Randomly zeros entire irrep copies during training, as in
[e3nn.nn.Dropout](https://github.com/e3nn/e3nn/blob/main/e3nn/nn/_dropout.py).
Components of an irrep copy share a mask. For `0 < p < 1`
(intermediate dimensions omitted),

```{math}
y_{bi}=\frac{m_{bi}}{1-p}x_{bi},\qquad m_{bi}\sim\operatorname{Bernoulli}(1-p).
```

{math}`b` indexes batch items and {math}`i` indexes irrep copies.
Each {math}`x_{bi}` is a vector of {math}`2l_i+1` components. The mask is shared
across components and intermediate dimensions, and sampled independently
for each batch item and irrep copy. Evaluation returns the input unchanged.
During training, `p <= 0` keeps all irrep copies and `p >= 1` drops all copies.
An existing e3nn Dropout can be passed instead of `irreps` and `p`.

:param irreps: Representations of the input features.
:type irreps: `str or Irreps`
:param p: Probability of dropping an irrep copy.
:type p: `float`

```{py:method} forward(x)

:param x: Input tensor of shape `(batch_size, irreps.dim)` or `(batch_size, num_items, irreps.dim)`. `batch_size` counts independent samples; `num_items` counts items sharing a mask. Additional intermediate dimensions also share the mask.
:type x: `torch.Tensor`
:returns: Tensor with the same shape as `x`.

```
````

Supports FP32 and FP64 on CUDA; CPU and FP16/BF16 inputs fall back to PyTorch.

## Function

```{py:function} LunaEquivariance.dropout(x, irreps, p, training=True, *, config=None)

See the module parameters above.

:param training: Apply dropout when `True`. Default: `True`.
:type training: `bool`
:param config: Optional prepared configuration.
:type config: `object or None`
:returns: Tensor with the same shape as `x`.

```

## Examples

`batch_size` is the number of samples, `num_atoms` the atoms per sample, and `irreps.dim` the feature dimension. The dropout mask is shared across atoms within each sample.

```python
import torch
from LunaEquivariance import nn

dropout = nn.Dropout("8x0e + 4x1o", p=0.1).cuda()
batch_size, num_atoms = 32, 128
# x: (batch_size, num_atoms, dropout.irreps.dim)
x = torch.randn(batch_size, num_atoms, dropout.irreps.dim, device="cuda", requires_grad=True)
y = dropout(x)  # (batch_size, num_atoms, dropout.irreps.dim)
assert y.shape == x.shape
y.square().mean().backward()

dropout.eval()
assert dropout(x) is x
```

## Limitations

- CUDA kernels require compute capability 8.0 or later. On older GPUs, eager execution falls back to PyTorch.
