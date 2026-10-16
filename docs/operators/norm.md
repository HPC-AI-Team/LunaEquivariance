# Norm

````{py:class} LunaEquivariance.o3.Norm(irreps_in, squared=False, *, dtype=None)

Computes the norm or squared norm of each irrep copy, as in
[e3nn.o3.Norm](https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_norm.py).
For irrep copy {math}`x_i` (batch indices omitted),

```{math}
n_i=\lVert x_i\rVert_2.
```

With `squared=True`, returns the squared norm instead. The output contains
one scalar per irrep copy. The module has no learnable parameters. For feature
normalization with learnable scale and bias, see [EquivariantNorm](equivariant_norm.md).
An existing e3nn Norm can be passed instead of `irreps_in`.

:param irreps_in: Representations of the input features.
:type irreps_in: `str or Irreps`
:param squared: Return squared norms instead of norms. Default: `False`.
:type squared: `bool`
:param dtype: Data type used to initialize the module. If `None`, uses the default dtype or the dtype of an existing module.
:type dtype: `torch.dtype or None`

```{py:method} forward(features)

:param features: Input tensor of shape `(batch_size, irreps_in.dim)`. Multiple leading batch dimensions are also supported and preserved in the output.
:type features: `torch.Tensor`
:returns: Tensor of shape `(batch_size, irreps_in.num_irreps)`.

```
````

## Function

```{py:function} LunaEquivariance.norm(features, irreps_in, squared=False, *, config=None)

See the module parameters above.

:param config: Optional prepared norm configuration.
:type config: `object or None`
:returns: Tensor of shape `(batch_size, irreps_in.num_irreps)`.

```

## Examples

```python
import torch
from LunaEquivariance import o3

norm = o3.Norm("8x0e + 4x1o").cuda()
num_atoms = 1024
# x: (num_atoms, norm.irreps_in.dim)
x = torch.randn(num_atoms, norm.irreps_in.dim, device="cuda", requires_grad=True)
y = norm(x)  # (num_atoms, norm.irreps_in.num_irreps)
y.sum().backward()
```

## Limitations

- GPU acceleration requires compute capability 8.0 or later.
- FP32 and FP64; CPU and FP16/BF16 inputs fall back to PyTorch.
