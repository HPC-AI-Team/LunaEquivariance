# NormActivation

````{py:class} LunaEquivariance.nn.NormActivation(irreps_in, scalar_nonlinearity, normalize=True, epsilon=None, bias=False, *, dtype=None)

Applies a scalar nonlinearity to each irrep norm and rescales the input features,
as in [e3nn.nn.NormActivation](https://docs.e3nn.org/en/stable/api/nn/nn_normact.html#e3nn.nn.NormActivation)
with `normalize=True`:

```{math}
n_i=\max(\lVert x_i\rVert_2,\epsilon),\qquad y_i=\phi(n_i+b_i)\frac{x_i}{n_i}.
```

{math}`x_i` is one irrep copy, {math}`\phi` is `scalar_nonlinearity`,
and {math}`b_i` is its optional scalar bias (zero when omitted). The batch index is omitted. An existing e3nn NormActivation can be passed as the first argument.

:param irreps_in: Representations of the input features.
:type irreps_in: `str or Irreps`
:param scalar_nonlinearity: [Supported activation](activation.md#supported-activations) applied directly to each norm, without the normalization used by Activation. Use `torch.nn.Identity()` for identity.
:type scalar_nonlinearity: `callable`
:param normalize: Normalize each input irrep before multiplying by the activated norm. Only `True` is currently supported. Default: `True`.
:type normalize: `bool`
:param epsilon: Positive lower bound for each irrep norm. `None` selects `1e-8`.
:type epsilon: `float or None`
:param bias: Add a learnable bias to each norm before applying the nonlinearity. Default: `False`.
:type bias: `bool`
:param dtype: Data type used to initialize the module. If `None`, uses the default dtype or the dtype of an existing module.
:type dtype: `torch.dtype or None`

```{py:method} forward(features)

:param features: Input tensor of shape `(batch_size, irreps_in.dim)`. Multiple leading batch dimensions are also supported and preserved in the output.
:type features: `torch.Tensor`
:returns: Tensor with the same shape and representations as `features`.

```
````

## Function

```{py:function} LunaEquivariance.norm_activation(features, irreps_in, scalar_nonlinearity, normalize=True, epsilon=None, bias=None, *, config=None)

See the module parameters above.

:param bias: Bias tensor of shape `(irreps_in.num_irreps,)`. `None` disables bias.
:type bias: `torch.Tensor or None`
:param config: Optional prepared configuration.
:type config: `object or None`
:returns: Tensor with the same shape and representations as `features`.

```

## Examples

```python
import torch
from LunaEquivariance import nn

activation = nn.NormActivation("8x0e + 4x1o", torch.sigmoid, bias=True).cuda()
num_atoms = 1024
# x: (num_atoms, activation.irreps_in.dim)
x = torch.randn(num_atoms, activation.irreps_in.dim, device="cuda", requires_grad=True)
y = activation(x)  # (num_atoms, activation.irreps_out.dim)
y.square().mean().backward()
```

## Limitations

- GPU acceleration requires compute capability 8.0 or later.
- Scalar nonlinearities are limited to the [supported activation functions](activation.md#supported-activations). Custom functions and in-place activations are not supported.
- FP32 and FP64; CPU and FP16/BF16 inputs fall back to PyTorch.
