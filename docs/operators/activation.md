# Activation

````{py:class} LunaEquivariance.nn.Activation(irreps_in, acts, *, dtype=None)

Applies normalized activation functions to scalar irreps, as in
[e3nn.nn.Activation](https://docs.e3nn.org/en/stable/api/nn/nn_gate.html#e3nn.nn.Activation).
Non-scalar irreps are unchanged. For each copy in scalar block {math}`i`,

```{math}
y_i=\phi_i(x_i).
```

{math}`\phi_i` is the activation in `acts[i]`, including the second-moment
normalization used by e3nn. `None` gives the identity. Odd scalar inputs need
an even or odd activation; `irreps_out` describes the resulting parity.
An existing e3nn Activation can be passed as the first argument instead.

:param irreps_in: Representations of the input features.
:type irreps_in: `str or Irreps`
:param acts: One activation function per irrep block: `len(acts) == len(irreps_in)`. Each entry is a supported callable or `None` (identity); non-scalar blocks require `None`. See [Supported activations](#supported-activations).
:type acts: `list of callable or None`
:param dtype: Data type used to initialize the module. `None` uses `torch.get_default_dtype()`, including when converting an existing module.
:type dtype: `torch.dtype or None`

```{py:method} forward(features, dim=-1)

:param features: Input of shape `(batch_size, irreps_in.dim)` when `dim=-1`; otherwise the feature dimension is axis `dim`. All other dimensions are batch dimensions.
:type features: `torch.Tensor`
:param dim: Dimension containing the features. Default: `-1`.
:type dim: `int`
:returns: Tensor with the same shape as `features`.

```
````

## Function

```{py:function} LunaEquivariance.activation(features, irreps_in=None, acts=None, dim=-1)

See the module parameters above.

:returns: Tensor with the same shape as `features`.

```

## Examples

```python
import torch
from LunaEquivariance import nn

activation = nn.Activation("8x0e + 4x1o", [torch.tanh, None]).cuda()
num_atoms = 1024
# x: (num_atoms, activation.irreps_in.dim)
x = torch.randn(num_atoms, activation.irreps_in.dim, device="cuda", requires_grad=True)
y = activation(x)  # (num_atoms, activation.irreps_out.dim)
y.square().mean().backward()
```

## Supported activations

Identity (`None`), abs, ReLU, SiLU, tanh, sigmoid, Softplus and GELU.
Accepts the corresponding `torch`/`torch.nn.functional` functions and standard
`torch.nn` modules; abs uses `torch.abs` or `torch.absolute`.

Softplus supports `beta` and `threshold`; GELU supports `approximate="none"` or
`"tanh"`. Set these on `torch.nn.Softplus` / `torch.nn.GELU`, or through `functools.partial`, for
example `partial(torch.nn.functional.softplus, beta=2.0, threshold=10.0)`.

## Limitations

- GPU acceleration requires compute capability 8.0 or later.
- Only the [listed activation functions](#supported-activations) are supported. Custom functions and in-place activations are not supported.
- FP32 and FP64; CPU and FP16/BF16 inputs fall back to PyTorch.
