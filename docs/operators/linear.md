# Linear

````{py:class} LunaEquivariance.o3.Linear(irreps_in, irreps_out, *, f_in=None, f_out=None, internal_weights=None, shared_weights=None, instructions=None, biases=False, path_normalization="element", _optimize_einsums=None, dtype=None)

Applies an equivariant linear transformation with the connections, normalization
and weight layout of [e3nn.o3.Linear](https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_linear.py).
For one connection between input and output irreps of the same degree and parity,

```{math}
y_{vi}=\alpha\sum_u w_{uv}x_{ui}+b_v.
```

`u` and `v` index input and output multiplicities, and `i` indexes irrep
components. {math}`w_{uv}` comes from `weight`, and {math}`\alpha` is the path
normalization factor. Contributions from all connections are summed, then
the bias is added once. {math}`b_v` is zero for non-scalar irreps, odd scalars,
or when bias is disabled.
With `f_in` and `f_out`, the contraction also sums over input feature channels.
An existing e3nn Linear can be passed instead of the irrep arguments.

:param irreps_in: Representations of the input features.
:type irreps_in: `str or Irreps`
:param irreps_out: Representations of the output features.
:type irreps_out: `str or Irreps`
:param f_in: Number of input feature channels, in addition to irrep multiplicities. Use `f_in` and `f_out` together; `None` omits these dimensions.
:type f_in: `int or None`
:param f_out: Optional number of output feature channels. Use together with f_in.
:type f_out: `int or None`
:param internal_weights: Store learnable weights and biases in the module. `None` uses the value of `shared_weights`; internal weights require `shared_weights=True`.
:type internal_weights: `bool or None`
:param shared_weights: Use one weight tensor for all batch elements. Defaults to `True`.
:type shared_weights: `bool or None`
:param instructions: Pairs `(i_in, i_out)` connecting input and output blocks. `None` connects all compatible blocks.
:type instructions: `list of tuple or None`
:param biases: `True` enables bias for all `0e` output blocks; `False` disables bias. A list of length `len(irreps_out)` selects blocks individually. Default: `False`.
:type biases: `bool or list of bool`
:param path_normalization: Path normalization. Default: `"element"`.

    - `"element"`: Normalize each output over all contributing input elements.
    - `"path"`: Normalize each path over its input elements, then normalize over paths.

:type path_normalization: `{"element", "path"}`
:param _optimize_einsums: Compatibility option for e3nn configuration.
:type _optimize_einsums: `bool or None`
:param dtype: Data type used to initialize the module. If `None`, uses the default dtype or the dtype of an existing module.
:type dtype: `torch.dtype or None`

`weight_numel` counts weights across all input/output block connections;
`bias_numel` counts enabled scalar output biases. The optional `f_in` and
`f_out` dimensions are separate from these counts.

```{py:method} forward(features, weight=None, bias=None)

:param features: Input of shape `(batch_size, irreps_in.dim)`, or `(batch_size, f_in, irreps_in.dim)` with feature channels. Multiple leading batch dimensions are also supported; input and weight batch dimensions broadcast.
:type features: `torch.Tensor`
:param weight: Shared weights have shape `(weight_numel,)`; unshared weights have shape `(batch_size, weight_numel)`. With feature channels, these become `(f_in, f_out, weight_numel)` and `(batch_size, f_in, f_out, weight_numel)`. Leading dimensions are broadcastable with the input batch. `None` uses the module weights.
:type weight: `torch.Tensor or None`
:param bias: Bias of shape `(bias_numel,)`, or `(f_out, bias_numel)` with feature channels. `None` uses the module bias.
:type bias: `torch.Tensor or None`
:returns: Tensor of shape `(batch_size, irreps_out.dim)`, or `(batch_size, f_out, irreps_out.dim)`.

```
````

Supports FP32 and FP64 on CUDA; CPU and FP16/BF16 inputs fall back to PyTorch.

## Function

```{py:function} LunaEquivariance.linear(features, weight=None, bias=None, *, irreps_in=None, irreps_out=None, f_in=None, f_out=None, shared_weights=True, instructions=None, biases=False, path_normalization="element", dtype=None, config=None)

See the module parameters above.

:param weight: Weights with the shapes described for `forward`. Required when `weight_numel > 0`.
:type weight: `torch.Tensor or None`
:param bias: Bias with the shapes described for `forward`; required when output biases are enabled.
:type bias: `torch.Tensor or None`
:param shared_weights: Whether weights are shared across batch items. Default: `True`.
:type shared_weights: `bool`
:param dtype: Computation dtype. `None` infers it from the supplied weights, bias, or input when no parameters are used.
:type dtype: `torch.dtype or None`
:param config: Prepared configuration, such as `module.config`. With `config`, the configuration arguments above are not used.
:type config: `object or None`
:returns: Tensor of shape `(batch_size, irreps_out.dim)`, or `(batch_size, f_out, irreps_out.dim)`.

```

## Examples

```python
import torch
from LunaEquivariance import o3

linear = o3.Linear("8x0e + 4x1o", "16x0e + 4x1o", biases=True).cuda()
num_atoms = 1024
# x: (num_atoms, linear.irreps_in.dim)
x = torch.randn(num_atoms, linear.irreps_in.dim, device="cuda", requires_grad=True)
y = linear(x)  # (num_atoms, linear.irreps_out.dim)
assert y.shape == (num_atoms, linear.irreps_out.dim)
y.square().mean().backward()
assert linear.weight.grad is not None
```

## Limitations

- CUDA execution requires compute capability 8.0 or later.
