# Gate

````{py:class} LunaEquivariance.nn.Gate(irreps_scalars=None, act_scalars=None, irreps_gates=None, act_gates=None, irreps_gated=None, *, gate=None, dtype=None)

Applies scalar activation functions and gates to equivariant features, as in
[e3nn.nn.Gate](https://docs.e3nn.org/en/stable/api/nn/nn_gate.html#e3nn.nn.Gate).
The output combines activated scalars with features multiplied by activated gates:

```{math}
\left(\bigoplus_i\phi_i(x_i)\right)\oplus\left(\bigoplus_j\phi_j(g_j)y_j\right).
```

{math}`x_i` denotes scalar features, {math}`g_j` scalar gates, and {math}`y_j` gated features.
The functions {math}`\phi_i` come from `act_scalars`; {math}`\phi_j` come from `act_gates`. Both include e3nn's second-moment normalization.
The scalar and gated outputs are concatenated. `irreps_in` describes the sorted
input representation, and `irreps_out` includes the parity changes from the
activations. An existing e3nn Gate can be passed as the first argument or as `gate=`.

:param irreps_scalars: Representations of the scalar features {math}`x_i`.
:type irreps_scalars: `str or Irreps`
:param act_scalars: One [supported activation](activation.md#supported-activations) or `None` (identity) per scalar block: `len(act_scalars) == len(irreps_scalars)`.
:type act_scalars: `list of callable or None`
:param irreps_gates: Representations of the scalar gates {math}`g_j`.
:type irreps_gates: `str or Irreps`
:param act_gates: One [supported activation](activation.md#supported-activations) or `None` (identity) per gate block: `len(act_gates) == len(irreps_gates)`.
:type act_gates: `list of callable or None`
:param irreps_gated: Representations of the gated features {math}`y_j`. Contains `irreps_gates.num_irreps` irrep copies.
:type irreps_gated: `str or Irreps`
:param gate: Existing Gate module, supplied instead of the irrep arguments.
:type gate: `e3nn.nn.Gate or None`
:param dtype: Data type used to initialize the module. `None` uses `torch.get_default_dtype()`, including when converting an existing module.
:type dtype: `torch.dtype or None`

```{py:method} forward(features)

:param features: Input tensor of shape `(batch_size, irreps_in.dim)`, ordered according to `irreps_in`. Multiple leading batch dimensions are also supported and preserved in the output.
:type features: `torch.Tensor`
:returns: Tensor of shape `(batch_size, irreps_out.dim)`.

```
````

## Function

```{py:function} LunaEquivariance.gate(features, irreps_scalars=None, act_scalars=None, irreps_gates=None, act_gates=None, irreps_gated=None)

See the module parameters above.
Input features follow `irreps_in` of the corresponding Gate module.

:returns: Tensor of shape `(batch_size, irreps_out.dim)`.

```

## Examples

```python
import torch
from LunaEquivariance import nn

gate = nn.Gate("8x0e", [torch.tanh], "4x0e", [torch.sigmoid], "4x1o").cuda()
num_atoms = 1024
# x: (num_atoms, gate.irreps_in.dim)
x = torch.randn(num_atoms, gate.irreps_in.dim, device="cuda", requires_grad=True)
y = gate(x)  # (num_atoms, gate.irreps_out.dim)
y.square().mean().backward()
```

## Limitations

- GPU acceleration requires compute capability 8.0 or later.
- Scalar and gate activations are limited to the [supported activation functions](activation.md#supported-activations). Custom functions and in-place activations are not supported.
- FP32 and FP64; CPU and FP16/BF16 inputs fall back to PyTorch.
