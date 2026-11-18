# Public API

Reference for operator modules and functions, including formulas, parameters,
input and output shapes, and examples.

```python
from LunaEquivariance import o3, nn, so2, TensorProduct, TensorProductConv
```

- `o3`: Linear, Norm, SphericalHarmonics and SymmetricContraction.
- `nn`: Activation, Gate, NormActivation, Dropout, EquivariantNorm, GraphSoftmax, VarlenAttention and VarlenAttentionConv.
- `so2`: rotation, rotate_gather, linear, recombine_activate, inverse_rotate_reduce, attention_scores, feed_forward, radial_network, segmented_linear and models.optimize.
- `TensorProduct` and `TensorProductConv`: tensor products and graph convolutions with `uuu`, `uvu` or `uvw` connections.

Modules store configuration and any trainable parameters. Functions accept
weights as arguments. See [Quick start](quickstart.md) for examples.

## Representations

```{toctree}
:maxdepth: 1

Irrep and Irreps <representations>
```

## Operators

```{toctree}
:glob:
:maxdepth: 1

operators/*
```

## SO(2) Operators

```{toctree}
:maxdepth: 2

operators/so2/index
```
