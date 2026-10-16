# Public API

Reference for operator modules and functions, including formulas, parameters,
input and output shapes, and examples.

```python
from LunaEquivariance import o3, nn, TensorProduct, TensorProductConv
```

- `o3`: Linear, Norm and SphericalHarmonics.
- `nn`: Activation, Gate, NormActivation, Dropout, EquivariantNorm, GraphSoftmax, VarlenAttention and VarlenAttentionConv.
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
