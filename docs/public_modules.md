# Using e3nn modules

LunaEquivariance provides functions and PyTorch modules. Linear, Norm,
SphericalHarmonics, Activation, Gate, NormActivation and Dropout also accept
existing e3nn modules.

Other symbols in `o3` and `nn` delegate to e3nn.

## Convert existing modules

Pass an e3nn module to the corresponding constructor to retain its configuration,
parameters, buffers and training mode:

```python
import torch
from e3nn import o3 as reference_o3, nn as reference_nn
from LunaEquivariance import o3, nn

reference = reference_o3.Linear("8x0e + 4x1o", "16x0e + 4x1o", biases=True)
linear = o3.Linear(reference)

reference_activation = reference_nn.NormActivation(reference.irreps_out, torch.sigmoid, bias=True)
activation = nn.NormActivation(reference_activation)
model = torch.nn.Sequential(linear, activation).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

```

Build the optimizer from the converted model. In a larger model, assign the converted module to the original registered attribute, such as `model.linear = o3.Linear(model.linear)`.

Activation, Gate and NormActivation support the
[listed activation functions](operators/activation.md#supported-activations),
including when converting an e3nn module.

## Checkpoints

Converted modules retain the e3nn checkpoint keys:

```python
state = reference.state_dict()
linear.load_state_dict(state, strict=True)
reference.load_state_dict(linear.state_dict(), strict=True)
```

## Convert tensor products

`TensorProduct` and `TensorProductConv` accept an existing e3nn tensor product
or a `TensorProductConfig`, preserving its weights:

```python
from LunaEquivariance import TensorProduct, TensorProductConv

product = TensorProduct(e3nn_tp)
convolution = TensorProductConv(e3nn_tp)
```

The tensor product's instructions and weight settings determine which operators
can use it. In particular, UVU convolution uses external per-edge weights.
See [Tensor Product](operators/tensor_product.md) and
[Tensor Product Convolution](operators/tensor_product_conv.md) for supported configurations.

For convolution, pass source-node features and edge features separately, and
provide the connectivity with `graph=`.

Internal weights are available as `module.weight`; checkpoint keys use the
`tp.` prefix. When loading an older checkpoint with `op.tp.` keys, rename that
prefix to `tp.` before loading.

For functional calls with explicit weights, see [Functions](quickstart.md#functions).
