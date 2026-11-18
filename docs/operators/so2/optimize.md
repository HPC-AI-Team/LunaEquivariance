# Model Optimization

````{py:function} LunaEquivariance.so2.models.optimize(model)

Replaces supported model layers in place with fused LunaEquivariance SO(2)
operators and returns the same `torch.nn.Module`.

Existing parameter objects and `state_dict` keys are preserved so checkpoints
and optimizers continue to work. Repeated calls are idempotent; if
installation fails on an unsupported configuration, the original module
attributes are restored before raising an error.

Supported models:

- **eSEN** (`eSEN_Backbone`, passed as the complete model with its backbone and heads)
- **UMA** (`eSCNMDBackbone`, passed as the complete model with its backbone and heads)
- **EquiformerV3** (`EquiformerV3_OC`)

:param model: A supported eSEN, UMA or EquiformerV3 `torch.nn.Module`.
:type model: `torch.nn.Module`
:returns: The same `model` instance with SO(2) operators installed in place.

````

## Examples

```python
import torch
from LunaEquivariance.so2.models import optimize

# Wrap an existing fairchem eSEN, UMA, or EquiformerV3 model on CUDA.
model = optimize(model.cuda())
```

Build the optimizer after calling `optimize(model)` (or continue using an
existing optimizer built from the same `model.parameters()`, since parameter
objects are preserved).

## Limitations

- Requires an NVIDIA GPU with compute capability 8.0 or later and FP32 inputs.
- **eSEN**: requires the complete model (backbone and heads), `mlp_type="spectral"`, `act_type="gate"` (SiLU scalars and sigmoid gates), `use_envelope=True`, and `activation_checkpointing=False`.
- **UMA**: requires the complete model (backbone and heads), `ff_type="spectral"`, and `activation_checkpointing=False` (so `activation_checkpoint_chunk_size` is `None`).
- **EquiformerV3**: does not support `gradient_checkpointing_block_list`.
