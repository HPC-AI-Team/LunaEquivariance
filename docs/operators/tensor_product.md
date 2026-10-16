# Tensor Product

````{py:class} LunaEquivariance.TensorProduct(irreps_in1, irreps_in2=None, irreps_out=None, instructions=None, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, internal_weights=None, shared_weights=None, *, dtype=None)

Computes a tensor product with the instruction, normalization and weight
conventions of [e3nn.o3.TensorProduct](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html#e3nn.o3.TensorProduct).
Instructions select one connection mode: `uuu`, `uvu` or `uvw`. The contractions
and weight shapes for each mode are given below. For graph aggregation, see
[Tensor Product Convolution](tensor_product_conv.md).

:param irreps_in1: Representations of the first input, or an existing `TensorProductConfig`/`e3nn.o3.TensorProduct`. With an existing configuration, omit the other irrep, instruction and weight-configuration arguments.
:type irreps_in1: `str, Irreps, TensorProductConfig or e3nn.o3.TensorProduct`
:param irreps_in2: Second-input representations; required when constructing from irreps.
:type irreps_in2: `str, Irreps or None`
:param irreps_out: Output representations; required when constructing from irreps.
:type irreps_out: `str, Irreps or None`
:param instructions: Paths `(i_in1, i_in2, i_out, mode, has_weight[, path_weight])`. Block indices are zero-based. All paths use the same mode: `"uuu"`, `"uvu"` or `"uvw"`. `has_weight` selects a weighted path; `path_weight` is a fixed nonnegative normalization factor (default `1.0`). Required when constructing from irreps.
:type instructions: `list of tuple or None`
:param in1_var: One variance per block in `irreps_in1`, with length `len(irreps_in1)`. `None` uses `1.0` for every block.
:type in1_var: `sequence of float or None`
:param in2_var: One variance per block in `irreps_in2`, with length `len(irreps_in2)`. `None` uses `1.0` for every block.
:type in2_var: `sequence of float or None`
:param out_var: One variance per block in `irreps_out`, with length `len(irreps_out)`. `None` uses `1.0` for every block.
:type out_var: `sequence of float or None`
:param irrep_normalization: Irrep normalization. `None` selects `"component"`.

    - `"component"`: Assume unit variance per input component; scale the output to the same convention.
    - `"norm"`: Assume unit expected squared norm per input irrep copy; scale the output to the same convention.
    - `"none"`: Do not apply irrep normalization.

:type irrep_normalization: `{"component", "norm", "none"} or None`
:param path_normalization: Path normalization. `None` selects `"element"`.

    - `"element"`: Normalize each output over all contributing input elements.
    - `"path"`: Normalize each path over its input elements, then normalize over paths.
    - `"none"`: Do not apply path normalization.

:type path_normalization: `{"element", "path", "none"} or None`
:param internal_weights: Store learnable weights in the module. Defaults to `True` when weights are shared and weighted paths exist.
:type internal_weights: `bool or None`
:param shared_weights: Use the same weights for every sample or edge. Defaults to `True`. Internal weights require `shared_weights=True`.
:type shared_weights: `bool or None`
:param dtype: Computation dtype. `None` uses the configuration weight dtype; a new configuration uses `torch.get_default_dtype()`.
:type dtype: `{torch.float32, torch.float64} or None`

```{py:method} forward(x, y, weight=None)

:param x: First input. UVU/UVW accept only `(batch_size, irreps_in1.dim)`. UUU accepts `(irreps_in1.dim,)` or any leading batch dimensions followed by `irreps_in1.dim`.
:type x: `torch.Tensor`
:param y: Second input, with last dimension `irreps_in2.dim`. UVU/UVW accept only `(batch_size, irreps_in2.dim)`, with the same `batch_size` as `x`; batch broadcasting is not supported. UUU broadcasts the leading dimensions of `x`, `y` and non-shared weights using PyTorch broadcasting rules.
:type y: `torch.Tensor`
:param weight: Shared weights `(weight_numel,)` or per-sample weights `(batch_size, weight_numel)`. UUU also accepts broadcastable leading batch dimensions for non-shared weights. UVU/UVW require the exact shapes shown. `None` uses internal weights or an unweighted product.
:type weight: `torch.Tensor or None`
:returns: Tensor of shape `(batch_size, irreps_out.dim)`. UUU broadcasts leading batch dimensions; UVU/UVW use matching batch sizes.

```
````

## Function

```{py:function} LunaEquivariance.tensor_product(x, y, weight=None, *, irreps_in1=None, irreps_in2=None, irreps_out=None, instructions=None, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, shared_weights=False, config=None)

See the module parameters above.

:param weight: Shared weights `(weight_numel,)` or per-sample weights `(batch_size, weight_numel)`. `None` is allowed only for an unweighted product.
:type weight: `torch.Tensor or None`
:param shared_weights: Use shared weights. Default: `False`.
:type shared_weights: `bool`
:param config: Result of `prepare_tensor_product(..., convolution=False)`. Supplies paths, normalization and weight sharing instead of the individual arguments.
:type config: `object or None`
:returns: Tensor of shape `(batch_size, irreps_out.dim)`. UUU broadcasts leading batch dimensions; UVU/UVW use matching batch sizes.

```

## Weight configuration

The combinations below apply to weighted paths.

| `internal_weights` | `shared_weights` | UUU | UVU | UVW | Weight argument |
|---|---|---|---|---|---|
| `True` | `True` | Yes | Yes | Yes | Optional; uses internal weights |
| `False` | `True` | Yes | Yes | Yes | Shared `(weight_numel,)` |
| `False` | `False` | Yes | Yes | Yes | Per-sample `(batch_size, weight_numel)` |
| `True` | `False` | No | No | No | — |

`internal_weights=False` means weights are supplied by the caller; it does not
make the paths unweighted. Only UUU supports `has_weight=False`.

## Connection modes

For one instruction, {math}`x` and {math}`y` are input features and {math}`z` is the output.
Indices {math}`u,v,w` label multiplicity copies; {math}`i,j,k` label irrep components.
{math}`C_{ijk}` is the real Wigner 3j tensor and {math}`\alpha` is the instruction's
normalization factor. Contributions to the same output block are summed; batch indices are omitted.

`mul_in1`, `mul_in2` and `mul_out` are the connected block multiplicities.
Flatten path weights in instruction order; `weight_numel` is their total size.
Unweighted paths contribute no weight entries.

### UUU

Input and output multiplicities are equal. Each path multiplies matching channels without summing over them. Unweighted paths use {math}`w_u=1`. Path weights have shape `(mul_in1,)`.

```{math}
z_{uk}=\alpha\,w_u\sum_{i,j}C_{ijk}\,x_{ui}y_{uj}.
```

This example broadcasts over batches and items.

```python
import torch
from LunaEquivariance import TensorProduct, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "8x1o", "8x0e",
    [(0, 0, 0, "uuu", True)],
    internal_weights=False, shared_weights=False,
)
op = TensorProduct(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "8x1o", "8x0e",
#     [(0, 0, 0, "uuu", True)],
#     internal_weights=False, shared_weights=False,
# ).to(device="cuda", dtype=torch.float32)
# op = TensorProduct(cfg)

batch_size, num_items = 4, 3
x = torch.randn(batch_size, 1, cfg.irreps_in1.dim, device="cuda")  # (batch_size, 1, cfg.irreps_in1.dim)
y = torch.randn(1, num_items, cfg.irreps_in2.dim, device="cuda")  # (1, num_items, cfg.irreps_in2.dim)
# w: (batch_size, num_items, cfg.weight_numel)
w = torch.randn(batch_size, num_items, cfg.weight_numel, device="cuda")
out = op(x, y, w)  # (batch_size, num_items, cfg.irreps_out.dim)
```

### UVU

The first-input and output multiplicities are equal. Each path sums over the second input multiplicity. All paths are weighted, with weight shape `(mul_in1, mul_in2)`.

```{math}
z_{uk}=\alpha\sum_{v,i,j}w_{uv}\,C_{ijk}\,x_{ui}y_{vj}.
```

```python
import torch
from LunaEquivariance import TensorProduct, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "2x1o", "8x0e",
    [(0, 0, 0, "uvu", True)],
    internal_weights=False, shared_weights=False,
)
op = TensorProduct(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "2x1o", "8x0e",
#     [(0, 0, 0, "uvu", True)],
#     internal_weights=False, shared_weights=False,
# ).to(device="cuda", dtype=torch.float32)
# op = TensorProduct(cfg)

num_edges = 16
x = torch.randn(num_edges, cfg.irreps_in1.dim, device="cuda")  # (num_edges, cfg.irreps_in1.dim)
y = torch.randn(num_edges, cfg.irreps_in2.dim, device="cuda")  # (num_edges, cfg.irreps_in2.dim)
w = torch.randn(num_edges, cfg.weight_numel, device="cuda")  # (num_edges, cfg.weight_numel)
out = op(x, y, w)  # (num_edges, cfg.irreps_out.dim)
```

### UVW

Including all compatible UVW paths gives the connectivity of
`e3nn.o3.FullyConnectedTensorProduct`.

The input and output multiplicities are independent. Each path sums over both input multiplicities. All paths are weighted, with weight shape `(mul_in1, mul_in2, mul_out)`.

```{math}
z_{wk}=\alpha\sum_{u,v,i,j}w_{uvw}\,C_{ijk}\,x_{ui}y_{vj}.
```

```python
import torch
from LunaEquivariance import TensorProduct, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "2x1o", "16x0e",
    [(0, 0, 0, "uvw", True)],
    internal_weights=True, shared_weights=True,
).cuda()
op = TensorProduct(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "2x1o", "16x0e",
#     [(0, 0, 0, "uvw", True)],
#     internal_weights=True, shared_weights=True,
# ).to(device="cuda", dtype=torch.float32)
# op = TensorProduct(cfg)

num_edges = 16
x = torch.randn(num_edges, cfg.irreps_in1.dim, device="cuda")  # (num_edges, cfg.irreps_in1.dim)
y = torch.randn(num_edges, cfg.irreps_in2.dim, device="cuda")  # (num_edges, cfg.irreps_in2.dim)

# Internal shared weights: (cfg.weight_numel,).
out = op(x, y)  # (num_edges, cfg.irreps_out.dim)
```

## TensorProductConfig

```{py:class} LunaEquivariance.TensorProductConfig(irreps_in1, irreps_in2, irreps_out, instructions, *, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, internal_weights=None, shared_weights=None)

Stores tensor-product instructions, normalization and optional learnable weights.
Pass this object to `TensorProduct` or `TensorProductConv`; it does not implement `forward`.

Parameters have the same meanings and defaults as
{py:class}`TensorProduct <LunaEquivariance.TensorProduct>`.

```

The attributes `irreps_in1`, `irreps_in2`, `irreps_out`, `instructions`,
`weight_numel` and `weight` describe the configured product.

The supplied `path_weight` scales a path by its square root; after variance and
irrep/path normalization, `instructions[r].path_weight` stores the coefficient
{math}`\alpha` used in the formulas.

## prepare_tensor_product

```{py:function} LunaEquivariance.prepare_tensor_product(irreps_in1, irreps_in2=None, irreps_out=None, instructions=None, *, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, shared_weights=False, dtype=torch.float32, convolution=False, mode=None)

Prepare a reusable configuration for the tensor-product functions. Irrep,
instruction and normalization arguments have the same meaning as above.
Unlike TensorProductConfig, this result has no learnable parameters.

:param irreps_in1: First-input irreps, or prebuilt UUU/UVU convolution metadata.
:type irreps_in1: `str, Irreps or Mapping`
:param irreps_in2: Second-input irreps. Required when constructing from irreps.
:type irreps_in2: `str, Irreps or None`
:param irreps_out: Output irreps. Required when constructing from irreps.
:type irreps_out: `str, Irreps or None`
:param instructions: Tensor-product paths as described above. Required when constructing from irreps.
:type instructions: `list of tuple or None`
:param shared_weights: Use shared weights in functional calls. Default: `False`.
:type shared_weights: `bool`
:param dtype: Computation dtype. Default: `torch.float32`.
:type dtype: `torch.dtype`
:param convolution: Prepare a convolution instead of a tensor product. Default: `False`.
:type convolution: `bool`
:param mode: Connection mode. `None` infers it from the instructions. Prebuilt convolution metadata requires `"uuu"` or `"uvu"`.
:type mode: `{"uuu", "uvu", "uvw"} or None`
:returns: Configuration passed as `config=` to tensor_product or tensor_product_conv.

```

Direct functional calls infer dtype from the input and cache their configuration.
For FP64 preparation, pass `dtype=torch.float64`. Convolution functions also
take an explicit graph.

## Limitations

- `internal_weights` and `shared_weights` have mode-specific restrictions; see [Weight configuration](#weight-configuration).
- For each path, UUU requires `mul_in1 = mul_in2 = mul_out`; UVU requires `mul_in1 = mul_out`.
- NVIDIA GPUs only; UVW requires compute capability 8.0 or later.
- Supported dtypes: FP32 and FP64.
