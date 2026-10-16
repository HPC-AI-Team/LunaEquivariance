# EquivariantNorm

````{py:class} LunaEquivariance.nn.EquivariantNorm(irreps, *, norm="layer", grouping="merged", weighting="component", layout="mul_ir", eps=None, affine=True, bias=None, stat_weights=None, device=None, dtype=None)

Applies layer or RMS normalization to groups of irreps. Each irrep copy has
one learnable scale; even scalars can also have
a learnable bias. For a single even-scalar block with `weighting="component"`,
this reduces to PyTorch layer or RMS normalization, as shown below.

Statistics are computed independently for each input row (batch indices are omitted).
In block {math}`i`, {math}`U_i` is the multiplicity, {math}`d_i=2l_i+1` the component
count, {math}`u` the copy index and {math}`m` the component index. Layer
normalization subtracts the channel mean of each even-scalar block; all other
blocks, and all RMS-mode blocks, remain uncentered:

```{math}
\mu_i=\begin{cases}
U_i^{-1}\sum_u x_{i,u,0}&\text{layer mode and }i\text{ is }0e,\\
0&\text{otherwise},
\end{cases}
\qquad \widetilde{x}_{i,u,m}=x_{i,u,m}-\mu_i.
```

For the group {math}`G` containing block {math}`i`, compute

```{math}
q_G=\sum_{j\in G}\omega_j\sum_{u,m}\widetilde{x}_{j,u,m}^2,\qquad
 y_{i,u,m}=\gamma_{i,u}\frac{\widetilde{x}_{i,u,m}}{\sqrt{q_G+\epsilon}}
 +\mathbf{1}_{i\text{ is }0e}\,\beta_{i,u}.
```

Here {math}`\gamma` is `weight`, {math}`\beta` is `bias`, {math}`\epsilon` is `eps`,
and {math}`\omega_i` weights each block in the group statistic. If {math}`K_G` is
the number of blocks in the group, the default coefficients are:

| `weighting` | Weight {math}`\omega_i` |
|---|---|
| `component` | {math}`1/(K_G U_i d_i)` |
| `norm` | {math}`1/(K_G U_i)` |
| `inverse_dim_squared` | {math}`1/(K_G d_i^2)` |

Grouping changes the denominator; scalar means are computed separately per block.

:param irreps: Input representations.
:type irreps: `str or Irreps`
:param norm: Normalization method. Default: `"layer"`.

    - `"layer"`: Subtract the mean of each `0e` block before normalization.
    - `"rms"`: Normalize without subtracting the mean.

:type norm: `{"layer", "rms"}`
:param grouping: Blocks that share normalization statistics. Default: `"merged"`.

    - `"merged"`: Use one group for all blocks (e.g. [EquiformerV3](https://arxiv.org/abs/2604.09130), [EquFlashV2](https://github.com/SamsungDS/GGNN/blob/main/GGNN/model/EquFlashV2/nn/normalization.py), and [UMA](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/layer_norm.py)).
    - `"per_irrep"`: Use a separate group for each block (e.g. [Equiformer](https://arxiv.org/abs/2206.11990)).
    - `"separable"`: Use a separate group for each `0e` block and one group for the remaining blocks (e.g. [EquiformerV2](https://arxiv.org/abs/2306.12059)).

:type grouping: `{"merged", "per_irrep", "separable"}`
:param weighting: Weights used to compute group statistics. Default: `"component"`. See the coefficients above.

    - `"component"`: Average squared components within each block.
    - `"norm"`: Average squared irrep norms within each block.
    - `"inverse_dim_squared"`: Divide each block's sum of squares by its squared irrep dimension.

:type weighting: `{"component", "norm", "inverse_dim_squared"}`
:param layout: Feature layout within each block, where `irrep_dim = 2*l + 1`. Default: `"mul_ir"`.

    - `"mul_ir"`: Each block has shape `(multiplicity, irrep_dim)` before flattening.
    - `"ir_mul"`: Each block has shape `(irrep_dim, multiplicity)` before flattening.

:type layout: `{"mul_ir", "ir_mul"}`
:param eps: Small positive constant added to the weighted sum of squares. `None` uses `1e-5`, or `1e-12` when `weighting="inverse_dim_squared"`.
:type eps: `float or None`
:param affine: Enable a learnable scale for each irrep copy; `False` uses unit scale. Default: `True`.
:type affine: `bool`
:param bias: Enable a learnable bias for each `0e` copy. `None` selects `True` only when `affine=True` and `norm="layer"`; otherwise it selects `False`.
:type bias: `bool or None`
:param stat_weights: Positive block coefficients of length `len(irreps)`, replacing the coefficients from `weighting` without renormalization. `None` uses `weighting`.
:type stat_weights: `sequence of float or None`
:param device: Device for the affine parameters.
:type device: `torch.device, str or None`
:param dtype: Data type for the affine parameters.
:type dtype: `torch.dtype or None`

```{py:method} forward(x)

:param x: Input of shape `(num_atoms, irreps.dim)`. With `layout="ir_mul"` and equal multiplicity across all blocks, also accepts `(num_atoms, num_components, num_channels)`: `num_channels` is that multiplicity, `num_components = sum(ir.dim for _, ir in irreps)`, and `num_components * num_channels == irreps.dim`. `num_atoms` counts independent input rows.
:type x: `torch.Tensor`
:returns: Tensor with the same shape and representations as `x`.

```
````

Supports FP32 and FP64 on CUDA. CPU and FP16/BF16 inputs use the PyTorch implementation.

## Function

```{py:function} LunaEquivariance.equivariant_norm(x, irreps, weight=None, bias=None, *, norm="layer", grouping="merged", weighting="component", layout="mul_ir", eps=None, stat_weights=None, config=None)

See the module parameters above.

:param weight: Scale of shape `(irreps.num_irreps,)`. `None` uses unit scale.
:type weight: `torch.Tensor or None`
:param bias: Bias of shape `(num_scalars,)`, where `num_scalars` is the total multiplicity of `0e` blocks. `None` disables bias.
:type bias: `torch.Tensor or None`
:param config: Optional prepared normalization configuration.
:type config: `object or None`
:returns: Tensor with the same shape and representations as `x`.

```

## Examples

```python
import torch
from LunaEquivariance import Irreps, nn

irreps = Irreps("8x0e + 4x1o")
normalization = nn.EquivariantNorm(irreps, norm="layer").cuda()
num_atoms = 1024
x = torch.randn(num_atoms, irreps.dim, device="cuda", requires_grad=True)  # (num_atoms, irreps.dim)
y = normalization(x)  # (num_atoms, irreps.dim)
assert y.shape == x.shape
y.square().mean().backward()
assert normalization.weight.grad is not None
```

For a single `0e` block with `weighting="component"`,
`affine=False` and `bias=False`, it reduces to:

```python
import torch.nn.functional as F

batch_size, num_channels = 32, 8
eps = 1e-5
x = torch.randn(batch_size, num_channels, device="cuda")
for mode in ("layer", "rms"):
    normalization = nn.EquivariantNorm(
        f"{num_channels}x0e", norm=mode, affine=False, bias=False, eps=eps,
    )
    reference = F.layer_norm if mode == "layer" else F.rms_norm
    torch.testing.assert_close(
        normalization(x), reference(x, (num_channels,), eps=eps),
    )
```
