# SphericalHarmonics

````{py:class} LunaEquivariance.o3.SphericalHarmonics(irreps_out, normalize, normalization="integral", irreps_in=None, *, dtype=None)

Evaluates real spherical harmonics {math}`Y^l(x)` using the basis and normalization
conventions of [e3nn.o3.SphericalHarmonics](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html):

```{math}
Y^l:\mathbb R^3\longrightarrow\mathbb R^{2l+1},\qquad Y^l(Rx)=D^l(R)Y^l(x).
```

{math}`l` is the degree and {math}`D^l(R)` is the degree-{math}`l` representation
of rotation {math}`R`. With `normalize=True`, input vectors are normalized
before evaluation. With `normalize=False`, the homogeneous polynomials are
evaluated directly on the input vectors. The output normalization is set by
`normalization`, as described below.
Repeated degrees are returned in the requested order. An existing e3nn
SphericalHarmonics module can be passed as the first argument.

:param irreps_out: Requested degrees or output representations. Degrees 0 through 12 are supported.
:type irreps_out: `int, list of int, str or Irreps`
:param normalize: Normalize input vectors before evaluating the spherical harmonics.
:type normalize: `bool`
:param normalization: Output normalization, independent of `normalize`. Default: `"integral"`. For unit input vectors:

    - `"integral"`: Each component satisfies {math}`\int_{S^2}Y^l_m(x)^2\,dx=1`.
    - `"component"`: {math}`\|Y^l(x)\|^2=2l+1`.
    - `"norm"`: {math}`\|Y^l(x)\|=1`.

:type normalization: `{"integral", "component", "norm"}`
:param irreps_in: One vector (`"1o"`) or pseudovector (`"1e"`). `None` chooses `"1e"` if the output irreps contain an odd degree with even parity, and `"1o"` otherwise. Output parity equals input parity raised to the degree.
:type irreps_in: `str, Irreps or None`
:param dtype: Data type used to initialize the module. `None` uses `torch.get_default_dtype()`, including when converting an existing module.
:type dtype: `torch.dtype or None`

```{py:method} forward(x)

:param x: Input vectors of shape `(batch_size, 3)`. Multiple leading batch dimensions are also supported and preserved in the output.
:type x: `torch.Tensor`
:returns: Tensor of shape `(batch_size, dim_out)`. For an integer degree `l`, `dim_out = 2*l + 1`; for a degree list, sum `2*l + 1` over that list; for output irreps, `dim_out = Irreps(irreps_out).dim`.

```
````

Supports FP32 and FP64 on CUDA; CPU and FP16/BF16 inputs fall back to PyTorch.

## Function

```{py:function} LunaEquivariance.o3.spherical_harmonics(l, x, normalize, normalization="integral", *, config=None)

See the module parameters above.

:param l: Requested degrees or output representations.
:type l: `int, list of int, str or Irreps`
:param config: Optional prepared configuration.
:type config: `object or None`
:returns: Tensor of shape `(batch_size, dim_out)`. For an integer degree `l`, `dim_out = 2*l + 1`; for a degree list, sum `2*l + 1` over that list; for output irreps, `dim_out = Irreps(l).dim`.

```

## Examples

```python
import torch
from LunaEquivariance import o3

num_edges = 1024
x = torch.randn(num_edges, 3, device="cuda", requires_grad=True)  # (num_edges, 3): Cartesian edge vectors
sh = o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component").cuda()
y = sh(x)  # (num_edges, sh.irreps_out.dim)
assert y.shape == (num_edges, sh.irreps_out.dim)
y.square().mean().backward()

functional = o3.spherical_harmonics([0, 1, 2, 3], x, True, "component")  # (num_edges, sh.irreps_out.dim)
torch.testing.assert_close(functional, y)
```

## Limitations

- CUDA execution requires compute capability 8.0 or later.
