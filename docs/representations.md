# Representations

`Irrep(l, p)` describes one irreducible representation of degree `l` and parity `p`,
with `2l + 1` components; `Irreps` describes an ordered
list with multiplicities. `.dim` is the total feature dimension, `.num_irreps`
is the total multiplicity, and `.slices()` gives slices along the feature dimension.

```python
from LunaEquivariance import Irrep, Irreps

irreps = Irreps("8x0e + 8x1o")
assert irreps.dim == 32
blocks = irreps.slices()
```

An irrep string such as `8x0e + 8x1o` contains eight even scalars and eight odd
vectors. Each vector has three components, giving 32 features in total.
Operators accept these strings directly or an `Irreps` object.

See [Representations and feature shapes](quickstart.md#representations-and-feature-shapes)
for a worked introduction.
