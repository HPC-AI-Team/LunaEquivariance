# Linear

````{py:function} LunaEquivariance.so2.linear(sectors, weights, bias=None, *, lmax, mmax)

Applies order-wise linear projections to local-frame SO(2) sector features
using caller-supplied weights.

An SO(2)-equivariant linear layer treats order `0` as real scalar-axis vectors
and each nonzero order {math}`m > 0` as a complex pair {math}`(+m, -m)`:

- **Order `0`**: projects `(num_edges, (lmax + 1) * Cin)` to `(num_edges, A + (lmax + 1) * Cout)`, where `A >= 0` is an optional leading scalar output block (such as gates or attention features) and `bias` adds to this output.
- **Order `m > 0`**: applies the same weight matrix of shape `(2 * (lmax + 1 - m) * Cout, (lmax + 1 - m) * Cin)`—containing stacked real and imaginary channel maps—independently to both the `+m` and `-m` planes.

The returned nonzero-order tensors hold these paired real/imaginary products
before complex recombination; the subsequent
{py:func}`LunaEquivariance.so2.recombine_activate` or
{py:func}`LunaEquivariance.so2.inverse_rotate_reduce` call fuses that
recombination directly into its kernel. All differentiable tensor inputs
support derivatives through third order.

:param sectors: Tuple of `mmax + 1` sector tensors (typically from {py:func}`LunaEquivariance.so2.rotate_gather` or {py:func}`LunaEquivariance.so2.recombine_activate`):

    - `sectors[0]`: shape `(num_edges, (lmax + 1) * Cin)` or `(num_edges, lmax + 1, Cin)`.
    - `sectors[m]` (`m > 0`): shape `(num_edges, 2, (lmax + 1 - m) * Cin)`, `(num_edges, 2 * (lmax + 1 - m), Cin)` or `(num_edges, 2, lmax + 1 - m, Cin)`.

:type sectors: `sequence of torch.Tensor`
:param weights: Tuple of `mmax + 1` projection weight matrices:

    - `weights[0]`: shape `(A + (lmax + 1) * Cout, (lmax + 1) * Cin)`.
    - `weights[m]` (`m > 0`): shape `(2 * (lmax + 1 - m) * Cout, (lmax + 1 - m) * Cin)`.

:type weights: `sequence of torch.Tensor`
:param bias: Optional order-`0` bias of shape `(A + (lmax + 1) * Cout,)`. `None` omits bias.
:type bias: `torch.Tensor or None`
:param lmax: Maximum angular degree {math}`L \ge 0`.
:type lmax: `int`
:param mmax: Maximum retained local angular order {math}`M`, with `0 <= mmax <= lmax`.
:type mmax: `int`
:returns: Tuple of `mmax + 1` tensors:

    - Order `0`: shape `(num_edges, A + (lmax + 1) * Cout)`.
    - Order `m > 0`: paired products of shape `(num_edges, 2, 2 * (lmax + 1 - m) * Cout)`.

````

## Examples

```python
import torch
from LunaEquivariance import so2

num_edges, lmax, mmax, cin, cout, aux = 64, 4, 2, 16, 8, 4
sectors = (
    torch.randn(num_edges, (lmax + 1) * cin, device="cuda"),
    *(torch.randn(num_edges, 2, (lmax + 1 - m) * cin, device="cuda") for m in range(1, mmax + 1)),
)
weights = (
    torch.randn(aux + (lmax + 1) * cout, (lmax + 1) * cin, device="cuda"),
    *(torch.randn(2 * (lmax + 1 - m) * cout, (lmax + 1 - m) * cin, device="cuda") for m in range(1, mmax + 1)),
)
bias = torch.randn(aux + (lmax + 1) * cout, device="cuda")
raw = so2.linear(sectors, weights, bias, lmax=lmax, mmax=mmax)
```

The equivalent PyTorch call is:

```python
import torch.nn.functional as F

out0 = F.linear(sectors[0].reshape(num_edges, -1), weights[0], bias)
out_orders = tuple(
    F.linear(sectors[m].reshape(num_edges * 2, -1), weights[m]).reshape(num_edges, 2, -1)
    for m in range(1, mmax + 1)
)
raw_ref = (out0, *out_orders)
```

## Limitations

- When `mmax > 0`, the order-`0` output width must be at least `(lmax + 1) * Cout`.
- Supported dtypes: FP32, FP64, FP16 and BF16 on CPU and CUDA.
