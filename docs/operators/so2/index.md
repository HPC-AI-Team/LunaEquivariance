# SO(2) Operators

The `LunaEquivariance.so2` namespace provides local-frame SO(2)-equivariant
operators with caller-owned parameter tensors.

Rotating node features into an edge's local coordinate frame aligns the local
{math}`m = 0` axis with the edge vector, so convolutions decouple across nonnegative
local angular orders {math}`m \in \{0, \dots, M\}` (`mmax <= lmax`). Keeping only
rows with {math}`|m| \le M` reduces rotation, projection and activation cost:

1. {py:func}`LunaEquivariance.so2.rotation` builds compact retained Wigner-D rows from `(num_edges, 3, 3)` local frames.
2. {py:func}`LunaEquivariance.so2.rotate_gather` gathers endpoint node features onto edges, applies the forward rotation and radial weights, and packs features by order `m`.
3. {py:func}`LunaEquivariance.so2.linear` applies order-wise linear projections, leaving nonzero-order complex recombination for the next fused stage.
4. {py:func}`LunaEquivariance.so2.recombine_activate` recombines complex products and applies gated or spherical-grid activations.
5. {py:func}`LunaEquivariance.so2.inverse_rotate_reduce` recombines complex products, rotates messages back to the global frame, and aggregates at receiver nodes (using sum or softmax attention weights).

Supporting operators cover attention logits
({py:func}`LunaEquivariance.so2.attention_scores`), node feed-forward blocks
({py:func}`LunaEquivariance.so2.feed_forward`), radial MLPs
({py:func}`LunaEquivariance.so2.radial_network`), segmented expert linear
maps ({py:func}`LunaEquivariance.so2.segmented_linear`), and in-place model
integration ({py:func}`LunaEquivariance.so2.models.optimize`).

See [Local-frame SO(2) convolution](../../quickstart.md#local-frame-so2-convolution)
for a complete runnable example.

## Layouts and conventions

Three tensor conventions are shared across the `so2` operators:

| Convention | Shape | Ordering |
|---|---|---|
| Degree-major node features | `(num_nodes, (lmax + 1) ** 2, channels)` | Degree `l` occupies rows `l * l : (l + 1) ** 2`, with signed order `m = -l .. +l` at `l * l + l + m`. |
| Order-sector tuple | `mmax + 1` tensors | Order `0` has shape `(num_edges, (lmax + 1) * C)`; each order `m > 0` has shape `(num_edges, 2, (lmax + 1 - m) * C)` with `+m` and `-m` planes on axis 1 and degrees `l = m .. lmax` along the last axis. |
| Compact retained rotations | `(num_edges, K)` | Keeps rows with {math}`\lvert m \rvert \le \min(l, M)` ordered `0, +1, -1, ...` within each degree `l = 0 .. lmax`, giving `K = sum((2 * min(l, mmax) + 1) * (2 * l + 1) for l in range(lmax + 1))`. |

```{toctree}
:maxdepth: 1

rotation
rotate_gather
linear
recombine_activate
inverse_rotate_reduce
attention_scores
feed_forward
radial_network
segmented_linear
optimize
```
