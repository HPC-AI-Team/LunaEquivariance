# Tensor Product Convolution

````{py:class} LunaEquivariance.TensorProductConv(irreps_in1, irreps_in2=None, irreps_out=None, instructions=None, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, internal_weights=None, shared_weights=None, *, dtype=None, mode=None)

Fuses an [e3nn-style tensor product](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html#e3nn.o3.TensorProduct)
on each edge with summation at destination nodes. An optional envelope scales
each message before aggregation:

```{math}
f_i^{\prime}=\sum_{e:\,\mathrm{dst}[e]=i}\mathrm{env}_e\,\bigl(f_{\mathrm{src}[e]}\otimes_{w_e}y_e\bigr).
```

{math}`f_j` is a source-node feature and {math}`f'_i`
is a destination-node output. `x` stores {math}`f`, `edge_attr` stores {math}`y_e`, and
`weight` stores {math}`w_e` (or shared weights). The graph defines {math}`j=\mathrm{src}[e]` and {math}`i=\mathrm{dst}[e]`. Without `env`, messages are summed without envelope scaling. The instructions select `"uuu"`, `"uvu"` or `"uvw"`; the tensor product includes its configured normalization.

The variance and normalization arguments follow [TensorProduct](tensor_product.md).
Both operators also accept a [TensorProductConfig](tensor_product.md#tensorproductconfig).
An existing e3nn tensor product can be passed directly as `TensorProductConv(tp)`.

:param irreps_in1: Representations of the first input, or an existing `TensorProductConfig`/`e3nn.o3.TensorProduct`. With an existing configuration, omit the other irrep, instruction and weight-configuration arguments. Prebuilt metadata is also accepted with an explicit `mode`.
:type irreps_in1: `str, Irreps, TensorProductConfig, e3nn.o3.TensorProduct or Mapping`
:param irreps_in2: Second-input representations; required when constructing from irreps.
:type irreps_in2: `str, Irreps or None`
:param irreps_out: Output representations; required when constructing from irreps.
:type irreps_out: `str, Irreps or None`
:param instructions: Paths `(i_in1, i_in2, i_out, mode, has_weight[, path_weight])`, as in [TensorProduct](tensor_product.md). All paths use a single connection mode.
:type instructions: `list of tuple or None`
:param in1_var: First-input block variances.
:type in1_var: `sequence of float or None`
:param in2_var: Second-input block variances.
:type in2_var: `sequence of float or None`
:param out_var: Output block variances.
:type out_var: `sequence of float or None`
:param irrep_normalization: Irrep normalization; `None` selects `"component"`.
:type irrep_normalization: `{"component", "norm", "none"} or None`
:param path_normalization: Path normalization; `None` selects `"element"`.
:type path_normalization: `{"element", "path", "none"} or None`
:param internal_weights: Store learnable weights in the module. For UUU/UVW, defaults to `True` when weights are shared and weighted paths exist. UVU requires `False`.
:type internal_weights: `bool or None`
:param shared_weights: Share weights across edges. Defaults to `True` for UUU/UVW; UVU requires `False`. Internal weights require shared weights.
:type shared_weights: `bool or None`
:param dtype: Computation dtype. `None` uses the configuration weight dtype; a new configuration uses `torch.get_default_dtype()`. Prebuilt metadata defaults to `torch.float32`.
:type dtype: `{torch.float32, torch.float64} or None`
:param mode: Connection mode. `None` infers it from instructions; an explicit value matches the instructions. Prebuilt metadata requires `"uuu"` or `"uvu"`.
:type mode: `{"uuu", "uvu", "uvw"} or None`

```{py:method} forward(x, edge_attr, weight=None, *, graph, **kwargs)

:param x: Node features `(num_atoms, irreps_in1.dim)`. UVW also accepts `(num_input_atoms, irreps_in1.dim)` with `num_input_atoms >= num_atoms`.
:type x: `torch.Tensor`
:param edge_attr: Edge features of shape `(num_edges, irreps_in2.dim)`.
:type edge_attr: `torch.Tensor`
:param weight: Per-edge weights `(num_edges, weight_numel)` or shared weights `(weight_numel,)`. `None` uses internal weights, or omits weights for an unweighted UUU product. UUU also accepts shared weights `(1, weight_numel)`. UVU requires explicit per-edge weights.
:type weight: `torch.Tensor or None`
:param graph: Graph returned by `make_graph` or `create_graph`. Use edges sorted by destination, with source indices, edge features, per-edge weights and envelopes in the same order.
:type graph: `LunaTPGraph`
:param env: Edge envelope of shape `(num_edges,)` or `(num_edges, 1)`, multiplying messages before aggregation. `None` disables envelope scaling.
:type env: `torch.Tensor or None`
:param return_l0_feat: UUU/UVU only: return scalar path contributions before envelope scaling and aggregation. Default: `False`.
:type return_l0_feat: `bool`
:returns: Node features of shape `(num_atoms, irreps_out.dim)`, where `num_atoms = graph.num_nodes`. For UUU/UVU with `return_l0_feat=True`, returns `(out, l0_feat)`, with `l0_feat` of shape `(num_edges, l0_dim)`.

```
````

## Function

```{py:function} LunaEquivariance.tensor_product_conv(x, edge_attr, weight=None, *, graph, irreps_in1=None, irreps_in2=None, irreps_out=None, instructions=None, in1_var=None, in2_var=None, out_var=None, irrep_normalization=None, path_normalization=None, shared_weights=False, config=None, **kwargs)

See the module parameters above.

:param weight: Per-edge weights `(num_edges, weight_numel)` or shared weights `(weight_numel,)`. `None` is allowed only for an unweighted UUU product. UUU also accepts shared weights `(1, weight_numel)`. UVU requires explicit per-edge weights.
:type weight: `torch.Tensor or None`
:param shared_weights: Share weights across edges. Default: `False`. UVU supports only `False`.
:type shared_weights: `bool`
:param config: Result of `prepare_tensor_product(..., convolution=True)`. Supplies paths, normalization and weight sharing instead of the individual arguments.
:type config: `object or None`
:returns: Node features of shape `(num_atoms, irreps_out.dim)`, where `num_atoms = graph.num_nodes`. For UUU/UVU with `return_l0_feat=True`, returns `(out, l0_feat)`, with `l0_feat` of shape `(num_edges, l0_dim)`.

```

## Graph construction

```{py:function} LunaEquivariance.make_graph(src, dst, num_nodes=None)

Build a graph once and reuse it across convolution layers with the same connectivity.
`TensorProductConv.create_graph(src, dst, num_nodes)` is an equivalent entry point.

:param src: Source-node indices of shape `(num_edges,)` or `(num_edges, 1)`, with dtype `torch.int32` or `torch.int64`.
:type src: `torch.Tensor`
:param dst: Destination-node indices with the same shape options and integer dtypes as `src`, sorted in ascending order. Keep `src`, edge features, per-edge weights and envelopes in the corresponding order.
:type dst: `torch.Tensor`
:param num_nodes: Total number of nodes, including isolated nodes. If omitted, inferred as the largest index in `src` or `dst` plus one, or zero for an empty graph.
:type num_nodes: `int or None`
:returns: A `LunaTPGraph` passed as `graph=` to the convolution.

```

## Weight configuration

The combinations below apply to weighted paths.

| `internal_weights` | `shared_weights` | UUU | UVU | UVW | Weight argument |
|---|---|---|---|---|---|
| `True` | `True` | Yes | No | Yes | Optional; uses internal weights |
| `False` | `True` | Yes | No | Yes | Shared `(weight_numel,)` |
| `False` | `False` | Yes | Yes | Yes | Per-edge `(num_edges, weight_numel)` |
| `True` | `False` | No | No | No | — |

UVU convolution uses external per-edge weights only. UUU also accepts shared
weights of shape `(1, weight_numel)` and unweighted paths (`has_weight=False`).

## Connection modes

UUU, UVU and UVW share the graph aggregation above and differ in how they
connect irrep multiplicities. Their contractions and path weight shapes are
given in [Tensor Product](tensor_product.md#connection-modes). The examples below
show each mode with node and edge inputs; `weight_numel` is the weight size per edge.

### UUU

Uses [UUU connections](tensor_product.md#uuu). Supports weighted and unweighted paths, shared or per-edge weights, and optional scalar path outputs.

```python
import torch
from LunaEquivariance import TensorProductConv, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "8x1o", "8x0e",
    [(0, 0, 0, "uuu", True)],
    internal_weights=False, shared_weights=False,
)
conv = TensorProductConv(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "8x1o", "8x0e",
#     [(0, 0, 0, "uuu", True)],
#     internal_weights=False, shared_weights=False,
# ).to(device="cuda", dtype=torch.float32)
# conv = TensorProductConv(cfg)

num_atoms = 3
src = torch.tensor([2, 0, 1], device="cuda")  # (num_edges,)
dst = torch.tensor([0, 2, 2], device="cuda")  # (num_edges,)
num_edges = src.numel()
# Build once in your model's graph preprocessing module and reuse for the same graph.
graph = conv.create_graph(src, dst, num_atoms)
x = torch.randn(num_atoms, cfg.irreps_in1.dim, device="cuda")  # (num_atoms, cfg.irreps_in1.dim)
y = torch.randn(num_edges, cfg.irreps_in2.dim, device="cuda")  # (num_edges, cfg.irreps_in2.dim)
w = torch.randn(num_edges, cfg.weight_numel, device="cuda")  # (num_edges, cfg.weight_numel)
env = torch.ones(num_edges, 1, device="cuda")  # (num_edges, 1)
# out: (num_atoms, cfg.irreps_out.dim); l0: (num_edges, l0_dim)
out, l0 = conv(x, y, w, graph=graph, env=env, return_l0_feat=True)
```

### UVU

Uses [UVU connections](tensor_product.md#uvu). Requires weighted paths and external per-edge weights (`internal_weights=False`, `shared_weights=False`). Supports optional scalar path outputs.

```python
import torch
from LunaEquivariance import TensorProductConv, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "2x1o", "8x0e",
    [(0, 0, 0, "uvu", True)],
    internal_weights=False, shared_weights=False,
)
conv = TensorProductConv(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "2x1o", "8x0e",
#     [(0, 0, 0, "uvu", True)],
#     internal_weights=False, shared_weights=False,
# ).to(device="cuda", dtype=torch.float32)
# conv = TensorProductConv(cfg)

num_atoms = 3
src = torch.tensor([1, 2, 0, 1], device="cuda")  # (num_edges,)
dst = torch.tensor([0, 0, 1, 2], device="cuda")  # (num_edges,)
num_edges = src.numel()
# Build once in your model's graph preprocessing module and reuse for the same graph.
graph = conv.create_graph(src, dst, num_atoms)
x = torch.randn(num_atoms, cfg.irreps_in1.dim, device="cuda")  # (num_atoms, cfg.irreps_in1.dim)
y = torch.randn(num_edges, cfg.irreps_in2.dim, device="cuda")  # (num_edges, cfg.irreps_in2.dim)
w = torch.randn(num_edges, cfg.weight_numel, device="cuda")  # (num_edges, cfg.weight_numel)
env = torch.ones(num_edges, 1, device="cuda")  # (num_edges, 1)
out = conv(x, y, w, graph=graph, env=env)
# With return_l0_feat=True: out, l0 = conv(..., return_l0_feat=True)
# l0: (num_edges, l0_dim), before envelope scaling and aggregation.
```

### UVW

Uses [UVW connections](tensor_product.md#uvw). Supports shared or per-edge weights and returns node features only.

```python
import torch
from LunaEquivariance import TensorProductConv, TensorProductConfig

cfg = TensorProductConfig(
    "8x1o", "2x1o", "16x0e",
    [(0, 0, 0, "uvw", True)],
    internal_weights=False, shared_weights=False,
)
conv = TensorProductConv(cfg, dtype=torch.float32)

# Alternatively, initialize from e3nn:
# from e3nn import o3
# cfg = o3.TensorProduct(
#     "8x1o", "2x1o", "16x0e",
#     [(0, 0, 0, "uvw", True)],
#     internal_weights=False, shared_weights=False,
# ).to(device="cuda", dtype=torch.float32)
# conv = TensorProductConv(cfg)

num_atoms = 3
src = torch.tensor([2, 0, 1], device="cuda")  # (num_edges,)
dst = torch.tensor([0, 2, 2], device="cuda")  # (num_edges,)
num_edges = src.numel()
# Build once in your model's graph preprocessing module and reuse for the same graph.
graph = conv.create_graph(src, dst, num_atoms)
x = torch.randn(num_atoms, cfg.irreps_in1.dim, device="cuda")  # (num_atoms, cfg.irreps_in1.dim)
y = torch.randn(num_edges, cfg.irreps_in2.dim, device="cuda")  # (num_edges, cfg.irreps_in2.dim)
w = torch.randn(num_edges, cfg.weight_numel, device="cuda")  # (num_edges, cfg.weight_numel)
env = torch.ones(num_edges, 1, device="cuda")  # (num_edges, 1)
out = conv(x, y, w, graph=graph, env=env)  # (num_atoms, cfg.irreps_out.dim)
```

## Equivalent e3nn computation

For an e3nn tensor product `tp` configured
with the same paths, normalization and weights, the node output is:

```python
# x: (num_atoms, tp.irreps_in1.dim)
# edge_attr: (num_edges, tp.irreps_in2.dim)
# src, dst: (num_edges,); weight: shared or per-edge TP weights
messages = tp(x[src], edge_attr, weight)  # (num_edges, tp.irreps_out.dim)
if env is not None:  # env: (num_edges,) or (num_edges, 1)
    messages = messages * env.reshape(-1, 1)
out = messages.new_zeros(num_atoms, tp.irreps_out.dim)
out.index_add_(0, dst, messages)  # (num_atoms, tp.irreps_out.dim)
```

## Additional outputs

With `return_l0_feat=True`, UUU and UVU return `(out, l0_feat)`: aggregated
node features and scalar contributions for each edge. `l0_feat` has shape
`(num_edges, l0_dim)` and contains values before envelope scaling and aggregation.

`l0_dim` sums `mul_out` over scalar-output instructions, or is zero if there are none.
Columns are ordered by output irrep, then instruction order; paths remain separate.
For UVU, second-input channels are summed within each path. L0 features follow
the input edge order.

## Limitations

- `internal_weights` and `shared_weights` have mode-specific restrictions; see [Weight configuration](#weight-configuration).
- For each path, UUU requires `mul_in1 = mul_in2 = mul_out`; UVU requires `mul_in1 = mul_out`.
- Input edges need to be sorted by destination; see [Graph construction](#graph-construction).
- NVIDIA GPUs only; UVW and GPU graph construction require compute capability 8.0 or later.
- Supported dtypes: FP32 and FP64.
