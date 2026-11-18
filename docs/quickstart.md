# Quick start

The examples below use FP32 on a GPU.

## Representations and feature shapes

An irrep string describes blocks in the last feature dimension. `8x0e` contains
eight scalar channels; `4x1o` contains four vector channels, each with three
components. Together they occupy 20 features:

```python
from LunaEquivariance import Irreps

irreps = Irreps("8x0e + 4x1o")
assert irreps.dim == 20
assert irreps.num_irreps == 12
```

`e` and `o` record even and odd parity under inversion. Within each block, the
usual layout stores all components of one copy together.

## Build a layer

A layer with trainable weights:

```python
import torch
from LunaEquivariance import o3, nn

model = torch.nn.Sequential(
    o3.Linear("8x0e + 4x1o", "16x0e + 4x1o", biases=True),
    nn.NormActivation("16x0e + 4x1o", torch.sigmoid, bias=True),
    nn.Dropout("16x0e + 4x1o", p=0.1),
).cuda()
x = torch.randn(32, 20, device="cuda", requires_grad=True)
y = model(x)  # (num_atoms, irreps_in.dim) -> (num_atoms, irreps_out.dim)
assert y.shape == (32, 28)
y.square().mean().backward()

model.eval()  # Disables dropout.
```

See [Training and integration](training.md) for optimizers and `torch.compile`,
and [Checkpoints](public_modules.md#checkpoints) for loading e3nn weights.

## Tensor product

Instructions connect irrep blocks. This example uses `uvu`: the first input and
output have eight channels, while the second input has two. Each sample supplies
its own weights.

```python
import torch
from LunaEquivariance import TensorProduct

product = TensorProduct(
    "8x1o", "2x1o", "8x0e",
    [(0, 0, 0, "uvu", True)],
    internal_weights=False, shared_weights=False,
).cuda()
x = torch.randn(16, product.irreps_in1.dim, device="cuda")  # (num_edges, product.irreps_in1.dim)
y = torch.randn(16, product.irreps_in2.dim, device="cuda")  # (num_edges, product.irreps_in2.dim)
weight = torch.randn(16, product.weight_numel, device="cuda")  # (num_edges, product.weight_numel)
out = product(x, y, weight)  # (num_edges, product.irreps_out.dim)
assert out.shape == (16, 8)
```

Use `uuu` to multiply matching channels, or `uvw` to combine all input channels. See [Weight configuration](operators/tensor_product.md#weight-configuration) for supported weight modes.

## Fused graph convolution

`src` identifies senders and `dst` identifies receivers. Build one graph for the
batch, then reuse it across convolution layers with the same connectivity.

```python
import torch
from LunaEquivariance import TensorProductConv, make_graph

conv = TensorProductConv(
    "8x1o", "2x1o", "8x0e",
    [(0, 0, 0, "uvu", True)],
    internal_weights=False, shared_weights=False,
).cuda()
src = torch.tensor([2, 0, 0, 1], device="cuda")
dst = torch.tensor([0, 1, 2, 2], device="cuda")
graph = make_graph(src, dst, num_nodes=3)  # src, dst: (num_edges,)

x = torch.randn(3, conv.irreps_in1.dim, device="cuda", requires_grad=True)  # (num_atoms, conv.irreps_in1.dim)
edge_attr = torch.randn(4, conv.irreps_in2.dim, device="cuda")  # (num_edges, conv.irreps_in2.dim)
weight = torch.randn(4, conv.weight_numel, device="cuda", requires_grad=True)  # (num_edges, conv.weight_numel)
out = conv(x, edge_attr, weight, graph=graph)  # (num_atoms, conv.irreps_out.dim)
assert out.shape == (3, 8)
out.square().mean().backward()
```

The graph builder requires ascending `dst` with paired `src` entries. Keep edge
features and weights in that same order. Both module and functional convolutions
require `graph=`.

## Functions

Use a function to pass weights and bias directly. This scalar-only Linear
example has `dim_in * dim_out` weights:

```python
import torch
from LunaEquivariance import linear, activation

x = torch.randn(32, 2, device="cuda", requires_grad=True)
weight = torch.randn(6, device="cuda", requires_grad=True)
bias = torch.zeros(3, device="cuda", requires_grad=True)
# x: (num_atoms, dim_in); weight: (dim_in * dim_out,); bias: (dim_out,)
y = linear(x, weight, bias, irreps_in="2x0e", irreps_out="3x0e", biases=True)  # (num_atoms, dim_out)
y = activation(y, "3x0e", [torch.tanh])  # (num_atoms, dim_out) -> (num_atoms, dim_out)
assert y.shape == (32, 3)
y.square().mean().backward()
```

Tensor products also accept irreps and instructions directly:

```python
import torch
from LunaEquivariance import tensor_product_conv, make_graph

src = torch.tensor([1, 2, 0], device="cuda")
dst = torch.tensor([0, 0, 1], device="cuda")
graph = make_graph(src, dst, num_nodes=3)  # src, dst: (num_edges,)
x = torch.randn(3, 4, device="cuda")
edge_attr = torch.randn(3, 1, device="cuda")
weight = torch.randn(3, 4, device="cuda")
# x: (num_atoms, dim_in1); edge_attr: (num_edges, dim_in2)
# weight: (num_edges, weight_numel); out: (num_atoms, dim_out)
out = tensor_product_conv(
    x, edge_attr, weight, graph=graph,
    irreps_in1="4x0e", irreps_in2="1x0e", irreps_out="4x0e",
    instructions=[(0, 0, 0, "uvu", True)],
)
assert out.shape == (3, 4)
```

For repeated calls, `prepare_tensor_product(..., convolution=True)` supplies a
reusable `config=`. Preparation is optional for eager functional calls.

See the [Public API](api.md) for parameters and examples for all operators,
including [GraphSoftmax](operators/graph_softmax.md),
[VarlenAttention](operators/varlen_attention.md) and the `so2` operators.

## Local-frame SO(2) convolution

The `so2` operators compose an edge convolution by rotating node features into
local edge frames, projecting and activating by angular order, and rotating back
to receiver nodes:

```python
import torch
from LunaEquivariance import make_graph, so2

num_nodes, num_edges, lmax, mmax, c = 8, 24, 4, 2, 16
dst = torch.randint(0, num_nodes, (num_edges,), device="cuda").sort().values
src = (dst + torch.randint(1, num_nodes, (num_edges,), device="cuda")) % num_nodes
graph = make_graph(src, dst, num_nodes=num_nodes)

pos = torch.randn(num_nodes, 3, device="cuda")
edge_dir = torch.nn.functional.normalize(pos[graph.src] - pos[graph.dst], dim=-1)
ref = torch.randn_like(edge_dir)
norm_z = torch.nn.functional.normalize(torch.linalg.cross(edge_dir, ref), dim=-1)
norm_y = torch.linalg.cross(norm_z, edge_dir)
frames = torch.stack((norm_z, edge_dir, norm_y), dim=1)  # (num_edges, 3, 3): row 1 aligns with the edge

x = torch.randn(num_nodes, (lmax + 1) ** 2, c, device="cuda", requires_grad=True)
rot = so2.rotation(frames, lmax=lmax, mmax=mmax)

num_groups = sum(lmax + 1 - m for m in range(mmax + 1))
radial = torch.randn(num_edges, 2 * c * num_groups, device="cuda", requires_grad=True)
sectors = so2.rotate_gather(x, rot, radial, graph, lmax=lmax, mmax=mmax)

w1 = (
    torch.randn((2 * lmax + 1) * c, (lmax + 1) * 2 * c, device="cuda", requires_grad=True),
    *(torch.randn(2 * (lmax + 1 - m) * c, (lmax + 1 - m) * 2 * c, device="cuda", requires_grad=True)
      for m in range(1, mmax + 1)),
)
w2 = (
    torch.randn((lmax + 1) * c, (lmax + 1) * c, device="cuda", requires_grad=True),
    *(torch.randn(2 * (lmax + 1 - m) * c, (lmax + 1 - m) * c, device="cuda", requires_grad=True)
      for m in range(1, mmax + 1)),
)
sectors, _ = so2.recombine_activate(
    so2.linear(sectors, w1, lmax=lmax, mmax=mmax), lmax=lmax, mmax=mmax,
)
raw = so2.linear(sectors, w2, lmax=lmax, mmax=mmax)
out = so2.inverse_rotate_reduce(raw, rot, graph, lmax=lmax, mmax=mmax)  # (num_nodes, (lmax + 1) ** 2, c)
out.square().mean().backward()
```

## Shape notation

Shape annotations use `batch_size` for independent samples. Linear, Norm, SphericalHarmonics,
Activation, Gate, NormActivation and UUU also support multiple leading batch dimensions. `.dim` is the number of scalar components
in an irrep specification, including all multiplicities; `.num_irreps` is the
number of irrep copies.

| Symbol | Meaning |
|---|---|
| `num_atoms` | Number of atoms (graph nodes); equals `graph.num_nodes` in convolutions. |
| `num_edges` | Number of directed edges. |
| `num_segments`, `num_sequences` | Number of CSR groups in GraphSoftmax and VarlenAttention, respectively. |
| `num_heads` | Number of attention heads. |
| `head_dim`, `value_dim` | Query/key and value dimensions per attention head. |
| `batch_size` | Number of independent samples or tensor products. |
| `irreps_in.dim`, `irreps_out.dim` | Input and output feature dimensions. |
| `irreps_in1.dim`, `irreps_in2.dim` | First and second tensor-product input dimensions. |
| `weight_numel` | Number of weights across all weighted paths; excludes batch dimensions. |
| `mul_in1`, `mul_in2`, `mul_out` | Multiplicities of the blocks connected by one instruction. |
| `l0_dim` | Sum of output multiplicities over scalar-output instructions, counting each path separately. |
