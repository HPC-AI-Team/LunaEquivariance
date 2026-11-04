# Training and integration

This example builds a three-layer equivariant GNN for a single molecular graph.
Each layer combines a tensor-product convolution, radial weights and a residual
update. Scalar and vector node features are mapped to a scalar graph energy.

## A three-layer GNN

```python
import torch
from LunaEquivariance import TensorProductConv, o3, nn


class MessagePassing(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        irreps = f"{channels}x0e + {channels}x1o"
        self.conv = TensorProductConv(
            irreps, "1x0e + 1x1o", irreps,
            [(0, 0, 0, "uvu", True), (0, 1, 1, "uvu", True),
             (1, 0, 1, "uvu", True), (1, 1, 0, "uvu", True)],
            internal_weights=False, shared_weights=False,
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, channels),
            torch.nn.SiLU(),
            torch.nn.Linear(channels, self.conv.weight_numel),
        )
        self.linear = o3.Linear(irreps, irreps)
        self.activation = nn.Activation(irreps, [torch.tanh, None])

    def forward(self, x, edge_attr, distances, graph):
        weight = self.radial(distances)  # (num_edges, self.conv.weight_numel)
        messages = self.conv(x, edge_attr, weight, graph=graph)
        return self.activation(x + self.linear(messages))


class GNN(torch.nn.Module):
    def __init__(self, num_species=10, channels=8, num_layers=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_species, channels)
        self.sh = o3.SphericalHarmonics(
            "1x0e + 1x1o", normalize=True, normalization="component",
        )
        self.layers = torch.nn.ModuleList(
            [MessagePassing(channels) for _ in range(num_layers)]
        )
        self.readout = o3.Linear(f"{channels}x0e + {channels}x1o", "1x0e")

    def forward(self, species, positions, graph):
        scalars = self.embedding(species)
        vectors = scalars.new_zeros((species.shape[0], 3 * scalars.shape[1]))
        x = torch.cat((scalars, vectors), dim=-1)
        displacement = positions[graph.src] - positions[graph.dst]
        distances = displacement.norm(dim=-1, keepdim=True)  # (num_edges, 1)
        edge_attr = self.sh(displacement)  # (num_edges, 4): 1x0e + 1x1o
        for layer in self.layers:
            x = layer(x, edge_attr, distances, graph)
        return self.readout(x).sum()  # Scalar energy for one graph
```

The radial networks produce per-edge weights; the convolutions aggregate
messages at destination atoms. All layers share the same graph topology.

## Training

The small graph below illustrates one training step. In a dataset pipeline,
replace it with the current molecule's atom types, coordinates and neighbor list.

```python
model = GNN(num_layers=3).cuda()
num_atoms = 4
species = torch.tensor([0, 1, 0, 1], device="cuda")  # (num_atoms,)
positions = torch.randn(num_atoms, 3, device="cuda")  # (num_atoms, 3)
# Directed edges, sorted by destination; no self edges.
src = torch.tensor([1, 3, 0, 2, 1, 3, 0, 2], device="cuda")
dst = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device="cuda")
# Build in graph preprocessing and reuse across all three layers.
graph = TensorProductConv.create_graph(src, dst, num_atoms)

target_energy = torch.tensor(0.5, device="cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
optimizer.zero_grad(set_to_none=True)
energy = model(species, positions, graph)
loss = (energy - target_energy).square()
loss.backward()
optimizer.step()
```

Rebuild the graph when connectivity changes. Coordinates and edge features can
change while the same topology is reused.

## torch.compile

Trace the GNN's energy and force loss, including parameter gradients, then
compile the training computation with `torch.compile`.

```python
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

positions.requires_grad_(True)
target_forces = torch.zeros_like(positions)  # Replace with reference forces.
parameters = dict(model.named_parameters())


def force_loss_and_grad(parameters, species, positions, target_energy, target_forces):
    energy = functional_call(model, parameters, (species, positions, graph))
    forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    loss = (energy - target_energy).square() + (forces - target_forces).square().mean()
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    return loss, gradients


args = (parameters, species, positions, target_energy, target_forces)
traced_step = make_fx(
    force_loss_and_grad, tracing_mode="fake", _allow_non_fake_inputs=True,
)(*args)
compiled_step = torch.compile(traced_step, fullgraph=True)

with torch.no_grad():
    loss, gradients = compiled_step(*args)
for parameter, gradient in zip(parameters.values(), gradients):
    parameter.grad = gradient
optimizer.step()
```

The compiled step returns gradients directly; the optimizer applies them without
another `backward()` call. This example captures the graph topology and model
buffers; retrace when these or input shapes change.
