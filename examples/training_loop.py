"""Train radial weights and node features with an energy/force loss."""

import torch
from LunaEquivariance import TensorProductConv, TensorProductConfig


def main():
    torch.manual_seed(17)
    tp = TensorProductConfig(
        "32x0e",
        "1x0e",
        "32x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = TensorProductConv(tp)
    src = torch.tensor([1, 2, 0, 2, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
    graph = conv.create_graph(src, dst, 3)
    radial = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.SiLU(), torch.nn.Linear(32, tp.weight_numel)
    ).cuda()
    features = torch.nn.Parameter(torch.randn(3, 32, device="cuda"))
    optimizer = torch.optim.Adam([features, *radial.parameters()], lr=1e-3)
    edge_attr = torch.ones(6, 1, device="cuda")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"
    )
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        xyz = positions.detach().requires_grad_()
        distances = (xyz[src] - xyz[dst]).norm(dim=-1, keepdim=True)
        weight = radial(distances)
        output = conv(features, edge_attr, weight, graph=graph)
        energy = output.square().mean()
        forces = -torch.autograd.grad(energy, xyz, create_graph=True)[0]
        loss = energy.square() + forces.square().mean()
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
