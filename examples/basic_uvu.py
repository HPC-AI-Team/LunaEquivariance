"""Independent tensor products, followed by a differentiable training loss."""

import torch
from LunaEquivariance import TensorProduct, TensorProductConfig


def main():
    torch.manual_seed(7)
    tp = TensorProductConfig(
        "32x0e",
        "1x0e",
        "32x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    op = TensorProduct(tp)
    x = torch.randn(128, 32, device="cuda", requires_grad=True)
    edge_attr = torch.randn(128, 1, device="cuda", requires_grad=True)
    weight = torch.randn(128, tp.weight_numel, device="cuda", requires_grad=True)
    out = op(x, edge_attr, weight)
    torch.testing.assert_close(out, x * edge_attr * weight, rtol=5e-4, atol=5e-5)
    out.square().mean().backward()
    print("output:", tuple(out.shape), "input gradient:", tuple(x.grad.shape))


if __name__ == "__main__":
    main()
