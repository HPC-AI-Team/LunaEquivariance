"""Replace an external-weight e3nn UVU tensor product in an eager model."""

import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct

from LunaEquivariance import TensorProduct


def main():
    torch.manual_seed(21)
    tp = E3nnTensorProduct(
        "32x0e",
        "1x0e",
        "32x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    reference = tp
    tp = TensorProduct(reference)
    inputs = (
        torch.randn(128, 32, device="cuda", requires_grad=True),
        torch.randn(128, 1, device="cuda", requires_grad=True),
        torch.randn(128, 32, device="cuda", requires_grad=True),
    )
    expected, actual = reference(*inputs), tp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    expected_grads = torch.autograd.grad(expected.square().mean(), inputs)
    actual_grads = torch.autograd.grad(actual.square().mean(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-4, atol=5e-5)
    print("e3nn replacement: outputs and all operand gradients agree")


if __name__ == "__main__":
    main()
