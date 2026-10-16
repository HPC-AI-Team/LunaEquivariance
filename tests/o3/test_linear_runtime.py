"""Entry ownership, live-role reuse and saved precision for Linear."""

import pytest
import torch

from LunaEquivariance.modules import linear


@pytest.mark.cuda
@pytest.mark.parametrize(
    "layout", ("shared", "unshared", "features_broadcast", "weight_broadcast")
)
@pytest.mark.parametrize("strided", (False, True))
def test_native_layout_preparation_preserves_through_triple_backward(layout, strided):
    op = linear.LunaEqLinear(
        "1x0e", "1x0e", internal_weights=False, shared_weights=layout == "shared"
    )
    rows = 1 if layout == "features_broadcast" else 4
    x = (
        torch.randn(rows, 2, dtype=torch.float64)[:, :1]
        if strided
        else torch.randn(rows, 1, dtype=torch.float64)
    )
    x = torch.nn.Parameter(x)
    shape = (
        (1,) if layout == "shared" else (1 if layout == "weight_broadcast" else 4, 1)
    )
    weight = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    x = x.detach().cuda().requires_grad_()
    weight = weight.detach().cuda().requires_grad_()
    actual = op(x, weight)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference = reference_x * reference_weight
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_loss, reference_loss = actual.pow(4).sum(), reference.pow(4).sum()
    for order in range(1, 4):
        actual_grad = torch.autograd.grad(actual_loss, (x, weight), create_graph=True)
        reference_grad = torch.autograd.grad(
            reference_loss, (reference_x, reference_weight), create_graph=True
        )
        for actual_value, reference_value in zip(actual_grad, reference_grad):
            torch.testing.assert_close(
                actual_value, reference_value, rtol=1e-12, atol=1e-12
            )
        actual_loss = sum(value.sum() for value in actual_grad)
        reference_loss = sum(value.sum() for value in reference_grad)
