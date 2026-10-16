"""Dropout values and derivatives against e3nn with the same random mask."""

import pytest
import torch
from e3nn import nn, o3

import LunaEquivariance as lunaeq


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.3, True), (1.0, True), (0.3, False)]
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "middle", "empty"])
def test_dropout(device, dtype, p, training, layout):
    irreps = o3.Irreps("2x0e + 2x1o + 1x2e")
    shape = (5, 3, irreps.dim) if layout == "middle" else (5, irreps.dim)
    if layout == "empty":
        shape = (0, irreps.dim)
    x = torch.randn(*shape, device=device, dtype=dtype)
    if layout == "strided":
        x = torch.randn(shape[0], 2 * shape[-1], device=device, dtype=dtype)[:, ::2]
    x.requires_grad_()
    reference = nn.Dropout(irreps, p).train(training)
    operator = lunaeq.Dropout(irreps, p).train(training)
    torch.manual_seed(42)
    expected = reference(x)
    torch.manual_seed(42)
    actual = operator(x)
    check_derivatives(actual, expected, x)


def check_derivatives(actual, expected, x):
    torch.testing.assert_close(actual, expected)
    # A nonlinear loss tests higher derivatives using the original forward mask.
    actual_loss, expected_loss = actual.sin().sum(), expected.sin().sum()
    for order in range(4):
        actual_grad = torch.autograd.grad(actual_loss, x, create_graph=order < 3)[0]
        expected_grad = torch.autograd.grad(expected_loss, x, create_graph=order < 3)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        direction = torch.randn_like(x)
        actual_loss = (actual_grad * direction).sum()
        expected_loss = (expected_grad * direction).sum()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_functional_dropout(device):
    irreps = "2x0e + 2x1o"
    x = torch.randn(5, 8, device=device, requires_grad=True)
    torch.manual_seed(42)
    expected = nn.Dropout(irreps, 0.3)(x)
    torch.manual_seed(42)
    actual = lunaeq.dropout(x, irreps, 0.3, training=True)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(x)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, grad)[0],
        torch.autograd.grad(expected, x, grad)[0],
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(37,), (33, 1), (13, 37)])
def test_large_dropout(dtype, batch_shape):
    irreps = o3.Irreps("128x0e + 64x1o + 32x2e + 16x3o")
    x = torch.randn(
        *batch_shape, irreps.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    torch.manual_seed(42)
    expected = nn.Dropout(irreps, 0.3)(x)
    torch.manual_seed(42)
    actual = lunaeq.Dropout(irreps, 0.3)(x)
    check_derivatives(actual, expected, x)


@pytest.mark.cuda
@pytest.mark.parametrize("irreps", [
    "2x0e+1x1o+1x3e", "3x0e+4x1o+1x2e", "1x0e+1x1o",
])
def test_compact_mask_word_boundaries(irreps):
    x = torch.randn(17, o3.Irreps(irreps).dim, device="cuda", requires_grad=True)
    operator = lunaeq.Dropout(irreps, 0.35)
    reference = nn.Dropout(irreps, 0.35)
    torch.manual_seed(71)
    actual = operator(x)
    actual_rng = torch.cuda.get_rng_state()
    torch.manual_seed(71)
    expected = reference(x)
    torch.testing.assert_close(actual_rng, torch.cuda.get_rng_state(), atol=0, rtol=0)
    check_derivatives(actual, expected, x)
