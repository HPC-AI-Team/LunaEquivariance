"""Native mixed backward contracts compared with the independent e3nn module."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance import SphericalHarmonics


def mixed_backward(output, x, g, rx, bx, bg):
    (first,) = torch.autograd.grad(output, x, g, create_graph=True)
    second = torch.autograd.grad(
        first, (x, g), rx, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second,
        (x, g, rx),
        (bx, bg),
        retain_graph=True,
        allow_unused=True,
    )
    return first, *second, *third


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", ("component", "norm", "integral"))
def test_native_forward_through_triple_backward(dtype, normalize, normalization):
    torch.manual_seed(932117)
    degrees = [3, 1, 3, 0]
    native = SphericalHarmonics(degrees, normalize, normalization, dtype=dtype)
    reference = o3.SphericalHarmonics(degrees, normalize, normalization).cuda()
    x = (torch.randn(5, 3, device="cuda", dtype=dtype) + 0.8).requires_grad_()
    actual, expected = native(x), reference(x)
    g = torch.randn_like(actual).requires_grad_()
    rx = torch.randn_like(x).requires_grad_()
    bx, bg = torch.randn_like(x), torch.randn_like(g)
    budget = 8e-5 if dtype == torch.float32 else 3e-12
    for value, target in zip(
        (actual, *mixed_backward(actual, x, g, rx, bx, bg)),
        (expected, *mixed_backward(expected, x, g, rx, bx, bg)),
    ):
        if value is None or target is None:
            assert value is target
        else:
            torch.testing.assert_close(value, target, rtol=budget, atol=budget)
    assert native.last_execution_backend == "cuda_fused"
    assert native.backward_backend == "cuda_fused_recompute"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_triple_backward_with_nonleaf_inputs(dtype):
    torch.manual_seed(932118)
    native = SphericalHarmonics([0, 1, 2, 3], True, "integral", dtype=dtype)
    reference = o3.SphericalHarmonics([0, 1, 2, 3], True, "integral").cuda()
    base = (torch.randn(4, 3, device="cuda", dtype=dtype) + 1.0).requires_grad_()
    x = base.sin() + 0.8
    g = torch.randn(4, 16, device="cuda", dtype=dtype).requires_grad_()
    rx = torch.randn_like(x).requires_grad_()
    bx, bg = torch.randn_like(x), torch.randn_like(g)
    actual = mixed_backward(native(x), x, g, rx, bx, bg)
    expected = mixed_backward(reference(x), x, g, rx, bx, bg)
    budget = 3e-4 if dtype == torch.float32 else 3e-11
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=budget, atol=budget)
