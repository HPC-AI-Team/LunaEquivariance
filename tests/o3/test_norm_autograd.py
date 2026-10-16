"""Norm higher derivatives and noncontiguous inputs."""

import pytest
import torch
from e3nn.o3 import Norm as ReferenceNorm

from LunaEquivariance.o3 import Norm


pytestmark = [
    pytest.mark.cuda,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_third_derivatives(dtype):
    reference = ReferenceNorm("2x0e+3x1o").cuda().to(dtype)
    native = Norm(reference).cuda().to(dtype)
    x = (
        torch.linspace(0.2, 1.4, 33, device="cuda", dtype=dtype)
        .reshape(3, 11)
        .requires_grad_()
    )
    actual, expected = native(x), reference(x)
    g = torch.full_like(actual, 0.23, requires_grad=True)
    rx = torch.full_like(x, 0.17, requires_grad=True)
    bx, bg = torch.full_like(x, 0.19), torch.full_like(actual, 0.31)

    def derivatives(output):
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (x, g, rx), (bx, bg)
        )
        return (first, *second, *third)

    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    for value, target in zip(derivatives(actual), derivatives(expected)):
        if value is None or target is None:
            assert value is target
        else:
            torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("layout", ["columns", "transpose", "expand"])
def test_strided_forward_and_mixed_backward(dtype, squared, layout):
    reference = ReferenceNorm("2x0e + 3x1o", squared=squared).cuda().to(dtype)
    native = Norm(reference).cuda().to(dtype)
    if layout == "columns":
        features = torch.linspace(0.2, 1.4, 132, device="cuda", dtype=dtype).reshape(
            2, 3, 22
        )[..., ::2]
    elif layout == "transpose":
        features = (
            torch.linspace(0.2, 1.4, 66, device="cuda", dtype=dtype)
            .reshape(2, 3, 11)
            .transpose(0, 1)
        )
    else:
        features = (
            torch.linspace(0.2, 1.4, 22, device="cuda", dtype=dtype)
            .reshape(2, 1, 11)
            .expand(2, 3, 11)
        )
    x = features.detach().requires_grad_()
    assert not x.is_contiguous()
    actual, expected = native(x), reference(x)
    assert native.last_execution_backend == "cuda_fused"
    g = (
        torch.linspace(0.13, 0.29, actual.numel(), device="cuda", dtype=dtype)
        .reshape_as(actual)
        .requires_grad_()
    )
    rx = (
        torch.linspace(0.17, 0.23, x.numel(), device="cuda", dtype=dtype)
        .reshape_as(x)
        .requires_grad_()
    )
    bx, bg = torch.full_like(x, 0.19), torch.full_like(actual, 0.31)

    def derivatives(output):
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(second, (x, g, rx), (bx, bg), allow_unused=True)
        return (first, *second, *third)

    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    for value, target in zip(
        (actual, *derivatives(actual)), (expected, *derivatives(expected))
    ):
        if value is None or target is None:
            assert value is target
        else:
            torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)
