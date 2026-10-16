"""Explicit Norm CUDA preserves missing seeds, zero inputs and index widths."""

import pytest
import torch
from e3nn.o3 import Norm as ReferenceNorm
from LunaEquivariance import Norm

pytestmark = [
    pytest.mark.cuda,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("seed", ["both", "features", "grad_out"])
def test_partial_third_derivatives(dtype, squared, seed):
    irreps = "2x0e+3x1o+2x2e"
    native = Norm(irreps, squared=squared, dtype=dtype).cuda()
    reference = ReferenceNorm(irreps, squared=squared).cuda().to(dtype)
    torch.manual_seed(371)
    x = (torch.randn(4, 21, device="cuda", dtype=dtype) + 0.3).requires_grad_()
    g = torch.randn(4, 7, device="cuda", dtype=dtype, requires_grad=True)
    rx = torch.randn_like(x, requires_grad=True)
    bx = torch.randn_like(x) if seed != "grad_out" else None
    bg = torch.randn_like(g) if seed != "features" else None
    results = []
    for model in (reference, native):
        output = model(x)
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        pairs = [
            (value, direction)
            for value, direction in zip(second, (bx, bg))
            if direction is not None
        ]
        third = torch.autograd.grad(
            tuple((v for v, _ in pairs)),
            (x, g, rx),
            tuple((d for _, d in pairs)),
            allow_unused=True,
        )
        results.append((output, first, *second, *third))
    tolerance = 3e-05 if dtype == torch.float32 else 3e-12
    for expected, actual in zip(*results):
        if expected is None or actual is None:
            assert expected is actual
        else:
            torch.testing.assert_close(
                actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
def test_zero_nonfinite_empty_and_strided(dtype, squared):
    irreps = "2x0e+3x1o+2x2e"
    native = Norm(irreps, squared=squared, dtype=dtype).cuda()
    reference = ReferenceNorm(irreps, squared=squared).cuda().to(dtype)
    tiny = 1e-30 if dtype == torch.float32 else 1e-200
    rows = torch.tensor(
        [0.0, tiny, float("inf"), float("nan"), 0.75], dtype=dtype, device="cuda"
    )
    x = rows[:, None].expand(-1, 21).detach().requires_grad_()
    expected, actual = (reference(x), native(x))
    tolerance = 3e-05 if dtype == torch.float32 else 3e-12
    for order in range(4):
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
        )
        if order < 3:
            assert actual.requires_grad == expected.requires_grad
            if not expected.requires_grad:
                break
            expected = torch.autograd.grad(
                expected.sum(), x, create_graph=order < 2, retain_graph=True
            )[0]
            actual = torch.autograd.grad(
                actual.sum(), x, create_graph=order < 2, retain_graph=True
            )[0]
    empty = torch.empty(2, 0, 21, dtype=dtype, device="cuda", requires_grad=True)
    torch.testing.assert_close(native(empty), reference(empty))
    torch.testing.assert_close(
        torch.autograd.grad(native(empty).sum(), empty)[0], empty
    )
