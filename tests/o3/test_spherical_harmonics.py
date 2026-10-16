"""Public metadata contracts independently checked against e3nn."""

import torch
from e3nn import o3
from LunaEquivariance import SphericalHarmonics


def test_dtype_conversion_and_cpu_fallback():
    op = SphericalHarmonics([0, 1], True)
    op.double()
    op.half()
    assert op.compute_dtype == torch.float16
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    actual = op(x)
    expected = o3.SphericalHarmonics([0, 1], True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert op.last_execution_backend == "torch_compat"
    seed = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, x, seed)
    (expected_grad,) = torch.autograd.grad(expected, x, seed)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
