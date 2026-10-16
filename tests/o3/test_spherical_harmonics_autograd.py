"""Focused regressions for SH layout, planning and replay ownership."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance.modules import spherical_harmonics as operator


def test_fallback_tracks_public_normalization():
    constructor = o3.SphericalHarmonics
    op = operator.LunaEqSphericalHarmonics("1e+2e", True, "integral", irreps_in="1e")
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    for normalize, normalization in (
        (True, "integral"),
        (True, "integral"),
        (False, "integral"),
        (False, "norm"),
        (True, "integral"),
    ):
        op.normalize, op.normalization = normalize, normalization
        actual = op(x)
        expected = constructor(op.irreps_out, normalize, normalization, op.irreps_in)(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seed = torch.randn_like(actual)
        (actual_grad,) = torch.autograd.grad(actual, x, seed)
        (expected_grad,) = torch.autograd.grad(expected, x, seed)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert not op.state_dict() and not tuple(op.children())


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ("transpose", "slice", "broadcast", "rank3"))
def test_strided_native_forward_and_derivative(layout):
    if layout == "transpose":
        x = torch.randn(3, 11, device="cuda").t()
    elif layout == "slice":
        x = torch.randn(11, 6, device="cuda")[:, ::2]
    elif layout == "broadcast":
        x = torch.randn(1, 3, device="cuda").expand(11, 3)
    else:
        x = torch.randn(2, 3, 5, device="cuda").permute(0, 2, 1)
    x.requires_grad_()
    op = operator.LunaEqSphericalHarmonics([3, 1, 3, 0], True, "component")
    actual = op(x)
    expected = o3.SphericalHarmonics([3, 1, 3, 0], True, "component").cuda()(x)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    seed = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, x, seed)
    (expected_grad,) = torch.autograd.grad(expected, x, seed)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=3e-5)
    assert op.last_execution_backend == "cuda_fused"
