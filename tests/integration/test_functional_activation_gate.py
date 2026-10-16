"""Standalone functions share the Module's numerical implementation."""

from functools import partial

import pytest
import torch
from e3nn import nn as reference

from LunaEquivariance.modules.activation import LunaEqActivation
from LunaEquivariance.modules.gate import LunaEqGate
from LunaEquivariance.nn.activation.interface import activation
from LunaEquivariance.nn.gate.interface import gate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "act",
    [
        torch.tanh,
        torch.sigmoid,
        torch.relu,
        torch.abs,
        torch.nn.functional.silu,
        partial(torch.nn.functional.softplus, beta=2.0, threshold=15.0),
        partial(torch.nn.functional.gelu, approximate="tanh"),
        None,
    ],
)
def test_activation_cpu(dtype, act):
    irreps = "3x0e+2x1o"
    acts = [act, None]
    expected = reference.Activation(irreps, acts)
    x = torch.randn(4, 9, dtype=dtype)
    actual = activation(x, irreps, acts)
    torch.testing.assert_close(actual, expected(x), rtol=0, atol=0)
    torch.testing.assert_close(
        activation(x.t(), irreps, acts, dim=0), expected(x.t(), dim=0), rtol=0, atol=0
    )


GATE_CASES = [
    ("2x0e", [torch.tanh], "3x0e", [torch.sigmoid], "1x1e+2x2o"),
    ("2x0o", [torch.abs], "1x0o+2x0e", [torch.tanh, torch.sigmoid], "3x2o"),
    ("", [], "2x0e", [None], "1x1o+1x2e"),
    ("3x0e", [torch.tanh], "", [], ""),
    ("", [], "", [], ""),
    ("1x0e+2x0e", [torch.tanh, torch.sigmoid], "3x0e", [torch.sigmoid], "3x1o"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("args", GATE_CASES)
def test_gate_cpu(dtype, args):
    module = LunaEqGate(*args, dtype=dtype)
    official = reference.Gate(*args)
    x = torch.randn(5, module.in_dim, dtype=dtype)
    actual = gate(x, *args)
    torch.testing.assert_close(actual, module(x), rtol=0, atol=0)
    torch.testing.assert_close(
        actual,
        official(x),
        rtol=2e-6 if dtype == torch.float32 else 1e-14,
        atol=1e-6 if dtype == torch.float32 else 1e-14,
    )


def test_invalid_activation():
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "1x1e", [torch.tanh])
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "3x0o", [torch.sigmoid])
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "3x0e", [])
    with pytest.raises(ValueError):
        gate(torch.randn(2, 3), "", [], "2x0e", [torch.sigmoid], "1x1o")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["activation", "gate"])
def test_cuda_three_orders(dtype, kind):
    if kind == "activation":
        args = ("32x0e+8x1o", [torch.tanh, None])
        module = LunaEqActivation(*args, dtype=dtype).cuda()
        official = reference.Activation(*args).cuda()
        function = activation
    else:
        args = ("8x0e", [torch.tanh], "32x0e", [torch.sigmoid], "16x1o+16x2e")
        module = LunaEqGate(*args, dtype=dtype).cuda()
        official = reference.Gate(*args).cuda()
        function = gate
    x = torch.randn(19, module.in_dim, device="cuda", dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    module_x = x.detach().clone().requires_grad_()
    actual, expected, wrapped = (
        function(x, *args),
        official(reference_x),
        module(module_x),
    )
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    for order in range(4):
        torch.testing.assert_close(actual, wrapped, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        if order < 3:
            seed = torch.randn_like(actual)
            (actual,) = torch.autograd.grad(actual, x, seed, create_graph=True)
            (wrapped,) = torch.autograd.grad(wrapped, module_x, seed, create_graph=True)
            (expected,) = torch.autograd.grad(
                expected, reference_x, seed, create_graph=True
            )
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            actual = function(x, *args)
            expected = official(x)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert actual.requires_grad == expected.requires_grad
            assert actual.is_inference() == expected.is_inference()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["activation", "gate"])
def test_low_precision_torch_path(dtype, kind):
    args = ("4x0e", [torch.tanh]) if kind == "activation" else GATE_CASES[0]
    module = (LunaEqActivation if kind == "activation" else LunaEqGate)(*args)
    function = activation if kind == "activation" else gate
    x = torch.randn(module.in_dim, 5, dtype=dtype).t().requires_grad_()
    actual = function(x, *args)
    torch.testing.assert_close(actual, module(x), atol=0, rtol=0)
    actual.sum().backward()
    assert x.grad is not None
