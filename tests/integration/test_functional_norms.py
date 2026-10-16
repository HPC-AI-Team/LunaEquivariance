"""Standalone normalization APIs share the Module's arithmetic and autograd."""

import pytest
import torch
from e3nn import nn, o3

from LunaEquivariance.modules.norm import LunaEqNorm
from LunaEquivariance.modules.norm_activation import LunaEqNormActivation
from LunaEquivariance.modules.equ_norm import LunaEqEquivariantNorm
from LunaEquivariance.o3.norm.interface import norm
from LunaEquivariance.nn.norm_activation.interface import norm_activation
from LunaEquivariance.nn.equ_norm.interface import equivariant_norm


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("irreps", ["2x0e + 3x1o + 2x2e", "", "0x1o + 3x2e"])
def test_norm_function_cpu(dtype, squared, irreps):
    reference = o3.Norm(irreps, squared=squared).to(dtype)
    module = LunaEqNorm(reference).to(dtype)
    x = torch.randn(5, o3.Irreps(irreps).dim, dtype=dtype, requires_grad=True)
    actual = norm(x, irreps, squared)
    torch.testing.assert_close(actual, reference(x))
    torch.testing.assert_close(module(x), actual, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "nonlinearity", [torch.tanh, torch.sigmoid, torch.nn.functional.silu]
)
def test_norm_activation_function_cpu(dtype, bias, nonlinearity):
    irreps = "2x0e + 3x1o + 2x2e"
    reference = nn.NormActivation(o3.Irreps(irreps), nonlinearity, bias=bias).to(dtype)
    module = LunaEqNormActivation(reference).to(dtype)
    x = torch.randn(5, o3.Irreps(irreps).dim, dtype=dtype, requires_grad=True)
    parameter = reference.biases if bias else None
    actual = norm_activation(x, irreps, nonlinearity, bias=parameter)
    torch.testing.assert_close(actual, reference(x))
    torch.testing.assert_close(module(x), actual, atol=0, rtol=0)
    inputs = (x, parameter) if bias else (x,)
    actual_grad = torch.autograd.grad(actual.sum(), inputs)
    expected_grad = torch.autograd.grad(reference(x).sum(), inputs)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["norm", "norm_activation", "equivariant_norm"])
def test_functional_norm_cuda_three_orders(dtype, kind):
    irreps = "2x0e + 3x1o + 2x2e"
    x = torch.randn(
        11, o3.Irreps(irreps).dim, device="cuda", dtype=dtype, requires_grad=True
    )
    if kind == "norm":
        module = LunaEqNorm(irreps, dtype=dtype).cuda()
        inputs = (x,)
        function = lambda: norm(x, irreps)
    elif kind == "norm_activation":
        module = LunaEqNormActivation(irreps, torch.tanh, bias=True, dtype=dtype).cuda()
        inputs = (x, module.biases)
        function = lambda: norm_activation(x, irreps, torch.tanh, bias=module.biases)
    else:
        module = LunaEqEquivariantNorm(irreps, dtype=dtype, device="cuda")
        inputs = (x, module.weight, module.bias)
        function = lambda: equivariant_norm(x, irreps, module.weight, module.bias)
    actual = function()
    expected = module(x)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for order in range(3):
        create_graph = order < 2
        seeds = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(
            actual, inputs, seeds, create_graph=create_graph, allow_unused=True
        )
        expected_grad = torch.autograd.grad(
            expected, inputs, seeds, create_graph=create_graph, allow_unused=True
        )
        for actual_part, expected_part in zip(actual_grad, expected_grad):
            if actual_part is None or expected_part is None:
                assert actual_part is expected_part
            else:
                torch.testing.assert_close(
                    actual_part, expected_part, atol=tolerance, rtol=tolerance
                )
        if create_graph:
            actual = torch.cat(
                [
                    value.reshape(-1)
                    for value in actual_grad
                    if value is not None and value.requires_grad
                ]
            )
            expected = torch.cat(
                [
                    value.reshape(-1)
                    for value in expected_grad
                    if value is not None and value.requires_grad
                ]
            )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("kind", ["norm", "norm_activation"])
def test_functional_norm_cuda_half_and_autocast(dtype, autocast, kind):
    irreps = o3.Irreps("2x0e + 3x1o + 2x2e")
    input_dtype = torch.float32 if autocast else dtype
    reference = (
        (
            o3.Norm(irreps)
            if kind == "norm"
            else nn.NormActivation(irreps, torch.tanh, bias=True)
        )
        .cuda()
        .to(input_dtype)
    )
    x = torch.randn(
        11, irreps.dim, device="cuda", dtype=input_dtype, requires_grad=True
    )
    inputs = (x,) if kind == "norm" else (x, reference.biases)
    with torch.autocast("cuda", dtype=dtype, enabled=autocast):
        actual = (
            norm(x, irreps)
            if kind == "norm"
            else norm_activation(x, irreps, torch.tanh, bias=reference.biases)
        )
        expected = reference(x)
    tolerance = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2}[dtype]
    if kind == "norm" and autocast:
        # Existing native Norm keeps FP32 under CUDA autocast. Compare its
        # unchanged FP32 arithmetic against the reference outside autocast.
        expected = reference(x)
        tolerance = 3e-5
        assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    seed = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, inputs, seed)
    expected_grad = torch.autograd.grad(expected, inputs, seed)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
