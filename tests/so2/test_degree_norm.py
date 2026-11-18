"""Degree-wise normalization and affine parameter gradients."""

import pytest
import torch

from LunaEquivariance.so2.utils.degree_norm import fused_equivariant_norm, fused_rms_norm
from tests.so2.checks import assert_directional_derivatives
from tests.so2.reference import norm as node_reference


def torch_reference(x, weight, bias, lmax):
    centered = torch.cat((x[:, :1] - x[:, :1].mean(-1, keepdim=True), x[:, 1:]), 1)
    variance = sum(
        centered[:, d * d : (d + 1) ** 2].square().mean((1, 2)) for d in range(lmax + 1)
    ) / (lmax + 1)
    degree = torch.tensor(
        [d for d in range(lmax + 1) for _ in range(2 * d + 1)], device=x.device
    )
    value = (
        centered * torch.rsqrt(variance + 1e-5)[:, None, None] * weight[degree][None]
    )
    return torch.cat((value[:, :1] + bias, value[:, 1:]), 1)


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,channels", [(4, 128), (4, 257), (6, 129)])
def test_large_normalization_all_derivatives(lmax, channels):
    torch.manual_seed(2026)
    x = torch.randn(2, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
    weight = torch.randn(lmax + 1, channels, device="cuda", requires_grad=True)
    bias = torch.randn(channels, device="cuda", requires_grad=True)
    inputs = (x, weight, bias)
    reference_inputs = tuple(v.detach().double().requires_grad_() for v in inputs)
    actual = fused_equivariant_norm(*inputs, 1e-5)
    expected = torch_reference(*reference_inputs, lmax)
    cotangent = torch.randn_like(x, requires_grad=True)
    ref_cotangent = cotangent.detach().double().requires_grad_()
    first = torch.autograd.grad(actual, inputs, cotangent, create_graph=True)
    ref_first = torch.autograd.grad(
        expected, reference_inputs, ref_cotangent, create_graph=True
    )
    directions = tuple(torch.randn_like(v) for v in inputs)
    second = torch.autograd.grad(
        first, (*inputs, cotangent), directions, allow_unused=True
    )
    ref_second = torch.autograd.grad(
        sum((a * b.double()).sum() for a, b in zip(ref_first, directions)),
        (*reference_inputs, ref_cotangent),
        allow_unused=True,
    )
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-5)
    for value, target in zip(first, ref_first):
        torch.testing.assert_close(value.double(), target, atol=1e-4, rtol=1e-5)
    for value, target, source in zip(second, ref_second, (*inputs, cotangent)):
        value = torch.zeros_like(source) if value is None else value
        target = torch.zeros_like(source).double() if target is None else target
        torch.testing.assert_close(value.double(), target, atol=1e-4, rtol=1e-5)


def test_degree_balanced_norm_and_storage_semantics():
    # Both degree mean squares are one, although degree one contains three rows.
    x = torch.ones(2, 4, 2, dtype=torch.float64)
    x[:, 0] = torch.tensor([1.0, 3.0])
    weight = torch.ones(2, 2, dtype=torch.float64)
    bias = torch.tensor([2.0, 4.0], dtype=torch.float64)
    result = node_reference.fused_rms_norm(
        x,
        weight,
        bias,
        lmax=1,
        epsilon=0.0,
        coefficient_major_output=True,
    )
    torch.testing.assert_close(
        result[:, 0], torch.tensor([[1.0, 5.0], [1.0, 5.0]], dtype=x.dtype)
    )
    torch.testing.assert_close(result[:, 1:], torch.ones_like(result[:, 1:]))
    assert result.stride() == (2, 4, 1)


@pytest.mark.parametrize("operation", ["rms", "equivariant"])
def test_degree_norm_rejects_cpu_input(operation):
    x = torch.randn(2, 4, 3)
    weight, bias = torch.ones(2, 3), torch.zeros(3)
    with pytest.raises(ValueError, match="CUDA FP32"):
        if operation == "rms":
            fused_rms_norm(x, weight, bias, lmax=1, epsilon=1e-5)
        else:
            fused_equivariant_norm(x, weight, bias, 1e-5)


@pytest.mark.parametrize("operation", ["rms", "equivariant"])
@pytest.mark.parametrize("invalid", ["strided_weight", "strided_bias", "different_device", "half_precision"])
def test_degree_norm_affine_contracts(operation, invalid):
    x = torch.empty(2, 4, 3)
    weight, bias = torch.empty(2, 3), torch.empty(3)
    if invalid == "strided_weight":
        weight = torch.empty(2, 6)[:, ::2]
    elif invalid == "strided_bias":
        bias = torch.empty(6)[::2]
    elif invalid == "different_device":
        weight = torch.empty(2, 3, device="meta")
    else:
        weight = weight.half()
    message = {
        "strided_weight": "contiguous",
        "strided_bias": "contiguous",
        "different_device": "share one device",
        "half_precision": "CUDA FP32",
    }[invalid]
    with pytest.raises(ValueError, match=message):
        if operation == "rms":
            fused_rms_norm(x, weight, bias, lmax=1, epsilon=1e-5)
        else:
            fused_equivariant_norm(x, weight, bias, 1e-5)


@pytest.mark.parametrize(
    "name,kwargs",
    [("fused_rms_norm", {"lmax": 1, "epsilon": 1e-4}), ("fused_equivariant_norm", {"eps": 1e-4})],
)
def test_degree_norm_reference_derivatives(name, kwargs):
    torch.manual_seed(814)
    inputs = tuple(
        (0.2 * torch.randn(shape, dtype=torch.float64)).requires_grad_()
        for shape in [(2, 4, 3), (2, 3), (3,)]
    )
    reference = getattr(node_reference, name)

    def function(*args):
        return reference(*args, **kwargs)

    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)


@pytest.mark.cuda
@pytest.mark.parametrize("operator", [fused_rms_norm, fused_equivariant_norm])
@pytest.mark.parametrize("lmax", [0, 2])
def test_degree_norm_against_fp64_reference(operator, lmax):
    torch.manual_seed(816)
    inputs = tuple(
        (0.2 * torch.randn(shape, device="cuda")).requires_grad_()
        for shape in [(2, (lmax + 1) ** 2, 3), (lmax + 1, 3), (3,)]
    )
    reference_inputs = tuple(value.detach().double().requires_grad_() for value in inputs)
    if operator is fused_rms_norm:
        actual = operator(*inputs, lmax=lmax, epsilon=1e-4)
        expected = node_reference.fused_rms_norm(*reference_inputs, lmax=lmax, epsilon=1e-4)
    else:
        actual = operator(*inputs, 1e-4)
        expected = node_reference.fused_equivariant_norm(*reference_inputs, 1e-4)
    assert_directional_derivatives(actual, expected, inputs, reference_inputs)


@pytest.mark.cuda
@pytest.mark.parametrize("operator", [fused_rms_norm, fused_equivariant_norm])
def test_degree_norm_training_tile_against_fp64(operator):
    torch.manual_seed(817)
    x = torch.randn(9, 679, 128, device="cuda").transpose(0, 1)
    inputs = tuple(
        value.requires_grad_() for value in (
            x,
            torch.randn(3, 128, device="cuda"),
            torch.randn(128, device="cuda"),
        )
    )
    reference = tuple(value.detach().double().requires_grad_() for value in inputs)
    if operator is fused_rms_norm:
        config = dict(lmax=2, epsilon=1e-5, coefficient_major_output=True)
        actual = operator(*inputs, **config)
        expected = node_reference.fused_rms_norm(*reference, **config)
    else:
        actual = operator(*inputs, 1e-5)
        expected = node_reference.fused_equivariant_norm(*reference, 1e-5)
    assert_directional_derivatives(actual, expected, inputs, reference)
