"""Coefficient gates on node and edge layouts."""

import pytest
import torch
from torch.nn import functional as F

from LunaEquivariance.so2.utils.coefficient_gate import fused_gate, node_gate
from tests.so2.checks import assert_directional_derivatives
from tests.so2.reference.gate import node_gate as reference_node_gate


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,channels", [(2, 16), (4, 64)])
def test_node_gate_outputs_and_two_derivatives(lmax, channels):
    x = torch.randn(5, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
    gates = torch.randn(5, lmax * channels, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_g = gates.detach().clone().requires_grad_()
    parts = [F.silu(ref_x[:, :1])]
    for degree in range(1, lmax + 1):
        gate = ref_g.reshape(5, lmax, channels)[:, degree - 1 : degree].sigmoid()
        parts.append(ref_x[:, degree * degree : (degree + 1) ** 2] * gate)
    expected = torch.cat(parts, dim=1)
    actual = node_gate(gates, x, lmax=lmax)
    cotangent = torch.randn_like(actual)
    directions = (torch.randn_like(x), torch.randn_like(gates))
    first = torch.autograd.grad(actual, (x, gates), cotangent, create_graph=True)
    ref_first = torch.autograd.grad(
        expected, (ref_x, ref_g), cotangent, create_graph=True
    )
    second = torch.autograd.grad(first, (x, gates), directions)
    ref_second = torch.autograd.grad(ref_first, (ref_x, ref_g), directions)
    for value, reference in zip(
        (actual, *first, *second), (expected, *ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("preserve_strides", [False, True], ids=["node", "edge"])
def test_gate_input_layout_and_strided_derivatives(preserve_strides):
    lmax, nodes, channels = 2, 3, 8
    if preserve_strides:
        x = torch.randn(nodes, 9, 2 * channels, device="cuda")[..., ::2]
    else:
        x = torch.randn(1, 9, channels, device="cuda").expand(nodes, -1, -1)
    x.requires_grad_()
    gates = torch.randn(nodes, lmax * channels * 2, device="cuda")[
        :, ::2
    ].requires_grad_()
    if preserve_strides:
        actual = fused_gate(gates, x, lmax=lmax, mmax=lmax)
        assert actual.stride() == x.stride()
    else:
        actual = node_gate(gates, x, lmax=lmax)
        assert actual.is_contiguous()
    expected = torch.cat(
        (
            F.silu(x[:, :1]),
            *(
                x[:, degree * degree : (degree + 1) ** 2]
                * gates.reshape(nodes, lmax, channels)[:, degree - 1 : degree].sigmoid()
                for degree in range(1, lmax + 1)
            ),
        ),
        dim=1,
    )
    cotangent = (
        torch.randn(nodes, channels, 9, device="cuda").transpose(1, 2).requires_grad_()
    )
    directions = (
        torch.randn(nodes, channels, 9, device="cuda").transpose(1, 2).requires_grad_(),
        torch.randn(nodes, lmax * channels * 2, device="cuda")[:, ::2].requires_grad_(),
    )
    targets = (x, gates)
    first = torch.autograd.grad(actual, targets, cotangent, create_graph=True)
    ref_first = torch.autograd.grad(expected, targets, cotangent, create_graph=True)
    second_targets = (*targets, cotangent)
    second = torch.autograd.grad(first, second_targets, directions, create_graph=True)
    ref_second = torch.autograd.grad(ref_first, second_targets, directions)
    for value, reference in zip(
        (actual, *first, *second), (expected, *ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        sum(value.sum() for value in second).backward()


def test_gate_scalar_and_positive_degree_semantics():
    x = torch.tensor(
        [[[2.0, -1.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64
    )
    result = reference_node_gate(torch.zeros(1, 1, 2), x, lmax=1)
    torch.testing.assert_close(result[:, 0], F.silu(x[:, 0]))
    torch.testing.assert_close(result[:, 1:], x[:, 1:] / 2)


def test_node_gate_rejects_cpu_input():
    with pytest.raises(ValueError, match="CUDA FP32"):
        node_gate(torch.randn(2, 3), torch.randn(2, 4, 3), lmax=1)


def test_coefficient_gate_reference_derivatives():
    torch.manual_seed(814)
    x = (0.2 * torch.randn(2, 4, 3, dtype=torch.float64)).requires_grad_()
    gates = (0.2 * torch.randn(2, 1, 3, dtype=torch.float64)).requires_grad_()

    def function(gates, x):
        return reference_node_gate(gates, x, lmax=1)

    assert torch.autograd.gradcheck(function, (gates, x), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (gates, x), fast_mode=True)


@pytest.mark.cuda
@pytest.mark.parametrize("lmax", [0, 2])
def test_coefficient_gate_against_fp64_reference(lmax):
    torch.manual_seed(816)
    x = (0.2 * torch.randn(2, (lmax + 1) ** 2, 3, device="cuda")).requires_grad_()
    gates = (0.2 * torch.randn(2, lmax, 3, device="cuda")).requires_grad_()
    reference = tuple(value.detach().double().requires_grad_() for value in (gates, x))
    actual = node_gate(gates, x, lmax=lmax)
    expected = reference_node_gate(*reference, lmax=lmax)
    assert_directional_derivatives(actual, expected, (gates, x), reference)
