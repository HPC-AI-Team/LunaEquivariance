"""Attention logits and the gradients of their input and parameters."""

import pytest
import torch

from LunaEquivariance.so2 import attention_scores
from tests.so2.reference.attention_score import attention_scores as attention_reference
from tests.so2.checks import compare_operator_derivatives


def reference(inputs, weight, bias, dot, *, eps):
    normalized = torch.nn.functional.layer_norm(inputs, (64,), weight, bias, eps)
    activated = 0.6 * normalized + 0.4 * normalized * torch.tanh(normalized * 0.5)
    return (activated * dot).sum(-1)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["contiguous", "edge_stride", "head_stride"])
@pytest.mark.parametrize("direction_slot", [None, 0, 1, 2, 3])
def test_strided_alpha_projection_forward_backward_double_backward(
    layout, direction_slot
):
    torch.manual_seed(371)
    source = torch.randn(31, 8, 64, device="cuda")
    if layout == "edge_stride":
        wide = torch.randn(31, 2048, device="cuda")
        inputs = wide[:, 13:525].view(31, 8, 64).requires_grad_()
    elif layout == "head_stride":
        inputs = source.transpose(0, 1).contiguous().transpose(0, 1).requires_grad_()
    else:
        inputs = source.requires_grad_()
    weight = torch.randn(64, device="cuda", requires_grad=True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    dot = torch.randn(8, 64, device="cuda", requires_grad=True)
    args = (inputs, weight, bias, dot)
    cotangent = torch.randn(8, 31, device="cuda").T.requires_grad_()
    directions = tuple(torch.randn_like(value) for value in args)
    results = []
    for implementation in (reference, attention_scores):
        output = implementation(*args, eps=1e-5)
        gradients = torch.autograd.grad(output, args, cotangent, create_graph=True)
        active = range(4) if direction_slot is None else (direction_slot,)
        contraction = sum((gradients[k] * directions[k]).sum() for k in active)
        second = torch.autograd.grad(contraction, (*args, cotangent), allow_unused=True)
        results.append((output, gradients, second))
    for wanted, actual in zip(results[0], results[1]):
        if isinstance(wanted, torch.Tensor):
            wanted, actual = (wanted,), (actual,)
        for a, b in zip(wanted, actual):
            if a is None:
                assert b is None or torch.count_nonzero(b) == 0
            else:
                torch.testing.assert_close(b, a, rtol=3e-4, atol=3e-4)


@pytest.mark.cuda
def test_alpha_projection_rejects_noncontiguous_channels():
    inputs = torch.randn(2, 8, 128, device="cuda")[:, :, ::2]
    with pytest.raises(ValueError, match="contiguous channels"):
        attention_scores(
            inputs,
            torch.ones(64, device="cuda"),
            torch.zeros(64, device="cuda"),
            torch.ones(8, 64, device="cuda"),
            eps=1e-5,
        )


def derivatives(output, leaves, cotangent, directions):
    first = torch.autograd.grad(output, leaves, cotangent, create_graph=True)
    second = torch.autograd.grad(
        first, (*leaves, cotangent), directions, allow_unused=True
    )
    return (output, *first, *second)


def compare(actual, expected, rtol=8e-4, atol=5e-5):
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if b is None:
            assert a is None or torch.count_nonzero(a) == 0
        else:
            torch.testing.assert_close(
                a.double(),
                b.double(),
                rtol=rtol,
                atol=atol,
                msg=lambda message: f"slot {index}: {message}",
            )


@pytest.mark.cuda
@pytest.mark.parametrize("heads,channels", [(1, 1), (3, 7), (5, 37), (8, 64), (2, 96)])
@pytest.mark.parametrize("direction_slot", [None, 0, 1, 2, 3])
def test_alpha_dimensions_all_derivatives(heads, channels, direction_slot):
    torch.manual_seed(2026)
    edges = 7
    storage = torch.randn(edges, heads * channels + 5, device="cuda")
    data = (
        storage[:, 2 : 2 + heads * channels].view(edges, heads, channels),
        torch.randn(channels, device="cuda"),
        torch.randn(channels, device="cuda"),
        torch.randn(heads, channels, device="cuda"),
    )
    cotangent_data = torch.randn(edges, heads, device="cuda")
    directions = tuple(
        torch.randn_like(v) * 0.1
        if direction_slot is None or direction_slot == i
        else torch.zeros_like(v)
        for i, v in enumerate(data)
    )
    result = []
    for reference in (False, True):
        dtype = torch.float64 if reference else torch.float32
        leaves = tuple(v.to(dtype).detach().requires_grad_() for v in data)
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        if reference:
            inputs, weight, bias, dot = leaves
            affine = torch.nn.functional.layer_norm(
                inputs, (channels,), weight, bias, 1e-5
            )
            activation = 0.6 * affine + 0.4 * affine * torch.tanh(0.5 * affine)
            output = (activation * dot).sum(-1)
        else:
            output = attention_scores(*leaves, eps=1e-5)
        result.append(
            derivatives(
                output, leaves, cotangent, tuple(v.to(dtype) for v in directions)
            )
        )
    compare(*result, rtol=1e-3, atol=3e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("channels", [1, 7, 32])
def test_attention_scores_strided_input_double_backward(channels):
    torch.manual_seed(731)
    x = torch.randn(8, 6, channels, device="cuda")[:, ::2].requires_grad_()
    weight = torch.randn(channels, device="cuda", requires_grad=True)
    bias = torch.randn(channels, device="cuda", requires_grad=True)
    projection = torch.randn(3, channels, device="cuda", requires_grad=True)
    compare_operator_derivatives(attention_scores, attention_reference, (x, weight, bias, projection))


@pytest.mark.cuda
def test_attention_force_double_backward_tiled():
    torch.manual_seed(732)
    inputs = (
        torch.randn(4096, 8, 16, device="cuda") * 0.3,
        torch.randn(16, device="cuda") * 0.02 + 1.0,
        torch.randn(16, device="cuda") * 0.02,
        torch.randn(8, 16, device="cuda") * 0.25,
    )
    seed = torch.randn(4096, 8, device="cuda") * 0.1
    direction = torch.randn_like(inputs[0]) * 0.1

    def force_hvp(operation, dtype):
        leaves = tuple(value.to(dtype).requires_grad_() for value in inputs)
        cotangent = seed.to(dtype).requires_grad_()
        logits = operation(*leaves)
        first, = torch.autograd.grad(logits, leaves[0], cotangent, create_graph=True)
        return torch.autograd.grad(first, (*leaves, cotangent), direction.to(dtype))

    compare(
        force_hvp(attention_scores, torch.float32),
        force_hvp(attention_reference, torch.float64),
        atol=3e-5,
        rtol=3e-4,
    )
