"""Frozen weights skip derivative outputs while preserving mixed input derivatives."""

import pytest
import torch
from LunaEquivariance import TensorProduct, TensorProductConv
from tests.lunatp.test_uvu_shapes import assert_close


def frozen_chain(out, x, y, g, directions, second_directions):
    first = torch.autograd.grad(out, (x, y), g, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(
        first, (x, y, g), directions, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second, (x, y, g, *directions), second_directions, retain_graph=True
    )
    return (out,), first, second, third


def verify_frozen_problem(tp, x, y, w, mode, shared, dtype):
    w.requires_grad_(False)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    expected = tp(x, y, w)
    actual = op(x, y, None if shared else w)
    g = torch.randn_like(expected, requires_grad=True)
    directions = tuple(torch.randn_like(v, requires_grad=True) for v in (x, y))
    second_directions = tuple(torch.randn_like(v) for v in (x, y, g))
    reference = frozen_chain(expected, x, y, g, directions, second_directions)
    result = frozen_chain(actual, x, y, g, directions, second_directions)
    for results, references in zip(result, reference):
        for value, wanted in zip(results, references):
            assert_close(value, wanted, dtype)
    assert w.grad is None
    first = torch.autograd.grad(actual, (x, y), g, retain_graph=True)
    for value, wanted in zip(first, reference[1]):
        assert_close(value, wanted, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_weights_through_third_order(make_problem, mode, shared, dtype):
    tp, x, y, w, *_ = make_problem(mode, shared, dtype=dtype)
    verify_frozen_problem(tp, x, y, w, mode, shared, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_weights_empty_inputs(make_problem, mode, shared, dtype):
    tp, x, y, w, *_ = make_problem(mode, shared, dtype=dtype)
    w.requires_grad_(False)
    x, y = x[:0], y[:0]
    w = w if shared else w[:0]
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    out = op(x, y, None if shared else w)
    g = torch.randn_like(out, requires_grad=True)
    directions = tuple(torch.randn_like(v, requires_grad=True) for v in (x, y))
    second_directions = tuple(torch.randn_like(v) for v in (x, y, g))
    for order in frozen_chain(out, x, y, g, directions, second_directions):
        for value in order:
            assert value.numel() == 0


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_external_weights_from_network_keep_parameter_gradients(
    make_problem, mode, dtype
):
    tp, x, y, w, *_ = make_problem(mode, dtype=dtype)
    parameter = w.detach().clone().requires_grad_()
    weight = torch.tanh(parameter)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    expected = tp(x, y, weight)
    actual = op(x, y, weight)
    inputs = (x, y, parameter)
    g = torch.randn_like(expected, requires_grad=True)
    directions = tuple(torch.randn_like(v, requires_grad=True) for v in inputs)
    second_directions = tuple(torch.randn_like(v) for v in (*inputs, g))

    def chain(out):
        first = torch.autograd.grad(
            out, inputs, g, create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, (*inputs, g), directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, g, *directions), second_directions, retain_graph=True
        )
        return first, second, third

    for values, wanted_values in zip(chain(actual), chain(expected)):
        for value, wanted in zip(values, wanted_values):
            assert_close(value, wanted, dtype)
