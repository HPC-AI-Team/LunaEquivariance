"""NormActivation shared-bias gradients, saved graphs and input pruning."""

from copy import deepcopy
import pytest
import torch
from e3nn.nn import NormActivation as ReferenceNormActivation
from e3nn.o3 import Irreps
from LunaEquivariance.nn import NormActivation

pytestmark = [
    pytest.mark.cuda,
]
IRREPS = "2x0e+1x1o"


def modules(function, dtype):
    reference = (
        ReferenceNormActivation(Irreps(IRREPS), function, epsilon=0.07, bias=True)
        .cuda()
        .to(dtype)
    )
    with torch.no_grad():
        reference.biases.copy_(
            torch.tensor([0.11, -0.08, 0.16], device="cuda", dtype=dtype)
        )
    return (reference, NormActivation(deepcopy(reference)).cuda().to(dtype))


def raw_input(dtype):
    raw = torch.linspace(-1.2, 1.4, 60, device="cuda", dtype=dtype).reshape(2, 3, 10)[
        ..., ::2
    ]
    raw[0, 0] = 0
    return raw


def assert_close(actual, expected, dtype):
    if actual is None or expected is None:
        assert actual is expected
        return
    tolerance = 3e-05 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True)])
def test_public_bias_edges_keep_higher_derivatives(dtype, needs):
    reference, native = modules(torch.tanh, dtype)
    xs = [raw_input(dtype).detach().clone().requires_grad_(needs[0]) for _ in range(2)]
    targets = []
    values = []
    for model, x in zip((reference, native), xs):
        model.biases.requires_grad_(needs[1])
        targets.append(
            tuple((value for value in (x, model.biases) if value.requires_grad))
        )
        values.append(model(x))
    assert native.last_execution_backend == "cuda_fused"
    for order in range(1, 4):
        derivatives = [
            torch.autograd.grad(value.sum(), target, create_graph=order < 3)
            for value, target in zip(values, targets)
        ]
        for expected, actual in zip(*derivatives):
            tolerance = 3e-5 if dtype == torch.float32 else 3e-12
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        values = [
            sum((gradient.sum() for gradient in result)) for result in derivatives
        ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonleaf_bias_and_shared_ancestors_keep_third_derivatives(dtype):
    reference, native = modules(torch.sigmoid, dtype)
    xs = [raw_input(dtype).detach().clone().requires_grad_() for _ in range(2)]
    biases = [
        model.biases.detach().clone().requires_grad_() for model in (reference, native)
    ]
    targets = list(zip(xs, biases))
    values = []
    for model, x, bias in zip((reference, native), xs, biases):
        output = torch.func.functional_call(model, {"biases": bias * 1.3 + 0.02}, (x,))
        values.append(0.17 * output.square().sum())
    for order in range(1, 4):
        derivatives = [
            torch.autograd.grad(value, target, create_graph=order < 3)
            for value, target in zip(values, targets)
        ]
        for expected, actual in zip(*derivatives):
            assert_close(actual, expected, dtype)
        values = [
            0.23 * sum((value.sum() for value in result)) for result in derivatives
        ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_empty_batch_keeps_input_and_bias_gradients(dtype, bias):
    reference = (
        ReferenceNormActivation(Irreps(IRREPS), torch.tanh, epsilon=0.07, bias=bias)
        .cuda()
        .to(dtype)
    )
    native = NormActivation(deepcopy(reference)).cuda().to(dtype)
    xs = [
        torch.empty(2, 0, 5, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    targets = [(x, *model.parameters()) for x, model in zip(xs, (reference, native))]
    outputs = [model(x) for model, x in zip((reference, native), xs)]
    assert outputs[1].shape == outputs[0].shape
    derivatives = [
        torch.autograd.grad(output.sum(), target)
        for output, target in zip(outputs, targets)
    ]
    for expected, actual in zip(*derivatives):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
