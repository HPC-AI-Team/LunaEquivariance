"""Supported scalar activations use fixed parameters and reject custom programs."""

from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from e3nn import nn as reference_nn, o3

from LunaEquivariance import nn


ACTIVATIONS = (
    pytest.param(torch.nn.Identity(), id="identity"),
    pytest.param(torch.abs, id="abs"),
    pytest.param(torch.relu, id="relu"),
    pytest.param(F.silu, id="silu"),
    pytest.param(torch.tanh, id="tanh"),
    pytest.param(torch.sigmoid, id="sigmoid"),
    pytest.param(F.softplus, id="softplus"),
    pytest.param(F.gelu, id="gelu"),
    pytest.param(torch.nn.Softplus(beta=1.3, threshold=7), id="softplus-module"),
    pytest.param(partial(F.softplus, beta=1.7, threshold=6), id="softplus-partial"),
    pytest.param(torch.nn.GELU(approximate="tanh"), id="gelu-module"),
    pytest.param(partial(F.gelu, approximate="tanh"), id="gelu-partial"),
)


def modules(name, function):
    if name == "Activation":
        reference = reference_nn.Activation("3x0e+1x1o", [deepcopy(function), None])
    elif name == "Gate":
        reference = reference_nn.Gate(
            "3x0e", [deepcopy(function)], "2x0e", [torch.sigmoid], "2x1o"
        )
    else:
        reference = reference_nn.NormActivation(
            o3.Irreps("3x0e+1x1o"), deepcopy(function), epsilon=0.07, bias=True
        )
        with torch.no_grad():
            reference.biases.copy_(torch.tensor([0.11, -0.08, 0.16, -0.03]))
    return reference, getattr(nn, name)(deepcopy(reference))


@pytest.mark.parametrize("name", ("Activation", "Gate", "NormActivation"))
@pytest.mark.parametrize("function", ACTIVATIONS[-4:])
def test_parameterized_builtin_cpu_matches_e3nn(name, function):
    reference, native = modules(name, function)
    x = torch.linspace(-0.8, 1.2, 2 * reference.irreps_in.dim).reshape(2, -1)
    torch.testing.assert_close(native(x), reference(x))


@pytest.mark.cuda
@pytest.mark.parametrize("name", ("Activation", "Gate", "NormActivation"))
@pytest.mark.parametrize("function", ACTIVATIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
def test_builtin_cuda_forward_through_triple_backward_matches_e3nn(
    name, function, dtype
):
    reference, native = modules(name, function)
    reference_dtype = torch.float64 if name == "NormActivation" else dtype
    reference, native = reference.cuda().to(reference_dtype), native.cuda().to(dtype)
    models, precisions = [reference, native], [reference_dtype, dtype]
    if name == "NormActivation" and dtype == torch.float32:
        # Norm derivatives can cancel. Measure the FP32 reference's own
        # rounding error against FP64 instead of treating that error as zero.
        models.append(deepcopy(reference).float())
        precisions.append(dtype)
    width = reference.irreps_in.dim
    raw = torch.linspace(-0.8, 1.2, 3 * width, device="cuda", dtype=dtype).reshape(
        3, width
    )
    inputs = [
        raw.detach().clone().to(precision).requires_grad_() for precision in precisions
    ]
    targets = [(x, *model.parameters()) for model, x in zip(models, inputs)]
    values = [(model(x),) for model, x in zip(models, inputs)]
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    assert native.last_execution_backend == "cuda_fused"
    for order in range(4):
        if order:
            derivatives = []
            for current, target in zip(values, targets):
                live = tuple(
                    value
                    for value in current
                    if value is not None and value.requires_grad
                )
                derivatives.append(
                    torch.autograd.grad(
                        tuple(value.sum() for value in live),
                        target,
                        create_graph=order < 3,
                        allow_unused=True,
                    )
                    if live
                    else (None,) * len(target)
                )
            values = derivatives
        for expected, actual, *same_precision_reference in zip(*values):
            if expected is None or actual is None:
                assert actual is expected
            elif same_precision_reference:
                rounded = same_precision_reference[0]
                assert rounded is not None
                assert actual.shape == rounded.shape == expected.shape
                assert actual.dtype == rounded.dtype == dtype
                reference_error = (rounded.double() - expected).abs()
                error = (actual.double() - expected).abs()
                budget = reference_error + tolerance + tolerance * expected.abs()
                assert torch.all(error <= budget), (
                    f"order {order}: error exceeds the reference rounding error and "
                    f"the original tolerance by {(error - budget).max().item()}"
                )
            else:
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
