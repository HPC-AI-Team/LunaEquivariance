"""Activation functional and module outputs and derivatives."""

from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from e3nn import nn as reference_nn

import LunaEquivariance as lunaeq
from LunaEquivariance import nn as luna_nn
from LunaEquivariance.modules.activation import LunaEqActivation
from LunaEquivariance.nn.activation.interface import activation
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import assert_aoti, assert_compiled, assert_reference_close


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


def builtin_modules(function):
    reference = reference_nn.Activation("3x0e+1x1o", [deepcopy(function), None])
    return (reference, getattr(luna_nn, "Activation")(deepcopy(reference)))


@pytest.mark.parametrize("function", ACTIVATIONS[-4:])
def test_parameterized_builtin_cpu_matches_e3nn(function):
    reference, native = builtin_modules(function)
    x = torch.linspace(-0.8, 1.2, 2 * reference.irreps_in.dim).reshape(2, -1)
    assert_reference_close(native(x), reference.double()(x.double()))


@pytest.mark.cuda
@pytest.mark.parametrize("function", ACTIVATIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("fp32", "fp64"))
def test_builtin_cuda_forward_through_triple_backward_matches_e3nn(function, dtype):
    reference, native = builtin_modules(function)
    reference_dtype = torch.float64
    reference, native = (reference.cuda().to(reference_dtype), native.cuda().to(dtype))
    models, precisions = ([reference, native], [reference_dtype, dtype])
    width = reference.irreps_in.dim
    raw = torch.linspace(-0.8, 1.2, 3 * width, device="cuda", dtype=dtype).reshape(
        3, width
    )
    inputs = [
        raw.detach().clone().to(precision).requires_grad_() for precision in precisions
    ]
    targets = [(x, *model.parameters()) for model, x in zip(models, inputs)]
    values = [(model(x),) for model, x in zip(models, inputs)]
    assert native.last_execution_backend == "cuda_fused"
    for order in range(4):
        if order:
            derivatives = []
            for current, target in zip(values, targets):
                live = tuple(
                    (
                        value
                        for value in current
                        if value is not None and value.requires_grad
                    )
                )
                derivatives.append(
                    torch.autograd.grad(
                        tuple((value.sum() for value in live)),
                        target,
                        create_graph=order < 3,
                        allow_unused=True,
                    )
                    if live
                    else (None,) * len(target)
                )
            values = derivatives
        for expected, actual in zip(*values):
            if expected is None or actual is None:
                assert actual is expected
            else:
                assert_reference_close(actual, expected)


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
    expected = reference_nn.Activation(irreps, acts).double()
    x = torch.randn(4, 9, dtype=dtype)
    actual = activation(x, irreps, acts)
    assert_reference_close(actual, expected(x.double()))
    assert_reference_close(
        activation(x.t(), irreps, acts, dim=0), expected(x.double().t(), dim=0)
    )


def test_invalid_activation():
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "1x1e", [torch.tanh])
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "3x0o", [torch.sigmoid])
    with pytest.raises(ValueError):
        activation(torch.randn(2, 3), "3x0e", [])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_three_orders(dtype):
    args = ("32x0e+8x1o", [torch.tanh, None])
    module = LunaEqActivation(*args, dtype=dtype).cuda()
    official = reference_nn.Activation(*args).cuda().double()
    function = activation
    x = torch.randn(19, module.in_dim, device="cuda", dtype=dtype, requires_grad=True)
    reference_x = x.detach().double().requires_grad_()
    module_x = x.detach().clone().requires_grad_()
    actual, expected, wrapped = (
        function(x, *args),
        official(reference_x),
        module(module_x),
    )
    for order in range(4):
        torch.testing.assert_close(actual, wrapped, rtol=0, atol=0)
        assert_reference_close(actual, expected)
        if order < 3:
            seed = torch.randn_like(actual)
            (actual,) = torch.autograd.grad(actual, x, seed, create_graph=True)
            (wrapped,) = torch.autograd.grad(wrapped, module_x, seed, create_graph=True)
            (expected,) = torch.autograd.grad(
                expected, reference_x, seed.double(), create_graph=True
            )
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            actual = function(x, *args)
            expected = official(x.double())
            assert_reference_close(actual, expected)
            assert actual.requires_grad == expected.requires_grad
            assert actual.is_inference() == expected.is_inference()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_torch_path(dtype):
    args = ("4x0e", [torch.tanh])
    module = LunaEqActivation(*args)
    function = activation
    x = torch.randn(module.in_dim, 5, dtype=dtype).t().requires_grad_()
    actual = function(x, *args)
    torch.testing.assert_close(actual, module(x), atol=0, rtol=0)
    actual.sum().backward()
    assert x.grad is not None


def test_operator_aliases():
    operator = getattr(lunaeq, "Activation")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "Activation") is operator
    assert getattr(getattr(lunaeq, "nn"), "Activation") is operator


def test_module_reference():
    module = luna_nn.Activation("2x0o + 1x1o", [torch.abs, None])
    width = 5
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import Activation

    torch.manual_seed(42)
    module = Activation("2x0e+1x1o", [torch.tanh, None]).to(device="cuda", dtype=dtype)
    width = 5
    x = torch.randn(5, width, device="cuda", dtype=dtype).requires_grad_()
    assert_cuda_report(module, (x,))


def assert_cuda_report(module, args, **kwargs):
    tolerance = 1e-09 if args[0].dtype == torch.float64 else 0.0003
    report = check_compatibility(
        module,
        args,
        kwargs=kwargs,
        derivative_order=3,
        loss_fn=lambda output: sum(
            (
                x.sin().mean()
                for x in (output if isinstance(output, tuple) else (output,))
            )
        ),
        warmup=1,
        repeats=2,
        atol=tolerance,
        rtol=tolerance,
        verbose=False,
    )
    assert report["passed"], report
    assert report["operator"] == type(module).__name__
    assert report["errors"]["gradient_3"]["passed"]
    assert all((item["speedup"] > 0 for item in report["timings"].values()))


def compilation_case(channels):
    irreps = f"{channels}x0e + {channels}x1o"
    module = lunaeq.Activation(irreps, [torch.tanh, None]).cuda()
    samples = [((torch.randn(rows, 4 * channels, device="cuda", requires_grad=True),), {})
               for rows in (7, 19)]
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (4, 8))
def test_compile(channels):
    module, samples = compilation_case(channels)
    assert_compiled(module, samples)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (4, 8)]
    assert_aoti(cases, tmp_path)
