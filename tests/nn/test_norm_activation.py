"""NormActivation functional and module outputs and derivatives."""

from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
from e3nn import nn as reference_nn, o3
from e3nn.nn import NormActivation as ReferenceNormActivation
from e3nn.o3 import Irreps

import LunaEquivariance as lunaeq
from LunaEquivariance import NormActivation, nn as luna_nn
from LunaEquivariance.modules.norm_activation import LunaEqNormActivation
from LunaEquivariance.nn.norm_activation.interface import norm_activation
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
    reference = reference_nn.NormActivation(
        o3.Irreps("3x0e+1x1o"), deepcopy(function), epsilon=0.07, bias=True
    )
    with torch.no_grad():
        reference.biases.copy_(torch.tensor([0.11, -0.08, 0.16, -0.03]))
    return (reference, getattr(luna_nn, "NormActivation")(deepcopy(reference)))


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


IRREPS = "2x0e+1x1o"


def modules(function, dtype):
    reference = (
        ReferenceNormActivation(Irreps(IRREPS), function, epsilon=0.07, bias=True)
        .cuda()
        .double()
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
    assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True)])
def test_public_bias_edges_keep_higher_derivatives(dtype, needs):
    reference, native = modules(torch.tanh, dtype)
    raw = raw_input(dtype)
    xs = [
        raw.detach().to(precision).requires_grad_(needs[0])
        for precision in (torch.float64, dtype)
    ]
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
            assert_reference_close(actual, expected)
        values = [
            sum((gradient.sum() for gradient in result)) for result in derivatives
        ]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonleaf_bias_and_shared_ancestors_keep_third_derivatives(dtype):
    reference, native = modules(torch.sigmoid, dtype)
    raw = raw_input(dtype)
    xs = [
        raw.detach().to(precision).requires_grad_()
        for precision in (torch.float64, dtype)
    ]
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


@pytest.mark.cuda
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


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "nonlinearity", [torch.tanh, torch.sigmoid, torch.nn.functional.silu]
)
def test_norm_activation_function_cpu(dtype, bias, nonlinearity):
    irreps = "2x0e + 3x1o + 2x2e"
    reference = reference_nn.NormActivation(
        o3.Irreps(irreps), nonlinearity, bias=bias
    ).to(dtype)
    module = LunaEqNormActivation(reference).to(dtype)
    x = torch.randn(5, o3.Irreps(irreps).dim, dtype=dtype, requires_grad=True)
    reference_x = x.detach().double().requires_grad_()
    parameter = reference.biases if bias else None
    reference64 = deepcopy(reference).double()
    actual = norm_activation(x, irreps, nonlinearity, bias=parameter)
    assert_reference_close(actual, reference64(reference_x))
    torch.testing.assert_close(module(x), actual, atol=0, rtol=0)
    inputs = (x, parameter) if bias else (x,)
    actual_grad = torch.autograd.grad(actual.sum(), inputs)

    # Near zero, the normalized activation can lose gradient precision in FP16/BF16.
    low_precision = dtype in (torch.float16, torch.bfloat16)
    if low_precision:
        reference_x = x.detach().requires_grad_()
    else:
        reference = reference64
    reference_inputs = (reference_x, reference.biases) if bias else (reference_x,)
    expected_grad = torch.autograd.grad(reference(reference_x).sum(), reference_inputs)
    for actual, expected in zip(actual_grad, expected_grad):
        if low_precision:
            torch.testing.assert_close(actual, expected)
        else:
            assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_functional_norm_cuda_three_orders(dtype):
    irreps = "2x0e + 3x1o + 2x2e"
    x = torch.randn(
        11, o3.Irreps(irreps).dim, device="cuda", dtype=dtype, requires_grad=True
    )
    module = LunaEqNormActivation(irreps, torch.tanh, bias=True, dtype=dtype).cuda()
    inputs = (x, module.biases)
    function = lambda: norm_activation(x, irreps, torch.tanh, bias=module.biases)
    actual = function()
    expected = module(x)
    tolerance = 3e-05 if dtype == torch.float32 else 3e-12
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
def test_functional_norm_cuda_half_and_autocast(dtype, autocast):
    irreps = o3.Irreps("2x0e + 3x1o + 2x2e")
    input_dtype = torch.float32 if autocast else dtype
    reference = (
        reference_nn.NormActivation(irreps, torch.tanh, bias=True)
        .cuda()
        .to(input_dtype)
    )
    x = torch.randn(
        11, irreps.dim, device="cuda", dtype=input_dtype, requires_grad=True
    )
    bias = reference.biases
    inputs = (x, bias)
    reference_x = x.detach().double().requires_grad_()
    reference = deepcopy(reference).double()
    reference_inputs = (reference_x, reference.biases)
    with torch.autocast("cuda", dtype=dtype, enabled=autocast):
        actual = norm_activation(x, irreps, torch.tanh, bias=bias)
    expected = reference(reference_x)
    assert_reference_close(actual, expected, dtype=dtype)
    seed = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, inputs, seed)
    expected_grad = torch.autograd.grad(expected, reference_inputs, seed.double())
    for actual, expected in zip(actual_grad, expected_grad):
        assert_reference_close(actual, expected, dtype=dtype)


def test_norm_activation_string_irreps_with_bias():
    report = check_compatibility(
        luna_nn.NormActivation("2x0e+1x1o", torch.sigmoid, bias=True),
        (torch.randn(3, 5, requires_grad=True),),
        warmup=0,
        repeats=1,
        verbose=False,
    )
    assert report["passed"], report


def test_operator_aliases():
    operator = getattr(lunaeq, "NormActivation")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "NormActivation") is operator
    assert getattr(getattr(lunaeq, "nn"), "NormActivation") is operator


def test_module_reference():
    module = luna_nn.NormActivation("2x0e + 1x1o", torch.sigmoid, bias=True)
    with torch.no_grad():
        module.biases.fill_(0.2)
    width = 5
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import NormActivation

    torch.manual_seed(42)
    module = NormActivation("2x0e+1x1o", torch.sigmoid, bias=True).to(
        device="cuda", dtype=dtype
    )
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
    module = lunaeq.NormActivation(irreps, torch.tanh, epsilon=0.03, bias=True).cuda()
    samples = [((torch.randn(rows, 4 * channels, device="cuda", requires_grad=True),), {})
               for rows in (7, 19)]
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (4, 8))
def test_compile(channels):
    module, samples = compilation_case(channels)
    assert_compiled(module, samples)


@pytest.mark.cuda
@pytest.mark.parametrize("order", (1, 2))
def test_make_fx_training_keeps_bias_gradients(order):
    torch.manual_seed(1827)
    native, samples = compilation_case(4)
    reference = ReferenceNormActivation(
        native.irreps_in, torch.tanh, epsilon=0.03, bias=True
    ).cuda().double()
    x = samples[0][0][0]
    bias = native.biases
    strain = x.new_tensor(0.02, requires_grad=True)
    readout = x.new_tensor(0.3, requires_grad=True)
    inputs = (x, bias, strain, readout)
    reference_inputs = tuple(value.detach().double().requires_grad_() for value in inputs)

    def predict(module, positions, bias, strain, readout):
        features = positions * (1 + strain)
        output = torch.func.functional_call(module, {"biases": bias}, (features,))
        energy = (output.sin() * readout).sum()
        first, stress = torch.autograd.grad(
            energy, (positions, strain), create_graph=True
        )
        if order == 1:
            return energy, -first, stress
        direction = torch.linspace(
            0.2, 0.8, positions.numel(), device=positions.device, dtype=positions.dtype
        ).reshape_as(positions)
        second = torch.autograd.grad(
            (first * direction).sum(), positions, create_graph=True
        )[0]
        return energy, -first, stress, second

    expected = predict(reference, *reference_inputs)
    expected_grad = torch.autograd.grad(
        sum(value.square().mean() for value in expected), reference_inputs
    )
    eager = predict(native, *inputs)
    eager_grad = torch.autograd.grad(
        sum(value.square().mean() for value in eager), inputs
    )
    graph = make_fx(
        lambda *values: predict(native, *values),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*inputs)
    torch._dynamo.reset()
    compiled = torch.compile(
        graph, fullgraph=True, options={"triton.cudagraphs": False}
    )
    actual = compiled(*inputs)
    actual_grad = torch.autograd.grad(
        sum(value.square().mean() for value in actual), inputs
    )
    for values in (eager, actual):
        for value, wanted in zip(values, expected):
            assert_reference_close(value, wanted)
    for values in (eager_grad, actual_grad):
        for value, wanted in zip(values, expected_grad):
            assert_reference_close(value, wanted)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (4, 8)]
    assert_aoti(cases, tmp_path)
