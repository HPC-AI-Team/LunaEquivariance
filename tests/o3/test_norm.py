"""Norm outputs, strided inputs and derivatives."""

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Norm as ReferenceNorm

import LunaEquivariance as lunaeq
from LunaEquivariance import Norm, o3 as luna_o3
from LunaEquivariance.modules.norm import LunaEqNorm
from LunaEquivariance.o3.norm.interface import norm
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_third_derivatives(dtype):
    reference = ReferenceNorm("2x0e+3x1o").cuda().double()
    native = Norm(reference).cuda().to(dtype)
    x = (
        torch.linspace(0.2, 1.4, 33, device="cuda", dtype=dtype)
        .reshape(3, 11)
        .requires_grad_()
    )
    actual = native(x)
    g = torch.full_like(actual, 0.23, requires_grad=True)
    rx = torch.full_like(x, 0.17, requires_grad=True)
    bx, bg = (torch.full_like(x, 0.19), torch.full_like(actual, 0.31))

    def derivatives(output, x, g, rx, bx, bg):
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(second, (x, g, rx), (bx, bg))
        return (first, *second, *third)

    ref_values = fp64((x, g, rx, bx, bg))
    expected = reference(ref_values[0])
    for value, target in zip(
        derivatives(actual, x, g, rx, bx, bg), derivatives(expected, *ref_values)
    ):
        if value is None or target is None:
            assert value is target
        else:
            assert_reference_close(value, target)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("layout", ["columns", "transpose", "expand"])
def test_strided_forward_and_mixed_backward(dtype, squared, layout):
    reference = ReferenceNorm("2x0e + 3x1o", squared=squared).cuda().double()
    native = Norm(reference).cuda().to(dtype)
    if layout == "columns":
        features = torch.linspace(0.2, 1.4, 132, device="cuda", dtype=dtype).reshape(
            2, 3, 22
        )[..., ::2]
    elif layout == "transpose":
        features = (
            torch.linspace(0.2, 1.4, 66, device="cuda", dtype=dtype)
            .reshape(2, 3, 11)
            .transpose(0, 1)
        )
    else:
        features = (
            torch.linspace(0.2, 1.4, 22, device="cuda", dtype=dtype)
            .reshape(2, 1, 11)
            .expand(2, 3, 11)
        )
    x = features.detach().requires_grad_()
    assert not x.is_contiguous()
    actual = native(x)
    assert native.last_execution_backend == "cuda_fused"
    g = (
        torch.linspace(0.13, 0.29, actual.numel(), device="cuda", dtype=dtype)
        .reshape_as(actual)
        .requires_grad_()
    )
    rx = (
        torch.linspace(0.17, 0.23, x.numel(), device="cuda", dtype=dtype)
        .reshape_as(x)
        .requires_grad_()
    )
    bx, bg = (torch.full_like(x, 0.19), torch.full_like(actual, 0.31))

    def derivatives(output, x, g, rx, bx, bg):
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(second, (x, g, rx), (bx, bg), allow_unused=True)
        return (first, *second, *third)

    ref_values = fp64((x, g, rx, bx, bg))
    expected = reference(ref_values[0])
    for value, target in zip(
        (actual, *derivatives(actual, x, g, rx, bx, bg)),
        (expected, *derivatives(expected, *ref_values)),
    ):
        if value is None or target is None:
            assert value is target
        else:
            assert_reference_close(value, target)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("seed", ["both", "features", "grad_out"])
def test_partial_third_derivatives(dtype, squared, seed):
    irreps = "2x0e+3x1o+2x2e"
    native = Norm(irreps, squared=squared, dtype=dtype).cuda()
    reference = ReferenceNorm(irreps, squared=squared).cuda().double()
    torch.manual_seed(371)
    x = (torch.randn(4, 21, device="cuda", dtype=dtype) + 0.3).requires_grad_()
    g = torch.randn(4, 7, device="cuda", dtype=dtype, requires_grad=True)
    rx = torch.randn_like(x, requires_grad=True)
    bx = torch.randn_like(x) if seed != "grad_out" else None
    bg = torch.randn_like(g) if seed != "features" else None
    ref_x, ref_g, ref_rx, ref_bx, ref_bg = fp64((x, g, rx, bx, bg))
    results = []
    for model, x, g, rx, bx, bg in (
        (reference, ref_x, ref_g, ref_rx, ref_bx, ref_bg),
        (native, x, g, rx, bx, bg),
    ):
        output = model(x)
        first = torch.autograd.grad(output, x, g, create_graph=True, retain_graph=True)[
            0
        ]
        second = torch.autograd.grad(
            first, (x, g), rx, create_graph=True, retain_graph=True
        )
        pairs = [
            (value, direction)
            for value, direction in zip(second, (bx, bg))
            if direction is not None
        ]
        third = torch.autograd.grad(
            tuple((v for v, _ in pairs)),
            (x, g, rx),
            tuple((d for _, d in pairs)),
            allow_unused=True,
        )
        results.append((output, first, *second, *third))
    for expected, actual in zip(*results):
        if expected is None or actual is None:
            assert expected is actual
        else:
            assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("squared", [False, True])
def test_zero_nonfinite_empty_and_strided(dtype, squared):
    irreps = "2x0e+3x1o+2x2e"
    native = Norm(irreps, squared=squared, dtype=dtype).cuda()
    reference = ReferenceNorm(irreps, squared=squared).cuda().to(dtype)
    tiny = 1e-30 if dtype == torch.float32 else 1e-200
    rows = torch.tensor(
        [0.0, tiny, float("inf"), float("nan"), 0.75], dtype=dtype, device="cuda"
    )
    x = rows[:, None].expand(-1, 21).detach().requires_grad_()
    expected, actual = (reference(x), native(x))
    tolerance = 3e-05 if dtype == torch.float32 else 3e-12
    for order in range(4):
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
        )
        if order < 3:
            assert actual.requires_grad == expected.requires_grad
            if not expected.requires_grad:
                break
            expected = torch.autograd.grad(
                expected.sum(), x, create_graph=order < 2, retain_graph=True
            )[0]
            actual = torch.autograd.grad(
                actual.sum(), x, create_graph=order < 2, retain_graph=True
            )[0]
    empty = torch.empty(2, 0, 21, dtype=dtype, device="cuda", requires_grad=True)
    torch.testing.assert_close(native(empty), reference(empty))
    torch.testing.assert_close(
        torch.autograd.grad(native(empty).sum(), empty)[0], empty
    )


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
    assert_reference_close(actual, reference.double()(x.double()))
    torch.testing.assert_close(module(x), actual, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_functional_norm_cuda_three_orders(dtype):
    irreps = "2x0e + 3x1o + 2x2e"
    x = torch.randn(
        11, o3.Irreps(irreps).dim, device="cuda", dtype=dtype, requires_grad=True
    )
    module = LunaEqNorm(irreps, dtype=dtype).cuda()
    inputs = (x,)
    function = lambda: norm(x, irreps)
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
    reference = o3.Norm(irreps).cuda().double()
    x = torch.randn(
        11, irreps.dim, device="cuda", dtype=input_dtype, requires_grad=True
    )
    inputs = (x,)
    reference_x = fp64((x,))[0]
    with torch.autocast("cuda", dtype=dtype, enabled=autocast):
        actual = norm(x, irreps)
    expected = reference(reference_x)
    if autocast:
        assert actual.dtype == torch.float32
    assert_reference_close(actual, expected, dtype=dtype)
    seed = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, inputs, seed)
    expected_grad = torch.autograd.grad(expected, reference_x, seed.double())
    for actual, expected in zip(actual_grad, expected_grad):
        assert_reference_close(actual, expected, dtype=dtype)


def test_operator_aliases():
    operator = getattr(lunaeq, "Norm")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "Norm") is operator
    assert getattr(getattr(lunaeq, "o3"), "Norm") is operator


def test_module_reference():
    module = luna_o3.Norm("2x0e + 1x1o", squared=True)
    width = 5
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import Norm

    torch.manual_seed(42)
    module = Norm("2x0e+1x1o").to(device="cuda", dtype=dtype)
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
    module = lunaeq.Norm(irreps).cuda()
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
