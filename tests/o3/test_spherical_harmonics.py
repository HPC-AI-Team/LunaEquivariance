"""Spherical harmonics normalization, equivariance and derivatives."""

import pytest
import torch
from e3nn import o3

import LunaEquivariance as lunaeq
from LunaEquivariance import SphericalHarmonics, o3 as luna_o3
from LunaEquivariance.modules import spherical_harmonics as operator
from LunaEquivariance.o3.spherical_harmonics.interface import spherical_harmonics
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    rotate_features,
)


def test_dtype_conversion_and_cpu_fallback():
    op = SphericalHarmonics([0, 1], True)
    op.double()
    op.half()
    assert op.compute_dtype == torch.float16
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    actual = op(x)
    expected = o3.SphericalHarmonics([0, 1], True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert op.last_execution_backend == "torch_compat"
    seed = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, x, seed)
    (expected_grad,) = torch.autograd.grad(expected, x, seed)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_fallback_tracks_public_normalization():
    constructor = o3.SphericalHarmonics
    op = operator.LunaEqSphericalHarmonics("1e+2e", True, "integral", irreps_in="1e")
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    for normalize, normalization in (
        (True, "integral"),
        (True, "integral"),
        (False, "integral"),
        (False, "norm"),
        (True, "integral"),
    ):
        op.normalize, op.normalization = (normalize, normalization)
        actual = op(x)
        expected = constructor(op.irreps_out, normalize, normalization, op.irreps_in)(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seed = torch.randn_like(actual)
        (actual_grad,) = torch.autograd.grad(actual, x, seed)
        (expected_grad,) = torch.autograd.grad(expected, x, seed)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert not op.state_dict() and (not tuple(op.children()))


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ("transpose", "slice", "broadcast", "rank3"))
def test_strided_native_forward_and_derivative(layout):
    if layout == "transpose":
        x = torch.randn(3, 11, device="cuda").t()
    elif layout == "slice":
        x = torch.randn(11, 6, device="cuda")[:, ::2]
    elif layout == "broadcast":
        x = torch.randn(1, 3, device="cuda").expand(11, 3)
    else:
        x = torch.randn(2, 3, 5, device="cuda").permute(0, 2, 1)
    x.requires_grad_()
    op = operator.LunaEqSphericalHarmonics([3, 1, 3, 0], True, "component")
    actual = op(x)
    reference_x = x.detach().double().requires_grad_()
    expected = (
        o3.SphericalHarmonics([3, 1, 3, 0], True, "component")
        .cuda()
        .double()(reference_x)
    )
    assert_reference_close(actual, expected)
    seed = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, x, seed)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, seed.double())
    assert_reference_close(actual_grad, expected_grad)
    assert op.last_execution_backend == "cuda_fused"


def mixed_backward(output, x, g, rx, bx, bg):
    (first,) = torch.autograd.grad(output, x, g, create_graph=True)
    second = torch.autograd.grad(
        first, (x, g), rx, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second, (x, g, rx), (bx, bg), retain_graph=True, allow_unused=True
    )
    return (first, *second, *third)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", ("component", "norm", "integral"))
def test_native_forward_through_triple_backward(dtype, normalize, normalization):
    torch.manual_seed(932117)
    degrees = [3, 1, 3, 0]
    native = SphericalHarmonics(degrees, normalize, normalization, dtype=dtype)
    reference = o3.SphericalHarmonics(degrees, normalize, normalization).cuda().double()
    x = (torch.randn(5, 3, device="cuda", dtype=dtype) + 0.8).requires_grad_()
    actual = native(x)
    g = torch.randn_like(actual).requires_grad_()
    rx = torch.randn_like(x).requires_grad_()
    bx, bg = (torch.randn_like(x), torch.randn_like(g))
    ref_values = fp64((x, g, rx, bx, bg))
    expected = reference(ref_values[0])
    for value, target in zip(
        (actual, *mixed_backward(actual, x, g, rx, bx, bg)),
        (expected, *mixed_backward(expected, *ref_values)),
    ):
        if value is None or target is None:
            assert value is target
        else:
            assert_reference_close(value, target)
    assert native.last_execution_backend == "cuda_fused"
    assert native.backward_backend == "cuda_fused_recompute"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_triple_backward_with_nonleaf_inputs(dtype):
    torch.manual_seed(932118)
    native = SphericalHarmonics([0, 1, 2, 3], True, "integral", dtype=dtype)
    reference = o3.SphericalHarmonics([0, 1, 2, 3], True, "integral").cuda().double()
    base = (torch.randn(4, 3, device="cuda", dtype=dtype) + 1.0).requires_grad_()
    x = base.sin() + 0.8
    g = torch.randn(4, 16, device="cuda", dtype=dtype).requires_grad_()
    rx = torch.randn_like(x).requires_grad_()
    bx, bg = (torch.randn_like(x), torch.randn_like(g))
    actual = mixed_backward(native(x), x, g, rx, bx, bg)
    ref_values = fp64((x, g, rx, bx, bg))
    expected = mixed_backward(reference(ref_values[0]), *ref_values)
    for value, target in zip(actual, expected):
        assert_reference_close(value, target)


@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("normalization", ("component", "norm", "integral"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_sh_without_module(normalize, normalization, dtype):
    x = torch.randn(7, 3, dtype=dtype, requires_grad=True)
    degrees = [3, 1, 3, 0]
    reference_x = x.detach().double().requires_grad_()
    expected = o3.spherical_harmonics(degrees, reference_x, normalize, normalization)
    actual = spherical_harmonics(degrees, x, normalize, normalization)
    assert_reference_close(actual, expected)
    seed = torch.randn_like(actual)
    assert_reference_close(
        torch.autograd.grad(actual, x, seed)[0],
        torch.autograd.grad(expected, reference_x, seed.double())[0],
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class TestEquivariance:
    def test_spherical_harmonics(self, device, dtype, rotation):
        module = SphericalHarmonics([0, 1, 2, 3], True, "component", dtype=dtype)
        positions = torch.randn(7, 3, device=device, dtype=dtype)
        actual = module(positions @ rotation.to(positions).T)
        expected = rotate_features(module(positions), module.irreps_out, rotation)
        tolerance = 3e-05 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_operator_aliases():
    operator = getattr(lunaeq, "SphericalHarmonics")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "SphericalHarmonics") is operator
    assert getattr(getattr(lunaeq, "o3"), "SphericalHarmonics") is operator


def test_module_reference():
    module = luna_o3.SphericalHarmonics("0e + 1o", normalize=True)
    width = 3
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import SphericalHarmonics

    torch.manual_seed(42)
    module = SphericalHarmonics("0e+1o+2e", normalize=True).to(
        device="cuda", dtype=dtype
    )
    width = 3
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


def compilation_case(lmax):
    module = lunaeq.SphericalHarmonics(list(range(lmax + 1)), True, "component").cuda()
    samples = [
        ((torch.randn(rows, 3, device="cuda", requires_grad=True),), {})
        for rows in (7, 19)
    ]
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("lmax", (2, 3))
def test_compile(lmax):
    module, samples = compilation_case(lmax)
    assert_compiled(module, samples)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(lmax) for lmax in (2, 3)]
    assert_aoti(cases, tmp_path)
