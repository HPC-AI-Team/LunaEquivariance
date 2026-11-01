"""Tensor-only SH backward schemas and compiled higher derivatives."""

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance import SphericalHarmonics
from LunaEquivariance.o3.spherical_harmonics.interface import register_ops


@pytest.mark.parametrize("degrees", ((0,), (0, 0), (1,), (0, 1, 2)))
def test_registered_backward_schema(degrees):
    name = register_ops(degrees, True, "component", 32, torch.float32)
    backward = name.removesuffix("_forward") + "_backward"
    if any(degrees):
        schema = getattr(torch.ops.lunaeq, backward).default._schema
        assert len(schema.returns) == 1
        assert str(schema.returns[0].type) == "Tensor"
    else:
        assert not hasattr(torch.ops.lunaeq, backward)


def derivatives(module, x, g, rx, bx, bg):
    output = module(x)
    (first,) = torch.autograd.grad(output, x, g, create_graph=True)
    second = torch.autograd.grad(first, (x, g), rx, create_graph=True)
    third = torch.autograd.grad(second, (x, g, rx), (bx, bg), allow_unused=True)
    return output, first, *second, *third


def inputs(dtype, rows):
    x = (torch.randn(rows, 3, device="cuda", dtype=dtype) + 0.8).requires_grad_()
    g = torch.randn(rows, 9, device="cuda", dtype=dtype).requires_grad_()
    rx = torch.randn_like(x).requires_grad_()
    return x, g, rx, torch.randn_like(x), torch.randn_like(g)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("normalize", (False, True))
def test_aoti_forward_through_triple_backward(tmp_path, dtype, normalize):
    torch.manual_seed(932119)
    native = SphericalHarmonics([0, 1, 2], normalize, "component", dtype=dtype).cuda()
    reference = o3.SphericalHarmonics([0, 1, 2], normalize, "component").cuda()
    args = inputs(dtype, 5)

    def function(*values):
        return derivatives(native, *values)

    traced = make_fx(function, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*args)
    assert any(
        "spherical_harmonics" in str(node.target) and str(node.target).endswith("_backward.default")
        for node in traced.graph.nodes
    )
    rows = torch.export.Dim("rows", min=2, max=32)
    exported = torch.export.export(traced, args, dynamic_shapes=tuple({0: rows} for _ in args))
    path = str(tmp_path / "spherical_harmonics.pt2")
    torch._inductor.aoti_compile_and_package(exported, package_path=path)
    compiled = torch._inductor.aoti_load_package(path)
    tolerance = 8e-5 if dtype == torch.float32 else 3e-11
    for count in (5, 7):
        values = inputs(dtype, count)
        expected = derivatives(reference, *values)
        with torch.no_grad():
            actual = compiled(*values)
        for value, target in zip(actual, expected):
            if target is None:
                assert value is None
            else:
                torch.testing.assert_close(value, target, rtol=tolerance, atol=tolerance)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("normalize", (False, True))
def test_constant_harmonics_preserve_none_gradient(dtype, normalize):
    name = register_ops((0, 0), normalize, "component", 32, dtype)
    operation = getattr(torch.ops.lunaeq, name)
    x = torch.randn(5, 3, device="cuda", dtype=dtype, requires_grad=True)
    output = operation(x)
    torch.testing.assert_close(output, torch.ones(5, 2, device="cuda", dtype=dtype))
    (gradient,) = torch.autograd.grad(output.sum(), x, allow_unused=True)
    assert gradient is None
    (gradient,) = torch.autograd.grad(operation(x).sum() + x.square().sum(), x)
    torch.testing.assert_close(gradient, 2 * x)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_empty_registered_backward_returns_tensor(dtype):
    name = register_ops((0, 1, 2), True, "component", 32, dtype)
    x = torch.empty(0, 3, device="cuda", dtype=dtype, requires_grad=True)
    output = getattr(torch.ops.lunaeq, name)(x)
    (gradient,) = torch.autograd.grad(output.sum(), x)
    assert gradient.shape == x.shape
    assert gradient.dtype == dtype
