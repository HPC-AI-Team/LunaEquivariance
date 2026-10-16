"""Equivariant normalization and gradients against an independent PyTorch expression."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance import EquivariantNorm
from tests.testing_utils import rotate_features


def reference_norm(x, weight, bias, *, norm, grouping, weighting, layout, eps):
    irreps = o3.Irreps("3x0e+2x1o+2x2e")
    blocks = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        block = x[:, section]
        block = (
            block.reshape(-1, mul, ir.dim)
            if layout == "mul_ir"
            else block.reshape(-1, ir.dim, mul).transpose(1, 2)
        )
        if norm == "layer" and ir.is_scalar():
            block = block - block.mean(1, keepdim=True)
        blocks.append(block)
    groups = (
        [[0], [1], [2]]
        if grouping == "per_irrep"
        else [[0], [1, 2]]
        if grouping == "separable"
        else [[0, 1, 2]]
    )
    output = [None] * len(blocks)
    channel_offsets = [0, 3, 5]
    for group in groups:
        moments = []
        for index in group:
            mul, ir = irreps[index]
            moment = blocks[index].square().sum((1, 2), keepdim=True)
            divisor = {
                "component": mul * ir.dim,
                "norm": mul,
                "inverse_dim_squared": ir.dim**2,
            }[weighting]
            moments.append(moment / divisor)
        scale = (sum(moments) / len(group) + eps).rsqrt()
        for index in group:
            mul, ir = irreps[index]
            start = channel_offsets[index]
            value = blocks[index] * scale
            if weight is not None:
                value = value * weight[start : start + mul][None, :, None]
            if bias is not None and ir.is_scalar():
                value = value + bias[None, :, None]
            if layout == "ir_mul":
                value = value.transpose(1, 2)
            output[index] = value.reshape(x.shape[0], -1)
    return torch.cat(output, -1)


class TestEquivariantNorm:
    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("norm", ["layer", "rms"])
    @pytest.mark.parametrize("grouping", ["merged", "separable", "per_irrep"])
    @pytest.mark.parametrize("weighting", ["component", "norm", "inverse_dim_squared"])
    @pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
    def test_forward_and_three_derivatives(
        self, dtype, norm, grouping, weighting, layout
    ):
        torch.manual_seed(732)
        options = dict(
            norm=norm, grouping=grouping, weighting=weighting, layout=layout, eps=1e-4
        )
        module = EquivariantNorm(
            "3x0e+2x1o+2x2e", bias=True, device="cuda", dtype=dtype, **options
        )
        with torch.no_grad():
            module.weight.uniform_(0.8, 1.2)
            module.bias.uniform_(-0.1, 0.1)
        x = torch.randn(4, 38, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
        actual = (module(x),)
        expected = (reference_norm(x, module.weight, module.bias, **options),)
        targets = (x, module.weight, module.bias)
        tolerance = 3e-4 if dtype == torch.float32 else 3e-10
        for order in range(4):
            for value, reference in zip(actual, expected):
                if value is None or reference is None:
                    assert value is reference
                else:
                    torch.testing.assert_close(
                        value, reference, rtol=tolerance, atol=tolerance
                    )
            if order == 3:
                break
            pairs = [
                (a, b)
                for a, b in zip(actual, expected)
                if a is not None and a.requires_grad
            ]
            seeds = tuple(torch.randn_like(a, requires_grad=True) for a, _ in pairs)
            actual = torch.autograd.grad(
                tuple(a for a, _ in pairs),
                targets,
                seeds,
                create_graph=order < 2,
                allow_unused=True,
            )
            expected = torch.autograd.grad(
                tuple(b for _, b in pairs),
                targets,
                seeds,
                create_graph=order < 2,
                allow_unused=True,
            )
            targets = (*targets, *seeds)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("norm", ["layer", "rms"])
    @pytest.mark.parametrize("grouping", ["merged", "separable", "per_irrep"])
    def test_equivariance(self, rotation, dtype, norm, grouping):
        irreps = "3x0e+2x1o+2x2e"
        module = EquivariantNorm(
            irreps, norm=norm, grouping=grouping, device="cuda", dtype=dtype
        )
        features = torch.randn(7, 19, device="cuda", dtype=dtype)
        actual = module(rotate_features(features, irreps, rotation))
        expected = rotate_features(module(features), irreps, rotation)
        tolerance = 5e-5 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


fallback_cases = [
    ("cpu", torch.float32), ("cpu", torch.float64),
    ("cpu", torch.float16), ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.cuda),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.cuda),
]


@pytest.mark.parametrize("device,dtype", fallback_cases)
@pytest.mark.parametrize("grouping", ["merged", "separable", "per_irrep"])
@pytest.mark.parametrize("norm,layout,weighting", [
    ("layer", "mul_ir", "component"),
    ("rms", "ir_mul", "inverse_dim_squared"),
])
def test_pytorch_fallback(device, dtype, grouping, norm, layout, weighting):
    torch.manual_seed(31)
    options = dict(norm=norm, grouping=grouping, weighting=weighting, layout=layout, eps=1e-4)
    module = EquivariantNorm("3x0e+2x1o+2x2e", bias=True, device=device, dtype=dtype, **options)
    x = torch.randn(4, 38, device=device, dtype=dtype)[:, ::2].requires_grad_()
    low = dtype in (torch.float16, torch.bfloat16)
    cast = lambda value: value.float() if low else value
    actual = module(x)
    expected = reference_norm(cast(x), cast(module.weight), cast(module.bias), **options).to(dtype)
    assert actual.dtype == dtype
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.003 if low else 1e-5 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual, expected = actual.sin().mean(), expected.sin().mean()
    targets = (x, module.weight, module.bias)
    for order in range(3):
        left = torch.autograd.grad(actual, targets, create_graph=order < 2)
        right = torch.autograd.grad(expected, targets, create_graph=order < 2)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        if order < 2:
            directions = [torch.randn_like(a) for a in left]
            actual = sum((a * v).sum() for a, v in zip(left, directions))
            expected = sum((b * v).sum() for b, v in zip(right, directions))


def test_fallback_function_and_compile():
    from LunaEquivariance import equivariant_norm

    module = EquivariantNorm("2x0e+2x1o", layout="ir_mul", affine=False)
    x = torch.randn(3, 4, 2, requires_grad=True)
    expected = module(x)
    actual = equivariant_norm(x, "2x0e+2x1o", layout="ir_mul")
    torch.testing.assert_close(actual, expected)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    output = compiled(x)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(torch.autograd.grad(output.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
