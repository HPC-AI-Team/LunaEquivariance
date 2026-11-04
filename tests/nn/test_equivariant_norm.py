"""Equivariant normalization statistics, layouts and derivatives."""

import pytest
import torch
from e3nn import o3

import LunaEquivariance as lunaeq
from LunaEquivariance import EquivariantNorm
from LunaEquivariance.modules.equ_norm import LunaEqEquivariantNorm
from LunaEquivariance.nn.equ_norm.interface import equivariant_norm
from LunaEquivariance.o3.norm.interface import norm
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    rotate_features,
)


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
                "inverse_dim_squared": ir.dim ** 2,
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
    def test_forward_and_three_derivatives(self, dtype, norm, grouping, weighting, layout):
        torch.manual_seed(732)
        options = dict(
            norm=norm, grouping=grouping, weighting=weighting, layout=layout, eps=0.0001
        )
        module = EquivariantNorm(
            "3x0e+2x1o+2x2e", bias=True, device="cuda", dtype=dtype, **options
        )
        with torch.no_grad():
            module.weight.uniform_(0.8, 1.2)
            module.bias.uniform_(-0.1, 0.1)
        x = torch.randn(4, 38, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
        actual = (module(x),)
        ref_targets = fp64((x, module.weight, module.bias))
        expected = (reference_norm(*ref_targets, **options),)
        targets = (x, module.weight, module.bias)
        for order in range(4):
            for value, reference in zip(actual, expected):
                assert_reference_close(value, reference)
            if order == 3:
                break
            pairs = [
                (a, b)
                for a, b in zip(actual, expected)
                if a is not None and a.requires_grad
            ]
            seeds = tuple((torch.randn_like(a, requires_grad=True) for a, _ in pairs))
            ref_seeds = fp64(seeds)
            actual = torch.autograd.grad(
                tuple((a for a, _ in pairs)),
                targets,
                seeds,
                create_graph=order < 2,
                allow_unused=True,
                materialize_grads=True,
            )
            expected = torch.autograd.grad(
                tuple((b for _, b in pairs)),
                ref_targets,
                ref_seeds,
                create_graph=order < 2,
                allow_unused=True,
                materialize_grads=True,
            )
            targets = (*targets, *seeds)
            ref_targets = (*ref_targets, *ref_seeds)

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
        tolerance = 5e-05 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


fallback_cases = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cpu", torch.float16),
    ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.cuda),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.cuda),
]


@pytest.mark.parametrize("device,dtype", fallback_cases)
@pytest.mark.parametrize("grouping", ["merged", "separable", "per_irrep"])
@pytest.mark.parametrize(
    "norm,layout,weighting",
    [("layer", "mul_ir", "component"), ("rms", "ir_mul", "inverse_dim_squared")],
)
def test_pytorch_fallback(device, dtype, grouping, norm, layout, weighting):
    torch.manual_seed(31)
    options = dict(
        norm=norm, grouping=grouping, weighting=weighting, layout=layout, eps=0.0001
    )
    module = EquivariantNorm(
        "3x0e+2x1o+2x2e", bias=True, device=device, dtype=dtype, **options
    )
    x = torch.randn(4, 38, device=device, dtype=dtype)[:, ::2].requires_grad_()
    actual = module(x)
    ref_targets = fp64((x, module.weight, module.bias))
    expected = reference_norm(*ref_targets, **options)
    assert actual.dtype == dtype
    assert_reference_close(actual, expected)
    actual, expected = (actual.sin().mean(), expected.sin().mean())
    targets = (x, module.weight, module.bias)
    for order in range(3):
        left = torch.autograd.grad(actual, targets, create_graph=order < 2)
        right = torch.autograd.grad(expected, ref_targets, create_graph=order < 2)
        for a, b in zip(left, right):
            assert_reference_close(a, b)
        if order < 2:
            directions = [torch.randn_like(a) for a in left]
            actual = sum(((a * v).sum() for a, v in zip(left, directions)))
            expected = sum(((b * v.double()).sum() for b, v in zip(right, directions)))


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
    torch.testing.assert_close(
        torch.autograd.grad(output.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_functional_norm_cuda_three_orders(dtype):
    irreps = "2x0e + 3x1o + 2x2e"
    x = torch.randn(
        11, o3.Irreps(irreps).dim, device="cuda", dtype=dtype, requires_grad=True
    )
    module = LunaEqEquivariantNorm(irreps, dtype=dtype, device="cuda")
    inputs = (x, module.weight, module.bias)
    function = lambda: equivariant_norm(x, irreps, module.weight, module.bias)
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


def test_operator_aliases():
    operator = getattr(lunaeq, "EquivariantNorm")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "EquivariantNorm") is operator
    assert getattr(getattr(lunaeq, "nn"), "EquivariantNorm") is operator


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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "grouping,norm,weighting,layout",
    [
        ("merged", "layer", "component", "mul_ir"),
        ("separable", "layer", "norm", "ir_mul"),
        ("per_irrep", "rms", "inverse_dim_squared", "mul_ir"),
        ("merged", "rms", "norm", "ir_mul"),
        ("separable", "rms", "component", "mul_ir"),
        ("per_irrep", "layer", "inverse_dim_squared", "ir_mul"),
        ("merged", "layer", "custom", "ir_mul"),
        ("separable", "rms", "custom", "mul_ir"),
    ],
)
def test_cuda_equivariant_norm(dtype, grouping, norm, weighting, layout):
    from LunaEquivariance import EquivariantNorm

    torch.manual_seed(42)
    options = (
        dict(stat_weights=[0.2, 0.3, 0.1])
        if weighting == "custom"
        else dict(weighting=weighting)
    )
    module = EquivariantNorm(
        "3x0e+2x1o+2x2e",
        grouping=grouping,
        norm=norm,
        layout=layout,
        bias=True,
        device="cuda",
        dtype=dtype,
        **options,
    )
    with torch.no_grad():
        module.weight.uniform_(0.8, 1.2)
        module.bias.uniform_(-0.1, 0.1)
    x = torch.randn(4, 19, device="cuda", dtype=dtype).requires_grad_()
    assert_cuda_report(module, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_pytorch_reference(dtype):
    from LunaEquivariance import EquivariantNorm, GraphSoftmax

    torch.manual_seed(12)
    module = EquivariantNorm("3x0e+2x1o", dtype=dtype)
    args = (torch.randn(4, 9, dtype=dtype, requires_grad=True),)
    report = check_compatibility(
        module,
        args,
        derivative_order=3,
        warmup=0,
        repeats=1,
        atol=0.03,
        rtol=0.03,
        verbose=False,
    )
    assert report["reference"] == "PyTorch"
    assert report["passed"], report


def compilation_case(channels):
    irreps = f"{channels}x0e + {channels}x1o"
    module = lunaeq.EquivariantNorm(irreps, affine=True, bias=True).cuda()
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
