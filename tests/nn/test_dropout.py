"""Dropout masks, boundaries and derivatives."""

import pytest
import torch
from e3nn import nn as reference_nn, o3

import LunaEquivariance as lunaeq
from LunaEquivariance import nn as luna_nn
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import assert_aoti, assert_reference_close, fp64


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.3, True), (1.0, True), (0.3, False)]
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "middle", "empty"])
def test_dropout(device, dtype, p, training, layout):
    irreps = o3.Irreps("2x0e + 2x1o + 1x2e")
    shape = (5, 3, irreps.dim) if layout == "middle" else (5, irreps.dim)
    if layout == "empty":
        shape = (0, irreps.dim)
    x = torch.randn(*shape, device=device, dtype=dtype)
    if layout == "strided":
        x = torch.randn(shape[0], 2 * shape[-1], device=device, dtype=dtype)[:, ::2]
    x.requires_grad_()
    reference = reference_nn.Dropout(irreps, p).train(training)
    operator = lunaeq.Dropout(irreps, p).train(training)
    reference_x = fp64((x,))[0]
    torch.manual_seed(42)
    mask = reference(torch.ones_like(x)) != 0
    expected = reference_x * mask / ((1 - p * training) or 1.0)
    torch.manual_seed(42)
    actual = operator(x)
    check_derivatives(actual, expected, x, reference_x)


def check_derivatives(actual, expected, x, reference_x):
    assert_reference_close(actual, expected)
    actual_loss, expected_loss = (actual.sin().sum(), expected.sin().sum())
    for order in range(4):
        actual_grad = torch.autograd.grad(actual_loss, x, create_graph=order < 3)[0]
        expected_grad = torch.autograd.grad(
            expected_loss, reference_x, create_graph=order < 3
        )[0]
        assert_reference_close(actual_grad, expected_grad)
        direction = torch.randn_like(x)
        actual_loss = (actual_grad * direction).sum()
        expected_loss = (expected_grad * direction.double()).sum()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_functional_dropout(device):
    irreps = "2x0e + 2x1o"
    x = torch.randn(5, 8, device=device, requires_grad=True)
    reference_x = fp64((x,))[0]
    torch.manual_seed(42)
    mask = reference_nn.Dropout(irreps, 0.3)(torch.ones_like(x)) != 0
    expected = reference_x * mask / (1 - 0.3)
    torch.manual_seed(42)
    actual = lunaeq.dropout(x, irreps, 0.3, training=True)
    assert_reference_close(actual, expected)
    grad = torch.randn_like(x)
    assert_reference_close(
        torch.autograd.grad(actual, x, grad)[0],
        torch.autograd.grad(expected, reference_x, grad.double())[0],
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(37,), (33, 1), (13, 37)])
def test_large_dropout(dtype, batch_shape):
    irreps = o3.Irreps("128x0e + 64x1o + 32x2e + 16x3o")
    x = torch.randn(
        *batch_shape, irreps.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    reference_x = fp64((x,))[0]
    torch.manual_seed(42)
    mask = reference_nn.Dropout(irreps, 0.3)(torch.ones_like(x)) != 0
    expected = reference_x * mask / (1 - 0.3)
    torch.manual_seed(42)
    actual = lunaeq.Dropout(irreps, 0.3)(x)
    check_derivatives(actual, expected, x, reference_x)


@pytest.mark.cuda
@pytest.mark.parametrize("irreps", ["2x0e+1x1o+1x3e", "3x0e+4x1o+1x2e", "1x0e+1x1o"])
def test_compact_mask_word_boundaries(irreps):
    x = torch.randn(17, o3.Irreps(irreps).dim, device="cuda", requires_grad=True)
    operator = lunaeq.Dropout(irreps, 0.35)
    reference = reference_nn.Dropout(irreps, 0.35)
    reference_x = fp64((x,))[0]
    torch.manual_seed(71)
    actual = operator(x)
    actual_rng = torch.cuda.get_rng_state()
    torch.manual_seed(71)
    mask = reference(torch.ones_like(x)) != 0
    expected = reference_x * mask / (1 - 0.35)
    torch.testing.assert_close(actual_rng, torch.cuda.get_rng_state(), atol=0, rtol=0)
    check_derivatives(actual, expected, x, reference_x)


def test_operator_aliases():
    operator = getattr(lunaeq, "Dropout")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "Dropout") is operator
    assert getattr(getattr(lunaeq, "nn"), "Dropout") is operator


def test_module_reference():
    module = luna_nn.Dropout("2x0e + 1x1o", p=0.3).eval()
    width = 5
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import Dropout

    torch.manual_seed(42)
    module = Dropout("2x0e+1x1o", p=0.3).to(device="cuda", dtype=dtype)
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


class DropoutChecks(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.operator = lunaeq.Dropout(f"{channels}x0e + {channels}x1o", p=0.25).cuda()

    def forward(self, x):
        output = self.operator(x)
        kept = output != 0
        vectors = kept[:, self.channels :].reshape(x.shape[0], self.channels, 3)
        return (
            torch.isclose(output, x * kept / 0.75, atol=1e-06, rtol=1e-06).all(),
            (vectors == vectors[:, :, :1]).all(),
            kept.any(),
        )


def compilation_case(channels):
    module = DropoutChecks(channels)
    samples = [(((torch.rand(rows, 4 * channels, device="cuda") + 0.2).requires_grad_(),), {})
               for rows in (7, 19)]
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (4, 8))
def test_compile(channels):
    module, samples = compilation_case(channels)
    torch._dynamo.reset()
    compiled = torch.compile(module.operator, fullgraph=True, dynamic=True,
                             options={"triton.cudagraphs": False})
    for (x,), _ in samples:
        output = compiled(x)
        kept = output != 0
        torch.testing.assert_close(output, x * kept / 0.75)
        vectors = kept[:, channels:].reshape(x.shape[0], channels, 3)
        assert torch.equal(vectors, vectors[:, :, :1].expand_as(vectors))
        seed = torch.randn_like(x)
        gradient = torch.autograd.grad(output, x, seed)[0]
        torch.testing.assert_close(gradient, seed * kept / 0.75)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (4, 8)]
    assert_aoti(cases, tmp_path)
