"""Graph softmax outputs, envelopes and derivatives."""

import pytest
import torch

from LunaEquivariance import build_graph_softmax_row_ptr, graph_softmax
from LunaEquivariance.utils.testing import check_compatibility
import LunaEquivariance as lunaeq
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
)


class TestGraphSoftmax:
    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("env_shape", [None, "edge", "column", "channel"])
    @pytest.mark.parametrize("smooth_out", [False, True])
    def test_graph_softmax_derivatives(self, dtype, index_dtype, env_shape, smooth_out):
        torch.manual_seed(731)
        offsets = [0, 3, 3, 7]
        csr = torch.tensor(offsets, device="cuda", dtype=index_dtype)
        data = torch.randn(7, 6, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
        env = None
        if env_shape is not None:
            shape = {"edge": (7,), "column": (7, 1), "channel": (7, 3)}[env_shape]
            env = (torch.rand(shape, device="cuda", dtype=dtype) + 0.2).requires_grad_()
        if smooth_out and env is None:
            with pytest.raises(ValueError, match="smooth_out requires env"):
                graph_softmax(data, csr, smooth_out=True)
            return
        eps = 1e-05
        ref_data, ref_env = fp64((data, env))
        parts = []
        for start, end in zip(offsets, offsets[1:]):
            logits = ref_data[start:end]
            if start == end:
                parts.append(logits)
                continue
            weights = 1 if ref_env is None else ref_env[start:end].reshape(end - start, -1)
            numerator = (logits - logits.detach().amax(0, keepdim=True)).exp() * weights
            part = numerator / (numerator.sum(0, keepdim=True) + eps)
            parts.append(part * weights if smooth_out else part)
        expected = (torch.cat(parts),)
        actual = (graph_softmax(data, csr, env=env, eps=eps, smooth_out=smooth_out),)
        targets = (data,) if env is None else (data, env)
        ref_targets = (ref_data,) if ref_env is None else (ref_data, ref_env)
        for order in range(4):
            for value, reference in zip(actual, expected):
                if value is None or reference is None:
                    assert value is reference
                else:
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
            )
            expected = torch.autograd.grad(
                tuple((b for _, b in pairs)),
                ref_targets,
                ref_seeds,
                create_graph=order < 2,
                allow_unused=True,
            )
            targets = (*targets, *seeds)
            ref_targets = (*ref_targets, *ref_seeds)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_row_ptr_preserves_empty_segments(self, dtype):
        ids = torch.tensor([0, 0, 2, 2, 2], device="cuda", dtype=dtype)
        expected = torch.tensor([0, 2, 2, 5, 5], device="cuda", dtype=torch.int64)
        torch.testing.assert_close(build_graph_softmax_row_ptr(ids, 4), expected)


fallback_cases = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cpu", torch.float16),
    ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.cuda),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.cuda),
]


@pytest.mark.parametrize("device,dtype", fallback_cases)
@pytest.mark.parametrize(
    "env_shape,smooth", [(None, False), ((7,), False), ((7, 1), True), ((7, 3), True)]
)
def test_pytorch_fallback(device, dtype, env_shape, smooth):
    from LunaEquivariance import GraphSoftmax

    torch.manual_seed(31)
    x = torch.randn(7, 6, device=device, dtype=dtype)[:, ::2].requires_grad_()
    env = (
        None
        if env_shape is None
        else (torch.rand(env_shape, device=device, dtype=dtype) + 0.2).requires_grad_()
    )
    csr = torch.tensor([0, 3, 3, 7], device=device, dtype=torch.int32)
    module = GraphSoftmax(eps=1e-05, smooth_out=smooth)
    actual = module(x, csr, env=env)
    values, ref_env = fp64((x, env))
    envelope = None if ref_env is None else ref_env.reshape(7, -1)
    parts = []
    for begin, end in ((0, 3), (3, 3), (3, 7)):
        scores = values[begin:end]
        if begin == end:
            parts.append(scores)
            continue
        weight = 1 if envelope is None else envelope[begin:end]
        numerator = (scores - scores.detach().amax(0, keepdim=True)).exp() * weight
        result = numerator / (numerator.sum(0, keepdim=True) + module.eps)
        parts.append(result * weight if smooth else result)
    expected = torch.cat(parts)
    assert actual.dtype == dtype
    assert_reference_close(actual, expected)
    targets = (x,) if env is None else (x, env)
    ref_targets = (values,) if ref_env is None else (values, ref_env)
    actual, expected = (actual.sin().mean(), expected.sin().mean())
    for order in range(3):
        left = torch.autograd.grad(actual, targets, create_graph=order < 2)
        right = torch.autograd.grad(expected, ref_targets, create_graph=order < 2)
        for a, b in zip(left, right):
            assert_reference_close(a, b)
        if order < 2:
            directions = [torch.randn_like(a) for a in left]
            actual = sum(((a * v).sum() for a, v in zip(left, directions)))
            expected = sum(((b * v.double()).sum() for b, v in zip(right, directions)))


def test_fallback_compile_and_empty_input():
    from LunaEquivariance import GraphSoftmax

    module = GraphSoftmax()
    x = torch.randn(5, 3, requires_grad=True)
    csr = torch.tensor([0, 2, 2, 5])
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    expected, actual = (module(x, csr), compiled(x, csr))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )
    empty = torch.empty(0, 3, requires_grad=True)
    output = module(empty, torch.tensor([0, 0, 0]))
    assert output.shape == empty.shape
    output.sum().backward()
    assert empty.grad.shape == empty.shape


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
    "env_shape,smooth",
    [
        (shape, smooth)
        for shape in (None, (7,), (7, 1), (7, 3))
        for smooth in (False, True)
        if shape is not None or not smooth
    ],
)
def test_cuda_graph_softmax(dtype, env_shape, smooth):
    from LunaEquivariance import GraphSoftmax

    torch.manual_seed(42)
    x = torch.randn(7, 3, device="cuda", dtype=dtype).requires_grad_()
    csr = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    kwargs = (
        {}
        if env_shape is None
        else dict(
            env=(
                torch.rand(env_shape, device="cuda", dtype=dtype) + 0.2
            ).requires_grad_()
        )
    )
    assert_cuda_report(GraphSoftmax(smooth_out=smooth), (x, csr), **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_pytorch_reference(dtype):
    from LunaEquivariance import EquivariantNorm, GraphSoftmax

    torch.manual_seed(12)
    module = GraphSoftmax()
    args = (torch.randn(4, 3, dtype=dtype, requires_grad=True), torch.tensor([0, 2, 4]))
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
    module = lunaeq.GraphSoftmax(eps=1e-5, smooth_out=True).cuda()
    samples = []
    for edges, nodes in ((17, 3), (41, 5)):
        x = torch.randn(edges, channels, device="cuda", requires_grad=True)
        csr = torch.linspace(0, edges, nodes + 1, device="cuda", dtype=torch.int32)
        env = (torch.rand(edges, 1, device="cuda") + 0.2).requires_grad_()
        samples.append(((x, csr), {"env": env}))
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
