"""UUU outputs, layouts, equivariance and derivatives."""

from copy import deepcopy

import pytest
import torch

from LunaEquivariance import TensorProduct
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    rotate_features,
)
import LunaEquivariance as lunaeq


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
def test_forward(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uuu", shared)
    op = TensorProduct(tp)
    kwargs = {}
    reference = deepcopy(tp).double()(x.double(), y.double(), w.double())
    for inference in (False, True):
        with torch.set_grad_enabled(not inference):
            actual = op(x, y, None if shared else w, **kwargs)
        assert_reference_close(actual, reference)


def assert_derivatives_close(actual, expected):
    assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_derivatives(make_problem, shared, dtype):
    tp, x, y, w, src, dst = make_problem("uuu", shared, dtype=dtype)
    op = TensorProduct(tp)
    kwargs = {}
    actual = op(x, y, None if shared else w, **kwargs)
    g = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, g)
    first_directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in inputs[:3])
    )
    second_directions = tuple((torch.randn_like(v) for v in inputs))

    ref_inputs = fp64(inputs)
    ref_first = fp64(first_directions)
    ref_second = fp64(second_directions)
    expected = deepcopy(tp).double()(*ref_inputs[:3])

    def chain(out, inputs, first_directions, second_directions):
        first = torch.autograd.grad(
            out, inputs[:3], inputs[3], create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, inputs, first_directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, *first_directions), second_directions, retain_graph=True
        )
        return ((out,), first, second, third)

    for reference_order, actual_order in zip(
        chain(expected, ref_inputs, ref_first, ref_second),
        chain(actual, inputs, first_directions, second_directions),
    ):
        for reference, result in zip(reference_order, actual_order):
            assert_derivatives_close(result, reference)
    reference_x = torch.autograd.grad(
        expected, ref_inputs[0], ref_inputs[3], retain_graph=True
    )[0]
    assert_derivatives_close(torch.autograd.grad(actual, x, g)[0], reference_x)
    if not shared:
        force = op(x, y.detach(), w.detach(), **kwargs)
        assert_derivatives_close(torch.autograd.grad(force, x, g)[0], reference_x)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
def test_fp64_finite_differences(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uuu", shared, dtype=torch.float64)
    op = TensorProduct(tp, dtype=torch.float64)
    kwargs = {}
    if shared:
        fn, values = (lambda a, b: op(a, b, **kwargs), (x, y))
    else:
        fn, values = (lambda a, b, c: op(a, b, c, **kwargs), (x, y, w))
    nondet_tol = 0.0
    assert torch.autograd.gradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-06,
        atol=1e-05,
        rtol=0.0001,
        nondet_tol=nondet_tol,
    )
    assert torch.autograd.gradgradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-06,
        atol=1e-05,
        rtol=0.0001,
        nondet_tol=nondet_tol,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
def test_fp64_empty_inputs(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uuu", shared, dtype=torch.float64)
    op = TensorProduct(tp)
    x, y = (x[:0], y[:0])
    w = w if shared else w[:0]
    kwargs = {}
    result = op(x, y, None if shared else w, **kwargs)
    assert result.shape == (len(x), tp.irreps_out.dim)
    first = torch.autograd.grad(result.sum(), (x, y, w))
    for value, gradient in zip((x, y, w), first):
        assert gradient.shape == value.shape
        assert not gradient.count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equivariance(make_problem, rotation, dtype):
    reference, features, edges, weight, src, dst = make_problem("uuu", dtype=dtype)
    module = TensorProduct(reference, dtype=dtype)
    kwargs = {}
    output = module(features, edges, weight, **kwargs)
    actual = module(
        rotate_features(features, reference.irreps_in1, rotation),
        rotate_features(edges, reference.irreps_in2, rotation),
        weight,
        **kwargs,
    )
    expected = rotate_features(output, reference.irreps_out, rotation)
    tolerance = 5e-05 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.cuda
def test_converted_module_forward_and_three_orders(make_problem):
    tp, x, y, w, src, dst = make_problem("uuu")
    module = TensorProduct(tp)
    kwargs = {}
    module(x, y, w, **kwargs)
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem("uuu", dtype=dtype)
        assert not hasattr(module.op, "graph")
        actual = module(x, y, w, **kwargs)
        grad = torch.randn_like(actual, requires_grad=True)
        inputs = (x, y, w, grad)
        first_seeds = tuple(
            (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
        )
        second_seeds = tuple((torch.randn_like(value) for value in inputs))
        ref_inputs = fp64(inputs)
        ref_first = fp64(first_seeds)
        ref_second = fp64(second_seeds)
        expected = deepcopy(reference).double()(*ref_inputs[:3])
        assert_reference_close(actual, expected)
        results = []
        for output, targets, directions, second_directions in (
            (expected, ref_inputs, ref_first, ref_second),
            (actual, inputs, first_seeds, second_seeds),
        ):
            first = torch.autograd.grad(
                output, targets[:3], targets[3], create_graph=True, retain_graph=True
            )
            second = torch.autograd.grad(
                first, targets, directions, create_graph=True, retain_graph=True
            )
            third = torch.autograd.grad(
                second, (*targets, *directions), second_directions, retain_graph=True
            )
            results.append((*first, *second, *third))
        for expected_grad, actual_grad in zip(*results):
            assert_reference_close(actual_grad, expected_grad)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_forward_and_three_orders(make_problem, dtype):
    tp, x, y, w, src, dst = make_problem("uuu", dtype=dtype)
    cls = TensorProduct
    instructions = [
        (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight)
        for ins in tp.instructions
    ]
    module = cls(
        tp.irreps_in1,
        tp.irreps_in2,
        tp.irreps_out,
        instructions,
        internal_weights=False,
        shared_weights=False,
        dtype=dtype,
    )
    baseline = TensorProduct(tp, dtype=dtype)
    kwargs = {}
    actual = module(x, y, w, **kwargs)
    original = baseline(x, y, w, **kwargs)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    grad = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, grad)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    ref_inputs = fp64(inputs)
    ref_first = fp64(first_seeds)
    ref_second = fp64(second_seeds)
    expected = deepcopy(tp).double()(*ref_inputs[:3])
    assert_reference_close(actual, expected)
    results = []
    for output, targets, directions, second_directions in (
        (expected, ref_inputs, ref_first, ref_second),
        (actual, inputs, first_seeds, second_seeds),
    ):
        first = torch.autograd.grad(
            output, targets[:3], targets[3], create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, targets, directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*targets, *directions), second_directions, retain_graph=True
        )
        results.append((*first, *second, *third))
    for reference, result in zip(*results):
        assert_reference_close(result, reference)


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
    "internal,shared,env_shape",
    [(False, False, None), (False, True, None), (True, True, None)],
)
def test_cuda_tensor_product_modes(dtype, internal, shared, env_shape):
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(42)
    cls = TensorProduct
    module = cls(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uuu", True)],
        internal_weights=internal,
        shared_weights=shared,
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(4, 6, device="cuda", dtype=dtype).requires_grad_()
    y = torch.randn(4, 6, device="cuda", dtype=dtype).requires_grad_()
    args = (x, y)
    if not internal:
        shape = (module.weight_numel,) if shared else (4, module.weight_numel)
        args += (torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(),)
    kwargs = {}
    assert_cuda_report(module, args, **kwargs)


def test_irreps_constructor_registers_internal_weights():
    module = TensorProduct(
        "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uuu", True)], dtype=torch.float64
    )
    assert module.internal_weights and module.shared_weights
    assert module.weight.dtype == torch.float64
    assert module.weight is dict(module.named_parameters())["tp.weight"]
    before = module.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(module.weight, before - 0.1)


def compilation_case(channels):
    irreps = lunaeq.Irreps(f"{channels}x0e + {channels}x1o")
    edge_irreps = irreps
    instructions = [(i, j, k, "uuu", True)
                    for i, (_, a) in enumerate(irreps)
                    for j, (_, b) in enumerate(edge_irreps)
                    for k, (_, c) in enumerate(irreps) if c in a * b]
    module = lunaeq.TensorProduct(irreps, edge_irreps, irreps, instructions,
                          internal_weights=False, shared_weights=False).cuda()
    weight_numel = module.weight_numel

    samples = []
    for rows, edges in ((7, 7), (19, 19)):
        x = (torch.randn(rows, irreps.dim, device="cuda") * 0.3).requires_grad_()
        edge = (torch.randn(edges, edge_irreps.dim, device="cuda") * 0.3).requires_grad_()
        weight = (torch.randn(edges, weight_numel, device="cuda") * 0.3).requires_grad_()
        samples.append(((x, edge, weight), {}))
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
