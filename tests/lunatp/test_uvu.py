"""UVU outputs, layouts, equivariance and derivatives."""

import unittest
from copy import deepcopy

import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct

from LunaEquivariance import TensorProduct, TensorProductConfig, TensorProductConv
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
    tp, x, y, w, src, dst = make_problem("uvu", shared)
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
    tp, x, y, w, src, dst = make_problem("uvu", shared, dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uvu", shared, dtype=torch.float64)
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
    tp, x, y, w, src, dst = make_problem("uvu", shared, dtype=torch.float64)
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
    reference, features, edges, weight, src, dst = make_problem("uvu", dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uvu")
    module = TensorProduct(tp)
    kwargs = {}
    module(x, y, w, **kwargs)
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem("uvu", dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uvu", dtype=dtype)
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
        [(0, 0, 0, "uvu", True)],
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
        "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uvu", True)], dtype=torch.float64
    )
    assert module.internal_weights and module.shared_weights
    assert module.weight.dtype == torch.float64
    assert module.weight is dict(module.named_parameters())["tp.weight"]
    before = module.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(module.weight, before - 0.1)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_local_configuration_normalization(
    shared, dtype, irrep_normalization, path_normalization
):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        for variances in (None, [1.2, 0.7, 0.0]):
            instructions = [
                (0, 0, 0, "uvu", True, 1.3),
                (1, 1, 0, "uvu", True, 0.25),
                (0, 1, 1, "uvu", True, 2.1),
                (1, 0, 1, "uvu", True, 0.0),
                (1, 1, 2, "uvu", True, 0.75),
            ]
            args = (
                "3x0e + 3x1o + 0x2e",
                "2x0e + 2x1o + 1x2e",
                "3x0e + 3x1o + 3x2e",
                instructions,
            )
            kwargs = dict(
                internal_weights=shared,
                shared_weights=shared,
                in1_var=variances,
                in2_var=variances,
                out_var=variances,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
            torch.manual_seed(981)
            expected = E3nnTensorProduct(
                *args, **kwargs, compile_left_right=False, compile_right=False
            )
            torch.manual_seed(981)
            actual = TensorProductConfig(*args, **kwargs)
            assert actual.instructions == expected.instructions
            assert actual.weight_numel == expected.weight_numel
            assert actual.weight.dtype == dtype and torch.equal(
                actual.weight, expected.weight
            )
            assert actual.internal_weights == expected.internal_weights
            assert actual.shared_weights == expected.shared_weights
            for field in ("irreps_in1", "irreps_in2", "irreps_out"):
                own, reference = (getattr(actual, field), getattr(expected, field))
                assert own == reference and own.dim == reference.dim
                assert own.slices() == reference.slices()
    finally:
        torch.set_default_dtype(previous)


def frozen_chain(out, x, y, g, directions, second_directions):
    first = torch.autograd.grad(out, (x, y), g, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(
        first, (x, y, g), directions, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second, (x, y, g, *directions), second_directions, retain_graph=True
    )
    return ((out,), first, second, third)


def verify_frozen_problem(tp, x, y, w, mode, shared, dtype):
    w.requires_grad_(False)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    actual = op(x, y, None if shared else w)
    g = torch.randn_like(actual, requires_grad=True)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in (x, y)))
    second_directions = tuple((torch.randn_like(v) for v in (x, y, g)))
    ref_x, ref_y, ref_w, ref_g, ref_directions, ref_second = fp64(
        (x, y, w, g, directions, second_directions)
    )
    expected = deepcopy(tp).double()(ref_x, ref_y, ref_w)
    reference = frozen_chain(expected, ref_x, ref_y, ref_g, ref_directions, ref_second)
    result = frozen_chain(actual, x, y, g, directions, second_directions)
    for results, references in zip(result, reference):
        for value, wanted in zip(results, references):
            assert_shape_close(value, wanted, dtype)
    assert w.grad is None
    first = torch.autograd.grad(actual, (x, y), g, retain_graph=True)
    for value, wanted in zip(first, reference[1]):
        assert_shape_close(value, wanted, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_weights_through_third_order(make_problem, shared, dtype):
    tp, x, y, w, *_ = make_problem("uvu", shared, dtype=dtype)
    verify_frozen_problem(tp, x, y, w, "uvu", shared, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_weights_empty_inputs(make_problem, shared, dtype):
    tp, x, y, w, *_ = make_problem("uvu", shared, dtype=dtype)
    w.requires_grad_(False)
    x, y = (x[:0], y[:0])
    w = w if shared else w[:0]
    op = TensorProduct(tp)
    out = op(x, y, None if shared else w)
    g = torch.randn_like(out, requires_grad=True)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in (x, y)))
    second_directions = tuple((torch.randn_like(v) for v in (x, y, g)))
    for order in frozen_chain(out, x, y, g, directions, second_directions):
        for value in order:
            assert value.numel() == 0


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_external_weights_from_network_keep_parameter_gradients(make_problem, dtype):
    tp, x, y, w, *_ = make_problem("uvu", dtype=dtype)
    parameter = w.detach().clone().requires_grad_()
    weight = torch.tanh(parameter)
    op = TensorProduct(tp)
    actual = op(x, y, weight)
    inputs = (x, y, parameter)
    g = torch.randn_like(actual, requires_grad=True)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in inputs))
    second_directions = tuple((torch.randn_like(v) for v in (*inputs, g)))

    ref_inputs, ref_g, ref_directions, ref_second = fp64(
        (inputs, g, directions, second_directions)
    )
    expected = deepcopy(tp).double()(
        ref_inputs[0], ref_inputs[1], torch.tanh(ref_inputs[2])
    )

    def chain(out, inputs, g, directions, second_directions):
        first = torch.autograd.grad(
            out, inputs, g, create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, (*inputs, g), directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, g, *directions), second_directions, retain_graph=True
        )
        return (first, second, third)

    for values, wanted_values in zip(
        chain(actual, inputs, g, directions, second_directions),
        chain(expected, ref_inputs, ref_g, ref_directions, ref_second),
    ):
        for value, wanted in zip(values, wanted_values):
            assert_shape_close(value, wanted, dtype)


SHAPES = {
    "tail": ("33x0e + 33x1o + 7x2e", "2x0e + 3x1o + 2x3o", 2),
    "vector_tail": ("35x2e + 1x0e", "2x0e", 1),
    "mixed": ("129x0e + 97x1o + 1x0e", "3x0e + 2x1o + 3x3o", 2),
    "wide": ("1025x0e + 1025x1o", "2x0e + 2x1o", 2),
    "grouped_tail": ("65x0e", "32x0e", 1),
    "grouped_wide": ("1057x0e", "32x0e", 1),
    "large_packed": ("65537x0e + 65601x1o", "1x0e", 2),
}


def make_uvu_shape(shape, shared, dtype, device):
    inputs, edges, active_irreps = SHAPES[shape]
    inputs, edges = (Irreps(inputs), Irreps(edges))
    instructions = [
        (i, j, k, "uvu", True)
        for i, (mul_in, ir_in) in enumerate(inputs[:active_irreps])
        for j, (_, ir_edge) in enumerate(edges)
        for k, (mul_out, ir_out) in enumerate(inputs[:active_irreps])
        if mul_in == mul_out and ir_out in ir_in * ir_edge
    ]
    if shape in ("tail", "mixed"):
        instructions.append(instructions[0])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return E3nnTensorProduct(
            inputs,
            edges,
            inputs,
            instructions,
            internal_weights=shared,
            shared_weights=shared,
        ).to(device)
    finally:
        torch.set_default_dtype(old_dtype)


def assert_shape_close(actual, expected, dtype):
    assert actual.dtype == dtype
    assert_reference_close(actual, expected)
    if not expected.count_nonzero():
        assert not actual.count_nonzero()


def derivative_chain(out, inputs, directions, second_directions):
    first = torch.autograd.grad(
        out, inputs[:3], inputs[3], create_graph=True, retain_graph=True
    )
    second = torch.autograd.grad(
        first, inputs, directions, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second, (*inputs, *directions), second_directions, retain_graph=True
    )
    return ((out,), first, second, third)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uvu_complete_shape_derivatives(shape, shared, dtype):
    torch.manual_seed(4921)
    tp = make_uvu_shape(shape, shared, dtype, "cuda")
    op = TensorProduct(tp)
    x = (
        torch.randn(tp.irreps_in1.dim, 21, device="cuda", dtype=dtype)
        .t()[2:19]
        .requires_grad_()
    )
    y = (
        torch.randn(tp.irreps_in2.dim, 21, device="cuda", dtype=dtype)
        .t()[2:19]
        .requires_grad_()
    )
    w = (
        tp.weight
        if shared
        else torch.randn(tp.weight_numel, 21, device="cuda", dtype=dtype)
        .t()[2:19]
        .requires_grad_()
    )
    g = torch.randn(
        17, tp.irreps_out.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    inputs = (x, y, w, g)
    first_directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in inputs[:3])
    )
    second_directions = tuple((torch.randn_like(v) for v in inputs))
    ref_inputs, ref_first, ref_second = fp64(
        (inputs, first_directions, second_directions)
    )
    expected = derivative_chain(
        deepcopy(tp).double()(*ref_inputs[:3]), ref_inputs, ref_first, ref_second
    )
    actual = derivative_chain(
        op(x, y, None if shared else w), inputs, first_directions, second_directions
    )
    for actual_order, reference_order in zip(actual, expected):
        for result, reference in zip(actual_order, reference_order):
            assert_shape_close(result, reference, dtype)
    force = op(x, y.detach(), None if shared else w.detach())
    assert_shape_close(torch.autograd.grad(force, x, g)[0], expected[1][0], dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        streamed = op(x, y, None if shared else w)
    torch.cuda.current_stream().wait_stream(stream)
    assert_shape_close(streamed, expected[0][0], dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        captured = op(x, y, None if shared else w)
    graph.replay()
    torch.cuda.synchronize()
    assert_shape_close(captured, expected[0][0], dtype)
    empty_x, empty_y = (x[:0], y[:0])
    empty_w = w if shared else w[:0]
    out = op(empty_x, empty_y, None if shared else empty_w)
    empty_g = g[:0]
    first = torch.autograd.grad(
        out, (empty_x, empty_y, empty_w), empty_g, create_graph=True
    )
    empty_inputs = (empty_x, empty_y, empty_w, empty_g)
    directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in empty_inputs[:3])
    )
    second = torch.autograd.grad(first, empty_inputs, directions, create_graph=True)
    third = torch.autograd.grad(
        second,
        (*empty_inputs, *directions),
        tuple((torch.randn_like(v) for v in second)),
    )
    stages = (
        (first, empty_inputs[:3]),
        (second, empty_inputs),
        (third, (*empty_inputs, *directions)),
    )
    for results, values in stages:
        for value, result in zip(values, results):
            assert result.shape == value.shape
            assert not result.count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_uvu_tail_vector_store_boundary(dtype, shared):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "35x2e",
            "1x0e",
            "35x2e + 1x0e",
            [(0, 0, 0, "uvu", True)],
            internal_weights=shared,
            shared_weights=shared,
        ).cuda()
    finally:
        torch.set_default_dtype(old_dtype)
    op = TensorProduct(tp)
    x = torch.randn(7, 175, device="cuda", dtype=dtype)
    y = torch.randn(7, 1, device="cuda", dtype=dtype)
    w = tp.weight if shared else torch.randn(7, 35, device="cuda", dtype=dtype)
    expected = deepcopy(tp).double()(x.double(), y.double(), w.double())
    with torch.no_grad():
        actual = op(x, y, None if shared else w)
        assert_shape_close(actual, expected, dtype)
        assert not actual[:, -1].count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uvu_shared_tail_partial_reductions(dtype):
    torch.manual_seed(5837)
    tp = make_uvu_shape("tail", True, dtype, "cuda")
    op = TensorProduct(tp)
    items = 2051
    x = torch.randn(
        items, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        items, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = tp.weight
    g = torch.randn(
        items, tp.irreps_out.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    inputs = (x, y, w, g)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in inputs[:3]))
    second_directions = tuple((torch.randn_like(v) for v in inputs))
    ref_inputs, ref_directions, ref_second = fp64(
        (inputs, directions, second_directions)
    )
    expected = derivative_chain(
        deepcopy(tp).double()(*ref_inputs[:3]), ref_inputs, ref_directions, ref_second
    )
    actual = derivative_chain(op(x, y), inputs, directions, second_directions)
    for results, references in zip(actual, expected):
        for result, reference in zip(results, references):
            assert_shape_close(result, reference, dtype)


@pytest.mark.cuda
@unittest.skipUnless(torch.cuda.is_available(), "UVU requires CUDA")
class UVUOutputTests(unittest.TestCase):
    def test_overlapping_outputs_and_derivatives(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                torch.manual_seed(939)
                inputs = Irreps("32x0e + 32x1o")
                edges = Irreps("1x0e + 1x1o")
                outputs = inputs + Irreps("32x2e")
                instructions = [
                    (i, j, k, "uvu", True)
                    for i, (_, a) in enumerate(inputs)
                    for j, (_, b) in enumerate(edges)
                    for k, (_, c) in enumerate(inputs)
                    if c in a * b
                ]
                tp = E3nnTensorProduct(
                    inputs,
                    edges,
                    outputs,
                    instructions,
                    internal_weights=shared,
                    shared_weights=shared,
                ).cuda()
                op = TensorProduct(tp)
                x = torch.randn(inputs.dim, 19, device="cuda").t().requires_grad_()
                y = torch.randn(edges.dim, 19, device="cuda").t().requires_grad_()
                weight = (
                    tp.weight
                    if shared
                    else torch.randn(19, tp.weight_numel, device="cuda", requires_grad=True)
                )
                grad_out = torch.randn(19, outputs.dim, device="cuda", requires_grad=True)
                operands = (x, y, weight, grad_out)
                first_directions = tuple(
                    (torch.randn_like(t, requires_grad=True) for t in operands[:3])
                )
                second_directions = tuple((torch.randn_like(t) for t in operands))
                ref_operands, ref_first, ref_second = fp64(
                    (operands, first_directions, second_directions)
                )
                expected = derivative_chain(
                    deepcopy(tp).double()(*ref_operands[:3]),
                    ref_operands,
                    ref_first,
                    ref_second,
                )
                actual = derivative_chain(
                    op(x, y, None if shared else weight),
                    operands,
                    first_directions,
                    second_directions,
                )
                for order, (reference, result) in enumerate(zip(expected, actual)):
                    with self.subTest(order=order):
                        for left, right in zip(reference, result):
                            assert_reference_close(right, left)
                for result in (actual[0][0], actual[2][3], actual[3][3]):
                    self.assertEqual(torch.count_nonzero(result[:, inputs.dim :]).item(), 0)
                with torch.no_grad():
                    inference = op(x, y, None if shared else weight)
                assert_reference_close(inference, expected[0][0])


def compilation_case(channels):
    irreps = lunaeq.Irreps(f"{channels}x0e + {channels}x1o")
    edge_irreps = lunaeq.Irreps("2x0e + 2x1o")
    instructions = [(i, j, k, "uvu", True)
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
