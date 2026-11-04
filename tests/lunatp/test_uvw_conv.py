"""UVW CONV outputs, layouts, equivariance and derivatives."""

import math
from copy import deepcopy
from functools import partial

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance import (
    TensorProductConv,
    make_graph,
    prepare_tensor_product,
    tensor_product_conv,
)
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    ConvolutionInputs,
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    graph_tensors,
    rotate_features,
)
import LunaEquivariance as lunaeq


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
def test_forward(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uvw_conv", shared)
    op = TensorProductConv(tp)
    kwargs = {}
    reference = deepcopy(tp).double()(x[src].double(), y.double(), w.double())
    kwargs["graph"] = op.create_graph(src, dst, len(x))
    reference = reference.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, reference
    )
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
    tp, x, y, w, src, dst = make_problem("uvw_conv", shared, dtype=dtype)
    op = TensorProductConv(tp)
    kwargs = {"graph": op.create_graph(src, dst, len(x))}
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
    expected = deepcopy(tp).double()(ref_inputs[0][src], ref_inputs[1], ref_inputs[2])
    expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, expected
    )

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
    tp, x, y, w, src, dst = make_problem("uvw_conv", shared, dtype=torch.float64)
    op = TensorProductConv(tp, dtype=torch.float64)
    kwargs = {"graph": op.create_graph(src, dst, len(x))}
    if shared:
        fn, values = (lambda a, b: op(a, b, **kwargs), (x, y))
    else:
        fn, values = (lambda a, b, c: op(a, b, c, **kwargs), (x, y, w))
    nondet_tol = 1e-12
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
    tp, x, y, w, src, dst = make_problem("uvw_conv", shared, dtype=torch.float64)
    op = TensorProductConv(tp)
    x, y = (x, y[:0])
    w = w if shared else w[:0]
    kwargs = {"graph": op.create_graph(src[:0], dst[:0], len(x))}
    result = op(x, y, None if shared else w, **kwargs)
    assert result.shape == (len(x), tp.irreps_out.dim)
    first = torch.autograd.grad(result.sum(), (x, y, w))
    for value, gradient in zip((x, y, w), first):
        assert gradient.shape == value.shape
        assert not gradient.count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equivariance(make_problem, rotation, dtype):
    reference, features, edges, weight, src, dst = make_problem("uvw_conv", dtype=dtype)
    module = TensorProductConv(reference, dtype=dtype)
    kwargs = {"graph": module.create_graph(src, dst, len(features))}
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
    tp, x, y, w, src, dst = make_problem("uvw_conv")
    module = TensorProductConv(tp)
    kwargs = {"graph": module.create_graph(src, dst, len(x))}
    module(x, y, w, **kwargs)
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem("uvw_conv", dtype=dtype)
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
        expected = deepcopy(reference).double()(
            ref_inputs[0][src], ref_inputs[1], ref_inputs[2]
        )
        expected = expected.new_zeros((len(x), reference.irreps_out.dim)).index_add(
            0, dst, expected
        )
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
    tp, x, y, w, src, dst = make_problem("uvw_conv", dtype=dtype)
    cls = TensorProductConv
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
    baseline = TensorProductConv(tp, dtype=dtype)
    kwargs = {}
    graph = module.create_graph(src, dst, len(x))
    kwargs["graph"] = graph
    actual = module(x, y, w, **kwargs)
    original = baseline(x, y, w, **kwargs)
    tolerance = 5e-05 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, original, rtol=tolerance, atol=tolerance)
    grad = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, grad)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    ref_inputs = fp64(inputs)
    ref_first = fp64(first_seeds)
    ref_second = fp64(second_seeds)
    expected = deepcopy(tp).double()(ref_inputs[0][src], ref_inputs[1], ref_inputs[2])
    expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, expected
    )
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
    [
        (False, False, (4,)),
        (False, False, (4, 1)),
        (False, True, (4,)),
        (False, True, (4, 1)),
        (True, True, (4,)),
        (True, True, (4, 1)),
    ],
)
def test_cuda_tensor_product_modes(dtype, internal, shared, env_shape):
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(42)
    cls = TensorProductConv
    module = cls(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=internal,
        shared_weights=shared,
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(3, 6, device="cuda", dtype=dtype).requires_grad_()
    y = torch.randn(4, 6, device="cuda", dtype=dtype).requires_grad_()
    args = (x, y)
    if not internal:
        shape = (module.weight_numel,) if shared else (4, module.weight_numel)
        args += (torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(),)
    kwargs = {}
    src = torch.tensor([1, 2, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1, 2], device="cuda")
    kwargs = dict(
        graph=module.create_graph(src, dst, 3),
        env=torch.rand(env_shape, device="cuda", dtype=dtype).requires_grad_(),
    )
    assert_cuda_report(module, args, **kwargs)


def test_irreps_constructor_registers_internal_weights():
    module = TensorProductConv(
        "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uvw", True)], dtype=torch.float64
    )
    assert module.internal_weights and module.shared_weights
    assert module.weight.dtype == torch.float64
    assert module.weight is dict(module.named_parameters())["tp.weight"]
    before = module.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(module.weight, before - 0.1)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [3, 4, 12])
def test_external_path_groups_through_triple_backward(dtype, channels):
    """Cover partial warps and complete warps within a partial thread block."""
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(61200 + channels)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            f"{channels}x2e",
            "1x2e",
            "5x2e",
            [(0, 0, 0, "uvw", True)],
            internal_weights=False,
            shared_weights=False,
        ).to(device="cuda", dtype=dtype)
    finally:
        torch.set_default_dtype(previous_dtype)
    rows = 29
    nodes = 7
    x = torch.randn(
        nodes, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        rows, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(
        rows, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True
    )
    src = torch.randint(nodes, (rows,), device="cuda")
    dst = torch.sort(torch.randint(nodes, (rows,), device="cuda"))[0]
    op = TensorProductConv(tp, dtype=dtype).cuda()
    graph = op.create_graph(src, dst, nodes)
    actual = op(x, y, w, graph=graph)
    inputs = (x, y, w)
    grad_out = torch.randn_like(actual, requires_grad=True)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, grad_out)))
    ref_inputs, ref_grad, ref_first, ref_second = fp64(
        (inputs, grad_out, first_seeds, second_seeds)
    )
    ref_tp = deepcopy(tp).double()
    messages = ref_tp(ref_inputs[0][src], ref_inputs[1], ref_inputs[2])
    expected = messages.new_zeros(nodes, tp.irreps_out.dim).index_add(0, dst, messages)
    results = []
    for value, targets, gradient, directions, second_directions in (
        (actual, inputs, grad_out, first_seeds, second_seeds),
        (expected, ref_inputs, ref_grad, ref_first, ref_second),
    ):
        first = torch.autograd.grad(value, targets, gradient, create_graph=True)
        second = torch.autograd.grad(
            first, (*targets, gradient), directions, create_graph=True
        )
        third = torch.autograd.grad(
            second, (*targets, gradient, *directions), second_directions
        )
        results.append((value, *first, *second, *third))
    for actual, expected in zip(*results):
        assert_reference_close(actual, expected)


def make_tp(dtype, shared):
    ir = Irreps("33x0e + 17x1o + 2x3o")
    edge = Irreps("2x0e + 3x1o + 2x2o")
    out = Irreps("19x0e + 11x1o + 5x2e + 3x4o")
    paths = [
        (i, j, k, "uvw", True)
        for i, (_, a) in enumerate(ir[:2])
        for j, (_, b) in enumerate(edge[:2])
        for k, (_, c) in enumerate(out[:3])
        if c in a * b
    ]
    paths.append(paths[0])
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            ir, edge, out, paths, internal_weights=shared, shared_weights=shared
        ).cuda()
    finally:
        torch.set_default_dtype(old)
    return tp


def compare(actual, expected):
    assert_reference_close(actual, expected)


def make_operand(shape, dtype, offset):
    if offset:
        value = torch.randn(math.prod(shape) + 1, device="cuda", dtype=dtype)[
            1:
        ].reshape(shape)
    elif len(shape) == 2:
        value = torch.randn(tuple(reversed(shape)), device="cuda", dtype=dtype).t()
    else:
        value = torch.randn(shape, device="cuda", dtype=dtype)
    return value.requires_grad_()


def derivative_orders(value, inputs, first_seeds, second_seeds):
    g = inputs[3]
    first = torch.autograd.grad(
        value, inputs[:3], g, create_graph=True, retain_graph=True
    )
    second = torch.autograd.grad(
        first, inputs, first_seeds, create_graph=True, retain_graph=True
    )
    third = torch.autograd.grad(
        second, (*inputs, *first_seeds), second_seeds, retain_graph=True
    )
    return ((value,), first, second, third)


def frozen_node_orders(value, x, g, rx, bg):
    first = torch.autograd.grad(value, x, g, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(first, g, rx, create_graph=True)[0]
    third = torch.autograd.grad(second, rx, bg)[0]
    return (value, first, second, third)


def evaluate(op, x, y, w, graph, seed):
    out = op(x, y, w, graph=graph)
    return (out, torch.autograd.grad(out, (x, y, w), seed))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_independent_edge_channels_all_derivatives(dtype, shared, offset):
    torch.manual_seed(2981)
    tp = make_tp(dtype, shared)
    op = TensorProductConv(tp, dtype=dtype)
    src = torch.tensor([0, 2, 1, 2, 0, 1, 1, 2, 0], device="cuda")
    dst = torch.tensor([2, 0, 2, 1, 0, 1, 2, 0, 1], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = op.create_graph(src, dst, 6)
    x = make_operand((6, tp.irreps_in1.dim), dtype, offset)
    y = make_operand((len(src), tp.irreps_in2.dim), dtype, offset)
    w = make_operand(
        (tp.weight_numel,) if shared else (len(src), tp.weight_numel), dtype, offset
    )
    ref_x, ref_y, ref_w = fp64((x, y, w))
    messages = deepcopy(tp).double()(ref_x[src], ref_y, ref_w)
    expected = messages.new_zeros((6, tp.irreps_out.dim)).index_add(0, dst, messages)
    actual = op(x, y, w, graph=graph)
    assert not actual[:, -27:].count_nonzero()
    assert not actual[3:].count_nonzero()
    g = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, g)
    first_seeds = tuple((torch.randn_like(v, requires_grad=True) for v in inputs[:3]))
    second_seeds = tuple((torch.randn_like(v) for v in inputs))
    ref_g, ref_first, ref_second = fp64((g, first_seeds, second_seeds))
    ref_inputs = (ref_x, ref_y, ref_w, ref_g)
    for ref_order, got_order in zip(
        derivative_orders(expected, ref_inputs, ref_first, ref_second),
        derivative_orders(actual, inputs, first_seeds, second_seeds),
    ):
        for ref, got in zip(ref_order, got_order):
            compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_edge_channels_frozen_edge_and_weight(dtype):
    torch.manual_seed(7321)
    tp = make_tp(dtype, shared=True)
    op = TensorProductConv(tp, dtype=dtype)
    src = torch.tensor([0, 2, 1, 0, 1], device="cuda")
    dst = torch.tensor([2, 0, 2, 1, 0], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = op.create_graph(src, dst, 6)
    x = torch.randn(
        6, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(5, tp.irreps_in2.dim, device="cuda", dtype=dtype)
    w = tp.weight.detach()
    ref_x, ref_y, ref_w = fp64((x, y, w))
    messages = deepcopy(tp).double()(ref_x[src], ref_y, ref_w)
    expected = messages.new_zeros((6, tp.irreps_out.dim)).index_add(0, dst, messages)
    actual = op(x, y, w, graph=graph)
    g = torch.randn_like(actual, requires_grad=True)
    rx = torch.randn_like(x, requires_grad=True)
    bg = torch.randn_like(actual)
    ref_g, ref_rx, ref_bg = fp64((g, rx, bg))
    first_ref = torch.autograd.grad(expected, ref_x, ref_g, retain_graph=True)[0]
    first_got = torch.autograd.grad(actual, x, g, retain_graph=True)[0]
    compare(first_got, first_ref)
    for ref, got in zip(
        frozen_node_orders(expected, ref_x, ref_g, ref_rx, ref_bg),
        frozen_node_orders(actual, x, g, rx, bg),
    ):
        compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("num_nodes", [0, 6])
def test_independent_edge_channels_empty_graph(dtype, shared, num_nodes):
    tp = make_tp(dtype, shared)
    op = TensorProductConv(tp, dtype=dtype)
    src = dst = torch.empty(0, device="cuda", dtype=torch.int64)
    graph = op.create_graph(src, dst, num_nodes)
    x = torch.randn(
        num_nodes, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.empty(
        0, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = (
        tp.weight
        if shared
        else torch.empty(
            0, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True
        )
    )
    actual = op(x, y, w, graph=graph)
    assert actual.shape == (num_nodes, tp.irreps_out.dim)
    assert not actual.count_nonzero()
    g = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, g)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    first = torch.autograd.grad(actual, inputs[:3], g, create_graph=True)
    second = torch.autograd.grad(first, inputs, first_seeds, create_graph=True)
    third = torch.autograd.grad(second, (*inputs, *first_seeds), second_seeds)
    assert all((not value.count_nonzero() for value in (*first, *second, *third)))


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False, True])
def test_independent_edge_channels_capture(shared):
    tp = make_tp(torch.float64, shared)
    op = TensorProductConv(tp, dtype=torch.float64)
    src = torch.tensor([0, 1, 2, 1, 0], device="cuda")
    dst = torch.tensor([2, 0, 1, 1, 0], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = op.create_graph(src, dst, 4)
    x = torch.randn(
        4, tp.irreps_in1.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    y = torch.randn(
        5, tp.irreps_in2.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    w = (
        tp.weight
        if shared
        else torch.randn(
            5, tp.weight_numel, device="cuda", dtype=torch.float64, requires_grad=True
        )
    )
    seed = torch.randn(4, tp.irreps_out.dim, device="cuda", dtype=torch.float64)
    expected, gradients = evaluate(op, x, y, w, graph, seed)
    expected = expected.detach()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate(op, x, y, w, graph, seed)
    torch.cuda.current_stream().wait_stream(stream)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        out, got = evaluate(op, x, y, w, graph, seed)
    capture.replay()
    torch.cuda.synchronize()
    compare(out, expected)
    for actual, reference in zip(got, gradients):
        compare(actual, reference)


def make_envelope_operand(shape, dtype, layout):
    if layout == "strided" and len(shape) == 2:
        value = torch.randn((shape[0], shape[1] * 2), device="cuda", dtype=dtype)[
            :, ::2
        ]
    elif layout == "offset":
        size = 1
        for width in shape:
            size *= width
        value = torch.randn(size + 1, device="cuda", dtype=dtype)[1:].reshape(shape)
    else:
        value = torch.randn(shape, device="cuda", dtype=dtype)
    return value.requires_grad_()


def node_only_derivatives(out, x, g, rx, bg):
    first = torch.autograd.grad(out, x, g, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(first, g, rx, create_graph=True)[0]
    third = torch.autograd.grad(second, (x, rx), bg, allow_unused=True)
    return (first, second, third[1])


def feature_derivatives(out, active, directions, cotangents):
    g = active[3]
    first = torch.autograd.grad(out, active[:3], g, create_graph=True)
    second = torch.autograd.grad(first, active, directions[:3], create_graph=True)
    third = torch.autograd.grad(second, (*active, *directions[:3]), cotangents[:4])
    return ((out,), first, second, third)


def scalar_only_derivatives(out, env, g, re, bg):
    first = torch.autograd.grad(out, env, g, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(first, g, re, create_graph=True)[0]
    third = torch.autograd.grad(second, re, bg)[0]
    return (first, second, third)


def make_envelope_problem(
    dtype, shared, *, nodes=5, empty=False, layout="dense", high_degree=False
):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        ir = Irreps("5x2e + 1x0e" if high_degree else "5x0e + 3x1o + 1x3o")
        edge = Irreps("2x2e + 1x0e" if high_degree else "2x0e + 2x1o + 1x3o")
        out = Irreps(
            "3x3e + 2x4e + 1x6e" if high_degree else "4x0e + 3x1o + 2x2e + 1x4o"
        )
        paths = [
            (i, j, k, "uvw", True)
            for i, (_, a) in enumerate(ir[:2])
            for j, (_, b) in enumerate(edge[:2])
            for k, (_, c) in enumerate(out[:3])
            if c in a * b
        ]
        paths.append(paths[0])
        if high_degree:
            paths = [(0, 0, 0, "uvw", True), (0, 0, 1, "uvw", True)]
        tp = E3nnTensorProduct(
            ir, edge, out, paths, internal_weights=shared, shared_weights=shared
        ).cuda()
    finally:
        torch.set_default_dtype(old_dtype)
    op = TensorProductConv(tp, dtype=dtype)
    src = torch.tensor(
        [] if empty else [0, 2, 1, 2, 0, 1, 1, 2, 0], device="cuda", dtype=torch.int64
    )
    dst = torch.tensor(
        [] if empty else [2, 0, 2, 1, 0, 1, 2, 0, 1], device="cuda", dtype=torch.int64
    )
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = op.create_graph(src, dst, nodes)
    x = make_envelope_operand((nodes + 2, ir.dim), dtype, layout)
    y = make_envelope_operand((src.numel(), edge.dim), dtype, layout)
    w = make_envelope_operand(
        (tp.weight_numel,) if shared else (src.numel(), tp.weight_numel), dtype, layout
    )
    env = make_envelope_operand((src.numel(), 1), dtype, layout)
    if env.numel():
        with torch.no_grad():
            env[0] = 0
            env[1] = -1
    g = make_envelope_operand((nodes, out.dim), dtype, layout)
    values = (x, y, w, g, env)
    directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in (x, y, w, env))
    )
    cotangents = tuple((torch.randn_like(v) for v in values))
    return (tp, op, graph, values, directions, cotangents)


def reference_envelope(tp, graph, values):
    x, y, w, _, env = values
    messages = tp(x[graph.src], y, w) * env
    return x.new_zeros((graph.num_nodes, tp.irreps_out.dim)).index_add(
        0, graph.dst, messages
    )


def envelope_chain(forward, values, directions, cotangents):
    x, y, w, g, env = values
    out = forward()
    first = torch.autograd.grad(out, (x, y, w, env), g, create_graph=True)
    second = torch.autograd.grad(first, values, directions, create_graph=True)
    third = torch.autograd.grad(second, (*values, *directions), cotangents)
    return ((out,), first, second, third)


def compare_envelope(actual, expected):
    if actual is None or expected is None:
        remaining = expected if actual is None else actual
        assert remaining is None or not remaining.count_nonzero()
        return
    assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("layout", ["dense", "strided", "offset"])
def test_envelope_all_inputs_through_third_order(dtype, shared, layout):
    torch.manual_seed(7423)
    tp, op, graph, values, directions, cotangents = make_envelope_problem(
        dtype, shared, layout=layout
    )
    ref_values, ref_directions, ref_cotangents = fp64((values, directions, cotangents))
    ref_tp = deepcopy(tp).double()
    expected = envelope_chain(
        lambda: reference_envelope(ref_tp, graph, ref_values),
        ref_values,
        ref_directions,
        ref_cotangents,
    )
    actual = envelope_chain(
        lambda: op(values[0], values[1], values[2], graph=graph, env=values[4]),
        values,
        directions,
        cotangents,
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare_envelope(got, ref)
    assert not actual[0][0][:, -9:].count_nonzero()
    assert not actual[1][0][3:].count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_frozen_inputs_and_missing_directions(dtype, shared):
    torch.manual_seed(5327)
    tp, op, graph, values, directions, _ = make_envelope_problem(dtype, shared)
    x, y, w, g, env = values
    y, w, env = (y.detach(), w.detach(), env.detach())
    values = (x, y, w, g, env)
    rx = directions[0]
    bg = torch.randn_like(g)
    ref_values, ref_rx, ref_bg = fp64((values, rx, bg))
    expected = reference_envelope(deepcopy(tp).double(), graph, ref_values)
    actual = op(x, y, w, graph=graph, env=env)
    compare_envelope(actual, expected)
    compare_envelope(
        torch.autograd.grad(actual, x, g, retain_graph=True)[0],
        torch.autograd.grad(expected, ref_values[0], ref_values[3], retain_graph=True)[
            0
        ],
    )
    for got, ref in zip(
        node_only_derivatives(actual, x, g, rx, bg),
        node_only_derivatives(expected, ref_values[0], ref_values[3], ref_rx, ref_bg),
    ):
        compare_envelope(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_high_degree_staged_external_and_grouped_shared(dtype, shared):
    torch.manual_seed(3187)
    tp, op, graph, values, directions, cotangents = make_envelope_problem(
        dtype, shared, high_degree=True
    )
    ref_values, ref_directions, ref_cotangents = fp64((values, directions, cotangents))
    ref_tp = deepcopy(tp).double()
    expected = envelope_chain(
        lambda: reference_envelope(ref_tp, graph, ref_values),
        ref_values,
        ref_directions,
        ref_cotangents,
    )
    actual = envelope_chain(
        lambda: op(values[0], values[1], values[2], graph=graph, env=values[4]),
        values,
        directions,
        cotangents,
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare_envelope(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("nodes", [0, 5])
def test_envelope_empty_graph_all_orders(dtype, shared, nodes):
    tp, op, graph, values, directions, cotangents = make_envelope_problem(
        dtype, shared, nodes=nodes, empty=True
    )
    actual = envelope_chain(
        lambda: op(values[0], values[1], values[2], graph=graph, env=values[4]),
        values,
        directions,
        cotangents,
    )
    for stage in actual:
        for value in stage:
            assert not value.count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_frozen_scalar_full_feature_chain(dtype, shared):
    torch.manual_seed(4587)
    tp, op, graph, values, directions, cotangents = make_envelope_problem(dtype, shared)
    x, y, w, g, env = values
    env = env.detach()
    active = (x, y, w, g)
    ref_active, ref_env, ref_directions, ref_cotangents = fp64(
        (active, env, directions, cotangents)
    )
    expected = feature_derivatives(
        reference_envelope(deepcopy(tp).double(), graph, (*ref_active, ref_env)),
        ref_active,
        ref_directions,
        ref_cotangents,
    )
    actual = feature_derivatives(
        op(x, y, w, graph=graph, env=env), active, directions, cotangents
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare_envelope(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_only_scalar_trainable(dtype, shared):
    torch.manual_seed(4329)
    tp, op, graph, values, directions, _ = make_envelope_problem(dtype, shared)
    x, y, w, g, env = values
    x, y, w = (x.detach(), y.detach(), w.detach())
    re = directions[3]
    bg = torch.randn_like(g)
    ref_values, ref_re, ref_bg = fp64(((x, y, w, g, env), re, bg))
    expected = reference_envelope(deepcopy(tp).double(), graph, ref_values)
    actual = op(x, y, w, graph=graph, env=env)
    compare_envelope(
        torch.autograd.grad(actual, env, g, retain_graph=True)[0],
        torch.autograd.grad(expected, ref_values[4], ref_values[3], retain_graph=True)[
            0
        ],
    )
    for got, ref in zip(
        scalar_only_derivatives(actual, env, g, re, bg),
        scalar_only_derivatives(expected, ref_values[4], ref_values[3], ref_re, ref_bg),
    ):
        compare_envelope(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_caller_stream_and_capture_all_orders(dtype, shared):
    torch.manual_seed(6151)
    tp, op, graph, values, directions, cotangents = make_envelope_problem(dtype, shared)
    ref_values, ref_directions, ref_cotangents = fp64((values, directions, cotangents))
    ref_tp = deepcopy(tp).double()
    raw_expected = envelope_chain(
        lambda: reference_envelope(ref_tp, graph, ref_values),
        ref_values,
        ref_directions,
        ref_cotangents,
    )
    expected = tuple((tuple((v.detach() for v in stage)) for stage in raw_expected))
    del raw_expected
    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        local_values = tuple((v.detach().clone().requires_grad_(True) for v in values))
        local_directions = tuple(
            (v.detach().clone().requires_grad_(True) for v in directions)
        )
        local_cotangents = tuple((v.detach().clone() for v in cotangents))
        execute = partial(
            envelope_chain,
            partial(op, *local_values[:3], graph=graph, env=local_values[4]),
            local_values,
            local_directions,
            local_cotangents,
        )
        streamed = execute()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            captured = execute()
    caller.wait_stream(stream)
    capture.replay()
    torch.cuda.synchronize()
    for actual in (streamed, captured):
        for got_stage, ref_stage in zip(actual, expected):
            for got, ref in zip(got_stage, ref_stage):
                compare_envelope(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_envelope_internal_parameter_through_third_order(dtype):
    torch.manual_seed(4735)
    tp, op, graph, values, directions, cotangents = make_envelope_problem(dtype, True)
    x, y, _, g, env = values
    values = (x, y, tp.weight, g, env)
    assert op.weight is tp.weight
    ref_values, ref_directions, ref_cotangents = fp64((values, directions, cotangents))
    ref_tp = deepcopy(tp).double()
    expected = envelope_chain(
        lambda: reference_envelope(ref_tp, graph, ref_values),
        ref_values,
        ref_directions,
        ref_cotangents,
    )
    actual = envelope_chain(
        lambda: op(x, y, graph=graph, env=env), values, directions, cotangents
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare_envelope(got, ref)


@pytest.mark.parametrize(
    "dtype,channels", ((torch.float32, 2048), (torch.float64, 1024))
)
def test_wide_uvw_conv_construction(dtype, channels):
    config = prepare_tensor_product(
        "4x0e",
        f"{channels}x2e",
        "4x2e",
        [(0, 0, 0, "uvw", True)],
        dtype=dtype,
        convolution=True,
    )
    config.prepare_indexing()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,channels", ((torch.float32, 2048), (torch.float64, 1024))
)
def test_wide_uvw_conv_forward(dtype, channels):
    torch.manual_seed(1043)
    reference = o3.TensorProduct(
        "4x0e",
        f"{channels}x2e",
        "4x2e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device="cuda", dtype=dtype)
    module = TensorProductConv(reference, dtype=dtype)
    src = torch.tensor([1, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1], device="cuda")
    graph = make_graph(src, dst, 2)
    x = torch.randn(2, 4, device="cuda", dtype=dtype)
    y = torch.randn(3, 5 * channels, device="cuda", dtype=dtype)
    weight = torch.randn(3, reference.weight_numel, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual = module(x, y, weight, graph=graph)
        messages = deepcopy(reference).double()(
            x[src].double(), y.double(), weight.double()
        )
        expected = messages.new_zeros(2, 20).index_add(0, dst, messages)
    assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_graph_forward_and_three_orders(dtype):
    torch.manual_seed(1042)
    src = torch.tensor([0, 2, 1, 2, 0, 1, 1], device="cuda")
    dst = torch.tensor([2, 0, 2, 1, 0, 1, 2], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = make_graph(src, dst, 4)
    irreps = "32x0e + 32x1o"
    edge_irreps = "1x0e + 1x1o"
    ir, edge = (o3.Irreps(irreps), o3.Irreps(edge_irreps))
    instructions = [
        (i, j, k, "uvw", True)
        for i, (_, a) in enumerate(ir)
        for j, (_, b) in enumerate(edge)
        for k, (_, c) in enumerate(ir)
        if c in a * b
    ]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        reference = o3.TensorProduct(
            ir, edge, ir, instructions, internal_weights=False, shared_weights=False
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    module = TensorProductConv(reference)
    x = torch.randn(4, ir.dim, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(len(src), edge.dim, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(
        len(src), reference.weight_numel, device="cuda", dtype=dtype, requires_grad=True
    )
    actual = module(x, y, weight, graph=graph)
    config = prepare_tensor_product(
        ir, edge, ir, instructions, dtype=dtype, convolution=True
    )
    functional = tensor_product_conv(x, y, weight, config=config, graph=graph)
    grad = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, weight, grad)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    ref_inputs, ref_first, ref_second = fp64((inputs, first_seeds, second_seeds))
    messages = deepcopy(reference).double()(
        ref_inputs[0][src], ref_inputs[1], ref_inputs[2]
    )
    expected = messages.new_zeros(4, ir.dim).index_add(0, dst, messages)
    assert_reference_close(actual, expected)
    assert_reference_close(functional, expected)
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
@pytest.mark.parametrize("env_shape", [(4,), (4, 1)])
def test_tensor_product_conv_envelope_compile(env_shape):
    from LunaEquivariance import TensorProductConv

    torch.manual_seed(42)
    module = TensorProductConv(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device="cuda", dtype=torch.float64)
    graph = module.create_graph(
        torch.tensor([1, 2, 0, 1], device="cuda"),
        torch.tensor([0, 0, 1, 2], device="cuda"),
        3,
    )
    args = tuple(
        (
            torch.randn(shape, device="cuda", dtype=torch.float64).requires_grad_()
            for shape in ((3, 6), (4, 6), (4, module.weight_numel), env_shape)
        )
    )

    def forward(x, edge_attr, weight, env):
        return module(x, edge_attr, weight, graph=graph, env=env)

    expected = forward(*args)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    actual = compiled(*args)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.sin().sum(), args)
    actual_grads = torch.autograd.grad(actual.sin().sum(), args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grads[-1].shape == env_shape


def make_case(module, *, with_env=True, strided=False, graph=None):
    if graph is None:
        graph = make_graph(
            torch.tensor([2, 0, 2, 1, 0], device="cuda"),
            torch.tensor([0, 1, 1, 2, 2], device="cuda"),
            3,
        )
    values = []
    shapes = [
        (3, module.irreps_in1.dim),
        (5, module.irreps_in2.dim),
        (5, module.weight_numel),
    ] + ([(5, 1)] if with_env else [])
    for i, (rows, width) in enumerate(shapes):
        columns = width * (2 if strided else 1)
        value = torch.linspace(
            0.1 + i * 0.07,
            0.8 + i * 0.09,
            rows * columns,
            device="cuda",
            dtype=torch.float64,
        ).reshape(rows, columns)
        if strided:
            value = value[:, ::2]
        values.append(value.requires_grad_())
    if not with_env:
        values.append(None)
    return (graph, tuple(values))


def compile_case(module, graph, values, order, *, return_l0=False, unused_weight=False):
    if unused_weight:
        values = (*values[:2], values[2].detach(), values[3])
    shape = (5, module.op.l0_dim) if return_l0 else (3, module.irreps_out.dim)
    seed = torch.linspace(
        0.2, 0.6, shape[0] * shape[1], device="cuda", dtype=torch.float64
    ).reshape(shape)

    def function(x, y, weight, env, seed):
        kwargs = {"return_l0_feat": True} if return_l0 else {}
        result = module(x, y, weight, graph=graph, env=env, **kwargs)
        outputs = (result[1] if return_l0 else result,)
        targets = tuple(
            (
                value
                for value in (x, y, weight, env)
                if value is not None and value.requires_grad
            )
        )
        directions = (seed,)
        for step in range(order):
            gradients = torch.autograd.grad(
                outputs,
                targets,
                directions,
                create_graph=step + 1 < order,
                retain_graph=True,
                allow_unused=True,
            )
            outputs = tuple((value for value in gradients if value is not None))
            if step + 1 < order:
                outputs = tuple((value for value in outputs if value.requires_grad))
                directions = tuple((torch.ones_like(value) for value in outputs))
        return outputs

    args = (*values, seed)
    expected = function(*args)
    traced = make_fx(function, tracing_mode="fake", _allow_non_fake_inputs=True)(*args)
    traced.graph.eliminate_dead_code()
    traced.recompile()
    compiled = torch.compile(
        traced, fullgraph=True, options={"triton.cudagraphs": False}
    )
    with torch.no_grad():
        actual = compiled(*args)
    torch.cuda.synchronize()
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=3e-11, atol=3e-12)
    return (compiled, args, expected)


@pytest.fixture(scope="module")
def uvw_convolution():
    return TensorProductConv(
        "2x0e + 2x1o",
        "1x0e + 1x1o",
        "2x0e + 2x1o",
        [
            (0, 0, 0, "uvw", True),
            (1, 1, 0, "uvw", True),
            (0, 1, 1, "uvw", True),
            (1, 0, 1, "uvw", True),
        ],
        internal_weights=False,
        shared_weights=False,
        dtype=torch.float64,
    ).cuda()


@pytest.mark.cuda
@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_uvw_compiled_derivatives(uvw_convolution, order):
    graph, values = make_case(uvw_convolution, with_env=order >= 2)
    compile_case(uvw_convolution, graph, values, order)


def compilation_case(channels):
    irreps = lunaeq.Irreps(f"{channels}x0e + {channels}x1o")
    edge_irreps = lunaeq.Irreps("2x0e + 2x1o")
    instructions = [(i, j, k, "uvw", True)
                    for i, (_, a) in enumerate(irreps)
                    for j, (_, b) in enumerate(edge_irreps)
                    for k, (_, c) in enumerate(irreps) if c in a * b]
    module = lunaeq.TensorProductConv(irreps, edge_irreps, irreps, instructions,
                          internal_weights=False, shared_weights=False).cuda()
    weight_numel = module.weight_numel
    module = ConvolutionInputs(module)
    samples = []
    for rows, edges in ((7, 17), (19, 41)):
        x = (torch.randn(rows, irreps.dim, device="cuda") * 0.3).requires_grad_()
        edge = (torch.randn(edges, edge_irreps.dim, device="cuda") * 0.3).requires_grad_()
        weight = (torch.randn(edges, weight_numel, device="cuda") * 0.3).requires_grad_()
        src = (torch.arange(edges, device="cuda") * 3 + 1) % rows
        dst = (torch.arange(edges, device="cuda") % rows).sort().values
        graph = lunaeq.make_graph(src, dst, rows)
        samples.append(((x, edge, weight, graph_tensors(graph)), {}))
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
