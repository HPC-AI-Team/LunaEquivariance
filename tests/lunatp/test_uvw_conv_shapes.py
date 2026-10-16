"""Independent edge channels in every fused convolution derivative layout."""

import math
import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProductConv


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
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert bool(torch.isfinite(actual).all())
    error = (actual - expected).double().norm() / expected.double().norm().clamp_min(
        1e-25
    )
    assert error.item() < (1e-11 if actual.dtype == torch.float64 else 2e-05)


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
    expected = x.new_zeros((6, tp.irreps_out.dim)).index_add(0, dst, tp(x[src], y, w))
    actual = op(x, y, w, graph=graph)
    assert not actual[:, -27:].count_nonzero()
    assert not actual[3:].count_nonzero()
    g = torch.randn_like(expected, requires_grad=True)
    inputs = (x, y, w, g)
    first_seeds = tuple((torch.randn_like(v, requires_grad=True) for v in inputs[:3]))
    second_seeds = tuple((torch.randn_like(v) for v in inputs))
    for ref_order, got_order in zip(
        derivative_orders(expected, inputs, first_seeds, second_seeds),
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
    expected = x.new_zeros((6, tp.irreps_out.dim)).index_add(0, dst, tp(x[src], y, w))
    actual = op(x, y, w, graph=graph)
    g = torch.randn_like(expected, requires_grad=True)
    rx = torch.randn_like(x, requires_grad=True)
    bg = torch.randn_like(expected)
    first_ref = torch.autograd.grad(expected, x, g, retain_graph=True)[0]
    first_got = torch.autograd.grad(actual, x, g, retain_graph=True)[0]
    compare(first_got, first_ref)
    for ref, got in zip(
        frozen_node_orders(expected, x, g, rx, bg),
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
