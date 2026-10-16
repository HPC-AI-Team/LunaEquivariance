"""Optional edge-envelope contracts and full multilinear derivative recipes."""

from functools import partial
import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProductConv


def make_operand(shape, dtype, layout):
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


def make_problem(
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
    x = make_operand((nodes + 2, ir.dim), dtype, layout)
    y = make_operand((src.numel(), edge.dim), dtype, layout)
    w = make_operand(
        (tp.weight_numel,) if shared else (src.numel(), tp.weight_numel), dtype, layout
    )
    env = make_operand((src.numel(), 1), dtype, layout)
    if env.numel():
        with torch.no_grad():
            env[0] = 0
            env[1] = -1
    g = make_operand((nodes, out.dim), dtype, layout)
    values = (x, y, w, g, env)
    directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in (x, y, w, env))
    )
    cotangents = tuple((torch.randn_like(v) for v in values))
    return (tp, op, graph, values, directions, cotangents)


def reference(tp, graph, values):
    x, y, w, _, env = values
    messages = tp(x[graph.src], y, w) * env
    return x.new_zeros((graph.num_nodes, tp.irreps_out.dim)).index_add(
        0, graph.dst, messages
    )


def chain(forward, values, directions, cotangents):
    x, y, w, g, env = values
    out = forward()
    first = torch.autograd.grad(out, (x, y, w, env), g, create_graph=True)
    second = torch.autograd.grad(first, values, directions, create_graph=True)
    third = torch.autograd.grad(second, (*values, *directions), cotangents)
    return ((out,), first, second, third)


def compare(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert bool(torch.isfinite(actual).all())
    difference = (actual - expected).double()
    relative = (difference.norm() / expected.double().norm().clamp_min(1e-25)).item()
    threshold = 1e-11 if actual.dtype == torch.float64 else 2e-05
    assert relative < threshold


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("layout", ["dense", "strided", "offset"])
def test_envelope_all_inputs_through_third_order(dtype, shared, layout):
    torch.manual_seed(7423)
    tp, op, graph, values, directions, cotangents = make_problem(
        dtype, shared, layout=layout
    )
    expected = chain(
        lambda: reference(tp, graph, values), values, directions, cotangents
    )
    actual = chain(
        lambda: op(values[0], values[1], values[2], graph=graph, env=values[4]),
        values,
        directions,
        cotangents,
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare(got, ref)
    assert not actual[0][0][:, -9:].count_nonzero()
    assert not actual[1][0][3:].count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_frozen_inputs_and_missing_directions(dtype, shared):
    torch.manual_seed(5327)
    tp, op, graph, values, directions, _ = make_problem(dtype, shared)
    x, y, w, g, env = values
    y, w, env = (y.detach(), w.detach(), env.detach())
    values = (x, y, w, g, env)
    rx = directions[0]
    bg = torch.randn_like(g)
    expected = reference(tp, graph, values)
    actual = op(x, y, w, graph=graph, env=env)
    compare(actual, expected)
    compare(
        torch.autograd.grad(actual, x, g, retain_graph=True)[0],
        torch.autograd.grad(expected, x, g, retain_graph=True)[0],
    )
    for got, ref in zip(
        node_only_derivatives(actual, x, g, rx, bg),
        node_only_derivatives(expected, x, g, rx, bg),
    ):
        compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_high_degree_staged_external_and_grouped_shared(dtype, shared):
    torch.manual_seed(3187)
    tp, op, graph, values, directions, cotangents = make_problem(
        dtype, shared, high_degree=True
    )
    expected = chain(
        lambda: reference(tp, graph, values), values, directions, cotangents
    )
    actual = chain(
        lambda: op(values[0], values[1], values[2], graph=graph, env=values[4]),
        values,
        directions,
        cotangents,
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("nodes", [0, 5])
def test_envelope_empty_graph_all_orders(dtype, shared, nodes):
    tp, op, graph, values, directions, cotangents = make_problem(
        dtype, shared, nodes=nodes, empty=True
    )
    actual = chain(
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
    tp, op, graph, values, directions, cotangents = make_problem(dtype, shared)
    x, y, w, g, env = values
    env = env.detach()
    active = (x, y, w, g)
    expected = feature_derivatives(
        reference(tp, graph, (*active, env)), active, directions, cotangents
    )
    actual = feature_derivatives(
        op(x, y, w, graph=graph, env=env), active, directions, cotangents
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_only_scalar_trainable(dtype, shared):
    torch.manual_seed(4329)
    tp, op, graph, values, directions, _ = make_problem(dtype, shared)
    x, y, w, g, env = values
    x, y, w = (x.detach(), y.detach(), w.detach())
    re = directions[3]
    bg = torch.randn_like(g)
    expected = reference(tp, graph, (x, y, w, g, env))
    actual = op(x, y, w, graph=graph, env=env)
    compare(
        torch.autograd.grad(actual, env, g, retain_graph=True)[0],
        torch.autograd.grad(expected, env, g, retain_graph=True)[0],
    )
    for got, ref in zip(
        scalar_only_derivatives(actual, env, g, re, bg),
        scalar_only_derivatives(expected, env, g, re, bg),
    ):
        compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_envelope_caller_stream_and_capture_all_orders(dtype, shared):
    torch.manual_seed(6151)
    tp, op, graph, values, directions, cotangents = make_problem(dtype, shared)
    raw_expected = chain(
        lambda: reference(tp, graph, values), values, directions, cotangents
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
            chain,
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
                compare(got, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_envelope_internal_parameter_through_third_order(dtype):
    torch.manual_seed(4735)
    tp, op, graph, values, directions, cotangents = make_problem(dtype, True)
    x, y, _, g, env = values
    values = (x, y, tp.weight, g, env)
    assert op.weight is tp.weight
    expected = chain(
        lambda: reference(tp, graph, values), values, directions, cotangents
    )
    actual = chain(
        lambda: op(x, y, graph=graph, env=env), values, directions, cotangents
    )
    for got_stage, ref_stage in zip(actual, expected):
        for got, ref in zip(got_stage, ref_stage):
            compare(got, ref)
