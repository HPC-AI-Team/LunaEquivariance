"""UVU channel tails, wide irreps and multiple edge channels."""

import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProduct

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


def assert_close(actual, expected, dtype):
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert bool(torch.isfinite(actual).all())
    diff = (actual - expected).double().norm()
    scale = expected.double().norm().clamp_min(1e-20)
    assert (diff / scale).item() < (1e-11 if dtype is torch.float64 else 2e-05)
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
    expected = derivative_chain(
        tp(x, y, w), inputs, first_directions, second_directions
    )
    actual = derivative_chain(
        op(x, y, None if shared else w), inputs, first_directions, second_directions
    )
    for actual_order, reference_order in zip(actual, expected):
        for result, reference in zip(actual_order, reference_order):
            assert_close(result, reference, dtype)
    force = op(x, y.detach(), None if shared else w.detach())
    assert_close(torch.autograd.grad(force, x, g)[0], expected[1][0], dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        streamed = op(x, y, None if shared else w)
    torch.cuda.current_stream().wait_stream(stream)
    assert_close(streamed, expected[0][0], dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        captured = op(x, y, None if shared else w)
    graph.replay()
    torch.cuda.synchronize()
    assert_close(captured, expected[0][0], dtype)
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
