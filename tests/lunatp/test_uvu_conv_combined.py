"""Complete combined envelope/raw-L0 derivatives and unused-output contracts."""

import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProductConv
from tests.lunatp.test_uvu_conv_shapes import (
    assert_shape_close,
    derivative_chain,
    make_shape_problem,
    reference_shape_outputs,
)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("first_input", ["all", "edge", "weight", "envelope"])
@pytest.mark.parametrize("output", ["both", "node", "l0"])
def test_combined_mixed_derivatives(dtype, first_input, output):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype)
    env[::2] = 0
    env.requires_grad_()
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, True)
    assert actual[1].shape == (len(y), 3 * 17)
    if output != "both":
        selection = slice(0, 1) if output == "node" else slice(1, 2)
        actual, expected = (actual[selection], expected[selection])
    inputs = (x, y, w, env)
    chosen = {"all": inputs, "edge": (y,), "weight": (w,), "envelope": (env,)}[
        first_input
    ]
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in chosen)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    if output == "l0" and first_input == "envelope":
        gradient = torch.autograd.grad(actual, env, seeds, allow_unused=True)[0]
        assert gradient is None or bool((gradient == 0).all())
        return
    for actual_order, expected_order in zip(
        derivative_chain(actual, chosen, inputs, seeds, directions, second_seeds),
        derivative_chain(expected, chosen, inputs, seeds, directions, second_seeds),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_combined_zero_scalar_columns(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "5x1o",
            "2x0e",
            "5x1o",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    torch.manual_seed(92381)
    src = torch.tensor([1, 0, 1, 0], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    x = torch.randn(
        3, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        4, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(4, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True)
    env = torch.randn(4, 1, device="cuda", dtype=dtype, requires_grad=True)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    assert actual[1].shape == (len(y), 0)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, True)
    inputs = (x, y, w, env)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(expected, inputs, inputs, seeds, directions, second_seeds),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_combined_frozen_envelope_and_empty_edges(dtype, empty):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype, empty=empty)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype)
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, True)
    inputs = (x, y, w)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(expected, inputs, inputs, seeds, directions, second_seeds),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_combined_nondefault_stream_and_capture(dtype):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype, requires_grad=True)
    inputs = (x, y, w, env)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, True)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in expected))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    expected_chain = derivative_chain(
        expected, inputs, inputs, seeds, directions, second_seeds
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x, y, w, env = (value.detach().clone().requires_grad_() for value in inputs)
        inputs = (x, y, w, env)
        seeds = tuple((value.detach().clone().requires_grad_() for value in seeds))
        directions = tuple(
            (value.detach().clone().requires_grad_() for value in directions)
        )
        second_seeds = tuple((value.clone() for value in second_seeds))
        for _ in range(3):
            actual = derivative_chain(
                op(x, y, w, graph=graph, env=env, return_l0_feat=True),
                inputs,
                inputs,
                seeds,
                directions,
                second_seeds,
            )
    torch.cuda.current_stream().wait_stream(stream)
    for actual_order, expected_order in zip(actual, expected_chain):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        captured = derivative_chain(
            op(x, y, w, graph=graph, env=env, return_l0_feat=True),
            inputs,
            inputs,
            seeds,
            directions,
            second_seeds,
        )
    for _ in range(2):
        capture.replay()
    torch.cuda.synchronize()
    for actual_order, expected_order in zip(captured, expected_chain):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("feature", ["l0", "combined"])
def test_high_irrep_nonunit_edge_derivatives(dtype, feature):
    """Exercise the existing staged FP64 factor schedule with nonunit copies."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "17x5o",
            "2x5o",
            "17x0e",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    x = torch.randn(
        3, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        4, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(4, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True)
    src = torch.tensor([1, 0, 2, 0], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    env = (
        torch.randn(4, 1, device="cuda", dtype=dtype, requires_grad=True)
        if feature == "combined"
        else None
    )
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, True)
    inputs = (x, y, w) if env is None else (x, y, w, env)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(expected, inputs, inputs, seeds, directions, second_seeds),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)
