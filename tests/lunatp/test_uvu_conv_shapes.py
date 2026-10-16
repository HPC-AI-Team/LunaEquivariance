"""e3nn weight-layout and derivative contracts for nonunit edge multiplicity."""

from math import prod
import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProductConv


def make_shape_problem(channels, dtype, *, device="cuda", empty=False):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            f"{channels}x0e + {channels}x1o + 1x2e",
            "2x0e + 3x1o + 1x2e",
            f"{channels}x0e + {channels}x1o + 1x2e",
            [
                (0, 0, 0, "uvu", True),
                (0, 0, 0, "uvu", True),
                (1, 1, 0, "uvu", True),
                (0, 1, 1, "uvu", True),
                (1, 0, 1, "uvu", True),
            ],
            internal_weights=False,
            shared_weights=False,
        ).to(device)
    finally:
        torch.set_default_dtype(previous)
    torch.manual_seed(72619)
    nodes, edges = (5, 0 if empty else 9)
    src = torch.tensor(
        [1, 0, 1, 2, 0, 1, 2, 0, 1][:edges], device=device, dtype=torch.int64
    )
    dst = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2][:edges], device=device, dtype=torch.int64
    )
    x = (
        torch.randn(tp.irreps_in1.dim, nodes, device=device, dtype=dtype)
        .t()
        .requires_grad_()
    )
    y = (
        torch.randn(tp.irreps_in2.dim, edges, device=device, dtype=dtype)
        .t()
        .requires_grad_()
    )
    w = (
        torch.randn(edges * tp.weight_numel + 1, device=device, dtype=dtype)[1:]
        .view(edges, tp.weight_numel)
        .requires_grad_()
    )
    return (tp, x, y, w, src, dst)


def reference_shape_outputs(tp, x, y, w, src, dst, env=None, l0=False):
    edge = tp(x[src], y, w)
    out = x.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, edge if env is None else edge * env
    )
    if not l0:
        return (out,)
    paths, offset = ([], 0)
    for instruction in tp.instructions:
        width = prod(instruction.path_shape)
        if tp.irreps_out[instruction.i_out].ir.l == 0:
            mask = torch.zeros_like(w)
            mask[:, offset : offset + width] = 1
            paths.append(
                (
                    instruction.i_out,
                    tp(x[src], y, w * mask)[
                        :, tp.irreps_out.slices()[instruction.i_out]
                    ],
                )
            )
        offset += width
    scalar = (
        torch.cat(
            [value for _, value in sorted(paths, key=lambda item: item[0])], dim=1
        )
        if paths
        else edge[:, :0]
    )
    return (out, scalar)


def assert_shape_close(actual, expected, dtype):
    if actual is None:
        assert expected is None or bool((expected == 0).all())
        return
    if expected is None:
        assert bool((actual == 0).all())
        return
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-11 if dtype == torch.float64 else 0.0007,
        atol=1e-11 if dtype == torch.float64 else 0.0002,
    )
    if actual.numel():
        relative = (
            actual - expected
        ).double().norm() / expected.double().norm().clamp_min(1e-20)
        assert relative.item() < (1e-11 if dtype == torch.float64 else 2e-05)


def derivative_chain(outputs, chosen, inputs, seeds, directions, second_seeds):
    first = torch.autograd.grad(
        outputs, chosen, seeds, create_graph=True, retain_graph=True, allow_unused=True
    )
    active_first = [
        (value, direction)
        for value, direction in zip(first, directions)
        if value is not None and value.requires_grad
    ]
    second_inputs = (*inputs, *seeds)
    second = torch.autograd.grad(
        [v for v, _ in active_first],
        second_inputs,
        [r for _, r in active_first],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    active_second = [
        (value, seed)
        for value, seed in zip(second, second_seeds)
        if value is not None and value.requires_grad
    ]
    third = torch.autograd.grad(
        [v for v, _ in active_second],
        (*second_inputs, *directions),
        [s for _, s in active_second],
        retain_graph=True,
        allow_unused=True,
    )
    return (outputs, first, second, third)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 33, 1025])
@pytest.mark.parametrize("feature", ["ordinary", "envelope", "l0", "combined"])
def test_nonunit_edge_complete_derivatives(dtype, channels, feature):
    tp, x, y, w, src, dst = make_shape_problem(channels, dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = (
        torch.randn(len(y), 1, device=x.device, dtype=dtype, requires_grad=True)
        if feature in ("envelope", "combined")
        else None
    )
    l0 = feature in ("l0", "combined")
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=l0)
    actual = actual if l0 else (actual,)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, env, l0)
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_mixed_width_nonunit_edge_derivatives(dtype):
    """Large unequal groups must not exceed the module constant-memory budget."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "65537x0e + 65601x1o + 1x2e",
            "2x0e + 3x1o + 1x2e",
            "65537x0e + 65601x0e + 65537x1o + 65601x1o + 1x2e",
            [
                (0, 0, 0, "uvu", True),
                (0, 0, 0, "uvu", True),
                (1, 1, 1, "uvu", True),
                (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True),
            ],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    torch.manual_seed(92671)
    src = torch.tensor([1, 0, 1, 0], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    x = (
        torch.randn(tp.irreps_in1.dim, 3, dtype=dtype, device="cuda")
        .t()
        .requires_grad_()
    )
    y = (
        torch.randn(tp.irreps_in2.dim, 4, dtype=dtype, device="cuda")
        .t()
        .requires_grad_()
    )
    w = (
        torch.randn(4 * tp.weight_numel + 1, dtype=dtype, device="cuda")[1:]
        .view(4, -1)
        .requires_grad_()
    )
    env = torch.tensor(
        [[0.0], [0.5], [-1.0], [0.0]], dtype=dtype, device="cuda", requires_grad=True
    )
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
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
