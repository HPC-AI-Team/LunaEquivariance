"""Complete optional derivatives, including edge/weight-first mixed chains."""

import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProductConv
from tests.lunatp.test_uvu_conv_shapes import reference_shape_outputs


@pytest.fixture(params=[torch.float32, torch.float64])
def optional_dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def assert_optional_close(actual, expected, **tolerances):
    torch.testing.assert_close(actual, expected, **tolerances)
    if actual.numel():
        error = (actual - expected).double().norm()
        scale = expected.double().norm().clamp_min(1e-20)
        limit = 1e-11 if actual.dtype == torch.float64 else 2e-05
        assert (error / scale).item() < limit


def optional_derivative_orders(
    chosen, seeds, second_inputs, directions, second_seeds, outputs
):
    first = torch.autograd.grad(
        outputs, chosen, seeds, create_graph=True, retain_graph=True
    )
    second = torch.autograd.grad(
        first,
        second_inputs,
        directions,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    active = [
        (v, s)
        for v, s in zip(second, second_seeds)
        if v is not None and v.requires_grad
    ]
    third = torch.autograd.grad(
        [v for v, _ in active],
        (*second_inputs, *directions),
        [s for _, s in active],
        allow_unused=True,
        retain_graph=True,
    )
    return (first, second, third)


def force_loss_derivatives(positions, parameters, outputs):
    energy = sum((v.square().sum() for v in outputs))
    force = -torch.autograd.grad(
        energy, positions, create_graph=True, retain_graph=True
    )[0]
    grads = torch.autograd.grad(
        force.square().mean(), (positions, parameters), retain_graph=True
    )
    return (force, *grads)


def make_wide_problem(mode, shared, dtype):
    tp = E3nnTensorProduct(
        "1057x0e",
        "1x0e",
        "1057x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device="cuda", dtype=dtype)
    n, e = (5, 9)
    x = torch.randn(n, 1057, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(e, 1, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(e, 1057, device="cuda", dtype=dtype, requires_grad=True)
    src = torch.randint(n, (e,), device="cuda")
    dst = torch.sort(torch.randint(n, (e,), device="cuda"))[0]
    return (tp, x, y, w, src, dst)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "feature,dtype",
    [
        ("envelope", torch.float32),
        ("envelope", torch.float64),
        ("l0", torch.float32),
        ("l0", torch.float64),
    ],
)
@pytest.mark.parametrize("first_inputs", ["all", "edge", "weight", "env_or_l0"])
def test_complete_optional_derivatives(
    make_problem, feature, dtype, first_inputs, frozen_env=False
):
    torch.manual_seed(831)
    tp, x, y, w, src, dst = make_problem("uvu_conv", False, dtype=dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(
        (len(y), 1), device="cuda", dtype=dtype, requires_grad=not frozen_env
    )
    l0 = feature == "l0"
    actual = op(
        x, y, w, graph=graph, **{"return_l0_feat": True} if l0 else {"env": env}
    )
    actual = actual if l0 else (actual,)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, None if l0 else env, l0)
    inputs = (x, y, w) if l0 or frozen_env else (x, y, w, env)
    seeds = tuple((torch.randn_like(v, requires_grad=True) for v in actual))
    if first_inputs == "env_or_l0" and l0:
        actual, expected, seeds = (actual[1:], expected[1:], seeds[1:])
    chosen = (
        inputs
        if first_inputs in ("all", "env_or_l0") and l0
        else {"all": inputs, "edge": (y,), "weight": (w,), "env_or_l0": (env,)}[
            first_inputs
        ]
    )
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in chosen))
    second_inputs = (*inputs, *seeds)
    second_seeds = tuple((torch.randn_like(v) for v in second_inputs))
    tol = (
        {"rtol": 1e-11, "atol": 1e-11}
        if dtype == torch.float64
        else {"rtol": 0.0005, "atol": 8e-05}
    )
    for a, b in zip(expected, actual):
        assert_optional_close(a, b, **tol)
    for reference, result in zip(
        optional_derivative_orders(
            chosen, seeds, second_inputs, directions, second_seeds, expected
        ),
        optional_derivative_orders(
            chosen, seeds, second_inputs, directions, second_seeds, actual
        ),
    ):
        for a, b in zip(reference, result):
            if a is None:
                assert b is None or bool((b == 0).all())
            else:
                assert_optional_close(a, torch.zeros_like(a) if b is None else b, **tol)


@pytest.mark.cuda
@pytest.mark.parametrize("feature", ["envelope", "l0"])
@pytest.mark.parametrize("trainable", ["edge", "weight"])
def test_optional_force_training(make_problem, feature, trainable, optional_dtype):
    torch.manual_seed(274)
    tp, x, y, w, src, dst = make_problem("uvu_conv", False, dtype=optional_dtype)
    x, y, w = (x.detach(), y.detach(), w.detach())
    parameters = (y if trainable == "edge" else w).requires_grad_()
    positions = torch.randn(len(x), 3, device="cuda", requires_grad=True)
    radius = (positions[src] - positions[dst]).square().sum(-1, keepdim=True)
    y, w = (y * torch.exp(-radius), w * (1 + radius))
    env = torch.exp(-0.2 * radius)
    op = TensorProductConv(tp, dtype=torch.get_default_dtype())
    graph = op.create_graph(src, dst, len(x))
    l0 = feature == "l0"
    actual = op(
        x, y, w, graph=graph, **{"return_l0_feat": True} if l0 else {"env": env}
    )
    actual = actual if l0 else (actual,)
    expected = reference_shape_outputs(tp, x, y, w, src, dst, None if l0 else env, l0)
    for a, b in zip(
        force_loss_derivatives(positions, parameters, expected),
        force_loss_derivatives(positions, parameters, actual),
    ):
        assert_optional_close(
            a,
            b,
            rtol=1e-11 if optional_dtype == torch.float64 else 0.0008,
            atol=1e-11 if optional_dtype == torch.float64 else 0.0005,
        )


@pytest.mark.cuda
def test_l0_fp64_wide_channels():
    """Cover FP64 streaming stores, channel chunks and a partial final warp."""
    test_complete_optional_derivatives(make_wide_problem, "l0", torch.float64, "all")
