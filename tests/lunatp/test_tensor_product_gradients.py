"""Tensor product outputs, mixed derivatives and finite differences."""

import pytest
import torch

from LunaEquivariance import TensorProduct, TensorProductConv


MODES = [
    ("uuu", False),
    ("uuu", True),
    ("uuu_conv", False),
    ("uvu", False),
    ("uvu", True),
    ("uvu_conv", False),
    ("uvw", False),
    ("uvw", True),
    ("uvw_conv", False),
    ("uvw_conv", True),
]


def assert_close(actual, expected):
    assert actual.dtype == expected.dtype
    assert bool(torch.isfinite(actual).all())
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-25)
    assert error.item() < (1e-11 if actual.dtype == torch.float64 else 2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("mode,shared", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_derivatives(make_problem, mode, shared, dtype):
    tp, x, y, w, src, dst = make_problem(mode, shared, dtype=dtype)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    kwargs = {}
    expected = tp(x[src] if mode.endswith("conv") else x, y, w)
    if mode.endswith("conv"):
        kwargs["graph"] = op.create_graph(src, dst, len(x))
        expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
            0, dst, expected
        )
    actual = op(x, y, None if shared else w, **kwargs)
    g = torch.randn_like(expected, requires_grad=True)
    inputs = (x, y, w, g)
    first_directions = tuple(
        torch.randn_like(v, requires_grad=True) for v in inputs[:3]
    )
    second_directions = tuple(torch.randn_like(v) for v in inputs)

    def chain(out):
        first = torch.autograd.grad(
            out, inputs[:3], g, create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, inputs, first_directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, *first_directions), second_directions, retain_graph=True
        )
        return (out,), first, second, third

    for reference_order, actual_order in zip(chain(expected), chain(actual)):
        for reference, result in zip(reference_order, actual_order):
            assert_close(result, reference)
    reference_x = torch.autograd.grad(expected, x, g, retain_graph=True)[0]
    assert_close(torch.autograd.grad(actual, x, g)[0], reference_x)
    if not shared:
        force = op(x, y.detach(), w.detach(), **kwargs)
        assert_close(torch.autograd.grad(force, x, g)[0], reference_x)


@pytest.mark.cuda
@pytest.mark.parametrize("mode,shared", MODES)
def test_fp64_finite_differences(make_problem, mode, shared):
    tp, x, y, w, src, dst = make_problem(mode, shared, dtype=torch.float64)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(
        tp, dtype=torch.float64
    )
    kwargs = (
        {"graph": (op.create_graph(src, dst, len(x)))} if mode.endswith("conv") else {}
    )
    if shared:
        fn, values = lambda a, b: op(a, b, **kwargs), (x, y)
    else:
        fn, values = lambda a, b, c: op(a, b, c, **kwargs), (x, y, w)
    # Atomic repeat differences measured on H100 are below 4e-15. This
    # only checks reentrancy, not the strict analytical derivative tolerance.
    nondet_tol = 1e-12 if (mode == "uvw" and shared) or mode.endswith("conv") else 0.0
    assert torch.autograd.gradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        nondet_tol=nondet_tol,
    )
    assert torch.autograd.gradgradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        nondet_tol=nondet_tol,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("mode,shared", MODES)
def test_fp64_empty_inputs(make_problem, mode, shared):
    tp, x, y, w, src, dst = make_problem(mode, shared, dtype=torch.float64)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    x, y = (x if mode.endswith("conv") else x[:0]), y[:0]
    w = w if shared else w[:0]
    kwargs = (
        {"graph": (op.create_graph(src[:0], dst[:0], len(x)))}
        if mode.endswith("conv")
        else {}
    )
    result = op(x, y, None if shared else w, **kwargs)
    assert result.shape == (len(x), tp.irreps_out.dim)
    first = torch.autograd.grad(result.sum(), (x, y, w))
    for value, gradient in zip((x, y, w), first):
        assert gradient.shape == value.shape
        assert not gradient.count_nonzero()
