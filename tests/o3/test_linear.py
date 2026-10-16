"""Linear first derivatives against e3nn, including frozen operand contracts."""

from itertools import product

import pytest
import torch
from e3nn import o3


LINEAR_CASES = {
    "scalar_tail": ("33x0e", "17x0e", None, True, "element"),
    "mlip": (
        "16x0e + 16x1o + 16x2e",
        "16x0e + 16x1o + 16x2e",
        None,
        True,
        "element",
    ),
    "repeated": (
        "5x0e + 3x1o + 7x0e + 2x2e",
        "3x0e + 11x1o + 2x0e + 1x3o",
        None,
        True,
        "path",
    ),
    "partial": (
        "5x0e + 3x1o + 7x0e + 2x2e",
        "3x0e + 11x1o + 2x0e + 1x3o",
        [(0, 0), (2, 0), (1, 1)],
        [True, False, False, False],
        "element",
    ),
    "zero_multiplicity": (
        "0x0e + 4x1o",
        "0x0e + 2x1o + 3x0e",
        None,
        False,
        "element",
    ),
}


def make_linear_problem(
    case, mode, dtype, device="cpu", items=7, *, feature_axes=False
):
    """Use e3nn's own instructions and fresh operands for each trial."""
    irreps_in, irreps_out, instructions, biases, normalization = LINEAR_CASES[case]
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        linear = o3.Linear(
            irreps_in,
            irreps_out,
            instructions=instructions,
            biases=biases,
            path_normalization=normalization,
            internal_weights=mode == "internal",
            shared_weights=mode != "external",
            # e3nn 0.6 builds per-item example biases while optimizing an
            # external-weight Linear, although its runtime bias is shared.
            _optimize_einsums=False if mode == "external" and biases else None,
            **(dict(f_in=2, f_out=3) if feature_axes else {}),
        ).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
    x = torch.randn(
        items,
        *((2,) if feature_axes else ()),
        linear.irreps_in.dim,
        dtype=dtype,
        device=device,
    )
    w = (
        linear.weight
        if mode == "internal"
        else torch.randn(
            *((items,) if mode == "external" else ()),
            *((2, 3) if feature_axes else ()),
            linear.weight_numel,
            dtype=dtype,
            device=device,
        )
    )
    b = (
        linear.bias
        if mode == "internal"
        else torch.randn(
            *((3,) if feature_axes else ()),
            linear.bias_numel,
            dtype=dtype,
            device=device,
        )
    )
    return linear, x, w, b


@pytest.mark.cuda
@pytest.mark.parametrize("case", ["scalar_tail", "mlip", "repeated", "partial"])
@pytest.mark.parametrize("mode", ["internal", "shared", "external"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "needs", [mask for mask in product((False, True), repeat=3) if any(mask)]
)
def test_linear_native_bwd_requested_outputs(case, mode, dtype, needs):
    from LunaEquivariance.modules.linear import LunaEqLinear

    torch.manual_seed(732)
    linear, x, w, b = make_linear_problem(case, mode, dtype, "cuda", items=19)
    g = torch.randn(19, linear.irreps_out.dim, device="cuda", dtype=dtype)
    op = LunaEqLinear(linear, dtype=dtype)
    for value, needed in zip((x, w, b), needs):
        value.requires_grad_(needed)
    inputs = tuple(value for value, needed in zip((x, w, b), needs) if needed)
    actual_output = op(x, w, b)
    reference_output = linear(x, w, b)
    actual = torch.autograd.grad(
        actual_output, inputs, g, allow_unused=True, materialize_grads=True
    )
    expected = torch.autograd.grad(
        reference_output, inputs, g, allow_unused=True, materialize_grads=True
    )
    tolerance = (
        dict(rtol=5e-5, atol=5e-6)
        if dtype == torch.float32
        else dict(rtol=3e-11, atol=3e-11)
    )
    torch.testing.assert_close(actual_output, reference_output, **tolerance)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, **tolerance)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["internal", "shared", "external"])
@pytest.mark.parametrize("case", ["scalar_tail", "repeated", "partial"])
def test_linear_mixed_derivatives(dtype, mode, case):
    from LunaEquivariance import Linear

    torch.manual_seed(733)
    reference, x, weight, bias = make_linear_problem(case, mode, dtype, "cuda", items=7)
    native = Linear(reference, dtype=dtype)
    inputs = (x, weight, bias)
    for value in inputs:
        value.requires_grad_()
    actual, expected = (native(*inputs),), (reference(*inputs),)
    targets = inputs
    tolerance = 5e-5 if dtype == torch.float32 else 3e-11
    for order in range(4):
        for value, target in zip(actual, expected):
            if value is None or target is None:
                # Unused bias paths may return an explicit zero gradient.
                remaining = target if value is None else value
                assert remaining is None or not remaining.count_nonzero()
            else:
                torch.testing.assert_close(
                    value, target, rtol=tolerance, atol=tolerance
                )
        if order == 3:
            break
        pairs = [
            (a, b)
            for a, b in zip(actual, expected)
            if a is not None and b is not None and a.requires_grad and b.requires_grad
        ]
        seeds = tuple(torch.randn_like(a, requires_grad=True) for a, _ in pairs)
        actual = torch.autograd.grad(
            tuple(a for a, _ in pairs),
            targets,
            seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        expected = torch.autograd.grad(
            tuple(b for _, b in pairs),
            targets,
            seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        targets = (*targets, *seeds)
