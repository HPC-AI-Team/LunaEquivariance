"""Cross-check order layouts, activation and weighting against PyTorch."""

import os
import subprocess
import sys
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from tests.so2.reference import recombination as activation_reference
from tests.so2.reference import attention_score as attention_reference
from tests.so2.reference import segmented_linear as experts_reference
from tests.so2.reference import projection as projection_reference


NAMES = (
    "raw_sector_linear",
    "project_sectors",
    "merge_sectors",
    "recombine_sectors",
    "activate_gated",
    "activate_grid",
    "fused_alpha_projection",
    "segmented_linear",
)


EDGE_REFERENCES = {
    "raw_sector_linear": projection_reference.raw_sector_linear,
    "project_sectors": projection_reference.project_sectors,
    "merge_sectors": projection_reference.merge_sectors,
    "recombine_sectors": activation_reference.recombine_sectors,
    "activate_gated": activation_reference.activate_gated,
    "activate_grid": activation_reference.activate_grid,
    "fused_alpha_projection": attention_reference.fused_alpha_projection,
    "segmented_linear": experts_reference.segmented_linear,
}


def flatten(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(tensor for output in outputs for tensor in flatten(output))


def case(
    name,
    function,
    *,
    device="cpu",
    dtype=torch.float64,
    lmax=2,
    mmax=1,
    channels=2,
):
    generator = torch.Generator(device=device).manual_seed(923)
    edges, extras = 3, 2

    def random(shape):
        return (
            torch.randn(shape, generator=generator, device=device, dtype=dtype) * 0.2
        ).requires_grad_()

    if name in {"raw_sector_linear", "project_sectors"}:
        cin = channels + 1
        shapes = [(edges, lmax + 1, cin)]
        shapes += [(edges, 2, (lmax + 1 - m) * cin) for m in range(1, mmax + 1)]
        weights = [(extras + (lmax + 1) * channels, (lmax + 1) * cin)]
        weights += [
            (2 * (lmax + 1 - m) * channels, (lmax + 1 - m) * cin)
            for m in range(1, mmax + 1)
        ]
        inputs = tuple(random(shape) for shape in (*shapes, *weights, (weights[0][0],)))

        def evaluate(*values):
            sectors, maps = values[: mmax + 1], values[mmax + 1 : -1]
            scalar = partial(F.linear, weight=maps[0], bias=values[-1])
            orders = tuple(partial(F.linear, weight=weight) for weight in maps[1:])
            options = (
                {"output_channels": channels, "scalar_prefix": extras}
                if name == "project_sectors"
                else {}
            )
            return flatten(
                function(sectors, scalar, orders, **options)
            )

    elif name in {"merge_sectors", "recombine_sectors"}:
        if name == "merge_sectors":
            shapes = [(edges, lmax + 1, channels)]
            shapes += [(edges, 2, lmax + 1 - m, channels) for m in range(1, mmax + 1)]
        else:
            shapes = [(edges, extras + (lmax + 1) * channels)]
            shapes += [
                (edges, 2, 2 * (lmax + 1 - m) * channels) for m in range(1, mmax + 1)
            ]
        inputs = tuple(random(shape) for shape in shapes)

        def evaluate(*values):
            options = {"extra_channels": extras} if name == "recombine_sectors" else {}
            return flatten(
                function(
                    values,
                    lmax=lmax,
                    mmax=mmax,
                    channels=channels,
                    **options,
                )
            )

    elif name == "activate_gated":
        shapes = [(edges, lmax * channels), (edges, lmax + 1, channels)]
        shapes += [
            (edges, 2, 2 * (lmax + 1 - m) * channels) for m in range(1, mmax + 1)
        ]
        inputs = tuple(random(shape) for shape in shapes)

        def evaluate(*values):
            return function(
                values[0],
                values[1],
                values[2:],
                lmax=lmax,
                mmax=mmax,
            )

    elif name == "activate_grid":
        shapes = [(edges, extras + 3 * channels + (lmax + 1) * 2 * channels)]
        shapes += [
            (edges, 2, 4 * (lmax + 1 - m) * channels) for m in range(1, mmax + 1)
        ]
        inputs = tuple(random(shape) for shape in shapes)
        rows = lmax + 1 + 2 * sum(lmax + 1 - m for m in range(1, mmax + 1))
        to_grid = random((11, rows)).detach()
        from_grid = random((rows, 11)).detach()

        def evaluate(*values):
            return flatten(
                function(
                    values,
                    to_grid.to(values[0]),
                    from_grid.to(values[0]),
                    lmax=lmax,
                    mmax=mmax,
                    channels=channels,
                    auxiliary_channels=extras,
                )
            )

    elif name == "fused_alpha_projection":
        inputs = tuple(random(shape) for shape in ((edges, 2, 3), (3,), (3,), (2, 3)))

        def evaluate(*values):
            return (function(*values, eps=1e-5),)

    else:
        inputs = tuple(random(shape) for shape in ((3, 2, 2), (2, 3, 2), (3, 2), (3,)))

        def evaluate(*values):
            return (function(*values, (1, 0, 2)),)

    return evaluate, inputs


def test_references_import_without_model_packages():
    script = """
import importlib.abc
import sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fairchem'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectRuntime())
from tests.so2.reference.recombination import activate_grid, activate_gated
assert callable(activate_grid) and callable(activate_gated)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, "-c", script], env=environment, check=True)


def test_recombination_signs_and_scalar_prefix():
    raw = (torch.tensor([[7.0, 11.0, 13.0]]), torch.tensor([[[2.0, 3.0], [5.0, 7.0]]]))
    coefficients, extras = activation_reference.recombine_sectors(
        raw,
        lmax=1,
        mmax=1,
        channels=1,
        extra_channels=1,
    )
    torch.testing.assert_close(
        coefficients, torch.tensor([[[11.0], [13.0], [-5.0], [8.0]]])
    )
    torch.testing.assert_close(extras, torch.tensor([[7.0]]))


def test_merge_row_order_and_degree_specific_gates():
    scalar = torch.tensor([[[2.0], [3.0], [4.0]]], dtype=torch.float64)
    raw = (
        torch.tensor(
            [[[5.0, 6.0, 7.0, 8.0], [11.0, 12.0, 13.0, 14.0]]], dtype=torch.float64
        ),
        torch.tensor([[[17.0, 19.0], [23.0, 29.0]]], dtype=torch.float64),
    )
    gates = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    sectors = activation_reference.activate_gated(gates, scalar, raw, lmax=2, mmax=2)
    g2 = torch.sigmoid(torch.tensor(2.0, dtype=torch.float64))
    expected = torch.stack(
        (
            F.silu(torch.tensor(2.0, dtype=torch.float64)),
            torch.tensor(1.5),
            4 * g2,
            torch.tensor(-4.0),
            -8 * g2,
            torch.tensor(9.0),
            20 * g2,
            -12 * g2,
            42 * g2,
        )
    ).reshape(1, 9, 1)
    merged = projection_reference.merge_sectors(sectors, lmax=2, mmax=2, channels=1)
    torch.testing.assert_close(merged, expected)


def test_grid_product_identity_tables_and_scalar_row():
    raw = (
        torch.tensor([[9.0, 2.0, 3.0, 0.0, 2.0, 3.0, 5.0, 7.0]]),
        torch.arange(1.0, 9.0).reshape(1, 2, 4),
    )
    sectors, auxiliary = activation_reference.activate_grid(
        raw,
        torch.eye(4),
        torch.eye(4),
        lmax=1,
        mmax=1,
        channels=1,
        auxiliary_channels=1,
    )
    actual = projection_reference.merge_sectors(sectors, lmax=1, mmax=1, channels=1)
    expected = torch.tensor([3.0, 17.5, 18.0, 40.0]).reshape(1, 4, 1)
    expected[:, 0] += F.silu(torch.tensor(2.0)) * 3
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(auxiliary, torch.tensor([[9.0]]))


def test_attention_constant_input_reduces_to_affine_bias():
    inputs = torch.full((2, 2, 3), 4.0, dtype=torch.float64)
    weight = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    bias = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    dot = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 2.0, 4.0]], dtype=torch.float64)
    result = attention_reference.fused_alpha_projection(inputs, weight, bias, dot, 1e-5)
    activated_bias = 0.6 * bias + 0.4 * bias * torch.tanh(bias / 2)
    torch.testing.assert_close(result, (activated_bias * dot).sum(-1).expand(2, -1))


def test_segmented_mixture_equals_mixture_of_expert_outputs():
    function, inputs = case("segmented_linear", experts_reference.segmented_linear)
    x, weights, coefficients, bias = inputs
    expert_outputs = torch.stack([F.linear(x, weight) for weight in weights], dim=1)
    edge_coefficients = coefficients.repeat_interleave(torch.tensor([1, 0, 2]), dim=0)
    expected = torch.einsum("nk,nkto->nto", edge_coefficients, expert_outputs) + bias
    torch.testing.assert_close(function(*inputs)[0], expected)


@pytest.mark.parametrize("name", NAMES)
def test_reference_first_and_second_derivatives(name):
    function, inputs = case(name, EDGE_REFERENCES[name], lmax=1, mmax=1, channels=1)
    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)


def derivatives(function, inputs, cotangents, directions):
    outputs = function(*inputs)
    first = torch.autograd.grad(
        outputs, inputs, cotangents, create_graph=True, allow_unused=True
    )
    first = tuple(
        value * 0 if gradient is None else gradient
        for value, gradient in zip(inputs, first, strict=True)
    )
    second = torch.autograd.grad(
        first, (*inputs, *cotangents), directions, allow_unused=True
    )
    targets = (*inputs, *cotangents)
    second = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(targets, second, strict=True)
    )
    return *outputs, *first, *second


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("lmax,mmax,channels", [(0, 0, 3), (3, 2, 7)])
def test_optimized_forward_backward_double_backward(name, lmax, mmax, channels):
    from LunaEquivariance.so2.linear.interface import raw_sector_linear, project_sectors
    from LunaEquivariance.so2.utils.sector_layout import merge_sectors
    from LunaEquivariance.so2.recombine_activate.interface import recombine_sectors
    from LunaEquivariance.so2.recombine_activate.interface import gated_activation
    from LunaEquivariance.so2.recombine_activate.interface import grid_activation
    from LunaEquivariance.so2 import attention_scores, segmented_linear
    optimized = dict(raw_sector_linear=raw_sector_linear, project_sectors=project_sectors,
        merge_sectors=merge_sectors, recombine_sectors=recombine_sectors,
        activate_gated=gated_activation, activate_grid=grid_activation,
        fused_alpha_projection=attention_scores, segmented_linear=segmented_linear)

    actual, inputs = case(
        name,
        optimized[name],
        device="cuda",
        dtype=torch.float32,
        lmax=lmax,
        mmax=mmax,
        channels=channels,
    )
    expected, reference_inputs = case(
        name,
        EDGE_REFERENCES[name],
        device="cuda",
        dtype=torch.float32,
        lmax=lmax,
        mmax=mmax,
        channels=channels,
    )
    # Keep the same FP32 values, then evaluate the independent formula in FP64.
    reference_inputs = tuple(
        value.detach().double().requires_grad_() for value in reference_inputs
    )
    cotangents = tuple(
        torch.randn_like(value).requires_grad_() for value in actual(*inputs)
    )
    reference_cotangents = tuple(
        value.detach().double().requires_grad_() for value in cotangents
    )
    directions = tuple(torch.randn_like(value) for value in inputs)
    observed = derivatives(actual, inputs, cotangents, directions)
    target = derivatives(
        expected,
        reference_inputs,
        reference_cotangents,
        tuple(value.double() for value in directions),
    )
    for value, expected_value in zip(observed, target, strict=True):
        torch.testing.assert_close(value.double(), expected_value, rtol=5e-4, atol=3e-5)
