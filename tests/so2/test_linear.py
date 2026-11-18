"""Orderwise channel projections and their derivatives."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from LunaEquivariance import so2 as op
from LunaEquivariance.so2.linear.interface import project_sectors
from tests.so2.checks import assert_autograd_chain, fp64
from tests.so2.reference.projection import linear as linear_reference
from tests.testing_utils import assert_reference_close


@pytest.mark.parametrize("shape", [(3, 3, 4), (3, 3, 2, 4)])
def test_projection_rejects_ambiguous_planes(shape):
    with pytest.raises(ValueError, match="two planes"):
        project_sectors(
            (torch.zeros(3, 6), torch.zeros(shape)),
            torch.nn.Identity(),
            (torch.nn.Identity(),),
            output_channels=2,
        )


def conditioned_projection(weight, x):
    return torch.bmm(x, weight.transpose(1, 2))


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("coefficient_axes", ["flat", "explicit", "merged"])
def test_projection_parameter_and_feature_derivatives(conditioned, coefficient_axes):
    torch.manual_seed(42)
    edges, cin, cout, extras = 3, 2, 3, 4
    x0 = torch.randn(edges, 3, cin, dtype=torch.double, requires_grad=True)
    x1 = torch.randn(edges, 2, 2, cin, dtype=torch.double, requires_grad=True)
    w0 = torch.randn(extras + 3 * cout, 3 * cin, dtype=torch.double, requires_grad=True)
    w1 = torch.randn(
        (edges, 4 * cout, 2 * cin) if conditioned else (4 * cout, 2 * cin),
        dtype=torch.double,
        requires_grad=True,
    )
    b0 = torch.randn(extras + 3 * cout, dtype=torch.double, requires_grad=True)
    sectors = {
        "flat": (x0.flatten(1), x1.flatten(2)),
        "explicit": (x0, x1),
        "merged": (x0, x1.flatten(1, 2)),
    }[coefficient_axes]
    options = {"apply_pair": lambda fn, x: fn(x)} if conditioned else {}
    maps = (
        partial(conditioned_projection, w1)
        if conditioned
        else partial(F.linear, weight=w1),
    )
    side, projected = project_sectors(
        sectors,
        partial(F.linear, weight=w0, bias=b0),
        maps,
        output_channels=cout,
        scalar_prefix=extras,
        **options,
    )
    scalar = F.linear(x0.flatten(1), w0, b0)
    pair = torch.stack(
        [
            torch.stack(
                [
                    F.linear(x1[e, p].flatten(), w1[e] if conditioned else w1)
                    for p in range(2)
                ]
            )
            for e in range(edges)
        ]
    )
    reference = (scalar[:, :extras], scalar[:, extras:].reshape(edges, 3, cout), pair)
    actual = (side, *projected)
    inputs = (x0, x1, w0, w1, b0)
    directions = tuple(torch.randn_like(value) for value in inputs)

    def derivatives(outputs):
        first = torch.autograd.grad(
            sum(o.sin().sum() for o in outputs), inputs, create_graph=True
        )
        return (*first, *torch.autograd.grad(first, inputs, directions))

    for result, expected in zip(
        (*actual, *derivatives(actual)), (*reference, *derivatives(reference))
    ):
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)


def test_external_projection_weights_support_second_derivatives():
    from LunaEquivariance.so2.linear.interface import project_sectors

    generator = torch.Generator().manual_seed(123)
    x0, x1, w0, w1 = [
        torch.randn(shape, generator=generator, dtype=torch.double, requires_grad=True)
        for shape in ((2, 2), (2, 2, 1), (3, 2), (2, 1))
    ]

    def project(x0, x1, w0, w1):
        side, sectors = project_sectors(
            (x0, x1),
            partial(F.linear, weight=w0),
            (partial(F.linear, weight=w1),),
            output_channels=1,
            scalar_prefix=1,
        )
        return side, *sectors

    side, scalar, pair = project(x0, x1, w0, w1)
    torch.testing.assert_close(side, F.linear(x0, w0)[:, :1])
    torch.testing.assert_close(scalar, F.linear(x0, w0)[:, 1:].reshape(2, 2, 1))
    torch.testing.assert_close(pair, F.linear(x1, w1))
    assert torch.autograd.gradcheck(project, (x0, x1, w0, w1))
    assert torch.autograd.gradgradcheck(project, (x0, x1, w0, w1))


@pytest.mark.parametrize("lmax,mmax,channels", [(0, 0, 3), (2, 1, 5), (4, 2, 8)])
@pytest.mark.parametrize("strided", [False, True])
def test_linear_cpu(lmax, mmax, channels, strided):
    torch.manual_seed(907)
    edges, outputs, auxiliary = 3, 7, 2
    sectors = [torch.randn(edges, lmax + 1, channels, requires_grad=True)]
    for order in range(1, mmax + 1):
        width = lmax + 1 - order
        value = torch.randn(edges, 2, width * channels * (2 if strided else 1))
        sectors.append((value[..., ::2] if strided else value).requires_grad_())
    weights = [torch.randn(auxiliary + (lmax + 1) * outputs, (lmax + 1) * channels, requires_grad=True)]
    weights += [torch.randn(2 * (lmax + 1 - m) * outputs,
                            (lmax + 1 - m) * channels, requires_grad=True) for m in range(1, mmax + 1)]
    bias = torch.randn(weights[0].shape[0], requires_grad=True)
    targets = (*sectors, *weights, bias)
    refs = fp64(targets)
    split = mmax + 1
    actual = op.linear(sectors, weights, bias, lmax=lmax, mmax=mmax)
    wanted = linear_reference(refs[:split], refs[split:2 * split], refs[-1], lmax=lmax, mmax=mmax)
    loss = sum(value.square().mean() for value in actual)
    expected_loss = sum(value.square().mean() for value in wanted)
    for value, target in zip(actual, wanted):
        assert_reference_close(value, target)
    assert_autograd_chain(loss, targets, expected_loss, refs)


@pytest.mark.parametrize("bad", ["missing_weight", "bad_plane", "bad_order"])
def test_linear_rejects_inconsistent_sectors(bad):
    sectors = (torch.randn(2, 6), torch.randn(2, 2, 3))
    weights = (torch.randn(6, 6), torch.randn(6, 3))
    if bad == "missing_weight":
        weights = weights[:1]
    elif bad == "bad_plane":
        sectors = (sectors[0], torch.randn(2, 3, 3))
    else:
        sectors = sectors[:1]
    with pytest.raises(ValueError):
        op.linear(sectors, weights, lmax=1, mmax=1)


def test_functional_compatibility_report():
    from tests.so2.compatibility import check_compatibility

    sectors = (torch.randn(2, 4, requires_grad=True), torch.randn(2, 2, 2, requires_grad=True))
    weights = (torch.randn(6, 4, requires_grad=True), torch.randn(6, 2, requires_grad=True))
    bias = torch.randn(6, requires_grad=True)
    report = check_compatibility(
        op.linear, (sectors, weights, bias), reference=linear_reference,
        kwargs={"lmax": 1, "mmax": 1}, derivative_order=2,
        warmup=0, repeats=1, verbose=False,
    )
    assert report["passed"]
    assert report["timings"]["forward_loss_gradient_2"]["lunaeq_ms"] > 0


@pytest.mark.parametrize("form", ["flat_planes", "coefficient_rows", "explicit_planes"])
def test_linear_physical_sector_layouts(form):
    lmax, mmax, channels, outputs = 2, 1, 3, 4
    scalar = torch.randn(2, lmax + 1, channels)
    planes = torch.randn(2, 2, 2, channels)
    value = {
        "flat_planes": planes.flatten(2),
        "coefficient_rows": planes.flatten(1, 2),
        "explicit_planes": planes,
    }[form]
    weights = (torch.randn((lmax + 1) * outputs, (lmax + 1) * channels),
               torch.randn(4 * outputs, 2 * channels))
    actual = op.linear((scalar, value), weights, lmax=lmax, mmax=mmax)
    expected = linear_reference((scalar, planes), weights, lmax=lmax, mmax=mmax)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("case", ["scalar_rows", "plane_axes", "extra_axes", "scalar_output"])
def test_linear_rejects_ambiguous_logical_layout(case):
    scalar = torch.randn(2, 3, 4)
    plane = torch.randn(2, 2, 8)
    weights = (torch.randn(6, 12), torch.randn(8, 8))
    expected = ""
    if case == "scalar_rows":
        scalar = scalar.reshape(2, 2, 6)
        expected = "L\\+1"
    elif case == "plane_axes":
        plane = plane.reshape(2, 1, 16)
        expected = "planes"
    elif case == "extra_axes":
        plane = plane.reshape(2, 2, 2, 2, 2)
        expected = "planes"
    else:
        weights = (torch.randn(5, 12), weights[1])
        expected = "output"
    with pytest.raises(ValueError, match=expected):
        op.linear((scalar, plane), weights, lmax=2, mmax=1)
