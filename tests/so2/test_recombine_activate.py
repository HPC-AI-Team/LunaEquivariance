"""Raw product recombination and gated or grid activation."""

import pytest
import torch

from LunaEquivariance.so2.recombine_activate.interface import (
    complex_recombination,
    gated_activation,
    grid_activation,
    recombine_activate,
    recombine_sectors,
    recombine_sectors_adjoint,
)
from LunaEquivariance.so2.utils.sector_layout import layout, merge_sectors
from tests.so2.checks import assert_directional_derivatives, make_cotangent, strided_copy
from tests.so2.reference.recombination import activate_gated, activate_grid


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 0), (3, 2)])
def test_gate_retains_physical_outputs_and_returns_raw_gradient_slots(
    monkeypatch, lmax, mmax
):
    """Check Autograd state and buffer contracts without executing CUDA arithmetic."""
    from LunaEquivariance.so2.recombine_activate import interface as gate

    from LunaEquivariance.so2.recombine_activate.kernels import backward

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return self.run

        def run(self, *arguments, **dimensions):
            calls.append((self, arguments))
            if self is gate.forward_kernel:
                destinations = arguments[2]
            elif self is backward.backward_kernel:
                destinations = (arguments[3], *arguments[4])
            else:
                destinations = (*arguments[5], arguments[6], *arguments[7])
            for tensor in destinations:
                tensor.fill_(1)

    monkeypatch.setattr(gate, "forward_kernel", Kernel())
    monkeypatch.setattr(backward, "backward_kernel", Kernel())
    monkeypatch.setattr(gate, "double_backward_kernel", Kernel())
    edges, channels = 2, 3
    gating = torch.randn(edges, lmax * channels, requires_grad=True)
    sectors = (torch.randn(edges, lmax + 1, channels, requires_grad=True),) + tuple(
        torch.randn(edges, 2, 2 * (lmax - order + 1) * channels, requires_grad=True)
        for order in range(1, mmax + 1)
    )
    result = gate.GroupedGate.apply(gating, sectors[0], lmax, mmax, *sectors[1:])
    outputs, carriers = result[:mmax + 1], result[mmax + 1:]
    saved = outputs[0].grad_fn.saved_tensors
    assert len(saved) == 2 * mmax + 2
    assert saved[0] is gating
    assert saved[1] is sectors[0]
    for retained, output, carrier, saved_carrier in zip(
        saved[2:mmax + 2], outputs[1:], carriers, saved[mmax + 2:], strict=True,
    ):
        assert retained.data_ptr() == output.data_ptr()
        assert retained.shape == output.shape
        assert saved_carrier is not None
        assert carrier is not output
        assert carrier.data_ptr() == saved_carrier.data_ptr() == output.data_ptr()
        assert carrier.shape == saved_carrier.shape == output.shape
    for raw in sectors[1:]:
        assert all(raw.data_ptr() != retained.data_ptr() for retained in saved)

    inputs = (gating, *sectors)
    cotangents = tuple(
        torch.randn_like(output, requires_grad=True) for output in outputs
    )
    first = torch.autograd.grad(outputs, inputs, cotangents, create_graph=True)
    second = torch.autograd.grad(
        first, (*inputs, *cotangents), tuple(torch.ones_like(value) for value in first)
    )
    assert tuple(value.shape for value in first) == tuple(
        value.shape for value in inputs
    )
    assert tuple(value.shape for value in second) == tuple(
        value.shape for value in (*inputs, *cotangents)
    )

    _, backward_args = next(call for call in calls if call[0] is backward.backward_kernel)
    _, double_backward_args = next(
        call for call in calls if call[0] is gate.double_backward_kernel
    )
    assert tuple(value.shape for value in backward_args[4]) == tuple(
        value.shape for value in sectors
    )
    assert tuple(value.shape for value in double_backward_args[4]) == tuple(
        value.shape for value in sectors
    )
    assert tuple(value.shape for value in double_backward_args[7]) == tuple(
        value.shape for value in outputs
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,mmax,channels",
    [(0, 0, 7), (3, 0, 33), (2, 2, 8), (3, 2, 8), (4, 4, 8),
     (5, 5, 7), (6, 5, 33), (3, 2, 257)],
)
@pytest.mark.parametrize("strided_direction", [False, True])
def test_gate_recombination_derivatives(lmax, mmax, channels, strided_direction):
    torch.manual_seed(27)
    edges = 4
    inputs = [
        torch.randn(edges, lmax * channels, device="cuda").requires_grad_(),
        torch.randn(edges, lmax + 1, channels, device="cuda").requires_grad_(),
    ]
    inputs += [
        torch.randn(
            edges, 2, 2 * (lmax - m + 1) * channels, device="cuda"
        ).requires_grad_()
        for m in range(1, mmax + 1)
    ]
    ref = [value.detach().double().requires_grad_() for value in inputs]
    expected = activate_gated(ref[0], ref[1], ref[2:], lmax=lmax, mmax=mmax)
    actual = gated_activation(inputs[0], inputs[1], inputs[2:], lmax=lmax, mmax=mmax)
    cotangents = [torch.randn_like(value).requires_grad_() for value in actual]
    ref_cotangents = [value.detach().double().requires_grad_() for value in cotangents]
    first = torch.autograd.grad(actual, inputs, cotangents, create_graph=True)
    ref_first = torch.autograd.grad(expected, ref, ref_cotangents, create_graph=True)
    directions = [torch.randn_like(value) for value in first]
    if strided_direction:
        directions = [
            torch.randn(*value.shape[:-1], 2 * value.shape[-1], device="cuda")[..., ::2]
            for value in first
        ]
    second = torch.autograd.grad(first, inputs + cotangents, directions)
    ref_second = torch.autograd.grad(
        ref_first, ref + ref_cotangents, [value.double() for value in directions]
    )
    for got, want in zip(
        [*actual, *first, *second], [*expected, *ref_first, *ref_second], strict=True
    ):
        torch.testing.assert_close(got.double(), want, atol=3e-5, rtol=3e-4)


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3)])
def test_grid_activation_forward_backward_double_backward(lmax, mmax):
    torch.manual_seed(19)
    edges = 3
    shapes = [(edges, 608 + (lmax + 1) * 64)]
    shapes += [
        (edges, 2, 2 * (lmax - order + 1) * 64) for order in range(1, mmax + 1)
    ]
    raw = tuple(
        (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
        for shape in shapes
    )
    ref = tuple(x.detach().clone().requires_grad_() for x in raw)
    rows = sum((1 if m == 0 else 2) * (lmax - m + 1) for m in range(mmax + 1))
    to_grid = torch.randn(112, rows, device="cuda") * 0.1
    from_grid = torch.randn(rows, 112, device="cuda") * 0.1
    sectors, alpha = grid_activation(
        raw,
        to_grid,
        from_grid,
        lmax=lmax,
        mmax=mmax,
        channels=32,
        auxiliary_channels=512,
    )
    got = (*sectors, alpha)
    expected_sectors, expected_alpha = activate_grid(
        ref, to_grid, from_grid, lmax=lmax, mmax=mmax, channels=32, auxiliary_channels=512,
    )
    expected = (*expected_sectors, expected_alpha)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    first = torch.autograd.grad(
        sum(x.square().sum() for x in got), raw, create_graph=True
    )
    first_ref = torch.autograd.grad(
        sum(x.square().sum() for x in expected), ref, create_graph=True
    )
    directions = tuple(torch.randn_like(x) for x in raw)
    second = torch.autograd.grad(first, raw, directions, create_graph=True)
    second_ref = torch.autograd.grad(first_ref, ref, directions)
    for values, refs in [(first, first_ref), (second, second_ref)]:
        for a, b in zip(values, refs):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=3e-4)


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("physical_input", [False, True])
def test_grid_activation_broadcast_cotangent_and_strided_second_directions(
    monkeypatch, physical_input,
):
    from LunaEquivariance.so2.recombine_activate import interface as activation

    torch.manual_seed(19)
    edges, lmax, mmax = 3, 4, 2
    divisor = 2 if physical_input else 1
    shapes = [(edges, 928), (edges, 2, 512 // divisor), (edges, 2, 384 // divisor)]
    if physical_input:
        monkeypatch.setattr(activation, "use_sector_cotangents", lambda *_: True)
    raw = (
        torch.randn(edges * 2, shapes[0][-1] + 16, device="cuda")[::2, 8:-8]
        .mul_(0.1)
        .requires_grad_(),
        *(
            torch.randn(edges * 2, 4, shape[-1] + 16, device="cuda")[::2, ::2, 8:-8]
            .mul_(0.1)
            .requires_grad_()
            for shape in shapes[1:]
        ),
    )
    ref = tuple(value.detach().double().requires_grad_() for value in raw)
    to_grid = torch.randn(112, 19, device="cuda") * 0.1
    from_grid = torch.randn(19, 112, device="cuda") * 0.1
    sectors, alpha = grid_activation(
        raw,
        to_grid,
        from_grid,
        lmax=lmax,
        mmax=mmax,
        channels=32,
        auxiliary_channels=512,
        physical_input=physical_input,
    )
    expected_sectors, expected_alpha = activate_grid(
        ref, to_grid.double(), from_grid.double(),
        lmax=lmax, mmax=mmax, channels=32, auxiliary_channels=512,
        physical_input=physical_input,
    )
    expected = (*expected_sectors, expected_alpha)
    outputs = (*sectors, alpha)
    cotangents = tuple(
        torch.randn((), device="cuda").expand_as(value).requires_grad_()
        for value in outputs
    )
    ref_cotangents = tuple(
        value.detach().double().requires_grad_() for value in cotangents
    )
    first = torch.autograd.grad(outputs, raw, cotangents, create_graph=True)
    first_ref = torch.autograd.grad(
        expected, ref, ref_cotangents, create_graph=True
    )
    directions = tuple(
        torch.randn(*shape[:-1], shape[-1] * 2, device="cuda")[..., ::2]
        for shape in shapes
    )
    second = torch.autograd.grad(
        first, raw + cotangents, directions, allow_unused=True
    )
    second_ref = torch.autograd.grad(
        first_ref,
        ref + ref_cotangents,
        tuple(value.double() for value in directions),
        allow_unused=True,
    )
    for actual, reference in zip((*first, *second), (*first_ref, *second_ref)):
        if actual is None or reference is None:
            assert actual is reference
        else:
            torch.testing.assert_close(
                actual.double(), reference, atol=2e-5, rtol=3e-4
            )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,mmax,c,auxiliary",
    [
        (0, 0, 1, 0),
        (2, 0, 7, 11),
        (2, 2, 17, 13),
        (4, 4, 33, 0),
        (5, 5, 8, 29),
        (4, 2, 64, 128),
    ],
)
def test_grid_dimensions(lmax, mmax, c, auxiliary):
    torch.manual_seed(2026)
    e = 3
    shapes = [(e, auxiliary + 3 * c + (lmax + 1) * 2 * c)] + [
        (e, 2, 4 * (lmax - m + 1) * c) for m in range(1, mmax + 1)
    ]
    raw = tuple((torch.randn(s, device="cuda") * 0.1).requires_grad_() for s in shapes)
    ref = tuple(x.detach().double().requires_grad_() for x in raw)
    rows = sum((1 if m == 0 else 2) * (lmax - m + 1) for m in range(mmax + 1))
    t = torch.randn(19, rows, device="cuda") * 0.1
    f = torch.randn(rows, 19, device="cuda") * 0.1
    sectors, alpha = grid_activation(
        raw, t, f, lmax=lmax, mmax=mmax, channels=c, auxiliary_channels=auxiliary
    )
    got = (*sectors, alpha)
    expected_sectors, expected_alpha = activate_grid(
        ref, t.double(), f.double(), lmax=lmax, mmax=mmax,
        channels=c, auxiliary_channels=auxiliary,
    )
    expected = (*expected_sectors, expected_alpha)
    go = tuple(torch.randn_like(x).requires_grad_() for x in got)
    ro = tuple(x.detach().double().requires_grad_() for x in go)
    d = torch.autograd.grad(got, raw, go, create_graph=True)
    dr = torch.autograd.grad(expected, ref, ro, create_graph=True)
    h = tuple(torch.randn_like(x) for x in raw)
    dd = torch.autograd.grad(d, (*raw, *go), h, allow_unused=True)
    ddr = torch.autograd.grad(
        dr, (*ref, *ro), tuple(x.double() for x in h), allow_unused=True
    )
    for x, y in zip((*got, *d, *dd), (*expected, *dr, *ddr)):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x.double(), y, atol=2e-5, rtol=3e-4)


@pytest.mark.cuda
def test_grouped_gate_strided_inputs_and_directions_gpu():
    torch.manual_seed(183)
    values = (
        torch.randn(4, 8, device="cuda"),
        torch.randn(4, 3, 4, device="cuda"),
        torch.randn(4, 2, 16, device="cuda"),
        torch.randn(4, 2, 8, device="cuda"),
    )

    def evaluate(dense):
        inputs = tuple(
            (x.clone() if dense else strided_copy(x)).requires_grad_() for x in values
        )
        outputs = gated_activation(inputs[0], inputs[1], inputs[2:], lmax=2, mmax=2)
        first = torch.autograd.grad(
            sum(y.sum() for y in outputs), inputs, create_graph=True
        )
        directions = tuple(
            torch.ones_like(x) if dense else make_cotangent(x, "sum") for x in first
        )
        second = torch.autograd.grad(first, inputs, directions, create_graph=True)
        return (*outputs, *first, *second)

    for got, want in zip(evaluate(False), evaluate(True), strict=True):
        torch.testing.assert_close(got, want, rtol=3e-5, atol=3e-5)


@pytest.mark.cuda
def test_packed_gate_scalar_derivatives():
    torch.manual_seed(183)
    shapes = ((4, 40), (4, 2, 32), (4, 2, 16))
    raw = tuple(
        strided_copy(torch.randn(shape, device="cuda")).requires_grad_()
        for shape in shapes
    )
    reference = tuple(value.detach().double().requires_grad_() for value in raw)
    actual, side = recombine_activate(raw, lmax=2, mmax=2, activation="gate")
    gates, coefficients = reference[0].split((16, 24), dim=-1)
    expected = activate_gated(gates, coefficients.reshape(4, 3, 8), reference[1:],
                              lmax=2, mmax=2)
    assert side.shape == (4, 0)
    assert_directional_derivatives(
        torch.cat([value.flatten(1) for value in actual], dim=1),
        torch.cat([value.flatten(1) for value in expected], dim=1),
        raw, reference,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("half", [3, 129])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_recombination_and_double_backward(half, strided, nonlinear):
    torch.manual_seed(42)
    data = torch.randn(5, 2, 2 * half, device="cuda")
    cotangent_data = torch.randn(5, 2, half * (2 if strided else 1), device="cuda")
    direction = torch.randn(5, 2, 2 * half * (2 if strided else 1), device="cuda")
    if strided:
        cotangent_data = cotangent_data[..., ::2]
        direction = direction[..., ::2]

    results = []
    for optimized in (False, True):
        raw = data.detach().requires_grad_()
        cotangent = cotangent_data.detach().requires_grad_()
        if optimized:
            output = complex_recombination(raw, half)
        else:
            q = raw.reshape(5, 2, 2, half)
            output = torch.stack((q[:, 0, 0] - q[:, 1, 1], q[:, 1, 0] + q[:, 0, 1]), 1)
        value = output.sin() if nonlinear else output
        gradient = torch.autograd.grad(value, raw, cotangent, create_graph=True)[0]
        second = torch.autograd.grad(
            gradient, (raw, cotangent), direction, allow_unused=True
        )
        results.append((output, gradient, *second))

    for actual, expected in zip(results[1], results[0], strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax,channels", [(3, 0, 3), (3, 2, 3), (3, 3, 3),
                                                (5, 0, 7), (5, 4, 7), (5, 5, 7)])
@pytest.mark.parametrize("extras", [0, 5])
def test_recombination_forward_adjoint_and_second_derivatives(lmax, mmax, channels, extras):
    edges = 2
    raw = (
        torch.randn(
            edges, extras + (lmax + 1) * channels, device="cuda", requires_grad=True
        ),
    ) + tuple(
        torch.randn(
            edges, 2, 2 * (lmax - m + 1) * channels, device="cuda", requires_grad=True
        )
        for m in range(1, mmax + 1)
    )
    parts = [raw[0][:, extras:].reshape(edges, lmax + 1, channels)]
    for m, value in enumerate(raw[1:], 1):
        width = (lmax - m + 1) * channels
        positive = value[:, 0, :width] - value[:, 1, width:]
        negative = value[:, 1, :width] + value[:, 0, width:]
        parts.extend(
            (
                positive.reshape(edges, -1, channels),
                negative.reshape(edges, -1, channels),
            )
        )
    reference = torch.cat(parts, dim=1), raw[0][:, :extras]
    actual = recombine_sectors(
        raw, lmax=lmax, mmax=mmax, channels=channels, extra_channels=extras
    )
    cotangents = tuple(torch.randn_like(value, requires_grad=True) for value in actual)
    directions = tuple(torch.randn_like(value) for value in raw)

    def differentiate(outputs):
        first = torch.autograd.grad(outputs, raw, cotangents, create_graph=True)
        second = torch.autograd.grad(first, cotangents, directions)
        return first, second

    first, second = differentiate(actual)
    expected_first, expected_second = differentiate(reference)
    adjoint = recombine_sectors_adjoint(
        *cotangents, lmax=lmax, mmax=mmax, channels=channels, extra_channels=extras
    )
    for result, expected in zip(
        (*actual, *first, *second, *adjoint),
        (*reference, *expected_first, *expected_second, *expected_first),
    ):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["merge", "recombine"])
def test_sector_assembly_rejects_cpu_inputs_before_launch(operation):
    if operation == "merge":
        sectors = (torch.empty(2, 6), torch.empty(2, 2, 4))
        call = lambda: merge_sectors(sectors, lmax=2, mmax=1, channels=2)
    else:
        raw = (torch.empty(2, 7), torch.empty(2, 2, 8))
        call = lambda: recombine_sectors(
            raw, lmax=2, mmax=1, channels=2, extra_channels=1
        )
    with pytest.raises(ValueError, match="CUDA FP32"):
        call()


@pytest.mark.parametrize(
    "raw",
    [
        (),
        (torch.empty(2, 3, 2), torch.empty(2, 2, 8)),
        (torch.empty(2, 6), torch.empty(2, 2, 8)),
        (torch.empty(2, 7), torch.empty(2, 4, 4)),
    ],
)
def test_recombination_rejects_inconsistent_raw_shapes(raw):
    with pytest.raises(ValueError, match="sector"):
        recombine_sectors(raw, lmax=2, mmax=1, channels=2, extra_channels=1)


@pytest.mark.cuda
@pytest.mark.parametrize("used_output", ["coefficients", "extras"])
def test_recombination_strided_inputs_and_unused_output(used_output):
    generator = torch.Generator(device="cuda").manual_seed(113)
    edges, lmax, mmax, channels, extras = 3, 2, 1, 3, 2
    raw = tuple(
        (
            torch.randn(
                (*shape[:-1], 2 * shape[-1]), generator=generator, device="cuda"
            )
            * 0.2
        )[..., ::2].requires_grad_()
        for shape in ((edges, extras + (lmax + 1) * channels), (edges, 2, 4 * channels))
    )
    reference_raw = tuple(value.detach().double().requires_grad_() for value in raw)
    actual = recombine_sectors(
        raw, lmax=lmax, mmax=mmax, channels=channels, extra_channels=extras
    )
    scalar, pair = reference_raw
    half = 2 * channels
    expected = (
        torch.cat(
            (
                scalar[:, extras:].reshape(edges, lmax + 1, channels),
                (pair[:, 0, :half] - pair[:, 1, half:]).reshape(edges, 2, channels),
                (pair[:, 1, :half] + pair[:, 0, half:]).reshape(edges, 2, channels),
            ),
            dim=1,
        ),
        scalar[:, :extras],
    )
    slot = 0 if used_output == "coefficients" else 1
    directions = tuple(torch.ones_like(value) * 0.1 for value in raw)

    def derivatives(outputs, values, directions):
        first = torch.autograd.grad(
            outputs[slot].square().sum(), values, create_graph=True, allow_unused=True
        )
        first = tuple(
            value * 0 if gradient is None else gradient
            for value, gradient in zip(values, first, strict=True)
        )
        second = torch.autograd.grad(first, values, directions, allow_unused=True)
        second = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for value, gradient in zip(values, second, strict=True)
        )
        return *outputs, *first, *second

    observed = derivatives(actual, raw, directions)
    reference = derivatives(
        expected, reference_raw, tuple(value.double() for value in directions)
    )
    for value, target in zip(observed, reference, strict=True):
        torch.testing.assert_close(value.double(), target, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
def test_empty_sector_dimensions():
    lmax, mmax, channels, extras = 5, 5, 7, 3
    raw = (
        torch.empty(
            0, extras + (lmax + 1) * channels, device="cuda", requires_grad=True
        ),
    ) + tuple(
        torch.empty(
            0, 2, 2 * (lmax - m + 1) * channels, device="cuda", requires_grad=True
        )
        for m in range(1, mmax + 1)
    )
    representation, extra = recombine_sectors(
        raw, lmax=lmax, mmax=mmax, channels=channels, extra_channels=extras
    )
    assert representation.shape == (0, (lmax + 1) ** 2, channels)
    assert extra.shape == (0, extras)
    first = torch.autograd.grad(
        representation.square().sum() + extra.square().sum(), raw, create_graph=True
    )
    second = torch.autograd.grad(sum(x.sum() for x in first), raw)
    for value, grad, grad2 in zip(raw, first, second):
        assert value.shape == grad.shape == grad2.shape
    physical = (
        torch.empty(0, (lmax + 1) * channels, device="cuda", requires_grad=True),
    ) + tuple(
        torch.empty(0, 2, (lmax - m + 1) * channels, device="cuda", requires_grad=True)
        for m in range(1, mmax + 1)
    )
    merged = merge_sectors(physical, lmax=lmax, mmax=mmax, channels=channels)
    gradients = torch.autograd.grad(merged.square().sum(), physical, create_graph=True)
    second = torch.autograd.grad(sum(x.sum() for x in gradients), physical)
    for value, grad, grad2 in zip(physical, gradients, second):
        assert value.shape == grad.shape == grad2.shape


@pytest.mark.parametrize("lmax,mmax", [(3, 4), (1, 2), (1, -1)])
def test_layout_rejects_unsupported_orders(lmax, mmax):
    with pytest.raises(ValueError, match="sector assembly"):
        layout(lmax, mmax)


@pytest.mark.parametrize(
    "sectors",
    [
        (),
        (torch.empty(2, 6),),
        (torch.empty(2, 6), torch.empty(2, 3, 4)),
        (torch.empty(2, 6), torch.empty(3, 2, 4)),
        (torch.empty(2, 6), torch.empty(2, 2, 3, 2)),
    ],
)
def test_merge_rejects_inconsistent_sector_shapes(sectors):
    with pytest.raises(ValueError, match="sector"):
        merge_sectors(sectors, lmax=2, mmax=1, channels=2)


@pytest.mark.cuda
@pytest.mark.parametrize("form", ["flat", "logical", "planes", "strided"])
@pytest.mark.parametrize("edges", [0, 4])
def test_merge_sector_forms_preserve_values_and_derivative_shapes(form, edges):
    generator = torch.Generator(device="cuda").manual_seed(77)
    lmax, mmax, channels = 3, 2, 5
    inputs = []
    for order in range(mmax + 1):
        width = lmax + 1 - order
        if form == "flat":
            shape = (
                (edges, width * channels)
                if order == 0
                else (edges, 2, width * channels)
            )
        elif form == "planes" and order > 0:
            shape = (edges, 2, width, channels)
        else:
            shape = (edges, width if order == 0 else 2 * width, channels)
        if form == "strided":
            shape = (*shape[:-1], 2 * shape[-1])
        value = torch.randn(shape, generator=generator, device="cuda") * 0.2
        if form == "strided":
            value = value[..., ::2]
        inputs.append(value.requires_grad_())
    references = tuple(value.detach().double().requires_grad_() for value in inputs)
    actual = merge_sectors(inputs, lmax=lmax, mmax=mmax, channels=channels)
    expected = torch.cat(
        tuple(
            value.reshape(
                edges, (lmax + 1 - order) * (1 if order == 0 else 2), channels
            )
            for order, value in enumerate(references)
        ),
        dim=1,
    )
    directions = tuple(torch.ones_like(value) * 0.1 for value in inputs)

    def derivatives(output, values, directions):
        loss = (output.sin() + 0.3 * output.square()).sum()
        first = torch.autograd.grad(loss, values, create_graph=True)
        second = torch.autograd.grad(first, values, directions)
        return output, *first, *second

    observed = derivatives(actual, inputs, directions)
    reference = derivatives(
        expected, references, tuple(value.double() for value in directions)
    )
    for value, target in zip(observed, reference, strict=True):
        torch.testing.assert_close(value.double(), target, rtol=2e-5, atol=2e-6)
