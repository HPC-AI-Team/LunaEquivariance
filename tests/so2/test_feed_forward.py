"""Coefficient and spherical-grid feed-forward maps and training gradients."""

import pytest
import torch
import torch.nn.functional as F

from LunaEquivariance.so2 import feed_forward
from LunaEquivariance.so2.feed_forward.interface import (
    StreamedGridProduct,
    fused_grid_product_grid_major,
)
from LunaEquivariance.so2.feed_forward.kernels.forward import (
    project_after_grid,
    project_before_grid,
)
from tests.so2.checks import (
    assert_directional_derivatives,
    directional_derivatives,
    output_derivatives,
)
from tests.so2.reference import ffn as ffn_reference
from tests.so2.reference.ffn import feed_forward as feed_forward_reference
from tests.testing_utils import assert_reference_close


def coefficient_reference(inputs, parameters, lmax):
    scalar_weight, scalar_bias, first_weight, first_bias, second_weight, second_bias = (
        parameters
    )
    hidden_by_degree = [
        F.linear(
            inputs[:, degree * degree : (degree + 1) ** 2],
            first_weight[degree],
        )
        for degree in range(lmax + 1)
    ]
    activated = [F.silu(hidden_by_degree[0] + first_bias)]
    if lmax:
        gates = torch.sigmoid(
            F.silu(
                F.linear(inputs[:, 0], scalar_weight, scalar_bias)
            )
        ).reshape(inputs.shape[0], lmax, first_bias.numel())
        activated.extend(
            hidden_by_degree[degree] * gates[:, degree - 1 : degree]
            for degree in range(1, lmax + 1)
        )
    return torch.cat(
        [
            F.linear(
                value,
                second_weight[degree],
                second_bias if degree == 0 else None,
            )
            for degree, value in enumerate(activated)
        ],
        dim=1,
    )


def random_direction(value, strided):
    if strided:
        return torch.randn(
            *value.shape[:-1],
            2 * value.shape[-1],
            device=value.device,
        )[..., ::2]
    return torch.randn_like(value)


def zero_unused(gradients, inputs):
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for gradient, value in zip(gradients, inputs, strict=True)
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,cin,hidden,cout", [(0, 7, 13, 5), (1, 5, 7, 11), (2, 9, 6, 7)]
)
@pytest.mark.parametrize(
    "mode", ["full", "weights_only", "partially_frozen", "input_only"]
)
@pytest.mark.parametrize("strided", [False, True])
def test_coefficient_ffn_forward_backward_double_backward(
    lmax, cin, hidden, cout, mode, strided
):
    from LunaEquivariance.so2.feed_forward.interface import coefficient_ffn

    torch.manual_seed(2026)
    shapes = [
        (lmax * hidden, cin),
        (lmax * hidden,),
        (lmax + 1, hidden, cin),
        (hidden,),
        (lmax + 1, cout, hidden),
        (cout,),
    ]
    parameters = [
        (0.1 * torch.randn(shape, device="cuda")).requires_grad_() for shape in shapes
    ]
    inputs = torch.randn(
        3, (lmax + 1) ** 2, cin, device="cuda", requires_grad=mode != "weights_only"
    )
    if mode == "input_only":
        for value in parameters:
            value.requires_grad_(False)
    elif mode == "partially_frozen":
        for index in (0, 3, 4):
            parameters[index].requires_grad_(False)
    scalar_weight, scalar_bias, first_weight, first_bias, second_weight, second_bias = (
        parameters
    )
    degree = torch.tensor(
        [degree for degree in range(lmax + 1) for _ in range(2 * degree + 1)],
        dtype=torch.int64,
        device="cuda",
    )
    actual = coefficient_ffn(
        inputs,
        scalar_weight,
        scalar_bias,
        first_weight,
        degree,
        first_bias,
        second_weight,
        degree,
        second_bias,
        lmax=lmax,
    )
    expected = coefficient_reference(inputs, parameters, lmax)
    active = tuple(value for value in (inputs, *parameters) if value.requires_grad)
    cotangent = random_direction(actual, strided).requires_grad_()
    actual_first = torch.autograd.grad(
        actual, active, cotangent, create_graph=True, allow_unused=True
    )
    expected_first = torch.autograd.grad(
        expected, active, cotangent, create_graph=True, allow_unused=True
    )
    directions = tuple(random_direction(value, strided) for value in active)

    def differentiate_contraction(gradients):
        contraction = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions, strict=True)
            if gradient is not None and gradient.requires_grad
        )
        return torch.autograd.grad(contraction, (*active, cotangent), allow_unused=True)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for observed, wanted in zip(
        zero_unused(actual_first, active),
        zero_unused(expected_first, active),
        strict=True,
    ):
        torch.testing.assert_close(observed, wanted, atol=1e-5, rtol=1e-5)
    actual_second = differentiate_contraction(actual_first)
    expected_second = differentiate_contraction(expected_first)
    assert actual_second[-1] is not None
    for observed, wanted in zip(
        zero_unused(actual_second, (*active, cotangent)),
        zero_unused(expected_second, (*active, cotangent)),
        strict=True,
    ):
        torch.testing.assert_close(observed, wanted, atol=1e-5, rtol=1e-5)


def parameters(lmax, channels, train=False):
    shapes = [
        (lmax + 1, channels),
        (channels,),
        (lmax * channels, channels),
        (lmax * channels,),
        (lmax + 1, channels, channels),
        (channels,),
        (lmax + 1, channels, channels),
        (channels,),
    ]
    values = [
        (torch.randn(shape, device="cuda") * 0.1).requires_grad_(train)
        for shape in shapes
    ]
    degrees = torch.tensor(
        [degree for degree in range(lmax + 1) for _ in range(2 * degree + 1)],
        device="cuda",
    )
    return values, degrees


def residual_ffn_reference(x, values, degrees, lmax):
    nw, nb, gw, gb, fw, fb, sw, sb = values
    scalar = x[:, :1] - x[:, :1].mean(-1, keepdim=True)
    centered = torch.cat((scalar, x[:, 1:]), 1)
    variance = sum(
        centered[:, d * d : (d + 1) ** 2].square().mean((1, 2)) for d in range(lmax + 1)
    ) / (lmax + 1)
    normalized = centered * torch.rsqrt(variance + 1e-5)[:, None, None]
    normalized = normalized * nw[degrees][None]
    normalized = torch.cat((normalized[:, :1] + nb, normalized[:, 1:]), 1)
    hidden = torch.einsum("nrc,rhc->nrh", normalized, fw[degrees])
    gates = F.linear(normalized[:, 0], gw, gb)
    gates = torch.sigmoid(F.silu(gates)).reshape(
        x.shape[0], lmax, hidden.shape[-1]
    )
    activated = torch.cat(
        (
            F.silu(hidden[:, :1] + fb),
            *[
                hidden[:, d * d : (d + 1) ** 2] * gates[:, d - 1 : d]
                for d in range(1, lmax + 1)
            ],
        ),
        1,
    )
    raw = torch.einsum("nrh,rch->nrc", activated, sw[degrees])
    return torch.cat((raw[:, :1] + sb, raw[:, 1:]), 1) + x


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,channels", [(1, 32), (2, 33)])
@pytest.mark.parametrize("strided", (False, True))
def test_training_ffn_bw1_bw2(lmax, channels, strided):
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    values, degrees = parameters(lmax, channels, train=True)
    x = torch.randn(3, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
    inputs = (x, *values)
    actual = norm_residual_ffn(x, *values, degrees, lmax=lmax, epsilon=1e-5)
    expected = residual_ffn_reference(x, values, degrees, lmax)
    grad = (
        torch.randn(
            actual.shape[0], actual.shape[2], actual.shape[1], device="cuda"
        ).transpose(1, 2)
        if strided
        else torch.randn_like(actual)
    )
    actual_first = torch.autograd.grad(actual, inputs, grad, create_graph=True)
    expected_first = torch.autograd.grad(expected, inputs, grad, create_graph=True)
    directions = tuple(
        torch.randn(*value.shape[:-1], value.shape[-1] * 2, device="cuda")[..., ::2]
        if strided
        else torch.randn_like(value)
        for value in inputs
    )
    actual_second = torch.autograd.grad(actual_first, inputs, directions)
    expected_directional = sum(
        (g * v).sum() for g, v in zip(expected_first, directions) if g.requires_grad
    )
    expected_second = torch.autograd.grad(
        expected_directional, inputs, allow_unused=True
    )
    expected_second = tuple(
        torch.zeros_like(x) if g is None else g for x, g in zip(inputs, expected_second)
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for a, b in zip(actual_first, expected_first):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)
    for a, b in zip(actual_second, expected_second):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,channels", [(2, 33), (3, 32)])
def test_ffn_no_grad_matches_training_forward(lmax, channels):
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    values, degrees = parameters(lmax, channels, train=True)
    x = torch.randn(3, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
    actual = norm_residual_ffn(
        x,
        *values,
        degrees,
        lmax=lmax,
        epsilon=1e-5,
    )
    with torch.no_grad():
        detached = norm_residual_ffn(
            x,
            *values,
            degrees,
            lmax=lmax,
            epsilon=1e-5,
        )
    assert actual.requires_grad
    assert not detached.requires_grad
    torch.testing.assert_close(detached, actual.detach(), atol=0, rtol=0)


@pytest.mark.cuda
def test_scalar_only_ffn_bias_gradients():
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    lmax, channels = 0, 8
    values, degrees = parameters(lmax, channels, train=True)
    x = torch.randn(3, 1, channels, device="cuda", requires_grad=True)
    actual = norm_residual_ffn(x, *values, degrees, lmax=lmax, epsilon=1e-5)
    expected = residual_ffn_reference(x, values, degrees, lmax)
    cotangent = torch.randn(1, 1, channels, device="cuda").expand_as(actual)
    targets = (x, values[5], values[7])
    result = torch.autograd.grad(actual, targets, cotangent)
    baseline = torch.autograd.grad(expected, targets, cotangent)
    for observed, reference_gradient in zip(result, baseline, strict=True):
        torch.testing.assert_close(observed, reference_gradient, atol=1e-4, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ("full", "partially_frozen", "input_only"))
@pytest.mark.parametrize("strided", (False, True))
def test_ffn_mixed_derivatives_and_saved_grad(mode, strided):
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    torch.manual_seed(2026)
    values, degrees = parameters(2, 33, train=True)
    if mode == "partially_frozen":
        for index in (0, 3, 4):
            values[index].requires_grad_(False)
    x = torch.randn(3, 9, 33, device="cuda", requires_grad=True)
    active = tuple(value for value in (x, *values) if value.requires_grad)
    actual = norm_residual_ffn(x, *values, degrees, lmax=2, epsilon=1e-5)
    expected = residual_ffn_reference(x, values, degrees, 2)
    cotangent = (
        torch.randn(3, 33, 9, device="cuda").transpose(1, 2)
        if strided
        else torch.randn_like(actual)
    ).requires_grad_()
    targets = (x,) if mode == "input_only" else active
    first = torch.autograd.grad(actual, targets, cotangent, create_graph=True)
    reference_first = torch.autograd.grad(
        expected, targets, cotangent, create_graph=True
    )

    # Inspect BW1 itself: neither its tensor nor a view of its storage may keep
    # the contiguous incoming cotangent alive. Do not fix the number of other
    # saved tensors; those may change with independent state improvements.
    backward = first[0].grad_fn
    assert type(backward).__name__ == "EquivariantFFNBackwardBackward"
    saved_storages = {
        value.untyped_storage().data_ptr() for value in backward.saved_tensors
    }
    assert cotangent.untyped_storage().data_ptr() not in saved_storages

    directions = tuple(
        torch.randn(*value.shape[:-1], 2 * value.shape[-1], device="cuda")[..., ::2]
        if strided
        else torch.randn_like(value)
        for value in targets
    )
    second = torch.autograd.grad(
        first,
        (*active, cotangent),
        directions,
        allow_unused=True,
    )
    reference_second = torch.autograd.grad(
        reference_first,
        (*active, cotangent),
        directions,
        allow_unused=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for observed, wanted in zip(first, reference_first, strict=True):
        torch.testing.assert_close(observed, wanted, atol=1e-4, rtol=1e-5)
    assert second[-1] is not None  # The incoming-cotangent D2 slot must survive.
    for observed, wanted, target in zip(
        second,
        reference_second,
        (*active, cotangent),
        strict=True,
    ):
        observed = torch.zeros_like(target) if observed is None else observed
        wanted = torch.zeros_like(target) if wanted is None else wanted
        torch.testing.assert_close(observed, wanted, atol=1e-4, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("rows,channels", [(1, 33), (16, 128), (25, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_epilogue_layout_and_input_gradient(rows, channels, dtype):
    from LunaEquivariance.so2.feed_forward.kernels.forward import epilogue_forward as forward
    from LunaEquivariance.so2.feed_forward.kernels.backward import input_vjp_from_shape

    for nodes in (1, 7, 16, 75, 291):
        raw = torch.randn(rows, nodes, channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)
        residual = torch.randn(nodes, rows, channels, device="cuda", dtype=dtype)
        expected = raw.float().permute(1, 0, 2) + residual.float()
        expected[:, 0] += bias.float()
        torch.testing.assert_close(
            forward(raw, bias, residual), expected.to(dtype), atol=0, rtol=0
        )
        gradient = torch.randn_like(residual)
        torch.testing.assert_close(
            input_vjp_from_shape(gradient, rows),
            gradient.permute(1, 0, 2).contiguous(),
            atol=0,
            rtol=0,
        )


@pytest.mark.cuda
def test_epilogue_reuses_compiled_code_for_different_node_counts():
    from LunaEquivariance.so2.feed_forward.kernels.forward import epilogue_kernel as forward_kernel
    from LunaEquivariance.so2.feed_forward.kernels.backward import input_vjp_kernel

    rows, channels, block = 9, 96, 256
    hashes = []
    for nodes in (131, 137, 1, 16, 149):
        raw = torch.empty(rows, nodes, channels, device="cuda")
        bias = torch.empty(channels, device="cuda")
        residual = torch.empty(nodes, rows, channels, device="cuda")
        output = torch.empty_like(residual)
        raw_gradient = torch.empty_like(raw)
        elements = output.numel()
        options = dict(
            nodes=nodes,
            rows=rows,
            channels=channels,
            block=block,
            num_warps=4,
            grid=((elements + block - 1) // block,),
        )
        forward = forward_kernel.warmup(
            raw, bias, residual, output, elements, **options
        )
        backward = input_vjp_kernel.warmup(residual, raw_gradient, elements, **options)
        hashes.append((forward.hash, backward.hash))
    assert len(set(hashes)) == 1


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["expanded", "permuted", "sliced"])
def test_epilogue_accepts_strided_cotangents(layout):
    from LunaEquivariance.so2.feed_forward.kernels.backward import input_vjp_from_shape

    nodes, rows, channels = 3, 9, 33
    if layout == "expanded":
        gradient = torch.ones((), device="cuda").expand(nodes, rows, channels)
    elif layout == "permuted":
        gradient = torch.randn(rows, nodes, channels, device="cuda").permute(1, 0, 2)
    else:
        gradient = torch.randn(nodes, rows, 2 * channels, device="cuda")[:, :, ::2]
    actual = input_vjp_from_shape(gradient, rows)
    torch.testing.assert_close(actual, gradient.permute(1, 0, 2), atol=0, rtol=0)


def rectangular_reference(x, values, degree, lmax):
    nw, nb, gw, gb, fw, fb, sw, sb = values
    centered = torch.cat((x[:, :1] - x[:, :1].mean(-1, keepdim=True), x[:, 1:]), 1)
    variance = sum(
        centered[:, d * d : (d + 1) ** 2].square().mean((1, 2)) for d in range(lmax + 1)
    ) / (lmax + 1)
    norm = centered * torch.rsqrt(variance + 1e-5)[:, None, None] * nw[degree][None]
    norm = torch.cat((norm[:, :1] + nb, norm[:, 1:]), 1)
    hidden = torch.einsum("nrc,rhc->nrh", norm, fw[degree])
    gates = torch.sigmoid(
        F.silu(F.linear(norm[:, 0], gw, gb))
    )
    gates = gates.reshape(x.shape[0], lmax, hidden.shape[-1])
    act = torch.cat(
        (
            F.silu(hidden[:, :1] + fb),
            *[
                hidden[:, d * d : (d + 1) ** 2] * gates[:, d - 1 : d]
                for d in range(1, lmax + 1)
            ],
        ),
        1,
    )
    result = torch.einsum("nrh,rch->nrc", act, sw[degree])
    return x + torch.cat((result[:, :1] + sb, result[:, 1:]), 1)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,channels,hidden",
    [(0, 7, 13), (1, 17, 31), (2, 33, 16), (4, 7, 13), (6, 129, 7)],
)
@pytest.mark.parametrize("input_only", [False, True])
def test_rectangular_ffn_forward_backward_double_backward(
    lmax, channels, hidden, input_only
):
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    torch.manual_seed(2026)
    shapes = [
        (lmax + 1, channels),
        (channels,),
        (lmax * hidden, channels),
        (lmax * hidden,),
        (lmax + 1, hidden, channels),
        (hidden,),
        (lmax + 1, channels, hidden),
        (channels,),
    ]
    values = [
        (torch.randn(shape, device="cuda") * 0.1).requires_grad_() for shape in shapes
    ]
    degree = torch.tensor(
        [d for d in range(lmax + 1) for _ in range(2 * d + 1)], device="cuda"
    )
    x = torch.randn(3, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
    inputs = (x, *values)
    actual = norm_residual_ffn(x, *values, degree, lmax=lmax, epsilon=1e-5)
    expected = rectangular_reference(x, values, degree, lmax)
    cotangent = (
        torch.randn(3, channels, (lmax + 1) ** 2, device="cuda")
        .transpose(1, 2)
        .requires_grad_()
    )
    differentiated = (x,) if input_only else inputs
    first = torch.autograd.grad(actual, differentiated, cotangent, create_graph=True)
    ref_first = torch.autograd.grad(
        expected, differentiated, cotangent, create_graph=True, allow_unused=True
    )
    ref_first = tuple(
        torch.zeros_like(v) if g is None else g
        for g, v in zip(ref_first, differentiated)
    )
    directions = tuple(torch.randn_like(v) for v in differentiated)
    second = torch.autograd.grad(
        first, (*inputs, cotangent), directions, allow_unused=True
    )
    objective = sum(
        (g * v).sum() for g, v in zip(ref_first, directions) if g.requires_grad
    )
    ref_second = torch.autograd.grad(objective, (*inputs, cotangent), allow_unused=True)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for value, target in zip(first, ref_first):
        torch.testing.assert_close(value, target, atol=1e-4, rtol=1e-5)
    for value, target, original in zip(second, ref_second, (*inputs, cotangent)):
        value = torch.zeros_like(original) if value is None else value
        target = torch.zeros_like(original) if target is None else target
        torch.testing.assert_close(value, target, atol=1e-4, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("expanded", (False, True))
def test_grid_major_product_strided_derivatives(expanded):
    grids, nodes, channels = 5, 3, 8
    pair = torch.randn(grids, nodes, 2 * channels, device="cuda", requires_grad=True)
    gate = torch.randn(nodes, 1, channels, device="cuda", requires_grad=True)
    actual = fused_grid_product_grid_major(pair, gate)
    expected = pair[..., :channels] * pair[..., channels:] * gate[:, 0].sigmoid()[None]
    if expanded:
        cotangent = torch.randn(1, nodes, channels, device="cuda").expand(grids, -1, -1)
    else:
        cotangent = torch.randn(grids, channels, nodes, device="cuda").transpose(1, 2)
    cotangent.requires_grad_()
    directions = (
        torch.randn(grids, 2 * channels, nodes, device="cuda")
        .transpose(1, 2)
        .requires_grad_(),
        torch.randn(nodes, 1, 2 * channels, device="cuda")[..., ::2].requires_grad_(),
    )
    targets = (pair, gate)
    first = torch.autograd.grad(actual, targets, cotangent, create_graph=True)
    ref_first = torch.autograd.grad(expected, targets, cotangent, create_graph=True)
    second_targets = (cotangent, *targets)
    second = torch.autograd.grad(first, second_targets, directions, create_graph=True)
    ref_second = torch.autograd.grad(ref_first, second_targets, directions, create_graph=True)
    for value, reference in zip(
        (actual, *first, *second), (expected, *ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
    third_targets = (*second_targets, *directions)
    third = torch.autograd.grad(sum(value.sum() for value in second), third_targets)
    ref_third = torch.autograd.grad(sum(value.sum() for value in ref_second), third_targets)
    for value, reference in zip(third, ref_third, strict=True):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)


def grid_tensor(generator, *shape):
    return (
        torch.randn(*shape, generator=generator, dtype=torch.float64) * 0.2
    ).requires_grad_()


def grid_derivatives(output, leaves, cotangent, directions):
    first = torch.autograd.grad(output, leaves, cotangent, create_graph=True)
    contraction = sum(
        (value * direction).sum() for value, direction in zip(first, directions)
    )
    second = torch.autograd.grad(contraction, (*leaves, cotangent), allow_unused=True)
    return (output, *first, *second)


@pytest.mark.cuda
def test_streamed_grid_product_derivatives():
    torch.manual_seed(17)
    rows, nodes, channels, grids = 25, 256, 512, 196

    def random(*shape):
        return torch.randn(shape, device="cuda") * 0.1

    pair = random(rows, nodes, 2 * channels).requires_grad_()
    gate = random(nodes, 1, channels).requires_grad_()
    to_grid = random(grids, rows)
    from_grid = random(grids, rows).T
    seed = random(nodes, rows, channels).transpose(0, 1).requires_grad_()
    directions = (random(nodes, rows, 2 * channels).transpose(0, 1), random(*gate.shape))
    actual = StreamedGridProduct.apply(pair, gate, to_grid, from_grid)

    ref_pair, ref_gate = (x.detach().double().requires_grad_() for x in (pair, gate))
    first, second = (to_grid.double() @ ref_pair.flatten(1)).view(grids, nodes, -1).chunk(2, -1)
    product = first * second * ref_gate[:, 0].sigmoid()
    expected = (from_grid.double() @ product.flatten(1)).view(rows, nodes, channels)
    actual_values = grid_derivatives(actual, (pair, gate), seed, directions)
    expected_values = grid_derivatives(
        expected, (ref_pair, ref_gate), seed.detach().double().requires_grad_(),
        tuple(x.double() for x in directions),
    )
    for value, reference in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(value.double(), reference, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("bias", [None, "vector", "broadcast"])
@pytest.mark.parametrize("node_major", [False, True])
def test_complete_grid_nonlinearity_and_all_parameter_derivatives(
    truncated, bias, node_major
):
    generator = torch.Generator().manual_seed(17)
    # An explicit subset of angular components is valid as long as G/H use it.
    index = torch.tensor(
        [0, 1, 1, 1, 2, 2] if truncated else [0, 1, 1, 1, 2, 2, 2, 2, 2]
    )
    components, nodes = len(index), 2
    x = grid_tensor(generator, nodes, components, 3)
    w1, a1 = grid_tensor(generator, 3, 5, 3), grid_tensor(generator, 4, 5)
    a2, w2 = grid_tensor(generator, 6, 4), grid_tensor(generator, 3, 2, 6)
    bias_prefix = (1, 1) if bias == "broadcast" else ()
    b1 = grid_tensor(generator, *bias_prefix, 5) if bias else None
    b2 = grid_tensor(generator, *bias_prefix, 2) if bias else None
    g = grid_tensor(generator, 13, components).detach()
    h = grid_tensor(generator, components, 13).detach()

    # Original graph: degree map, expansion, channel map, nonlinearity,
    # channel map, inverse expansion, degree map.
    hidden = torch.einsum("nki,koi->nko", x, w1[index])
    if bias:
        hidden = torch.cat((hidden[:, :1] + b1, hidden[:, 1:]), dim=1)
    grid = torch.einsum("gk,nkc->ngc", g, hidden)
    grid = F.linear(F.linear(grid, a1).tanh(), a2)
    hidden = torch.einsum("kg,ngc->nkc", h, grid)
    original = torch.einsum("nki,koi->nko", hidden, w2[index])
    if bias:
        original = torch.cat((original[:, :1] + b2, original[:, 1:]), dim=1)

    pair = project_before_grid(
        x if node_major else x.permute(1, 0, 2),
        w1,
        a1,
        index,
        b1,
        input_node_major=node_major,
    )
    grid = (g @ pair.flatten(1)).view(13, nodes, 4).tanh()
    coefficients = (h @ grid.flatten(1)).view(components, nodes, 4)
    actual = project_after_grid(
        coefficients.permute(1, 0, 2) if node_major else coefficients,
        a2,
        w2,
        index,
        b2,
        input_node_major=node_major,
    ).permute(1, 0, 2)
    leaves = (x, w1, a1, a2, w2, *((b1, b2) if bias else ()))
    cotangent = grid_tensor(generator, *actual.shape)
    directions = tuple(grid_tensor(generator, *value.shape).detach() for value in leaves)
    expected_values = grid_derivatives(original, leaves, cotangent, directions)
    actual_values = grid_derivatives(actual, leaves, cotangent, directions)
    for expected, result in zip(expected_values, actual_values):
        if expected is None:
            assert result is None
        else:
            torch.testing.assert_close(result, expected, rtol=2e-11, atol=2e-12)


def test_reordering_paper_feature_derivative_identities():
    generator = torch.Generator().manual_seed(19)
    index = torch.tensor([0, 1, 1, 1])
    x = grid_tensor(generator, 2, 4, 3)
    w, a = grid_tensor(generator, 2, 5, 3), grid_tensor(generator, 2, 5)
    g = grid_tensor(generator, 7, 4).detach()
    coefficients = project_before_grid(x, w, a, index)
    u = (g @ coefficients.flatten(1)).view(7, 2, 2)
    bar_u = grid_tensor(generator, *u.shape)
    bar_x = torch.autograd.grad(u, x, bar_u, create_graph=True)[0]
    p = grid_tensor(generator, *x.shape).detach()
    second = torch.autograd.grad(bar_x, bar_u, p)[0]

    # The first derivative reverses the coefficient-space composed map.
    b = a @ w
    returned = (g.T @ bar_u.flatten(1)).view(4, 2, 2)
    expected_first = torch.bmm(returned, b[index]).permute(1, 0, 2)
    # The second derivative's incoming direction follows the forward map.
    expected_second = (g @ project_before_grid(p, w, a, index).flatten(1)).view_as(u)
    torch.testing.assert_close(bar_x, expected_first)
    torch.testing.assert_close(second, expected_second)


def test_composed_maps_use_current_original_weights():
    generator = torch.Generator().manual_seed(23)
    x = grid_tensor(generator, 1, 4, 2)
    index = torch.tensor([0, 1, 1, 1])
    weight, channel = grid_tensor(generator, 2, 3, 2), grid_tensor(generator, 2, 3)
    before = project_before_grid(x, weight, channel, index)
    with torch.no_grad():
        channel.mul_(2)
    after = project_before_grid(x, weight, channel, index)
    torch.testing.assert_close(after, 2 * before)


def tensor(shape, *, device="cpu", dtype=torch.float64):
    return (0.2 * torch.randn(shape, device=device, dtype=dtype)).requires_grad_()


def case(name, *, device="cpu", dtype=torch.float64, lmax=1):
    """Small unequal widths expose swapped channels and scalar bias placement."""
    n, rows, cin, hidden, cout = 2, (lmax + 1) ** 2, 3, 5, 2

    def make(*shape):
        return tensor(shape, device=device, dtype=dtype)

    degree = torch.tensor(
        [degree for degree in range(lmax + 1) for _ in range(2 * degree + 1)],
        device=device,
    )
    x = make(n, rows, cin)
    if name == "coefficient_channel_linear":
        values = (x, make(lmax + 1, cout, cin), make(1, 1, cout))
        return values, lambda function, args: function(*args[:2], degree, args[2])
    if name == "project_before_grid":
        values = (x, make(lmax + 1, hidden, cin), make(cout, hidden), make(hidden))
        return values, lambda function, args: function(*args[:3], degree, args[3])
    if name == "project_after_grid":
        values = (
            x.transpose(0, 1).detach().requires_grad_(),
            make(hidden, cin),
            make(lmax + 1, cout, hidden),
            make(cout),
        )
        return values, lambda function, args: function(*args[:3], degree, args[3])
    if name == "coefficient_ffn":
        values = (
            x,
            make(lmax * hidden, cin),
            make(lmax * hidden),
            make(lmax + 1, hidden, cin),
            make(hidden),
            make(lmax + 1, cout, hidden),
            make(cout),
        )
        return values, lambda function, a: function(
            *a[:4],
            degree,
            a[4],
            a[5],
            degree,
            a[6],
            lmax=lmax,
        )
    if name == "norm_residual_ffn":
        values = (
            x,
            make(lmax + 1, cin),
            make(cin),
            make(lmax * hidden, cin),
            make(lmax * hidden),
            make(lmax + 1, hidden, cin),
            make(hidden),
            make(lmax + 1, cin, hidden),
            make(cin),
        )
        return values, lambda function, a: function(*a, degree, lmax=lmax, epsilon=1e-4)
    if name == "grid_ffn":
        grid, first_width, last_width = 7, 4, 6
        values = (
            x,
            make(lmax + 1, first_width, cin),
            make(2 * hidden, first_width),
            make(first_width),
            make(grid, rows),
            make(hidden, cin),
            make(hidden),
            make(rows, grid),
            make(last_width, hidden),
            make(lmax + 1, cout, last_width),
            make(cout),
        )
        return values, lambda function, a: function(
            *a[:3],
            degree,
            *a[3:10],
            degree,
            a[10],
        )
    if name == "merge_scalar_branch":
        values = (make(rows, n, cout), make(n, 1, hidden), make(lmax + 1, cout, hidden))
        return values, lambda function, args: function(*args)
    raise AssertionError(name)


NAMES = (
    "coefficient_channel_linear",
    "coefficient_ffn",
    "norm_residual_ffn",
    "grid_ffn",
    "merge_scalar_branch",
    "project_before_grid",
    "project_after_grid",
)


@pytest.mark.parametrize("name", NAMES)
def test_reference_ffn_parts_gradcheck_and_gradgradcheck(name):
    torch.manual_seed(814)
    values, call = case(name)

    def function(*args):
        return call(getattr(ffn_reference, name), args)

    assert torch.autograd.gradcheck(function, values, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, values, fast_mode=True)


@pytest.fixture
def ffn_operators():
    pytest.importorskip("triton")
    from LunaEquivariance.so2.feed_forward.interface import coefficient_ffn, grid_ffn, norm_residual_ffn
    from LunaEquivariance.so2.feed_forward.kernels.forward import (
        coefficient_channel_linear,
        merge_scalar_branch,
        project_after_grid,
        project_before_grid,
    )

    return {
        "coefficient_ffn": coefficient_ffn,
        "grid_ffn": grid_ffn,
        "coefficient_channel_linear": coefficient_channel_linear,
        "norm_residual_ffn": norm_residual_ffn,
        "project_before_grid": project_before_grid,
        "project_after_grid": project_after_grid,
        "merge_scalar_branch": merge_scalar_branch,
    }


@pytest.mark.parametrize(
    "name",
    (
        "coefficient_channel_linear",
        "project_before_grid",
        "project_after_grid",
        "merge_scalar_branch",
    ),
)
def test_pytorch_channel_maps_match_separate_reference_through_training_derivatives(
    name, ffn_operators,
):
    torch.manual_seed(815)
    values, call = case(name)
    copied = tuple(value.detach().clone().requires_grad_() for value in values)
    actual = call(ffn_operators[name], values)
    expected = call(getattr(ffn_reference, name), copied)
    cotangent = torch.randn_like(actual, requires_grad=True)
    directions = tuple(torch.randn_like(value) for value in values)
    actual_values = directional_derivatives(actual, values, cotangent, directions)
    expected_values = directional_derivatives(expected, copied, cotangent, directions)
    for observed, wanted in zip(actual_values, expected_values, strict=True):
        if wanted is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, wanted)


@pytest.mark.cuda
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("lmax", (0, 2))
def test_ffn_parts_against_fp64_reference(name, lmax, ffn_operators):
    torch.manual_seed(816)
    values, call = case(name, device="cuda", dtype=torch.float32, lmax=lmax)
    copied = tuple(value.detach().double().requires_grad_() for value in values)
    actual = call(ffn_operators[name], values)
    expected = call(getattr(ffn_reference, name), copied)
    assert_directional_derivatives(actual, expected, values, copied)


def test_channel_composition_preserves_external_weight_derivatives():
    from LunaEquivariance.so2.feed_forward.kernels.forward import coefficient_channel_linear
    from LunaEquivariance.so2.feed_forward.kernels.forward import project_before_grid

    generator = torch.Generator().manual_seed(73)
    shapes = ((2, 4, 3), (2, 5, 3), (6, 5), (5,))
    parameters = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in shapes
    )
    reference_parameters = tuple(
        x.detach().clone().requires_grad_() for x in parameters
    )
    degrees = torch.tensor([0, 1, 1, 1])
    actual = project_before_grid(*parameters[:3], degrees, parameters[3])
    x, weight, channel_weight, bias = reference_parameters
    intermediate = coefficient_channel_linear(x, weight, degrees, bias)
    expected = F.linear(intermediate, channel_weight)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    first = torch.autograd.grad(actual, parameters, cotangent, create_graph=True)
    reference_first = torch.autograd.grad(
        expected, reference_parameters, cotangent, create_graph=True
    )
    directions = tuple(
        torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in parameters
    )
    second = torch.autograd.grad(first, parameters, directions, allow_unused=True)
    reference_second = torch.autograd.grad(
        reference_first, reference_parameters, directions, allow_unused=True
    )
    for actual_values, expected_values in (
        (first, reference_first),
        (second, reference_second),
    ):
        for value, reference in zip(actual_values, expected_values):
            if reference is None:
                assert value is None
            else:
                torch.testing.assert_close(value, reference)


def parameter(shape, *, device="cpu", dtype=torch.float64):
    return (torch.randn(shape, device=device, dtype=dtype) * 0.15).requires_grad_()


def ffn_problem(lmax, channels, hidden, *, grid=False, normalization=False, device="cpu", dtype=torch.float64):
    torch.manual_seed(341)
    make = lambda shape: parameter(shape, device=device, dtype=dtype)
    rows = (lmax + 1) ** 2
    values = [make((3, rows, channels)), make((lmax+1, hidden, channels)),
              make((lmax+1, channels, hidden))]
    kwargs = dict(input_bias=make((hidden,)), output_bias=make((channels,)),
                  gate_bias=make((hidden if grid else lmax*hidden,)))
    values.append(make((hidden if grid else lmax*hidden, channels)))
    if grid:
        points = rows + 3
        kwargs.update(
            to_grid=make((points, rows)), from_grid=make((rows, points)),
            grid_input_weight=make((2*hidden, hidden)),
            grid_output_weight=make((hidden, hidden)),
        )
        kwargs["scalar_branch"] = (make((2*hidden, channels)), make((2*hidden,)))
    if normalization:
        kwargs.update(
            norm_weight=make((lmax+1, channels)), norm_bias=make((channels,)),
            norm_eps=1e-3, residual=True,
        )
    return values, kwargs


def floating_leaves(args, kwargs):
    leaves = list(args) + list(kwargs.values())
    values = []
    for leaf in leaves:
        if isinstance(leaf, tuple):
            values.extend(leaf)
        elif isinstance(leaf, torch.Tensor):
            values.append(leaf)
    return tuple(v for v in values if v.requires_grad)


def convert(args, kwargs):
    def fp64(value):
        if isinstance(value, torch.Tensor):
            return value.detach().double().requires_grad_(value.requires_grad)
        if isinstance(value, tuple):
            return tuple(fp64(item) for item in value)
        return value
    return [fp64(v) for v in args], {key: fp64(v) for key,v in kwargs.items()}


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("normalization", [False, True])
def test_reference_ffn_complete_gradients(grid, normalization):
    args, kwargs = ffn_problem(1, 3, 4, grid=grid, normalization=normalization)
    output = feed_forward_reference(*args, **kwargs)
    targets = floating_leaves(args,kwargs)
    seed = torch.randn_like(output, requires_grad=True)
    first = torch.autograd.grad(output, targets, seed, create_graph=True)
    second = torch.autograd.grad(first, (*targets,seed), tuple(torch.randn_like(v) for v in first), allow_unused=True)
    assert all(torch.isfinite(value).all() for value in (output,*first,*[v for v in second if v is not None]))


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,channels,hidden", [(1,3,5),(2,8,12),(4,7,13)])
@pytest.mark.parametrize("grid,normalization", [(False,False),(False,True),(True,False),(True,True)])
def test_ffn_forward_backward_double_backward(lmax,channels,hidden,grid,normalization):
    args, kwargs = ffn_problem(lmax,channels,hidden,grid=grid,normalization=normalization,
                               device="cuda",dtype=torch.float32)
    if not normalization:
        args[0] = args[0].transpose(0,1).contiguous().transpose(0,1)
    refs, ref_kwargs = convert(args,kwargs)
    actual = feed_forward(*args, **kwargs)
    expected = feed_forward_reference(*refs, **ref_kwargs)
    targets = floating_leaves(args,kwargs)
    reference_targets = floating_leaves(refs,ref_kwargs)
    seed = torch.randn_like(actual, requires_grad=True)
    directions = tuple(torch.randn_like(v) for v in targets)
    ref_seed = seed.detach().double().requires_grad_()
    for result, wanted in zip(
        output_derivatives(actual,targets,seed,directions),
        output_derivatives(expected,reference_targets,ref_seed,tuple(v.double() for v in directions)),
        strict=True,
    ):
        assert_reference_close(result,wanted)


@pytest.mark.cuda
@pytest.mark.parametrize("expanded", [False, True])
def test_feed_forward_with_frozen_parameters(expanded):
    from tests.so2.reference.ffn import degree_index

    args, kwargs = ffn_problem(2, 8, 12, device="cuda", dtype=torch.float32)
    x, *weights = args
    weights = [weight.detach() for weight in weights]
    biases = {name: bias.detach() for name, bias in kwargs.items()}
    if expanded:
        degree = degree_index(x.shape[1], x.device)
        weights[:2] = [weight[degree] for weight in weights[:2]]
    actual = feed_forward(x, *weights, **biases, input_only=True)
    ref_x = x.detach().double().requires_grad_()
    ref_args, ref_biases = convert(weights, biases)
    expected = feed_forward_reference(ref_x, *ref_args, **ref_biases, input_only=True)
    seed = torch.randn_like(actual)
    assert_reference_close(actual, expected)
    assert_reference_close(
        torch.autograd.grad(actual, x, seed)[0],
        torch.autograd.grad(expected, ref_x, seed.double())[0],
    )


def test_ffn_rejects_incomplete_degrees():
    args,kwargs = ffn_problem(1,3,4)
    args[0] = args[0][:,:3]
    with pytest.raises(ValueError,match="complete degrees"):
        feed_forward(*args,**kwargs)


@pytest.mark.parametrize(("name", "message"), [
    ("grid_input_weight", "provide all four grid parameters together"),
    ("norm_weight", "provide norm_weight and norm_bias together"),
])
def test_ffn_rejects_incomplete_optional_parameters(name, message):
    args, kwargs = ffn_problem(1, 3, 4)
    kwargs[name] = args[0]
    with pytest.raises(ValueError, match=message):
        feed_forward(*args, **kwargs)


@pytest.mark.parametrize(
    "operation", ("residual_ffn", "coefficient_ffn")
)
def test_triton_ffn_entries_reject_cpu_before_launch(operation):
    from LunaEquivariance.so2.feed_forward.interface import coefficient_ffn, norm_residual_ffn

    x = torch.randn(2, 4, 3)
    weight, bias = torch.ones(2, 3), torch.zeros(3)
    degrees = torch.tensor([0, 1, 1, 1])
    scalar_weight, scalar_bias = torch.randn(5, 3), torch.zeros(5)
    first, second = torch.randn(2, 5, 3), torch.randn(2, 3, 5)
    calls = {
        "residual_ffn": lambda: norm_residual_ffn(
            x,
            weight,
            bias,
            scalar_weight,
            scalar_bias,
            first,
            torch.zeros(5),
            second,
            bias,
            degrees,
            lmax=1,
            epsilon=1e-5,
        ),
        "coefficient_ffn": lambda: coefficient_ffn(
            x,
            scalar_weight,
            scalar_bias,
            first,
            degrees,
            torch.zeros(5),
            second,
            degrees,
            bias,
            lmax=1,
        ),
    }
    with pytest.raises(ValueError, match="CUDA FP32"):
        calls[operation]()


@pytest.mark.parametrize(
    "invalid", ("strided_weight", "strided_bias", "different_device", "half_precision")
)
def test_ffn_affine_contracts_without_a_gpu(invalid):
    from LunaEquivariance.so2.feed_forward.interface import norm_residual_ffn

    x = torch.empty(2, 4, 3)
    weight, bias = torch.empty(2, 3), torch.empty(3)
    if invalid == "strided_weight":
        weight = torch.empty(2, 6)[:, ::2]
    elif invalid == "strided_bias":
        bias = torch.empty(6)[::2]
    elif invalid == "different_device":
        weight = torch.empty(2, 3, device="meta")
    else:
        weight = weight.half()
    call = lambda: norm_residual_ffn(
        x,
        weight,
        bias,
        torch.empty(5, 3),
        torch.empty(5),
        torch.empty(2, 5, 3),
        torch.empty(5),
        torch.empty(2, 3, 5),
        torch.empty(3),
        torch.empty(4, dtype=torch.int64),
        lmax=1,
        epsilon=1e-5,
    )
    message = (
        "contiguous"
        if invalid.startswith("strided")
        else "share one device"
        if invalid == "different_device"
        else "CUDA FP32"
    )
    with pytest.raises(ValueError, match=message):
        call()
