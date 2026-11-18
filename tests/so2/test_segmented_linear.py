"""Segmented channel maps and gradients of expert mixtures."""

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance.so2 import segmented_linear
from LunaEquivariance.so2.segmented_linear.interface import concat_with_split_backward
from LunaEquivariance.so2.segmented_linear.interface import (
    autograd_segmented_linear,
    segmented_linear_with_gradients,
)
from tests.so2.reference.segmented_linear import segmented_linear as segmented_reference
from tests.so2.checks import (
    assert_derivative_slots,
    compare_operator_derivatives,
    first_and_second_gradients,
)


SIZES = (1, 0, 2)


def make_inputs():
    generator = torch.Generator().manual_seed(194)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in ((3, 2, 2), (2, 2, 2), (3, 2), (2,))
    )


@pytest.mark.parametrize("mode", ("input", "full", "force"))
def test_segmented_linear_forward_and_input_vjp(mode):
    inputs = make_inputs()
    actual = segmented_linear_with_gradients(*inputs, SIZES, gradient_mode=mode)
    expected = autograd_segmented_linear(*inputs, SIZES)
    cotangent = torch.linspace(
        -0.7, 0.9, actual.numel(), dtype=actual.dtype
    ).reshape_as(actual)
    torch.testing.assert_close(actual, expected)
    actual_dx = torch.autograd.grad(actual, inputs[0], cotangent)[0]
    expected_dx = torch.autograd.grad(expected, inputs[0], cotangent)[0]
    torch.testing.assert_close(actual_dx, expected_dx)


def test_force_parameter_double_backward():
    inputs = make_inputs()
    actual = segmented_linear_with_gradients(*inputs, SIZES, gradient_mode="force")
    expected = autograd_segmented_linear(*inputs, SIZES)
    cotangent = torch.linspace(
        -0.7, 0.9, actual.numel(), dtype=actual.dtype
    ).reshape_as(actual)
    cotangent.requires_grad_()
    actual_first = torch.autograd.grad(
        actual, inputs, cotangent, create_graph=True, allow_unused=True
    )
    # Force mode deliberately exposes only the input VJP in the first graph.
    assert all(value is None for value in actual_first[1:])
    expected_dx = torch.autograd.grad(
        expected, inputs[0], cotangent, create_graph=True
    )[0]
    torch.testing.assert_close(actual_first[0], expected_dx)
    differentiated = (inputs[1], inputs[2], cotangent)
    actual_second = torch.autograd.grad(actual_first[0].square().sum(), differentiated)
    expected_second = torch.autograd.grad(expected_dx.square().sum(), differentiated)
    for candidate, baseline in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(candidate, baseline)
    # The empty segment contributes no coefficient derivative.
    torch.testing.assert_close(
        actual_second[1][1], torch.zeros_like(actual_second[1][1])
    )


def test_full_finite_difference_derivatives():
    inputs = make_inputs()

    def function(*values):
        return segmented_linear(*values, SIZES)

    assert torch.autograd.gradcheck(function, inputs)
    assert torch.autograd.gradgradcheck(function, inputs)


def test_force_mode_with_frozen_input_supports_parameter_derivatives():
    x, *parameters = make_inputs()
    x = x.detach()
    actual = segmented_linear_with_gradients(
        x, *parameters, SIZES, gradient_mode="force"
    )
    expected = autograd_segmented_linear(x, *parameters, SIZES)
    actual_first = torch.autograd.grad(
        actual.sin().sum(), parameters, create_graph=True
    )
    expected_first = torch.autograd.grad(
        expected.sin().sum(), parameters, create_graph=True
    )
    actual_second = torch.autograd.grad(
        sum(value.square().sum() for value in actual_first), parameters
    )
    expected_second = torch.autograd.grad(
        sum(value.square().sum() for value in expected_first), parameters
    )
    for result, reference in zip(
        (*actual_first, *actual_second),
        (*expected_first, *expected_second),
        strict=True,
    ):
        torch.testing.assert_close(result, reference)


def test_input_mode_with_frozen_parameters_supports_force_loss():
    x, *parameters = make_inputs()
    parameters = tuple(value.detach() for value in parameters)
    actual = segmented_linear_with_gradients(
        x, *parameters, SIZES, gradient_mode="input"
    )
    expected = autograd_segmented_linear(x, *parameters, SIZES)
    actual_dx = torch.autograd.grad(actual.sin().sum(), x, create_graph=True)[0]
    expected_dx = torch.autograd.grad(expected.sin().sum(), x, create_graph=True)[0]
    actual_second = torch.autograd.grad(actual_dx.square().sum(), x)[0]
    expected_second = torch.autograd.grad(expected_dx.square().sum(), x)[0]
    torch.testing.assert_close(actual_dx, expected_dx)
    torch.testing.assert_close(actual_second, expected_second)


@pytest.mark.parametrize("mode", ("full", "force"))
def test_segmented_linear_strided_input_and_cotangent(mode):
    x, *parameters = make_inputs()
    x = x.transpose(1, 2).detach().requires_grad_()
    inputs = (x, *parameters)
    actual = segmented_linear_with_gradients(*inputs, SIZES, gradient_mode=mode)
    expected = autograd_segmented_linear(*inputs, SIZES)
    cotangent = torch.randn_like(actual).transpose(1, 2).requires_grad_()
    actual_dx = torch.autograd.grad(actual, x, cotangent, create_graph=True)[0]
    expected_dx = torch.autograd.grad(expected, x, cotangent, create_graph=True)[0]
    targets = (*parameters[:2], cotangent)
    actual_second = torch.autograd.grad(actual_dx.square().sum(), targets)
    expected_second = torch.autograd.grad(expected_dx.square().sum(), targets)
    torch.testing.assert_close(actual_dx, expected_dx)
    for result, reference in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("mode", ("full", "force"))
def test_third_derivative_support_by_gradient_mode(mode):
    x, weights, coefficients, bias = make_inputs()
    actual = segmented_linear_with_gradients(
        x, weights, coefficients, bias, SIZES, gradient_mode=mode
    )
    cotangent = torch.randn_like(actual, requires_grad=True)
    dx = torch.autograd.grad(actual, x, cotangent, create_graph=True)[0]
    direction = torch.randn_like(dx, requires_grad=True)
    second = torch.autograd.grad(dx, weights, direction, create_graph=True)[0]
    assert torch.isfinite(second).all()
    if mode == "force":
        with pytest.raises(RuntimeError, match="once_differentiable"):
            second.sum().backward()
        return

    expected = autograd_segmented_linear(x, weights, coefficients, bias, SIZES)
    ref_dx = torch.autograd.grad(expected, x, cotangent, create_graph=True)[0]
    ref_second = torch.autograd.grad(ref_dx, weights, direction, create_graph=True)[0]
    targets = (coefficients, cotangent, direction)
    third = torch.autograd.grad(second.sum(), targets)
    ref_third = torch.autograd.grad(ref_second.sum(), targets)
    for value, reference in zip(third, ref_third, strict=True):
        torch.testing.assert_close(value, reference)


def derivatives(function, inputs, cotangent, directions):
    output = function(*inputs)
    variables = (*inputs, cotangent)
    first = torch.autograd.grad(
        (output * cotangent).sum(), variables, create_graph=True
    )
    second = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(first, directions, strict=True)
        ),
        variables,
    )
    return output, *first, *second


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("sizes", [(2, 0, 3, 1), (0,), (3,)])
def test_concat_strided_segments_double_backward(dtype, dim, sizes):
    generator = torch.Generator().manual_seed(341)
    values = []
    for size in sizes:
        shape = [2, 3, 4]
        shape[dim] = size
        storage = torch.randn(
            (*shape[:-1], 2 * shape[-1]), dtype=dtype, generator=generator
        )
        values.append(storage[..., ::2].requires_grad_())
    inputs = tuple(values)
    reference = torch.cat(inputs, dim=dim)
    cotangent = torch.randn(
        reference.shape, dtype=dtype, generator=generator, requires_grad=True
    )
    directions = tuple(
        torch.randn(value.shape, dtype=dtype, generator=generator)
        for value in (*inputs, cotangent)
    )
    original_values = tuple(value.detach().clone() for value in inputs)
    expected = derivatives(
        lambda *args: torch.cat(args, dim=dim), inputs, cotangent, directions
    )
    actual = derivatives(
        lambda *args: concat_with_split_backward(args, dim=dim),
        inputs,
        cotangent,
        directions,
    )
    for candidate, baseline in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    for value, original in zip(inputs, original_values, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_concat_aliased_input_finite_differences():
    value = torch.randn(3, 4, dtype=torch.float64).T.requires_grad_()

    def function(x):
        return concat_with_split_backward((x, x[:0], x), dim=0)

    assert torch.autograd.gradcheck(function, (value,))
    assert torch.autograd.gradgradcheck(function, (value,))


def native_segmented(x, weights, coefficients, bias):
    mixed = (coefficients @ weights.flatten(1)).view(
        coefficients.shape[0], weights.shape[1], weights.shape[2]
    )
    return torch.cat(
        [
            torch.nn.functional.linear(local, weight, bias)
            for local, weight in zip(x.split((2, 0, 3)), mixed.unbind(0), strict=True)
        ]
    )


def segmented(x, weights, coefficients, bias):
    return autograd_segmented_linear(x, weights, coefficients, bias, (2, 0, 3))


def mole_inputs(dtype):
    generator = torch.Generator().manual_seed(342)
    shapes = ((5, 3), (2, 4, 3), (3, 2), (4,))
    return tuple(
        torch.randn(shape, dtype=dtype, generator=generator).requires_grad_()
        for shape in shapes
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segmented_linear_all_derivatives(dtype):
    inputs = mole_inputs(dtype)
    cotangent = torch.randn(5, 4, dtype=dtype, requires_grad=True)
    directions = tuple(torch.randn_like(value) for value in (*inputs, cotangent))
    expected = derivatives(native_segmented, inputs, cotangent, directions)
    actual = derivatives(segmented, inputs, cotangent, directions)
    for candidate, baseline in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_first_order_segmented_external_double_backward(dtype):
    inputs = mole_inputs(dtype)

    def first_order(*values):
        output = segmented(*values)
        gradients = torch.autograd.grad(
            output.square().sum(), values, create_graph=True
        )
        return output, *gradients

    def native_first_order(*values):
        output = native_segmented(*values)
        gradients = torch.autograd.grad(
            output.square().sum(), values, create_graph=True
        )
        return output, *gradients

    traced = make_fx(first_order)(*inputs)
    compiled = torch.compile(traced, backend="aot_eager", fullgraph=True)
    actual = compiled(*inputs)
    actual_bw2 = torch.autograd.grad(
        sum(value.square().sum() for value in actual), inputs
    )
    expected = native_first_order(*inputs)
    expected_bw2 = torch.autograd.grad(
        sum(value.square().sum() for value in expected), inputs
    )
    for candidate, baseline in zip(
        (*actual, *actual_bw2), (*expected, *expected_bw2), strict=True
    ):
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_segmented_linear_preserves_all_external_parameter_derivatives(
    frozen_input, with_bias
):
    from LunaEquivariance.so2 import segmented_linear

    generator = torch.Generator().manual_seed(991)
    shapes = ((5, 2, 3), (2, 4, 3), (3, 2), (4,))
    inputs = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64)
        for shape in shapes[: 4 if with_bias else 3]
    )
    actual_inputs = tuple(
        value.clone().requires_grad_(index != 0 or not frozen_input)
        for index, value in enumerate(inputs)
    )
    reference_inputs = tuple(
        value.clone().requires_grad_(index != 0 or not frozen_input)
        for index, value in enumerate(inputs)
    )
    sizes = (2, 0, 3)

    def evaluate(values, reference):
        x, weights, coefficients = values[:3]
        bias = values[3] if with_bias else None
        if reference:
            mixed = torch.einsum("se,eoi->soi", coefficients, weights)
            output = torch.cat(
                tuple(
                    F.linear(local, weight, bias)
                    for local, weight in zip(x.split(sizes), mixed, strict=True)
                )
            )
        else:
            output = segmented_linear(x, weights, coefficients, bias, sizes)
        targets = tuple(value for value in values if value.requires_grad)
        first = torch.autograd.grad(
            (output.sin() + 0.2 * output.square()).sum(), targets, create_graph=True
        )
        assert all(gradient is not None for gradient in first)
        second = torch.autograd.grad(
            first, targets, tuple(torch.full_like(value, 0.1) for value in targets)
        )
        assert all(gradient is not None for gradient in second)
        return output, *first, *second

    actual = evaluate(actual_inputs, reference=False)
    expected = evaluate(reference_inputs, reference=True)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "invalid", ["input_width", "mixture_width", "sizes", "bias", "dtype"]
)
def test_segmented_linear_rejects_inconsistent_tensor_contracts(invalid):
    from LunaEquivariance.so2 import segmented_linear

    x = torch.empty(5, 3)
    weights = torch.empty(2, 4, 3)
    coefficients = torch.empty(3, 2)
    bias = torch.empty(4)
    sizes = (2, 0, 3)
    if invalid == "input_width":
        x = torch.empty(5, 2)
    elif invalid == "mixture_width":
        coefficients = torch.empty(3, 3)
    elif invalid == "sizes":
        sizes = (1, 0, 3)
    elif invalid == "bias":
        bias = torch.empty(1, 4)
    else:
        weights = weights.double()
    with pytest.raises(ValueError):
        segmented_linear(x, weights, coefficients, bias, sizes)


def test_segmented_linear_asymmetric_dimensions():
    from LunaEquivariance.so2 import segmented_linear
    from LunaEquivariance.so2.segmented_linear.interface import autograd_segmented_linear

    torch.manual_seed(2026)
    data = tuple(
        torch.randn(shape, dtype=torch.float64) * 0.2
        for shape in [(6, 2, 5), (3, 7, 5), (4, 3), (7,)]
    )
    directions = tuple(torch.randn_like(v) for v in data)
    cotangent_data = torch.randn(6, 2, 7, dtype=torch.float64)
    results = []
    for reference in (False, True):
        inputs = tuple(v.detach().requires_grad_() for v in data)
        cotangent = cotangent_data.detach().requires_grad_()
        if reference:
            output = autograd_segmented_linear(*inputs, (1, 0, 3, 2))
        else:
            output = segmented_linear(*inputs, (1, 0, 3, 2))
        results.append(first_and_second_gradients(output, inputs, cotangent, directions))
    assert_derivative_slots(*results)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layout", ["matrix", "paired"])
@pytest.mark.parametrize("sizes", [(1, 0, 3, 2), (6,), (0,)])
def test_segmented_linear_cpu_double_backward(bias, layout, sizes):
    torch.manual_seed(731)
    shape = (sum(sizes), 2, 10) if layout == "paired" else (sum(sizes), 10)
    x = torch.randn(shape, dtype=torch.float64)[..., ::2].requires_grad_()
    weight = torch.randn(3, 7, 5, dtype=torch.float64, requires_grad=True)
    mixing = torch.randn(len(sizes), 3, dtype=torch.float64, requires_grad=True)
    offset = torch.randn(7, dtype=torch.float64, requires_grad=True) if bias else None
    for operation in (segmented_linear, autograd_segmented_linear):
        compare_operator_derivatives(
            operation, segmented_reference, (x, weight, mixing, offset, sizes),
        )


@pytest.mark.cuda
def test_segmented_linear_cuda_all_parameter_gradients():
    torch.manual_seed(731)
    values = [torch.randn(shape, device="cuda", requires_grad=True)
              for shape in ((6, 2, 5), (3, 7, 5), (4, 3), (7,))]
    compare_operator_derivatives(segmented_linear, segmented_reference,
                        (*values, (1, 0, 3, 2)))


@pytest.mark.cuda
def test_force_mixture_accumulates_energy_and_force_gradients():
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        inputs = tuple(value.detach().cuda().float().requires_grad_() for value in make_inputs())
        for weight in (0., 0.7):
            actual = segmented_linear_with_gradients(*inputs, SIZES, gradient_mode="force")
            expected = autograd_segmented_linear(*inputs, SIZES)
            result = []
            for output in (actual, expected):
                energy = output.sin().sum()
                force, = torch.autograd.grad(energy, inputs[0], create_graph=True)
                gradients = torch.autograd.grad(force.square().sum() + weight * energy, inputs)
                result.append((force, *gradients))
            torch.testing.assert_close(result[0], result[1], rtol=2e-5, atol=1e-5)
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.cuda
@pytest.mark.parametrize(("cin", "cout"), [(256, 256), (128, 256)])
@pytest.mark.parametrize("negative_view", [False, True])
def test_grouped_linear_large_segments(cin, cout, negative_view):
    from LunaEquivariance.so2.segmented_linear.kernels.utils import (
        grouped_device,
        use_grouped,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    if not grouped_device(device):
        pytest.skip("grouped cuBLAS is unavailable")

    sizes = (4608,) + (512,) * 15
    generator = torch.Generator(device=device).manual_seed(731)

    def value(shape):
        return torch.randn(shape, device=device, generator=generator).mul_(0.1)

    x = value((sum(sizes), 2, cin))
    if negative_view:
        x = torch._neg_view(x)
    x.requires_grad_()
    weights = value((3, cout, cin)).requires_grad_()
    mixing = value((len(sizes), 3)).requires_grad_()
    bias = value((cout,)).requires_grad_()
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        assert use_grouped(x, value((len(sizes), cout, cin)), sizes)
        compare_operator_derivatives(
            segmented_linear, segmented_reference, (x, weights, mixing, bias, sizes),
        )
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.cuda
def test_grouped_ieee_weight_keeps_ninth_split_and_cancellation():
    from LunaEquivariance.so2.segmented_linear.kernels.backward import segmented_weight_gradient
    from LunaEquivariance.so2.segmented_linear.kernels.utils import (
        WEIGHT, grouped_device, grouped_gemm, row_sizes,
    )

    device = torch.device("cuda")
    if not grouped_device(device):
        pytest.skip("Split weight gradients require an A100 or H100 and the CUDA toolkit")

    sizes = [4608] + [700] * 15
    x = torch.zeros((sum(sizes), 2, 256), dtype=torch.float32, device=device)
    dy = torch.zeros_like(x)
    gradient = x.new_empty((len(sizes), 256, 256))
    x_rows, dy_rows = x.flatten(0, 1), dy.flatten(0, 1)
    x_rows[8192, 0], dy_rows[8192, 0] = 2.0, 3.0
    start = 2 * sizes[0]
    h = 2.0 ** -8
    p = (1 + h) ** 2
    x_rows[start:start + 3, 0] = x.new_tensor((1 + h, -p, 2.0 ** -34)) * 2.0 ** 20
    dy_rows[start:start + 3, 0] = x.new_tensor((1 + h, 1, 1)) * 2.0 ** 20

    rows = row_sizes(x, sizes)
    grouped_gemm(x_rows, dy_rows, gradient, rows, WEIGHT, False)
    expected = torch.zeros_like(gradient)
    expected[0, 0, 0], expected[1, 0, 0] = 6.0, 64.0
    torch.testing.assert_close(gradient, expected, rtol=0.0, atol=0.0)

    neg_dy = torch._neg_view(dy)
    neg_result = segmented_weight_gradient(neg_dy, x, sizes, gradient)
    torch.testing.assert_close(neg_result[0, 0, 0], expected[0, 0, 0].neg(), rtol=0.0, atol=0.0)

    with pytest.raises(RuntimeError, match="partition"):
        grouped_gemm(x_rows, dy_rows, gradient, [rows[0] - 2] + rows[1:], WEIGHT, False)
    aliased = x_rows.flatten()[:gradient.numel()].view_as(gradient)
    with pytest.raises(RuntimeError, match="must not alias"):
        grouped_gemm(x_rows, dy_rows, aliased, rows, WEIGHT, False)


@pytest.mark.cuda
@pytest.mark.parametrize("rank, cout, cin", [(2, 384, 384), (3, 512, 256), (3, 256, 128)])
def test_grouped_ieee_weight_keeps_torch_for_gemm2(monkeypatch, rank, cout, cin):
    from LunaEquivariance.so2.segmented_linear.kernels import backward
    from LunaEquivariance.so2.segmented_linear.kernels.utils import grouped_device

    device = torch.device("cuda")
    if not grouped_device(device):
        pytest.skip("Grouped CUDA extension requires an A100 or H100 and the CUDA toolkit")
    sizes = [80] * 16
    rows = (sum(sizes),) if rank == 2 else (sum(sizes), 2)
    x = torch.zeros((*rows, cin), dtype=torch.float32, device=device)
    dy = torch.zeros((*rows, cout), dtype=torch.float32, device=device)
    mixed = x.new_empty((len(sizes), cout, cin))
    x.flatten(0, -2)[0, 0], dy.flatten(0, -2)[0, 0] = 2.0, 3.0
    expected = torch.zeros_like(mixed)
    expected[0, 0, 0] = 6.0

    def unexpected(*args):
        pytest.fail("GEMM2 used the weight-adjoint fast path")

    monkeypatch.setattr(backward, "grouped_weight_device", lambda _: True)
    monkeypatch.setattr(backward, "streamed_weight_device", lambda _: False)
    monkeypatch.setattr(backward, "grouped_gemm", unexpected)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert backward.use_grouped(x, mixed, sizes)
        torch.testing.assert_close(backward.segmented_weight_gradient(dy, x, sizes, mixed),
                                   expected, rtol=0.0, atol=0.0)
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.cuda
def test_streamed_weight_replays_changed_inputs():
    from LunaEquivariance.so2.segmented_linear.kernels.backward import segmented_weight_gradient
    from LunaEquivariance.so2.segmented_linear.kernels.utils import (
        grouped_device, streamed_weight_device,
    )

    device = torch.device("cuda")
    if not grouped_device(device) or not streamed_weight_device(device):
        pytest.skip("Streamed weight gradients require an H100 and the CUDA toolkit")

    sizes = (80,) * 16
    generator = torch.Generator(device=device).manual_seed(731)
    x = torch.randn(sum(sizes), 2, 128, device=device, generator=generator)
    dy = torch.randn(sum(sizes), 2, 256, device=device, generator=generator)
    mixed = x.new_empty((len(sizes), 256, 128))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.cuda.stream(stream):
            segmented_weight_gradient(dy, x, sizes, mixed)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            observed = segmented_weight_gradient(dy, x, sizes, mixed)
        for scale in (1.0, -2.0):
            x.mul_(scale)
            graph.replay()
            reference = torch.stack([
                y.flatten(0, 1).T @ value.flatten(0, 1)
                for value, y in zip(x.split(sizes), dy.split(sizes), strict=True)
            ])
            torch.testing.assert_close(observed, reference, rtol=1e-5, atol=3e-5)
    finally:
        torch.set_float32_matmul_precision(precision)
