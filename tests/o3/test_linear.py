"""Linear outputs, layouts, parameter ownership and derivatives."""

from copy import deepcopy
from itertools import product

import pytest
import torch
from e3nn import o3

import LunaEquivariance as lunaeq
from LunaEquivariance import Linear, o3 as luna_o3
from LunaEquivariance.modules import linear as linear_module
from LunaEquivariance.modules.linear import LunaEqLinear
from LunaEquivariance.o3.linear.interface import linear
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    rotate_features,
)


LINEAR_CASES = {
    "scalar_tail": ("33x0e", "17x0e", None, True, "element"),
    "uniform": ("16x0e + 16x1o + 16x2e", "16x0e + 16x1o + 16x2e", None, True, "element"),
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
    "zero_multiplicity": ("0x0e + 4x1o", "0x0e + 2x1o + 3x0e", None, False, "element"),
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
            _optimize_einsums=False if mode == "external" and biases else None,
            **dict(f_in=2, f_out=3) if feature_axes else {},
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
    return (linear, x, w, b)


@pytest.mark.cuda
@pytest.mark.parametrize("case", ["scalar_tail", "uniform", "repeated", "partial"])
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
    inputs = tuple((value for value, needed in zip((x, w, b), needs) if needed))
    actual_output = op(x, w, b)
    ref_x, ref_w, ref_b, ref_g = fp64((x, w, b, g))
    ref_inputs = tuple(value for value in (ref_x, ref_w, ref_b) if value.requires_grad)
    reference_output = deepcopy(linear).double()(ref_x, ref_w, ref_b)
    actual = torch.autograd.grad(
        actual_output, inputs, g, allow_unused=True, materialize_grads=True
    )
    expected = torch.autograd.grad(
        reference_output, ref_inputs, ref_g, allow_unused=True, materialize_grads=True
    )
    assert_reference_close(actual_output, reference_output)
    for result, reference in zip(actual, expected):
        assert_reference_close(result, reference)


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
    ref_targets = fp64(inputs)
    actual = (native(*inputs),)
    expected = (deepcopy(reference).double()(*ref_targets),)
    targets = inputs
    for order in range(4):
        for value, target in zip(actual, expected):
            if value is None or target is None:
                remaining = target if value is None else value
                assert remaining is None or not remaining.count_nonzero()
            else:
                assert_reference_close(value, target)
        if order == 3:
            break
        pairs = [
            (a, b)
            for a, b in zip(actual, expected)
            if a is not None and b is not None and a.requires_grad and b.requires_grad
        ]
        seeds = tuple((torch.randn_like(a, requires_grad=True) for a, _ in pairs))
        ref_seeds = fp64(seeds)
        actual = torch.autograd.grad(
            tuple((a for a, _ in pairs)),
            targets,
            seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        expected = torch.autograd.grad(
            tuple((b for _, b in pairs)),
            ref_targets,
            ref_seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        targets = (*targets, *seeds)
        ref_targets = (*ref_targets, *ref_seeds)


@pytest.mark.parametrize("method", ["half", "bfloat16"])
def test_low_precision_conversion_updates_owned_tensors_and_uses_compatibility(method):
    op = LunaEqLinear("2x0e", "2x0e", biases=True, dtype=torch.float32)
    ref = o3.Linear("2x0e", "2x0e", biases=True)
    ref.load_state_dict(op.state_dict(), strict=True)
    weight = op.weight.detach().clone()
    assert getattr(op, method)() is op
    getattr(ref, method)()
    dtype = torch.float16 if method == "half" else torch.bfloat16
    assert op.weight.dtype == op.bias.dtype == op.output_mask.dtype == dtype
    assert op.weight is op.tp.weight and op.bias is op.tp.bias
    torch.testing.assert_close(op.weight, weight.to(dtype), rtol=0, atol=0)
    features = torch.randn(3, 2, dtype=dtype)
    torch.testing.assert_close(op(features), ref(features), rtol=0, atol=0)
    assert op.last_execution_backend == "torch_compat"


@pytest.mark.parametrize("shared", (True, False))
@pytest.mark.parametrize("feature_axes", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
def test_wrap_scripted_module_infers_external_feature_axes(
    shared, feature_axes, explicit
):
    axes = dict(f_in=2, f_out=3) if feature_axes else {}
    reference = o3.Linear(
        "2x0e+1x1o", "2x0e+1x1o", internal_weights=False, shared_weights=shared, **axes
    )
    reference._compiled_main = torch.jit.script(reference._compiled_main)
    native = LunaEqLinear(reference, **axes if explicit else {})
    assert (native.f_in, native.f_out) == ((2, 3) if feature_axes else (None, None))
    x = torch.randn((4,) + ((2,) if feature_axes else ()) + (5,))
    weight = torch.randn(
        (() if shared else (4,))
        + ((2, 3) if feature_axes else ())
        + (reference.weight_numel,)
    )
    torch.testing.assert_close(native(x, weight), reference(x, weight), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", (True, False))
def test_wrap_scripted_external_feature_axes_selects_native(shared):
    reference = o3.Linear(
        "2x0e+1x1o",
        "2x0e+1x1o",
        internal_weights=False,
        shared_weights=shared,
        f_in=2,
        f_out=3,
    ).cuda()
    reference._compiled_main = torch.jit.script(reference._compiled_main)
    native = LunaEqLinear(reference)
    x = torch.randn(7, 2, 5, device="cuda", requires_grad=True)
    weight = torch.randn(
        (() if shared else (7,)) + (2, 3, reference.weight_numel),
        device="cuda",
        requires_grad=True,
    )
    reference_x, reference_weight = fp64((x, weight))
    reference = deepcopy(reference).double()
    expected, actual = (reference(reference_x, reference_weight), native(x, weight))
    assert native.last_execution_backend == "cuda_fused"
    assert_reference_close(actual, expected)
    g = torch.randn_like(actual)
    for calculated, target in zip(
        torch.autograd.grad(actual, (x, weight), g),
        torch.autograd.grad(expected, (reference_x, reference_weight), g.double()),
    ):
        assert_reference_close(calculated, target)


CASES = (
    pytest.param(1, 3, True, False, id="shared-dot-direct"),
    pytest.param(1, 65, True, False, id="shared-dot-split"),
    pytest.param(1, 3, False, False, id="external-dot-dx-dw-dg-db"),
    pytest.param(1, 65, False, False, id="external-dot-medium-dx-dw-dg-db"),
    pytest.param(1, 4096, False, False, id="external-dot-path-major"),
    pytest.param(32, 3, True, False, id="shared-gemm-direct"),
    pytest.param(32, 65, True, False, id="shared-gemm-split"),
    pytest.param(32, 3, False, False, id="external-power-of-two"),
    pytest.param(33, 3, False, False, id="external-padded"),
    pytest.param(1, 3, True, True, id="shared-feature-axes"),
    pytest.param(1, 3, False, True, id="external-feature-axes"),
)


def mixed_derivatives(module, x, weight, g, rx, rw, bx, bw, bg):
    y = module(x, weight)
    first = torch.autograd.grad(y, (x, weight), g, create_graph=True)
    second = torch.autograd.grad(first, (x, weight, g), (rx, rw), create_graph=True)
    third = torch.autograd.grad(second, (x, weight, g, rx, rw), (bx, bw, bg))
    return (y, *first, *second, *third)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("width,items,shared,feature_axes", CASES)
def test_ignored_weight_slots_match_e3nn_in_all_orders(
    dtype, width, items, shared, feature_axes
):
    torch.manual_seed(47212)
    kwargs = dict(
        internal_weights=False,
        shared_weights=shared,
        instructions=[(0, 0), (0, -1)],
        **dict(f_in=2, f_out=3) if feature_axes else {},
    )
    reference = o3.Linear(f"{width}x0e", f"{width}x0e", **kwargs).cuda().to(dtype)
    native = LunaEqLinear(reference)
    xshape = (items,) + ((2,) if feature_axes else ()) + (width,)
    yshape = (items,) + ((3,) if feature_axes else ()) + (width,)
    wshape = (
        (() if shared else (items,))
        + ((2, 3) if feature_axes else ())
        + (2 * width * width,)
    )
    shapes = (xshape, wshape, yshape, xshape, wshape, xshape, wshape, yshape)
    operands = tuple(
        (
            torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            for shape in shapes
        )
    )
    expected = mixed_derivatives(deepcopy(reference).double(), *fp64(operands))
    actual = mixed_derivatives(native, *operands)
    assert native.last_execution_backend == "cuda_fused"
    for result, target in zip(actual, expected):
        assert_reference_close(result, target)
    gaps = slice(width * width, 2 * width * width)
    for index in (2, 4, 7, 10):
        assert not actual[index][..., gaps].count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "layout", ("shared", "unshared", "features_broadcast", "weight_broadcast")
)
@pytest.mark.parametrize("strided", (False, True))
def test_native_layout_preparation_preserves_through_triple_backward(layout, strided):
    op = linear_module.LunaEqLinear(
        "1x0e", "1x0e", internal_weights=False, shared_weights=layout == "shared"
    )
    rows = 1 if layout == "features_broadcast" else 4
    x = (
        torch.randn(rows, 2, dtype=torch.float64)[:, :1]
        if strided
        else torch.randn(rows, 1, dtype=torch.float64)
    )
    x = torch.nn.Parameter(x)
    shape = (
        (1,) if layout == "shared" else (1 if layout == "weight_broadcast" else 4, 1)
    )
    weight = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    x = x.detach().cuda().requires_grad_()
    weight = weight.detach().cuda().requires_grad_()
    actual = op(x, weight)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference = reference_x * reference_weight
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_loss, reference_loss = (actual.pow(4).sum(), reference.pow(4).sum())
    for order in range(1, 4):
        actual_grad = torch.autograd.grad(actual_loss, (x, weight), create_graph=True)
        reference_grad = torch.autograd.grad(
            reference_loss, (reference_x, reference_weight), create_graph=True
        )
        for actual_value, reference_value in zip(actual_grad, reference_grad):
            assert_reference_close(actual_value, reference_value.double())
        actual_loss = sum((value.sum() for value in actual_grad))
        reference_loss = sum((value.sum() for value in reference_grad))


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize("axes", (False, True))
@pytest.mark.parametrize("shared", (False, True))
def test_linear_without_module_and_explicit_parameters(dtype, axes, shared):
    options = dict(f_in=2, f_out=3) if axes else {}
    ref = o3.Linear(
        "2x0e+1x1o",
        "3x0e+2x1o",
        internal_weights=False,
        shared_weights=shared,
        **options,
    ).to(dtype)
    x = torch.randn(
        (5,) + ((2,) if axes else ()) + (5,), dtype=dtype, requires_grad=True
    )
    w = torch.randn(
        (() if shared else (5,)) + ((2, 3) if axes else ()) + (ref.weight_numel,),
        dtype=dtype,
        requires_grad=True,
    )
    expected = ref(x, w)
    actual = linear(
        x,
        w,
        irreps_in="2x0e+1x1o",
        irreps_out="3x0e+2x1o",
        shared_weights=shared,
        **options,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    seed = torch.randn_like(actual)
    for value, target in zip(
        torch.autograd.grad(actual, (x, w), seed),
        torch.autograd.grad(expected, (x, w), seed),
    ):
        torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("normalization", ("element", "path"))
def test_functional_linear_bias_and_irreps_objects(normalization):
    irreps_in, irreps_out = (o3.Irreps("2x0e+1x1o"), o3.Irreps("3x0e+2x1o"))
    reference = o3.Linear(
        irreps_in, irreps_out, biases=True, path_normalization=normalization
    ).double()
    x = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    actual = linear(
        x,
        reference.weight,
        reference.bias,
        irreps_in=list(irreps_in),
        irreps_out=irreps_out,
        biases=True,
        path_normalization=normalization,
    )
    torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class TestEquivariance:
    def test_linear(self, device, dtype, rotation):
        irreps_in = "3x0e+2x1o+1x2e"
        irreps_out = "2x0e+3x1o+2x2e"
        module = Linear(irreps_in, irreps_out, biases=True, dtype=dtype).to(device)
        features = torch.randn(7, module.irreps_in.dim, device=device, dtype=dtype)
        actual = module(rotate_features(features, irreps_in, rotation))
        expected = rotate_features(module(features), irreps_out, rotation)
        tolerance = 3e-05 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_operator_aliases():
    operator = getattr(lunaeq, "Linear")
    assert issubclass(operator, torch.nn.Module)
    assert getattr(lunaeq, "LunaEq" + "Linear") is operator
    assert getattr(getattr(lunaeq, "o3"), "Linear") is operator


def test_frozen_parameters():
    module = luna_o3.Linear("2x0e", "3x0e").requires_grad_(False)
    x = torch.randn(3, 2, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report
    assert all((not p.requires_grad for p in module.parameters()))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_module_three_derivatives(dtype):
    from LunaEquivariance import Linear

    torch.manual_seed(42)
    module = Linear("2x0e+1x1o", "3x0e+2x1o", biases=True).to(
        device="cuda", dtype=dtype
    )
    width = 5
    x = torch.randn(5, width, device="cuda", dtype=dtype).requires_grad_()
    assert_cuda_report(module, (x,))


def assert_cuda_report(module, args, **kwargs):
    tolerance = 1e-09 if args[0].dtype == torch.float64 else 0.0003
    report = check_compatibility(
        module,
        args,
        kwargs=kwargs,
        derivative_order=3,
        loss_fn=lambda output: sum(
            (
                x.sin().mean()
                for x in (output if isinstance(output, tuple) else (output,))
            )
        ),
        warmup=1,
        repeats=2,
        atol=tolerance,
        rtol=tolerance,
        verbose=False,
    )
    assert report["passed"], report
    assert report["operator"] == type(module).__name__
    assert report["errors"]["gradient_3"]["passed"]
    assert all((item["speedup"] > 0 for item in report["timings"].values()))


def compilation_case(channels):
    irreps = f"{channels}x0e + {channels}x1o"
    module = lunaeq.Linear(irreps, irreps, biases=True).cuda()
    samples = [
        ((torch.randn(rows, 4 * channels, device="cuda", requires_grad=True),), {})
        for rows in (7, 19)
    ]
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (4, 8))
def test_compile(channels):
    module, samples = compilation_case(channels)
    assert_compiled(module, samples)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (4, 8)]
    assert_aoti(cases, tmp_path)
