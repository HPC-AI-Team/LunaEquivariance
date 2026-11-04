"""UVU CONV outputs, layouts, equivariance and derivatives."""

from copy import deepcopy
from math import prod

import pytest
import torch
from e3nn import o3
from e3nn.o3 import TensorProduct as E3nnTensorProduct
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance import (
    TensorProductConv,
    make_graph,
    prepare_tensor_product,
    tensor_product_conv,
)
from LunaEquivariance.utils.testing import check_compatibility
from tests.testing_utils import (
    ConvolutionInputs,
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
    graph_tensors,
    rotate_features,
)
import LunaEquivariance as lunaeq


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False])
def test_forward(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uvu_conv", shared)
    op = TensorProductConv(tp)
    kwargs = {}
    reference = deepcopy(tp).double()(x[src].double(), y.double(), w.double())
    kwargs["graph"] = op.create_graph(src, dst, len(x))
    reference = reference.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, reference
    )
    for inference in (False, True):
        with torch.set_grad_enabled(not inference):
            actual = op(x, y, None if shared else w, **kwargs)
        assert_reference_close(actual, reference)


def assert_derivatives_close(actual, expected):
    assert_reference_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_derivatives(make_problem, shared, dtype):
    tp, x, y, w, src, dst = make_problem("uvu_conv", shared, dtype=dtype)
    op = TensorProductConv(tp)
    kwargs = {"graph": op.create_graph(src, dst, len(x))}
    actual = op(x, y, None if shared else w, **kwargs)
    g = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, g)
    first_directions = tuple(
        (torch.randn_like(v, requires_grad=True) for v in inputs[:3])
    )
    second_directions = tuple((torch.randn_like(v) for v in inputs))

    ref_inputs = fp64(inputs)
    ref_first = fp64(first_directions)
    ref_second = fp64(second_directions)
    expected = deepcopy(tp).double()(ref_inputs[0][src], ref_inputs[1], ref_inputs[2])
    expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, expected
    )

    def chain(out, inputs, first_directions, second_directions):
        first = torch.autograd.grad(
            out, inputs[:3], inputs[3], create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, inputs, first_directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, *first_directions), second_directions, retain_graph=True
        )
        return ((out,), first, second, third)

    for reference_order, actual_order in zip(
        chain(expected, ref_inputs, ref_first, ref_second),
        chain(actual, inputs, first_directions, second_directions),
    ):
        for reference, result in zip(reference_order, actual_order):
            assert_derivatives_close(result, reference)
    reference_x = torch.autograd.grad(
        expected, ref_inputs[0], ref_inputs[3], retain_graph=True
    )[0]
    assert_derivatives_close(torch.autograd.grad(actual, x, g)[0], reference_x)
    if not shared:
        force = op(x, y.detach(), w.detach(), **kwargs)
        assert_derivatives_close(torch.autograd.grad(force, x, g)[0], reference_x)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False])
def test_fp64_finite_differences(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uvu_conv", shared, dtype=torch.float64)
    op = TensorProductConv(tp, dtype=torch.float64)
    kwargs = {"graph": op.create_graph(src, dst, len(x))}
    if shared:
        fn, values = (lambda a, b: op(a, b, **kwargs), (x, y))
    else:
        fn, values = (lambda a, b, c: op(a, b, c, **kwargs), (x, y, w))
    nondet_tol = 1e-12
    assert torch.autograd.gradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-06,
        atol=1e-05,
        rtol=0.0001,
        nondet_tol=nondet_tol,
    )
    assert torch.autograd.gradgradcheck(
        fn,
        values,
        fast_mode=True,
        eps=1e-06,
        atol=1e-05,
        rtol=0.0001,
        nondet_tol=nondet_tol,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("shared", [False])
def test_fp64_empty_inputs(make_problem, shared):
    tp, x, y, w, src, dst = make_problem("uvu_conv", shared, dtype=torch.float64)
    op = TensorProductConv(tp)
    x, y = (x, y[:0])
    w = w if shared else w[:0]
    kwargs = {"graph": op.create_graph(src[:0], dst[:0], len(x))}
    result = op(x, y, None if shared else w, **kwargs)
    assert result.shape == (len(x), tp.irreps_out.dim)
    first = torch.autograd.grad(result.sum(), (x, y, w))
    for value, gradient in zip((x, y, w), first):
        assert gradient.shape == value.shape
        assert not gradient.count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equivariance(make_problem, rotation, dtype):
    reference, features, edges, weight, src, dst = make_problem("uvu_conv", dtype=dtype)
    module = TensorProductConv(reference, dtype=dtype)
    kwargs = {"graph": module.create_graph(src, dst, len(features))}
    output = module(features, edges, weight, **kwargs)
    actual = module(
        rotate_features(features, reference.irreps_in1, rotation),
        rotate_features(edges, reference.irreps_in2, rotation),
        weight,
        **kwargs,
    )
    expected = rotate_features(output, reference.irreps_out, rotation)
    tolerance = 5e-05 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.cuda
def test_converted_module_forward_and_three_orders(make_problem):
    tp, x, y, w, src, dst = make_problem("uvu_conv")
    module = TensorProductConv(tp)
    kwargs = {"graph": module.create_graph(src, dst, len(x))}
    module(x, y, w, **kwargs)
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem("uvu_conv", dtype=dtype)
        assert not hasattr(module.op, "graph")
        actual = module(x, y, w, **kwargs)
        grad = torch.randn_like(actual, requires_grad=True)
        inputs = (x, y, w, grad)
        first_seeds = tuple(
            (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
        )
        second_seeds = tuple((torch.randn_like(value) for value in inputs))
        ref_inputs = fp64(inputs)
        ref_first = fp64(first_seeds)
        ref_second = fp64(second_seeds)
        expected = deepcopy(reference).double()(
            ref_inputs[0][src], ref_inputs[1], ref_inputs[2]
        )
        expected = expected.new_zeros((len(x), reference.irreps_out.dim)).index_add(
            0, dst, expected
        )
        assert_reference_close(actual, expected)
        results = []
        for output, targets, directions, second_directions in (
            (expected, ref_inputs, ref_first, ref_second),
            (actual, inputs, first_seeds, second_seeds),
        ):
            first = torch.autograd.grad(
                output, targets[:3], targets[3], create_graph=True, retain_graph=True
            )
            second = torch.autograd.grad(
                first, targets, directions, create_graph=True, retain_graph=True
            )
            third = torch.autograd.grad(
                second, (*targets, *directions), second_directions, retain_graph=True
            )
            results.append((*first, *second, *third))
        for expected_grad, actual_grad in zip(*results):
            assert_reference_close(actual_grad, expected_grad)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_forward_and_three_orders(make_problem, dtype):
    tp, x, y, w, src, dst = make_problem("uvu_conv", dtype=dtype)
    cls = TensorProductConv
    instructions = [
        (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight)
        for ins in tp.instructions
    ]
    module = cls(
        tp.irreps_in1,
        tp.irreps_in2,
        tp.irreps_out,
        instructions,
        internal_weights=False,
        shared_weights=False,
        dtype=dtype,
    )
    baseline = TensorProductConv(tp, dtype=dtype)
    kwargs = {}
    graph = module.create_graph(src, dst, len(x))
    kwargs["graph"] = graph
    actual = module(x, y, w, **kwargs)
    original = baseline(x, y, w, **kwargs)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    grad = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, w, grad)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    ref_inputs = fp64(inputs)
    ref_first = fp64(first_seeds)
    ref_second = fp64(second_seeds)
    expected = deepcopy(tp).double()(ref_inputs[0][src], ref_inputs[1], ref_inputs[2])
    expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
        0, dst, expected
    )
    assert_reference_close(actual, expected)
    results = []
    for output, targets, directions, second_directions in (
        (expected, ref_inputs, ref_first, ref_second),
        (actual, inputs, first_seeds, second_seeds),
    ):
        first = torch.autograd.grad(
            output, targets[:3], targets[3], create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, targets, directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*targets, *directions), second_directions, retain_graph=True
        )
        results.append((*first, *second, *third))
    for reference, result in zip(*results):
        assert_reference_close(result, reference)


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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "internal,shared,env_shape", [(False, False, (4,)), (False, False, (4, 1))]
)
def test_cuda_tensor_product_modes(dtype, internal, shared, env_shape):
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(42)
    cls = TensorProductConv
    module = cls(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=internal,
        shared_weights=shared,
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(3, 6, device="cuda", dtype=dtype).requires_grad_()
    y = torch.randn(4, 6, device="cuda", dtype=dtype).requires_grad_()
    args = (x, y)
    if not internal:
        shape = (module.weight_numel,) if shared else (4, module.weight_numel)
        args += (torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(),)
    kwargs = {}
    src = torch.tensor([1, 2, 0, 1], device="cuda")
    dst = torch.tensor([0, 0, 1, 2], device="cuda")
    kwargs = dict(
        graph=module.create_graph(src, dst, 3),
        env=torch.rand(env_shape, device="cuda", dtype=dtype).requires_grad_(),
    )
    kwargs["return_l0_feat"] = True
    assert_cuda_report(module, args, **kwargs)


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
    assert actual.dtype == dtype
    assert_reference_close(actual, expected)


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
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, l0
    )
    inputs = (x, y, w) if env is None else (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_inputs,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
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
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, True
    )
    inputs = (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_inputs,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.fixture(params=[torch.float32, torch.float64])
def optional_dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def assert_optional_close(actual, expected):
    assert_reference_close(actual, expected)


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
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(),
        ref_x,
        ref_y,
        ref_w,
        src,
        dst,
        None if l0 else ref_env,
        l0,
    )
    inputs = (x, y, w) if l0 or frozen_env else (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
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
    ref_chosen = tuple(ref_by_id[id(value)] for value in chosen)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in chosen))
    second_inputs = (*inputs, *seeds)
    second_seeds = tuple((torch.randn_like(v) for v in second_inputs))
    for a, b in zip(expected, actual):
        assert_optional_close(b, a)
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for reference, result in zip(
        optional_derivative_orders(
            ref_chosen,
            ref_seeds,
            (*ref_inputs, *ref_seeds),
            ref_directions,
            ref_second_seeds,
            expected,
        ),
        optional_derivative_orders(
            chosen, seeds, second_inputs, directions, second_seeds, actual
        ),
    ):
        for a, b in zip(reference, result):
            if a is None:
                assert b is None or bool((b == 0).all())
            else:
                assert_optional_close(
                    torch.zeros_like(a, dtype=dtype) if b is None else b, a
                )


@pytest.mark.cuda
@pytest.mark.parametrize("feature", ["envelope", "l0"])
@pytest.mark.parametrize("trainable", ["edge", "weight"])
def test_optional_force_training(make_problem, feature, trainable, optional_dtype):
    torch.manual_seed(274)
    tp, x, y, w, src, dst = make_problem("uvu_conv", False, dtype=optional_dtype)
    x, y, w = (x.detach(), y.detach(), w.detach())
    parameters = (y if trainable == "edge" else w).requires_grad_()
    positions = torch.randn(len(x), 3, device="cuda", requires_grad=True)
    ref_x, ref_y, ref_w, ref_positions = fp64((x, y, w, positions))
    ref_parameters = ref_y if trainable == "edge" else ref_w
    ref_radius = (
        (ref_positions[src] - ref_positions[dst]).square().sum(-1, keepdim=True)
    )
    ref_y, ref_w = ref_y * torch.exp(-ref_radius), ref_w * (1 + ref_radius)
    ref_env = torch.exp(-0.2 * ref_radius)
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
    expected = reference_shape_outputs(
        deepcopy(tp).double(),
        ref_x,
        ref_y,
        ref_w,
        src,
        dst,
        None if l0 else ref_env,
        l0,
    )
    for a, b in zip(
        force_loss_derivatives(ref_positions, ref_parameters, expected),
        force_loss_derivatives(positions, parameters, actual),
    ):
        assert_reference_close(b, a, dtype=positions.dtype)


@pytest.mark.cuda
def test_l0_fp64_wide_channels():
    """Cover FP64 streaming stores, channel chunks and a partial final warp."""
    test_complete_optional_derivatives(make_wide_problem, "l0", torch.float64, "all")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("first_input", ["all", "edge", "weight", "envelope"])
@pytest.mark.parametrize("output", ["both", "node", "l0"])
def test_combined_mixed_derivatives(dtype, first_input, output):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype)
    env[::2] = 0
    env.requires_grad_()
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, True
    )
    assert actual[1].shape == (len(y), 3 * 17)
    if output != "both":
        selection = slice(0, 1) if output == "node" else slice(1, 2)
        actual, expected = (actual[selection], expected[selection])
    inputs = (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    chosen = {"all": inputs, "edge": (y,), "weight": (w,), "envelope": (env,)}[
        first_input
    ]
    ref_chosen = tuple(ref_by_id[id(value)] for value in chosen)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in chosen)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    if output == "l0" and first_input == "envelope":
        gradient = torch.autograd.grad(actual, env, seeds, allow_unused=True)[0]
        assert gradient is None or bool((gradient == 0).all())
        return
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, chosen, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_chosen,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_combined_zero_scalar_columns(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "5x1o",
            "2x0e",
            "5x1o",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    torch.manual_seed(92381)
    src = torch.tensor([1, 0, 1, 0], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    x = torch.randn(
        3, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        4, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(4, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True)
    env = torch.randn(4, 1, device="cuda", dtype=dtype, requires_grad=True)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    assert actual[1].shape == (len(y), 0)
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, True
    )
    inputs = (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_inputs,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_combined_frozen_envelope_and_empty_edges(dtype, empty):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype, empty=empty)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype)
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, True
    )
    inputs = (x, y, w)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_inputs,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_combined_nondefault_stream_and_capture(dtype):
    tp, x, y, w, src, dst = make_shape_problem(17, dtype)
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    env = torch.randn(len(y), 1, device=x.device, dtype=dtype, requires_grad=True)
    inputs = (x, y, w, env)
    ref_inputs = fp64(inputs)
    expected = reference_shape_outputs(
        deepcopy(tp).double(), *ref_inputs[:3], src, dst, ref_inputs[3], True
    )
    seeds = tuple(
        (torch.randn_like(value, dtype=dtype, requires_grad=True) for value in expected)
    )
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second = fp64((seeds, directions, second_seeds))
    expected_chain = derivative_chain(
        expected, ref_inputs, ref_inputs, ref_seeds, ref_directions, ref_second
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x, y, w, env = (value.detach().clone().requires_grad_() for value in inputs)
        inputs = (x, y, w, env)
        seeds = tuple((value.detach().clone().requires_grad_() for value in seeds))
        directions = tuple(
            (value.detach().clone().requires_grad_() for value in directions)
        )
        second_seeds = tuple((value.clone() for value in second_seeds))
        for _ in range(3):
            actual = derivative_chain(
                op(x, y, w, graph=graph, env=env, return_l0_feat=True),
                inputs,
                inputs,
                seeds,
                directions,
                second_seeds,
            )
    torch.cuda.current_stream().wait_stream(stream)
    for actual_order, expected_order in zip(actual, expected_chain):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        captured = derivative_chain(
            op(x, y, w, graph=graph, env=env, return_l0_feat=True),
            inputs,
            inputs,
            seeds,
            directions,
            second_seeds,
        )
    for _ in range(2):
        capture.replay()
    torch.cuda.synchronize()
    for actual_order, expected_order in zip(captured, expected_chain):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("feature", ["l0", "combined"])
def test_high_irrep_nonunit_edge_derivatives(dtype, feature):
    """Exercise the existing staged FP64 factor schedule with nonunit copies."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "17x5o",
            "2x5o",
            "17x0e",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    x = torch.randn(
        3, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        4, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(4, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True)
    src = torch.tensor([1, 0, 2, 0], device="cuda")
    dst = torch.tensor([0, 0, 1, 1], device="cuda")
    env = (
        torch.randn(4, 1, device="cuda", dtype=dtype, requires_grad=True)
        if feature == "combined"
        else None
    )
    op = TensorProductConv(tp, dtype=dtype)
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, w, graph=graph, env=env, return_l0_feat=True)
    ref_x, ref_y, ref_w, ref_env = fp64((x, y, w, env))
    ref_by_id = dict(zip(map(id, (x, y, w, env)), (ref_x, ref_y, ref_w, ref_env)))
    expected = reference_shape_outputs(
        deepcopy(tp).double(), ref_x, ref_y, ref_w, src, dst, ref_env, True
    )
    inputs = (x, y, w) if env is None else (x, y, w, env)
    ref_inputs = tuple(ref_by_id[id(value)] for value in inputs)
    seeds = tuple((torch.randn_like(value, requires_grad=True) for value in actual))
    directions = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs)
    )
    second_seeds = tuple((torch.randn_like(value) for value in (*inputs, *seeds)))
    ref_seeds, ref_directions, ref_second_seeds = fp64(
        (seeds, directions, second_seeds)
    )
    for actual_order, expected_order in zip(
        derivative_chain(actual, inputs, inputs, seeds, directions, second_seeds),
        derivative_chain(
            expected,
            ref_inputs,
            ref_inputs,
            ref_seeds,
            ref_directions,
            ref_second_seeds,
        ),
    ):
        for value, reference in zip(actual_order, expected_order):
            assert_shape_close(value, reference, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_graph_forward_and_three_orders(dtype):
    torch.manual_seed(1042)
    src = torch.tensor([0, 2, 1, 2, 0, 1, 1], device="cuda")
    dst = torch.tensor([2, 0, 2, 1, 0, 1, 2], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = make_graph(src, dst, 4)
    irreps = "32x0e + 32x1o"
    edge_irreps = "1x0e + 1x1o"
    ir, edge = (o3.Irreps(irreps), o3.Irreps(edge_irreps))
    instructions = [
        (i, j, k, "uvu", True)
        for i, (_, a) in enumerate(ir)
        for j, (_, b) in enumerate(edge)
        for k, (_, c) in enumerate(ir)
        if c in a * b
    ]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        reference = o3.TensorProduct(
            ir, edge, ir, instructions, internal_weights=False, shared_weights=False
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    module = TensorProductConv(reference)
    x = torch.randn(4, ir.dim, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(len(src), edge.dim, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(
        len(src), reference.weight_numel, device="cuda", dtype=dtype, requires_grad=True
    )
    actual = module(x, y, weight, graph=graph)
    config = prepare_tensor_product(
        ir, edge, ir, instructions, dtype=dtype, convolution=True
    )
    functional = tensor_product_conv(x, y, weight, config=config, graph=graph)
    grad = torch.randn_like(actual, requires_grad=True)
    inputs = (x, y, weight, grad)
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
    )
    second_seeds = tuple((torch.randn_like(value) for value in inputs))
    ref_inputs, ref_first, ref_second = fp64((inputs, first_seeds, second_seeds))
    messages = deepcopy(reference).double()(
        ref_inputs[0][src], ref_inputs[1], ref_inputs[2]
    )
    expected = messages.new_zeros(4, ir.dim).index_add(0, dst, messages)
    assert_reference_close(actual, expected)
    assert_reference_close(functional, expected)
    results = []
    for output, targets, directions, second_directions in (
        (expected, ref_inputs, ref_first, ref_second),
        (actual, inputs, first_seeds, second_seeds),
    ):
        first = torch.autograd.grad(
            output, targets[:3], targets[3], create_graph=True, retain_graph=True
        )
        second = torch.autograd.grad(
            first, targets, directions, create_graph=True, retain_graph=True
        )
        third = torch.autograd.grad(
            second, (*targets, *directions), second_directions, retain_graph=True
        )
        results.append((*first, *second, *third))
    for expected_grad, actual_grad in zip(*results):
        assert_reference_close(actual_grad, expected_grad)


@pytest.mark.cuda
@pytest.mark.parametrize("env_shape", [(4,), (4, 1)])
def test_tensor_product_conv_envelope_compile(env_shape):
    from LunaEquivariance import TensorProductConv

    torch.manual_seed(42)
    module = TensorProductConv(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device="cuda", dtype=torch.float64)
    graph = module.create_graph(
        torch.tensor([1, 2, 0, 1], device="cuda"),
        torch.tensor([0, 0, 1, 2], device="cuda"),
        3,
    )
    args = tuple(
        (
            torch.randn(shape, device="cuda", dtype=torch.float64).requires_grad_()
            for shape in ((3, 6), (4, 6), (4, module.weight_numel), env_shape)
        )
    )

    def forward(x, edge_attr, weight, env):
        return module(x, edge_attr, weight, graph=graph, env=env)

    expected = forward(*args)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    actual = compiled(*args)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.sin().sum(), args)
    actual_grads = torch.autograd.grad(actual.sin().sum(), args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grads[-1].shape == env_shape


@pytest.fixture(scope="module")
def convolution():
    mode = "uvu"
    return TensorProductConv(
        "2x0e + 2x1o",
        "2x0e + 2x1o" if mode == "uuu" else "1x0e + 1x1o",
        "2x0e + 2x1o",
        [
            (0, 0, 0, mode, True),
            (1, 1, 0, mode, True),
            (0, 1, 1, mode, True),
            (1, 0, 1, mode, True),
        ],
        internal_weights=False,
        shared_weights=False,
        dtype=torch.float64,
    ).cuda()


def make_case(module, *, with_env=True, strided=False, graph=None):
    if graph is None:
        graph = make_graph(
            torch.tensor([2, 0, 2, 1, 0], device="cuda"),
            torch.tensor([0, 1, 1, 2, 2], device="cuda"),
            3,
        )
    values = []
    shapes = [
        (3, module.irreps_in1.dim),
        (5, module.irreps_in2.dim),
        (5, module.weight_numel),
    ] + ([(5, 1)] if with_env else [])
    for i, (rows, width) in enumerate(shapes):
        columns = width * (2 if strided else 1)
        value = torch.linspace(
            0.1 + i * 0.07,
            0.8 + i * 0.09,
            rows * columns,
            device="cuda",
            dtype=torch.float64,
        ).reshape(rows, columns)
        if strided:
            value = value[:, ::2]
        values.append(value.requires_grad_())
    if not with_env:
        values.append(None)
    return (graph, tuple(values))


def compile_case(module, graph, values, order, *, return_l0=False, unused_weight=False):
    if unused_weight:
        values = (*values[:2], values[2].detach(), values[3])
    shape = (5, module.op.l0_dim) if return_l0 else (3, module.irreps_out.dim)
    seed = torch.linspace(
        0.2, 0.6, shape[0] * shape[1], device="cuda", dtype=torch.float64
    ).reshape(shape)

    def function(x, y, weight, env, seed):
        kwargs = {"return_l0_feat": True} if return_l0 else {}
        result = module(x, y, weight, graph=graph, env=env, **kwargs)
        outputs = (result[1] if return_l0 else result,)
        targets = tuple(
            (
                value
                for value in (x, y, weight, env)
                if value is not None and value.requires_grad
            )
        )
        directions = (seed,)
        for step in range(order):
            gradients = torch.autograd.grad(
                outputs,
                targets,
                directions,
                create_graph=step + 1 < order,
                retain_graph=True,
                allow_unused=True,
            )
            outputs = tuple((value for value in gradients if value is not None))
            if step + 1 < order:
                outputs = tuple((value for value in outputs if value.requires_grad))
                directions = tuple((torch.ones_like(value) for value in outputs))
        return outputs

    args = (*values, seed)
    expected = function(*args)
    traced = make_fx(function, tracing_mode="fake", _allow_non_fake_inputs=True)(*args)
    traced.graph.eliminate_dead_code()
    traced.recompile()
    compiled = torch.compile(
        traced, fullgraph=True, options={"triton.cudagraphs": False}
    )
    with torch.no_grad():
        actual = compiled(*args)
    torch.cuda.synchronize()
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=3e-11, atol=3e-12)
    return (compiled, args, expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "order,with_env,return_l0",
    [
        (0, False, False),
        (1, False, False),
        (2, True, False),
        (3, True, False),
        (1, True, True),
        (3, False, True),
    ],
)
def test_compiled_conv_derivatives(convolution, order, with_env, return_l0):
    graph, values = make_case(convolution, with_env=with_env)
    compile_case(convolution, graph, values, order, return_l0=return_l0)


@pytest.mark.cuda
def test_compiled_conv_strided_inputs_and_unused_weight(convolution):
    graph, values = make_case(convolution, strided=True)
    assert all((not value.is_contiguous() for value in values))
    compile_case(convolution, graph, values, 3, unused_weight=True)


def compilation_case(channels):
    irreps = lunaeq.Irreps(f"{channels}x0e + {channels}x1o")
    edge_irreps = lunaeq.Irreps("2x0e + 2x1o")
    instructions = [(i, j, k, "uvu", True)
                    for i, (_, a) in enumerate(irreps)
                    for j, (_, b) in enumerate(edge_irreps)
                    for k, (_, c) in enumerate(irreps) if c in a * b]
    module = lunaeq.TensorProductConv(irreps, edge_irreps, irreps, instructions,
                          internal_weights=False, shared_weights=False).cuda()
    weight_numel = module.weight_numel
    module = ConvolutionInputs(module)
    samples = []
    for rows, edges in ((7, 17), (19, 41)):
        x = (torch.randn(rows, irreps.dim, device="cuda") * 0.3).requires_grad_()
        edge = (torch.randn(edges, edge_irreps.dim, device="cuda") * 0.3).requires_grad_()
        weight = (torch.randn(edges, weight_numel, device="cuda") * 0.3).requires_grad_()
        src = (torch.arange(edges, device="cuda") * 3 + 1) % rows
        dst = (torch.arange(edges, device="cuda") % rows).sort().values
        graph = lunaeq.make_graph(src, dst, rows)
        samples.append(((x, edge, weight, graph_tensors(graph)), {}))
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
