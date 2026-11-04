"""UUU CONV outputs, layouts, equivariance and derivatives."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct, wigner_3j
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
    tp, x, y, w, src, dst = make_problem("uuu_conv", shared)
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
    tp, x, y, w, src, dst = make_problem("uuu_conv", shared, dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uuu_conv", shared, dtype=torch.float64)
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
    tp, x, y, w, src, dst = make_problem("uuu_conv", shared, dtype=torch.float64)
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
    reference, features, edges, weight, src, dst = make_problem("uuu_conv", dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uuu_conv")
    module = TensorProductConv(tp)
    kwargs = {"graph": module.create_graph(src, dst, len(x))}
    module(x, y, w, **kwargs)
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem("uuu_conv", dtype=dtype)
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
    tp, x, y, w, src, dst = make_problem("uuu_conv", dtype=dtype)
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
    "internal,shared,env_shape",
    [
        (False, False, (4,)),
        (False, False, (4, 1)),
        (False, True, (4,)),
        (False, True, (4, 1)),
        (True, True, (4,)),
        (True, True, (4, 1)),
    ],
)
def test_cuda_tensor_product_modes(dtype, internal, shared, env_shape):
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(42)
    cls = TensorProductConv
    module = cls(
        "2x1o",
        "2x1o",
        "2x0e",
        [(0, 0, 0, "uuu", True)],
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
    assert_cuda_report(module, args, **kwargs)


def test_irreps_constructor_registers_internal_weights():
    module = TensorProductConv(
        "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uuu", True)], dtype=torch.float64
    )
    assert module.internal_weights and module.shared_weights
    assert module.weight.dtype == torch.float64
    assert module.weight is dict(module.named_parameters())["tp.weight"]
    before = module.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(module.weight, before - 0.1)


def make_tensor_product(dtype, *, shared=False, weighted=True):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        irreps = Irreps("3x0e + 3x1o + 3x2e")
        instructions = [
            (0, 0, 0, "uuu", weighted),
            (1, 1, 0, "uuu", weighted),
            (0, 1, 1, "uuu", weighted),
            (1, 0, 1, "uuu", weighted),
            (1, 1, 2, "uuu", weighted),
        ]
        tp = E3nnTensorProduct(
            irreps,
            irreps,
            irreps,
            instructions,
            internal_weights=shared and weighted,
            shared_weights=shared,
        )
    finally:
        torch.set_default_dtype(previous)
    return tp


def scalar_features(tp, x, y, weight):
    features = []
    weight_offset = 0
    in1, in2 = (tp.irreps_in1.slices(), tp.irreps_in2.slices())
    for instruction in tp.instructions:
        channels, a = tp.irreps_in1[instruction.i_in1]
        _, b = tp.irreps_in2[instruction.i_in2]
        _, c = tp.irreps_out[instruction.i_out]
        if c.l == 0:
            left = x[:, in1[instruction.i_in1]].reshape(len(x), channels, a.dim)
            right = y[:, in2[instruction.i_in2]].reshape(len(y), channels, b.dim)
            coefficient = wigner_3j(a.l, b.l, c.l, dtype=x.dtype, device=x.device)
            value = torch.einsum("eui,euj,ijk->euk", left, right, coefficient)
            value = value.squeeze(-1) * instruction.path_weight
            if instruction.has_weight:
                value = value * weight[..., weight_offset : weight_offset + channels]
            features.append(value)
        if instruction.has_weight:
            weight_offset += channels
    return torch.cat(features, dim=-1)


def differentiate(outputs, inputs, seeds):
    selected = [
        (value, seed)
        for value, seed in zip(outputs, seeds)
        if value is not None and value.requires_grad and (seed is not None)
    ]
    return torch.autograd.grad(
        tuple((value for value, _ in selected)),
        inputs,
        tuple((seed for _, seed in selected)),
        allow_unused=True,
        create_graph=True,
        retain_graph=True,
    )


def derivative_chain(outputs, values, seeds, first_seeds, second_seeds):
    first = differentiate(outputs, values, seeds)
    second_inputs = (*values, *(seed for seed in seeds if seed is not None))
    second = differentiate(first, second_inputs, first_seeds)
    third = differentiate(second, (*second_inputs, *first_seeds), second_seeds)
    return (first, second, third)


def assert_orders_match(actual, expected, dtype):
    for actual_order, expected_order in zip(actual, expected):
        for value, reference in zip(actual_order, expected_order):
            assert (value is None) == (reference is None)
            if value is not None:
                assert_reference_close(value, reference)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train_env", [False, True])
@pytest.mark.parametrize("scalar_only", [False, True])
def test_envelope_and_scalar_features_through_triple_backward(
    dtype, train_env, scalar_only
):
    torch.manual_seed(712)
    tp = make_tensor_product(dtype).cuda()
    op = TensorProductConv(tp, dtype=dtype)
    x = (
        torch.randn(
            tp.irreps_in1.dim, 5, dtype=dtype, device="cuda", requires_grad=True
        ).t()
        * 0.7
    )
    y = (
        torch.randn(
            tp.irreps_in2.dim, 11, dtype=dtype, device="cuda", requires_grad=True
        ).t()
        * 0.8
    )
    weight = torch.randn(
        11, tp.weight_numel, dtype=dtype, device="cuda", requires_grad=True
    )
    env = torch.randn(11, 1, dtype=dtype, device="cuda", requires_grad=train_env)
    src = torch.tensor([1, 0, 2, 0, 3, 1, 2, 0, 1, 3, 2], device="cuda")
    dst = torch.tensor([3, 0, 1, 2, 0, 1, 3, 1, 0, 2, 3], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    y, weight, env = (y[order], weight[order], env[order])
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, weight, graph=graph, env=env, return_l0_feat=True)
    ref_x, ref_y, ref_weight, ref_env = fp64((x, y, weight, env))
    reference_tp = deepcopy(tp).double()
    messages = reference_tp(ref_x[src], ref_y, ref_weight) * ref_env
    expected = (
        messages.new_zeros((len(x), tp.irreps_out.dim)).index_add(0, dst, messages),
        scalar_features(reference_tp, ref_x[src], ref_y, ref_weight),
    )
    assert_orders_match((actual,), (expected,), dtype)
    values = (x, y, weight) + ((env,) if train_env and (not scalar_only) else ())
    seeds = (
        None if scalar_only else torch.randn_like(actual[0], requires_grad=True),
        torch.randn_like(actual[1], requires_grad=True),
    )
    first_seeds = tuple(
        (torch.randn_like(value, requires_grad=True) for value in values)
    )
    second_inputs = (*values, *(seed for seed in seeds if seed is not None))
    second_seeds = tuple((torch.randn_like(value) for value in second_inputs))
    ref_values = dict(
        zip(map(id, (x, y, weight, env)), (ref_x, ref_y, ref_weight, ref_env))
    )
    ref_targets = tuple(ref_values[id(value)] for value in values)
    ref_seeds, ref_first, ref_second = fp64((seeds, first_seeds, second_seeds))
    assert_orders_match(
        derivative_chain(actual, values, seeds, first_seeds, second_seeds),
        derivative_chain(expected, ref_targets, ref_seeds, ref_first, ref_second),
        dtype,
    )
    first = torch.autograd.grad(
        actual[0], x, torch.ones_like(actual[0]), create_graph=True, retain_graph=True
    )[0]
    reference = torch.autograd.grad(
        expected[0],
        ref_x,
        torch.ones_like(expected[0]),
        create_graph=True,
        retain_graph=True,
    )[0]
    seed = torch.randn_like(x)
    second = torch.autograd.grad(first, (y, weight), seed, retain_graph=True)
    expected_second = torch.autograd.grad(
        reference, (ref_y, ref_weight), seed.double(), retain_graph=True
    )
    assert_orders_match((second,), (expected_second,), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_internal_or_unweighted_paths_with_destination_pointer(dtype, weighted):
    tp = make_tensor_product(dtype, shared=True, weighted=weighted).cuda()
    op = TensorProductConv(tp, dtype=dtype)
    src = torch.tensor([1, 0, 2, 0], device="cuda")
    dst = torch.tensor([0, 1, 1, 2], device="cuda")
    graph = op.create_graph(src, dst, 4)
    x = torch.randn(
        4, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        4, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    actual = op(x, y, graph=graph)
    ref_x, ref_y, ref_weight = fp64((x, y, tp.weight))
    messages = deepcopy(tp).double()(ref_x[src], ref_y, ref_weight)
    expected = messages.new_zeros((4, tp.irreps_out.dim)).index_add(0, dst, messages)
    assert_orders_match(((actual,),), ((expected,),), dtype)
    values = (x, y) + ((tp.weight,) if weighted else ())
    actual_grad = torch.autograd.grad(actual.sum(), values)
    ref_values = (ref_x, ref_y) + ((ref_weight,) if weighted else ())
    expected_grad = torch.autograd.grad(expected.sum(), ref_values)
    assert_orders_match((actual_grad,), (expected_grad,), dtype)


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
    edge_irreps = irreps
    ir, edge = (o3.Irreps(irreps), o3.Irreps(edge_irreps))
    instructions = [
        (i, j, k, "uuu", True)
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
        [(0, 0, 0, "uuu", True)],
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
    mode = "uuu"
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
    edge_irreps = irreps
    instructions = [(i, j, k, "uuu", True)
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
