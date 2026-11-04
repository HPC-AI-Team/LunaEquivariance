"""FP32 symmetric contractions against repeated Clebsch–Gordan products."""

from itertools import product

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

import LunaEquivariance as lunaeq
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
)


def reference_contraction(x, weights, attrs, irreps_in, irreps_out, correlation):
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    channels = max(mul for mul, _ in (*irreps_in, *irreps_out))
    features, offset = [], 0
    for copies, irrep in irreps_in:
        if x.ndim == 3:
            value = x[..., offset:offset + irrep.dim]
            offset += irrep.dim
        else:
            value = x[:, offset:offset + copies * irrep.dim].reshape(x.shape[0], copies, irrep.dim)
            value = torch.nn.functional.pad(value, (0, 0, 0, channels - copies))
            offset += copies * irrep.dim
        features.append((irrep, value))
    products = {1: features}
    for degree in range(2, correlation + 1):
        values = features
        for order in range(2, degree + 1):
            coupled = []
            for left_irrep, left in values:
                for right_irrep, right in features:
                    for irrep in left_irrep * right_irrep:
                        if degree == 4 and (irrep.l >= 12 or irrep.p != (-1) ** irrep.l):
                            continue
                        coefficients = o3.wigner_3j(
                            irrep.l, left_irrep.l, right_irrep.l, dtype=x.dtype, device=x.device
                        ) * irrep.dim ** .5
                        coupled.append((irrep, torch.einsum("nci,ncj,kij->nck", left, right, coefficients)))
            values = sorted(coupled, key=lambda item: item[0])
        products[degree] = values
    outputs, path, weight_offset = [], 0, 0
    for copies, irrep in irreps_out:
        output = x.new_zeros(x.shape[0], copies, irrep.dim)
        for degree in range(correlation, 0, -1):
            matches = [value for candidate, value in products[degree] if candidate == irrep]
            for value in matches:
                if weights.ndim == 3:
                    parameter = weights[:, path]
                else:
                    parameter = weights[:, weight_offset:weight_offset + copies]
                mixed_weight = torch.einsum("ns,sc->nc", attrs, parameter)
                output = output + mixed_weight[..., None] * value[:, :copies]
                path += 1
                weight_offset += copies
            if not matches:
                path += 1
                weight_offset += copies
        outputs.append(output.reshape(x.shape[0], copies * irrep.dim))
    assert (path if weights.ndim == 3 else weight_offset) == weights.shape[1]
    return torch.cat(outputs, dim=-1)


def mixed_derivatives(output, inputs, grad_out, directions, second_directions):
    first = torch.autograd.grad(output, inputs, grad_out, create_graph=True, retain_graph=True)
    selected = [(value, seed) for value, seed in zip(first, directions) if seed is not None]
    second = torch.autograd.grad(
        tuple(value for value, _ in selected), (*inputs, grad_out),
        tuple(seed for _, seed in selected), create_graph=True, retain_graph=True, allow_unused=True,
    )
    selected = [(value, seed) for value, seed in zip(second, second_directions)
                if value is not None and value.requires_grad and seed is not None]
    third = torch.autograd.grad(
        tuple(value for value, _ in selected),
        (*inputs, grad_out, *(seed for seed in directions if seed is not None)),
        tuple(seed for _, seed in selected), allow_unused=True, retain_graph=True,
    )
    return output, *first, *second, *third


@pytest.fixture(params=[("4x0e+4x1o", "4x0e+4x1o", 3),
                       ("3x0e+3x1o+3x2e", "3x0e+3x1o", 2),
                       ("7x0e+5x1o+3x2e", "5x0e+3x1o", 3),
                       ("4x0e+4x1o", "4x0e+4x1o", 4),
                       ("4x0e+3x1o+2x2e", "4x0e+3x1o", 4)],
                ids=["L1-correlation3", "L2-correlation2", "nonuniform-correlation3",
                     "L1-correlation4", "nonuniform-correlation4"])
def problem(request):
    torch.manual_seed(1937)
    irreps_in, irreps_out, correlation = request.param
    # The ordered-path reference below uses unreduced weights.
    module = lunaeq.SymmetricContraction(
        irreps_in, irreps_out, correlation, num_elements=3,
        use_reduced_cg=False, device="cuda", dtype=torch.float32,
    )
    channels = o3.Irreps(irreps_in)[0].mul
    components = sum(irrep.dim for _, irrep in o3.Irreps(irreps_in))
    shape = (7, channels, components) if module.meta["input_layout"] == "channels" else (7, module.irreps_in.dim)
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * .3).requires_grad_()
    attrs = torch.nn.functional.one_hot(torch.randint(3, (7,), device="cuda"), 3).float()
    return module, x, attrs


def test_public_entries_and_state_dict():
    assert lunaeq.o3.SymmetricContraction is lunaeq.SymmetricContraction
    assert lunaeq.o3.symmetric_contraction is lunaeq.symmetric_contraction
    assert lunaeq.LunaEq.SymmetricContraction is lunaeq.SymmetricContraction
    assert lunaeq.LunaEq.symmetric_contraction is lunaeq.symmetric_contraction
    module = lunaeq.SymmetricContraction("4x0e+4x1o", "4x0e+4x1o", num_elements=3)
    restored = lunaeq.SymmetricContraction("4x0e+4x1o", "4x0e+4x1o", num_elements=3)
    restored.load_state_dict(module.state_dict())
    assert module.use_reduced_cg
    assert module.weight.shape == (3, 9, 4)
    assert module.weight.requires_grad
    assert restored.config == module.config
    torch.testing.assert_close(restored.weight, module.weight, rtol=0, atol=0)


@pytest.mark.parametrize("correlation,paths", [(2, 9), (4, 83), (6, 532)])
def test_reduced_basis_configuration(correlation, paths):
    module = lunaeq.SymmetricContraction(
        "4x0e+4x1o+4x2e+4x3o", "4x0e+4x1o", correlation,
        num_elements=3, use_reduced_cg=True,
    )
    restored = lunaeq.SymmetricContraction(
        module.irreps_in, module.irreps_out, correlation, num_elements=3, use_reduced_cg=True,
    )
    restored.load_state_dict(module.state_dict())
    assert module.weight.shape == (3, paths, 4)
    assert restored.use_reduced_cg
    assert restored.config == module.config
    torch.testing.assert_close(restored.weight, module.weight, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("irreps_in,irreps_out", [
    ("4x0e+4x1o", "4x0e+4x1o"),
    ("4x0e+3x1o+2x2e", "4x0e+3x1o"),
])
def test_reduced_basis_parameter_gradients(irreps_in, irreps_out):
    torch.manual_seed(1943)
    raw = lunaeq.SymmetricContraction(
        irreps_in, irreps_out, 4, num_elements=3, use_reduced_cg=False, device="cuda",
    )
    reduced = lunaeq.SymmetricContraction(irreps_in, irreps_out, 4, num_elements=3,
                                         use_reduced_cg=True, device="cuda")
    mapping = torch.zeros(reduced.meta["weight_numel"], raw.meta["weight_numel"], device="cuda")
    for basis, members in enumerate(raw.meta["cg_dag"]["weight_members"]):
        target = reduced.meta["path_offsets"][basis]
        channels = reduced.meta["path_channels"][basis]
        for path, coefficient in members:
            source = raw.meta["path_offsets"][path]
            mapping[target:target + channels, source:source + channels] += coefficient * torch.eye(channels, device="cuda")
    weights = (raw.weight.flatten(1) @ mapping.t()).reshape_as(reduced.weight)
    shape = ((3, raw.meta["channels"], raw.meta["input_dim"])
             if raw.meta["input_layout"] == "channels" else (3, raw.meta["input_size"]))
    x = (torch.randn(shape, device="cuda") * .7).requires_grad_()
    attrs = torch.randn(3, 3, device="cuda", requires_grad=True)
    actual = lunaeq.symmetric_contraction(x, weights, attrs, irreps_in, irreps_out, 4, use_reduced_cg=True)
    expected = raw(x, attrs)
    inputs = (x, raw.weight, attrs)
    grad_out = torch.randn_like(actual).requires_grad_()
    directions = tuple((torch.randn_like(value) * .2).requires_grad_() for value in inputs)
    second_directions = tuple(torch.randn_like(value) * .2 for value in (*inputs, grad_out))
    for value, target in zip(mixed_derivatives(actual, inputs, grad_out, directions, second_directions),
                             mixed_derivatives(expected, inputs, grad_out, directions, second_directions)):
        if value is None or target is None:
            assert value is None and target is None
        else:
            torch.testing.assert_close(value, target, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("kwargs,match", [
    (dict(irreps_in=""), "nonempty"),
    (dict(irreps_out=""), "nonempty"),
    (dict(irreps_in="0x0e+4x1o"), "multiplicities"),
    (dict(irreps_out="0x0e"), "multiplicities"),
    (dict(correlation=0), "positive"),
    (dict(num_elements=0), "positive"),
    (dict(dtype=torch.float16), "float32 and float64"),
])
def test_invalid_configuration(kwargs, match):
    arguments = dict(irreps_in="4x0e+4x1o", irreps_out="4x0e+4x1o", num_elements=3)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        lunaeq.SymmetricContraction(**arguments)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["one_hot", "soft", "zero", "strided", "empty"])
def test_forward_through_triple_backward(problem, kind):
    module, x, attrs = problem
    weights = module.weight
    if kind in ("soft", "strided"):
        attrs = torch.randn_like(attrs) * 0.3
    elif kind == "zero":
        attrs = torch.zeros_like(attrs)
    elif kind == "empty":
        x, attrs = x[:0], attrs[:0]
    if kind == "strided":
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2).detach()
        weights = weights.transpose(0, 1).contiguous().transpose(0, 1).detach()
        attrs = attrs.t().contiguous().t()
        assert not any(value.is_contiguous() for value in (x, weights, attrs))
    inputs = tuple(value.detach().requires_grad_() for value in (x, weights, attrs))
    actual = lunaeq.symmetric_contraction(
        *inputs, module.irreps_in, module.irreps_out, module.correlation,
        use_reduced_cg=module.use_reduced_cg,
    )
    ref_inputs = fp64(inputs)
    expected = reference_contraction(
        *ref_inputs, module.irreps_in, module.irreps_out, module.correlation
    )
    grad_out = (torch.randn_like(actual) * 0.3).requires_grad_()
    directions = tuple(
        (torch.randn_like(value) * 0.3).requires_grad_() for value in inputs
    )
    second_directions = tuple(
        torch.randn_like(value) * 0.3 for value in (*inputs, grad_out)
    )
    for value, target in zip(
        mixed_derivatives(actual, inputs, grad_out, directions, second_directions),
        mixed_derivatives(
            expected, ref_inputs, *fp64((grad_out, directions, second_directions))
        ),
    ):
        assert_reference_close(value, target)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "needs", [mask for mask in product((False, True), repeat=3) if any(mask)]
)
def test_requested_input_gradients(problem, needs):
    module, x, attrs = problem
    inputs = tuple(
        value.detach().requires_grad_(needed)
        for value, needed in zip((x, module.weight, attrs), needs)
    )
    targets = tuple(value for value in inputs if value.requires_grad)
    actual = lunaeq.symmetric_contraction(
        *inputs, module.irreps_in, module.irreps_out, module.correlation,
        use_reduced_cg=module.use_reduced_cg,
    )
    ref_inputs = fp64(inputs)
    ref_targets = tuple(value for value in ref_inputs if value.requires_grad)
    expected = reference_contraction(
        *ref_inputs, module.irreps_in, module.irreps_out, module.correlation
    )
    grad_out = torch.randn_like(actual)
    for value, target in zip(
        torch.autograd.grad(actual, targets, grad_out),
        torch.autograd.grad(expected, ref_targets, grad_out.double()),
    ):
        assert_reference_close(value, target)


@pytest.mark.cuda
def test_frozen_weights_through_triple_backward(problem):
    module, x, attrs = problem
    module.requires_grad_(False)
    actual = module(x, attrs)
    assert actual.grad_fn.name() == "SymmetricContractionForwardBackward"
    ref_x, ref_weight, ref_attrs = fp64((x, module.weight, attrs))
    expected = reference_contraction(
        ref_x,
        ref_weight,
        ref_attrs,
        module.irreps_in,
        module.irreps_out,
        module.correlation,
    )
    grad_out = (torch.randn_like(actual) * 0.3).requires_grad_()
    direction = (torch.randn_like(x) * 0.3).requires_grad_()
    seeds = (torch.randn_like(x) * 0.3, torch.randn_like(grad_out) * 0.3)
    for value, target in zip(
        mixed_derivatives(actual, (x,), grad_out, (direction,), seeds),
        mixed_derivatives(expected, (ref_x,), *fp64((grad_out, (direction,), seeds))),
    ):
        assert_reference_close(value, target)
    assert module.weight.grad is None


@pytest.mark.cuda
@pytest.mark.parametrize("operand", ["features", "weights", "attributes"])
def test_invalid_input_shape(problem, operand):
    module, x, attrs = problem
    weights = module.weight
    if operand == "features":
        x = x[..., :-1]
    elif operand == "weights":
        weights = weights[:, :-1]
    else:
        attrs = attrs[:-1]
    with pytest.raises(RuntimeError, match="shape"):
        lunaeq.symmetric_contraction(
            x, weights, attrs, module.irreps_in, module.irreps_out, module.correlation,
            use_reduced_cg=module.use_reduced_cg,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("first_seed,second_seed", [(0, 0), (1, 0), (2, 0), (None, 3)])
def test_missing_backward_directions(problem, first_seed, second_seed):
    module, x, attrs = problem
    inputs = (x, module.weight, attrs.requires_grad_())
    actual = module(x, attrs)
    ref_inputs = fp64(inputs)
    expected = reference_contraction(
        *ref_inputs, module.irreps_in, module.irreps_out, module.correlation
    )
    grad_out = (torch.randn_like(actual) * 0.3).requires_grad_()
    directions = tuple(
        (torch.randn_like(value) * 0.3).requires_grad_()
        if first_seed is None or first_seed == index
        else None
        for index, value in enumerate(inputs)
    )
    second_directions = tuple(
        torch.randn_like(value) * 0.3 if second_seed == index else None
        for index, value in enumerate((*inputs, grad_out))
    )
    for value, target in zip(
        mixed_derivatives(actual, inputs, grad_out, directions, second_directions),
        mixed_derivatives(
            expected, ref_inputs, *fp64((grad_out, directions, second_directions))
        ),
    ):
        if value is None or target is None:
            remaining = target if value is None else value
            assert remaining is None or not torch.count_nonzero(remaining)
        else:
            assert_reference_close(value, target)


@pytest.mark.cuda
def test_compiled_training_updates(problem):
    module, x, attrs = problem
    reference_weight = torch.nn.Parameter(module.weight.detach().double())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD([reference_weight], lr=0.01)
    compiled = torch.compile(module, fullgraph=True)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = compiled(x.detach(), attrs)
        expected = reference_contraction(
            x.detach().double(),
            reference_weight,
            attrs.double(),
            module.irreps_in,
            module.irreps_out,
            module.correlation,
        )
        actual.square().mean().backward()
        expected.square().mean().backward()
        assert_reference_close(actual, expected)
        assert_reference_close(module.weight.grad, reference_weight.grad)
        optimizer.step()
        reference_optimizer.step()
        assert_reference_close(module.weight, reference_weight)


@pytest.mark.cuda
def test_compiled_second_derivative_training(problem):
    module, x, attrs = problem

    def predictions(features, weights, attributes):
        output = lunaeq.symmetric_contraction(
            features,
            weights,
            attributes,
            module.irreps_in,
            module.irreps_out,
            module.correlation,
            use_reduced_cg=module.use_reduced_cg,
        )
        first = torch.autograd.grad(output.square().sum(), features, create_graph=True)[
            0
        ]
        second = torch.autograd.grad(first.sum(), features, create_graph=True)[0]
        return output, first, second

    predictions(x, module.weight, attrs)
    graph = make_fx(predictions)(x, module.weight, attrs)
    compiled = torch.compile(graph, fullgraph=True)
    actual = compiled(x, module.weight, attrs)
    ref_x, ref_weight, ref_attrs = fp64((x, module.weight, attrs))
    expected_output = reference_contraction(
        ref_x,
        ref_weight,
        ref_attrs,
        module.irreps_in,
        module.irreps_out,
        module.correlation,
    )
    expected_first = torch.autograd.grad(
        expected_output.square().sum(), ref_x, create_graph=True
    )[0]
    expected_second = torch.autograd.grad(
        expected_first.sum(), ref_x, create_graph=True
    )[0]
    expected = (expected_output, expected_first, expected_second)
    actual_gradient = torch.autograd.grad(
        sum(value.square().mean() for value in actual), module.weight
    )[0]
    expected_gradient = torch.autograd.grad(
        sum(value.square().mean() for value in expected), ref_weight
    )[0]
    for value, target in zip(
        (*actual, actual_gradient), (*expected, expected_gradient)
    ):
        assert_reference_close(value, target)


def compilation_case(channels):
    irreps = f"{channels}x0e + {channels}x1o"
    module = lunaeq.SymmetricContraction(irreps, irreps, 2, num_elements=2, device="cuda")
    samples = []
    for rows in (7, 19):
        shape = (rows, channels, 4) if module.meta["input_layout"] == "channels" else (rows, 4 * channels)
        x = (torch.randn(shape, device="cuda") * 0.3).requires_grad_()
        attrs = torch.nn.functional.one_hot(torch.arange(rows, device="cuda") % 2, 2).float()
        samples.append(((x, attrs.requires_grad_()), {}))
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (2, 4))
def test_compile(channels):
    module, samples = compilation_case(channels)
    assert_compiled(module, samples)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (2, 4)]
    assert_aoti(cases, tmp_path)
