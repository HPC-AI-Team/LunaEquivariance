"""Radial networks, grouped inputs and cache lifetime."""

import importlib
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from LunaEquivariance.so2 import radial_network
from LunaEquivariance.so2.radial_network import interface as runtime
from LunaEquivariance.so2.radial_network.interface import RadialParameters
from tests.so2.reference.radial import radial_network as radial_reference
from tests.so2.checks import (
    assert_derivative_slots,
    compare_operator_derivatives,
    first_and_second_gradients,
)


class LaunchSpy:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, kwargs))

        return launch


@pytest.fixture
def restore_deterministic_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def prepare_call(monkeypatch, grouped, operation):
    name = "grouped_layer_norm_silu" if grouped else "layer_norm_silu"
    stage = "double_backward" if operation == "double_backward" else "backward"
    module = import_module(f"LunaEquivariance.so2.radial_network.kernels.{stage}")
    inputs = torch.empty((2, 3, 128) if grouped else (3, 128))
    weight = torch.empty((2, 128) if grouped else (128,))
    stats = torch.empty(6 if grouped else 3)
    monkeypatch.setattr(
        module, f"{name}_validate", lambda *_: (2, 3, 128) if grouped else (3, 128)
    )
    spy = LaunchSpy()
    kernel = (
        "double_backward_kernel"
        if operation == "double_backward"
        else "backward_kernel"
    )
    if not grouped:
        kernel = "layer_norm_silu_" + kernel
    elif stage == "backward":
        kernel = "grouped_layer_norm_silu_" + kernel
    monkeypatch.setattr(module, kernel, spy)
    args = [inputs, inputs]
    if grouped:
        args.append(weight)
    args.extend([weight, weight, stats, stats])
    if operation == "double_backward":
        args.append(inputs)
        if grouped:
            args.append(weight)
        args.extend([None, None])
    elif grouped:
        args.append(4)
    return lambda: getattr(module, f"{name}_{operation}")(*args), spy


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("operation", ["backward", "double_backward"])
def test_atomic_gradient_launches_honor_deterministic_mode(
    monkeypatch, restore_deterministic_mode, grouped, operation
):
    call, spy = prepare_call(monkeypatch, grouped, operation)
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="atomic reductions"):
        call()
    assert not spy.calls

    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(RuntimeWarning, match="atomic reductions"):
        call()
    assert len(spy.calls) == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_input_only_gradient_remains_available_in_deterministic_mode(
    monkeypatch, restore_deterministic_mode, grouped
):
    call, spy = prepare_call(monkeypatch, grouped, "backward_input")
    torch.use_deterministic_algorithms(True)
    call()
    assert len(spy.calls) == 1
    assert spy.calls[0][1]["compute_parameter_gradients"] is False


@pytest.mark.cuda
def test_radial_mlp_no_grad_matches_training_forward():
    from LunaEquivariance.so2.radial_network.interface import (
        RadialParameters,
        fused_radial_mlp,
    )

    def tensor(*shape):
        return torch.randn(*shape, device="cuda", requires_grad=True)

    parameters = RadialParameters(
        tensor(16, 8),
        tensor(16),
        tensor(16),
        tensor(16),
        1e-5,
        tensor(16, 16),
        tensor(16),
        tensor(16),
        tensor(16),
        1e-5,
        tensor(12, 16),
        tensor(12),
    )
    x = tensor(7, 8)
    expected = fused_radial_mlp(parameters, x)
    with torch.no_grad():
        actual = fused_radial_mlp(parameters, x)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.cuda
def test_grouped_radial_no_grad_matches_training_forward():
    from LunaEquivariance.so2.radial_network.interface import (
        grouped_radial_first,
    )
    from LunaEquivariance.so2.radial_network.interface import fused_layer_norm_silu

    def tensor(*shape):
        return torch.randn(*shape, device="cuda", requires_grad=True)

    args = (
        tensor(7, 4),
        torch.tensor([0, 1, 2, 0, 1, 2, 0], device="cuda"),
        torch.tensor([1, 2, 0, 1, 2, 0, 1], device="cuda"),
        tuple(tensor(128, 8) for _ in range(2)),
        tuple(tensor(128) for _ in range(2)),
        tuple(tensor(3, 2) for _ in range(2)),
        tuple(tensor(3, 2) for _ in range(2)),
        tuple(tensor(128) for _ in range(2)),
        tuple(tensor(128) for _ in range(2)),
    )
    options = dict(distance_channels=4, embedding_channels=2, norm_eps=1e-5)
    expected = grouped_radial_first(*args, **options)
    weight, bias = tensor(128), tensor(128)
    tail = fused_layer_norm_silu(expected[0], weight, bias, 1e-5)
    with torch.no_grad():
        actual = grouped_radial_first(*args, **options)
        actual_tail = fused_layer_norm_silu(actual[0], weight, bias, 1e-5)
    for observed, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, wanted)
        assert not observed.requires_grad
    torch.testing.assert_close(actual_tail, tail)


def layer_norm_silu_reference(x, input_bias, weight, bias):
    value = x + input_bias[:, None]
    centered = value - value.mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
    return torch.nn.functional.silu(normalized * weight[:, None] + bias[:, None])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "warps,trainable,compiled",
    [
        (1, False, False),
        (1, False, True),
        (8, True, False),
        (8, False, True),
    ],
)
def test_radial_norm_derivatives(warps, trainable, compiled):
    from LunaEquivariance.so2.radial_network.interface import fused_grouped_layer_norm_silu

    parameters = [
        torch.randn(4, 128, device="cuda", requires_grad=trainable) for _ in range(3)
    ]

    def operation(x):
        torch._dynamo.mark_static(x, 2)
        return fused_grouped_layer_norm_silu(
            x,
            *parameters,
            1e-5,
            forward_num_warps=1 if warps == 1 else 8,
            backward_num_warps=4,
        )

    if compiled:
        operation = torch.compile(operation, fullgraph=True, dynamic=True)
    for rows in [31, 53]:
        x = torch.randn(4, rows, 128, device="cuda", requires_grad=True)
        leaves = (x, *parameters) if trainable else (x,)
        actual = operation(x)
        expected = layer_norm_silu_reference(x, *parameters)
        cotangent = torch.randn_like(actual)
        observed = torch.autograd.grad(
            actual, leaves, cotangent, create_graph=trainable
        )
        wanted = torch.autograd.grad(
            expected, leaves, cotangent, create_graph=trainable
        )
        for a, b in [(actual, expected), *zip(observed, wanted)]:
            assert torch.isfinite(a).all()
            assert (a - b).double().norm() / b.double().norm() < 1e-5
        if trainable:
            directions = [torch.randn_like(value) for value in leaves]
            observed_hvp = torch.autograd.grad(observed, leaves, directions)
            wanted_hvp = torch.autograd.grad(wanted, leaves, directions)
            for a, b in zip(observed_hvp, wanted_hvp):
                assert torch.isfinite(a).all()
                assert (a - b).double().norm() / b.double().norm() < 2e-5


@pytest.fixture
def radial_case(monkeypatch):
    torch.manual_seed(731)
    first_calls = []

    def first_stage(distance, source, receiver, weights, biases, sources, targets,
                    norm_weights, norm_biases, *, norm_eps, **kwargs):
        first_calls.append(len(weights))
        outputs = []
        for weight, bias, source_table, target_table, scale, shift in zip(
            weights, biases, sources, targets, norm_weights, norm_biases, strict=True,
        ):
            inputs = torch.cat((distance, source_table[source], target_table[receiver]), -1)
            hidden = F.linear(inputs, weight, bias)
            outputs.append(F.silu(F.layer_norm(hidden, (hidden.shape[-1],), scale, shift, norm_eps)))
        return torch.stack(outputs)

    monkeypatch.setattr(runtime, "grouped_radial_first", first_stage)
    monkeypatch.setattr(runtime, "fused_layer_norm_silu",
                        lambda x, w, b, eps: F.silu(F.layer_norm(x, (x.shape[-1],), w, b, eps)))
    def parameter(*shape):
        return torch.nn.Parameter(torch.randn(*shape, dtype=torch.float64) * 0.3)
    params = tuple(RadialParameters(parameter(7, 11), parameter(7), parameter(7), parameter(7), 1e-5,
                                   parameter(9, 7), parameter(9), parameter(9), parameter(9), 2e-5,
                                   parameter(4, 9), parameter(4)) for _ in range(2))
    sources = tuple(parameter(4, 3) for _ in params)
    targets = tuple(parameter(4, 3) for _ in params)
    distance = parameter(6, 5)
    source = torch.tensor([0, 2, 3, 1, 2, 0])
    receiver = torch.tensor([1, 0, 2, 3, 1, 2])
    owner = runtime.AtomPairRadialRuntime(params, sources, targets, distance_channels=5)
    return owner, distance, source, receiver, first_calls


def runtime_reference(owner, index, distance, source, receiver):
    params = owner.parameters[index]
    hidden = torch.cat((distance, owner.source_embeddings[index][source],
                        owner.target_embeddings[index][receiver]), -1)
    hidden = F.linear(hidden, params.first_weight, params.first_bias)
    hidden = F.silu(F.layer_norm(hidden, (hidden.shape[-1],), params.first_norm_weight,
                                params.first_norm_bias, params.first_eps))
    hidden = F.linear(hidden, params.second_weight, params.second_bias)
    hidden = F.silu(F.layer_norm(hidden, (hidden.shape[-1],), params.second_norm_weight,
                                params.second_norm_bias, params.second_eps))
    return F.linear(hidden, params.final_weight, params.final_bias)


def test_grouped_first_stage_and_consumed_hidden_lifetimes(radial_case):
    owner, distance, source, receiver, calls = radial_case
    first = owner.consume(0, distance, source, receiver)
    assert calls == [2]
    assert owner.hidden[0] is None and owner.hidden[1] is not None
    torch.testing.assert_close(first, runtime_reference(owner, 0, distance, source, receiver))
    second = owner.consume(1, distance, source, receiver)
    assert calls == [2]
    assert owner.hidden is None and owner.input_key is None
    torch.testing.assert_close(second, runtime_reference(owner, 1, distance, source, receiver))
    params = tuple(value for group in owner.parameters for value in group if isinstance(value, torch.Tensor))
    targets = (distance, *params, *owner.source_embeddings, *owner.target_embeddings)
    actual = first.square().sum() + second.square().sum()
    expected = sum(runtime_reference(owner, i, distance, source, receiver).square().sum() for i in range(2))
    actual_grads = torch.autograd.grad(actual, targets, create_graph=True)
    expected_grads = torch.autograd.grad(expected, targets, create_graph=True)
    for got, wanted in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, wanted)
    directions = tuple(torch.randn_like(value) for value in actual_grads)
    actual_second = torch.autograd.grad(actual_grads, targets, directions, allow_unused=True)
    expected_second = torch.autograd.grad(expected_grads, targets, directions, allow_unused=True)
    torch.testing.assert_close(actual_second, expected_second)


def test_interruption_wrong_order_and_changed_input_reset_cache(radial_case):
    owner, distance, source, receiver, calls = radial_case
    first = owner.consume(0, distance, source, receiver)
    torch.testing.assert_close(owner.consume(0, distance, source, receiver), first)
    assert calls == [2, 2]
    with pytest.raises(RuntimeError, match="block order"):
        owner.consume(2, distance, source, receiver)
    assert owner.hidden is None
    owner.consume(0, distance, source, receiver)
    with pytest.raises(RuntimeError, match="share one radial input"):
        owner.consume(1, distance.clone(), source, receiver)
    assert owner.hidden is None
    with pytest.raises(RuntimeError, match="share one radial input"):
        owner.consume(1, distance, source, receiver)


def test_deepcopy_resets_hidden_and_preserves_copied_parameter_identity(radial_case):
    owner, distance, source, receiver, _ = radial_case
    model = torch.nn.Module()
    model.parameters_for_radial = torch.nn.ParameterList(
        [value for group in owner.parameters for value in group if isinstance(value, torch.Tensor)]
        + list(owner.source_embeddings) + list(owner.target_embeddings)
    )
    model.runtime = owner
    owner.consume(0, distance, source, receiver)
    copied = deepcopy(model)
    assert copied.runtime.hidden is None and copied.runtime.next_index == 0
    assert copied.runtime.parameters[0].first_weight is copied.parameters_for_radial[0]
    assert copied.runtime.parameters[0].first_weight is not owner.parameters[0].first_weight
    torch.testing.assert_close(copied.runtime.consume(0, distance, source, receiver),
                               runtime_reference(copied.runtime, 0, distance, source, receiver))


def test_incompatible_parameter_groups_rejected(radial_case):
    owner, _, _, _, _ = radial_case
    bad = owner.parameters[1]._replace(first_eps=3e-5)
    with pytest.raises(ValueError, match="uniform"):
        runtime.AtomPairRadialRuntime((owner.parameters[0], bad), owner.source_embeddings,
                                     owner.target_embeddings, distance_channels=5)


radial_equations = importlib.import_module("tests.so2.reference.radial")


FIELDS = (
    "first_weight",
    "first_bias",
    "first_norm_weight",
    "first_norm_bias",
    "second_weight",
    "second_bias",
    "second_norm_weight",
    "second_norm_bias",
    "final_weight",
    "final_bias",
)


def make_parameters(values, record=SimpleNamespace):
    fields = dict(zip(FIELDS, values, strict=True))
    return record(**fields, first_eps=1e-3, second_eps=2e-3)


def case(name, *, final_bias=True, device="cpu", dtype=torch.float64):
    edges, cin, h1, h2, cout = 5, 4, 7, 5, 3

    def make(*shape):
        return (0.2 * torch.randn(shape, device=device, dtype=dtype)).requires_grad_()

    packed = name == "packed_radial_net"
    mlp_cin = cin + 2 * 3 if packed else cin
    parameters = (
        make(h1, mlp_cin),
        make(h1),
        make(h1),
        make(h1),
        make(h2, h1),
        make(h2),
        make(h2),
        make(h2),
        make(cout, h2),
    )
    parameters = (*parameters, *((make(cout),) if final_bias else ()))
    if not packed:
        values = (make(edges, cin), *parameters)

        def call(function, args, parameter_class=SimpleNamespace):
            parameters = args[1:] if final_bias else (*args[1:], None)
            return function(make_parameters(parameters, parameter_class), args[0])

        return values, call

    values = (make(edges, cin), make(4, 3), make(4, 3), *parameters)
    atoms = torch.tensor([1, 2, 1, 0], device=device)
    edge_index = torch.tensor([[0, 0, 2, 3, 1], [1, 2, 1, 1, 0]], device=device)

    def call(function, args, parameter_class=None):
        fields = dict(zip(FIELDS, args[3:], strict=True))
        fields["output_weight"] = fields.pop("final_weight")
        fields["output_bias"] = fields.pop("final_bias")
        return function(
            args[0],
            atoms,
            edge_index,
            args[1],
            args[2],
            **fields,
            first_norm_eps=1e-3,
            second_norm_eps=2e-3,
        )

    return values, call


def derivatives(output, values, cotangent, directions):
    first = torch.autograd.grad(output, values, cotangent, create_graph=True)
    contraction = sum((a * b).sum() for a, b in zip(first, directions, strict=True))
    second = torch.autograd.grad(contraction, (*values, cotangent), allow_unused=True)
    return (output, *first, *second)


@pytest.mark.parametrize(
    "name,final_bias",
    (
        ("fused_radial_mlp", True),
        ("fused_radial_mlp", False),
        ("packed_radial_net", True),
    ),
)
def test_radial_reference_gradcheck_and_gradgradcheck(name, final_bias):
    torch.manual_seed(819)
    values, call = case(name, final_bias=final_bias)

    def function(*args):
        return call(getattr(radial_equations, name), args)

    assert torch.autograd.gradcheck(function, values, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, values, fast_mode=True)


def test_radial_reference_matches_sequential_linear_layernorm_silu():
    torch.manual_seed(820)
    values, call = case("fused_radial_mlp")
    layers = torch.nn.Sequential(
        torch.nn.Linear(4, 7),
        torch.nn.LayerNorm(7, eps=1e-3),
        torch.nn.SiLU(),
        torch.nn.Linear(7, 5),
        torch.nn.LayerNorm(5, eps=2e-3),
        torch.nn.SiLU(),
        torch.nn.Linear(5, 3),
    ).double()
    module_parameters = tuple(layers.parameters())
    assert len(module_parameters) == len(values) - 1
    with torch.no_grad():
        for destination, source in zip(module_parameters, values[1:], strict=True):
            destination.copy_(source)
    actual = call(radial_equations.fused_radial_mlp, values)
    expected = layers(values[0])
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "name,final_bias",
    (
        ("fused_radial_mlp", True),
        ("fused_radial_mlp", False),
        ("packed_radial_net", True),
    ),
)
@pytest.mark.parametrize("mode", ("full", "input_only", "parameters_only"))
def test_radial_operators_match_fp64_reference_through_double_backward(
    name, final_bias, mode
):
    pytest.importorskip("triton")
    optimized = importlib.import_module("LunaEquivariance.so2.radial_network.interface")
    torch.manual_seed(821)
    values, call = case(name, final_bias=final_bias, device="cuda", dtype=torch.float32)
    if mode == "input_only":
        for value in values[1:]:
            value.requires_grad_(False)
    elif mode == "parameters_only":
        values[0].requires_grad_(False)
    copied = tuple(
        value.detach().double().requires_grad_(value.requires_grad) for value in values
    )
    active = tuple(value for value in values if value.requires_grad)
    reference_active = tuple(value for value in copied if value.requires_grad)
    actual = call(getattr(optimized, name), values, optimized.RadialParameters)
    expected = call(getattr(radial_equations, name), copied)
    cotangent = torch.randn_like(actual, requires_grad=True)
    directions = tuple(torch.randn_like(value) for value in active)
    actual_values = derivatives(actual, active, cotangent, directions)
    expected_values = derivatives(
        expected,
        reference_active,
        cotangent.double(),
        tuple(value.double() for value in directions),
    )
    for observed, wanted in zip(actual_values, expected_values, strict=True):
        if wanted is None:
            assert observed is None or not torch.count_nonzero(observed)
        elif observed is None:
            assert not torch.count_nonzero(wanted)
        else:
            torch.testing.assert_close(observed.double(), wanted, atol=5e-5, rtol=5e-4)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "field", ("first_bias", "first_norm_weight", "first_norm_bias")
)
@pytest.mark.parametrize("invalid", ("stride", "device"))
def test_grouped_radial_affine_contract_is_checked_before_launch(field, invalid):
    from LunaEquivariance.so2.radial_network.interface import RadialParameters
    from LunaEquivariance.so2.radial_network.interface import fused_radial_mlp

    values, _ = case("fused_radial_mlp", device="cuda", dtype=torch.float32)
    parameters = make_parameters(values[1:], RadialParameters)
    parameter = getattr(parameters, field)
    if invalid == "stride":
        parameter = torch.empty(2 * parameter.numel(), device="cuda")[::2]
    else:
        parameter = parameter.cpu()
    parameters = parameters._replace(**{field: parameter})
    with pytest.raises(
        ValueError,
        match="affine parameters must match input device/dtype and be contiguous",
    ):
        fused_radial_mlp(parameters, values[0])


@pytest.mark.cuda
def test_radial_mlp_asymmetric_nonpower_widths():
    from LunaEquivariance.so2.radial_network.interface import (
        RadialParameters,
        fused_radial_mlp,
    )

    torch.manual_seed(2026)
    shapes = [
        (9, 7),
        (13, 7),
        (13,),
        (13,),
        (13,),
        (19, 13),
        (19,),
        (19,),
        (19,),
        (11, 19),
        (11,),
    ]
    data = tuple(torch.randn(shape, device="cuda") * 0.2 for shape in shapes)
    directions = tuple(torch.randn_like(v) * 0.1 for v in data)
    cotangent_data = torch.randn(9, 11, device="cuda")
    results = []
    for reference in (False, True):
        dtype = torch.float64 if reference else torch.float32
        inputs = tuple(v.to(dtype).detach().requires_grad_() for v in data)
        x, w1, b1, nw1, nb1, w2, b2, nw2, nb2, w3, b3 = inputs
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        if reference:
            hidden = torch.nn.functional.linear(x, w1, b1)
            hidden = torch.nn.functional.silu(
                torch.nn.functional.layer_norm(hidden, (13,), nw1, nb1, 1e-5)
            )
            hidden = torch.nn.functional.linear(hidden, w2, b2)
            hidden = torch.nn.functional.silu(
                torch.nn.functional.layer_norm(hidden, (19,), nw2, nb2, 1e-5)
            )
            output = torch.nn.functional.linear(hidden, w3, b3)
        else:
            parameters = RadialParameters(
                w1, b1, nw1, nb1, 1e-5, w2, b2, nw2, nb2, 1e-5, w3, b3
            )
            output = fused_radial_mlp(parameters, x)
        results.append(
            first_and_second_gradients(output, inputs, cotangent, tuple(v.to(dtype) for v in directions))
        )
    assert_derivative_slots(*results, rtol=2e-3, atol=8e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("channels", [1, 13, 19, 128])
@pytest.mark.parametrize("direction_slot", [None, 0, 1, 2])
def test_ungrouped_radial_norm_dimensions(channels, direction_slot):
    from LunaEquivariance.so2.radial_network.interface import fused_layer_norm_silu

    torch.manual_seed(2026)
    data = tuple(
        torch.randn(shape, device="cuda") * 0.2
        for shape in [(2, 7, channels), (channels,), (channels,)]
    )
    directions = tuple(
        torch.randn_like(v) * 0.1
        if direction_slot is None or direction_slot == i
        else torch.zeros_like(v)
        for i, v in enumerate(data)
    )
    cotangent_data = torch.randn_like(data[0])
    results = []
    for reference in (False, True):
        dtype = torch.float64 if reference else torch.float32
        inputs = tuple(v.to(dtype).detach().requires_grad_() for v in data)
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        if reference:
            output = torch.nn.functional.silu(
                torch.nn.functional.layer_norm(
                    inputs[0], (channels,), inputs[1], inputs[2], 1e-5
                )
            )
        else:
            output = fused_layer_norm_silu(*inputs, 1e-5)
        results.append(
            first_and_second_gradients(output, inputs, cotangent, tuple(v.to(dtype) for v in directions))
        )
    assert_derivative_slots(*results, rtol=2e-3, atol=8e-5)


def radial_inputs(*, groups=None, indexed=False, device="cuda", output_bias=True):
    torch.manual_seed(731)
    prefix = () if groups is None else (groups,)
    def tensor(*shape):
        return (torch.randn(*shape, device=device) * 0.3).requires_grad_()
    distance, embedding, first, second, output = 5, 3, 7, 9, 4
    cin = distance + 2 * embedding if indexed else distance
    inputs = tensor(6, distance)
    weights = (tensor(*prefix, first, cin), tensor(*prefix, second, first),
               tensor(*prefix, output, second))
    biases = (tensor(*prefix, first), tensor(*prefix, second),
              tensor(*prefix, output) if output_bias else None)
    norm_weights = (tensor(*prefix, first), tensor(*prefix, second))
    norm_biases = (tensor(*prefix, first), tensor(*prefix, second))
    kwargs = {"eps": (1e-5, 2e-5)}
    if indexed:
        kwargs.update(
            atom_types=torch.tensor([2, 0, 1, 2], device=device),
            edge_index=torch.tensor([[0, 1, 3, 2, 0, 3], [1, 0, 0, 3, 2, 2]], device=device),
            embeddings=(tensor(*prefix, 3, embedding), tensor(*prefix, 3, embedding)),
        )
    return (inputs, weights, biases, norm_weights, norm_biases), kwargs


@pytest.mark.cuda
@pytest.mark.parametrize("groups", [None, 1, 3])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("output_bias", [False, True])
def test_radial_parameters_and_input_double_backward(groups, indexed, output_bias):
    args, kwargs = radial_inputs(groups=groups, indexed=indexed, output_bias=output_bias)
    compare_operator_derivatives(radial_network, radial_reference, args, kwargs)


def test_radial_reference_grouping_keeps_networks_independent():
    args, kwargs = radial_inputs(groups=2, indexed=True, device="cpu")
    result = radial_reference(*args, **kwargs)
    inputs, weights, biases, norm_weights, norm_biases = args
    for group in range(2):
        select = lambda values: tuple(value[group] for value in values)
        single = radial_reference(inputs, select(weights), select(biases),
                                          select(norm_weights), select(norm_biases),
                                          **dict(kwargs, embeddings=select(kwargs["embeddings"])))
        torch.testing.assert_close(result[group], single)


def test_invalid_radial_topology():
    args, kwargs = radial_inputs(device="cpu")
    inputs, weights, biases, norm_weights, norm_biases = args
    with pytest.raises(ValueError, match="three linear"):
        radial_network(inputs, weights[:2], biases, norm_weights, norm_biases)
    with pytest.raises(ValueError, match="eps"):
        radial_network(*args, eps=0)
    with pytest.raises(ValueError, match="normalization"):
        radial_network(inputs, weights, biases, norm_weights[:1], norm_biases)
