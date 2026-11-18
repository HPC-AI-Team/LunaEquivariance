"""Initial atom features and distance-based radial inputs."""

import pytest
import torch

from tests.so2.checks import make_cotangent, strided_copy


from LunaEquivariance.so2.inverse_rotate_reduce import interface as optimized
from tests.so2.reference import inverse_rotation_reduction as reference
from tests.so2.reference.radial import fused_radial_input as reference_radial_input
from LunaEquivariance.so2.radial_network.interface import fused_radial_input


def case(name, device="cpu", dtype=torch.float64):
    def make(*shape):
        return (0.2 * torch.randn(shape, device=device, dtype=dtype)).requires_grad_()

    atoms = torch.tensor([2, 1, 2, 0], dtype=torch.int64, device=device)
    if name == "fused_initial_atom":
        values = (make(4, 4, 3), make(5, 3))
        return values, lambda function, a: function(a[0].clone(), a[1], atoms)
    if name == "fused_initial_scalar":
        values = (make(4, 4, 3), make(4, 3))
        return values, lambda function, a: function(a[0].clone(), a[1])
    if name == "fused_radial_input":
        edges = torch.tensor([[0, 0, 2, 3, 1], [1, 2, 1, 1, 0]], device=device)
        values = (make(5, 2), make(5, 3), make(5, 3))
        return values, lambda function, a: function(*a, atoms, edges)
    raise AssertionError(name)


NAMES = ("fused_initial_atom", "fused_initial_scalar", "fused_radial_input")


def test_atom_update_changes_only_scalars_and_returns_same_storage():
    x = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
    original = x.clone()
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=x.dtype)
    result = reference.fused_initial_atom(x, weight, torch.tensor([1, 0]))
    assert result is x
    torch.testing.assert_close(result[:, 0], original[:, 0] + weight.flip(0))
    torch.testing.assert_close(result[:, 1:], original[:, 1:])


def test_radial_feature_channel_order_and_repeated_atom_gradients():
    distance = torch.tensor([[10.0], [20.0]], dtype=torch.float64, requires_grad=True)
    source = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    receiver = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64, requires_grad=True
    )
    result = reference_radial_input(
        distance,
        source,
        receiver,
        torch.tensor([1, 1, 0]),
        torch.tensor([[0, 1], [2, 2]]),
    )
    expected = torch.tensor(
        [[10.0, 3.0, 4.0, 5.0, 6.0], [20.0, 3.0, 4.0, 5.0, 6.0]], dtype=distance.dtype
    )
    torch.testing.assert_close(result, expected)
    first = torch.autograd.grad(result.sum(), (source, receiver))
    torch.testing.assert_close(
        first[0], torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=source.dtype)
    )
    torch.testing.assert_close(
        first[1], torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=receiver.dtype)
    )


@pytest.mark.parametrize("name", NAMES)
def test_embedding_reference_gradcheck_and_gradgradcheck(name):
    torch.manual_seed(817)
    values, call = case(name)

    def function(*args):
        return call((reference_radial_input if name == "fused_radial_input" else getattr(reference, name)), args)

    assert torch.autograd.gradcheck(function, values, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, values, fast_mode=True)


def derivatives(output, values, cotangent, directions):
    first = torch.autograd.grad(output, values, cotangent, create_graph=True)
    contraction = sum((a * b).sum() for a, b in zip(first, directions, strict=True))
    second = torch.autograd.grad(contraction, (*values, cotangent), allow_unused=True)
    return (output, *first, *second)


@pytest.mark.cuda
@pytest.mark.parametrize("name", NAMES)
def test_optimized_embedding_matches_reference_and_training_derivatives(name):
    pytest.importorskip("triton")
    torch.manual_seed(818)
    values, call = case(name, device="cuda", dtype=torch.float32)
    copied = tuple(value.detach().double().requires_grad_() for value in values)
    actual = call((fused_radial_input if name == "fused_radial_input" else getattr(optimized, name)), values)
    expected = call((reference_radial_input if name == "fused_radial_input" else getattr(reference, name)), copied)
    cotangent = torch.randn_like(actual, requires_grad=True)
    directions = tuple(torch.randn_like(value) for value in values)
    actual_values = derivatives(actual, values, cotangent, directions)
    expected_values = derivatives(
        expected,
        copied,
        cotangent.double(),
        tuple(value.double() for value in directions),
    )
    for observed, wanted in zip(actual_values, expected_values, strict=True):
        if wanted is None:
            assert observed is None or not torch.count_nonzero(observed)
        elif observed is None:
            assert not torch.count_nonzero(wanted)
        else:
            torch.testing.assert_close(observed.double(), wanted, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("operation", ("atom", "scalar", "radial"))
@pytest.mark.parametrize(
    "invalid", ("strided_weight", "different_device", "half_precision")
)
def test_embedding_external_parameter_contracts_without_a_gpu(operation, invalid):
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as embedding

    x = torch.empty(2, 4, 3)
    shape = (2, 3) if operation == "scalar" else (5, 3)
    weight = torch.empty(shape)
    if invalid == "strided_weight":
        weight = torch.empty(shape[0], 6)[:, ::2]
    elif invalid == "different_device":
        weight = torch.empty(shape, device="meta")
    else:
        weight = weight.half()
    atoms = torch.empty(2, dtype=torch.int64)
    if operation == "atom":
        call = lambda: embedding.fused_initial_atom(x, weight, atoms)
    elif operation == "scalar":
        call = lambda: embedding.fused_initial_scalar(x, weight)
    else:
        call = lambda: fused_radial_input(
            torch.empty(3, 4),
            weight,
            torch.empty(5, 3),
            atoms,
            torch.empty(2, 3, dtype=torch.int64),
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


@pytest.mark.parametrize("operation", ("atom", "scalar"))
@pytest.mark.parametrize("view", (False, True))
def test_initial_embedding_rejects_autograd_leaves_before_launch(
    monkeypatch, operation, view
):
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as scalar

    def unexpected_launch(*args):
        pytest.fail("invalid in-place input reached the kernel")

    monkeypatch.setattr(scalar.AtomAdd, "apply", unexpected_launch)
    monkeypatch.setattr(scalar.ScalarAdd, "apply", unexpected_launch)
    leaf = torch.randn(2, 4, 3, requires_grad=True)
    x = leaf.view_as(leaf) if view else leaf
    original, version = leaf.detach().clone(), leaf._version
    with pytest.raises(RuntimeError, match="leaf tensor"):
        if operation == "atom":
            scalar.fused_initial_atom(
                x, torch.empty(5, 3), torch.empty(2, dtype=torch.int64)
            )
        else:
            scalar.fused_initial_scalar(x, torch.empty(2, 3))
    torch.testing.assert_close(leaf, original, rtol=0, atol=0)
    assert leaf._version == version


@pytest.mark.cuda
@pytest.mark.parametrize("operation", ("atom", "scalar"))
@pytest.mark.parametrize("kind", ("sum", "strided"))
def test_initial_embedding_inplace_and_strided_derivatives_gpu(operation, kind):
    from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
        fused_initial_atom,
        fused_initial_scalar,
    )

    torch.manual_seed(191)
    features = torch.randn(3, 4, 7, device="cuda", requires_grad=True)
    indices = torch.tensor([0, 2, 0], device="cuda")
    parameters = torch.randn(
        4 if operation == "atom" else 3, 7, device="cuda", requires_grad=True
    )
    nodes = features.clone()
    before = nodes._version
    if operation == "atom":
        actual = fused_initial_atom(nodes, parameters, indices)
        scalar = parameters.index_select(0, indices)
    else:
        actual = fused_initial_scalar(nodes, parameters)
        scalar = parameters
    assert actual is nodes
    assert nodes._version > before
    expected = torch.cat((features[:, :1] + scalar[:, None], features[:, 1:]), dim=1)
    cotangent = make_cotangent(actual, kind).requires_grad_()
    first = torch.autograd.grad(
        actual, (features, parameters), cotangent, create_graph=True
    )
    reference_first = torch.autograd.grad(
        expected, (features, parameters), cotangent, create_graph=True
    )
    directions = tuple(
        strided_copy(torch.randn_like(value)) for value in (features, parameters)
    )
    (second,) = torch.autograd.grad(first, cotangent, directions)
    (reference_second,) = torch.autograd.grad(reference_first, cotangent, directions)
    for value, reference in zip(
        (actual, *first, second),
        (expected, *reference_first, reference_second),
        strict=True,
    ):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)
