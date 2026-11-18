"""Input layouts and derivative checks shared by SO(2) tests."""

from types import SimpleNamespace

import torch
from torch.utils._pytree import tree_leaves, tree_map_only

from tests.testing_utils import assert_reference_close


def fp64(values):
    """Copy floating leaves in FP64 and preserve integer graph indices."""
    return tree_map_only(
        torch.Tensor,
        lambda value: (
            value.detach().double().requires_grad_(value.requires_grad)
            if value.is_floating_point()
            else value.detach().clone()
        ),
        values,
    )


def compare_operator_derivatives(function, ref_function, args, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    ref_args, ref_kwargs = fp64((args, kwargs))
    targets = tuple(value for value in tree_leaves((args, kwargs))
                    if isinstance(value, torch.Tensor) and value.requires_grad)
    ref_targets = tuple(value for value in tree_leaves((ref_args, ref_kwargs))
                        if isinstance(value, torch.Tensor) and value.requires_grad)
    actual, expected = (function(*args, **kwargs),), (ref_function(*ref_args, **ref_kwargs),)
    for order in range(3):
        for value, wanted in zip(actual, expected, strict=True):
            if value is None or wanted is None:
                assert value is wanted
            else:
                assert_reference_close(value, wanted)
        if order == 2:
            break
        pairs = [(value, wanted) for value, wanted in zip(actual, expected, strict=True)
                 if value is not None and value.requires_grad]
        seeds = tuple(torch.randn_like(value, requires_grad=True) for value, _ in pairs)
        ref_seeds = fp64(seeds)
        actual = torch.autograd.grad(tuple(value for value, _ in pairs), targets, seeds,
                                     create_graph=order == 0, allow_unused=True)
        expected = torch.autograd.grad(tuple(wanted for _, wanted in pairs), ref_targets,
                                       ref_seeds, create_graph=order == 0, allow_unused=True)
        targets, ref_targets = (*targets, *seeds), (*ref_targets, *ref_seeds)


def output_derivatives(output, parameters, seed, directions):
    first = torch.autograd.grad(output, parameters, seed, create_graph=True)
    second = torch.autograd.grad(first, (*parameters, seed), directions, allow_unused=True)
    return (output, *first, *(torch.zeros_like(v) if g is None else g
                              for v,g in zip((*parameters,seed), second)))


def strided_copy(value):
    backing = value.new_full((*value.shape[:-1], 2 * value.shape[-1]), 17)
    result = backing[..., ::2]
    result.copy_(value)
    return result


def make_cotangent(value, kind):
    if kind == "sum":
        return value.new_ones(()).expand_as(value)
    return strided_copy(torch.ones_like(value))


def first_and_second_gradients(output, inputs, cotangent, directions):
    first = torch.autograd.grad(output, inputs, cotangent, create_graph=True)
    second = torch.autograd.grad(
        first, (*inputs, cotangent), directions, allow_unused=True
    )
    return output, *first, *second


def assert_derivative_slots(actual, expected, rtol=1e-8, atol=1e-9):
    for a, b in zip(actual, expected, strict=True):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            torch.testing.assert_close(a.double(), b.double(), rtol=rtol, atol=atol)


def tiny_so2_graph(device):
    # Two receivers, multiple senders and one isolated node.
    src = torch.tensor([1, 2, 0, 2, 0, 1], device=device, dtype=torch.int32)
    dst = torch.tensor([0, 0, 1, 1, 2, 2], device=device, dtype=torch.int32)
    src_order = src.argsort(stable=True).to(torch.int32)
    source_csr = torch.tensor([0, 2, 4, 6, 6], device=device, dtype=torch.int32)
    receiver_csr = torch.tensor([0, 2, 4, 6, 6], device=device, dtype=torch.int32)
    return SimpleNamespace(src=src, dst=dst, num_nodes=4, edge_ids_by_src=src_order,
                           src_rowptr=source_csr, dst_rowptr=receiver_csr, _tables={})


def derivative_chain(output, targets, cotangent, directions):
    first = torch.autograd.grad(output, targets, cotangent, create_graph=True, allow_unused=True)
    active = [(g, h) for g, h in zip(first, directions) if g is not None and g.requires_grad]
    second = torch.autograd.grad(tuple(g for g, _ in active), targets,
                                 tuple(h for _, h in active), allow_unused=True)
    return first, second


def assert_autograd_chain(output, targets, expected, reference_targets):
    assert_reference_close(output, expected)
    cotangent = torch.randn_like(output)
    directions = tuple(torch.randn_like(value) for value in targets)
    actual = derivative_chain(output, targets, cotangent, directions)
    wanted = derivative_chain(expected, reference_targets, cotangent.double(),
                              tuple(value.double() for value in directions))
    for current, reference_order in zip(actual, wanted):
        for result, target in zip(current, reference_order):
            assert (result is None) == (target is None)
            if result is not None:
                assert_reference_close(result, target)


def directional_derivatives(output, values, cotangent, directions):
    first = torch.autograd.grad(
        output, values, cotangent, create_graph=True, allow_unused=True
    )
    contraction = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(first, directions, strict=True)
        if gradient is not None and gradient.requires_grad
    )
    second = torch.autograd.grad(contraction, (*values, cotangent), allow_unused=True)
    return (output, *first, *second)


def assert_directional_derivatives(actual, expected, inputs, reference_inputs):
    cotangent = torch.randn_like(actual, requires_grad=True)
    directions = tuple(torch.randn_like(value) for value in inputs)
    observed = directional_derivatives(actual, inputs, cotangent, directions)
    reference = directional_derivatives(
        expected, reference_inputs, cotangent.double(),
        tuple(value.double() for value in directions),
    )
    for value, wanted in zip(observed, reference, strict=True):
        if wanted is None:
            assert value is None or not torch.count_nonzero(value)
        elif value is None:
            assert not torch.count_nonzero(wanted)
        else:
            torch.testing.assert_close(value.double(), wanted, atol=4e-5, rtol=4e-4)
