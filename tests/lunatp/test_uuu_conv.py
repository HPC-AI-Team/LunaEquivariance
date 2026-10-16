"""Current UUU convolution topology, precision, envelope and raw scalar contract."""

import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct, wigner_3j

from LunaEquivariance import TensorProductConv


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
    in1, in2 = tp.irreps_in1.slices(), tp.irreps_in2.slices()
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
        if value is not None and value.requires_grad and seed is not None
    ]
    return torch.autograd.grad(
        tuple(value for value, _ in selected),
        inputs,
        tuple(seed for _, seed in selected),
        allow_unused=True,
        create_graph=True,
        retain_graph=True,
    )


def derivative_chain(outputs, values, seeds, first_seeds, second_seeds):
    first = differentiate(outputs, values, seeds)
    second_inputs = (*values, *(seed for seed in seeds if seed is not None))
    second = differentiate(first, second_inputs, first_seeds)
    third = differentiate(second, (*second_inputs, *first_seeds), second_seeds)
    return first, second, third


def assert_orders_match(actual, expected, dtype):
    rtol, atol = (2e-11, 1e-11) if dtype == torch.float64 else (5e-4, 5e-5)
    for actual_order, expected_order in zip(actual, expected):
        for value, reference in zip(actual_order, expected_order):
            assert (value is None) == (reference is None)
            if value is not None:
                torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)


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
    # Sort paired topology and edge data while retaining the isolated node.
    src = torch.tensor([1, 0, 2, 0, 3, 1, 2, 0, 1, 3, 2], device="cuda")
    dst = torch.tensor([3, 0, 1, 2, 0, 1, 3, 1, 0, 2, 3], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = src[order], dst[order]
    y, weight, env = y[order], weight[order], env[order]
    graph = op.create_graph(src, dst, len(x))
    actual = op(x, y, weight, graph=graph, env=env, return_l0_feat=True)
    messages = tp(x[src], y, weight) * env
    expected = (
        messages.new_zeros((len(x), tp.irreps_out.dim)).index_add(0, dst, messages),
        scalar_features(tp, x[src], y, weight),
    )
    assert_orders_match((actual,), (expected,), dtype)
    values = (x, y, weight) + ((env,) if train_env and not scalar_only else ())
    seeds = (
        None if scalar_only else torch.randn_like(actual[0], requires_grad=True),
        torch.randn_like(actual[1], requires_grad=True),
    )
    first_seeds = tuple(torch.randn_like(value, requires_grad=True) for value in values)
    second_inputs = (*values, *(seed for seed in seeds if seed is not None))
    second_seeds = tuple(torch.randn_like(value) for value in second_inputs)
    assert_orders_match(
        derivative_chain(actual, values, seeds, first_seeds, second_seeds),
        derivative_chain(expected, values, seeds, first_seeds, second_seeds),
        dtype,
    )
    # Requesting only x in double backward leaves y and weight directions absent.
    first = torch.autograd.grad(
        actual[0], x, torch.ones_like(actual[0]), create_graph=True, retain_graph=True
    )[0]
    reference = torch.autograd.grad(
        expected[0],
        x,
        torch.ones_like(expected[0]),
        create_graph=True,
        retain_graph=True,
    )[0]
    seed = torch.randn_like(x)
    second = torch.autograd.grad(first, (y, weight), seed, retain_graph=True)
    expected_second = torch.autograd.grad(
        reference, (y, weight), seed, retain_graph=True
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
    messages = tp(x[src], y)
    expected = messages.new_zeros((4, tp.irreps_out.dim)).index_add(0, dst, messages)
    assert_orders_match(((actual,),), ((expected,),), dtype)
    values = (x, y) + ((tp.weight,) if weighted else ())
    actual_grad = torch.autograd.grad(actual.sum(), values)
    expected_grad = torch.autograd.grad(expected.sum(), values)
    assert_orders_match((actual_grad,), (expected_grad,), dtype)
