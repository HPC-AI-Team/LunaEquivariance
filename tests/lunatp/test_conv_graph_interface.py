"""All convolution modes consume the same explicit graph and edge order."""

import pytest
import torch
from e3nn import o3
from LunaEquivariance import (
    TensorProductConv,
    make_graph,
    prepare_tensor_product,
    tensor_product_conv,
)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_same_graph_across_modes_and_three_orders(dtype):
    torch.manual_seed(1042)
    src = torch.tensor([0, 2, 1, 2, 0, 1, 1], device="cuda")
    dst = torch.tensor([2, 0, 2, 1, 0, 1, 2], device="cuda")
    order = dst.argsort(stable=True)
    src, dst = (src[order], dst[order])
    graph = make_graph(src, dst, 4)
    for mode in ("uuu", "uvu", "uvw"):
        irreps = "32x0e + 32x1o"
        edge_irreps = irreps if mode == "uuu" else "1x0e + 1x1o"
        ir, edge = (o3.Irreps(irreps), o3.Irreps(edge_irreps))
        instructions = [
            (i, j, k, mode, True)
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
        y = torch.randn(
            len(src), edge.dim, device="cuda", dtype=dtype, requires_grad=True
        )
        weight = torch.randn(
            len(src),
            reference.weight_numel,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        actual = module(x, y, weight, graph=graph)
        config = prepare_tensor_product(
            ir, edge, ir, instructions, dtype=dtype, convolution=True
        )
        functional = tensor_product_conv(x, y, weight, config=config, graph=graph)
        messages = reference(x[src], y, weight)
        expected = x.new_zeros(4, ir.dim).index_add(0, dst, messages)
        tolerance = 5e-05 if dtype == torch.float32 else 3e-12
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(functional, expected, rtol=tolerance, atol=tolerance)
        grad = torch.randn_like(expected, requires_grad=True)
        inputs = (x, y, weight, grad)
        first_seeds = tuple(
            (torch.randn_like(value, requires_grad=True) for value in inputs[:3])
        )
        second_seeds = tuple((torch.randn_like(value) for value in inputs))
        results = []
        for output in (expected, actual):
            first = torch.autograd.grad(
                output, inputs[:3], grad, create_graph=True, retain_graph=True
            )
            second = torch.autograd.grad(
                first, inputs, first_seeds, create_graph=True, retain_graph=True
            )
            third = torch.autograd.grad(
                second, (*inputs, *first_seeds), second_seeds, retain_graph=True
            )
            results.append((*first, *second, *third))
        for expected_grad, actual_grad in zip(*results):
            relative_error = (
                expected_grad - actual_grad
            ).double().norm() / expected_grad.double().norm().clamp_min(1e-20)
            assert relative_error.item() < (2e-05 if dtype == torch.float32 else 3e-12)
