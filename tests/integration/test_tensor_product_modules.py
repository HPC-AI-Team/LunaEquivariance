"""Public tensor-product modules retain weight ownership and CUDA derivatives."""

import pytest
import torch

from LunaEquivariance import TensorProduct, TensorProductConv


@pytest.mark.parametrize(
    "cls,mode",
    (
        (TensorProduct, "uuu"),
        (TensorProduct, "uvu"),
        (TensorProduct, "uvw"),
        (TensorProductConv, "uuu"),
        (TensorProductConv, "uvw"),
    ),
)
def test_irreps_constructor_registers_internal_weights(cls, mode):
    module = cls("2x0e", "2x0e", "2x0e", [(0, 0, 0, mode, True)], dtype=torch.float64)
    assert module.internal_weights and module.shared_weights
    assert module.weight.dtype == torch.float64
    assert module.weight is dict(module.named_parameters())["tp.weight"]
    before = module.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(module.weight, before - 0.1)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "mode", ("uuu", "uvu", "uvw", "uuu_conv", "uvu_conv", "uvw_conv")
)
def test_converted_module_forward_and_three_orders(make_problem, mode):
    tp, x, y, w, src, dst = make_problem(mode)
    conv = mode.endswith("_conv")
    module = (TensorProductConv if conv else TensorProduct)(tp)
    kwargs = {"graph": module.create_graph(src, dst, len(x))} if conv else {}
    module(
        x, y, w, **kwargs
    )  # Populate the FP32 launch cache before changing precision.
    parent = torch.nn.Sequential(module)
    for dtype in (torch.float64, torch.float32):
        parent.double() if dtype == torch.float64 else parent.float()
        reference, x, y, w, _, _ = make_problem(mode, dtype=dtype)
        assert not hasattr(module.op, "graph")
        expected = reference(x[src] if conv else x, y, w)
        if conv:
            expected = expected.new_zeros((len(x), reference.irreps_out.dim)).index_add(
                0, dst, expected
            )
        actual = module(x, y, w, **kwargs)
        tolerance = 5e-5 if dtype == torch.float32 else 3e-12
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        grad = torch.randn_like(expected, requires_grad=True)
        inputs = (x, y, w, grad)
        first_seeds = tuple(
            torch.randn_like(value, requires_grad=True) for value in inputs[:3]
        )
        second_seeds = tuple(torch.randn_like(value) for value in inputs)
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
            error = (
                expected_grad - actual_grad
            ).double().norm() / expected_grad.double().norm().clamp_min(1e-20)
            assert error.item() < (2e-5 if dtype == torch.float32 else 3e-12)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "mode", ("uuu", "uvu", "uvw", "uuu_conv", "uvu_conv", "uvw_conv")
)
def test_forward_and_three_orders(make_problem, mode, dtype):
    tp, x, y, w, src, dst = make_problem(mode, dtype=dtype)
    conv = mode.endswith("_conv")
    cls = TensorProductConv if conv else TensorProduct
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
    baseline = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(
        tp, dtype=dtype
    )
    kwargs = {}
    if conv:
        graph = module.create_graph(src, dst, len(x))
        kwargs["graph"] = graph
    actual = module(x, y, w, **kwargs)
    original = baseline(x, y, w, **kwargs)
    # UVW convolution uses atomic sums; repeated launches can round differently.
    if mode == "uvw_conv":
        tolerance = 5e-5 if dtype == torch.float32 else 3e-12
        torch.testing.assert_close(actual, original, rtol=tolerance, atol=tolerance)
    else:
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    expected = tp(x[src] if conv else x, y, w)
    if conv:
        expected = expected.new_zeros((len(x), tp.irreps_out.dim)).index_add(
            0, dst, expected
        )
    tolerance = 5e-5 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    grad = torch.randn_like(expected, requires_grad=True)
    inputs = (x, y, w, grad)
    first_seeds = tuple(
        torch.randn_like(value, requires_grad=True) for value in inputs[:3]
    )
    second_seeds = tuple(torch.randn_like(value) for value in inputs)
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
    for reference, result in zip(*results):
        error = (
            reference - result
        ).double().norm() / reference.double().norm().clamp_min(1e-20)
        assert error.item() < (2e-5 if dtype == torch.float32 else 3e-12)
