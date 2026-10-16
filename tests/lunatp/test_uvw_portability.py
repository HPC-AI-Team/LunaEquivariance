"""Device-specific native architecture dispatch and FP64 warp fragments."""

import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["uvw", "uvw_conv"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [3, 4, 12])
def test_external_path_groups_through_triple_backward(mode, dtype, channels):
    """Cover partial warps and complete warps within a partial thread block."""

    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(61200 + channels)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            f"{channels}x2e",
            "1x2e",
            "5x2e",
            [(0, 0, 0, "uvw", True)],
            internal_weights=False,
            shared_weights=False,
        ).to(device="cuda", dtype=dtype)
    finally:
        torch.set_default_dtype(previous_dtype)
    rows = 29
    nodes = 7 if mode == "uvw_conv" else rows
    x = torch.randn(
        nodes, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        rows, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = torch.randn(
        rows, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True
    )
    src = torch.randint(nodes, (rows,), device="cuda")
    dst = torch.sort(torch.randint(nodes, (rows,), device="cuda"))[0]
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(
        tp, dtype=dtype
    ).cuda()
    if mode == "uvw_conv":
        graph = op.create_graph(src, dst, nodes)
        actual = op(x, y, w, graph=graph)
        expected = x.new_zeros((nodes, tp.irreps_out.dim)).index_add(
            0, dst, tp(x[src], y, w)
        )
    else:
        actual, expected = op(x, y, w), tp(x, y, w)
    inputs = (x, y, w)
    grad_out = torch.randn_like(expected, requires_grad=True)
    first_seeds = tuple(torch.randn_like(value, requires_grad=True) for value in inputs)
    second_seeds = tuple(torch.randn_like(value) for value in (*inputs, grad_out))
    results = []
    for value in (actual, expected):
        first = torch.autograd.grad(value, inputs, grad_out, create_graph=True)
        second = torch.autograd.grad(
            first, (*inputs, grad_out), first_seeds, create_graph=True
        )
        third = torch.autograd.grad(
            second, (*inputs, grad_out, *first_seeds), second_seeds
        )
        results.append((value, *first, *second, *third))
    rtol, atol = (5e-4, 5e-5) if dtype == torch.float32 else (1e-11, 1e-11)
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
