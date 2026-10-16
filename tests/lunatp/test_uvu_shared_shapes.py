"""Large-batch shared UVU reductions with channel tails and edge copies."""

import pytest
import torch
from e3nn.o3 import TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProduct
from tests.lunatp.test_uvu_shapes import assert_close, derivative_chain, make_uvu_shape


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shared", [False, True])
def test_uvu_tail_vector_store_boundary(dtype, shared):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        tp = E3nnTensorProduct(
            "35x2e",
            "1x0e",
            "35x2e + 1x0e",
            [(0, 0, 0, "uvu", True)],
            internal_weights=shared,
            shared_weights=shared,
        ).cuda()
    finally:
        torch.set_default_dtype(old_dtype)
    op = TensorProduct(tp)
    x = torch.randn(7, 175, device="cuda", dtype=dtype)
    y = torch.randn(7, 1, device="cuda", dtype=dtype)
    w = tp.weight if shared else torch.randn(7, 35, device="cuda", dtype=dtype)
    expected = tp(x, y, w)
    with torch.no_grad():
        actual = op(x, y, None if shared else w)
        assert_close(actual, expected, dtype)
        assert not actual[:, -1].count_nonzero()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uvu_shared_tail_partial_reductions(dtype):
    torch.manual_seed(5837)
    tp = make_uvu_shape("tail", True, dtype, "cuda")
    op = TensorProduct(tp)
    items = 2051
    x = torch.randn(
        items, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        items, tp.irreps_in2.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    w = tp.weight
    g = torch.randn(
        items, tp.irreps_out.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    inputs = (x, y, w, g)
    directions = tuple((torch.randn_like(v, requires_grad=True) for v in inputs[:3]))
    second_directions = tuple((torch.randn_like(v) for v in inputs))
    expected = derivative_chain(tp(x, y, w), inputs, directions, second_directions)
    actual = derivative_chain(op(x, y), inputs, directions, second_directions)
    for results, references in zip(actual, expected):
        for result, reference in zip(results, references):
            assert_close(result, reference, dtype)
