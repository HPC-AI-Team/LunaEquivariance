"""Public inference and training outputs against the e3nn expression."""

import pytest
import torch
from LunaEquivariance import TensorProduct, TensorProductConv


@pytest.mark.cuda
@pytest.mark.parametrize(
    "mode,shared",
    [
        ("uuu", False),
        ("uuu", True),
        ("uuu_conv", False),
        ("uvu", False),
        ("uvu", True),
        ("uvw", False),
        ("uvw", True),
        ("uvu_conv", False),
        ("uvw_conv", False),
        ("uvw_conv", True),
    ],
)
def test_forward(make_problem, mode, shared):
    tp, x, y, w, src, dst = make_problem(mode, shared)
    op = (TensorProductConv if mode.endswith("_conv") else TensorProduct)(tp)
    kwargs = {}
    reference = tp(x[src] if mode.endswith("conv") else x, y, w)
    if mode.endswith("conv"):
        kwargs["graph"] = op.create_graph(src, dst, len(x))
        reference = reference.new_zeros((len(x), tp.irreps_out.dim)).index_add(
            0, dst, reference
        )
    for inference in (False, True):
        with torch.set_grad_enabled(not inference):
            actual = op(x, y, None if shared else w, **kwargs)
        torch.testing.assert_close(actual, reference, rtol=0.0005, atol=5e-05)
