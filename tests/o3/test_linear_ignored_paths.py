"""Accepted ignored e3nn paths retain zero weight slots through mixed derivatives."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance.modules.linear import LunaEqLinear


CASES = (
    pytest.param(1, 3, True, False, id="shared-dot-direct"),
    pytest.param(1, 65, True, False, id="shared-dot-split"),
    pytest.param(1, 3, False, False, id="external-dot-dx-dw-dg-db"),
    pytest.param(1, 65, False, False, id="external-dot-medium-dx-dw-dg-db"),
    pytest.param(1, 4096, False, False, id="external-dot-path-major"),
    pytest.param(32, 3, True, False, id="shared-gemm-direct"),
    pytest.param(32, 65, True, False, id="shared-gemm-split"),
    pytest.param(32, 3, False, False, id="external-power-of-two"),
    pytest.param(33, 3, False, False, id="external-padded"),
    pytest.param(1, 3, True, True, id="shared-feature-axes"),
    pytest.param(1, 3, False, True, id="external-feature-axes"),
)


def mixed_derivatives(module, x, weight, g, rx, rw, bx, bw, bg):
    y = module(x, weight)
    first = torch.autograd.grad(y, (x, weight), g, create_graph=True)
    second = torch.autograd.grad(first, (x, weight, g), (rx, rw), create_graph=True)
    third = torch.autograd.grad(second, (x, weight, g, rx, rw), (bx, bw, bg))
    return y, *first, *second, *third


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("width,items,shared,feature_axes", CASES)
def test_ignored_weight_slots_match_e3nn_in_all_orders(
    dtype, width, items, shared, feature_axes
):
    torch.manual_seed(47212)
    kwargs = dict(
        internal_weights=False,
        shared_weights=shared,
        instructions=[(0, 0), (0, -1)],
        **(dict(f_in=2, f_out=3) if feature_axes else {}),
    )
    reference = o3.Linear(f"{width}x0e", f"{width}x0e", **kwargs).cuda().to(dtype)
    native = LunaEqLinear(reference)
    xshape = (items,) + ((2,) if feature_axes else ()) + (width,)
    yshape = (items,) + ((3,) if feature_axes else ()) + (width,)
    wshape = (
        (() if shared else (items,))
        + ((2, 3) if feature_axes else ())
        + (2 * width * width,)
    )
    shapes = (xshape, wshape, yshape, xshape, wshape, xshape, wshape, yshape)
    operands = tuple(
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in shapes
    )
    expected = mixed_derivatives(reference, *operands)
    actual = mixed_derivatives(native, *operands)
    assert native.last_execution_backend == "cuda_fused"
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=tolerance, atol=tolerance)
    gaps = slice(width * width, 2 * width * width)
    # D1 DW, D2 SW, D3 TW/TRW reuse ordinary and dx/dw/dg/db weight-gradient emitters.
    for index in (2, 4, 7, 10):
        assert not actual[index][..., gaps].count_nonzero()
