"""Module conversion must update generated-code precision and owned tensors."""

import pytest
import torch
from e3nn import o3

from LunaEquivariance.modules.linear import LunaEqLinear


@pytest.mark.parametrize("method", ["half", "bfloat16"])
def test_low_precision_conversion_updates_owned_tensors_and_uses_compatibility(method):
    op = LunaEqLinear("2x0e", "2x0e", biases=True, dtype=torch.float32)
    ref = o3.Linear("2x0e", "2x0e", biases=True)
    ref.load_state_dict(op.state_dict(), strict=True)
    weight = op.weight.detach().clone()
    assert getattr(op, method)() is op
    getattr(ref, method)()
    dtype = torch.float16 if method == "half" else torch.bfloat16
    assert op.weight.dtype == op.bias.dtype == op.output_mask.dtype == dtype
    assert op.weight is op.tp.weight and op.bias is op.tp.bias
    torch.testing.assert_close(op.weight, weight.to(dtype), rtol=0, atol=0)
    features = torch.randn(3, 2, dtype=dtype)
    torch.testing.assert_close(op(features), ref(features), rtol=0, atol=0)
    assert op.last_execution_backend == "torch_compat"


@pytest.mark.parametrize("shared", (True, False))
@pytest.mark.parametrize("feature_axes", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
def test_wrap_scripted_module_infers_external_feature_axes(
    shared, feature_axes, explicit
):
    axes = dict(f_in=2, f_out=3) if feature_axes else {}
    reference = o3.Linear(
        "2x0e+1x1o", "2x0e+1x1o", internal_weights=False, shared_weights=shared, **axes
    )
    reference._compiled_main = torch.jit.script(reference._compiled_main)
    native = LunaEqLinear(reference, **(axes if explicit else {}))
    assert (native.f_in, native.f_out) == ((2, 3) if feature_axes else (None, None))
    x = torch.randn((4,) + ((2,) if feature_axes else ()) + (5,))
    weight = torch.randn(
        (() if shared else (4,))
        + ((2, 3) if feature_axes else ())
        + (reference.weight_numel,)
    )
    torch.testing.assert_close(native(x, weight), reference(x, weight), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("shared", (True, False))
def test_wrap_scripted_external_feature_axes_selects_native(shared):
    reference = o3.Linear(
        "2x0e+1x1o",
        "2x0e+1x1o",
        internal_weights=False,
        shared_weights=shared,
        f_in=2,
        f_out=3,
    ).cuda()
    reference._compiled_main = torch.jit.script(reference._compiled_main)
    native = LunaEqLinear(reference)
    x = torch.randn(7, 2, 5, device="cuda", requires_grad=True)
    weight = torch.randn(
        (() if shared else (7,)) + (2, 3, reference.weight_numel),
        device="cuda",
        requires_grad=True,
    )
    expected, actual = reference(x, weight), native(x, weight)
    assert native.last_execution_backend == "cuda_fused"
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    g = torch.randn_like(actual)
    for calculated, target in zip(
        torch.autograd.grad(actual, (x, weight), g),
        torch.autograd.grad(expected, (x, weight), g),
    ):
        torch.testing.assert_close(calculated, target, rtol=3e-5, atol=3e-6)
