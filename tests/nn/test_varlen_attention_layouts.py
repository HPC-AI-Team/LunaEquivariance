"""2D and explicit-head inputs share the same CUDA computation."""

import itertools

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance.nn import VarlenAttention
from tests.nn.test_varlen_attention import gradients


@pytest.mark.cuda
def test_head_layouts_four_orders():
    torch.manual_seed(908)
    csr = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    base = tuple(
        (torch.randn(7, 2, width, device="cuda") * 0.2).requires_grad_()
        for width in (8, 8, 5)
    )
    flat = tuple(value.flatten(1).detach().requires_grad_() for value in base)
    flat_module = VarlenAttention(num_heads=2)
    expected = gradients(lambda *values: flat_module(*values, csr), flat)
    for split in itertools.product((False, True), repeat=3):
        values = tuple(
            (value if headed else value.flatten(1)).detach().requires_grad_()
            for value, headed in zip(base, split)
        )
        # Head count comes from a 3D operand; no extra user argument is needed.
        module = VarlenAttention() if any(split) else flat_module

        def fn(*args):
            return module(*args, csr)

        output = fn(*values)
        assert output.shape == values[2].shape
        actual = gradients(fn, values)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.reshape(-1), b.reshape(-1), atol=0, rtol=0)
        traced = make_fx(
            lambda *args: gradients(fn, args),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(*values)
        compiled = torch.compile(traced, fullgraph=True)
        with torch.no_grad():
            actual = compiled(*values)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(
                a.reshape(-1), b.reshape(-1), atol=2e-5, rtol=2e-5
            )
    module = VarlenAttention()
    torch._dynamo.reset()
    output = torch.compile(module, fullgraph=True)(*base, csr)
    torch.testing.assert_close(output.flatten(1), expected[0], atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
def test_head_layout_bias_and_stride():
    csr = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int64)
    q, k, v = (torch.randn(7, 2, 16, device="cuda")[:, :, ::2] for _ in range(3))
    bias = torch.randn(7, 2, device="cuda")
    actual = VarlenAttention()(q, k, v, csr, bias=bias)
    expected = VarlenAttention(num_heads=2)(
        q.flatten(1), k.flatten(1), v.flatten(1), csr, bias=bias
    )
    torch.testing.assert_close(actual.flatten(1), expected, rtol=0, atol=0)
