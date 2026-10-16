"""Packed CSR attention through triple-backward, including mixed gradient seeds."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance.nn import VarlenAttention


def gradients(fn, inputs):
    values = (fn(*inputs),)
    result = list(values)
    targets = inputs
    for order in range(3):
        seeds = tuple(
            torch.full_like(value, 0.7, requires_grad=True) for value in values
        )
        grads = torch.autograd.grad(
            values, targets, seeds, create_graph=order < 2, allow_unused=True
        )
        result.extend(value for value in grads if value is not None)
        values = tuple(
            value for value in grads if value is not None and value.requires_grad
        )
        targets = (*targets, *seeds)
    return tuple(result)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets,heads,dim,value_dim,bias_kind,causal,strided,index_dtype",
    [
        ([0, 3, 3, 7], 2, 8, 5, 0, False, False, torch.int32),
        ([0, 5, 12], 2, 8, 8, 1, True, True, torch.int64),
        ([0, 3, 7], 1, 8, 8, 2, False, False, torch.int32),
        ([0, 33, 70], 1, 8, 8, 2, True, False, torch.int64),
    ],
)
def test_packed_csr_four_orders(
    offsets, heads, dim, value_dim, bias_kind, causal, strided, index_dtype
):
    torch.manual_seed(918)
    rows = offsets[-1]
    max_length = max(b - a for a, b in zip(offsets, offsets[1:]))
    csr = torch.tensor(offsets, device="cuda", dtype=index_dtype)
    inputs = []
    for width in (heads * dim, heads * dim, heads * value_dim):
        value = torch.randn(rows, width * (2 if strided else 1), device="cuda") * 0.2
        inputs.append((value[:, ::2] if strided else value).requires_grad_())
    if bias_kind:
        shape = (
            (rows, heads)
            if bias_kind == 1
            else (len(offsets) - 1, heads, max_length, max_length)
        )
        inputs.append((torch.randn(shape, device="cuda") * 0.1).requires_grad_())
    inputs = tuple(inputs)
    module = VarlenAttention(num_heads=heads, causal=causal)

    def fn(*args):
        return module(
            *args[:3], csr, bias=args[3] if bias_kind else None
        )

    def reference(*args):
        output = []
        for segment, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            q, k, v = (
                value[begin:end]
                .reshape(end - begin, heads, value.shape[1] // heads)
                .transpose(0, 1)
                for value in args[:3]
            )
            scores = q @ k.transpose(-1, -2) * dim**-0.5
            if bias_kind == 1:
                scores = scores + args[3][begin:end].transpose(0, 1)[:, None, :]
            elif bias_kind == 2:
                scores = scores + args[3][segment, :, : end - begin, : end - begin]
            if causal:
                mask = torch.ones(
                    end - begin, end - begin, device="cuda", dtype=torch.bool
                ).triu(1)
                scores = scores.masked_fill(mask, -torch.inf)
            output.append(
                (scores.softmax(-1) @ v)
                .transpose(0, 1)
                .reshape(end - begin, heads * value_dim)
            )
        return torch.cat(output)

    expected = gradients(reference, inputs)
    actual = gradients(fn, inputs)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    traced = make_fx(
        lambda *args: gradients(fn, args),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*inputs)
    compiled = torch.compile(traced, fullgraph=True)
    with torch.no_grad():
        actual = compiled(*inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
def test_concentrated_softmax_four_orders():
    # Large logits expose inconsistent forward and recomputed normalization.
    torch.manual_seed(472)
    offsets = [0, 0, 1, 18, 51, 116]
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    inputs = tuple(
        (torch.randn(116, 2, width, device="cuda") * 3).requires_grad_()
        for width in (96, 96, 65)
    )
    module = VarlenAttention(num_heads=2)

    def reference():
        outputs = []
        for begin, end in zip(offsets, offsets[1:]):
            q, k, v = (
                value[begin:end].transpose(0, 1).double() for value in inputs
            )
            scores = (q @ k.transpose(-1, -2)) * 96**-0.5
            outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        return torch.cat(outputs).float()

    expected = (reference(),)
    actual = (module(*inputs, csr),)
    targets = inputs
    generator = torch.Generator(device="cuda").manual_seed(731)
    for order in range(4):
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
        if order == 3:
            break
        active = tuple(i for i, value in enumerate(expected) if value.requires_grad)
        seeds = tuple(
            (torch.randn(expected[i].shape, device="cuda", generator=generator) * .1)
            .requires_grad_()
            for i in active
        )
        expected, actual = (
            torch.autograd.grad(
                tuple(values[i] for i in active), targets, seeds,
                create_graph=order < 2, allow_unused=True,
            )
            for values in (expected, actual)
        )
        targets = (*targets, *seeds)
