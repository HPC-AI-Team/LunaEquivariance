"""Packed CSR attention through triple-backward, including mixed gradient seeds."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance.nn import VarlenAttention, VarlenAttentionConv


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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,dim,atol,rtol", [
    (torch.float32, 128, 2e-5, 2e-5),
    (torch.float16, 64, 2e-4, 1e-2),
    (torch.bfloat16, 64, 3e-3, 3e-2),
])
@pytest.mark.parametrize("lengths", [
    [0, 1, 64, 65, 128, 129, 256],
    [0, 1, 129, 385, 769],
], ids=["tile_boundaries", "long_sequences"])
def test_attention_resident_tile(dtype, dim, atol, rtol, lengths):
    torch.manual_seed(819)
    offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    inputs = tuple(
        (torch.randn(offsets[-1], 2, dim, device="cuda", dtype=dtype) * 0.2)
        .requires_grad_()
        for _ in range(3)
    )
    refs = tuple(x.detach().double().requires_grad_() for x in inputs)
    outputs = []
    for begin, end in zip(offsets, offsets[1:]):
        q, k, v = (x[begin:end].transpose(0, 1) for x in refs)
        scores = q @ k.transpose(-1, -2) * dim**-0.5
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(outputs)
    actual = VarlenAttention(num_heads=2)(*inputs, csr)
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)
    seed = torch.randn_like(actual) * 0.1
    got = torch.autograd.grad(actual, inputs, seed)
    want = torch.autograd.grad(expected, refs, seed.double())
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,dim,value_dim,length", [
    (torch.float32, 8, 8, 256),
    (torch.float32, 128, 128, 256),
    (torch.float16, 128, 128, 256),
    (torch.bfloat16, 128, 128, 256),
    (torch.float16, 8, 8, 256),
    (torch.float16, 17, 7, 256),
    (torch.bfloat16, 8, 5, 256),
    (torch.float16, 128, 128, 128),
    (torch.bfloat16, 128, 128, 128),
])
def test_streaming_gradients(dtype, dim, value_dim, length):
    torch.manual_seed(481)
    csr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    inputs = tuple(
        (torch.randn(length, 1, width, device="cuda", dtype=dtype) * 0.2)
        .requires_grad_() for width in (dim, dim, value_dim)
    )
    refs = tuple(x.detach().double().requires_grad_() for x in inputs)
    seeds = tuple(torch.randn_like(x) * 0.1 for x in inputs)
    module = VarlenAttention()

    def derivatives(values, output):
        result = [output]
        loss = output.square().sum()
        for order in range(3):
            grads = torch.autograd.grad(loss, values, create_graph=order < 2)
            result.extend(grads)
            loss = sum((g * seed.to(g.dtype)).sum() for g, seed in zip(grads, seeds))
        return tuple(result)

    def fn(*values):
        return derivatives(values, module(*values, csr))

    q, k, v = (x.transpose(0, 1) for x in refs)
    expected = derivatives(refs, ((q @ k.transpose(-1, -2) * dim**-0.5)
                                  .softmax(-1) @ v).transpose(0, 1))
    actual = fn(*inputs)
    atol, rtol = ((2e-5, 2e-5) if dtype == torch.float32 else
                  (2e-3, 1e-2) if dtype == torch.float16 else (2e-2, 3e-2))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)
    traced = make_fx(fn, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
    with torch.no_grad():
        compiled = torch.compile(traced, fullgraph=True)(*inputs)
    for got, want in zip(compiled, expected):
        torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)

    output = torch.compile(module, fullgraph=True)(*inputs, csr)
    grads = torch.autograd.grad(output.square().sum(), inputs)
    for got, want in zip(grads, expected[1:4]):
        torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,needs,key_bias", [
    (torch.float32, (True, True, True), False),
    (torch.float16, (True, True, True), False),
    (torch.bfloat16, (True, True, True), False),
    (torch.float16, (True, False, False), False),
    (torch.float16, (False, True, False), True),
    (torch.bfloat16, (False, False, True), False),
])
def test_attention_storage_offset(dtype, needs, key_bias):
    torch.manual_seed(819)
    rows, heads, dim = 514, 2, 64
    inputs = tuple(
        (torch.randn(rows * heads * dim + 1, device="cuda", dtype=dtype) * 0.3)[1:]
        .view(rows, heads, dim).requires_grad_(need)
        for need in needs
    )
    reference = tuple(x.detach().double().requires_grad_(need) for x, need in zip(inputs, needs))
    csr = torch.tensor([0, 257, 514], device="cuda", dtype=torch.int32)
    bias = torch.randn(rows, heads, device="cuda", dtype=dtype) * 0.1 if key_bias else None
    actual = VarlenAttention(num_heads=heads)(*inputs, csr, bias=bias)
    expected = []
    for start, end in ((0, 257), (257, 514)):
        q, k, v = (x[start:end].transpose(0, 1) for x in reference)
        scores = q @ k.transpose(-1, -2) * dim**-0.5
        if bias is not None:
            scores = scores + bias[start:end].double().T[:, None, :]
        expected.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(expected)
    seed = torch.randn_like(actual) * 0.1
    actual_grad = torch.autograd.grad(actual, tuple(x for x in inputs if x.requires_grad), seed)
    expected_grad = torch.autograd.grad(expected, tuple(x for x in reference if x.requires_grad), seed.double())
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (2e-4, 1e-2) if dtype == torch.float16 else (3e-3, 3e-2)
    for result, ref in zip((actual, *actual_grad), (expected, *expected_grad)):
        torch.testing.assert_close(result.double(), ref, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_ragged_strided(dtype, causal):
    torch.manual_seed(927)
    offsets, heads, dim = [0, 0, 1, 128, 512], 2, 128
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int64).repeat_interleave(2)[::2]
    inputs = tuple(
        (torch.randn(offsets[-1], heads, 2 * dim + 1, device="cuda", dtype=dtype) * 0.3)
        [..., 1::2].requires_grad_() for _ in range(3)
    )
    refs = tuple(x.detach().double().requires_grad_() for x in inputs)
    actual = VarlenAttention(num_heads=heads, causal=causal)(*inputs, csr)
    chunks = []
    for begin, end in zip(offsets, offsets[1:]):
        q, k, v = (x[begin:end].transpose(0, 1) for x in refs)
        scores = q @ k.transpose(-1, -2) * dim**-0.5
        if causal:
            mask = torch.ones(end - begin, end - begin, device="cuda", dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        chunks.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(chunks)
    seed = torch.randn_like(actual) * 0.1
    actual_grad = torch.autograd.grad(actual, inputs, seed)
    expected_grad = torch.autograd.grad(expected, refs, seed.double())
    atol, rtol = ((2e-5, 2e-5) if dtype == torch.float32 else
                  (2e-4, 1e-2) if dtype == torch.float16 else (3e-3, 3e-2))
    for got, want in zip((actual, *actual_grad), (expected, *expected_grad)):
        torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("conv,envelope", [(False, False), (True, False), (True, True)])
def test_attention_streams_and_aliases(dtype, conv, envelope):
    torch.manual_seed(673)
    offsets = [0, 0, 1, 256, 769]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        csr_storage = torch.empty(10, device="cuda", dtype=torch.int64)
        csr = csr_storage[::2]
        csr.copy_(torch.tensor(offsets, device="cuda"))
        module = (VarlenAttentionConv if conv else VarlenAttention)(num_heads=2)
        for alias in (False, True, False):
            inputs = [
                (torch.randn(769, 2, 65, device="cuda", dtype=dtype) * 0.2)[
                    :, :, 1::2
                ].requires_grad_()
                for _ in range(1 if alias else 3)
            ]
            env = (
                torch.rand(769, device="cuda", dtype=dtype, requires_grad=True)
                if envelope
                else None
            )
            values = inputs * 3 if alias else inputs
            output = module(*values, csr, envelope=env) if conv else module(*values, csr)
            leaves = tuple(inputs) + ((env,) if envelope else ())
            refs = tuple(x.detach().double().requires_grad_() for x in leaves)
            q, k, v = refs[:1] * 3 if alias else refs[:3]
            segments = []
            for begin, end in zip(offsets, offsets[1:]):
                y = torch.nn.functional.scaled_dot_product_attention(
                    q[begin:end].transpose(0, 1),
                    k[begin:end].transpose(0, 1),
                    v[begin:end].transpose(0, 1),
                ).transpose(0, 1)
                if conv:
                    if envelope:
                        y = y * refs[-1][begin:end, None, None]
                    y = y.sum(0)
                segments.append(y)
            expected = torch.stack(segments) if conv else torch.cat(segments)
            seed = torch.randn_like(output) * 0.1
            actual_grads = torch.autograd.grad(output, leaves, seed)
            ref_grads = torch.autograd.grad(expected, refs, seed.double())
            atol, rtol = (
                (2e-5, 2e-5)
                if dtype == torch.float32
                else (1e-2, 1e-2)
                if dtype == torch.float16
                else (0.08, 0.05)
            )
            for got, want in zip((output, *actual_grads), (expected, *ref_grads)):
                torch.testing.assert_close(got.double(), want, atol=atol, rtol=rtol)
    stream.synchronize()
