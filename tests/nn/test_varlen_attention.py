"""Variable-length attention layouts, compilation and derivatives."""

import itertools

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from LunaEquivariance import VarlenAttention, VarlenAttentionConv
from LunaEquivariance.utils.testing import check_compatibility
import LunaEquivariance as lunaeq
from tests.testing_utils import (
    assert_aoti,
    assert_compiled,
    assert_reference_close,
    fp64,
)


def gradients(fn, inputs):
    values = (fn(*inputs),)
    result = list(values)
    targets = inputs
    for order in range(3):
        seeds = tuple(
            (
                torch.full_like(value, 0.7, dtype=torch.float32)
                .to(value.dtype)
                .requires_grad_()
                for value in values
            )
        )
        grads = torch.autograd.grad(
            values, targets, seeds, create_graph=order < 2, allow_unused=True
        )
        result.extend((value for value in grads if value is not None))
        values = tuple(
            (value for value in grads if value is not None and value.requires_grad)
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
    max_length = max((b - a for a, b in zip(offsets, offsets[1:])))
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
        return module(*args[:3], csr, bias=args[3] if bias_kind else None)

    def reference(*args):
        output = []
        for segment, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            q, k, v = (
                value[begin:end]
                .reshape(end - begin, heads, value.shape[1] // heads)
                .transpose(0, 1)
                for value in args[:3]
            )
            scores = q @ k.transpose(-1, -2) * dim ** (-0.5)
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

    expected = gradients(reference, fp64(inputs))
    actual = gradients(fn, inputs)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert_reference_close(a, b)
    traced = make_fx(
        lambda *args: gradients(fn, args),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*inputs)
    compiled = torch.compile(traced, fullgraph=True)
    with torch.no_grad():
        actual = compiled(*inputs)
    for a, b in zip(actual, expected):
        assert_reference_close(a, b)


@pytest.mark.cuda
def test_concentrated_softmax_four_orders():
    torch.manual_seed(472)
    offsets = [0, 0, 1, 18, 51, 116]
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    inputs = tuple(
        (
            (torch.randn(116, 2, width, device="cuda") * 3).requires_grad_()
            for width in (96, 96, 65)
        )
    )
    module = VarlenAttention(num_heads=2)

    ref_targets = fp64(inputs)

    def reference():
        outputs = []
        for begin, end in zip(offsets, offsets[1:]):
            q, k, v = (
                value[begin:end].transpose(0, 1).double() for value in ref_targets
            )
            scores = q @ k.transpose(-1, -2) * 96 ** (-0.5)
            outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        return torch.cat(outputs)

    expected = (reference(),)
    actual = (module(*inputs, csr),)
    targets = inputs
    generator = torch.Generator(device="cuda").manual_seed(731)
    for order in range(4):
        for got, want in zip(actual, expected):
            assert_reference_close(got, want)
        if order == 3:
            break
        active = tuple((i for i, value in enumerate(expected) if value.requires_grad))
        seeds = tuple(
            (
                (
                    torch.randn(expected[i].shape, device="cuda", generator=generator)
                    * 0.1
                ).requires_grad_()
                for i in active
            )
        )
        ref_seeds = fp64(seeds)
        expected = torch.autograd.grad(
            tuple(expected[i] for i in active),
            ref_targets,
            ref_seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        actual = torch.autograd.grad(
            tuple(actual[i] for i in active),
            targets,
            seeds,
            create_graph=order < 2,
            allow_unused=True,
        )
        targets = (*targets, *seeds)
        ref_targets = (*ref_targets, *ref_seeds)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,dim", [(torch.float32, 128), (torch.float16, 64), (torch.bfloat16, 64),],
)
@pytest.mark.parametrize(
    "lengths",
    [[0, 1, 64, 65, 128, 129, 256], [0, 1, 129, 385, 769]],
    ids=["tile_boundaries", "long_sequences"],
)
def test_attention_resident_tile(dtype, dim, lengths):
    torch.manual_seed(819)
    offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    inputs = tuple(
        (
            (
                torch.randn(offsets[-1], 2, dim, device="cuda", dtype=dtype) * 0.2
            ).requires_grad_()
            for _ in range(3)
        )
    )
    refs = tuple((x.detach().double().requires_grad_() for x in inputs))
    outputs = []
    for begin, end in zip(offsets, offsets[1:]):
        q, k, v = (x[begin:end].transpose(0, 1) for x in refs)
        scores = q @ k.transpose(-1, -2) * dim ** (-0.5)
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(outputs)
    actual = VarlenAttention(num_heads=2)(*inputs, csr)
    assert_reference_close(actual, expected)
    seed = torch.randn_like(actual) * 0.1
    got = torch.autograd.grad(actual, inputs, seed)
    want = torch.autograd.grad(expected, refs, seed.double())
    for a, b in zip(got, want):
        assert_reference_close(a, b)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,dim,value_dim,length",
    [
        (torch.float32, 8, 8, 256),
        (torch.float32, 128, 128, 256),
        (torch.float16, 128, 128, 256),
        (torch.bfloat16, 128, 128, 256),
        (torch.float16, 8, 8, 256),
        (torch.float16, 17, 7, 256),
        (torch.bfloat16, 8, 5, 256),
        (torch.float16, 128, 128, 128),
        (torch.bfloat16, 128, 128, 128),
    ],
)
def test_streaming_gradients(dtype, dim, value_dim, length):
    torch.manual_seed(481)
    csr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    inputs = tuple(
        (
            (
                torch.randn(length, 1, width, device="cuda", dtype=dtype) * 0.2
            ).requires_grad_()
            for width in (dim, dim, value_dim)
        )
    )
    refs = tuple((x.detach().double().requires_grad_() for x in inputs))
    seeds = tuple((torch.randn_like(x) * 0.1 for x in inputs))
    module = VarlenAttention()

    def derivatives(values, output):
        result = [output]
        loss = output.square().sum()
        for order in range(3):
            grads = torch.autograd.grad(loss, values, create_graph=order < 2)
            result.extend(grads)
            loss = sum(((g * seed.to(g.dtype)).sum() for g, seed in zip(grads, seeds)))
        return tuple(result)

    def fn(*values):
        return derivatives(values, module(*values, csr))

    q, k, v = (x.transpose(0, 1) for x in refs)
    expected = derivatives(
        refs,
        ((q @ k.transpose(-1, -2) * dim ** (-0.5)).softmax(-1) @ v).transpose(0, 1),
    )
    actual = fn(*inputs)
    for got, want in zip(actual, expected):
        assert_reference_close(got, want)
    traced = make_fx(fn, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
    with torch.no_grad():
        compiled = torch.compile(traced, fullgraph=True)(*inputs)
    for got, want in zip(compiled, expected):
        assert_reference_close(got, want)
    output = torch.compile(module, fullgraph=True)(*inputs, csr)
    grads = torch.autograd.grad(output.square().sum(), inputs)
    for got, want in zip(grads, expected[1:4]):
        assert_reference_close(got, want)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,needs,key_bias",
    [
        (torch.float32, (True, True, True), False),
        (torch.float16, (True, True, True), False),
        (torch.bfloat16, (True, True, True), False),
        (torch.float16, (True, False, False), False),
        (torch.float16, (False, True, False), True),
        (torch.bfloat16, (False, False, True), False),
    ],
)
def test_attention_storage_offset(dtype, needs, key_bias):
    torch.manual_seed(819)
    rows, heads, dim = (514, 2, 64)
    inputs = tuple(
        (
            (torch.randn(rows * heads * dim + 1, device="cuda", dtype=dtype) * 0.3)[1:]
            .view(rows, heads, dim)
            .requires_grad_(need)
            for need in needs
        )
    )
    reference = tuple(
        (x.detach().double().requires_grad_(need) for x, need in zip(inputs, needs))
    )
    csr = torch.tensor([0, 257, 514], device="cuda", dtype=torch.int32)
    bias = (
        torch.randn(rows, heads, device="cuda", dtype=dtype) * 0.1 if key_bias else None
    )
    actual = VarlenAttention(num_heads=heads)(*inputs, csr, bias=bias)
    expected = []
    for start, end in ((0, 257), (257, 514)):
        q, k, v = (x[start:end].transpose(0, 1) for x in reference)
        scores = q @ k.transpose(-1, -2) * dim ** (-0.5)
        if bias is not None:
            scores = scores + bias[start:end].double().T[:, None, :]
        expected.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(expected)
    seed = torch.randn_like(actual) * 0.1
    actual_grad = torch.autograd.grad(
        actual, tuple((x for x in inputs if x.requires_grad)), seed
    )
    expected_grad = torch.autograd.grad(
        expected, tuple((x for x in reference if x.requires_grad)), seed.double()
    )
    for result, ref in zip((actual, *actual_grad), (expected, *expected_grad)):
        assert_reference_close(result, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_ragged_strided(dtype, causal):
    torch.manual_seed(927)
    offsets, heads, dim = ([0, 0, 1, 128, 512], 2, 128)
    csr = torch.tensor(offsets, device="cuda", dtype=torch.int64).repeat_interleave(2)[
        ::2
    ]
    inputs = tuple(
        (
            (
                torch.randn(offsets[-1], heads, 2 * dim + 1, device="cuda", dtype=dtype)
                * 0.3
            )[..., 1::2].requires_grad_()
            for _ in range(3)
        )
    )
    refs = tuple((x.detach().double().requires_grad_() for x in inputs))
    actual = VarlenAttention(num_heads=heads, causal=causal)(*inputs, csr)
    chunks = []
    for begin, end in zip(offsets, offsets[1:]):
        q, k, v = (x[begin:end].transpose(0, 1) for x in refs)
        scores = q @ k.transpose(-1, -2) * dim ** (-0.5)
        if causal:
            mask = torch.ones(
                end - begin, end - begin, device="cuda", dtype=torch.bool
            ).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        chunks.append((scores.softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(chunks)
    seed = torch.randn_like(actual) * 0.1
    actual_grad = torch.autograd.grad(actual, inputs, seed)
    expected_grad = torch.autograd.grad(expected, refs, seed.double())
    for got, want in zip((actual, *actual_grad), (expected, *expected_grad)):
        assert_reference_close(got, want)


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
            output = (
                module(*values, csr, envelope=env) if conv else module(*values, csr)
            )
            leaves = tuple(inputs) + ((env,) if envelope else ())
            refs = tuple((x.detach().double().requires_grad_() for x in leaves))
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
            for got, want in zip((output, *actual_grads), (expected, *ref_grads)):
                assert_reference_close(got, want)
    stream.synchronize()


@pytest.mark.cuda
def test_head_layouts_four_orders():
    torch.manual_seed(908)
    csr = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    base = tuple(
        (
            (torch.randn(7, 2, width, device="cuda") * 0.2).requires_grad_()
            for width in (8, 8, 5)
        )
    )
    flat = tuple((value.flatten(1).detach().requires_grad_() for value in base))
    flat_module = VarlenAttention(num_heads=2)
    expected = gradients(lambda *values: flat_module(*values, csr), flat)
    for split in itertools.product((False, True), repeat=3):
        values = tuple(
            (
                (value if headed else value.flatten(1)).detach().requires_grad_()
                for value, headed in zip(base, split)
            )
        )
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
                a.reshape(-1), b.reshape(-1), atol=2e-05, rtol=2e-05
            )
    module = VarlenAttention()
    torch._dynamo.reset()
    output = torch.compile(module, fullgraph=True)(*base, csr)
    torch.testing.assert_close(output.flatten(1), expected[0], atol=2e-05, rtol=2e-05)


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


def assert_cuda_report(module, args, **kwargs):
    tolerance = 1e-09 if args[0].dtype == torch.float64 else 0.0003
    report = check_compatibility(
        module,
        args,
        kwargs=kwargs,
        derivative_order=3,
        loss_fn=lambda output: sum(
            (
                x.sin().mean()
                for x in (output if isinstance(output, tuple) else (output,))
            )
        ),
        warmup=1,
        repeats=2,
        atol=tolerance,
        rtol=tolerance,
        verbose=False,
    )
    assert report["passed"], report
    assert report["operator"] == type(module).__name__
    assert report["errors"]["gradient_3"]["passed"]
    assert all((item["speedup"] > 0 for item in report["timings"].values()))


@pytest.mark.cuda
@pytest.mark.parametrize("env_shape", [None])
@pytest.mark.parametrize(
    "bias_kind,causal,flatten", [(0, False, False), (1, False, True), (2, True, False)]
)
def test_cuda_attention(env_shape, bias_kind, causal, flatten):
    from LunaEquivariance import VarlenAttention, VarlenAttentionConv

    torch.manual_seed(42)
    cls = VarlenAttention
    module = cls(num_heads=2, causal=causal, softmax_scale=0.4)
    args = tuple(
        (
            (torch.randn(7, 2, d, device="cuda") * 0.2).requires_grad_()
            for d in (8, 8, 5)
        )
    )
    if flatten:
        args = tuple((x.flatten(1) for x in args))
    csr = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    kwargs = {}
    if bias_kind:
        shape = (7, 2) if bias_kind == 1 else (3, 2, 4, 4)
        kwargs["bias"] = (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
    assert_cuda_report(module, (*args, csr), **kwargs)


def compilation_case(channels):
    module = lunaeq.VarlenAttention(num_heads=2).cuda()
    samples = []
    for edges, nodes in ((17, 3), (41, 5)):
        q, k, value = tuple((torch.randn(edges, 2, width, device="cuda") * 0.2).requires_grad_()
                            for width in (channels, channels, channels + 1))
        csr = torch.linspace(0, edges, nodes + 1, device="cuda", dtype=torch.int32)
        kwargs = {}
        samples.append(((q, k, value, csr), kwargs))
    return module, samples


@pytest.mark.cuda
@pytest.mark.parametrize("channels", (4, 8))
def test_compile(channels):
    module, samples = compilation_case(channels)
    assert_compiled(module, samples)


@pytest.mark.cuda
def test_aoti(tmp_path):
    cases = [compilation_case(channels) for channels in (4, 8)]
    assert_aoti(cases, tmp_path)
