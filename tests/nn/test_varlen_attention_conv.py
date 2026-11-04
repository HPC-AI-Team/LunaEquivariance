"""Attention convolution outputs, envelopes and derivatives."""

import pytest
import torch

from LunaEquivariance import varlen_attention_conv
from LunaEquivariance.utils.testing import check_compatibility
import LunaEquivariance as lunaeq
from tests.testing_utils import assert_aoti, assert_compiled, assert_reference_close


def reference(q, k, v, bias, env, lengths, causal, return_out):
    ys = []
    os = []
    start = 0
    for atom, n in enumerate(lengths):
        end = start + n
        score = torch.einsum("ihd,jhd->hij", q[start:end], k[start:end]) * q.shape[
            -1
        ] ** (-0.5)
        if bias is not None:
            score = score + (
                bias[start:end].T[:, None, :]
                if bias.ndim == 2
                else bias[atom, :, :n, :n]
            )
        if causal:
            score = score.masked_fill(
                torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1),
                -float("inf"),
            )
        out = torch.einsum("hij,jhv->ihv", score.softmax(-1), v[start:end])
        os.append(out)
        ys.append(
            (out if env is None else out * env.reshape(-1)[start:end, None, None]).sum(
                0
            )
        )
        start = end
    y = torch.stack(ys) if ys else v[:0]
    return (y, torch.cat(os)) if return_out else (y,)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "envelope,return_out,bias_kind,causal,strided,flatten,edge_only",
    [
        (True, True, 1, False, False, False, False),
        (True, True, 1, False, False, False, True),
        (True, False, 0, False, True, False, False),
        (False, True, 0, False, False, True, False),
        (True, True, 2, True, True, False, False),
    ],
)
def test_attention_conv(
    bias_kind,
    causal,
    envelope,
    return_out,
    strided,
    flatten,
    edge_only,
    lengths=(0, 1, 3, 33),
    h=2,
    d=7,
    v=5,
):
    torch.manual_seed(314)
    e = sum(lengths)

    def tensor(shape):
        x = torch.randn(*shape, device="cuda") * 0.25
        if strided:
            storage = torch.empty((*shape[:-1], shape[-1] * 2), device="cuda")
            storage[..., ::2] = x
            x = storage[..., ::2]
        return x.requires_grad_()

    q, k, value = (tensor((e, h, d)), tensor((e, h, d)), tensor((e, h, v)))
    bias = (
        tensor((e, h))
        if bias_kind == 1
        else tensor((len(lengths), h, max(1, max(lengths)), max(1, max(lengths))))
        if bias_kind == 2
        else None
    )
    env = tensor((e, 1)) if envelope else None
    xs = (q, k, value, bias, env)
    refs = tuple(
        (x.detach().double().requires_grad_() if x is not None else None for x in xs)
    )
    csr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int64 if strided else torch.int32,
    )
    args = tuple((x.flatten(1) for x in xs[:3])) if flatten else xs[:3]
    result = varlen_attention_conv(
        *args,
        csr,
        envelope=env,
        bias=bias,
        causal=causal,
        return_attn_out=return_out,
        num_heads=h,
    )
    actual = result if return_out else (result,)
    if flatten:
        actual = tuple((x.unflatten(1, (h, -1)) for x in actual))
    expected = reference(*refs, lengths, causal, return_out)
    targets = tuple((x for x in xs if x is not None))
    reference_targets = tuple((x for x in refs if x is not None))
    for order in range(4):
        for a, b in zip(actual, expected):
            if a is None or b is None:
                if a is not None:
                    torch.testing.assert_close(a, torch.zeros_like(a))
                if b is not None:
                    torch.testing.assert_close(b, torch.zeros_like(b))
            elif a.numel():
                assert_reference_close(a, b)
        if order == 3:
            break
        if order == 0 and edge_only:
            actual, expected = (actual[1:], expected[1:])
        pairs = [
            (a, b)
            for a, b in zip(actual, expected)
            if a is not None and b is not None and a.requires_grad and b.requires_grad
        ]
        seeds = tuple((torch.randn_like(a).requires_grad_() for a, b in pairs))
        ref_seeds = tuple((x.detach().double().requires_grad_() for x in seeds))
        actual = torch.autograd.grad(
            tuple((a for a, b in pairs)),
            targets,
            seeds,
            create_graph=True,
            allow_unused=True,
        )
        expected = torch.autograd.grad(
            tuple((b for a, b in pairs)),
            reference_targets,
            ref_seeds,
            create_graph=True,
            allow_unused=True,
        )
        targets = (*targets, *seeds)
        reference_targets = (*reference_targets, *ref_seeds)


@pytest.mark.cuda
@pytest.mark.parametrize("d,v", [(7, 5), (32, 32), (33, 17)])
def test_attention_conv_tile_boundaries(d, v):
    test_attention_conv(
        bias_kind=1,
        causal=False,
        envelope=True,
        return_out=True,
        strided=False,
        flatten=False,
        edge_only=False,
        lengths=(0, 1, 31, 32, 33, 64, 65),
        d=d,
        v=v,
    )


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
@pytest.mark.parametrize("env_shape", [(7,), (7, 1)])
@pytest.mark.parametrize(
    "bias_kind,causal,flatten", [(0, False, False), (1, False, True), (2, True, False)]
)
def test_cuda_attention(env_shape, bias_kind, causal, flatten):
    from LunaEquivariance import VarlenAttention, VarlenAttentionConv

    torch.manual_seed(42)
    cls = VarlenAttentionConv
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
    kwargs.update(
        envelope=torch.rand(env_shape, device="cuda").requires_grad_(),
        return_attn_out=True,
    )
    assert_cuda_report(module, (*args, csr), **kwargs)


@pytest.mark.cuda
def test_attention_conv_default_tuple_loss():
    from LunaEquivariance import VarlenAttentionConv

    args = tuple(
        (torch.randn(3, 1, 8, device="cuda").requires_grad_() for _ in range(3))
    )
    csr = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    report = check_compatibility(
        VarlenAttentionConv(),
        (*args, csr),
        kwargs={"return_attn_out": True},
        derivative_order=3,
        warmup=0,
        repeats=1,
        verbose=False,
    )
    assert report["reference"] == "PyTorch"
    assert report["passed"], report


def compilation_case(channels):
    module = lunaeq.VarlenAttentionConv(num_heads=2).cuda()
    samples = []
    for edges, nodes in ((17, 3), (41, 5)):
        q, k, value = tuple((torch.randn(edges, 2, width, device="cuda") * 0.2).requires_grad_()
                            for width in (channels, channels, channels + 1))
        csr = torch.linspace(0, edges, nodes + 1, device="cuda", dtype=torch.int32)
        kwargs = {"envelope": (torch.rand(edges, 1, device="cuda") + 0.2).requires_grad_(),
                  "return_attn_out": True}
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
