"""CSR attention aggregation against FP64 Torch through third derivatives."""

import pytest
import torch

from LunaEquivariance import varlen_attention_conv


def reference(q, k, v, bias, env, lengths, causal, return_out):
    ys = []
    os = []
    start = 0
    for atom, n in enumerate(lengths):
        end = start + n
        score = (
            torch.einsum("ihd,jhd->hij", q[start:end], k[start:end])
            * q.shape[-1] ** -0.5
        )
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
@pytest.mark.parametrize("envelope,return_out,bias_kind,causal,strided,flatten", [
    (True, True, 1, False, False, False),
    (True, False, 0, False, True, False),
    (False, True, 0, False, False, True),
    (True, True, 2, True, True, False),
])
def test_attention_conv(
    bias_kind,
    causal,
    envelope,
    return_out,
    strided,
    flatten,
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

    q, k, value = tensor((e, h, d)), tensor((e, h, d)), tensor((e, h, v))
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
        x.detach().double().requires_grad_() if x is not None else None for x in xs
    )
    csr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int64 if strided else torch.int32,
    )
    args = tuple(x.flatten(1) for x in xs[:3]) if flatten else xs[:3]
    result = varlen_attention_conv(
        *args,
        csr,
        envelope=env,
        bias=bias,
        causal=causal,
        return_attn_out=return_out,
        num_heads=h
    )
    actual = result if return_out else (result,)
    if flatten:
        actual = tuple(x.unflatten(1, (h, -1)) for x in actual)
    expected = reference(*refs, lengths, causal, return_out)
    targets = tuple(x for x in xs if x is not None)
    reference_targets = tuple(x for x in refs if x is not None)
    for order in range(4):
        for a, b in zip(actual, expected):
            if a is None or b is None:
                if a is not None:
                    torch.testing.assert_close(a, torch.zeros_like(a))
                if b is not None:
                    torch.testing.assert_close(b, torch.zeros_like(b))
            elif a.numel():
                torch.testing.assert_close(a.double(), b, atol=3e-4, rtol=3e-4)
        if order == 3:
            break
        pairs = [
            (a, b)
            for a, b in zip(actual, expected)
            if a is not None and b is not None and a.requires_grad and b.requires_grad
        ]
        seeds = tuple(torch.randn_like(a).requires_grad_() for a, b in pairs)
        ref_seeds = tuple(x.detach().double().requires_grad_() for x in seeds)
        actual = torch.autograd.grad(
            tuple(a for a, b in pairs),
            targets,
            seeds,
            create_graph=True,
            allow_unused=True,
        )
        expected = torch.autograd.grad(
            tuple(b for a, b in pairs),
            reference_targets,
            ref_seeds,
            create_graph=True,
            allow_unused=True,
        )
        targets = (*targets, *seeds)
        reference_targets = (*reference_targets, *ref_seeds)
