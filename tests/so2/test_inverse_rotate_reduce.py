"""Inverse rotation and receiver sum or attention reduction."""

import gc
from functools import partial
from unittest.mock import patch
from weakref import ref

import pytest
import torch
import torch.nn.functional as F

from LunaEquivariance.so2.inverse_rotate_reduce.kernels import utils as receiver_utils
from LunaEquivariance.so2.inverse_rotate_reduce.kernels import double_backward as receiver_double_backward
from LunaEquivariance.so2.inverse_rotate_reduce.kernels import backward as receiver_backward
from LunaEquivariance.so2.inverse_rotate_reduce.kernels import forward as receiver_forward
from LunaEquivariance.so2.inverse_rotate_reduce import interface as receiver
from LunaEquivariance import so2 as op
from LunaEquivariance.so2.utils.layout import get_retained_layout
from LunaEquivariance.so2.linear.interface import project_sectors
from LunaEquivariance.so2.linear.interface import project_recombined_sectors
from tests.so2.reference.recombination import recombine_sectors
from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
    attention_aggregation,
    build_receiver_csr,
)
from LunaEquivariance.so2.inverse_rotate_reduce.kernels.utils import attention_dimensions
from tests.so2.reference.inverse_rotation_reduction import (
    aggregate_attention,
    aggregate_sum,
    inverse_rotate_reduce as reduce_reference,
)
from LunaEquivariance.so2.inverse_rotate_reduce.kernels.utils import use_receiver_double_backward
from LunaEquivariance.so2.rotation import interface as wigner
from tests.testing_utils import assert_reference_close
from tests.so2.checks import make_cotangent, strided_copy
from tests.so2.checks import fp64
from tests.so2.checks import assert_autograd_chain, tiny_so2_graph


def pack_values(raw, lmax, mmax):
    values, _ = recombine_sectors(
        (raw[0].flatten(1), *raw[1:]),
        lmax=lmax,
        mmax=mmax,
        channels=raw[0].shape[-1],
        extra_channels=0,
    )
    return values


def case(lmax, mmax):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(912 + lmax + mmax)
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=128)
    receivers = torch.repeat_interleave(
        torch.arange(6, device="cuda"), torch.tensor([9, 0, 2, 1, 0, 16], device="cuda")
    )
    edges = receivers.numel()
    raw = [torch.randn(edges, lmax + 1, 128, device="cuda") * 0.2]
    raw.extend(
        torch.randn(edges, 2, 2 * width * 128, device="cuda") * 0.2
        for width in spec.sector_widths[1:]
    )
    logits = torch.randn(edges, 8, device="cuda")
    envelope = torch.rand(edges, 1, device="cuda") + 0.1
    envelope[3] = 0
    wigner = torch.randn(edges, spec.compact_elements, device="cuda") * 0.2
    return spec, (*raw, logits, envelope, wigner), receivers, 6


def differentiate(output, inputs, cotangent, directions):
    first = torch.autograd.grad(
        output, inputs, cotangent, create_graph=True, allow_unused=True
    )
    active = [(g, v) for g, v in zip(first, directions, strict=True) if g is not None]
    second = torch.autograd.grad(
        tuple(g for g, _ in active),
        (*inputs, cotangent),
        tuple(v for _, v in active),
        allow_unused=True,
    )
    return (output, *first, *second)


def assert_slots(actual, expected, rtol=3e-4, atol=2e-5):
    for slot, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert (value is None) == (reference is None), f"unused gradient slot {slot}"
        if value is None:
            continue
        torch.testing.assert_close(
            value.double(),
            reference.double(),
            rtol=rtol,
            atol=atol,
            msg=lambda text: f"slot {slot}: {text}",
        )


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(2, 0), (2, 2), (4, 2), (4, 4)])
def test_raw_attention_all_input_and_cotangent_d2(lmax, mmax):
    spec, data, receivers, nodes = case(lmax, mmax)
    row_ptr = build_receiver_csr(receivers, nodes)
    cotangent_data = torch.randn(6, spec.full_components, 128, device="cuda")
    directions = tuple(torch.randn_like(value) * 0.2 for value in data)

    def evaluate(mode):
        dtype = torch.float64 if mode == "fp64" else torch.float32
        inputs = tuple(value.to(dtype).detach().requires_grad_() for value in data)
        raw, logits, envelope, wigner = inputs[: mmax + 1], *inputs[mmax + 1 :]
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        if mode == "fp64":
            output = aggregate_attention(
                raw,
                logits,
                envelope,
                wigner,
                receivers,
                row_ptr,
                nodes,
                1e-6,
                lmax=lmax,
                mmax=mmax,
            )
        else:
            if mode == "signed":
                values = [raw[0]]
                for sector in raw[1:]:
                    pp, pn, np, nn = sector.reshape(sector.shape[0], 4, -1).unbind(1)
                    values.append(torch.stack((pp - nn, np + pn), dim=1))
                values = tuple(values)
            else:
                values = raw if mode == "raw" else pack_values(raw, lmax, mmax)
            output = attention_aggregation(
                values,
                logits,
                envelope,
                wigner,
                receivers,
                row_ptr,
                nodes,
                1e-6,
                lmax=lmax,
                mmax=mmax,
            )
        return differentiate(
            output, inputs, cotangent, tuple(v.to(dtype) for v in directions)
        )

    actual = evaluate("raw")
    assert_slots(actual, evaluate("packed"))
    assert_slots(actual, evaluate("signed"))
    assert_slots(actual, evaluate("fp64"))


@pytest.mark.cuda
@pytest.mark.parametrize("mmax", [0, 2])
def test_raw_attention_projection_parameter_d2(mmax):
    lmax, cin, channels = 4, 4, 128
    spec, data, receivers, nodes = case(lmax, mmax)
    edges = receivers.numel()
    row_ptr = build_receiver_csr(receivers, 6)
    sectors = (torch.randn(edges, (lmax + 1) * cin, device="cuda"),) + tuple(
        torch.randn(edges, 2, width * cin, device="cuda")
        for width in spec.sector_widths[1:]
    )
    weights = (
        torch.randn((lmax + 1) * channels, (lmax + 1) * cin, device="cuda") * 0.1,
    ) + tuple(
        torch.randn(2 * width * channels, width * cin, device="cuda") * 0.1
        for width in spec.sector_widths[1:]
    )
    bias = torch.randn((lmax + 1) * channels, device="cuda") * 0.1
    leaves = (*sectors, *weights, bias, *data[-3:])
    cotangent_data = torch.randn(6, spec.full_components, channels, device="cuda")
    directions = tuple(torch.randn_like(value) * 0.1 for value in leaves)

    def evaluate(mode):
        dtype = torch.float64 if mode == "fp64" else torch.float32
        inputs = tuple(value.to(dtype).detach().requires_grad_() for value in leaves)
        count = mmax + 1
        x, w = inputs[:count], inputs[count : 2 * count]
        b, logits, envelope, wigner = inputs[2 * count :]
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        if mode == "fp64":
            raw = (F.linear(x[0], w[0], b).reshape(edges, lmax + 1, channels),) + tuple(
                F.linear(value, weight)
                for value, weight in zip(x[1:], w[1:], strict=True)
            )
            output = aggregate_attention(
                raw,
                logits,
                envelope,
                wigner,
                receivers,
                row_ptr,
                nodes,
                1e-6,
                lmax=lmax,
                mmax=mmax,
            )
        else:
            if mode == "signed":
                scalar, *planes = project_recombined_sectors(x, w, b)
                values = (scalar.reshape(edges, lmax + 1, channels), *planes)
            else:
                _, values = project_sectors(
                    x,
                    partial(F.linear, weight=w[0], bias=b),
                    tuple(partial(F.linear, weight=weight) for weight in w[1:]),
                    output_channels=channels,
                )
            output = attention_aggregation(
                values,
                logits,
                envelope,
                wigner,
                receivers,
                row_ptr,
                nodes,
                1e-6,
                lmax=lmax,
                mmax=mmax,
            )
        return differentiate(
            output, inputs, cotangent, tuple(v.to(dtype) for v in directions)
        )

    expected = evaluate("fp64")
    assert_slots(evaluate("raw"), expected, rtol=5e-4, atol=5e-5)
    assert_slots(evaluate("signed"), expected, rtol=5e-4, atol=5e-5)


@pytest.mark.cuda
def test_raw_attention_broadcast_directions():
    spec, data, receivers, nodes = case(2, 2)
    inputs = tuple(value.detach().requires_grad_() for value in data)
    row_ptr = build_receiver_csr(receivers, 6)
    cotangent = (
        torch.ones((), device="cuda")
        .expand(6, spec.full_components, 128)
        .requires_grad_()
    )
    directions = tuple(
        torch.ones((), device="cuda").expand_as(value).requires_grad_()
        for value in inputs
    )

    def third_derivative(operation, inputs, cotangent, directions):
        output = operation(
            inputs[:3], *inputs[3:], receivers, row_ptr, nodes, 1e-6, lmax=2, mmax=2
        )
        first = torch.autograd.grad(output, inputs, cotangent, create_graph=True)
        second = torch.autograd.grad(
            first, (*inputs, cotangent), directions, create_graph=True
        )
        assert all(torch.isfinite(value).all() for value in second)
        return torch.autograd.grad(
            sum(value.sum() for value in second), (*inputs, cotangent, *directions),
            materialize_grads=True,
        )

    actual = third_derivative(attention_aggregation, inputs, cotangent, directions)
    expected = third_derivative(aggregate_attention, *fp64((inputs, cotangent, directions)))
    for value, reference in zip(actual, expected, strict=True):
        assert_reference_close(value, reference)


def derivatives(output, leaves, cotangent, directions):
    first = torch.autograd.grad(output, leaves, cotangent, create_graph=True)
    second = torch.autograd.grad(
        first, (*leaves, cotangent), directions, allow_unused=True
    )
    return (output, *first, *second)


def compare(actual, expected, rtol=8e-4, atol=5e-5):
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if b is None:
            assert a is None or torch.count_nonzero(a) == 0
        else:
            torch.testing.assert_close(
                a.double(),
                b.double(),
                rtol=rtol,
                atol=atol,
                msg=lambda message: f"slot {index}: {message}",
            )


@pytest.mark.parametrize("heads,width", [(1, 1), (3, 7), (5, 24), (8, 16), (2, 48)])
def test_attention_dimension_contract(heads, width):
    assert attention_dimensions(
        torch.empty(2, 9, heads * width), torch.empty(2, heads)
    ) == (heads * width, heads, width)


@pytest.mark.parametrize("channels,heads", [(0, 2), (16, 0), (7, 2)])
def test_attention_inconsistent_dimensions(channels, heads):
    with pytest.raises(ValueError, match="positive and divisible"):
        attention_dimensions(torch.empty(2, 9, channels), torch.empty(2, heads))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "heads,width,lmax,mmax",
    [
        (1, 1, 0, 0),
        (3, 7, 2, 2),
        (5, 24, 2, 0),
        (8, 16, 2, 2),
        (2, 48, 2, 1),
        (3, 7, 8, 5),
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_attention_dimensions_all_derivatives(heads, width, lmax, mmax, raw):
    torch.manual_seed(2026)
    channels = heads * width
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    receivers = torch.tensor([0, 0, 2, 2, 2, 3], device="cuda")
    nodes = torch.empty(4, 1, device="cuda")
    row_ptr = build_receiver_csr(receivers, 4)
    edges = receivers.numel()
    if raw:
        features = [torch.randn(edges, lmax + 1, channels, device="cuda") * 0.2]
        features += [
            torch.randn(edges, 2, 2 * w * channels, device="cuda") * 0.2
            for w in spec.sector_widths[1:]
        ]
    else:
        features = [
            torch.randn(edges, spec.reduced_components, channels, device="cuda") * 0.2
        ]
    data = (
        *features,
        torch.randn(edges, heads, device="cuda") * 0.2,
        torch.rand(edges, 1, device="cuda") + 0.2,
        torch.randn(edges, spec.compact_elements, device="cuda") * 0.2,
    )
    cotangent_data = torch.randn(4, spec.full_components, channels, device="cuda")
    directions = tuple(torch.randn_like(v) * 0.1 for v in data)
    result = []
    for reference in (False, True):
        dtype = torch.float64 if reference else torch.float32
        leaves = tuple(v.to(dtype).detach().requires_grad_() for v in data)
        features, logits, envelope, wigner = leaves[:-3], *leaves[-3:]
        cotangent = cotangent_data.to(dtype).detach().requires_grad_()
        operation = aggregate_attention if reference else attention_aggregation
        output = operation(
            features if raw else features[0],
            logits,
            envelope,
            wigner,
            receivers,
            row_ptr,
            nodes.shape[0],
            1e-6,
            lmax=lmax,
            mmax=mmax,
        )
        result.append(
            derivatives(
                output, leaves, cotangent, tuple(v.to(dtype) for v in directions)
            )
        )
    compare(*result)


class Launches:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            self.calls.append((args, kwargs))

        return record


@pytest.mark.parametrize("raw", [False, True])
def test_m4_launches_and_derivative_slots(raw):
    messages = tuple(
        torch.randn(3, 5, 8)
        if m == 0
        else torch.randn(3, 2, 2 * (5 - m) * 8)
        if raw
        else torch.randn(3, 2 * (5 - m), 8)
        for m in range(5)
    )
    envelope, wigner, indices = (
        torch.ones(3),
        torch.randn(3, 25, 25),
        torch.tensor([0, 0, 2]),
    )
    grad = torch.randn(4, 25, 8)
    names = (
        "sector_message_backward_kernel",
        "sector_parameter_backward_kernel",
        "sector_message_double_backward_kernel",
        "sector_parameter_double_backward_kernel",
    )
    calls = [Launches() for _ in range(5)]
    with patch.object(receiver_forward, "receiver_forward_kernel", calls[0]):
        result = receiver_forward.receiver_sector_tail_forward(
            messages,
            envelope,
            wigner,
            receiver.receiver_row_ptr(indices, 4),
            4,
            lmax=4,
            mmax=4,
            raw_complex=raw,
        )
    assert result.shape == (4, 25, 8)
    assert all(calls[0].calls[0][0][0][i] is messages[i] for i in range(5))
    with patch.object(receiver_backward, names[0], calls[1]), patch.object(receiver_backward, names[1], calls[2]):
        g = receiver_backward.sum_backward(
            grad,
            messages,
            envelope,
            wigner,
            indices,
            4,
            lmax=4,
            mmax=4,
            wigner_layout=0,
            raw_complex=raw,
        )
    with patch.object(receiver_double_backward, names[2], calls[3]), patch.object(receiver_double_backward, names[3], calls[4]):
        h = receiver_double_backward.sum_double_backward(
            grad,
            messages,
            envelope,
            wigner,
            indices,
            4,
            torch.ones_like(envelope),
            torch.ones_like(wigner),
            tuple(torch.ones_like(x) for x in messages),
            lmax=4,
            mmax=4,
            wigner_layout=0,
            raw_complex=raw,
        )
    for c in calls[1:]:
        assert [kw["order"] for _, kw in c.calls] == list(range(5))
        assert c.calls[-1][1]["rows"] == 2
        assert c.calls[-1][1]["raw_complex"] == raw
    assert len(g) == 7 and len(h) == 8
    assert [x.shape for x in g[2:]] == [x.shape for x in messages]
    assert [x.shape for x in h[3:]] == [x.shape for x in messages]


def test_compact_m4_schema():
    schema = torch.ops.so2.compact_sector_tail_input_backward.default._schema
    fifth = next(a for a in schema.arguments if a.name == "message_gradients")
    assert fifth.alias_info.is_write
    assert "messages" in [a.name for a in schema.arguments]


@pytest.mark.parametrize("raw", [False, True])
def test_compact_m4_backward_passes_fifth_input_and_output(raw):
    messages = tuple(
        torch.randn(3, 5, 8)
        if m == 0
        else torch.randn(3, 2, 2 * (5 - m) * 8)
        if raw
        else torch.randn(3, 2 * (5 - m), 8)
        for m in range(5)
    )
    args_seen = []
    with patch.object(
        torch.ops.so2,
        "compact_sector_tail_input_backward",
        side_effect=lambda *args: args_seen.append(args),
    ):
        gradients = receiver_backward.compact_tail_backward(
            torch.randn(4, 25, 8),
            messages,
            torch.ones(3),
            torch.randn(3, 165, 1),
            torch.tensor([0, 0, 2]),
            lmax=4,
            mmax=4,
            raw_complex=raw,
            wigner_layout=8,
        )
    args = args_seen[0]
    assert all(args[1][i] is messages[i] for i in range(5))
    assert args[5][4] is gradients[5]  # fifth destination in the registered op
    assert [g.shape for g in gradients[1:]] == [x.shape for x in messages]


@pytest.mark.parametrize("mmax", [2, 4])
def test_retained_receiver_layout_is_explicit(mmax):
    from LunaEquivariance.so2.utils.layout import get_retained_layout

    spec = get_retained_layout(lmax=4, mmax=mmax, channels=8)
    rotations = torch.randn(3, spec.compact_elements)
    messages = tuple(
        torch.randn(3, 5 if m == 0 else 2 * (5 - m), 8) for m in range(mmax + 1)
    )
    with (
        patch.object(
            receiver,
            "prepare_inputs",
            side_effect=lambda messages, env, w, r, *args, **kw: (env, w, r),
        ),
        patch.object(receiver.SumAggregation, "apply", return_value=None) as apply,
    ):
        receiver.sum_aggregation(
            messages,
            torch.ones(3),
            rotations,
            torch.tensor([0, 0, 2]),
            4,
            lmax=4,
            mmax=mmax,
            rotation_storage="retained_rows",
        )
    assert apply.call_args.args[1].data_ptr() == rotations.data_ptr()
    assert apply.call_args.args[1].shape == (3, spec.compact_elements, 1)
    assert apply.call_args.args[8] == 4 + mmax


def test_scalar_consumer_preserves_rotation_maximum_order():
    from LunaEquivariance.so2.utils.layout import get_retained_layout

    spec = get_retained_layout(lmax=4, mmax=2, channels=8)
    rotations = torch.randn(3, spec.compact_elements)
    message = torch.randn(3, 5, 8)
    with (
        patch.object(
            receiver,
            "prepare_inputs",
            side_effect=lambda messages, env, w, r, *args, **kw: (env, w, r),
        ),
        patch.object(receiver.ScatterSum, "apply", return_value=None) as apply,
    ):
        receiver.structured_sector_tail(
            (message,),
            torch.ones(3),
            rotations,
            torch.tensor([0, 0, 2]),
            4,
            lmax=4,
            mmax=0,
            wigner_layout="retained_rows",
            wigner_mmax=2,
        )
    assert apply.call_args.args[6] == 6
    receiver_utils.sum_forward_validate(
        (message,),
        torch.ones(3),
        rotations.unsqueeze(-1),
        torch.tensor([0, 0, 2]),
        4,
        lmax=4,
        mmax=0,
        wigner_layout=6,
    )


def check_sum_derivatives(
    lmax, mmax, channels, scalar_only, raw, wigner_direction_style="contiguous"
):
    from LunaEquivariance.so2.utils.layout import get_angular_layout, get_retained_layout

    torch.manual_seed(19)
    nodes = 5
    active_mmax = 0 if scalar_only else mmax
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    sectors = get_angular_layout(lmax=lmax, mmax=active_mmax, channels=channels)
    indices = torch.tensor([0, 0, 0, 2, 2], device="cuda")
    if (lmax, mmax, channels) == (2, 2, 128) and not scalar_only:
        indices = indices.repeat_interleave(17)
    edges = indices.numel()
    inputs = [
        torch.randn(edges, device="cuda"),
        torch.randn(edges, retained.compact_elements, device="cuda"),
    ]
    inputs += [
        torch.randn(
            (edges, 2, 2 * (lmax - m + 1) * channels)
            if raw and m
            else (edges, lmax + 1 if m == 0 else 2 * (lmax - m + 1), channels),
            device="cuda",
        )
        for m in range(active_mmax + 1)
    ]
    actual_inputs = [x.requires_grad_() for x in inputs]
    ref_inputs = [x.detach().double().requires_grad_() for x in inputs]

    def reference(env, rotation, *messages):
        if raw:
            combined = [messages[0]]
            for m, value in enumerate(messages[1:], 1):
                value = value.reshape(edges, 2, 2, lmax - m + 1, channels)
                combined.append(
                    torch.cat(
                        (
                            value[:, 0, 0] - value[:, 1, 1],
                            value[:, 1, 0] + value[:, 0, 1],
                        ),
                        dim=1,
                    )
                )
            messages = combined
        pieces = []
        for degree in range(lmax + 1):
            values = []
            for order in range(min(degree, active_mmax) + 1):
                width = lmax - order + 1
                for sign in (0,) if order == 0 else (1, -1):
                    row = degree - order + (width if sign == -1 else 0)
                    reduced_row = sectors.dense_rows.index(
                        degree * degree + degree + sign * order
                    )
                    # Full retained layout can contain more orders than the consumer.
                    full_sectors = get_angular_layout(
                        lmax=lmax, mmax=mmax, channels=channels
                    )
                    retained_row = full_sectors.dense_rows.index(
                        sectors.dense_rows[reduced_row]
                    )
                    start = retained.compact_row_offsets[retained_row]
                    values.append(
                        rotation[:, start : start + 2 * degree + 1, None]
                        * messages[order][:, row, None, :]
                    )
            pieces.append(sum(values))
        edge_values = torch.cat(pieces, dim=1) * env[:, None, None]
        return torch.zeros(
            nodes, (lmax + 1) ** 2, channels, device="cuda", dtype=torch.float64
        ).index_add(0, indices, edge_values)

    if scalar_only:
        actual = receiver.structured_sector_tail(
            (actual_inputs[2],),
            actual_inputs[0],
            actual_inputs[1],
            indices,
            nodes,
            lmax=lmax,
            mmax=0,
            wigner_layout="retained_rows",
            wigner_mmax=mmax,
        )
    else:
        actual = receiver.sum_aggregation(
            tuple(actual_inputs[2:]),
            actual_inputs[0],
            actual_inputs[1],
            indices,
            nodes,
            lmax=lmax,
            mmax=mmax,
            rotation_storage="retained_rows",
            raw_complex=raw,
        )
    expected = reference(*ref_inputs)
    torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-5)
    cotangent = torch.randn_like(actual).requires_grad_()
    ref_cotangent = cotangent.detach().double().requires_grad_()
    first = torch.autograd.grad(
        (actual * cotangent).sum(), actual_inputs, create_graph=True
    )
    ref_first = torch.autograd.grad(
        (expected * ref_cotangent).sum(), ref_inputs, create_graph=True
    )
    directions = [torch.randn_like(x) for x in first]
    if wigner_direction_style != "contiguous":
        # A full-sized backing allocation makes the old wrong broadcast address
        # readable, so this test reports a numerical error without an illegal read.
        backing = torch.randn(edges, 2 * retained.compact_elements, device="cuda")
        directions[1] = (
            backing[:, :1].expand(edges, retained.compact_elements)
            if wigner_direction_style == "broadcast"
            else backing[:, ::2]
        )
    second = torch.autograd.grad(first, actual_inputs + [cotangent], directions)
    ref_second = torch.autograd.grad(
        ref_first,
        ref_inputs + [ref_cotangent],
        tuple(value.double() for value in directions),
    )
    for got, want in zip(first + second, ref_first + ref_second):
        torch.testing.assert_close(got.double(), want, atol=3e-4, rtol=3e-4)

    if not scalar_only:
        # Exercise the input-only VJP used when the envelope has no gradients.
        # Force the existing H100 L2 reuse schedule on A100 to compile/check it too.
        reuse = (lmax, mmax, channels) == (2, 2, 128)
        with patch.object(receiver, "use_l2_tail_reuse", return_value=reuse):
            fast = receiver.sum_aggregation(
                tuple(actual_inputs[2:]),
                actual_inputs[0].detach(),
                actual_inputs[1],
                indices,
                nodes,
                lmax=lmax,
                mmax=mmax,
                rotation_storage="retained_rows",
                raw_complex=raw,
            )
            fast_first = torch.autograd.grad(
                (fast * cotangent).sum(), actual_inputs[1:]
            )
        torch.testing.assert_close(
            fast.double(), expected.detach(), atol=3e-5, rtol=3e-5
        )
        for got, want in zip(fast_first, ref_first[1:]):
            torch.testing.assert_close(
                got.double(), want.detach(), atol=3e-4, rtol=3e-4
            )


@pytest.mark.cuda
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("edges", [0, 7])
def test_receiver_hvp_dense_and_empty_edges(raw, edges):
    nodes, lmax, mmax, channels = 6, 2, 2, 8
    torch.manual_seed(29)
    receivers = torch.tensor(
        [0, 0, 0, 2, 2, 4, 4][:edges], device="cuda", dtype=torch.int64
    )

    def rand(*shape):
        return torch.randn(*shape, device="cuda") * 0.2

    messages = tuple(
        rand(edges, lmax + 1, channels)
        if order == 0
        else rand(edges, 2, 2 * (lmax - order + 1) * channels)
        if raw
        else rand(edges, 2 * (lmax - order + 1), channels)
        for order in range(mmax + 1)
    )
    rotations = rand(edges, 9, 9)
    arguments = (
        rand(nodes, 9, channels),
        messages,
        rand(edges),
        rotations,
        receivers,
        nodes,
        rand(edges),
        torch.randn_like(rotations),
        tuple(torch.randn_like(value) for value in messages),
    )
    reference = receiver_double_backward.sum_double_backward(
        *arguments, lmax=lmax, mmax=mmax, wigner_layout=0, raw_complex=raw
    )
    actual = receiver_double_backward.sum_double_backward(
        *arguments,
        lmax=lmax,
        mmax=mmax,
        wigner_layout=0,
        raw_complex=raw,
        row_ptr=receiver.receiver_row_ptr(receivers, nodes),
    )
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, atol=3e-5, rtol=3e-5)
    assert torch.count_nonzero(actual[0][[1, 3, 5]]) == 0


@pytest.mark.cuda
@pytest.mark.parametrize("raw_complex", [False, True])
def test_compact_receiver_hvp_releases_tensor_arguments(monkeypatch, raw_complex):
    if torch.cuda.get_device_capability() not in ((8, 0), (9, 0)):
        pytest.skip("compact receiver HVP uses SM80 and SM90 schedules")
    torch.manual_seed(19)
    nodes, channels = 128, 128
    counts = torch.tensor([0, 13, 17, 35], device="cuda").repeat(nodes // 4)
    counts[-1] += 1
    index_dtype = torch.int32 if raw_complex else torch.int64
    receivers = torch.arange(nodes, device="cuda", dtype=index_dtype).repeat_interleave(counts)
    edges = receivers.numel()
    row_ptr = receiver.receiver_row_ptr(receivers, nodes)
    references = []
    launch = receiver.compact_receiver_double_backward

    def track(grad_output, messages, envelope, wigner, receivers, row_ptr,
              message_directions, wigner_direction, *, raw_complex):
        references.extend(ref(tensor) for tensor in (*messages, *message_directions))
        return launch(grad_output, messages, envelope, wigner, receivers, row_ptr,
                      message_directions, wigner_direction, raw_complex=raw_complex)

    def differentiate_once():
        shapes = ((edges, 2, 4 * channels), (edges, 2, 2 * channels)) if raw_complex else (
            (edges, 4, channels), (edges, 2, channels)
        )
        messages = tuple(
            (torch.randn(shape, device="cuda") * 0.2).requires_grad_()
            for shape in ((edges, 3, channels), *shapes)
        )
        wigner = (torch.randn(edges, 35, device="cuda") * 0.2).requires_grad_()
        envelope = torch.rand(edges, device="cuda")
        output = receiver.sum_aggregation(
            messages, envelope, wigner, receivers, nodes,
            lmax=2, mmax=2, row_ptr=row_ptr, raw_complex=raw_complex,
            rotation_storage="retained_rows", wigner_mmax=2,
        )
        inputs = (*messages, wigner)
        directions = tuple(torch.randn_like(tensor) for tensor in inputs)
        cotangent = torch.randn_like(output, requires_grad=True)
        actual = differentiate(output, inputs, cotangent, directions)

        ref_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
        expected = aggregate_sum(
            ref_inputs[:-1], envelope.double(), ref_inputs[-1], receivers, nodes,
            lmax=2, mmax=2, raw_complex=raw_complex, rotation_storage="retained_rows",
        )
        expected = differentiate(
            expected, ref_inputs, cotangent.detach().double().requires_grad_(),
            tuple(direction.double() for direction in directions),
        )
        assert_slots(actual, expected, atol=1e-4)

    monkeypatch.setattr(receiver, "compact_receiver_double_backward", track)
    differentiate_once()
    torch.cuda.synchronize()
    gc.collect()
    assert references and all(reference() is None for reference in references)


@pytest.mark.cuda
@pytest.mark.parametrize("direction_style", ["broadcast", "sliced"])
def test_retained_receiver_strided_rotation_direction(direction_style):
    check_sum_derivatives(
        3,
        2,
        8,
        False,
        True,
        wigner_direction_style=direction_style,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,mmax,channels", [(2, 2, 128), (3, 2, 128), (4, 2, 8), (4, 4, 8)]
)
@pytest.mark.parametrize("scalar_only", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_retained_sum_derivatives(lmax, mmax, channels, scalar_only, raw):
    check_sum_derivatives(lmax, mmax, channels, scalar_only, raw)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,mmax,channels", [(0, 0, 7), (3, 0, 33), (5, 5, 7), (3, 2, 257)]
)
@pytest.mark.parametrize("raw", [False, True])
def test_sum_dimensions(lmax, mmax, channels, raw):
    check_sum_derivatives(lmax, mmax, channels, False, raw)


@pytest.mark.parametrize(
    "capability,expected", [((8, 0), False), ((8, 6), True), ((9, 0), True)]
)
def test_dense_hvp_architecture_selection(capability, expected):
    with patch("torch.cuda.get_device_capability", return_value=capability) as query:
        assert (
            use_receiver_double_backward(
                torch.device("cuda:1"), nodes=512, lmax=4, channels=128, wigner_layout=0
            )
            is expected
        )
        query.assert_called_once_with(torch.device("cuda:1"))


@pytest.mark.parametrize(
    "nodes,lmax,channels,layout",
    [
        (128, 4, 128, 0),
        (256, 5, 256, 0),
        (512, 2, 128, 0),
        (512, 4, 33, 0),
        (512, 4, 64, 0),
        (2048, 4, 128, 6),
        (2048, 4, 128, 1),
        (0, 4, 128, 0),
    ],
)
def test_preserved_schedules_do_not_query_cuda(nodes, lmax, channels, layout):
    with patch(
        "torch.cuda.get_device_capability", side_effect=AssertionError("unneeded query")
    ):
        assert use_receiver_double_backward(
            torch.device("cuda:1"),
            nodes=nodes,
            lmax=lmax,
            channels=channels,
            wigner_layout=layout,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("receiver_owned", [False, True])
def test_dispatched_derivatives_against_reference(
    mmax, receiver_owned, dataflow_reference
):
    synthetic_graph = dataflow_reference.synthetic_graph
    legal_wigner = dataflow_reference.legal_wigner
    wigner_block_mask = dataflow_reference.wigner_block_mask
    complex_reference = dataflow_reference.complex_reference
    from LunaEquivariance.so2.utils.layout import get_angular_layout
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as api

    torch.manual_seed(42)
    device = torch.device("cuda")
    nodes, lmax, channels = 8, 3, 7
    graph = synthetic_graph(nodes, 3, seed=42, device=device)
    edges = graph.shape[1]
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    rows = torch.tensor(spec.dense_rows, device=device)
    messages = (torch.randn(edges, lmax + 1, channels, device=device),) + tuple(
        torch.randn(edges, 2, 2 * width * channels, device=device)
        for width in spec.sector_widths[1:]
    )
    inputs = (
        *messages,
        torch.rand(edges, device=device),
        legal_wigner(edges, lmax, seed=42, device=device),
    )
    mask = wigner_block_mask(lmax, device)
    masks = (None,) * (len(inputs) - 1) + (mask,)

    def reference(*values):
        logical = (values[0],) + tuple(
            complex_reference(v, width * channels).reshape(edges, 2 * width, channels)
            for v, width in zip(values[1:-2], spec.sector_widths[1:])
        )
        rotation = values[-1].index_select(1, rows)
        edge = torch.bmm(rotation.transpose(1, 2), torch.cat(logical, dim=1))
        edge = edge * values[-2][:, None, None]
        return edge.new_zeros(nodes, spec.full_components, channels).index_add(
            0, graph[1], edge
        )

    def operator(*values):
        return api.sum_aggregation(
            values[:-2],
            values[-2],
            values[-1],
            graph[1],
            nodes,
            lmax=lmax,
            mmax=mmax,
            raw_complex=True,
        )

    cotangent = torch.randn(nodes, spec.full_components, channels, device=device)
    directions = tuple(
        torch.randn_like(v) if m is None else torch.randn_like(v) * m
        for v, m in zip(inputs, masks)
    )

    def evaluate(fn):
        values = tuple(v.detach().clone().requires_grad_() for v in inputs)
        cot = cotangent.detach().clone().requires_grad_()
        output = fn(*values)
        first = torch.autograd.grad((output * cot).sum(), values, create_graph=True)
        second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(first, directions)), values + (cot,)
        )
        return (
            (output.detach(),),
            tuple(g.detach() for g in first),
            tuple(g.detach() for g in second),
        )

    expected = evaluate(reference)
    with patch.object(api, "use_receiver_double_backward", return_value=receiver_owned):
        actual = evaluate(operator)
    for computed, wanted, group_masks in zip(
        actual, expected, [(None,), masks, masks + (None,)]
    ):
        for got, want, m in zip(computed, wanted, group_masks):
            if m is not None:
                got, want = got * m, want * m
            torch.testing.assert_close(got, want, atol=3e-4, rtol=3e-4)


def test_public_sum_csr_handles_empty_receivers():
    from LunaEquivariance.so2.utils.graph import prepare_graph

    receivers = torch.tensor([0, 0, 2, 4], dtype=torch.int64)
    edges = torch.stack((torch.tensor([1, 3, 0, 5]), receivers))
    actual = prepare_graph(edges, 6).receiver_row_ptr
    torch.testing.assert_close(
        actual, torch.tensor([0, 2, 2, 3, 3, 4, 4], dtype=torch.int32)
    )


@pytest.mark.cuda
@pytest.mark.parametrize("stored_mmax", [0, 1, 2])
@pytest.mark.parametrize("shaped", [False, True])
def test_initial_message_uses_shared_rotation_storage(stored_mmax, shaped):
    torch.manual_seed(261)
    topology = tiny_so2_graph("cuda")
    edges, lmax, channels = topology.src.numel(), 2, 5
    compact = sum((2 * min(l, stored_mmax) + 1) * (2 * l + 1) for l in range(lmax + 1))
    message = torch.randn(edges, lmax + 1, channels, device="cuda", requires_grad=True)
    rotations = torch.randn(edges, compact, device="cuda", requires_grad=True)
    envelope = torch.rand(edges, device="cuda", requires_grad=True)
    targets = (message, rotations, envelope)
    ref_message, ref_rotation, ref_envelope = fp64(targets)
    raw = (message if shaped else message.flatten(1),)
    ref_raw = (ref_message if shaped else ref_message.flatten(1),)
    actual = op.inverse_rotate_reduce(raw, rotations, topology, lmax=lmax, mmax=0, edge_weight=envelope)
    wanted = reduce_reference(ref_raw, ref_rotation, topology, lmax=lmax, mmax=0, edge_weight=ref_envelope)
    assert_autograd_chain(actual, targets, wanted, (ref_message, ref_rotation, ref_envelope))


@pytest.mark.parametrize("case", ["bad_shape", "bad_rotation_width", "missing_rotation_orders", "attention_extra_orders"])
def test_inverse_reduce_rejects_inconsistent_storage(case):
    topology = tiny_so2_graph("cpu")
    lmax, mmax, channels = 2, 0, 3
    scalar = torch.randn(6, lmax + 1, channels)
    raw = (scalar,)
    rotations = torch.randn(6, 25)  # L=2, retained M=1.
    logits = None
    if case == "bad_shape":
        raw = (torch.randn(6, lmax, channels),)
    elif case == "bad_rotation_width":
        rotations = rotations[:, :-1]
    elif case == "missing_rotation_orders":
        mmax = 2
        raw = (scalar, torch.randn(6, 2, 12), torch.randn(6, 2, 6))
    else:
        logits = torch.randn(6, 1)
    expected = {
        "bad_shape": "shaped order zero",
        "bad_rotation_width": "rotation width",
        "missing_rotation_orders": "fewer orders",
        "attention_extra_orders": "matching rotation and message orders",
    }
    with pytest.raises(ValueError, match=expected[case]):
        op.inverse_rotate_reduce(raw, rotations, topology, lmax=lmax, mmax=mmax, logits=logits)


@pytest.mark.cuda
@pytest.mark.parametrize("raw", [False, True])
def test_compact_tail_narrow_message_gradients_gpu(raw):
    from LunaEquivariance.so2.utils.layout import get_retained_layout

    torch.manual_seed(184)
    spec = get_retained_layout(lmax=2, mmax=2, channels=4)
    rotation = torch.randn(4, spec.compact_elements, device="cuda")
    messages = tuple(
        torch.randn(
            (4, 2, 2 * (3 - m) * 4)
            if raw and m
            else (4, 3 if m == 0 else 2 * (3 - m), 4),
            device="cuda",
        )
        for m in range(3)
    )
    receivers = torch.tensor([0, 0, 1, 2], device="cuda")
    env = torch.rand(4, device="cuda")

    def evaluate(dense):
        inputs = []
        for value in messages:
            if dense:
                x = value.clone()
            else:
                backing = value.new_empty(
                    (value.shape[0], value.shape[1] + 2, value.shape[2])
                )
                x = backing[:, : value.shape[1], :]
                x.copy_(value)
            inputs.append(x.requires_grad_())
        w = rotation.clone().requires_grad_()
        output = receiver.sum_aggregation(
            inputs,
            env if dense else strided_copy(env),
            w,
            receivers if dense else strided_copy(receivers),
            3,
            lmax=2,
            mmax=2,
            rotation_storage="retained_rows",
            raw_complex=raw,
        )
        gradients = torch.autograd.grad(output.sum(), (*inputs, w))
        return output, *gradients

    for got, want in zip(evaluate(False), evaluate(True), strict=True):
        torch.testing.assert_close(got, want, rtol=3e-5, atol=3e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["sum", "strided"])
def test_attention_internal_workspace_cotangent_gpu(kind):
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as attention
    from LunaEquivariance.so2.utils.layout import get_retained_layout

    torch.manual_seed(185)
    spec = get_retained_layout(lmax=2, mmax=2, channels=128)
    values = (
        torch.randn(4, spec.reduced_components, 128, device="cuda"),
        torch.randn(4, 8, device="cuda"),
        torch.rand(4, 1, device="cuda") + 0.5,
        torch.randn(4, spec.compact_elements, device="cuda"),
    )
    receivers = torch.tensor([0, 0, 1, 2], device="cuda")
    row_ptr = attention.build_receiver_csr(receivers, 3)

    def evaluate(dense):
        inputs = tuple(x.detach().requires_grad_() for x in values)
        workspace = attention.packed_attention_op(
            *inputs, receivers, row_ptr, 3, 1e-6, 2, 2
        )
        g = make_cotangent(workspace, kind)
        if dense:
            g = g.contiguous()
        first = torch.autograd.grad(workspace, inputs, g, create_graph=True)
        second = torch.autograd.grad(
            first, inputs, tuple(torch.ones_like(x) for x in first), create_graph=True
        )
        return *first, *second

    for got, want in zip(evaluate(False), evaluate(True), strict=True):
        torch.testing.assert_close(got, want, rtol=5e-5, atol=5e-5)
