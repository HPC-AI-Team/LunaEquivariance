"""Complete convolution and the three dataflow equations."""

import pytest
import torch

from LunaEquivariance import so2 as op
from LunaEquivariance.so2.linear.interface import project_after_inverse_reduce
from tests.so2.reference.recombination import recombine_activate as recombine_reference
from tests.so2.reference import gather_rotation as gather_equations
from tests.so2.reference.gather_rotation import rotate_gather as gather_reference
from tests.so2.reference.projection import linear as linear_reference
from tests.so2.reference import inverse_rotation_reduction as reduction_reference
from tests.so2.reference.inverse_rotation_reduction import inverse_rotate_reduce as reduce_reference
from tests.testing_utils import assert_reference_close
from tests.so2.checks import fp64
from tests.so2.checks import assert_autograd_chain, tiny_so2_graph


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax,channels", [(0, 0, 3), (2, 1, 5), (4, 2, 8)])
@pytest.mark.parametrize("activation,reduction", [("gate", "sum"), ("grid", "attention")])
def test_complete_convolution(lmax, mmax, channels, activation, reduction):
    torch.manual_seed(197)
    device = "cuda"
    topology = tiny_so2_graph(device)
    nodes, edges, output_channels = topology.num_nodes, topology.src.numel(), 4
    rows = (lmax + 1) ** 2
    compact = sum((2 * min(l, mmax) + 1) * (2 * l + 1) for l in range(lmax + 1))
    radial_width = 2 * channels * sum(lmax + 1 - m for m in range(mmax + 1))
    auxiliary = 2
    scale = 0.1

    def rand(*shape):
        return (scale * torch.randn(*shape, device=device)).requires_grad_()

    x = rand(nodes, rows, channels)
    rotations = rand(edges, compact)
    radial = rand(edges, radial_width)
    hidden_channels = output_channels * (2 if activation == "grid" else 1)
    extras = auxiliary + (3 * output_channels if activation == "grid" else lmax * output_channels)
    first = [rand(extras + (lmax + 1) * hidden_channels, (lmax + 1) * 2 * channels)]
    first += [rand(2 * (lmax + 1 - m) * hidden_channels, (lmax + 1 - m) * 2 * channels)
              for m in range(1, mmax + 1)]
    bias1 = rand(first[0].shape[0])
    second = [rand((lmax + 1) * output_channels, (lmax + 1) * output_channels)]
    second += [rand(2 * (lmax + 1 - m) * output_channels, (lmax + 1 - m) * output_channels)
               for m in range(1, mmax + 1)]
    bias2 = rand(second[0].shape[0])
    edge_weight = (1 + rand(edges)).detach().requires_grad_()
    logits = rand(edges, 2)
    retained_rows = sum(2 * min(l, mmax) + 1 for l in range(lmax + 1))
    to_grid = torch.randn(9, retained_rows, device=device) * scale
    from_grid = torch.randn(retained_rows, 9, device=device) * scale
    inputs = (x, rotations, radial, *first, bias1, *second, bias2, edge_weight, logits)
    refs = fp64(inputs)
    count = mmax + 1

    def run(functions, values, double=False):
        gather, linear, activate, reduce = functions
        x, rotation, radial = values[:3]
        first = values[3:3 + count]
        b1 = values[3 + count]
        second = values[4 + count:4 + 2 * count]
        b2, envelope, alpha = values[4 + 2 * count:]
        sectors = gather(x, rotation, radial, topology, lmax=lmax, mmax=mmax)
        raw = linear(sectors, first, b1, lmax=lmax, mmax=mmax)
        active, side = activate(
            raw, lmax=lmax, mmax=mmax, activation=activation, auxiliary_channels=auxiliary,
            to_grid=(to_grid.double() if double else to_grid) if activation == "grid" else None,
            from_grid=(from_grid.double() if double else from_grid) if activation == "grid" else None,
        )
        projected = linear(active, second, b2, lmax=lmax, mmax=mmax)
        output = reduce(
            projected, rotation, topology, lmax=lmax, mmax=mmax,
            edge_weight=envelope, logits=alpha if reduction == "attention" else None,
        )
        # The auxiliary branch also participates in the differentiated graph.
        return output.square().sum() + side.square().sum()

    actual = run((op.rotate_gather, op.linear, op.recombine_activate, op.inverse_rotate_reduce), inputs)
    wanted = run((gather_reference, linear_reference, recombine_reference, reduce_reference), refs, double=True)
    assert_autograd_chain(actual, inputs, wanted, refs)


@pytest.mark.cuda
@pytest.mark.parametrize(("groups", "experts", "nodes_per_group"), [
    (1, 2, 4), (1, 2, 36), (2, 2, 2), (16, 64, 2),
])
def test_segmented_projection_after_reduction(monkeypatch, groups, experts, nodes_per_group):
    torch.manual_seed(197)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    sizes = (nodes_per_group,) * groups
    nodes = sum(sizes)
    edge_sizes = (7,) if groups == 1 else (3, 4) * (groups // 2)
    offsets = torch.arange(0, nodes + 1, sizes[0], dtype=torch.int32, device="cuda")
    pattern = torch.tensor([0, 0, 0, 2, 2, 2, 3], device="cuda")
    receivers = pattern if groups == 1 else (
        torch.arange(0, nodes, 4, device="cuda")[:, None] + pattern
    ).flatten()
    row_ptr = torch.searchsorted(receivers, torch.arange(nodes + 1, device="cuda")).to(torch.int32)
    edges = receivers.numel()

    def rand(*shape):
        return (0.1 * torch.randn(*shape, device="cuda")).requires_grad_()

    sectors = tuple(rand(edges, width, 128) for width in (3, 4, 2))
    rotation = rand(edges, 35)
    weights = (rand(experts, 384, 384), rand(experts, 512, 256), rand(experts, 256, 128))
    coefficients, bias = rand(groups, experts), rand(384)
    inputs = (*sectors, rotation, *weights, coefficients, bias)
    reference_inputs = fp64(inputs)

    def reference(values):
        x0, x1, x2, rotation, *parameters = values
        w0, w1, w2, mix, bias = parameters
        weights = tuple(torch.einsum("se,eoi->soi", mix, w) for w in (w0, w1, w2))
        raw = tuple(
            torch.cat(tuple(
                torch.nn.functional.linear(part, weight, b)
                    for part, weight in zip(sector.split(edge_sizes), weight, strict=True)
            ))
            for sector, weight, b in zip(
                    (x0.flatten(1), x1.reshape(edges, 2, 256), x2.reshape(edges, 2, 128)),
                weights, (bias, None, None), strict=True,
            )
        )
        return reduction_reference.aggregate_sum(
            (raw[0].reshape(edges, 3, 128), *raw[1:]), rotation.new_ones(edges), rotation,
            receivers, nodes, lmax=2, mmax=2, raw_complex=True, rotation_storage="retained_rows",
        )

    actual = project_after_inverse_reduce(
        sectors, rotation, receivers, row_ptr, weights, coefficients, bias, sizes, offsets,
    )
    expected = reference(reference_inputs)
    assert_reference_close(actual, expected)
    seed = torch.randn_like(actual).requires_grad_()
    energy = torch.randn_like(actual)
    directions = tuple(torch.randn_like(x) for x in inputs[:4])

    def derivatives(output, values, seed, energy, directions):
        force = torch.autograd.grad(output, values[:4], seed, create_graph=True)
        gradients = torch.autograd.grad(
            (output, *force), (*values, seed), (energy, *directions),
        )
        return (*force, *gradients)

    actual_grads = derivatives(actual, inputs, seed, energy, directions)
    expected_grads = derivatives(
        expected, reference_inputs, seed.double(), energy.double(),
        tuple(value.double() for value in directions),
    )
    for value, wanted in zip(actual_grads, expected_grads, strict=True):
        assert_reference_close(value, wanted)


def tensor(*shape, device="cpu", requires_grad=False):
    return (
        torch.randn(*shape, dtype=torch.float64, device=device) * 0.2
    ).requires_grad_(requires_grad)


def pack_by_degree(dense, lmax, mmax):
    """Independent storage equation: degree, local order 0,+1,-1, full row."""
    blocks = []
    for degree in range(lmax + 1):
        center = degree * degree + degree
        orders = [0] + [
            sign * order
            for order in range(1, min(degree, mmax) + 1)
            for sign in (1, -1)
        ]
        blocks.extend(
            dense[:, center + order, degree**2 : (degree + 1) ** 2] for order in orders
        )
    return torch.cat(blocks, dim=1)


def receiver_graph(device="cpu"):
    receivers = torch.tensor([0, 0, 2], device=device)
    edges = torch.stack((torch.tensor([1, 2, 0], device=device), receivers))
    return edges, receivers, torch.tensor([0, 2, 2, 3, 3], device=device)


def test_gather_endpoint_and_local_order_contract():
    x = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4, 1)
    edges = torch.tensor([[0, 1], [1, 2]])
    dense = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    compact = pack_by_degree(dense, 1, 1)
    radial = torch.tensor([[[2.0, 3.0], [5.0, 7.0]], [[11.0, 13.0], [17.0, 19.0]]])
    zero, one = gather_equations.node_to_edge(
        x,
        edges,
        radial,
        compact,
        lmax=1,
        mmax=1,
        radial_sharing="degree",
        rotation_storage="retained_rows",
    )
    expected_zero = torch.stack(
        (
            torch.cat((x[0, 0] * 2, x[1, 0] * 3, x[0, 2] * 5, x[1, 2] * 7)),
            torch.cat((x[1, 0] * 11, x[2, 0] * 13, x[1, 2] * 17, x[2, 2] * 19)),
        )
    )
    expected_one = torch.tensor(
        [[[20.0, 56.0], [10.0, 42.0]], [[136.0, 228.0], [102.0, 190.0]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(zero, expected_zero)
    torch.testing.assert_close(one, expected_one)


def test_attention_envelope_is_used_in_normalization_and_final_weight():
    values = torch.tensor([[[2.0]], [[5.0]]], dtype=torch.float64)
    logits = torch.log(torch.tensor([[2.0], [3.0]], dtype=torch.float64))
    envelope = torch.tensor([[0.5], [1.0]], dtype=torch.float64)
    receivers = torch.zeros(2, dtype=torch.int64)
    result = reduction_reference.aggregate_attention(
        values,
        logits,
        envelope,
        torch.ones(2, 1, dtype=torch.float64),
        receivers,
        torch.tensor([0, 2, 2]),
        2,
        0.1,
        lmax=0,
        mmax=0,
    )
    shifted = torch.tensor([2 / 3, 1.0], dtype=torch.float64)
    probability = shifted * envelope[:, 0] / (4 / 3 + 0.1)
    expected = (probability * envelope[:, 0] * values[:, 0, 0]).sum()
    torch.testing.assert_close(result[0, 0, 0], expected)
    assert result[1].count_nonzero() == 0


def test_raw_recombination_and_unused_stored_orders():
    dense = torch.eye(9, dtype=torch.float64)[None]
    compact = pack_by_degree(dense, 2, 2)
    zero = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)
    # Input plane 0: maps A/B, input plane 1: maps A/B.
    raw = torch.tensor(
        [[[10.0, 20.0, 1.0, 2.0], [30.0, 40.0, 3.0, 4.0]]], dtype=torch.float64
    )
    result = reduction_reference.aggregate_sum(
        (zero, raw),
        torch.tensor([2.0]),
        compact,
        torch.tensor([0]),
        2,
        lmax=2,
        mmax=1,
        raw_complex=True,
        rotation_storage="retained_rows",
        wigner_mmax=2,
    )
    expected = torch.tensor(
        [1.0, 31.0, 2.0, 7.0, 0.0, 42.0, 3.0, 16.0, 0.0], dtype=torch.float64
    )
    torch.testing.assert_close(result[0, :, 0], expected * 2)
    assert result[1].count_nonzero() == 0


def dataflow_case(
    kind,
    *,
    lmax=1,
    mmax=1,
    channels=2,
    raw=False,
    storage="retained_rows",
    sharing="degree",
    heads=1,
):
    """Return floating inputs and a direct public signature for both paths."""
    edges, receivers, row_ptr = receiver_graph()
    nodes, count, full = 4, 3, (lmax + 1) ** 2
    dense = tensor(count, full, full)
    compact = pack_by_degree(dense, lmax, mmax)
    rotation = dense if storage == "dense" else compact
    if kind == "node_to_edge":
        x = tensor(nodes, full, channels)
        groups = sum(lmax + 1 - m for m in range(mmax + 1))
        radial = (
            tensor(count, lmax + 1, 2 * channels)
            if sharing == "degree"
            else tensor(count, groups * 2 * channels)
        )

        def call(function, values):
            feature, radial, rotation = values
            index = edges.to(feature.device)
            return function(
                feature,
                index,
                radial,
                rotation,
                lmax=lmax,
                mmax=mmax,
                radial_sharing=sharing,
                rotation_storage=storage,
            )

        return (x, radial, rotation), call
    messages = tuple(
        tensor(count, lmax + 1, channels)
        if m == 0
        else tensor(count, 2, 2 * (lmax + 1 - m) * channels)
        if raw
        else tensor(count, 2 * (lmax + 1 - m), channels)
        for m in range(mmax + 1)
    )
    envelope = torch.rand(count, 1, dtype=torch.float64) + 0.5
    if kind == "aggregate_sum":

        def call(function, values):
            *messages, envelope, rotation = values
            return function(
                tuple(messages),
                envelope,
                rotation,
                receivers.to(rotation.device),
                nodes,
                row_ptr=row_ptr.to(device=rotation.device, dtype=torch.int32),
                lmax=lmax,
                mmax=mmax,
                raw_complex=raw,
                rotation_storage=storage,
            )

        return (*messages, envelope, rotation), call
    if kind == "aggregate_attention":
        components = sum(2 * min(degree, mmax) + 1 for degree in range(lmax + 1))
        values = messages if raw else (tensor(count, components, channels),)
        logits = tensor(count, heads)

        def call(function, inputs):
            *values, logits, envelope, rotation = inputs
            data = tuple(values) if raw else values[0]
            return function(
                data,
                logits,
                envelope,
                rotation,
                receivers.to(rotation.device),
                row_ptr.to(rotation.device),
                nodes,
                1e-10,
                lmax=lmax,
                mmax=mmax,
            )

        return (*values, logits, envelope, compact), call
    if kind == "scalar_edge_to_node":
        radial = tensor(count, lmax + 1, channels)

        def call(function, values):
            radial, envelope, rotation = values
            return function(
                radial,
                envelope,
                rotation,
                receivers.to(rotation.device),
                row_ptr.to(rotation.device),
                1.7,
                lmax=lmax,
                mmax=mmax,
            )

        return (radial, envelope, compact), call
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ["node_to_edge", "aggregate_sum", "aggregate_attention", "scalar_edge_to_node"],
)
def test_reference_fp64_first_and_second_derivatives(kind):
    torch.manual_seed(104)
    values, call = dataflow_case(
        kind, channels=1, raw=kind in ("aggregate_sum", "aggregate_attention")
    )
    values = tuple(value.requires_grad_() for value in values)

    def function(*values):
        return call(getattr(gather_equations if kind == "node_to_edge" else reduction_reference, kind), values)

    assert torch.autograd.gradcheck(function, values, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, values, fast_mode=True)


def differentiate(function, call, values, cotangent, directions):
    output = call(function, values)
    outputs = output if isinstance(output, tuple) else (output,)
    first = torch.autograd.grad(outputs, values, cotangent, create_graph=True)
    second = torch.autograd.grad(first, (*values, *cotangent), directions)
    return (*outputs, *first, *second)


GPU_CASES = [
    ("node_to_edge", dict(lmax=0, mmax=0, channels=3)),
    ("node_to_edge", dict(lmax=2, mmax=1, channels=3)),
    ("node_to_edge", dict(lmax=4, mmax=2, channels=7, sharing="order_degree")),
    (
        "node_to_edge",
        dict(lmax=2, mmax=2, channels=3, sharing="order_degree", storage="dense"),
    ),
    ("aggregate_sum", dict(lmax=0, mmax=0, channels=3)),
    ("aggregate_sum", dict(lmax=2, mmax=1, channels=3)),
    ("aggregate_sum", dict(lmax=2, mmax=2, channels=3, raw=True)),
    ("aggregate_sum", dict(lmax=3, mmax=2, channels=7, raw=True, storage="dense")),
    ("aggregate_attention", dict(lmax=0, mmax=0, channels=3)),
    ("aggregate_attention", dict(lmax=2, mmax=1, channels=3)),
    ("aggregate_attention", dict(lmax=2, mmax=2, channels=3, raw=True)),
    ("aggregate_attention", dict(lmax=2, mmax=1, channels=6, heads=2, raw=True)),
    ("scalar_edge_to_node", dict(lmax=0, mmax=0, channels=3)),
    ("scalar_edge_to_node", dict(lmax=3, mmax=2, channels=7)),
]


@pytest.mark.cuda
@pytest.mark.parametrize("kind,configuration", GPU_CASES)
def test_cuda_dataflows_against_independent_fp64_equations(kind, configuration):
    from LunaEquivariance.so2.rotate_gather.interface import gather_rotated_sectors as node_to_edge
    from LunaEquivariance.so2.inverse_rotate_reduce.interface import (
        sum_aggregation as aggregate_sum,
        attention_aggregation as aggregate_attention,
        scalar_edge_to_node,
    )

    optimized = dict(
        node_to_edge=node_to_edge,
        aggregate_sum=aggregate_sum,
        aggregate_attention=aggregate_attention,
        scalar_edge_to_node=scalar_edge_to_node,
    )[kind]
    torch.manual_seed(108)
    values, call = dataflow_case(kind, **configuration)
    expected_values = tuple(value.detach().requires_grad_() for value in values)
    actual_values = tuple(
        value.float().cuda().detach().requires_grad_() for value in values
    )
    sample = call(getattr(gather_equations if kind == "node_to_edge" else reduction_reference, kind), expected_values)
    sample = sample if isinstance(sample, tuple) else (sample,)
    cotangent = tuple(tensor(*value.shape, requires_grad=True) for value in sample)
    directions = tuple(tensor(*value.shape) for value in values)
    expected = differentiate(
        getattr(gather_equations if kind == "node_to_edge" else reduction_reference, kind), call, expected_values, cotangent, directions
    )
    actual = differentiate(
        optimized,
        call,
        actual_values,
        tuple(value.float().cuda().detach().requires_grad_() for value in cotangent),
        tuple(value.float().cuda() for value in directions),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value.double().cpu(), expected_value, atol=3e-5, rtol=3e-4
        )
