"""Node gathering, retained rotation and radial scaling."""

import sys
from types import SimpleNamespace

import pytest
import torch

from LunaEquivariance.so2.rotate_gather.interface import gather_rotated_sectors as node_to_edge
from LunaEquivariance.so2.rotate_gather import interface as prefix
from LunaEquivariance.so2.rotate_gather.interface import (
    compact_prefix,
    rotate_gather_order_degree,
)
from LunaEquivariance.so2.utils.graph import prefix_csr_metadata
from LunaEquivariance.so2.utils.layout import get_angular_layout, get_retained_layout


def inputs():
    return (
        torch.empty(3, 9, 32),
        torch.empty(2, 4, dtype=torch.int64),
        torch.empty(4, 1),
        torch.empty(4, 35),
    )


def test_degree_rejects_csr_metadata():
    with pytest.raises(ValueError, match="does not consume CSR metadata"):
        node_to_edge(
            *inputs(),
            lmax=2,
            mmax=2,
            radial_sharing="degree",
            rotation_storage="retained_rows",
            metadata=(),
        )


@pytest.mark.parametrize(
    "sharing,storage", [("degree", "dense"), ("invalid", "retained_rows")]
)
def test_unsupported_radial_storage_combination(sharing, storage):
    with pytest.raises(ValueError, match="unsupported radial sharing"):
        node_to_edge(
            *inputs(),
            lmax=2,
            mmax=2,
            radial_sharing=sharing,
            rotation_storage=storage,
        )


def test_public_degree_uses_default_launch(monkeypatch):
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(prefix, "rotate_gather_degree", dispatch)
    result = node_to_edge(
        *inputs(),
        lmax=2,
        mmax=2,
        radial_sharing="degree",
        rotation_storage="retained_rows",
    )
    assert result == "result"
    assert calls[0]["num_warps"] == 2


def test_order_degree_keeps_metadata_and_layout(monkeypatch):
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(prefix, "rotate_gather_order_degree", dispatch)
    metadata = object()
    result = node_to_edge(
        *inputs(),
        lmax=2,
        mmax=2,
        radial_sharing="order_degree",
        rotation_storage="retained_rows",
        metadata=metadata,
    )
    assert result == "result"
    assert calls[0]["metadata"] is metadata
    assert calls[0]["sectorized"] is True
    assert "num_warps" not in calls[0]


def test_sector_metadata_preserves_old_slots_and_adds_m4():
    from LunaEquivariance.so2.rotate_gather.kernels.utils import sector_metadata
    from LunaEquivariance.so2.utils.layout import get_angular_layout

    for lmax in (2, 3, 4):
        for mmax in range(lmax + 1):
            spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=128)
            starts, elements = sector_metadata(spec)
            assert len(starts) == mmax
            assert len(elements) == mmax + 1
            assert starts[:mmax] == spec.sector_row_offsets[1:]
            for order in range(mmax + 1):
                width = spec.sector_widths[order]
                assert elements[order] == width * (1 if order == 0 else 2) * 256
    spec = get_angular_layout(lmax=4, mmax=4, channels=128)
    assert sector_metadata(spec) == ((5, 13, 19, 23), (1280, 2048, 1536, 1024, 512))


@pytest.mark.cuda
@pytest.mark.parametrize("channels", [32, 64, 128, 256])
@pytest.mark.parametrize("mmax", [0, 1, 2])
def test_compact_prefix_backward(channels, mmax):
    from LunaEquivariance.so2.rotate_gather.interface import compact_prefix

    from LunaEquivariance.so2.utils.layout import get_angular_layout, get_retained_layout

    spec = get_angular_layout(lmax=2, mmax=mmax, channels=channels)
    retained = get_retained_layout(lmax=2, mmax=mmax, channels=channels)
    edges, nodes = 17, 5
    x = torch.randn(nodes, 9, channels, device="cuda", requires_grad=True)
    edge_index = torch.randint(nodes, (2, edges), device="cuda")
    radial_rows = sum(3 - order for order in range(mmax + 1))
    radial = torch.randn(
        edges, radial_rows * 2 * channels, device="cuda", requires_grad=True
    )
    wigner = torch.randn(
        edges, retained.compact_elements, device="cuda", requires_grad=True
    )
    gathered = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
    rotated_rows = []
    for row, degree in enumerate(spec.reduced_degrees):
        width = 2 * degree + 1
        offset = retained.compact_row_offsets[row]
        rotation_row = wigner[:, offset : offset + width]
        value = torch.einsum(
            "ej,ejc->ec", rotation_row, gathered[:, degree**2 : (degree + 1) ** 2]
        )
        radial_offset = spec.radial_row_offsets[row]
        rotated_rows.append(
            value * radial[:, radial_offset : radial_offset + 2 * channels]
        )
    rotated = torch.stack(rotated_rows, dim=1)
    reference = []
    for order, width in enumerate(spec.sector_widths):
        offset = spec.sector_row_offsets[order]
        rows = width if order == 0 else 2 * width
        reference.append(rotated[:, offset : offset + rows].reshape(edges, -1))
    actual = compact_prefix(
        x, edge_index, radial, wigner, lmax=2, mmax=mmax, channels=channels
    )
    cotangents = [torch.randn_like(value) for value in reference]
    expected_grads = torch.autograd.grad(reference, (x, radial, wigner), cotangents)
    actual_grads = torch.autograd.grad(
        actual,
        (x, radial, wigner),
        [g.reshape_as(a) for g, a in zip(cotangents, actual)],
    )
    for expected, observed in zip(expected_grads, actual_grads):
        relative = (expected - observed).double().norm() / expected.double().norm()
        assert torch.isfinite(observed).all()
        assert relative < 1e-5


@pytest.mark.parametrize(
    "sectorized,expected", [(False, "split_csr"), (True, "fused_atomic")]
)
@pytest.mark.parametrize("schedule", [None, "split_csr", "fused_csr", "fused_atomic"])
def test_layout_can_change_without_changing_derivative_policy(
    monkeypatch, sectorized, expected, schedule
):
    x = torch.empty(3, 9, 32)
    edges = torch.empty(2, 4, dtype=torch.int64)
    metadata = tuple(torch.empty(4, dtype=torch.int32) for _ in range(3))
    calls = []
    monkeypatch.setattr(prefix.CSRPrefix, "apply", lambda *args: calls.append(args))
    prefix.rotate_gather_order_degree(
        x,
        edges,
        torch.empty(4, 1),
        torch.empty(4, 35),
        lmax=2,
        mmax=2,
        metadata=metadata,
        sectorized=sectorized,
        derivative_schedule=schedule,
    )
    assert calls[0][-2:] == (sectorized, expected if schedule is None else schedule)


def test_unknown_schedule_is_rejected_before_building_metadata():
    with pytest.raises(ValueError, match="unsupported prefix derivative schedule"):
        prefix.rotate_gather_order_degree(
            None, None, None, None, lmax=2, mmax=2, derivative_schedule="invalid"
        )


@pytest.mark.parametrize("schedule", ["split_csr", "fused_csr", "fused_atomic"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("channels", [32, 128])
@pytest.mark.parametrize(
    "grad_enabled,deterministic", [(False, False), (False, True), (True, False), (True, True)]
)
def test_derivative_dispatch_preserves_csr_guards(
    monkeypatch, schedule, compact, channels, grad_enabled, deterministic
):
    x, radial = torch.empty(3, 9, channels), torch.empty(4, 1)
    wigner = torch.empty((4, 35) if compact else (4, 9, 9))
    ctx = SimpleNamespace(
        saved_tensors=(x, torch.empty(2, 4), radial, wigner, None, None, None),
        dimensions=(2, 2, schedule),
    )
    calls = []
    gradients = (x, radial, wigner)

    def csr(*args, **kwargs):
        calls.append(("csr", kwargs["fused_vjp"], kwargs.get("atomic_input", False)))
        return gradients

    def atomic(*args, **kwargs):
        calls.append(("compact", True, True))
        return gradients

    def differentiable(*args):
        calls.append(("differentiable", args[9], args[10]))
        return gradients

    monkeypatch.setattr(prefix, "gather_backward_source_csr", csr)
    monkeypatch.setattr(prefix, "gather_backward_atomic", atomic)
    monkeypatch.setattr(prefix.CSRPrefixBackward, "apply", differentiable)
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: deterministic
    )
    with torch.set_grad_enabled(grad_enabled):
        result = prefix.CSRPrefix.backward(ctx, torch.ones(4, 1))
    if grad_enabled or deterministic:
        atomic = compact and schedule == "fused_atomic" and channels == 128 and not deterministic
        expected = ("differentiable", schedule != "split_csr", atomic)
    elif compact and schedule == "fused_atomic":
        expected = ("compact", True, True)
    else:
        expected = ("csr", schedule != "split_csr", schedule == "fused_atomic")
    assert calls == [expected]
    assert len(result) == 11
    assert result[0] is x and result[2] is radial and result[3] is wigner
    assert all(result[index] is None for index in (1, 4, 5, 6, 7, 8, 9, 10))


@pytest.mark.parametrize("lmax,mmax,channels", [(2, 2, 32), (5, 5, 7)])
def test_flat_prefix_first_call_compile(monkeypatch, lmax, mmax, channels):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from LunaEquivariance.so2.utils.layout import get_angular_layout, get_retained_layout
    from LunaEquivariance.so2.rotate_gather import interface as mod

    class NoLaunch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(mod, "wrap_triton", lambda kernel: NoLaunch())
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)

    def forward(x, indices, radial, rotation):
        return mod.flat_prefix_forward(
            x,
            indices,
            radial,
            rotation,
            lmax=lmax,
            mmax=mmax,
            block_channels=128,
            num_warps=1,
        )

    torch._dynamo.reset()
    try:
        with FakeTensorMode():
            inputs = (
                torch.empty(4, spec.full_components, channels, device="cuda"),
                torch.empty(2, 8, device="cuda", dtype=torch.int64),
                torch.empty(8, spec.radial_elements, device="cuda"),
                torch.empty(8, retained.compact_elements, device="cuda"),
            )
            output = torch.compile(forward, fullgraph=True, backend="eager")(*inputs)
            assert output.shape == (8, spec.reduced_components, 2 * channels)
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
def test_compiled_prefix_metadata_preserves_receiver_storage_offset():
    torch = pytest.importorskip("torch")
    from LunaEquivariance.so2.utils.graph import prefix_csr_metadata

    nodes = 9
    compiled = torch.compile(
        lambda edges: prefix_csr_metadata(edges, nodes), dynamic=True
    )
    for receiver_values in (
        [0, 0, 1, 1, 1, 3, 4, 4, 6, 8, 8, 8, 8],
        [0, 0, 0, 2, 2, 2, 2, 3, 5, 5, 5, 6, 6, 6, 7, 7, 8],
    ):
        receivers = torch.tensor(receiver_values, device="cuda", dtype=torch.int64)
        sources = (torch.arange(receivers.numel(), device="cuda") * 5 + 3) % nodes
        edges = torch.stack((sources, receivers))
        # The receiver row is contiguous but starts after the source row.
        assert edges[1].storage_offset() == receivers.numel()
        expected = torch.cat(
            (
                receivers.new_zeros(1),
                torch.bincount(receivers, minlength=nodes).cumsum(0),
            )
        ).to(torch.int32)
        eager_receiver = prefix_csr_metadata(edges, nodes)[2]
        compiled_receiver = compiled(edges)[2]
        torch.testing.assert_close(eager_receiver, expected, rtol=0, atol=0)
        torch.testing.assert_close(compiled_receiver, expected, rtol=0, atol=0)


def torch_reference(
    x, edges, radial, rotations, storage, spec, retained, radial_sharing
):
    channels = x.shape[-1]
    endpoints = torch.cat((x[edges[0]], x[edges[1]]), dim=-1)
    rows = []
    for row, (degree, dense_row) in enumerate(
        zip(spec.reduced_degrees, spec.dense_rows)
    ):
        width = 2 * degree + 1
        if storage == "dense":
            rotation = rotations[:, dense_row, degree**2 : (degree + 1) ** 2]
        else:
            offset = retained.compact_row_offsets[row]
            rotation = rotations[:, offset : offset + width]
        value = torch.einsum(
            "ej,ejc->ec", rotation, endpoints[:, degree**2 : (degree + 1) ** 2]
        )
        if radial_sharing == "degree":
            factor = radial[:, degree]
        else:
            offset = spec.radial_row_offsets[row]
            factor = radial[:, offset : offset + 2 * channels]
        rows.append(value * factor)
    matrix = torch.stack(rows, dim=1)
    outputs = [matrix[:, : spec.lmax + 1].flatten(1)]
    for width, offset in zip(spec.sector_widths[1:], spec.sector_row_offsets[1:]):
        outputs.append(
            matrix[:, offset : offset + 2 * width].reshape(edges.shape[1], 2, -1)
        )
    return tuple(outputs)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "storage,radial_sharing,sectorized",
    [
        ("dense", "order_degree", True),
        ("retained_rows", "degree", True),
        ("retained_rows", "order_degree", True),
        ("retained_rows", "order_degree", False),
    ],
)
@pytest.mark.parametrize(
    "lmax,mmax,channels", [(2, 2, 32), (4, 2, 32), (4, 4, 32), (2, 2, 128)]
)
def test_prefix_all_input_derivatives(
    storage, radial_sharing, sectorized, lmax, mmax, channels
):
    check_prefix_derivatives(storage, radial_sharing, sectorized, lmax, mmax, channels)


@pytest.mark.cuda
@pytest.mark.parametrize("edge_count", [1024, 8193])
def test_order_degree_gather_edgewise_derivatives(edge_count):
    check_prefix_derivatives(
        "retained_rows", "order_degree", True, 2, 2, 128, edge_count=edge_count,
    )


@pytest.mark.cuda
def test_degree_shared_gather_strided_directions(monkeypatch):
    if (not torch.cuda.is_available() or prefix.degree_shared_double_backward is None
            or torch.cuda.get_device_capability() not in ((8, 0), (9, 0))):
        pytest.skip("degree-shared HVP schedule requires A100 or H100")
    monkeypatch.setattr(prefix, "use_degree_shared_gather", lambda *_: True)
    check_prefix_derivatives("retained_rows", "degree", True, 4, 2, 128, strided_directions=True)


def test_degree_shared_gather_checks_shape_before_device(monkeypatch):
    monkeypatch.setattr(prefix, "degree_shared_double_backward", object())
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    device = object()
    x = SimpleNamespace(is_cuda=True, device=device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    assert prefix.use_degree_shared_gather(x, 17940, 4, 2, 128)
    assert not prefix.use_degree_shared_gather(None, 0, 4, 2, 128)
    assert not prefix.use_degree_shared_gather(None, 17940, 4, 2, 64)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 9))
    assert not prefix.use_degree_shared_gather(x, 17940, 4, 2, 128)


def check_prefix_derivatives(
    storage, radial_sharing, sectorized, lmax, mmax, channels, operation=None,
    *, strided_directions=False, edge_count=11,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator(device="cuda").manual_seed(42)
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)

    def rand(shape):
        return (
            torch.randn(shape, generator=generator, device="cuda") * 0.2
        ).requires_grad_()

    nodes = max(7, edge_count // 64)
    edges = torch.randint(nodes, (2, edge_count), generator=generator, device="cuda")
    # CSR receiver reduction requires edges grouped by receiving node.
    edges = edges[:, edges[1].argsort(stable=True)].contiguous()
    radial_shape = (
        (edge_count, lmax + 1, 2 * channels)
        if radial_sharing == "degree"
        else (edge_count, spec.radial_elements)
    )
    rotation_shape = {
        "dense": (edge_count, spec.full_components, spec.full_components),
        "retained_rows": (edge_count, retained.compact_elements),
    }[storage]
    inputs = (
        rand((nodes, spec.full_components, channels)),
        rand(radial_shape),
        rand(rotation_shape),
    )
    refs = tuple(x.detach().double().requires_grad_() for x in inputs)
    kwargs = dict(lmax=lmax, mmax=mmax)
    if operation is not None:
        kwargs.update(
            rotation_storage=storage,
            radial_sharing=radial_sharing,
            sectorized=sectorized,
        )
    elif sectorized:
        operation = node_to_edge
        kwargs.update(rotation_storage=storage, radial_sharing=radial_sharing)
    else:
        assert radial_sharing == "order_degree"
        operation = rotate_gather_order_degree
        kwargs["sectorized"] = False
    actual = operation(inputs[0], edges, *inputs[1:], **kwargs)
    if strided_directions:
        assert all(value.is_contiguous() for value in actual)
    expected = torch_reference(
        refs[0], edges, *refs[1:], storage, spec, retained, radial_sharing
    )
    # A nonlinear scalar loss tests the operator Hessian and incoming gradient
    # dependence together, rather than only a constant-cotangent local VJP.
    if strided_directions:
        directions = tuple(
            x.new_full((*x.shape[:-1], 2 * x.shape[-1]), 0.1)[..., ::2]
            for x in inputs
        )
    else:
        directions = tuple(torch.ones_like(x) * 0.1 for x in inputs)

    def derivatives(outputs, values):
        loss = sum((o.sin() + o.square() * 0.3).sum() for o in outputs)
        first = torch.autograd.grad(loss, values, create_graph=True)
        second = torch.autograd.grad(first, values, directions)
        return (*outputs, *first, *second)

    for value, reference in zip(
        derivatives(actual, inputs), derivatives(expected, refs)
    ):
        torch.testing.assert_close(value.double(), reference, rtol=2e-4, atol=2e-5)
    # A VJP without a higher-order graph can use a separate fused kernel.
    detached = tuple(value.detach().requires_grad_() for value in inputs)
    direct = operation(detached[0], edges, *detached[1:], **kwargs)
    reference_inputs = tuple(
        value.detach().double().requires_grad_() for value in inputs
    )
    reference_outputs = torch_reference(
        reference_inputs[0],
        edges,
        *reference_inputs[1:],
        storage,
        spec,
        retained,
        radial_sharing,
    )
    first = torch.autograd.grad(sum(value.square().sum() for value in direct), detached)
    reference_first = torch.autograd.grad(
        sum(value.square().sum() for value in reference_outputs), reference_inputs
    )
    for value, reference in zip(first, reference_first):
        torch.testing.assert_close(value.double(), reference, rtol=2e-4, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("sectorized", [False, True])
@pytest.mark.parametrize("schedule", ["split_csr", "fused_csr", "fused_atomic"])
def test_output_layout_independent_of_derivative_schedule(sectorized, schedule):
    from LunaEquivariance.so2.rotate_gather.interface import rotate_gather_order_degree

    def operation(
        x, edges, radial, rotations, *, radial_sharing, rotation_storage, **kwargs
    ):
        assert radial_sharing == "order_degree" and rotation_storage == "retained_rows"
        return rotate_gather_order_degree(
            x,
            edges,
            radial,
            rotations,
            derivative_schedule=schedule,
            **kwargs,
        )

    check_prefix_derivatives(
        "retained_rows", "order_degree", sectorized, 3, 2, 128, operation
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "storage,sharing,sectorized",
    [
        ("retained_rows", "degree", True),
        ("retained_rows", "order_degree", True),
        ("dense", "order_degree", True),
        ("retained_rows", "order_degree", False),
    ],
)
@pytest.mark.parametrize(
    "lmax,mmax,channels", [(0, 0, 7), (3, 0, 33), (5, 5, 7), (3, 2, 257)]
)
def test_prefix_dimensions(storage, sharing, sectorized, lmax, mmax, channels):
    check_prefix_derivatives(storage, sharing, sectorized, lmax, mmax, channels)


@pytest.mark.cuda
@pytest.mark.parametrize("sharing", ["degree", "order_degree"])
def test_empty_prefix_dimensions(sharing):
    import torch
    from LunaEquivariance.so2.rotate_gather.interface import gather_rotated_sectors as node_to_edge
    from LunaEquivariance.so2.utils.layout import get_angular_layout, get_retained_layout

    lmax, mmax, channels = 3, 3, 7
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    x = torch.randn(3, 16, channels, device="cuda", requires_grad=True)
    radial = torch.empty(
        (0, 4, 2 * channels) if sharing == "degree" else (0, spec.radial_elements),
        device="cuda",
        requires_grad=True,
    )
    rotation = torch.empty(
        0, retained.compact_elements, device="cuda", requires_grad=True
    )
    edge_index = torch.empty(2, 0, device="cuda", dtype=torch.int64)
    values = (x, radial, rotation)
    outputs = node_to_edge(
        x,
        edge_index,
        radial,
        rotation,
        lmax=lmax,
        mmax=mmax,
        radial_sharing=sharing,
        rotation_storage="retained_rows",
    )
    assert all(value.shape[0] == 0 for value in outputs)
    first = torch.autograd.grad(
        sum(value.square().sum() for value in outputs), values, create_graph=True
    )
    second = torch.autograd.grad(sum(value.sum() for value in first), values)
    for value, grad, grad2 in zip(values, first, second):
        assert value.shape == grad.shape == grad2.shape
        assert torch.count_nonzero(grad) == torch.count_nonzero(grad2) == 0


def run(lmax, mmax, device, helpers):
    synthetic_graph, legal_wigner = helpers.synthetic_graph, helpers.legal_wigner
    prefix_outputs, comparison = helpers.prefix_outputs, helpers.comparison
    clone_leaves, as_tuple = helpers.clone_leaves, helpers.as_tuple
    torch.manual_seed(42)
    nodes, channels = 6, 128
    graph = synthetic_graph(nodes, 3, seed=42, device=device)
    edges = graph.shape[1]
    spec = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
    dense = legal_wigner(edges, lmax, seed=42, device=device)
    retained = get_retained_layout(lmax=lmax, mmax=mmax, channels=channels)
    compact = torch.cat(
        [
            dense[:, spec.dense_rows[row], degree**2 : (degree + 1) ** 2]
            for degree, degree_rows in enumerate(retained.degree_rows)
            for row in degree_rows
        ],
        dim=1,
    ).contiguous()
    values = (
        torch.randn(nodes, (lmax + 1) ** 2, channels, device=device),
        torch.randn(edges, spec.radial_elements, device=device),
        compact,
    )
    metadata = prefix_csr_metadata(graph, nodes)

    def reference(x, radial, wigner):
        endpoints = torch.cat((x[graph[0]], x[graph[1]]), dim=-1)
        rotated = []
        for row, degree in enumerate(spec.reduced_degrees):
            start = retained.compact_row_offsets[row]
            width = 2 * degree + 1
            rotated.append(
                torch.einsum(
                    "ej,ejc->ec",
                    wigner[:, start : start + width],
                    endpoints[:, degree**2 : (degree + 1) ** 2],
                )
            )
        rotated = torch.stack(rotated, dim=1)
        scale = torch.stack(
            [
                radial[:, offset : offset + 2 * channels]
                for offset in spec.radial_row_offsets
            ],
            dim=1,
        )
        return prefix_outputs(rotated * scale, spec)

    def csr(x, radial, wigner):
        return rotate_gather_order_degree(
            x,
            graph,
            radial,
            wigner,
            lmax=lmax,
            mmax=mmax,
            metadata=metadata,
            sectorized=True,
        )

    def direct(x, radial, wigner):
        return compact_prefix(
            x, graph, radial, wigner, lmax=lmax, mmax=mmax, channels=channels
        )

    with torch.no_grad():
        cotangents = tuple(torch.randn_like(v) for v in reference(*values))
    directions = tuple(torch.randn_like(v) for v in values)

    def evaluate(fn, selected, second_index, need_second=True):
        x = clone_leaves(values, requires_grad=True)
        c = clone_leaves(cotangents, requires_grad=True)
        outputs = as_tuple(fn(*x))
        active = (
            range(len(c))
            if selected == "all"
            else (0,)
            if selected == "m0"
            else (mmax,)
        )
        objective = sum((outputs[i] * c[i]).sum() for i in active)
        gradients = torch.autograd.grad(objective, x, create_graph=need_second)
        result = [
            tuple(v.detach() for v in outputs),
            tuple(v.detach() for v in gradients),
        ]
        if need_second:
            slots = range(3) if second_index == "all" else (second_index,)
            contraction = sum((gradients[i] * directions[i]).sum() for i in slots)
            raw = torch.autograd.grad(contraction, x + c, allow_unused=True)
            result.append(
                tuple(
                    torch.zeros_like(v) if g is None else g.detach()
                    for g, v in zip(raw, x + c)
                )
            )
        return result

    results = {}
    for selected in ("all", "m0", "last"):
        for second_index in ("all", 0, 1, 2):
            expected = evaluate(reference, selected, second_index)
            observed = evaluate(csr, selected, second_index)
            errors = {
                stage: comparison(a, b)
                for stage, a, b in zip(
                    ("forward", "first", "second"), expected, observed
                )
            }
            results[f"csr-{selected}-second-{second_index}"] = {
                "status": "passed"
                if all(e["finite"] and e["relative_l2"] < 1e-4 for e in errors.values())
                else "failed",
                "errors": errors,
            }
        # The direct torch.library compact API is the existing terminal-D1 path.
        # Higher-order training uses the CSR autograd wrapper tested above.
        expected = evaluate(reference, selected, "all", need_second=False)
        observed = evaluate(direct, selected, "all", need_second=False)
        errors = {
            stage: comparison(a, b)
            for stage, a, b in zip(("forward", "first"), expected, observed)
        }
        results[f"direct-{selected}"] = {
            "status": "passed"
            if all(e["finite"] and e["relative_l2"] < 1e-4 for e in errors.values())
            else "failed",
            "errors": errors,
        }
    return {
        "configuration": {
            "lmax": lmax,
            "mmax": mmax,
            "channels": channels,
            "nodes": nodes,
            "edges": edges,
        },
        "cases": results,
        "status": "passed"
        if all(r["status"] == "passed" for r in results.values())
        else "failed",
    }


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (4, 2), (4, 4)])
def test_partial_output_and_derivative_slots(lmax, mmax, dataflow_reference):
    assert (
        run(lmax, mmax, torch.device("cuda"), dataflow_reference)["status"] == "passed"
    )
