"""Wigner basis, rotation storage and geometry derivatives."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module

import pytest
import torch
import triton
from e3nn import o3
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from LunaEquivariance.so2 import rotation
from LunaEquivariance.so2.utils.layout import (
    AngularLayout,
    RetainedRotationLayout,
    get_angular_layout,
    get_retained_layout,
)
from LunaEquivariance.so2.inverse_rotate_reduce import interface as tail
from LunaEquivariance.so2.rotation import interface as edge_quaternion
from LunaEquivariance.so2.rotation.interface import (
    pack_inverse_rotations,
    pack_retained_rotations,
    retained_wigner_from_frames,
)
from LunaEquivariance.so2.rotation.kernels.utils import (
    degree_basis,
    device_degree_basis,
    standard_basis,
)
from LunaEquivariance.so2.rotation import interface as wigner
from LunaEquivariance.so2.rotation.kernels import utils as wigner_utils
from LunaEquivariance.so2.rotation.interface import compact_inverse_rescale
from LunaEquivariance.so2.rotation import interface as retained_rows
from LunaEquivariance.so2.rotation.kernels import utils as retained_rows_utils
from LunaEquivariance.so2.rotation.kernels import forward as retained_rows_forward
from LunaEquivariance.so2.rotation.kernels import backward as retained_rows_backward
from LunaEquivariance.so2.rotation.kernels import double_backward as retained_rows_double_backward
from tests.so2.reference import rotation as rotation_reference
from tests.so2.reference.rotation import rotation as reference_rotation, z_rotation
from tests.testing_utils import assert_reference_close
from tests.so2.checks import make_cotangent, output_derivatives


@contextmanager
def fp64_reference():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("lmax", [0, 1, 4, 8])
def test_standard_basis_matches_e3nn(lmax):
    matrices = standard_basis(lmax, device="cpu", dtype=torch.float64)
    with fp64_reference():
        permutation = torch.tensor([[0., -1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        for degree, matrix in enumerate(matrices):
            expected = o3.Irrep(degree, (-1)**degree).D_from_matrix(permutation)
            torch.testing.assert_close(matrix, expected, atol=2e-13, rtol=2e-13)
            identity = torch.eye(2 * degree + 1)
            torch.testing.assert_close(matrix @ matrix, identity, atol=2e-13, rtol=2e-13)
            assert matrix.is_contiguous() and not matrix.requires_grad


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 7])
def test_standard_basis_euler_factorization(degree):
    matrix = standard_basis(degree, dtype=torch.float64)[-1]
    with fp64_reference():
        alpha = torch.tensor([-1.1, 0., 2.4])
        beta = torch.tensor([0.6, 1.8, -0.4])
        gamma = torch.tensor([0.3, -0.9, 1.3])
        actual = (z_rotation(alpha, degree) @ matrix @ z_rotation(beta, degree)
                  @ matrix @ z_rotation(gamma, degree))
        expected = o3.wigner_D(degree, alpha, beta, gamma)
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_standard_basis_preserves_default_and_reuses_constants(dtype, monkeypatch):
    previous = torch.get_default_dtype()
    degree_basis.cache_clear()
    device_degree_basis.cache_clear()
    # A production call must not modify global dtype, even temporarily.
    def forbidden(*args, **kwargs):
        raise AssertionError("basis setup changed the default dtype")
    monkeypatch.setattr(torch, "set_default_dtype", forbidden)
    first = standard_basis(6, dtype=dtype)
    second = standard_basis(6, device=torch.device("cpu"), dtype=dtype)
    assert torch.get_default_dtype() == previous
    for left, right in zip(first, second, strict=True):
        assert left.dtype == dtype
        assert left.data_ptr() == right.data_ptr()


def test_cached_constants_are_not_inference_tensors():
    with torch.inference_mode():
        matrices = standard_basis(9, dtype=torch.float64)
    assert all(not matrix.is_inference() for matrix in matrices)
    inputs = torch.randn(2, 19, dtype=torch.float64, requires_grad=True)
    result = inputs @ matrices[-1]
    torch.autograd.grad(result.sum(), inputs)


@pytest.mark.parametrize("lmax,dtype,error", [
    (-1, torch.float32, ValueError),
    (1.5, torch.float32, TypeError),
    (True, torch.float32, TypeError),
    (2, torch.float16, TypeError),
])
def test_standard_basis_rejects_invalid_settings(lmax,dtype,error):
    with pytest.raises(error):
        standard_basis(lmax,dtype=dtype)


@pytest.mark.parametrize("arch", [80, 90], ids=["a100", "h100"])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize(
    "kernel",
    [
        retained_rows_forward.retained_forward_kernel,
        retained_rows_backward.retained_backward_kernel,
        retained_rows_double_backward.retained_hvp_kernel,
    ],
    ids=["forward", "backward", "double-backward"],
)
def test_retained_rotation_compiles(kernel, arch, mmax, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    constants = retained_rows_utils.launch_constants(mmax)
    signature = {
        name: "constexpr" if name in constants else "*fp32" for name in kernel.arg_names
    }
    compiled = triton.compile(
        ASTSource(kernel, signature, constants),
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": 4},
    )
    assert compiled.name == kernel.__name__
    assert compiled.asm["cubin"]


from LunaEquivariance.so2.rotation import interface as monomials


from LunaEquivariance.so2.rotation import interface as retained
from LunaEquivariance.so2.rotation.kernels import utils as retained_utils


@pytest.mark.parametrize("kind", ["edge", "monomial", "retained"])
@pytest.mark.parametrize("compile_visible", [False, True])
def test_backward_keeps_its_forward_dispatch(monkeypatch, kind, compile_visible):
    function, extras, prefix = {
        "edge": (edge_quaternion.EdgeQuaternion, (torch.zeros(2),), "edge_quaternion"),
        "monomial": (monomials.QuaternionMonomials, (torch.zeros(1, 4),), "monomials"),
        "retained": (retained.RetainedQuaternion, (torch.zeros(1), torch.zeros(1), 2),
                     "retained_rows"),
    }[kind]
    calls = []

    def forward(q, *args):
        calls.append(("forward", args[-2] if kind == "monomial" else args[-1]))
        return q.square()

    def vjp(gradient, q, *args):
        calls.append(("vjp", args[-2] if kind == "monomial" else args[-1]))
        return 2 * gradient * q

    def hvp(gradient, q, *args):
        calls.append(("hvp", args[-2] if kind == "monomial" else args[-1]))
        direction = args[1] if kind == "edge" else args[0]
        return 2 * direction * q, 2 * direction * gradient

    monkeypatch.setattr(wigner, f"{prefix}_forward", forward)
    monkeypatch.setattr(wigner, f"{prefix}_backward", vjp)
    second_pass = "monomials_double_backward" if kind == "monomial" else f"{prefix}_hvp"
    monkeypatch.setattr(wigner, second_pass, hvp)
    q = torch.randn(2, 4, requires_grad=True)
    cotangent = torch.randn_like(q, requires_grad=True)
    direction = torch.randn_like(q)
    tail = (12,) if kind == "monomial" else ()
    output = function.apply(q, *extras, compile_visible, *tail)
    function.apply(q, *extras, not compile_visible, *tail)
    (first,) = torch.autograd.grad(output, q, cotangent, create_graph=True)
    second_q, second_cotangent = torch.autograd.grad(first, (q, cotangent), direction)
    torch.testing.assert_close(first, 2 * cotangent * q)
    torch.testing.assert_close(second_q, 2 * direction * cotangent)
    torch.testing.assert_close(second_cotangent, 2 * direction * q)
    assert calls == [
        ("forward", compile_visible),
        ("forward", not compile_visible),
        ("vjp", compile_visible),
        ("hvp", compile_visible),
    ]


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(0, 0), (1, 0), (1, 1), (3, 3), (5, 4), (7, 7)])
def test_arbitrary_degree_assembly_and_derivatives(monkeypatch, lmax, mmax):
    from LunaEquivariance.so2.rotation import interface as edge_quaternion
    from LunaEquivariance.so2.rotation import interface as polynomial
    from LunaEquivariance.so2.rotation import interface as retained_rows
    from LunaEquivariance.so2.rotation.kernels import utils as retained_rows_utils
    from LunaEquivariance.so2.rotation.kernels import utils as polynomial_utils
    from LunaEquivariance.so2.rotation.kernels.utils import monomial_exponents

    torch.manual_seed(2026)
    monkeypatch.setattr(
        edge_quaternion, "fused_edge_quaternion", lambda q, gamma, **kwargs: q
    )
    tables = []
    for degree in range(1, lmax + 1):
        powers = (
            torch.tensor(
                polynomial_utils.QUADRATIC_EXPONENTS, dtype=torch.int32, device="cuda"
            )
            if degree == 1
            else monomial_exponents(2 * degree, torch.device("cuda"))
        )
        full = torch.randn((2 * degree + 1) ** 2, powers.shape[0], device="cuda") * 0.1
        selected = retained_rows_utils.select_retained_coefficients(
            full, degrees=(degree,), mmax=mmax
        )
        tables.append((powers, selected))
    quadratic = tables[0][1] if lmax else torch.empty(0, device="cuda")
    quartic = tables[1][1] if lmax >= 2 else torch.empty(0, device="cuda")
    q = torch.nn.functional.normalize(
        torch.randn(3, 4, device="cuda"), dim=1
    ).requires_grad_()
    rq = q.detach().double().requires_grad_()
    actual = retained_rows.quaternion_wigner_from_edges_retained(
        q,
        quadratic,
        quartic,
        lmax=lmax,
        mmax=mmax,
        polynomial_blocks=tuple(
            retained_rows_utils.PolynomialBlock((degree,), powers, table)
            for degree, (powers, table) in enumerate(tables[2:], 3)
        ),
    )
    expected = torch.cat(
        [
            rq[:, :1].square() * 0 + 1,
            *(
                (rq[:, None] ** powers[None].double()).prod(-1) @ table.double().T
                for powers, table in tables
            ),
        ],
        dim=1,
    )
    cotangent = torch.randn_like(actual, requires_grad=True)
    rc = cotangent.detach().double().requires_grad_()
    direction = torch.randn_like(q)
    (first,) = torch.autograd.grad(actual, q, cotangent, create_graph=True)
    (ref_first,) = torch.autograd.grad(expected, rq, rc, create_graph=True)
    second = torch.autograd.grad(first, (q, cotangent), direction)
    ref_second = torch.autograd.grad(ref_first, (rq, rc), direction.double())
    for value, target in zip(
        (actual, first, *second), (expected, ref_first, *ref_second)
    ):
        torch.testing.assert_close(value.double(), target, atol=1e-4, rtol=6e-5)


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (1, 0), (1, 1), (7, 7), (8, 5)])
def test_retained_dimensions_cover_each_degree(lmax, mmax):
    from LunaEquivariance.so2.rotation.kernels.utils import select_retained_coefficients
    from LunaEquivariance.so2.utils.layout import get_retained_layout

    count = 1
    for degree in range(1, lmax + 1):
        width = 2 * degree + 1
        original = torch.arange(width * width).reshape(width * width, 1)
        selected = select_retained_coefficients(original, degrees=(degree,), mmax=mmax)
        order = [
            degree,
            *(
                v
                for m in range(1, min(degree, mmax) + 1)
                for v in (degree + m, degree - m)
            ),
        ]
        torch.testing.assert_close(
            selected, original.reshape(width, width)[order].reshape(-1, 1)
        )
        count += selected.numel()
    assert (
        count == get_retained_layout(lmax=lmax, mmax=mmax, channels=1).compact_elements
    )


def retained_euler_reference(angles, matrices, mmax, scale):
    def z(angle, degree):
        width = 2 * degree + 1
        out = angle.new_zeros((angle.numel(), width, width))
        for row in range(width):
            phase = (degree - row) * angle
            out[:, row, width - row - 1] = phase.sin()
            out[:, row, row] = phase.cos()
        return out

    rows = wigner_utils.selected_local_rows(len(matrices) - 1, mmax)
    blocks = tuple(
        (z(angles[0], l) @ j @ z(angles[1], l) @ j @ z(angles[2], l))[:, selected]
        for l, (j, selected) in enumerate(zip(matrices, rows, strict=True))
    )
    direct = torch.cat(tuple(value.flatten(1) for value in blocks), dim=1)
    return (direct,) if scale is None else (direct, direct * scale)


def retained_derivatives(outputs, inputs, cotangents, directions):
    first = torch.autograd.grad(outputs, inputs, cotangents, create_graph=True)
    second = torch.autograd.grad(first, (*inputs, *cotangents), directions)
    return (*outputs, *first, *second)


def view(value, kind):
    if kind == "broadcast":
        return value.new_tensor(0.3).expand_as(value)
    backing = value.new_empty((*value.shape, 2))
    backing[..., 0] = value
    return backing[..., 0]


@pytest.mark.parametrize("pair", [False, True])
def test_retained_int32_offset_boundary_without_large_allocation(monkeypatch, pair):
    from types import SimpleNamespace

    spec = get_retained_layout(lmax=2, mmax=2, channels=1)
    maximum_edges = (1 << 31) // spec.compact_elements
    matrices = tuple(torch.empty(2 * l + 1, 2 * l + 1) for l in range(3))
    scale = torch.ones(spec.compact_elements)
    monkeypatch.setattr(wigner, "validate_inputs", lambda *args: ())
    calls = []
    sentinel = object()

    def apply(*args):
        calls.append(args)
        return (sentinel, sentinel) if pair else (sentinel,)

    monkeypatch.setattr(wigner.RetainedWigner, "apply", apply)

    def invoke(edges):
        angle = SimpleNamespace(
            shape=(edges,), device=torch.device("cpu"), dtype=torch.float32
        )
        if pair:
            return wigner.sparse_wigner_pair(
                angle, angle, angle, matrices, scale, mmax=2
            )
        return wigner.sparse_wigner(
            angle, angle, angle, matrices, mmax=2, storage="retained_rows"
        )

    invoke(maximum_edges)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="retained Wigner carrier exceeds.*32-bit"):
        invoke(maximum_edges + 1)
    assert len(calls) == 1


@pytest.mark.parametrize("pair", [False, True])
def test_retained_autograd_cpu_contract(monkeypatch, pair):
    spec = get_retained_layout(lmax=2, mmax=1, channels=1)
    torch.manual_seed(541)
    angles = tuple(
        torch.randn(3, dtype=torch.double).requires_grad_() for _ in range(3)
    )
    matrices = tuple(
        torch.randn(2 * l + 1, 2 * l + 1, dtype=torch.double) for l in range(3)
    )
    scale = torch.rand(spec.compact_elements, dtype=torch.double) if pair else None

    def forward(a, b, c, js, scale, spec):
        return retained_euler_reference((a, b, c), js, spec.mmax, scale)

    def backward(gd, gi, a, b, c, js, scale, spec):
        with torch.enable_grad():
            xs = tuple(x.detach().requires_grad_() for x in (a, b, c))
            return torch.autograd.grad(
                retained_euler_reference(xs, js, spec.mmax, scale),
                xs,
                (gd,) if gi is None else (gd, gi),
            )

    def hvp(gd, gi, a, b, c, ds, js, scale, spec):
        with torch.enable_grad():
            xs = tuple(x.detach().requires_grad_() for x in (a, b, c))
            gs = tuple(
                g.detach().requires_grad_() for g in ((gd,) if gi is None else (gd, gi))
            )
            first = torch.autograd.grad(
                retained_euler_reference(xs, js, spec.mmax, scale), xs, gs, create_graph=True
            )
            result = torch.autograd.grad(first, (*gs, *xs), ds)
            return (result[0], result[1] if pair else None), result[len(gs) :]

    monkeypatch.setattr(wigner, "retained_forward", forward)
    monkeypatch.setattr(wigner, "retained_backward", backward)
    monkeypatch.setattr(wigner, "retained_hvp", hvp)
    outputs = wigner.RetainedWigner.apply(*angles, scale, *matrices, spec)
    reference = retained_euler_reference(angles, matrices, spec.mmax, scale)
    gs = tuple(torch.randn_like(value).requires_grad_() for value in outputs)
    ds = tuple(torch.randn_like(value) for value in angles)
    for actual, expected in zip(
        retained_derivatives(outputs, angles, gs, ds),
        retained_derivatives(reference, angles, gs, ds),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 0), (2, 2), (3, 2), (4, 2), (6, 3)])
@pytest.mark.parametrize("pair", [False, True])
def test_retained_gpu_against_fp64_all_slots(lmax, mmax, pair):
    torch.manual_seed(1541 + lmax + mmax)
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=1)
    angles = tuple(torch.randn(4, device="cuda").requires_grad_() for _ in range(3))
    matrices = tuple(
        torch.randn(2 * l + 1, 2 * l + 1, device="cuda") / (2 * l + 1) ** 0.5
        for l in range(lmax + 1)
    )
    scale = torch.rand(spec.compact_elements, device="cuda") + 0.5 if pair else None
    if pair:
        outputs = wigner.sparse_wigner_pair(*angles, matrices, scale, mmax=mmax)
    else:
        outputs = (
            wigner.sparse_wigner(*angles, matrices, mmax=mmax, storage="retained_rows"),
        )
    ref_angles = tuple(x.detach().double().requires_grad_() for x in angles)
    reference = retained_euler_reference(
        ref_angles,
        tuple(j.double() for j in matrices),
        mmax,
        scale.double() if pair else None,
    )
    gs = tuple(view(torch.randn_like(x), "strided").requires_grad_() for x in outputs)
    ds = tuple(view(torch.randn_like(x), "strided") for x in angles)
    ref_gs = tuple(g.detach().double().requires_grad_() for g in gs)
    actual = retained_derivatives(outputs, angles, gs, ds)
    expected = retained_derivatives(reference, ref_angles, ref_gs, tuple(d.double() for d in ds))
    for slot, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(
            got.double(),
            want,
            rtol=4e-4,
            atol=2e-4,
            msg=lambda text: f"slot {slot}: {text}",
        )


@pytest.mark.cuda
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("active", ["direct", "inverse", "both"])
def test_retained_broadcast_gradients_and_unused_carrier(pair, active):
    if not pair and active != "direct":
        pytest.skip("single carrier")
    torch.manual_seed(1852)
    spec = get_retained_layout(lmax=4, mmax=2, channels=1)
    values = tuple(torch.randn(4, device="cuda") for _ in range(3))
    matrices = tuple(
        torch.randn(2 * l + 1, 2 * l + 1, device="cuda") / (2 * l + 1) ** 0.5
        for l in range(5)
    )
    scale = torch.rand(spec.compact_elements, device="cuda") + 0.5

    def evaluate(dense):
        angles = tuple(x.detach().requires_grad_() for x in values)
        outputs = (
            wigner.sparse_wigner_pair(*angles, matrices, scale, mmax=2)
            if pair
            else (
                wigner.sparse_wigner(
                    *angles, matrices, mmax=2, storage="retained_rows"
                ),
            )
        )
        selected = (
            outputs
            if active == "both"
            else outputs[1:]
            if active == "inverse"
            else outputs[:1]
        )
        gs = tuple(
            view(torch.empty_like(x), "broadcast").requires_grad_() for x in selected
        )
        ds = tuple(view(torch.empty_like(x), "broadcast") for x in angles)
        if dense:
            gs = tuple(g.detach().contiguous().requires_grad_() for g in gs)
            ds = tuple(d.contiguous() for d in ds)
        return retained_derivatives(selected, angles, gs, ds)

    for actual, expected in zip(evaluate(False), evaluate(True), strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("lmax,mmax", [(l, m) for l in range(7) for m in range(l + 1)])
def test_retained_layout_uses_common_sector_order(lmax, mmax):
    sectors = AngularLayout.create(lmax=lmax, mmax=mmax, channels=128)
    retained = RetainedRotationLayout.create(lmax=lmax, mmax=mmax, channels=128)
    for field in (
        "full_components",
        "reduced_components",
        "reduced_degrees",
        "sector_widths",
        "sector_row_offsets",
    ):
        assert getattr(retained, field) == getattr(sectors, field)

    # Every selected full-matrix row appears exactly once in degree storage,
    # in the 0,+1,-1,+2,-2 ordering used by the analytic Wigner producer.
    addresses = []
    base = 0
    for degree, rows in enumerate(retained.degree_rows):
        width = 2 * degree + 1
        signed_orders = (0,) + tuple(
            s * m for m in range(1, min(degree, mmax) + 1) for s in (1, -1)
        )
        assert (
            tuple(sectors.dense_rows[row] - degree * degree - degree for row in rows)
            == signed_orders
        )
        for local, row in enumerate(rows):
            offset = retained.compact_row_offsets[row]
            assert offset == base + local * width
            addresses.extend(range(offset, offset + width))
        base += len(rows) * width
    assert addresses == list(range(retained.compact_elements))
    assert retained.degree_row_counts == sectors.degree_reduced_counts
    assert retained.degree_row_offsets == sectors.degree_reduced_offsets
    assert retained.compact_elements <= sum(
        (2 * degree + 1) ** 2 for degree in range(lmax + 1)
    )


from LunaEquivariance.so2.rotation import interface as original
from LunaEquivariance.so2.rotation.kernels import utils as original_utils


get_retained_layout = import_module(
    "LunaEquivariance.so2.utils.layout"
).get_retained_layout


def quaternion_reference(q, quadratic, quartic):
    def evaluate(table, powers):
        powers = q.new_tensor(powers)
        monomials = (q[:, None, :] ** powers[None, :, :]).prod(-1)
        return monomials @ table.T

    return torch.cat(
        (
            q.new_ones((q.shape[0], 1)),
            evaluate(quadratic, original_utils.QUADRATIC_EXPONENTS),
            evaluate(quartic, original_utils.QUARTIC_EXPONENTS),
        ),
        dim=1,
    )


def make_tables(device):
    # The quartic table is deliberately general: cancellations or zero entries
    # in a particular rotation convention must not hide missing derivatives.
    generator = torch.Generator().manual_seed(891)
    return (
        original_utils.quadratic_coefficients(torch.device(device)),
        torch.randn(25, len(original_utils.QUARTIC_EXPONENTS), generator=generator).to(
            device
        ),
    )


@pytest.mark.parametrize("lmax,mmax", [(2, 0), (2, 1), (2, 2), (4, 2), (4, 4), (6, 3)])
def test_selected_coefficient_rows_match_common_layout(lmax, mmax):
    degrees = tuple(range(lmax + 1))
    size = sum((2 * degree + 1) ** 2 for degree in degrees)
    table = torch.arange(size * 2).reshape(size, 2)
    selected = retained_utils.select_retained_coefficients(table, degrees=degrees, mmax=mmax)
    expected_indices = []
    start = 0
    for degree in degrees:
        width = 2 * degree + 1
        rows = [degree]
        for order in range(1, min(degree, mmax) + 1):
            rows += [degree + order, degree - order]
        for row in rows:
            expected_indices.extend(
                range(start + row * width, start + (row + 1) * width)
            )
        start += width * width
    torch.testing.assert_close(selected, table[expected_indices])
    assert (
        selected.shape[0]
        == get_retained_layout(lmax=lmax, mmax=mmax, channels=1).compact_elements
    )


@pytest.mark.cuda
@pytest.mark.parametrize("mmax", [0, 1, 2])
@pytest.mark.parametrize("compile_visible", [False, True])
def test_retained_l2_forward_vjp_hvp(mmax, compile_visible):
    quadratic, quartic = make_tables("cuda")
    quadratic = retained_utils.select_retained_coefficients(
        quadratic, degrees=(1,), mmax=mmax
    )
    quartic = retained_utils.select_retained_coefficients(quartic, degrees=(2,), mmax=mmax)
    q = torch.randn(7, 4, device="cuda")
    q = torch.nn.functional.normalize(q, dim=1).requires_grad_()
    reference_q = q.detach().double().requires_grad_()
    actual = retained.retained_quaternion_l2(
        q, quadratic, quartic, mmax=mmax, compile_visible=compile_visible
    )
    expected = quaternion_reference(reference_q, quadratic.double(), quartic.double())
    cotangent = torch.randn_like(actual, requires_grad=True)
    reference_cotangent = cotangent.detach().double().requires_grad_()
    direction = torch.randn_like(q)
    (actual_first,) = torch.autograd.grad(actual, q, cotangent, create_graph=True)
    (expected_first,) = torch.autograd.grad(
        expected, reference_q, reference_cotangent, create_graph=True
    )
    actual_second = torch.autograd.grad(actual_first, (q, cotangent), direction)
    expected_second = torch.autograd.grad(
        expected_first, (reference_q, reference_cotangent), direction.double()
    )
    for value, target in zip(
        (actual, actual_first, *actual_second),
        (expected, expected_first, *expected_second),
        strict=True,
    ):
        torch.testing.assert_close(value.double(), target, atol=8e-5, rtol=5e-5)


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(3, 1), (4, 2), (4, 4), (6, 3)])
def test_high_degree_selected_polynomial_assembly(monkeypatch, lmax, mmax):
    from LunaEquivariance.so2.rotation import interface as edge_quaternion
    from LunaEquivariance.so2.rotation.kernels.utils import monomial_exponents

    # Isolate assembly and quaternion differentiation from the independently
    # tested edge-frame construction. Complete source tables are CPU reference
    # data, not edge-dependent rotation tensors.
    monkeypatch.setattr(
        edge_quaternion, "fused_edge_quaternion", lambda q, gamma, **kwargs: q
    )
    quadratic, quartic = make_tables("cuda")
    quadratic = retained_utils.select_retained_coefficients(
        quadratic, degrees=(1,), mmax=mmax
    )
    quartic = retained_utils.select_retained_coefficients(quartic, degrees=(2,), mmax=mmax)
    tables = []
    for degrees in ((3,) if lmax == 3 else (3, 4), *((d,) for d in range(5, lmax + 1))):
        powers = monomial_exponents(2 * max(degrees), torch.device("cuda"))
        full = (
            torch.randn(
                sum((2 * d + 1) ** 2 for d in degrees), powers.shape[0], device="cuda"
            )
            * 0.2
        )
        selected = retained_utils.select_retained_coefficients(
            full, degrees=degrees, mmax=mmax
        )
        tables.append(retained_utils.PolynomialBlock(degrees, powers, selected))
    q = torch.nn.functional.normalize(
        torch.randn(3, 4, device="cuda"), dim=1
    ).requires_grad_()
    ref_q = q.detach().double().requires_grad_()
    actual = retained.quaternion_wigner_from_edges_retained(
        q,
        quadratic,
        quartic,
        lmax=lmax,
        mmax=mmax,
        polynomial_blocks=tuple(tables),
    )
    expected = torch.cat(
        [
            quaternion_reference(ref_q, quadratic.double(), quartic.double()),
            *(
                (ref_q[:, None, :] ** powers.double()[None]).prod(-1)
                @ table.double().T
                for _, powers, table in tables
            ),
        ],
        dim=1,
    )
    cotangent = torch.randn_like(actual, requires_grad=True)
    ref_cotangent = cotangent.detach().double().requires_grad_()
    direction = torch.randn_like(q)
    (first,) = torch.autograd.grad(actual, q, cotangent, create_graph=True)
    (ref_first,) = torch.autograd.grad(
        expected, ref_q, ref_cotangent, create_graph=True
    )
    second = torch.autograd.grad(first, (q, cotangent), direction)
    ref_second = torch.autograd.grad(
        ref_first, (ref_q, ref_cotangent), direction.double()
    )
    for value, target in zip(
        (actual, first, *second), (expected, ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(value.double(), target, atol=1e-4, rtol=6e-5)


@pytest.mark.parametrize("mmax", [0, 1, 2])
def test_launch_constants_serialize_for_compilation_hook(mmax):
    import inspect
    import json
    from types import SimpleNamespace
    from triton.runtime.jit import serialize_specialization_data

    constants = retained_utils.launch_constants(mmax)
    arguments = dict(
        name="retained",
        signature={},
        constants={(index,): value for index, value in enumerate(constants.values())},
        attrs={},
        options=SimpleNamespace(num_warps=4),
        key="test",
    )
    if "target" in inspect.signature(serialize_specialization_data).parameters:
        arguments["target"] = SimpleNamespace(backend="cuda", arch=80, warp_size=32)
    serialized = serialize_specialization_data(**arguments)
    assert json.loads(serialized)["constant_vals"] == json.loads(
        json.dumps(list(constants.values()))
    )


def polynomial_block(degrees, *, mmax=2):
    rows = sum((2 * min(degree, mmax) + 1) * (2 * degree + 1) for degree in degrees)
    return retained_utils.PolynomialBlock(
        degrees, torch.zeros(1, 4, dtype=torch.int32), torch.zeros(rows, 1)
    )


@pytest.mark.parametrize(
    "groups,lmax",
    [
        ((), 3),
        (((4,),), 4),
        (((3,), (5,)), 5),
        (((3, 4), (4,)), 4),
        (((3, 4, 5),), 4),
        (((3, 5),), 5),
        (((2,),), 2),
    ],
)
def test_polynomial_blocks_require_complete_ordered_degrees(groups, lmax):
    blocks = tuple(polynomial_block(degrees) for degrees in groups)
    with pytest.raises(ValueError, match="polynomial block"):
        retained.quaternion_wigner_from_edges_retained(
            torch.zeros(2, 3),
            torch.empty(0),
            torch.empty(0),
            lmax=lmax,
            mmax=min(lmax, 2),
            polynomial_blocks=blocks,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "rows",
        "columns",
        "trainable",
        "dtype",
        "power_rank",
        "power_dtype",
        "empty_powers",
    ],
)
def test_polynomial_blocks_require_fixed_matching_tables(invalid):
    block = polynomial_block((3, 4))
    if invalid == "rows":
        block = block._replace(coefficients=block.coefficients[:-1])
    elif invalid == "columns":
        block = block._replace(coefficients=torch.zeros(block.coefficients.shape[0], 2))
    elif invalid == "trainable":
        block.coefficients.requires_grad_()
    elif invalid == "dtype":
        block = block._replace(coefficients=block.coefficients.double())
    elif invalid == "power_rank":
        block = block._replace(exponents=block.exponents.flatten())
    elif invalid == "power_dtype":
        block = block._replace(exponents=block.exponents.long())
    else:
        block = block._replace(exponents=block.exponents[:0])
    with pytest.raises(ValueError, match="polynomial (coefficients|powers)"):
        retained.quaternion_wigner_from_edges_retained(
            torch.zeros(2, 3),
            torch.empty(0),
            torch.empty(0),
            lmax=4,
            mmax=2,
            polynomial_blocks=(block,),
        )


def dense_euler_reference(angles, matrices_j, rows):
    def rotation(angle, degree):
        width = 2 * degree + 1
        matrix = angle.new_zeros((angle.numel(), width, width))
        for index, frequency in enumerate(range(degree, -degree - 1, -1)):
            matrix[:, index, width - 1 - index] = torch.sin(frequency * angle)
            matrix[:, index, index] = torch.cos(frequency * angle)
        return matrix

    blocks = tuple(
        (
            rotation(angles[0], degree)
            @ matrix_j
            @ rotation(angles[1], degree)
            @ matrix_j
            @ rotation(angles[2], degree)
        )[:, selected, :]
        for degree, (matrix_j, selected) in enumerate(
            zip(matrices_j, rows, strict=True)
        )
    )
    return blocks


def derivatives(outputs, angles, cotangents, directions):
    first = torch.autograd.grad(outputs, angles, cotangents, create_graph=True)
    second = torch.autograd.grad(first, (*angles, *cotangents), directions)
    return (*outputs, *first, *second)


def make_inputs(device, dtype, lmax=2):
    generator = torch.Generator(device=device).manual_seed(473)
    angles = tuple(
        torch.randn(3, device=device, dtype=dtype, generator=generator).requires_grad_()
        for _ in range(3)
    )
    matrices_j = tuple(
        torch.randn(
            2 * l + 1, 2 * l + 1, device=device, dtype=dtype, generator=generator
        )
        for l in range(lmax + 1)
    )
    return angles, matrices_j


@pytest.mark.parametrize("mmax", [0, 2])
def test_autograd_state_and_cotangent_contract(monkeypatch, mmax):
    # CPU substitutes only the launches. It exercises the actual shared custom
    # autograd state, including cotangents of every output in the second pass.
    def forward(a, b, c, js, rows):
        return dense_euler_reference((a, b, c), js, rows)

    def backward(gs, a, b, c, js, rows):
        with torch.enable_grad():
            angles = tuple(x.detach().requires_grad_() for x in (a, b, c))
            return torch.autograd.grad(dense_euler_reference(angles, js, rows), angles, gs)

    def hvp(gs, a, b, c, directions, js, rows):
        with torch.enable_grad():
            angles = tuple(x.detach().requires_grad_() for x in (a, b, c))
            gs = tuple(g.detach().requires_grad_() for g in gs)
            first = torch.autograd.grad(
                dense_euler_reference(angles, js, rows), angles, gs, create_graph=True
            )
            values = torch.autograd.grad(first, (*gs, *angles), directions)
            return values[:-3], values[-3:]

    monkeypatch.setattr(wigner, "wigner_forward", forward)
    monkeypatch.setattr(wigner, "wigner_backward", backward)
    monkeypatch.setattr(wigner, "wigner_hvp", hvp)
    angles, js = make_inputs("cpu", torch.float64)
    rows = wigner_utils.selected_local_rows(2, mmax)
    expected = dense_euler_reference(angles, js, rows)
    actual = wigner.Wigner.apply(*angles, *js, rows)
    cotangents = tuple(torch.ones_like(y, requires_grad=True) for y in expected)
    directions = tuple(torch.ones_like(a) for a in angles)
    references = derivatives(expected, angles, cotangents, directions)
    candidates = derivatives(actual, angles, cotangents, directions)
    for ref, result in zip(references, candidates, strict=True):
        torch.testing.assert_close(result, ref)


@pytest.mark.usefixtures("without_tf32")
@pytest.mark.cuda
@pytest.mark.parametrize("storage", ["blocks", "retained_rows"])
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (4, 2), (4, 4)])
def test_wigner_gpu_against_float64(storage, lmax, mmax):
    angles, js = make_inputs("cuda", torch.float32, lmax)
    refs = tuple(a.detach().double().requires_grad_() for a in angles)
    rows = wigner_utils.selected_local_rows(lmax, mmax)
    expected = dense_euler_reference(refs, tuple(j.double() for j in js), rows)
    if storage == "retained_rows":
        expected = (
            torch.cat(tuple(block.flatten(1) for block in expected), dim=1),
        )
    actual = wigner.sparse_wigner(*angles, js, mmax=mmax, storage=storage)
    if storage != "blocks":
        actual = (actual,)
    cotangents = tuple(torch.randn_like(y).requires_grad_() for y in actual)
    ref_cotangents = tuple(g.detach().double().requires_grad_() for g in cotangents)
    directions = tuple(torch.randn_like(a) for a in angles)
    references = derivatives(
        expected, refs, ref_cotangents, tuple(d.double() for d in directions)
    )
    candidates = derivatives(actual, angles, cotangents, directions)
    for ref, result in zip(references, candidates, strict=True):
        torch.testing.assert_close(result.double(), ref, atol=3e-3, rtol=3e-4)


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 0), (2, 1), (4, 4), (7, 5)])
def test_packing_defines_forward_and_inverse_maps(lmax, mmax):
    torch.manual_seed(73)
    width = (lmax + 1) ** 2
    full = torch.randn(2, width, width, dtype=torch.float64, requires_grad=True)
    packed = rotation_reference.pack_retained_rotations(full, lmax=lmax, mmax=mmax)
    torch.testing.assert_close(
        packed, pack_retained_rotations(full, lmax=lmax, mmax=mmax)
    )
    rows = rotation_reference.unpack_retained_rotations(packed, lmax=lmax, mmax=mmax)
    degree_mask = torch.block_diag(
        *[torch.ones(2 * degree + 1, 2 * degree + 1) for degree in range(lmax + 1)]
    )
    indices = [degree * degree + degree for degree in range(lmax + 1)] + [
        degree * degree + degree + sign * m
        for m in range(1, mmax + 1)
        for sign in (1, -1)
        for degree in range(m, lmax + 1)
    ]
    expected = (full * degree_mask)[:, indices]
    torch.testing.assert_close(rows, expected)
    inverse = expected.transpose(1, 2)
    torch.testing.assert_close(
        pack_inverse_rotations(inverse, lmax=lmax, mmax=mmax), packed
    )
    torch.testing.assert_close(
        rotation_reference.pack_inverse_rotations(inverse, lmax=lmax, mmax=mmax), packed
    )
    torch.testing.assert_close(
        rotation_reference.pack_retained_rotations(
            expected, lmax=lmax, mmax=mmax, input_layout="reduced_rows"
        ),
        packed,
    )
    x = torch.randn(2, width, 3, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(x)
    local = rows @ x
    (first,) = torch.autograd.grad(local.square().sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(first, x, direction)
    torch.testing.assert_close(second, 2 * rows.transpose(1, 2) @ (rows @ direction))


def test_inverse_scale_and_packing_have_second_derivatives():
    value = torch.randn(1, 9, 7, dtype=torch.float64, requires_grad=True)

    def operation(scale):
        return rotation_reference.compact_inverse_rescale(scale, lmax=2, mmax=1)

    torch.testing.assert_close(
        operation(value), compact_inverse_rescale(value, lmax=2, mmax=1)
    )
    assert torch.autograd.gradcheck(operation, (value,))
    assert torch.autograd.gradgradcheck(operation, (value,))


@pytest.mark.parametrize("convention", ["masked", "pole_clamped"])
def test_frame_reference_gradcheck_and_poles(convention):
    torch.manual_seed(79)
    frames = torch.linalg.qr(torch.randn(2, 3, 3, dtype=torch.float64))[
        0
    ].requires_grad_()
    matrices_j = tuple(
        torch.randn(2 * degree + 1, 2 * degree + 1, dtype=torch.float64)
        for degree in range(3)
    )

    def operation(value):
        return rotation_reference.retained_wigner_from_frames(
            value, matrices_j, mmax=1, angle_convention=convention
        )

    assert torch.autograd.gradcheck(operation, (frames,))
    assert torch.autograd.gradgradcheck(operation, (frames,))
    poles = torch.stack((torch.eye(3), torch.diag(torch.tensor([1.0, -1.0, -1.0]))))
    poles = poles.double().requires_grad_()
    angles = rotation_reference.rotation_angles(poles, angle_convention=convention)
    (gradient,) = torch.autograd.grad(
        sum(a.sum() for a in angles), poles, create_graph=True
    )
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    (second,) = torch.autograd.grad(gradient.sum(), poles)
    assert torch.isfinite(second).all()
    torch.testing.assert_close(second, torch.zeros_like(second), atol=0, rtol=0)


def sample_frames(edges):
    alpha, beta, gamma = torch.rand(3, edges, device="cuda")
    return o3.angles_to_matrix(
        (2 * alpha - 1) * torch.pi, 0.3 + beta * (torch.pi - 0.6),
        (2 * gamma - 1) * torch.pi,
    ).requires_grad_()


@pytest.mark.cuda
@pytest.mark.parametrize("convention", ["masked", "pole_clamped"])
@pytest.mark.parametrize(
    "lmax,mmax,edges", [(0, 0, 5), (3, 1, 5), (4, 4, 5), (4, 2, 16385)]
)
def test_frame_generator_and_complete_vjp_against_dense_reference(
    convention, lmax, mmax, edges
):
    torch.manual_seed(83)
    frames = sample_frames(edges)
    refs = frames.detach().double().requires_grad_()
    matrices_j = tuple(
        torch.randn(2 * degree + 1, 2 * degree + 1, device="cuda") * 0.3
        for degree in range(lmax + 1)
    )
    ref_j = tuple(j.double() for j in matrices_j)
    compact_width = sum(
        (2 * degree + 1) * (2 * min(degree, mmax) + 1) for degree in range(lmax + 1)
    )
    scale = torch.randn(compact_width, device="cuda")
    actual = retained_wigner_from_frames(
        frames,
        matrices_j,
        mmax=mmax,
        angle_convention=convention,
        inverse_rescale=scale,
    )
    expected = rotation_reference.retained_wigner_from_frames(
        refs,
        ref_j,
        mmax=mmax,
        angle_convention=convention,
        inverse_rescale=scale.double(),
    )
    cotangents = tuple(torch.randn_like(y, requires_grad=True) for y in actual)
    ref_cotangents = tuple(g.detach().double().requires_grad_() for g in cotangents)
    (first,) = torch.autograd.grad(actual, frames, cotangents, create_graph=True)
    (ref_first,) = torch.autograd.grad(
        expected, refs, ref_cotangents, create_graph=True
    )
    direction = torch.randn_like(frames)
    second = torch.autograd.grad(first, (frames, *cotangents), direction)
    ref_second = torch.autograd.grad(
        ref_first, (refs, *ref_cotangents), direction.double()
    )
    for result, target in zip(
        (*actual, first, *second), (*expected, ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(result.double(), target, atol=4e-4, rtol=6e-4)


def test_quaternion_reference_is_normalized_and_differentiable():
    torch.manual_seed(89)
    edges = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([0.0, 0.7, 2.1], dtype=torch.float64)

    def operation(x):
        return rotation_reference.fused_edge_quaternion(x, gamma)

    result = operation(edges)
    torch.testing.assert_close(
        result.square().sum(1), torch.ones(3, dtype=torch.float64)
    )
    assert torch.autograd.gradcheck(operation, (edges,))
    assert torch.autograd.gradgradcheck(operation, (edges,))
    poles = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    result = rotation_reference.fused_edge_quaternion(poles, gamma[:2])
    (first,) = torch.autograd.grad(result.sum(), poles, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), poles)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.cuda
@pytest.mark.parametrize("compile_visible", [False, True])
def test_quaternion_generator_matches_full_geometry_derivatives(compile_visible):
    from LunaEquivariance.so2.rotation.interface import fused_edge_quaternion

    torch.manual_seed(97)
    edges = torch.cat(
        (
            torch.randn(7, 3, device="cuda"),
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], device="cuda"),
        )
    )
    edges.requires_grad_()
    refs = edges.detach().double().requires_grad_()
    gamma = torch.rand(edges.shape[0], device="cuda") * (2 * torch.pi)
    actual = fused_edge_quaternion(edges, gamma, compile_visible=compile_visible)
    expected = rotation_reference.fused_edge_quaternion(refs, gamma.double())
    cotangent = torch.randn_like(actual, requires_grad=True)
    ref_cotangent = cotangent.detach().double().requires_grad_()
    (first,) = torch.autograd.grad(actual, edges, cotangent, create_graph=True)
    (ref_first,) = torch.autograd.grad(expected, refs, ref_cotangent, create_graph=True)
    direction = torch.randn_like(edges)
    second = torch.autograd.grad(first, (edges, cotangent), direction)
    ref_second = torch.autograd.grad(
        ref_first, (refs, ref_cotangent), direction.double()
    )
    for result, target in zip(
        (actual, first, *second), (expected, ref_first, *ref_second), strict=True
    ):
        torch.testing.assert_close(result.double(), target, atol=1e-4, rtol=5e-4)


@pytest.mark.cuda
def test_quaternion_compiled_launches(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    torch.manual_seed(97)
    edges = torch.randn(33, 3, device="cuda")
    gamma = torch.rand(33, device="cuda")
    cotangent = torch.randn(33, 4, device="cuda")
    direction = torch.randn_like(edges)

    def derivatives(edges, gamma, cotangent, direction):
        output = torch.empty_like(cotangent)
        gradient = torch.empty_like(edges)
        q_direction = torch.empty_like(cotangent)
        hvp = torch.empty_like(edges)
        torch.ops.so2.edge_quaternion_forward(edges, gamma, output)
        torch.ops.so2.edge_quaternion_backward(cotangent, edges, gamma, gradient)
        torch.ops.so2.edge_quaternion_hvp(
            cotangent, edges, gamma, direction, q_direction, hvp
        )
        return output, gradient, q_direction, hvp

    expected = derivatives(edges, gamma, cotangent, direction)
    compiled = torch.compile(derivatives, fullgraph=True, dynamic=True)
    actual = compiled(edges, gamma, cotangent, direction)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "device", "trainable", "edge_rank"]
)
def test_quaternion_rejects_an_invalid_frame_angle(invalid):
    from LunaEquivariance.so2.rotation.interface import fused_edge_quaternion

    edges = torch.randn(3, 3, device="cuda")
    gamma = torch.zeros(3, device="cuda")
    if invalid == "shape":
        gamma = gamma[:, None]
    elif invalid == "dtype":
        gamma = gamma.double()
    elif invalid == "device":
        gamma = gamma.cpu()
    elif invalid == "trainable":
        gamma.requires_grad_()
    else:
        edges = edges[None]
    with pytest.raises(ValueError):
        fused_edge_quaternion(edges, gamma)


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (1, 0), (2, 2), (4, 2), (7, 5)])
def test_rotation_packing_matches_degree_rows_and_derivatives(lmax, mmax):
    spec = get_retained_layout(lmax=lmax, mmax=mmax, channels=3)
    dense = torch.randn(
        2,
        spec.full_components,
        spec.full_components,
        dtype=torch.float64,
        requires_grad=True,
    )
    packed = pack_retained_rotations(dense, lmax=lmax, mmax=mmax)
    expected = torch.cat(
        [
            dense[
                :, degree * degree + degree + order, degree * degree : (degree + 1) ** 2
            ]
            for degree in range(lmax + 1)
            for order in [
                0,
                *(s * m for m in range(1, min(degree, mmax) + 1) for s in (1, -1)),
            ]
        ],
        dim=1,
    )
    torch.testing.assert_close(packed, expected)
    assert packed.shape == (2, spec.compact_elements)
    cotangent = torch.randn_like(packed, requires_grad=True)
    (first,) = torch.autograd.grad(packed, dense, cotangent, create_graph=True)
    direction = torch.randn_like(dense)
    (second,) = torch.autograd.grad(first, cotangent, direction)
    torch.testing.assert_close(
        second, pack_retained_rotations(direction, lmax=lmax, mmax=mmax)
    )


def test_inverse_scale_packing():
    layout = get_angular_layout(lmax=3, mmax=1, channels=1)
    spec = get_retained_layout(lmax=3, mmax=1, channels=1)
    scale = torch.arange(layout.full_components * layout.reduced_components).reshape(
        1, layout.full_components, layout.reduced_components
    )
    packed = compact_inverse_rescale(scale, lmax=3, mmax=1)
    for degree, rows in enumerate(spec.degree_rows):
        for row in rows:
            offset = spec.compact_row_offsets[row]
            assert torch.equal(
                packed[offset : offset + 2 * degree + 1],
                scale[0, degree * degree : (degree + 1) ** 2, row],
            )


def test_inverse_scale_packing_preserves_first_and_second_derivatives():
    layout = get_angular_layout(lmax=2, mmax=1, channels=1)
    scale = torch.randn(
        1,
        layout.full_components,
        layout.reduced_components,
        dtype=torch.float64,
        requires_grad=True,
    )
    # Rebuild each graph, as required for differentiable convention tensors.
    mask = torch.zeros_like(scale)
    for reduced_row, degree in enumerate(layout.reduced_degrees):
        mask[:, degree * degree : (degree + 1) ** 2, reduced_row] = 1
    for _ in range(2):
        packed = compact_inverse_rescale(scale, lmax=2, mmax=1)
        (first,) = torch.autograd.grad(packed.square().sum(), scale, create_graph=True)
        torch.testing.assert_close(first, 2 * scale * mask)
        direction = torch.randn_like(scale)
        (second,) = torch.autograd.grad(first, scale, direction)
        torch.testing.assert_close(second, 2 * direction * mask)


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 1), (2, 2), (4, 2)])
def test_rectangular_carriers_preserve_rows_scaling_and_gradients(lmax, mmax):
    layout = get_angular_layout(lmax=lmax, mmax=mmax, channels=1)
    dense = torch.randn(
        2,
        layout.full_components,
        layout.full_components,
        dtype=torch.float64,
        requires_grad=True,
    )
    rows = torch.tensor(layout.dense_rows)
    reduced = dense.index_select(1, rows)
    scale = torch.randn(
        1, layout.reduced_components, layout.full_components, dtype=dense.dtype
    )
    inverse = (reduced * scale).transpose(1, 2)
    direct = pack_retained_rotations(
        reduced, lmax=lmax, mmax=mmax, input_layout="reduced_rows"
    )
    packed_inverse = pack_inverse_rotations(inverse, lmax=lmax, mmax=mmax)
    expected_direct = pack_retained_rotations(dense, lmax=lmax, mmax=mmax)
    expected_scale = pack_retained_rotations(
        scale, lmax=lmax, mmax=mmax, input_layout="reduced_rows"
    )
    torch.testing.assert_close(direct, expected_direct)
    torch.testing.assert_close(packed_inverse, expected_direct * expected_scale)
    # Nonlinear loss checks input-input second derivatives as well as the
    # input-cotangent derivatives already tested for full dense packing.
    actual_loss = direct.square().sum() + packed_inverse.square().sum()
    reference_loss = (
        expected_direct.square().sum()
        + (expected_direct * expected_scale).square().sum()
    )
    (actual_first,) = torch.autograd.grad(actual_loss, dense, create_graph=True)
    (expected_first,) = torch.autograd.grad(reference_loss, dense, create_graph=True)
    torch.testing.assert_close(actual_first, expected_first)
    direction = torch.randn_like(dense)
    (actual_second,) = torch.autograd.grad(actual_first, dense, direction)
    (expected_second,) = torch.autograd.grad(expected_first, dense, direction)
    torch.testing.assert_close(actual_second, expected_second)


def test_full_and_reduced_rows_are_not_inferred_from_equal_shapes():
    layout = get_angular_layout(lmax=2, mmax=2, channels=1)
    dense = torch.arange(81.0).reshape(1, 9, 9)
    reduced = dense[:, layout.dense_rows, :]
    assert reduced.shape == dense.shape
    expected = pack_retained_rotations(dense, lmax=2, mmax=2)
    actual = pack_retained_rotations(
        reduced, lmax=2, mmax=2, input_layout="reduced_rows"
    )
    assert torch.equal(actual, expected)
    assert not torch.equal(pack_retained_rotations(reduced, lmax=2, mmax=2), expected)


def test_invalid_rotation_preparation():
    from LunaEquivariance.so2.rotation.interface import retained_wigner_from_frames

    with pytest.raises(ValueError, match="angle_convention"):
        retained_wigner_from_frames(None, (), mmax=0, angle_convention="automatic")
    with pytest.raises(ValueError, match="shape"):
        pack_retained_rotations(torch.zeros(2, 3, 3), lmax=1, mmax=1)
    with pytest.raises(ValueError, match="input_layout"):
        pack_retained_rotations(
            torch.zeros(1, 1, 1), lmax=0, mmax=0, input_layout="auto"
        )
    with pytest.raises(ValueError, match="shape"):
        pack_inverse_rotations(torch.zeros(1, 3, 3), lmax=1, mmax=1)


@pytest.mark.parametrize("convention", ["pole_clamped","masked"])
def test_rotation_reference_pole_derivatives(convention):
    frames = torch.tensor([[[1.,0,0],[0,1,0],[0,0,1]],
                           [[1.,0,0],[0,-1,0],[0,0,-1]]],dtype=torch.float64,requires_grad=True)
    basis = tuple(torch.eye(2*l+1,dtype=torch.float64) for l in range(3))
    direct = reference_rotation(frames,basis,mmax=1,convention=convention)
    grad = torch.autograd.grad(direct.sum(),frames,create_graph=True)[0]
    assert torch.isfinite(direct).all()
    assert torch.equal(grad,torch.zeros_like(grad))


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(1,0),(2,1),(4,2)])
@pytest.mark.parametrize("convention", ["pole_clamped","masked"])
def test_frame_rotation_derivatives(lmax,mmax,convention):
    torch.manual_seed(162)
    frames = torch.randn(4,3,3,device="cuda",requires_grad=True)
    # Keep the independent frame entries off the pole chart in both dtypes.
    with torch.no_grad():
        frames[:,1,1] = 0.2
    matrices = tuple(torch.randn(2*l+1,2*l+1,device="cuda") for l in range(lmax+1))
    expected_frames = frames.detach().double().requires_grad_()
    actual = rotation(frames,matrices,mmax=mmax,convention=convention)
    expected = reference_rotation(expected_frames,tuple(m.double() for m in matrices),mmax=mmax,convention=convention)
    seed = torch.randn_like(actual,requires_grad=True)
    direction = (torch.randn_like(frames),)
    for result,wanted in zip(
        output_derivatives(actual,(frames,),seed,direction),
        output_derivatives(expected,(expected_frames,),seed.detach().double().requires_grad_(),tuple(v.double() for v in direction)),
        strict=True,
    ):
        assert_reference_close(result,wanted)


@pytest.mark.cuda
def test_masked_rotation_near_poles_against_fp64():
    torch.manual_seed(5432)
    near = torch.linspace(0.006, 0.035, 33, device="cuda")
    beta = torch.cat((near, torch.pi - near))
    alpha, gamma = (2 * torch.rand(2, beta.numel(), device="cuda") - 1) * torch.pi
    frames = o3.angles_to_matrix(alpha, beta, gamma).requires_grad_()
    ref_frames = frames.detach().double().requires_grad_()
    observed = rotation(frames, lmax=2, mmax=2, convention="masked")
    expected = reference_rotation(ref_frames, lmax=2, mmax=2, convention="masked")
    seed = torch.randn_like(observed, requires_grad=True)
    ref_seed = seed.detach().double().requires_grad_()
    direction = (torch.randn_like(frames),)
    for result, target in zip(
        output_derivatives(observed, (frames,), seed, direction),
        output_derivatives(expected, (ref_frames,), ref_seed, (direction[0].double(),)),
        strict=True,
    ):
        error = (result.double() - target).norm()
        assert error <= 1e-4 * target.norm() + 1e-8


@pytest.mark.cuda
@pytest.mark.parametrize("lmax,mmax", [(1,0),(1,1),(2,1),(2,2)])
def test_quaternion_rotation_preserves_rng_and_derivatives(lmax,mmax):
    from LunaEquivariance.so2.rotation.kernels.utils import quadratic_coefficients
    from LunaEquivariance.so2.rotation.kernels.utils import select_retained_coefficients

    torch.manual_seed(199)
    geometry = torch.randn(4,3,device="cuda",requires_grad=True)
    quadratic = select_retained_coefficients(quadratic_coefficients(geometry.device),degrees=(1,),mmax=mmax)
    quartic = select_retained_coefficients(torch.randn(25,35,device="cuda"),degrees=(2,),mmax=mmax)
    ref_geometry = geometry.detach().double().requires_grad_()
    torch.cuda.manual_seed(951)
    actual = wigner.quaternion_wigner_from_edges_retained(
        geometry, quadratic, quartic, lmax=lmax, mmax=mmax,
    )
    torch.cuda.manual_seed(951)
    expected = rotation_reference.quaternion_wigner_from_edges_retained(
        ref_geometry, quadratic.double(), quartic.double(), lmax=lmax, mmax=mmax,
    )
    seed = torch.randn_like(actual,requires_grad=True)
    direction = (torch.randn_like(geometry),)
    for result,wanted in zip(
        output_derivatives(actual,(geometry,),seed,direction),
        output_derivatives(expected,(ref_geometry,),seed.detach().double().requires_grad_(),tuple(v.double() for v in direction)),
        strict=True,
    ):
        assert_reference_close(result,wanted)


@pytest.mark.cuda
def test_frame_rotation_scaled_inverse():
    geometry = torch.randn(3,3,3,device="cuda",requires_grad=True)
    with torch.no_grad():
        geometry[:,1,1] = 0.4
    matrices = tuple(torch.randn(2*l+1,2*l+1,device="cuda") for l in range(3))
    direct = rotation(geometry,matrices,mmax=1)
    scale = torch.full((direct.shape[1],),0.6,device="cuda")
    forward,inverse = rotation(geometry,matrices,mmax=1,inverse_scale=scale)
    torch.testing.assert_close(forward,direct)
    torch.testing.assert_close(inverse,direct*scale)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lmax,mmax,edges", [(1, 0, 4), (4, 2, 4), (6, 4, 4), (6, 2, 16385)]
)
def test_rotation_with_standard_basis(lmax, mmax, edges):
    torch.manual_seed(162)
    frames = sample_frames(edges)
    expected_frames = frames.detach().double().requires_grad_()
    actual = rotation(frames,lmax=lmax,mmax=mmax)
    expected = reference_rotation(expected_frames,lmax=lmax,mmax=mmax)
    seed = torch.randn_like(actual,requires_grad=True)
    direction = (torch.randn_like(frames),)
    for result,wanted in zip(
        output_derivatives(actual,(frames,),seed,direction),
        output_derivatives(expected,(expected_frames,),seed.detach().double().requires_grad_(),
                         tuple(v.double() for v in direction)),
        strict=True,
    ):
        assert_reference_close(result,wanted)


@pytest.mark.parametrize("function", [rotation,reference_rotation])
def test_rotation_requires_consistent_degree(function):
    frames = torch.eye(3).unsqueeze(0)
    with pytest.raises(ValueError,match="requires lmax"):
        function(frames,mmax=0)
    with pytest.raises(ValueError,match="must match"):
        function(frames,(torch.ones(1,1),),lmax=1,mmax=0)


@pytest.mark.parametrize("name", ["masked_angles", "pole_clamped_angles"])
def test_angle_derivative_storage_cpu(monkeypatch, name):
    wrapper = wigner.AnglesBackward if name == "masked_angles" else wigner.RotationAnglesBackward
    seen = []

    def backward(a, b, c, rotation):
        seen.append(tuple(x.stride() for x in (a, b, c)))
        return torch.zeros_like(rotation)

    def hvp(a, b, c, rotation, direction):
        seen.append(tuple(x.stride() for x in (a, b, c)))
        return (
            torch.zeros_like(a),
            torch.zeros_like(b),
            torch.zeros_like(c),
            torch.zeros_like(rotation),
        )

    monkeypatch.setattr(wigner, f"{name}_backward", backward)
    monkeypatch.setattr(wigner, f"{name}_hvp", hvp)
    rotation = torch.randn(3, 3, 3, requires_grad=True)
    cotangents = tuple(torch.ones(()).expand(3).requires_grad_() for _ in range(3))
    first = wrapper.apply(*cotangents, rotation)
    direction = torch.ones_like(first, requires_grad=True)
    second = torch.autograd.grad(
        first, (*cotangents, rotation), direction, create_graph=True
    )
    assert seen == [((1,), (1,), (1,))] * 2
    third = torch.autograd.grad(
        sum(value.sum() for value in second), (*cotangents, rotation, direction)
    )
    assert all(torch.isfinite(value).all() for value in third)


def test_wigner_broadcast_d2_directions_cpu(monkeypatch):
    seen = []
    monkeypatch.setattr(
        wigner,
        "wigner_backward",
        lambda gs, a, b, c, js, rows: tuple(torch.zeros_like(x) for x in (a, b, c)),
    )

    def hvp(gs, a, b, c, directions, js, rows):
        seen.extend(direction.stride() for direction in directions)
        return tuple(torch.zeros_like(g) for g in gs), tuple(
            torch.zeros_like(x) for x in (a, b, c)
        )

    monkeypatch.setattr(wigner, "wigner_hvp", hvp)
    angles = tuple(torch.randn(3, requires_grad=True) for _ in range(3))
    g = torch.ones(3, 1, 1, requires_grad=True)
    first = wigner.WignerBackward.apply(g, *angles, torch.ones(1, 1), ((0,),))
    torch.autograd.grad(sum(x.sum() for x in first), (*angles, g), create_graph=True)
    assert seen == [(1,)] * 3


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["masked_angles", "pole_clamped_angles"])
@pytest.mark.parametrize("kind", ["sum", "strided"])
def test_angle_strided_d1_d2_gpu(name, kind):
    angles = getattr(wigner, f"{name}_fused_rotation_angles")
    torch.manual_seed(181)
    rotation = torch.linalg.qr(torch.randn(4, 3, 3, device="cuda"))[0]
    direction = torch.randn_like(rotation)

    def evaluate(dense):
        r = rotation.detach().requires_grad_()
        gs = tuple(
            (
                make_cotangent(r[:, 0, 0], kind).contiguous()
                if dense
                else make_cotangent(r[:, 0, 0], kind)
            ).requires_grad_()
            for _ in range(3)
        )
        outputs = angles(r)
        first = torch.autograd.grad(outputs, r, gs, create_graph=True)[0]
        second = torch.autograd.grad(first, (r, *gs), direction, create_graph=True)
        return (*outputs, first, *second)

    actual, expected = evaluate(False), evaluate(True)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-5, atol=3e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["sum", "strided"])
def test_wigner_strided_d2_gpu(kind):
    torch.manual_seed(182)
    values = tuple(torch.randn(4, device="cuda") for _ in range(3))
    js = tuple(torch.randn(2 * l + 1, 2 * l + 1, device="cuda") for l in range(3))

    def evaluate(dense):
        angles = tuple(x.detach().requires_grad_() for x in values)
        outputs = wigner.sparse_wigner(*angles, js, mmax=2)
        gs = tuple(torch.ones_like(y, requires_grad=True) for y in outputs)
        first = torch.autograd.grad(outputs, angles, gs, create_graph=True)
        directions = tuple(make_cotangent(x, kind) for x in angles)
        if dense:
            directions = tuple(x.contiguous() for x in directions)
        return torch.autograd.grad(first, (*angles, *gs), directions, create_graph=True)

    for got, want in zip(evaluate(False), evaluate(True), strict=True):
        torch.testing.assert_close(got, want, rtol=3e-5, atol=3e-5)
