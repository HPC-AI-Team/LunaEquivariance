"""Recombine raw SO(2) products and apply the selected activation."""

from __future__ import annotations

from functools import partial

import torch
import triton
from torch.autograd.function import once_differentiable

from ..utils.layout import check_orders
from ..utils.sector_layout import Pack, Unpack, layout, merge_sectors, validate_float_inputs
from ..utils.tensor_contracts import require_int32_offsets
from .kernels.backward import gate_backward, raw_activation_derivative, recombine_vjp_kernel
from .kernels.double_backward import double_backward_kernel
from .kernels.forward import forward_kernel, grid_forward, recombine
from .kernels.triple_backward import (
    ActivationDoubleBackward,
    activation_third_backward,
    gate_third_backward,
    grid_hvp,
    projected_edge_grid_third,
)
from .kernels.utils import gate_gradient_buffers, gate_scalar_views


def recombine_activate(
    raw, *, lmax, mmax, activation="gate", auxiliary_channels=0,
    to_grid=None, from_grid=None,
):
    """Consume raw products, apply the selected activation, and return sectors.

    Returns ``(sectors, auxiliary)``. Auxiliary channels are the leading
    unchanged scalar output, of shape ``(E,auxiliary_channels)``.

    ``activation="gate"`` expects the remaining scalar output to contain
    ``L*C`` gates followed by ``(L+1)*C`` coefficient channels. Degree zero
    uses SiLU and higher degrees use sigmoid gates.

    ``activation="grid"`` expects three ``C`` scalar blocks (scalar a,
    scalar b, gate), followed by ``(L+1)*2*C`` coefficient channels. Fixed
    ``to_grid`` and ``from_grid`` matrices act on retained components.
    The two recombined grid fields are multiplied, gated and transformed
    back. Both forms use CUDA FP32 and support derivatives through third order.
    """
    raw = tuple(raw)
    check_orders(lmax, mmax, raw)
    if raw[0].ndim != 2 or auxiliary_channels < 0:
        raise ValueError("raw order zero must be a matrix with a nonnegative auxiliary width")
    width = raw[0].shape[-1] - auxiliary_channels
    if activation == "gate":
        if to_grid is not None or from_grid is not None:
            raise ValueError("grid transforms apply only to the grid activation")
        divisor = 2 * lmax + 1
        if width <= 0 or width % divisor:
            raise ValueError("gate output must contain L*C gates and (L+1)*C coefficients")
        channels = width // divisor
        if auxiliary_channels == 0:
            sectors = apply_gate(raw[0], None, raw[1:], lmax, mmax, channels)
        else:
            start = auxiliary_channels
            gates = raw[0][:, start:start + lmax * channels]
            coefficients = raw[0][:, start + lmax * channels:].reshape(
                raw[0].shape[0], lmax + 1, channels
            )
            sectors = gated_activation(gates, coefficients, raw[1:], lmax=lmax, mmax=mmax)
        return sectors, raw[0][:, :auxiliary_channels]
    if activation == "grid":
        if to_grid is None or from_grid is None:
            raise ValueError("grid activation requires both transform matrices")
        divisor = 2 * (lmax + 1) + 3
        if width <= 0 or width % divisor:
            raise ValueError("grid output must contain paired coefficients and three scalar blocks")
        return grid_activation(
            raw, to_grid, from_grid, lmax=lmax, mmax=mmax,
            channels=width // divisor, auxiliary_channels=auxiliary_channels,
        )
    raise ValueError("activation must be 'gate' or 'grid'")


class GroupedGateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scalar, input0, lmax, mmax, raw_complex, *args):
        count = mmax + 1
        grads, sectors = args[:count], args[count:count + mmax]
        ctx.carriers = len(args) - (count + mmax)
        values = gate_backward(
            grads, scalar, input0, sectors, lmax=lmax, mmax=mmax, raw_complex=raw_complex,
        )
        ctx.save_for_backward(scalar, input0, *grads, *sectors)
        ctx.dimensions = (lmax, mmax, raw_complex)
        return values

    @staticmethod
    def backward(ctx, scalar_direction, input0_direction, *sector_directions):
        lmax, mmax, raw_complex = ctx.dimensions
        count = mmax + 1
        saved = ctx.saved_tensors
        scalar, input0 = saved[:2]
        grads, sectors = saved[2:count + 2], saved[count + 2:]
        if scalar_direction is None:
            scalar_direction = torch.zeros_like(scalar)
        if input0 is not None and input0_direction is None:
            input0_direction = torch.zeros_like(input0)
        if scalar_direction.stride(-1) != 1:
            scalar_direction = scalar_direction.contiguous()
        if input0_direction is not None:
            input0_direction = input0_direction.contiguous()
        edges = scalar.shape[0]
        channels = scalar.shape[1] // (2 * lmax + 1) if input0 is None else input0.shape[-1]
        directions = tuple(
            direction if direction is not None else scalar.new_zeros(
                (edges, 2, 2 * (lmax - order + 1) * channels) if raw_complex else sector.shape
            )
            for order, (direction, sector) in enumerate(zip(sector_directions, sectors, strict=True), 1)
        )
        args = scalar, input0, scalar_direction, input0_direction, *grads, *sectors, *directions
        if torch.is_grad_enabled():
            values = ActivationDoubleBackward.apply(
                partial(gate_double_backward, lmax=lmax, mmax=mmax, raw_complex=raw_complex),
                partial(gate_third_backward, lmax=lmax, mmax=mmax, raw_complex=raw_complex),
                (), None, *args,
            )
        else:
            values = gate_double_backward(
                *args, lmax=lmax, mmax=mmax, raw_complex=raw_complex,
                fold_gate_adjoint=bool(ctx.carriers),
            )
            if ctx.carriers:
                return (values[0], values[1], None, None, None, *values[2:2 + count],
                        *(None,) * mmax, *values[2 + count:])
        return values[0], values[1], None, None, None, *values[2:], *(None,) * ctx.carriers


def gate_double_backward(scalar, input0, scalar_direction, input0_direction, *inputs,
                         lmax, mmax, raw_complex=True, fold_gate_adjoint=False):
    count = mmax + 1
    grads, sectors, sector_directions = inputs[:count], inputs[count:2 * count - 1], inputs[2 * count - 1:]
    gating, x0 = gate_scalar_views(scalar, input0, lmax)
    gate_direction, x0_direction = gate_scalar_views(scalar_direction, input0_direction, lmax)
    edges, channels = x0.shape[0], x0.shape[-1]
    directions = x0_direction, *sector_directions
    dgrads = tuple(torch.empty_like(value) for value in grads)
    ds, d0, dgates, dx0 = gate_gradient_buffers(scalar, input0, lmax)
    dsectors = tuple(torch.empty_like(value) for value in sectors)
    block = min(256, triton.next_power_of_2(channels))
    double_backward_kernel[(edges, triton.cdiv(channels, block))](
        grads, gating, (x0, *sectors),
        gate_direction, directions,
        dgrads, dgates, (dx0, *dsectors),
        gating.stride(0), x0.stride(0), dgates.stride(0), dx0.stride(0),
        gate_direction.stride(0), x0_direction.stride(0), x0_direction.stride(1),
        tuple(value.stride() for value in directions),
        channels=channels,
        lmax=lmax,
        mmax=mmax,
        raw_complex=raw_complex,
        block_channels=block,
        fold_gate_adjoint=fold_gate_adjoint,
        num_warps=4,
    )
    return ds, d0, *dgrads, *dsectors


def gate_forward(scalar, input0, sectors, lmax, mmax, *, raw_complex=True, three_products=False):
    gating, x0 = gate_scalar_views(scalar, input0, lmax)
    edges, channels = x0.shape[0], x0.shape[-1]
    outputs = (x0.new_empty(x0.shape),) + tuple(
        x0.new_empty((edges, 2 * (lmax - order + 1), channels))
        for order in range(1, mmax + 1)
    )
    block = min(256, triton.next_power_of_2(channels))
    forward_kernel[(edges, triton.cdiv(channels, block))](
        gating, (x0, *sectors), outputs, gating.stride(0), x0.stride(0),
        channels=channels, lmax=lmax, mmax=mmax, raw_complex=raw_complex,
        block_channels=block, three_products=three_products, num_warps=4,
    )
    return outputs


class GroupedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scalar, input0, lmax, mmax, *sectors):
        _, x0 = gate_scalar_views(scalar, input0, lmax)
        raw_complex = not sectors or sectors[0].shape[-1] != x0.shape[-1]
        outputs = gate_forward(scalar, input0, sectors, lmax, mmax, raw_complex=raw_complex)
        # The private outputs share storage with the gated sectors. Their
        # adjoint returns only projection gradients after D2 folds the gate.
        carriers = tuple(value.view_as(value) for value in outputs[1:])
        ctx.save_for_backward(scalar, input0, *outputs[1:], *carriers)
        ctx.dimensions = (lmax, mmax, raw_complex)
        ctx.set_materialize_grads(False)
        return (*outputs, *carriers)

    @staticmethod
    def backward(ctx, *grads):
        lmax, mmax, raw_complex = ctx.dimensions
        scalar, input0, *saved = ctx.saved_tensors
        sectors, carriers = saved[:mmax], saved[mmax:]
        visible, raw_only = grads[:mmax + 1], grads[mmax + 1:]
        if torch.is_grad_enabled():
            _, x0 = gate_scalar_views(scalar, input0, lmax)
            grads = tuple(
                sector.new_zeros(sector.shape) if grad is None else grad.contiguous()
                for sector, grad in zip((x0, *sectors), visible, strict=True)
            )
            values = GroupedGateBackward.apply(
                scalar, input0, lmax, mmax, raw_complex, *grads, *sectors, *carriers,
            )
        else:
            visible = tuple(grad.contiguous() if grad is not None else None for grad in visible)
            raw_only = tuple(grad.contiguous() if grad is not None else None for grad in raw_only)
            raw_cotangents = raw_only if any(v is not None for v in raw_only) else ()
            skip_sectors = ()
            if raw_complex and not torch.compiler.is_compiling() and hasattr(ctx, "maybe_clear_saved_tensors"):
                skip_sectors = tuple(
                    grad is None and raw is None for grad, raw in zip(visible[1:], raw_only, strict=True)
                )
                if not any(skip_sectors):
                    skip_sectors = ()
            gate_options = dict(
                lmax=lmax, mmax=mmax, raw_complex=raw_complex, raw_cotangents=raw_cotangents,
            )
            if skip_sectors:
                shapes = tuple((value.shape[0], 2, value.shape[1] * value.shape[2]) for value in sectors)
                empty = not any(value is not None for value in visible) and not raw_cotangents
                if not empty:
                    values = gate_backward(visible, scalar, input0, sectors, **gate_options, skip_sectors=skip_sectors)
                # Materialize unused input gradients after dropping the saved gated sectors.
                del saved, sectors, carriers
                ctx.maybe_clear_saved_tensors()
                if empty:
                    values = (
                        scalar.new_zeros(scalar.shape),
                        input0.new_zeros(input0.shape) if input0 is not None else None,
                        *(None,) * mmax,
                    )
                values = (*values[:2], *(scalar.new_zeros(shape) if skip else value
                           for value, skip, shape in zip(values[2:], skip_sectors, shapes, strict=True)))
            else:
                values = gate_backward(visible, scalar, input0, sectors, **gate_options)
        return values[0], values[1], None, None, *values[2:]


class ThreeProductGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gating, input0, lmax, mmax, *products):
        outputs = gate_forward(gating, input0, products, lmax, mmax, three_products=True)
        ctx.save_for_backward(gating, input0, *outputs[1:])
        ctx.dimensions = lmax, mmax
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        gating, input0, *sectors = ctx.saved_tensors
        grads = tuple(
            value.new_zeros(value.shape) if grad is None else grad.resolve_neg().contiguous()
            for value, grad in zip((input0, *sectors), grads, strict=True)
        )
        lmax, mmax = ctx.dimensions
        dg, dx, *dproducts = gate_backward(
            grads, gating, input0, sectors, lmax=lmax, mmax=mmax, three_products=True,
        )
        return dg, dx, None, None, *dproducts


def gated_activation(gating, input0, inputs, *, lmax, mmax):
    """Recombine raw products and apply a sigmoid gate for each degree.

    Let ``L=lmax``, ``M=mmax``, and ``w_m=L+1-m``. For ``E`` edges and
    ``C`` channels, ``gating`` is ``(E,L*C)`` and ``input0`` is
    ``(E,L+1,C)``. ``inputs[m-1]`` is ``(E,2,2*w_m*C)``. Reshape it to
    ``q[E,plane,map,degree,channel]`` with two real/imaginary maps.
    The physical planes are ``q[:,0,0]-q[:,1,1]`` and
    ``q[:,1,0]+q[:,0,1]``. Their degrees increase from ``m`` to ``L``.

    Return a tuple of physical sectors. Sector zero is ``(E,L+1,C)``.
    Its degree-zero row uses SiLU, and degree ``ell>0`` is multiplied by
    ``sigmoid(gating.reshape(E,L,C)[:,ell-1])``. Nonzero sectors are
    ``(E,2*w_m,C)`` with positive then negative rows. Each row uses the
    same gate as its degree in sector zero.

    All inputs are CUDA FP32 on one device. Derivatives through third
    order include gates and all raw products. Recombination and gating share
    one kernel, avoiding a separate recombined edge tensor.
    """
    if not 0 <= mmax <= lmax or len(inputs) != mmax:
        raise ValueError(
            "grouped raw gate requires one sector for each 0 <= m <= M <= L"
        )
    if input0.ndim != 3 or input0.shape[-1] <= 0:
        raise ValueError(
            "grouped raw gate requires rank-three sectors with positive channels"
        )
    edges, channels = input0.shape[0], input0.shape[-1]
    if tuple(input0.shape) != (edges, lmax + 1, channels):
        raise ValueError("grouped raw gate received inconsistent sector shapes")
    return apply_gate(gating, input0, inputs, lmax, mmax, channels)


def apply_gate(scalar, input0, inputs, lmax, mmax, channels, *, raw_complex=True):
    edges = scalar.shape[0] if input0 is None else input0.shape[0]
    scalar_width = (2 * lmax + 1 if input0 is None else lmax) * channels
    expected = tuple(
        (edges, 2, 2 * (lmax - order + 1) * channels) if raw_complex else
        (edges, 2 * (lmax - order + 1), channels)
        for order in range(1, mmax + 1)
    )
    if tuple(scalar.shape) != (edges, scalar_width) or tuple(x.shape for x in inputs) != expected:
        raise ValueError("grouped raw gate received inconsistent sector shapes")
    values = (scalar, *inputs) if input0 is None else (scalar, input0, *inputs)
    if not all(
        value.is_cuda and value.dtype == torch.float32 and value.device == scalar.device
        for value in values
    ):
        raise ValueError("grouped raw gate requires CUDA FP32 tensors")
    # Preserve the projection buffer edge stride; inner axes must be dense.
    if scalar.stride(1) != 1:
        scalar = scalar.contiguous()
    if input0 is not None and (input0.stride(1) != channels or input0.stride(2) != 1):
        input0 = input0.contiguous()
    inputs = tuple(value.contiguous() for value in inputs)
    return GroupedGate.apply(scalar, input0, lmax, mmax, *inputs)[:mmax + 1]


def use_sector_cotangents(scalar, physical_input, lmax, mmax, channels):
    return (
        physical_input
        and (lmax, mmax, channels) == (4, 2, 32)
        and 8192 <= scalar.shape[0] <= 32768
        and not torch.compiler.is_compiling()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(scalar.device) in ((8, 0), (9, 0))
    )


class GridActivationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dimensions, physical_input, daux, to_grid, from_grid, coefficients, polynomial, *values):
        count = dimensions[1] + 1
        raw, cotangents = values[:count], values[count:]
        dy = cotangents if cotangents[0].ndim == 2 else cotangents[0]
        gradients, _ = raw_activation_derivative(
            raw, dy, daux, to_grid, from_grid, *dimensions, physical_input=physical_input,
            coefficients=coefficients,
        )
        ctx.save_for_backward(to_grid, from_grid, polynomial, *values)
        ctx.dimensions = dimensions
        ctx.physical_input = physical_input
        ctx.coefficients = coefficients
        return gradients

    @staticmethod
    def backward(ctx, *directions):
        to_grid, from_grid, polynomial, *values = ctx.saved_tensors
        constants = to_grid, from_grid, polynomial
        count = ctx.dimensions[1] + 1
        raw, cotangents = values[:count], values[count:]
        tangents = []
        for value, direction in zip(raw, directions):
            if direction is None:
                direction = torch.zeros_like(value)
            elif direction.stride(-1) != 1:
                direction = direction.contiguous()
            tangents.append(direction)
        directions = tuple(tangents)
        args = *raw, *directions, *cotangents
        options = {
            "dimensions": ctx.dimensions,
            "physical_input": ctx.physical_input,
        }
        second = partial(grid_double_backward, **options, coefficients=ctx.coefficients)
        if torch.is_grad_enabled():
            third = partial(activation_third_backward, partial(grid_hvp, **options))
            if (
                polynomial is not None and ctx.physical_input and len(cotangents) == count
                and cotangents[0].ndim == 2 and not torch.compiler.is_compiling()
                and torch.version.hip is None and torch.cuda.get_device_capability(raw[0].device) == (9, 0)
            ):
                third = partial(projected_edge_grid_third, dimensions=ctx.dimensions)
            gradients = ActivationDoubleBackward.apply(
                second, third, constants, 1024 if polynomial is not None else None, *args,
            )
        else:
            gradients = second(*constants, *args)
        auxiliary_channels = ctx.dimensions[-1]
        return None, None, directions[0][:, :auxiliary_channels], None, None, None, None, *gradients


def grid_double_backward(to_grid, from_grid, polynomial, *inputs, dimensions, physical_input, coefficients):
    count = dimensions[1] + 1
    raw, directions, cotangents = inputs[:count], inputs[count:2 * count], inputs[2 * count:]
    sectors = cotangents[0].ndim == 2
    dy = cotangents if sectors else cotangents[0]
    gradients, d_dy = raw_activation_derivative(
        raw, dy, directions[0], to_grid, from_grid, *dimensions,
        directions=directions, physical_input=physical_input, coefficients=coefficients,
    )
    return (*gradients, *d_dy) if sectors else (*gradients, d_dy)


class GridActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dimensions, physical_input, to_grid, from_grid, coefficients, polynomial, *raw):
        outputs = grid_forward(
            raw, to_grid, from_grid, *dimensions, physical_input=physical_input, coefficients=coefficients,
        )
        ctx.save_for_backward(to_grid, from_grid, polynomial, *raw)
        ctx.dimensions = dimensions
        ctx.physical_input = physical_input
        ctx.coefficients = coefficients
        return (*outputs, raw[0][:, : dimensions[-1]])

    @staticmethod
    def backward(ctx, *gradients):
        to_grid, from_grid, polynomial, *raw = ctx.saved_tensors
        lmax, mmax, channels, auxiliary_channels = ctx.dimensions
        active, daux = gradients[:-1], gradients[-1]
        if use_sector_cotangents(raw[0], ctx.physical_input, lmax, mmax, channels):
            cotangents = tuple(grad.resolve_neg() for grad in active)
            for grad in cotangents:
                require_int32_offsets(grad.shape, grad.stride(), name="grid activation cotangent")
        else:
            cotangents = (merge_sectors(active, lmax=lmax, mmax=mmax, channels=channels),)
        if daux is None:
            daux = raw[0].new_zeros((raw[0].shape[0], auxiliary_channels))
        draw = GridActivationBackward.apply(
            ctx.dimensions, ctx.physical_input, daux, to_grid, from_grid, ctx.coefficients, polynomial,
            *raw, *cotangents,
        )
        return (None, None, None, None, None, None, *draw)


def grid_activation(
    raw, to_grid, from_grid, *, lmax, mmax, channels, auxiliary_channels, physical_input=False,
    projected=None,
):
    """Apply the gated spherical-grid product to projected sectors.

    Let C=channels, A=auxiliary_channels, L=lmax, M=mmax, and
    R=sum((1 if m == 0 else 2)*(L+1-m) for m in range(M+1)).
    ``raw[0]`` is (E, A+3*C+(L+1)*2*C); ``raw[m]`` is
    (E, 2, 4*(L+1-m)*C) for m>0. The leading A scalar channels are
    returned unchanged as a separate (E, A) tensor. The next scalar blocks
    are ``scalar_a``, ``scalar_b``, and ``gate``, each with C channels.
    The remaining coefficients have 2*C channels, split into ``a,b``
    after complex recombination. For each order m>0, reshape its raw
    product to ``q[E,plane,map,degree,2*C]``. Physical positive and
    negative planes are ``q[:,0,0]-q[:,1,1]`` and
    ``q[:,1,0]+q[:,0,1]``. Coefficient rows follow ``0,+1,-1,+2,-2,...``
    with increasing degrees inside each signed order. With
    ``physical_input=True`` the nonzero orders already contain these signed
    planes and have shape (E, 2, 2*(L+1-m)*C).

    Fixed ``to_grid`` and ``from_grid`` tables are (G, R) and (R, G).
    The output coefficients are
    ``sigmoid(gate) * from_grid @ ((to_grid @ a) * (to_grid @ b))``
    independently for each edge and channel. Row zero also receives
    ``SiLU(scalar_a) * scalar_b``. Inputs must be CUDA FP32 on one device,
    with contiguous raw channel axes. Return ``(sectors, auxiliary)``. Sector zero has shape
    ``(E, (L+1)*C)`` and sector m>0 has shape ``(E, 2, (L+1-m)*C)``.
    These flat sectors feed the next SO(2) matrix product without packing.
    ``merge_sectors`` assembles them into ``(E, R, C)`` when needed.
    Auxiliary output has shape ``(E, A)``. Raw inputs support derivatives
    through triple backward. Grid tables are fixed and do not accept gradients.
    Recombination, grid products, gating, and output packing avoid separate
    recombined and grid-domain edge arrays.
    """
    if lmax < 0 or not 0 <= mmax <= lmax or len(raw) != mmax + 1:
        raise ValueError("raw grid activation requires one sector per retained order")
    if channels <= 0 or auxiliary_channels < 0:
        raise ValueError(
            "grid activation requires positive channels and nonnegative auxiliary width"
        )
    if raw[0].ndim != 2:
        raise ValueError(
            "the raw zero-order sector must have shape (E, A+3*C+(L+1)*2*C)"
        )
    edges = raw[0].shape[0]
    hidden = 2 * channels
    extras = auxiliary_channels + 3 * channels
    expected = [(edges, extras + (lmax + 1) * hidden)]
    maps = 1 if physical_input else 2
    expected.extend(
        (edges, 2, maps * (lmax - order + 1) * hidden) for order in range(1, mmax + 1)
    )
    if tuple(value.shape for value in raw) != tuple(expected):
        raise ValueError(
            "raw sectors must contain the declared paired channels and scalar side outputs"
        )
    values = (*raw, to_grid, from_grid)
    if any(
        not value.is_cuda
        or value.dtype != torch.float32
        or value.device != raw[0].device
        for value in values
    ):
        raise ValueError("raw grid activation requires CUDA FP32 tensors on one device")
    if any(value.stride(-1) != 1 for value in raw):
        raise ValueError("raw sector channels must be contiguous")
    rows = sum(
        (1 if order == 0 else 2) * (lmax - order + 1) for order in range(mmax + 1)
    )
    if (
        to_grid.ndim != 2
        or to_grid.shape[1] != rows
        or from_grid.shape != (rows, to_grid.shape[0])
    ):
        raise ValueError("grid transforms must match the retained angular components")
    if to_grid.requires_grad or from_grid.requires_grad:
        raise ValueError("grid transform tables must be fixed")
    for value in values:
        require_int32_offsets(value.shape, value.stride(), name="grid activation input")
    require_int32_offsets(
        (edges, rows, channels),
        (rows * channels, channels, 1),
        name="grid activation output workspace",
    )
    dimensions = (lmax, mmax, channels, auxiliary_channels)
    coefficients, polynomial = (None, None) if projected is None else projected
    outputs = GridActivation.apply(dimensions, physical_input, to_grid, from_grid, coefficients, polynomial, *raw)
    return tuple(outputs[:-1]), outputs[-1]


def recombine_sectors(
    raw: tuple[torch.Tensor, ...],
    *,
    lmax: int,
    mmax: int,
    channels: int,
    extra_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine complex products into m-major coefficients and scalar extras.

    Zero-order input is ``(E, extra_channels + (L+1)*C)``. It starts with
    scalar side outputs followed by the zero-order coefficients.
    For ``m>0``, let ``w=L+1-m`` and reshape raw ``(E,2,2*w*C)`` to
    ``q[E,plane,map,degree,channel]`` of shape ``(E,2,2,w,C)``. The map
    index selects real then imaginary weights. Positive coefficients are
    ``q[:,0,0]-q[:,1,1]`` and negative coefficients are
    ``q[:,1,0]+q[:,0,1]``.

    Return ``(coefficients, extras)``. Coefficients have shape ``(E,R,C)``
    with rows ``0,+1,-1,+2,-2,...`` and increasing degrees within each
    signed order. ``R=L+1+2*sum(L+1-m for m in range(1,M+1))``. Extras
    have shape ``(E,extra_channels)``. Inputs are CUDA FP32 on one device.
    Noncontiguous channel axes are copied. Derivatives through third order
    cover every raw input. The fused implementation assembles the final
    coefficient tensor without separate per-order recombination arrays.
    """

    widths, _, _ = layout(lmax, mmax)
    if channels <= 0 or extra_channels < 0:
        raise ValueError(
            "recombination requires positive channels and nonnegative scalar extras"
        )
    if len(raw) != mmax + 1:
        raise ValueError("raw sector count does not match mmax")
    if raw[0].ndim != 2:
        raise ValueError(
            "the raw zero-order sector must have shape (E, extra_channels + (L+1)*C)"
        )
    edges = raw[0].shape[0]
    expected = [(edges, extra_channels + widths[0] * channels)]
    expected.extend((edges, 2, 2 * width * channels) for width in widths[1:])
    if tuple(tuple(value.shape) for value in raw) != tuple(expected):
        raise ValueError("raw sector shapes do not match L, M, C, and scalar extras")
    validate_float_inputs(raw, "complex recombination")
    raw = tuple(value if value.stride(-1) == 1 else value.contiguous() for value in raw)
    return Pack.apply(*raw, lmax, mmax, channels, extra_channels)


def recombine_sectors_adjoint(
    representation: torch.Tensor,
    extra: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    channels: int,
    extra_channels: int,
) -> tuple[torch.Tensor, ...]:
    """Apply the adjoint of complex recombination to coefficient cotangents.

    This distributes each signed coefficient derivative to its two contributing
    raw products. It is an adjoint, not an inverse of ``recombine_sectors``.
    """

    outputs = Unpack.apply(
        representation, extra, lmax, mmax, channels, extra_channels
    )
    return tuple(outputs[: mmax + 1])


class ComplexRecombinationBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, half):
        ctx.half = half
        grad = grad.contiguous()
        elements = grad.shape[0] * half
        raw_gradient = torch.empty(
            (grad.shape[0], 2, 2 * half), device=grad.device, dtype=grad.dtype
        )
        recombine_vjp_kernel[(triton.cdiv(elements, 256),)](
            grad, raw_gradient, elements, half, block=256, num_warps=4
        )
        return raw_gradient

    @staticmethod
    def backward(ctx, raw_direction):
        # Recombination is linear: differentiating its adjoint applies it again.
        return recombine(raw_direction.contiguous(), ctx.half), None


class ComplexRecombination(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, half):
        ctx.half = half
        return recombine(raw, half)

    @staticmethod
    def backward(ctx, grad):
        return ComplexRecombinationBackward.apply(grad, ctx.half), None


def complex_recombination(raw, half):
    """Combine the two real matrix products for one nonzero angular order.

    Contiguous CUDA FP32 ``raw`` has shape ``(E, 2, 2*half)``. Reshape it
    to ``q[E, input_plane, weight_map, half]``. The returned ``(E,2,half)``
    tensor contains ``q[:,0,0]-q[:,1,1]`` and ``q[:,1,0]+q[:,0,1]``.
    Backward applies the transpose of this linear map, and double backward
    reuses its forward kernel. No edge features are saved for differentiation.
    """
    return ComplexRecombination.apply(raw, half)
