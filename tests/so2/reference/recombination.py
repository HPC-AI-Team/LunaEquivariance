"""PyTorch recombination and edge activations with explicit intermediates."""

import torch
from torch.nn import functional as F

from tests.so2.reference.projection import merge_sectors



def activate_gated(gating, input0, inputs, *, lmax, mmax):
    """Recombine each raw sector, then apply degree-specific sigmoid gates.

    ``gating`` is ``(E, L*C)`` and ``input0`` is ``(E, L+1, C)``.
    Degree zero uses SiLU. Degree ``ell>0`` uses the gate at ``ell-1``
    for both signed-order coefficients. Return a tuple of physical sectors:
    zero ``(E, L+1, C)``, others ``(E, 2*w_m, C)``.
    """
    edges, _, channels = input0.shape
    gates = gating.reshape(edges, lmax, channels).sigmoid()
    scalar = torch.cat((F.silu(input0[:, :1]), input0[:, 1:] * gates), dim=1)
    outputs = [scalar]
    for order, raw in enumerate(inputs, start=1):
        width = lmax + 1 - order
        q = raw.reshape(edges, 2, 2, width, channels)
        physical = torch.cat((q[:, 0, 0] - q[:, 1, 1], q[:, 1, 0] + q[:, 0, 1]), dim=1)
        degree_gates = gates[:, order - 1 :]
        outputs.append(physical * torch.cat((degree_gates, degree_gates), dim=1))
    return tuple(outputs[: mmax + 1])


def activate_grid(
    raw, to_grid, from_grid, *, lmax, mmax, channels, auxiliary_channels, physical_input=False,
):
    """Recombine paired channels and apply a gated spherical-grid product.

    Raw products contain ``2*C`` coefficient channels. The zero-order
    prefix is ``[auxiliary(A), scalar_a(C), scalar_b(C), gate(C)]``.
    Recombination yields ``coefficients[E,R,2*C]``. Split channels into
    ``a,b`` and compute ``sigmoid(gate) * F[(T a) * (T b)]``. Add
    ``SiLU(scalar_a) * scalar_b`` only to angular row zero. ``T`` and ``F``
    are fixed ``(G,R)`` and ``(R,G)`` tables. Return ``(sectors, auxiliary)``
    with flat zero-order ``(E,(L+1)*C)``, nonzero ``(E,2,w_m*C)``, and
    auxiliary ``(E,A)``. The optimized operator differentiates raw products
    through double backward and treats both grid tables as constants.
    """
    edges = raw[0].shape[0]
    prefix = auxiliary_channels + 3 * channels
    if physical_input:
        scalars = raw[0][:, :prefix]
        coefficients = merge_sectors(
            (raw[0][:, prefix:], *raw[1:]),
            lmax=lmax, mmax=mmax, channels=2 * channels,
        )
    else:
        coefficients, scalars = recombine_sectors(
            raw, lmax=lmax, mmax=mmax, channels=2 * channels, extra_channels=prefix,
        )
    a, b = coefficients.chunk(2, dim=-1)
    grid_a = torch.einsum("gr,erc->egc", to_grid, a)
    grid_b = torch.einsum("gr,erc->egc", to_grid, b)
    product = torch.einsum("rg,egc->erc", from_grid, grid_a * grid_b)
    scalar_a, scalar_b, gate = scalars[:, auxiliary_channels:].split(channels, dim=-1)
    product = product * gate.sigmoid().unsqueeze(1)
    scalar = F.silu(scalar_a) * scalar_b
    physical = torch.cat((product[:, :1] + scalar.unsqueeze(1), product[:, 1:]), dim=1)
    sectors = [physical[:, : lmax + 1].reshape(edges, (lmax + 1) * channels)]
    start = lmax + 1
    for order in range(1, mmax + 1):
        width = lmax + 1 - order
        sectors.append(
            physical[:, start : start + 2 * width].reshape(edges, 2, width * channels)
        )
        start += 2 * width
    return tuple(sectors), scalars[:, :auxiliary_channels]


def recombine_activate(
    raw, *, lmax, mmax, activation="gate", auxiliary_channels=0,
    to_grid=None, from_grid=None,
):
    edges = raw[0].shape[0]
    width = raw[0].shape[-1] - auxiliary_channels
    if activation == "gate":
        channels = width // (2 * lmax + 1)
        gates = raw[0][:, auxiliary_channels:auxiliary_channels + lmax * channels]
        coefficients = raw[0][:, auxiliary_channels + lmax * channels:].reshape(edges, lmax + 1, channels)
        result = activate_gated(gates, coefficients, raw[1:], lmax=lmax, mmax=mmax)
        return result, raw[0][:, :auxiliary_channels]
    if activation == "grid":
        channels = width // (2 * (lmax + 1) + 3)
        return activate_grid(
            raw, to_grid, from_grid, lmax=lmax, mmax=mmax,
            channels=channels, auxiliary_channels=auxiliary_channels,
        )
    raise ValueError("activation must be 'gate' or 'grid'")


def recombine_sectors(raw, *, lmax, mmax, channels, extra_channels):
    """Materialize complex recombination and return ``(coefficients, extras)``.

    ``raw[0]`` is ``(E, A+(L+1)*C)``. At order ``m>0`` reshape the raw
    ``(E, 2, 2*w_m*C)`` product to ``q[E, plane, map, degree, channel]``.
    The physical positive plane is ``q[:,0,0] - q[:,1,1]`` and the
    negative plane is ``q[:,1,0] + q[:,0,1]``. The map index selects the
    real and imaginary weight maps. Return coefficients ``(E, R, C)``
    in the row order of ``merge_sectors`` and leading extras ``(E, A)``.
    """
    edges = raw[0].shape[0]
    extras = raw[0][:, :extra_channels]
    physical = [raw[0][:, extra_channels:].reshape(edges, lmax + 1, channels)]
    for order in range(1, mmax + 1):
        width = lmax + 1 - order
        q = raw[order].reshape(edges, 2, 2, width, channels)
        positive = q[:, 0, 0] - q[:, 1, 1]
        negative = q[:, 1, 0] + q[:, 0, 1]
        physical.append(torch.cat((positive, negative), dim=1))
    return merge_sectors(physical, lmax=lmax, mmax=mmax, channels=channels), extras
