"""PyTorch order-wise projection and physical sector assembly."""

from math import prod

import torch
from torch.nn import functional as F


def native_pair_linear(channel_map, paired):
    edges = paired.shape[0]
    projected = channel_map(paired.reshape(-1, paired.shape[-1]))
    return projected.reshape(edges, 2, projected.shape[-1])


def raw_sector_linear(sectors, scalar_map, order_maps, *, apply_pair=native_pair_linear):
    """Apply each caller-owned map, preserving the two nonzero-order planes.

    The zero-order input flattens to ``(E, K0)``. Each remaining input
    flattens to ``(E, 2, Km)``. Its map acts on the last axis with shared
    weights for both planes. Return ``(scalar_output, *raw_order_outputs)``.
    Each raw output is ``(E, 2, output_width)``. No real/imaginary products
    are added or subtracted here. ``apply_pair`` can preserve the edge axis
    for edge-conditioned maps instead of flattening it with the plane axis.
    """
    scalar = sectors[0]
    outputs = [scalar_map(scalar.reshape(scalar.shape[0], prod(scalar.shape[1:])))]
    for sector, channel_map in zip(sectors[1:], order_maps, strict=True):
        paired = sector.reshape(sector.shape[0], 2, prod(sector.shape[1:]) // 2)
        outputs.append(apply_pair(channel_map, paired))
    return tuple(outputs)


def project_sectors(
    sectors,
    scalar_map,
    order_maps,
    *,
    output_channels,
    scalar_prefix=0,
    apply_pair=native_pair_linear,
):
    """Project sectors and separate the zero-order scalar prefix.

    Return ``(side, projected)``. If the zero-order map outputs
    ``(E, A + (L+1)*Cout)``, ``side`` is ``(E, A)`` and
    ``projected[0]`` is ``(E, L+1, Cout)``. Other projected sectors keep
    the raw ``(E, 2, 2*w_m*Cout)`` complex-product layout.
    """
    raw = raw_sector_linear(sectors, scalar_map, order_maps, apply_pair=apply_pair)
    side = raw[0][:, :scalar_prefix]
    rows = (raw[0].shape[-1] - scalar_prefix) // output_channels
    scalar = raw[0][:, scalar_prefix:].reshape(raw[0].shape[0], rows, output_channels)
    return side, (scalar, *raw[1:])


def merge_sectors(sectors, *, lmax, mmax, channels):
    """Concatenate physical sectors into an order-major ``(E, R, C)`` tensor.

    ``R = L+1 + 2*sum(L+1-m for m in range(1, M+1))``. Rows are
    ``m=0``, then ``+1, -1, +2, -2, ...``. Inside each signed order,
    degrees increase. Inputs can have flat, logical, or explicit-plane
    shapes described by the optimized ``merge_sectors`` interface.
    """
    edges = sectors[0].shape[0]
    return torch.cat(
        [
            sector.reshape(
                edges, (1 if order == 0 else 2) * (lmax + 1 - order), channels
            )
            for order, sector in enumerate(sectors[: mmax + 1])
        ],
        dim=1,
    )


def linear(sectors, weights, bias=None, *, lmax, mmax):
    """Apply each real channel map to each input plane independently."""
    edges = sectors[0].shape[0]
    output = [F.linear(sectors[0].reshape(edges, prod(sectors[0].shape[1:])), weights[0], bias)]
    for order in range(1, mmax + 1):
        sector = sectors[order]
        planes = sector.reshape(edges, 2, prod(sector.shape[1:]) // 2)
        output.append(torch.stack(tuple(F.linear(planes[:, plane], weights[order]) for plane in (0, 1)), dim=1))
    return tuple(output)
