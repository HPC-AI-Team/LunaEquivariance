"""Materialized PyTorch equations for gather, rotation and radial scaling."""

import torch

from tests.so2.reference.rotation import dense_rotation_rows, unpack_retained_rotations


def node_to_edge(
    x, edge_index, radial, rotations, *,
    lmax, mmax, radial_sharing, rotation_storage, metadata=None,
):
    """Reference for :func:`LunaEquivariance.so2.rotate_gather.node_to_edge`.

    The mathematical order is gather both endpoints, rotate each degree,
    multiply radial weights, and group coefficients by local order. See the
    optimized function for the input and sector contracts. ``metadata`` has
    no mathematical effect, since PyTorch indexing does not require CSR.
    """
    del metadata
    source, receiver = edge_index
    gathered = torch.cat((x[source], x[receiver]), dim=-1)
    if rotation_storage == "retained_rows":
        rows = unpack_retained_rotations(rotations, lmax=lmax, mmax=mmax)
    elif rotation_storage == "dense" and radial_sharing == "order_degree":
        rows = dense_rotation_rows(rotations, lmax, mmax)
    else:
        raise ValueError("unsupported radial sharing and rotation storage combination")
    rotated = torch.bmm(rows, gathered)

    edges, _, endpoint_channels = rotated.shape
    sectors = []
    component_start = radial_start = 0
    for order in range(mmax + 1):
        width = lmax + 1 - order
        components = width if order == 0 else 2 * width
        values = rotated[:, component_start : component_start + components]
        if radial_sharing == "degree":
            weights = radial[:, order : lmax + 1]
        elif radial_sharing == "order_degree":
            weights = radial[:, radial_start : radial_start + width * endpoint_channels]
            weights = weights.reshape(edges, width, endpoint_channels)
            radial_start += width * endpoint_channels
        else:
            raise ValueError("radial_sharing must be 'degree' or 'order_degree'")
        if order:
            # +m and -m use the same radial map at a given degree.
            weights = torch.cat((weights, weights), dim=1)
        values = values * weights
        shape = (
            (edges, width * endpoint_channels)
            if order == 0
            else (edges, 2, width * endpoint_channels)
        )
        sectors.append(values.reshape(shape))
        component_start += components
    return tuple(sectors)


def rotate_gather(
    x, rotations, radial, graph, *, lmax, mmax, radial_sharing="order_degree"
):
    indices = torch.stack((graph.src, graph.dst)).to(torch.int64)
    return node_to_edge(
        x, indices, radial, rotations, lmax=lmax, mmax=mmax,
        radial_sharing=radial_sharing, rotation_storage="retained_rows",
    )
