"""PyTorch inverse rotation, edge weighting and receiver reduction."""

import torch

from tests.so2.reference.rotation import dense_rotation_rows, unpack_retained_rotations


def recombined_messages(messages, lmax, mmax, raw_complex):
    """Return m-major physical coefficients, using explicit complex products."""
    edges, _, channels = messages[0].shape
    physical = [messages[0]]
    for order in range(1, mmax + 1):
        if raw_complex:
            width = lmax + 1 - order
            # Axes are input plane, weight-map output, degree, channel.
            products = messages[order].reshape(edges, 2, 2, width, channels)
            positive = products[:, 0, 0] - products[:, 1, 1]
            negative = products[:, 1, 0] + products[:, 0, 1]
            physical.append(torch.cat((positive, negative), dim=1))
        else:
            physical.append(messages[order])
    return torch.cat(physical, dim=1)


def aggregate_sum(
    messages, envelope, wigner, receivers, nodes, *, lmax, mmax,
    row_ptr=None, raw_complex=False, rotation_storage=None, wigner_mmax=None,
):
    """Reference for :func:`LunaEquivariance.so2.inverse_rotate_reduce.interface.sum_aggregation`.

    Recombine raw products when requested, apply the transpose of each
    retained forward-rotation block, multiply the envelope once, and sum into
    receivers. ``row_ptr`` describes scheduling only. ``wigner_mmax`` can
    exceed ``mmax`` when the same rotation storage serves several consumers.
    """
    del row_ptr
    local = recombined_messages(messages, lmax, mmax, raw_complex)
    if rotation_storage == "retained_rows":
        stored_mmax = mmax if wigner_mmax is None else wigner_mmax
        rows = unpack_retained_rotations(wigner, lmax=lmax, mmax=stored_mmax)
        rows = rows[:, : local.shape[1]]
    elif rotation_storage in (None, "dense"):
        rows = dense_rotation_rows(wigner, lmax, mmax)
    else:
        raise ValueError("unknown rotation storage")
    edge_messages = torch.bmm(rows.transpose(1, 2), local)
    edge_messages = edge_messages * envelope.reshape(-1, 1, 1)
    output = local.new_zeros(nodes, (lmax + 1) ** 2, local.shape[-1])
    return output.index_add(0, receivers, edge_messages)


def aggregate_attention(
    values, logits, envelope, wigner_inv,
    receivers, row_ptr, nodes, eps, *, lmax, mmax,
):
    """Reference for :func:`LunaEquivariance.so2.inverse_rotate_reduce.interface.attention_aggregation`.

    ``values`` is packed m-major coefficients or a tuple of raw matrix-product
    sectors. Attention uses a detached per-receiver maximum for stability.
    The envelope appears in the normalization and again on the weighted values.
    ``wigner_inv`` stores inverse-map columns. Decoding those vectors as rows
    and transposing recovers the rectangular inverse map, including any scale
    already present in its entries. ``row_ptr`` has no mathematical effect.
    """
    del row_ptr
    local = (
        values
        if isinstance(values, torch.Tensor)
        else recombined_messages(values, lmax, mmax, raw_complex=True)
    )
    edges, components, channels = local.shape
    heads = logits.shape[1]
    envelope = envelope.reshape(edges, 1)
    index = receivers[:, None].expand(edges, heads)
    maxima = (
        logits.new_full((nodes, heads), -torch.inf)
        .scatter_reduce(0, index, logits, reduce="amax", include_self=True)
        .detach()
    )
    shifted = torch.exp(logits - maxima[receivers])
    unnormalized = shifted * envelope
    denominator = logits.new_zeros(nodes, heads).index_add(0, receivers, unnormalized)
    probability = unnormalized / (denominator[receivers] + eps)
    weight = probability * envelope
    weighted = local.reshape(edges, components, heads, channels // heads)
    weighted = (weighted * weight[:, None, :, None]).reshape(edges, components, channels)
    inverse_columns = unpack_retained_rotations(
        wigner_inv, lmax=lmax, mmax=mmax,
    ).transpose(1, 2)
    edge_messages = torch.bmm(inverse_columns, weighted)
    output = local.new_zeros(nodes, (lmax + 1) ** 2, channels)
    return output.index_add(0, receivers, edge_messages)


def scalar_edge_to_node(
    radial, envelope, wigner, receivers, row_ptr, rescale, *, lmax, mmax,
):
    """Reference for :func:`LunaEquivariance.so2.inverse_rotate_reduce.scalar_edge_to_node`.

    Each degree has one local m=0 radial coefficient. Multiply it by the
    corresponding inverse-rotation column, weight by the envelope, sum into
    the receiving nodes, and divide by the constant ``rescale``. Other local
    orders stored in ``wigner`` do not enter this scalar embedding.
    """
    nodes = row_ptr.numel() - 1
    zero_order_rows = unpack_retained_rotations(
        wigner,
        lmax=lmax,
        mmax=mmax,
    )[:, : lmax + 1]
    edge_messages = torch.bmm(zero_order_rows.transpose(1, 2), radial)
    edge_messages = edge_messages * envelope.reshape(-1, 1, 1)
    output = radial.new_zeros(nodes, (lmax + 1) ** 2, radial.shape[-1])
    return output.index_add(0, receivers, edge_messages) / rescale


def inverse_rotate_reduce(
    raw, rotations, graph, *, lmax, mmax, edge_weight=None, logits=None, eps=1e-8
):
    edges = graph.src.numel()
    channels = raw[0].shape[-1] if raw[0].ndim == 3 else raw[0].shape[-1] // (lmax + 1)
    messages = raw if raw[0].ndim == 3 else (raw[0].reshape(edges, lmax + 1, channels), *raw[1:])
    widths = tuple(sum((2 * min(l, m) + 1) * (2 * l + 1) for l in range(lmax + 1)) for m in range(lmax + 1))
    stored_mmax = widths.index(rotations.shape[1])
    if edge_weight is None:
        edge_weight = raw[0].new_ones(edges)
    receivers = graph.dst.to(torch.int64)
    if logits is None:
        return aggregate_sum(
            messages, edge_weight, rotations, receivers, graph.num_nodes,
            lmax=lmax, mmax=mmax, raw_complex=True,
            rotation_storage="retained_rows", wigner_mmax=stored_mmax,
        )
    if stored_mmax != mmax:
        raise ValueError("attention reduction requires matching rotation and message orders")
    return aggregate_attention(
        messages, logits, edge_weight, rotations, receivers, graph.dst_rowptr,
        graph.num_nodes, eps, lmax=lmax, mmax=mmax,
    )


def fused_initial_atom(node_features, weight, atomic_numbers):
    """Add weight[atomic_numbers] to coefficient zero of (N, rows, C) in place."""
    node_features[:, 0] = node_features[:, 0] + weight[atomic_numbers]
    return node_features


def fused_initial_scalar(node_features, scalar_features):
    """Add (N, C) scalar features to coefficient zero in place."""
    node_features[:, 0] = node_features[:, 0] + scalar_features
    return node_features
