"""PyTorch equations for radial MLPs with caller-owned parameter tensors.

Each hidden stage is ``linear -> channel LayerNorm -> SiLU``. The final
stage is linear. These references accept CPU or CUDA floating-point tensors
and use ordinary autograd, including double backward through all parameters.
"""

import torch
from torch.nn import functional as F


def linear_norm_silu(inputs, weight, bias, norm_weight, norm_bias, epsilon):
    hidden = F.linear(inputs, weight, bias)
    normalized = F.layer_norm(
        hidden,
        (weight.shape[0],),
        norm_weight,
        norm_bias,
        epsilon,
    )
    return F.silu(normalized)


def fused_radial_mlp(parameters, inputs):
    """Evaluate two hidden stages and a final linear map.

    ``parameters`` has the fields documented by
    ``LunaEquivariance.so2.radial_network.interface.RadialParameters``. The
    reference accesses only these tensor and epsilon fields. It does not
    import the optimized parameter class or depend on Triton.
    """
    p = parameters
    first = linear_norm_silu(
        inputs,
        p.first_weight,
        p.first_bias,
        p.first_norm_weight,
        p.first_norm_bias,
        p.first_eps,
    )
    second = linear_norm_silu(
        first,
        p.second_weight,
        p.second_bias,
        p.second_norm_weight,
        p.second_norm_bias,
        p.second_eps,
    )
    return F.linear(second, p.final_weight, p.final_bias)


def packed_radial_net(
    edge_distance,
    atomic_numbers,
    edge_index,
    source_embedding_weight,
    target_embedding_weight,
    *,
    first_weight,
    first_bias,
    first_norm_weight,
    first_norm_bias,
    first_norm_eps,
    second_weight,
    second_bias,
    second_norm_weight,
    second_norm_bias,
    second_norm_eps,
    output_weight,
    output_bias,
):
    """Concatenate distance/atom-pair features, then apply the radial MLP.

    The first linear map receives ``[distance_basis, source_atom_embedding,
    receiver_atom_embedding]``. Shapes and channel order match the optimized
    function. Table and edge indices remain integer inputs.
    """
    source, receiver = edge_index
    packed = torch.cat(
        (
            edge_distance,
            source_embedding_weight[atomic_numbers[source]],
            target_embedding_weight[atomic_numbers[receiver]],
        ),
        dim=-1,
    )
    first = linear_norm_silu(
        packed,
        first_weight,
        first_bias,
        first_norm_weight,
        first_norm_bias,
        first_norm_eps,
    )
    second = linear_norm_silu(
        first,
        second_weight,
        second_bias,
        second_norm_weight,
        second_norm_bias,
        second_norm_eps,
    )
    return F.linear(second, output_weight, output_bias)


def radial_network(
    inputs, weights, biases, norm_weights, norm_biases, *, eps=1e-5,
    atom_types=None, edge_index=None, embeddings=None,
):
    """Materialize indexed inputs and evaluate each radial network separately."""
    eps = (eps, eps) if isinstance(eps, (int, float)) else tuple(eps)
    grouped = weights[0].ndim == 3
    groups = weights[0].shape[0] if grouped else 1
    outputs = []
    for index in range(groups):
        select = lambda value: value[index] if grouped and value is not None else value
        hidden = inputs
        if embeddings is not None:
            source, receiver = edge_index
            source_table, receiver_table = (
                table[index] if table.ndim == 3 else table for table in embeddings
            )
            hidden = torch.cat((inputs, source_table[atom_types[source]],
                                receiver_table[atom_types[receiver]]), dim=-1)
        for layer in range(2):
            hidden = F.linear(hidden, select(weights[layer]), select(biases[layer]))
            hidden = F.layer_norm(hidden, (hidden.shape[-1],), select(norm_weights[layer]),
                                  select(norm_biases[layer]), eps[layer])
            hidden = F.silu(hidden)
        outputs.append(F.linear(hidden, select(weights[2]), select(biases[2])))
    return torch.stack(outputs) if grouped else outputs[0]


def fused_radial_input(
    distance, source_weight, target_weight, atomic_numbers, edge_index
):
    """Concatenate distance features and the source/receiver embedding rows."""
    source, receiver = edge_index
    return torch.cat(
        (
            distance,
            source_weight[atomic_numbers[source]],
            target_weight[atomic_numbers[receiver]],
        ),
        dim=-1,
    )
