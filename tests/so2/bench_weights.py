"""Representative radial, expert-linear and attention-score workloads."""

import math

import torch

from LunaEquivariance.so2 import attention_scores, radial_network, segmented_linear


def radial_problem(device, small, grouped, indexed, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    nodes, edges = (17, 41) if small else (2083, 85876)
    distance, embedding, hidden, output = (8, 4, 8, 12) if small else (64, 32, 128, 128)
    groups = 4 if grouped else 1
    prefix = (groups,) if grouped else ()
    inputs_width = distance + 2 * embedding if indexed else distance

    def parameter(shape, scale=0.1, offset=0.0):
        value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        return (value * scale + offset).requires_grad_()

    inputs = parameter((edges, distance), scale=0.3)
    weights = tuple(parameter((*prefix, target, source), scale=1 / math.sqrt(source))
                    for source, target in ((inputs_width, hidden), (hidden, hidden), (hidden, output)))
    biases = tuple(parameter((*prefix, width), scale=0.02) for width in (hidden, hidden, output))
    norm_weights = tuple(parameter((*prefix, hidden), scale=0.02, offset=1.0) for _ in range(2))
    norm_biases = tuple(parameter((*prefix, hidden), scale=0.02) for _ in range(2))
    kwargs = {}
    if indexed:
        elements = 10 if small else 100
        kwargs = {
            "atom_types": torch.randint(elements, (nodes,), generator=generator, device=device),
            "edge_index": torch.randint(nodes, (2, edges), generator=generator, device=device),
            "embeddings": tuple(parameter((*prefix, elements, embedding), scale=0.3) for _ in range(2)),
        }
    metadata = {
        "operator": "radial_network", "nodes": nodes, "edges": edges,
        "networks": groups, "indexed": indexed, "input_channels": distance,
        "embedding_channels": embedding if indexed else 0,
        "hidden_channels": hidden, "output_channels": output, "dtype": "float32",
    }
    return (inputs, weights, biases, norm_weights, norm_biases), kwargs, metadata


def segmented_problem(device, small, channels, paired, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    edges = 41 if small else 85876
    channels = (12 if paired else 8) if small else channels
    experts, systems = 8, 4

    def parameter(shape, scale=0.1):
        value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        return (value * scale).requires_grad_()

    shape = (edges, 2, channels) if paired else (edges, channels)
    inputs = parameter(shape, scale=0.3)
    weights = parameter((experts, channels, channels), scale=1 / math.sqrt(channels))
    mixing = torch.randn(systems, experts, generator=generator, device=device,
                         dtype=torch.float32).softmax(-1).requires_grad_()
    bias = parameter((channels,), scale=0.02)
    size, remainder = divmod(edges, systems)
    sizes = tuple(size + (index < remainder) for index in range(systems))
    metadata = {
        "operator": "segmented_linear", "edges": edges, "systems": systems,
        "experts": experts, "channels": channels, "components": 2 if paired else 1,
        "dtype": "float32",
    }
    return (inputs, weights, mixing, bias, sizes), {}, metadata


def attention_problem(device, small, width, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    edges, heads = (41, 2) if small else (85876, 8)
    width = width // 4 if small else width

    def parameter(shape, scale=0.1, offset=0.0):
        value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        return (value * scale + offset).requires_grad_()

    inputs = parameter((edges, heads, width), scale=0.3)
    weight = parameter((width,), scale=0.02, offset=1.0)
    bias = parameter((width,), scale=0.02)
    projection = parameter((heads, width), scale=1 / math.sqrt(width))
    metadata = {
        "operator": "attention_scores", "edges": edges, "heads": heads,
        "head_channels": width, "dtype": "float32",
    }
    return (inputs, weight, bias, projection), {}, metadata


def cases(device="cuda", small=False, operators=None):
    """Yield one workload at a time, with external trainable tensor parameters."""
    for index, (grouped, indexed) in enumerate(((False, False), (True, False),
                                               (False, True), (True, True))):
        if operators and "radial_network" not in operators:
            break
        name = "radial_" + ("indexed" if indexed else "plain") + ("_grouped" if grouped else "")
        args, kwargs, metadata = radial_problem(device, small, grouped, indexed, 731 + index)
        yield name, radial_network, args, kwargs, metadata
        del args, kwargs, metadata
    for index, (channels, paired) in enumerate(((64, False), (128, True))):
        if operators and "segmented_linear" not in operators:
            break
        args, kwargs, metadata = segmented_problem(device, small, channels, paired, 741 + index)
        name = "segmented_" + ("paired" if paired else "matrix")
        yield name, segmented_linear, args, kwargs, metadata
        del args, kwargs, metadata
    for index, width in enumerate((16, 32)):
        if operators and "attention_scores" not in operators:
            break
        args, kwargs, metadata = attention_problem(device, small, width, 751 + index)
        yield f"attention_width{width}", attention_scores, args, kwargs, metadata
        del args, kwargs, metadata
