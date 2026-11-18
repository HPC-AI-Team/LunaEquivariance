"""Convolution workloads with normal graph sizes and valid rotations."""

import torch
from e3nn import o3

from LunaEquivariance import make_graph, so2
from LunaEquivariance.so2.utils.layout import get_angular_layout
from tests.so2.reference.rotation import rotation
from tests.so2.bench_transforms import spherical_grid


def problem(name, lmax, variant, index, device, small):
    generator = torch.Generator(device=device).manual_seed(920 + index)
    nodes, edges, channels = (7, 19, 4) if small else (2083, 85876, 64)
    # Grid products retain two G*C fields in the PyTorch reference.
    # Use the same smaller graph for both implementations on 40 GB GPUs.
    if variant == "grid" and not small:
        nodes, edges = 512, 16384
    mmax = 2
    metadata = {"operator": name, "nodes": nodes, "edges": edges,
                "lmax": lmax, "mmax": mmax, "channels": channels}
    kwargs = {"lmax": lmax, "mmax": mmax}

    def parameter(shape, scale=0.3):
        values = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return (values * scale).requires_grad_()

    if name in ("rotate_gather", "inverse_rotate_reduce"):
        src = torch.randint(nodes, (edges,), device=device, generator=generator)
        dst = torch.sort(torch.randint(nodes, (edges,), device=device, generator=generator))[0]
        graph = make_graph(src, dst, nodes)
        angles = torch.rand(3, edges, device=device, dtype=torch.float32, generator=generator)
        frames = o3.angles_to_matrix(angles[0] * 6.28,
                                    0.25 + angles[1] * 2.6, angles[2] * 6.28)
        with torch.no_grad():
            rotations = rotation(frames, lmax=lmax, mmax=mmax).detach().requires_grad_()

    if name == "rotate_gather":
        sharing = variant or "order_degree"
        x = parameter((nodes, (lmax + 1)**2, channels))
        shape = ((edges, lmax + 1, 2 * channels) if sharing == "degree" else
                 (edges, 2 * channels * sum(lmax + 1 - m for m in range(mmax + 1))))
        radial = parameter(shape)
        args = x, rotations, radial, graph
        kwargs["radial_sharing"] = sharing
        metadata["variant"] = sharing
    elif name == "linear":
        cin, cout = 2 * channels, channels
        sectors = [parameter((edges, (lmax + 1) * cin))]
        sectors += [parameter((edges, 2, (lmax + 1 - m) * cin)) for m in range(1, mmax + 1)]
        weights = [parameter(((lmax + 1) * cout, (lmax + 1) * cin),
                             ((lmax + 1) * cin)**-0.5)]
        weights += [parameter((2 * (lmax + 1 - m) * cout, (lmax + 1 - m) * cin),
                              ((lmax + 1 - m) * cin)**-0.5) for m in range(1, mmax + 1)]
        args = tuple(sectors), tuple(weights), parameter(((lmax + 1) * cout,), 0.02)
        metadata.update(input_channels=cin, output_channels=cout, variant="paired_gemm")
    else:
        activation = variant or "gate"
        grid = name == "recombine_activate" and activation == "grid"
        width = 2 * channels if grid else channels
        extras = ((3 * channels if grid else lmax * channels)
                  if name == "recombine_activate" else 0)
        raw = [parameter((edges, extras + (lmax + 1) * width))]
        raw += [parameter((edges, 2, 2 * (lmax + 1 - m) * width)) for m in range(1, mmax + 1)]
        if name == "recombine_activate":
            args = (tuple(raw),)
            kwargs["activation"] = activation
            metadata["variant"] = activation
            if grid:
                forward, inverse = spherical_grid(lmax, device)
                rows = get_angular_layout(lmax=lmax, mmax=mmax, channels=1).dense_rows
                kwargs.update(to_grid=forward[:, rows].contiguous(),
                              from_grid=inverse[list(rows)].contiguous())
                metadata["grid_resolution"] = [2 * (lmax + 3)] * 2
        else:
            args = tuple(raw), rotations, graph
            edge_weight = (torch.rand(edges, device=device, generator=generator) + 0.5).requires_grad_()
            kwargs["edge_weight"] = edge_weight
            metadata["variant"] = variant or "sum"
            if variant == "attention":
                heads = 2 if small else 8
                kwargs["logits"] = parameter((edges, heads))
                metadata["heads"] = heads
    return args, kwargs, metadata


def cases(device="cuda", small=False, operators=None):
    variants = [(name, degree, None) for name in (
        "rotate_gather", "linear", "recombine_activate", "inverse_rotate_reduce"
    ) for degree in (2, 4, 6)]
    variants += [("rotate_gather", 4, "degree"),
                 ("recombine_activate", 4, "grid"),
                 ("inverse_rotate_reduce", 4, "attention")]
    for index, (name, lmax, variant) in enumerate(variants):
        if operators and name not in operators:
            continue
        args, kwargs, metadata = problem(name, lmax, variant, index, device, small)
        yield f"{name}-l{lmax}-{metadata['variant']}", getattr(so2, name), args, kwargs, metadata
        del args, kwargs
