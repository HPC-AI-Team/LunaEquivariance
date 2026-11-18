"""Regular frame and node workloads for the SO(2) performance checks."""

import math
from itertools import product

import torch
from e3nn import o3

from LunaEquivariance import so2
from LunaEquivariance.so2 import feed_forward
from LunaEquivariance.so2.rotation.interface import quaternion_wigner_from_edges_retained


def spherical_grid(lmax, device):
    """Dense angular maps from e3nn's component-normalized S2 transforms."""
    resolution = 2 * (lmax + 3)
    shape = (resolution, resolution)
    components = (lmax + 1) ** 2
    to_grid = o3.ToS2Grid(lmax, shape, dtype=torch.float64, device="cpu")
    from_grid = o3.FromS2Grid(shape, lmax, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        forward = to_grid(torch.eye(components, dtype=torch.float64))
        inverse = from_grid(torch.eye(resolution**2, dtype=torch.float64).reshape(-1, *shape))
    return (forward.reshape(components, -1).T.contiguous().to(device=device, dtype=torch.float32),
            inverse.T.contiguous().to(device=device, dtype=torch.float32))


def quaternion_basis(device):
    """Construct degree-two quaternion coefficients through the 1 x 1 coupling."""
    from LunaEquivariance.so2.rotation.kernels.utils import (
        QUADRATIC_EXPONENTS,
        QUARTIC_EXPONENTS,
        quadratic_coefficients,
    )
    from LunaEquivariance.so2.rotation.kernels.utils import select_retained_coefficients

    quadratic = quadratic_coefficients(torch.device("cpu")).double()
    powers = QUADRATIC_EXPONENTS[1:]
    homogeneous = quadratic[:, 1:].clone()
    for variable in range(4):
        exponent = tuple(2 if axis == variable else 0 for axis in range(4))
        homogeneous[:, powers.index(exponent)] += quadratic[:, 0]
    coupling = o3.wigner_3j(1, 1, 2, dtype=torch.float64, device="cpu")
    matrix = homogeneous.reshape(3, 3, -1)
    products = 5 * torch.einsum("ija,ikp,jlq,klb->abpq", coupling, matrix, matrix, coupling)
    quartic = torch.zeros(25, len(QUARTIC_EXPONENTS), dtype=torch.float64)
    for first, second in product(range(len(powers)), repeat=2):
        exponent = tuple(a + b for a, b in zip(powers[first], powers[second]))
        quartic[:, QUARTIC_EXPONENTS.index(exponent)] += products[:, :, first, second].reshape(-1)
    return (
        select_retained_coefficients(quadratic.float().to(device), degrees=(1,), mmax=2),
        select_retained_coefficients(quartic.float().to(device), degrees=(2,), mmax=2),
    )


def cases(device="cuda", small=False, operators=None):
    """Yield independent workloads, retaining only the current case's tensors.

    Regular sizes follow Luna's node and edge benchmarks. ``small=True``
    keeps the same computational variants with fewer rows and channels for
    correctness checks. Grid maps and rotations use valid angular geometry.
    """
    nodes, edges, channels = (7, 19, 4) if small else (2083, 85876, 64)
    generator = torch.Generator(device=device)
    for lmax, convention in ((2, "pole_clamped"), (4, "pole_clamped"),
                             (6, "pole_clamped"), (4, "masked")):
        if operators and "rotation" not in operators:
            break
        generator.manual_seed(714 + lmax)
        angles = torch.rand(3, edges, generator=generator, device=device, dtype=torch.float32)
        alpha, gamma = (2 * math.pi * angles[index] - math.pi for index in (0, 2))
        # Pole behavior has separate correctness cases. These frames stay
        # inside the regular Euler chart used for the performance comparison.
        beta = 0.25 + (math.pi - 0.5) * angles[1]
        frames = o3.angles_to_matrix(alpha, beta, gamma).detach().requires_grad_()
        name = f"rotation-l{lmax}-m2-{convention}"
        yield (name, so2.rotation, (frames,),
               {"lmax": lmax, "mmax": 2, "convention": convention},
               {"operator": "rotation", "variant": convention, "lmax": lmax,
                "mmax": 2, "edges": edges, "frame_shape": [edges, 3, 3],
                "beta_interval": [0.25, math.pi - 0.25]})

    if not operators or "rotation" in operators:
        generator.manual_seed(718)
        directions = torch.randn(edges, 3, generator=generator, device=device, dtype=torch.float32)
        directions = (directions / directions.norm(dim=-1, keepdim=True)).requires_grad_()
        yield ("rotation-l2-m2-quaternion", quaternion_wigner_from_edges_retained,
               (directions, *quaternion_basis(device)),
               {"lmax": 2, "mmax": 2}, {"operator": "rotation", "variant": "quaternion", "lmax": 2,
                             "mmax": 2, "edges": edges, "geometry_shape": [edges, 3]})

    variants = ((2, "coefficient"), (4, "coefficient"), (6, "coefficient"),
                (4, "norm_residual"), (4, "grid_scalar"))
    for lmax, variant in variants:
        if operators and "feed_forward" not in operators:
            break
        generator.manual_seed(820 + lmax)
        hidden = 4 * channels if variant == "grid_scalar" else channels
        degree_channels = channels if variant == "grid_scalar" else hidden

        def parameter(shape, scale):
            return (torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
                    * scale).requires_grad_()

        inputs = parameter((nodes, (lmax + 1)**2, channels), 1.0)
        first = parameter((lmax + 1, degree_channels, channels), channels**-0.5)
        second = parameter((lmax + 1, channels, degree_channels), degree_channels**-0.5)
        gate_rows = hidden if variant == "grid_scalar" else lmax * hidden
        gate = parameter((gate_rows, channels), channels**-0.5)
        kwargs = {
            "input_bias": parameter((degree_channels,), 0.02),
            "output_bias": parameter((channels,), 0.02),
            "gate_bias": parameter((gate_rows,), 0.02),
        }
        metadata = {"operator": "feed_forward", "variant": variant,
                    "nodes": nodes, "lmax": lmax, "channels": channels,
                    "hidden_channels": hidden, "degree_channels": degree_channels}
        if variant == "norm_residual":
            kwargs.update(
                norm_weight=torch.ones(lmax + 1, channels, device=device, dtype=torch.float32, requires_grad=True),
                norm_bias=parameter((channels,), 0.02),
                residual=True,
            )
        if variant == "grid_scalar":
            to_grid, from_grid = spherical_grid(lmax, device)
            kwargs.update(
                to_grid=to_grid, from_grid=from_grid,
                grid_input_weight=parameter((2 * hidden, degree_channels), degree_channels**-0.5),
                grid_output_weight=parameter((degree_channels, hidden), hidden**-0.5),
            )
            kwargs["scalar_branch"] = (
                parameter((2 * degree_channels, channels), channels**-0.5),
                parameter((2 * degree_channels,), 0.02),
            )
            metadata["grid_resolution"] = [2 * (lmax + 3)] * 2
            metadata["grid_points"] = to_grid.shape[0]
        yield (f"feed-forward-l{lmax}-{variant}", feed_forward,
               (inputs, first, second, gate), kwargs, metadata)
