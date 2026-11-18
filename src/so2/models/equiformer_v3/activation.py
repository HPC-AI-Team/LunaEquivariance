"""Bind EquiformerV3 channel maps to spherical grid activation."""

from functools import cache
from math import sqrt
from weakref import WeakKeyDictionary

import torch
from LunaEquivariance.so2 import recombine_activate
from LunaEquivariance.so2.linear.interface import project_recombined_sectors
from LunaEquivariance.so2.recombine_activate.interface import grid_activation
from LunaEquivariance.so2.recombine_activate.kernels import utils as grid_kernels

from .linear import project_raw_sectors


prepare_projected_product = getattr(grid_kernels, "prepare_projected_product", None)
projected_grids = WeakKeyDictionary()


@cache
def stock_edge_grid():
    from e3nn.o3 import FromS2Grid, ToS2Grid

    options = dict(normalization="component", dtype=torch.float32, device="cpu")
    with torch.inference_mode(False):
        to = ToS2Grid(4, (14, 8), **options)
        back = FromS2Grid((14, 8), 4, **options)
        rows = [(degree, order) for order in (0, 1, -1, 2, -2) for degree in range(abs(order), 5)]
        degree, order = torch.tensor(rows, device="cpu").T
        index = degree * (degree + 1) + order
        scale = torch.tensor(
            [sqrt((2 * l + 1) / 5) if l > 2 else 1 for l, _ in rows], dtype=torch.float32, device="cpu",
        )
        tables = []
        for transform in (to, back):
            latitude = transform.shb[order + 4, :, index].T
            longitude = transform.sha[:, order + 4]
            tables.append((latitude[:, None, :] * longitude[None, :, :] * scale).reshape(112, 19))
    return tables[0], tables[1].T


@cache
def stock_projected_product(device):
    product = prepare_projected_product(*stock_edge_grid())
    if product is not None:
        coefficients, polynomial = product
        with torch.inference_mode(False):
            return coefficients, polynomial.to(device)


def edge_grid_product(module, edges, to_grid, from_grid):
    if (
        prepare_projected_product is None
        or not 16384 <= edges <= 32768
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.cuda.get_device_capability(to_grid.device) != (9, 0)
        or any(value.requires_grad or value.is_inference() for value in (to_grid, from_grid))
    ):
        return None
    with torch.cuda.device(to_grid.device):
        if torch.cuda.is_current_stream_capturing():
            return None
    versions = tuple((value._version, value.data_ptr()) for value in (to_grid, from_grid))
    saved = projected_grids.get(module)
    if saved is not None and saved[0] is to_grid and saved[1] is from_grid and saved[2] == versions:
        return saved[3]
    product = None
    stock_to, stock_from = stock_edge_grid()
    if torch.equal(to_grid.cpu(), stock_to) and torch.equal(from_grid.cpu(), stock_from):
        product = stock_projected_product(to_grid.device)
    projected_grids[module] = to_grid, from_grid, versions, product
    return product


def validate_linear_grid_activation(module):
    if not 0 <= module.mmax <= module.lmax:
        raise ValueError("grid activation requires 0 <= M <= L")
    hidden = module.num_out_channels
    if hidden <= 0 or hidden % 2:
        raise ValueError("grid product requires two equal channel groups")
    if module.extra_m0_out_channels is None or module.extra_m0_out_channels < 3 * (
        hidden // 2
    ):
        raise ValueError(
            "grid activation requires scalar product and gate side channels"
        )


def use_recombined_projection(module, sectors, to_grid, from_grid):
    if (torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
            or torch.is_autocast_enabled("cuda") or torch.are_deterministic_algorithms_enabled()):
        return False
    if (module.lmax, module.mmax, module.num_in_channels, module.num_out_channels,
            module.extra_m0_out_channels) != (4, 2, 256, 64, 608) or len(sectors) != 3:
        return False
    edges = sectors[0].shape[0]
    if not 8192 <= edges <= 32768:
        return False
    if tuple(sector.shape for sector in sectors) != (
        (edges, 1280), (edges, 2, 1024), (edges, 2, 768),
    ) or to_grid.shape != (112, 19) or from_grid.shape != (19, 112):
        return False
    device = sectors[0].device
    values = (*sectors, to_grid, from_grid)
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) in ((8, 0), (9, 0))
        and all(value.dtype == torch.float32 and value.device == device for value in values)
        and all(sector.is_contiguous() for sector in sectors)
    )


def linear_grid_activation(
    module,
    sectors: tuple[torch.Tensor, ...],
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Project each angular order and consume raw products in the activation."""

    lmax, mmax = module.lmax, module.mmax
    validate_linear_grid_activation(module)
    channels = module.num_out_channels // 2
    auxiliary_channels = module.extra_m0_out_channels - 3 * channels
    if use_recombined_projection(module, sectors, to_grid, from_grid):
        projected = project_recombined_sectors(
            sectors, (module.fc_m0.weight, *(layer.fc.weight for layer in module.so2_m_linear)),
            module.fc_m0.bias,
        )
        product = edge_grid_product(module, projected[0].shape[0], to_grid, from_grid)
        return grid_activation(
            projected, to_grid, from_grid, lmax=lmax, mmax=mmax, channels=channels,
            auxiliary_channels=auxiliary_channels, physical_input=True,
            **({"projected": product} if product is not None else {}),
        )

    raw = project_raw_sectors(module, sectors)
    return recombine_activate(
        raw, activation="grid",
        to_grid=to_grid, from_grid=from_grid,
        lmax=lmax,
        mmax=mmax,
        auxiliary_channels=auxiliary_channels,
    )
