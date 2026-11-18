"""Small reference fixtures for public dataflow correctness tests."""

from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

TensorTuple = tuple[torch.Tensor, ...]


def complex_reference(raw: torch.Tensor, half: int) -> torch.Tensor:
    real, imaginary = raw.split(half, dim=2)
    return torch.cat(
        (
            real[:, 0:1] - imaginary[:, 1:2],
            real[:, 1:2] + imaginary[:, 0:1],
        ),
        dim=1,
    )


def as_tuple(value: torch.Tensor | Sequence[torch.Tensor]) -> TensorTuple:
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(value)


def clone_leaves(values, *, requires_grad):
    return tuple(
        value.detach().clone().requires_grad_(requires_grad) for value in values
    )


def wigner_block_mask(lmax: int, device: torch.device) -> torch.Tensor:
    components = (lmax + 1) ** 2
    mask = torch.zeros(components, components, device=device)
    for degree in range(lmax + 1):
        start, end = degree * degree, (degree + 1) ** 2
        mask[start:end, start:end] = 1.0
    return mask


def comparison(reference, actual):
    tensors = []
    for expected, observed in zip(reference, actual, strict=True):
        delta = observed.detach().double() - expected.detach().double()
        tensors.append(
            {
                "squared_error": delta.square().sum().item(),
                "reference_squared_norm": expected.detach()
                .double()
                .square()
                .sum()
                .item(),
                "max_abs": delta.abs().max().item() if delta.numel() else 0.0,
                "finite": bool(
                    torch.isfinite(observed).all() and torch.isfinite(expected).all()
                ),
            }
        )
    error2 = sum(row["squared_error"] for row in tensors)
    norm2 = sum(row["reference_squared_norm"] for row in tensors)
    relative = (error2 / max(norm2, 1e-60)) ** 0.5
    return {
        "relative_l2": relative,
        "squared_error": error2,
        "reference_squared_norm": norm2,
        "finite": all(row["finite"] for row in tensors),
        "tensors": tensors,
    }


def synthetic_graph(nodes, neighbors, *, seed, device):
    if not 0 < neighbors < nodes:
        raise ValueError("neighbors must be positive and less than nodes")
    generator = torch.Generator().manual_seed(seed)
    sources, targets = [], []
    for source in range(nodes):
        candidates = torch.cat((torch.arange(source), torch.arange(source + 1, nodes)))
        selected = candidates[
            torch.randperm(nodes - 1, generator=generator)[:neighbors]
        ]
        sources.append(torch.full((neighbors,), source))
        targets.append(selected)
    edges = torch.stack((torch.cat(sources), torch.cat(targets)))
    return edges[:, edges[1].argsort(stable=True)].to(device)


def prefix_outputs(value, spec):
    outputs = []
    for order, width in enumerate(spec.sector_widths):
        sector = value.narrow(
            1, spec.sector_row_offsets[order], width if order == 0 else 2 * width
        )
        outputs.append(
            sector.reshape(value.shape[0], width * 2 * spec.channels)
            if order == 0
            else sector.reshape(value.shape[0], 2, width * 2 * spec.channels)
        )
    return tuple(outputs)


def legal_wigner(edges, lmax, *, seed, device):
    """Build physical rotation blocks on CPU, outside every measured region.

    e3nn versions used by the model environments create SO(3) generators on
    CPU. Chunked CPU construction avoids both device mismatches and a large
    batched matrix-exponential workspace; the finished FP32 tensor moves once.
    """
    from e3nn.o3 import wigner_D

    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(3, edges, generator=generator, dtype=torch.float64)
    alpha, beta, gamma = (
        angles[0] * (2 * torch.pi),
        (2 * angles[1] - 1).acos(),
        angles[2] * (2 * torch.pi),
    )
    result = torch.zeros(edges, (lmax + 1) ** 2, (lmax + 1) ** 2)
    with torch.no_grad():
        for begin in range(0, edges, 1024):
            end = min(begin + 1024, edges)
            for degree in range(lmax + 1):
                first, last = degree**2, (degree + 1) ** 2
                result[begin:end, first:last, first:last] = wigner_D(
                    degree, alpha[begin:end], beta[begin:end], gamma[begin:end]
                ).float()
    return result.to(device)


@pytest.fixture
def dataflow_reference():
    return SimpleNamespace(
        synthetic_graph=synthetic_graph,
        legal_wigner=legal_wigner,
        wigner_block_mask=wigner_block_mask,
        complex_reference=complex_reference,
        prefix_outputs=prefix_outputs,
        comparison=comparison,
        clone_leaves=clone_leaves,
        as_tuple=as_tuple,
    )


@pytest.fixture
def without_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
