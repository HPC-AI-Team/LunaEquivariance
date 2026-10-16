"""Common small workloads; GPU tests use the same public facade as applications."""

import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct


@pytest.fixture
def make_problem():
    def build(mode, shared=False, dtype=torch.float32, device="cuda"):
        device = torch.device(device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(731)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(731)
            old = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                ir = Irreps("32x0e + 32x1o")
                edge = ir if mode.startswith("uuu") else Irreps("1x0e + 1x1o")
                instructions = [
                    (i, j, k, mode[:3], True)
                    for i, (_, a) in enumerate(ir)
                    for j, (_, b) in enumerate(edge)
                    for k, (_, c) in enumerate(ir)
                    if c in a * b
                ]
                tp = TensorProduct(
                    ir,
                    edge,
                    ir,
                    instructions,
                    internal_weights=shared,
                    shared_weights=shared,
                ).to(device)
            finally:
                torch.set_default_dtype(old)
            conv = mode.endswith("conv")
            nodes, items = (7, 19) if conv else (19, 19)
            x = (
                torch.randn(ir.dim, nodes, device=device, dtype=dtype)
                .t()
                .requires_grad_()
            )
            y = (
                torch.randn(edge.dim, items, device=device, dtype=dtype)
                .t()
                .requires_grad_()
            )
            w = (
                tp.weight
                if shared
                else torch.randn(
                    items,
                    tp.weight_numel,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            src = torch.randint(nodes, (items,), device=device)
            dst = torch.sort(torch.randint(nodes, (items,), device=device))[0]
            return tp, x, y, w, src, dst

    return build


def pytest_collection_modifyitems(items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(
            reason="CUDA device required; use the GPU workflow for numerical kernels"
        )
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def seed_fixture():
    torch.manual_seed(42)


@pytest.fixture
def rotation(seed_fixture):
    from e3nn import o3

    return o3.rand_matrix(dtype=torch.float64)
