"""Weight-side reassociation for the EqV3 grid feed-forward network."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache, partial
from math import isqrt
from types import MethodType
from typing import Callable, Iterator, Sequence
from weakref import WeakKeyDictionary

import torch
from LunaEquivariance.so2 import feed_forward
from LunaEquivariance.so2.feed_forward import interface as grid_ops


grid_feed_forward = getattr(grid_ops, "grid_feed_forward", None)


grid_factor_cache = WeakKeyDictionary()


@lru_cache(maxsize=1)
def stock_grid_basis():
    from e3nn.o3 import FromS2Grid, ToS2Grid

    with torch.inference_mode(False):
        to = ToS2Grid(4, (14, 14), normalization="component", dtype=torch.float32, device="cpu")
        back = FromS2Grid((14, 14), 4, normalization="component", dtype=torch.float32, device="cpu")
        rows = torch.arange(25, device="cpu")
        degree = torch.tensor([isqrt(row) for row in range(25)], device="cpu")
        mode = rows - degree * (degree + 1) + 4
        alpha = to.sha.contiguous()
        beta_to = to.shb[mode, :, rows].T.contiguous()
        beta_from = back.shb[mode, :, rows].T.contiguous()
        longitude = alpha.index_select(1, mode)
        to_grid = (beta_to[:, None, :] * longitude[None, :, :]).view(196, 25)
        from_grid = (beta_from[:, None, :] * longitude[None, :, :]).view(196, 25).T
    return (to_grid, from_grid), (alpha, beta_to, beta_from, mode.int())


@lru_cache(maxsize=8)
def stock_grid_factors(device):
    with torch.inference_mode(False):
        factors = stock_grid_basis()[1]
        compact = getattr(grid_ops, "compact_grid_factors", None)
        if compact is not None:
            alpha, _, _, row_mode = factors
            factors = compact(alpha, row_mode)
        return tuple(value.to(device) for value in factors)


def grid_factors(grid, inputs):
    if (
        torch.compiler.is_compiling()
        or not 512 <= inputs.shape[0] <= 1024
        or torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.is_autocast_enabled("cuda")
        or torch.are_deterministic_algorithms_enabled()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(inputs.device) != (9, 0)
    ):
        return ()
    to, back = grid.to_grid_mat, grid.from_grid_mat
    if any(value.device != inputs.device or value.dtype != torch.float32 or value.requires_grad
           or value.is_inference() for value in (to, back)):
        return ()
    with torch.cuda.device(inputs.device):
        if torch.cuda.is_current_stream_capturing():
            return ()
    versions = to._version, back._version, to.device, to.data_ptr(), back.data_ptr()
    cached = grid_factor_cache.get(grid)
    if cached is not None and cached[0] is to and cached[1] is back and cached[2] == versions:
        return cached[3]

    stock, _ = stock_grid_basis()
    factors = ()
    if torch.equal(to.cpu(), stock[0]) and torch.equal(back.cpu(), stock[1]):
        factors = stock_grid_factors(inputs.device)
        plan = getattr(grid_ops, "CanonicalGridFactors", None)
        if plan is not None and factors[1].shape == (7, 25):
            factors = plan.bind(to, back, factors)
    grid_factor_cache[grid] = to, back, versions, factors
    return factors


def validate_grid_ffn(ffn):
    if (
        ffn.num_hidden_channels <= 0
        or ffn.num_out_channels != ffn.num_in_channels
        or ffn.activation != "sep-merge_gates2_swiglu"
        or not ffn.use_grid_mlp
    ):
        raise ValueError("unexpected reassociated EqV3 grid-FFN contract")
    grid_mlp = ffn.grid_mlp
    required = ("gating_linear", "grid_linear_1", "grid_linear_2", "grid_drop")
    if not all(hasattr(grid_mlp, name) for name in required):
        raise ValueError("unexpected EqV3 grid-MLP contract")
    if not isinstance(grid_mlp.grid_drop, torch.nn.Identity):
        raise ValueError("the reassociated grid FFN requires zero dropout")
    if (
        ffn.so3_linear_1.bias is None
        or ffn.so3_linear_2.bias is None
        or grid_mlp.grid_linear_1.bias is not None
        or grid_mlp.grid_linear_2.bias is not None
    ):
        raise ValueError("unexpected grid-FFN bias contract")


def reassociated_grid_ffn(ffn: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run an EqV3 grid FFN with both grid-channel linears moved to weights.

    The SO(3) transforms mix representation components while the grid MLP
    linears mix channels.  These actions commute.  Composing their weights
    moves channel work from grid points to degree-weight products while
    retaining cuBLAS for every matrix multiply.
    """

    if (
        inputs.ndim != 3
        or inputs.shape[1] != (ffn.lmax + 1) ** 2
        or inputs.shape[2] != ffn.num_in_channels
    ):
        raise ValueError("unexpected EqV3 grid-FFN input shape")
    if inputs.dtype != torch.float32 or not inputs.is_cuda:
        raise ValueError("the reassociated grid FFN requires CUDA FP32")

    first = ffn.so3_linear_1
    second = ffn.so3_linear_2
    grid_mlp = ffn.grid_mlp
    scalar = ffn.scalar_mlp[0].linear
    factors = ()
    if grid_feed_forward is not None and ffn.num_hidden_channels == 512:
        factors = grid_factors(ffn.so3_grid, inputs)
    run = feed_forward
    if factors:
        index = grid_ops.degree_index(inputs.shape[1], inputs.device)
        run = partial(grid_feed_forward, index=index, factors=factors)
    return run(
        inputs, first.weight, second.weight, grid_mlp.gating_linear.weight,
        input_bias=first.bias, output_bias=second.bias,
        gate_bias=grid_mlp.gating_linear.bias,
        to_grid=ffn.so3_grid.to_grid_mat, from_grid=ffn.so3_grid.from_grid_mat,
        grid_input_weight=grid_mlp.grid_linear_1.weight,
        grid_output_weight=grid_mlp.grid_linear_2.weight,
        scalar_branch=(scalar.weight, scalar.bias),
    )


def install_grid_ffns(
    model: torch.nn.Module,
    block_indices: Sequence[int],
) -> Callable[[], None]:
    """Install the weight-reassociated grid FFN in selected EqV3 blocks."""

    indices = tuple(block_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("block indices must be a non-empty unique sequence")
    if min(indices) < 0 or max(indices) >= len(model.blocks):
        raise ValueError(f"block index outside [0, {len(model.blocks)})")
    blocks = [model.blocks[index] for index in indices]
    for block in blocks:
        validate_grid_ffn(block.ffn)
        if block.drop_path is None or block.drop_path.drop_prob != 0.0:
            raise ValueError("the common EqV3 schedule requires zero drop path")
    old_ffn_forwards = [block.ffn.__dict__.get("forward") for block in blocks]
    old_drop_paths = [block.drop_path for block in blocks]
    for block in blocks:
        block.ffn.forward = MethodType(reassociated_grid_ffn, block.ffn)
        block.drop_path = None

    def restore():
        for block, old_forward, drop_path in zip(
            blocks,
            old_ffn_forwards,
            old_drop_paths,
            strict=True,
        ):
            if old_forward is None:
                del block.ffn.forward
            else:
                block.ffn.forward = old_forward
            block.drop_path = drop_path

    return restore


@contextmanager
def reassociated_grid_ffn_blocks(
    model: torch.nn.Module, block_indices: Sequence[int]
) -> Iterator[None]:
    restore = install_grid_ffns(model, block_indices)
    try:
        yield
    finally:
        restore()
