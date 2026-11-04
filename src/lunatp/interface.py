"""Tensor-product preparation and functional calls with explicit weights."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from functools import lru_cache, partial
from threading import RLock

import torch
from torch.utils._python_dispatch import _disable_current_modes

from ..o3.irreps import Irreps
from ..utils.operator_config import encode_config
from .utils.layout import (
    Instruction, TensorProductPaths, configure_indexing, prepare_indexing, compiled_config,
)
from .uuu.interface import lunatp_uuu_func
from .uuu.kernels.utils import build_uuu_metadata
from .uuu_conv.interface import lunatp_uuu_conv_func
from .uuu_conv.kernels.utils import build_uuu_conv_metadata, configure_uuu_conv
from .uvu.interface import lunatp_uvu_func
from .uvu.kernels.utils import build_uvu_metadata
from .uvu_conv.interface import lunatp_uvu_conv_func
from .uvu_conv.kernels.utils import build_uvu_conv_metadata, configure_uvu_conv
from .uvw.interface import lunatp_uvw_func
from .uvw.kernels.utils import build_uvw_metadata
from .uvw_conv.interface import lunatp_uvw_conv_func
from .uvw_conv.kernels.utils import (
    FORWARD_SPLIT_CHANNELS, BACKWARD_NEIGHBOR_SPLITS,
    NODE_NEIGHBOR_SPLITS, reset_launch_caches, get_uvw_conv_backward_grid_tasks,
    get_uvw_conv_tr_grid_tasks,
)


BOUND_OPERATORS = {}
OPERATOR_BINDING_LOCK = RLock()


@torch.compiler.assume_constant_result
def register_ops(op):
    """Bind a prepared configuration for model adapters that store an op name."""
    operator_name = f"{op.index_mode}_forward"
    key = operator_name, op._torch_config
    with OPERATOR_BINDING_LOCK:
        name = BOUND_OPERATORS.get(key)
        if name is None:
            name = f"_lunaeq_legacy_{len(BOUND_OPERATORS)}"
            function = partial(getattr(torch.ops.lunatp, operator_name).default,
                               config=op._torch_config)
            function.default = function
            setattr(torch.ops.lunatp, name, function)
            BOUND_OPERATORS[key] = name
        return name


def compiled_operator_config(op):
    """Keep the full static layout in the graph; item counts and topology stay inputs."""
    config = dict(
        mode=op.index_mode, dtype=op.compute_dtype, index_bits=op.index_bits,
        out_dim=op.meta["out_dim"], l0_dim=getattr(op, "l0_dim", 0),
        weight_mode=getattr(op, "weight_mode", "external"),
    )
    if op.index_mode in ("uuu_conv", "uvu_conv"):
        config["meta"] = op.meta
    else:
        tp = op.tp
        config["paths"] = (
            str(tp.irreps_in1), str(tp.irreps_in2), str(tp.irreps_out),
            tuple(tuple(instruction) for instruction in tp.instructions),
            tp.internal_weights, tp.shared_weights, tp.weight_numel,
        )
    return encode_config(config)


@lru_cache(maxsize=128)
def restore_compiled_operator(config):
    settings = compiled_config(config)
    mode, dtype, bits = settings["mode"], settings["dtype"], settings["index_bits"]
    with _disable_current_modes():
        if mode in ("uuu_conv", "uvu_conv"):
            operator = LunaTPUUUConv if mode == "uuu_conv" else LunaTPUVUConv
            return operator(settings["meta"], dtype=dtype, index_bits=bits)
        in1, in2, out, instructions, internal, shared, weights = settings["paths"]
        # Paths are already normalized; restore attributes without normalizing twice.
        paths = TensorProductPaths.__new__(TensorProductPaths)
        paths.irreps_in1, paths.irreps_in2, paths.irreps_out = Irreps(in1), Irreps(in2), Irreps(out)
        paths.instructions = tuple(Instruction(*instruction) for instruction in instructions)
        paths.internal_weights, paths.shared_weights = internal, shared
        paths.weight_numel, paths.dtype = weights, dtype
        operator = {"uuu": LunaTPUUU, "uvu": LunaTPUVU, "uvw": LunaTPUVW,
                    "uvw_conv": LunaTPUVWConv}[mode]
        return operator(paths, dtype=dtype, index_bits=bits)


def create_uuu_conv_operator(official_tp_or_meta, *, dtype=None):
    if not isinstance(official_tp_or_meta, Mapping):
        official_tp_or_meta = TensorProductPaths.from_tensor_product(official_tp_or_meta)
    if dtype is None:
        dtype = (
            torch.float32
            if isinstance(official_tp_or_meta, Mapping)
            else official_tp_or_meta.dtype
        )
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("UUU convolution requires float32 or float64")

    if isinstance(official_tp_or_meta, Mapping):
        meta = official_tp_or_meta
    else:
        meta = build_uuu_conv_metadata(official_tp_or_meta, coefficient_dtype=dtype)

    op = LunaTPUUUConv(meta, dtype=dtype)
    if not isinstance(official_tp_or_meta, Mapping):
        op.tp = official_tp_or_meta
    op.lunatp_backend = (
        "lunatp_uuuconv_fp64" if dtype == torch.float64 else "lunatp_uuuconv_fp32"
    )
    op.lunatp_dtype = dtype
    return op


def create_uvu_conv_operator(official_tp_or_meta, *, dtype=None):
    if not isinstance(official_tp_or_meta, Mapping):
        official_tp_or_meta = TensorProductPaths.from_tensor_product(official_tp_or_meta)
    if dtype is None:
        dtype = torch.float32
    elif dtype is not torch.float32 and dtype is not torch.float64:
        raise ValueError("LunaTP uvu_conv requires torch.float32 or torch.float64.")

    if isinstance(official_tp_or_meta, Mapping):
        meta = official_tp_or_meta
    else:
        meta = build_uvu_conv_metadata(official_tp_or_meta, dtype)
    if not meta["generated_specs"] or any(
        int(spec["channels"]) <= 0 for spec in meta["generated_specs"]
    ):
        raise ValueError("UVU convolution requires positive channels for every path")
    op = LunaTPUVUConv(meta, dtype=dtype)
    op.lunatp_backend = (
        "lunatp_uvuconv_fp64" if dtype == torch.float64 else "lunatp_uvuconv_fp32"
    )
    op.lunatp_dtype = dtype
    return op


def prepare_tensor_product(
    irreps_in1, irreps_in2=None, irreps_out=None, instructions=None, *,
    in1_var=None, in2_var=None, out_var=None,
    irrep_normalization=None, path_normalization=None, shared_weights=False,
    dtype=torch.float32, convolution=False, mode=None,
):
    """Prepare parameter-free paths and launch caches for repeated calls.

    Functions also accept irreps directly and cache this preparation. Prebuilt
    UUU/UVU convolution metadata requires convolution=True and an explicit mode.
    """
    if isinstance(irreps_in1, Mapping):
        if not convolution or mode not in ("uuu", "uvu"):
            raise ValueError('Metadata requires convolution=True and mode="uuu" or "uvu"')
        factory = create_uuu_conv_operator if mode == "uuu" else create_uvu_conv_operator
        config = factory(irreps_in1, dtype=dtype)
    else:
        if isinstance(irreps_in1, TensorProductPaths):
            paths = copy(irreps_in1)
        else:
            if irreps_in2 is None or irreps_out is None or instructions is None:
                raise TypeError("All three irreps and instructions are required")
            paths = TensorProductPaths(
                irreps_in1, irreps_in2, irreps_out, instructions,
                in1_var=in1_var, in2_var=in2_var, out_var=out_var,
                irrep_normalization=irrep_normalization, path_normalization=path_normalization,
                internal_weights=False, shared_weights=shared_weights,
            )
        paths.dtype = dtype
        modes = {instruction.connection_mode for instruction in paths.instructions}
        if len(modes) != 1 or not modes <= {"uuu", "uvu", "uvw"}:
            raise ValueError("Tensor products require one connection mode: uuu, uvu or uvw")
        inferred = modes.pop()
        if mode is not None and mode != inferred:
            raise ValueError("mode must match the tensor-product instructions")
        mode = inferred
        if not convolution:
            return {"uuu": LunaTPUUU, "uvu": LunaTPUVU, "uvw": LunaTPUVW}[mode](
                paths, dtype=dtype,
            )
        if mode == "uuu":
            config = create_uuu_conv_operator(paths, dtype=dtype)
        elif mode == "uvu":
            config = create_uvu_conv_operator(paths, dtype=dtype)
        else:
            return LunaTPUVWConv(paths, dtype=dtype)
    return config


@lru_cache(maxsize=128)
def cached_tensor_product(
    irreps_in1, irreps_in2, irreps_out, instructions,
    in1_var, in2_var, out_var, irrep_normalization, path_normalization,
    shared_weights, dtype, convolution,
):
    return prepare_tensor_product(
        irreps_in1, irreps_in2, irreps_out, instructions,
        in1_var=in1_var, in2_var=in2_var, out_var=out_var,
        irrep_normalization=irrep_normalization, path_normalization=path_normalization,
        shared_weights=shared_weights, dtype=dtype, convolution=convolution,
    )


def tensor_product(
    x, y, weight=None, *, irreps_in1=None, irreps_in2=None, irreps_out=None,
    instructions=None, in1_var=None, in2_var=None, out_var=None,
    irrep_normalization=None, path_normalization=None, shared_weights=False,
    config=None,
):
    """Compute independent tensor products; weights are explicit tensors.

    Supply irreps/instructions, or reuse prepare_tensor_product(...). Shared
    weights have shape [weight_numel]; external weights have one row per item.
    UUU also supports unweighted paths and broadcast batch dimensions.
    """
    if config is None:
        if irreps_in1 is None or irreps_in2 is None or irreps_out is None or instructions is None:
            raise TypeError("Supply irreps and instructions, or a prepared config")
        config = cached_tensor_product(
            irreps_in1 if isinstance(irreps_in1, str) else str(Irreps(irreps_in1)),
            irreps_in2 if isinstance(irreps_in2, str) else str(Irreps(irreps_in2)),
            irreps_out if isinstance(irreps_out, str) else str(Irreps(irreps_out)),
            tuple(tuple(i) for i in instructions),
            None if in1_var is None else tuple(in1_var),
            None if in2_var is None else tuple(in2_var),
            None if out_var is None else tuple(out_var),
            irrep_normalization, path_normalization, shared_weights, x.dtype, False,
        )
    return config.run(config, x, y, weight)


def tensor_product_conv(
    x, edge_attr, weight=None, *, graph, irreps_in1=None, irreps_in2=None, irreps_out=None,
    instructions=None, in1_var=None, in2_var=None, out_var=None,
    irrep_normalization=None, path_normalization=None, shared_weights=False,
    config=None, **kwargs,
):
    """Contract edge messages and sum by destination, using explicit weights.

    All connection modes require an explicit LunaTPGraph. Reuse it
    across layers with the same topology; edge data follows its input ordering.
    """
    if config is None:
        if irreps_in1 is None or irreps_in2 is None or irreps_out is None or instructions is None:
            raise TypeError("Supply irreps and instructions, or a prepared config")
        config = cached_tensor_product(
            irreps_in1 if isinstance(irreps_in1, str) else str(Irreps(irreps_in1)),
            irreps_in2 if isinstance(irreps_in2, str) else str(Irreps(irreps_in2)),
            irreps_out if isinstance(irreps_out, str) else str(Irreps(irreps_out)),
            tuple(tuple(i) for i in instructions),
            None if in1_var is None else tuple(in1_var),
            None if in2_var is None else tuple(in2_var),
            None if out_var is None else tuple(out_var),
            irrep_normalization, path_normalization, shared_weights, x.dtype, True,
        )
    return config.run(config, x, edge_attr, weight, graph=graph, **kwargs)


__all__ = ["tensor_product", "tensor_product_conv", "prepare_tensor_product"]


class LunaTPUUU:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uuu_func)
    index_bits = 32
    compute_dtype = torch.float32

    def __init__(self, tp, *, dtype=None, index_bits=32):
        tp = TensorProductPaths.from_tensor_product(tp)
        self.tp = tp
        self.compute_dtype = tp.dtype if dtype is None else dtype
        if self.compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("UUU supports float32 and float64 inputs")

        self.meta = build_uuu_metadata(tp, self.compute_dtype)
        self.weight_mode = self.meta["weight_mode"]
        configure_indexing(self, self.meta, "uuu", index_bits)
        self.kernels = {}
        self._torch_config = compiled_operator_config(self)


class LunaTPUVU:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uvu_func)
    index_bits = 32
    compute_dtype = torch.float32

    def __init__(self, tp, *, dtype=None, index_bits=32):
        tp = TensorProductPaths.from_tensor_product(tp)
        self.tp = tp
        self.compute_dtype = tp.dtype if dtype is None else dtype
        if self.compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("UVU supports float32 and float64 inputs")

        self.meta = build_uvu_metadata(tp, self.compute_dtype)
        self.weight_mode = self.meta["weight_mode"]
        configure_indexing(self, self.meta, "uvu", index_bits)
        self.kernels = {}
        self._torch_config = compiled_operator_config(self)


class LunaTPUVW:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uvw_func)
    index_bits = 32
    compute_dtype = torch.float32

    def __init__(self, tp, *, dtype=None, index_bits=32):
        tp = TensorProductPaths.from_tensor_product(tp)
        self.tp = tp
        self.compute_dtype = tp.dtype if dtype is None else dtype
        if self.compute_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"UVW requires float32 or float64, got {self.compute_dtype}"
            )

        self.meta = build_uvw_metadata(tp, self.compute_dtype)
        self.weight_mode = self.meta["weight_mode"]
        configure_indexing(self, self.meta, "uvw", index_bits)
        self.wide_forward = all(
            spec["out_channels"] >= 128 for spec in self.meta["generated_specs"]
        )
        self.arch = None
        self.arch_device = -1
        self.kernels = {}
        self._torch_config = compiled_operator_config(self)


class LunaTPUUUConv:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uuu_conv_func)
    mode = "uuu_conv"

    def __init__(
        self, meta: Mapping[str, object], *, dtype=torch.float32, index_bits=32
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("UUU convolution requires float32 or float64.")
        configure_uuu_conv(self, meta, dtype, index_bits)
        self._torch_config = compiled_operator_config(self)


class LunaTPUVUConv:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uvu_conv_func)
    mode = "uvu_conv"

    def __init__(
        self,
        meta: Mapping[str, object],
        *,
        dtype: torch.dtype = torch.float32,
        index_bits=32,
    ):
        configure_uvu_conv(self, meta, dtype, index_bits)
        self._torch_config = compiled_operator_config(self)


class LunaTPUVWConv:
    register_ops = register_ops
    prepare_indexing = prepare_indexing
    run = staticmethod(lunatp_uvw_conv_func)
    mode = "uvw_conv"
    index_bits = 32
    compute_dtype = torch.float32

    def __init__(self, official_tp, *, dtype=None, index_bits=32):
        official_tp = TensorProductPaths.from_tensor_product(official_tp)
        self.tp = official_tp
        self.compute_dtype = official_tp.dtype if dtype is None else dtype
        if self.compute_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"UVW convolution requires float32 or float64, got {self.compute_dtype}"
            )
        self.dtype_tag = "fp64" if self.compute_dtype == torch.float64 else "fp32"
        self.arch = None
        self.arch_device = -1

        self.meta = build_uvw_metadata(official_tp, self.compute_dtype)
        self.weight_mode = self.meta["weight_mode"]
        configure_indexing(self, self.meta, "uvw_conv", index_bits)
        if self.weight_mode != "shared":
            # Node partitions can exhaust the launch grid before feature offsets.
            specs = self.meta["generated_specs"]
            tile_n = 32
            groups = {}
            for spec in specs:
                key = (spec["out_offset"], spec["out_channels"], spec["d3"])
                groups[key] = max(groups.get(key, 0), spec["path_channels"])
            forward_tasks = NODE_NEIGHBOR_SPLITS * sum(
                ((channels + tile_n - 1) // tile_n)
                * ((paths + FORWARD_SPLIT_CHANNELS - 1) // FORWARD_SPLIT_CHANNELS)
                for (_, channels, _), paths in groups.items()
            )
            node_tasks = max(forward_tasks, get_uvw_conv_backward_grid_tasks(self.meta), 1)
            edge_tasks = get_uvw_conv_tr_grid_tasks(self.meta)
            self.index_edge_limit = min(self.index_edge_limit, ((1 << 31) - 1) // edge_tasks)
            self.index_node_limit = min(self.index_node_limit, ((1 << 31) - 1) // node_tasks)
            self.wide_node_limit = min(self.wide_node_limit, ((1 << 63) - 1024) // node_tasks)
            self.wide_edge_limit = min(
                self.wide_edge_limit,
                ((1 << 63) - 1024) // max(edge_tasks, NODE_NEIGHBOR_SPLITS, BACKWARD_NEIGHBOR_SPLITS),
            )
        reset_launch_caches(self)
        self._torch_config = compiled_operator_config(self)
