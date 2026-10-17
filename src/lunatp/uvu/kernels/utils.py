"""UVU path metadata, launch geometry and cached backward launches."""

import re
from collections import Counter
from math import prod

import torch

from ....utils.extension import load_tp_extension
from ...utils.cg import build_cg_terms, wigner_3j
from ...utils.layout import UvuTensorPathSpec


SHARED_WEIGHT_ROWS = 64

def build_uvu_metadata(tp, dtype):
    internal = bool(tp.internal_weights)
    shared = bool(tp.shared_weights)
    if shared:
        weight_mode = "shared"
    elif not internal and not shared:
        weight_mode = "external"
    else:
        raise ValueError(
            "UVU generated kernels support only "
            "(internal_weights=True, shared_weights=True) or "
            f"(internal_weights=False, shared_weights=False); got internal_weights={internal}, shared_weights={shared}."
        )

    in_slices = tp.irreps_in1.slices()
    edge_slices = tp.irreps_in2.slices()
    out_slices = tp.irreps_out.slices()
    output_uses = Counter(ins.i_out for ins in tp.instructions)
    specs: list[UvuTensorPathSpec] = []
    weight_numel = int(tp.weight_numel)
    weight_offset = 0
    max_channels = 0

    for ins in tp.instructions:
        if ins.connection_mode != "uvu":
            raise ValueError(
                f"Only connection_mode='uvu' is supported, got {ins.connection_mode!r}."
            )
        if not ins.has_weight:
            raise ValueError("UVU generated kernels require weighted instructions.")

        mul_in, ir_in = tp.irreps_in1[ins.i_in1]
        mul_edge, ir_edge = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]
        channels = int(mul_in)
        edge_channels = int(mul_edge)
        if channels <= 0 or edge_channels <= 0:
            raise ValueError(
                "UVU weighted paths require positive input and edge multiplicities."
            )
        if channels != int(mul_out):
            raise ValueError(
                "UVU generated kernels require mul_in == mul_out for every path."
            )
        if tuple(int(v) for v in ins.path_shape) != (channels, edge_channels):
            raise ValueError(
                "UVU generated kernels expect uvu path_shape=(channels, edge_channels)."
            )
        max_channels = max(max_channels, channels)

        cg = (
            wigner_3j(ir_in.l, ir_edge.l, ir_out.l)
            if dtype is torch.float32
            else wigner_3j(ir_in.l, ir_edge.l, ir_out.l, dtype=dtype)
        )
        if cg.dtype is not dtype or cg.device.type != "cpu":
            cg = cg.to(dtype=dtype, device="cpu")
        path_weight = float(ins.path_weight)
        terms = build_cg_terms(cg, path_weight)

        for edge_channel in range(edge_channels):
            spec: UvuTensorPathSpec = {
                "x_offset": int(in_slices[ins.i_in1].start),
                "y_offset": int(edge_slices[ins.i_in2].start)
                + edge_channel * ir_edge.dim,
                "out_offset": int(out_slices[ins.i_out].start),
                "accumulate_output": output_uses[ins.i_out] > 1
                or edge_channels > 1,
                "weight_offset": weight_offset + edge_channel,
                "channels": channels,
                "d1": int(ir_in.dim),
                "d2": int(ir_edge.dim),
                "d3": int(ir_out.dim),
                "l1": int(ir_in.l),
                "l2": int(ir_edge.l),
                "l3": int(ir_out.l),
                "terms": terms,
            }
            # e3nn flattens each weighted UVU path in [u, v] order. Expanding
            # v only at code generation keeps all edge channels in one launch.
            if edge_channels != 1:
                spec["weight_stride"] = edge_channels
            specs.append(spec)
        weight_offset += prod(ins.path_shape)

    if not specs:
        raise ValueError("No uvu instructions found.")
    if weight_offset != weight_numel:
        raise ValueError(
            f"lowered weight_numel={weight_offset}, configuration reports {tp.weight_numel}."
        )
    meta = {
        "in1_dim": int(tp.irreps_in1.dim),
        "in2_dim": int(tp.irreps_in2.dim),
        "out_dim": int(tp.irreps_out.dim),
        "weight_numel": weight_numel,
        "max_channels": max_channels,
        "weight_mode": weight_mode,
        "generated_specs": specs,
        # Shared outputs accumulate path contributions; unused components stay zero.
        "zero_output": any(spec["accumulate_output"] for spec in specs)
        or sum(
            spec["channels"] * len({term[2] for term in spec["terms"]})
            for spec in specs
        )
        != int(tp.irreps_out.dim),
    }
    if dtype is torch.float64:
        meta["dtype"] = dtype
    return meta


def make_grid_source(source: str, channel_tile: int) -> str:
    """Promote grid-rounding constants only when their int32 sums overflow."""
    limit = (1 << 31) - 1
    for name, tile in (
        ("GEN_MAX_CHANNELS", channel_tile),
        ("GEN_WEIGHT_NUMEL", 32),
        ("GEN_NUM_WARP_TASKS", 8),
    ):
        match = re.search(f"constexpr int {name} = (\\d+);", source)
        if match and int(match.group(1)) + tile - 1 > limit:
            declaration = match.group(0)
            source = source.replace(
                declaration, declaration.replace("int ", "int64_t ", 1)
            )

    return re.sub(
        r"\((\d+) \+ GEN_CHANNEL_TILE - 1\)",
        lambda match: (
            f"(int64_t({int(match.group(1))}) + GEN_CHANNEL_TILE - 1)"
            if int(match.group(1)) + channel_tile - 1 > limit
            else match.group(0)
        ),
        source,
    )


def get_uvu_launch_config(
    max_channels: int, channels: object | None = None
) -> tuple[int, int, int]:
    # heuristic config
    if channels is not None:
        channel_set = set(channels)
        if len(channel_set) > 1:
            if max_channels <= 384:
                return 64, 64, 8
            return 128, 128, 4
    if max_channels <= 64:
        return 64, 64, 8
    if max_channels <= 256:
        return 128, 128, 4
    return 256, 256, 2


def run_backward(op, order, *inputs, make_source, only_x=False, needs=None):
    native_needs = tuple(needs) if needs is not None else (True,) * (3, 4, 7)[order - 1]
    if not any(native_needs) and not only_x:
        return (None,) * len(native_needs)
    key = order, native_needs
    cached = op.kernels.get(key)
    if cached is None:
        from .backward import use_uvu_dx_dy_dw_fusion

        # The tiled schedules already select W/rW. Keep their existing feature
        # consumers; exact output subsets reuse only the bounded fused schedule.
        outputs = list(native_needs if use_uvu_dx_dy_dw_fusion(op.meta, order)
                       else (True,) * len(native_needs))
        outputs[2] = native_needs[2]
        if order == 3:
            outputs[6] = native_needs[6]
        outputs = tuple(outputs)
        need_grad_weight = outputs[2]
        need_grad_rw = order == 3 and outputs[6]
        dtype_name = "fp64" if op.compute_dtype is torch.float64 else "fp32"
        phase = ("bwd", "double_bwd", "triple_bwd")[order - 1]
        source = (
            make_source(
                op.meta, need_grad_weight=need_grad_weight, need_grad_rw=need_grad_rw,
                needs=outputs,
            )
            if order == 3
            else make_source(op.meta, need_grad_weight=need_grad_weight, needs=outputs)
        )
        extension = load_tp_extension(
            f"lunatp_uvu_{phase}_{op.weight_mode}_{dtype_name}",
            source,
            max_channels=op.meta["max_channels"],
            index_bits=op.index_bits,
        )
        if order == 1:
            launch = (
                getattr(extension, f"lunatp_uvu_bwd_{op.weight_mode}_{dtype_name}"),
                getattr(extension, f"lunatp_uvu_bwd_x_{op.weight_mode}_{dtype_name}"),
            )
        else:
            launch = getattr(
                extension, f"lunatp_uvu_{phase}_{op.weight_mode}_{dtype_name}"
            )
        cached = op.kernels[key] = launch, outputs
    launch, outputs = cached
    input_only = order == 1 and native_needs == (True, False, False)
    if order == 1:
        launch = launch[only_x or input_only]
    result = launch(*(inputs[1:] if input_only and not only_x else inputs))
    if only_x:
        return result
    if input_only:
        return result, None, None
    values = iter(result)
    gradients = tuple(next(values) if needed else None for needed in outputs)
    return tuple(value if needed else None for value, needed in zip(gradients, native_needs))
