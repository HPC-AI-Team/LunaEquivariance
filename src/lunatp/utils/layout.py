"""Compile-time tensor-product path layouts and exact coefficient classes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import prod, sqrt
from typing import Any, Literal, NamedTuple, TypedDict

import torch
from typing_extensions import NotRequired

from ...o3.irreps import Irreps

# Two index variants reserve half of the 64 KiB CUDA constant budget.
CUDA_TASK_TABLE_WARP_LIMIT = 2048
_FP64_SIGN_BIT = 0x8000000000000000

Mode = Literal["uuu", "uvu", "uvw", "uuu_conv", "uvu_conv", "uvw_conv"]
CGTerm = tuple[int, int, int, float]


class PathSpec(TypedDict):
    x_offset: int
    y_offset: int
    out_offset: int
    weight_offset: int
    d1: int
    d2: int
    d3: int
    l1: int
    l2: int
    l3: int
    terms: list[CGTerm]


class UvuPathSpec(PathSpec):
    channels: int
    weight_stride: NotRequired[int]
    l0_path: NotRequired[int]


class UvuTensorPathSpec(UvuPathSpec):
    accumulate_output: bool


class UuuPathSpec(PathSpec):
    channels: int
    has_weight: bool
    path_weight: NotRequired[float]
    l0_path: NotRequired[int]


class UuuTensorPathSpec(UuuPathSpec):
    accumulate_output: bool


class UvwPathSpec(PathSpec):
    in_channels: int
    edge_channels: int
    out_channels: int
    path_channels: int


class Metadata(TypedDict):
    in1_dim: int
    in2_dim: int
    out_dim: int
    weight_numel: int


class UvuConvMetadata(Metadata):
    max_channels: int
    generated_specs: list[UvuPathSpec]


class UuuConvMetadata(Metadata):
    max_channels: int
    generated_specs: list[UuuPathSpec]


class UuuMetadata(Metadata):
    dtype: NotRequired[torch.dtype]
    max_channels: int
    generated_specs: list[UuuTensorPathSpec]
    weight_mode: Literal["shared", "external"]
    zero_output: bool


@dataclass(frozen=True)
class SignedIJMember:
    """One input pair in an exact, up-to-sign coefficient class."""

    i_idx: int
    j_name: int
    relative_sign: int
    terms: tuple[tuple[int, int, int, float], ...]


@dataclass(frozen=True)
class SignedIJClass:
    """Path-local input pairs with identical double-precision coefficient signatures."""

    signature_bits: tuple[tuple[int, int], ...]
    members: tuple[SignedIJMember, ...]


def canonical_signed_signature(
    entries: list[tuple[int, float]],
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Return an exact double-precision signature and its global orientation."""
    # Compare exact IEEE double bits; narrowing would merge distinct FP64 coefficients.
    raw = tuple(
        sorted(
            (int(index), struct.unpack("<Q", struct.pack("<d", float(coefficient)))[0])
            for index, coefficient in entries
        )
    )
    negated = tuple((index, bits ^ _FP64_SIGN_BIT) for index, bits in raw)
    canonical = min(raw, negated)
    return canonical, (1 if raw == canonical else -1)


def build_signed_ij_classes(
    terms: list[tuple[int, int, int, float]],
    *,
    spec_y_offset: int,
    y_name_base: int,
) -> list[SignedIJClass]:
    """Group complete ``(i,j)->k`` coefficient vectors by exact double-precision sign.

    Classes group static coefficients within a path; dynamic weights are
    applied separately.
    """
    terms_by_ij: dict[tuple[int, int], list[tuple[int, int, int, float]]] = {}
    for i_idx, j_idx, k_idx, coeff in terms:
        pair = (int(i_idx), spec_y_offset + int(j_idx) - y_name_base)
        terms_by_ij.setdefault(pair, []).append(
            (int(i_idx), int(j_idx), int(k_idx), float(coeff))
        )

    classes: dict[
        tuple[tuple[int, int], ...],
        list[tuple[tuple[int, int], int, tuple[tuple[int, int, int, float], ...]]],
    ] = {}
    for pair, pair_terms_list in sorted(terms_by_ij.items()):
        pair_terms = tuple(pair_terms_list)
        signature, orientation = canonical_signed_signature(
            [(int(k_idx), float(coeff)) for _, _, k_idx, coeff in pair_terms]
        )
        classes.setdefault(signature, []).append((pair, orientation, pair_terms))

    result: list[SignedIJClass] = []
    for signature, entries in classes.items():
        representative_orientation = entries[0][1]
        members = tuple(
            SignedIJMember(
                i_idx=pair[0],
                j_name=pair[1],
                relative_sign=representative_orientation * orientation,
                terms=pair_terms,
            )
            for pair, orientation, pair_terms in entries
        )
        result.append(SignedIJClass(signature_bits=signature, members=members))
    return result


def l0_feature_layout(
    specs: list[UvuPathSpec],
) -> tuple[dict[int, int], int]:
    """Return each scalar-output path's column offset and the total width."""
    offsets: dict[int, int] = {}
    path_offsets: dict[tuple[str, int], int] = {}
    width = 0
    for path_idx, spec in sorted(
        enumerate(specs), key=lambda item: int(item[1]["out_offset"])
    ):
        if int(spec["l3"]) != 0:
            continue
        channels = spec["channels"]
        path = ("instruction", int(spec["l0_path"])) if "l0_path" in spec else ("path", path_idx)
        if path not in path_offsets:
            path_offsets[path] = width
            width += int(channels)
        offsets[path_idx] = path_offsets[path]
    return offsets, width


def group_forward_paths(
    specs: list[dict[str, Any]],
    *,
    axis: str,
    live_output_cap: int | None,
) -> list[tuple[tuple[int, int, int, int], list[tuple[int, dict[str, Any]]]]]:
    """Group paths sharing an input traversal, then cap live output accumulators.

    First-fit bins and the final stable cost sort determine launch order.
    """
    offset_field, width_field = {"x": ("x_offset", "d1"), "y": ("y_offset", "d2")}[axis]
    grouped: dict[tuple[int, int, int, int], list[tuple[int, dict[str, Any]]]] = {}
    for path_idx, spec in enumerate(specs):
        input_width = int(spec[width_field])
        channels = int(spec["channels"])
        # Launch planners use (input offset, unused offset, width, channels).
        key = (int(spec[offset_field]), -1, input_width, channels)
        grouped.setdefault(key, []).append((path_idx, spec))
    fused_groups = []
    for key, group_specs in grouped.items():
        if live_output_cap is None:
            fused_groups.append((key, group_specs))
            continue
        if sum(int(spec["d3"]) for _, spec in group_specs) <= live_output_cap:
            fused_groups.append((key, group_specs))
            continue
        bins: list[tuple[int, list[tuple[int, dict[str, Any]]]]] = []
        for item in sorted(
            group_specs,
            key=lambda item: (-int(item[1]["d3"]), -len(item[1]["terms"]), item[0]),
        ):
            outputs = int(item[1]["d3"])
            for bin_idx, (live_outputs, bin_specs) in enumerate(bins):
                if live_outputs + outputs <= live_output_cap:
                    bin_specs.append(item)
                    bins[bin_idx] = (live_outputs + outputs, bin_specs)
                    break
            else:
                bins.append((outputs, [item]))
        fused_groups.extend((key, bin_specs) for _, bin_specs in bins)
    return sorted(
        fused_groups,
        key=lambda item: sum(len(spec["terms"]) for _, spec in item[1]),
        reverse=True,
    )


# Path normalization adapted from e3nn 0.6.0; operation order is retained.
class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple[int, ...]


class TensorProductPaths:
    """Normalized tensor-product paths without parameters or Module state."""

    dtype: torch.dtype

    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        instructions: list[tuple[Any, ...]],
        *,
        in1_var: list[float] | None = None,
        in2_var: list[float] | None = None,
        out_var: list[float] | None = None,
        irrep_normalization: str | None = None,
        path_normalization: str | None = None,
        internal_weights: bool | None = None,
        shared_weights: bool | None = None,
    ) -> None:
        if irrep_normalization is None:
            irrep_normalization = "component"
        if path_normalization is None:
            path_normalization = "element"
        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        paths = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]
        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(
                self.irreps_in1
            ), "Len of ir1_var must be equal to len(irreps_in1)"
        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(
                self.irreps_in2
            ), "Len of ir2_var must be equal to len(irreps_in2)"
        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(
                self.irreps_out
            ), "Len of out_var must be equal to len(irreps_out)"

        normalization_coefficients = []
        for ins in paths:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )
            assert ins.connection_mode in ["uuu", "uvw", "uvu"]
            if ins.connection_mode == "uuu":
                assert (
                    mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                ), "UUU requires equal input and output multiplicities"
            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1
            if path_normalization == "element":
                x = sum(
                    (
                        in1_var[i.i_in1] * in2_var[i.i_in2] * {
                            "uvw": self.irreps_in1[i.i_in1].mul * self.irreps_in2[i.i_in2].mul,
                            "uvu": self.irreps_in2[i.i_in2].mul,
                            "uuu": 1,
                        }[i.connection_mode]
                        for i in paths
                        if i.i_out == ins.i_out
                    )
                )
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * {
                    "uvw": self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul,
                    "uvu": self.irreps_in2[ins.i_in2].mul,
                    "uuu": 1,
                }[ins.connection_mode]
                x *= len([i for i in paths if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1
            if x > 0.0:
                alpha /= x
            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight
            normalization_coefficients += [sqrt(alpha)]
        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(paths, normalization_coefficients)
        ]
        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = shared_weights and any((i.has_weight for i in self.instructions))
        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.weight_numel = sum(
            (prod(ins.path_shape) for ins in self.instructions if ins.has_weight)
        )

    @classmethod
    def from_tensor_product(cls, tp: Any) -> TensorProductPaths:
        """Copy only static path attributes; never retain a parameter owner."""
        if isinstance(tp, cls):
            return tp
        paths = cls.__new__(cls)
        for name in ("irreps_in1", "irreps_in2", "irreps_out", "instructions",
                     "internal_weights", "shared_weights", "weight_numel"):
            setattr(paths, name, getattr(tp, name))
        paths.dtype = tp.weight.dtype
        return paths


class TensorProductConfig(torch.nn.Module):  # type: ignore[misc]
    """Tensor-product paths and optional registered shared weights."""

    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        instructions: list[tuple[Any, ...]],
        *,
        in1_var: list[float] | None = None,
        in2_var: list[float] | None = None,
        out_var: list[float] | None = None,
        irrep_normalization: str | None = None,
        path_normalization: str | None = None,
        internal_weights: bool | None = None,
        shared_weights: bool | None = None,
    ) -> None:
        super().__init__()
        paths = TensorProductPaths(
            irreps_in1, irreps_in2, irreps_out, instructions,
            in1_var=in1_var, in2_var=in2_var, out_var=out_var,
            irrep_normalization=irrep_normalization, path_normalization=path_normalization,
            internal_weights=internal_weights, shared_weights=shared_weights,
        )
        self.__dict__.update(vars(paths))
        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.Tensor())


def configure_indexing(op: Any, meta: dict[str, Any], mode: Mode, index_bits: int) -> None:
    """Precompute conservative bounds covering addresses, workspaces and grids."""
    op.index_bits = index_bits
    op.index_mode = mode
    specs = meta["generated_specs"]
    shared = meta.get("weight_mode") == "shared"
    dims = (meta["in1_dim"], meta["in2_dim"], meta["out_dim"])
    if max(*dims, meta["weight_numel"]) > (1 << 31) - 1:
        raise OverflowError(
            "Feature layout dimensions must fit signed int32; batch and edge offsets use int64"
        )
    channels = [s.get("channels", s.get("in_channels", 1)) for s in specs]
    parts = sum((c + 31) // 32 for c in channels)
    node_span = max(*dims, sum(c * s["d3"] for c, s in zip(channels, specs)), 4 * parts)
    if mode == "uvw_conv":
        node_span = max(
            node_span, sum(s["path_channels"] * s["d3"] for s in specs)
        )
    edge_span = max(
        meta["in2_dim"] * max(1, parts), 1 if shared else meta["weight_numel"]
    )
    # Source segments include one empty segment for every isolated node.
    if mode == "uvu_conv":
        node_span = max(node_span, 2 * meta["in1_dim"])
        edge_span = max(edge_span, 2 * meta["in1_dim"])
    if mode == "uuu_conv":
        node_span = max(1, node_span, 2 * meta["in1_dim"],
                        sum(c * s["d1"] for c, s in zip(channels, specs)))
        edge_span = max(1, *dims, meta["weight_numel"],
                        sum(c * s["d3"] for c, s in zip(channels, specs)), 4 * parts)
    if mode == "uuu":
        node_span = max(1, node_span)
        edge_span = max(1, edge_span)
    if mode in ("uuu", "uvu", "uvw"):
        node_span = edge_span = max(node_span, edge_span)
    op.wide_node_limit = ((1 << 63) - 1024) // node_span
    op.wide_edge_limit = ((1 << 63) - 1024) // edge_span
    limit = (1 << 31) - 1024  # Leave room for tile rounding and loop increments.
    op.index_node_limit = limit // node_span
    op.index_edge_limit = limit // edge_span
    if mode == "uvu" and shared:
        op.index_edge_limit = min(op.index_edge_limit, 64 * 65535)
    if mode == "uvu_conv":
        # The compact forward split schedule packs a chunk ID into 16 bits.
        op.index_edge_limit = min(op.index_edge_limit, 40 * 65535)
    if mode == "uvw_conv" and shared:
        op.index_node_limit = min(
            op.index_node_limit, 16 * 65535 // max(s["d3"] for s in specs)
        )
    if parts > 65535 or (mode == "uvw_conv" and len(specs) * 8 > 65535):
        op.index_node_limit = op.index_edge_limit = 0
    op.index_item_limit = min(op.index_node_limit, op.index_edge_limit)
    op.index64_operator = None


@torch.compiler.assume_constant_result  # type: ignore[untyped-decorator]
def prepare_indexing(op: Any) -> None:
    """Create the static wide configuration once, outside symbolic shape tracing."""
    if op.index64_operator is None:
        if op.index_mode in ("uuu_conv", "uvu_conv"):
            wide = type(op)(op.meta, dtype=op.compute_dtype, index_bits=64)
        else:
            wide = type(op)(op.tp, dtype=op.compute_dtype, index_bits=64)
        # Saved autograd contexts retain their width and independent launch caches.
        object.__setattr__(op, "index64_operator", wide)


def select_indexing(op: Any, nodes: int, edges: int) -> Any:
    """Use wide offsets after the public wrapper detects a 32-bit limit."""
    if nodes > op.wide_node_limit or edges > op.wide_edge_limit:
        raise OverflowError(
            "Tensor, workspace or task offsets exceed signed int64 capacity"
        )
    if op.index_bits == 64:
        return op
    op.prepare_indexing()
    return op.index64_operator
