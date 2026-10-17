"""Tensor-product modules and fused graph convolutions for UUU, UVU and UVW."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..lunatp.interface import (
    LunaTPUUU, LunaTPUVU, LunaTPUVW, LunaTPUVWConv,
    create_uuu_conv_operator, create_uvu_conv_operator,
    tensor_product, tensor_product_conv,
)
from ..lunatp.utils.layout import TensorProductConfig
from ..lunatp.utils.cg import wigner_3j, build_cg_terms
from ..lunatp.graph import make_graph


class TensorProduct(torch.nn.Module):
    """CUDA tensor product with e3nn path and normalization arguments.

    Pass irreps and instructions, or an existing TensorProduct/configuration.
    Instructions select one connection mode: uuu, uvu or uvw. Shared weights
    remain registered in the constructed or supplied configuration. Mixed connection modes are not
    supported by the fused kernels.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2=None,
        irreps_out=None,
        instructions=None,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization=None,
        path_normalization=None,
        internal_weights=None,
        shared_weights=None,
        *,
        dtype=None,
    ):
        super().__init__()
        if isinstance(irreps_in1, torch.nn.Module):
            if any(value is not None for value in (
                irreps_in2, irreps_out, instructions, in1_var, in2_var, out_var,
                irrep_normalization, path_normalization, internal_weights, shared_weights,
            )):
                raise TypeError("Pass either an existing tensor product or irreps and path options")
            tp = irreps_in1
        else:
            if irreps_in2 is None or irreps_out is None or instructions is None:
                raise TypeError("irreps_in2, irreps_out and instructions are required")
            tp = TensorProductConfig(
                irreps_in1, irreps_in2, irreps_out, instructions,
                in1_var=in1_var, in2_var=in2_var, out_var=out_var,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
                internal_weights=internal_weights, shared_weights=shared_weights,
            )
            if dtype is not None and tp.weight.dtype != dtype:
                tp.to(dtype=dtype)
        modes = {instruction.connection_mode for instruction in tp.instructions}
        if len(modes) != 1 or not modes <= {"uuu", "uvu", "uvw"}:
            raise ValueError("TensorProduct requires one connection mode: uuu, uvu or uvw")
        self.mode = modes.pop()
        operators = {"uuu": LunaTPUUU, "uvu": LunaTPUVU, "uvw": LunaTPUVW}
        self.op = operators[self.mode](tp, dtype=dtype)
        self.tp = tp
        self.irreps_in1, self.irreps_in2, self.irreps_out = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
        self.instructions = tp.instructions
        self.weight_numel = tp.weight_numel
        self.internal_weights, self.shared_weights = tp.internal_weights, tp.shared_weights
        self.train(tp.training)

    @property
    def weight(self):
        tp = getattr(self, "tp", None)
        return tp.weight if tp is not None else None

    def _apply(self, fn, recurse=True):
        dtype = fn(torch.empty(0, dtype=self.op.compute_dtype)).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Tensor products support float32 and float64 inputs")
        result = super()._apply(fn, recurse=recurse)
        if dtype != self.op.compute_dtype:
            index_bits = self.op.index_bits
            if self.mode in ("uuu", "uvu", "uvw", "uvw_conv"):
                self.op = type(self.op)(self.tp, dtype=dtype, index_bits=index_bits)
            elif hasattr(self, "tp"):
                factory = create_uuu_conv_operator if self.mode == "uuu_conv" else create_uvu_conv_operator
                self.op = factory(self.tp, dtype=dtype)
            else:
                meta = dict(self.op.meta)
                specs = []
                for original in self.op.generated_specs:
                    spec = dict(original)
                    if "path_weight" in spec:
                        cg = wigner_3j(spec["l1"], spec["l2"], spec["l3"], dtype=dtype, device="cpu")
                        spec["terms"] = [] if spec["path_weight"] == 0 else [
                            (i, j, k, coefficient * spec["path_weight"])
                            for i, j, k, coefficient in build_cg_terms(cg, 1.0)
                        ]
                    specs.append(spec)
                meta["generated_specs"] = specs
                factory = create_uuu_conv_operator if self.mode == "uuu_conv" else create_uvu_conv_operator
                self.op = factory(meta, dtype=dtype)
            if self.op.index_bits != index_bits:
                self.op = type(self.op)(self.op.meta, dtype=dtype, index_bits=index_bits)
        return result

    def forward(self, x, y, weight=None):
        if weight is None and self.internal_weights:
            weight = self.weight
        return tensor_product(x, y, weight, config=self.op)


class TensorProductConv(TensorProduct):
    """Fused tensor product and graph aggregation.

    Pass irreps and instructions, an existing TensorProduct/configuration, or
    prebuilt metadata with mode="uuu" or "uvu". Metadata uses external weights
    and defaults to FP32. Configurations supply their own default dtype.
    Prepare topology with create_graph() and pass graph explicitly on every call.
    Radial networks and neighbor normalization stay in the caller.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2=None,
        irreps_out=None,
        instructions=None,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization=None,
        path_normalization=None,
        internal_weights=None,
        shared_weights=None,
        *,
        dtype=None,
        mode=None,
    ):
        torch.nn.Module.__init__(self)
        metadata = isinstance(irreps_in1, Mapping)
        if metadata or isinstance(irreps_in1, torch.nn.Module):
            if any(value is not None for value in (
                irreps_in2, irreps_out, instructions, in1_var, in2_var, out_var,
                irrep_normalization, path_normalization, internal_weights, shared_weights,
            )):
                raise TypeError("Pass either an existing tensor product/metadata or irreps and path options")
            tp = irreps_in1
        else:
            if irreps_in2 is None or irreps_out is None or instructions is None:
                raise TypeError("irreps_in2, irreps_out and instructions are required")
            tp = TensorProductConfig(
                irreps_in1, irreps_in2, irreps_out, instructions,
                in1_var=in1_var, in2_var=in2_var, out_var=out_var,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
                internal_weights=internal_weights, shared_weights=shared_weights,
            )
            if dtype is not None and tp.weight.dtype != dtype:
                tp.to(dtype=dtype)
        if metadata:
            if mode not in ("uuu", "uvu"):
                raise ValueError('Prebuilt metadata requires mode="uuu" or mode="uvu"')
            dtype = torch.float32 if dtype is None else dtype
        else:
            modes = {instruction.connection_mode for instruction in tp.instructions}
            if len(modes) != 1 or not modes <= {"uuu", "uvu", "uvw"}:
                raise ValueError("TensorProductConv requires one connection mode: uuu, uvu or uvw")
            inferred_mode = modes.pop()
            if mode is not None and mode != inferred_mode:
                raise ValueError("mode must match the tensor-product instructions")
            mode = inferred_mode
            dtype = tp.weight.dtype if dtype is None else dtype
        self.mode = mode + "_conv"
        if mode == "uuu":
            self.op = create_uuu_conv_operator(tp, dtype=dtype)
        elif mode == "uvu":
            self.op = create_uvu_conv_operator(tp, dtype=dtype)
        else:
            self.op = LunaTPUVWConv(tp, dtype=dtype)
        if metadata:
            self.weight_numel = tp["weight_numel"]
            self.internal_weights = self.shared_weights = False
        else:
            self.tp = tp
            self.irreps_in1, self.irreps_in2, self.irreps_out = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
            self.instructions = tp.instructions
            self.weight_numel = tp.weight_numel
            self.internal_weights, self.shared_weights = tp.internal_weights, tp.shared_weights
            self.train(tp.training)

    @staticmethod
    def create_graph(src, dst, num_nodes=None, **kwargs):
        """Prepare shared topology once per batch, independent of connection mode."""
        return make_graph(src, dst, num_nodes, **kwargs)

    def forward(self, x, edge_attr, weight=None, *, graph, **kwargs):
        if weight is None and self.internal_weights:
            weight = self.weight
        return tensor_product_conv(x, edge_attr, weight, graph=graph, config=self.op, **kwargs)


__all__ = ["TensorProduct", "TensorProductConv"]
