"""PyTorch module for gate."""

from collections import OrderedDict
from math import sqrt

import torch

try:
    from e3nn.nn import Gate
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    Gate = None

from ..nn.gate.interface import gate, register_module_ops
from ..nn.activation.kernels.utils import parse_activation, parse_activation_paths
from ..o3.irreps import PublicIrreps as Irreps
from .activation import LunaEqActivation


def input_column(spans, column):
    for span in spans:
        offset = column - span["output_offset"]
        if 0 <= offset < span["width"]:
            return span["input_offset"] + offset
    raise ValueError("Gate extraction does not contain the requested column")


class LunaEqGate(LunaEqActivation):
    """Scalar gates with mathematical settings fixed at construction."""

    def __init__(
        self,
        irreps_scalars=None,
        act_scalars=None,
        irreps_gates=None,
        act_gates=None,
        irreps_gated=None,
        *,
        gate=None,
        dtype=None,
    ):
        # Share e3nn's normalized activations and sorted field extraction.
        torch.nn.Module.__init__(self)
        parsed_scalars = parsed_gates = None
        if gate is not None:
            if irreps_scalars is not None:
                raise TypeError("Gate module was supplied twice")
            official = gate
        elif hasattr(irreps_scalars, "sc") and act_scalars is None:
            official = irreps_scalars
        else:
            if Gate is None:
                raise ModuleNotFoundError("No module named 'e3nn'", name="e3nn")

            act_scalars, act_gates = list(act_scalars), list(act_gates)
            parsed_scalars = [parse_activation(activation) for activation in act_scalars]
            parsed_gates = [parse_activation(activation) for activation in act_gates]
            official = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)

        self.compute_dtype = torch.get_default_dtype() if dtype is None else dtype
        self.sc, self.act_scalars, self.act_gates, self.mul = (
            official.sc,
            official.act_scalars,
            official.act_gates,
            official.mul,
        )
        self.training = official.training
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = (
            official.irreps_scalars,
            official.irreps_gates,
            official.irreps_gated,
        )
        self.irreps_in, self.irreps_out = official.irreps_in, official.irreps_out
        self.meta = self.build_metadata(parsed_scalars, parsed_gates)
        self.in_dim, self.out_dim = self.irreps_in.dim, self.irreps_out.dim
        self.activation_paths = self.meta["scalar_activation"]
        self.irreps_in, self.irreps_out = self.meta["irreps_in"], self.meta["irreps_out"]
        self.kernel_cache = OrderedDict()
        object.__setattr__(self, "last_execution_backend", None)
        self._device_marker = torch.empty(0, dtype=self.compute_dtype)

    def build_metadata(self, parsed_scalars, parsed_gates):
        dtype = self.compute_dtype
        scalar_activation = parse_activation_paths(self.act_scalars, parsed_scalars)
        gate_activation = parse_activation_paths(self.act_gates, parsed_gates)

        cut = self.sc.cut
        in_slices = cut.irreps_in.slices()
        extract = []
        for irreps, instructions in zip(cut.irreps_outs, cut.instructions):
            spans = []
            for out_slice, index in zip(irreps.slices(), instructions):
                in_slice = in_slices[index]
                width = out_slice.stop - out_slice.start
                if width != in_slice.stop - in_slice.start:
                    raise ValueError("Gate Extract has inconsistent field widths")
                spans.append(
                    dict(input_offset=in_slice.start, output_offset=out_slice.start, width=width)
                )
            extract.append(tuple(spans))

        scalar_extract, gate_extract, gated_extract = extract

        scalar_paths = []
        offset = 0
        for width, spec in scalar_activation:
            for span in scalar_extract:
                begin = max(offset, span["output_offset"])
                end = min(offset + width, span["output_offset"] + span["width"])
                if begin < end:
                    scalar_paths.append(
                        dict(
                            width=end - begin,
                            input_offset=input_column(scalar_extract, begin),
                            output_offset=begin,
                            activation=spec,
                        )
                    )
            offset += width
        scalar_dim = offset

        gate_ranges = []
        offset = 0
        for width, spec in gate_activation:
            gate_ranges.append((offset, offset + width, spec))
            offset += width

        gated_paths = []
        product = self.mul
        gated_slices, gate_slices, output_slices = (
            product.irreps_in1.slices(),
            product.irreps_in2.slices(),
            product.irreps_out.slices(),
        )
        for instruction in product.instructions:
            if instruction.connection_mode != "uuu" or instruction.has_weight:
                raise ValueError("Gate requires the original unweighted elementwise tensor product")
            gated_index, gate_index, output_index = instruction.i_in1, instruction.i_in2, instruction.i_out
            components = product.irreps_in1[gated_index].ir.dim
            coefficient = float(instruction.path_weight) / sqrt(components)
            if coefficient != 1.0:
                raise ValueError("Gate requires original component-normalized scalar products")
            copies = product.irreps_in1[gated_index].mul
            for copy_index in range(copies):
                gate_column = gate_slices[gate_index].start + copy_index
                descriptor = next(
                    spec for begin, end, spec in gate_ranges
                    if begin <= gate_column < end
                )
                gated_paths.append(
                    dict(
                        copies=1,
                        components=components,
                        input_offset=input_column(gated_extract, gated_slices[gated_index].start + copy_index * components),
                        gate_offset=input_column(gate_extract, gate_column),
                        output_offset=scalar_dim + output_slices[output_index].start + copy_index * components,
                        activation=descriptor,
                    )
                )

        # Adjacent copies share one static CUDA path when every address advances
        # by the same physical scalar/vector width.
        merged_paths = []
        for path in gated_paths:
            if merged_paths:
                previous = merged_paths[-1]
                copies, components = previous["copies"], previous["components"]
                if (
                    path["components"] == components
                    and path["activation"] == previous["activation"]
                    and path["input_offset"] == previous["input_offset"] + copies * components
                    and path["gate_offset"] == previous["gate_offset"] + copies
                    and path["output_offset"] == previous["output_offset"] + copies * components
                ):
                    previous["copies"] += 1
                    continue
            merged_paths.append(path)
        return dict(
            in_dim=self.irreps_in.dim,
            out_dim=self.irreps_out.dim,
            irreps_in=Irreps(str(self.irreps_in)),
            irreps_out=Irreps(str(self.irreps_out)),
            scalar_paths=tuple(scalar_paths),
            gated_paths=tuple(merged_paths),
            extract_specs=tuple(extract),
            extract_identity=tuple(
                tuple(indices) == tuple(range(len(cut.irreps_in)))
                for indices in cut.instructions
            ),
            extract_dims=tuple(irreps.dim for irreps in cut.irreps_outs),
            scalar_activation=scalar_activation,
            gate_activation=gate_activation,
            tp_irreps1=Irreps(str(product.irreps_in1)),
            tp_irreps2=Irreps(str(product.irreps_in2)),
            dtype=dtype,
            has_grad=bool(scalar_activation or offset),
        )

    register_module_ops = register_module_ops

    def forward(self, features):
        native = (features.is_cuda and features.dtype in (torch.float32, torch.float64)
                  and features.ndim and features.shape[-1] == self.in_dim)
        if torch.compiler.is_compiling() and native:
            return getattr(torch.ops.lunaeq, self.register_module_ops()).default(features)
        backend = "cuda_fused" if native else "torch_compat"
        object.__setattr__(self, "last_execution_backend", backend)
        return gate(features, layout=self.meta, kernel_cache=self.kernel_cache)


__all__ = ['LunaEqGate']
