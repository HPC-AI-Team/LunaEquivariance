"""PyTorch module for linear."""

from math import prod
import operator

import torch

try:
    from e3nn.o3 import Linear as E3nnLinear
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    E3nnLinear = None

from ..o3.linear.interface import MISSING, linear
from ..o3.linear.metadata import LinearConfig, LinearKernelConfig


class LunaEqLinear(torch.nn.Module):
    """e3nn Linear API with native CUDA FP32/FP64 derivatives through order 3.

    CPU, low precision, heterogeneous operands and layouts beyond the CUDA
    dimension capacity use Torch computation. The last selected
    path is exposed as ``last_execution_backend``; only ``cuda_fused`` denotes
    the native CUDA contraction path.
    """

    def __init__(
        self,
        irreps_in=MISSING,
        irreps_out=MISSING,
        *,
        f_in=None,
        f_out=None,
        internal_weights=None,
        shared_weights=None,
        instructions=None,
        biases=False,
        path_normalization="element",
        _optimize_einsums=None,
        dtype=None,
    ):
        super().__init__()
        if irreps_in is MISSING:
            raise TypeError("missing required argument: irreps_in")
        is_linear = all(
            hasattr(irreps_in, name)
            for name in ("irreps_in", "irreps_out", "instructions", "weight", "bias")
        )
        if is_linear and (irreps_out is MISSING or irreps_out is None):
            tp = irreps_in
            if f_in is None or f_out is None:
                if hasattr(tp, "f_in"):
                    inferred_in, inferred_out = tp.f_in, tp.f_out
                elif tp.weight.ndim == 3:
                    inferred_in, inferred_out = tuple(tp.weight.shape[:2])
                else:
                    graph = getattr(getattr(tp, "_compiled_main", None), "graph", None)
                    if callable(getattr(graph, "nodes", None)):
                        # TorchScript retains reshape sizes as graph inputs.
                        inputs = {value.debugName().split(".")[0]: value for value in graph.inputs()}
                        weight_input = inputs.get("w", inputs.get("weight"))
                        feature_input = inputs.get("x", inputs.get("features"))
                        feature_rank, input_axis, output_axis = 1, None, None
                        for node in graph.nodes():
                            values = tuple(node.inputs())
                            if node.kind() == "aten::reshape":
                                shape_node = values[1].node()
                                shape = (
                                    tuple(value.toIValue() for value in shape_node.inputs())
                                    if shape_node.kind() == "prim::ListConstruct"
                                    else values[1].toIValue()
                                )
                                if values[0] == weight_input and shape is not None and len(shape) == 4:
                                    if not all(isinstance(size, int) for size in shape):
                                        raise ValueError("Cannot infer Linear feature axes; supply f_in and f_out")
                                    inferred_in, inferred_out = shape[1], shape[2]
                                    break
                                if values[0] == feature_input and shape is not None and len(shape) == 3:
                                    input_axis = shape[1]
                                    if not isinstance(input_axis, int):
                                        raise ValueError("Cannot infer Linear feature axes; supply f_in and f_out")
                            elif node.kind() == "aten::slice" and values[2].toIValue() == -2:
                                feature_rank = 2
                            elif node.kind() == "aten::add" and values[1].node().kind() == "prim::ListConstruct":
                                shape = tuple(value.toIValue() for value in values[1].node().inputs())
                                if len(shape) == 2:
                                    output_axis = shape[0]
                        else:
                            if feature_rank == 2:
                                if not isinstance(output_axis, int):
                                    raise ValueError("Cannot infer Linear feature axes; supply f_in and f_out")
                                inferred_in, inferred_out = input_axis or 1, output_axis
                            else:
                                inferred_in, inferred_out = None, None
                    else:
                        # FX reshape arguments carry the same feature axes.
                        for node in getattr(graph, "nodes", ()):
                            if node.op != "call_method" or node.target != "reshape":
                                continue
                            value = node.args[0]
                            if getattr(value, "op", None) != "placeholder" or value.target not in ("w", "weight"):
                                continue
                            shape = tuple(node.args[1:])
                            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                                shape = tuple(shape[0])
                            if len(shape) == 4:
                                if not all(isinstance(size, int) for size in shape):
                                    raise ValueError("Cannot infer Linear feature axes; supply f_in and f_out")
                                inferred_in, inferred_out = shape[1], shape[2]
                                break
                        else:
                            # Pruned contractions may retain only output allocation axes.
                            feature_rank, input_axis, output_axis = 1, None, None
                            for node in getattr(graph, "nodes", ()):
                                if node.op == "call_function" and node.target is operator.getitem:
                                    index = node.args[1]
                                    if isinstance(index, slice) and index.stop == -2:
                                        feature_rank = 2
                                if node.op == "call_function" and node.target is operator.add:
                                    shape = node.args[1]
                                    if isinstance(shape, tuple) and len(shape) == 2:
                                        output_axis = shape[0]
                                if node.op == "call_method" and node.target == "reshape":
                                    value, *shape = node.args
                                    if (getattr(value, "op", None) == "placeholder"
                                            and value.target in ("x", "features") and len(shape) == 3):
                                        input_axis = shape[1]
                            inferred_in, inferred_out = (
                                (input_axis or 1, output_axis)
                                if feature_rank == 2 and output_axis is not None else (None, None)
                            )
                f_in = inferred_in if f_in is None else f_in
                f_out = inferred_out if f_out is None else f_out
            if dtype is not None and tp.weight.dtype != dtype:
                tp.to(dtype=dtype)
        else:
            if irreps_out is MISSING:
                raise TypeError("missing required argument: irreps_out")
            constructor = LinearConfig if E3nnLinear is None else E3nnLinear
            options = dict(
                f_in=f_in, f_out=f_out, internal_weights=internal_weights,
                shared_weights=shared_weights, instructions=instructions,
                biases=biases, path_normalization=path_normalization,
                _optimize_einsums=_optimize_einsums,
            )
            if constructor is LinearConfig:
                tp = constructor(irreps_in, irreps_out, dtype=dtype, **options)
            else:
                tp = constructor(irreps_in, irreps_out, **options)
                if dtype is not None:
                    tp = tp.to(dtype=dtype)

        # Own parameters at the same names as e3nn, including when embedded in
        # another nn.Module. The source module shares these parameters,
        # but is not an extra registered level in state_dict/named_parameters.
        self.training = getattr(tp, "training", self.training)
        object.__setattr__(self, "tp", tp)
        for name in ("weight", "bias", "output_mask"):
            if name in tp._parameters:
                self.register_parameter(name, getattr(tp, name))
            else:
                self.register_buffer(name, getattr(tp, name))
        if hasattr(tp, "_compiled_main"):
            self._compiled_main = tp._compiled_main

        self.f_in, self.f_out = f_in, f_out
        self.irreps_in, self.irreps_out = self.tp.irreps_in, self.tp.irreps_out
        self.instructions = self.tp.instructions
        self.weight_index_slices = self.tp.weight_index_slices
        self._optimize_einsums = self.tp._optimize_einsums
        self.weight_numel, self.bias_numel = self.tp.weight_numel, self.tp.bias_numel
        self.internal_weights, self.shared_weights = (
            self.tp.internal_weights,
            self.tp.shared_weights,
        )

        self.config = LinearKernelConfig(
            tp, f_in=f_in, f_out=f_out, dtype=dtype
        )
        if dtype is None:
            self.config.compute_dtype = tp.weight.dtype
            self.config.meta['dtype'] = tp.weight.dtype

    @property
    def compute_dtype(self):
        return self.config.compute_dtype

    @property
    def meta(self):
        return self.config.meta

    @property
    def kernels(self):
        return self.config.kernels

    @property
    def raw_layout_kernels(self):
        return self.config.raw_layout_kernels

    @property
    def last_execution_backend(self):
        return self.config.last_execution_backend

    @property
    def last_execution_dtype(self):
        return self.config.last_execution_dtype

    @property
    def index_batch_span(self):
        return self.config.index_batch_span

    @property
    def index_workspace_span(self):
        return self.config.index_workspace_span

    def sync_parameters(self):
        for name in ("weight", "bias", "output_mask"):
            value = getattr(self, name)
            if getattr(self.tp, name) is not value:
                setattr(self.tp, name, value)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.sync_parameters()

    def weight_view_for_instruction(self, instruction, weight=None):
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return weight.narrow(-1, offset, prod(ins.path_shape)).view(
            weight.shape[:-1] + ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        offset = 0
        for index, ins in enumerate(self.instructions):
            size = prod(ins.path_shape)
            value = weight.narrow(-1, offset, size).view(weight.shape[:-1] + ins.path_shape)
            offset += size
            yield (index, ins, value) if yield_instruction else value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def _apply(self, fn, recurse=True):
        # Empty external buffers can have a different dtype from the selected
        # computation. Probe the computation dtype before mutating parameters.
        dtype = fn(torch.empty(0, dtype=self.compute_dtype, device=self.weight.device)).dtype
        result = super()._apply(fn, recurse=recurse)
        self.sync_parameters()
        if dtype != self.compute_dtype:
            self.config.compute_dtype = dtype
            self.config.meta = {**self.meta, "dtype": dtype}
            self.config.kernels = {}
        return result

    def forward(self, features, weight=None, bias=None):
        if weight is None and self.weight_numel and not self.internal_weights:
            raise RuntimeError("Weights must be provided when internal_weights = False")
        if bias is None and self.bias_numel and not self.internal_weights:
            raise RuntimeError("Biases must be provided when internal_weights = False")
        return linear(
            features, self.weight if weight is None else weight,
            self.bias if bias is None else bias, config=self.config,
        )


__all__ = ['LunaEqLinear']
