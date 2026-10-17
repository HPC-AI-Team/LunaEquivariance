"""Linear configuration, e3nn conversion and contraction metadata.

Only construction uses this module. Execution and autograd live in interface.py.
"""

from collections import OrderedDict, namedtuple
from functools import lru_cache
from math import prod
import torch

from ..irreps import Irreps
from .kernels.utils import MAX_WEIGHT_SPLITS, TASK_ALIGNMENT


Instruction = namedtuple("Instruction", "i_in i_out path_shape path_weight")
LinearSlices = namedtuple("LinearSlices", "slice_1D shape_2D")


class LinearLayout:
    """Parameter-free irreps, instructions and normalization."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        f_in=None,
        f_out=None,
        internal_weights=None,
        shared_weights=None,
        instructions=None,
        biases=False,
        path_normalization="element",
        dtype=None,
        _optimize_einsums=None,
    ):
        if path_normalization not in ("element", "path"):
            raise ValueError("path_normalization must be element or path")
        if (f_in is None) != (f_out is None) or (f_in is not None and min(f_in, f_out) <= 0):
            raise ValueError("f_in and f_out must both be positive or both be None")

        self.irreps_in, self.irreps_out = Irreps(irreps_in), Irreps(irreps_out)
        self._optimize_einsums = True if _optimize_einsums is None else _optimize_einsums
        self.f_in, self.f_out = f_in, f_out
        self.shared_weights = shared_weights is not False
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("internal_weights requires shared_weights")

        paths = (
            [
                (i, j)
                for i, (_, a) in enumerate(self.irreps_in)
                for j, (_, b) in enumerate(self.irreps_out)
                if a == b
            ]
            if instructions is None
            else instructions
        )
        self.instructions = []
        for i, j in paths:
            if i < 0 or j < 0 or i >= len(self.irreps_in) or j >= len(self.irreps_out):
                raise IndexError("Linear instruction index out of range")
            if self.irreps_in[i].ir != self.irreps_out[j].ir:
                raise ValueError("Linear paths must connect identical irreps")
            count = sum(
                self.irreps_in[a if path_normalization == "element" else i].mul
                for a, b in paths
                if b == j
            ) * (f_in or 1)
            self.instructions.append(
                Instruction(
                    i, j, (self.irreps_in[i].mul, self.irreps_out[j].mul), max(1, count) ** -0.5
                )
            )

        if biases is None:
            biases = False
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in self.irreps_out]
        if len(biases) != len(self.irreps_out) or any(
            b and not ir.is_scalar() for b, (_, ir) in zip(biases, self.irreps_out)
        ):
            raise ValueError("Linear biases are allowed only on even scalar irreps")
        self.instructions += [
            Instruction(-1, j, (ir.dim,), 1.0)
            for j, (b, ir) in enumerate(zip(biases, self.irreps_out))
            if b
        ]

        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.i_in >= 0)
        self.bias_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.i_in == -1)
        self.weight_index_slices = []
        offset = 0
        for ins in self.instructions:
            size = prod(ins.path_shape)
            self.weight_index_slices.append(
                LinearSlices(slice(offset, offset + size), ins.path_shape)
            )
            offset += size


class LinearConfig(torch.nn.Module, LinearLayout):
    """Parameter initialization when e3nn is unavailable."""

    def __init__(self, irreps_in, irreps_out, *, dtype=None, **options):
        torch.nn.Module.__init__(self)
        LinearLayout.__init__(self, irreps_in, irreps_out, dtype=dtype, **options)
        f_in, f_out = self.f_in, self.f_out
        weight_shape = (() if f_in is None else (f_in, f_out)) + (self.weight_numel,)
        bias_shape = (() if f_out is None else (f_out,)) + (self.bias_numel,)
        if self.internal_weights and self.weight_numel:
            self.weight = torch.nn.Parameter(torch.randn(weight_shape, dtype=dtype))
        else:
            self.register_buffer("weight", torch.empty(0, dtype=dtype))
        if self.internal_weights and self.bias_numel:
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape, dtype=dtype))
        else:
            self.register_buffer("bias", torch.empty(0, dtype=dtype))

        active = {ins.i_out for ins in self.instructions if 0 not in ins.path_shape}
        self.register_buffer(
            "output_mask",
            (
                torch.cat(
                    [
                        torch.full((ir.dim,), float(j in active), dtype=dtype)
                        for j, ir in enumerate(self.irreps_out)
                    ]
                )
                if len(self.irreps_out)
                else torch.empty(0, dtype=dtype)
            ),
        )


def build_linear_metadata(linear, dtype=torch.float32, *, f_in=None, f_out=None):
    """Lower exact path normalization, bias and optional feature axes."""
    fin, fout = f_in or 1, f_out or 1
    in_slices, out_slices = linear.irreps_in.slices(), linear.irreps_out.slices()
    specs, biases, weight_gaps = [], [], []
    wi = bi = 0
    for ins in linear.instructions:
        if ins.i_in == -1:
            count = prod(ins.path_shape)
            for fo in range(fout) if count and ins.i_out >= 0 else ():
                biases.append(
                    dict(
                        out_offset=fo * linear.irreps_out.dim + out_slices[ins.i_out].start,
                        bias_offset=fo * linear.bias_numel + bi,
                        channels=count,
                    )
                )
            bi += count
            continue
        cin, cout = ins.path_shape
        components = linear.irreps_in[ins.i_in].ir.dim
        # e3nn accepts Python negative input indices. Negative output indices
        # own weights but are absent from its enumerated output instructions.
        if cin and cout and ins.i_out >= 0:
            for fi in range(fin):
                for fo in range(fout):
                    specs.append(
                        dict(
                            x_offset=fi * linear.irreps_in.dim + in_slices[ins.i_in].start,
                            out_offset=fo * linear.irreps_out.dim + out_slices[ins.i_out].start,
                            weight_offset=(fi * fout + fo) * linear.weight_numel + wi,
                            in_channels=cin,
                            out_channels=cout,
                            components=components,
                            path_weight=float(ins.path_weight),
                        )
                    )
        elif cin and cout:
            for fi in range(fin):
                for fo in range(fout):
                    weight_gaps.append(((fi * fout + fo) * linear.weight_numel + wi, cin * cout))
        wi += cin * cout

    if wi != linear.weight_numel or bi != linear.bias_numel:
        raise ValueError("Linear instruction counts disagree with weight/bias size")
    return dict(
        in_dim=fin * linear.irreps_in.dim,
        out_dim=fout * linear.irreps_out.dim,
        weight_numel=fin * fout * wi,
        bias_numel=fout * bi,
        shared_weights=bool(linear.shared_weights),
        dtype=dtype,
        max_channels=max((max(s["in_channels"], s["out_channels"]) for s in specs), default=1),
        generated_specs=specs,
        bias_specs=biases,
        weight_gaps=tuple(weight_gaps),
    )


class LinearKernelConfig:
    """Parameter-free layout and reusable kernel records, also usable without Module."""

    def __init__(self, layout, *, f_in=None, f_out=None, dtype=None, index_bits=32):
        self.irreps_in, self.irreps_out = layout.irreps_in, layout.irreps_out
        self.input_dim, self.output_dim = layout.irreps_in.dim, layout.irreps_out.dim
        self.instructions = tuple(layout.instructions)
        self.weight_numel, self.bias_numel = layout.weight_numel, layout.bias_numel
        self.shared_weights = layout.shared_weights
        self.optimize_einsums = getattr(layout, "_optimize_einsums", True)
        self.f_in, self.f_out = f_in, f_out
        self.compute_dtype = torch.get_default_dtype() if dtype is None else dtype
        self._explicit_dtype = dtype is not None
        if index_bits not in (32, 64):
            raise ValueError("index_bits must be 32 or 64")
        self.index_bits = index_bits
        self.meta = build_linear_metadata(layout, self.compute_dtype, f_in=f_in, f_out=f_out)
        dimensions = (
            self.meta["in_dim"],
            self.meta["out_dim"],
            self.meta["weight_numel"],
            self.meta["bias_numel"],
        )
        # CUDA tile coordinates use signed int32. Keep legal e3nn metadata
        # constructible and select compatibility before any native compile.
        self._cuda_dimensions_supported = (
            not (self.meta["generated_specs"] or self.meta["bias_specs"])
            or max(dimensions) <= (1 << 31) - 1
        )
        virtual_weights = sum(
            ((s["in_channels"] * s["out_channels"] + TASK_ALIGNMENT - 1) // TASK_ALIGNMENT)
            * TASK_ALIGNMENT
            for s in self.meta["generated_specs"]
        )
        max_components = max((s["components"] for s in self.meta["generated_specs"]), default=1)
        self.max_components = max_components
        self.index_batch_span = max(
            self.meta["in_dim"] + self.meta["out_dim"] + virtual_weights,
            max_components,
            *dimensions,
            1,
        )
        self.index_workspace_span = MAX_WEIGHT_SPLITS * self.meta["weight_numel"]

        self.kernels = {}
        self._programs = {}
        self.raw_layout_kernels = OrderedDict()
        self.last_execution_backend = None
        self.layout_key = (
            str(self.irreps_in), str(self.irreps_out), self.instructions,
            self.weight_numel, self.bias_numel, self.shared_weights,
            self.optimize_einsums, self.f_in, self.f_out, self.index_bits,
        )

    @torch.compiler.assume_constant_result
    def register_ops(self, dtype):
        from .interface import register_ops

        return register_ops(self.layout_key, dtype)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def linear_config(irreps_in, irreps_out, f_in, f_out, shared_weights, instructions,
                  biases, path_normalization, dtype, index_bits):
    layout = LinearLayout(
        irreps_in, irreps_out, f_in=f_in, f_out=f_out, shared_weights=shared_weights,
        internal_weights=False, instructions=instructions, biases=biases,
        path_normalization=path_normalization,
    )
    return LinearKernelConfig(layout, f_in=f_in, f_out=f_out, dtype=dtype, index_bits=index_bits)

