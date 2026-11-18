"""Channel projections on order-grouped SO(2) features."""

import ctypes
from functools import cache
import math
from math import prod
from pathlib import Path
from shutil import which
from typing import NamedTuple
from weakref import WeakValueDictionary, finalize

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import custom_op, triton_op, wrap_triton
from torch.nn import functional as F

from ..segmented_linear.interface import MixedSegmentVJP, project_mixed_segments
from ..segmented_linear.kernels.backward import mixed_input_vjp, mixed_weight_vjp
from ..segmented_linear.kernels.utils import mix_weights
from ...utils.extension import find_cutlass_include
from ...utils.jit import build_cuda_library, load_cuda_library, load_dispatch
from ..rotate_gather.interface import gather_rotated_sectors
from ..rotate_gather.kernels.utils import FUSED_WIGNER_GRID_E_STRIDE
from ..inverse_rotate_reduce.interface import (
    assemble_receiver_rows, pack_projection_weights, preprojection_reduce, project_frozen_receiver,
)
from ..utils.layout import check_orders
from .kernels.backward import (
    AMPERE_GATHER_VJP_SOURCE, degree_assembly_backward, fold_signed_weight, merge_complex_input_gradient, projected_bias_gradient,
    projection_input_vjp, small_degree_input_vjp,
)
from .kernels.double_backward import degree_assembly_double_backward
from .kernels.forward import (
    AMPERE_CORRECTION_SOURCE, AMPERE_GATHER_FORWARD_SOURCE, PROJECTION_LT_SOURCE,
    HOPPER_ROW_DISPATCH_SOURCE, HOPPER_ROW_SCALED_SOURCE, apply_sparse, assemble_projected_degrees,
    flattened_so2_linear, gather_with_row_scale, launch_degree_assembly, mix_receiver_weights, pack_signed_weight,
    project_small_degrees, raw_sector_linear,
)


class FrozenComplexProjection(torch.autograd.Function):
    """Use three real products for a frozen complex linear map."""

    @staticmethod
    def forward(ctx, features, weight):
        a, b = features.unbind(1)
        real, imag = weight.chunk(2, dim=0)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(real, imag)
        return F.linear(a, real), F.linear(b, imag), F.linear(a + b, real + imag)

    @staticmethod
    @once_differentiable
    def backward(ctx, dp, dq, dt):
        if dp is None and dq is None and dt is None:
            return None, None
        real, imag = ctx.saved_tensors
        if dp is not None and dq is not None and dt is not None:
            output = frozen_complex_input_vjp(dp, dq, dt, real, imag)
            if output is not None:
                return output, None
        left = None if dp is None else dp @ real
        right = None if dq is None else dq @ imag
        shared = None if dt is None else dt @ (real + imag)
        sample = next(value for value in (left, right, shared) if value is not None)
        edges, width = sample.shape
        output = sample.new_empty((edges, 2, width))
        if edges:
            block = 256
            merge_complex_input_gradient[(triton.cdiv(sample.numel(), block),)](
                sample if left is None else left, sample if right is None else right,
                sample if shared is None else shared, output, sample.numel(), width,
                left is not None, right is not None, shared is not None, block, num_warps=4,
            )
        return output, None


def project_sectors(
    sectors, scalar_map, order_maps, *,
    output_channels, scalar_prefix=0, apply_pair=flattened_so2_linear,
):
    """Apply channel maps and return ``(side, projected_sectors)``.

    Inputs and map shapes follow ``raw_sector_linear``. Let
    ``A=scalar_prefix``, ``Cout=output_channels``, and ``w_m=L+1-m``.
    The zero-order map produces ``(E,A+(L+1)*Cout)``. ``side`` is its
    leading ``(E,A)`` view, while ``projected_sectors[0]`` is the remaining
    ``(E,L+1,Cout)`` view. For ``m>0``, projected sectors are raw products
    ``(E,2,2*w_m*Cout)`` with real-map then imaginary-map outputs in
    each plane. They have not been recombined into physical coefficients.

    ``apply_pair`` defines map batching. Ordinary maps flatten edge and
    plane axes. Edge-conditioned maps retain the edge axis for per-edge
    weights. Device, dtype, and parameter derivatives follow those maps.
    PyTorch linear maps support derivatives through third order. The slicing
    uses views and leaves recombination to activation or aggregation.
    """
    if output_channels <= 0 or scalar_prefix < 0:
        raise ValueError("invalid output channel count or scalar prefix")
    raw = raw_sector_linear(sectors, scalar_map, order_maps, apply_pair=apply_pair)
    scalar = raw[0]
    width = scalar.shape[-1] - scalar_prefix
    if width < 0 or width % output_channels:
        raise ValueError("zero-order projection does not contain complete coefficients")
    side = scalar.narrow(-1, 0, scalar_prefix)
    output0 = scalar.narrow(-1, scalar_prefix, width).reshape(
        scalar.shape[0], width // output_channels, output_channels
    )
    return side, (output0, *raw[1:])


def linear(sectors, weights, bias=None, *, lmax, mmax):
    """Apply tensor-owned channel maps while retaining raw complex products.

    ``sectors`` and ``weights`` are tuples indexed by order ``0..M``.
    Sector zero flattens to ``(E,(L+1)*Cin)``. Other sectors have two
    physical planes, each of width ``(L+1-m)*Cin``. Weight zero has shape
    ``(A+(L+1)*Cout,(L+1)*Cin)``. A nonzero-order weight has shape
    ``(2*(L+1-m)*Cout,(L+1-m)*Cin)``. Optional bias belongs to order
    zero only. The scalar prefix ``A`` can hold gates or attention features.

    Return the complete zero-order output ``(E,A+(L+1)*Cout)`` and raw
    nonzero-order products ``(E,2,2*(L+1-m)*Cout)``. Recombination belongs
    to the subsequent activation or inverse-rotation reduction. Uses PyTorch
    matrix products across their supported devices and floating dtypes, with
    derivatives through third order.
    """
    sectors, weights = tuple(sectors), tuple(weights)
    check_orders(lmax, mmax, sectors)
    if len(weights) != len(sectors):
        raise ValueError("one weight matrix is required for each order")
    scalar = sectors[0]
    if scalar.ndim not in (2, 3) or (scalar.ndim == 3 and scalar.shape[1] != lmax + 1):
        raise ValueError("the scalar sector must have shape (E,K) or (E,L+1,C)")
    edges = scalar.shape[0]
    scalar_width = prod(scalar.shape[1:])
    if scalar_width == 0 or scalar_width % (lmax + 1):
        raise ValueError("order-zero features must contain all L+1 degree rows")
    channels = scalar_width // (lmax + 1)
    if weights[0].ndim != 2 or weights[0].shape[1] != scalar_width:
        raise ValueError("order-zero weights do not match the input channels")
    if bias is not None and tuple(bias.shape) != (weights[0].shape[0],):
        raise ValueError("the scalar bias must match the order-zero output width")
    output = [F.linear(scalar.reshape(edges, scalar_width), weights[0], bias)]
    output_channels = None
    for order in range(1, mmax + 1):
        value, weight = sectors[order], weights[order]
        width = lmax + 1 - order
        input_width = width * channels
        plane_shape = tuple(value.shape[1:])
        if (value.ndim not in (3, 4) or value.shape[0] != edges
                or plane_shape not in ((2, input_width), (2 * width, channels), (2, width, channels))):
            raise ValueError("nonzero sectors must contain positive and negative order planes")
        if weight.ndim != 2 or weight.shape[1] != input_width or weight.shape[0] % (2 * width):
            raise ValueError("nonzero-order weights require complete real and imaginary maps")
        current_channels = weight.shape[0] // (2 * width)
        if current_channels == 0 or (output_channels is not None and current_channels != output_channels):
            raise ValueError("nonzero-order maps must share a positive output channel count")
        output_channels = current_channels
        paired = value.reshape(edges * 2, input_width)
        output.append(F.linear(paired, weight).reshape(edges, 2, weight.shape[0]))
    if output_channels is not None and weights[0].shape[0] < (lmax + 1) * output_channels:
        raise ValueError("order-zero output cannot hold all degree coefficients at the output channel count")
    return tuple(output)


def gather_project(
    x, index, radial, rotation, scalar_map=None, order_maps=(), *, lmax=2, mmax=2,
    metadata=None, apply_pair=flattened_so2_linear, frozen=None, pair_metadata=None,
):
    """Gather, rotate, scale and apply the first order-wise projection.

    The trainable path accepts external channel maps and returns the raw
    real and imaginary products. A frozen plan returns the signed packed
    sectors consumed by the fused receiver, with one input derivative.
    Reciprocal radial rows on Ampere also require their pair indices.
    """
    if frozen is not None:
        if frozen.bias_plan is not None:
            if pair_metadata is None:
                raise ValueError("the reciprocal projection requires pair indices")
            pair_ptr, pairs = pair_metadata
            return project_orders(
                x, index, radial, rotation, pair_ptr, pairs, frozen.weights, frozen.bias,
                frozen.encoding, frozen.bias_plan,
            )
        return projection(
            x, index, rotation, radial, frozen.weights[0], frozen.bias,
            *frozen.vjp_weights, *frozen.encoding,
        )

    sectors = gather_rotated_sectors(
        x, index, radial, rotation, lmax=lmax, mmax=mmax, metadata=metadata,
        radial_sharing="order_degree", rotation_storage="retained_rows",
    )
    return raw_sector_linear(
        tuple(sector.contiguous() for sector in sectors), scalar_map, order_maps,
        apply_pair=apply_pair,
    )


def signed_projection_weight(weight):
    real, imag = weight.chunk(2, dim=0)
    return torch.cat((weight, torch.cat((-imag, real), dim=0)), dim=1)


class PackedSignedWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return pack_signed_weight(weight.resolve_neg())

    @staticmethod
    def backward(ctx, gradient):
        return FoldSignedWeight.apply(gradient)


class FoldSignedWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient):
        return fold_signed_weight(gradient.resolve_neg())

    @staticmethod
    def backward(ctx, gradient):
        return PackedSignedWeight.apply(gradient)


def project_recombined_sectors(sectors, weights, bias, *, pack_weight=signed_projection_weight):
    """Project flat sectors and return the signed planes for nonzero orders."""
    outputs = [F.linear(sectors[0], weights[0], bias)]
    for sector, weight in zip(sectors[1:], weights[1:], strict=True):
        # Each output plane is one row block of the real complex-product matrix.
        paired = pack_weight(weight)
        outputs.append(F.linear(sector.flatten(1), paired).unflatten(1, (2, -1)))
    return tuple(outputs)


class DegreeAssemblyVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, scale, bias, needs):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad, scale, bias)
        return degree_assembly_backward(grad, scale, bias, needs)

    @staticmethod
    @once_differentiable
    def backward(ctx, hy0, hy1, hy2, hs, hb):
        grad, scale, bias = ctx.saved_tensors
        nodes, _, channels = grad.shape
        hy = tuple(h.resolve_neg().contiguous() if h is not None else None for h in (hy0, hy1, hy2))
        hs = hs.resolve_neg().contiguous() if hs is not None else None
        hb = hb.resolve_neg().contiguous() if hb is not None else None
        has_y = tuple(h is not None for h in hy)
        has_scale, has_bias = hs is not None, hb is not None
        need_grad, need_scale, need_bias = ctx.needs_input_grad[:3]
        need_grad &= any(has_y) or has_scale or has_bias
        need_scale &= has_bias
        need_bias &= has_scale
        dg = torch.empty_like(grad) if need_grad else None
        ds = torch.empty_like(scale) if need_scale else None
        if need_grad or need_scale:
            launch_degree_assembly(degree_assembly_double_backward, nodes, channels,
                   (grad, scale, bias, *hy, hs, hb, dg, ds),
                   has_y, has_scale, has_bias, need_grad, need_scale)
        db = projected_bias_gradient(grad, hs, bias) if need_bias else None
        return dg, ds, db, None


class DegreeAssembly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, y1, y2, scale, bias):
        nodes, _, channels = y0.shape
        output = y0.new_empty((nodes, 9, channels))
        ctx.save_for_backward(scale, bias)
        launch_degree_assembly(assemble_projected_degrees, nodes, channels, (y0, y1, y2, scale, bias, output))
        return output

    @staticmethod
    def backward(ctx, grad):
        scale, bias = ctx.saved_tensors
        grad = grad.resolve_neg().contiguous()
        if torch.is_grad_enabled():
            return DegreeAssemblyVJP.apply(grad, scale, bias, ctx.needs_input_grad)
        return degree_assembly_backward(grad, scale, bias, ctx.needs_input_grad)


def assemble_degrees_with_bias(values, scale, bias):
    """Assemble three node tensors and apply the degree-specific rotated bias."""
    return DegreeAssembly.apply(*(value.contiguous() for value in values), scale.contiguous(), bias.contiguous())


class ReceiverWeightMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, *experts):
        ctx.save_for_backward(coefficients, *experts)
        return mix_receiver_weights(experts, coefficients)

    @staticmethod
    def backward(ctx, degree0, degree1, degree2):
        coefficients, *experts = ctx.saved_tensors
        # Invert the degree packing to use each order's original expert layout.
        order0 = torch.cat((degree0, degree1[..., :384], degree2[..., :384]), dim=1)
        order1 = torch.cat((degree1[..., 384:640], degree2[..., 384:640],
                            degree1[..., 640:896], degree2[..., 640:896]), dim=1)
        order2 = torch.cat((degree2[..., 896:1024], degree2[..., 1024:]), dim=1)
        gradients = tuple(value.flatten(1) for value in (order0, order1, order2))
        need_mix, *need_experts = ctx.needs_input_grad
        dcoefficients = None
        if need_mix:
            c0, c1, c2 = (grad @ w.flatten(1).T for grad, w in zip(gradients, experts, strict=True))
            dcoefficients = c0 + c1 + c2
        dexperts = tuple((coefficients.T @ grad).view_as(w) if needed else None
                         for grad, w, needed in zip(gradients, experts, need_experts, strict=True))
        return dcoefficients, *dexperts


def receiver_weights(experts, coefficients, channels):
    if (
        channels == 128
        and coefficients.is_cuda and coefficients.dtype == torch.float32
        and coefficients.ndim == 2 and 16 <= coefficients.shape[0] <= 64
        and coefficients.shape[1] == 64
        and tuple(w.shape for w in experts) == ((64, 384, 384), (64, 512, 256), (64, 256, 128))
        and all(w.is_contiguous() and not w.is_neg() and w.dtype == coefficients.dtype
                and w.device == coefficients.device for w in (coefficients, *experts))
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.is_autocast_enabled("cuda")
        and torch.cuda.get_device_capability(coefficients.device) in ((8, 0), (9, 0))
    ):
        return ReceiverWeightMix.apply(coefficients, *experts)

    mixed = tuple(mix_weights(weight, coefficients) for weight in experts)
    return pack_projection_weights(mixed[0], mixed[1:], channels)


class ReceiverForceVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, scale, w0, w1, w2, bias, offsets, ampere):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad, scale, w0, w1, w2, bias, offsets)
        ctx.ampere = ampere
        dx, ds = small_degree_input_vjp(grad, (w0, w1, w2), bias, ampere=ampere)
        db = projected_bias_gradient(grad, scale, bias) if ctx.needs_input_grad[5] else None
        return *dx, ds, db

    @staticmethod
    @once_differentiable
    def backward(ctx, hx0, hx1, hx2, hs, hb):
        grad, scale, w0, w1, w2, bias, offsets = ctx.saved_tensors
        weights = (w0, w1, w2)
        directions = tuple(h.resolve_neg().contiguous() if h is not None else None for h in (hx0, hx1, hx2))
        hs = hs.resolve_neg().contiguous() if hs is not None else None
        hb = hb.resolve_neg().contiguous() if hb is not None else None
        nodes = scale.shape[0]
        need_grad, need_scale = ctx.needs_input_grad[:2]
        need_weights = tuple(
            needed and h is not None for needed, h in zip(ctx.needs_input_grad[2:5], directions, strict=True)
        )
        need_bias = ctx.needs_input_grad[5]
        *dy, ds, db = degree_assembly_backward(
            grad, hs if hs is not None else scale, hb if hb is not None else bias,
            (*need_weights, need_scale and hb is not None, need_bias and hs is not None),
        )
        dw = tuple(mixed_weight_vjp(y, h, w, (nodes,), offsets) if needed else None
                   for y, h, w, needed in zip(dy, directions, weights, need_weights, strict=True))
        dg = None
        if need_grad:
            # Apply the same projection to the incoming feature and scale directions.
            x = tuple(h if h is not None else grad.new_zeros((nodes, 2 * degree + 1, w.shape[-1]))
                      for degree, (h, w) in enumerate(zip(directions, weights, strict=True)))
            dg = project_small_degrees(x, weights, hs if hs is not None else torch.zeros_like(scale),
                                       bias, ampere=ctx.ampere)
            if hb is not None:
                for degree in range(3):
                    rows = slice(degree ** 2, (degree + 1) ** 2)
                    dg[:, rows].add_(scale[:, rows, None] * hb[degree * 128:(degree + 1) * 128])
        return dg, ds, *dw, db, None, None


class SmallReceiverProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, x2, scale, w0, w1, w2, bias, offsets, ampere):
        ctx.save_for_backward(x0, x1, x2, scale, w0, w1, w2, bias, offsets)
        ctx.ampere = ampere
        return project_small_degrees((x0, x1, x2), (w0, w1, w2), scale, bias, ampere=ampere)

    @staticmethod
    def backward(ctx, grad):
        x0, x1, x2, scale, w0, w1, w2, bias, offsets = ctx.saved_tensors
        terms, weights = (x0, x1, x2), (w0, w1, w2)
        need_x = ctx.needs_input_grad[:3]
        need_scale = ctx.needs_input_grad[3]
        need_w = ctx.needs_input_grad[4:7]
        need_bias = ctx.needs_input_grad[7]
        grad = grad.resolve_neg().contiguous()
        sizes = (scale.shape[0],)

        fuse_input_vjp = all(need_x) and need_scale
        if torch.is_grad_enabled() and fuse_input_vjp:
            *dx, ds, db = ReceiverForceVJP.apply(grad, scale, *weights, bias, offsets, ctx.ampere)
            dw = (None, None, None)
        elif torch.is_grad_enabled():
            # The force VJP defers parameter gradients to double backward.
            active_w = tuple(w and not x for x, w in zip(need_x, need_w, strict=True))
            split = tuple(x or w for x, w in zip(need_x, active_w, strict=True))
            *dy, ds, db = DegreeAssemblyVJP.apply(grad, scale, bias, (*split, need_scale, need_bias))
            parts = tuple(
                MixedSegmentVJP.apply(g, x, w, sizes, offsets, False, nx, nw) if g is not None else (None, None)
                for g, x, w, nx, nw in zip(dy, terms, weights, need_x, active_w, strict=True)
            )
            dx, dw = zip(*parts, strict=True)
        else:
            split = need_w if fuse_input_vjp else tuple(x or w for x, w in zip(need_x, need_w, strict=True))
            *dy, ds, db = degree_assembly_backward(
                grad, scale, bias, (*split, need_scale and not fuse_input_vjp, need_bias),
            )
            if fuse_input_vjp:
                dx, ds = small_degree_input_vjp(grad, weights, bias, ampere=ctx.ampere)
            else:
                dx = tuple(mixed_input_vjp(g, w, sizes, False) if needed else None
                           for g, w, needed in zip(dy, weights, need_x, strict=True))
            dw = tuple(mixed_weight_vjp(g, x, w, sizes, offsets) if needed else None
                       for g, x, w, needed in zip(dy, terms, weights, need_w, strict=True))
        return *dx, ds, *dw, db, None, None


def project_after_inverse_reduce(
    sectors, rotation, receivers, row_ptr, experts=None, coefficients=None, bias=None, sizes=None,
    offsets=None, *, frozen=None,
):
    """Reduce messages before applying maps shared by the receiver's incoming edges.

    Training supplies activated physical sectors and the expert maps. With a
    prepared frozen receiver, supply the raw first-projection outputs instead.
    Its reduction applies the gate as it loads each edge.
    """
    if frozen is not None:
        return project_frozen_receiver(sectors, rotation, receivers, row_ptr, frozen)
    terms, scale = preprojection_reduce(sectors, rotation, receivers, row_ptr)
    weights = receiver_weights(experts, coefficients, bias.numel() // 3)
    nodes = scale.shape[0]
    if (
        len(sizes) == 1 and sizes[0] == nodes and 0 < nodes <= 80
        and 0 < rotation.shape[0] <= 4096
        and bias.shape == (384,) and bias.dtype == scale.dtype and bias.device == scale.device
        and offsets.shape == (2,) and offsets.dtype == torch.int32 and offsets.device == scale.device
        and offsets.is_contiguous() and not offsets.is_neg()
        and tuple(w.shape for w in weights) == ((1, 128, 384), (1, 128, 896), (1, 128, 1152))
        and all(w.dtype == scale.dtype and w.device == scale.device and w.is_contiguous() for w in weights)
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.is_autocast_enabled("cuda") and not torch.backends.cuda.matmul.allow_tf32
    ):
        capability = torch.cuda.get_device_capability(rotation.device)
        ampere = capability == (8, 0)
        if capability == (9, 0) or (ampere and nodes <= 16 and rotation.shape[0] <= 512):
            return SmallReceiverProjection.apply(
                *terms, scale, *weights, bias.resolve_neg().contiguous(), offsets, ampere,
            )
    values = tuple(project_mixed_segments(value, weight, sizes, offsets, force_only=True)
                   for value, weight in zip(terms, weights, strict=True))
    return assemble_degrees_with_bias(values, scale, bias)


def project_reduced_sectors(sectors, rotation, receivers, row_ptr, weights, bias, *, lmax, mmax):
    """Apply one shared order-wise projection after inverse rotation and reduction."""
    terms, scale = preprojection_reduce(
        sectors, rotation, receivers, row_ptr, lmax=lmax, mmax=mmax,
    )
    channels = bias.numel() // (lmax + 1)
    scalar, *pairs = weights
    weights = pack_projection_weights(scalar, pairs, channels)
    bias_rows = torch.cat(
        tuple(row.expand(2 * degree + 1, -1) for degree, row in enumerate(bias.reshape(-1, channels)))
    )
    values = tuple(F.linear(value, weight) for value, weight in zip(terms, weights, strict=True))
    return assemble_receiver_rows(values, scale, bias_rows)


class Residual(ctypes.Structure):
    _fields_ = [
        ("active", ctypes.c_void_p),
        ("total", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("columns", ctypes.c_void_p),
        ("edge_pair", ctypes.c_void_p),
        ("values", ctypes.c_void_p),
        ("scale", ctypes.c_float),
        ("cap", ctypes.c_int),
    ]


@cache
def ampere_projection_kernels():
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    encode = load_cuda_library("sm80_forward_strip.cu", AMPERE_GATHER_FORWARD_SOURCE, "80").launch_gather_forward
    encode.argtypes = [ptr] * 8 + [ctypes.POINTER(Residual), integer, integer, ptr]
    encode.restype = integer
    correct = load_cuda_library("sm80_correction.cu", AMPERE_CORRECTION_SOURCE, "80").launch_correction
    correct.argtypes = [ptr] * 6 + [ctypes.POINTER(Residual), ptr] + [integer] * 3 + [ptr]
    correct.restype = integer
    adjoint = load_cuda_library("sm80_gather_strip.cu", AMPERE_GATHER_VJP_SOURCE, "80").launch_gather_strip
    adjoint.argtypes = [ptr] * 12 + [integer] * 3 + [ptr]
    adjoint.restype = integer

    gemm = ctypes.CDLL("libcublas.so.12").cublasGemmEx
    gemm.argtypes = [ptr, integer, integer, integer, integer, integer, ptr, ptr, integer,
                     integer, ptr, integer, integer, ptr, ptr, integer, integer, integer, integer]
    gemm.restype = integer
    return encode, correct, adjoint, gemm


@cache
def projection_lt_library():
    lib = load_cuda_library("projection_lt.cpp", PROJECTION_LT_SOURCE, "80", flags=("-lcublasLt",))
    ptr, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    lib.plans_create.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_bool, ptr, size, integer, ctypes.POINTER(integer)]
    lib.plans_create.restype = ptr
    lib.plans_count.argtypes = [ptr]
    lib.plans_count.restype = integer
    lib.plans_run.argtypes = [ptr, integer] + [ptr] * 4 + [size, ctypes.c_float, ptr]
    lib.plans_run.restype = integer
    lib.plans_destroy.argtypes = [ptr]
    lib.plans_destroy.restype = None
    lib.plans_create_frozen.argtypes = [ctypes.c_uint64] * 3 + [ctypes.POINTER(integer)]
    lib.plans_create_frozen.restype = ptr
    lib.plans_run_frozen.argtypes = [ptr] * 8
    lib.plans_run_frozen.restype = integer
    return lib


@cache
def frozen_projection_library():
    from torch.utils.cpp_extension import CUDA_HOME, get_cxx_compiler

    if CUDA_HOME is None or not which(get_cxx_compiler()) or torch.version.cuda.split(".")[0] != "12":
        return None
    toolkit = Path(CUDA_HOME)
    if not all((toolkit / path).is_file() for path in (
        "bin/nvcc", "include/cublasLt.h", "lib64/libcublasLt.so", "lib64/libcublasLt.so.12",
    )):
        return None
    return projection_lt_library()


class FrozenProjectionPlan:
    def __init__(self, library, handle):
        self.run = library.plans_run_frozen
        self.handle = handle
        finalize(self, library.plans_destroy, handle)


# At most one recent plan for each supported order and CUDA device.
frozen_projection_plans = {}


def frozen_complex_input_vjp(dp, dq, dt, real, imag):
    edges = dp.shape[0]
    if (
        not dp.is_cuda or torch.version.hip is not None or torch.version.cuda is None
        or not 16384 <= edges <= 32768 or real.shape not in ((384, 768), (256, 512))
        or imag.shape != real.shape or dp.shape != (edges, real.shape[0])
        or dq.shape != dp.shape or dt.shape != dp.shape
        or torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled()
        or torch.get_float32_matmul_precision() != "highest" or torch.is_autocast_enabled("cuda")
        or torch.backends.cuda.matmul.allow_tf32
        or torch.cuda.get_device_capability(dp.device) != (9, 0)
    ):
        return None
    operands = dp, dq, dt, real, imag
    if any(t.device != dp.device or t.dtype != torch.float32 or not t.is_contiguous()
           or t.is_neg() or t.data_ptr() % 256 for t in operands):
        return None

    key = dp.device, real.shape
    planned_edges, plan = frozen_projection_plans.get(key, (None, None))
    with torch.cuda.device(dp.device):
        if planned_edges != edges:
            if torch.cuda.is_current_stream_capturing():
                return None
            library = frozen_projection_library()
            if library is None:
                return None
            status = ctypes.c_int()
            handle = library.plans_create_frozen(edges, real.shape[1], real.shape[0], ctypes.byref(status))
            if status.value:
                raise RuntimeError(f"cuBLASLt frozen projection plan failed ({status.value})")
            if not handle:
                frozen_projection_plans[key] = edges, None
                return None
            plan = FrozenProjectionPlan(library, handle)
        if plan is None:
            return None
        shared = dt @ (real + imag)
        output = dp.new_empty((edges, 2, real.shape[1]))
        stream = torch.cuda.current_stream(dp.device)
        status = plan.run(
            plan.handle, dp.data_ptr(), dq.data_ptr(), real.data_ptr(), imag.data_ptr(),
            shared.data_ptr(), output.data_ptr(), stream.cuda_stream,
        )
        if status:
            raise RuntimeError(f"cuBLASLt frozen projection input VJP failed ({status})")
        if planned_edges != edges:
            frozen_projection_plans[key] = edges, plan
    return output


class BiasPlan:
    def __init__(self, edges, weight, bias, workspace):
        self.workspace = workspace
        self.edge_count = edges
        self.lib = projection_lt_library()
        status = ctypes.c_int()
        self.handle = self.lib.plans_create(
            edges, *weight.shape, True, bias.data_ptr(), workspace.numel(), 12, ctypes.byref(status)
        )
        if not self.handle:
            raise RuntimeError(f"cuBLASLt bias plan creation failed with status {status.value}")
        self.release = finalize(self, self.lib.plans_destroy, self.handle)
        if not self.lib.plans_count(self.handle):
            self.release()
            raise RuntimeError("cuBLASLt has no FP16 bias algorithm for the calibrated graph")

    def run(self, weight, input, output, alpha, stream):
        return self.lib.plans_run(
            self.handle, 0, weight.data_ptr(), input.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), self.workspace.numel(), alpha, stream,
        )


plans = WeakValueDictionary()


def rounded(value):
    """Round FP32 operands to the A100 cuBLAS TF32 format."""
    bits = value.contiguous().view(torch.int32)
    return ((bits + 4096) & -8192).view(torch.float32)


def scale_for(value):
    maximum = float(value.abs().max())
    return 2.0 ** min(60, math.floor(math.log2(65504.0 / maximum))) if maximum else 1.0


def prepare_ampere_projection(weights, bias, input_scales, edge_count, workspace):
    """Encode frozen sector weights and prepare the order-zero cuBLASLt plan."""
    m0, m1, m2 = weights
    weights = m0.T.contiguous().T, m1.T.contiguous().T, m2
    if len(input_scales) != 3 or any(not math.isfinite(s) or s <= 0 or math.frexp(s)[0] != 0.5 for s in input_scales):
        raise ValueError("one positive power-of-two scale is required per order")
    ampere_projection_kernels()
    half_weights, transposed, weight_scales = [], [], []
    with torch.cuda.device(weights[0].device):
        for weight in weights:
            reference = rounded(weight.detach())
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("expected finite first-projection weights")
            scale = scale_for(reference)
            encoded = (reference * scale).half()
            normal = (encoded == 0) | (encoded.abs() >= 2.0**-14)
            if not torch.equal(encoded.float() / scale, reference) or not bool(normal.all()):
                raise ValueError("first-projection weights cannot be encoded exactly as scaled normal FP16")
            half_weights.append(encoded)
            transposed.append(reference.T.contiguous())
            weight_scales.append(scale)
        plan = BiasPlan(edge_count, half_weights[0], bias, workspace)
    plans[plan.handle] = plan
    return weights, (half_weights, transposed, list(input_scales), weight_scales), plan


class PackedInputs:
    def __init__(self, x, index, weights, scales):
        self.edges = index.shape[1]
        cap = 16
        self.inputs = [x.new_empty((self.edges, w.shape[1]), dtype=torch.float16) for w in weights]
        self.outputs = [x.new_empty((self.edges, w.shape[0])) for w in weights]
        # The producer writes counts for active rows before publishing them.
        self.totals = x.new_zeros((3,), dtype=torch.int32)
        self.metadata = []
        self.residuals = (Residual * 3)()
        for order, scale in enumerate(scales):
            active = x.new_empty((self.edges,), dtype=torch.int32)
            counts = torch.empty_like(active)
            columns = active.new_empty((self.edges, cap))
            edge_pair = torch.empty_like(active)
            values = x.new_empty((self.edges, cap))
            record = active, self.totals[order], counts, columns, edge_pair, values
            self.metadata.append(record)
            self.residuals[order] = Residual(*(v.data_ptr() for v in record), scale, cap)


@custom_op("trim_a100::project_orders", mutates_args=())
def project(x: Tensor, index: Tensor, radial: Tensor, rotation: Tensor, pairs: Tensor,
            bias: Tensor, half_weights: list[Tensor], transposed_weights: list[Tensor],
            input_scales: list[float], weight_scales: list[float], plan_handle: int) -> tuple[Tensor, Tensor, Tensor]:
    edges = index.shape[1]
    geometry = x, index, radial, rotation, pairs
    assert all(t.is_cuda and t.is_contiguous() for t in geometry)
    assert index.dtype == torch.int64 and pairs.dtype == torch.int32
    assert all(t.dtype == torch.float32 for t in (x, radial, rotation, bias, *transposed_weights))
    assert all(t.dtype == torch.float16 for t in half_weights)
    assert x.shape[1:] == (9, 128) and rotation.shape == (edges, 35)
    assert index.shape[0] == pairs.shape[0] == 2 and radial.shape == (pairs.shape[1], 1536)
    assert torch.backends.cuda.matmul.allow_tf32 and not torch.are_deterministic_algorithms_enabled()
    if edges == 0:
        return tuple(x.new_empty((0, w.shape[0])) for w in half_weights)
    encode, correct, _, gemm = ampere_projection_kernels()
    plan = plans[plan_handle]
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream().cuda_stream
        handle = torch.cuda.current_blas_handle()
        packed = PackedInputs(x, index, half_weights, input_scales)
        status = encode(*(t.data_ptr() for t in (*geometry, *packed.inputs)),
                        packed.residuals, edges, pairs.shape[1], stream)
        if status:
            raise RuntimeError(f"CUDA first-projection packing failed with status {status}")
        for order, (encoded, result, weight, transpose, sx, sw) in enumerate(zip(
            packed.inputs, packed.outputs, half_weights, transposed_weights, input_scales, weight_scales, strict=True,
        )):
            bias_order = bias if order == 0 else None
            alpha = 1.0 / (sx * sw)
            n, k = weight.shape
            if bias_order is not None and edges == plan.edge_count:
                status = plan.run(weight, encoded, result, alpha, stream)
            else:
                alpha_value, beta_value = ctypes.c_float(alpha), ctypes.c_float(0.0)
                # Column-major view of the transposed result. FP16 inputs, FP32 accumulation.
                status = gemm(handle, 1, 0, n, edges, k, ctypes.byref(alpha_value), weight.data_ptr(), 2, k,
                              encoded.data_ptr(), 2, k, ctypes.byref(beta_value), result.data_ptr(), 0, n, 68, 99)
                if status == 0 and bias_order is not None:
                    result.add_(bias_order)
            if status:
                raise RuntimeError(f"cuBLAS first projection failed with status {status}")
            status = correct(result.data_ptr(), transpose.data_ptr(), x.data_ptr(), index.data_ptr(),
                             radial.data_ptr(), rotation.data_ptr(), ctypes.byref(packed.residuals[order]),
                             None if bias_order is None else bias_order.data_ptr(), edges, n, order, stream)
            if status:
                raise RuntimeError(f"CUDA first-projection correction failed with status {status}")
    return tuple(packed.outputs)


@project.register_fake
def project_fake(x, index, radial, rotation, pairs, bias, half_weights, transposed_weights,
                 input_scales, weight_scales, plan_handle):
    return tuple(x.new_empty((index.shape[1], w.shape[0])) for w in half_weights)


@custom_op("trim_a100::projection_input_vjp", mutates_args=())
def gather_vjp(q0: Tensor, q1: Tensor, q2: Tensor, x: Tensor, index: Tensor,
               radial: Tensor, rotation: Tensor, pairs: Tensor, ptr: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    seeds = tuple(v.resolve_neg().contiguous() for v in (q0, q1, q2))
    dx, dr, dw = torch.zeros_like(x), torch.empty_like(radial), torch.empty_like(rotation)
    values = (*seeds, x, index, radial, rotation, pairs, ptr, dx, dr, dw)
    assert all(t.is_cuda and t.is_contiguous() for t in values)
    assert index.dtype == torch.int64 and ptr.dtype == pairs.dtype == torch.int32
    assert all(t.dtype == torch.float32 for t in (*seeds, x, radial, rotation))
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream()
        status = ampere_projection_kernels()[2](*(v.data_ptr() for v in values), index.shape[1], pairs.shape[1], x.shape[0],
                              stream.cuda_stream)
    if status:
        raise RuntimeError(f"CUDA projection input VJP failed with status {status}")
    return dx, dr, dw


@gather_vjp.register_fake
def gather_vjp_fake(q0, q1, q2, x, index, radial, rotation, pairs, ptr):
    return torch.empty_like(x), torch.empty_like(radial), torch.empty_like(rotation)


class ProjectOrders(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, radial, rotation, ptr, pairs, w0, w1, w2, bias,
                half_weights, transposed_weights, input_scales, weight_scales, plan_handle):
        ctx.save_for_backward(x, index, radial, rotation, pairs, ptr, w0, w1, w2)
        return project(x, index, radial, rotation, pairs, bias, half_weights, transposed_weights,
                       input_scales, weight_scales, plan_handle)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        *geometry, w0, w1, w2 = ctx.saved_tensors
        seeds = tuple(g.resolve_neg().contiguous() @ w for g, w in zip(gradients, (w0, w1, w2), strict=True))
        dx, dr, dw = gather_vjp(*seeds, *geometry)
        return dx, None, dr, dw, *([None] * 11)


def project_orders(x, index, radial, rotation, pair_row_ptr, pairs, weights, bias, encoding, plan):
    return ProjectOrders.apply(
        x.contiguous(), index.contiguous(), radial, rotation.contiguous(), pair_row_ptr, pairs,
        *weights, bias, *encoding, plan.handle,
    )


@cache
def configure_projection():
    """Load the Hopper GEMM and register its FakeTensor implementation before tracing."""
    cutlass_include = find_cutlass_include()
    row_scaled = build_cuda_library(
        "sm90_row_scaled.cu", HOPPER_ROW_SCALED_SOURCE, "90a",
        flags=("-Xptxas=-v", "--std=c++17", "--expt-relaxed-constexpr",
               f"-I{cutlass_include}"),
    )
    load_dispatch(
        "trim_row_scaled_dispatch", (("sm90_row_dispatch.cpp", HOPPER_ROW_DISPATCH_SOURCE),),
        arch="90", link=(str(row_scaled),),
    )

    @torch.library.register_fake("trim_h100::uma_row_scaled_matmul")
    def row_matmul_fake(x, weight, scale):
        return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


ROW_SCALE_RESIDUAL_CAPACITY = 8


@triton_op("trim_h100::uma_row_scaled_projection", mutates_args=())
def projection(
    x: Tensor, index: Tensor, rotation: Tensor, radial: Tensor,
    w0: Tensor, bias: Tensor, vjp0: Tensor, vjp1: Tensor, vjp2: Tensor,
    h1: Tensor, h2: Tensor, t1: Tensor, t2: Tensor, shift1: int, shift2: int,
) -> tuple[Tensor, Tensor, Tensor]:
    x, index, rotation, radial = (v.contiguous() for v in (x, index, rotation, radial))
    edges = index.shape[1]
    x0 = x.new_empty((edges, 768))
    half1, half2 = (x.new_empty((edges, k), dtype=torch.float16) for k in (1024, 512))
    inv1, inv2 = (x.new_empty(edges) for _ in range(2))
    count1, count2 = (x.new_empty(edges, dtype=torch.int32) for _ in range(2))
    columns1, columns2 = (count1.new_empty((edges, ROW_SCALE_RESIDUAL_CAPACITY)) for _ in range(2))
    residual1, residual2 = (x.new_empty((edges, ROW_SCALE_RESIDUAL_CAPACITY)) for _ in range(2))
    stride = FUSED_WIGNER_GRID_E_STRIDE
    wrap_triton(gather_with_row_scale)[(torch.sym_max(1, torch.sym_min(edges, stride)),)](
        x, index, rotation, radial, x0, half1, half2, inv1, inv2,
        count1, count2, columns1, columns2, residual1, residual2,
        edges, 128, *x.stride(), index.stride(0), 128, stride, shift1, shift2, ROW_SCALE_RESIDUAL_CAPACITY, num_warps=1,
    )
    y0 = F.linear(x0, w0, bias)
    y1 = torch.ops.trim_h100.uma_row_scaled_matmul.default(half1, h1, inv1)
    y2 = torch.ops.trim_h100.uma_row_scaled_matmul.default(half2, h2, inv2)
    rows_per_block = 4 if edges <= 8192 else 32
    wrap_triton(apply_sparse)[(triton.cdiv(edges, rows_per_block), 2)](
        y1, y2, t1, t2, count1, count2, columns1, columns2, residual1, residual2,
        half1, half2, inv1, inv2, x, index, rotation, radial, edges, *x.stride(), index.stride(0),
        shift1, shift2, ROW_SCALE_RESIDUAL_CAPACITY, rows_per_block, num_warps=4,
    )
    return y0, y1, y2


class FrozenProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, vjp_weight, bias):
        ctx.save_for_backward(vjp_weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return F.linear(grad.contiguous(), weight), None, None, None


def tf32(x):
    bits = x.view(torch.int32)
    rounded = (bits + 4095 + ((bits >> 13) & 1)) & -8192
    return torch.where((bits & 0x7f800000) == 0x7f800000, bits, rounded).view(torch.float32)


def encode_weight(weight):
    weight = tf32(weight.contiguous())
    magnitude = weight.abs().max()
    shift = 16 - torch.frexp(magnitude)[1]
    shift = torch.where(magnitude == 0, 0, shift)
    half = torch.ldexp(weight, shift).to(torch.float16)
    if not torch.equal(torch.ldexp(half.float(), -shift), weight):
        raise ValueError("TF32 weights do not fit the scaled FP16 representation")
    return half, weight.T.contiguous(), int(shift)


class FrozenGatherProjection(NamedTuple):
    weights: tuple[Tensor, ...]
    bias: Tensor
    encoding: tuple
    vjp_weights: tuple[Tensor, ...] = ()
    bias_plan: BiasPlan | None = None

    def supports(self, x, index, radial, rotation):
        return (
            x.dtype == torch.float32 and rotation.ndim == 2 and rotation.shape[1] == 35
            and x.shape[1:] == (9, 128) and radial.shape[1] == 1536 and index.shape[1] > 0
            and not torch.are_deterministic_algorithms_enabled()
        )

    def from_packed(self, *sectors):
        """Project packed sectors from an external gather implementation."""
        return tuple(
            FrozenProjection.apply(sector, weight, adjoint, bias)
            for sector, weight, adjoint, bias in zip(
                sectors, self.weights, self.vjp_weights, (self.bias, None, None), strict=True,
            )
        )


def prepare_gather_project(
    scalar_weight, paired_weights, bias, *, input_scales=None, edge_count=None, workspace=None,
):
    """Prepare the frozen first projection for the weight device."""
    paired_weights = tuple(paired_weights)
    weights = scalar_weight, *paired_weights
    if any(value.requires_grad for value in (*weights, bias)):
        raise ValueError("the packed first projection requires frozen evaluation")
    if any(not w.is_cuda or w.device != scalar_weight.device or w.dtype != torch.float32
           for w in (*weights, bias)):
        raise ValueError("first-projection weights and bias must be CUDA FP32 tensors on one device")
    if tuple(w.shape for w in weights) != ((640, 768), (512, 512), (256, 256)) or bias.shape != (640,):
        raise ValueError("the packed first projection requires the L=M=2, C=128 sector layout")

    packed = (scalar_weight.detach(), *(
        signed_projection_weight(weight.detach()).contiguous() for weight in paired_weights
    ))

    capability = torch.cuda.get_device_capability(scalar_weight.device)
    if capability == (8, 0):
        if input_scales is None or edge_count is None or workspace is None:
            raise ValueError("Ampere requires input scales, edge count and cuBLASLt workspace")
        weights, encoding, plan = prepare_ampere_projection(
            packed, bias, input_scales, edge_count, workspace,
        )
        return FrozenGatherProjection(weights, bias, encoding, bias_plan=plan)
    if capability == (9, 0):
        configure_projection()
        h1, t1, shift1 = encode_weight(packed[1])
        h2, t2, shift2 = encode_weight(packed[2])
        adjoints = tuple(weight.T.contiguous() for weight in packed)
        return FrozenGatherProjection(
            packed, bias.detach(), (h1, h2, t1, t2, shift1, shift2), adjoints,
        )
    raise ValueError("the packed first projection requires Ampere SM80 or Hopper SM90")


def register_gather_projection_vjp(gather_vjp):
    """Bind the input derivative for the external Hopper gather layout."""
    def save_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:4], *inputs[6:9])

    def backward(ctx, *grad):
        x, index, rotation, radial, *weights = ctx.saved_tensors
        dx, dw, dr = projection_input_vjp(x, index, rotation, radial, weights, grad, gather_vjp)
        return dx, None, dw, dr, *([None] * 11)

    projection.register_autograd(backward, setup_context=save_context)
