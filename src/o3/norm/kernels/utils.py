"""Norm specialization, source assembly, and CUDA launch caching."""

from math import gcd, prod

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from ....utils.indexing import LAYOUT_SOURCE, LAYOUT_HOST

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
NATIVE_DTYPES = (torch.float32, torch.float64)
BLOCK_SIZE = 128
MAX_BLOCKS = 65535
MAX_SHARED_BYTES = 48 * 1024
BACKWARD_MIN_BLOCKS_FP32 = 16
VECTORIZE_MIN_ROWS = 16384
MAX_PACKET_ELEMENTS = 32
TRIPLE_MIN_BLOCKS_FP32 = 8
TRIPLE_MIN_BLOCKS_FP64 = 5
TRIPLE_REGISTER_BOUND_COMPONENTS = 9

CUDA_HEADERS = """
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <optional>
#include <type_traits>
"""
CUDA_MATH = r"""
__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double mul_rn(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ double add_rn(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ float div_rn(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ double div_rn(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ float sqrt_rn(float a) { return __fsqrt_rn(a); }
__device__ __forceinline__ double sqrt_rn(double a) { return __dsqrt_rn(a); }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ double fma_rn(double a, double b, double c) { return __fma_rn(a, b, c); }

template<int Components>
__device__ __forceinline__ Element product(Element a, Element b) {
    if constexpr (Components == 1) return mul_rn(a, b);
    else return fma_rn(a, b, Element(0));
}

// Keep divisions outside the ReLU mask. A C++ conditional around a rounded
// division can become a divergent branch after the intrinsic is expanded.
__device__ __forceinline__ float relu_backward(float value, float input) {
    float result;
    asm("{ .reg .pred p; setp.le.f32 p, %1, 0f00000000; "
        "selp.f32 %0, 0f00000000, %2, p; }"
        : "=f"(result) : "f"(input), "f"(value));
    return result;
}
__device__ __forceinline__ double relu_backward(double value, double input) {
    double result;
    asm("{ .reg .pred p; setp.le.f64 p, %1, 0d0000000000000000; "
        "selp.f64 %0, 0d0000000000000000, %2, p; }"
        : "=d"(result) : "d"(input), "d"(value));
    return result;
}

// e3nn uses a scalar product for l=0, and an ordered matrix product otherwise.
template<int Components>
__device__ __forceinline__ Element dot(const Element* a, const Element* b) {
    Element result = 0;
    #pragma unroll
    for (int c = 0; c < Components; ++c) {
        if constexpr (Components == 1) result = mul_rn(a[c], b[c]);
        else result = fma_rn(a[c], b[c], result);
    }
    return result;
}
"""


def make_source(meta, order, *, strided=False, needs=(), seeds=(False, False)):
    from . import forward, backward, double_backward, triple_backward

    sources = (forward.CUDA_SOURCE, backward.CUDA_SOURCE,
               double_backward.CUDA_SOURCE, triple_backward.CUDA_SOURCE)
    scalar = "double" if meta["dtype"] == torch.float64 else "float"
    min_blocks = TRIPLE_MIN_BLOCKS_FP64 if scalar == "double" else TRIPLE_MIN_BLOCKS_FP32
    if any(path["components"] > TRIPLE_REGISTER_BOUND_COMPONENTS for path in meta["paths"]):
        min_blocks = 1
    max_components = max((path["components"] for path in meta["paths"]), default=1)
    vectorized = (order == 0 and not strided and scalar == "float" and not meta["squared"]
                  and 4 * max_components <= MAX_PACKET_ELEMENTS
                  and bool(meta["paths"]) and all(path["copies"] % 4 == 0 for path in meta["paths"]))
    # Aligned copy groups keep vector stores within a row.
    copy_tile = gcd(32, *(path["copies"] for path in meta["paths"]))
    shared_bytes = max_components * BLOCK_SIZE * (8 if scalar == "double" else 4)
    if order == 3:
        shared_bytes *= 2
    coalesced = order > 0 and copy_tile >= 8 and shared_bytes <= MAX_SHARED_BYTES
    flags = dict(COALESCED=int(coalesced), MAX_COMPONENTS=max_components,
                 COPY_TILE=copy_tile, ROWS_PER_WARP=32 // copy_tile,
                 PATH_TILES=sum((p["copies"] + copy_tile - 1) // copy_tile for p in meta["paths"]),
                 BLOCK_SIZE=BLOCK_SIZE, MAX_BLOCKS=MAX_BLOCKS,
                 VECTORIZE_MIN_ROWS=VECTORIZE_MIN_ROWS,
                 TRIPLE_MIN_BLOCKS=min_blocks,
                 INPUT_DIM=meta["in_dim"], OUTPUT_DIM=meta["out_dim"],
                 SQUARED=int(meta["squared"]), STRIDED=int(strided), VECTORIZED=int(vectorized),
                 WRITE_X=int(bool(needs and needs[0])),
                 WRITE_G=int(len(needs) > 1 and needs[1]),
                 WRITE_RX=int(len(needs) > 2 and needs[2]),
                 HAS_BX=int(seeds[0]), HAS_BG=int(seeds[1]))
    if order == 1:
        flags["BACKWARD_MIN_BLOCKS"] = BACKWARD_MIN_BLOCKS_FP32 if coalesced and scalar == "float" else 1
    prefix = CUDA_HEADERS + f"\nusing Element = {scalar};\n"
    prefix += "".join(f"#define {name} {value}\n" for name, value in flags.items())
    prefix += CUDA_MATH
    if order > 0:
        prefix += backward.CUDA_TILE_SOURCE
    if order == 0:
        prefix += LAYOUT_SOURCE + "\nForwardLayout make_forward_layout(at::Tensor x) {\n" + LAYOUT_HOST + "\n    return x_layout;\n}\n"
    paths, packets, tiles = [], [], []
    tile_offset = 0
    for path in meta["paths"]:
        copies, components = path["copies"], path["components"]
        offset, start = path["input_offset"], path["scalar_offset"]
        condition = f"if (group >= {start} && group < {start + copies})"
        position = f"({offset} + (group - {start}) * {components})"
        vector = f"(row * INPUT_DIM + {position})"
        output = "(row * OUTPUT_DIM + group)"
        if order == 0:
            call = f"out[{output}] = norm_forward<{components}>(x + x_row + {position} * stride, stride);"
            packets.append(f"{condition} *reinterpret_cast<float4*>(out + {output}) = norm_forward_packet<{components}>(x + {vector});")
        elif order == 1:
            call = f"norm_backward<{components}>(x, grad_out[{output}], dx, {vector}, {vector});"
        elif order == 2:
            call = f"norm_double_backward<{components}>(x, grad_out[{output}], rx, sx, sg, {vector}, {output}, {vector});"
        else:
            call = f"norm_triple_backward<{components}>(x, grad_out[{output}], rx, bx, HAS_BG ? bg[{output}] : Element(0), tx, tg, trx, {vector}, {output}, {vector});"
        paths.append(f"{condition} {call}")
        if order == 1:
            tiles.append(f"norm_backward_tile<{components}, {copies}, {offset}, {start}, {tile_offset}>(x, grad_out, dx, task, rows, shared);")
        elif order == 2:
            tiles.append(f"norm_double_backward_tile<{components}, {copies}, {offset}, {start}, {tile_offset}>(x, grad_out, rx, sx, sg, task, rows, shared);")
        elif order == 3:
            tiles.append(f"norm_triple_backward_tile<{components}, {copies}, {offset}, {start}, {tile_offset}>(x, grad_out, rx, bx, bg, tx, tg, trx, task, rows, shared_x, shared_rx);")
        tile_offset += (copies + copy_tile - 1) // copy_tile
    return prefix + (
        sources[order].replace("/* paths */", ("\n            " if order == 0 else "\n        ").join(paths))
        .replace("/* tiles */", "\n        ".join(tiles))
        .replace("/* packets */", "\n            ".join(packets))
    )


def load_forward_backward(op, *, arch, index_bits, strided=False):
    source = combine_cuda_sources({
        "forward": make_source(op.meta, 0, strided=strided),
        "backward": make_source(op.meta, 1),
    })
    extension = load_cuda_extension(
        "norm_forward_backward", source, arch=arch, index_bits=index_bits,
        use_fast_math=False, minimal_headers=True,
    )
    op.kernels[(1, (), (False, False), index_bits, arch, False)] = extension.backward.launch
    return extension.forward.launch


class NormKernels:
    """A fixed Norm configuration and its compiled CUDA entry points."""

    def __init__(self, *, meta, compute_dtype, index_bits, kernels):
        self.meta = meta
        self.compute_dtype = compute_dtype
        self.index_bits = index_bits
        self.kernels = kernels
        self.last_execution_backend = None
        self.backward_backend = None

    def run(self, order, x, grad_out=None, rx=None, bx=None, bg=None, *, needs=()):
        batch = x.shape[:-1]
        rows = prod(batch)
        width, out_width = self.meta["in_dim"], self.meta["out_dim"]
        strided = order == 0 and x.ndim <= 64 and not x.is_contiguous()
        bound = rows * max(width, out_width, 1) + MAX_BLOCKS * BLOCK_SIZE
        if strided and rows:
            bound = max(bound, sum((size - 1) * stride for size, stride in zip(x.shape, x.stride())) + MAX_BLOCKS * BLOCK_SIZE)
        if bound > (1 << 63) - 1:
            raise OverflowError("Norm offsets exceed signed int64 capacity")
        index_bits = 64 if bound > (1 << 31) - 1 else self.index_bits
        seeds = (bx is not None, bg is not None)
        if order == 3:
            needs = (needs[0] and (not self.meta["squared"] or seeds[1]),
                     needs[1] and seeds[0], needs[2])
        if order > 1 and not any(needs):
            return (None,) * len(needs)
        if out_width == 0:
            if order == 0:
                return x.new_empty((*batch, 0))
            return (None,) * len(needs) if order > 1 else x.new_empty(x.shape)
        if rows == 0:
            if order == 0:
                return x.new_empty((*batch, out_width))
            if order == 1:
                return x.new_empty(x.shape)
            return tuple(x.new_empty((*batch, dim)) if needed else None
                         for needed, dim in zip(needs, (width, out_width, width)))
        arch = get_cuda_arch(x.device)
        key = (order, tuple(needs), seeds, index_bits, arch, strided)
        launch = self.kernels.get(key)
        if launch is None:
            if order in (0, 1):
                forward = load_forward_backward(self, arch=arch, index_bits=index_bits, strided=strided)
                launch = forward if order == 0 else self.kernels[key]
            else:
                source = make_source(self.meta, order, needs=needs, seeds=seeds)
                launch = load_cuda_extension(
                    "norm_double_backward" if order == 2 else "norm_triple_backward",
                    source, arch=arch, index_bits=index_bits, use_fast_math=False,
                    minimal_headers=True,
                ).launch
            self.kernels[key] = launch
            while len(self.kernels) > 128:
                self.kernels.popitem(last=False)
        if order == 0:
            value = x if strided else x.contiguous()
            return launch(value, rows).view(*batch, out_width)
        if order == 1:
            return launch(x.contiguous(), grad_out.contiguous(), rows).view_as(x)
        if order == 2:
            result = launch(x.contiguous(), grad_out.contiguous(), rx.contiguous(), rows)
        else:
            result = launch(
                x.contiguous(), grad_out.contiguous(), rx.contiguous(),
                bx.contiguous() if bx is not None else None,
                bg.contiguous() if bg is not None else None, rows,
            )
        widths = (width, out_width, width)
        return tuple(value.view(*batch, dim) if value is not None else None
                     for value, dim in zip(result, widths))


def norm_product(components, x1, x2):
    shape = torch.broadcast_tensors(x1[..., :1], x2[..., :1])[0].shape[:-1]
    left = x1.expand(shape + (-1,)).reshape(-1, components)
    right = x2.expand(shape + (-1,)).reshape(-1, components)
    rows = left.shape[0]
    if components == 1:
        value = torch.einsum("ba,ba->ba", right, left)
    else:
        left = left.reshape(rows, 1, components)
        right = right.reshape(rows, 1, components)
        value = torch.einsum("cba,cba->cb", right, left)
    value = value.reshape(rows, 1)
    return value.reshape(shape + (1,))


