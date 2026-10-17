"""Precision-specific CUDA source generation; never used on a launch hot path."""

from __future__ import annotations

import re

import torch


def float_literal(x: float, dtype: torch.dtype | None = None) -> str:
    # Seventeen fractional digits round-trip a Python/C++ double.
    # The trailing ``f`` selects a float literal for FP32 kernels.
    # ``cuda_source_to_fp64`` strips it to preserve double-precision coefficients.
    return f"{x:.17e}" + ("" if dtype == torch.float64 else "f")


def suffix_float32_literals(source: str) -> str:
    """Add float suffixes to generated decimal literals, leaving their text intact."""
    return re.sub(
        r"(?<![\w.])(?:\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])",
        r"\g<0>f",
        source,
    )


def cuda_source_to_fp64(source: str) -> str:
    """Convert the generated CUDA dialect, with a guarded PTX select template.

    Converts the type tokens and intrinsics used by generated kernels.
    Unrecognized select2_f32 helpers raise RuntimeError.
    """
    select2_f32_body = """    float result;
    asm volatile(
        "{ .reg .pred p0; "
        "setp.eq.s32 p0, %1, 0; "
        "selp.f32 %0, %2, %3, p0; }"
        : "=f"(result)
        : "r"(index), "f"(value0), "f"(value1));
    return result;"""
    if "select2_f32" in source:
        converted = source.replace(
            select2_f32_body,
            "    return index == 0 ? value0 : value1;",
        )
        if converted == source:
            raise RuntimeError("unrecognized select2_f32 helper in FP64 source")
        source = converted
    source = source.replace("_fp32", "_fp64")
    source = source.replace("torch::kFloat32", "torch::kFloat64")
    source = re.sub(r"\bfloat32\b", "float64", source)
    # Convert complete CUDA type tokens, never substrings of identifiers.
    source = re.sub(r"\bfloat([234]?)\b", r"double\1", source)
    source = re.sub(
        r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fF]\b",
        r"\1",
        source,
    )
    source = source.replace(".f32", ".f64").replace('"=f"', '"=d"').replace('"f"(', '"d"(')
    source = re.sub(r"\b__fmaf_rn\b", "__fma_rn", source)
    source = re.sub(r"\bfmaf(?=\s*\()", "fma", source)
    return source


def cuda_source_for_dtype(source: str, dtype: torch.dtype = torch.float32) -> str:
    """Finish a generated CUDA dialect without changing FP32 source text."""
    if dtype == torch.float32:
        return source
    if dtype == torch.float64:
        return cuda_source_to_fp64(source)
    raise ValueError(f"CUDA generation requires float32 or float64, got {dtype}")


def get_fp64_mma(arch: str) -> tuple[str, str]:
    """Return the CUDA header and matrix-product type for a compile target.

    CUTLASS supports 16x8x16 on SM90+ and 8x8x4 on SM80+. Compose the
    smaller operation in the same register layout on SM8x. FP64 throughput
    depends on the GPU.
    """
    capability = int(arch.rstrip("af"))
    if capability < 80:
        raise ValueError("FP64 MMA requires compute capability 8.0 or newer")
    if capability >= 90:
        return (
            "#include <cute/arch/mma_sm90.hpp>",
            "cute::SM90::MMA_16x8x16_F64F64F64F64_TN",
        )
    header = """
#include <cuda_runtime.h>
#include <cute/arch/mma_sm80.hpp>
struct Fp64Mma16x8x16 {
    __device__ __forceinline__ static void fma(
        double& d0, double& d1, double& d2, double& d3,
        double a0, double a1, double a2, double a3,
        double a4, double a5, double a6, double a7,
        double b0, double b1, double b2, double b3,
        double c0, double c1, double c2, double c3) {
        const double a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
        const double b[4] = {b0, b1, b2, b3};
        d0 = c0; d1 = c1; d2 = c2; d3 = c3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            cute::SM80_8x8x4_F64F64F64F64_TN::fma(
                d0, d1, a[k * 2], b[k], d0, d1);
            cute::SM80_8x8x4_F64F64F64F64_TN::fma(
                d2, d3, a[k * 2 + 1], b[k], d2, d3);
        }
    }
};
"""
    return header, "Fp64Mma16x8x16"

