"""UVW forward CUDA templates."""

import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal, get_fp64_mma
from ....utils.extension import get_cuda_arch, load_cuda_extension
from .utils import (
    SCALAR_CUDA_SOURCE, SCALAR_PARALLEL_CUDA_SOURCE, UVW_FWD_ACCUMULATOR_LIMIT,
    UVW_FWD_LARGE_BATCH_THRESHOLD, UVW_SCALAR_WORKSPACE_ELEMENTS, UVW_FWD_MAX_SPLITS,
    UVW_GEMM_BLOCKS_PER_SM, UVW_GEMM_TARGET_BLOCKS_PER_SM,
)


def make_uvw_forward_external_kernel(
    meta,
    products=(((("x", "y"),), "w"),),
    kernel_name="lunatp_uvw_fwd_external_fused_fp32",
    tile_channels=128,
    pipeline_weights=False,
    *,
    arch,
):
    """Return CUDA kernel, grid expression, zeroing flag and input pointer order.

    Each product is (CG input pairs, weight pointer). Grouping pairs that share a
    weight avoids repeating its matrix product in double and triple backward.
    """
    if meta.get("dtype") is torch.float64 and pipeline_weights:
        specs = meta["generated_specs"]
        components = max(spec["d3"] for spec in specs)
        aligned_tiles = all(
            spec["out_channels"] % 64 == 0 and spec["path_channels"] % 16 == 0
            for spec in specs
        )
        # One M16 fragment bounds the register state; complete N64/K16 tiles
        # avoid repeating CG work for padded external-weight products.
        if (
            aligned_tiles
            and 8 <= components <= 16
            and products == (((("x", "y"),), "w"),)
        ):
            return make_uvw_forward_external_mma_kernel(
                meta, kernel_name, tile_channels, arch=arch
            )
        # Cheap CG rows benefit from a smaller double accumulator fragment.
        if aligned_tiles and components * 2 <= UVW_FWD_ACCUMULATOR_LIMIT:
            tile_channels = min(tile_channels, 32)
    groups = {}
    for spec in meta["generated_specs"]:
        groups.setdefault(
            (spec["out_offset"], spec["out_channels"], spec["d3"]), []
        ).append(spec)
    max_tiles = max(
        (channels + tile_channels - 1) // tile_channels for _, channels, _ in groups
    )
    zero_output = sum(c * d for _, c, d in groups) != meta["out_dim"]
    used = {name for pairs, weight in products for pair in pairs for name in pair}
    used.update(weight for _, weight in products)
    inputs = tuple(
        name
        for name in ("x", "y", "w", "rx", "ry", "rw", "bx", "by", "bw")
        if name in used
    )
    multiple = len(products) > 1
    # Groups share one CUDA register allocation, so size the ring for the
    # largest per-lane accumulator fragment in this kernel.
    accumulator_values = max(
        d3 * min(tile_channels // 32, (channels + 31) // 32)
        for _, channels, d3 in groups
    )
    rows = (
        2 if pipeline_weights and accumulator_values <= UVW_FWD_ACCUMULATOR_LIMIT else 1
    )
    cases = []
    for group, ((offset, channels, d3), specs) in enumerate(groups.items()):
        columns = min(tile_channels // 32, (channels + 31) // 32)
        vector = (
            pipeline_weights
            and rows == 1
            and columns in (2, 4)
            and channels % columns == 0
            and meta["weight_numel"] % columns == 0
            and all(spec["weight_offset"] % columns == 0 for spec in specs)
        )
        width = columns * 32
        lines = [
            r"""if (group == {group}) {{
Index item=(task/{max_tiles})*4+(threadIdx.x>>5);
int tile_column=(task%{max_tiles})*{width};
if (item>=items || tile_column>={channels}) return;
{column_declaration}
using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{d3},{columns},1>,
float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
Mma::FragmentC accum; accum.clear();""".format(
                group=group,
                max_tiles=max_tiles,
                width=width,
                channels=channels,
                column_declaration=f"int column=tile_column+lane*{columns};"
                if vector
                else "int column=tile_column+lane;",
                d3=d3,
                columns=columns,
            )
        ]
        for spec in specs:
            size, edges = (
                spec["path_channels"],
                spec["edge_channels"],
            )
            lines.append("{")
            if multiple:
                # Keep one matrix loop instead of duplicating it for each derivative term.
                lines += [
                    "#pragma unroll 1",
                    f"for(int product=0;product<{len(products)};++product) {{",
                ]
                choices = {
                    "a": [pairs[0][0] for pairs, _ in products],
                    "b": [pairs[0][1] for pairs, _ in products],
                    "a1": [
                        pairs[1][0] if len(pairs) == 2 else "nullptr"
                        for pairs, _ in products
                    ],
                    "b1": [
                        pairs[1][1] if len(pairs) == 2 else "nullptr"
                        for pairs, _ in products
                    ],
                    "weight": [weight for _, weight in products],
                }
                for name, values in choices.items():
                    expression = values[-1]
                    for q in reversed(range(len(values) - 1)):
                        expression = f"(product=={q}?{values[q]}:{expression})"
                    lines.append(f"const float* {name}={expression};")
            weight = "weight" if multiple else products[0][1]
            lines += [
                f"constexpr Index begin=0,end={size};",
                "#pragma unroll 1",
                "for (int base=begin;base<end;base+=32) {",
            ]
            buffers = ("next",) if rows == 1 else ("next0", "next1")
            for slot, buffer in enumerate(buffers):
                lines.append(f"Mma::FragmentB {buffer};")
                row = "base" if rows == 1 else f"(base+{slot})"
                weight_channels = spec["out_channels"]
                weight_offset = spec["weight_offset"]
                row_mask = f"base+{slot}<end && " if rows > 1 else ""
                if vector:
                    lines += [f"{buffer}[{n}]=0.f;" for n in range(columns)]
                    lines += [
                        f"if({row_mask}column+{columns - 1}<{weight_channels}) {{",
                        f"""const float* src={weight}+int64_t(item)*{meta["weight_numel"]}+{weight_offset}+{row}*{weight_channels}+column;""",
                    ]
                    if meta.get("dtype") is torch.float64 and columns == 4:
                        lines += [
                            f'asm volatile("ld.global.L1::no_allocate.v2.f32 {{%0,%1}},[%2];":"=f"({buffer}[0]),"=f"({buffer}[1]):"l"(src));',
                            f'asm volatile("ld.global.L1::no_allocate.v2.f32 {{%0,%1}},[%2];":"=f"({buffer}[2]),"=f"({buffer}[3]):"l"(src+2));',
                        ]
                    else:
                        values = ",".join(f"%{n}" for n in range(columns))
                        constraints = ",".join(
                            f'"=f"({buffer}[{n}])' for n in range(columns)
                        )
                        lines.append(
                            f'asm volatile("ld.global.L1::no_allocate.v{columns}.f32 {{{values}}},[%{columns}];":{constraints}:"l"(src));'
                        )
                    lines.append("}")
                else:
                    for n in range(columns):
                        lines += [
                            r"""{buffer}[{n}]=0.f; if({row_mask}column+{channel_offset}<{weight_channels}) {{
const float* src={weight}+int64_t(item)*{weight_numel}+{weight_offset}+{row}*{weight_channels}+column+{channel_offset};
asm volatile("ld.global.L1::no_allocate.f32 %0,[%1];":"=f"({buffer}[{n}]):"l"(src));
}}""".format(
                                buffer=buffer,
                                n=n,
                                row_mask=row_mask,
                                channel_offset=n * 32,
                                weight_channels=weight_channels,
                                weight=weight,
                                weight_numel=meta["weight_numel"],
                                weight_offset=weight_offset,
                                row=row,
                            )
                        ]
            lines += [f"float z{k}=0.f;" for k in range(d3)]
            lines += ["int p=base+lane;", f"if(p<end) {{ int u=p/{edges},v=p%{edges};"]
            if multiple:
                lines += [emit_cg_source(meta, spec, pairs=(("a", "b"),))]
                has_optional_pair = any(len(pairs) == 1 for pairs, _ in products)
                lines += [
                    "if(a1!=nullptr) {" if has_optional_pair else "{",
                    emit_cg_source(meta, spec, pairs=(("a1", "b1"),)),
                    "}",
                ]
            else:
                lines += [emit_cg_source(meta, spec, pairs=products[0][0])]
            lines += [
                "}",
                "#pragma unroll 4",
                (
                    "for(int k=0;k<32 && base+k<end;++k) {"
                    if rows == 1
                    else f"for(int k=0;k<32 && base+k<end;k+={rows}) {{"
                ),
            ]
            for slot, buffer in enumerate(buffers):
                if rows > 1:
                    lines.append(f"if(k+{slot}<32 && base+k+{slot}<end) {{")
                lines += ["Mma::FragmentA a; Mma::FragmentB b;", f"b={buffer};"]
                lines.append(f"if(k+{slot + rows}<32 && base+k+{slot + rows}<end) {{")
                weight_channels = spec["out_channels"]
                weight_offset = spec["weight_offset"]
                row = f"(base+k+{slot + rows})"
                if vector:
                    lines += [
                        f"if(column+{columns - 1}<{weight_channels}) {{",
                        f"""const float* src={weight}+int64_t(item)*{meta["weight_numel"]}+{weight_offset}+{row}*{weight_channels}+column;""",
                    ]
                    if meta.get("dtype") is torch.float64 and columns == 4:
                        lines += [
                            f'asm volatile("ld.global.L1::no_allocate.v2.f32 {{%0,%1}},[%2];":"=f"({buffer}[0]),"=f"({buffer}[1]):"l"(src));',
                            f'asm volatile("ld.global.L1::no_allocate.v2.f32 {{%0,%1}},[%2];":"=f"({buffer}[2]),"=f"({buffer}[3]):"l"(src+2));',
                        ]
                    else:
                        values = ",".join(f"%{n}" for n in range(columns))
                        constraints = ",".join(
                            f'"=f"({buffer}[{n}])' for n in range(columns)
                        )
                        lines.append(
                            f'asm volatile("ld.global.L1::no_allocate.v{columns}.f32 {{{values}}},[%{columns}];":{constraints}:"l"(src));'
                        )
                    lines.append("}")
                else:
                    for n in range(columns):
                        lines += [
                            r"""if(column+{channel_offset}<{weight_channels}) {{
const float* src={weight}+int64_t(item)*{weight_numel}+{weight_offset}+{row}*{weight_channels}+column+{channel_offset};
asm volatile("ld.global.L1::no_allocate.f32 %0,[%1];":"=f"({buffer}[{n}]):"l"(src));
}}""".format(
                                channel_offset=n * 32,
                                weight_channels=weight_channels,
                                weight=weight,
                                weight_numel=meta["weight_numel"],
                                weight_offset=weight_offset,
                                row=row,
                                buffer=buffer,
                                n=n,
                            )
                        ]
                shuffle = "k" if rows == 1 else f"k+{slot}"
                lines += [
                    "}",
                    *[
                        f"a[{r}]=__shfl_sync(0xffffffff,z{r},{shuffle});"
                        for r in range(d3)
                    ],
                    "Mma()(accum,a,b,accum);",
                ]
                if rows > 1:
                    lines.append("}")
            lines += ["}", "}"]
            if multiple:
                lines.append("}")
            lines.append("}")
        lines.append(
            r"""#pragma unroll
for(int r=0;r<{d3};++r) {{
int component=r;
#pragma unroll
for(int n=0;n<{columns};++n) {{
{column_declaration}
if(item<items && c<{channels}) {{
float* dst=out+int64_t(item)*{out_dim}+{offset}+c*{d3}+component;
*dst=accum[r*{columns}+n];
}}
}}
}}
return;
}}""".format(
                d3=d3,
                columns=columns,
                column_declaration="int c=column+n;"
                if vector
                else "int c=column+n*32;",
                channels=channels,
                out_dim=meta["out_dim"],
                offset=offset,
            )
        )
        cases.append("\n".join(lines))
    pointers = ",".join(f"const float* __restrict__ {name}" for name in inputs)
    kernel = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
__global__ __launch_bounds__(128,4)
void {kernel_name}({pointers},float* __restrict__ out,int64_t items) {{
    int lane=threadIdx.x&31;
    int group=blockIdx.x%{groups_count}; Index task=blockIdx.x/{groups_count};
{cases}
}}
""".format(
        kernel_name=kernel_name,
        pointers=pointers,
        groups_count=len(groups),
        cases=chr(10).join(cases),
    )
    grid = f"{len(groups) * max_tiles}*((items+3)/4)"
    return kernel, grid, zero_output, inputs


def make_uvw_forward_external_mma_kernel(meta, kernel_name, tile_channels, *, arch):
    """Keep the sparse CG and external FP64 tensor product in one warp."""
    mma_header, mma = get_fp64_mma(arch)
    groups = {}
    for spec in meta["generated_specs"]:
        groups.setdefault(
            (spec["out_offset"], spec["out_channels"], spec["d3"]), []
        ).append(spec)
    width = min(tile_channels, 64)
    max_tiles = max((channels + width - 1) // width for _, channels, _ in groups)
    zero_output = sum(c * d for _, c, d in groups) != meta["out_dim"]
    cases = []
    for group, ((offset, channels, d3), specs) in enumerate(groups.items()):
        rows = (d3 + 15) // 16
        tiles = (min(width, channels) + 7) // 8
        lines = [
            r"""if(group=={group}) {{
Index item=(task/{max_tiles})*4+(threadIdx.x>>5);
int tile_column=(task%{max_tiles})*{width};
if(item>=items || tile_column>={channels}) return;
double accum[{rows}][{tiles}][4]={{{{0.}}}};""".format(
                group=group,
                max_tiles=max_tiles,
                width=width,
                channels=channels,
                rows=rows,
                tiles=tiles,
            )
        ]
        for spec in specs:
            size, edges = spec["path_channels"], spec["edge_channels"]
            lines += [
                "{",
                "#pragma unroll 1",
                f"for(int base=0;base<{size};base+=32) {{",
                *[f"double z{k}=0.;" for k in range(d3)],
                "int p=base+lane;",
                f"if(p<{size}) {{ int u=p/{edges},v=p%{edges};",
                emit_cg_source(meta, spec),
                "}",
                "#pragma unroll 1",
                f"for(int k=0;k<32 && base+k<{size};k+=16) {{",
            ]
            for row in range(rows):
                lines.append(f"double a{row}[8]={{0.}};")
                for component in range(row * 16, min((row + 1) * 16, d3)):
                    half, lane_group = divmod(component - row * 16, 8)
                    for step in range(4):
                        lines.append(
                            f"double s{component}_{step}=__shfl_sync(0xffffffff,z{component},k+(lane&3)+{step * 4});"
                        )
                        lines.append(
                            f"if((lane>>2)=={lane_group}) a{row}[{step * 2 + half}]=s{component}_{step};"
                        )
            lines += [
                r"""#pragma unroll
for(int n=0;n<{tiles};++n) {{
int column=tile_column+n*8+(lane>>2);
double b[4]={{0.}};
#pragma unroll
for(int i=0;i<4;++i) {{
if(base+k+(lane&3)+i*4<{size} && column<{channels}) {{
const double* source=w+int64_t(item)*{weight_numel}+{weight_offset}+(base+k+(lane&3)+i*4)*{channels}+column;
asm volatile("ld.global.L1::no_allocate.f64 %0,[%1];":"=d"(b[i]):"l"(source));
}}
}}""".format(
                    tiles=tiles,
                    size=size,
                    channels=channels,
                    weight_numel=meta["weight_numel"],
                    weight_offset=spec["weight_offset"],
                )
            ]
            for row in range(rows):
                lines.append(r"""Mma::fma(accum[{row}][n][0],accum[{row}][n][1],accum[{row}][n][2],accum[{row}][n][3],
a{row}[0],a{row}[1],a{row}[2],a{row}[3],a{row}[4],a{row}[5],a{row}[6],a{row}[7],
b[0],b[1],b[2],b[3],
accum[{row}][n][0],accum[{row}][n][1],accum[{row}][n][2],accum[{row}][n][3]);""".format(row=row))
            lines += ["}\n}\n}\n}"]
        lines += [
            r"""#pragma unroll
for(int m=0;m<{rows};++m) {{
#pragma unroll
for(int n=0;n<{tiles};++n) {{
#pragma unroll
for(int i=0;i<4;++i) {{
int component=m*16+(lane>>2)+(i>>1)*8;
int column=tile_column+n*8+(lane&3)*2+(i&1);
if(component<{d3} && column<{channels})
out[int64_t(item)*{out_dim}+{offset}+column*{d3}+component]=accum[m][n][i];
}}
}}
}}
return;
}}""".format(
                rows=rows,
                tiles=tiles,
                d3=d3,
                channels=channels,
                out_dim=meta["out_dim"],
                offset=offset,
            )
        ]
        cases.append("\n".join(lines))
    kernel = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
{mma_header}
__global__ __launch_bounds__(128,2)
void {kernel_name}(const double* __restrict__ x,const double* __restrict__ y,
const double* __restrict__ w,double* __restrict__ out,int64_t items) {{
int lane=threadIdx.x&31;
using Mma={mma};
int group=blockIdx.x%{groups_count};
Index task=blockIdx.x/{groups_count};
{cases}
}}
""".format(
        mma_header=mma_header,
        kernel_name=kernel_name,
        mma=mma,
        groups_count=len(groups),
        cases=chr(10).join(cases),
    )
    return (
        kernel,
        f"{len(groups) * max_tiles}*((items+3)/4)",
        zero_output,
        ("x", "y", "w"),
    )


def make_weight_loads(
    tile_k, stage, *, fp64, vector_weights, vector_width, b_pointer_cases
):
    index = "(index ^ (((index / TileN) & 3) << 2))" if fp64 else "index"
    copy = (
        f"cutlass::arch::cp_async_zfill<16>(shared_b[{stage}] + {index}, source, valid);"
        if vector_weights
        else f"shared_b[{stage}][{index}] = valid ? *source : 0.0f;"
    )
    if fp64 and vector_weights:
        copy = f"if(weight_aligned) {{ {copy} }} else {{ shared_b[{stage}][{index}]=valid?source[0]:0.0; shared_b[{stage}][{index}+1]=valid?source[1]:0.0; }}"
    return r"""            for (int vector_index = threadIdx.x;
                 vector_index < TileK * TileN / {weight_alignment};
                 vector_index += blockDim.x) {{
                const int index = vector_index * {weight_alignment};
                const int local_k = index / TileN;
                const int logical_k = {tile_k} + local_k;
                const int column = tile_column + index - local_k * TileN;
                const bool valid = logical_k < group_k && column < out_channels;
                const float* source = weight;
                if (valid) {{
                    switch (group) {{
{b_pointer_cases}
                    default:
                        break;
                    }}
                }}
                {copy}
            }}
            cutlass::arch::cp_async_fence();
""".format(
        weight_alignment=vector_width if vector_weights else 1,
        tile_k=tile_k,
        b_pointer_cases=chr(10).join(b_pointer_cases),
        copy=copy,
    )


def make_uvw_forward_shared_kernel(
    meta,
    kernel_name,
    products,
    wide_channels=None,
    adaptive_splits=False,
    prefetch_weights=False,
    reuse_cg=False,
    *,
    arch,
    include_mma=True,
):
    """Implicit CG/W GEMM, optionally summing two CG products per weight."""
    fp64 = meta.get("dtype") is torch.float64
    used = {v for pairs, weight in products for pair in pairs for v in pair}
    used.update(weight for _, weight in products)
    pointer_order = tuple(
        v for v in ("x", "y", "w", "rx", "ry", "rw", "bx", "by", "bw") if v in used
    )
    parameters = ",".join(f"const float* __restrict__ input_{v}" for v in pointer_order)
    parameters += (
        ",float* __restrict__ out,int64_t num_items,int out_stride,Index total_tasks"
    )
    aliases = {
        "x": [pairs[0][0] for pairs, _ in products],
        "y": [pairs[0][1] for pairs, _ in products],
        "weight": [weight for _, weight in products],
        "x1": [pairs[-1][0] for pairs, _ in products],
        "y1": [pairs[-1][1] for pairs, _ in products],
    }
    # Expanding two products removes operand-selection branches; larger
    # programs stay looped to keep their instruction footprint manageable.
    lines = [
        f"#pragma unroll {(2 if len(products) == 2 else 1)}",
        f"for(int product=0;product<{len(products)};++product) {{",
    ]
    for role, values in aliases.items():
        expression = "".join(
            f"product=={i}?input_{v}:" for i, v in enumerate(values[:-1])
        )
        expression += f"input_{values[-1]}"
        lines.append(f"const float* {role}={expression};")
    paired = [i for i, (pairs, _) in enumerate(products) if len(pairs) == 2]
    expression = (
        "true"
        if len(paired) == len(products)
        else " || ".join(f"product=={i}" for i in paired) or "false"
    )
    lines.append(f"const bool pair_active={expression};")
    if fp64:
        lines.append(
            "const bool weight_aligned=(reinterpret_cast<uintptr_t>(weight)&15)==0;"
        )
    product_open = "\n".join(lines)
    product_close = ("__syncthreads();\n" if len(products) > 1 else "") + "}"
    specs = meta["generated_specs"]
    # Keep the 128-thread MMA tile useful for narrow output groups: exchange
    # padded columns for rows, preserving the same per-thread accumulator.
    wide = (
        all(s["out_channels"] >= 128 for s in specs)
        if wide_channels is None
        else wide_channels
    )
    # Wide output groups retain the compiler's CG register reuse; narrow
    # FP64 forward groups benefit from more resident blocks.
    limit_registers = fp64 and prefetch_weights and not wide
    if fp64:
        wide = False
        reuse_cg = True
        threads = 128
        mma_rows, mma_columns = 4, 4
        tile_columns, tile_rows = 64, 32
        a_store = "index ^ (((index / TileK) & 3) << 2)"
        a_k = "k"
    else:
        threads = 256 if wide and len(products) == 1 else 128
        mma_rows = 8 if wide else 2
        mma_columns = 8 if wide and threads == 128 else 4
        tile_columns = (
            128 if wide else (32 if all(s["out_channels"] <= 32 for s in specs) else 64)
        )
        tile_rows = threads * mma_rows * mma_columns // tile_columns
        a_store = "index ^ ((index & 256) >> 4)" if wide and threads == 128 else "index"
        a_k = "(k ^ ((local_row & 8) << 1))" if wide and threads == 128 else "k"
    # Wide forward assigns 32 columns to a 32-thread warp, so A is broadcast
    # without bank conflicts. The 128-thread higher-order derivative layout assigns two
    # distinct row groups per warp and needs the XOR bank mapping.
    column_scale = 1 if wide else mma_columns
    column_offset = f"n * {tile_columns // mma_columns}" if wide else "n"
    fragment_loads = "\n".join(
        f"""                fragment_a[{i}] = shared_a[stage][{("local_row" if i == 0 else f"(local_row + {i})")} * TileK + {a_k}];"""
        for i in range(mma_rows)
    )
    tile_reduction = 16 if fp64 and len(products) > 1 else 32
    in1_dim = meta["in1_dim"]
    in2_dim = meta["in2_dim"]
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for spec in specs:
        key = (spec["out_offset"], spec["out_channels"], spec["d3"])
        groups.setdefault(key, []).append(spec)
    for group_specs in groups.values():
        group_specs.sort(
            key=lambda spec: (
                -spec["path_channels"] * len(spec["terms"]),
                spec["weight_offset"],
            )
        )
    group_items = sorted(
        groups.items(),
        key=lambda item: (
            -sum(
                spec["path_channels"] * (len(spec["terms"]) + item[0][1] * item[0][2])
                for spec in item[1]
            ),
            item[0][0],
        ),
    )

    zero_output = sum(c * d for _, c, d in groups) != meta["out_dim"] or any(
        sum(s["path_channels"] for s in specs) > 256 for specs in groups.values()
    )
    vector_width = 2 if fp64 else 4
    vector_weights = all(
        s["out_channels"] % vector_width == 0 and s["weight_offset"] % vector_width == 0
        for s in specs
    )
    a_cases = []
    vector_a_cases = []
    b_pointer_cases = []
    task_mapping = []
    total_tasks = []
    row_decode_cases = []
    for group_idx, ((out_offset, out_channels, d3), group_specs) in enumerate(
        group_items
    ):
        group_k = sum(spec["path_channels"] for spec in group_specs)
        split_k = (
            1
            if group_k <= 256
            else 2
            if group_k <= 512
            else 4
            if group_k <= 1024
            else 8
        )
        a_lines = [
            f"case {group_idx}: {{",
            f"    const Index n = row / {d3};",
            f"    const int component = row - n * {d3};",
        ]
        vector_lines = [
            f"case {group_idx}: {{",
            f"const Index item=tile_row/{d3}+index/TileK;",
            *(f"float z{k}=0.f;" for k in range(d3)),
            "if(item<num_items && logical_k<group_k) {",
        ]
        b_pointer_lines = [f"case {group_idx}: {{"]
        path_base = 0
        for spec_pos, spec in enumerate(group_specs):
            path_channels = spec["path_channels"]
            edge_channels = spec["edge_channels"]
            d1 = spec["d1"]
            d2 = spec["d2"]
            x_offset = spec["x_offset"]
            y_offset = spec["y_offset"]
            weight_offset = spec["weight_offset"]
            predicate = (
                f"logical_k < {path_base + path_channels}"
                if spec_pos == 0
                else f"logical_k >= {path_base} && logical_k < {path_base + path_channels}"
            )
            if wide or reuse_cg:
                vector_lines += [
                    f"if({predicate}) {{",
                    f"int p=logical_k-{path_base},u=p/{edge_channels},v=p%{edge_channels};",
                    "{",
                    emit_cg_source(
                        meta, spec, coefficient_first=wide or limit_registers
                    ),
                    "}",
                    "if(pair_active) {",
                    emit_cg_source(
                        meta,
                        spec,
                        pairs=(("x1", "y1"),),
                        coefficient_first=wide or limit_registers,
                    ),
                    "}",
                    "}",
                ]
            a_lines.append(f"    if ({predicate}) {{")
            a_lines.append(f"        const int p = logical_k - {path_base};")
            if edge_channels == 1:
                a_lines.append("        const int u = p;")
                a_lines.append("        const int v = 0;")
            else:
                a_lines.append(f"        const int u = p / {edge_channels};")
                a_lines.append(f"        const int v = p - u * {edge_channels};")
            a_lines.append("        switch (component) {")
            for k in range(d3):
                a_lines.append(f"        case {k}: {{")
                a_lines.append("            float acc;")
                a_lines.append(
                    '            asm volatile("mov.b32 %0, 0;" : "=f"(acc));'
                )
                for i_idx, j_idx, k_idx, coeff in spec["terms"]:
                    if k_idx != k:
                        continue
                    a_lines.append(
                        r"""            acc = fmaf({coefficient}, __ldg(x + int64_t(n) * {in1_dim} + {x_offset} + u * {d1} + {i}) * __ldg(y + int64_t(n) * {in2_dim} + {y_offset} + v * {d2} + {j}), acc);
            if (pair_active) acc = fmaf({coefficient}, __ldg(x1 + int64_t(n) * {in1_dim} + {x_offset} + u * {d1} + {i}) * __ldg(y1 + int64_t(n) * {in2_dim} + {y_offset} + v * {d2} + {j}), acc);""".format(
                            coefficient=float_literal(coeff, meta.get("dtype")),
                            in1_dim=in1_dim,
                            in2_dim=in2_dim,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            d1=d1,
                            d2=d2,
                            i=i_idx,
                            j=j_idx,
                        )
                    )
                a_lines.append("            value = acc;")
                a_lines.append("            break;")
                a_lines.append("        }")
            a_lines.append("        default:")
            a_lines.append("            break;")
            a_lines.append("        }")
            a_lines.append("    }")
            b_pointer_lines.append(f"    if ({predicate}) {{")
            b_pointer_lines.append(f"        const int p = logical_k - {path_base};")
            b_pointer_lines.append(
                f"        source = weight + {weight_offset} + int64_t(p) * {out_channels} + column;"
            )
            b_pointer_lines.append("    }")
            path_base += path_channels
        a_lines.append("    break;")
        a_lines.append("}")
        b_pointer_lines.append("    break;")
        b_pointer_lines.append("}")
        vector_lines.append("}")
        for k in range(d3):
            vector_lines += [
                r"""{{
Index row=item*{d3}+{k}-tile_row;
if(row>=0 && row<TileM) {{
int address=row*TileK+local_k;
shared_a[stage][{feature_store}]=z{k};
}}
}}""".format(
                    d3=d3,
                    k=k,
                    feature_store=a_store.replace("index", "address"),
                )
            ]
        vector_lines += ["break;", "}"]
        vector_a_cases.append("\n".join(vector_lines))
        a_cases.append("\n".join(a_lines))
        b_pointer_cases.append("\n".join(b_pointer_lines))
        out_tiles = (out_channels + tile_columns - 1) // tile_columns
        row_tiles = f"((x.size(0) * int64_t({d3}) + {tile_rows - 1}) / {tile_rows}) * {out_tiles}"
        if wide:
            split_k = min(split_k, 2)
        cuda_splits = host_splits = str(split_k)
        k_range = r"""const int tiles_per_split = (k_tiles + {split_k} - 1) / {split_k};
                k_begin = split_id * tiles_per_split * TileK;
                k_end = k_begin + tiles_per_split * TileK;""".format(split_k=split_k)
        if len(products) > 1 or (adaptive_splits and wide and group_k > 256):
            # Small output grids need more independent reductions to hide
            # latency. Balance whole K tiles across splits, avoiding empty CTAs.
            target_tiles = 1024 if len(products) > 1 else 512
            k_tiles = max(1, (group_k + tile_reduction - 1) // tile_reduction)
            cuda_splits = f"min(Index({k_tiles}), max(Index({split_k}), ({target_tiles} + output_tiles - 1) / max(Index(1), output_tiles)))"
            host_splits = f"std::min<int64_t>({k_tiles}, std::max<int64_t>({split_k}, ({target_tiles} + ({row_tiles}) - 1) / std::max<int64_t>(1, ({row_tiles}))))"
            k_range = r"""k_begin = (k_tiles * split_id / selected_splits) * TileK;
                k_end = (k_tiles * (split_id + 1) / selected_splits) * TileK;"""
        task_mapping.append(
            r"""        {{
            const Index output_tiles = Index((num_items * int64_t({d3}) + TileM - 1) / TileM) * {out_tiles};
            const int selected_splits = {cuda_splits};
            const Index count = output_tiles * selected_splits;
            if ({owns_task}) {{
                const Index local = task - cursor;
                const Index output_tile = local / selected_splits;
                group = {group_idx};
                d3 = {d3};
                group_k = {group_k};
                split_count = selected_splits;
                split_id = local - output_tile * selected_splits;
                out_offset = {out_offset};
                out_channels = {out_channels};
                tile_row = (output_tile / {out_tiles}) * TileM;
                tile_column = (output_tile - (output_tile / {out_tiles}) * {out_tiles}) * TileN;
                const int k_tiles = ({group_k} + TileK - 1) / TileK;
                {k_range}
                if (k_end > {group_k}) k_end = {group_k};
                found = true;
            }}
            cursor += count;
        }}""".format(
                d3=d3,
                out_tiles=out_tiles,
                cuda_splits=cuda_splits,
                owns_task="true"
                if len(group_items) == 1
                else "!found && task < cursor + count",
                group_idx=group_idx,
                group_k=group_k,
                out_offset=out_offset,
                out_channels=out_channels,
                k_range=k_range,
            )
        )
        total_tasks.append(f"({row_tiles}) * ({host_splits})")
        row_decode_cases.append(
            f"case {group_idx}: item = logical_row / {d3}; component = logical_row - item * {d3}; break;"
        )

    # A producer owns one item/path and computes every CG component once,
    # reusing the same input vectors before scattering to the shared GEMM tile.
    a_loads = r"""            for (int index = threadIdx.x; index < TileM * TileK; index += blockDim.x) {{
                const int local_row = index / TileK;
                const Index row = tile_row + local_row;
                const int logical_k = tile_k + index - local_row * TileK;
                float value;
                asm volatile("mov.b32 %0, 0;" : "=f"(value));
                if (row < num_items * int64_t(d3) && logical_k < group_k) {{
                    switch (group) {{
{a_cases}
                    default:
                        break;
                    }}
                }}
                shared_a[stage][{a_store}] = value;
            }}
""".format(
        a_cases=chr(10).join(a_cases), a_store=a_store
    )
    if wide or reuse_cg:
        a_loads = r"""
            int row_items=(tile_row%d3+TileM+d3-1)/d3;
            for(int index=threadIdx.x;index<row_items*TileK;index+=blockDim.x) {{
                int local_k=index%TileK,logical_k=tile_k+local_k;
                switch(group) {{
{vector_a_cases}
                }}
            }}
""".format(
            vector_a_cases=chr(10).join(vector_a_cases)
        )

    first_weights = ""
    next_weights = ""
    weight_loads = make_weight_loads(
        "tile_k",
        "stage",
        fp64=fp64,
        vector_weights=vector_weights,
        vector_width=vector_width,
        b_pointer_cases=b_pointer_cases,
    )
    # The current barrier protects the other buffer before its next write.
    # Narrow tiles retain their original copy/compute ordering.
    if prefetch_weights and (wide or fp64) and vector_weights:
        first_weights = make_weight_loads(
            "k_begin",
            "(k_begin / TileK) & 1",
            fp64=fp64,
            vector_weights=vector_weights,
            vector_width=vector_width,
            b_pointer_cases=b_pointer_cases,
        )
        next_weights = (
            "            if (tile_k + TileK < k_end) {\n"
            + make_weight_loads(
                "(tile_k + TileK)",
                "stage ^ 1",
                fp64=fp64,
                vector_weights=vector_weights,
                vector_width=vector_width,
                b_pointer_cases=b_pointer_cases,
            )
            + "            }\n"
        )
        weight_loads = ""

    if fp64:
        mma_header, mma = get_fp64_mma(arch)
        mma_definition = f"    using Mma = {mma};\n"
        mma_init = "    double accumulators[4][4] = {};\n"
        mma_compute = r"""            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int local_row = (warp >> 1) * 16 + (lane >> 2);
            const int local_column = (warp & 1) * 32;
            #pragma unroll
            for (int k = 0; k < TileK; k += 16) {
                double a[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    a[i] = shared_a[stage][(local_row + (i & 1) * 8) * TileK + ((k + (lane & 3) + (i >> 1) * 4) ^ ((local_row & 3) << 2))];
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    double b[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        b[i] = shared_b[stage][(k + (lane & 3) + i * 4) * TileN + ((local_column + n * 8 + (lane >> 2)) ^ ((lane & 3) << 2))];
                    Mma::fma(accumulators[n][0], accumulators[n][1], accumulators[n][2], accumulators[n][3],
                        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                        b[0], b[1], b[2], b[3],
                        accumulators[n][0], accumulators[n][1], accumulators[n][2], accumulators[n][3]);
                }
            }
"""
        mma_stores = (r"""    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const Index logical_row = tile_row + (warp >> 1) * 16 + (lane >> 2) + (i >> 1) * 8;
            const int column = tile_column + (warp & 1) * 32 + n * 8 + (lane & 3) * 2 + (i & 1);
            if (logical_row < num_items * int64_t(d3) && column < out_channels) {
                Index item;
                int component;
                switch (group) {
""" + "\n".join(row_decode_cases) + r"""
                default: return;
                }
                double* output = out + int64_t(item) * out_stride + out_offset + column * d3 + component;
                if (!Accumulate && split_count == 1) *output = accumulators[n][i];
                else atomicAdd(output, accumulators[n][i]);
            }
        }
    }""")
    else:
        mma_definition = r"""    using ThreadMma = cutlass::gemm::thread::Mma<
        cutlass::gemm::GemmShape<{mma_rows}, {mma_columns}, 1>,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>;
""".format(
            mma_rows=mma_rows, mma_columns=mma_columns
        )
        mma_init = r"""    typename ThreadMma::FragmentC accumulators;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < {fragment_size}; ++index) asm volatile("mov.b32 %0, 0;" : "=f"(accumulators[index]));
""".format(fragment_size=mma_rows * mma_columns)
        mma_compute = r"""            const int local_row = (int(threadIdx.x) / {row_threads}) * {mma_rows};
            const int local_column = (int(threadIdx.x) & {column_mask}) * {column_scale};
            typename ThreadMma::FragmentA fragment_a;
            typename ThreadMma::FragmentB fragment_b;
            {unroll}
            for (int k = 0; k < TileK; ++k) {{
{fragment_loads}
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < {mma_columns}; ++n) {{
                    fragment_b[n] = shared_b[stage][k * TileN + local_column + {column_offset}];
                }}
                ThreadMma()(accumulators, fragment_a, fragment_b, accumulators);
            }}
""".format(
            row_threads=tile_columns // mma_columns,
            mma_rows=mma_rows,
            column_mask=tile_columns // mma_columns - 1,
            column_scale=column_scale,
            unroll="#pragma unroll 4" if wide else "CUTLASS_PRAGMA_UNROLL",
            fragment_loads=fragment_loads,
            mma_columns=mma_columns,
            column_offset=column_offset,
        )
        mma_stores = r"""    const Index row = tile_row + (int(threadIdx.x) / {row_threads}) * {mma_rows};
    const int column = tile_column + (int(threadIdx.x) & {column_mask}) * {column_scale};
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < {mma_rows}; ++m) {{
        if (row + m < num_items * int64_t(d3)) {{
            const Index logical_row = row + m;
            Index item;
            int component;
            switch (group) {{
{row_decode_cases}
            default:
                return;
            }}
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < {mma_columns}; ++n) {{
                if (column + {column_offset} < out_channels) {{
                    if (!Accumulate && split_count == 1) {{
                        out[int64_t(item) * out_stride + out_offset +
                            (column + {column_offset}) * d3 + component] = accumulators[m * {mma_columns} + n];
                    }} else {{
                        atomicAdd(
                            out + int64_t(item) * out_stride + out_offset +
                                (column + {column_offset}) * d3 + component,
                            accumulators[m * {mma_columns} + n]);
                    }}
                }}
            }}
        }}
    }}""".format(
            row_threads=tile_columns // mma_columns,
            mma_rows=mma_rows,
            column_mask=tile_columns // mma_columns - 1,
            column_scale=column_scale,
            row_decode_cases=chr(10).join(row_decode_cases),
            mma_columns=mma_columns,
            column_offset=column_offset,
        )

    mma_include = ("\n" + (mma_header if include_mma else "")) if fp64 else ""
    kernel = r"""
#include <algorithm>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/gemm/thread/mma.h>{mma_include}
#include <cutlass/layout/matrix.h>

namespace {{


template<bool Accumulate>
__global__ __launch_bounds__({threads}, {resident_blocks})
void {kernel_name}(
    {parameters}) {{
    constexpr int TileM = {tile_rows};
    constexpr int TileN = {tile_columns};
    constexpr int TileK = {tile_reduction};
    __shared__ float shared_a[2][TileM * TileK];
    __shared__ float shared_b[2][TileK * TileN];
{mma_definition}    const Index task = Index(blockIdx.x);
    if (task >= total_tasks) return;

    int group = 0;
    int d3 = 0;
    int group_k = 0;
    int split_count = 1;
    int split_id = 0;
    int out_offset = 0;
    int out_channels = 0;
    Index tile_row = 0;
    int tile_column = 0;
    Index k_begin = 0;
    Index k_end = 0;
    Index cursor = 0;
    bool found = false;
{task_mapping}

{mma_init}    {product_open}
{first_weights}    for (Index tile_k = k_begin; tile_k < k_end; tile_k += TileK) {{
            const int stage = (tile_k / TileK) & 1;
{weight_loads}{a_loads}            cutlass::arch::cp_async_wait<0>();
            __syncthreads();
{next_weights}
{mma_compute}        }}

    {product_close}
{mma_stores}
}}

}}  // namespace

""".format(
        mma_include=mma_include,
        threads=threads,
        resident_blocks=4
        if limit_registers
        else 2
        if fp64
        else 3
        if threads == 256
        else 6
        if len(products) > 1 and (not wide)
        else 2,
        kernel_name=kernel_name,
        parameters=parameters,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        tile_reduction=tile_reduction,
        mma_definition=mma_definition,
        task_mapping=chr(10).join(task_mapping),
        mma_init=mma_init,
        product_open=product_open,
        first_weights=first_weights,
        weight_loads=weight_loads,
        a_loads=a_loads,
        next_weights=next_weights,
        mma_compute=mma_compute,
        product_close=product_close,
        mma_stores=mma_stores,
    )
    kernel = cuda_source_for_dtype(kernel, meta.get("dtype", torch.float32))
    return kernel, " + ".join(total_tasks) or "0", zero_output, pointer_order, threads


def make_uvw_forward_external_source(meta, tile_channels=128, *, arch):
    kernel, grid, zero_output, _ = make_uvw_forward_external_kernel(
        meta, tile_channels=tile_channels, pipeline_weights=True, arch=arch
    )
    clear = (
        (
            "C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(),0,"
            "out.numel()*sizeof(float),at::cuda::getCurrentCUDAStream()));"
        )
        if zero_output
        else ""
    )
    source = kernel + r"""
at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=x.size(0);
    auto out=at::empty({{items,int64_t({out_dim})}},x.options());
    if(items==0) return out;
    // Resolve the registered function before CUDA 12.8's first launch.
    static thread_local int prepared_device=-1;
    if(prepared_device != x.get_device()) {{
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,lunatp_uvw_fwd_external_fused_fp32));
        prepared_device=x.get_device();
    }}
    {clear}
    lunatp_uvw_fwd_external_fused_fp32<<<{grid},128,0,at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),y.data_ptr<float>(),w.data_ptr<float>(),out.data_ptr<float>(),items);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{ m.def("launch",&launch); }}
""".format(
        out_dim=meta["out_dim"], clear=clear, grid=grid
    )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def make_uvw_forward_scalar_source(meta):
    """Contract scalar channels with GEMM, then reduce angular input channels."""
    specs = meta["generated_specs"]
    regions = sorted({(s["out_offset"], s["out_offset"] + s["out_channels"] * s["d3"]) for s in specs})
    full_output = (regions[0][0] == 0 and regions[-1][1] == meta["out_dim"]
                   and all(left[1] == right[0] for left, right in zip(regions, regions[1:])))
    initialized_outputs = set()
    calls = []
    angular = []
    angular_width = 0
    packed_total = 0
    main_width = 0
    serial_partial_width = 0
    parallel_conditions = []
    split_capacities = []
    partial_widths = []
    max_tile_columns = 1
    for index, spec in enumerate(specs):
        u, v, o = (spec[k] for k in ("in_channels", "edge_channels", "out_channels"))
        d = spec["d1"]
        coefficient = float_literal(spec["terms"][0][3], torch.float32)
        xoff, yoff, woff, ooff = (spec[k] for k in ("x_offset", "y_offset", "weight_offset", "out_offset"))
        if d == 1:
            vector = o % 4 == 0 and (u * v) % 4 == 0 and woff % 4 == 0
            main_width = max(main_width, u * v)
            serial_partial_width = max(serial_partial_width, o * UVW_FWD_MAX_SPLITS)
            capacity = f"scalar_capacity_{index}"
            split_capacities.append(f"""const int64_t scalar_tiles_{index}=((items+planning_tile-1)/planning_tile)*(({o}+planning_tile-1)/planning_tile);
    const int {capacity}=std::min({max(1,u*v//64)},std::max(1,int((int64_t(sm_count)*{UVW_GEMM_TARGET_BLOCKS_PER_SM}+scalar_tiles_{index}-1)/scalar_tiles_{index})));""")
            partial_widths.append(f"int64_t({o})*{capacity}")
            parallel_condition = f"({capacity}>{UVW_FWD_MAX_SPLITS} && ((items+127)/128)*{(o+127)//128}*{UVW_FWD_MAX_SPLITS}<sm_count*{UVW_GEMM_TARGET_BLOCKS_PER_SM})"
            parallel_conditions.append(parallel_condition)
            accumulate = not full_output or ooff in initialized_outputs
            initialized_outputs.add(ooff)
            max_tile_columns = max(max_tile_columns, (o + 127) // 128)
            vector_choice = "(reinterpret_cast<uintptr_t>(wp+%d)&15)==0" % woff if vector else "false"
            calls.append(f"""
    for(int64_t start=0;start<items;start+=chunk) {{
        const int rows=std::min<int64_t>(chunk,items-start);
        const bool parallel_path={parallel_condition};
        const int tile=parallel_path && rows<=64 ? 64 : 128;
        const int tiles=((rows+tile-1)/tile)*(({o}+tile-1)/tile);
        const int capacity=parallel_path ? {capacity} : {UVW_FWD_MAX_SPLITS};
        const int requested=uvw_grouped::gemm_splits(tiles,{u*v},capacity,sm_count*(tile==64 ? 4 : {UVW_GEMM_BLOCKS_PER_SM}),sm_count*{UVW_GEMM_TARGET_BLOCKS_PER_SM});
        const int slice_k=(({u*v}+requested*8-1)/(requested*8))*8;
        const int splits=({u*v}+slice_k-1)/slice_k;
        const int64_t count=int64_t(rows)*{u*v};
        scalar_outer<{u},{v}><<<(count+1023)/1024,256,0,stream>>>(
            xp+start*{meta['in1_dim']}+{xoff},yp+start*{meta['in2_dim']}+{yoff},tmp,count,flags,parallel_path ? 0 : tiles);
        if(parallel_path) {{
            if(tile==64) uvw_grouped::scalar_gemm_parallel<uvw_grouped::SmallGemmNN>(
                tmp,wp+{woff},partial,rows,{o},{u*v},splits,slice_k,stream);
            else if({vector_choice}) uvw_grouped::scalar_gemm_parallel<uvw_grouped::GemmNN>(
                tmp,wp+{woff},partial,rows,{o},{u*v},splits,slice_k,stream);
            else uvw_grouped::scalar_gemm_parallel<uvw_grouped::GemmNNBase>(
                tmp,wp+{woff},partial,rows,{o},{u*v},splits,slice_k,stream);
            if(tile==64 && splits>=32)
                scalar_split_sum<{o},{meta["out_dim"]},{str(accumulate).lower()},4><<<(int64_t(rows)*{o}*4+255)/256,256,0,stream>>>(
                    partial,op+start*{meta['out_dim']}+{ooff},int64_t(rows)*{o},splits,{coefficient});
            else
                scalar_split_sum<{o},{meta["out_dim"]},{str(accumulate).lower()}><<<(int64_t(rows)*{o}+255)/256,256,0,stream>>>(
                    partial,op+start*{meta['out_dim']}+{ooff},int64_t(rows)*{o},splits,{coefficient});
        }} else {{
            if({vector_choice}) scalar_gemm_serial<uvw_grouped::GemmNN>(
                tmp,wp+{woff},op+start*{meta['out_dim']}+{ooff},rows,{o},{u*v},splits,flags,{coefficient},{1 if accumulate else 0}.f,stream);
            else scalar_gemm_serial<uvw_grouped::GemmNNBase>(
                tmp,wp+{woff},op+start*{meta['out_dim']}+{ooff},rows,{o},{u*v},splits,flags,{coefficient},{1 if accumulate else 0}.f,stream);
        }}
    }}""")
        else:
            angular.append((u, v, o, d, xoff, yoff, woff, ooff, coefficient, packed_total, angular_width))
            packed_total += (u * v * o + 3) // 4 * 4
            angular_width += u * o
    planning_width = max(main_width + serial_partial_width, angular_width)
    workspace = max(main_width, angular_width)
    packed_size = packed_total
    angular_vector = all(v % 4 == 0 and (u * o) % 4 == 0 and yoff % 4 == 0 and meta["in2_dim"] % 4 == 0
                         for u, v, o, d, xoff, yoff, woff, ooff, coefficient, poff, aoff in angular)
    angular_gemm = "uvw_grouped::GemmNN" if angular_vector else "GroupedNN"
    angular_call = f"uvw_grouped::launch<{angular_gemm}>(table,{len(angular)},sm_count*{UVW_GEMM_BLOCKS_PER_SM},1.f,0.f,stream)"
    if angular_vector:
        small_call = f"uvw_grouped::launch<uvw_grouped::SmallGemmNN>(table,{len(angular)},sm_count*{UVW_GEMM_BLOCKS_PER_SM},1.f,0.f,stream)"
        angular_call = f"((reinterpret_cast<uintptr_t>(yp)&15)==0 ? (rows<=64 ? {small_call} : {angular_call}) : uvw_grouped::launch<uvw_grouped::GemmNNBase>(table,{len(angular)},sm_count*{UVW_GEMM_BLOCKS_PER_SM},1.f,0.f,stream))"
    pack_source = []
    contract_source = []
    output_groups = {}
    for group, (u, v, o, d, xoff, yoff, woff, ooff, coefficient, poff, aoff) in enumerate(angular):
        output_groups.setdefault((ooff, o, d), []).append((u, xoff, coefficient, aoff))
        if angular_vector:
            descriptor = f"table.set({group},rows,{u*o},{v},y+{yoff},packed+{poff},tmp+chunk*{aoff},tmp+chunk*{aoff},{meta['in2_dim']},{u*o},{u*o},{u*o});"
        else:
            descriptor = f"table.set({group},{u*o},rows,{v},packed+{poff},y+{yoff},tmp+chunk*{aoff},tmp+chunk*{aoff},{u*o},{meta['in2_dim']},{u*o},{u*o});"
        pack_source.append(f"""
    if(idx=={group}) {{ {descriptor} }}
    if(idx>={poff} && idx<{poff+u*v*o}) {{
        const int64_t i=idx-{poff};
        const int o=i%{o},u=(i/{o})%{u},v=i/{u*o};
        packed[idx]=w[{woff}+(int64_t(u)*{v}+v)*{o}+o];
    }}""")
    output_width = sum(o for _, o, _ in output_groups)
    offset = 0
    for (ooff, o, d), paths in output_groups.items():
        initialization = f"float value[{d}]={{}};" if full_output else f"""float value[{d}];
        #pragma unroll
        for(int m=0;m<{d};++m) value[m]=out[row*OUT_DIM+{ooff}+o*{d}+m];"""
        lines = [f"""
    if(channel>={offset} && channel<{offset+o}) {{
        const int o=channel-{offset};
        {initialization}
        """]
        for u, xoff, coefficient, aoff in paths:
            lines.append(f"""
        {{
            float part[{d}]={{}};
            #pragma unroll 1
            for(int u=0;u<{u};++u) {{
                const float weight=tmp[chunk*{aoff}+(row*{u}+u)*{o}+o];
                #pragma unroll
                for(int m=0;m<{d};++m) part[m]=fmaf(x[row*IN1_DIM+{xoff}+u*{d}+m],weight,part[m]);
            }}
            #pragma unroll
            for(int m=0;m<{d};++m) value[m]+=part[m]*{coefficient};
        }}""")
        lines.append(f"""
        #pragma unroll
        for(int m=0;m<{d};++m) out[row*OUT_DIM+{ooff}+o*{d}+m]=value[m];
    }}""")
        contract_source.extend(lines)
        offset += o
    if angular:
        calls.append(f"""
    for(int64_t start=0;start<items;start+=chunk) {{
        const int rows=std::min<int64_t>(chunk,items-start);
        scalar_pack_groups<<<({packed_size}+255)/256,256,0,stream>>>(
            wp,yp+start*{meta['in2_dim']},packed,tmp,table,rows,chunk);
        const auto status={angular_call};
        TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS grouped GEMM failed");
        scalar_contract_groups<<<(int64_t(rows)*{output_width}+255)/256,256,0,stream>>>(
            xp+start*{meta['in1_dim']},tmp,op+start*{meta['out_dim']},rows,chunk);
    }}""")
    chunk_rows = max(1, (UVW_SCALAR_WORKSPACE_ELEMENTS // planning_width // 128) * 128)
    source = SCALAR_CUDA_SOURCE + SCALAR_PARALLEL_CUDA_SOURCE + r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
using GroupedNN=uvw_grouped::Kernel<cutlass::layout::ColumnMajor,cutlass::layout::ColumnMajor,cutlass::layout::ColumnMajor,128,128,8,cutlass::layout::NoPermute,64,64>;

template<class Grouped>
void scalar_gemm_serial(const float* a,const float* b,float* c,int m,int n,int k,
                        int splits,int* flags,float scale,float beta,cudaStream_t stream) {
    using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Kernel=cutlass::gemm::kernel::Gemm<typename Grouped::Mma,typename Grouped::Epilogue,Swizzle,true>;
    Swizzle swizzle;
    auto tiled=swizzle.get_tiled_shape({m,n,k},{128,128,8},splits);
    typename Kernel::Params params({m,n,k},tiled,
        {const_cast<float*>(a),k},{const_cast<float*>(b),n},
        {c,OUT_DIM},{c,OUT_DIM},{scale,beta},flags);
    void* arguments[]={&params};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(cutlass::Kernel<Kernel>),swizzle.get_grid_shape(tiled),dim3(Kernel::kThreadCount),arguments,sizeof(typename Kernel::SharedStorage),stream));
}

template<int U,int V>
__global__ void scalar_outer(const float* x,const float* y,float* output,int64_t count,
                            int* flags,int tiles) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tiles;i+=int64_t(blockDim.x)*gridDim.x) flags[i]=0;
    const int64_t idx=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;
    if(idx>=count) return;
    float4 value;
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int64_t element=idx+j,row=element/(U*V);
        const int u=(element/V)%U,v=element%V;
        reinterpret_cast<float*>(&value)[j]=element<count ? x[row*IN1_DIM+u]*y[row*IN2_DIM+v] : 0.f;
    }
    if(idx+3<count) reinterpret_cast<float4*>(output)[idx/4]=value;
    else {
        #pragma unroll
        for(int j=0;j<4;++j) if(idx+j<count) output[idx+j]=reinterpret_cast<float*>(&value)[j];
    }
}

__global__ void scalar_pack_groups(const float* w,const float* y,float* packed,float* tmp,
                                   uvw_grouped::Table table,int rows,int64_t chunk) {
    const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
PACK_SOURCE
}

__global__ void scalar_contract_groups(const float* x,const float* tmp,float* out,int rows,int64_t chunk) {
    const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx>=int64_t(rows)*OUTPUT_WIDTH) return;
    const int64_t row=idx/OUTPUT_WIDTH;
    const int channel=idx%OUTPUT_WIDTH;
CONTRACT_SOURCE
}

at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w) {
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=x.size(0);
    auto out=at::OUTPUT_ALLOCATION({items,int64_t(OUT_DIM)},x.options());
    if(items==0) return out;
    const int sm_count=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int planning_tile=items<=64 ? 64 : 128;
    SPLIT_CAPACITIES
    const bool parallel_splits=PARALLEL_CONDITION;
    const int64_t parallel_width=std::max(int64_t(ANGULAR_WIDTH),int64_t(MAIN_WIDTH)+std::max({PARTIAL_WIDTHS}));
    const int64_t parallel_chunk=std::max(int64_t(1),(int64_t(WORKSPACE_ELEMENTS)/parallel_width/128)*128);
    const int64_t chunk=std::min<int64_t>(items,parallel_splits ? parallel_chunk : CHUNK_ROWS);
    auto workspace=at::empty({chunk*(parallel_splits ? parallel_width : WORKSPACE)},x.options());
    const int flag_count=int((chunk+127)/128)*MAX_TILE_COLUMNS;
    const int64_t table_floats=(uvw_grouped::Table::bytes(TABLE_CAPACITY)+sizeof(float)-1)/sizeof(float);
    auto table_buffer=at::empty({int64_t(table_floats+flag_count)},x.options());
    float* tmp=workspace.data_ptr<float>();
    float* partial=tmp+chunk*MAIN_WIDTH;
    // Scalar products are dead before angular packing begins on this stream.
    at::Tensor weight_buffer;
    float* packed=tmp+chunk*ANGULAR_WIDTH;
    if(workspace.numel()-chunk*ANGULAR_WIDTH<PACKED_SIZE) {
        weight_buffer=at::empty({int64_t(PACKED_SIZE)},x.options());
        packed=weight_buffer.data_ptr<float>();
    }
    int* flags=reinterpret_cast<int*>(table_buffer.data_ptr<float>()+table_floats);
    uvw_grouped::Table table(table_buffer.data_ptr(),TABLE_CAPACITY);
    const float* xp=x.data_ptr<float>();
    const float* yp=y.data_ptr<float>();
    const float* wp=w.data_ptr<float>();
    float* op=out.data_ptr<float>();
    auto stream=at::cuda::getCurrentCUDAStream();
CALLS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&launch);}
""".replace("IN1_DIM",str(meta["in1_dim"])).replace("IN2_DIM",str(meta["in2_dim"])).replace("OUT_DIM",str(meta["out_dim"])).replace("CHUNK_ROWS",str(chunk_rows)).replace("WORKSPACE_ELEMENTS",str(UVW_SCALAR_WORKSPACE_ELEMENTS)).replace("WORKSPACE",str(workspace)).replace("SPLIT_CAPACITIES","\n    ".join(split_capacities)).replace("PARTIAL_WIDTHS",",".join(partial_widths) or "int64_t(0)").replace("PARALLEL_CONDITION"," || ".join(parallel_conditions) or "false").replace("OUTPUT_ALLOCATION","empty" if full_output else "zeros").replace("PACKED_SIZE",str(packed_size)).replace("CALLS","\n".join(calls)).replace("TABLE_CAPACITY",str(max(1,len(angular)))).replace("MAX_TILE_COLUMNS",str(max_tile_columns)).replace("ANGULAR_WIDTH",str(angular_width)).replace("MAIN_WIDTH",str(main_width))
    return source.replace("PACK_SOURCE", "\n".join(pack_source)).replace("CONTRACT_SOURCE", "\n".join(contract_source)).replace("OUTPUT_WIDTH", str(max(1,output_width))).replace("IN1_DIM", str(meta["in1_dim"])).replace("OUT_DIM", str(meta["out_dim"]))


def make_uvw_forward_shared_source(meta, adaptive_splits=False, *, arch):
    if (meta.get("dtype", torch.float32) is torch.float32 and meta["generated_specs"]
            and all(s["d2"] == 1 and s["d1"] == s["d3"]
                    for s in meta["generated_specs"])):
        return make_uvw_forward_scalar_source(meta)
    name = "lunatp_uvw_fwd_shared_fused_fp32"
    code, grid, clear, pointers, threads = make_uvw_forward_shared_kernel(
        meta,
        name,
        products=(((("x", "y"),), "w"),),
        adaptive_splits=adaptive_splits,
        prefetch_weights=True,
        reuse_cg=True,
        arch=arch,
    )
    arguments = ",".join(f"{value}.data_ptr<float>()" for value in pointers)
    prepare = r"""// Resolve the registered function before CUDA 12.8's first launch.
    static thread_local int prepared_device=-1;
    if(prepared_device != x.get_device()) {{
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,{name}<false>));
        prepared_device=x.get_device();
    }}
    """.format(name=name)
    reset = (
        "C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(),0,"
        "out.numel()*sizeof(float),stream));"
        if clear
        else ""
    )
    source = code + r"""
at::Tensor launch(at::Tensor x, at::Tensor y, at::Tensor w) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto out=at::empty({{x.size(0),int64_t({out_dim})}},x.options());
    const int64_t tasks={grid};
    if(tasks==0) return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    {reset}
    {prepare}{name}<false><<<tasks,{threads},0,stream>>>(
        {arguments},out.data_ptr<float>(),x.size(0),out.size(1),tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""".format(
        out_dim=meta["out_dim"],
        grid=grid,
        reset=reset,
        prepare=prepare,
        name=name,
        threads=threads,
        arguments=arguments,
    )

    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, x, edge_attr, weight):
    # Wide shared weights use more K partitions on large batches.
    large_batch = op.wide_forward and x.size(0) >= UVW_FWD_LARGE_BATCH_THRESHOLD
    device = x.get_device()
    if device != op.arch_device:
        op.arch = get_cuda_arch(x.device)
        op.arch_device = device
    key = op.arch, 0, (), large_batch
    launch = op.kernels.get(key)
    if launch is None:
        dtype_tag = "fp64" if op.compute_dtype == torch.float64 else "fp32"
        source = (
            make_uvw_forward_shared_source(
                op.meta, adaptive_splits=large_batch, arch=op.arch
            )
            if op.weight_mode == "shared"
            else make_uvw_forward_external_source(
                op.meta,
                arch=op.arch,
                tile_channels=64 if op.wide_forward and not large_batch else 128,
            )
        )
        extension = load_cuda_extension(
            f"lunatp_uvw_fwd_{op.weight_mode}_fused_{dtype_tag}",
            source,
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )
        launch = op.kernels[key] = extension.launch
    return launch(x, edge_attr, weight)
