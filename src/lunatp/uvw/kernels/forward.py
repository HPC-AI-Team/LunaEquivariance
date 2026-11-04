"""UVW forward CUDA templates."""

from textwrap import dedent, indent

import torch

from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal, get_fp64_mma
from ....utils.extension import (
    get_cuda_arch,
    get_device_properties,
    load_cuda_extension,
)
from .utils import (
    SCALAR_WEIGHT_LOADER,
    UVW_FWD_ACCUMULATOR_LIMIT,
    UVW_FWD_LARGE_BATCH_THRESHOLD,
    UVW_FWD_MAX_SPLITS,
    UVW_GEMM_TARGET_BLOCKS_PER_SM,
    UVW_GEMM_THREADS,
    UVW_GEMM_TILE_ELEMENTS,
    UVW_SCALAR_TARGET_BLOCKS,
    UVW_TARGET_BLOCKS,
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
            rf"""if (group == {group}) {{
    Index item=(task/{max_tiles})*4+(threadIdx.x>>5);
    int tile_column=(task%{max_tiles})*{width};
    if (item>=items || tile_column>={channels}) return;
    {f'int column=tile_column+lane*{columns};' if vector else 'int column=tile_column+lane;'}
    using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{d3},{columns},1>,
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC accum; accum.clear();"""
        ]
        path_padding = "            " if multiple else "        "
        for spec in specs:
            size, edges = (
                spec["path_channels"],
                spec["edge_channels"],
            )
            lines.append("    {")
            if multiple:
                # Keep one matrix loop instead of duplicating it for each derivative term.
                lines += [
                    "        #pragma unroll 1",
                    f"        for(int product=0;product<{len(products)};++product) {{",
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
                    lines.append(f"            const float* {name}={expression};")
            weight = "weight" if multiple else products[0][1]
            weight_channels = spec["out_channels"]
            weight_offset = spec["weight_offset"]
            lines += [
                f"{path_padding}constexpr Index begin=0,end={size};",
                f"{path_padding}auto load_weight = [&](Mma::FragmentB& fragment,int row) {{",
            ]
            if vector:
                lines += [
                    f"{path_padding}    if(column+{columns - 1}<{weight_channels}) {{",
                    f"{path_padding}        const float* src={weight}+int64_t(item)*{meta['weight_numel']}+{weight_offset}+int64_t(row)*{weight_channels}+column;",
                ]
                if meta.get("dtype") is torch.float64 and columns == 4:
                    lines += [
                        path_padding + '        asm volatile("ld.global.L1::no_allocate.v2.f32 {%0,%1},[%2];":"=f"(fragment[0]),"=f"(fragment[1]):"l"(src));',
                        path_padding + '        asm volatile("ld.global.L1::no_allocate.v2.f32 {%0,%1},[%2];":"=f"(fragment[2]),"=f"(fragment[3]):"l"(src+2));',
                    ]
                else:
                    values = ",".join(f"%{n}" for n in range(columns))
                    constraints = ",".join(f'"=f"(fragment[{n}])' for n in range(columns))
                    lines.append(f'{path_padding}        asm volatile("ld.global.L1::no_allocate.v{columns}.f32 {{{values}}},[%{columns}];":{constraints}:"l"(src));')
                lines.append(path_padding + "    }")
            else:
                for n in range(columns):
                    lines.append(indent(rf"""if(column+{n * 32}<{weight_channels}) {{
    const float* src={weight}+int64_t(item)*{meta['weight_numel']}+{weight_offset}+int64_t(row)*{weight_channels}+column+{n * 32};
    asm volatile("ld.global.L1::no_allocate.f32 %0,[%1];":"=f"(fragment[{n}]):"l"(src));
}}""", path_padding + "    "))
            lines += [path_padding + "};", path_padding + "#pragma unroll 1", path_padding + "for (int base=begin;base<end;base+=32) {"]
            buffers = ("next",) if rows == 1 else ("next0", "next1")
            for slot, buffer in enumerate(buffers):
                lines.append(f"{path_padding}    Mma::FragmentB {buffer};\n{path_padding}    {buffer}.clear();")
                guard = f"if(base+{slot}<end) " if rows > 1 else ""
                lines.append(f"{path_padding}    {guard}load_weight({buffer},base+{slot});")
            lines += [f"{path_padding}    float z{k}=0.f;" for k in range(d3)]
            lines += [f"{path_padding}    int p=base+lane;", f"{path_padding}    if(p<end) {{\n{path_padding}        int u=p/{edges},v=p%{edges};"]
            if multiple:
                lines += [indent(emit_cg_source(meta, spec, pairs=(("a", "b"),)), path_padding + "        ")]
                has_optional_pair = any(len(pairs) == 1 for pairs, _ in products)
                lines += [
                    path_padding + ("        if(a1!=nullptr) {" if has_optional_pair else "        {"),
                    indent(emit_cg_source(meta, spec, pairs=(("a1", "b1"),)), path_padding + "            "),
                    path_padding + "        }",
                ]
            else:
                lines += [indent(emit_cg_source(meta, spec, pairs=products[0][0]), path_padding + "        ")]
            lines += [
                path_padding + "    }",
                path_padding + "    #pragma unroll 4",
                (
                    path_padding + "    for(int k=0;k<32 && base+k<end;++k) {"
                    if rows == 1
                    else f"{path_padding}    for(int k=0;k<32 && base+k<end;k+={rows}) {{"
                ),
            ]
            mma_padding = path_padding + ("            " if rows > 1 else "        ")
            for slot, buffer in enumerate(buffers):
                if rows > 1:
                    lines.append(f"{path_padding}        if(k+{slot}<32 && base+k+{slot}<end) {{")
                lines += [f"{mma_padding}Mma::FragmentA a;\n{mma_padding}Mma::FragmentB b;", f"{mma_padding}b={buffer};"]
                lines.append(f"{mma_padding}if(k+{slot + rows}<32 && base+k+{slot + rows}<end) {{")
                lines.append(f"{mma_padding}    load_weight({buffer},base+k+{slot + rows});")
                shuffle = "k" if rows == 1 else f"k+{slot}"
                lines += [
                    mma_padding + "}",
                    *[
                        f"{mma_padding}a[{r}]=__shfl_sync(0xffffffff,z{r},{shuffle});"
                        for r in range(d3)
                    ],
                    mma_padding + "Mma()(accum,a,b,accum);",
                ]
                if rows > 1:
                    lines.append(path_padding + "        }")
            lines += [path_padding + "    }", path_padding + "}"]
            if multiple:
                lines.append("        }")
            lines.append("    }")
        lines.append(
            rf"""#pragma unroll
    for(int r=0;r<{d3};++r) {{
        int component=r;
#pragma unroll
        for(int n=0;n<{columns};++n) {{
            {'int c=column+n;' if vector else 'int c=column+n*32;'}
            if(item<items && c<{channels}) {{
                float* dst=out+int64_t(item)*{meta['out_dim']}+{offset}+c*{d3}+component;
                *dst=accum[r*{columns}+n];
            }}
        }}
    }}
    return;
}}"""
        )
        cases.append("\n".join(lines))
    pointers = ",".join(f"const float* __restrict__ {name}" for name in inputs)
    kernel = rf"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
__global__ __launch_bounds__(128, 4) void {kernel_name}({pointers}, float *__restrict__ out, int64_t items) {{
    int lane = threadIdx.x & 31;
    int group = blockIdx.x % {len(groups)};
    Index task = blockIdx.x / {len(groups)};
{indent(chr(10).join(cases), "    ")}
}}
"""
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
            rf"""if(group=={group}) {{
    Index item=(task/{max_tiles})*4+(threadIdx.x>>5);
    int tile_column=(task%{max_tiles})*{width};
    if(item>=items || tile_column>={channels}) return;
    double accum[{rows}][{tiles}][4]={{{{0.}}}};"""
        ]
        for spec in specs:
            size, edges = spec["path_channels"], spec["edge_channels"]
            lines += [
                "    {",
                "        #pragma unroll 1",
                f"        for(int base=0;base<{size};base+=32) {{",
                *[f"            double z{k}=0.;" for k in range(d3)],
                "            int p=base+lane;",
                f"            if(p<{size}) {{\n                int u=p/{edges},v=p%{edges};",
                indent(emit_cg_source(meta, spec), "                "),
                "            }",
                "            #pragma unroll 1",
                f"            for(int k=0;k<32 && base+k<{size};k+=16) {{",
            ]
            for row in range(rows):
                lines.append(f"                double a{row}[8]={{0.}};")
                for component in range(row * 16, min((row + 1) * 16, d3)):
                    half, lane_group = divmod(component - row * 16, 8)
                    for step in range(4):
                        lines.append(
                            f"                double s{component}_{step}=__shfl_sync(0xffffffff,z{component},k+(lane&3)+{step * 4});"
                        )
                        lines.append(
                            f"                if((lane>>2)=={lane_group}) a{row}[{step * 2 + half}]=s{component}_{step};"
                        )
            lines += [
                indent(rf"""#pragma unroll
for(int n=0;n<{tiles};++n) {{
    int column=tile_column+n*8+(lane>>2);
    double b[4]={{0.}};
#pragma unroll
    for(int i=0;i<4;++i) {{
        if(base+k+(lane&3)+i*4<{size} && column<{channels}) {{
            const double* source=w+int64_t(item)*{meta['weight_numel']}+{spec['weight_offset']}+(base+k+(lane&3)+i*4)*{channels}+column;
            asm volatile("ld.global.L1::no_allocate.f64 %0,[%1];":"=d"(b[i]):"l"(source));
        }}
    }}""", "                ")
            ]
            for row in range(rows):
                lines.append(indent(rf"""Mma::fma(accum[{row}][n][0],accum[{row}][n][1],accum[{row}][n][2],accum[{row}][n][3],
    a{row}[0],a{row}[1],a{row}[2],a{row}[3],a{row}[4],a{row}[5],a{row}[6],a{row}[7],
    b[0],b[1],b[2],b[3],
    accum[{row}][n][0],accum[{row}][n][1],accum[{row}][n][2],accum[{row}][n][3]);""", "                    "))
            lines += ["                }\n            }\n        }\n    }"]
        lines += [
            rf"""#pragma unroll
    for(int m=0;m<{rows};++m) {{
#pragma unroll
        for(int n=0;n<{tiles};++n) {{
#pragma unroll
            for(int i=0;i<4;++i) {{
                int component=m*16+(lane>>2)+(i>>1)*8;
                int column=tile_column+n*8+(lane&3)*2+(i&1);
                if(component<{d3} && column<{channels})
                    out[int64_t(item)*{meta['out_dim']}+{offset}+column*{d3}+component]=accum[m][n][i];
            }}
        }}
    }}
    return;
}}"""
        ]
        cases.append("\n".join(lines))
    kernel = rf"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
{mma_header}
__global__ __launch_bounds__(128, 2) void {kernel_name}(const double *__restrict__ x, const double *__restrict__ y,
                                                          const double *__restrict__ w, double *__restrict__ out, int64_t items) {{
    int lane = threadIdx.x & 31;
    using Mma = {mma};
    int group = blockIdx.x % {len(groups)};
    Index task = blockIdx.x / {len(groups)};
{indent(chr(10).join(cases), "    ")}
}}
"""
    return (
        kernel,
        f"{len(groups) * max_tiles}*((items+3)/4)",
        zero_output,
        ("x", "y", "w"),
    )


def make_weight_loads(tile_k, stage, *, vector_weights, vector_width, b_pointer_cases):
    """Emit weight loads for the shared FP64 kernel."""
    index = "(index ^ (((index / TileN) & 3) << 2))"
    copy = (
        f"cutlass::arch::cp_async_zfill<16>(shared_b[{stage}] + {index}, source, valid);"
        if vector_weights
        else f"shared_b[{stage}][{index}] = valid ? *source : 0.0f;"
    )
    if vector_weights:
        copy = f"""if(weight_aligned) {{
    {copy}
}} else {{
    shared_b[{stage}][{index}]=valid?source[0]:0.0;
    shared_b[{stage}][{index}+1]=valid?source[1]:0.0;
}}"""
    return rf"""for (int vector_index = threadIdx.x;
    vector_index < TileK * TileN / {vector_width if vector_weights else 1};
    vector_index += blockDim.x) {{
    const int index = vector_index * {vector_width if vector_weights else 1};
    const int local_k = index / TileN;
    const int logical_k = {tile_k} + local_k;
    const int column = tile_column + index - local_k * TileN;
    const bool valid = logical_k < group_k && column < out_channels;
    const float* source = weight;
    if (valid) {{
        switch (group) {{
{indent(chr(10).join(b_pointer_cases), "            ")}
            default:
                break;
        }}
    }}
{indent(copy, "    ")}
}}
cutlass::arch::cp_async_fence();
"""


def make_uvw_forward_scalar_fused_kernel(meta, kernel_name, products):
    groups = {}
    for spec in meta["generated_specs"]:
        groups.setdefault(
            (spec["out_offset"], spec["out_channels"], spec["d3"]), []
        ).append(spec)
    used = {r for pairs, w in products for pair in pairs for r in pair}
    used.update(w for pairs, w in products)
    pointers = tuple(
        r for r in ("x", "y", "w", "rx", "ry", "rw", "bx", "by", "bw") if r in used
    )
    group_rows = {key: 32 if key[2] <= 3 else 16 for key in groups}
    output_tiles = "+".join(
        f"((num_items+{group_rows[key] - 1})/{group_rows[key]})*{(key[1] + 31) // 32}"
        for key in groups
    )
    max_splits = min(
        UVW_FWD_MAX_SPLITS,
        max(
            (s["path_channels"] // s["edge_channels"] + 7) // 8
            for s in meta["generated_specs"]
        ),
    )
    splits = f"min(int64_t({max_splits}),max(int64_t(1),(int64_t({UVW_SCALAR_TARGET_BLOCKS})+({output_tiles})-1)/({output_tiles})))"
    dispatch = []
    totals = []
    for (offset, channels, d), specs in groups.items():
        rows = group_rows[(offset, channels, d)]
        ntiles = (channels + 31) // 32
        tasks = f"(((num_items+{rows - 1})/{rows})*{ntiles})*({splits})"
        totals.append(tasks.replace("num_items", "x.size(0)"))
        calls = []
        for spec in specs:
            u, v = spec["path_channels"] // spec["edge_channels"], spec["edge_channels"]
            coefficient = float_literal(spec["terms"][0][3], torch.float32)
            by_input = {}
            for pairs, weight in products:
                for x, y in pairs:
                    by_input.setdefault(x, []).append((y, weight))
            for x, terms in by_input.items():
                count = len(terms)
                y_pointers = ",".join(
                    f"{y}+tile_row*{meta['in2_dim']}+{spec['y_offset']}"
                    for y, _ in terms
                )
                w_pointers = ",".join(
                    f"{weight}+{spec['weight_offset']}" for _, weight in terms
                )
                if count == 1:
                    iterator_a = "typename BaseMma::IteratorA"
                    prepare_a = f"""
            typename BaseMma::IteratorA::Params params_a(cutlass::layout::RowMajor({meta["in2_dim"]}));
            typename BaseMma::IteratorA a(params_a,const_cast<float*>({y_pointers}),
            {{int(min(Index({rows}),num_items-tile_row)),{v}}},threadIdx.x,{{0,0}});
"""
                else:
                    iterator_a = f"InputIterator<{rows},{v},{meta['in2_dim']},{count}>"
                    prepare_a = f"""
            const float* inputs[{count}]={{{y_pointers}}};
            {iterator_a} a(inputs,int(min(Index({rows}),num_items-tile_row)),threadIdx.x);
"""
                iterator_b = f"WeightIterator<{rows},{u},{v},{channels},{str(spec['weight_offset'] % 4 == 0).lower()},{count}>"
                matmul = f"""
                {{
{indent(dedent(prepare_a), "                    ")}
                    using PathMma=cutlass::gemm::threadblock::MmaPipelined<typename BaseMma::Shape,
                        {iterator_a},typename BaseMma::SmemIteratorA,{iterator_b},
                        cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<8,256>,
                        float,cutlass::layout::RowMajor,0,typename {iterator_b}::ThreadMap>,
                        float,cutlass::layout::RowMajor,typename BaseMma::Policy>;
                    const float* weights[{count}]={{{w_pointers}}};
                    {iterator_b} b(weights,u0,tile_column,threadIdx.x);
                    PathMma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
                    mma({((v + 7) // 8) * count},accum,a,b,accum);
                    __syncthreads();
                }}
"""
                consume = f"""
            #pragma unroll
            for(int m=0;m<{rows // 4};++m) {{
                const int row=(m/4)*16+lane.row()*4+m%4;
                #pragma unroll
                for(int z=0;z<2;++z) {{
                    float4* dest=reinterpret_cast<float4*>(storage.h+row*256+(threadIdx.x/32)*64+z*32+lane.column()*4);
                    *dest=make_float4(accum[m*8+z*4],accum[m*8+z*4+1],accum[m*8+z*4+2],accum[m*8+z*4+3]);
                }}
            }}
            __syncthreads();
            #pragma unroll
            for(int j=0;j<{rows // 4};++j) {{
                const int position=threadIdx.x+j*128,row=position/32,column=position%32;
                const Index item=tile_row+row;
                if(item<num_items && tile_column+column<{channels}) {{
                    #pragma unroll
                    for(int angular=0;angular<{d};++angular) {{
                        float value=0.f;
                        #pragma unroll
                        for(int z=0;z<8;++z) {{
                            if(u0+z<{u})value=fmaf({x}[item*{meta["in1_dim"]}+{spec["x_offset"]}+(u0+z)*{d}+angular],storage.h[row*256+z*32+column],value);
                        }}
                        result[j*{d}+angular]+={coefficient}*value;
                    }}
                }}
            }}
"""
                calls.append(f"""
        for(Index u0=split*8;u0<{u};u0+=8*({splits})) {{
            typename BaseMma::FragmentC accum;accum.clear();
{indent(dedent(matmul), "            ")}
{indent(dedent(consume), "            ")}
            __syncthreads();
        }}
""")
        initialize = f"float result[{rows // 4}*{d}]={{}};"
        output = f"""
        #pragma unroll
        for(int j=0;j<{rows // 4};++j) {{
            const int position=threadIdx.x+j*128,row=position/32,column=position%32;
            const Index item=tile_row+row;
            if(item<num_items && tile_column+column<{channels}) {{
                #pragma unroll
                for(int angular=0;angular<{d};++angular)
                    atomicAdd(out+item*out_stride+{offset}+(tile_column+column)*{d}+angular,result[j*{d}+angular]);
            }}
        }}
"""
        dispatch.append(f"""
    if(task<cursor+{tasks}) {{
        using BaseMma=Mma<{rows}>;
        auto& storage=shared.m{rows};
        const int split=(task-cursor)%({splits});
        const Index local=(task-cursor)/({splits}),tile_row=(local/{ntiles})*{rows};
        const int tile_column=(local%{ntiles})*32;
        {initialize}
        auto lane=BaseMma::Operator::Policy::get_lane_layout().inverse(threadIdx.x%32);
{chr(10).join(calls)}
{output}
        return;
    }}
    cursor+={tasks};
""")
    code = f"""
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
namespace {kernel_name}_detail {{
template <int Rows>
using Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<Rows, 256, 8>, cutlass::gemm::GemmShape<Rows, 64, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel;
template <int Rows>
using Mma = typename Default<Rows>::Mma;
template <int Rows>
union Storage {{
    typename Mma<Rows>::SharedStorage main_loop;
    float h[Rows * 256];
}};
template <int Rows, int V, int Stride, int Terms>
struct InputIterator {{
    using BaseMma = Mma<Rows>;
    using Original = typename BaseMma::IteratorA;
    using Element = float;
    using Fragment = typename Original::Fragment;
    using ThreadMap = typename Original::ThreadMap;
    static constexpr int64_t PaddedV = (int64_t(V) + 7) / 8 * 8;
    using KIndex = std::conditional_t<(PaddedV * Terms + 16 <= INT32_MAX), int, int64_t>;
    const float *inputs[Terms];
    KIndex k;
    int rows, row;
    bool enabled = true;
    CUTLASS_DEVICE InputIterator(const float *const (&ptr)[Terms], int count, int thread)
        : rows(count) {{
        #pragma unroll
        for (int i = 0; i < Terms; ++i)
            inputs[i] = ptr[i];
        auto offset = ThreadMap::initial_offset(thread);
        row = offset.strided();
        k = offset.contiguous();
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE InputIterator &operator++() {{
        k += 8;
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &frag) const {{
        const KIndex term = k / PaddedV, first_v = k - term * PaddedV;
        const float *input = inputs[0];
        #pragma unroll
        for (int i = 1; i < Terms; ++i)
            input = term == i ? inputs[i] : input;
        const bool active = enabled && k < PaddedV * Terms;
        #pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            const int item = row + s * ThreadMap::Delta::kStrided;
            #pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                #pragma unroll
                for (int j = 0; j < ThreadMap::kElementsPerAccess; ++j) {{
                    const Index v = Index(first_v + c * ThreadMap::Delta::kContiguous + j);
                    const bool valid = active && item < rows && v < V;
                    const int index = (s * ThreadMap::Iterations::kContiguous + c) * ThreadMap::kElementsPerAccess + j;
                    frag[index] = valid ? input[Index(item) * Stride + v] : 0.f;
                }}
            }}
        }}
    }}
}};
{SCALAR_WEIGHT_LOADER}
template <int Rows, int U, int V, int O, bool Aligned = true, int Terms = 1>
using WeightIterator = ScalarWeightLoader<Mma<Rows>, U, V, O, Aligned, Terms>;
}}
constexpr int {kernel_name}_shared_bytes = 0;
template <bool Accumulate>
__global__ __launch_bounds__(128, 3) void {kernel_name}({",".join("const float* " + r for r in pointers)}, float *out, Index num_items, int out_stride, Index total_tasks) {{
    using namespace {kernel_name}_detail;
    __shared__ union {{
        Storage<16> m16;
        Storage<32> m32;
    }} shared;

    const Index task = blockIdx.x;
    Index cursor = 0;
{chr(10).join(dispatch)}
}}
"""
    clear = True
    return code, "+".join(totals), clear, pointers, 128


def make_uvw_forward_cutlass_kernel(meta, kernel_name, products, *, accumulate=False):
    """Fuse the CG operand into CUTLASS's pipelined GEMM iterator."""
    groups = {}
    for index, spec in enumerate(meta['generated_specs']):
        groups.setdefault((spec['out_offset'],spec['out_channels'],spec['d3']),[]).append((index,spec))
    used = {r for pairs,w in products for pair in pairs for r in pair}
    used.update(w for pairs,w in products)
    pointers=tuple(r for r in ('x','y','w','rx','ry','rw','bx','by','bw') if r in used)
    scalar = all(s["d2"] == 1 and s["d1"] == s["d3"] for s in meta["generated_specs"])
    channel_alignment = min(s["out_channels"] & -s["out_channels"] for s in meta["generated_specs"])
    tile_columns = min(256, max(64, channel_alignment)) if scalar else 64
    tile_rows = min(128, UVW_GEMM_TILE_ELEMENTS // tile_columns) if scalar else 64
    threads = UVW_GEMM_THREADS if scalar else 128
    warp_columns = min(64, tile_columns // 2) if scalar else 32
    alignment_b = min(4, tile_columns * 8 // threads) if scalar else 1
    if any(s["out_channels"] % alignment_b or s["weight_offset"] % alignment_b for s in meta["generated_specs"]):
        alignment_b = 1
    output_tiles = '+'.join(f'(((num_items*int64_t({d3})+{tile_rows-1})/{tile_rows})*{(channels+tile_columns-1)//tile_columns})' for _,channels,d3 in groups)
    split_source = ""
    if scalar:
        multiprocessors = meta["multiprocessors"]
        target_blocks = multiprocessors * UVW_GEMM_TARGET_BLOCKS_PER_SM
        reduction_tiles = max((spec["path_channels"] + 7) // 8 for spec in meta["generated_specs"])
        split_source = f"""
CUTLASS_HOST_DEVICE int64_t {kernel_name}_split_k(int64_t tiles) {{
    if(tiles<=0 || tiles>={target_blocks})return 1;
    const int64_t lower=min(int64_t({UVW_FWD_MAX_SPLITS}),max(int64_t(1),int64_t({target_blocks})/tiles));
    const int64_t upper=min(int64_t({UVW_FWD_MAX_SPLITS}),(int64_t({target_blocks})+tiles-1)/tiles);
    const int64_t lower_work=((tiles*lower+{multiprocessors-1})/{multiprocessors})*(({reduction_tiles}+lower-1)/lower);
    const int64_t upper_work=((tiles*upper+{multiprocessors-1})/{multiprocessors})*(({reduction_tiles}+upper-1)/upper);
    int best=lower_work<upper_work ? lower : upper;
    int64_t best_work=min(lower_work,upper_work);
    #pragma unroll
    for(int split=1;split<={UVW_FWD_MAX_SPLITS};++split) {{
        if(split>upper)continue;
        const int64_t work=((tiles*split+{multiprocessors-1})/{multiprocessors})*(({reduction_tiles}+split-1)/split);
        if(work<best_work) {{best=split;best_work=work;}}
    }}
    return best;
}}
"""
    row_index = "Index" if scalar else "int64_t"
    definitions=[]
    dispatch=[]
    totals=[]
    for (offset, channels, d3), specs in groups.items():
        ntiles=(channels+tile_columns-1)//tile_columns
        tasks=f'((num_items*int64_t({d3})+{tile_rows-1})/{tile_rows})*{ntiles}'
        contributions=len(specs)*len(products)
        requested=f'(1+{UVW_TARGET_BLOCKS-1}/({output_tiles}))'
        splits=f'({requested}<{contributions}?{requested}:{contributions})' if accumulate else '1'
        if accumulate:
            tasks=f'({tasks})*({splits})'
        ksplit = f'{kernel_name}_split_k({output_tiles})' if scalar else '1'
        tasks=f'({tasks})*({ksplit})' if scalar else tasks
        totals.append(tasks.replace('num_items','x.size(0)'))
        calls=[]
        for index,spec in specs:
            if scalar:
                coefficient = float_literal(spec['terms'][0][3], torch.float32)
                cg_method = f'''
    CUTLASS_DEVICE float value(Index row,int k,bool row_valid,bool column_valid) const {{
        if constexpr(!HasSecond){{
            if(!row_valid || !column_valid)return 0.f;
            const Index item=row/{d3};
            const int component=row%{d3},u=k/{spec['edge_channels']},v=k%{spec['edge_channels']};
            const Index xi=item*{meta['in1_dim']}+{spec['x_offset']}+u*{d3}+component;
            const Index yi=item*{meta['in2_dim']}+{spec['y_offset']}+v;
            float z={coefficient}*(x[xi]*y[yi]);
            return z;
        }}
        const bool valid=row_valid && column_valid;
        const Index safe_row=valid ? row : Index(0);
        const int safe_k=column_valid ? k : 0;
        const Index item=safe_row/{d3};
        const int component=safe_row%{d3},u=safe_k/{spec['edge_channels']},v=safe_k%{spec['edge_channels']};
        const Index xi=item*{meta['in1_dim']}+{spec['x_offset']}+u*{d3}+component;
        const Index yi=item*{meta['in2_dim']}+{spec['y_offset']}+v;
        float left=0.f,right=0.f;
        cutlass::arch::global_load<float,4>(left,x+xi,valid);
        cutlass::arch::global_load<float,4>(right,y+yi,valid);
        float z={coefficient}*(left*right);
        if constexpr(HasSecond){{
            float other_left=0.f,other_right=0.f;
            cutlass::arch::global_load<float,4>(other_left,x1+xi,valid);
            cutlass::arch::global_load<float,4>(other_right,y1+yi,valid);
            z+={coefficient}*(other_left*other_right);
        }}
        return z;
    }}
'''
            else:
                scratch_stores = [
                    f'''{{
    int64_t r=item*{d3}+{component}-tile;
    if(r>=0 && r<{tile_rows})scratch[r*8+i%8]=z{component};
}}'''
                    for component in range(d3)
                ]
                cg_method = f'''
    CUTLASS_DEVICE void produce(float* scratch,int64_t tile,int64_t rows,int begin,int end,bool enabled) const {{
        const int count=(tile%{d3}+{tile_rows}+{d3}-1)/{d3};
        for(int i=threadIdx.x;i<count*8;i+=blockDim.x) {{
            const int64_t item=tile/{d3}+i/8;
            const int k=begin+i%8,u=k/{spec['edge_channels']},v=k%{spec['edge_channels']};
{indent(chr(10).join(f'float z{c}=0.f;' for c in range(d3)), '            ')}
            if(enabled && item*{d3}<rows && k<end) {{
                {{
{indent(emit_cg_source(meta,spec,coefficient_first=False), "                    ")}
                }}
                if(x1) {{
{indent(emit_cg_source(meta,spec,pairs=(("x1","y1"),),coefficient_first=False), "                    ")}
                }}
            }}
{indent(chr(10).join(scratch_stores), '            ')}
        }}
    }}
'''
            definitions.append(f'''
{"template<bool HasSecond>" if scalar else ""}
struct {kernel_name}_CG{index} {{
    const float *x, *y, *x1, *y1;
    {cg_method}
}};''')
            if scalar:
                for pairs, weight in products:
                    x, y = pairs[0]
                    x1, y1 = pairs[1] if len(pairs) > 1 else ('nullptr', 'nullptr')
                    select = f'({len(calls)})%({splits})==split' if accumulate else 'true'
                    calls.append(f"""{{
    if ({select}) {{
        const int chunk=({spec['path_channels']}+8*({ksplit})-1)/(8*({ksplit}))*8;
        const int begin_k=split_k*chunk,end_k=min({spec['path_channels']},begin_k+chunk);
        if(begin_k<end_k) {{
            using A={kernel_name}_Iterator<{kernel_name}_CG{index}<{str(len(pairs)>1).lower()}>>;
            using Mma={kernel_name}_Mma<A>;
            A a({{{x},{y},{x1},{y1}}},num_items*int64_t({d3}),end_k,tile_row,threadIdx.x,scratch,begin_k);
            typename {kernel_name}_WeightIterator::Params bparams(cutlass::layout::RowMajor({channels}));
            {kernel_name}_WeightIterator b(bparams,const_cast<float*>({weight}+{spec['weight_offset']}),
            {{end_k,{channels}}},threadIdx.x,{{begin_k,tile_column}});
            Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
            mma((end_k-begin_k+7)/8,accum,a,b,accum);
            __syncthreads();
        }}
    }}
}}""")
                continue
            choices = []
            for pairs, weight in products:
                x, y = pairs[0]
                x1, y1 = pairs[1] if len(pairs)>1 else ('nullptr', 'nullptr')
                choices.append((x, y, x1, y1, weight))
            pointer_choices = []
            for j, name in enumerate(('input_x', 'input_y', 'other_x', 'other_y', 'weight')):
                expression = choices[-1][j]
                for product in range(len(choices)-2, -1, -1):
                    expression = f"product=={product} ? {choices[product][j]} : ({expression})"
                pointer_choices.append(f"const float* {name}={expression};")
            pointer_choices = "\n".join(pointer_choices)
            condition = f"if(({len(calls)}*{len(products)}+product)%({splits})!=split)continue;" if accumulate else ''
            begin_k = f'const int begin_k=0,end_k={spec["path_channels"]};'
            calls.append(f'''{{
    #pragma unroll 1
    for(int product=0;product<{len(products)};++product) {{
        {condition}
{indent(pointer_choices, "        ")}
        {begin_k}
        using A={kernel_name}_Iterator<{kernel_name}_CG{index}>;
        using Mma={kernel_name}_Mma<A>;
        A a({{input_x,input_y,other_x,other_y}},num_items*int64_t({d3}),end_k,tile_row,threadIdx.x,scratch,begin_k);
        typename BaseMma::IteratorB::Params bparams(cutlass::layout::RowMajor({channels}));
        typename BaseMma::IteratorB b(bparams,const_cast<float*>(weight+{spec['weight_offset']}),
        {{end_k,{channels}}},threadIdx.x,{{begin_k,tile_column}});
        Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
        mma((end_k-begin_k+7)/8,accum,a,b,accum);
        __syncthreads();
    }}
}}''')
        tile_decode = f'const int64_t local=task-cursor,tile_row=(local/{ntiles})*{tile_rows};'
        if accumulate:
            tile_decode=f'const int64_t local=(task-cursor)/({splits}),split=(task-cursor)%({splits}),tile_row=(local/{ntiles})*{tile_rows};'
        if scalar:
            local = f'(task-cursor)/({ksplit})'
            if accumulate:
                tile_decode = f'const int64_t local=({local})/({splits}),split=((task-cursor)/({ksplit}))%({splits});'
            else:
                tile_decode = f'const int64_t local={local};'
            tile_decode += f' const Index tile_row=Index((local/{ntiles})*{tile_rows}); const int split_k=(task-cursor)%({ksplit});'
        dispatch.append(f'''if(task<cursor+({tasks})) {{
    {tile_decode}
    const int tile_column=(local%{ntiles})*{tile_columns};
    typename BaseMma::FragmentC accum;accum.clear();
{indent(chr(10).join(calls), "    ")}
    using E={kernel_name}_Epilogue<{d3},{'true' if scalar else 'Accumulate'}>;
    typename E::OutputTileIterator::Params params{{cutlass::layout::RowMajor(out_stride)}};
    typename E::OutputTileIterator output(params,out+{offset},tile_row,{{int(min(int64_t({tile_rows}),num_items*int64_t({d3})-tile_row)),{channels}}},threadIdx.x,{{0,tile_column}});
    E epilogue(storage.epilogue,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    typename E::OutputOp op(1.f,0.f);
    epilogue(op,output,accum,output);
    return;
}}
cursor+=({tasks});''')
    cg_load = r'''        cg.produce(scratch,tile,rows,k-local_k,limit,enabled);
        __syncthreads();
        #pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s) {
            const int64_t r=row+s*ThreadMap::Delta::kStrided;
            #pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c) {
                const int column=k+c*ThreadMap::Delta::kContiguous;
                frag[s*ThreadMap::Iterations::kContiguous+c]=scratch[(r-tile)*8+local_k+c*ThreadMap::Delta::kContiguous];
            }
        }
        __syncthreads();'''
    scalar_load = """
        #pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s) {
            const Index r=row+s*ThreadMap::Delta::kStrided;
            #pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c) {
                const int column=k+c*ThreadMap::Delta::kContiguous;
                frag[s*ThreadMap::Iterations::kContiguous+c] = cg.value(r,column,r<rows,Full || (enabled && column<limit));
            }
        }
    """
    weight_iterators = f'''
using {kernel_name}_WeightMap=cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<{tile_columns},8>,{threads},{alignment_b}>;
using {kernel_name}_WeightIterator=cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::MatrixShape<8,{tile_columns}>,float,cutlass::layout::RowMajor,0,{kernel_name}_WeightMap,{alignment_b}>;
using {kernel_name}_WeightSmemIterator=cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<8,{tile_columns}>,float,cutlass::layout::RowMajor,0,{kernel_name}_WeightMap>;
''' if scalar else f'''
using {kernel_name}_WeightIterator=typename {kernel_name}_Default::Mma::IteratorB;
using {kernel_name}_WeightSmemIterator=typename {kernel_name}_Default::Mma::SmemIteratorB;
'''
    pipelined_mma=f'''
// Full K8 tiles keep masks fixed; the final tile drains the CUTLASS pipeline.
template <class IteratorA>
struct {kernel_name}_Mma : cutlass::gemm::threadblock::MmaPipelined<
                                 typename {kernel_name}_Default::Mma::Shape,
                                 IteratorA, typename {kernel_name}_Default::Mma::SmemIteratorA,
                                 {kernel_name}_WeightIterator,
                                 {kernel_name}_WeightSmemIterator,
                                 float, cutlass::layout::RowMajor,
                                 typename {kernel_name}_Default::Mma::Policy> {{
    using Base = cutlass::gemm::threadblock::MmaPipelined<
        typename {kernel_name}_Default::Mma::Shape,
        IteratorA, typename {kernel_name}_Default::Mma::SmemIteratorA,
        {kernel_name}_WeightIterator,
        {kernel_name}_WeightSmemIterator,
        float, cutlass::layout::RowMajor,
        typename {kernel_name}_Default::Mma::Policy>;
    using FragmentA = typename Base::FragmentA;
    using FragmentB = typename Base::FragmentB;
    using FragmentC = typename Base::FragmentC;
    using WarpA = typename Base::Operator::FragmentA;
    using WarpB = typename Base::Operator::FragmentB;
    using IteratorB = typename Base::IteratorB;
    using Base::Base;
    static_assert(Base::kWarpGemmIterations == 8 && Base::kStages == 2);

    template <bool Fetch>
    CUTLASS_DEVICE void phase(int read_stage, FragmentC &accum, IteratorA &a, IteratorB &b,
                              WarpA (&warp_a)[2], WarpB (&warp_b)[2]) {{
        FragmentA global_a;
        FragmentB global_b;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {{
            if (k == 7) {{
                this->smem_iterator_A_.store(this->transform_A_(global_a));
                this->smem_iterator_B_.store(this->transform_B_(global_b));
                this->gmem_wait();
                ++this->smem_iterator_A_;
                ++this->smem_iterator_B_;
                if (read_stage == 0) {{
                    this->smem_iterator_A_.add_tile_offset({{0, -2}});
                    this->smem_iterator_B_.add_tile_offset({{-2, 0}});
                }} else {{
                    this->warp_tile_iterator_A_.add_tile_offset({{0, -16}});
                    this->warp_tile_iterator_B_.add_tile_offset({{-16, 0}});
                }}
            }}
            this->warp_tile_iterator_A_.set_kgroup_index((k + 1) % 8);
            this->warp_tile_iterator_B_.set_kgroup_index((k + 1) % 8);
            this->warp_tile_iterator_A_.load(warp_a[(k + 1) % 2]);
            this->warp_tile_iterator_B_.load(warp_b[(k + 1) % 2]);
            ++this->warp_tile_iterator_A_;
            ++this->warp_tile_iterator_B_;
            if (k == 0) {{
                global_a.clear();
                global_b.clear();
                if constexpr (Fetch) {{
                    a.template load<true>(global_a);
                    ++a;
                    b.load(global_b);
                    ++b;
                }}
            }}
            this->warp_mma(accum, warp_a[k % 2], warp_b[k % 2], accum);
        }}
    }}

    CUTLASS_DEVICE void operator()(int iterations, FragmentC &accum, IteratorA a, IteratorB b,
                                   FragmentC const &source_accum) {{
        this->prologue(a, b, iterations);
        this->gmem_wait();
        accum = source_accum;
        WarpA warp_a[2];
        WarpB warp_b[2];
        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_A_.load(warp_a[0]);
        ++this->warp_tile_iterator_A_;
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_b[0]);
        ++this->warp_tile_iterator_B_;
        int read_stage = 0;
        #pragma unroll 1
        for (; iterations > 1; --iterations) {{
            phase<true>(read_stage, accum, a, b, warp_a, warp_b);
            read_stage ^= 1;
        }}
        phase<false>(read_stage, accum, a, b, warp_a, warp_b);
    }}
}};
'''
    code=f'''
#include <algorithm>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/arch/memory.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
namespace {{
{split_source}
using {kernel_name}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tile_rows}, {tile_columns}, 8>, cutlass::gemm::GemmShape<32, {warp_columns}, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel;
{weight_iterators}
{''.join(definitions)}
template <class CG>
struct {kernel_name}_Iterator {{
    using Original = typename {kernel_name}_Default::Mma::IteratorA;
    using Element = float;
    using Fragment = typename Original::Fragment;
    using ThreadMap = typename Original::ThreadMap;
    CG cg;
    {row_index} rows, row, tile;
    float *scratch;
    int local_k;
    int columns, k, limit;
    bool enabled = true, first = true;
    CUTLASS_DEVICE {kernel_name}_Iterator(CG cg_, {row_index} rows_, int columns_, {row_index} tile_, int thread, float *scratch_, int begin = 0)
        : cg(cg_), rows(rows_), tile(tile_), scratch(scratch_), columns(columns_) {{
        auto q = ThreadMap::initial_offset(thread);
        row = tile + q.strided();
        local_k = q.contiguous();
        k = begin + local_k;
        limit = begin + (columns % 8 ? columns % 8 : 8);
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {kernel_name}_Iterator &operator++() {{
        k += first ? (columns % 8 ? columns % 8 : 8) : 8;
        first = false;
        limit = columns;
        return *this;
    }}
    template <bool Full = false>
    CUTLASS_DEVICE void load(Fragment &frag) const {{
{scalar_load if scalar else cg_load}
    }}
}};
{pipelined_mma}template<int Components, bool Accumulate> struct {kernel_name}_Output : {kernel_name}_Default::Epilogue::OutputTileIterator {{
    using Base = typename {kernel_name}_Default::Epilogue::OutputTileIterator;
    using ThreadMap = typename Base::ThreadMap;
    using Fragment = typename Base::Fragment;
    using Params = typename Base::Params;
    float *pointer;
    int64_t stride, row_base;
    CUTLASS_DEVICE {kernel_name}_Output(Params const &p, float *ptr, int64_t row_base_, cutlass::MatrixCoord extent, int thread, cutlass::MatrixCoord offset = {{}}, int const *indices = nullptr)
        : Base(p, ptr, extent, thread, offset, indices), pointer(ptr), stride(p.stride / sizeof(float)), row_base(row_base_) {{}}
    CUTLASS_DEVICE void store(Fragment const &frag) {{
        typename Base::Mask mask;
        this->get_mask(mask);
        #pragma unroll
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster)
            #pragma unroll
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group)
                #pragma unroll
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {{
                    const int r = this->thread_start_row() + row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
                    const int f = row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
                    #pragma unroll
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {{
                        const int c = this->thread_start_column() + column * ThreadMap::Delta::kColumn;
                        if (r < this->extent_row() && mask.predicates[column]) {{
                            float *p = pointer + ((row_base + r) / Components) * stride + int64_t(c) * Components + (row_base + r) % Components;
                            if (Accumulate)
                                atomicAdd(p, frag[f * ThreadMap::Iterations::kColumn + column]);
                            else
                                *p = frag[f * ThreadMap::Iterations::kColumn + column];
                        }}
                    }}
                }}
    }}
}};
template <int Components, bool Accumulate>
using {kernel_name}_Epilogue = cutlass::epilogue::threadblock::Epilogue<
    typename {kernel_name}_Default::Epilogue::Shape, typename {kernel_name}_Default::Epilogue::WarpMmaOperator,
    {kernel_name}_Default::Epilogue::kPartitionsK, {kernel_name}_Output<Components, Accumulate>,
    typename {kernel_name}_Default::Epilogue::AccumulatorFragmentIterator, typename {kernel_name}_Default::Epilogue::WarpTileIterator,
    typename {kernel_name}_Default::Epilogue::SharedLoadIterator, typename {kernel_name}_Default::Epilogue::OutputOp,
    typename {kernel_name}_Default::Epilogue::Padding>;
template <bool Accumulate>
__global__ {f"__launch_bounds__({threads},2)" if scalar else ""} void {kernel_name}({','.join('const float* __restrict__ '+r for r in pointers)}, float *__restrict__ out, int64_t num_items, int out_stride, Index total_tasks) {{
    using BaseMma = typename {kernel_name}_Default::Mma;
    union Storage {{
        typename BaseMma::SharedStorage main_loop;
        typename {kernel_name}_Default::Epilogue::SharedStorage epilogue;
    }};
    __shared__ Storage storage;
    __shared__ float scratch[{tile_rows} * 8];
    const int64_t task = blockIdx.x;
    int64_t cursor = 0;
{indent(chr(10).join(dispatch), "    ")}
}}
}}
'''
    clear=scalar or sum(c*d for _,c,d in groups)!=meta['out_dim']
    return code,'+'.join(totals),clear,pointers,threads


def make_uvw_forward_shared_kernel(
    meta,
    kernel_name,
    products,
    prefetch_weights=False,
    accumulate=False,
    *,
    arch,
    include_mma=True,
):
    """Implicit CG/W GEMM, optionally summing two CG products per weight."""
    if meta.get("dtype") is not torch.float64:
        specs = meta["generated_specs"]
        if all(s["d2"] == 1 and s["d1"] == s["d3"] for s in specs) and any(
            s["d3"] > 1 for s in specs
        ):
            return make_uvw_forward_scalar_fused_kernel(meta, kernel_name, products)
        code, grid, clear, pointers, threads = make_uvw_forward_cutlass_kernel(
            meta, kernel_name, products, accumulate=accumulate
        )
        code += f"\nconstexpr int {kernel_name}_shared_bytes=0;\n"
        return code, grid, clear, pointers, threads
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
        lines.append(f"    const float* {role}={expression};")
    paired = [i for i, (pairs, _) in enumerate(products) if len(pairs) == 2]
    expression = (
        "true"
        if len(paired) == len(products)
        else " || ".join(f"product=={i}" for i in paired) or "false"
    )
    lines.append(f"    const bool pair_active={expression};")
    lines.append(
        "    const bool weight_aligned=(reinterpret_cast<uintptr_t>(weight)&15)==0;"
    )
    product_open = "\n".join(lines)
    product_close = ("    __syncthreads();\n" if len(products) > 1 else "") + "}"
    specs = meta["generated_specs"]
    # Preserve the narrower FP64 forward's coefficient order and occupancy.
    limit_registers = prefetch_weights and not all(
        spec["out_channels"] >= 128 for spec in specs
    )
    threads = 128
    tile_columns, tile_rows = 64, 32
    a_store = "index ^ (((index / TileK) & 3) << 2)"
    tile_reduction = 16 if len(products) > 1 else 32
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
    vector_width = 2
    vector_weights = all(
        s["out_channels"] % vector_width == 0 and s["weight_offset"] % vector_width == 0
        for s in specs
    )
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
        vector_lines = [
            f"case {group_idx}: {{",
            f"    const Index item=tile_row/{d3}+index/TileK;",
            *(f"    float z{k}=0.f;" for k in range(d3)),
            "    if(item<num_items && logical_k<group_k) {",
        ]
        b_pointer_lines = [f"case {group_idx}: {{"]
        path_base = 0
        for spec_pos, spec in enumerate(group_specs):
            path_channels = spec["path_channels"]
            edge_channels = spec["edge_channels"]
            weight_offset = spec["weight_offset"]
            predicate = (
                f"logical_k < {path_base + path_channels}"
                if spec_pos == 0
                else f"logical_k >= {path_base} && logical_k < {path_base + path_channels}"
            )
            vector_lines += [
                f"        if({predicate}) {{",
                f"            int p=logical_k-{path_base},u=p/{edge_channels},v=p%{edge_channels};",
                "            {",
                indent(emit_cg_source(meta, spec, coefficient_first=limit_registers), "                "),
                "            }",
                "            if(pair_active) {",
                indent(emit_cg_source(
                    meta,
                    spec,
                    pairs=(("x1", "y1"),),
                    coefficient_first=limit_registers,
                ), "                "),
                "            }",
                "        }",
            ]
            b_pointer_lines.append(f"    if ({predicate}) {{")
            b_pointer_lines.append(f"        const int p = logical_k - {path_base};")
            b_pointer_lines.append(
                f"        source = weight + {weight_offset} + int64_t(p) * {out_channels} + column;"
            )
            b_pointer_lines.append("    }")
            path_base += path_channels
        b_pointer_lines.append("    break;")
        b_pointer_lines.append("}")
        vector_lines.append("    }")
        for k in range(d3):
            vector_lines += [
                indent(rf"""{{
    Index row=item*{d3}+{k}-tile_row;
    if(row>=0 && row<TileM) {{
        int address=row*TileK+local_k;
        shared_a[stage][{a_store.replace('index', 'address')}]=z{k};
    }}
}}""", "    ")
            ]
        vector_lines += ["    break;", "}"]
        vector_a_cases.append("\n".join(vector_lines))
        b_pointer_cases.append("\n".join(b_pointer_lines))
        out_tiles = (out_channels + tile_columns - 1) // tile_columns
        row_tiles = f"((x.size(0) * int64_t({d3}) + {tile_rows - 1}) / {tile_rows}) * {out_tiles}"
        cuda_splits = host_splits = str(split_k)
        k_range = rf"""const int tiles_per_split = (k_tiles + {split_k} - 1) / {split_k};
k_begin = split_id * tiles_per_split * TileK;
k_end = k_begin + tiles_per_split * TileK;"""
        if len(products) > 1:
            # Small output grids need more independent reductions to hide
            # latency. Balance whole K tiles across splits, avoiding empty CTAs.
            target_tiles = UVW_TARGET_BLOCKS
            k_tiles = max(1, (group_k + tile_reduction - 1) // tile_reduction)
            cuda_splits = f"min(Index({k_tiles}), max(Index({split_k}), ({target_tiles} + output_tiles - 1) / max(Index(1), output_tiles)))"
            host_splits = f"std::min<int64_t>({k_tiles}, std::max<int64_t>({split_k}, ({target_tiles} + ({row_tiles}) - 1) / std::max<int64_t>(1, ({row_tiles}))))"
            # A split endpoint fits the path, while its pre-division product
            # can require a wider integer. Keep ordinary generated code intact.
            tile_product = (
                "int64_t(k_tiles)"
                if k_tiles * min(k_tiles, max(split_k, target_tiles)) > 0x7FFFFFFF
                else "k_tiles"
            )
            k_range = f"""k_begin = ({tile_product} * split_id / selected_splits) * TileK;
                k_end = ({tile_product} * (split_id + 1) / selected_splits) * TileK;"""
        task_mapping.append(
            rf"""        {{
            const Index output_tiles = Index((num_items * int64_t({d3}) + TileM - 1) / TileM) * {out_tiles};
            const int selected_splits = {cuda_splits};
            const Index count = output_tiles * selected_splits;
            if ({'true' if len(group_items) == 1 else '!found && task < cursor + count'}) {{
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
{indent(dedent(k_range), "                ")}
                if (k_end > {group_k}) k_end = {group_k};
                found = true;
            }}
            cursor += count;
        }}"""
        )
        total_tasks.append(f"({row_tiles}) * ({host_splits})")
        row_decode_cases.append(
            f"case {group_idx}: item = logical_row / {d3}; component = logical_row - item * {d3}; break;"
        )

    # A producer owns one item/path and computes every CG component once,
    # reusing the same input vectors before scattering to the shared GEMM tile.

    first_weights = ""
    next_weights = ""
    weight_loads = make_weight_loads(
        "tile_k",
        "stage",
        vector_weights=vector_weights,
        vector_width=vector_width,
        b_pointer_cases=b_pointer_cases,
    )
    # The current barrier protects the other buffer before its next write.
    # Narrow tiles retain their original copy/compute ordering.
    if prefetch_weights and vector_weights:
        first_weights = make_weight_loads(
            "k_begin",
            "(k_begin / TileK) & 1",
            vector_weights=vector_weights,
            vector_width=vector_width,
            b_pointer_cases=b_pointer_cases,
        )
        next_weights = (
            "if (tile_k + TileK < k_end) {\n"
            + indent(make_weight_loads(
                "(tile_k + TileK)",
                "stage ^ 1",
                vector_weights=vector_weights,
                vector_width=vector_width,
                b_pointer_cases=b_pointer_cases,
            ), "    ")
            + "}\n"
        )
        weight_loads = ""

    mma_header, mma = get_fp64_mma(arch)

    tile_loop = rf"""
{first_weights}
for (Index tile_k = k_begin; tile_k < k_end; tile_k += TileK) {{
    const int stage = (tile_k / TileK) & 1;
{indent(weight_loads, "    ")}
    int row_items = (tile_row % d3 + TileM + d3 - 1) / d3;
    for (int index = threadIdx.x; index < row_items * TileK; index += blockDim.x) {{
        int local_k = index % TileK, logical_k = tile_k + local_k;
        switch (group) {{
{indent(chr(10).join(vector_a_cases), "            ")}
        }}
    }}
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
{indent(next_weights, "    ")}
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int local_row = (warp >> 1) * 16 + (lane >> 2);
    const int local_column = (warp & 1) * 32;
    #pragma unroll
    for (int k = 0; k < TileK; k += 16) {{
        double a[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            a[i] = shared_a[stage][(local_row + (i & 1) * 8) * TileK + ((k + (lane & 3) + (i >> 1) * 4) ^ ((local_row & 3) << 2))];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {{
            double b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                b[i] = shared_b[stage][(k + (lane & 3) + i * 4) * TileN + ((local_column + n * 8 + (lane >> 2)) ^ ((lane & 3) << 2))];
            Mma::fma(accumulators[n][0], accumulators[n][1], accumulators[n][2], accumulators[n][3],
                     a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                     b[0], b[1], b[2], b[3],
                     accumulators[n][0], accumulators[n][1], accumulators[n][2], accumulators[n][3]);
        }}
    }}
}}
"""

    kernel = rf"""
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
#include <cutlass/gemm/thread/mma.h>{chr(10) + (mma_header if include_mma else '')}
#include <cutlass/layout/matrix.h>

namespace {{

template <bool Accumulate>
__global__ __launch_bounds__({threads}, {4 if limit_registers else 2}) void {kernel_name}(
    {parameters}) {{
    constexpr int TileM = {tile_rows};
    constexpr int TileN = {tile_columns};
    constexpr int TileK = {tile_reduction};
    __shared__ float shared_a[2][TileM * TileK];
    __shared__ float shared_b[2][TileK * TileN];
    using Mma = {mma};
    const Index task = Index(blockIdx.x);
    if (task >= total_tasks)
        return;

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
{indent(dedent(chr(10).join(task_mapping)), "    ")}

    double accumulators[4][4] = {{}};
{indent(product_open, "    ")}
{indent(tile_loop, "        ")}

{indent(product_close, "    ")}
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {{
        #pragma unroll
        for (int i = 0; i < 4; ++i) {{
            const Index logical_row = tile_row + (warp >> 1) * 16 + (lane >> 2) + (i >> 1) * 8;
            const int column = tile_column + (warp & 1) * 32 + n * 8 + (lane & 3) * 2 + (i & 1);
            if (logical_row < num_items * int64_t(d3) && column < out_channels) {{
                Index item;
                int component;
                switch (group) {{
{indent(chr(10).join(row_decode_cases), "                    ")}
                    default:
                        return;
                }}
                double *output = out + int64_t(item) * out_stride + out_offset + column * d3 + component;
                if (!Accumulate && split_count == 1)
                    *output = accumulators[n][i];
                else
                    atomicAdd(output, accumulators[n][i]);
            }}
        }}
    }}
}}

}} // namespace

"""
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
    source = kernel + rf"""
at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=x.size(0);
    auto out=at::empty({{items,int64_t({meta['out_dim']})}},x.options());
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
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def make_uvw_forward_shared_source(meta, *, arch):
    kernels = [("lunatp_uvw_fwd_shared_fused_fp32", meta)]
    if meta.get("dtype") is not torch.float64 and meta.get("scalar_input"):
        scalar_specs = [spec for spec in meta["generated_specs"] if spec["d3"] == 1]
        angular_specs = [spec for spec in meta["generated_specs"] if spec["d3"] > 1]
        if scalar_specs and angular_specs:
            kernels = [
                (
                    "lunatp_uvw_fwd_shared_scalar_fused_fp32",
                    {**meta, "generated_specs": scalar_specs},
                ),
                (
                    "lunatp_uvw_fwd_shared_angular_fused_fp32",
                    {**meta, "generated_specs": angular_specs},
                ),
            ]
    sources, tasks, prepare, launches = [], [], [], []
    clear_output = False
    for index, (name, kernel_meta) in enumerate(kernels):
        code, grid, clear, pointers, threads = make_uvw_forward_shared_kernel(
            kernel_meta,
            name,
            products=(((("x", "y"),), "w"),),
            prefetch_weights=True,
            arch=arch,
        )
        sources.append(code)
        tasks.append(f"const int64_t tasks_{index}={grid};")
        shared_bytes = (
            f"{name}_shared_bytes" if meta.get("dtype") is not torch.float64 else "0"
        )
        arguments = ",".join(f"{value}.data_ptr<float>()" for value in pointers)
        prepare.append(f"""
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,{name}<false>));
        if constexpr({shared_bytes}>0) C10_CUDA_CHECK(cudaFuncSetAttribute({name}<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_bytes}));
""")
        launches.append(f"""
    {name}<false><<<tasks_{index},{threads},{shared_bytes},stream>>>(
        {arguments},out.data_ptr<float>(),x.size(0),out.size(1),tasks_{index});
""")
        clear_output |= clear
    total_tasks = "+".join(f"tasks_{index}" for index in range(len(kernels)))
    reset = (
        "C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(),0,"
        "out.numel()*sizeof(float),stream));"
        if clear_output
        else ""
    )
    source = (
        "\n".join(sources)
        + f"""
at::Tensor launch(at::Tensor x, at::Tensor y, at::Tensor w) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto out=at::empty({{x.size(0),int64_t({meta["out_dim"]})}},x.options());
{indent(chr(10).join(tasks), "    ")}
    if(({total_tasks})==0) return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    {reset}
    static thread_local int prepared_device=-1;
    if(prepared_device != x.get_device()) {{
        cudaFuncAttributes attributes;
{indent(dedent(chr(10).join(prepare)), "        ")}
        prepared_device=x.get_device();
    }}
{indent(chr(10).join(launches), "    ")}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
"""
    )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, x, edge_attr, weight):
    # Only per-item weights change their channel tile with the batch size.
    large_batch = (
        op.weight_mode == "external"
        and op.wide_forward
        and x.size(0) >= UVW_FWD_LARGE_BATCH_THRESHOLD
    )
    device = x.get_device()
    if device != op.arch_device:
        op.arch = get_cuda_arch(x.device)
        op.arch_device = device
    key = op.arch, 0, (), large_batch, device
    launch = op.kernels.get(key)
    if launch is None:
        dtype_tag = "fp64" if op.compute_dtype == torch.float64 else "fp32"
        kernel_meta = {
            **op.meta,
            "multiprocessors": get_device_properties(x.device).multi_processor_count,
        }
        source = (
            make_uvw_forward_shared_source(kernel_meta, arch=op.arch)
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
