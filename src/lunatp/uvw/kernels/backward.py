"""UVW backward CUDA templates."""

from itertools import zip_longest
from operator import itemgetter

from textwrap import indent

import torch

from ...utils.runtime import DERIVATIVE_STREAM_SOURCE
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal, get_fp64_mma
from ....utils.extension import get_cuda_arch, get_device_properties, load_cuda_extension
from ...utils.contractions import INPUTS, PROGRAMS, OUTPUT_SHAPES
from .forward import make_uvw_forward_external_kernel, make_uvw_forward_shared_kernel
from .utils import (
    SCALAR_WEIGHT_LOADER,
    UVW_EXTERNAL_TILE_CHANNELS,
    UVW_THREADS,
    UVW_EXTERNAL_BLOCKS_PER_SM,
    UVW_STATIC_SHARED_BYTES,
    UVW_DY_MAX_TILE_DIM,
    UVW_DY_MIN_TILE_COLUMNS,
    UVW_DY_TILE_K,
    UVW_GEMM_TILE_ELEMENTS,
    UVW_GEMM_THREADS,
    UVW_GEMM_BLOCKS_PER_SM,
    UVW_TARGET_BLOCKS,
    UVW_WEIGHT_MAX_SPLITS,
)


def make_uvw_backward_dxdy_shared_kernel(meta, program, inputs, outputs, name, *, arch):
    fp64 = meta.get("dtype") is torch.float64
    specs = meta["generated_specs"]
    threads = 128
    # An irrep wider than a tile is reduced in component chunks. The CG
    # epilogue adds each chunk directly to its final gradient.
    segmented = any(spec["d3"] > 32 for spec in specs)
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    needed = {
        z for pairs in (*x_outputs.values(), *y_outputs.values()) for _, z in pairs
    }
    contractions = {k: v for k, v in program["z"].items() if k in needed}
    weights = sorted({w for pairs in contractions.values() for w, _ in pairs})
    gradients = sorted({g for pairs in contractions.values() for _, g in pairs})
    slots = {z: i for i, z in enumerate(contractions)}
    if fp64:
        buffers = 1
        resident_blocks = 2
    else:
        buffers = 2 if len(weights) + len(gradients) <= 3 else 1
        resident_blocks = (3 if len(contractions) <= 2 else 2) * 256 // threads
    xdim, ydim, odim = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    tasks, totals, epilogues = [], [], []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        edge, channels, paths = (
            spec[k] for k in ("edge_channels", "out_channels", "path_channels")
        )
        tile_items, column_tiles = max(1, 32 // d3), (paths + 63) // 64
        chunks = (d3 + 31) // 32
        tiles_per_item = column_tiles * chunks
        count = f"((items+{tile_items - 1})/{tile_items})*{column_tiles}"
        if segmented:
            count = f"((items+{tile_items - 1})/{tile_items})*{tiles_per_item}"
        totals.append(count)
        tasks.append(
            rf"""{{
    Index count = {count};
    if (task >= cursor && task < cursor + count) {{
        Index local = task - cursor;
        spec_id = {index};
        components = {d3};
        channels = {channels};
        paths = {paths};
        out_offset = {oo};
        weight_offset = {wo};
        tile_items = {tile_items};
        first_item = (local / {tiles_per_item}) * {tile_items};
        {f'row_components={min(d3, 32)}; first_component=((local/{column_tiles})%{chunks})*32;' if segmented else ''}
        first_path = (local % {column_tiles}) * 64;
    }}
    cursor += count;
}}"""
        )
        lines = [
            rf"""case {index}: {{
#pragma unroll 1
    for (int linear = threadIdx.x; linear < {tile_items * 64}; linear += {threads}) {{
        int local_item = linear / 64, local_path = linear % 64;
        Index item = first_item + local_item, p = first_path + local_path;
        int u = p / {edge}, v = p % {edge};
        bool valid = item < items && p < {paths};"""
        ]
        for target in y_outputs:
            lines += [f"        float {target}_{j}=0.f;" for j in range(d2)]
        lines.append("        if(valid) {")
        for target, pairs in x_outputs.items():
            lines.append("            {")
            for src in sorted({a for a, _ in pairs}):
                lines += [
                    f"                float {src}_{j}=__ldg({src}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j});"
                    for j in sorted({t[1] for t in spec["terms"]})
                ]
            lines += [f"                float acc{i}=0.f;" for i in range(d1)]
            for i, j, k, c in spec["terms"]:
                for src, z in pairs:
                    row = f"local_item*{d3}+{k}"
                    guard = ""
                    if segmented:
                        row = f"local_item*row_components+{k}-first_component"
                        guard = f"if({k}>=first_component && {k}<first_component+row_components) "
                    lines.append(
                        f"""                {guard}acc{i}=fmaf({float_literal(c, meta.get("dtype"))}*{src}_{j},shared.z[{slots[z]}][({row})*64+local_path],acc{i});"""
                    )
            lines += [
                f"                atomicAdd({target}+int64_t(item)*{xdim}+{xo}+u*{d1}+{i},acc{i});"
                for i in range(d1)
            ]
            lines.append("            }")
        for target, pairs in y_outputs.items():
            lines.append("            {")
            for src in sorted({a for a, _ in pairs}):
                lines += [
                    f"                float {src}_{i}=__ldg({src}+int64_t(item)*{xdim}+{xo}+u*{d1}+{i});"
                    for i in sorted({t[0] for t in spec["terms"]})
                ]
            for i, j, k, c in spec["terms"]:
                for src, z in pairs:
                    row = f"local_item*{d3}+{k}"
                    guard = ""
                    if segmented:
                        row = f"local_item*row_components+{k}-first_component"
                        guard = f"if({k}>=first_component && {k}<first_component+row_components) "
                    lines.append(
                        f"""                {guard}{target}_{j}=fmaf({float_literal(c, meta.get("dtype"))}*{src}_{i},shared.z[{slots[z]}][({row})*64+local_path],{target}_{j});"""
                    )
            lines.append("            }")
        lines.append("        }")
        for target in y_outputs:
            for j in range(d2):
                if edge == 1:
                    lines += [
                        rf"""#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            {target}_{j} +=
                __shfl_down_sync(0xffffffff, {target}_{j}, delta);
        if ((threadIdx.x & 31) == 0 && item < items)
            atomicAdd({target} + int64_t(item) * {ydim} + {yo + j},
                      {target}_{j});"""
                    ]
                else:
                    lines += [
                        f"        if(valid) atomicAdd({target}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j},{target}_{j});"
                    ]
        lines += ["    }", "    break;", "}"]
        epilogues.append("\n".join(lines))

    params = [f"const float* __restrict__ {v}" for v in inputs]
    params += [f"float* __restrict__ {v}" for v in outputs] + ["Index items"]
    if fp64:
        loads = []
        for gi, g in enumerate(gradients):
            loads.append(
                rf"""for (int index = threadIdx.x; index < 32 * 32; index += {threads}) {{
    Index row = index / 32, k = index % 32, item = first_item + row / {'row_components' if segmented else 'components'};
    int component = row % {'row_components+first_component' if segmented else 'components'};
    shared.tiles.a[buffer][{gi}][row * 33 + k] =
        row < tile_items * {'row_components' if segmented else 'components'} {'&& component<components' if segmented else ''} && item < items && tile_k + k < channels
            ? __ldg({g} + int64_t(item) * {odim} + out_offset + (tile_k + k) * components +
                    component)
            : 0.f;
}}"""
            )
        for wi, w in enumerate(weights):
            loads.append(rf"""for (int index = threadIdx.x; index < 64 * 32; index += {threads}) {{
    int path = index / 32, k = index % 32;
    shared.tiles.b[buffer][{wi}][path * 33 + k] =
        first_path + path < paths && tile_k + k < channels
            ? __ldg({w} + weight_offset + (first_path + path) * channels + tile_k + k)
            : 0.f;
}}""")
        shared_bytes = 8 * max(
            buffers * (len(gradients) * 32 * 33 + len(weights) * 64 * 33),
            len(contractions) * 32 * 64,
        )
        shared_declaration = f"""union Storage {{
    struct {{
        float a[{buffers}][{len(gradients)}][32 * 33];
        float b[{buffers}][{len(weights)}][64 * 33];
    }} tiles;
    float z[{len(contractions)}][32 * 64];
}};
extern __shared__ __align__(16) unsigned char storage[];
Storage &shared = *reinterpret_cast<Storage *>(storage);
"""
        mma_definition = f"""double acc[{len(contractions)}][4][4] = {{}};
"""
        tensor_gradient_loads = "\n".join(
            rf"""double a{gi}[8];
#pragma unroll
for (int i = 0; i < 8; ++i)
    a{gi}[i] =
        shared.tiles.a[buffer][{gi}][(local_row + (i & 1) * 8) * 33 + k + (lane & 3) + (i >> 1) * 4];"""
            for gi in range(len(gradients))
        )
        tensor_weight_loads = "\n".join(
            rf"""double b{wi}[4];
#pragma unroll
for (int i = 0; i < 4; ++i)
    b{wi}[i] =
        shared.tiles.b[buffer][{wi}][(local_column + n * 8 + (lane >> 2)) * 33 + k + (lane & 3) + i * 4];"""
            for wi in range(len(weights))
        )
        mma = get_fp64_mma(arch)[1]
        tensor_contractions = []
        for z, pairs in contractions.items():
            zi = slots[z]
            for w, g in pairs:
                a = f"a{gradients.index(g)}"
                b = f"b{weights.index(w)}"
                c = f"acc[{zi}][n]"
                tensor_contractions.append(
                    f"{mma}::fma({c}[0],{c}[1],{c}[2],{c}[3],{a}[0],{a}[1],{a}[2],{a}[3],{a}[4],{a}[5],{a}[6],{a}[7],{b}[0],{b}[1],{b}[2],{b}[3],{c}[0],{c}[1],{c}[2],{c}[3]);"
                )
        mma_compute = rf"""for (Index tile_k = 0; tile_k < channels; tile_k += 32) {{
    int buffer = (tile_k / 32) % {buffers};
{indent(chr(10).join(loads), '    ')}
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int local_row = (warp >> 1) * 16 + (lane >> 2), local_column = (warp & 1) * 32;
#pragma unroll
    for (int k = 0; k < 32; k += 16) {{
{indent(tensor_gradient_loads, '        ')}
#pragma unroll
        for (int n = 0; n < 4; ++n) {{
{indent(tensor_weight_loads, '            ')}
{indent(chr(10).join(tensor_contractions), '            ')}
        }}
    }}
    __syncthreads();
}}
"""
        mma_stores = rf"""const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
for (int n = 0; n < 4; ++n) {{
#pragma unroll
    for (int i = 0; i < 4; ++i) {{
        int row = (warp >> 1) * 16 + (lane >> 2) + (i >> 1) * 8,
            column = (warp & 1) * 32 + n * 8 + (lane & 3) * 2 + (i & 1);
{indent(chr(10).join((f'shared.z[{zi}][row*64+column]=acc[{zi}][n][i];' for zi in range(len(contractions)))), '        ')}
    }}
}}
__syncthreads();
"""
    else:
        shared_declaration = f"""struct Storage {{
    typename {name}_Mma::SharedStorage mma;
    float z[{len(contractions)}][32 * 64];
}};
__shared__ Storage shared;
"""
        mma_definition = f"""using Mma = {name}_Mma;
"""
        mma_compute = ""
        for z, pairs in contractions.items():
            mma_compute += """{
    typename Mma::FragmentC acc;
    acc.clear();
"""
            for w, g in pairs:
                mma_compute += f"""    {{
        {name}_IteratorA a({g} + int64_t(first_item) * {odim} + out_offset,
                                      int(min(Index(tile_items), items - first_item)), components, channels,
                                      threadIdx.x, {('row_components,first_component' if segmented else 'components,0')});
        {name}_IteratorB b({w} + weight_offset, channels, paths, first_path, threadIdx.x);
        Mma mma(shared.mma, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
        mma((channels + 7) / 8, acc, a, b, acc);
        __syncthreads();
    }}
"""
            mma_compute += f"""    typename Mma::Operator::IteratorC c({{shared.z[{slots[z]}], 64}}, threadIdx.x % 32);
    c.add_tile_offset({{int(threadIdx.x / 32) % 2, int(threadIdx.x / 32) / 2}});
    c.store(acc);
    __syncthreads();
}}
"""
        mma_stores = ""
    code = rf"""
__global__ __launch_bounds__({threads}, {resident_blocks}) void {name}({','.join(params)}) {{
{indent(shared_declaration, '    ')}
    Index task = blockIdx.x, cursor = 0;
    int spec_id = 0, components = 0, channels = 0, paths = 0;
    int out_offset = 0, weight_offset = 0, tile_items = 0;
    Index first_item = 0;
    int first_path = 0;
{indent('int row_components=0,first_component=0;' if segmented else '', '    ')}
{indent(chr(10).join(tasks), '    ')}
{indent(mma_definition, '    ')}
{indent(mma_compute, '    ')}
    // Every warp must finish reading tiles before the union becomes Z storage.
    __syncthreads();
{indent(mma_stores, '    ')}
    switch (spec_id) {{
{indent(chr(10).join(epilogues), '        ')}
    }}
}}
"""
    if not fp64:
        code = f"""
#include <cutlass/gemm/device/gemm.h>
using {name}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<16, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel::Mma;
struct {name}_IteratorA {{
    using Element = float;
    using ThreadMap = typename {name}_Default::IteratorA::ThreadMap;
    using Fragment = typename {name}_Default::IteratorA::Fragment;
    const float *pointer;
    int components, columns;
    int64_t row_offsets[ThreadMap::Iterations::kStrided];
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_IteratorA(const float *p, int r, int d, int k, int thread, int rd, int cd)
        : pointer(p), components(d), columns(k), offset(ThreadMap::initial_offset(thread)) {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            int row = offset.strided() + s * ThreadMap::Delta::kStrided;
            int component = row % rd + cd;
            row_offsets[s] = row < r * rd && component < d ? int64_t(row / rd) * {odim} + component : -1;
        }}
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorA &operator++() {{
        offset.contiguous() += 8;
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int column = offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                float value = 0;
                if (enabled && row_offsets[s] >= 0 && column < columns)
                    value = pointer[row_offsets[s] + int64_t(column) * components];
                fragment[s * ThreadMap::Iterations::kContiguous + c] = value;
            }}
        }}
    }}
}};
struct {name}_IteratorB {{
    using Element = float;
    using ThreadMap = typename {name}_Default::IteratorB::ThreadMap;
    using Fragment = typename {name}_Default::IteratorB::Fragment;
    const float *pointer;
    int channels, paths, first_path;
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_IteratorB(const float *p, int k, int n, int first, int thread)
        : pointer(p), channels(k), paths(n), first_path(first), offset(ThreadMap::initial_offset(thread)) {{
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorB &operator++() {{
        offset.contiguous() += 8;
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int k = offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                int p = first_path + offset.strided() + s * ThreadMap::Delta::kStrided;
                fragment[s * ThreadMap::Iterations::kContiguous + c] =
                    enabled && k < channels && p < paths ? pointer[int64_t(p) * channels + k] : 0.f;
            }}
        }}
    }}
}};
using {name}_Mma =
    cutlass::gemm::threadblock::MmaPipelined<typename {name}_Default::Shape, {name}_IteratorA,
                                             typename {name}_Default::SmemIteratorA,
                                             {name}_IteratorB,
                                             typename {name}_Default::SmemIteratorB, float,
                                             cutlass::layout::RowMajor, typename {name}_Default::Policy>;
""" + code
    if fp64:
        code += f"""
constexpr int {name}_shared_bytes = {shared_bytes};
"""
    return code, "+".join(totals), threads


def make_uvw_backward_dw_shared_kernel(meta, program, inputs, outputs, name, *, arch):
    if meta.get("dtype") is not torch.float64:
        targets = {k: v for k, v in program['w'].items() if k in outputs}
        products = {q: program['cg'][q] for pairs in targets.values() for q, _ in pairs}
        max_pairs = max(len(pairs) for pairs in products.values())
        pair_names = [f'{side}{i}' for i in range(max_pairs) for side in ('left','right')]
        pair_declarations = """;
""".join(f'const float* {name}=nullptr' for name in pair_names)+';'
        pair_arguments = ','.join(pair_names)
        pair_parameters = ','.join(f'const float* {name}' for name in pair_names)
        pair_initializers = ','.join(f'{name}({name})' for name in pair_names)
        left_select = ''.join(f'pair=={i}?left{i}:' for i in range(max_pairs-1))+f'left{max_pairs-1}'
        right_select = ''.join(f'pair=={i}?right{i}:' for i in range(max_pairs-1))+f'right{max_pairs-1}'
        odim = meta['out_dim']
        tile_columns = 128 if all(s['out_channels'] >= 128 for s in meta['generated_specs']) else 64
        tile_rows = tile_columns // 2
        threads = tile_columns * 2
        cases, tasks, totals = [], [], []
        tiles = sum(((s['path_channels']+tile_rows-1)//tile_rows)*((s['out_channels']+tile_columns-1)//tile_columns) for s in meta['generated_specs'])
        split_capacity = max(1, min(UVW_WEIGHT_MAX_SPLITS, (UVW_TARGET_BLOCKS + tiles - 1) // tiles))
        for index,spec in enumerate(meta['generated_specs']):
            d3 = spec['d3']
            paths,channels,edge=(spec[k] for k in ('path_channels','out_channels','edge_channels'))
            oo, wo = spec['out_offset'], spec['weight_offset']
            rows,columns=(paths+tile_rows-1)//tile_rows,(channels+tile_columns-1)//tile_columns
            split_expr=f'min(int64_t({split_capacity}),(int64_t(items)*{d3}+63)/64)'
            count=f'({rows*columns}*({split_expr}))'
            totals.append(count)
            tasks.append(f'''if (task >= cursor && task < cursor + {count}) {{
    int splits = int({split_expr});
    Index local = task - cursor, tile = local / splits;
    int split = local % splits;
    spec_id = {index};
    components = {d3};
    paths = {paths};
    channels = {channels};
    out_offset = {oo};
    weight_offset = {wo};
    first_path = (tile / {columns}) * {tile_rows};
    first_channel = (tile % {columns}) * {tile_columns};
    int64_t tiles = (int64_t(items) * {d3} + 15) / 16, per_split = (tiles + splits - 1) / splits;
    k_begin = split * per_split * 16;
    k_end = k_begin + min(int64_t(items) * {d3} - k_begin, per_split * 16);
}}
cursor += {count};''')
            body=[f'case {index}: {{ if constexpr(Components=={d3}) {{ int u=path/{edge},v=path%{edge};']
            body.append(indent('\n'.join(f'float z{k}=0.f;' for k in range(d3)), '        '))
            body.append('        if(enabled && path<paths && item*Components<end) {')
            body.append(f"""
#pragma unroll 1
            for (int pair = 0; pair < pair_count; ++pair) {{
                const float *lhs = {left_select};
                const float *rhs = {right_select};""")
            body.append(indent(emit_cg_source(meta,spec,pairs=(("lhs","rhs"),),coefficient_first=False), '                '))
            body.append('            }}')
            for k in range(d3):
                body.append(f"""            {{
                int64_t r = item * Components + {k} - start;
                if (r >= 0 && r < 16)
                    scratch[r * {tile_rows} + i % {tile_rows}] =
                        (start + r < end) ? z{k} : 0.f;
            }}""")
            body.append('        } break; }')
            cases.append('\n'.join(body))
        ptrs=', '.join(f'const float* {v}' for v in inputs)
        terms = [(q, g) for pairs in targets.values() for q, g in pairs]
        term_cases = []
        for term, (q, grad) in enumerate(terms):
            assignments = '\n'.join(f"""left{i} = {x};
right{i} = {y};""" for i,(x,y) in enumerate(products[q]))
            term_cases.append(f"""case {term}:
{indent(assignments, '    ')}
    gradient = {grad} + out_offset;
    pair_count = {len(products[q])};
    break;""")
        target_cases, offset = [], 0
        for target, pairs in targets.items():
            target_cases.append(f"""case {len(target_cases)}:
    output = {target};
    first_term = {offset};
    last_term = {offset+len(pairs)};
    break;""")
            offset += len(pairs)
        operations=f'''
#pragma unroll 1
for (int target = 0; target < {len(targets)}; ++target) {{
    typename Mma::FragmentC acc;
    acc.clear();
    float *output = nullptr;
    int first_term = 0, last_term = 0;
    switch (target) {{
{indent(chr(10).join(target_cases), '        ')}
    }}
#pragma unroll 1
    for (int term = first_term; term < last_term; ++term) {{
{indent(pair_declarations, '        ')}
        const float *gradient = nullptr;
        int pair_count = 0;
        switch (term) {{
{indent(chr(10).join(term_cases), '            ')}
        }}
        for (int64_t chunk = k_begin; chunk < k_end;) {{
            const int64_t chunk_end = chunk + min(int64_t(1) << 30, k_end - chunk);
            {name}_A<Components> a({pair_arguments}, pair_count, spec_id, paths, first_path, chunk,
                                              chunk_end, threadIdx.x, shared.tiles.scratch);
            {name}_B<Components> b(gradient, channels, first_channel, chunk, chunk_end, threadIdx.x);
            Mma mma(reinterpret_cast<typename Mma::SharedStorage &>(shared.tiles.mma), threadIdx.x, threadIdx.x / 32,
                    threadIdx.x % 32);
            mma(int((chunk_end - chunk + 15) / 16), acc, a, b, acc);
            __syncthreads();
            chunk = chunk_end;
        }}
    }}
    typename Mma::Operator::IteratorC c({{shared.result, {tile_columns}}}, threadIdx.x % 32);
    c.add_tile_offset({{int(threadIdx.x / 32) % {tile_rows//16}, int(threadIdx.x / 32) / {tile_rows//16}}});
    c.store(acc);
    __syncthreads();
    for (int i = threadIdx.x; i < {tile_rows} * {tile_columns}; i += {threads}) {{
        int p = first_path + i / {tile_columns}, c = first_channel + i % {tile_columns};
        if (p < paths && c < channels)
            atomicAdd(output + int64_t(weight_offset) + int64_t(p) * channels + c, shared.result[i]);
    }}
    __syncthreads();
}}'''
        dispatch = '\n'.join(f"""case {d}: {{
    constexpr int Components = {d};
    using Mma = {name}_Mma<Components>;"""+indent(operations, '    ')+'    break;\n}' for d in sorted({s['d3'] for s in meta['generated_specs']}))
        code=f'''
#include <cutlass/gemm/device/gemm.h>
using {name}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<{tile_rows}, {tile_columns}, 16>,
    cutlass::gemm::GemmShape<16, {tile_columns//2}, 16>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel::Mma;
template <int Components> struct {name}_A {{
    using Element = float;
    using ThreadMap = typename {name}_Default::IteratorA::ThreadMap;
    using Fragment = typename {name}_Default::IteratorA::Fragment;
{indent(pair_declarations, '    ')}
    int pair_count;
    int spec, paths, first_path;
    Index start, end;
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    float *scratch;
    CUTLASS_DEVICE {name}_A({pair_parameters}, int pairs, int s, int p, int first, Index begin, Index last,
                                       int thread, float *buffer)
        : {pair_initializers}, pair_count(pairs), spec(s), paths(p), first_path(first), start(begin), end(last),
          offset(ThreadMap::initial_offset(thread)), scratch(buffer) {{
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_A &operator++() {{
        start += min(Index(16), end - start);
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
        Index first_item = start / Components;
        const int count = (int(start % Components) + 15) / Components + 1;
        for (int i = threadIdx.x; i < count * {tile_rows}; i += {threads}) {{
            int64_t item = int64_t(first_item) + i / {tile_rows};
            int path = first_path + i % {tile_rows};
            switch (spec) {{
{indent(chr(10).join(cases), '                ')}
            }}
        }}
        __syncthreads();
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int path = offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                int reduction = offset.strided() + s * ThreadMap::Delta::kStrided;
                fragment[s * ThreadMap::Iterations::kContiguous + c] = scratch[reduction * {tile_rows} + path];
            }}
        }}
        __syncthreads();
    }}
}};
template <int Components> struct {name}_B {{
    using Element = float;
    using ThreadMap = typename {name}_Default::IteratorB::ThreadMap;
    using Fragment = typename {name}_Default::IteratorB::Fragment;
    const float *pointer;
    int channels, first_channel;
    Index start, end;
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_B(const float *p, int n, int first, Index begin, Index last, int thread)
        : pointer(p), channels(n), first_channel(first), start(begin), end(last),
          offset(ThreadMap::initial_offset(thread)) {{
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_B &operator++() {{
        start += min(Index(16), end - start);
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int column = first_channel + offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                int local = offset.strided() + s * ThreadMap::Delta::kStrided;
                float value = 0;
                if (enabled && column < channels && local < end - start) {{
                    Index reduction = start + local;
                    Index item = reduction / Components;
                    int component = int(reduction % Components);
                    value = pointer[int64_t(item) * {odim} + int64_t(column) * Components + component];
                }}
                fragment[s * ThreadMap::Iterations::kContiguous + c] = value;
            }}
        }}
    }}
}};
template <int Components>
using {name}_Mma = cutlass::gemm::threadblock::MmaPipelined<
    typename {name}_Default::Shape, {name}_A<Components>,
    typename {name}_Default::SmemIteratorA, {name}_B<Components>,
    typename {name}_Default::SmemIteratorB, float, cutlass::layout::RowMajor,
    typename {name}_Default::Policy>;
__global__ __launch_bounds__({threads}) void {name}({ptrs}, {','.join(f'float* {o}' for o in outputs)},
                                                                         Index items) {{
    using Mma = {name}_Mma<1>;
    union Storage {{
        struct {{
            typename Mma::SharedStorage mma;
            float scratch[{tile_rows} * 16];
        }} tiles;
        float result[{tile_rows} * {tile_columns}];
    }};
    __shared__ Storage shared;
    Index task = blockIdx.x, cursor = 0;
    int64_t k_begin = 0, k_end = 0;
    int spec_id = 0, components = 0, paths = 0, channels = 0, out_offset = 0, weight_offset = 0, first_path = 0,
        first_channel = 0;
{indent(chr(10).join(tasks), '    ')}
    if (k_begin >= k_end)
        return;
    switch (components) {{
{indent(dispatch, '        ')}
    }}
}}
'''
        return code,'+'.join(totals),threads

    targets = {k: v for k, v in program["w"].items() if k in outputs}
    needed = {q for pairs in targets.values() for q, _ in pairs}
    products = {q: v for q, v in program["cg"].items() if q in needed}
    gradients = sorted({g for pairs in targets.values() for _, g in pairs})
    product_indices = {q: index for index, q in enumerate(products)}
    gradient_indices = {g: index for index, g in enumerate(gradients)}
    tile_columns, tile_rows, tile_reduction, mma_rows = 64, 64, 16, 8
    threads = tile_rows * tile_columns // (mma_rows * 4)
    buffers = 1
    resident_blocks = 2
    odim = meta["out_dim"]
    # Schedule long reductions first so the last wave contains lighter tiles.
    specs = sorted(
        meta["generated_specs"],
        key=lambda spec: (
            -(len(spec["terms"]) + tile_columns * spec["d3"])
            / (2 if spec["d3"] == 1 else 4),
            spec["weight_offset"],
        ),
    )
    output_tiles = sum(
        ((s["path_channels"] + tile_rows - 1) // tile_rows)
        * ((s["out_channels"] + tile_columns - 1) // tile_columns)
        for s in specs
    )
    # Round up: a grid just below the target should still fill the next wave.
    split_scale = min(4, (128 + max(1, output_tiles) - 1) // max(1, output_tiles))
    tasks, totals, vector_cases = [], [], []
    for index, spec in enumerate(specs):
        d3 = spec["d3"]
        oo, wo = spec["out_offset"], spec["weight_offset"]
        paths, channels, edge = (
            spec[k] for k in ("path_channels", "out_channels", "edge_channels")
        )
        splits = (2 if d3 == 1 else 4) * split_scale
        columns, rows = (
            (channels + tile_columns - 1) // tile_columns,
            (paths + tile_rows - 1) // tile_rows,
        )
        count = rows * columns * splits
        totals.append(str(count))
        tasks.append(
            rf"""{{
    if (task >= cursor && task < cursor + {count}) {{
        Index local = task - cursor, tile = local / {splits};
        int split = local % {splits};
        spec_id = {index};
        components = {d3};
        paths = {paths};
        channels = {channels};
        out_offset = {oo};
        weight_offset = {wo};
        first_path = (tile / {columns}) * {tile_rows};
        first_channel = (tile % {columns}) * {tile_columns};
        Index tiles = (items * {d3} + {tile_reduction - 1}) / {tile_reduction},
              per_split = (tiles + {splits - 1}) / {splits};
        k_begin = split * per_split * {tile_reduction};
        k_end = min(items * {d3}, k_begin + per_split * {tile_reduction});
    }}
    cursor += {count};
}}"""
        )
        vector_lines = [f"case {index}: {{", f"    int u=p/{edge},v=p%{edge};"]
        for qi, pairs in enumerate(products.values()):
            vector_lines += [
                "    {",
                *(f"        float z{k}=0.f;" for k in range(d3)),
                "        if(p<paths && item<items) {",
                indent(emit_cg_source(meta, spec, pairs), '            '),
                "        }",
            ]
            for k in range(d3):
                vector_lines += [
                    rf"""        {{
            Index reduction = item * {d3} + {k};
            if (reduction >= tile_k && reduction < tile_k + {tile_reduction}) {{
                shared_a[buffer][{qi}][(reduction - tile_k) * {tile_rows} + local_path] =
                    reduction < k_end ? z{k} : 0.f;
            }}
        }}"""
                ]
            vector_lines.append("    }")
        vector_lines += ["    break;", "}"]
        vector_cases.append("\n".join(vector_lines))

    loads = []
    for gi, g in enumerate(gradients):
        loads += [
            rf"""for (int index = threadIdx.x; index < {tile_reduction} * {tile_columns}; index += {threads}) {{
    Index reduction = tile_k + index / {tile_columns}, channel = first_channel + index % {tile_columns};
    Index item = reduction / components;
    int component = reduction % components;
    shared_b[buffer][{gi}][index] = reduction < k_end && channel < channels
                                                     ? __ldg({g} + int64_t(item) * {odim} +
                                                             out_offset + channel * components + component)
                                                     : 0.f;
}}"""
        ]

    params = [f"const float* __restrict__ {v}" for v in inputs]
    params += [f"float* __restrict__ {v}" for v in outputs] + ["Index items"]
    shared_bytes = (
        8
        * buffers
        * (
            len(products) * tile_rows * tile_reduction
            + len(gradients) * tile_reduction * tile_columns
        )
    )
    mma_definition = f"""double acc[{len(targets)}][2][4][4] = {{}};
"""
    tensor_product_loads = "\n".join(
        rf"""double a{qi}[8];
#pragma unroll
for (int i = 0; i < 8; ++i)
    a{qi}[i] = shared_a[buffer][{qi}][((lane & 3) + (i >> 1) * 4) * {tile_rows} +
                                                                local_row + m * 16 + (i & 1) * 8];"""
        for qi in range(len(products))
    )
    tensor_gradient_loads = "\n".join(
        rf"""double b{gi}[4];
#pragma unroll
for (int i = 0; i < 4; ++i)
    b{gi}[i] = shared_b[buffer][{gi}]
                                    [((lane & 3) + i * 4) * {tile_columns} + local_column + n * 8 + (lane >> 2)];"""
        for gi in range(len(gradients))
    )
    mma = get_fp64_mma(arch)[1]
    tensor_contractions = []
    for ti, (target, pairs) in enumerate(targets.items()):
        for q, g in pairs:
            a = f"a{product_indices[q]}"
            b = f"b{gradient_indices[g]}"
            c = f"acc[{ti}][m][n]"
            tensor_contractions.append(
                f"{mma}::fma({c}[0],{c}[1],{c}[2],{c}[3],{a}[0],{a}[1],{a}[2],{a}[3],{a}[4],{a}[5],{a}[6],{a}[7],{b}[0],{b}[1],{b}[2],{b}[3],{c}[0],{c}[1],{c}[2],{c}[3]);"
            )
    code = rf"""
__global__ __launch_bounds__({threads}, {resident_blocks}) void {name}({','.join(params)}) {{
    extern __shared__ __align__(16) unsigned char storage[];
    auto shared_a = reinterpret_cast<float(*)[{len(products)}][{tile_rows} * {tile_reduction}]>(storage);
    auto shared_b = reinterpret_cast<float(*)[{len(gradients)}][{tile_reduction} * {tile_columns}]>(
        storage + sizeof(float) * {buffers * len(products) * tile_rows * tile_reduction});
    Index task = blockIdx.x, cursor = 0;
    int spec_id = 0, components = 0, channels = 0, paths = 0;
    int out_offset = 0, weight_offset = 0, first_path = 0, first_channel = 0;
    Index k_begin = 0, k_end = 0;
{indent(chr(10).join(tasks), '    ')}
{indent(mma_definition, '    ')}    for (Index tile_k = k_begin; tile_k < k_end; tile_k += {tile_reduction}) {{
        int buffer = (tile_k / {tile_reduction}) % {buffers};

        int item_count = (tile_k % components + {tile_reduction} + components - 1) / components;
        for (int index = threadIdx.x; index < item_count * {tile_rows}; index += {threads}) {{
            int local_path = index % {tile_rows}, p = first_path + local_path;
            Index item = tile_k / components + index / {tile_rows};
            switch (spec_id) {{
{indent(chr(10).join(vector_cases), '                ')}
            }}
        }}
{indent(chr(10).join(loads), '        ')}
        __syncthreads();
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int local_row = (warp >> 1) * 32 + (lane >> 2), local_column = (warp & 1) * 32;
#pragma unroll
        for (int m = 0; m < 2; ++m) {{
{indent(tensor_product_loads, '            ')}
#pragma unroll
            for (int n = 0; n < 4; ++n) {{
{indent(tensor_gradient_loads, '                ')}
{indent(chr(10).join(tensor_contractions), '                ')}
            }}
        }}
        __syncthreads();
    }}
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 0; m < 2; ++m) {{
#pragma unroll
        for (int n = 0; n < 4; ++n) {{
#pragma unroll
            for (int i = 0; i < 4; ++i) {{
                int p = first_path + (warp >> 1) * 32 + m * 16 + (lane >> 2) + (i >> 1) * 8,
                    channel = first_channel + (warp & 1) * 32 + n * 8 + (lane & 3) * 2 + (i & 1);
                if (p < paths && channel < channels) {{
{indent(chr(10).join((f'atomicAdd({target}+weight_offset+int64_t(p)*channels+channel,acc[{ti}][m][n][i]);' for ti, target in enumerate(targets))), '                    ')}
                }}
            }}
        }}
    }}
}}
"""
    code += f"""
constexpr int {name}_shared_bytes = {shared_bytes};
"""
    return code, "+".join(totals), threads


def make_uvw_backward_external_tiled_kernel(meta, program, inputs, outputs, name):
    """Reuse FP32 operand pairs in bounded native SIMT contraction tiles."""
    specs = meta['generated_specs']
    x_outputs = {k: v for k, v in program['x'].items() if k in outputs}
    y_outputs = {k: v for k, v in program['y'].items() if k in outputs}
    w_outputs = {k: v for k, v in program['w'].items() if k in outputs}
    forward = program['forward'] if 'gg' in outputs else ()
    needed_z = {z for pairs in (*x_outputs.values(), *y_outputs.values()) for _, z in pairs}
    contractions = {k: v for k, v in program['z'].items() if k in needed_z}
    needed_q = {q for pairs in w_outputs.values() for q, _ in pairs} | {q for q, _ in forward}
    products = {k: v for k, v in program['cg'].items() if k in needed_q}
    weights = sorted({w for pairs in contractions.values() for w, _ in pairs} | {w for _, w in forward})
    gradients = sorted({g for pairs in (*contractions.values(), *w_outputs.values()) for _, g in pairs})
    edge_sources = sorted({b for pairs in products.values() for _, b in pairs} | {b for pairs in x_outputs.values() for b, _ in pairs})
    feature_sources = sorted({a for pairs in products.values() for a, _ in pairs} | {a for pairs in y_outputs.values() for a, _ in pairs})
    feature_slots = {a: i for i, a in enumerate(feature_sources)}
    q_slots = {q: i for i, q in enumerate(products)}
    z_slots = {z: i for i, z in enumerate(contractions)}
    w_slots = {w: i for i, w in enumerate(weights)}
    g_slots = {g: i for i, g in enumerate(gradients)}
    edge_slots = {b: i for i, b in enumerate(edge_sources)}
    tile_u = tile_c = UVW_EXTERNAL_TILE_CHANNELS
    weight_stride = tile_c + 2
    threads = UVW_THREADS
    max_p1 = max((s['d1'] + 1) // 2 * 2 for s in specs)
    max_p3 = max((s['d3'] + 1) // 2 * 2 for s in specs)
    # An odd float2 row stride distributes columns across shared bank pairs.
    max_grad_stride = 4 * (max_p3 // 4) + 2
    max_tiles = max(((s['in_channels'] + tile_u - 1) // tile_u) * s['edge_channels'] for s in specs)
    xdim, ydim, odim, wdim = (meta[k] for k in ('in1_dim', 'in2_dim', 'out_dim', 'weight_numel'))
    storage, storage_aliases = [], []
    overlap_feature = bool(feature_sources and weights)
    cache_feature = overlap_feature and bool(y_outputs)
    if edge_sources:
        storage.append(f'__shared__ __align__(16) float edge[{len(edge_sources)}][{max_p1 * max_p3}];')
    if overlap_feature:
        prefix = f'float retained[{len(feature_sources) - 1}][{tile_u * max_p1}];' if len(feature_sources) > 1 else ''
        storage.append(f'''union __align__(16) FeatureWeightStorage {{
    float feature[{len(feature_sources)}][{tile_u * max_p1}];
    struct {{
        {prefix}float weight[{len(weights)}][{tile_u * weight_stride}];
    }} matrix;
}};
__shared__ FeatureWeightStorage operands;''')
        storage_aliases += ['auto& feature=operands.feature;', 'auto& weight=operands.matrix.weight;']
    elif feature_sources:
        storage.append(f'__shared__ __align__(16) float feature[{len(feature_sources)}][{tile_u * max_p1}];')
    if products or y_outputs:
        members = []
        if products:
            members.append(f'float q[{len(products)}][{tile_u * max_p3}];')
        if y_outputs:
            members.append(f'float h[{len(y_outputs)}][{max_p1 * max_p3}];')
        storage.append('union __align__(16) IntermediateStorage {' + '\n'.join(members) + """};
__shared__ IntermediateStorage intermediate;""")
    if contractions:
        storage.append(f'__shared__ __align__(16) float z[{len(contractions)}][{tile_u * max_p3}];')
    if weights and not overlap_feature:
        storage.append(f'__shared__ __align__(16) float weight[{len(weights)}][{tile_u * weight_stride}];')
    if gradients:
        storage.append(f'__shared__ __align__(16) float grad[{len(gradients)}][{tile_c * max_grad_stride}];')
    feature_elements = len(feature_sources) * tile_u * max_p1
    weight_elements = len(weights) * tile_u * weight_stride
    operand_elements = (
        max(feature_elements, (len(feature_sources) - 1) * tile_u * max_p1 + weight_elements)
        if overlap_feature else feature_elements + weight_elements
    )
    shared_bytes = 4 * (
        len(edge_sources) * max_p1 * max_p3
        + operand_elements
        + max(len(products) * tile_u * max_p3,
              len(y_outputs) * max_p1 * max_p3)
        + len(contractions) * tile_u * max_p3
        + len(gradients) * tile_c * max_grad_stride
    )
    dynamic_bytes = shared_bytes if shared_bytes > UVW_STATIC_SHARED_BYTES else 0
    if dynamic_bytes:
        fields = "\n".join(storage).replace("__shared__ ", "")
        names = [name for name, needed in (
            ("edge", edge_sources), ("operands", overlap_feature),
            ("feature", feature_sources and not overlap_feature),
            ("intermediate", products or y_outputs), ("z", contractions),
            ("weight", weights and not overlap_feature), ("grad", gradients),
        ) if needed]
        storage = ["struct SharedStorage {", fields, "};",
                   f'static_assert(sizeof(SharedStorage)=={shared_bytes},"shared storage size");',
                   "extern __shared__ __align__(16) unsigned char shared_memory[];",
                   "auto& shared=*reinterpret_cast<SharedStorage*>(shared_memory);",
                   *(f"auto& {name}=shared.{name};" for name in names)]
    cases = []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ('d1', 'd2', 'd3'))
        p1, p3 = (d1 + 1) // 2 * 2, (d3 + 1) // 2 * 2
        grad_stride = 4 * (p3 // 4) + 2
        xo, yo, oo, wo = (spec[k] for k in ('x_offset', 'y_offset', 'out_offset', 'weight_offset'))
        edges, channels, in_channels = (spec[k] for k in ('edge_channels', 'out_channels', 'in_channels'))
        groups = p3 // 2
        ik_terms = {}
        for i, j, k, coefficient in spec['terms']:
            ik_terms.setdefault((i, k), []).append((j, coefficient))
        lines = [f'case {index}: {{', f'    if(tile>={((in_channels + tile_u - 1) // tile_u) * edges}) return;', f'    int v=tile%{edges},first_u=(tile/{edges})*{tile_u};']
        if edge_sources:
            lines += ['#pragma unroll 1', f"""    for (int index = threadIdx.x; index < {len(edge_sources) * p1 * p3}; index += {threads})
        edge[index / {p1*p3}][index % {p1*p3}] = 0.f;""", '    __syncthreads();']
            lines += ['#pragma unroll 1', f"""    for (int projection = threadIdx.x; projection < {len(ik_terms)}; projection += {threads}) {{""", '        switch(projection) {']
            for projection, ((i, k), terms) in enumerate(ik_terms.items()):
                lines.append(f'            case {projection}: {{')
                for b in edge_sources:
                    lines.append('                { float value=0.f;')
                    for j, coefficient in terms:
                        lines.append(f'                    value=fmaf({float_literal(coefficient)},__ldg({b}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j}),value);')
                    lines.append(f'                    edge[{edge_slots[b]}][{i*p3+k}]=value;')
                    lines.append('                }')
                lines.append('                break; }')
            lines += ['        }', '    }']
        if feature_sources:
            if cache_feature:
                feature_registers = (tile_u * p1 + threads - 1) // threads
                lines += [f'    float cached_feature[{feature_registers}]={{}};', '#pragma unroll', f"""    for (int slot = 0; slot < {feature_registers}; ++slot) {{""", f'        int index=threadIdx.x+slot*{threads};', f'        if(index<{tile_u*p1}) {{', f'            int u=first_u+index/{p1},component=index%{p1};']
            else:
                lines += ['#pragma unroll 1', f"""    for (int index = threadIdx.x; index < {tile_u*p1}; index += {threads}) {{""", f'        int u=first_u+index/{p1},component=index%{p1};']
            for a in feature_sources:
                value = f'u<{in_channels} && component<{d1}?__ldg({a}+int64_t(item)*{xdim}+{xo}+int64_t(u)*{d1}+component):0.f'
                if cache_feature and a == feature_sources[-1]:
                    lines.append(f'    cached_feature[slot]={value};')
                    value = 'cached_feature[slot]'
                lines.append(f'    feature[{feature_slots[a]}][index]={value};')
            lines.append('}')
            if cache_feature:
                lines.append('}')
        lines += ['    __syncthreads();', '#pragma unroll 1', f"""    for (int index = threadIdx.x; index < {tile_u//2*groups}; index += {threads}) {{""", f'        int local_u=(index/{groups})*2,component=(index%{groups})*2;']
        for q in products:
            lines.append(f"""        RowMma::FragmentC value_{q};
        value_{q}.clear();""")
        if products:
            q_features = sorted({a for pairs in products.values() for a, _ in pairs})
            q_edges = {b for pairs in products.values() for _, b in pairs}
            lines += ['#pragma unroll 1', f"""        for (int i = 0; i < {p1}; i += 2) {{"""]
            for a in q_features:
                lines.append(f'            RowMma::FragmentA a_{a};')
                for m in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(feature[{feature_slots[a]}] +
                                                                (local_u + {m}) * {p1} + i);
                a_{a}[{m*2}] = pair.x;
                a_{a}[{m*2+1}] = pair.y;
            }}""")
            for b in ('y', 'by', 'ry'):
                if b not in q_edges:
                    continue
                lines.append('{RowMma::FragmentB operand;')
                for k in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(
                    edge[{edge_slots[b]}] + (i + {k}) * {p3} + component);
                operand[{k*2}] = pair.x;
                operand[{k*2+1}] = pair.y;
            }}""")
                for q, pairs in products.items():
                    for a, edge in pairs:
                        if edge == b:
                            lines.append(f'            RowMma()(value_{q},a_{a},operand,value_{q});')
                lines.append('        }')
            lines.append('        }')
            for q in products:
                for m in range(2):
                    for n in range(2):
                        lines.append(f'        intermediate.q[{q_slots[q]}][(local_u+{m})*{p3}+component+{n}]=first_u+local_u+{m}<{in_channels} && component+{n}<{d3}?value_{q}[{m*2+n}]:0.f;')
        for z in contractions:
            for m in range(2):
                for n in range(2):
                    lines.append(f'        z[{z_slots[z]}][(local_u+{m})*{p3}+component+{n}]=0.f;')
        lines += ['    }', '    __syncthreads();', f"""    for (int channel_tile = 0; channel_tile < {(channels + tile_c - 1) // tile_c}; ++channel_tile) {{
        int first_c = channel_tile * {tile_c};"""]
        if weights:
            lines += ['#pragma unroll 1', f"""        for (int index = threadIdx.x; index < {tile_u*tile_c}; index += {threads}) {{""", f'            int u=first_u+index/{tile_c},c=first_c+index%{tile_c};']
            for w in weights:
                lines.append(f'''            {{
                float *destination = weight[{w_slots[w]}] + (index / {tile_c}) * {weight_stride} +
                                     index % {tile_c};
                bool valid = u < {in_channels} && c < {channels};
                const float *source =
                    {w} + (valid ? int64_t(item) * {wdim} + {wo} +
                                                     (int64_t(u) * {edges} + v) * {channels} + c
                                               : 0);
                unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
                int bytes = valid ? 4 : 0;
                asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;" ::"r"(address), "l"(source),
                             "r"(bytes)
                             : "memory");
            }}''')
            lines.append('        }')
        if gradients:
            lines += ['#pragma unroll 1', f"""        for (int index = threadIdx.x; index < {tile_c*grad_stride}; index += {threads}) {{""", f'            int c=first_c+index/{grad_stride},component=index%{grad_stride};']
            for g in gradients:
                lines.append(f'''            {{
                float *destination = grad[{g_slots[g]}] + index;
                bool valid = c < {channels} && component < {d3};
                const float *source =
                    {g} + (valid ? int64_t(item) * {odim} + {oo} +
                                                     int64_t(c) * {d3} + component
                                               : 0);
                unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
                int bytes = valid ? 4 : 0;
                asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;" ::"r"(address), "l"(source),
                             "r"(bytes)
                             : "memory");
            }}''')
            lines.append('        }')
        if weights or gradients:
            lines += ["""        asm volatile("cp.async.commit_group;" ::: "memory");""",
                      """        asm volatile("cp.async.wait_group 0;" ::: "memory");"""]
        lines.append('        __syncthreads();')
        if contractions:
            lines += ['#pragma unroll 1', f"""        for (int index = threadIdx.x; index < {tile_u//2*groups}; index += {threads}) {{""", f'            int local_u=(index/{groups})*2,component=(index%{groups})*2;']
            for z in contractions:
                lines.append(f'            RowMma::FragmentC value_{z};')
                for m in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(
                    z[{z_slots[z]}] + (local_u + {m}) * {p3} + component);
                value_{z}[{m*2}] = pair.x;
                value_{z}[{m*2+1}] = pair.y;
            }}""")
            lines += ['#pragma unroll 1', f"""            for (int c = 0; c < {tile_c}; c += 2) {{"""]
            for g in gradients:
                lines.append(f'                RowMma::FragmentB b_{g};')
                for k in range(2):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(
                        grad[{g_slots[g]}] + (c + {k}) * {grad_stride} + component);
                    b_{g}[{k*2}] = pair.x;
                    b_{g}[{k*2+1}] = pair.y;
                }}""")
            for w in ('bw', 'w', 'rw'):
                if w not in weights:
                    continue
                lines.append('{RowMma::FragmentA operand;')
                for m in range(2):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(
                        weight[{w_slots[w]}] + (local_u + {m}) * {weight_stride} + c);
                    operand[{m*2}] = pair.x;
                    operand[{m*2+1}] = pair.y;
                }}""")
                for z, pairs in contractions.items():
                    for weight, g in pairs:
                        if weight == w:
                            lines.append(f'                RowMma()(value_{z},operand,b_{g},value_{z});')
                lines.append('            }')
            lines.append('            }')
            for z in contractions:
                for m in range(2):
                    for n in range(2):
                        lines.append(f'            z[{z_slots[z]}][(local_u+{m})*{p3}+component+{n}]=component+{n}<{d3}?value_{z}[{m*2+n}]:0.f;')
            lines.append('        }')
        if w_outputs:
            lines += ['#pragma unroll 1', f"""        for (int index = threadIdx.x; index < {tile_u//2*(tile_c//4)}; index += {threads}) {{""", f'            int local_u=(index/{tile_c//4})*2,local_c=index%{tile_c//4};']
            for out in w_outputs:
                lines.append(f"""            WeightMma::FragmentC value_{out};
            value_{out}.clear();""")
            lines += ['#pragma unroll 1', f"""            for (int component = 0; component < {p3}; component += 2) {{"""]
            w_gradients = sorted({g for pairs in w_outputs.values() for _, g in pairs})
            for g in w_gradients:
                lines.append(f'                WeightMma::FragmentB b_{g};')
                for n in range(4):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(
                        grad[{g_slots[g]}] + (local_c + {n*(tile_c//4)}) * {grad_stride} + component);
                    b_{g}[{n*2}] = pair.x;
                    b_{g}[{n*2+1}] = pair.y;
                }}""")
            for q in dict.fromkeys(q for pairs in w_outputs.values() for q, _ in pairs):
                lines.append('{WeightMma::FragmentA operand;')
                for m in range(2):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(intermediate.q[{q_slots[q]}] +
                                                                    (local_u + {m}) * {p3} +
                                                                    component);
                    operand[{m*2}] = pair.x;
                    operand[{m*2+1}] = pair.y;
                }}""")
                for out, pairs in w_outputs.items():
                    for product, g in pairs:
                        if product == q:
                            lines.append(f'                WeightMma()(value_{out},operand,b_{g},value_{out});')
                lines.append('            }')
            lines.append('            }')
            for m in range(2):
                for n in range(4):
                    lines.append(f'            {{int u=first_u+local_u+{m},c=first_c+local_c+{n*(tile_c//4)};if(u<{in_channels} && c<{channels}) {{')
                    for out in w_outputs:
                        lines.append(f'                    {out}[int64_t(item)*{wdim}+{wo}+(int64_t(u)*{edges}+v)*{channels}+c]=value_{out}[{m*4+n}];')
                    lines.append('                }}')
            lines.append('        }')
        if forward:
            lines += ['#pragma unroll 1', f"""        for (int index = threadIdx.x; index < {tile_c//2*groups}; index += {threads}) {{""", f'            int local_c=(index/{groups})*2,component=(index%{groups})*2;', """            TransposedMma::FragmentC value;
            value.clear();""", '#pragma unroll 1', f"""            for (int u = 0; u < {tile_u}; u += 2) {{"""]
            for q in dict.fromkeys(q for q, _ in forward):
                lines.append(f'                TransposedMma::FragmentB b_{q};')
                for k in range(2):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(
                        intermediate.q[{q_slots[q]}] + (u + {k}) * {p3} + component);
                    b_{q}[{k*2}] = pair.x;
                    b_{q}[{k*2+1}] = pair.y;
                }}""")
            for w in dict.fromkeys(w for _, w in forward):
                lines.append('{TransposedMma::FragmentA operand;')
                for k in range(2):
                    lines.append(f"""                {{
                    float2 pair = *reinterpret_cast<const float2 *>(
                        weight[{w_slots[w]}] + (u + {k}) * {weight_stride} + local_c);
                    operand[{k*2}] = pair.x;
                    operand[{k*2+1}] = pair.y;
                }}""")
                for q, weight in forward:
                    if weight == w:
                        lines.append(f'                TransposedMma()(value,operand,b_{q},value);')
                lines.append('            }')
            lines.append('            }')
            for m in range(2):
                for n in range(2):
                    lines.append(f'            if(first_c+local_c+{m}<{channels} && component+{n}<{d3}) atomicAdd(gg+int64_t(item)*{odim}+{oo}+int64_t(first_c+local_c+{m})*{d3}+component+{n},value[{m*2+n}]);')
            lines.append('        }')
        lines += ['        __syncthreads();', '    }']
        if cache_feature:
            # Weight readers have finished; restore the feature before H uses it.
            lines += ['#pragma unroll', f"""    for (int slot = 0; slot < {feature_registers}; ++slot) {{""", f'        int index=threadIdx.x+slot*{threads};', f'        if(index<{tile_u*p1}) feature[{len(feature_sources) - 1}][index]=cached_feature[slot];', '    }']
        if x_outputs:
            lines += ['#pragma unroll 1', f"""    for (int index = threadIdx.x; index < {tile_u//2*(p1//2)}; index += {threads}) {{""", f'        int local_u=(index/{p1//2})*2,component=(index%{p1//2})*2;']
            for out in x_outputs:
                lines.append(f"""        ColMma::FragmentC value_{out};
        value_{out}.clear();""")
            lines += ['#pragma unroll 1', f"""        for (int k = 0; k < {p3}; k += 2) {{"""]
            for b in edge_sources:
                lines.append(f'            ColMma::FragmentB b_{b};')
                for n in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(
                    edge[{edge_slots[b]}] + (component + {n}) * {p3} + k);
                b_{b}[{n*2}] = pair.x;
                b_{b}[{n*2+1}] = pair.y;
            }}""")
            for z in contractions:
                if not any(z == source for pairs in x_outputs.values() for _, source in pairs):
                    continue
                lines.append('{ColMma::FragmentA operand;')
                for m in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(z[{z_slots[z]}] +
                                                                (local_u + {m}) * {p3} + k);
                operand[{m*2}] = pair.x;
                operand[{m*2+1}] = pair.y;
            }}""")
                for out, pairs in x_outputs.items():
                    for b, source in pairs:
                        if source == z:
                            lines.append(f'            ColMma()(value_{out},operand,b_{b},value_{out});')
                lines.append('        }')
            lines.append('        }')
            for m in range(2):
                for n in range(2):
                    lines.append(f'        {{int u=first_u+local_u+{m};if(u<{in_channels} && component+{n}<{d1}) {{')
                    for out in x_outputs:
                        lines.append(f'                atomicAdd({out}+int64_t(item)*{xdim}+{xo}+int64_t(u)*{d1}+component+{n},value_{out}[{m*2+n}]);')
                    lines.append('            }}')
            lines.append('    }')
        if y_outputs:
            if cache_feature:
                lines.append('    __syncthreads();')
            lines += ['#pragma unroll 1', f"""    for (int index = threadIdx.x; index < {p1//2*groups}; index += {threads}) {{""", f'        int i=(index/{groups})*2,component=(index%{groups})*2;']
            for out in y_outputs:
                lines.append(f"""        TransposedMma::FragmentC value_{out};
        value_{out}.clear();""")
            lines += ['#pragma unroll 1', f"""        for (int u = 0; u < {tile_u}; u += 2) {{"""]
            y_features = sorted({a for pairs in y_outputs.values() for a, _ in pairs})
            for a in y_features:
                lines.append(f'            TransposedMma::FragmentA a_{a};')
                for k in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(feature[{feature_slots[a]}] +
                                                                (u + {k}) * {p1} + i);
                a_{a}[{k*2}] = pair.x;
                a_{a}[{k*2+1}] = pair.y;
            }}""")
            for z in contractions:
                if not any(z == source for pairs in y_outputs.values() for _, source in pairs):
                    continue
                lines.append('{TransposedMma::FragmentB operand;')
                for k in range(2):
                    lines.append(f"""            {{
                float2 pair = *reinterpret_cast<const float2 *>(
                    z[{z_slots[z]}] + (u + {k}) * {p3} + component);
                operand[{k*2}] = pair.x;
                operand[{k*2+1}] = pair.y;
            }}""")
                for out, pairs in y_outputs.items():
                    for a, source in pairs:
                        if source == z:
                            lines.append(f'            TransposedMma()(value_{out},a_{a},operand,value_{out});')
                lines.append('        }')
            lines.append('        }')
            for output_index, out in enumerate(y_outputs):
                for m in range(2):
                    for n in range(2):
                        lines.append(f'        intermediate.h[{output_index}][(i+{m})*{p3}+component+{n}]=i+{m}<{d1} && component+{n}<{d3}?value_{out}[{m*2+n}]:0.f;')
            lines += ['    }', '    __syncthreads();', '#pragma unroll 1', f"""    for (int j = threadIdx.x; j < {d2}; j += {threads}) {{"""]
            for output_index, out in enumerate(y_outputs):
                lines += ['// Limit live values in the sparse CG epilogue.', f"""        {{
            const volatile float *partial = intermediate.h[{output_index}];
            float value = 0.f;
            switch (j) {{"""]
                for j in range(d2):
                    lines.append(f'                case {j}: {{')
                    for i, jj, k, coefficient in spec['terms']:
                        if jj == j:
                            lines.append(f'                    value=fmaf({float_literal(coefficient)},partial[{i*p3+k}],value);')
                    lines.append('                    break; }')
                lines.append(f'            }}atomicAdd({out}+int64_t(item)*{ydim}+{yo}+int64_t(v)*{d2}+j,value);}}')
            lines.append('    }')
        lines += ['    return;', '}']
        cases.append('\n'.join(lines))
    parameters = [f'const float* __restrict__ {n}' for n in inputs] + [f'float* __restrict__ {n}' for n in outputs]
    aliases = []
    for alias, layout_a, layout_b in [('RowMma', 'RowMajor', 'RowMajor'), ('ColMma', 'RowMajor', 'ColumnMajor'), ('TransposedMma', 'ColumnMajor', 'RowMajor')]:
        aliases.append(f'using {alias}=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<2,2,2>,float,cutlass::layout::{layout_a},float,cutlass::layout::{layout_b},float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;')
    if w_outputs:
        aliases.append('using WeightMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<2,4,2>,float,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;')
    code = f'''__global__ __launch_bounds__({threads}, {UVW_EXTERNAL_BLOCKS_PER_SM}) void {name}({','.join(parameters)}) {{
    Index item = blockIdx.x / {max_tiles};
    int tile = blockIdx.x % {max_tiles};
{indent(chr(10).join(aliases), '    ')}
{indent(chr(10).join(storage), '    ')}
{indent(chr(10).join(storage_aliases), '    ')}
    switch (blockIdx.y) {{
{indent(chr(10).join(cases), "    ")}
    }}
}}
'''
    return code, f'dim3(items*{max_tiles},{len(specs)})', dynamic_bytes


def make_uvw_backward_external_kernel(meta, program, inputs, outputs, kernel_name):
    """Emit a path-major kernel; eight lanes cooperate on each path channel."""
    specs = meta["generated_specs"]
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    w_outputs = {k: v for k, v in program["w"].items() if k in outputs}
    needed_z = {
        z for pairs in (*x_outputs.values(), *y_outputs.values()) for _, z in pairs
    }
    contractions = {k: v for k, v in program["z"].items() if k in needed_z}
    needed_q = {q for pairs in w_outputs.values() for q, _ in pairs}
    products = {k: v for k, v in program["cg"].items() if k in needed_q}
    weights = sorted({w for pairs in contractions.values() for w, _ in pairs})
    gradients = sorted(
        {g for pairs in (*contractions.values(), *w_outputs.values()) for _, g in pairs}
    )
    max_edge = max(s["d2"] * s["edge_channels"] for s in specs)
    max_d3 = max(s["d3"] for s in specs)
    xdim, ydim, odim, wdim = (
        meta[k] for k in ("in1_dim", "in2_dim", "out_dim", "weight_numel")
    )
    cases = []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        edges, channels = spec["edge_channels"], spec["out_channels"]
        edge_size = edges * d2
        # Large third-order contractions otherwise keep Z, all CG epilogue
        # inputs and both edge accumulators live together. Shared memory bounds
        # that register lifetime without materializing a global intermediate.
        stage_z = len(contractions) * d3 > 24
        lines = [f"case {index}: {{"]
        lines += [f"    float sums_{target}[{edge_size}]={{0.f}};" for target in y_outputs]
        lines += [
            rf"""#pragma unroll 1
    for (int p = path_group; p < {spec['path_channels']}; p += 16) {{
        int u = p / {edges}, v = p % {edges};
        const unsigned mask = {'0xffffffffu' if spec['path_channels'] % 4 == 0 else '__activemask()'};"""
        ]
        if stage_z:
            lines.append("        {")
        lines.append(rf"""        using ZMma = cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1, {d3}, 1>, float,
                                                cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
                                                cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;""")
        for name in contractions:
            lines += [f"""        ZMma::FragmentC contract_{name};
        contract_{name}.clear();"""]
        if products:
            lines.append(rf"""        using WMma = cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1, 1, {d3}>, float,
                                                cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
                                                cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;""")
        for name, pairs in products.items():
            lines += [f"        WMma::FragmentA product_{name};", "        {"]
            lines += [f"            float z{k}=0.f;" for k in range(d3)]
            lines.append("            if(lane==0) {")
            # Reuse the CG accumulator while releasing each pair of inputs.
            for pair in pairs:
                lines += ["                {", indent(emit_cg_source(meta, spec, pairs=(pair,)), '                    '), "                }"]
            lines.append("            }")
            lines += [
                f"            product_{name}[{k}]=__shfl_sync(mask,z{k},0,8);" for k in range(d3)
            ]
            lines.append("        }")
        lines += ["#pragma unroll 1", f"""        for (int c = lane; c < {channels}; c += 8) {{"""]
        for name in weights:
            lines.append(
                f"""            ZMma::FragmentA weight_{name};
            weight_{name}[0] = __ldg({name} + int64_t(item) * {wdim} +
                                                {wo} + p * {channels} + c);"""
            )
        for name in gradients:
            lines += [f"            ZMma::FragmentB grad_{name};"]
            lines += [
                f"            grad_{name}[{k}]=__ldg({name}+int64_t(item)*{odim}+{oo}+c*{d3}+{k});"
                for k in range(d3)
            ]
        for target, pairs in contractions.items():
            for w, g in pairs:
                lines.append(
                    f"            ZMma()(contract_{target},weight_{w},grad_{g},contract_{target});"
                )
        for target, pairs in w_outputs.items():
            products_source = "\n".join(
                f"WMma()(delta_{target},product_{q},grad_{g},delta_{target});"
                for q, g in pairs
            )
            lines.append(
                rf"""            WMma::FragmentC delta_{target};
            delta_{target}.clear();
{indent(products_source, '            ')}
            {target}[int64_t(item) * {wdim} + {wo} + p * {channels} + c] =
                delta_{target}[0];"""
            )
        lines.append("        }")
        if contractions:
            lines += ["#pragma unroll", """        for (int delta = 4; delta > 0; delta >>= 1) {"""]
            for name in contractions:
                lines += [
                    f"            contract_{name}[{k}]+=__shfl_down_sync(mask,contract_{name}[{k}],delta,8);"
                    for k in range(d3)
                ]
            lines.append("        }")
        # Release high-order Z before loading the CG epilogue inputs.
        z_values = {}
        for z_index, name in enumerate(contractions):
            for k in range(d3):
                if stage_z:
                    address = f"path_group*{len(contractions) * d3}+{z_index * d3 + k}"
                    lines += [f"        if(lane==0) reduced_z[{address}]=contract_{name}[{k}];"]
                    z_values[name, k] = f"reduced_z[{address}]"
                else:
                    z_values[name, k] = f"contract_{name}[{k}]"
        if stage_z:
            lines.append("    }")
        if x_outputs or y_outputs:
            lines.append("        if(lane==0) {")
            # A single contraction can feed X and Y together, overlapping
            # their loads and arithmetic. Higher orders release one output's
            # inputs before starting the next to bound register lifetime.
            roles = [(target, pairs, "x") for target, pairs in x_outputs.items()]
            roles += [(target, pairs, "y") for target, pairs in y_outputs.items()]
            groups = [roles] if len(contractions) == 1 else [[role] for role in roles]
            for group in groups:
                lines.append("            {")
                for target, pairs, axis in sorted(
                    group, key=itemgetter(2), reverse=True
                ):
                    source_axis = 1 if axis == "x" else 0
                    dim, offset, width, channel = (
                        (ydim, yo, d2, "v") if axis == "x" else (xdim, xo, d1, "u")
                    )
                    for name in sorted({name for name, _ in pairs}):
                        for component in sorted(
                            {t[source_axis] for t in spec["terms"]}
                        ):
                            lines += [
                                f"                float value_{name}_{component}=__ldg({name}+int64_t(item)*{dim}+{offset}+{channel}*{width}+{component});"
                            ]
                    prefix = target if len(group) > 1 else "acc"
                    for component in sorted(
                        {t[1 - source_axis] for t in spec["terms"]}
                    ):
                        lines += [f"                float {prefix}{component}=0.f;"]
                for i, j, k, coeff in spec["terms"]:
                    for target, pairs, axis in group:
                        component, source = (i, j) if axis == "x" else (j, i)
                        prefix = target if len(group) > 1 else "acc"
                        for name, z in pairs:
                            lines.append(
                                f"                {prefix}{component}=fmaf({float_literal(coeff)}*value_{name}_{source},{z_values[z, k]},{prefix}{component});"
                            )
                for target, pairs, axis in group:
                    prefix = target if len(group) > 1 else "acc"
                    for component in sorted(
                        {t[0 if axis == "x" else 1] for t in spec["terms"]}
                    ):
                        if axis == "x":
                            lines += [
                                f"                atomicAdd({target}+int64_t(item)*{xdim}+{xo}+u*{d1}+{component},{prefix}{component});"
                            ]
                        else:
                            lines += [
                                f"                sums_{target}[v*{d2}+{component}]+={prefix}{component};"
                            ]
                lines.append("            }")
            lines.append("        }")
        lines.append("    }")
        if y_outputs:
            partial_stores = "\n".join(
                f"edge_partial[{target_index * 16 * edge_size}+path_group*{edge_size}+j]=sums_{target}[j];"
                for target_index, target in enumerate(y_outputs)
            )
            reductions = "\n".join(
                rf"""{{
    float value = 0.f;
#pragma unroll
    for (int group = 0; group < 16; ++group)
        value += edge_partial[{target_index * 16 * edge_size} + group * {edge_size} + j];
    atomicAdd({target} + int64_t(item) * {ydim} + {yo} + j, value);
}}"""
                for target_index, target in enumerate(y_outputs)
            )
            lines.append(
                rf"""    if (lane == 0) {{
#pragma unroll
        for (int j = 0; j < {edge_size}; ++j) {{
{indent(partial_stores, '            ')}
        }}
    }}
    __syncthreads();
    for (int j = threadIdx.x; j < {edge_size}; j += 128) {{
{indent(reductions, '        ')}
    }}"""
            )
        lines += ["    return;", "}"]
        cases.append("\n".join(lines))

    parameters = [f"const float* __restrict__ {name}" for name in inputs]
    parameters += [f"float* __restrict__ {name}" for name in outputs]
    storage = []
    if y_outputs:
        storage.append(
            f"__shared__ float edge_partial[{16 * len(y_outputs) * max_edge}];"
        )
    if len(contractions) * max_d3 > 24:
        storage.append(
            f"__shared__ volatile float reduced_z[{16 * len(contractions) * max_d3}];"
        )
    # Large directional contractions need a larger per-thread register budget
    # to avoid spilling the Z fragments into local memory.
    resident_blocks = 2 if len(contractions) * max_d3 > 48 else 4
    return rf"""
__global__ __launch_bounds__(128, {resident_blocks}) void {kernel_name}({','.join(parameters)}) {{
    int lane = threadIdx.x & 7, path_group = threadIdx.x >> 3;
    Index item = blockIdx.x;
{indent(chr(10).join(storage), '    ')}
    switch (blockIdx.y) {{
{indent(chr(10).join(cases), "    ")}
    }}
}}
"""


def make_uvw_backward_external_source(meta, program, inputs, outputs, phase, *, arch):
    prefix = f"lunatp_uvw_{phase}_external"
    tiled = meta.get("dtype") is not torch.float64
    matrix_outputs = outputs if tiled else tuple(name for name in outputs if name != "gg")
    # Separate XY and weight work when their combined fragments would crowd registers.
    phases = (matrix_outputs,)
    if not tiled and (len(program["z"]) + len(program["cg"])) * max(s["d3"] for s in meta["generated_specs"]) > 64:
        phases = (
            tuple(k for k in matrix_outputs if k not in program["w"]),
            tuple(k for k in matrix_outputs if k in program["w"]),
        )
    kernels, launches, configuration = [], [], []
    for index, phase_outputs in enumerate(phases):
        if not phase_outputs:
            continue
        name = f"{prefix}_fused_{index}_fp32"
        shared_bytes = 0
        if tiled:
            kernel, grid, shared_bytes = make_uvw_backward_external_tiled_kernel(
                meta, program, inputs, phase_outputs, name
            )
        else:
            kernel = make_uvw_backward_external_kernel(
                meta, program, inputs, phase_outputs, name
            )
            grid = f"dim3(items,{len(meta['generated_specs'])})"
        kernels.append(kernel)
        args = ",".join(
            f"{name}.data_ptr<float>()" for name in (*inputs, *phase_outputs)
        )
        launches.append(
            f"""{name}<<<{grid},128,{shared_bytes},stream>>>({args});"""
        )
        if shared_bytes:
            configuration.append(
                f"C10_CUDA_CHECK(cudaFuncSetAttribute({name},cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_bytes}));"
            )
    if "gg" in outputs and not tiled:
        products = tuple((program["cg"][q], w) for q, w in program["forward"])
        name = f"{prefix}_g_fp32"
        kernel, grid, clear, pointer_order = make_uvw_forward_external_kernel(
            meta, products, name, arch=arch
        )
        kernels.insert(0, kernel)
        if clear:
            launches.append(
                "C10_CUDA_CHECK(cudaMemsetAsync(gg.data_ptr<float>(),0,gg.numel()*sizeof(float),stream));"
            )
        args = ",".join(f"{name}.data_ptr<float>()" for name in pointer_order)
        launches.append(
            f"{name}<<<{grid},128,0,stream>>>({args},gg.data_ptr<float>(),items);"
        )
    allocations = [
        f"auto {name}=at::empty_like({OUTPUT_SHAPES[name]});" for name in outputs
    ]
    clears = [
        f"C10_CUDA_CHECK(cudaMemsetAsync({name}.data_ptr<float>(),0,{name}.numel()*sizeof(float),stream));"
        for name in outputs
        if name in (*program["x"], *program["y"]) or (tiled and name == "gg")
    ]
    configure_source = ""
    binding = 'm.def("launch",&launch);'
    if configuration:
        configure_source = f'''void configure(at::Tensor x) {{
    c10::cuda::CUDAGuard guard(x.device());
{indent(chr(10).join(configuration), "    ")}
}}
'''
        binding += 'm.def("configure",&configure);'
    source = rf"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>
{chr(10).join(kernels)}
std::vector<at::Tensor> launch({','.join((f'at::Tensor {name}' for name in inputs))}) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items = x.size(0);
{indent(chr(10).join(allocations), "    ")}
    if (items == 0)
        return {{{','.join(outputs)}}};
    auto stream = at::cuda::getCurrentCUDAStream();
{indent(chr(10).join(clears), "    ")}
{indent(chr(10).join(launches), "        ")}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{','.join(outputs)}}};
}}
{configure_source}PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
{indent(binding, '    ')}
}}
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def make_uvw_scalar_dxdy_kernel(meta, program, inputs, outputs, name):
    """Reuse a shared gW tile for both input gradients without global storage."""
    m,n,k=64,128,8
    xdim,ydim,odim=(meta[key] for key in ("in1_dim","in2_dim","out_dim"))
    source=f"""
#include <cutlass/gemm/device/gemm.h>
using {name}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{m}, {n}, {k}>,
    cutlass::gemm::GemmShape<{m//2}, {n//2}, {k}>,
    cutlass::gemm::GemmShape<1, 1, 1>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel::Mma;
using {name}_MapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<{k}, {m}>, 128, 4>;
using {name}_MapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<{k}, {n}>, 128, 4>;
template <int D, int O, bool Aligned> struct {name}_IteratorA {{
    using Element = float;
    using ThreadMap = {name}_MapA;
    using Fragment = cutlass::Array<float, {m*k//128}>;
    const float *pointer;
    Index offsets[ThreadMap::Iterations::kStrided];
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_IteratorA(const float *p, Index first, Index items, int thread)
        : pointer(p), offset(ThreadMap::initial_offset(thread)) {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            int row = offset.strided() + s * ThreadMap::Delta::kStrided;
            Index item = first + row / D;
            offsets[s] =
                row < ({m} / D) * D && item < items ? Index(item) * {odim} + row % D : -1;
        }}
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorA &operator++() {{
        offset.contiguous() += {k};
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &f) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int col = offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                int index = (s * ThreadMap::Iterations::kContiguous + c) * 4;
                if constexpr (Aligned) {{
                    float4 v = make_float4(0, 0, 0, 0);
                    if (enabled && offsets[s] >= 0 && col < O)
                        v = *reinterpret_cast<const float4 *>(pointer + offsets[s] + col);
                    f[index] = v.x;
                    f[index + 1] = v.y;
                    f[index + 2] = v.z;
                    f[index + 3] = v.w;
                }} else {{
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        f[index + j] =
                            enabled && offsets[s] >= 0 && col + j < O ? pointer[offsets[s] + (col + j) * D] : 0.f;
                }}
            }}
        }}
    }}
}};
template <int U, int V, int O, bool Aligned> struct {name}_IteratorB {{
    using Element = float;
    using ThreadMap = {name}_MapB;
    using Fragment = cutlass::Array<float, {n*k//128}>;
    const float *pointer;
    int first_u, first_v;
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    static constexpr int TileV = V < 8 ? (V <= 1 ? 1 : V <= 2 ? 2 : V <= 4 ? 4 : 8) : 8;
    CUTLASS_DEVICE {name}_IteratorB(const float *p, int u, int v, int thread)
        : pointer(p), first_u(u), first_v(v), offset(ThreadMap::initial_offset(thread)) {{
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorB &operator++() {{
        offset.contiguous() += {k};
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &f) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            int local = offset.strided() + s * ThreadMap::Delta::kStrided;
            int u = first_u + local / TileV, v = first_v + local % TileV;
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                int col = offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                int index = (s * ThreadMap::Iterations::kContiguous + c) * 4;
                if constexpr (Aligned) {{
                    float4 value = make_float4(0, 0, 0, 0);
                    if (enabled && u < U && v < V && col < O)
                        value = *reinterpret_cast<const float4 *>(pointer + (Index(u) * V + v) * O + col);
                    f[index] = value.x;
                    f[index + 1] = value.y;
                    f[index + 2] = value.z;
                    f[index + 3] = value.w;
                }} else {{
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        f[index + j] =
                            enabled && u < U && v < V && col + j < O ? pointer[(Index(u) * V + v) * O + col + j] : 0.f;
                }}
            }}
        }}
    }}
}};
template <bool IsA> struct {name}_Shared {{
    using Map = typename cutlass::platform::conditional<IsA, {name}_MapA, {name}_MapB>::type;
    using Element = float;
    using Layout =
        typename cutlass::platform::conditional<IsA, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using TensorRef = cutlass::TensorRef<float, Layout>;
    using Fragment = cutlass::Array<float, (IsA ? {m*k//128} : {n*k//128})>;
    TensorRef ref;
    int outer, inner;
    CUTLASS_DEVICE {name}_Shared(TensorRef r, int thread) : ref(r) {{
        auto p = Map::initial_offset(thread);
        outer = p.strided();
        inner = p.contiguous();
    }}
    CUTLASS_DEVICE void store(Fragment const &f) {{
#pragma unroll
        for (int s = 0; s < Map::Iterations::kStrided; ++s)
#pragma unroll
            for (int c = 0; c < Map::Iterations::kContiguous; ++c)
#pragma unroll
                for (int v = 0; v < 4; ++v) {{
                    int o = outer + s * Map::Delta::kStrided, i = inner + c * Map::Delta::kContiguous + v;
                    ref.at(IsA ? cutlass::MatrixCoord(o, i) : cutlass::MatrixCoord(i, o)) =
                        f[(s * Map::Iterations::kContiguous + c) * 4 + v];
                }}
    }}
    CUTLASS_DEVICE {name}_Shared &operator++() {{
        inner += {k};
        return *this;
    }}
    CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const &p) {{
        outer += IsA ? p.row() * {m} : p.column() * {n};
        inner += (IsA ? p.column() : p.row()) * {k};
    }}
}};
template <int D, int U, int V, int O, bool AAligned, bool BAligned>
using {name}_Mma = cutlass::gemm::threadblock::MmaPipelined<
    typename {name}_Default::Shape, {name}_IteratorA<D, O, AAligned>,
    {name}_Shared<true>, {name}_IteratorB<U, V, O, BAligned>, {name}_Shared<false>,
    float, cutlass::layout::RowMajor, typename {name}_Default::Policy>;
"""
    params=[f'const float* __restrict__ {v}' for v in inputs]+[f'float* __restrict__ {v}' for v in outputs]+['Index items']
    source+=f"""__global__ __launch_bounds__(128) void {name}({','.join(params)}) {{
    union Storage {{
        typename {name}_Default::SharedStorage mma;
        float z[{m*n}];
    }};
    __shared__ Storage storage;
    Index task = blockIdx.x;
"""
    grids=[]
    for spec in meta['generated_specs']:
        u,v,o,d=(spec[key] for key in ('in_channels','edge_channels','out_channels','d1'))
        xo,yo,oo,wo=(spec[key] for key in ('x_offset','y_offset','out_offset','weight_offset'))
        a_aligned=str(d==1 and odim%4==0 and oo%4==0 and o%4==0).lower();b_aligned=str(o%4==0 and wo%4==0).lower()
        tile_items=m//d;tile_v=min(8,1<<(v-1).bit_length());tile_u=n//tile_v;v_tiles=(v+tile_v-1)//tile_v;column_tiles=((u+tile_u-1)//tile_u)*v_tiles
        grid=f'((items+{tile_items-1})/{tile_items})*{column_tiles}'
        grids.append(grid)
        source+=f"""    {{
        Index count = {grid};
        if (task < count) {{
            Index first_item = (task / {column_tiles}) * {tile_items};
            int first_u = ((task % {column_tiles}) / {v_tiles}) * {tile_u},
                first_v = (task % {v_tiles}) * {tile_v};
            using Mma = {name}_Mma<{d}, {u}, {v},
                                              {o}, {a_aligned}, {b_aligned}>;
"""
        for z,pairs in program['z'].items():
            xp={out:tuple(src for src,zz in pp if zz==z) for out,pp in program['x'].items() if out in outputs}
            yp={out:tuple(src for src,zz in pp if zz==z) for out,pp in program['y'].items() if out in outputs}
            if not any(xp.values()) and not any(yp.values()):continue
            source+="""            {
                typename Mma::FragmentC acc;
                acc.clear();
"""
            for w,g in pairs:
                source+=f"""                {{
                    {name}_IteratorA<{d}, {o}, {a_aligned}> a(
                        {g} + {oo}, first_item, items, threadIdx.x);
                    {name}_IteratorB<{u}, {v}, {o},
                                                {b_aligned}>
                        b({w} + {wo}, first_u, first_v, threadIdx.x);
                    Mma mma(storage.mma, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
                    mma(({o} + {k-1}) / {k}, acc, a, b, acc);
                    __syncthreads();
                }}
"""
            source+=f"""                typename Mma::Operator::IteratorC c({{storage.z, {n}}}, threadIdx.x % 32);
                c.add_tile_offset({{int(threadIdx.x / 32) % 2, int(threadIdx.x / 32) / 2}});
                c.store(acc);
                __syncthreads();
            }}
"""
            scale=float_literal(spec['terms'][0][3],torch.float32)
            for out,sources in xp.items():
                if not sources:continue
                source+=f"""            for (int index = threadIdx.x; index < {tile_items*tile_u*d}; index += 128) {{
                int component = index % {d}, local_u = (index / {d}) % {tile_u},
                    local_item = index / {tile_u*d};
                Index item = first_item + local_item;
                int u = first_u + local_u;
                float value = 0.f;
                if (item < items && u < {u}) {{
"""
                source+=f"""#pragma unroll
                    for (int v = 0; v < {tile_v}; ++v)
                        if (first_v + v < {v}) {{
"""
                val='+'.join(f'{src}[Index(item)*{ydim}+{yo}+first_v+v]' for src in sources)
                source+=f"""                            value = fmaf(({val}),
                                         storage.z[(local_item * {d} + component) * {n} +
                                                   local_u * {tile_v} + v],
                                         value);
                        }}
"""
                source+=f"""                    atomicAdd({out} + Index(item) * {xdim} + {xo} +
                                  u * {d} + component,
                              value * {scale});
                }}
            }}
"""
            for out,sources in yp.items():
                if not sources:continue
                source+=f"""            for (int index = threadIdx.x; index < {tile_items*tile_v}; index += 128) {{
                int local_v = index % {tile_v}, local_item = index / {tile_v};
                Index item = first_item + local_item;
                int v = first_v + local_v;
                float value = 0.f;
                if (item < items && v < {v}) {{
"""
                source+=f"""#pragma unroll
                    for (int u = 0; u < {tile_u}; ++u)
                        if (first_u + u < {u}) {{
"""
                for comp in range(d):
                    val='+'.join(f'{src}[Index(item)*{xdim}+{xo}+(first_u+u)*{d}+{comp}]' for src in sources)
                    source+=f"""                            value =
                                fmaf(({val}),
                                     storage.z[(local_item * {d} + {comp}) * {n} +
                                               u * {tile_v} + local_v],
                                     value);
"""
                source+=f"""                        }}
                    atomicAdd({out} + Index(item) * {ydim} + {yo} + v,
                              value * {scale});
                }}
            }}
"""
            source+="""            __syncthreads();
"""
        source+="""            return;
        }
        task -= count;
    }
"""
    source+='}\n'
    return source,'+'.join(grids),128

def make_uvw_scalar_dx_kernel(meta, program, inputs, outputs, name):
    """Contract yW tiles with output gradients directly into input gradients."""
    prefix=r"""
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
namespace {
using Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel;
using BaseMma = typename Default::Mma;
""" + SCALAR_WEIGHT_LOADER + r"""
template <int U, int V, int O, bool Aligned = true>
using WeightIterator = ScalarWeightLoader<BaseMma, U, V, O, Aligned>;
"""
    grids=[];dispatch=[]
    for spec in meta['generated_specs']:
        u,v,o,d=(spec[k] for k in ('in_channels','edge_channels','out_channels','d1'))
        xo,yo,oo,wo=(spec[k] for k in ('x_offset','y_offset','out_offset','weight_offset'))
        scale=float_literal(spec['terms'][0][3],torch.float32)
        aligned=str(wo%4==0).lower()
        grid=f'((items+63)/64)*{(u+3)//4}*{(o+31)//32}'
        for target,pairs in program['x'].items():
            if target not in outputs:continue
            grouped={}
            for feature,z in pairs:
                for weight,grad in program['z'][z]:grouped.setdefault(grad,[]).append((feature,weight))
            groups=list(grouped.items())
            max_terms=max(len(terms) for _,terms in groups)
            feature=[];weight=[]
            for term in range(max_terms):
                for operand, pointers in enumerate((feature, weight)):
                    values = [terms[min(term, len(terms)-1)][operand] for _, terms in groups]
                    pointers.append(''.join(f'group=={i}?{value}:' for i, value in enumerate(values[:-1])) + values[-1])
            term_counts = ''.join(f'group=={i}?{len(terms)}:' for i, (_, terms) in enumerate(groups[:-1])) + str(len(groups[-1][1]))
            grad_pointer = ''.join(f'group=={i}?{grad}:' for i, (grad, _) in enumerate(groups[:-1])) + groups[-1][0]
            mma_call=f"""{{
    const float *feature_ptr = {''.join(f'term=={i}?({value}):' for i,value in enumerate(feature[:-1]))}{feature[-1]};
    const float *weight_ptr = {''.join(f'term=={i}?({value}):' for i,value in enumerate(weight[:-1]))}{weight[-1]};
    typename BaseMma::IteratorA::Params params(cutlass::layout::RowMajor({meta['in2_dim']}));
    typename BaseMma::IteratorA a(params,
                                  const_cast<float *>(feature_ptr + first * {meta['in2_dim']} + {yo}),
                                  {{int(min(Index(64), items - first)), {v}}}, threadIdx.x, {{0, 0}});
    const float *weights[1] = {{weight_ptr + {wo}}};
    WeightIterator<{u}, {v}, {o}, {aligned}> b(weights, u0, o0,
                                                                                                    threadIdx.x);
    PathMma mma(storage, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
    mma(({v} + 7) / 8, acc, a, b, acc);
    __syncthreads();
}}"""
            compute=f"""
{{
    const int o0 = (task % {(o+31)//32}) * 32;
    typename BaseMma::FragmentC acc;
    acc.clear();
#pragma unroll 1
    for (int term = 0; term < term_count; ++term) {{
{indent(mma_call, '        ')}
    }}
#pragma unroll
    for (int m = 0; m < 8; ++m) {{
        const Index row = first + (threadIdx.x / 32 % 2) * 32 + (m / 4) * 16 + lane.row() * 4 + m % 4;
#pragma unroll
        for (int component = 0; component < {d}; ++component) {{
            float grad_values[4] = {{}};
#pragma unroll
            for (int c = 0; c < 4; ++c) {{
                const int out = o0 + lane.column() * 4 + c;
                if (row < items && out < {o})
                    grad_values[c] =
                        grad_ptr[row * {meta['out_dim']} + {oo} + out * {d} + component];
            }}
#pragma unroll
            for (int z = 0; z < 2; ++z) {{
                const int channel = u0 + (threadIdx.x / 32 / 2) * 2 + z;
                float value = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {{
                    const int out = o0 + lane.column() * 4 + c;
                    if (row < items && out < {o})
                        value = fmaf(acc[m * 8 + c + z * 4], grad_values[c], value);
                }}
#pragma unroll
                for (int delta = 4; delta > 0; delta /= 2)
                    value +=
                        __shfl_sync(0xffffffff, value,
                                    BaseMma::Operator::Policy::get_lane_layout()({{lane.row(), lane.column() ^ delta}}));
                if (lane.column() == 0 && row < items && channel < {u})
                    atomicAdd({target} + row * {meta['in1_dim']} + {xo} +
                                  channel * {d} + component,
                              value * {scale});
            }}
        }}
    }}
    __syncthreads();
}}
"""
            dispatch.append(f'''{{
    Index count = ({grid})*{len(groups)};
    if (task < count) {{
        const int group = task % {len(groups)};
        task /= {len(groups)};
        const int term_count = {term_counts};
        const float *grad_ptr = {grad_pointer};
        const Index first = (task / {((u+3)//4)*((o+31)//32)}) * 64;
        const int u0 = ((task / {(o+31)//32}) % {(u+3)//4}) * 4;
        using PathMma = cutlass::gemm::threadblock::MmaPipelined<
            typename BaseMma::Shape, typename BaseMma::IteratorA, typename BaseMma::SmemIteratorA,
            WeightIterator<{u}, {v}, {o}, {aligned}>,
            cutlass::transform::threadblock::RegularTileIterator<
                cutlass::MatrixShape<8, 128>, float, cutlass::layout::RowMajor, 0,
                typename WeightIterator<{u}, {v}, {o},
                                        {aligned}>::ThreadMap>,
            float, cutlass::layout::RowMajor, typename BaseMma::Policy>;
{indent(compute, '        ')}
        return;
    }}
    task -= count;
}}
''');grids.append(f'({grid})*{len(groups)}')
    params=[f'const float* __restrict__ {p}' for p in inputs]+[f'float* __restrict__ {p}' for p in outputs]+['Index items']
    source=prefix+f'''    __global__ __launch_bounds__(128, 4) void {name}({','.join(params)}) {{
        __shared__ typename BaseMma::SharedStorage storage;
        Index task = blockIdx.x;
        auto lane = BaseMma::Operator::Policy::get_lane_layout().inverse(threadIdx.x % 32);
{indent(chr(10).join(dispatch), '        ')}
    }}
}}
'''
    for symbol in ('Default','BaseMma','WeightIterator'):
        source=source.replace(symbol,name+'_'+symbol)
    return source,'+'.join(grids),128


def make_uvw_scalar_dy_kernel(meta, program, inputs, outputs, name):
    """Multiply implicit angular contractions by weights without a global workspace."""
    threads = UVW_THREADS
    width = max(spec["edge_channels"] for spec in meta["generated_specs"])
    tile_n = min(UVW_DY_MAX_TILE_DIM, max(UVW_DY_MIN_TILE_COLUMNS, 1 << (width - 1).bit_length()))
    tile_m = min(UVW_DY_MAX_TILE_DIM, UVW_GEMM_TILE_ELEMENTS * threads // UVW_GEMM_THREADS // tile_n)
    tile_k = UVW_DY_TILE_K
    access = min(4, tile_n * tile_k // threads)
    # Include the split roundup and the last masked pipeline advances.
    wide_k = max(s["in_channels"] * s["out_channels"] for s in meta["generated_specs"]) + 16 * tile_k > (1 << 31) - 1
    k_type = "int64_t" if wide_k else "int"
    coordinate_type = "cutlass::Coord<2,int64_t>" if wide_k else "cutlass::PitchLinearCoord"
    coordinate_k = "offset[0]" if wide_k else "offset.contiguous()"
    coordinate_row = "offset[1]" if wide_k else "offset.strided()"
    vector_store = ''.join(f'fragment[index{f"+{i}" if i else ""}]=value.{field};' for i, field in enumerate('xyzw'[:access]))
    xdim, ydim, odim = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    source = f"""
#include <cutlass/gemm/device/gemm.h>
using {name}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tile_k}>,
    cutlass::gemm::GemmShape<{tile_m//2}, {tile_n//2}, {tile_k}>,
    cutlass::gemm::GemmShape<1, 1, 1>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel::Mma;
using {name}_MapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<{tile_k}, {tile_n}>, {threads}, {access}>;
template <int U, int V, int O, int D, int Count> struct {name}_IteratorA {{
    using Element = float;
    using ThreadMap = typename {name}_Default::IteratorA::ThreadMap;
    using Fragment = typename {name}_Default::IteratorA::Fragment;
    const float *feature[Count];
    const float *grad[Count];
    Index item[ThreadMap::Iterations::kStrided];
    {coordinate_type} offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_IteratorA(const float *const *f, const float *const *g, int64_t first,
                                                int64_t items, int thread)
        : offset(ThreadMap::initial_offset(thread)) {{
#pragma unroll
        for (int t = 0; t < Count; ++t) {{
            feature[t] = f[t];
            grad[t] = g[t];
        }}
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            int64_t row = first + {coordinate_row} + s * ThreadMap::Delta::kStrided;
            item[s] = row;
            if (item[s] >= items)
                item[s] = -1;
        }}
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorA &operator++() {{
        {coordinate_k} += {tile_k};
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                {k_type} k = {coordinate_k} + c * ThreadMap::Delta::kContiguous;
                float value = 0.f;
                if (enabled && item[s] >= 0 && k < (U * O)) {{
                    {{
#pragma unroll
                        for (int t = 0; t < Count; ++t)
#pragma unroll
                            for (int d = 0; d < D; ++d)
                                value = fmaf(feature[t][item[s] * {xdim} + (k / O) * D + d],
                                             grad[t][item[s] * {odim} + (k % O) * D + d], value);
                    }}
                }}
                fragment[s * ThreadMap::Iterations::kContiguous + c] = value;
            }}
        }}
    }}
}};
template <int U, int V, int O, int D, int64_t Offset> struct {name}_IteratorB {{
    using Element = float;
    using ThreadMap = {name}_MapB;
    using Fragment = cutlass::Array<float, {tile_n*tile_k//threads}>;
    const float *weight;
    int first;
    {coordinate_type} offset;
    bool enabled = true;
    CUTLASS_DEVICE {name}_IteratorB(const float *p, int column, int thread)
        : weight(p), first(column), offset(ThreadMap::initial_offset(thread)) {{
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {name}_IteratorB &operator++() {{
        {coordinate_k} += {tile_k};
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
#pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
            int column = first + {coordinate_row} + s * ThreadMap::Delta::kStrided;
#pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                {k_type} k = {coordinate_k} + c * ThreadMap::Delta::kContiguous;
                int index = (s * ThreadMap::Iterations::kContiguous + c) * {access};
                if constexpr (O % {access} == 0 && Offset % {access} == 0) {{
                    float{access} value = make_float{access}({','.join('0' for _ in range(access))});
                    if (enabled && column < V && k < (U * O)) {{
                        int64_t address = (int64_t(k / O) * V + column) * O + k % O;
                        value = *reinterpret_cast<const float{access} *>(weight + address);
                    }}
{indent(vector_store, '                    ')}
                }} else {{
#pragma unroll
                    for (int j = 0; j < {access}; ++j) {{
                        {k_type} kk = k + j;
                        float value = 0.f;
                        if (enabled && column < V && kk < (U * O)) {{
                            value = weight[(int64_t(kk / O) * V + column) * O + kk % O];
                        }}
                        fragment[index + j] = value;
                    }}
                }}
            }}
        }}
    }}
}};
struct {name}_Shared {{
    using Map = {name}_MapB;
    using Element = float;
    using Layout = cutlass::layout::RowMajor;
    using TensorRef = cutlass::TensorRef<float, Layout>;
    using Fragment = cutlass::Array<float, {tile_n*tile_k//threads}>;
    TensorRef ref;
    int outer, inner;
    CUTLASS_DEVICE {name}_Shared(TensorRef r, int thread) : ref(r) {{
        auto p = Map::initial_offset(thread);
        outer = p.strided();
        inner = p.contiguous();
    }}
    CUTLASS_DEVICE void store(Fragment const &f) {{
#pragma unroll
        for (int s = 0; s < Map::Iterations::kStrided; ++s)
#pragma unroll
            for (int c = 0; c < Map::Iterations::kContiguous; ++c)
#pragma unroll
                for (int v = 0; v < {access}; ++v) {{
                    int o = outer + s * Map::Delta::kStrided, i = inner + c * Map::Delta::kContiguous + v;
                    ref.at(cutlass::MatrixCoord(i, o)) =
                        f[(s * Map::Iterations::kContiguous + c) * {access} + v];
                }}
    }}
    CUTLASS_DEVICE {name}_Shared &operator++() {{
        inner += {tile_k};
        return *this;
    }}
    CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const &p) {{
        outer += p.column() * {tile_n};
        inner += p.row() * {tile_k};
    }}
}};
template <int U, int V, int O, int D, int64_t Offset, int Count>
using {name}_Mma = cutlass::gemm::threadblock::MmaPipelined<
    typename {name}_Default::Shape, {name}_IteratorA<U, V, O, D, Count>,
    typename {name}_Default::SmemIteratorA, {name}_IteratorB<U, V, O, D, Offset>,
    {name}_Shared, float, cutlass::layout::RowMajor, typename {name}_Default::Policy>;
"""
    params = [f"const float* __restrict__ {v}" for v in inputs]
    params += [f"float* __restrict__ {v}" for v in outputs] + ["Index items"]
    source += f"""
__global__ __launch_bounds__({threads}) void {name}({','.join(params)}) {{
"""
    source += f"""    __shared__ typename {name}_Default::SharedStorage storage;
    __shared__ float result[{tile_m} * {tile_n}];
    Index task = blockIdx.x;
"""
    grids=[]
    for spec in meta["generated_specs"]:
        u,v,o,d=(spec[k] for k in ("in_channels","edge_channels","out_channels","d1"))
        xo,yo,oo,wo=(spec[k] for k in ("x_offset","y_offset","out_offset","weight_offset"))
        scale=float_literal(spec["terms"][0][3],torch.float32)
        columns=v
        rows=1
        reduce=u*o
        column_tiles=(columns+tile_n-1)//tile_n
        base_grid=f"(((items*{rows}+{tile_m-1})/{tile_m})*{column_tiles})"
        parts=f"std::min(Index({min(16,max(1,reduce//128))}),std::max(Index(1),(Index({UVW_TARGET_BLOCKS})+{base_grid}-1)/{base_grid}))"
        grid=f"({base_grid}*({parts}))"
        for target,pairs in program["y"].items():
            if target not in outputs:continue
            grids.append(grid)
            reduction = f"int64_t({reduce})" if wide_k else str(reduce)
            source+=f"""    {{
        Index count = {grid};
        if (task < count) {{
            const Index parts = {parts}, tile = task / parts;
            const int part = task % parts;
            const {k_type} slice =
                (({reduction} + {tile_k} * parts - 1) / ({tile_k} * parts)) * {tile_k};
            const {k_type} first_k = part * slice;
            const Index first_row = (tile / {column_tiles}) * {tile_m};
            const int first_column = (tile % {column_tiles}) * {tile_n};
"""
            grouped={}
            for feature,z in pairs:
                for weight,grad in program['z'][z]:
                    grouped.setdefault(weight, []).append((feature, grad))
            source+=f"""            typename {name}_Default::FragmentC acc;
            acc.clear();
"""
            for weight, terms in grouped.items():
                features=','.join(f'{f}+{xo}' for f,g in terms)
                gradients=','.join(f'{g}+{oo}' for f,g in terms)
                source+=f"""            {{
                using Mma = {name}_Mma<{u}, {v}, {o},
                                                  {d}, {wo}, {len(terms)}>;
                const float *features[] = {{{features}}};
                const float *gradients[] = {{{gradients}}};
                {name}_IteratorA<{u}, {v}, {o},
                                            {d}, {len(terms)}>
                    a(features, gradients, first_row, items, threadIdx.x);
                {name}_IteratorB<{u}, {v}, {o},
                                             {d}, {wo}>
                    b({weight} + {wo}, first_column, threadIdx.x);
                a.{coordinate_k} += first_k;
                b.{coordinate_k} += first_k;
                Mma mma(storage, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
                mma((min(slice, {reduce} - first_k) + {tile_k-1}) / {tile_k}, acc, a, b,
                    acc);
                __syncthreads();
            }}
"""
            source+=f"""            typename {name}_Default::Operator::IteratorC c({{result, {tile_n}}}, threadIdx.x % 32);
            c.add_tile_offset({{int(threadIdx.x / 32) % 2, int(threadIdx.x / 32) / 2}});
            c.store(acc);
            __syncthreads();
            for (int index = threadIdx.x; index < {tile_m*tile_n}; index += {threads}) {{
                Index row = first_row + index / {tile_n};
                int column = first_column + index % {tile_n};
                if (row < items * {rows} && column < {columns})
                    atomicAdd({target} + {yo} + row * {ydim} + column,
                              result[index] * {scale});
            }}
            return;
        }}
        task -= count;
    }}
"""
    source+="}\n"
    return source,'+'.join(grids),threads


def make_uvw_scalar_dw_kernel(meta, program, inputs, outputs, name):
    """Generate CG products inside CUTLASS; accumulate directly into weights."""
    threads = UVW_GEMM_THREADS
    xd, yd, gd = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    kernels, grids, tile_counts, shared_types, calls = [], [], [], [], []
    tile_work = []
    ptrs = ','.join(f'const float* __restrict__ {a}' for a in inputs)
    for path, spec in enumerate(meta["generated_specs"]):
        u, v, o, d = (spec[k] for k in ("in_channels", "edge_channels", "out_channels", "d1"))
        xo, yo, go, wo = (spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset"))
        scale = float_literal(spec["terms"][0][3], torch.float32)
        cg_terms, angular_terms, grouped_terms = [], [], []
        for target in outputs:
            if target not in program['w']: continue
            by_feature, by_scalar, by_gradient = {}, {}, {}
            for q,g in program['w'][target]:
                by_gradient.setdefault(g,[]).extend(program['cg'][q])
                for x,y in program['cg'][q]:
                    by_feature.setdefault(x,[]).append((y,g))
                    by_scalar.setdefault(y,[]).append((x,g))
            grouped_terms.extend((target,g,pairs) for g,pairs in by_gradient.items())
            cg_terms.extend((target,x,pairs) for x,pairs in by_feature.items())
            angular_terms.extend((target,y,pairs) for y,pairs in by_scalar.items())
        pair_count=sum(len(pairs) for _,_,pairs in cg_terms)
        # Fuse repeated output gradients before GEMM when that reduces MMA count.
        product_on_left = len(grouped_terms) < len(cg_terms)
        if product_on_left:
            cg_terms = grouped_terms
        cg_height, cg_width = (u*v, o) if product_on_left else (u, v*o)
        cg_rows=min(128,max(32,1<<(cg_height-1).bit_length()))
        cg_columns=64 if cg_width<=64 else 128
        angular_rows=min(128,max(32,1<<(v-1).bit_length()))
        angular_columns=64 if u*o<=64 else 128
        cg_padded=((cg_height+cg_rows-1)//cg_rows)*cg_rows*((cg_width+cg_columns-1)//cg_columns)*cg_columns
        angular_padded=((v+angular_rows-1)//angular_rows)*angular_rows*((u*o+angular_columns-1)//angular_columns)*angular_columns
        cg_products=u*v*((o+cg_columns-1)//cg_columns) if product_on_left else v*o*((u+cg_rows-1)//cg_rows)
        cg_cost=d*(cg_padded*len(cg_terms)+cg_products*pair_count)
        angular_cost=angular_padded*len(angular_terms)+d*u*o*pair_count*((v+angular_rows-1)//angular_rows)
        contract_components=angular_cost<cg_cost
        terms=angular_terms if contract_components else cg_terms
        product_on_left = product_on_left and not contract_components
        height,width,components=(v,u*o,1) if contract_components else (cg_height,cg_width,d)
        rows=min(128,max(32,1<<(height-1).bit_length()))
        columns=64 if width<=64 else 128
        access = columns * 8 // threads
        a_access = rows * 8 // threads
        if contract_components or d != 1 or xd % a_access or xo % a_access or u % a_access:
            a_access = 1
        fields = "xyzw"[:access]
        tiles=((height+rows-1)//rows)*((width+columns-1)//columns)
        tile_counts.append(tiles)
        cg_values = (rows if product_on_left else columns) * max(len(pairs) for _, _, pairs in terms)
        tile_work.append(components * (rows * columns + cg_values * (d if contract_components else 1)))
        splits=f'{name}_split_capacity'
        grids.append(f"{tiles*len(terms)}*min(int64_t({splits}),(int64_t(items)*{components}+127)/128)")
        n=f'{name}_path_{path}'
        shared_types.append(f'typename {n}_Mma::SharedStorage path_{path};')
        calls.append(f"""if (task < {grids[-1]}) {{
    {n}("""+','.join((*inputs,*outputs))+""",items,task,&shared);
    return;
}
task -="""+grids[-1]+';')
        a_bodies,b_bodies,vector_bodies,a_vector_bodies=[],[],[],[]
        a_arguments,b_arguments=[],[]
        for _,multiplier,pairs in terms:
            a_values=[value for pair in pairs for value in pair] if product_on_left else [multiplier]
            b_values=[multiplier] if product_on_left else [value for pair in pairs for value in pair]
            a_arguments.append(','.join(a_values+['nullptr']*(4-len(a_values))))
            b_arguments.append(','.join(b_values+['nullptr']*(4-len(b_values))))
            multiplier='operand0'
            pairs=[(f'operand{2*i}',f'operand{2*i+1}') for i in range(len(pairs))]
            if contract_components:
                left=f'value={multiplier}[item*{yd}+{yo}+column];'
                right=''.join(f'value=fmaf({x}[item*{xd}+{xo}+(column/{o})*{d}+component],{g}[item*{gd}+{go}+(column%{o})*{d}+component],value);' for x,g in pairs)
                right=f'\n#pragma unroll\nfor(int component=0;component<{d};++component){{{right}}}'
                vector=''.join(f'{{float left={x}[item*{xd}];float{access} right=*reinterpret_cast<const float{access}*>({g}+item*{gd});'+''.join(f'value[{j}]=fmaf(left,right.{field},value[{j}]);' for j,field in enumerate(fields))+'}' for x,g in pairs)
                if d != 1 or gd%4 or go%4:
                    vector=''.join(f'{{float left={x}[item*{xd}+component];\n#pragma unroll\nfor(int j=0;j<{access};++j)value[j]=fmaf(left,{g}[item*{gd}+j*{d}+component],value[j]);}}' for x,g in pairs)
                    vector=f'\n#pragma unroll\nfor(int component=0;component<{d};++component){{{vector}}}'

            elif product_on_left:
                left=''.join(f'value=fmaf({x}[item*{xd}+{xo}+(column%{u})*{d}+component],{y}[item*{yd}+{yo}+column/{u}],value);' for x,y in pairs)
                right=f'value={multiplier}[item*{gd}+{go}+column*{d}+component];'
                vector=f'float{access} vector_value=*reinterpret_cast<const float{access}*>({multiplier}+item*{gd});'+''.join(f'value[{j}]=vector_value.{field};' for j,field in enumerate(fields))
            else:
                left=f'value={multiplier}[item*{xd}+{xo}+column*{d}+component];'
                right=''.join(f'value=fmaf({y}[item*{yd}+{yo}+column/{o}],{g}[item*{gd}+{go}+(column%{o})*{d}+component],value);' for y,g in pairs)
                vector=''.join(f'{{float left={y}[item*{yd}];float{access} right=*reinterpret_cast<const float{access}*>({g}+item*{gd});'+''.join(f'value[{j}]=fmaf(left,right.{field},value[{j}]);' for j,field in enumerate(fields))+'}' for y,g in pairs)
            if product_on_left:
                # Row order [v,u] lets one Y value scale a native X vector.
                a_vector=''.join((f'if({x})' if i else '')+f'{{float{a_access} left=*reinterpret_cast<const float{a_access}*>({x}+k*{xd}+{xo}+column%{u});float right={y}[k*{yd}+{yo}+column/{u}];'+''.join(f'value.{field}=fmaf(left.{field},right,value.{field});' for field in 'xyzw'[:a_access])+'}' for i,(x,y) in enumerate(pairs))
            else:
                a_vector=f'value=*reinterpret_cast<const float{a_access}*>(operand0+k*{xd}+{xo}+column);'
            a_vector_bodies.append(a_vector)
            a_bodies.append(left)
            b_bodies.append(right)
            vector_bodies.append(vector)
        widest=max(range(len(terms)),key=lambda i:len(terms[i][2]))
        a_body=a_bodies[widest]
        a_vector_body=a_vector_bodies[widest]
        b_body=b_bodies[widest]
        vector_body=vector_bodies[widest]
        vector_load=(contract_components and o%access==0) or (d==1 and o%access==0 and gd%access==0 and go%access==0)
        # Column addresses do not change between K tiles.
        if contract_components:
            offset0=f'int64_t({xo})+int64_t(column/{o})*{d}'
            offset1=f'int64_t({go})+int64_t(column%{o})*{d}'
        elif product_on_left:
            offset0=f'int64_t({go})+int64_t(column)*{d}'
        else:
            offset0=f'int64_t({yo})+column/{o}'
            offset1=f'int64_t({go})+int64_t(column%{o})*{d}'
        pointer_bias=f'operand0+={offset0};'
        if not product_on_left:
            pointer_bias+=f"""operand1 += {offset1};
if (operand2) {{
    operand2 += {offset0};
    operand3 += {offset1};
}}"""
        a_body=a_body.replace('value=fmaf(operand2[','if(operand2)value=fmaf(operand2[')
        b_body=b_body.replace('value=fmaf(operand2[','if(operand2)value=fmaf(operand2[')
        vector_body=vector_body.replace('{float left=operand2[','if(operand2){float left=operand2[')
        selections=[]
        for i,(target,_,_) in enumerate(terms):
            assignments=''.join(f'a{j}={value};' for j,value in enumerate(a_arguments[i].split(',')))+''.join(f'b{j}={value};' for j,value in enumerate(b_arguments[i].split(',')))
            selections.append(f"""case {i}: {{
    {assignments}result = {target};
    break;
}}""")
        weight_index=f'(int64_t(c/{o})*{v}+p)*{o}+c%{o}' if contract_components else f'(int64_t(p%{u})*{v}+p/{u})*{o}+c' if product_on_left else f'int64_t(p)*{v*o}+c'
        operation=f'''
{{
    using Mma = {n}_Mma;
    typename Mma::FragmentC acc;
    acc.clear();
    {n}_A a(a0, a1, a2, a3, first_row, begin, end, threadIdx.x);
    {n}_B b(b0, b1, b2, b3, first_column, begin, end, threadIdx.x);
    Mma mma(shared, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
    mma(int((end - begin + 7) / 8), acc, a, b, acc);
    __syncthreads();
    using IteratorC = typename Mma::Operator::IteratorC;
    using Policy = typename IteratorC::Policy;
    auto lane = Policy::get_lane_layout().inverse(threadIdx.x % 32);
    int row = first_row + (threadIdx.x / 32 % 4) * {rows//4} + lane.row() * Policy::LaneMmaShape::kM;
    int column = first_column + (threadIdx.x / 32 / 4) * {columns//2} + lane.column() * Policy::LaneMmaShape::kN;
#pragma unroll
    for (int mm = 0; mm < IteratorC::Iterations::kRow; ++mm) {{
#pragma unroll
        for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {{
            int p = row + mm * IteratorC::Delta::kRow + m;
#pragma unroll
            for (int nn = 0; nn < IteratorC::Iterations::kColumn; ++nn) {{
#pragma unroll
                for (int j = 0; j < Policy::LaneMmaShape::kN; ++j) {{
                    int c = column + nn * IteratorC::Delta::kColumn + j;
                    int element = j + Policy::LaneMmaShape::kN *
                                          (nn + IteratorC::Iterations::kColumn * (m + mm * Policy::LaneMmaShape::kM));
                    if (p < {height} && c < {width})
                        atomicAdd(result + int64_t({wo}) + {weight_index},
                                  acc[element] * {scale});
                }}
            }}
        }}
    }}
    __syncthreads();
}}
'''
        # A K coordinate is an item in this layout. Bias addresses once and
        # carry only the remaining rows through the official mainloop.
        a_stride = yd if contract_components else xd
        a_offset = f"{yo}+column" if contract_components else f"{xo}+column"
        if product_on_left:
            a_offset = f"{xo}+column%{u}"
        a_bias = f"operand0+=(int64_t(start)+offset.strided())*{a_stride}+{a_offset};"
        a_advance = f"operand0+=int64_t(8)*{a_stride};"
        if product_on_left:
            a_bias += f"""operand1 += (int64_t(start) + offset.strided()) * {yd} + {yo} + column / {u};
if (operand2) {{
    operand2 +=
        (int64_t(start) + offset.strided()) * {xd} + {xo} + column % {u};
    operand3 +=
        (int64_t(start) + offset.strided()) * {yd} + {yo} + column / {u};
}}"""
            a_advance += f"""operand1 += int64_t(8) * {yd};
if (operand2) {{
    operand2 += int64_t(8) * {xd};
    operand3 += int64_t(8) * {yd};
}}"""
        a_direct = a_body
        for address in (f"item*{xd}+{xo}+(column%{u})*{d}+component", f"item*{xd}+{xo}+column*{d}+component"):
            a_direct = a_direct.replace(address, f"int64_t(delta)*{xd}")
        for address in (f"item*{yd}+{yo}+column/{u}", f"item*{yd}+{yo}+column"):
            a_direct = a_direct.replace(address, f"int64_t(delta)*{yd}")
        a_vector_direct = a_vector_body
        for address in (f"k*{xd}+{xo}+column%{u}", f"k*{xd}+{xo}+column"):
            a_vector_direct = a_vector_direct.replace(address, f"int64_t(delta)*{xd}")
        a_vector_direct = a_vector_direct.replace(f"k*{yd}+{yo}+column/{u}", f"int64_t(delta)*{yd}")
        b_strides = (gd,) if product_on_left else ((xd, gd) if contract_components else (yd, gd))
        b_bias = "".join(f"operand{i}+=(int64_t(start)+offset.strided())*{stride};" for i, stride in enumerate(b_strides))
        b_advance = "".join(f"operand{i}+=int64_t(8)*{stride};" for i, stride in enumerate(b_strides))
        if not product_on_left:
            b_bias += f"""if (operand2) {{
    operand2 += (int64_t(start) + offset.strided()) * {b_strides[0]};
    operand3 += (int64_t(start) + offset.strided()) * {b_strides[1]};
}}"""
            b_advance += f"""if (operand2) {{
    operand2 += int64_t(8) * {b_strides[0]};
    operand3 += int64_t(8) * {b_strides[1]};
}}"""
        b_direct = vector_body
        for stride in (xd, yd, gd):
            b_direct = b_direct.replace(f"item*{stride}", f"int64_t(delta)*{stride}")
        a_vector_load = f'if(enabled&&delta<remaining){{{a_vector_direct}}}'
        memory_include = ''
        if product_on_left and len(terms[widest][2]) > 1 and a_access > 1:
            memory_include = '#include <cutlass/arch/memory.h>\n'
            a_vector_load = f'''
    bool valid = enabled && delta < remaining, second_valid = valid && operand2;
    float{a_access} left = {{}}, second_left = {{}};
    float right = 0.f, second_right = 0.f;
    cutlass::arch::global_load<float{a_access}, {4*a_access}>(
        left, operand0 + int64_t(delta) * {xd}, valid);
    cutlass::arch::global_load<float, 4>(right, operand1 + int64_t(delta) * {yd}, valid);
    cutlass::arch::global_load<float{a_access}, {4*a_access}>(
        second_left, (operand2 ? operand2 : operand0) + int64_t(delta) * {xd}, second_valid);
    cutlass::arch::global_load<float, 4>(
        second_right, (operand2 ? operand3 : operand1) + int64_t(delta) * {yd}, second_valid);
''' + ''.join(f'value.{field}=fmaf(left.{field},right,value.{field});' for field in 'xyzw'[:a_access])
            a_vector_load += 'if(operand2){' + ''.join(f'value.{field}=fmaf(second_left.{field},second_right,value.{field});' for field in 'xyzw'[:a_access]) + '}'
        kernels.append(f'''
{memory_include}#include <type_traits>
#include <cutlass/gemm/device/gemm.h>
using {n}_Default = typename cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{rows}, {columns}, 8>,
    cutlass::gemm::GemmShape<{rows//4}, {columns//2}, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>::GemmKernel::Mma;
template <bool IsA> struct {n}_Iterator {{
    using Element = float;
    static constexpr int Access = IsA ? {a_access} : {access};
    static constexpr int Columns = IsA ? {height} : {width};
    static constexpr bool Vector = IsA ? ({a_access} > 1) : {str(vector_load).lower()};
    static constexpr bool Biased = ({components} == 1) && (IsA || {str(vector_load).lower()});
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<IsA ? {rows} : {columns}, 8>, {threads},
        Access>;
    using Fragment = std::conditional_t<IsA, typename {n}_Default::IteratorA::Fragment,
                                        typename {n}_Default::IteratorB::Fragment>;
    const float *operand0, *operand1, *operand2, *operand3;
    int first;
    Index start, end, remaining;
    cutlass::PitchLinearCoord offset;
    bool enabled = true;
    CUTLASS_DEVICE {n}_Iterator(const float *p0, const float *p1, const float *p2, const float *p3,
                                               int f, Index b, Index e, int tid)
        : operand0(p0), operand1(p1), operand2(p2), operand3(p3), first(f), start(b), end(e),
          offset(ThreadMap::initial_offset(tid)) {{
        if constexpr (IsA && {components} == 1) {{
            const int column = first + offset.contiguous();
            enabled = column + Access - 1 < Columns;
            if (enabled) {{
{indent(a_bias, '                ')}
            }}
        }} else if constexpr (!IsA && Vector) {{
            const int column = first + offset.contiguous();
            enabled = column + Access - 1 < Columns;
            if (enabled) {{
{indent(pointer_bias, '                ')}
            }}
            if constexpr (Biased) {{
                if (enabled) {{
{indent(b_bias, '                    ')}
                }}
            }}
        }}
        if constexpr (Biased) {{
            static_assert(ThreadMap::Iterations::kContiguous == 1);
            remaining = Index(int64_t(end) - start - offset.strided());
        }}
    }}
    CUTLASS_DEVICE void clear_mask(bool clear = true) {{
        if (clear)
            enabled = false;
    }}
    CUTLASS_DEVICE {n}_Iterator &operator++() {{
        if constexpr (Biased) {{
            if constexpr (IsA) {{
{indent(a_advance, '                ')}
            }} else {{
{indent(b_advance, '                ')}
            }}
            remaining -= 8;
        }} else
            start += min(Index(8), end - start);
        return *this;
    }}
    CUTLASS_DEVICE void load(Fragment &fragment) const {{
        if constexpr (Biased) {{
#pragma unroll
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
                const int delta = s * ThreadMap::Delta::kStrided;
                if constexpr (IsA && Vector) {{
                    float{a_access} value = {{}};
{indent(a_vector_load, '                    ')}
                    const float *values = reinterpret_cast<const float *>(&value);
#pragma unroll
                    for (int j = 0; j < Access; ++j)
                        fragment[s * Access + j] = values[j];
                }} else if constexpr (IsA) {{
                    float value = 0.f;
                    if (enabled && delta < remaining) {{
{indent(a_direct, '                        ')}
                    }}
                    fragment[s] = value;
                }} else {{
                    float value[Access] = {{}};
                    if (enabled && delta < remaining) {{
{indent(b_direct, '                        ')}
                    }}
#pragma unroll
                    for (int j = 0; j < Access; ++j)
                        fragment[s * Access + j] = value[j];
                }}
            }}
        }} else {{
#pragma unroll
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {{
#pragma unroll
                for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {{
                    int column = first + offset.contiguous() + c * ThreadMap::Delta::kContiguous;
                    const Index k = start + offset.strided() + s * ThreadMap::Delta::kStrided;
                    const int index = (s * ThreadMap::Iterations::kContiguous + c) * Access;
                    if constexpr (IsA && Vector) {{
                        float{a_access} value = {{}};
                        if (enabled && column + Access - 1 < Columns && k < end) {{
{indent(a_vector_body, '                            ')}
                        }}
                        const float *values = reinterpret_cast<const float *>(&value);
#pragma unroll
                        for (int j = 0; j < Access; ++j)
                            fragment[index + j] = values[j];
                    }} else if constexpr (!IsA && Vector) {{
                        float value[Access] = {{}};
                        if (enabled && k < end) {{
                            Index item = k;
{indent(vector_body, '                            ')}
                        }}
#pragma unroll
                        for (int j = 0; j < Access; ++j)
                            fragment[index + j] = value[j];
                    }} else {{
#pragma unroll
                        for (int j = 0; j < Access; ++j) {{
                            float value = 0.f;
                            if (enabled && column + j < Columns && k < end) {{
                                Index item = k / {components};
                                int component = k % {components};
                                const int column = first + offset.contiguous() + c * ThreadMap::Delta::kContiguous + j;
                                if constexpr (IsA) {{
{indent(a_body, '                                    ')}
                                }} else {{
{indent(b_body, '                                    ')}
                                }}
                            }}
                            fragment[index + j] = value;
                        }}
                    }}
                }}
            }}
        }}
    }}
}};
using {n}_A = {n}_Iterator<true>;
using {n}_B = {n}_Iterator<false>;
using {n}_SharedA =
    cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<{rows}, 8>, float,
                                                         cutlass::layout::ColumnMajor, 1,
                                                         typename {n}_A::ThreadMap>;
using {n}_SharedB =
    cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<8, {columns}>, float,
                                                         cutlass::layout::RowMajor, 0,
                                                         typename {n}_B::ThreadMap>;
using {n}_Mma = cutlass::gemm::threadblock::MmaPipelined<
    typename {n}_Default::Shape, {n}_A, {n}_SharedA, {n}_B,
    {n}_SharedB, float, cutlass::layout::RowMajor, typename {n}_Default::Policy>;
__device__ __forceinline__ void {n}({ptrs}, {','.join(f'float* __restrict__ {a}' for a in outputs)}, Index items, Index task,
                                                   void *storage) {{
    auto &shared = *reinterpret_cast<typename {n}_Mma::SharedStorage *>(storage);
    int64_t size = int64_t(items) * {components};
    int splits = int(min(int64_t({splits}), (size + 127) / 128));
    int term_index = task % {len(terms)};
    task /= {len(terms)};
    const float *a0, *a1, *a2, *a3, *b0, *b1, *b2, *b3;
    float *result;
    switch (term_index) {{
{indent(chr(10).join(selections), '        ')}
    }}
    int tile = task / splits, split = task % splits;
    int first_row = (tile / {((width+columns-1)//columns)}) * {rows},
        first_column = (tile % {((width+columns-1)//columns)}) * {columns};
    Index span = ((size + splits * 8 - 1) / (splits * 8)) * 8;
    Index begin = split * span, end = min(Index(size), begin + span);
    if (begin >= end)
        return;
{indent(operation, '    ')}
}}
''')
    capacity=max(1,min(UVW_WEIGHT_MAX_SPLITS,(UVW_TARGET_BLOCKS//2+sum(tile_counts)-1)//sum(tile_counts)))
    source=f"""constexpr int {name}_split_capacity = {capacity};
"""+'\n'.join(kernels)
    # Start longer paths first so shorter paths can fill the last scheduling wave.
    calls = [call for _, call in sorted(zip(tile_work, calls), key=itemgetter(0), reverse=True)]
    source+=f"""
__global__ __launch_bounds__({threads}, {UVW_GEMM_BLOCKS_PER_SM}) void {name}_0({ptrs},
                                                                                            {','.join(f'float* __restrict__ {a}' for a in outputs)},
                                                                                            Index items) {{
    union Storage {{
{indent(chr(10).join(shared_types), '        ')}
    }};
    __shared__ Storage shared;
    Index task = blockIdx.x;
{indent(chr(10).join(calls), '    ')}
}}
"""
    return source,'+'.join(grids),threads


def make_uvw_backward_shared_source(meta, program, inputs, outputs, phase, *, arch):
    scalar = meta.get('dtype', torch.float32) is torch.float32 and all(s['d2'] == 1 and s['d1'] == s['d3'] for s in meta['generated_specs'])
    kernels, launches = [], []
    for role, emitter in (
        ("xy", make_uvw_backward_dxdy_shared_kernel),
        ("weight", make_uvw_backward_dw_shared_kernel),
    ):
        targets = tuple(
            v
            for v in outputs
            if v
            in (program["w"] if role == "weight" else (*program["x"], *program["y"]))
        )
        if not targets:
            continue
        name = f"lunatp_uvw_{phase}_shared_{role}_fused_fp32"
        if scalar and role == "weight":
            code, grid, threads = make_uvw_scalar_dw_kernel(meta, program, inputs, targets, name)
            kernels.append(code)
            args = ",".join(f"{v}.data_ptr<float>()" for v in (*inputs, *targets))
            launches.append(f"{name}_0<<<{grid},{threads},0,stream>>>({args},items);")
            continue
        if scalar:
            xy_launches = []
            scalar_specs = tuple(spec for spec in meta["generated_specs"] if spec["d1"] == 1)
            angular_specs = tuple(spec for spec in meta["generated_specs"] if spec["d1"] > 1)
            for specs, selected, build, suffix in (
                (scalar_specs, targets, make_uvw_scalar_dxdy_kernel, "scalar"),
                (angular_specs, tuple(v for v in targets if v in program["x"]), make_uvw_scalar_dx_kernel, "x"),
                (angular_specs, tuple(v for v in targets if v in program["y"]), make_uvw_scalar_dy_kernel, "y"),
            ):
                if not specs or not selected:
                    continue
                config = dict(meta, generated_specs=specs)
                kernel = f"{name}_{suffix}"
                code, grid, threads = build(config, program, inputs, selected, kernel)
                kernels.append(code)
                pointers = ",".join(f"{v}.data_ptr<float>()" for v in (*inputs, *selected))
                xy_launches.append(f"{kernel}<<<{grid},{threads},0,stream>>>({pointers},items);")
            launches.append("\n".join(xy_launches))
            continue
        code, grid, threads = emitter(meta, program, inputs, targets, name, arch=arch)
        kernels.append(code)
        args = ",".join(f"{v}.data_ptr<float>()" for v in (*inputs, *targets))
        if meta.get("dtype") is torch.float64:
            launches.append(
                f"""C10_CUDA_CHECK(cudaFuncSetAttribute({name}, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    {name}_shared_bytes));
{name}<<<{grid}, {threads}, {name}_shared_bytes,stream>>>({args},
                                                                                                    items);"""
            )
        else:
            launches.append(f"{name}<<<{grid},{threads},0,stream>>>({args},items);")
    independent_launches = len(launches)
    if "gg" in outputs and program["forward"]:
        output_groups = {}
        for output_spec in meta["generated_specs"]:
            output_key = (
                output_spec["out_offset"],
                output_spec["out_channels"],
                output_spec["d3"],
            )
            output_groups.setdefault(output_key, []).append(output_spec)
        # Bound the sparse-CG instruction body of each output tile. Splits
        # still reduce directly into GG, so there is no global CG workspace.
        output_chunks = []
        for output_group in output_groups.values():
            output_chunk, output_terms = [], 0
            for output_spec in output_group:
                if output_chunk and output_terms + len(output_spec["terms"]) > 128:
                    output_chunks.append(output_chunk)
                    output_chunk, output_terms = [], 0
                output_chunk.append(output_spec)
                output_terms += len(output_spec["terms"])
            output_chunks.append(output_chunk)
        if scalar and meta["scalar_input"]:
            scalar_chunks, angular_groups = [], {}
            for chunk in output_chunks:
                first = chunk[0]
                if first["d3"] == 1:
                    scalar_chunks.append(chunk)
                else:
                    key = first["out_offset"], first["out_channels"], first["d3"]
                    angular_groups.setdefault(key, []).append(chunk)
            output_chunks = scalar_chunks
            # Pack disjoint outputs together, retaining each output's CG chunks.
            for chunks in zip_longest(*angular_groups.values(), fillvalue=()):
                output_chunks.append([spec for chunk in chunks for spec in chunk])
        output_products = tuple(
            (program["cg"][q], weight) for q, weight in program["forward"]
        )
        for output_index, output_specs in enumerate(output_chunks):
            output_name = f"lunatp_uvw_{phase}_shared_g_{output_index}_fused_fp32"
            output_code, output_grid, _, output_pointers, output_threads = (
                make_uvw_forward_shared_kernel(
                    {**meta, "generated_specs": output_specs},
                    output_name,
                    output_products,
                    arch=arch,
                    accumulate=True,
                    include_mma=int(arch.rstrip("af")) >= 90,
                )
            )
            kernels.append(output_code)
            output_args = ",".join(
                f"{output_v}.data_ptr<float>()" for output_v in output_pointers
            )
            output_shared = f"{output_name}_shared_bytes" if meta.get("dtype") is not torch.float64 else "0"
            prepare_output = f"""if constexpr ({output_shared} > 0) {{
    static thread_local int prepared = -1;
    if (prepared != x.get_device()) {{
        C10_CUDA_CHECK(cudaFuncSetAttribute({output_name}<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            {output_shared}));
        prepared = x.get_device();
    }}
}}"""
            launches.append(
                prepare_output + f"{output_name}<true><<<{output_grid},{output_threads},{output_shared},stream>>>({output_args},gg.data_ptr<float>(),items,gg.size(1),{output_grid});"
            )

    allocations = [f"auto {v}=at::empty_like({OUTPUT_SHAPES[v]});" for v in outputs]
    clears = [
        f"if({v}.numel()) C10_CUDA_CHECK(cudaMemsetAsync({v}.data_ptr<float>(),0,{v}.numel()*sizeof(float),stream));"
        for v in outputs
    ]
    # Keep saturated FP64 first/second orders serial; graph replay favors that
    # schedule. Independent output branches overlap in the other shared paths.
    stream_source = ""
    if (
        independent_launches and len(launches) > 1
        and (meta.get("dtype") is not torch.float64 or phase == "triple_bwd")
    ):
        stream_source = DERIVATIVE_STREAM_SOURCE + "\n"
        launches[0] = (
            "auto& lane = derivative_stream(x.get_device(), stream);"
            "DerivativeFork fork(lane, stream);"
            + launches[0].replace(",stream>>>", ",lane.stream>>>")
        )
        launches.append("fork.join();")
    source = rf"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>
{stream_source}{chr(10).join(kernels)}
std::vector<at::Tensor> launch({','.join((f'at::Tensor {v}' for v in inputs))}) {{
    c10::cuda::CUDAGuard guard(x.device());
    Index items = x.size(0);
{indent(chr(10).join(allocations), "    ")}
    auto stream = at::cuda::getCurrentCUDAStream();
{indent(chr(10).join(clears), "    ")}
    if (items) {{
{indent(chr(10).join(launches), "        ")}
    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{','.join(outputs)}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch", &launch);
}}
"""

    if meta.get("dtype") is torch.float64:
        source = get_fp64_mma(arch)[0] + "\n" + source
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, order, *inputs, needs=None):
    outputs = PROGRAMS[order]["outputs"]
    selected = (
        outputs
        if needs is None
        else tuple(name for name, needed in zip(outputs, needs) if needed)
    )
    missing = 0
    if order >= 2:
        missing = (inputs[4] is None) | ((inputs[5] is None) << 1) | ((inputs[6] is None) << 2)
    if order == 3:
        missing |= ((inputs[7] is None) << 3) | ((inputs[8] is None) << 4) | ((inputs[9] is None) << 5) | ((inputs[10] is None) << 6)
    if missing and not (
        op.weight_mode == "shared" and op.compute_dtype is torch.float32
        and op.meta["scalar_input"]
    ):
        x, y, w, g = inputs[:4]
        templates = (x, y, w, g, x, y, w, x, y, w, g)
        inputs = tuple(torch.zeros_like(template) if value is None else value
                       for value, template in zip(inputs, templates))
        missing = 0
    device = inputs[0].get_device()
    if device != op.arch_device:
        op.arch = get_cuda_arch(inputs[0].device)
        op.arch_device = device
    key = op.arch, order, selected, missing, device
    launch = op.kernels.get(key)
    if launch is None:
        program = PROGRAMS[order]
        input_names = INPUTS[:len(inputs)]
        if missing:
            input_names = tuple(name for name, value in zip(input_names, inputs)
                                if value is not None)
            # None is a zero incoming gradient. Remove its products at JIT time;
            # requested zero outputs keep the existing zero-tensor semantics.
            filtered = dict(program)
            for contraction in ("cg", "z"):
                filtered[contraction] = {}
                for name, pairs in program[contraction].items():
                    kept = tuple(pair for pair in pairs
                                 if all(v in input_names for v in pair))
                    if kept:
                        filtered[contraction][name] = kept
            for feature in ("x", "y"):
                filtered[feature] = {}
                for out, pairs in program[feature].items():
                    kept = tuple((v, term) for v, term in pairs
                                 if v in input_names and term in filtered["z"])
                    if kept:
                        filtered[feature][out] = kept
            filtered["w"] = {}
            for out, pairs in program["w"].items():
                kept = tuple((term, v) for term, v in pairs
                             if v in input_names and term in filtered["cg"])
                if kept:
                    filtered["w"][out] = kept
            filtered["forward"] = tuple(
                (term, v) for term, v in program["forward"]
                if v in input_names and term in filtered["cg"]
            )
            program = filtered
        dtype_tag = "fp64" if op.compute_dtype == torch.float64 else "fp32"
        phase = ("bwd", "double_bwd", "triple_bwd")[order - 1]
        make_source = (
            make_uvw_backward_shared_source
            if op.weight_mode == "shared"
            else make_uvw_backward_external_source
        )
        kernel_meta = {**op.meta, "multiprocessors": get_device_properties(inputs[0].device).multi_processor_count}
        source = make_source(
            kernel_meta,
            program,
            input_names,
            selected,
            phase,
            arch=op.arch,
        )
        extension = load_cuda_extension(
            f"lunatp_uvw_{phase}_{op.weight_mode}_fused_{dtype_tag}",
            source,
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )
        configure = getattr(extension, "configure", None)
        if configure is not None:
            configure(inputs[0])
        launch = op.kernels[key] = extension.launch
    result = launch(*(value for value in inputs if value is not None)) if missing else launch(*inputs)
    if needs is None:
        return tuple(result)
    values = iter(result)
    return tuple(next(values) if needed else None for needed in needs)
