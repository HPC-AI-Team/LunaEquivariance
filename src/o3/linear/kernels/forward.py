"""Linear forward CUDA kernels and bindings."""

from __future__ import annotations

from collections import defaultdict
from textwrap import indent

import torch

from ....utils.precision import get_fp64_mma, float_literal


def make_feature_kernel(meta, name, pairs, broadcast, *, backward=False):
    roles = tuple(dict.fromkeys(role for term in (*pairs, *broadcast) for role in term))
    parameters = ", ".join(f"const Element* __restrict__ {role}" for role in roles)
    dest = "in_dim" if backward else "out_dim"
    src = "out_dim" if backward else "in_dim"
    groups = defaultdict(list)
    for spec in meta["generated_specs"]:
        offset = spec["x_offset" if backward else "out_offset"]
        channels = spec["in_channels" if backward else "out_channels"]
        groups[(offset, channels, spec["components"])].append(spec)

    lines = []
    for (offset, channels, components), specs in groups.items():
        lines.append((r"""if (q >= {offset} && q < {end}) {{
    const int channel = (q - {offset}) / {components};
    const int component = (q - {offset}) % {components};""").format(
            offset=offset, end=offset+channels*components, components=components,
        ))
        for spec in specs:
            cin, cout = spec["in_channels"], spec["out_channels"]
            width = cout if backward else cin
            source_offset = spec["out_offset" if backward else "x_offset"]
            weight_offset = spec["weight_offset"]
            windex = f"channel*{cout}+k" if backward else f"k*{cout}+channel"
            batch = "" if meta["shared_weights"] else f"n*{meta['weight_numel']}+"
            coeff = float_literal(spec["path_weight"], meta["dtype"])
            for a, w in pairs:
                feature_index = f"n*{meta[src]}+{source_offset}+k*{components}+component"
                weight_index = f"{batch}{weight_offset}+{windex}"
                feature = f"{a}[{feature_index}]"
                weight = f"{w}[{weight_index}]"
                lines.append((r"""    {{
        Element part = Element(0);
#pragma unroll {unroll}
        for (int k = 0; k < {width}; ++k)
            part = {fma}({feature}, {weight}, part);
        value = {fma}(part, Element({coefficient}), value);
    }}""").format(
                    unroll=1 if width>32 else 4, width=width,
                    fma='__fma_rn',
                    feature=feature, weight=weight, coefficient=coeff,
                ))
        lines.append("}")
    if not backward:
        for spec in meta["bias_specs"]:
            start, count = spec["out_offset"], spec["channels"]
            for (b,) in broadcast:
                bias_index = f"q-{start}+{spec['bias_offset']}"
                bias = f"{b}[{bias_index}]"
                lines.append(("if(q>={start} && q<{end}) value+={bias};").format(start=start, end=start+count, bias=bias))

    code = (
        (r"""
__global__ void {name}({parameters}{parameter_separator} Element* __restrict__ out, Index count) {{
    for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += Index(blockDim.x) * gridDim.x) {{
        const Index n = index / {dimension};
        const int q = index % {dimension};
        Element value = Element(0);
{contractions}
        out[index] = value;
    }}
}}
""").format(
            name=name,
            parameters=parameters,
            parameter_separator=',' if roles else '',
            dimension=meta[dest],
            contractions=indent("\n".join(lines), "        "),
        )
        if meta[dest]
        else ""
    )

    args = ", ".join(f"{role}.data_ptr<Element>()" for role in roles)
    count = f"num_items*int64_t({meta[dest]})"
    launch = (
        f"if ({count}>0) {name}<<<std::min<int64_t>(65535,({count}+255)/256),256,0,stream>>>({args}{',' if args else ''} out.data_ptr<Element>(),{count});"
        if code
        else ""
    )
    return {"source": code, "launch": launch}


def make_tiled_feature_kernel(
    meta, name, pairs, broadcast, *, backward=False, arch
):
    """Implicit AoS GEMM with static path bodies and one CTA dispatcher."""
    tile_m, tile_n, tile_k, threads = (32, 64, 16, 128)
    dest = meta["in_dim" if backward else "out_dim"]
    src = meta["out_dim" if backward else "in_dim"]
    groups = defaultdict(list)
    for spec in meta["generated_specs"]:
        groups[
            (
                spec["x_offset" if backward else "out_offset"],
                spec["in_channels" if backward else "out_channels"],
                spec["components"],
            )
        ].append(spec)
    # Unconnected destinations are output-owned zero groups in this same launch.
    spans = sorted((offset, offset + channels * d) for offset, channels, d in groups)
    cursor = 0
    for start, end in spans:
        if start > cursor:
            groups[(cursor, start - cursor, 1)] = []
        cursor = max(cursor, end)
    if cursor < dest:
        groups[(cursor, dest - cursor, 1)] = []
    if not groups:
        return dict(source="", launch="")

    roles = tuple(dict.fromkeys(role for term in (*pairs, *broadcast) for role in term))
    parameters = ", ".join(f"const Element* __restrict__ {role}" for role in roles)
    body_args = ", ".join(roles)
    dispatch, loads_a, loads_b, bias_add = [], [], [], []
    totals = []
    widths, component_counts, offsets, reductions = [], [], [], []
    for group, ((offset, channels, d), specs) in enumerate(groups.items()):
        nt = (channels + tile_n - 1) // tile_n
        ktotal = sum(s["out_channels" if backward else "in_channels"] for s in specs)
        tasks = f"((num_items*int64_t({d})+{tile_m-1})/{tile_m})*{nt}"
        totals.append(tasks)
        widths.append(channels)
        component_counts.append(d)
        offsets.append(offset)
        reductions.append(ktotal)
        dispatch.append(
            ("if(task<cursor+({tasks})) {{\n    {name}_body<{group}>({body_args}{separator}out,shared_a,shared_b,((task-cursor)/{tiles})*{tile_m},((task-cursor)%{tiles})*{tile_n},num_items);\n    return;\n}}\ncursor+=({tasks});").format(
                tasks=tasks, name=name, group=group, body_args=body_args,
                separator=',' if roles else '', tiles=nt, tile_m=tile_m, tile_n=tile_n,
            )
        )
        la = [(r"""case {group}: {{
    if (row >= num_items * int64_t({components}))
        return Element(0);
    const Index n = row / {components};
    const int component = row % {components};""").format(group=group, components=d)]
        lb = [(r"""case {group}: {{
    if (column >= {channels})
        return Element(0);""").format(group=group, channels=channels)]
        base = 0
        for spec in specs:
            count = spec["out_channels" if backward else "in_channels"]
            so = spec["out_offset" if backward else "x_offset"]
            wo, cout = spec["weight_offset"], spec["out_channels"]
            wi = f"column*{cout}+local_k" if backward else f"local_k*{cout}+column"
            feature_index = f"n*{src}+{so}+local_k*{d}+component"
            weight_index = f"{wo}+{wi}"
            feature = f"a[{feature_index}]"
            weight = f"b[{weight_index}]"
            la += [
                ("    if(k>={base} && k<{end}) {{\n        const int local_k=k-{base};\n        return {feature};\n    }}").format(base=base, end=base+count, feature=feature)
            ]
            lb += [
                ("    if(k>={base} && k<{end}) {{\n        const int local_k=k-{base};\n        return {weight}*Element({coefficient});\n    }}").format(base=base, end=base+count, weight=weight,
                                        coefficient=float_literal(spec['path_weight'], meta['dtype']))
            ]
            base += count
        la += ["    return Element(0);\n}"]
        lb += ["    return Element(0);\n}"]
        loads_a.extend(la)
        loads_b.extend(lb)
        for bs in meta["bias_specs"]:
            if (
                not backward
                and bs["out_offset"] >= offset
                and bs["out_offset"] + bs["channels"] <= offset + channels * d
            ):
                for (role,) in broadcast:
                    bias_index = f"{bs['bias_offset']}+column*components+component-{bs['out_offset']-offset}"
                    bias = f"{role}[{bias_index}]"
                    bias_add.append(
                        ("if(group=={group} && column*components+component>={start} && column*components+component<{end}) value+={bias};").format(group=group, start=bs['out_offset']-offset,
                                                 end=bs['out_offset']-offset+bs['channels'], bias=bias)
                    )

    loader_args = "const Element* a,int group,Index row,int k,int64_t num_items"
    header, mma = get_fp64_mma(arch)
    unique = name + "_Mma"
    header = header.replace("Fp64Mma16x8x16", unique)
    mma = mma.replace("Fp64Mma16x8x16", unique)
    init = ("Element acc[4][4]={}; const int lane=threadIdx.x&31,warp=threadIdx.x>>5;")
    calculate = (r"""
#pragma unroll
for (int k = 0; k < {tile_k}; k += 16) {{
    Element a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        a[i] = shared_a[((warp >> 1) * 16 + (lane >> 2) + (i & 1) * 8) * {tile_k} +
                        ((k + (lane & 3) + (i >> 1) * 4) ^ (((lane >> 2) & 3) << 2))];
#pragma unroll
    for (int n = 0; n < 4; ++n) {{
        Element b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            b[i] = shared_b[(k + (lane & 3) + i * 4) * {tile_n} +
                            (((warp & 1) * 32 + n * 8 + (lane >> 2)) ^ ((lane & 3) << 2))];
        {mma}::fma(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0],
            b[1], b[2], b[3], acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
    }}
}}
""").format(
        tile_k=tile_k,
        tile_n=tile_n,
        mma=mma,
    )
    loops = ("for(int j=0;j<4;++j) {\n    for(int i=0;i<4;++i) {")
    coords = ("const Index row=tile_row+(warp>>1)*16+(lane>>2)+(i>>1)*8;\nconst int column=tile_column+(warp&1)*32+j*8+(lane&3)*2+(i&1);\nElement value=acc[j][i];")
    a_store = ("(index/{tile_k})*{tile_k}+((index%{tile_k})^(((index/{tile_k})&3)<<{shift}))").format(tile_k=tile_k, shift=2)
    b_store = "(index/{tile_n})*{tile_n}+((index%{tile_n})^(((index/{tile_n})&3)<<2))".format(tile_n=tile_n)

    b_k, b_column = f"index/{tile_n}", f"index%{tile_n}"

    products = []
    for a, b in pairs:
        products.append(
            (r"""
for (Index first_k = 0; first_k < reduction_size; first_k += {tile_k}) {{
    for (int index = threadIdx.x; index < {a_elements}; index += {threads})
        shared_a[{a_store}] = {name}_a({a}, group, tile_row + index / {tile_k}, first_k + index % {tile_k}, num_items);
    for (int index = threadIdx.x; index < {b_elements}; index += {threads})
        shared_b[{b_store}] = {name}_b({b}, group, first_k + {b_k}, tile_column + {b_column});
    __syncthreads();
{calculate}
    __syncthreads();
}}
""").format(
                tile_k=tile_k,
                a_elements=tile_m * tile_k,
                threads=threads,
                a_store=a_store,
                name=name,
                a=a,
                b_elements=tile_k * tile_n,
                b_store=b_store,
                b_k=b_k,
                b_column=b_column,
                b=b,
                tile_n=tile_n,
                calculate=indent(calculate, "    "),
            )
        )

    total = "+".join(f"({t})" for t in totals) or "0"
    source = (
        header
        + (r"""
__device__ __forceinline__ Element {name}_a({loader_args}) {{
    switch (group) {{
{feature_loads}
        default: return Element(0);
    }}
}}
__device__ __forceinline__ Element {name}_b(const Element* b, int group, int k, int column) {{
    switch (group) {{
{weight_loads}
        default: return Element(0);
    }}
}}
template <int Group>
__device__ __forceinline__ void {name}_body({parameters}{parameter_separator} Element* out, Element* shared_a, Element* shared_b, Index tile_row,
    int tile_column, int64_t num_items) {{
    constexpr int group_widths[] = {{{group_widths}}};
    constexpr int group_components[] = {{{group_components}}};
    constexpr int group_offsets[] = {{{group_offsets}}};
    constexpr int group_reductions[] = {{{group_reductions}}};
    constexpr int group = Group, width = group_widths[Group], components = group_components[Group];
    constexpr int out_offset = group_offsets[Group], reduction_size = group_reductions[Group];
{init}
{products}
#pragma unroll
{loops}
{coords}
            if (row < num_items * int64_t(components) && column < width) {{
                const Index item = row / components;
                const int component = row % components;
{bias_add}
                out[item * {dest} + out_offset + column * components + component] = value;
            }}
        }}
    }}
}}
__global__ __launch_bounds__({threads}, 2) void {name}({parameters}{parameter_separator} Element* out, int64_t num_items) {{
    __shared__ Element shared_a[{a_elements}], shared_b[{b_elements}];
    const Index task = blockIdx.x;
    Index cursor = 0;
{task_dispatch}
}}
""").format(
            name=name,
            loader_args=loader_args,
            feature_loads=indent("\n".join(loads_a), "        "),
            weight_loads=indent("\n".join(loads_b), "        "),
            parameters=parameters,
            parameter_separator=',' if parameters else '',
            group_widths=','.join(map(str, widths)),
            group_components=','.join(map(str, component_counts)),
            group_offsets=','.join(map(str, offsets)),
            group_reductions=','.join(map(str, reductions)),
            init=indent(init, "    "),
            products=indent("\n".join(products), "    "),
            loops=indent(loops, "    "),
            coords=indent(coords, "            "),
            bias_add=indent("\n".join(bias_add), "                "),
            dest=dest,
            threads=threads,
            a_elements=tile_m * tile_k,
            b_elements=tile_k * tile_n,
            task_dispatch=indent("\n".join(dispatch), "    "),
        )
    )

    args = ", ".join(f"{r}.data_ptr<Element>()" for r in roles)
    launch = f"if(num_items>0 && {dest}>0) {name}<<<{total},{threads},0,stream>>>({args}{',' if args else ''}out.data_ptr<Element>(),num_items);"
    return dict(source=source, launch=launch)


def make_cutlass_feature_kernel(meta, name, pairs, broadcast, *, backward=False):
    """Group AoS GEMMs in one launch without packing features or weights."""
    dest = meta["in_dim" if backward else "out_dim"]
    source_dim = meta["out_dim" if backward else "in_dim"]
    groups = defaultdict(list)
    for spec in meta["generated_specs"]:
        groups[(spec["x_offset" if backward else "out_offset"], spec["in_channels" if backward else "out_channels"], spec["components"])].append(spec)
    cursor = 0
    for offset, channels, components in sorted(groups):
        if offset > cursor:
            groups[(cursor, offset - cursor, 1)] = []
        cursor = max(cursor, offset + channels * components)
    if cursor < dest:
        groups[(cursor, dest - cursor, 1)] = []
    if not groups:
        return dict(source="", launch="")

    tile_m, tile_n = 64, 64
    max_components = max(d for _, _, d in groups)
    tile_bound = sum(((channels + tile_n - 1) // tile_n) * d for _, channels, d in groups)
    chunk_items = min((2**31 - tile_m) // max_components, (2**31 - 1) // max(1, tile_bound))
    shared = meta["shared_weights"]
    dispatch, totals, storage = [], [], []
    roles = tuple(dict.fromkeys(role for term in (*pairs, *broadcast) for role in term))
    for group, ((offset, channels, components), specs) in enumerate(groups.items()):
        ntiles = (channels + tile_n - 1) // tile_n
        rows = f"num_items*int64_t({components})" if shared else str(components)
        mtiles = f"(({rows}+{tile_m-1})/{tile_m})"
        tasks = f"{mtiles}*{ntiles}" + ("" if shared else "*num_items")
        totals.append(tasks)
        body = []
        term = 0
        bias = next((b for b in meta["bias_specs"] if b["out_offset"] == offset and b["channels"] == channels and components == 1 and not backward), None)
        for spec in specs:
            for feature, weight in pairs:
                gemm = f"{name}_Kernel<{components}>"
                storage.append(f"typename {gemm}::GemmKernel::SharedStorage s{group}_{term};")
                if term:
                    source, ldc, beta = f"out+{offset}", dest, "1.0f"
                elif bias and broadcast:
                    source = f"{broadcast[0][0]}+{bias['bias_offset']}"
                    ldc, beta = 0, "1.0f"
                else:
                    source, ldc, beta = f"out+{offset}", dest, "0.0f"
                mode = "kGemm" if shared else "kBatched"
                batches = "1" if shared else "int(num_items)"
                a_stride = 0 if shared else source_dim
                b_stride = 0 if shared else meta["weight_numel"]
                c_stride = dest if (not shared and term) else 0
                d_stride = 0 if shared else dest
                body.append(f"""{{
    using Kernel = typename {gemm}::GemmKernel;
    typename Kernel::Params p;
    p.problem_size = {{int({rows}), {channels}, {spec['out_channels' if backward else 'in_channels']}}};
    p.grid_tiled_shape = {{int({mtiles}), {ntiles}, {batches}}};
    p.gemm_k_size = {spec['out_channels' if backward else 'in_channels']};
    p.mode = cutlass::gemm::GemmUniversalMode::{mode};
    p.batch_count = {batches};
    p.params_A = typename Kernel::Mma::IteratorA::Params(cutlass::layout::RowMajor({source_dim}));
    p.params_B = typename Kernel::Mma::IteratorB::Params(cutlass::layout::{'ColumnMajor' if backward else 'RowMajor'}({spec['out_channels']}));
    p.params_C = typename Kernel::Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor({ldc}));
    p.params_D = typename Kernel::Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor({dest}));
    p.output_op = typename Kernel::EpilogueOutputOp::Params({float_literal(spec['path_weight'],meta['dtype'])}, {beta});
    p.ptr_A = const_cast<Element*>({feature} + {spec['out_offset' if backward else 'x_offset']});
    p.ptr_B = const_cast<Element*>({weight} + {spec['weight_offset']});
    p.ptr_C = const_cast<Element*>({source});
    p.ptr_D = out + {offset};
    p.batch_stride_A = {a_stride};
    p.batch_stride_B = {b_stride};
    p.batch_stride_C = {c_stride};
    p.batch_stride_D = {d_stride};
    p.ptr_gather_A_indices = nullptr;
    p.ptr_gather_B_indices = nullptr;
    p.ptr_scatter_D_indices = nullptr;
    Kernel kernel;
    kernel.run_with_swizzle(p, shared_storage.s{group}_{term}, swizzle);
    __syncthreads();
}}""")
                term += 1
        # An output with no contraction is still written, including bias-only outputs.
        if not term:
            additions=[]
            for b in (() if backward else meta["bias_specs"]):
                if offset <= b["out_offset"] and b["out_offset"] + b["channels"] <= offset + channels * components:
                    for (role,) in broadcast:
                        additions.append(f"if(q>={b['out_offset']} && q<{b['out_offset']+b['channels']}) value+={role}[q-{b['out_offset']}+{b['bias_offset']}];")
            body.append(f"""for (int i = threadIdx.x; i < {tile_m*tile_n}; i += 128) {{
    const int64_t row = tile_m * {tile_m} + i / {tile_n};
    const int column = tile_n * {tile_n} + i % {tile_n};
    if (row < {rows} && column < {channels}) {{
        const int64_t item = {'row/'+str(components) if shared else 'batch'};
        const int q = {offset} + column * {components} + row % {components};
        Element value = 0;
{indent(chr(10).join(additions), '        ')}
        out[item * {dest} + q] = value;
    }}
}}""")
        dispatch.append(f"""if(task<cursor+({tasks})) {{
    const int64_t local=task-cursor;
    const int tile_n=local%{ntiles};
    const int tile_m=(local/{ntiles})%{mtiles};
    const int batch={'0' if shared else f'local/({ntiles}*{mtiles})'};
    {name}_Swizzle swizzle;
    swizzle.tile={{tile_m,tile_n,batch}};
{indent(chr(10).join(body), '    ')}
    return;
}}
cursor+=({tasks});""")
    params = ', '.join(f'const Element* {r}' for r in roles)
    feature_roles = {a for a, _ in pairs}
    weight_roles = {b for _, b in pairs}
    args = ', '.join(
        f'{r}.data_ptr<Element>()+first_item*int64_t({source_dim if r in feature_roles else meta["weight_numel"] if r in weight_roles and not shared else 0})'
        for r in roles
    )
    source = f"""
#include <cutlass/gemm/device/gemm_universal.h>

// Permute logical [item * component, channel] coordinates into e3nn storage.
template <int Components> struct {name}_Layout {{
    using Index = int32_t;
    using LongIndex = int64_t;
    Index stride;
    CUTLASS_HOST_DEVICE {name}_Layout(cutlass::MatrixCoord, Index s) : stride(s) {{}}
    CUTLASS_HOST_DEVICE {name}_Layout(cutlass::layout::PitchLinearCoord, Index s) : stride(s) {{}}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::MatrixCoord q) const {{
        return LongIndex(q.row() / Components) * stride + LongIndex(q.column()) * Components + q.row() % Components;
    }}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord q) const {{
        return operator()(cutlass::MatrixCoord(q.strided(), q.contiguous()));
    }}
}};
struct {name}_Swizzle : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> {{
    cutlass::gemm::GemmCoord tile;
    CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int) const {{ return tile; }}
}};
template <int Components>
using {name}_Gemm = cutlass::gemm::device::GemmUniversal<float, cutlass::layout::RowMajor, float, cutlass::layout::{'ColumnMajor' if backward else 'RowMajor'},
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tile_m}, {tile_n}, 8>, cutlass::gemm::GemmShape<32, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>, {name}_Swizzle, 2, 1, 1,
    cutlass::arch::OpMultiplyAdd, cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone, false, false,
    false, {name}_Layout<Components>, {name}_Layout<Components>>;
// CUTLASS's PermuteD iterator permutes stores only. Read C with the same
// AoS addressing when several paths or derivative terms share an output.
template <typename Iterator, int Components> struct {name}_OutputIterator : Iterator {{
    using ThreadMap = typename Iterator::ThreadMap;
    using Fragment = typename Iterator::Fragment;
    using Params = typename Iterator::Params;
    Element* pointer;
    int64_t stride;
    CUTLASS_DEVICE {name}_OutputIterator(Params const& params, Element* p, cutlass::MatrixCoord extent, int thread,
        cutlass::MatrixCoord offset = {{}}, int const* indices = nullptr)
        : Iterator(params, p, extent, thread, offset, indices), pointer(p), stride(params.stride / sizeof(Element)) {{}}
    CUTLASS_DEVICE void add_pointer_offset(int64_t offset) {{
        Iterator::add_pointer_offset(offset);
        pointer += offset;
    }}
    CUTLASS_DEVICE void load(Fragment& fragment) const {{
        typename Iterator::Mask mask;
        this->get_mask(mask);
#pragma unroll
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {{
#pragma unroll
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {{
#pragma unroll
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {{
                    const int r = this->thread_start_row() + row * ThreadMap::Delta::kRow +
                                  group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
                    const int fragment_row =
                        row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
#pragma unroll
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {{
                        const int c = this->thread_start_column() + column * ThreadMap::Delta::kColumn;
                        Element value = 0;
                        if (r < this->extent_row() && mask.predicates[column])
                            value =
                                pointer[int64_t(r / Components) * stride + int64_t(c) * Components + r % Components];
                        fragment[fragment_row * ThreadMap::Iterations::kColumn + column] = value;
                    }}
                }}
            }}
        }}
    }}
}};
template <int Components> struct {name}_Kernel {{
    using DefaultKernel = typename {name}_Gemm<Components>::GemmKernel;
    using DefaultEpilogue = typename DefaultKernel::Epilogue;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<typename DefaultEpilogue::Shape,
        typename DefaultEpilogue::WarpMmaOperator, DefaultEpilogue::kPartitionsK,
        {name}_OutputIterator<typename DefaultEpilogue::OutputTileIterator, Components>,
        typename DefaultEpilogue::AccumulatorFragmentIterator, typename DefaultEpilogue::WarpTileIterator,
        typename DefaultEpilogue::SharedLoadIterator, typename DefaultEpilogue::OutputOp,
        typename DefaultEpilogue::Padding>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<typename DefaultKernel::Mma, Epilogue, {name}_Swizzle>;
}};
__global__ void {name}({params}{',' if params else ''} Element* out, int64_t num_items) {{
    union Storage {{
{indent(chr(10).join(storage) if storage else 'float unused;', '        ')}
    }};
    __shared__ Storage shared_storage;
    const int64_t task = blockIdx.x;
    int64_t cursor = 0;
{indent(chr(10).join(dispatch), '    ')}
}}
"""
    total = '+'.join(f'({t})' for t in totals)
    launch = f"""if (num_items > 0) {{
    const int64_t total_items = num_items;
    for (int64_t first_item = 0; first_item < total_items; first_item += {chunk_items}) {{
        const int64_t num_items = std::min<int64_t>({chunk_items}, total_items - first_item);
        {name}<<<{total}, 128, 0, stream>>>({args}{',' if args else ''}out.data_ptr<Element>() + first_item * int64_t({dest}), num_items);
    }}
}}"""
    return dict(source=source, launch=launch)


def make_forward_source(meta, program, *, arch):
    sources = []
    launches = []
    returns = []
    for slot, name in enumerate(program["outputs"]):
        if name not in program["output"] and name not in program["broadcast"]:
            returns.append("pybind11::none()")
            continue
        pairs = program["output"].get(name, ())
        phase = ('forward', 'backward', 'double_backward', 'triple_backward')[program['order']]
        kernel = f"linear_{phase}_{name}"
        broadcast = program["broadcast"].get(name, ())
        if meta["dtype"] is torch.float32:
            emitted = make_cutlass_feature_kernel(meta, kernel, pairs, broadcast)
        elif meta["shared_weights"] and meta["max_channels"] >= 32:
            emitted = make_tiled_feature_kernel(
                meta, kernel, pairs, broadcast, arch=arch,
            )
        else:
            emitted = make_feature_kernel(meta, kernel, pairs, broadcast)
        shape = f"{{num_items,int64_t({meta['out_dim']})}}"
        sources.append(emitted["source"])
        launches.append(
            f"auto result_{slot}=torch::empty({shape},options);\n{{\n    auto out=result_{slot};\n"
            + indent(emitted["launch"], "    ") + "\n}"
        )
        returns.append(f"pybind11::cast(result_{slot})")

    parameters = ", ".join(f"torch::Tensor {role}" for role in program["inputs"])
    if not parameters:
        raise ValueError("A native Linear program needs at least one tensor dependency")
    first = program["inputs"][0]
    scalar = "double" if meta["dtype"] is torch.float64 else "float"
    prefix = (r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>
using Element = {scalar};

{kernels}
""").format(
        scalar=scalar,
        kernels="\n".join(sources),
    )
    body = (r"""
const c10::cuda::CUDAGuard guard({first}.device());
auto options = {first}.options();
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
{launches}
C10_CUDA_KERNEL_LAUNCH_CHECK();
return pybind11::make_tuple({results});
""").format(
        first=first,
        launches="\n".join(launches),
        results=','.join(returns),
    )
    parameters += ",int64_t num_items"
    return (
        prefix
        + f"pybind11::tuple launch({parameters}) {{\n" + indent(body.lstrip("\n"), "    ") + "}\n"
        + 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n    m.def("launch",&launch);\n}\n'
    )
