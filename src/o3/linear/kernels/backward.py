"""Linear backward CUDA kernels and bindings."""

from __future__ import annotations

from textwrap import indent

import torch

from ....utils.streams import STREAM_SOURCE
from ....utils.precision import get_fp64_mma, float_literal
from .forward import make_cutlass_feature_kernel, make_feature_kernel, make_tiled_feature_kernel
from .utils import MAX_WEIGHT_BLOCKS, MAX_WEIGHT_SPLITS


def make_weight_kernel(meta, kernel_name, pairs, *, arch):
    """Generate shared/per-item Linear weight contractions for every derivative order.

    ``pairs`` contains one or two ``(feature_role, output_seed_role)`` terms.
    Source needs the common ``Element`` and ``Index`` aliases.  Its launch
    fragment uses role tensors, ``out``, ``num_items`` and ``stream``.  Shared
    reductions use deterministic split-K partials and a final fused sum;
    one split writes directly.  Per-item weights have only a short component
    reduction, so every weight element is written once with no workspace.
    """
    pairs = tuple(tuple(pair) for pair in pairs)
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Linear weight contractions require feature/seed pairs")
    pointer_order = tuple(dict.fromkeys(role for pair in pairs for role in pair))
    specs = tuple(meta["generated_specs"])
    if not specs:
        return {"source": "", "pointer_order": pointer_order, "launch": ""}
    parameters = [f"const Element* __restrict__ {role}" for role in pointer_order]
    pointers = [f"{role}.data_ptr<Element>()" for role in pointer_order]
    if not meta["shared_weights"]:
        return make_external_weight_kernel(
            meta, kernel_name, pairs, pointer_order, parameters, pointers
        )
    if meta.get("dtype") == torch.float32:
        return make_cutlass_weight_kernel(meta, kernel_name, pairs, pointer_order)
    tile_m = 32 if max(int(s["in_channels"]) for s in specs) <= 32 else 64
    tile_n = 32 if max(int(s["out_channels"]) for s in specs) <= 32 else 64
    tile_k, threads = 16, 128

    useful = sum(
        int(s["in_channels"]) * int(s["out_channels"]) * int(s["components"]) for s in specs
    )
    padded = sum(
        ((int(s["in_channels"]) + tile_m - 1) // tile_m)
        * tile_m
        * ((int(s["out_channels"]) + tile_n - 1) // tile_n)
        * tile_n
        * int(s["components"])
        for s in specs
    )
    # Dot products lose channel reuse.  Use them only when GEMM padding
    # multiplies the useful work enough to offset that loss.
    if (
        max(int(s["in_channels"]) * int(s["out_channels"]) for s in specs) <= 1024
        and padded >= 8 * useful
    ):
        return make_small_weight_kernel(
            meta, kernel_name, pairs, pointer_order, parameters, pointers
        )
    # Regular square paths expose constant channel addressing to the FP64
    # compiler. Keep the original emitter for rectangular or padded paths.
    static_path = all(
        int(s["in_channels"]) == int(s["out_channels"])
        and int(s["in_channels"]) >= 32
        and int(s["in_channels"]) & (int(s["in_channels"]) - 1) == 0
        for s in specs
    )
    return make_fp64_weight_kernel(
        meta,
        kernel_name,
        pairs,
        pointer_order,
        parameters,
        pointers,
        arch=arch,
        tile_shape=(tile_m, tile_n, tile_k, threads),
        static_path=static_path,
    )


def make_cutlass_weight_kernel(meta, name, pairs, pointer_order):
    """Group native FP32 X.T @ grad_out with bounded split-K workspace."""
    weights = int(meta["weight_numel"])
    specs = tuple(meta["generated_specs"])
    output_tiles = sum(
        ((s["in_channels"] + 31) // 32) * ((s["out_channels"] + 63) // 64)
        for s in specs
    )
    # Wider paths already expose more output tiles; keep split-K workspace
    # bounded by the grouped work budget instead of growing with channel width.
    split_limit = min(MAX_WEIGHT_SPLITS, max(1, MAX_WEIGHT_BLOCKS // output_tiles))
    split_limit = 1 << (split_limit.bit_length() - 1)
    max_components = max(s["components"] for s in specs)
    chunk_items = (2**31 - 32) // max_components
    prefix = f"""
#include <cutlass/gemm/device/gemm_universal.h>
// Both operands have channels contiguous in the logical matrix and irreps
// components contiguous in storage: [item, channel, component].
template <int Components, int Stride> struct {name}_layout {{
    using LongIndex = int64_t;
    CUTLASS_HOST_DEVICE {name}_layout(cutlass::layout::PitchLinearCoord, int) {{}}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord q) const {{
        return int64_t(q.strided() / Components) * Stride + int64_t(q.contiguous()) * Components +
               q.strided() % Components;
    }}
}};
struct {name}_swizzle : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> {{
    cutlass::gemm::GemmCoord tile;
    CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int) const {{ return tile; }}
}};
template <int Components>
using {name}_gemm = cutlass::gemm::device::GemmUniversal<float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<16, 32, 32>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>, {name}_swizzle, 2, 1, 1,
    cutlass::arch::OpMultiplyAdd, cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone, false, false,
    false, cutlass::layout::NoPermute, {name}_layout<Components, {meta["in_dim"]}>, {name}_layout<Components, {meta["out_dim"]}>>;
"""

    fields, storage, initializers, dispatch, counts, reductions = [], [], [], [], [], []
    for term, (x, g) in enumerate(pairs):
        for path, spec in enumerate(specs):
            cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
            xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
            alpha = float_literal(spec["path_weight"], meta["dtype"])
            field = f"p{term}_{path}"
            tiles_m, tiles_n = (cin + 31) // 32, (cout + 63) // 64
            fields.append(f"typename {name}_gemm<{d}>::GemmKernel::Params {field};")
            storage.append(f"typename {name}_gemm<{d}>::GemmKernel::SharedStorage {field};")
            count = f"({tiles_m * tiles_n}*params.{field}.grid_tiled_shape.k())"
            counts.append(count)
            if term == 0:
                reductions.append(f"if(index>={wo} && index<{wo+cin*cout}) split_count=params.{field}.grid_tiled_shape.k();")
            initializers.append(f"""{{
    using Gemm = {name}_gemm<{d}>;
    auto partial = workspace.data_ptr<float>() + int64_t({term}) * splits * {weights} + {wo};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
        {{{cin}, {cout}, int(num_items * {d})}}, splits, {{float({alpha}), 0.0f}},
        {x}.data_ptr<float>() + first_item * int64_t({meta['in_dim']}) + {xo},
        {g}.data_ptr<float>() + first_item * int64_t({meta['out_dim']}) + {go}, partial, partial, 0, 0, 0, {weights}, {meta['in_dim']}, {meta['out_dim']},
        {cout}, {cout});
    params.{field} = typename Gemm::GemmKernel::Params(args, 1, 1);
}}""")
            dispatch.append(f"""if (task < cursor + {count}) {{
    const int local = task - cursor;
    {name}_swizzle swizzle;
    swizzle.tile = {{local % {tiles_m}, (local / {tiles_m}) % {tiles_n}, local / {tiles_m*tiles_n}}};
    typename {name}_gemm<{d}>::GemmKernel kernel;
    kernel.run_with_swizzle(params.{field}, shared_storage.{field}, swizzle);
    return;
}}
cursor += {count};""")
    source = prefix + f"""
struct {name}_params {{
{indent(chr(10).join(fields), '    ')}
}};
__global__ void {name}({name}_params params) {{
    union Storage {{
{indent(chr(10).join(storage), '        ')}
    }};
    __shared__ Storage shared_storage;
    const int task = blockIdx.x;
    int cursor = 0;
{indent(chr(10).join(dispatch), '    ')}
}}
__global__ void {name}_reduce({name}_params params, const float* partial, float* out, int splits, bool accumulate) {{
    const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= {weights})
        return;
    int split_count = 0;
{indent(chr(10).join(reductions), '    ')}
    float total = 0.0f;
    for (int term = 0; term < {len(pairs)}; ++term) {{
        float sum = 0.0f, correction = 0.0f;
        for (int split = 0; split < split_count; ++split) {{
            float value = partial[(int64_t(term) * splits + split) * {weights} + index] - correction;
            float next = sum + value;
            correction = (next - sum) - value;
            sum = next;
        }}
        total += sum;
    }}
    out[index] = accumulate ? out[index] + total : total;
}}
"""
    launch = f"""const int64_t total_items = num_items;
if (total_items == 0)
    out.zero_();
for (int64_t first_item = 0; first_item < total_items; first_item += {chunk_items}) {{
    const int64_t num_items = std::min<int64_t>({chunk_items}, total_items - first_item);
    const int splits = std::min<int64_t>({split_limit}, std::max<int64_t>(1, (num_items + 31) / 32));
    auto workspace = torch::empty({{{len(pairs)}, splits, {weights}}}, out.options());
    {name}_params params;
{indent(chr(10).join(initializers), '    ')}
    {name}<<<{'+'.join(counts)}, 128, 0, stream>>>(params);
    {name}_reduce<<<({weights} + 255) / 256, 256, 0, stream>>>(
        params, workspace.data_ptr<float>(), out.data_ptr<float>(), splits, first_item > 0);
}}"""
    return {"source":source,"pointer_order":pointer_order,"launch":launch}


def make_external_weight_kernel(meta, name, pairs, pointer_order, parameters, pointers):
    specs = tuple(meta["generated_specs"])
    counts = tuple(int(s["in_channels"]) * int(s["out_channels"]) for s in specs)
    if not counts or max(counts) > 256:
        if counts and all(c >= 1024 and c & (c - 1) == 0 for c in counts):
            # Round each path once over items, preserving component FMA order.
            fp64 = meta.get("dtype") == torch.float64
            fma = "__fma_rn" if fp64 else "__fmaf_rn"
            weights = int(meta["weight_numel"])
            tile, threads, vector = 1024, 256, 4
            declarations, blocks = [], []
            previous = "Index(0)"
            for path, spec in enumerate(meta["generated_specs"]):
                cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
                count = cin * cout
                xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
                alpha = float_literal(spec["path_weight"], meta.get("dtype"))
                end = f"end_path_{path}"
                declarations.append(
                    f"const Index {end}={previous}+(num_items*{count}+{tile-1})/{tile};"
                )
                expressions = [
                    ("value={fma}({feature}[item*{in_dim}+{x_offset}+u*{components}+{component}],{seed}[item*{out_dim}+{out_offset}+v*{components}+{component}],value);").format(
                            fma=fma, feature=x, seed=g, in_dim=meta['in_dim'], out_dim=meta['out_dim'],
                            x_offset=xo, out_offset=go, components=d, component=component,
                        )
                    for component in range(d)
                    for x, g in pairs
                ]
                blocks.append(
                    (r"""if (block >= {previous} && block < {end}) {{
    Index first = (block - {previous}) * {tile} + threadIdx.x * {vector};
#pragma unroll
    for (int j = 0; j < {vector}; ++j) {{
        Index pair = first + j;
        if (pair < num_items * {count}) {{
            Index item = pair / {count};
            int q = pair % {count};
            int u = q / {cout}, v = q % {cout};
            Element value = Element(0);
{products}
            out[item * {weights} + {wo} + q] = Element({alpha}) * value;
        }}
    }}
}}""").format(
                        previous=previous,
                        end=end,
                        tile=tile,
                        vector=vector,
                        count=count,
                        cout=cout,
                        products=indent("\n".join(expressions), '            '),
                        weights=weights,
                        wo=wo,
                        alpha=alpha,
                    )
                )
                previous = end
            source = (r"""
__global__ __launch_bounds__({threads}, 4) void {name}({parameters}, Element* __restrict__ out, Index num_items) {{
{path_extents}
    for (Index block = blockIdx.x; block < {previous}; block += gridDim.x) {{
{path_bodies}
    }}
}}
""").format(
                threads=threads,
                name=name,
                parameters=','.join(parameters),
                path_extents=indent("\n".join(declarations), '    '),
                previous=previous,
                path_bodies=indent("\n".join(blocks), '        '),
            )
            launch = (r"""if (num_items > 0) {{
{path_extents}
    unsigned int {name}_grid = static_cast<unsigned int>(std::min<int64_t>({previous}, 65535));
    {name}<<<{name}_grid, {threads}, 0, stream>>>({pointers}, out.data_ptr<Element>(), num_items);
}}""").format(
                path_extents=indent("\n".join(declarations), '    '),
                name=name,
                previous=previous,
                threads=threads,
                pointers=','.join(pointers),
            )
            return {"source": source, "pointer_order": pointer_order, "launch": launch}

        fp64 = meta.get("dtype") == torch.float64
        fma = "__fma_rn" if fp64 else "__fmaf_rn"
        weights = int(meta["weight_numel"])
        tile, threads, vector = 1024, 256, 4
        blocks = []
        cursor = 0
        for spec in meta["generated_specs"]:
            cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
            count = cin * cout
            ntile = (count + tile - 1) // tile
            xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
            alpha = float_literal(spec["path_weight"], meta.get("dtype"))
            expressions = []
            for component in range(d):
                for x, g in pairs:
                    expressions.append(
                        ("value={fma}({feature}[item*{in_dim}+{x_offset}+u*{components}+{component}],{seed}[item*{out_dim}+{out_offset}+v*{components}+{component}],value);").format(
                            fma=fma, feature=x, seed=g, in_dim=meta['in_dim'], out_dim=meta['out_dim'],
                            x_offset=xo, out_offset=go, components=d, component=component,
                        )
                    )
            blocks.append(
                (r"""if (task >= {cursor} && task < {end}) {{
    Index first = (task - {cursor}) * {tile} + threadIdx.x * {vector};
#pragma unroll
    for (int j = 0; j < {vector}; ++j) {{
        Index p = first + j;
        if (p < {count}) {{
            int u = p / {cout}, v = p % {cout};
            Element value = Element(0);
{products}
            out[item * {weights} + {wo} + p] = Element({alpha}) * value;
        }}
    }}
}}""").format(
                    cursor=cursor,
                    end=cursor + ntile,
                    tile=tile,
                    vector=vector,
                    count=count,
                    cout=cout,
                    products=indent("\n".join(expressions), '            '),
                    weights=weights,
                    wo=wo,
                    alpha=alpha,
                )
            )
            cursor += ntile
        source = (r"""
__global__ __launch_bounds__({threads}, 4) void {name}({parameters}, Element* __restrict__ out, Index num_items) {{
    Index total = num_items * {cursor};
    for (Index block = blockIdx.x; block < total; block += gridDim.x) {{
        Index item = block / {cursor}, task = block % {cursor};
{path_bodies}
    }}
}}
""").format(
            threads=threads,
            name=name,
            parameters=','.join(parameters),
            cursor=cursor,
            path_bodies=indent("\n".join(blocks), '        '),
        )
        launch = (r"""if (num_items > 0) {{
    Index {name}_blocks = num_items * {cursor};
    unsigned int {name}_grid = static_cast<unsigned int>(std::min<int64_t>({name}_blocks, 2147483647));
    {name}<<<{name}_grid, {threads}, 0, stream>>>({pointers}, out.data_ptr<Element>(), num_items);
}}""").format(
            name=name,
            cursor=cursor,
            threads=threads,
            pointers=','.join(pointers),
        )
        return {"source": source, "pointer_order": pointer_order, "launch": launch}

    threads = 256
    slots = 1 << (max(counts) - 1).bit_length()
    items_per_cta = threads // slots
    paths = len(specs)
    fma = "__fma_rn" if meta.get("dtype") == torch.float64 else "__fmaf_rn"
    weights = int(meta["weight_numel"])

    blocks = []
    for path, spec in enumerate(specs):
        cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
        count = cin * cout
        xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
        alpha = float_literal(spec["path_weight"], meta.get("dtype"))
        expressions = [
            ("value={fma}({feature}[item*{in_dim}+{x_offset}+u*{components}+{component}],{seed}[item*{out_dim}+{out_offset}+v*{components}+{component}],value);").format(
                            fma=fma, feature=x, seed=g, in_dim=meta['in_dim'], out_dim=meta['out_dim'],
                            x_offset=xo, out_offset=go, components=d, component=component,
                        )
            for component in range(d)
            for x, g in pairs
        ]
        blocks.append(
            (r"""if (task == {path} && q < {count} && item < num_items) {{
    int u = q / {cout}, v = q % {cout};
    Element value = Element(0);
{products}
    out[item * {weights} + {wo} + q] = Element({alpha}) * value;
}}""").format(
                path=path,
                count=count,
                cout=cout,
                products=indent("\n".join(expressions), '    '),
                weights=weights,
                wo=wo,
                alpha=alpha,
            )
        )

    source = (r"""
__global__ __launch_bounds__({threads}, 4) void {name}({parameters}, Element* __restrict__ out, Index num_items) {{
    Index total = ((num_items + {item_padding}) / {items_per_cta}) * {paths};
    for (Index block = blockIdx.x; block < total; block += gridDim.x) {{
        int task = block % {paths};
        Index item = (block / {paths}) * {items_per_cta} + threadIdx.x / {slots};
        int q = threadIdx.x % {slots};
{path_bodies}
    }}
}}
""").format(
        threads=threads,
        name=name,
        parameters=','.join(parameters),
        item_padding=items_per_cta - 1,
        items_per_cta=items_per_cta,
        paths=paths,
        slots=slots,
        path_bodies=indent("\n".join(blocks), '        '),
    )
    launch = (r"""if (num_items > 0) {{
    Index {name}_blocks = ((num_items + {item_padding}) / {items_per_cta}) * {paths};
    unsigned int {name}_grid = static_cast<unsigned int>(std::min<int64_t>({name}_blocks, 65535));
    {name}<<<{name}_grid, {threads}, 0, stream>>>({pointers}, out.data_ptr<Element>(), num_items);
}}""").format(
        name=name,
        item_padding=items_per_cta - 1,
        items_per_cta=items_per_cta,
        paths=paths,
        threads=threads,
        pointers=','.join(pointers),
    )
    return {"source": source, "pointer_order": pointer_order, "launch": launch}


def make_small_weight_kernel(meta, name, pairs, pointer_order, parameters, pointers):
    """Warp dot products avoid GEMM padding for small channel matrices."""
    fma = "__fma_rn"
    threads, lanes, tile = 256, 4, 64
    weights = int(meta["weight_numel"])
    specs = sorted(meta["generated_specs"], key=lambda spec: (-spec["components"], spec["weight_offset"]))
    tasks, gains, cursor = [], [], 0
    for spec in specs:
        cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
        count = cin * cout
        ntile = (count + tile - 1) // tile
        xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
        alpha = float_literal(spec["path_weight"], meta.get("dtype"))
        expressions = []
        for component in range(d):
            for x, g in pairs:
                expressions.append(
                    ("value={fma}({feature}[item*{in_dim}+{x_offset}+u*{components}+{component}],{seed}[item*{out_dim}+{out_offset}+v*{components}+{component}],value);").format(
                            fma=fma, feature=x, seed=g, in_dim=meta['in_dim'], out_dim=meta['out_dim'],
                            x_offset=xo, out_offset=go, components=d, component=component,
                        )
                )
        tasks.append(
            (r"""if (task >= {cursor} && task < {end}) {{
    Index p = (task - {cursor}) * {tile} + threadIdx.x / {lanes};
    const int lane = threadIdx.x % {lanes};
    Element value = Element(0);
    if (p < {count}) {{
        int u = p / {cout}, v = p % {cout};
        for (Index item = begin + lane; item < end; item += {lanes}) {{
{products}
        }}
    }}
#pragma unroll
    for (int step = {reduction_step}; step > 0; step >>= 1)
        value += __shfl_down_sync(0xffffffff, value, step, {lanes});
    if (lane == 0 && p < {count})
        out[Index(split) * {weights} + {wo} + p] = splits == 1 ? Element({alpha}) * value : value;
}}""").format(
                cursor=cursor,
                end=cursor + ntile,
                tile=tile,
                lanes=lanes,
                count=count,
                cout=cout,
                products=indent("\n".join(expressions), '            '),
                reduction_step=lanes // 2,
                weights=weights,
                wo=wo,
                alpha=alpha,
            )
        )
        gains.append(f"if(index>={wo} && index<{wo + count}) value*=Element({alpha});")
        cursor += ntile

    source = (r"""
__global__ __launch_bounds__({threads}, 4) void {name}({parameters}, Element* __restrict__ out, Index num_items, int splits) {{
    Index task = blockIdx.x / splits;
    int split = blockIdx.x % splits;
    Index per_split = (num_items + splits - 1) / splits;
    Index begin = Index(split) * per_split, end = min(num_items, begin + per_split);
{path_tasks}
}}

__global__ __launch_bounds__(256, 4) void {name}_reduce(
    const Element* __restrict__ partial, Element* __restrict__ out, int splits) {{
    Index index = Index(blockIdx.x) * 256 + threadIdx.x;
    if (index < {weights}) {{
        Element value = Element(0);
        for (int split = 0; split < splits; ++split)
            value += partial[Index(split) * {weights} + index];
{path_gains}
        out[index] = value;
    }}
}}
""").format(
        threads=threads,
        name=name,
        parameters=','.join(parameters),
        path_tasks=indent("\n".join(tasks), '    '),
        weights=weights,
        path_gains=indent("\n".join(gains), '        '),
    )

    target_splits = min(64, max(1, (432 + cursor - 1) // cursor))
    partial_initialization = "".join(
        f"        {name}_partial.narrow(-1,{offset},{count}).zero_();\n"
        for offset, count in meta.get("weight_gaps", ())
    )
    launch = (r"""if (num_items == 0) {{
    out.zero_();
}} else {{
    int {name}_splits = static_cast<int>(std::min<int64_t>({target_splits}, (num_items + 31) / 32));
    if ({name}_splits == 1) {{
        {name}<<<{cursor}, {threads}, 0, stream>>>({pointers}, out.data_ptr<Element>(), num_items, 1);
    }} else {{
        auto {name}_partial = torch::empty({{{name}_splits, {weights}}}, out.options());
        {partial_initialization} {name}<<<{cursor} * {name}_splits, {threads}, 0, stream>>>(
            {pointers}, {name}_partial.data_ptr<Element>(), num_items, {name}_splits);
        {name}_reduce<<<{reduction_blocks}, 256, 0, stream>>>(
            {name}_partial.data_ptr<Element>(), out.data_ptr<Element>(), {name}_splits);
    }}
}}""").format(
        name=name,
        target_splits=target_splits,
        cursor=cursor,
        threads=threads,
        pointers=','.join(pointers),
        weights=weights,
        partial_initialization=partial_initialization,
        reduction_blocks=(weights + 255) // 256,
    )
    return {"source": source, "pointer_order": pointer_order, "launch": launch}


def make_fp64_weight_kernel(
    meta,
    name,
    pairs,
    pointer_order,
    parameters,
    pointers,
    *,
    arch,
    tile_shape,
    static_path=False,
):
    compute_pairs = pairs
    specs = tuple(meta["generated_specs"])
    tile_m, tile_n, tile_k, threads = tile_shape
    xroles = tuple(dict.fromkeys(x for x, _ in compute_pairs))
    groles = tuple(dict.fromkeys(g for _, g in compute_pairs))
    xindex, gindex = ({role: i for i, role in enumerate(roles)} for roles in (xroles, groles))
    weights = int(meta["weight_numel"])
    tasks, gains, plans = [], [], []
    min_components = min(int(spec["components"]) for spec in specs)
    for spec in specs:
        cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
        rows, cols = (cin + tile_m - 1) // tile_m, (cout + tile_n - 1) // tile_n
        plans.append((spec, rows * cols, cols))
    output_tiles = sum(count for _, count, _ in plans)
    weighted_tiles = sum(count * int(spec["components"]) for spec, count, _ in plans)
    base_splits = max(1, (432 + weighted_tiles - 1) // weighted_tiles)
    caps = {
        int(spec["components"]): (
            1
            if output_tiles >= 216
            else min(MAX_WEIGHT_SPLITS, base_splits * int(spec["components"]))
        )
        for spec in specs
    }
    max_splits = max(caps.values())
    limit = f"(num_items*{min_components}+{tile_k - 1})/{tile_k}"
    grid_terms = []
    for path, (spec, count, cols) in enumerate(plans):
        cin, cout, d = (int(spec[k]) for k in ("in_channels", "out_channels", "components"))
        xo, go, wo = (int(spec[k]) for k in ("x_offset", "out_offset", "weight_offset"))
        alpha = float_literal(spec["path_weight"], meta.get("dtype"))
        cap = caps[d]
        if static_path:
            first_u = "0" if (cin + tile_m - 1) // tile_m == 1 else f"(tile/{cols})*{tile_m}"
            first_v = "0" if cols == 1 else f"(tile%{cols})*{tile_n}"
            tasks.append(
                (r"""const int path_splits_{path} = int(min(Index({cap}), limit));
const Index path_tasks_{path} = Index({count}) * path_splits_{path};
if (task >= cursor && task < cursor + path_tasks_{path}) {{
    const int splits = path_splits_{path};
    const Index tile = (task - cursor) / splits;
    const int split = (task - cursor) % splits;
    {name}_path_body<{path}>({pointers}, out, shared_a, shared_b, num_items, write_output, {first_u}, {first_v}, split, splits, 0);
    return;
}}
cursor += path_tasks_{path};""").format(
                    path=path,
                    cap=cap,
                    count=count,
                    name=name,
                    pointers=','.join(pointer_order),
                    first_u=first_u,
                    first_v=first_v,
                )
            )
        else:
            tasks.append(
                (r"""const int path_splits_{path} = int(min(Index({cap}), limit));
const Index path_tasks_{path} = Index({count}) * path_splits_{path};
if (task >= cursor && task < cursor + path_tasks_{path}) {{
    splits = path_splits_{path};
    Index tile = (task - cursor) / splits;
    split = (task - cursor) % splits;
    first_u = (tile / {cols}) * {tile_m};
    first_v = (tile % {cols}) * {tile_n};
    in_channels = {cin};
    out_channels = {cout};
    components = {d};
    x_offset = {xo};
    g_offset = {go};
    weight_offset = {wo};
    alpha = Element({alpha});
}}
cursor += path_tasks_{path};""").format(
                    path=path,
                    cap=cap,
                    count=count,
                    cols=cols,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    cin=cin,
                    cout=cout,
                    d=d,
                    xo=xo,
                    go=go,
                    wo=wo,
                    alpha=alpha,
                )
            )
        gains.append(
            f"if(index>={wo} && index<{wo + cin * cout}) {{splits=int(min(Index({cap}),limit));alpha=Element({alpha});}}"
        )
        grid_terms.append(f"int64_t({count})*std::min<int64_t>({cap},{limit})")
    header = ""
    tile_initializer = ""
    warp_m, warp_n = tile_m // 2, tile_n // 2
    m_count, n_count = warp_m // 16, warp_n // 8
    mma_header, mma_type = get_fp64_mma(str(arch))
    # Index variants live in separate namespaces.  A preprocessor guard
    # would suppress the helper definition in the second namespace.
    if mma_type == "Fp64Mma16x8x16":
        unique_type = name + "_Fp64Mma16x8x16"
        mma_header = mma_header.replace(mma_type, unique_type)
        mma_type = unique_type
    header = "\n" + mma_header + "\n"
    compute = (r"""const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
const int local_row = (warp >> 1) * {warp_m} + (lane >> 2), local_column = (warp & 1) * {warp_n};
Element acc[{m_count}][{n_count}][4] = {{}};""").format(
        warp_m=warp_m, warp_n=warp_n, m_count=m_count, n_count=n_count,
    )
    mma = (r"""#pragma unroll
for (int m = 0; m < {m_count}; ++m) {{
{feature_loads}
#pragma unroll
    for (int n = 0; n < {n_count}; ++n) {{
{seed_loads}
{products}
    }}
}}""").format(
        m_count=m_count, n_count=n_count,
        feature_loads=indent("\n".join(
            (r"""Element a{index}[8];
#pragma unroll
for (int i = 0; i < 8; ++i)
    a{index}[i] = shared_a[{index}][((lane & 3) + (i >> 1) * 4) * {tile_m} + local_row + m * 16 + (i & 1) * 8];""").format(index=index, tile_m=tile_m)
            for index in range(len(xroles))
        ), '    '),
        seed_loads=indent("\n".join(
            (r"""Element b{index}[4];
#pragma unroll
for (int i = 0; i < 4; ++i)
    b{index}[i] = shared_b[{index}][((lane & 3) + i * 4) * {tile_n} + local_column + n * 8 + (lane >> 2)];""").format(index=index, tile_n=tile_n)
            for index in range(len(groles))
        ), '        '),
        products=indent("\n".join(
            ("{mma_type}::fma(acc[m][n][0],acc[m][n][1],acc[m][n][2],acc[m][n][3],{a}[0],{a}[1],{a}[2],{a}[3],{a}[4],{a}[5],{a}[6],{a}[7],{b}[0],{b}[1],{b}[2],{b}[3],acc[m][n][0],acc[m][n][1],acc[m][n][2],acc[m][n][3]);").format(mma_type=mma_type, a=f'a{xindex[x]}', b=f'b{gindex[g]}')
            for x, g in compute_pairs
        ), '        '),
    )
    stores = (r"""
#pragma unroll
for (int m = 0; m < {m_count}; ++m) {{
#pragma unroll
    for (int n = 0; n < {n_count}; ++n) {{
#pragma unroll
        for (int i = 0; i < 4; ++i) {{
            int u = first_u + (warp >> 1) * {warp_m} + m * 16 + (lane >> 2) + (i >> 1) * 8;
            int v = first_v + (warp & 1) * {warp_n} + n * 8 + (lane & 3) * 2 + (i & 1);
            if (u < in_channels && v < out_channels) {{
                Element value = acc[m][n][i];
                out[Index(split) * {weights} + weight_offset + Index(u) * out_channels + v] =
                    write_output ? alpha * value : value;
            }}
        }}
    }}
}}
""").format(
        m_count=m_count,
        n_count=n_count,
        warp_m=warp_m,
        warp_n=warp_n,
        weights=weights,
    )
    load_cases = []
    for d in sorted({int(spec["components"]) for spec in specs}):
        x_values = [f"{role}[item*{meta['in_dim']}+x_offset+u*{d}+component]" for role in xroles]
        load_a = "\n".join(
            ("shared_a[{index}][index]=u<in_channels && reduction<k_end?{value}:Element(0);").format(index=i, value=value)
            for i, value in enumerate(x_values)
        )
        load_b = "\n".join(
            ("shared_b[{index}][index]=v<out_channels && reduction<k_end?{role}[item*{out_dim}+g_offset+v*{components}+component]:Element(0);").format(index=i, role=role, out_dim=meta['out_dim'], components=d)
            for i, role in enumerate(groles)
        )
        load_cases.append(
            (r"""case {d}: {{
    for (int index = threadIdx.x; index < {a_elements}; index += {threads}) {{
        int u = first_u + index % {tile_m};
        Index reduction = tile_k + index / {tile_m};
        Index item = reduction / {d};
        int component = reduction % {d};
{load_a}
    }}
    for (int index = threadIdx.x; index < {b_elements}; index += {threads}) {{
        int v = first_v + index % {tile_n};
        Index reduction = tile_k + index / {tile_n};
        Index item = reduction / {d};
        int component = reduction % {d};
{load_b}
    }}
    break;
}}""").format(
                d=d,
                a_elements=tile_m * tile_k,
                threads=threads,
                tile_m=tile_m,
                load_a=indent(load_a, '        '),
                b_elements=tile_k * tile_n,
                tile_n=tile_n,
                load_b=indent(load_b, '        '),
            )
        )
    reduce_sum = (r"""Element value = Element(0);
for (int split = 0; split < splits; ++split)
    value += partial[Index(split) * {weights} + index];""").format(
        weights=weights,
    )
    reduce_output = "alpha*value"
    task_assignment = f"Index task=blockIdx.x,cursor=0,limit={limit};int split=0,splits=1;"
    math_body = (r"""Index reduction_size = num_items * components;
Index tiles = (reduction_size + {tile_padding}) / {tile_k};
Index quotient = tiles / splits, remainder = tiles % splits;
Index k_begin = (Index(split) * quotient + min(Index(split), remainder)) * {tile_k};
Index k_end = min(reduction_size, (Index(split + 1) * quotient + min(Index(split + 1), remainder)) * {tile_k});
{accumulators}
for (Index tile_k = k_begin; tile_k < k_end; tile_k += {tile_k}) {{
    {tile_initializer}switch (components) {{
{tile_loads}
    }}
    __syncthreads();
{multiply_accumulate}
    __syncthreads();
}}
{stores}""").format(
        tile_padding=tile_k - 1,
        tile_k=tile_k,
        accumulators=compute,
        tile_initializer=tile_initializer,
        tile_loads=indent("\n".join(load_cases), "        "),
        multiply_accumulate=indent(mma, '    '),
        stores=stores,
    )
    if static_path:
        task_assignment = f"Index task=blockIdx.x,cursor=0,limit={limit};"
    prefix = (r"""__shared__ Element shared_a[{feature_count}][{a_elements}];
__shared__ Element shared_b[{seed_count}][{b_elements}];
{task_assignment}
""").format(
        feature_count=len(xroles),
        a_elements=tile_m * tile_k,
        seed_count=len(groles),
        b_elements=tile_k * tile_n,
        task_assignment=task_assignment,
    )
    if static_path:
        constexprs = []
        for cpp_name, key in (
            ("in_channels", "in_channels"),
            ("out_channels", "out_channels"),
            ("components", "components"),
            ("x_offset", "x_offset"),
            ("g_offset", "out_offset"),
            ("weight_offset", "weight_offset"),
        ):
            values = ",".join(str(int(s[key])) for s in specs)
            constexprs += [
                f"constexpr int path_{cpp_name}[]={{{values}}};",
                f"constexpr int {cpp_name}=path_{cpp_name}[Path];",
            ]
        alphas = ",".join(
            f"Element({float_literal(s['path_weight'], meta['dtype'])})" for s in specs
        )
        constexprs += [
            f"constexpr Element path_alpha[]={{{alphas}}};",
            "constexpr Element alpha=path_alpha[Path];",
        ]
        path_body = (r"""template<int Path>
__device__ __forceinline__ void {name}_path_body(
    {parameters},Element* __restrict__ out,Element(*shared_a)[{a_elements}],Element(*shared_b)[{b_elements}],
    Index num_items,bool write_output,int first_u,int first_v,int split,int splits,int term) {{
{path_parameters}
{math_body}
}}
""").format(
            name=name,
            parameters=','.join(parameters),
            a_elements=tile_m * tile_k,
            b_elements=tile_k * tile_n,
            path_parameters=indent("\n".join(constexprs), '    '),
            math_body=indent(math_body, "    "),
        )
        kernel_body = prefix + "\n".join(tasks)
    else:
        path_body = ""
        kernel_body = (
            prefix
            + (r"""int first_u = 0, first_v = 0, in_channels = 0, out_channels = 0, components = 1;
int x_offset = 0, g_offset = 0, weight_offset = 0;
Element alpha = Element(0);
{path_tasks}
{math_body}""").format(
                path_tasks="\n".join(tasks),
                math_body=math_body,
            )
        )
    source = (
        header
        + (r"""
{path_body}__global__ __launch_bounds__({threads}, 4) void {name}(
    {parameters}, Element* __restrict__ out, Index num_items, bool write_output) {{
{kernel_body}
}}

__global__ __launch_bounds__(256, 4) void {name}_reduce(
    const Element* __restrict__ partial, Element* __restrict__ out, Index num_items) {{
    Index index = Index(blockIdx.x) * 256 + threadIdx.x;
    if (index < {weights}) {{
        Index limit = {limit};
        int splits = 0;
        Element alpha = Element(0);
{path_gains}
{reduce_sum}
        out[index] = {reduce_output};
    }}
}}
""").format(
            path_body=path_body,
            threads=threads,
            name=name,
            parameters=','.join(parameters),
            kernel_body=indent(kernel_body, '    '),
            weights=weights,
            limit=limit,
            path_gains=indent("\n".join(gains), '        '),
            reduce_sum=indent(reduce_sum, '        '),
            reduce_output=reduce_output,
        )
    )
    partial_initialization = "".join(
        f"        {name}_partial.narrow(-1,{offset},{count}).zero_();\n"
        for offset, count in meta.get("weight_gaps", ())
    )
    launch = (r"""if (num_items == 0) {{
    out.zero_();
}} else {{
    int {name}_splits = static_cast<int>(std::min<int64_t>({max_splits}, {limit}));
    int64_t {name}_blocks = {grid_terms};
    if ({name}_splits == 1) {{
        {name}<<<{name}_blocks, {threads}, 0, stream>>>({pointers}, out.data_ptr<Element>(), num_items, true);
    }} else {{
        auto {name}_partial = torch::empty({{{name}_splits, {weights}}}, out.options());
        {partial_initialization} {name}<<<{name}_blocks, {threads}, 0, stream>>>({pointers}, {name}_partial.data_ptr<Element>(), num_items, false);
        {name}_reduce<<<{reduction_blocks}, 256, 0, stream>>>({name}_partial.data_ptr<Element>(), out.data_ptr<Element>(), num_items);
    }}
}}""").format(
        name=name,
        max_splits=max_splits,
        limit=limit,
        grid_terms='+'.join(grid_terms),
        threads=threads,
        pointers=','.join(pointers),
        weights=weights,
        partial_initialization=partial_initialization,
        reduction_blocks=(weights + 255) // 256,
    )
    return {"source": source, "pointer_order": pointer_order, "launch": launch}


def make_bias_kernel(meta, name, terms):
    roles = tuple(dict.fromkeys(role for term in terms for role in term))
    lines = []
    for spec in meta["bias_specs"]:
        start, count = spec["bias_offset"], spec["channels"]
        for (g,) in terms:
            lines += [
                ("if(q>={start} && q<{end}) for(Index n=threadIdx.x;n<num_items;n+=blockDim.x) value+={role}[n*{out_dim}+q-{start}+{out_offset}];").format(start=start, end=start+count, role=g, out_dim=meta['out_dim'], out_offset=spec['out_offset'])
            ]
    parameters = ", ".join(f"const Element* __restrict__ {role}" for role in roles)
    code = (r"""
__global__ void {name}({parameters}, Element* out, int64_t num_items) {{
    const int q = blockIdx.x;
    Element value = Element(0);
{contractions}
    __shared__ Element shared[256];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {{
        if (threadIdx.x < step)
            shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }}
    if (threadIdx.x == 0)
        out[q] = shared[0];
}}
""").format(
        name=name,
        parameters=parameters,
        contractions=indent("\n".join(lines), '    '),
    )
    args = ", ".join(f"{r}.data_ptr<Element>()" for r in roles)
    return {
        "source": code,
        "launch": f"if({meta['bias_numel']}>0) {name}<<<{meta['bias_numel']},256,0,stream>>>({args},out.data_ptr<Element>(),num_items);",
    }


def make_backward_source(meta, program, *, arch):
    sources = []
    launches = []
    allocations = []
    returns = []
    kinds = {
        name: kind
        for kind in ("input", "weight", "output", "bias")
        for name in program.get(kind, {})
    }
    kinds.update({name: "output" for name in program.get("broadcast", {})})
    for slot, name in enumerate(program["outputs"]):
        kind = kinds.get(name)
        if kind is None:
            returns.append("pybind11::none()")
            continue
        pairs = program.get(kind, {}).get(name, ())
        phase = ('forward', 'backward', 'double_backward', 'triple_backward')[program['order']]
        kernel = f"linear_{phase}_{name}"
        if kind in ("input", "output"):
            broadcast = program.get("broadcast", {}).get(name, ())
            if meta["dtype"] == torch.float32:
                emitted = make_cutlass_feature_kernel(
                    meta, kernel, pairs, broadcast, backward=kind == "input",
                )
            elif meta["shared_weights"] and meta["max_channels"] >= 32:
                emitted = make_tiled_feature_kernel(
                    meta, kernel, pairs, broadcast, backward=kind == "input", arch=arch,
                )
            else:
                emitted = make_feature_kernel(
                    meta, kernel, pairs, broadcast, backward=kind == "input",
                )
            dim = meta["in_dim" if kind == "input" else "out_dim"]
            shape = f"{{num_items,int64_t({dim})}}"
        elif kind == "weight":
            emitted = make_weight_kernel(meta, kernel, pairs, arch=arch)
            shape = (
                f"{{int64_t({meta['weight_numel']})}}"
                if meta["shared_weights"]
                else f"{{num_items,int64_t({meta['weight_numel']})}}"
            )
        else:
            emitted = make_bias_kernel(meta, kernel, pairs)
            shape = f"{{int64_t({meta['bias_numel']})}}"
        sources.append(emitted["source"])
        initialization = (
            "".join(
                f"result_{slot}.narrow(-1,{offset},{count}).zero_();\n"
                for offset, count in meta.get("weight_gaps", ())
            ) if kind == "weight" and not (
                meta["dtype"] == torch.float32 and meta["shared_weights"]
                and meta["generated_specs"]
            ) else ""
        )
        allocations.append(f"auto result_{slot}=torch::empty({shape},options);")
        launches.append(f"{initialization}{{\n    auto out=result_{slot};\n"
                        + indent(emitted["launch"], "    ") + "\n}")
        returns.append(f"pybind11::cast(result_{slot})")
    stream_source = ""
    if len(launches) > 1:
        workers = len(launches) - 1
        lane_declaration = ""
        lane_index = "i"
        if meta["dtype"] is torch.float32:
            worker_lanes = [
                int(kinds[name] == "weight")
                for name in program["outputs"]
                if name in kinds
            ][:-1]
            lane_indices = ", ".join(map(str, worker_lanes))
            lane_declaration = f"    const int lane_indices[{workers}] = {{{lane_indices}}};\n"
            lane_index = "lane_indices[i]"
        stream_source = STREAM_SOURCE
        branches = []
        for index, code in enumerate(launches[:-1]):
            branches.append(f"""{{
    auto& lane = *lanes[{index}];
    auto worker = c10::cuda::getStreamFromExternal(lane.stream, current.device_index());
    C10_CUDA_CHECK(cudaStreamWaitEvent(worker, start.ready, 0));
    launched = {index + 1};
    c10::cuda::CUDAStreamGuard worker_guard(worker);
    cudaStream_t stream = worker;
{indent(code, '    ')}
    C10_CUDA_CHECK(cudaEventRecord(lane.done, worker));
}}""")
        launches = [f"""
DerivativeStream* lanes[{workers}];
{lane_declaration}for (int i = 0; i < {workers}; ++i)
    lanes[i] = &derivative_stream(current.device_index(), current, {lane_index});
auto& start = *lanes[0];
C10_CUDA_CHECK(cudaEventRecord(start.ready, current));
int launched = 0;
try {{
{indent(chr(10).join(branches), '    ')}
{indent(launches[-1], '    ')}
    for (int i = 0; i < {workers}; ++i)
        C10_CUDA_CHECK(cudaStreamWaitEvent(current, lanes[i]->done, 0));
}} catch (...) {{
    // Allocation failures must join workers before caller-owned outputs die.
    for (int i = 0; i < launched; ++i) {{
        auto& lane = *lanes[i];
        cudaEventRecord(lane.done, lane.stream);
        cudaStreamWaitEvent(current, lane.done, 0);
    }}
    throw;
}}
"""]
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
{stream_source}
using Element = {scalar};
{kernels}
""").format(
        scalar=scalar,
        stream_source=stream_source,
        kernels="\n".join(sources),
    )
    body = (r"""
const c10::cuda::CUDAGuard guard({first}.device());
auto options = {first}.options();
auto current = at::cuda::getCurrentCUDAStream();
cudaStream_t stream = current;
{allocations}
{launches}
C10_CUDA_KERNEL_LAUNCH_CHECK();
return pybind11::make_tuple({results});
""").format(
        first=first,
        allocations="\n".join(allocations),
        launches="\n".join(launches),
        results=','.join(returns),
    )
    return (
        prefix
        + f"pybind11::tuple launch({parameters},int64_t num_items) {{\n"
        + indent(body.lstrip("\n"), "    ") + "}\n"
        + 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n    m.def("launch",&launch);\n}\n'
    )
