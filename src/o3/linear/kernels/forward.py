"""Linear forward CUDA kernels and bindings."""

from __future__ import annotations

from collections import defaultdict

import torch

from ....utils.precision import get_fp64_mma, float_literal
from .utils import MAX_BATCH_RANK


def make_feature_kernel(meta, name, pairs, broadcast, *, backward=False, layout=False):
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
const int channel=(q-{offset})/{components};
const int component=(q-{offset})%{components};""").format(
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
                feature = (
                    f"layout_{a}({a},{feature_index},layout)" if layout else f"{a}[{feature_index}]"
                )
                weight = (
                    f"layout_{w}({w},{weight_index},layout)" if layout else f"{w}[{weight_index}]"
                )
                lines.append((r"""{{ Element part=Element(0);
#pragma unroll {unroll}
for(int k=0;k<{width};++k) part={fma}({feature}, {weight}, part);
value={fma}(part,Element({coefficient}),value); }}""").format(
                    unroll=1 if width>32 else 4, width=width,
                    fma='__fma_rn' if meta['dtype'] is torch.float64 else '__fmaf_rn',
                    feature=feature, weight=weight, coefficient=coeff,
                ))
        lines.append("}")
    if not backward:
        for spec in meta["bias_specs"]:
            start, count = spec["out_offset"], spec["channels"]
            for (b,) in broadcast:
                bias_index = f"q-{start}+{spec['bias_offset']}"
                bias = f"layout_{b}({b},{bias_index},layout)" if layout else f"{b}[{bias_index}]"
                lines.append(("if(q>={start} && q<{end}) value+={bias};").format(start=start, end=start+count, bias=bias))

    code = (
        (r"""
__global__ void {name}({parameters}{parameter_separator} Element* __restrict__ out, Index count{layout_parameter}) {{
    for(Index index=Index(blockIdx.x)*blockDim.x+threadIdx.x;index<count;index+=Index(blockDim.x)*gridDim.x) {{
        const Index n=index/{dimension};
        const int q=index%{dimension};
        Element value=Element(0);
        {contractions}
        out[index]=value;
    }}
}}
""").format(
            name=name,
            parameters=parameters,
            parameter_separator=',' if roles else '',
            layout_parameter=',ForwardLayout layout' if layout else '',
            dimension=meta[dest],
            contractions="\n".join(lines),
        )
        if meta[dest]
        else ""
    )

    args = ", ".join(f"{role}.data_ptr<Element>()" for role in roles)
    count = f"num_items*int64_t({meta[dest]})"
    launch = (
        f"if ({count}>0) {name}<<<std::min<int64_t>(65535,({count}+255)/256),256,0,stream>>>({args}{',' if args else ''} out.data_ptr<Element>(),{count}{',layout_' if layout else ''});"
        if code
        else ""
    )
    return {"source": code, "launch": launch}


def make_tiled_feature_kernel(
    meta, name, pairs, broadcast, *, backward=False, arch, layout=False
):
    """Implicit AoS GEMM with static path bodies and one CTA dispatcher."""
    fp64 = meta["dtype"] is torch.float64
    tile_m, tile_n, tile_k, threads = (32, 64, 16, 128) if fp64 else (32, 64, 32, 128)
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
            ("if(task<cursor+({tasks})) {{{name}_body<{group}>({body_args}{separator}out,shared_a,shared_b,((task-cursor)/{tiles})*{tile_m},((task-cursor)%{tiles})*{tile_n},num_items{layout_argument}); return;}} cursor+=({tasks});").format(
                tasks=tasks, name=name, group=group, body_args=body_args,
                separator=',' if roles else '', tiles=nt, tile_m=tile_m, tile_n=tile_n,
                layout_argument=',layout' if layout else '',
            )
        )
        la = [(r"""case {group}: {{
if(row>=num_items*int64_t({components})) return Element(0);
const Index n=row/{components}; const int component=row%{components};""").format(group=group, components=d)]
        lb = [(r"""case {group}: {{
if(column>={channels}) return Element(0);""").format(group=group, channels=channels)]
        base = 0
        for spec in specs:
            count = spec["out_channels" if backward else "in_channels"]
            so = spec["out_offset" if backward else "x_offset"]
            wo, cout = spec["weight_offset"], spec["out_channels"]
            wi = f"column*{cout}+local_k" if backward else f"local_k*{cout}+column"
            feature_index = f"n*{src}+{so}+local_k*{d}+component"
            weight_index = f"{wo}+{wi}"
            feature = f"layout_x(a,{feature_index},layout)" if layout else f"a[{feature_index}]"
            weight = f"layout_weight(b,{weight_index},layout)" if layout else f"b[{weight_index}]"
            la += [
                ("if(k>={base} && k<{end}) {{const int local_k=k-{base}; return {feature};}}").format(base=base, end=base+count, feature=feature)
            ]
            lb += [
                ("if(k>={base} && k<{end}) {{const int local_k=k-{base}; return {weight}*Element({coefficient});}}").format(base=base, end=base+count, weight=weight,
                                        coefficient=float_literal(spec['path_weight'], meta['dtype']))
            ]
            base += count
        la += ["return Element(0);}"]
        lb += ["return Element(0);}"]
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
                    bias = (
                        f"layout_{role}({role},{bias_index},layout)" if layout else f"{role}[{bias_index}]"
                    )
                    bias_add.append(
                        ("if(group=={group} && column*components+component>={start} && column*components+component<{end}) value+={bias};").format(group=group, start=bs['out_offset']-offset,
                                                 end=bs['out_offset']-offset+bs['channels'], bias=bias)
                    )

    layout_parameter = ",const ForwardLayout& layout" if layout else ""
    layout_argument = ",layout" if layout else ""
    loader_args = "const Element* a,int group,Index row,int k,int64_t num_items" + layout_parameter
    header = ""
    if fp64:
        header, mma = get_fp64_mma(arch)
        unique = name + "_Mma"
        header = header.replace("Fp64Mma16x8x16", unique)
        mma = mma.replace("Fp64Mma16x8x16", unique)
        init = ("Element acc[4][4]={}; const int lane=threadIdx.x&31,warp=threadIdx.x>>5;")
        calculate = (r"""
        #pragma unroll
        for(int k=0;k<{tile_k};k+=16) {{
            Element a[8];
            #pragma unroll
            for(int i=0;i<8;++i) a[i]=shared_a[((warp>>1)*16+(lane>>2)+(i&1)*8)*{tile_k}+((k+(lane&3)+(i>>1)*4)^(((lane>>2)&3)<<2))];
            #pragma unroll
            for(int n=0;n<4;++n) {{
                Element b[4];
                #pragma unroll
                for(int i=0;i<4;++i) b[i]=shared_b[(k+(lane&3)+i*4)*{tile_n}+(((warp&1)*32+n*8+(lane>>2))^((lane&3)<<2))];
                {mma}::fma(acc[n][0],acc[n][1],acc[n][2],acc[n][3],a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],b[0],b[1],b[2],b[3],acc[n][0],acc[n][1],acc[n][2],acc[n][3]);
            }}
        }}
""").format(
            tile_k=tile_k,
            tile_n=tile_n,
            mma=mma,
        )
        loops = ("for(int j=0;j<4;++j) { for(int i=0;i<4;++i) {")
        coords = ("const Index row=tile_row+(warp>>1)*16+(lane>>2)+(i>>1)*8; const int column=tile_column+(warp&1)*32+j*8+(lane&3)*2+(i&1); Element value=acc[j][i];")
    else:
        init = ("Element acc[4][4]={}; const int local_row=(threadIdx.x>>4)*4,local_column=(threadIdx.x&15);")
        shared_b_index = (
            f"k*{tile_n}+((local_column+j*16)^k)" if backward
            else f"k*{tile_n}+local_column+j*16"
        )
        calculate = (r"""
        #pragma unroll
        for(int k=0;k<{tile_k};++k) {{
            Element a[4],b[4];
            #pragma unroll
            for(int i=0;i<4;++i) a[i]=shared_a[(local_row+i)*{tile_k}+(k^(((local_row+i)&3)<<3))];
            #pragma unroll
            for(int j=0;j<4;++j) b[j]=shared_b[{shared_b_index}];
            #pragma unroll
            for(int i=0;i<4;++i) {{
                #pragma unroll
                for(int j=0;j<4;++j) acc[i][j]=__fmaf_rn(a[i],b[j],acc[i][j]);
            }}
        }}
""").format(
            tile_k=tile_k,
            tile_n=tile_n,
            shared_b_index=shared_b_index,
        )
        loops = ("for(int i=0;i<4;++i) { for(int j=0;j<4;++j) {")
        coords = ("const Index row=tile_row+local_row+i; const int column=tile_column+local_column+j*16; Element value=acc[i][j];")
    a_store = ("(index/{tile_k})*{tile_k}+((index%{tile_k})^(((index/{tile_k})&3)<<{shift}))").format(tile_k=tile_k, shift=2 if fp64 else 3)
    b_store = (
        ("(index/{tile_n})*{tile_n}+((index%{tile_n})^(((index/{tile_n})&3)<<2))").format(tile_n=tile_n)
        if fp64
        else "index"
    )

    b_k, b_column = f"index/{tile_n}", f"index%{tile_n}"
    if not fp64 and backward:
        # Read transposed weights along contiguous K, then transpose in shared memory.
        b_k, b_column = f"index%{tile_k}", f"index/{tile_k}"
        b_store = f"(index%{tile_k})*{tile_n}+((index/{tile_k})^(index%{tile_k}))"

    products = []
    for a, b in pairs:
        products.append(
            (r"""
        for(Index first_k=0;first_k<reduction_size;first_k+={tile_k}) {{
            for(int index=threadIdx.x;index<{a_elements};index+={threads}) shared_a[{a_store}]={name}_a({a},group,tile_row+index/{tile_k},first_k+index%{tile_k},num_items{layout_argument});
            for(int index=threadIdx.x;index<{b_elements};index+={threads}) shared_b[{b_store}]={name}_b({b},group,first_k+{b_k},tile_column+{b_column}{layout_argument});
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
                layout_argument=layout_argument,
                b_elements=tile_k * tile_n,
                b_store=b_store,
                b_k=b_k,
                b_column=b_column,
                b=b,
                tile_n=tile_n,
                calculate=calculate,
            )
        )

    total = "+".join(f"({t})" for t in totals) or "0"
    source = (
        header
        + (r"""
__device__ __forceinline__ Element {name}_a({loader_args}) {{ switch(group) {{{feature_loads} default:return Element(0);}} }}
__device__ __forceinline__ Element {name}_b(const Element* b,int group,int k,int column{layout_parameter}) {{ switch(group) {{{weight_loads} default:return Element(0);}} }}
template<int Group>
__device__ __forceinline__ void {name}_body({parameters}{parameter_separator} Element* out,Element* shared_a,Element* shared_b,Index tile_row,int tile_column,int64_t num_items{layout_parameter}) {{
    constexpr int group_widths[]={{{group_widths}}};
    constexpr int group_components[]={{{group_components}}};
    constexpr int group_offsets[]={{{group_offsets}}};
    constexpr int group_reductions[]={{{group_reductions}}};
    constexpr int group=Group,width=group_widths[Group],components=group_components[Group];
    constexpr int out_offset=group_offsets[Group],reduction_size=group_reductions[Group];
    {init}
    {products}
    #pragma unroll
    {loops}
        {coords}
        if(row<num_items*int64_t(components) && column<width) {{
            const Index item=row/components;
            const int component=row%components;
            {bias_add}
            out[item*{dest}+out_offset+column*components+component]=value;
        }}
    }} }}
}}
__global__ __launch_bounds__({threads},2) void {name}({parameters}{parameter_separator} Element* out,int64_t num_items{kernel_layout_parameter}) {{
    __shared__ Element shared_a[{a_elements}],shared_b[{b_elements}];
    const Index task=blockIdx.x;
    Index cursor=0;
    {task_dispatch}
}}
""").format(
            name=name,
            loader_args=loader_args,
            feature_loads="\n".join(loads_a),
            layout_parameter=layout_parameter,
            weight_loads="\n".join(loads_b),
            parameters=parameters,
            parameter_separator=',' if parameters else '',
            group_widths=','.join(map(str, widths)),
            group_components=','.join(map(str, component_counts)),
            group_offsets=','.join(map(str, offsets)),
            group_reductions=','.join(map(str, reductions)),
            init=init,
            products="\n".join(products),
            loops=loops,
            coords=coords,
            bias_add="\n".join(bias_add),
            dest=dest,
            threads=threads,
            kernel_layout_parameter=',ForwardLayout layout' if layout else '',
            a_elements=tile_m * tile_k,
            b_elements=tile_k * tile_n,
            task_dispatch="\n".join(dispatch),
        )
    )

    args = ", ".join(f"{r}.data_ptr<Element>()" for r in roles)
    launch = f"if(num_items>0 && {dest}>0) {name}<<<{total},{threads},0,stream>>>({args}{',' if args else ''}out.data_ptr<Element>(),num_items{',layout_' if layout else ''});"
    return dict(source=source, launch=launch)


def make_forward_source(
    meta, program, *, arch, layout=False, f_in=None, f_out=None, scaled=None,
):
    """Generate forward with dense or explicitly described strided inputs."""
    headers, declarations = "", ""
    if layout:
        xi = max(1, meta["in_dim"] // (f_in or 1))
        wi = max(1, meta["weight_numel"] // ((f_in or 1) * (f_out or 1)))
        bi = max(1, meta["bias_numel"] // (f_out or 1))
        fo = f_out or 1
        weight_batch = "Index(0)" if meta["shared_weights"] else f"flat/{max(1,meta['weight_numel'])}"
        weight_field = "flat" if meta["shared_weights"] else f"flat%{max(1,meta['weight_numel'])}"
        prefix = (r"""
struct ForwardLayout {{
    int rank;
    Index sizes[{MAX_BATCH_RANK}],x_stride[{MAX_BATCH_RANK}],w_stride[{MAX_BATCH_RANK}];
    Index x_feature[2],w_feature[3],b_feature[2];
}};
__device__ __forceinline__ Index layout_batch(const ForwardLayout& layout,Index item,bool weight) {{
    Index offset=0;
    for(int axis=layout.rank-1;axis>0;--axis) {{
        const Index coordinate=item%layout.sizes[axis]; item/=layout.sizes[axis];
        offset+=coordinate*(weight?layout.w_stride[axis]:layout.x_stride[axis]);
    }}
    if(layout.rank) offset+=item*(weight?layout.w_stride[0]:layout.x_stride[0]);
    return offset;
}}
""").format(
            MAX_BATCH_RANK=MAX_BATCH_RANK,
        )
        bodies = (
            (r"""    const Index item=flat/{dimension}; const int q=flat%{dimension};
    const Index address=layout_batch(layout,item,false)+(q/{xi})*layout.x_feature[0]+(q%{xi})*layout.x_feature[1];
    return pointer[address];""").format(
                dimension=max(1, meta['in_dim']),
                xi=xi,
            ),
            (r"""    const Index item={weight_batch}; const int q={weight_field};
    const Index address={batch_address}+
        (q/{wi}/{fo})*layout.w_feature[0]+((q/{wi})%{fo})*layout.w_feature[1]+(q%{wi})*layout.w_feature[2];
    return pointer[address];""").format(
                weight_batch=weight_batch,
                weight_field=weight_field,
                batch_address='Index(0)' if meta['shared_weights'] else 'layout_batch(layout,item,true)',
                wi=wi,
                fo=fo,
            ),
            f"    return pointer[(flat/{bi})*layout.b_feature[0]+(flat%{bi})*layout.b_feature[1]];",
        )
        flags = (tuple(scaled) + (False, False, False))[:3] if scaled is not None else (False,) * 3
        readers = []
        for role, body, flag, field in zip(
            ("x", "weight", "bias"),
            bodies,
            flags,
            ("x_feature[1]", "w_feature[2]", "b_feature[1]"),
        ):
            if flag:
                body = f"    return pointer[flat*layout.{field}];"
            readers.append(
                f"__device__ __forceinline__ Element layout_{role}(const Element* pointer,Index flat,const ForwardLayout& layout) {{\n"
                + body
                + "\n}\n"
            )
        declarations = prefix + "".join(readers)
        headers = "\n#include <vector>\n#include <limits>\n#include <pybind11/stl.h>"

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
        if meta["shared_weights"] and meta["max_channels"] >= 32:
            emitted = make_tiled_feature_kernel(
                meta, kernel, pairs, broadcast, arch=arch, layout=layout,
            )
        else:
            emitted = make_feature_kernel(meta, kernel, pairs, broadcast, layout=layout)
        shape = f"{{num_items,int64_t({meta['out_dim']})}}"
        sources.append(emitted["source"])
        launches.append(
            f"auto result_{slot}=torch::empty({shape},options);\n{{auto out=result_{slot};\n{emitted['launch']}\n}}"
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
#include <cmath>{headers}
using Element={scalar};{declarations}
{kernels}
""").format(
        headers=headers,
        scalar=scalar,
        declarations=declarations,
        kernels="\n".join(sources),
    )
    body = (r"""
    const c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    {launches}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pybind11::make_tuple({results});
""").format(
        first=first,
        launches="\n".join(launches),
        results=','.join(returns),
    )
    parameters += ",int64_t num_items"
    if not layout:
        return (
            prefix
            + f"pybind11::tuple launch({parameters}) {{{body}}}\n"
            + 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&launch);}\n'
        )
    source = prefix
    launch_body = body.removesuffix("\n")
    source += (r"""
class RawLayoutLauncher {{
    ForwardLayout layout_{{}};
public:
    explicit RawLayoutLauncher(const std::vector<int64_t>& values) {{
        const int64_t rank=values[0];
        layout_.rank=int(rank);
        auto value=[&](int64_t index) {{
            return Index(values[index]);
        }};
        for(int64_t axis=0;axis<rank;++axis) {{
            layout_.sizes[axis]=value(1+axis); layout_.x_stride[axis]=value(1+rank+axis); layout_.w_stride[axis]=value(1+2*rank+axis);
        }}
        for(int j=0;j<2;++j) layout_.x_feature[j]=value(1+3*rank+j);
        for(int j=0;j<3;++j) layout_.w_feature[j]=value(3+3*rank+j);
        for(int j=0;j<2;++j) layout_.b_feature[j]=value(6+3*rank+j);
    }}
    pybind11::tuple launch({parameters}) {{{launch_body}
    }}
}};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
    pybind11::class_<RawLayoutLauncher>(m,"RawLayoutLauncher",pybind11::module_local())
        .def(pybind11::init<const std::vector<int64_t>&>()).def("launch",&RawLayoutLauncher::launch);
}}
""").format(
        MAX_BATCH_RANK=MAX_BATCH_RANK,
        parameters=parameters,
        launch_body=launch_body,
    )
    return source
