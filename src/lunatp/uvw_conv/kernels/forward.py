"""UVW convolution forward CUDA and native launch."""

from __future__ import annotations
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal
from ....utils.extension import load_cuda_extension
from .utils import set_arch


def make_uvw_conv_forward_external_kernel(
    meta, products=(((("x", "y"),), "w"),), kernel_name="lunatp_uvwconv_fwd_fused_fp32"
):
    """Return CUDA kernel, grid expression, zeroing flag and input pointer order.

    Each product is (CG input pairs, weight pointer). Grouping pairs that share a
    weight avoids repeating its matrix product in double and triple backward.
    """
    groups = {}
    for spec in meta["generated_specs"]:
        groups.setdefault(
            (spec["out_offset"], spec["out_channels"], spec["d3"]), []
        ).append(spec)
    max_tiles = max((channels + 127) // 128 for _, channels, _ in groups)
    zero_output = True  # Multiple edges contribute to each destination.
    used = {
        name
        for pairs, weight in products
        for pair in pairs
        for name in pair
        if name is not None
    }
    used.update(weight for _, weight in products)
    inputs = tuple(
        name
        for name in ("x", "y", "w", "rx", "ry", "rw", "bx", "by", "bw", "e", "re", "be")
        if name in used
    )
    multiple = len(products) > 1
    weighted = any(len(pair) == 3 for pairs, _ in products for pair in pairs)
    cases = []
    for group, ((offset, channels, d3), specs) in enumerate(groups.items()):
        columns = min(4, (channels + 31) // 32)
        width = columns * 32
        lines = r"""if (group == {group}) {{
Index item=(task/{max_tiles})*4+(threadIdx.x>>5);
int tile_column=(task%{max_tiles})*{width};
if (item>=items || tile_column>={channels}) return;
int column=tile_column+lane;
const int source_node=__ldg(graph_src+item);
const int dest_node=__ldg(graph_dst+item);
using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{d3},{columns},1>,
float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
Mma::FragmentC accum; accum.clear();""".format(
            group=group,
            max_tiles=max_tiles,
            width=width,
            channels=channels,
            d3=d3,
            columns=columns,
        ).split("\n")
        for spec in specs:
            size, edges, wo = (
                spec["path_channels"],
                spec["edge_channels"],
                spec["weight_offset"],
            )
            path_type = "int64_t" if size > (1 << 31) - 32 else "int"
            path_bound_type = "int64_t" if path_type == "int64_t" else "Index"
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
                if weighted:
                    choices = {"weight": [weight for _, weight in products]}
                    for pair_index in range(max(len(pairs) for pairs, _ in products)):
                        for axis, stem in enumerate(("a", "b", "scalar")):
                            choices[f"{stem}{pair_index}"] = [
                                pairs[pair_index][axis]
                                if pair_index < len(pairs)
                                and pairs[pair_index][axis] is not None
                                else "nullptr"
                                for pairs, _ in products
                            ]
                for name, values in choices.items():
                    expression = values[-1]
                    for q in reversed(range(len(values) - 1)):
                        expression = f"(product=={q}?{values[q]}:{expression})"
                    lines.append(f"const float* {name}={expression};")
            weight = "weight" if multiple else products[0][1]
            lines += r"""constexpr {path_bound_type} begin=0,end={size};
#pragma unroll 1
for ({path_type} base=begin;base<end;base+=32) {{
Mma::FragmentB next;""".format(
                path_bound_type=path_bound_type,
                size=size,
                path_type=path_type,
            ).split("\n")
            for n in range(columns):
                lines += r"""next[{fragment_column}]=0.f; if(column+{column_offset}<{channels}) {{
const float* src={weight}+int64_t(item)*{weight_numel}+{weight_offset}+base*{channels}+column+{column_offset};
asm volatile("ld.global.L1::no_allocate.f32 %0,[%1];":"=f"(next[{fragment_column}]):"l"(src));
}}""".format(
                    fragment_column=n,
                    column_offset=n * 32,
                    channels=channels,
                    weight=weight,
                    weight_numel=meta["weight_numel"],
                    weight_offset=wo,
                ).split("\n")
            lines += [f"float z{k}=0.f;" for k in range(d3)]
            lines += [
                f"{path_type} p=base+lane;",
                f"if(p<end) {{ int u=p/{edges},v=p%{edges};",
            ]
            if multiple:
                if weighted:
                    for pair_index in range(max(len(pairs) for pairs, _ in products)):
                        lines += [
                            f"if(a{pair_index}!=nullptr) {{",
                            emit_cg_source(
                                meta,
                                spec,
                                pairs=(
                                    (
                                        f"a{pair_index}",
                                        f"b{pair_index}",
                                        f"scalar{pair_index}",
                                    ),
                                ),
                                x_row="source_node",
                            ),
                            "}",
                        ]
                else:
                    lines += [
                        emit_cg_source(
                            meta, spec, pairs=(("a", "b"),), x_row="source_node"
                        )
                    ]
                    has_optional_pair = any(len(pairs) == 1 for pairs, _ in products)
                    lines += [
                        "if(a1!=nullptr) {" if has_optional_pair else "{",
                        emit_cg_source(
                            meta, spec, pairs=(("a1", "b1"),), x_row="source_node"
                        ),
                        "}",
                    ]
            else:
                lines += [
                    emit_cg_source(
                        meta, spec, pairs=products[0][0], x_row="source_node"
                    )
                ]
            prefetch_lines = []
            for n in range(columns):
                prefetch_lines += r"""if(column+{column_offset}<{channels}) {{
const float* src={weight}+int64_t(item)*{weight_numel}+{weight_offset}+(base+k+1)*{channels}+column+{column_offset};
asm volatile("ld.global.L1::no_allocate.f32 %0,[%1];":"=f"(next[{fragment_column}]):"l"(src));
}}""".format(
                    column_offset=n * 32,
                    channels=channels,
                    weight=weight,
                    weight_numel=meta["weight_numel"],
                    weight_offset=wo,
                    fragment_column=n,
                ).split("\n")
            lines += r"""}}
#pragma unroll 4
for(int k=0;k<32 && base+k<end;++k) {{
Mma::FragmentA a; Mma::FragmentB b;
b=next;
if(k+1<32 && base+k+1<end) {{
{weight_prefetch}
}}
{fragment_loads}
Mma()(accum,a,b,accum);
}}
}}""".format(
                weight_prefetch="\n".join(prefetch_lines),
                fragment_loads="\n".join(
                    (f"a[{k}]=__shfl_sync(0xffffffff,z{k},k);" for k in range(d3))
                ),
            ).split("\n")
            if multiple:
                lines.append("}")
            lines.append("}")
        lines += r"""#pragma unroll
for(int r=0;r<{d3};++r) {{
int component=r;
#pragma unroll
for(int n=0;n<{columns};++n) {{
int c=column+n*32;
if(item<items && c<{channels}) {{
float* dst=out+int64_t(dest_node)*{out_dim}+{offset}+c*{d3}+component;
atomicAdd(dst,accum[r*{columns}+n]);
}}
}}
}}
return;
}}""".format(
            d3=d3,
            columns=columns,
            channels=channels,
            out_dim=meta["out_dim"],
            offset=offset,
        ).split("\n")
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
void {kernel_name}({pointers},const int32_t* graph_src,const int32_t* graph_dst,float* __restrict__ out,int64_t items) {{
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
    return (
        cuda_source_for_dtype(kernel, meta.get("dtype", torch.float32)),
        grid,
        zero_output,
        inputs,
    )


def make_uvw_conv_forward_external_source(meta, *, env=False):
    products = (((("x", "y", "e"),), "w"),) if env else (((("x", "y"),), "w"),)
    kernel, grid, _, _ = make_uvw_conv_forward_external_kernel(meta, products)
    env_tensor = ",at::Tensor e" if env else ""
    env_argument = "e.data_ptr<float>()," if env else ""
    return cuda_source_for_dtype(
        kernel
        + r"""
at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w{env_tensor},at::Tensor src,at::Tensor dst,int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=y.size(0);
    auto out=at::zeros({{num_nodes,int64_t({out_dim})}},x.options());
    if(items==0) return out;
    lunatp_uvwconv_fwd_fused_fp32<<<{grid},128,0,at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),y.data_ptr<float>(),w.data_ptr<float>(),{env_argument}src.data_ptr<int32_t>(),dst.data_ptr<int32_t>(),out.data_ptr<float>(),items);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{ m.def("launch",&launch); }}
""".format(
            env_tensor=env_tensor,
            out_dim=meta["out_dim"],
            grid=grid,
            env_argument=env_argument,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_forward_shared_source(meta, *, env=False):
    component_rows = max(spec["d3"] for spec in meta["generated_specs"])
    if env:
        env_load = [f"float ev=__ldg(env+e);"]
        env_scale = "*ev"
        env_parameter = ",const float* env"
        env_tensor = ",at::Tensor env"
        env_argument = "env.data_ptr<float>(),"
    else:
        env_load = []
        env_scale = env_parameter = env_tensor = env_argument = ""
    cases = []
    for index, spec in enumerate(meta["generated_specs"]):
        d1, d3 = (spec[k] for k in ("d1", "d3"))
        channels, outputs = spec["path_channels"], spec["out_channels"]
        path_type = "int64_t" if channels > (1 << 31) - 64 else "int"
        edge_channels, d2 = spec["edge_channels"], spec["d2"]
        decode = (
            f"{path_type} p=u; u=p/{edge_channels}; int v=p%{edge_channels};"
            if edge_channels != 1
            else ""
        )
        edge_offset = f"v*{d2}+" if edge_channels != 1 else ""
        lines = [
            f"case {index}: {{",
            f"using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{d3},1,1>,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;",
            "Mma::FragmentC acc; acc.clear();",
            "const int column=threadIdx.x+blockIdx.z*128;",
            f"for({path_type} base=0;base<{channels};base+=64) {{",
            "{",
            f"{path_type} u=base+threadIdx.x%64;",
            *(f"float z{k}=0.f;" for k in range(d3)),
            f"if(u<{channels}) {{",
            *([decode] if decode else []),
            "for(Index pos=rowptr[node]+threadIdx.x/64;pos<rowptr[node+1];pos+=2) {",
            "Index s=src[pos],e=edge[pos];",
        ]
        lines += env_load
        for i in sorted({t[0] for t in spec["terms"]}):
            lines.append(
                f"""float xv{i}=__ldg(x+int64_t(s)*{meta["in1_dim"]}+{spec["x_offset"]}+u*{d1}+{i});"""
            )
        for j in sorted({t[1] for t in spec["terms"]}):
            lines.append(
                f"""float yv{j}=__ldg(y+int64_t(e)*{meta["in2_dim"]}+{spec["y_offset"]}+{edge_offset}{j}){env_scale};"""
            )
        for i, j, k, c in spec["terms"]:
            lines.append(
                f"""z{k}=fmaf({float_literal(c, meta.get("dtype"))},xv{i}*yv{j},z{k});"""
            )
        lines += ["}", "}"]
        for k in range(d3):
            value = f"z{k}"
            lines.append(
                f"a[(threadIdx.x/64)*{component_rows * 64}+{k}*64+threadIdx.x%64]={value};"
            )
        lines += r"""}}
__syncthreads();
for(int i=threadIdx.x;i<{tile_values};i+=128) a[i]+=a[{tile_stride}+i];
__syncthreads();
if(column<{outputs}) {{
#pragma unroll 4
for(int k=0;k<64;++k) {{
Mma::FragmentA av;
{fragment_loads}
Mma::FragmentB bv;
bv[0]=(base+k<{channels} && column<{outputs})?__ldg(w+{weight_offset}+(base+k)*{outputs}+column):0.f;
Mma()(acc,av,bv,acc);
}}
}}
__syncthreads();
}}
if(column<{outputs}) {{
#pragma unroll
for(int r=0;r<{components};++r) {{
if(column<{outputs}) atomicAdd(out+int64_t(node)*{out_dim}+{out_offset}+column*{components}+r,acc[r]);
}}
}}
break;
}}""".format(
            tile_values=d3 * 64,
            tile_stride=component_rows * 64,
            outputs=outputs,
            fragment_loads="\n".join((f"av[{r}]=a[{r}*64+k];" for r in range(d3))),
            channels=channels,
            weight_offset=spec["weight_offset"],
            components=d3,
            out_dim=meta["out_dim"],
            out_offset=spec["out_offset"],
        ).split("\n")
        cases.append("\n".join(lines))
    out_tiles = max((s["out_channels"] + 127) // 128 for s in meta["generated_specs"])
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
__global__ __launch_bounds__(128,4)
void lunatp_uvwconv_fwd_shared_fused_fp32(const float* x,const float* y,const float* w{env_parameter},const int32_t* src,const EdgeIndex* edge,const EdgeIndex* rowptr,float* out) {{
    __shared__ float a[2*{component_rows}*64];
    Index node=blockIdx.x;
    switch(blockIdx.y) {{
{cases}
    }}
}}
at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w{env_tensor},at::Tensor src,at::Tensor edge,at::Tensor rowptr) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_nodes=rowptr.numel()-1;
    auto out=at::zeros({{num_nodes,int64_t({out_dim})}},x.options());
    if(num_nodes==0) return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_fwd_shared_fused_fp32<<<dim3(num_nodes,{cases_count},{out_tiles}),128,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),w.data_ptr<float>(),{env_argument}src.data_ptr<int32_t>(),edge.data_ptr<EdgeIndex>(),rowptr.data_ptr<EdgeIndex>(),out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""".format(
            env_parameter=env_parameter,
            component_rows=component_rows,
            cases=chr(10).join(cases),
            env_tensor=env_tensor,
            out_dim=meta["out_dim"],
            cases_count=len(cases),
            out_tiles=out_tiles,
            env_argument=env_argument,
        ),
        meta.get("dtype", torch.float32),
    )


def run_external(op, inputs, src, dst, *, num_nodes, env=False):
    key = (0, (), True) if env else (0, ())
    launch = op.external_kernels.get(key)
    if launch is None:
        launch = load_cuda_extension(
            f"lunatp_uvwconv_fwd_external_{op.dtype_tag}",
            make_uvw_conv_forward_external_source(op.meta, env=env),
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        ).launch
        op.external_kernels[key] = launch
    return launch(*inputs, src, dst, num_nodes)


def run(op, x, edge_attr, weight, *, graph, env=None):
    set_arch(op, x.get_device())
    src, dst = graph.src, graph.dst
    num_nodes = graph.num_nodes
    if env is not None:
        if op.weight_mode != "shared":
            return run_external(
                op,
                (x, edge_attr, weight, env),
                src,
                dst,
                num_nodes=num_nodes,
                env=True,
            )
        if op.envelope_forward is None:
            op.envelope_forward = load_cuda_extension(
                f"lunatp_uvwconv_fwd_shared_node_{op.dtype_tag}",
                make_uvw_conv_forward_shared_source(op.meta, env=True),
                arch=op.arch,
                use_fast_math=False,
                index_bits=op.index_bits,
            ).launch
        return op.envelope_forward(
            x,
            edge_attr,
            weight,
            env,
            graph.src,
            graph.edge_ids_by_dst,
            graph.dst_rowptr,
        )
    if op.weight_mode == "shared":
        if op.fused_forward is None:
            op.fused_forward = load_cuda_extension(
                f"lunatp_uvwconv_fwd_shared_node_{op.dtype_tag}",
                make_uvw_conv_forward_shared_source(op.meta),
                arch=op.arch,
                use_fast_math=False,
                index_bits=op.index_bits,
            ).launch
        return op.fused_forward(
            x,
            edge_attr,
            weight,
            graph.src,
            graph.edge_ids_by_dst,
            graph.dst_rowptr,
        )

    return run_external(op, (x, edge_attr, weight), src, dst, num_nodes=num_nodes)
