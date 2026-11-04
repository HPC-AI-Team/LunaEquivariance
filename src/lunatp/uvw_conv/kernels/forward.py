"""UVW convolution forward CUDA and native launch."""

from __future__ import annotations
import textwrap
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal
from ....utils.extension import load_cuda_extension
from .utils import FORWARD_SPLIT_CHANNELS, NODE_NEIGHBOR_SPLITS, set_arch


CUDA_HEADERS = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
"""


def make_uvw_conv_forward_external_kernel(
    meta, products=(((("x", "y"),), "w"),), kernel_name="lunatp_uvwconv_fwd_fused_fp32"
):
    """Assign one output channel per lane and reuse inputs over each CSR partition."""
    used = {name for pairs, weight in products for pair in pairs for name in pair if name is not None}
    used.update(weight for _, weight in products)
    inputs = tuple(name for name in ("x", "y", "w", "rx", "ry", "rw", "bx", "by", "bw", "e", "re", "be") if name in used)
    groups = {}
    for spec in meta["generated_specs"]:
        groups.setdefault((spec["out_offset"], spec["out_channels"], spec["d3"]), []).append(spec)
    cases, tasks = [], 0
    for (offset, n, d), group in groups.items():
        splits = (max(spec["path_channels"] for spec in group) + FORWARD_SPLIT_CHANNELS - 1) // FORWARD_SPLIT_CHANNELS
        count = ((n + 31) // 32) * splits
        terms = []
        for spec in group:
            k = spec["path_channels"]
            for pairs, weight in products:
                d1, d2, edges = (spec[name] for name in ("d1", "d2", "edge_channels"))
                features = tuple(dict.fromkeys(pair[0] for pair in pairs))
                rank = len(features) * d1
                edge_slices = ((k // FORWARD_SPLIT_CHANNELS) * min(edges, FORWARD_SPLIT_CHANNELS)
                               + min(edges, k % FORWARD_SPLIT_CHANNELS))
                cg_first_cost = d * k * n + len(pairs) * len(spec["terms"]) * k * ((n + 31) // 32)
                projection_cost = rank * k * n + len(pairs) * len(spec["terms"]) * edge_slices * n
                scalar_slots = 2 if meta.get("dtype") == torch.float64 else 1
                live_values = d + max(2 * rank + 16, rank + len(pairs) * (d2 + 1))
                # Choose the smaller equivalent contraction before generating CUDA.
                if pairs and rank <= d and projection_cost < cg_first_cost and scalar_slots * live_values <= 255:
                    loads = "".join(
                        f"                    const float x_{feature}{i}={feature}[Index(source_node)*{meta['in1_dim']}+{spec['x_offset']}+u*{d1}+{i}];"
                        for feature in features for i in range(d1)
                    )
                    updates = "".join(
                        f"                    projected_{feature}[{i}]=fmaf(x_{feature}{i},value,projected_{feature}[{i}]);"
                        for feature in features for i in range(d1)
                    )
                    couplings = []
                    for pair_index, pair in enumerate(pairs):
                        feature, edge = pair[:2]
                        scale = pair[2] if len(pair) > 2 else None
                        yloads = "".join(
                            f"            const float y{pair_index}_{j}={edge}[Index(item)*{meta['in2_dim']}+{spec['y_offset']}+v*{d2}+{j}];"
                            for j in sorted({term[1] for term in spec["terms"]})
                        )
                        if scale is not None:
                            yloads += f"const float envelope{pair_index}={scale}[item];"
                        update = "".join(
                            f"result[{r}]=fmaf({float_literal(coefficient, meta.get('dtype'))}*projected_{feature}[{i}],y{pair_index}_{j}"
                            + (f"*envelope{pair_index}" if scale is not None else "")
                            + f"                    ,result[{r}]);"
                            for i, j, r, coefficient in spec["terms"]
                        )
                        couplings.append(f"{{ {yloads} if(channel<{n}) {{ {update} }} }}")
                    terms.append(f"""
for(Index v=0;v<{edges};++v) {{
    const Index first_u=Index((int64_t(begin)+{edges - 1}-v)/{edges});
    const Index last_u=min(Index({spec['in_channels']}),Index((int64_t(min(Index({k}),begin+{FORWARD_SPLIT_CHANNELS}))+{edges - 1}-v)/{edges}));
    if(first_u<last_u) {{
        {''.join(f'float projected_{feature}[{d1}]={{}};' for feature in features)}
        for(Index base=first_u;base<last_u;base+=16) {{
            float prefetched[16];
#pragma unroll
            for(int j=0;j<16;++j) {{
                const Index u=base+j;
                prefetched[j]=(u<last_u && channel<{n})?{weight}[Index(item)*{meta['weight_numel']}+{spec['weight_offset']}+(u*{edges}+v)*{n}+channel]:0.f;
            }}
#pragma unroll
            for(int j=0;j<16;++j) {{
                const Index u=base+j;
                if(u<last_u && channel<{n}) {{
{textwrap.indent(loads, '                    ')}
                    const float value=prefetched[j];
{textwrap.indent(updates, '                    ')}
                }}
            }}
        }}
{textwrap.indent(''.join(couplings), '        ')}
    }}
}}""")
                    continue
                update = "".join(f'                    result[{r}]=fmaf(q{r},value,result[{r}]);' for r in range(d))
                terms.append(f"""
for(Index chunk=begin;chunk<min(Index({k}),begin+{FORWARD_SPLIT_CHANNELS});chunk+=32) {{
    const Index p=chunk+lane,u=p/{spec['edge_channels']},v=p%{spec['edge_channels']};
    {''.join(f'float z{r}=0.f;' for r in range(d))}
    if(p<{k} && p<begin+{FORWARD_SPLIT_CHANNELS}) {{
{textwrap.indent(emit_cg_source(meta,spec,pairs=pairs,x_row='source_node').replace('int64_t', 'Index'), '        ')}
    }}
    const int length=int(min(Index(32),min(Index({k}),begin+{FORWARD_SPLIT_CHANNELS})-chunk));
#pragma unroll
    for(int first=0;first<32;first+=16) {{
        float prefetched[16];
#pragma unroll
        for(int j=0;j<16;++j) {{
            prefetched[j]=(channel<{n} && first+j<length)?{weight}[Index(item)*{meta['weight_numel']}+{spec['weight_offset']}+(chunk+first+j)*{n}+channel]:0.f;
        }}
#pragma unroll
        for(int j=0;j<16;++j) {{
            const int s=first+j;
            if(s<length) {{
                {''.join(f'const float q{r}=__shfl_sync(0xffffffff,z{r},s);' for r in range(d))}
                if(channel<{n}) {{
                    const float value=prefetched[j];
{textwrap.indent(update, '                    ')}
                }}
            }}
        }}
    }}
}}""")
        store = "".join(
            f"atomicAdd(out+Index(dest_node)*{meta['out_dim']}+{offset}+Index(channel)*{d}+{r},result[{r}]);"
            for r in range(d)
        )
        cases.append(f"""
if (local < {tasks+count}) {{
    const int channel = ((local - {tasks}) / {splits}) * 32 + lane;
    const Index begin = ((local - {tasks}) % {splits}) * {FORWARD_SPLIT_CHANNELS};
    float result[{d}] = {{}};
    for (Index position = Index(edge_begin); position < edge_end; ++position) {{
        const EdgeIndex item = edge_by_dst[position];
        const int source_node = graph_src[item];
{textwrap.indent(''.join(terms), '        ')}
    }}
    if (channel < {n}) {{
        {store}
    }}
    return;
}}""")
        tasks += count
    pointers = ','.join(f'const float* {name}' for name in inputs)
    tasks_per_node = tasks * NODE_NEIGHBOR_SPLITS
    grid = f'(nodes*{tasks_per_node}+3)/4'
    launch_bounds = "128,8" if meta.get("dtype", torch.float32) == torch.float32 else "128"
    source = f"""
__global__ __launch_bounds__({launch_bounds})
    void {kernel_name}({pointers},const int32_t* graph_src,const EdgeIndex* edge_by_dst,const EdgeIndex* dst_rowptr,float* out,int64_t nodes) {{
    const int lane=threadIdx.x%32;
    const int64_t task=int64_t(blockIdx.x)*4+threadIdx.x/32;
    if(task>=nodes*{tasks_per_node})return;
    const int64_t dest_node=task/{tasks_per_node},local=task%{tasks};
    const int partition=task/{tasks}%{NODE_NEIGHBOR_SPLITS};
    const int64_t first=dst_rowptr[dest_node],degree=dst_rowptr[dest_node+1]-first;
    const int64_t edge_begin=first+degree*partition/{NODE_NEIGHBOR_SPLITS};
    const int64_t edge_end=first+degree*(partition+1)/{NODE_NEIGHBOR_SPLITS};
    if(edge_begin==edge_end)return;
{textwrap.indent(chr(10).join(cases), '    ')}
}}
"""
    return cuda_source_for_dtype(source,meta.get("dtype",torch.float32)),grid,True,inputs,128


def make_uvw_conv_forward_external_source(meta, *, env=False):
    products = (((("x", "y", "e"),), "w"),) if env else (((("x", "y"),), "w"),)
    kernel, grid, _, _, threads = make_uvw_conv_forward_external_kernel(meta, products)
    env_tensor = ",at::Tensor e" if env else ""
    env_argument = "e.data_ptr<float>()," if env else ""
    return cuda_source_for_dtype(
        CUDA_HEADERS + kernel
        + r"""
at::Tensor launch(at::Tensor x, at::Tensor y, at::Tensor w{env_tensor}, at::Tensor src, at::Tensor edge_by_dst,
                                    at::Tensor dst_rowptr, int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t nodes = num_nodes;
    auto out = at::zeros({{num_nodes, int64_t({out_dim})}}, x.options());
    if (nodes == 0 || y.size(0) == 0)
        return out;
    lunatp_uvwconv_fwd_fused_fp32<<<{grid}, {threads}, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), {env_argument}src.data_ptr<int32_t>(),
        edge_by_dst.data_ptr<EdgeIndex>(), dst_rowptr.data_ptr<EdgeIndex>(), out.data_ptr<float>(), nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch", &launch);
}}
""".format(
            env_tensor=env_tensor,
            out_dim=meta["out_dim"],
            grid=grid,
            threads=threads,
            env_argument=env_argument,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_forward_shared_source(meta, *, env=False):
    """Fuse CSR contraction into the input tiles of the CUTLASS mainloop."""
    definitions, dispatch, totals = [], [], []
    element_bytes = 8 if meta.get("dtype") is torch.float64 else 4
    partition = FORWARD_SPLIT_CHANNELS
    max_columns = max(spec["out_channels"] for spec in meta["generated_specs"])
    tile_n = min(512 // element_bytes, max(32, 1 << (max_columns - 1).bit_length()))
    max_components = max(spec["d3"] for spec in meta["generated_specs"])
    tile_m = min(64 // element_bytes, max(4, 1 << (2 * max_components - 1).bit_length()))
    tile_k = 32
    tile_channels = min(partition, (max(spec["path_channels"] for spec in meta["generated_specs"]) + 31) // 32 * 32)
    warp_m = tile_m
    warp_n = tile_n // 4
    for index, spec in enumerate(meta["generated_specs"]):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        k, n = spec["path_channels"], spec["out_channels"]
        xloads = '\n'.join(f'const float xv{i}=x[int64_t(source)*{meta["in1_dim"]}+{spec["x_offset"]}+int64_t(u)*{d1}+{i}];' for i in sorted({t[0] for t in spec['terms']}))
        yloads = '\n'.join(f'const float yv{j}=y[int64_t(e)*{meta["in2_dim"]}+{spec["y_offset"]}+int64_t(v)*{d2}+{j}];' for j in sorted({t[1] for t in spec['terms']}))
        compute = '\n'.join(f'z[{r}]=fmaf({float_literal(c, torch.float32)},xv{i}*yv{j}' + ('*envelope[e]' if env else '') + f',z[{r}]);' for i,j,r,c in spec['terms'])
        definitions.append(f"""
struct Contraction{index} {{
    const float *x,*y,*envelope;
    const int32_t* src;
    const EdgeIndex *edge,*rowptr;
    CUTLASS_DEVICE void produce(float* scratch,int64_t first,int64_t rows,int64_t begin,int64_t limit) const {{
        const int64_t first_node=first/{d3};
        const int nodes=({tile_m}+first%{d3}+{d3}-1)/{d3};
        for(int work=threadIdx.x;work<nodes*{tile_channels};work+=128) {{
            const int local_k=work%{tile_channels};const int64_t p=begin+local_k;
            const int64_t node=first_node+work/{tile_channels};
            const {"int64_t" if k > (1 << 31) - 1 else "int"} u=p/{spec['edge_channels']},v=p%{spec['edge_channels']};
            float z[{d3}]={{}};
            if(node<rows/{d3}&&p<limit) {{
                for(EdgeIndex pos=rowptr[node];pos<rowptr[node+1];++pos) {{
                    const int32_t source=src[pos];const EdgeIndex e=edge[pos];
{textwrap.indent(xloads, '                    ')}
{textwrap.indent(yloads, '                    ')}
{textwrap.indent(compute, '                    ')}
                }}
            }}
#pragma unroll
            for(int component=0;component<{d3};++component) {{
                const int row=node*{d3}+component-first;
                if(row>=0&&row<{tile_m})scratch[row*{tile_channels}+local_k]=z[component];
            }}
        }}
    }}
}};
""")
        columns = (n+tile_n-1)//tile_n
        splits = (k+partition-1)//partition
        tasks = f'((nodes*int64_t({d3})+{tile_m-1})/{tile_m})*{columns}*{splits}'
        totals.append(tasks)
        dispatch.append(f"""
if(task<cursor+({tasks})) {{
    const int64_t task_in_path=task-cursor,local=task_in_path/{splits},row_base=(local/{columns})*{tile_m};
    const int64_t begin=(task_in_path%{splits})*{partition};
    const int length=int(min(int64_t({partition}),int64_t({k})-begin));
    const int column=(local%{columns})*{tile_n};
    using A=InputIterator;
    using Mma=cutlass::gemm::threadblock::MmaSingleStage<typename BaseMma::Shape,A,
    typename BaseMma::SmemIteratorA,typename BaseMma::IteratorB,typename BaseMma::SmemIteratorB,
    float,cutlass::layout::RowMajor,typename BaseMma::Policy>;
    Contraction{index} producer{{x,y,envelope,src,edge,rowptr}};
    producer.produce(scratch,row_base,nodes*int64_t({d3}),begin,begin+length);
    __syncthreads();
    A a(scratch,row_base,length,threadIdx.x);
    typename BaseMma::IteratorB b(typename BaseMma::IteratorB::Params(cutlass::layout::RowMajor({n})),
    const_cast<float*>(w+{spec['weight_offset']}+begin*{n}),{{length,{n}}},threadIdx.x,{{0,column}});
    typename Mma::FragmentC accum;accum.clear();
    Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    mma((length+{tile_k-1})/{tile_k},accum,a,b,accum);
    __syncthreads();
    using E=Epilogue<{d3}>;
    typename E::OutputTileIterator output(typename E::OutputTileIterator::Params(cutlass::layout::RowMajor({meta['out_dim']})),
    out+{spec['out_offset']},row_base,{{int(min(int64_t({tile_m}),nodes*int64_t({d3})-row_base)),{n}}},threadIdx.x,{{0,column}});
    E epilogue(storage.epilogue,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    epilogue(typename E::OutputOp(1.f,0.f),output,accum,output);
    return;
}}
cursor+=({tasks});
""")
    env_tensor = ',at::Tensor envelope' if env else ''
    env_argument = 'envelope.data_ptr<float>()' if env else 'nullptr'
    source = f"""
#include <algorithm>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_singlestage.h>
namespace {{
using Kernel=typename cutlass::gemm::device::Gemm<
    float,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
    float,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tile_m},{tile_n},{tile_k}>,cutlass::gemm::GemmShape<{warp_m},{warp_n},{tile_k}>,cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,1,1>::GemmKernel;
using BaseMma=typename Kernel::Mma;
{''.join(definitions)}
struct InputIterator {{
    using Original=typename BaseMma::IteratorA;
    using Element=float;
    using Fragment=typename Original::Fragment;
    using ThreadMap=typename Original::ThreadMap;
    float* scratch;
    int64_t row,tile;
    int k,columns;
    bool enabled=true,first=true;
    CUTLASS_DEVICE InputIterator(float* data,int64_t row_base,int columns_,int thread):scratch(data),tile(row_base),columns(columns_) {{
        auto q=ThreadMap::initial_offset(thread);
        row=row_base+q.contiguous();k=q.strided();
    }}
    CUTLASS_DEVICE void clear_mask(bool clear=true) {{if(clear)enabled=false;}}
    CUTLASS_DEVICE InputIterator& operator++() {{k+=first?(columns%{tile_k}?columns%{tile_k}:{tile_k}):{tile_k};first=false;return *this;}}
    CUTLASS_DEVICE void load(Fragment& fragment) const {{
        if(!enabled) {{fragment.clear();return;}}

#pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s) {{
            const int p=k+s*ThreadMap::Delta::kStrided;
#pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c) {{
                const int64_t r=row+c*ThreadMap::Delta::kContiguous;
                const int offset=(r-tile)*{tile_channels}+p;
                fragment[s*ThreadMap::Iterations::kContiguous+c]=scratch[offset];
            }}
        }}
    }}
}};
template<int Components> struct OutputIterator:Kernel::Epilogue::OutputTileIterator {{
    using Base=typename Kernel::Epilogue::OutputTileIterator;
    using ThreadMap=typename Base::ThreadMap;
    using Fragment=typename Base::Fragment;
    using Params=typename Base::Params;
    float* pointer;
    int64_t stride,row_base;
    CUTLASS_DEVICE OutputIterator(Params const& p,float* out,int64_t first,cutlass::MatrixCoord extent,int thread,cutlass::MatrixCoord offset={{}},int const* indices=nullptr):
        Base(p,out,extent,thread,offset,indices),pointer(out),stride(p.stride/sizeof(float)),row_base(first) {{}}
    CUTLASS_DEVICE void store(Fragment const& fragment) const {{
        typename Base::Mask mask;this->get_mask(mask);
#pragma unroll
        for(int cluster=0;cluster<ThreadMap::Iterations::kCluster;++cluster)
#pragma unroll
            for(int group=0;group<ThreadMap::Iterations::kGroup;++group)
#pragma unroll
                for(int row=0;row<ThreadMap::Iterations::kRow;++row) {{
                    const int r=this->thread_start_row()+row*ThreadMap::Delta::kRow+group*ThreadMap::Delta::kGroup+cluster*ThreadMap::Delta::kCluster;
                    const int fr=row+ThreadMap::Iterations::kRow*(group+ThreadMap::Iterations::kGroup*cluster);
#pragma unroll
                    for(int col=0;col<ThreadMap::Iterations::kColumn;++col) {{
                        const int c=this->thread_start_column()+col*ThreadMap::Delta::kColumn;
                        if(r<this->extent_row()&&mask.predicates[col])
                            atomicAdd(pointer+((row_base+r)/Components)*stride+int64_t(c)*Components+(row_base+r)%Components,fragment[fr*ThreadMap::Iterations::kColumn+col]);
            }}
        }}
    }}
}};
template<int Components> using Epilogue=cutlass::epilogue::threadblock::Epilogue<
    typename Kernel::Epilogue::Shape,typename Kernel::Epilogue::WarpMmaOperator,Kernel::Epilogue::kPartitionsK,
    OutputIterator<Components>,typename Kernel::Epilogue::AccumulatorFragmentIterator,typename Kernel::Epilogue::WarpTileIterator,
    typename Kernel::Epilogue::SharedLoadIterator,typename Kernel::Epilogue::OutputOp,typename Kernel::Epilogue::Padding>;
__global__ __launch_bounds__(128)
    void lunatp_uvwconv_fwd_shared_fused_fp32(const float* x,const float* y,const float* w,const float* envelope,
                                                                const int32_t* src,const EdgeIndex* edge,const EdgeIndex* rowptr,float* out,int64_t nodes) {{
    union Storage {{typename cutlass::gemm::threadblock::MmaBase<typename BaseMma::Shape,typename BaseMma::Policy,1>::SharedStorage main_loop;typename Kernel::Epilogue::SharedStorage epilogue;}};
    __shared__ Storage storage;
    __shared__ float scratch[{tile_m}*{tile_channels}];
    const int64_t task=blockIdx.x;
    int64_t cursor=0;
{textwrap.indent(''.join(dispatch), '    ')}
}}
}}
at::Tensor launch(at::Tensor x,at::Tensor y,at::Tensor w{env_tensor},at::Tensor src,at::Tensor edge,at::Tensor rowptr) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t nodes=rowptr.numel()-1;
    auto out=at::zeros({{nodes,int64_t({meta['out_dim']})}},x.options());
    if(nodes==0)return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_fwd_shared_fused_fp32<<<{'+'.join(totals)},128,0,stream>>>(
        x.data_ptr<float>(),y.data_ptr<float>(),w.data_ptr<float>(),{env_argument},
        src.data_ptr<int32_t>(),edge.data_ptr<EdgeIndex>(),rowptr.data_ptr<EdgeIndex>(),out.data_ptr<float>(),nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
"""
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, x, edge_attr, weight, *, graph, env=None):
    set_arch(op, x.get_device())
    if op.weight_mode != "shared":
        has_env = env is not None
        key = (0, (), True) if has_env else (0, ())
        launch = op.external_kernels.get(key)
        if launch is None:
            launch = load_cuda_extension(
                f"lunatp_uvwconv_fwd_external_{op.dtype_tag}",
                make_uvw_conv_forward_external_source(op.meta, env=has_env),
                arch=op.arch,
                use_fast_math=False,
                index_bits=op.index_bits,
            ).launch
            op.external_kernels[key] = launch
        inputs = (x, edge_attr, weight, env) if has_env else (x, edge_attr, weight)
        return launch(*inputs, graph.src, graph.edge_ids_by_dst, graph.dst_rowptr, graph.num_nodes)

    if env is not None:
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
    if op.fused_forward is None:
        op.fused_forward = load_cuda_extension(
            f"lunatp_uvwconv_fwd_shared_node_{op.dtype_tag}",
            make_uvw_conv_forward_shared_source(op.meta),
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        ).launch
    return op.fused_forward(
        x, edge_attr, weight, graph.src, graph.edge_ids_by_dst, graph.dst_rowptr,
    )
