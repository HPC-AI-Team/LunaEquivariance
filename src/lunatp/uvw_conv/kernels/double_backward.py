"""UVW convolution double backward CUDA and native launch."""

from __future__ import annotations
import textwrap
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype
from ....utils.extension import load_cuda_extension
from ...utils.runtime import auxiliary_streams
from .backward import run_external, load_node_contraction, load_weight_contraction
from .utils import MAX_OUTPUT_SPLITS, TARGET_OUTPUT_BLOCKS, get_uvw_conv_weight_layout, set_arch


def make_uvw_conv_q_shared_source(meta, queries, inputs):
    """Keep the existing Q layout/CTA while compiling weighted CG recipes."""
    specs = meta["generated_specs"]
    groups, rows, spec_to_group = get_uvw_conv_weight_layout(specs)
    threads = 128 if max(int(s["path_channels"]) for s in specs) <= 128 else 256
    arrays = tuple(
        sorted(
            {
                role
                for products in queries.values()
                for product in products
                for role in product
                if role is not None
            }
        )
    )
    output_pointers = []
    allocations = []
    for query_index in range(len(queries)):
        allocations.append("q_groups.emplace_back();")
        for group_index, ((_, _, d3), _) in enumerate(groups):
            output_pointers.append(f"float* __restrict__ q_{query_index}_{group_index}")
            allocations.append(
                f"q_groups.back().push_back(at::empty({{{rows[group_index]},num_nodes*int64_t({d3})}},x.options()));"
            )
    query_cases = []
    for query_index, products in enumerate(queries.values()):
        cases = []
        for spec_index, spec in enumerate(specs):
            group_index, row_offset = spec_to_group[spec_index]
            channels, edges, d3 = (
                int(spec[k]) for k in ("path_channels", "edge_channels", "d3")
            )
            channel_type = (
                "int64_t" if row_offset + channels > (1 << 31) - threads else "int"
            )
            lines = [
                f"case {spec_index}: {{",
                f"    for({channel_type} p=threadIdx.x;p<{channels};p+=blockDim.x) {{",
                f"        int u=p/{edges},v=p%{edges};",
                *[f"float acc{k}=0.f;" for k in range(d3)],
                "        for(Index pos=begin;pos<end;++pos) {",
                "            const Index item=__ldg(edge_by_dst+pos);",
                "            const int source_node=__ldg(src+pos);",
            ]
            for product in products:
                pair = tuple(
                    f"ptr_{role}" if role is not None else None for role in product
                )
                lines += [
                    "            {",
                    textwrap.indent(emit_cg_source(
                        meta,
                        spec,
                        pairs=(pair,),
                        x_row="source_node",
                        coefficient_first=False,
                        accumulator="acc",
                    ), '                '),
                    "            }",
                ]
            lines.append("        }")
            lines += [
                f"        q_{query_index}_{group_index}[({row_offset}+p)*(num_nodes*int64_t({d3}))+Index(node)*{d3}+{k}]=acc{k};"
                for k in range(d3)
            ]
            lines += ["    }", "    break;", "}"]
            cases.append("\n".join(lines))
        query_cases.append(
            f"case {query_index}: {{ switch(blockIdx.y) {{ {chr(10).join(cases)} }} break; }}"
        )

    arguments = [f"{name}.data_ptr<float>()" for name in arrays]
    arguments += [
        "src.data_ptr<int32_t>()",
        "edge_by_dst.data_ptr<EdgeIndex>()",
        "rowptr.data_ptr<EdgeIndex>()",
    ]
    arguments += [
        f"q_groups[{q}][{group}].data_ptr<float>()"
        for q in range(len(queries))
        for group in range(len(groups))
    ]
    arguments.append("num_nodes")
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
__global__ __launch_bounds__({threads}, 4) void lunatp_uvwconv_bwd_q_shared_fp32({input_pointers}, const int32_t *src,
                                                                                     const EdgeIndex *edge_by_dst,
                                                                                     const EdgeIndex *rowptr,
                                                                                     {output_pointers}, int64_t num_nodes) {{
    const Index node = blockIdx.x;
    if (node >= num_nodes)
        return;
    const Index begin = __ldg(rowptr + node), end = __ldg(rowptr + node + 1);
    switch (blockIdx.z) {{ {query_cases} }}
}}
std::vector<std::vector<at::Tensor>> launch({input_parameters}, at::Tensor src, at::Tensor edge_by_dst, at::Tensor rowptr,
                                            int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    std::vector<std::vector<at::Tensor>> q_groups;
{allocations}
    auto stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0)
        lunatp_uvwconv_bwd_q_shared_fp32<<<dim3(static_cast<unsigned int>(num_nodes), {specs_count}, {queries_count}),
                                           {threads}, 0, stream>>>({arguments});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch", &launch);
}}
""".format(
            threads=threads,
            input_pointers=",".join(
                (f"const float* __restrict__ ptr_{name}" for name in arrays)
            ),
            output_pointers=",".join(output_pointers),
            query_cases=chr(10).join(query_cases),
            input_parameters=",".join((f"at::Tensor {name}" for name in inputs)),
            allocations=textwrap.indent(chr(10).join(allocations), '    '),
            specs_count=len(specs),
            queries_count=len(queries),
            arguments=",".join(arguments),
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_output_cutlass_source(meta):
    """Contract grouped Q and weights directly into the node output layout."""
    groups, group_rows, _ = get_uvw_conv_weight_layout(meta["generated_specs"])
    output_tiles = " + ".join(f"((nodes*int64_t({key[2]})+63)/64)*{(key[1]+63)//64}" for key, _ in groups)
    declarations = []
    cases = []
    for group, ((offset, channels, d), specs) in enumerate(groups):
        stores = f'atomicAdd(pointer+int64_t(r/{d})*{meta["out_dim"]}+{offset}+int64_t(c)*{d}+r%{d},value);'
        weight_rows = '\n'.join(f'if(q.strided()>={t["row_offset"]} && q.strided()<{t["row_offset"]+t["path_channels"]}) return int64_t({t["weight_offset"]})+int64_t(q.strided()-{t["row_offset"]})*{channels}+q.contiguous();' for t in specs)
        declarations.append(f'''
struct WeightLayout{group} {{
    using Index = int32_t;
    using LongIndex = int64_t;
    CUTLASS_HOST_DEVICE WeightLayout{group}(cutlass::layout::PitchLinearCoord, Index) {{
    }}
    CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord q) const {{
{textwrap.indent(weight_rows, '        ')}
        return 0;
    }}
}};
template <typename Iterator> struct OutputIterator{group} : Iterator {{
    using ThreadMap = typename Iterator::ThreadMap;
    using Fragment = typename Iterator::Fragment;
    using Params = typename Iterator::Params;
    float *pointer;
    CUTLASS_DEVICE OutputIterator{group}(Params const &p, float *out, cutlass::MatrixCoord extent, int thread,
    cutlass::MatrixCoord offset = {{}}, int const *indices = nullptr)
    : Iterator(p, out, extent, thread, offset, indices), pointer(out) {{
    }}
    CUTLASS_DEVICE void store(Fragment const &fragment) const {{
        typename Iterator::Mask mask;
        this->get_mask(mask);
#pragma unroll
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {{
#pragma unroll
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {{
#pragma unroll
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {{
                    int r = this->thread_start_row() + row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                    cluster * ThreadMap::Delta::kCluster;
                    int fr = row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
#pragma unroll
                    for (int col = 0; col < ThreadMap::Iterations::kColumn; ++col) {{
                        int c = this->thread_start_column() + col * ThreadMap::Delta::kColumn;
                        float value = fragment[fr * ThreadMap::Iterations::kColumn + col];
                        if (r < this->extent_row() && mask.predicates[col]) {{
                            {stores}
                        }}
                    }}
                }}
            }}
        }}
    }}
}};
using Gemm{group} = cutlass::gemm::device::GemmUniversal<
float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 8>,
cutlass::gemm::GemmShape<32, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, cutlass::arch::OpMultiplyAdd,
cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone, false, false, false, cutlass::layout::NoPermute,
cutlass::layout::NoPermute, WeightLayout{group}>;
''')
        cases.append(f'''
case {group}: {{
    using Kernel=typename Gemm{group}::GemmKernel;
    using Mma=typename Kernel::Mma;
    using Base=typename Kernel::Epilogue;
    using Epilogue=cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,typename Base::WarpMmaOperator,Base::kPartitionsK,
    OutputIterator{group}<typename Base::OutputTileIterator>,typename Base::AccumulatorFragmentIterator,typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,typename Base::OutputOp,typename Base::Padding>;
    auto& storage=shared.s{group};
    const int64_t first_node=int64_t(blockIdx.y)*64/{d};
    const int row=int64_t(blockIdx.y)*64%{d},column=blockIdx.x*64;
    if(first_node>=nodes || column>={channels}) return;
    const int rows=min(int64_t(64+{d}),(nodes-first_node)*{d});
    const int total_k={group_rows[group]},tiles=(total_k+7)/8,per_split=(tiles+splits-1)/splits;
    const int begin=split*per_split*8,end=min(total_k,begin+per_split*8);
    if(begin>=end) return;
    typename Mma::FragmentC accum;accum.clear();
#pragma unroll
    for(int p=0;p<Products;++p) {{
        typename Mma::IteratorA a(typename Mma::IteratorA::Params(cutlass::layout::ColumnMajor(nodes*{d})),
        const_cast<float*>(inputs.q[p][{group}]+first_node*{d}),{{rows,end}},threadIdx.x,{{row,begin}});
        typename Mma::IteratorB b(typename Mma::IteratorB::Params(cutlass::layout::RowMajor({channels})),
        const_cast<float*>(inputs.w[p]),{{end,{channels}}},threadIdx.x,{{begin,column}});
        Mma mma(storage.main_loop,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
        mma((end-begin+7)/8,accum,a,b,accum);__syncthreads();
    }}
    typename Epilogue::OutputTileIterator output(typename Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor({channels})),out+first_node*{meta['out_dim']},{{rows,{channels}}},threadIdx.x,{{row,column}});
    Epilogue epilogue(storage.epilogue,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    epilogue(typename Kernel::EpilogueOutputOp(1.0f,0.0f),output,accum,output);
    return;
}}
''')
    source = f'''
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cutlass/gemm/device/gemm_universal.h>
#include <limits.h>
struct Inputs {{const float* q[2][{len(groups)}];const float* w[2];}};
{''.join(declarations)}
template<int Products> __global__ void lunatp_uvwconv_bwd_output_shared_fp32(Inputs inputs,float* out,int64_t nodes) {{
    union Storage {{ {' '.join(f"typename Gemm{i}::GemmKernel::SharedStorage s{i};" for i in range(len(groups)))} }};
    __shared__ Storage shared;
    const int64_t tiles={output_tiles};
    const int splits=min(int64_t({MAX_OUTPUT_SPLITS}),max(int64_t(1),({TARGET_OUTPUT_BLOCKS}+tiles-1)/tiles));
    const int split=blockIdx.z%splits;
    switch(blockIdx.z/splits) {{{chr(10).join(cases)}}}
}}

at::Tensor launch_pair(std::vector<at::Tensor> q0,std::vector<at::Tensor> q1,at::Tensor w0,at::Tensor w1,int64_t nodes) {{
    Inputs inputs{{}};
    for(int i=0;i<{len(groups)};++i) {{inputs.q[0][i]=q0[i].data_ptr<float>();inputs.q[1][i]=q1[i].data_ptr<float>();}}
    inputs.w[0]=w0.data_ptr<float>();inputs.w[1]=w1.data_ptr<float>();
    c10::cuda::CUDAGuard guard(w0.device());
    auto out=at::zeros({{nodes,int64_t({meta['out_dim']})}},w0.options());
    if(nodes==0) return out;
    const int64_t tiles={output_tiles};
    const int splits=std::min<int64_t>({MAX_OUTPUT_SPLITS},std::max<int64_t>(1,({TARGET_OUTPUT_BLOCKS}+tiles-1)/tiles));
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_output_shared_fp32<2><<<dim3({max((key[1]+63)//64 for key,_ in groups)},(nodes*{max(key[2] for key,_ in groups)}+63)/64,{len(groups)}*splits),128,0,stream>>>(inputs,out.data_ptr<float>(),nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}}
void accumulate(std::vector<at::Tensor> q,at::Tensor weight,at::Tensor out) {{
    c10::cuda::CUDAGuard guard(out.device());
    const int64_t nodes=out.size(0);
    if(nodes==0) return;
    Inputs inputs{{}};
    for(int i=0;i<{len(groups)};++i) inputs.q[0][i]=q[i].data_ptr<float>();
    inputs.w[0]=weight.data_ptr<float>();
    const int64_t tiles={output_tiles};
    const int splits=std::min<int64_t>({MAX_OUTPUT_SPLITS},std::max<int64_t>(1,({TARGET_OUTPUT_BLOCKS}+tiles-1)/tiles));
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_output_shared_fp32<1><<<dim3({max((key[1]+63)//64 for key,_ in groups)},(nodes*{max(key[2] for key,_ in groups)}+63)/64,{len(groups)}*splits),128,0,stream>>>(inputs,out.data_ptr<float>(),nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch_pair",&launch_pair);m.def("accumulate",&accumulate);}}
'''
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(
    op,
    x,
    edge_attr,
    weight,
    grad_out,
    rx,
    ry,
    rw,
    *,
    graph,
    needs=None,
):
    set_arch(op, x.get_device())
    num_nodes = graph.num_nodes
    if op.weight_mode == "shared":
        if op.node_contraction is None:
            load_node_contraction(op)
        if op.weight_contraction is None:
            load_weight_contraction(op)
        if op.output_contraction is None:
            op.output_contraction = load_cuda_extension(
                f"lunatp_uvwconv_bwd_output_shared_{op.dtype_tag}",
                make_uvw_conv_output_cutlass_source(
                    op.meta
                ),
                arch=op.arch,
                use_fast_math=False,
                index_bits=op.index_bits,
            )

        current_stream = torch.cuda.current_stream(x.device)
        contract_stream = auxiliary_streams(
            current_stream.device_index, 1, scope=current_stream.cuda_stream
        )[0]
        contract_stream.wait_stream(current_stream)
        # Both consumers share the same CG pair and stream. Their temporary
        # tensors can be released in stream order, without cross-stream holds.
        with torch.cuda.stream(contract_stream):
            q1 = op.precontract_q_pair(
                rx,
                edge_attr,
                x,
                ry,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            q0 = op.precontract_q(
                x,
                edge_attr,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            grad_weight = op.weight_reduce.launch(q1, grad_out)
            grad_grad_out = op.output_contraction.launch_pair(
                q1, q0, weight, rw, num_nodes
            )
            del q0, q1

        z_w = op.precontract_z(weight, grad_out, num_nodes)
        z_rw = op.precontract_z(rw, grad_out, num_nodes)
        grad_x, grad_edge = op.pair_from_z(
            ry,
            z_w,
            edge_attr,
            z_rw,
            rx,
            x,
            graph.src,
            graph.dst,
            graph.src_rowptr,
            graph.edge_ids_by_src,
            num_nodes,
        )
        current_stream.wait_stream(contract_stream)
        grad_weight.record_stream(current_stream)
        grad_grad_out.record_stream(current_stream)
        return grad_x, grad_edge, grad_weight, grad_grad_out
    return run_external(
        op, 2, (x, edge_attr, weight, grad_out, rx, ry, rw), graph, needs=needs
    )
