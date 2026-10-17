"""UVW convolution double backward CUDA and native launch."""

from __future__ import annotations
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import get_fp64_mma, cuda_source_for_dtype
from ....utils.extension import load_cuda_extension
from ...utils.runtime import auxiliary_streams
from .backward import make_uvw_conv_gemm_mma, run_external, load_node_contraction, load_weight_contraction
from .utils import get_uvw_conv_weight_layout, set_arch


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
                f"for({channel_type} p=threadIdx.x;p<{channels};p+=blockDim.x) {{",
                f"int u=p/{edges},v=p%{edges};",
                *[f"float acc{k}=0.f;" for k in range(d3)],
                "for(Index pos=begin;pos<end;++pos) {",
                "const Index item=__ldg(edge_by_dst+pos);",
                "const int source_node=__ldg(src+pos);",
            ]
            for product in products:
                pair = tuple(
                    f"ptr_{role}" if role is not None else None for role in product
                )
                lines += [
                    "{",
                    emit_cg_source(
                        meta,
                        spec,
                        pairs=(pair,),
                        x_row="source_node",
                        coefficient_first=False,
                        accumulator="acc",
                    ),
                    "}",
                ]
            lines.append("}")
            lines += [
                f"q_{query_index}_{group_index}[({row_offset}+p)*(num_nodes*int64_t({d3}))+Index(node)*{d3}+{k}]=acc{k};"
                for k in range(d3)
            ]
            lines += ["}", "break;", "}"]
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
__global__ __launch_bounds__({threads},4)
void lunatp_uvwconv_bwd_q_shared_fp32({input_pointers},const int32_t* src,const EdgeIndex* edge_by_dst,const EdgeIndex* rowptr,{output_pointers},int64_t num_nodes) {{
    const Index node=blockIdx.x;
    if(node>=num_nodes) return;
    const Index begin=__ldg(rowptr+node),end=__ldg(rowptr+node+1);
    switch(blockIdx.z) {{ {query_cases} }}
}}
std::vector<std::vector<at::Tensor>> launch({input_parameters},at::Tensor src,at::Tensor edge_by_dst,at::Tensor rowptr,int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    std::vector<std::vector<at::Tensor>> q_groups;
    {allocations}
    auto stream=at::cuda::getCurrentCUDAStream();
    if(num_nodes>0) lunatp_uvwconv_bwd_q_shared_fp32<<<dim3(static_cast<unsigned int>(num_nodes),{specs_count},{queries_count}),{threads},0,stream>>>({arguments});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{ m.def("launch",&launch); }}
""".format(
            threads=threads,
            input_pointers=",".join(
                (f"const float* __restrict__ ptr_{name}" for name in arrays)
            ),
            output_pointers=",".join(output_pointers),
            query_cases=chr(10).join(query_cases),
            input_parameters=",".join((f"at::Tensor {name}" for name in inputs)),
            allocations=chr(10).join(allocations),
            specs_count=len(specs),
            queries_count=len(queries),
            arguments=",".join(arguments),
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_double_backward_output_shared_source(meta, *, arch):
    groups, rows, _ = get_uvw_conv_weight_layout(meta["generated_specs"])
    cases = []
    a_store = "i"
    b_store = "i"
    mma = {
        "declaration": r"""using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<2,4,1>,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC acc;acc.clear();""",
        "compute": r"""#pragma unroll 4
        for(int k=0;k<32;++k) {
            #pragma unroll
            for(int p=0;p<Products;++p) {
                Mma::FragmentA av;av[0]=a[p][k*16+local_row];av[1]=a[p][k*16+local_row+1];
                Mma::FragmentB bv;
                #pragma unroll
                for(int c=0;c<4;++c) bv[c]=b[p][k*64+local_column+c*16];
                Mma()(acc,av,bv,acc);
            }
        }""",
        "row": "row_base+local_row+r",
        "column": "column_base+local_column+c*16",
        "value": "acc[r*4+c]",
        "rows": 2,
        "columns": 4,
    }
    if meta.get("dtype") == torch.float64:
        a_store = "(i/16)*16+((i%16)^(((i/16)&3)<<2))"
        b_store = "(i/64)*64+((i%64)^(((i/64)&3)<<2))"
        mma = make_uvw_conv_gemm_mma(
            meta,
            "a[p][input*16+(row^((input&3)<<2))]",
            "b[p][input*64+(column^((input&3)<<2))]",
            arch=arch,
        )
    for group, ((offset, channels, d3), specs) in enumerate(groups):
        path_type = "int64_t" if rows[group] > (1 << 31) - 32 else "int"
        bound_type = "int64_t" if path_type == "int64_t" else "Index"
        dimensions = f"constexpr int K={rows[group]},Columns={channels};"
        weight_position = "int k=base+i/64,column=column_base+i%64;"
        q_position = "Index row=row_base+i%16,k=base+i/16;"
        if path_type == "int64_t":
            dimensions = (
                f"constexpr int64_t K={rows[group]}; constexpr int Columns={channels};"
            )
            weight_position = "int64_t k=base+i/64; int column=column_base+i%64;"
            q_position = "Index row=row_base+i%16; int64_t k=base+i/16;"
        weights = []
        for s in specs:
            begin = s["row_offset"]
            end = begin + s["path_channels"]
            weights.append(
                f"""if(k>={begin} && k<{end}) pointer=inputs.w[p]+{s["weight_offset"]}+int64_t(k-{begin})*{channels}+column;"""
            )
        cases.append(
            r"""
case {group}: {{
    {dimensions}
    const Index total_m=nodes*{d3};
    if(row_base>=total_m || column_base>=Columns) return;
    const {bound_type} tiles=(K+31)/32,per_split=(tiles+3)/4;
    const {bound_type} begin=split*per_split*32,end=min({bound_type}(K),begin+per_split*32);
    if(begin>=end) return;
    {declaration}
    for({path_type} base=begin;base<end;base+=32) {{
        #pragma unroll
        for(int p=0;p<Products;++p) {{
            for(int i=threadIdx.x;i<512;i+=128) {{
                {q_position}
                a[p][{a_store}]=(row<total_m && k<end)?__ldg(inputs.q[p][{group}]+int64_t(k)*total_m+row):0.f;
            }}
            for(int i=threadIdx.x;i<2048;i+=128) {{
                {weight_position}
                const float* pointer=inputs.w[p];
                bool valid=k<end && column<Columns;
                if(valid) {{
                    {weights}
                }}
                b[p][{b_store}]=valid?__ldg(pointer):0.f;
            }}
        }}
        __syncthreads();
        {compute}
        __syncthreads();
    }}
    #pragma unroll
    for(int r=0;r<{rows};++r) {{
        Index row={row};
        #pragma unroll
        for(int c=0;c<{columns};++c) {{
            int column={column};
            if(row<total_m && column<Columns) atomicAdd(out+int64_t(row/{d3})*{out_dim}+{offset}+column*{d3}+row%{d3},{value});
        }}
    }}
    return;
}}
""".format(
                group=group,
                dimensions=dimensions,
                d3=d3,
                bound_type=bound_type,
                declaration=mma["declaration"],
                path_type=path_type,
                q_position=q_position,
                a_store=a_store,
                weight_position=weight_position,
                weights=chr(10).join(weights),
                b_store=b_store,
                compute=mma["compute"],
                rows=mma["rows"],
                row=mma["row"],
                columns=mma["columns"],
                column=mma["column"],
                out_dim=meta["out_dim"],
                offset=offset,
                value=mma["value"],
            )
        )
    out_tiles = (max(channels for (_, channels, _), _ in groups) + 63) // 64
    max_d3 = max(components for (_, _, components), _ in groups)
    mma_include = (
        "\n" + get_fp64_mma(arch)[0] if meta.get("dtype") == torch.float64 else ""
    )
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>{mma_include}
#include <vector>
struct Inputs {{const float* q[2][{groups_count}];const float* w[2];}};
template<int Products>
__global__ __launch_bounds__(128,4)
void lunatp_uvwconv_bwd_output_shared_fp32(Inputs inputs,float* out,Index nodes) {{
    __shared__ float a[Products][512];
    __shared__ float b[Products][2048];
    const Index row_base=blockIdx.y*16; const int column_base=blockIdx.x*64;
    const int local_row=(threadIdx.x/16)*2,local_column=threadIdx.x%16;
    const int split=blockIdx.z%4;
    switch(blockIdx.z/4) {{
{cases}
    }}
}}
at::Tensor launch_pair(std::vector<at::Tensor> q0,std::vector<at::Tensor> q1,at::Tensor w0,at::Tensor w1,int64_t nodes) {{
    Inputs inputs{{}};
    for(int i=0;i<{groups_count};++i) {{inputs.q[0][i]=q0[i].data_ptr<float>();inputs.q[1][i]=q1[i].data_ptr<float>();}}
    inputs.w[0]=w0.data_ptr<float>();inputs.w[1]=w1.data_ptr<float>();
    c10::cuda::CUDAGuard guard(w0.device());
    auto out=at::zeros({{nodes,int64_t({out_dim})}},w0.options());
    if(nodes==0) return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_output_shared_fp32<2><<<dim3({out_tiles},(nodes*{max_d3}+15)/16,{output_tasks}),128,0,stream>>>(inputs,out.data_ptr<float>(),nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}}
void accumulate(std::vector<at::Tensor> q,at::Tensor weight,at::Tensor out) {{
    c10::cuda::CUDAGuard guard(out.device());
    const int64_t nodes=out.size(0);
    if(nodes==0) return;
    Inputs inputs{{}};
    for(int i=0;i<{groups_count};++i) inputs.q[0][i]=q[i].data_ptr<float>();
    inputs.w[0]=weight.data_ptr<float>();
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_output_shared_fp32<1><<<dim3({out_tiles},(nodes*{max_d3}+15)/16,{output_tasks}),128,0,stream>>>(inputs,out.data_ptr<float>(),nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch_pair",&launch_pair);m.def("accumulate",&accumulate);}}
""".format(
            mma_include=mma_include,
            groups_count=len(groups),
            cases=chr(10).join(cases),
            out_dim=meta["out_dim"],
            out_tiles=out_tiles,
            max_d3=max_d3,
            output_tasks=len(groups) * 4,
        ),
        meta.get("dtype", torch.float32),
    )


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
    src, dst = graph.src, graph.dst
    num_nodes = graph.num_nodes
    if op.weight_mode == "shared":
        if op.node_contraction is None:
            load_node_contraction(op)
        if op.weight_contraction is None:
            load_weight_contraction(op)
        if op.output_contraction is None:
            op.output_contraction = load_cuda_extension(
                f"lunatp_uvwconv_bwd_output_shared_{op.dtype_tag}",
                make_uvw_conv_double_backward_output_shared_source(
                    op.meta, arch=op.arch
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
    dx, dy, dw, dg = run_external(
        op, 2, (x, edge_attr, weight, grad_out, rx, ry, rw), src, dst, needs=needs
    )
    return dx, dy, dw, dg
