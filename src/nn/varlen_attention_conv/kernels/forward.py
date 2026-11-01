"""Online CUTLASS attention with envelope and CSR pooling in the epilogue."""

from functools import lru_cache

import torch

from .utils import input_layout, load_kernel


SOURCE = r"""
#include <cute/tensor.hpp>
constexpr int ForwardThreads=!RETURN_OUT && !LOW_PRECISION ? 256:128;
constexpr int ForwardWorkers=LOW_PRECISION ? 1:4;
#if !RETURN_OUT
constexpr int NodeWarps=LOW_PRECISION ? 4:8,NodeThreads=NodeWarps*32;
constexpr int NodeRows=64;
constexpr int NodeKeys=LOW_PRECISION ? 64:32;
// Keep 1 KiB below the 48 KiB static limit for launch resources. The replay
// cache and alpha consume 17 KiB, leaving a 32-aligned Q/K capacity of 96.
constexpr int NodeSharedBudget=47*1024;
constexpr int NodeReplayBytes=(NodeRows*NodeKeys+4*NodeKeys)*sizeof(float);
constexpr int NodeCapacity=(NodeSharedBudget-NodeReplayBytes)/
    ((NodeRows+NodeKeys)*sizeof(GemmElement))/32*32;
constexpr int NodeReduction=LOW_PRECISION ? ((DIM+31)/32*32<NodeCapacity ? (DIM+31)/32*32:NodeCapacity):32;
constexpr bool NodeStageValue=LOW_PRECISION && VALUE_DIM<=NodeReduction;
constexpr int NodeQuad=LOW_PRECISION ? 4:8;
constexpr int NodeColumns=LOW_PRECISION ? 2:4;
constexpr int NodeGroups=NodeKeys/(NodeQuad*NodeColumns);
using NodeScoreShape=cutlass::gemm::GemmShape<NodeRows,NodeKeys,NodeReduction>;
#if LOW_PRECISION
using NodeScoreOperator=typename cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<16,64,NodeReduction>,GemmInstruction,
    GemmElement,cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16,32>,
    GemmElement,cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16,32>,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>::Type;
using NodeScorePolicy=cutlass::gemm::threadblock::MmaPolicy<NodeScoreOperator,
    cutlass::MatrixShape<0,0>,cutlass::MatrixShape<0,0>,1>;

#else
using NodeScoreCore=cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<64,32,8>,cutlass::gemm::GemmShape<8,32,8>,GemmInstruction,
    GemmElement,cutlass::layout::RowMajor,GemmElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,GemmClass,2,cutlass::arch::OpMultiplyAdd>;
using NodeScorePolicy=typename NodeScoreCore::MmaPolicy;
using NodeScoreOperator=typename NodeScorePolicy::Operator;
#endif
using NodeScoreAccumulator=typename NodeScoreOperator::FragmentC;
#if LOW_PRECISION
using NodeScoreStorage=typename cutlass::gemm::threadblock::MmaBase<
    NodeScoreShape,NodeScorePolicy,1>::SharedStorage;
#else
namespace native_score {
using namespace cute;
using Mma=decltype(make_tiled_mma(UniversalFMA<float,float,float>{},
    Layout<Shape<_4,_8,_1>,Stride<_8,_1,_0>>{},
    Tile<Layout<Shape<_4,_2>,Stride<_2,_1>>,
         Layout<Shape<_8,_4>,Stride<_4,_1>>,_1>{}));
using Copy=decltype(make_tiled_copy(Copy_Atom<UniversalCopy<float>,float>{},
    Layout<Shape<_32,_8>,Stride<_8,_1>>{},Layout<Shape<_1,_1>>{}));
using ScoreLayout=Layout<Shape<_1,_2,_4>,Stride<_0,_4,_1>>;
using QueryLayout=Layout<Shape<_64,_32>,Stride<_1,Int<68>>>;
using KeyLayout=Layout<Shape<_32,_32>,Stride<_1,Int<36>>>;
struct alignas(16) Storage {
    float query[2*cosize_v<QueryLayout>],key[2*cosize_v<KeyLayout>];
};

__device__ __forceinline__ void prefetch(Storage& shared,Params const& p,
    int slot,int64_t qb,int64_t kb,int64_t end,int head,int channel) {
    auto query=make_tensor(make_smem_ptr(shared.query+slot*cosize_v<QueryLayout>),QueryLayout{});
    auto key=make_tensor(make_smem_ptr(shared.key+slot*cosize_v<KeyLayout>),KeyLayout{});
    auto thread=Copy{}.get_slice(threadIdx.x);
    auto qd=thread.partition_D(query);
    auto kd=thread.partition_D(key);
    auto qc=thread.partition_S(make_identity_tensor(Shape<_64,_32>{}));
    auto kc=thread.partition_S(make_identity_tensor(Shape<_32,_32>{}));
    CUTE_UNROLL
    for(int i=0;i<size(qd);++i) {
        auto coord=qc(i);
        const int row=int(get<0>(coord)),column=int(get<1>(coord));
        const bool valid=qb+row<end && channel+column<DIM;
        auto* source=p.in[0].data+(valid ? input_offset(p,0,0,qb+row,head,channel+column):0);
        cutlass::arch::cp_async_zfill<4,cutlass::arch::CacheOperation::Always>(&qd(i),source,valid);
    }
    CUTE_UNROLL
    for(int i=0;i<size(kd);++i) {
        auto coord=kc(i);
        const int row=int(get<0>(coord)),column=int(get<1>(coord));
        const bool valid=kb+row<end && channel+column<DIM;
        auto* source=p.in[1].data+(valid ? input_offset(p,1,1,kb+row,head,channel+column):0);
        cutlass::arch::cp_async_zfill<4,cutlass::arch::CacheOperation::Always>(&kd(i),source,valid);
    }
    cutlass::arch::cp_async_fence();
}
}
using NodeScoreStorage=native_score::Storage;
#endif

#if LOW_PRECISION
template<bool Query>
__device__ __forceinline__ void node_score_load(NodeScoreStorage& shared,
    Params const& p,int64_t begin,int64_t end,int head,int channel) {
    constexpr int Slot=Query ? 0:1,Rows=Query ? NodeRows:NodeKeys;
    constexpr int InputType=(INPUT_DTYPES>>(2*Slot))&3u;
    if constexpr(!(DIM%8) && (CONTIGUOUS&(1u<<Slot)) && InputType==(LOWP_BF16 ? 2:1)) {
        auto* base=static_cast<GemmElement const*>(p.in[Slot].data);
        if((reinterpret_cast<uintptr_t>(base)&15u)==0) {
            for(int pos=threadIdx.x*8;pos<Rows*NodeReduction;pos+=128*8) {
                const int row=pos/NodeReduction,column=pos%NodeReduction;
                const bool valid=begin+row<end && channel+column<DIM;
                auto* pointer=base+(valid ? input_offset(p,Slot,Slot,begin+row,head,channel+column):0);
                GemmElement* destination;
                if constexpr(Query) destination=&shared.operand_A_ref().at({row,column});
                else destination=&shared.operand_B_ref().at({column,row});
                cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(destination,pointer,valid);
            }
            return;
        }
    }
    for(int pos=threadIdx.x;pos<Rows*NodeReduction;pos+=128) {
        const int row=pos/NodeReduction,column=pos%NodeReduction;
        const GemmElement value=GemmElement(begin+row<end && channel+column<DIM ?
            input<1>(p,Slot,begin+row,head,channel+column).x[0]:0.f);
        if constexpr(Query) shared.operand_A_ref().at({row,column})=value;
        else shared.operand_B_ref().at({column,row})=value;
    }
}

// Once QK has completed, the K operand is free until the next score tile.
__device__ __forceinline__ void node_value_load(NodeScoreStorage& shared,
    Params const& p,int64_t begin,int64_t end,int head) {
    auto* tile=shared.operand_B.data();
    constexpr int InputType=(INPUT_DTYPES>>4)&3u;
    if constexpr(!(VALUE_DIM%8) && (CONTIGUOUS&4u) && InputType==(LOWP_BF16 ? 2:1)) {
        auto* base=static_cast<GemmElement const*>(p.in[2].data);
        if((reinterpret_cast<uintptr_t>(base)&15u)==0) {
            for(int pos=threadIdx.x*8;pos<NodeKeys*NodeReduction;pos+=128*8) {
                const int row=pos/NodeReduction,column=pos%NodeReduction;
                const bool valid=begin+row<end && column<VALUE_DIM;
                auto* pointer=base+(valid ? input_offset(p,2,2,begin+row,head,column):0);
                cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(tile+pos,pointer,valid);
            }
            return;
        }
    }
    for(int pos=threadIdx.x;pos<NodeKeys*NodeReduction;pos+=128) {
        const int row=pos/NodeReduction,column=pos%NodeReduction;
        tile[pos]=GemmElement(begin+row<end && column<VALUE_DIM ? input<1>(p,2,begin+row,head,column).x[0]:0.f);
    }
}
#endif

// CUTLASS TensorOp and SIMT accumulator coordinates, with each warp owning its fixed query rows.
__device__ __forceinline__ int node_score_row(int lane,int warp,int r) {
    return warp*(NodeRows/NodeWarps)+(LOW_PRECISION ? (lane/4+r*8):(lane/8*2+r));
}
__device__ __forceinline__ int node_score_index(int r,int n,int j) {
    return LOW_PRECISION ? n*4+r*2+j:(r*NodeGroups+n)*4+j;
}

template<bool Forward>
__device__ __forceinline__ NodeScoreAccumulator node_score_fragment(Params const& p,
    NodeScoreStorage& shared,int64_t atom,int64_t begin,int64_t end,int64_t qb,int64_t kb,int head) {
    NodeScoreAccumulator score;
    score.clear();
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
#if LOW_PRECISION
    using Mma=cutlass::gemm::threadblock::MmaSingleStage<NodeScoreShape,
        ConvReadyOperand,ConvReadyOperand,ConvReadyOperand,ConvReadyOperand,
        float,cutlass::layout::RowMajor,NodeScorePolicy>;
    for(int channel=0;channel<DIM;channel+=NodeReduction) {
        if constexpr(DIM>NodeReduction) {
            node_score_load<true>(shared,p,qb,end,head,channel);
            node_score_load<false>(shared,p,kb,end,head,channel);
            cutlass::arch::cp_async_fence();
        } else if constexpr(!Forward) {
            node_score_load<false>(shared,p,kb,end,head,channel);
            cutlass::arch::cp_async_fence();
        }
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
        Mma mma(shared,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
        mma(1,score,ConvReadyOperand{},ConvReadyOperand{},score);
    }
    if constexpr(Forward && DIM<=NodeReduction) {
        if(kb+NodeKeys<end) {
            node_score_load<false>(shared,p,kb+NodeKeys,end,head,0);
            cutlass::arch::cp_async_fence();
        }
    }
    if constexpr(NodeStageValue) {
        if(!Forward || kb+NodeKeys>=end) {
            node_value_load(shared,p,kb,end,head);
            cutlass::arch::cp_async_fence();
        }
    }
#else
    using namespace cute;
    native_score::Mma mma;
    auto thread=mma.get_slice(lane);
    auto c=make_tensor(make_rmem_ptr(score.data()),native_score::ScoreLayout{});
    const bool active=qb+warp*8<end;
    native_score::prefetch(shared,p,0,qb,kb,end,head,0);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    for(int channel=0;channel<DIM;channel+=NodeReduction) {
        const int slot=(channel/NodeReduction)%2;
        if(channel+NodeReduction<DIM) {
            native_score::prefetch(shared,p,slot^1,qb,kb,end,head,channel+NodeReduction);
        }
        auto query=make_tensor(make_smem_ptr(shared.query+slot*cosize_v<native_score::QueryLayout>),native_score::QueryLayout{});
        auto key=make_tensor(make_smem_ptr(shared.key+slot*cosize_v<native_score::KeyLayout>),native_score::KeyLayout{});
        auto qa=local_tile(query,Shape<_8,_32>{},make_coord(warp,_0{}));
        auto a=thread.partition_A(qa);
        auto b=thread.partition_B(key);
        if(active) cute::gemm(mma,a,b,c);
        // Producers publish the next panel only after all warps retire this one.
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }
#endif
    #pragma unroll
    for(int n=0;n<NodeGroups;++n) {
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int64_t q=qb+node_score_row(lane,warp,r);
            #pragma unroll
            for(int j=0;j<NodeColumns;++j) {
                const int i=node_score_index(r,n,j);
                const int64_t k=kb+n*(NodeQuad*NodeColumns)+(lane%NodeQuad)*NodeColumns+j;
                float value=-INFINITY;
                if(q<end && k<end && (!CAUSAL || k<=q)) {
                    value=score[i]*p.scale;
                    if constexpr(BIAS_KIND==1) value+=input<1>(p,3,k,head).x[0];
                    if constexpr(BIAS_KIND==2) value+=input<1>(p,3,atom,head,q-begin,k-begin).x[0];
                }
                score[i]=value;
            }
        }
    }
    return score;
}

__device__ __forceinline__ void node_forward_registers(Params const& p,NodeScoreStorage& gemm,
    float (*alpha)[NodeKeys],float* cache,int64_t atom,int head,int query_worker) {
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int64_t begin=p.offset(atom),end=p.offset(atom+1);
#if !LOW_PRECISION
    if(begin+query_worker*NodeRows>=end) return;
#endif
    const bool cache_first=end-begin>NodeKeys;
    typename NodeScoreOperator::IteratorC cached_score(
        {cache+warp*(NodeRows/NodeWarps)*NodeKeys,cutlass::layout::RowMajor(NodeKeys)},lane);
    float node[(VALUE_DIM+NodeThreads-1)/NodeThreads]={};
#if LOW_PRECISION
    for(int64_t qb=begin;qb<end;qb+=NodeRows) {
#else
    for(int64_t qb=begin+query_worker*NodeRows;qb<end;qb+=ForwardWorkers*NodeRows) {
#endif
#if LOW_PRECISION
        if constexpr(DIM<=NodeReduction) {
            node_score_load<true>(gemm,p,qb,end,head,0);
            node_score_load<false>(gemm,p,begin,end,head,0);
            cutlass::arch::cp_async_fence();
        }
#endif
        float maximum[2]={-INFINITY,-INFINITY},denominator[2]={};
        NodeScoreAccumulator score;
        int64_t last_key=begin;
        for(int64_t kb=begin;kb<end;kb+=NodeKeys) {
            score=node_score_fragment<true>(p,gemm,atom,begin,end,qb,kb,head);
            if(cache_first && kb==begin) cached_score.store(score);
            #pragma unroll
            for(int r=0;r<2;++r) {
                float next=-INFINITY;
                #pragma unroll
                for(int n=0;n<NodeGroups;++n) {
                    #pragma unroll
                    for(int j=0;j<NodeColumns;++j) next=fmaxf(next,score[node_score_index(r,n,j)]);
                }
                #pragma unroll
                for(int delta=1;delta<NodeQuad;delta*=2)
                    next=fmaxf(next,__shfl_xor_sync(0xffffffff,next,delta));
                next=fmaxf(next,maximum[r]);
                float sum=0.f;
                #pragma unroll
                for(int n=0;n<NodeGroups;++n) {
                    #pragma unroll
                    for(int j=0;j<NodeColumns;++j) {
                        const float value=score[node_score_index(r,n,j)];
                        sum+=isfinite(value) ? __expf(value-next):0.f;
                    }
                }
                #pragma unroll
                for(int delta=1;delta<NodeQuad;delta*=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta);
                float factor=isfinite(next) ? __expf(maximum[r]-next):1.f;
                maximum[r]=next;
                denominator[r]=denominator[r]*factor+sum;
            }
            last_key=kb;
        }
        #pragma unroll
        for(int r=0;r<2;++r) denominator[r]=denominator[r]>0.f ? 1.f/denominator[r]:0.f;
        for(int64_t kb=last_key;kb>=begin;kb-=NodeKeys) {
            if(kb!=last_key) {
                if(cache_first && kb==begin) cached_score.load(score);
                else score=node_score_fragment<false>(p,gemm,atom,begin,end,qb,kb,head);
            }
            const int row0=node_score_row(lane,warp,0),row1=node_score_row(lane,warp,1);
            const float weight0=qb+row0<end ? denominator[0]*((PRESENT&16u) ? input<1>(p,4,qb+row0).x[0]:1.f):0.f;
            const float weight1=qb+row1<end ? denominator[1]*((PRESENT&16u) ? input<1>(p,4,qb+row1).x[0]:1.f):0.f;
            #pragma unroll
            for(int n=0;n<NodeGroups;++n) {
                #pragma unroll
                for(int j=0;j<NodeColumns;++j) {
                    const float a=score[node_score_index(0,n,j)],b=score[node_score_index(1,n,j)];
                    float sum=(isfinite(a) ? __expf(a-maximum[0])*weight0:0.f)+
                              (isfinite(b) ? __expf(b-maximum[1])*weight1:0.f);
                    #pragma unroll
                    for(int delta=NodeQuad;delta<32;delta*=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta);
                    if(lane<NodeQuad) alpha[warp][n*(NodeQuad*NodeColumns)+lane*NodeColumns+j]=sum;
                }
            }
#if LOW_PRECISION
            if constexpr(NodeStageValue) cutlass::arch::cp_async_wait<0>();
#endif
            __syncthreads();
            if(tid<NodeKeys) {
                alpha[0][tid]=alpha[0][tid]+alpha[1][tid]+alpha[2][tid]+alpha[3][tid];
#if !LOW_PRECISION
                alpha[0][tid]=alpha[0][tid]+alpha[4][tid]+alpha[5][tid]+alpha[6][tid]+alpha[7][tid];
#endif
            }
            __syncthreads();
            for(int c=tid;c<VALUE_DIM;c+=NodeThreads) {
                float sum=0.f;
                for(int k=0;k<NodeKeys && kb+k<end;++k) {
                    float value;
#if LOW_PRECISION
                    if constexpr(NodeStageValue) value=float(gemm.operand_B.data()[k*NodeReduction+c]);
                    else
#endif
                    value=input<1>(p,2,kb+k,head,c).x[0];
                    sum=fmaf(alpha[0][k],value,sum);
                }
                node[c/NodeThreads]+=sum;
            }
            __syncthreads();
#if LOW_PRECISION
            if constexpr(NodeStageValue) {
                if(cache_first && kb-NodeKeys==begin) {
                    node_value_load(gemm,p,begin,end,head);
                    cutlass::arch::cp_async_fence();
                }
            }
#endif
        }
    }
    for(int c=tid;c<VALUE_DIM;c+=NodeThreads) {
#if LOW_PRECISION
        p.out[0].store(p.out[0].index(atom,head,c),node[c/NodeThreads]);
#else
        p.out[0].atomic_add(p.out[0].index(atom,head,c),node[c/NodeThreads]);
#endif
    }
}
#endif

__global__ __launch_bounds__(ForwardThreads) void forward_kernel(const __grid_constant__ Params p) {
#if !RETURN_OUT
    __shared__ NodeScoreStorage score;
    __shared__ float alpha[NodeWarps][NodeKeys],cache[NodeRows*NodeKeys];
    node_forward_registers(p,score,alpha,cache,blockIdx.x,blockIdx.y,int(blockIdx.z));
#else
    __shared__ union {
        AttentionGemmStorage score;
        AttentionGemmTile<32,true> output;
    } gemm;
    __shared__ float scores[1][1024], product[1024], output[1024];
    __shared__ float maximum[32], denominator[32], correction[32];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
#if !LOW_PRECISION
    const int query_worker=blockIdx.z;
    if(query_worker>0 && begin+query_worker*32>=end) return;
#endif
    for(int channel=0;channel<VALUE_DIM;channel+=32) {
        float aggregate=0.f;
#if LOW_PRECISION
        for(int64_t qb=begin;qb<end;qb+=32) {
#else
        for(int64_t qb=begin+query_worker*32;qb<end;qb+=4*32) {
#endif
            if(tid<32) { maximum[tid]=-INFINITY; denominator[tid]=0.f; }
            for(int i=tid;i<1024;i+=128) output[i]=0.f;
            __syncthreads();
            for(int64_t kb=begin;kb<end;kb+=32) {
                attention_scores<1>(p,gemm.score,scores,atom,begin,end,qb,kb,head);
                for(int row=warp;row<32;row+=4) {
                    float score=scores[0][row*32+lane],next=score;
                    #pragma unroll
                    for(int delta=16;delta;delta/=2)
                        next=fmaxf(next,__shfl_xor_sync(0xffffffff,next,delta));
                    next=fmaxf(maximum[row],next);
                    float factor=isfinite(next) ? __expf(maximum[row]-next):1.f;
                    float weight=isfinite(score) ? __expf(score-next):0.f;
                    float sum=attention_sum(weight);
                    __syncwarp();
                    scores[0][row*32+lane]=weight;
                    if(lane==0) {
                        maximum[row]=next;
                        denominator[row]=denominator[row]*factor+sum;
                        correction[row]=factor;
                    }
                }
                __syncthreads();
                attention_gemm(gemm.output,product,32,
                    [&](int row,int k) {return scores[0][row*32+k];},
                    [&](int k,int column) {return kb+k<end && channel+column<VALUE_DIM ?
                        input<1>(p,2,kb+k,head,channel+column).x[0]:0.f;});
                for(int i=tid;i<1024;i+=128) output[i]=output[i]*correction[i/32]+product[i];
                __syncthreads();
            }
            if(tid<32) denominator[tid]=denominator[tid]>0.f ? 1.f/denominator[tid]:0.f;
            __syncthreads();
            for(int i=tid;i<1024;i+=128) {
                int row=i/32,c=channel+i%32;
                if(qb+row<end && c<VALUE_DIM)
                    p.out[1].store(p.out[1].index(qb+row,head,c),output[i]*denominator[row]);
            }
            if(tid<32 && channel+tid<VALUE_DIM) {
                for(int row=0;row<32 && qb+row<end;++row) {
                    float env=(PRESENT&16u) ? input<1>(p,4,qb+row).x[0]:1.f;
                    aggregate=fmaf(output[row*32+tid]*denominator[row],env,aggregate);
                }
            }
            __syncthreads();
        }
        if(tid<32 && channel+tid<VALUE_DIM) {
#if LOW_PRECISION
            p.out[0].store(p.out[0].index(atom,head,channel+tid),aggregate);
#else
            p.out[0].atomic_add(p.out[0].index(atom,head,channel+tid),aggregate);
#endif
        }
        __syncthreads();
    }
#endif
}

#if !LOW_PRECISION
__global__ void initialize_nodes(const __grid_constant__ Params p) {
    for(int c=threadIdx.x;c<VALUE_DIM;c+=128)
        p.out[0].store(p.out[0].index(blockIdx.x,blockIdx.y,c),0.f);
}
#endif

void run(std::vector<std::optional<at::Tensor>> inputs,at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if(csr.numel()>1) {
        Params p=params(inputs,outputs,csr,scale);
#if !LOW_PRECISION
        initialize_nodes<<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        forward_kernel<<<dim3(csr.numel()-1,HEADS,ForwardWorkers),ForwardThreads,0,at::cuda::getCurrentCUDAStream()>>>(p);
#else
        forward_kernel<<<dim3(csr.numel()-1,HEADS),ForwardThreads,0,at::cuda::getCurrentCUDAStream()>>>(p);
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
"""


@lru_cache(maxsize=128)
def get_forward_kernel(config, return_attn_out, device, dtype, layout):
    source = f"#define RETURN_OUT {int(return_attn_out)}\n" + SOURCE
    return load_kernel("attention_conv_forward", source, config, device.index, dtype, layout).run


def attention_conv_forward(inputs, csr, config, return_attn_out):
    value = inputs[2]
    output = torch.empty((csr.numel() - 1, *value.shape[1:]), device=value.device, dtype=value.dtype)
    edge_output = (
        torch.empty(value.shape, device=value.device, dtype=value.dtype) if return_attn_out else None
    )
    get_forward_kernel(config, return_attn_out, value.device, value.dtype, input_layout(inputs))(
        inputs, csr, (output, edge_output), config[3]
    )
    return output, edge_output
