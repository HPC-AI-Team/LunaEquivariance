"""Recompute attention tiles and fuse all first-order input gradients."""

from functools import lru_cache

import torch

from .utils import finish_gradients, input_layout, load_kernel


GRADIENT_SOURCE = r"""
constexpr unsigned GradientComponents=1u;
template<int N>
__device__ __forceinline__ Differential<N> softmax_gradient(Differential<N> probability,Differential<N> dot,Differential<N> mean) {
    return probability*(dot-mean);
}
template<int N>
__device__ __forceinline__ void gradient(Params const& p,int slot,Differential<N> value,
    int64_t i,int64_t j=0,int64_t k=0,int64_t l=0) {
    auto const& output=p.out[slot];
    if(NEEDS&(1u<<slot)) {
        if(slot==4 || (QUERY_WORKERS>1 && (slot==1 || slot==2 || (slot==3 && BIAS_KIND==1)))) output.atomic_add(output.index(i,j,k,l),value.x[0]);
        else output.accumulate(output.index(i,j,k,l),value.x[0]);
    }
}
template<int N>
__device__ __forceinline__ void upstream_gradient(Params const&,int,Differential<N>,int64_t,int,int) {}
"""


BACKWARD_SOURCE = r"""

#ifndef PACKED_GRADIENTS
#define PACKED_GRADIENTS 0
#endif
#if defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD
constexpr int BackwardColumns=64;
#else
constexpr int BackwardColumns=32;
#endif
#if PACKED_GRADIENTS
template<int N>
struct BackwardStorage {
    union {
        PackedScoreStorage score;
        PackedOutputStorage output;
    } gemm;
    float probability[N][1024];
    float maximum[32];
    Differential<N> denominator[32],delta[32],node_dot[32],alpha[32];
};

#else
template<int N>
struct BackwardStorage {
    union {
        AttentionGemmStorage score;
        AttentionGemmTile<BackwardColumns,true> output;
    } gemm;
#if EDGE_GRAD
    float probability[N][1024],product[N][1024];
#else
    // P is consumed before the terminal dQ/dK products. Only their selected
    // derivative components remain live, so both contractions share storage.
    union {
        float probability[N][1024];
        float product[N==4 ? 2:1][32*BackwardColumns];
    };
#endif
    float adjoint[N][1024];
    float maximum[32];
    Differential<N> denominator[32],delta[32],node_dot[32],alpha[32];
};

#endif

#if defined(CONV_NATIVE_HIGHER_SUMMARY)
template<int N>
__global__ void node_projection_kernel(const __grid_constant__ Params p,View summary) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    for(int64_t key=begin+warp;key<end;key+=4) {
        Differential<N> dot;
        if constexpr(NEEDS&3483u) {
            for(int c=lane;c<VALUE_DIM;c+=32)
                dot=dot+input<N>(p,2,key,head,c)*upstream<N>(p,5,atom,head,c);
            dot=warp_sum(dot);
        }
        if(lane==0) {
            #pragma unroll
            for(int c=0;c<N;++c) {
                summary.data[(key*HEADS+head)*2*N+c]=dot.x[c];
                summary.data[(key*HEADS+head)*2*N+N+c]=0.f;
            }
        }
    }
}

// Node dV is rank one in channels. Its scalar coefficient also supplies the
// node-adjoint gradient, so both outputs consume each reduced alpha once.
template<int N>
__global__ void node_value_kernel(const __grid_constant__ Params p,View summary) {
    using F=Differential<N>;
    constexpr int Channels=(VALUE_DIM+127)/128;
    const int tid=threadIdx.x,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    __shared__ float coefficient[N][32];
    F node[Channels],seed[Channels];
    if constexpr(NEEDS&516u) {
        #pragma unroll
        for(int group=0;group<Channels;++group) if(tid+group*128<VALUE_DIM)
            seed[group]=upstream<N>(p,5,atom,head,tid+group*128);
    }
    for(int64_t batch=begin;batch<end;batch+=32) {
        if(tid<32) {
            #pragma unroll
            for(int c=0;c<N;++c) coefficient[c][tid]=batch+tid<end ?
                summary.data[((batch+tid)*HEADS+head)*2*N+N+c]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int group=0;group<Channels;++group) {
            const int channel=tid+group*128;
            if(channel<VALUE_DIM) for(int key=0;key<32 && batch+key<end;++key) {
                F alpha;
                #pragma unroll
                for(int c=0;c<N;++c) alpha.x[c]=coefficient[c][key];
                if constexpr(NEEDS&516u) {
                    F value=alpha*seed[group];
                    if constexpr(NEEDS&4u)
                        p.out[2].store(p.out[2].index(batch+key,head,channel),value.x[N-1]);
                    if constexpr(N==4 && (NEEDS&512u))
                        p.out[9].store(p.out[9].index(batch+key,head,channel),value.x[2]);
                }
                if constexpr(NEEDS&32u)
                    node[group]=node[group]+alpha*input<N>(p,2,batch+key,head,channel);
            }
        }
        __syncthreads();
    }
    if constexpr(NEEDS&32u) {
        #pragma unroll
        for(int group=0;group<Channels;++group) if(tid+group*128<VALUE_DIM)
            p.out[5].store(p.out[5].index(atom,head,tid+group*128),node[group].x[N-1]);
    }
}
#endif

template<int N>
__device__ __forceinline__ void attention_adjoint(Params const& p,BackwardStorage<N>& shared,
    int64_t atom,int64_t qb,int64_t kb,int64_t end,int head
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
    ,View const& summary
#endif
    ) {
    using F=Differential<N>;
    if constexpr(EDGE_GRAD) {
        differential_gemm<N,true,false>(shared.gemm.score,shared.adjoint,VALUE_DIM,
            [&](int row,int c) {return qb+row<end ? output_gradient<N>(p,atom,qb+row,head,c):F();},
            [&](int c,int column) {return kb+column<end ? input<N>(p,2,kb+column,head,c):F();});
    } else {
        int lane=threadIdx.x%32,warp=threadIdx.x/32;
#if defined(CONV_FIRST_BACKWARD)
        if(threadIdx.x<32) {
            auto workspace=static_cast<float const*>(p.out[5].data);
            shared.node_dot[threadIdx.x]=Differential<N>(kb+threadIdx.x<end ?
                workspace[2*((kb+threadIdx.x)*HEADS+head)+1]:0.f);
        }
#elif defined(CONV_NATIVE_HIGHER_SUMMARY)
        if(threadIdx.x<32) {
            #pragma unroll
            for(int c=0;c<N;++c) shared.node_dot[threadIdx.x].x[c]=kb+threadIdx.x<end ?
                summary.data[((kb+threadIdx.x)*HEADS+head)*2*N+c]:0.f;
        }
#else
        for(int key=warp;key<32;key+=4) {
            F dot;
            if(kb+key<end) for(int c=lane;c<VALUE_DIM;c+=32)
                dot=dot+input<N>(p,2,kb+key,head,c)*upstream<N>(p,5,atom,head,c);
            dot=warp_sum(dot);
            if(lane==0) shared.node_dot[key]=dot;
        }
#endif
        __syncthreads();
#if !PACKED_GRADIENTS && !(defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD)
        for(int i=threadIdx.x;i<1024;i+=128) {
            F env=qb+i/32<end ? ((PRESENT&16u) ? input<N>(p,4,qb+i/32):F(1.f)):F();
            matrix_store<N>(shared.adjoint,i,env*shared.node_dot[i%32]);
        }
        __syncthreads();
#endif
    }
}

// Normalization and the softmax row mean are streamed; no score matrix or
// edge output survives the call. Higher orders differentiate these same sums.
template<int N>
#if defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD
__global__ __launch_bounds__(128,8)
#else
__global__ __launch_bounds__(128)
#endif
void backward_kernel(const __grid_constant__ Params p
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
    ,View summary
#endif
    ) {
    extern __shared__ __align__(16) char memory[];
    auto& shared=*reinterpret_cast<BackwardStorage<N>*>(memory);
#if PACKED_GRADIENTS
    // The mainloop has finished reading operands before the two terminal
    // planes reuse their storage. P has already been consumed as dS.
    static_assert(sizeof(PackedOutputStorage)>=2*1024*sizeof(float));
    auto product=reinterpret_cast<float(*)[1024]>(&shared.gemm.output);
    auto adjoint=shared.probability;
#else
    auto product=shared.product;
    auto adjoint=shared.adjoint;
#endif
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    using F=Differential<N>;
    for(int64_t qb=begin+int64_t(blockIdx.z)*32;qb<end;qb+=32*QUERY_WORKERS) {
        if(tid<32) {
            shared.maximum[tid]=-INFINITY;
            shared.denominator[tid]=F();
            shared.delta[tid]=F();
        }
        __syncthreads();
        int64_t last_key=begin;
        for(int64_t kb=begin;kb<end;kb+=32) {
            attention_scores<N>(p,shared.gemm.score,shared.probability,atom,begin,end,qb,kb,head);
            attention_adjoint<N>(p,shared,atom,qb,kb,end,head
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
                ,summary
#endif
                );
            for(int row=warp;row<32;row+=4) {
                F score=matrix_value<N>(shared.probability,row*32+lane);
                float next=score.x[0];
                #pragma unroll
                for(int shift=16;shift;shift/=2) next=fmaxf(next,__shfl_xor_sync(0xffffffff,next,shift));
                next=fmaxf(next,shared.maximum[row]);
                float correction=isfinite(next) ? __expf(shared.maximum[row]-next):1.f;
                F weight=isfinite(score.x[0]) ? exponential(score-F(next)):F();
                F sum=warp_sum(weight);
#if PACKED_GRADIENTS || (defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD)
                F env=qb+row<end ? ((PRESENT&16u) ? input<N>(p,4,qb+row):F(1.f)):F();
                F dot=env*shared.node_dot[lane];
#else
                F dot=matrix_value<N>(shared.adjoint,row*32+lane);
#endif
                F mean=warp_sum(weight*dot);
                if(lane==0) {
                    shared.maximum[row]=next;
                    shared.denominator[row]=shared.denominator[row]*F(correction)+sum;
                    shared.delta[row]=shared.delta[row]*F(correction)+mean;
                }
            }
            __syncthreads();
            last_key=kb;
        }
        if(tid<32 && shared.denominator[tid].x[0]>0.f) {
            shared.denominator[tid]=reciprocal(shared.denominator[tid]);
            shared.delta[tid]=shared.delta[tid]*shared.denominator[tid];
        }
        __syncthreads();
        // The last score/adjoint tile is still live. Short segments use it
        // directly; longer rows only recompute earlier tiles.
        for(int64_t kb=last_key;kb>=begin;kb-=32) {
            if(kb!=last_key) {
                attention_scores<N>(p,shared.gemm.score,shared.probability,atom,begin,end,qb,kb,head);
                attention_adjoint<N>(p,shared,atom,qb,kb,end,head
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
                    ,summary
#endif
                    );
            }
            for(int i=tid;i<1024;i+=128) {
                F score=matrix_value<N>(shared.probability,i);
                F probability=isfinite(score.x[0]) ? exponential(score-F(shared.maximum[i/32]))*shared.denominator[i/32]:F();
                matrix_store<N>(shared.probability,i,probability);
#if !PACKED_GRADIENTS
#if defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD
                F env=qb+i/32<end ? ((PRESENT&16u) ? input<N>(p,4,qb+i/32):F(1.f)):F();
                matrix_store<N>(shared.adjoint,i,
                    softmax_gradient(probability,env*shared.node_dot[i%32],shared.delta[i/32]));
#else
                matrix_store<N>(shared.adjoint,i,
                    softmax_gradient(probability,matrix_value<N>(shared.adjoint,i),shared.delta[i/32]));
#endif
#endif
            }
            __syncthreads();
            if constexpr(!EDGE_GRAD) {
#if defined(CONV_FIRST_BACKWARD)
                if constexpr(NEEDS&4u)
#elif defined(CONV_NATIVE_HIGHER_SUMMARY)
                if constexpr(NEEDS&548u)
#endif
                if(tid<32) {
                    F alpha;
                    for(int row=0;row<32 && qb+row<end;++row) {
                        F env=(PRESENT&16u) ? input<N>(p,4,qb+row):F(1.f);
                        alpha=alpha+matrix_value<N>(shared.probability,row*32+tid)*env;
                    }
                    shared.alpha[tid]=alpha;
                }
                for(int row=warp;row<32;row+=4) {
                    F value=warp_sum(matrix_value<N>(shared.probability,row*32+lane)*shared.node_dot[lane]);
                    if(lane==0 && qb+row<end) gradient<N>(p,4,value,qb+row);
                }
                __syncthreads();
#if defined(CONV_FIRST_BACKWARD)
                if constexpr(NEEDS&4u) if(tid<32 && kb+tid<end)
                    p.out[5].atomic_add(p.out[5].index(kb+tid,head,0),shared.alpha[tid].x[0]);
#elif defined(CONV_NATIVE_HIGHER_SUMMARY)
                if constexpr(NEEDS&548u) if(tid<32 && kb+tid<end) {
                    #pragma unroll
                    for(int c=0;c<N;++c)
                        atomicAdd(summary.data+((kb+tid)*HEADS+head)*2*N+N+c,shared.alpha[tid].x[c]);
                }
#else
                for(int c=tid;c<VALUE_DIM;c+=128) {
                    F value;
                    for(int key=0;key<32 && kb+key<end;++key) {
                        F alpha=shared.alpha[key];
                        gradient<N>(p,2,alpha*upstream<N>(p,5,atom,head,c),kb+key,head,c);
                        value=value+alpha*input<N>(p,2,kb+key,head,c);
                    }
                    upstream_gradient<N>(p,5,value,atom,head,c);
                }
#endif
                __syncthreads();
#if PACKED_GRADIENTS
                // Node output adjoints are an outer product. Consume the
                // probability contractions before replacing P with dS.
                for(int i=tid;i<1024;i+=128) {
                    F probability=matrix_value<N>(shared.probability,i);
                    F env=qb+i/32<end ? ((PRESENT&16u) ? input<N>(p,4,qb+i/32):F(1.f)):F();
                    matrix_store<N>(shared.probability,i,
                        softmax_gradient(probability,env*shared.node_dot[i%32],shared.delta[i/32]));
                }
                __syncthreads();
#endif
            }
            if constexpr(BIAS_KIND==1) {
                for(int column=tid;column<32;column+=128) if(kb+column<end) {
                    F value;
                    for(int row=0;row<32;++row) value=value+matrix_value<N>(adjoint,row*32+column);
                    gradient<N>(p,3,value,kb+column,head);
                }
            }
            if constexpr(BIAS_KIND==2) {
                for(int i=tid;i<1024;i+=128) if(qb+i/32<end && kb+i%32<end)
                    gradient<N>(p,3,matrix_value<N>(adjoint,i),atom,head,qb+i/32-begin,kb+i%32-begin);
            }
            __syncthreads();
            for(int channel=0;channel<DIM;channel+=BackwardColumns) {
                if constexpr(NEEDS&((1u<<0)|(1u<<7))) {
#if LOW_PRECISION
                    if constexpr(N==1)
                        attention_gemm(shared.gemm.output,product[0],32,
                            [&](int row,int k) {return adjoint[0][row*32+k];},
                            ConvGemmFeature<1,false>{p,kb,end,head,channel});
                    else
#endif
                    differential_gemm<N,true,false,GradientComponents>(shared.gemm.output,product,32,
                        [&](int row,int k) {return matrix_value<N>(adjoint,row*32+k);},
                        [&](int k,int column) {return kb+k<end && channel+column<DIM ? input<N>(p,1,kb+k,head,channel+column):F();});
                    for(int i=tid;i<32*BackwardColumns;i+=128) if(qb+i/BackwardColumns<end && channel+i%BackwardColumns<DIM)
                        gradient<N>(p,0,matrix_value<N,GradientComponents>(product,i)*F(p.scale),qb+i/BackwardColumns,head,channel+i%BackwardColumns);
#if PACKED_GRADIENTS
                    __syncthreads();
#endif
                }
                if constexpr(NEEDS&((1u<<1)|(1u<<8))) {
#if LOW_PRECISION
                    if constexpr(N==1)
                        attention_gemm(shared.gemm.output,product[0],32,
                            [&](int row,int k) {return adjoint[0][k*32+row];},
                            ConvGemmFeature<0,false>{p,qb,end,head,channel});
                    else
#endif
                    differential_gemm<N,true,false,GradientComponents>(shared.gemm.output,product,32,
                        [&](int row,int k) {return matrix_value<N>(adjoint,k*32+row);},
                        [&](int k,int column) {return qb+k<end && channel+column<DIM ? input<N>(p,0,qb+k,head,channel+column):F();});
#if defined(CONV_FIRST_BACKWARD) && !LOW_PRECISION && !EDGE_GRAD && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
                    if constexpr(DIM%4==0) {
                        for(int i=tid*4;i<32*BackwardColumns;i+=128*4) {
                            int row=i/BackwardColumns,c=channel+i%BackwardColumns;
                            if(kb+row<end && c<DIM) {
                                float4 value=*reinterpret_cast<float4 const*>(product[0]+i);
                                value.x*=p.scale;value.y*=p.scale;value.z*=p.scale;value.w*=p.scale;
                                auto pointer=reinterpret_cast<float4*>(static_cast<float*>(p.out[1].data)
                                    +((kb+row)*HEADS+head)*DIM+c);
                                if constexpr(QUERY_WORKERS>1) atomicAdd(pointer,value);
                                else {
                                    float4 previous=*pointer;
                                    previous.x+=value.x;previous.y+=value.y;previous.z+=value.z;previous.w+=value.w;
                                    *pointer=previous;
                                }
                            }
                        }
                    } else
#endif
                    for(int i=tid;i<32*BackwardColumns;i+=128) if(kb+i/BackwardColumns<end && channel+i%BackwardColumns<DIM)
                        gradient<N>(p,1,matrix_value<N,GradientComponents>(product,i)*F(p.scale),kb+i/BackwardColumns,head,channel+i%BackwardColumns);
                }
                __syncthreads();
            }
            if constexpr(EDGE_GRAD) for(int channel=0;channel<VALUE_DIM;channel+=32) {
                if constexpr(NEEDS&((1u<<2)|(1u<<9))) {
                    differential_gemm<N,true,true,GradientComponents>(shared.gemm.output,product,32,
                        [&](int row,int k) {return matrix_value<N>(shared.probability,k*32+row);},
                        [&](int k,int column) {return qb+k<end && channel+column<VALUE_DIM ? output_gradient<N>(p,atom,qb+k,head,channel+column):F();});
                    for(int i=tid;i<1024;i+=128) if(kb+i/32<end && channel+i%32<VALUE_DIM)
                        gradient<N>(p,2,matrix_value<N,GradientComponents>(product,i),kb+i/32,head,channel+i%32);
                }
                if constexpr(NEEDS&((1u<<4)|(1u<<5)|(1u<<6)|(1u<<11))) {
                    differential_gemm<N,true,false>(shared.gemm.output,product,32,
                        [&](int row,int k) {return matrix_value<N>(shared.probability,row*32+k);},
                        [&](int k,int column) {return kb+k<end && channel+column<VALUE_DIM ? input<N>(p,2,kb+k,head,channel+column):F();});
                    for(int row=warp;row<32;row+=4) {
                        F env_grad;
                        for(int column=lane;column<32;column+=32) {
                            F value=matrix_value<N>(product,row*32+column);
                            F node_grad=channel+column<VALUE_DIM ? upstream<N>(p,5,atom,head,channel+column):F();
                            env_grad=env_grad+value*node_grad;
                            if(qb+row<end && channel+column<VALUE_DIM)
                                upstream_gradient<N>(p,6,value,qb+row,head,channel+column);
                        }
                        env_grad=warp_sum(env_grad);
                        if(lane==0 && qb+row<end) gradient<N>(p,4,env_grad,qb+row);
                    }
                    if(tid<32 && channel+tid<VALUE_DIM) {
                        F node;
                        for(int row=0;row<32 && qb+row<end;++row) {
                            F env=(PRESENT&16u) ? input<N>(p,4,qb+row):F(1.f);
                            node=node+matrix_value<N>(product,row*32+tid)*env;
                        }
                        upstream_gradient<N>(p,5,node,atom,head,channel+tid);
                    }
                }
                __syncthreads();
            }
        }
    }
}

#if defined(CONV_FIRST_BACKWARD) && LOW_PRECISION
namespace conv_register {
constexpr int Reduction=((DIM+31)/32*32<128 ? (DIM+31)/32*32:128);

struct ScoreTraits {
    using Shape=cutlass::gemm::GemmShape<32,64,Reduction>;
    using SmemA=cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16,32>;
    using SmemB=cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16,32>;
    using Operator=typename cutlass::gemm::warp::DefaultMmaTensorOp<
        cutlass::gemm::GemmShape<16,32,Reduction>,GemmInstruction,
        GemmElement,SmemA,GemmElement,SmemB,float,cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>::Type;
    using Policy=cutlass::gemm::threadblock::MmaPolicy<Operator,
        cutlass::MatrixShape<0,0>,cutlass::MatrixShape<0,0>,1>;
    using Base=cutlass::gemm::threadblock::MmaBase<Shape,Policy,1>;
    using Storage=typename Base::SharedStorage;
    using Mainloop=cutlass::gemm::threadblock::MmaSingleStage<Shape,
        ConvReadyOperand,ConvReadyOperand,ConvReadyOperand,ConvReadyOperand,
        float,cutlass::layout::RowMajor,Policy>;
};
using Operator=typename ScoreTraits::Operator;
using Accumulator=typename Operator::FragmentC;
using ScoreStorage=typename ScoreTraits::Storage;

template<bool Query>
__device__ __forceinline__ void load_score(ScoreStorage& shared,Params const& p,
    int64_t begin,int64_t end,int head,int channel) {
    constexpr int Slot=Query ? 0:1,Rows=Query ? 32:64;
    constexpr int InputType=(INPUT_DTYPES>>(2*Slot))&3u;
#if DIM%8==0
    if constexpr(InputType!=0 && (CONTIGUOUS&(1u<<Slot))) {
        auto base=static_cast<GemmElement const*>(p.in[Slot].data);
        if((reinterpret_cast<uintptr_t>(base)&15u)==0) {
            for(int pos=threadIdx.x*8;pos<Rows*Reduction;pos+=128*8) {
                const int row=pos/Reduction,column=pos%Reduction;
                const bool valid=begin+row<end && channel+column<DIM;
                auto pointer=base+(valid ? ((begin+row)*HEADS+head)*DIM+channel+column:0);
                GemmElement* destination;
                if constexpr(Query) destination=&shared.operand_A_ref().at({row,column});
                else destination=&shared.operand_B_ref().at({column,row});
                cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(destination,pointer,valid);
            }
            return;
        }
    }
#endif
    ConvGemmFeature<Slot,false> source{p,begin,end,head,channel};
    for(int pos=threadIdx.x;pos<Rows*Reduction;pos+=128) {
        const int row=pos/Reduction,column=pos%Reduction;
        if constexpr(Query) shared.operand_A_ref().at({row,column})=source(row,column);
        else shared.operand_B_ref().at({column,row})=source(row,column);
    }
}
static_assert(Accumulator::kElements==16);
struct RowStats {
    float maximum[32],denominator[32],delta[32];
    float partial[3][2][32];
};

// The unchanged library Core has two M warps and two N warps.
// IteratorC maps i=4*n+2*r+j to (16*warp_m+lane/4+8*r,
// 32*warp_n+8*n+2*(lane%4)+j). Only row statistics cross the N warps.
__device__ __forceinline__ void normalize(Accumulator& accumulator,RowStats& stats,
    float const* node_dot,float const* env) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,part=warp/2;
    #pragma unroll
    for(int r=0;r<2;++r) {
        float value=-INFINITY;
        #pragma unroll
        for(int n=0;n<4;++n) {
            value=fmaxf(value,accumulator[4*n+2*r]);
            value=fmaxf(value,accumulator[4*n+2*r+1]);
        }
        value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,1));
        value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,2));
        const int row=(warp%2)*16+lane/4+r*8;
        if(lane%4==0) stats.partial[0][part][row]=value;
    }
    __syncthreads();
    float next[2],correction[2];
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        next[r]=fmaxf(stats.maximum[row],fmaxf(stats.partial[0][0][row],stats.partial[0][1][row]));
        correction[r]=isfinite(next[r]) ? __expf(stats.maximum[row]-next[r]):1.f;
        float sum=0.f,mean=0.f;
        #pragma unroll
        for(int n=0;n<4;++n) {
            #pragma unroll
            for(int j=0;j<2;++j) {
                const int i=4*n+2*r+j,column=part*32+n*8+(lane%4)*2+j;
                float value=isfinite(accumulator[i]) ? __expf(accumulator[i]-next[r]):0.f;
                accumulator[i]=value;
                sum+=value;
                mean=fmaf(value,node_dot[column],mean);
            }
        }
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        mean+=__shfl_xor_sync(0xffffffff,mean,1);
        mean+=__shfl_xor_sync(0xffffffff,mean,2);
        if(lane%4==0) {
            stats.partial[1][part][row]=sum;
            stats.partial[2][part][row]=mean*env[row];
        }
    }
    __syncthreads();
    if(part==0 && lane%4==0) {
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int row=(warp%2)*16+lane/4+r*8;
            stats.maximum[row]=next[r];
            stats.denominator[row]=stats.denominator[row]*correction[r]+stats.partial[1][0][row]+stats.partial[1][1][row];
            stats.delta[row]=stats.delta[row]*correction[r]+stats.partial[2][0][row]+stats.partial[2][1][row];
        }
    }
    __syncthreads();
}

// P remains in the library accumulator. The terminal contractions consume
// Half/BF16 dS directly, and alpha retains the node output's rank-one structure.
template<bool EnvGradient,bool BiasGradient>
__device__ __forceinline__ void probabilities(Accumulator& accumulator,RowStats& stats,
    float const* node_dot,float const* env,GemmElement* adjoint,float* alpha,float* bias,
    bool last_weights) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,part=warp/2;
    float alphas[4][2]={},biases[4][2]={},envgrad[2]={};
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        #pragma unroll
        for(int n=0;n<4;++n) {
            #pragma unroll
            for(int j=0;j<2;++j) {
                const int i=4*n+2*r+j,column=part*32+n*8+(lane%4)*2+j;
                float value=last_weights ? accumulator[i] :
                    (isfinite(accumulator[i]) ? __expf(accumulator[i]-stats.maximum[row]):0.f);
                float probability=value*stats.denominator[row];
                float ds=probability*(env[row]*node_dot[column]-stats.delta[row]);
                adjoint[row*64+column]=GemmElement(ds);
                accumulator[i]=ds;
                alphas[n][j]=fmaf(probability,env[row],alphas[n][j]);
                if constexpr(BiasGradient) biases[n][j]+=ds;
                if constexpr(EnvGradient) envgrad[r]=fmaf(probability,node_dot[column],envgrad[r]);
            }
        }
        if constexpr(EnvGradient) {
            envgrad[r]+=__shfl_xor_sync(0xffffffff,envgrad[r],1);
            envgrad[r]+=__shfl_xor_sync(0xffffffff,envgrad[r],2);
            if(lane%4==0) stats.partial[0][part][row]=envgrad[r];
        }
    }
    #pragma unroll
    for(int n=0;n<4;++n) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            float value=alphas[n][j];
            value+=__shfl_xor_sync(0xffffffff,value,4);
            value+=__shfl_xor_sync(0xffffffff,value,8);
            value+=__shfl_xor_sync(0xffffffff,value,16);
            const int column=part*32+n*8+(lane%4)*2+j;
            if(lane<4) alpha[(warp%2)*64+column]=value;
            if constexpr(BiasGradient) {
                float derivative=biases[n][j];
                derivative+=__shfl_xor_sync(0xffffffff,derivative,4);
                derivative+=__shfl_xor_sync(0xffffffff,derivative,8);
                derivative+=__shfl_xor_sync(0xffffffff,derivative,16);
                if(lane<4) bias[(warp%2)*64+column]=derivative;
            }
        }
    }
    __syncthreads();
}
} // namespace conv_register
struct RegisterBackwardStorage {
    // Q survives both score passes while K and terminal products share bytes.
    decltype(conv_register::ScoreStorage::operand_A) query;
    union {
        decltype(conv_register::ScoreStorage::operand_B) key;
        AttentionGemmTile<32,true> output;
    } gemm;
    __align__(16) GemmElement adjoint[2048];
    conv_register::RowStats stats;
    float node_dot[64],env[32],alpha[128];
    float bias[(BIAS_KIND==1 && (NEEDS&8u)) ? 128:1];
};

static_assert(sizeof(RegisterBackwardStorage)<=32*1024,
    "Native score and terminal storage must remain bounded");
static_assert(offsetof(RegisterBackwardStorage,gemm)==offsetof(conv_register::ScoreStorage,operand_B),
    "Persistent Q and reusable K must address the library score storage");

// The library mainloop returns the live accumulator without a C store. The
// FragmentC coordinate formula is the actual CUTLASS IteratorC mapping.
__device__ __forceinline__ void register_scores(Params const& p,RegisterBackwardStorage& shared,
    conv_register::Accumulator& accumulator,int64_t atom,int64_t begin,int64_t end,int64_t qb,int64_t kb,int head) {
    accumulator.clear();
    auto& operands=*reinterpret_cast<conv_register::ScoreStorage*>(&shared);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    for(int channel=0;channel<DIM;channel+=conv_register::Reduction) {
        if constexpr(DIM>conv_register::Reduction)
            conv_register::load_score<true>(operands,p,qb,end,head,channel);
        conv_register::load_score<false>(operands,p,kb,end,head,channel);
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        typename conv_register::ScoreTraits::Mainloop mma(operands,threadIdx.x,warp,lane);
        mma(1,accumulator,ConvReadyOperand{},ConvReadyOperand{},accumulator);
    }
    #pragma unroll
    for(int n=0;n<4;++n)
        for(int r=0;r<2;++r)
            for(int j=0;j<2;++j) {
                int row=(warp%2)*16+lane/4+r*8,column=(warp/2)*32+n*8+(lane%4)*2+j;
                int64_t q=qb+row,k=kb+column;
                float value=-INFINITY;
                if(q<end && k<end && (!CAUSAL || k<=q)) {
                    value=accumulator[4*n+2*r+j]*p.scale;
                    if constexpr(BIAS_KIND==1) value+=input<1>(p,3,k,head).x[0];
                    if constexpr(BIAS_KIND==2) value+=input<1>(p,3,atom,head,q-begin,k-begin).x[0];
                }
                accumulator[4*n+2*r+j]=value;
            }
    if(threadIdx.x<64) {
        const int64_t key=kb+threadIdx.x;
        auto workspace=static_cast<float const*>(p.out[5].data);
        shared.node_dot[threadIdx.x]=key<end ? workspace[2*(key*HEADS+head)+1]:0.f;
    }
    __syncthreads();
}

template<bool Transpose>
struct RegisterAdjoint {
    GemmElement const* data;
    __device__ __forceinline__ GemmElement operator()(int row,int column) const {
        if constexpr(Transpose) return data[column*64+row];
        else return data[row*64+column];
    }
    template<bool First,typename Fragment>
    __device__ __forceinline__ void load8(Fragment& fragment,int index,int row,int column) const {
        if constexpr(!Transpose && !First) {
            uint4 packed=*reinterpret_cast<uint4 const*>(data+row*64+column);
            auto values=reinterpret_cast<GemmElement const*>(&packed);
            #pragma unroll
            for(int i=0;i<8;++i) fragment[index+i]=values[i];
        } else {
            #pragma unroll
            for(int i=0;i<8;++i) fragment[index+i]=operator()(row+(First ? i:0),column+(First ? 0:i));
        }
    }
};

template<bool Key,typename Left,typename Right>
__device__ __forceinline__ void register_gradient(Params const& p,AttentionGemmTile<32,true>& shared,
    int reduction,Left left,Right right,int64_t begin,int64_t end,int head,int channel,bool first=false,bool last=false) {
    constexpr int Rows=Key ? 64:32,Partitions=Key ? 1:2,Stride=40;
    using Shape=cutlass::gemm::GemmShape<Rows,32,32>;
    using Core=cutlass::gemm::threadblock::DefaultMmaCore<Shape,
        cutlass::gemm::GemmShape<16,32,32/Partitions>,
        cutlass::gemm::GemmShape<16,8,Key ? 16:8>,
        GemmElement,cutlass::layout::RowMajor,GemmElement,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,GemmClass,2,cutlass::arch::OpMultiplyAdd>;
    using IteratorA=AttentionProducerIterator<typename Core::IteratorThreadMapA,Left,true,true,32>;
    using IteratorB=AttentionProducerIterator<typename Core::IteratorThreadMapB,Right,false,true,32>;
    using Mma=cutlass::gemm::threadblock::MmaSingleStage<Shape,
        IteratorA,typename Core::SmemIteratorA,IteratorB,typename Core::SmemIteratorB,
        float,cutlass::layout::RowMajor,typename Core::MmaPolicy>;
    using Storage=typename Mma::SharedStorage;
    static_assert(sizeof(shared)>=sizeof(Storage) && alignof(decltype(shared))>=alignof(Storage));
    static_assert(sizeof(shared)>=Partitions*Rows*Stride*sizeof(float));
    typename Mma::FragmentC accumulator;
    accumulator.clear();
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    Mma mma(*reinterpret_cast<Storage*>(&shared),tid,warp,lane);
    mma(Key ? 1:(reduction+31)/32,accumulator,
        IteratorA(left,reduction),IteratorB(right,reduction),accumulator);
    if constexpr(Key) {
        // Each warp owns 16 key rows. Consume its final C directly, without a
        // shared-memory store/reload or a K-partition reduction.
        #pragma unroll
        for(int n=0;n<4;++n)
            for(int row=0;row<2;++row)
                for(int j=0;j<2;++j) {
                    const int key=warp*16+lane/4+row*8;
                    const int column=channel+n*8+(lane%4)*2+j;
                    if(begin+key<end && column<DIM)
                        gradient<1>(p,1,Differential<1>(accumulator[4*n+2*row+j]*p.scale),begin+key,head,column);
                }
    } else {
        __syncthreads();
        float* result=reinterpret_cast<float*>(&shared);
        using Operator=typename Core::MmaPolicy::Operator;
        typename Operator::IteratorC iterator(
            {result+(warp/2)*32*Stride+(warp%2)*16*Stride,cutlass::layout::RowMajor(Stride)},lane);
        iterator.store(accumulator);
        __syncthreads();
        for(int i=tid;i<32*32;i+=128) if(begin+i/32<end && channel+i%32<DIM) {
            const int index=(i/32)*Stride+i%32;
            float value=(result[index]+result[32*Stride+index])*p.scale;
            auto const& partial=p.out[6];
            auto destination=static_cast<float*>(partial.data)+partial.index(begin+i/32,head,channel+i%32);
            if(!first) value=*destination+value;
            // One CTA owns each query row. Keep partial sums in FP32 and
            // round only its final contribution into the public gradient.
            if(last) {
                auto const& output=p.out[0];
                static_cast<GemmElement*>(output.data)[output.index(begin+i/32,head,channel+i%32)]=GemmElement(value);
            } else *destination=value;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(128) void backward_register_kernel(const __grid_constant__ Params p) {
    extern __shared__ __align__(16) char memory[];
    auto& shared=*reinterpret_cast<RegisterBackwardStorage*>(memory);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    for(int64_t qb=begin+int64_t(blockIdx.z)*32;qb<end;qb+=32*QUERY_WORKERS) {
        if constexpr(DIM<=conv_register::Reduction) {
            auto& operands=*reinterpret_cast<conv_register::ScoreStorage*>(&shared);
            conv_register::load_score<true>(operands,p,qb,end,head,0);
            cutlass::arch::cp_async_fence();
        }
        if(tid<32) {
            shared.stats.maximum[tid]=-INFINITY;
            shared.stats.denominator[tid]=0.f;
            shared.stats.delta[tid]=0.f;
            shared.env[tid]=qb+tid<end ? ((PRESENT&16u) ? input<1>(p,4,qb+tid).x[0]:1.f):0.f;
        }
        __syncthreads();
        int64_t last_key=begin;
        conv_register::Accumulator accumulator;
        for(int64_t kb=begin;kb<end;kb+=64) {
            register_scores(p,shared,accumulator,atom,begin,end,qb,kb,head);
            conv_register::normalize(accumulator,shared.stats,shared.node_dot,shared.env);
            last_key=kb;
        }
        if(tid<32) {
            float inverse=shared.stats.denominator[tid]>0.f ? 1.f/shared.stats.denominator[tid]:0.f;
            shared.stats.denominator[tid]=inverse;
            shared.stats.delta[tid]*=inverse;
        }
        __syncthreads();
        for(int64_t kb=last_key;kb>=begin;kb-=64) {
            if(kb!=last_key) register_scores(p,shared,accumulator,atom,begin,end,qb,kb,head);
            conv_register::probabilities<(NEEDS&16u)!=0,BIAS_KIND==1 && (NEEDS&8u)>(accumulator,shared.stats,
                shared.node_dot,shared.env,shared.adjoint,shared.alpha,shared.bias,kb==last_key);
            if constexpr(BIAS_KIND==2 && (NEEDS&8u)) {
                #pragma unroll
                for(int n=0;n<4;++n)
                    for(int r=0;r<2;++r)
                        for(int j=0;j<2;++j) {
                            int row=(warp%2)*16+lane/4+r*8,column=(warp/2)*32+n*8+(lane%4)*2+j;
                            if(qb+row<end && kb+column<end)
                                gradient<1>(p,3,Differential<1>(accumulator[4*n+2*r+j]),atom,head,qb+row-begin,kb+column-begin);
                        }
            }
            accumulator.clear();
            if constexpr(NEEDS&16u) if(tid<32 && qb+tid<end)
                gradient<1>(p,4,Differential<1>(shared.stats.partial[0][0][tid]+shared.stats.partial[0][1][tid]),qb+tid);
            if constexpr(BIAS_KIND==1 && (NEEDS&8u)) if(tid<64 && kb+tid<end)
                gradient<1>(p,3,Differential<1>(shared.bias[tid]+shared.bias[64+tid]),kb+tid,head);
            if(tid<64) shared.alpha[tid]+=shared.alpha[64+tid];
            __syncthreads();
            if constexpr(NEEDS&4u) if(tid<64 && kb+tid<end)
                p.out[5].atomic_add(p.out[5].index(kb+tid,head),shared.alpha[tid]);
            for(int channel=0;channel<DIM;channel+=32) {
                if constexpr(NEEDS&1u) {
                    register_gradient<false>(p,shared.gemm.output,64,RegisterAdjoint<false>{shared.adjoint},
                        ConvGemmFeature<1,false>{p,kb,end,head,channel},qb,end,head,channel,kb==last_key,kb==begin);
                }
                if constexpr(NEEDS&2u) {
                    register_gradient<true>(p,shared.gemm.output,32,RegisterAdjoint<true>{shared.adjoint},
                        ConvGemmFeature<0,false>{p,qb,end,head,channel},kb,end,head,channel);
                }
            }
        }
    }
}

#endif

#if defined(CONV_FIRST_BACKWARD) && !EDGE_GRAD
// Node dV is rank one in its channel dimension. Sum its scalar coefficient
// before broadcasting the node gradient, avoiding a full FP32 dV atomic plane.
__global__ void backward_value_kernel(const __grid_constant__ Params p) {
    const int head=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    constexpr int Channels=(VALUE_DIM+31)/32;
    float seed[Channels];
    #pragma unroll
    for(int c=0;c<Channels;++c)
        seed[c]=lane+c*32<VALUE_DIM ? upstream<1>(p,5,atom,head,lane+c*32).x[0]:0.f;
    auto coefficient=static_cast<float const*>(p.out[5].data);
    auto output=static_cast<GemmElement*>(p.out[2].data);
    for(int64_t key=begin+warp;key<end;key+=blockDim.x/32) {
        const float alpha=coefficient[2*(key*HEADS+head)];
        #pragma unroll
        for(int c=0;c<Channels;++c) if(lane+c*32<VALUE_DIM)
            output[(key*HEADS+head)*VALUE_DIM+lane+c*32]=GemmElement(alpha*seed[c]);
    }
}

#if !LOW_PRECISION
template<int Slot>
__device__ __forceinline__ float4 load_projection_channels(Params const& p,int64_t row,int head,int channel) {
    if constexpr((PRESENT&(1u<<Slot)) && (CONTIGUOUS&(1u<<Slot))) {
        auto pointer=static_cast<float const*>(p.in[Slot].data)+input_offset(p,Slot,Slot,row,head,channel);
        if(channel+3<VALUE_DIM && (reinterpret_cast<uintptr_t>(pointer)&15u)==0)
            return *reinterpret_cast<float4 const*>(pointer);
    }
    return {channel<VALUE_DIM ? input<1>(p,Slot,row,head,channel).x[0]:0.f,
        channel+1<VALUE_DIM ? input<1>(p,Slot,row,head,channel+1).x[0]:0.f,
        channel+2<VALUE_DIM ? input<1>(p,Slot,row,head,channel+2).x[0]:0.f,
        channel+3<VALUE_DIM ? input<1>(p,Slot,row,head,channel+3).x[0]:0.f};
}
#endif

// Zero the reduction outputs and reuse one node-gradient projection per edge.
// Alpha and the projection have disjoint fields throughout the backward call.
__global__ void backward_initialize_kernel(const __grid_constant__ Params p) {
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.y;
    const int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if constexpr(NEEDS&2u) {
        auto output=static_cast<float*>(p.out[1].data);
        for(int64_t i=tid;i<(end-begin)*DIM;i+=128)
            output[((begin+i/DIM)*HEADS+head)*DIM+i%DIM]=0.f;
    }
    if constexpr(NEEDS&16u) if(head==0)
        for(int64_t edge=begin+tid;edge<end;edge+=128)
            p.out[4].store(p.out[4].index(edge),0.f);
    if constexpr((NEEDS&8u) && BIAS_KIND==1)
        for(int64_t edge=begin+tid;edge<end;edge+=128)
            p.out[3].store(p.out[3].index(edge,head),0.f);
    if constexpr((NEEDS&8u) && BIAS_KIND==2)
        for(int64_t i=tid;i<p.out[3].stride[1];i+=128)
            p.out[3].store(atom*p.out[3].stride[0]+head*p.out[3].stride[1]+i,0.f);
    auto workspace=static_cast<float*>(p.out[5].data);
    constexpr int Channels=(VALUE_DIM+31)/32;
#if LOW_PRECISION
    float seed[Channels];
    if constexpr(NEEDS&27u) {
        #pragma unroll
        for(int c=0;c<Channels;++c)
            seed[c]=lane+c*32<VALUE_DIM ? upstream<1>(p,5,atom,head,lane+c*32).x[0]:0.f;
    }
    for(int64_t key=begin+warp;key<end;key+=4) {
        float value=0.f;
        if constexpr(NEEDS&27u) {
            #pragma unroll
            for(int c=0;c<Channels;++c) if(lane+c*32<VALUE_DIM)
                value=fmaf(input<1>(p,2,key,head,lane+c*32).x[0],seed[c],value);
            value=attention_sum(value);
        }
        if(lane==0) {
            if constexpr(NEEDS&4u) workspace[2*(key*HEADS+head)]=0.f;
            workspace[2*(key*HEADS+head)+1]=value;
        }
    }
#else
    float4 seed[Channels];
    if constexpr(NEEDS&27u) {
        #pragma unroll
        for(int c=0;c<Channels;++c)
            seed[c]=load_projection_channels<5>(p,atom,head,(tid%8)*4+c*32);
    }
    // All groups, including tail rows, participate in the width-eight shuffles.
    for(int64_t batch=begin;batch<end;batch+=16) {
        const int64_t key=batch+tid/8;
        float value=0.f;
        if constexpr(NEEDS&27u) {
            if(key<end) {
                #pragma unroll
                for(int c=0;c<Channels;++c) {
                    float4 v=load_projection_channels<2>(p,key,head,(tid%8)*4+c*32);
                    value=fmaf(v.x,seed[c].x,value);
                    value=fmaf(v.y,seed[c].y,value);
                    value=fmaf(v.z,seed[c].z,value);
                    value=fmaf(v.w,seed[c].w,value);
                }
            }
            value+=__shfl_xor_sync(0xffffffff,value,4,8);
            value+=__shfl_xor_sync(0xffffffff,value,2,8);
            value+=__shfl_xor_sync(0xffffffff,value,1,8);
        }
        if(tid%8==0 && key<end) {
            if constexpr(NEEDS&4u) workspace[2*(key*HEADS+head)]=0.f;
            workspace[2*(key*HEADS+head)+1]=value;
        }
    }
#endif
}

#endif
"""


SOURCE = r"""
void run(std::vector<std::optional<at::Tensor>> inputs,at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if(csr.numel()>1) {
#if !EDGE_GRAD
        auto p=params(inputs,outputs,csr,scale);
        backward_initialize_kernel<<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p);
#if LOW_PRECISION
        backward_register_kernel<<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,sizeof(RegisterBackwardStorage),at::cuda::getCurrentCUDAStream()>>>(p);
#else
        backward_kernel<1><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,sizeof(BackwardStorage<1>),at::cuda::getCurrentCUDAStream()>>>(p);
#endif
        if constexpr(NEEDS&4u)
            backward_value_kernel<<<dim3(csr.numel()-1,HEADS),256,0,at::cuda::getCurrentCUDAStream()>>>(p);
#else
        constexpr int bytes=sizeof(BackwardStorage<1>);
        backward_kernel<1><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,bytes,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
"""


@lru_cache(maxsize=128)
def get_backward_kernel(config, device, dtype, needed, edge_grad, layout, query_workers):
    defines = (
        "#define CONV_FIRST_BACKWARD 1\n"
        f"#define QUERY_WORKERS {query_workers}\n#define NEEDS {needed}u\n"
        f"#define EDGE_GRAD {int(edge_grad)}\n"
    )
    source = defines + GRADIENT_SOURCE + BACKWARD_SOURCE + SOURCE
    return load_kernel("attention_conv_backward", source, config, device.index, dtype, layout).run


def attention_conv_backward(inputs, csr, config, needs):
    node_backward = inputs[6] is None
    query_owner = inputs[0].dtype != torch.float32 and node_backward
    outputs = tuple(
        (
            torch.empty(x.shape, device=x.device, dtype=x.dtype)
            if query_owner and i in (0, 2) else
            torch.empty(x.shape, device=x.device, dtype=torch.float32)
            if query_owner or (node_backward and i in (1, 2)) else
            torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        )
        if x is not None and need else None
        for i, (x, need) in enumerate(zip(inputs[:5], needs))
    )
    workspace = (
        (
            torch.empty(
                (inputs[0].shape[0], config[2], 2), device=inputs[0].device, dtype=torch.float32
            ),
            torch.empty(inputs[0].shape, device=inputs[0].device, dtype=torch.float32)
            if query_owner and needs[0] else None,
        )
        if node_backward else ()
    )
    needed = sum(1 << i for i, output in enumerate(outputs) if output is not None)
    query_workers = 4 if inputs[0].shape[0] > 32 * (csr.numel() - 1) else 1
    kernel = get_backward_kernel(
        config, inputs[0].device, inputs[0].dtype, needed,
        inputs[6] is not None, input_layout(inputs), query_workers,
    )
    kernel(inputs, csr, (*outputs, *workspace), config[3])
    del workspace
    return finish_gradients(outputs, inputs)
