"""Online CSR attention with CUTLASS SIMT and TensorOp matrix products."""

import torch

from .utils import CUDA_SOURCE, GEMM_SOURCE, ATTENTION_SOURCE, dtype_defines
from ....utils.extension import get_cuda_arch, load_cuda_extension

KERNELS = {}

FORWARD_SOURCE = r"""
constexpr int ForwardRows=32;
constexpr int ForwardKeys=LOW_PRECISION ? 64:32;
constexpr int ForwardThreads=128;


#if !LOW_PRECISION
#include <cute/tensor.hpp>
namespace native_score {
using namespace cute;
using Mma=decltype(make_tiled_mma(UniversalFMA<float,float,float>{},
    Layout<Shape<_4,_8,_1>,Stride<_8,_1,_0>>{},
    Tile<Layout<Shape<_4,_2>,Stride<_2,_1>>,
         Layout<Shape<_8,_4>,Stride<_4,_1>>,_1>{}));
using Copy=decltype(make_tiled_copy(Copy_Atom<UniversalCopy<float>,float>{},
    Layout<Shape<_32,_4>,Stride<_4,_1>>{},Layout<Shape<_1,_1>>{}));
using ScoreLayout=Layout<Shape<_1,_2,_4>,Stride<_0,_4,_1>>;
using OperandLayout=Layout<Shape<_32,_32>,Stride<_1,Int<36>>>;
struct alignas(16) Storage {
    float query[2*cosize_v<OperandLayout>],key[2*cosize_v<OperandLayout>];
};
using Accumulator=cutlass::Array<float,8>;

__device__ __forceinline__ void prefetch(Storage& shared,AttentionParams const& p,
    int slot,int64_t qb,int64_t kb,int64_t end,int head,int channel) {
    auto query=make_tensor(make_smem_ptr(shared.query+slot*cosize_v<OperandLayout>),OperandLayout{});
    auto key=make_tensor(make_smem_ptr(shared.key+slot*cosize_v<OperandLayout>),OperandLayout{});
    auto thread=Copy{}.get_slice(threadIdx.x);
    auto qd=thread.partition_D(query);
    auto kd=thread.partition_D(key);
    auto coords=thread.partition_S(make_identity_tensor(Shape<_32,_32>{}));
    CUTE_UNROLL
    for(int i=0;i<size(qd);++i) {
        auto coord=coords(i);
        const int row=int(get<0>(coord)),column=channel+int(get<1>(coord));
        const bool valid=qb+row<end && column<DIM;
        auto const& q=p.inputs[0];
#if COMPACT
        const int64_t offset=((qb+row)*HEADS+head)*DIM+column;
#else
        const int64_t offset=(qb+row)*q.stride[0]+head*q.stride[1]+column*q.stride[2];
#endif
        cutlass::arch::cp_async_zfill<4,cutlass::arch::CacheOperation::Always>(
            &qd(i),q.data+(valid ? offset:0),valid);
    }
    CUTE_UNROLL
    for(int i=0;i<size(kd);++i) {
        auto coord=coords(i);
        const int row=int(get<0>(coord)),column=channel+int(get<1>(coord));
        const bool valid=kb+row<end && column<DIM;
        auto const& k=p.inputs[1];
#if COMPACT
        const int64_t offset=((kb+row)*HEADS+head)*DIM+column;
#else
        const int64_t offset=(kb+row)*k.stride[0]+head*k.stride[1]+column*k.stride[2];
#endif
        cutlass::arch::cp_async_zfill<4,cutlass::arch::CacheOperation::Always>(
            &kd(i),k.data+(valid ? offset:0),valid);
    }
    cutlass::arch::cp_async_fence();
}

__device__ __forceinline__ Accumulator gemm(Storage& shared,AttentionParams const& p,
    int64_t qb,int64_t kb,int64_t end,int head) {
    Mma mma;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    auto thread=mma.get_slice(lane);
    Accumulator scores;
    scores.clear();
    auto c=make_tensor(make_rmem_ptr(scores.data()),ScoreLayout{});
    const bool active=qb+warp*8<end;
    prefetch(shared,p,0,qb,kb,end,head,0);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    for(int channel=0;channel<DIM;channel+=32) {
        const int slot=(channel/32)%2;
        if(channel+32<DIM) prefetch(shared,p,slot^1,qb,kb,end,head,channel+32);
        auto query=make_tensor(make_smem_ptr(shared.query+slot*cosize_v<OperandLayout>),OperandLayout{});
        auto key=make_tensor(make_smem_ptr(shared.key+slot*cosize_v<OperandLayout>),OperandLayout{});
        auto qa=local_tile(query,Shape<_8,_32>{},make_coord(warp,_0{}));
        auto a=thread.partition_A(qa);
        auto b=thread.partition_B(key);
        if(active) cute::gemm(mma,a,b,c);
        // Retire this panel before its slot is reused, and publish the next one.
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }
    return scores;
}
}
#endif

#if LOW_PRECISION
struct ForwardValueTraits {
    static constexpr int PartitionsK=FORWARD_WIDTH==32 ? 2:1;
    static constexpr int WarpColumns=FORWARD_WIDTH/(2/PartitionsK);
    using Shape=cutlass::gemm::GemmShape<ForwardRows,FORWARD_WIDTH,ForwardKeys>;
    using Core=cutlass::gemm::threadblock::DefaultMmaCore<Shape,
        cutlass::gemm::GemmShape<16,WarpColumns,ForwardKeys/PartitionsK>,GemmInstruction,
        GemmElement,cutlass::layout::RowMajor,GemmElement,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,GemmClass,2,cutlass::arch::OpMultiplyAdd>;
    using Base=cutlass::gemm::threadblock::MmaBase<Shape,typename Core::MmaPolicy,1>;
    using Operator=typename Core::MmaPolicy::Operator;
    using Accumulator=typename Operator::FragmentC;
};

struct ForwardValueTile {
    typename ForwardValueTraits::Base::SharedStorage operands;
    float result[ForwardRows*FORWARD_WIDTH*ForwardValueTraits::PartitionsK];
};

// The producer has already populated this native shared operand. CUTLASS still
// owns shared synchronization and every warp-K step of the complete mainloop.
struct ForwardReadyOperand {
    using Fragment=cutlass::Array<GemmElement,1>;
    __device__ __forceinline__ ForwardReadyOperand() {}
    template<typename Ref>
    __device__ __forceinline__ ForwardReadyOperand(Ref const&,int) {}
    __device__ __forceinline__ void clear_mask(bool) {}
    __device__ __forceinline__ ForwardReadyOperand& operator++() {return *this;}
    __device__ __forceinline__ void load(Fragment& fragment) const {fragment.clear();}
    __device__ __forceinline__ void store(Fragment const&) const {}
};

__device__ __forceinline__ void forward_value_load(ForwardValueTile& shared,AttentionGemmFeature<> right) {
    // Score has released K; native P and B occupy disjoint regions.
#if COMPACT && VALUE_DIM%8==0
    if(right.aligned) {
        for(int pos=threadIdx.x*8;pos<ForwardKeys*FORWARD_WIDTH;pos+=ForwardThreads*8) {
            const int row=pos/FORWARD_WIDTH,column=pos%FORWARD_WIDTH;
            const bool valid=right.begin+row<right.end && right.channel_base+column<VALUE_DIM;
            const Storage* source=right.source.data+(valid ?
                ((right.begin+row)*HEADS+right.head)*VALUE_DIM+right.channel_base+column:0);
            auto* destination=&shared.operands.operand_B_ref().at({row,column});
            cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(destination,source,valid);
        }
        cutlass::arch::cp_async_fence();
    } else
#endif
    {
        for(int pos=threadIdx.x;pos<ForwardKeys*FORWARD_WIDTH;pos+=ForwardThreads) {
            const int row=pos/FORWARD_WIDTH,column=pos%FORWARD_WIDTH;
            shared.operands.operand_B_ref().at({row,column})=GemmElement(right(row,column));
        }
    }
}

__device__ __forceinline__ void forward_value_gemm(ForwardValueTile& shared,
    typename ForwardValueTraits::Accumulator& output) {
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    using Mma=cutlass::gemm::threadblock::MmaSingleStage<typename ForwardValueTraits::Shape,
        ForwardReadyOperand,ForwardReadyOperand,ForwardReadyOperand,ForwardReadyOperand,
        float,cutlass::layout::RowMajor,typename ForwardValueTraits::Core::MmaPolicy>;
    Mma mma(shared.operands,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    mma(1,output,ForwardReadyOperand{},ForwardReadyOperand{},output);
    __syncthreads();
}

constexpr int ForwardReduction=((DIM+31)/32*32<256 ? (DIM+31)/32*32:256);
struct ForwardScoreTraits {
    using Shape=cutlass::gemm::GemmShape<ForwardRows,ForwardKeys,ForwardReduction>;
    using SmemA=cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16,32>;
    using SmemB=cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16,32>;
    using Operator=typename cutlass::gemm::warp::DefaultMmaTensorOp<
        cutlass::gemm::GemmShape<16,ForwardKeys/2,ForwardReduction>,GemmInstruction,
        GemmElement,SmemA,GemmElement,SmemB,float,cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>::Type;
    using Policy=cutlass::gemm::threadblock::MmaPolicy<Operator,
        cutlass::MatrixShape<0,0>,cutlass::MatrixShape<0,0>,1>;
    using Base=cutlass::gemm::threadblock::MmaBase<Shape,Policy,1>;
    using SharedStorage=typename Base::SharedStorage;
    using Mainloop=cutlass::gemm::threadblock::MmaSingleStage<Shape,
        ForwardReadyOperand,ForwardReadyOperand,ForwardReadyOperand,ForwardReadyOperand,
        float,cutlass::layout::RowMajor,Policy>;
};
using ForwardScoreStorage=typename ForwardScoreTraits::SharedStorage;

template<bool Query>
__device__ __forceinline__ void forward_score_load(ForwardScoreStorage& shared,
    AttentionTensor const& source,int64_t begin,int64_t end,int head,int channel_base) {
    constexpr int Rows=Query ? ForwardRows:ForwardKeys;
#if COMPACT && DIM%8==0
    if((reinterpret_cast<uintptr_t>(source.data)&15)==0) {
        // Match each warp's copy rows to the native 32-element crosswise tile.
        constexpr int ThreadsPerRow=ForwardScoreTraits::SmemA::Base::kCrosswise/8;
        for(int row=threadIdx.x/ThreadsPerRow;row<Rows;row+=ForwardThreads/ThreadsPerRow)
        for(int column=(threadIdx.x%ThreadsPerRow)*8;column<ForwardReduction;column+=ThreadsPerRow*8) {
            const bool valid=begin+row<end && channel_base+column<DIM;
            const Storage* pointer=source.data+(valid ? ((begin+row)*HEADS+head)*DIM+channel_base+column:0);
            GemmElement* destination;
            if constexpr(Query) destination=&shared.operand_A_ref().at({row,column});
            else destination=&shared.operand_B_ref().at({column,row});
            cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(destination,pointer,valid);
        }
        return;
    }
#endif
    for(int pos=threadIdx.x;pos<Rows*ForwardReduction;pos+=ForwardThreads) {
        const int row=pos/ForwardReduction,column=pos%ForwardReduction;
        const GemmElement value=GemmElement(begin+row<end && channel_base+column<DIM ?
            attention_feature(source,begin+row,head,channel_base+column,DIM):0.f);
        if constexpr(Query) shared.operand_A_ref().at({row,column})=value;
        else shared.operand_B_ref().at({column,row})=value;
    }
}
#endif

struct AttentionForwardShared {
#if LOW_PRECISION
    // Q remains live while K and the complete PV tile share the following bytes.
    decltype(ForwardScoreStorage::operand_A) query;
    union GemmStorage {
        decltype(ForwardScoreStorage::operand_B) key;
        ForwardValueTile value;
    } gemm;
#else
    union GemmStorage {
        native_score::Storage score;
        AttentionGemmTile<FORWARD_WIDTH,true> value;
    } gemm;
    GemmElement probability[ForwardRows*ForwardKeys];
#endif
    float maximum[ForwardRows],normalizer[ForwardRows],rescale[ForwardRows];
#if LOW_PRECISION
    float partial[2][ForwardRows];
#endif
};
#if LOW_PRECISION
static_assert(offsetof(AttentionForwardShared,gemm)==offsetof(ForwardScoreStorage,operand_B),
    "The library score storage must address the persistent Q and reusable K tile");
#endif

#if LOW_PRECISION
using ForwardScoreMma=typename ForwardScoreTraits::Operator;

// IteratorC owns two rows per lane. Four lanes share a row, and two N warps
// share its 64 keys; only the row maximum and sum cross the N warps.
__device__ __forceinline__ void forward_probabilities(AttentionForwardShared& shared,
    ForwardScoreMma::FragmentC& scores,AttentionParams const& p,
    int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,part=warp/2;
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        float maximum=-INFINITY;
        #pragma unroll
        for(int n=0;n<4;++n) {
            #pragma unroll
            for(int j=0;j<2;++j) {
                const int i=4*n+2*r+j,column=part*32+n*8+(lane%4)*2+j;
                const bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
                scores[i]=valid ? scores[i]*p.scale+
                    attention_bias<3>(p,segment,head,begin,qb+row,kb+column):-INFINITY;
                maximum=fmaxf(maximum,scores[i]);
            }
        }
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
        if(lane%4==0) shared.partial[part][row]=maximum;
    }
    __syncthreads();
    float next[2],alpha[2];
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        next[r]=fmaxf(shared.maximum[row],fmaxf(shared.partial[0][row],shared.partial[1][row]));
        alpha[r]=shared.maximum[row]>-INFINITY ? __expf(shared.maximum[row]-next[r]):0.f;
    }
    // Every N warp must read the maxima before the same summaries hold sums.
    __syncthreads();
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        float sum=0.f;
        #pragma unroll
        for(int n=0;n<4;++n) {
            const int column=part*32+n*8+(lane%4)*2;
            cutlass::Array<float,2> values;
            #pragma unroll
            for(int j=0;j<2;++j) {
                const int i=4*n+2*r+j;
                values[j]=scores[i]>-INFINITY && next[r]>-INFINITY ? __expf(scores[i]-next[r]):0.f;
                sum+=values[j];
            }
            auto converted=cutlass::NumericArrayConverter<GemmElement,float,2>{}(values);
            *reinterpret_cast<uint32_t*>(&shared.gemm.value.operands.operand_A_ref().at({row,column}))=
                *reinterpret_cast<uint32_t const*>(&converted);
        }
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        if(lane%4==0) shared.partial[part][row]=sum;
    }
    __syncthreads();
    if(part==0 && lane%4==0) {
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int row=(warp%2)*16+lane/4+r*8;
            shared.maximum[row]=next[r];
            shared.normalizer[row]=shared.normalizer[row]*alpha[r]+shared.partial[0][row]+shared.partial[1][row];
            shared.rescale[row]=alpha[r];
        }
    }
    __syncthreads();
}

template<int Tiles>
__device__ __forceinline__ void forward_rescale(
    typename AttentionGemmTraits<FORWARD_WIDTH,true>::Accumulator (&output)[Tiles],float const* rescale) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int row=(warp%2)*16+lane/4+r*8;
        const float alpha=rescale[row];
        #pragma unroll
        for(int tile=0;tile<Tiles;++tile)
            #pragma unroll
            for(int n=0;n<AttentionGemmTraits<FORWARD_WIDTH,true>::WarpColumns/8;++n) {
                output[tile][4*n+2*r]*=alpha;
                output[tile][4*n+2*r+1]*=alpha;
            }
    }
}
#else
template<int Tiles>
__device__ __forceinline__ void forward_rescale(
    typename AttentionGemmTraits<FORWARD_WIDTH,true>::Accumulator (&output)[Tiles],float const* rescale) {
    using Iterator=typename AttentionGemmTraits<FORWARD_WIDTH,true>::Operator::IteratorC;
    using Policy=typename Iterator::Policy;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const auto offset=Policy::get_lane_layout().inverse(lane);
    #pragma unroll
    for(int m=0;m<Iterator::Iterations::kRow;++m) {
        #pragma unroll
        for(int r=0;r<Policy::LaneMmaShape::kM;++r) {
            const int row=(warp%2)*16+offset.row()*Policy::LaneMmaShape::kM+m*Iterator::Delta::kRow+r;
            const float alpha=rescale[row];
            #pragma unroll
            for(int tile=0;tile<Tiles;++tile)
                #pragma unroll
                for(int n=0;n<Iterator::ThreadShape::kColumn;++n)
                    output[tile][(m*Policy::LaneMmaShape::kM+r)*Iterator::ThreadShape::kColumn+n]*=alpha;
        }
    }
}
#endif

__global__ __launch_bounds__(ForwardThreads) void attention_forward_kernel(
    const __grid_constant__ AttentionParams p) {
    const int segment=blockIdx.x,head=blockIdx.y;
    const int64_t begin=p.offset(segment),end=p.offset(segment+1);
    extern __shared__ char storage[];
    auto& shared=*reinterpret_cast<AttentionForwardShared*>(storage);
    auto& gemm=shared.gemm;
#if !LOW_PRECISION
    auto& probability=shared.probability;
#endif
    constexpr int Values=(VALUE_DIM+FORWARD_WIDTH-1)/FORWARD_WIDTH;
#if LOW_PRECISION
    using ValueTraits=ForwardValueTraits;
#else
    using ValueTraits=AttentionGemmTraits<FORWARD_WIDTH,true>;
#endif
    using ValueMma=typename ValueTraits::Operator;
    constexpr int WarpsMN=ForwardThreads/(32*ValueTraits::PartitionsK);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int value_offset=(warp/WarpsMN)*ForwardRows*FORWARD_WIDTH+
        (warp%2)*16*FORWARD_WIDTH+
        ((warp%WarpsMN)/2)*ValueTraits::WarpColumns;
    for(int64_t qb=begin+int64_t(blockIdx.z)*ForwardRows;qb<end;qb+=int64_t(gridDim.z)*ForwardRows) {
        typename ValueMma::FragmentC output[Values];
        #pragma unroll
        for(int tile=0;tile<Values;++tile) output[tile].clear();
#if LOW_PRECISION
        auto& score_storage=*reinterpret_cast<ForwardScoreStorage*>(&shared);
        if constexpr(DIM<=ForwardReduction) {
            forward_score_load<true>(score_storage,p.inputs[0],qb,end,head,0);
            cutlass::arch::cp_async_fence();
        }
#endif
        for(int row=threadIdx.x;row<ForwardRows;row+=ForwardThreads) {
            shared.maximum[row]=-INFINITY;
            shared.normalizer[row]=0.f;
        }
        __syncthreads();
        for(int64_t kb=begin;kb<end;kb+=ForwardKeys) {
#if LOW_PRECISION
            ForwardScoreMma::FragmentC scores;
            scores.clear();
            for(int channel=0;channel<DIM;channel+=ForwardReduction) {
                if constexpr(DIM>ForwardReduction)
                    forward_score_load<true>(score_storage,p.inputs[0],qb,end,head,channel);
                forward_score_load<false>(score_storage,p.inputs[1],kb,end,head,channel);
                cutlass::arch::cp_async_fence();
                cutlass::arch::cp_async_wait<0>();
                typename ForwardScoreTraits::Mainloop mma(score_storage,threadIdx.x,warp,lane);
                mma(1,scores,ForwardReadyOperand{},ForwardReadyOperand{},scores);
            }
            forward_value_load(gemm.value,
                AttentionGemmFeature<>(p.inputs[2],kb,end,head,VALUE_DIM,0));
            forward_probabilities(shared,scores,p,begin,end,qb,kb,segment,head);
            forward_rescale(output,shared.rescale);
#else
            auto scores=native_score::gemm(gemm.score,p,qb,kb,end,head);
            #pragma unroll
            for(int r=0;r<2;++r) {
                const int row=warp*8+lane/8*2+r;
                float maximum=-INFINITY;
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int column=lane%8*4+j;
                    const bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
                    scores[r*4+j]=valid ? scores[r*4+j]*p.scale+
                        attention_bias<3>(p,segment,head,begin,qb+row,kb+column):-INFINITY;
                    maximum=fmaxf(maximum,scores[r*4+j]);
                }
                #pragma unroll
                for(int shift=4;shift;shift/=2)
                    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift));
                const float next=fmaxf(shared.maximum[row],maximum);
                const float alpha=shared.maximum[row]>-INFINITY ? __expf(shared.maximum[row]-next):0.f;
                float sum=0.f;
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const float value=scores[r*4+j]>-INFINITY && next>-INFINITY ? __expf(scores[r*4+j]-next):0.f;
                    probability[row*ForwardKeys+lane%8*4+j]=value;
                    sum+=value;
                }
                #pragma unroll
                for(int shift=4;shift;shift/=2) sum+=__shfl_xor_sync(0xffffffff,sum,shift);
                __syncwarp();
                if(lane%8==0) {
                    shared.maximum[row]=next;
                    shared.normalizer[row]=shared.normalizer[row]*alpha+sum;
                    shared.rescale[row]=alpha;
                }
            }
            __syncthreads();
            forward_rescale(output,shared.rescale);
#endif
            #pragma unroll
            for(int tile=0;tile<Values;++tile) {
#if LOW_PRECISION
                if(tile) forward_value_load(gemm.value,
                    AttentionGemmFeature<>(p.inputs[2],kb,end,head,VALUE_DIM,tile*FORWARD_WIDTH));
                forward_value_gemm(gemm.value,output[tile]);
#else
                attention_gemm(gemm.value,gemm.value.result,int(min(int64_t(ForwardKeys),end-kb)),
                    [&](int row,int k) {return probability[row*ForwardKeys+k];},
                    [&](int k,int column) {return kb+k<end && tile*FORWARD_WIDTH+column<VALUE_DIM ?
                        attention_feature(p.inputs[2],kb+k,head,tile*FORWARD_WIDTH+column,VALUE_DIM):0.f;},
                    &output[tile]);
#endif
            }
        }
        for(int row=threadIdx.x;row<ForwardRows;row+=ForwardThreads)
            shared.normalizer[row]=shared.normalizer[row]>0.f ? 1.f/shared.normalizer[row]:0.f;
        __syncthreads();
        #pragma unroll
        for(int tile=0;tile<Values;++tile) {
            typename ValueMma::IteratorC iterator({gemm.value.result+value_offset,cutlass::layout::RowMajor(FORWARD_WIDTH)},lane);
            iterator.store(output[tile]);
            __syncthreads();
            if constexpr(ValueTraits::PartitionsK==2) {
                for(int pos=threadIdx.x;pos<ForwardRows*FORWARD_WIDTH;pos+=ForwardThreads)
                    gemm.value.result[pos]+=gemm.value.result[ForwardRows*FORWARD_WIDTH+pos];
                __syncthreads();
            }
            constexpr int Pack=16/sizeof(Storage);
            if constexpr(VALUE_DIM%Pack==0) {
                for(int pos=threadIdx.x*Pack;pos<ForwardRows*FORWARD_WIDTH;pos+=ForwardThreads*Pack) {
                    const int row=pos/FORWARD_WIDTH,column=tile*FORWARD_WIDTH+pos%FORWARD_WIDTH;
                    if(qb+row<end && column+Pack<=VALUE_DIM) {
                        cutlass::Array<float,Pack> values;
                        const float inverse=shared.normalizer[row];
                        #pragma unroll
                        for(int i=0;i<Pack;++i) values[i]=gemm.value.result[pos+i]*inverse;
                        auto converted=cutlass::NumericArrayConverter<GemmElement,float,Pack>{}(values);
                        *reinterpret_cast<uint4*>(p.outputs[0].data+((qb+row)*HEADS+head)*VALUE_DIM+column)=
                            *reinterpret_cast<uint4 const*>(&converted);
                    }
                }
            } else {
                for(int pos=threadIdx.x;pos<ForwardRows*FORWARD_WIDTH;pos+=ForwardThreads) {
                    int row=pos/FORWARD_WIDTH,column=tile*FORWARD_WIDTH+pos%FORWARD_WIDTH;
                    if(qb+row<end && column<VALUE_DIM)
                        p.outputs[0].data[((qb+row)*HEADS+head)*VALUE_DIM+column]=Storage(gemm.value.result[pos]*shared.normalizer[row]);
                }
            }
            __syncthreads();
        }
        for(int row=threadIdx.x;row<ForwardRows;row+=ForwardThreads) if(qb+row<end) {
            const int64_t index=((qb+row)*HEADS+head)*2;
            p.state.data[index]=shared.maximum[row];
            p.state.data[index+1]=shared.normalizer[row];
        }
        __syncthreads();
    }
}

std::vector<at::Tensor> attention_forward(std::vector<std::optional<at::Tensor>> inputs,
    at::Tensor rowptr,double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto output=at::empty(inputs[2]->sizes(),inputs[2]->options());
    auto state=at::empty({inputs[0]->size(0),int64_t(HEADS),int64_t(2)},inputs[0]->options().dtype(at::kFloat));
    auto p=make_params(inputs,{output},rowptr,state,scale);
    if(p.groups) {
        attention_forward_kernel<<<dim3(p.groups,HEADS,WORKERS),ForwardThreads,
            sizeof(AttentionForwardShared),at::cuda::getCurrentCUDAStream()>>>(p);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output,state};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run",&attention_forward);
    m.def("configure",[]() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_forward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttentionForwardShared)));
    });
}

"""


def attention_forward(inputs, rowptr, config):
    if not inputs[0].shape[0]:
        value = inputs[2]
        return (torch.empty_like(value),
                torch.empty((*value.shape[:2], 2), device=value.device, dtype=torch.float32))
    dim, value_dim, heads, scale, causal, bias_kind = config
    width = min(128, 1 << max(5, (value_dim-1).bit_length()))
    rows = 32
    workers = min(4, max(1, (inputs[0].shape[0]+rows*max(1, rowptr.numel()-1)-1)
                            // (rows*max(1, rowptr.numel()-1))))
    compact = all(value.is_contiguous() for i, value in enumerate(inputs)
                  if value is not None and i not in (3, 8, 12))
    key = compact, dim, value_dim, heads, causal, bias_kind, workers, inputs[0].device, inputs[0].dtype
    kernel = KERNELS.get(key)
    if kernel is None:
        defines = (f"#define FORWARD_WIDTH {width}\n#define DIM {dim}\n#define VALUE_DIM {value_dim}\n"
                   f"#define COMPACT {int(compact)}\n#define HEADS {heads}\n#define CAUSAL {int(causal)}\n"
                   f"#define BIAS_KIND {bias_kind}\n#define PRESENT {15 if bias_kind else 7}\n"
                   f"#define NEEDS 1\n#define WORKERS {workers}\n")
        module = load_cuda_extension(
            "attention_forward", dtype_defines(inputs[0].dtype) + defines + CUDA_SOURCE
            + GEMM_SOURCE + ATTENTION_SOURCE + FORWARD_SOURCE,
            arch=get_cuda_arch(inputs[0].device), use_fast_math=False, minimal_headers=True,
        )
        with torch.cuda.device(inputs[0].device):
            module.configure()
        kernel = module.run
        KERNELS[key] = kernel
    return tuple(kernel(inputs, rowptr, scale))
