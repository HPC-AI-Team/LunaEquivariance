"""Fused first backward from recomputed probability tiles."""

import torch

from .utils import CUDA_SOURCE, GEMM_SOURCE, ATTENTION_SOURCE, BIAS_SOURCE, dtype_defines
from ....utils.extension import get_cuda_arch, load_cuda_extension

KERNELS = {}

BACKWARD_SOURCE = r"""
#if !LOW_PRECISION
template<int Left,int Right>
__device__ __forceinline__ void attention_pair(AttentionParams const& p,AttentionGemmStorage& shared,
    float* output,int64_t qb,int64_t kb,int64_t end,int head,int channels) {
    if constexpr ((PRESENT&(1<<Left)) && (PRESENT&(1<<Right))) {
        attention_gemm(shared,output,channels,
            [&](int row,int k) {return qb+row<end ? attention_feature(p.inputs[Left],qb+row,head,k,channels):0.f;},
            [&](int k,int column) {return kb+column<end ? attention_feature(p.inputs[Right],kb+column,head,k,channels):0.f;});
    } else {
        for(int i=threadIdx.x;i<1024;i+=AttentionThreads) output[(i/32)*AttentionScoreStride+i%32]=0.f;
        __syncthreads();
    }
}

template<int Target,int Source,bool Transpose=false,int Columns,bool RowMajorB>
__device__ __forceinline__ void attention_output(AttentionParams const& p,AttentionGemmTile<Columns,RowMajorB>& shared,
    float const* coefficient,float* product,int64_t qb,int64_t kb,int64_t end,
    int head,int channels,float scale=1.f,bool first=false) {
    if constexpr ((NEEDS&(1<<Target)) && (PRESENT&(1<<Source))) {
        const int64_t target_base = Transpose ? kb:qb;
        const int64_t source_base = Transpose ? qb:kb;
        for(int channel=0;channel<channels;channel+=Columns) {
            attention_gemm(shared,product,int(min(int64_t(32),end-source_base)),
                [&](int row,int k) {return coefficient[Transpose ? k*AttentionScoreStride+row:row*AttentionScoreStride+k];},
                [&](int k,int column) {return source_base+k<end && channel+column<channels ?
                    attention_feature(p.inputs[Source],source_base+k,head,channel+column,channels):0.f;});
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            if constexpr(Transpose && (Target==2 ? VALUE_DIM:DIM)%4==0) {
                for(int i=threadIdx.x*4;i<32*Columns;i+=AttentionThreads*4) {
                    int row=i/Columns,c=channel+i%Columns;
                    if(target_base+row<end && c<channels) {
                        float4 value=make_float4(product[i]*scale,product[i+1]*scale,
                            product[i+2]*scale,product[i+3]*scale);
                        atomicAdd(reinterpret_cast<float4*>(reinterpret_cast<float*>(p.outputs[Target].data)
                            +((target_base+row)*HEADS+head)*channels+c),value);
                    }
                }
            } else
#endif
            {
                for(int i=threadIdx.x;i<32*Columns;i+=AttentionThreads) {
                    int row=i/Columns,c=channel+i%Columns;
                    if(target_base+row<end && c<channels) {
                        float* destination=reinterpret_cast<float*>(p.outputs[Target].data)
                            +((target_base+row)*HEADS+head)*channels+c;
                        if constexpr(Transpose) atomicAdd(destination,product[i]*scale);
                        else if(first) *destination=product[i]*scale;
                        else *destination+=product[i]*scale;
                    }
                }
            }
            __syncthreads();
        }
    }
}
#endif

#if LOW_PRECISION
#include <cute/tensor.hpp>
namespace backward_ready {
using namespace cute;
constexpr int Threads=128;
constexpr int KeyCapacity=((DIM<256 ? DIM:256)+63)/64*64;
constexpr int ValueCapacity=((VALUE_DIM<256 ? VALUE_DIM:256)+63)/64*64;
using Atom=std::conditional_t<std::is_same_v<GemmElement,cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
using Mma=TiledMMA<Atom,Layout<Shape<_2,_2,_1>>,Tile<_32,_32,_16>>;
using CopyN=Copy_Atom<SM75_U32x4_LDSM_N,GemmElement>;
using CopyT=Copy_Atom<SM75_U16x8_LDSM_T,GemmElement>;
using FeatureAtom=decltype(composition(Swizzle<3,3,3>{},Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
template<int Rows,int Columns>
using FeatureLayout=decltype(tile_to_shape(FeatureAtom{},Shape<Int<Rows>,Int<Columns>>{}));
using CoefficientAtom=decltype(composition(Swizzle<2,3,3>{},Layout<Shape<_8,_32>,Stride<_32,_1>>{}));
using CoefficientLayout=decltype(tile_to_shape(CoefficientAtom{},Shape<_32,_32>{}));

template<typename Tensor>
__device__ __forceinline__ auto transpose(Tensor const& tensor) {
    return make_tensor(tensor.data(),composition(tensor.layout(),
        make_layout(make_shape(size<1>(tensor),size<0>(tensor)),GenRowMajor{})));
}

template<typename Acc,typename SA,typename SB,typename Mma,typename CA,typename CB>
__device__ __forceinline__ void gemm(Acc& acc,SA const& a,SB const& b,Mma mma,CA ca,CB cb) {
    auto thr=mma.get_thread_slice(threadIdx.x);
    auto ra=thr.partition_fragment_A(make_tensor(a.data(),get_nonswizzle_portion(a.layout())));
    auto rb=thr.partition_fragment_B(make_tensor(b.data(),get_nonswizzle_portion(b.layout())));
    auto copya=make_tiled_copy_A(ca,mma);
    auto copyb=make_tiled_copy_B(cb,mma);
    auto ta=copya.get_thread_slice(threadIdx.x);
    auto tb=copyb.get_thread_slice(threadIdx.x);
    auto sa=ta.partition_S(a);
    auto sb=tb.partition_S(b);
    auto va=ta.retile_D(ra);
    auto vb=tb.retile_D(rb);
    cute::copy(copya,sa(_,_,_0{}),va(_,_,_0{}));
    cute::copy(copyb,sb(_,_,_0{}),vb(_,_,_0{}));
    #pragma unroll
    for(int k=0;k<size<2>(ra);++k) {
        if(k+1<size<2>(ra)) {
            cute::copy(copya,sa(_,_,k+1),va(_,_,k+1));
            cute::copy(copyb,sb(_,_,k+1),vb(_,_,k+1));
        }
        cute::gemm(mma,ra(_,_,k),rb(_,_,k),acc);
    }
}

template<int Channels,int Rows,int Columns>
__device__ __forceinline__ void load_feature(GemmElement* output,AttentionTensor const& source,
    int64_t begin,int64_t end,int head,int base) {
    using Layout=FeatureLayout<Rows,Columns>;
#if COMPACT
    if constexpr(Channels%8==0) if((reinterpret_cast<uintptr_t>(source.data)&15)==0) {
        for(int pos=threadIdx.x*8;pos<Rows*Columns;pos+=Threads*8) {
            int row=pos/Columns,column=pos%Columns;
            bool valid=begin+row<end && base+column<Channels;
            auto pointer=source.data+(valid ? ((begin+row)*HEADS+head)*Channels+base+column:0);
            cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Global>(
                output+Layout{}(row,column),pointer,valid);
        }
        return;
    }
#endif
    for(int pos=threadIdx.x;pos<Rows*Columns;pos+=Threads) {
        int row=pos/Columns,column=pos%Columns;
        output[Layout{}(row,column)]=GemmElement(begin+row<end && base+column<Channels ?
            attention_feature(source,begin+row,head,base+column,Channels):0.f);
    }
}
__device__ __forceinline__ void wait_feature() {
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
}

template<int Channels,int Capacity,typename Acc>
__device__ __forceinline__ void score(Acc& acc,GemmElement* left_resident,GemmElement* right_resident,
    GemmElement* stream_left,GemmElement* stream_right,AttentionTensor const& left,AttentionTensor const& right,
    int64_t qb,int64_t kb,int64_t end,int head,int channel_base) {
    auto a=make_tensor(make_smem_ptr(left_resident),FeatureLayout<32,Capacity>{});
    auto b=make_tensor(make_smem_ptr(right_resident),FeatureLayout<32,Capacity>{});
    for(int c=0;c<Channels;c+=64) {
        if(c>=channel_base && c<channel_base+Capacity) {
            auto at=local_tile(a,Shape<_32,_64>{},make_coord(_0{},(c-channel_base)/64));
            auto bt=local_tile(b,Shape<_32,_64>{},make_coord(_0{},(c-channel_base)/64));
            gemm(acc,at,bt,Mma{},CopyN{},CopyN{});
        } else {
            load_feature<Channels,32,64>(stream_left,left,qb,end,head,c);
            load_feature<Channels,32,64>(stream_right,right,kb,end,head,c);
            wait_feature();
            auto at=make_tensor(make_smem_ptr(stream_left),FeatureLayout<32,64>{});
            auto bt=make_tensor(make_smem_ptr(stream_right),FeatureLayout<32,64>{});
            gemm(acc,at,bt,Mma{},CopyN{},CopyN{});
            __syncthreads();
        }
    }
}
} // namespace backward_ready

#endif
#if !LOW_PRECISION
namespace score_value {

// Two independent dot products. There are no cross-component products.
struct alignas(8) Element {
    float score, value;

    __host__ __device__ __forceinline__ Element(float x = 0.f) : score(x), value(0.f) {}
    __host__ __device__ __forceinline__ Element(float score_, float value_)
        : score(score_), value(value_) {}
};

struct Product {
    Element left, right;

    __host__ __device__ __forceinline__ Element operator+(Element const& accumulator) const {
        return {fmaf(left.score, right.score, accumulator.score),
                fmaf(left.value, right.value, accumulator.value)};
    }
};

__host__ __device__ __forceinline__ Product operator*(Element const& left, Element const& right) {
    return {left, right};
}

static_assert(sizeof(Element) == 8 && alignof(Element) == 8);
using Shape = cutlass::gemm::GemmShape<32, 32, 8>;
using Core = cutlass::gemm::threadblock::DefaultMmaCore<
    Shape, cutlass::gemm::GemmShape<16, 16, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, 2, cutlass::arch::OpMultiplyAdd>;
using Base = cutlass::gemm::threadblock::MmaBase<Shape, typename Core::MmaPolicy, 1>;
using Operator = typename Core::MmaPolicy::Operator;

struct alignas(16) Shared {
    typename Base::SharedStorage operands;
};

template<typename ThreadMap, typename Producer, bool Left>
struct ProducerIterator {
    using Element = score_value::Element;
    using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    Producer producer;
    cutlass::PitchLinearCoord origin;
    int base = 0, reduction;
    bool enabled = true;

    __device__ __forceinline__ ProducerIterator(Producer producer_, int reduction_)
        : producer(producer_), origin(ThreadMap::initial_offset(threadIdx.x)), reduction(reduction_) {}
    __device__ __forceinline__ void clear_mask(bool clear) { enabled = enabled && !clear; }
    __device__ __forceinline__ ProducerIterator& operator++() { base += 8; return *this; }
    __device__ __forceinline__ void load(Fragment& fragment) const {
        #pragma unroll
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s)
            #pragma unroll
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c)
                #pragma unroll
                for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                    int k = base + origin.contiguous() + c * ThreadMap::Delta::kContiguous + e;
                    int outer = origin.strided() + s * ThreadMap::Delta::kStrided;
                    int index = (s * ThreadMap::Iterations::kContiguous + c) * ThreadMap::kElementsPerAccess + e;
                    Element element;
                    if (enabled && k < reduction) {
                        if constexpr (Left) element = producer(outer, k);
                        else element = producer(k, outer);
                    }
                    fragment[index] = element;
                }
    }
};

template<typename Fragment, typename Project>
__device__ __forceinline__ void store_plane(Fragment const& accumulator, float* output, Project project) {
    using IteratorC = typename Operator::IteratorC;
    using FloatIterator = cutlass::gemm::warp::MmaSimtTileIterator<
        typename IteratorC::Shape, cutlass::gemm::Operand::kC, float,
        cutlass::layout::RowMajor, typename IteratorC::Policy>;
    static_assert(FloatIterator::Fragment::kElements == Fragment::kElements);
    typename FloatIterator::Fragment field;
    #pragma unroll
    for (int i = 0; i < Fragment::kElements; ++i) field[i] = project(accumulator[i]);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    FloatIterator iterator(
        {output + (warp % 2) * 16 * AttentionScoreStride + (warp / 2) * 16, cutlass::layout::RowMajor(AttentionScoreStride)}, lane);
    iterator.store(field);
}

template<typename Left, typename Right>
__device__ __forceinline__ void gemm(Shared& shared, float* score, float* value,
    int reduction, Left left, Right right) {
    using IteratorA = ProducerIterator<typename Core::IteratorThreadMapA, Left, true>;
    using IteratorB = ProducerIterator<typename Core::IteratorThreadMapB, Right, false>;
    using ThreadblockMma = cutlass::gemm::threadblock::MmaSingleStage<
        Shape, IteratorA, typename Core::SmemIteratorA, IteratorB, typename Core::SmemIteratorB,
        Element, cutlass::layout::RowMajor, typename Core::MmaPolicy>;
    typename ThreadblockMma::FragmentC accumulator;
    accumulator.clear();
    ThreadblockMma mma(shared.operands, threadIdx.x, threadIdx.x / 32, threadIdx.x % 32);
    mma((reduction + 7) / 8, accumulator, IteratorA(left, reduction), IteratorB(right, reduction), accumulator);
    store_plane(accumulator, score, [](Element const& x) { return x.score; });
    store_plane(accumulator, value, [](Element const& x) { return x.value; });
    __syncthreads();
}

} // namespace score_value
#endif

#if LOW_PRECISION
struct alignas(16) AttentionBackwardShared {
    alignas(16) GemmElement query[32*backward_ready::KeyCapacity];
    alignas(16) GemmElement gradient[32*backward_ready::ValueCapacity];
    alignas(16) GemmElement key[32*backward_ready::KeyCapacity];
    alignas(16) GemmElement feature_value[(NEEDS&11) ? 32*backward_ready::ValueCapacity:1];
    alignas(16) GemmElement stream_left[(DIM>256 || VALUE_DIM>256) ? 32*64:1];
    alignas(16) GemmElement stream_right[(DIM>256 || VALUE_DIM>256) ? 32*64:1];
    alignas(16) GemmElement probability[(NEEDS&4) ? 32*32:1];
    alignas(16) GemmElement value[(NEEDS&3) ? 32*32:1];
    float bias_gradient[((NEEDS&8) && BIAS_KIND!=0) ? 32*32:1];
    float delta[32];
};

#else
union AttentionBackwardScore {
    AttentionGemmStorage scalar;
    score_value::Shared packed;
};

__device__ __forceinline__ void backward_pairs(AttentionParams const& p,AttentionBackwardScore& gemm,
    float* probability,float* value,int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head) {
    if constexpr ((NEEDS&11) && (PRESENT&7)==7) {
        score_value::gemm(gemm.packed,probability,value,DIM>VALUE_DIM ? DIM:VALUE_DIM,
            [&](int row,int k) {
                score_value::Element x;
                if(qb+row<end) {
                    if(k<DIM) x.score=attention_feature(p.inputs[0],qb+row,head,k,DIM);
                    if constexpr(PRESENT&16)
                        if(k<VALUE_DIM) x.value=attention_feature(p.inputs[4],qb+row,head,k,VALUE_DIM);
                }
                return x;
            },
            [&](int k,int column) {
                score_value::Element x;
                if(kb+column<end) {
                    if(k<DIM) x.score=attention_feature(p.inputs[1],kb+column,head,k,DIM);
                    if(k<VALUE_DIM) x.value=attention_feature(p.inputs[2],kb+column,head,k,VALUE_DIM);
                }
                return x;
            });
    } else {
        attention_pair<0,1>(p,gemm.scalar,probability,qb,kb,end,head,DIM);
        if constexpr(NEEDS&11) attention_pair<4,2>(p,gemm.scalar,value,qb,kb,end,head,VALUE_DIM);
    }
    for(int i=threadIdx.x;i<1024;i+=128) {
        int row=i/32,column=i%32;
        bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
        float prob=0.f;
        if(valid && p.state.at(qb+row,head,1)>0.f) {
            float score=probability[(i/32)*AttentionScoreStride+i%32]*p.scale+attention_bias<3>(p,segment,head,begin,qb+row,kb+column);
            prob=__expf(score-p.state.at(qb+row,head,0))*p.state.at(qb+row,head,1);
        }
        probability[(i/32)*AttentionScoreStride+i%32]=prob;
    }
    __syncthreads();
}

union AttentionBackwardGemm {
    AttentionBackwardScore score;
    AttentionGemmTile<BACKWARD_WIDTH,true> output;
};
struct AttentionBackwardShared {
    AttentionBackwardGemm gemm;
    float probability[32*AttentionScoreStride],value[32*AttentionScoreStride],delta[32];
};

#endif

#if LOW_PRECISION && (NEEDS&9)
// The existing workspace initialization also computes one softmax-centering
// scalar per query, shared by all key owners without keeping g and O live.
__global__ void attention_center_initialize_kernel(
    const __grid_constant__ AttentionParams p) {
    const int segment=blockIdx.x,head=blockIdx.y;
    const int64_t begin=p.offset(segment),end=p.offset(segment+1);
    attention_initialize<4>(p,segment,head,begin,end);
    const int lane=threadIdx.x%8,row_group=threadIdx.x/8;
    const unsigned mask=0xffu<<((threadIdx.x%32)/8*8);
    for(int64_t row=begin+row_group;row<end;row+=16) {
        float sum=0.f;
        AttentionGemmFeature<> grad(p.inputs[4],row,row+1,head,VALUE_DIM);
        AttentionGemmFeature<> output(p.inputs[5],row,row+1,head,VALUE_DIM);
        for(int c=lane*8;c<VALUE_DIM;c+=8*8) {
            alignas(16) cutlass::Array<GemmElement,8> g,o;
            grad.template load8<false>(g,0,0,c);
            output.template load8<false>(o,0,0,c);
            auto gf=cutlass::NumericArrayConverter<float,GemmElement,8>{}(g);
            auto of=cutlass::NumericArrayConverter<float,GemmElement,8>{}(o);
            #pragma unroll
            for(int i=0;i<8;++i) sum+=gf[i]*of[i];
        }
        #pragma unroll
        for(int delta=4;delta>0;delta/=2) sum+=__shfl_xor_sync(mask,sum,delta,8);
        if(lane==0) reinterpret_cast<float*>(p.inputs[6].data)[row*HEADS+head]=sum;
    }
}
#endif

__global__ __launch_bounds__(128) void attention_backward_kernel(
    const __grid_constant__ AttentionParams p) {
    int segment=blockIdx.x,head=blockIdx.y;
    int64_t begin=p.offset(segment),end=p.offset(segment+1);
    extern __shared__ char storage[];
    auto& shared=*reinterpret_cast<AttentionBackwardShared*>(storage);
#if !LOW_PRECISION
    auto& gemm=shared.gemm;
#endif
    auto& probability=shared.probability;
    auto& value=shared.value;
    auto& delta=shared.delta;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
#if LOW_PRECISION
    using namespace backward_ready;
    constexpr int QueryChannels=(NEEDS&3) ? DIM:0;
    constexpr int ValueChannels=(NEEDS&4) ? VALUE_DIM:0;
    constexpr int Channels=QueryChannels>ValueChannels ? QueryChannels:ValueChannels;
    constexpr int ChannelGroups=(Channels+255)/256;
    const int worker=blockIdx.z%4;
    const int channel_base=ChannelGroups<=1 ? 0:(blockIdx.z/4)*256;
    auto native_query=make_tensor(make_smem_ptr(shared.query),FeatureLayout<32,KeyCapacity>{});
    auto native_gradient=make_tensor(make_smem_ptr(shared.gradient),FeatureLayout<32,ValueCapacity>{});
    auto native_key=make_tensor(make_smem_ptr(shared.key),FeatureLayout<32,KeyCapacity>{});
    auto coefficient=make_tensor(make_smem_ptr(shared.value),CoefficientLayout{});
    auto prob_coefficient=make_tensor(make_smem_ptr(shared.probability),CoefficientLayout{});
    auto coords=Mma{}.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_32,_32>{}));
    auto dq_coords=Mma{}.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_32,_64>{}));
    auto key_coords=Mma{}.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_32,Int<KeyCapacity>>{}));
    auto value_coords=Mma{}.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_32,Int<ValueCapacity>>{}));
    for(int64_t kb=begin+int64_t(worker)*32;kb<end;kb+=4*32) {
        load_feature<DIM,32,KeyCapacity>(shared.key,p.inputs[1],kb,end,head,channel_base);
        if constexpr(NEEDS&11)
            load_feature<VALUE_DIM,32,ValueCapacity>(shared.feature_value,p.inputs[2],kb,end,head,channel_base);
        wait_feature();
        auto dk=partition_fragment_C(Mma{},Shape<_32,Int<KeyCapacity>>{});
        auto dv=partition_fragment_C(Mma{},Shape<_32,Int<ValueCapacity>>{});
        if constexpr(NEEDS&2) clear(dk);
        if constexpr(NEEDS&4) clear(dv);
        for(int64_t qb=begin;qb<end;qb+=32) {
            load_feature<DIM,32,KeyCapacity>(shared.query,p.inputs[0],qb,end,head,channel_base);
            load_feature<VALUE_DIM,32,ValueCapacity>(shared.gradient,p.inputs[4],qb,end,head,channel_base);
            if constexpr(NEEDS&9) {
                if(threadIdx.x<32) delta[threadIdx.x]=qb+threadIdx.x<end ?
                    reinterpret_cast<float const*>(p.inputs[6].data)[(qb+threadIdx.x)*HEADS+head]:0.f;
            } else if constexpr(NEEDS&11) {
                for(int row=warp;row<32;row+=4) {
                    float sum=0.f;
                    if(qb+row<end) for(int c=lane;c<VALUE_DIM;c+=32)
                        sum+=attention_feature(p.inputs[4],qb+row,head,c,VALUE_DIM)*
                             attention_feature(p.inputs[5],qb+row,head,c,VALUE_DIM);
                    sum=attention_sum(sum);
                    if(lane==0) delta[row]=sum;
                }
            }
            wait_feature();
            {
                auto scores=partition_fragment_C(Mma{},Shape<_32,_32>{});
                auto products=partition_fragment_C(Mma{},Shape<_32,_32>{});
                clear(scores);clear(products);
                score<DIM,KeyCapacity>(scores,shared.query,shared.key,shared.stream_left,shared.stream_right,
                    p.inputs[0],p.inputs[1],qb,kb,end,head,channel_base);
                if constexpr(NEEDS&11)
                    score<VALUE_DIM,ValueCapacity>(products,shared.gradient,shared.feature_value,shared.stream_left,shared.stream_right,
                        p.inputs[4],p.inputs[2],qb,kb,end,head,channel_base);
                #pragma unroll
                for(int i=0;i<size(scores);++i) {
                    const int row=get<0>(coords(i)),column=get<1>(coords(i));
                    const bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
                    float prob=0.f;
                    if(valid && p.state.at(qb+row,head,1)>0.f) {
                        const float score=scores(i)*p.scale+attention_bias<3>(p,segment,head,begin,qb+row,kb+column);
                        prob=__expf(score-p.state.at(qb+row,head,0))*p.state.at(qb+row,head,1);
                    }
                    if constexpr(NEEDS&4) prob_coefficient(row,column)=GemmElement(prob);
                    if constexpr(NEEDS&11) {
                        const float ds=prob*(products(i)-delta[row]);
                        if constexpr(NEEDS&3) coefficient(row,column)=GemmElement(ds);
                        if constexpr((NEEDS&8) && BIAS_KIND!=0) shared.bias_gradient[row*32+column]=ds;
                    }
                }
            }
            __syncthreads();
            if constexpr(NEEDS&1) for(int local=0;local<KeyCapacity && channel_base+local<DIM;local+=64) {
                auto dq=partition_fragment_C(Mma{},Shape<_32,_64>{});
                clear(dq);
                auto key_tile=local_tile(native_key,Shape<_32,_64>{},make_coord(_0{},local/64));
                gemm(dq,coefficient,transpose(key_tile),Mma{},CopyN{},CopyT{});
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
                if constexpr(DIM%4==0) {
                    #pragma unroll
                    for(int i=0;i<size(dq);i+=2) {
                        float x=dq(i)*p.scale,y=dq(i+1)*p.scale;
                        float z=__shfl_down_sync(0xffffffff,x,1),w=__shfl_down_sync(0xffffffff,y,1);
                        int row=get<0>(dq_coords(i)),c=channel_base+local+get<1>(dq_coords(i));
                        if(lane%2==0 && qb+row<end && c<DIM)
                            atomicAdd(reinterpret_cast<float4*>(reinterpret_cast<float*>(p.outputs[0].data)
                                +((qb+row)*HEADS+head)*DIM+c),make_float4(x,y,z,w));
                    }
                } else
#endif
                {
                    #pragma unroll
                    for(int i=0;i<size(dq);++i) {
                        int row=get<0>(dq_coords(i)),c=channel_base+local+get<1>(dq_coords(i));
                        if(qb+row<end && c<DIM)
                            atomicAdd(reinterpret_cast<float*>(p.outputs[0].data)+((qb+row)*HEADS+head)*DIM+c,dq(i)*p.scale);
                    }
                }
                __syncthreads();
            }
            if constexpr(NEEDS&2)
                gemm(dk,transpose(coefficient),transpose(native_query),Mma{},CopyT{},CopyT{});
            if constexpr(NEEDS&4)
                gemm(dv,transpose(prob_coefficient),transpose(native_gradient),Mma{},CopyT{},CopyT{});
            __syncthreads();
            if constexpr((NEEDS&8) && BIAS_KIND!=0) if(channel_base==0) {
                for(int i=threadIdx.x;i<32*32;i+=128) {
                    int row=i/32,column=i%32;
                    if(qb+row<end && kb+column<end) {
                        if constexpr(BIAS_KIND==2)
                            p.outputs[3].fp32_at(segment,head,qb+row-begin,kb+column-begin)=shared.bias_gradient[i];
                    }
                }
                if constexpr(BIAS_KIND==1) if(threadIdx.x<32 && kb+threadIdx.x<end) {
                    float sum=0.f;
                    for(int row=0;row<32;++row) sum+=shared.bias_gradient[row*32+threadIdx.x];
                    atomicAdd(&p.outputs[3].fp32_at(kb+threadIdx.x,head),sum);
                }
                __syncthreads();
            }
        }
        if constexpr(NEEDS&2) {
            if constexpr(DIM%2==0) {
                #pragma unroll
                for(int i=0;i<size(dk);i+=2) {
                    int row=get<0>(key_coords(i)),c=channel_base+get<1>(key_coords(i));
                    if(kb+row<end && c<DIM) {
                        Storage x=Storage(dk(i)*p.scale),y=Storage(dk(i+1)*p.scale);
                        uint32_t packed=uint32_t(x.x)|(uint32_t(y.x)<<16);
                        *reinterpret_cast<uint32_t*>(p.outputs[1].data+((kb+row)*HEADS+head)*DIM+c)=packed;
                    }
                }
            } else {
                #pragma unroll
                for(int i=0;i<size(dk);++i) {
                    int row=get<0>(key_coords(i)),c=channel_base+get<1>(key_coords(i));
                    if(kb+row<end && c<DIM)
                        p.outputs[1].data[((kb+row)*HEADS+head)*DIM+c]=Storage(dk(i)*p.scale);
                }
            }
        }
        if constexpr(NEEDS&4) {
            if constexpr(VALUE_DIM%2==0) {
                #pragma unroll
                for(int i=0;i<size(dv);i+=2) {
                    int row=get<0>(value_coords(i)),c=channel_base+get<1>(value_coords(i));
                    if(kb+row<end && c<VALUE_DIM) {
                        Storage x=Storage(dv(i)),y=Storage(dv(i+1));
                        uint32_t packed=uint32_t(x.x)|(uint32_t(y.x)<<16);
                        *reinterpret_cast<uint32_t*>(p.outputs[2].data+((kb+row)*HEADS+head)*VALUE_DIM+c)=packed;
                    }
                }
            } else {
                #pragma unroll
                for(int i=0;i<size(dv);++i) {
                    int row=get<0>(value_coords(i)),c=channel_base+get<1>(value_coords(i));
                    if(kb+row<end && c<VALUE_DIM)
                        p.outputs[2].data[((kb+row)*HEADS+head)*VALUE_DIM+c]=Storage(dv(i));
                }
            }
        }
        __syncthreads();
    }

#else
    for(int64_t qb=begin+int64_t(blockIdx.z)*32;qb<end;qb+=int64_t(gridDim.z)*32) {
        if constexpr(NEEDS&11) {
            // Softmax centering is dot(g, the saved forward output).
            for(int row=warp;row<32;row+=4) {
                float sum=0.f;
                if(qb+row<end) for(int c=lane;c<VALUE_DIM;c+=32)
                    sum+=attention_feature(p.inputs[4],qb+row,head,c,VALUE_DIM)*
                         attention_feature(p.inputs[5],qb+row,head,c,VALUE_DIM);
                sum=attention_sum(sum);
                if(lane==0) delta[row]=sum;
            }
        }
        __syncthreads();
        for(int64_t kb=begin;kb<end;kb+=32) {
            backward_pairs(p,gemm.score,probability,value,begin,end,qb,kb,segment,head);
            for(int i=threadIdx.x;i<1024;i+=128) {
                if constexpr(NEEDS&11) {
                    value[(i/32)*AttentionScoreStride+i%32]=probability[(i/32)*AttentionScoreStride+i%32]*(value[(i/32)*AttentionScoreStride+i%32]-delta[i/32]);
                }
            }
            __syncthreads();
            attention_output<0,1>(p,gemm.output,value,gemm.output.result,qb,kb,end,head,DIM,p.scale,kb==begin);
            attention_output<1,0,true>(p,gemm.output,value,gemm.output.result,qb,kb,end,head,DIM,p.scale);
            attention_output<2,4,true>(p,gemm.output,probability,gemm.output.result,qb,kb,end,head,VALUE_DIM);
            attention_bias_output<3,AttentionScoreStride>(p,value,segment,head,begin,qb,kb,end);
        }
    }
#endif
}

void attention_backward(std::vector<std::optional<at::Tensor>> inputs,at::Tensor rowptr,at::Tensor state,
    std::vector<std::optional<at::Tensor>> outputs,double scale,std::vector<std::optional<at::Tensor>> partials) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p=make_params(inputs,partials,rowptr,state,scale);
    if(p.groups) {
#if LOW_PRECISION && (NEEDS&9)
        attention_center_initialize_kernel<<<dim3(p.groups,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p);
#elif !LOW_PRECISION
        attention_initialize_kernel<4><<<dim3(p.groups,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p);
#endif
#if LOW_PRECISION
        p.outputs[1]=make_tensor(outputs[1]);
        p.outputs[2]=make_tensor(outputs[2]);
        constexpr int QueryChannels=(NEEDS&3) ? DIM:0;
        constexpr int ValueChannels=(NEEDS&4) ? VALUE_DIM:0;
        constexpr int Channels=QueryChannels>ValueChannels ? QueryChannels:ValueChannels;
        constexpr int Blocks=(Channels+255)/256;
        constexpr int Workers=4*(Blocks>0 ? Blocks:1);
#else
        constexpr int Workers=4;
#endif
        attention_backward_kernel<<<dim3(p.groups,HEADS,Workers),128,sizeof(AttentionBackwardShared),at::cuda::getCurrentCUDAStream()>>>(p);
#if LOW_PRECISION && (NEEDS&9)
        auto conversion=make_params(partials,{outputs[0],std::nullopt,std::nullopt,outputs[3]},rowptr,state,scale);
        attention_convert_kernel<4><<<dim3(p.groups,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(conversion);
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run",&attention_backward);
    m.def("configure",[]() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttentionBackwardShared)));
    });
}
"""


def attention_backward(inputs, rowptr, state, config, needs, output):
    results = tuple(torch.empty(value.shape, device=value.device, dtype=value.dtype)
                    if need and value is not None else None
                    for value, need in zip(inputs[:4], needs))
    if not any(value is not None and value.numel() for value in results):
        return results
    inputs = (*inputs, output)
    dim, value_dim, heads, scale, causal, bias_kind = config
    wanted = sum(1 << i for i, value in enumerate(results) if value is not None)
    if inputs[0].dtype != torch.float32 and wanted & 9:
        inputs = (*inputs, torch.empty(inputs[0].shape[:2], device=inputs[0].device, dtype=torch.float32))
    present = sum(1 << i for i, value in enumerate(inputs[:5]) if value is not None)
    compact = all(value.is_contiguous() for i, value in enumerate(inputs)
                  if value is not None and i not in (3, 8, 12))
    key = compact, dim, value_dim, heads, causal, bias_kind, wanted, present, inputs[0].device, inputs[0].dtype
    kernel = KERNELS.get(key)
    if kernel is None:
        defines = (f"#define DIM {dim}\n#define VALUE_DIM {value_dim}\n#define HEADS {heads}\n"
                   f"#define COMPACT {int(compact)}\n#define CAUSAL {int(causal)}\n#define BIAS_KIND {bias_kind}\n"
                   f"#define PRESENT {present}\n#define NEEDS {wanted}\n")
        if inputs[0].dtype == torch.float32:
            width = min(128, max(32, 32*((max(dim, value_dim)+31)//32)))
            defines += f"#define BACKWARD_WIDTH {width}\n#define ATTENTION_BACKWARD_SINGLE_STAGE 1\n"
        module = load_cuda_extension(
            "attention_backward", dtype_defines(inputs[0].dtype) + defines + CUDA_SOURCE
            + GEMM_SOURCE + ATTENTION_SOURCE + BIAS_SOURCE + BACKWARD_SOURCE,
            arch=get_cuda_arch(inputs[0].device), use_fast_math=False, minimal_headers=True,
        )
        with torch.cuda.device(inputs[0].device):
            module.configure()
        kernel = module.run
        KERNELS[key] = kernel
    partials = tuple(None if i in (1, 2) else
                     torch.empty_like(value, dtype=torch.float32)
                     if value is not None and value.dtype != torch.float32 else value
                     for i, value in enumerate(results)) if inputs[0].dtype != torch.float32 else results
    kernel(inputs, rowptr, state, results, scale, partials)
    return results
