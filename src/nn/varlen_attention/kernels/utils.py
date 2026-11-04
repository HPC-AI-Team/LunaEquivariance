"""Shared dtype-aware tensor addressing and FP32 accumulation for CSR attention.

Input slots follow interface.py: q, k, v, bias, grad_out, then incoming
backward gradients (rq, rk, rv, rb) and double-backward gradients
(sq, sk, sv, sb, sg).
Each output tuple follows its differentiated inputs.
"""

import torch


def dtype_defines(dtype):
    if dtype == torch.float32:
        return "#define LOW_PRECISION 0\nusing Storage = float;\n"
    storage = "at::Half" if dtype == torch.float16 else "at::BFloat16"
    mma = "cutlass::half_t" if dtype == torch.float16 else "cutlass::bfloat16_t"
    return ("#include <cutlass/arch/mma_sm80.h>\n#include <cutlass/numeric_types.h>\n"
            f"using LowMmaElement = {mma};\n#define LOW_PRECISION 1\n"
            f"#include <torch/extension.h>\nusing Storage = {storage};\n")


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/mma.h>
#include <cutlass/arch/mma_sm50.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <optional>
#include <vector>

using Element = float;
struct AttentionTensor {
    Storage* data = nullptr;
    bool fp32 = false;
    int64_t stride[4] = {};
    int64_t size[4] = {};

#if LOW_PRECISION
    struct Reference {
        void* pointer;
        bool fp32;
        __device__ __forceinline__ operator float() const {
            return fp32 ? *static_cast<float*>(pointer) : float(*static_cast<Storage*>(pointer));
        }
    };
    __device__ __forceinline__ Reference at(
        int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        int64_t offset = i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3];
        return {fp32 ? static_cast<void*>(reinterpret_cast<float*>(data) + offset) :
                       static_cast<void*>(data + offset), fp32};
    }
#else
    __device__ __forceinline__ Element& at(
        int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        return data[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
    }
#endif
    // Split reductions use a float workspace even when inputs are low precision.
    __device__ __forceinline__ float& fp32_at(int64_t i, int64_t j, int64_t k = 0, int64_t l = 0) const {
        return reinterpret_cast<float*>(data)[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
    }
};

AttentionTensor make_tensor(const std::optional<at::Tensor>& tensor) {
    AttentionTensor view;
    if (tensor.has_value()) {
        view.data = reinterpret_cast<Storage*>(tensor->data_ptr());
        view.fp32 = tensor->scalar_type() == at::kFloat;
        for (int axis = 0; axis < tensor->dim(); ++axis) {
            view.stride[axis] = tensor->stride(axis);
            view.size[axis] = tensor->size(axis);
        }
    }
    return view;
}

struct AttentionState {
    float* data;
    int64_t heads;
    __device__ __forceinline__ float at(int64_t row, int64_t head, int channel) const {
        return data[(row * heads + head) * 2 + channel];
    }
};

struct AttentionParams {
    AttentionTensor inputs[14], outputs[9];
    AttentionState state;
    const void* rowptr;
    bool csr64;
    int64_t groups, rowptr_stride;
    Element scale;

    __device__ __forceinline__ int64_t offset(int64_t group) const {
        return csr64 ? static_cast<const int64_t*>(rowptr)[group * rowptr_stride]
                     : static_cast<const int32_t*>(rowptr)[group * rowptr_stride];
    }
};

AttentionParams make_params(
    const std::vector<std::optional<at::Tensor>>& inputs,
    const std::vector<std::optional<at::Tensor>>& outputs,
    const at::Tensor& rowptr, const at::Tensor& state, double scale) {
    AttentionParams params{};
    for (size_t i = 0; i < inputs.size(); ++i) params.inputs[i] = make_tensor(inputs[i]);
    for (size_t i = 0; i < outputs.size(); ++i) params.outputs[i] = make_tensor(outputs[i]);
    params.state = {state.data_ptr<float>(), state.size(1)};
    params.rowptr = rowptr.data_ptr();
    params.csr64 = rowptr.scalar_type() == at::kLong;
    params.groups = rowptr.numel() - 1;
    params.rowptr_stride = rowptr.stride(0);
    params.scale = static_cast<Element>(scale);
    return params;
}
"""


GEMM_SOURCE = r"""
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/mma_singlestage.h>
#include <cutlass/arch/memory.h>

constexpr int AttentionThreads = 128;
#if !LOW_PRECISION && defined(ATTENTION_BACKWARD_SINGLE_STAGE)
constexpr int AttentionScoreStride=36;
#else
constexpr int AttentionScoreStride=32;
#endif
#if LOW_PRECISION
using GemmElement = LowMmaElement;
using GemmClass = cutlass::arch::OpClassTensorOp;
using GemmInstruction = cutlass::gemm::GemmShape<16,8,16>;
constexpr int GemmReductionTile = 64;
#else
using GemmElement = float;
using GemmClass = cutlass::arch::OpClassSimt;
using GemmInstruction = cutlass::gemm::GemmShape<1,1,1>;
constexpr int GemmReductionTile = 8;
#endif
#if !LOW_PRECISION && defined(ATTENTION_BACKWARD_SINGLE_STAGE)
// A scalar B map and lane grouping valid for every 32-column multiple.
template<int Columns>
struct AttentionNativeOutputCore : cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<32,64,8>,cutlass::gemm::GemmShape<16,32,8>,
    GemmInstruction,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
    float,cutlass::layout::RowMajor,GemmClass,2,cutlass::arch::OpMultiplyAdd> {
    using Shape=cutlass::gemm::GemmShape<32,Columns,8>;
    using WarpShape=cutlass::gemm::GemmShape<16,Columns/2,8>;
    static constexpr int ContiguousThreads=cutlass::gcd(Columns,AttentionThreads);
    static constexpr int StridedThreads=AttentionThreads/ContiguousThreads;
    struct IteratorThreadMapB : cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<ContiguousThreads,8>,AttentionThreads,1> {
        using Iterations=cutlass::layout::PitchLinearShape<Columns/ContiguousThreads,8/StridedThreads>;
        using Delta=cutlass::layout::PitchLinearShape<ContiguousThreads,StridedThreads>;
    };
    using SmemIteratorB=cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<8,Columns>,float,cutlass::layout::RowMajor,0,IteratorThreadMapB>;
    static constexpr int LaneN=cutlass::gcd(WarpShape::kN/8,128/cutlass::sizeof_bits<float>::value);
    using Policy=cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4,8>,
        cutlass::layout::RowMajorInterleaved<1>,cutlass::gemm::GemmShape<4,LaneN,1>>;
    using MmaWarpSimt=cutlass::gemm::warp::MmaSimt<WarpShape,float,cutlass::layout::ColumnMajor,
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,Policy>;
    using MmaPolicy=cutlass::gemm::threadblock::MmaPolicy<MmaWarpSimt,
        cutlass::MatrixShape<4,0>,cutlass::MatrixShape<0,0>,1>;
};
#endif
template<int Columns,bool RowMajorB=false>
struct AttentionGemmTraits {
#if defined(ATTENTION_BACKWARD_SINGLE_STAGE)
    static constexpr bool SingleStage=true;
#else
    static constexpr bool SingleStage=LOW_PRECISION;
#endif
    static constexpr int ReductionTile=LOW_PRECISION && SingleStage ? 32:GemmReductionTile;
    static constexpr int Stages=SingleStage ? 1:2;
    static constexpr int PartitionsK=LOW_PRECISION && RowMajorB && Columns==32 ? 2:1;
    static constexpr int WarpColumns=Columns/(2/PartitionsK);
#if !LOW_PRECISION && defined(ATTENTION_BACKWARD_SINGLE_STAGE)
    static constexpr int ResultStride=Columns+(RowMajorB ? 0:AttentionScoreStride-32);
#else
    static constexpr int ResultStride=Columns;
#endif
    using Shape = cutlass::gemm::GemmShape<32,Columns,ReductionTile>;
    using Instruction = typename std::conditional<PartitionsK==2,
        cutlass::gemm::GemmShape<16,8,8>,GemmInstruction>::type;
    using DefaultCore = cutlass::gemm::threadblock::DefaultMmaCore<
        Shape,cutlass::gemm::GemmShape<16,WarpColumns,ReductionTile/PartitionsK>,Instruction,
        GemmElement,cutlass::layout::RowMajor,GemmElement,
        typename std::conditional<RowMajorB,cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>::type,
        float,cutlass::layout::RowMajor,GemmClass,2,cutlass::arch::OpMultiplyAdd>;
#if !LOW_PRECISION && defined(ATTENTION_BACKWARD_SINGLE_STAGE)
    using Core=typename std::conditional<RowMajorB,AttentionNativeOutputCore<Columns>,DefaultCore>::type;
#else
    using Core=DefaultCore;
#endif
    using Base = cutlass::gemm::threadblock::MmaBase<Shape,typename Core::MmaPolicy,Stages>;
    using Operator = typename Core::MmaPolicy::Operator;
    using Accumulator = typename Operator::FragmentC;
};

template<int Columns,bool RowMajorB=false>
struct alignas(16) AttentionGemmTile {
    typename AttentionGemmTraits<Columns,RowMajorB>::Base::SharedStorage operands;
    float result[32*AttentionGemmTraits<Columns,RowMajorB>::ResultStride*AttentionGemmTraits<Columns,RowMajorB>::PartitionsK];
};
using AttentionGemmStorage = AttentionGemmTile<32>;

#if LOW_PRECISION
template<bool First,typename Producer,typename Fragment>
__device__ __forceinline__ auto attention_load8(Producer const& producer,Fragment& fragment,
    int index,int row,int column,int) -> decltype(producer.template load8<First>(fragment,index,row,column),void()) {
    producer.template load8<First>(fragment,index,row,column);
}
template<bool First,typename Producer,typename Fragment>
__device__ __forceinline__ void attention_load8(Producer const& producer,Fragment& fragment,
    int index,int row,int column,long) {
    #pragma unroll
    for(int i=0;i<8;++i) fragment[index+i]=GemmElement(producer(row+(First ? i:0),column+(First ? 0:i)));
}
#endif

// A and B are generated from feature tensors or live shared-memory probability
// tiles. CUTLASS owns prefetch, shared-memory buffering and the full GEMM loop.
template<typename ThreadMap,typename Producer,bool Left,bool RowMajorB,int ReductionTile=GemmReductionTile>
struct AttentionProducerIterator {
    using Element = GemmElement;
    using Fragment = cutlass::Array<Element,ThreadMap::Iterations::kCount*ThreadMap::kElementsPerAccess>;
    Producer producer;
    cutlass::PitchLinearCoord origin;
    int base=0,reduction;
    bool enabled=true;

    __device__ __forceinline__ AttentionProducerIterator(Producer producer_,int reduction_)
        :producer(producer_),origin(ThreadMap::initial_offset(threadIdx.x)),reduction(reduction_) {}
    __device__ __forceinline__ void clear_mask(bool clear) {enabled=enabled && !clear;}
    __device__ __forceinline__ AttentionProducerIterator& operator++() {base+=ReductionTile;return *this;}
    __device__ __forceinline__ void load(Fragment& fragment) const {
#if LOW_PRECISION
        fragment.clear();
        if(!enabled) return;
        #pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s)
            #pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c) {
                const int contiguous=origin.contiguous()+c*ThreadMap::Delta::kContiguous;
                const int strided=origin.strided()+s*ThreadMap::Delta::kStrided;
                const int k=base+((Left || !RowMajorB) ? contiguous:strided);
                const int outer=(Left || !RowMajorB) ? strided:contiguous;
                const int index=(s*ThreadMap::Iterations::kContiguous+c)*8;
                if(k>=reduction) continue;
                if(Left || !RowMajorB) {
                    if(k+7>=reduction) {
                        #pragma unroll
                        for(int e=0;e<8;++e) if(k+e<reduction)
                            fragment[index+e]=Element(Left ? producer(outer,k+e):producer(k+e,outer));
                        continue;
                    }
                }
                attention_load8<!Left && !RowMajorB>(producer,fragment,index,Left ? outer:k,Left ? k:outer,0);
            }
#else
        #pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s)
            #pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c)
                #pragma unroll
                for(int e=0;e<ThreadMap::kElementsPerAccess;++e) {
                    int contiguous=origin.contiguous()+c*ThreadMap::Delta::kContiguous+e;
                    int strided=origin.strided()+s*ThreadMap::Delta::kStrided;
                    int k=base+((Left || !RowMajorB) ? contiguous:strided);
                    int outer=(Left || !RowMajorB) ? strided:contiguous;
                    int index=(s*ThreadMap::Iterations::kContiguous+c)*ThreadMap::kElementsPerAccess+e;
                    float value=0.f;
                    if(enabled && k<reduction) {
                        if constexpr(Left) value=producer(outer,k);
                        else value=producer(k,outer);
                    }
                    fragment[index]=Element(value);
                }
#endif
    }
};

template<int Columns,bool RowMajorB,typename Left,typename Right>
__device__ __forceinline__ void attention_gemm(AttentionGemmTile<Columns,RowMajorB>& shared,float* output,
    int reduction,Left left,Right right,
    typename AttentionGemmTraits<Columns,RowMajorB>::Accumulator* previous=nullptr) {
    using Traits = AttentionGemmTraits<Columns,RowMajorB>;
    using Core = typename Traits::Core;
    using Operator = typename Core::MmaPolicy::Operator;
    using IteratorA = AttentionProducerIterator<typename Core::IteratorThreadMapA,Left,true,RowMajorB,Traits::ReductionTile>;
    using IteratorB = AttentionProducerIterator<typename Core::IteratorThreadMapB,Right,false,RowMajorB,Traits::ReductionTile>;
    using Pipelined = cutlass::gemm::threadblock::MmaPipelined<
        typename AttentionGemmTraits<Columns,RowMajorB>::Shape,IteratorA,typename Core::SmemIteratorA,
        IteratorB,typename Core::SmemIteratorB,float,cutlass::layout::RowMajor,
        typename Core::MmaPolicy>;
    using SingleStage = cutlass::gemm::threadblock::MmaSingleStage<
        typename Traits::Shape,IteratorA,typename Core::SmemIteratorA,
        IteratorB,typename Core::SmemIteratorB,float,cutlass::layout::RowMajor,
        typename Core::MmaPolicy>;
    using ThreadblockMma=typename std::conditional<Traits::SingleStage,SingleStage,Pipelined>::type;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    constexpr int PartitionsK=AttentionGemmTraits<Columns,RowMajorB>::PartitionsK;
    constexpr int WarpColumns=AttentionGemmTraits<Columns,RowMajorB>::WarpColumns;
    constexpr int WarpsMN=4/PartitionsK;
    const int partition=warp/WarpsMN,warp_mn=warp%WarpsMN;
    constexpr int ResultStride=Traits::ResultStride;
    const int tile_offset=(warp_mn%2)*16*ResultStride+(warp_mn/2)*WarpColumns;
    typename ThreadblockMma::FragmentC local;
    auto& accumulator=previous ? *previous:local;
    typename Operator::IteratorC iterator_c(
        {output+tile_offset,cutlass::layout::RowMajor(ResultStride)},lane);
    if(!previous) accumulator.clear();
    ThreadblockMma mma(shared.operands,threadIdx.x,warp,lane);
    mma((reduction+Traits::ReductionTile-1)/Traits::ReductionTile,accumulator,
        IteratorA(left,reduction),IteratorB(right,reduction),accumulator);
    if(previous) {
        __syncthreads();
        return;
    }
    if constexpr(PartitionsK==1) iterator_c.store(accumulator);
    else {
        typename Operator::IteratorC partial_c(
            {shared.result+partition*32*ResultStride+tile_offset,cutlass::layout::RowMajor(ResultStride)},lane);
        partial_c.store(accumulator);
        __syncthreads();
        for(int i=threadIdx.x;i<32*Columns;i+=128) {
            int index=(i/Columns)*ResultStride+i%Columns;
            output[index]=shared.result[index]+shared.result[32*ResultStride+index];
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float attention_sum(float value) {
    #pragma unroll
    for (int shift=16;shift;shift/=2)
        value += __shfl_xor_sync(0xffffffff,value,shift);
    return value;
}

"""


ATTENTION_SOURCE = r"""
template<int Count>
__device__ __forceinline__ void attention_initialize(AttentionParams const& p,int segment,int head,
    int64_t begin,int64_t end) {
    #pragma unroll
    for(int target=0;target<Count;++target) {
#if !LOW_PRECISION && defined(ATTENTION_BACKWARD_SINGLE_STAGE)
        // Query rows have one owner; their first key tile writes the sum.
        if(target==0) continue;
#endif
        if(!p.outputs[target].data) continue;
        if constexpr(Count>=5) {
            if(target==0 || target==4 || (Count==9 && target==5)) continue;
        }
        const auto& output=p.outputs[target];
        if(target==3 || target==8) {
            if constexpr(BIAS_KIND==2) {
                for(int64_t i=threadIdx.x;i<output.size[2]*output.size[3];i+=AttentionThreads)
                    output.fp32_at(segment,head,i/output.size[3],i%output.size[3])=0.f;
            } else if constexpr(BIAS_KIND==1) {
                for(int64_t row=begin+threadIdx.x;row<end;row+=AttentionThreads)
                    output.fp32_at(row,head)=0.f;
            }
        } else {
            const int channels=(target==2 || target==4 || target==7) ? VALUE_DIM:DIM;
            for(int64_t i=threadIdx.x;i<(end-begin)*channels;i+=AttentionThreads)
                reinterpret_cast<float*>(output.data)[((begin+i/channels)*HEADS+head)*channels+i%channels]=0.f;
        }
    }
    __syncthreads();
}

template<int Count>
__global__ void attention_initialize_kernel(const __grid_constant__ AttentionParams p) {
    const int segment=blockIdx.x,head=blockIdx.y;
    attention_initialize<Count>(p,segment,head,p.offset(segment),p.offset(segment+1));
}

template<int Count>
__global__ void attention_convert_kernel(const __grid_constant__ AttentionParams p) {
    const int64_t thread=(int64_t(blockIdx.x)*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
    const int64_t threads=int64_t(gridDim.x)*gridDim.y*blockDim.x;
    #pragma unroll
    for(int target=0;target<Count;++target) {
        if(!p.outputs[target].data || p.outputs[target].fp32) continue;
        const auto& output=p.outputs[target];
        const auto& input=p.inputs[target];
        const int64_t count=output.size[0]*output.stride[0];
        const auto* source=reinterpret_cast<float const*>(input.data);
        for(int64_t offset=thread*8;offset<count;offset+=threads*8) {
            if(offset+7<count) {
                alignas(16) cutlass::Array<float,8> values;
                reinterpret_cast<float4*>(&values)[0]=*reinterpret_cast<float4 const*>(source+offset);
                reinterpret_cast<float4*>(&values)[1]=*reinterpret_cast<float4 const*>(source+offset+4);
                auto converted=cutlass::NumericArrayConverter<GemmElement,float,8>{}(values);
                *reinterpret_cast<uint4*>(output.data+offset)=*reinterpret_cast<uint4 const*>(&converted);
            } else {
                for(int e=0;offset+e<count;++e) output.data[offset+e]=Storage(source[offset+e]);
            }
        }
    }
}

__device__ __forceinline__ float attention_feature(AttentionTensor const& source,
    int64_t row,int head,int channel,int channels) {
#if COMPACT
    return source.data[(row*HEADS+head)*channels+channel];
#else
    return source.data[row*source.stride[0]+head*source.stride[1]+channel*source.stride[2]];
#endif
}

#if LOW_PRECISION
template<bool Transpose=false>
struct AttentionGemmFeature {
    AttentionTensor const& source;
    int64_t begin,end;
    int head,channels,channel_base;
    bool aligned;
    __device__ __forceinline__ AttentionGemmFeature(AttentionTensor const& source_,int64_t begin_,
        int64_t end_,int head_,int channels_,int channel_base_=0)
        :source(source_),begin(begin_),end(end_),head(head_),channels(channels_),channel_base(channel_base_),
         aligned(channels_%8==0 && (reinterpret_cast<uintptr_t>(source_.data)&15)==0) {}
    __device__ __forceinline__ float operator()(int row,int column) const {
        const int64_t edge=begin+(Transpose ? column:row);
        const int channel=channel_base+(Transpose ? row:column);
        return edge<end && channel<channels ? attention_feature(source,edge,head,channel,channels):0.f;
    }
    template<bool First,typename Fragment>
    __device__ __forceinline__ void load8(Fragment& fragment,int index,int row,int column) const {
#if COMPACT
        if constexpr(First==Transpose) {
            const int64_t edge=begin+(Transpose ? column:row);
            const int channel=channel_base+(Transpose ? row:column);
            if(aligned && edge<end && channel+7<channels) {
                auto& packed=*reinterpret_cast<uint4*>(reinterpret_cast<char*>(&fragment)+index*sizeof(GemmElement));
                cutlass::arch::global_load<uint4,16>(packed,
                    source.data+(edge*HEADS+head)*channels+channel,true);
                return;
            }
        }
#endif
        #pragma unroll
        for(int i=0;i<8;++i) fragment[index+i]=GemmElement((*this)(row+(First ? i:0),column+(First ? 0:i)));
    }
};
#endif

template<int Input>
__device__ __forceinline__ float attention_bias(AttentionParams const& p,int segment,int head,
    int64_t begin,int64_t query,int64_t key) {
    if constexpr (!(PRESENT & (1<<Input)) || BIAS_KIND==0) return 0.f;
    if constexpr (BIAS_KIND==1) return p.inputs[Input].at(key,head);
    else return p.inputs[Input].at(segment,head,query-begin,key-begin);
}
"""

BIAS_SOURCE = r"""template<int Target,int Stride=32>
__device__ __forceinline__ void attention_bias_output(AttentionParams const& p,float const* coefficient,
    int segment,int head,int64_t begin,int64_t qb,int64_t kb,int64_t end) {
    if constexpr((NEEDS&(1<<Target)) && BIAS_KIND!=0) {
        if constexpr(BIAS_KIND==2) {
            for(int i=threadIdx.x;i<32*32;i+=AttentionThreads)
                if(qb+i/32<end && kb+i%32<end)
                    p.outputs[Target].fp32_at(segment,head,qb+i/32-begin,kb+i%32-begin)=coefficient[(i/32)*Stride+i%32];
        } else {
            for(int column=threadIdx.x;column<32;column+=AttentionThreads) if(kb+column<end) {
                float total=0.f;
                for(int row=0;row<32;++row) total+=coefficient[row*Stride+column];
                atomicAdd(reinterpret_cast<float*>(p.outputs[Target].data)+(kb+column)*HEADS+head,total);
            }
        }
        __syncthreads();
    }
}
"""

PACKED_GEMM_SOURCE = r"""
#include <cutlass/gemm/warp/mma_simt_tile_iterator.h>

namespace packed_pair {

struct alignas(16) FeatureElement {
    float value, first, second, padding0;

    __host__ __device__ __forceinline__ FeatureElement(float value_ = 0.f)
        : value(value_), first(0.f), second(0.f), padding0(0.f) {}
    __host__ __device__ __forceinline__ FeatureElement(float value_, float first_, float second_)
        : value(value_), first(first_), second(second_), padding0(0.f) {}
};

struct alignas(8) DoubleFeatureElement {
    float value,first;
    __host__ __device__ __forceinline__ DoubleFeatureElement(float scalar=0.f)
        :value(scalar),first(0.f) {}
    __host__ __device__ __forceinline__ DoubleFeatureElement(FeatureElement const& feature)
        :value(feature.value),first(feature.first) {}
};

// The value-side input has no first direction. Preserve that structural zero
// through CUTLASS's scalar conversion instead of loading a shared zero field.
struct alignas(8) ValueSecondElement {
    float value,second;
    __host__ __device__ __forceinline__ ValueSecondElement(float scalar=0.f)
        :value(scalar),second(0.f) {}
    __host__ __device__ __forceinline__ ValueSecondElement(FeatureElement const& feature)
        :value(feature.value),second(feature.second) {}
};

struct alignas(16) PairElement {
    float value, first, second, mixed;

    __host__ __device__ __forceinline__ PairElement(float value_ = 0.f)
        : value(value_), first(0.f), second(0.f), mixed(0.f) {}
    __host__ __device__ __forceinline__ PairElement(FeatureElement const& feature)
        : value(feature.value), first(feature.first), second(feature.second), mixed(0.f) {}
    __host__ __device__ __forceinline__ PairElement(DoubleFeatureElement const& feature)
        :value(feature.value),first(feature.first),second(0.f),mixed(0.f) {}
    __host__ __device__ __forceinline__ PairElement(ValueSecondElement const& feature)
        :value(feature.value),first(0.f),second(feature.second),mixed(0.f) {}
};

// ADL supplies only the scalar multiply-add used by CUTLASS's generic Mma.
struct Product {
    PairElement left, right;

    __host__ __device__ __forceinline__ PairElement operator+(PairElement const& accumulator) const {
        PairElement result;
        result.value = fmaf(left.value, right.value, accumulator.value);
        result.first = fmaf(left.first, right.value, fmaf(left.value, right.first, accumulator.first));
        result.second = fmaf(left.second, right.value, fmaf(left.value, right.second, accumulator.second));
        result.mixed = fmaf(left.first, right.second, fmaf(left.second, right.first, accumulator.mixed));
        return result;
    }
};

__host__ __device__ __forceinline__ Product operator*(PairElement const& left, PairElement const& right) {
    return {left, right};
}

static_assert(sizeof(FeatureElement) == 16 && sizeof(PairElement) == 16);
static_assert(alignof(FeatureElement) == 16 && alignof(PairElement) == 16);

// The accumulator only retains the two terminal gradient components. The
// other fields carry operands through CUTLASS's generic multiply_add casts.
struct alignas(16) Terminal4 {
    float value,first,second,mixed;
    __host__ __device__ __forceinline__ Terminal4(float scalar=0.f)
        :value(scalar),first(0.f),second(0.f),mixed(0.f) {}
    __host__ __device__ __forceinline__ Terminal4(FeatureElement const& feature)
        :value(feature.value),first(feature.first),second(feature.second),mixed(0.f) {}
    __host__ __device__ __forceinline__ Terminal4(PairElement const& pair)
        :value(pair.value),first(pair.first),second(pair.second),mixed(pair.mixed) {}
    __host__ __device__ __forceinline__ Terminal4(ValueSecondElement const& feature)
        :value(feature.value),first(0.f),second(feature.second),mixed(0.f) {}
};

struct TerminalProduct {
    Terminal4 left,right;
    __host__ __device__ __forceinline__ Terminal4 operator+(Terminal4 const& accumulator) const {
        Terminal4 result;
        result.second=fmaf(left.second,right.value,
                          fmaf(left.value,right.second,accumulator.second));
        result.mixed=fmaf(left.mixed,right.value,
                         fmaf(left.first,right.second,
                              fmaf(left.second,right.first,accumulator.mixed)));
        return result;
    }
};

__host__ __device__ __forceinline__ TerminalProduct operator*(Terminal4 left,Terminal4 right) {
    return {left,right};
}

// Rebind only Element to float. Keep the packed accumulator's exact Shape and
// Policy: the ordinary float GEMM uses a different per-lane fragment mapping.
template<int Stride,typename IteratorC, typename Fragment, typename Project>
__device__ __forceinline__ void store_plane(Fragment const& accumulator, float* output, Project project) {
    using FloatIterator = cutlass::gemm::warp::MmaSimtTileIterator<
        typename IteratorC::Shape, cutlass::gemm::Operand::kC, float,
        cutlass::layout::RowMajor, typename IteratorC::Policy>;
    static_assert(IteratorC::Shape::kRow == 16);
    static_assert(FloatIterator::Fragment::kElements == Fragment::kElements);
    typename FloatIterator::Fragment field;
    #pragma unroll
    for (int i = 0; i < Fragment::kElements; ++i)
        field[i] = project(accumulator[i]);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    FloatIterator iterator(
        {output + (warp % 2) * 16 * Stride + (warp / 2) * IteratorC::Shape::kColumn, cutlass::layout::RowMajor(Stride)}, lane);
    iterator.store(field);
}

#if LOW_PRECISION
constexpr int DoubleBackwardScore=0;
constexpr int DoubleBackwardOutput=1;
constexpr int TripleBackwardScore=2;
constexpr int TripleBackwardOutput=3;

// Named scalar CUTLASS fragments keep all derivative fields in registers.
template<typename Fragment> struct TwoFields {
    Fragment value,first;
    __device__ __forceinline__ void clear() {
        value.clear();
        first.clear();
    }
};
template<typename Fragment> struct ThreeFields {
    Fragment value,first,second;
    __device__ __forceinline__ void clear() {
        value.clear();
        first.clear();
        second.clear();
    }
};
template<typename Fragment> struct FourFields {
    Fragment value,first,second,mixed;
    __device__ __forceinline__ void clear() {
        value.clear();
        first.clear();
        second.clear();
        mixed.clear();
    }
};
template<typename Fragment> struct FirstField {
    Fragment first;
    __device__ __forceinline__ void clear() {first.clear();}
};
template<typename Fragment> struct TerminalFields {
    Fragment second,mixed;
    __device__ __forceinline__ void clear() {
        second.clear();
        mixed.clear();
    }
};

template<typename Iterator,typename Fragment_,int Fields>
struct CompoundWarpIterator {
    using Fragment=Fragment_;
    using TensorRef=typename Iterator::TensorRef;
    Iterator iterator;
    __device__ __forceinline__ CompoundWarpIterator(TensorRef const& ref,int lane):iterator(ref,lane) {}
    __device__ __forceinline__ void add_tile_offset(cutlass::MatrixCoord offset) {iterator.add_tile_offset(offset);}
    __device__ __forceinline__ void set_kgroup_index(int index) {iterator.set_kgroup_index(index);}
    __device__ __forceinline__ CompoundWarpIterator& operator++() {++iterator;return *this;}
    __device__ __forceinline__ void load(Fragment& fragment) const {
        iterator.load(fragment.value);
        iterator.load_with_pointer_offset(fragment.first,1024);
        if constexpr(Fields>=3) iterator.load_with_pointer_offset(fragment.second,2048);
        if constexpr(Fields==4) iterator.load_with_pointer_offset(fragment.mixed,3072);
    }
};

template<typename Iterator,typename Fragment_,int Fields>
struct CompoundSharedIterator {
    using Fragment=Fragment_;
    using TensorRef=typename Iterator::TensorRef;
    Iterator iterator;
    __device__ __forceinline__ CompoundSharedIterator(TensorRef ref,int thread):iterator(ref,thread) {}
    __device__ __forceinline__ void store(Fragment const& fragment) {
        iterator.store(fragment.value);
        iterator.store_with_pointer_offset(fragment.first,1024);
        if constexpr(Fields>=3) iterator.store_with_pointer_offset(fragment.second,2048);
        if constexpr(Fields==4) iterator.store_with_pointer_offset(fragment.mixed,3072);
    }
};

// Each library warp GEMM consumes the same loaded fields. Only these explicit
// derivative products differ from the scalar TensorOp operator.
template<typename WarpMma,int Kind>
struct CompoundWarpMma {
    using Shape=typename WarpMma::Shape;
    using Policy=typename WarpMma::Policy;
    using ElementA=typename WarpMma::ElementA;
    using ElementB=typename WarpMma::ElementB;
    using ElementC=float;
    using LayoutA=typename WarpMma::LayoutA;
    using LayoutB=typename WarpMma::LayoutB;
    using LayoutC=cutlass::layout::RowMajor;
    static constexpr int FieldsA=Kind<TripleBackwardScore ? 2:(Kind==TripleBackwardScore ? 3:4);
    static constexpr int FieldsB=Kind<TripleBackwardScore ? 2:3;
    using FragmentA=typename std::conditional<Kind<TripleBackwardScore,TwoFields<typename WarpMma::FragmentA>,
        typename std::conditional<Kind==TripleBackwardScore,ThreeFields<typename WarpMma::FragmentA>,
            FourFields<typename WarpMma::FragmentA>>::type>::type;
    using FragmentB=typename std::conditional<Kind<TripleBackwardScore,TwoFields<typename WarpMma::FragmentB>,
        ThreeFields<typename WarpMma::FragmentB>>::type;
    using FragmentC=typename std::conditional<Kind==DoubleBackwardScore,TwoFields<typename WarpMma::FragmentC>,
        typename std::conditional<Kind==DoubleBackwardOutput,FirstField<typename WarpMma::FragmentC>,
            typename std::conditional<Kind==TripleBackwardScore,FourFields<typename WarpMma::FragmentC>,
                TerminalFields<typename WarpMma::FragmentC>>::type>::type>::type;
    using IteratorA=CompoundWarpIterator<typename WarpMma::IteratorA,FragmentA,FieldsA>;
    using IteratorB=CompoundWarpIterator<typename WarpMma::IteratorB,FragmentB,FieldsB>;
    static constexpr auto kTransformA=cutlass::ComplexTransform::kNone;
    static constexpr auto kTransformB=cutlass::ComplexTransform::kNone;
    static_assert(std::is_same<typename WarpMma::FragmentA,typename WarpMma::TransformedFragmentA>::value);
    static_assert(std::is_same<typename WarpMma::FragmentB,typename WarpMma::TransformedFragmentB>::value);

    __device__ __forceinline__ void operator()(FragmentC& d,FragmentA const& a,
        FragmentB const& b,FragmentC const& c) const {
        d=c;
        WarpMma mma;
        if constexpr(Kind==DoubleBackwardScore || Kind==TripleBackwardScore) mma(d.value,a.value,b.value,d.value);
        if constexpr(Kind<TripleBackwardOutput) {
            mma(d.first,a.value,b.first,d.first);
            mma(d.first,a.first,b.value,d.first);
        }
        if constexpr(Kind>=TripleBackwardScore) {
            mma(d.second,a.value,b.second,d.second);
            mma(d.second,a.second,b.value,d.second);
            mma(d.mixed,a.first,b.second,d.mixed);
            mma(d.mixed,a.second,b.first,d.mixed);
            if constexpr(Kind==TripleBackwardOutput) mma(d.mixed,a.mixed,b.value,d.mixed);
        }
    }
};

template<int Field,bool First,typename Input,typename Fragment>
__device__ __forceinline__ auto compound_load8(Input const& input,Fragment& fragment,int index,int row,int column,int)
    -> decltype(input.template load_field8<Field,First>(fragment,index,row,column),void()) {
    input.template load_field8<Field,First>(fragment,index,row,column);
}
template<int Field,bool First,typename Input,typename Fragment>
__device__ __forceinline__ void compound_load8(Input const& input,Fragment& fragment,int index,int row,int column,long) {
    cutlass::Array<float,8> values;
    #pragma unroll
    for(int i=0;i<8;++i) values[i]=input.template field<Field>(row+(First ? i:0),column+(First ? 0:i));
    auto converted=cutlass::NumericArrayConverter<GemmElement,float,8>{}(values);
    #pragma unroll
    for(int i=0;i<8;++i) fragment[index+i]=GemmElement(converted[i]);
}
template<int Field,typename Input>
struct CompoundInputField {
    Input const& input;
    __device__ __forceinline__ float operator()(int row,int column) const {
        return input.template field<Field>(row,column);
    }
    template<bool First,typename Fragment>
    __device__ __forceinline__ void load8(Fragment& fragment,int index,int row,int column) const {
        compound_load8<Field,First>(input,fragment,index,row,column,0);
    }
};

template<typename ThreadMap,typename Input,typename Fragment_,int Fields,bool Left,bool RowMajorB>
struct CompoundProducerIterator {
    using Element=GemmElement;
    using Fragment=Fragment_;
    Input input;
    int base=0,reduction;
    bool enabled=true;
    __device__ __forceinline__ CompoundProducerIterator(Input input_,int reduction_):input(input_),reduction(reduction_) {}
    __device__ __forceinline__ void clear_mask(bool clear) {enabled=enabled && !clear;}
    __device__ __forceinline__ CompoundProducerIterator& operator++() {base+=32;return *this;}
    template<int Field,typename ScalarFragment>
    __device__ __forceinline__ void load_field(ScalarFragment& fragment) const {
        CompoundInputField<Field,Input> field{input};
        AttentionProducerIterator<ThreadMap,decltype(field),Left,RowMajorB,32> iterator(field,reduction);
        iterator.base=base;
        iterator.enabled=enabled;
        iterator.load(fragment);
    }
    __device__ __forceinline__ void load(Fragment& fragment) const {
        load_field<0>(fragment.value);
        load_field<1>(fragment.first);
        if constexpr(Fields>=3) load_field<2>(fragment.second);
        if constexpr(Fields==4) load_field<3>(fragment.mixed);
    }
};

template<int Kind>
struct CompoundTraits {
    static constexpr bool RowMajorB=Kind==DoubleBackwardOutput || Kind==TripleBackwardOutput;
    using Matrix=AttentionGemmTraits<32,RowMajorB>;
    using WarpMma=typename Matrix::Operator;
    using Core=typename Matrix::Core;
    using Operator=CompoundWarpMma<WarpMma,Kind>;
    using Policy=cutlass::gemm::threadblock::MmaPolicy<Operator,
        cutlass::MatrixShape<(Operator::FieldsA-1)*32,0>,
        cutlass::MatrixShape<RowMajorB ? (Operator::FieldsB-1)*32:0,
            RowMajorB ? 0:(Operator::FieldsB-1)*32>,Matrix::PartitionsK>;
    using Base=cutlass::gemm::threadblock::MmaBase<typename Matrix::Shape,Policy,1>;
    using GlobalA=typename std::conditional<Operator::FieldsA==2,TwoFields<typename Core::SmemIteratorA::Fragment>,
        typename std::conditional<Operator::FieldsA==3,ThreeFields<typename Core::SmemIteratorA::Fragment>,
            FourFields<typename Core::SmemIteratorA::Fragment>>::type>::type;
    using GlobalB=typename std::conditional<Operator::FieldsB==2,TwoFields<typename Core::SmemIteratorB::Fragment>,
        ThreeFields<typename Core::SmemIteratorB::Fragment>>::type;
    using SharedA=CompoundSharedIterator<typename Core::SmemIteratorA,GlobalA,Operator::FieldsA>;
    using SharedB=CompoundSharedIterator<typename Core::SmemIteratorB,GlobalB,Operator::FieldsB>;
};

template<typename A,typename B,typename C,bool RowMajorB>
struct LowStorage {
    static constexpr bool Double=std::is_same<A,DoubleFeatureElement>::value;
    static constexpr int Kind=Double
        ? (RowMajorB ? DoubleBackwardOutput:DoubleBackwardScore)
        : (RowMajorB ? TripleBackwardOutput:TripleBackwardScore);
    using T=CompoundTraits<Kind>;
    static constexpr int RetainedElements=RowMajorB
        ? T::Base::SharedStorage::ShapeA::kCount*sizeof(GemmElement)/sizeof(float):0;
    static constexpr int ResultStride=RowMajorB ? 40:32;
    union {
        typename T::Base::SharedStorage operands;
        // Preserve the ready coefficient planes while the terminal epilogue
        // reuses the right operand region for partition sums and outputs.
        float result[RetainedElements+2*32*ResultStride];
    };
};

template<int Stride,typename T,typename Fragment>
__device__ __forceinline__ void compound_store(Fragment const& field,float* output) {
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    typename T::WarpMma::IteratorC iterator({output+(warp%2)*16*Stride+(warp/2)*16,
        cutlass::layout::RowMajor(Stride)},lane);
    iterator.store(field);
}
template<typename T,typename Shared,typename Fragment>
__device__ __forceinline__ void compound_reduce(Shared& shared,Fragment const& field,float (&output)[8]) {
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    float* partial=shared.result+Shared::RetainedElements;
    constexpr int Stride=Shared::ResultStride;
    typename T::WarpMma::IteratorC iterator({partial+(warp/2)*32*Stride+(warp%2)*16*Stride,
        cutlass::layout::RowMajor(Stride)},lane);
    iterator.store(field);
    __syncthreads();
    #pragma unroll
    for(int j=0;j<8;++j) {
        const int i=threadIdx.x+j*128;
        const int pos=(i/32)*Stride+i%32;
        output[j]=partial[pos]+partial[32*Stride+pos];
    }
    __syncthreads();
}

template<typename Fragment_>
struct ReadyCoefficientIterator {
    using Element=GemmElement;
    using Fragment=Fragment_;
    template<typename Input>
    __device__ __forceinline__ ReadyCoefficientIterator(Input const&,int) {}
    __device__ __forceinline__ void clear_mask(bool) {}
    __device__ __forceinline__ ReadyCoefficientIterator& operator++() {return *this;}
    __device__ __forceinline__ void load(Fragment&) const {}
};
template<typename Iterator>
struct ReadyCoefficientStore {
    using Fragment=typename Iterator::Fragment;
    using TensorRef=typename Iterator::TensorRef;
    __device__ __forceinline__ ReadyCoefficientStore(TensorRef,int) {}
    __device__ __forceinline__ void store(Fragment const&) {}
};

template<typename Shared,typename Left>
__device__ __forceinline__ void prepare_coefficients(Shared& shared,Left const& left,int reduction) {
    using T=typename Shared::T;
    using Iterator=CompoundProducerIterator<typename T::Core::IteratorThreadMapA,Left,
        typename T::GlobalA,T::Operator::FieldsA,true,true>;
    typename Iterator::Fragment fragment;
    Iterator(left,reduction).load(fragment);
    typename T::SharedA store(shared.operands.operand_A_ref(),threadIdx.x);
    store.store(fragment);
    __syncthreads();
}

template<int Stride,typename A,typename B,typename C,bool RowMajorB,typename Left,typename Right>
__device__ __forceinline__ void low_gemm(LowStorage<A,B,C,RowMajorB>& shared,
    float* value,float* first,float* second,float* mixed,int reduction,Left left,Right right) {
    using Storage=LowStorage<A,B,C,RowMajorB>;
    using T=typename Storage::T;
    using Core=typename T::Core;
    constexpr bool terminal=Storage::Kind==DoubleBackwardOutput || Storage::Kind==TripleBackwardOutput;
    using IteratorA=typename std::conditional<terminal,ReadyCoefficientIterator<typename T::GlobalA>,
        CompoundProducerIterator<typename Core::IteratorThreadMapA,Left,
            typename T::GlobalA,T::Operator::FieldsA,true,RowMajorB>>::type;
    using SharedA=typename std::conditional<terminal,ReadyCoefficientStore<typename T::SharedA>,typename T::SharedA>::type;
    using IteratorB=CompoundProducerIterator<typename Core::IteratorThreadMapB,Right,
        typename T::GlobalB,T::Operator::FieldsB,false,RowMajorB>;
    using Mma=cutlass::gemm::threadblock::MmaSingleStage<typename T::Matrix::Shape,
        IteratorA,SharedA,IteratorB,typename T::SharedB,float,cutlass::layout::RowMajor,typename T::Policy>;
    typename Mma::FragmentC accumulator;
    accumulator.clear();
    Mma mma(shared.operands,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    mma(terminal ? 1:(reduction+31)/32,accumulator,IteratorA(left,reduction),IteratorB(right,reduction),accumulator);
    __syncthreads();
    if constexpr(Storage::Kind==DoubleBackwardScore || Storage::Kind==TripleBackwardScore) {
        if(value) compound_store<Stride,T>(accumulator.value,value);
        if(first) compound_store<Stride,T>(accumulator.first,first);
        if constexpr(Storage::Kind==TripleBackwardScore) {
            if(second) compound_store<Stride,T>(accumulator.second,second);
            if(mixed) compound_store<Stride,T>(accumulator.mixed,mixed);
        }
    } else if constexpr(Storage::Kind==DoubleBackwardOutput) {
        float field[8];
        compound_reduce<T>(shared,accumulator.first,field);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int i=threadIdx.x+j*128;
            if(first) first[(i/32)*Stride+i%32]=field[j];
        }
    } else {
        float second_field[8],mixed_field[8];
        if(second) compound_reduce<T>(shared,accumulator.second,second_field);
        if(mixed) compound_reduce<T>(shared,accumulator.mixed,mixed_field);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int i=threadIdx.x+j*128,pos=(i/32)*Stride+i%32;
            if(second) second[pos]=second_field[j];
            if(mixed) mixed[pos]=mixed_field[j];
        }
    }
    __syncthreads();
}

#endif

template<typename A,typename B,typename C,bool RowMajorB=false,int Columns=32>
struct Traits {
    static constexpr int Stages=RowMajorB && Columns>32 ? 1:2;
    using Shape=cutlass::gemm::GemmShape<32,Columns,8>;
    using Core=cutlass::gemm::threadblock::DefaultMmaCore<
        Shape,cutlass::gemm::GemmShape<16,Columns/2,8>,cutlass::gemm::GemmShape<1,1,1>,
        A,cutlass::layout::RowMajor,B,
        typename std::conditional<RowMajorB,cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>::type,
        C,cutlass::layout::RowMajor,cutlass::arch::OpClassSimt,2,cutlass::arch::OpMultiplyAdd>;
    using Base=cutlass::gemm::threadblock::MmaBase<Shape,typename Core::MmaPolicy,Stages>;
    using Operator=typename Core::MmaPolicy::Operator;
};

template<typename A,typename B,typename C,bool RowMajorB=false,int Columns=32>
struct alignas(16) GemmStorage
#if LOW_PRECISION
    : LowStorage<A,B,C,RowMajorB>
#endif
{
#if !LOW_PRECISION
    typename Traits<A,B,C,RowMajorB,Columns>::Base::SharedStorage operands;
#endif
};

using Storage=GemmStorage<FeatureElement,FeatureElement,PairElement>;
using DoubleStorage=GemmStorage<DoubleFeatureElement,DoubleFeatureElement,PairElement>;

template<typename ThreadMap,typename Producer,typename Element_,bool Left,bool RowMajorB>
struct ProducerIterator {
    using Element=Element_;
    using Fragment=cutlass::Array<Element,ThreadMap::Iterations::kCount*ThreadMap::kElementsPerAccess>;
    Producer producer;
    cutlass::PitchLinearCoord origin;
    int base=0,reduction;
    bool enabled=true;

    __device__ __forceinline__ ProducerIterator(Producer producer_,int reduction_)
        :producer(producer_),origin(ThreadMap::initial_offset(threadIdx.x)),reduction(reduction_) {}
    __device__ __forceinline__ void clear_mask(bool clear) {enabled=enabled && !clear;}
    __device__ __forceinline__ ProducerIterator& operator++() {base+=8;return *this;}
    __device__ __forceinline__ void load(Fragment& fragment) const {
        #pragma unroll
        for(int s=0;s<ThreadMap::Iterations::kStrided;++s)
            #pragma unroll
            for(int c=0;c<ThreadMap::Iterations::kContiguous;++c)
                #pragma unroll
                for(int e=0;e<ThreadMap::kElementsPerAccess;++e) {
                    int contiguous=origin.contiguous()+c*ThreadMap::Delta::kContiguous+e;
                    int strided=origin.strided()+s*ThreadMap::Delta::kStrided;
                    int k=base+((Left || !RowMajorB) ? contiguous:strided);
                    int outer=(Left || !RowMajorB) ? strided:contiguous;
                    int index=(s*ThreadMap::Iterations::kContiguous+c)*ThreadMap::kElementsPerAccess+e;
                    Element value;
                    if(enabled && k<reduction) {
                        if constexpr(Left) value=producer(outer,k);
                        else value=producer(k,outer);
                    }
                    fragment[index]=value;
                }
    }
};

// CUTLASS owns all prefetch, operand buffering, warp MMA, and the K loop.
template<int Stride=32,typename A,typename B,typename C,bool RowMajorB,int Columns,typename Left,typename Right>
__device__ __forceinline__ void gemm(GemmStorage<A,B,C,RowMajorB,Columns>& shared,
    float* value,float* first,float* second,float* mixed,int reduction,Left left,Right right) {
#if LOW_PRECISION
    low_gemm<Stride>(static_cast<LowStorage<A,B,C,RowMajorB>&>(shared),
        value,first,second,mixed,reduction,left,right);
#else
    using T=Traits<A,B,C,RowMajorB,Columns>;
    using Core=typename T::Core;
    using Operator=typename T::Operator;
    using IteratorA=ProducerIterator<typename Core::IteratorThreadMapA,Left,A,true,RowMajorB>;
    using IteratorB=ProducerIterator<typename Core::IteratorThreadMapB,Right,B,false,RowMajorB>;
    using Pipelined=cutlass::gemm::threadblock::MmaPipelined<
        typename T::Shape,IteratorA,typename Core::SmemIteratorA,IteratorB,typename Core::SmemIteratorB,
        C,cutlass::layout::RowMajor,typename Core::MmaPolicy>;
    using SingleStage=cutlass::gemm::threadblock::MmaSingleStage<
        typename T::Shape,IteratorA,typename Core::SmemIteratorA,IteratorB,typename Core::SmemIteratorB,
        C,cutlass::layout::RowMajor,typename Core::MmaPolicy>;
    using ThreadblockMma=typename std::conditional<T::Stages==1,SingleStage,Pipelined>::type;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    typename ThreadblockMma::FragmentC accumulator;
    accumulator.clear();
    ThreadblockMma mma(shared.operands,threadIdx.x,warp,lane);
    mma((reduction+7)/8,accumulator,IteratorA(left,reduction),IteratorB(right,reduction),accumulator);
    // An epilogue may reuse operand storage after every warp has consumed it.
    __syncthreads();
    using IteratorC=typename Operator::IteratorC;
    if(value) store_plane<Stride,IteratorC>(accumulator,value,[](C const& x) {return x.value;});
    if(first) store_plane<Stride,IteratorC>(accumulator,first,[](C const& x) {return x.first;});
    if(second) store_plane<Stride,IteratorC>(accumulator,second,[](C const& x) {return x.second;});
    if(mixed) store_plane<Stride,IteratorC>(accumulator,mixed,[](C const& x) {return x.mixed;});
    __syncthreads();
#endif
}

} // namespace packed_pair

template<int Value,int First,int Second>
__device__ __forceinline__ packed_pair::FeatureElement attention_packed_feature(
    AttentionParams const& p,int64_t row,int64_t end,int head,int channel,int channels) {
    packed_pair::FeatureElement result;
    if(row<end) {
        if constexpr(Value>=0 && (PRESENT&(1<<Value)))
            result.value=attention_feature(p.inputs[Value],row,head,channel,channels);
        if constexpr(First>=0 && (PRESENT&(1<<First)))
            result.first=attention_feature(p.inputs[First],row,head,channel,channels);
        if constexpr(Second>=0 && (PRESENT&(1<<Second)))
            result.second=attention_feature(p.inputs[Second],row,head,channel,channels);
    }
    return result;
}

#if LOW_PRECISION
template<int Value,int First,int Second,bool Transpose,int Channels>
struct AttentionPackedFeature {
    AttentionParams const& p;
    int64_t begin,end;
    int head,channel_base;

    template<int Field>
    __device__ __forceinline__ float field(int row,int column) const {
        constexpr int Slot=Field==0 ? Value:(Field==1 ? First:Second);
        if constexpr(Slot>=0 && (PRESENT&(1u<<Slot))) {
            const int64_t edge=begin+(Transpose ? column:row);
            const int channel=channel_base+(Transpose ? row:column);
            if(edge<end && channel<Channels)
                return attention_feature(p.inputs[Slot],edge,head,channel,Channels);
        }
        return 0.f;
    }

    template<int Field,bool Left,typename Fragment>
    __device__ __forceinline__ void load_field8(Fragment& fragment,int index,int row,int column) const {
        constexpr int Slot=Field==0 ? Value:(Field==1 ? First:Second);
        if constexpr(Slot>=0 && (PRESENT&(1u<<Slot))) {
#if COMPACT
            if constexpr(Left==Transpose && Channels%8==0) {
                const int64_t edge=begin+(Transpose ? column:row);
                const int channel=channel_base+(Transpose ? row:column);
                auto data=p.inputs[Slot].data;
                if(edge<end && channel+7<Channels && (reinterpret_cast<uintptr_t>(data)&15)==0) {
                    auto& packed=*reinterpret_cast<uint4*>(reinterpret_cast<char*>(&fragment)+index*sizeof(GemmElement));
                    cutlass::arch::global_load<uint4,16>(packed,data+(edge*HEADS+head)*Channels+channel,true);
                    return;
                }
            }
#endif
            #pragma unroll
            for(int i=0;i<8;++i)
                fragment[index+i]=GemmElement(field<Field>(row+(Left ? i:0),column+(Left ? 0:i)));
        } else {
            #pragma unroll
            for(int i=0;i<8;++i) fragment[index+i]=GemmElement(0.f);
        }
    }
};

template<bool Transpose,int Stride>
struct AttentionPackedCoefficient {
    float const* data[4];
    template<int Field>
    __device__ __forceinline__ float field(int row,int column) const {
        const int index=Transpose ? column*Stride+row:row*Stride+column;
        return data[Field][index];
    }
};

#endif

"""
