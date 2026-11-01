"""Mixed backward directions through all inputs and both output adjoints."""

from functools import lru_cache

import torch

from .backward import BACKWARD_SOURCE
from .utils import finish_gradients, input_layout, load_kernel


GRADIENT_SOURCE = r"""
#define PACKED_GRADIENTS (!EDGE_GRAD)
constexpr unsigned GradientComponents=12u;
// dS_rs includes all four product-rule terms. CUTLASS then contracts
// dQ_rs = dS_rs K + dS_r K_s + dS_s K_r (and the transposed dK terms).
// G_rs also includes envelope_r * node_grad_s in the value-gradient path.
template<int N>
__device__ __forceinline__ Differential<N> softmax_gradient(Differential<N> probability,Differential<N> dot,Differential<N> mean) {
    Differential<N> value;
    float centered[4];
    #pragma unroll
    for(int i=0;i<4;++i) centered[i]=dot.x[i]-mean.x[i];
    value.x[0]=probability.x[0]*centered[0];
    value.x[1]=fmaf(probability.x[1],centered[0],probability.x[0]*centered[1]);
    value.x[2]=fmaf(probability.x[2],centered[0],probability.x[0]*centered[2]);
    value.x[3]=fmaf(probability.x[3],centered[0],fmaf(probability.x[1],centered[2],
                    fmaf(probability.x[2],centered[1],probability.x[0]*centered[3])));
    return value;
}
// Mixed components include the third primal derivative and the incoming
// node/edge-adjoint direction. The second component differentiates the first
// incoming direction. Both original output adjoints receive mixed outputs.
template<int N>
__device__ __forceinline__ void gradient(Params const& p,int slot,Differential<N> value,
    int64_t i,int64_t j=0,int64_t k=0,int64_t l=0) {
    if(NEEDS&(1u<<slot)) {
        if(slot==4 || (QUERY_WORKERS>1 && (slot==1 || slot==2 || (slot==3 && BIAS_KIND==1)))) p.out[slot].atomic_add(p.out[slot].index(i,j,k,l),value.x[3]);
        else p.out[slot].accumulate(p.out[slot].index(i,j,k,l),value.x[3]);
    }
    if(NEEDS&(1u<<(7+slot))) {
        if(slot==4 || (QUERY_WORKERS>1 && (slot==1 || slot==2 || (slot==3 && BIAS_KIND==1)))) p.out[7+slot].atomic_add(p.out[7+slot].index(i,j,k,l),value.x[2]);
        else p.out[7+slot].accumulate(p.out[7+slot].index(i,j,k,l),value.x[2]);
    }
}
template<int N>
__device__ __forceinline__ void upstream_gradient(Params const& p,int slot,Differential<N> value,int64_t i,int h,int c) {
    if(NEEDS&(1u<<slot)) {
        if(QUERY_WORKERS>1 && slot==5) p.out[slot].atomic_add(p.out[slot].index(i,h,c),value.x[3]);
        else p.out[slot].accumulate(p.out[slot].index(i,h,c),value.x[3]);
    }
}
"""


PACKED_SOURCE = r"""
#if PACKED_GRADIENTS
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/warp/mma_simt_tile_iterator.h>

namespace packed_pair {

template<typename A,typename B,typename C,bool RowMajorB=false>
struct Traits {
    using Shape=cutlass::gemm::GemmShape<32,32,8>;
    using Core=cutlass::gemm::threadblock::DefaultMmaCore<
        Shape,cutlass::gemm::GemmShape<16,16,8>,cutlass::gemm::GemmShape<1,1,1>,
        A,cutlass::layout::RowMajor,B,
        typename std::conditional<RowMajorB,cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>::type,
        C,cutlass::layout::RowMajor,cutlass::arch::OpClassSimt,2,cutlass::arch::OpMultiplyAdd>;
    using Base=cutlass::gemm::threadblock::MmaBase<Shape,typename Core::MmaPolicy,2>;
    using Operator=typename Core::MmaPolicy::Operator;
};

template<typename A,typename B,typename C,bool RowMajorB=false>
struct alignas(16) GemmStorage {
    typename Traits<A,B,C,RowMajorB>::Base::SharedStorage operands;
};

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

template<typename IteratorC, typename Fragment, typename Project>
__device__ __forceinline__ void store_plane(Fragment const& accumulator, float* output, Project project) {
    using FloatIterator = cutlass::gemm::warp::MmaSimtTileIterator<
        typename IteratorC::Shape, cutlass::gemm::Operand::kC, float,
        cutlass::layout::RowMajor, typename IteratorC::Policy>;
    static_assert(IteratorC::Shape::kRow == 16 && IteratorC::Shape::kColumn == 16);
    static_assert(FloatIterator::Fragment::kElements == Fragment::kElements);
    typename FloatIterator::Fragment field;
    #pragma unroll
    for (int i = 0; i < Fragment::kElements; ++i)
        field[i] = project(accumulator[i]);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    FloatIterator iterator(
        {output + (warp % 2) * 16 * 32 + (warp / 2) * 16, cutlass::layout::RowMajor(32)}, lane);
    iterator.store(field);
}


// The library owns every operand load and all K-loop and warp-MMA iterations.
template<typename A,typename B,typename C,bool RowMajorB,typename Left,typename Right>
__device__ __forceinline__ void multiply(GemmStorage<A,B,C,RowMajorB>& shared,
    typename Traits<A,B,C,RowMajorB>::Operator::FragmentC& accumulator,
    int reduction,Left left,Right right) {
    using T=Traits<A,B,C,RowMajorB>;
    using Core=typename T::Core;
    using IteratorA=ProducerIterator<typename Core::IteratorThreadMapA,Left,A,true,RowMajorB>;
    using IteratorB=ProducerIterator<typename Core::IteratorThreadMapB,Right,B,false,RowMajorB>;
    using ThreadblockMma=cutlass::gemm::threadblock::MmaPipelined<
        typename T::Shape,IteratorA,typename Core::SmemIteratorA,IteratorB,typename Core::SmemIteratorB,
        C,cutlass::layout::RowMajor,typename Core::MmaPolicy>;
    accumulator.clear();
    ThreadblockMma mma(shared.operands,threadIdx.x,threadIdx.x/32,threadIdx.x%32);
    mma((reduction+7)/8,accumulator,IteratorA(left,reduction),IteratorB(right,reduction),accumulator);
}

// Planes may overwrite operands after the complete mainloop has returned.
template<typename A,typename B,typename C,bool RowMajorB,typename Left,typename Right>
__device__ __forceinline__ void gemm_planes(GemmStorage<A,B,C,RowMajorB>& shared,
    float* value,float* first,float* second,float* mixed,int reduction,Left left,Right right) {
    using Operator=typename Traits<A,B,C,RowMajorB>::Operator;
    typename Operator::FragmentC accumulator;
    multiply(shared,accumulator,reduction,left,right);
    __syncthreads();
    using IteratorC=typename Operator::IteratorC;
    if(value) store_plane<IteratorC>(accumulator,value,[](C const& x) {return x.value;});
    if(first) store_plane<IteratorC>(accumulator,first,[](C const& x) {return x.first;});
    if(second) store_plane<IteratorC>(accumulator,second,[](C const& x) {return x.second;});
    if(mixed) store_plane<IteratorC>(accumulator,mixed,[](C const& x) {return x.mixed;});
    __syncthreads();
}

struct alignas(16) FeatureElement {
    float value, first, second, padding0;

    __host__ __device__ __forceinline__ FeatureElement(float value_ = 0.f)
        : value(value_), first(0.f), second(0.f), padding0(0.f) {}
    __host__ __device__ __forceinline__ FeatureElement(float value_, float first_, float second_)
        : value(value_), first(first_), second(second_), padding0(0.f) {}
};

struct alignas(16) PairElement {
    float value, first, second, mixed;

    __host__ __device__ __forceinline__ PairElement(float value_ = 0.f)
        : value(value_), first(0.f), second(0.f), mixed(0.f) {}
    __host__ __device__ __forceinline__ PairElement(FeatureElement const& feature)
        : value(feature.value), first(feature.first), second(feature.second), mixed(0.f) {}
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

} // namespace packed_pair

using PackedScoreStorage=packed_pair::GemmStorage<packed_pair::FeatureElement,
    packed_pair::FeatureElement,packed_pair::PairElement>;
using PackedOutputStorage=packed_pair::GemmStorage<packed_pair::PairElement,
    packed_pair::FeatureElement,packed_pair::Terminal4,true>;


namespace packed_pair {
template<int N,bool LeftMixed=true,bool RightMixed=true,unsigned Components=(1u<<N)-1,
    typename Left,typename Right>
__device__ __forceinline__ void differential_gemm(PackedScoreStorage& shared,float (*output)[1024],
    int reduction,Left left,Right right) {
    static_assert(N==4 && !LeftMixed && !RightMixed && Components==15u);
    packed_pair::gemm_planes(shared,output[0],output[1],output[2],output[3],reduction,
        [&](int row,int k) {
            auto x=left(row,k);
            return packed_pair::FeatureElement(x.x[0],x.x[1],x.x[2]);
        },
        [&](int k,int column) {
            auto x=right(k,column);
            return packed_pair::FeatureElement(x.x[0],x.x[1],x.x[2]);
        });
}

template<int N,bool LeftMixed=true,bool RightMixed=true,unsigned Components=(1u<<N)-1,
    typename Left,typename Right>
__device__ __forceinline__ void differential_gemm(PackedOutputStorage& shared,float (*output)[1024],
    int reduction,Left left,Right right) {
    static_assert(N==4 && LeftMixed && !RightMixed && Components==12u);
    packed_pair::gemm_planes(shared,nullptr,nullptr,output[0],output[1],reduction,
        [&](int row,int k) {
            auto x=left(row,k);
            packed_pair::PairElement result;
            result.value=x.x[0];
            result.first=x.x[1];
            result.second=x.x[2];
            result.mixed=x.x[3];
            return result;
        },
        [&](int k,int column) {
            auto x=right(k,column);
            return packed_pair::FeatureElement(x.x[0],x.x[1],x.x[2]);
        });
}


}

#endif
"""


SOURCE = r"""
void run(std::vector<std::optional<at::Tensor>> inputs,at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,double scale
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
         ,at::Tensor summary
#endif
         ) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if(csr.numel()>1) {
        constexpr int bytes=sizeof(BackwardStorage<4>);
        static thread_local int configured_device=-1;
        int device=inputs[0]->get_device();
        if(configured_device!=device) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(backward_kernel<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
            configured_device=device;
        }
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
        auto p=params(inputs,outputs,csr,scale);
        auto projected=view(summary);
        node_projection_kernel<4><<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p,projected);
        backward_kernel<4><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,bytes,at::cuda::getCurrentCUDAStream()>>>(p,projected);
        if constexpr(NEEDS&548u)
            node_value_kernel<4><<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p,projected);
#else
        backward_kernel<4><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,bytes,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
"""


@lru_cache(maxsize=128)
def get_triple_backward_kernel(config, device, dtype, needed, edge_grad, layout, query_workers):
    defines = (
        f"#define QUERY_WORKERS {query_workers}\n"
        f"#define NEEDS {needed}u\n"
        f"#define EDGE_GRAD {int(edge_grad)}\n"
    )
    if dtype == torch.float32 and not edge_grad:
        defines += "#define CONV_NATIVE_HIGHER_SUMMARY 1\n"
    source = (
        defines
        + GRADIENT_SOURCE
        + PACKED_SOURCE
        + BACKWARD_SOURCE
        + SOURCE
    )
    return load_kernel("attention_conv_triple_backward", source, config, device.index, dtype, layout).run


def attention_conv_triple_backward(inputs, csr, config, needs):
    native_node = inputs[0].dtype == torch.float32 and inputs[6] is None
    outputs = tuple(
        (torch.empty if native_node and i in (2, 5, 9) else torch.zeros)(
            x.shape, device=x.device, dtype=torch.float32
        ) if x is not None and need else None
        for i, (x, need) in enumerate(zip(inputs[:12], needs))
    )
    needed = sum(1 << i for i, output in enumerate(outputs) if output is not None)
    query_workers = 4 if inputs[0].shape[0] > 32 * (csr.numel() - 1) else 1
    kernel = get_triple_backward_kernel(
        config, inputs[0].device, inputs[0].dtype, needed,
        inputs[6] is not None, input_layout(inputs), query_workers,
    )
    if native_node:
        summary = torch.empty(
            (*inputs[0].shape[:2], 8), device=inputs[0].device, dtype=torch.float32
        )
        kernel(inputs, csr, outputs, config[3], summary)
    else:
        kernel(inputs, csr, outputs, config[3])
    return finish_gradients(outputs, inputs)
