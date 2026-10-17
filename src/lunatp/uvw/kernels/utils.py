"""Path metadata shared by UVW and UVW convolution, and UVW tuning constants."""

from math import prod

import torch

from ...utils.cg import build_cg_terms, wigner_3j
from ...utils.layout import UvwPathSpec

# Maximum per-lane FP32 FragmentC values for two-row weight lookahead.
UVW_FWD_ACCUMULATOR_LIMIT = 10
UVW_FWD_LARGE_BATCH_THRESHOLD = 1024
UVW_SCALAR_WORKSPACE_ELEMENTS = 67108864
UVW_FWD_MAX_SPLITS = 16
UVW_GEMM_BLOCKS_PER_SM = 2
UVW_GEMM_TARGET_BLOCKS_PER_SM = 4


SCALAR_CUDA_SOURCE = r"""#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h>

namespace uvw_grouped {
inline int gemm_splits(int tiles,int k,int capacity,int blocks,int target) {
    if(capacity==1) return 1;
    int initial=std::min(capacity,std::max(1,int((int64_t(target)+tiles-1)/tiles)));
    int64_t waves=(int64_t(tiles)*initial+blocks-1)/blocks;
    int choices[]={initial,std::max(1,int(std::min<int64_t>(capacity,(waves-1)*blocks/tiles))),
                   std::max(1,int(std::min<int64_t>(capacity,waves*blocks/tiles)))};
    int best=initial;
    int64_t best_cost=INT64_MAX;
    for(int requested:choices) {
        int64_t step=int64_t(requested)*8;
        int64_t span=(int64_t(k)+step-1)/step*8;
        int parts=int((int64_t(k)+span-1)/span);
        int64_t cost=((int64_t(tiles)*parts+blocks-1)/blocks)*span;
        if(cost<best_cost || (cost==best_cost && parts<best)) {best=parts;best_cost=cost;}
    }
    return best;
}

struct Table {
    cutlass::gemm::GemmCoord* sizes;
    float **a, **b, **c, **d;
    int64_t *lda, *ldb, *ldc, *ldd;

    // One small device allocation; producers may populate entries in their existing launch.
    __host__ __device__ Table(void* memory, int capacity) {
        auto p = static_cast<int64_t*>(memory);
        a = reinterpret_cast<float**>(p); b = a + capacity;
        c = b + capacity; d = c + capacity;
        lda = reinterpret_cast<int64_t*>(d + capacity);
        ldb = lda + capacity; ldc = ldb + capacity; ldd = ldc + capacity;
        sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(ldd + capacity);
    }
    static constexpr size_t bytes(int capacity) { return size_t(capacity) * (8 * sizeof(int64_t) + sizeof(cutlass::gemm::GemmCoord)); }
    __device__ void set(int i, int m, int n, int k, const float* pa, const float* pb,
                        float* pc, float* pd, int64_t sa, int64_t sb, int64_t sc, int64_t sd) {
        sizes[i] = {m,n,k}; a[i] = const_cast<float*>(pa); b[i] = const_cast<float*>(pb);
        c[i] = pc; d[i] = pd; lda[i] = sa; ldb[i] = sb; ldc[i] = sc; ldd[i] = sd;
    }
};

template<class LayoutA, class LayoutB, class LayoutC = cutlass::layout::ColumnMajor,
         int M = 128, int N = 128, int K = 8,
         class PermuteD = cutlass::layout::NoPermute, int WarpM = 32, int WarpN = 64, int Stages = 2>
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, LayoutC, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<M,N,K>, cutlass::gemm::GemmShape<WarpM,WarpN,K>,
    cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone,
    PermuteD>::GemmKernel;

template<class K>
cutlass::Status launch(Table table, int count, int blocks, float alpha, float beta, cudaStream_t stream) {
    using Gemm = cutlass::gemm::device::GemmGrouped<K>;
    typename Gemm::Arguments args(table.sizes,count,blocks,{alpha,beta},
        table.a,table.b,table.c,table.d,table.lda,table.ldb,table.ldc,table.ldd);
    Gemm gemm;
    auto status = gemm.initialize(args,nullptr,stream);
    if(status != cutlass::Status::kSuccess) return status;
    return gemm.run(stream);
}

template<class LayoutA, class LayoutB, int Stages=2,int M=128,int N=128,int WarpM=64,int WarpN=64>
cutlass::Status single(cudaStream_t stream, int m, int n, int k, const float* a, int64_t lda,
                       const float* b, int64_t ldb, float* c, int64_t ldc, float beta, float alpha=1.f) {
    using Gemm = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, float, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M,N,8>, cutlass::gemm::GemmShape<WarpM,WarpN,8>,
        cutlass::gemm::GemmShape<1,1,1>,
        cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, 1, 1,
        false, cutlass::arch::OpMultiplyAdd>;
    typename Gemm::Arguments args({m,n,k},{a,lda},{b,ldb},{c,ldc},{c,ldc},{alpha,beta});
    Gemm gemm;
    return gemm(args,nullptr,stream);
}

inline cutlass::Status scalar_gemm(cudaStream_t stream, bool transpose_a, bool transpose_b,
        int m,int n,int k,const float* a,int64_t lda,const float* b,int64_t ldb,
        float* c,int64_t ldc,float beta,float alpha=1.f) {
    using Row=cutlass::layout::RowMajor;
    using Col=cutlass::layout::ColumnMajor;
    if(transpose_a) return single<Row,Col>(stream,m,n,k,a,lda,b,ldb,c,ldc,beta,alpha);
    if(transpose_b) {
        if(n<=64 && m<=64*65535) return single<Col,Row,2,64,64,32,32>(stream,m,n,k,a,lda,b,ldb,c,ldc,beta,alpha);
        return single<Col,Row>(stream,m,n,k,a,lda,b,ldb,c,ldc,beta,alpha);
    }
    return single<Col,Col>(stream,m,n,k,a,lda,b,ldb,c,ldc,beta,alpha);
}

template<class PermuteD=cutlass::layout::NoPermute,int M=128,int N=128,int WarpM=64,int WarpN=64>
struct VectorNT {
    using Base=uvw_grouped::Kernel<cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,cutlass::layout::RowMajor,M,N,8,PermuteD,WarpM,WarpN>;
    static constexpr int Threads=(M/WarpM)*(N/WarpN)*32;
    static constexpr int AccessA=std::min(4,M*8/Threads);
    static constexpr int AccessB=std::min(4,N*8/Threads);
    using MapA=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<M,8>,Threads,AccessA>;
    using MapB=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<N,8>,Threads,AccessB>;
    using IteratorA=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<M,8>,float,cutlass::layout::ColumnMajor,1,MapA,AccessA>;
    using IteratorB=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<8,N>,float,cutlass::layout::RowMajor,0,MapB,AccessB>;
    using SharedA=cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<M,8>,float,cutlass::layout::ColumnMajor,1,MapA>;
    using SharedB=cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<8,N>,float,cutlass::layout::RowMajor,0,MapB>;
    using Mma=cutlass::gemm::threadblock::MmaPipelined<cutlass::gemm::GemmShape<M,N,8>,IteratorA,SharedA,IteratorB,SharedB,float,cutlass::layout::RowMajor,typename Base::Mma::Policy>;
    using Kernel=cutlass::gemm::kernel::GemmGrouped<Mma,typename Base::Epilogue,cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>;
};

template<int Stages,int Count=1,int M=128,int N=128,int WarpM=64,int WarpN=64>
struct AsyncNT {
    using Parent=VectorNT<cutlass::layout::NoPermute,M,N,WarpM,WarpN>;
    using IteratorA=cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<M,8>,float,cutlass::layout::ColumnMajor,1,
        typename Parent::MapA,cutlass::Array<float,Parent::AccessA>>;
    using IteratorB=cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<8,N>,float,cutlass::layout::RowMajor,0,
        typename Parent::MapB,cutlass::Array<float,Parent::AccessB>>;
    using SharedA=cutlass::transform::threadblock::RegularTileAccessIterator<
        cutlass::MatrixShape<M,8>,float,cutlass::layout::ColumnMajor,1,typename Parent::MapA>;
    using SharedB=cutlass::transform::threadblock::RegularTileAccessIterator<
        cutlass::MatrixShape<8,N>,float,cutlass::layout::RowMajor,0,typename Parent::MapB>;
    using Mma=cutlass::gemm::threadblock::MmaMultistage<
        cutlass::gemm::GemmShape<M,N,8>,IteratorA,SharedA,
        cutlass::arch::CacheOperation::Global,IteratorB,SharedB,
        cutlass::arch::CacheOperation::Global,float,cutlass::layout::RowMajor,
        typename Parent::Mma::Policy,Stages>;
    using Warp=typename Mma::Policy::Operator;
    using Output=cutlass::epilogue::thread::LinearCombination<float,Count,float,float>;
    using Epi=cutlass::epilogue::threadblock::DefaultEpilogueSimt<cutlass::gemm::GemmShape<M,N,8>,Warp,Output,Count>;
    using Padding=cutlass::MatrixShape<Epi::Padding::kRow,(Epi::Padding::kColumn+Count-1)/Count*Count>;
    using Epilogue=cutlass::epilogue::threadblock::Epilogue<cutlass::gemm::GemmShape<M,N,8>,Warp,1,
        typename Epi::OutputTileIterator,typename Epi::AccumulatorFragmentIterator,
        typename Epi::WarpTileIterator,typename Epi::SharedLoadIterator,Output,Padding>;
};

template<int M=128,int N=64,int WM=64,int WN=32>
struct VectorTN {
    using Base=Kernel<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,M,N,8,cutlass::layout::NoPermute,WM,WN>;
    static constexpr int Threads=(M/WM)*(N/WN)*32;
    using MapA=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<8,M>,Threads,4>;
    using MapB=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<8,N>,Threads,4>;
    using IteratorA=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<M,8>,float,cutlass::layout::RowMajor,1,MapA,4>;
    using IteratorB=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<8,N>,float,cutlass::layout::ColumnMajor,0,MapB,4>;
    template<bool IsA> struct Shared {
        using Map=typename cutlass::platform::conditional<IsA,MapA,MapB>::type;
        using Element=float;
        using Layout=typename cutlass::platform::conditional<IsA,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor>::type;
        using TensorRef=cutlass::TensorRef<float,Layout>;
        using Fragment=typename cutlass::platform::conditional<IsA,typename IteratorA::Fragment,typename IteratorB::Fragment>::type;
        TensorRef ref;int outer,inner;
        __device__ Shared(TensorRef r,int t):ref(r){auto p=Map::initial_offset(t);outer=p.strided();inner=p.contiguous();}
        __device__ void store(Fragment const& f){
            #pragma unroll
            for(int s=0;s<Map::Iterations::kStrided;++s)
                #pragma unroll
                for(int c=0;c<Map::Iterations::kContiguous;++c)
                    #pragma unroll
                    for(int v=0;v<4;++v){
                        int o=outer+s*Map::Delta::kStrided,i=inner+c*Map::Delta::kContiguous+v;
                        ref.at(IsA?cutlass::MatrixCoord(o,i):cutlass::MatrixCoord(i,o))=f[(s*Map::Iterations::kContiguous+c)*4+v];
                    }
        }
        __device__ Shared& operator++(){inner+=8;return *this;}
        __device__ void add_tile_offset(cutlass::MatrixCoord const& p){outer+=(IsA?p.row()*M:p.column()*N);inner+=(IsA?p.column():p.row())*8;}
    };
    using Mma=cutlass::gemm::threadblock::MmaPipelined<cutlass::gemm::GemmShape<M,N,8>,IteratorA,Shared<true>,IteratorB,Shared<false>,float,cutlass::layout::RowMajor,typename Base::Mma::Policy>;
    using Kernel=cutlass::gemm::kernel::GemmGrouped<Mma,typename Base::Epilogue,cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>;
};
inline void scalar_tn(cudaStream_t stream,int m,int n,int k,const float* a,int64_t lda,
        const float* b,int64_t ldb,float* c,int64_t ldc,float beta) {
    if(((k|lda|ldb)&3) || ((reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(b))&15)) {
        scalar_gemm(stream,true,false,m,n,k,a,lda,b,ldb,c,ldc,beta);
        return;
    }
    using Vector=VectorTN<>;
    using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using K=cutlass::gemm::kernel::Gemm<typename Vector::Mma,typename Vector::Base::Epilogue,Swizzle,false>;
    Swizzle swizzle;
    auto tiled=swizzle.get_tiled_shape({n,m,k},{128,64,8},1);
    typename K::Params params({n,m,k},tiled,
        {const_cast<float*>(b),ldb},{const_cast<float*>(a),lda},
        {c,ldc},{c,ldc},{1.f,beta});
    void* arguments[]={&params};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(cutlass::Kernel<K>),
        swizzle.get_grid_shape(tiled),dim3(K::kThreadCount),arguments,sizeof(typename K::SharedStorage),stream));
}

using GemmNNBase=Kernel<cutlass::layout::RowMajor,cutlass::layout::RowMajor,cutlass::layout::RowMajor,128,128,8,cutlass::layout::NoPermute,64,64>;
using GemmNNThreadMapA=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<8,128>,128,4>;
using GemmNNThreadMapB=cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<128,8>,128,4>;
using GemmNNIteratorA=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<128,8>,float,cutlass::layout::RowMajor,1,GemmNNThreadMapA,4>;
using GemmNNIteratorB=cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<8,128>,float,cutlass::layout::RowMajor,0,GemmNNThreadMapB,4>;

// Global A vectors run along K; scatter them into the existing transposed shared layout.
struct GemmNNSharedA {
    using Element=float;
    using Layout=cutlass::layout::ColumnMajor;
    using TensorRef=cutlass::TensorRef<float,Layout>;
    using Fragment=GemmNNIteratorA::Fragment;
    TensorRef ref;
    int row, column;
    __device__ GemmNNSharedA(TensorRef ref_,int thread):ref(ref_) {
        auto offset=GemmNNThreadMapA::initial_offset(thread);
        row=offset.strided(); column=offset.contiguous();
    }
    __device__ void store(Fragment const& fragment) {
        #pragma unroll
        for(int s=0;s<GemmNNThreadMapA::Iterations::kStrided;++s)
            #pragma unroll
            for(int c=0;c<GemmNNThreadMapA::Iterations::kContiguous;++c)
                #pragma unroll
                for(int v=0;v<4;++v)
                    ref.at({row+s*GemmNNThreadMapA::Delta::kStrided,column+c*GemmNNThreadMapA::Delta::kContiguous+v})=
                        fragment[(s*GemmNNThreadMapA::Iterations::kContiguous+c)*4+v];
    }
    __device__ GemmNNSharedA& operator++() {column+=8;return *this;}
    __device__ void add_tile_offset(cutlass::MatrixCoord const& offset) {row+=offset.row()*128;column+=offset.column()*8;}
};
using GemmNNSharedB=cutlass::transform::threadblock::RegularTileIterator<cutlass::MatrixShape<8,128>,float,cutlass::layout::RowMajor,0,GemmNNThreadMapB>;
using GemmNNWarpPolicy=cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<8,4>,cutlass::layout::RowMajorInterleaved<2>,cutlass::gemm::GemmShape<4,4,1>>;
using GemmNNWarp=cutlass::gemm::warp::MmaSimt<cutlass::gemm::GemmShape<64,64,8>,float,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,GemmNNWarpPolicy>;
using GemmNNPolicy=cutlass::gemm::threadblock::MmaPolicy<GemmNNWarp,cutlass::MatrixShape<4,0>,cutlass::MatrixShape<0,0>,1>;
using GemmNNOutput=cutlass::epilogue::thread::LinearCombination<float,1,float,float>;
using GemmNNEpilogue=typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<cutlass::gemm::GemmShape<128,128,8>,GemmNNWarp,GemmNNOutput,1>::Epilogue;
using GemmNNMma=cutlass::gemm::threadblock::MmaPipelined<cutlass::gemm::GemmShape<128,128,8>,GemmNNIteratorA,GemmNNSharedA,GemmNNIteratorB,GemmNNSharedB,float,cutlass::layout::RowMajor,GemmNNPolicy>;
using GemmNN=cutlass::gemm::kernel::GemmGrouped<GemmNNMma,GemmNNEpilogue,cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>;

using GemmNNOutput4=cutlass::epilogue::thread::LinearCombination<float,4,float,float>;
using GemmNNEpilogue4Config=cutlass::epilogue::threadblock::DefaultEpilogueSimt<cutlass::gemm::GemmShape<128,128,8>,GemmNNWarp,GemmNNOutput4,4>;
// SIMT warp stores use scalar padding; round it up for aligned float4 shared loads.
using GemmNNEpilogue4=cutlass::epilogue::threadblock::Epilogue<
    cutlass::gemm::GemmShape<128,128,8>,GemmNNWarp,1,
    GemmNNEpilogue4Config::OutputTileIterator,GemmNNEpilogue4Config::AccumulatorFragmentIterator,
    GemmNNEpilogue4Config::WarpTileIterator,GemmNNEpilogue4Config::SharedLoadIterator,
    GemmNNOutput4,cutlass::MatrixShape<0,20>>;

inline cutlass::Status scalar_nn(cudaStream_t stream,int m,int n,int k,const float* a,int64_t lda,
        const float* b,int64_t ldb,float* c,int64_t ldc,float beta,float alpha=1.f) {
    if(((m|k|lda|ldb|ldc)&3)!=0 || ((reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(b)|reinterpret_cast<uintptr_t>(c))&15)!=0)
        return scalar_gemm(stream,false,false,m,n,k,a,lda,b,ldb,c,ldc,beta,alpha);
    using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Main=cutlass::gemm::kernel::Gemm<GemmNNMma,GemmNNEpilogue4,Swizzle,false>;
    cutlass::gemm::GemmCoord problem(n,m,k);
    auto tiled=Swizzle().get_tiled_shape(problem,{128,128,8},1);
    typename Main::Params params(problem,tiled,{const_cast<float*>(b),ldb},
        {const_cast<float*>(a),lda},{c,ldc},{c,ldc},{alpha,beta});
    void* args[]={&params};
    auto status=cudaLaunchKernel(reinterpret_cast<const void*>(cutlass::Kernel<Main>),
        Swizzle().get_grid_shape(tiled),dim3(Main::kThreadCount),args,sizeof(typename Main::SharedStorage),stream);
    return status==cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

}

using uvw_grouped::scalar_gemm;
using uvw_grouped::scalar_tn;
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

"""

SCALAR_PARALLEL_CUDA_SOURCE = r"""
#include <cutlass/gemm/kernel/gemm_splitk_parallel.h>

namespace uvw_grouped {
using SmallGemmNN=Kernel<cutlass::layout::RowMajor,cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,64,64,8,cutlass::layout::NoPermute,32,32>;

template<class Grouped>
void scalar_gemm_parallel(const float* a,const float* b,float* partial,
        int m,int n,int k,int splits,int slice_k,cudaStream_t stream) {
    using Swizzle=cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle;
    using Gemm=cutlass::gemm::kernel::GemmSplitKParallel<
        typename Grouped::Mma,typename Grouped::Epilogue,Swizzle>;
    Swizzle swizzle;
    auto tiled=swizzle.get_tiled_shape({m,n,k},
        {Grouped::Mma::Shape::kM,Grouped::Mma::Shape::kN,8},splits);
    typename Gemm::Params params({m,n,k},tiled,
        {const_cast<float*>(a),k},{const_cast<float*>(b),n},
        {partial,n},{1.f,0.f},int64_t(m)*n);
    params.gemm_k_size=slice_k;
    void* arguments[]={&params};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(cutlass::Kernel<Gemm>),
        swizzle.get_grid_shape(tiled),dim3(Gemm::kThreadCount),arguments,
        sizeof(typename Gemm::SharedStorage),stream));
}

} // namespace uvw_grouped

template<int Columns,int Stride,bool Accumulate,int Lanes=1>
__global__ void scalar_split_sum(const float* partial,float* out,
                                int64_t count,int splits,float scale) {
    static_assert(Lanes>=1 && Lanes<=32 && (Lanes&(Lanes-1))==0);
    const int64_t thread=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int64_t idx=thread/Lanes;
    const int lane=threadIdx.x%Lanes;
    float sum=0.f;
    if(idx<count)
        for(int s=lane;s<splits;s+=Lanes) sum+=partial[int64_t(s)*count+idx];
    #pragma unroll
    for(int offset=Lanes/2;offset>0;offset/=2)
        sum+=__shfl_down_sync(0xffffffff,sum,offset,Lanes);
    if(lane==0 && idx<count) {
        float* dst=out+(idx/Columns)*Stride+idx%Columns;
        if constexpr(Accumulate) *dst+=sum*scale;
        else *dst=sum*scale;
    }
}
"""


def build_uvw_metadata(tp, dtype):
    internal = bool(tp.internal_weights)
    shared = bool(tp.shared_weights)
    if shared:
        weight_mode = "shared"
    elif not internal and not shared:
        weight_mode = "external"
    else:
        raise ValueError(
            "uvw generated kernels support only "
            "(internal_weights=True, shared_weights=True) or "
            f"(internal_weights=False, shared_weights=False); got internal_weights={internal}, shared_weights={shared}."
        )

    x_slices = tp.irreps_in1.slices()
    y_slices = tp.irreps_in2.slices()
    out_slices = tp.irreps_out.slices()
    specs: list[UvwPathSpec] = []
    weight_offset = 0

    for ins in tp.instructions:
        if ins.connection_mode != "uvw":
            raise ValueError(
                f"Only connection_mode='uvw' is supported, got {ins.connection_mode!r}."
            )
        if not ins.has_weight:
            raise ValueError("uvw generated kernels require weighted instructions.")

        mul_in, ir_in = tp.irreps_in1[ins.i_in1]
        mul_edge, ir_edge = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]
        shape = tuple(int(v) for v in ins.path_shape)
        in_channels = int(mul_in)
        edge_channels = int(mul_edge)
        out_channels = int(mul_out)
        expected = (in_channels, edge_channels, out_channels)
        if shape != expected:
            raise ValueError(
                f"Unexpected uvw path_shape={shape}, expected {expected}."
            )
        path_channels = in_channels * edge_channels

        cg = (
            wigner_3j(ir_in.l, ir_edge.l, ir_out.l, dtype=torch.float64)
            if dtype == torch.float64
            else wigner_3j(ir_in.l, ir_edge.l, ir_out.l)
        )
        if cg.dtype is not dtype or cg.device.type != "cpu":
            cg = cg.to(dtype=dtype, device="cpu")
        path_weight = float(ins.path_weight)
        d1, d2, d3 = int(ir_in.dim), int(ir_edge.dim), int(ir_out.dim)
        terms = build_cg_terms(cg, path_weight)
        specs.append(
            {
                "x_offset": int(x_slices[ins.i_in1].start),
                "y_offset": int(y_slices[ins.i_in2].start),
                "out_offset": int(out_slices[ins.i_out].start),
                "weight_offset": weight_offset,
                "in_channels": in_channels,
                "edge_channels": edge_channels,
                "out_channels": out_channels,
                "path_channels": path_channels,
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "l1": int(ir_in.l),
                "l2": int(ir_edge.l),
                "l3": int(ir_out.l),
                "terms": terms,
            }
        )
        weight_offset += prod(shape)

    weight_numel = int(tp.weight_numel)
    if weight_offset != weight_numel:
        raise ValueError(
            f"lowered weight_numel={weight_offset}, configuration reports {tp.weight_numel}."
        )

    meta = {
        "in1_dim": int(tp.irreps_in1.dim),
        "in2_dim": int(tp.irreps_in2.dim),
        "out_dim": int(tp.irreps_out.dim),
        "weight_numel": weight_numel,
        "max_in_channels": max((spec["in_channels"] for spec in specs), default=0),
        "max_out_channels": max(
            (spec["out_channels"] for spec in specs), default=0
        ),
        "forward_costs": [
            len(spec["terms"]) * spec["in_channels"] for spec in specs
        ],
        "grad_x_costs": [
            len(spec["terms"]) * spec["out_channels"] for spec in specs
        ],
        "weight_mode": weight_mode,
        "generated_specs": specs,
        "scalar_input": bool(specs) and all(
            spec["d2"] == 1 and spec["d1"] == spec["d3"]
            and len(spec["terms"]) == spec["d1"]
            and all(i == k and j == 0 and value == spec["terms"][0][3]
                    for i, j, k, value in spec["terms"])
            for spec in specs
        ),
    }
    if dtype == torch.float64:
        meta["dtype"] = dtype
    return meta
