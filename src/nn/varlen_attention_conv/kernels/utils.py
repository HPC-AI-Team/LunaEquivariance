"""Shared tensor views, directional arithmetic, and CUTLASS GEMM source."""

import torch

from ....utils.extension import get_cuda_arch, load_cuda_extension
from ...varlen_attention.kernels.utils import GEMM_SOURCE, dtype_defines


CUDA_SOURCE = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Value, two directional derivatives, and their mixed derivative.
// Compile-time width removes unused arithmetic from lower derivative orders.
template <int N> struct Differential {
    float x[N];
    __device__ __forceinline__ Differential(float value = 0.f) {
        x[0] = value;
        for (int i = 1; i < N; ++i)
            x[i] = 0.f;
    }
};
template <int N> __device__ __forceinline__ Differential<N> operator+(Differential<N> a, Differential<N> b) {
    for (int i = 0; i < N; ++i)
        a.x[i] += b.x[i];
    return a;
}
template <int N> __device__ __forceinline__ Differential<N> operator-(Differential<N> a, Differential<N> b) {
    for (int i = 0; i < N; ++i)
        a.x[i] -= b.x[i];
    return a;
}
template <int N> __device__ __forceinline__ Differential<N> operator*(Differential<N> a, Differential<N> b) {
    Differential<N> c(a.x[0] * b.x[0]);
    if constexpr (N >= 2)
        c.x[1] = fmaf(a.x[1], b.x[0], a.x[0] * b.x[1]);
    if constexpr (N == 4) {
        c.x[2] = fmaf(a.x[2], b.x[0], a.x[0] * b.x[2]);
        c.x[3] = fmaf(a.x[3], b.x[0], fmaf(a.x[1], b.x[2], fmaf(a.x[2], b.x[1], a.x[0] * b.x[3])));
    }
    return c;
}
template <int N> __device__ __forceinline__ Differential<N> reciprocal(Differential<N> a) {
    float inv = 1.f / a.x[0];
    Differential<N> b(inv);
    if constexpr (N >= 2)
        b.x[1] = -a.x[1] * inv * inv;
    if constexpr (N == 4) {
        b.x[2] = -a.x[2] * inv * inv;
        b.x[3] = (2.f * a.x[1] * a.x[2] * inv - a.x[3]) * inv * inv;
    }
    return b;
}
template <int N> __device__ __forceinline__ Differential<N> exponential(Differential<N> a) {
    float e = __expf(a.x[0]);
    Differential<N> b(e);
    if constexpr (N >= 2)
        b.x[1] = e * a.x[1];
    if constexpr (N == 4) {
        b.x[2] = e * a.x[2];
        b.x[3] = e * fmaf(a.x[1], a.x[2], a.x[3]);
    }
    return b;
}
template <int N> __device__ __forceinline__ Differential<N> warp_sum(Differential<N> a) {
    for (int offset = 16; offset; offset /= 2)
        for (int c = 0; c < N; ++c)
            a.x[c] += __shfl_xor_sync(0xffffffff, a.x[c], offset);
    return a;
}
// Only storage access depends on dtype; addressing and FP32 arithmetic are shared.
struct View {
#if defined(LOWP_BF16)
    void *data = nullptr;
    int dtype = 0;
#else
    float *data = nullptr;
#endif
    int64_t stride[4] = {};
    __device__ __forceinline__ int64_t index(int64_t i, int64_t j = 0, int64_t k = 0, int64_t l = 0) const {
        return i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3];
    }
    __device__ __forceinline__ float load(int64_t offset, int slot) const {
#if defined(LOWP_BF16)
        constexpr uint64_t types=INPUT_DTYPES;
        int type=(types>>(2*slot))&3u;
        if (type == 0) return static_cast<const float *>(data)[offset];
        if (type == 1) return __half2float(static_cast<const __half *>(data)[offset]);
        return __bfloat162float(static_cast<const __nv_bfloat16 *>(data)[offset]);
#else
        return data[offset];
#endif
    }
    __device__ __forceinline__ void store(int64_t offset, float v) const {
#if defined(LOWP_BF16)
        if (dtype == 0) static_cast<float *>(data)[offset] = v;
        else if (dtype == 1) static_cast<__half *>(data)[offset] = __float2half_rn(v);
        else static_cast<__nv_bfloat16 *>(data)[offset] = __float2bfloat16_rn(v);
#else
        data[offset] = v;
#endif
    }
    // Gradient storage is allocated once in FP32 for every input dtype.
    __device__ __forceinline__ void accumulate(int64_t offset, float v) const {
        static_cast<float*>(data)[offset] += v;
    }
    __device__ __forceinline__ void atomic_add(int64_t offset, float v) const {
        atomicAdd(static_cast<float*>(data) + offset, v);
    }

};
View view(std::optional<at::Tensor> const &t) {
    View v;
    if (t) {
#if defined(LOWP_BF16)
        v.data = t->data_ptr();
        v.dtype = t->scalar_type() == at::kFloat ? 0 : (t->scalar_type() == at::kHalf ? 1 : 2);
#else
        v.data = t->data_ptr<float>();
#endif
        for (int i = 0; i < t->dim(); ++i) {
            v.stride[i] = t->stride(i);
        }
    }
    return v;
}
struct Params {
    View in[19], out[12];
    const void *csr;
    bool csr64;
    int64_t csr_stride;
    float scale;
    __device__ __forceinline__ int64_t offset(int64_t a) const {
        return csr64 ? static_cast<int64_t const *>(csr)[a * csr_stride]
                     : static_cast<int const *>(csr)[a * csr_stride];
    }
};
Params params(std::vector<std::optional<at::Tensor>> const &in,
              std::vector<std::optional<at::Tensor>> const &out, at::Tensor csr, double scale) {
    Params p{};
    for (size_t i = 0; i < in.size(); ++i)
        p.in[i] = view(in[i]);
    for (size_t i = 0; i < out.size(); ++i)
        p.out[i] = view(out[i]);
    p.csr = csr.data_ptr();
    p.csr64 = csr.scalar_type() == at::kLong;
    p.csr_stride = csr.stride(0);
    p.scale = scale;
    return p;
}
// Compile-time presence removes missing bias, envelope and gradient seeds.
// Packed tensors use the known feature width; arbitrary strides use View.
__device__ __forceinline__ int64_t input_offset(Params const& p,int actual,int slot,
    int64_t i,int64_t j,int64_t k,int64_t l=0) {
    if(CONTIGUOUS&(1u<<actual)) {
        if(slot==0 || slot==1) return (i*HEADS+j)*DIM+k;
        if(slot==2 || slot==5 || slot==6) return (i*HEADS+j)*VALUE_DIM+k;
        if(slot==4) return i;
    }
    return p.in[actual].index(i,j,k,l);
}
template <int N>
__device__ __forceinline__ Differential<N> input(Params const& p,int slot,int64_t i,int64_t j=0,
    int64_t k=0,int64_t l=0) {
    Differential<N> value;
    if(PRESENT&(1u<<slot)) value.x[0]=p.in[slot].load(input_offset(p,slot,slot,i,j,k,l),slot);
    if constexpr(N>=2)
        if(PRESENT&(1u<<(7+slot))) value.x[1]=p.in[7+slot].load(input_offset(p,7+slot,slot,i,j,k,l),7+slot);
    if constexpr(N==4)
        if(PRESENT&(1u<<(12+slot))) value.x[2]=p.in[12+slot].load(input_offset(p,12+slot,slot,i,j,k,l),12+slot);
    return value;
}
template <int N>
__device__ __forceinline__ Differential<N> upstream(Params const& p,int slot,int64_t i,int h,int c) {
    Differential<N> value;
    if(PRESENT&(1u<<slot)) value.x[0]=p.in[slot].load(input_offset(p,slot,slot,i,h,c),slot);
    if constexpr(N==4)
        if(PRESENT&(1u<<(12+slot))) value.x[2]=p.in[12+slot].load(input_offset(p,12+slot,slot,i,h,c),12+slot);
    return value;
}

"""


GEMM_DIFFERENTIAL_SOURCE = r"""
#if LOW_PRECISION
// CUTLASS still requires iterator interfaces when operands are already staged.
struct ConvReadyOperand {
    using Fragment=cutlass::Array<GemmElement,1>;
    __device__ __forceinline__ ConvReadyOperand() {}
    template<typename Ref> __device__ __forceinline__ ConvReadyOperand(Ref const&,int) {}
    __device__ __forceinline__ void clear_mask(bool) {}
    __device__ __forceinline__ ConvReadyOperand& operator++() {return *this;}
    __device__ __forceinline__ void load(Fragment& fragment) const {fragment.clear();}
    __device__ __forceinline__ void store(Fragment const&) const {}
};

// Raw Q/K/V tiles retain their storage type until the TensorOp mainloop.
// Coefficients and all higher-order products keep their FP32 arithmetic.
template<int Slot,bool Transpose>
struct ConvGemmFeature {
    Params const& p;
    int64_t begin,end;
    int head,channel=0;
    static constexpr int Width=Slot==2 ? VALUE_DIM:DIM;
    static constexpr int InputType=(INPUT_DTYPES>>(2*Slot))&3u;

    __device__ __forceinline__ GemmElement operator()(int row,int column) const {
        int64_t edge=begin+(Transpose ? column:row);
        int c=channel+(Transpose ? row:column);
        if(edge>=end || c>=Width) return GemmElement(0.f);
        int64_t offset=input_offset(p,Slot,Slot,edge,head,c);
        if constexpr(InputType==0) return GemmElement(static_cast<float const*>(p.in[Slot].data)[offset]);
        else return static_cast<GemmElement const*>(p.in[Slot].data)[offset];
    }

    template<bool First,typename Fragment>
    __device__ __forceinline__ void load8(Fragment& fragment,int index,int row,int column) const {
        if constexpr(First==Transpose && InputType!=0 && (CONTIGUOUS&(1u<<Slot))) {
            int64_t edge=begin+(Transpose ? column:row);
            int c=channel+(Transpose ? row:column);
            auto pointer=static_cast<GemmElement const*>(p.in[Slot].data)+input_offset(p,Slot,Slot,edge,head,c);
            if(edge<end && c+7<Width && (reinterpret_cast<uintptr_t>(pointer)&15u)==0) {
                uint4 packed=*reinterpret_cast<uint4 const*>(pointer);
                auto values=reinterpret_cast<GemmElement const*>(&packed);
                #pragma unroll
                for(int i=0;i<8;++i) fragment[index+i]=values[i];
                return;
            }
        }
        #pragma unroll
        for(int i=0;i<8;++i)
            fragment[index+i]=operator()(row+(First ? i:0),column+(First ? 0:i));
    }
};
#endif

// Concatenate product-rule terms along K. Each requested derivative
// component is one complete CUTLASS GEMM, with one accumulator epilogue.
// Selected components occupy consecutive output tiles.
template<int N, bool LeftMixed=true, bool RightMixed=true,
    unsigned Components=(1u<<N)-1, int Columns, bool RowMajorB, typename Left, typename Right>
__device__ __forceinline__ void differential_gemm(AttentionGemmTile<Columns,RowMajorB>& shared, float (*output)[32*Columns],
    int reduction, Left left, Right right) {
    if constexpr(Components&1u)
        attention_gemm(shared,output[0],reduction,
            [&](int row,int k) {return left(row,k).x[0];},
            [&](int k,int column) {return right(k,column).x[0];});
    if constexpr(N>=2 && (Components&2u))
        attention_gemm(shared,output[int(bool(Components&1u))],2*reduction,
            [&](int row,int k) {return k<reduction ? left(row,k).x[1]:left(row,k-reduction).x[0];},
            [&](int k,int column) {return k<reduction ? right(k,column).x[0]:right(k-reduction,column).x[1];});
    if constexpr(N==4 && (Components&4u))
        attention_gemm(shared,output[int(bool(Components&1u))+int(bool(Components&2u))],2*reduction,
            [&](int row,int k) {return k<reduction ? left(row,k).x[2]:left(row,k-reduction).x[0];},
            [&](int k,int column) {return k<reduction ? right(k,column).x[0]:right(k-reduction,column).x[2];});
    if constexpr(N==4 && (Components&8u)) {
        constexpr int terms=2+int(LeftMixed)+int(RightMixed);
        attention_gemm(shared,output[int(bool(Components&1u))+int(bool(Components&2u))+int(bool(Components&4u))],terms*reduction,
            [&](int row,int k) {
                if(k<reduction) return left(row,k).x[1];
                if(k<2*reduction) return left(row,k-reduction).x[2];
                if constexpr(LeftMixed) {
                    if(k<3*reduction) return left(row,k-2*reduction).x[3];
                    if constexpr(RightMixed) return left(row,k-3*reduction).x[0];
                } else if constexpr(RightMixed) return left(row,k-2*reduction).x[0];
                return 0.f;
            },
            [&](int k,int column) {
                if(k<reduction) return right(k,column).x[2];
                if(k<2*reduction) return right(k-reduction,column).x[1];
                if constexpr(LeftMixed) {
                    if(k<3*reduction) return right(k-2*reduction,column).x[0];
                    if constexpr(RightMixed) return right(k-3*reduction,column).x[3];
                } else if constexpr(RightMixed) return right(k-2*reduction,column).x[3];
                return 0.f;
            });
    }
}

template<int N, unsigned Components=(1u<<N)-1, int Elements>
__device__ __forceinline__ Differential<N> matrix_value(float const (*matrix)[Elements],int index) {
    Differential<N> value;
    int selected=0;
    #pragma unroll
    for(int component=0;component<N;++component)
        if(Components&(1u<<component)) value.x[component]=matrix[selected++][index];
    return value;
}

template<int N>
__device__ __forceinline__ void matrix_store(float (*matrix)[1024],int index,Differential<N> value) {
    #pragma unroll
    for(int component=0;component<N;++component) matrix[component][index]=value.x[component];
}

template<int N>
__device__ __forceinline__ Differential<N> output_gradient(Params const& p,int64_t atom,int64_t row,int head,int c) {
    auto envelope=(PRESENT&16u) ? input<N>(p,4,row):Differential<N>(1.f);
    return envelope*upstream<N>(p,5,atom,head,c)+upstream<N>(p,6,row,head,c);
}

template<int N,typename Shared>
__device__ __forceinline__ void attention_scores(Params const& p,Shared& shared,float (*scores)[1024],
    int64_t atom,int64_t begin,int64_t end,int64_t qb,int64_t kb,int head) {
#if LOW_PRECISION
    if constexpr(N==1)
        attention_gemm(shared,scores[0],DIM,
            ConvGemmFeature<0,false>{p,qb,end,head},ConvGemmFeature<1,true>{p,kb,end,head});
    else
#endif
    differential_gemm<N,false,false>(shared,scores,DIM,
        [&](int row,int c) {return qb+row<end ? input<N>(p,0,qb+row,head,c):Differential<N>();},
        [&](int c,int column) {return kb+column<end ? input<N>(p,1,kb+column,head,c):Differential<N>();});
    for(int i=threadIdx.x;i<1024;i+=128) {
        int64_t q=qb+i/32,k=kb+i%32;
        Differential<N> score(-INFINITY);
        if(q<end && k<end && (!CAUSAL || k<=q)) {
            score=matrix_value<N>(scores,i)*Differential<N>(p.scale);
            if constexpr(BIAS_KIND==1) score=score+input<N>(p,3,k,head);
            if constexpr(BIAS_KIND==2) score=score+input<N>(p,3,atom,head,q-begin,k-begin);
        }
        matrix_store<N>(scores,i,score);
    }
    __syncthreads();
}
"""


def load_kernel(name, source, config, device_index, dtype, layout):
    dim, value_dim, heads, _, causal, bias_kind = config
    present, contiguous, input_types = layout
    defines = (
        f"#define DIM {dim}\n#define VALUE_DIM {value_dim}\n#define HEADS {heads}\n"
        f"#define CAUSAL {int(causal)}\n#define BIAS_KIND {bias_kind}\n"
    )
    defines += f"#define PRESENT {present}u\n#define CONTIGUOUS {contiguous}u\n"
    if dtype != torch.float32:
        defines += f"#define LOWP_BF16 {int(dtype == torch.bfloat16)}\n"
        defines += f"#define INPUT_DTYPES {input_types}ull\n"
    return load_cuda_extension(
        name,
        dtype_defines(dtype) + defines + CUDA_SOURCE + GEMM_SOURCE + GEMM_DIFFERENTIAL_SOURCE + source,
        arch=get_cuda_arch(torch.device("cuda", device_index)),
        use_fast_math=False,
        minimal_headers=True,
    )


def input_layout(inputs):
    present = contiguous = input_types = 0
    for i, value in enumerate(inputs):
        if value is not None:
            bit = 1 << i
            present |= bit
            if value.is_contiguous():
                contiguous |= bit
            dtype = value.dtype
            input_types |= (0 if dtype == torch.float32 else 1 if dtype == torch.float16 else 2) << (2 * i)
    return present, contiguous, input_types


def finish_gradients(outputs, inputs):
    return tuple(
        y.to(x.dtype) if y is not None and y.dtype != x.dtype else y
        for y, x in zip(outputs, inputs)
    )
