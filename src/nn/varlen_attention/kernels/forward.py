"""Online CSR attention with FP32 softmax and dtype-specific CUTLASS products."""

from functools import lru_cache

from .utils import dtype_defines

import torch

from ....utils.extension import get_cuda_arch, get_device_properties, load_cuda_extension

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_conversion.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/layout/matrix.h>
#include <cuda_runtime.h>

struct Tensor3 {
    const Storage *data;
    int64_t edge_stride, head_stride, channel_stride;
    template<int Channels, bool Contiguous>
    __device__ __forceinline__ float load(int64_t edge, int head, int channel) const {
        if constexpr (Contiguous) return data[(edge*HEADS+head)*Channels+channel];
        return data[edge * edge_stride + int64_t(head) * head_stride + int64_t(channel) * channel_stride];
    }
};
struct Bias {
    const Storage *data;
    bool fp32;
    int64_t strides[4];
    __device__ float load(int64_t index) const {
        return fp32 ? reinterpret_cast<const float*>(data)[index] : float(data[index]);
    }
};

template<int Rows, int Threads, int Channels, int Width, bool Contiguous, typename SharedElement, int Pitch>
__device__ __forceinline__ void load_forward_tile(
    Tensor3 const& source, SharedElement (*destination)[Pitch],
    int64_t begin, int64_t end, int head, int tid) {
    constexpr int Pack = 16 / sizeof(Storage);
    using Packed = uint4;
    if constexpr (Contiguous && Channels%Pack==0) {
        for (int i=tid*Pack;i<Rows*Width;i+=Threads*Pack) {
            int r=i/Width, d=i%Width;
            Packed value={0,0,0,0};
            if (begin+r<end && d<Channels)
                value=*reinterpret_cast<const Packed*>(source.data+((begin+r)*HEADS+head)*Channels+d);
            *reinterpret_cast<Packed*>(&destination[r][d])=value;
        }
    } else {
        for (int i=tid;i<Rows*Width;i+=Threads) {
            int r=i/Width, d=i%Width;
            destination[r][d]=SharedElement(begin+r<end && d<Channels ? source.template load<Channels, Contiguous>(begin+r, head, d) : 0);
        }
    }
}
"""


FORWARD_SOURCE = r"""
template<int Dim, int ValueDim, int Kind, bool Causal, bool Contiguous, typename CsrIndex>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS_PER_SM) void attention_forward_kernel(
    const __grid_constant__ Tensor3 query, const __grid_constant__ Tensor3 key, const __grid_constant__ Tensor3 value, const __grid_constant__ Bias bias, const CsrIndex *csr, int64_t csr_stride,
    Storage *output, float *state, float scale) {
    constexpr int Tile=FORWARD_TILE;
    using SharedElement = LowMmaElement;
    constexpr int D = (Dim+15)/16*16;
    constexpr int V = (ValueDim+7)/8*8;
    constexpr int QPitch = D + 8;
    using Mma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32,
        LowMmaElement, cutlass::layout::RowMajor, LowMmaElement,
        cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>;
    using Convert = cutlass::NumericConverter<LowMmaElement, float, cutlass::FloatRoundStyle::round_to_nearest>;
#if SHARED_BYTES > 0
    extern __shared__ __align__(16) SharedElement storage[];
    auto queries=reinterpret_cast<SharedElement (*)[QPitch]>(storage);
    auto keys=reinterpret_cast<SharedElement (*)[QPitch]>(storage+Tile*QPitch);
    auto values=reinterpret_cast<SharedElement (*)[V+8]>(storage+2*Tile*QPitch);
    auto probability=reinterpret_cast<SharedElement (*)[Tile+4]>(storage+2*Tile*QPitch+Tile*(V+8));
#else
    __shared__ __align__(16) SharedElement queries[Tile][QPitch], keys[Tile][QPitch], values[Tile][V+8], probability[Tile][Tile+4];
#endif
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    int row=warp*16+lane/4, col=(lane%4)*2;
    int segment=(FORWARD_ADJACENT ? blockIdx.x/FORWARD_WORKERS : blockIdx.x), head=blockIdx.y;
    int64_t begin=csr[int64_t(segment)*csr_stride], end=csr[(int64_t(segment)+1)*csr_stride];
    if(end-begin<FORWARD_MIN_LENGTH || (FORWARD_MAX_LENGTH && end-begin>FORWARD_MAX_LENGTH)) return;
    for (int64_t query_start=begin+(FORWARD_ADJACENT ? blockIdx.x%FORWARD_WORKERS : blockIdx.z)*Tile;query_start<end;query_start+=FORWARD_WORKERS*Tile) {
        load_forward_tile<Tile, THREADS, Dim, D, Contiguous>(query, queries, query_start, end, head, tid);
        typename Mma::FragmentC accum[V/8];
        #pragma unroll
        for (int n=0;n<V/8;++n) accum[n].clear();
        float maximum[2]={-INFINITY, -INFINITY}, denominator[2]={0, 0};
        for (int64_t key_start=begin;key_start<end && (!Causal || key_start<=query_start+Tile-1);key_start+=Tile) {
            load_forward_tile<Tile, THREADS, Dim, D, Contiguous>(key, keys, key_start, end, head, tid);
            load_forward_tile<Tile, THREADS, ValueDim, V, Contiguous>(value, values, key_start, end, head, tid);
            __syncthreads();
            typename Mma::FragmentC scores[Tile/8];
            #pragma unroll
            for (int n=0;n<Tile/8;++n) scores[n].clear();
            #pragma unroll 1
            for (int d=0;d<D;d+=16) {
                typename Mma::FragmentA a;
                #pragma unroll
                for (int i=0;i<Mma::FragmentA::kElements;++i)
                    a[i]=Convert{}(queries[row+((i/2)%2)*8][d+(lane%4)*2+i%2+(i/4)*8]);
                #pragma unroll
                for (int n=0;n<Tile/8;++n) {
                    typename Mma::FragmentB b;
                    #pragma unroll
                    for (int i=0;i<Mma::FragmentB::kElements;++i)
                        b[i]=Convert{}(keys[n*8+lane/4][d+(lane%4)*2+i%2+(i/2)*8]);
                    Mma{}(scores[n], a, b, scores[n]);
                }
            }
            #pragma unroll
            for (int m=0;m<2;++m) {
                int64_t query_index=query_start+row+m*8;
                float tile_max=-INFINITY;
                #pragma unroll
                for (int n=0;n<Tile/8;++n) {
                    #pragma unroll
                    for (int j=0;j<2;++j) {
                        int64_t key_index=key_start+n*8+col+j;
                        float s=-INFINITY;
                        if (query_index<end && key_index<end && (!Causal || key_index<=query_index)) {
                            s=scores[n][m*2+j]*scale;
                            if constexpr(Kind==1) s+=bias.load(key_index*bias.strides[0]+head*bias.strides[1]);
                            if constexpr(Kind==2) s+=bias.load(int64_t(segment)*bias.strides[0]+head*bias.strides[1]+(query_index-begin)*bias.strides[2]+(key_index-begin)*bias.strides[3]);
                        }
                        scores[n][m*2+j]=s;
                        tile_max=fmaxf(tile_max, s);
                    }
                }
                tile_max=fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1, 4));
                tile_max=fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2, 4));
                float updated=fmaxf(maximum[m], tile_max);
                float alpha=updated>-INFINITY ? __expf(maximum[m]-updated) : 0;
                float sum=0;
                #pragma unroll
                for (int n=0;n<Tile/8;++n) {
                    #pragma unroll
                    for (int j=0;j<2;++j) {
                        float p=updated>-INFINITY ? __expf(scores[n][m*2+j]-updated) : 0;
                        probability[row+m*8][n*8+col+j]=SharedElement(p);
                        sum+=p;
                    }
                }
                sum+=__shfl_xor_sync(0xffffffff, sum, 1, 4);
                sum+=__shfl_xor_sync(0xffffffff, sum, 2, 4);
                denominator[m]=denominator[m]*alpha+sum;
                maximum[m]=updated;
                #pragma unroll
                for (int n=0;n<V/8;++n) {
                    accum[n][m*2]*=alpha;
                    accum[n][m*2+1]*=alpha;
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int d=0;d<Tile;d+=16) {
                typename Mma::FragmentA a;
                #pragma unroll
                for (int i=0;i<Mma::FragmentA::kElements;++i)
                    a[i]=Convert{}(probability[row+((i/2)%2)*8][d+(lane%4)*2+i%2+(i/4)*8]);
                #pragma unroll
                for (int n=0;n<V/8;++n) {
                    typename Mma::FragmentB b;
                    #pragma unroll
                    for (int i=0;i<Mma::FragmentB::kElements;++i)
                        b[i]=Convert{}(values[d+(lane%4)*2+i%2+(i/2)*8][n*8+lane/4]);
                    Mma{}(accum[n], a, b, accum[n]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int m=0;m<2;++m) if (query_start+row+m*8<end) {
            int64_t index=(query_start+row+m*8)*HEADS+head;
            float inverse=denominator[m]>0 ? 1.0f/denominator[m] : 0;
            if (lane%4==0) {
                state[index*2]=maximum[m];
                state[index*2+1]=inverse;
            }
            #pragma unroll
            for (int n=0;n<V/8;++n) {
                int d=n*8+col;
                #pragma unroll
                for(int j=0;j<2;++j) if(d+j<ValueDim)
                    values[row+m*8][d+j]=SharedElement(accum[n][m*2+j]*inverse);
            }
        }
        __syncthreads();
        if constexpr (ValueDim%8==0) {
            for(int i=tid*8;i<Tile*V;i+=THREADS*8) {
                int r=i/V,d=i%V;
                if(query_start+r<end && d<ValueDim)
                    *reinterpret_cast<uint4*>(output+((query_start+r)*HEADS+head)*ValueDim+d)=
                        *reinterpret_cast<const uint4*>(&values[r][d]);
            }
        } else {
            for(int i=tid;i<Tile*V;i+=THREADS) {
                int r=i/V,d=i%V;
                if(query_start+r<end && d<ValueDim)
                    output[((query_start+r)*HEADS+head)*ValueDim+d]=float(values[r][d]);
            }
        }
        __syncthreads();
    }
}
"""


FORWARD_SIMT_SOURCE = r"""
template<int Dim, int ValueDim, int Kind, bool Causal, bool Contiguous, typename CsrIndex>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS_PER_SM) void attention_forward_simt_kernel(
    const __grid_constant__ Tensor3 query, const __grid_constant__ Tensor3 key, const __grid_constant__ Tensor3 value, const __grid_constant__ Bias bias, const CsrIndex *csr, int64_t csr_stride,
    Storage *output, float *state, float scale) {
    constexpr int Tile = FORWARD_TILE;
    constexpr int Rows = Tile/16;
    constexpr int Lanes = THREADS/16;
    constexpr int PairColumns = Tile/Lanes;
    constexpr int Features = Dim > ValueDim ? Dim : ValueDim;
    constexpr int FeatureWidth = Features <= FORWARD_FEATURE_LIMIT ? (Features+31)/32*32 : 64;
    constexpr int Width = FeatureWidth < Tile ? Tile : FeatureWidth;
    constexpr int ValueTile = ValueDim < Width ? ValueDim : Width;
    constexpr int Columns = LOW_PRECISION ? ((ValueTile+7)/8<4 ? (ValueTile+7)/8 : 4) : (ValueTile+Lanes-1)/Lanes;
    constexpr int Chunks = (ValueTile + Lanes * Columns - 1) / (Lanes * Columns);
    using Mma = cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<Rows, PairColumns, 1>,
        float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
    using ValueMma = cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<Rows, Columns, 1>,
        float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
    extern __shared__ float storage[];
    constexpr bool RowMajor = !LOW_PRECISION && Tile>=48;
    constexpr int LeftSize = RowMajor ? Tile*(Width+4) : Width*(Tile+1);
    auto qcache = reinterpret_cast<float (*)[Width+4]>(storage);
    auto kcache = reinterpret_cast<float (*)[Width+4]>(storage+LeftSize);
    auto left = reinterpret_cast<float (*)[Tile+1]>(storage);
    auto right = reinterpret_cast<float (*)[Tile+1]>(storage+LeftSize);
    int tid = threadIdx.x, lane = tid % Lanes;
    constexpr int KeyStride = LOW_PRECISION ? 1 : Lanes;
    int row = tid / Lanes * Rows, column = LOW_PRECISION ? lane * PairColumns : lane;
    int segment = (FORWARD_ADJACENT ? blockIdx.x/FORWARD_WORKERS : blockIdx.x), head = blockIdx.y;
    int64_t begin = csr[int64_t(segment)*csr_stride], end = csr[(int64_t(segment)+1)*csr_stride];
    if(end-begin<FORWARD_MIN_LENGTH || (FORWARD_MAX_LENGTH && end-begin>FORWARD_MAX_LENGTH)) return;
    for (int64_t query_start = begin+(FORWARD_ADJACENT ? blockIdx.x%FORWARD_WORKERS : blockIdx.z)*Tile; query_start < end; query_start += FORWARD_WORKERS*Tile) {
        for (int vb = 0; vb < ValueDim; vb += ValueTile) {
            float maximum[Rows];
            #pragma unroll
            for (int m=0;m<Rows;++m) maximum[m]=-INFINITY;
            float denominator[Rows] = {};
            typename ValueMma::FragmentC accumulator[Chunks];
            #pragma unroll
            for (int c=0;c<Chunks;++c) accumulator[c].clear();
            for (int64_t key_start = begin; key_start < end && (!Causal || key_start <= query_start+Tile-1); key_start += Tile) {
                typename Mma::FragmentC scores;
                scores.clear();
                for (int fb = 0; fb < Dim; fb += Width) {
                    if constexpr (!LOW_PRECISION && Contiguous && Dim%4==0 && Width%4==0) {
                        for (int i=tid*4;i<Tile*Width;i+=THREADS*4) {
                            int r=i/Width, d=i%Width;
                            float4 q={0, 0, 0, 0}, k={0, 0, 0, 0};
                            if (query_start+r<end && fb+d<Dim) q=*reinterpret_cast<const float4*>(query.data+((query_start+r)*HEADS+head)*Dim+fb+d);
                            if (key_start+r<end && fb+d<Dim) k=*reinterpret_cast<const float4*>(key.data+((key_start+r)*HEADS+head)*Dim+fb+d);
                            if constexpr (RowMajor) {
                                *reinterpret_cast<float4*>(&qcache[r][d])=q;
                                *reinterpret_cast<float4*>(&kcache[r][d])=k;
                            } else {
                                left[d][r]=q.x;
                                left[d+1][r]=q.y;
                                left[d+2][r]=q.z;
                                left[d+3][r]=q.w;
                                right[d][r]=k.x;
                                right[d+1][r]=k.y;
                                right[d+2][r]=k.z;
                                right[d+3][r]=k.w;
                            }
                        }
                    } else {
                        for (int i=tid;i<Tile*Width;i+=THREADS) {
                            int r=i/Width, d=i%Width;
                            float q=query_start+r<end && fb+d<Dim ? query.template load<Dim, Contiguous>(query_start+r, head, fb+d) : 0;
                            if constexpr (RowMajor) qcache[r][d]=q;
                            else left[d][r]=q;
                            float k=key_start+r<end && fb+d<Dim ? key.template load<Dim, Contiguous>(key_start+r, head, fb+d) : 0;
                            if constexpr (RowMajor) kcache[r][d]=k;
                            else right[d][r]=k;
                        }
                    }
                    __syncthreads();
                    if (query_start+row<end && key_start+column<end) {
                        #pragma unroll 2
                        for (int d=0;d<Width && fb+d<Dim;++d) {
                            typename Mma::FragmentA a;
                            typename Mma::FragmentB b;
                            #pragma unroll
                            for (int m=0;m<Rows;++m) a[m]=RowMajor ? qcache[row+m][d] : left[d][row+m];
                            #pragma unroll
                            for (int n=0;n<PairColumns;++n) b[n]=RowMajor ? kcache[column+n*KeyStride][d] : right[d][column+n*KeyStride];
                            Mma{}(scores, a, b, scores);
                        }
                    }
                    if constexpr (Dim>Width || RowMajor) __syncthreads();
                    else __syncwarp();
                }
                #pragma unroll
                for (int m=0;m<Rows;++m) {
                    int64_t query_index=query_start+row+m;
                    float tile_max=-INFINITY;
                    #pragma unroll
                    for (int n=0;n<PairColumns;++n) {
                        int64_t key_index=key_start+column+n*KeyStride;
                        float s=-INFINITY;
                        if (query_index<end && key_index<end && (!Causal || key_index<=query_index)) {
                            s=scores[m*PairColumns+n]*scale;
                            if constexpr (Kind==1) s+=bias.load(key_index*bias.strides[0]+head*bias.strides[1]);
                            if constexpr (Kind==2) s+=bias.load(int64_t(segment)*bias.strides[0]+head*bias.strides[1]+(query_index-begin)*bias.strides[2]+(key_index-begin)*bias.strides[3]);
                        }
                        scores[m*PairColumns+n]=s;
                        tile_max=fmaxf(tile_max, s);
                    }
                    #pragma unroll
                    for (int shift=1;shift<Lanes;shift*=2) tile_max=fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, shift, Lanes));
                    float updated=fmaxf(maximum[m], tile_max);
                    float alpha=key_start!=begin && updated>-INFINITY ? __expf(maximum[m]-updated) : 0;
                    float sum=0;
                    #pragma unroll
                    for (int n=0;n<PairColumns;++n) {
                        float p=updated>-INFINITY ? __expf(scores[m*PairColumns+n]-updated) : 0;
                        left[column+n*KeyStride][row+m]=p;
                        sum+=p;
                    }
                    #pragma unroll
                    for (int shift=1;shift<Lanes;shift*=2) sum+=__shfl_xor_sync(0xffffffff, sum, shift, Lanes);
                    denominator[m]=denominator[m]*alpha+sum;
                    maximum[m]=updated;
                    if (key_start!=begin) {
                        #pragma unroll
                        for (int c=0;c<Chunks;++c) {
                            #pragma unroll
                            for (int n=0;n<Columns;++n) accumulator[c][m*Columns+n]*=alpha;
                        }
                    }
                }
                __syncthreads();
                if constexpr (!LOW_PRECISION && Contiguous && ValueDim%4==0 && Width%4==0) {
                    for (int i=tid*4;i<Tile*Width;i+=THREADS*4) {
                        int r=i/Width, d=i%Width;
                        float4 v={0, 0, 0, 0};
                        if (key_start+r<end && vb+d<ValueDim) v=*reinterpret_cast<const float4*>(value.data+((key_start+r)*HEADS+head)*ValueDim+vb+d);
                        if constexpr (RowMajor) *reinterpret_cast<float4*>(&kcache[r][d])=v;
                        else {
                            right[d][r]=v.x;
                            right[d+1][r]=v.y;
                            right[d+2][r]=v.z;
                            right[d+3][r]=v.w;
                        }
                    }
                } else {
                    for (int i=tid;i<Tile*Width;i+=THREADS) {
                        int r=i/Width, d=i%Width;
                        float v=key_start+r<end && vb+d<ValueDim ? value.template load<ValueDim, Contiguous>(key_start+r, head, vb+d) : 0;
                        if constexpr (RowMajor) kcache[r][d]=v;
                        else right[d][r]=v;
                    }
                }
                __syncthreads();
                int keys=int(end-key_start<Tile ? end-key_start : Tile);
                if (query_start+row<end) {
                    #pragma unroll 2
                    for (int k=0;k<keys;++k) {
                        typename ValueMma::FragmentA a;
                        #pragma unroll
                        for (int m=0;m<Rows;++m) a[m]=left[k][row+m];
                        #pragma unroll
                        for (int c=0;c<Chunks;++c) {
                            typename ValueMma::FragmentB b;
                            #pragma unroll
                            for (int n=0;n<Columns;++n) {
                                int d=lane*Columns+c*Lanes*Columns+n;
                                b[n]=d<Width ? (RowMajor ? kcache[k][d] : right[d][k]) : 0;
                            }
                            ValueMma{}(accumulator[c], a, b, accumulator[c]);
                        }
                    }
                }
                if (key_start+Tile<end && (!Causal || key_start+Tile<=query_start+Tile-1) || vb+ValueTile<ValueDim || query_start+FORWARD_WORKERS*Tile<end) __syncthreads();
            }
            #pragma unroll
            for (int m=0;m<Rows;++m) if (query_start+row+m<end) {
                int64_t index=(query_start+row+m)*HEADS+head;
                float inverse=denominator[m]>0 ? 1.0f/denominator[m] : 0;
                if (lane==0) {
                    state[index*2]=maximum[m];
                    state[index*2+1]=inverse;
                }
                #pragma unroll
                for (int c=0;c<Chunks;++c) {
                    int d=vb+lane*Columns+c*Lanes*Columns;
                    if constexpr (!LOW_PRECISION && Columns%4==0 && ValueDim%4==0) {
                        #pragma unroll
                        for (int n=0; n<Columns; n+=4) if (d+n<ValueDim) {
                            float4 result;
                            result.x=accumulator[c][m*Columns+n]*inverse;
                            result.y=accumulator[c][m*Columns+n+1]*inverse;
                            result.z=accumulator[c][m*Columns+n+2]*inverse;
                            result.w=accumulator[c][m*Columns+n+3]*inverse;
                            *reinterpret_cast<float4*>(output+index*ValueDim+d+n)=result;
                        }
                    } else if constexpr (!LOW_PRECISION && Columns%2==0 && ValueDim%2==0) {
                        #pragma unroll
                        for (int n=0; n<Columns; n+=2) if (d+n<ValueDim)
                            *reinterpret_cast<float2*>(output+index*ValueDim+d+n)=make_float2(accumulator[c][m*Columns+n]*inverse, accumulator[c][m*Columns+n+1]*inverse);
                    } else {
                        #pragma unroll
                        for (int n=0;n<Columns;++n) if (d+n<ValueDim) output[index*ValueDim+d+n]=accumulator[c][m*Columns+n]*inverse;
                    }
                }
            }
        }
    }
}
"""


BINDING_SOURCE = r"""
void attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    c10::optional<torch::Tensor> b, torch::Tensor csr, torch::Tensor output,
    torch::Tensor state, double scale) {
    c10::cuda::CUDAGuard device_guard(q.device());
    Tensor3 query{reinterpret_cast<Storage*>(q.data_ptr()), q.stride(0), q.stride(1), q.stride(2)};
    Tensor3 key{reinterpret_cast<Storage*>(k.data_ptr()), k.stride(0), k.stride(1), k.stride(2)};
    Tensor3 value{reinterpret_cast<Storage*>(v.data_ptr()), v.stride(0), v.stride(1), v.stride(2)};
    Bias bias{};
    if (b.has_value()) {
        bias.data = reinterpret_cast<Storage*>(b->data_ptr());
        bias.fp32 = b->scalar_type() == at::kFloat;
        for (int i = 0; i < b->dim(); ++i) bias.strides[i] = b->stride(i);
    }
    dim3 grid((csr.numel() - 1)*(FORWARD_ADJACENT ? FORWARD_WORKERS:1), q.size(1), FORWARD_ADJACENT ? 1:FORWARD_WORKERS);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    bool aligned = ((reinterpret_cast<uintptr_t>(query.data) |
                     reinterpret_cast<uintptr_t>(key.data) |
                     reinterpret_cast<uintptr_t>(value.data)) & 15) == 0;
    if (csr.scalar_type() == at::kInt) {
        if (aligned) {
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int32_t><<<grid, THREADS, SHARED_BYTES, stream>>>(query, key, value, bias,
                    csr.data_ptr<int32_t>(), csr.stride(0), reinterpret_cast<Storage*>(output.data_ptr()),
                    state.data_ptr<float>(), float(scale));
        } else {
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int32_t><<<grid, THREADS, SHARED_BYTES, stream>>>(query, key, value, bias,
                    csr.data_ptr<int32_t>(), csr.stride(0), reinterpret_cast<Storage*>(output.data_ptr()),
                    state.data_ptr<float>(), float(scale));
        }
    } else {
        if (aligned) {
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int64_t><<<grid, THREADS, SHARED_BYTES, stream>>>(query, key, value, bias,
                    csr.data_ptr<int64_t>(), csr.stride(0), reinterpret_cast<Storage*>(output.data_ptr()),
                    state.data_ptr<float>(), float(scale));
        } else {
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int64_t><<<grid, THREADS, SHARED_BYTES, stream>>>(query, key, value, bias,
                    csr.data_ptr<int64_t>(), csr.stride(0), reinterpret_cast<Storage*>(output.data_ptr()),
                    state.data_ptr<float>(), float(scale));
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward);
#if SHARED_BYTES > 0
    m.def("configure", []() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int32_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
#if !LOW_PRECISION && FORWARD_TILE >= 48
        C10_CUDA_CHECK(cudaFuncSetAttribute(FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int32_t>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
#endif
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int64_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
#if !LOW_PRECISION && FORWARD_TILE >= 48
        C10_CUDA_CHECK(cudaFuncSetAttribute(FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, CONTIGUOUS, int64_t>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
#endif
        if constexpr (CONTIGUOUS) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int32_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
#if !LOW_PRECISION && FORWARD_TILE >= 48
        C10_CUDA_CHECK(cudaFuncSetAttribute(FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int32_t>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
#endif
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int64_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
#if !LOW_PRECISION && FORWARD_TILE >= 48
        C10_CUDA_CHECK(cudaFuncSetAttribute(FORWARD_KERNEL<DIM, VALUE_DIM, BIAS_KIND, CAUSAL, false, int64_t>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
#endif
        }
    });
#endif
}
"""

@lru_cache(maxsize=None)
def get_forward_kernel(dim, value_dim, heads, bias_kind, causal, contiguous, device,
                       dtype=torch.float32, tile=32, query_workers=1, lower=0, upper=0):
    arch = get_cuda_arch(device)
    properties = get_device_properties(device)
    shared_capacity = properties.shared_memory_per_multiprocessor
    shared_bytes = 2 * tile * (
        2 * ((dim + 15) // 16 * 16 + 8) + (value_dim + 7) // 8 * 8 + tile + 12
    )
    # Bound tensor-core fragments and fall back to SIMT for wider features.
    tensor_core = dtype != torch.float32 and max(dim, value_dim) <= 256
    if tensor_core and shared_bytes > 48 * 1024:
        tensor_core = shared_bytes <= properties.shared_memory_per_block_optin
    dynamic_bytes = shared_bytes if tensor_core and shared_bytes > 48 * 1024 else 0
    feature_limit = 64
    if dtype == torch.float32:
        feature_limit = ((shared_capacity // 4 - 1024) // (2 * 33 * 4) // 32) * 32
    if tensor_core:
        kernel_name = "attention_forward_kernel"
        threads = 2 * tile
        minimum_blocks = min(8, shared_capacity // (shared_bytes + 1024))
    else:
        kernel_name = "attention_forward_simt_kernel"
        threads = 256 if dtype == torch.float32 and tile == 32 else 128
        features = max(dim, value_dim)
        width = max(tile, (features + 31) // 32 * 32 if features <= feature_limit else 64)
        row_major = dtype == torch.float32 and tile >= 48
        dynamic_bytes = (2 * tile * (width + 4) if row_major else 2 * width * (tile + 1)) * 4
        minimum_blocks = min(6 if tile == 48 else 4, shared_capacity // (dynamic_bytes + 1024))
    adjacent_workers = query_workers > 1 and (dtype != torch.float32 or tile >= 48)
    constants = (
        dtype_defines(dtype) +
        f"#define HEADS {heads}\n#define DIM {dim}\n#define VALUE_DIM {value_dim}\n"
        f"#define BIAS_KIND {bias_kind}\n#define CAUSAL {str(causal).lower()}\n"
        f"#define CONTIGUOUS {str(contiguous).lower()}\n"
        f"#define FORWARD_MIN_LENGTH {lower}\n#define FORWARD_MAX_LENGTH {upper}\n"
        f"#define FORWARD_FEATURE_LIMIT {feature_limit}\n#define FORWARD_TILE {tile}\n"
        f"#define FORWARD_KERNEL {kernel_name}\n#define THREADS {threads}\n"
        f"#define FORWARD_WORKERS {query_workers}\n#define FORWARD_ADJACENT {int(adjacent_workers)}\n"
        f"#define SHARED_BYTES {dynamic_bytes}\n#define MIN_BLOCKS_PER_SM {minimum_blocks}\n"
    )
    source = FORWARD_SOURCE if tensor_core else FORWARD_SIMT_SOURCE
    module = load_cuda_extension(
        "attention_forward", constants + CUDA_SOURCE + source + BINDING_SOURCE,
        arch=arch, minimal_headers=True, use_fast_math=False,
    )
    if dynamic_bytes:
        with torch.cuda.device(device):
            module.configure()
    return module.forward


@lru_cache(maxsize=None)
def get_forward_kernels(dim, value_dim, heads, bias_kind, causal, contiguous, device,
                        dtype, large):
    ranges = ((0, 0, 32),)
    if large and dtype == torch.float32:
        ranges = ((0, 48, 48), (49, 0, 64))
    elif large and max(dim, value_dim) <= 256:
        shared64 = 128 * (2 * ((dim + 15) // 16 * 16 + 8)
                          + (value_dim + 7) // 8 * 8 + 76)
        capacity = get_device_properties(device).shared_memory_per_multiprocessor
        if 2 * (shared64 + 1024) <= capacity:
            ranges = ((0, 32, 32), (33, 0, 64))
    kernels = []
    for lower, upper, tile in ranges:
        workers = 1
        if dtype == torch.float32 and not (upper and upper <= tile):
            workers = 2 if large else 4
        kernels.append(get_forward_kernel(
            dim, value_dim, heads, bias_kind, causal, contiguous, device, dtype,
            tile, workers, lower, upper,
        ))
    return tuple(kernels)


def attention_forward(inputs, rowptr, config):
    q, k, v, bias = inputs[:4]
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    state = torch.empty((*v.shape[:2], 2), device=v.device, dtype=torch.float32)
    if v.shape[0]:
        kernels = get_forward_kernels(
            config[0], config[1], config[2], config[6], config[4],
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous(),
            v.device, v.dtype, v.shape[0] > 32 * (rowptr.numel() - 1),
        )
        for kernel in kernels:
            kernel(q, k, v, bias, rowptr, output, state, config[3])
    return output, state
