"""Fused attention, optional envelope and CSR sum; no edge-output workspace."""
from functools import lru_cache
import torch
from ....utils.extension import get_device_properties
from .utils import load_kernel

SOURCE = r"""


constexpr int SHARED_STRIDE = DIM < 32 && VALUE_DIM < 32 ? 34 : 33;
constexpr int LEFT_ROWS = RETURN_OUT ? 32 : (DIM < 32 ? (DIM + 3) / 4 * 4 : 32);
constexpr int RIGHT_CHANNELS = DIM > VALUE_DIM ? DIM : VALUE_DIM;
constexpr int RIGHT_ROWS = RIGHT_CHANNELS < 32 ? (RIGHT_CHANNELS + 3) / 4 * 4 : 32;
constexpr int SCORE_CACHE_KEYS = 64;
constexpr int VALUE_CHUNKS = VALUE_DIM <= 128 ? (VALUE_DIM + 31) / 32 : 2;

// Vector loads are optional; arbitrary strides and unaligned storage use the same tile.
template<int Channels, int Rows>
__device__ void load_tile(const View &x, int64_t start, int64_t end, int head, int fb,
                          float (&tile)[Rows][SHARED_STRIDE]) {
    bool aligned = (reinterpret_cast<uintptr_t>(x.data) % 16 == 0);
    if constexpr (!CONTIGUOUS)
        aligned = aligned && x.stride[2] == 1 && (x.stride[0] % 4 == 0) && (x.stride[1] % 4 == 0);
    else
        aligned = aligned && Channels % 4 == 0;
    constexpr int Width = Channels < 32 ? (Channels + 3) / 4 * 4 : 32;
    for (int i = threadIdx.x * 4; i < 32 * Width; i += 512) {
        int row = i / Width, c = i % Width;
        float4 values = {0.f, 0.f, 0.f, 0.f};
        if (start + row < end) {
            int64_t base;
            if constexpr (CONTIGUOUS) base = ((start + row) * HEADS + head) * Channels + fb + c;
            else base = x.index(start + row, head, fb + c);
            int64_t stride = CONTIGUOUS ? 1 : x.stride[2];
            if (aligned && fb + c + 3 < Channels) {
                values = *reinterpret_cast<const float4 *>(x.data + base);
            } else {
                values.x = fb + c < Channels ? x.data[base + 0 * stride] : 0.f;
                values.y = fb + c + 1 < Channels ? x.data[base + 1 * stride] : 0.f;
                values.z = fb + c + 2 < Channels ? x.data[base + 2 * stride] : 0.f;
                values.w = fb + c + 3 < Channels ? x.data[base + 3 * stride] : 0.f;
            }
        }
        tile[c][row] = values.x;
        tile[c + 1][row] = values.y;
        tile[c + 2][row] = values.z;
        tile[c + 3][row] = values.w;
    }
}

__device__ void score_tile(const Params &p, int64_t atom, int64_t begin, int64_t end,
                          int64_t query_start, int64_t key_start, int head,
                          float (&left)[LEFT_ROWS][SHARED_STRIDE], float (&right)[RIGHT_ROWS][SHARED_STRIDE], float (&scores)[2][4]) {
    int tid = threadIdx.x, row = (tid / 8) * 2, column = (tid % 8) * 4;
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int j = 0; j < 4; ++j) scores[m][j] = 0.f;
    for (int fb = 0; fb < DIM; fb += 32) {
        load_tile<DIM>(p.in[0], query_start, end, head, fb, left);
        load_tile<DIM>(p.in[1], key_start, end, head, fb, right);
        __syncthreads();
        if (query_start + row < end && key_start + column < end) {
        #pragma unroll 4
        for (int c = 0; c < 32 && fb + c < DIM; ++c) {
            float q0 = left[c][row], q1 = left[c][row + 1];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float key = right[c][column + j];
                scores[0][j] = fmaf(q0, key, scores[0][j]);
                scores[1][j] = fmaf(q1, key, scores[1][j]);
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t q = query_start + row + m, k = key_start + column + j;
            float value = -INFINITY;
            if (q < end && k < end && (!CAUSAL || k <= q)) {
                value = scores[m][j] * p.scale;
                if constexpr (BIAS_KIND == 1) value += p.in[3].get(k, head);
                if constexpr (BIAS_KIND == 2) value += p.in[3].get(atom, head, q - begin, k - begin);
            }
            scores[m][j] = value;
        }
}

template<bool Short>
__device__ __forceinline__ void process_atom(const Params &p, int64_t atom, int head,
    float (&left)[LEFT_ROWS][SHARED_STRIDE], float (&right)[RIGHT_ROWS][SHARED_STRIDE], float (&alpha)[32], float (&score_cache)[SCORE_CACHE_KEYS][33], int query_worker, int query_workers) {
    int64_t begin = p.offset(atom), end = p.offset(atom + 1);
    int tid = threadIdx.x;
    bool cache_scores = !Short && end - begin <= SCORE_CACHE_KEYS;
    int row = (tid / 8) * 2, column = (tid % 8) * 4;
    for (int vb = 0; vb < VALUE_DIM; vb += VALUE_CHUNKS * 32) {
        float aggregate[VALUE_CHUNKS] = {};
        int64_t query_tiles = Short ? int64_t(end > begin) : (end - begin + 31) / 32;
        for (int64_t query_tile = query_worker; query_tile < query_tiles; query_tile += query_workers) {
            int64_t query_start = begin + query_tile * 32;
            int64_t key_tiles = Short ? 1 : (end - begin + 31) / 32;
            if constexpr (CAUSAL)
                key_tiles = key_tiles < query_tile + 1 ? key_tiles : query_tile + 1;
            float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {};
            float scores[2][4], output[VALUE_CHUNKS][2][4] = {};
            int64_t last_key = begin;
            // Keep the last score tile live: short segments never recompute QK.
            for (int64_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                int64_t key = begin + key_tile * 32;
                score_tile(p, atom, begin, end, query_start, key, head, left, right, scores);
                if (cache_scores) {
                    #pragma unroll
                    for (int m = 0; m < 2; ++m)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            score_cache[key_tile * 32 + column + j][row + m] = scores[m][j];
                }
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    float tile_max = -INFINITY;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tile_max = fmaxf(tile_max, scores[m][j]);
                    #pragma unroll
                    for (int d = 1; d < 8; d *= 2)
                        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, d, 8));
                    float next = Short ? tile_max : fmaxf(maximum[m], tile_max);
                    float correction = Short ? 0.f : (isfinite(next) ? __expf(maximum[m] - next) : 1.f);
                    float sum = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        scores[m][j] = isfinite(scores[m][j]) ? __expf(scores[m][j] - next) : 0.f;
                        sum += scores[m][j];
                    }
                    #pragma unroll
                    for (int d = 1; d < 8; d *= 2)
                        sum += __shfl_xor_sync(0xffffffff, sum, d, 8);
                    denominator[m] = Short ? sum : fmaf(denominator[m], correction, sum);
                    maximum[m] = next;
                }
                last_key = key;
            }
            float inverse[2], envelope[2];
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                inverse[m] = denominator[m] > 0.f ? 1.f / denominator[m] : 0.f;
                int64_t q = query_start + row + m;
                envelope[m] = q < end ? (p.in[4].data ? p.in[4].get(q) : 1.f) : 0.f;
            }
            for (int64_t key_tile = key_tiles; key_tile > 0; --key_tile) {
                int64_t key = begin + (key_tile - 1) * 32;
                if (!Short && key != last_key) {
                    if (cache_scores) {
                        #pragma unroll
                        for (int m = 0; m < 2; ++m)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j)
                                scores[m][j] = score_cache[(key_tile - 1) * 32 + column + j][row + m];
                    } else {
                        score_tile(p, atom, begin, end, query_start, key, head, left, right, scores);
                    }
                    #pragma unroll
                    for (int m = 0; m < 2; ++m)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            scores[m][j] = isfinite(scores[m][j]) ? __expf(scores[m][j] - maximum[m]) : 0.f;
                }
                // Reduce eight query contributions inside each warp before staging.
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float p0 = scores[0][j] * inverse[0], p1 = scores[1][j] * inverse[1];
                    if constexpr (RETURN_OUT) {
                        left[column + j][row] = p0;
                        left[column + j][row + 1] = p1;
                    }
                    float partial = fmaf(p0, envelope[0], p1 * envelope[1]);
                    partial += __shfl_xor_sync(0xffffffff, partial, 8);
                    partial += __shfl_xor_sync(0xffffffff, partial, 16);
                    if (tid % 32 < 8) right[tid / 32][column + j] = partial;
                }
                __syncthreads();
                if (tid < 32)
                    alpha[tid] = (right[0][tid] + right[1][tid]) + (right[2][tid] + right[3][tid]);
                __syncthreads();
                #pragma unroll
                for (int chunk = 0; chunk < VALUE_CHUNKS; ++chunk) {
                    load_tile<VALUE_DIM>(p.in[2], key, end, head, vb + chunk * 32, right);
                    __syncthreads();
                    if (tid < 32 && vb + chunk * 32 + tid < VALUE_DIM) {
                        float value = aggregate[chunk];
                        #pragma unroll 4
                        for (int k = 0; k < 32; ++k) value = fmaf(alpha[k], right[tid][k], value);
                        aggregate[chunk] = value;
                    }
                    if constexpr (RETURN_OUT) {
                        #pragma unroll 4
                        for (int k = 0; k < 32; ++k) {
                            float p0 = left[k][row], p1 = left[k][row + 1];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                float value = vb + chunk * 32 + column + j < VALUE_DIM ? right[column + j][k] : 0.f;
                                output[chunk][0][j] = fmaf(p0, value, output[chunk][0][j]);
                                output[chunk][1][j] = fmaf(p1, value, output[chunk][1][j]);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            if constexpr (RETURN_OUT) {
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    int64_t q = query_start + row + m;
                    if (q < end) {
                        #pragma unroll
                        for (int chunk = 0; chunk < VALUE_CHUNKS; ++chunk)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                int feature = vb + chunk * 32 + column + j;
                                if (feature < VALUE_DIM)
                                    p.out[1].data[(q * HEADS + head) * VALUE_DIM + feature] = output[chunk][m][j];
                            }
                    }
                }
            }
        }
        if (tid < 32) {
            #pragma unroll
            for (int chunk = 0; chunk < VALUE_CHUNKS; ++chunk) {
                int feature = vb + chunk * 32 + tid;
                if (feature < VALUE_DIM)
                    {
                    int64_t offset=(atom*HEADS+head)*VALUE_DIM+feature;
                    if(!Short && query_workers>1) atomicAdd(p.out[0].data+offset,aggregate[chunk]);
                    else p.out[0].data[offset]=aggregate[chunk];
                    }
            }
        }
        __syncthreads();
    }
}
__global__ void initialize_nodes(const __grid_constant__ Params p) {
    int64_t atom=blockIdx.x;
    if(p.offset(atom+1)-p.offset(atom)<=32) return;
    for(int t=threadIdx.x;t<HEADS*VALUE_DIM;t+=blockDim.x)
        p.out[0].data[atom*HEADS*VALUE_DIM+t]=0.f;
}
__global__ __launch_bounds__(128, (RETURN_OUT || !CONTIGUOUS) ? 4 : (LEFT_ROWS < 32 && RIGHT_ROWS < 32 ? 10 : 8))
void forward_kernel(const __grid_constant__ Params p) {
    __shared__ float left[LEFT_ROWS][SHARED_STRIDE], right[RIGHT_ROWS][SHARED_STRIDE], alpha[32];
    __shared__ float score_cache[SCORE_CACHE_KEYS][33];
    int64_t atom = blockIdx.x;
    int64_t length=p.offset(atom+1)-p.offset(atom);
    int workers=int(gridDim.z),worker=int(blockIdx.z);
    if(worker>=(length>0?(length+31)/32:1)) return;
    if (p.offset(atom + 1) - p.offset(atom) <= 32)
        process_atom<true>(p, atom, blockIdx.y, left, right, alpha, score_cache,0,1);
    else
        process_atom<false>(p, atom, blockIdx.y, left, right, alpha, score_cache,worker,workers);
}
void run(std::vector<std::optional<at::Tensor>> inputs, at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs, double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if (csr.numel() > 1) {
        Params p=params(inputs,outputs,csr,scale);
        int workers=inputs[0]->size(0)>(csr.numel()-1)*32?4:1;
        if(workers>1) {
            initialize_nodes<<<csr.numel()-1,128,0,at::cuda::getCurrentCUDAStream()>>>(p);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        forward_kernel<<<dim3(csr.numel() - 1, HEADS,workers), 128, 0,
                         at::cuda::getCurrentCUDAStream()>>>(params(inputs, outputs, csr, scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""


LOWP_SOURCE = r"""
using Lowp = std::conditional_t<LOWP_BF16, __nv_bfloat16, __half>;
namespace wmma = nvcuda::wmma;
__device__ Lowp to_lowp(float x) {
#if LOWP_BF16
    return __float2bfloat16_rn(x);
#else
    return __float2half_rn(x);
#endif
}

__device__ float from_lowp(Lowp x) {
#if LOWP_BF16
    return __bfloat162float(x);
#else
    return __half2float(x);
#endif
}
template<int Channels>
__device__ Lowp feature(const View &x, int64_t edge, int head, int feature) {
    int64_t offset;
    if constexpr (CONTIGUOUS) offset = (edge * HEADS + head) * Channels + feature;
    else offset = x.index(edge,head,feature);
    return static_cast<const Lowp *>(x.data)[offset];
}

template<int Channels>
__device__ void load_feature_tile(const View &x, int64_t start, int64_t end,
                                  int head, int fb, Lowp *tile) {
    bool aligned = reinterpret_cast<uintptr_t>(x.data) % 8 == 0;
    if constexpr (CONTIGUOUS) aligned = aligned && Channels % 4 == 0;
    else aligned = aligned && x.stride[2] == 1 && x.stride[0] % 4 == 0 && x.stride[1] % 4 == 0;
    for (int i=threadIdx.x*4;i<1024;i+=512) {
        int row=i/32,col=i%32;
        uint2 values={0,0};
        if (start+row<end && fb+col<Channels) {
            if (aligned && fb+col+3<Channels) {
                int64_t offset = CONTIGUOUS ? ((start+row)*HEADS+head)*Channels+fb+col : x.index(start+row,head,fb+col);
                values=*reinterpret_cast<const uint2 *>(static_cast<const Lowp *>(x.data)+offset);
            } else {
                Lowp *v=reinterpret_cast<Lowp *>(&values);
                #pragma unroll
                for (int j=0;j<4;++j) v[j]=fb+col+j<Channels ? feature<Channels>(x,start+row,head,fb+col+j) : to_lowp(0.f);
            }
        }
        *reinterpret_cast<uint2 *>(tile+i)=values;
    }
}

__device__ void lowp_score_tile(const Params &p, int64_t qb, int64_t kb, int64_t end,
                                int head, Lowp *left, Lowp *right, float *scores) {
    int tid=threadIdx.x,warp=tid/32;
    int tile_row=warp/2*16,tile_col=warp%2*16;
    wmma::fragment<wmma::accumulator,16,16,16,float> score;
    wmma::fill_fragment(score, 0.f);
    for (int fb = 0; fb < DIM; fb += 32) {
        load_feature_tile<DIM>(p.in[0],qb,end,head,fb,left);
        load_feature_tile<DIM>(p.in[1],kb,end,head,fb,right);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < (DIM <= 16 ? 16 : 32); k += 16) {
            wmma::fragment<wmma::matrix_a,16,16,16,Lowp,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,Lowp,wmma::col_major> b;
            wmma::load_matrix_sync(a, left + tile_row * 32 + k, 32);
            wmma::load_matrix_sync(b, right + tile_col * 32 + k, 32);
            wmma::mma_sync(score, a, b, score);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(scores + tile_row * 32 + tile_col, score, 32, wmma::mem_row_major);
    __syncthreads();

}

#include <cutlass/arch/mma_sm80.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#if LOWP_BF16
using NodeElement = cutlass::bfloat16_t;
#else
using NodeElement = cutlass::half_t;
#endif
constexpr int NodeWidth = ((DIM > VALUE_DIM ? DIM : VALUE_DIM) + 15) / 16 * 16;
constexpr int NodePitch = NodeWidth + 8;
template<int Tile> struct NodeForwardLayout {
    static constexpr int FeatureWords = Tile * NodePitch / 2;
    static constexpr int LeftWords = FeatureWords > Tile * Tile ? FeatureWords : Tile * Tile;
    static constexpr int Bytes = (LeftWords + FeatureWords + 3 * Tile) * sizeof(float);
};
template<int Tile, int Channels>
__device__ void load_node_forward(View const& input, NodeElement (*cache)[NodePitch],
    int64_t begin, int64_t end, int head) {
    bool aligned = CONTIGUOUS && Channels % 8 == 0 && reinterpret_cast<uintptr_t>(input.data) % 16 == 0;
    for(int i=threadIdx.x*8;i<Tile*NodeWidth;i+=blockDim.x*8) {
        int row=i/NodeWidth,c=i%NodeWidth;
        if(aligned) {
            uint4 value={0,0,0,0};
            if(begin+row<end && c<Channels)
                value=*reinterpret_cast<const uint4*>(static_cast<Lowp const*>(input.data)+((begin+row)*HEADS+head)*Channels+c);
            *reinterpret_cast<uint4*>(&cache[row][c])=value;
        } else {
            #pragma unroll
            for(int j=0;j<8;++j)
                cache[row][c+j]=NodeElement(begin+row<end && c+j<Channels?input.get(begin+row,head,c+j):0.f);
        }
    }
}

// QK fragments stay live until all reads finish; the query cache then stores P.
// Contract sum_q(P * envelope) first, avoiding an edge-output PV matrix.
template<int Tile, int Minimum>
__global__ __launch_bounds__(2*Tile,2)
void node_forward_lowp(const __grid_constant__ Params p) {
    using L=NodeForwardLayout<Tile>;
    using Mma=cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
        NodeElement,cutlass::layout::RowMajor,NodeElement,cutlass::layout::ColumnMajor,
        float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if(end-begin<=Minimum || end-begin>Tile) return;
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.y;
    int row=warp*16+lane/4,column=(lane%4)*2;
    extern __shared__ __align__(16) float memory[];
    auto queries=reinterpret_cast<NodeElement (*)[NodePitch]>(memory);
    auto keys=reinterpret_cast<NodeElement (*)[NodePitch]>(memory+L::LeftWords);
    float* probability=memory;
    float* alpha=memory+L::LeftWords+L::FeatureWords;
    float* partial=alpha+Tile;
    load_node_forward<Tile,DIM>(p.in[0],queries,begin,end,head);
    load_node_forward<Tile,DIM>(p.in[1],keys,begin,end,head);
    __syncthreads();
    typename Mma::FragmentC scores[Tile/8];
    #pragma unroll
    for(int n=0;n<Tile/8;++n) scores[n].clear();
    #pragma unroll 1
    for(int d=0;d<(DIM+15)/16*16;d+=16) {
        typename Mma::FragmentA a;
        #pragma unroll
        for(int i=0;i<Mma::FragmentA::kElements;++i)
            a[i]=queries[row+(i/2%2)*8][d+(lane%4)*2+i%2+(i/4)*8];
        #pragma unroll
        for(int n=0;n<Tile/8;++n) {
            typename Mma::FragmentB b;
            #pragma unroll
            for(int i=0;i<Mma::FragmentB::kElements;++i)
                b[i]=keys[n*8+lane/4][d+(lane%4)*2+i%2+(i/2)*8];
            Mma{}(scores[n],a,b,scores[n]);
        }
    }
    __syncthreads();
    #pragma unroll
    for(int m=0;m<2;++m) {
        int64_t q=begin+row+m*8;
        float maximum=-INFINITY;
        #pragma unroll
        for(int n=0;n<Tile/8;++n)
            #pragma unroll
            for(int j=0;j<2;++j) {
                int64_t key=begin+n*8+column+j;
                float s=-INFINITY;
                if(q<end && key<end && (!CAUSAL || key<=q)) {
                    s=scores[n][m*2+j]*p.scale;
                    if constexpr(BIAS_KIND==1) s+=p.in[3].get(key,head);
                    if constexpr(BIAS_KIND==2) s+=p.in[3].get(atom,head,q-begin,key-begin);
                }
                scores[n][m*2+j]=s; maximum=fmaxf(maximum,s);
            }
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1,4));
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2,4));
        float sum=0.f;
        #pragma unroll
        for(int n=0;n<Tile/8;++n)
            #pragma unroll
            for(int j=0;j<2;++j) {
                float weight=isfinite(scores[n][m*2+j])?__expf(scores[n][m*2+j]-maximum):0.f;
                scores[n][m*2+j]=weight;sum+=weight;
            }
        sum+=__shfl_xor_sync(0xffffffff,sum,1,4);
        sum+=__shfl_xor_sync(0xffffffff,sum,2,4);
        float env=q<end?(p.in[4].data?p.in[4].get(q):1.f):0.f;
        float factor=sum>0.f?env/sum:0.f;
        #pragma unroll
        for(int n=0;n<Tile/8;++n)
            #pragma unroll
            for(int j=0;j<2;++j)
                probability[(row+m*8)*Tile+n*8+column+j]=scores[n][m*2+j]*factor;
    }
    __syncthreads();
    int key=tid%Tile;
    float weight=0.f;
    for(int q=tid/Tile;q<Tile;q+=2) weight+=probability[q*Tile+key];
    partial[tid]=weight;
    __syncthreads();
    if(tid<Tile) alpha[tid]=partial[tid]+partial[tid+Tile];
    load_node_forward<Tile,VALUE_DIM>(p.in[2],keys,begin,end,head);
    __syncthreads();
    for(int c=tid;c<VALUE_DIM;c+=blockDim.x) {
        float result=0.f;
        for(int k=0;k<Tile;++k) result=fmaf(alpha[k],float(keys[k][c]),result);
        p.out[0].store((atom*HEADS+head)*VALUE_DIM+c,result);
    }
}

// FlashAttention's online normalization: only a 32-row score/output tile is
// resident. QK and PV use FP16/BF16 Tensor Cores with FP32 accumulators.
__global__ __launch_bounds__(128)
void forward_lowp_kernel(const __grid_constant__ Params p) {
    __shared__ __align__(32) Lowp left[32 * 32], right[32 * 32], probability[32 * 32];
    __shared__ __align__(32) float scores[32 * 32], output[32 * 32];
    __shared__ float maximum[32], denominator[32], rescale[32], envelope[32], partial[128];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int tile_row = warp / 2 * 16, tile_col = warp % 2 * 16;
    int64_t atom = blockIdx.x;
    int head = blockIdx.y;
    int64_t begin = p.offset(atom), end = p.offset(atom + 1);
    if constexpr(!RETURN_OUT && NODE_LONG_TILE>=64)
        if(end-begin>32 && end-begin<=NODE_LONG_TILE) return;

    // Node-only attention contracts weighted query probabilities first. The
    // remaining alpha*V operation is a vector reduction, not a dense PV matrix.
    if constexpr (!RETURN_OUT) {
        if (end-begin <= 32) {
            if (tid < 32) envelope[tid] = begin+tid < end && p.in[4].data ? p.in[4].get(begin+tid) : 1.f;
            lowp_score_tile(p,begin,begin,end,head,left,right,scores);
            for (int r=tid/16;r<32;r+=8) {
                float values[2]={-INFINITY,-INFINITY};
                #pragma unroll
                for (int j=0;j<2;++j) {
                    int col=lane%16+j*16;
                    if (begin+r < end && begin+col < end && (!CAUSAL || col<=r)) {
                        values[j]=scores[r*32+col]*p.scale;
                        if constexpr (BIAS_KIND == 1) values[j]+=p.in[3].get(begin+col,head);
                        if constexpr (BIAS_KIND == 2) values[j]+=p.in[3].get(atom,head,r,col);
                    }
                }
                float m=fmaxf(values[0],values[1]);
                #pragma unroll
                for (int delta=8;delta;delta/=2) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,delta,16));
                float weights[2];
                #pragma unroll
                for (int j=0;j<2;++j) weights[j]=values[j] == -INFINITY ? 0.f : __expf(values[j]-m);
                float sum=weights[0]+weights[1];
                #pragma unroll
                for (int delta=8;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta,16);
                #pragma unroll
                for (int j=0;j<2;++j) scores[r*32+lane%16+j*16] = sum>0.f ? weights[j]/sum*envelope[r] : 0.f;
            }
            __syncthreads();
            if (tid<32) {
                float alpha=0.f;
                for (int r=0;r<32;++r) alpha+=scores[r*32+tid];
                maximum[tid]=alpha;
            }
            __syncthreads();
            for (int vb=0;vb<VALUE_DIM;vb+=32) {
                load_feature_tile<VALUE_DIM>(p.in[2],begin,end,head,vb,right);
                __syncthreads();
                float sum=0.f;
                for (int r=warp;r<32;r+=4)
                    sum=fmaf(from_lowp(right[r*32+lane]),maximum[r],sum);
                partial[tid]=sum;
                __syncthreads();
                if (tid<32 && vb+tid<VALUE_DIM) {
                    float total=(partial[tid]+partial[tid+32])+(partial[tid+64]+partial[tid+96]);
                    static_cast<Lowp *>(p.out[0].data)[(atom*HEADS+head)*VALUE_DIM+vb+tid]=to_lowp(total);
                }
                __syncthreads();
            }
            return;
        }
    }
    for (int vb = 0; vb < VALUE_DIM; vb += 32) {
        float aggregate = 0.f;
        for (int64_t qb = begin; qb < end; qb += 32) {
            if (tid < 32) {
                maximum[tid] = -INFINITY;
                denominator[tid] = 0.f;
                envelope[tid] = qb+tid < end && p.in[4].data ? p.in[4].get(qb+tid) : 1.f;
            }
            for (int i = tid; i < 1024; i += 128) output[i] = 0.f;
            __syncthreads();
            for (int64_t kb = begin; kb < end; kb += 32) {
                lowp_score_tile(p,qb,kb,end,head,left,right,scores);
                // A warp handles a complete row; no row maxima leave the CTA.
                for (int r = warp; r < 32; r += 4) {
                    int64_t q = qb+r, k = kb+lane;
                    float s = -INFINITY;
                    if (q < end && k < end && (!CAUSAL || k <= q)) {
                        s = scores[r*32+lane] * p.scale;
                        if constexpr (BIAS_KIND == 1) s += p.in[3].get(k,head);
                        if constexpr (BIAS_KIND == 2) s += p.in[3].get(atom,head,q-begin,k-begin);
                    }
                    float m = s;
                    #pragma unroll
                    for (int delta=16; delta; delta/=2) m = fmaxf(m,__shfl_xor_sync(0xffffffff,m,delta));
                    m = fmaxf(m,maximum[r]);
                    float correction = maximum[r] == -INFINITY ? 0.f : __expf(maximum[r]-m);
                    float v = s == -INFINITY ? 0.f : __expf(s-m);
                    float sum = v;
                    #pragma unroll
                    for (int delta=16; delta; delta/=2) sum += __shfl_xor_sync(0xffffffff,sum,delta);
                    probability[r*32+lane] = to_lowp(v);
                    if (!lane) {
                        maximum[r] = m;
                        denominator[r] = denominator[r]*correction + sum;
                        rescale[r] = correction;
                    }
                }
                load_feature_tile<VALUE_DIM>(p.in[2],kb,end,head,vb,right);
                __syncthreads();
                for (int i=tid; i<1024; i+=128) output[i] *= rescale[i/32];
                __syncthreads();
                wmma::fragment<wmma::accumulator,16,16,16,float> result;
                wmma::load_matrix_sync(result, output + tile_row*32+tile_col, 32, wmma::mem_row_major);
                #pragma unroll
                for (int k=0;k<32;k+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,Lowp,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,Lowp,wmma::row_major> b;
                    wmma::load_matrix_sync(a, probability + tile_row*32+k, 32);
                    wmma::load_matrix_sync(b, right + k*32+tile_col, 32);
                    wmma::mma_sync(result,a,b,result);
                }
                wmma::store_matrix_sync(output+tile_row*32+tile_col,result,32,wmma::mem_row_major);
                __syncthreads();
            }
            float sum = 0.f;
            if (vb+lane < VALUE_DIM) {
                for (int r=warp;r<32 && qb+r<end;r+=4) {
                    float o = output[r*32+lane]/denominator[r];
                    sum = fmaf(o,envelope[r],sum);
                    if constexpr (RETURN_OUT)
                        static_cast<Lowp *>(p.out[1].data)[((qb+r)*HEADS+head)*VALUE_DIM+vb+lane] = to_lowp(o);
                }
            }
            partial[tid] = sum;
            __syncthreads();
            if (tid < 32) aggregate += (partial[tid]+partial[tid+32])+(partial[tid+64]+partial[tid+96]);
            __syncthreads();
        }
        if (tid < 32 && vb+tid < VALUE_DIM)
            static_cast<Lowp *>(p.out[0].data)[(atom*HEADS+head)*VALUE_DIM+vb+tid] = to_lowp(aggregate);
        __syncthreads();
    }
}
void run(std::vector<std::optional<at::Tensor>> inputs, at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs, double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if (csr.numel()>1) {
        if constexpr(!RETURN_OUT && NODE_LONG_TILE>=64) {
            static thread_local int configured_device=-1;
            int device=inputs[0]->get_device();
            if(configured_device!=device) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_forward_lowp<64,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeForwardLayout<64>::Bytes));
                if constexpr(NODE_LONG_TILE>64)
                    C10_CUDA_CHECK(cudaFuncSetAttribute(node_forward_lowp<NODE_LONG_TILE,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeForwardLayout<NODE_LONG_TILE>::Bytes));
                configured_device=device;
            }
            node_forward_lowp<64,32><<<dim3(csr.numel()-1,HEADS),128,NodeForwardLayout<64>::Bytes,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if constexpr(NODE_LONG_TILE>64) {
                node_forward_lowp<NODE_LONG_TILE,64><<<dim3(csr.numel()-1,HEADS),2*NODE_LONG_TILE,NodeForwardLayout<NODE_LONG_TILE>::Bytes,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
        forward_lowp_kernel<<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
"""


@lru_cache(maxsize=128)
def get_forward_kernel(config, return_attn_out, contiguous, device, dtype, small_grid):
    source = f"#define RETURN_OUT {int(return_attn_out)}\n#define CONTIGUOUS {int(contiguous)}\n" + (SOURCE if dtype == torch.float32 else LOWP_SOURCE)
    if dtype != torch.float32:
        props = get_device_properties(device)
        budget = min(props.shared_memory_per_multiprocessor // 2, props.shared_memory_per_block_optin)
        pitch = (max(config[0], config[1]) + 15) // 16 * 16 + 8
        tile = 32
        for size in range(64, 257, 16):
            feature = size * pitch // 2
            if (max(feature, size * size) + feature + 3 * size) * 4 <= budget:
                tile = size
        if small_grid:
            tile = 32
        source = f"#define NODE_LONG_TILE {tile}\n" + source
    return load_kernel("attention_conv_forward", source, config, device.index, dtype).run


def attention_conv_forward(inputs, csr, config, return_attn_out):
    v = inputs[2]
    output = torch.empty((csr.numel() - 1, *v.shape[1:]), device=v.device, dtype=v.dtype)
    edge_output = torch.empty(v.shape, device=v.device, dtype=v.dtype) if return_attn_out else None
    contiguous = all(x.is_contiguous() for x in inputs[:3])
    small_grid = (v.dtype != torch.float32
                  and (csr.numel() - 1) * config[2] <= 2 * get_device_properties(v.device).multi_processor_count
                  and inputs[0].shape[0] <= 32 * (csr.numel() - 1))
    kernel = get_forward_kernel(config, return_attn_out, contiguous, v.device, v.dtype, small_grid)
    kernel(inputs, csr, (output, edge_output), config[3])
    return output, edge_output
