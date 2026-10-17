"""Second reverse derivatives of attention and envelope-weighted CSR reduction."""
from functools import lru_cache
import torch
from ....utils.extension import get_device_properties
from .utils import allocate, load_kernel
from .backward import TILED_SOURCE

LOWP_SOURCE = r"""
__device__ __forceinline__ float attention_feature_load(View const &v, int64_t offset) {
#ifdef LOWP_BF16
    if (!v.data) return 0.f;
#if LOWP_BF16
    return __bfloat162float(static_cast<__nv_bfloat16 const *>(v.data)[offset]);
#else
    return __half2float(static_cast<__half const *>(v.data)[offset]);
#endif
#else
    return v.load(offset);
#endif
}
#ifdef LOWP_BF16
__device__ __forceinline__ unsigned attention_pack(float a, float b) {
#if LOWP_BF16
    __nv_bfloat162 value = __floats2bfloat162_rn(a, b);
#else
    __half2 value = __floats2half2_rn(a, b);
#endif
    return *reinterpret_cast<unsigned *>(&value);
}

// Register fragments follow the m16n8k16 row/column mapping used by
// FlashAttention. Feature tiles are loaded once, then reused by all jet products.
template <int Width, int Pitch, int Planes, bool ColumnMajor = false>
__device__ void cached_qk_product(float const *cache, float *out, int left, int right,
                                  int n, int channels, int group_threads = blockDim.x) {
    constexpr int width = Width;
    constexpr int plane_size = ColumnMajor ? 33 * Width : 32 * Pitch;
    int tid = threadIdx.x % group_threads;
    int lane = tid % 32;
    for (int tile = tid / 32; tile < 8; tile += group_threads / 32) {
        int row = tile / 4 * 16 + lane / 4;
        int col = tile % 4 * 8;
        float d[4] = {};
        for (int base = 0; base < width; base += 16) {
            float a[8], b[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int c = base + (lane % 4) * 2 + i % 2 + (i / 4) * 8;
                int r = row + (i / 2 % 2) * 8;
                a[i] = r < n && c < channels ? cache[left * plane_size + (ColumnMajor ? c * 33 + r : r * Pitch + c)] : 0.f;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int c = base + (lane % 4) * 2 + i % 2 + (i / 2) * 8;
                int r = col + lane / 4;
                b[i] = r < n && c < channels ? cache[(Planes + right) * plane_size + (ColumnMajor ? c * 33 + r : r * Pitch + c)] : 0.f;
            }
            unsigned aa[4], bb[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) aa[i] = attention_pack(a[2 * i], a[2 * i + 1]);
            #pragma unroll
            for (int i = 0; i < 2; ++i) bb[i] = attention_pack(b[2 * i], b[2 * i + 1]);
#if LOWP_BF16
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#else
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#endif
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(bb[0]), "r"(bb[1]));
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            out[(row + i / 2 * 8) * 32 + col + lane % 4 * 2 + i % 2] = d[i];
    }
    __syncthreads();
}


#endif
"""


NODE_SOURCE = r"""
#ifdef LOWP_BF16
#define NODE_DOUBLE_SOURCE 1
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>
#if LOWP_BF16
using LowMmaElement=cutlass::bfloat16_t;
#else
using LowMmaElement=cutlass::half_t;
#endif
using NodeDoubleMma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    LowMmaElement,cutlass::layout::RowMajor,LowMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;

template<int Pitch, bool Transpose>
__device__ __forceinline__ NodeDoubleMma::FragmentA node_double_load_a(
    LowMmaElement const* pointer, int row, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/16)*8)*Pitch+row+(lane/8%2)*8
                           : (row+lane%16)*Pitch+base+(lane/16)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    NodeDoubleMma::FragmentA result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    return result;
}

template<int Pitch, bool Transpose>
__device__ __forceinline__ NodeDoubleMma::FragmentB node_double_load_b(
    LowMmaElement const* pointer, int column, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/8%2)*8)*Pitch+column
                           : (column+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    NodeDoubleMma::FragmentB result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    return result;
}

constexpr int NODE_DOUBLE_PITCH=((DIM>VALUE_DIM ? DIM:VALUE_DIM)+15)/16*16+8;
constexpr int NODE_DOUBLE_PLANE=32*NODE_DOUBLE_PITCH;
constexpr int NODE_DOUBLE_BYTES=2*(7*NODE_DOUBLE_PLANE+2*32*40)+4*(448+VALUE_DIM);

template<int Input,int Slot,int Channels>
__device__ void node_double_feature(Params const& p,LowMmaElement* cache,int64_t begin,int n,int head) {
    auto const tensor=p.in[Input];
    if constexpr(PRESENT&(1u<<Input)) {
        if(Channels%8==0 && tensor.stride[2]==1 && tensor.stride[0]%8==0 && tensor.stride[1]%8==0
            && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
            for(int t=threadIdx.x*8;t<NODE_DOUBLE_PLANE;t+=128*8) {
                int row=t/NODE_DOUBLE_PITCH,c=t%NODE_DOUBLE_PITCH;
                uint4 value={0,0,0,0};
                if(row<n && c<Channels)
                    value=*reinterpret_cast<uint4 const*>(static_cast<LowMmaElement const*>(tensor.data)+tensor.index(begin+row,head,c));
                *reinterpret_cast<uint4*>(cache+Slot*NODE_DOUBLE_PLANE+t)=value;
            }
            return;
        }
    }
    for(int t=threadIdx.x;t<NODE_DOUBLE_PLANE;t+=128) {
        int row=t/NODE_DOUBLE_PITCH,c=t%NODE_DOUBLE_PITCH;
        float value=0.f;
        if constexpr(PRESENT&(1u<<Input)) if(row<n && c<Channels)
            value=attention_feature_load(tensor,tensor.index(begin+row,head,c));
        cache[Slot*NODE_DOUBLE_PLANE+t]=LowMmaElement(value);
    }
}

template<int Target,int C0,int F0,int C1,int F1,bool Transpose>
__device__ void node_double_output(Params const& p,LowMmaElement* cache,LowMmaElement const* coefficients,
                                  LowMmaElement* temporary,int64_t begin,int n,int head) {
    if constexpr(NEEDS&(1<<Target)) {
        constexpr int Tiles=(DIM+7)/8;
        int lane=threadIdx.x%32;
        for(int tile=threadIdx.x/32;tile<2*Tiles;tile+=4) {
            int row=tile/Tiles*16,c=tile%Tiles*8;
            NodeDoubleMma::FragmentC value;value.clear();
            #pragma unroll
            for(int base=0;base<32;base+=16) {
                auto a=node_double_load_a<40,Transpose>(coefficients+C0*32*40,row,base,lane);
                auto b=node_double_load_b<NODE_DOUBLE_PITCH,true>(cache+F0*NODE_DOUBLE_PLANE,c,base,lane);
                NodeDoubleMma{}(value,a,b,value);
                a=node_double_load_a<40,Transpose>(coefficients+C1*32*40,row,base,lane);
                b=node_double_load_b<NODE_DOUBLE_PITCH,true>(cache+F1*NODE_DOUBLE_PLANE,c,base,lane);
                NodeDoubleMma{}(value,a,b,value);
            }
            #pragma unroll
            for(int i=0;i<4;++i)
                temporary[(row+lane/4+i/2*8)*NODE_DOUBLE_PITCH+c+lane%4*2+i%2]=LowMmaElement(value[i]*p.scale);
        }
        __syncthreads();
        if constexpr(DIM%8==0) {
            for(int t=threadIdx.x*8;t<n*DIM;t+=128*8) {
                int row=t/DIM,c=t%DIM;
                *reinterpret_cast<uint4*>(static_cast<LowMmaElement*>(p.out[Target].data)+((begin+row)*HEADS+head)*DIM+c)=
                    *reinterpret_cast<uint4 const*>(temporary+row*NODE_DOUBLE_PITCH+c);
            }
        } else {
            for(int t=threadIdx.x;t<n*DIM;t+=128) {
                int row=t/DIM,c=t%DIM;
                p.out[Target].store(((begin+row)*HEADS+head)*DIM+c,float(temporary[row*NODE_DOUBLE_PITCH+c]));
            }
        }
        __syncthreads();
    }
}

// A node adjoint is shared across queries: g_q=e_q*g_node. Its first
// direction is rank one as well, so value/upstream contractions reduce to
// two vectors instead of generating dense g/rg or attention output matrices.
__device__ void node_double_short(Params const& p,int64_t atom,int64_t begin,int n) {
    extern __shared__ float shared[];
    auto* cache=reinterpret_cast<LowMmaElement*>(shared);
    auto* coefficients=cache+6*NODE_DOUBLE_PLANE;
    auto* temporary=coefficients+2*32*40;
    auto* envelopes=reinterpret_cast<float*>(temporary+NODE_DOUBLE_PLANE);
    auto* envelope_r=envelopes+32;
    auto* dot=envelope_r+32;
    auto* dot_r=dot+32;
    auto* node_grad=dot_r+32;
    auto* alpha=node_grad+VALUE_DIM;
    auto* alpha_r=alpha+32;
    auto* partial=alpha_r+32;
    auto* bias_partial=partial+128;
    auto* env_gradient=bias_partial+64;
    float envelope_total=0.f;
    if(threadIdx.x<32) {
        envelopes[threadIdx.x]=(PRESENT&16) && threadIdx.x<n ? p.in[4].get(begin+threadIdx.x):1.f;
        envelope_r[threadIdx.x]=(PRESENT&(1u<<11)) && threadIdx.x<n ? p.in[11].get(begin+threadIdx.x):0.f;
    }
    __syncthreads();
    for(int head=0;head<HEADS;++head) {
        if(threadIdx.x<VALUE_DIM)
            node_grad[threadIdx.x]=(PRESENT&32) ? attention_feature_load(p.in[5],p.in[5].index(atom,head,threadIdx.x)):0.f;
        node_double_feature<0,0,DIM>(p,cache,begin,n,head);
        node_double_feature<1,1,DIM>(p,cache,begin,n,head);
        node_double_feature<2,2,VALUE_DIM>(p,cache,begin,n,head);
        node_double_feature<7,3,DIM>(p,cache,begin,n,head);
        node_double_feature<8,4,DIM>(p,cache,begin,n,head);
        node_double_feature<9,5,VALUE_DIM>(p,cache,begin,n,head);
        if constexpr(BIAS_KIND==2 && (NEEDS&8))
            for(int64_t t=threadIdx.x;t<p.out[3].size[2]*p.out[3].size[3];t+=128)
                p.out[3].store(p.out[3].index(atom,head,t/p.out[3].size[3],t%p.out[3].size[3]),0.f);
        __syncthreads();
        {
            int row=threadIdx.x/4,c=threadIdx.x%4;
            float d=0.f,dr=0.f;
            #pragma unroll
            for(int j=c;j<VALUE_DIM;j+=4) {
                d=fmaf(float(cache[2*NODE_DOUBLE_PLANE+row*NODE_DOUBLE_PITCH+j]),node_grad[j],d);
                dr=fmaf(float(cache[5*NODE_DOUBLE_PLANE+row*NODE_DOUBLE_PITCH+j]),node_grad[j],dr);
            }
            #pragma unroll
            for(int s=1;s<4;s*=2) {
                d+=__shfl_xor_sync(0xffffffff,d,s);
                dr+=__shfl_xor_sync(0xffffffff,dr,s);
            }
            if(c==0) {dot[row]=d;dot_r[row]=dr;}
        }
        __syncthreads();
        int lane=threadIdx.x%32,warp=threadIdx.x/32;
        if(threadIdx.x<64) {
            int row=warp*16;
            NodeDoubleMma::FragmentC scores[4],directions[4];
            #pragma unroll
            for(int j=0;j<4;++j) {scores[j].clear();directions[j].clear();}
            #pragma unroll
            for(int base=0;base<(DIM+15)/16*16;base+=16) {
                auto q=node_double_load_a<NODE_DOUBLE_PITCH,false>(cache,row,base,lane);
                auto rq=node_double_load_a<NODE_DOUBLE_PITCH,false>(cache+3*NODE_DOUBLE_PLANE,row,base,lane);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    auto k=node_double_load_b<NODE_DOUBLE_PITCH,false>(cache+NODE_DOUBLE_PLANE,j*8,base,lane);
                    auto rk=node_double_load_b<NODE_DOUBLE_PITCH,false>(cache+4*NODE_DOUBLE_PLANE,j*8,base,lane);
                    NodeDoubleMma{}(scores[j],q,k,scores[j]);
                    NodeDoubleMma{}(directions[j],rq,k,directions[j]);
                    NodeDoubleMma{}(directions[j],q,rk,directions[j]);
                }
            }
            float alpha_part[4][2]={},alpha_r_part[4][2]={};
            #pragma unroll
            for(int r=0;r<2;++r) {
                int qr=row+lane/4+r*8;
                float maximum=-INFINITY;
                #pragma unroll
                for(int j=0;j<4;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+lane%4*2+c,index=r*2+c;
                        float s=-INFINITY,a=0.f;
                        if(qr<n && key<n && (!CAUSAL || key<=qr)) {
                            s=scores[j][index]*p.scale;
                            a=directions[j][index]*p.scale;
                            if constexpr(BIAS_KIND==1) {
                                s+=p.in[3].get(begin+key,head);
                                if constexpr(PRESENT&(1u<<10)) a+=p.in[10].get(begin+key,head);
                            }
                            if constexpr(BIAS_KIND==2) {
                                s+=p.in[3].get(atom,head,qr,key);
                                if constexpr(PRESENT&(1u<<10)) a+=p.in[10].get(atom,head,qr,key);
                            }
                        }
                        scores[j][index]=s;directions[j][index]=a;
                        maximum=fmaxf(maximum,s);
                    }
                maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
                maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
                float norm=0.f,mean=0.f;
                #pragma unroll
                for(int j=0;j<4;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int index=r*2+c;
                        float prob=isfinite(maximum) ? __expf(scores[j][index]-maximum):0.f;
                        scores[j][index]=prob;
                        norm+=prob;mean=fmaf(prob,directions[j][index],mean);
                    }
                #pragma unroll
                for(int s=1;s<4;s*=2) {
                    norm+=__shfl_xor_sync(0xffffffff,norm,s);
                    mean+=__shfl_xor_sync(0xffffffff,mean,s);
                }
                float inv=norm>0.f ? 1.f/norm:0.f;mean*=inv;
                float delta=0.f,delta_r=0.f;
                #pragma unroll
                for(int j=0;j<4;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+lane%4*2+c,index=r*2+c;
                        float prob=scores[j][index]*inv,pr=prob*(directions[j][index]-mean);
                        scores[j][index]=prob;directions[j][index]=pr;
                        delta=fmaf(prob,dot[key],delta);
                        delta_r=fmaf(pr,dot[key],fmaf(prob,dot_r[key],delta_r));
                    }
                #pragma unroll
                for(int s=1;s<4;s*=2) {
                    delta+=__shfl_xor_sync(0xffffffff,delta,s);
                    delta_r+=__shfl_xor_sync(0xffffffff,delta_r,s);
                }
                if(lane%4==0) env_gradient[qr]=delta_r;
                float e=envelopes[qr],er=envelope_r[qr];
                #pragma unroll
                for(int j=0;j<4;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+lane%4*2+c,index=r*2+c;
                        float prob=scores[j][index],pr=directions[j][index];
                        float ep=e*prob,epr=fmaf(e,pr,er*prob);
                        float a=ep*(dot[key]-delta);
                        float ar=fmaf(epr,dot[key]-delta,ep*(dot_r[key]-delta_r));
                        coefficients[qr*40+key]=LowMmaElement(a);
                        coefficients[32*40+qr*40+key]=LowMmaElement(ar);
                        alpha_part[j][c]+=ep;
                        alpha_r_part[j][c]+=epr;
                        scores[j][index]=ar;
                        if constexpr(BIAS_KIND==2 && (NEEDS&8)) if(qr<n && key<n)
                            p.out[3].store(p.out[3].index(atom,head,qr,key),ar);
                    }
            }
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int c=0;c<2;++c) {
                    float a=alpha_part[j][c],ar=alpha_r_part[j][c];
                    #pragma unroll
                    for(int s=4;s<32;s*=2) {
                        a+=__shfl_xor_sync(0xffffffff,a,s);
                        ar+=__shfl_xor_sync(0xffffffff,ar,s);
                    }
                    if(lane<4) {
                        int key=j*8+lane*2+c;
                        partial[warp*32+key]=a;
                        partial[64+warp*32+key]=ar;
                    }
                    if constexpr(BIAS_KIND==1 && (NEEDS&8)) {
                        float db=scores[j][c]+scores[j][2+c];
                        #pragma unroll
                        for(int s=4;s<32;s*=2) db+=__shfl_xor_sync(0xffffffff,db,s);
                        if(lane<4) bias_partial[warp*32+j*8+lane*2+c]=db;
                    }
                }
        }
        __syncthreads();
        if(threadIdx.x<32) {
            alpha[threadIdx.x]=partial[threadIdx.x]+partial[32+threadIdx.x];
            alpha_r[threadIdx.x]=partial[64+threadIdx.x]+partial[96+threadIdx.x];
            envelope_total+=env_gradient[threadIdx.x];
            if constexpr(BIAS_KIND==1 && (NEEDS&8)) if(threadIdx.x<n)
                p.out[3].store(p.out[3].index(begin+threadIdx.x,head),bias_partial[threadIdx.x]+bias_partial[32+threadIdx.x]);
        }
        __syncthreads();
        node_double_output<0,1,1,0,4,false>(p,cache,coefficients,temporary,begin,n,head);
        node_double_output<1,1,0,0,3,true>(p,cache,coefficients,temporary,begin,n,head);
        if constexpr(NEEDS&4)
            for(int t=threadIdx.x;t<n*VALUE_DIM;t+=128) {
                int row=t/VALUE_DIM,c=t%VALUE_DIM;
                p.out[2].store(((begin+row)*HEADS+head)*VALUE_DIM+c,alpha_r[row]*node_grad[c]);
            }
        if constexpr(NEEDS&32) if(threadIdx.x<VALUE_DIM) {
            float result=0.f;
            for(int row=0;row<n;++row) {
                result=fmaf(alpha_r[row],float(cache[2*NODE_DOUBLE_PLANE+row*NODE_DOUBLE_PITCH+threadIdx.x]),result);
                result=fmaf(alpha[row],float(cache[5*NODE_DOUBLE_PLANE+row*NODE_DOUBLE_PITCH+threadIdx.x]),result);
            }
            p.out[5].store(p.out[5].index(atom,head,threadIdx.x),result);
        }
        __syncthreads();
    }
    if constexpr(NEEDS&16) if(threadIdx.x<n)
        p.out[4].store(p.out[4].index(begin+threadIdx.x),envelope_total);
}
#endif
"""

DERIVATIVE_SOURCE = r"""


// Accumulate each nonzero derivative product directly with IEEE FP32 FMA.
// Input features and upstream adjoints have zero mixed input component.
template <bool LeftMixed = true, bool RightMixed = true, int N>
__device__ __forceinline__ Differential<N> jet_fma(Differential<N> sum, Differential<N> a, Differential<N> b) {
    sum.x[0] = fmaf(a.x[0], b.x[0], sum.x[0]);
    if constexpr (N >= 2) sum.x[1] = fmaf(a.x[1], b.x[0], fmaf(a.x[0], b.x[1], sum.x[1]));
    if constexpr (N == 4) {
        sum.x[2] = fmaf(a.x[2], b.x[0], fmaf(a.x[0], b.x[2], sum.x[2]));
        sum.x[3] = fmaf(a.x[1], b.x[2], fmaf(a.x[2], b.x[1], sum.x[3]));
        if constexpr (LeftMixed) sum.x[3] = fmaf(a.x[3], b.x[0], sum.x[3]);
        if constexpr (RightMixed) sum.x[3] = fmaf(a.x[0], b.x[3], sum.x[3]);
    }
    return sum;
}

// A constant scalar has no derivative components; omit zero cross-products.
template <int N>
__device__ __forceinline__ Differential<N> jet_scale(Differential<N> value, float scalar) {
    #pragma unroll
    for (int c = 0; c < N; ++c) value.x[c] *= scalar;
    return value;
}

// Packed features use compile-time widths while keeping the edge product int64.
// Arbitrary input strides retain the general view path.
__device__ __forceinline__ int64_t input_offset(Params const &p, int actual, int field,
                                               int64_t i, int64_t j, int64_t k, int64_t l = 0) {
    if (CONTIGUOUS & (1u << actual)) {
        if (field == 0 || field == 1) return (i * HEADS + j) * DIM + k;
        if (field == 2 || field == 5 || field == 6) return (i * HEADS + j) * VALUE_DIM + k;
        if (field == 4) return i;
    }
    return p.in[actual].index(i,j,k,l);
}

// Presence masks remove absent raw-output adjoints and unused directions at compile time.
template <int N>
__device__ __forceinline__ Differential<N> static_input(Params const &p, int slot, int64_t i, int64_t j = 0, int64_t k = 0, int64_t l = 0) {
    Differential<N> value;
    if (PRESENT & (1u << slot)) value.x[0] = (slot < 3 || slot > 4 ? attention_feature_load(p.in[slot], input_offset(p,slot,slot,i,j,k,l)) : p.in[slot].load(input_offset(p,slot,slot,i,j,k,l)));
    if constexpr (N >= 2)
        if (PRESENT & (1u << (7 + slot))) value.x[1] = (slot < 3 || slot > 4 ? attention_feature_load(p.in[7 + slot], input_offset(p,7+slot,slot,i,j,k,l)) : p.in[7 + slot].load(input_offset(p,7+slot,slot,i,j,k,l)));
    if constexpr (N == 4)
        if (PRESENT & (1u << (12 + slot))) value.x[2] = (slot < 3 || slot > 4 ? attention_feature_load(p.in[12 + slot], input_offset(p,12+slot,slot,i,j,k,l)) : p.in[12 + slot].load(input_offset(p,12+slot,slot,i,j,k,l)));
    return value;
}
template <int N>
__device__ __forceinline__ Differential<N> static_upstream(Params const &p, int slot, int64_t i, int h, int c) {
    Differential<N> value;
    if (PRESENT & (1u << slot)) value.x[0] = (slot < 3 || slot > 4 ? attention_feature_load(p.in[slot], input_offset(p,slot,slot,i,h,c)) : p.in[slot].load(input_offset(p,slot,slot,i,h,c)));
    if constexpr (N == 4)
        if (PRESENT & (1u << (12 + slot))) value.x[2] = (slot < 3 || slot > 4 ? attention_feature_load(p.in[12 + slot], input_offset(p,12+slot,slot,i,h,c)) : p.in[12 + slot].load(input_offset(p,12+slot,slot,i,h,c)));
    return value;
}

// Fixed channel tiles bound shared storage for arbitrary feature widths. Each
// warps own fixed query tiles and reuse feature loads across products.
template <int Order>
__device__ __forceinline__ void short_derivative(Params const &p, int64_t atom, int64_t begin, int n) {
    constexpr int N = Order == 2 ? 2 : 4;
    constexpr int planes = N == 4 ? 3 : N;
    constexpr int extent = DIM > VALUE_DIM ? DIM : VALUE_DIM;
    constexpr int width = extent < 32 ? extent : 32;
    constexpr int pitch = width + 1;
    constexpr int plane_size = 32 * pitch;
    using F = Differential<N>;
    constexpr bool raw_adjoint = (PRESENT & 64) || (N == 4 && (PRESENT & (1u << 18)));
    constexpr int feature_size = 3 * planes * plane_size;
    constexpr int pv_size = 2 * N * 1024 + planes * plane_size;
    constexpr int qk_size = N * 1024 + 2 * planes * plane_size;
    constexpr int base_arena = feature_size > pv_size ? (feature_size > qk_size ? feature_size : qk_size) : (pv_size > qk_size ? pv_size : qk_size);
#ifdef LOWP_BF16
    constexpr int arena = base_arena > feature_size + 1024 ? base_arena : feature_size + 1024;
#else
    constexpr int arena = base_arena;
#endif
    extern __shared__ float storage[];
    F *ds = reinterpret_cast<F *>(storage);
    F *probability = ds + 1024;
    F *full_dot = reinterpret_cast<F *>(storage + arena);
    F *envelope_dot = full_dot + 32;
    float *cache_data = storage;
    auto feature = [&](int slot, int row, int c) {
        F result;
        #pragma unroll
        for (int component = 0; component < planes; ++component)
            result.x[component] = cache_data[(slot * planes + component) * plane_size + row * pitch + c];
        return result;
    };
    // Copy second-order feature planes directly into the existing shared arena.
    // Every consuming tile waits for its copies before the block barrier.
    auto load_feature = [&](int slot, int target, int h, int base) {
        for (int t = threadIdx.x; t < n * width; t += blockDim.x) {
            int row = t / width, c = t % width;
            #ifndef LOWP_BF16
            if constexpr (N == 2) {
                #pragma unroll
                for (int component = 0; component < planes; ++component) {
                    int actual = component == 0 ? slot : 7 + slot;
                    float *target_ptr = cache_data + (target * planes + component) * plane_size + row * pitch + c;
                    if ((PRESENT & (1u << actual)) && base + c < (slot < 2 ? DIM : VALUE_DIM)) {
                        int64_t offset = input_offset(p, actual, slot, begin + row, h, base + c);
                        unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(target_ptr));
                        const float *source_ptr = p.in[actual].data + offset;
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(address), "l"(source_ptr));
                    } else {
                        *target_ptr = 0.f;
                    }
                }
            } else
#endif
            {
                F value;
                if (base + c < (slot < 2 ? DIM : VALUE_DIM))
                    value = static_input<N>(p, slot, begin + row, h, base + c);
                #pragma unroll
                for (int component = 0; component < planes; ++component)
                    cache_data[(target * planes + component) * plane_size + row * pitch + c] = value.x[component];
            }
        }
    };
    constexpr int Rows = Order == 2 ? 4 : 2;
    int lane = threadIdx.x % 32;
    int row = threadIdx.x / 32 * Rows;
    // A half warp owns two keys per lane in the second-order score tile.
    constexpr int ScoreCols = Order == 2 ? 2 : 1;
    constexpr int ScoreRows = Rows / ScoreCols;
    constexpr int GroupWidth = 32 / ScoreCols;
    int score_row = (threadIdx.x / GroupWidth) * ScoreRows;
    int key_start = (lane % GroupWidth) * ScoreCols;
    unsigned group_mask = (0xffffffffu >> (32 - GroupWidth)) << (lane / GroupWidth * GroupWidth);
    constexpr int EnvelopeRows = raw_adjoint ? Rows : ScoreRows;
    int envelope_row = raw_adjoint ? row : score_row;
    bool envelope_leader = raw_adjoint ? lane == 0 : lane % GroupWidth == 0;
    F envelope_gradient[EnvelopeRows];
    for (int h = 0; h < HEADS; ++h) {
        if (threadIdx.x < n) full_dot[threadIdx.x] = F();
        cache_data = storage;
        auto group_sum = [&](F value) {
            #pragma unroll
            for (int d = GroupWidth / 2; d; d /= 2)
                #pragma unroll
                for (int component = 0; component < N; ++component)
                    value.x[component] += __shfl_xor_sync(group_mask, value.x[component], d, GroupWidth);
            return value;
        };
        F scores[ScoreRows][ScoreCols], adjoints[ScoreRows][ScoreCols];
        for (int base = 0; base < extent; base += width) {
            load_feature(0, 0, h, base);
            load_feature(1, 1, h, base);
            load_feature(2, 2, h, base);
            if constexpr (N == 2) asm volatile("cp.async.wait_all;" ::: "memory");
            __syncthreads();
            if (threadIdx.x < n) {
                F dot;
                for (int c = 0; c < width && base + c < VALUE_DIM; ++c)
                    dot = jet_fma<false, false>(dot, feature(2, threadIdx.x, c), static_upstream<N>(p, 5, atom, h, base + c));
                full_dot[threadIdx.x] = full_dot[threadIdx.x] + dot;
            }
#ifdef LOWP_BF16
            if (base < DIM) {
                float *product = storage + feature_size;
                #pragma unroll
                for (int component = 0; component < N; ++component) {
                    #pragma unroll
                    for (int term = 0; term < (component == 0 ? 1 : 2); ++term) {
                        int left = component == 0 ? 0 : component == 3 ? (term == 0 ? 1 : 2) : term == 0 ? component : 0;
                        int right = component == 0 ? 0 : component == 3 ? (term == 0 ? 2 : 1) : term == 1 ? component : 0;
                        cached_qk_product<width, pitch, planes>(cache_data, product, left, right, n, min(width, DIM - base));
                        #pragma unroll
                        for (int r = 0; r < ScoreRows; ++r)
                            #pragma unroll
                            for (int j = 0; j < ScoreCols; ++j)
                                scores[r][j].x[component] += product[(score_row + r) * 32 + key_start + j];
                        __syncthreads();
                    }
                }
            }
#endif
            for (int c = 0; c < width; ++c) {
                F keys[ScoreCols], values[ScoreCols];
                #pragma unroll
                for (int j = 0; j < ScoreCols; ++j) {
                    if (key_start + j < n) {
                        keys[j] = feature(1, key_start + j, c);
                        if constexpr (raw_adjoint) values[j] = feature(2, key_start + j, c);
                    }
                }
                #pragma unroll
                for (int r = 0; r < ScoreRows; ++r) {
                    int i = score_row + r;
                    F query, upstream;
                    if (i < n) {
                        query = feature(0, i, c);
                        if constexpr (raw_adjoint)
                            if (base + c < VALUE_DIM) upstream = static_upstream<N>(p, 6, begin + i, h, base + c);
                    }
                    #pragma unroll
                    for (int j = 0; j < ScoreCols; ++j) {
                        int k = key_start + j;
                        if (i < n && k < n && (!CAUSAL || k <= i)) {
#ifndef LOWP_BF16
                            if (base + c < DIM)
                                scores[r][j] = jet_fma<false, false>(scores[r][j], query, keys[j]);
#endif
                            if constexpr (raw_adjoint)
                                if (base + c < VALUE_DIM)
                                    adjoints[r][j] = jet_fma<false, false>(adjoints[r][j], values[j], upstream);
                        }
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < ScoreRows; ++r) if (score_row + r < n) {
            int i = score_row + r;
            F score[ScoreCols], weight[ScoreCols], prob[ScoreCols], adjoint[ScoreCols];
            float maximum = -INFINITY;
            #pragma unroll
            for (int j = 0; j < ScoreCols; ++j) {
                int k = key_start + j;
                score[j] = F(-INFINITY);
                if (k < n && (!CAUSAL || k <= i)) {
                    score[j] = jet_scale(scores[r][j], p.scale);
                    if constexpr (BIAS_KIND == 1) score[j] = score[j] + static_input<N>(p, 3, begin + k, h);
                    if constexpr (BIAS_KIND == 2) score[j] = score[j] + static_input<N>(p, 3, atom, h, i, k);
                }
                maximum = fmaxf(maximum, score[j].x[0]);
            }
            #pragma unroll
            for (int d = GroupWidth / 2; d; d /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(group_mask, maximum, d, GroupWidth));
            #pragma unroll
            for (int j = 0; j < ScoreCols; ++j)
                if (isfinite(score[j].x[0]) && isfinite(maximum))
                    weight[j] = exponential(score[j] - F(maximum));
            F den = weight[0];
            #pragma unroll
            for (int j = 1; j < ScoreCols; ++j) den = den + weight[j];
            den = group_sum(den);
            F inverse = den.x[0] > 0.f ? reciprocal(den) : F();
            F env = (PRESENT & 16) ? static_input<N>(p, 4, begin + i) : F(1.f);
            #pragma unroll
            for (int j = 0; j < ScoreCols; ++j) {
                int k = key_start + j;
                prob[j] = weight[j] * inverse;
                probability[i * 32 + k] = prob[j];
                adjoint[j] = k < n ? full_dot[k] : F();
                if constexpr (raw_adjoint)
                    adjoint[j] = jet_fma<false, true>(adjoints[r][j], env, adjoint[j]);
            }
            F delta = prob[0] * adjoint[0];
            #pragma unroll
            for (int j = 1; j < ScoreCols; ++j) delta = delta + prob[j] * adjoint[j];
            delta = group_sum(delta);
            if constexpr (!raw_adjoint)
                if (lane % GroupWidth == 0) envelope_gradient[r] = envelope_gradient[r] + delta;
            #pragma unroll
            for (int j = 0; j < ScoreCols; ++j) {
                int k = key_start + j;
                F score_gradient = prob[j] * (adjoint[j] - delta);
                if constexpr (!raw_adjoint) score_gradient = score_gradient * env;
                ds[i * 32 + k] = score_gradient;
                if constexpr (BIAS_KIND == 2)
                    if (k < n) gradient_unique<Order>(p, 3, score_gradient, atom, h, i, k);
            }
        }
        __syncthreads();
        // Envelope derivatives contract P with V dot g_node, avoiding edge O.
        #pragma unroll
        for (int r = 0; r < Rows; ++r) if (row + r < n) {
            if constexpr (raw_adjoint && (NEEDS & ((1u << 4) | (1u << 11)))) {
                F de = lane < n ? probability[(row + r) * 32 + lane] * full_dot[lane] : F();
                de = warp_sum(de);
                if (lane == 0) envelope_gradient[r] = envelope_gradient[r] + de;
            }
        }
        // Reuse the dot scratch for alpha_k = sum_q envelope_q P_qk.
        if (threadIdx.x < n) {
            F alpha;
            for (int i = 0; i < n; ++i) {
                F env = (PRESENT & 16) ? static_input<N>(p, 4, begin + i) : F(1.f);
                alpha = jet_fma<false, true>(alpha, env, probability[i * 32 + threadIdx.x]);
            }
            envelope_dot[threadIdx.x] = alpha;
        }
        // The subsequent V-cache barrier also makes alpha visible to all warps.
        if constexpr (BIAS_KIND == 1) {
            #pragma unroll
            for (int r = 0; r < Rows; ++r) if (row + r < n) {
                F db = lane < n ? ds[lane * 32 + row + r] : F();
                db = warp_sum(db);
                if (lane == 0) gradient_unique<Order>(p, 3, db, begin + row + r, h);
            }
        }
        // Consume probabilities before reusing their storage for Q/K features.
        cache_data = storage + 2 * N * 1024;
        for (int base = 0; base < VALUE_DIM; base += width) {
            load_feature(2, 0, h, base);
            if constexpr (N == 2) asm volatile("cp.async.wait_all;" ::: "memory");
            __syncthreads();
            int c = base + lane;
            F gnode = lane < width && c < VALUE_DIM ? static_upstream<N>(p, 5, atom, h, c) : F();
            F dv[Rows], output[Rows];
            if (lane < width && c < VALUE_DIM && row < n) {
                #pragma unroll
                for (int r = 0; r < Rows; ++r) if (row + r < n)
                    dv[r] = envelope_dot[row + r] * gnode;
                if constexpr ((PRESENT & 64) || (N == 4 && (PRESENT & (1u << 18))) || (NEEDS & 64)) {
                    for (int j = 0; j < n; ++j) {
                        F g = static_upstream<N>(p, 6, begin + j, h, c);
                        F value = feature(0, j, lane);
                        #pragma unroll
                        for (int r = 0; r < Rows; ++r) if (row + r < n) {
                            if constexpr ((PRESENT & 64) || (N == 4 && (PRESENT & (1u << 18))))
                                dv[r] = jet_fma<true, false>(dv[r], probability[j * 32 + row + r], g);
                            if constexpr (NEEDS & 64)
                                output[r] = jet_fma<true, false>(output[r], probability[(row + r) * 32 + j], value);
                        }
                    }
                }
                #pragma unroll
                for (int r = 0; r < Rows; ++r) if (row + r < n) {
                    int64_t i = begin + row + r;
                    gradient_unique<Order>(p, 2, dv[r], i, h, c);
                    if constexpr (NEEDS & 64)
                        p.out[6].store((i * HEADS + h) * VALUE_DIM + c, output[r].x[N - 1]);
                }
            }
            if (threadIdx.x < width && base + threadIdx.x < VALUE_DIM && (NEEDS & 32)) {
                F node;
                for (int j = 0; j < n; ++j)
                    node = jet_fma<true, false>(node, envelope_dot[j], feature(0, j, threadIdx.x));
                p.out[5].store(p.out[5].index(atom, h, base + threadIdx.x), node.x[N - 1]);
            }
            __syncthreads();
        }
        // Probabilities are dead; their storage now holds Q and K feature jets.
        cache_data = storage + N * 1024;
        for (int base = 0; base < DIM; base += width) {
            load_feature(0, 0, h, base);
            load_feature(1, 1, h, base);
            if constexpr (N == 2) asm volatile("cp.async.wait_all;" ::: "memory");
            __syncthreads();
            int c = base + lane;
            if (lane < width && c < DIM && row < n) {
                F dq[Rows], dk[Rows];
                for (int j = 0; j < n; ++j) {
                    F key = feature(1, j, lane), query = feature(0, j, lane);
                    #pragma unroll
                    for (int r = 0; r < Rows; ++r) if (row + r < n) {
                        dq[r] = jet_fma<true, false>(dq[r], ds[(row + r) * 32 + j], key);
                        dk[r] = jet_fma<true, false>(dk[r], ds[j * 32 + row + r], query);
                    }
                }
                #pragma unroll
                for (int r = 0; r < Rows; ++r) if (row + r < n) {
                    gradient_unique<Order>(p, 0, jet_scale(dq[r], p.scale), begin + row + r, h, c);
                    gradient_unique<Order>(p, 1, jet_scale(dk[r], p.scale), begin + row + r, h, c);
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int r = 0; r < EnvelopeRows; ++r)
        if (envelope_leader && envelope_row + r < n)
            gradient_unique<Order>(p, 4, envelope_gradient[r], begin + envelope_row + r);
}

template <int Order>
__device__ void clear_optional_outputs(Params const &p, int64_t atom, int64_t begin, int64_t end) {
    constexpr int slots = Order == 2 ? 7 : 12;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        if (!(NEEDS & (1u << slot))) continue;
        View v = p.out[slot];
        int field = slot >= 7 ? slot - 7 : slot;
        if (field == 4) {
            int64_t row = begin + threadIdx.x;
            if (row < end) v.store(v.index(row), 0.f);
        } else if (field == 3 && BIAS_KIND == 2) {
            for (int h = 0; h < HEADS; ++h)
                for (int64_t t = threadIdx.x; t < v.size[2] * v.size[3]; t += blockDim.x)
                    v.store(v.index(atom, h, t / v.size[3], t % v.size[3]), 0.f);
        }
    }
    __syncthreads();
}

template<int Order>
__device__ void finish_higher_rows(Params const& p,Params const& final_p,int64_t atom,bool resident) {
    int64_t begin=p.offset(atom),end=p.offset(atom+1);
    bool owned=end-begin<=32 || (resident && end-begin<=64);
    for(int slot=0;slot<(Order==2?7:12);++slot) {
        if(!(NEEDS&(1u<<slot)))continue;
        int field=slot>=7?slot-7:slot;
        if(field<3 && owned)continue;
        View const& from=p.out[slot];View const& to=final_p.out[slot];
        if(field==4) {
            for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x)to.store(i,from.fp32_data()[i]);
        } else if(field==3 && BIAS_KIND==2) {
            int64_t count=HEADS*to.size[2]*to.size[3];
            for(int64_t i=threadIdx.x;i<count;i+=blockDim.x)to.store(atom*count+i,from.fp32_data()[atom*count+i]);
        } else {
            int width=field<2?DIM:field==3?1:VALUE_DIM;
            int64_t first=field==5?atom:begin,last=field==5?atom+1:end;
            for(int64_t i=first*HEADS*width+threadIdx.x;i<last*HEADS*width;i+=blockDim.x)
                to.store(i,from.fp32_data()[i]);
        }
    }
}

template<int Order>
__global__ void finish_higher(const __grid_constant__ Params p,
    const __grid_constant__ Params final_p,bool resident) {
    finish_higher_rows<Order>(p,final_p,blockIdx.x,resident);
}

// A small grid keeps short work and a bounded streaming fallback in one CTA.
template<int Order>
__device__ void small_higher_long(Params const& p,Params const& final_p,
    int64_t atom,int64_t begin,int64_t end,float* envelope) {
    for(int head=0;head<HEADS;++head)
        tiled_backward<Order>(p,atom,begin,end,envelope,head,0,1);
    __syncthreads();
    if(envelope) {
        constexpr int Planes=Order==3?2:1;
        for(int64_t i=begin+threadIdx.x;i<end;i+=blockDim.x) {
            float primary=0.f,direction=0.f;
            for(int h=0;h<HEADS;++h) {
                if(NEEDS&16)primary+=envelope[(i*HEADS+h)*Planes];
                if constexpr(Order==3)if(NEEDS&(1u<<11))direction+=envelope[(i*HEADS+h)*Planes+1];
            }
            if(NEEDS&16)p.out[4].fp32_data()[i]=primary;
            if constexpr(Order==3)if(NEEDS&(1u<<11))p.out[11].fp32_data()[i]=direction;
        }
    }
    __syncthreads();
    finish_higher_rows<Order>(p,final_p,atom,false);
}

template<int Threads>
__global__ __launch_bounds__(Threads,Threads==128?2:4)
void small_double_kernel(const __grid_constant__ Params p,
    const __grid_constant__ Params native_p,
    const __grid_constant__ Params final_p,float* envelope) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if(end-begin<=32) {
#if defined(NODE_DOUBLE_SOURCE) && DIM<=64 && VALUE_DIM<=64 && !(PRESENT&64) && !(NEEDS&64)
        node_double_short(native_p,atom,begin,int(end-begin));
#else
        clear_optional_outputs<2>(native_p,atom,begin,end);
        short_derivative<2>(native_p,atom,begin,int(end-begin));
#endif
        __syncthreads();
        finish_higher_rows<2>(p,final_p,atom,false);
    } else small_higher_long<2>(p,final_p,atom,begin,end,envelope);
}

#if defined(NODE_DOUBLE_SOURCE) && DIM<=64 && VALUE_DIM<=64 && !(PRESENT&64) && !(NEEDS&64)
template<int Order> __global__ __launch_bounds__(128,2)
#else
template<int Order> __global__ __launch_bounds__(Order==2 ? 256:512,Order==2 ? 4:2)
#endif
void derivative_kernel(Params p,Params native_p,int query_workers,bool resident=true) {
    constexpr int N = Order == 2 ? 2 : 4;
    using F = Differential<N>;
    int64_t atom = blockIdx.x, begin = p.offset(atom), end = p.offset(atom + 1);
    if(end-begin>32) {
        if(query_workers>1) tiled_initialize<Order>(p,atom,resident);
        return;
    }
    if(end-begin<=32) {
#if defined(NODE_DOUBLE_SOURCE) && DIM<=64 && VALUE_DIM<=64 && !(PRESENT&64) && !(NEEDS&64)
        if constexpr(Order==2) {node_double_short(native_p,atom,begin,int(end-begin));return;}
#endif
        clear_optional_outputs<Order>(native_p, atom, begin, end);
        short_derivative<Order>(native_p, atom, begin, int(end - begin));
        return;
    }

}
"""


GRADIENT_SOURCE = r"""


// Direction r differentiates J(x)^T g. The matching upstream gradient is J(x)r.
template <int Order, int N>
__device__ void gradient(Params const &p, int slot, Differential<N> value, int64_t i, int64_t j = 0,
                         int64_t k = 0, int64_t l = 0) {
    if (p.out[slot].data)
        p.out[slot].atomic_add(p.out[slot].index(i, j, k, l), value.x[1]);
}
template <int Order, int N>
__device__ __forceinline__ void gradient_unique(Params const &p, int slot, Differential<N> value, int64_t i,
                                int64_t j = 0, int64_t k = 0, int64_t l = 0) {
    if (NEEDS & (1u << (0 + slot))) {
        auto offset = (slot < 2 ? (i * HEADS + j) * DIM + k : slot == 2 ? (i * HEADS + j) * VALUE_DIM + k : slot == 4 ? i : BIAS_KIND == 1 ? i * HEADS + j : p.out[0 + slot].index(i, j, k, l));
        if (slot == 4)
            p.out[0 + slot].accumulate(offset, value.x[1]);
        else
            p.out[0 + slot].store(offset, value.x[1]);
    }
}
"""

SOURCE = r"""


// Outputs: Hessian(loss) times incoming direction, then the attention/reduction
// directional output for each upstream adjoint. All are fused in one launch.
void run(std::vector<std::optional<at::Tensor>> inputs, at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,
         std::vector<std::optional<at::Tensor>> final_outputs, double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if (csr.numel() > 1) {
        constexpr int extent = DIM > VALUE_DIM ? DIM : VALUE_DIM;
        constexpr int width = extent < 32 ? extent : 32;
        constexpr int feature_size = 3 * 32 * (width + 1) * 2;
        constexpr int pv_size = 2 * 2 * 1024 + 32 * (width + 1) * 2;
        constexpr int qk_size = 2 * 1024 + 2 * 32 * (width + 1) * 2;
        constexpr int base_arena = feature_size > pv_size ? (feature_size > qk_size ? feature_size : qk_size) : (pv_size > qk_size ? pv_size : qk_size);
#ifdef LOWP_BF16
    constexpr int arena = base_arena > feature_size + 1024 ? base_arena : feature_size + 1024;
#else
    constexpr int arena = base_arena;
#endif
#if defined(NODE_DOUBLE_SOURCE) && DIM<=64 && VALUE_DIM<=64 && !(PRESENT&64) && !(NEEDS&64)
        constexpr int dynamic_bytes=NODE_DOUBLE_BYTES,threads=128;
#else
        #ifdef LOWP_BF16
        constexpr int dynamic_bytes=(arena+2*64)*sizeof(float),threads=256;
#else
        constexpr int short_bytes = (arena+2*64)*sizeof(float);
        constexpr int dynamic_bytes = short_bytes, threads=256;
#endif
#endif
        static thread_local int configured_device=-1;
        static thread_local int multiprocessors=0;
        int device=inputs[0]->get_device();
        if(configured_device!=device) {
            if constexpr(dynamic_bytes>0)
                C10_CUDA_CHECK(cudaFuncSetAttribute(derivative_kernel<2>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_bytes));
            C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<2,256,4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,TiledBackwardLayout<2>::SharedBytes));
            C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<2,256,4,false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,TiledBackwardLayout<2>::SharedBytes));
            if constexpr(TiledBackwardLayout<2>::NodeOnly) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<2,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeHigherLayout<2,64>::Bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<2,64>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<2,48>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeHigherLayout<2,48>::Bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<2,48>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
            }
#ifdef LOWP_BF16
            if constexpr(TiledBackwardLayout<2>::NodeOnly) {
                constexpr int small_bytes=dynamic_bytes>TiledBackwardLayout<2>::SharedBytes?
                    dynamic_bytes:TiledBackwardLayout<2>::SharedBytes;
                C10_CUDA_CHECK(cudaFuncSetAttribute(small_double_kernel<threads>,cudaFuncAttributeMaxDynamicSharedMemorySize,small_bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(small_double_kernel<threads>,cudaFuncAttributePreferredSharedMemoryCarveout,50));
            }
#endif
            multiprocessors=at::cuda::getDeviceProperties(device)->multiProcessorCount;
            configured_device=device;
        }
        Params p=params(inputs,outputs,csr,scale);
        Params final_p=params(inputs,final_outputs,csr,scale);
        Params native_p=p;
#ifdef LOWP_BF16
        for(int slot=0;slot<7;++slot)
            if(slot<3) native_p.out[slot]=final_p.out[slot];
#endif
        float* envelope=nullptr;
        int query_workers=1;
        bool resident=false;
        auto workspace=tiled_envelope_workspace<2>(*inputs[0]);
        if(workspace.defined()) envelope=workspace.data_ptr<float>();
        query_workers=tiled_query_workers<2>(inputs[0]->get_device(),csr.numel()-1);
        // Amortize two resident launches only when average degree exceeds a streaming tile.
        if constexpr(TiledBackwardLayout<2>::NodeOnly)
            resident=inputs[0]->size(0)>32*(csr.numel()-1);
        if(resident) query_workers=4;
#ifdef LOWP_BF16
        if constexpr(TiledBackwardLayout<2>::NodeOnly) {
            if(!resident && (csr.numel()-1)*HEADS<=2*multiprocessors) {
                constexpr int small_bytes=dynamic_bytes>TiledBackwardLayout<2>::SharedBytes?
                    dynamic_bytes:TiledBackwardLayout<2>::SharedBytes;
                small_double_kernel<threads><<<csr.numel()-1,threads,small_bytes,at::cuda::getCurrentCUDAStream()>>>(p,native_p,final_p,envelope);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
        }
#endif


        derivative_kernel<2><<<csr.numel()-1,threads,dynamic_bytes,at::cuda::getCurrentCUDAStream()>>>(p,native_p,query_workers,resident);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if(resident) {
        long_rows_kernel<2,256,4><<<dim3(csr.numel()-1,HEADS,query_workers),256,
            TiledBackwardLayout<2>::SharedBytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
        long_rows_kernel<2,256,4,false><<<dim3(csr.numel()-1,HEADS,query_workers),256,
            TiledBackwardLayout<2>::SharedBytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if constexpr(TiledBackwardLayout<2>::NodeOnly) if(resident) {
            node_higher_cached_kernel<2,64><<<dim3(csr.numel()-1,HEADS),(NodeHigherLayout<2,64>::Threads),(NodeHigherLayout<2,64>::Bytes),at::cuda::getCurrentCUDAStream()>>>(native_p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            node_higher_cached_kernel<2,48><<<dim3(csr.numel()-1,HEADS),(NodeHigherLayout<2,48>::Threads),(NodeHigherLayout<2,48>::Bytes),at::cuda::getCurrentCUDAStream()>>>(native_p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if(envelope) {
            tiled_envelope_reduce<2><<<csr.numel()-1,128,0,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
#ifdef LOWP_BF16
        finish_higher<2><<<csr.numel()-1,256,0,at::cuda::getCurrentCUDAStream()>>>(p,final_p,resident);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
"""


@lru_cache(maxsize=128)
def get_double_backward_kernel(config, device, dtype, present, contiguous, needed):
    defines = f"#define PRESENT {present}u\n#define NEEDS {needed}u\n#define CONTIGUOUS {contiguous}u\n"
    native_score = 0
    if dtype in (torch.float16, torch.bfloat16):
        pitch = (config[0] + 15) // 16 * 16 + 8
        required = 4 * max(2 * 64 * pitch, 2 * 64 * 72 // 2 + 2 * (64 * 64 + 4 * 64))
        native_score = int(required <= get_device_properties(device).shared_memory_per_block_optin)
    defines += f"#define HIGHER_NATIVE_SCORE {native_score}\n"
    return load_kernel(
        "attention_conv_double_backward", defines + LOWP_SOURCE + GRADIENT_SOURCE + NODE_SOURCE + TILED_SOURCE + DERIVATIVE_SOURCE + SOURCE,
        config, device.index, dtype,
    ).run


def attention_conv_double_backward(inputs, csr, config, needs):
    present = sum(1 << i for i, value in enumerate(inputs) if value is not None)
    contiguous = sum(1 << i for i, value in enumerate(inputs) if value is not None and value.is_contiguous())
    needed = sum(1 << i for i, need in enumerate(needs) if need and inputs[i] is not None)
    outputs = allocate(inputs[:7], needs, accumulate=True)
    final_outputs = allocate(inputs[:7], needs) if inputs[0].dtype in (torch.float16, torch.bfloat16) else outputs
    kernel = get_double_backward_kernel(config, inputs[0].device, inputs[0].dtype, present, contiguous, needed)
    kernel(inputs, csr, outputs, final_outputs, config[3])
    return final_outputs
