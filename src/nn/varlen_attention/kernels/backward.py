"""Fused CSR attention backward with FP32 accumulation."""

import torch

from .utils import CUDA_SOURCE, dtype_defines
from ....utils.extension import get_cuda_arch, get_device_properties, load_cuda_extension

COMPILED = {}

BACKWARD_SOURCE = r'''#if LOW_PRECISION
using BackwardMma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    LowMmaElement,cutlass::layout::RowMajor,LowMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;

template<int Pitch, bool Transpose>
__device__ __forceinline__ BackwardMma::FragmentA backward_load_a(
    LowMmaElement const* pointer, int row, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/16)*8)*Pitch+row+(lane/8%2)*8
                           : (row+lane%16)*Pitch+base+(lane/16)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    BackwardMma::FragmentA result;
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
__device__ __forceinline__ BackwardMma::FragmentB backward_load_b(
    LowMmaElement const* pointer, int column, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/8%2)*8)*Pitch+column
                           : (column+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    BackwardMma::FragmentB result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    return result;
}

template<int Tile,int Pitch,int Channels,int Heads,int Source,int Target,bool Transpose,int Threads>
__device__ void backward_lowp_output(AttentionParams const& p, LowMmaElement* cache,
    LowMmaElement const* weight, int64_t begin,int64_t end,int head) {
    constexpr int Tiles=(Channels+7)/8;
    int lane=threadIdx.x%32;
    auto* output=cache+(Target==0 ? 0:Tile*Pitch);
    for(int tile=threadIdx.x/32;tile<(Tile/16)*Tiles;tile+=Threads/32) {
        int row=(tile/Tiles)*16,column=(tile%Tiles)*8;
        BackwardMma::FragmentC result;result.clear();
        #pragma unroll
        for(int base=0;base<Tile;base+=16) {
            auto a=backward_load_a<Tile+8,Transpose>(weight,row,base,lane);
            auto b=backward_load_b<Pitch,true>(cache+Source*Tile*Pitch,column,base,lane);
            BackwardMma{}(result,a,b,result);
        }
        #pragma unroll
        for(int i=0;i<4;++i) {
            int r=row+lane/4+(i/2)*8,c=column+(lane%4)*2+i%2;
            output[r*Pitch+c]=LowMmaElement(result[i]*(Target<2?p.scale:1.f));
        }
    }
    __syncthreads();
    if constexpr (Channels%8==0) {
        for(int x=threadIdx.x*8;x<Tile*Channels;x+=Threads*8) {
            int r=x/Channels,c=x%Channels;
            if(begin+r<end)
                *reinterpret_cast<uint4*>(p.outputs[Target].data+((begin+r)*Heads+head)*Channels+c)=
                    *reinterpret_cast<const uint4*>(output+r*Pitch+c);
        }
    } else {
        for(int x=threadIdx.x;x<Tile*Channels;x+=Threads) {
            int r=x/Channels,c=x%Channels;
            if(begin+r<end) p.outputs[Target].store(begin+r,head,c,float(output[r*Pitch+c]));
        }
    }
    __syncthreads();
}

template<int Tile,int Pitch,int Threads>
__device__ void backward_lowp_cache(AttentionTensor const& tensor,LowMmaElement* cache,
    int64_t begin,int64_t end,int head,int width) {
    if(width%8==0 && tensor.stride[2]==1 && tensor.stride[0]%8==0 && tensor.stride[1]%8==0 && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
        for(int x=threadIdx.x*8;x<Tile*Pitch;x+=Threads*8) {
            int r=x/Pitch,c=x%Pitch;
            uint4 packed={0,0,0,0};
            if(begin+r<end && c<width)
                packed=*reinterpret_cast<const uint4*>(tensor.data+(begin+r)*tensor.stride[0]+head*tensor.stride[1]+c);
            *reinterpret_cast<uint4*>(cache+r*Pitch+c)=packed;
        }
    } else {
        for(int x=threadIdx.x;x<Tile*Pitch;x+=Threads) {
            int r=x/Pitch,c=x%Pitch;
            cache[x]=LowMmaElement(begin+r<end&&c<width?tensor.load(begin+r,head,c):0.f);
        }
    }
}

template<int Tile,int D,int V,int H,bool COMPACT,int BIAS,bool CAUSAL,int NEEDS,int Threads>
__device__ void backward_lowp_segment(AttentionParams const& p,int64_t begin,int64_t end,
    int64_t segment,int head,float* smem) {
    constexpr int W=D>V?D:V, Pitch=(W+15)/16*16+8;
    auto* cache=reinterpret_cast<LowMmaElement*>(smem);
    auto* probability=cache+3*Tile*Pitch;
    auto* adjoint=probability+Tile*(Tile+8);
    auto* bias_partial=reinterpret_cast<float*>(adjoint+Tile*(Tile+8));
    int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    backward_lowp_cache<Tile,Pitch,Threads>(p.inputs[0],cache,begin,end,head,D);
    backward_lowp_cache<Tile,Pitch,Threads>(p.inputs[1],cache+Tile*Pitch,begin,end,head,D);
    backward_lowp_cache<Tile,Pitch,Threads>(p.inputs[4],cache+2*Tile*Pitch,begin,end,head,V);
    __syncthreads();
    for(int row=warp*16;row<Tile;row+=(Threads/32)*16) {
        BackwardMma::FragmentC scores[Tile/8];
        #pragma unroll
        for(int n=0;n<Tile/8;++n) scores[n].clear();
        #pragma unroll
        for(int base=0;base<D;base+=16) {
            auto a=backward_load_a<Pitch,false>(cache,row,base,lane);
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                auto b=backward_load_b<Pitch,false>(cache+Tile*Pitch,n*8,base,lane);
                BackwardMma{}(scores[n],a,b,scores[n]);
            }
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            int qr=row+lane/4+r*8;
            int64_t q=begin+qr;
            float maximum=q<end?p.state.data[(q*H+head)*2]:0.f;
            float inverse=q<end?p.state.data[(q*H+head)*2+1]:0.f;
            float norm=0.f;
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    int c=n*8+(lane%4)*2+j;
                    int64_t k=begin+c;
                    float value=0.f;
                    if(q<end&&k<end&&(!CAUSAL||k<=q)&&inverse>0.f) {
                        float score=scores[n][r*2+j]*p.scale;
                        if constexpr (BIAS==1) score+=p.inputs[3].at(k,head);
                        if constexpr (BIAS==2) score+=p.inputs[3].at(segment,head,qr,c);
                        value=__expf(score-maximum)*inverse;
                    }
                    scores[n][r*2+j]=value;
                    norm+=value;
                }
            }
            #pragma unroll
            for(int shift=1;shift<4;shift*=2) norm+=__shfl_xor_sync(0xffffffff,norm,shift);
            float inv=norm>0.f?1.f/norm:0.f;
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    int c=n*8+(lane%4)*2+j;
                    probability[qr*(Tile+8)+c]=LowMmaElement(scores[n][r*2+j]*inv);
                }
            }
        }
    }
    __syncthreads();
    backward_lowp_cache<Tile,Pitch,Threads>(p.inputs[2],cache,begin,end,head,V);
    __syncthreads();
    for(int row=warp*16;row<Tile;row+=(Threads/32)*16) {
        BackwardMma::FragmentC values[Tile/8];
        #pragma unroll
        for(int n=0;n<Tile/8;++n) values[n].clear();
        #pragma unroll
        for(int base=0;base<V;base+=16) {
            auto a=backward_load_a<Pitch,false>(cache+2*Tile*Pitch,row,base,lane);
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                auto b=backward_load_b<Pitch,false>(cache,n*8,base,lane);
                BackwardMma{}(values[n],a,b,values[n]);
            }
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            int qr=row+lane/4+r*8;
            float total=0.f;
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    int c=n*8+(lane%4)*2+j;
                    total=fmaf(float(probability[qr*(Tile+8)+c]),values[n][r*2+j],total);
                }
            }
            #pragma unroll
            for(int shift=1;shift<4;shift*=2) total+=__shfl_xor_sync(0xffffffff,total,shift);
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    int c=n*8+(lane%4)*2+j;
                    float ds=float(probability[qr*(Tile+8)+c])*(values[n][r*2+j]-total);
                    adjoint[qr*(Tile+8)+c]=LowMmaElement(ds);
                    values[n][r*2+j]=ds;
                    if constexpr (BIAS==2 && (NEEDS&8))
                        if(begin+qr<end&&begin+c<end) p.outputs[3].at(segment,head,qr,c)=ds;
                }
            }
        }
        if constexpr (BIAS==1 && (NEEDS&8)) {
            #pragma unroll
            for(int n=0;n<Tile/8;++n) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    float value=values[n][j]+values[n][2+j];
                    #pragma unroll
                    for(int shift=4;shift<32;shift*=2) value+=__shfl_xor_sync(0xffffffff,value,shift);
                    if(lane<4) bias_partial[(row/16)*Tile+n*8+lane*2+j]=value;
                }
            }
        }
    }
    __syncthreads();
    if constexpr (BIAS==1 && (NEEDS&8))
        if(tid<Tile&&begin+tid<end) {
            float sum=0.f;
            #pragma unroll
            for(int r=0;r<Tile/16;++r) sum+=bias_partial[r*Tile+tid];
            p.outputs[3].at(begin+tid,head)=sum;
        }
    if constexpr (NEEDS&1) backward_lowp_output<Tile,Pitch,D,H,1,0,false,Threads>(p,cache,adjoint,begin,end,head);
    if constexpr (NEEDS&2) {
        backward_lowp_cache<Tile,Pitch,Threads>(p.inputs[0],cache,begin,end,head,D);
        __syncthreads();
        backward_lowp_output<Tile,Pitch,D,H,0,1,true,Threads>(p,cache,adjoint,begin,end,head);
    }
    if constexpr (NEEDS&4) backward_lowp_output<Tile,Pitch,V,H,2,2,true,Threads>(p,cache,probability,begin,end,head);
}

#endif

template <int H, bool COMPACT>
__device__ __forceinline__ decltype(auto) backward_feature(AttentionTensor const& tensor,
    int64_t row, int head, int channel, int width) {
    if constexpr (COMPACT && !LOW_PRECISION) return (tensor.data[(row*H+head)*width+channel]);
    return tensor.feature(row, head, channel);
}

// Cache Q/K/V/G in channel-major tiles shared by all products and outputs.
template <int Source, int Channels, int Width, int H, bool COMPACT, int Threads>
__device__ __forceinline__ void backward_cache(AttentionParams const& p, float* cache,
    int64_t begin, int64_t end, int head, int channel_base) {
    const int tid = threadIdx.x;
    if constexpr (!LOW_PRECISION && COMPACT && Channels%4==0) {
        if ((reinterpret_cast<uintptr_t>(p.inputs[Source].data)&15)==0) {
            for (int x=tid*4; x<32*Width; x+=Threads*4) {
                int r=x/Width, c=x%Width;
                float4 value={0,0,0,0};
                if (begin+r<end && channel_base+c<Channels)
                    value=*reinterpret_cast<const float4*>(&backward_feature<H, true>(p.inputs[Source], begin+r, head, channel_base+c, Channels));
                cache[c*33+r]=value.x;
                cache[(c+1)*33+r]=value.y;
                cache[(c+2)*33+r]=value.z;
                cache[(c+3)*33+r]=value.w;
            }
            return;
        }
    }
    for (int x=tid; x<32*Width; x+=Threads) {
        int r=x/Width, c=x%Width;
        cache[c*33+r]=begin+r<end && channel_base+c<Channels ?
            backward_feature<H, COMPACT>(p.inputs[Source], begin+r, head, channel_base+c, Channels):0.f;
    }
}

template <int D, int V, int H, bool COMPACT, int BIAS, bool CAUSAL, int Threads>
__device__ void backward_pairs(AttentionParams const& p, int64_t qb, int64_t kb,
    int64_t begin, int64_t end, int64_t segment, int head, float* cache, float* prob, float* adj) {
    constexpr int Width=BACKWARD_WIDTH, Plane=33*Width, Rows=1024/Threads;
    constexpr int Features=D>V ? D:V;
    int tid=threadIdx.x, lane=tid%32, row=(tid/32)*Rows;
    const int query_rows=int(min(int64_t(32),end-qb));
    const int key_rows=int(min(int64_t(32),end-kb));
    const bool transpose=query_rows<key_rows;
    const int primary=1 << (32-__clz(unsigned(min(query_rows,key_rows)-1)));
    const int group=32/primary, reduction_lane=lane/primary, small=lane%primary;
    const int score_large=transpose ? Plane:0, score_small=transpose ? 0:Plane;
    const int value_large=transpose ? 2*Plane:3*Plane, value_small=transpose ? 3*Plane:2*Plane;
    float score[Rows]={}, value[Rows]={};
    if (primary<32) {
        for (int i=tid; i<1024; i+=Threads) if ((transpose ? i/32:i%32)>=primary) {
            prob[i]=0.f;
            adj[i]=0.f;
        }
    }
    for (int base=0; base<Features; base+=Width) {
        if constexpr (Features>Width) {
            backward_cache<0,D,Width,H,COMPACT,Threads>(p,cache,qb,end,head,base);
            backward_cache<4,V,Width,H,COMPACT,Threads>(p,cache+3*Plane,qb,end,head,base);
        }
        backward_cache<1,D,Width,H,COMPACT,Threads>(p,cache+Plane,kb,end,head,base);
        backward_cache<2,V,Width,H,COMPACT,Threads>(p,cache+2*Plane,kb,end,head,base);
        __syncthreads();
        if ((transpose ? kb:qb)+row<end) {
            #pragma unroll 4
            for (int c=reduction_lane; c<Width && base+c<Features; c+=group) {
                float k=cache[score_small+c*33+small], v=cache[value_small+c*33+small];
                #pragma unroll
                for (int m=0; m<Rows; ++m) {
                    if (base+c<D) score[m]=multiply_add(cache[score_large+c*33+row+m],k,score[m]);
                    if (base+c<V) value[m]=multiply_add(cache[value_large+c*33+row+m],v,value[m]);
                }
            }
        }
        if constexpr (Features>Width) __syncthreads();
    }
    for (int offset=primary; offset<32; offset*=2) {
        #pragma unroll
        for (int m=0; m<Rows; ++m) {
            score[m]+=__shfl_xor_sync(0xffffffff,score[m],offset);
            value[m]+=__shfl_xor_sync(0xffffffff,value[m],offset);
        }
    }
    #pragma unroll
    for (int m=0; m<Rows; ++m) if (reduction_lane==0) {
        int qr=transpose ? small:row+m, kr=transpose ? row+m:small;
        int64_t q=qb+qr, k=kb+kr;
        float probability=0.f;
        if (q<end && k<end && (!CAUSAL || k<=q)) {
            float s=score[m]*p.scale;
            if constexpr (BIAS==1) s+=p.inputs[3].at(k,head);
            if constexpr (BIAS==2) s+=p.inputs[3].at(segment,head,q-begin,k-begin);
            float inverse=p.state.data[(q*H+head)*2+1];
            if (inverse>0.f) probability=__expf(s-p.state.data[(q*H+head)*2])*inverse;
        }
        prob[qr*32+kr]=probability;
        adj[qr*32+kr]=value[m];
    }
    __syncthreads();
}

template <int Rows, int D, int V, int H, bool COMPACT, int Threads>
__device__ __forceinline__ void backward_contract(AttentionParams const& p,
    float const* prob, float const* adj, float* cache, int64_t begin, int64_t end,
    int64_t qb, int64_t kb, int head, int output, int workers) {
    constexpr int Width=BACKWARD_WIDTH, Plane=33*Width;
    constexpr bool Cached=D<=Width && V<=Width;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int width=output<2 ? D:V, source=output==0 ? 1:output==1 ? 0:4;
    const int slot=output==0 ? 1:output==1 ? 0:3;
    const int64_t source_base=output==0 ? kb:qb, target_base=output==0 ? qb:kb;
    const int target_rows=int(min(int64_t(32),end-target_base));
    const int reduction=int(min(int64_t(32),end-source_base));
    for (int base=0; base<width; base+=Width) {
        const int count=min(Width,width-base), channel_tiles=(count+31)/32;
        if constexpr (!Cached) {
            for (int x=threadIdx.x; x<32*count; x+=Threads) {
                int r=x/count, c=x%count;
                cache[c*33+r]=source_base+r<end ? backward_feature<H,COMPACT>(p.inputs[source],source_base+r,head,base+c,width):0.f;
            }
            __syncthreads();
        }
        for (int tile=warp; tile<(target_rows+Rows-1)/Rows*channel_tiles; tile+=Threads/32) {
            const int row=tile/channel_tiles*Rows, channel=tile%channel_tiles*32+lane;
            if (channel<count) {
                float sum[Rows]={};
                #pragma unroll 4
                for (int k=0; k<reduction; ++k) {
                    float feature=cache[(Cached ? slot*Plane:0)+channel*33+k];
                    #pragma unroll
                    for (int m=0; m<Rows; ++m) {
                        float weight=output==2 ? prob[k*32+row+m]:output==0 ? adj[(row+m)*32+k]:adj[k*32+row+m];
                        sum[m]=multiply_add(weight,feature,sum[m]);
                    }
                }
                #pragma unroll
                for (int m=0; m<Rows; ++m) if (target_base+row+m<end) {
                    auto&& out=backward_feature<H,true>(p.outputs[output],target_base+row+m,head,base+channel,width);
                    bool accumulate=output==0 ? kb+32<end:qb>begin;
                    if constexpr (!LOW_PRECISION) {
                        if (output!=0 && end-begin>64 && workers>1)
                            atomicAdd(&out,sum[m]*(output<2 ? p.scale:1.f));
                        else out=sum[m]*(output<2 ? p.scale:1.f)+(accumulate ? out:0.f);
                    } else out=sum[m]*(output<2 ? p.scale:1.f)+(accumulate ? float(out):0.f);
                }
            }
        }
        if constexpr (!Cached) __syncthreads();
    }
}

template <int D, int V, int H, bool COMPACT, int BIAS, bool CAUSAL, int NEEDS, int Threads>
__device__ void backward_tiled(AttentionParams const& p, int64_t begin, int64_t end,
    int64_t segment, int head, float* smem, int worker, int workers) {
    constexpr int Width=BACKWARD_WIDTH, Plane=33*Width, Rows=1024/Threads;
    constexpr bool Cached=D<=Width && V<=Width;
    int tid=threadIdx.x, lane=tid%32, row=(tid/32)*Rows;
    float* pairs=smem+4*Plane;
    float* delta=pairs+BACKWARD_CACHE_TILES*2048;
    float* norm=delta+32;
    const int64_t last_key=begin+(end-begin-1)/32*32;
    const int64_t first_saved=max(begin,last_key-(BACKWARD_CACHE_TILES-1)*32);
    for (int64_t qb=begin+(BIAS!=2 && end-begin>64 ? worker*32:0); qb<end; qb+=(BIAS!=2 && end-begin>64 ? workers:1)*32) {
        if (tid<32) {
            delta[tid]=0.f;
            norm[tid]=0.f;
        }
        if constexpr (Cached) {
            backward_cache<0,D,Width,H,COMPACT,Threads>(p,smem,qb,end,head,0);
            backward_cache<4,V,Width,H,COMPACT,Threads>(p,smem+3*Plane,qb,end,head,0);
        }
        __syncthreads();
        for (int64_t kb=begin; kb<end; kb+=32) {
            float* prob=pairs+((kb-begin)/32%BACKWARD_CACHE_TILES)*2048;
            float* adj=prob+1024;
            backward_pairs<D,V,H,COMPACT,BIAS,CAUSAL,Threads>(p,qb,kb,begin,end,segment,head,smem,prob,adj);
            #pragma unroll
            for (int m=0; m<Rows; ++m) {
                float probability=prob[(row+m)*32+lane];
                float sum=probability*adj[(row+m)*32+lane];
                #pragma unroll
                for (int shift=16; shift>0; shift/=2) {
                    sum+=__shfl_xor_sync(0xffffffff,sum,shift);
                    probability+=__shfl_xor_sync(0xffffffff,probability,shift);
                }
                if (lane==0) {
                    delta[row+m]+=sum;
                    norm[row+m]+=probability;
                }
            }
            __syncthreads();
        }
        for (int64_t kb=last_key; kb>=begin; kb-=32) {
            float* prob=pairs+((kb-begin)/32%BACKWARD_CACHE_TILES)*2048;
            float* adj=prob+1024;
            if (kb<first_saved)
                backward_pairs<D,V,H,COMPACT,BIAS,CAUSAL,Threads>(p,qb,kb,begin,end,segment,head,smem,prob,adj);
            else if constexpr (Cached && (NEEDS&1)) {
                if (kb!=last_key)
                    backward_cache<1,D,Width,H,COMPACT,Threads>(p,smem+Plane,kb,end,head,0);
            }
            #pragma unroll
            for (int m=0; m<Rows; ++m) {
                float inverse=norm[row+m]>0.f ? 1.f/norm[row+m]:0.f;
                prob[(row+m)*32+lane]*=inverse;
                adj[(row+m)*32+lane]=prob[(row+m)*32+lane]*(adj[(row+m)*32+lane]-delta[row+m]*inverse);
            }
            __syncthreads();
            #pragma unroll
            for (int output=0; output<3; ++output) if (NEEDS&(1<<output)) {
                int64_t target_base=output==0 ? qb:kb;
                if (end-target_base<Rows)
                    backward_contract<1,D,V,H,COMPACT,Threads>(p,prob,adj,smem,begin,end,qb,kb,head,output,workers);
                else
                    backward_contract<Rows,D,V,H,COMPACT,Threads>(p,prob,adj,smem,begin,end,qb,kb,head,output,workers);
            }
            if constexpr (BIAS==1 && (NEEDS&8)) {
                #pragma unroll
                for (int m=0; m<Rows; ++m) {
                    float sum=adj[lane*32+row+m];
                    #pragma unroll
                    for (int shift=16; shift>0; shift/=2) sum+=__shfl_xor_sync(0xffffffff,sum,shift);
                    if (lane==0 && kb+row+m<end) {
                        auto&& out=p.outputs[3].at(kb+row+m,head);
#if !LOW_PRECISION
                        if (end-begin>64 && workers>1) atomicAdd(&out,sum);
                        else out=sum+(qb>begin ? out:0.f);
#else
                        out=sum+(qb>begin ? float(out):0.f);
#endif
                    }
                }
            }
            if constexpr (BIAS==2 && (NEEDS&8)) {
                #pragma unroll
                for (int m=0; m<Rows; ++m) if (qb+row+m<end && kb+lane<end)
                    p.outputs[3].at(segment,head,qb+row+m-begin,kb+lane-begin)=adj[(row+m)*32+lane];
            }
            __syncthreads();
        }
    }
}

#if !LOW_PRECISION
template<int Tile,int Pitch,int H,bool COMPACT,int Threads>
__device__ void backward_row_cache(AttentionTensor const& tensor,float* cache,
    int64_t begin,int64_t end,int head,int base,int width,int count) {
    if constexpr (COMPACT) {
        if (width%4==0 && count%4==0 && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
            for (int x=threadIdx.x*4;x<Tile*count;x+=Threads*4) {
                int r=x/count,c=x%count;
                float4 value={0.f,0.f,0.f,0.f};
                if (begin+r<end) value=*reinterpret_cast<const float4*>(&backward_feature<H,true>(tensor,begin+r,head,base+c,width));
                *reinterpret_cast<float4*>(cache+r*Pitch+c)=value;
            }
            return;
        }
    }
    for (int x=threadIdx.x;x<Tile*count;x+=Threads) {
        int r=x/count,c=x%count;
        cache[r*Pitch+c]=begin+r<end ? backward_feature<H,COMPACT>(tensor,begin+r,head,base+c,width):0.f;
    }
}

// A complete segment up to 64 tokens fits in bounded shared memory.
template <int Tile,int D, int V, int H, bool COMPACT, int BIAS, bool CAUSAL, int NEEDS,int Threads>
__device__ void backward_segment(AttentionParams const& p, int64_t begin, int64_t end,
    int64_t segment, int head, float* smem) {
    constexpr int Rows=Tile/(Threads/16), KeyTiles=Tile/16;
    int tid = threadIdx.x, lane = tid%16, row = (tid/16)*Rows, col = lane;
    constexpr int KeyStride = 16;
    constexpr int FeatureTile = 32;
    constexpr int CacheWidth = (D>V ? D:V);
    constexpr int CachePitch = (CacheWidth+3)/4*4+4;
    constexpr int Plane = Tile*(CachePitch>72 ? CachePitch:72);
    float* prob = smem+Plane;
    float* adj = prob+Tile*Tile;
    #pragma unroll
    for (int product = 0; product<2; ++product) {
        float matrix[Rows][KeyTiles] = {{0.f}};
        int width = product==0 ? D:V;
        for (int base = 0; base<width; base+=FeatureTile) {
            int count = min(FeatureTile, width-base);
            backward_row_cache<Tile,36,H,COMPACT,Threads>(p.inputs[product==0 ? 0:4],smem,begin,end,head,base,width,count);
            backward_row_cache<Tile,36,H,COMPACT,Threads>(p.inputs[product==0 ? 1:2],smem+Tile*36,begin,end,head,base,width,count);
            __syncthreads();
            if (begin+row<end) {
                #pragma unroll 4
                for (int c = 0; c<count; ++c) {
                    #pragma unroll
                    for (int m = 0; m<Rows; ++m) {
                        float a = smem[(row+m)*36+c];
                        #pragma unroll
                        for (int n = 0; n<KeyTiles; ++n) matrix[m][n] = multiply_add(a, smem[Tile*36+(col+n*KeyStride)*36+c], matrix[m][n]);
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int m = 0; m<Rows; ++m) {
            int64_t q = begin+row+m;
            if (product==0) {
                float maximum = q<end ? p.state.data[(q*H+head)*2]:0.f;
                float inverse = q<end ? p.state.data[(q*H+head)*2+1]:0.f;
                #pragma unroll
                for (int n = 0; n<KeyTiles; ++n) {
                    int64_t k = begin+col+n*KeyStride;
                    float probability = 0.f;
                    if (q<end && k<end && (!CAUSAL || k<=q) && inverse>0.f) {
                        float score = matrix[m][n]*p.scale;
                        if constexpr (BIAS==1) score+=p.inputs[3].at(k, head);
                        if constexpr (BIAS==2) score+=p.inputs[3].at(segment, head, row+m, col+n*KeyStride);
                        probability = __expf(score-maximum)*inverse;
                    }
                    prob[(row+m)*Tile+col+n*KeyStride] = probability;
                }
            } else {
                float delta = 0.f, norm = 0.f;
                #pragma unroll
                for (int n = 0; n<KeyTiles; ++n) {
                    delta = multiply_add(prob[(row+m)*Tile+col+n*KeyStride], matrix[m][n], delta);
                    norm+=prob[(row+m)*Tile+col+n*KeyStride];
                }
                #pragma unroll
                for (int shift = 1; shift<16; shift*=2) {
                    delta+=__shfl_xor_sync(0xffffffff, delta, shift);
                    norm+=__shfl_xor_sync(0xffffffff, norm, shift);
                }
                float inverse_norm = norm>0.f ? 1.f/norm : 0.f;
                delta*=inverse_norm;
                #pragma unroll
                for (int n = 0; n<KeyTiles; ++n) {
                    prob[(row+m)*Tile+col+n*KeyStride]*=inverse_norm;
                    float value = prob[(row+m)*Tile+col+n*KeyStride]*(matrix[m][n]-delta);
                    adj[(row+m)*Tile+col+n*KeyStride] = value;
                    if constexpr (BIAS==2 && (NEEDS&8)) if (q<end && begin+col+n*KeyStride<end)
                        p.outputs[3].at(segment, head, row+m, col+n*KeyStride) = value;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int output=0;output<3;++output) if (NEEDS&(1<<output)) {
        int width=output<2 ? D:V, source=output==0 ? 1:output==1 ? 0:4;
        constexpr int Columns=(CacheWidth+15)/16;
        backward_row_cache<Tile,CachePitch,H,COMPACT,Threads>(p.inputs[source],smem,begin,end,head,0,width,width);
        __syncthreads();
        float sum[Rows][Columns]={};
        if (begin+row<end) {
            #pragma unroll 2
            for (int k=0;k<end-begin;++k) {
                float weights[Rows];
                #pragma unroll
                for (int m=0;m<Rows;++m)
                    weights[m]=output==2 ? prob[k*Tile+row+m]:output==0 ? adj[(row+m)*Tile+k]:adj[k*Tile+row+m];
                #pragma unroll
                for (int c=0;c<Columns;++c) if (lane+c*16<width) {
                    float value=smem[k*CachePitch+lane+c*16];
                    #pragma unroll
                    for (int m=0;m<Rows;++m) sum[m][c]=multiply_add(weights[m],value,sum[m][c]);
                }
            }
            #pragma unroll
            for (int m=0;m<Rows;++m) if (begin+row+m<end) {
                #pragma unroll
                for (int c=0;c<Columns;++c) if (lane+c*16<width)
                    backward_feature<H,true>(p.outputs[output],begin+row+m,head,lane+c*16,width)=sum[m][c]*(output<2 ? p.scale:1.f);
            }
        }
        __syncthreads();
    }
    if constexpr (BIAS==1 && (NEEDS&8)) {
        int lane32 = tid%32, key = (tid/32)*(Tile/(Threads/32));
        #pragma unroll
        for (int m = 0; m<Tile/(Threads/32); ++m) {
            float sum = adj[lane32*Tile+key+m]+(lane32+32<Tile ? adj[(lane32+32)*Tile+key+m]:0.f);
            #pragma unroll
            for (int shift = 16; shift>0; shift/=2) sum+=__shfl_xor_sync(0xffffffff, sum, shift);
            if (lane32==0 && begin+key+m<end) p.outputs[3].at(begin+key+m, head) = sum;
        }
    }
}

#endif

template <int D, int V, int H, bool COMPACT, int BIAS, bool CAUSAL, int NEEDS, int Threads, int TileLimit, int Minimum>
__global__ __launch_bounds__(Threads, Minimum ? BACKWARD_LONG_BLOCKS : BACKWARD_MIN_BLOCKS) void attention_backward_kernel(const __grid_constant__ AttentionParams p) {
    extern __shared__ float smem[];
    constexpr int Workers=LOW_PRECISION || Minimum==0 ? 1:BACKWARD_WORKERS;
    int64_t segment = blockIdx.x/Workers;
    int worker=blockIdx.x%Workers;
    int head = blockIdx.y;
    int64_t begin = p.offset(segment), end = p.offset(segment+1);
    if constexpr (Minimum>0)
        if (end-begin<=Minimum || (end-begin<=64 && worker>0)) return;
#if LOW_PRECISION
    if constexpr (Minimum==0) if (end-begin>TileLimit) return;
#endif
#if !LOW_PRECISION
    if constexpr (Minimum==0) if (end-begin>TileLimit) {
        if constexpr (BACKWARD_WORKERS>1) if(end-begin>64) {
            #pragma unroll
            for (int output=1;output<4;++output) if ((NEEDS&(1<<output)) && (output<3 || BIAS==1)) {
                int width=output==1 ? D:output==2 ? V:1;
                for(int64_t x=threadIdx.x;x<(end-begin)*width;x+=Threads) {
                    int64_t row=begin+x/width; int column=x%width;
                    if(output<3) p.outputs[output].at(row,head,column)=0.f;
                    else p.outputs[output].at(row,head)=0.f;
                }
            }
        }
        return;
    }
#endif
    if constexpr (BIAS==2 && (NEEDS&8)) {
        auto out = p.outputs[3];
        for (int64_t x = threadIdx.x; x<out.size[2]*out.size[3]; x+=Threads)
            out.at(segment, head, x/out.size[3], x%out.size[3]) = 0.f;
        __syncthreads();
    }
    if (begin==end) return;
#if LOW_PRECISION
    if (!BACKWARD_RESIDENT || end-begin>TileLimit) {
        backward_tiled<D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem,0,1);
        return;
    }
#endif

#if LOW_PRECISION
    if (end-begin<=32) backward_lowp_segment<32,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
    else backward_lowp_segment<TileLimit,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
    return;
#endif
#if !LOW_PRECISION
    if constexpr (Minimum>0 && D<=128 && V<=128) {
        if (end-begin<=64) {
            backward_segment<64,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
            return;
        }
    }
    if constexpr (Minimum==0 && D<=128 && V<=128) {
        if (end-begin<=32) backward_segment<32,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
        else if (end-begin<=48) backward_segment<48,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
        else backward_segment<64,D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem);
    } else backward_tiled<D,V,H,COMPACT,BIAS,CAUSAL,NEEDS,Threads>(p,begin,end,segment,head,smem,worker,Workers);
#endif
}
'''


def attention_backward(inputs, rowptr, state, config, needs):
    parallel = inputs[0].dtype == torch.float32 and inputs[0].shape[0] > 32 * (rowptr.numel()-1) and config[6] != 2
    results = tuple(
        torch.empty(value.shape, device=value.device, dtype=value.dtype)
        if need and value is not None else None
        for value, need in zip(inputs[:4], needs)
    )
    if not any(value is not None and value.numel() for value in results):
        return results
    dim, value_dim, heads, scale, causal, steps, bias_kind = config
    mask = sum(1 << i for i, value in enumerate(results) if value is not None)
    compact = all(value.is_contiguous() for i, value in enumerate(inputs) if i != 3 and value is not None)
    key = dim, value_dim, heads, compact, bias_kind, causal, mask, inputs[0].device, inputs[0].dtype, parallel
    kernel = COMPILED.get(key)
    if kernel is None:
        properties = get_device_properties(inputs[0].device)
        maximum = max(dim, value_dim)
        low_precision = inputs[0].dtype != torch.float32
        threads = 128 if low_precision and maximum <= 64 else 256
        cache_width, cached_tiles = 64, 1
        minimum_blocks = 5 if threads == 128 else 4
        short_tile = 64 if low_precision else 48
        resident = True
        if low_precision:
            pitch = (maximum + 15) // 16 * 16 + 8
            shared = 3 * short_tile * pitch // 2 + short_tile * (short_tile + 8) + short_tile * short_tile // 16
            shared_capacity = properties.shared_memory_per_block_optin
            if shared * 4 > shared_capacity:
                short_tile = 32
                shared = 3 * short_tile * pitch // 2 + short_tile * (short_tile + 8) + short_tile * short_tile // 16
            resident = shared * 4 <= shared_capacity
            cached_tiles = 3
            stream_width = (shared_capacity // 4 - cached_tiles * 2048 - 64) // 132 // 32 * 32
            cache_width = min((maximum + 31) // 32 * 32, stream_width)
            if not resident:
                shared = 132 * cache_width + cached_tiles * 2048 + 64
            shared_per_sm = properties.shared_memory_per_multiprocessor
            minimum_blocks = min(4, shared_per_sm // (shared * 4))
        else:
            shared_capacity = properties.shared_memory_per_multiprocessor
            cache_width = (maximum + 31) // 32 * 32
            if 2 * (132 * cache_width + 2112) * 4 > shared_capacity:
                cache_width = 64
            resident_blocks = min(1024 // threads, shared_capacity // ((132 * cache_width + 2112) * 4))
            minimum_blocks = min(minimum_blocks, resident_blocks)
            cached_tiles = max(1, min(8, (shared_capacity // resident_blocks - (132 * cache_width + 64) * 4 - 1024) // 8192))
            shared = 132 * cache_width + cached_tiles * 2048 + 64
        if not low_precision and threads==256 and maximum<=128:
            shared=max(shared,64*max(72,(maximum+3)//4*4+4)+8192)
        stream_shared, long_blocks = shared, minimum_blocks
        if not low_precision and maximum<=128:
            shared=short_tile*max(72,(maximum+3)//4*4+4)+2*short_tile*short_tile
            minimum_blocks=min(8,shared_capacity//(shared*4+1024))
        if not low_precision:
            threads=128
        kernel_symbol = f"attention_backward_kernel<{dim},{value_dim},{heads},{str(compact).lower()},{bias_kind},{str(causal).lower()},{mask},{threads},{short_tile},0>"
        long_shared = 0
        if low_precision:
            long_tile = short_tile
            while resident:
                tile = long_tile + 16
                required = 3 * tile * pitch // 2 + tile * (tile + 8) + tile * tile // 16
                if required * 4 > shared_capacity:
                    break
                long_tile = tile
            resident_shared = 3 * long_tile * pitch // 2 + long_tile * (long_tile + 8) + long_tile * long_tile // 16
            long_shared = max(resident_shared if resident else 0, 132 * cache_width + cached_tiles * 2048 + 64)
            long_symbol = f"attention_backward_kernel<{dim},{value_dim},{heads},{str(compact).lower()},{bias_kind},{str(causal).lower()},{mask},256,{long_tile},{short_tile}>"
        else:
            long_shared=stream_shared
            long_symbol=f"attention_backward_kernel<{dim},{value_dim},{heads},{str(compact).lower()},{bias_kind},{str(causal).lower()},{mask},256,64,{short_tile}>"
        configure = ""
        if shared * 4 > 48 * 1024:
            configure = f'''C10_CUDA_CHECK(cudaFuncSetAttribute({kernel_symbol}, cudaFuncAttributeMaxDynamicSharedMemorySize, {shared*4}));
    C10_CUDA_CHECK(cudaFuncSetAttribute({kernel_symbol}, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));'''
        configure += f"C10_CUDA_CHECK(cudaFuncSetAttribute({long_symbol}, cudaFuncAttributeMaxDynamicSharedMemorySize, {long_shared*4}));"
        long_launch = f"{long_symbol}<<<dim3(p.groups*{4 if parallel else 1},{heads}),256,{long_shared*4},at::cuda::getCurrentCUDAStream()>>>(p);"
        binding = f'''
void run(std::vector<std::optional<torch::Tensor>> inputs, torch::Tensor rowptr,
         torch::Tensor state, std::vector<std::optional<torch::Tensor>> outputs, double scale) {{
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p=make_params(inputs,outputs,rowptr,state,scale);
    {kernel_symbol}<<<dim3(p.groups,{heads}),{threads},{shared*4},at::cuda::getCurrentCUDAStream()>>>(p);
    {long_launch}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
    m.def("run",&run);
    m.def("configure", []() {{ {configure} }});
}}
'''
        constants = f"#define BACKWARD_RESIDENT {int(resident)}\n#define BACKWARD_WORKERS {4 if parallel else 1}\n#define BACKWARD_LONG_BLOCKS {1 if low_precision else long_blocks}\n#define BACKWARD_WIDTH {cache_width}\n#define BACKWARD_CACHE_TILES {cached_tiles}\n#define BACKWARD_MIN_BLOCKS {minimum_blocks}\n"
        module = load_cuda_extension("attention_backward", constants + dtype_defines(inputs[0].dtype) + CUDA_SOURCE + BACKWARD_SOURCE + binding, arch=get_cuda_arch(inputs[0].device), use_fast_math=False, minimal_headers=True)
        with torch.cuda.device(inputs[0].device):
            module.configure()
        kernel = module.run
        COMPILED[key] = kernel
    kernel(inputs, rowptr, state, results, scale)
    return results
