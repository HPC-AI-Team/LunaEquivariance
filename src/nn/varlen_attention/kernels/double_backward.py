"""Fused CSR attention double backward with FP32 accumulation."""

import torch
from .utils import CUDA_SOURCE, dtype_defines
from ....utils.extension import get_cuda_arch, get_device_properties, load_cuda_extension

KERNELS = {}

FEATURE_SOURCE = r'''
__device__ __forceinline__ float higher_sum(float x) {
    #pragma unroll
    for (int offset = 16;  offset;  offset/=2) x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}
constexpr int FEATURE_WIDTH = (DIM > VALUE_DIM ? DIM : VALUE_DIM) <= FEATURE_LIMIT ? (DIM > VALUE_DIM ? DIM : VALUE_DIM) : 64;
__host__ __device__ constexpr int feature_slot(int input) {
    return input < 3 ? input : input < 8 ? input-1 : input < 12 ? input-2 : 10;
}
template <int INPUT> __device__ __forceinline__ float feature_value(
    AttentionParams const& p, float const* cache, int row, int channel,
    int64_t query_base, int64_t key_base, int head) {
    if constexpr (!(PRESENT & (1 << INPUT))) return 0.f;
    if constexpr (DIM<=FEATURE_WIDTH && VALUE_DIM<=FEATURE_WIDTH)
        return cache[feature_slot(INPUT)*33*FEATURE_WIDTH+channel*33+row];
    else return p.inputs[INPUT].feature((INPUT==0 || INPUT==4 || INPUT==5 || INPUT==9 || INPUT==13 ? query_base:key_base)+row, head, channel);
}
template <int INPUT, int SLOT = feature_slot(INPUT),
    int WIDTH = FEATURE_WIDTH, int ROWS = 32> __device__ __forceinline__ void cache_feature(
    AttentionParams const& p, float* cache, int64_t query_base, int64_t key_base, int64_t end, int head, int channel_base) {
    constexpr int channels = INPUT==0 || INPUT==1 || INPUT==5 || INPUT==6 || INPUT==9 || INPUT==10 ? DIM:VALUE_DIM;
    constexpr int Pack = 16 / sizeof(Storage);
    using Packed = std::conditional_t<LOW_PRECISION, uint4, float4>;
    const int64_t base = INPUT==0 || INPUT==4 || INPUT==5 || INPUT==9 || INPUT==13 ? query_base:key_base;
    if constexpr ((PRESENT & (1<<INPUT)) && channels%Pack==0 && WIDTH%Pack==0) {
        const auto& source = p.inputs[INPUT];
        if (source.stride[2]==1 && source.stride[0]%Pack==0 && source.stride[1]%Pack==0
           && (reinterpret_cast<uintptr_t>(source.data)&15)==0) {
            for (int i = threadIdx.x*Pack; i<ROWS*WIDTH; i+=blockDim.x*Pack) {
                const int row = i/WIDTH, c = i%WIDTH;
                Packed value = {0, 0, 0, 0};
                if (base+row<end && channel_base+c<channels)
                    value = *reinterpret_cast<const Packed*>(&source.feature(base+row, head, channel_base+c));
#if LOW_PRECISION
                cutlass::Array<LowMmaElement, 8> packed;
                *reinterpret_cast<uint4*>(&packed) = value;
                auto values = cutlass::NumericArrayConverter<float, LowMmaElement, 8>{}(packed);
#else
                const float values[4] = {value.x, value.y, value.z, value.w};
#endif
                #pragma unroll
                for (int j = 0; j<Pack; ++j)
                    cache[SLOT*(ROWS+1)*WIDTH+(c+j)*(ROWS+1)+row] = values[j];
            }
            return;
        }
    }
    for (int i = threadIdx.x; i<ROWS*WIDTH; i+=blockDim.x) {
        const int row = i/WIDTH, c = i%WIDTH;
        float value = 0;
        if constexpr (PRESENT & (1<<INPUT))
            if (base+row<end && channel_base+c<channels) value = p.inputs[INPUT].feature(base+row, head, channel_base+c);
        cache[SLOT*(ROWS+1)*WIDTH+c*(ROWS+1)+row] = value;
    }
}
template <int INPUT> __device__ __forceinline__ float higher_bias(AttentionParams const& p, int segment, int head,
    int64_t query, int64_t key, int64_t absolute_key) {
    if constexpr (!(PRESENT & (1<<INPUT)) || BIAS_KIND==0) return 0;
    else if constexpr (BIAS_KIND==1) return p.inputs[INPUT].at(absolute_key, head);
    else return p.inputs[INPUT].at(segment, head, query, key);
}
template <int TARGET, int COEFFICIENT, bool Atomic = false> __device__ __forceinline__ void higher_bias_gradient(
    AttentionParams const& p, float const* adjoints, int segment, int head, int64_t begin, int64_t end, int64_t qb,
        int64_t kb) {
    if constexpr ((NEEDS & (1<<TARGET)) && BIAS_KIND!=0) {
        const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
        #pragma unroll
        for (int c = 0; c<4; c++) {
            if constexpr (BIAS_KIND==1) {
                float value = higher_sum(adjoints[COEFFICIENT*1024+lane*32+row+c]);
                if (lane==0 && kb+row+c<end) {
                    auto&& out = p.outputs[TARGET].at(kb+row+c, head);
                    if constexpr(Atomic) atomicAdd(&p.outputs[TARGET].fp32_at(kb+row+c,head),value);
                    else out = value+(qb>begin ? out:0.f);
                }
            } else if (qb+row+c<end && kb+lane<end)
                p.outputs[TARGET].at(segment, head, qb-begin+row+c,
                    kb-begin+lane) = adjoints[COEFFICIENT*1024+(row+c)*32+lane];
        }
    }
}
'''


LOWP_SOURCE = r"""
#if LOW_PRECISION
using DoubleMma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    LowMmaElement,cutlass::layout::RowMajor,LowMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;

template<int Pitch, bool Transpose>
__device__ __forceinline__ DoubleMma::FragmentA double_load_a(
    LowMmaElement const* pointer, int row, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/16)*8)*Pitch+row+(lane/8%2)*8
                           : (row+lane%16)*Pitch+base+(lane/16)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    DoubleMma::FragmentA result;
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
__device__ __forceinline__ DoubleMma::FragmentB double_load_b(
    LowMmaElement const* pointer, int column, int base, int lane) {
    int offset = Transpose ? (base+lane%8+(lane/8%2)*8)*Pitch+column
                           : (column+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address = unsigned(__cvta_generic_to_shared(pointer+offset));
    DoubleMma::FragmentB result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    if constexpr (Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]) : "r"(address));
    return result;
}

constexpr int DOUBLE_WIDTH=((DIM>VALUE_DIM ? DIM:VALUE_DIM)+15)/16*16+8;
constexpr int DOUBLE_PLANE=32*DOUBLE_WIDTH;
constexpr int DOUBLE_WORK_FLOATS=(4*32*40+DOUBLE_PLANE)/2+64;

template<int Target,int C0,int F0,int C1=-1,int F1=-1,bool Transpose=false>
__device__ void double_register_output(AttentionParams const& p,LowMmaElement* features,
    LowMmaElement const* coefficients,LowMmaElement* temporary,int64_t begin,int n,int head) {
    if constexpr(NEEDS&(1<<Target)) {
        constexpr int Channels=Target<2 ? DIM:VALUE_DIM,Tiles=(Channels+7)/8;
        int lane=threadIdx.x%32;
        for(int tile=threadIdx.x/32;tile<2*Tiles;tile+=4) {
            int row=tile/Tiles*16,col=tile%Tiles*8;
            DoubleMma::FragmentC value;value.clear();
            #pragma unroll
            for(int base=0;base<32;base+=16) {
                auto a=double_load_a<40,Transpose>(coefficients+C0*32*40,row,base,lane);
                auto b=double_load_b<DOUBLE_WIDTH,true>(features+F0*DOUBLE_PLANE,col,base,lane);
                DoubleMma{}(value,a,b,value);
                if constexpr(C1>=0) {
                    a=double_load_a<40,Transpose>(coefficients+C1*32*40,row,base,lane);
                    b=double_load_b<DOUBLE_WIDTH,true>(features+F1*DOUBLE_PLANE,col,base,lane);
                    DoubleMma{}(value,a,b,value);
                }
            }
            #pragma unroll
            for(int i=0;i<4;++i) {
                int r=row+lane/4+(i/2)*8,c=col+(lane%4)*2+i%2;
                temporary[r*DOUBLE_WIDTH+c]=LowMmaElement(value[i]*(Target<2 ? p.scale:1.f));
            }
        }
        __syncthreads();
        if (Channels%8==0 && !p.outputs[Target].fp32) {
            for(int x=threadIdx.x*8;x<32*Channels;x+=128*8) {
                int r=x/Channels,c=x%Channels;
                if(r<n)
                    *reinterpret_cast<uint4*>(p.outputs[Target].data+((begin+r)*HEADS+head)*Channels+c)=
                        *reinterpret_cast<const uint4*>(temporary+r*DOUBLE_WIDTH+c);
            }
        } else {
            for(int x=threadIdx.x;x<n*Channels;x+=128) {
                int r=x/Channels,c=x%Channels;
                p.outputs[Target].store(begin+r,head,c,float(temporary[r*DOUBLE_WIDTH+c]));
            }
        }
        __syncthreads();
    }
}

__device__ void double_lowp(AttentionParams const& p,int segment,int head,int64_t begin,int n) {
    extern __shared__ float shared[];
    auto* features=reinterpret_cast<LowMmaElement*>(shared);
    auto* coefficients=features+7*DOUBLE_PLANE;
    auto* temporary=coefficients+4*32*40;
    float* bias_partial=reinterpret_cast<float*>(temporary+DOUBLE_PLANE);
    constexpr int slots[7]={0,1,2,4,5,6,7};
    #pragma unroll
    for(int slot=0;slot<7;++slot) {
        int input=slots[slot],channels=slot==0 || slot==1 || slot==4 || slot==5 ? DIM:VALUE_DIM;
        auto const& tensor=p.inputs[input];
        if(channels%8==0 && tensor.stride[2]==1 && tensor.stride[0]%8==0 && tensor.stride[1]%8==0
            && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
            for(int x=threadIdx.x*8;x<DOUBLE_PLANE;x+=128*8) {
                int row=x/DOUBLE_WIDTH,c=x%DOUBLE_WIDTH;
                uint4 value={0,0,0,0};
                if(row<n && c<channels && (PRESENT&(1<<input)))
                    value=*reinterpret_cast<uint4 const*>(tensor.data+(begin+row)*tensor.stride[0]+head*tensor.stride[1]+c);
                *reinterpret_cast<uint4*>(features+slot*DOUBLE_PLANE+x)=value;
            }
        } else {
            for(int x=threadIdx.x;x<DOUBLE_PLANE;x+=128) {
                int row=x/DOUBLE_WIDTH,c=x%DOUBLE_WIDTH;
                features[slot*DOUBLE_PLANE+x]=LowMmaElement(row<n && c<channels && (PRESENT&(1<<input)) ?
                    tensor.load(begin+row,head,c):0.f);
            }
        }
    }
    if constexpr(BIAS_KIND==2 && (NEEDS&8))
        for(int64_t t=threadIdx.x;t<p.outputs[3].size[2]*p.outputs[3].size[3];t+=128)
            p.outputs[3].at(segment,head,t/p.outputs[3].size[3],t%p.outputs[3].size[3])=0.f;
    __syncthreads();
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(threadIdx.x<64) {
        int row=warp*16;
        DoubleMma::FragmentC scores[4],directions[4],values[4],value_directions[4];
        #pragma unroll
        for(int j=0;j<4;++j) {scores[j].clear();directions[j].clear();values[j].clear();value_directions[j].clear();}
        #pragma unroll
        for(int base=0;base<DOUBLE_WIDTH-8;base+=16) {
            if(base<DIM) {
                auto query=double_load_a<DOUBLE_WIDTH,false>(features,row,base,lane);
                auto rq=double_load_a<DOUBLE_WIDTH,false>(features+4*DOUBLE_PLANE,row,base,lane);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    auto key=double_load_b<DOUBLE_WIDTH,false>(features+DOUBLE_PLANE,j*8,base,lane);
                    auto rk=double_load_b<DOUBLE_WIDTH,false>(features+5*DOUBLE_PLANE,j*8,base,lane);
                    DoubleMma{}(scores[j],query,key,scores[j]);
                    DoubleMma{}(directions[j],rq,key,directions[j]);
                    DoubleMma{}(directions[j],query,rk,directions[j]);
                }
            }
            if(base<VALUE_DIM) {
                auto g=double_load_a<DOUBLE_WIDTH,false>(features+3*DOUBLE_PLANE,row,base,lane);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    auto v=double_load_b<DOUBLE_WIDTH,false>(features+2*DOUBLE_PLANE,j*8,base,lane);
                    auto rv=double_load_b<DOUBLE_WIDTH,false>(features+6*DOUBLE_PLANE,j*8,base,lane);
                    DoubleMma{}(values[j],g,v,values[j]);
                    DoubleMma{}(value_directions[j],g,rv,value_directions[j]);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            int qr=row+lane/4+r*8;
            float maximum=qr<n ? p.state.at(begin+qr,head,0):0.f;
            float inverse=qr<n ? p.state.at(begin+qr,head,1):0.f;
            float norm=0.f,mean=0.f;
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int c=0;c<2;++c) {
                    int key=j*8+(lane%4)*2+c,index=r*2+c;
                    float probability=0.f,a=0.f;
                    if(qr<n && key<n && (!CAUSAL || key<=qr) && inverse>0.f) {
                        float score=scores[j][index]*p.scale+higher_bias<3>(p,segment,head,qr,key,begin+key);
                        probability=__expf(score-maximum)*inverse;
                        a=directions[j][index]*p.scale+higher_bias<8>(p,segment,head,qr,key,begin+key);
                    }
                    scores[j][index]=probability;
                    directions[j][index]=a;
                    norm+=probability;
                    mean=fmaf(probability,a,mean);
                }
            #pragma unroll
            for(int shift=1;shift<4;shift*=2) {
                norm+=__shfl_xor_sync(0xffffffff,norm,shift);
                mean+=__shfl_xor_sync(0xffffffff,mean,shift);
            }
            float inv=norm>0.f ? 1.f/norm:0.f;
            mean*=inv;
            float delta=0.f,delta_r=0.f;
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int c=0;c<2;++c) {
                    int index=r*2+c;
                    float prob=scores[j][index]*inv,pr=prob*(directions[j][index]-mean);
                    scores[j][index]=prob;
                    directions[j][index]=pr;
                    delta=fmaf(prob,values[j][index],delta);
                    delta_r=fmaf(pr,values[j][index],fmaf(prob,value_directions[j][index],delta_r));
                }
            #pragma unroll
            for(int shift=1;shift<4;shift*=2) {
                delta+=__shfl_xor_sync(0xffffffff,delta,shift);
                delta_r+=__shfl_xor_sync(0xffffffff,delta_r,shift);
            }
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int c=0;c<2;++c) {
                    int key=j*8+(lane%4)*2+c,index=r*2+c,pos=qr*40+key;
                    float prob=scores[j][index],pr=directions[j][index];
                    float a=prob*(values[j][index]-delta);
                    float ar=fmaf(pr,values[j][index]-delta,prob*(value_directions[j][index]-delta_r));
                    coefficients[pos]=LowMmaElement(prob);
                    coefficients[32*40+pos]=LowMmaElement(pr);
                    coefficients[2*32*40+pos]=LowMmaElement(a);
                    coefficients[3*32*40+pos]=LowMmaElement(ar);
                    values[j][index]=ar;
                    if constexpr(BIAS_KIND==2 && (NEEDS&8)) if(qr<n && key<n)
                        p.outputs[3].at(segment,head,qr,key)=ar;
                }
        }
        if constexpr(BIAS_KIND==1 && (NEEDS&8)) {
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int c=0;c<2;++c) {
                    float sum=values[j][c]+values[j][2+c];
                    #pragma unroll
                    for(int shift=4;shift<32;shift*=2) sum+=__shfl_xor_sync(0xffffffff,sum,shift);
                    if(lane<4) bias_partial[warp*32+j*8+lane*2+c]=sum;
                }
        }
    }
    __syncthreads();
    if constexpr(BIAS_KIND==1 && (NEEDS&8)) if(threadIdx.x<n)
        p.outputs[3].at(begin+threadIdx.x,head)=bias_partial[threadIdx.x]+bias_partial[32+threadIdx.x];
    double_register_output<0,3,1,2,5>(p,features,coefficients,temporary,begin,n,head);
    double_register_output<1,3,0,2,4,true>(p,features,coefficients,temporary,begin,n,head);
    double_register_output<2,1,3,-1,-1,true>(p,features,coefficients,temporary,begin,n,head);
    double_register_output<4,1,2,0,6>(p,features,coefficients,temporary,begin,n,head);
}
#endif
"""

NATIVE_RESIDENT = r'''
#if LOW_PRECISION
template<int Tile> struct DoubleResident {
static constexpr int DOUBLE_WIDTH=((DIM>VALUE_DIM ? DIM:VALUE_DIM)+15)/16*16+8;
static constexpr int DOUBLE_PLANE=Tile*DOUBLE_WIDTH;
static constexpr int DOUBLE_WORK_FLOATS=(4*Tile*(Tile+8)+DOUBLE_PLANE)/2+(Tile/16)*Tile;

template<int Target,int C0,int F0,int C1=-1,int F1=-1,bool Transpose=false>
static __device__ void double_register_output(AttentionParams const& p,LowMmaElement* features,
    LowMmaElement const* coefficients,LowMmaElement* temporary,int64_t begin,int n,int head) {
    if constexpr(NEEDS&(1<<Target)) {
        static constexpr int Channels=Target<2 ? DIM:VALUE_DIM,Tiles=(Channels+7)/8;
        int lane=threadIdx.x%32;
        for(int tile=threadIdx.x/32;tile<(Tile/16)*Tiles;tile+=4) {
            int row=tile/Tiles*16,col=tile%Tiles*8;
            DoubleMma::FragmentC value;value.clear();
            #pragma unroll
            for(int base=0;base<Tile;base+=16) {
                auto a=double_load_a<Tile+8,Transpose>(coefficients+C0*Tile*(Tile+8),row,base,lane);
                auto b=double_load_b<DOUBLE_WIDTH,true>(features+F0*DOUBLE_PLANE,col,base,lane);
                DoubleMma{}(value,a,b,value);
                if constexpr(C1>=0) {
                    a=double_load_a<Tile+8,Transpose>(coefficients+C1*Tile*(Tile+8),row,base,lane);
                    b=double_load_b<DOUBLE_WIDTH,true>(features+F1*DOUBLE_PLANE,col,base,lane);
                    DoubleMma{}(value,a,b,value);
                }
            }
            #pragma unroll
            for(int i=0;i<4;++i) {
                int r=row+lane/4+(i/2)*8,c=col+(lane%4)*2+i%2;
                temporary[r*DOUBLE_WIDTH+c]=LowMmaElement(value[i]*(Target<2 ? p.scale:1.f));
            }
        }
        __syncthreads();
        if (Channels%8==0 && !p.outputs[Target].fp32) {
            for(int x=threadIdx.x*8;x<Tile*Channels;x+=128*8) {
                int r=x/Channels,c=x%Channels;
                if(r<n)
                    *reinterpret_cast<uint4*>(p.outputs[Target].data+((begin+r)*HEADS+head)*Channels+c)=
                        *reinterpret_cast<const uint4*>(temporary+r*DOUBLE_WIDTH+c);
            }
        } else {
            for(int x=threadIdx.x;x<n*Channels;x+=128) {
                int r=x/Channels,c=x%Channels;
                p.outputs[Target].store(begin+r,head,c,float(temporary[r*DOUBLE_WIDTH+c]));
            }
        }
        __syncthreads();
    }
}

template<int Input,int Slot>
static __device__ void load_feature(AttentionParams const& p,LowMmaElement* features,
    int64_t begin,int n,int head) {
    constexpr int Channels=Input==0 || Input==1 || Input==5 || Input==6 ? DIM:VALUE_DIM;
    auto const& tensor=p.inputs[Input];
    if(Channels%8==0 && tensor.stride[2]==1 && tensor.stride[0]%8==0 && tensor.stride[1]%8==0
        && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
        for(int x=threadIdx.x*8;x<DOUBLE_PLANE;x+=128*8) {
            int row=x/DOUBLE_WIDTH,c=x%DOUBLE_WIDTH;
            uint4 value={0,0,0,0};
            if(row<n && c<Channels && (PRESENT&(1<<Input)))
                value=*reinterpret_cast<const uint4*>(tensor.data+(begin+row)*tensor.stride[0]+head*tensor.stride[1]+c);
            *reinterpret_cast<uint4*>(features+Slot*DOUBLE_PLANE+x)=value;
        }
    } else {
        for(int x=threadIdx.x;x<DOUBLE_PLANE;x+=128) {
            int row=x/DOUBLE_WIDTH,c=x%DOUBLE_WIDTH;
            features[Slot*DOUBLE_PLANE+x]=LowMmaElement(row<n && c<Channels && (PRESENT&(1<<Input)) ?
                tensor.load(begin+row,head,c):0.f);
        }
    }
}

static __device__ void double_lowp(AttentionParams const& p,int segment,int head,int64_t begin,int n) {
    extern __shared__ float shared[];
    auto* features=reinterpret_cast<LowMmaElement*>(shared);
    auto* coefficients=features+4*DOUBLE_PLANE;
    auto* temporary=coefficients+4*Tile*(Tile+8);
    float* bias_partial=reinterpret_cast<float*>(temporary+DOUBLE_PLANE);
    load_feature<0,0>(p,features,begin,n,head);
    load_feature<1,1>(p,features,begin,n,head);
    load_feature<5,2>(p,features,begin,n,head);
    load_feature<6,3>(p,features,begin,n,head);
    if constexpr(BIAS_KIND==2 && (NEEDS&8))
        for(int64_t t=threadIdx.x;t<p.outputs[3].size[2]*p.outputs[3].size[3];t+=128)
            p.outputs[3].at(segment,head,t/p.outputs[3].size[3],t%p.outputs[3].size[3])=0.f;
    __syncthreads();
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    {
        int stripe=warp,row=warp*16;
        DoubleMma::FragmentC scores[Tile/8],directions[Tile/8],values[Tile/8],value_directions[Tile/8];
        #pragma unroll
        for(int j=0;j<Tile/8;++j) {scores[j].clear();directions[j].clear();values[j].clear();value_directions[j].clear();}
        if(warp<Tile/16) {
            #pragma unroll
            for(int base=0;base<DOUBLE_WIDTH-8;base+=16) {
                if(base<DIM) {
                    auto query=double_load_a<DOUBLE_WIDTH,false>(features,row,base,lane);
                    auto rq=double_load_a<DOUBLE_WIDTH,false>(features+2*DOUBLE_PLANE,row,base,lane);
                    #pragma unroll
                    for(int j=0;j<Tile/8;++j) {
                        auto key=double_load_b<DOUBLE_WIDTH,false>(features+DOUBLE_PLANE,j*8,base,lane);
                        auto rk=double_load_b<DOUBLE_WIDTH,false>(features+3*DOUBLE_PLANE,j*8,base,lane);
                        DoubleMma{}(scores[j],query,key,scores[j]);
                        DoubleMma{}(directions[j],rq,key,directions[j]);
                        DoubleMma{}(directions[j],query,rk,directions[j]);
                    }
            }
            }
        }
        // Score fragments stay in registers while the feature cache is reused.
        __syncthreads();
        load_feature<4,0>(p,features,begin,n,head);
        load_feature<2,1>(p,features,begin,n,head);
        load_feature<7,2>(p,features,begin,n,head);
        __syncthreads();
        if(warp<Tile/16) {
            #pragma unroll
            for(int base=0;base<DOUBLE_WIDTH-8;base+=16) {
                if(base<VALUE_DIM) {
                    auto g=double_load_a<DOUBLE_WIDTH,false>(features,row,base,lane);
                    #pragma unroll
                    for(int j=0;j<Tile/8;++j) {
                        auto v=double_load_b<DOUBLE_WIDTH,false>(features+DOUBLE_PLANE,j*8,base,lane);
                        auto rv=double_load_b<DOUBLE_WIDTH,false>(features+2*DOUBLE_PLANE,j*8,base,lane);
                        DoubleMma{}(values[j],g,v,values[j]);
                        DoubleMma{}(value_directions[j],g,rv,value_directions[j]);
                    }
            }
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                int qr=row+lane/4+r*8;
                float maximum=qr<n ? p.state.at(begin+qr,head,0):0.f;
                float inverse=qr<n ? p.state.at(begin+qr,head,1):0.f;
                float norm=0.f,mean=0.f;
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,index=r*2+c;
                        float probability=0.f,a=0.f;
                        if(qr<n && key<n && (!CAUSAL || key<=qr) && inverse>0.f) {
                            float score=scores[j][index]*p.scale+higher_bias<3>(p,segment,head,qr,key,begin+key);
                            probability=__expf(score-maximum)*inverse;
                            a=directions[j][index]*p.scale+higher_bias<8>(p,segment,head,qr,key,begin+key);
                        }
                        scores[j][index]=probability;
                        directions[j][index]=a;
                        norm+=probability;
                        mean=fmaf(probability,a,mean);
                    }
                #pragma unroll
                for(int shift=1;shift<4;shift*=2) {
                    norm+=__shfl_xor_sync(0xffffffff,norm,shift);
                    mean+=__shfl_xor_sync(0xffffffff,mean,shift);
                }
                float inv=norm>0.f ? 1.f/norm:0.f;
                mean*=inv;
                float delta=0.f,delta_r=0.f;
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int index=r*2+c;
                        float prob=scores[j][index]*inv,pr=prob*(directions[j][index]-mean);
                        scores[j][index]=prob;
                        directions[j][index]=pr;
                        delta=fmaf(prob,values[j][index],delta);
                        delta_r=fmaf(pr,values[j][index],fmaf(prob,value_directions[j][index],delta_r));
                    }
                #pragma unroll
                for(int shift=1;shift<4;shift*=2) {
                    delta+=__shfl_xor_sync(0xffffffff,delta,shift);
                    delta_r+=__shfl_xor_sync(0xffffffff,delta_r,shift);
                }
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,index=r*2+c,pos=qr*(Tile+8)+key;
                        float prob=scores[j][index],pr=directions[j][index];
                        float a=prob*(values[j][index]-delta);
                        float ar=fmaf(pr,values[j][index]-delta,prob*(value_directions[j][index]-delta_r));
                        coefficients[pos]=LowMmaElement(prob);
                        coefficients[Tile*(Tile+8)+pos]=LowMmaElement(pr);
                        coefficients[2*Tile*(Tile+8)+pos]=LowMmaElement(a);
                        coefficients[3*Tile*(Tile+8)+pos]=LowMmaElement(ar);
                        values[j][index]=ar;
                        if constexpr(BIAS_KIND==2 && (NEEDS&8)) if(qr<n && key<n)
                            p.outputs[3].at(segment,head,qr,key)=ar;
                    }
            }
            if constexpr(BIAS_KIND==1 && (NEEDS&8)) {
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        float sum=values[j][c]+values[j][2+c];
                        #pragma unroll
                        for(int shift=4;shift<32;shift*=2) sum+=__shfl_xor_sync(0xffffffff,sum,shift);
                        if(lane<4) bias_partial[stripe*Tile+j*8+lane*2+c]=sum;
                    }
            }
        }
    }
    __syncthreads();
    if constexpr(BIAS_KIND==1 && (NEEDS&8)) if(threadIdx.x<n) {
        float sum=0.f;
        for(int stripe=0;stripe<Tile/16;++stripe) sum+=bias_partial[stripe*Tile+threadIdx.x];
        p.outputs[3].at(begin+threadIdx.x,head)=sum;
    }
    double_register_output<2,1,0,-1,-1,true>(p,features,coefficients,temporary,begin,n,head);
    double_register_output<4,1,1,0,2>(p,features,coefficients,temporary,begin,n,head);
    load_feature<0,0>(p,features,begin,n,head);
    load_feature<1,1>(p,features,begin,n,head);
    load_feature<5,2>(p,features,begin,n,head);
    load_feature<6,3>(p,features,begin,n,head);
    __syncthreads();
    double_register_output<0,3,1,2,3>(p,features,coefficients,temporary,begin,n,head);
    double_register_output<1,3,0,2,2,true>(p,features,coefficients,temporary,begin,n,head);
}

static constexpr int Bytes=4*DOUBLE_PLANE*sizeof(LowMmaElement)+DOUBLE_WORK_FLOATS*sizeof(float);
};
#endif
'''

SOURCE = r'''
#if LOW_PRECISION
constexpr bool NATIVE_SHORT_SUPPORTED=7*DOUBLE_PLANE*sizeof(LowMmaElement)+4*DOUBLE_WORK_FLOATS<=RESIDENT_BUDGET;
#else
constexpr bool NATIVE_SHORT_SUPPORTED=false;
#endif

#if LOW_PRECISION
constexpr int DOUBLE_CACHE_PITCH=FEATURE_WIDTH+8;
constexpr int DOUBLE_CACHE_PLANE=32*DOUBLE_CACHE_PITCH;
constexpr int DOUBLE_LONG_BYTES=2*7*DOUBLE_CACHE_PLANE+4*(4096+32)+2*4*32*40;

template<int Input>
__device__ void double_cache(AttentionParams const& p,LowMmaElement* cache,
    int64_t qb,int64_t kb,int64_t end,int head,int channel_base) {
    constexpr int Channels=Input==0 || Input==1 || Input==5 || Input==6 ? DIM:VALUE_DIM;
    constexpr int Slot=feature_slot(Input);
    int64_t base=Input==0 || Input==4 || Input==5 ? qb:kb;
    if constexpr((PRESENT&(1<<Input)) && Channels%8==0) {
        auto const& source=p.inputs[Input];
        if(!source.fp32 && source.stride[2]==1 && source.stride[0]%8==0 && source.stride[1]%8==0
            && (reinterpret_cast<uintptr_t>(source.data)&15)==0) {
            for(int i=threadIdx.x*8;i<32*FEATURE_WIDTH;i+=blockDim.x*8) {
                int row=i/FEATURE_WIDTH,c=i%FEATURE_WIDTH;
                uint4 value={0,0,0,0};
                if(base+row<end && channel_base+c<Channels)
                    value=*reinterpret_cast<const uint4*>(&source.feature(base+row,head,channel_base+c));
                *reinterpret_cast<uint4*>(cache+Slot*DOUBLE_CACHE_PLANE+row*DOUBLE_CACHE_PITCH+c)=value;
            }
            return;
        }
    }
    for(int i=threadIdx.x;i<32*FEATURE_WIDTH;i+=blockDim.x) {
        int row=i/FEATURE_WIDTH,c=i%FEATURE_WIDTH;
        float value=0;
        if constexpr(PRESENT&(1<<Input))
            if(base+row<end && channel_base+c<Channels)
                value=float(p.inputs[Input].feature(base+row,head,channel_base+c));
        cache[Slot*DOUBLE_CACHE_PLANE+row*DOUBLE_CACHE_PITCH+c]=LowMmaElement(value);
    }
}

__device__ void double_cache_inputs(AttentionParams const& p,LowMmaElement* cache,
    int64_t qb,int64_t kb,int64_t end,int head,int base) {
    double_cache<0>(p,cache,qb,kb,end,head,base);
    double_cache<1>(p,cache,qb,kb,end,head,base);
    double_cache<2>(p,cache,qb,kb,end,head,base);
    double_cache<4>(p,cache,qb,kb,end,head,base);
    double_cache<5>(p,cache,qb,kb,end,head,base);
    double_cache<6>(p,cache,qb,kb,end,head,base);
    double_cache<7>(p,cache,qb,kb,end,head,base);
    __syncthreads();
}

__device__ void double_pairs(AttentionParams const& p,LowMmaElement* cache,
    float* probability,float* adjoint,int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=warp/4*16,col=warp%4*8;
    DoubleMma::FragmentC score,direction,value,value_direction;
    score.clear();direction.clear();value.clear();value_direction.clear();
    for(int chunk=0;chunk<max(DIM,VALUE_DIM);chunk+=FEATURE_WIDTH) {
        double_cache_inputs(p,cache,qb,kb,end,head,chunk);
        #pragma unroll
        for(int c=0;c<FEATURE_WIDTH;c+=16) {
            if(chunk+c<DIM) {
                auto q=double_load_a<DOUBLE_CACHE_PITCH,false>(cache,row,c,lane);
                auto k=double_load_b<DOUBLE_CACHE_PITCH,false>(cache+DOUBLE_CACHE_PLANE,col,c,lane);
                DoubleMma{}(score,q,k,score);
                if constexpr(PRESENT&(1<<5)) {
                    auto rq=double_load_a<DOUBLE_CACHE_PITCH,false>(cache+4*DOUBLE_CACHE_PLANE,row,c,lane);
                    DoubleMma{}(direction,rq,k,direction);
                }
                if constexpr(PRESENT&(1<<6)) {
                    auto rk=double_load_b<DOUBLE_CACHE_PITCH,false>(cache+5*DOUBLE_CACHE_PLANE,col,c,lane);
                    DoubleMma{}(direction,q,rk,direction);
                }
            }
            if(chunk+c<VALUE_DIM) {
                auto g=double_load_a<DOUBLE_CACHE_PITCH,false>(cache+3*DOUBLE_CACHE_PLANE,row,c,lane);
                auto v=double_load_b<DOUBLE_CACHE_PITCH,false>(cache+2*DOUBLE_CACHE_PLANE,col,c,lane);
                DoubleMma{}(value,g,v,value);
                if constexpr(PRESENT&(1<<7)) {
                    auto rv=double_load_b<DOUBLE_CACHE_PITCH,false>(cache+6*DOUBLE_CACHE_PLANE,col,c,lane);
                    DoubleMma{}(value_direction,g,rv,value_direction);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        int r=row+lane/4+(i/2)*8,c=col+(lane%4)*2+i%2,pos=r*32+c;
        float prob=0,a=0;
        if(qb+r<end && kb+c<end && (!CAUSAL || kb+c<=qb+r)) {
            float s=score[i]*p.scale+higher_bias<3>(p,segment,head,qb-begin+r,kb-begin+c,kb+c);
            float inv=p.state.at(qb+r,head,1);
            if(inv>0 && s>-INFINITY) prob=__expf(s-p.state.at(qb+r,head,0))*inv;
            a=direction[i]*p.scale+higher_bias<8>(p,segment,head,qb-begin+r,kb-begin+c,kb+c);
        }
        probability[pos]=prob;probability[1024+pos]=a;
        adjoint[pos]=value[i];adjoint[1024+pos]=value_direction[i];
    }
    __syncthreads();
}

template<int Target,int C0,int F0,int C1,int F1,bool Transpose,bool Atomic>
__device__ void double_tile_output(AttentionParams const& p,LowMmaElement const* cache,
    LowMmaElement const* coefficients,int64_t begin,int64_t end,int64_t qb,int64_t kb,int head,int channel_base) {
    if constexpr(NEEDS&(1<<Target)) {
        constexpr int Channels=Target<2 ? DIM:VALUE_DIM,Tiles=FEATURE_WIDTH/8;
        int lane=threadIdx.x%32;
        for(int tile=threadIdx.x/32;tile<2*Tiles;tile+=8) {
            int row=tile/Tiles*16,col=tile%Tiles*8;
            DoubleMma::FragmentC result;result.clear();
            #pragma unroll
            for(int base=0;base<32;base+=16) {
                auto a=double_load_a<40,Transpose>(coefficients+C0*32*40,row,base,lane);
                auto b=double_load_b<DOUBLE_CACHE_PITCH,true>(cache+F0*DOUBLE_CACHE_PLANE,col,base,lane);
                DoubleMma{}(result,a,b,result);
                if constexpr(C1>=0) {
                    a=double_load_a<40,Transpose>(coefficients+C1*32*40,row,base,lane);
                    b=double_load_b<DOUBLE_CACHE_PITCH,true>(cache+F1*DOUBLE_CACHE_PLANE,col,base,lane);
                    DoubleMma{}(result,a,b,result);
                }
            }
            #pragma unroll
            for(int i=0;i<4;++i) {
                int64_t r=(Transpose ? kb:qb)+row+lane/4+(i/2)*8;
                int c=channel_base+col+(lane%4)*2+i%2;
                if(r<end && c<Channels) {
                    float value=result[i]*(Target<2 ? p.scale:1.f);
                    if constexpr(Atomic && Transpose) atomicAdd(&p.outputs[Target].fp32_at(r,head,c),value);
                    else {
                        auto&& out=p.outputs[Target].feature(r,head,c);
                        out=value+((Transpose ? qb:kb)>begin ? float(out):0.f);
                    }
                }
            }
        }
    }
}

template<bool Atomic=false>
__device__ void double_gradients(AttentionParams const& p,float const* probability,float const* adjoint,
    LowMmaElement* cache,int64_t begin,int64_t end,int64_t qb,int64_t kb,int head) {
    auto* coefficients=reinterpret_cast<LowMmaElement*>(const_cast<float*>(adjoint)+2048+32);
    for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
        int pos=i/32*40+i%32;
        coefficients[pos]=LowMmaElement(probability[i]);
        coefficients[32*40+pos]=LowMmaElement(probability[1024+i]);
        coefficients[2*32*40+pos]=LowMmaElement(adjoint[i]);
        coefficients[3*32*40+pos]=LowMmaElement(adjoint[1024+i]);
    }
    __syncthreads();
    for(int base=0;base<max(DIM,VALUE_DIM);base+=FEATURE_WIDTH) {
        if constexpr(DIM>FEATURE_WIDTH || VALUE_DIM>FEATURE_WIDTH)
            double_cache_inputs(p,cache,qb,kb,end,head,base);
        double_tile_output<0,3,1,2,5,false,Atomic>(p,cache,coefficients,begin,end,qb,kb,head,base);
        double_tile_output<1,3,0,2,4,true,Atomic>(p,cache,coefficients,begin,end,qb,kb,head,base);
        double_tile_output<2,1,3,-1,-1,true,Atomic>(p,cache,coefficients,begin,end,qb,kb,head,base);
        double_tile_output<4,1,2,0,6,false,Atomic>(p,cache,coefficients,begin,end,qb,kb,head,base);
        __syncthreads();
    }
}
#else
constexpr int DOUBLE_LONG_BYTES=4*(7*33*FEATURE_WIDTH+4096+32);
#endif

#if !LOW_PRECISION
__device__ __forceinline__ void double_pairs(AttentionParams const& p, float* cache, float* probability, float* adjoint,
    int64_t begin, int64_t end, int64_t qb, int64_t kb, int segment, int head) {
    const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
    float score[4] = {}, direction[4] = {}, value[4] = {}, value_direction[4] = {};
    constexpr int extent = DIM>VALUE_DIM ? DIM:VALUE_DIM;
    for (int channel_base = 0; channel_base<extent; channel_base+=FEATURE_WIDTH) {
        cache_feature<0>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<1>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<2>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<4>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<5>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<6>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<7>(p, cache, qb, kb, end, head, channel_base);
        __syncthreads();
        if (qb+row<end) {
            #pragma unroll 4
            for (int c = 0; c<FEATURE_WIDTH; c++) {
                #pragma unroll
                for (int r = 0; r<4; r++) {
                    int q = c*33+row+r, k = c*33+lane;
                    if (channel_base+c<DIM) {
                        score[r] = multiply_add(cache[q], cache[33*FEATURE_WIDTH+k], score[r]);
                        if constexpr (PRESENT&(1<<5)) direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q],
                            cache[33*FEATURE_WIDTH+k], direction[r]);
                        if constexpr (PRESENT&(1<<6)) direction[r] = multiply_add(cache[q],
                            cache[5*33*FEATURE_WIDTH+k], direction[r]);
                    }
                    if (channel_base+c<VALUE_DIM) {
                        value[r] = multiply_add(cache[3*33*FEATURE_WIDTH+q], cache[2*33*FEATURE_WIDTH+k], value[r]);
                        if constexpr (PRESENT&(1<<7)) value_direction[r] = multiply_add(cache[3*33*FEATURE_WIDTH+q],
                            cache[6*33*FEATURE_WIDTH+k], value_direction[r]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r<4; r++) {
        float probability_value = 0, direction_value = 0;
        if (qb+row+r<end && kb+lane<end && (!CAUSAL || kb+lane<=qb+row+r)) {
            float s = score[r]*p.scale+higher_bias<3>(p, segment, head, qb-begin+row+r, kb-begin+lane, kb+lane);
            float inv = p.state.at(qb+row+r, head, 1);
            if (inv>0 && s>-INFINITY) probability_value = __expf(s-p.state.at(qb+row+r, head, 0))*inv;
            direction_value = direction[r]*p.scale+higher_bias<8>(p, segment, head, qb-begin+row+r, kb-begin+lane, kb+lane);
        }
        int pos = (row+r)*32+lane;
        probability[pos] = probability_value;
        probability[1024+pos] = direction_value;
        adjoint[pos] = value[r];
        adjoint[1024+pos] = value_direction[r];
    }
    __syncthreads();
}

template<bool Atomic = false>
__device__ __forceinline__ void double_gradients(
    AttentionParams const& p, float const* probability, float const* adjoint, float const* cache,
    int64_t begin, int64_t end, int64_t qb, int64_t kb, int head) {
    const int lane = threadIdx.x%32, row = (threadIdx.x%128)/32*8;
    constexpr int extent = DIM>VALUE_DIM ? DIM:VALUE_DIM;
    if (threadIdx.x<128) {
        if (qb+row>=end) return;
        const int count = int(min(int64_t(32), end-kb));
        #pragma unroll
        for (int chunk = 0; chunk<(extent+31)/32; chunk++) {
            int channel = lane+chunk*32;
            float query[8] = {}, grad_out[8] = {};
            #pragma unroll 2
            for (int key = 0; key<count; key++) {
                float k = 0, rk = 0, v = 0, rv = 0;
                if constexpr (NEEDS&1) if (channel<DIM) {
                    k = feature_value<1>(p, cache, key, channel, qb, kb, head);
                    rk = feature_value<6>(p, cache, key, channel, qb, kb, head);
                }
                if constexpr (NEEDS&16) if (channel<VALUE_DIM) {
                    v = feature_value<2>(p, cache, key, channel, qb, kb, head);
                    rv = feature_value<7>(p, cache, key, channel, qb, kb, head);
                }
                #pragma unroll
                for (int r = 0; r<8; r++) {
                    int pos = (row+r)*32+key;
                    if constexpr (NEEDS&1) if (channel<DIM) {
                        query[r] = multiply_add(adjoint[1024+pos], k, query[r]);
                        query[r] = multiply_add(adjoint[pos], rk, query[r]);
                    }
                    if constexpr (NEEDS&16) if (channel<VALUE_DIM) {
                        grad_out[r] = multiply_add(probability[1024+pos], v, grad_out[r]);
                        grad_out[r] = multiply_add(probability[pos], rv, grad_out[r]);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r<8; r++) if (qb+row+r<end) {
                if constexpr (NEEDS&1) if (channel<DIM) {
                    auto&& out = p.outputs[0].feature(qb+row+r, head, channel);
                    out = query[r]*p.scale+(kb>begin ? out:0.f);
                }
                if constexpr (NEEDS&16) if (channel<VALUE_DIM) {
                    auto&& out = p.outputs[4].feature(qb+row+r, head, channel);
                    out = grad_out[r]+(kb>begin ? out:0.f);
                }
            }
        }
    } else {
        if (kb+row>=end) return;
        const int count = int(min(int64_t(32), end-qb));
        #pragma unroll
        for (int chunk = 0; chunk<(extent+31)/32; chunk++) {
            int channel = lane+chunk*32;
            float key_gradient[8] = {}, value_gradient[8] = {};
            #pragma unroll 2
            for (int key = 0; key<count; key++) {
                float q = 0, rq = 0, g = 0;
                if constexpr (NEEDS&2) if (channel<DIM) {
                    q = feature_value<0>(p, cache, key, channel, qb, kb, head);
                    rq = feature_value<5>(p, cache, key, channel, qb, kb, head);
                }
                if constexpr (NEEDS&4) if (channel<VALUE_DIM)
                    g = feature_value<4>(p, cache, key, channel, qb, kb, head);
                #pragma unroll
                for (int r = 0; r<8; r++) {
                    int pos = key*32+row+r;
                    if constexpr (NEEDS&2) if (channel<DIM) {
                        key_gradient[r] = multiply_add(adjoint[1024+pos], q, key_gradient[r]);
                        key_gradient[r] = multiply_add(adjoint[pos], rq, key_gradient[r]);
                    }
                    if constexpr (NEEDS&4) if (channel<VALUE_DIM)
                        value_gradient[r] = multiply_add(probability[1024+pos], g, value_gradient[r]);
                }
            }
            #pragma unroll
            for (int r = 0; r<8; r++) if (kb+row+r<end) {
                if constexpr (NEEDS&2) if (channel<DIM) {
                    auto&& out = p.outputs[1].feature(kb+row+r, head, channel);
                    if constexpr(Atomic) atomicAdd(&p.outputs[1].fp32_at(kb+row+r,head,channel),key_gradient[r]*p.scale);
                    else out = key_gradient[r]*p.scale+(qb>begin ? out:0.f);
                }
                if constexpr (NEEDS&4) if (channel<VALUE_DIM) {
                    auto&& out = p.outputs[2].feature(kb+row+r, head, channel);
                    if constexpr(Atomic) atomicAdd(&p.outputs[2].fp32_at(kb+row+r,head,channel),value_gradient[r]);
                    else out = value_gradient[r]+(qb>begin ? out:0.f);
                }
            }
        }
    }
}

#endif

constexpr int SEGMENT_WIDTH=(DIM>VALUE_DIM ? DIM:VALUE_DIM)<=96 ? (DIM>VALUE_DIM ? DIM:VALUE_DIM):64;
template<int Tile> constexpr int SEGMENT_CACHE=(SEGMENT_WIDTH>128 ? SEGMENT_WIDTH:128)*(Tile+1);

template<int Tile>
__device__ void double_segment_scores(AttentionParams const& p,float* cache,float* target,
    int64_t begin,int64_t end,int head) {
    constexpr int Rows=Tile/16,Plane=32*(Tile+1);
    const int row=threadIdx.x/16*Rows,col=threadIdx.x%16*Rows;
    float score[Rows][Rows]={},derivative[Rows][Rows]={};
    for(int base=0;base<DIM;base+=32) {
        cache_feature<0,0,32,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<1,1,32,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<5,2,32,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<6,3,32,Tile>(p,cache,begin,begin,end,head,base);
        __syncthreads();
        if(begin+row<end) {
            #pragma unroll 2
            for(int c=0;c<min(32,DIM-base);++c) {
                #pragma unroll
                for(int r=0;r<Rows;++r) {
                    float q=cache[c*(Tile+1)+row+r],rq=cache[2*Plane+c*(Tile+1)+row+r];
                    #pragma unroll
                    for(int k=0;k<Rows;++k) {
                        float v=cache[Plane+c*(Tile+1)+col+k],rv=cache[3*Plane+c*(Tile+1)+col+k];
                        score[r][k]=multiply_add(q,v,score[r][k]);
                        if constexpr(PRESENT&(1<<5)) derivative[r][k]=multiply_add(rq,v,derivative[r][k]);
                        if constexpr(PRESENT&(1<<6)) derivative[r][k]=multiply_add(q,rv,derivative[r][k]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<Rows;++r) {
        #pragma unroll
        for(int k=0;k<Rows;++k) {
            int pos=(row+r)*Tile+col+k;
            target[pos]=score[r][k];target[Tile*Tile+pos]=derivative[r][k];
        }
    }
    __syncthreads();
}

template<int Tile,int Source,int Target,int Plane,bool Transpose>
__device__ __forceinline__ void double_segment_add(AttentionParams const& p,float* cache,float const* weights,
    int64_t begin,int64_t end,int head,int base,float (&result)[Tile/8][(SEGMENT_WIDTH+31)/32]) {
    constexpr int Width=Target<2 ? DIM:VALUE_DIM;
    const int row=threadIdx.x/32*(Tile/8),lane=threadIdx.x%32;
    cache_feature<Source,0,SEGMENT_WIDTH,Tile>(p,cache,begin,begin,end,head,base);
    __syncthreads();
    if(begin+row<end) {
        #pragma unroll 2
        for(int k=0;k<end-begin;k++) {
            #pragma unroll
            for(int r=0;r<Tile/8;r++) {
                int pos=Transpose ? k*Tile+row+r:(row+r)*Tile+k;
                float a=weights[Plane*Tile*Tile+pos];
                #pragma unroll
                for(int c=0;c<(SEGMENT_WIDTH+31)/32;c++) {
                    int channel=lane+c*32;
                    if(base+channel<Width && channel<SEGMENT_WIDTH)
                        result[r][c]=multiply_add(a,cache[channel*(Tile+1)+k],result[r][c]);
                }
            }
        }
    }
    __syncthreads();
}

template<int Tile,int Target,int A,int B,int S0,int S1,bool Transpose>
__device__ void double_segment_output(AttentionParams const& p,float* cache,float const* weights,
    int64_t begin,int64_t end,int head) {
    if constexpr(NEEDS&(1<<Target)) {
        constexpr int Width=Target<2 ? DIM:VALUE_DIM;
        const int row=threadIdx.x/32*(Tile/8),lane=threadIdx.x%32;
        for(int base=0;base<Width;base+=SEGMENT_WIDTH) {
            float result[Tile/8][(SEGMENT_WIDTH+31)/32]={};
            double_segment_add<Tile,S0,Target,A,Transpose>(p,cache,weights,begin,end,head,base,result);
            if constexpr(S1>=0) double_segment_add<Tile,S1,Target,B,Transpose>(p,cache,weights,begin,end,head,base,result);
            #pragma unroll
            for(int r=0;r<Tile/8;r++) if(begin+row+r<end) {
                #pragma unroll
                for(int c=0;c<(SEGMENT_WIDTH+31)/32;c++) {
                    int channel=base+lane+c*32;
                    if(channel<Width && lane+c*32<SEGMENT_WIDTH)
                        p.outputs[Target].feature(begin+row+r,head,channel)=result[r][c]*(Target<2 ? p.scale:1.f);
                }
            }
        }
    }
}

template<int Tile>
__device__ void double_segment_adjoint(AttentionParams const& p,float* cache,float* weights,
    int64_t begin,int64_t end,int head) {
    const int row=threadIdx.x/16*(Tile/16),col=threadIdx.x%16*(Tile/16);
    float t[Tile/16][Tile/16]={},tr[Tile/16][Tile/16]={};
    for(int base=0;base<VALUE_DIM;base+=32) {
        cache_feature<4,0,32,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<2,1,32,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<7,2,32,Tile>(p,cache,begin,begin,end,head,base);
        __syncthreads();
        if(begin+row<end) {
            #pragma unroll 2
            for(int c=0;c<min(32,VALUE_DIM-base);c++) {
                #pragma unroll
                for(int r=0;r<Tile/16;r++) {
                    float g=cache[c*(Tile+1)+row+r];
                    #pragma unroll
                    for(int k=0;k<Tile/16;k++) {
                        t[r][k]=multiply_add(g,cache[32*(Tile+1)+c*(Tile+1)+col+k],t[r][k]);
                        if constexpr(PRESENT&(1<<7)) tr[r][k]=multiply_add(g,cache[2*32*(Tile+1)+c*(Tile+1)+col+k],tr[r][k]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<Tile/16;r++) {
        float p0[Tile/16],pr[Tile/16],delta=0,delta_r=0;
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=(row+r)*Tile+col+k;
            p0[k]=weights[pos];pr[k]=weights[Tile*Tile+pos];
            tr[r][k]=pr[k]*t[r][k]+p0[k]*tr[r][k];t[r][k]*=p0[k];
            delta+=t[r][k];delta_r+=tr[r][k];
        }
        #pragma unroll
        for(int shift=1;shift<16;shift*=2) {
            delta+=__shfl_xor_sync(0xffffffff,delta,shift);delta_r+=__shfl_xor_sync(0xffffffff,delta_r,shift);
        }
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=(row+r)*Tile+col+k;
            weights[pos]=t[r][k]-p0[k]*delta;
            weights[Tile*Tile+pos]=tr[r][k]-pr[k]*delta-p0[k]*delta_r;
        }
    }
    __syncthreads();
}

template<int Tile>
__device__ __forceinline__ void double_segment(AttentionParams const& p,int segment,int head) {
    const int64_t begin=p.offset(segment),end=p.offset(segment+1);
    if(end-begin<=Tile-16 || end-begin>Tile) return;
    extern __shared__ float shared[];
    float* probability=shared+SEGMENT_CACHE<Tile>;
    float* adjoint=probability;
    double_segment_scores<Tile>(p,shared,probability,begin,end,head);
    const int row=threadIdx.x/16*(Tile/16),col=threadIdx.x%16*(Tile/16);
    #pragma unroll
    for(int r=0;r<Tile/16;r++) {
        float prob[Tile/16],a[Tile/16];
        float total=0,mean=0;
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=(row+r)*Tile+col+k;
            const bool valid=begin+row+r<end && begin+col+k<end && (!CAUSAL || col+k<=row+r);
            float score=valid ? probability[pos]*p.scale+higher_bias<3>(p,segment,head,row+r,col+k,begin+col+k):0.f;
            float inverse=valid ? p.state.at(begin+row+r,head,1):0.f;
            prob[k]=valid && inverse>0 && score>-INFINITY ? __expf(score-p.state.at(begin+row+r,head,0))*inverse:0.f;
            a[k]=probability[Tile*Tile+pos]*p.scale+(valid ? higher_bias<8>(p,segment,head,row+r,col+k,begin+col+k):0.f);
            total+=prob[k];mean+=prob[k]*a[k];
        }
        #pragma unroll
        for(int shift=1;shift<16;shift*=2) {
            total+=__shfl_xor_sync(0xffffffff,total,shift);
            mean+=__shfl_xor_sync(0xffffffff,mean,shift);
        }
        float inverse=total>0 ? 1.f/total:0.f;
        mean*=inverse;
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=(row+r)*Tile+col+k;
            float pr=(prob[k]*=inverse)*(a[k]-mean);
            probability[pos]=prob[k];probability[Tile*Tile+pos]=pr;
        }
    }
    __syncthreads();
    double_segment_output<Tile,2,1,0,4,-1,true>(p,shared,probability,begin,end,head);
    double_segment_output<Tile,4,1,0,2,7,false>(p,shared,probability,begin,end,head);
    double_segment_adjoint<Tile>(p,shared,probability,begin,end,head);
    double_segment_output<Tile,0,1,0,1,6,false>(p,shared,adjoint,begin,end,head);
    double_segment_output<Tile,1,1,0,0,5,true>(p,shared,adjoint,begin,end,head);
    if constexpr((NEEDS&8) && BIAS_KIND==1) {
        for(int k=threadIdx.x;k<end-begin;k+=blockDim.x) {
            float sum=0;
            for(int q=0;q<end-begin;q++) sum+=adjoint[Tile*Tile+q*Tile+k];
            p.outputs[3].at(begin+k,head)=sum;
        }
    } else if constexpr((NEEDS&8) && BIAS_KIND==2) {
        for(int64_t i=threadIdx.x;i<p.outputs[3].size[2]*p.outputs[3].size[3];i+=blockDim.x) {
            int64_t q=i/p.outputs[3].size[3],k=i%p.outputs[3].size[3];
            p.outputs[3].at(segment,head,q,k)=q<end-begin && k<end-begin ? adjoint[Tile*Tile+q*Tile+k]:0.f;
        }
    }
}
template<int Tile> constexpr int RESIDENT_BYTES =
#if LOW_PRECISION
    DoubleResident<Tile>::Bytes<=RESIDENT_BUDGET ? DoubleResident<Tile>::Bytes:
#endif
    4*(SEGMENT_CACHE<Tile>+2*Tile*Tile);
template<int Tile> constexpr int RESIDENT_THREADS =
#if LOW_PRECISION
    DoubleResident<Tile>::Bytes<=RESIDENT_BUDGET ? 128:
#endif
    256;
template<int Tile>
__global__ __launch_bounds__(RESIDENT_THREADS<Tile>,LOW_PRECISION ? 1:(Tile==48 ? 4:3)) void attention_double_backward_segment(const __grid_constant__ AttentionParams p) {
    #if LOW_PRECISION
    if constexpr(DoubleResident<Tile>::Bytes<=RESIDENT_BUDGET) {
        int segment=blockIdx.x,head=blockIdx.y;
        int64_t begin=p.offset(segment),end=p.offset(segment+1);
        if(end-begin>(Tile==48?32:48) && end-begin<=Tile)
            DoubleResident<Tile>::double_lowp(p,segment,head,begin,end-begin);
    } else
#endif
    double_segment<Tile>(p,blockIdx.x,blockIdx.y);
}
#define MAIN_SHARED_BYTES (4*(COMBINE_SEGMENTS ? ((7*33*FEATURE_WIDTH+4096+32)>(SEGMENT_CACHE<64>+2*64*64) ? (7*33*FEATURE_WIDTH+4096+32):(SEGMENT_CACHE<64>+2*64*64)):(7*33*FEATURE_WIDTH+4096+32)))


template<bool Split = false, bool InitializeOnly = false>
#if LOW_PRECISION
__global__ __launch_bounds__(!Split && NATIVE_SHORT_SUPPORTED ? 128:256, 2)
#else
__global__ __launch_bounds__(256,Split ? 2:1)
#endif
void attention_double_backward_kernel(const __grid_constant__ AttentionParams p) {
    const int segment = Split ? blockIdx.x/4:blockIdx.x, head = blockIdx.y;
    const int64_t begin = p.offset(segment), end = p.offset(segment+1);
    if constexpr(InitializeOnly) if(end-begin<=64) return;
    if(end-begin>32 && end-begin<=64) {
        if constexpr(COMBINE_SEGMENTS && !Split) {
            if(end-begin<=48) double_segment<48>(p,segment,head);
            else double_segment<64>(p,segment,head);
        }
        return;
    }
    if constexpr(Split) {
        if(end-begin<=64) return;
    } else if(end-begin>64) {
        if constexpr(LOW_PRECISION && !InitializeOnly) return;
        for(int64_t i=threadIdx.x;i<(end-begin)*max(DIM,VALUE_DIM);i+=blockDim.x) {
            int64_t r=begin+i/max(DIM,VALUE_DIM);
            int c=i%max(DIM,VALUE_DIM);
            if constexpr(NEEDS&(1<<1)) if(c<DIM) p.outputs[1].feature(r,head,c)=0;
            if constexpr(NEEDS&(1<<2)) if(c<VALUE_DIM) p.outputs[2].feature(r,head,c)=0;
        }
        if constexpr((NEEDS&(1<<3)) && BIAS_KIND==1) {
            for(int64_t r=begin+threadIdx.x;r<end;r+=blockDim.x) p.outputs[3].at(r,head)=0;
        } else if constexpr((NEEDS&(1<<3)) && BIAS_KIND==2) {
            for(int64_t i=threadIdx.x;i<p.outputs[3].size[2]*p.outputs[3].size[3];i+=blockDim.x)
                p.outputs[3].at(segment,head,i/p.outputs[3].size[3],i%p.outputs[3].size[3])=0;
        }
        return;
    }

#if LOW_PRECISION
    if constexpr(!Split && NATIVE_SHORT_SUPPORTED) {
        double_lowp(p,segment,head,begin,int(end-begin));
        return;
    }
#endif
    const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
    extern __shared__ float shared[];
#if LOW_PRECISION
    auto* cache=reinterpret_cast<LowMmaElement*>(shared);
    float* probability=reinterpret_cast<float*>(cache+7*DOUBLE_CACHE_PLANE);
#else
    float* cache = shared;
    float* probability = cache+7*33*FEATURE_WIDTH;
#endif
    float* adjoint = probability+2048;
    float* row_sum = adjoint+2048;
    if constexpr (!Split && BIAS_KIND==2 && (NEEDS&(1<<3)))
        for (int64_t i = threadIdx.x; i<p.outputs[3].size[2]*p.outputs[3].size[3]; i+=256)
            p.outputs[3].at(segment, head, i/p.outputs[3].size[3], i%p.outputs[3].size[3]) = 0;

    if (begin==end) return;
    if (end-begin<=32) {
        double_pairs(p, cache, probability, adjoint, begin, end, begin, begin, segment, head);
        #pragma unroll
        for (int r = 0; r<4; r++) {
            int pos = (row+r)*32+lane;
            float prob = probability[pos], a = probability[1024+pos];
            const float total = higher_sum(prob);
            prob *= total>0 ? 1.f/total : 0.f;
            probability[pos] = prob;
            float pr = prob*(a-higher_sum(prob*a));
            float t = adjoint[pos], tr = adjoint[1024+pos];
            float delta = higher_sum(prob*t), delta_r = higher_sum(pr*t+prob*tr);
            probability[1024+pos] = pr;
            adjoint[pos] = prob*(t-delta);
            adjoint[1024+pos] = pr*(t-delta)+prob*(tr-delta_r);
        }
        __syncthreads();

        double_gradients<Split>(p, probability, adjoint, cache, begin, end, begin, begin, head);
        higher_bias_gradient<3, 1, Split>(p, adjoint, segment, head, begin, end, begin, begin);
        return;
    }
    for (int64_t qb = begin+(Split ? (blockIdx.x%4)*32:0); qb<end; qb+=Split ? 128:32) {
        float mean[4] = {}, delta[4] = {}, delta_r[4] = {};
        if (threadIdx.x<32) row_sum[threadIdx.x] = 0.f;
        // Collect raw moments together; centering after the scan avoids a rescan.
        for (int64_t kb = begin; kb<end; kb+=32) {
            double_pairs(p, cache, probability, adjoint, begin, end, qb, kb, segment, head);
            #pragma unroll
            for (int r = 0; r<4; r++) {
                int pos = (row+r)*32+lane;
                float prob = probability[pos], a = probability[1024+pos];
                float t = adjoint[pos], tr = adjoint[1024+pos];
                const float total = higher_sum(prob);
                if (lane==0) row_sum[row+r]+=total;
                mean[r]+=higher_sum(prob*a);
                delta[r]+=higher_sum(prob*t);
                delta_r[r]+=higher_sum(prob*a*t+prob*tr);
            }
            __syncthreads();
        }
        if (threadIdx.x<32) {
            const float total = row_sum[threadIdx.x];
            row_sum[threadIdx.x] = total>0 ? 1.f/total : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r<4; r++) {
            const float inv_sum = row_sum[row+r];
            mean[r] *= inv_sum;
            delta[r] *= inv_sum;
            delta_r[r] = delta_r[r]*inv_sum-mean[r]*delta[r];
        }
        for (int64_t kb = begin; kb<end; kb+=32) {
            if (end-begin>32) {
                double_pairs(p, cache, probability, adjoint, begin, end, qb, kb, segment, head);
                #pragma unroll
                for (int r = 0; r<4; r++) {
                    int pos = (row+r)*32+lane;
                    probability[pos] *= row_sum[row+r];
                    probability[1024+pos] = probability[pos]*(probability[1024+pos]-mean[r]);
                }
            }
            #pragma unroll
            for (int r = 0; r<4; r++) {
                int pos = (row+r)*32+lane;
                float t = adjoint[pos]-delta[r], tr = adjoint[1024+pos]-delta_r[r];
                adjoint[pos] = probability[pos]*t;
                adjoint[1024+pos] = probability[1024+pos]*t+probability[pos]*tr;
            }
            __syncthreads();

            double_gradients<Split>(p, probability, adjoint, cache, begin, end, qb, kb, head);
        higher_bias_gradient<3, 1, Split>(p, adjoint, segment, head, begin, end, qb, kb);
            __syncthreads();
        }
    }
}

#if LOW_PRECISION
__global__ void higher_convert_long(const __grid_constant__ AttentionParams p) {
    int segment=blockIdx.x,head=blockIdx.y;
    int64_t begin=p.offset(segment),end=p.offset(segment+1);
    if(end-begin<=64) return;
    #pragma unroll
    for(int output=0;output<5;++output) {
        if(!p.outputs[output].data) continue;
        if(output==3 || output==8) {
            if constexpr(BIAS_KIND==2) {
                int64_t columns=p.outputs[output].size[3];
                for(int64_t x=threadIdx.x;x<p.outputs[output].size[2]*columns;x+=blockDim.x)
                    p.outputs[output].at(segment,head,x/columns,x%columns)=float(p.inputs[output].at(segment,head,x/columns,x%columns));
            } else if constexpr(BIAS_KIND==1) {
                for(int64_t row=begin+threadIdx.x;row<end;row+=blockDim.x)
                    p.outputs[output].at(row,head)=float(p.inputs[output].at(row,head));
            }
        } else {
            int channels=p.outputs[output].size[2];
            for(int64_t x=threadIdx.x;x<(end-begin)*channels;x+=blockDim.x) {
                int64_t row=begin+x/channels;int channel=x%channels;
                p.outputs[output].feature(row,head,channel)=float(p.inputs[output].feature(row,head,channel));
            }
        }
    }
}
#endif

void attention_double_backward(std::vector<c10::optional<at::Tensor>> inputs, at::Tensor rowptr, at::Tensor state,
    std::vector<c10::optional<at::Tensor>> outputs, double scale, std::vector<c10::optional<at::Tensor>> partials) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p = make_params(inputs, outputs, rowptr, state, scale);
#if LOW_PRECISION
    auto partial = make_params(inputs, partials, rowptr, state, scale);
    attention_double_backward_kernel<false,true><<<dim3(p.groups,inputs[0]->size(1)),128,0,at::cuda::getCurrentCUDAStream()>>>(partial);
#else
    auto const& partial=p;
#endif
#if LOW_PRECISION
    constexpr int threads=(NATIVE_SHORT_SUPPORTED) ? 128:256;
    constexpr int bytes=(NATIVE_SHORT_SUPPORTED) ? 2*7*DOUBLE_PLANE+4*DOUBLE_WORK_FLOATS:DOUBLE_LONG_BYTES;
#else
    constexpr int threads=256;
    constexpr int bytes=MAIN_SHARED_BYTES;
#endif
    attention_double_backward_kernel<false><<<dim3(p.groups, inputs[0]->size(1)), threads, bytes,
        at::cuda::getCurrentCUDAStream()>>>(p);
    attention_double_backward_kernel<true><<<dim3(p.groups*4,inputs[0]->size(1)),256,DOUBLE_LONG_BYTES,
        at::cuda::getCurrentCUDAStream()>>>(partial);
#if !COMBINE_SEGMENTS
    attention_double_backward_segment<48><<<dim3(p.groups,inputs[0]->size(1)),RESIDENT_THREADS<48>,
        RESIDENT_BYTES<48>,at::cuda::getCurrentCUDAStream()>>>(p);
    attention_double_backward_segment<64><<<dim3(p.groups,inputs[0]->size(1)),RESIDENT_THREADS<64>,
        RESIDENT_BYTES<64>,at::cuda::getCurrentCUDAStream()>>>(p);
#endif
#if LOW_PRECISION
    auto conversion=make_params(partials,outputs,rowptr,state,scale);
    higher_convert_long<<<dim3(p.groups,inputs[0]->size(1)),256,0,at::cuda::getCurrentCUDAStream()>>>(conversion);
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &attention_double_backward);
    m.def("configure", []() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_double_backward_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
#if LOW_PRECISION
            (NATIVE_SHORT_SUPPORTED) ? 2*7*DOUBLE_PLANE+4*DOUBLE_WORK_FLOATS:DOUBLE_LONG_BYTES
#else
            MAIN_SHARED_BYTES
#endif
            ));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_double_backward_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,DOUBLE_LONG_BYTES));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_double_backward_segment<48>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,RESIDENT_BYTES<48>));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_double_backward_segment<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,RESIDENT_BYTES<64>));
    });
}
'''


def attention_double_backward(inputs, rowptr, state, config, needs):
    results = tuple(
        torch.empty(value.shape, device=value.device, dtype=value.dtype)
        if need and value is not None else None
        for value, need in zip(inputs[:5], needs)
    )
    if any(value is not None and value.numel() for value in results):
        present = sum(1 << i for i, value in enumerate(inputs) if value is not None)
        wanted = sum(1 << i for i, value in enumerate(results) if value is not None)
        # A small grid cannot occupy more than two CTAs per SM; combine resident paths to avoid empty launches.
        device = inputs[0].device
        combine_segments = (inputs[0].dtype == torch.float32
                            and get_cuda_arch(device) == "90"
                            and (rowptr.numel()-1)*config[2] <= 2*get_device_properties(device).multi_processor_count)
        key = (combine_segments, config[0], config[1], config[2], config[4], config[6], present, wanted, inputs[0].device.index, inputs[0].dtype)
        kernel = KERNELS.get(key)
        if kernel is None:
            arch = get_cuda_arch(device)
            properties = get_device_properties(device)
            # Keep two CTAs within Hopper's 228 KiB shared-memory budget.
            feature_limit = 64
            if arch == "90":
                feature_limit = (((228 * 1024 // 2 - 1024) // 4 - 4128) // (7 * 33) // 32) * 32
            defines = (f"#define COMBINE_SEGMENTS {int(combine_segments)}\n#define RESIDENT_BUDGET {properties.shared_memory_per_block_optin-1024}\n"
                       f"#define FEATURE_LIMIT {feature_limit}\n"
                       f"#define DIM {config[0]}\n#define VALUE_DIM {config[1]}\n"
                       f"#define HEADS {config[2]}\n#define CAUSAL {int(config[4])}\n#define BIAS_KIND {config[6]}\n"
                       f"#define PRESENT {present}\n#define NEEDS {wanted}\n")
            module = load_cuda_extension(
                "attention_double_backward", dtype_defines(inputs[0].dtype) + CUDA_SOURCE + defines + FEATURE_SOURCE + LOWP_SOURCE + NATIVE_RESIDENT + SOURCE,
                arch=arch,
                use_fast_math=False, minimal_headers=True,
            )
            with torch.cuda.device(inputs[0].device):
                module.configure()
            kernel = module.run
            KERNELS[key] = kernel
        partials = tuple(torch.empty_like(value, dtype=torch.float32) if value is not None else None
                         for value in results) if inputs[0].dtype != torch.float32 else results
        kernel(inputs, rowptr, state, results, config[3], partials)
    return results
