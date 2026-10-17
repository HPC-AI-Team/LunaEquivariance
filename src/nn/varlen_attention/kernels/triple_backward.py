"""Fused CSR attention triple backward with FP32 accumulation."""

import torch
from .utils import CUDA_SOURCE, dtype_defines
from .double_backward import FEATURE_SOURCE
from ....utils.extension import get_cuda_arch, get_device_properties, load_cuda_extension

KERNELS = {}

SOFTMAX_SOURCE = r"""
// Both product backends use the same FP32 softmax moments and mixed adjoint.
template<bool StoreProbability>
__device__ __forceinline__ void softmax_triple_backward(
    float prob, float a, float b, float c, float* probability, float* adjoint, int pos) {
    a-=higher_sum(prob*a);
    b-=higher_sum(prob*b);
    float pr = prob*a, ps = prob*b, prs = prob*(a*b+c-higher_sum(prob*(c+a*b)));
    float t = adjoint[pos], tr = adjoint[1024+pos], ts = adjoint[2048+pos], trs = adjoint[3072+pos];
    float delta = higher_sum(prob*t), delta_r = higher_sum(pr*t+prob*tr);
    float delta_s = higher_sum(ps*t+prob*ts), delta_rs = higher_sum(prs*t+pr*ts+ps*tr+prob*trs);
    t-=delta;
    tr-=delta_r;
    ts-=delta_s;
    trs-=delta_rs;
    if constexpr (StoreProbability) probability[pos] = prob;
    probability[1024+pos] = pr;
    probability[2048+pos] = ps;
    probability[3072+pos] = prs;
    adjoint[pos] = prob*t;
    adjoint[1024+pos] = pr*t+prob*tr;
    adjoint[2048+pos] = ps*t+prob*ts;
    adjoint[3072+pos] = prs*t+pr*ts+ps*tr+prob*trs;
}
"""

NATIVE_RESIDENT = r'''
#if LOW_PRECISION
using NativeTripleMma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16,8,16>,32,
    LowMmaElement,cutlass::layout::RowMajor,LowMmaElement,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;

template<int Pitch,bool Transpose>
__device__ __forceinline__ NativeTripleMma::FragmentA native_triple_a(
    LowMmaElement const* pointer,int row,int base,int lane) {
    int offset=Transpose ? (base+lane%8+(lane/16)*8)*Pitch+row+(lane/8%2)*8
                         : (row+lane%16)*Pitch+base+(lane/16)*8;
    unsigned address=unsigned(__cvta_generic_to_shared(pointer+offset));
    NativeTripleMma::FragmentA result;
    auto* words=reinterpret_cast<unsigned*>(&result);
    if constexpr(Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]):"r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]):"r"(address));
    return result;
}

template<int Pitch,bool Transpose>
__device__ __forceinline__ NativeTripleMma::FragmentB native_triple_b(
    LowMmaElement const* pointer,int column,int base,int lane) {
    int offset=Transpose ? (base+lane%8+(lane/8%2)*8)*Pitch+column
                         : (column+lane%8)*Pitch+base+(lane/8%2)*8;
    unsigned address=unsigned(__cvta_generic_to_shared(pointer+offset));
    NativeTripleMma::FragmentB result;
    auto* words=reinterpret_cast<unsigned*>(&result);
    if constexpr(Transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]):"r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(words[0]),"=r"(words[1]):"r"(address));
    return result;
}

__device__ __forceinline__ float native_triple_sum(float value) {
    value+=__shfl_xor_sync(0xffffffff,value,1);
    value+=__shfl_xor_sync(0xffffffff,value,2);
    return value;
}

template<int Tile> struct TripleNative {
    static constexpr int Width=(DIM>VALUE_DIM?DIM:VALUE_DIM),Pitch=(Width+15)/16*16+8;
    static constexpr int Plane=Tile*Pitch,CPitch=Tile+8,CPlane=Tile*CPitch;
    static constexpr int Bytes=(5*Plane+4*CPlane)*sizeof(LowMmaElement)+2*(Tile/16)*Tile*sizeof(float);

    template<int Input,int Slot>
    static __device__ void load(AttentionParams const& p,LowMmaElement* features,int64_t begin,int n,int head) {
        constexpr int Channels=Input==0 || Input==1 || Input==5 || Input==6 || Input==9 || Input==10?DIM:VALUE_DIM;
        auto const& tensor=p.inputs[Input];
        if constexpr(PRESENT&(1<<Input)) {
            if(Channels%8==0 && tensor.stride[2]==1 && tensor.stride[0]%8==0 && tensor.stride[1]%8==0
                && (reinterpret_cast<uintptr_t>(tensor.data)&15)==0) {
                for(int x=threadIdx.x*8;x<Plane;x+=128*8) {
                    int row=x/Pitch,c=x%Pitch;uint4 packed={0,0,0,0};
                    if(row<n && c<Channels) packed=*reinterpret_cast<uint4 const*>(tensor.data+(begin+row)*tensor.stride[0]+head*tensor.stride[1]+c);
                    *reinterpret_cast<uint4*>(features+Slot*Plane+x)=packed;
                }
                return;
            }
        }
        for(int x=threadIdx.x;x<Plane;x+=128) {
            int row=x/Pitch,c=x%Pitch;float value=0.f;
            if constexpr(PRESENT&(1<<Input)) if(row<n && c<Channels) value=tensor.load(begin+row,head,c);
            features[Slot*Plane+x]=LowMmaElement(value);
        }
    }

    template<int Target,int C0,int F0,int C1,int F1,int C2=-1,int F2=-1,bool Transpose=false>
    static __device__ void output(AttentionParams const& p,LowMmaElement const* features,
        LowMmaElement const* coefficients,int64_t begin,int n,int head) {
        if constexpr(NEEDS&(1<<Target)) {
            constexpr int Channels=Target==0 || Target==1 || Target==5 || Target==6?DIM:VALUE_DIM;
            constexpr int Columns=(Channels+7)/8;
            int lane=threadIdx.x%32;
            for(int tile=threadIdx.x/32;tile<(Tile/16)*Columns;tile+=4) {
                int row=(tile/Columns)*16,column=(tile%Columns)*8;
                NativeTripleMma::FragmentC result;result.clear();
                #pragma unroll
                for(int base=0;base<Tile;base+=16) {
                    auto a=native_triple_a<CPitch,Transpose>(coefficients+C0*CPlane,row,base,lane);
                    auto b=native_triple_b<Pitch,true>(features+F0*Plane,column,base,lane);
                    NativeTripleMma{}(result,a,b,result);
                    a=native_triple_a<CPitch,Transpose>(coefficients+C1*CPlane,row,base,lane);
                    b=native_triple_b<Pitch,true>(features+F1*Plane,column,base,lane);
                    NativeTripleMma{}(result,a,b,result);
                    if constexpr(C2>=0) {
                        a=native_triple_a<CPitch,Transpose>(coefficients+C2*CPlane,row,base,lane);
                        b=native_triple_b<Pitch,true>(features+F2*Plane,column,base,lane);
                        NativeTripleMma{}(result,a,b,result);
                    }
                }
                #pragma unroll
                for(int i=0;i<4;++i) {
                    int r=row+lane/4+(i/2)*8,c=column+(lane%4)*2+i%2;
                    if(r<n && c<Channels) p.outputs[Target].feature(begin+r,head,c)=result[i]*(Target==0 || Target==1 || Target==5 || Target==6?p.scale:1.f);
                }
            }
            __syncthreads();
        }
    }

    static __device__ void run(AttentionParams const& p,int segment,int head,int64_t begin,int n) {
        extern __shared__ float shared[];
        auto* features=reinterpret_cast<LowMmaElement*>(shared);
        auto* coefficients=features+5*Plane;
        auto* bias_partial=reinterpret_cast<float*>(coefficients+4*CPlane);
        int lane=threadIdx.x%32,warp=threadIdx.x/32,row=warp*16;
        NativeTripleMma::FragmentC jets[4][Tile/8];
        #pragma unroll
        for(int jet=0;jet<4;++jet)
            #pragma unroll
            for(int j=0;j<Tile/8;++j) jets[jet][j].clear();
        load<0,0>(p,features,begin,n,head);load<1,1>(p,features,begin,n,head);
        load<5,2>(p,features,begin,n,head);load<6,3>(p,features,begin,n,head);load<9,4>(p,features,begin,n,head);
        if constexpr(BIAS_KIND==2) {
            if constexpr(NEEDS&8) for(int64_t x=threadIdx.x;x<p.outputs[3].size[2]*p.outputs[3].size[3];x+=128)
                p.outputs[3].at(segment,head,x/p.outputs[3].size[3],x%p.outputs[3].size[3])=0.f;
            if constexpr(NEEDS&256) for(int64_t x=threadIdx.x;x<p.outputs[8].size[2]*p.outputs[8].size[3];x+=128)
                p.outputs[8].at(segment,head,x/p.outputs[8].size[3],x%p.outputs[8].size[3])=0.f;
        }
        __syncthreads();
        if(row<Tile) {
            for(int base=0;base<DIM;base+=16) {
                auto q=native_triple_a<Pitch,false>(features,row,base,lane);
                auto rq=native_triple_a<Pitch,false>(features+2*Plane,row,base,lane);
                auto sq=native_triple_a<Pitch,false>(features+4*Plane,row,base,lane);
                #pragma unroll
                for(int j=0;j<Tile/8;++j) {
                    auto k=native_triple_b<Pitch,false>(features+Plane,j*8,base,lane);
                    auto rk=native_triple_b<Pitch,false>(features+3*Plane,j*8,base,lane);
                    NativeTripleMma{}(jets[0][j],q,k,jets[0][j]);
                    NativeTripleMma{}(jets[1][j],rq,k,jets[1][j]);NativeTripleMma{}(jets[1][j],q,rk,jets[1][j]);
                    NativeTripleMma{}(jets[2][j],sq,k,jets[2][j]);NativeTripleMma{}(jets[3][j],sq,rk,jets[3][j]);
                }
            }
        }
        __syncthreads();
        // K is dead after these products; reuse its plane for the second K direction.
        load<10,1>(p,features,begin,n,head);
        __syncthreads();
        if(row<Tile) {
            for(int base=0;base<DIM;base+=16) {
                auto q=native_triple_a<Pitch,false>(features,row,base,lane);
                auto rq=native_triple_a<Pitch,false>(features+2*Plane,row,base,lane);
                #pragma unroll
                for(int j=0;j<Tile/8;++j) {
                    auto sk=native_triple_b<Pitch,false>(features+Plane,j*8,base,lane);
                    NativeTripleMma{}(jets[2][j],q,sk,jets[2][j]);NativeTripleMma{}(jets[3][j],rq,sk,jets[3][j]);
                }
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                int qr=row+lane/4+r*8;
                float maximum=qr<n?p.state.at(begin+qr,head,0):0.f,inverse=qr<n?p.state.at(begin+qr,head,1):0.f;
                float norm=0.f,ma=0.f,mb=0.f,mc=0.f;
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,i=r*2+c;
                        float prob=0.f,a=0.f,b=0.f,mixed=0.f;
                        if(qr<n && key<n && (!CAUSAL || key<=qr) && inverse>0.f) {
                            float score=jets[0][j][i]*p.scale+higher_bias<3>(p,segment,head,qr,key,begin+key);
                            prob=__expf(score-maximum)*inverse;
                            a=jets[1][j][i]*p.scale+higher_bias<8>(p,segment,head,qr,key,begin+key);
                            b=jets[2][j][i]*p.scale+higher_bias<12>(p,segment,head,qr,key,begin+key);
                            mixed=jets[3][j][i]*p.scale;
                        }
                        jets[0][j][i]=prob;jets[1][j][i]=a;jets[2][j][i]=b;jets[3][j][i]=mixed;
                        norm+=prob;ma+=prob*a;mb+=prob*b;mc+=prob*(mixed+a*b);
                    }
                norm=native_triple_sum(norm);float inv=norm>0.f?1.f/norm:0.f;
                ma=native_triple_sum(ma)*inv;mb=native_triple_sum(mb)*inv;mc=native_triple_sum(mc)*inv-ma*mb;
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,i=r*2+c,pos=qr*CPitch+key;
                        float prob=jets[0][j][i]*inv,a=jets[1][j][i]-ma,b=jets[2][j][i]-mb;
                        coefficients[pos]=LowMmaElement(prob);coefficients[CPlane+pos]=LowMmaElement(prob*a);
                        coefficients[2*CPlane+pos]=LowMmaElement(prob*b);
                        coefficients[3*CPlane+pos]=LowMmaElement(prob*(a*b+jets[3][j][i]-mc));
                    }
            }
        }
        __syncthreads();
        load<4,0>(p,features,begin,n,head);load<2,1>(p,features,begin,n,head);
        load<7,2>(p,features,begin,n,head);load<11,3>(p,features,begin,n,head);load<13,4>(p,features,begin,n,head);
        __syncthreads();
        output<2,3,0,1,4,-1,-1,true>(p,features,coefficients,begin,n,head);
        output<7,2,0,0,4,-1,-1,true>(p,features,coefficients,begin,n,head);
        output<4,3,1,1,3,2,2>(p,features,coefficients,begin,n,head);
        #pragma unroll
        for(int jet=0;jet<4;++jet)
            #pragma unroll
            for(int j=0;j<Tile/8;++j) jets[jet][j].clear();
        if(row<Tile) {
            for(int base=0;base<VALUE_DIM;base+=16) {
                auto g=native_triple_a<Pitch,false>(features,row,base,lane);
                auto sg=native_triple_a<Pitch,false>(features+4*Plane,row,base,lane);
                #pragma unroll
                for(int j=0;j<Tile/8;++j) {
                    auto v=native_triple_b<Pitch,false>(features+Plane,j*8,base,lane);
                    auto rv=native_triple_b<Pitch,false>(features+2*Plane,j*8,base,lane);
                    auto sv=native_triple_b<Pitch,false>(features+3*Plane,j*8,base,lane);
                    NativeTripleMma{}(jets[0][j],g,v,jets[0][j]);NativeTripleMma{}(jets[1][j],g,rv,jets[1][j]);
                    NativeTripleMma{}(jets[2][j],g,sv,jets[2][j]);NativeTripleMma{}(jets[2][j],sg,v,jets[2][j]);
                    NativeTripleMma{}(jets[3][j],sg,rv,jets[3][j]);
                }
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                int qr=row+lane/4+r*8;float delta=0.f,dr=0.f,ds=0.f,drs=0.f;
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,i=r*2+c,pos=qr*CPitch+key;
                        float prob=float(coefficients[pos]),pr=float(coefficients[CPlane+pos]);
                        float ps=float(coefficients[2*CPlane+pos]),prs=float(coefficients[3*CPlane+pos]);
                        float t=jets[0][j][i],tr=jets[1][j][i],ts=jets[2][j][i],trs=jets[3][j][i];
                        delta+=prob*t;dr+=pr*t+prob*tr;ds+=ps*t+prob*ts;drs+=prs*t+pr*ts+ps*tr+prob*trs;
                    }
                delta=native_triple_sum(delta);dr=native_triple_sum(dr);ds=native_triple_sum(ds);drs=native_triple_sum(drs);
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        int key=j*8+(lane%4)*2+c,i=r*2+c,pos=qr*CPitch+key;
                        float prob=float(coefficients[pos]),pr=float(coefficients[CPlane+pos]);
                        float ps=float(coefficients[2*CPlane+pos]),prs=float(coefficients[3*CPlane+pos]);
                        float t=jets[0][j][i]-delta,tr=jets[1][j][i]-dr,ts=jets[2][j][i]-ds,trs=jets[3][j][i]-drs;
                        float a0=prob*t,a1=pr*t+prob*tr,a2=ps*t+prob*ts,a3=prs*t+pr*ts+ps*tr+prob*trs;
                        coefficients[pos]=LowMmaElement(a0);coefficients[CPlane+pos]=LowMmaElement(a1);
                        coefficients[2*CPlane+pos]=LowMmaElement(a2);coefficients[3*CPlane+pos]=LowMmaElement(a3);
                        jets[0][j][i]=a3;jets[1][j][i]=a2;
                        if constexpr(BIAS_KIND==2) if(qr<n && key<n) {
                            if constexpr(NEEDS&8) p.outputs[3].at(segment,head,qr,key)=a3;
                            if constexpr(NEEDS&256) p.outputs[8].at(segment,head,qr,key)=a2;
                        }
                    }
            }
            if constexpr(BIAS_KIND==1) {
                #pragma unroll
                for(int j=0;j<Tile/8;++j)
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        float sum=jets[0][j][c]+jets[0][j][2+c],direction=jets[1][j][c]+jets[1][j][2+c];
                        #pragma unroll
                        for(int shift=4;shift<32;shift*=2) {
                            sum+=__shfl_xor_sync(0xffffffff,sum,shift);direction+=__shfl_xor_sync(0xffffffff,direction,shift);
                        }
                        if(lane<4) {
                            bias_partial[warp*Tile+j*8+lane*2+c]=sum;
                            bias_partial[(Tile/16)*Tile+warp*Tile+j*8+lane*2+c]=direction;
                        }
                    }
            }
        }
        __syncthreads();
        if constexpr(BIAS_KIND==1) if(threadIdx.x<n) {
            float sum=0.f,direction=0.f;
            for(int stripe=0;stripe<Tile/16;++stripe) {
                sum+=bias_partial[stripe*Tile+threadIdx.x];direction+=bias_partial[(Tile/16)*Tile+stripe*Tile+threadIdx.x];
            }
            if constexpr(NEEDS&8) p.outputs[3].at(begin+threadIdx.x,head)=sum;
            if constexpr(NEEDS&256) p.outputs[8].at(begin+threadIdx.x,head)=direction;
        }
        load<1,0>(p,features,begin,n,head);load<6,1>(p,features,begin,n,head);load<10,2>(p,features,begin,n,head);
        __syncthreads();
        output<0,3,0,1,2,2,1>(p,features,coefficients,begin,n,head);
        output<5,2,0,0,2>(p,features,coefficients,begin,n,head);
        load<0,0>(p,features,begin,n,head);load<5,1>(p,features,begin,n,head);load<9,2>(p,features,begin,n,head);
        __syncthreads();
        output<1,3,0,1,2,2,1,true>(p,features,coefficients,begin,n,head);
        output<6,2,0,0,2,-1,-1,true>(p,features,coefficients,begin,n,head);
    }
};
#endif
'''

SOURCE = r'''
constexpr int CACHE_PLANES = FEATURE_WIDTH <= 32 ? 5 : 6;
#if LOW_PRECISION
constexpr int TRIPLE_LONG_PITCH=FEATURE_WIDTH+8;
constexpr int TRIPLE_LONG_PLANE=32*TRIPLE_LONG_PITCH;
constexpr int TRIPLE_LONG_BYTES=6*TRIPLE_LONG_PLANE*sizeof(LowMmaElement)+4*(8192+256)+4*32*40*sizeof(LowMmaElement);

template<int Input,int Slot>
__device__ void triple_cache(AttentionParams const& p,LowMmaElement* cache,
    int64_t qb,int64_t kb,int64_t end,int head,int channel_base) {
    constexpr int Channels=Input==0 || Input==1 || Input==5 || Input==6 || Input==9 || Input==10 ? DIM:VALUE_DIM;
    int64_t base=Input==0 || Input==4 || Input==5 || Input==9 || Input==13 ? qb:kb;
    auto const& source=p.inputs[Input];
    if constexpr((PRESENT&(1<<Input)) && Channels%8==0) {
        if(!source.fp32 && source.stride[2]==1 && source.stride[0]%8==0 && source.stride[1]%8==0
            && (reinterpret_cast<uintptr_t>(source.data)&15)==0) {
            for(int i=threadIdx.x*8;i<32*FEATURE_WIDTH;i+=blockDim.x*8) {
                int row=i/FEATURE_WIDTH,c=i%FEATURE_WIDTH;uint4 value={0,0,0,0};
                if(base+row<end && channel_base+c<Channels)
                    value=*reinterpret_cast<uint4 const*>(&source.feature(base+row,head,channel_base+c));
                *reinterpret_cast<uint4*>(cache+Slot*TRIPLE_LONG_PLANE+row*TRIPLE_LONG_PITCH+c)=value;
            }
            return;
        }
    }
    for(int i=threadIdx.x;i<32*FEATURE_WIDTH;i+=blockDim.x) {
        int row=i/FEATURE_WIDTH,c=i%FEATURE_WIDTH;float value=0;
        if constexpr(PRESENT&(1<<Input)) if(base+row<end && channel_base+c<Channels)
            value=float(source.feature(base+row,head,channel_base+c));
        cache[Slot*TRIPLE_LONG_PLANE+row*TRIPLE_LONG_PITCH+c]=LowMmaElement(value);
    }
}

__device__ void triple_pairs(AttentionParams const& p,LowMmaElement* cache,float* probability,float* adjoint,
    int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=warp/4*16,column=warp%4*8;
    NativeTripleMma::FragmentC score,direction,second,mixed;
    score.clear();direction.clear();second.clear();mixed.clear();
    for(int chunk=0;chunk<DIM;chunk+=FEATURE_WIDTH) {
        triple_cache<0,0>(p,cache,qb,kb,end,head,chunk);triple_cache<1,1>(p,cache,qb,kb,end,head,chunk);
        triple_cache<5,2>(p,cache,qb,kb,end,head,chunk);triple_cache<6,3>(p,cache,qb,kb,end,head,chunk);
        triple_cache<9,4>(p,cache,qb,kb,end,head,chunk);triple_cache<10,5>(p,cache,qb,kb,end,head,chunk);
        __syncthreads();
        #pragma unroll
        for(int c=0;c<FEATURE_WIDTH;c+=16) if(chunk+c<DIM) {
            auto q=native_triple_a<TRIPLE_LONG_PITCH,false>(cache,row,c,lane);
            auto k=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+TRIPLE_LONG_PLANE,column,c,lane);
            auto rq=native_triple_a<TRIPLE_LONG_PITCH,false>(cache+2*TRIPLE_LONG_PLANE,row,c,lane);
            auto rk=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+3*TRIPLE_LONG_PLANE,column,c,lane);
            auto sq=native_triple_a<TRIPLE_LONG_PITCH,false>(cache+4*TRIPLE_LONG_PLANE,row,c,lane);
            auto sk=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+5*TRIPLE_LONG_PLANE,column,c,lane);
            NativeTripleMma{}(score,q,k,score);
            if constexpr(PRESENT&(1<<5)) NativeTripleMma{}(direction,rq,k,direction);
            if constexpr(PRESENT&(1<<6)) NativeTripleMma{}(direction,q,rk,direction);
            if constexpr(PRESENT&(1<<9)) NativeTripleMma{}(second,sq,k,second);
            if constexpr(PRESENT&(1<<10)) NativeTripleMma{}(second,q,sk,second);
            if constexpr((PRESENT&(1<<9)) && (PRESENT&(1<<6))) NativeTripleMma{}(mixed,sq,rk,mixed);
            if constexpr((PRESENT&(1<<5)) && (PRESENT&(1<<10))) NativeTripleMma{}(mixed,rq,sk,mixed);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        int r=row+lane/4+(i/2)*8,c=column+(lane%4)*2+i%2,pos=r*32+c;
        float prob=0,a=0,b=0,d=0;
        if(qb+r<end && kb+c<end && (!CAUSAL || kb+c<=qb+r)) {
            float s=score[i]*p.scale+higher_bias<3>(p,segment,head,qb-begin+r,kb-begin+c,kb+c);
            float inv=p.state.at(qb+r,head,1);
            if(inv>0 && s>-INFINITY) prob=__expf(s-p.state.at(qb+r,head,0))*inv;
            a=direction[i]*p.scale+higher_bias<8>(p,segment,head,qb-begin+r,kb-begin+c,kb+c);
            b=second[i]*p.scale+higher_bias<12>(p,segment,head,qb-begin+r,kb-begin+c,kb+c);
            d=mixed[i]*p.scale;
        }
        probability[pos]=prob;probability[1024+pos]=a;probability[2048+pos]=b;probability[3072+pos]=d;
    }
    NativeTripleMma::FragmentC value,first_value,second_value,mixed_value;
    value.clear();first_value.clear();second_value.clear();mixed_value.clear();
    for(int chunk=0;chunk<VALUE_DIM;chunk+=FEATURE_WIDTH) {
        triple_cache<4,0>(p,cache,qb,kb,end,head,chunk);triple_cache<2,1>(p,cache,qb,kb,end,head,chunk);
        triple_cache<7,2>(p,cache,qb,kb,end,head,chunk);triple_cache<11,3>(p,cache,qb,kb,end,head,chunk);
        triple_cache<13,4>(p,cache,qb,kb,end,head,chunk);
        __syncthreads();
        #pragma unroll
        for(int c=0;c<FEATURE_WIDTH;c+=16) if(chunk+c<VALUE_DIM) {
            auto g=native_triple_a<TRIPLE_LONG_PITCH,false>(cache,row,c,lane);
            auto v=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+TRIPLE_LONG_PLANE,column,c,lane);
            auto rv=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+2*TRIPLE_LONG_PLANE,column,c,lane);
            auto sv=native_triple_b<TRIPLE_LONG_PITCH,false>(cache+3*TRIPLE_LONG_PLANE,column,c,lane);
            auto sg=native_triple_a<TRIPLE_LONG_PITCH,false>(cache+4*TRIPLE_LONG_PLANE,row,c,lane);
            NativeTripleMma{}(value,g,v,value);
            if constexpr(PRESENT&(1<<7)) NativeTripleMma{}(first_value,g,rv,first_value);
            if constexpr(PRESENT&(1<<11)) NativeTripleMma{}(second_value,g,sv,second_value);
            if constexpr(PRESENT&(1<<13)) NativeTripleMma{}(second_value,sg,v,second_value);
            if constexpr((PRESENT&(1<<13)) && (PRESENT&(1<<7))) NativeTripleMma{}(mixed_value,sg,rv,mixed_value);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        int r=row+lane/4+(i/2)*8,c=column+(lane%4)*2+i%2,pos=r*32+c;
        adjoint[pos]=value[i];adjoint[1024+pos]=first_value[i];
        adjoint[2048+pos]=second_value[i];adjoint[3072+pos]=mixed_value[i];
    }
    __syncthreads();
}

template<int Target,int Direction,bool Transpose,int S0,int S1,int S2=-1,bool Atomic=false>
__device__ void triple_contract(AttentionParams const& p,float const* pairs,LowMmaElement* cache,
    int64_t begin,int64_t end,int64_t qb,int64_t kb,int head) {
    auto* coefficients=reinterpret_cast<LowMmaElement*>(const_cast<float*>(pairs)+(Target==2 || Target==4 ? 8192:4096)+256);
    if constexpr(Target==2 || Target==0) {
        for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
            int pos=i/32*40+i%32;
            #pragma unroll
            for(int j=0;j<4;++j) coefficients[j*32*40+pos]=LowMmaElement(pairs[j*1024+i]);
        }
        __syncthreads();
    }
    constexpr unsigned DirectionMask=Direction<0 ? 0u:1u<<Direction;
    if constexpr((NEEDS&(1<<Target)) || (NEEDS&DirectionMask)) {
        constexpr bool QK=Target==0 || Target==1,Cached=(Target==2 || Target==4) && VALUE_DIM<=FEATURE_WIDTH;
        constexpr int Width=QK ? DIM:VALUE_DIM,Columns=FEATURE_WIDTH/8;
        constexpr int Slot0=Cached ? (Target==4 ? 1:0):0,Slot1=Cached ? (Target==2 ? 4:3):1;
        int lane=threadIdx.x%32;
        for(int chunk=0;chunk<Width;chunk+=FEATURE_WIDTH) {
            if constexpr(!Cached) {
                triple_cache<S0,0>(p,cache,qb,kb,end,head,chunk);triple_cache<S1,1>(p,cache,qb,kb,end,head,chunk);
                if constexpr(S2>=0) triple_cache<S2,2>(p,cache,qb,kb,end,head,chunk);
                __syncthreads();
            }
            for(int tile=threadIdx.x/32;tile<2*Columns;tile+=8) {
                int row=tile/Columns*16,column=tile%Columns*8;
                NativeTripleMma::FragmentC result,direction;result.clear();direction.clear();
                #pragma unroll
                for(int base=0;base<32;base+=16) {
                    auto f0=native_triple_b<TRIPLE_LONG_PITCH,true>(cache+Slot0*TRIPLE_LONG_PLANE,column,base,lane);
                    auto f1=native_triple_b<TRIPLE_LONG_PITCH,true>(cache+Slot1*TRIPLE_LONG_PLANE,column,base,lane);
                    if constexpr(NEEDS&(1<<Target)) {
                        auto a=native_triple_a<40,Transpose>(coefficients+3*32*40,row,base,lane);
                        NativeTripleMma{}(result,a,f0,result);
                        a=native_triple_a<40,Transpose>(coefficients+32*40,row,base,lane);
                        NativeTripleMma{}(result,a,f1,result);
                        if constexpr(S2>=0) {
                            a=native_triple_a<40,Transpose>(coefficients+2*32*40,row,base,lane);
                            auto f2=native_triple_b<TRIPLE_LONG_PITCH,true>(cache+2*TRIPLE_LONG_PLANE,column,base,lane);
                            NativeTripleMma{}(result,a,f2,result);
                        }
                    }
                    if constexpr(NEEDS&DirectionMask) {
                        auto a=native_triple_a<40,Transpose>(coefficients+2*32*40,row,base,lane);
                        NativeTripleMma{}(direction,a,f0,direction);
                        a=native_triple_a<40,Transpose>(coefficients,row,base,lane);
                        NativeTripleMma{}(direction,a,f1,direction);
                    }
                }
                #pragma unroll
                for(int i=0;i<4;++i) {
                    int64_t r=(Transpose ? kb:qb)+row+lane/4+(i/2)*8;
                    int c=chunk+column+(lane%4)*2+i%2;
                    if(r<end && c<Width) {
                        float scale=QK ? p.scale:1.f;bool add=(Transpose ? qb:kb)>begin;
                        if constexpr(NEEDS&(1<<Target)) {
                            if constexpr(Atomic && Transpose) atomicAdd(&p.outputs[Target].fp32_at(r,head,c),result[i]*scale);
                            else {auto&& out=p.outputs[Target].feature(r,head,c);out=result[i]*scale+(add?float(out):0.f);}
                        }
                        if constexpr(NEEDS&DirectionMask) {
                            if constexpr(Atomic && Transpose) atomicAdd(&p.outputs[Direction].fp32_at(r,head,c),direction[i]*scale);
                            else {auto&& out=p.outputs[Direction].feature(r,head,c);out=direction[i]*scale+(add?float(out):0.f);}
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();
}
#else
constexpr int TRIPLE_LONG_BYTES=4*(CACHE_PLANES*33*FEATURE_WIDTH+8192+256);
#endif


#if !LOW_PRECISION
template <int TARGET, int DIRECTION, bool TRANSPOSE, int S0, int S1, int S2 = -1, bool Atomic = false>
__device__ __forceinline__ void triple_contract(
    AttentionParams const& p, float const* pairs, float* cache,
    int64_t begin, int64_t end, int64_t qb, int64_t kb, int head) {
    constexpr unsigned DirectionMask = DIRECTION<0 ? 0u : (1u<<DIRECTION);
    if constexpr ((NEEDS & (1<<TARGET)) || (NEEDS & DirectionMask)) {
        constexpr bool qk = TARGET==0 || TARGET==1;
        constexpr int width = qk ? DIM : VALUE_DIM;
        // Consume the value tile left by triple_pairs before Q/K reuse it.
        constexpr bool cached = (TARGET==2 || TARGET==4) && VALUE_DIM<=FEATURE_WIDTH;
        constexpr int Slot0 = TARGET==4 ? 1 : 0;
        constexpr int Slot1 = TARGET==2 ? 4 : TARGET==4 ? 3 : 1;
        const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
        const int64_t base = TRANSPOSE ? kb : qb;
        const int count = int(min(int64_t(32), end-(TRANSPOSE ? qb : kb)));
        for (int channel_base = 0; channel_base<width; channel_base+=FEATURE_WIDTH) {
            if constexpr (!cached) {
                cache_feature<S0, 0>(p, cache, qb, kb, end, head, channel_base);
                cache_feature<S1, 1>(p, cache, qb, kb, end, head, channel_base);
                if constexpr (S2>=0) cache_feature<S2, 2>(p, cache, qb, kb, end, head, channel_base);
                __syncthreads();
            }
            #pragma unroll
            for (int chunk = 0; chunk<(FEATURE_WIDTH+31)/32; chunk++) {
                const int local_channel = lane+chunk*32;
                const int channel = channel_base+local_channel;
                if (channel<width && local_channel<FEATURE_WIDTH && base+row<end) {
                    float result[4] = {}, direction[4] = {};
                    #pragma unroll 2
                    for (int key = 0; key<count; key++) {
                        const float f0 = cache[(cached ? Slot0 : 0)*33*FEATURE_WIDTH+local_channel*33+key];
                        const float f1 = cache[(cached ? Slot1 : 1)*33*FEATURE_WIDTH+local_channel*33+key];
                        float f2 = 0;
                        if constexpr (S2>=0 && (NEEDS & (1<<TARGET)))
                            f2 = cache[66*FEATURE_WIDTH+local_channel*33+key];
                        #pragma unroll
                        for (int c = 0; c<4; c++) {
                            const int pos = TRANSPOSE ? key*32+row+c : (row+c)*32+key;
                            const float a2 = pairs[2048+pos];
                            if constexpr (NEEDS & (1<<TARGET)) {
                                result[c] = multiply_add(pairs[3072+pos], f0, result[c]);
                                result[c] = multiply_add(pairs[1024+pos], f1, result[c]);
                                if constexpr (S2>=0) result[c] = multiply_add(a2, f2, result[c]);
                            }
                            if constexpr (NEEDS & DirectionMask) {
                                direction[c] = multiply_add(a2, f0, direction[c]);
                                direction[c] = multiply_add(pairs[pos], f1, direction[c]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int c = 0; c<4; c++) if (base+row+c<end) {
                        const float scale = qk ? p.scale : 1.f;
                        const bool accumulate = (TRANSPOSE ? qb : kb)>begin;
                        if constexpr (NEEDS & (1<<TARGET)) {
                            auto&& out = p.outputs[TARGET].feature(base+row+c,head,channel);
                            if constexpr(Atomic && TRANSPOSE) atomicAdd(&p.outputs[TARGET].fp32_at(base+row+c,head,channel),result[c]*scale);
                            else out = result[c]*scale+(accumulate ? float(out) : 0.f);
                        }
                        if constexpr (NEEDS & DirectionMask) {
                            auto&& out = p.outputs[DIRECTION].feature(base+row+c,head,channel);
                            if constexpr(Atomic && TRANSPOSE) atomicAdd(&p.outputs[DIRECTION].fp32_at(base+row+c,head,channel),direction[c]*scale);
                            else out = direction[c]*scale+(accumulate ? float(out) : 0.f);
                        }
                    }
                }
            }
            if constexpr (!cached) __syncthreads();
        }
    }
}

__device__ __forceinline__ void triple_pairs(AttentionParams const& p, float* cache, float* probability, float* adjoint,
    int64_t begin, int64_t end, int64_t qb, int64_t kb, int segment, int head) {
    const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
    float score[4] = {}, direction[4] = {}, second_direction[4] = {}, mixed_direction[4] = {};
    for (int channel_base = 0; channel_base<DIM; channel_base+=FEATURE_WIDTH) {
        cache_feature<0, 0>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<1, 1>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<5, 2>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<6, 3>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<9, 4>(p, cache, qb, kb, end, head, channel_base);
        if constexpr (CACHE_PLANES==6)
            cache_feature<10, 5>(p, cache, qb, kb, end, head, channel_base);
        __syncthreads();
        #pragma unroll 4
        for (int c = 0; c<FEATURE_WIDTH; c++) {
            #pragma unroll
            for (int r = 0; r<4; r++) {
                int q = c*33+row+r, k = c*33+lane;
                if (channel_base+c<DIM && qb+row<end) {
                    score[r] = multiply_add(cache[q], cache[33*FEATURE_WIDTH+k], score[r]);
                    if constexpr (PRESENT&(1<<5)) direction[r] = multiply_add(cache[2*33*FEATURE_WIDTH+q],
                        cache[33*FEATURE_WIDTH+k], direction[r]);
                    if constexpr (PRESENT&(1<<6)) direction[r] = multiply_add(cache[q],
                        cache[3*33*FEATURE_WIDTH+k], direction[r]);
                    if constexpr (PRESENT&(1<<9)) second_direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q],
                        cache[33*FEATURE_WIDTH+k], second_direction[r]);
                    if constexpr (CACHE_PLANES==6 && (PRESENT&(1<<10))) second_direction[r] = multiply_add(cache[q], cache[5*33*FEATURE_WIDTH+k], second_direction[r]);
                    if constexpr (CACHE_PLANES==6 && ((PRESENT&(1<<5)) && (PRESENT&(1<<10)))) mixed_direction[r] = multiply_add(cache[2*33*FEATURE_WIDTH+q], cache[5*33*FEATURE_WIDTH+k], mixed_direction[r]);
                    if constexpr (CACHE_PLANES==6 && ((PRESENT&(1<<9)) && (PRESENT&(1<<6)))) mixed_direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q], cache[3*33*FEATURE_WIDTH+k], mixed_direction[r]);
                }
            }
        }
        __syncthreads();
        if constexpr (CACHE_PLANES==5) {
            // Reuse K after its last product, keeping four CTAs resident at width 32.
            cache_feature<10, 1>(p, cache, qb, kb, end, head, channel_base);
            __syncthreads();
            #pragma unroll 4
            for (int c = 0; c<FEATURE_WIDTH; c++) {
                #pragma unroll
                for (int r = 0; r<4; r++) {
                    int q = c*33+row+r, k = c*33+lane;
                    if (channel_base+c<DIM && qb+row<end) {
                        if constexpr (PRESENT&(1<<10)) second_direction[r] = multiply_add(cache[q],
                            cache[33*FEATURE_WIDTH+k], second_direction[r]);
                        if constexpr ((PRESENT&(1<<5)) && (PRESENT&(1<<10))) mixed_direction[r] = multiply_add(cache[2*33*FEATURE_WIDTH+q], cache[33*FEATURE_WIDTH+k], mixed_direction[r]);
                        if constexpr ((PRESENT&(1<<9)) && (PRESENT&(1<<6))) mixed_direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q], cache[3*33*FEATURE_WIDTH+k], mixed_direction[r]);
                    }
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int r = 0; r<4; r++) {
        float probability_value = 0, direction_value = 0, second_direction_value = 0, mixed_direction_value = 0;
        if (qb+row+r<end && kb+lane<end && (!CAUSAL || kb+lane<=qb+row+r)) {
            float s = score[r]*p.scale+higher_bias<3>(p, segment, head, qb-begin+row+r, kb-begin+lane, kb+lane);
            float inv = p.state.data[((qb+row+r)*HEADS+head)*2+1];
            if (inv>0 && s>-INFINITY) probability_value = __expf(s-p.state.data[((qb+row+r)*HEADS+head)*2+0])*inv;
            direction_value = direction[r]*p.scale+higher_bias<8>(p, segment, head, qb-begin+row+r, kb-begin+lane, kb+lane);
            second_direction_value = second_direction[r]*p.scale+higher_bias<12>(p, segment, head, qb-begin+row+r,
                kb-begin+lane, kb+lane);
            mixed_direction_value = mixed_direction[r]*p.scale;
        }
        int pos = (row+r)*32+lane;
        probability[pos] = probability_value;
        probability[1024+pos] = direction_value;
        probability[2048+pos] = second_direction_value;
        probability[3072+pos] = mixed_direction_value;
    }
    float value[4] = {}, value_direction[4] = {}, second_value_direction[4] = {}, mixed_value_direction[4] = {};
    for (int channel_base = 0; channel_base<VALUE_DIM; channel_base+=FEATURE_WIDTH) {
        cache_feature<4, 0>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<2, 1>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<7, 2>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<11, 3>(p, cache, qb, kb, end, head, channel_base);
        cache_feature<13, 4>(p, cache, qb, kb, end, head, channel_base);
        __syncthreads();
        #pragma unroll 4
        for (int c = 0; c<FEATURE_WIDTH; c++) {
            #pragma unroll
            for (int r = 0; r<4; r++) {
                int q = c*33+row+r, k = c*33+lane;
                if (channel_base+c<VALUE_DIM && qb+row<end) {
                    value[r] = multiply_add(cache[0*33*FEATURE_WIDTH+q], cache[1*33*FEATURE_WIDTH+k], value[r]);
                    if constexpr (PRESENT&(1<<7)) value_direction[r] = multiply_add(cache[0*33*FEATURE_WIDTH+q],
                        cache[2*33*FEATURE_WIDTH+k], value_direction[r]);
                    if constexpr (PRESENT&(1<<11)) second_value_direction[r] = multiply_add(cache[0*33*FEATURE_WIDTH+q], cache[3*33*FEATURE_WIDTH+k], second_value_direction[r]);
                    if constexpr (PRESENT&(1<<13)) second_value_direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q], cache[1*33*FEATURE_WIDTH+k], second_value_direction[r]);
                    if constexpr ((PRESENT&(1<<13)) && (PRESENT&(1<<7))) mixed_value_direction[r] = multiply_add(cache[4*33*FEATURE_WIDTH+q], cache[2*33*FEATURE_WIDTH+k], mixed_value_direction[r]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r<4; r++) {
        int pos = (row+r)*32+lane;
        adjoint[pos] = value[r];
        adjoint[1024+pos] = value_direction[r];
        adjoint[2048+pos] = second_value_direction[r];
        adjoint[3072+pos] = mixed_value_direction[r];
    }
    __syncthreads();
}

#endif

template<int Tile>
__device__ __forceinline__ int triple_segment_index(int row,int column) {
    return row*Tile+column;
}
constexpr int TRIPLE_SEGMENT_WIDTH=SEGMENT_LIMIT;
template<int Tile> constexpr int TRIPLE_SEGMENT_FEATURES=Tile==64 ? SEGMENT_FEATURE64:32;
template<int Tile> constexpr int TRIPLE_SEGMENT_CACHE=
    (TRIPLE_SEGMENT_WIDTH>6*TRIPLE_SEGMENT_FEATURES<Tile> ? TRIPLE_SEGMENT_WIDTH:6*TRIPLE_SEGMENT_FEATURES<Tile>)*(Tile+1);
template<int Tile>
__device__ void triple_segment_scores(AttentionParams const& p,float* cache,float* target,
    int64_t begin,int64_t end,int head) {
    constexpr int Features=TRIPLE_SEGMENT_FEATURES<Tile>;
    constexpr int Rows=Tile/16,Plane=Features*(Tile+1);
    const int row=threadIdx.x/16*Rows,col=threadIdx.x%16;
    float score[Rows][Rows]={},dr[Rows][Rows]={},ds[Rows][Rows]={},drs[Rows][Rows]={};
    for(int base=0;base<DIM;base+=Features) {
        cache_feature<0,0,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<1,1,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<5,2,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<6,3,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<9,4,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<10,5,Features,Tile>(p,cache,begin,begin,end,head,base);
        __syncthreads();
        if(begin+row<end) {
            #pragma unroll 2
            for(int c=0;c<min(Features,DIM-base);++c) {
                #pragma unroll
                for(int r=0;r<Rows;++r) {
                    float q=cache[c*(Tile+1)+row+r];
                    float rq=cache[2*Plane+c*(Tile+1)+row+r];
                    float sq=cache[4*Plane+c*(Tile+1)+row+r];
                    #pragma unroll
                    for(int k=0;k<Rows;++k) {
                        float v=cache[Plane+c*(Tile+1)+col+k*16];
                        float rv=cache[3*Plane+c*(Tile+1)+col+k*16];
                        float sv=cache[5*Plane+c*(Tile+1)+col+k*16];
                        score[r][k]=multiply_add(q,v,score[r][k]);
                        if constexpr(PRESENT&(1<<5)) dr[r][k]=multiply_add(rq,v,dr[r][k]);
                        if constexpr(PRESENT&(1<<6)) dr[r][k]=multiply_add(q,rv,dr[r][k]);
                        if constexpr(PRESENT&(1<<9)) ds[r][k]=multiply_add(sq,v,ds[r][k]);
                        if constexpr(PRESENT&(1<<10)) ds[r][k]=multiply_add(q,sv,ds[r][k]);
                        if constexpr((PRESENT&(1<<5)) && (PRESENT&(1<<10))) drs[r][k]=multiply_add(rq,sv,drs[r][k]);
                        if constexpr((PRESENT&(1<<9)) && (PRESENT&(1<<6))) drs[r][k]=multiply_add(sq,rv,drs[r][k]);
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
            int pos=triple_segment_index<Tile>(row+r,col+k*16);
            target[pos]=score[r][k];target[Tile*Tile+pos]=dr[r][k];
            target[2*Tile*Tile+pos]=ds[r][k];target[3*Tile*Tile+pos]=drs[r][k];
        }
    }
    __syncthreads();
}

template<int Tile,int Source,int Target,int Direction,bool Transpose,int Primary,int Derivative>
__device__ __forceinline__ void triple_segment_add(AttentionParams const& p,float* cache,float const* weights,
    int64_t begin,int64_t end,int head,int base,
    float (&result)[Tile/8][(TRIPLE_SEGMENT_WIDTH+31)/32],float (&direction)[Tile/8][(TRIPLE_SEGMENT_WIDTH+31)/32]) {
    constexpr unsigned DirectionMask=Direction<0 ? 0u:(1u<<Direction);
    constexpr int Width=Target<2 ? DIM:VALUE_DIM;
    const int row=threadIdx.x/32*(Tile/8),lane=threadIdx.x%32;
    cache_feature<Source,0,TRIPLE_SEGMENT_WIDTH,Tile>(p,cache,begin,begin,end,head,base);
    __syncthreads();
    if(begin+row<end) {
        #pragma unroll 2
        for(int k=0;k<end-begin;k++) {
            #pragma unroll
            for(int r=0;r<Tile/8;r++) {
                int pos=Transpose ? triple_segment_index<Tile>(k,row+r):triple_segment_index<Tile>(row+r,k);
                float a=0,b=0;
                if constexpr(NEEDS&(1<<Target)) a=weights[Primary*Tile*Tile+pos];
                if constexpr(Derivative>=0 && (NEEDS&DirectionMask)) b=weights[Derivative*Tile*Tile+pos];
                #pragma unroll
                for(int c=0;c<(TRIPLE_SEGMENT_WIDTH+31)/32;c++) {
                    int channel=lane+c*32;
                    if(base+channel<Width && channel<TRIPLE_SEGMENT_WIDTH) {
                        float feature=cache[channel*(Tile+1)+k];
                        if constexpr(NEEDS&(1<<Target)) result[r][c]=multiply_add(a,feature,result[r][c]);
                        if constexpr(Derivative>=0 && (NEEDS&DirectionMask)) direction[r][c]=multiply_add(b,feature,direction[r][c]);
                    }
                }
            }
        }
    }
    __syncthreads();
}

template<int Tile,int Target,int Direction,bool Transpose,int S0,int S1,int S2=-1>
__device__ void triple_segment_output(AttentionParams const& p,float* cache,float const* weights,
    int64_t begin,int64_t end,int head) {
    constexpr unsigned DirectionMask=Direction<0 ? 0u:(1u<<Direction);
    if constexpr((NEEDS&(1<<Target)) || (NEEDS&DirectionMask)) {
        constexpr int Width=Target<2 ? DIM:VALUE_DIM;
        const int row=threadIdx.x/32*(Tile/8),lane=threadIdx.x%32;
        for(int base=0;base<Width;base+=TRIPLE_SEGMENT_WIDTH) {
            float result[Tile/8][(TRIPLE_SEGMENT_WIDTH+31)/32]={},direction[Tile/8][(TRIPLE_SEGMENT_WIDTH+31)/32]={};
            triple_segment_add<Tile,S0,Target,Direction,Transpose,3,2>(p,cache,weights,begin,end,head,base,result,direction);
            triple_segment_add<Tile,S1,Target,Direction,Transpose,1,0>(p,cache,weights,begin,end,head,base,result,direction);
            if constexpr(S2>=0 && (NEEDS&(1<<Target)))
                triple_segment_add<Tile,S2,Target,Direction,Transpose,2,-1>(p,cache,weights,begin,end,head,base,result,direction);
            #pragma unroll
            for(int r=0;r<Tile/8;r++) if(begin+row+r<end) {
                #pragma unroll
                for(int c=0;c<(TRIPLE_SEGMENT_WIDTH+31)/32;c++) if(base+lane+c*32<Width && lane+c*32<TRIPLE_SEGMENT_WIDTH) {
                    const float scale=Target<2 ? p.scale:1.f;
                    if constexpr(NEEDS&(1<<Target)) p.outputs[Target].feature(begin+row+r,head,base+lane+c*32)=result[r][c]*scale;
                    if constexpr(NEEDS&DirectionMask) p.outputs[Direction].feature(begin+row+r,head,base+lane+c*32)=direction[r][c]*scale;
                }
            }
        }
    }
}

// Consume probability jets, then overwrite them with centered adjoint jets.
template<int Tile>
__device__ void triple_segment_adjoint(AttentionParams const& p,float* cache,float* weights,
    int64_t begin,int64_t end,int head) {
    constexpr int Features=TRIPLE_SEGMENT_FEATURES<Tile>;
    const int row=threadIdx.x/16*(Tile/16),col=threadIdx.x%16;
    float t[Tile/16][Tile/16]={},tr[Tile/16][Tile/16]={},ts[Tile/16][Tile/16]={},trs[Tile/16][Tile/16]={};
    for(int base=0;base<VALUE_DIM;base+=Features) {
        cache_feature<4,0,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<2,1,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<7,2,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<11,3,Features,Tile>(p,cache,begin,begin,end,head,base);
        cache_feature<13,4,Features,Tile>(p,cache,begin,begin,end,head,base);
        __syncthreads();
        if(begin+row<end) {
            #pragma unroll 2
            for(int c=0;c<min(Features,VALUE_DIM-base);c++) {
                #pragma unroll
                for(int r=0;r<Tile/16;r++) {
                    float g=cache[c*(Tile+1)+row+r],sg=cache[4*Features*(Tile+1)+c*(Tile+1)+row+r];
                    #pragma unroll
                    for(int k=0;k<Tile/16;k++) {
                        float v=cache[Features*(Tile+1)+c*(Tile+1)+col+k*16];
                        float rv=cache[2*Features*(Tile+1)+c*(Tile+1)+col+k*16];
                        float sv=cache[3*Features*(Tile+1)+c*(Tile+1)+col+k*16];
                        t[r][k]=multiply_add(g,v,t[r][k]);
                        if constexpr(PRESENT&(1<<7)) tr[r][k]=multiply_add(g,rv,tr[r][k]);
                        if constexpr(PRESENT&(1<<11)) ts[r][k]=multiply_add(g,sv,ts[r][k]);
                        if constexpr(PRESENT&(1<<13)) ts[r][k]=multiply_add(sg,v,ts[r][k]);
                        if constexpr((PRESENT&(1<<13)) && (PRESENT&(1<<7))) trs[r][k]=multiply_add(sg,rv,trs[r][k]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<Tile/16;r++) {
        float p0[Tile/16],pr[Tile/16],ps[Tile/16],prs[Tile/16];
        float delta[4]={};
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=triple_segment_index<Tile>(row+r,col+k*16);
            p0[k]=weights[pos];pr[k]=weights[Tile*Tile+pos];ps[k]=weights[2*Tile*Tile+pos];prs[k]=weights[3*Tile*Tile+pos];
            float v=t[r][k],vr=tr[r][k],vs=ts[r][k],vrs=trs[r][k];
            t[r][k]=p0[k]*v;
            tr[r][k]=pr[k]*v+p0[k]*vr;
            ts[r][k]=ps[k]*v+p0[k]*vs;
            trs[r][k]=prs[k]*v+pr[k]*vs+ps[k]*vr+p0[k]*vrs;
            delta[0]+=t[r][k];delta[1]+=tr[r][k];delta[2]+=ts[r][k];delta[3]+=trs[r][k];
        }
        #pragma unroll
        for(int i=0;i<4;i++) {
            #pragma unroll
            for(int shift=1;shift<16;shift*=2) delta[i]+=__shfl_xor_sync(0xffffffff,delta[i],shift);
        }
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=triple_segment_index<Tile>(row+r,col+k*16);
            weights[pos]=t[r][k]-p0[k]*delta[0];
            weights[Tile*Tile+pos]=tr[r][k]-pr[k]*delta[0]-p0[k]*delta[1];
            weights[2*Tile*Tile+pos]=ts[r][k]-ps[k]*delta[0]-p0[k]*delta[2];
            weights[3*Tile*Tile+pos]=trs[r][k]-prs[k]*delta[0]-pr[k]*delta[2]-ps[k]*delta[1]-p0[k]*delta[3];
        }
    }
    __syncthreads();
}

template<int Tile>
__device__ __forceinline__ void triple_segment(AttentionParams const& p,int segment,int head) {
    const int64_t begin=p.offset(segment),end=p.offset(segment+1);
    if(end-begin<=Tile-16 || end-begin>Tile) return;
    extern __shared__ float shared[];
    float* probability=shared+TRIPLE_SEGMENT_CACHE<Tile>;
    float* adjoint=probability;
    triple_segment_scores<Tile>(p,shared,probability,begin,end,head);
    const int row=threadIdx.x/16*(Tile/16),col=threadIdx.x%16;
    #pragma unroll
    for(int r=0;r<Tile/16;r++) {
        float prob[Tile/16],a[Tile/16],b[Tile/16],c[Tile/16],stat[4]={};
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int key=col+k*16,pos=triple_segment_index<Tile>(row+r,key);
            bool valid=begin+row+r<end && begin+key<end && (!CAUSAL || key<=row+r);
            float score=valid ? probability[pos]*p.scale+higher_bias<3>(p,segment,head,row+r,key,begin+key):0.f;
            float inverse=valid ? p.state.at(begin+row+r,head,1):0.f;
            prob[k]=valid && inverse>0 && score>-INFINITY ? __expf(score-p.state.at(begin+row+r,head,0))*inverse:0.f;
            a[k]=probability[Tile*Tile+pos]*p.scale+(valid ? higher_bias<8>(p,segment,head,row+r,key,begin+key):0.f);
            b[k]=probability[2*Tile*Tile+pos]*p.scale+(valid ? higher_bias<12>(p,segment,head,row+r,key,begin+key):0.f);
            c[k]=probability[3*Tile*Tile+pos]*p.scale;
            stat[0]+=prob[k];stat[1]+=prob[k]*a[k];stat[2]+=prob[k]*b[k];stat[3]+=prob[k]*(c[k]+a[k]*b[k]);
        }
        #pragma unroll
        for(int i=0;i<4;i++) {
            #pragma unroll
            for(int shift=1;shift<16;shift*=2) stat[i]+=__shfl_xor_sync(0xffffffff,stat[i],shift);
        }
        float inverse=stat[0]>0 ? 1.f/stat[0]:0.f;
        stat[1]*=inverse;stat[2]*=inverse;stat[3]=stat[3]*inverse-stat[1]*stat[2];
        #pragma unroll
        for(int k=0;k<Tile/16;k++) {
            int pos=triple_segment_index<Tile>(row+r,col+k*16);
            prob[k]*=inverse;a[k]-=stat[1];b[k]-=stat[2];
            probability[pos]=prob[k];probability[Tile*Tile+pos]=prob[k]*a[k];probability[2*Tile*Tile+pos]=prob[k]*b[k];
            probability[3*Tile*Tile+pos]=prob[k]*(a[k]*b[k]+c[k]-stat[3]);
        }
    }
    __syncthreads();
    triple_segment_output<Tile,2,7,true,4,13>(p,shared,probability,begin,end,head);
    triple_segment_output<Tile,4,-1,false,2,11,7>(p,shared,probability,begin,end,head);
    triple_segment_adjoint<Tile>(p,shared,probability,begin,end,head);
    triple_segment_output<Tile,0,5,false,1,10,6>(p,shared,adjoint,begin,end,head);
    triple_segment_output<Tile,1,6,true,0,9,5>(p,shared,adjoint,begin,end,head);
    if constexpr((NEEDS&(1<<3)) && BIAS_KIND==1) {
        for(int k=threadIdx.x;k<end-begin;k+=blockDim.x) {
            float sum=0;
            for(int q=0;q<end-begin;q++) sum+=adjoint[3*Tile*Tile+triple_segment_index<Tile>(q,k)];
            p.outputs[3].at(begin+k,head)=sum;
        }
    } else if constexpr((NEEDS&(1<<3)) && BIAS_KIND==2) {
        for(int64_t i=threadIdx.x;i<p.outputs[3].size[2]*p.outputs[3].size[3];i+=blockDim.x) {
            int64_t q=i/p.outputs[3].size[3],k=i%p.outputs[3].size[3];
            p.outputs[3].at(segment,head,q,k)=q<end-begin && k<end-begin ? adjoint[3*Tile*Tile+triple_segment_index<Tile>(q,k)]:0.f;
        }
    }
    if constexpr((NEEDS&(1<<8)) && BIAS_KIND==1) {
        for(int k=threadIdx.x;k<end-begin;k+=blockDim.x) {
            float sum=0;
            for(int q=0;q<end-begin;q++) sum+=adjoint[2*Tile*Tile+triple_segment_index<Tile>(q,k)];
            p.outputs[8].at(begin+k,head)=sum;
        }
    } else if constexpr((NEEDS&(1<<8)) && BIAS_KIND==2) {
        for(int64_t i=threadIdx.x;i<p.outputs[8].size[2]*p.outputs[8].size[3];i+=blockDim.x) {
            int64_t q=i/p.outputs[8].size[3],k=i%p.outputs[8].size[3];
            p.outputs[8].at(segment,head,q,k)=q<end-begin && k<end-begin ? adjoint[2*Tile*Tile+triple_segment_index<Tile>(q,k)]:0.f;
        }
    }
}
template<int Tile> constexpr int RESIDENT_BYTES =
#if LOW_PRECISION
    TripleNative<Tile>::Bytes<=RESIDENT_BUDGET ? TripleNative<Tile>::Bytes:
#endif
    4*(TRIPLE_SEGMENT_CACHE<Tile>+4*Tile*Tile);
template<int Tile> constexpr int RESIDENT_THREADS =
#if LOW_PRECISION
    TripleNative<Tile>::Bytes<=RESIDENT_BUDGET ? 128:
#endif
    256;
template<int Tile>
__global__ __launch_bounds__(RESIDENT_THREADS<Tile>,Tile==48 ? 3:2) void attention_triple_backward_segment(const __grid_constant__ AttentionParams p) {
    #if LOW_PRECISION
    if constexpr(TripleNative<Tile>::Bytes<=RESIDENT_BUDGET) {
        int segment=blockIdx.x,head=blockIdx.y;
        int64_t begin=p.offset(segment),end=p.offset(segment+1);
        if(end-begin>(Tile==48?32:48) && end-begin<=Tile)
            TripleNative<Tile>::run(p,segment,head,begin,end-begin);
    } else
#endif
    triple_segment<Tile>(p,blockIdx.x,blockIdx.y);
}
#define MAIN_SHARED_BYTES (4*(COMBINE_SEGMENTS ? ((CACHE_PLANES*33*FEATURE_WIDTH+8192+256)>(TRIPLE_SEGMENT_CACHE<64>+4*64*64) ? (CACHE_PLANES*33*FEATURE_WIDTH+8192+256):(TRIPLE_SEGMENT_CACHE<64>+4*64*64)):(CACHE_PLANES*33*FEATURE_WIDTH+8192+256)))

template<bool Split = false, bool InitializeOnly = false, bool SerialLong = false>
__device__ __forceinline__ void triple_backward_segment(AttentionParams const& p,int64_t block_x,int block_y) {
    const int segment = Split ? block_x/8:block_x, head = block_y;
    const int64_t begin = p.offset(segment), end = p.offset(segment+1);
    if constexpr(InitializeOnly) if(end-begin<=64) return;
    if(end-begin>32 && end-begin<=64) {
        if constexpr(COMBINE_SEGMENTS && !Split) {
#if LOW_PRECISION
            if(end-begin<=48) TripleNative<48>::run(p,segment,head,begin,end-begin);
            else TripleNative<64>::run(p,segment,head,begin,end-begin);
#else
            if(end-begin<=48) triple_segment<48>(p,segment,head);
            else triple_segment<64>(p,segment,head);
#endif
        }
        return;
    }
    if constexpr(Split) {
        if(end-begin<=64) return;
    } else if(end-begin>64) {
        if constexpr(LOW_PRECISION && !InitializeOnly && !SerialLong) return;
        for(int64_t i=threadIdx.x;i<(end-begin)*max(DIM,VALUE_DIM);i+=blockDim.x) {
            int64_t r=begin+i/max(DIM,VALUE_DIM);
            int c=i%max(DIM,VALUE_DIM);
            if constexpr(NEEDS&(1<<1)) if(c<DIM) p.outputs[1].feature(r,head,c)=0;
            if constexpr(NEEDS&(1<<2)) if(c<VALUE_DIM) p.outputs[2].feature(r,head,c)=0;
            if constexpr(NEEDS&(1<<6)) if(c<DIM) p.outputs[6].feature(r,head,c)=0;
            if constexpr(NEEDS&(1<<7)) if(c<VALUE_DIM) p.outputs[7].feature(r,head,c)=0;
        }
        if constexpr((NEEDS&(1<<3)) && BIAS_KIND==1) {
            for(int64_t r=begin+threadIdx.x;r<end;r+=blockDim.x) p.outputs[3].at(r,head)=0;
        } else if constexpr((NEEDS&(1<<3)) && BIAS_KIND==2) {
            for(int64_t i=threadIdx.x;i<p.outputs[3].size[2]*p.outputs[3].size[3];i+=blockDim.x)
                p.outputs[3].at(segment,head,i/p.outputs[3].size[3],i%p.outputs[3].size[3])=0;
        }
        if constexpr((NEEDS&(1<<8)) && BIAS_KIND==1) {
            for(int64_t r=begin+threadIdx.x;r<end;r+=blockDim.x) p.outputs[8].at(r,head)=0;
        } else if constexpr((NEEDS&(1<<8)) && BIAS_KIND==2) {
            for(int64_t i=threadIdx.x;i<p.outputs[8].size[2]*p.outputs[8].size[3];i+=blockDim.x)
                p.outputs[8].at(segment,head,i/p.outputs[8].size[3],i%p.outputs[8].size[3])=0;
        }
        if constexpr(!SerialLong) return;
        __syncthreads();
    }

#if LOW_PRECISION
    if constexpr(!Split && !SerialLong && DIM<=128 && VALUE_DIM<=128) {
        TripleNative<32>::run(p,segment,head,begin,int(end-begin));
        return;
    }
#endif

    const int lane = threadIdx.x%32, row = threadIdx.x/32*4;
    extern __shared__ float shared[];
    #if LOW_PRECISION
    auto* cache=reinterpret_cast<LowMmaElement*>(shared);
    float* probability=reinterpret_cast<float*>(cache+6*TRIPLE_LONG_PLANE);
#else
    float* cache=shared;
    float* probability=cache+CACHE_PLANES*33*FEATURE_WIDTH;
#endif
    float* adjoint = probability+4096;
    float* statistics = adjoint+4096;
    if constexpr (!Split && BIAS_KIND==2) {
        if constexpr (NEEDS&(1<<3))
            for (int64_t i = threadIdx.x; i<p.outputs[3].size[2]*p.outputs[3].size[3]; i+=256)
                p.outputs[3].at(segment, head, i/p.outputs[3].size[3], i%p.outputs[3].size[3]) = 0;
        if constexpr (NEEDS&(1<<8))
            for (int64_t i = threadIdx.x; i<p.outputs[8].size[2]*p.outputs[8].size[3]; i+=256)
                p.outputs[8].at(segment, head, i/p.outputs[8].size[3], i%p.outputs[8].size[3]) = 0;
    }

    if (begin==end) return;
    if (end-begin<=32) {
        triple_pairs(p, cache, probability, adjoint, begin, end, begin, begin, segment, head);
        #pragma unroll
        for (int r = 0; r<4; r++) {
            int pos = (row+r)*32+lane;
            float prob = probability[pos], a = probability[1024+pos], b = probability[2048+pos], c = probability[3072+pos];
            const float total = higher_sum(prob);
            prob *= total>0 ? 1.f/total : 0.f;
            probability[pos] = prob;
            softmax_triple_backward<false>(prob,a,b,c,probability,adjoint,pos);
        }
        __syncthreads();
        triple_contract<2, 7, true, 4, 13, -1, Split>(p, probability, cache, begin, end, begin, begin, head);
        triple_contract<4, -1, false, 2, 11, 7, Split>(p, probability, cache, begin, end, begin, begin, head);
        // Both value contractions read this cache; synchronize before Q/K reuses it.
        if constexpr (VALUE_DIM<=FEATURE_WIDTH && (NEEDS & ((1<<2)|(1<<4)|(1<<7))))
            __syncthreads();
        triple_contract<0, 5, false, 1, 10, 6, Split>(p, adjoint, cache, begin, end, begin, begin, head);
        triple_contract<1, 6, true, 0, 9, 5, Split>(p, adjoint, cache, begin, end, begin, begin, head);
        higher_bias_gradient<3, 3, Split>(p, adjoint, segment, head, begin, end, begin, begin);
        higher_bias_gradient<8, 2, Split>(p, adjoint, segment, head, begin, end, begin, begin);
        return;
    }
    for (int64_t qb = begin+(Split ? (block_x%8)*32:0); qb<end; qb+=Split ? 256:32) {
        statistics[threadIdx.x] = 0.f;
        __syncthreads();
        for (int64_t kb = begin; kb<end; kb+=32) {
            triple_pairs(p, cache, probability, adjoint, begin, end, qb, kb, segment, head);
            #pragma unroll
            for (int r = 0; r<4; r++) {
                const int pos = (row+r)*32+lane;
                const float prob = probability[pos], a = probability[1024+pos];
                const float b = probability[2048+pos], c = probability[3072+pos];
                const float t = adjoint[pos], tr = adjoint[1024+pos];
                const float ts = adjoint[2048+pos], trs = adjoint[3072+pos];
                const float total = higher_sum(prob);
                const float mean_r = higher_sum(prob*a);
                const float mean_s = higher_sum(prob*b);
                const float mean_rs = higher_sum(prob*(c+a*b));
                const float delta = higher_sum(prob*t);
                const float delta_r = higher_sum(prob*(a*t+tr));
                const float delta_s = higher_sum(prob*(b*t+ts));
                const float delta_rs = higher_sum(prob*((a*b+c)*t+a*ts+b*tr+trs));
                if (lane==0) {
                    statistics[224+row+r]+=total;
                    statistics[row+r]+=mean_r;
                    statistics[32+row+r]+=mean_s;
                    statistics[64+row+r]+=mean_rs;
                    statistics[96+row+r]+=delta;
                    statistics[128+row+r]+=delta_r;
                    statistics[160+row+r]+=delta_s;
                    statistics[192+row+r]+=delta_rs;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x<32) {
            const int r = threadIdx.x;
            const float total = statistics[224+r];
            const float inv_sum = total>0 ? 1.f/total : 0.f;
            statistics[224+r] = inv_sum;
            #pragma unroll
            for (int i=0; i<7; i++) statistics[i*32+r] *= inv_sum;
            const float mean_r = statistics[r], mean_s = statistics[32+r];
            const float mean_rs = statistics[64+r]-mean_r*mean_s;
            const float delta = statistics[96+r];
            statistics[64+r] = mean_rs;
            statistics[192+r]-=mean_r*statistics[160+r]+mean_s*statistics[128+r]
                -(mean_r*mean_s-mean_rs)*delta;
            statistics[128+r]-=mean_r*delta;
            statistics[160+r]-=mean_s*delta;
        }
        __syncthreads();
        for (int64_t kb = begin; kb<end; kb+=32) {
            if (end-begin>32) {
                triple_pairs(p, cache, probability, adjoint, begin, end, qb, kb, segment, head);
                #pragma unroll
                for (int r = 0; r<4; r++) {
                    int pos = (row+r)*32+lane;
                    probability[pos] *= statistics[224+row+r];
                    float prob = probability[pos], a = probability[1024+pos]-statistics[row+r],
                        b = probability[2048+pos]-statistics[32+row+r];
                    probability[1024+pos] = prob*a;
                    probability[2048+pos] = prob*b;
                    probability[3072+pos] = prob*(a*b+probability[3072+pos]-statistics[64+row+r]);
                }
            }
            #pragma unroll
            for (int r = 0; r<4; r++) {
                int pos = (row+r)*32+lane;
                float t = adjoint[pos]-statistics[96+row+r], tr = adjoint[1024+pos]-statistics[128+row+r];
                float ts = adjoint[2048+pos]-statistics[160+row+r], trs = adjoint[3072+pos]-statistics[192+row+r];
                float prob = probability[pos], pr = probability[1024+pos], ps = probability[2048+pos],
                    prs = probability[3072+pos];
                adjoint[pos] = prob*t;
                adjoint[1024+pos] = pr*t+prob*tr;
                adjoint[2048+pos] = ps*t+prob*ts;
                adjoint[3072+pos] = prs*t+pr*ts+ps*tr+prob*trs;
            }
            __syncthreads();
            triple_contract<2, 7, true, 4, 13, -1, Split>(p, probability, cache, begin, end, qb, kb, head);
            triple_contract<4, -1, false, 2, 11, 7, Split>(p, probability, cache, begin, end, qb, kb, head);
            // Both value contractions read this cache; synchronize before Q/K reuses it.
            if constexpr (VALUE_DIM<=FEATURE_WIDTH && (NEEDS & ((1<<2)|(1<<4)|(1<<7))))
                __syncthreads();
            triple_contract<0, 5, false, 1, 10, 6, Split>(p, adjoint, cache, begin, end, qb, kb, head);
            triple_contract<1, 6, true, 0, 9, 5, Split>(p, adjoint, cache, begin, end, qb, kb, head);
            higher_bias_gradient<3, 3, Split>(p, adjoint, segment, head, begin, end, qb, kb);
            higher_bias_gradient<8, 2, Split>(p, adjoint, segment, head, begin, end, qb, kb);
            __syncthreads();
        }
    }
}

template<bool Split = false, bool InitializeOnly = false>
__global__ __launch_bounds__(LOW_PRECISION && !Split && DIM<=128 && VALUE_DIM<=128 ? 128:256, LOW_PRECISION ? 2:(FEATURE_WIDTH<=32 ? 4:2))
void attention_triple_backward_kernel(const __grid_constant__ AttentionParams p) {
    triple_backward_segment<Split,InitializeOnly>(p,blockIdx.x,blockIdx.y);
}

#if LOW_PRECISION
__device__ __forceinline__ void triple_convert_segment(AttentionParams const& p,int segment,int head) {
    int64_t begin=p.offset(segment),end=p.offset(segment+1);
    if(end-begin<=64) return;
    #pragma unroll
    for(int output=0;output<9;++output) {
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
__global__ void higher_convert_long(const __grid_constant__ AttentionParams p) {
    triple_convert_segment(p,blockIdx.x,blockIdx.y);
}

__global__ __launch_bounds__(256,2) void triple_small_long(
    const __grid_constant__ AttentionParams partial,const __grid_constant__ AttentionParams conversion) {
    int64_t begin=partial.offset(blockIdx.x),end=partial.offset(blockIdx.x+1);
    if(end-begin<=64) return;
    triple_backward_segment<false,false,true>(partial,blockIdx.x,blockIdx.y);
    __syncthreads();
    triple_convert_segment(conversion,blockIdx.x,blockIdx.y);
}

#endif

void attention_triple_backward(std::vector<c10::optional<at::Tensor>> inputs, at::Tensor rowptr, at::Tensor state,
    std::vector<c10::optional<at::Tensor>> outputs, double scale, std::vector<c10::optional<at::Tensor>> partials) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p = make_params(inputs, outputs, rowptr, state, scale);
#if LOW_PRECISION
    auto partial = make_params(inputs, partials, rowptr, state, scale);
#if COMBINE_SEGMENTS
    attention_triple_backward_kernel<false><<<dim3(p.groups,inputs[0]->size(1)),128,TripleNative<64>::Bytes,
        at::cuda::getCurrentCUDAStream()>>>(p);
    auto small_conversion=make_params(partials,outputs,rowptr,state,scale);
    triple_small_long<<<dim3(p.groups,inputs[0]->size(1)),256,TRIPLE_LONG_BYTES,
        at::cuda::getCurrentCUDAStream()>>>(partial,small_conversion);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
#endif
    attention_triple_backward_kernel<false,true><<<dim3(p.groups,inputs[0]->size(1)),128,0,at::cuda::getCurrentCUDAStream()>>>(partial);
#else
    auto const& partial=p;
#endif
#if LOW_PRECISION
    constexpr int bytes = COMBINE_SEGMENTS ? TripleNative<64>::Bytes:(DIM<=128 && VALUE_DIM<=128 ? TripleNative<32>::Bytes:TRIPLE_LONG_BYTES);
#else
    constexpr int bytes = MAIN_SHARED_BYTES;
#endif
    attention_triple_backward_kernel<false><<<dim3(p.groups, inputs[0]->size(1)), LOW_PRECISION && DIM<=128 && VALUE_DIM<=128 ? 128:256, bytes,
        at::cuda::getCurrentCUDAStream()>>>(p);
    attention_triple_backward_kernel<true><<<dim3(p.groups*8,inputs[0]->size(1)),256,TRIPLE_LONG_BYTES,
        at::cuda::getCurrentCUDAStream()>>>(partial);
#if !COMBINE_SEGMENTS
    attention_triple_backward_segment<48><<<dim3(p.groups,inputs[0]->size(1)),RESIDENT_THREADS<48>,
        RESIDENT_BYTES<48>,at::cuda::getCurrentCUDAStream()>>>(p);
    attention_triple_backward_segment<64><<<dim3(p.groups,inputs[0]->size(1)),RESIDENT_THREADS<64>,
        RESIDENT_BYTES<64>,at::cuda::getCurrentCUDAStream()>>>(p);
#endif
#if LOW_PRECISION
    auto conversion=make_params(partials,outputs,rowptr,state,scale);
    higher_convert_long<<<dim3(p.groups,inputs[0]->size(1)),256,0,at::cuda::getCurrentCUDAStream()>>>(conversion);
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &attention_triple_backward);
    m.def("configure", []() {
#if LOW_PRECISION && COMBINE_SEGMENTS
        C10_CUDA_CHECK(cudaFuncSetAttribute(triple_small_long,
            cudaFuncAttributeMaxDynamicSharedMemorySize,TRIPLE_LONG_BYTES));
#endif
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,TRIPLE_LONG_BYTES));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_segment<48>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,RESIDENT_BYTES<48>));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_segment<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,RESIDENT_BYTES<64>));
        if constexpr (CACHE_PLANES==5)
            C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_kernel<false>,
                cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
#if LOW_PRECISION
            COMBINE_SEGMENTS ? TripleNative<64>::Bytes:(DIM<=128 && VALUE_DIM<=128 ? TripleNative<32>::Bytes:TRIPLE_LONG_BYTES)
#else
            MAIN_SHARED_BYTES
#endif
            ));
    });
}
'''

def attention_triple_backward(inputs, rowptr, state, config, needs):
    results = tuple(
        torch.empty(value.shape, device=value.device, dtype=value.dtype)
        if need and value is not None else None
        for value, need in zip(inputs[:9], needs)
    )
    if any(value is not None and value.numel() for value in results):
        present = sum(1 << i for i, value in enumerate(inputs) if value is not None)
        wanted = sum(1 << i for i, value in enumerate(results) if value is not None)
        # A small grid cannot occupy more than two CTAs per SM; combine resident paths to avoid empty launches.
        device = inputs[0].device
        arch = get_cuda_arch(device)
        properties = get_device_properties(device)
        combine_segments = (arch == "90"
                            and (rowptr.numel()-1)*config[2] <= 2*properties.multi_processor_count)
        if combine_segments and inputs[0].dtype != torch.float32:
            native_shared = 640*max(config[:2])+44032
            combine_segments = (max(config[:2]) <= 128
                                and 2*(native_shared+1024) <= properties.shared_memory_per_multiprocessor)
        key = (combine_segments, config[0], config[1], config[2], config[4], config[6], present, wanted, inputs[0].device.index, inputs[0].dtype)
        kernel = KERNELS.get(key)
        if kernel is None:
            # Keep two CTAs within Hopper's 228 KiB shared-memory budget.
            feature_limit = 64
            if arch == "90":
                feature_limit = (((228 * 1024 // 2 - 1024) // 4 - 8448) // (6 * 33) // 32) * 32
            segment_limit = min(max(config[:2]), 96) if arch == "90" else 32
            segment_feature64 = 8 if arch == "80" else 32
            defines = (f"#define COMBINE_SEGMENTS {int(combine_segments)}\n#define RESIDENT_BUDGET {properties.shared_memory_per_block_optin-1024}\n"
                       f"#define FEATURE_LIMIT {feature_limit}\n#define SEGMENT_LIMIT {segment_limit}\n#define SEGMENT_FEATURE64 {segment_feature64}\n"
                       f"#define DIM {config[0]}\n#define VALUE_DIM {config[1]}\n"
                       f"#define HEADS {config[2]}\n#define CAUSAL {int(config[4])}\n#define BIAS_KIND {config[6]}\n"
                       f"#define PRESENT {present}\n#define NEEDS {wanted}\n")
            module = load_cuda_extension(
                "attention_triple_backward", dtype_defines(inputs[0].dtype) + CUDA_SOURCE + defines + FEATURE_SOURCE + SOFTMAX_SOURCE + NATIVE_RESIDENT + SOURCE,
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
