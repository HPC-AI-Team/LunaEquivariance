"""Complete third backward from mixed softmax moments and CUTLASS GEMMs."""

import torch

from .utils import (
    CUDA_SOURCE, GEMM_SOURCE, ATTENTION_SOURCE, PACKED_GEMM_SOURCE, BIAS_SOURCE, dtype_defines,
)
from ....utils.extension import get_cuda_arch, load_cuda_extension

KERNELS = {}

TRIPLE_BACKWARD_SOURCE = r"""
// The CUTLASS producer reads four rows and eight columns per warp.
constexpr int AttentionPairStride=36;

struct AttentionTripleShared {
    union GemmStorage {
        union {
            packed_pair::Storage operands;
#if !LOW_PRECISION
            packed_pair::GemmStorage<packed_pair::ValueSecondElement,packed_pair::FeatureElement,
                packed_pair::PairElement> value;
#endif
        } packed;
        struct {
            packed_pair::GemmStorage<packed_pair::PairElement,packed_pair::FeatureElement,
                packed_pair::Terminal4,true> operands;
        } terminal;
    } gemm;
    float probability[2][4][32*AttentionPairStride],value[4][32*AttentionPairStride],statistics[8][32];
};

template<int MixedTarget,int SecondTarget,int Base,int First,int Second,bool Transpose=false,bool SecondOutput=true
#if LOW_PRECISION
    ,int Channels=DIM
#endif
>
__device__ __forceinline__ void triple_backward_packed_output(AttentionParams const& p,
    AttentionTripleShared::GemmStorage& work,float const coefficient[4][32*AttentionPairStride],
    int64_t qb,int64_t kb,int64_t end,int head,int channels,float scale=1.f) {
    if constexpr((NEEDS&(1<<MixedTarget)) || (SecondOutput && (NEEDS&(1<<SecondTarget)))) {
        auto& terminal=work.terminal;
#if LOW_PRECISION
        auto& operands=terminal.operands;
        float* second_result=operands.result+operands.RetainedElements;
#else
        using Feature=typename std::conditional<(First<0),packed_pair::ValueSecondElement,packed_pair::FeatureElement>::type;
        using OutputStorage=packed_pair::GemmStorage<packed_pair::PairElement,Feature,packed_pair::Terminal4,true>;
        static_assert(sizeof(OutputStorage)<=sizeof(terminal.operands));
        auto& operands=*reinterpret_cast<OutputStorage*>(&terminal.operands);
        float* second_result=reinterpret_cast<float*>(&operands);
#endif
        float* mixed_result=second_result+32*AttentionPairStride;
        const int64_t target_base=Transpose ? kb:qb;
        const int64_t source_base=Transpose ? qb:kb;
#if LOW_PRECISION
        packed_pair::prepare_coefficients(operands,
            AttentionPackedCoefficient<Transpose,AttentionPairStride>{{coefficient[0],coefficient[1],coefficient[2],coefficient[3]}},
            (end-source_base<32 ? int(end-source_base):32));
        static_assert(sizeof(operands)>=sizeof(float)*(decltype(terminal.operands)::RetainedElements+2*32*AttentionPairStride));
#endif
        for(int channel=0;channel<channels;channel+=32) {
            packed_pair::gemm<AttentionPairStride>(operands,nullptr,nullptr,SecondOutput && (NEEDS&(1<<SecondTarget)) ? second_result:nullptr,
                (NEEDS&(1<<MixedTarget)) ? mixed_result:nullptr,(end-source_base<32 ? int(end-source_base):32),
#if LOW_PRECISION
                AttentionPackedCoefficient<Transpose,AttentionPairStride>{{coefficient[0],coefficient[1],coefficient[2],coefficient[3]}},
                AttentionPackedFeature<Base,First,Second,false,Channels>{p,source_base,end,head,channel}
#else
                [&](int row,int k) {
                    int index=Transpose ? k*AttentionPairStride+row:row*AttentionPairStride+k;
                    packed_pair::PairElement result;
                    result.value=coefficient[0][index];
                    result.first=coefficient[1][index];
                    result.second=coefficient[2][index];
                    result.mixed=coefficient[3][index];
                    return result;
                },
                [&](int k,int column) {
                    if(channel+column>=channels) return Feature();
                    return Feature(attention_packed_feature<Base,First,Second>(
                        p,source_base+k,end,head,channel+column,channels));
                }
#endif
            );
            for(int i=threadIdx.x;i<32*32;i+=AttentionThreads) {
                const int pos=(i/32)*AttentionPairStride+i%32;
                int64_t row=target_base+i/32;
                int column=channel+i%32;
                if(row<end && column<channels) {
                    if constexpr(NEEDS&(1<<MixedTarget)) {
                        float* target=reinterpret_cast<float*>(p.outputs[MixedTarget].data)+
                            (row*HEADS+head)*channels+column;
                        if constexpr(Transpose) atomicAdd(target,scale*mixed_result[pos]);
                        else if(kb+32>=end) *target=scale*mixed_result[pos];
                        else *target+=scale*mixed_result[pos];
                    }
                    if constexpr(SecondOutput && (NEEDS&(1<<SecondTarget))) {
                        float* target=reinterpret_cast<float*>(p.outputs[SecondTarget].data)+
                            (row*HEADS+head)*channels+column;
                        if constexpr(Transpose) atomicAdd(target,scale*second_result[pos]);
                        else if(kb+32>=end) *target=scale*second_result[pos];
                        else *target+=scale*second_result[pos];
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void triple_backward_pairs(AttentionParams const& p,
    AttentionTripleShared::GemmStorage& work,
    float probability[4][32*AttentionPairStride],float value[4][32*AttentionPairStride],
    int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head,
    bool compute_probability=true,bool compute_value=true) {
    auto& packed=work.packed;
    if(compute_probability) {
        packed_pair::gemm<AttentionPairStride>(packed.operands,probability[0],probability[1],probability[2],probability[3],DIM,
#if LOW_PRECISION
            AttentionPackedFeature<0,5,9,false,DIM>{p,qb,end,head,0},
            AttentionPackedFeature<1,6,10,true,DIM>{p,kb,end,head,0}
#else
            [&](int row,int channel) {
                return attention_packed_feature<0,5,9>(p,qb+row,end,head,channel,DIM);
            },
            [&](int channel,int column) {
                return attention_packed_feature<1,6,10>(p,kb+column,end,head,channel,DIM);
            }
#endif
        );
        for(int i=threadIdx.x;i<32*32;i+=AttentionThreads) {
            const int pos=(i/32)*AttentionPairStride+i%32;
            int row=i/32,column=i%32;
            bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
            float score_value=probability[0][pos],first_value=probability[1][pos];
            float second_value=probability[2][pos],mixed_value=probability[3][pos];
            float prob=0.f,first=0.f;
            if(valid && p.state.at(qb+row,head,1)>0.f) {
                float score=score_value*p.scale+attention_bias<3>(p,segment,head,begin,qb+row,kb+column);
                prob=__expf(score-p.state.at(qb+row,head,0))*p.state.at(qb+row,head,1);
                first=first_value*p.scale+attention_bias<8>(p,segment,head,begin,qb+row,kb+column);
            }
            probability[0][pos]=prob;
            probability[1][pos]=first;
            probability[2][pos]=valid ? second_value*p.scale+
                attention_bias<12>(p,segment,head,begin,qb+row,kb+column):0.f;
            probability[3][pos]=valid ? mixed_value*p.scale:0.f;
        }
        __syncthreads();
    }
    if(compute_value) {
        packed_pair::gemm<AttentionPairStride>(
#if LOW_PRECISION
            packed.operands,
#else
            packed.value,
#endif
            value[0],value[1],value[2],value[3],VALUE_DIM,
#if LOW_PRECISION
            AttentionPackedFeature<4,-1,13,false,VALUE_DIM>{p,qb,end,head,0},
            AttentionPackedFeature<2,7,11,true,VALUE_DIM>{p,kb,end,head,0}
#else
            [&](int row,int channel) {
                return packed_pair::ValueSecondElement(
                    attention_packed_feature<4,-1,13>(p,qb+row,end,head,channel,VALUE_DIM));
            },
            [&](int channel,int column) {
                return attention_packed_feature<2,7,11>(p,kb+column,end,head,channel,VALUE_DIM);
            }
#endif
        );
    }
}

__global__ __launch_bounds__(AttentionThreads) void attention_triple_backward_kernel(
    const __grid_constant__ AttentionParams p) {
    int segment=blockIdx.x,head=blockIdx.y;
    int64_t begin=p.offset(segment),end=p.offset(segment+1);
    extern __shared__ char storage[];
    auto& shared=*reinterpret_cast<AttentionTripleShared*>(storage);
    auto& gemm=shared.gemm;
    auto& statistics=shared.statistics;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int64_t qb=begin+int64_t(blockIdx.z)*32;qb<end;qb+=4*32) {
        for(int i=threadIdx.x;i<8*32;i+=AttentionThreads) statistics[i/32][i%32]=0.f;
        __syncthreads();
        int64_t last_key=begin;
        for(int64_t kb=begin;kb<end;kb+=32) {
            int slot=int((kb-begin)/32)%2;
            auto& probability=shared.probability[slot];
            auto& value=shared.value;
            triple_backward_pairs(p,gemm,probability,value,begin,end,qb,kb,segment,head);
            last_key=kb;
            for(int row=warp;row<32;row+=AttentionThreads/32) {
                float m[8]={};
                for(int column=lane;column<32;column+=32) {
                    int pos=row*AttentionPairStride+column;
                    float prob=probability[0][pos],a=probability[1][pos],b=probability[2][pos];
                    float c=probability[3][pos]+a*b;
                    float t=value[0][pos],tr=value[1][pos],ts=value[2][pos],trs=value[3][pos];
                    m[0]+=prob;
                    m[1]+=prob*a;
                    m[2]+=prob*b;
                    m[3]+=prob*c;
                    m[4]+=prob*t;
                    m[5]+=prob*(a*t+tr);
                    m[6]+=prob*(b*t+ts);
                    m[7]+=prob*(c*t+a*ts+b*tr+trs);
                }
                #pragma unroll
                for(int i=0;i<8;++i) {
                    float sum=attention_sum(m[i]);
                    if(lane==0) statistics[i][row]+=sum;
                }
            }
            __syncthreads();
        }
        if(threadIdx.x<32) {
            int row=threadIdx.x;
            float m[8];
            #pragma unroll
            for(int i=0;i<8;++i) m[i]=statistics[i][row];
            float inverse=m[0]>0.f ? 1.f/m[0]:0.f;
            #pragma unroll
            for(int i=1;i<8;++i) m[i]*=inverse;
            statistics[0][row]=inverse;
            statistics[1][row]=m[1];
            statistics[2][row]=m[2];
            statistics[3][row]=m[3]-m[1]*m[2];
            statistics[4][row]=m[4];
            statistics[5][row]=m[5]-m[1]*m[4];
            statistics[6][row]=m[6]-m[2]*m[4];
            statistics[7][row]=m[7]-m[1]*m[6]-m[2]*m[5]+(2.f*m[1]*m[2]-m[3])*m[4];
        }
        __syncthreads();
        for(int64_t kb=last_key;kb>=begin;kb-=32) {
            int slot=int((kb-begin)/32)%2;
            auto& probability=shared.probability[slot];
            auto& value=shared.value;
            triple_backward_pairs(p,gemm,probability,value,begin,end,qb,kb,segment,head,
                last_key-kb>=2*32,kb!=last_key);
            for(int i=threadIdx.x;i<32*32;i+=AttentionThreads) {
                const int pos=(i/32)*AttentionPairStride+i%32;
                int row=i/32;
                float prob=probability[0][pos]*statistics[0][row];
                float a=probability[1][pos]-statistics[1][row];
                float b=probability[2][pos]-statistics[2][row];
                float pr=prob*a,ps=prob*b;
                float prs=prob*(a*b+probability[3][pos]-statistics[3][row]);
                float t=value[0][pos]-statistics[4][row];
                float tr=value[1][pos]-statistics[5][row];
                float ts=value[2][pos]-statistics[6][row];
                float trs=value[3][pos]-statistics[7][row];
                probability[0][pos]=prob;
                probability[1][pos]=pr;
                probability[2][pos]=ps;
                probability[3][pos]=prs;
                value[0][pos]=prob*t;
                value[1][pos]=pr*t+prob*tr;
                value[2][pos]=ps*t+prob*ts;
                value[3][pos]=prs*t+pr*ts+ps*tr+prob*trs;
            }
            __syncthreads();
            triple_backward_packed_output<0,5,1,6,10>(p,gemm,value,qb,kb,end,head,DIM,p.scale);
            triple_backward_packed_output<1,6,0,5,9,true>(p,gemm,value,qb,kb,end,head,DIM,p.scale);
#if LOW_PRECISION
            triple_backward_packed_output<2,7,4,-1,13,true,true,VALUE_DIM>(p,gemm,probability,qb,kb,end,head,VALUE_DIM);
#else
            triple_backward_packed_output<2,7,4,-1,13,true>(p,gemm,probability,qb,kb,end,head,VALUE_DIM);
#endif
#if LOW_PRECISION
            triple_backward_packed_output<4,4,2,7,11,false,false,VALUE_DIM>(p,gemm,probability,qb,kb,end,head,VALUE_DIM);
#else
            triple_backward_packed_output<4,4,2,7,11,false,false>(p,gemm,probability,qb,kb,end,head,VALUE_DIM);
#endif
            attention_bias_output<3,AttentionPairStride>(p,value[3],segment,head,begin,qb,kb,end);
            attention_bias_output<8,AttentionPairStride>(p,value[2],segment,head,begin,qb,kb,end);
        }
    }
}

void attention_triple_backward(std::vector<std::optional<at::Tensor>> inputs,at::Tensor rowptr,at::Tensor state,
    std::vector<std::optional<at::Tensor>> outputs,double scale,std::vector<std::optional<at::Tensor>> partials) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p=make_params(inputs,partials,rowptr,state,scale);
    if(p.groups) {
        attention_initialize_kernel<9><<<dim3(p.groups,HEADS),AttentionThreads,0,at::cuda::getCurrentCUDAStream()>>>(p);
        attention_triple_backward_kernel<<<dim3(p.groups,HEADS,4),AttentionThreads,sizeof(AttentionTripleShared),at::cuda::getCurrentCUDAStream()>>>(p);
#if LOW_PRECISION
        auto conversion=make_params(partials,outputs,rowptr,state,scale);
        attention_convert_kernel<9><<<dim3(p.groups,HEADS),AttentionThreads,0,at::cuda::getCurrentCUDAStream()>>>(conversion);
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run",&attention_triple_backward);
    m.def("configure",[]() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_triple_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttentionTripleShared)));
    });
}
"""


def attention_triple_backward(inputs, rowptr, state, config, needs):
    results = tuple(torch.empty(value.shape, device=value.device, dtype=value.dtype)
                    if need and value is not None else None
                    for value, need in zip(inputs[:9], needs))
    if not any(value is not None and value.numel() for value in results):
        return results
    dim, value_dim, heads, scale, causal, bias_kind = config
    wanted = sum(1 << i for i, value in enumerate(results) if value is not None)
    present = sum(1 << i for i, value in enumerate(inputs) if value is not None)
    compact = all(value.is_contiguous() for i, value in enumerate(inputs)
                  if value is not None and i not in (3, 8, 12))
    key = compact, dim, value_dim, heads, causal, bias_kind, wanted, present, inputs[0].device, inputs[0].dtype
    kernel = KERNELS.get(key)
    if kernel is None:
        defines = (f"#define DIM {dim}\n#define VALUE_DIM {value_dim}\n#define HEADS {heads}\n"
                   f"#define COMPACT {int(compact)}\n#define CAUSAL {int(causal)}\n#define BIAS_KIND {bias_kind}\n"
                   f"#define PRESENT {present}\n#define NEEDS {wanted}\n")
        module = load_cuda_extension(
            "attention_triple_backward", dtype_defines(inputs[0].dtype) + defines + CUDA_SOURCE
            + GEMM_SOURCE + ATTENTION_SOURCE + PACKED_GEMM_SOURCE + BIAS_SOURCE + TRIPLE_BACKWARD_SOURCE,
            arch=get_cuda_arch(inputs[0].device), use_fast_math=False, minimal_headers=True,
        )
        with torch.cuda.device(inputs[0].device):
            module.configure()
        kernel = module.run
        KERNELS[key] = kernel
    partials = tuple(torch.empty_like(value, dtype=torch.float32)
                     if value is not None and value.dtype != torch.float32 else value
                     for value in results) if inputs[0].dtype != torch.float32 else results
    kernel(inputs, rowptr, state, results, scale, partials)
    return results
