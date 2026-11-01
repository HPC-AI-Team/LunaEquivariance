"""Complete second backward using FP32 softmax moments and CUTLASS products."""

import torch

from .utils import (
    CUDA_SOURCE, GEMM_SOURCE, ATTENTION_SOURCE, PACKED_GEMM_SOURCE, BIAS_SOURCE, dtype_defines,
)
from ....utils.extension import get_cuda_arch, load_cuda_extension

KERNELS = {}

DOUBLE_BACKWARD_SOURCE = r"""
constexpr int AttentionDoubleSlots=2;
constexpr int AttentionDoubleColumns=LOW_PRECISION ? 32:64;
using AttentionDoubleOutputStorage=packed_pair::GemmStorage<packed_pair::DoubleFeatureElement,
    packed_pair::DoubleFeatureElement,packed_pair::PairElement,true,AttentionDoubleColumns>;
struct AttentionDoubleShared {
    union GemmStorage {
        packed_pair::DoubleStorage packed;
        AttentionDoubleOutputStorage terminal;
#if !LOW_PRECISION
        float result[32*AttentionDoubleColumns];
#endif
    } gemm;
    float probability[AttentionDoubleSlots][32*32],direction[AttentionDoubleSlots][32*32];
    float value[AttentionDoubleSlots][32*32],value_direction[AttentionDoubleSlots][32*32];
    float statistics[4*32];
};

__device__ __forceinline__ void double_backward_packed_pairs(AttentionParams const& p,
    AttentionDoubleShared::GemmStorage& work,
    float* probability,float* direction,float* value,float* value_direction,
    int64_t begin,int64_t end,int64_t qb,int64_t kb,int segment,int head) {
    auto& packed=work.packed;
    packed_pair::gemm(packed,probability,direction,nullptr,nullptr,DIM,
#if LOW_PRECISION
        AttentionPackedFeature<0,5,-1,false,DIM>{p,qb,end,head,0},
        AttentionPackedFeature<1,6,-1,true,DIM>{p,kb,end,head,0}
#else
        [&](int row,int channel) {
            return attention_packed_feature<0,5,-1>(p,qb+row,end,head,channel,DIM);
        },
        [&](int channel,int column) {
            return attention_packed_feature<1,6,-1>(p,kb+column,end,head,channel,DIM);
        }
#endif
    );
    for(int i=threadIdx.x;i<32*32;i+=AttentionThreads) {
        int row=i/32,column=i%32;
        bool valid=qb+row<end && kb+column<end && (!CAUSAL || kb+column<=qb+row);
        float prob=0.f,first=0.f;
        if(valid && p.state.at(qb+row,head,1)>0.f) {
            float score=probability[i]*p.scale+attention_bias<3>(p,segment,head,begin,qb+row,kb+column);
            prob=__expf(score-p.state.at(qb+row,head,0))*p.state.at(qb+row,head,1);
            first=direction[i]*p.scale+attention_bias<8>(p,segment,head,begin,qb+row,kb+column);
        }
        probability[i]=prob;
        direction[i]=first;
    }
    __syncthreads();
    packed_pair::gemm(packed,value,value_direction,nullptr,nullptr,VALUE_DIM,
#if LOW_PRECISION
        AttentionPackedFeature<4,-1,-1,false,VALUE_DIM>{p,qb,end,head,0},
        AttentionPackedFeature<2,7,-1,true,VALUE_DIM>{p,kb,end,head,0}
#else
        [&](int row,int channel) {
            return attention_packed_feature<4,-1,-1>(p,qb+row,end,head,channel,VALUE_DIM);
        },
        [&](int channel,int column) {
            return attention_packed_feature<2,7,-1>(p,kb+column,end,head,channel,VALUE_DIM);
        }
#endif
    );
}

template<int Target,int Base,int Direction,bool Transpose=false
#if LOW_PRECISION
    ,int Channels=DIM
#endif
>
__device__ __forceinline__ void double_backward_packed_output(AttentionParams const& p,
    AttentionDoubleShared::GemmStorage& work,float const* base,float const* direction,
    int64_t qb,int64_t kb,int64_t end,int head,int channels,float scale=1.f) {
    if constexpr(NEEDS&(1<<Target)) {
        auto& packed=work.terminal;
        const int64_t target_base=Transpose ? kb:qb;
        const int64_t source_base=Transpose ? qb:kb;
#if LOW_PRECISION
        packed_pair::prepare_coefficients(packed,AttentionPackedCoefficient<Transpose,32>{{base,direction,nullptr,nullptr}},
            (end-source_base<32 ? int(end-source_base):32));
        float* result=packed.result+packed.RetainedElements;
#else
        float* result=work.result;
#endif
        for(int channel=0;channel<channels;channel+=AttentionDoubleColumns) {
            packed_pair::gemm<AttentionDoubleColumns>(packed,nullptr,result,nullptr,nullptr,(end-source_base<32 ? int(end-source_base):32),
#if LOW_PRECISION
                AttentionPackedCoefficient<Transpose,32>{{base,direction,nullptr,nullptr}},
                AttentionPackedFeature<Base,Direction,-1,false,Channels>{p,source_base,end,head,channel}
#else
                [&](int row,int k) {
                    int index=Transpose ? k*32+row:row*32+k;
                    return packed_pair::FeatureElement(base[index],direction[index],0.f);
                },
                [&](int k,int column) {
                    if(channel+column>=channels) return packed_pair::FeatureElement();
                    return attention_packed_feature<Base,Direction,-1>(
                        p,source_base+k,end,head,channel+column,channels);
                }
#endif
            );
            for(int i=threadIdx.x;i<32*AttentionDoubleColumns;i+=AttentionThreads) {
                int64_t row=target_base+i/AttentionDoubleColumns;
                int column=channel+i%AttentionDoubleColumns;
                if(row<end && column<channels) {
                    float* output=reinterpret_cast<float*>(p.outputs[Target].data)+
                        (row*HEADS+head)*channels+column;
                    if constexpr(Transpose) atomicAdd(output,scale*result[i]);
                    else if(kb+32>=end) *output=scale*result[i];
                    else *output+=scale*result[i];
                }
            }
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(AttentionThreads) void attention_double_backward_kernel(
    const __grid_constant__ AttentionParams p) {
    int segment=blockIdx.x,head=blockIdx.y;
    int64_t begin=p.offset(segment),end=p.offset(segment+1);
    extern __shared__ char storage[];
    auto& shared=*reinterpret_cast<AttentionDoubleShared*>(storage);
    auto& gemm=shared.gemm;
    auto& statistics=shared.statistics;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int64_t qb=begin+int64_t(blockIdx.z)*32;qb<end;qb+=4*32) {
        float norm[32/(AttentionThreads/32)]={},mean[32/(AttentionThreads/32)]={},
            delta[32/(AttentionThreads/32)]={},delta_direction[32/(AttentionThreads/32)]={};
        int64_t last_key=begin;
        for(int64_t kb=begin;kb<end;kb+=32) {
            const int slot=int((kb-begin)/32)%AttentionDoubleSlots;
            auto& probability=shared.probability[slot];
            auto& direction=shared.direction[slot];
            auto& value=shared.value[slot];
            auto& value_direction=shared.value_direction[slot];
            double_backward_packed_pairs(p,gemm,probability,direction,value,value_direction,begin,end,qb,kb,segment,head);
            last_key=kb;
            #pragma unroll
            for(int r=0;r<32/(AttentionThreads/32);++r) {
                for(int column=lane;column<32;column+=32) {
                    int pos=(warp+r*(AttentionThreads/32))*32+column;
                    float prob=probability[pos],a=direction[pos],t=value[pos],tr=value_direction[pos];
                    norm[r]+=prob;
                    mean[r]+=prob*a;
                    delta[r]+=prob*t;
                    delta_direction[r]+=prob*(a*t+tr);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<32/(AttentionThreads/32);++r) {
            float n=attention_sum(norm[r]),m=attention_sum(mean[r]);
            float d=attention_sum(delta[r]),dr=attention_sum(delta_direction[r]);
            float inverse=n>0.f ? 1.f/n:0.f;
            if(lane==0) {
                int row=warp+r*(AttentionThreads/32);
                statistics[row]=inverse;
                statistics[32+row]=m*inverse;
                statistics[2*32+row]=d*inverse;
                statistics[3*32+row]=dr*inverse-m*d*inverse*inverse;
            }
        }
        __syncthreads();
        for(int64_t kb=last_key;kb>=begin;kb-=32) {
            const int slot=int((kb-begin)/32)%AttentionDoubleSlots;
            auto& probability=shared.probability[slot];
            auto& direction=shared.direction[slot];
            auto& value=shared.value[slot];
            auto& value_direction=shared.value_direction[slot];
            if(last_key-kb>=AttentionDoubleSlots*32) {
                double_backward_packed_pairs(p,gemm,probability,direction,value,value_direction,begin,end,qb,kb,segment,head);
            }
            for(int i=threadIdx.x;i<32*32;i+=AttentionThreads) {
                int row=i/32;
                float prob=probability[i]*statistics[row];
                float pr=prob*(direction[i]-statistics[32+row]);
                float t=value[i]-statistics[2*32+row],tr=value_direction[i]-statistics[3*32+row];
                probability[i]=prob;
                direction[i]=pr;
                value[i]=prob*t;
                value_direction[i]=pr*t+prob*tr;
            }
            __syncthreads();
            double_backward_packed_output<0,1,6>(p,gemm,value,value_direction,qb,kb,end,head,DIM,p.scale);
            double_backward_packed_output<1,0,5,true>(p,gemm,value,value_direction,qb,kb,end,head,DIM,p.scale);
#if LOW_PRECISION
            double_backward_packed_output<2,4,-1,true,VALUE_DIM>(p,gemm,direction,direction,qb,kb,end,head,VALUE_DIM);
#else
            double_backward_packed_output<2,4,-1,true>(p,gemm,direction,direction,qb,kb,end,head,VALUE_DIM);
#endif
#if LOW_PRECISION
            double_backward_packed_output<4,2,7,false,VALUE_DIM>(p,gemm,probability,direction,qb,kb,end,head,VALUE_DIM);
#else
            double_backward_packed_output<4,2,7>(p,gemm,probability,direction,qb,kb,end,head,VALUE_DIM);
#endif
            attention_bias_output<3>(p,value_direction,segment,head,begin,qb,kb,end);
        }
    }
}

void attention_double_backward(std::vector<std::optional<at::Tensor>> inputs,at::Tensor rowptr,at::Tensor state,
    std::vector<std::optional<at::Tensor>> outputs,double scale,std::vector<std::optional<at::Tensor>> partials) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    auto p=make_params(inputs,partials,rowptr,state,scale);
    if(p.groups) {
        attention_initialize_kernel<5><<<dim3(p.groups,HEADS),AttentionThreads,0,at::cuda::getCurrentCUDAStream()>>>(p);
        attention_double_backward_kernel<<<dim3(p.groups,HEADS,4),AttentionThreads,sizeof(AttentionDoubleShared),at::cuda::getCurrentCUDAStream()>>>(p);
#if LOW_PRECISION
        auto conversion=make_params(partials,outputs,rowptr,state,scale);
        attention_convert_kernel<5><<<dim3(p.groups,HEADS),AttentionThreads,0,at::cuda::getCurrentCUDAStream()>>>(conversion);
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run",&attention_double_backward);
    m.def("configure",[]() {
        C10_CUDA_CHECK(cudaFuncSetAttribute(attention_double_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttentionDoubleShared)));
    });
}
"""


def attention_double_backward(inputs, rowptr, state, config, needs):
    results = tuple(torch.empty(value.shape, device=value.device, dtype=value.dtype)
                    if need and value is not None else None
                    for value, need in zip(inputs[:5], needs))
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
            "attention_double_backward", dtype_defines(inputs[0].dtype) + defines + CUDA_SOURCE
            + GEMM_SOURCE + ATTENTION_SOURCE + PACKED_GEMM_SOURCE + BIAS_SOURCE + DOUBLE_BACKWARD_SOURCE,
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
