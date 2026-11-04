"""Differentiate the fused backward and both output adjoints once."""

from functools import lru_cache

import torch

from .backward import BACKWARD_SOURCE
from .utils import finish_gradients, input_layout, load_kernel


GRADIENT_SOURCE = r"""
constexpr unsigned GradientComponents=2u;
// dS_r = P_r * (D - delta) + P * (D_r - delta_r).
// The shared CUTLASS products apply dQ_r = dS_r K + dS K_r,
// dK_r = dS_r^T Q + dS^T Q_r, and dV_r = P_r^T G + P^T G_r.
template<int N>
__device__ __forceinline__ Differential<N> softmax_gradient(Differential<N> probability,Differential<N> dot,Differential<N> mean) {
    Differential<N> value;
    float centered=dot.x[0]-mean.x[0];
    value.x[0]=probability.x[0]*centered;
    value.x[1]=fmaf(probability.x[1],centered,probability.x[0]*(dot.x[1]-mean.x[1]));
    return value;
}
// The incoming direction differentiates J(x)^T g; the output adjoints receive
// the directional forward output, including the envelope product rule.
template<int N>
__device__ __forceinline__ void gradient(Params const& p,int slot,Differential<N> value,
    int64_t i,int64_t j=0,int64_t k=0,int64_t l=0) {
    auto const& output=p.out[slot];
    if(NEEDS&(1u<<slot)) {
        if(slot==4 || (QUERY_WORKERS>1 && (slot==1 || slot==2 || (slot==3 && BIAS_KIND==1)))) output.atomic_add(output.index(i,j,k,l),value.x[1]);
        else output.accumulate(output.index(i,j,k,l),value.x[1]);
    }
}
template<int N>
__device__ __forceinline__ void upstream_gradient(Params const& p,int slot,Differential<N> value,int64_t i,int h,int c) {
    if(NEEDS&(1u<<slot)) {
        if(QUERY_WORKERS>1 && slot==5) p.out[slot].atomic_add(p.out[slot].index(i,h,c),value.x[1]);
        else p.out[slot].accumulate(p.out[slot].index(i,h,c),value.x[1]);
    }
}
"""


SOURCE = r"""
void run(std::vector<std::optional<at::Tensor>> inputs,at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,double scale
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
         ,at::Tensor summary
#endif
         ) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if(csr.numel()>1) {
        constexpr int bytes=sizeof(BackwardStorage<2>);
        static thread_local int configured_device=-1;
        int device=inputs[0]->get_device();
        if(configured_device!=device) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(backward_kernel<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
            configured_device=device;
        }
#if defined(CONV_NATIVE_HIGHER_SUMMARY)
        auto p=params(inputs,outputs,csr,scale);
        auto projected=view(summary);
        node_projection_kernel<2><<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p,projected);
        backward_kernel<2><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,bytes,at::cuda::getCurrentCUDAStream()>>>(p,projected);
        if constexpr(NEEDS&548u)
            node_value_kernel<2><<<dim3(csr.numel()-1,HEADS),128,0,at::cuda::getCurrentCUDAStream()>>>(p,projected);
#else
        backward_kernel<2><<<dim3(csr.numel()-1,HEADS,QUERY_WORKERS),128,bytes,at::cuda::getCurrentCUDAStream()>>>(params(inputs,outputs,csr,scale));
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
"""


@lru_cache(maxsize=128)
def get_double_backward_kernel(config, device, dtype, needed, edge_grad, layout, query_workers):
    defines = (
        f"#define QUERY_WORKERS {query_workers}\n"
        f"#define NEEDS {needed}u\n"
        f"#define EDGE_GRAD {int(edge_grad)}\n"
    )
    if dtype == torch.float32 and not edge_grad:
        defines += "#define CONV_NATIVE_HIGHER_SUMMARY 1\n"
    source = (
        defines
        + GRADIENT_SOURCE
        + BACKWARD_SOURCE
        + SOURCE
    )
    return load_kernel("attention_conv_double_backward", source, config, device.index, dtype, layout).run


def attention_conv_double_backward(inputs, csr, config, needs):
    native_node = inputs[0].dtype == torch.float32 and inputs[6] is None
    outputs = tuple(
        (torch.empty if native_node and i in (2, 5) else torch.zeros)(
            x.shape, device=x.device, dtype=torch.float32
        ) if x is not None and need else None
        for i, (x, need) in enumerate(zip(inputs[:7], needs))
    )
    needed = sum(1 << i for i, output in enumerate(outputs) if output is not None)
    query_workers = 4 if inputs[0].shape[0] > 32 * (csr.numel() - 1) else 1
    kernel = get_double_backward_kernel(
        config, inputs[0].device, inputs[0].dtype, needed,
        inputs[6] is not None, input_layout(inputs), query_workers,
    )
    if native_node:
        summary = torch.empty(
            (*inputs[0].shape[:2], 4), device=inputs[0].device, dtype=torch.float32
        )
        kernel(inputs, csr, outputs, config[3], summary)
    else:
        kernel(inputs, csr, outputs, config[3])
    return finish_gradients(outputs, inputs)
