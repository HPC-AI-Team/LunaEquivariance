"""Third reverse derivatives including both output adjoints and envelope mixtures."""
from functools import lru_cache
import torch
from ....utils.extension import get_device_properties
from .double_backward import DERIVATIVE_SOURCE, LOWP_SOURCE
from .utils import allocate, load_kernel
from .backward import TILED_SOURCE

GRADIENT_SOURCE = r"""


// x: D_s D_r[J(x)^T g], including the incoming upstream direction s_g.
// r: D_s[J(x)^T g]. g: D_s D_r F(x), for both node and raw edge outputs.
template <int Order, int N>
__device__ void gradient(Params const &p, int slot, Differential<N> value, int64_t i, int64_t j = 0,
                         int64_t k = 0, int64_t l = 0) {
    if (p.out[slot].data)
        p.out[slot].atomic_add(p.out[slot].index(i, j, k, l), value.x[3]);
    if (p.out[7 + slot].data)
        p.out[7 + slot].atomic_add(p.out[7 + slot].index(i, j, k, l), value.x[2]);
}
template <int Order, int N>
__device__ __forceinline__ void gradient_unique(Params const &p, int slot, Differential<N> value, int64_t i,
                                int64_t j = 0, int64_t k = 0, int64_t l = 0) {
    if (NEEDS & (1u << (0 + slot))) {
        auto offset = (slot < 2 ? (i * HEADS + j) * DIM + k : slot == 2 ? (i * HEADS + j) * VALUE_DIM + k : slot == 4 ? i : BIAS_KIND == 1 ? i * HEADS + j : p.out[0 + slot].index(i, j, k, l));
        if (slot == 4)
            p.out[0 + slot].accumulate(offset, value.x[3]);
        else
            p.out[0 + slot].store(offset, value.x[3]);
    }
    if (NEEDS & (1u << (7 + slot))) {
        auto offset = (slot < 2 ? (i * HEADS + j) * DIM + k : slot == 2 ? (i * HEADS + j) * VALUE_DIM + k : slot == 4 ? i : BIAS_KIND == 1 ? i * HEADS + j : p.out[7 + slot].index(i, j, k, l));
        if (slot == 4)
            p.out[7 + slot].accumulate(offset, value.x[2]);
        else
            p.out[7 + slot].store(offset, value.x[2]);
    }
}
"""

NODE_TRIPLE_SOURCE = r"""
#ifdef LOWP_BF16
using NodeTripleTile = typename std::conditional<LOWP_BF16,__nv_bfloat16,__half>::type;
constexpr int NODE_TRIPLE_WIDTH = ((DIM>VALUE_DIM ? DIM:VALUE_DIM)+15)/16*16;
constexpr int NODE_TRIPLE_PITCH = NODE_TRIPLE_WIDTH+8;
constexpr int NODE_TRIPLE_PLANE = 32*NODE_TRIPLE_PITCH;
constexpr int NODE_TRIPLE_BYTES = 6*NODE_TRIPLE_PLANE*sizeof(NodeTripleTile)+(8192+256)*sizeof(float);
constexpr bool NODE_TRIPLE_ENABLED = (DIM>32 || VALUE_DIM>32) && DIM<=128 && VALUE_DIM<=128 && !(PRESENT&64) && !(PRESENT&(1u<<18)) && !(NEEDS&64);
__device__ __forceinline__ NodeTripleTile node_triple_cast(float x) {
#if LOWP_BF16
    return __float2bfloat16_rn(x);
#else
    return __float2half_rn(x);
#endif
}

__device__ __forceinline__ float node_triple_float(NodeTripleTile x) {
#if LOWP_BF16
    return __bfloat162float(x);
#else
    return __half2float(x);
#endif
}

template<int Input,int Field,int Slot>
__device__ void node_triple_load(Params const& p,NodeTripleTile* features,int64_t begin,int n,int head) {
    constexpr int channels=Field<2 ? DIM:VALUE_DIM;
    auto const& source=p.in[Input];
    if constexpr((PRESENT&(1u<<Input)) && channels%8==0) {
        if(source.stride[2]==1 && source.stride[0]%8==0 && source.stride[1]%8==0 && (reinterpret_cast<uintptr_t>(source.data)&15)==0) {
            for(int t=threadIdx.x*8;t<NODE_TRIPLE_PLANE;t+=128*8) {
                int row=t/NODE_TRIPLE_PITCH,c=t%NODE_TRIPLE_PITCH;
                uint4 value={0,0,0,0};
                if(row<n && c<channels) value=*reinterpret_cast<uint4 const*>(reinterpret_cast<NodeTripleTile const*>(source.data)+(begin+row)*source.stride[0]+head*source.stride[1]+c);
                *reinterpret_cast<uint4*>(features+Slot*NODE_TRIPLE_PLANE+t)=value;
            }
            return;
        }
    }
    for(int t=threadIdx.x;t<NODE_TRIPLE_PLANE;t+=128) {
        int row=t/NODE_TRIPLE_PITCH,c=t%NODE_TRIPLE_PITCH;
        NodeTripleTile value=node_triple_cast(0.f);
        if constexpr(PRESENT&(1u<<Input)) if(row<n && c<channels)
            value=reinterpret_cast<NodeTripleTile const*>(source.data)[input_offset(p,Input,Field,begin+row,head,c)];
        features[Slot*NODE_TRIPLE_PLANE+t]=value;
    }
}

template<int Left,int Right,int Left1=-1,int Right1=-1>
__device__ void node_triple_score(NodeTripleTile const* features,float* output) {
    using namespace nvcuda;
    int warp=threadIdx.x/32,q=warp/2*16,k=warp%2*16;
    wmma::fragment<wmma::accumulator,16,16,16,float> d;
    wmma::fill_fragment(d,0.f);
    for(int c=0;c<DIM;c+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,NodeTripleTile,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,NodeTripleTile,wmma::col_major> b;
        wmma::load_matrix_sync(a,features+Left*NODE_TRIPLE_PLANE+q*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
        wmma::load_matrix_sync(b,features+Right*NODE_TRIPLE_PLANE+k*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
        wmma::mma_sync(d,a,b,d);
        if constexpr(Left1>=0) {
            wmma::load_matrix_sync(a,features+Left1*NODE_TRIPLE_PLANE+q*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
            wmma::load_matrix_sync(b,features+Right1*NODE_TRIPLE_PLANE+k*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
            wmma::mma_sync(d,a,b,d);
        }
    }
    wmma::store_matrix_sync(output+q*32+k,d,32,wmma::mem_row_major);
}

template<int Target,int C0,int F0,int C1,int F1,int C2=-1,int F2=-1,bool Transpose=false>
__device__ void node_triple_output(Params const& p,NodeTripleTile const* features,NodeTripleTile const* coefficients,float* temporary,
                                 int64_t begin,int n,int head) {
    if constexpr(NEEDS&(1u<<Target)) {
        using namespace nvcuda;
        constexpr int padded=(DIM+15)/16*16;
        constexpr int tiles=2*(padded/16);
        using Layout=typename std::conditional<Transpose,wmma::col_major,wmma::row_major>::type;
        for(int tile=threadIdx.x/32;tile<tiles;tile+=4) {
            int q=tile/(padded/16)*16,c=tile%(padded/16)*16;
            wmma::fragment<wmma::accumulator,16,16,16,float> d;
            wmma::fill_fragment(d,0.f);
            #pragma unroll
            for(int key=0;key<32;key+=16) {
                wmma::fragment<wmma::matrix_a,16,16,16,NodeTripleTile,Layout> a;
                wmma::fragment<wmma::matrix_b,16,16,16,NodeTripleTile,wmma::row_major> b;
                int offset=Transpose ? key*40+q:q*40+key;
                wmma::load_matrix_sync(a,coefficients+C0*1280+offset,40);
                wmma::load_matrix_sync(b,features+F0*NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
                wmma::mma_sync(d,a,b,d);
                wmma::load_matrix_sync(a,coefficients+C1*1280+offset,40);
                wmma::load_matrix_sync(b,features+F1*NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
                wmma::mma_sync(d,a,b,d);
                if constexpr(C2>=0) {
                    wmma::load_matrix_sync(a,coefficients+C2*1280+offset,40);
                    wmma::load_matrix_sync(b,features+F2*NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c,NODE_TRIPLE_PITCH);
                    wmma::mma_sync(d,a,b,d);
                }
            }
            wmma::store_matrix_sync(temporary+q*padded+c,d,padded,wmma::mem_row_major);
        }
        __syncthreads();
        for(int t=threadIdx.x;t<n*DIM;t+=128) {
            int row=t/DIM,c=t%DIM;
            p.out[Target].store((int64_t(begin+row)*HEADS+head)*DIM+c,temporary[row*padded+c]*p.scale);
        }
        __syncthreads();
    }
}

// Each CTA owns the atom across all heads. Envelope sums stay FP32 until
// both requested envelope derivatives are stored once after the final head.
__device__ void node_triple_derivative(Params const& p,int64_t atom,int64_t begin,int n) {
    using F=Differential<4>;
    extern __shared__ float storage[];
    auto* features=reinterpret_cast<NodeTripleTile*>(storage);
    float* work=reinterpret_cast<float*>(features+6*NODE_TRIPLE_PLANE);
    F* full_dot=reinterpret_cast<F*>(work+8192);
    F* alpha=full_dot+32;
    int lane=threadIdx.x%32,row=threadIdx.x/32;
    float envelope_third[8]={},envelope_direction[8]={};
    for(int h=0;h<HEADS;++h) {
        node_triple_load<0,0,0>(p,features,begin,n,h);
        node_triple_load<1,1,1>(p,features,begin,n,h);
        node_triple_load<7,0,2>(p,features,begin,n,h);
        node_triple_load<8,1,3>(p,features,begin,n,h);
        node_triple_load<12,0,4>(p,features,begin,n,h);
        node_triple_load<13,1,5>(p,features,begin,n,h);
        __syncthreads();
        node_triple_score<0,1>(features,work);
        node_triple_score<2,1,0,3>(features,work+1024);
        node_triple_score<4,1,0,5>(features,work+2048);
        node_triple_score<2,5,4,3>(features,work+3072);
        __syncthreads();
        node_triple_load<2,2,0>(p,features,begin,n,h);
        node_triple_load<9,2,1>(p,features,begin,n,h);
        node_triple_load<14,2,2>(p,features,begin,n,h);
        __syncthreads();
        for(int key=row;key<n;key+=4) {
            F dot;
            for(int c=lane;c<VALUE_DIM;c+=32) {
                F value;
                value.x[0]=node_triple_float(features[key*NODE_TRIPLE_PITCH+c]);
                value.x[1]=node_triple_float(features[NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c]);
                value.x[2]=node_triple_float(features[2*NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c]);
                dot=jet_fma<false,false>(dot,value,static_upstream<4>(p,5,atom,h,c));
            }
            dot=warp_sum(dot);
            if(lane==0) full_dot[key]=dot;
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<8;++r) {
            int q=row+r*4,pos=q*32+lane;
            F score(-INFINITY),prob;
            if(q<n && lane<n && (!CAUSAL || lane<=q)) {
                #pragma unroll
                for(int component=0;component<4;++component) score.x[component]=work[component*1024+pos]*p.scale;
                if constexpr(BIAS_KIND==1) score=score+static_input<4>(p,3,begin+lane,h);
                if constexpr(BIAS_KIND==2) score=score+static_input<4>(p,3,atom,h,q,lane);
            }
            float maximum=score.x[0];
            #pragma unroll
            for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            F weight=isfinite(maximum) && isfinite(score.x[0]) ? exponential(score-F(maximum)):F();
            F denominator=warp_sum(weight);
            if(denominator.x[0]>0.f) prob=weight*reciprocal(denominator);
            F dot=lane<n ? full_dot[lane]:F();
            F delta=warp_sum(prob*dot);
            F env=q<n && (PRESENT&16) ? static_input<4>(p,4,begin+q):F(1.f);
            F ds=prob*(dot-delta)*env;
            #pragma unroll
            for(int component=0;component<4;++component) {
                work[component*1024+pos]=prob.x[component];
                work[(component+4)*1024+pos]=ds.x[component];
            }
            if(lane==0 && q<n) {
                envelope_third[r]+=delta.x[3];
                envelope_direction[r]+=delta.x[2];
            }
            if constexpr(BIAS_KIND==2) if(q<n && lane<n) {
                if constexpr(NEEDS&8) p.out[3].store(p.out[3].index(atom,h,q,lane),ds.x[3]);
                if constexpr(NEEDS&(1u<<10)) p.out[10].store(p.out[10].index(atom,h,q,lane),ds.x[2]);
            }
        }
        __syncthreads();
        if(threadIdx.x<n) {
            F sum;
            for(int q=0;q<n;++q) {
                F prob;
                #pragma unroll
                for(int component=0;component<4;++component) prob.x[component]=work[component*1024+q*32+threadIdx.x];
                F env=(PRESENT&16) ? static_input<4>(p,4,begin+q):F(1.f);
                sum=jet_fma<false,true>(sum,env,prob);
            }
            alpha[threadIdx.x]=sum;
        }
        if constexpr(BIAS_KIND==1) if(threadIdx.x<n) {
            float third=0.f,direction=0.f;
            for(int q=0;q<n;++q) {third+=work[7168+q*32+threadIdx.x];direction+=work[6144+q*32+threadIdx.x];}
            if constexpr(NEEDS&8) p.out[3].store(p.out[3].index(begin+threadIdx.x,h),third);
            if constexpr(NEEDS&(1u<<10)) p.out[10].store(p.out[10].index(begin+threadIdx.x,h),direction);
        }
        __syncthreads();
        for(int t=threadIdx.x;t<n*VALUE_DIM;t+=128) {
            int key=t/VALUE_DIM,c=t%VALUE_DIM;
            F g=static_upstream<4>(p,5,atom,h,c),a=alpha[key];
            float third=fmaf(a.x[3],g.x[0],a.x[1]*g.x[2]);
            float direction=fmaf(a.x[2],g.x[0],a.x[0]*g.x[2]);
            if constexpr(NEEDS&4) p.out[2].store((int64_t(begin+key)*HEADS+h)*VALUE_DIM+c,third);
            if constexpr(NEEDS&(1u<<9)) p.out[9].store((int64_t(begin+key)*HEADS+h)*VALUE_DIM+c,direction);
        }
        if constexpr(NEEDS&32) for(int c=threadIdx.x;c<VALUE_DIM;c+=128) {
            float sum=0.f;
            for(int key=0;key<n;++key) {
                F a=alpha[key];
                sum=fmaf(a.x[3],node_triple_float(features[key*NODE_TRIPLE_PITCH+c]),sum);
                sum=fmaf(a.x[1],node_triple_float(features[2*NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c]),sum);
                sum=fmaf(a.x[2],node_triple_float(features[NODE_TRIPLE_PLANE+key*NODE_TRIPLE_PITCH+c]),sum);
            }
            p.out[5].store(p.out[5].index(atom,h,c),sum);
        }
        __syncthreads();
        auto* coefficients=reinterpret_cast<NodeTripleTile*>(work);
        for(int t=threadIdx.x;t<4*1280;t+=128) {
            int plane=t/1280,row=t%1280/40,col=t%40;
            coefficients[t]=node_triple_cast(col<32 ? work[4096+plane*1024+row*32+col]:0.f);
        }
        node_triple_load<0,0,0>(p,features,begin,n,h);
        node_triple_load<1,1,1>(p,features,begin,n,h);
        node_triple_load<7,0,2>(p,features,begin,n,h);
        node_triple_load<8,1,3>(p,features,begin,n,h);
        node_triple_load<12,0,4>(p,features,begin,n,h);
        node_triple_load<13,1,5>(p,features,begin,n,h);
        __syncthreads();
        float* temporary=work+4096;
        node_triple_output<0,3,1,1,5,2,3>(p,features,coefficients,temporary,begin,n,h);
        node_triple_output<7,2,1,0,5>(p,features,coefficients,temporary,begin,n,h);
        node_triple_output<1,3,0,1,4,2,2,true>(p,features,coefficients,temporary,begin,n,h);
        node_triple_output<8,2,0,0,4,-1,-1,true>(p,features,coefficients,temporary,begin,n,h);
    }
    if(lane==0) {
        #pragma unroll
        for(int r=0;r<8;++r) if(row+r*4<n) {
            if constexpr(NEEDS&16) p.out[4].store(begin+row+r*4,envelope_third[r]);
            if constexpr(NEEDS&(1u<<11)) p.out[11].store(begin+row+r*4,envelope_direction[r]);
        }
    }
}
__global__ __launch_bounds__(128,2)
void small_triple_kernel(const __grid_constant__ Params p,
    const __grid_constant__ Params native_p,
    const __grid_constant__ Params final_p,float* envelope) {
    int64_t atom=blockIdx.x,begin=p.offset(atom),end=p.offset(atom+1);
    if(end-begin<=32) {
        clear_optional_outputs<3>(native_p,atom,begin,end);
        node_triple_derivative(native_p,atom,begin,int(end-begin));
        __syncthreads();
        finish_higher_rows<3>(p,final_p,atom,false);
    } else small_higher_long<3>(p,final_p,atom,begin,end,envelope);
}

#endif

"""

SOURCE = r"""


// The mixed component of the analytic gradient gives the primal third
// derivative plus the second-output-adjoint term. Its second direction gives
// the gradient of the first incoming direction; the output mixed component
// gives the gradient of the original node/edge adjoints.
void run(std::vector<std::optional<at::Tensor>> inputs, at::Tensor csr,
         std::vector<std::optional<at::Tensor>> outputs,
         std::vector<std::optional<at::Tensor>> final_outputs, double scale) {
    c10::cuda::CUDAGuard guard(inputs[0]->device());
    if (csr.numel() > 1) {
        constexpr int extent = DIM > VALUE_DIM ? DIM : VALUE_DIM;
        constexpr int width = extent < 32 ? extent : 32;
        constexpr int feature_size = 3 * 32 * (width + 1) * 3;
        constexpr int pv_size = 2 * 4 * 1024 + 32 * (width + 1) * 3;
        constexpr int qk_size = 4 * 1024 + 2 * 32 * (width + 1) * 3;
        constexpr int base_arena = feature_size > pv_size ? (feature_size > qk_size ? feature_size : qk_size) : (pv_size > qk_size ? pv_size : qk_size);
#ifdef LOWP_BF16
    constexpr int arena = base_arena > feature_size + 1024 ? base_arena : feature_size + 1024;
#else
    constexpr int arena = base_arena;
#endif

        #ifdef LOWP_BF16
        constexpr int dynamic_bytes = (arena+4*64)*sizeof(float);
#else
        constexpr int short_bytes = (arena+4*64)*sizeof(float);
        constexpr int dynamic_bytes = short_bytes;
#endif
        static thread_local int configured_device=-1;
        static thread_local int multiprocessors=0;
        int device=inputs[0]->get_device();
        if(configured_device!=device) {
            if constexpr(dynamic_bytes>0)
                C10_CUDA_CHECK(cudaFuncSetAttribute(derivative_kernel<3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_bytes));
            C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<3,512,2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,TiledBackwardLayout<3>::SharedBytes));
            C10_CUDA_CHECK(cudaFuncSetAttribute(long_rows_kernel<3,512,2,false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,TiledBackwardLayout<3>::SharedBytes));
            if constexpr(TiledBackwardLayout<3>::NodeOnly) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<3,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeHigherLayout<3,64>::Bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<3,64>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<3,48>,cudaFuncAttributeMaxDynamicSharedMemorySize,NodeHigherLayout<3,48>::Bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(node_higher_cached_kernel<3,48>,cudaFuncAttributePreferredSharedMemoryCarveout,100));
            }
#ifdef LOWP_BF16
            if constexpr(NODE_TRIPLE_ENABLED) {
                constexpr int small_bytes=NODE_TRIPLE_BYTES>TiledBackwardLayout<3>::SharedBytes?
                    NODE_TRIPLE_BYTES:TiledBackwardLayout<3>::SharedBytes;
                C10_CUDA_CHECK(cudaFuncSetAttribute(small_triple_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,small_bytes));
                C10_CUDA_CHECK(cudaFuncSetAttribute(small_triple_kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
            }
#endif
            multiprocessors=at::cuda::getDeviceProperties(device)->multiProcessorCount;
            configured_device=device;
        }
        Params p=params(inputs,outputs,csr,scale);
        Params final_p=params(inputs,final_outputs,csr,scale);
        Params native_p=p;
#ifdef LOWP_BF16
        for(int slot=0;slot<12;++slot)
            if((slot>=7?slot-7:slot)<3)native_p.out[slot]=final_p.out[slot];
#endif
        float* envelope=nullptr;
        int query_workers=1;
        bool resident=false;
        auto workspace=tiled_envelope_workspace<3>(*inputs[0]);
        if(workspace.defined()) envelope=workspace.data_ptr<float>();
        query_workers=tiled_query_workers<3>(inputs[0]->get_device(),csr.numel()-1);
        // Amortize two resident launches only when average degree exceeds a streaming tile.
        if constexpr(TiledBackwardLayout<3>::NodeOnly)
            resident=inputs[0]->size(0)>32*(csr.numel()-1);
        if(resident) query_workers=4;
#ifdef LOWP_BF16
        if constexpr(NODE_TRIPLE_ENABLED) {
            if(!resident && (csr.numel()-1)*HEADS<=2*multiprocessors) {
                constexpr int small_bytes=NODE_TRIPLE_BYTES>TiledBackwardLayout<3>::SharedBytes?
                    NODE_TRIPLE_BYTES:TiledBackwardLayout<3>::SharedBytes;
                small_triple_kernel<<<csr.numel()-1,128,small_bytes,at::cuda::getCurrentCUDAStream()>>>(p,native_p,final_p,envelope);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
        }
#endif


        derivative_kernel<3><<<csr.numel()-1,512,dynamic_bytes,at::cuda::getCurrentCUDAStream()>>>(p,native_p,query_workers,resident);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if(resident) {
        long_rows_kernel<3,512,2><<<dim3(csr.numel()-1,HEADS,query_workers),512,
            TiledBackwardLayout<3>::SharedBytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
        long_rows_kernel<3,512,2,false><<<dim3(csr.numel()-1,HEADS,query_workers),512,
            TiledBackwardLayout<3>::SharedBytes,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if constexpr(TiledBackwardLayout<3>::NodeOnly) if(resident) {
            node_higher_cached_kernel<3,64><<<dim3(csr.numel()-1,HEADS),(NodeHigherLayout<3,64>::Threads),(NodeHigherLayout<3,64>::Bytes),at::cuda::getCurrentCUDAStream()>>>(native_p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            node_higher_cached_kernel<3,48><<<dim3(csr.numel()-1,HEADS),(NodeHigherLayout<3,48>::Threads),(NodeHigherLayout<3,48>::Bytes),at::cuda::getCurrentCUDAStream()>>>(native_p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if(envelope) {
            tiled_envelope_reduce<3><<<csr.numel()-1,128,0,at::cuda::getCurrentCUDAStream()>>>(p,envelope);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
#ifdef LOWP_BF16
        finish_higher<3><<<csr.numel()-1,256,0,at::cuda::getCurrentCUDAStream()>>>(p,final_p,resident);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
"""


@lru_cache(maxsize=128)
def get_triple_backward_kernel(config, device, dtype, present, contiguous, needed):
    defines = f"#define PRESENT {present}u\n#define NEEDS {needed}u\n#define CONTIGUOUS {contiguous}u\n"
    native_score = 0
    if dtype in (torch.float16, torch.bfloat16):
        pitch = (config[0] + 15) // 16 * 16 + 8
        required = 4 * max(3 * 64 * pitch, 4 * 64 * 72 // 2 + 4 * (64 * 64 + 4 * 64))
        native_score = int(required <= get_device_properties(device).shared_memory_per_block_optin)
    defines += f"#define HIGHER_NATIVE_SCORE {native_score}\n"
    return load_kernel(
        "attention_conv_triple_backward", defines + LOWP_SOURCE + GRADIENT_SOURCE + TILED_SOURCE + DERIVATIVE_SOURCE + NODE_TRIPLE_SOURCE + SOURCE,
        config, device.index, dtype,
    ).run


def attention_conv_triple_backward(inputs, csr, config, needs):
    present = sum(1 << i for i, value in enumerate(inputs) if value is not None)
    contiguous = sum(1 << i for i, value in enumerate(inputs) if value is not None and value.is_contiguous())
    needed = sum(1 << i for i, need in enumerate(needs) if need and inputs[i] is not None)
    outputs = allocate(inputs[:12], needs, accumulate=True)
    final_outputs = allocate(inputs[:12], needs) if inputs[0].dtype in (torch.float16, torch.bfloat16) else outputs
    kernel = get_triple_backward_kernel(config, inputs[0].device, inputs[0].dtype, present, contiguous, needed)
    kernel(inputs, csr, outputs, final_outputs, config[3])
    return final_outputs
