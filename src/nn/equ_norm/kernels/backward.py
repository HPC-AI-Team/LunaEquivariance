"""Equivariant normalization backward CUDA."""

import torch

from ....utils.precision import suffix_float32_literals
from .utils import color_segments, fast_view_source, make_binding, make_reduction_prelude


def make_backward_source(plan, dtype):
    """Fuse feature and affine gradients with bounded copy reductions."""
    block_descriptors = {
        b: "{%d,%d,%d,%d,%d,%d,%d,0,%.17e}" % (
            b.offset, b.channels, b.components, b.param_offset,
            b.bias_offset, b.group, b.center_index, b.weight,
        )
        for b in plan.blocks
    }
    common_source = make_reduction_prelude(plan, dtype, 1)

    # Native Element atomics need no global workspace.
    word = 2 if dtype == torch.float64 else 1
    gang = 256
    for candidate in (32, 64, 128, 256):
        owners, counts = color_segments(plan, candidate)
        if word * (2 * counts[0] + 2 * counts[1] + counts[2]) <= 72:
            gang = candidate
            break
    owners, (nd, np, nb) = color_segments(plan, gang)
    if word * (2 * nd + 2 * np + nb) > 72:
        cpp = make_binding(
            dtype,
            r"""int64_t workspace_size(int64_t n,bool parameters=true);
""",
            stats_width=0, parameter_outputs=(1, 2),
        )
        cuda = ""
        cuda += r"""
__device__ __forceinline__ Acc backward_stream_sum(Acc value) {
    __shared__ Acc shared[4];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value+=__shfl_down_sync(0xffffffff,value,d);
    if(lane==0)shared[warp]=value;
    __syncthreads();
    value=0;
    #pragma unroll
    for(int w=0;w<4;++w)value+=shared[w];
    __syncthreads();
    return value;
}
__global__ void backward_stream_zero(Outputs out) {
    const int p=blockIdx.x*128+threadIdx.x;
    if(out.v[1] && p<P)out.v[1][p]=0;
    if(out.v[2] && p<BIAS)out.v[2][p]=0;
}
__global__ __launch_bounds__(128) void backward_stream_rows(Inputs in,Outputs out,const Acc* fs,int64_t N) {
    const int group=blockIdx.y,tid=threadIdx.x;
    for(int64_t row=blockIdx.x;row<N;row+=gridDim.x) {
        const Acc r=fs[row*FW+NCENTER+group];
        Acc H=0;
"""
        for block in plan.blocks:
            cuda += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group)break;

        const Acc anchor=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
        const Acc mean=b.center_index>=0?fs[row*FW+b.center_index]:0;
        for(int c=tid;c<b.channels;c+=128) {{
            const Acc gamma=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
            Acc gp=0,gb=0;
            #pragma unroll 1
            for(int m=0;m<b.components;++m) {{
                const int j=b.offset+local_index(b,c,m);
                const Acc x=in.v[0].get<UnitStride>(row,j),g=in.v[3].get<UnitStride>(row,j);
                const Acc z=(b.center_index>=0?(x-anchor)-mean:x)*r;
                gp+=g*z;
                if(m==0)gb=g;
            }}
            H+=gamma*gp;
            if(out.v[1])atomicAdd(out.v[1]+b.param_offset+c,Element(gp));
            if(b.bias_offset>=0 && out.v[2])atomicAdd(out.v[2]+b.bias_offset+c,Element(gb));
        }}
        }}while(false);
"""
        cuda += r"""H=backward_stream_sum(H);
if(out.v[0]) {
"""
        for block in plan.blocks:
            cuda += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group)break;

        const Acc anchor=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
        const Acc mean=b.center_index>=0?fs[row*FW+b.center_index]:0;
"""
            if block.center_index >= 0 and block.channels == 1:
                # The centering projection on a one-channel scalar is exactly zero.
                cuda += r"""
        if(tid==0) out.v[0][row*D+b.offset]=Element(0);
"""
            elif block.center_index >= 0:
                cuda += r"""
        Acc projection=0;
        for(int c=tid;c<b.channels;c+=128) {
            const Acc gamma=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
            const Acc x=in.v[0].get<UnitStride>(row,b.offset+c),g=in.v[3].get<UnitStride>(row,b.offset+c);
            const Acc z=((x-anchor)-mean)*r;
            projection+=r*(gamma*g-b.weight*z*H);
        }
        projection=backward_stream_sum(projection)/b.channels;
        for(int c=tid;c<b.channels;c+=128) {
            const Acc gamma=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
            const Acc x=in.v[0].get<UnitStride>(row,b.offset+c),g=in.v[3].get<UnitStride>(row,b.offset+c);
            const Acc z=((x-anchor)-mean)*r;
            out.v[0][row*D+b.offset+c]=Element(r*(gamma*g-b.weight*z*H)-projection);
        }
"""
            else:
                cuda += r"""
        for(int c=tid;c<b.channels;c+=128) {
            const Acc gamma=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
            #pragma unroll 1
            for(int m=0;m<b.components;++m) {
                const int j=b.offset+local_index(b,c,m);
                const Acc x=in.v[0].get<UnitStride>(row,j),g=in.v[3].get<UnitStride>(row,j),z=x*r;
                out.v[0][row*D+j]=Element(r*(gamma*g-b.weight*z*H));
            }
        }
"""
            cuda += "}while(false);\n"
        cuda += r"""
}
}
}
int64_t workspace_size(int64_t,bool) {return 0;}
void launch(Inputs in,Outputs out,const Acc* fs,Acc*,int64_t N,cudaStream_t stream) {
    if(out.v[1] || out.v[2])backward_stream_zero<<<(max(P,BIAS)+127)/128,128,0,stream>>>(out);
    if(N>0)backward_stream_rows<<<dim3(int(min(N,int64_t(65535))),NG),128,0,stream>>>(in,out,fs,N);
}
"""
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        return fast_view_source(
            {"cpp": cpp, "body": cuda, "prefix": common_source, "cuda_workspace": True},
            plan, dtype, active_inputs=(0, 1, 3),
        )

    nb = max(1, nb)
    threads = max(128, gang)
    cpp = make_binding(
        dtype,
        r"""int64_t workspace_size(int64_t n,bool parameters=true);
""",
        stats_width=0, parameter_outputs=(1, 2),
    )
    cuda = ""
    cuda += "\n\n\n"
    cuda += f"""constexpr int OWN_THREADS={threads}, GANG={gang}, GANGS={threads // gang};
constexpr int DATA_SLOTS={nd}, PARAM_SLOTS={np}, BETA_SLOTS={nb};
"""
    cuda += r"""
__device__ __forceinline__ Acc gang_sum(Acc value,Acc* shared) {
    const int lane=threadIdx.x&31, local_lane=threadIdx.x%GANG;
    const int gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value+=__shfl_down_sync(0xffffffff,value,d);
    if constexpr(GANG==32) return __shfl_sync(0xffffffff,value,0);
    else {
        if(lane==0) shared[threadIdx.x/32]=value;
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        value=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) value+=shared[gang_id*(GANG/32)+w];
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        return value;
    }
}
"""
    cuda += r"""__global__ __launch_bounds__(OWN_THREADS) void backward_channel_colored(
        Inputs in,Outputs out,const Acc* fs,Acc* partials,int64_t N,int workers) {
    __shared__ Acc row_shared[OWN_THREADS/32];
    __shared__ Acc parameter_shared[GANGS*PARAM_SLOTS*GANG];
    __shared__ Acc beta_shared[GANGS*BETA_SLOTS*GANG];
    const int worker=blockIdx.x/NG,first_group=blockIdx.x%NG;
    const int gang_id=threadIdx.x/GANG,lane=threadIdx.x%GANG;
    for(int group=first_group;group<NG;group+=gridDim.x) {
        Acc z_cache[DATA_SLOTS],g_cache[DATA_SLOTS],gamma_cache[PARAM_SLOTS];
        Acc gamma_acc[PARAM_SLOTS]={0},beta_acc[BETA_SLOTS]={0};
"""
    for o in owners:
        cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(owns) gamma_cache[{o['param']}]=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
}} while(false);
"""
    cuda += r"""for(int64_t start=int64_t(worker)*GANGS;start<N;start+=int64_t(workers)*GANGS) {
const int64_t row=start+gang_id; const bool active=row<N;
const Acc r=active?fs[row*FW+NCENTER+group]:0;
Acc H=0;
"""
    for o in owners:
        b, data_slot, parameter_slot, bias_slot = o["block"], o["data"], o["param"], o["bias"]
        cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
const Acc anchor=active && b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
const Acc mean_delta=active && b.center_index>=0?fs[row*FW+b.center_index]:0;
Acc gp=0;
"""
        cuda += f"""#pragma unroll
for(int m=0;m<b.components;++m) {{ const int j=local_index(b,c,m);
const Acc u=active?in.v[0].get<UnitStride>(row,b.offset+j):0,g=active?in.v[3].get<UnitStride>(row,b.offset+j):0;
const Acc z=(b.center_index>=0?(u-anchor)-mean_delta:u)*r;
z_cache[{data_slot}+m]=z;g_cache[{data_slot}+m]=g;gp+=g*z;
}}
H+=gamma_cache[{parameter_slot}]*gp; if(out.v[1]) gamma_acc[{parameter_slot}]+=gp;
"""
        if b.bias_offset >= 0:
            cuda += f"""if(out.v[2]) beta_acc[{bias_slot}]+=g_cache[{data_slot}];
"""
        cuda += "} while(false);\n"
    cuda += r"""H=gang_sum(H,row_shared);
if(out.v[0]) {
"""
    for b in plan.blocks:
        own = [o for o in owners if o["block"] is b]
        if b.center_index >= 0:
            cuda += r"""Acc projection%d=0;
""" % b.center_index
        for o in own:
            data_slot, parameter_slot = o["data"], o["param"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
#pragma unroll
for(int m=0;m<b.components;++m) {{
const Acc dx=r*(g_cache[{data_slot}+m]*gamma_cache[{parameter_slot}]-b.weight*z_cache[{data_slot}+m]*H);
"""
            if b.center_index >= 0:
                cuda += f"""z_cache[{data_slot}+m]=dx; projection{b.center_index}+=dx;
"""
            else:
                cuda += r"""if(active) out.v[0][row*D+b.offset+local_index(b,c,m)]=dx;
"""
            cuda += "}\n} while(false);\n"
        if b.center_index >= 0:
            cuda += f"""projection{b.center_index}=gang_sum(projection{b.center_index},row_shared)/{b.channels};
"""
            for o in own:
                cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(owns && active) out.v[0][row*D+b.offset+c]=z_cache[{o['data']}]-projection{b.center_index};
}} while(false);
"""
    cuda += r"""}
}
if(out.v[1] || out.v[2]) {
"""
    for o in owners:
        b, parameter_slot, bias_slot = o["block"], o["param"], o["bias"]
        cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(owns) {{ parameter_shared[gang_id*PARAM_SLOTS*GANG+{parameter_slot}*GANG+lane]=gamma_acc[{parameter_slot}];
"""
        if b.bias_offset >= 0:
            cuda += f"beta_shared[gang_id*BETA_SLOTS*GANG+{bias_slot}*GANG+lane]=beta_acc[{bias_slot}];\n"
        cuda += "}\n} while(false);\n"
    cuda += r"""__syncthreads();
if(gang_id==0) {
"""
    for o in owners:
        b, parameter_slot, bias_slot = o["block"], o["param"], o["bias"]
        cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
Acc gp=0,bp=0;
#pragma unroll
for(int gang=0;gang<GANGS;++gang) {{ gp+=parameter_shared[gang*PARAM_SLOTS*GANG+{parameter_slot}*GANG+lane];
"""
        if b.bias_offset >= 0:
            cuda += f"bp+=beta_shared[gang*BETA_SLOTS*GANG+{bias_slot}*GANG+lane];\n"
        cuda += r"""}
if(out.v[1]) atomicAdd(out.v[1]+b.param_offset+c,gp);
"""
        if b.bias_offset >= 0:
            cuda += r"""if(out.v[2]) atomicAdd(out.v[2]+b.bias_offset+c,bp);
"""
        cuda += "} while(false);\n"
    cuda += r"""}
__syncthreads();
}
}
"""
    cuda += r"""}
__global__ void backward_colored_zero(Outputs out) {
    const int p=blockIdx.x*128+threadIdx.x;
    if(out.v[1] && p<P) out.v[1][p]=Element(0);
    if(out.v[2] && p<BIAS) out.v[2][p]=Element(0);
}
"""
    cuda += r"""thread_local int row_sm_count=1;
int row_capacity() {
    static thread_local int cached_device=-1, cached_capacity=0;
    int device=0; cudaGetDevice(&device);
    if(device!=cached_device) {
        int per_sm=0; cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop,device);
        cudaError_t err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,backward_channel_colored,OWN_THREADS,0);
        if(err!=cudaSuccess || per_sm<=0) throw std::runtime_error("backward row capacity is unavailable");
        cached_device=device; cached_capacity=per_sm*prop.multiProcessorCount; row_sm_count=prop.multiProcessorCount;
    }
    return cached_capacity;
}
int worker_count(int64_t N) {
    const int available=max(1,row_capacity()/NG);
    const int minimum_grid=max(1,(row_sm_count+NG-1)/NG);
    const int64_t contractions=(N+4*GANGS-1)/(4*GANGS);
    const int64_t workers=min(int64_t(available),max(int64_t(minimum_grid),contractions));
    return int(min(workers,(N+GANGS-1)/GANGS));
}
int64_t workspace_size(int64_t n,bool parameters) {
    return 0;
}
"""
    cuda += r"""void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    const int workers=worker_count(N),blocks=min(row_capacity(),workers*NG);
    if(out.v[1] || out.v[2]) backward_colored_zero<<<(max(P,BIAS)+127)/128,128,0,stream>>>(out);
    backward_channel_colored<<<blocks,OWN_THREADS,0,stream>>>(in,out,fs,stats,N,workers);
}
"""
    if dtype == torch.float32:
        cuda = suffix_float32_literals(cuda)
    return fast_view_source(
        {"cpp": cpp, "body": cuda, "prefix": common_source, "cuda_workspace": True,
         "trailing_includes": ("#include <stdexcept>",)},
        plan, dtype, active_inputs=(0, 1, 3),
    )
