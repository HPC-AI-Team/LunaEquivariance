"""Equivariant normalization double-backward CUDA."""

from operator import attrgetter

import torch

from ....utils.precision import suffix_float32_literals
from .utils import FEATURE_THREADS, color_segments, fast_view_source, make_binding, make_reduction_prelude


def make_feature_reduce(name):
    return r"""__device__ __forceinline__ void streaming_reduce(Acc (&values)[FEATURE_K],Acc* shared) {
    const int lane=threadIdx.x&31;
    #pragma unroll
    for(int k=0;k<FEATURE_K;++k) {
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1) values[k]+=__shfl_down_sync(0xffffffff,values[k],delta);
        if(lane==0) shared[k*(FEATURE_THREADS/32)+threadIdx.x/32]=values[k];
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<FEATURE_K;++k) {
        Acc total=0;
        #pragma unroll
        for(int w=0;w<FEATURE_THREADS/32;++w) total+=shared[k*(FEATURE_THREADS/32)+w];
        values[k]=total;
    }
    __syncthreads();
}
""".replace("streaming_reduce", name)


def append_offset_launch(cuda, kernel_name):
    """Add the offset kernel and launch wrapper for large grids."""
    start = cuda.index("template<int ROWS> __global__")
    offset_kernel = cuda[start:].replace(f"{kernel_name}(", f"{kernel_name}_offset(", 1)
    offset_kernel = offset_kernel.replace("const Acc* fs,int64_t N)", "const Acc* fs,int64_t N,int64_t block_start)", 1)
    offset_kernel = offset_kernel.replace(
        "const int group=blockIdx.x%NG;\n    const int64_t first=int64_t(blockIdx.x/NG)*ROWS;",
        "const int64_t block=block_start+int64_t(blockIdx.x);\n"
        "    const int group=int(block%NG);\n    const int64_t first=(block/NG)*ROWS;",
    )
    cuda += offset_kernel
    launch = r"""
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    if(out.v[1] || out.v[2]) parameter_zero<<<(max(P,BIAS)+127)/128,128,0,stream>>>(out);
    if(N<256) double_backward_streaming<1><<<N*NG,FEATURE_THREADS,0,stream>>>(in,out,fs,N);
    else {
        const int64_t blocks=((N+7)/8)*NG;
        if(blocks<=2147483647)
            double_backward_streaming<8><<<blocks,FEATURE_THREADS,0,stream>>>(in,out,fs,N);
        else for(int64_t start=0;start<blocks;) {
            const int64_t count=min(blocks-start,int64_t(2147483647));
            double_backward_streaming_offset<8><<<count,FEATURE_THREADS,0,stream>>>(in,out,fs,N,start);
            start+=count;
        }
    }
}
"""
    return cuda + launch.replace("double_backward_streaming", kernel_name)


def make_double_backward_source(plan, dtype):
    block_descriptors = {
        b: "{%d,%d,%d,%d,%d,%d,%d,0,%.17e}" % (
            b.offset, b.channels, b.components, b.param_offset,
            b.bias_offset, b.group, b.center_index, b.weight,
        )
        for b in plan.blocks
    }

    # z=r*P(x), hr=r*P(rx); inverse powers are folded into reduced moments.
    emitted = make_reduction_prelude(plan, dtype, 2)
    cuda = ""
    cpp = make_binding(
        dtype,
        r"""int64_t workspace_size(int64_t n,bool parameters=true) { return 0; }
""",
        stats_width=0, parameter_outputs=(1,),
    )

    row = FEATURE_THREADS
    offsets = {}
    sizes = [0] * plan.num_groups
    for bi, b in enumerate(plan.blocks):
        offsets[bi] = sizes[b.group]
        sizes[b.group] += (b.channels * b.components + row - 1) // row
    slots = max(sizes)
    if slots * 4 * (2 if dtype == torch.float64 else 1) > 64:
        K = 3 + 3 * plan.num_centers
        cuda += f"""
constexpr int FEATURE_THREADS={FEATURE_THREADS}, FEATURE_K={K};
"""
        cuda += "\n" + make_feature_reduce("streaming_reduce")
        cuda += r"""__global__ void parameter_zero(Outputs out) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(out.v[1] && p<P) out.v[1][p]=0;
    if(out.v[2] && p<BIAS) out.v[2][p]=0;
}
template<int ROWS> __global__ __launch_bounds__(FEATURE_THREADS) void double_backward_streaming(
    Inputs in,Outputs out,const Acc* fs,int64_t N) {
    __shared__ Acc reduced[FEATURE_K*(FEATURE_THREADS/32)];
    const int group=blockIdx.x%NG;
    const int64_t first=int64_t(blockIdx.x/NG)*ROWS;
    const int lane32=threadIdx.x&31;
    for(int step=0;step<ROWS && first+step<N;++step) {
        const int64_t row=first+step;
        const Acc r=fs[row*FW+NCENTER+group];
        Acc sums[FEATURE_K]={0};
"""
        anchors = r"""const Acc ax=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
const Acc mx=b.center_index>=0?fs[row*FW+b.center_index]:0;
const Acc ah=b.center_index>=0?in.v[4].get<UnitStride>(row,b.offset):0;
"""
        for b in plan.blocks:
            cuda += f"""do {{ constexpr Block b={block_descriptors[b]}; if(b.group!=group) break;
{anchors}"""
            cuda += r"""for(int j=threadIdx.x;j<b.channels*b.components;j+=FEATURE_THREADS) {
const int c=channel(b,j);
const Acc xv=in.v[0].get<UnitStride>(row,b.offset+j),hv=in.v[4].get<UnitStride>(row,b.offset+j);
const Acc g=in.v[3].get<UnitStride>(row,b.offset+j);
const Acc z=(b.center_index>=0?(xv-ax)-mx:xv)*r,h=(b.center_index>=0?hv-ah:hv)*r;
const Acc ga=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0),rg=in.v[5].parameter<UnitStride>(b.param_offset+c);
sums[0]+=g*ga*z;sums[1]+=b.weight*z*h;sums[2]+=h*g*ga+g*rg*z;
"""
            if b.center_index >= 0:
                ci = b.center_index
                cuda += f"""sums[{3 + 3 * ci}]+=h;sums[{4 + 3 * ci}]+=g*ga;sums[{5 + 3 * ci}]+=g*rg;
"""
            cuda += "}\n} while(false);\n"
        cuda += r"""streaming_reduce(sums,reduced);
const Acc S=sums[0],C=sums[1];Acc T=sums[2];
"""
        for b in sorted((b for b in plan.blocks if b.center_index >= 0), key=attrgetter("center_index")):
            ci = b.center_index
            cuda += f"""const Acc hm{ci}=sums[{3 + 3 * ci}]/{b.channels}.0;T-=hm{ci}*sums[{4 + 3 * ci}];
const Acc pm{ci}=r*(sums[{5 + 3 * ci}]/{b.channels}.0-C*sums[{4 + 3 * ci}]/{b.channels}.0);
"""
        cuda += r"""const Acc Q=T-3.0*S*C;
"""
        for b in plan.blocks:
            hm = f"hm{b.center_index}" if b.center_index >= 0 else "0.0"
            pm = f"pm{b.center_index}" if b.center_index >= 0 else "0.0"
            cuda += f"""do {{ constexpr Block b={block_descriptors[b]}; if(b.group!=group) break;
{anchors}const Acc wh=b.weight*S,wz=b.weight*Q;
"""
            cuda += r"""for(int chunk=0;chunk<(b.channels*b.components+FEATURE_THREADS-1)/FEATURE_THREADS;++chunk) {
const int j=threadIdx.x+chunk*FEATURE_THREADS,c=channel(b,j);
Acc z=0,h=0,g=0,ga=1.0,rg=0;
const bool valid=j<b.channels*b.components;
if(valid) {
const Acc xv=in.v[0].get<UnitStride>(row,b.offset+j),hv=in.v[4].get<UnitStride>(row,b.offset+j);
z=(b.center_index>=0?(xv-ax)-mx:xv)*r;
g=in.v[3].get<UnitStride>(row,b.offset+j);
ga=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);rg=in.v[5].parameter<UnitStride>(b.param_offset+c);
"""
            cuda += f"h=(b.center_index>=0?hv-ah:hv)*r-{hm};\n"
            dx = "0.0" if b.center_index >= 0 and b.channels == 1 else f"r*(g*(rg-ga*C)-h*wh-z*wz)-{pm}"
            rb = r"""in.v[6].parameter<UnitStride>(b.bias_offset+c)""" if b.bias_offset >= 0 else "0.0"
            cuda += f"""if(out.v[0]) out.v[0][row*D+b.offset+j]={dx};
"""
            cuda += f"""if(out.v[3]) out.v[3][row*D+b.offset+j]=ga*h+(rg-ga*C)*z+{rb};
}}
"""
            cuda += r"""if(out.v[1]) { Acc gp=valid?g*(h-z*C):0;
"""
            if plan.layout == "mul_ir":
                cuda += r"""#pragma unroll
for(int delta=1;delta<min(32,b.components);delta<<=1) {
const Acc other=__shfl_down_sync(0xffffffff,gp,delta);
const int other_copy=__shfl_down_sync(0xffffffff,c,delta);
if(lane32+delta<32 && other_copy==c) gp+=other;
}
if(valid && (lane32==0 || j%b.components==0)) atomicAdd(out.v[1]+b.param_offset+c,gp);
"""
            else:
                cuda += r"""if(valid) atomicAdd(out.v[1]+b.param_offset+c,gp);
"""
            cuda += "}\n}\n} while(false);\n"
        cuda += "}\n}\n"
        cuda = append_offset_launch(cuda, "double_backward_streaming")
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        feature_source = {"cpp": cpp, "body": cuda}
    else:
        K = 3 + 3 * plan.num_centers
        cuda += f"""
constexpr int FEATURE_THREADS={row}, FEATURE_SLOTS={slots}, FEATURE_K={K};
"""
        cuda += "\n" + make_feature_reduce("feature_reduce")
        cuda += r"""__global__ void parameter_zero(Outputs out) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(out.v[1] && p<P) out.v[1][p]=0;
    if(out.v[2] && p<BIAS) out.v[2][p]=0;
}
template<int ROWS> __global__ __launch_bounds__(FEATURE_THREADS) void double_backward_feature(
    Inputs in,Outputs out,const Acc* fs,int64_t N) {
    __shared__ Acc reduced[FEATURE_K*(FEATURE_THREADS/32)];
    const int group=blockIdx.x%NG;
    const int64_t first=int64_t(blockIdx.x/NG)*ROWS;
    const int lane32=threadIdx.x&31;
    Acc gamma_acc[FEATURE_SLOTS]={0};
    for(int step=0;step<ROWS && first+step<N;++step) {
        const int64_t row=first+step;
        const Acc r=fs[row*FW+NCENTER+group];
        Acc z_cache[FEATURE_SLOTS],h_cache[FEATURE_SLOTS],g_cache[FEATURE_SLOTS];
        Acc sums[FEATURE_K]={0};
"""
        for bi, b in enumerate(plan.blocks):
            cuda += f"""do {{ constexpr Block b={block_descriptors[b]}; if(b.group!=group) break;
"""
            cuda += r"""const Acc ax=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
const Acc mx=b.center_index>=0?fs[row*FW+b.center_index]:0;
const Acc ah=b.center_index>=0?in.v[4].get<UnitStride>(row,b.offset):0;
"""
            for slot in range((b.channels * b.components + row - 1) // row):
                pos = offsets[bi] + slot
                cuda += f"""do {{ const int j=threadIdx.x+{slot}*FEATURE_THREADS;
Acc z=0,h=0,g=0;
if(j<b.channels*b.components) {{ const int c=channel(b,j);
const Acc xv=in.v[0].get<UnitStride>(row,b.offset+j),hv=in.v[4].get<UnitStride>(row,b.offset+j);
g=in.v[3].get<UnitStride>(row,b.offset+j);
z=(b.center_index>=0?(xv-ax)-mx:xv)*r; h=(b.center_index>=0?hv-ah:hv)*r;
const Acc ga=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0),rg=in.v[5].parameter<UnitStride>(b.param_offset+c);
sums[0]+=g*ga*z;sums[1]+=b.weight*z*h;sums[2]+=h*g*ga+g*rg*z;
"""
                if b.center_index >= 0:
                    ci = b.center_index
                    cuda += f"""sums[{3 + 3 * ci}]+=h;sums[{4 + 3 * ci}]+=g*ga;sums[{5 + 3 * ci}]+=g*rg;
"""
                cuda += f"""}}
z_cache[{pos}]=z;h_cache[{pos}]=h;g_cache[{pos}]=g;
}} while(false);
"""
            cuda += "} while(false);\n"
        cuda += r"""feature_reduce(sums,reduced);
const Acc S=sums[0],C=sums[1];Acc T=sums[2];
"""
        for b in sorted((b for b in plan.blocks if b.center_index >= 0), key=attrgetter("center_index")):
            ci = b.center_index
            cuda += f"""const Acc hm{ci}=sums[{3 + 3 * ci}]/{b.channels}.0;T-=hm{ci}*sums[{4 + 3 * ci}];
const Acc pm{ci}=r*(sums[{5 + 3 * ci}]/{b.channels}.0-C*sums[{4 + 3 * ci}]/{b.channels}.0);
"""
        cuda += r"""const Acc Q=T-3.0*S*C;
"""
        for bi, b in enumerate(plan.blocks):
            hm = f"hm{b.center_index}" if b.center_index >= 0 else "0.0"
            pm = f"pm{b.center_index}" if b.center_index >= 0 else "0.0"
            cuda += f"""do {{ constexpr Block b={block_descriptors[b]}; if(b.group!=group) break;
const Acc wh=b.weight*S,wz=b.weight*Q;
"""
            for slot in range((b.channels * b.components + row - 1) // row):
                pos = offsets[bi] + slot
                cuda += f"""do {{ const int j=threadIdx.x+{slot}*FEATURE_THREADS;
if(j>=b.channels*b.components) break; const int c=channel(b,j);
const Acc z=z_cache[{pos}],h=h_cache[{pos}]-{hm},g=g_cache[{pos}];
const Acc ga=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0),rg=in.v[5].parameter<UnitStride>(b.param_offset+c),A=rg-ga*C;
if(out.v[1]) gamma_acc[{pos}]+=g*(h-z*C);
"""
                dx = "0.0" if b.center_index >= 0 and b.channels == 1 else f"r*(g*A-h*wh-z*wz)-{pm}"
                cuda += f"""if(out.v[0]) out.v[0][row*D+b.offset+j]={dx};
"""
                rb = r"""in.v[6].parameter<UnitStride>(b.bias_offset+c)""" if b.bias_offset >= 0 else "0.0"
                cuda += f"""if(out.v[3]) out.v[3][row*D+b.offset+j]=ga*h+A*z+{rb};
}} while(false);
"""
            cuda += "} while(false);\n"
        cuda += r"""}
if(out.v[1]) {
"""
        for bi, b in enumerate(plan.blocks):
            cuda += f"""do {{ constexpr Block b={block_descriptors[b]}; if(b.group!=group) break;
"""
            for slot in range((b.channels * b.components + row - 1) // row):
                pos = offsets[bi] + slot
                cuda += f"""do {{ const int j=threadIdx.x+{slot}*FEATURE_THREADS,c=channel(b,j);
Acc gp=gamma_acc[{pos}];
"""
                if plan.layout == "mul_ir":
                    cuda += r"""#pragma unroll
for(int delta=1;delta<min(32,b.components);delta<<=1) {
    const Acc other=__shfl_down_sync(0xffffffff,gp,delta);
    const int other_copy=__shfl_down_sync(0xffffffff,c,delta);
    if(lane32+delta<32 && other_copy==c) gp+=other;
}
"""
                    cuda += r"""if(j<b.channels*b.components && (lane32==0 || j%b.components==0)) atomicAdd(out.v[1]+b.param_offset+c,gp);
"""
                else:
                    cuda += r"""if(j<b.channels*b.components) atomicAdd(out.v[1]+b.param_offset+c,gp);
"""
                cuda += "} while(false);\n"
            cuda += "} while(false);\n"
        cuda += "}\n}\n"
        cuda = append_offset_launch(cuda, "double_backward_feature")
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        feature_source = {"cpp": cpp, "body": cuda}

    wide = (
        max(sum(b.channels * b.components for b in plan.blocks if b.group == g) for g in range(plan.num_groups)) > 1024
    )
    copy_budget = any(
        3 * counts[0] + 3 * counts[1] + counts[2] <= 48
        for _, counts in (color_segments(plan, gang // 2, 2) for gang in (128, 256, 512))
    )
    narrow_double_copy = (
        dtype == torch.float64
        and plan.layout == "ir_mul"
        and plan.param_dim < plan.dim
        and any(
            2 * (3 * counts[0] + 3 * counts[1] + counts[2]) <= 80
            for _, counts in (color_segments(plan, gang, 1) for gang in (128, 256, 512))
        )
    )
    if (
        dtype == torch.float32 and wide and plan.layout == "mul_ir" and copy_budget
    ) or narrow_double_copy:
        cuda = ""
        cpp = make_binding(
            dtype, r"""int64_t workspace_size(int64_t n,bool parameters=true);
""", stats_width=0, parameter_outputs=(1,),
        )
        component_lanes = 1 if plan.layout == "ir_mul" else 2
        gang = 512
        word = 2 if dtype == torch.float64 else 1
        for candidate in (128, 256, 512):
            owners, counts = color_segments(plan, candidate // component_lanes, component_lanes)
            if word * (counts[0] * 3 + counts[1] * 3 + counts[2]) <= (80 if word == 2 else 48):
                gang = candidate
                break
        owners, (nd, np, nb) = color_segments(plan, gang // component_lanes, component_lanes)
        nb = max(1, nb)
        workers_threads = max(128, gang)
        gangs = workers_threads // gang
        center_blocks = sorted(
            (b for b in plan.blocks if b.center_index >= 0), key=attrgetter("center_index")
        )
        K = 3 + 3 * plan.num_centers
        cuda += "\n\n"
        cuda += f"""constexpr int OWN_THREADS={workers_threads}, GANG={gang}, GANGS={gangs}, CL={component_lanes}, CHANNELS={gang // component_lanes}, K={K};
constexpr int DATA_SLOTS={nd}, PARAM_SLOTS={np}, BETA_SLOTS={nb};
"""
        cuda += r"""
__device__ __forceinline__ void gang_reduce(Acc (&values)[K],Acc* shared) {
    const int lane32=threadIdx.x&31, gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int k=0;k<K;++k) {
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1) values[k]+=__shfl_down_sync(0xffffffff,values[k],delta);
        if constexpr(GANG==32) values[k]=__shfl_sync(0xffffffff,values[k],0);
        else if(lane32==0) shared[k*(OWN_THREADS/32)+threadIdx.x/32]=values[k];
    }
    if constexpr(GANG>32) {
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        #pragma unroll
        for(int k=0;k<K;++k) {
            values[k]=0;
            #pragma unroll
            for(int w=0;w<GANG/32;++w) values[k]+=shared[k*(OWN_THREADS/32)+gang_id*(GANG/32)+w];
        }
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
    }
}
__global__ __launch_bounds__(OWN_THREADS) void double_backward_colored(
    Inputs in,Outputs out,const Acc* fs,Acc* partials,int64_t N,int workers) {
    __shared__ Acc row_shared[K*(OWN_THREADS/32)];
    __shared__ Acc parameter_shared[GANGS*PARAM_SLOTS*CHANNELS];
    const int worker=blockIdx.x/NG, first_group=blockIdx.x%NG;
    const int gang_id=threadIdx.x/GANG,lane=(threadIdx.x%GANG)/CL, mlane=threadIdx.x%CL;
    for(int group=first_group;group<NG;group+=gridDim.x) {
        Acc z_cache[DATA_SLOTS],h_cache[DATA_SLOTS],g_cache[DATA_SLOTS];
        Acc gamma_cache[PARAM_SLOTS],rgamma_cache[PARAM_SLOTS],rbeta_cache[BETA_SLOTS];
        Acc gamma_acc[PARAM_SLOTS]={0};
"""
        for o in owners:
            b, parameter_slot, bias_slot = o["block"], o["param"], o["bias"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
"""
            cuda += f"""if(owns) {{ gamma_cache[{parameter_slot}]=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
rgamma_cache[{parameter_slot}]=in.v[5].parameter<UnitStride>(b.param_offset+c); }}
"""
            if b.bias_offset >= 0:
                cuda += f"""if(owns) rbeta_cache[{bias_slot}]=in.v[6].parameter<UnitStride>(b.bias_offset+c);
"""
            cuda += "} while(false);\n"
        cuda += r"""for(int64_t start=int64_t(worker)*GANGS;start<N;start+=int64_t(workers)*GANGS) {
const int64_t row=start+gang_id; const bool active=row<N;
const Acc r=active?fs[row*FW+NCENTER+group]:0;
Acc sums[K]={0};
"""
        for o in owners:
            b, data_slot, parameter_slot = o["block"], o["data"], o["param"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
"""
            cuda += r"""const Acc anchor_x=active && b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
const Acc mean_x=active && b.center_index>=0?fs[row*FW+b.center_index]:0;
const Acc anchor_h=active && b.center_index>=0?in.v[4].get<UnitStride>(row,b.offset):0;
Acc gz=0,gh=0,zh=0;
"""
            cuda += f"""#pragma unroll
for(int slot=0;slot<(b.components+CL-1)/CL;++slot) {{ const int m=mlane+slot*CL; const int j=local_index(b,c,m);
const Acc xv=active && m<b.components?in.v[0].get<UnitStride>(row,b.offset+j):0;
const Acc hv=active && m<b.components?in.v[4].get<UnitStride>(row,b.offset+j):0;
const Acc g=active && m<b.components?in.v[3].get<UnitStride>(row,b.offset+j):0;
const Acc z=m<b.components?(b.center_index>=0?(xv-anchor_x)-mean_x:xv)*r:0;
const Acc h=m<b.components?(b.center_index>=0?hv-anchor_h:hv)*r:0;
z_cache[{data_slot}+slot]=z; h_cache[{data_slot}+slot]=h; g_cache[{data_slot}+slot]=g;
gz+=g*z; gh+=g*h; zh+=z*h;
"""
            if b.center_index >= 0:
                ci = b.center_index
                cuda += f"""if(m<b.components) {{ sums[{3 + 3 * ci}]+=h; sums[{4 + 3 * ci}]+=g*gamma_cache[{parameter_slot}]; sums[{5 + 3 * ci}]+=g*rgamma_cache[{parameter_slot}]; }}
"""
            cuda += f"""}}
sums[0]+=gamma_cache[{parameter_slot}]*gz; sums[1]+=b.weight*zh; sums[2]+=gamma_cache[{parameter_slot}]*gh+rgamma_cache[{parameter_slot}]*gz;
}} while(false);
"""
        cuda += r"""gang_reduce(sums,row_shared);
const Acc S=sums[0],C=sums[1];
Acc T=sums[2];
"""
        for b in center_blocks:
            ci = b.center_index
            cuda += f"""const Acc hm{ci}=sums[{3 + 3 * ci}]/{b.channels}.0;
T-=hm{ci}*sums[{4 + 3 * ci}];
const Acc pm{ci}=r*(sums[{5 + 3 * ci}]/{b.channels}.0-C*sums[{4 + 3 * ci}]/{b.channels}.0);
"""
        for o in owners:
            b, data_slot, parameter_slot, bias_slot = o["block"], o["data"], o["param"], o["bias"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
"""
            hm = f"hm{b.center_index}" if b.center_index >= 0 else "0.0"
            pm = f"pm{b.center_index}" if b.center_index >= 0 else "0.0"
            rb = f"""rbeta_cache[{bias_slot}]""" if b.bias_offset >= 0 else "0.0"
            dx = (
                "0.0"
                if b.center_index >= 0 and b.channels == 1
                else f"r*(g*rg-g*ga*C-b.weight*h*S-b.weight*z*T+3.0*b.weight*z*S*C)-{pm}"
            )
            cuda += f"""const Acc ga=gamma_cache[{parameter_slot}],rg=rgamma_cache[{parameter_slot}],rb={rb};
Acc gp=0;
#pragma unroll
for(int slot=0;slot<(b.components+CL-1)/CL;++slot) {{ const int m=mlane+slot*CL;
const Acc z=z_cache[{data_slot}+slot],h=h_cache[{data_slot}+slot]-{hm},g=g_cache[{data_slot}+slot];
const Acc e=h-z*C; gp+=g*e;
const int j=local_index(b,c,m);
if(active && m<b.components && out.v[0]) out.v[0][row*D+b.offset+j]={dx};
if(active && m<b.components && out.v[3]) out.v[3][row*D+b.offset+j]=ga*h+rg*z-ga*z*C+rb;
}}
#pragma unroll
for(int delta=CL/2;delta>0;delta>>=1) gp+=__shfl_down_sync(__activemask(),gp,delta,CL);
if(out.v[1] && mlane==0) gamma_acc[{parameter_slot}]+=gp;
"""
            if b.bias_offset >= 0:
                cuda += r"""if(active && row==0 && mlane==0 && out.v[2]) out.v[2][b.bias_offset+c]=0;
"""
            cuda += "} while(false);\n"
        cuda += r"""}
if(out.v[1]) {
"""
        for o in owners:
            b, parameter_slot = o["block"], o["param"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(owns && mlane==0) parameter_shared[gang_id*PARAM_SLOTS*CHANNELS+{parameter_slot}*CHANNELS+lane]=gamma_acc[{parameter_slot}];
}} while(false);
"""
        cuda += r"""__syncthreads();
if(gang_id==0 && mlane==0) {
"""
        for o in owners:
            b, parameter_slot = o["block"], o["param"]
            cuda += f"""do {{ constexpr Block b={block_descriptors[o['block']]}; if(b.group!=group) break;
const int c=lane+({o['channel_shift']}); const bool owns=lane>={o['lo']} && lane<{o['hi']};
if(!owns) break;
Acc gp=0;
#pragma unroll
for(int gang=0;gang<GANGS;++gang) gp+=parameter_shared[gang*PARAM_SLOTS*CHANNELS+{parameter_slot}*CHANNELS+lane];
atomicAdd(out.v[1]+b.param_offset+c,Element(gp));
}} while(false);
"""
        cuda += r"""}
__syncthreads();
}
}
}
"""
        cuda += r"""
__global__ void colored_parameter_zero(Outputs out) {
    const int p=blockIdx.x*128+threadIdx.x;
    if(out.v[1] && p<P) out.v[1][p]=Element(0);
    if(out.v[2] && p<BIAS) out.v[2][p]=Element(0);
}
int row_capacity() {
    static thread_local int cached_device=-1,cached_capacity=0;
    int device=0;cudaGetDevice(&device);
    if(device!=cached_device) {
        int per_sm=0;cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,device);
        cudaError_t err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,double_backward_colored,OWN_THREADS,0);
        if(err!=cudaSuccess || per_sm<=0) throw std::runtime_error("double-backward row capacity unavailable");
        cached_device=device;cached_capacity=per_sm*prop.multiProcessorCount;
    }
    return cached_capacity;
}
int worker_count(int64_t N) {
    return int(min(int64_t(max(1,row_capacity()/NG)),(N+GANGS-1)/GANGS));
}
int64_t workspace_size(int64_t n,bool parameters) {
    return 0;
}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    const int workers=worker_count(N),blocks=min(row_capacity(),workers*NG);
    if(out.v[1] || out.v[2]) colored_parameter_zero<<<(max(P,BIAS)+127)/128,128,0,stream>>>(out);
    double_backward_colored<<<blocks,OWN_THREADS,0,stream>>>(in,out,fs,stats,N,workers);
}
"""
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        copy_source = {"cpp": cpp, "body": cuda, "include": "#include <stdexcept>"}
        cuda = ""
        cuda += f"""
namespace feature_path {{
{feature_source['body']}
}}
namespace copy_path {{
{copy_source['body']}
}}
"""
        if narrow_double_copy:
            cuda += r"""int64_t workspace_size(int64_t n,bool parameters) {
    return n>=64 && n<256?copy_path::workspace_size(n,parameters):0;
}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    if(N>=64 && N<256) copy_path::launch(in,out,fs,stats,N,stream);
    else feature_path::launch(in,out,fs,stats,N,stream);
}
"""
        else:
            cuda += r"""int64_t workspace_size(int64_t n,bool parameters) {
    return n>=512?copy_path::workspace_size(n,parameters):0;
}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    if(N>=512) copy_path::launch(in,out,fs,stats,N,stream);
    else feature_path::launch(in,out,fs,stats,N,stream);
}
"""
        source = dict(cpp=copy_source["cpp"], body=cuda, prefix=emitted,
                      includes=(copy_source["include"],), cuda_workspace=True)
    else:
        source = {
            "cpp": feature_source["cpp"],
            "body": feature_source["body"],
            "prefix": emitted,
        }
    return fast_view_source(source, plan, dtype, active_inputs=(0, 1, 3, 4, 5, 6))
