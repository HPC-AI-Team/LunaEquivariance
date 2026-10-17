"""Equivariant normalization forward CUDA."""

import torch

from ....utils.precision import suffix_float32_literals
from .utils import color_segments, fast_view_source, make_binding, make_reduction_prelude


def make_forward_source(plan, dtype):
    block_descriptors = {
        b: "{%d,%d,%d,%d,%d,%d,%d,0,%.17e}" % (
            b.offset, b.channels, b.components, b.param_offset,
            b.bias_offset, b.group, b.center_index, b.weight,
        )
        for b in plan.blocks
    }

    base = make_forward_rows(plan, dtype, row_cap=128)
    copy = base
    word = 2 if dtype == torch.float64 else 1
    budget = 56 if word == 2 else 32
    gang = 256
    for candidate in (32, 64, 128, 256):
        owners, counts = color_segments(plan, candidate)
        if word * sum(counts) <= budget:
            gang = candidate
            break
    owners, (data_slots, parameter_slots, bias_slots) = color_segments(plan, gang)
    if word * (data_slots + parameter_slots + bias_slots) <= budget:
        affine_loads = square_sums = center_subtractions = ""
        resummed_squares = maximum_magnitudes = scaled_squares = affine_outputs = ""
        for owner in owners:
            block = owner["block"]
            affine_loads += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
gamma[{owner["param"]}]=in.v[1].parameter<UnitStride>(b.param_offset+c,1.0);
"""
            if block.bias_offset >= 0:
                affine_loads += f"""beta[{owner["bias"]}]=in.v[2].parameter<UnitStride>(b.bias_offset+c);
"""
            affine_loads += "}while(false);\n"
            if plan.num_centers:
                square_sums += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
const Acc anchor=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0;
"""
                square_sums += f"""#pragma unroll
for(int m=0;m<b.components;++m) {{const Acc delta=in.v[0].get<UnitStride>(row,b.offset+local_index(b,c,m))-anchor;cache[{owner["data"]}+m]=delta; q+=b.weight*delta*delta; if(b.center_index>=0) means[b.center_index]+=delta;}}
}}while(false);
"""
                if block.center_index >= 0:
                    center_subtractions += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
#pragma unroll
for(int m=0;m<b.components;++m) cache[{owner["data"]}+m]-=means[b.center_index];
}}while(false);
"""
                resummed_squares += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
#pragma unroll
for(int m=0;m<b.components;++m) {{const Acc u=cache[{owner["data"]}+m];q+=b.weight*u*u;}}
}}while(false);
"""
            else:
                square_sums += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
const Acc anchor=b.center_index>=0?in.v[0].get<UnitStride>(row,b.offset):0; const Acc mean=b.center_index>=0?means[b.center_index]:0;
"""
                square_sums += f"""#pragma unroll
for(int m=0;m<b.components;++m) {{const Acc u=in.v[0].get<UnitStride>(row,b.offset+local_index(b,c,m))-anchor-mean;cache[{owner["data"]}+m]=u; q+=b.weight*u*u;}}
}}while(false);
"""
            maximum_magnitudes += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
#pragma unroll
for(int m=0;m<b.components;++m) scale=fmax(scale,fabs(cache[{owner["data"]}+m]));
}}while(false);
"""
            scaled_squares += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
#pragma unroll
for(int m=0;m<b.components;++m) {{const Acc u=cache[{owner["data"]}+m]/scale;scaled+=b.weight*u*u;}}
}}while(false);
"""
            affine_outputs += f"""do {{ constexpr Block b={block_descriptors[block]}; if(b.group!=group) break;const int c=lane+({owner["channel_shift"]});const bool owns=lane>={owner["lo"]} && lane<{owner["hi"]};if(!owns) break;
const Acc ga=gamma[{owner["param"]}],be="""
            affine_outputs += f"beta[{owner['bias']}];\n" if block.bias_offset >= 0 else "0;\n"
            affine_outputs += f"""#pragma unroll
for(int m=0;m<b.components;++m) if(out.v[0]) out.v[0][row*D+b.offset+local_index(b,c,m)]=Element(ga*(cache[{owner["data"]}+m]*inv)+be);
}}while(false);
"""

        center_means = "".join(
            f"""if(group=={block.group}) {{const Acc sum=means[{block.center_index}];means[{block.center_index}]=sum/{block.channels};q-={block.weight:.17e}*means[{block.center_index}]*sum;if(lane==0)stats[row*FW+{block.center_index}]=means[{block.center_index}];}}
"""
            for block in plan.blocks if block.center_index >= 0
        )
        cuda = (f"""constexpr int GANG={gang},OWN_THREADS={max(128, gang)},DATA_SLOTS={data_slots},PARAM_SLOTS={parameter_slots},BETA_SLOTS={max(1, bias_slots)};

__device__ __forceinline__ Acc gang_sum(Acc value,Acc* shared) {{
    const int lane=threadIdx.x&31,gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value+=__shfl_down_sync(0xffffffff,value,d);
    if constexpr(GANG==32) return __shfl_sync(0xffffffff,value,0);
    else {{
        if(lane==0) shared[threadIdx.x/32]=value;
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        value=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) value+=shared[gang_id*(GANG/32)+w];
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        return value;
    }}
}}
__device__ __forceinline__ Acc gang_max(Acc value,Acc* shared) {{
    const int lane=threadIdx.x&31,gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value=fmax(value,__shfl_down_sync(0xffffffff,value,d));
    if constexpr(GANG==32) return __shfl_sync(0xffffffff,value,0);
    else {{
        if(lane==0) shared[threadIdx.x/32]=value;
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");value=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) value=fmax(value,shared[gang_id*(GANG/32)+w]);
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");return value;
    }}
}}

__device__ __forceinline__ void gang_sum_squares_and_scalar_sums(Acc& q,Acc* sums,Acc* shared) {{
    const int lane=threadIdx.x&31,gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) {{
        q+=__shfl_down_sync(0xffffffff,q,d);
        #pragma unroll
        for(int k=0;k<NCENTER;++k) sums[k]+=__shfl_down_sync(0xffffffff,sums[k],d);
    }}
    if constexpr(GANG==32) {{
        q=__shfl_sync(0xffffffff,q,0);
        #pragma unroll
        for(int k=0;k<NCENTER;++k) sums[k]=__shfl_sync(0xffffffff,sums[k],0);
    }} else {{
        if(lane==0) {{
            shared[threadIdx.x/32]=q;
            #pragma unroll
            for(int k=0;k<NCENTER;++k) shared[(k+1)*(blockDim.x/32)+threadIdx.x/32]=sums[k];
        }}
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        q=0;
        #pragma unroll
        for(int k=0;k<NCENTER;++k) sums[k]=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) {{
            q+=shared[gang_id*(GANG/32)+w];
            #pragma unroll
            for(int k=0;k<NCENTER;++k) sums[k]+=shared[(k+1)*(blockDim.x/32)+gang_id*(GANG/32)+w];
        }}
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
    }}
}}
template<int T> __global__ __launch_bounds__(T) void copy_forward(Inputs in,Outputs out,Acc* stats,int64_t N,int workers) {{
    __shared__ Acc row_shared[(NCENTER+1)*T/32];
    const int group=blockIdx.y,gang_id=threadIdx.x/GANG,lane=threadIdx.x%GANG;
    Acc cache[DATA_SLOTS],gamma[PARAM_SLOTS],beta[BETA_SLOTS];
{affine_loads}for(int64_t row=int64_t(blockIdx.x)*(T/GANG)+gang_id;row<N;row+=int64_t(workers)*(T/GANG)) {{
Acc means[{max(1, plan.num_centers)}]={{}};
Acc q=0;
{square_sums}gang_sum_squares_and_scalar_sums(q,means,row_shared);const Acc qraw=q;
{center_means}{center_subtractions}if(q<=0.02*qraw || !isfinite(q)) {{q=0;
{resummed_squares}q=gang_sum(q,row_shared); }}
Acc inv=inverse_sqrt(q+{plan.eps:.17e});
if(!isfinite(q+{plan.eps:.17e})) {{Acc scale=0;
{maximum_magnitudes}scale=gang_max(scale,row_shared);Acc scaled=0;
{scaled_squares}scaled=gang_sum(scaled,row_shared);inv=(Acc(1)/scale)*inverse_sqrt(scaled+({plan.eps:.17e}/scale)/scale);}}
if(lane==0) stats[row*FW+NCENTER+group]=inv;
{affine_outputs}
}}
}}
int persistent_capacity() {{
    static thread_local int cached_device=-1,cached=0;int device=0;cudaGetDevice(&device);
    if(device!=cached_device){{int per_sm=0;cudaDeviceProp prop{{}};cudaGetDeviceProperties(&prop,device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,copy_forward<OWN_THREADS>,OWN_THREADS,0);
        cached_device=device;cached=max(1,per_sm*prop.multiProcessorCount);}}
    return cached;
}}
void launch(Inputs in,Outputs out,const Acc*,Acc* stats,int64_t N,cudaStream_t stream) {{
    if(N<=256) copy_forward<GANG><<<dim3(N,NG),GANG,0,stream>>>(in,out,stats,N,int(N));
    else {{const int workers=int(min((N+(OWN_THREADS/GANG)-1)/(OWN_THREADS/GANG),int64_t(persistent_capacity())));
        copy_forward<OWN_THREADS><<<dim3(workers,NG),OWN_THREADS,0,stream>>>(in,out,stats,N,workers);}}
}}
""" if plan.num_centers else f"""constexpr int GANG={gang},OWN_THREADS={max(128, gang)},DATA_SLOTS={data_slots},PARAM_SLOTS={parameter_slots},BETA_SLOTS={max(1, bias_slots)};

__device__ __forceinline__ Acc gang_sum(Acc value,Acc* shared) {{
    const int lane=threadIdx.x&31,gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value+=__shfl_down_sync(0xffffffff,value,d);
    if constexpr(GANG==32) return __shfl_sync(0xffffffff,value,0);
    else {{
        if(lane==0) shared[threadIdx.x/32]=value;
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        value=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) value+=shared[gang_id*(GANG/32)+w];
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");
        return value;
    }}
}}
__device__ __forceinline__ Acc gang_max(Acc value,Acc* shared) {{
    const int lane=threadIdx.x&31,gang_id=threadIdx.x/GANG;
    #pragma unroll
    for(int d=16;d>0;d>>=1) value=fmax(value,__shfl_down_sync(0xffffffff,value,d));
    if constexpr(GANG==32) return __shfl_sync(0xffffffff,value,0);
    else {{
        if(lane==0) shared[threadIdx.x/32]=value;
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");value=0;
        #pragma unroll
        for(int w=0;w<GANG/32;++w) value=fmax(value,shared[gang_id*(GANG/32)+w]);
        asm volatile("bar.sync %0, %1;"::"r"(gang_id+1),"n"(GANG):"memory");return value;
    }}
}}
template<int T> __global__ __launch_bounds__(T) void copy_forward(Inputs in,Outputs out,Acc* stats,int64_t N,int workers) {{
    __shared__ Acc row_shared[T/32];
    const int group=blockIdx.y,gang_id=threadIdx.x/GANG,lane=threadIdx.x%GANG;
    Acc cache[DATA_SLOTS],gamma[PARAM_SLOTS],beta[BETA_SLOTS];
{affine_loads}for(int64_t row=int64_t(blockIdx.x)*(T/GANG)+gang_id;row<N;row+=int64_t(workers)*(T/GANG)) {{
Acc means[{max(1, plan.num_centers)}]={{}};
Acc q=0;
{square_sums}q=gang_sum(q,row_shared);Acc inv=inverse_sqrt(q+{plan.eps:.17e});
if(!isfinite(q+{plan.eps:.17e})) {{Acc scale=0;
{maximum_magnitudes}scale=gang_max(scale,row_shared);Acc scaled=0;
{scaled_squares}scaled=gang_sum(scaled,row_shared);inv=(Acc(1)/scale)*inverse_sqrt(scaled+({plan.eps:.17e}/scale)/scale);}}
if(lane==0) stats[row*FW+NCENTER+group]=inv;
{affine_outputs}
}}
}}
int persistent_capacity() {{
    static thread_local int cached_device=-1,cached=0;int device=0;cudaGetDevice(&device);
    if(device!=cached_device){{int per_sm=0;cudaDeviceProp prop{{}};cudaGetDeviceProperties(&prop,device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,copy_forward<OWN_THREADS>,OWN_THREADS,0);
        cached_device=device;cached=max(1,per_sm*prop.multiProcessorCount);}}
    return cached;
}}
void launch(Inputs in,Outputs out,const Acc*,Acc* stats,int64_t N,cudaStream_t stream) {{
    if(N<=256) copy_forward<GANG><<<dim3(N,NG),GANG,0,stream>>>(in,out,stats,N,int(N));
    else {{const int workers=int(min((N+(OWN_THREADS/GANG)-1)/(OWN_THREADS/GANG),int64_t(persistent_capacity())));
        copy_forward<OWN_THREADS><<<dim3(workers,NG),OWN_THREADS,0,stream>>>(in,out,stats,N,workers);}}
}}
""")
        if dtype == torch.float32:
            cuda = suffix_float32_literals(cuda)
        copy = {"cpp": base["cpp"], "body": cuda, "prefix": base["prefix"]}
    width = max(sum(b.channels * b.components for b in plan.blocks if b.group == g) for g in range(plan.num_groups))
    if width >= 1024:
        shared = make_forward_rows(
            plan, dtype, row_cap=32,
            row_cache_budget=96 if dtype == torch.float64 else 64, warp=True,
        )
        # Low-wave rows retain one CTA each. Bulk rows share affine staging.
        source = "\n" * 8
        for namespace, variant in (("narrow", shared), ("wide", copy)):
            prefix = variant["prefix"]
            body = prefix["constants"] + prefix["helpers"] + variant["body"]
            source += f"""namespace {namespace} {{
{body}
}}
"""
        source += r"""void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
    if(N<=256) wide::launch(in,out,fs,stats,N,stream);
    else narrow::launch(in,out,fs,stats,N,stream);
}
"""
        source = {"cpp": shared["cpp"], "body": source}
    else:
        source = copy
    return fast_view_source(source, plan, dtype, active_inputs=(0, 1, 2))


def make_forward_rows(
    plan, dtype, *, row_cap, row_cache_budget=None,
    warp=False,
):
    """Render row loops and launch choices directly, without rewriting CUDA bodies."""
    emitted = make_reduction_prelude(
        plan, dtype, 0, row_cap=row_cap, row_cache_budget=row_cache_budget,
        warp=warp,
    )
    resources = emitted["resources"]
    row_cache = resources["cache_enabled"]
    cache_fields = resources["cache_fields"]
    cache_slots = resources["cache_slots"]
    sw = resources["stats_width"]
    ns = plan.num_centers
    descriptors = (
        tuple("constexpr Block b=" + value + ";" for value in resources["descriptors"])
        if warp else ("const Block b=block_at(bi);",)
    )
    loop_begin = "\n" if warp else "\n#pragma unroll\nfor(int bi=0;bi<NB;++bi) {"
    block_begin = "do {" if warp else ""
    block_end = "} while(false);\n" if warp else "}"
    thread = "(threadIdx.x&31)" if warp else "threadIdx.x"
    source = f"""__device__ __forceinline__ void row_body(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,int64_t row,int group) {{
    if(row>=N || {thread}>=ROW_THREADS) return;
"""
    if warp:
        width = max(1, ns)
        source += f"""__shared__ Acc warp_means[{4 * width}];
Acc* means=warp_means+(threadIdx.x>>5)*{width};
"""
    else:
        source += f"""__shared__ Acc means[{max(1, ns)}];
"""
    if row_cache:
        for name in cache_fields:
            wide = name in ("u", "h", "k") and plan.num_centers != 0
            source += f"{'Acc' if wide else 'Element'} cache_{name}[{cache_slots}];\n"
    if plan.num_centers:
        source += loop_begin
        for descriptor in descriptors:
            source += block_begin + f"""
    {descriptor}
    if(b.group!=group || b.center_index<0) continue;
Acc val[1]={{0.0}};
const Acc anchor0=in.v[0].get<UnitStride>(row,b.offset);
for(int c={thread};c<b.channels;c+=ROW_THREADS) {{
val[0]+=in.v[0].get<UnitStride>(row,b.offset+c)-anchor0;
}}
reduce<1,ROW_THREADS>(val);
if({thread}==0) means[0*NCENTER+b.center_index]=val[0]/b.channels;
row_sync();
if({thread}==0) stats[row*SW+0*NCENTER+b.center_index]=means[0*NCENTER+b.center_index];
""" + block_end
        source += "\n"
    source += """Acc sums[1]={0.0};
const Acc r=0.0;
"""
    feature_loop = (
        f"\n#pragma unroll\nfor(int slot=0;slot<(b.channels*b.components+ROW_THREADS-1)/ROW_THREADS;++slot) {{\nconst int j={thread}+slot*ROW_THREADS;\nif(j>=b.channels*b.components) continue;\n"
        if row_cache else f"for(int j={thread};j<b.channels*b.components;j+=ROW_THREADS) {{\n"
    )
    source += loop_begin
    for descriptor in descriptors:
        source += block_begin + f" {descriptor} if(b.group!=group) continue;\n"
        source += feature_loop + """const int p=b.param_offset+channel(b,j);
const Acc weight=b.weight;
const Acc u=centered(in.v[0],row,b,j,means,0);
const Acc ga=in.v[1].parameter<UnitStride>(p,1.0);
const Acc beta=b.bias_offset>=0?in.v[2].parameter<UnitStride>(b.bias_offset+channel(b,j)):0.0;
"""
        if row_cache:
            source += "cache_u[b.cache_offset+slot]=u;\n"
        source += "const Acc z=u*r;\n"
        source += "sums[0]+=weight*u*u;\n}" + block_end
    source += f"""
reduce<1,ROW_THREADS>(sums);
Acc inv=inverse_sqrt(sums[0]+{plan.eps:.17e});
// Rare overflow fallback: scaled squares, no feature-sized workspace.
if(!isfinite(sums[0]+{plan.eps:.17e})) {{
    Acc scale=0.0;
    """
    source += loop_begin
    for descriptor in descriptors:
        source += block_begin + f"""
        {descriptor} if(b.group!=group) continue;
        for(int j={thread};j<b.channels*b.components;j+=ROW_THREADS)
            scale=fmax(scale,fabs(centered(in.v[0],row,b,j,means,0)));
    """ + block_end
    source += """
    scale=reduce_max(scale);
    Acc scaled[1]={0.0};
    """ + loop_begin
    for descriptor in descriptors:
        source += block_begin + f"""
        {descriptor} if(b.group!=group) continue;
        for(int j={thread};j<b.channels*b.components;j+=ROW_THREADS) {{
            const Acc value=centered(in.v[0],row,b,j,means,0)/scale;
            scaled[0]+=b.weight*value*value;
        }}
    """ + block_end
    source += f"""
    reduce<1,ROW_THREADS>(scaled);
    inv=__ddiv_rn(1.0,scale)*inverse_sqrt(scaled[0]+({plan.eps:.17e}/scale)/scale);
}}
if({thread}==0) stats[row*SW+NCENTER+group]=inv;
""" + loop_begin
    for descriptor in descriptors:
        source += block_begin + f" {descriptor} if(b.group!=group) continue;\nconst Acc r=inv;\n"
        source += feature_loop + f"""const int p=b.param_offset+channel(b,j);
const Acc weight=b.weight;
const Acc u={"cache_u[b.cache_offset+slot]" if row_cache else "centered(in.v[0],row,b,j,means,0)"};
const Acc ga=in.v[1].parameter<UnitStride>(p,1.0);
const Acc beta=b.bias_offset>=0?in.v[2].parameter<UnitStride>(b.bias_offset+channel(b,j)):0.0;
const Acc z=u*r;
"""
        source += "if(out.v[0]) out.v[0][row*D+b.offset+j]=Element((ga * z + beta));\n}" + block_end
    source += "\n}\n"
    source += r"""__global__ void rows(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N) {"""
    if warp and (plan.param_dim + plan.bias_dim) * (8 if dtype == torch.float64 else 4) <= 24 * 1024:
        source += f"""
__shared__ Element affine_gamma[{max(1, plan.param_dim)}];
__shared__ Element affine_beta[{max(1, plan.bias_dim)}];
if(in.v[1].data)
    for(int p=threadIdx.x;p<P;p+=blockDim.x) affine_gamma[p]=Element(in.v[1].parameter<UnitStride>(p));
if(in.v[2].data)
    for(int p=threadIdx.x;p<BIAS;p+=blockDim.x) affine_beta[p]=Element(in.v[2].parameter<UnitStride>(p));
__syncthreads();
if(in.v[1].data) {{in.v[1].data=affine_gamma;in.v[1].s1=1;in.v[1].width=1;}}
if(in.v[2].data) {{in.v[2].data=affine_beta;in.v[2].s1=1;in.v[2].width=1;}}
"""
    row = "int64_t(blockIdx.x)*4+(threadIdx.x>>5)" if warp else "blockIdx.x"
    source += f"""
    row_body(in,out,fs,stats,N,{row},blockIdx.y);
}}
"""
    source += f"""__global__ void rows_offset(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,int64_t row_start) {{
    row_body(in,out,fs,stats,N,row_start+({row}),blockIdx.y);
}}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {{
    const int64_t blocks={"(N+3)/4" if warp else "N"};
    if(blocks<=2147483647) {{
        rows<<<dim3(blocks,NG),{4 if warp else 1}*ROW_THREADS,0,stream>>>(in,out,fs,stats,N);
    }} else {{
        for(int64_t start=0;start<blocks;) {{
            const int64_t count=min(blocks-start,int64_t(2147483647));
            rows_offset<<<dim3(count,NG),{4 if warp else 1}*ROW_THREADS,0,stream>>>(in,out,fs,stats,N,start*{4 if warp else 1});
            start+=count;
        }}
    }}
}}
"""
    workspace = f"""int64_t workspace_size(int64_t n,bool parameters=true) {{
    return n*{sw};
}}
"""
    host = make_binding(dtype, workspace, stats_width=sw)
    if dtype == torch.float32:
        source = suffix_float32_literals(source.replace("__ddiv_rn", "__fdiv_rn").replace("__dsqrt_rn", "__fsqrt_rn"))
    return {"cpp": host, "body": source, "prefix": emitted}
