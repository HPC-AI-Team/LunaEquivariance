"""Equivariant normalization triple-backward CUDA."""

import torch

from ....utils.precision import suffix_float32_literals
from .utils import (
    PAIRED_ROUTE_SLOT, TRIPLE_BACKWARD_ACTIVE_INPUTS,
    make_binding, make_reduction_prelude, make_tensor_views,
)


def triple_backward_row_loads(cache_fields, *, cached=False, store=False):
    u = "cache_u[b.cache_offset+slot]" if cached else "centered(in.v[0],row,b,j,means,0)"
    h = "cache_h[b.cache_offset+slot]" if cached else "centered(in.v[4],row,b,j,means,1)"
    k = "cache_k[b.cache_offset+slot]" if cached else "centered(in.v[7],row,b,j,means,2)"
    g = "cache_g[b.cache_offset+slot]" if cached else "in.v[3].get<UnitStride>(row,b.offset+j)"
    sg = "cache_sg[b.cache_offset+slot]" if cached and "sg" in cache_fields else "in.v[10].get<UnitStride>(row,b.offset+j)"
    lines = [
        "const int p=b.param_offset+channel(b,j);",
        "const Acc weight=b.weight;",
        f"const Acc u={u};",
        f"const Acc h={h};",
        f"const Acc k={k};",
        f"const Acc g={g};",
        "const Acc ga=in.v[1].parameter<UnitStride>(p,1.0);",
        "const Acc rg=in.v[5].parameter<UnitStride>(p,0.0);",
        "const Acc sgam=in.v[8].parameter<UnitStride>(p,0.0);",
        f"const Acc sg={sg};",
    ]
    if store:
        lines.extend(f"cache_{name}[b.cache_offset+slot]={name};" for name in cache_fields)
    lines.extend(("const Acc z=u*r,hr=h*r,kr=k*r;", "const Acc t = g * ga;", "const Acc w = sg * ga + g * sgam;"))
    return "\n".join(lines)


def triple_backward_row_iteration(inner, bound, block, offset, resources, thread):
    """Emit a known block's cached slots or the bounded streaming loop."""
    if not resources["cache_enabled"]:
        return f"for(int j={thread};j<{bound};j+=ROW_THREADS) {{\n" + inner + "}"

    row = resources["row_threads"]
    first = offset // row
    end = (offset + block.channels * block.components + row - 1) // row
    source = "\n"
    for slot in range(first, end):
        source += f"""{{ const int j=int({thread})+{slot * row - offset};
constexpr int slot={slot - first};
if(j>=0 && j<{bound}) {{
{inner}}}
}}
"""
    return source


def make_triple_backward_source(plan, dtype):
    """Render single and paired schedules from explicit plan pieces."""
    # r: inverse denominator; z/hr/kr: centered features and input seeds times r.
    # Native output slots: x, gamma, beta, g, rx, rgamma, rbeta.
    dimensions = [0] * plan.num_groups
    offsets = []
    for block in plan.blocks:
        offsets.append(dimensions[block.group])
        dimensions[block.group] += block.channels * block.components
    row = min(128, max(32, 1 << (max(dimensions) - 1).bit_length()))
    slots = (max(dimensions) + row - 1) // row
    word = 1 if dtype == torch.float32 else 2
    retain_sg = dtype == torch.float32 or slots * 5 * word <= 48
    fields = ("u", "h", "k", "g", "sg") if retain_sg else ("u", "h", "k", "g")
    block_group_slots = [0] * plan.num_groups
    for block in plan.blocks:
        block_group_slots[block.group] += (block.channels * block.components + row - 1) // row
    cache_enabled = slots * len(fields) * word <= (96 if dtype == torch.float32 else 160)
    budget = max(block_group_slots) * 5 * word if cache_enabled else 0
    layout = {
        "offsets": tuple(offset // row for offset in offsets),
        "slots": slots,
        "fields": fields,
    }

    prefix = make_reduction_prelude(
        plan,
        dtype,
        3,
        row_cache_budget=budget,
        row_cache_layout=layout,
        workspace_width=0,
    )
    scalar = "float" if dtype == torch.float32 else "double"
    header = (
        f"""#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <optional>
#include <vector>
using Element={scalar};using Acc={scalar};
{make_tensor_views(plan, dtype)}
"""
    )
    launch_declarations = wrapper = registrations = ""
    row = prefix["resources"]["row_threads"]
    maximum = max(block.channels * block.components for block in plan.blocks)
    width = max(dimensions)
    word = 4 if dtype == torch.float32 else 8
    common = max(1, 3 * plan.num_centers) + 32 * (7 + 2 + 1)
    slots = (width + row - 1) // row
    pair_eligible = (
        row == 128
        and prefix["resources"]["cache_enabled"]
        and 2 * 3 * maximum * word <= 32768
        and 2 * (3 * maximum + common) * word <= 48 * 1024
        and slots * 5 * (word // 4) <= 48
        and 2 * (3 * width + common) * word <= 48 * 1024
    )
    variants = []
    for paired in ((False, True) if pair_eligible else (False,)):
        if paired:
            prefix = make_reduction_prelude(
                plan,
                dtype,
                3,
                row_cache_budget=budget,
                row_cache_layout=layout,
                workspace_width=0,
                paired=True,
            )
        resources = prefix["resources"]
        descriptors = resources["descriptors"]
        fields = resources["cache_fields"]
        cached = resources["cache_enabled"]
        maximum = max(block.channels * block.components for block in plan.blocks)
        shared = maximum * 3 * (4 if dtype == torch.float32 else 8) <= 32768
        width = max(dimensions)
        thread = "row_thread()" if paired else "threadIdx.x"
        valid_argument = ",bool row_valid" if paired else ""
        source = f"""__device__ __forceinline__ void row_body(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,int64_t row,int group{valid_argument}) {{
"""
        if not paired:
            source += "    if(row>=N || threadIdx.x>=ROW_THREADS) return;\n"
        if shared:
            source += (
                f"""__shared__ Acc parameter_storage[2*3*{width}];
Acc* parameter_shared=parameter_storage+(int(threadIdx.x)/ROW_THREADS)*3*{width};
"""
                if paired
                else f"""__shared__ Acc parameter_shared[3*{maximum}];
"""
            )
        means = max(1, 3 * plan.num_centers)
        source += (
            f"""__shared__ Acc means_storage[2*{means}];
Acc* means=means_storage+(int(threadIdx.x)/ROW_THREADS)*{means};
"""
            if paired
            else f"""__shared__ Acc means[{means}];
"""
        )
        if cached:
            for name in fields:
                scalar = "Acc" if name in ("u", "h", "k") and plan.num_centers else "Element"
                source += f"{scalar} cache_{name}[{resources['cache_slots']}];\n"
        if plan.num_centers:
            mean_source = "\n"
            for descriptor in descriptors:
                mean_source += f"""do {{
    constexpr Block b={descriptor};
    if(b.group!=group || b.center_index<0) continue;
if({thread}==0) means[b.center_index]=fs[row*FW+b.center_index];
Acc val[2]={{0.0}};
const Acc anchor0=in.v[4].get<UnitStride>(row,b.offset);
const Acc anchor1=in.v[7].get<UnitStride>(row,b.offset);
for(int c={thread};c<b.channels;c+=ROW_THREADS) {{
val[0]+=in.v[4].get<UnitStride>(row,b.offset+c)-anchor0;
val[1]+=in.v[7].get<UnitStride>(row,b.offset+c)-anchor1;
}}
reduce<2,ROW_THREADS>(val);
if({thread}==0) means[1*NCENTER+b.center_index]=val[0]/b.channels;
if({thread}==0) means[2*NCENTER+b.center_index]=val[1]/b.channels;
row_sync();
}} while(false);
"""
            source += mean_source + "\n"
        source += r"""const Acc r=fs[row*FW+NCENTER+group];
Acc sums[7]={0.0};

"""
        for block, descriptor, offset in zip(plan.blocks, descriptors, offsets):
            source += f"""do {{ constexpr Block b={descriptor}; if(b.group!=group) continue;
"""
            inner = triple_backward_row_loads(fields, store=cached) + "\n"
            inner += """sums[0]+=t * z;
sums[1]+=weight * z * hr;
sums[2]+=hr * t + g * rg * z;
sums[3]+=weight * z * kr;
sums[4]+=w * z + t * kr;
sums[5]+=weight * kr * hr;
sums[6]+=hr * w + sg * rg * z + g * rg * kr;
"""
            source += triple_backward_row_iteration(
                inner, "b.channels*b.components", block, offset, resources, thread
            )
            source += "} while(false);\n"
        source += "\nreduce<7,ROW_THREADS>(sums);\n"
        source += """const Acc m0=sums[0];
const Acc m1=sums[1];
const Acc m2=sums[2];
const Acc m3=sums[3];
const Acc m4=sums[4];
const Acc m5=sums[5];
const Acc m6=sums[6];
const Acc c0=-m6 + 3.0 * (m3 * m2 + m4 * m1 + m0 * m5) - 15.0 * m3 * m0 * m1;
const Acc c1=-m5 + 3.0 * m3 * m1;
const Acc c2=-m4 + 3.0 * m3 * m0;
const Acc c3=-m3;
const Acc c4=-m2 + 3.0 * m0 * m1;
const Acc c5=-m1;
const Acc c6=-m0;
"""
        source += "\n"
        zero_scalar = any(block.center_index >= 0 and block.channels == 1 for block in plan.blocks)
        for block, descriptor, offset in zip(plan.blocks, descriptors, offsets):
            source += f"""do {{ constexpr Block b={descriptor}; if(b.group!=group) continue;
"""
            source += r"""Acc projection[2]={0.0};
if(b.center_index>=0) {
"""
            inner = triple_backward_row_loads(fields, cached=cached) + "\n"
            inner += f"""projection[0]+=r * (weight * z * c0 + t * c1 + weight * hr * c2 + g * rg * c3 + weight * kr * c4 + w * c5 + sg * rg);
"""
            inner += f"""projection[1]+=r * (weight * z * c2 + t * c3 + weight * kr * c6 + w);
"""
            source += triple_backward_row_iteration(inner, "b.channels", block, offset, resources, thread)
            source += r"""
reduce<2,ROW_THREADS>(projection);
projection[0]/=b.channels;
projection[1]/=b.channels;
}
"""
            inner = triple_backward_row_loads(fields, cached=cached) + "\n"
            local = r"""const Acc local_a=z*c1+hr*c3+kr*c5;
const Acc local_b=z*c5+hr;
const Acc local_c=z*c3+kr;
"""
            if shared:
                indices = (
                    (f"{offset}+j", f"{width}+{offset}+j", f"2*{width}+{offset}+j")
                    if paired
                    else ("j", f"{maximum}+j", f"2*{maximum}+j")
                )
                guard, zero = ("row_valid ? ", " : Acc(0)") if paired else ("", "")
                local += f"""if(out.v[1] || out.v[5] || out.v[6]) {{
    parameter_shared[{indices[0]}]={guard}g*local_a+sg*local_b{zero};
    parameter_shared[{indices[1]}]={guard}g*local_c+sg*z{zero};
    if(b.bias_offset>=0) parameter_shared[{indices[2]}]={guard}sg{zero};
}}
"""
            else:
                local = "const int c=channel(b,j);\n" + local
                local += r"""if(out.v[1] || out.v[5]) {
    Acc gp=g*local_a+sg*local_b,rp=g*local_c+sg*z;
"""
                if plan.layout == "mul_ir":
                    local += r"""    const unsigned mask=__activemask();
    const int lane=threadIdx.x&31;
    #pragma unroll
    for(int delta=1;delta<min(32,b.components);delta<<=1) {
        const Acc other_gp=__shfl_down_sync(mask,gp,delta),other_rp=__shfl_down_sync(mask,rp,delta);
        const int other_copy=__shfl_down_sync(mask,c,delta);
        if(lane+delta<32 && j+delta<b.channels*b.components && other_copy==c) { gp+=other_gp;rp+=other_rp; }
    }
    if(lane==0 || j%b.components==0) {
"""
                else:
                    local += "    {\n"
                local = (
                    f"""{local}        if(out.v[1]) atomicAdd(out.v[1]+b.param_offset+c,gp);
        if(out.v[5]) atomicAdd(out.v[5]+b.param_offset+c,rp);
    }}
}}
if(out.v[6] && b.bias_offset>=0) atomicAdd(out.v[6]+b.bias_offset+c,sg);
"""
                )
            inner += local
            input_guard = "row_valid && out.v[0]" if paired else "out.v[0]"
            output_guard = "row_valid && out.v[3]" if paired else "out.v[3]"
            direction_guard = "row_valid && out.v[4]" if paired else "out.v[4]"
            scalar_zero = r"""b.center_index>=0 && b.channels==1 ? Element(0) : """ if zero_scalar else ""
            inner += f"""if({input_guard}) out.v[0][row*D+b.offset+j]={scalar_zero}Element((r * (weight * z * c0 + t * c1 + weight * hr * c2 + g * rg * c3 + weight * kr * c4 + w * c5 + sg * rg)) - projection[0]);
if({output_guard}) out.v[3][row*D+b.offset+j]=Element((ga*local_a+sgam*local_b+rg*local_c));
if({direction_guard}) out.v[4][row*D+b.offset+j]={scalar_zero}Element((r * (weight * z * c2 + t * c3 + weight * kr * c6 + w)) - projection[1]);
"""
            source += triple_backward_row_iteration(
                inner, "b.channels*b.components", block, offset, resources, thread
            )
            if shared:
                source += "\n\n" if paired else "\n"
                if not paired:
                    source += f"""if(out.v[1] || out.v[5] || out.v[6]) {{
    row_sync();
    for(int c=threadIdx.x;c<b.channels;c+=ROW_THREADS) {{
        Acc total0=0, total1=0, total2=0;
        #pragma unroll
        for(int m=0;m<b.components;++m) {{
            const int ji=local_index(b,c,m);
            total0+=parameter_shared[ji];
            total1+=parameter_shared[{maximum}+ji];
            if(b.bias_offset>=0) total2+=parameter_shared[2*{maximum}+ji];
        }}
        const int p=b.param_offset+c;
        if(out.v[1]) atomicAdd(out.v[1]+p,total0);
        if(out.v[5]) atomicAdd(out.v[5]+p,total1);
        if(out.v[6] && b.bias_offset>=0) atomicAdd(out.v[6]+b.bias_offset+c,total2);
    }}
    row_sync();
}}
"""
            source += "} while(false);\n"
        source += "\n"
        if paired:
            tail = r"""
if(out.v[1] || out.v[5] || out.v[6]) {
__syncthreads();
"""
            for descriptor, offset in zip(descriptors, offsets):
                tail += f"""do {{ constexpr Block b={descriptor}; if(b.group!=group) continue;
for(int c=int(threadIdx.x);c<b.channels;c+=2*ROW_THREADS) {{
    Acc total0=0,total1=0,total2=0,pair0=0,pair1=0,pair2=0;
    #pragma unroll
    for(int m=0;m<b.components;++m) {{
        const int ji={offset}+local_index(b,c,m);
        total0+=parameter_storage[ji];
        total1+=parameter_storage[{width}+ji];
        pair0+=parameter_storage[3*{width}+ji];
        pair1+=parameter_storage[4*{width}+ji];
        if(b.bias_offset>=0) {{
            total2+=parameter_storage[2*{width}+ji];
            pair2+=parameter_storage[5*{width}+ji];
        }}
    }}
    const int p=b.param_offset+c;
    if(out.v[1]) atomicAdd(out.v[1]+p,total0+pair0);
    if(out.v[5]) atomicAdd(out.v[5]+p,total1+pair1);
    if(out.v[6] && b.bias_offset>=0) atomicAdd(out.v[6]+b.bias_offset+c,total2+pair2);
}}
}} while(false);
"""
            # The final consumer needs no second CTA barrier before all threads return.
            source += tail + "}\n"
        source += "}\n"
        source += f"""__global__ void zero_parameters(Outputs out) {{
    const int p=blockIdx.x*blockDim.x+{thread};
    if(p<P) {{
        if(out.v[1]) out.v[1][p]=Element(0);
        if(out.v[5]) out.v[5][p]=Element(0);
    }}
    if(p<BIAS) {{
        if(out.v[2]) out.v[2][p]=Element(0);
        if(out.v[6]) out.v[6][p]=Element(0);
    }}
}}
__global__ void rows(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N) {{
"""
        if paired:
            source += r"""    const int64_t raw_row=2*int64_t(blockIdx.x)+int(threadIdx.x)/ROW_THREADS;
    const bool valid=raw_row<N;
    const int64_t read_row=valid ? raw_row : raw_row-1;
    row_body(in,out,fs,stats,N,read_row,blockIdx.y,valid);
"""
        else:
            source += r"""    row_body(in,out,fs,stats,N,blockIdx.x,blockIdx.y);
"""
        source += r"""}
__global__ void rows_offset(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,int64_t row_start) {
"""
        if paired:
            source += r"""    const int64_t raw_row=row_start+2*int64_t(blockIdx.x)+int(threadIdx.x)/ROW_THREADS;
    const bool valid=raw_row<N;
    const int64_t read_row=valid ? raw_row : raw_row-1;
    row_body(in,out,fs,stats,N,read_row,blockIdx.y,valid);
"""
        else:
            source += r"""    row_body(in,out,fs,stats,N,row_start+int64_t(blockIdx.x),blockIdx.y);
"""
        source += r"""}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {
if(out.v[1] || out.v[2] || out.v[5] || out.v[6]) zero_parameters<<<(P+127)/128,128,0,stream>>>(out);
"""
        grid, threads = ("(N+1)/2", "2*ROW_THREADS") if paired else ("N", "ROW_THREADS")
        source += f"""const int64_t blocks={grid};
if(blocks<=2147483647) {{
    rows<<<dim3(blocks,NG),{threads},0,stream>>>(in,out,fs,stats,N);
}} else {{
    for(int64_t block_start=0;block_start<blocks;) {{
        const int64_t count=min(blocks-block_start,int64_t(2147483647));
        rows_offset<<<dim3(count,NG),{threads},0,stream>>>(in,out,fs,stats,N,block_start*{2 if paired else 1});
        block_start+=count;
    }}
}}
}}
"""
        row_source = suffix_float32_literals(source) if dtype == torch.float32 else source
        body = prefix["cuda_body_prefix"] + row_source
        source = ""
        for name, fast in (("unit", "true"), ("general", "false")):
            source += f"""namespace {name} {{ constexpr bool UnitStride={fast};
{body}
}}
"""
        condition = (" && ".join(f"""(!in.v[{i}].data || (in.v[{i}].width==1 && in.v[{i}].s1==1))""" for i in TRIPLE_BACKWARD_ACTIVE_INPUTS) or "true")
        source += f"""void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {{
if({condition}) unit::launch(in,out,fs,stats,N,stream);
else general::launch(in,out,fs,stats,N,stream);
}}
"""
        variants.append(source)
    single = variants[0]
    if pair_eligible:
        pair = variants[1]
        cuda = f"""{header}#include <array>
namespace single {{
{single}
}}
"""
        condition = (" && ".join(f"""(!in.v[{i}].data || (in.v[{i}].width==1 && in.v[{i}].s1==1))""" for i in TRIPLE_BACKWARD_ACTIVE_INPUTS) or "true")
        route_source = f"""
std::array<int64_t,9> triple_backward_route(Inputs in,int64_t N) {{
    const bool unit={condition};
    static thread_local int cached_device=-1,cached_sm_count=0;
    static thread_local int cached_blocks[2]={{-1,-1}};
    int device=0,sm_count=0,active_blocks=0;
    cudaError_t error=cudaGetDevice(&device);
    if(error==cudaSuccess && cached_device!=device) {{
        error=cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device);
        if(error==cudaSuccess) {{
            cached_device=device;
            cached_sm_count=sm_count;
            cached_blocks[0]=cached_blocks[1]=-1;
        }}
    }}
    if(error==cudaSuccess) {{
        sm_count=cached_sm_count;
        int& blocks=cached_blocks[unit];
        if(blocks<0) {{
            if(unit) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,single::unit::rows,single::unit::ROW_THREADS,0);
            else error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,single::general::rows,single::general::ROW_THREADS,0);
            if(error==cudaSuccess) blocks=active_blocks;
        }} else active_blocks=blocks;
    }}
    const int64_t capacity=error==cudaSuccess ? int64_t(sm_count)*active_blocks : 0;
    const int64_t grid=N*{plan.num_groups};
    const bool paired=grid>=512 && grid<=2*capacity;
    return {{grid,{plan.num_groups},512,sm_count,active_blocks,capacity,
             paired,unit,int64_t(error)}};
}}
void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {{
    const auto route=triple_backward_route(in,N);
    if(route[{PAIRED_ROUTE_SLOT}]) paired::launch(in,out,fs,stats,N,stream);
    else single::launch(in,out,fs,stats,N,stream);
}}
"""
        cuda += f"""namespace paired {{
{pair}
}}
{route_source}"""
        launch_declarations = "\n#include <array>\nstd::array<int64_t,9> triple_backward_route(Inputs,int64_t);"
        wrapper = r"""std::vector<int64_t> route_info(std::vector<std::optional<torch::Tensor>> inputs) {
    const auto& x=inputs[0].value();
    c10::cuda::CUDAGuard guard(x.device());
    Inputs in{};
    for(size_t i=0;i<inputs.size();++i) in.v[i]=view(inputs[i]);
    const auto route=triple_backward_route(in,x.size(0));
    return std::vector<int64_t>(route.begin(),route.end());
}
"""
        registrations = '\n    m.def("route_info",&route_info);'
    else:
        cuda = header + single
    cpp = make_binding(
        dtype,
        r"""int64_t workspace_size(int64_t n,bool parameters=true) {
    return 0;
}
""",
        stats_width=0,
        launch_declarations=launch_declarations,
        wrapper=wrapper,
        registrations=registrations,
    )
    return {"cpp": cpp, "cuda": cuda}
