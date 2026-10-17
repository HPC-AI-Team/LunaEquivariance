"""Shared tensor views, CUDA reductions, bindings and extension compilation."""

import re

import torch

from ....utils.extension import combine_cuda_sources
from ....utils.jit import load_extension
from ....utils.indexing import name_cuda_kernels
from ....utils.precision import suffix_float32_literals


FEATURE_THREADS = 256

TRIPLE_BACKWARD_ACTIVE_INPUTS = (0, 1, 3, 4, 5, 7, 8, 10)
PAIRED_ROUTE_SLOT = 6


def make_reduction_prelude(
    plan,
    dtype,
    order,
    *,
    row_cap=None,
    row_cache_budget=None,
    row_cache_layout=None,
    workspace_width=None,
    paired=False,
    warp=False,
):
    """Render shared tensor/reduction code and its static row layout.

    Backward and double-backward reuse the reduction prefix. Triple-backward
    also consumes tensor types and helpers for its stride and paired schedules.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("equ_norm supports float32 and float64")
    ns = (1, 1, 2, 3)[order] * plan.num_centers
    saved_count = int(order == 2)
    sw = 0 if order == 1 else ns + plan.num_groups * (saved_count + 1)

    max_group_dim = max(
        sum(b.channels * b.components for b in plan.blocks if b.group == g) for g in range(plan.num_groups)
    )
    row_threads = min(row_cap or 128, max(32, 1 << (max_group_dim - 1).bit_length()))
    offsets = []
    group_slots = [0] * plan.num_groups
    for b in plan.blocks:
        offsets.append(group_slots[b.group])
        group_slots[b.group] += (b.channels * b.components + row_threads - 1) // row_threads
    cache_fields = (("u",), ("u", "g"), ("u", "h", "g"), ("u", "h", "k", "g", "sg"))[order]
    cache_slots = max(group_slots)
    if row_cache_layout is not None:
        offsets = row_cache_layout["offsets"]
        cache_slots = row_cache_layout["slots"]
        cache_fields = row_cache_layout["fields"]

    cache_regs = cache_slots * len(cache_fields) * (2 if dtype == torch.float64 else 1)
    row_cache = cache_regs <= (
        16 if dtype == torch.float64 and order == 0 else 32 if order in (0, 3) else 48
    )
    if row_cache_budget is not None:
        row_cache = cache_regs <= row_cache_budget
    bdesc = [
        "{%d,%d,%d,%d,%d,%d,%d,%d,%.17e}"
        % (
            b.offset,
            b.channels,
            b.components,
            b.param_offset,
            b.bias_offset,
            b.group,
            b.center_index,
            offsets[i],
            b.weight,
        )
        for i, b in enumerate(plan.blocks)
    ]
    block_cases = "\n".join(f"case {i}: return {value};" for i, value in enumerate(bdesc[:-1]))
    thread = "row_thread()" if paired else "(threadIdx.x&31)" if warp else "threadIdx.x"
    row_thread = (
        "__device__ __forceinline__ int row_thread() { return int(threadIdx.x)&(ROW_THREADS-1); }\n"
        if paired
        else ""
    )
    sync_body = (
        '''    const int barrier=1+int(threadIdx.x)/ROW_THREADS;
    asm volatile("bar.sync %0, %1;" :: "r"(barrier), "n"(ROW_THREADS) : "memory");'''
        if paired
        else '''    if constexpr(ROW_THREADS==32) __syncwarp();
    else __syncthreads();'''
    )
    reduce_storage = (
        "__shared__ Acc reduce_storage[2*K*32];\n    Acc* shared=reduce_storage+(int(threadIdx.x)/ROW_THREADS)*K*32;"
        if paired
        else "__shared__ Acc shared[K*32];"
    )
    max_storage = (
        "__shared__ Acc max_storage[2*32];\n    Acc* shared=max_storage+(int(threadIdx.x)/ROW_THREADS)*32;"
        if paired
        else "__shared__ Acc shared[32];"
    )
    block_sync = "row_sync();" if paired else "__syncthreads();"
    constants = f"""constexpr int D={plan.dim}, P={plan.param_dim}, BIAS={plan.bias_dim};
constexpr int NB={len(plan.blocks)}, NG={plan.num_groups}, NCENTER={plan.num_centers};
constexpr int FW={(plan.num_centers + plan.num_groups)}, SW={sw if workspace_width is None else workspace_width}, THREADS={max(128, row_threads)}, ROW_THREADS={row_threads};
constexpr bool IR_MUL={str(plan.layout == "ir_mul").lower()};
struct Block {{ int offset, channels, components, param_offset, bias_offset, group, center_index, cache_offset; Acc weight; }};
__device__ __forceinline__ Block block_at(int i) {{
    switch(i) {{ {block_cases} default: return {bdesc[-1]}; }}
}}
"""
    maximum = r"""__device__ __forceinline__ Acc reduce_max(Acc value) {
    #pragma unroll
    for(int d=16;d>0;d>>=1) value=fmax(value,__shfl_down_sync(0xffffffff,value,d));
    return __shfl_sync(0xffffffff,value,0);
}
""" if warp else f"""__device__ __forceinline__ Acc reduce_max(Acc value) {{
    {max_storage}
    const int lane={thread}&31, warp={thread}>>5;
    for(int d=16;d>0;d>>=1) value=fmax(value,__shfl_down_sync(0xffffffff,value,d));
    if(lane==0) shared[warp]=value;
    row_sync();
    if(warp==0) {{
        value=lane<ROW_THREADS/32?shared[lane]:0.0;
        for(int d=ROW_THREADS/64;d>0;d>>=1) value=fmax(value,__shfl_down_sync(0xffffffff,value,d));
        if(lane==0) shared[0]=value;
    }}
    row_sync();
    return shared[0];
}}
"""
    helpers = f"""{row_thread}__device__ __forceinline__ int channel(const Block& b,int j) {{ return IR_MUL ? j%b.channels : j/b.components; }}
__device__ __forceinline__ int local_index(const Block& b,int c,int m) {{ return IR_MUL ? m*b.channels+c : c*b.components+m; }}
__device__ __forceinline__ Acc inverse_sqrt(Acc x) {{ return __ddiv_rn(1.0,__dsqrt_rn(x)); }}
__device__ __forceinline__ void row_sync() {{
{sync_body}
}}
template<int K,int Workers=THREADS> __device__ __forceinline__ void reduce(Acc (&v)[K]) {{
    if constexpr(Workers==32) {{
        #pragma unroll
        for(int k=0;k<K;++k) {{
            #pragma unroll
            for(int d=16;d>0;d>>=1) v[k]+=__shfl_down_sync(0xffffffff,v[k],d);
            v[k]=__shfl_sync(0xffffffff,v[k],0);
        }}
    }} else {{
    {reduce_storage}
    const int lane={thread}&31, warp={thread}>>5;
    #pragma unroll
    for(int k=0;k<K;++k) {{
        #pragma unroll
        for(int d=16;d>0;d>>=1) v[k]+=__shfl_down_sync(0xffffffff,v[k],d);
        if(lane==0) shared[k*32+warp]=v[k];
    }}
    {block_sync}
    if(warp==0) {{
        #pragma unroll
        for(int k=0;k<K;++k) {{
            Acc t=lane<Workers/32?shared[k*32+lane]:0.0;
            #pragma unroll
            for(int d=Workers/64;d>0;d>>=1) t+=__shfl_down_sync(0xffffffff,t,d);
            if(lane==0) shared[k*32]=t;
        }}
    }}
    {block_sync}
    #pragma unroll
    for(int k=0;k<K;++k) v[k]=shared[k*32];
    }}
}}
{maximum}__device__ __forceinline__ Acc centered(View v,int64_t row,const Block& b,int j,const Acc* means,int mean_slot) {{
    const Acc val=v.get<UnitStride>(row,b.offset+j);
    if(b.center_index<0 || !v.data) return val;
    return (val-v.get<UnitStride>(row,b.offset))-means[mean_slot*NCENTER+b.center_index];
}}
"""
    if dtype == torch.float32:
        bdesc = [suffix_float32_literals(descriptor) for descriptor in bdesc]
        constants = suffix_float32_literals(constants)
        helpers = suffix_float32_literals(
            helpers.replace("__ddiv_rn", "__fdiv_rn").replace("__dsqrt_rn", "__fsqrt_rn")
        )
    return {
        "constants": constants,
        "helpers": helpers,
        # Keep the rendered source stable for content-addressed caching.
        "cuda_body_prefix": "\n" * 7 + constants + "\n" + helpers,
        "resources": {
            "row_threads": row_threads,
            "cache_enabled": row_cache,
            "descriptors": tuple(bdesc),
            "stats_width": sw,
            "cache_fields": cache_fields,
            "cache_slots": cache_slots,
        },
    }


def make_binding(
    dtype,
    workspace_definition,
    *,
    stats_width,
    parameter_outputs=(),
    launch_declarations="",
    wrapper="",
    registrations="",
):
    """Render the native ABI and the chosen workspace contract directly."""
    scalar = "float" if dtype == torch.float32 else "double"
    parameter_guard = " || ".join(f"out.v[{i}]" for i in parameter_outputs) or "false"
    host = f"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>
using Element = {scalar};
using Acc = {scalar};
struct View {{ const Element* data; int64_t s0,s1,s2,width; }};
struct Inputs {{ View v[11]; }};
struct Outputs {{ Element* v[7]; }};
void launch(Inputs,Outputs,const Acc*,Acc*,int64_t,cudaStream_t);{launch_declarations}
{workspace_definition}"""
    host += """View view(const std::optional<torch::Tensor>& t) {
    if(!t.has_value()) return {nullptr,0,0,0,1};
    const auto& a=t.value();
    if(a.dim()==1) return {a.data_ptr<Element>(),0,a.stride(0),0,1};
    if(a.dim()==2) return {a.data_ptr<Element>(),a.stride(0),a.stride(1),0,1};
    return {a.data_ptr<Element>(),a.stride(0),a.stride(1),a.stride(2),a.size(2)};
}
void run(std::vector<std::optional<torch::Tensor>> inputs,
         std::vector<std::optional<torch::Tensor>> outputs,
         torch::Tensor fwd_stats,torch::Tensor workspace) {
    const auto& x=inputs[0].value();
    c10::cuda::CUDAGuard guard(x.device());
    Inputs in{}; Outputs out{};
    for(size_t i=0;i<inputs.size();++i) in.v[i]=view(inputs[i]);
    for(size_t i=0;i<outputs.size();++i) out.v[i]=outputs[i].has_value()?outputs[i]->data_ptr<Element>():nullptr;
    const int64_t N=x.size(0);
    if(N==0) {
        for(auto& a:outputs) if(a.has_value()) a->zero_();
        return;
    }
    auto stream=at::cuda::getCurrentCUDAStream();
"""
    host += f"""TORCH_CHECK(workspace.numel()>=workspace_size(N,{parameter_guard}),"equ_norm workspace is too small");
    launch(in,out,fwd_stats.data_ptr<Acc>(),workspace.data_ptr<Acc>(),N,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
{wrapper}PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{{registrations}
    m.def("run",&run); m.attr("stats_width")={stats_width};
    m.def("workspace_size",&workspace_size,pybind11::arg("n"),pybind11::arg("parameters")=true);
}}
"""
    return host


def color_segments(plan, gang, component_lanes=1):
    result = []
    maxima = [0, 0, 0]
    for group in range(plan.num_groups):
        owners, cursor = [], 0
        for block in plan.blocks:
            if block.group != group:
                continue
            for start in range(
                (cursor // gang) * gang,
                (cursor + block.channels + gang - 1) // gang * gang,
                gang,
            ):
                left, right = (
                    max(cursor, start),
                    min(cursor + block.channels, start + gang),
                )
                if right <= left:
                    continue
                owners.append(
                    dict(
                        block=block,
                        lo=left - start,
                        hi=right - start,
                        channel_shift=start - cursor,
                    )
                )
            cursor += block.channels

        allocated = [[], [], []]
        for owner in owners:
            b = owner["block"]
            for kind, width in enumerate(
                (
                    (b.components + component_lanes - 1) // component_lanes,
                    1,
                    1 if b.bias_offset >= 0 else 0,
                )
            ):
                offset = 0
                if width:
                    while any(
                        owner["lo"] < rhi
                        and rlo < owner["hi"]
                        and offset < roff + rw
                        and roff < offset + width
                        for rlo, rhi, roff, rw in allocated[kind]
                    ):
                        offset += 1
                    allocated[kind].append((owner["lo"], owner["hi"], offset, width))
                    maxima[kind] = max(maxima[kind], offset + width)
                owner[("data", "param", "bias")[kind]] = offset
            result.append(owner)
    return result, maxima


def make_tensor_views(plan, dtype):
    """The native input layout shared by every CUDA specialization."""
    input_c = plan.blocks[0].channels
    uniform_channels = all(b.channels == input_c for b in plan.blocks)
    views = f"""struct View {{
    const Element* data; int64_t s0,s1,s2,width;
    template<bool Unit> __device__ __forceinline__ Acc get(int64_t row,int j,Acc missing=0.0) const {{
if constexpr(Unit) return data?Acc(data[row*s0+j]):missing;
else {{
        if(!data) return missing;
        if({str(not uniform_channels).lower()} || width==1) return Acc(data[row*s0+int64_t(j)*s1]);
        return Acc(data[row*s0+(j/{input_c if uniform_channels else "width"})*s1+(j%{input_c if uniform_channels else "width"})*s2]);
    }}
}}
    template<bool Unit> __device__ __forceinline__ Acc parameter(int j,Acc missing=0.0) const {{
if constexpr(Unit) return data?Acc(data[j]):missing;
else {{
        return data?Acc(data[int64_t(j)*s1]):missing;
    }}
}}
}};
struct Inputs {{ View v[11]; }};
struct Outputs {{ Element* v[7]; }};"""
    return suffix_float32_literals(views) if dtype == torch.float32 else views


def fast_view_source(source, plan, dtype, active_inputs=None):
    """Assemble tensor types, row kernels and stride dispatch."""
    scalar = "float" if dtype == torch.float32 else "double"
    views = make_tensor_views(plan, dtype)
    includes = (
        *source.get("includes", ()),
        "#include <cuda.h>",
        "#include <cuda_runtime.h>",
        "#include <cmath>",
        "#include <optional>",
        "#include <vector>",
        *source.get("trailing_includes", ()),
    )
    prefix = source.get("prefix")
    body = source["body"]
    if prefix is not None:
        body = "\n" * len(source.get("includes", ())) + prefix["cuda_body_prefix"] + body
    result = (
        "\n".join(dict.fromkeys(includes)) + "\n"
        + f"using Element={scalar};using Acc={scalar};\n" + views + "\n"
    )
    for name, fast in (("unit", "true"), ("general", "false")):
        result += f"namespace {name} {{ constexpr bool UnitStride={fast};\n" + body + "\n}\n"

    if active_inputs is None:
        active_inputs = tuple(range(11))
    predicate = (" && ".join(f"(!in.v[{i}].data || (in.v[{i}].width==1 && in.v[{i}].s1==1))" for i in active_inputs) or "true")
    result += (
        "void launch(Inputs in,Outputs out,const Acc* fs,Acc* stats,int64_t N,cudaStream_t stream) {\n"
        + f"if({predicate}) unit::launch(in,out,fs,stats,N,stream);\nelse general::launch(in,out,fs,stats,N,stream);\n}}\n"
    )
    if source.get("cuda_workspace", False):
        result += "int64_t workspace_size(int64_t n,bool parameters) { return max(unit::workspace_size(n,parameters),general::workspace_size(n,parameters)); }\n"
    return {"cpp": source["cpp"], "cuda": result}



def load_kernel(plan, order, dtype, arch):
    from .forward import make_forward_source
    from .backward import make_backward_source
    from .double_backward import make_double_backward_source
    from .triple_backward import make_triple_backward_source

    if order < 2:
        # Keep the two translation units and CUDA bodies intact. Namespaces
        # separate their launchers and workspace contracts in one extension.
        sources = {
            "forward": make_forward_source(plan, dtype),
            "backward": make_backward_source(plan, dtype),
        }
        source = {"cpp": combine_cuda_sources({name: code["cpp"] for name, code in sources.items()})}
        includes, bodies = [], []
        for name, emitted in sources.items():
            code = name_cuda_kernels(emitted["cuda"], f"equ_norm_{name}", index_bits=64)
            includes.extend(re.findall(r"^\s*#include[^\n]*", code, flags=re.M))
            body = re.sub(r"^\s*#include[^\n]*", "", code, flags=re.M)
            bodies.append(f"namespace {name} {{\n{body}\n}}\n")
        source["cuda"] = "\n".join(dict.fromkeys(includes)) + "\n" + "\n".join(bodies)
    else:
        make_source = make_double_backward_source if order == 2 else make_triple_backward_source
        source = make_source(plan, dtype)
        phase = "double_backward" if order == 2 else "triple_backward"
        source["cuda"] = name_cuda_kernels(source["cuda"], f"equ_norm_{phase}", index_bits=64)

    return load_extension(
        "equ_norm",
        cpp_sources=source["cpp"], cuda_sources=source["cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "--prec-div=true", "--prec-sqrt=true",
            f"-gencode=arch=compute_{arch[0]}{arch[1]},code=sm_{arch[0]}{arch[1]}",
        ],
        no_implicit_headers=True,
    )
