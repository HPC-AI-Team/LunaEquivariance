"""UVU forward CUDA templates."""

from collections.abc import Mapping
import torch
from ...utils.cg import weight_channel_offset
from ....utils.precision import cuda_source_for_dtype, float_literal
from ....utils.extension import load_tp_extension


def make_uvu_forward_source(meta: Mapping[str, object]) -> str:
    """Give each warp one output channel tile and traverse its CG paths."""
    fp64 = meta.get("dtype") is torch.float64
    vector_width = 2 if fp64 else 4
    groups = {}
    for spec in meta["generated_specs"]:
        key = spec["out_offset"], spec["channels"], spec["d3"]
        groups.setdefault(key, []).append(spec)
    groups = sorted(groups.items())
    cases, task_index = [], []
    tasks = 0
    max_d3 = max(key[2] for key, _ in groups)
    for group, ((out_offset, channels, d3), specs) in enumerate(groups):
        count = (channels + 31) // 32
        branch = "if" if group == 0 else "else if"
        task_index.append(f"{branch}(task < {tasks + count}) {{ group={group}; u_base=(task-{tasks})*32; }}")
        tasks += count
        lines = [f"case {group}: {{", f"    if(u >= {channels}) return;"]
        lines.extend(f"    float acc{k}=0.f;" for k in range(d3))
        x_groups = {}
        for spec in specs:
            x_groups.setdefault((spec["x_offset"], spec["d1"]), []).append(spec)
        for (x_offset, d1), x_specs in x_groups.items():
            lines.append("    {")
            for i in range(d1):
                lines.append(f"        const float xv{i}=__ldg(x+item_x+{x_offset}+u*{d1}+{i});")
            y_groups = {}
            for spec in x_specs:
                y_groups.setdefault((spec["y_offset"], spec["d2"]), []).append(spec)
            for (y_offset, d2), y_specs in y_groups.items():
                lines.append("        {")
                for j in range(d2):
                    lines.append(f"            const float yv{j}=__ldg(y+item_y+{y_offset+j});")
                for spec in y_specs:
                    lines.append(f"            {{ const float ww=__ldg(weight+item_w+{spec['weight_offset']}+{weight_channel_offset(spec)});")
                    for i, j, k, coefficient in spec["terms"]:
                        lines.append(f"                acc{k}=fmaf({float_literal(coefficient)}*yv{j},ww*xv{i},acc{k});")
                    lines.append("            }")
                lines.append("        }")
            lines.append("    }")
        # Preserve the existing coalesced warp epilogue when vector aligned.
        if channels % 32 == 0 and d3 >= (3 if fp64 else 5) and meta["out_dim"] % 4 == 0 and out_offset % 4 == 0:
            for k in range(d3):
                lines.append(f"    out_tile[warp*32*GEN_MAX_D3+lane*{d3}+{k}]=acc{k};")
            lines.append("    __syncwarp();")
            for iteration in range((d3 + vector_width - 1) // vector_width):
                base = iteration * 32 * vector_width
                remaining = 32*d3-base
                if remaining < 32*vector_width:
                    lines.append(f"    if(lane < {remaining//vector_width}) {{")
                lines.append(f"    const int q{iteration}={base}+lane*{vector_width};")
                lines.append(f"    const float{vector_width} packed{iteration}=*reinterpret_cast<const float{vector_width}*>(out_tile+warp*32*GEN_MAX_D3+q{iteration});")
                lines.append(f"    *reinterpret_cast<float{vector_width}*>(out+item_out+{out_offset}+u_base*{d3}+q{iteration})=packed{iteration};")
                if remaining < 32*vector_width:
                    lines.append("    }")
            lines.append("    __syncwarp();")
        else:
            for k in range(d3):
                lines.append(f"    out[item_out+{out_offset}+u*{d3}+{k}]=acc{k};")
        lines.extend(("    break;", "}"))
        cases.append("\n".join(lines))
    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {{
constexpr int GEN_IN1_DIM={in1_dim};
constexpr int GEN_IN2_DIM={in2_dim};
constexpr int GEN_OUT_DIM={out_dim};
constexpr int GEN_WEIGHT_NUMEL={weight_numel};
constexpr int GEN_MAX_D3={max_d3};
constexpr int GEN_WARP_TASKS={tasks};
constexpr int GEN_THREADS=256;
constexpr int GEN_SHARED_BYTES=GEN_THREADS*GEN_MAX_D3*sizeof(float);

__global__ __launch_bounds__(GEN_THREADS)
void lunatp_uvu_fwd_fp32(const float* __restrict__ x,const float* __restrict__ y,
    const float* __restrict__ weight,float* __restrict__ out,int64_t num_items) {{
    extern __shared__ __align__(16) unsigned char storage[];
    float* out_tile=reinterpret_cast<float*>(storage);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const uint64_t block=uint64_t(blockIdx.y)*gridDim.x+blockIdx.x;
    const uint64_t work=block*(GEN_THREADS/32)+warp;
    const int64_t n=work/GEN_WARP_TASKS;
    if(n>=num_items) return;
    const int task=work%GEN_WARP_TASKS;
    int group=0,u_base=0;
    {task_index}
    const int u=u_base+lane;
    const int64_t item_x=n*GEN_IN1_DIM,item_y=n*GEN_IN2_DIM,item_out=n*GEN_OUT_DIM;
    const int64_t item_w={weight_item};
    switch(group) {{
    {cases}
    }}
}}

torch::Tensor run_forward(torch::Tensor x,torch::Tensor y,torch::Tensor weight) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::{allocator}({{x.size(0),GEN_OUT_DIM}},x.options());
    if(x.size(0)==0) return out;
    const int64_t blocks=(x.size(0)*GEN_WARP_TASKS+GEN_THREADS/32-1)/(GEN_THREADS/32);
    const auto gx=static_cast<unsigned int>(std::min<int64_t>(blocks,2147483647));
    const dim3 grid(gx,static_cast<unsigned int>((blocks+gx-1)/gx));
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    lunatp_uvu_fwd_fp32<<<grid,GEN_THREADS,GEN_SHARED_BYTES,stream>>>(
        x.data_ptr<float>(),y.data_ptr<float>(),weight.data_ptr<float>(),out.data_ptr<float>(),x.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
    cudaFuncAttributes attributes;
    auto status=cudaFuncGetAttributes(&attributes,lunatp_uvu_fwd_fp32);
    TORCH_CHECK(status==cudaSuccess,"UVU forward kernel initialization: ",cudaGetErrorString(status));
    if constexpr(GEN_SHARED_BYTES>48*1024) {{
        status=cudaFuncSetAttribute(lunatp_uvu_fwd_fp32,cudaFuncAttributeMaxDynamicSharedMemorySize,GEN_SHARED_BYTES);
        TORCH_CHECK(status==cudaSuccess,"UVU forward shared memory: ",cudaGetErrorString(status));
    }}
    m.def("lunatp_uvu_fwd_{weight_mode}_fp32",&run_forward);
}}
""".format(
        in1_dim=meta["in1_dim"],in2_dim=meta["in2_dim"],out_dim=meta["out_dim"],
        weight_numel=meta["weight_numel"],max_d3=max_d3,tasks=tasks,
        weight_item="0" if meta["weight_mode"]=="shared" else "n*GEN_WEIGHT_NUMEL",
        task_index="\n".join(task_index),cases="\n".join(cases),weight_mode=meta["weight_mode"],
        allocator="empty" if sum(channels*d3 for (_,channels,d3),_ in groups)==meta["out_dim"] else "zeros",
    )
    return cuda_source_for_dtype(source,meta.get("dtype",torch.float32))


def run(op, x, edge_attr, weight):
    key = (0, True, False)
    launch = op.kernels.get(key)
    if launch is None:
        dtype_name = "fp64" if op.compute_dtype is torch.float64 else "fp32"
        name = f"lunatp_uvu_fwd_{op.weight_mode}_{dtype_name}"
        extension = load_tp_extension(
            name,
            make_uvu_forward_source(op.meta),
            max_channels=op.meta["max_channels"],
            index_bits=op.index_bits,
        )
        launch = op.kernels[key] = getattr(extension, name)
    return launch(x, edge_attr, weight)
