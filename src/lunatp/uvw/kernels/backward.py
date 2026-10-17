"""UVW backward CUDA templates."""

from operator import itemgetter

import torch

from ...utils.runtime import DERIVATIVE_STREAM_SOURCE
from ...utils.cg import emit_cg_source
from ....utils.precision import cuda_source_for_dtype, float_literal, get_fp64_mma
from ....utils.extension import get_cuda_arch, load_cuda_extension
from ...utils.contractions import INPUTS, PROGRAMS, OUTPUT_SHAPES
from .forward import make_uvw_forward_external_kernel, make_uvw_forward_shared_kernel
from .utils import (SCALAR_CUDA_SOURCE, SCALAR_PARALLEL_CUDA_SOURCE,
                    UVW_SCALAR_WORKSPACE_ELEMENTS)


def make_uvw_grouped_backward_launch(meta, groups, workspace_elements, x_terms, y_terms, w_terms, forward_terms):
    """Combine angular paths sharing scalar features into wider FP32 GEMMs."""
    XD, YD, GD = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    contraction_groups = {}
    for out, terms in x_terms.items():
        multipliers = {}
        for y, w, g in terms:
            multipliers.setdefault(g, []).append((y, w))
        for g, pairs in multipliers.items():
            contraction_groups.setdefault(tuple(sorted(pairs)), []).append((out, g))
    multipliers = {}
    for x, y, w in forward_terms:
        multipliers.setdefault(x, []).append((y, w))
    for x, pairs in multipliers.items():
        contraction_groups.setdefault(tuple(sorted(pairs)), []).append(("gg", x))
    product_groups = {}
    for out, terms in y_terms.items():
        multipliers = {}
        for x, w, g in terms:
            multipliers.setdefault(w, []).append((x, g))
        for w, pairs in multipliers.items():
            product_groups.setdefault(tuple(pairs), []).append((out, w, False))
    for out, terms in w_terms.items():
        multipliers = {}
        for x, y, g in terms:
            multipliers.setdefault(y, []).append((x, g))
        for y, pairs in multipliers.items():
            product_groups.setdefault(tuple(pairs), []).append((out, y, True))
    weight_names = sorted({w for pairs in contraction_groups for _, w in pairs}
                          | {multiplier for consumers in product_groups.values() for _, multiplier, is_weight in consumers if not is_weight})
    last_weight_group = {}
    for pairs, consumers in product_groups.items():
        for out, multiplier, weight in consumers:
            if weight: last_weight_group[out] = (pairs, multiplier)
    code = ""
    for (yo, V), specs in groups.items():
        width = sum(s["path_channels"] // V * s["out_channels"] for s in specs)
        tile_v = min(128, ((V + 63) // 64) * 64)
        warp_v = min(tile_v, 64)
        blocks_per_sm = 128 // tile_v
        chunk = max(1, (workspace_elements // width // 128) * 128)
        code += "{\n"
        for w in weight_names:
            code += f"auto packed_{w}=at::empty({{{V*width}}},x.options());\n"
        for out in w_terms:
            code += f"auto packed_{out}=at::zeros({{{V*width}}},x.options());\n"
        for w in weight_names:
            code += f"lunaeq_uvw_bwd_pack_group_{yo}_{V}_fp32<<<{min(4096,(V*width+255)//256)},256,0,stream>>>({w}.data_ptr<float>(),packed_{w}.data_ptr<float>());\n"
        overlap = bool(contraction_groups and product_groups)
        if overlap:
            code += "auto& lane=derivative_stream(x.get_device(),stream); DerivativeFork fork(lane,stream); auto dx_stream=lane.stream;\n"
        else:
            code += "auto dx_stream=stream;\n"
        if product_groups:
            code += f"const int split_capacity=std::max(1,std::min(sm_count*4,std::max({width//256},{chunk//256}))); auto split_memory=at::empty({{int64_t(uvw_grouped::Table::bytes(split_capacity))}},x.options().dtype(at::kByte)); uvw_grouped::Table split_table(split_memory.data_ptr(),split_capacity);\n"
            code += f"const int64_t initial_count=std::min(items,int64_t({chunk})); const int sm_target=sm_count*4;\n"
            code += f"const int64_t dy_reserve=std::min(int64_t({V*(width//256)})*initial_count,int64_t({V})*((int64_t(sm_target)*128+{(V+63)//64}-1)/{(V+63)//64}+initial_count));\n"
            code += f"const int64_t dw_reserve=initial_count/256<2 ? 0 : int64_t({width*V})*std::min((sm_target+{((V+63)//64)*((width+127)//128)}-1)/{((V+63)//64)*((width+127)//128)},int(initial_count/256));\n"
            code += f"const int64_t reserve=std::min(h_elements/2,std::max({('dy_reserve' if y_terms else 'int64_t(0)')},{('dw_reserve' if w_terms else 'int64_t(0)')})); const int64_t angular_chunk=std::max(int64_t(1),std::min(int64_t({chunk}),(h_elements-reserve)/{width}));\n"
        else:
            code += f"const int64_t angular_chunk={chunk};\n"
        code += "for(int64_t start=0;start<items;start+=angular_chunk) {int count=int(std::min(angular_chunk,items-start));\n"
        if w_terms:
            code += f"const int dw_capacity=std::max(1,std::min(count/256,int((h_elements-int64_t(count)*{width})/{V*width}))); const int dw_parts=uvw_grouped::gemm_splits({((V+tile_v-1)//tile_v)*((width+255)//256)},count,dw_capacity,sm_count*{blocks_per_sm},sm_target);\n"
        if y_terms:
            code += f"const int dy_capacity=std::max(1,std::min({width//256},int((h_elements-int64_t(count)*{width})/(int64_t(count)*{V})))); const bool dy_wide=int64_t((count+255)/256)*{((V+63)//64)}*dy_capacity>=sm_count; const int dy_rows=dy_wide ? 256 : 128; const int dy_parts=uvw_grouped::gemm_splits({((V+63)//64)}*((count+dy_rows-1)/dy_rows),{width},dy_capacity,sm_target/2,sm_target);\n"
        for pairs, consumers in product_groups.items():
            overlap_product = any(weight for _, _, weight in consumers) and any(not weight for _, _, weight in consumers)
            dy_stream = "dy_stream" if overlap_product else "stream"
            if overlap_product:
                code += "{\n"
            x, g = pairs[0]
            x1, g1 = pairs[-1]
            final_weights = [(out,multiplier) for out,multiplier,weight in consumers if weight and last_weight_group[out]==(pairs,multiplier)]
            if all(s["out_channels"] % 4 == 0 for s in specs):
                code += f"lunaeq_uvw_bwd_product_group_{yo}_{V}_fp32<{len(pairs)}><<<(int64_t(count)*{width//4}+255)/256,256,0,stream>>>({x}.data_ptr<float>(),{g}.data_ptr<float>(),{x1}.data_ptr<float>(),{g1}.data_ptr<float>(),h_buffer.data_ptr<float>(),start,count);\n"
            else:
                offset = 0
                for s in specs:
                    U, O, D = s["path_channels"] // V, s["out_channels"], s["d1"]
                    xo, oo = s["x_offset"], s["out_offset"]
                    scale = float_literal(s["terms"][0][3], torch.float32)
                    code += f"lunaeq_uvw_bwd_product_fp32<{U},{O},{D},{XD},{GD},{len(pairs)},{width},{offset}><<<(int64_t(count)*{U*((O+3)//4)}+255)/256,256,0,stream>>>({x}.data_ptr<float>()+{xo},{g}.data_ptr<float>()+{oo},{x1}.data_ptr<float>()+{xo},{g1}.data_ptr<float>()+{oo},h_buffer.data_ptr<float>()+{offset},start,count,{scale});\n"
                    offset += U*O
            if overlap_product:
                single_tile = 64 if V <= 64 and width <= 64*65535 else 128
                single_tiles = ((width+single_tile-1)//single_tile)*((V+single_tile-1)//single_tile)
                code += "std::optional<DerivativeFork> dy_fork; cudaStream_t dy_stream=stream;\n"
                code += f"if(dw_parts==1 && int64_t(count)*{single_tiles}>=int64_t(sm_count)*{single_tile}) {{ auto& lane=derivative_stream(x.get_device(),stream,1); dy_fork.emplace(lane,stream); dy_stream=lane.stream; }}\n"
            for out, multiplier, weight_gradient in consumers:
                if weight_gradient:
                    final_sum = f"lunaeq_uvw_bwd_weight_sum_{yo}_{V}_fp32<<<(int64_t({width})*{V}+255)/256,256,0,stream>>>"
                    code += f"{{ int splits=dw_parts; if(splits==1) {{ scalar_gemm(stream,false,true,{width},{V},count,h_buffer.data_ptr<float>(),{width},{multiplier}.data_ptr<float>()+start*{YD}+{yo},{YD},packed_{out}.data_ptr<float>(),{width},1.f);\n"
                    if (out,multiplier) in final_weights:
                        code += f"if(start+count==items) {final_sum}(nullptr,packed_{out}.data_ptr<float>(),{out}.data_ptr<float>(),0);\n"
                    code += f"}} else {{ int span=((count+splits-1)/splits+3)/4*4; splits=(count+span-1)/span; float* partial=h_buffer.data_ptr<float>()+int64_t(count)*{width};\n"
                    code += f"lunaeq_uvw_bwd_split_table_fp32<false,true><<<1,128,0,stream>>>(split_table,splits,{width},{V},count,span,h_buffer.data_ptr<float>(),{width},{multiplier}.data_ptr<float>()+start*{YD}+{yo},{YD},partial);\n"
                    kernel = f"uvw_grouped::VectorNT<cutlass::layout::NoPermute,{tile_v},256,{warp_v},64>::Kernel" if V%4==0 and width%4==0 and YD%4==0 and yo%4==0 else f"uvw_grouped::Kernel<cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,cutlass::layout::RowMajor,{tile_v},256,8,cutlass::layout::NoPermute,{warp_v},64>"
                    launch = f"uvw_grouped::launch<{kernel}>(split_table,splits,sm_count*{blocks_per_sm},1.f,0.f,stream)"
                    if kernel.startswith("uvw_grouped::VectorNT"):
                        launch = f"((reinterpret_cast<uintptr_t>({multiplier}.data_ptr<float>())&15)==0 ? {launch} : uvw_grouped::launch<uvw_grouped::VectorNT<cutlass::layout::NoPermute,{tile_v},256,{warp_v},64>::Base>(split_table,splits,sm_count*{blocks_per_sm},1.f,0.f,stream))"
                    code += f"auto status={launch}; TORCH_CHECK(status==cutlass::Status::kSuccess,\"CUTLASS angular DW failed\");\n"
                    if (out,multiplier) in final_weights:
                        code += f"if(start+count==items) {final_sum}(partial,packed_{out}.data_ptr<float>(),{out}.data_ptr<float>(),splits); else "
                    code += f"lunaeq_uvw_bwd_split_sum_fp32<<<({width*V}+255)/256,256,0,stream>>>(partial,packed_{out}.data_ptr<float>(),{width},{V},{width},splits); }} }}\n"
                else:
                    code += f"{{ int splits=dy_parts; if(splits==1) scalar_gemm({dy_stream},true,false,{V},count,{width},packed_{multiplier}.data_ptr<float>(),{width},h_buffer.data_ptr<float>(),{width},{out}.data_ptr<float>()+start*{YD}+{yo},{YD},1.f); else {{ int span=(({width}+splits-1)/splits+3)/4*4; splits=({width}+span-1)/span; float* partial=h_buffer.data_ptr<float>()+int64_t(count)*{width};\n"
                    code += f"lunaeq_uvw_bwd_split_table_fp32<true,false><<<1,128,0,{dy_stream}>>>(split_table,splits,{V},count,{width},span,packed_{multiplier}.data_ptr<float>(),{width},h_buffer.data_ptr<float>(),{width},partial);\n"
                    code += f"using G=uvw_grouped::Kernel<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,128,64,8,cutlass::layout::NoPermute,64,32>; using Wide=uvw_grouped::Kernel<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,256,64,8,cutlass::layout::NoPermute,64,64>; auto status=dy_wide ? uvw_grouped::launch<Wide>(split_table,splits,sm_count*2,1.f,0.f,{dy_stream}) : uvw_grouped::launch<G>(split_table,splits,sm_count*2,1.f,0.f,{dy_stream}); TORCH_CHECK(status==cutlass::Status::kSuccess,\"CUTLASS angular dy failed\");\n"
                    code += f"lunaeq_uvw_bwd_split_sum_fp32<<<(int64_t(count)*{V}+255)/256,256,0,{dy_stream}>>>(partial,{out}.data_ptr<float>()+start*{YD}+{yo},{V},count,{YD},splits); }} }}\n"
            if overlap_product:
                code += "if(dy_fork) dy_fork->join();\n}\n"
        for pairs, consumers in contraction_groups.items():
            for index, (y,w) in enumerate(pairs):
                beta = "0.f" if index == 0 else "1.f"
                code += f"uvw_grouped::scalar_nn(dx_stream,{width},count,{V},packed_{w}.data_ptr<float>(),{width},{y}.data_ptr<float>()+start*{YD}+{yo},{YD},a_buffer.data_ptr<float>(),{width},{beta});\n"
            offset = 0
            for s in specs:
                U, O, D = s["path_channels"] // V, s["out_channels"], s["d1"]
                xo, oo = s["x_offset"], s["out_offset"]
                scale = float_literal(s["terms"][0][3], torch.float32)
                for out, multiplier in consumers:
                    if out == "gg":
                        code += f"lunaeq_uvw_bwd_output_fp32<{U},{O},{D},{XD},{GD},{width}><<<(int64_t(count)*{O}+255)/256,256,0,dx_stream>>>(a_buffer.data_ptr<float>()+{offset},{multiplier}.data_ptr<float>()+{xo},gg.data_ptr<float>()+{oo},start,count,{scale});\n"
                    else:
                        code += f"lunaeq_uvw_bwd_dx_fp32<{U},{O},{D},{XD},{GD},{width}><<<(int64_t(count)*{U*8}+255)/256,256,0,dx_stream>>>(a_buffer.data_ptr<float>()+{offset},{multiplier}.data_ptr<float>()+{oo},{out}.data_ptr<float>()+{xo},start,count,{scale});\n"
                offset += U*O
        code += "}\n"
        if overlap:
            code += "fork.join();\n"
        code += "}\n"
    return code


def make_uvw_scalar_backward_source(meta, program, inputs, outputs, phase):
    """Contract scalar features before angular components; recompute bounded tiles."""
    # Expand each requested derivative into scalar-input contractions at compile
    # time. The CUDA launcher reuses y*W and x*g tiles across these terms.
    x_terms = {
        target: tuple(
            (y, weight, grad)
            for y, z in program["x"][target]
            for weight, grad in program["z"][z]
        )
        for target in outputs
        if target in program["x"]
    }
    y_terms = {
        target: tuple(
            (x, weight, grad)
            for x, z in program["y"][target]
            for weight, grad in program["z"][z]
        )
        for target in outputs
        if target in program["y"]
    }
    w_terms = {
        target: tuple(
            (x, y, grad)
            for q, grad in program["w"][target]
            for x, y in program["cg"][q]
        )
        for target in outputs
        if target in program["w"]
    }
    forward_terms = (
        tuple(
            (x, y, weight)
            for q, weight in program["forward"]
            for x, y in program["cg"][q]
        )
        if "gg" in outputs
        else ()
    )
    specs = meta["generated_specs"]
    max_width = max(
        (
            s["path_channels"]
            if s["d1"] == 1
            else s["path_channels"] // s["edge_channels"] * s["out_channels"]
        )
        for s in specs
    )
    path_groups = {}
    for spec in specs:
        if spec["d1"] > 1:
            path_groups.setdefault((spec["y_offset"], spec["edge_channels"]), []).append(spec)
    max_width = max([max_width] + [sum(s["path_channels"] // v * s["out_channels"] for s in group) for (_, v), group in path_groups.items()])
    workspace_elements = max(max_width, UVW_SCALAR_WORKSPACE_ELEMENTS)
    code = SCALAR_CUDA_SOURCE + (SCALAR_PARALLEL_CUDA_SOURCE if forward_terms else "") + r"""
#include <optional>
template<bool TransposeA,bool TransposeB>
__global__ void lunaeq_uvw_bwd_split_table_fp32(uvw_grouped::Table table,int splits,int m,int n,int k,int span,const float* a,int lda,const float* b,int ldb,float* partial) {
    for(int i=threadIdx.x;i<splits;i+=blockDim.x) {
        int first=i*span, count=min(span,k-first);
        const float* aa=a+(TransposeA ? int64_t(first) : int64_t(first)*lda);
        const float* bb=b+(TransposeB ? int64_t(first)*ldb : int64_t(first));
        float* out=partial+int64_t(i)*m*n;
        table.set(i,n,m,count,bb,aa,out,out,ldb,lda,m,m);
    }
}
__global__ void lunaeq_uvw_bwd_split_sum_fp32(const float* partial,float* out,int m,int n,int ld,int splits) {
    int64_t total=int64_t(m)*n;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=int64_t(blockDim.x)*gridDim.x) {
        float value=0.f;
        for(int s=0;s<splits;++s) value+=partial[int64_t(s)*total+i];
        out[(i/m)*ld+i%m]+=value;
    }
}
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
template<int U,int O,int D,int XD,int GD,int AD=U*O>
__global__ void lunaeq_uvw_bwd_dx_fp32(const float* a,const float* g,float* dx,
        int64_t start,int count,float scale) {
    int lane=threadIdx.x%8;
    for(int64_t i=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/8;i<int64_t(count)*U;i+=int64_t(blockDim.x/8)*gridDim.x) {
        int u=i%U,n=i/U; float z[D]={};
        for(int o=lane;o<O;o+=8) {
            float av=a[int64_t(n)*AD+u*O+o];
#pragma unroll
            for(int d=0;d<D;++d) z[d]=fmaf(av,g[(start+n)*GD+o*D+d],z[d]);
        }
#pragma unroll
        for(int d=0;d<D;++d) {
            unsigned mask=0xffu << (threadIdx.x%32/8*8);
            for(int offset=4;offset>0;offset/=2) z[d]+=__shfl_down_sync(mask,z[d],offset,8);
            if(lane==0) dx[(start+n)*XD+u*D+d]+=scale*z[d];
        }
    }
}
template<int U,int O,int D,int XD,int GD,int AD=U*O>
__global__ void lunaeq_uvw_bwd_output_fp32(const float* a,const float* x,float* out,
        int64_t start,int count,float scale) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(count)*O;i+=int64_t(blockDim.x)*gridDim.x) {
        int o=i%O,n=i/O; float z[D]={};
        for(int u=0;u<U;++u) {
            float value=a[int64_t(n)*AD+u*O+o];
            #pragma unroll
            for(int d=0;d<D;++d) z[d]=fmaf(x[(start+n)*XD+u*D+d],value,z[d]);
        }
        #pragma unroll
        for(int d=0;d<D;++d) out[(start+n)*GD+o*D+d]+=scale*z[d];
    }
}
template<int U,int O,int D,int XD,int GD,int Products,int HD=U*O,int Offset=0>
__global__ void lunaeq_uvw_bwd_product_fp32(const float* x,const float* g,const float* x1,const float* g1,float* h,
        int64_t start,int count,float scale) {
    constexpr int columns=(O+3)/4;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(count)*U*columns;i+=int64_t(blockDim.x)*gridDim.x) {
        int o=(i%columns)*4,u=(i/columns)%U,n=i/(U*columns);
        float z[4]={};
        #pragma unroll
        for(int d=0;d<D;++d) {
            float value=x[(start+n)*XD+u*D+d];
            #pragma unroll
            for(int j=0;j<4;++j) if(o+j<O) z[j]=fmaf(value,g[(start+n)*GD+(o+j)*D+d],z[j]);
        }
        if constexpr(Products==2) {
            #pragma unroll
            for(int d=0;d<D;++d) {
                float value=x1[(start+n)*XD+u*D+d];
                #pragma unroll
                for(int j=0;j<4;++j) if(o+j<O) z[j]=fmaf(value,g1[(start+n)*GD+(o+j)*D+d],z[j]);
            }
        }
        #pragma unroll
        for(int j=0;j<4;++j) z[j]*=scale;
        float* dst=h+int64_t(n)*HD+u*O+o;
        if constexpr(O%4==0 && HD%4==0 && Offset%4==0)
            *reinterpret_cast<float4*>(dst)=make_float4(z[0],z[1],z[2],z[3]);
        else {
            #pragma unroll
            for(int j=0;j<4;++j) if(o+j<O) dst[j]=z[j];
        }
    }
}
"""
    code += r"""
template<int U,int V,int XD,int YD>
__global__ void lunaeq_uvw_bwd_scalar_dx_fp32(const float* t,const float* y,float* dx,int64_t start,int count,float scale) {
 int lane=threadIdx.x%32;
 for(int64_t i=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;i<int64_t(count)*U;i+=int64_t(blockDim.x/32)*gridDim.x) {
  int u=i%U,n=i/U;float z=0;
  for(int v=lane;v<V;v+=32) z=fmaf(t[(int64_t(n)*U+u)*V+v],y[(start+n)*YD+v],z);
  for(int k=16;k>0;k/=2)z+=__shfl_down_sync(0xffffffff,z,k);
  if(lane==0)dx[(start+n)*XD+u]+=scale*z;
 }
}
template<int U,int V,int XD,int YD>
__global__ void lunaeq_uvw_bwd_scalar_dy_fp32(const float* t,const float* x,float* dy,int64_t start,int count,float scale) {
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(count)*V;i+=int64_t(blockDim.x)*gridDim.x) {
  int v=i%V,n=i/V;float z=0;
  for(int u=0;u<U;++u)z=fmaf(t[(int64_t(n)*U+u)*V+v],x[(start+n)*XD+u],z);
  dy[(start+n)*YD+v]+=scale*z;
 }
}
template<int U,int V,int XD,int YD,int NX,int NY>
__global__ void lunaeq_uvw_bwd_scalar_dx_dy_fp32(
 const float* t,const float* y0,const float* y1,const float* x0,const float* x1,
 float* dx0,float* dx1,float* dy0,float* dy1,int64_t start,float scale) {
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const int64_t n=blockIdx.x;
 const float* ys[2]={y0,y1};const float* xs[2]={x0,x1};
 float* dxs[2]={dx0,dx1};float* dys[2]={dy0,dy1};
 float dy_acc[NY][(V+31)/32]={};
 __shared__ float partial[NY][8][32];
 for(int u=warp;u<U;u+=8) {
  float dx_acc[NX]={};float xv[NY];
  #pragma unroll
  for(int c=0;c<NY;++c)xv[c]=xs[c][(start+n)*XD+u];
  #pragma unroll
  for(int j=0;j<(V+31)/32;++j) {
   int v=j*32+lane;
   if(v<V) {
    float tv=t[(n*U+u)*V+v];
    #pragma unroll
    for(int c=0;c<NX;++c)dx_acc[c]=fmaf(tv,ys[c][(start+n)*YD+v],dx_acc[c]);
    #pragma unroll
    for(int c=0;c<NY;++c)dy_acc[c][j]=fmaf(tv,xv[c],dy_acc[c][j]);
   }
  }
  #pragma unroll
  for(int c=0;c<NX;++c) {
   for(int k=16;k>0;k/=2)dx_acc[c]+=__shfl_down_sync(0xffffffff,dx_acc[c],k);
   if(lane==0)dxs[c][(start+n)*XD+u]+=scale*dx_acc[c];
  }
 }
 #pragma unroll
 for(int j=0;j<(V+31)/32;++j) {
  #pragma unroll
  for(int c=0;c<NY;++c)partial[c][warp][lane]=dy_acc[c][j];
  __syncthreads();
  if(warp==0 && j*32+lane<V) {
   #pragma unroll
   for(int c=0;c<NY;++c) {
    float sum=0;
    #pragma unroll
    for(int w=0;w<8;++w)sum+=partial[c][w][lane];
    dys[c][(start+n)*YD+j*32+lane]+=scale*sum;
   }
  }
  __syncthreads();
 }
}
template<class Grouped,bool SplitK=true>
void scalar_gemm_serial(const float* a,const float* b,float* out,int m,int n,int k,int lda,int ldb,int ldc,int splits,int* flags,cudaStream_t stream) {
 using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
 using Kernel=cutlass::gemm::kernel::Gemm<typename Grouped::Mma,typename Grouped::Epilogue,Swizzle,SplitK>;
 Swizzle swizzle;auto tiled=swizzle.get_tiled_shape({m,n,k},{Grouped::Mma::Shape::kM,Grouped::Mma::Shape::kN,8},splits);
 typename Kernel::Params params({m,n,k},tiled,{const_cast<float*>(a),lda},{const_cast<float*>(b),ldb},{out,ldc},{out,ldc},{1.f,1.f},flags);
 void* args[]={&params};
 C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(cutlass::Kernel<Kernel>),swizzle.get_grid_shape(tiled),dim3(Kernel::kThreadCount),args,sizeof(typename Kernel::SharedStorage),stream));
}

template<int U,int V,int XD,int YD,int Products>
__global__ void lunaeq_uvw_bwd_scalar_outer_fp32(const float* x,const float* y,const float* x1,const float* y1,float* q,int64_t start,int count,float scale,int* flags,int flag_count) {
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<flag_count;i+=int64_t(blockDim.x)*gridDim.x) flags[i]=0;
 int64_t total=int64_t(count)*U*V;
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total/4;i+=int64_t(blockDim.x)*gridDim.x) {
  int64_t base=4*i;
  int v=base%V,u=(base/V)%U,n=base/(U*V);
  float value[4];
  if(V%4==0 || v+3<V) {
   float xv=x[(start+n)*XD+u];
   float xv1=0.f;
   if constexpr(Products==2) xv1=x1[(start+n)*XD+u];
   #pragma unroll
   for(int j=0;j<4;++j) {
    value[j]=xv*y[(start+n)*YD+v+j];
    if constexpr(Products==2) value[j]=fmaf(xv1,y1[(start+n)*YD+v+j],value[j]);
    value[j]*=scale;
   }
  } else {
   #pragma unroll
   for(int j=0;j<4;++j) {
    int vv=(base+j)%V,uu=((base+j)/V)%U,nn=(base+j)/(U*V);
    value[j]=x[(start+nn)*XD+uu]*y[(start+nn)*YD+vv];
    if constexpr(Products==2) value[j]=fmaf(x1[(start+nn)*XD+uu],y1[(start+nn)*YD+vv],value[j]);
    value[j]*=scale;
   }
  }
  reinterpret_cast<float4*>(q)[i]=make_float4(value[0],value[1],value[2],value[3]);
 }
 if(blockIdx.x==0 && threadIdx.x<total%4) {
  int64_t i=(total/4)*4+threadIdx.x;
  int v=i%V,u=(i/V)%U,n=i/(U*V);
  float value=x[(start+n)*XD+u]*y[(start+n)*YD+v];
  if constexpr(Products==2) value=fmaf(x1[(start+n)*XD+u],y1[(start+n)*YD+v],value);
  q[i]=scale*value;
 }

}
"""
    for (yo, V), group_specs in path_groups.items():
        width = sum(s["path_channels"] // V * s["out_channels"] for s in group_specs)
        code += f"""
__global__ void lunaeq_uvw_bwd_pack_group_{yo}_{V}_fp32(const float* weight,float* packed) {{
    constexpr int width={width}, channels={V};
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            i<int64_t(width)*channels;i+=int64_t(blockDim.x)*gridDim.x) {{
        const int column=int(i%width), v=int(i/width);
"""
        offset = 0
        for index, spec in enumerate(group_specs):
            U, O = spec["path_channels"] // V, spec["out_channels"]
            prefix = "if" if index == 0 else "else if"
            code += f"""
        {prefix}(column<{offset+U*O}) {{
            const int local=column-{offset};
            packed[i]=weight[int64_t({spec['weight_offset']})+int64_t(local/{O})*channels*{O}
                +int64_t(v)*{O}+local%{O}];
        }}
"""
            offset += U*O
        code += "    }\n}\n"
    for (yo, V), group_specs in path_groups.items():
        if any(spec["out_channels"] % 4 for spec in group_specs):
            continue
        width = sum(spec["in_channels"] * spec["out_channels"] for spec in group_specs)
        code += f"""
template<int Products>
__global__ void lunaeq_uvw_bwd_product_group_{yo}_{V}_fp32(
        const float* x,const float* g,const float* x1,const float* g1,float* h,
        int64_t start,int count) {{
    constexpr int width={width}, columns=width/4;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            i<int64_t(count)*columns;i+=int64_t(blockDim.x)*gridDim.x) {{
        const int column=int(i%columns)*4;
        const int64_t row=start+i/columns;
        float value[4]={{}};
"""
        offset = 0
        for index, spec in enumerate(group_specs):
            U, O, D = spec["in_channels"], spec["out_channels"], spec["d1"]
            xo, oo = spec["x_offset"], spec["out_offset"]
            scale = float_literal(spec["terms"][0][3], torch.float32)
            prefix = "if" if index == 0 else "else if"
            code += f"""
        {prefix}(column<{offset+U*O}) {{
            const int local=column-{offset}, u=local/{O}, o=local%{O};
            #pragma unroll
            for(int d=0;d<{D};++d) {{
                const float a=x[row*{meta['in1_dim']}+{xo}+u*{D}+d];
                #pragma unroll
                for(int j=0;j<4;++j)
                    value[j]=fmaf(a,g[row*{meta['out_dim']}+{oo}+(o+j)*{D}+d],value[j]);
            }}
            if constexpr(Products==2) {{
                #pragma unroll
                for(int d=0;d<{D};++d) {{
                    const float a=x1[row*{meta['in1_dim']}+{xo}+u*{D}+d];
                    #pragma unroll
                    for(int j=0;j<4;++j)
                        value[j]=fmaf(a,g1[row*{meta['out_dim']}+{oo}+(o+j)*{D}+d],value[j]);
                }}
            }}
            #pragma unroll
            for(int j=0;j<4;++j) value[j]*={scale};
        }}
"""
            offset += U*O
        code += "        reinterpret_cast<float4*>(h)[i]=make_float4(value[0],value[1],value[2],value[3]);\n    }\n}\n"
    if w_terms:
        for (yo, V), group_specs in path_groups.items():
            width = sum(s["path_channels"] // V * s["out_channels"] for s in group_specs)
            code += f"""
__global__ void lunaeq_uvw_bwd_weight_sum_{yo}_{V}_fp32(
        const float* partial,const float* packed,float* weight,int splits) {{
    constexpr int width={width}, channels={V};
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            i<int64_t(width)*channels;i+=int64_t(blockDim.x)*gridDim.x) {{
        float value=0.f;
        for(int s=0;s<splits;++s) value+=partial[int64_t(s)*width*channels+i];
        value+=packed[i];
        const int column=int(i%width), v=int(i/width);
"""
            offset = 0
            for index, s in enumerate(group_specs):
                U, O = s["path_channels"] // V, s["out_channels"]
                prefix = "if" if index == 0 else "else if"
                code += f"""
        {prefix}(column<{offset+U*O}) {{
            const int local=column-{offset};
            weight[int64_t({s['weight_offset']})+int64_t(local/{O})*channels*{O}
                +int64_t(v)*{O}+local%{O}]+=value;
        }}
"""
                offset += U*O
            code += "    }\n}\n"
    parallel = (bool(x_terms or forward_terms) and bool(y_terms or w_terms)) or (
        bool(y_terms) and bool(w_terms or forward_terms)
    ) or (
        bool(x_terms) and bool(forward_terms) and any(s["d1"] == 1 for s in specs)
    )
    if parallel:
        code += DERIVATIVE_STREAM_SOURCE
    code += (
        "\nstd::vector<at::Tensor> launch("
        + ",".join(f"at::Tensor {v}" for v in inputs)
        + ") {\n"
    )
    code += "c10::cuda::CUDAGuard guard(x.device()); auto stream=at::cuda::getCurrentCUDAStream(); int64_t items=x.size(0);\n"
    for out in outputs:
        code += f"auto {out}=at::zeros_like({OUTPUT_SHAPES[out]});\n"
    code += "if(!items) return {" + ",".join(outputs) + "};\n"
    code += "const int sm_count=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;\n"
    z_groups = sorted({z for out in x_terms for _,z in program["x"][out]} | {z for out in y_terms for _,z in program["y"][out]})
    has_scalar = any(spec["d1"] == 1 for spec in specs)
    has_angular = any(spec["d1"] != 1 for spec in specs)
    if x_terms or (has_scalar and y_terms) or (has_angular and forward_terms):
        a_sizes = []
        if z_groups:
            for spec in specs:
                if spec["d1"] == 1:
                    width = spec["path_channels"]
                    chunk = max(1, (workspace_elements // (width*len(z_groups)) // 128) * 128)
                    a_sizes.append(f"std::min(items,int64_t({chunk}))*{width}")
        if x_terms or forward_terms:
            for (_, v), group in path_groups.items():
                width = sum(s["path_channels"] // v * s["out_channels"] for s in group)
                chunk = max(1, (workspace_elements // width // 128) * 128)
                a_sizes.append(f"std::min(items,int64_t({chunk}))*{width}")
        code += f"auto a_buffer=at::empty({{std::max({{{','.join(dict.fromkeys(a_sizes))}}})}},x.options());\n"
    if w_terms or (has_scalar and forward_terms) or (has_angular and y_terms):
        dw_flags = max((((s["path_channels"]+127)//128)*((s["out_channels"]+127)//128) for s in specs if s["d1"]==1),default=0) if w_terms else 0
        flag_counts = [f"int64_t({dw_flags})"]
        if forward_terms:
            for s in specs:
                if s["d1"] == 1:
                    flag_counts.append(f"((std::min(items,h_elements/{s['path_channels']})+127)/128)*{(s['out_channels']+127)//128}")
        h_widths = [f"int64_t({max_width})"]
        if forward_terms:
            for s in specs:
                if s["d1"] == 1:
                    uv, o = s["path_channels"], s["out_channels"]
                    tiles = f"((items+127)/128)*{(o+127)//128}"
                    capacity = f"std::min({max(1,uv//64)},std::max(1,int((int64_t(sm_count)*4+({tiles})-1)/({tiles}))))"
                    enabled = f"({capacity}>16 && ({tiles})*16<sm_count*4)"
                    h_widths.append(f"({enabled} ? int64_t({uv})+int64_t({o})*({capacity}) : int64_t(0))")
        code += f"const int64_t h_width=std::max({{{','.join(h_widths)}}}); int64_t h_elements=std::min(items*h_width,int64_t({workspace_elements}));\n"
        if y_terms:
            # A single DW partition needs no partial buffer. Keep the short row
            # batch together and reserve only its actual DY partials.
            for (_, v), group in path_groups.items():
                width = sum(s["in_channels"] * s["out_channels"] for s in group)
                code += f"if(items/256<2) h_elements=std::max(h_elements,std::min(items*int64_t({width+v*(width//256)}),int64_t({workspace_elements})));\n"
        code += f"const int64_t flag_count=std::max({{{','.join(flag_counts)}}}); auto h_buffer=at::empty({{h_elements+flag_count}},x.options()); int* gemm_flags=reinterpret_cast<int*>(h_buffer.data_ptr<float>()+h_elements);\n"
    for spec in specs:
        if path_groups and spec["d1"] > 1:
            continue
        U, V, O, D = (
            spec["path_channels"] // spec["edge_channels"],
            spec["edge_channels"],
            spec["out_channels"],
            spec["d1"],
        )
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        XD, YD, GD = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
        scale = float_literal(spec["terms"][0][3], torch.float32)
        width = U * V if D == 1 else U * O
        chunk = max(1, (workspace_elements // width // 128) * 128)
        if D == 1:
            code += "{\n"
            tiles=((O+127)//128)*((U*V+127)//128)
            if forward_terms:
                budget_guard = (f" && h_elements>=int64_t({U*V})+int64_t({O})*gg_parallel_capacity"
                                if U*V+O*max(1,U*V//64)>workspace_elements else "")
                code += f"const int64_t gg_tiles_total=((items+127)/128)*{(O+127)//128}; const int gg_parallel_capacity=std::min({max(1,U*V//64)},std::max(1,int((int64_t(sm_count)*4+gg_tiles_total-1)/gg_tiles_total))); const bool gg_parallel=gg_parallel_capacity>16 && gg_tiles_total*16<sm_count*4{budget_guard}; const int gg_capacity=gg_parallel ? gg_parallel_capacity : std::max(1,std::min(sm_count*4,{U*V//256}));\n"
                code += f"const int64_t path_chunk=std::max(int64_t(1),std::min(int64_t({chunk}),h_elements/({U*V}+int64_t({O})*(gg_parallel ? gg_capacity : 0))));\n"
            elif w_terms:
                code += f"const int64_t path_chunk=std::max(int64_t(1),std::min(int64_t({chunk}),h_elements/{U*V}));\n"
            else:
                code += f"const int64_t path_chunk={chunk};\n"
            d1_parallel = bool(x_terms or y_terms) and bool(w_terms or forward_terms)
            if d1_parallel:
                code += "auto& lane=derivative_stream(x.get_device(),stream); DerivativeFork fork(lane,stream); auto xy_stream=lane.stream;\n"
            else:
                code += "auto xy_stream=stream;\n"
            separate_z = len(z_groups) > 1
            if separate_z:
                code += f"const int64_t a_rows=std::min(items,a_buffer.numel()/{U*V}); int64_t z_chunk=std::max(int64_t(1),a_rows);\n"
                code += "if(z_chunk<items && z_chunk>=128) { const int64_t tiles=(items+127)/128,capacity=z_chunk/128; const int64_t chunks=(tiles+capacity-1)/capacity; z_chunk=((tiles+chunks-1)/chunks)*128; }\n"
                code += "for(int64_t start=0;start<items;start+=z_chunk) { int count=int(std::min(z_chunk,items-start));\n"
            else:
                code += "for(int64_t start=0;start<items;start+=path_chunk) { int count=int(std::min(path_chunk,items-start));\n"
            # Multiple Z results reuse A independently of the Q/H row chunks.
            for z in z_groups:
                for wave, (w, g) in enumerate(program["z"][z]):
                    code += f"scalar_tn(xy_stream,{U*V},count,{O},{w}.data_ptr<float>()+{wo},{O},{g}.data_ptr<float>()+start*{GD}+{oo},{GD},a_buffer.data_ptr<float>(),{U*V},{0 if wave==0 else 1}.f);\n"
                x_consumers = [(out,y) for out in x_terms for y,zz in program["x"][out] if zz==z]
                y_consumers = [(out,x) for out in y_terms for x,zz in program["y"][out] if zz==z]
                if x_consumers and y_consumers:
                    pointers = []
                    for consumers, entry, offset in (
                        (x_consumers, 1, yo), (y_consumers, 1, xo),
                        (x_consumers, 0, xo), (y_consumers, 0, yo),
                    ):
                        for i in range(2):
                            pointers.append(
                                f"{consumers[i][entry]}.data_ptr<float>()+{offset}"
                                if i < len(consumers) else "nullptr"
                            )
                    code += (
                        f"lunaeq_uvw_bwd_scalar_dx_dy_fp32<{U},{V},{XD},{YD},{len(x_consumers)},{len(y_consumers)}>"
                        f"<<<count,256,0,xy_stream>>>(a_buffer.data_ptr<float>(),{','.join(pointers)},start,{scale});\n"
                    )
                    continue
                for out in x_terms:
                    for y, zz in program["x"][out]:
                        if zz == z:
                            code += (
                                f"lunaeq_uvw_bwd_scalar_dx_fp32<{U},{V},{XD},{YD}><<<(int64_t(count)*{U*32}+255)/256,"
                                f"256,0,xy_stream>>>(a_buffer.data_ptr<float>(),{y}.data_ptr<float>()+{yo},"
                                f"{out}.data_ptr<float>()+{xo},start,count,{scale});\n"
                            )
                for out in y_terms:
                    for x, zz in program["y"][out]:
                        if zz == z:
                            code += (
                                f"lunaeq_uvw_bwd_scalar_dy_fp32<{U},{V},{XD},{YD}><<<(int64_t(count)*{V}+255)/256,"
                                f"256,0,xy_stream>>>(a_buffer.data_ptr<float>(),{x}.data_ptr<float>()+{xo},"
                                f"{out}.data_ptr<float>()+{yo},start,count,{scale});\n"
                            )
            if separate_z:
                code += "}\n"
            products = sorted(
                {q for out in w_terms for q, _ in program["w"][out]}
                | ({q for q, _ in program["forward"]} if forward_terms else set())
            )
            if separate_z and products:
                code += "for(int64_t start=0;start<items;start+=path_chunk) { int count=int(std::min(path_chunk,items-start));\n"
            if w_terms:
                code += f"const int dw_parts=uvw_grouped::gemm_splits({tiles},count,std::max(1,std::min(16,count/1024)),sm_count*2,sm_count*4);\n"
            if forward_terms:
                code += f"const int gg_tile=gg_parallel && count<=64 ? 64 : 128; const int gg_tiles=((count+gg_tile-1)/gg_tile)*(({O}+gg_tile-1)/gg_tile); const int gg_parts=uvw_grouped::gemm_splits(gg_tiles,{U*V},gg_capacity,sm_count*(gg_tile==64 ? 4 : 2),sm_count*4); const int gg_slice=(({U*V}+gg_parts*8-1)/(gg_parts*8))*8; const int gg_splits=({U*V}+gg_slice-1)/gg_slice;\n"
            for q in products:
                overlap_product = (any(qq == q for out in w_terms for qq, _ in program["w"][out])
                                   and bool(forward_terms) and any(qq == q for qq, _ in program["forward"]))
                gg_stream = "gg_stream" if overlap_product else "stream"
                if overlap_product:
                    code += "{\n"
                pairs = program["cg"][q]
                x, y = pairs[0]
                x1, y1 = pairs[-1]
                code += (
                    f"lunaeq_uvw_bwd_scalar_outer_fp32<{U},{V},{XD},{YD},{len(pairs)}><<<((int64_t(count)*{U*V}+3)/4+255)/256,"
                    f"256,0,stream>>>({x}.data_ptr<float>()+{xo},{y}.data_ptr<float>()+{yo},{x1}.data_ptr<float>()+{xo},"
                    f"{y1}.data_ptr<float>()+{yo},h_buffer.data_ptr<float>(),start,count,{scale},gemm_flags,std::max({tiles if w_terms else 0},{((O+127)//128) if forward_terms else 0}*((count+127)/128)));\n"
                )
                if overlap_product:
                    code += "std::optional<DerivativeFork> gg_fork; cudaStream_t gg_stream=stream;\n"
                    code += "if(gg_parallel && gg_tile==128 && dw_parts==1) { auto& lane=derivative_stream(x.get_device(),stream,1); gg_fork.emplace(lane,stream); gg_stream=lane.stream; }\n"
                for out in w_terms:
                    for qq, g in program["w"][out]:
                        if qq == q:
                            arguments=f"h_buffer.data_ptr<float>(),{g}.data_ptr<float>()+start*{GD}+{oo},{out}.data_ptr<float>()+{wo},{U*V},{O},count,{U*V},{GD},{O},dw_parts,gemm_flags,stream"
                            if all(value%4==0 for value in (U*V,O,GD,oo)):
                                kernel="uvw_grouped::AsyncNT<3,4>" if wo%4==0 else "uvw_grouped::AsyncNT<3>"
                                if (O+63)//64<=65535:
                                    short=f"uvw_grouped::AsyncNT<3,{2 if wo%2==0 else 1},64,64,32,32>"
                                    code += f"if((reinterpret_cast<uintptr_t>({g}.data_ptr<float>())&15)==0) {{ if(count<2048 && dw_parts==1) scalar_gemm_serial<{short},false>({arguments}); else scalar_gemm_serial<{kernel}>({arguments}); }} else "
                                else:
                                    code += f"if((reinterpret_cast<uintptr_t>({g}.data_ptr<float>())&15)==0) scalar_gemm_serial<{kernel}>({arguments}); else "
                            code += f"scalar_gemm_serial<uvw_grouped::Kernel<cutlass::layout::ColumnMajor,cutlass::layout::RowMajor,cutlass::layout::RowMajor,128,128,8,cutlass::layout::NoPermute,64,64>>({arguments});\n"
                if forward_terms:
                    for qq, w in program["forward"]:
                        if qq == q:
                            code += f"if(gg_parallel) {{ float* partial=h_buffer.data_ptr<float>()+int64_t(count)*{U*V};\n"
                            arguments=f"h_buffer.data_ptr<float>(),{w}.data_ptr<float>()+{wo},partial,count,{O},{U*V},gg_splits,gg_slice,{gg_stream}"
                            code += f"if(gg_tile==64) uvw_grouped::scalar_gemm_parallel<uvw_grouped::SmallGemmNN>({arguments}); else "
                            if O%4==0 and (U*V)%4==0 and wo%4==0:
                                code += f"if((reinterpret_cast<uintptr_t>({w}.data_ptr<float>())&15)==0) uvw_grouped::scalar_gemm_parallel<uvw_grouped::GemmNN>({arguments}); else "
                            code += f"uvw_grouped::scalar_gemm_parallel<uvw_grouped::GemmNNBase>({arguments});\n"
                            code += f"scalar_split_sum<{O},{GD},true><<<(int64_t(count)*{O}+255)/256,256,0,{gg_stream}>>>(partial,gg.data_ptr<float>()+start*{GD}+{oo},int64_t(count)*{O},gg_splits,1.f);\n}} else {{\n"
                            arguments=f"h_buffer.data_ptr<float>(),{w}.data_ptr<float>()+{wo},gg.data_ptr<float>()+start*{GD}+{oo},count,{O},{U*V},{U*V},{O},{GD},gg_parts,gemm_flags,{gg_stream}"
                            if O%4==0 and (U*V)%4==0 and wo%4==0:
                                code += f"if((reinterpret_cast<uintptr_t>({w}.data_ptr<float>())&15)==0) scalar_gemm_serial<uvw_grouped::GemmNN>({arguments}); else "
                            code += f"scalar_gemm_serial<uvw_grouped::GemmNNBase>({arguments});\n}}\n"
                if overlap_product:
                    code += "if(gg_fork) gg_fork->join();\n}\n"
            if not separate_z or products:
                code += "}\n"
            if d1_parallel:
                code += "fork.join();\n"
            code += "}\n"
            continue
    if path_groups:
        code += make_uvw_grouped_backward_launch(meta, path_groups, workspace_elements, x_terms, y_terms, w_terms, forward_terms)
    code += (
        "C10_CUDA_KERNEL_LAUNCH_CHECK(); return {"
        + ",".join(outputs)
        + '};}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&launch);}\n'
    )
    return code


def make_uvw_backward_dxdy_shared_kernel(meta, program, inputs, outputs, name, *, arch):
    fp64 = meta.get("dtype") is torch.float64
    specs = meta["generated_specs"]
    threads = 128 if fp64 or all(s["out_channels"] >= 128 for s in specs) else 256
    # An irrep wider than a tile is reduced in component chunks. The CG
    # epilogue adds each chunk directly to its final gradient.
    segmented = any(spec["d3"] > 32 for spec in specs)
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    needed = {
        z for pairs in (*x_outputs.values(), *y_outputs.values()) for _, z in pairs
    }
    contractions = {k: v for k, v in program["z"].items() if k in needed}
    # Warp-wide columns reuse each weight fragment across more component rows.
    mma_columns = 2 if threads == 128 and len(contractions) > 2 else 4
    lanes = 64 // mma_columns
    mma_rows = 32 * lanes // threads
    weights = sorted({w for pairs in contractions.values() for w, _ in pairs})
    gradients = sorted({g for pairs in contractions.values() for _, g in pairs})
    slots = {z: i for i, z in enumerate(contractions)}
    if fp64:
        buffers = 1
        resident_blocks = 2
    else:
        buffers = 2 if len(weights) + len(gradients) <= 3 else 1
        resident_blocks = (3 if len(contractions) <= 2 else 2) * 256 // threads
    xdim, ydim, odim = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    tasks, totals, epilogues = [], [], []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        edge, channels, paths = (
            spec[k] for k in ("edge_channels", "out_channels", "path_channels")
        )
        tile_items, column_tiles = max(1, 32 // d3), (paths + 63) // 64
        chunks = (d3 + 31) // 32
        tiles_per_item = column_tiles * chunks
        count = f"((items+{tile_items - 1})/{tile_items})*{column_tiles}"
        if segmented:
            count = f"((items+{tile_items - 1})/{tile_items})*{tiles_per_item}"
        totals.append(count)
        tasks.append(
            r"""{{
            Index count={count};
            if(task>=cursor && task<cursor+count) {{
                Index local=task-cursor;
                spec_id={index}; components={d3}; channels={channels}; paths={paths};
                out_offset={oo}; weight_offset={wo}; tile_items={tile_items};
                first_item=(local/{tiles_per_item})*{tile_items};
                {component_setup}
                first_path=(local%{column_tiles})*64;
            }}
            cursor+=count;
        }}""".format(
                count=count,
                index=index,
                d3=d3,
                channels=channels,
                paths=paths,
                oo=oo,
                wo=wo,
                tile_items=tile_items,
                tiles_per_item=tiles_per_item,
                component_setup=f"row_components={min(d3, 32)}; first_component=((local/{column_tiles})%{chunks})*32;"
                if segmented
                else "",
                column_tiles=column_tiles,
            )
        )
        lines = [
            r"""case {index}: {{
#pragma unroll 1
for(int linear=threadIdx.x;linear<{feature_rows};linear+={threads}) {{
int local_item=linear/64, local_path=linear%64;
Index item=first_item+local_item,p=first_path+local_path;
int u=p/{edge},v=p%{edge};
bool valid=item<items && p<{paths};""".format(
                index=index,
                feature_rows=tile_items * 64,
                threads=threads,
                edge=edge,
                paths=paths,
            )
        ]
        for target in y_outputs:
            lines += [f"float {target}_{j}=0.f;" for j in range(d2)]
        lines.append("if(valid) {")
        for target, pairs in x_outputs.items():
            lines.append("{")
            for src in sorted({a for a, _ in pairs}):
                lines += [
                    f"float {src}_{j}=__ldg({src}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j});"
                    for j in sorted({t[1] for t in spec["terms"]})
                ]
            lines += [f"float acc{i}=0.f;" for i in range(d1)]
            for i, j, k, c in spec["terms"]:
                for src, z in pairs:
                    row = f"local_item*{d3}+{k}"
                    guard = ""
                    if segmented:
                        row = f"local_item*row_components+{k}-first_component"
                        guard = f"if({k}>=first_component && {k}<first_component+row_components) "
                    lines.append(
                        f"""{guard}acc{i}=fmaf({float_literal(c, meta.get("dtype"))}*{src}_{j},shared.z[{slots[z]}][({row})*64+local_path],acc{i});"""
                    )
            lines += [
                f"atomicAdd({target}+int64_t(item)*{xdim}+{xo}+u*{d1}+{i},acc{i});"
                for i in range(d1)
            ]
            lines.append("}")
        for target, pairs in y_outputs.items():
            lines.append("{")
            for src in sorted({a for a, _ in pairs}):
                lines += [
                    f"float {src}_{i}=__ldg({src}+int64_t(item)*{xdim}+{xo}+u*{d1}+{i});"
                    for i in sorted({t[0] for t in spec["terms"]})
                ]
            for i, j, k, c in spec["terms"]:
                for src, z in pairs:
                    row = f"local_item*{d3}+{k}"
                    guard = ""
                    if segmented:
                        row = f"local_item*row_components+{k}-first_component"
                        guard = f"if({k}>=first_component && {k}<first_component+row_components) "
                    lines.append(
                        f"""{guard}{target}_{j}=fmaf({float_literal(c, meta.get("dtype"))}*{src}_{i},shared.z[{slots[z]}][({row})*64+local_path],{target}_{j});"""
                    )
            lines.append("}")
        lines.append("}")
        for target in y_outputs:
            for j in range(d2):
                if edge == 1:
                    lines += [
                        r"""#pragma unroll
for(int delta=16;delta>0;delta>>=1)
    {target}_{j}+=__shfl_down_sync(0xffffffff,{target}_{j},delta);
if((threadIdx.x&31)==0 && item<items) atomicAdd({target}+int64_t(item)*{ydim}+{component_offset},{target}_{j});""".format(
                            target=target,
                            j=j,
                            ydim=ydim,
                            component_offset=yo + j,
                        )
                    ]
                else:
                    lines += [
                        f"if(valid) atomicAdd({target}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j},{target}_{j});"
                    ]
        lines += ["}", "break;", "}"]
        epilogues.append("\n".join(lines))

    params = [f"const float* __restrict__ {v}" for v in inputs]
    params += [f"float* __restrict__ {v}" for v in outputs] + ["Index items"]
    vector_grad = (
        not fp64
        and not segmented
        and odim % 4 == 0
        and all(s["out_offset"] % 4 == 0 and s["out_channels"] % 4 == 0 for s in specs)
    )
    vector_weight = not fp64 and all(
        s["weight_offset"] % 4 == 0 and s["out_channels"] % 4 == 0 for s in specs
    )

    loads, fragments, contractions_cpp, stores = [], [], [], []
    for gi, g in enumerate(gradients):
        if vector_grad:
            scatter = []
            for offset, member in enumerate(("x", "y", "z", "w")):
                scatter.append(
                    f"shared.tiles.a[buffer][{gi}][(local_item*components+(inside+{offset})%components)*33+(inside+{offset})/components]=value.{member};"
                )
            loads.append(
                r"""for(int index=threadIdx.x;index<tile_items*8*components;index+={threads}) {{
                int scalar=index*4,local_item=scalar/(32*components);
                Index inside=scalar-local_item*32*components,item=first_item+local_item;
                float4 value={{0.f,0.f,0.f,0.f}};
                if(item<items && tile_k*components+inside<channels*components)
                    value=*reinterpret_cast<const float4*>({g}+int64_t(item)*{odim}+out_offset+tile_k*components+inside);
                {scatter}
            }}
            for(int index=threadIdx.x;index<(32-tile_items*components)*32;index+={threads}) {{
                Index row=tile_items*components+index/32;
                shared.tiles.a[buffer][{gi}][row*33+index%32]=0.f;
            }}""".format(
                    threads=threads,
                    g=g,
                    odim=odim,
                    scatter=chr(10).join(scatter),
                    gi=gi,
                )
            )
        else:
            loads.append(
                r"""for(int index=threadIdx.x;index<32*32;index+={threads}) {{
                Index row=index/32,k=index%32,item=first_item+row/{component_width};
                int component=row%{component_end};
                shared.tiles.a[buffer][{gi}][row*33+k]=
                    row<tile_items*{component_width} {component_guard} && item<items && tile_k+k<channels ?
                    __ldg({g}+int64_t(item)*{odim}+out_offset+(tile_k+k)*components+component) : 0.f;
            }}""".format(
                    threads=threads,
                    component_width="row_components" if segmented else "components",
                    component_end="row_components+first_component"
                    if segmented
                    else "components",
                    gi=gi,
                    component_guard="&& component<components" if segmented else "",
                    g=g,
                    odim=odim,
                )
            )
        fragments.append(r"""Mma::FragmentA a{gi};
a{gi}[0]=shared.tiles.a[buffer][{gi}][local_row*33+k];
a{gi}[1]=shared.tiles.a[buffer][{gi}][(local_row+1)*33+k];""".format(gi=gi))
        fragments += [
            f"a{gi}[{m}]=shared.tiles.a[buffer][{gi}][(local_row+{m})*33+k];"
            for m in range(2, mma_rows)
        ]
    for wi, w in enumerate(weights):
        if vector_weight:
            scatter = [
                f"shared.tiles.b[buffer][{wi}][path*33+k+{offset}]=value.{member};"
                for offset, member in enumerate(("x", "y", "z", "w"))
            ]
            loads.append(
                r"""for(int index=threadIdx.x;index<64*8;index+={threads}) {{
                int path=index/8,k=(index%8)*4;
                float4 value={{0.f,0.f,0.f,0.f}};
                if(first_path+path<paths && tile_k+k<channels)
                    value=*reinterpret_cast<const float4*>({w}+weight_offset+(first_path+path)*channels+tile_k+k);
                {scatter}
            }}""".format(
                    threads=threads,
                    w=w,
                    scatter=chr(10).join(scatter),
                )
            )
        else:
            loads.append(r"""for(int index=threadIdx.x;index<64*32;index+={threads}) {{
                int path=index/32,k=index%32;
                shared.tiles.b[buffer][{wi}][path*33+k]=
                    first_path+path<paths && tile_k+k<channels ?
                    __ldg({w}+weight_offset+(first_path+path)*channels+tile_k+k) : 0.f;
            }}""".format(threads=threads, wi=wi, w=w))
        fragments += [f"Mma::FragmentB b{wi};"]
        fragments += [
            f"b{wi}[{n}]=shared.tiles.b[buffer][{wi}][(local_column+{n * lanes})*33+k];"
            for n in range(mma_columns)
        ]
    for z, pairs in contractions.items():
        for w, g in pairs:
            contractions_cpp.append(
                f"Mma()(acc[{slots[z]}],a{gradients.index(g)},b{weights.index(w)},acc[{slots[z]}]);"
            )
        stores += [
            f"shared.z[{slots[z]}][(local_row+m)*64+local_column+n*{lanes}]=acc[{slots[z]}][m*{mma_columns}+n];"
        ]
    if fp64:
        shared_bytes = 8 * max(
            buffers * (len(gradients) * 32 * 33 + len(weights) * 64 * 33),
            len(contractions) * 32 * 64,
        )
        shared_declaration = r"""    extern __shared__ __align__(16) unsigned char storage[];
    Storage& shared=*reinterpret_cast<Storage*>(storage);
"""
        mma_definition = f"    double acc[{len(contractions)}][4][4]={{}};\n"
        tensor_gradient_loads = "\n".join(
            r"""double a{gradient}[8];
#pragma unroll
for(int i=0;i<8;++i) a{gradient}[i]=shared.tiles.a[buffer][{gradient}][(local_row+(i&1)*8)*33+k+(lane&3)+(i>>1)*4];""".format(gradient=gi)
            for gi in range(len(gradients))
        )
        tensor_weight_loads = "\n".join(
            r"""double b{weight}[4];
#pragma unroll
for(int i=0;i<4;++i) b{weight}[i]=shared.tiles.b[buffer][{weight}][(local_column+n*8+(lane>>2))*33+k+(lane&3)+i*4];""".format(weight=wi)
            for wi in range(len(weights))
        )
        tensor_contractions = []
        for z, pairs in contractions.items():
            zi = slots[z]
            for w, g in pairs:
                a = f"a{gradients.index(g)}"
                b = f"b{weights.index(w)}"
                c = f"acc[{zi}][n]"
                tensor_contractions.append(
                    f"{get_fp64_mma(arch)[1]}::fma({c}[0],{c}[1],{c}[2],{c}[3],{a}[0],{a}[1],{a}[2],{a}[3],{a}[4],{a}[5],{a}[6],{a}[7],{b}[0],{b}[1],{b}[2],{b}[3],{c}[0],{c}[1],{c}[2],{c}[3]);"
                )
        mma_compute = r"""    for(Index tile_k=0;tile_k<channels;tile_k+=32) {{
        int buffer=(tile_k/32)%{buffers};
        {loads}
        __syncthreads();
            const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
        const int local_row=(warp>>1)*16+(lane>>2),local_column=(warp&1)*32;
        #pragma unroll
        for(int k=0;k<32;k+=16) {{
{gradient_loads}
#pragma unroll
for(int n=0;n<4;++n) {{
{weight_loads}
{contractions}
}}
}}
        __syncthreads();
    }}
""".format(
            buffers=buffers,
            loads="\n".join(loads),
            gradient_loads=tensor_gradient_loads,
            weight_loads=tensor_weight_loads,
            contractions="\n".join(tensor_contractions),
        )
        mma_stores = r"""const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
for(int n=0;n<4;++n) {{
#pragma unroll
for(int i=0;i<4;++i) {{
int row=(warp>>1)*16+(lane>>2)+(i>>1)*8,column=(warp&1)*32+n*8+(lane&3)*2+(i&1);
{stores}
}}
}}
__syncthreads();
""".format(
            stores="\n".join(
                f"shared.z[{zi}][row*64+column]=acc[{zi}][n][i];"
                for zi in range(len(contractions))
            )
        )
    else:
        shared_declaration = f"    __shared__ Storage shared;\n"
        mma_definition = r"""    using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{mma_rows},{mma_columns},1>,
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC acc[{contractions_count}];
    #pragma unroll
    for(int z=0;z<{contractions_count};++z) acc[z].clear();
    int local_row=(threadIdx.x/{lanes})*{mma_rows},local_column=threadIdx.x&{column_mask};
""".format(
            mma_rows=mma_rows,
            mma_columns=mma_columns,
            contractions_count=len(contractions),
            lanes=lanes,
            column_mask=lanes - 1,
        )
        mma_compute = r"""    for(Index tile_k=0;tile_k<channels;tile_k+=32) {{
        int buffer=(tile_k/32)%{buffers};
        {loads}
        __syncthreads();
        {unroll}
        for(int k=0;k<32;++k) {{
            {fragments}
            {contractions_cpp}
        }}
        {barrier}
    }}
""".format(
            buffers=buffers,
            loads="\n".join(loads),
            unroll="#pragma unroll 8"
            if mma_rows >= 4 and len(contractions) > 1
            else "#pragma unroll",
            fragments=chr(10).join(fragments),
            contractions_cpp=chr(10).join(contractions_cpp),
            barrier="__syncthreads();" if buffers == 1 else "",
        )
        mma_stores = r"""    #pragma unroll
    for(int m=0;m<{mma_rows};++m) {{
        #pragma unroll
        for(int n=0;n<{mma_columns};++n) {{
            {stores}
        }}
    }}
    __syncthreads();
""".format(
            mma_rows=mma_rows,
            mma_columns=mma_columns,
            stores=chr(10).join(stores),
        )
    code = r"""
__global__ __launch_bounds__({threads},{resident_blocks}) void {name}({params}) {{
    union Storage {{
        struct {{float a[{buffers}][{gradients_count}][32*33];
                 float b[{buffers}][{weights_count}][64*33];}} tiles;
        float z[{contractions_count}][32*64];
    }};
{shared_declaration}    Index task=blockIdx.x,cursor=0; int spec_id=0,components=0,channels=0,paths=0;
    int out_offset=0,weight_offset=0,tile_items=0; Index first_item=0; int first_path=0;
    {component_declaration}
    {tasks}
{mma_definition}{mma_compute}    // Every warp must finish reading tiles before the union becomes Z storage.
    __syncthreads();
{mma_stores}    switch(spec_id) {{
        {epilogues}
    }}
}}
""".format(
        threads=threads,
        resident_blocks=resident_blocks,
        name=name,
        params=",".join(params),
        buffers=buffers,
        gradients_count=len(gradients),
        weights_count=len(weights),
        contractions_count=len(contractions),
        shared_declaration=shared_declaration,
        component_declaration="int row_components=0,first_component=0;"
        if segmented
        else "",
        tasks=chr(10).join(tasks),
        mma_definition=mma_definition,
        mma_compute=mma_compute,
        mma_stores=mma_stores,
        epilogues=chr(10).join(epilogues),
    )
    if fp64:
        code += f"\nconstexpr int {name}_shared_bytes={shared_bytes};\n"
    return (
        cuda_source_for_dtype(code, meta.get("dtype", torch.float32)),
        "+".join(totals),
        threads,
    )


def make_uvw_backward_dw_shared_kernel(meta, program, inputs, outputs, name, *, arch):
    fp64 = meta.get("dtype") is torch.float64
    targets = {k: v for k, v in program["w"].items() if k in outputs}
    needed = {q for pairs in targets.values() for q, _ in pairs}
    products = {q: v for q, v in program["cg"].items() if q in needed}
    gradients = sorted({g for pairs in targets.values() for _, g in pairs})
    product_indices = {q: index for index, q in enumerate(products)}
    gradient_indices = {g: index for index, g in enumerate(gradients)}
    # Reuse multiple CG products across a wider channel tile when it does not
    # add padded channel work. Scope A fragments to each product so the
    # larger per-thread accumulator does not keep every CG operand live.
    if fp64:
        tile_columns, tile_rows, tile_reduction, mma_rows = 64, 64, 16, 8
    else:
        tile_columns = (
            128
            if len(products) > 1
            and all(
                ((spec["out_channels"] + 63) // 64) % 2 == 0
                for spec in meta["generated_specs"]
            )
            else 64
        )
        # The wide tile reuses G across more paths. A shorter reduction tile keeps
        # its CG staging area unchanged while giving each producer one element.
        tile_rows = 64 if tile_columns == 128 else 32
        tile_reduction = 16 if tile_columns == 128 else 32
        mma_rows = 8 if tile_columns == 128 else 2
    threads = tile_rows * tile_columns // (mma_rows * 4)
    if fp64:
        buffers = 1
        resident_blocks = 2
    else:
        buffers = (
            2
            if len(products) * tile_rows * tile_reduction
            + len(gradients) * tile_reduction * tile_columns
            <= 6144
            else 1
        )
        resident_blocks = max(1, (4 if len(products) == 1 else 2) * 256 // threads)
    xdim, ydim, odim = (meta[k] for k in ("in1_dim", "in2_dim", "out_dim"))
    # Schedule long reductions first so the last wave contains lighter tiles.
    specs = sorted(
        meta["generated_specs"],
        key=lambda spec: (
            -(len(spec["terms"]) + tile_columns * spec["d3"])
            / (2 if spec["d3"] == 1 else 4),
            spec["weight_offset"],
        ),
    )
    output_tiles = sum(
        ((s["path_channels"] + tile_rows - 1) // tile_rows)
        * ((s["out_channels"] + tile_columns - 1) // tile_columns)
        for s in specs
    )
    # Round up: a grid just below the target should still fill the next wave.
    split_scale = min(4, (128 + max(1, output_tiles) - 1) // max(1, output_tiles))
    tasks, totals, cg_cases, vector_cases = [], [], [], []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        paths, channels, edge = (
            spec[k] for k in ("path_channels", "out_channels", "edge_channels")
        )
        splits = (2 if d3 == 1 else 4) * split_scale
        columns, rows = (
            (channels + tile_columns - 1) // tile_columns,
            (paths + tile_rows - 1) // tile_rows,
        )
        count = rows * columns * splits
        totals.append(str(count))
        tasks.append(
            r"""{{
            if(task>=cursor && task<cursor+{count}) {{
                Index local=task-cursor,tile=local/{splits}; int split=local%{splits};
                spec_id={index}; components={d3}; paths={paths}; channels={channels};
                out_offset={oo}; weight_offset={wo};
                first_path=(tile/{columns})*{tile_rows}; first_channel=(tile%{columns})*{tile_columns};
                Index tiles=(items*{d3}+{reduction_mask})/{tile_reduction}, per_split=(tiles+{split_mask})/{splits};
                k_begin=split*per_split*{tile_reduction};k_end=min(items*{d3},k_begin+per_split*{tile_reduction});
            }}
            cursor+={count};
        }}""".format(
                count=count,
                splits=splits,
                index=index,
                d3=d3,
                paths=paths,
                channels=channels,
                oo=oo,
                wo=wo,
                columns=columns,
                tile_rows=tile_rows,
                tile_columns=tile_columns,
                reduction_mask=tile_reduction - 1,
                tile_reduction=tile_reduction,
                split_mask=splits - 1,
            )
        )
        lines = [
            f"case {index}: {{",
            f"int u=p/{edge},v=p%{edge};",
            "switch(component) {",
        ]
        for component in range(d3):
            lines.append(f"case {component}: {{")
            for qi, (q, pairs) in enumerate(products.items()):
                lines.append("{ float value=0.f;")
                for x, y in pairs:
                    for i, j, k, c in spec["terms"]:
                        if k == component:
                            lines.append(
                                f"""value=fmaf({float_literal(c, meta.get("dtype"))},__ldg({x}+int64_t(item)*{xdim}+{xo}+u*{d1}+{i})*__ldg({y}+int64_t(item)*{ydim}+{yo}+v*{d2}+{j}),value);"""
                            )
                lines.append(f"shared_a[buffer][{qi}][index]=value; }}")
            lines += ["break;", "}"]
        lines += ["}", "break;", "}"]
        cg_cases.append("\n".join(lines))
        vector_lines = [f"case {index}: {{", f"int u=p/{edge},v=p%{edge};"]
        for qi, pairs in enumerate(products.values()):
            vector_lines += [
                "{",
                *(f"float z{k}=0.f;" for k in range(d3)),
                "if(p<paths && item<items) {",
                emit_cg_source(meta, spec, pairs),
                "}",
            ]
            for k in range(d3):
                vector_lines += [
                    r"""{{
Index reduction=item*{d3}+{k};
if(reduction>=tile_k && reduction<tile_k+{tile_reduction}) {{
shared_a[buffer][{qi}][(reduction-tile_k)*{tile_rows}+local_path]=reduction<k_end?z{k}:0.f;
}}
}}""".format(
                        d3=d3,
                        k=k,
                        tile_reduction=tile_reduction,
                        qi=qi,
                        tile_rows=tile_rows,
                    )
                ]
            vector_lines.append("}")
        vector_lines += ["break;", "}"]
        vector_cases.append("\n".join(vector_lines))

    loads, fragments, mma, stores = [], [], [], []
    for gi, g in enumerate(gradients):
        if (
            not fp64
            and tile_columns == 128
            and odim % 4 == 0
            and all(
                s["out_offset"] % 4 == 0 and s["out_channels"] % 4 == 0 for s in specs
            )
        ):
            scatter = []
            for offset, member in enumerate(("x", "y", "z", "w")):
                scatter.append(
                    r"""{{
                    int component=(inside+{offset})%components,column=(inside+{offset})/components;
                    Index reduction=item*components+component;
                    if(reduction>=tile_k && reduction<tile_k+{tile_reduction})
                        shared_b[buffer][{gi}][(reduction-tile_k)*{tile_columns}+column]=reduction<k_end?value.{member}:0.f;
                }}""".format(
                        offset=offset,
                        tile_reduction=tile_reduction,
                        gi=gi,
                        tile_columns=tile_columns,
                        member=member,
                    )
                )
            loads.append(
                r"""
            int rows_{g}=(tile_k%components+{tile_reduction}+components-1)/components;
            for(int index=threadIdx.x;index<rows_{g}*{tile_columns}*components/4;index+={threads}) {{
                int scalar=index*4,local_item=scalar/({tile_columns}*components);
                Index inside=scalar-local_item*{tile_columns}*components,item=tile_k/components+local_item;
                float4 value={{0.f,0.f,0.f,0.f}};
                if(item<items && first_channel*components+inside+3<channels*components)
                    value=*reinterpret_cast<const float4*>({g}+int64_t(item)*{odim}+out_offset+first_channel*components+inside);
                {scatter}
            }}""".format(
                    g=g,
                    tile_reduction=tile_reduction,
                    tile_columns=tile_columns,
                    threads=threads,
                    odim=odim,
                    scatter=chr(10).join(scatter),
                )
            )
        else:
            loads += [
                r"""for(int index=threadIdx.x;index<{tile_reduction}*{tile_columns};index+={threads}) {{
            Index reduction=tile_k+index/{tile_columns},channel=first_channel+index%{tile_columns};
            Index item=reduction/components; int component=reduction%components;
            shared_b[buffer][{gi}][index]=reduction<k_end && channel<channels ?
                __ldg({g}+int64_t(item)*{odim}+out_offset+channel*components+component) : 0.f;
        }}""".format(
                    tile_reduction=tile_reduction,
                    tile_columns=tile_columns,
                    threads=threads,
                    gi=gi,
                    g=g,
                    odim=odim,
                )
            ]
        # Components are uniform within a CTA. Specializing this small load
        # body avoids repeating general integer division for every G element.
        if tile_columns == 128:
            body = loads[-1]
            loads[-1] = (
                "switch(components) {\n"
                + "\n".join(
                    f"""case {d}: {{ {body.replace("components", str(d))} break; }}"""
                    for d in sorted({spec["d3"] for spec in specs})
                )
                + "\n}"
            )
        fragments += [f"Mma::FragmentB b{gi};"]
        fragments += [
            f"b{gi}[{n}]=shared_b[buffer][{gi}][k*{tile_columns}+local_column+{n}];"
            for n in range(4)
        ]
    if tile_columns == 64:
        for qi, q in enumerate(products):
            fragments.append(
                f"Mma::FragmentA a{qi};\na{qi}[0]=shared_a[buffer][{qi}][k*{tile_rows}+local_row];"
            )
            fragments += [
                f"a{qi}[{i}]=shared_a[buffer][{qi}][k*{tile_rows}+local_row+{i}];"
                for i in range(1, mma_rows)
            ]
    for ti, (target, pairs) in enumerate(targets.items()):
        for q, g in pairs:
            if tile_columns == 128:
                qi = product_indices[q]
                mma += ["{", "Mma::FragmentA a;"]
                mma += [
                    f"a[{i}]=shared_a[buffer][{qi}][k*{tile_rows}+local_row+{i}];"
                    for i in range(mma_rows)
                ]
                mma.extend(
                    (f"Mma()(acc[{ti}],a,b{gradient_indices[g]},acc[{ti}]);", "}")
                )
            else:
                mma.append(
                    f"Mma()(acc[{ti}],a{product_indices[q]},b{gradient_indices[g]},acc[{ti}]);"
                )
        stores.append(
            f"atomicAdd({target}+weight_offset+p*channels+channel,acc[{ti}][m*4+n]);"
        )
    cg_loads = r"""        for(int index=threadIdx.x;index<{tile_rows}*{tile_reduction};index+={threads}) {{
            Index p=first_path+index%{tile_rows},reduction=tile_k+index/{tile_rows};
            Index item=reduction/components; int component=reduction%components;
            if(p<paths && reduction<k_end) {{
                switch(spec_id) {{ {cg_cases} }}
            }} else {{
                #pragma unroll
                for(int q=0;q<{products_count};++q) shared_a[buffer][q][index]=0.f;
            }}
        }}
""".format(
        tile_rows=tile_rows,
        tile_reduction=tile_reduction,
        threads=threads,
        cg_cases=chr(10).join(cg_cases),
        products_count=len(products),
    )
    if tile_columns == 128 or fp64:
        cg_loads = r"""
        int item_count=(tile_k%components+{tile_reduction}+components-1)/components;
        for(int index=threadIdx.x;index<item_count*{tile_rows};index+={threads}) {{
            int local_path=index%{tile_rows},p=first_path+local_path;
            Index item=tile_k/components+index/{tile_rows};
            switch(spec_id) {{ {vector_cases} }}
        }}
""".format(
            tile_reduction=tile_reduction,
            tile_rows=tile_rows,
            threads=threads,
            vector_cases=chr(10).join(vector_cases),
        )

    params = [f"const float* __restrict__ {v}" for v in inputs]
    params += [f"float* __restrict__ {v}" for v in outputs] + ["Index items"]
    if fp64:
        shared_bytes = (
            8
            * buffers
            * (
                len(products) * tile_rows * tile_reduction
                + len(gradients) * tile_reduction * tile_columns
            )
        )
        shared_declaration = r"""    extern __shared__ __align__(16) unsigned char storage[];
    auto shared_a=reinterpret_cast<float (*)[{products_count}][{tile_rows}*{tile_reduction}]>(storage);
    auto shared_b=reinterpret_cast<float (*)[{gradients_count}][{tile_reduction}*{tile_columns}]>(
        storage+sizeof(float)*{feature_storage});
""".format(
            products_count=len(products),
            tile_rows=tile_rows,
            tile_reduction=tile_reduction,
            gradients_count=len(gradients),
            tile_columns=tile_columns,
            feature_storage=buffers * len(products) * tile_rows * tile_reduction,
        )
        mma_definition = f"    double acc[{len(targets)}][2][4][4]={{}};\n"
        tensor_product_loads = "\n".join(
            r"""double a{product}[8];
#pragma unroll
for(int i=0;i<8;++i) a{product}[i]=shared_a[buffer][{product}][((lane&3)+(i>>1)*4)*{tile_rows}+local_row+m*16+(i&1)*8];""".format(product=qi, tile_rows=tile_rows)
            for qi in range(len(products))
        )
        tensor_gradient_loads = "\n".join(
            r"""double b{gradient}[4];
#pragma unroll
for(int i=0;i<4;++i) b{gradient}[i]=shared_b[buffer][{gradient}][((lane&3)+i*4)*{tile_columns}+local_column+n*8+(lane>>2)];""".format(
                gradient=gi, tile_columns=tile_columns
            )
            for gi in range(len(gradients))
        )
        tensor_contractions = []
        for ti, (target, pairs) in enumerate(targets.items()):
            for q, g in pairs:
                a = f"a{product_indices[q]}"
                b = f"b{gradient_indices[g]}"
                c = f"acc[{ti}][m][n]"
                tensor_contractions.append(
                    f"{get_fp64_mma(arch)[1]}::fma({c}[0],{c}[1],{c}[2],{c}[3],{a}[0],{a}[1],{a}[2],{a}[3],{a}[4],{a}[5],{a}[6],{a}[7],{b}[0],{b}[1],{b}[2],{b}[3],{c}[0],{c}[1],{c}[2],{c}[3]);"
                )
        mma_compute = r"""    for(Index tile_k=k_begin;tile_k<k_end;tile_k+={tile_reduction}) {{
        int buffer=(tile_k/{tile_reduction})%{buffers};
{cg_loads}        {loads}
        __syncthreads();
            const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
        const int local_row=(warp>>1)*32+(lane>>2),local_column=(warp&1)*32;
        #pragma unroll
        for(int m=0;m<2;++m) {{
{product_loads}
#pragma unroll
for(int n=0;n<4;++n) {{
{gradient_loads}
{contractions}
}}
}}
        __syncthreads();
    }}
""".format(
            tile_reduction=tile_reduction,
            buffers=buffers,
            cg_loads=cg_loads,
            loads="\n".join(loads),
            product_loads=tensor_product_loads,
            gradient_loads=tensor_gradient_loads,
            contractions="\n".join(tensor_contractions),
        )
        mma_stores = r"""const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
for(int m=0;m<2;++m) {{
#pragma unroll
for(int n=0;n<4;++n) {{
#pragma unroll
for(int i=0;i<4;++i) {{
int p=first_path+(warp>>1)*32+m*16+(lane>>2)+(i>>1)*8,channel=first_channel+(warp&1)*32+n*8+(lane&3)*2+(i&1);
if(p<paths && channel<channels) {{
{stores}
}}
}}
}}
}}""".format(
            stores="\n".join(
                f"atomicAdd({target}+weight_offset+int64_t(p)*channels+channel,acc[{ti}][m][n][i]);"
                for ti, target in enumerate(targets)
            )
        )
    else:
        shared_declaration = f"    __shared__ float shared_a[{buffers}][{len(products)}][{tile_rows}*{tile_reduction}];\n    __shared__ float shared_b[{buffers}][{len(gradients)}][{tile_reduction}*{tile_columns}];\n"
        mma_definition = r"""    using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<{mma_rows},4,1>,
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC acc[{targets_count}];
    #pragma unroll
    for(int t=0;t<{targets_count};++t) acc[t].clear();
    int local_row=(threadIdx.x/{row_threads})*{mma_rows},local_column=(threadIdx.x&{column_mask})*4;
""".format(
            mma_rows=mma_rows,
            targets_count=len(targets),
            row_threads=tile_columns // 4,
            column_mask=tile_columns // 4 - 1,
        )
        mma_compute = r"""    for(Index tile_k=k_begin;tile_k<k_end;tile_k+={tile_reduction}) {{
        int buffer=(tile_k/{tile_reduction})%{buffers};
{cg_loads}        {loads}
        __syncthreads();
        {unroll}
        for(int k=0;k<{tile_reduction};++k) {{
            {fragments}
            {mma}
        }}
        {barrier}
    }}
""".format(
            buffers=buffers,
            cg_loads=cg_loads,
            loads="\n".join(loads),
            unroll="#pragma unroll 4" if mma_rows > 2 else "#pragma unroll",
            tile_reduction=tile_reduction,
            fragments=chr(10).join(fragments),
            mma=chr(10).join(mma),
            barrier="__syncthreads();" if buffers == 1 else "",
        )
        mma_stores = r"""    #pragma unroll
    for(int m=0;m<{mma_rows};++m) {{
        #pragma unroll
        for(int n=0;n<4;++n) {{
            int p=first_path+local_row+m,channel=first_channel+local_column+n;
            if(p<paths && channel<channels) {{ {stores} }}
        }}
    }}""".format(
            mma_rows=mma_rows,
            stores=chr(10).join(stores),
        )
    code = r"""
__global__ __launch_bounds__({threads},{resident_blocks}) void {name}({params}) {{
{shared_declaration}    Index task=blockIdx.x,cursor=0; int spec_id=0,components=0,channels=0,paths=0;
    int out_offset=0,weight_offset=0,first_path=0,first_channel=0; Index k_begin=0,k_end=0;
    {tasks}
{mma_definition}{mma_compute}{mma_stores}
}}
""".format(
        threads=threads,
        resident_blocks=resident_blocks,
        name=name,
        params=",".join(params),
        shared_declaration=shared_declaration,
        tasks=chr(10).join(tasks),
        mma_definition=mma_definition,
        mma_compute=mma_compute,
        mma_stores=mma_stores,
    )
    if fp64:
        code += f"\nconstexpr int {name}_shared_bytes={shared_bytes};\n"
    return (
        cuda_source_for_dtype(code, meta.get("dtype", torch.float32)),
        "+".join(totals),
        threads,
    )


def make_uvw_backward_external_kernel(meta, program, inputs, outputs, kernel_name):
    """Emit a path-major kernel; eight lanes cooperate on each path channel."""
    specs = meta["generated_specs"]
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    w_outputs = {k: v for k, v in program["w"].items() if k in outputs}
    needed_z = {
        z for pairs in (*x_outputs.values(), *y_outputs.values()) for _, z in pairs
    }
    contractions = {k: v for k, v in program["z"].items() if k in needed_z}
    needed_q = {q for pairs in w_outputs.values() for q, _ in pairs}
    products = {k: v for k, v in program["cg"].items() if k in needed_q}
    weights = sorted({w for pairs in contractions.values() for w, _ in pairs})
    gradients = sorted(
        {g for pairs in (*contractions.values(), *w_outputs.values()) for _, g in pairs}
    )
    max_edge = max(s["d2"] * s["edge_channels"] for s in specs)
    max_d3 = max(s["d3"] for s in specs)
    xdim, ydim, odim, wdim = (
        meta[k] for k in ("in1_dim", "in2_dim", "out_dim", "weight_numel")
    )
    cases = []
    for index, spec in enumerate(specs):
        d1, d2, d3 = (spec[k] for k in ("d1", "d2", "d3"))
        xo, yo, oo, wo = (
            spec[k] for k in ("x_offset", "y_offset", "out_offset", "weight_offset")
        )
        edges, channels = spec["edge_channels"], spec["out_channels"]
        edge_size = edges * d2
        # Large third-order contractions otherwise keep Z, all CG epilogue
        # inputs and both edge accumulators live together. Shared memory bounds
        # that register lifetime without materializing a global intermediate.
        stage_z = len(contractions) * d3 > 24
        lines = [f"case {index}: {{"]
        lines += [f"float sums_{target}[{edge_size}]={{0.f}};" for target in y_outputs]
        lines += [
            r"""#pragma unroll 1
for(int p=path_group;p<{path_channels};p+=16) {{
int u=p/{edges},v=p%{edges};
const unsigned mask={mask};""".format(
                path_channels=spec["path_channels"],
                edges=edges,
                # Four eight-lane path groups fill a warp. Aligned path counts
                # give every lane the same loop trip count within that warp.
                mask="0xffffffffu"
                if spec["path_channels"] % 4 == 0
                else "__activemask()",
            )
        ]
        if stage_z:
            lines.append("{")
        lines.append(r"""using ZMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1,{d3},1>,
float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;""".format(d3=d3))
        for name in contractions:
            lines += [f"ZMma::FragmentC contract_{name}; contract_{name}.clear();"]
        if products:
            lines.append(r"""using WMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1,1,{d3}>,
float,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,
float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;""".format(d3=d3))
        for name, pairs in products.items():
            lines += [f"WMma::FragmentA product_{name};", "{"]
            lines += [f"float z{k}=0.f;" for k in range(d3)]
            lines.append("if(lane==0) {")
            # Reuse the CG accumulator while releasing each pair of inputs.
            for pair in pairs:
                lines += ["{", emit_cg_source(meta, spec, pairs=(pair,)), "}"]
            lines.append("}")
            lines += [
                f"product_{name}[{k}]=__shfl_sync(mask,z{k},0,8);" for k in range(d3)
            ]
            lines.append("}")
        lines += ["#pragma unroll 1", f"for(int c=lane;c<{channels};c+=8) {{"]
        for name in weights:
            lines.append(
                f"ZMma::FragmentA weight_{name};\nweight_{name}[0]=__ldg({name}+int64_t(item)*{wdim}+{wo}+p*{channels}+c);"
            )
        for name in gradients:
            lines += [f"ZMma::FragmentB grad_{name};"]
            lines += [
                f"grad_{name}[{k}]=__ldg({name}+int64_t(item)*{odim}+{oo}+c*{d3}+{k});"
                for k in range(d3)
            ]
        for target, pairs in contractions.items():
            for w, g in pairs:
                lines.append(
                    f"ZMma()(contract_{target},weight_{w},grad_{g},contract_{target});"
                )
        for target, pairs in w_outputs.items():
            products_source = "\n".join(
                f"WMma()(delta_{target},product_{q},grad_{g},delta_{target});"
                for q, g in pairs
            )
            lines.append(
                r"""WMma::FragmentC delta_{target}; delta_{target}.clear();
{products}
{target}[int64_t(item)*{wdim}+{wo}+p*{channels}+c]=delta_{target}[0];""".format(
                    target=target,
                    products=products_source,
                    wdim=wdim,
                    wo=wo,
                    channels=channels,
                )
            )
        lines.append("}")
        if contractions:
            lines += ["#pragma unroll", "for(int delta=4;delta>0;delta>>=1) {"]
            for name in contractions:
                lines += [
                    f"contract_{name}[{k}]+=__shfl_down_sync(mask,contract_{name}[{k}],delta,8);"
                    for k in range(d3)
                ]
            lines.append("}")
        # Release high-order Z before loading the CG epilogue inputs.
        z_values = {}
        for z_index, name in enumerate(contractions):
            for k in range(d3):
                if stage_z:
                    address = f"path_group*{len(contractions) * d3}+{z_index * d3 + k}"
                    lines += [f"if(lane==0) reduced_z[{address}]=contract_{name}[{k}];"]
                    z_values[name, k] = f"reduced_z[{address}]"
                else:
                    z_values[name, k] = f"contract_{name}[{k}]"
        if stage_z:
            lines.append("}")
        if x_outputs or y_outputs:
            lines.append("if(lane==0) {")
            # A single contraction can feed X and Y together, overlapping
            # their loads and arithmetic. Higher orders release one output's
            # inputs before starting the next to bound register lifetime.
            roles = [(target, pairs, "x") for target, pairs in x_outputs.items()]
            roles += [(target, pairs, "y") for target, pairs in y_outputs.items()]
            groups = [roles] if len(contractions) == 1 else [[role] for role in roles]
            for group in groups:
                lines.append("{")
                for target, pairs, axis in sorted(
                    group, key=itemgetter(2), reverse=True
                ):
                    source_axis = 1 if axis == "x" else 0
                    dim, offset, width, channel = (
                        (ydim, yo, d2, "v") if axis == "x" else (xdim, xo, d1, "u")
                    )
                    for name in sorted({name for name, _ in pairs}):
                        for component in sorted(
                            {t[source_axis] for t in spec["terms"]}
                        ):
                            lines += [
                                f"float value_{name}_{component}=__ldg({name}+int64_t(item)*{dim}+{offset}+{channel}*{width}+{component});"
                            ]
                    prefix = target if len(group) > 1 else "acc"
                    for component in sorted(
                        {t[1 - source_axis] for t in spec["terms"]}
                    ):
                        lines += [f"float {prefix}{component}=0.f;"]
                for i, j, k, coeff in spec["terms"]:
                    for target, pairs, axis in group:
                        component, source = (i, j) if axis == "x" else (j, i)
                        prefix = target if len(group) > 1 else "acc"
                        for name, z in pairs:
                            lines.append(
                                f"{prefix}{component}=fmaf({float_literal(coeff)}*value_{name}_{source},{z_values[z, k]},{prefix}{component});"
                            )
                for target, pairs, axis in group:
                    prefix = target if len(group) > 1 else "acc"
                    for component in sorted(
                        {t[0 if axis == "x" else 1] for t in spec["terms"]}
                    ):
                        if axis == "x":
                            lines += [
                                f"atomicAdd({target}+int64_t(item)*{xdim}+{xo}+u*{d1}+{component},{prefix}{component});"
                            ]
                        else:
                            lines += [
                                f"sums_{target}[v*{d2}+{component}]+={prefix}{component};"
                            ]
                lines.append("}")
            lines.append("}")
        lines.append("}")
        if y_outputs:
            partial_stores = "\n".join(
                f"edge_partial[{target_index * 16 * edge_size}+path_group*{edge_size}+j]=sums_{target}[j];"
                for target_index, target in enumerate(y_outputs)
            )
            reductions = "\n".join(
                r"""{{
float value=0.f;
#pragma unroll
for(int group=0;group<16;++group) value+=edge_partial[{partial_offset}+group*{edge_size}+j];
atomicAdd({target}+int64_t(item)*{ydim}+{yo}+j,value);
}}""".format(
                    partial_offset=target_index * 16 * edge_size,
                    edge_size=edge_size,
                    target=target,
                    ydim=ydim,
                    yo=yo,
                )
                for target_index, target in enumerate(y_outputs)
            )
            lines.append(
                r"""if(lane==0) {{
#pragma unroll
for(int j=0;j<{edge_size};++j) {{
{partial_stores}
}}
}}
__syncthreads();
for(int j=threadIdx.x;j<{edge_size};j+=128) {{
{reductions}
}}""".format(
                    edge_size=edge_size,
                    partial_stores=partial_stores,
                    reductions=reductions,
                )
            )
        lines += ["return;", "}"]
        cases.append("\n".join(lines))

    parameters = [f"const float* __restrict__ {name}" for name in inputs]
    parameters += [f"float* __restrict__ {name}" for name in outputs]
    storage = []
    if y_outputs:
        storage.append(
            f"__shared__ float edge_partial[{16 * len(y_outputs) * max_edge}];"
        )
    if len(contractions) * max_d3 > 24:
        storage.append(
            f"__shared__ volatile float reduced_z[{16 * len(contractions) * max_d3}];"
        )
    # Large directional contractions need a larger per-thread register budget
    # to avoid spilling the Z fragments into local memory.
    resident_blocks = 2 if len(contractions) * max_d3 > 48 else 4
    return r"""
__global__ __launch_bounds__(128,{resident_blocks})
void {kernel_name}({parameters}) {{
    int lane=threadIdx.x&7,path_group=threadIdx.x>>3;
    Index item=blockIdx.x;
    {storage}
    switch(blockIdx.y) {{
{cases}
    }}
}}
""".format(
        resident_blocks=resident_blocks,
        kernel_name=kernel_name,
        parameters=",".join(parameters),
        storage=chr(10).join(storage),
        cases=chr(10).join(cases),
    )


def make_uvw_backward_external_source(meta, program, inputs, outputs, phase, *, arch):
    prefix = f"lunatp_uvw_{phase}_external"
    matrix_outputs = tuple(name for name in outputs if name != "gg")
    max_d3 = max(s["d3"] for s in meta["generated_specs"])
    # Separate XY and weight work when their combined fragments would crowd
    # registers. Both phases use this same emitter and write final outputs.
    if (len(program["z"]) + len(program["cg"])) * max_d3 > 64:
        phases = (
            tuple(k for k in matrix_outputs if k not in program["w"]),
            tuple(k for k in matrix_outputs if k in program["w"]),
        )
    else:
        phases = (matrix_outputs,)
    kernels, launches = [], []
    for index, phase_outputs in enumerate(phases):
        if not phase_outputs:
            continue
        name = f"{prefix}_fused_{index}_fp32"
        kernels.append(
            make_uvw_backward_external_kernel(
                meta, program, inputs, phase_outputs, name
            )
        )
        args = ",".join(
            f"{name}.data_ptr<float>()" for name in (*inputs, *phase_outputs)
        )
        launches.append(
            f"""{name}<<<dim3(items,{len(meta["generated_specs"])}),128,0,stream>>>({args});"""
        )
    if "gg" in outputs:
        products = tuple((program["cg"][q], w) for q, w in program["forward"])
        name = f"{prefix}_g_fp32"
        kernel, grid, clear, pointer_order = make_uvw_forward_external_kernel(
            meta, products, name, arch=arch
        )
        kernels.insert(0, kernel)
        if clear:
            launches.append(
                "C10_CUDA_CHECK(cudaMemsetAsync(gg.data_ptr<float>(),0,gg.numel()*sizeof(float),stream));"
            )
        args = ",".join(f"{name}.data_ptr<float>()" for name in pointer_order)
        launches.append(
            f"{name}<<<{grid},128,0,stream>>>({args},gg.data_ptr<float>(),items);"
        )
    allocations = [
        f"auto {name}=at::empty_like({OUTPUT_SHAPES[name]});" for name in outputs
    ]
    clears = [
        f"C10_CUDA_CHECK(cudaMemsetAsync({name}.data_ptr<float>(),0,{name}.numel()*sizeof(float),stream));"
        for name in outputs
        if name in (*program["x"], *program["y"])
    ]
    source = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>
{kernels}
std::vector<at::Tensor> launch({contractions}) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=x.size(0);
    {allocations}
    if(items==0) return {{{outputs}}};
    auto stream=at::cuda::getCurrentCUDAStream();
    {clears}
    {launches}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""".format(
        kernels="\n".join(kernels),
        contractions=",".join((f"at::Tensor {name}" for name in inputs)),
        allocations=chr(10).join(allocations),
        outputs=",".join(outputs),
        clears=chr(10).join(clears),
        launches=chr(10).join(launches),
    )
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))



def make_uvw_backward_shared_source(meta, program, inputs, outputs, phase, *, arch):
    if (meta.get("dtype", torch.float32) is torch.float32
        and meta["generated_specs"]
        and all(s["d2"] == 1 and s["d1"] == s["d3"]
                and len(s["terms"]) == s["d1"]
                and all(i == k and j == 0 and value == s["terms"][0][3] for i,j,k,value in s["terms"])
                for s in meta["generated_specs"])):
        return make_uvw_scalar_backward_source(meta, program, inputs, outputs, phase)
    kernels, launches = [], []
    for role, emitter in (
        ("xy", make_uvw_backward_dxdy_shared_kernel),
        ("weight", make_uvw_backward_dw_shared_kernel),
    ):
        targets = tuple(
            v
            for v in outputs
            if v
            in (program["w"] if role == "weight" else (*program["x"], *program["y"]))
        )
        if not targets:
            continue
        name = f"lunatp_uvw_{phase}_shared_{role}_fused_fp32"
        code, grid, threads = emitter(meta, program, inputs, targets, name, arch=arch)
        kernels.append(code)
        args = ",".join(f"{v}.data_ptr<float>()" for v in (*inputs, *targets))
        if meta.get("dtype") is torch.float64:
            launches.append(
                f"C10_CUDA_CHECK(cudaFuncSetAttribute({name},cudaFuncAttributeMaxDynamicSharedMemorySize,{name}_shared_bytes));{name}<<<{grid},{threads},{name}_shared_bytes,stream>>>({args},items);"
            )
        else:
            launches.append(f"{name}<<<{grid},{threads},0,stream>>>({args},items);")
    independent_launches = len(launches)
    if "gg" in outputs:
        output_kernels, output_launches = [], []
        output_groups = {}
        for output_spec in meta["generated_specs"]:
            output_key = (
                output_spec["out_offset"],
                output_spec["out_channels"],
                output_spec["d3"],
            )
            output_groups.setdefault(output_key, []).append(output_spec)
        # Bound the sparse-CG instruction body of each output tile. Splits
        # still reduce directly into GG, so there is no global CG workspace.
        output_chunks = []
        for output_group in output_groups.values():
            output_chunk, output_terms = [], 0
            for output_spec in output_group:
                if output_chunk and output_terms + len(output_spec["terms"]) > 128:
                    output_chunks.append(output_chunk)
                    output_chunk, output_terms = [], 0
                output_chunk.append(output_spec)
                output_terms += len(output_spec["terms"])
            output_chunks.append(output_chunk)
        for output_index, output_specs in enumerate(output_chunks):
            output_name = f"lunatp_uvw_{phase}_shared_g_{output_index}_fused_fp32"
            output_products = tuple(
                (program["cg"][output_q], output_w)
                for output_q, output_w in program["forward"]
            )
            output_code, output_grid, _, output_pointers, output_threads = (
                make_uvw_forward_shared_kernel(
                    {**meta, "generated_specs": output_specs},
                    output_name,
                    output_products,
                    arch=arch,
                    include_mma=int(arch.rstrip("af")) >= 90,
                    wide_channels=all(
                        output_s["out_channels"] >= 128
                        for output_s in meta["generated_specs"]
                    ),
                )
            )
            output_kernels.append(output_code)
            output_args = ",".join(
                f"{output_v}.data_ptr<float>()" for output_v in output_pointers
            )
            output_launches.append(
                f"{output_name}<true><<<{output_grid},{output_threads},0,stream>>>({output_args},gg.data_ptr<float>(),items,gg.size(1),{output_grid});"
            )
        kernels.extend(output_kernels)
        launches.extend(output_launches)

    allocations = [f"auto {v}=at::empty_like({OUTPUT_SHAPES[v]});" for v in outputs]
    clears = [
        f"if({v}.numel()) C10_CUDA_CHECK(cudaMemsetAsync({v}.data_ptr<float>(),0,{v}.numel()*sizeof(float),stream));"
        for v in outputs
    ]
    # Keep saturated FP64 first/second orders serial; graph replay favors that
    # schedule. Independent output branches overlap in the other shared paths.
    stream_source = ""
    if (
        independent_launches and len(launches) > 1
        and (meta.get("dtype") is not torch.float64 or phase == "triple_bwd")
    ):
        stream_source = DERIVATIVE_STREAM_SOURCE + "\n"
        launches[0] = (
            "auto& lane = derivative_stream(x.get_device(), stream);"
            "DerivativeFork fork(lane, stream);"
            + launches[0].replace(",stream>>>", ",lane.stream>>>")
        )
        launches.append("fork.join();")
    source = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>
{stream_source}{kernels}
std::vector<at::Tensor> launch({contractions}) {{
    c10::cuda::CUDAGuard guard(x.device());
    Index items=x.size(0);
    {allocations}
    auto stream=at::cuda::getCurrentCUDAStream();
    {clears}
    if(items) {{ {launches} }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""".format(
        stream_source=stream_source,
        kernels="\n".join(kernels),
        contractions=",".join((f"at::Tensor {v}" for v in inputs)),
        allocations=chr(10).join(allocations),
        clears=chr(10).join(clears),
        launches=chr(10).join(launches),
        outputs=",".join(outputs),
    )

    if meta.get("dtype") is torch.float64:
        source = get_fp64_mma(arch)[0] + "\n" + source
    return cuda_source_for_dtype(source, meta.get("dtype", torch.float32))


def run(op, order, *inputs, needs=None):
    outputs = PROGRAMS[order]["outputs"]
    selected = (
        outputs
        if needs is None
        else tuple(name for name, needed in zip(outputs, needs) if needed)
    )
    missing = 0
    if order >= 2:
        missing = (inputs[4] is None) | ((inputs[5] is None) << 1) | ((inputs[6] is None) << 2)
    if order == 3:
        missing |= ((inputs[7] is None) << 3) | ((inputs[8] is None) << 4) | ((inputs[9] is None) << 5) | ((inputs[10] is None) << 6)
    if missing and not (
        op.weight_mode == "shared" and op.compute_dtype is torch.float32
        and op.meta["scalar_input"]
    ):
        x, y, w, g = inputs[:4]
        templates = (x, y, w, g, x, y, w, x, y, w, g)
        inputs = tuple(torch.zeros_like(template) if value is None else value
                       for value, template in zip(inputs, templates))
        missing = 0
    device = inputs[0].get_device()
    if device != op.arch_device:
        op.arch = get_cuda_arch(inputs[0].device)
        op.arch_device = device
    key = op.arch, order, selected, missing
    launch = op.kernels.get(key)
    if launch is None:
        program = PROGRAMS[order]
        input_names = INPUTS[:len(inputs)]
        if missing:
            input_names = tuple(name for name, value in zip(input_names, inputs)
                                if value is not None)
            # None is a zero incoming gradient. Remove its products at JIT time;
            # requested zero outputs keep the existing zero-tensor semantics.
            filtered = dict(program)
            for contraction in ("cg", "z"):
                filtered[contraction] = {}
                for name, pairs in program[contraction].items():
                    kept = tuple(pair for pair in pairs
                                 if all(v in input_names for v in pair))
                    if kept:
                        filtered[contraction][name] = kept
            for feature in ("x", "y"):
                filtered[feature] = {}
                for out, pairs in program[feature].items():
                    kept = tuple((v, term) for v, term in pairs
                                 if v in input_names and term in filtered["z"])
                    if kept:
                        filtered[feature][out] = kept
            filtered["w"] = {}
            for out, pairs in program["w"].items():
                kept = tuple((term, v) for term, v in pairs
                             if v in input_names and term in filtered["cg"])
                if kept:
                    filtered["w"][out] = kept
            filtered["forward"] = tuple(
                (term, v) for term, v in program["forward"]
                if v in input_names and term in filtered["cg"]
            )
            program = filtered
        dtype_tag = "fp64" if op.compute_dtype == torch.float64 else "fp32"
        phase = ("bwd", "double_bwd", "triple_bwd")[order - 1]
        make_source = (
            make_uvw_backward_shared_source
            if op.weight_mode == "shared"
            else make_uvw_backward_external_source
        )
        source = make_source(
            op.meta,
            program,
            input_names,
            selected,
            phase,
            arch=op.arch,
        )
        extension = load_cuda_extension(
            f"lunatp_uvw_{phase}_{op.weight_mode}_fused_{dtype_tag}",
            source,
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )
        launch = op.kernels[key] = extension.launch
    result = launch(*(value for value in inputs if value is not None)) if missing else launch(*inputs)
    if needs is None:
        return tuple(result)
    values = iter(result)
    return tuple(next(values) if needed else None for needed in needs)
