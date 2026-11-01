"""NormActivation backward CUDA kernels and bias reduction."""

from math import gcd
from textwrap import indent

import torch

from .graph import BIAS_OUTPUTS, VECTOR_INPUTS, build_irrep_program
from .utils import (
    MAX_SHARED_BYTES,
    COMPUTATION_NAMES,
    output_width,
    emit_scalar_lines,
)
from ....utils.precision import float_literal
from ...activation.kernels.scalar import activation_cuda_prelude


def emit_radial_geometry(meta, program, components, input_address):
    """Shared ordered component sums and activation coefficients for FP32."""
    order = program["order"]
    input_at = lambda role, component=0: (
        input_address(role, component) if role in program["inputs"] else "Element(0)"
    )
    summaries = ["q", "p"]
    if order >= 2:
        summaries += ["tr", "h"]
    if order == 3:
        summaries += ["j", "m", "kk", "e", "ff"]
    lines = [f"Element {name}=0;" for name in summaries]
    for component in range(components):
        lines += f"""{{
    const Element X={input_at('x', component)};
    const Element G={input_at('grad_out', component)};
    const Element R={input_at('rx', component)};
    const Element T={input_at('bx', component)};
    const Element K={input_at('bg', component)};""".splitlines()
        lines += ["    q=__fmaf_rn(X,X,q);", "    p=__fmaf_rn(X,G,p);"]
        if order >= 2:
            lines += ["    tr=__fmaf_rn(X,R,tr);", "    h=__fmaf_rn(G,R,h);"]
        if order == 3:
            lines += [
                "    j=__fmaf_rn(X,T,j);", "    m=__fmaf_rn(G,T,m);",
                "    kk=__fmaf_rn(R,T,kk);", "    e=__fmaf_rn(X,K,e);", "    ff=__fmaf_rn(R,K,ff);",
            ]
        lines += ["}"]

    eps = float_literal(meta["eps_squared"], meta["dtype"])
    activation = "::tanhf" if meta["activation"]["kind"] == "tanh" else "activation_sigmoid"
    lines += f"""const bool clamped=q<{eps};
const Element radius=__fsqrt_rn(clamped?{eps}:q);
const Element iv=Element(1)/radius;
const Element F={activation}(radius+{input_at('bias')});""".splitlines()
    if meta["activation"]["kind"] == "tanh":
        lines += r"""const Element F1=__fsub_rn(Element(1),__fmul_rn(F,F));
const Element F2=-Element(2)*F*F1;
const Element F3=-Element(2)*F1*(Element(1)-Element(3)*F*F);""".splitlines()
    else:
        lines += r"""const Element F1=F*(Element(1)-F);
const Element F2=F1*(Element(1)-Element(2)*F);
const Element F3=F1*(Element(1)-Element(6)*F+Element(6)*F*F);""".splitlines()
    lines += f"""const Element a=F*iv,ab=F1*iv;
const Element U={input_at('rb')},Z={input_at('bb')};
if(clamped) {{
    const Element abb=F2*iv,abbb=F3*iv;""".splitlines()
    return lines


def emit_radial_backward(meta, program, components, input_address, output_address):
    """Emit radial backward expressions."""
    input_at = lambda role, component=0: (
        input_address(role, component) if role in program["inputs"] else "Element(0)"
    )
    outputs = program["active_outputs"]
    lines = emit_radial_geometry(meta, program, components, input_address)
    if "db" in outputs:
        lines.append(f"    {output_address('db', 0)}=ab*p;")
    for component in range(components):
        if "dx" in outputs:
            lines.append(f"    {output_address('dx', component)}=a*{input_at('grad_out', component)};")
    lines += ["} else {"]
    if components == 1:
        lines += f"""    if(isfinite(radius) && radius==fabsf({input_at('x')})) {{
        const Element sign=({input_at('x')}<Element(0))?Element(-1):Element(1);
        const Element direction={input_at('rx')}+sign*U;
        const Element gradient={input_at('bx')}+sign*Z;""".splitlines()
        if "dx" in outputs:
            lines.append(f"        {output_address('dx', 0)}=F1*{input_at('grad_out', 0)};")
        if "db" in outputs:
            lines.append(f"        {output_address('db', 0)}=sign*F1*{input_at('grad_out', 0)};")
        lines += ["    } else {"]
    radial_indent = "        " if components == 1 else "    "
    lines += [radial_indent + "const Element eta=p*iv,c=F1-a,d=c*iv;"]
    if "db" in outputs:
        lines.append(radial_indent + f"{output_address('db', 0)}=F1*eta;")
    for component in range(components):
        if "dx" in outputs:
            lines.append(
                radial_indent + f"{output_address('dx', component)}=a*{input_at('grad_out', component)}"
                f"+c*eta*iv*{input_at('x', component)};"
            )
    if components == 1:
        lines += ["    }"]
    lines += ["}"]
    return lines, [(meta["activation"]["kind"], {})]


def make_backward_source(meta, program):
    names = program["active_outputs"]
    count = meta["bias_dim"]
    scalar = "double" if meta["dtype"] == torch.float64 else "float"
    biased = tuple(name for name in names if name in BIAS_OUTPUTS)
    max_components = max((path["components"] for path in meta["paths"]), default=1)
    vectors = tuple(name for name in names if name not in biased)
    shared_bytes = 128 * (max_components * len(vectors) + len(biased)) * (8 if meta["dtype"] == torch.float64 else 4)
    copy_tile = gcd(32, *(path["copies"] for path in meta["paths"]))
    coalesced = meta["dtype"] == torch.float32 and copy_tile >= 8 and shared_bytes <= MAX_SHARED_BYTES
    lines, atoms, tiles = [], [], []
    tile_offset = 0
    begin = 0
    for path in meta["paths"]:
        components, copies = path["components"], path["copies"]

        local = build_irrep_program(meta, program, path)
        lines += [
            f"if(group >= {begin} && group < {begin + copies}) {{",
            f"    const Index copy=group-{begin};",
        ]

        input_address = lambda role, column: (
            f"{role}[row*{role}_row_stride+"
            f"({path['input_offset'] if role in VECTOR_INPUTS or role in ('grad_out', 'bg') else path['scalar_offset']}"
            f"+copy*{components if role in VECTOR_INPUTS or role in ('grad_out', 'bg') else 1}+{column})"
            f"*{role}_col_stride]"
        )

        output_address = lambda name, column: (
            (f"contribution_{name}" if coalesced else f"acc_{name}")
            if name in biased
            else f"values_{name}[lane*{components}+{column}]"
            if coalesced and components * (8 if meta["dtype"] == torch.float64 else 4) > 16
            else f"out_{name}[row*{output_width(meta, name)}+{path['input_offset']}"
            f"+copy*{components}+{column}]"
        )

        if (
            meta["dtype"] == torch.float32
            and meta["normalize"]
            and meta.get("norm_squared", True)
            and meta["eps_squared"] >= 1e-16
            and meta["activation"]["kind"] in ("tanh", "sigmoid")
        ):
            if program["order"] == 1:
                body, extra = emit_radial_backward(meta, program, components, input_address, output_address)
            elif program["order"] == 2:
                from .double_backward import emit_radial_double_backward
                body, extra = emit_radial_double_backward(meta, program, components, input_address, output_address)
            else:
                from .triple_backward import emit_radial_triple_backward
                body, extra = emit_radial_triple_backward(meta, program, components, input_address, output_address)
        else:
            body, extra = emit_scalar_lines(meta["dtype"], local, input_address, output_address)
        lines.extend("    " + line for line in body)
        lines.append("}")
        atoms += extra
        if coalesced:
            width = copies // copy_tile
            stores = []
            if components * (8 if meta["dtype"] == torch.float64 else 4) > 16:
                stores.append("__syncwarp();\nif(row<num_items) {")
                for name in vectors:
                    stores.append(f"""    #pragma unroll
    for(int c=0;c<{components};++c) {{
        const Index column=Index(c)*{copy_tile}+lane;
        out_{name}[row*{output_width(meta,name)}+{path['input_offset']}+copy_base*{components}+column]=values_{name}[column];
    }}""")
                stores.append("}\n__syncwarp();")
            reductions = []
            for name in biased:
                reduction = [f"Element sum_{name}=acc_{name};"]
                for delta in (8, 16):
                    if delta >= copy_tile:
                        reduction.append(f"sum_{name}+=__shfl_down_sync(0xffffffff,sum_{name},{delta});")
                reduction.append(f"if((threadIdx.x&31)<{copy_tile}) shared_{name}[warp*{copy_tile}+lane]=sum_{name};")
                reductions.extend(reduction)
            if biased:
                reductions.append("__syncthreads();")
                for name in biased:
                    reductions.append(f"""if(warp==0 && (threadIdx.x&31)<{copy_tile}) {{
    Element sum=shared_{name}[lane];
    for(int w=1;w<4;++w) sum+=shared_{name}[w*{copy_tile}+lane];
    out_{name}[row_tile*{count}+group]=sum;
}}""")
                reductions.append("__syncthreads();")
            declarations = [f"Element* values_{name}=shared_values_{name}+row_group*{copy_tile}*{max_components};" for name in vectors]
            declarations += [f"Element acc_{name}=0;" for name in biased]
            contributions = [f"Element contribution_{name}=0;" for name in biased]
            accumulate = [f"acc_{name}+=contribution_{name};" for name in biased]
            tiles.append(f"""if(group_tile>={tile_offset} && group_tile<{tile_offset+width}) {{
    const Index copy_base=(group_tile-{tile_offset})*{copy_tile};
    const Index copy=copy_base+lane;
    const Index group={path['scalar_offset']}+copy;
{indent(chr(10).join(declarations), '    ')}
    #pragma unroll 1
    for(int step=0;step<{copy_tile//8};++step) {{
        const Index row=row_tile*16+step*{128//copy_tile}+row_group;
{indent(chr(10).join(contributions), '        ')}
        if(row<num_items) {{
{indent(chr(10).join(body), '            ')}
        }}
{indent(chr(10).join(accumulate), '        ')}
{indent(chr(10).join(stores), '        ')}
    }}
{indent(chr(10).join(reductions), '    ')}
}}""")
            tile_offset += width
        begin += copies
    assert begin == count
    parameters = [
        f"const Element* __restrict__ {role},Index {role}_row_stride,Index {role}_col_stride"
        for role in program["inputs"]
    ]
    parameters += [f"Element* __restrict__ out_{name}" for name in names]
    args = [
        f"{role}.data_ptr<Element>(),Index({role}.dim()==1?0:{role}.stride(0)),Index({role}.stride(-1))"
        for role in program["inputs"]
    ]
    args += [
        f"{'partial' if name in biased else 'result'}_{name}.data_ptr<Element>()" for name in names
    ]
    allocations = []
    for name in names:
        if name in biased:
            allocations += [
                f"auto result_{name}=torch::empty({{int64_t({count})}},options);",
                f"auto partial_{name}=torch::empty({{row_tiles,int64_t({count})}},options);",
            ]
        else:
            allocations.append(
                f"auto result_{name}=torch::empty({{num_items,int64_t({output_width(meta, name)})}},options);"
            )
    summaries = []
    for name in biased:
        summaries += r"""Element sum_{name}=acc_{name};
sum_{name}+=__shfl_down_sync(0xffffffff,sum_{name},8);
sum_{name}+=__shfl_down_sync(0xffffffff,sum_{name},16);
if((threadIdx.x&31)<8) shared_{name}[(threadIdx.x>>5)*8+col]=sum_{name};""".format(name=name).splitlines()
    summary_writes = []
    for name in biased:
        summary_writes += r"""Element value_{name}=shared_{name}[col];
for(int warp=1;warp<4;++warp) value_{name}+=shared_{name}[warp*8+col];
out_{name}[row_tile*{count}+group]=value_{name};""".format(name=name, count=count).splitlines()
    reduction_parameters = [
        f"const Element* partial_{name},Element* result_{name}" for name in biased
    ]
    reduction_body, reduction_writes = [], []
    for name in biased:
        reduction_body += r"""Element sum_{name}=0;
if(group<{count}) for(Index row=threadIdx.x/8;row<row_tiles;row+=32) sum_{name}+=partial_{name}[row*{count}+group];
sum_{name}+=__shfl_down_sync(0xffffffff,sum_{name},8);
sum_{name}+=__shfl_down_sync(0xffffffff,sum_{name},16);
if((threadIdx.x&31)<8) shared_{name}[warp*8+lane]=sum_{name};""".format(name=name, count=count).splitlines()
        reduction_writes += r"""Element value_{name}=shared_{name}[lane];
for(int w=1;w<8;++w) value_{name}+=shared_{name}[w*8+lane];
result_{name}[group]=value_{name};""".format(name=name).splitlines()
    reduction_kernel = ""
    reduction_launch = ""
    if biased:
        reduction_kernel = r"""
__global__ void norm_activation_shared_bias_reduce({parameters},int64_t row_tiles) {{
{shared_memory}
    const int lane=threadIdx.x&7,warp=threadIdx.x>>5;
    const Index group=Index(blockIdx.x)*8+lane;
{partial_sums}
    __syncthreads();
    if(threadIdx.x<8 && group<{count}) {{
{final_stores}
    }}
}}
""".format(
            parameters=','.join(reduction_parameters),
            shared_memory=indent(chr(10).join((f'__shared__ Element shared_{name}[64];' for name in biased)), '    '),
            partial_sums=indent(chr(10).join(reduction_body), "    "),
            count=count,
            final_stores=indent(chr(10).join(reduction_writes), "        "),
        )
        reduction_launch = (
            "norm_activation_shared_bias_reduce<<<"
            + str((count + 7) // 8)
            + ",256,0,stream>>>("
            + ",".join(
                f"partial_{name}.data_ptr<Element>(),result_{name}.data_ptr<Element>()"
                for name in biased
            )
            + ",row_tiles);"
        )
    returns = [
        f"pybind11::cast(result_{name})" if name in names else "pybind11::none()"
        for name in program["outputs"]
    ]
    first = program["inputs"][0]
    if coalesced:
        kernel_shared = [f"__shared__ __align__(16) Element shared_values_{name}[128*{max_components}];" for name in vectors]
        kernel_shared += [f"__shared__ Element shared_{name}[128];" for name in biased]
        kernel_body = f"""{chr(10).join(kernel_shared)}
const int lane=threadIdx.x%{copy_tile},warp=threadIdx.x>>5;
const int row_group=threadIdx.x/{copy_tile};
const Index row_tiles=(num_items+15)/16;
for(Index tile=blockIdx.x;tile<row_tiles*{tile_offset};tile+=gridDim.x) {{
    const Index row_tile=tile/{tile_offset};
    const Index group_tile=tile%{tile_offset};
{indent(chr(10).join(tiles), "    ")}
}}"""
        group_tiles = tile_offset
    else:
        kernel_body = r"""{shared_memory}
const Index group_tiles=({count}+7)/8;
const Index row_tiles=(num_items+15)/16;
const int col=threadIdx.x&7;
for(Index tile=blockIdx.x;tile<row_tiles*group_tiles;tile+=gridDim.x) {{
    const Index row_tile=tile/group_tiles;
    const Index group=(tile%group_tiles)*8+col;
    const Index row=row_tile*16+threadIdx.x/8;
{bias_accumulators}
    if(row<num_items && group<{count}) {{
{statements}
    }}
{warp_summaries}
    __syncthreads();
    if(threadIdx.x<8 && group<{count}) {{
{summary_writes}
    }}
    __syncthreads();
}}""".format(
            shared_memory=chr(10).join(f'__shared__ Element shared_{name}[32];' for name in biased),
            count=count,
            bias_accumulators=indent(chr(10).join(f'Element acc_{name}=0;' for name in biased), '    '),
            statements=indent(chr(10).join(lines), "        "), warp_summaries=indent(chr(10).join(summaries), "    "),
            summary_writes=indent(chr(10).join(summary_writes), "        "),
        )
        group_tiles = (count + 7) // 8
    return r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
using Element={scalar};
{prelude}
__global__ __launch_bounds__(128,1) void {kernel}({parameters},int64_t num_items) {{
{kernel_body}
}}
{reduction_kernel}
pybind11::tuple launch({tensor_parameters},int64_t num_items) {{
    c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();
    const int64_t row_tiles=(num_items+15)/16;
    const int64_t group_tiles={group_tiles};
{allocations}
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    if(num_items>0) {{
        {kernel}<<<std::min<int64_t>(65535,row_tiles*group_tiles),128,0,stream>>>({arguments},num_items);
        {reduction_launch}
    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pybind11::make_tuple({returns});
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{
    m.def("launch",&launch);
}}
""".format(
        kernel=f"norm_activation_{COMPUTATION_NAMES[program['order']]}",
        kernel_body=indent(kernel_body, "    "), group_tiles=group_tiles,
        scalar=scalar,
        prelude=activation_cuda_prelude(meta['dtype'], atoms),
        parameters=','.join(parameters),
        reduction_kernel=reduction_kernel,
        tensor_parameters=','.join((f'torch::Tensor {role}' for role in program['inputs'])),
        first=first,
        allocations=indent(chr(10).join(allocations), "    "),
        arguments=','.join(args),
        reduction_launch=reduction_launch,
        returns=','.join(returns),
    )
