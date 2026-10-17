"""UVW convolution backward CUDA and native launch."""

from __future__ import annotations
from collections.abc import Mapping
from operator import itemgetter
import torch
from ...utils.cg import emit_cg_source
from ....utils.precision import get_fp64_mma, cuda_source_for_dtype, float_literal
from ....utils.extension import load_tp_extension, load_cuda_extension
from ...utils.runtime import auxiliary_streams, get_overlap_state
from ...utils.contractions import (
    ENV_INPUTS,
    ENV_PROGRAMS,
    INPUTS,
    PROGRAMS,
    ENV_OUTPUT_SHAPES,
    OUTPUT_SHAPES,
)
from .forward import make_uvw_conv_forward_external_kernel
from .utils import get_uvw_conv_node_layout, get_uvw_conv_weight_layout, set_arch


def make_uvw_conv_gemm_mma(meta, a_read, b_read, *, arch):
    """Emit FP64 warp MMA for the existing 16x64, K32 shared GEMM tiles.

    Each of the four warps owns 16 columns and all 16 rows. Read expressions
    use local row/column/input coordinates and product index p. The same
    accumulator sums the existing one- or two-product contraction.
    """
    if meta.get("dtype") != torch.float64:
        raise ValueError("The warp MMA fragment requires FP64 metadata")
    return {
        "declaration": r"""using Mma={mma_type};
    const int mma_lane=threadIdx.x&31,mma_warp=threadIdx.x>>5;
    double acc[2][4]={{}};""".format(mma_type=get_fp64_mma(arch)[1]),
        "compute": r"""#pragma unroll
        for(int k=0;k<32;k+=16) {{
            #pragma unroll
            for(int p=0;p<Products;++p) {{
                double av[8];
                #pragma unroll
                for(int i=0;i<8;++i) {{
                    const int row=(mma_lane>>2)+(i&1)*8;
                    const int input=k+(mma_lane&3)+(i>>1)*4;
                    av[i]={a_read};
                }}
                #pragma unroll
                for(int n=0;n<2;++n) {{
                    double bv[4];
                    #pragma unroll
                    for(int i=0;i<4;++i) {{
                        const int column=mma_warp*16+n*8+(mma_lane>>2);
                        const int input=k+(mma_lane&3)+i*4;
                        bv[i]={b_read};
                    }}
                    Mma::fma(acc[n][0],acc[n][1],acc[n][2],acc[n][3],
                        av[0],av[1],av[2],av[3],av[4],av[5],av[6],av[7],
                        bv[0],bv[1],bv[2],bv[3],
                        acc[n][0],acc[n][1],acc[n][2],acc[n][3]);
                }}
            }}
        }}""".format(
            a_read=a_read,
            b_read=b_read,
        ),
        "row": "row_base+(mma_lane>>2)+(r>>1)*8",
        "column": "column_base+mma_warp*16+c*8+(mma_lane&3)*2+(r&1)",
        "value": "acc[c][r]",
        "rows": 4,
        "columns": 2,
    }


def make_uvw_conv_node_shared_source(meta, kernel_name, *, arch):
    specs = meta["generated_specs"]
    offsets, total = get_uvw_conv_node_layout(specs)
    column_index = "int64_t(column)" if total > (1 << 31) - 1 else "column"
    cases = []
    a_store = "i"
    b_store = "u*32+((i%32)^(u&31))"
    mma = {
        "declaration": r"""using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<2,4,1>,
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC acc;acc.clear();""",
        "compute": r"""#pragma unroll 4
        for(int k=0;k<32;++k) {
            #pragma unroll
            for(int p=0;p<Products;++p) {
                Mma::FragmentA av;av[0]=a[p][k*16+local_row];av[1]=a[p][k*16+local_row+1];
                Mma::FragmentB bv;
                #pragma unroll
                for(int c=0;c<4;++c) {
                    int u=local_column+c*16;
                    bv[c]=b[p][u*32+(k^(u&31))];
                }
                Mma()(acc,av,bv,acc);
            }
        }""",
        "row": "row_base+local_row+r",
        "column": "column_base+local_column+c*16",
        "value": "acc[r*4+c]",
        "rows": 2,
        "columns": 4,
    }
    if meta.get("dtype") == torch.float64:
        a_store = "(i/16)*16+((i%16)^(((i/16)&3)<<2))"
        b_store = "u*32+((i%32)^((u&3)<<2))"
        mma = make_uvw_conv_gemm_mma(
            meta,
            "a[p][input*16+(row^((input&3)<<2))]",
            "b[p][column*32+(input^((column&3)<<2))]",
            arch=arch,
        )
    column_type = (
        "int64_t" if max(s["path_channels"] for s in specs) > (1 << 31) - 64 else "int"
    )
    column_base = (
        "int64_t(blockIdx.x)*64" if column_type == "int64_t" else "blockIdx.x*64"
    )
    for index, s in enumerate(specs):
        d = s["d3"]
        columns = s["path_channels"]
        kdim = s["out_channels"]
        input_type = "int64_t" if kdim > (1 << 31) - 32 else "int"
        if input_type == "int64_t":
            dimensions = f"constexpr {column_type} Columns={columns}; constexpr int64_t K={kdim};"
            load_position = (
                f"int u=i/32; int64_t k=base+i%32; {column_type} column=column_base+u;"
            )
            grad_position = "Index row=row_base+i%16; int64_t k=base+i/16;"
        else:
            grad_position = "Index row=row_base+i%16,k=base+i/16;"
            if column_type == "int64_t":
                dimensions = (
                    f"constexpr int64_t Columns={columns}; constexpr int K={kdim};"
                )
                load_position = "int u=i/32,k=base+i%32; int64_t column=column_base+u;"
            else:
                dimensions = f"constexpr int Columns={columns},K={kdim};"
                load_position = "int u=i/32,k=base+i%32,column=column_base+u;"
        cases.append(
            r"""
case {index}: {{
    {dimensions}
    const Index rows=nodes*{d};
    if(row_base>=rows || column_base>=Columns) return;
    {declaration}
    for({input_type} base=0;base<K;base+=32) {{
        #pragma unroll
        for(int p=0;p<Products;++p) {{
            const float* grad=p==0?g0:g1;
            const float* weight=p==0?w0:w1;
            for(int i=threadIdx.x;i<512;i+=128) {{
                {grad_position}
                a[p][{a_store}]=(row<rows && k<K)?__ldg(grad+int64_t(row/{d})*{out_dim}+{out_offset}+k*{d}+row%{d}):0.f;
            }}
            for(int i=threadIdx.x;i<2048;i+=128) {{
                {load_position}
                b[p][{b_store}]=(column<Columns && k<K)?__ldg(weight+{weight_offset}+int64_t(column)*K+k):0.f;
            }}
        }}
        __syncthreads();
        {compute}
        __syncthreads();
    }}
    #pragma unroll
    for(int r=0;r<{rows};++r) {{
        Index row={row};
        #pragma unroll
        for(int c=0;c<{columns};++c) {{
            {column_type} column={column};
            if(row<rows && column<Columns) z[int64_t(row/{d})*{total}+{offsets_value}+{column_index}*{d}+row%{d}]={value};
        }}
    }}
    return;
}}
""".format(
                index=index,
                dimensions=dimensions,
                d=d,
                declaration=mma["declaration"],
                input_type=input_type,
                grad_position=grad_position,
                a_store=a_store,
                out_dim=meta["out_dim"],
                out_offset=s["out_offset"],
                load_position=load_position,
                b_store=b_store,
                weight_offset=s["weight_offset"],
                compute=mma["compute"],
                rows=mma["rows"],
                row=mma["row"],
                columns=mma["columns"],
                column_type=column_type,
                column=mma["column"],
                total=total,
                offsets_value=offsets[index],
                column_index=column_index,
                value=mma["value"],
            )
        )
    return cuda_source_for_dtype(
        r"""
// W*G stays in node-major Z; source-CSR and edge consumers share this layout.
// XOR-swizzled weights give coalesced global loads and conflict-free MMA reads.
template<int Products>
__global__ __launch_bounds__(128,4)
void {kernel_name}(const float* w0,const float* g0,const float* w1,const float* g1,float* z,int64_t nodes) {{
    __shared__ float a[Products][512],b[Products][2048];
    const Index row_base=blockIdx.y*16; const {column_type} column_base={column_base};
    const int local_row=(threadIdx.x/16)*2,local_column=threadIdx.x%16;
    switch(blockIdx.z) {{
{cases}
    }}
}}
""".format(
            kernel_name=kernel_name,
            column_type=column_type,
            column_base=column_base,
            cases=chr(10).join(cases),
        ),
        meta.get("dtype", torch.float32),
    )


def grad_x_allocator(meta):
    """Check contiguous input-irrep coverage without enumerating channels."""
    intervals = sorted(
        {
            (
                int(spec["x_offset"]),
                int(spec["x_offset"]) + int(spec["in_channels"]) * int(spec["d1"]),
            )
            for spec in meta["generated_specs"]
        }
    )
    end = 0
    for begin, stop in intervals:
        if begin > end:
            return "torch::zeros"
        end = max(end, stop)
    return "torch::empty" if end == int(meta["in1_dim"]) else "torch::zeros"


def emit_dx_lines(
    meta,
    z_offsets,
    kernel_name,
    threads,
    channel_bound,
    *,
    paired=False,
    wide_z=False,
    products=None,
):
    """Emit source-CSR contraction; large accumulator sets own separate regions.

    Keeping at most one input irrep live avoids the high register pressure of
    accumulating every input component across all neighbors. Small contractions
    retain one block per source/channel tile. Both variants write final outputs.
    """
    specs = list(meta["generated_specs"])
    in1_dim, in2_dim = int(meta["in1_dim"]), int(meta["in2_dim"])
    slots = []
    for spec in specs:
        for component in range(int(spec["d1"])):
            key = (
                int(spec["x_offset"]),
                int(spec["in_channels"]),
                int(spec["d1"]),
                component,
            )
            if key not in slots:
                slots.append(key)
    slot_index = {
        (offset, dim, component): index
        for index, (offset, _, dim, component) in enumerate(slots)
    }
    groups = {}
    for index, spec in enumerate(specs):
        # Limit the compile-time accumulator count.
        key = int(spec["x_offset"]) if len(slots) > 32 else 0
        groups.setdefault(key, []).append((index, spec))
    arrays = ("edge0", "z0", "edge1", "z1") if paired else ("edge_attr", "z")
    if products is not None:
        arrays = tuple(
            sorted(
                {
                    role
                    for terms in products.values()
                    for term in terms
                    for role in term
                    if role is not None
                }
            )
        )
    targets = tuple(products) if products is not None else ("grad_x",)
    parameters = [
        f"""    const float* __restrict__ {("ptr_" if products is not None else "")}{name},"""
        for name in arrays
    ]
    lines = [
        r"""__global__ __launch_bounds__({threads}, 4)
void {kernel_name}(
{parameters}
    const int32_t* __restrict__ dst,
    const EdgeIndex* __restrict__ src_rowptr,
    const EdgeIndex* __restrict__ edge_by_src,
{output_parameters}
    int64_t num_nodes) {{
    const int s = blockIdx.x;
    const int u = blockIdx.y * {threads} + threadIdx.x;
    if (s >= num_nodes || u >= {channel_bound}) return;""".format(
            threads=threads,
            kernel_name=kernel_name,
            parameters="\n".join(parameters),
            output_parameters="\n".join(
                [f"    float* __restrict__ {target}," for target in targets]
            ),
            channel_bound=channel_bound,
        )
    ]
    for group, group_specs in enumerate(groups.values()):
        offsets = {int(spec["x_offset"]) for _, spec in group_specs}
        owned = [
            (index, slot) for index, slot in enumerate(slots) if slot[0] in offsets
        ]
        if len(groups) > 1:
            lines.append(f"    if (blockIdx.z == {group}) {{")
        for slot, _ in owned:
            if products is None:
                lines.append(f"    float acc_x{slot} = 0.0f;")
            else:
                lines += [f"    float acc_{target}{slot} = 0.0f;" for target in targets]
        lines += [r"""    const Index e_begin = __ldg(src_rowptr + s);
    const Index e_end = __ldg(src_rowptr + s + 1);
    for (Index pos = e_begin; pos < e_end; ++pos) {
        const Index e = __ldg(edge_by_src + pos);
        const int t = __ldg(dst + e);"""]
        if products is not None:
            for scalar in sorted(
                {
                    term[2]
                    for terms in products.values()
                    for term in terms
                    if term[2] is not None
                }
            ):
                lines.append(
                    f"        const float scale_{scalar}=__ldg(ptr_{scalar}+e);"
                )
        for spec_idx, spec in group_specs:
            x_offset, y_offset = int(spec["x_offset"]), int(spec["y_offset"])
            in_channels, d1, d3 = (
                int(spec["in_channels"]),
                int(spec["d1"]),
                int(spec["d3"]),
            )
            terms = list(spec["terms"])
            lines.append(f"        if (u < {in_channels}) {{")
            edge_channels, d2 = int(spec["edge_channels"]), int(spec["d2"])
            if edge_channels != 1:
                lines.append(f"        for (int v = 0; v < {edge_channels}; ++v) {{")
            edge_offset = f"v * {d2} + " if edge_channels != 1 else ""
            channel = "int64_t(u)" if wide_z else "u"
            path = (
                f"({channel} * {edge_channels} + v)" if edge_channels != 1 else channel
            )
            for j in sorted({int(t[1]) for t in terms}):
                if products is not None:
                    for feature in sorted(
                        {term[0] for terms in products.values() for term in terms}
                    ):
                        lines.append(
                            f"            const float value_{feature}_{spec_idx}_{j}=__ldg(ptr_{feature}+e*{in2_dim}+{edge_offset}{y_offset + j});"
                        )
                elif paired:
                    for product in range(2):
                        lines.append(
                            f"            const float y{product}v{spec_idx}_{j} = __ldg(edge{product} + e * {in2_dim} + {edge_offset}{y_offset + j});"
                        )
                else:
                    lines.append(
                        f"            const float yv{spec_idx}_{j} = __ldg(edge_attr + e * {in2_dim} + {edge_offset}{y_offset + j});"
                    )
            for k in sorted({int(t[2]) for t in terms}):
                if products is not None:
                    for matrix in sorted(
                        {term[1] for terms in products.values() for term in terms}
                    ):
                        lines.append(
                            f"            const float value_{matrix}_{spec_idx}_{k}=__ldg(ptr_{matrix}+Index(t)*GEN_TOTAL_Z+{z_offsets[spec_idx]}+{path}*{d3}+{k});"
                        )
                elif paired:
                    for product in range(2):
                        lines.append(
                            f"            const float z{product}v{spec_idx}_{k} = __ldg(z{product} + Index(t) * GEN_TOTAL_Z + {z_offsets[spec_idx]} + {path} * {d3} + {k});"
                        )
                else:
                    lines.append(
                        f"            const float zv{spec_idx}_{k} = __ldg(z + Index(t) * GEN_TOTAL_Z + {z_offsets[spec_idx]} + {path} * {d3} + {k});"
                    )
            for i, j, k, coeff in terms:
                slot = slot_index[x_offset, d1, int(i)]
                if products is not None:
                    for target, contractions in products.items():
                        for feature, matrix, scalar in contractions:
                            factor = f"*scale_{scalar}" if scalar is not None else ""
                            lines.append(
                                f"""            acc_{target}{slot}=fmaf({float_literal(float(coeff), meta.get("dtype"))}*value_{feature}_{spec_idx}_{int(j)}{factor},value_{matrix}_{spec_idx}_{int(k)},acc_{target}{slot});"""
                            )
                elif paired:
                    lines.append(
                        f"""            acc_x{slot} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, y0v{spec_idx}_{int(j)} * z0v{spec_idx}_{int(k)} + y1v{spec_idx}_{int(j)} * z1v{spec_idx}_{int(k)}, acc_x{slot});"""
                    )
                else:
                    lines.append(
                        f"""            acc_x{slot} = fmaf({float_literal(float(coeff), meta.get("dtype"))} * yv{spec_idx}_{int(j)}, zv{spec_idx}_{int(k)}, acc_x{slot});"""
                    )
            if edge_channels != 1:
                lines.append("        }")
            lines.append("        }")
        lines.append("    }")
        for slot, (offset, channels, dim, component) in owned:
            if products is None:
                lines.append(
                    f"    if (u < {channels}) grad_x[Index(s) * {in1_dim} + {offset} + u * {dim} + {component}] = acc_x{slot};"
                )
            else:
                lines += [
                    f"    if (u < {channels}) {target}[Index(s)*{in1_dim}+{offset}+u*{dim}+{component}]=acc_{target}{slot};"
                    for target in targets
                ]
        if len(groups) > 1:
            lines += ["    return;", "    }"]
    lines.append("}")
    return lines, (f", {len(groups)}" if len(groups) > 1 else "")


def make_uvw_conv_backward_dx_shared_source(meta: Mapping[str, object], *, arch) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    z_offsets, total_z = get_uvw_conv_node_layout(specs)
    wide_z = total_z > (1 << 31) - 1
    z_type = "int64_t" if wide_z else "int"
    max_channels = max(int(s["in_channels"]) for s in specs)
    max_paths = max(int(s["path_channels"]) for s in specs)
    z_channels = (
        "GEN_MAX_PATH_CHANNELS" if max_paths != max_channels else "GEN_MAX_CHANNELS"
    )
    path_type = "int64_t" if max_paths > (1 << 31) - 64 else "int"
    path_constant = (
        f"constexpr {path_type} GEN_MAX_PATH_CHANNELS = {max_paths};\n"
        if max_paths != max_channels
        else ""
    )
    max_d3 = max(int(s["d3"]) for s in specs)
    x_tile = 64 if max_channels <= 64 else 128
    channel_type = "int64_t" if max_channels > (1 << 31) - x_tile else "int"

    grad_x_alloc = grad_x_allocator(meta)

    x_lines, x_group_grid = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_x_src_apply_fp32",
        "GEN_X_TILE",
        "GEN_MAX_CHANNELS",
        wide_z=wide_z,
    )

    mma_include = (
        "\n" + get_fp64_mma(arch)[0] if meta.get("dtype") == torch.float64 else ""
    )
    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>{mma_include}
#include <cuda.h>
#include <cuda_runtime.h>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_IN1_DIM = {in1_dim};
constexpr {z_type} GEN_TOTAL_Z = {total_z};
{path_constant}constexpr {channel_type} GEN_MAX_CHANNELS = {max_channels};
constexpr int GEN_MAX_D3 = {max_d3};
constexpr int GEN_X_TILE = {x_tile};
constexpr int GEN_Z_TILE_M = 16;
constexpr int GEN_Z_TILE_N = 64;


{make_uvw_conv_node_shared_source}

{x_lines}
}}  // namespace

torch::Tensor lunatp_uvwconv_bwd_dx_shared_launch_fp32(
    torch::Tensor edge_attr,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge_attr.device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, edge_attr.options());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, edge_attr.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_x_zgemm_fp32<1><<<dim3(
        static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
        static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
        GEN_NUM_SPECS), 128, 0, stream>>>(
        weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    if (num_nodes > 0) lunatp_uvwconv_bwd_x_src_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
        static_cast<unsigned int>((GEN_MAX_CHANNELS + GEN_X_TILE - 1) / GEN_X_TILE){x_group_grid}), GEN_X_TILE, 0, stream>>>(
        edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
        src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_dx_shared_launch_fp32", &lunatp_uvwconv_bwd_dx_shared_launch_fp32, "Graph-conv shared grad_x source-CSR backward launch");
}}
""".format(
            mma_include=mma_include,
            specs_count=len(specs),
            in1_dim=in1_dim,
            z_type=z_type,
            total_z=total_z,
            path_constant=path_constant,
            channel_type=channel_type,
            max_channels=max_channels,
            max_d3=max_d3,
            x_tile=x_tile,
            make_uvw_conv_node_shared_source=make_uvw_conv_node_shared_source(
                meta, "lunatp_uvwconv_bwd_x_zgemm_fp32", arch=arch
            ),
            x_lines=chr(10).join(x_lines),
            grad_x_alloc=grad_x_alloc,
            z_channels=z_channels,
            x_group_grid=x_group_grid,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_dxdy_shared_source(
    meta: Mapping[str, object], *, arch
) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    in2_dim = int(meta["in2_dim"])
    z_offsets, total_z = get_uvw_conv_node_layout(specs)
    wide_z = total_z > (1 << 31) - 1
    z_type = "int64_t" if wide_z else "int"
    max_in_channels = max(int(s["in_channels"]) for s in specs)
    max_paths = max(int(s["path_channels"]) for s in specs)
    z_channels = (
        "GEN_MAX_PATH_CHANNELS"
        if max_paths != max_in_channels
        else "GEN_MAX_IN_CHANNELS"
    )
    path_type = "int64_t" if max_paths > (1 << 31) - 64 else "int"
    path_constant = (
        f"constexpr {path_type} GEN_MAX_PATH_CHANNELS = {max_paths};\n"
        if max_paths != max_in_channels
        else ""
    )
    max_d2 = max(int(s["d2"]) for s in specs)
    max_d3 = max(int(s["d3"]) for s in specs)
    x_threads = 64 if max_in_channels <= 64 else 128
    y_threads = 64 if max_in_channels <= 64 else 128
    channel_type = "int64_t" if max_in_channels > (1 << 31) - x_threads else "int"

    grad_x_alloc = grad_x_allocator(meta)

    x_lines, x_group_grid = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_xy_source_x_apply_fp32",
        "GEN_X_THREADS",
        "GEN_MAX_IN_CHANNELS",
        wide_z=wide_z,
    )
    x_pair_lines, _ = emit_dx_lines(
        meta,
        z_offsets,
        "lunatp_uvwconv_bwd_xy_source_x_pair_apply_fp32",
        "GEN_X_THREADS",
        "GEN_MAX_IN_CHANNELS",
        paired=True,
        wide_z=wide_z,
    )

    edge_kernels = []
    for paired, edge_kernel_name in (
        (False, "lunatp_uvwconv_bwd_xy_source_y_apply_fp32"),
        (True, "lunatp_uvwconv_bwd_xy_source_y_pair_apply_fp32"),
    ):
        edge_specs = list(meta["generated_specs"])
        edge_in1_dim, edge_in2_dim = int(meta["in1_dim"]), int(meta["in2_dim"])
        # Paired contractions keep two sets of feature and node values live.
        # Their smaller budget preserves occupancy without splitting compact singles.
        edge_component_budget = 32 if paired else 48
        edge_groups = {}
        for edge_index, edge_spec in enumerate(edge_specs):
            edge_key = (
                int(edge_spec["y_offset"])
                if edge_in2_dim > edge_component_budget
                else 0
            )
            edge_groups.setdefault(edge_key, []).append((edge_index, edge_spec))
        edge_covered = {
            int(edge_spec["y_offset"]) + edge_j
            for edge_spec in edge_specs
            for edge_j in range(int(edge_spec["edge_channels"]) * int(edge_spec["d2"]))
        }
        edge_arrays = ("x0", "z0", "x1", "z1") if paired else ("x", "z")
        edge_lines = [
            "__global__ __launch_bounds__(GEN_Y_THREADS, 4)",
            f"void {edge_kernel_name}(",
            *[
                f"    const float* __restrict__ {edge_name},"
                for edge_name in edge_arrays
            ],
            "    const int32_t* __restrict__ src,",
            "    const int32_t* __restrict__ dst,",
            "    float* __restrict__ grad_edge,",
            "    int64_t num_edges) {",
            "    __shared__ float smem[GEN_Y_WARPS * GEN_IN2_DIM];",
            "    const Index e = blockIdx.x;",
            "    const int u = threadIdx.x;",
            "    if (e >= num_edges) return;",
            "    const int s = __ldg(src + e);",
            "    const int t = __ldg(dst + e);",
        ]
        for edge_group, edge_group_specs in enumerate(edge_groups.values()):
            edge_components = {
                int(edge_spec["y_offset"]) + edge_j
                for _, edge_spec in edge_group_specs
                for edge_j in range(
                    int(edge_spec["edge_channels"]) * int(edge_spec["d2"])
                )
            }
            if edge_group == 0:
                # Unused edge irreps still receive zeros, without a clearing kernel.
                edge_components.update(set(range(edge_in2_dim)) - edge_covered)
            edge_components = sorted(edge_components)
            if len(edge_groups) > 1:
                edge_lines.append(f"    if (blockIdx.y == {edge_group}) {{")
            edge_lines += [
                f"    float acc_y{edge_j} = 0.0f;" for edge_j in edge_components
            ]
            for edge_spec_idx, edge_spec, edge_v in (
                (edge_spec_idx, edge_spec, edge_v)
                for edge_spec_idx, edge_spec in edge_group_specs
                for edge_v in range(int(edge_spec["edge_channels"]))
            ):
                edge_x_offset = int(edge_spec["x_offset"])
                edge_in_channels, edge_d1, edge_d3 = (
                    int(edge_spec["in_channels"]),
                    int(edge_spec["d1"]),
                    int(edge_spec["d3"]),
                )
                edge_terms = list(edge_spec["terms"])
                edge_channels = int(edge_spec["edge_channels"])
                edge_channel = "int64_t(u)" if wide_z else "u"
                edge_path = (
                    f"({edge_channel} * {edge_channels} + {edge_v})"
                    if edge_channels != 1
                    else edge_channel
                )
                if edge_channels != 1:
                    edge_lines.append("    {")
                edge_channel_type = (
                    "int64_t" if edge_in_channels > (1 << 31) - y_threads else "int"
                )
                edge_lines.append(
                    f"    for ({edge_channel_type} u = threadIdx.x; u < {edge_in_channels}; u += GEN_Y_THREADS) {{"
                    if edge_in_channels > y_threads
                    else f"    if (u < {edge_in_channels}) {{"
                )
                for edge_i in sorted({int(edge_t[0]) for edge_t in edge_terms}):
                    if paired:
                        for edge_product in range(2):
                            edge_lines.append(
                                f"        const float x{edge_product}v{edge_spec_idx}_{edge_i} = __ldg(x{edge_product} + Index(s) * {edge_in1_dim} + {edge_x_offset} + u * {edge_d1} + {edge_i});"
                            )
                    else:
                        edge_lines.append(
                            f"        const float xv{edge_spec_idx}_{edge_i} = __ldg(x + Index(s) * {edge_in1_dim} + {edge_x_offset} + u * {edge_d1} + {edge_i});"
                        )
                for edge_k in sorted({int(edge_t[2]) for edge_t in edge_terms}):
                    if paired:
                        for edge_product in range(2):
                            edge_lines.append(
                                f"        const float z{edge_product}v{edge_spec_idx}_{edge_k} = __ldg(z{edge_product} + Index(t) * GEN_TOTAL_Z + {z_offsets[edge_spec_idx]} + {edge_path} * {edge_d3} + {edge_k});"
                            )
                    else:
                        edge_lines.append(
                            f"        const float zv{edge_spec_idx}_{edge_k} = __ldg(z + Index(t) * GEN_TOTAL_Z + {z_offsets[edge_spec_idx]} + {edge_path} * {edge_d3} + {edge_k});"
                        )
                for edge_i, edge_j, edge_k, edge_coeff in edge_terms:
                    edge_component = (
                        int(edge_spec["y_offset"])
                        + edge_v * int(edge_spec["d2"])
                        + int(edge_j)
                    )
                    if paired:
                        edge_lines.append(
                            f"""        acc_y{edge_component} = fmaf({float_literal(float(edge_coeff), meta.get("dtype"))}, x0v{edge_spec_idx}_{int(edge_i)} * z0v{edge_spec_idx}_{int(edge_k)} + x1v{edge_spec_idx}_{int(edge_i)} * z1v{edge_spec_idx}_{int(edge_k)}, acc_y{edge_component});"""
                        )
                    else:
                        edge_lines.append(
                            f"""        acc_y{edge_component} = fmaf({float_literal(float(edge_coeff), meta.get("dtype"))} * xv{edge_spec_idx}_{int(edge_i)}, zv{edge_spec_idx}_{int(edge_k)}, acc_y{edge_component});"""
                        )
                edge_lines.append("    }")
                if edge_channels != 1:
                    edge_lines.append("    }")
            for edge_j in edge_components:
                edge_lines += r"""    float sum{j} = acc_y{j};
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum{j} += __shfl_down_sync(0xffffffff, sum{j}, offset);
    if ((threadIdx.x & 31) == 0) smem[(threadIdx.x >> 5) * GEN_IN2_DIM + {j}] = sum{j};""".format(j=edge_j).split("\n")
            edge_lines += "    __syncthreads();\n    if (threadIdx.x < 32) {".split(
                "\n"
            )
            for edge_j in edge_components:
                edge_lines += r"""        float block_sum{j} = (threadIdx.x < GEN_Y_WARPS) ? smem[threadIdx.x * GEN_IN2_DIM + {j}] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) block_sum{j} += __shfl_down_sync(0xffffffff, block_sum{j}, offset);
        if (threadIdx.x == 0) grad_edge[e * GEN_IN2_DIM + {j}] = block_sum{j};""".format(j=edge_j).split("\n")
            edge_lines.append("    }")
            if len(edge_groups) > 1:
                edge_lines += ["    return;", "    }"]
        edge_lines.append("}")
        edge_grid = "static_cast<unsigned int>(num_edges)"
        if len(edge_groups) > 1:
            edge_grid = f"dim3({edge_grid}, {len(edge_groups)})"
        edge_kernels.append((edge_lines, edge_grid))
    (y_kernel_lines, y_grid), (y_pair_kernel_lines, y_pair_grid) = edge_kernels

    mma_include = (
        "\n" + get_fp64_mma(arch)[0] if meta.get("dtype") == torch.float64 else ""
    )
    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>{mma_include}
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_X_THREADS = {x_threads};
constexpr int GEN_Y_THREADS = {y_threads};
constexpr int GEN_Y_WARPS = (GEN_Y_THREADS + 31) / 32;
constexpr int GEN_IN1_DIM = {in1_dim};
constexpr int GEN_IN2_DIM = {in2_dim};
constexpr int GEN_MAX_D2 = {max_d2};
constexpr {z_type} GEN_TOTAL_Z = {total_z};
{path_constant}constexpr {channel_type} GEN_MAX_IN_CHANNELS = {max_in_channels};
constexpr int GEN_MAX_D3 = {max_d3};
constexpr int GEN_Z_TILE_M = 16;
constexpr int GEN_Z_TILE_N = 64;


{make_uvw_conv_node_shared_source}

{x_lines}

{x_pair_lines}

{y_kernel_lines}

{y_pair_kernel_lines}
}}  // namespace

template<int Products>
void prepare_uvwconv_z_fp32(int device) {{
    static thread_local int prepared_device=-1;
    if(prepared_device != device) {{
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,lunatp_uvwconv_bwd_xy_source_zgemm_fp32<Products>));
        prepared_device=device;
    }}
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_z_fp32(
    torch::Tensor weight,
    torch::Tensor grad_out,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(weight.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<1>(weight.get_device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, grad_out.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<1><<<dim3(
        static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
        static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
        GEN_NUM_SPECS), 128, 0, stream>>>(
        weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_z_pair_fp32(
    torch::Tensor weight0, torch::Tensor grad0, torch::Tensor weight1,
    torch::Tensor grad1, int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(weight0.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<2>(weight0.get_device());
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, grad0.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<2><<<dim3(
        ({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N,
        (num_nodes * GEN_MAX_D3 + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M,
        GEN_NUM_SPECS), 128, 0, stream>>>(
        weight0.data_ptr<float>(), grad0.data_ptr<float>(),
        weight1.data_ptr<float>(), grad1.data_ptr<float>(), z.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return z;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_x_from_z_fp32(
    torch::Tensor edge_attr,
    torch::Tensor z,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge_attr.device());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, edge_attr.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
        static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
        edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
        src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}}

torch::Tensor lunatp_uvwconv_bwd_xy_source_y_from_z_fp32(
    torch::Tensor x,
    torch::Tensor z,
    torch::Tensor src,
    torch::Tensor dst) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_edges = src.size(0);
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_apply_fp32<<<{y_grid}, GEN_Y_THREADS, 0, stream>>>(
        x.data_ptr<float>(), z.data_ptr<float>(), src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(),
        grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_edge;
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32(
    torch::Tensor edge0,
    torch::Tensor z0,
    torch::Tensor edge1,
    torch::Tensor z1,
    torch::Tensor x0,
    torch::Tensor x1,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(edge0.device());
    const int64_t num_edges = src.size(0);
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, x0.options());
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x0.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_pair_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
        static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
        edge0.data_ptr<float>(), z0.data_ptr<float>(), edge1.data_ptr<float>(), z1.data_ptr<float>(),
        dst.data_ptr<int32_t>(), src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(),
        grad_x.data_ptr<float>(), num_nodes);
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_pair_apply_fp32<<<{y_pair_grid}, GEN_Y_THREADS, 0, stream>>>(
        x0.data_ptr<float>(), z0.data_ptr<float>(), x1.data_ptr<float>(), z1.data_ptr<float>(),
        src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_x, grad_edge}};
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_dxdy_shared_launch_fp32(
    torch::Tensor x,
    torch::Tensor edge_attr,
    torch::Tensor weight,
    torch::Tensor grad_out,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor src_rowptr,
    torch::Tensor edge_by_src,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    if(num_nodes>0) prepare_uvwconv_z_fp32<1>(x.get_device());
    const int64_t num_edges = src.size(0);
    auto z = torch::empty({{num_nodes, GEN_TOTAL_Z}}, x.options());
    auto grad_x = {grad_x_alloc}({{num_nodes, GEN_IN1_DIM}}, x.options());
    auto grad_edge = torch::empty({{num_edges, GEN_IN2_DIM}}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_zgemm_fp32<1><<<dim3(
        static_cast<unsigned int>(({z_channels} + GEN_Z_TILE_N - 1) / GEN_Z_TILE_N),
        static_cast<unsigned int>((num_nodes * int64_t(GEN_MAX_D3) + GEN_Z_TILE_M - 1) / GEN_Z_TILE_M),
        GEN_NUM_SPECS), 128, 0, stream>>>(
        weight.data_ptr<float>(), grad_out.data_ptr<float>(), nullptr, nullptr, z.data_ptr<float>(), num_nodes);
    if (num_nodes > 0) lunatp_uvwconv_bwd_xy_source_x_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),
        static_cast<unsigned int>((GEN_MAX_IN_CHANNELS + GEN_X_THREADS - 1) / GEN_X_THREADS){x_group_grid}), GEN_X_THREADS, 0, stream>>>(
        edge_attr.data_ptr<float>(), z.data_ptr<float>(), dst.data_ptr<int32_t>(),
        src_rowptr.data_ptr<EdgeIndex>(), edge_by_src.data_ptr<EdgeIndex>(), grad_x.data_ptr<float>(), num_nodes);
    if (num_edges > 0) lunatp_uvwconv_bwd_xy_source_y_apply_fp32<<<{y_grid}, GEN_Y_THREADS, 0, stream>>>(
        x.data_ptr<float>(), z.data_ptr<float>(), src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(),
        grad_edge.data_ptr<float>(), num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_x, grad_edge}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_xy_source_z_pair_fp32", &lunatp_uvwconv_bwd_xy_source_z_pair_fp32);
    m.def("lunatp_uvwconv_bwd_xy_source_z_fp32", &lunatp_uvwconv_bwd_xy_source_z_fp32, "Graph-conv shared first z precontract");
    m.def("lunatp_uvwconv_bwd_xy_source_x_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_x_from_z_fp32, "Graph-conv shared first grad_x from z");
    m.def("lunatp_uvwconv_bwd_xy_source_y_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_y_from_z_fp32, "Graph-conv shared first grad_edge from z");
    m.def("lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32", &lunatp_uvwconv_bwd_xy_source_pair_from_z_fp32, "Graph-conv shared paired grad_x+grad_edge from z");
    m.def("lunatp_uvwconv_bwd_dxdy_shared_launch_fp32", &lunatp_uvwconv_bwd_dxdy_shared_launch_fp32, "Graph-conv shared grad_x+grad_edge source-CSR backward launch");
}}
""".format(
            mma_include=mma_include,
            specs_count=len(specs),
            x_threads=x_threads,
            y_threads=y_threads,
            in1_dim=in1_dim,
            in2_dim=in2_dim,
            max_d2=max_d2,
            z_type=z_type,
            total_z=total_z,
            path_constant=path_constant,
            channel_type=channel_type,
            max_in_channels=max_in_channels,
            max_d3=max_d3,
            make_uvw_conv_node_shared_source=make_uvw_conv_node_shared_source(
                meta, "lunatp_uvwconv_bwd_xy_source_zgemm_fp32", arch=arch
            ),
            x_lines=chr(10).join(x_lines),
            x_pair_lines=chr(10).join(x_pair_lines),
            y_kernel_lines=chr(10).join(y_kernel_lines),
            y_pair_kernel_lines=chr(10).join(y_pair_kernel_lines),
            z_channels=z_channels,
            grad_x_alloc=grad_x_alloc,
            x_group_grid=x_group_grid,
            y_grid=y_grid,
            y_pair_grid=y_pair_grid,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_dw_preprocess_shared_source(
    meta: Mapping[str, object],
) -> str:
    specs = list(meta["generated_specs"])
    in1_dim = int(meta["in1_dim"])
    in2_dim = int(meta["in2_dim"])
    groups, group_rows, spec_to_group = get_uvw_conv_weight_layout(specs)
    threads = 128 if max(int(s["path_channels"]) for s in specs) <= 128 else 256

    params = []
    allocs = []
    ptrs = []
    for group_id, ((_out_offset, _out_channels, d3), _group_specs) in enumerate(groups):
        params.append(f"    float* __restrict__ q_group{group_id},")
        allocs.append(
            f"    q_groups.push_back(torch::empty({{{group_rows[group_id]}, num_nodes * int64_t({d3})}}, q_options));"
        )
        ptrs.append(f"        q_groups[{group_id}].data_ptr<float>(),")
    cases = []
    pair_cases = []
    for spec_idx, spec in enumerate(specs):
        group_id, row_offset = spec_to_group[spec_idx]
        x_offset = int(spec["x_offset"])
        y_offset = int(spec["y_offset"])
        in_channels = int(spec["path_channels"])
        channel_type = (
            "int64_t" if row_offset + in_channels > (1 << 31) - threads else "int"
        )
        edge_channels, d2 = int(spec["edge_channels"]), int(spec["d2"])
        input_channel = f"(u / {edge_channels})" if edge_channels != 1 else "u"
        edge_offset = f"(u % {edge_channels}) * {d2} + " if edge_channels != 1 else ""
        d1 = int(spec["d1"])
        d3 = int(spec["d3"])
        terms = list(spec["terms"])
        unique_i = sorted({int(t[0]) for t in terms})
        unique_j = sorted({int(t[1]) for t in terms})
        lines = [f"case {spec_idx}: {{"]
        lines.append(
            f"    for ({channel_type} u = threadIdx.x; u < {in_channels}; u += blockDim.x) {{"
        )
        for k in range(d3):
            lines.append(f"        float acc{k} = 0.0f;")
        lines.extend(r"""        for (Index p = e_begin; p < e_end; ++p) {{
            const Index e = __ldg(edge_by_dst + p);
            const int s = __ldg(src + p);""".format().split("\n"))
        for i in unique_i:
            lines.append(
                f"            const float xv{i} = __ldg(x + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});"
            )
        for j in unique_j:
            lines.append(
                f"            const float yv{j} = __ldg(edge_attr + e * {in2_dim} + {edge_offset}{y_offset + j});"
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            lines.append(
                f"""            acc{int(k_idx)} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, xv{int(i_idx)} * yv{int(j_idx)}, acc{int(k_idx)});"""
            )
        lines.append("        }")
        for k in range(d3):
            lines.append(
                f"        q_group{group_id}[({row_offset} + u) * (num_nodes * int64_t({d3})) + Index(node) * {d3} + {k}] = acc{k};"
            )
        lines.append("    }")
        lines.append("    break;")
        lines.append("}")
        cases.append("\n".join(lines))

        pair_lines = [f"case {spec_idx}: {{"]
        pair_lines.append(
            f"    for ({channel_type} u = threadIdx.x; u < {in_channels}; u += blockDim.x) {{"
        )
        for k in range(d3):
            pair_lines.append(f"        float acc{k} = 0.0f;")
        pair_lines.extend(r"""        for (Index p = e_begin; p < e_end; ++p) {{
            const Index e = __ldg(edge_by_dst + p);
            const int s = __ldg(src + p);""".format().split("\n"))
        for i in unique_i:
            pair_lines.extend(
                f"            const float x0v{i} = __ldg(x0 + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});\n            const float x1v{i} = __ldg(x1 + Index(s) * {in1_dim} + {x_offset} + {input_channel} * {d1} + {i});".split(
                    "\n"
                )
            )
        for j in unique_j:
            pair_lines.extend(
                f"            const float y0v{j} = __ldg(edge0 + e * {in2_dim} + {edge_offset}{y_offset + j});\n            const float y1v{j} = __ldg(edge1 + e * {in2_dim} + {edge_offset}{y_offset + j});".split(
                    "\n"
                )
            )
        for i_idx, j_idx, k_idx, coeff in terms:
            pair_lines.append(
                f"""            acc{int(k_idx)} = fmaf({float_literal(float(coeff), meta.get("dtype"))}, x0v{int(i_idx)} * y0v{int(j_idx)} + x1v{int(i_idx)} * y1v{int(j_idx)}, acc{int(k_idx)});"""
            )
        pair_lines.append("        }")
        for k in range(d3):
            pair_lines.append(
                f"        q_group{group_id}[({row_offset} + u) * (num_nodes * int64_t({d3})) + Index(node) * {d3} + {k}] = acc{k};"
            )
        pair_lines.append("    }")
        pair_lines.append("    break;")
        pair_lines.append("}")
        pair_cases.append("\n".join(pair_lines))

    return cuda_source_for_dtype(
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {{
constexpr int GEN_NUM_SPECS = {specs_count};
constexpr int GEN_THREADS = {threads};
constexpr int GEN_NUM_GROUPS = {groups_count};


__global__ __launch_bounds__(GEN_THREADS, 4)
void lunatp_uvwconv_bwd_weight_grouped_q_fp32(
    const float* __restrict__ x,
    const float* __restrict__ edge_attr,
    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ edge_by_dst,
    const EdgeIndex* __restrict__ rowptr,
{params}
    int64_t num_nodes) {{
    const Index node = blockIdx.x;
    const int spec = blockIdx.y;
    if (node >= num_nodes) return;
    const Index e_begin = __ldg(rowptr + node);
    const Index e_end = __ldg(rowptr + node + 1);
    switch (spec) {{
{cases}
    default:
        break;
    }}
}}

__global__ __launch_bounds__(GEN_THREADS, 4)
void lunatp_uvwconv_bwd_weight_grouped_q_pair_fp32(
    const float* __restrict__ x0,
    const float* __restrict__ edge0,
    const float* __restrict__ x1,
    const float* __restrict__ edge1,
    const int32_t* __restrict__ src,
    const EdgeIndex* __restrict__ edge_by_dst,
    const EdgeIndex* __restrict__ rowptr,
{params}
    int64_t num_nodes) {{
    const Index node = blockIdx.x;
    const int spec = blockIdx.y;
    if (node >= num_nodes) return;
    const Index e_begin = __ldg(rowptr + node);
    const Index e_end = __ldg(rowptr + node + 1);
    switch (spec) {{
{pair_cases}
    default:
        break;
    }}
}}

}} // namespace

std::vector<torch::Tensor> lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32(
    torch::Tensor x,
    torch::Tensor edge_attr,
    torch::Tensor src,
    torch::Tensor edge_by_dst,
    torch::Tensor rowptr,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    auto q_options = x.options();
    std::vector<torch::Tensor> q_groups;
{allocs}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_weight_grouped_q_fp32<<<dim3(static_cast<unsigned int>(num_nodes), GEN_NUM_SPECS), GEN_THREADS, 0, stream>>>(
        x.data_ptr<float>(),
        edge_attr.data_ptr<float>(),
        src.data_ptr<int32_t>(),
        edge_by_dst.data_ptr<EdgeIndex>(),
        rowptr.data_ptr<EdgeIndex>(),
{ptrs}
        num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}

std::vector<torch::Tensor> lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32(
    torch::Tensor x0,
    torch::Tensor edge0,
    torch::Tensor x1,
    torch::Tensor edge1,
    torch::Tensor src,
    torch::Tensor edge_by_dst,
    torch::Tensor rowptr,
    int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x0.device());
    auto q_options = x0.options();
    std::vector<torch::Tensor> q_groups;
{allocs}
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_nodes > 0) lunatp_uvwconv_bwd_weight_grouped_q_pair_fp32<<<dim3(static_cast<unsigned int>(num_nodes), GEN_NUM_SPECS), GEN_THREADS, 0, stream>>>(
        x0.data_ptr<float>(),
        edge0.data_ptr<float>(),
        x1.data_ptr<float>(),
        edge1.data_ptr<float>(),
        src.data_ptr<int32_t>(),
        edge_by_dst.data_ptr<EdgeIndex>(),
        rowptr.data_ptr<EdgeIndex>(),
{ptrs}
        num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_groups;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32", &lunatp_uvwconv_bwd_weight_grouped_q_launch_fp32, "Graph-conv shared grouped Q precontraction");
    m.def("lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32", &lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_fp32, "Graph-conv shared grouped pair-summed Q precontraction");
}}
""".format(
            specs_count=len(specs),
            threads=threads,
            groups_count=len(groups),
            params=chr(10).join(params),
            cases=chr(10).join(cases),
            pair_cases=chr(10).join(pair_cases),
            allocs=chr(10).join(allocs),
            ptrs=chr(10).join(ptrs),
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_dw_reduce_shared_source(meta, *, arch):
    groups, group_rows, _ = get_uvw_conv_weight_layout(meta["generated_specs"])
    fp64 = meta.get("dtype") == torch.float64
    mma = (
        make_uvw_conv_gemm_mma(
            meta,
            "a[p][row*32+(input^((row&3)<<2))]",
            "b[p][input*64+(column^((input&3)<<2))]",
            arch=arch,
        )
        if fp64
        else {
            "declaration": r"""using Mma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<2,4,1>,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
    Mma::FragmentC acc;acc.clear();""",
            "compute": r"""#pragma unroll 4
        for(int k=0;k<32;++k) {
            #pragma unroll
            for(int p=0;p<Products;++p) {
                Mma::FragmentA av;av[0]=a[p][local_row*32+(k^(local_row&7))];av[1]=a[p][(local_row+1)*32+(k^((local_row+1)&7))];
                Mma::FragmentB bv;
                #pragma unroll
                for(int c=0;c<4;++c) bv[c]=b[p][k*64+local_column+c*16];
                Mma()(acc,av,bv,acc);
            }
        }""",
            "row": "row_base+local_row+r",
            "column": "column_base+local_column+c*16",
            "value": "acc[r*4+c]",
            "rows": 2,
            "columns": 4,
        }
    )
    a_index = (
        "(i/32)*32+((i%32)^(((i/32)&3)<<2))"
        if fp64
        else "(i/32)*32+((i%32)^((i/32)&7))"
    )
    b_index = "(i/64)*64+((i%64)^(((i/64)&3)<<2))" if fp64 else "i"
    work_index = "Index" if fp64 else "int"
    grad_position = (
        "Index k=base+i/64; int column=column_base+i%64;"
        if fp64
        else "int k=base+i/64,column=column_base+i%64;"
    )
    row_type = "int64_t" if max(group_rows) > (1 << 31) - 16 else "Index"
    row_base = "int64_t(blockIdx.y)*16" if row_type == "int64_t" else "blockIdx.y*16"
    q_position = "Index row=row_base+i/32,k=base+i%32;"
    if row_type == "int64_t":
        q_position = "int64_t row=row_base+i/32; Index k=base+i%32;"
    cases = []
    for group, ((offset, channels, d3), specs) in enumerate(groups):
        dimensions = f"constexpr int Rows={group_rows[group]},Columns={channels};"
        if group_rows[group] > (1 << 31) - 16:
            dimensions = f"constexpr int64_t Rows={group_rows[group]}; constexpr int Columns={channels};"
        stores = []
        for spec in specs:
            begin = spec["row_offset"]
            end = begin + spec["path_channels"]
            stores.append(
                f"""if(row>={begin} && row<{end}) atomicAdd(out+{spec["weight_offset"]}+int64_t(row-{begin})*{channels}+column,{mma["value"]});"""
            )
        cases.append(
            r"""
case {group}: {{
    {dimensions}
    const Index total_k=nodes*{d3};
    if(row_base>=Rows || column_base>=Columns) return;
    const Index tiles=(total_k+31)/32,per_split=(tiles+7)/8;
    const Index begin=split*per_split*32,end=min(total_k,begin+per_split*32);
    if(begin>=end) return;
    {declaration}
    for({work_index} base=begin;base<end;base+=32) {{
        #pragma unroll
        for(int p=0;p<Products;++p) {{
            for(int i=threadIdx.x;i<512;i+=128) {{
                {q_position}
                a[p][{a_index}]=(row<Rows && k<end)?__ldg(inputs.q[p][{group}]+int64_t(row)*total_k+k):0.f;
            }}
            for(int i=threadIdx.x;i<2048;i+=128) {{
                {grad_position}
                b[p][{b_index}]=(k<end && column<Columns)?__ldg(inputs.g[p]+int64_t(k/{d3})*{out_dim}+{offset}+column*{d3}+k%{d3}):0.f;
            }}
        }}
        __syncthreads();
        {compute}
        __syncthreads();
    }}
    #pragma unroll
    for(int r=0;r<{rows};++r) {{
        {row_type} row={row};
        #pragma unroll
        for(int c=0;c<{columns};++c) {{
            int column={column};
            if(row<Rows && column<Columns) {{
                {stores}
            }}
        }}
    }}
    return;
}}
""".format(
                group=group,
                dimensions=dimensions,
                d3=d3,
                declaration=mma["declaration"],
                work_index=work_index,
                q_position=q_position,
                a_index=a_index,
                grad_position=grad_position,
                b_index=b_index,
                out_dim=meta["out_dim"],
                offset=offset,
                compute=mma["compute"],
                rows=mma["rows"],
                row_type=row_type,
                row=mma["row"],
                columns=mma["columns"],
                column=mma["column"],
                stores=chr(10).join(stores),
            )
        )
    mma_include = (
        "\n" + get_fp64_mma(arch)[0] if meta.get("dtype") == torch.float64 else ""
    )
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>{mma_include}
struct Inputs {{const float* q[2][{groups_count}];const float* g[2];}};
template<int Products>
__global__ __launch_bounds__(128,4)
void lunatp_uvwconv_bwd_weight_shared_fp32(Inputs inputs,float* out,Index nodes) {{
    __shared__ float a[Products][512];
    __shared__ float b[Products][2048];
    const {row_type} row_base={row_base}; const int column_base=blockIdx.x*64;
    const int local_row=(threadIdx.x/16)*2,local_column=threadIdx.x%16;
    const int split=blockIdx.z%8;
    switch(blockIdx.z/8) {{
{cases}
    }}
}}
template<int Products>
at::Tensor reduce(Inputs inputs,at::Tensor grad) {{
    c10::cuda::CUDAGuard guard(grad.device());
    auto out=at::zeros({{{weight_numel}}},grad.options());
    if(grad.size(0)==0) return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    lunatp_uvwconv_bwd_weight_shared_fp32<Products><<<dim3({column_tiles},{row_tiles},{reduction_tasks}),128,0,stream>>>(inputs,out.data_ptr<float>(),grad.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}}
at::Tensor launch(std::vector<at::Tensor> q,at::Tensor g) {{
    Inputs inputs{{}};
    for(int i=0;i<{groups_count};++i) inputs.q[0][i]=q[i].data_ptr<float>();
    inputs.g[0]=g.data_ptr<float>();
    return reduce<1>(inputs,g);
}}
at::Tensor launch_pair(std::vector<at::Tensor> q0,std::vector<at::Tensor> q1,at::Tensor g0,at::Tensor g1) {{
    Inputs inputs{{}};
    for(int i=0;i<{groups_count};++i) {{inputs.q[0][i]=q0[i].data_ptr<float>();inputs.q[1][i]=q1[i].data_ptr<float>();}}
    inputs.g[0]=g0.data_ptr<float>();inputs.g[1]=g1.data_ptr<float>();
    return reduce<2>(inputs,g0);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);m.def("launch_pair",&launch_pair);}}
""".format(
            mma_include=mma_include,
            groups_count=len(groups),
            row_type=row_type,
            row_base=row_base,
            cases=chr(10).join(cases),
            weight_numel=meta["weight_numel"],
            column_tiles=(max((key[1] for key, _ in groups)) + 63) // 64,
            row_tiles=(max(group_rows) + 15) // 16,
            reduction_tasks=len(groups) * 8,
        ),
        meta.get("dtype", torch.float32),
    )


def make_uvw_conv_backward_external_kernel(meta, program, inputs, outputs, kernel_name):
    """Emit a path-major kernel; eight lanes cooperate on each path channel."""
    specs = meta["generated_specs"]
    x_outputs = {k: v for k, v in program["x"].items() if k in outputs}
    y_outputs = {k: v for k, v in program["y"].items() if k in outputs}
    w_outputs = {k: v for k, v in program["w"].items() if k in outputs}
    e_outputs = {k: v for k, v in program.get("e", {}).items() if k in outputs}
    needed_z = {
        pair[1]
        for pairs in (*x_outputs.values(), *y_outputs.values())
        for pair in pairs
    }
    needed_z.update(z for pairs in e_outputs.values() for _, z in pairs)
    contractions = {k: v for k, v in program["z"].items() if k in needed_z}
    forward_pairs = program["forward"] if "gg" in outputs else ()
    needed_q = {q for pairs in w_outputs.values() for q, _ in pairs}
    needed_q.update(q for q, _ in forward_pairs)
    products = {k: v for k, v in program["cg"].items() if k in needed_q}
    weights = sorted(
        {w for pairs in contractions.values() for w, _ in pairs}
        | {w for _, w in forward_pairs}
    )
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
        path_type = "int64_t" if spec["path_channels"] > (1 << 31) - 16 else "int"
        edge_size = edges * d2
        # Large third-order contractions otherwise keep Z, all CG epilogue
        # inputs and both edge accumulators live together. Shared memory bounds
        # that register lifetime without materializing a global intermediate.
        stage_z = len(contractions) * d3 > 24
        lines = [f"case {index}: {{"]
        lines += [f"float sums_{target}[{edge_size}]={{0.f}};" for target in y_outputs]
        lines += [f"float scalar_{target}=0.f;" for target in e_outputs]
        if forward_pairs:
            lines += r"""using FMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1,{d3},1>,
float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
float,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;
FMma::FragmentC forward_sum[{channels}];
#pragma unroll
for(int c=0;c<{channels};++c) forward_sum[c].clear();""".format(
                d3=d3, channels=(channels + 7) // 8
            ).split("\n")
        lines += r"""#pragma unroll 1
for({path_type} p=path_group;p<{path_channels};p+=16) {{
int u=p/{edges},v=p%{edges};
const unsigned mask={mask};""".format(
            path_type=path_type,
            path_channels=spec["path_channels"],
            edges=edges,
            # Four eight-lane path groups fill a warp. Tail groups retain
            # their active mask; complete warps use a compile-time mask.
            mask="0xffffffffu" if spec["path_channels"] % 4 == 0 else "__activemask()",
        ).split("\n")
        if stage_z:
            lines.append("{")
        lines += f"using ZMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1,{d3},1>,\nfloat,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,\nfloat,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;".split(
            "\n"
        )
        for name in contractions:
            lines += [f"ZMma::FragmentC contract_{name}; contract_{name}.clear();"]
        if products:
            lines += f"using WMma=cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<1,1,{d3}>,\nfloat,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,\nfloat,cutlass::layout::RowMajor,cutlass::arch::OpMultiplyAdd>;".split(
                "\n"
            )
        for name, pairs in products.items():
            lines += [f"WMma::FragmentA product_{name};", "{"]
            lines += [f"float z{k}=0.f;" for k in range(d3)]
            lines.append("if(lane==0) {")
            # Reuse the CG accumulator while releasing each pair of inputs.
            for pair in pairs:
                lines += [
                    "{",
                    emit_cg_source(meta, spec, pairs=(pair,), x_row="source_node"),
                    "}",
                ]
            lines.append("}")
            lines += [
                f"product_{name}[{k}]=__shfl_sync(mask,z{k},0,8);" for k in range(d3)
            ]
            lines.append("}")
        lines += [
            "#pragma unroll" if forward_pairs else "#pragma unroll 1",
            f"for(int c=lane;c<{channels};c+=8) {{",
        ]
        for name in weights:
            lines += f"ZMma::FragmentA weight_{name};\nweight_{name}[0]=__ldg({name}+int64_t(item)*{wdim}+{wo}+p*{channels}+c);".split(
                "\n"
            )
        for name in gradients:
            lines += [f"ZMma::FragmentB grad_{name};"]
            lines += [
                f"grad_{name}[{k}]=__ldg({name}+int64_t(dest_node)*{odim}+{oo}+c*{d3}+{k});"
                for k in range(d3)
            ]
        for target, pairs in contractions.items():
            for w, g in pairs:
                lines += [
                    f"ZMma()(contract_{target},weight_{w},grad_{g},contract_{target});"
                ]
        for target, pairs in w_outputs.items():
            lines += [f"WMma::FragmentC delta_{target}; delta_{target}.clear();"]
            for q, g in pairs:
                lines += [
                    f"WMma()(delta_{target},product_{q},grad_{g},delta_{target});"
                ]
            lines += [
                f"{target}[int64_t(item)*{wdim}+{wo}+p*{channels}+c]=delta_{target}[0];"
            ]
        for q, w in forward_pairs:
            lines.append(
                f"FMma()(forward_sum[c/8],weight_{w},product_{q},forward_sum[c/8]);"
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
        with_env = bool(program.get("e"))
        if x_outputs or (y_outputs and not with_env):
            lines.append("if(lane==0) {")
            # A single contraction can feed X and Y together, overlapping
            # their loads and arithmetic. Higher orders release one output's
            # inputs before starting the next to bound register lifetime.
            roles = [(target, pairs, "x") for target, pairs in x_outputs.items()]
            if not with_env:
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
                    for name in sorted({pair[0] for pair in pairs}):
                        for component in sorted(
                            {t[source_axis] for t in spec["terms"]}
                        ):
                            lines += [
                                f"""float value_{name}_{component}=__ldg({name}+int64_t({("item" if axis == "x" else "source_node")})*{dim}+{offset}+{channel}*{width}+{component});"""
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
                        for pair in pairs:
                            name, z = pair[:2]
                            scalar = pair[2] if len(pair) == 3 else None
                            factor = (
                                f"*__ldg({scalar}+item)" if scalar is not None else ""
                            )
                            lines += [
                                f"""{prefix}{component}=fmaf({float_literal(coeff, meta.get("dtype"))}*value_{name}_{source}{factor},{z_values[z, k]},{prefix}{component});"""
                            ]
                for target, pairs, axis in group:
                    prefix = target if len(group) > 1 else "acc"
                    for component in sorted(
                        {t[0 if axis == "x" else 1] for t in spec["terms"]}
                    ):
                        if axis == "x":
                            lines += [
                                f"atomicAdd({target}+int64_t(source_node)*{xdim}+{xo}+u*{d1}+{component},{prefix}{component});"
                            ]
                        else:
                            lines += [
                                f"sums_{target}[v*{d2}+{component}]+={prefix}{component};"
                            ]
                lines.append("}")
            lines.append("}")
        if with_env and (y_outputs or e_outputs):
            edge_scalar_products = {
                edge_target: tuple(
                    (edge_x, edge_y, edge_z)
                    for edge_q, edge_z in edge_products
                    for edge_x, edge_y, edge__ in program["cg"][edge_q]
                )
                for edge_target, edge_products in e_outputs.items()
            }
            edge_families = tuple(
                dict.fromkeys(
                    [
                        (edge_x, edge_z)
                        for edge_products in y_outputs.values()
                        for edge_x, edge_z, edge__ in edge_products
                    ]
                    + [
                        (edge_x, edge_z)
                        for edge_products in edge_scalar_products.values()
                        for edge_x, _, edge_z in edge_products
                    ]
                )
            )
            edge_components = sorted({int(edge_term[1]) for edge_term in spec["terms"]})
            edge_lines = ["if(lane==0) {"]
            for edge_x, edge_z in edge_families:
                # A family stays live only until both of its consumers have used it.
                # This avoids repeating CG or retaining all families' accumulators.
                edge_lines.append("{")
                for edge_i in sorted(
                    {int(edge_term[0]) for edge_term in spec["terms"]}
                ):
                    edge_lines.append(
                        f"""float value_x{edge_i}=__ldg({edge_x}+int64_t(source_node)*{meta["in1_dim"]}+{spec["x_offset"]}+u*{spec["d1"]}+{edge_i});"""
                    )
                edge_lines += [
                    f"float value_y{edge_j}=0.f;" for edge_j in edge_components
                ]
                for edge_i, edge_j, edge_k, edge_coefficient in spec["terms"]:
                    edge_lines.append(
                        f"""value_y{edge_j}=fmaf({float_literal(edge_coefficient, meta.get("dtype"))}*value_x{edge_i},{z_values[edge_z, edge_k]},value_y{edge_j});"""
                    )
                for edge_target, edge_products in y_outputs.items():
                    for edge_feature, edge_matrix, edge_scalar in edge_products:
                        if (edge_feature, edge_matrix) == (edge_x, edge_z):
                            edge_scale = (
                                f"*__ldg({edge_scalar}+item)"
                                if edge_scalar is not None
                                else ""
                            )
                            edge_lines += [
                                f"""sums_{edge_target}[v*{spec["d2"]}+{edge_j}]+=value_y{edge_j}{edge_scale};"""
                                for edge_j in edge_components
                            ]
                for edge_target, edge_products in edge_scalar_products.items():
                    for edge_feature, edge_edge, edge_matrix in edge_products:
                        if (edge_feature, edge_matrix) == (edge_x, edge_z):
                            for edge_j in edge_components:
                                edge_value = f"""__ldg({edge_edge}+int64_t(item)*{meta["in2_dim"]}+{spec["y_offset"]}+v*{spec["d2"]}+{edge_j})"""
                                edge_lines.append(
                                    f"scalar_{edge_target}=fmaf(value_y{edge_j},{edge_value},scalar_{edge_target});"
                                )
                edge_lines.append("}")
            edge_lines.append("}")
            lines += edge_lines
        lines.append("}")
        if e_outputs:
            lines.append("if(lane==0) {")
            for target_index, target in enumerate(e_outputs):
                lines.append(
                    f"scalar_partial[{target_index * 16}+path_group]=scalar_{target};"
                )
            lines += "}\n__syncthreads();".split("\n")
            for target_index, target in enumerate(e_outputs):
                lines += r"""if(threadIdx.x=={target_thread}) {{
float value=0.f;
#pragma unroll
for(int group=0;group<16;++group) value+=scalar_partial[{partial_base}+group];
atomicAdd({target}+item,value);
}}""".format(
                    target_thread=target_index,
                    partial_base=target_index * 16,
                    target=target,
                ).split("\n")
        if forward_pairs:
            # Each lane owns output columns; reduce the sixteen path groups
            # once per CTA rather than atomically updating for every channel.
            lines += [
                "#pragma unroll",
                f"for(int c=lane;c<{channels};c+=8) {{",
                *[
                    f"forward_partial[(path_group*{channels}+c)*{d3}+{k}]=forward_sum[c/8][{k}];"
                    for k in range(d3)
                ],
                "}",
                "__syncthreads();",
                f"for(int j=threadIdx.x;j<{channels * d3};j+=128) {{",
                "float value=0.f;",
                "#pragma unroll",
                f"for(int group=0;group<16;++group) value+=forward_partial[group*{channels * d3}+j];",
                f"atomicAdd(gg+int64_t(dest_node)*{odim}+{oo}+j,value);",
                "}",
            ]
        if y_outputs:
            lines += [
                "if(lane==0) {",
                "#pragma unroll",
                f"for(int j=0;j<{edge_size};++j) {{",
            ]
            for target_index, target in enumerate(y_outputs):
                lines += [
                    f"edge_partial[{target_index * 16 * edge_size}+path_group*{edge_size}+j]=sums_{target}[j];"
                ]
            lines += r"""}}
}}
__syncthreads();
for(int j=threadIdx.x;j<{size};j+=128) {{""".format(size=edge_size).split("\n")
            for target_index, target in enumerate(y_outputs):
                lines += r"""{{
float value=0.f;
#pragma unroll
for(int group=0;group<16;++group) value+=edge_partial[{size_target_index}+group*{size}+j];
atomicAdd({target}+int64_t(item)*{ydim}+{yo}+j,value);
}}""".format(
                    size_target_index=target_index * 16 * edge_size,
                    size=edge_size,
                    target=target,
                    ydim=ydim,
                    yo=yo,
                ).split("\n")
            lines.append("}")
        lines += ["return;", "}"]
        cases.append("\n".join(lines))
    parameters = [f"const float* __restrict__ {name}" for name in inputs]
    parameters += ["const int32_t* graph_src", "const int32_t* graph_dst"]
    parameters += [f"float* __restrict__ {name}" for name in outputs]
    storage = []
    if forward_pairs:
        forward_size = max(s["out_channels"] * s["d3"] for s in specs)
        storage.append(f"__shared__ float forward_partial[{16 * forward_size}];")
    if y_outputs:
        storage.append(
            f"__shared__ float edge_partial[{16 * len(y_outputs) * max_edge}];"
        )
    if e_outputs:
        storage.append(f"__shared__ float scalar_partial[{16 * len(e_outputs)}];")
    if len(contractions) * max_d3 > 24:
        storage.append(
            f"__shared__ volatile float reduced_z[{16 * len(contractions) * max_d3}];"
        )
    # Large directional contractions need a larger per-thread register budget
    # to avoid spilling the Z fragments into local memory.
    resident_blocks = 2 if len(contractions) * max_d3 > 48 else 4
    return cuda_source_for_dtype(
        r"""
__global__ __launch_bounds__(128,{resident_blocks})
void {kernel_name}({parameters}) {{
    int lane=threadIdx.x&7,path_group=threadIdx.x>>3;
    Index item=blockIdx.x;
    const int source_node=__ldg(graph_src+item),dest_node=__ldg(graph_dst+item);
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
        ),
        meta.get("dtype", torch.float32),
    )


def make_external_backward_source(meta, program, inputs, outputs, phase, *, env=False):
    """Bind one derivative program and emit its per-edge contractions."""
    prefix = f"lunatp_uvwconv_{phase}_external"
    # Fuse the output derivative when its per-lane fragments fit the register budget.
    # Wider contractions use a separate forward emitter.
    forward_budget = max(
        ((s["out_channels"] + 7) // 8) * s["d3"] for s in meta["generated_specs"]
    )
    fuse_forward = phase == "double_bwd" and "gg" in outputs and forward_budget <= 40
    matrix_outputs = tuple(name for name in outputs if name != "gg" or fuse_forward)
    max_d3 = max(s["d3"] for s in meta["generated_specs"])
    fragment_count = len(program["z"]) + len(program["cg"])
    if env:
        # Envelope-only CG terms are consumed after the matrix loop; they do
        # not keep WMma fragments live. Count only the actual fused consumers.
        needed_q = {
            q for target in matrix_outputs for q, _ in program["w"].get(target, ())
        }
        if fuse_forward:
            needed_q.update(q for q, _ in program["forward"])
        needed_z = {
            term[1]
            for axis in ("x", "y", "e")
            for target in matrix_outputs
            for term in program[axis].get(target, ())
        }
        fragment_count = len(needed_z) + len(needed_q)
    # Separate XY and weight work when their combined fragments would crowd
    # registers. Both phases use this same emitter and write final outputs.
    if fragment_count * max_d3 > 64:
        phases = (
            tuple(k for k in matrix_outputs if k not in program["w"]),
            tuple(k for k in matrix_outputs if k in program["w"]),
        )
    else:
        phases = (matrix_outputs,)
    kernels, branches = [], []
    for index, phase_outputs in enumerate(phases):
        if not phase_outputs:
            continue
        name = f"{prefix}_fused_{index}_fp32"
        kernels.append(
            make_uvw_conv_backward_external_kernel(
                meta, program, inputs, phase_outputs, name
            )
        )
        args = ",".join(f"{name}.data_ptr<float>()" for name in inputs)
        args += ",src.data_ptr<int32_t>(),dst.data_ptr<int32_t>()," + ",".join(
            f"{name}.data_ptr<float>()" for name in phase_outputs
        )
        branches.append([
            f"""{name}<<<dim3(items,{len(meta["generated_specs"])}),128,0,stream>>>({args});"""
        ])
    if "gg" in outputs and not fuse_forward:
        products = tuple((program["cg"][q], w) for q, w in program["forward"])
        name = f"{prefix}_g_fp32"
        kernel, grid, clear, pointer_order = make_uvw_conv_forward_external_kernel(
            meta, products, name
        )
        kernels.insert(0, kernel)
        output_branch = []
        if clear:
            output_branch.append(
                "C10_CUDA_CHECK(cudaMemsetAsync(gg.data_ptr<float>(),0,gg.numel()*sizeof(float),stream));"
            )
        args = ",".join(f"{name}.data_ptr<float>()" for name in pointer_order)
        output_branch.append(
            f"{name}<<<{grid},128,0,stream>>>({args},src.data_ptr<int32_t>(),dst.data_ptr<int32_t>(),gg.data_ptr<float>(),items);"
        )
        branches.append(output_branch)
    launches = [line for branch in branches for line in branch]
    stream_state, state_parameter, stream_binding = "", "", ""
    # Each branch writes distinct outputs. Fork after zeroing the accumulators,
    # and join before returning them to the caller's stream.
    if len(branches) > 1:
        count = len(branches) - 1
        stream_state = r"""
struct BranchStreams {
    BranchStreams(int device, const std::vector<int64_t>& handles) {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        for (auto handle : handles)
            streams.push_back(at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handle), device));
        for (auto& event : done)
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~BranchStreams() noexcept {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        const int device = streams.front().device_index();
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        cudaEventDestroy(start);
        for (auto event : done) cudaEventDestroy(event);
        if (previous != device) cudaSetDevice(previous);
    }
    BranchStreams(const BranchStreams&) = delete;
    BranchStreams& operator=(const BranchStreams&) = delete;
    std::vector<at::cuda::CUDAStream> streams;
    cudaEvent_t start = nullptr;
    cudaEvent_t done[COUNT] = {};
};
""".replace("COUNT", str(count))
        state_parameter = "BranchStreams& state,"
        stream_binding = f"""
pybind11::class_<BranchStreams>(m,"_Streams",pybind11::module_local())
    .def(pybind11::init<int,const std::vector<int64_t>&>());
m.attr("_Streams").attr("stream_count")={count};
m.attr("_Streams").attr("stream_priority")=0;
"""
        launches = ["C10_CUDA_CHECK(cudaEventRecord(state.start, stream));"]
        for index, branch in enumerate(branches[1:]):
            launches += [f"C10_CUDA_CHECK(cudaStreamWaitEvent(state.streams[{index}], state.start, 0));"]
            launches += [line.replace(",0,stream", f",0,state.streams[{index}]")
                         .replace(",stream));", f",state.streams[{index}]));") for line in branch]
            launches += [f"C10_CUDA_CHECK(cudaEventRecord(state.done[{index}], state.streams[{index}]));"]
        launches += branches[0]
        launches += [f"C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done[{index}], 0));" for index in range(count)]
    output_shapes = ENV_OUTPUT_SHAPES if env else OUTPUT_SHAPES
    allocations = [
        f"auto {name}=at::empty_like({output_shapes[name]});" for name in outputs
    ]
    clears = [
        f"C10_CUDA_CHECK(cudaMemsetAsync({name}.data_ptr<float>(),0,{name}.numel()*sizeof(float),stream));"
        for name in outputs
        if name in (*program["x"], *program["y"], *program.get("e", {}))
        or (name == "gg" and fuse_forward)
    ]
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/thread/mma.h>
#include <vector>
"""
        + "\n".join(kernels) + stream_state
        + r"""
std::vector<at::Tensor> launch({state_parameter}{input_parameters},at::Tensor src,at::Tensor dst) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t items=y.size(0);
    {allocations}
    if(items==0) {{
        {clear_outputs}
        return {{{outputs}}};
    }}
    auto stream=at::cuda::getCurrentCUDAStream();
    {clears}
    {launches}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{outputs}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{{stream_binding}m.def("launch",&launch);}}
""".format(
            state_parameter=state_parameter,
            stream_binding=stream_binding,
            input_parameters=",".join((f"at::Tensor {name}" for name in inputs)),
            allocations=chr(10).join(allocations),
            clear_outputs=chr(10).join((f"{name}.zero_();" for name in outputs)),
            outputs=",".join(outputs),
            clears=chr(10).join(clears),
            launches=chr(10).join(launches),
        ),
        meta.get("dtype", torch.float32),
    )


def run_external(op, order, inputs, src, dst, *, needs=None, env=False):
    """Native graph gather/scatter with common first-, second- and third-order expressions."""
    outputs = (ENV_PROGRAMS if env else PROGRAMS)[order]["outputs"]
    selected = (
        outputs
        if needs is None
        else tuple(name for name, need in zip(outputs, needs) if need)
    )

    key = (order, selected)
    if env:
        key += (True,)
    launch = op.external_kernels.get(key)
    if launch is None:
        meta = op.meta
        program = (ENV_PROGRAMS if env else PROGRAMS)[order]
        roles = (ENV_INPUTS if env else INPUTS)[
            : (5, 9, 14)[order - 1] if env else (4, 7, 11)[order - 1]
        ]
        source = make_external_backward_source(
            meta,
            program,
            roles,
            selected,
            ("bwd", "double_bwd", "triple_bwd")[order - 1],
            env=env,
        )
        extension = load_cuda_extension(
            f"""lunatp_uvwconv_{("fwd", "bwd", "double_bwd", "triple_bwd")[order]}_external_{op.dtype_tag}""",
            source,
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )
        launch = (extension.launch, getattr(extension, "_Streams", None))
        op.external_kernels[key] = launch

    function, stream_factory = launch
    if stream_factory is None:
        result = function(*inputs, src, dst)
    else:
        state = get_overlap_state(op, inputs[0].device, stream_factory)
        result = function(state, *inputs, src, dst)
    if needs is None:
        return tuple(result)
    values = iter(result)
    return tuple(next(values) if needed else None for needed in needs)


def run(op, x, edge_attr, weight, grad_out, *, graph, needs=None):
    set_arch(op, x.get_device())
    src, dst = graph.src, graph.dst
    num_nodes = graph.num_nodes
    if op.weight_mode == "shared":
        # Frozen edge attributes and weights still require propagation into
        # upstream node features. Reuse the source-CSR input contraction.
        if needs is not None and needs[0] and not any(needs[1:]):
            if op.input_gradient is None:
                extension = load_cuda_extension(
                    f"lunatp_uvwconv_bwd_x_shared_{op.dtype_tag}",
                    make_uvw_conv_backward_dx_shared_source(op.meta, arch=op.arch),
                    arch=op.arch,
                    use_fast_math=False,
                    index_bits=op.index_bits,
                )
                op.input_gradient = getattr(
                    extension,
                    f"lunatp_uvwconv_bwd_dx_shared_launch_{op.dtype_tag}",
                )
            grad_x = op.input_gradient(
                edge_attr,
                weight,
                grad_out,
                graph.dst,
                graph.src_rowptr,
                graph.edge_ids_by_src,
                num_nodes,
            )
            return grad_x, None, None
        if op.node_contraction is None:
            load_node_contraction(op)
        if op.weight_contraction is None:
            load_weight_contraction(op)

        current_stream = torch.cuda.current_stream(x.device)
        weight_stream = auxiliary_streams(
            current_stream.device_index, 1, scope=current_stream.cuda_stream
        )[0]
        # Weight reduction is independent of x/edge gradients. Both streams
        # first see caller inputs, and the caller joins before returning.
        weight_stream.wait_stream(current_stream)
        with torch.cuda.stream(weight_stream):
            q = op.precontract_q(
                x,
                edge_attr,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            grad_weight = op.weight_reduce.launch(q, grad_out)
            del q
        grad_x, grad_edge = op.launch_dxdy(
            x,
            edge_attr,
            weight,
            grad_out,
            graph.src,
            graph.dst,
            graph.src_rowptr,
            graph.edge_ids_by_src,
            num_nodes,
        )
        current_stream.wait_stream(weight_stream)
        grad_weight.record_stream(current_stream)
        return grad_x, grad_edge, grad_weight
    return tuple(
        run_external(op, 1, (x, edge_attr, weight, grad_out), src, dst, needs=needs)
    )


def load_node_contraction(op):
    extension = load_cuda_extension(
        f"lunatp_uvwconv_bwd_xy_shared_{op.dtype_tag}",
        make_uvw_conv_backward_dxdy_shared_source(op.meta, arch=op.arch),
        arch=op.arch,
        use_fast_math=False,
        index_bits=op.index_bits,
    )
    op.precontract_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_z_{op.dtype_tag}"
    )
    op.precontract_z_pair = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_z_pair_{op.dtype_tag}"
    )
    op.pair_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_pair_from_z_{op.dtype_tag}"
    )
    op.launch_dxdy = getattr(
        extension, f"lunatp_uvwconv_bwd_dxdy_shared_launch_{op.dtype_tag}"
    )
    op.grad_x_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_x_from_z_{op.dtype_tag}"
    )
    op.grad_edge_from_z = getattr(
        extension, f"lunatp_uvwconv_bwd_xy_source_y_from_z_{op.dtype_tag}"
    )
    op.node_contraction = extension


def load_weight_contraction(op):
    extension = load_tp_extension(
        f"lunatp_uvwconv_bwd_weight_precontract_shared_{op.dtype_tag}",
        make_uvw_conv_backward_dw_preprocess_shared_source(op.meta),
        max_channels=max(op.meta["max_in_channels"], op.meta["max_out_channels"]),
        index_bits=op.index_bits,
        arch=op.arch,
    )
    op.weight_reduce = load_cuda_extension(
        f"lunatp_uvwconv_bwd_weight_shared_{op.dtype_tag}",
        make_uvw_conv_backward_dw_reduce_shared_source(op.meta, arch=op.arch),
        arch=op.arch,
        use_fast_math=False,
        index_bits=op.index_bits,
    )
    op.precontract_q = getattr(
        extension, f"lunatp_uvwconv_bwd_weight_grouped_q_launch_{op.dtype_tag}"
    )
    op.precontract_q_pair = getattr(
        extension, f"lunatp_uvwconv_bwd_weight_grouped_q_pair_launch_{op.dtype_tag}"
    )
    op.weight_contraction = extension
