"""UVW convolution triple backward CUDA and native launch."""

from __future__ import annotations
import torch
from ....utils.precision import cuda_source_for_dtype, float_literal
from ....utils.extension import load_tp_extension, load_cuda_extension
from ...utils.runtime import auxiliary_streams
from ...utils.contractions import ENV_INPUTS, ENV_PROGRAMS, ENV_OUTPUT_SHAPES
from .backward import (
    emit_dx_lines,
    grad_x_allocator,
    run_external,
    load_node_contraction,
    load_weight_contraction,
)
from .utils import get_uvw_conv_node_layout, set_arch
from .double_backward import (
    make_uvw_conv_double_backward_output_shared_source,
    make_uvw_conv_q_shared_source,
)


def make_uvw_conv_apply_shared_source(meta, program, inputs, outputs, *, phase):
    """Apply a derivative recipe to existing node-major GEMM contractions."""
    specs = meta["generated_specs"]
    z_offsets, total_z = get_uvw_conv_node_layout(specs)
    wide_z = total_z > (1 << 31) - 1
    x_products = {
        name: products for name, products in program["x"].items() if name in outputs
    }
    y_products = {
        name: products for name, products in program["y"].items() if name in outputs
    }
    e_products = {
        name: tuple((x, y, z) for q, z in products for x, y, _ in program["cg"][q])
        for name, products in program["e"].items()
        if name in outputs
    }
    needed_z = tuple(
        name
        for name in program["z"]
        if any(
            name == term[1]
            for terms in (*x_products.values(), *y_products.values())
            for term in terms
        )
        or any(name == z for terms in e_products.values() for _, _, z in terms)
    )
    channels = max(int(s["in_channels"]) for s in specs)
    threads = 64 if channels <= 64 else 128
    parameters = ",".join(f"at::Tensor {name}" for name in inputs)
    cg_outputs = tuple(
        name for name in outputs if name in (*x_products, *y_products, *e_products)
    )
    allocations = []
    for name in cg_outputs:
        shape = ENV_OUTPUT_SHAPES[name]
        allocation = (
            f"""{grad_x_allocator(meta).replace("torch::", "at::")}({{num_nodes,GEN_IN1_DIM}},x.options())"""
            if shape == "x"
            else f"at::empty_like({shape})"
        )
        allocations.append(f"auto {name}={allocation};")

    kernels, launches, clears = [], [], []
    if x_products:
        lines, group_grid = emit_dx_lines(
            meta,
            z_offsets,
            f"lunatp_uvwconv_{phase}_x_apply_fp32",
            "GEN_X_THREADS",
            "GEN_MAX_IN_CHANNELS",
            wide_z=wide_z,
            products=x_products,
        )
        kernels.append("\n".join(lines))
        arrays = tuple(
            sorted(
                {
                    role
                    for terms in x_products.values()
                    for term in terms
                    for role in term
                    if role is not None
                }
            )
        )
        arguments = [
            f"zs[{needed_z.index(name)}].data_ptr<float>()"
            if name in needed_z
            else f"{name}.data_ptr<float>()"
            for name in arrays
        ]
        arguments += [
            "dst.data_ptr<int32_t>()",
            "src_rowptr.data_ptr<EdgeIndex>()",
            "edge_by_src.data_ptr<EdgeIndex>()",
        ]
        arguments += [f"{name}.data_ptr<float>()" for name in x_products] + [
            "num_nodes"
        ]
        launches.append(
            f"""if(num_nodes>0) lunatp_uvwconv_{phase}_x_apply_fp32<<<dim3(static_cast<unsigned int>(num_nodes),(GEN_MAX_IN_CHANNELS+GEN_X_THREADS-1)/GEN_X_THREADS{group_grid}),GEN_X_THREADS,0,stream>>>({",".join(arguments)});"""
        )
    if y_products or e_products:
        edge_kernel_name = f"lunatp_uvwconv_{phase}_y_apply_fp32"
        edge_specs = meta["generated_specs"]
        edge_in1_dim, edge_in2_dim = int(meta["in1_dim"]), int(meta["in2_dim"])
        edge_families = tuple(
            dict.fromkeys(
                [
                    (edge_x, edge_z)
                    for edge_terms in y_products.values()
                    for edge_x, edge_z, edge__ in edge_terms
                ]
                + [
                    (edge_x, edge_z)
                    for edge_terms in e_products.values()
                    for edge_x, _, edge_z in edge_terms
                ]
            )
        )
        edge_arrays = tuple(
            sorted(
                {
                    edge_role
                    for edge_terms in y_products.values()
                    for edge_term in edge_terms
                    for edge_role in edge_term
                    if edge_role is not None
                }
                | {
                    edge_role
                    for edge_terms in e_products.values()
                    for edge_term in edge_terms
                    for edge_role in edge_term
                }
            )
        )
        edge_budget = max(1, 48 // len(edge_families))
        edge_groups = {}
        for edge_index, edge_spec in enumerate(edge_specs):
            edge_groups.setdefault(
                int(edge_spec["y_offset"]) if edge_in2_dim > edge_budget else 0, []
            ).append((edge_index, edge_spec))
        edge_covered = {
            int(edge_s["y_offset"]) + edge_j
            for edge_s in edge_specs
            for edge_j in range(int(edge_s["edge_channels"]) * int(edge_s["d2"]))
        }
        edge_group_components = [
            sorted(
                {
                    int(edge_s["y_offset"]) + edge_j
                    for _, edge_s in edge_group
                    for edge_j in range(
                        int(edge_s["edge_channels"]) * int(edge_s["d2"])
                    )
                }
            )
            for edge_group in edge_groups.values()
        ]
        edge_width = max(map(len, edge_group_components))
        edge_lines = [
            r"""__global__ __launch_bounds__(GEN_Y_THREADS,4)
void {edge_kernel_name}(
{input_pointers}
    const int32_t* src,const int32_t* dst,
{output_pointers}
    int64_t num_edges) {{
    __shared__ float smem[GEN_Y_WARPS*{shared_edge_stride}];
    const Index e=blockIdx.x;
    if(e>=num_edges) return;
    const int s=__ldg(src+e),t=__ldg(dst+e);
    const int u=threadIdx.x;""".format(
                edge_kernel_name=edge_kernel_name,
                input_pointers="\n".join(
                    [
                        f"    const float* __restrict__ ptr_{edge_name},"
                        for edge_name in edge_arrays
                    ]
                ),
                output_pointers="\n".join(
                    [
                        f"    float* __restrict__ {edge_name},"
                        for edge_name in (*y_products, *e_products)
                    ]
                ),
                shared_edge_stride=len(edge_families) * edge_width,
            )
        ]
        for edge_group_index, ((_, edge_group_specs), edge_components) in enumerate(
            zip(edge_groups.items(), edge_group_components)
        ):
            if len(edge_groups) > 1:
                edge_lines.append(f"    if(blockIdx.y=={edge_group_index}) {{")
            for edge_family in range(len(edge_families)):
                edge_lines += [
                    f"    float acc_{edge_family}_{edge_j}=0.f;"
                    for edge_j in edge_components
                ]
            for edge_spec_index, edge_spec in edge_group_specs:
                edge_d1, edge_d2, edge_d3 = (
                    int(edge_spec[edge_k]) for edge_k in ("d1", "d2", "d3")
                )
                edge_channels, edge_edges = (
                    int(edge_spec["in_channels"]),
                    int(edge_spec["edge_channels"]),
                )
                edge_channel_type = (
                    "int64_t" if edge_channels > (1 << 31) - threads else "int"
                )
                for edge_v in range(edge_edges):
                    edge_path = (
                        f"""({("int64_t(u)" if wide_z else "u")}*{edge_edges}+{edge_v})"""
                        if edge_edges != 1
                        else "int64_t(u)"
                        if wide_z
                        else "u"
                    )
                    edge_lines += [
                        "    {",
                        f"    for({edge_channel_type} u=threadIdx.x;u<{edge_channels};u+=GEN_Y_THREADS) {{",
                    ]
                    for edge_x in sorted({edge_x for edge_x, edge__ in edge_families}):
                        edge_lines += [
                            f"""        const float xv_{edge_x}_{edge_i}=__ldg(ptr_{edge_x}+Index(s)*{edge_in1_dim}+{edge_spec["x_offset"]}+u*{edge_d1}+{edge_i});"""
                            for edge_i in sorted(
                                {int(edge_term[0]) for edge_term in edge_spec["terms"]}
                            )
                        ]
                    for edge_z in sorted({edge_z for _, edge_z in edge_families}):
                        edge_lines += [
                            f"        const float zv_{edge_z}_{edge_k}=__ldg(ptr_{edge_z}+Index(t)*GEN_TOTAL_Z+{z_offsets[edge_spec_index]}+{edge_path}*{edge_d3}+{edge_k});"
                            for edge_k in sorted(
                                {int(edge_term[2]) for edge_term in edge_spec["terms"]}
                            )
                        ]
                    for edge_i, edge_j, edge_k, edge_coefficient in edge_spec["terms"]:
                        edge_component = (
                            int(edge_spec["y_offset"]) + edge_v * edge_d2 + int(edge_j)
                        )
                        for edge_family, (edge_x, edge_z) in enumerate(edge_families):
                            edge_lines.append(
                                f"""        acc_{edge_family}_{edge_component}=fmaf({float_literal(float(edge_coefficient), meta.get("dtype"))}*xv_{edge_x}_{int(edge_i)},zv_{edge_z}_{int(edge_k)},acc_{edge_family}_{edge_component});"""
                            )
                    edge_lines += ["    }", "    }"]
            for edge_family in range(len(edge_families)):
                for edge_component_index, edge_j in enumerate(edge_components):
                    edge_lines += [
                        r"""    float sum_{edge_family}_{edge_j}=acc_{edge_family}_{edge_j};
    #pragma unroll
    for(int delta=16;delta>0;delta>>=1) sum_{edge_family}_{edge_j}+=__shfl_down_sync(0xffffffff,sum_{edge_family}_{edge_j},delta);
    if((threadIdx.x&31)==0) smem[(threadIdx.x>>5)*{shared_edge_stride}+{component_offset}]=sum_{edge_family}_{edge_j};""".format(
                            edge_family=edge_family,
                            edge_j=edge_j,
                            shared_edge_stride=len(edge_families) * edge_width,
                            component_offset=edge_family * edge_width
                            + edge_component_index,
                        )
                    ]
            edge_lines += "    __syncthreads();\n    if(threadIdx.x<32) {".split("\n")
            for edge_family in range(len(edge_families)):
                for edge_component_index, edge_j in enumerate(edge_components):
                    edge_lines += [
                        r"""        float block_{edge_family}_{edge_j}=threadIdx.x<GEN_Y_WARPS?smem[threadIdx.x*{shared_edge_stride}+{component_offset}]:0.f;
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1) block_{edge_family}_{edge_j}+=__shfl_down_sync(0xffffffff,block_{edge_family}_{edge_j},delta);""".format(
                            edge_family=edge_family,
                            edge_j=edge_j,
                            shared_edge_stride=len(edge_families) * edge_width,
                            component_offset=edge_family * edge_width
                            + edge_component_index,
                        )
                    ]
            edge_lines.append("        if(threadIdx.x==0) {")
            for edge_target, edge_terms in y_products.items():
                for edge_j in edge_components:
                    edge_expression = "+".join(
                        f"block_{edge_families.index((edge_x, edge_z))}_{edge_j}"
                        + (
                            f"*__ldg(ptr_{edge_scalar}+e)"
                            if edge_scalar is not None
                            else ""
                        )
                        for edge_x, edge_z, edge_scalar in edge_terms
                    )
                    edge_lines.append(
                        f"            {edge_target}[e*{edge_in2_dim}+{edge_j}]={edge_expression};"
                    )
            for edge_target, edge_terms in e_products.items():
                edge_lines.append(f"            float value_{edge_target}=0.f;")
                for edge_x, edge_y, edge_z in edge_terms:
                    for edge_j in edge_components:
                        edge_lines.append(
                            f"            value_{edge_target}=fmaf(block_{edge_families.index((edge_x, edge_z))}_{edge_j},__ldg(ptr_{edge_y}+e*{edge_in2_dim}+{edge_j}),value_{edge_target});"
                        )
                edge_lines.append(
                    f"            atomicAdd({edge_target}+e,value_{edge_target});"
                    if len(edge_groups) > 1
                    else f"            {edge_target}[e]=value_{edge_target};"
                )
            if edge_group_index == 0:
                for edge_target in y_products:
                    edge_lines += [
                        f"            {edge_target}[e*{edge_in2_dim}+{edge_j}]=0.f;"
                        for edge_j in sorted(set(range(edge_in2_dim)) - edge_covered)
                    ]
            edge_lines += ["        }", "    }"]
            if len(edge_groups) > 1:
                edge_lines += ["    return;", "    }"]
        edge_lines.append("}")
        edge_grid = "static_cast<unsigned int>(num_edges)"
        if len(edge_groups) > 1:
            edge_grid = f"dim3({edge_grid},{len(edge_groups)})"
        lines, grid, arrays, clear_env = (
            edge_lines,
            edge_grid,
            edge_arrays,
            len(edge_groups) > 1,
        )
        kernels.append("\n".join(lines))
        arguments = [
            f"zs[{needed_z.index(name)}].data_ptr<float>()"
            if name in needed_z
            else f"{name}.data_ptr<float>()"
            for name in arrays
        ]
        arguments += ["src.data_ptr<int32_t>()", "dst.data_ptr<int32_t>()"]
        arguments += [
            f"{name}.data_ptr<float>()" for name in (*y_products, *e_products)
        ] + ["num_edges"]
        launches.append(
            f"""if(num_edges>0) {edge_kernel_name}<<<{grid},GEN_Y_THREADS,0,stream>>>({",".join(arguments)});"""
        )
        if clear_env:
            clears += [
                f"C10_CUDA_CHECK(cudaMemsetAsync({name}.data_ptr<float>(),0,{name}.numel()*sizeof(float),stream));"
                for name in e_products
            ]
    return cuda_source_for_dtype(
        r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
namespace {{
constexpr int GEN_X_THREADS={threads},GEN_Y_THREADS={threads};
constexpr int GEN_Y_WARPS=(GEN_Y_THREADS+31)/32;
constexpr int GEN_IN1_DIM={in1_dim},GEN_IN2_DIM={in2_dim};
constexpr {channel_type} GEN_MAX_IN_CHANNELS={channels};
constexpr {offset_type} GEN_TOTAL_Z={total_z};
{kernels}
}}
std::vector<at::Tensor> launch({parameters},std::vector<at::Tensor> zs,at::Tensor src,at::Tensor dst,at::Tensor src_rowptr,at::Tensor edge_by_src,int64_t num_nodes) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_edges=y.size(0);
    {allocations}
    auto stream=at::cuda::getCurrentCUDAStream();
    {clears}
    {launches}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{{cg_outputs}}};
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""".format(
            threads=threads,
            in1_dim=meta["in1_dim"],
            in2_dim=meta["in2_dim"],
            channel_type="int64_t" if channels > (1 << 31) - threads else "int",
            channels=channels,
            offset_type="int64_t" if wide_z else "int",
            total_z=total_z,
            kernels=chr(10).join(kernels),
            parameters=parameters,
            allocations=chr(10).join(allocations),
            clears=chr(10).join(clears),
            launches=chr(10).join(launches),
            cg_outputs=",".join(cg_outputs),
        ),
        meta.get("dtype", torch.float32),
    )


@torch.no_grad()
def run_envelope_backward(op, order, inputs, graph, needs=None):
    """Apply edge-scalar product rules around the existing fused GEMMs.

    Node-major Z cannot contain a per-edge scalar. Its consumers apply the
    scalar and share their unscaled reduction with the envelope gradient.
    Q retains the original grouped layout and includes edge scalars before
    destination aggregation. Only native CUDA performs the contractions.
    """
    set_arch(op, inputs[0].get_device())
    if op.weight_mode != "shared":
        return run_external(
            op, order, inputs, graph.src, graph.dst, needs=needs, env=True
        )
    program = ENV_PROGRAMS[order]
    outputs = program["outputs"]
    selected = (
        outputs
        if needs is None
        else tuple(name for name, need in zip(outputs, needs) if need)
    )
    if not selected:
        return (None,) * len(outputs)

    key = (order, selected)
    kernels = op.envelope_backward_kernels.get(key)
    if kernels is None:
        roles = ENV_INPUTS[: len(inputs)]
        cg_outputs = tuple(
            name
            for name in selected
            if name in (*program["x"], *program["y"], *program["e"])
        )
        needed_z = tuple(
            name
            for name in program["z"]
            if any(
                term[1] == name
                for axis in ("x", "y", "e")
                for target in cg_outputs
                for term in program[axis].get(target, ())
            )
        )
        needed_q = {q for target in selected for q, _ in program["w"].get(target, ())}
        if "gg" in selected:
            needed_q.update(q for q, _ in program["forward"])
        queries = {q: pairs for q, pairs in program["cg"].items() if q in needed_q}
        apply = None
        if cg_outputs:
            phase = ("fwd", "bwd", "doublebwd", "triplebwd")[order]
            apply = load_tp_extension(
                f"lunatp_uvwconv_{phase}_apply_shared_{op.dtype_tag}",
                make_uvw_conv_apply_shared_source(op.meta, program, roles, cg_outputs, phase=phase),
                max_channels=op.meta["max_in_channels"],
                index_bits=op.index_bits,
                arch=op.arch,
            ).launch
        q_launch = None
        if queries:
            q_launch = load_tp_extension(
                f"lunatp_uvwconv_bwd_q_shared_{op.dtype_tag}",
                make_uvw_conv_q_shared_source(op.meta, queries, roles),
                max_channels=op.meta["max_in_channels"],
                index_bits=op.index_bits,
                arch=op.arch,
            ).launch
        kernels = apply, q_launch, roles, cg_outputs, needed_z, queries
        op.envelope_backward_kernels[key] = kernels

    apply, q_launch, roles, cg_outputs, needed_z, queries = kernels
    values = dict(zip(roles, inputs))
    if needed_z and op.node_contraction is None:
        load_node_contraction(op)
    if queries and op.weight_contraction is None:
        load_weight_contraction(op)
    if "gg" in selected and op.output_contraction is None:
        op.output_contraction = load_cuda_extension(
            f"lunatp_uvwconv_bwd_output_shared_{op.dtype_tag}",
            make_uvw_conv_double_backward_output_shared_source(op.meta, arch=op.arch),
            arch=op.arch,
            use_fast_math=False,
            index_bits=op.index_bits,
        )

    result = {}
    current_stream = torch.cuda.current_stream(inputs[0].device)
    contract_stream = None
    if queries:
        contract_stream = auxiliary_streams(
            current_stream.device_index, 1, scope=current_stream.cuda_stream
        )[0]
        contract_stream.wait_stream(current_stream)
        with torch.cuda.stream(contract_stream):
            q = dict(
                zip(
                    queries,
                    q_launch(
                        *inputs,
                        graph.src,
                        graph.edge_ids_by_dst,
                        graph.dst_rowptr,
                        graph.num_nodes,
                    ),
                )
            )
            reduce = op.weight_reduce
            for target in selected:
                products = program["w"].get(target)
                if products is None:
                    continue
                if len(products) == 1:
                    query, gradient = products[0]
                    result[target] = reduce.launch(q[query], values[gradient])
                else:
                    (q0, g0), (q1, g1) = products
                    result[target] = reduce.launch_pair(
                        q[q0], q[q1], values[g0], values[g1]
                    )
            if "gg" in selected:
                (q0, w0), (q1, w1), *remaining = program["forward"]
                result["gg"] = op.output_contraction.launch_pair(
                    q[q0], q[q1], values[w0], values[w1], graph.num_nodes
                )
                for query, weight in remaining:
                    op.output_contraction.accumulate(
                        q[query], values[weight], result["gg"]
                    )
            del q
    if cg_outputs:
        z = []
        node = op.node_contraction
        for name in needed_z:
            products = program["z"][name]
            if len(products) == 1:
                weight, gradient = products[0]
                z.append(
                    op.precontract_z(values[weight], values[gradient], graph.num_nodes)
                )
            else:
                (w0, g0), (w1, g1) = products
                z.append(
                    op.precontract_z_pair(
                        values[w0],
                        values[g0],
                        values[w1],
                        values[g1],
                        graph.num_nodes,
                    )
                )
        result.update(
            zip(
                cg_outputs,
                apply(
                    *inputs,
                    z,
                    graph.src,
                    graph.dst,
                    graph.src_rowptr,
                    graph.edge_ids_by_src,
                    graph.num_nodes,
                ),
            )
        )
    if contract_stream is not None:
        current_stream.wait_stream(contract_stream)
        for name in (*program["w"], "gg"):
            if name in result:
                result[name].record_stream(current_stream)
    return tuple(result.get(name) for name in outputs)


def run(
    op, x, edge_attr, weight, grad_out, rx, ry, rw, bx, by, bw, bg, *, graph, needs=None
):
    set_arch(op, x.get_device())
    src, dst = graph.src, graph.dst
    num_nodes = graph.num_nodes
    if op.weight_mode != "shared":
        gx, gy, gw, gg, grx, gry, grw = tuple(
            run_external(
                op,
                3,
                (x, edge_attr, weight, grad_out, rx, ry, rw, bx, by, bw, bg),
                src,
                dst,
                needs=needs,
            )
        )
    else:
        if op.node_contraction is None:
            load_node_contraction(op)
        if op.weight_contraction is None:
            load_weight_contraction(op)

        if op.output_contraction is None:
            op.output_contraction = load_cuda_extension(
                f"lunatp_uvwconv_bwd_output_shared_{op.dtype_tag}",
                make_uvw_conv_double_backward_output_shared_source(
                    op.meta, arch=op.arch
                ),
                arch=op.arch,
                use_fast_math=False,
                index_bits=op.index_bits,
            )

        with torch.no_grad():
            node_products = op.precontract_z_pair(weight, bg, bw, grad_out, num_nodes)
            grad_x = op.grad_x_from_z(
                ry,
                node_products,
                graph.dst,
                graph.src_rowptr,
                graph.edge_ids_by_src,
                num_nodes,
            )
            grad_rx = op.grad_x_from_z(
                edge_attr,
                node_products,
                graph.dst,
                graph.src_rowptr,
                graph.edge_ids_by_src,
                num_nodes,
            )
            grad_edge = op.grad_edge_from_z(rx, node_products, graph.src, graph.dst)
            grad_ry = op.grad_edge_from_z(x, node_products, graph.src, graph.dst)
            del node_products

            node_products = op.precontract_z(rw, bg, num_nodes)
            grad_x.add_(
                op.grad_x_from_z(
                    edge_attr,
                    node_products,
                    graph.dst,
                    graph.src_rowptr,
                    graph.edge_ids_by_src,
                    num_nodes,
                )
            )
            grad_edge.add_(op.grad_edge_from_z(x, node_products, graph.src, graph.dst))
            del node_products

            node_products = op.precontract_z(rw, grad_out, num_nodes)
            grad_x.add_(
                op.grad_x_from_z(
                    by,
                    node_products,
                    graph.dst,
                    graph.src_rowptr,
                    graph.edge_ids_by_src,
                    num_nodes,
                )
            )
            grad_edge.add_(op.grad_edge_from_z(bx, node_products, graph.src, graph.dst))
            del node_products

            node_products = op.precontract_z(weight, grad_out, num_nodes)
            grad_rx.add_(
                op.grad_x_from_z(
                    by,
                    node_products,
                    graph.dst,
                    graph.src_rowptr,
                    graph.edge_ids_by_src,
                    num_nodes,
                )
            )
            grad_ry.add_(op.grad_edge_from_z(bx, node_products, graph.src, graph.dst))
            del node_products

            # Both consumers read the weight-gradient CG layout directly.
            # Consume two CG families before creating the next; at most two remain live.
            direction_products = op.precontract_q_pair(
                rx,
                edge_attr,
                x,
                ry,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            mixed_products = op.precontract_q_pair(
                bx,
                ry,
                rx,
                by,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            grad_weight = op.weight_reduce.launch_pair(
                direction_products, mixed_products, bg, grad_out
            )
            grad_grad_out = op.output_contraction.launch_pair(
                direction_products, mixed_products, bw, weight, num_nodes
            )
            del direction_products, mixed_products
            grad_products = op.precontract_q_pair(
                bx,
                edge_attr,
                x,
                by,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            op.output_contraction.accumulate(grad_products, rw, grad_grad_out)

            base_products = op.precontract_q(
                x,
                edge_attr,
                graph.src,
                graph.edge_ids_by_dst,
                graph.dst_rowptr,
                num_nodes,
            )
            grad_rw = op.weight_reduce.launch_pair(
                base_products, grad_products, bg, grad_out
            )
            gx, gy, gw, gg, grx, gry, grw = (
                grad_x,
                grad_edge,
                grad_weight,
                grad_grad_out,
                grad_rx,
                grad_ry,
                grad_rw,
            )
    return gx, gy, gw, gg, grx, gry, grw
