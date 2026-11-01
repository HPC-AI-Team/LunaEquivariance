"""NormActivation forward CUDA kernel and binding."""

from functools import partial
from math import gcd
from textwrap import indent

import torch
from .utils import CUDA_HEADERS, FORWARD_BLOCK_SIZE, FORWARD_MIN_BLOCKS, MAX_SHARED_BYTES, emit_copy_program
from ....utils.indexing import LAYOUT_HOST, LAYOUT_SOURCE
from ...activation.kernels.scalar import activation_cuda_prelude


def forward_expression_source(node, args, dtype, *, eps_squared):
    """Keep the registered radius/scaling operations explicitly rounded."""
    precision = "d" if dtype == torch.float64 else "f"
    if node.op == "fma":
        suffix = "" if dtype == torch.float64 else "f"
        if node.args[0] is node.args[1] and node.args[0].op == "input":
            return f"__fma{suffix}_rn({args[0]},{args[0]},{args[2]})"
        if any(argument.op == "input" and argument.value[0] == "x" for argument in node.args[:2]):
            return f"__fma{suffix}_rn({args[0]},{args[1]},{args[2]})"
    elif node.op == "mul":
        if node.args[0] is node.args[1] and node.args[0].op == "input":
            return f"__{precision}mul_rn({args[0]},{args[0]})"
        if any(argument.op == "input" and argument.value[0] == "x" for argument in node.args):
            return f"__{precision}mul_rn({args[0]},{args[1]})"
    elif node.op == "sqrt" and node.args[0].op in ("where", "aten.relu.default"):
        return f"__{precision}sqrt_rn({args[0]})"
    elif node.op == "lt" and node.args[1].op == "constant":
        if float.fromhex(node.args[1].value) == eps_squared:
            return f"({args[0]}<{args[1]})"
    elif node.op == "where" and node.args[1].op == "constant":
        if float.fromhex(node.args[1].value) == eps_squared:
            return f"({args[0]}?{args[1]}:{args[2]})"
    elif node.op == "add":
        if node.args[1].op == "input" and node.args[1].value[0] == "bias":
            return f"__{precision}add_rn({args[0]},{args[1]})"
    elif node.op == "div" and node.args[1].op == "sqrt":
        return f"__{precision}div_rn({args[0]},{args[1]})"
    return None


def make_forward_source(meta, program, *, strided=False, generic=False, save_bias=False):
    block_size = FORWARD_BLOCK_SIZE
    warps = block_size // 32
    max_components = max((path["components"] for path in meta["paths"]), default=1)
    copy_tile = gcd(32, *(path["copies"] for path in meta["paths"]))
    rows_per_warp = 32 // copy_tile
    shared_outputs = (
        meta["dtype"] == torch.float32 and copy_tile >= 8 and not (generic or strided)
        and max_components * block_size * 4 <= MAX_SHARED_BYTES
    )
    lines, atoms, count, paths = emit_copy_program(
        meta, program, strided=strided, generic=generic, save_bias=save_bias,
        expression_source=partial(forward_expression_source, eps_squared=meta["eps_squared"]),
        shared_outputs=shared_outputs,
    )
    fields, names = (program["fields"], program["active_outputs"])
    scalar = "double" if meta["dtype"] == torch.float64 else "float"
    parameters = [f"const Element* __restrict__ {role}" for role in program["inputs"]]
    parameters += [f"Element* __restrict__ out_{name}" for name in names]
    arguments = [f"{role}.data_ptr<Element>()" for role in program["inputs"]]
    arguments += [f"result_{name}.data_ptr<Element>()" for name in names]
    allocations = [
        f"auto result_{name}=torch::empty({{num_items,int64_t({len(fields[name])})}},options);"
        for name in names
    ]
    returns = [
        f"pybind11::cast(result_{name})" if name in names else "pybind11::none()"
        for name in program["outputs"]
    ]
    if save_bias:
        assert meta["kind"] == "norm_activation" and program["order"] == 0
        assert program["inputs"] == ("x", "bias") and names == ("out",)
        assert count == meta["bias_dim"] and count > 0
        parameters.append("Element* __restrict__ saved_state")
        arguments.append("bias_state.data_ptr<Element>()")
        allocations.append(f"auto bias_state=torch::empty({{int64_t({count})}},options);")
        returns.append("pybind11::cast(bias_state)")
    layout_declarations = ""
    layout_host = ""
    if generic or strided:
        assert program["order"] == 0 and program["inputs"] in (("x",), ("x", "bias"))
        bias_stride = "Index(bias.stride(0))" if "bias" in program["inputs"] else "Index(1)"
        if generic:
            layout_declarations = "\n" + LAYOUT_SOURCE
            layout_host = LAYOUT_HOST + "\n"
            parameters.extend(("ForwardLayout x_layout", "Index bias_stride"))
            arguments.extend(("x_layout", bias_stride))
        else:
            parameters.extend(("Index x_row_stride", "Index x_col_stride", "Index bias_stride"))
            arguments.extend(("Index(x.stride(0))", "Index(x.stride(1))", bias_stride))

    first = program["inputs"][0]
    kernel = "norm_activation_forward"
    if shared_outputs:
        tiles = sum((path["copies"] + copy_tile - 1) // copy_tile for path in meta["paths"])
        paths_source = []
        offset = 0
        for path, statements in paths:
            copies, components = path["copies"], path["components"]
            if not copies:
                continue
            width = copies // copy_tile
            saved = ""
            if save_bias:
                saved = f"if(row==0) saved_state[{path['scalar_offset']}+copy]=bias[{path['scalar_offset']}+copy];"
            stores = ""
            if components * (8 if meta["dtype"] == torch.float64 else 4) > 16:
                stores = f"""__syncwarp();
if(row<num_items) {{
    #pragma unroll
    for(int c=0;c<{components};++c) {{
        const Index column=Index(c)*{copy_tile}+lane;
        out_out[row*{meta['in_dim']}+{path['input_offset']}+copy_base*{components}+column]=values[column];
    }}
}}
__syncwarp();"""
            paths_source.append(f"""if(tile>={offset} && tile<{offset + width}) {{
    const Index copy_base=(tile-{offset})*{copy_tile};
    const Index copy=copy_base+lane;
    Element* values=shared+group*{copy_tile}*{max_components};
    {saved}
    if(row<num_items) {{
{indent(chr(10).join(statements), "        ")}
    }}
{indent(stores, "    ")}
}}""")
            offset += width
        device_kernel = f"""__global__ __launch_bounds__({block_size},{FORWARD_MIN_BLOCKS}) void {kernel}({','.join(parameters)},int64_t num_items) {{
    __shared__ __align__(16) Element shared[{block_size}*{max_components}];
    const int lane=threadIdx.x%{copy_tile},warp=threadIdx.x>>5;
    const int group=threadIdx.x/{copy_tile};
    for(Index task=Index(blockIdx.x)*{warps}+warp;task<((num_items+{rows_per_warp-1})/{rows_per_warp})*{tiles};task+=Index(gridDim.x)*{warps}) {{
        const Index row=(task/{tiles})*{rows_per_warp}+(threadIdx.x&31)/{copy_tile};
        const Index tile=task%{tiles};
{indent(chr(10).join(paths_source), "        ")}
    }}
}}"""
        geometry = f"<<<std::min<int64_t>(65535,(((num_items+{rows_per_warp-1})/{rows_per_warp})*{tiles}+{warps-1})/{warps}),{block_size},0,stream>>>"
    else:
        device_kernel = None
        geometry = f"<<<std::min<int64_t>(65535,(num_items*{count}+{block_size-1})/{block_size}),{block_size},0,stream>>>"
    launch = f"if(num_items>0) {kernel}{geometry}({','.join(arguments)},num_items);"
    prelude = activation_cuda_prelude(meta["dtype"], atoms)
    device_kernel = device_kernel or r"""__global__ __launch_bounds__({block_size},1) void {kernel}({parameters},int64_t num_items) {{
    for(Index item=Index(blockIdx.x)*{block_size}+threadIdx.x;item<num_items*{count};item+=Index(gridDim.x)*{block_size}) {{
{statements}
    }}
}}""".format(
        kernel=kernel,
        block_size=block_size,
        parameters=','.join(parameters),
        count=count,
        statements=indent(chr(10).join(lines), "        "),
    )
    body = r"""using Element={scalar};{layout_declarations}
{prelude}
{device_kernel}
pybind11::tuple launch({tensor_parameters},int64_t num_items) {{
    c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();
{allocations}
{layout_host}    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    {launch}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pybind11::make_tuple({returns});
}}
""".format(
        scalar=scalar,
        layout_declarations=layout_declarations,
        prelude=prelude,
        device_kernel=device_kernel,
        tensor_parameters=','.join((f'torch::Tensor {role}' for role in program['inputs'])),
        first=first,
        allocations=indent(chr(10).join(allocations), "    "),
        layout_host=layout_host,
        launch=launch,
        returns=','.join(returns),
    )
    binding = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n    m.def("launch",&launch);\n}\n'
    return "\n" + "\n".join(CUDA_HEADERS) + "\n" + body + binding
