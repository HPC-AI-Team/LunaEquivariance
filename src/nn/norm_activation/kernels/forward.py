"""NormActivation radius/scaling and forward CUDA copy kernel binding."""

from functools import partial

import torch
from .utils import CUDA_HEADERS, emit_copy_program
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
    lines, atoms, count, _ = emit_copy_program(
        meta, program, strided=strided, generic=generic, save_bias=save_bias,
        expression_source=partial(forward_expression_source, eps_squared=meta["eps_squared"]),
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
    geometry = f"<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
    launch = f"if(num_items>0) {kernel}{geometry}({','.join(arguments)},num_items);"
    prelude = activation_cuda_prelude(meta["dtype"], atoms)
    device_kernel = r"""__global__ __launch_bounds__(128,1) void {kernel}({parameters},int64_t num_items) {{
    for(Index item=Index(blockIdx.x)*128+threadIdx.x;item<num_items*{count};item+=Index(gridDim.x)*128) {{
        {statements}
    }}
}}""".format(
        kernel=kernel,
        parameters=','.join(parameters),
        count=count,
        statements=chr(10).join(lines),
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
        allocations=chr(10).join(allocations),
        layout_host=layout_host,
        launch=launch,
        returns=','.join(returns),
    )
    binding = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch",&launch);}\n'
    return "\n" + "\n".join(CUDA_HEADERS) + "\n" + body + binding
