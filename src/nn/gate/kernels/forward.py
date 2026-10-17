"""Gate CUDA forward, dense/strided schedules, and training native binding."""

import torch

from ...activation.kernels.backward import (
    BACKWARD_CALL, BACKWARD_CHECKS, TENSOR_BINDING as BACKWARD_TENSOR_BINDING,
    TENSOR_MODULE_BINDING as BACKWARD_TENSOR_MODULE_BINDING,
)
from ...activation.kernels.forward import (
    BINDING, FORWARD_CALL, MODULE_BINDING, TENSOR_BINDING,
    TENSOR_MODULE_BINDING,
)
from .utils import make_binding, make_path_types, make_schedule

FORWARD_MATH = r'''template<class Path>
__device__ __forceinline__ void gate_forward_scalar(
    const Element* x, int64_t x_row, int64_t feature_stride,
    Element* out, Index row, Index copy) {
    const Element value = x[x_row + (Path::input_offset + copy) * feature_stride];
    out[row * OutWidth + Path::output_offset + copy] =
        activation_forward_value<Path::kind, Path::normalize>(value, Path::parameters());
}
template<class Path>
__device__ __forceinline__ void gate_forward_product(
    const Element* x, int64_t x_row, int64_t feature_stride,
    Element* out, Index row, Index copy) {
    const Element value = x[x_row + (Path::gate_offset + copy) * feature_stride];
    const Element gate = activation_forward_value<Path::kind, Path::normalize>(
        value, Path::parameters());
    #pragma unroll
    for (int component = 0; component < Path::components; ++component) {
        const Index input = Path::input_offset + copy * Path::components + component;
        const Index output = row * OutWidth + Path::output_offset +
                             copy * Path::components + component;
        out[output] = gate_multiply(gate, x[x_row + input * feature_stride]);
    }
}'''


FORWARD_KERNEL = r'''__global__ __launch_bounds__(128, 1) void gate_forward(
    const Element* __restrict__ x, Element* __restrict__ out,
    int64_t num_items{layout_parameter}) {{
    for (Index item = Index(blockIdx.x) * 128 + threadIdx.x;
         item < num_items * {count}; item += Index(gridDim.x) * 128) {{
        const Index row = item / {count};
        const Index group = item % {count};
        {input_layout}
        {schedule}
    }}
}}'''


GATE_FORWARD_BINDING = r'''
torch::Tensor launch_fast_forward(torch::Tensor x) {{
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t num_items=x.numel()/{width};
    constexpr int64_t padding=65535*128;
    TORCH_CHECK(num_items<=(std::numeric_limits<int64_t>::max()-padding)/{max_width},"Gate task offsets exceed int64");
    pybind11::tuple results;
    if(x.is_contiguous() || x.dim()>64) {{
        results=dense_forward::launch(x.reshape({{num_items,int64_t({width})}}).contiguous(),num_items);
    }} else {{
        results=strided_forward::launch(x,num_items);
    }}
    auto out=results[0].cast<torch::Tensor>();
    std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end());
    shape.back()={out_width};
    return out.reshape(shape);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch_fast_forward",&launch_fast_forward);}}
'''


def make_source(meta, config, *, strided=False, tensor_only=False, fragments=False):
    assert config["order"] == 0 and config["inputs"] == ("x",)
    assert config["active_outputs"] == ("out",)
    declarations, paths = make_path_types(meta, config["dtype"])
    count, schedule = make_schedule(
        paths,
        "gate_forward_scalar<{path}>(x, x_row, feature_stride, out, row, copy);",
        "gate_forward_product<{path}>(x, x_row, feature_stride, out, row, copy);",
    )
    declarations += [f"constexpr int OutWidth = {meta['out_dim']};", FORWARD_MATH]
    kernel = FORWARD_KERNEL.format(
        count=count, schedule=schedule,
        layout_parameter=",ForwardLayout x_layout" if strided else "",
        input_layout=("const int64_t x_row = forward_row_offset(row, x_layout);\n"
                      "const int64_t feature_stride = x_layout.feature_stride;" if strided else
                      f"const int64_t x_row = row * {meta['in_dim']};\n"
                      "const int64_t feature_stride = 1;"),
    )
    layout_argument = ",x_layout" if strided else ""
    launch = (
        f"if(num_items>0) gate_forward<<<std::min<int64_t>(65535,(num_items*{count}+127)/128),128,0,stream>>>"
        f"(x.data_ptr<Element>(),result_out.data_ptr<Element>(),num_items{layout_argument});"
    )
    return make_binding(
        meta, config, declarations=declarations, kernel_body=kernel,
        binding=BINDING, module_binding=MODULE_BINDING, launch=launch,
        strided=strided, tensor_only=tensor_only, fragments=fragments,
    )


def make_training_source(meta, forward, backward):
    """Compile the actual dense/strided forward and terminal backward together."""
    from .backward import make_source as make_backward_source

    kernels = []
    for config in (forward, backward):
        order = config["order"]
        assert order in (0, 1) and config["active_outputs"] == (("out",) if order == 0 else ("dx",))
        assert config["inputs"] == (("x",) if order == 0 else ("x", "grad_out"))
        build = make_backward_source if order else make_source
        includes, dense = build(meta, config, tensor_only=True, fragments=True)
        bodies = "\nnamespace contiguous {\n" + dense + "\n}\n"
        if order == 0:
            strided_headers, strided_source = make_source(
                meta, config, strided=True, tensor_only=True, fragments=True
            )
            includes += strided_headers
            bodies += "\nnamespace strided {\n" + strided_source + "\n}\n"
        includes += ["#include <limits>", "#include <vector>"]
        width, out_width = meta["in_dim"], meta["out_dim"]
        args = "torch::Tensor x" + (",torch::Tensor grad_out" if order else "")
        checks = BACKWARD_CHECKS.format(out_width=out_width) if order else ""
        call = (BACKWARD_CALL.format(width=width, out_width=out_width) if order else
                FORWARD_CALL.format(results_type="torch::Tensor", width=width))
        binding = BACKWARD_TENSOR_BINDING if order else TENSOR_BINDING
        wrapper = binding.format(
            args=args, scalar="Double" if meta["dtype"] == torch.float64 else "Float",
            width=width, max_width=max(width, out_width, 1), checks=checks, call=call,
            result_value="results", result_width=width if order else out_width,
        )
        kernels.append((includes, bodies + wrapper))
    (includes0, forward_impl), (includes1, backward_impl) = kernels
    return ("\n".join(dict.fromkeys([*includes0, *includes1])) +
            "\nnamespace forward_impl {\n" + forward_impl + "\n}\n" +
            "\nnamespace backward_impl {\n" + backward_impl + "\n}\n" +
            TENSOR_MODULE_BINDING + BACKWARD_TENSOR_MODULE_BINDING)


def make_host_source(meta, config):
    """Bind sorted Gate outputs while reading the input's original strides."""
    assert meta["dtype"] == torch.float32
    dense_headers, dense_source = make_source(meta, config, fragments=True)
    strided_headers, strided_source = make_source(meta, config, strided=True, fragments=True)
    includes = "\n".join(dict.fromkeys([
        *dense_headers, *strided_headers, "#include <limits>", "#include <vector>",
    ]))
    source = (includes + "\nnamespace dense_forward {\n" + dense_source + "\n}\n" +
              "\nnamespace strided_forward {\n" + strided_source + "\n}\n")
    return source + GATE_FORWARD_BINDING.format(
        width=meta["in_dim"], out_width=meta["out_dim"],
        max_width=max(meta["in_dim"], meta["out_dim"], 1),
    )
