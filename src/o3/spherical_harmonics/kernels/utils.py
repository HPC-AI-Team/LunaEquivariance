"""Shared spherical-harmonic kernel configuration and host binding."""

INPUT_ROLES = ('x', 'grad_out', 'rx', 'bx', 'bg')

BACKWARD_INPUTS = ("x", "grad_out")
BACKWARD_OUTPUTS = ("dx",)

TRIPLE_BACKWARD_INPUTS = ("x", "grad_out", "rx", "bx", "bg")
TRIPLE_BACKWARD_OUTPUTS = ("tx", "tg", "trx")


HOST_BINDING = r"""pybind11::tuple launch({launch_parameters},int64_t num_items) {{
    c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();
    {allocations}
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
{layout_setup}    if(num_items>0) {kernel}<<<std::min<int64_t>(65535,(num_items+127)/128),128,0,stream>>>({arguments},num_items{layout_argument});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pybind11::make_tuple({results});
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
"""
