"""First derivative of a per-irrep norm; recomputes the square sum."""

CUDA_SOURCE = r"""
template<int Components>
__device__ __forceinline__ void norm_backward(
    const Element* x, Element grad_out, Element* dx, Index feature_offset
) {
    Element scale = grad_out;
    if constexpr (!SQUARED) {
        const Element square = dot<Components>(x + feature_offset, x + feature_offset);
        const Element denominator = mul_rn(sqrt_rn(square), Element(2));
        const Element first_scale = div_rn(grad_out, denominator);
        scale = relu_backward(first_scale, square);
    }
    #pragma unroll
    for (int c = 0; c < Components; ++c)
        dx[feature_offset + c] = add_rn(product<Components>(scale, x[feature_offset + c]), product<Components>(x[feature_offset + c], scale));
}

__global__ __launch_bounds__(BLOCK_SIZE, 1) void norm_backward_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    Element* __restrict__ dx, int64_t rows
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        const Index row = item / OUTPUT_DIM;
        const Index group = item % OUTPUT_DIM;
        /* paths */
    }
}

#if PATH_MAJOR
__global__ __launch_bounds__(BLOCK_SIZE, 1) void norm_backward_path_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    Element* __restrict__ dx, int64_t rows
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        /* path_major */
    }
}

#endif

at::Tensor launch(at::Tensor x, at::Tensor grad_out, int64_t rows) {
    c10::cuda::CUDAGuard guard(x.device());
    auto dx = at::empty({rows, int64_t(INPUT_DIM)}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows > 0 && OUTPUT_DIM > 0) {
        #if PATH_MAJOR
        if (rows >= PATH_MAJOR_MIN_ROWS) {
        norm_backward_path_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), dx.data_ptr<Element>(), rows);
        } else
        #endif
        norm_backward_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), dx.data_ptr<Element>(), rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return dx;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
