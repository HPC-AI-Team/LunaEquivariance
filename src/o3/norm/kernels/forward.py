"""Per-irrep square sum and norm, including strided and vectorized loads."""

CUDA_SOURCE = r"""
template<int Components>
__device__ __forceinline__ Element norm_forward(const Element* x, Index stride) {
    Element square = 0;
    #pragma unroll
    for (int c = 0; c < Components; ++c) {
        const Element value = x[c * stride];
        if constexpr (Components == 1) square = mul_rn(value, value);
        else square = fma_rn(value, value, square);
    }
    return SQUARED ? square : sqrt_rn(square < 0 ? Element(0) : square);
}

__global__ __launch_bounds__(BLOCK_SIZE, 1) void norm_forward_kernel(
    const Element* __restrict__ x, Element* __restrict__ out,
    int64_t rows
    #if STRIDED
    , ForwardLayout layout
    #endif
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        const Index row = item / OUTPUT_DIM;
        const Index group = item % OUTPUT_DIM;
        #if STRIDED
        const Index x_row = forward_row_offset(row, layout);
        const Index stride = layout.feature_stride;
        #else
        const Index x_row = row * INPUT_DIM;
        const Index stride = 1;
        #endif
        /* paths */
    }
}

#if VECTORIZED
// Four adjacent irreps share aligned float4 loads; each sum keeps component order.
template<int Components>
__device__ __forceinline__ float4 norm_forward_packet(const float* x) {
    alignas(16) float values[4 * Components];
    #pragma unroll
    for (int c = 0; c < Components; ++c)
        reinterpret_cast<float4*>(values)[c] = reinterpret_cast<const float4*>(x)[c];
    float result[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        result[k] = norm_forward<Components>(values + k * Components, 1);
    return make_float4(result[0], result[1], result[2], result[3]);
}

__global__ __launch_bounds__(BLOCK_SIZE, 1) void norm_forward_packet_kernel(
    const float* __restrict__ x, float* __restrict__ out, int64_t rows
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * (OUTPUT_DIM / 4); item += Index(gridDim.x) * BLOCK_SIZE) {
        const Index row = item / (OUTPUT_DIM / 4);
        const Index group = (item % (OUTPUT_DIM / 4)) * 4;
        /* packets */
    }
}
#endif

at::Tensor launch(at::Tensor x, int64_t rows) {
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({rows, int64_t(OUTPUT_DIM)}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    #if STRIDED
    ForwardLayout layout = make_forward_layout(x);
    #endif
    if (rows > 0 && OUTPUT_DIM > 0) {
        #if VECTORIZED
        if (rows >= VECTORIZE_MIN_ROWS && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0) {
            norm_forward_packet_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * (OUTPUT_DIM / 4) + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
                x.data_ptr<Element>(), out.data_ptr<Element>(), rows);
        } else
        #endif
        norm_forward_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), out.data_ptr<Element>(), rows
            #if STRIDED
            , layout
            #endif
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
