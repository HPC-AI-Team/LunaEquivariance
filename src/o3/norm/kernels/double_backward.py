"""Second derivatives with respect to features and the incoming gradient."""

CUDA_SOURCE = r"""
template<int Components>
__device__ __forceinline__ void norm_double_backward(
    const Element* x, Element grad_out, const Element* rx,
    Element* sx, Element* sg, Index feature_offset, Index norm_offset, Index output_offset
) {
    const Element square = SQUARED ? Element(0) : dot<Components>(x + feature_offset, x + feature_offset);
    const Element twice_dot = add_rn(dot<Components>(rx + feature_offset, x + feature_offset), dot<Components>(x + feature_offset, rx + feature_offset));
    if constexpr (SQUARED) {
        if constexpr (WRITE_G) sg[norm_offset] = twice_dot;
        if constexpr (WRITE_X) {
            #pragma unroll
            for (int c = 0; c < Components; ++c)
                sx[output_offset + c] = add_rn(product<Components>(rx[feature_offset + c], grad_out), product<Components>(grad_out, rx[feature_offset + c]));
        }
    } else {
        const Element denominator = mul_rn(sqrt_rn(square), Element(2));
        const Element masked_dot = relu_backward(twice_dot, square);
        if constexpr (WRITE_X) {
            const Element first_scale = div_rn(grad_out, denominator);
            const Element scale = relu_backward(first_scale, square);
            const Element curvature = div_rn(
                mul_rn(mul_rn(-masked_dot, div_rn(first_scale, denominator)), Element(2)),
                denominator);
            const Element radial = relu_backward(add_rn(Element(0), curvature), square);
            #pragma unroll
            for (int c = 0; c < Components; ++c) {
                const Element left = add_rn(product<Components>(rx[feature_offset + c], scale), product<Components>(radial, x[feature_offset + c]));
                const Element right = add_rn(product<Components>(scale, rx[feature_offset + c]), product<Components>(x[feature_offset + c], radial));
                sx[output_offset + c] = add_rn(left, right);
            }
        }
        if constexpr (WRITE_G) sg[norm_offset] = div_rn(masked_dot, denominator);
    }
}

#if COALESCED
template<int Components, int Copies, int InputOffset, int NormOffset, int TileOffset>
__device__ __forceinline__ void norm_double_backward_tile(
    const Element* x, const Element* grad_out, const Element* rx,
    Element* sx, Element* sg, Index task, int64_t rows, Element* shared
) {
    constexpr int Tiles = Copies / COPY_TILE;
    const Index tile = task % PATH_TILES;
    if (tile < TileOffset || tile >= TileOffset + Tiles) return;
    const int lane = threadIdx.x % COPY_TILE;
    const int group = threadIdx.x / COPY_TILE;
    const Index row = (task / PATH_TILES) * ROWS_PER_WARP + (threadIdx.x & 31) / COPY_TILE;
    const Index copy_base = (tile - TileOffset) * COPY_TILE;
    const Index copy = copy_base + lane;
    const Index feature = (row < rows ? row : 0) * INPUT_DIM + InputOffset + copy * Components;
    const Index norm = (row < rows ? row : 0) * OUTPUT_DIM + NormOffset + copy;
    if constexpr (Components * sizeof(Element) <= 16) {
        if (row < rows)
            norm_double_backward<Components>(
                x, grad_out[row * OUTPUT_DIM + NormOffset + copy],
                rx, sx, sg, feature, norm, feature);
    } else {
        Element* values = WRITE_X ? shared + group * COPY_TILE * MAX_COMPONENTS : nullptr;
        if (row < rows)
            norm_double_backward<Components>(
                x, grad_out[row * OUTPUT_DIM + NormOffset + copy],
                rx, values, sg, feature, norm, lane * Components);
        __syncwarp();
        if constexpr (WRITE_X)
            if (row < rows) norm_store_tile<Components, InputOffset>(values, sx, row, copy_base);
        __syncwarp();
    }
}
#endif

__global__ __launch_bounds__(BLOCK_SIZE, 1) void norm_double_backward_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, Element* __restrict__ sx,
    Element* __restrict__ sg, int64_t rows
) {
    #if COALESCED
    __shared__ __align__(16) Element shared[WRITE_X ? BLOCK_SIZE * MAX_COMPONENTS : 1];
    const int warp = threadIdx.x >> 5;
    for (Index task = Index(blockIdx.x) * (BLOCK_SIZE / 32) + warp;
         task < ((rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP) * PATH_TILES;
         task += Index(gridDim.x) * (BLOCK_SIZE / 32)) {
        /* tiles */
    }
    #else
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        const Index row = item / OUTPUT_DIM;
        const Index group = item % OUTPUT_DIM;
        /* paths */
    }
    #endif
}

pybind11::tuple launch(at::Tensor x, at::Tensor grad_out, at::Tensor rx, int64_t rows) {
    c10::cuda::CUDAGuard guard(x.device());
    at::Tensor sx, sg;
    if constexpr (WRITE_X) sx = at::empty({rows, int64_t(INPUT_DIM)}, x.options());
    if constexpr (WRITE_G) sg = at::empty({rows, int64_t(OUTPUT_DIM)}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows > 0 && OUTPUT_DIM > 0) {
        const int64_t work = COALESCED
            ? ((rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP) * PATH_TILES * 32
            : rows * OUTPUT_DIM;
        const int blocks = std::min<int64_t>(MAX_BLOCKS, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
        norm_double_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), rx.data_ptr<Element>(),
            WRITE_X ? sx.data_ptr<Element>() : nullptr,
            WRITE_G ? sg.data_ptr<Element>() : nullptr, rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(WRITE_X ? pybind11::cast(sx) : pybind11::none(),
                                WRITE_G ? pybind11::cast(sg) : pybind11::none());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
