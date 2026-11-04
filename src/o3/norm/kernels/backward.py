"""First derivative of a per-irrep norm; recomputes the square sum."""

CUDA_TILE_SOURCE = r"""
template<int Components, int InputOffset>
__device__ __forceinline__ void norm_store_tile(
    const Element* values, Element* output, Index row, Index copy_base
) {
    const int lane = threadIdx.x % COPY_TILE;
    constexpr int Pack = sizeof(Element) == sizeof(double) ? 2 : 4;
    using Vector = std::conditional_t<sizeof(Element) == sizeof(double), double2, float4>;
    Vector* destination = reinterpret_cast<Vector*>(
        output + row * INPUT_DIM + InputOffset + copy_base * Components);
    #pragma unroll
    for (int column = lane; column < Components * COPY_TILE / Pack; column += COPY_TILE)
        destination[column] = reinterpret_cast<const Vector*>(values)[column];
}
"""

CUDA_SOURCE = r"""
template<int Components>
__device__ __forceinline__ void norm_backward(
    const Element* x, Element grad_out, Element* dx, Index feature_offset, Index output_offset
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
        dx[output_offset + c] = add_rn(product<Components>(scale, x[feature_offset + c]), product<Components>(x[feature_offset + c], scale));
}

#if COALESCED
template<int Components, int Copies, int InputOffset, int NormOffset, int TileOffset>
__device__ __forceinline__ void norm_backward_tile(
    const Element* x, const Element* grad_out, Element* dx,
    Index task, int64_t rows, Element* shared
) {
    constexpr int Tiles = Copies / COPY_TILE;
    const Index tile = task % PATH_TILES;
    if (tile < TileOffset || tile >= TileOffset + Tiles) return;
    const int lane = threadIdx.x % COPY_TILE;
    const int group = threadIdx.x / COPY_TILE;
    const Index row = (task / PATH_TILES) * ROWS_PER_WARP + (threadIdx.x & 31) / COPY_TILE;
    const Index copy_base = (tile - TileOffset) * COPY_TILE;
    const Index copy = copy_base + lane;
    if constexpr (Components * sizeof(Element) <= 16) {
        if (row < rows)
            norm_backward<Components>(x, grad_out[row * OUTPUT_DIM + NormOffset + copy], dx,
                row * INPUT_DIM + InputOffset + copy * Components,
                row * INPUT_DIM + InputOffset + copy * Components);
    } else {
        Element* values = shared + group * COPY_TILE * MAX_COMPONENTS;
        if (row < rows)
            norm_backward<Components>(x, grad_out[row * OUTPUT_DIM + NormOffset + copy],
                                     values, row * INPUT_DIM + InputOffset + copy * Components,
                                     lane * Components);
        __syncwarp();
        if (row < rows) norm_store_tile<Components, InputOffset>(values, dx, row, copy_base);
        __syncwarp();
    }
}
#endif

__global__ __launch_bounds__(BLOCK_SIZE, BACKWARD_MIN_BLOCKS) void norm_backward_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    Element* __restrict__ dx, int64_t rows
) {
    #if COALESCED
    __shared__ __align__(16) Element shared[BLOCK_SIZE * MAX_COMPONENTS];
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

at::Tensor launch(at::Tensor x, at::Tensor grad_out, int64_t rows) {
    c10::cuda::CUDAGuard guard(x.device());
    auto dx = at::empty({rows, int64_t(INPUT_DIM)}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows > 0 && OUTPUT_DIM > 0) {
        const int64_t work = COALESCED
            ? ((rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP) * PATH_TILES * 32
            : rows * OUTPUT_DIM;
        const int blocks = std::min<int64_t>(MAX_BLOCKS, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
        norm_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), dx.data_ptr<Element>(), rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return dx;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
