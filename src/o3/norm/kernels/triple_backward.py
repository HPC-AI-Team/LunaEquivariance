"""Third derivatives; absent second-order seeds are compiled out."""

CUDA_SOURCE = r"""
template<int Components>
__device__ __forceinline__ void norm_triple_backward(
    const Element* x, Element grad_out, const Element* rx,
    const Element* bx, Element bg, Element* tx, Element* tg, Element* trx,
    Index feature_offset, Index norm_offset
) {
    if constexpr (SQUARED) {
        if constexpr (WRITE_G)
            tg[norm_offset] = add_rn(dot<Components>(rx + feature_offset, bx + feature_offset), dot<Components>(bx + feature_offset, rx + feature_offset));
        #pragma unroll
        for (int c = 0; c < Components; ++c) {
            if constexpr (WRITE_X)
                tx[feature_offset + c] = add_rn(product<Components>(bg, rx[feature_offset + c]), product<Components>(rx[feature_offset + c], bg));
            if constexpr (WRITE_RX) {
                Element left = HAS_BG ? product<Components>(bg, x[feature_offset + c]) : Element(0);
                Element right = HAS_BG ? product<Components>(x[feature_offset + c], bg) : Element(0);
                if constexpr (HAS_BX) {
                    const Element from_left = product<Components>(grad_out, bx[feature_offset + c]);
                    const Element from_right = product<Components>(bx[feature_offset + c], grad_out);
                    left = HAS_BG ? add_rn(left, from_left) : from_left;
                    right = HAS_BG ? add_rn(from_right, right) : from_right;
                }
                trx[feature_offset + c] = add_rn(left, right);
            }
        }
    } else {
        const Element square = dot<Components>(x + feature_offset, x + feature_offset);
        const Element denominator = mul_rn(sqrt_rn(square), Element(2));
        const Element twice_rx = add_rn(dot<Components>(rx + feature_offset, x + feature_offset), dot<Components>(x + feature_offset, rx + feature_offset));
        const Element masked_rx = relu_backward(twice_rx, square);
        Element scale = 0, curvature = 0, radial = 0;
        Element mixed = HAS_BG ? div_rn(bg, denominator) : Element(0);
        Element radial_grad = HAS_BG ? mul_rn(-bg, div_rn(div_rn(masked_rx, denominator), denominator)) : Element(0);
        Element radial_denominator = 0;
        Element grad_g = 0;
        if constexpr (HAS_BX) {
            const Element first_scale = div_rn(grad_out, denominator);
            scale = relu_backward(first_scale, square);
            const Element scale_over_denominator = div_rn(first_scale, denominator);
            curvature = div_rn(mul_rn(mul_rn(-masked_rx, scale_over_denominator), Element(2)), denominator);
            radial = relu_backward(add_rn(Element(0), curvature), square);
            const Element twice_bx = add_rn(dot<Components>(bx + feature_offset, x + feature_offset), dot<Components>(x + feature_offset, bx + feature_offset));
            const Element masked_bx = relu_backward(twice_bx, square);
            const Element bx_scale = mul_rn(div_rn(masked_bx, denominator), Element(2));
            const Element from_bx = -mul_rn(bx_scale, scale_over_denominator);
            mixed = HAS_BG ? add_rn(mixed, from_bx) : from_bx;
            radial_denominator = mul_rn(mul_rn(-masked_bx, div_rn(curvature, denominator)), Element(2));
            const Element product = mul_rn(bx_scale, -masked_rx);
            const Element contribution = mul_rn(-product, div_rn(scale_over_denominator, denominator));
            radial_grad = HAS_BG ? add_rn(radial_grad, contribution) : contribution;
            radial_grad = add_rn(radial_grad, mul_rn(-div_rn(product, denominator), scale_over_denominator));
            const Element twice_mixed = add_rn(dot<Components>(rx + feature_offset, bx + feature_offset), dot<Components>(bx + feature_offset, rx + feature_offset));
            const Element masked_mixed = relu_backward(twice_mixed, square);
            radial_grad = add_rn(radial_grad, mul_rn(-masked_mixed, scale_over_denominator));
            if constexpr (WRITE_G)
                grad_g = add_rn(div_rn(div_rn(product, denominator), denominator), div_rn(masked_mixed, denominator));
        }
        mixed = relu_backward(mixed, square);
        Element radial_second = mul_rn(radial_grad, Element(2));
        if constexpr (HAS_BX) radial_second = add_rn(radial_denominator, radial_second);
        radial_second = add_rn(Element(0), div_rn(radial_second, denominator));
        radial_second = relu_backward(radial_second, square);
        #pragma unroll
        for (int c = 0; c < Components; ++c) {
            if constexpr (WRITE_X) {
                Element left = product<Components>(mixed, rx[feature_offset + c]);
                Element right = product<Components>(rx[feature_offset + c], mixed);
                if constexpr (HAS_BX) {
                    left = add_rn(product<Components>(bx[feature_offset + c], radial), left);
                    right = add_rn(product<Components>(radial, bx[feature_offset + c]), right);
                }
                left = add_rn(left, product<Components>(radial_second, x[feature_offset + c]));
                right = add_rn(right, product<Components>(x[feature_offset + c], radial_second));
                tx[feature_offset + c] = add_rn(left, right);
            }
        }
        if constexpr (WRITE_G) tg[norm_offset] = grad_g;
        #pragma unroll
        for (int c = 0; c < Components; ++c) {
            if constexpr (WRITE_RX) {
                Element left = product<Components>(mixed, x[feature_offset + c]);
                Element right = product<Components>(x[feature_offset + c], mixed);
                if constexpr (HAS_BX) {
                    left = add_rn(left, product<Components>(scale, bx[feature_offset + c]));
                    right = add_rn(product<Components>(bx[feature_offset + c], scale), right);
                }
                trx[feature_offset + c] = add_rn(left, right);
            }
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, TRIPLE_MIN_BLOCKS) void norm_triple_backward_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ bx,
    const Element* __restrict__ bg, Element* __restrict__ tx,
    Element* __restrict__ tg, Element* __restrict__ trx, int64_t rows
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        const Index row = item / OUTPUT_DIM;
        const Index group = item % OUTPUT_DIM;
        /* paths */
    }
}

#if PATH_MAJOR
__global__ __launch_bounds__(BLOCK_SIZE, TRIPLE_MIN_BLOCKS) void norm_triple_backward_path_kernel(
    const Element* __restrict__ x, const Element* __restrict__ grad_out,
    const Element* __restrict__ rx, const Element* __restrict__ bx,
    const Element* __restrict__ bg, Element* __restrict__ tx,
    Element* __restrict__ tg, Element* __restrict__ trx, int64_t rows
) {
    for (Index item = Index(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         item < rows * OUTPUT_DIM; item += Index(gridDim.x) * BLOCK_SIZE) {
        /* path_major */
    }
}

#endif

pybind11::tuple launch(at::Tensor x, at::Tensor grad_out, at::Tensor rx,
                      std::optional<at::Tensor> bx, std::optional<at::Tensor> bg, int64_t rows) {
    c10::cuda::CUDAGuard guard(x.device());
    at::Tensor tx, tg, trx;
    if constexpr (WRITE_X) tx = at::empty({rows, int64_t(INPUT_DIM)}, x.options());
    if constexpr (WRITE_G) tg = at::empty({rows, int64_t(OUTPUT_DIM)}, x.options());
    if constexpr (WRITE_RX) trx = at::empty({rows, int64_t(INPUT_DIM)}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows > 0 && OUTPUT_DIM > 0) {
        #if PATH_MAJOR
        if (rows >= PATH_MAJOR_MIN_ROWS) {
        norm_triple_backward_path_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), rx.data_ptr<Element>(),
            HAS_BX ? bx->data_ptr<Element>() : nullptr, HAS_BG ? bg->data_ptr<Element>() : nullptr,
            WRITE_X ? tx.data_ptr<Element>() : nullptr, WRITE_G ? tg.data_ptr<Element>() : nullptr,
            WRITE_RX ? trx.data_ptr<Element>() : nullptr, rows);
        } else
        #endif
        norm_triple_backward_kernel<<<std::min<int64_t>(MAX_BLOCKS, (rows * OUTPUT_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<Element>(), grad_out.data_ptr<Element>(), rx.data_ptr<Element>(),
            HAS_BX ? bx->data_ptr<Element>() : nullptr, HAS_BG ? bg->data_ptr<Element>() : nullptr,
            WRITE_X ? tx.data_ptr<Element>() : nullptr, WRITE_G ? tg.data_ptr<Element>() : nullptr,
            WRITE_RX ? trx.data_ptr<Element>() : nullptr, rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return pybind11::make_tuple(WRITE_X ? pybind11::cast(tx) : pybind11::none(),
                               WRITE_G ? pybind11::cast(tg) : pybind11::none(),
                               WRITE_RX ? pybind11::cast(trx) : pybind11::none());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
