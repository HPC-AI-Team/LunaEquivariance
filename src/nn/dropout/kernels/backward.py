"""Apply the saved compact mask; backward and higher orders consume no RNG."""


def make_backward_kernels(group_layout):
    source = r'''__global__ __launch_bounds__(BLOCK,4) void dropout_apply_compact(
        const Element *input,Element *output,const uint8_t *state,Layout layout,Element scale) {
    const int64_t middle=layout.row_items/WIDTH;
    for(int64_t tile=blockIdx.x;tile<layout.batch*WIDTH_TILES;tile+=gridDim.x) {
        const int64_t batch=tile/WIDTH_TILES;
        const int column=int(tile%WIDTH_TILES)*BLOCK+threadIdx.x;
        if(column>=WIDTH) continue;
        int group=0,component=0; int64_t copy=0,mask_copy=0;
        copy_for(column,batch,group,component,copy,mask_copy);
        const Element factor=Element(state[mask_copy])*scale;
        // One compact lookup serves all components and all middle slices;
        // no random-state regeneration or full-width mask is materialized.
        for(int64_t m=0;m<middle;++m) {
            const int64_t flat=batch*layout.row_items+m*WIDTH+column;
            output[flat]=input[input_offset(flat,layout)]*factor;
        }
    }
}

// Contiguous fast path uses only small launch metadata. Arbitrary strides keep
// the original general Layout path and the same compact-state ABI.
__device__ __forceinline__ uint32_t compact_index32(uint32_t column,uint32_t batch) {
    '''
    source += '\n    '.join(
        f"if(column>={feature_begin}U && column<{feature_begin + mul * dim}U) "
        f"return batch*uint32_t(COPIES)+{copy_begin}U"
        f"+(column-{feature_begin}U)/{dim}U;"
        for group, mul, dim, feature_begin, copy_begin in group_layout
        if mul
    )
    source += r'''
    return 0;
}

template<bool OneMiddle>
__global__ __launch_bounds__(VBLOCK,8) void dropout_apply_compact_vec32(
        const ElementVector *__restrict__ input,ElementVector *__restrict__ output,
        const uint8_t *__restrict__ state,uint32_t batches,uint32_t row_items,
        uint32_t middle,Element scale) {
    for(uint32_t tile=blockIdx.x;tile<batches*WIDTH_TILES;tile+=gridDim.x) {
        const uint32_t batch=tile/WIDTH_TILES;
        const uint32_t column=(tile%WIDTH_TILES)*BLOCK+threadIdx.x*2;
        if(column>=WIDTH) continue;
        const Element f0=Element(state[compact_index32(column,batch)])*scale;
        const Element f1=Element(state[compact_index32(column+1,batch)])*scale;
        if constexpr(OneMiddle) {
            const uint32_t index=(batch*WIDTH+column)/2;
            const ElementVector v=input[index];
            output[index]=ElementVector{v.x*f0,v.y*f1};
        } else {
            for(uint32_t m=0;m<middle;++m) {
                const uint32_t index=(batch*row_items+m*WIDTH+column)/2;
                const ElementVector v=input[index];
                output[index]=ElementVector{v.x*f0,v.y*f1};
            }
        }
    }
}

template<bool OneMiddle>
__global__ __launch_bounds__(BLOCK,4) void dropout_apply_compact_scalar32(
        const Element *__restrict__ input,Element *__restrict__ output,
        const uint8_t *__restrict__ state,uint32_t batches,uint32_t row_items,
        uint32_t middle,Element scale) {
    for(uint32_t tile=blockIdx.x;tile<batches*WIDTH_TILES;tile+=gridDim.x) {
        const uint32_t batch=tile/WIDTH_TILES;
        const uint32_t column=(tile%WIDTH_TILES)*BLOCK+threadIdx.x;
        if(column>=WIDTH) continue;
        const Element factor=Element(state[compact_index32(column,batch)])*scale;
        if constexpr(OneMiddle) {
            const uint32_t index=batch*WIDTH+column;
            output[index]=input[index]*factor;
        } else {
            for(uint32_t m=0;m<middle;++m) {
                const uint32_t index=batch*row_items+m*WIDTH+column;
                output[index]=input[index]*factor;
            }
        }
    }
}

template<bool UseMap>
__device__ __forceinline__ ElementVector4 compact_factors4(uint32_t column,uint32_t batch,
        const uint8_t *state,Element scale) {
    if constexpr(UseMap) {
        if constexpr(COPIES%4==0) {
            const uint32_t packed=__ldg(PACKED_COPY4+column/4);
            const uint32_t selectors=packed&65535;
            const uint32_t index=(batch*COPIES+(packed>>16))/4;
            const auto *words=reinterpret_cast<const uint32_t*>(state);
            const uint32_t low=__ldg(words+index);
            // Bit 2 of each byte selector chooses the second mask word.
            const uint32_t high=(selectors&0x4444) ? __ldg(words+index+1) : 0;
            const uint32_t masks=__byte_perm(low,high,selectors);
            return ElementVector4{Element(masks&255)*scale,
                                  Element((masks>>8)&255)*scale,
                                  Element((masks>>16)&255)*scale,
                                  Element(masks>>24)*scale};
        } else {
            const unsigned long long packed=__ldg(PACKED_COPY4+column/4);
            const uint32_t base=batch*COPIES;
            const Element f0=Element(state[base+uint32_t(packed&65535)])*scale;
            const Element f1=Element(state[base+uint32_t((packed>>16)&65535)])*scale;
            const Element f2=Element(state[base+uint32_t((packed>>32)&65535)])*scale;
            const Element f3=Element(state[base+uint32_t(packed>>48)])*scale;
            return ElementVector4{f0,f1,f2,f3};
        }
    } else {
        return ElementVector4{Element(state[compact_index32(column,batch)])*scale,
                             Element(state[compact_index32(column+1,batch)])*scale,
                             Element(state[compact_index32(column+2,batch)])*scale,
                             Element(state[compact_index32(column+3,batch)])*scale};
    }
}

template<bool OneMiddle,bool UseMap,int Threads>
__global__ __launch_bounds__(Threads) void dropout_apply_flat4_fast(
        const ElementVector4 *__restrict__ input,ElementVector4 *__restrict__ output,
        const uint8_t *__restrict__ state,uint32_t vectors,uint32_t row_items,Element scale) {
    for(uint32_t index=blockIdx.x*Threads+threadIdx.x;index<vectors;index+=gridDim.x*Threads) {
        const uint32_t flat=index*4;
        const uint32_t column=flat%WIDTH;
        const uint32_t batch=OneMiddle ? flat/WIDTH : flat/row_items;
        const ElementVector4 f=compact_factors4<UseMap>(column,batch,state,scale);
        const ElementVector4 v=input[index];
        output[index]=ElementVector4{v.x*f.x,v.y*f.y,v.z*f.z,v.w*f.w};
    }
}


'''
    return source


SHARED_MASK_KERNEL = r'''// A CTA owns a bounded contiguous output tile. Shared factors live only
// during this kernel; the recursive autograd ABI retains compact bytes only.
template<int MiddleTile>
__global__ __launch_bounds__(FORWARD_BLOCK,8) void dropout_apply_shared_tiles(
        const ForwardVector *__restrict__ input,ForwardVector *__restrict__ output,
        const uint8_t *__restrict__ state,uint32_t batches,uint32_t middle,Element scale) {
    extern __shared__ Element factors[];
    const uint32_t begin=uint32_t(blockIdx.x)*MiddleTile;
    const uint32_t rows=std::min(uint32_t(MiddleTile),middle-begin);
    for(uint32_t batch=uint32_t(blockIdx.y);batch<batches;batch+=uint32_t(gridDim.y)) {
        for(uint32_t copy=threadIdx.x;copy<uint32_t(COPIES);copy+=FORWARD_BLOCK) {
            const Element factor=Element(state[batch*uint32_t(COPIES)+copy])*scale;
            write_expanded_factor(copy,factor,factors);
        }
        __syncthreads();
        const uint32_t base=(batch*middle+begin)*(WIDTH/FORWARD_VECTOR_WIDTH);
        const uint32_t vectors=rows*(WIDTH/FORWARD_VECTOR_WIDTH);
        const auto *vector_factors=reinterpret_cast<const ForwardVector*>(factors);
        for(uint32_t local=threadIdx.x;local<vectors;local+=FORWARD_BLOCK) {
            const uint32_t column=local%(WIDTH/FORWARD_VECTOR_WIDTH);
            const ForwardVector v=input[base+local];
            const ForwardVector f=vector_factors[column];
            output[base+local]=ForwardVector{v.x*f.x,v.y*f.y};
        }
        __syncthreads();
    }
}

'''


def make_backward_launch(double_precision):
    source = r'''at::Tensor apply_mask(const at::Tensor &input,const at::Tensor &state,double probability) {
    c10::cuda::CUDAGuard guard(input.device());
    const Layout layout=layout_for(input);
    const int mode=probability_mode(probability);
    auto output=at::empty(input.sizes(),input.options().requires_grad(false));
    if(layout.numel>0) {
        const int64_t blocks=std::min<int64_t>(65535,layout.batch*WIDTH_TILES);
        const Element scale=mode==0 ? Element(1.0/(1.0-probability)) : Element(1);
        const cudaStream_t stream=at::cuda::getCurrentCUDAStream(input.get_device());
        const bool compact32=layout.contiguous && layout.numel<=INT32_MAX &&
                             layout.row_items<=INT32_MAX && state.numel()<=INT32_MAX;
        if(compact32) {
            const uint32_t batches=uint32_t(layout.batch);
            const uint32_t row_items=uint32_t(layout.row_items);
            const uint32_t middle=uint32_t(layout.row_items/WIDTH);
            const bool vector=(WIDTH%2==0) &&
                              uintptr_t(input.data_ptr())%alignof(ElementVector)==0;
            const bool packed_flat=sizeof(Element)==4 && HAS_PACKED_MAP &&
                                   uintptr_t(input.data_ptr())%alignof(ElementVector4)==0;
            '''
    if double_precision:
        source += r'''const bool shared_tiles=middle>=8 &&
                WIDTH/FORWARD_VECTOR_WIDTH>=FORWARD_BLOCK &&
                WIDTH%FORWARD_VECTOR_WIDTH==0 &&
                uintptr_t(input.data_ptr())%alignof(ForwardVector)==0 &&
                uint64_t(WIDTH)*sizeof(Element)<=32*1024;
            if(shared_tiles) {
                const dim3 grid((middle+7)/8,std::min(batches,65535U));
                dropout_apply_shared_tiles<8><<<grid,FORWARD_BLOCK,WIDTH*sizeof(Element),stream>>>(
                    reinterpret_cast<const ForwardVector*>(input.data_ptr<Element>()),
                    reinterpret_cast<ForwardVector*>(output.data_ptr<Element>()),
                    state.data_ptr<uint8_t>(),batches,middle,scale);
            } else '''
    source += r'''if(packed_flat) {
                const uint32_t vectors=uint32_t(layout.numel/4);
                const uint32_t sm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
                const uint32_t flat_blocks=std::min((vectors+255)/256,sm*16);
                if(middle==1)
                    dropout_apply_flat4_fast<true,true,256><<<flat_blocks,256,0,stream>>>(
                        reinterpret_cast<const ElementVector4*>(input.data_ptr<Element>()),
                        reinterpret_cast<ElementVector4*>(output.data_ptr<Element>()),
                        state.data_ptr<uint8_t>(),vectors,row_items,scale);
                else
                    dropout_apply_flat4_fast<false,true,256><<<flat_blocks,256,0,stream>>>(
                        reinterpret_cast<const ElementVector4*>(input.data_ptr<Element>()),
                        reinterpret_cast<ElementVector4*>(output.data_ptr<Element>()),
                        state.data_ptr<uint8_t>(),vectors,row_items,scale);
            } else if(vector) {
                const auto *in=reinterpret_cast<const ElementVector*>(input.data_ptr<Element>());
                auto *out=reinterpret_cast<ElementVector*>(output.data_ptr<Element>());
                if(middle==1)
                    dropout_apply_compact_vec32<true><<<blocks,VBLOCK,0,stream>>>(
                        in,out,state.data_ptr<uint8_t>(),batches,row_items,middle,scale);
                else
                    dropout_apply_compact_vec32<false><<<blocks,VBLOCK,0,stream>>>(
                        in,out,state.data_ptr<uint8_t>(),batches,row_items,middle,scale);
            } else if(middle==1) {
                dropout_apply_compact_scalar32<true><<<blocks,BLOCK,0,stream>>>(
                    input.data_ptr<Element>(),output.data_ptr<Element>(),state.data_ptr<uint8_t>(),
                    batches,row_items,middle,scale);
            } else {
                dropout_apply_compact_scalar32<false><<<blocks,BLOCK,0,stream>>>(
                    input.data_ptr<Element>(),output.data_ptr<Element>(),state.data_ptr<uint8_t>(),
                    batches,row_items,middle,scale);
            }
        } else {
            dropout_apply_compact<<<blocks,BLOCK,0,stream>>>(input.data_ptr<Element>(),
                output.data_ptr<Element>(),state.data_ptr<uint8_t>(),layout,scale);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

'''
    return source


MASK_BINDING = '    module.def("apply_mask",&apply_mask);'
